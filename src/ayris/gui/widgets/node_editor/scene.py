"""Сцена нодового редактора: ноды и связи одной команды на ``QGraphicsScene``.

Сцена — тонкое представление того же :class:`~ayris.gui.widgets.action_list.ActionListModel`,
что и список-режим: она не хранит своего дерева, а перестраивается из модели. Держит
координаты нод (метаданные UI), рисует фон (стеклянное цветное сияние варианта Б + точечная
сетка), подсвечивает текущий блок при отладке и ставит точки останова. Структурные правки
(связать, удалить) идут через модель, поэтому переключение «Список ↔ Ноды» ничего не теряет.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QRadialGradient
from PySide6.QtWidgets import QGraphicsScene, QGraphicsSceneMouseEvent

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.gui.widgets.action_list import ActionListModel, BlockPath
from ayris.gui.widgets.node_editor.bridge import (
    MAIN_PORT,
    DisplayEdge,
    GraphEdge,
    GraphNode,
    NodeGraph,
    can_connect,
    collapse_delays,
    command_from_graph,
    graph_from_command,
    make_delay_block,
    role_of,
    summary_of,
)
from ayris.gui.widgets.node_editor.edge_item import EdgeItem, bezier_path
from ayris.gui.widgets.node_editor.layout import GRID, auto_layout
from ayris.gui.widgets.node_editor.node_item import NodeItem

if TYPE_CHECKING:
    from PySide6.QtWidgets import QGraphicsPathItem

    from ayris.gui.theme import ThemeManager

__all__ = ["NodeScene"]


class NodeScene(QGraphicsScene):
    """Nodes and edges of one command, rebuilt from an :class:`ActionListModel`."""

    #: The selected block's full path, or an empty tuple on deselect.
    block_selected = Signal(object)
    #: A structural change made through the model (connect, disconnect, delete).
    changed_command = Signal()
    #: A connection the user attempted was refused, with a Russian reason.
    connection_rejected = Signal(str)
    #: A breakpoint was toggled on or off (its set changed), for persistence.
    breakpoint_changed = Signal()

    def __init__(
        self,
        model: ActionListModel,
        theme: ThemeManager,
        *,
        catalog: BlockCatalog | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._theme = theme
        self._catalog = catalog if catalog is not None else BlockCatalog()
        self._positions: dict[str, tuple[float, float]] = {}
        self._nodes: dict[str, NodeItem] = {}
        self._edges: list[EdgeItem] = []
        self._path_by_id: dict[str, BlockPath] = {}
        self._graph = NodeGraph()
        self._snap = False
        self._running_id: str | None = None
        #: Breakpoint node ids (block paths), kept apart from the items so they
        #: survive a rebuild and can be loaded from the debugger session store.
        self._breakpoints: set[str] = set()
        self.selectionChanged.connect(self._on_selection)

    # -- state --------------------------------------------------------------

    @property
    def graph(self) -> NodeGraph:
        return self._graph

    def set_snap(self, snap: bool) -> None:
        self._snap = snap
        for item in self._nodes.values():
            item.set_snap(snap)

    def set_layout(self, positions: dict[str, tuple[float, float]]) -> None:
        """Load saved node coordinates (UI metadata) before the next rebuild."""
        self._positions = dict(positions)

    def layout(self) -> dict[str, tuple[float, float]]:
        """Current node coordinates keyed by block path, for saving with the command."""
        return {node_id: (item.node.x, item.node.y) for node_id, item in self._nodes.items()}

    def positions_by_block(self) -> dict[int, tuple[float, float]]:
        """Snapshot the current coordinates keyed by *block identity*, not by path.

        Node coordinates are stored under the block's positional path (``actions[1]``),
        but a structural edit renumbers those paths — inserting or removing a block shifts
        every later sibling by one. Re-keyed by path alone, each saved coordinate would
        stay put and so land on the neighbour that inherited the old path, flinging the
        real nodes across the canvas. Taken by ``id(block)`` right before the edit, this
        snapshot survives the renumbering; :meth:`restore_positions` reapplies it after.
        """
        current = {**self._positions, **self.layout()}
        by_block: dict[int, tuple[float, float]] = {}
        for row in self._model.rows():
            pos = current.get(row.path_text)
            if pos is not None:
                by_block[id(row.block)] = pos
        return by_block

    def restore_positions(self, snapshot: dict[int, tuple[float, float]]) -> None:
        """Re-key a :meth:`positions_by_block` snapshot onto the current tree's paths.

        Call after the model mutation: each surviving block keeps its coordinate under its
        new path, a removed block drops out (its id is gone), and a freshly inserted block
        is simply absent — left for :meth:`rebuild`'s auto-layout (or an explicit placement)
        rather than stealing a shifted neighbour's spot.
        """
        positions: dict[str, tuple[float, float]] = {}
        for row in self._model.rows():
            pos = snapshot.get(id(row.block))
            if pos is not None:
                positions[row.path_text] = pos
        self.set_layout(positions)

    # -- building -----------------------------------------------------------

    def rebuild(self) -> None:
        """Redraw the whole graph from the model, keeping and completing the layout."""
        selected = self.selected_path()
        # Drop the item references and mute the selection signal BEFORE clearing the
        # scene: ``clear()`` deletes the C++ items and fires ``selectionChanged``, and a
        # handler that still walked the stale ``_nodes`` would touch a deleted object
        # («Internal C++ object already deleted»). Cleared dicts and blocked signals
        # keep that mid-clear callback harmless; the real selection is restored below.
        blocked = self.blockSignals(True)
        self._nodes.clear()
        self._edges.clear()
        self.clear()
        self.blockSignals(blocked)
        command = self._model.command
        if command is None:
            self._graph = NodeGraph()
            return
        self._path_by_id = {row.path_text: row.path for row in self._model.rows()}
        graph = graph_from_command(command, catalog=self._catalog, positions=self._positions)
        # Fold every linear «Пауза» onto its wire: such a block draws as a delay chip on the
        # connection, not as its own card. ``hidden`` names the folded nodes to skip and
        # ``display`` the wires to draw, each carrying the summed delay. A pause that cannot
        # fold (the root, disabled, or with no successor) is absent from ``hidden`` and keeps
        # its card, so a delay is never hidden without a place left to edit it.
        hidden, display = collapse_delays(graph)
        visible = [node for node in graph.nodes if node.id not in hidden]
        if any(node.id not in self._positions for node in visible):
            self._layout_collapsed(graph, visible, display)
            for node in visible:  # saved coordinates win over the auto pass
                if node.id in self._positions:
                    node.x, node.y = self._positions[node.id]
        self._graph = graph
        for node in visible:
            danger = self._is_dangerous(node.block.type)
            item = NodeItem(
                node,
                self._theme,
                summary=summary_of(node.block, self._catalog),
                danger=danger,
                snap=self._snap,
            )
            self.addItem(item)
            self._nodes[node.id] = item
        for wire_edge in display:
            source = self._nodes.get(wire_edge.source_id)
            target = self._nodes.get(wire_edge.target_id)
            if source is None or target is None:
                continue
            wire = EdgeItem(
                source,
                wire_edge.source_port,
                target,
                self._theme,
                edge=wire_edge,
                delay_ms=wire_edge.delay_ms,
            )
            self.addItem(wire)
            self._edges.append(wire)
        self._positions = self.layout()
        # Re-apply breakpoints to the freshly built items; drop any whose block is
        # gone (a deleted node) so the set never carries a path with no node.
        self._breakpoints &= set(self._nodes)
        for node_id in self._breakpoints:
            self._nodes[node_id].set_breakpoint(True)
        if selected:
            self.select_path(selected)

    def _is_dangerous(self, block_type: str) -> bool:
        meta = self._catalog.try_get(block_type)
        return bool(meta is not None and meta.is_dangerous)

    def _layout_collapsed(
        self,
        graph: NodeGraph,
        visible: list[GraphNode],
        display: list[DisplayEdge],
    ) -> None:
        """Auto-layout only the visible nodes, wired by the collapsed display edges.

        Folding a «Пауза» onto a wire removes its card, so laying out the *full* graph would
        leave the folded pause's column empty and stretch the wire across the gap. Laying out
        a throwaway graph of just the visible nodes — joined by the display edges, which
        already skip the folded pauses — packs the flow one column per visible step. The
        :class:`GraphNode` objects are shared, so their ``x``/``y`` are written straight onto
        the real graph.
        """
        compact = NodeGraph(
            nodes=visible,
            edges=[GraphEdge(edge.source_id, edge.source_port, edge.target_id) for edge in display],
            root_id=graph.root_id,
        )
        auto_layout(compact)

    def auto_arrange(self) -> None:
        """Re-lay-out the visible flow left-to-right (folded pauses excluded) and keep it.

        The toolbar's «Упорядочить» runs this: it lays out the collapsed topology so a wire
        with a delay does not open a hole where its chip-folded pause used to sit, then keeps
        the fresh coordinates for the rebuild that frames the graph.
        """
        hidden, display = collapse_delays(self._graph)
        visible = [node for node in self._graph.nodes if node.id not in hidden]
        self._layout_collapsed(self._graph, visible, display)
        self.set_layout({node.id: (node.x, node.y) for node in visible})

    # -- selection ----------------------------------------------------------

    def selected_path(self) -> BlockPath:
        for node_id, item in self._nodes.items():
            if item.isSelected():
                return self._path_by_id.get(node_id, ())
        return ()

    def select_path(self, path: BlockPath) -> None:
        # Exclusive by construction: clear any prior selection first, so selecting a node
        # never leaves an older one highlighted. Otherwise «+ Нода» (which selects the new
        # node while its anchor is still selected) would leave two nodes lit, and Delete —
        # reading ``selected_path()``, the first selected — would remove the wrong block.
        # The clear is muted so only the final selection emits ``block_selected`` once.
        target: NodeItem | None = None
        for node_id, item in self._nodes.items():
            if self._path_by_id.get(node_id) == path:
                target = item
                break
        blocked = self.blockSignals(True)
        self.clearSelection()
        self.blockSignals(blocked)
        if target is not None:
            target.setSelected(True)

    def _on_selection(self) -> None:
        self.block_selected.emit(self.selected_path())

    # -- movement -----------------------------------------------------------

    def notify_node_moved(self, item: NodeItem) -> None:
        """A node item moved: refresh the wires touching it and remember its position."""
        for wire in self._edges:
            if wire.source_id == item.node_id or wire.target_id == item.node_id:
                wire.update_path()
        self._positions[item.node_id] = (item.node.x, item.node.y)

    # -- connecting ---------------------------------------------------------

    def _rebuild_from_graph(self) -> dict[str, str] | None:
        """Rebuild the command from the current graph, keeping node coordinates in place.

        A structural edit reorders ``actions`` (the flow lives in list order), which renumbers
        the positional node ids coordinates are keyed by. :func:`command_from_graph` reports how
        each old id maps to its new one; the coordinates ride that remap so a wired node keeps
        its spot instead of jumping onto the neighbour that inherited its old path. Returns the
        remap (empty when nothing moved), or ``None`` when there is no command to rebuild.
        """
        command = self._model.command
        if command is None:
            return None
        remap: dict[str, str] = {}
        rebuilt = command_from_graph(self._graph, command, id_remap=remap)
        old_positions = {**self._positions, **self.layout()}
        moved = {remap.get(node_id, node_id): pos for node_id, pos in old_positions.items()}
        self.set_layout(moved)
        self._model.set_command(rebuilt)
        self.rebuild()
        self.changed_command.emit()
        return remap

    def request_connect(self, source_id: str, port: str, target_id: str) -> bool:
        """Try to add a connection; reconstructs the tree through the model if valid."""
        ok, reason = can_connect(self._graph, source_id, port, target_id)
        if not ok:
            self.connection_rejected.emit(reason)
            return False
        if self._model.command is None:
            return False
        self._graph.edges.append(GraphEdge(source_id, port, target_id))
        return self._rebuild_from_graph() is not None

    def can_start_from(self, node_id: str, port: str) -> bool:
        """Whether an output port is free to start a new wire from."""
        return self._graph.out_edge(node_id, port) is None

    def begin_reconnect(self, target_id: str) -> tuple[str, str] | None:
        """Tear the wire off ``target_id``'s input so its freed source end can be dragged.

        Grabbing a node's input port pulls the incoming wire off it: the wire is removed (its
        old target becomes a free node) and the now-loose source end is handed back so the view
        can keep dragging it — dropped on another node it reconnects, dropped in empty space it
        stays detached. Returns ``(source id, source port)`` remapped to the rebuilt tree, or
        ``None`` when the node has no incoming wire to grab.
        """
        edge = self._graph.in_edge(target_id)
        if edge is None or self._model.command is None:
            return None
        source_id, source_port = edge.source_id, edge.source_port
        self._graph.edges = [wire for wire in self._graph.edges if wire.key != edge.key]
        remap = self._rebuild_from_graph()
        if remap is None:
            return None
        return remap.get(source_id, source_id), source_port

    def begin_reconnect_from_source(self, source_id: str, port: str) -> str | None:
        """Tear the wire off ``source_id``'s output port so its head can be re-anchored.

        The mirror of :meth:`begin_reconnect`: grabbing an occupied output port pulls the wire
        off its origin, leaving the far (target) end loose. The wire is removed — its old target
        becomes a free node — and that target's id (remapped to the rebuilt tree) is handed back
        so the view can drag a new origin onto it: dropped on another node it reconnects from
        there, dropped in empty space the target stays detached. Returns ``None`` when the port
        has no wire to grab.
        """
        edge = self._graph.out_edge(source_id, port)
        if edge is None or self._model.command is None:
            return None
        target_id = edge.target_id
        self._graph.edges = [wire for wire in self._graph.edges if wire.key != edge.key]
        remap = self._rebuild_from_graph()
        if remap is None:
            return None
        return remap.get(target_id, target_id)

    def disconnect_edge(self, source_id: str, source_port: str, target_id: str) -> bool:
        """Remove one wire; the block it fed becomes a free node again.

        Dropping the wire makes its target unreachable from the flow's start, so
        :func:`command_from_graph` re-derives it (and its whole subtree) as detached —
        the block stays put but no longer runs until it is wired back in.
        """
        if self._model.command is None:
            return False
        key = (source_id, source_port, target_id)
        kept = [edge for edge in self._graph.edges if edge.key != key]
        if len(kept) == len(self._graph.edges):
            return False
        self._graph.edges = kept
        return self._rebuild_from_graph() is not None

    def edge_at_chip(self, scene_point: QPointF) -> EdgeItem | None:
        """The wire whose delay chip is under a scene point — the click target for editing."""
        for wire in self._edges:
            if wire.chip_contains(scene_point):
                return wire
        return None

    def set_wire_delay(self, display: DisplayEdge, ms: int) -> bool:
        """Set the «Пауза» a wire carries (0 removes it), editing the model through the graph.

        The chip maps onto a graph edit, uniform with connect/disconnect: the run of folded
        pause nodes on the wire is replaced by a single «Пауза» of ``ms`` — or, when ``ms`` is
        0, by a direct wire with no pause at all — then the command is rebuilt so both views
        and the ``.ayris`` file agree. A lone existing pause keeps its comment and flags; extra
        folded pauses collapse into the one. Returns ``False`` when there is nothing to edit (no
        command loaded, or the wire's ends are gone).
        """
        if self._model.command is None:
            return False
        source = self._graph.node_by_id(display.source_id)
        target = self._graph.node_by_id(display.target_id)
        if source is None or target is None:
            return False
        ms = max(0, int(ms))
        reused = self._graph.node_by_id(display.sleep_ids[0]) if display.sleep_ids else None
        doomed = set(display.sleep_ids)
        run_key = (display.source_id, display.source_port, display.target_id)
        self._graph.nodes = [node for node in self._graph.nodes if node.id not in doomed]
        self._graph.edges = [
            edge
            for edge in self._graph.edges
            if edge.source_id not in doomed and edge.target_id not in doomed and edge.key != run_key
        ]
        if ms <= 0:
            self._graph.edges.append(
                GraphEdge(display.source_id, display.source_port, display.target_id)
            )
        else:
            node = self._make_delay_node(reused, ms)
            self._graph.nodes.append(node)
            self._graph.edges.append(GraphEdge(display.source_id, display.source_port, node.id))
            self._graph.edges.append(GraphEdge(node.id, MAIN_PORT, display.target_id))
        return self._rebuild_from_graph() is not None

    def _make_delay_node(self, reused: GraphNode | None, ms: int) -> GraphNode:
        """A «Пауза» graph node of ``ms``, reusing an existing pause's id, comment and flags."""
        block = make_delay_block(ms)
        if reused is not None:
            block.enabled = reused.block.enabled
            block.comment = reused.block.comment
            block.on_error = reused.block.on_error
            if reused.block.sound is not None:
                block.sound = reused.block.sound.model_copy(deep=True)
            node_id = reused.id
        else:
            node_id = "__wire_delay__"
        meta = self._catalog.try_get(block.type)
        return GraphNode(
            id=node_id,
            block=block,
            role=role_of(block.type, self._catalog),
            title=meta.title_ru if meta is not None else block.type,
        )

    def _selected_edge(self) -> EdgeItem | None:
        for wire in self._edges:
            if wire.isSelected():
                return wire
        return None

    def delete_selection(self) -> bool:
        """Delete whatever is selected: a wire is disconnected, a node is removed.

        A selected wire wins over a node, so pressing Delete on a highlighted wire cuts
        that connection rather than deleting a node that happens to be selected too.
        """
        wire = self._selected_edge()
        if wire is not None:
            return self.disconnect_edge(wire.source_id, wire.source_port, wire.target_id)
        return self.delete_selected()

    # -- deleting -----------------------------------------------------------

    def delete_selected(self) -> bool:
        """Remove the selected block (and its whole subtree) through the model.

        Deletes exactly what the list view's «Удалить» does — the block at the selected
        path, with any branches it holds — so both views and the shared undo stack stay
        in step. Clearing the selection before the rebuild empties the inspector and stops
        a survivor that inherited the freed path index from silently becoming selected.
        """
        path = self.selected_path()
        if not path:
            return False
        container, index = path[:-1], path[-1]
        if not isinstance(index, int):
            return False
        if self._model.command is None:
            return False
        snapshot = self.positions_by_block()
        if self._model.remove(container, index) is None:
            return False
        self.restore_positions(snapshot)
        self.clearSelection()
        self.rebuild()
        self.changed_command.emit()
        return True

    def valid_target(self, source_id: str, port: str, target_id: str) -> bool:
        return can_connect(self._graph, source_id, port, target_id)[0]

    def node_at_point(self, scene_point: QPointF) -> NodeItem | None:
        for item in self.items(scene_point):
            if isinstance(item, NodeItem):
                return item
        return None

    def node_item(self, node_id: str) -> NodeItem | None:
        return self._nodes.get(node_id)

    def preview_wire(self, source_id: str, port: str, cursor: QPointF) -> QGraphicsPathItem | None:
        """A dashed line item from a port to the cursor, for the live-wire drag."""
        source = self._nodes.get(source_id)
        if source is None:
            return None
        start = source.output_scene_pos(port)
        return self.addPath(bezier_path(start, cursor))

    def preview_wire_to_input(self, target_id: str, cursor: QPointF) -> QGraphicsPathItem | None:
        """A live wire drawn *toward* a fixed input, for the reverse (origin) drag.

        Mirror of :meth:`preview_wire`: the cursor is the loose origin end being placed and the
        curve lands on ``target_id``'s input port, so re-anchoring a wire's start looks the same
        as drawing one.
        """
        target = self._nodes.get(target_id)
        if target is None:
            return None
        end = target.input_scene_pos()
        return self.addPath(bezier_path(cursor, end))

    # -- debug --------------------------------------------------------------

    def highlight_block(self, block_path: str) -> None:
        """Mark the block at ``block_path`` (``actions[1].then[0]``) as running.

        Usually that block is a node card; but a running «Пауза» folded onto a wire has no
        card, so its wire is lit instead — the delay chip turns to the running colour.
        """
        self.clear_running()
        item = self._nodes.get(block_path)
        if item is not None:
            item.set_running(True)
            self._running_id = block_path
            return
        for wire in self._edges:
            if block_path in wire.edge.sleep_ids:
                wire.set_running(True)
                self._running_id = block_path
                return

    def clear_running(self) -> None:
        for wire in self._edges:
            wire.set_running(False)
        if self._running_id is not None:
            item = self._nodes.get(self._running_id)
            if item is not None:
                item.set_running(False)
            self._running_id = None

    def running_item(self) -> NodeItem | None:
        return self._nodes.get(self._running_id) if self._running_id else None

    def toggle_breakpoint(self, node_id: str) -> bool:
        item = self._nodes.get(node_id)
        if item is None:
            return False
        now_on = not item.has_breakpoint()
        item.set_breakpoint(now_on)
        if now_on:
            self._breakpoints.add(node_id)
        else:
            self._breakpoints.discard(node_id)
        self.breakpoint_changed.emit()
        return now_on

    def set_breakpoints(self, node_ids: set[str]) -> None:
        """Replace the breakpoint set (e.g. loaded from the session store) and redraw.

        Silent by design: this reflects stored state into the view, so it does not
        emit :attr:`breakpoint_changed` — only a user toggle should trigger a save.
        """
        self._breakpoints = set(node_ids)
        for node_id, item in self._nodes.items():
            item.set_breakpoint(node_id in self._breakpoints)

    def breakpoints(self) -> set[str]:
        return set(self._breakpoints)

    # -- theme --------------------------------------------------------------

    def refresh_theme(self) -> None:
        self.invalidate(self.sceneRect(), QGraphicsScene.SceneLayer.BackgroundLayer)
        for item in self._nodes.values():
            item.update()
        for wire in self._edges:
            wire.update()

    # -- background ---------------------------------------------------------

    def _visible_scene_rect(self) -> QRectF | None:
        """The scene rectangle currently shown by the first view, or ``None`` if unattached."""
        views = self.views()
        if not views:
            return None
        view = views[0]
        viewport = view.viewport()
        top_left = view.mapToScene(0, 0)
        bottom_right = view.mapToScene(viewport.width(), viewport.height())
        return QRectF(top_left, bottom_right)

    def drawBackground(  # noqa: N802 — Qt override.
        self, painter: QPainter, rect: QRectF | QRect
    ) -> None:
        rect = QRectF(rect)
        # «Глубже и темнее», как в браузерном макете: база холста — фон темы, затемнённый
        # к чёрному (мокап: color-mix(bg 90%, #000)). Здесь чуть сильнее (×0.86), а сияние
        # ниже и компактнее — иначе большие пятна поднимали весь холст и он читался светлее
        # браузерной версии. Множитель — чистое затенение, не палитра, поэтому не токен.
        base = QColor(self._theme.theme.color("background"))
        base = QColor(
            round(base.red() * 0.86), round(base.green() * 0.86), round(base.blue() * 0.86)
        )
        painter.fillRect(rect, base)
        # «Стеклянное» сияние варианта Б: несколько мягких цветных пятен по углам холста
        # поверх тёмной базы, под точечной сеткой. Пятна привязаны к видимой области, а не
        # к dirty-``rect``: Qt отдаёт ``drawBackground`` только грязный прямоугольник, и при
        # перетаскивании ноды это маленький кусок вокруг неё. Центрируй пятно на ``rect`` —
        # каждый частичный перерисованный кусок нарисует своё смещённое сияние и за нодой
        # потянется «шлейф». Видимый прямоугольник стабилен на протяжении перетаскивания,
        # поэтому все частичные перерисовки берут одно и то же сияние.
        bloom_rect = self._visible_scene_rect() or rect
        span = max(bloom_rect.width(), bloom_rect.height())
        # (токен, x-доля, y-доля, радиус-доля, альфа) — расположение пятен как в макете:
        # акцент сверху-слева, info сверху-справа, «звук» снизу, «действие» снизу-слева.
        # Радиусы уменьшены (пятно локально в углу, а не на пол-холста) и альфы снижены,
        # чтобы центр холста оставался тёмным — «глубже», как в браузерной версии.
        aurora = (
            ("accent", 0.20, 0.24, 0.40, 0.15),
            ("info", 0.82, 0.16, 0.36, 0.11),
            ("role_sound", 0.66, 0.92, 0.44, 0.11),
            ("role_action", 0.26, 0.86, 0.40, 0.09),
        )
        for token, fx, fy, fr, alpha in aurora:
            centre = QPointF(
                bloom_rect.left() + bloom_rect.width() * fx,
                bloom_rect.top() + bloom_rect.height() * fy,
            )
            glow = QColor(self._theme.theme.color(token))
            glow.setAlphaF(alpha)
            gradient = QRadialGradient(centre, span * fr)
            gradient.setColorAt(0.0, glow)
            faded = QColor(glow)
            faded.setAlphaF(0.0)
            gradient.setColorAt(1.0, faded)
            painter.fillRect(rect, QBrush(gradient))
        # Dotted grid — dimmed so the darker canvas stays deep and the dots don't lift it.
        dot = QColor(self._theme.theme.color("text_muted"))
        dot.setAlphaF(0.11)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(dot))
        left = rect.left() - (rect.left() % GRID)
        top = rect.top() - (rect.top() % GRID)
        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawEllipse(QPointF(x, y), 1.1, 1.1)
                y += GRID
            x += GRID

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802 — Qt.
        # A left click on empty canvas clears selection so the inspector empties too.
        if event.button() == Qt.MouseButton.LeftButton:
            hit = any(
                isinstance(item, NodeItem | EdgeItem) for item in self.items(event.scenePos())
            )
            if not hit:
                self.clearSelection()
        super().mousePressEvent(event)
