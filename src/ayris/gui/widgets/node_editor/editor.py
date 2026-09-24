"""Виджет нодового редактора: холст, плавающая капсула-пульт, палитра — на одной модели.

:class:`NodeEditor` — второе представление того же
:class:`~ayris.gui.widgets.action_list.ActionListModel`, что и список-режим. Он повторяет
контракт :class:`~ayris.gui.widgets.action_list.ActionListView` (сигналы ``block_selected``
и ``changed``, методы ``rebuild``/``selected_path``/``select_path``), поэтому редактор
команды подключает его как альтернативную вкладку без второй копии дерева. Инструменты
холста собраны в плавающую капсулу у нижнего края (вариант «Кинематограф»): добавить ноду
из палитры, удалить, «Упорядочить», привязка к сетке, «Показать всё» и сброс масштаба.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.gui.widgets.action_list import ActionListModel, BlockPath
from ayris.gui.widgets.block_palette import BlockPalette
from ayris.gui.widgets.node_editor.bridge import layout_from_json, layout_to_json
from ayris.gui.widgets.node_editor.edge_item import EdgeItem
from ayris.gui.widgets.node_editor.layout import free_slot
from ayris.gui.widgets.node_editor.scene import NodeScene
from ayris.gui.widgets.node_editor.view import NodeView

if TYPE_CHECKING:
    from ayris.core.events import DebugPaused, DebugStepFinished
    from ayris.gui.theme import ThemeManager

__all__ = ["NodeEditor"]


class NodeEditor(QWidget):
    """Node-graph editor over an :class:`ActionListModel`, mirroring the list view."""

    #: The selected block's full path, or an empty tuple — same as the list view.
    block_selected = Signal(object)
    #: A structural change made through the model.
    changed = Signal()
    #: A breakpoint was toggled (double-click on a node); the host persists the set.
    breakpoints_changed = Signal()

    def __init__(
        self,
        model: ActionListModel,
        theme: ThemeManager,
        *,
        catalog: BlockCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._model = model
        self._theme = theme
        self._catalog = catalog if catalog is not None else BlockCatalog()

        self._scene = NodeScene(model, theme, catalog=self._catalog)
        self._view = NodeView(self._scene)
        self._palette_popup: BlockPalette | None = None

        # Set once a fresh command is loaded (reset_layout): the graph must be framed to
        # fit the viewport the first time it is shown at a real size, so a wide command
        # never opens with its branches clipped off the right edge. Cleared after one fit.
        self._needs_frame = False

        self._scene.block_selected.connect(self.block_selected)
        self._scene.block_selected.connect(self._on_scene_selection)
        self._scene.changed_command.connect(self._on_command_changed)
        self._scene.connection_rejected.connect(self._on_rejected)
        # The double-click reaches the scene through the view, which also emits
        # ``breakpoint_toggled``; the scene's own ``breakpoint_changed`` is the single
        # source of truth for the set, so the host persists off that.
        self._scene.breakpoint_changed.connect(self.breakpoints_changed)
        self._view.delete_requested.connect(self.delete_selected)
        self._view.delay_chip_clicked.connect(self._edit_delay)
        self._view.context_menu_requested.connect(self._show_context_menu)
        theme.theme_changed.connect(self._on_theme_changed)

        # Canvas keyboard shortcuts matching the «Кинематограф» context menu: F2 renames the
        # selected node, Ctrl+D duplicates it (Delete is handled by the view). Scoped to the
        # view so they fire only while the canvas has focus and never clash with the window's
        # own bindings; the menu shows the same hints so they aren't a lie.
        rename_sc = QShortcut(QKeySequence(Qt.Key.Key_F2), self._view)
        rename_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        rename_sc.activated.connect(self._rename_selected)
        duplicate_sc = QShortcut(QKeySequence("Ctrl+D"), self._view)
        duplicate_sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        duplicate_sc.activated.connect(self._duplicate_selected)

        # Floating command capsule over the canvas — a child of the view (not its viewport),
        # positioned on resize/show, built before the layout so its first placement has real
        # button sizes.
        self._build_capsule()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._view, 1)

        self._status = ""
        self._position_overlays()

    # -- capsule toolbar ----------------------------------------------------

    def _build_capsule(self) -> None:
        """Build the floating «пульт» capsule at the bottom-centre of the canvas.

        Per the «Кинематограф» variant this single capsule now carries the whole bottom
        bar: an optional host slot on the LEFT (the editor's undo/redo island), then the
        canvas tools (add · delete · arrange · snap · fit · 1:1), then an optional host
        slot on the RIGHT (the editor's статус · Тест · Сохранить). The host slots are
        filled by :meth:`mount_controls` when the node view becomes the active mode and
        hidden otherwise, so the tools stand alone until the editor hands its controls
        over. It is a child of :attr:`_view`, positioned by :meth:`_position_overlays`.
        """
        capsule = QFrame(self._view)
        capsule.setProperty("capsule", True)
        self._capsule = capsule
        layout = QHBoxLayout(capsule)
        gap = self._theme.metric("spacing_xs")
        layout.setContentsMargins(gap, gap, gap, gap)
        layout.setSpacing(gap)

        # Left host slot (undo/redo island) + its separator, ahead of the tools.
        self._host_left = self._make_host()
        self._sep_left = self._make_separator(capsule)
        layout.addWidget(self._host_left)
        layout.addWidget(self._sep_left)

        self._add_button = self._capsule_button("＋", "Добавить ноду")
        self._add_button.clicked.connect(self._open_palette)
        self._delete_button = self._capsule_button("🗑", "Удалить выбранную ноду (Delete)")
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self.delete_selected)
        self._arrange_button = self._capsule_button("⇄", "Упорядочить")
        self._arrange_button.clicked.connect(self.arrange)
        self._snap_button = self._capsule_button("▦", "Привязка к сетке")
        self._snap_button.setCheckable(True)
        self._snap_button.toggled.connect(self._scene.set_snap)
        self._fit_button = self._capsule_button("⤢", "Показать всё")
        self._fit_button.clicked.connect(self._view.fit_all)
        self._reset_button = self._capsule_button("1:1", "Масштаб 1:1")
        self._reset_button.clicked.connect(self._view.reset_zoom)

        layout.addWidget(self._add_button)
        layout.addWidget(self._delete_button)
        layout.addWidget(self._make_separator(capsule))
        layout.addWidget(self._arrange_button)
        layout.addWidget(self._snap_button)
        layout.addWidget(self._make_separator(capsule))
        layout.addWidget(self._fit_button)
        layout.addWidget(self._reset_button)

        # Right host slot (статус · Тест · Сохранить) + its separator, after the tools.
        self._sep_right = self._make_separator(capsule)
        self._host_right = self._make_host()
        layout.addWidget(self._sep_right)
        layout.addWidget(self._host_right)

        for slot in (self._host_left, self._sep_left, self._sep_right, self._host_right):
            slot.hide()
        capsule.adjustSize()

    def _make_host(self) -> QWidget:
        """A transparent inline container for controls the editor mounts into the capsule."""
        host = QWidget(self._capsule)
        host.setProperty("transparent", True)
        host_layout = QHBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(self._theme.metric("spacing_xs"))
        return host

    def _capsule_button(self, glyph: str, tooltip: str) -> QPushButton:
        button = QPushButton(glyph)
        button.setToolTip(tooltip)
        button.setProperty("iconButton", True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        return button

    def _make_separator(self, parent: QWidget) -> QFrame:
        line = QFrame(parent)
        line.setProperty("vline", True)
        line.setFixedWidth(1)
        line.setFixedHeight(self._theme.metric("icon_md"))
        return line

    def _position_overlays(self) -> None:
        """Centre the floating command capsule along the bottom edge of the canvas.

        The capsule carries the whole bottom bar (tools plus the editor's mounted
        undo/redo and статус · Тест · Сохранить), floating over the graph so the canvas
        fills the whole panel beneath it — no separate docked strip, no dead band below.
        """
        if not hasattr(self, "_capsule"):
            return
        viewport = self._view.viewport()
        margin = self._theme.metric("spacing_lg")
        # Never let the capsule spill past the canvas edges: cap it to the viewport so the
        # status note (its one elastic, eliding item) gives up width instead of shoving
        # «Тест / Сохранить» off the right edge («и в нижней панели … Сохранить → хран»).
        # The toolbar and buttons keep their natural size; only the status elides.
        self._capsule.setMaximumWidth(max(1, viewport.width() - 2 * margin))
        self._capsule.adjustSize()
        x = max(0, (viewport.width() - self._capsule.width()) // 2)
        y = max(0, viewport.height() - self._capsule.height() - margin)
        self._capsule.move(x, y)
        self._capsule.raise_()

    def mount_controls(self, left: QWidget | None, right: QWidget | None) -> None:
        """Mount the editor's host controls into the capsule's left / right slots.

        The macro editor hands over its undo/redo island (``left``) and its
        статус · Тест · Сохранить group (``right``) while the node canvas is the shown
        mode, so the whole bottom bar reads as one «Кинематограф» capsule instead of a
        separate footer row. Passing ``None`` for a slot hides it; the caller then
        re-parents that group back under the tabs.
        """
        self._mount_host(self._host_left, self._sep_left, left)
        self._mount_host(self._host_right, self._sep_right, right)
        self._position_overlays()

    def _mount_host(self, host: QWidget, separator: QFrame, widget: QWidget | None) -> None:
        if widget is None:
            host.hide()
            separator.hide()
            return
        current = widget.parentWidget()
        current_layout = current.layout() if current is not None else None
        if current_layout is not None:
            current_layout.removeWidget(widget)
        host_layout = host.layout()
        if host_layout is not None:
            host_layout.addWidget(widget)
        widget.show()
        host.show()
        separator.show()

    # -- palette ------------------------------------------------------------

    def _open_palette(self) -> None:
        if self._palette_popup is None:
            popup = BlockPalette(self._theme, catalog=self._catalog, parent=self)
            # Frameless + translucent so only the QSS-rounded surface shows — no square
            # window corners peeking out behind the radius («у панели есть углы»).
            popup.setWindowFlags(
                Qt.WindowType.Popup
                | Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.NoDropShadowWindowHint
            )
            popup.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            popup.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            popup.setFixedWidth(264)
            popup.block_chosen.connect(self._insert_block)
            self._palette_popup = popup
        popup = self._palette_popup
        popup.resize(264, 360)
        # The capsule sits at the bottom of the canvas, so the palette opens UPWARDS —
        # above the «＋» button rather than below it, where it would run off the window.
        above = self._add_button.mapToGlobal(QPoint(0, -popup.height() - 4))
        popup.move(above)
        popup.show()

    def _insert_block(self, block_type: str) -> None:
        if self._palette_popup is not None:
            self._palette_popup.hide()
        command = self._model.command
        if command is None:
            return
        selected = self._scene.selected_path()
        # Snapshot coordinates by block identity BEFORE the insert renumbers the paths, so
        # the blocks after the insertion point keep their spots instead of inheriting a
        # shifted neighbour's — otherwise the tail of the flow flings itself off-canvas.
        snapshot = self._scene.positions_by_block()
        # A new node is added FREE: it drops in unwired (detached) and the user drags the
        # wires to it. Position in the list no longer decides the flow — the wires do — so
        # it always lands at the root tail; the selected (or last) node is only an anchor
        # for a tidy spot on the canvas.
        end = len(command.actions)
        anchor_path: BlockPath | None = (
            selected if selected else (("actions", end - 1) if end > 0 else None)
        )
        path = self._model.insert_type(block_type, ("actions",), end)
        if path is None:
            self._status = "Слишком глубокая вложенность блоков."
            return
        new_block = self._model.block_at(path)
        if new_block is not None:
            new_block.detached = True
        # Keep every surviving node where it was, then rebuild so the new node has a real
        # path and the layout knows every existing coordinate.
        self._scene.restore_positions(snapshot)
        self.rebuild()
        self._place_new_node(path, anchor_path)
        self.select_path(path)
        self._ensure_visible(path)
        self._on_command_changed()

    def _place_new_node(self, path: BlockPath, anchor_path: BlockPath | None) -> None:
        """Give the freshly inserted node a free spot in flow next to its anchor.

        Placed by :func:`~ayris.gui.widgets.node_editor.layout.free_slot`: one step right of
        the anchor block (the node it follows), nudged down if a later sibling already holds
        that cell. This replaces the old «drop at viewport centre», which stacked repeated
        inserts on one point and flung a node off-canvas whenever the view was panned or
        zoomed away from the flow.
        """
        node_id = self._node_id_for_path(path)
        if node_id is None:
            return
        layout = self._scene.layout()
        anchor_id = self._node_id_for_path(anchor_path) if anchor_path is not None else None
        anchor = layout.get(anchor_id) if anchor_id is not None else None
        occupied = [pos for nid, pos in layout.items() if nid != node_id]
        layout[node_id] = free_slot(anchor, occupied)
        self._scene.set_layout(layout)
        self.rebuild()

    def _ensure_visible(self, path: BlockPath) -> None:
        """Scroll the minimum amount to bring a node into view, if it isn't already."""
        node_id = self._node_id_for_path(path)
        if node_id is None:
            return
        item = self._scene.node_item(node_id)
        if item is not None:
            self._view.ensureVisible(item)

    def _node_id_for_path(self, path: BlockPath) -> str | None:
        for node_id, node_path in self._scene._path_by_id.items():
            if node_path == path:
                return node_id
        return None

    # -- deleting -----------------------------------------------------------

    def delete_selected(self) -> None:
        """Delete the selected node through the model, like the list view's «Удалить».

        The scene removes the block and its subtree, rebuilds and announces the change;
        the shared undo stack (Ctrl+Z) can take it back, so no confirmation is needed.
        A selected wire is cut instead — its block becomes a free node again.
        """
        self._scene.delete_selection()

    def _on_scene_selection(self, path: object) -> None:
        self._delete_button.setEnabled(bool(path))

    # -- context menu -------------------------------------------------------

    def _show_context_menu(self, global_pos: QPoint, node_id: object) -> None:
        """Open the right-click menu — «в браузер версии … ПКМ … добавить/упорядочить/показать всё».

        Over a node (the view has already selected it) the menu duplicates, toggles or deletes
        it; over empty canvas it adds a node, arranges the flow or fits it to the view — the
        same commands the bottom capsule carries, reachable without aiming for the pill.
        """
        menu = QMenu(self)
        # Frameless + translucent so the QSS border-radius isn't boxed by square window
        # corners («в двойном клике у панели есть углы»).
        menu.setWindowFlags(
            menu.windowFlags()
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        menu.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        if node_id:
            block = self._model.block_at(self._scene.selected_path())
            enabled = block.enabled if block is not None else True
            # Shortcut hints are shown right-aligned by Qt (the mockup's `.ci .k` spans);
            # the matching QShortcuts on the view make them work from the canvas too.
            rename = menu.addAction("Переименовать", self._rename_selected)
            rename.setShortcut(QKeySequence(Qt.Key.Key_F2))
            duplicate = menu.addAction("Дублировать", self._duplicate_selected)
            duplicate.setShortcut(QKeySequence("Ctrl+D"))
            menu.addAction("Выключить" if enabled else "Включить", self._toggle_enabled)
            menu.addSeparator()
            delete = menu.addAction("Удалить", self.delete_selected)
            delete.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        else:
            menu.addAction("Добавить ноду", self._open_palette)
            menu.addAction("Упорядочить", self.arrange)
            menu.addAction("Показать всё", self._view.fit_all)
        menu.exec(global_pos)

    def _duplicate_selected(self) -> None:
        """Deep-copy the selected node as a fresh free node beside it, like the list «Дублировать».

        The copy drops in detached (unwired) next to the original — the canvas paradigm where
        wires, not list order, decide the flow — so duplicating never silently reroutes the
        chain. Coordinates are snapshotted by block identity so the renumbered siblings keep
        their spots while the new node is placed by :meth:`_place_new_node`.
        """
        path = self._scene.selected_path()
        index = path[-1] if path else None
        if not isinstance(index, int):
            return
        container = path[:-1]
        snapshot = self._scene.positions_by_block()
        new_path = self._model.duplicate(container, index)
        if new_path is None:
            return
        new_block = self._model.block_at(new_path)
        if new_block is not None:
            new_block.detached = True
        self._scene.restore_positions(snapshot)
        self.rebuild()
        self._place_new_node(new_path, path)
        self.select_path(new_path)
        self._ensure_visible(new_path)
        self._on_command_changed()

    def _toggle_enabled(self) -> None:
        """Flip the selected node's «выключен» state through the model, then redraw it dimmed."""
        path = self._scene.selected_path()
        index = path[-1] if path else None
        if not isinstance(index, int):
            return
        block = self._model.block_at(path)
        if block is None:
            return
        self._model.set_enabled(path[:-1], index, enabled=not block.enabled)
        self.rebuild()
        self.select_path(path)
        self._on_command_changed()

    def _rename_selected(self) -> None:
        """Rename the selected node — the "Переименовать" (F2) of the «Кинематограф» menu.

        A block has no name field of its own, so the node's name is stored in its
        ``comment``: it already doubles as the card's title (see ``bridge._build_list``) and
        as the list view's row suffix, and it round-trips with the command. The prompt is
        seeded with the current comment — clearing it reverts the card to the block's default
        title. Mirrors the list view's «Комментарий…» and the enable-toggle model pattern.
        """
        path = self._scene.selected_path()
        index = path[-1] if path else None
        if not isinstance(index, int):
            return
        block = self._model.block_at(path)
        if block is None:
            return
        text, ok = QInputDialog.getText(self, "Переименовать ноду", "Название:", text=block.comment)
        if ok:
            self._model.set_comment(path[:-1], index, text)
            self.rebuild()
            self.select_path(path)
            self._on_command_changed()

    # -- wire delay ---------------------------------------------------------

    def _edit_delay(self, edge: object) -> None:
        """Edit the «Пауза» a wire carries, from a click on its delay chip.

        Opens a small millisecond prompt seeded with the wire's current delay; confirming
        sets it on the model through the scene (which rebuilds and announces the change), so
        «0» drops the pause and any other value folds a single «Пауза» onto the wire.
        Cancelling leaves everything untouched.
        """
        if not isinstance(edge, EdgeItem):
            return
        value, ok = QInputDialog.getInt(
            self,
            "Задержка на связи",
            "Пауза перед следующим блоком, мс:",
            edge.delay_ms,
            0,
            3_600_000,
            50,
        )
        if ok:
            self._scene.set_wire_delay(edge.edge, value)

    # -- list-view contract -------------------------------------------------

    def rebuild(self) -> None:
        self._scene.rebuild()
        self._view.grow_scene_rect()
        self._frame_if_pending()

    def selected_path(self) -> BlockPath:
        return self._scene.selected_path()

    def select_path(self, path: BlockPath) -> None:
        self._scene.select_path(path)

    def arrange(self) -> None:
        """Auto-layout the graph left-to-right and frame it.

        Delays folded onto wires are excluded from the layout, so «Упорядочить» packs one
        column per visible node instead of leaving a hole where a chip-folded «Пауза» sits.
        """
        self._scene.auto_arrange()
        self.rebuild()
        self._view.fit_all()

    # -- layout persistence -------------------------------------------------

    def layout_json(self) -> dict[str, list[float]]:
        """Node coordinates as JSON-ready UI metadata, saved alongside the command."""
        return layout_to_json(self._scene.graph)

    def set_layout_json(self, data: object) -> None:
        self._scene.set_layout(layout_from_json(data))

    def reset_layout(self) -> None:
        """Forget saved coordinates so the next rebuild auto-lays-out a fresh command.

        A fresh command also needs framing: its auto-layout starts near the scene origin
        and a branching or long command spills past the viewport, so the graph is fitted
        to the view the first time it is shown (or right now if it is already visible).
        """
        self._scene.set_layout({})
        self._needs_frame = True

    def _frame_if_pending(self) -> None:
        """Fit the graph to the viewport once, when it is finally shown at a real size.

        The frame is deferred until the canvas actually has a viewport (a command can be
        loaded while the node view is hidden behind the list view, where the viewport is
        0×0 and ``fit_all`` would compute nonsense). Framing on the first real show keeps
        every node inside the visible area instead of clipped at the right edge.
        """
        if not self._needs_frame:
            return
        viewport = self._view.viewport()
        if not self.isVisible() or viewport.width() <= 1 or viewport.height() <= 1:
            return
        self._needs_frame = False
        self._view.fit_all()

    # -- debug --------------------------------------------------------------

    def on_debug_paused(self, event: DebugPaused) -> None:
        self._scene.highlight_block(event.block_path)
        self._view.center_on_node(event.block_path)

    def on_debug_step(self, event: DebugStepFinished) -> None:
        self._scene.highlight_block(event.block_path)

    def clear_debug(self) -> None:
        self._scene.clear_running()

    def breakpoints(self) -> set[str]:
        return self._scene.breakpoints()

    def set_breakpoints(self, node_ids: set[str]) -> None:
        """Reflect stored breakpoints into the canvas (no save is triggered)."""
        self._scene.set_breakpoints(node_ids)

    # -- signals ------------------------------------------------------------

    def _on_command_changed(self) -> None:
        self.changed.emit()

    def _on_rejected(self, message: str) -> None:
        self._status = message

    @property
    def status(self) -> str:
        return self._status

    def _on_theme_changed(self, _theme: object) -> None:
        self._scene.refresh_theme()

    # -- events -------------------------------------------------------------

    def resizeEvent(self, event: object) -> None:  # noqa: N802 — Qt override.
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._position_overlays()
        # A pending frame may have been deferred while the viewport was 0×0; now that the
        # canvas has a real size, fit the fresh command to it.
        self._frame_if_pending()

    def showEvent(self, event: object) -> None:  # noqa: N802 — Qt override.
        super().showEvent(event)  # type: ignore[arg-type]
        self._position_overlays()
        # Opening a command while the node view sat behind the list view left the frame
        # pending (no viewport to fit to); the switch to «Ноды» shows it — frame it now.
        self._frame_if_pending()
