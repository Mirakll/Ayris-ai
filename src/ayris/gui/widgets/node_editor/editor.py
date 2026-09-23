"""Виджет нодового редактора: холст, тулбар, миникарта, палитра — на одной модели.

:class:`NodeEditor` — второе представление того же
:class:`~ayris.gui.widgets.action_list.ActionListModel`, что и список-режим. Он повторяет
контракт :class:`~ayris.gui.widgets.action_list.ActionListView` (сигналы ``block_selected``
и ``changed``, методы ``rebuild``/``selected_path``/``select_path``), поэтому редактор
команды подключает его как альтернативную вкладку без второй копии дерева. Тулбар несёт
палитру блоков (с категориями и поиском), «Упорядочить», привязку к сетке, «Показать всё» и
сброс масштаба; миникарта скрывается крестиком и возвращается кнопкой.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.gui.widgets.action_list import ActionListModel, BlockPath
from ayris.gui.widgets.block_palette import BlockPalette
from ayris.gui.widgets.node_editor.bridge import layout_from_json, layout_to_json
from ayris.gui.widgets.node_editor.layout import auto_layout, free_slot
from ayris.gui.widgets.node_editor.minimap import Minimap
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
        self._minimap = Minimap(self._scene, self._view, theme, parent=self._view)
        self._mm_toggle = QPushButton("🗺", self._view)
        self._mm_toggle.setFixedSize(34, 34)
        self._mm_toggle.hide()
        self._mm_close = QPushButton("✕", self._minimap)
        self._mm_close.setFixedSize(20, 20)
        self._mm_close.move(self._minimap.width() - 24, 4)
        self._mm_close.clicked.connect(self.hide_minimap)
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
        self._mm_toggle.clicked.connect(self._show_minimap)
        theme.theme_changed.connect(self._on_theme_changed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_xs"))
        outer.addLayout(self._build_toolbar())
        outer.addWidget(self._view, 1)

        self._status = ""

    # -- toolbar ------------------------------------------------------------

    def _build_toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(self._theme.metric("spacing_xs"))
        self._add_button = QPushButton("＋ Нода")
        self._add_button.clicked.connect(self._open_palette)
        self._delete_button = QPushButton("🗑 Удалить")
        self._delete_button.setToolTip("Удалить выбранную ноду (Delete)")
        self._delete_button.setEnabled(False)
        self._delete_button.clicked.connect(self.delete_selected)
        self._arrange_button = QPushButton("⇄ Упорядочить")
        self._arrange_button.clicked.connect(self.arrange)
        self._snap_button = QPushButton("▦ Сетка")
        self._snap_button.setCheckable(True)
        self._snap_button.toggled.connect(self._scene.set_snap)
        self._fit_button = QPushButton("⤢ Показать всё")
        self._fit_button.clicked.connect(self._view.fit_all)
        self._reset_button = QPushButton("1:1")
        self._reset_button.clicked.connect(self._view.reset_zoom)
        for button in (
            self._add_button,
            self._delete_button,
            self._arrange_button,
            self._snap_button,
            self._fit_button,
            self._reset_button,
        ):
            bar.addWidget(button)
        bar.addStretch(1)
        return bar

    # -- palette ------------------------------------------------------------

    def _open_palette(self) -> None:
        if self._palette_popup is None:
            popup = BlockPalette(self._theme, catalog=self._catalog, parent=self)
            popup.setWindowFlags(Qt.WindowType.Popup)
            popup.setFixedWidth(264)
            popup.block_chosen.connect(self._insert_block)
            self._palette_popup = popup
        popup = self._palette_popup
        below = self._add_button.mapToGlobal(QPoint(0, self._add_button.height() + 4))
        popup.move(below)
        popup.resize(264, 360)
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

    # -- list-view contract -------------------------------------------------

    def rebuild(self) -> None:
        self._scene.rebuild()
        self._view.grow_scene_rect()
        self._minimap.update()
        self._update_minimap_geometry()
        self._frame_if_pending()

    def selected_path(self) -> BlockPath:
        return self._scene.selected_path()

    def select_path(self, path: BlockPath) -> None:
        self._scene.select_path(path)

    def arrange(self) -> None:
        """Auto-layout the graph left-to-right and frame it."""
        auto_layout(self._scene.graph)
        self._scene.set_layout(self._scene.graph.positions())
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
        self._minimap.update()
        self.changed.emit()

    def _on_rejected(self, message: str) -> None:
        self._status = message

    @property
    def status(self) -> str:
        return self._status

    def _on_theme_changed(self, _theme: object) -> None:
        self._scene.refresh_theme()
        self._minimap.update()

    # -- minimap placement --------------------------------------------------

    def _show_minimap(self) -> None:
        self._minimap.show()
        self._mm_toggle.hide()
        self._update_minimap_geometry()

    def _update_minimap_geometry(self) -> None:
        margin = 14
        vw = self._view.viewport().width()
        vh = self._view.viewport().height()
        self._minimap.move(
            vw - self._minimap.width() - margin, vh - self._minimap.height() - margin
        )
        self._mm_toggle.move(
            vw - self._mm_toggle.width() - margin, vh - self._mm_toggle.height() - margin
        )

    def resizeEvent(self, event: object) -> None:  # noqa: N802 — Qt override.
        super().resizeEvent(event)  # type: ignore[arg-type]
        self._update_minimap_geometry()
        # A pending frame may have been deferred while the viewport was 0×0; now that the
        # canvas has a real size, fit the fresh command to it.
        self._frame_if_pending()

    def showEvent(self, event: object) -> None:  # noqa: N802 — Qt override.
        super().showEvent(event)  # type: ignore[arg-type]
        # Opening a command while the node view sat behind the list view left the frame
        # pending (no viewport to fit to); the switch to «Ноды» shows it — frame it now.
        self._frame_if_pending()

    def hide_minimap(self) -> None:
        self._minimap.hide()
        self._mm_toggle.show()
        self._update_minimap_geometry()
