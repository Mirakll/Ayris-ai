"""Холст нодового редактора: панорамирование, зум к курсору, выделение, живой провод.

``QGraphicsView`` над :class:`~ayris.gui.widgets.node_editor.scene.NodeScene`. Панорама —
средней кнопкой и пробелом с левой; зум колесом к позиции курсора с ограничением масштаба;
выделение рамкой и групповое перемещение — штатным rubber-band. От выходного порта тянется
живой провод: валидная цель подсвечивается, невалидная связь не создаётся. «Показать всё» и
сброс масштаба — для тулбара.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtWidgets import QGraphicsView

from ayris.gui.widgets.node_editor.bridge import MAIN_PORT
from ayris.gui.widgets.node_editor.edge_item import EdgeItem
from ayris.gui.widgets.node_editor.node_item import NodeItem
from ayris.gui.widgets.node_editor.scene import NodeScene

if TYPE_CHECKING:
    from PySide6.QtWidgets import QGraphicsPathItem, QWidget

__all__ = ["NodeView"]

_MIN_SCALE: Final = 0.4
_MAX_SCALE: Final = 2.2
_ZOOM_IN: Final = 1.1
_ZOOM_OUT: Final = 0.9

#: The scrollable canvas when the graph is small — big enough to pan comfortably around a
#: few nodes. A larger graph grows the scene rect to fit it (see :meth:`grow_scene_rect`),
#: so a long left-to-right chain never runs off a fixed edge and out of reach.
_BASE_SCENE_RECT: Final = QRectF(-2000, -2000, 4000, 4000)
#: Padding kept around the graph's bounding box so there is room to pan past the last node.
_SCENE_MARGIN: Final = 600.0


class NodeView(QGraphicsView):
    """Pannable, zoomable canvas with rubber-band selection and live wiring."""

    #: Emitted when a node is double-clicked to toggle a breakpoint (its node id).
    breakpoint_toggled = Signal(str)
    #: Emitted when the user presses Delete/Backspace to remove the selected node.
    delete_requested = Signal()

    def __init__(self, scene: NodeScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self._node_scene = scene
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Снять штатную рамку QFrame: сцена сама заливает фон, а утопленный контур
        # вьюпорта проступал по верхнему и нижнему краю холста «квадратной рамкой»
        # (слева и справа его закрывают панели сплиттера). Так же делает _ScrollPage.
        # Вьюпорт делаем прозрачным, иначе за сценой остаётся залитая панель.
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setStyleSheet("QGraphicsView { border: none; background: transparent; }")
        self.viewport().setAutoFillBackground(False)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSceneRect(_BASE_SCENE_RECT)
        self._space = False
        self._panning = False
        self._pan_start = QPoint()
        self._connect_source: tuple[str, str] | None = None
        #: Reverse drag: a fixed input end whose origin is being re-anchored (the wire's
        #: output end was grabbed). Only one of the two is ever set at a time.
        self._connect_target: str | None = None
        self._preview: QGraphicsPathItem | None = None

    # -- zoom ---------------------------------------------------------------

    @property
    def scale_factor(self) -> float:
        return self.transform().m11()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 — Qt override.
        factor = _ZOOM_IN if event.angleDelta().y() > 0 else _ZOOM_OUT
        target = self.scale_factor * factor
        if target < _MIN_SCALE or target > _MAX_SCALE:
            return
        self.scale(factor, factor)

    def reset_zoom(self) -> None:
        self.resetTransform()

    def fit_all(self) -> None:
        """Frame the whole graph, then clamp the zoom into the allowed range."""
        rect = self._node_scene.itemsBoundingRect()
        if rect.isEmpty():
            return
        self.fitInView(rect.adjusted(-60, -60, 60, 60), Qt.AspectRatioMode.KeepAspectRatio)
        if self.scale_factor > _MAX_SCALE:
            self.resetTransform()
            self.scale(_MAX_SCALE, _MAX_SCALE)
        elif self.scale_factor < _MIN_SCALE:
            self.resetTransform()
            self.scale(_MIN_SCALE, _MIN_SCALE)

    def center_on_node(self, node_id: str) -> None:
        item = self._node_scene.node_item(node_id)
        if item is not None:
            self.centerOn(item)

    def grow_scene_rect(self) -> None:
        """Grow the scrollable canvas to cover the whole graph, keeping a base minimum.

        The scene rect used to be pinned at a fixed ±5000 box, so a long left-to-right
        chain (auto-layout places each node one column further right, unbounded) eventually
        ran past the edge and its tail became unreachable — you could not pan to it. Uniting
        the graph's padded bounding box with a comfortable base keeps short graphs roomy and
        long ones fully reachable, while never shrinking below what is on screen.
        """
        items = self._node_scene.itemsBoundingRect()
        if items.isEmpty():
            self.setSceneRect(_BASE_SCENE_RECT)
            return
        padded = items.adjusted(-_SCENE_MARGIN, -_SCENE_MARGIN, _SCENE_MARGIN, _SCENE_MARGIN)
        self.setSceneRect(padded.united(_BASE_SCENE_RECT))

    # -- panning ------------------------------------------------------------

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override.
        if event.key() == Qt.Key.Key_Space:
            self._space = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            # Ask for a delete only when a node OR a wire is selected, so the key is a
            # no-op on empty canvas rather than a swallowed event. The scene decides which
            # to act on — a selected wire is cut, otherwise the node is removed.
            if self.selected_nodes() or self.selected_edges():
                self.delete_requested.emit()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802 — Qt override.
        if event.key() == Qt.Key.Key_Space:
            self._space = False
            self.unsetCursor()
        super().keyReleaseEvent(event)

    # -- mouse: pan, live wire, selection -----------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if event.button() == Qt.MouseButton.MiddleButton or (
            self._space and event.button() == Qt.MouseButton.LeftButton
        ):
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            scene_point = self.mapToScene(event.position().toPoint())
            node = self._node_scene.node_at_point(scene_point)
            if node is not None:
                port = node.output_port_at(scene_point)
                if port is not None:
                    if self._node_scene.can_start_from(node.node_id, port):
                        self.begin_connection(node.node_id, port)
                        event.accept()
                        return
                    # Occupied output port: grab that wire by its origin and tear it off, then
                    # drag a new origin onto the freed (fixed) target end — the mirror of the
                    # input-port grab below, so a wire is re-routable from either end.
                    target = self._node_scene.begin_reconnect_from_source(node.node_id, port)
                    if target is not None:
                        self.begin_reverse_connection(target)
                        event.accept()
                        return
                # Grab the wire by the input port and tear it off: the freed source end is
                # handed back to keep dragging, so the drop either reconnects it elsewhere or
                # (in empty space) leaves the block detached.
                if node.input_port_at(scene_point):
                    reconnect = self._node_scene.begin_reconnect(node.node_id)
                    if reconnect is not None:
                        self.begin_connection(*reconnect)
                        event.accept()
                        return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if self._panning:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            h = self.horizontalScrollBar()
            v = self.verticalScrollBar()
            h.setValue(h.value() - delta.x())
            v.setValue(v.value() - delta.y())
            event.accept()
            return
        if self._connecting:
            self.update_connection(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if self._panning and event.button() in (
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        ):
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        if self._connecting:
            self.finish_connection(self.mapToScene(event.position().toPoint()))
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        scene_point = self.mapToScene(event.position().toPoint())
        node = self._node_scene.node_at_point(scene_point)
        if node is not None:
            self._node_scene.toggle_breakpoint(node.node_id)
            self.breakpoint_toggled.emit(node.node_id)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # -- live wire (also driven directly by tests) --------------------------

    @property
    def _connecting(self) -> bool:
        return self._connect_source is not None or self._connect_target is not None

    def begin_connection(self, source_id: str, port: str) -> None:
        """Start dragging a wire *from* a fixed output end toward a target's input."""
        self._connect_source = (source_id, port)
        self._connect_target = None
        self._clear_preview()

    def begin_reverse_connection(self, target_id: str) -> None:
        """Start dragging a new origin *onto* a fixed input end (the output end was grabbed)."""
        self._connect_target = target_id
        self._connect_source = None
        self._clear_preview()

    def update_connection(self, scene_point: QPointF) -> None:
        self._clear_preview()
        if self._connect_source is not None:
            source_id, port = self._connect_source
            self._preview = self._node_scene.preview_wire(source_id, port, scene_point)
        elif self._connect_target is not None:
            self._preview = self._node_scene.preview_wire_to_input(
                self._connect_target, scene_point
            )

    def finish_connection(self, scene_point: QPointF) -> bool:
        self._clear_preview()
        source = self._connect_source
        target = self._connect_target
        self._connect_source = None
        self._connect_target = None
        # A drop anywhere on the target node counts — its input (or, in reverse, its main
        # output) is the connection point; a drop in empty space leaves the block detached.
        node = self._node_scene.node_at_point(scene_point)
        if node is None:
            return False
        if source is not None:
            source_id, port = source
            return self._node_scene.request_connect(source_id, port, node.node_id)
        if target is not None:
            return self._node_scene.request_connect(node.node_id, MAIN_PORT, target)
        return False

    def _clear_preview(self) -> None:
        if self._preview is not None:
            self._node_scene.removeItem(self._preview)
            self._preview = None

    # -- misc ---------------------------------------------------------------

    def selected_nodes(self) -> list[NodeItem]:
        return [item for item in self.scene().selectedItems() if isinstance(item, NodeItem)]

    def selected_edges(self) -> list[EdgeItem]:
        return [item for item in self.scene().selectedItems() if isinstance(item, EdgeItem)]
