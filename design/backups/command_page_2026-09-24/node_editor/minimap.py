"""Скрываемая миникарта: обзор графа с рамкой текущего вида.

Обзор сцены в углу холста: ноды — прямоугольники цвета роли, связи — линии, текущий вид —
рамка. Рамку можно перетаскивать, чтобы двигать холст. Карта прячется крестиком и
возвращается кнопкой; автоскрытие — когда граф целиком влезает в окно.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from ayris.gui.widgets.node_editor.bridge import ROLE_TOKENS
from ayris.gui.widgets.node_editor.layout import NODE_HEIGHT, NODE_WIDTH

if TYPE_CHECKING:
    from ayris.gui.theme import ThemeManager
    from ayris.gui.widgets.node_editor.scene import NodeScene
    from ayris.gui.widgets.node_editor.view import NodeView

__all__ = ["Minimap"]

_MARGIN = 40.0


class Minimap(QWidget):
    """A small overview of the whole graph with a draggable current-view rectangle."""

    def __init__(
        self,
        scene: NodeScene,
        view: NodeView,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._scene = scene
        self._view = view
        self._theme = theme
        self.setFixedSize(190, 130)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    # -- geometry -----------------------------------------------------------

    def scene_bounds(self) -> QRectF:
        """The bounding box of all nodes, padded — the area the minimap maps."""
        graph = self._scene.graph
        if not graph.nodes:
            return QRectF(0, 0, 1, 1)
        xs = [node.x for node in graph.nodes]
        ys = [node.y for node in graph.nodes]
        left, top = min(xs), min(ys)
        right = max(x + NODE_WIDTH for x in xs)
        bottom = max(y + NODE_HEIGHT for y in ys)
        return QRectF(left, top, right - left, bottom - top).adjusted(
            -_MARGIN, -_MARGIN, _MARGIN, _MARGIN
        )

    def _scale(self, bounds: QRectF) -> float:
        if bounds.width() <= 0 or bounds.height() <= 0:
            return 1.0
        return min(self.width() / bounds.width(), self.height() / bounds.height())

    def viewport_scene_rect(self) -> QRectF:
        """The scene rectangle currently visible in the view."""
        top_left = self._view.mapToScene(0, 0)
        bottom_right = self._view.mapToScene(
            self._view.viewport().width(), self._view.viewport().height()
        )
        return QRectF(top_left, bottom_right)

    def should_autohide(self) -> bool:
        """Whether the whole graph already fits the view — then the map hides itself."""
        return self.viewport_scene_rect().contains(self.scene_bounds())

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002 — Qt override.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(self._theme.theme.color("surface")))
        bounds = self.scene_bounds()
        scale = self._scale(bounds)

        def to_widget(x: float, y: float) -> QPointF:
            return QPointF((x - bounds.left()) * scale, (y - bounds.top()) * scale)

        graph = self._scene.graph
        accent = QColor(self._theme.theme.color("accent"))
        accent.setAlphaF(0.5)
        painter.setPen(QPen(accent, 1.0))
        for edge in graph.edges:
            source = graph.node_by_id(edge.source_id)
            target = graph.node_by_id(edge.target_id)
            if source is None or target is None:
                continue
            start = to_widget(source.x + NODE_WIDTH, source.y + NODE_HEIGHT / 2)
            end = to_widget(target.x, target.y + NODE_HEIGHT / 2)
            painter.drawLine(start, end)
        painter.setPen(Qt.PenStyle.NoPen)
        for node in graph.nodes:
            painter.setBrush(QBrush(QColor(self._theme.theme.color(ROLE_TOKENS[node.role]))))
            top_left = to_widget(node.x, node.y)
            painter.drawRect(
                QRectF(
                    top_left,
                    QPointF(top_left.x() + NODE_WIDTH * scale, top_left.y() + NODE_HEIGHT * scale),
                )
            )

        view_rect = self.viewport_scene_rect()
        frame = QRectF(
            to_widget(view_rect.left(), view_rect.top()),
            to_widget(view_rect.right(), view_rect.bottom()),
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(self._theme.theme.color("accent")), 1.5))
        painter.drawRect(frame.intersected(QRectF(self.rect())))

    # -- interaction --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        self._center_view_on(event.position())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._center_view_on(event.position())

    def _center_view_on(self, widget_point: QPointF) -> None:
        bounds = self.scene_bounds()
        scale = self._scale(bounds)
        if scale <= 0:
            return
        scene_x = bounds.left() + widget_point.x() / scale
        scene_y = bounds.top() + widget_point.y() / scale
        self._view.centerOn(scene_x, scene_y)
        self.update()
