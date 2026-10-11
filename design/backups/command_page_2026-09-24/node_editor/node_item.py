"""Нода-карточка блока на сцене: заголовок с ролью, сводка, порты.

Компактная карточка по эталонному мокапу: левый accent-bar цвета роли, шапка с иконкой,
русским названием и ролью капсом, одна строка сводки с обрезкой, порты вход-слева /
выход(ы)-справа. Состояния «выключен», «опасный», «выделен» и «выполняется» — визуально.
Цвета и подписи берутся из токенов темы и каталога задачи 33, не хардкодятся. Рисование —
только заливки и текст, без blur-эффектов; отрисовка кэшируется, пересчёта раскладки в
``paint`` нет.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)

from ayris.gui.widgets.node_editor.bridge import (
    BRANCH_LABELS,
    MAIN_PORT,
    ROLE_LABELS,
    ROLE_TOKENS,
    GraphNode,
)
from ayris.gui.widgets.node_editor.layout import GRID, NODE_HEIGHT, NODE_WIDTH

if TYPE_CHECKING:
    from ayris.gui.theme import ThemeManager

__all__ = ["NodeItem"]

_RADIUS: Final = 12.0
_ACCENT_BAR: Final = 4.0
_HEADER_H: Final = 40.0
_PORT_R: Final = 6.5
_PAD: Final = 12.0


class NodeItem(QGraphicsObject):
    """One command block drawn as a movable, selectable card with ports."""

    def __init__(
        self,
        node: GraphNode,
        theme: ThemeManager,
        *,
        summary: str,
        danger: bool,
        snap: bool = False,
    ) -> None:
        super().__init__()
        self._node = node
        self._theme = theme
        self._summary = summary
        self._danger = danger
        self._snap = snap
        self._running = False
        self._breakpoint = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)
        self.setZValue(1.0)
        self.setPos(node.x, node.y)

    # -- identity -----------------------------------------------------------

    @property
    def node(self) -> GraphNode:
        return self._node

    @property
    def node_id(self) -> str:
        return self._node.id

    def output_ports(self) -> list[str]:
        """Output port names top-to-bottom: the sequential main plus any branches."""
        return [MAIN_PORT, *self._node.branch_ports]

    # -- geometry -----------------------------------------------------------

    def boundingRect(self) -> QRectF:  # noqa: N802 — Qt override.
        # A little margin so the port circles and the selection glow are not clipped.
        return QRectF(-_PORT_R - 2, -6, NODE_WIDTH + 2 * _PORT_R + 4, NODE_HEIGHT + 12)

    def input_scene_pos(self) -> QPointF:
        return self.mapToScene(QPointF(0.0, NODE_HEIGHT / 2))

    def output_port_at(self, scene_point: QPointF) -> str | None:
        """The output port near a scene point (within a grab radius), or ``None``."""
        local = self.mapFromScene(scene_point)
        ports = self.output_ports()
        for index, port in enumerate(ports):
            fraction = (index + 1) / (len(ports) + 1)
            centre = QPointF(NODE_WIDTH, NODE_HEIGHT * fraction)
            if _distance(local, centre) <= _PORT_R + 4:
                return port
        return None

    def input_port_at(self, scene_point: QPointF) -> bool:
        local = self.mapFromScene(scene_point)
        return _distance(local, QPointF(0.0, NODE_HEIGHT / 2)) <= _PORT_R + 4

    def output_scene_pos(self, port: str) -> QPointF:
        ports = self.output_ports()
        try:
            index = ports.index(port)
        except ValueError:
            index = 0
        fraction = (index + 1) / (len(ports) + 1)
        return self.mapToScene(QPointF(NODE_WIDTH, NODE_HEIGHT * fraction))

    # -- state --------------------------------------------------------------

    def set_running(self, running: bool) -> None:
        if running != self._running:
            self._running = running
            self.update()

    def set_breakpoint(self, value: bool) -> None:
        if value != self._breakpoint:
            self._breakpoint = value
            self.update()

    def has_breakpoint(self) -> bool:
        return self._breakpoint

    def set_snap(self, snap: bool) -> None:
        self._snap = snap

    def refresh(self, *, summary: str, danger: bool, title: str) -> None:
        self._summary = summary
        self._danger = danger
        self._node.title = title
        self.update()

    # -- movement -----------------------------------------------------------

    def itemChange(  # noqa: N802 — Qt override.
        self,
        change: QGraphicsItem.GraphicsItemChange,
        value: object,
    ) -> object:
        change_kind = QGraphicsItem.GraphicsItemChange
        if change == change_kind.ItemPositionChange and self._snap and isinstance(value, QPointF):
            x = round(value.x() / GRID) * GRID
            y = round(value.y() / GRID) * GRID
            return QPointF(x, y)
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            if isinstance(value, QPointF):
                self._node.x = value.x()
                self._node.y = value.y()
            scene = self.scene()
            notify = getattr(scene, "notify_node_moved", None)
            if callable(notify):
                notify(self)
        return super().itemChange(change, value)

    # -- painting -----------------------------------------------------------

    def _color(self, token: str) -> QColor:
        return QColor(self._theme.theme.color(token))

    def _role_color(self) -> QColor:
        return self._color(ROLE_TOKENS[self._node.role])

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,  # noqa: ARG002
        widget: QWidget | None = None,  # noqa: ARG002
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        role = self._role_color()
        surface = self._color("surface")
        border = self._color("border")
        body = QRectF(0, 0, NODE_WIDTH, NODE_HEIGHT)

        if not self._node.enabled:
            painter.setOpacity(0.55)

        # Card body.
        card = QPainterPath()
        card.addRoundedRect(body, _RADIUS, _RADIUS)
        painter.fillPath(card, QBrush(surface))

        # Header band tinted by the role colour — intersected with the card so the top
        # corners follow the rounded outline and the fill never bleeds past the edge.
        header_path = QPainterPath()
        header_path.addRect(QRectF(0, 0, NODE_WIDTH, _HEADER_H))
        painter.fillPath(
            header_path.intersected(card),
            QBrush(_mix(role, self._color("surface_highlight"), 0.22)),
        )

        # Left accent bar, likewise clipped to the rounded card so its corners stay inside.
        accent_path = QPainterPath()
        accent_path.addRect(QRectF(0, 0, _ACCENT_BAR, NODE_HEIGHT))
        painter.fillPath(accent_path.intersected(card), QBrush(role))

        # Icon square.
        icon_rect = QRectF(_PAD, 9, 22, 22)
        icon_path = QPainterPath()
        icon_path.addRoundedRect(icon_rect, 6, 6)
        painter.fillPath(icon_path, QBrush(_mix(role, surface, 0.30)))

        # Title (elided) and role label.
        painter.setPen(QPen(self._color("text_primary")))
        title_font = QFont(painter.font())
        title_font.setPixelSize(13)
        title_font.setBold(True)
        painter.setFont(title_font)
        title_rect = QRectF(_PAD + 30, 6, NODE_WIDTH - _PAD - 40, 17)
        metrics = painter.fontMetrics()
        elided = metrics.elidedText(
            self._node.title, Qt.TextElideMode.ElideRight, int(title_rect.width())
        )
        painter.drawText(title_rect, int(Qt.AlignmentFlag.AlignVCenter), elided)

        role_font = QFont(painter.font())
        role_font.setPixelSize(9)
        role_font.setBold(True)
        painter.setFont(role_font)
        painter.setPen(QPen(role))
        role_rect = QRectF(_PAD + 30, 22, NODE_WIDTH - _PAD - 40, 13)
        painter.drawText(
            role_rect, int(Qt.AlignmentFlag.AlignVCenter), ROLE_LABELS[self._node.role].upper()
        )

        # Danger glyph.
        if self._danger:
            painter.setPen(QPen(self._color("error")))
            painter.drawText(
                QRectF(NODE_WIDTH - 24, 6, 18, 18),
                int(Qt.AlignmentFlag.AlignCenter),
                "⚠",
            )

        # Summary line, ellipsised.
        painter.setPen(QPen(self._color("text_secondary")))
        summary_font = QFont(painter.font())
        summary_font.setPixelSize(12)
        summary_font.setBold(False)
        painter.setFont(summary_font)
        summary_rect = QRectF(
            _PAD, _HEADER_H + 6, NODE_WIDTH - 2 * _PAD, NODE_HEIGHT - _HEADER_H - 10
        )
        summary_metrics = painter.fontMetrics()
        summary = summary_metrics.elidedText(
            self._summary or "Не настроено",
            Qt.TextElideMode.ElideRight,
            int(summary_rect.width()),
        )
        painter.drawText(
            summary_rect, int(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft), summary
        )

        painter.setOpacity(1.0)

        # Border / selection / running.
        if self._running:
            pen = QPen(self._color("focus"), 2.0)
        elif self.isSelected():
            pen = QPen(self._color("accent"), 2.0)
        elif self._danger:
            pen = QPen(self._color("error"), 1.0)
        else:
            pen = QPen(border, 1.0)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(body, _RADIUS, _RADIUS)

        self._paint_ports(painter, role)

    def _paint_ports(self, painter: QPainter, role: QColor) -> None:
        surface = self._color("surface")
        # Input port (left).
        self._draw_port(painter, QPointF(0, NODE_HEIGHT / 2), role, surface, filled=False)
        # Output ports (right).
        ports = self.output_ports()
        for index, port in enumerate(ports):
            fraction = (index + 1) / (len(ports) + 1)
            centre = QPointF(NODE_WIDTH, NODE_HEIGHT * fraction)
            colour = _branch_colour(self, port) or role
            self._draw_port(painter, centre, colour, surface, filled=False)
            label = BRANCH_LABELS.get(port, "")
            if label:
                painter.setPen(QPen(colour))
                lbl_font = QFont(painter.font())
                lbl_font.setPixelSize(9)
                lbl_font.setBold(True)
                painter.setFont(lbl_font)
                painter.drawText(
                    QRectF(NODE_WIDTH - 52, centre.y() - 14, 44, 12),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    label.upper(),
                )

        # Breakpoint dot on the left of the header.
        if self._breakpoint:
            painter.setBrush(QBrush(self._color("error")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(_ACCENT_BAR + 5, _HEADER_H / 2), 4.0, 4.0)

    def _draw_port(
        self,
        painter: QPainter,
        centre: QPointF,
        colour: QColor,
        surface: QColor,
        *,
        filled: bool,
    ) -> None:
        painter.setPen(QPen(colour, 2.0))
        painter.setBrush(QBrush(colour if filled else surface))
        painter.drawEllipse(centre, _PORT_R, _PORT_R)


def _branch_colour(item: NodeItem, port: str) -> QColor | None:
    """The success/error colour that labels ``then``/``else`` ports, else ``None``."""
    if port == "then":
        return QColor(item._theme.theme.color("success"))
    if port in ("else", "catch"):
        return QColor(item._theme.theme.color("error"))
    return None


def _distance(a: QPointF, b: QPointF) -> float:
    return float(((a.x() - b.x()) ** 2 + (a.y() - b.y()) ** 2) ** 0.5)


def _mix(a: QColor, b: QColor, ratio: float) -> QColor:
    """``ratio`` of ``a`` over ``b`` — the mockup's ``color-mix`` for tinted surfaces."""
    inverse = 1.0 - ratio
    return QColor(
        round(a.red() * ratio + b.red() * inverse),
        round(a.green() * ratio + b.green() * inverse),
        round(a.blue() * ratio + b.blue() * inverse),
    )
