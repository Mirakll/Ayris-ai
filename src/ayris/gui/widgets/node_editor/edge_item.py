"""Связь между нодами: светящаяся кривая Безье с чипом задержки и указателем направления.

Провод — кубическая кривая от выходного порта к входному, нарисованная тремя проходами
(широкое гало + средний + яркое ядро), без blur-эффектов, как в эталонном мокапе. Цвет —
цвет ветки (``то``=success, ``иначе``/``ошибка``=error) или цвет роли исходной ноды.
Наведение и выделение подсвечивают провод. Путь пересчитывается только при перемещении
концов, не в ``paint``.

На середине каждого провода — жёлтый чип задержки: свёрнутая «Пауза» между двумя нодами
(см. :func:`~ayris.gui.widgets.node_editor.bridge.collapse_delays`). Чип есть на любом
проводе; ``0 мс`` рисуется приглушённо, ненулевая задержка — акцентом ``warning``. Клик по
чипу редактор ловит через :meth:`chip_contains`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QStyleOptionGraphicsItem, QWidget

from ayris.gui.widgets.node_editor.bridge import ROLE_TOKENS, DisplayEdge, format_delay

if TYPE_CHECKING:
    from ayris.gui.theme import ThemeManager
    from ayris.gui.widgets.node_editor.node_item import NodeItem

__all__ = ["EdgeItem", "bezier_path"]

_MIN_CTRL: Final = 60.0
_CHIP_H: Final = 18.0
_CHIP_PAD: Final = 9.0
_CHIP_FONT_PX: Final = 10


def bezier_path(start: QPointF, end: QPointF) -> QPainterPath:
    """The left-to-right cubic Bézier between two points, control offset by the span."""
    dx = max(_MIN_CTRL, abs(end.x() - start.x()) * 0.5)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(start.x() + dx, start.y()),
        QPointF(end.x() - dx, end.y()),
        end,
    )
    return path


class EdgeItem(QGraphicsPathItem):
    """A glowing connection from one node's output port to another node's input.

    Carries the wire's delay chip: ``delay_ms`` is the collapsed «Пауза» duration (0 when
    the wire has none) and ``edge`` is the :class:`DisplayEdge` the scene edits when the chip
    is clicked.
    """

    def __init__(
        self,
        source: NodeItem,
        source_port: str,
        target: NodeItem,
        theme: ThemeManager,
        *,
        edge: DisplayEdge | None = None,
        delay_ms: int = 0,
    ) -> None:
        super().__init__()
        self._source = source
        self._source_port = source_port
        self._target = target
        self._theme = theme
        self._edge = (
            edge if edge is not None else DisplayEdge(source.node_id, source_port, target.node_id)
        )
        self._delay_ms = delay_ms
        self._hovered = False
        self._running = False
        self._chip_rect = QRectF()
        self.setZValue(0.0)
        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.update_path()

    @property
    def source_id(self) -> str:
        return self._source.node_id

    @property
    def source_port(self) -> str:
        return self._source_port

    @property
    def target_id(self) -> str:
        return self._target.node_id

    @property
    def edge(self) -> DisplayEdge:
        return self._edge

    @property
    def delay_ms(self) -> int:
        return self._delay_ms

    def set_running(self, running: bool) -> None:
        if running != self._running:
            self._running = running
            self.update()

    def update_path(self) -> None:
        """Recompute the curve from the current port positions, then re-place the chip."""
        start = self._source.output_scene_pos(self._source_port)
        end = self._target.input_scene_pos()
        path = bezier_path(start, end)
        self.setPath(path)
        self._chip_rect = self._compute_chip_rect(path)

    def _compute_chip_rect(self, path: QPainterPath) -> QRectF:
        centre = path.pointAtPercent(0.5)
        font = QFont()
        font.setPixelSize(_CHIP_FONT_PX)
        font.setBold(True)
        width = QFontMetricsF(font).horizontalAdvance(format_delay(self._delay_ms)) + 2 * _CHIP_PAD
        return QRectF(centre.x() - width / 2, centre.y() - _CHIP_H / 2, width, _CHIP_H)

    def chip_contains(self, scene_point: QPointF) -> bool:
        """Whether a scene point is inside the delay chip — the editor's click target."""
        return self._chip_rect.contains(scene_point)

    def _colour(self) -> QColor:
        if self._source_port == "then":
            return QColor(self._theme.theme.color("success"))
        if self._source_port in ("else", "catch"):
            return QColor(self._theme.theme.color("error"))
        return QColor(self._theme.theme.color(ROLE_TOKENS[self._source.node.role]))

    def hoverEnterEvent(self, event: object) -> None:  # noqa: N802, ARG002 — Qt override.
        self._hovered = True
        self.update()

    def hoverLeaveEvent(self, event: object) -> None:  # noqa: N802, ARG002 — Qt override.
        self._hovered = False
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 — Qt override.
        return self.path().boundingRect().united(self._chip_rect).adjusted(-6, -6, 6, 6)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,  # noqa: ARG002
        widget: QWidget | None = None,  # noqa: ARG002
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        path = self.path()
        colour = self._colour()
        bright = self.isSelected() or self._hovered or self._running
        # Three-pass glow: wide halo, mid, bright core — no blur.
        for width, alpha in ((8.0, 0.14), (4.0, 0.38), (2.0, 0.92)):
            glow = QColor(colour)
            glow.setAlphaF(min(1.0, alpha * (1.4 if bright else 1.0)))
            pen = QPen(glow, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPath(path)
        self._paint_arrow(painter, path, colour)
        self._paint_chip(painter)

    def _paint_chip(self, painter: QPainter) -> None:
        active = self._delay_ms > 0
        warning = QColor(self._theme.theme.color("warning"))
        surface = QColor(self._theme.theme.color("surface"))
        muted = QColor(self._theme.theme.color("text_muted"))
        if self._running:
            border = QColor(self._theme.theme.color("focus"))
        elif active:
            border = warning
        else:
            border = QColor(self._theme.theme.color("border"))
        fill = QColor(surface)
        fill.setAlphaF(0.95)
        painter.setPen(QPen(border, 1.4 if active else 1.0))
        painter.setBrush(fill)
        painter.drawRoundedRect(self._chip_rect, _CHIP_H / 2, _CHIP_H / 2)
        font = QFont(painter.font())
        font.setPixelSize(_CHIP_FONT_PX)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(warning if active else muted))
        painter.drawText(
            self._chip_rect, int(Qt.AlignmentFlag.AlignCenter), format_delay(self._delay_ms)
        )

    def _paint_arrow(self, painter: QPainter, path: QPainterPath, colour: QColor) -> None:
        end = path.pointAtPercent(1.0)
        before = path.pointAtPercent(0.94)
        direction = end - before
        length = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
        if length < 1e-6:
            return
        ux, uy = direction.x() / length, direction.y() / length
        size = 7.0
        base = QPointF(end.x() - ux * size, end.y() - uy * size)
        left = QPointF(base.x() - uy * size * 0.5, base.y() + ux * size * 0.5)
        right = QPointF(base.x() + uy * size * 0.5, base.y() - ux * size * 0.5)
        arrow = QPainterPath(end)
        arrow.lineTo(left)
        arrow.lineTo(right)
        arrow.closeSubpath()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawPath(arrow)
