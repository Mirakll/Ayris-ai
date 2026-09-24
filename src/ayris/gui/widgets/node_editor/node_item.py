"""Нода-карточка блока на сцене: заголовок с ролью, сводка, порты.

Компактная карточка по варианту «Кинематограф»: цветная шапка (насыщенный тон роли) и
рамка того же цвета вместо прежнего левого accent-bar, шапка с иконкой, русским названием
и ролью капсом, одна строка сводки с обрезкой, порты вход-слева / выход(ы)-справа с мягким
свечением. Состояния «выключен», «опасный», «выделен» и «выполняется» — визуально. Цвета и
подписи берутся из токенов темы и каталога задачи 33, не хардкодятся. Рисование — заливки,
текст и лёгкий радиальный ореол портов; отрисовка кэшируется, пересчёта раскладки в
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
    QRadialGradient,
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
from ayris.gui.widgets.node_editor.icons import paint_block_glyph
from ayris.gui.widgets.node_editor.layout import GRID, NODE_HEIGHT, NODE_WIDTH

if TYPE_CHECKING:
    from ayris.gui.theme import ThemeManager

__all__ = ["NodeItem"]

_RADIUS: Final = 12.0
_HEADER_H: Final = 40.0
_PORT_R: Final = 6.5
_PORT_HALO: Final = 6.0
_PAD: Final = 12.0
#: Отступ подписи ветки (то/иначе/тело/ошибка) от правого края карточки. Подпись рисуется
#: ВНУТРИ ноды у правого края с выравниванием вправо (``.port-label{right:16px}`` макета),
#: на уровне своего порта, — а не на внешней «полке», где она заезжала и «съезжала».
_PORT_LABEL_INSET: Final = 14.0
#: Правый жёлоб сводки, когда у ноды есть подписи веток: сводка эллипсируется раньше, чтобы
#: её текст не подлезал под капсовые подписи «ТО»/«ИНАЧЕ»/«ТЕЛО»/«ОШИБКА» у правого края.
_SUMMARY_GUTTER: Final = 40.0
#: Тёмный якорь для затемнения токена (color-mix c #000 из макета «Кинематограф»):
#: не палитра, а операция затенения, поэтому это чёрный, а не цветовой токен темы.
_SHADE: Final = QColor(0, 0, 0)


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
        # A margin fits the port circles AND their soft halo so neither is clipped; extra
        # room below covers the soft drop shadow (drawn under the card so it reads as a
        # solid, lifted panel — the mockup's `box-shadow: 0 6px 18px`). Branch labels are
        # drawn INSIDE the card now, so no extra right shelf is reserved.
        margin = _PORT_R + _PORT_HALO + 2
        return QRectF(-margin, -8, NODE_WIDTH + 2 * margin, NODE_HEIGHT + 30)

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

        # Soft drop shadow under the card — QPainter has no blur, so the mockup's
        # `box-shadow: 0 6px 18px rgba(0,0,0,.6)` is approximated by a stack of translucent
        # rounded rects that grow outward and sink down; their overlap darkens toward the
        # centre and fades at the edge. Drawn first, beneath the opaque body, it lifts the
        # node off the canvas so it no longer reads as a flat, semi-transparent tint.
        for step in range(6, 0, -1):
            grow = step * 1.7
            drop = 6.0 * step / 6.0
            shadow_path = QPainterPath()
            shadow_path.addRoundedRect(
                body.adjusted(-grow, drop - grow * 0.35, grow, drop + grow),
                _RADIUS + grow * 0.5,
                _RADIUS + grow * 0.5,
            )
            shade = QColor(0, 0, 0)
            shade.setAlpha(11)
            painter.fillPath(shadow_path, shade)

        # Card body — the surface tone shaded toward black (the mockup's
        # `--node-bg: color-mix(surface 84%, #000)`), so the card reads as a solid,
        # deeper panel over the glowing canvas instead of a near-transparent tint.
        card = QPainterPath()
        card.addRoundedRect(body, _RADIUS, _RADIUS)
        painter.fillPath(card, QBrush(_mix(surface, _SHADE, 0.84)))

        # Header band tinted by the role colour — saturated, «цветная шапка» of the
        # cinematic variant. Intersected with the card so the top corners follow the
        # rounded outline and the fill never bleeds past the edge.
        header_path = QPainterPath()
        header_path.addRect(QRectF(0, 0, NODE_WIDTH, _HEADER_H))
        painter.fillPath(
            header_path.intersected(card),
            QBrush(_mix(role, self._color("surface_highlight"), 0.42)),
        )

        # Hairline sheen along the very top edge lifts the coloured header (the mockup's
        # inset top highlight); clipped to the card so its corners stay inside.
        sheen = QPainterPath()
        sheen.addRect(QRectF(0, 0, NODE_WIDTH, 1.0))
        painter.fillPath(
            sheen.intersected(card), QBrush(_mix(self._color("text_primary"), role, 0.28))
        )

        # Icon square with the block glyph inside — «у нод нет иконок» / «почти все значки
        # одинаковые» fix. The square is a soft role tint (matching the mockup's `.ico`
        # background); the lucide-style glyph is specific to THIS block type (keyboard for a
        # key press, clock for a wait…), stroked over it in the full role colour, and falls
        # back to the role glyph for a type this build doesn't know.
        icon_rect = QRectF(_PAD, 9, 22, 22)
        icon_path = QPainterPath()
        icon_path.addRoundedRect(icon_rect, 6, 6)
        painter.fillPath(icon_path, QBrush(_mix(role, surface, 0.30)))
        paint_block_glyph(painter, icon_rect, self._node.block.type, self._node.role, role)

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

        # Summary line, ellipsised. When the node carries branch labels (то/иначе/тело/
        # ошибка) they are drawn inside the right edge at each port's height, so the summary
        # reserves a right gutter and elides before it reaches them — no more overlap.
        painter.setPen(QPen(self._color("text_secondary")))
        summary_font = QFont(painter.font())
        summary_font.setPixelSize(12)
        summary_font.setBold(False)
        painter.setFont(summary_font)
        has_labels = any(BRANCH_LABELS.get(port) for port in self._node.branch_ports)
        right_reserve = _SUMMARY_GUTTER if has_labels else _PAD
        summary_rect = QRectF(
            _PAD, _HEADER_H + 6, NODE_WIDTH - _PAD - right_reserve, NODE_HEIGHT - _HEADER_H - 10
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

        # Soft outer glow when selected or running — the mockup's `0 0 26px accent` /
        # `0 0 30px focus`. Stacked translucent outlines fake the blur (no primitive), so a
        # picked node lights up instead of only swapping its border colour.
        glow = None
        if self._running:
            glow = self._color("focus")
        elif self.isSelected():
            glow = self._color("accent")
        if glow is not None:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for step in range(4, 0, -1):
                ring = QColor(glow)
                ring.setAlpha(22)
                painter.setPen(QPen(ring, step * 2.2))
                painter.drawRoundedRect(
                    body.adjusted(-step, -step, step, step), _RADIUS + step, _RADIUS + step
                )

        # Border / selection / running. The resting border is tinted with the role colour
        # (the cinematic node's role frame) and follows the rounded corners cleanly — it
        # replaces the old left accent bar, so no straight strip pokes past the radius.
        if self._running:
            pen = QPen(self._color("focus"), 2.0)
        elif self.isSelected():
            pen = QPen(self._color("accent"), 2.0)
        elif self._danger:
            pen = QPen(self._color("error"), 1.0)
        else:
            pen = QPen(_mix(role, border, 0.42), 1.6)
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
                # INSIDE the card, right-aligned near the right edge at the port's height
                # (the mockup's `.port-label{right:16px}`) — never on an external shelf,
                # which drifted and clipped. The summary reserves a matching right gutter,
                # so label and summary coexist without overlap.
                painter.drawText(
                    QRectF(
                        NODE_WIDTH * 0.42,
                        centre.y() - 7,
                        NODE_WIDTH * 0.58 - _PORT_LABEL_INSET,
                        14,
                    ),
                    int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                    label.upper(),
                )

        # Breakpoint dot on the left of the header.
        if self._breakpoint:
            painter.setBrush(QBrush(self._color("error")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(11.0, _HEADER_H / 2), 4.0, 4.0)

    def _draw_port(
        self,
        painter: QPainter,
        centre: QPointF,
        colour: QColor,
        surface: QColor,
        *,
        filled: bool,
    ) -> None:
        # Soft halo so the ports read as lit connectors, not flat dots (the mockup's
        # per-port glow). Painted first, under the crisp circle drawn on top.
        halo_r = _PORT_R + _PORT_HALO
        halo = QRadialGradient(centre, halo_r)
        inner = QColor(colour)
        inner.setAlphaF(0.55 if filled else 0.38)
        outer = QColor(colour)
        outer.setAlphaF(0.0)
        halo.setColorAt(0.0, inner)
        halo.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(halo))
        painter.drawEllipse(centre, halo_r, halo_r)
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
