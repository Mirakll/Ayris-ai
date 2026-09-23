"""A spin box that paints its own stepper chip instead of the native arrows.

Qt draws a spin box's up/down buttons through the active *style*: two tiny stacked
arrows crammed into a 14×16 column on the right. They read as a cramped «double
switch», ignore the theme, and — like the combobox arrow — misbehave at a fractional
device pixel ratio. Rather than restyle them, this widget removes the button
subcontrols entirely and paints one rounded stepper chip itself: a compact ▴ over ▾,
the half under the cursor highlighted, each half stepping the value on click. Exactly
one, theme-driven control at any DPI.

This mirrors :class:`~ayris.gui.widgets.combo_box.ThemedComboBox` — every visual value
(chip colours, size, radius, arrow size) is a Qt property fed from the theme via QSS
``qproperty-*``, so the widget stays theme-driven and needs no ThemeManager reference.
It is a drop-in :class:`QSpinBox`: value, range, suffix/prefix and signals are
unchanged, so existing call sites and tests keep working.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainterPath, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QSpinBox,
    QStyle,
    QStyleOptionSpinBox,
    QStylePainter,
    QWidget,
)

__all__ = ["ThemedSpinBox"]


class ThemedSpinBox(QSpinBox):
    """Spin box that owns its stepper chip instead of leaving it to the style."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Sensible fallbacks; QSS qproperty-* overrides these with theme tokens.
        self._chip_color = QColor("#29223C")
        self._chip_color_active = QColor("#7A5CFF")
        self._arrow_color = QColor("#C3BAD9")
        self._arrow_color_active = QColor("#FFFFFF")
        self._arrow_color_disabled = QColor("#6B6684")
        self._chip_size = 22
        self._chip_inset = 6
        self._chip_radius = 8
        self._arrow_size = 12
        # Which half is under the cursor right now: "up", "down" or "" (none).
        self._hover = ""
        self.setMouseTracking(True)
        self._reserve_text_room()

    # --- Qt properties driven from QSS (qproperty-*) -----------------------
    def _get_chip_color(self) -> QColor:
        return self._chip_color

    def _set_chip_color(self, value: QColor) -> None:
        self._chip_color = value
        self.update()

    chipColor = Property(QColor, _get_chip_color, _set_chip_color)  # type: ignore[call-arg]  # noqa: N815

    def _get_chip_color_active(self) -> QColor:
        return self._chip_color_active

    def _set_chip_color_active(self, value: QColor) -> None:
        self._chip_color_active = value
        self.update()

    chipColorActive = Property(QColor, _get_chip_color_active, _set_chip_color_active)  # type: ignore[call-arg]  # noqa: N815

    def _get_arrow_color(self) -> QColor:
        return self._arrow_color

    def _set_arrow_color(self, value: QColor) -> None:
        self._arrow_color = value
        self.update()

    arrowColor = Property(QColor, _get_arrow_color, _set_arrow_color)  # type: ignore[call-arg]  # noqa: N815

    def _get_arrow_color_active(self) -> QColor:
        return self._arrow_color_active

    def _set_arrow_color_active(self, value: QColor) -> None:
        self._arrow_color_active = value
        self.update()

    arrowColorActive = Property(QColor, _get_arrow_color_active, _set_arrow_color_active)  # type: ignore[call-arg]  # noqa: N815

    def _get_arrow_color_disabled(self) -> QColor:
        return self._arrow_color_disabled

    def _set_arrow_color_disabled(self, value: QColor) -> None:
        self._arrow_color_disabled = value
        self.update()

    arrowColorDisabled = Property(  # type: ignore[call-arg]  # noqa: N815
        QColor, _get_arrow_color_disabled, _set_arrow_color_disabled
    )

    def _get_chip_size(self) -> int:
        return self._chip_size

    def _set_chip_size(self, value: int) -> None:
        self._chip_size = int(value)
        self.updateGeometry()  # the chip footprint feeds sizeHint, so re-ask for it
        self._reserve_text_room()
        self.update()

    chipSize = Property(int, _get_chip_size, _set_chip_size)  # type: ignore[call-arg]  # noqa: N815

    def _get_chip_inset(self) -> int:
        return self._chip_inset

    def _set_chip_inset(self, value: int) -> None:
        self._chip_inset = int(value)
        self.updateGeometry()  # the chip footprint feeds sizeHint, so re-ask for it
        self._reserve_text_room()
        self.update()

    chipInset = Property(int, _get_chip_inset, _set_chip_inset)  # type: ignore[call-arg]  # noqa: N815

    def _get_chip_radius(self) -> int:
        return self._chip_radius

    def _set_chip_radius(self, value: int) -> None:
        self._chip_radius = int(value)
        self.update()

    chipRadius = Property(int, _get_chip_radius, _set_chip_radius)  # type: ignore[call-arg]  # noqa: N815

    def _get_arrow_size(self) -> int:
        return self._arrow_size

    def _set_arrow_size(self, value: int) -> None:
        self._arrow_size = int(value)
        self.update()

    arrowSize = Property(int, _get_arrow_size, _set_arrow_size)  # type: ignore[call-arg]  # noqa: N815

    # --- geometry ----------------------------------------------------------
    def _chip_footprint(self) -> int:
        """Horizontal room the stepper chip claims on the right, in layout pixels.

        The base ``QSpinBox`` QSS gives the field a text-width edit area; we paint the
        chip on top of the right edge, so reserve its footprint here or the chip would
        sit over the last digits.
        """
        return max(1, self._chip_size) + 2 * self._chip_inset

    def _reserve_text_room(self) -> None:
        """Keep the editable text clear of the chip by insetting the line edit.

        Unlike the combobox — which paints its own label and elides it — a spin box
        edits through a real child ``QLineEdit`` that lays its text across the whole
        field. Without a right margin the number would slide under the chip. The
        left/top/bottom margins stay at zero; only the right side reserves the chip.
        """
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.setTextMargins(0, 0, self._chip_footprint(), 0)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(hint.width() + self._chip_footprint(), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        return QSize(hint.width() + self._chip_footprint(), hint.height())

    def _chip_rect(self) -> QRectF:
        """The stepper chip's rectangle, hugging the right edge and vertically centred."""
        rect = self.rect()
        size = max(1, self._chip_size)
        inset = self._chip_inset
        chip_h = max(1, rect.height() - 2 * inset)
        left = rect.right() - inset - size
        top = rect.top() + (rect.height() - chip_h) / 2.0
        return QRectF(left, top, size, chip_h)

    # --- interaction -------------------------------------------------------
    def _half_at(self, y: float, chip: QRectF) -> str:
        """Which stepper half a y-coordinate falls in: ``"up"`` / ``"down"``."""
        return "up" if y < chip.center().y() else "down"

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """A click on the chip steps the value; elsewhere is normal editing."""
        chip = self._chip_rect()
        pos = event.position()
        if self.isEnabled() and chip.adjusted(-2, -2, 2, 2).contains(pos):
            if self._half_at(pos.y(), chip) == "up":
                self.stepUp()
            else:
                self.stepDown()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Track which half the cursor is over so it can highlight."""
        chip = self._chip_rect()
        pos = event.position()
        hover = self._half_at(pos.y(), chip) if chip.contains(pos) else ""
        if hover != self._hover:
            self._hover = hover
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event: object) -> None:  # noqa: N802
        if self._hover:
            self._hover = ""
            self.update()
        super().leaveEvent(event)  # type: ignore[arg-type]

    # --- painting ----------------------------------------------------------
    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        painter = QStylePainter(self)
        option = QStyleOptionSpinBox()
        self.initStyleOption(option)
        # Drop the up/down subcontrols so neither the base style nor the stylesheet
        # paints the native arrows. We render the chip ourselves below.
        option.subControls &= ~QStyle.SubControl.SC_SpinBoxUp  # type: ignore[attr-defined]
        option.subControls &= ~QStyle.SubControl.SC_SpinBoxDown  # type: ignore[attr-defined]
        painter.drawComplexControl(QStyle.ComplexControl.CC_SpinBox, option)

        chip = self._chip_rect()
        enabled = self.isEnabled()
        painter.setRenderHint(QStylePainter.RenderHint.Antialiasing, True)

        # Chip background is always drawn so the stepper reads as one solid control
        # (like the combobox chip), not a pair of ghost arrows. The half under the
        # cursor is tinted with the accent so it is clear each half clicks on its own.
        if enabled:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._chip_color)
            painter.drawRoundedRect(chip, self._chip_radius, self._chip_radius)
            if self._hover:
                painter.setBrush(self._chip_color_active)
                painter.drawPath(self._half_path(chip, self._hover))

        up_reachable = enabled and self.value() < self.maximum()
        down_reachable = enabled and self.value() > self.minimum()
        self._draw_chevron(painter, chip, "up", active=self._hover == "up", reachable=up_reachable)
        self._draw_chevron(
            painter, chip, "down", active=self._hover == "down", reachable=down_reachable
        )

    def _half_path(self, chip: QRectF, half: str) -> QPainterPath:
        """A rounded path covering the top or bottom half of the chip.

        Only the outer corners of the chip are rounded; the edge meeting the other
        half stays square, so the two halves tile into one chip without a seam.
        """
        radius = float(self._chip_radius)
        mid = chip.center().y()
        path = QPainterPath()
        if half == "up":
            rect = QRectF(chip.left(), chip.top(), chip.width(), mid - chip.top())
            path.moveTo(rect.left(), rect.bottom())
            path.lineTo(rect.left(), rect.top() + radius)
            path.quadTo(rect.left(), rect.top(), rect.left() + radius, rect.top())
            path.lineTo(rect.right() - radius, rect.top())
            path.quadTo(rect.right(), rect.top(), rect.right(), rect.top() + radius)
            path.lineTo(rect.right(), rect.bottom())
        else:
            rect = QRectF(chip.left(), mid, chip.width(), chip.bottom() - mid)
            path.moveTo(rect.left(), rect.top())
            path.lineTo(rect.left(), rect.bottom() - radius)
            path.quadTo(rect.left(), rect.bottom(), rect.left() + radius, rect.bottom())
            path.lineTo(rect.right() - radius, rect.bottom())
            path.quadTo(rect.right(), rect.bottom(), rect.right(), rect.bottom() - radius)
            path.lineTo(rect.right(), rect.top())
        path.closeSubpath()
        return path

    def _draw_chevron(
        self, painter: QStylePainter, chip: QRectF, direction: str, *, active: bool, reachable: bool
    ) -> None:
        """Draw one compact chevron in the upper or lower half of the chip."""
        if not self.isEnabled() or not reachable:
            colour = self._arrow_color_disabled
        elif active:
            colour = self._arrow_color_active
        else:
            colour = self._arrow_color

        a = max(4, self._arrow_size) / 2.0
        cx = chip.center().x()
        quarter = chip.height() / 4.0
        cy = chip.top() + (quarter if direction == "up" else 3 * quarter)

        pen = QPen(colour)
        pen.setWidthF(max(1.4, self._arrow_size / 9.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if direction == "up":
            points = [
                QPointF(cx - a, cy + a / 2.0),
                QPointF(cx, cy - a / 2.0),
                QPointF(cx + a, cy + a / 2.0),
            ]
        else:
            points = [
                QPointF(cx - a, cy - a / 2.0),
                QPointF(cx, cy + a / 2.0),
                QPointF(cx + a, cy - a / 2.0),
            ]
        painter.drawPolyline(points)
