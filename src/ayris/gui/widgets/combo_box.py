"""A combo box that paints its own chevron, immune to the double-arrow bug.

Qt draws a combobox arrow through the active *style*. Styling ``::drop-down``
as a floating chip plus a ``::down-arrow`` image works at 100 % scale, but at a
fractional device pixel ratio (125 %, 150 %) the style keeps painting its own
native arrow in the middle of the field *and* our image in the chip — two
arrows. Rather than fight the style, this widget removes the arrow subcontrol
entirely and paints one chevron itself, so there is exactly one arrow at any DPI.

Every visual value (colours, chip size, radius, arrow size) is a Qt property fed
from the theme via QSS ``qproperty-*`` in the stylesheet, so the widget stays
theme-driven and needs no ThemeManager reference.
"""

from __future__ import annotations

from PySide6.QtCore import Property, QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QWidget,
)

__all__ = ["ThemedComboBox"]


class ThemedComboBox(QComboBox):
    """Combo box that owns its chevron instead of leaving it to the style."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Sensible fallbacks; QSS qproperty-* overrides these with theme tokens.
        self._chip_color = QColor("#29223C")
        self._chip_color_active = QColor("#7A5CFF")
        self._arrow_color = QColor("#C3BAD9")
        self._arrow_color_active = QColor("#FFFFFF")
        self._arrow_color_disabled = QColor("#6B6684")
        self._chip_size = 36
        self._chip_inset = 6
        self._chip_radius = 8
        self._arrow_size = 16

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
        self.update()

    chipSize = Property(int, _get_chip_size, _set_chip_size)  # type: ignore[call-arg]  # noqa: N815

    def _get_chip_inset(self) -> int:
        return self._chip_inset

    def _set_chip_inset(self, value: int) -> None:
        self._chip_inset = int(value)
        self.updateGeometry()  # the chip footprint feeds sizeHint, so re-ask for it
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
        """Horizontal room the chevron chip claims on the right, in layout pixels.

        Qt sizes the field for the text alone — the QSS drop-down is zero width, so
        the style leaves no room for the chip we paint ourselves. Without adding it
        back, the field is exactly the text's width and ``paintEvent`` then carves
        the chip out of that text, leaving a sliver like «Авто: о». Reserving the
        same footprint here keeps the visible label as wide as it looks.
        """
        return max(1, self._chip_size) + 2 * self._chip_inset

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = super().sizeHint()
        return QSize(hint.width() + self._chip_footprint(), hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        hint = super().minimumSizeHint()
        return QSize(hint.width() + self._chip_footprint(), hint.height())

    # --- painting ----------------------------------------------------------
    def paintEvent(self, _event: QPaintEvent) -> None:  # noqa: N802
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        # Drop the arrow subcontrol so neither the base style nor the stylesheet
        # paints an arrow (or its chip). We render both ourselves below.
        option.subControls &= ~QStyle.SubControl.SC_ComboBoxArrow  # type: ignore[attr-defined]
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)

        rect = self.rect()
        size = max(1, self._chip_size)
        inset = self._chip_inset
        chip_h = max(1, rect.height() - 2 * inset)
        left = rect.right() - inset - size
        top = rect.top() + (rect.height() - chip_h) / 2.0
        chip = QRectF(left, top, size, chip_h)

        # Paint the label into a rect that stops before the chip, otherwise the
        # style lays the text across the full width and the chip covers its tail.
        label_option = QStyleOptionComboBox(option)
        label_option.rect = option.rect.adjusted(  # type: ignore[attr-defined]
            0, 0, -(size + 2 * inset), 0
        )
        # Elide the label to the field that survives the chip cut, so a long item
        # ends in «…» instead of being sliced through a glyph. drawControl clips
        # but never elides; QComboBox does this itself in its own paintEvent.
        field = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            label_option,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        label_option.currentText = self.fontMetrics().elidedText(  # type: ignore[attr-defined]
            self.currentText(), Qt.TextElideMode.ElideRight, max(0, field.width())
        )
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, label_option)

        enabled = self.isEnabled()
        popup_open = self.view().isVisible() if self.view() is not None else False
        active = enabled and (self.underMouse() or popup_open)

        painter.setRenderHint(QStylePainter.RenderHint.Antialiasing, True)
        if enabled:
            chip_color = self._chip_color_active if active else self._chip_color
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(chip_color)
            painter.drawRoundedRect(chip, self._chip_radius, self._chip_radius)

        if not enabled:
            arrow_color = self._arrow_color_disabled
        elif active:
            arrow_color = self._arrow_color_active
        else:
            arrow_color = self._arrow_color

        a = max(6, self._arrow_size) / 2.0
        cx = chip.center().x()
        cy = chip.center().y()
        pen = QPen(arrow_color)
        pen.setWidthF(max(1.5, self._arrow_size / 8.0))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(
            [
                QPointF(cx - a, cy - a / 2.0),
                QPointF(cx, cy + a / 2.0),
                QPointF(cx + a, cy - a / 2.0),
            ]
        )
