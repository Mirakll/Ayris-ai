"""Accessible animated toggle drawn entirely from theme tokens."""

from __future__ import annotations

from PySide6.QtCore import Property, QEasingCurve, QPropertyAnimation, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["ToggleSwitch"]


class ToggleSwitch(QWidget):
    toggled = Signal(bool)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        checked: bool = False,
        label: str = "Переключатель",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._checked = checked
        self._position = 1.0 if checked else 0.0
        self.setAccessibleName(label)
        self.setAccessibleDescription("Включено" if checked else "Выключено")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._animation = QPropertyAnimation(self, b"position", self)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        theme.theme_changed.connect(self._theme_changed)
        self._theme_changed()

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, checked: bool) -> None:  # noqa: N802
        if checked == self._checked:
            return
        self._checked = checked
        self.setAccessibleDescription("Включено" if checked else "Выключено")
        self._animation.stop()
        self._animation.setStartValue(self._position)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()
        self.toggled.emit(checked)

    def get_position(self) -> float:
        return self._position

    def set_position(self, value: float) -> None:
        self._position = value
        self.update()

    position = Property(float, get_position, set_position, None, "Положение ползунка")

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.isEnabled():
            self.setChecked(not self._checked)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.setChecked(not self._checked)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colours = self._theme.theme.colors
        track = (
            (colours.accent if self._checked else colours.surface_highlight)
            if self.isEnabled()
            else colours.accent_disabled
        )
        border = colours.focus if self.hasFocus() else colours.border
        width = self.width()
        height = self.height()
        border_width = self._theme.metric("border_width")
        painter.setPen(QColor(border))
        painter.setBrush(QColor(track))
        painter.drawRoundedRect(
            QRectF(border_width / 2, border_width / 2, width - border_width, height - border_width),
            height / 2,
            height / 2,
        )
        margin = self._theme.metric("toggle_knob_margin")
        diameter = height - margin * 2
        x = margin + (width - height) * self._position
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(colours.on_accent if self._checked else colours.text_secondary))
        painter.drawEllipse(QRectF(x, margin, diameter, diameter))

    def _theme_changed(self, _theme: object | None = None) -> None:
        self._animation.setDuration(self._theme.metric("animation_normal"))
        self.setFixedSize(self._theme.metric("toggle_width"), self._theme.metric("toggle_height"))
        self.update()
