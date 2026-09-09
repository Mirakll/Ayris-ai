"""Token-coloured indeterminate loading indicator."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["BusyIndicator"]


class BusyIndicator(QWidget):
    def __init__(
        self, theme: ThemeManager, *, active: bool = True, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.setAccessibleName("Загрузка")
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self.setActive(active)

    def isActive(self) -> bool:  # noqa: N802
        return self._timer.isActive()

    def setActive(self, active: bool) -> None:  # noqa: N802
        if active:
            self._timer.start(max(1, self._theme.metric("animation_fast") // 4))
        else:
            self._timer.stop()
        self.update()

    def _advance(self) -> None:
        self._angle = (self._angle + 30) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width = self._theme.metric("focus_width")
        pen = QPen(QColor(self._theme.theme.color("accent")), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        inset = width + self._theme.metric("spacing_xs")
        rect = self.rect().adjusted(inset, inset, -inset, -inset)
        painter.drawArc(rect, (90 - self._angle) * 16, 250 * 16)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        size = self._theme.metric("busy_size")
        self.setFixedSize(size, size)
        if self._timer.isActive():
            self._timer.setInterval(max(1, self._theme.metric("animation_fast") // 4))
        self.update()
