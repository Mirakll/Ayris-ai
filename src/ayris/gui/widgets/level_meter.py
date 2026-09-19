"""A horizontal input-level bar with a VAD threshold marker.

The meter draws two things over one track: how loud the microphone is right now
and where the voice-activity gate sits. When the level crosses the gate the fill
turns the accent colour, so a glance answers the question calibration is really
about — «слышит ли меня Ayris сейчас».

The widget is a pure display. It owns no timer and opens no stream: the audio
worker publishes :class:`~ayris.core.events.AudioLevelChanged`, and whoever holds
the meter feeds it through :meth:`set_level`. That keeps it testable without a
sound card and lets the «Голос» tab stop feeding it the moment it is hidden.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["LevelMeter"]


class LevelMeter(QWidget):
    """A level bar in ``0.0-1.0`` with a movable threshold line."""

    def __init__(
        self,
        theme: ThemeManager,
        *,
        level: float = 0.0,
        threshold: float = 0.5,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._level = _clamp(level)
        self._peak = self._level
        self._threshold = _clamp(threshold)
        self.setAccessibleName("Уровень сигнала микрофона")
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def level(self) -> float:
        """Current level in ``0.0-1.0``."""
        return self._level

    def threshold(self) -> float:
        """Where the VAD gate sits, in ``0.0-1.0``."""
        return self._threshold

    def set_level(self, level: float) -> None:
        """Set the live level; keeps a short-lived peak marker above it."""
        self._level = _clamp(level)
        self._peak = max(self._level, self._peak * 0.9)
        self.update()

    def set_threshold(self, threshold: float) -> None:
        """Move the gate marker."""
        self._threshold = _clamp(threshold)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colours = self._theme.theme
        radius = self._theme.metric("radius_sm")
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)

        painter.setPen(QColor(colours.color("border")))
        painter.setBrush(QColor(colours.color("surface_highlight")))
        painter.drawRoundedRect(rect, radius, radius)

        if self._level > 0.0:
            active = self._level >= self._threshold
            fill = QColor(colours.color("success" if active else "text_secondary"))
            fill_rect = QRectF(rect)
            fill_rect.setWidth(rect.width() * self._level)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill)
            painter.drawRoundedRect(fill_rect, radius, radius)

        if self._peak > 0.01:
            peak_x = rect.left() + rect.width() * self._peak
            pen = painter.pen()
            pen.setColor(QColor(colours.color("text_primary")))
            pen.setWidthF(max(1.0, self._theme.metric("border_width")))
            painter.setPen(pen)
            painter.drawLine(int(peak_x), int(rect.top()), int(peak_x), int(rect.bottom()))

        marker_x = rect.left() + rect.width() * self._threshold
        pen = painter.pen()
        pen.setColor(QColor(colours.color("accent")))
        pen.setWidthF(max(1.5, self._theme.metric("focus_width")))
        painter.setPen(pen)
        painter.drawLine(int(marker_x), int(rect.top()), int(marker_x), int(rect.bottom()))

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self.setMinimumHeight(self._theme.metric("control_height"))
        self.update()


def _clamp(value: float) -> float:
    """Keep a level or a threshold inside ``0.0-1.0``."""
    return max(0.0, min(1.0, float(value)))
