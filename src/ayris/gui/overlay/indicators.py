"""Small status widgets for the overlay header: network and microphone.

Both reflect *confirmed* state pushed from the event bus. Neither issues a
command or guesses ahead of the assistant: pressing a control elsewhere sends a
request, and the indicator only changes when the state owner answers with an
event.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QWidget

from ayris.core.state import MicMode
from ayris.gui.theme import ThemeManager

__all__ = ["MicIndicator", "NetworkIndicator"]


class _Pill(QLabel):
    """A rounded, theme-coloured status chip with an accessible label."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._color_token = "text_muted"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setProperty("overlayPill", True)
        theme.theme_changed.connect(self._refresh)

    def _set(self, *, text: str, accessible: str, color_token: str) -> None:
        self._color_token = color_token
        self.setText(text)
        self.setAccessibleName(accessible)
        self.setToolTip(accessible)
        self._refresh()

    def _refresh(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color(self._color_token)
        surface = self._theme.theme.color("surface_highlight")
        radius = self._theme.metric("radius_sm")
        pad_x = self._theme.metric("spacing_sm")
        pad_y = self._theme.metric("spacing_xs")
        self.setStyleSheet(
            "QLabel {"
            f"color: {color};"
            f"background: {surface};"
            f"border-radius: {radius}px;"
            f"padding: {pad_y}px {pad_x}px;"
            "}"
        )


class NetworkIndicator(_Pill):
    """Online / offline, driven by :class:`~ayris.core.events.OnlineStatusChanged`."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(theme, parent)
        self._online = False
        self.set_online(online=False)

    @property
    def online(self) -> bool:
        return self._online

    def set_online(self, *, online: bool, detail: str = "") -> None:
        self._online = online
        if online:
            self._set(text="Сеть", accessible="Сеть доступна", color_token="success")
        else:
            label = detail or "Офлайн-режим"
            self._set(text="Офлайн", accessible=label, color_token="text_secondary")


class MicIndicator(_Pill):
    """Microphone state and arming mode, driven by :class:`MicToggled`/`ModeChanged`."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(theme, parent)
        self._enabled = True
        self._mode = MicMode.HYBRID
        self.set_state(enabled=True, mode=MicMode.HYBRID)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def mode(self) -> MicMode:
        return self._mode

    def set_state(self, *, enabled: bool, mode: MicMode) -> None:
        self._enabled = enabled
        self._mode = mode
        if not enabled:
            self._set(text="Микрофон выкл.", accessible="Микрофон выключен", color_token="warning")
            return
        self._set(
            text=f"Микрофон · {mode.label}",
            accessible=f"Микрофон включён, режим: {mode.label}",
            color_token="text_primary",
        )
