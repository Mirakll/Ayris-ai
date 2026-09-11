"""Tray context menu: event commands in, authoritative state out."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

from ayris.core.events import (
    EventBus,
    MicModeRequested,
    MicToggleRequested,
    OverlayVisibilityRequested,
    ProfileSwitchRequested,
)
from ayris.core.models import Profile
from ayris.core.state import MicMode, StatusSnapshot

__all__ = ["TrayMenu"]


class TrayMenu(QMenu):
    """A menu that reflects supplied state and never owns application state."""

    def __init__(
        self,
        bus: EventBus,
        *,
        profiles: Callable[[], Sequence[Profile]],
        active_profile: Callable[[], Profile],
        show_settings: Callable[[], None],
        quit_application: Callable[[], None],
    ) -> None:
        super().__init__()
        self._bus = bus
        self._profiles = profiles
        self._active_profile = active_profile
        self._overlay_visible = True

        self.microphone = self.addAction("Микрофон включён")
        self.microphone.setCheckable(True)
        self.microphone.triggered.connect(lambda _checked: self._bus.publish(MicToggleRequested()))

        mode_menu = self.addMenu("Режим")
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.always = mode_menu.addAction("Always Listening")
        self.ptt = mode_menu.addAction("Push-to-Talk")
        for action, mode in ((self.always, MicMode.ALWAYS), (self.ptt, MicMode.PTT)):
            action.setCheckable(True)
            self.mode_group.addAction(action)
            action.triggered.connect(
                lambda _checked, selected=mode: self._bus.publish(MicModeRequested(selected))
            )

        self.overlay = self.addAction("Показывать основной оверлей")
        self.overlay.setCheckable(True)
        self.overlay.triggered.connect(self._request_overlay)
        self.settings = self.addAction("Настройки")
        self.settings.triggered.connect(show_settings)
        self.profile_menu = self.addMenu("Профиль")
        self.profile_menu.aboutToShow.connect(self.reload_profiles)
        self.aboutToShow.connect(self.reload_profiles)
        self.addSeparator()
        self.quit = self.addAction("Выход")
        self.quit.triggered.connect(quit_application)

    def sync(self, snapshot: StatusSnapshot, *, overlay_visible: bool) -> None:
        widgets = (self.microphone, self.always, self.ptt, self.overlay)
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.microphone.setChecked(snapshot.mic_enabled)
            self.microphone.setText(
                "Микрофон включён" if snapshot.mic_enabled else "Микрофон выключен"
            )
            self.always.setChecked(snapshot.mic_mode in (MicMode.ALWAYS, MicMode.HYBRID))
            self.ptt.setChecked(snapshot.mic_mode is MicMode.PTT)
            self._overlay_visible = overlay_visible
            self.overlay.setChecked(overlay_visible)
        finally:
            for widget, blocked in zip(widgets, previous, strict=True):
                widget.blockSignals(blocked)

    def _request_overlay(self, visible: bool) -> None:
        self._bus.publish(OverlayVisibilityRequested(visible=visible))

    def reload_profiles(self) -> None:
        self.profile_menu.clear()
        active = self._active_profile()
        group = QActionGroup(self.profile_menu)
        group.setExclusive(True)
        for profile in self._profiles():
            action = QAction(profile.name, self.profile_menu)
            action.setCheckable(True)
            action.setChecked(profile.id == active.id)
            if profile.id is not None:
                action.triggered.connect(
                    lambda _checked, profile_id=profile.id: self._bus.publish(
                        ProfileSwitchRequested(profile_id)
                    )
                )
            group.addAction(action)
            self.profile_menu.addAction(action)
