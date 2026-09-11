"""Headless tray status and menu/event behaviour."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication

from ayris.core.events import (
    Event,
    EventBus,
    MicModeRequested,
    MicToggleRequested,
    OverlayVisibilityRequested,
    ProfileSwitchRequested,
)
from ayris.core.models import Profile
from ayris.core.state import AssistantState, MicMode, StatusSnapshot
from ayris.gui.tray import TrayLevel, _icon, status_for
from ayris.gui.tray_menu import TrayMenu

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


def test_status_uses_text_as_well_as_colour() -> None:
    assert status_for(StatusSnapshot()).level is TrayLevel.HEALTHY
    paused = status_for(StatusSnapshot(mic_enabled=False))
    assert paused.level is TrayLevel.PAUSED
    assert "микрофон выключен" in paused.tooltip
    warning = status_for(StatusSnapshot(state=AssistantState.ERROR, detail="нет микрофона"))
    assert warning.level is TrayLevel.WARNING
    assert "нет микрофона" in warning.tooltip


def test_bundled_status_icons_exist_for_light_and_dark_panels() -> None:
    for level in TrayLevel:
        assert not _icon(level, light_panel=True).isNull()
        assert not _icon(level, light_panel=False).isNull()


def test_menu_reflects_state_and_publishes_commands(app: QApplication) -> None:
    bus = EventBus(thread_id=None)
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    profiles = [Profile("Основной", id=1, is_active=True), Profile("Игры", id=2)]
    menu = TrayMenu(
        bus,
        profiles=lambda: profiles,
        active_profile=lambda: profiles[0],
        show_settings=lambda: None,
        quit_application=lambda: None,
    )
    menu.sync(StatusSnapshot(mic_enabled=False, mic_mode=MicMode.PTT), overlay_visible=False)
    assert not menu.microphone.isChecked()
    assert menu.ptt.isChecked()
    assert not menu.overlay.isChecked()
    menu.microphone.trigger()
    menu.always.trigger()
    menu.overlay.trigger()
    assert any(isinstance(event, MicToggleRequested) for event in events)
    assert any(
        isinstance(event, MicModeRequested) and event.mode is MicMode.ALWAYS for event in events
    )
    assert any(isinstance(event, OverlayVisibilityRequested) and event.visible for event in events)
    menu.reload_profiles()
    assert [action.text() for action in menu.profile_menu.actions()] == ["Основной", "Игры"]
    assert menu.profile_menu.actions()[0].isChecked()
    menu.profile_menu.actions()[1].trigger()
    assert any(
        isinstance(event, ProfileSwitchRequested) and event.profile_id == 2 for event in events
    )
    menu.close()
