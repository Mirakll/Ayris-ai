"""The single dashboard window, offscreen.

One window replaced the settings window and the floating overlay, so these tests
assert the assembled behaviour: the sphere and status line follow the assistant
state, the dialogue log fills from bus events, the command field routes text,
the microphone button asks the state owner to toggle, the profile selector
switches profiles, and the settings layer opens and closes over the dashboard
rather than as a second window.

The WebGL sphere needs Qt WebEngine and must never be spun up in CI (it hangs),
so ``make_sphere`` is patched to the lightweight QPainter sphere for every test.
Windows are closed in ``finally`` so one test never leaks into the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QWidget

from ayris.core.config import ConfigManager
from ayris.core.events import (
    ActionFailed,
    EventBus,
    MicToggleRequested,
    ModeChanged,
    OnlineStatusChanged,
    OverlayVisibilityRequested,
    TranscriptReady,
    TtsStarted,
)
from ayris.core.models import Profile
from ayris.core.profile import ProfileSwitched
from ayris.core.state import AssistantState, MicMode
from ayris.gui.dashboard import showcase as showcase_module
from ayris.gui.main_window import MainWindow
from ayris.gui.overlay.dialog_log import DialogKind
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.sphere_widget import SphereWidget as PainterSphere

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    return ThemeManager(app)


@pytest.fixture(autouse=True)
def _painter_sphere(monkeypatch: pytest.MonkeyPatch, theme: ThemeManager) -> None:
    def factory(theme_: ThemeManager, parent: QWidget | None = None) -> QWidget:
        return PainterSphere(theme_, parent=parent)

    monkeypatch.setattr(showcase_module, "make_sphere", factory)


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    return ConfigManager(tmp_path / "config.toml")


def _window(
    theme: ThemeManager,
    manager: ConfigManager,
    *,
    bus: EventBus | None = None,
    **kwargs: object,
) -> MainWindow:
    return MainWindow(theme=theme, manager=manager, bus=bus, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# One window, two columns
# --------------------------------------------------------------------------- #


def test_state_drives_sphere_status_and_voice(theme: ThemeManager, manager: ConfigManager) -> None:
    window = _window(theme, manager)
    try:
        window.set_state(AssistantState.LISTENING)
        assert window._showcase.sphere.state.value == "listening"
        assert window._dialog.input_bar._voice_active is True
        window.set_state(AssistantState.IDLE)
        assert window._dialog.status_label.text() == "Привет, чем помочь?"
        assert window._dialog.input_bar._voice_active is False
    finally:
        window.exit()


def test_add_message_switches_from_status_to_log(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    window = _window(theme, manager)
    try:
        assert window._dialog._center.currentWidget() is window._dialog._empty_page
        window.add_message("assistant", "готово")
        assert window._dialog._center.currentWidget() is window._dialog.log
        kinds = [(e.kind, e.text) for e in window._dialog.log.entries]
        assert (DialogKind.ANSWER, "готово") in kinds
    finally:
        window.exit()


def test_settings_layer_opens_and_closes_in_window(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    window = _window(theme, manager)
    try:
        window.show()
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.settings_open is False
        window.open_settings()
        assert window.settings_open is True
        # No second top-level window appears — settings live inside this one.
        extra = [
            w
            for w in QApplication.topLevelWidgets()
            if w is not window and w.isWindow() and w.isVisible()
        ]
        assert extra == []
        window.close_settings()
        assert window.settings_open is False
        assert window.created_sections  # sections still registered
    finally:
        window.exit()


def test_fullscreen_toggles_chrome_and_persists(
    theme: ThemeManager, manager: ConfigManager
) -> None:
    window = _window(theme, manager)
    try:
        window.show()
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.fullscreen is False

        # The corner button drives the toggle.
        window._showcase.fullscreen_button.click()
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.fullscreen is True
        assert window.isFullScreen() is True
        assert window._showcase._flat is True
        assert window._dialog._flat is True
        assert manager.settings.window.fullscreen is True

        window.toggle_fullscreen()
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.fullscreen is False
        assert window._showcase._flat is False
        assert window._dialog._flat is False
        assert manager.settings.window.fullscreen is False
    finally:
        window.exit()


def test_saved_fullscreen_restores_on_show(theme: ThemeManager, manager: ConfigManager) -> None:
    manager.apply({"window.fullscreen": True})
    window = _window(theme, manager)
    try:
        assert window.fullscreen is False  # deferred until the window is shown
        window.show()
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.fullscreen is True
        assert window.isFullScreen() is True
    finally:
        window.exit()


# --------------------------------------------------------------------------- #
# Event bus in, dashboard out
# --------------------------------------------------------------------------- #


def test_bus_events_reflect_on_the_window(theme: ThemeManager, manager: ConfigManager) -> None:
    bus = EventBus(thread_id=None)
    window = _window(theme, manager, bus=bus)
    try:
        bus.publish(
            ModeChanged(
                state=AssistantState.LISTENING,
                previous=AssistantState.IDLE,
                mic_mode=MicMode.PTT,
            )
        )
        assert window._showcase.sphere.state.value == "listening"
        bus.publish(OnlineStatusChanged(online=False, detail="нет сети"))
        assert window._dialog.offline_hint.isHidden() is False
        bus.publish(TranscriptReady(text="привет", is_final=True))
        bus.publish(TtsStarted(text="здравствуйте"))
        bus.publish(ActionFailed(action="open", error="boom", user_message="не вышло"))
        kinds = [(e.kind, e.text) for e in window._dialog.log.entries]
        assert (DialogKind.HEARD, "привет") in kinds
        assert (DialogKind.ANSWER, "здравствуйте") in kinds
        assert (DialogKind.ERROR, "не вышло") in kinds
    finally:
        window.exit()


def test_typed_command_logs_and_routes(theme: ThemeManager, manager: ConfigManager) -> None:
    seen: list[str] = []
    window = _window(theme, manager, submit_text=seen.append)
    try:
        window._dialog.input_bar.command.setText("выключи музыку")
        window._dialog.input_bar.command.returnPressed.emit()
        assert seen == ["выключи музыку"]
        kinds = [(e.kind, e.text) for e in window._dialog.log.entries]
        assert (DialogKind.HEARD, "выключи музыку") in kinds
    finally:
        window.exit()


def test_microphone_button_asks_to_toggle(theme: ThemeManager, manager: ConfigManager) -> None:
    bus = EventBus(thread_id=None)
    requests: list[MicToggleRequested] = []
    bus.subscribe(MicToggleRequested, requests.append)
    window = _window(theme, manager, bus=bus)
    try:
        window._dialog.input_bar.mic_button.click()
        assert len(requests) == 1
    finally:
        window.exit()


def test_profile_selector_switches(theme: ThemeManager, manager: ConfigManager) -> None:
    chosen: list[Profile] = []
    profiles = [Profile("Базовый", id=1), Profile("Тихий", id=2)]
    window = _window(
        theme,
        manager,
        profiles=lambda: profiles,
        switch_profile=chosen.append,
    )
    try:
        window._dialog.profile_selected.emit(profiles[1])
        assert chosen == [profiles[1]]
    finally:
        window.exit()


def test_visibility_request_shows_and_hides(theme: ThemeManager, manager: ConfigManager) -> None:
    bus = EventBus(thread_id=None)
    window = _window(theme, manager, bus=bus)
    try:
        bus.publish(OverlayVisibilityRequested(visible=True))
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.isVisible() is True
        bus.publish(OverlayVisibilityRequested(visible=False))
        QApplication.instance().processEvents()  # type: ignore[union-attr]
        assert window.isVisible() is False
    finally:
        window.exit()


def test_exit_unsubscribes_from_the_bus(theme: ThemeManager, manager: ConfigManager) -> None:
    bus = EventBus(thread_id=None)
    window = _window(theme, manager, bus=bus)
    window.exit()
    # After exit, further events must not reach a torn-down window.
    bus.publish(ProfileSwitched(profile=Profile("Игры", id=3)))
    assert window._dialog.profile_button.text().startswith("По умолчанию")
