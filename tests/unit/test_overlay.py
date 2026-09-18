"""Offscreen behaviour of the single main overlay and its parts.

These tests assert state, signals, geometry hand-off and that a hidden panel
stops its animation and its update timer. They do not assert transparency, focus
or flicker — those need a live desktop and are checked by hand. Widgets are
closed in ``finally`` so one test never leaks a window into the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from ayris.core.config import OverlayConfig
from ayris.core.events import (
    ActionFailed,
    AudioLevelChanged,
    EventBus,
    MicToggled,
    ModeChanged,
    OnlineStatusChanged,
    OverlayToggleRequested,
    OverlayVisibilityRequested,
    TranscriptReady,
    TtsStarted,
)
from ayris.core.models import Profile, utc_now
from ayris.core.profile import ProfileSwitched
from ayris.core.state import AssistantState, MicMode, StatusSnapshot
from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.overlay.dialog_log import DialogKind, DialogLog
from ayris.gui.overlay.indicators import MicIndicator, NetworkIndicator
from ayris.gui.overlay.main import MainOverlay, OverlayController
from ayris.gui.overlay.timers_panel import ActiveTimer, TimersPanel, format_remaining
from ayris.gui.overlay.window_flags import (
    WindowFlags,
    overlay_qt_flags,
    tool_window_bits,
    with_no_activate,
)
from ayris.gui.theme import ThemeManager
from ayris.utils import winapi
from ayris.utils.monitors import MonitorInfo

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


class _FakeNative:
    def __init__(self) -> None:
        self.style = 0
        self.topmost = 0

    def ex_style(self) -> int:
        return self.style

    def set_ex_style(self, value: int) -> None:
        self.style = value

    def assert_topmost(self) -> None:
        self.topmost += 1


class _FakeTimers:
    def __init__(self, timers: list[ActiveTimer]) -> None:
        self._timers = timers
        self.cancelled: list[int] = []

    def active_timers(self) -> list[ActiveTimer]:
        return self._timers

    def cancel_timer(self, timer_id: int) -> None:
        self.cancelled.append(timer_id)


def _primary() -> MonitorInfo:
    rect = winapi.Rect(0, 0, 1920, 1080)
    return MonitorInfo(handle=1, index=0, rect=rect, work=rect, name="DISPLAY0", primary=True)


# --------------------------------------------------------------------------- #
# Pure window-flag arithmetic
# --------------------------------------------------------------------------- #


def test_overlay_qt_flags_are_frameless_tool_topmost() -> None:
    flags = overlay_qt_flags()
    assert flags & int(Qt.WindowType.FramelessWindowHint)
    assert flags & int(Qt.WindowType.Tool)
    assert flags & int(Qt.WindowType.WindowStaysOnTopHint)


def test_tool_window_bits_leave_taskbar_and_alt_tab() -> None:
    bits = tool_window_bits(winapi.WS_EX_APPWINDOW)
    assert bits & winapi.WS_EX_TOOLWINDOW
    assert not bits & winapi.WS_EX_APPWINDOW


def test_no_activate_bit_only_when_asked() -> None:
    assert with_no_activate(0, enabled=True) == winapi.WS_EX_NOACTIVATE
    assert with_no_activate(winapi.WS_EX_NOACTIVATE, enabled=False) == 0


def test_window_flags_apply_over_a_fake_native() -> None:
    native = _FakeNative()
    flags = WindowFlags(native)
    flags.apply_tool_window()
    assert native.style & winapi.WS_EX_TOOLWINDOW
    flags.confirm_topmost()
    assert native.topmost == 1


def test_window_flags_without_native_are_safe_noops() -> None:
    flags = WindowFlags(None)
    assert flags.available is False
    flags.apply_tool_window()
    flags.confirm_topmost()  # must not raise


# --------------------------------------------------------------------------- #
# Dialogue log
# --------------------------------------------------------------------------- #


def test_dialog_log_is_bounded_to_the_last_lines(theme: ThemeManager) -> None:
    log = DialogLog(theme, max_lines=3)
    try:
        for index in range(5):
            log.add_entry(DialogKind.HEARD, f"строка {index}")
        assert len(log.entries) == 3
        assert [entry.text for entry in log.entries] == ["строка 2", "строка 3", "строка 4"]
    finally:
        log.close()


def test_dialog_log_ignores_blank_lines(theme: ThemeManager) -> None:
    log = DialogLog(theme)
    try:
        log.add_entry(DialogKind.ANSWER, "   ")
        assert log.entries == ()
    finally:
        log.close()


def test_dialog_log_copy_selected(theme: ThemeManager, app: QApplication) -> None:
    log = DialogLog(theme)
    try:
        log.add_entry(DialogKind.ERROR, "сбой")
        log._list.setCurrentRow(0)
        assert log.copy_selected() is True
        assert "сбой" in QApplication.clipboard().text()
    finally:
        log.close()


# --------------------------------------------------------------------------- #
# Command input
# --------------------------------------------------------------------------- #


def test_command_input_submits_and_clears(theme: ThemeManager) -> None:
    field = CommandInput(theme)
    try:
        seen: list[str] = []
        field.submitted.connect(seen.append)
        field.setText("который час")
        field.returnPressed.emit()
        assert seen == ["который час"]
        assert field.text() == ""
        assert field.history == ("который час",)
    finally:
        field.close()


def test_command_input_history_walks_with_arrows(theme: ThemeManager) -> None:
    field = CommandInput(theme)
    try:
        for phrase in ("первая", "вторая"):
            field.setText(phrase)
            field.returnPressed.emit()
        up = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.NoModifier)
        field.keyPressEvent(up)
        assert field.text() == "вторая"
        field.keyPressEvent(up)
        assert field.text() == "первая"
        down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Down, Qt.KeyboardModifier.NoModifier)
        field.keyPressEvent(down)
        assert field.text() == "вторая"
    finally:
        field.close()


# --------------------------------------------------------------------------- #
# Indicators
# --------------------------------------------------------------------------- #


def test_indicators_reflect_confirmed_state(theme: ThemeManager) -> None:
    network = NetworkIndicator(theme)
    mic = MicIndicator(theme)
    try:
        network.set_online(online=True)
        assert network.online is True
        mic.set_state(enabled=False, mode=MicMode.PTT)
        assert mic.enabled is False and mic.mode is MicMode.PTT
    finally:
        network.close()
        mic.close()


# --------------------------------------------------------------------------- #
# Timers panel
# --------------------------------------------------------------------------- #


def test_format_remaining_shows_hours_only_when_needed() -> None:
    assert format_remaining(90) == "01:30"
    assert format_remaining(3661) == "1:01:01"


def test_timers_panel_renders_countdown_and_cancels(theme: ThemeManager, app: QApplication) -> None:
    now = utc_now()
    provider = _FakeTimers([ActiveTimer(id=7, label="Чай", due=now + timedelta(seconds=90))])
    panel = TimersPanel(theme, provider=provider, clock=lambda: now)
    try:
        panel.refresh()
        assert panel.timers[0].label == "Чай"
        assert panel._rows[7].remaining.text() == "01:30"
        panel._cancel(7)
        assert provider.cancelled == [7]
    finally:
        panel.close()


def test_timers_panel_updates_only_while_visible(theme: ThemeManager, app: QApplication) -> None:
    panel = TimersPanel(theme, provider=_FakeTimers([]))
    try:
        panel.show()
        app.processEvents()
        assert panel.is_updating() is True
        panel.hide()
        app.processEvents()
        assert panel.is_updating() is False
    finally:
        panel.close()


# --------------------------------------------------------------------------- #
# The overlay window
# --------------------------------------------------------------------------- #


def test_overlay_state_drives_the_sphere(theme: ThemeManager) -> None:
    overlay = MainOverlay(theme)
    try:
        overlay.set_assistant_state(AssistantState.LISTENING)
        assert overlay.sphere.state.value == "listening"
    finally:
        overlay.close()


def test_hidden_overlay_stops_animation_and_updates(theme: ThemeManager, app: QApplication) -> None:
    overlay = MainOverlay(theme)
    try:
        overlay.show()
        app.processEvents()
        assert overlay.sphere.is_animation_running() is True
        overlay.hide()
        app.processEvents()
        assert overlay.sphere.is_animation_running() is False
        assert overlay.timers.is_updating() is False
    finally:
        overlay.close()


# --------------------------------------------------------------------------- #
# The controller: event bus in, overlay state out
# --------------------------------------------------------------------------- #


def _controller(
    app: QApplication,
    theme: ThemeManager,
    *,
    submit_text: object = None,
) -> tuple[EventBus, OverlayController]:
    bus = EventBus(thread_id=None)
    controller = OverlayController(
        bus,
        theme,
        lambda: OverlayConfig(enabled=False),
        monitors=lambda: [_primary()],
        native_factory=lambda _handle: None,
        submit_text=submit_text,  # type: ignore[arg-type]
    )
    return bus, controller


def test_controller_mirrors_bus_state(theme: ThemeManager, app: QApplication) -> None:
    bus, controller = _controller(app, theme)
    try:
        bus.publish(
            ModeChanged(
                state=AssistantState.LISTENING,
                previous=AssistantState.IDLE,
                mic_mode=MicMode.PTT,
            )
        )
        assert controller.overlay.sphere.state.value == "listening"
        bus.publish(OnlineStatusChanged(online=True))
        assert controller.overlay.network.online is True
        bus.publish(MicToggled(enabled=False, mic_mode=MicMode.PTT))
        assert controller.overlay.mic.enabled is False
        bus.publish(AudioLevelChanged(rms=0.5))
        bus.publish(ProfileSwitched(profile=Profile("Игры", id=2)))
        assert controller.overlay.profile.text() == "Игры"
    finally:
        controller.close()


def test_controller_logs_dialogue_from_events(theme: ThemeManager, app: QApplication) -> None:
    bus, controller = _controller(app, theme)
    try:
        bus.publish(TranscriptReady(text="привет", is_final=True))
        bus.publish(TtsStarted(text="здравствуйте"))
        bus.publish(ActionFailed(action="open", error="boom", user_message="не вышло"))
        kinds = [(entry.kind, entry.text) for entry in controller.overlay.log.entries]
        assert (DialogKind.HEARD, "привет") in kinds
        assert (DialogKind.ANSWER, "здравствуйте") in kinds
        assert (DialogKind.ERROR, "не вышло") in kinds
    finally:
        controller.close()


def test_controller_routes_typed_command(theme: ThemeManager, app: QApplication) -> None:
    seen: list[str] = []
    bus, controller = _controller(app, theme, submit_text=seen.append)
    try:
        controller.overlay.command.setText("выключи музыку")
        controller.overlay.command.returnPressed.emit()
        assert seen == ["выключи музыку"]
    finally:
        controller.close()


def test_controller_show_hide_and_toggle(theme: ThemeManager, app: QApplication) -> None:
    bus, controller = _controller(app, theme)
    try:
        bus.publish(OverlayVisibilityRequested(visible=True))
        app.processEvents()
        assert controller.overlay.isVisible() is True
        bus.publish(OverlayVisibilityRequested(visible=False))
        app.processEvents()
        assert controller.overlay.isVisible() is False
        bus.publish(OverlayToggleRequested())
        app.processEvents()
        assert controller.overlay.isVisible() is True
    finally:
        controller.close()


def test_controller_syncs_from_snapshot(theme: ThemeManager, app: QApplication) -> None:
    bus = EventBus(thread_id=None)
    snapshot = StatusSnapshot(
        state=AssistantState.THINKING, mic_mode=MicMode.ALWAYS, mic_enabled=True, online=True
    )
    controller = OverlayController(
        bus,
        theme,
        lambda: OverlayConfig(enabled=False),
        monitors=lambda: [_primary()],
        native_factory=lambda _handle: None,
        snapshot=snapshot,
    )
    try:
        assert controller.overlay.sphere.state.value == "thinking"
        assert controller.overlay.network.online is True
    finally:
        controller.close()
