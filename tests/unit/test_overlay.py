"""Offscreen behaviour of the reusable dialogue-panel widgets.

These parts once lived in a floating overlay window; that window is gone and its
pieces now sit in the dashboard's right column (see ``test_main_window.py`` for
the assembled window). The tests here still cover each widget on its own:
the bounded dialogue log, the command field's history, the indicators and the
active-timers panel. Widgets are closed in ``finally`` so one test never leaks a
window into the next.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from ayris.core.models import utc_now
from ayris.core.state import MicMode
from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.overlay.dialog_log import DialogKind, DialogLog
from ayris.gui.overlay.indicators import MicIndicator, NetworkIndicator
from ayris.gui.overlay.timers_panel import ActiveTimer, TimersPanel, format_remaining
from ayris.gui.theme import ThemeManager

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


class _FakeTimers:
    def __init__(self, timers: list[ActiveTimer]) -> None:
        self._timers = timers
        self.cancelled: list[int] = []

    def active_timers(self) -> list[ActiveTimer]:
        return self._timers

    def cancel_timer(self, timer_id: int) -> None:
        self.cancelled.append(timer_id)


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


def test_command_input_submit_method_matches_enter(theme: ThemeManager) -> None:
    field = CommandInput(theme)
    try:
        seen: list[str] = []
        field.submitted.connect(seen.append)
        field.setText("открой браузер")
        field.submit()
        assert seen == ["открой браузер"]
        assert field.text() == ""
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
