"""Tab «Общие» (task 48): live theme, live priority, restart plaques, eco note.

The page carries the section's un-boring corners — a theme that applies the
instant it changes, an audio-priority combo that asks before «Реальное время»,
plaques that appear when a saved change needs a worker restarted, and a resource
panel that must stop sampling the moment it is hidden. Each is checked here
against a fake sampler and a fake supervisor, offscreen, with every widget
closed in teardown: an un-closed settings page keeps a config subscription and
hangs CI.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.core.config import ConfigManager, RestartScope
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.general import GeneralTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    ResourceMonitor,
    Sampler,
    WorkerControl,
    set_active_worker_control,
)
from ayris.workers.manager import WorkerStatus, WorkerSummary

pytestmark = pytest.mark.unit


class _FakeSampler(Sampler):
    """Fixed numbers, so a hidden panel and a shown one differ only in polling."""

    def __init__(self, rss: int = 100 * 1024 * 1024, cpu: float = 5.0) -> None:
        self._rss = rss
        self._cpu = cpu

    def sample(self, pid: int) -> tuple[int, float] | None:
        del pid
        return self._rss, self._cpu


class _FakeControl(WorkerControl):
    """A supervisor stand-in that records restarts instead of doing them."""

    def __init__(self, summaries: tuple[WorkerSummary, ...] = ()) -> None:
        self._summaries = summaries
        self.restarted: list[tuple[RestartScope, str]] = []

    def status(self) -> tuple[WorkerSummary, ...]:
        return self._summaries

    def restart_scope(self, scope: RestartScope, settings_reason: str = "") -> int:
        self.restarted.append((scope, settings_reason))
        return 1


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


@pytest.fixture(autouse=True)
def _clear_control() -> Iterator[None]:
    """Never leak a fake supervisor into another test's global."""
    yield
    set_active_worker_control(None)


def _make_tab(manager: ConfigManager, theme: ThemeManager) -> GeneralTab:
    tab = GeneralTab(manager, theme, sampler=_FakeSampler())
    tab.load_from_config()
    return tab


def test_tab_registers_itself_as_the_general_factory() -> None:
    assert tab_spec("general").factory is GeneralTab


def test_tab_assembles_and_loads_defaults(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    assert tab._theme_combo.currentData() == "dark_purple"
    assert tab._process_combo.currentData() == "normal"
    assert tab._audio_combo.currentData() == "high"
    assert tab._ram_combo.currentData() == 4096
    tab.dispose()
    tab.close()


def test_theme_applies_the_instant_the_combo_changes(
    app: QApplication, manager: ConfigManager
) -> None:
    theme = ThemeManager(app)
    tab = _make_tab(manager, theme)
    assert theme.mode == "dark"

    tab._theme_combo.setCurrentIndex(tab._theme_combo.findData("light"))
    assert theme.mode == "light"

    tab.flush_pending()
    assert manager.settings.general.theme == "light"
    tab.dispose()
    tab.close()


def test_priority_and_thread_pool_choices_reach_config(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))

    tab._process_combo.setCurrentIndex(tab._process_combo.findData("high"))
    tab._ram_combo.setCurrentIndex(tab._ram_combo.findData(8192))
    tab._bindings["performance.stt_threads"].widget.setValue(4)

    tab.flush_pending()
    assert manager.settings.performance.process_priority == "high"
    assert manager.settings.performance.ram_limit_mb == 8192
    assert manager.settings.performance.stt_threads == 4
    assert tab._resources._ram_limit_mb == 8192
    tab.dispose()
    tab.close()


def test_realtime_priority_reverts_when_declined(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._confirm_realtime = lambda: False  # type: ignore[method-assign]

    tab._audio_combo.setCurrentIndex(tab._audio_combo.findData("realtime"))

    assert tab._audio_combo.currentData() == "high"
    assert "performance.audio_priority" not in tab._pending
    assert not tab.dirty_label.isVisibleTo(tab)
    tab.dispose()
    tab.close()


def test_realtime_priority_saved_when_confirmed(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._confirm_realtime = lambda: True  # type: ignore[method-assign]

    tab._audio_combo.setCurrentIndex(tab._audio_combo.findData("realtime"))
    tab.flush_pending()

    assert manager.settings.performance.audio_priority == "realtime"
    tab.dispose()
    tab.close()


def test_restart_plaque_appears_then_clears_on_restart(
    app: QApplication, manager: ConfigManager
) -> None:
    control = _FakeControl()
    set_active_worker_control(control)
    tab = _make_tab(manager, ThemeManager(app))
    bar = tab._restart_bars[RestartScope.AUDIO]
    assert bar.isHidden()

    manager.apply({"performance.audio_priority": "above_normal"})
    app.processEvents()
    assert not bar.isHidden()
    assert bar.button.isEnabled()

    bar.button.click()
    assert control.restarted == [(RestartScope.AUDIO, "перезапуск из настроек")]
    assert RestartScope.AUDIO not in manager.pending_restarts
    assert bar.isHidden()
    tab.dispose()
    tab.close()


def test_restart_button_disabled_without_a_supervisor(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    manager.apply({"performance.audio_priority": "above_normal"})
    app.processEvents()

    bar = tab._restart_bars[RestartScope.AUDIO]
    assert not bar.isHidden()
    assert not bar.button.isEnabled()
    tab.dispose()
    tab.close()


def test_worker_rows_reflect_supervisor_status(app: QApplication, manager: ConfigManager) -> None:
    control = _FakeControl(
        (
            WorkerSummary(name="stt-1", kind="stt", status=WorkerStatus.READY, pid=4242),
            WorkerSummary(name="tts-1", kind="tts", status=WorkerStatus.STOPPED),
        )
    )
    set_active_worker_control(control)
    tab = _make_tab(manager, ThemeManager(app))

    rows = tab._worker_rows()
    assert [row.pid for row in rows] == [4242, None]
    assert "Распознавание речи" in rows[0].label
    assert rows[1].note  # a stopped worker shows its status instead of numbers
    tab.dispose()
    tab.close()


def test_eco_note_explains_deferred_engines(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    assert "Выключено" in tab._eco_note.text()

    manager.apply({"performance.eco_mode": True})
    app.processEvents()
    assert "Выключено" not in tab._eco_note.text()
    tab.dispose()
    tab.close()


def test_resource_panel_polls_only_while_visible(app: QApplication) -> None:
    monitor = ResourceMonitor(ThemeManager(app), sampler=_FakeSampler())
    assert not monitor.is_polling()

    monitor.show()
    app.processEvents()
    assert monitor.is_polling()

    monitor.hide()
    app.processEvents()
    assert not monitor.is_polling()
    monitor.close()


def test_resource_panel_flags_when_over_the_ram_limit(app: QApplication) -> None:
    monitor = ResourceMonitor(
        ThemeManager(app),
        sampler=_FakeSampler(rss=8 * 1024 * 1024 * 1024),
        ram_limit_mb=2048,
    )
    monitor.show()
    app.processEvents()

    assert monitor._summary.property("status") == "warning"
    assert "превышен лимит" in monitor._summary.text()
    monitor.close()
