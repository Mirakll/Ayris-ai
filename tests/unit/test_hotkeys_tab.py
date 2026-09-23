"""Tab «Горячие клавиши» (task 55): the model and the writes, offscreen.

No window is shown and no capture dialog is ever opened — a left-open modal hangs
CI. The tab's public seams (:meth:`HotkeysTab.assign_system`, ``assign_command``,
``clear_*``, ``reset_*`` and ``set_interception``) are driven directly and checked
against the config, the command store and the resolved rows. ``probe_interception``
is monkeypatched so the games toggle can be tested without the kernel driver. Every
widget is closed in teardown so a lingering config subscription cannot wedge the run.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.core.config import ConfigManager, Settings
from ayris.core.database import Database
from ayris.core.events import EventBus, OpenCommandRequested
from ayris.core.models import Command
from ayris.core.repositories import Repositories
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.hotkeys import HotkeysTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.hotkey_table import (
    SYSTEM_HOTKEY_ORDER,
    CommandHotkeyStatus,
    SystemHotkeyStatus,
)
from ayris.utils.hotkey_backends.interception import InterceptionState, InterceptionStatus
from ayris.utils.hotkeys import Hotkey, try_parse_hotkey

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
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


@pytest.fixture
def repos() -> Iterator[Repositories]:
    database = Database.open(":memory:")
    yield Repositories(database)
    database.close()


@pytest.fixture
def store(repos: Repositories) -> CommandTreeStore:
    profile = repos.profiles.create("Основной", activate=True)
    assert profile.id is not None
    return CommandTreeStore(repos, profile.id)


def _hotkey(text: str) -> Hotkey:
    parsed = try_parse_hotkey(text)
    assert parsed is not None, text
    return parsed


def _make_tab(
    manager: ConfigManager,
    theme: ThemeManager,
    store: CommandTreeStore,
    *,
    bus: EventBus | None = None,
) -> HotkeysTab:
    tab = HotkeysTab(manager, theme, bus, store=store)
    tab.load_from_config()
    return tab


def _add_command(
    repos: Repositories, store: CommandTreeStore, name: str, *, enabled: bool = True
) -> int:
    command = repos.commands.create(
        Command(name=name, profile_id=store.profile_id, enabled=enabled)
    )
    assert command.id is not None
    return command.id


def test_registers_itself_as_the_hotkeys_factory() -> None:
    assert tab_spec("hotkeys").factory is HotkeysTab


def test_system_order_matches_the_manager_labels() -> None:
    # The table must list exactly the hotkeys the manager registers, in order.
    from ayris.utils.hotkey_manager import _SYSTEM_LABELS

    assert tuple(_SYSTEM_LABELS) == SYSTEM_HOTKEY_ORDER


def test_assign_system_writes_config_and_marks_active(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.assign_system("push_to_talk", _hotkey("ctrl+alt+p")) is True
    assert manager.settings.hotkeys.push_to_talk == "ctrl+alt+p"
    row = next(r for r in tab.system_rows() if r.spec.field == "push_to_talk")
    assert row.status is SystemHotkeyStatus.ACTIVE
    assert not row.has_problem
    tab.dispose()
    tab.close()


def test_clear_system_empties_the_field(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    tab = _make_tab(manager, ThemeManager(app), store)
    tab.clear_system("toggle_overlay")
    assert manager.settings.hotkeys.toggle_overlay == ""
    row = next(r for r in tab.system_rows() if r.spec.field == "toggle_overlay")
    assert row.status is SystemHotkeyStatus.EMPTY
    tab.dispose()
    tab.close()


def test_reset_system_restores_the_default(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    default_cancel = Settings().hotkeys.cancel
    tab = _make_tab(manager, ThemeManager(app), store)
    tab.clear_system("cancel")
    assert manager.settings.hotkeys.cancel == ""
    tab.reset_system("cancel")
    assert manager.settings.hotkeys.cancel == default_cancel
    tab.dispose()
    tab.close()


def test_system_clash_is_refused_and_leaves_config_intact(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    before = manager.settings.hotkeys.push_to_talk
    tab = _make_tab(manager, ThemeManager(app), store)
    # toggle_wake owns ctrl+shift+a by default; push_to_talk may not also claim it.
    taken = _hotkey(manager.settings.hotkeys.toggle_wake)
    assert tab.assign_system("push_to_talk", taken) is False
    assert manager.settings.hotkeys.push_to_talk == before
    tab.dispose()
    tab.close()


def test_reset_all_system_reports_nothing_to_do_at_defaults(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    tab = _make_tab(manager, ThemeManager(app), store)
    # Fresh config is already at the defaults, so there is nothing to confirm and
    # nothing to write — the dialog must never open (it would hang the run).
    tab.reset_all_system()
    assert manager.settings.hotkeys.push_to_talk == Settings().hotkeys.push_to_talk
    tab.dispose()
    tab.close()


def test_assign_command_writes_a_trigger_and_marks_active(
    app: QApplication, manager: ConfigManager, repos: Repositories, store: CommandTreeStore
) -> None:
    command_id = _add_command(repos, store, "Свет")
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.assign_command(command_id, _hotkey("ctrl+alt+l")) is True
    row = next(r for r in tab.command_rows() if r.command_id == command_id)
    assert row.combo == "ctrl+alt+l"
    assert row.status is CommandHotkeyStatus.ACTIVE
    tab.dispose()
    tab.close()


def test_clear_command_drops_the_hotkey_trigger(
    app: QApplication, manager: ConfigManager, repos: Repositories, store: CommandTreeStore
) -> None:
    command_id = _add_command(repos, store, "Свет")
    tab = _make_tab(manager, ThemeManager(app), store)
    tab.assign_command(command_id, _hotkey("ctrl+alt+l"))
    assert tab.clear_command(command_id) is True
    assert all(r.command_id != command_id for r in tab.command_rows())
    tab.dispose()
    tab.close()


def test_disabled_command_hotkey_reads_as_disabled(
    app: QApplication, manager: ConfigManager, repos: Repositories, store: CommandTreeStore
) -> None:
    command_id = _add_command(repos, store, "Тьма", enabled=False)
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.assign_command(command_id, _hotkey("ctrl+alt+d")) is True
    row = next(r for r in tab.command_rows() if r.command_id == command_id)
    assert row.status is CommandHotkeyStatus.DISABLED
    tab.dispose()
    tab.close()


def test_conflict_between_command_and_system_marks_both_rows(
    app: QApplication, manager: ConfigManager, repos: Repositories, store: CommandTreeStore
) -> None:
    command_id = _add_command(repos, store, "Свет")
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.assign_system("push_to_talk", _hotkey("ctrl+alt+k")) is True
    # The command is allowed to take the same combo; it is not refused but shown
    # as a conflict in both rows, like the manager keeping only the first claimant.
    assert tab.assign_command(command_id, _hotkey("ctrl+alt+k")) is True

    system_row = next(r for r in tab.system_rows() if r.spec.field == "push_to_talk")
    command_row = next(r for r in tab.command_rows() if r.command_id == command_id)
    assert any("Свет" in owner for owner in system_row.conflict_with)
    assert "Push-to-Talk" in command_row.conflict_with
    assert command_row.status is CommandHotkeyStatus.UNREGISTERED
    assert tab.problem_count >= 2
    tab.dispose()
    tab.close()


def test_set_interception_is_refused_without_a_driver(
    app: QApplication,
    manager: ConfigManager,
    store: CommandTreeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ayris.gui.tabs.hotkeys.probe_interception",
        lambda: InterceptionStatus(InterceptionState.MISSING, "нет драйвера"),
    )
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.set_interception(True) is False
    assert manager.settings.hotkeys.use_interception is False
    # The switch was snapped back off so it never lies about the live backend.
    assert tab._interception_toggle.isChecked() is False
    tab.dispose()
    tab.close()


def test_set_interception_enabled_with_a_ready_driver(
    app: QApplication,
    manager: ConfigManager,
    store: CommandTreeStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "ayris.gui.tabs.hotkeys.probe_interception",
        lambda: InterceptionStatus(InterceptionState.READY, "готово"),
    )
    tab = _make_tab(manager, ThemeManager(app), store)
    assert tab.set_interception(True) is True
    assert manager.settings.hotkeys.use_interception is True
    tab.dispose()
    tab.close()


def test_open_command_publishes_the_request(
    app: QApplication, manager: ConfigManager, store: CommandTreeStore
) -> None:
    bus = EventBus()
    received: list[int] = []
    bus.subscribe(OpenCommandRequested, lambda e: received.append(e.command_id), weak=False)
    tab = _make_tab(manager, ThemeManager(app), store, bus=bus)
    tab._open_command(7)
    assert received == [7]
    tab.dispose()
    tab.close()
