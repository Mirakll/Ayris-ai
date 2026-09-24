"""Task 59: the «Приватность» tab on a temporary database, offscreen.

The tab's engine (retention, cleanup, backup, the audit reader, the secrets
store) is proven elsewhere; the checks here are the tab's own guarantees, the
part the engine cannot cover on its own:

* telemetry is off, unbound and opens no outgoing socket when the page loads;
* every destructive button confirms with the *real* count before it runs, and a
  declined confirmation changes nothing;
* the full reset makes a backup *before* it clears and keeps the commands;
* retention deletes only expired rows, each category on its own clock;
* the composition toggles actually change behaviour, not just their checkbox;
* an API key value never reaches the profile export, the logs or a bug report.

Everything is offscreen; the tab is disposed and closed in teardown, or the
DataInventory size-walk thread hangs CI on the join that never comes.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import sqlite3
import subprocess
import sys
import zipfile
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication, QDialog

from ayris.actions.system.clipboard import ClipboardKind, ClipboardSnapshot, record_clipboard
from ayris.core import paths as paths_module
from ayris.core.audit import AuditFilter, AuditReader
from ayris.core.config import ConfigManager
from ayris.core.database import Database, reset_database
from ayris.core.events import EventBus
from ayris.core.models import (
    AuditEntry,
    Command,
    ExecutionResult,
    HistoryEntry,
    VariableScope,
    to_db_timestamp,
    utc_now,
)
from ayris.core.portable_profile import export_bundle
from ayris.core.repositories import Repositories
from ayris.core.secrets import SecretsStore, mask
from ayris.gui.tabs import privacy as privacy_module
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.privacy import PrivacyServices, PrivacyTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.audit_view import AuditView, serialize_audit_rows
from ayris.utils.bug_report import build_bug_report
from ayris.utils.logger import setup_logging, shutdown_logging

pytestmark = pytest.mark.unit


# -- fixtures -------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> paths_module.AppPaths:
    return paths_module.init_paths(profile=tmp_path / "root")


@pytest.fixture
def database(paths: paths_module.AppPaths) -> Iterator[Database]:
    handle = Database.open(paths.database_file)
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)


@pytest.fixture
def bus() -> EventBus:
    return EventBus(thread_id=None)


@pytest.fixture
def config(paths: paths_module.AppPaths) -> ConfigManager:
    manager = ConfigManager(paths.config_file)
    manager.load()
    return manager


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


# -- test doubles and helpers --------------------------------------------


class FakeKeyring:
    """In-memory stand-in for :mod:`keyring`, so no test touches Windows.

    A missing entry raises ``KeyError`` on delete — which the store reads as
    "nothing to remove", exactly like the real backend does.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self._store.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self._store[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self._store[(service_name, username)]


def _services(
    repos: Repositories,
    paths: paths_module.AppPaths,
    *,
    secrets: SecretsStore | None = None,
) -> PrivacyServices:
    return PrivacyServices(
        repositories=repos,
        paths=paths,
        secrets=(
            secrets if secrets is not None else SecretsStore("AyrisTest", backend=FakeKeyring())
        ),
        audit_reader=AuditReader(repos.audit),
        open_folder=lambda _folder: None,
    )


@contextlib.contextmanager
def _privacy_tab(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    services: PrivacyServices,
) -> Iterator[PrivacyTab]:
    """Build the tab and always dispose it — the size-walk thread must be joined."""
    tab = PrivacyTab(config, theme, bus, services=services)
    try:
        yield tab
    finally:
        tab.dispose()
        tab.close()


def _run_synchronously(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the background DB runner fire on the calling thread.

    ``_run`` emits ``finished``/``failed`` directly, and under a live
    ``QApplication`` a same-thread emit invokes the connected slot inline, so the
    whole cleanup completes before the handler returns — no thread, no waiting.
    """
    monkeypatch.setattr(privacy_module._DbRunner, "run", privacy_module._DbRunner._run)


def _seed_history(repos: Repositories, count: int) -> None:
    for index in range(count):
        repos.history.add(HistoryEntry(stt_raw=f"фраза {index}", intent="command:1"))


def _backdate_clipboard(database: Database, entry_id: int | None, days: int) -> None:
    """Clipboard rows are always stamped ``now`` on insert; age one by hand."""
    assert entry_id is not None
    when = utc_now() - timedelta(days=days)
    with database.transaction():
        database.execute(
            "UPDATE clipboard_history SET ts = ? WHERE id = ?",
            (to_db_timestamp(when), entry_id),
        )


# -- registration ---------------------------------------------------------


def test_tab_registers_itself_as_the_privacy_factory() -> None:
    assert tab_spec("privacy").factory is PrivacyTab


def test_privacy_page_is_wired_through_the_package_import() -> None:
    """A clean interpreter must reach the real page through ``ayris.gui.tabs`` alone.

    The in-process assertion above only passes because this module imported the
    submodule; a fresh subprocess proves the ``register_tab`` wire that swaps the
    placeholder for the real page at startup.
    """
    src = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(src), env.get("PYTHONPATH", "")) if p)
    code = (
        "from ayris.gui.tabs import tab_spec\n"
        "spec = tab_spec('privacy')\n"
        "assert spec.factory is not None, 'privacy still shows the placeholder'\n"
        "print(spec.factory.__name__)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "PrivacyTab"


# -- 1. telemetry ---------------------------------------------------------


def test_telemetry_is_off_unbound_and_opens_no_socket(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[object] = []

    class _BlockedSocket(socket.socket):
        def connect(self, address: object) -> None:  # type: ignore[override]
            attempts.append(address)
            raise AssertionError(f"вкладка открыла исходящее соединение: {address!r}")

        def connect_ex(self, address: object) -> int:  # type: ignore[override]
            attempts.append(address)
            raise AssertionError(f"вкладка открыла исходящее соединение: {address!r}")

    monkeypatch.setattr(socket, "socket", _BlockedSocket)
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        tab.load_from_config()
        app.processEvents()
        # Telemetry is a dead control: never bound, so flipping it cannot persist.
        assert "privacy.telemetry" not in tab._bindings
        assert config.settings.privacy.telemetry is False
        assert attempts == []


# -- 2. deletion with counts and confirmation -----------------------------


def test_deleting_history_confirms_with_the_real_count_then_empties_it(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_history(repos, 3)
    _run_synchronously(monkeypatch)
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        seen: dict[str, str] = {}

        def _confirm(title: str, text: str, confirm_text: str) -> bool:
            seen["text"] = text
            return True

        tab._confirm = _confirm  # type: ignore[method-assign]
        tab._delete_history()
        app.processEvents()

        assert "Будет удалено записей: 3" in seen["text"]
        assert repos.history.count() == 0


def test_a_declined_confirmation_deletes_nothing(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_history(repos, 2)
    _run_synchronously(monkeypatch)
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        tab._confirm = lambda *_args: False  # type: ignore[method-assign]
        tab._delete_history()
        app.processEvents()

        assert repos.history.count() == 2


def test_deleting_clipboard_warns_about_pinned_and_empties_it(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repos.clipboard.add("обычное")
    repos.clipboard.add("важное", pinned=True)
    _run_synchronously(monkeypatch)
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        seen: dict[str, str] = {}

        def _confirm(title: str, text: str, confirm_text: str) -> bool:
            seen["text"] = text
            return True

        tab._confirm = _confirm  # type: ignore[method-assign]
        tab._delete_clipboard()
        app.processEvents()

        assert "Будет удалено записей: 2 (включая закреплённые)" in seen["text"]
        assert repos.clipboard.count() == 0


def test_full_reset_backs_up_before_clearing_and_keeps_commands(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
    database: Database,
    app: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = repos.profiles.create("Тест", activate=True)
    assert profile.id is not None
    _seed_history(repos, 4)
    repos.audit.add(AuditEntry(command_name="громкость", require_admin=True))
    repos.clipboard.add("буфер")
    repos.variables.set(
        "город", "Москва", scope=VariableScope.PROFILE, profile_id=profile.id, persistent=True
    )
    repos.commands.create(Command(name="Свет", profile_id=profile.id))
    commands_before = repos.commands.count()

    class _AcceptDialog:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(privacy_module, "_TypedConfirmDialog", _AcceptDialog)
    _run_synchronously(monkeypatch)
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        tab._full_reset()
        app.processEvents()

    # A backup was written before the wipe, and it still holds the seeded rows.
    backups = list(paths.database_file.parent.glob(f"{paths.database_file.stem}_backup_*.db"))
    assert len(backups) == 1
    with contextlib.closing(sqlite3.connect(backups[0])) as backup_db:
        assert backup_db.execute("SELECT COUNT(*) FROM history").fetchone()[0] == 4

    # The live profile is empty, but the commands themselves survived.
    stats = repos.maintenance.statistics()
    assert stats["history"] == 0
    assert stats["audit"] == 0
    assert stats["clipboard_history"] == 0
    assert stats["variables"] == 0
    assert repos.commands.count() == commands_before


# -- 3. per-category retention -------------------------------------------


def test_retention_removes_only_expired_rows_each_category_on_its_own_clock(
    repos: Repositories, database: Database
) -> None:
    # History: one old, one fresh.
    repos.history.add(HistoryEntry(stt_raw="старое", ts=utc_now() - timedelta(days=40)))
    repos.history.add(HistoryEntry(stt_raw="свежее", ts=utc_now() - timedelta(days=1)))
    # Audit: one old, one fresh.
    repos.audit.add(AuditEntry(command_name="старая", ts=utc_now() - timedelta(days=40)))
    repos.audit.add(AuditEntry(command_name="свежая", ts=utc_now() - timedelta(days=1)))
    # Clipboard: old unpinned, old pinned, fresh unpinned.
    old_unpinned = repos.clipboard.add("старое")
    old_pinned = repos.clipboard.add("важное", pinned=True)
    repos.clipboard.add("свежее")
    _backdate_clipboard(database, old_unpinned.id, days=40)
    _backdate_clipboard(database, old_pinned.id, days=40)

    # History alone: audit and clipboard untouched.
    repos.maintenance.apply_retention(history_days=30)
    stats = repos.maintenance.statistics()
    assert stats["history"] == 1
    assert stats["audit"] == 2
    assert stats["clipboard_history"] == 3

    # Audit alone.
    repos.maintenance.apply_retention(audit_days=30)
    assert repos.maintenance.statistics()["audit"] == 1

    # Clipboard alone: the pinned old row and the fresh row both survive.
    repos.maintenance.apply_retention(clipboard_days=30)
    assert repos.maintenance.statistics()["clipboard_history"] == 2
    assert repos.clipboard.count(pinned=True) == 1


# -- 4. audit journal: filters, search, export ---------------------------


def test_audit_reader_filters_by_result_and_command(repos: Repositories) -> None:
    repos.audit.add(AuditEntry(command_name="громкость", result=ExecutionResult.OK))
    repos.audit.add(AuditEntry(command_name="музыка", result=ExecutionResult.ERROR))
    repos.audit.add(AuditEntry(command_name="музыка стоп", result=ExecutionResult.OK))
    reader = AuditReader(repos.audit)

    assert reader.page(AuditFilter()).total == 3
    errors = reader.page(AuditFilter(result=ExecutionResult.ERROR))
    assert errors.total == 1
    assert errors.items[0].command_name == "музыка"
    by_command = reader.page(AuditFilter(command="музыка"))
    assert by_command.total == 2


def test_serialize_audit_rows_renders_csv_json_and_rejects_other() -> None:
    rows = (AuditEntry(command_name="громкость", params={"value": 50}, result=ExecutionResult.OK),)
    payload = json.loads(serialize_audit_rows(rows, "json"))
    assert payload[0]["command"] == "громкость"
    assert payload[0]["result"] == "ok"
    assert payload[0]["params"] == {"value": 50}
    assert serialize_audit_rows(rows, "csv").splitlines()[0].startswith("ts,command,params")
    with pytest.raises(ValueError):
        serialize_audit_rows(rows, "xml")


def test_audit_view_exports_every_shown_row_when_nothing_is_selected(
    repos: Repositories, theme: ThemeManager, app: QApplication, tmp_path: Path
) -> None:
    repos.audit.add(AuditEntry(command_name="громкость", result=ExecutionResult.OK))
    repos.audit.add(AuditEntry(command_name="музыка", result=ExecutionResult.ERROR))
    view = AuditView(AuditReader(repos.audit), theme)
    try:
        assert view._table.rowCount() == 2
        target = tmp_path / "audit.json"
        assert view.export_to(str(target), "json") == 2
        exported = json.loads(target.read_text(encoding="utf-8"))
        assert {row["command"] for row in exported} == {"громкость", "музыка"}
    finally:
        view.close()


# -- 6. what gets recorded: the composition toggles ----------------------


def test_composition_toggles_persist_to_config(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
) -> None:
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        tab.load_from_config()
        targets: dict[str, bool] = {}
        for path in (
            "privacy.record_transcript",
            "actions.clipboard.monitor",
            "privacy.audit_params",
        ):
            widget = tab._bindings[path].widget
            new_value = not widget.isChecked()
            widget.setChecked(new_value)
            targets[path] = new_value
        tab.flush_pending()

    assert config.settings.privacy.record_transcript is targets["privacy.record_transcript"]
    assert config.settings.actions.clipboard.monitor is targets["actions.clipboard.monitor"]
    assert config.settings.privacy.audit_params is targets["privacy.audit_params"]


def test_clipboard_monitor_switch_actually_gates_recording(
    config: ConfigManager, repos: Repositories
) -> None:
    snapshot = ClipboardSnapshot(kind=ClipboardKind.TEXT, text="скопированный текст")

    config.apply({"actions.clipboard.monitor": False})
    off = record_clipboard(
        snapshot, store=repos.clipboard, settings=config.settings.actions.clipboard
    )
    assert off.stored is False
    assert off.reason == "disabled"
    assert repos.clipboard.count() == 0

    config.apply({"actions.clipboard.monitor": True})
    on = record_clipboard(
        snapshot, store=repos.clipboard, settings=config.settings.actions.clipboard
    )
    assert on.stored is True
    assert repos.clipboard.count() == 1


def test_retention_combos_persist_each_category(
    config: ConfigManager,
    theme: ThemeManager,
    bus: EventBus,
    repos: Repositories,
    paths: paths_module.AppPaths,
) -> None:
    wanted = {
        "privacy.retention_history_days": 7,
        "privacy.retention_clipboard_days": 30,
        "privacy.retention_audit_days": 90,
    }
    with _privacy_tab(config, theme, bus, _services(repos, paths)) as tab:
        tab.load_from_config()
        for path, days in wanted.items():
            combo = tab._bindings[path].widget
            combo.setCurrentIndex(combo.findData(days))
        tab.flush_pending()

    privacy = config.settings.privacy
    assert privacy.retention_history_days == 7
    assert privacy.retention_clipboard_days == 30
    assert privacy.retention_audit_days == 90


# -- 7. the API key never leaves the store -------------------------------


def _zip_holds(archive: Path, needle: bytes) -> bool:
    """Whether ``needle`` appears in the *contents* of any archive member."""
    with zipfile.ZipFile(archive) as bundle:
        return any(needle in bundle.read(name) for name in bundle.namelist())


def test_api_key_value_never_reaches_export_logs_or_bug_report(
    config: ConfigManager,
    repos: Repositories,
    paths: paths_module.AppPaths,
    database: Database,
) -> None:
    secret = "sk-secret-DONOTLEAK-9f8e7d6c5b4a3210"
    needle = secret.encode("utf-8")
    store = SecretsStore("AyrisTest", backend=FakeKeyring())
    profile = repos.profiles.create("Тест", activate=True)
    assert profile.id is not None
    config.apply({"ai.credential_ref": "openai"})  # the name lives in config, never the key

    # Send this profile's logs to a place the test can read back.
    shutdown_logging()
    setup_logging(log_dir=paths.logs_dir, console=False)
    try:
        store.save("openai", secret)
        assert store.get("openai") == secret  # the key itself is in the store
    finally:
        shutdown_logging()

    bundle = paths.database_file.parent / "profile.zip"
    export_bundle(repos, bundle, profile=profile, settings=config.settings, include_sounds=False)
    assert not _zip_holds(bundle, needle)

    report = build_bug_report(paths.cache_dir / "report.zip", paths=paths, database=database)
    assert not _zip_holds(report.path, needle)

    logs = [
        path.read_text(encoding="utf-8", errors="replace") for path in paths.logs_dir.glob("*.log")
    ]
    assert logs, "лог не был записан"
    assert all(secret not in blob for blob in logs)
    assert any(mask(secret) in blob for blob in logs)  # the store logged the mask, not the key

    assert store.delete("openai") is True  # the «delete keys» button clears the store entry
    assert store.get("openai") is None
