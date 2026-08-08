"""Task 03: the SQLite layer — connection, migrations, repositories, cleanup.

Everything here runs against a database in ``tmp_path``. That is deliberate
rather than convenient: an in-memory database silently behaves differently in
the two places this task cares most about — WAL mode is a no-op without a file,
and the backup helper has nothing to copy — so the tests that assert on those
would pass while the shipped code was broken.

The tests are grouped by what could plausibly break in production:

* :class:`TestConnection` — pragmas and transaction semantics.
* :class:`TestMigrations` — a clean start, an upgrade from an older file, and
  the guarantee that a normal start is a no-op.
* Repository classes — CRUD, and the referential rules that make a delete safe.
* :class:`TestMaintenance` — retention, the Privacy tab's erase, backup.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ayris.core.database import Database, get_database, init_database, reset_database
from ayris.core.errors import DatabaseError
from ayris.core.migrations import (
    MIGRATIONS,
    SCHEMA_VERSION,
    Migration,
    apply_migrations,
    current_version,
    schema_version_row,
)
from ayris.core.models import (
    AuditEntry,
    Command,
    CommandFolder,
    ExecutionResult,
    HistoryEntry,
    ModelRecord,
    Timer,
    TimerKind,
    Trigger,
    TriggerType,
    VariableScope,
    VariableType,
    utc_now,
)
from ayris.core.repositories import CleanupCategory, Repositories

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: Every table the task requires, plus the version marker.
EXPECTED_TABLES = frozenset(
    {
        "schema_version",
        "profiles",
        "command_folders",
        "commands",
        "command_versions",
        "triggers",
        "variables",
        "history",
        "audit",
        "timers",
        "clipboard_history",
        "models",
    }
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Location of the test database. The file itself is created on demand."""
    return tmp_path / "ayris.db"


@pytest.fixture
def database(db_path: Path) -> Iterator[Database]:
    """An open, migrated database on disk."""
    handle = Database.open(db_path)
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    """Repositories over the test database."""
    return Repositories(database)


@pytest.fixture
def profile_id(repos: Repositories) -> int:
    """An active profile to hang commands off. Most tests need one."""
    profile = repos.profiles.create("тест", activate=True)
    assert profile.id is not None
    return profile.id


def table_names(database: Database) -> set[str]:
    rows = database.query_all("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {str(row["name"]) for row in rows}


def make_command(profile_id: int, name: str = "Свет", **kwargs: object) -> Command:
    """A command with the fields a test does not care about already filled in."""
    return Command(name=name, profile_id=profile_id, **kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _FakePaths:
    """Stands in for ``AppPaths`` where only the database location matters."""

    database_file: Path


# ----------------------------------------------------------------------


class TestConnection:
    """Pragmas and the transaction wrapper."""

    def test_open_creates_the_file_and_its_parents(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "ayris.db"

        with Database.open(target) as database:
            assert database.path == target

        assert target.exists()

    def test_wal_and_foreign_keys_are_on(self, database: Database) -> None:
        assert str(database.query_value("PRAGMA journal_mode")).lower() == "wal"
        assert database.query_value("PRAGMA foreign_keys") == 1

    def test_busy_timeout_is_set(self, database: Database) -> None:
        assert database.query_value("PRAGMA busy_timeout") > 0

    def test_transaction_commits(self, database: Database) -> None:
        with database.transaction():
            database.execute(
                "INSERT INTO profiles (name, created_at) VALUES ('a', '2026-01-01T00:00:00+00:00')"
            )

        assert database.query_value("SELECT COUNT(*) FROM profiles") == 1

    def test_transaction_rolls_back_on_error(self, database: Database) -> None:
        with pytest.raises(RuntimeError), database.transaction():
            database.execute(
                "INSERT INTO profiles (name, created_at) VALUES ('a', '2026-01-01T00:00:00+00:00')"
            )
            raise RuntimeError("boom")

        assert database.query_value("SELECT COUNT(*) FROM profiles") == 0

    def test_nested_transaction_rolls_back_only_the_inner_block(self, database: Database) -> None:
        """A failing inner block must not undo the outer one.

        This is what lets a repository open a transaction without caring whether
        its caller already did.
        """
        with database.transaction():
            database.execute(
                "INSERT INTO profiles (name, created_at)"
                " VALUES ('внешний', '2026-01-01T00:00:00+00:00')"
            )
            with pytest.raises(RuntimeError), database.transaction():
                database.execute(
                    "INSERT INTO profiles (name, created_at)"
                    " VALUES ('внутренний', '2026-01-01T00:00:00+00:00')"
                )
                raise RuntimeError("boom")

        names = [row["name"] for row in database.query_all("SELECT name FROM profiles")]
        assert names == ["внешний"]

    def test_sqlite_errors_arrive_as_database_error(self, database: Database) -> None:
        """Callers catch one exception type, not sqlite3's hierarchy."""
        with pytest.raises(DatabaseError):
            database.execute("SELECT * FROM no_such_table")

    def test_constraint_violation_carries_a_russian_message(
        self, repos: Repositories, profile_id: int
    ) -> None:
        repos.commands.create(make_command(profile_id))

        with pytest.raises(DatabaseError) as caught:
            repos.commands.create(make_command(profile_id))

        assert caught.value.user_message
        assert caught.value.user_message.isascii() is False

    def test_each_thread_gets_its_own_connection(self, database: Database) -> None:
        """Thread-local connections: sqlite3 objects are not shareable."""
        seen: list[int] = []

        def record() -> None:
            seen.append(id(database.connect()))

        worker = threading.Thread(target=record)
        worker.start()
        worker.join()

        assert seen and seen[0] != id(database.connect())

    def test_close_is_idempotent(self, db_path: Path) -> None:
        database = Database.open(db_path)
        database.close()
        database.close()

        assert database.is_closed

    def test_integrity_check_passes(self, database: Database) -> None:
        assert database.integrity_check() is True


class TestGlobalHandle:
    """``init_database`` / ``get_database`` — the handle the app shares."""

    def test_get_opens_lazily(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same lazy contract as ``get_paths`` and ``get_config_manager``.

        A script or a test that only reads must not have to call
        ``init_database`` first.
        """
        reset_database()
        monkeypatch.setattr(
            "ayris.core.database.get_paths",
            lambda: _FakePaths(tmp_path / "lazy.db"),
        )

        try:
            database = get_database()
            assert get_database() is database
            assert current_version(database) == SCHEMA_VERSION
        finally:
            reset_database()

    def test_init_then_get_returns_the_same_object(self, db_path: Path) -> None:
        created = init_database(db_path)
        try:
            assert get_database() is created
        finally:
            reset_database()

    def test_reset_closes_the_handle(self, db_path: Path) -> None:
        created = init_database(db_path)
        reset_database()

        assert created.is_closed


class TestMigrations:
    """Schema versioning from scratch, and over an existing file."""

    def test_clean_start_creates_every_table(self, database: Database) -> None:
        assert table_names(database) >= EXPECTED_TABLES

    def test_clean_start_records_the_version(self, database: Database) -> None:
        assert current_version(database) == SCHEMA_VERSION

        row = schema_version_row(database)
        assert row is not None
        version, applied_at = row
        assert version == SCHEMA_VERSION
        assert applied_at

    def test_indexes_for_the_hot_queries_exist(self, database: Database) -> None:
        """The three access paths the task calls out by name."""
        rows = database.query_all("SELECT name FROM sqlite_master WHERE type = 'index'")
        names = {str(row["name"]) for row in rows}

        assert {
            "idx_triggers_command",  # triggers reached through their command
            "idx_commands_profile",  # ... whose profile is filtered here
            "idx_history_ts",
            "idx_timers_due",
        } <= names

    def test_second_open_does_not_rerun_migrations(self, db_path: Path) -> None:
        """Restarting must be a no-op, not a re-application.

        Compares the recorded ``applied_at``: a re-run would refresh it even
        though the tables already exist and nothing would visibly break.
        """
        with Database.open(db_path) as first:
            before = schema_version_row(first)

        with Database.open(db_path) as second:
            after = schema_version_row(second)
            applied = apply_migrations(second)

        assert before == after
        assert applied == SCHEMA_VERSION

    def test_data_survives_a_reopen(self, db_path: Path) -> None:
        with Database.open(db_path) as first:
            Repositories(first).profiles.create("сохранённый")

        with Database.open(db_path) as second:
            assert Repositories(second).profiles.get_by_name("сохранённый") is not None

    def test_migration_over_an_older_database(self, db_path: Path) -> None:
        """The real upgrade path: a file at v1 meets a build that knows v2."""
        extra = Migration(
            version=SCHEMA_VERSION + 1,
            description="тестовая таблица",
            statements=("CREATE TABLE later (id INTEGER PRIMARY KEY, note TEXT NOT NULL)",),
        )

        with Database.open(db_path) as database:
            repos = Repositories(database)
            repos.profiles.create("старый профиль")
            assert current_version(database) == SCHEMA_VERSION

            applied = apply_migrations(database, migrations=[*MIGRATIONS, extra])

            assert applied == extra.version
            assert current_version(database) == extra.version
            assert "later" in table_names(database)
            # The upgrade must not disturb what was already there.
            assert repos.profiles.get_by_name("старый профиль") is not None

    def test_migrations_run_in_order_from_empty(self, db_path: Path) -> None:
        order: list[int] = []
        extra = Migration(
            version=SCHEMA_VERSION + 1,
            description="колонка",
            statements=("ALTER TABLE profiles ADD COLUMN note TEXT NOT NULL DEFAULT ''",),
            callback=lambda _db: order.append(SCHEMA_VERSION + 1),
        )

        with Database.open(db_path, migrate=False) as database:
            assert current_version(database) == 0
            apply_migrations(database, migrations=[*MIGRATIONS, extra])

        assert order == [SCHEMA_VERSION + 1]

    def test_target_stops_early(self, db_path: Path) -> None:
        extra = Migration(
            version=SCHEMA_VERSION + 1,
            description="не должна примениться",
            statements=("CREATE TABLE untouched (id INTEGER PRIMARY KEY)",),
        )

        with Database.open(db_path, migrate=False) as database:
            apply_migrations(database, target=SCHEMA_VERSION, migrations=[*MIGRATIONS, extra])

            assert current_version(database) == SCHEMA_VERSION
            assert "untouched" not in table_names(database)

    def test_a_failed_migration_leaves_the_old_version(self, db_path: Path) -> None:
        """A broken upgrade must roll back, so the older build still starts."""
        broken = Migration(
            version=SCHEMA_VERSION + 1,
            description="сломанная",
            statements=(
                "CREATE TABLE half (id INTEGER PRIMARY KEY)",
                "THIS IS NOT SQL",
            ),
        )

        with Database.open(db_path) as database:
            with pytest.raises(DatabaseError):
                apply_migrations(database, migrations=[*MIGRATIONS, broken])

            assert current_version(database) == SCHEMA_VERSION
            assert "half" not in table_names(database)

    def test_a_newer_database_is_refused(self, db_path: Path) -> None:
        """Opening a file from a future build read-write would corrupt it."""
        with Database.open(db_path) as database:
            database.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")

            with pytest.raises(DatabaseError) as caught:
                apply_migrations(database)

        assert caught.value.recoverable is False


class TestProfileRepository:
    def test_create_and_read_back(self, repos: Repositories) -> None:
        created = repos.profiles.create("основной")

        assert created.id is not None
        assert created.created_at is not None
        assert repos.profiles.get(created.id) == created

    def test_only_one_profile_is_active(self, repos: Repositories) -> None:
        first = repos.profiles.create("первый", activate=True)
        second = repos.profiles.create("второй", activate=True)

        active = repos.profiles.active()
        assert active is not None
        assert active.id == second.id
        assert repos.profiles.get(first.id or 0) is not None
        assert repos.profiles.count() == 2

    def test_set_active_switches(self, repos: Repositories) -> None:
        first = repos.profiles.create("первый", activate=True)
        second = repos.profiles.create("второй")
        assert second.id is not None

        repos.profiles.set_active(second.id)

        active = repos.profiles.active()
        assert active is not None and active.id == second.id
        stored_first = repos.profiles.get(first.id or 0)
        assert stored_first is not None and stored_first.is_active is False

    def test_duplicate_names_are_rejected(self, repos: Repositories) -> None:
        repos.profiles.create("дубль")

        with pytest.raises(DatabaseError):
            repos.profiles.create("дубль")

    def test_delete_removes_the_whole_profile(self, repos: Repositories, profile_id: int) -> None:
        """Deleting a profile must not strand its commands or folders."""
        folder = repos.folders.create(CommandFolder(name="Папка", profile_id=profile_id))
        command = repos.commands.create(make_command(profile_id, folder_id=folder.id))
        repos.triggers.add_voice(command.id or 0, "тест")
        repos.variables.set("x", 1, scope=VariableScope.PROFILE, profile_id=profile_id)

        assert repos.profiles.delete(profile_id) is True

        assert repos.commands.count(profile_id) == 0
        assert repos.folders.list_for_profile(profile_id) == []
        assert repos.triggers.list_for_command(command.id or 0) == []
        assert repos.variables.list_all(profile_id=profile_id) == []


class TestFolderRepository:
    def test_tree_navigation(self, repos: Repositories, profile_id: int) -> None:
        root = repos.folders.create(CommandFolder(name="Система", profile_id=profile_id))
        child = repos.folders.create(
            CommandFolder(name="Звук", profile_id=profile_id, parent_id=root.id)
        )

        assert repos.folders.children(None) == [root]
        assert repos.folders.children(root.id) == [child]

    def test_reorder_renumbers(self, repos: Repositories, profile_id: int) -> None:
        first = repos.folders.create(CommandFolder(name="A", profile_id=profile_id))
        second = repos.folders.create(CommandFolder(name="B", profile_id=profile_id))

        repos.folders.reorder([second.id or 0, first.id or 0])

        ordered = [f.name for f in repos.folders.list_for_profile(profile_id)]
        assert ordered == ["B", "A"]

    def test_deleting_a_folder_keeps_its_commands(
        self, repos: Repositories, profile_id: int
    ) -> None:
        """Commands move to the root instead of vanishing with the folder."""
        folder = repos.folders.create(CommandFolder(name="Временная", profile_id=profile_id))
        command = repos.commands.create(make_command(profile_id, folder_id=folder.id))

        repos.folders.delete(folder.id or 0)

        stored = repos.commands.get(command.id or 0)
        assert stored is not None
        assert stored.folder_id is None

    def test_deleting_a_folder_removes_its_subtree(
        self, repos: Repositories, profile_id: int
    ) -> None:
        root = repos.folders.create(CommandFolder(name="Родитель", profile_id=profile_id))
        child = repos.folders.create(
            CommandFolder(name="Ребёнок", profile_id=profile_id, parent_id=root.id)
        )

        repos.folders.delete(root.id or 0)

        assert repos.folders.get(child.id or 0) is None

    def test_a_folder_cannot_contain_itself(self, repos: Repositories, profile_id: int) -> None:
        folder = repos.folders.create(CommandFolder(name="Петля", profile_id=profile_id))

        with pytest.raises(DatabaseError):
            repos.folders.update(replace(folder, parent_id=folder.id))


class TestCommandRepository:
    def test_round_trip_preserves_json_columns(self, repos: Repositories, profile_id: int) -> None:
        """Tags and actions must survive encode/decode unchanged."""
        actions = ({"type": "TypeText", "text": "привет"}, {"type": "Delay", "ms": 250})
        created = repos.commands.create(
            make_command(profile_id, tags=("дом", "свет"), actions=actions)
        )

        stored = repos.commands.get(created.id or 0)

        assert stored is not None
        assert stored.tags == ("дом", "свет")
        assert stored.actions == actions

    def test_timestamps_are_filled_in(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(make_command(profile_id))

        assert created.created_at is not None
        assert created.updated_at is not None
        assert created.created_at.tzinfo is not None

    def test_update_refreshes_updated_at(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(make_command(profile_id))

        updated = repos.commands.update(replace(created, description="новое"))

        assert updated.updated_at is not None
        assert created.updated_at is not None
        assert updated.updated_at >= created.updated_at
        stored = repos.commands.get(created.id or 0)
        assert stored is not None and stored.description == "новое"

    def test_list_filters(self, repos: Repositories, profile_id: int) -> None:
        repos.commands.create(make_command(profile_id, "Включён"))
        repos.commands.create(make_command(profile_id, "Выключен", enabled=False))

        assert len(repos.commands.list_for_profile(profile_id)) == 2
        enabled = repos.commands.list_for_profile(profile_id, enabled_only=True)
        assert [c.name for c in enabled] == ["Включён"]

    def test_list_is_ordered_by_priority(self, repos: Repositories, profile_id: int) -> None:
        repos.commands.create(make_command(profile_id, "Обычная"))
        repos.commands.create(make_command(profile_id, "Важная", priority=10))

        names = [c.name for c in repos.commands.list_for_profile(profile_id)]
        assert names == ["Важная", "Обычная"]

    def test_search_covers_name_and_description(self, repos: Repositories, profile_id: int) -> None:
        repos.commands.create(make_command(profile_id, "Свет"))
        repos.commands.create(make_command(profile_id, "Музыка", description="включить свет"))

        assert len(repos.commands.search(profile_id, "свет")) == 2

    def test_search_ignores_case_in_cyrillic(self, repos: Repositories, profile_id: int) -> None:
        """SQLite's own LIKE folds ASCII only, which would break search here."""
        repos.commands.create(make_command(profile_id, "Выключить Свет"))

        assert len(repos.commands.search(profile_id, "свет")) == 1
        assert len(repos.commands.search(profile_id, "СВЕТ")) == 1
        assert len(repos.commands.search(profile_id, "ВыКлЮчИтЬ")) == 1

    def test_unsaved_command_is_rejected(self, repos: Repositories, profile_id: int) -> None:
        """Catch the missing id here rather than as a foreign key error later."""
        with pytest.raises(DatabaseError):
            repos.commands.update(make_command(profile_id))


class TestCommandVersions:
    def test_versions_increment_per_command(self, repos: Repositories, profile_id: int) -> None:
        first = repos.commands.create(make_command(profile_id, "Первая"))
        second = repos.commands.create(make_command(profile_id, "Вторая"))

        repos.commands.save_version(first)
        repos.commands.save_version(first)
        repos.commands.save_version(second)

        assert [v.version for v in repos.commands.list_versions(first.id or 0)] == [2, 1]
        assert [v.version for v in repos.commands.list_versions(second.id or 0)] == [1]

    def test_update_can_snapshot_the_previous_state(
        self, repos: Repositories, profile_id: int
    ) -> None:
        created = repos.commands.create(make_command(profile_id, description="исходное"))

        repos.commands.update(replace(created, description="изменённое"), save_version=True)

        versions = repos.commands.list_versions(created.id or 0)
        assert len(versions) == 1
        assert versions[0].snapshot["description"] == "исходное"

    def test_restore_brings_back_the_old_state(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(
            make_command(profile_id, description="исходное", tags=("a",))
        )
        repos.commands.update(
            replace(created, description="изменённое", tags=("b",)), save_version=True
        )

        restored = repos.commands.restore_version(created.id or 0, 1)

        assert restored.description == "исходное"
        assert restored.tags == ("a",)
        stored = repos.commands.get(created.id or 0)
        assert stored is not None and stored.description == "исходное"

    def test_restore_snapshots_the_current_state_first(
        self, repos: Repositories, profile_id: int
    ) -> None:
        """An undo must itself be undoable."""
        created = repos.commands.create(make_command(profile_id, description="v1"))
        repos.commands.update(replace(created, description="v2"), save_version=True)

        repos.commands.restore_version(created.id or 0, 1)

        versions = repos.commands.list_versions(created.id or 0)
        assert [v.snapshot["description"] for v in versions] == ["v2", "v1"]

    def test_restoring_a_missing_version_raises(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(make_command(profile_id))

        with pytest.raises(DatabaseError):
            repos.commands.restore_version(created.id or 0, 99)

    def test_prune_keeps_the_newest(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(make_command(profile_id))
        for _ in range(5):
            repos.commands.save_version(created)

        removed = repos.commands.prune_versions(created.id or 0, keep=2)

        assert removed == 3
        assert [v.version for v in repos.commands.list_versions(created.id or 0)] == [5, 4]

    def test_versions_die_with_their_command(self, repos: Repositories, profile_id: int) -> None:
        created = repos.commands.create(make_command(profile_id))
        repos.commands.save_version(created)

        repos.commands.delete(created.id or 0)

        assert repos.commands.list_versions(created.id or 0) == []


class TestTriggerRepository:
    def test_voice_shorthand(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))

        trigger = repos.triggers.add_voice(command.id or 0, "включи свет")

        assert trigger.type is TriggerType.VOICE
        assert trigger.phrase == "включи свет"
        assert trigger.fuzzy is True

    def test_lookup_by_profile_joins_through_commands(
        self, repos: Repositories, profile_id: int
    ) -> None:
        """The matcher's startup query, including the enabled filter."""
        enabled = repos.commands.create(make_command(profile_id, "Включена"))
        disabled = repos.commands.create(make_command(profile_id, "Выключена", enabled=False))
        repos.triggers.add_voice(enabled.id or 0, "работает")
        repos.triggers.add_voice(disabled.id or 0, "не работает")

        phrases = [t.phrase for t in repos.triggers.list_for_profile(profile_id)]
        assert phrases == ["работает"]

        every = repos.triggers.list_for_profile(profile_id, enabled_only=False)
        assert len(every) == 2

    def test_lookup_can_filter_by_type(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.triggers.add_voice(command.id or 0, "фраза")
        repos.triggers.add(
            Trigger(
                command_id=command.id or 0,
                type=TriggerType.HOTKEY,
                payload={"keys": "ctrl+alt+l"},
            )
        )

        hotkeys = repos.triggers.list_for_profile(profile_id, trigger_type=TriggerType.HOTKEY)

        assert len(hotkeys) == 1
        assert hotkeys[0].payload["keys"] == "ctrl+alt+l"

    def test_triggers_are_ordered_by_priority(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.triggers.add(
            Trigger(command_id=command.id or 0, payload={"phrase": "обычная"}, priority=0)
        )
        repos.triggers.add(
            Trigger(command_id=command.id or 0, payload={"phrase": "важная"}, priority=5)
        )

        phrases = [t.phrase for t in repos.triggers.list_for_profile(profile_id)]
        assert phrases == ["важная", "обычная"]

    def test_replace_swaps_the_whole_set(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.triggers.add_voice(command.id or 0, "старая")

        repos.triggers.replace_for_command(
            command.id or 0, [Trigger(command_id=0, payload={"phrase": "новая"})]
        )

        phrases = [t.phrase for t in repos.triggers.list_for_command(command.id or 0)]
        assert phrases == ["новая"]

    def test_unknown_trigger_type_is_rejected(self, repos: Repositories, profile_id: int) -> None:
        """A CHECK constraint guards the column even if Python is bypassed."""
        command = repos.commands.create(make_command(profile_id))

        with pytest.raises(DatabaseError):
            repos.database.execute(
                'INSERT INTO "triggers" (command_id, type) VALUES (?, ?)',
                (command.id, "выдумка"),
            )

    def test_a_trigger_needs_an_existing_command(self, repos: Repositories) -> None:
        """Proves foreign keys are actually enforced, not merely declared."""
        with pytest.raises(DatabaseError):
            repos.triggers.add_voice(999, "сирота")

    def test_cascade_delete_removes_triggers(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.triggers.add_voice(command.id or 0, "фраза")

        repos.commands.delete(command.id or 0)

        assert repos.triggers.list_for_command(command.id or 0) == []


class TestVariableRepository:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("текст", VariableType.STRING),
            (42, VariableType.INT),
            (3.5, VariableType.FLOAT),
            (True, VariableType.BOOL),
            ([1, 2], VariableType.ARRAY),
            ({"a": 1}, VariableType.DICT),
        ],
    )
    def test_type_is_inferred(
        self, repos: Repositories, value: object, expected: VariableType
    ) -> None:
        """``bool`` before ``int``: it is a subclass and would be misfiled."""
        stored = repos.variables.set("v", value)

        assert stored.type is expected
        assert stored.value == value

    def test_set_overwrites_instead_of_duplicating(self, repos: Repositories) -> None:
        repos.variables.set("счётчик", 1)
        repos.variables.set("счётчик", 2)

        assert repos.variables.get_value("счётчик") == 2
        assert len(repos.variables.list_all()) == 1

    def test_scopes_are_independent(self, repos: Repositories, profile_id: int) -> None:
        repos.variables.set("режим", "глобальный")
        repos.variables.set(
            "режим", "профильный", scope=VariableScope.PROFILE, profile_id=profile_id
        )

        assert repos.variables.get_value("режим") == "глобальный"
        assert (
            repos.variables.get_value("режим", scope=VariableScope.PROFILE, profile_id=profile_id)
            == "профильный"
        )

    def test_missing_variable_returns_the_default(self, repos: Repositories) -> None:
        assert repos.variables.get("нет") is None
        assert repos.variables.get_value("нет", "запасное") == "запасное"

    def test_transient_variables_are_cleared(self, repos: Repositories) -> None:
        repos.variables.set("живёт", 1)
        repos.variables.set("умрёт", 2, persistent=False)

        assert repos.variables.clear_transient() == 1
        assert [v.name for v in repos.variables.list_all()] == ["живёт"]


class TestHistoryRepository:
    def test_add_stamps_the_time(self, repos: Repositories) -> None:
        entry = repos.history.add(HistoryEntry(stt_raw="привет"))

        assert entry.id is not None
        assert entry.ts is not None
        stored = repos.history.get(entry.id)
        assert stored is not None and stored.stt_raw == "привет"

    def test_recent_is_newest_first(self, repos: Repositories) -> None:
        for index in range(3):
            repos.history.add(HistoryEntry(stt_raw=f"фраза {index}"))

        assert [e.stt_raw for e in repos.history.recent()] == ["фраза 2", "фраза 1", "фраза 0"]

    def test_failures_excludes_successes(self, repos: Repositories) -> None:
        repos.history.add(HistoryEntry(stt_raw="ок"))
        repos.history.add(HistoryEntry(stt_raw="мимо", result=ExecutionResult.UNMATCHED))

        assert [e.stt_raw for e in repos.history.failures()] == ["мимо"]

    def test_deleting_a_command_keeps_its_history(
        self, repos: Repositories, profile_id: int
    ) -> None:
        """History is a record of what was said; that stays true afterwards."""
        command = repos.commands.create(make_command(profile_id))
        entry = repos.history.add(
            HistoryEntry(stt_raw="включи свет", matched_command_id=command.id)
        )

        repos.commands.delete(command.id or 0)

        stored = repos.history.get(entry.id or 0)
        assert stored is not None
        assert stored.stt_raw == "включи свет"
        assert stored.matched_command_id is None

    def test_purge_by_age(self, repos: Repositories) -> None:
        fresh = repos.history.add(HistoryEntry(stt_raw="свежая"))
        old = repos.history.add(HistoryEntry(stt_raw="старая", ts=utc_now() - timedelta(days=40)))

        removed = repos.history.delete_older_than(30)

        assert removed == 1
        assert repos.history.get(old.id or 0) is None
        assert repos.history.get(fresh.id or 0) is not None

    def test_purge_with_zero_days_keeps_everything(self, repos: Repositories) -> None:
        """``0`` means "no age limit", not "delete everything"."""
        repos.history.add(HistoryEntry(stt_raw="фраза"))

        assert repos.history.delete_older_than(0) == 0
        assert repos.history.count() == 1

    def test_trim_keeps_the_newest(self, repos: Repositories) -> None:
        for index in range(5):
            repos.history.add(HistoryEntry(stt_raw=f"фраза {index}"))

        removed = repos.history.trim_to_limit(2)

        assert removed == 3
        assert [e.stt_raw for e in repos.history.recent()] == ["фраза 4", "фраза 3"]

    def test_list_between_filters_the_window(self, repos: Repositories) -> None:
        repos.history.add(HistoryEntry(stt_raw="вчера", ts=utc_now() - timedelta(days=1)))
        repos.history.add(HistoryEntry(stt_raw="сейчас"))

        window = repos.history.list_between(utc_now() - timedelta(hours=1), utc_now())

        assert [e.stt_raw for e in window] == ["сейчас"]


class TestAuditRepository:
    def test_elevated_filter(self, repos: Repositories) -> None:
        repos.audit.add(AuditEntry(command_name="Обычная"))
        repos.audit.add(AuditEntry(command_name="От админа", require_admin=True, elevated=True))

        elevated = repos.audit.recent(elevated_only=True)

        assert [e.command_name for e in elevated] == ["От админа"]
        assert repos.audit.count() == 2

    def test_purge_by_age(self, repos: Repositories) -> None:
        repos.audit.add(AuditEntry(command_name="Свежая"))
        repos.audit.add(AuditEntry(command_name="Старая", ts=utc_now() - timedelta(days=400)))

        assert repos.audit.delete_older_than(365) == 1
        assert repos.audit.count() == 1


class TestTimerRepository:
    def test_due_returns_only_what_has_fired(self, repos: Repositories) -> None:
        past = repos.timers.create(
            Timer(label="прошедший", fire_at=utc_now() - timedelta(minutes=5))
        )
        repos.timers.create(Timer(label="будущий", fire_at=utc_now() + timedelta(hours=1)))

        due = repos.timers.due()

        assert [t.id for t in due] == [past.id]

    def test_disabled_timers_are_not_due(self, repos: Repositories) -> None:
        timer = repos.timers.create(
            Timer(label="выключенный", fire_at=utc_now() - timedelta(minutes=5))
        )
        repos.timers.set_enabled(timer.id or 0, enabled=False)

        assert repos.timers.due() == []

    def test_recurring_timers_wait_for_a_concrete_fire_at(self, repos: Repositories) -> None:
        """A cron entry only becomes due once the scheduler expands it."""
        timer = repos.timers.create(Timer(label="каждый день", cron="0 9 * * *"))

        assert repos.timers.due() == []

        repos.timers.reschedule(timer.id or 0, utc_now() - timedelta(minutes=1))

        assert [t.id for t in repos.timers.due()] == [timer.id]

    def test_a_timer_without_a_schedule_is_rejected(self, repos: Repositories) -> None:
        """It could never fire, so accepting it would only hide a bug."""
        with pytest.raises(DatabaseError):
            repos.timers.create(Timer(label="никогда"))

    def test_update_round_trip(self, repos: Repositories) -> None:
        timer = repos.timers.create(
            Timer(label="чай", kind=TimerKind.REMINDER, fire_at=utc_now(), payload={"n": 1})
        )

        repos.timers.update(replace(timer, label="кофе", payload={"n": 2}))

        stored = repos.timers.get(timer.id or 0)
        assert stored is not None
        assert stored.label == "кофе"
        assert stored.payload == {"n": 2}
        assert stored.kind is TimerKind.REMINDER

    def test_timestamps_survive_the_round_trip(self, repos: Repositories) -> None:
        """Microseconds included — the scheduler compares these for equality."""
        moment = datetime(2026, 8, 8, 12, 30, 15, 123456, tzinfo=UTC)
        timer = repos.timers.create(Timer(label="точный", fire_at=moment))

        stored = repos.timers.get(timer.id or 0)

        assert stored is not None and stored.fire_at == moment


class TestClipboardRepository:
    def test_pinned_entries_come_first(self, repos: Repositories) -> None:
        repos.clipboard.add("обычная")
        repos.clipboard.add("закреплённая", pinned=True)

        assert [e.content for e in repos.clipboard.recent()] == ["закреплённая", "обычная"]

    def test_clear_keeps_pinned_by_default(self, repos: Repositories) -> None:
        repos.clipboard.add("одна")
        repos.clipboard.add("закреплённая", pinned=True)

        assert repos.clipboard.clear() == 1
        assert [e.content for e in repos.clipboard.recent()] == ["закреплённая"]
        assert repos.clipboard.clear(keep_pinned=False) == 1

    def test_trim_never_drops_pinned(self, repos: Repositories) -> None:
        repos.clipboard.add("закреплённая", pinned=True)
        for index in range(4):
            repos.clipboard.add(f"запись {index}")

        repos.clipboard.trim_to_limit(2)

        contents = [e.content for e in repos.clipboard.recent()]
        assert "закреплённая" in contents
        assert len(contents) == 3


class TestModelRepository:
    def test_one_active_model_per_kind(self, repos: Repositories) -> None:
        stt = repos.models.add(ModelRecord(kind="stt", name="vosk", is_active=True))
        tts = repos.models.add(ModelRecord(kind="tts", name="silero", is_active=True))

        stt_active = repos.models.active("stt")
        tts_active = repos.models.active("tts")
        assert stt_active is not None and stt_active.id == stt.id
        assert tts_active is not None and tts_active.id == tts.id

    def test_set_active_switches_within_a_kind(self, repos: Repositories) -> None:
        first = repos.models.add(ModelRecord(kind="stt", name="small", is_active=True))
        second = repos.models.add(ModelRecord(kind="stt", name="large"))

        repos.models.set_active(second.id or 0)

        active = repos.models.active("stt")
        assert active is not None and active.id == second.id
        stored_first = repos.models.get(first.id or 0)
        assert stored_first is not None and stored_first.is_active is False

    def test_activating_a_missing_model_raises(self, repos: Repositories) -> None:
        with pytest.raises(DatabaseError):
            repos.models.set_active(404)

    def test_unknown_kind_is_rejected(self, repos: Repositories) -> None:
        with pytest.raises(DatabaseError):
            repos.database.execute("INSERT INTO models (kind, name) VALUES ('квантовая', 'x')")


class TestMaintenance:
    def test_retention_applies_both_limits(self, repos: Repositories) -> None:
        for index in range(4):
            repos.history.add(HistoryEntry(stt_raw=f"свежая {index}"))
        repos.history.add(HistoryEntry(stt_raw="старая", ts=utc_now() - timedelta(days=90)))

        removed = repos.maintenance.apply_retention(history_days=30, history_limit=2)

        assert removed == 3
        assert repos.history.count() == 2

    def test_clear_erases_the_chosen_categories(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.history.add(HistoryEntry(stt_raw="фраза"))
        repos.audit.add(AuditEntry(command_name="Свет"))
        repos.clipboard.add("буфер")

        report = repos.maintenance.clear([CleanupCategory.HISTORY, CleanupCategory.AUDIT])

        assert report.removed[CleanupCategory.HISTORY] == 1
        assert report.removed[CleanupCategory.AUDIT] == 1
        assert report.total == 2
        # Untouched categories stay, and commands are never erased by cleanup.
        assert len(repos.clipboard.recent()) == 1
        assert repos.commands.get(command.id or 0) is not None

    def test_clear_reports_in_russian(self, repos: Repositories) -> None:
        repos.history.add(HistoryEntry(stt_raw="фраза"))

        report = repos.maintenance.clear([CleanupCategory.HISTORY])

        assert "история команд" in report.summary()

    def test_clear_with_nothing_selected_is_a_no_op(self, repos: Repositories) -> None:
        repos.history.add(HistoryEntry(stt_raw="фраза"))

        report = repos.maintenance.clear([])

        assert report.total == 0
        assert repos.history.count() == 1

    def test_clear_all_empties_every_category(self, repos: Repositories, profile_id: int) -> None:
        command = repos.commands.create(make_command(profile_id))
        repos.commands.save_version(command)
        repos.history.add(HistoryEntry(stt_raw="фраза"))
        repos.audit.add(AuditEntry(command_name="Свет"))
        repos.clipboard.add("буфер", pinned=True)
        repos.variables.set("v", 1)
        repos.timers.create(Timer(label="таймер", fire_at=utc_now()))

        repos.maintenance.clear_all()

        stats = repos.maintenance.statistics()
        assert stats["history"] == 0
        assert stats["audit"] == 0
        assert stats["clipboard_history"] == 0
        assert stats["variables"] == 0
        assert stats["timers"] == 0
        assert stats["command_versions"] == 0
        # The user's commands and profiles are their work, not their traces.
        assert stats["commands"] == 1
        assert stats["profiles"] == 1

    def test_statistics_count_every_table(self, repos: Repositories, profile_id: int) -> None:
        repos.commands.create(make_command(profile_id))

        stats = repos.maintenance.statistics()

        assert stats["profiles"] == 1
        assert stats["commands"] == 1
        assert stats["size_bytes"] > 0

    def test_backup_produces_a_usable_database(
        self, repos: Repositories, profile_id: int, tmp_path: Path
    ) -> None:
        """A backup nobody can open is not a backup."""
        repos.commands.create(make_command(profile_id, "Сохранённая"))
        target = tmp_path / "backup.db"

        created = repos.maintenance.backup(target)

        assert created == target
        assert target.stat().st_size > 0

        with Database.open(target, migrate=False) as restored:
            assert current_version(restored) == SCHEMA_VERSION
            names = [c.name for c in Repositories(restored).commands.list_for_profile(profile_id)]
            assert names == ["Сохранённая"]

    def test_backup_defaults_to_a_timestamped_neighbour(
        self, repos: Repositories, db_path: Path
    ) -> None:
        created = repos.maintenance.backup()

        assert created.parent == db_path.parent
        assert created.name.startswith("ayris_backup_")

    def test_backup_captures_uncommitted_work_after_commit(
        self, repos: Repositories, profile_id: int, tmp_path: Path
    ) -> None:
        """WAL content must be checkpointed, or the copy silently lags behind."""
        repos.commands.create(make_command(profile_id, "Только что"))
        target = tmp_path / "wal.db"

        repos.maintenance.backup(target)

        with Database.open(target, migrate=False) as restored:
            assert Repositories(restored).commands.count(profile_id) == 1

    def test_vacuum_runs_and_reports(self, repos: Repositories) -> None:
        for index in range(200):
            repos.history.add(HistoryEntry(stt_raw=f"фраза {index}"))
        repos.history.clear()

        freed = repos.maintenance.vacuum()

        assert freed >= 0
        assert repos.database.integrity_check() is True

    def test_restore_replaces_the_current_database(
        self, repos: Repositories, profile_id: int, tmp_path: Path
    ) -> None:
        repos.commands.create(make_command(profile_id, "До бэкапа"))
        target = tmp_path / "restore.db"
        repos.maintenance.backup(target)
        repos.commands.create(make_command(profile_id, "После бэкапа"))

        repos.database.restore(target)

        names = {c.name for c in repos.commands.list_for_profile(profile_id)}
        assert names == {"До бэкапа"}


class TestPrivacyGuarantees:
    """Section 11 promises the user their data is really gone."""

    def test_erased_history_leaves_no_readable_trace(
        self, repos: Repositories, db_path: Path
    ) -> None:
        """VACUUM after a cleanup is what actually reclaims the pages.

        Without it the rows sit in free pages and a hex editor still finds them,
        which would make the Privacy tab a lie.
        """
        secret = "секретная фраза для проверки очистки"
        for _ in range(50):
            repos.history.add(HistoryEntry(stt_raw=secret))

        repos.maintenance.clear([CleanupCategory.HISTORY], vacuum=True)
        repos.database.checkpoint()

        assert secret.encode() not in db_path.read_bytes()

    def test_cleanup_is_all_or_nothing(self, repos: Repositories) -> None:
        """A cleanup that fails part-way must leave the data as it was."""
        repos.history.add(HistoryEntry(stt_raw="фраза"))
        repos.audit.add(AuditEntry(command_name="Свет"))

        with pytest.raises(RuntimeError), repos.database.transaction():
            repos.maintenance.clear([CleanupCategory.HISTORY], vacuum=False)
            raise RuntimeError("сбой на середине")

        assert repos.history.count() == 1
        assert repos.audit.count() == 1


class TestPersistenceAcrossRestarts:
    """The full lifecycle, as the application actually experiences it."""

    def test_a_command_survives_a_restart_with_everything_attached(self, db_path: Path) -> None:
        with Database.open(db_path) as first:
            repos = Repositories(first)
            profile = repos.profiles.create("основной", activate=True)
            folder = repos.folders.create(CommandFolder(name="Дом", profile_id=profile.id))
            command = repos.commands.create(
                Command(
                    name="Свет",
                    profile_id=profile.id or 0,
                    folder_id=folder.id,
                    tags=("дом",),
                    actions=({"type": "Delay", "ms": 100},),
                )
            )
            repos.triggers.add_voice(command.id or 0, "включи свет")
            repos.variables.set("яркость", 80)

        with Database.open(db_path) as second:
            repos = Repositories(second)
            active = repos.profiles.active()
            assert active is not None and active.name == "основной"

            stored = repos.commands.get_by_name(active.id or 0, "Свет")
            assert stored is not None
            assert stored.tags == ("дом",)
            assert stored.actions == ({"type": "Delay", "ms": 100},)

            triggers = repos.triggers.list_for_profile(active.id or 0)
            assert [t.phrase for t in triggers] == ["включи свет"]
            assert repos.variables.get_value("яркость") == 80

    def test_two_connections_see_each_other(self, db_path: Path) -> None:
        """WAL lets a reader work while a writer holds the lock."""
        with Database.open(db_path) as writer:
            Repositories(writer).profiles.create("общий")

            reader = sqlite3.connect(db_path)
            try:
                reader.row_factory = sqlite3.Row
                rows = reader.execute("SELECT name FROM profiles").fetchall()
            finally:
                reader.close()

        assert [row["name"] for row in rows] == ["общий"]
