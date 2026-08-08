"""Schema versioning.

The schema is a list of numbered migrations. On startup
:func:`apply_migrations` compares ``PRAGMA user_version`` against
:data:`SCHEMA_VERSION` and runs whatever is missing, inside one transaction per
migration, so an interrupted upgrade leaves the previous version intact rather
than a half-migrated table.

``user_version`` is used instead of a bookkeeping table because SQLite stores it
in the file header: there is no chicken-and-egg problem reading it from an empty
database, and it cannot drift out of sync with the schema it describes. The task
calls this ``schema_version``; :func:`current_version` and
:func:`schema_version_row` expose it under that name, and a one-row
``schema_version`` table mirrors it for anyone inspecting the file by hand.

Adding a migration means appending to :data:`MIGRATIONS` — never editing a
released one, since databases in the field have already run it. Downgrades are
not supported; :func:`apply_migrations` refuses to touch a database written by a
newer Ayris rather than guessing.

Conventions in the DDL below:

* ``ON DELETE CASCADE`` throughout, so deleting a profile takes its commands,
  folders, variables and timers with it, and deleting a command takes its
  triggers and versions. This relies on ``PRAGMA foreign_keys = ON``, which
  :mod:`ayris.core.database` sets on every connection.
* ``history.matched_command_id`` is ``ON DELETE SET NULL`` instead: the record
  that a phrase was heard stays true after the command it matched is gone.
* Timestamps are ISO-8601 UTC text, which sorts chronologically as text.
* Columns holding JSON keep the ``_json`` suffix from the task;
  :mod:`ayris.core.models` decodes them at the boundary.
* ``command_folders.sort_order`` is not called ``order``: that is a reserved
  keyword and would need quoting in every statement touching it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayris.core.errors import DatabaseError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.database import Database

__all__ = [
    "MIGRATIONS",
    "SCHEMA_VERSION",
    "Migration",
    "apply_migrations",
    "current_version",
    "schema_version_row",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Migration:
    """One irreversible step forward in the schema.

    Args:
        version: Target version after this migration runs.
        description: Short Russian summary for the log.
        statements: DDL/DML executed in order.
        callback: Optional Python step for data moves SQL cannot express.
    """

    version: int
    description: str
    statements: Sequence[str] = ()
    callback: Callable[[Database], None] | None = None


_INITIAL_SCHEMA: Final = (
    # ------------------------------------------------------------------
    # schema_version — mirror of PRAGMA user_version, for humans
    # ------------------------------------------------------------------
    """
    CREATE TABLE schema_version (
        id          INTEGER PRIMARY KEY CHECK (id = 1),
        version     INTEGER NOT NULL,
        applied_at  TEXT    NOT NULL
    )
    """,
    # ------------------------------------------------------------------
    # profiles
    # ------------------------------------------------------------------
    """
    CREATE TABLE profiles (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        name        TEXT    NOT NULL UNIQUE,
        created_at  TEXT    NOT NULL,
        is_active   INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
    )
    """,
    # Partial unique index: any number of inactive profiles, at most one
    # active. Enforced by SQLite rather than by Python so a crash mid-switch
    # cannot leave two active profiles behind.
    "CREATE UNIQUE INDEX idx_profiles_active ON profiles (is_active) WHERE is_active = 1",
    # ------------------------------------------------------------------
    # command_folders — category tree
    # ------------------------------------------------------------------
    """
    CREATE TABLE command_folders (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id  INTEGER REFERENCES profiles (id) ON DELETE CASCADE,
        parent_id   INTEGER REFERENCES command_folders (id) ON DELETE CASCADE,
        name        TEXT    NOT NULL,
        sort_order  INTEGER NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX idx_folders_parent ON command_folders (parent_id, sort_order)",
    "CREATE INDEX idx_folders_profile ON command_folders (profile_id)",
    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    """
    CREATE TABLE commands (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id    INTEGER NOT NULL REFERENCES profiles (id) ON DELETE CASCADE,
        folder_id     INTEGER REFERENCES command_folders (id) ON DELETE SET NULL,
        name          TEXT    NOT NULL,
        description   TEXT    NOT NULL DEFAULT '',
        tags          TEXT    NOT NULL DEFAULT '[]',
        enabled       INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
        priority      INTEGER NOT NULL DEFAULT 0,
        cooldown_ms   INTEGER NOT NULL DEFAULT 0 CHECK (cooldown_ms >= 0),
        require_admin INTEGER NOT NULL DEFAULT 0 CHECK (require_admin IN (0, 1)),
        actions_json  TEXT    NOT NULL DEFAULT '[]',
        created_at    TEXT    NOT NULL,
        updated_at    TEXT    NOT NULL,
        UNIQUE (profile_id, name)
    )
    """,
    # The matcher loads the enabled commands of the active profile on every
    # utterance; this index answers that without touching the table.
    "CREATE INDEX idx_commands_profile ON commands (profile_id, enabled)",
    "CREATE INDEX idx_commands_folder ON commands (folder_id)",
    # ------------------------------------------------------------------
    # command_versions — undo/redo and export
    # ------------------------------------------------------------------
    """
    CREATE TABLE command_versions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        command_id    INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
        version       INTEGER NOT NULL,
        snapshot_json TEXT    NOT NULL,
        comment       TEXT    NOT NULL DEFAULT '',
        created_at    TEXT    NOT NULL,
        UNIQUE (command_id, version)
    )
    """,
    "CREATE INDEX idx_versions_command ON command_versions (command_id, version DESC)",
    # ------------------------------------------------------------------
    # triggers
    # ------------------------------------------------------------------
    # Quoted: TRIGGER is a keyword. Unquoted "triggers" happens to parse, but
    # confuses editors and schema tools.
    """
    CREATE TABLE "triggers" (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        command_id   INTEGER NOT NULL REFERENCES commands (id) ON DELETE CASCADE,
        type         TEXT    NOT NULL DEFAULT 'voice'
                     CHECK (type IN ('voice', 'hotkey', 'event', 'timer')),
        payload_json TEXT    NOT NULL DEFAULT '{}',
        fuzzy        INTEGER NOT NULL DEFAULT 1 CHECK (fuzzy IN (0, 1)),
        priority     INTEGER NOT NULL DEFAULT 0
    )
    """,
    # "Triggers by profile" from the task is a join: triggers -> commands ->
    # profile. This index is the second half of that plan, idx_commands_profile
    # the first; together they answer it without scanning either table.
    'CREATE INDEX idx_triggers_command ON "triggers" (command_id)',
    'CREATE INDEX idx_triggers_type ON "triggers" (type, priority DESC)',
    # ------------------------------------------------------------------
    # variables
    # ------------------------------------------------------------------
    """
    CREATE TABLE variables (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        scope       TEXT    NOT NULL DEFAULT 'global'
                    CHECK (scope IN ('local', 'profile', 'global')),
        profile_id  INTEGER REFERENCES profiles (id) ON DELETE CASCADE,
        name        TEXT    NOT NULL,
        type        TEXT    NOT NULL DEFAULT 'string'
                    CHECK (type IN ('string', 'int', 'float', 'bool', 'array', 'dict')),
        value_json  TEXT    NOT NULL DEFAULT 'null',
        persistent  INTEGER NOT NULL DEFAULT 1 CHECK (persistent IN (0, 1))
    )
    """,
    # A global variable has no profile, so (profile_id, name) would allow
    # duplicate globals: NULL is never equal to NULL in a UNIQUE index.
    # Coalescing to 0 gives global variables one shared namespace.
    """
    CREATE UNIQUE INDEX idx_variables_name
        ON variables (scope, COALESCE(profile_id, 0), name)
    """,
    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------
    """
    CREATE TABLE history (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        ts                 TEXT    NOT NULL,
        stt_raw            TEXT    NOT NULL DEFAULT '',
        matched_command_id INTEGER REFERENCES commands (id) ON DELETE SET NULL,
        intent             TEXT    NOT NULL DEFAULT '',
        params_json        TEXT    NOT NULL DEFAULT '{}',
        result             TEXT    NOT NULL DEFAULT 'ok',
        error              TEXT    NOT NULL DEFAULT '',
        duration_ms        INTEGER NOT NULL DEFAULT 0
    )
    """,
    # Descending: history is read newest-first, so this serves both the recent-N
    # query and the "older than N days" cleanup as an index scan with no sort.
    "CREATE INDEX idx_history_ts ON history (ts DESC)",
    "CREATE INDEX idx_history_command ON history (matched_command_id)",
    # ------------------------------------------------------------------
    # audit
    # ------------------------------------------------------------------
    """
    CREATE TABLE audit (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        ts            TEXT    NOT NULL,
        command_name  TEXT    NOT NULL,
        params_json   TEXT    NOT NULL DEFAULT '{}',
        result        TEXT    NOT NULL DEFAULT 'ok',
        require_admin INTEGER NOT NULL DEFAULT 0 CHECK (require_admin IN (0, 1)),
        elevated      INTEGER NOT NULL DEFAULT 0 CHECK (elevated IN (0, 1))
    )
    """,
    "CREATE INDEX idx_audit_ts ON audit (ts DESC)",
    # Section 11 asks "what ran with elevation"; a partial index keeps it small.
    "CREATE INDEX idx_audit_elevated ON audit (ts DESC) WHERE elevated = 1",
    # ------------------------------------------------------------------
    # timers
    # ------------------------------------------------------------------
    """
    CREATE TABLE timers (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        kind         TEXT    NOT NULL DEFAULT 'timer'
                     CHECK (kind IN ('timer', 'reminder', 'alarm')),
        label        TEXT    NOT NULL DEFAULT '',
        fire_at      TEXT,
        cron         TEXT    NOT NULL DEFAULT '',
        enabled      INTEGER NOT NULL DEFAULT 1 CHECK (enabled IN (0, 1)),
        sound        TEXT    NOT NULL DEFAULT '',
        payload_json TEXT    NOT NULL DEFAULT '{}',
        -- One-shot entries carry fire_at, recurring ones a cron expression.
        -- An entry with neither would never fire and can only be a bug.
        CHECK (fire_at IS NOT NULL OR cron <> '')
    )
    """,
    # The scheduler asks "what is due" on a tick; enabled first so the index
    # covers the filter, fire_at second so the answer comes out ordered.
    "CREATE INDEX idx_timers_due ON timers (enabled, fire_at)",
    # ------------------------------------------------------------------
    # clipboard_history
    # ------------------------------------------------------------------
    """
    CREATE TABLE clipboard_history (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        ts       TEXT    NOT NULL,
        content  TEXT    NOT NULL,
        pinned   INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0, 1))
    )
    """,
    "CREATE INDEX idx_clipboard_ts ON clipboard_history (ts DESC)",
    # ------------------------------------------------------------------
    # models
    # ------------------------------------------------------------------
    """
    CREATE TABLE models (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        kind         TEXT    NOT NULL CHECK (kind IN ('stt', 'tts', 'wake', 'llm')),
        name         TEXT    NOT NULL,
        version      TEXT    NOT NULL DEFAULT '',
        path         TEXT    NOT NULL DEFAULT '',
        sha256       TEXT    NOT NULL DEFAULT '',
        installed_at TEXT,
        is_active    INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1)),
        UNIQUE (kind, name, version)
    )
    """,
    # At most one active model per kind, same trick as profiles.
    "CREATE UNIQUE INDEX idx_models_active ON models (kind) WHERE is_active = 1",
)


#: Every migration ever released, in order. Append only.
MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(
        version=1,
        description="исходная схема",
        statements=_INITIAL_SCHEMA,
    ),
)

#: Schema version this build expects.
SCHEMA_VERSION: Final = MIGRATIONS[-1].version


def current_version(database: Database) -> int:
    """Return the schema version stored in the file. ``0`` means empty."""
    return int(database.query_value("PRAGMA user_version", default=0))


def schema_version_row(database: Database) -> tuple[int, str] | None:
    """Return ``(version, applied_at)`` from the ``schema_version`` table.

    ``None`` before the first migration has created the table.
    """
    row = database.query_one("SELECT version, applied_at FROM schema_version WHERE id = 1")
    if row is None:
        return None
    return int(row["version"]), str(row["applied_at"])


def apply_migrations(
    database: Database,
    *,
    target: int | None = None,
    migrations: Sequence[Migration] = MIGRATIONS,
) -> int:
    """Bring the schema up to ``target``, running only what is missing.

    Idempotent: with nothing pending it makes no write at all, so a normal
    startup costs one ``PRAGMA user_version`` read.

    Args:
        database: Database to migrate.
        target: Stop at this version. Defaults to the newest available.
        migrations: Sequence to apply. Overridden only by the tests that
            exercise the engine itself.

    Returns:
        The version in effect afterwards.

    Raises:
        DatabaseError: If the file was written by a newer Ayris, or a migration
            failed. A failed migration is rolled back, leaving the previous
            version intact and the database usable by the older build.
    """
    newest = migrations[-1].version if migrations else 0
    goal = newest if target is None else target
    version = current_version(database)

    if version > newest:
        raise DatabaseError(
            f"database schema v{version} is newer than supported v{newest}",
            user_message=(
                "База данных создана более новой версией Ayris "
                f"(схема v{version}, поддерживается v{newest}).\n"
                "Обновите приложение, иначе данные могут быть повреждены."
            ),
            recoverable=False,
        )
    if version >= goal:
        return version

    pending = [m for m in migrations if version < m.version <= goal]
    if not pending:
        return version

    _log.info("применение миграций: v%d → v%d (%d шт.)", version, goal, len(pending))
    for migration in pending:
        _apply_one(database, migration)
        version = migration.version

    return version


def _apply_one(database: Database, migration: Migration) -> None:
    """Run one migration atomically, version bump included."""
    from ayris.core.models import to_db_timestamp, utc_now

    try:
        with database.transaction():
            for statement in migration.statements:
                database.execute(statement)
            if migration.callback is not None:
                migration.callback(database)

            # PRAGMA takes no bound parameters; the value is an int from our own
            # migration list, so the interpolation cannot carry user input.
            database.execute(f"PRAGMA user_version = {int(migration.version)}")
            database.execute(
                """
                INSERT INTO schema_version (id, version, applied_at) VALUES (1, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET version = excluded.version,
                                                   applied_at = excluded.applied_at
                """,
                (migration.version, to_db_timestamp(utc_now())),
            )
    except DatabaseError as exc:
        raise DatabaseError(
            f"migration v{migration.version} ({migration.description}) failed: {exc}",
            user_message=(
                f"Не удалось обновить базу данных до версии {migration.version}.\n"
                "Изменения отменены, данные не повреждены."
            ),
            recoverable=False,
        ) from exc

    _log.info("миграция v%d применена: %s", migration.version, migration.description)
