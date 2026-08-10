"""Typed repositories, one per table, plus maintenance helpers.

Each repository takes a :class:`~ayris.core.database.Database` and speaks in the
entities from :mod:`ayris.core.models`. SQL lives here and nowhere else: the GUI,
the macro engine and the NLU matcher never build a statement themselves, which
is what makes a schema change a one-file change.

Grouped access goes through :class:`Repositories`, so a caller holds one object
instead of ten::

    repos = Repositories(database)
    command = repos.commands.create(Command(name="Свет", profile_id=profile.id))
    repos.triggers.add(Trigger(command_id=command.id, payload={"phrase": "свет"}))

Write methods that create a row return the entity with ``id`` and timestamps
filled in, so the caller never has to re-read what it just wrote.

Multi-statement writes are wrapped in :meth:`Database.transaction`. Because that
context manager nests through SAVEPOINTs, a caller may wrap several repository
calls in a transaction of its own and still get all-or-nothing behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.core.errors import DatabaseError
from ayris.core.models import (
    AuditEntry,
    ClipboardEntry,
    Command,
    CommandFolder,
    CommandVersion,
    ExecutionResult,
    HistoryEntry,
    JsonObject,
    ModelRecord,
    Profile,
    Timer,
    TimerKind,
    Trigger,
    TriggerType,
    Variable,
    VariableScope,
    VariableType,
    dump_json,
    to_db_timestamp,
    utc_now,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.database import Database
    from ayris.core.paths import ModelKind

__all__ = [
    "AuditRepository",
    "CleanupCategory",
    "ClipboardRepository",
    "CommandRepository",
    "FolderRepository",
    "HistoryRepository",
    "MaintenanceRepository",
    "ModelRepository",
    "ProfileRepository",
    "Repositories",
    "TimerRepository",
    "TriggerRepository",
    "VariableRepository",
]

_log = get_logger(__name__)


class _Repository:
    """Common base: holds the database handle."""

    __slots__ = ("_db",)

    def __init__(self, database: Database) -> None:
        self._db = database

    @property
    def database(self) -> Database:
        return self._db


def _require_id(value: int | None, entity: str) -> int:
    """Reject an entity that has not been persisted yet.

    Catches the "insert a child before its parent" mistake at the call site
    instead of letting SQLite report a foreign key violation on ``NULL``.
    """
    if value is None:
        raise DatabaseError(
            f"{entity} has no id; save it before referencing it",
            user_message="Внутренняя ошибка: объект ещё не сохранён в базе данных.",
        )
    return value


# ----------------------------------------------------------------------
# profiles
# ----------------------------------------------------------------------


class ProfileRepository(_Repository):
    """Profiles: named sets of commands, folders and variables."""

    _COLUMNS: Final = "id, name, created_at, is_active"

    def create(self, name: str, *, activate: bool = False) -> Profile:
        """Create a profile. ``activate`` makes it the active one atomically."""
        now = utc_now()
        with self._db.transaction():
            if activate:
                self._deactivate_all()
            profile_id = self._db.insert(
                "INSERT INTO profiles (name, created_at, is_active) VALUES (?, ?, ?)",
                (name, to_db_timestamp(now), int(activate)),
            )
        return Profile(id=profile_id, name=name, created_at=now, is_active=activate)

    def get(self, profile_id: int) -> Profile | None:
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM profiles WHERE id = ?", (profile_id,)
        )
        return Profile.from_row(row) if row is not None else None

    def get_by_name(self, name: str) -> Profile | None:
        row = self._db.query_one(f"SELECT {self._COLUMNS} FROM profiles WHERE name = ?", (name,))
        return Profile.from_row(row) if row is not None else None

    def list_all(self) -> list[Profile]:
        rows = self._db.query_all(f"SELECT {self._COLUMNS} FROM profiles ORDER BY name")
        return [Profile.from_row(row) for row in rows]

    def active(self) -> Profile | None:
        """The currently active profile, if one has been chosen."""
        row = self._db.query_one(f"SELECT {self._COLUMNS} FROM profiles WHERE is_active = 1")
        return Profile.from_row(row) if row is not None else None

    def set_active(self, profile_id: int) -> None:
        """Switch the active profile.

        Deactivating first is required, not cosmetic: a partial unique index
        allows only one active row, so setting the new one first would collide.
        """
        with self._db.transaction():
            self._deactivate_all()
            self._db.execute("UPDATE profiles SET is_active = 1 WHERE id = ?", (profile_id,))

    def rename(self, profile_id: int, name: str) -> None:
        self._db.execute("UPDATE profiles SET name = ? WHERE id = ?", (name, profile_id))

    def delete(self, profile_id: int) -> bool:
        """Delete a profile and, by cascade, everything inside it."""
        cursor = self._db.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
        return cursor.rowcount > 0

    def count(self) -> int:
        return int(self._db.query_value("SELECT COUNT(*) FROM profiles", default=0))

    def _deactivate_all(self) -> None:
        self._db.execute("UPDATE profiles SET is_active = 0 WHERE is_active = 1")


# ----------------------------------------------------------------------
# command_folders
# ----------------------------------------------------------------------


class FolderRepository(_Repository):
    """The command library tree."""

    _COLUMNS: Final = "id, profile_id, parent_id, name, sort_order"

    def create(self, folder: CommandFolder) -> CommandFolder:
        folder_id = self._db.insert(
            """
            INSERT INTO command_folders (profile_id, parent_id, name, sort_order)
            VALUES (?, ?, ?, ?)
            """,
            (folder.profile_id, folder.parent_id, folder.name, folder.sort_order),
        )
        return replace(folder, id=folder_id)

    def get(self, folder_id: int) -> CommandFolder | None:
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM command_folders WHERE id = ?", (folder_id,)
        )
        return CommandFolder.from_row(row) if row is not None else None

    def list_for_profile(self, profile_id: int | None) -> list[CommandFolder]:
        """Folders of one profile plus the shared ones (``profile_id IS NULL``)."""
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM command_folders
            WHERE profile_id IS ? OR profile_id IS NULL
            ORDER BY sort_order, name
            """,
            (profile_id,),
        )
        return [CommandFolder.from_row(row) for row in rows]

    def children(self, parent_id: int | None) -> list[CommandFolder]:
        """Direct children of a node. ``None`` returns the roots."""
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM command_folders
            WHERE parent_id IS ? ORDER BY sort_order, name
            """,
            (parent_id,),
        )
        return [CommandFolder.from_row(row) for row in rows]

    def update(self, folder: CommandFolder) -> None:
        folder_id = _require_id(folder.id, "folder")
        if folder.parent_id == folder_id:
            raise DatabaseError(
                f"folder {folder_id} cannot be its own parent",
                user_message="Папка не может находиться внутри самой себя.",
            )
        self._db.execute(
            """
            UPDATE command_folders
               SET profile_id = ?, parent_id = ?, name = ?, sort_order = ?
             WHERE id = ?
            """,
            (folder.profile_id, folder.parent_id, folder.name, folder.sort_order, folder_id),
        )

    def rename(self, folder_id: int, name: str) -> None:
        self._db.execute("UPDATE command_folders SET name = ? WHERE id = ?", (name, folder_id))

    def reorder(self, ordered_ids: Sequence[int]) -> None:
        """Renumber ``sort_order`` to match the given order. Used by drag-and-drop."""
        with self._db.transaction():
            self._db.executemany(
                "UPDATE command_folders SET sort_order = ? WHERE id = ?",
                [(position, folder_id) for position, folder_id in enumerate(ordered_ids)],
            )

    def delete(self, folder_id: int) -> bool:
        """Delete a folder and its subtree.

        Commands inside are kept: ``commands.folder_id`` is ``ON DELETE SET
        NULL``, so they move to the root rather than disappearing with the
        folder the user dragged to the bin.
        """
        cursor = self._db.execute("DELETE FROM command_folders WHERE id = ?", (folder_id,))
        return cursor.rowcount > 0


# ----------------------------------------------------------------------
# commands and their versions
# ----------------------------------------------------------------------


class CommandRepository(_Repository):
    """Commands, with the version history used for undo/redo and export."""

    _COLUMNS: Final = (
        "id, profile_id, folder_id, name, description, tags, enabled, priority, "
        "cooldown_ms, require_admin, actions_json, created_at, updated_at"
    )

    def create(self, command: Command) -> Command:
        now = utc_now()
        stamp = to_db_timestamp(now)
        command_id = self._db.insert(
            """
            INSERT INTO commands (
                profile_id, folder_id, name, description, tags, enabled, priority,
                cooldown_ms, require_admin, actions_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command.profile_id,
                command.folder_id,
                command.name,
                command.description,
                dump_json(list(command.tags)),
                int(command.enabled),
                command.priority,
                command.cooldown_ms,
                int(command.require_admin),
                dump_json(list(command.actions)),
                stamp,
                stamp,
            ),
        )
        return replace(command, id=command_id, created_at=now, updated_at=now)

    def get(self, command_id: int) -> Command | None:
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM commands WHERE id = ?", (command_id,)
        )
        return Command.from_row(row) if row is not None else None

    def get_by_name(self, profile_id: int, name: str) -> Command | None:
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM commands WHERE profile_id = ? AND name = ?",
            (profile_id, name),
        )
        return Command.from_row(row) if row is not None else None

    def list_for_profile(
        self,
        profile_id: int,
        *,
        enabled_only: bool = False,
        folder_id: int | None = None,
    ) -> list[Command]:
        """Commands of a profile, highest priority first.

        Args:
            enabled_only: Skip disabled commands. The matcher sets this.
            folder_id: Restrict to one folder. ``None`` means every folder.
        """
        sql = [f"SELECT {self._COLUMNS} FROM commands WHERE profile_id = ?"]
        params: list[object] = [profile_id]
        if enabled_only:
            sql.append("AND enabled = 1")
        if folder_id is not None:
            sql.append("AND folder_id = ?")
            params.append(folder_id)
        sql.append("ORDER BY priority DESC, name")
        rows = self._db.query_all(" ".join(sql), params)
        return [Command.from_row(row) for row in rows]

    def search(self, profile_id: int, text: str) -> list[Command]:
        """Case-insensitive search over name and description for the library tab.

        Uses ``ulower`` rather than ``LIKE``: SQLite's own case folding is
        ASCII-only, so a plain ``LIKE`` would not match "свет" against "Свет".
        """
        pattern = f"%{text.lower()}%"
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM commands
             WHERE profile_id = ?
               AND (ulower(name) LIKE ? OR ulower(description) LIKE ?)
             ORDER BY priority DESC, name
            """,
            (profile_id, pattern, pattern),
        )
        return [Command.from_row(row) for row in rows]

    def update(self, command: Command, *, save_version: bool = False, comment: str = "") -> Command:
        """Persist a changed command and refresh ``updated_at``.

        Args:
            save_version: Snapshot the *previous* state into ``command_versions``
                first, so the editor can undo back to it.
        """
        command_id = _require_id(command.id, "command")
        now = utc_now()
        with self._db.transaction():
            if save_version:
                previous = self.get(command_id)
                if previous is not None:
                    self.save_version(previous, comment=comment)
            self._db.execute(
                """
                UPDATE commands
                   SET folder_id = ?, name = ?, description = ?, tags = ?, enabled = ?,
                       priority = ?, cooldown_ms = ?, require_admin = ?, actions_json = ?,
                       updated_at = ?
                 WHERE id = ?
                """,
                (
                    command.folder_id,
                    command.name,
                    command.description,
                    dump_json(list(command.tags)),
                    int(command.enabled),
                    command.priority,
                    command.cooldown_ms,
                    int(command.require_admin),
                    dump_json(list(command.actions)),
                    to_db_timestamp(now),
                    command_id,
                ),
            )
        return replace(command, updated_at=now)

    def set_enabled(self, command_id: int, *, enabled: bool) -> None:
        self._db.execute(
            "UPDATE commands SET enabled = ?, updated_at = ? WHERE id = ?",
            (int(enabled), to_db_timestamp(utc_now()), command_id),
        )

    def move_to_folder(self, command_id: int, folder_id: int | None) -> None:
        self._db.execute(
            "UPDATE commands SET folder_id = ?, updated_at = ? WHERE id = ?",
            (folder_id, to_db_timestamp(utc_now()), command_id),
        )

    def delete(self, command_id: int) -> bool:
        """Delete a command; triggers and versions go with it by cascade."""
        cursor = self._db.execute("DELETE FROM commands WHERE id = ?", (command_id,))
        return cursor.rowcount > 0

    def count(self, profile_id: int | None = None) -> int:
        if profile_id is None:
            return int(self._db.query_value("SELECT COUNT(*) FROM commands", default=0))
        return int(
            self._db.query_value(
                "SELECT COUNT(*) FROM commands WHERE profile_id = ?", (profile_id,), default=0
            )
        )

    # --- versions -----------------------------------------------------

    def save_version(self, command: Command, *, comment: str = "") -> CommandVersion:
        """Snapshot a command. The version number increments per command."""
        command_id = _require_id(command.id, "command")
        snapshot: JsonObject = {
            "name": command.name,
            "description": command.description,
            "tags": list(command.tags),
            "enabled": command.enabled,
            "priority": command.priority,
            "cooldown_ms": command.cooldown_ms,
            "require_admin": command.require_admin,
            "actions": list(command.actions),
            "folder_id": command.folder_id,
        }
        now = utc_now()
        with self._db.transaction():
            next_version = (
                int(
                    self._db.query_value(
                        "SELECT MAX(version) FROM command_versions WHERE command_id = ?",
                        (command_id,),
                        default=0,
                    )
                )
                + 1
            )
            version_id = self._db.insert(
                """
                INSERT INTO command_versions
                    (command_id, version, snapshot_json, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (command_id, next_version, dump_json(snapshot), comment, to_db_timestamp(now)),
            )
        return CommandVersion(
            id=version_id,
            command_id=command_id,
            version=next_version,
            snapshot=snapshot,
            comment=comment,
            created_at=now,
        )

    def list_versions(self, command_id: int, *, limit: int = 50) -> list[CommandVersion]:
        rows = self._db.query_all(
            """
            SELECT id, command_id, version, snapshot_json, comment, created_at
              FROM command_versions WHERE command_id = ?
             ORDER BY version DESC LIMIT ?
            """,
            (command_id, limit),
        )
        return [CommandVersion.from_row(row) for row in rows]

    def get_version(self, command_id: int, version: int) -> CommandVersion | None:
        row = self._db.query_one(
            """
            SELECT id, command_id, version, snapshot_json, comment, created_at
              FROM command_versions WHERE command_id = ? AND version = ?
            """,
            (command_id, version),
        )
        return CommandVersion.from_row(row) if row is not None else None

    def restore_version(self, command_id: int, version: int) -> Command:
        """Roll a command back to a snapshot.

        The current state is snapshotted first, so an undo can itself be undone.

        Raises:
            DatabaseError: The command or the version does not exist.
        """
        with self._db.transaction():
            current = self.get(command_id)
            if current is None:
                raise DatabaseError(
                    f"command {command_id} not found",
                    user_message="Команда не найдена — возможно, она уже удалена.",
                )
            snapshot_version = self.get_version(command_id, version)
            if snapshot_version is None:
                raise DatabaseError(
                    f"command {command_id} has no version {version}",
                    user_message=f"Версия {version} этой команды не найдена.",
                )

            data = snapshot_version.snapshot
            restored = replace(
                current,
                name=str(data.get("name", current.name)),
                description=str(data.get("description", current.description)),
                tags=tuple(str(tag) for tag in data.get("tags", [])),
                enabled=bool(data.get("enabled", current.enabled)),
                priority=int(data.get("priority", current.priority)),
                cooldown_ms=int(data.get("cooldown_ms", current.cooldown_ms)),
                require_admin=bool(data.get("require_admin", current.require_admin)),
                actions=tuple(item for item in data.get("actions", []) if isinstance(item, dict)),
                folder_id=data.get("folder_id", current.folder_id),
            )
            return self.update(restored, save_version=True, comment=f"откат к версии {version}")

    def prune_versions(self, command_id: int, *, keep: int = 20) -> int:
        """Drop all but the newest ``keep`` versions. Returns rows deleted."""
        cursor = self._db.execute(
            """
            DELETE FROM command_versions
             WHERE command_id = ? AND version NOT IN (
                   SELECT version FROM command_versions
                    WHERE command_id = ? ORDER BY version DESC LIMIT ?
             )
            """,
            (command_id, command_id, keep),
        )
        return cursor.rowcount


# ----------------------------------------------------------------------
# triggers
# ----------------------------------------------------------------------


class TriggerRepository(_Repository):
    """Voice phrases, hotkeys, events and schedules that fire commands."""

    _COLUMNS: Final = "id, command_id, type, payload_json, fuzzy, priority"

    def add(self, trigger: Trigger) -> Trigger:
        trigger_id = self._db.insert(
            """
            INSERT INTO "triggers" (command_id, type, payload_json, fuzzy, priority)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                trigger.command_id,
                str(trigger.type),
                dump_json(trigger.payload),
                int(trigger.fuzzy),
                trigger.priority,
            ),
        )
        return replace(trigger, id=trigger_id)

    def add_voice(self, command_id: int, phrase: str, *, fuzzy: bool = True) -> Trigger:
        """Shorthand for the most common trigger: a spoken phrase."""
        return self.add(
            Trigger(
                command_id=command_id,
                type=TriggerType.VOICE,
                payload={"phrase": phrase},
                fuzzy=fuzzy,
            )
        )

    def get(self, trigger_id: int) -> Trigger | None:
        row = self._db.query_one(
            f'SELECT {self._COLUMNS} FROM "triggers" WHERE id = ?', (trigger_id,)
        )
        return Trigger.from_row(row) if row is not None else None

    def list_for_command(self, command_id: int) -> list[Trigger]:
        rows = self._db.query_all(
            f'SELECT {self._COLUMNS} FROM "triggers" WHERE command_id = ? ORDER BY priority DESC',
            (command_id,),
        )
        return [Trigger.from_row(row) for row in rows]

    def list_for_profile(
        self,
        profile_id: int,
        *,
        trigger_type: TriggerType | None = None,
        enabled_only: bool = True,
    ) -> list[Trigger]:
        """Every trigger of a profile — the matcher's startup query.

        Served by ``idx_commands_profile`` and ``idx_triggers_command`` together;
        this is the "triggers by profile_id" access path the task asks for.
        """
        sql = [
            "SELECT t.id, t.command_id, t.type, t.payload_json, t.fuzzy, t.priority",
            'FROM "triggers" t JOIN commands c ON c.id = t.command_id',
            "WHERE c.profile_id = ?",
        ]
        params: list[object] = [profile_id]
        if enabled_only:
            sql.append("AND c.enabled = 1")
        if trigger_type is not None:
            sql.append("AND t.type = ?")
            params.append(str(trigger_type))
        sql.append("ORDER BY t.priority DESC, c.priority DESC")
        rows = self._db.query_all(" ".join(sql), params)
        return [Trigger.from_row(row) for row in rows]

    def update(self, trigger: Trigger) -> None:
        trigger_id = _require_id(trigger.id, "trigger")
        self._db.execute(
            """
            UPDATE "triggers"
               SET command_id = ?, type = ?, payload_json = ?, fuzzy = ?, priority = ?
             WHERE id = ?
            """,
            (
                trigger.command_id,
                str(trigger.type),
                dump_json(trigger.payload),
                int(trigger.fuzzy),
                trigger.priority,
                trigger_id,
            ),
        )

    def delete(self, trigger_id: int) -> bool:
        cursor = self._db.execute('DELETE FROM "triggers" WHERE id = ?', (trigger_id,))
        return cursor.rowcount > 0

    def delete_for_command(self, command_id: int) -> int:
        """Drop every trigger of a command. Used when the editor replaces them wholesale."""
        cursor = self._db.execute('DELETE FROM "triggers" WHERE command_id = ?', (command_id,))
        return cursor.rowcount

    def replace_for_command(self, command_id: int, triggers: Iterable[Trigger]) -> list[Trigger]:
        """Swap a command's triggers atomically."""
        with self._db.transaction():
            self.delete_for_command(command_id)
            return [self.add(replace(trigger, command_id=command_id)) for trigger in triggers]


# ----------------------------------------------------------------------
# variables
# ----------------------------------------------------------------------


class VariableRepository(_Repository):
    """Macro variables in local, profile and global scope."""

    _COLUMNS: Final = "id, scope, profile_id, name, type, value_json, persistent"

    def set(
        self,
        name: str,
        value: object,
        *,
        scope: VariableScope = VariableScope.GLOBAL,
        profile_id: int | None = None,
        var_type: VariableType | None = None,
        persistent: bool = True,
    ) -> Variable:
        """Create or overwrite a variable.

        One upsert rather than select-then-write: two macros setting the same
        variable concurrently would otherwise race between the two statements.
        """
        resolved_type = var_type if var_type is not None else _infer_type(value)
        self._db.execute(
            """
            INSERT INTO variables (scope, profile_id, name, type, value_json, persistent)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (scope, COALESCE(profile_id, 0), name) DO UPDATE
                SET type = excluded.type,
                    value_json = excluded.value_json,
                    persistent = excluded.persistent
            """,
            (
                str(scope),
                profile_id,
                name,
                str(resolved_type),
                dump_json(value),
                int(persistent),
            ),
        )
        stored = self.get(name, scope=scope, profile_id=profile_id)
        if stored is None:  # pragma: no cover - the upsert just wrote this row
            raise DatabaseError(
                f"variable {name!r} vanished after upsert",
                user_message="Не удалось сохранить переменную.",
            )
        return stored

    def get(
        self,
        name: str,
        *,
        scope: VariableScope = VariableScope.GLOBAL,
        profile_id: int | None = None,
    ) -> Variable | None:
        row = self._db.query_one(
            f"""
            SELECT {self._COLUMNS} FROM variables
             WHERE scope = ? AND name = ? AND COALESCE(profile_id, 0) = COALESCE(?, 0)
            """,
            (str(scope), name, profile_id),
        )
        return Variable.from_row(row) if row is not None else None

    def get_value(
        self,
        name: str,
        default: object = None,
        *,
        scope: VariableScope = VariableScope.GLOBAL,
        profile_id: int | None = None,
    ) -> object:
        """Value of a variable, or ``default`` when it is not set."""
        variable = self.get(name, scope=scope, profile_id=profile_id)
        return default if variable is None else variable.value

    def list_all(
        self,
        *,
        scope: VariableScope | None = None,
        profile_id: int | None = None,
    ) -> list[Variable]:
        sql = [f"SELECT {self._COLUMNS} FROM variables WHERE 1 = 1"]
        params: list[object] = []
        if scope is not None:
            sql.append("AND scope = ?")
            params.append(str(scope))
        if profile_id is not None:
            sql.append("AND profile_id = ?")
            params.append(profile_id)
        sql.append("ORDER BY scope, name")
        rows = self._db.query_all(" ".join(sql), params)
        return [Variable.from_row(row) for row in rows]

    def delete(
        self,
        name: str,
        *,
        scope: VariableScope = VariableScope.GLOBAL,
        profile_id: int | None = None,
    ) -> bool:
        cursor = self._db.execute(
            """
            DELETE FROM variables
             WHERE scope = ? AND name = ? AND COALESCE(profile_id, 0) = COALESCE(?, 0)
            """,
            (str(scope), name, profile_id),
        )
        return cursor.rowcount > 0

    def clear_transient(self) -> int:
        """Drop non-persistent variables. Called at startup: they died with the process."""
        cursor = self._db.execute("DELETE FROM variables WHERE persistent = 0")
        return cursor.rowcount


def _infer_type(value: object) -> VariableType:
    """Map a Python value onto a declared variable type.

    ``bool`` is checked before ``int`` because ``bool`` is a subclass of ``int``
    and would otherwise be stored as one.
    """
    if isinstance(value, bool):
        return VariableType.BOOL
    if isinstance(value, int):
        return VariableType.INT
    if isinstance(value, float):
        return VariableType.FLOAT
    if isinstance(value, list | tuple):
        return VariableType.ARRAY
    if isinstance(value, dict):
        return VariableType.DICT
    return VariableType.STRING


# ----------------------------------------------------------------------
# history
# ----------------------------------------------------------------------


class HistoryRepository(_Repository):
    """Recognised phrases and what came of them."""

    _COLUMNS: Final = (
        "id, ts, stt_raw, matched_command_id, intent, params_json, result, error, duration_ms"
    )

    def add(self, entry: HistoryEntry) -> HistoryEntry:
        ts = entry.ts if entry.ts is not None else utc_now()
        entry_id = self._db.insert(
            """
            INSERT INTO history (
                ts, stt_raw, matched_command_id, intent, params_json, result, error, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                to_db_timestamp(ts),
                entry.stt_raw,
                entry.matched_command_id,
                entry.intent,
                dump_json(entry.params),
                str(entry.result),
                entry.error,
                entry.duration_ms,
            ),
        )
        return replace(entry, id=entry_id, ts=ts)

    def get(self, entry_id: int) -> HistoryEntry | None:
        row = self._db.query_one(f"SELECT {self._COLUMNS} FROM history WHERE id = ?", (entry_id,))
        return HistoryEntry.from_row(row) if row is not None else None

    def recent(self, limit: int = 100, *, offset: int = 0) -> list[HistoryEntry]:
        """Newest entries first — what the history tab shows."""
        rows = self._db.query_all(
            f"SELECT {self._COLUMNS} FROM history ORDER BY ts DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [HistoryEntry.from_row(row) for row in rows]

    def list_between(self, start: datetime, end: datetime) -> list[HistoryEntry]:
        """Entries in a time window, oldest first."""
        rows = self._db.query_all(
            f"SELECT {self._COLUMNS} FROM history WHERE ts >= ? AND ts <= ? ORDER BY ts",
            (to_db_timestamp(start), to_db_timestamp(end)),
        )
        return [HistoryEntry.from_row(row) for row in rows]

    def failures(self, limit: int = 100) -> list[HistoryEntry]:
        """Entries that did not succeed, for the diagnostics view."""
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM history
             WHERE result <> ? ORDER BY ts DESC LIMIT ?
            """,
            (str(ExecutionResult.OK), limit),
        )
        return [HistoryEntry.from_row(row) for row in rows]

    def count(self) -> int:
        return int(self._db.query_value("SELECT COUNT(*) FROM history", default=0))

    def delete_older_than(self, days: int) -> int:
        """Delete entries older than ``days``. Returns rows removed."""
        if days <= 0:
            return 0
        cutoff = utc_now() - timedelta(days=days)
        cursor = self._db.execute("DELETE FROM history WHERE ts < ?", (to_db_timestamp(cutoff),))
        removed = cursor.rowcount
        if removed:
            _log.info("удалено записей истории старше %d дн.: %d", days, removed)
        return removed

    def trim_to_limit(self, limit: int) -> int:
        """Keep only the newest ``limit`` entries.

        Backs ``privacy.history_limit``; ``0`` means unlimited and does nothing.
        """
        if limit <= 0:
            return 0
        cursor = self._db.execute(
            """
            DELETE FROM history WHERE id NOT IN (
                SELECT id FROM history ORDER BY ts DESC, id DESC LIMIT ?
            )
            """,
            (limit,),
        )
        return cursor.rowcount

    def clear(self) -> int:
        cursor = self._db.execute("DELETE FROM history")
        return cursor.rowcount


# ----------------------------------------------------------------------
# audit
# ----------------------------------------------------------------------


class AuditRepository(_Repository):
    """Security journal: what ran, with which rights."""

    _COLUMNS: Final = "id, ts, command_name, params_json, result, require_admin, elevated"

    def add(self, entry: AuditEntry) -> AuditEntry:
        ts = entry.ts if entry.ts is not None else utc_now()
        entry_id = self._db.insert(
            """
            INSERT INTO audit (ts, command_name, params_json, result, require_admin, elevated)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                to_db_timestamp(ts),
                entry.command_name,
                dump_json(entry.params),
                str(entry.result),
                int(entry.require_admin),
                int(entry.elevated),
            ),
        )
        return replace(entry, id=entry_id, ts=ts)

    def recent(self, limit: int = 100, *, elevated_only: bool = False) -> list[AuditEntry]:
        sql = [f"SELECT {self._COLUMNS} FROM audit"]
        if elevated_only:
            sql.append("WHERE elevated = 1")
        sql.append("ORDER BY ts DESC, id DESC LIMIT ?")
        rows = self._db.query_all(" ".join(sql), (limit,))
        return [AuditEntry.from_row(row) for row in rows]

    def count(self) -> int:
        return int(self._db.query_value("SELECT COUNT(*) FROM audit", default=0))

    def delete_older_than(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = utc_now() - timedelta(days=days)
        cursor = self._db.execute("DELETE FROM audit WHERE ts < ?", (to_db_timestamp(cutoff),))
        return cursor.rowcount

    def clear(self) -> int:
        cursor = self._db.execute("DELETE FROM audit")
        return cursor.rowcount


# ----------------------------------------------------------------------
# timers
# ----------------------------------------------------------------------


class TimerRepository(_Repository):
    """Timers, reminders and alarms."""

    _COLUMNS: Final = "id, kind, label, fire_at, cron, enabled, sound, payload_json"

    def create(self, timer: Timer) -> Timer:
        if timer.fire_at is None and not timer.cron:
            raise DatabaseError(
                "timer needs either fire_at or cron",
                user_message="Укажите время срабатывания или расписание.",
            )
        timer_id = self._db.insert(
            """
            INSERT INTO timers (kind, label, fire_at, cron, enabled, sound, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(timer.kind),
                timer.label,
                to_db_timestamp(timer.fire_at),
                timer.cron,
                int(timer.enabled),
                timer.sound,
                dump_json(timer.payload),
            ),
        )
        return replace(timer, id=timer_id)

    def get(self, timer_id: int) -> Timer | None:
        row = self._db.query_one(f"SELECT {self._COLUMNS} FROM timers WHERE id = ?", (timer_id,))
        return Timer.from_row(row) if row is not None else None

    def list_all(self, *, kind: TimerKind | None = None, enabled_only: bool = False) -> list[Timer]:
        sql = [f"SELECT {self._COLUMNS} FROM timers WHERE 1 = 1"]
        params: list[object] = []
        if kind is not None:
            sql.append("AND kind = ?")
            params.append(str(kind))
        if enabled_only:
            sql.append("AND enabled = 1")
        sql.append("ORDER BY fire_at IS NULL, fire_at")
        rows = self._db.query_all(" ".join(sql), params)
        return [Timer.from_row(row) for row in rows]

    def due(self, moment: datetime | None = None) -> list[Timer]:
        """Enabled one-shot timers whose moment has passed.

        Recurring entries are excluded: the scheduler expands their cron into a
        concrete ``fire_at`` and they show up here once that time arrives.
        """
        now = moment if moment is not None else utc_now()
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM timers
             WHERE enabled = 1 AND fire_at IS NOT NULL AND fire_at <= ?
             ORDER BY fire_at
            """,
            (to_db_timestamp(now),),
        )
        return [Timer.from_row(row) for row in rows]

    def update(self, timer: Timer) -> None:
        timer_id = _require_id(timer.id, "timer")
        self._db.execute(
            """
            UPDATE timers
               SET kind = ?, label = ?, fire_at = ?, cron = ?, enabled = ?,
                   sound = ?, payload_json = ?
             WHERE id = ?
            """,
            (
                str(timer.kind),
                timer.label,
                to_db_timestamp(timer.fire_at),
                timer.cron,
                int(timer.enabled),
                timer.sound,
                dump_json(timer.payload),
                timer_id,
            ),
        )

    def reschedule(self, timer_id: int, fire_at: datetime) -> None:
        """Point a timer at its next occurrence. Called after a cron entry fires."""
        self._db.execute(
            "UPDATE timers SET fire_at = ? WHERE id = ?", (to_db_timestamp(fire_at), timer_id)
        )

    def set_enabled(self, timer_id: int, *, enabled: bool) -> None:
        self._db.execute("UPDATE timers SET enabled = ? WHERE id = ?", (int(enabled), timer_id))

    def delete(self, timer_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM timers WHERE id = ?", (timer_id,))
        return cursor.rowcount > 0

    def delete_expired(self) -> int:
        """Remove fired one-shot timers. Recurring entries are kept."""
        cursor = self._db.execute(
            """
            DELETE FROM timers
             WHERE enabled = 0 AND cron = '' AND fire_at IS NOT NULL AND fire_at < ?
            """,
            (to_db_timestamp(utc_now()),),
        )
        return cursor.rowcount


# ----------------------------------------------------------------------
# clipboard
# ----------------------------------------------------------------------


class ClipboardRepository(_Repository):
    """Clipboard history for the ClipboardGet/ClipboardSet action blocks."""

    _COLUMNS: Final = "id, ts, content, pinned"

    def add(self, content: str, *, pinned: bool = False) -> ClipboardEntry:
        now = utc_now()
        entry_id = self._db.insert(
            "INSERT INTO clipboard_history (ts, content, pinned) VALUES (?, ?, ?)",
            (to_db_timestamp(now), content, int(pinned)),
        )
        return ClipboardEntry(id=entry_id, ts=now, content=content, pinned=pinned)

    def recent(self, limit: int = 50) -> list[ClipboardEntry]:
        """Pinned entries first, then newest — the order the picker shows."""
        rows = self._db.query_all(
            f"""
            SELECT {self._COLUMNS} FROM clipboard_history
             ORDER BY pinned DESC, ts DESC LIMIT ?
            """,
            (limit,),
        )
        return [ClipboardEntry.from_row(row) for row in rows]

    def set_pinned(self, entry_id: int, *, pinned: bool) -> None:
        self._db.execute(
            "UPDATE clipboard_history SET pinned = ? WHERE id = ?", (int(pinned), entry_id)
        )

    def delete(self, entry_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM clipboard_history WHERE id = ?", (entry_id,))
        return cursor.rowcount > 0

    def trim_to_limit(self, limit: int) -> int:
        """Keep the newest ``limit`` unpinned entries; pinned ones always stay."""
        if limit <= 0:
            return 0
        cursor = self._db.execute(
            """
            DELETE FROM clipboard_history
             WHERE pinned = 0 AND id NOT IN (
                   SELECT id FROM clipboard_history WHERE pinned = 0
                    ORDER BY ts DESC LIMIT ?
             )
            """,
            (limit,),
        )
        return cursor.rowcount

    def clear(self, *, keep_pinned: bool = True) -> int:
        sql = "DELETE FROM clipboard_history"
        if keep_pinned:
            sql += " WHERE pinned = 0"
        cursor = self._db.execute(sql)
        return cursor.rowcount


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------


class ModelRepository(_Repository):
    """Installed STT/TTS/wake-word/LLM models."""

    _COLUMNS: Final = (
        "id, kind, name, engine, version, path, sha256, size_bytes, "
        "catalog_id, installed_at, is_active"
    )

    def add(self, model: ModelRecord) -> ModelRecord:
        installed = model.installed_at if model.installed_at is not None else utc_now()
        with self._db.transaction():
            if model.is_active:
                self._deactivate(model.kind)
            model_id = self._db.insert(
                """
                INSERT INTO models
                    (kind, name, engine, version, path, sha256, size_bytes,
                     catalog_id, installed_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model.kind,
                    model.name,
                    model.engine,
                    model.version,
                    model.path,
                    model.sha256,
                    model.size_bytes,
                    model.catalog_id,
                    to_db_timestamp(installed),
                    int(model.is_active),
                ),
            )
        return replace(model, id=model_id, installed_at=installed)

    def get(self, model_id: int) -> ModelRecord | None:
        row = self._db.query_one(f"SELECT {self._COLUMNS} FROM models WHERE id = ?", (model_id,))
        return ModelRecord.from_row(row) if row is not None else None

    def by_catalog_id(self, catalog_id: str) -> ModelRecord | None:
        """The installed copy of a catalog entry, if there is one.

        How :mod:`ayris.models.registry` answers "is this already installed" for
        the settings list, and how a repeated download becomes an update of the
        existing row instead of a UNIQUE violation.
        """
        if not catalog_id:
            return None
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM models WHERE catalog_id = ?", (catalog_id,)
        )
        return ModelRecord.from_row(row) if row is not None else None

    def find(self, kind: ModelKind, name: str, version: str = "") -> ModelRecord | None:
        """The row matching the UNIQUE key ``(kind, name, version)``."""
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM models WHERE kind = ? AND name = ? AND version = ?",
            (kind, name, version),
        )
        return ModelRecord.from_row(row) if row is not None else None

    def list_by_kind(self, kind: ModelKind) -> list[ModelRecord]:
        rows = self._db.query_all(
            f"SELECT {self._COLUMNS} FROM models WHERE kind = ? ORDER BY name, version", (kind,)
        )
        return [ModelRecord.from_row(row) for row in rows]

    def list_all(self) -> list[ModelRecord]:
        rows = self._db.query_all(f"SELECT {self._COLUMNS} FROM models ORDER BY kind, name")
        return [ModelRecord.from_row(row) for row in rows]

    def active(self, kind: ModelKind) -> ModelRecord | None:
        """The model currently selected for a kind."""
        row = self._db.query_one(
            f"SELECT {self._COLUMNS} FROM models WHERE kind = ? AND is_active = 1", (kind,)
        )
        return ModelRecord.from_row(row) if row is not None else None

    def update(self, model: ModelRecord) -> ModelRecord:
        """Overwrite the mutable columns of an existing row.

        ``is_active`` is deliberately not among them: it is an invariant across
        rows, not a property of one, and :meth:`set_active` is the only thing
        allowed to move it.
        """
        if model.id is None:
            raise DatabaseError(
                "cannot update a model without an id",
                user_message="Модель не найдена.",
            )
        installed = model.installed_at if model.installed_at is not None else utc_now()
        cursor = self._db.execute(
            """
            UPDATE models
               SET name = ?, engine = ?, version = ?, path = ?, sha256 = ?,
                   size_bytes = ?, catalog_id = ?, installed_at = ?
             WHERE id = ?
            """,
            (
                model.name,
                model.engine,
                model.version,
                model.path,
                model.sha256,
                model.size_bytes,
                model.catalog_id,
                to_db_timestamp(installed),
                model.id,
            ),
        )
        if cursor.rowcount == 0:
            raise DatabaseError(
                f"model {model.id} not found",
                user_message="Модель не найдена.",
            )
        return replace(model, installed_at=installed)

    def set_active(self, model_id: int) -> None:
        """Make a model the active one for its kind.

        Same two-step dance as profiles: a partial unique index permits one
        active model per kind, so the previous one is cleared first.
        """
        with self._db.transaction():
            row = self._db.query_one("SELECT kind FROM models WHERE id = ?", (model_id,))
            if row is None:
                raise DatabaseError(
                    f"model {model_id} not found",
                    user_message="Модель не найдена.",
                )
            self._deactivate(row["kind"])
            self._db.execute("UPDATE models SET is_active = 1 WHERE id = ?", (model_id,))

    def clear_active(self, kind: ModelKind) -> bool:
        """Leave a kind with no active model.

        Deleting the active model has to go through this first: the row is gone
        either way, but the subsystem using it needs to hear about it before its
        files disappear.

        Returns:
            Whether a model was actually deselected.
        """
        cursor = self._db.execute(
            "UPDATE models SET is_active = 0 WHERE kind = ? AND is_active = 1", (kind,)
        )
        return cursor.rowcount > 0

    def delete(self, model_id: int) -> bool:
        cursor = self._db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        return cursor.rowcount > 0

    def total_size(self, kind: ModelKind | None = None) -> int:
        """Recorded bytes on disk, for one kind or for everything."""
        if kind is None:
            return int(self._db.query_value("SELECT SUM(size_bytes) FROM models", default=0))
        return int(
            self._db.query_value(
                "SELECT SUM(size_bytes) FROM models WHERE kind = ?", (kind,), default=0
            )
        )

    def _deactivate(self, kind: str) -> None:
        self._db.execute(
            "UPDATE models SET is_active = 0 WHERE kind = ? AND is_active = 1", (kind,)
        )


# ----------------------------------------------------------------------
# maintenance
# ----------------------------------------------------------------------


class CleanupCategory(StrEnum):
    """A group of rows the Privacy tab can erase independently.

    Mirrors the GDPR-like buttons in section 11, so each button maps to exactly
    one member instead of the tab knowing which tables exist.
    """

    HISTORY = "history"
    AUDIT = "audit"
    CLIPBOARD = "clipboard"
    VARIABLES = "variables"
    TIMERS = "timers"
    COMMAND_VERSIONS = "command_versions"


#: Russian labels for the Privacy tab checkboxes.
CLEANUP_LABELS: Final[dict[CleanupCategory, str]] = {
    CleanupCategory.HISTORY: "История команд",
    CleanupCategory.AUDIT: "Журнал аудита",
    CleanupCategory.CLIPBOARD: "История буфера обмена",
    CleanupCategory.VARIABLES: "Сохранённые переменные",
    CleanupCategory.TIMERS: "Таймеры и напоминания",
    CleanupCategory.COMMAND_VERSIONS: "История версий команд",
}

_CLEANUP_STATEMENTS: Final[dict[CleanupCategory, str]] = {
    CleanupCategory.HISTORY: "DELETE FROM history",
    CleanupCategory.AUDIT: "DELETE FROM audit",
    CleanupCategory.CLIPBOARD: "DELETE FROM clipboard_history",
    # Commands keep working after this: only stored state is dropped.
    CleanupCategory.VARIABLES: "DELETE FROM variables",
    CleanupCategory.TIMERS: "DELETE FROM timers",
    CleanupCategory.COMMAND_VERSIONS: "DELETE FROM command_versions",
}


@dataclass(frozen=True, slots=True)
class CleanupReport:
    """What a cleanup removed, per category."""

    removed: dict[CleanupCategory, int]
    freed_bytes: int = 0

    @property
    def total(self) -> int:
        return sum(self.removed.values())

    def summary(self) -> str:
        """One-line Russian summary for the toast shown after cleanup."""
        if not self.total:
            return "нечего удалять"
        parts = [
            f"{CLEANUP_LABELS[category].lower()}: {count}"
            for category, count in self.removed.items()
            if count
        ]
        return ", ".join(parts)


class MaintenanceRepository(_Repository):
    """Retention, cleanup and backup — the Privacy tab's engine room."""

    def apply_retention(self, *, history_days: int = 0, history_limit: int = 0) -> int:
        """Enforce the retention settings. Returns rows removed.

        Runs at startup and after the settings change. Both limits are optional
        and ``0`` disables either one.
        """
        removed = 0
        with self._db.transaction():
            history = HistoryRepository(self._db)
            removed += history.delete_older_than(history_days)
            removed += history.trim_to_limit(history_limit)
        return removed

    def clear(
        self,
        categories: Iterable[CleanupCategory],
        *,
        vacuum: bool = True,
    ) -> CleanupReport:
        """Erase whole categories of data.

        Everything happens in one transaction, so a failure part-way leaves the
        user's data as it was rather than half-erased.

        Args:
            categories: What to remove.
            vacuum: Shrink the file afterwards. Worth it here — this is exactly
                the case where a lot of pages are freed at once — but it rewrites
                the file, so callers doing it repeatedly may turn it off.
        """
        requested = list(dict.fromkeys(categories))
        if not requested:
            return CleanupReport(removed={})

        size_before = self._db.file_size()
        removed: dict[CleanupCategory, int] = {}
        with self._db.transaction():
            for category in requested:
                cursor = self._db.execute(_CLEANUP_STATEMENTS[category])
                removed[category] = cursor.rowcount

        if vacuum:
            self._db.vacuum()

        freed = max(0, size_before - self._db.file_size())
        report = CleanupReport(removed=removed, freed_bytes=freed)
        _log.info("очистка данных: %s", report.summary())
        return report

    def clear_all(self, *, vacuum: bool = True) -> CleanupReport:
        """Erase every category. The "полный сброс" button, commands excepted."""
        return self.clear(list(CleanupCategory), vacuum=vacuum)

    def backup(self, destination: Path | str | None = None) -> Path:
        """Write a consistent copy of the database.

        Args:
            destination: Target file. Defaults to a timestamped file next to the
                database, which is what the "создать резервную копию" button uses.
        """
        if destination is None:
            stamp = utc_now().strftime("%Y%m%d_%H%M%S")
            destination = self._db.path.with_name(f"{self._db.path.stem}_backup_{stamp}.db")
        self._db.checkpoint()
        return self._db.backup(destination)

    def vacuum(self) -> int:
        """Compact the database. Returns the bytes freed."""
        before = self._db.file_size()
        self._db.vacuum()
        return max(0, before - self._db.file_size())

    def statistics(self) -> dict[str, int]:
        """Row counts per table, for the Privacy tab and DevTools."""
        tables = (
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
        )
        counts: dict[str, int] = {}
        for table in tables:
            # Table names come from this tuple, never from a caller.
            counts[table] = int(self._db.query_value(f'SELECT COUNT(*) FROM "{table}"', default=0))
        counts["size_bytes"] = self._db.file_size()
        return counts


# ----------------------------------------------------------------------
# facade
# ----------------------------------------------------------------------


class Repositories:
    """Every repository over one database.

    Built once at startup and handed to the subsystems that need storage.
    """

    __slots__ = (
        "_db",
        "audit",
        "clipboard",
        "commands",
        "folders",
        "history",
        "maintenance",
        "models",
        "profiles",
        "timers",
        "triggers",
        "variables",
    )

    def __init__(self, database: Database) -> None:
        self._db = database
        self.profiles = ProfileRepository(database)
        self.folders = FolderRepository(database)
        self.commands = CommandRepository(database)
        self.triggers = TriggerRepository(database)
        self.variables = VariableRepository(database)
        self.history = HistoryRepository(database)
        self.audit = AuditRepository(database)
        self.timers = TimerRepository(database)
        self.clipboard = ClipboardRepository(database)
        self.models = ModelRepository(database)
        self.maintenance = MaintenanceRepository(database)

    @property
    def database(self) -> Database:
        return self._db
