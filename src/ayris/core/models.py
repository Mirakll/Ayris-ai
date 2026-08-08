"""Typed entities for every table in the Ayris database.

One frozen dataclass per table. Repositories return these instead of
:class:`sqlite3.Row`, so the rest of the application never indexes a row by a
string key and mypy catches a renamed column at the call site.

Three conventions run through the module:

*Timestamps are ISO-8601 UTC strings.* The implicit ``datetime`` adapter that
:mod:`sqlite3` used to provide is deprecated as of Python 3.12, and the project
runs pytest with ``filterwarnings = ["error"]``, so relying on it would fail the
suite. Conversion is explicit through :func:`to_db_timestamp` and
:func:`from_db_timestamp`. The format is fixed and always UTC, which means the
lexicographic order of the stored text matches chronological order — that is
what lets ``history`` be sorted and ``timers`` be selected by ``fire_at``
without a functional index.

*JSON columns are parsed at the boundary.* The column keeps the ``_json`` suffix
required by the task, the dataclass field drops it and holds the decoded value:
``actions_json`` in SQL is :attr:`Command.actions` in Python.

*Identifiers are optional until the row exists.* ``id`` is ``None`` on an entity
that has not been inserted yet; :meth:`Repository.create` returns a copy with it
filled in via :func:`dataclasses.replace`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from sqlite3 import Row
from typing import Any, Final, Self, TypeVar, get_args

from ayris.core.paths import ModelKind

__all__ = [
    "MODEL_KINDS",
    "AuditEntry",
    "ClipboardEntry",
    "Command",
    "CommandFolder",
    "CommandVersion",
    "ExecutionResult",
    "HistoryEntry",
    "JsonObject",
    "ModelRecord",
    "Profile",
    "Timer",
    "TimerKind",
    "Trigger",
    "TriggerType",
    "Variable",
    "VariableScope",
    "VariableType",
    "dump_json",
    "from_db_timestamp",
    "load_json_array",
    "load_json_object",
    "load_string_array",
    "to_db_timestamp",
    "utc_now",
]

#: A decoded JSON object. ``Any`` for the values: action parameters are
#: arbitrary user data by design, and narrowing them here would only push casts
#: into every caller.
JsonObject = dict[str, Any]

#: Allowed values of ``models.kind``, mirroring the model directories in
#: :mod:`ayris.core.paths` so a record can always be resolved to a folder.
MODEL_KINDS: Final[tuple[str, ...]] = get_args(ModelKind)


class TriggerType(StrEnum):
    """How a command gets fired. Section 7.1 of the specification."""

    VOICE = "voice"
    HOTKEY = "hotkey"
    EVENT = "event"
    TIMER = "timer"


class VariableScope(StrEnum):
    """Lifetime of a variable. ``LOCAL`` lives for one command invocation."""

    LOCAL = "local"
    PROFILE = "profile"
    GLOBAL = "global"


class VariableType(StrEnum):
    """Declared type of a variable's value."""

    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ARRAY = "array"
    DICT = "dict"


class TimerKind(StrEnum):
    """What a scheduled entry represents."""

    TIMER = "timer"
    REMINDER = "reminder"
    ALARM = "alarm"


class ExecutionResult(StrEnum):
    """Outcome recorded in ``history`` and ``audit``."""

    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    DENIED = "denied"
    UNMATCHED = "unmatched"


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def to_db_timestamp(value: datetime | None) -> str | None:
    """Render a datetime for storage.

    A naive datetime is assumed to be UTC rather than local time: every producer
    inside Ayris uses :func:`utc_now`, so a missing tzinfo means the value came
    from a caller that did not think about zones, and guessing local time would
    silently shift it.
    """
    if value is None:
        return None
    aware = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return aware.isoformat(timespec="microseconds")


def from_db_timestamp(raw: object) -> datetime | None:
    """Parse a stored timestamp, tolerating rows written by an older build.

    Returns ``None`` for anything unparseable instead of raising: a corrupted
    timestamp in one history row must not break the whole history tab.
    """
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def dump_json(value: object) -> str:
    """Encode a JSON column. ``ensure_ascii`` is off so Russian text stays readable."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(raw: object) -> Any:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def load_json_object(raw: object) -> JsonObject:
    """Decode a JSON object column, falling back to ``{}``."""
    value = _loads(raw)
    return value if isinstance(value, dict) else {}


def load_json_array(raw: object) -> tuple[JsonObject, ...]:
    """Decode a JSON array of objects, dropping entries that are not objects."""
    value = _loads(raw)
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, dict))


def load_string_array(raw: object) -> tuple[str, ...]:
    """Decode a JSON array of strings, used for ``commands.tags``."""
    value = _loads(raw)
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value)


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _enum_or(raw: object, fallback: _EnumT) -> _EnumT:
    """Coerce a stored string to an enum member, falling back on unknown input.

    CHECK constraints keep the column honest, but a database written by a newer
    build may carry a member this one does not know yet.
    """
    if not isinstance(raw, str):
        return fallback
    try:
        return type(fallback)(raw)
    except ValueError:
        return fallback


@dataclass(frozen=True, slots=True)
class Profile:
    """A named set of commands, variables and folders."""

    name: str
    id: int | None = None
    created_at: datetime | None = None
    is_active: bool = False

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            name=row["name"],
            created_at=from_db_timestamp(row["created_at"]),
            is_active=bool(row["is_active"]),
        )


@dataclass(frozen=True, slots=True)
class CommandFolder:
    """A node of the command library tree.

    ``profile_id`` is ``NULL`` for a folder shared by every profile; the column
    is not in the task's field list but without it deleting a profile would
    leave its folders behind.

    The ordering column is called ``sort_order``: ``ORDER`` is a reserved SQL
    keyword and would need quoting in every statement that touches it.
    """

    name: str
    id: int | None = None
    profile_id: int | None = None
    parent_id: int | None = None
    sort_order: int = 0

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            profile_id=row["profile_id"],
            parent_id=row["parent_id"],
            name=row["name"],
            sort_order=row["sort_order"],
        )


@dataclass(frozen=True, slots=True)
class Command:
    """A command: metadata plus the ordered list of action blocks."""

    name: str
    profile_id: int
    id: int | None = None
    folder_id: int | None = None
    description: str = ""
    tags: tuple[str, ...] = ()
    enabled: bool = True
    priority: int = 0
    cooldown_ms: int = 0
    require_admin: bool = False
    actions: tuple[JsonObject, ...] = ()
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            profile_id=row["profile_id"],
            folder_id=row["folder_id"],
            name=row["name"],
            description=row["description"],
            tags=load_string_array(row["tags"]),
            enabled=bool(row["enabled"]),
            priority=row["priority"],
            cooldown_ms=row["cooldown_ms"],
            require_admin=bool(row["require_admin"]),
            actions=load_json_array(row["actions_json"]),
            created_at=from_db_timestamp(row["created_at"]),
            updated_at=from_db_timestamp(row["updated_at"]),
        )


@dataclass(frozen=True, slots=True)
class CommandVersion:
    """A snapshot of a command, kept for undo/redo and for export.

    ``snapshot`` holds the whole command as it looked, so restoring a version
    never has to reconstruct it from a diff.
    """

    command_id: int
    version: int
    snapshot: JsonObject
    id: int | None = None
    comment: str = ""
    created_at: datetime | None = None

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            command_id=row["command_id"],
            version=row["version"],
            snapshot=load_json_object(row["snapshot_json"]),
            comment=row["comment"],
            created_at=from_db_timestamp(row["created_at"]),
        )


@dataclass(frozen=True, slots=True)
class Trigger:
    """One way of firing a command.

    ``payload`` is shaped by :attr:`type`: a phrase for ``voice``, a key
    combination for ``hotkey``, an event name for ``event``, a schedule for
    ``timer``.
    """

    command_id: int
    type: TriggerType = TriggerType.VOICE
    payload: JsonObject = field(default_factory=dict)
    id: int | None = None
    fuzzy: bool = True
    priority: int = 0

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            command_id=row["command_id"],
            type=_enum_or(row["type"], TriggerType.VOICE),
            payload=load_json_object(row["payload_json"]),
            fuzzy=bool(row["fuzzy"]),
            priority=row["priority"],
        )

    @property
    def phrase(self) -> str:
        """Spoken phrase of a voice trigger, empty for other types."""
        value = self.payload.get("phrase", "")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True, slots=True)
class Variable:
    """A stored variable. ``value`` is the decoded ``value_json`` column."""

    name: str
    id: int | None = None
    scope: VariableScope = VariableScope.GLOBAL
    profile_id: int | None = None
    type: VariableType = VariableType.STRING
    value: Any = None
    persistent: bool = True

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            scope=_enum_or(row["scope"], VariableScope.GLOBAL),
            profile_id=row["profile_id"],
            name=row["name"],
            type=_enum_or(row["type"], VariableType.STRING),
            value=_loads(row["value_json"]),
            persistent=bool(row["persistent"]),
        )


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    """One pass through the pipeline, matched or not."""

    id: int | None = None
    ts: datetime | None = None
    stt_raw: str = ""
    matched_command_id: int | None = None
    intent: str = ""
    params: JsonObject = field(default_factory=dict)
    result: ExecutionResult = ExecutionResult.OK
    error: str = ""
    duration_ms: int = 0

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            ts=from_db_timestamp(row["ts"]),
            stt_raw=row["stt_raw"],
            matched_command_id=row["matched_command_id"],
            intent=row["intent"],
            params=load_json_object(row["params_json"]),
            result=_enum_or(row["result"], ExecutionResult.OK),
            error=row["error"],
            duration_ms=row["duration_ms"],
        )


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """Security journal required by section 11.

    Separate from ``history`` on purpose: history is a UX convenience the user
    may clear at will, the audit trail records privileged execution and is
    written even when history is switched off.
    """

    command_name: str
    id: int | None = None
    ts: datetime | None = None
    params: JsonObject = field(default_factory=dict)
    result: ExecutionResult = ExecutionResult.OK
    require_admin: bool = False
    elevated: bool = False

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            ts=from_db_timestamp(row["ts"]),
            command_name=row["command_name"],
            params=load_json_object(row["params_json"]),
            result=_enum_or(row["result"], ExecutionResult.OK),
            require_admin=bool(row["require_admin"]),
            elevated=bool(row["elevated"]),
        )


@dataclass(frozen=True, slots=True)
class Timer:
    """A one-shot timer, a reminder or a recurring alarm.

    Either ``fire_at`` (one-shot) or ``cron`` (recurring) is set; the scheduler
    in task 28 recomputes ``fire_at`` from ``cron`` after each firing.
    """

    label: str
    id: int | None = None
    kind: TimerKind = TimerKind.TIMER
    fire_at: datetime | None = None
    cron: str = ""
    enabled: bool = True
    sound: str = ""
    payload: JsonObject = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            kind=_enum_or(row["kind"], TimerKind.TIMER),
            label=row["label"],
            fire_at=from_db_timestamp(row["fire_at"]),
            cron=row["cron"],
            enabled=bool(row["enabled"]),
            sound=row["sound"],
            payload=load_json_object(row["payload_json"]),
        )


@dataclass(frozen=True, slots=True)
class ClipboardEntry:
    """A clipboard snapshot. Pinned entries survive history trimming."""

    content: str
    id: int | None = None
    ts: datetime | None = None
    pinned: bool = False

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            ts=from_db_timestamp(row["ts"]),
            content=row["content"],
            pinned=bool(row["pinned"]),
        )


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """An installed STT/TTS/wake-word/LLM model."""

    kind: ModelKind
    name: str
    id: int | None = None
    version: str = ""
    path: str = ""
    sha256: str = ""
    installed_at: datetime | None = None
    is_active: bool = False

    @classmethod
    def from_row(cls, row: Row) -> Self:
        return cls(
            id=row["id"],
            kind=row["kind"],
            name=row["name"],
            version=row["version"],
            path=row["path"],
            sha256=row["sha256"],
            installed_at=from_db_timestamp(row["installed_at"]),
            is_active=bool(row["is_active"]),
        )
