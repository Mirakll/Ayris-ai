"""What the assistant remembers between two utterances.

A command is rarely said in isolation. «Открой браузер» is followed by «закрой
его», «который час» by «повтори», and a phrase that means one thing in Photoshop
means another in a game. All of that needs exactly one thing: a small, honest
record of what just happened, with an expiry date on every field.

**One writer.** The context is read from the pipeline thread, written from the
pipeline, from the bus (a cancel, a profile switch) and from the settings window.
So nothing outside this module touches a field: every mutation is a method on
:class:`DialogContext`, taken under one lock, and every read goes through
:meth:`DialogContext.snapshot`, which hands back a frozen
:class:`ContextSnapshot`. A caller that held a reference to the live object could
otherwise see the last object change between the anaphora check and the match.

**Everything expires, and not at the same rate.** Three deadlines, because they
answer three different questions:

* :attr:`ContextTtl.followup` — «are we still in a dialogue?» The last command
  can be repeated as an action and referred to inside this window; it is extended
  by every utterance, so a conversation stays alive while it is being had.
* :attr:`ContextTtl.objects` — «is «его» still unambiguous?» Longer than a single
  follow-up hop, because «закрой его» half a minute later still means the browser,
  but not indefinitely: pointing a pronoun at a window from ten minutes ago is how
  an assistant closes the wrong thing.
* :attr:`ContextTtl.answer` — «can I repeat myself?» Much longer, and on purpose.
  «Повтори» is a request to hear the last answer again, and the user who asks has
  usually just walked back to the desk.

The pending clarification has its own, short deadline: a question the user never
answered must not swallow the next unrelated command.

**Time is wall-clock, deliberately.** The obvious choice is
:func:`time.monotonic`, and it is the wrong one here: the context outlives the
process. A monotonic reading persisted before a restart means nothing after it,
so the deadlines would have to be recomputed from a second, wall-clock, stamp —
two clocks, one of which is only there to fix the other. The clock is injected
(:class:`DialogContext` takes ``clock``), which is what the tests use instead of
sleeping, and what makes a clock jump a testable case rather than a rumour.

**The active window is not asked for twice in a row.** Every matched phrase would
otherwise cost a ``GetForegroundWindow`` plus a ``GetWindowTextW`` plus a process
lookup, on the UI thread, for a value that cannot meaningfully change inside one
utterance. :func:`get_active_window` caches for :data:`WINDOW_CACHE_TTL` seconds
and never raises: off Windows, and on a Windows where the call fails, it returns
``None``, and a trigger filtered by window simply does not fire.
"""

from __future__ import annotations

import ctypes
import sys
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Final, Protocol, TypeVar

from ayris.core.events import CancelRequested, TtsStarted
from ayris.core.models import JsonObject, VariableType, utc_now
from ayris.nlu.normalize import fold_letters
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from ayris.core.config import CommandsConfig
    from ayris.core.events import EventBus, Unsubscribe
    from ayris.core.repositories import VariableRepository

#: Bound to :class:`~enum.StrEnum` rather than :class:`~enum.Enum`: every enum
#: stored in the context is a string enum, and the narrower bound is what lets
#: :func:`_enum_member` be called with a plain JSON string.
_EnumT = TypeVar("_EnumT", bound=StrEnum)

__all__ = [
    "CONTEXT_VARIABLE",
    "DEFAULT_ANSWER_TTL",
    "DEFAULT_FOLLOWUP_TTL",
    "DEFAULT_OBJECT_TTL",
    "DEFAULT_PENDING_TTL",
    "MAX_OBJECTS",
    "WINDOW_CACHE_TTL",
    "ContextObject",
    "ContextSnapshot",
    "ContextStore",
    "ContextTtl",
    "DialogContext",
    "Gender",
    "LastAnswer",
    "LastCommand",
    "MemoryContextStore",
    "ObjectKind",
    "PendingKind",
    "PendingRequest",
    "TimeOfDay",
    "VariableContextStore",
    "WindowInfo",
    "get_active_window",
    "guess_gender",
    "invalidate_active_window",
]

_log = get_logger(__name__)

#: Follow-up window: how long the last command stays referable. Section 5.2.
DEFAULT_FOLLOWUP_TTL: Final = 30.0
#: How long «его» keeps pointing at the last object.
DEFAULT_OBJECT_TTL: Final = 60.0
#: How long «повтори» can still fetch the last answer.
DEFAULT_ANSWER_TTL: Final = 600.0
#: How long an unanswered clarifying question keeps the floor.
DEFAULT_PENDING_TTL: Final = 20.0
#: Objects kept, newest first. Deeper than that is not memory, it is guessing.
MAX_OBJECTS: Final = 8
#: Lifetime of one active-window reading, in seconds.
WINDOW_CACHE_TTL: Final = 0.3
#: Global variable the persisted context lives in.
CONTEXT_VARIABLE: Final = "__dialog_context__"

#: Version of the persisted payload. A restart across an upgrade that changed the
#: shape must drop the context, not misread it.
_STATE_VERSION: Final = 1


class TimeOfDay(StrEnum):
    """Part of the day, for greetings and for time-conditioned triggers."""

    NIGHT = "night"
    MORNING = "morning"
    DAY = "day"
    EVENING = "evening"

    @classmethod
    def from_hour(cls, hour: int) -> TimeOfDay:
        """The part of the day an hour belongs to, on the usual Russian split."""
        if 5 <= hour < 11:
            return cls.MORNING
        if 11 <= hour < 17:
            return cls.DAY
        if 17 <= hour < 23:
            return cls.EVENING
        return cls.NIGHT

    @property
    def label(self) -> str:
        """Russian name, for the trigger editor and for «доброе утро»."""
        return _TIME_OF_DAY_LABELS[self]


_TIME_OF_DAY_LABELS: Final[Mapping[TimeOfDay, str]] = {
    TimeOfDay.NIGHT: "ночь",
    TimeOfDay.MORNING: "утро",
    TimeOfDay.DAY: "день",
    TimeOfDay.EVENING: "вечер",
}

#: What a user may write in ``when_time``. Both languages, because the settings
#: window is Russian and an imported command may well be English.
TIME_OF_DAY_WORDS: Final[Mapping[str, TimeOfDay]] = {
    "ночь": TimeOfDay.NIGHT,
    "ночью": TimeOfDay.NIGHT,
    "night": TimeOfDay.NIGHT,
    "утро": TimeOfDay.MORNING,
    "утром": TimeOfDay.MORNING,
    "morning": TimeOfDay.MORNING,
    "день": TimeOfDay.DAY,
    "днём": TimeOfDay.DAY,
    "day": TimeOfDay.DAY,
    "вечер": TimeOfDay.EVENING,
    "вечером": TimeOfDay.EVENING,
    "evening": TimeOfDay.EVENING,
}


class Gender(StrEnum):
    """Grammatical gender of an object's name, for pronoun agreement."""

    MASCULINE = "m"
    FEMININE = "f"
    NEUTER = "n"
    PLURAL = "p"


def guess_gender(name: str) -> Gender:
    """Guess the gender of a Russian noun from its ending.

    Crude on purpose. It is used to *prefer* one remembered object over another
    when several could answer «его», never to reject one, so a wrong guess costs
    a slightly worse choice between two candidates and nothing else. A Latin name
    — «Chrome», «Discord» — comes out masculine, which is how they are spoken
    about in Russian.
    """
    word = fold_letters(name).strip().split()[-1] if name.strip() else ""
    if not word:
        return Gender.MASCULINE
    if word.endswith(("ы", "и")) and len(word) > 3:
        return Gender.PLURAL
    if word.endswith(("а", "я")):
        return Gender.FEMININE
    if word.endswith(("о", "е")):
        return Gender.NEUTER
    if word.endswith("ь"):
        return Gender.FEMININE
    return Gender.MASCULINE


@dataclass(frozen=True, slots=True)
class WindowInfo:
    """The foreground window, as much of it as matching needs."""

    title: str = ""
    process: str = ""
    pid: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the reading carries nothing to match against."""
        return not (self.title or self.process)

    def matches(self, pattern: str) -> bool:
        """Whether ``pattern`` describes this window.

        A pattern with ``*`` or ``?`` is a glob, which is what the specification's
        ``"*Photoshop*"`` expects. A pattern without either is treated as a
        substring, because a user who wrote ``Photoshop`` meant the same thing and
        should not have to know about globs to get it.
        """
        needle = fold_letters(pattern).strip()
        if not needle:
            return True
        subjects = (fold_letters(self.title), fold_letters(self.process))
        if any(char in needle for char in "*?["):
            return any(fnmatchcase(subject, needle) for subject in subjects)
        return any(needle in subject for subject in subjects)

    def as_json(self) -> JsonObject:
        """Serialisable form, for the persisted snapshot and for DevTools."""
        return {"title": self.title, "process": self.process, "pid": self.pid}


class _WindowCache:
    """Holds the last active-window reading and the moment it was taken.

    A class rather than two module globals so that the lock and the value cannot
    drift apart, and so that a test can reset both at once.
    """

    __slots__ = ("_at", "_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._at = 0.0
        self._value: WindowInfo | None = None

    def get(self, probe: Callable[[], WindowInfo | None], ttl: float) -> WindowInfo | None:
        now = time.monotonic()
        with self._lock:
            if self._at and now - self._at < ttl:
                return self._value
        value = probe()
        with self._lock:
            self._at = now
            self._value = value
        return value

    def invalidate(self) -> None:
        with self._lock:
            self._at = 0.0
            self._value = None


_WINDOW_CACHE: Final = _WindowCache()


def _win_function(library: str, function: str) -> Any | None:
    """Look up a WinAPI entry point, or ``None`` when it is unavailable."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        return getattr(getattr(windll, library), function)
    except (AttributeError, OSError):
        return None


def _window_title(handle: int) -> str:
    """Title of a window, or ``""`` when it has none or cannot be read."""
    get_length = _win_function("user32", "GetWindowTextLengthW")
    get_text = _win_function("user32", "GetWindowTextW")
    if get_length is None or get_text is None:
        return ""
    length = int(get_length(ctypes.c_void_p(handle)))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    get_text(ctypes.c_void_p(handle), buffer, length + 1)
    return str(buffer.value)


def _window_pid(handle: int) -> int:
    """Owning process id of a window, or ``0``."""
    get_pid = _win_function("user32", "GetWindowThreadProcessId")
    if get_pid is None:
        return 0
    pid = ctypes.c_ulong(0)
    get_pid(ctypes.c_void_p(handle), ctypes.byref(pid))
    return int(pid.value)


def _process_name(pid: int) -> str:
    """Executable name of a process, or ``""``.

    ``QueryFullProcessImageNameW`` needs only ``PROCESS_QUERY_LIMITED_INFORMATION``,
    which an unelevated Ayris is granted for a normal foreground application. An
    elevated window returns nothing, and that is a fact about Windows, not an
    error: the filter falls back to the title.
    """
    if pid <= 0:
        return ""
    open_process = _win_function("kernel32", "OpenProcess")
    query_name = _win_function("kernel32", "QueryFullProcessImageNameW")
    close_handle = _win_function("kernel32", "CloseHandle")
    if open_process is None or query_name is None or close_handle is None:
        return ""
    handle = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        size = ctypes.c_ulong(_MAX_PATH)
        buffer = ctypes.create_unicode_buffer(_MAX_PATH)
        if not query_name(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return str(buffer.value).rsplit("\\", 1)[-1]
    finally:
        close_handle(handle)


#: ``PROCESS_QUERY_LIMITED_INFORMATION`` — the least right that answers the name.
_PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
_MAX_PATH: Final = 260


def _probe_active_window() -> WindowInfo | None:
    """Ask Windows what the foreground window is. ``None`` when it cannot say."""
    # sys.platform rather than os.name: mypy reads it as a platform guard, so the
    # other branch is not reported as unreachable under warn_unreachable.
    if sys.platform != "win32":
        return None
    get_foreground = _win_function("user32", "GetForegroundWindow")
    if get_foreground is None:
        return None
    try:
        handle = int(get_foreground() or 0)
        if handle == 0:
            return None
        title = _window_title(handle)
        pid = _window_pid(handle)
        process = _process_name(pid)
    except OSError as exc:
        _log.debug("Активное окно недоступно: %s", exc)
        return None
    info = WindowInfo(title=title, process=process, pid=pid)
    return None if info.is_empty else info


def get_active_window(*, ttl: float = WINDOW_CACHE_TTL) -> WindowInfo | None:
    """The foreground window, cached for ``ttl`` seconds.

    The single seam the rest of Ayris goes through: substituting it is how the
    trigger filters are tested off Windows, and it is the only place the WinAPI
    calls live. Never raises.
    """
    return _WINDOW_CACHE.get(_probe_active_window, ttl)


def invalidate_active_window() -> None:
    """Forget the cached reading. Called when a test or a window switch says so."""
    _WINDOW_CACHE.invalidate()


class ObjectKind(StrEnum):
    """What kind of thing was mentioned. Decides which pronoun can reach it."""

    APP = "app"
    WINDOW = "window"
    FILE = "file"
    URL = "url"
    DEVICE = "device"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class ContextObject:
    """The last thing talked about, and enough of it to act on again.

    ``value`` is the machine-readable half — an application id, a path, a URL —
    and ``name`` the half a human said and would recognise in a question. Both are
    kept: the action needs the first, «Закрыть Google Chrome?» needs the second.
    """

    kind: ObjectKind
    name: str
    value: str = ""
    gender: Gender = Gender.MASCULINE
    at: datetime = field(default_factory=utc_now)

    @property
    def target(self) -> str:
        """What to act on: the machine-readable value, or the name if there is none."""
        return self.value or self.name

    def as_json(self) -> JsonObject:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "value": self.value,
            "gender": self.gender.value,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ContextObject | None:
        kind = _enum_member(ObjectKind, data.get("kind"))
        name = _as_str(data.get("name"))
        at = _as_time(data.get("at"))
        if kind is None or not name or at is None:
            return None
        gender = _enum_member(Gender, data.get("gender")) or guess_gender(name)
        return cls(kind=kind, name=name, value=_as_str(data.get("value")), gender=gender, at=at)


@dataclass(frozen=True, slots=True)
class LastCommand:
    """The command that ran last, with everything needed to run it again.

    ``dangerous`` is set by the caller, not guessed here: the registry of task 19
    is what knows that ``power.shutdown`` is not to be repeated on a whim, and
    guessing it from the intent name would be a security decision made by string
    matching.
    """

    command_id: int | None = None
    intent: str = ""
    action: str = ""
    phrase: str = ""
    slots: Mapping[str, Any] = field(default_factory=dict)
    result: str = ""
    dangerous: bool = False
    at: datetime = field(default_factory=utc_now)

    @property
    def title(self) -> str:
        """Shortest honest description, for a confirmation question."""
        return self.phrase or self.intent or self.action

    def as_json(self) -> JsonObject:
        return {
            "command_id": self.command_id,
            "intent": self.intent,
            "action": self.action,
            "phrase": self.phrase,
            "slots": {name: _jsonable(value) for name, value in self.slots.items()},
            "result": self.result,
            "dangerous": self.dangerous,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LastCommand | None:
        at = _as_time(data.get("at"))
        if at is None:
            return None
        raw_slots = data.get("slots")
        slots = dict(raw_slots) if isinstance(raw_slots, dict) else {}
        command_id = data.get("command_id")
        return cls(
            command_id=command_id if isinstance(command_id, int) else None,
            intent=_as_str(data.get("intent")),
            action=_as_str(data.get("action")),
            phrase=_as_str(data.get("phrase")),
            slots=slots,
            result=_as_str(data.get("result")),
            dangerous=bool(data.get("dangerous", False)),
            at=at,
        )


@dataclass(frozen=True, slots=True)
class LastAnswer:
    """The last thing Ayris said out loud, so that it can say it again."""

    text: str
    at: datetime = field(default_factory=utc_now)

    def as_json(self) -> JsonObject:
        return {"text": self.text, "at": self.at.isoformat()}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> LastAnswer | None:
        text = _as_str(data.get("text"))
        at = _as_time(data.get("at"))
        if not text or at is None:
            return None
        return cls(text=text, at=at)


class PendingKind(StrEnum):
    """What Ayris is waiting to hear."""

    SLOT = "slot"
    CONFIRM = "confirm"


@dataclass(frozen=True, slots=True)
class PendingRequest:
    """A question that was asked out loud and whose answer is the next utterance.

    Deliberately not persisted: a clarification that survived a restart would meet
    a user who has no idea a question is outstanding, and eat their first command
    as its answer. :meth:`DialogContext.restore` drops it.
    """

    kind: PendingKind
    question: str
    command_id: int | None = None
    intent: str = ""
    slot: str = ""
    slot_type: str = ""
    known: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 0
    at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ContextTtl:
    """The four deadlines, in seconds. ``0`` switches a feature off entirely."""

    followup: float = DEFAULT_FOLLOWUP_TTL
    objects: float = DEFAULT_OBJECT_TTL
    answer: float = DEFAULT_ANSWER_TTL
    pending: float = DEFAULT_PENDING_TTL

    @classmethod
    def from_config(cls, commands: CommandsConfig) -> ContextTtl:
        """Read the deadlines out of the «Команды» tab."""
        return cls(
            followup=commands.followup_ttl_s,
            objects=commands.object_ttl_s,
            answer=commands.answer_ttl_s,
            pending=commands.clarify_timeout_s,
        )


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    """The context as it was at one moment. Frozen, so it is safe to hand around."""

    at: datetime = field(default_factory=utc_now)
    profile_id: int | None = None
    touched_at: datetime | None = None
    command: LastCommand | None = None
    objects: tuple[ContextObject, ...] = ()
    answer: LastAnswer | None = None
    pending: PendingRequest | None = None
    window: WindowInfo | None = None
    variables: Mapping[str, Any] = field(default_factory=dict)
    ttl: ContextTtl = field(default_factory=ContextTtl)

    @property
    def time_of_day(self) -> TimeOfDay:
        """Which part of the day it is, for greetings and for trigger conditions."""
        return TimeOfDay.from_hour(self.at.astimezone().hour)

    @property
    def hour(self) -> int:
        """Local hour, ``0..23``. The context stores UTC; conditions are local."""
        return self.at.astimezone().hour

    @property
    def follow_up_active(self) -> bool:
        """Whether the follow-up window is still open."""
        if self.touched_at is None or self.ttl.followup <= 0:
            return False
        return _elapsed(self.at, self.touched_at) <= self.ttl.followup

    @property
    def last_object(self) -> ContextObject | None:
        """The most recently mentioned object, of any kind."""
        return self.objects[0] if self.objects else None

    def object_of(
        self,
        kinds: Iterable[ObjectKind] | None = None,
        *,
        gender: Gender | None = None,
    ) -> ContextObject | None:
        """The newest remembered object of a suitable kind.

        ``gender`` is a preference, not a filter: «закрой его» about a program the
        user calls «программа» is normal speech, and refusing to answer it because
        the noun is feminine would be pedantry with a cost. A gender match wins
        when there is one, recency decides otherwise.
        """
        allowed = tuple(kinds) if kinds is not None else ()
        candidates = [item for item in self.objects if not allowed or item.kind in allowed]
        if not candidates:
            return None
        if gender is not None:
            for item in candidates:
                if item.gender is gender:
                    return item
        return candidates[0]

    def variable(self, name: str, default: Any = None) -> Any:
        """Value of a profile variable, folded case, or ``default``."""
        if name in self.variables:
            return self.variables[name]
        folded = fold_letters(name)
        for key, value in self.variables.items():
            if fold_letters(key) == folded:
                return value
        return default

    @property
    def can_repeat_answer(self) -> bool:
        """Whether there is an answer left to repeat."""
        return self.answer is not None

    @property
    def can_repeat_action(self) -> bool:
        """Whether the last command is still close enough to be repeated."""
        return self.command is not None and self.follow_up_active


def _elapsed(now: datetime, then: datetime) -> float:
    """Seconds between two stamps, never negative.

    A clock that jumped backwards — an NTP correction, a user fixing the date —
    would otherwise produce a negative age, which compares as «younger than
    fresh» and quietly extends every deadline. Clamping means the worst a
    backwards jump can do is keep the context alive one extra utterance.
    """
    return max(0.0, (now - then).total_seconds())


def _as_str(value: Any) -> str:
    """Whatever came out of JSON, as a string; anything else becomes ``""``."""
    return value if isinstance(value, str) else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    """A JSON object as a mapping; anything else as an empty one."""
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return {}


def _as_time(value: Any) -> datetime | None:
    """Parse an ISO stamp written by :meth:`ContextObject.as_json`.

    A naive stamp is read as UTC: the context is stored in UTC, and a stamp
    without an offset can only have come from a version that did the same.
    """
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed


def _enum_member(enum_type: type[_EnumT], value: Any) -> _EnumT | None:
    """A member of ``enum_type``, or ``None`` for anything unrecognised."""
    if not isinstance(value, str):
        return None
    try:
        return enum_type(value)
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    """Make a slot value survive a round trip through JSON.

    Slots hold parsed things — a :class:`~datetime.datetime` from the time
    parser, an :class:`~enum.Enum` from a choice slot, a
    :class:`~ayris.nlu.apps.AppMatch`. ``json.dumps`` would raise on those and
    take the whole save down with it, so anything without an obvious
    representation is stored as its ``str``: a repeated command is matched again
    from ``phrase`` anyway, and the slots are there for the confirmation
    question and the history line.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    return str(value)


class ContextStore(Protocol):
    """Where the context sleeps between two runs of the program.

    Deliberately narrow: three methods over one JSON blob. The context is read
    and written as a whole, exactly once per change, and a real table for it
    would buy nothing but a migration.
    """

    def load(self) -> JsonObject | None:
        """The stored payload, or ``None`` when there is none."""
        ...

    def save(self, state: JsonObject) -> None:
        """Replace the stored payload."""
        ...

    def clear(self) -> None:
        """Forget the stored payload."""
        ...


class MemoryContextStore:
    """A store that forgets on exit. The default, and what the tests use."""

    __slots__ = ("_state",)

    def __init__(self) -> None:
        self._state: JsonObject | None = None

    def load(self) -> JsonObject | None:
        return self._state

    def save(self, state: JsonObject) -> None:
        self._state = state

    def clear(self) -> None:
        self._state = None


class VariableContextStore:
    """Keeps the context in the ``variables`` table, as one global variable.

    Chosen over a table of its own on purpose. The context is a single opaque
    blob with no query pattern beyond «give me the last one», and
    :class:`~ayris.core.repositories.VariableRepository` already stores exactly
    that, with JSON encoding and a schema that will not need a migration to
    hold one more field.

    Never raises: a database that is locked or gone must not take down the
    dialogue, it only means the context does not survive this restart.
    """

    __slots__ = ("_name", "_variables")

    def __init__(self, variables: VariableRepository, *, name: str = CONTEXT_VARIABLE) -> None:
        self._variables = variables
        self._name = name

    def load(self) -> JsonObject | None:
        try:
            value = self._variables.get_value(self._name)
        except Exception:
            _log.exception("контекст: не удалось прочитать сохранённое состояние")
            return None
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
        return None

    def save(self, state: JsonObject) -> None:
        try:
            self._variables.set(self._name, state, var_type=VariableType.DICT)
        except Exception:
            _log.exception("контекст: не удалось сохранить состояние")

    def clear(self) -> None:
        try:
            self._variables.delete(self._name)
        except Exception:
            _log.exception("контекст: не удалось удалить сохранённое состояние")


class DialogContext:
    """What the assistant remembers, and the only thing allowed to change it.

    Args:
        ttl: The four deadlines. Defaults to the module constants; production
            builds it from the config with :meth:`ContextTtl.from_config`.
        store: Where to persist to. Defaults to a memory store, i.e. nothing
            survives the process — production passes a
            :class:`VariableContextStore`.
        clock: Source of «now». Wall-clock UTC by default; tests hand in a
            counter and get deterministic expiry without sleeping.
        window_probe: How to learn the active window. Defaults to the cached
            WinAPI reading; tests hand in a stub.
        profile_id: Which profile this context belongs to. A switch to another
            profile clears everything — see :meth:`reset`.
        autosave: Persist after every change. Off makes the caller responsible
            for calling :meth:`save`, which is what a batch of updates wants.

    Every public method takes the lock; every read goes out as a frozen
    :class:`ContextSnapshot`. The lock is an :class:`~threading.RLock` because a
    handler on the bus thread can end up calling back in — a cancel that clears
    a pending request while :meth:`snapshot` is being built one frame down.
    """

    __slots__ = (
        "_answer",
        "_autosave",
        "_clock",
        "_command",
        "_lock",
        "_objects",
        "_pending",
        "_profile_id",
        "_store",
        "_touched_at",
        "_ttl",
        "_unsubscribe",
        "_variables",
        "_window_probe",
    )

    def __init__(
        self,
        *,
        ttl: ContextTtl | None = None,
        store: ContextStore | None = None,
        clock: Callable[[], datetime] = utc_now,
        window_probe: Callable[[], WindowInfo | None] = get_active_window,
        profile_id: int | None = None,
        autosave: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self._ttl = ttl or ContextTtl()
        self._store: ContextStore = store or MemoryContextStore()
        self._clock = clock
        self._window_probe = window_probe
        self._autosave = autosave
        self._profile_id = profile_id
        self._touched_at: datetime | None = None
        self._command: LastCommand | None = None
        self._objects: tuple[ContextObject, ...] = ()
        self._answer: LastAnswer | None = None
        self._pending: PendingRequest | None = None
        self._variables: dict[str, Any] = {}
        self._unsubscribe: list[Unsubscribe] = []

    # ------------------------------------------------------------------ reading

    def snapshot(self) -> ContextSnapshot:
        """The context as it is now, with everything expired already dropped.

        Pruning happens here rather than on a timer: there is no thread to run a
        timer on, an expiry that nobody looks at has no consequences, and doing
        it on read means a snapshot can be trusted without a second age check.
        """
        with self._lock:
            now = self._now()
            self._prune(now)
            return ContextSnapshot(
                at=now,
                profile_id=self._profile_id,
                touched_at=self._touched_at,
                command=self._command,
                objects=self._objects,
                answer=self._answer,
                pending=self._pending,
                window=self._window(),
                variables=dict(self._variables),
                ttl=self._ttl,
            )

    @property
    def ttl(self) -> ContextTtl:
        """The deadlines in force."""
        return self._ttl

    @property
    def profile_id(self) -> int | None:
        """Profile this context belongs to."""
        with self._lock:
            return self._profile_id

    def pending(self) -> PendingRequest | None:
        """The outstanding question, or ``None``. Expired ones are dropped here."""
        with self._lock:
            self._prune(self._now())
            return self._pending

    # ------------------------------------------------------------------ writing

    def touch(self) -> None:
        """Note that the user just said something: the dialogue is alive.

        Called for every recognised utterance, including ones that matched
        nothing — a misheard phrase is still a person talking to the assistant,
        and letting the follow-up window close under them is how «закрой его»
        stops working right after a failed command.
        """
        with self._lock:
            self._touched_at = self._now()
            self._persist()

    def remember_object(
        self,
        kind: ObjectKind,
        name: str,
        value: str = "",
        *,
        gender: Gender | None = None,
    ) -> ContextObject | None:
        """Note that something was mentioned, and can now be called «его».

        A repeated mention of the same thing moves it to the front instead of
        being stored twice, so a list of eight really holds eight distinct
        things. Returns ``None`` for a nameless object, which is a caller bug
        that is not worth an exception.
        """
        if not name.strip():
            return None
        with self._lock:
            now = self._now()
            item = ContextObject(
                kind=kind,
                name=name.strip(),
                value=value.strip(),
                gender=gender if gender is not None else guess_gender(name),
                at=now,
            )
            rest = [
                other
                for other in self._objects
                if not (other.kind is item.kind and other.target == item.target)
            ]
            self._objects = (item, *rest)[:MAX_OBJECTS]
            self._touched_at = now
            self._persist()
            return item

    def remember_command(
        self,
        *,
        intent: str = "",
        action: str = "",
        phrase: str = "",
        slots: Mapping[str, Any] | None = None,
        result: str = "",
        dangerous: bool = False,
        command_id: int | None = None,
    ) -> LastCommand:
        """Note the command that just ran, so that it can be repeated.

        ``dangerous`` comes from the command definition, not from a guess about
        the name: whether repeating something without asking is acceptable is a
        decision for whoever declared the action.
        """
        with self._lock:
            now = self._now()
            command = LastCommand(
                command_id=command_id,
                intent=intent,
                action=action,
                phrase=phrase,
                slots=dict(slots or {}),
                result=result,
                dangerous=dangerous,
                at=now,
            )
            self._command = command
            self._touched_at = now
            self._persist()
            return command

    def remember_answer(self, text: str) -> LastAnswer | None:
        """Note what was said out loud, for «повтори».

        Does not extend the follow-up window: the assistant talking is not the
        user talking, and a long answer should not keep the dialogue open longer
        than a short one.
        """
        cleaned = text.strip()
        if not cleaned:
            return None
        with self._lock:
            answer = LastAnswer(text=cleaned, at=self._now())
            self._answer = answer
            self._persist()
            return answer

    def set_pending(self, request: PendingRequest) -> PendingRequest:
        """Record that a question was asked and its answer is due next."""
        with self._lock:
            stamped = replace(request, at=self._now())
            self._pending = stamped
            self._touched_at = stamped.at
            return stamped

    def clear_pending(self) -> PendingRequest | None:
        """Stop waiting for an answer. Returns what was dropped, if anything."""
        with self._lock:
            pending = self._pending
            self._pending = None
            return pending

    def set_variable(self, name: str, value: Any) -> None:
        """Publish a variable into the context, for conditions and templates."""
        with self._lock:
            self._variables[name] = value

    def load_variables(self, values: Mapping[str, Any]) -> None:
        """Replace the visible variables wholesale, on a profile load."""
        with self._lock:
            self._variables = dict(values)

    def cancel(self, *, reason: str = "") -> None:
        """«Отмена». Drop everything that could still act on its own.

        The pending question, the last command and the mentioned objects go:
        after a cancel, «повтори» must not re-run what was just aborted, and
        «его» must not point at the thing the user just backed away from. The
        last *answer* stays — «отмена» stops an action, it does not mean the
        user no longer wants to hear what was said.
        """
        with self._lock:
            self._pending = None
            self._command = None
            self._objects = ()
            self._touched_at = None
            self._persist()
        _log.debug("контекст: сброшен по отмене (%s)", reason or "без причины")

    def reset(self, *, profile_id: int | None = None) -> None:
        """Forget everything, including the stored copy.

        Called on a profile switch: another profile has other commands, other
        variables and quite possibly another user, and a pronoun that survived
        the switch would resolve to something that no longer exists.
        """
        with self._lock:
            self._profile_id = profile_id
            self._touched_at = None
            self._command = None
            self._objects = ()
            self._answer = None
            self._pending = None
            self._variables = {}
            self._store_clear()

    # -------------------------------------------------------------- persistence

    def save(self) -> None:
        """Write the context out through the store."""
        with self._lock:
            self._store_save()

    def restore(self) -> bool:
        """Read the context back in. Returns whether anything was restored.

        Expired fields are dropped on the way in, and the pending question is
        never restored: a user who restarts the program has no idea a question
        was outstanding, and their first command would be eaten as its answer.
        A payload from another version, or from another profile, is discarded
        rather than half-read.
        """
        state = self._store_load()
        if not state or state.get("version") != _STATE_VERSION:
            return False
        stored_profile = state.get("profile_id")
        stored_profile = stored_profile if isinstance(stored_profile, int) else None
        with self._lock:
            if self._profile_id is not None and stored_profile != self._profile_id:
                return False
            command = _mapping(state.get("command"))
            answer = _mapping(state.get("answer"))
            self._profile_id = stored_profile
            self._touched_at = _as_time(state.get("touched_at"))
            self._command = LastCommand.from_json(command) if command else None
            self._answer = LastAnswer.from_json(answer) if answer else None
            self._objects = tuple(
                item
                for item in (
                    ContextObject.from_json(raw)
                    for raw in state.get("objects", [])
                    if isinstance(raw, dict)
                )
                if item is not None
            )[:MAX_OBJECTS]
            self._pending = None
            self._prune(self._now())
            return self._command is not None or self._answer is not None or bool(self._objects)

    # ---------------------------------------------------------------------- bus

    def attach(self, bus: EventBus, *, track_answers: bool = True) -> None:
        """Listen for cancels, and optionally for what the assistant says.

        Subscribed with ``weak=False``: the bus holds handlers weakly, and a
        bound method has no other owner, so a weak subscription would be
        collected immediately and «отмена» would silently stop clearing the
        context. :meth:`detach` is therefore not optional.
        """
        with self._lock:
            self.detach()
            self._unsubscribe.append(bus.subscribe(CancelRequested, self._on_cancel, weak=False))
            if track_answers:
                self._unsubscribe.append(
                    bus.subscribe(TtsStarted, self._on_tts_started, weak=False)
                )

    def detach(self) -> None:
        """Stop listening. Idempotent."""
        with self._lock:
            handles, self._unsubscribe = self._unsubscribe, []
        for unsubscribe in handles:
            unsubscribe()

    def _on_cancel(self, event: CancelRequested) -> None:
        self.cancel(reason=event.reason)

    def _on_tts_started(self, event: TtsStarted) -> None:
        """Remember what was said, unless it was a question we are waiting on.

        A clarifying question is spoken like anything else, and recording it as
        the last answer would make «повтори» repeat the question instead of the
        answer — mildly useful, but it would also overwrite the answer the user
        actually asked to hear again.
        """
        with self._lock:
            if self._pending is not None:
                return
        self.remember_answer(event.text)

    # ------------------------------------------------------------------ internal

    def _now(self) -> datetime:
        """Current time from the injected clock, always timezone-aware."""
        now = self._clock()
        return now if now.tzinfo is not None else now.replace(tzinfo=UTC)

    def _window(self) -> WindowInfo | None:
        """Active window, or ``None`` if it cannot be had."""
        try:
            return self._window_probe()
        except Exception:
            _log.exception("контекст: не удалось определить активное окно")
            return None

    def _prune(self, now: datetime) -> None:
        """Drop what has expired. Must be called with the lock held."""
        ttl = self._ttl
        if (
            ttl.followup > 0
            and self._touched_at is not None
            and _elapsed(now, self._touched_at) > ttl.followup
        ):
            self._touched_at = None
            self._command = None
        if ttl.objects > 0 and self._objects:
            self._objects = tuple(
                item for item in self._objects if _elapsed(now, item.at) <= ttl.objects
            )
        if (
            ttl.answer > 0
            and self._answer is not None
            and _elapsed(now, self._answer.at) > ttl.answer
        ):
            self._answer = None
        if (
            ttl.pending > 0
            and self._pending is not None
            and _elapsed(now, self._pending.at) > ttl.pending
        ):
            _log.info("контекст: уточнение «%s» просрочено", self._pending.question)
            self._pending = None

    def _state(self) -> JsonObject:
        """The persistable payload. Must be called with the lock held."""
        return {
            "version": _STATE_VERSION,
            "profile_id": self._profile_id,
            "touched_at": self._touched_at.isoformat() if self._touched_at else None,
            "command": self._command.as_json() if self._command else None,
            "objects": [item.as_json() for item in self._objects],
            "answer": self._answer.as_json() if self._answer else None,
        }

    def _persist(self) -> None:
        """Save if autosave is on. Must be called with the lock held."""
        if self._autosave:
            self._store_save()

    def _store_save(self) -> None:
        """Write through the store, swallowing its failures.

        The shipped stores promise not to raise, but the context is the single
        writer the whole dialogue goes through: a store that breaks its promise
        — a database locked by a backup, a plugin's own store — must cost the
        user a context that does not survive the restart, not «отмена» that
        stops working mid-sentence.
        """
        try:
            self._store.save(self._state())
        except Exception:
            _log.exception("контекст: не удалось сохранить состояние")

    def _store_load(self) -> JsonObject | None:
        """Read through the store, swallowing its failures."""
        try:
            return self._store.load()
        except Exception:
            _log.exception("контекст: не удалось прочитать сохранённое состояние")
            return None

    def _store_clear(self) -> None:
        """Forget the stored copy, swallowing the store's failures."""
        try:
            self._store.clear()
        except Exception:
            _log.exception("контекст: не удалось удалить сохранённое состояние")
