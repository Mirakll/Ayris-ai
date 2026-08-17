"""When a trigger is allowed to fire at all.

«Сохрани» means one thing in Photoshop and another in a text editor, and the way
to let both exist in one library is to give a trigger conditions: it is only a
candidate when the foreground window is Photoshop, or when it is evening, or when
a variable says the user is at work. This module is those conditions — how they
are written, how they are read out of a trigger's payload, and how they become
the predicate :meth:`ayris.nlu.matcher.Matcher.match` filters with.

**Filtering happens before ranking, not after.** A condition that is checked after
the winner is picked does not disambiguate anything: «сохрани» would match the
Photoshop trigger, win on priority, fail its condition and then have nothing to
fall back to, because the alternative was never scored. So the predicate goes
into the matcher and inactive entries are dropped while candidates are still being
collected — which also means the fuzzy sweep does not spend time on triggers that
cannot fire.

**A malformed condition never fires, and says so once.** The conditions come from
a JSON payload the user edited by hand or a plugin generated, so «``when_time``:
"полшестого"» is a thing that will happen. Parsing is total —
:meth:`TriggerConditions.parse` returns a condition set and a list of Russian
complaints, never raises — and an unparseable condition is dropped rather than
treated as satisfied. The command editor of task 44 shows the complaints; the
matcher just gets a trigger with one fewer condition on it.

**Cost is why the window is asked for once.** :func:`context_predicate` closes
over one :class:`~ayris.nlu.context.ContextSnapshot`, which already holds the
window reading taken at the start of the utterance. Nothing in here calls
:func:`~ayris.nlu.context.get_active_window`, so a library of a thousand
conditional triggers costs one WinAPI round trip per phrase, not a thousand.

**An absent fact fails a condition that needs it.** Off Windows, and on a Windows
that would not say what the foreground window is, the reading is ``None`` and
every ``when_window`` condition is unsatisfied. The alternative — treating an
unknown window as a match — would make «сохрани» in Photoshop fire on a machine
where the probe is broken, which is exactly the case where being wrong is
invisible.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from ayris.nlu.context import TIME_OF_DAY_WORDS, TimeOfDay
from ayris.nlu.normalize import fold_letters
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.nlu.context import ContextSnapshot, WindowInfo
    from ayris.nlu.index import IndexedTrigger

__all__ = [
    "CONDITION_KEYS",
    "KEY_PROCESS",
    "KEY_PROFILE",
    "KEY_TIME",
    "KEY_VARIABLE",
    "KEY_WINDOW",
    "UNCONDITIONAL",
    "HourRange",
    "TriggerConditions",
    "TriggerPredicate",
    "VariableCondition",
    "VariableTest",
    "conditions_from_payload",
    "context_predicate",
    "describe_conditions",
    "validate_conditions",
]

_log = get_logger(__name__)

#: Payload key holding a window-title glob. The specification's example is
#: ``"*Photoshop*"``; a pattern without wildcards is a substring test, which
#: :meth:`ayris.nlu.context.WindowInfo.matches` decides.
KEY_WINDOW: Final = "when_window"
#: Payload key holding an executable name: ``"photoshop.exe"``.
KEY_PROCESS: Final = "when_process"
#: Payload key holding a part of the day or an hour range.
KEY_TIME: Final = "when_time"
#: Payload key holding one variable test, or a list of them.
KEY_VARIABLE: Final = "when_variable"
#: Payload key holding a profile id, or a list of them.
KEY_PROFILE: Final = "when_profile"

#: Every key this module reads. Public so the command editor can tell a condition
#: it does not render from a typo in a key name.
CONDITION_KEYS: Final[frozenset[str]] = frozenset(
    {KEY_WINDOW, KEY_PROCESS, KEY_TIME, KEY_VARIABLE, KEY_PROFILE}
)

#: ``9-18``, ``21-6``, ``9:30-18:00``. The minutes are accepted and ignored: an
#: hour is the granularity a trigger condition needs, and silently rounding is
#: friendlier than refusing a value someone reasonably typed.
_HOUR_RANGE: Final = re.compile(
    r"^\s*(?P<from>\d{1,2})(?::\d{1,2})?\s*[-–—]\s*(?P<to>\d{1,2})(?::\d{1,2})?\s*$"
)


class VariableTest(StrEnum):
    """How a variable's value is compared to the expected one."""

    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    TRUTHY = "truthy"
    FALSY = "falsy"
    EXISTS = "exists"
    MISSING = "missing"


#: What a user may write for a test, including the spellings they will actually
#: write. ``==`` and ``!=`` are here because a payload edited by hand tends to
#: reach for them.
_TEST_WORDS: Final[Mapping[str, VariableTest]] = {
    "eq": VariableTest.EQUALS,
    "==": VariableTest.EQUALS,
    "=": VariableTest.EQUALS,
    "equals": VariableTest.EQUALS,
    "ne": VariableTest.NOT_EQUALS,
    "!=": VariableTest.NOT_EQUALS,
    "<>": VariableTest.NOT_EQUALS,
    "not_equals": VariableTest.NOT_EQUALS,
    "contains": VariableTest.CONTAINS,
    "in": VariableTest.CONTAINS,
    "truthy": VariableTest.TRUTHY,
    "true": VariableTest.TRUTHY,
    "falsy": VariableTest.FALSY,
    "false": VariableTest.FALSY,
    "exists": VariableTest.EXISTS,
    "set": VariableTest.EXISTS,
    "missing": VariableTest.MISSING,
    "unset": VariableTest.MISSING,
}

#: Strings that count as false when a variable is tested for truth. A variable
#: read out of JSON, or typed into the settings window, is a string far more often
#: than it is a boolean, and ``bool("false")`` is ``True``.
_FALSY_WORDS: Final[frozenset[str]] = frozenset(
    {"", "0", "false", "нет", "off", "выкл", "выключено", "none", "null"}
)


@dataclass(frozen=True, slots=True)
class HourRange:
    """Hours of the day a trigger is active in, ``start`` inclusive, ``end`` not.

    Wrapping is the point of having the class at all: ``21-6`` is a night shift,
    and reading it as an empty range would silently disable the trigger.
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if not (0 <= self.start <= 24 and 0 <= self.end <= 24):
            raise ValueError("часы задаются числами 0..24")

    @property
    def wraps(self) -> bool:
        """Whether the range runs through midnight."""
        return self.end <= self.start

    def contains(self, hour: int) -> bool:
        """Whether an hour falls inside the range."""
        hour %= 24
        if self.start == self.end:
            return True
        if self.wraps:
            return hour >= self.start or hour < self.end
        return self.start <= hour < self.end

    def describe(self) -> str:
        """Russian description for the command editor."""
        return f"с {self.start}:00 до {self.end}:00"

    @classmethod
    def parse(cls, raw: str) -> HourRange | None:
        """Read ``9-18`` or ``21:30-6:00``, or ``None`` when it is neither."""
        found = _HOUR_RANGE.match(raw)
        if found is None:
            return None
        start = int(found.group("from"))
        end = int(found.group("to"))
        if start > 24 or end > 24:
            return None
        return cls(start=start % 24, end=end % 24)


@dataclass(frozen=True, slots=True)
class VariableCondition:
    """One test against one profile variable."""

    name: str
    test: VariableTest = VariableTest.TRUTHY
    expected: Any = None

    def holds(self, snapshot: ContextSnapshot) -> bool:
        """Whether the context satisfies this test."""
        missing = object()
        value = snapshot.variable(self.name, missing)
        if self.test is VariableTest.MISSING:
            return value is missing
        if value is missing:
            # Every other test is a statement about a value. Without one there is
            # nothing to be true about, so the condition fails rather than
            # accidentally passing on a `None` that looks falsy.
            return False
        if self.test is VariableTest.EXISTS:
            return True
        if self.test is VariableTest.TRUTHY:
            return _truthy(value)
        if self.test is VariableTest.FALSY:
            return not _truthy(value)
        if self.test is VariableTest.CONTAINS:
            return _text(self.expected) in _text(value)
        equal = _equal(value, self.expected)
        return equal if self.test is VariableTest.EQUALS else not equal

    def describe(self) -> str:
        """Russian description for the command editor."""
        if self.test is VariableTest.TRUTHY:
            return f"переменная {self.name} включена"
        if self.test is VariableTest.FALSY:
            return f"переменная {self.name} выключена"
        if self.test is VariableTest.EXISTS:
            return f"переменная {self.name} задана"
        if self.test is VariableTest.MISSING:
            return f"переменная {self.name} не задана"
        if self.test is VariableTest.CONTAINS:
            return f"переменная {self.name} содержит «{_text(self.expected)}»"
        sign = "=" if self.test is VariableTest.EQUALS else "≠"
        return f"переменная {self.name} {sign} «{_text(self.expected)}»"

    @classmethod
    def parse(cls, raw: Any) -> tuple[VariableCondition | None, str]:
        """Read one test, returning it and a Russian complaint, one of them empty.

        Two shapes, because two kinds of author. A string — ``"режим=работа"``,
        ``"тихий_час"`` — is what someone types by hand. A mapping —
        ``{"name": "режим", "test": "eq", "value": "работа"}`` — is what a plugin
        or the editor generates, and the only shape that can carry a non-string
        expected value.
        """
        if isinstance(raw, str):
            return cls._parse_text(raw)
        if isinstance(raw, Mapping):
            return cls._parse_mapping(raw)
        return None, "условие по переменной должно быть строкой или объектом"

    @classmethod
    def _parse_text(cls, raw: str) -> tuple[VariableCondition | None, str]:
        text = raw.strip()
        if not text:
            return None, "условие по переменной пустое"
        for operator in ("!=", "<>", "==", "="):
            if operator in text:
                name, _, expected = text.partition(operator)
                if not name.strip():
                    return None, f"в условии «{raw}» не указано имя переменной"
                test = _TEST_WORDS[operator]
                return cls(name=name.strip(), test=test, expected=expected.strip()), ""
        if text.startswith("!"):
            stripped = text[1:].strip()
            if not stripped:
                return None, f"в условии «{raw}» не указано имя переменной"
            return cls(name=stripped, test=VariableTest.FALSY), ""
        return cls(name=text, test=VariableTest.TRUTHY), ""

    @classmethod
    def _parse_mapping(cls, raw: Mapping[str, Any]) -> tuple[VariableCondition | None, str]:
        name = raw.get("name") or raw.get("variable") or raw.get("var")
        if not isinstance(name, str) or not name.strip():
            return None, "в условии по переменной не указано имя"
        test_raw = raw.get("test") or raw.get("op") or ""
        if not isinstance(test_raw, str):
            return None, f"условие для переменной {name}: проверка задана не строкой"
        expected = raw.get("value", raw.get("expected"))
        if not test_raw:
            test = VariableTest.EQUALS if expected is not None else VariableTest.TRUTHY
        else:
            found = _TEST_WORDS.get(fold_letters(test_raw).strip())
            if found is None:
                return None, f"условие для переменной {name}: неизвестная проверка «{test_raw}»"
            test = found
        return cls(name=name.strip(), test=test, expected=expected), ""


@dataclass(frozen=True, slots=True)
class TriggerConditions:
    """Everything that has to hold for one trigger to be a candidate.

    Conditions are conjunctive — every one of them must hold — while the values
    *inside* one condition are alternatives: two window patterns mean «either
    window», because that is what a list of windows can sensibly mean, and a
    trigger needing two windows at once needs a different feature.
    """

    windows: tuple[str, ...] = ()
    processes: tuple[str, ...] = ()
    times: tuple[TimeOfDay, ...] = ()
    hours: tuple[HourRange, ...] = ()
    variables: tuple[VariableCondition, ...] = ()
    profiles: tuple[int, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the trigger is unconditional. The overwhelmingly common case."""
        return not (
            self.windows
            or self.processes
            or self.times
            or self.hours
            or self.variables
            or self.profiles
        )

    @property
    def needs_window(self) -> bool:
        """Whether evaluating this needs the foreground window."""
        return bool(self.windows or self.processes)

    def holds(self, snapshot: ContextSnapshot) -> bool:
        """Whether ``snapshot`` satisfies every condition."""
        if self.is_empty:
            return True
        if self.profiles and snapshot.profile_id not in self.profiles:
            return False
        if (self.times or self.hours) and not self._time_holds(snapshot):
            return False
        if self.needs_window and not self._window_holds(snapshot.window):
            return False
        return all(condition.holds(snapshot) for condition in self.variables)

    def _time_holds(self, snapshot: ContextSnapshot) -> bool:
        """Whether the part of the day or the hour range is satisfied.

        The two are alternatives to each other: a trigger written as
        ``["вечером", "9-12"]`` means «evening or late morning», because that is
        one condition with two values and the values inside a condition are
        alternatives.
        """
        if self.times and snapshot.time_of_day in self.times:
            return True
        return any(window.contains(snapshot.hour) for window in self.hours)

    def _window_holds(self, window: WindowInfo | None) -> bool:
        """Whether the foreground window satisfies the window conditions.

        ``None`` fails: see the module docstring. The two keys are alternatives,
        because ``when_window: "*Photoshop*"`` and ``when_process: "photoshop.exe"``
        are two ways of naming the same intention and a user who wrote both meant
        «this application», not «both patterns at once».
        """
        if window is None:
            return False
        if any(window.matches(pattern) for pattern in self.windows):
            return True
        folded = fold_letters(window.process)
        return any(folded == fold_letters(name) for name in self.processes)

    def describe(self) -> str:
        """Russian one-liner for the trigger list. ``""`` when unconditional."""
        return "; ".join(describe_conditions(self))

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> tuple[TriggerConditions, tuple[str, ...]]:
        """Read the conditions out of a trigger payload.

        Returns the conditions and the complaints. Never raises: a payload is
        data, and one bad condition must not stop the library from loading.
        """
        problems: list[str] = []
        windows = _strings(payload.get(KEY_WINDOW), KEY_WINDOW, problems)
        processes = _strings(payload.get(KEY_PROCESS), KEY_PROCESS, problems)
        times, hours = _times(payload.get(KEY_TIME), problems)
        variables = _variables(payload.get(KEY_VARIABLE), problems)
        profiles = _profiles(payload.get(KEY_PROFILE), problems)
        conditions = cls(
            windows=windows,
            processes=processes,
            times=times,
            hours=hours,
            variables=variables,
            profiles=profiles,
        )
        return conditions, tuple(problems)


#: A trigger with no conditions on it. Shared, because the index holds one of
#: these per unconditional trigger and they are all the same object.
UNCONDITIONAL: Final = TriggerConditions()


def conditions_from_payload(payload: Mapping[str, Any]) -> TriggerConditions:
    """The conditions in a payload, complaining to the log rather than returning.

    What the index builder calls: it has nowhere to show a message and the log is
    where a user who wonders why their trigger never fires will be pointed.
    """
    if not payload or CONDITION_KEYS.isdisjoint(payload.keys()):
        return UNCONDITIONAL
    conditions, problems = TriggerConditions.parse(payload)
    for problem in problems:
        _log.warning("условие триггера пропущено: %s", problem)
    return conditions


def validate_conditions(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Complaints about a payload's conditions, for the command editor.

    Called when a trigger is saved, which is the only moment the user can still
    fix it.
    """
    return TriggerConditions.parse(payload)[1]


def describe_conditions(conditions: TriggerConditions) -> tuple[str, ...]:
    """Each condition as a Russian phrase, for the trigger list and DevTools."""
    parts: list[str] = []
    if conditions.windows:
        parts.append("окно: " + ", ".join(conditions.windows))
    if conditions.processes:
        parts.append("процесс: " + ", ".join(conditions.processes))
    if conditions.times:
        parts.append("время: " + ", ".join(item.label for item in conditions.times))
    if conditions.hours:
        parts.append("часы: " + ", ".join(item.describe() for item in conditions.hours))
    parts.extend(item.describe() for item in conditions.variables)
    if conditions.profiles:
        parts.append("профиль: " + ", ".join(str(item) for item in conditions.profiles))
    return tuple(parts)


#: What :meth:`ayris.nlu.matcher.Matcher.match` filters candidates with. Takes an
#: index entry rather than a :class:`~ayris.nlu.matcher.Trigger` so that the
#: predicate can read the conditions the index compiled once, instead of parsing
#: a payload per phrase.
TriggerPredicate = Callable[["IndexedTrigger"], bool]


def context_predicate(
    snapshot: ContextSnapshot | None,
    *,
    extra: TriggerPredicate | None = None,
) -> TriggerPredicate | None:
    """A predicate that keeps the triggers active in ``snapshot``.

    Returns ``None`` when there is nothing to filter by — no snapshot and no
    ``extra`` — so the matcher can skip the per-candidate call entirely rather
    than invoking a predicate that always says yes.

    ``extra`` composes another restriction in: the pipeline uses it to limit
    matching to one command when it is re-running a repeat, and the tests use it
    to stand in for conditions that do not exist yet.
    """
    if snapshot is None:
        return extra

    def predicate(entry: IndexedTrigger) -> bool:
        if not entry.conditions.holds(snapshot):
            return False
        return extra is None or extra(entry)

    return predicate


def _truthy(value: Any) -> bool:
    """Whether a variable's value counts as true.

    A string is judged by :data:`_FALSY_WORDS` rather than by emptiness, because
    the value came from JSON or from a text field and ``"false"`` means false to
    the person who typed it.
    """
    if isinstance(value, str):
        return fold_letters(value).strip() not in _FALSY_WORDS
    if isinstance(value, bool | int | float):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, Sequence | Mapping | set | frozenset):
        return bool(value)
    return True


def _text(value: Any) -> str:
    """A value as folded text, for comparison against a user-written string."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return fold_letters(str(value)).strip()


def _equal(value: Any, expected: Any) -> bool:
    """Whether a variable equals what a condition expects.

    Compared as text unless both sides are numbers. A variable holding ``40`` and
    a condition written ``"40"`` are the same thing to the user, and this is the
    one place where being strict about types would only ever be wrong.
    """
    if isinstance(value, bool) or isinstance(expected, bool):
        return _text(value) == _text(expected)
    if isinstance(value, int | float) and isinstance(expected, int | float):
        return float(value) == float(expected)
    return _text(value) == _text(expected)


def _strings(raw: Any, key: str, problems: list[str]) -> tuple[str, ...]:
    """One string or a list of them, cleaned. Anything else is a complaint."""
    if raw is None:
        return ()
    values = raw if isinstance(raw, list | tuple) else (raw,)
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            problems.append(f"{key}: ожидается строка, получено {type(item).__name__}")
            continue
        cleaned = item.strip()
        if cleaned:
            result.append(cleaned)
    return tuple(result)


def _times(raw: Any, problems: list[str]) -> tuple[tuple[TimeOfDay, ...], tuple[HourRange, ...]]:
    """Parts of the day and hour ranges out of one ``when_time`` value."""
    if raw is None:
        return (), ()
    values = raw if isinstance(raw, list | tuple) else (raw,)
    parts: list[TimeOfDay] = []
    hours: list[HourRange] = []
    for item in values:
        if not isinstance(item, str):
            problems.append(f"{KEY_TIME}: ожидается строка, получено {type(item).__name__}")
            continue
        text = fold_letters(item).strip()
        if not text:
            continue
        part = TIME_OF_DAY_WORDS.get(text)
        if part is not None:
            parts.append(part)
            continue
        span = HourRange.parse(text)
        if span is None:
            problems.append(f"{KEY_TIME}: «{item}» — не время суток и не диапазон часов вида 9-18")
            continue
        hours.append(span)
    return tuple(dict.fromkeys(parts)), tuple(hours)


def _variables(raw: Any, problems: list[str]) -> tuple[VariableCondition, ...]:
    """One variable test or a list of them."""
    if raw is None:
        return ()
    values = raw if isinstance(raw, list | tuple) else (raw,)
    result: list[VariableCondition] = []
    for item in values:
        condition, problem = VariableCondition.parse(item)
        if condition is None:
            problems.append(problem or f"{KEY_VARIABLE}: условие не распознано")
            continue
        result.append(condition)
    return tuple(result)


def _profiles(raw: Any, problems: list[str]) -> tuple[int, ...]:
    """Profile ids a trigger is limited to."""
    if raw is None:
        return ()
    values = raw if isinstance(raw, list | tuple) else (raw,)
    result: list[int] = []
    for item in values:
        if isinstance(item, bool) or not isinstance(item, int | str):
            problems.append(f"{KEY_PROFILE}: ожидается id профиля, получено {item!r}")
            continue
        try:
            result.append(int(item))
        except ValueError:
            problems.append(f"{KEY_PROFILE}: «{item}» не похоже на id профиля")
    return tuple(result)
