"""Clock readings and durations, as people say them out loud.

Two questions look alike and are not: «поставь таймер на пять минут» asks for a
:class:`~datetime.timedelta`, «разбуди в семь утра» asks for a
:class:`~datetime.datetime`. :func:`parse_duration` answers the first,
:func:`parse_clock` plus :func:`resolve_clock` the second, and
:func:`parse_moment` picks between them for a caller that does not know which
one it was given.

**A clock reading is not a moment yet.** «в семь» names an hour on a dial, and
which of the two sevens the user meant depends on when it was said — in the
afternoon it is 19:00, before dawn it is 07:00. So parsing stops at
:class:`ClockTime`, a reading with everything the phrase actually stated, and
:func:`resolve_clock` turns it into a moment against a ``now`` the caller passes
in. Nothing here reads the system clock, which is what makes the rules testable:
a test pins ``now`` to a fixed instant in a named timezone and the answers stop
depending on the machine that runs it.

**The disambiguation lives in one place.** :class:`ClockRules` and
:func:`resolve_clock` hold every guess the parser makes, and there are exactly
three: a bare hour may mean either half of the day, an hour that has already
passed means tomorrow, and an hour inside the quiet window is not what someone
meant unless there was no other option. Keeping them in a function of their own
means the guesses can be changed, and argued about, without touching the
grammar; keeping them out of :func:`parse_clock` means the grammar can be tested
without a clock at all.

**Russian names the hour it is walking towards.** «полвторого» is half past one,
not half past two — the genitive ordinal points at the hour being approached,
and the same holds for «половина восьмого» (7:30), «четверть девятого» (8:15)
and «без четверти семь» (6:45). Getting this backwards is the single most likely
way to set an alarm an hour off, so all four forms are parsed here rather than
being left to a regex in a command template.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Final

from ayris.nlu.numbers import FRACTION_WORDS, ORDINAL_WORDS, parse_number, tokenize

__all__ = [
    "DAY_WORDS",
    "DEFAULT_CLOCK_RULES",
    "DURATION_UNITS",
    "MERIDIEM_WORDS",
    "ClockRules",
    "ClockTime",
    "Meridiem",
    "ParsedDuration",
    "apply_meridiem",
    "hour_candidates",
    "parse_clock",
    "parse_duration",
    "parse_moment",
    "resolve_clock",
]


class Meridiem(StrEnum):
    """Which part of the day the phrase named, when it named one."""

    NIGHT = "night"
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"


#: Words that pin the half of the day. Both the genitive that follows an hour
#: («в семь утра») and the instrumental that stands on its own («утром»), since
#: dictation produces either.
MERIDIEM_WORDS: Final[dict[str, Meridiem]] = {
    "ночи": Meridiem.NIGHT,
    "ночью": Meridiem.NIGHT,
    "утра": Meridiem.MORNING,
    "утром": Meridiem.MORNING,
    "дня": Meridiem.AFTERNOON,
    "днем": Meridiem.AFTERNOON,
    "полудня": Meridiem.AFTERNOON,
    "вечера": Meridiem.EVENING,
    "вечером": Meridiem.EVENING,
}

#: Day words and how many days they move. «вчера» is here for the history
#: queries of later tasks; a timer will simply never see it.
DAY_WORDS: Final[dict[str, int]] = {
    "позавчера": -2,
    "вчера": -1,
    "сегодня": 0,
    "завтра": 1,
    "послезавтра": 2,
}

#: Prepositions that may stand before a clock reading and mean nothing by
#: themselves. Their presence is what tells «в два часа» (a moment) from «два
#: часа» (a duration), so they are consumed rather than ignored. ``часов`` is
#: among them because of «часов в семь», where it stands in front and softens
#: the hour instead of naming a unit.
_CLOCK_PREPOSITIONS: Final[frozenset[str]] = frozenset(
    {"в", "во", "к", "ко", "около", "примерно", "ровно", "часов"}
)

#: Prepositions that may stand before a duration. «в течение» arrives as two
#: tokens, so both halves are listed.
_DURATION_PREFIXES: Final[frozenset[str]] = frozenset(
    {"через", "спустя", "на", "за", "в", "течение", "примерно", "около", "ровно"}
)

#: Prefixes that make a duration into a moment: «через десять секунд».
_DELAY_PREFIXES: Final[frozenset[str]] = frozenset({"через", "спустя"})

#: Forms of «час» that may follow the hour in a clock reading, and mean 1 when
#: they stand in place of it: «в час дня».
_HOUR_WORDS: Final[frozenset[str]] = frozenset({"час", "часа", "часов", "часу"})

#: Forms of «минута» that close the minutes of a clock reading.
_MINUTE_WORDS: Final[frozenset[str]] = frozenset({"минута", "минуты", "минуту", "минут", "мин"})

#: How the units of a duration are spelled, in the cases dictation produces.
#: Months and years are deliberately absent: neither has a fixed length, and a
#: :class:`~datetime.timedelta` that pretends otherwise is a bug with a delay.
_UNIT_FORMS: Final[tuple[tuple[timedelta, tuple[str, ...]], ...]] = (
    (
        timedelta(seconds=1),
        ("секунда", "секунды", "секунду", "секунд", "секундам", "секундами", "сек"),
    ),
    (timedelta(minutes=1), ("минута", "минуты", "минуту", "минут", "минутам", "минутами", "мин")),
    (timedelta(hours=1), ("час", "часа", "часов", "часу", "часам", "часами")),
    (
        timedelta(days=1),
        ("день", "дня", "дней", "дню", "дням", "сутки", "суток", "суткам", "сутками"),
    ),
    (timedelta(weeks=1), ("неделя", "недели", "неделю", "недель", "неделям")),
)

#: Every duration unit form, flattened for lookup.
DURATION_UNITS: Final[dict[str, timedelta]] = {
    form: delta for delta, forms in _UNIT_FORMS for form in forms
}

#: The unit forms that stand for one of themselves when said with no number in
#: front: «поставь таймер на минуту», «через час», «на сутки». Only the singular
#: nominative and accusative, because a bare genitive plural is the tail of a
#: phrase rather than a length of time — reading «минут» as one minute would turn
#: a half-heard «на пять минут» into a silent one-minute timer.
_LONE_UNITS: Final[frozenset[str]] = frozenset(
    {
        "секунда",
        "секунду",
        "минута",
        "минуту",
        "час",
        "часу",
        "день",
        "сутки",
        "неделя",
        "неделю",
    }
)

#: A reading written in digits: ``19:30``, ``9:05``, ``07:00:30``.
_HHMM: Final = re.compile(r"^(?P<hour>\d{1,2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?$")

#: Nouns that name a fixed hour: «полдень», «полночь». Both arrive split by the
#: tokeniser, which separates the ``пол`` prefix from whatever follows it. The
#: half of the day comes along because these readings are not ambiguous — noon
#: is 12:00 and nothing else — and without it 12 would still offer two hours.
_NAMED_HOURS: Final[dict[str, tuple[int, Meridiem]]] = {
    "ночь": (0, Meridiem.NIGHT),
    "ночи": (0, Meridiem.NIGHT),
    "день": (12, Meridiem.AFTERNOON),
    "дня": (12, Meridiem.AFTERNOON),
}


@dataclass(frozen=True, slots=True)
class ParsedDuration:
    """A length of time, and how many tokens it took to say it."""

    value: timedelta
    words: int

    @property
    def seconds(self) -> float:
        """The duration in seconds, for callers scheduling a timer."""
        return self.value.total_seconds()


@dataclass(frozen=True, slots=True)
class ClockTime:
    """A reading off a dial: only what the phrase actually said.

    ``hour`` is as spoken, so 1 to 12 for the ambiguous forms; ``meridiem`` and
    ``day_offset`` are ``None`` when the phrase did not name them, which is the
    difference between «в семь утра» (one possible moment) and «в семь» (two,
    and :func:`resolve_clock` chooses).
    """

    hour: int
    minute: int = 0
    second: int = 0
    meridiem: Meridiem | None = None
    day_offset: int | None = None
    words: int = 0

    @property
    def is_exact(self) -> bool:
        """Whether the reading names a single hour of the day on its own."""
        return len(hour_candidates(self)) == 1


@dataclass(frozen=True, slots=True)
class ClockRules:
    """Everything the resolver guesses when the phrase left it a choice.

    ``quiet_start`` and ``quiet_end`` bound the hours a bare numeral is assumed
    *not* to mean: at 17:00 «в четыре» is tomorrow afternoon rather than four in
    the morning, because nobody sets an alarm eleven hours out by saying the
    hour alone. The window is only a preference — «в четыре утра» names its half
    of the day explicitly and lands where it was told to.
    """

    quiet_start: int = 0
    quiet_end: int = 6

    def is_quiet(self, hour: int) -> bool:
        """Whether ``hour`` falls inside the quiet window, wrap-around included."""
        if self.quiet_start <= self.quiet_end:
            return self.quiet_start <= hour < self.quiet_end
        return hour >= self.quiet_start or hour < self.quiet_end


#: The rules used when a caller does not care to pass its own.
DEFAULT_CLOCK_RULES: Final = ClockRules()


def apply_meridiem(hour: int, meridiem: Meridiem) -> int:
    """Fold a spoken hour onto the 24-hour clock using the named half of day.

    «двенадцать ночи» is midnight and «двенадцать дня» is noon, which is the one
    place the arithmetic is not simply plus twelve. An hour already past twelve
    is left alone: «в 19 вечера» is redundant, not thirty-one o'clock.
    """
    if hour > 12:
        return hour
    if hour == 12:
        # Only the night reading wraps. «двенадцать утра» is noon, not midnight —
        # midnight is «двенадцать ночи» and nothing else, so grouping twelve with
        # the morning would move a lunchtime reminder twelve hours back.
        return 0 if meridiem is Meridiem.NIGHT else 12
    if meridiem in (Meridiem.NIGHT, Meridiem.MORNING):
        return hour
    return hour + 12


def hour_candidates(clock: ClockTime) -> tuple[int, ...]:
    """The 24-hour readings ``clock`` could stand for, earliest first.

    One when the phrase settled the question — a named half of the day, or an
    hour above twelve — and two when it did not.
    """
    if clock.meridiem is not None:
        return (apply_meridiem(clock.hour, clock.meridiem),)
    if clock.hour > 12 or clock.hour == 0:
        return (clock.hour,)
    if clock.hour == 12:
        return (0, 12)
    return (clock.hour, clock.hour + 12)


def resolve_clock(
    clock: ClockTime,
    *,
    now: datetime,
    rules: ClockRules = DEFAULT_CLOCK_RULES,
) -> datetime:
    """Turn a reading into the moment it points at.

    Three rules, and they are the whole of the module's guessing:

    1. An hour that has already passed today means tomorrow — «в семь» said at
       eight in the evening is seven in the morning.
    2. A bare hour takes whichever of its two readings comes sooner, so «в семь»
       in the afternoon is 19:00.
    3. Between two readings, one inside the quiet window loses to one outside
       it, however much later it falls.

    A phrase that named its day skips all three: «завтра в девять» is nine in
    the morning tomorrow, because the day was stated and the choice of half is
    then made from the start of that day rather than from ``now``.

    Args:
        clock: The reading, as :func:`parse_clock` returned it.
        now: The moment to resolve against. Timezone-aware or naive; the answer
            follows whichever ``now`` is, and nothing here consults the system
            clock.
        rules: The quiet window. Defaults to :data:`DEFAULT_CLOCK_RULES`.

    Returns:
        The moment, with seconds as spoken and microseconds cleared.
    """
    if clock.day_offset is not None:
        base = (now + timedelta(days=clock.day_offset)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        reference = base
        roll = False
    else:
        base = now
        reference = now
        roll = True

    best: datetime | None = None
    best_rank: tuple[int, timedelta] | None = None
    for hour in hour_candidates(clock):
        moment = base.replace(hour=hour, minute=clock.minute, second=clock.second, microsecond=0)
        if roll and moment <= now:
            moment += timedelta(days=1)
        rank = (int(rules.is_quiet(hour)), moment - reference)
        if best_rank is None or rank < best_rank:
            best, best_rank = moment, rank

    # ``hour_candidates`` never returns an empty tuple, so a resolved moment
    # always exists; mypy needs to be told in a way that survives a refactor.
    if best is None:  # pragma: no cover - unreachable while candidates are non-empty
        return base
    return best


def _read_fraction_minutes(word: str) -> int | None:
    """Read «четверти» or «половины» as minutes, for the «без …» forms."""
    fraction = FRACTION_WORDS.get(word)
    if fraction is None:
        return None
    return int(fraction * 60)


def _parse_to_hour(words: list[str], index: int) -> tuple[int, int, int, int] | None:
    """Read the forms built on the *upcoming* hour, or return ``None``.

    Covers «без четверти семь» and «без десяти семь» (6:45 and 6:50), and
    «половина восьмого», «полвторого», «четверть девятого» (7:30, 1:30, 8:15).
    All of them name the hour being approached, so the answer is one less than
    the ordinal — the mistake that sets an alarm an hour late.
    """
    word = words[index]

    if word == "без" and index + 2 < len(words):
        minutes = _read_fraction_minutes(words[index + 1])
        consumed = index + 2
        if minutes is None:
            # The last token is held back: the minutes and the hour are both
            # numerals, and «двадцати восемь» composes to 28 the moment the whole
            # tail is offered — eating the hour this form exists to name. The
            # guard above leaves at least one token in the slice, and «без
            # двадцати пяти восемь» still composes its 25 out of two.
            number = parse_number(" ".join(words[index + 1 : -1]))
            if number is None or number.as_int is None:
                return None
            minutes = number.as_int
            consumed = index + 1 + number.words
        if consumed >= len(words) or not 0 < minutes < 60:
            return None
        target = parse_number(" ".join(words[consumed:]), allow_ordinal=True)
        if target is None or target.as_int is None:
            return None
        hour = (target.as_int - 1) % 24
        return hour, 60 - minutes, 0, consumed + target.words

    fraction = FRACTION_WORDS.get(word)
    if fraction is not None and index + 1 < len(words):
        target_ordinal = ORDINAL_WORDS.get(words[index + 1])
        if target_ordinal is None:
            return None
        return (target_ordinal - 1) % 24, int(fraction * 60), 0, index + 2

    return None


def _parse_digital(word: str) -> tuple[int, int, int] | None:
    """Read ``19:30`` and its relatives, or return ``None``."""
    found = _HHMM.match(word)
    if found is None:
        return None
    hour = int(found.group("hour"))
    minute = int(found.group("minute"))
    second = int(found.group("second") or 0)
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour, minute, second


def _parse_spoken_hour(
    words: list[str],
    index: int,
    *,
    marked: bool = False,
) -> tuple[int, int, int, int, bool] | None:
    """Read «семь», «семь тридцать», «семь часов тридцать минут», «час».

    The trailing flag says whether the reading carried a marker of its own — a
    unit or explicit minutes. Without one, «два часа» and «две минуты» are told
    apart only by the preposition in front, which the caller checks.

    Args:
        words: The tokenised phrase.
        index: Where the reading is expected to start.
        marked: Whether the caller has already seen something that makes this a
            reading. Only «дня» needs it, and it needs it badly: the word is both
            a half of the day and a unit of duration, so «в три дня» is 15:00 and
            «три дня» is three days.
    """
    number = parse_number(" ".join(words[index:]))
    if number is None:
        # «в час дня»: the unit word stands in for the numeral one. Unmarked on
        # its own, because a bare «час» is one hour, not one o'clock.
        if words[index] in _HOUR_WORDS:
            return 1, 0, 0, index + 1, False
        return None

    hour = number.as_int
    if hour is None or not 0 <= hour <= 23:
        return None
    cursor = index + number.words
    marked_here = False

    if cursor < len(words):
        unit = words[cursor]
        if unit in _HOUR_WORDS:
            # Consumed but not counted as a marker: «два часа» is a duration
            # until a preposition or a half of the day says otherwise.
            cursor += 1
        elif unit in DURATION_UNITS and (unit not in MERIDIEM_WORDS or not marked):
            # «пять минут» is a duration however it is punctuated; reading it as
            # five o'clock would turn a timer into an alarm. The one exception is
            # «дня», which is also a half of the day: «в три дня» is 15:00 and
            # «три дня» is three days, so a marker is what decides. Left in place
            # either way, for the caller's meridiem loop to read.
            return None

    if cursor >= len(words):
        return hour, 0, 0, cursor, marked_here

    minutes = parse_number(" ".join(words[cursor:]))
    if minutes is None or minutes.as_int is None or not 0 <= minutes.as_int <= 59:
        return hour, 0, 0, cursor, marked_here
    cursor += minutes.words
    if cursor < len(words) and words[cursor] in _MINUTE_WORDS:
        cursor += 1
    return hour, minutes.as_int, 0, cursor, True


def parse_clock(text: str, *, expected: bool = False) -> ClockTime | None:
    """Read a clock reading out of ``text``, or return ``None``.

    Accepts what people say to an assistant: «в семь утра», «в 19:30», «завтра в
    девять», «полвторого», «без четверти семь», «в полдень», «в час дня».

    A bare «два часа» is *not* a reading — it is two hours, and only the
    preposition, a named half of the day or explicit minutes make it one. That
    refusal is deliberate: :func:`parse_moment` falls through to
    :func:`parse_duration`, and a timer misread as an alarm fires eleven hours
    late.

    Args:
        text: The phrase, or the stretch of it a slot captured.
        expected: Whether the caller already knows a clock reading belongs here.
            Set by :class:`ayris.nlu.slot_types.TimeType`, because a template
            written «разбуди в {time}» keeps the preposition in its literal and
            hands the slot a bare «семь» — the marker the refusal above looks for
            is present, just on the other side of the brace. Off by default:
            :func:`parse_moment` must keep falling through to a duration.

    Returns:
        The reading, with only what the phrase stated filled in, or ``None``.
    """
    words = tokenize(text)
    index = 0
    day_offset: int | None = None
    marked = expected

    while index < len(words):
        word = words[index]
        if word in DAY_WORDS:
            day_offset = DAY_WORDS[word]
            marked = True
            index += 1
            continue
        if word in _CLOCK_PREPOSITIONS:
            marked = True
            index += 1
            continue
        break

    if index >= len(words):
        return None

    hour, minute, second = 0, 0, 0
    exact_form = False

    named = None
    if words[index] == "пол" and index + 1 < len(words):
        named = _NAMED_HOURS.get(words[index + 1])
    to_hour = _parse_to_hour(words, index)
    digital = _parse_digital(words[index])
    named_meridiem: Meridiem | None = None

    if named is not None:
        hour, named_meridiem = named
        exact_form = True
        index += 2
    elif to_hour is not None:
        hour, minute, second, index = to_hour
        exact_form = True
    elif digital is not None:
        hour, minute, second = digital
        exact_form = True
        index += 1
    else:
        spoken = _parse_spoken_hour(words, index, marked=marked)
        if spoken is None:
            return None
        hour, minute, second, index, spoken_marked = spoken
        marked = marked or spoken_marked

    meridiem: Meridiem | None = named_meridiem
    while index < len(words):
        word = words[index]
        if word in MERIDIEM_WORDS and meridiem is None:
            meridiem = MERIDIEM_WORDS[word]
            index += 1
            continue
        if word in DAY_WORDS and day_offset is None:
            day_offset = DAY_WORDS[word]
            index += 1
            continue
        break

    if not exact_form and not marked and meridiem is None:
        return None

    return ClockTime(
        hour=hour,
        minute=minute,
        second=second,
        meridiem=meridiem,
        day_offset=day_offset,
        words=index,
    )


def parse_duration(text: str) -> ParsedDuration | None:
    """Read a length of time out of ``text``, or return ``None``.

    Accepts «пять минут», «полтора часа», «полчаса», «два с половиной часа»,
    «через десять секунд», «час тридцать минут» and a bare «час», which is one
    hour the same way «минуту» is one minute. Units add up, so a phrase may name
    several of them. A unit with no number in front counts as one only in the
    forms listed in :data:`_LONE_UNITS`; «минут» on its own is a fragment.

    Returns:
        The duration and how many tokens it consumed, or ``None`` when the text
        does not start with one.
    """
    words = tokenize(text)
    index = 0
    while index < len(words) and words[index] in _DURATION_PREFIXES:
        index += 1

    total = timedelta()
    seen = False

    while index < len(words):
        number = parse_number(" ".join(words[index:]))
        if number is not None:
            unit_index = index + number.words
            if unit_index >= len(words):
                break
            unit = DURATION_UNITS.get(words[unit_index])
            if unit is None:
                break
            # Decimal all the way to microseconds: a third of an hour через
            # float lands on 19:59.999999, and a timer that fires a microsecond
            # early is a test that fails once a week.
            micros = Decimal(unit.total_seconds()) * number.value * 1_000_000
            total += timedelta(microseconds=int(round(micros)))
            index = unit_index + 1
            seen = True
            continue

        unit = DURATION_UNITS.get(words[index]) if words[index] in _LONE_UNITS else None
        if unit is None:
            break
        total += unit
        index += 1
        seen = True

    if not seen:
        return None
    return ParsedDuration(value=total, words=index)


def parse_moment(
    text: str,
    *,
    now: datetime,
    rules: ClockRules = DEFAULT_CLOCK_RULES,
    expected: bool = False,
) -> datetime | None:
    """Read the moment ``text`` points at, however it was expressed.

    A clock reading is resolved against ``now``; a duration behind «через» is
    added to it. Anything else — a bare duration, a phrase with no time in it —
    returns ``None``, because «пять минут» is a length and turning it into a
    moment is the caller's decision, not the parser's.

    Args:
        text: The phrase, or the stretch of it a slot captured.
        now: The moment to resolve against. Nothing here reads the system clock.
        rules: The quiet window used to pick between two readings of a bare hour.
        expected: Passed through to :func:`parse_clock`, which then reads a bare
            «семь» as 07:00 — right for a ``{time}`` slot, whose preposition sits
            in the template's literal, and wrong for a free phrase. A unit word
            still outranks the flag, so «пять минут» stays a length either way.
    """
    clock = parse_clock(text, expected=expected)
    if clock is not None:
        return resolve_clock(clock, now=now, rules=rules)

    words = tokenize(text)
    if not words or words[0] not in _DELAY_PREFIXES:
        return None
    duration = parse_duration(text)
    if duration is None:
        return None
    return (now + duration.value).replace(microsecond=0)
