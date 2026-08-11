"""Task 16: clock readings and durations, as people say them out loud.

Nothing here reads the system clock. Every test pins ``now`` to a literal moment
with an explicit :class:`~zoneinfo.ZoneInfo`, and that is not tidiness: the CI
runners are UTC, the user's machine is not, and «в семь» resolves differently
depending on which side of noon it was said on. A test that took the real time
would pass here and fail in CI three hours later, or the other way round.

Three decisions are the whole of the module's guessing, and each has its own
group below.

*A bare hour has two readings, and the sooner one wins.* «в семь» said at noon is
19:00; said at eight in the evening it is 07:00 tomorrow, because seven in the
evening has already gone. This is what a user means often enough that guessing is
better than asking, and it is exactly the sort of rule that needs pinning down.

*A stated day skips the guessing.* «завтра в девять» is nine in the morning, not
nine in the evening, because the choice between the two halves is then made from
the start of that day rather than from ``now``.

*A reading and a duration are different things.* «два часа» is a length — two
hours — and «в два часа» is a moment. Getting that backwards makes a timer fire
eleven hours late, so :func:`parse_clock` refuses a bare number unless the
caller says a reading is expected, which is what a ``{time}`` slot does: the
preposition is in the template's literal, on the other side of the brace.

Groups:

* :class:`TestClockForms` — every shape of reading, without resolving it.
* :class:`TestHalfHours` — «полвторого» is 13:30, and the off-by-one behind it.
* :class:`TestMeridiem` — «утра», «вечера», «дня», «ночи» folded onto 24 hours.
* :class:`TestExpected` — a bare hour inside a slot versus in a free phrase.
* :class:`TestResolve` — the two-reading rule, the quiet window, stated days.
* :class:`TestMoment` — the whole path, clock and «через» alike.
* :class:`TestDurations` — «пять минут», «полтора часа», compound lengths.
* :class:`TestRefusals` — what must *not* read as a time.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ayris.nlu.timeparse import (
    ClockRules,
    ClockTime,
    Meridiem,
    apply_meridiem,
    hour_candidates,
    parse_clock,
    parse_duration,
    parse_moment,
    resolve_clock,
)

#: The timezone every test pins to. Moscow has no DST, so a literal moment here
#: means one thing forever — which is the point: an offset that shifts twice a
#: year would make «в семь» resolve differently in March than in October.
MSK = ZoneInfo("Europe/Moscow")

#: Midday, Tuesday 11 August 2026. Noon so that both readings of a bare hour are
#: available: one has passed and one has not, which is the case the resolver's
#: rules are actually about.
NOON = datetime(2026, 8, 11, 12, 0, tzinfo=MSK)

#: Late evening the same day. Every reading of a small hour has gone by, so the
#: answer must roll into tomorrow.
NIGHT = datetime(2026, 8, 11, 22, 30, tzinfo=MSK)

#: Early morning. Nothing has passed yet, so the morning reading wins on its own
#: without the quiet-window rule having to break a tie.
MORNING = datetime(2026, 8, 11, 6, 0, tzinfo=MSK)


class TestClockForms:
    """Reading a dial, before anything is resolved against a date."""

    @pytest.mark.parametrize(
        ("text", "hour", "minute"),
        [
            ("в семь", 7, 0),
            ("в 7", 7, 0),
            ("в семь тридцать", 7, 30),
            ("в 19:30", 19, 30),
            ("19:30", 19, 30),
            ("в 9:05", 9, 5),
            ("в девять часов", 9, 0),
            ("в девять часов пятнадцать минут", 9, 15),
            ("в двадцать один тридцать", 21, 30),
            ("в час", 1, 0),
            ("в полдень", 12, 0),
            ("в полночь", 0, 0),
        ],
    )
    def test_shapes(self, text: str, hour: int, minute: int) -> None:
        clock = parse_clock(text)
        assert clock is not None
        assert (clock.hour, clock.minute) == (hour, minute)

    @pytest.mark.parametrize(
        ("text", "hour", "minute"),
        [
            ("без четверти семь", 6, 45),
            ("без пятнадцати семь", 6, 45),
            ("без двадцати восемь", 7, 40),
            ("без пяти двенадцать", 11, 55),
        ],
    )
    def test_counting_back_from_the_hour(self, text: str, hour: int, minute: int) -> None:
        """«без четверти семь» is 6:45 — the hour named is the one being reached."""
        clock = parse_clock(text)
        assert clock is not None
        assert (clock.hour, clock.minute) == (hour, minute)

    @pytest.mark.parametrize(
        ("text", "offset"),
        [
            ("завтра в девять", 1),
            ("послезавтра в девять", 2),
            ("сегодня в девять", 0),
        ],
    )
    def test_day_words(self, text: str, offset: int) -> None:
        clock = parse_clock(text)
        assert clock is not None
        assert clock.day_offset == offset

    def test_seconds(self) -> None:
        clock = parse_clock("в 9:05:30")
        assert clock is not None
        assert (clock.hour, clock.minute, clock.second) == (9, 5, 30)

    def test_words_consumed_is_reported(self) -> None:
        """A slot has to know how much of its text the reading took."""
        clock = parse_clock("в семь утра")
        assert clock is not None
        assert clock.words == 3


class TestHalfHours:
    """«полвторого» is half *before* two, i.e. 13:30 — the one everyone gets wrong."""

    @pytest.mark.parametrize(
        ("text", "hour", "minute"),
        [
            ("полвторого", 1, 30),
            ("полпервого", 0, 30),
            ("полтретьего", 2, 30),
            ("полседьмого", 6, 30),
            ("полдвенадцатого", 11, 30),
            ("пол седьмого", 6, 30),
        ],
    )
    def test_half_before_the_named_hour(self, text: str, hour: int, minute: int) -> None:
        clock = parse_clock(text)
        assert clock is not None
        assert (clock.hour, clock.minute) == (hour, minute)

    def test_polvtorogo_resolves_to_half_past_one(self) -> None:
        """The checklist item: «полвторого» said at noon is 13:30, not 01:30."""
        assert parse_moment("полвторого", now=NOON) == datetime(2026, 8, 11, 13, 30, tzinfo=MSK)

    def test_half_hour_needs_no_preposition(self) -> None:
        """«пол-» is an exact form on its own — nothing else it could mean."""
        assert parse_clock("полвторого") is not None


class TestMeridiem:
    """The named halves of the day, and folding a spoken hour onto 24."""

    @pytest.mark.parametrize(
        ("hour", "meridiem", "expected"),
        [
            (7, Meridiem.MORNING, 7),
            (7, Meridiem.EVENING, 19),
            (12, Meridiem.MORNING, 12),
            (12, Meridiem.NIGHT, 0),
            (1, Meridiem.NIGHT, 1),
            (3, Meridiem.AFTERNOON, 15),
            (11, Meridiem.EVENING, 23),
        ],
    )
    def test_apply(self, hour: int, meridiem: Meridiem, expected: int) -> None:
        assert apply_meridiem(hour, meridiem) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("в семь утра", datetime(2026, 8, 11, 7, 0, tzinfo=MSK)),
            ("в семь вечера", datetime(2026, 8, 11, 19, 0, tzinfo=MSK)),
            ("в три дня", datetime(2026, 8, 11, 15, 0, tzinfo=MSK)),
            ("в два ночи", datetime(2026, 8, 12, 2, 0, tzinfo=MSK)),
        ],
    )
    def test_named_half_removes_the_guess(self, text: str, expected: datetime) -> None:
        """With the half stated there is only one reading, so no rule applies."""
        assert parse_moment(text, now=MORNING) == expected

    def test_meridiem_alone_marks_a_reading(self) -> None:
        """«семь вечера» without «в» is still a reading: the half of day says so."""
        clock = parse_clock("семь вечера")
        assert clock is not None
        assert clock.hour == 7
        assert clock.meridiem is Meridiem.EVENING

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("полседьмого вечера", datetime(2026, 8, 11, 18, 30, tzinfo=MSK)),
            ("полседьмого утра", datetime(2026, 8, 11, 6, 30, tzinfo=MSK)),
        ],
    )
    def test_half_hour_with_a_named_half(self, text: str, expected: datetime) -> None:
        assert parse_moment(text, now=MORNING) == expected


class TestExpected:
    """A bare hour: refused in a free phrase, accepted inside a ``{time}`` slot."""

    @pytest.mark.parametrize("text", ["семь", "7", "девять", "два часа"])
    def test_refused_without_a_marker(self, text: str) -> None:
        """Otherwise «два часа» — a length — becomes an alarm eleven hours off."""
        assert parse_clock(text) is None

    @pytest.mark.parametrize(
        ("text", "hour"),
        [("семь", 7), ("7", 7), ("девять", 9), ("девять тридцать", 9)],
    )
    def test_accepted_when_the_caller_expects_one(self, text: str, hour: int) -> None:
        """A template written «разбуди в {time}» keeps «в» outside the brace."""
        clock = parse_clock(text, expected=True)
        assert clock is not None
        assert clock.hour == hour

    def test_expected_reaches_parse_moment(self) -> None:
        assert parse_moment("семь", now=NOON) is None
        assert parse_moment("семь", now=NOON, expected=True) == datetime(
            2026, 8, 11, 19, 0, tzinfo=MSK
        )

    def test_expected_does_not_shadow_a_unit(self) -> None:
        """The flag loosens a bare hour and nothing else: «пять минут» is a length."""
        assert parse_clock("пять минут", expected=True) is None
        assert parse_moment("пять минут", now=NOON, expected=True) is None
        assert parse_moment("через пять минут", now=NOON, expected=True) == datetime(
            2026, 8, 11, 12, 5, tzinfo=MSK
        )


class TestResolve:
    """Turning a reading into the moment it points at."""

    @pytest.mark.parametrize(
        ("clock", "expected"),
        [
            (ClockTime(hour=7), (7, 19)),
            # Twelve is ambiguous like any small hour: «в двенадцать» is noon or
            # midnight, and the resolver picks between them by the same rules.
            (ClockTime(hour=12), (0, 12)),
            (ClockTime(hour=0), (0,)),
            (ClockTime(hour=19), (19,)),
            (ClockTime(hour=7, meridiem=Meridiem.EVENING), (19,)),
        ],
    )
    def test_candidates(self, clock: ClockTime, expected: tuple[int, ...]) -> None:
        """A bare hour under twelve stands for two readings; anything else for one."""
        assert hour_candidates(clock) == expected

    def test_afternoon_prefers_the_evening_reading(self) -> None:
        """The checklist item: «в семь» said at noon is 19:00."""
        assert parse_moment("в семь", now=NOON) == datetime(2026, 8, 11, 19, 0, tzinfo=MSK)

    def test_a_passed_hour_rolls_into_tomorrow(self) -> None:
        assert parse_moment("в семь", now=NIGHT) == datetime(2026, 8, 12, 7, 0, tzinfo=MSK)

    def test_morning_prefers_the_morning_reading(self) -> None:
        assert parse_moment("в семь", now=MORNING) == datetime(2026, 8, 11, 7, 0, tzinfo=MSK)

    def test_a_stated_day_skips_the_guessing(self) -> None:
        """«завтра в девять» is the morning: the day was stated, so nine means nine."""
        assert parse_moment("завтра в девять", now=NIGHT) == datetime(2026, 8, 12, 9, 0, tzinfo=MSK)

    def test_a_stated_day_keeps_a_named_half(self) -> None:
        assert parse_moment("завтра в девять вечера", now=NOON) == datetime(
            2026, 8, 12, 21, 0, tzinfo=MSK
        )

    def test_quiet_window_loses_to_a_later_reading(self) -> None:
        """Between two readings, «в три» at midday means the afternoon, not 03:00."""
        assert parse_moment("в три", now=NOON) == datetime(2026, 8, 11, 15, 0, tzinfo=MSK)

    def test_quiet_window_is_configurable(self) -> None:
        rules = ClockRules(quiet_start=0, quiet_end=9)
        clock = parse_clock("в восемь")
        assert clock is not None
        assert resolve_clock(clock, now=MORNING, rules=rules) == datetime(
            2026, 8, 11, 20, 0, tzinfo=MSK
        )

    @pytest.mark.parametrize(
        ("hour", "expected"),
        [(0, True), (3, True), (5, True), (6, False), (12, False), (23, False)],
    )
    def test_quiet_hours(self, hour: int, expected: bool) -> None:
        assert ClockRules().is_quiet(hour) is expected

    def test_wrapping_quiet_window(self) -> None:
        """A window that crosses midnight is the normal case for «не беспокоить»."""
        rules = ClockRules(quiet_start=23, quiet_end=7)
        assert rules.is_quiet(23) is True
        assert rules.is_quiet(2) is True
        assert rules.is_quiet(7) is False

    def test_the_answer_follows_the_timezone_of_now(self) -> None:
        """Nothing consults the system clock, so the tzinfo comes from ``now``."""
        resolved = parse_moment("в семь", now=NOON)
        assert resolved is not None
        assert resolved.tzinfo is MSK

    def test_a_naive_now_gives_a_naive_answer(self) -> None:
        resolved = parse_moment("в семь", now=datetime(2026, 8, 11, 12, 0))
        assert resolved is not None
        assert resolved.tzinfo is None

    def test_microseconds_are_cleared(self) -> None:
        resolved = parse_moment("в семь", now=NOON.replace(microsecond=123_456))
        assert resolved is not None
        assert resolved.microsecond == 0


class TestMoment:
    """The whole path: a reading, or a delay behind «через»."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("через пять минут", datetime(2026, 8, 11, 12, 5, tzinfo=MSK)),
            ("через час", datetime(2026, 8, 11, 13, 0, tzinfo=MSK)),
            ("через полтора часа", datetime(2026, 8, 11, 13, 30, tzinfo=MSK)),
            ("через 30 секунд", datetime(2026, 8, 11, 12, 0, 30, tzinfo=MSK)),
            ("через сутки", datetime(2026, 8, 12, 12, 0, tzinfo=MSK)),
        ],
    )
    def test_delays(self, text: str, expected: datetime) -> None:
        assert parse_moment(text, now=NOON) == expected

    def test_a_bare_duration_is_not_a_moment(self) -> None:
        """«пять минут» is a length; making it a moment is the caller's decision."""
        assert parse_moment("пять минут", now=NOON) is None

    @pytest.mark.parametrize("text", ["", "бубубу", "открой браузер", "громче"])
    def test_nothing_to_read(self, text: str) -> None:
        assert parse_moment(text, now=NOON) is None


class TestDurations:
    """Lengths of time, which a timer and a reminder both need."""

    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("пять минут", 300),
            ("5 минут", 300),
            ("одну минуту", 60),
            ("две минуты", 120),
            ("час", 3_600),
            ("два часа", 7_200),
            ("полтора часа", 5_400),
            ("полчаса", 1_800),
            ("30 секунд", 30),
            ("сутки", 86_400),
            ("день", 86_400),
            ("неделю", 604_800),
        ],
    )
    def test_simple(self, text: str, seconds: int) -> None:
        parsed = parse_duration(text)
        assert parsed is not None
        assert parsed.value == timedelta(seconds=seconds)
        assert parsed.seconds == pytest.approx(seconds)

    def test_polтора_часа_is_ninety_minutes(self) -> None:
        """The checklist item, spelled out on its own."""
        parsed = parse_duration("полтора часа")
        assert parsed is not None
        assert parsed.value == timedelta(minutes=90)

    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("час тридцать минут", 5_400),
            ("два часа пятнадцать минут", 8_100),
            ("одну минуту тридцать секунд", 90),
        ],
    )
    def test_compound(self, text: str, seconds: int) -> None:
        parsed = parse_duration(text)
        assert parsed is not None
        assert parsed.value == timedelta(seconds=seconds)

    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("через пять минут", 300),
            ("на пять минут", 300),
            ("спустя пять минут", 300),
        ],
    )
    def test_leading_prepositions(self, text: str, seconds: int) -> None:
        parsed = parse_duration(text)
        assert parsed is not None
        assert parsed.value == timedelta(seconds=seconds)

    def test_word_count_is_reported(self) -> None:
        parsed = parse_duration("пять минут и что-то ещё")
        assert parsed is not None
        assert parsed.words == 2

    @pytest.mark.parametrize("text", ["", "пять", "минут", "бубубу", "громче"])
    def test_not_a_duration(self, text: str) -> None:
        """A bare number is not a length — the unit is what makes it one."""
        assert parse_duration(text) is None


class TestRefusals:
    """The things that must not read as a time, each for a concrete reason."""

    @pytest.mark.parametrize("text", ["", "   ", "громче", "открой браузер", "в"])
    def test_no_reading(self, text: str) -> None:
        assert parse_clock(text) is None

    @pytest.mark.parametrize("text", ["в 25:00", "в 12:70", "25:00", "в 99"])
    def test_out_of_range(self, text: str) -> None:
        """A recogniser's misheard digit must not become hour 25."""
        assert parse_clock(text) is None

    def test_a_duration_is_not_a_reading(self) -> None:
        assert parse_clock("два часа") is None
        assert parse_duration("два часа") is not None

    def test_a_reading_is_not_a_duration(self) -> None:
        assert parse_clock("в два часа") is not None
