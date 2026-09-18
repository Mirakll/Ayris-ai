"""When a scheduled entry fires next — pure time math, no waiting, no I/O.

One-shot entries carry an absolute ``fire_at``; recurring ones carry a five-field
cron expression that is evaluated in a real time zone so daylight-saving jumps do
not silently drop or double a firing. Everything here is a function of a moment
and a time zone, so it is tested against a frozen clock without any sleeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, tzinfo
from enum import StrEnum
from typing import Final

from ayris.core.errors import AyrisError
from ayris.core.models import Timer, TimerKind, utc_now

__all__ = [
    "CronError",
    "CronExpr",
    "MissedPolicy",
    "cron_daily",
    "cron_weekly",
    "next_fire",
    "remaining",
]

# Bound the search for a cron match so an impossible expression (e.g. Feb 30)
# fails loudly instead of spinning. Four years covers every leap-year cycle.
_MAX_MINUTES_AHEAD: Final = 4 * 366 * 24 * 60


class CronError(AyrisError):
    """A cron expression cannot be parsed."""


class MissedPolicy(StrEnum):
    """What to do at startup with a one-shot entry whose moment already passed."""

    FIRE_NOW = "fire_now"
    MARK_MISSED = "mark_missed"
    SKIP = "skip"


def _parse_field(
    field: str, low: int, high: int, *, names: dict[str, int] | None = None
) -> frozenset[int]:
    """Expand one cron field (``*``, ``a-b``, ``*/n``, ``a,b`` and combinations)."""
    result: set[int] = set()
    for part in field.split(","):
        token = part.strip().lower()
        if not token:
            raise CronError(f"пустое поле в cron: {field!r}")
        step = 1
        if "/" in token:
            token, _, step_text = token.partition("/")
            if not step_text.isdigit() or int(step_text) == 0:
                raise CronError(f"неверный шаг в cron: {part!r}")
            step = int(step_text)
        if token in ("*", ""):
            start, stop = low, high
        elif "-" in token:
            start_text, _, stop_text = token.partition("-")
            start = _field_value(start_text, low, high, names)
            stop = _field_value(stop_text, low, high, names)
        else:
            start = stop = _field_value(token, low, high, names)
        if start > stop:
            raise CronError(f"диапазон задом наперёд в cron: {part!r}")
        result.update(range(start, stop + 1, step))
    return frozenset(result)


def _field_value(text: str, low: int, high: int, names: dict[str, int] | None) -> int:
    token = text.strip().lower()
    if names is not None and token in names:
        return names[token]
    if not token.lstrip("-").isdigit():
        raise CronError(f"неожиданное значение в cron: {text!r}")
    value = int(token)
    if not low <= value <= high:
        raise CronError(f"значение {value} вне диапазона {low}..{high}")
    return value


_MONTHS: Final[dict[str, int]] = {
    name: index
    for index, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
_DOW: Final[dict[str, int]] = {
    name: index for index, name in enumerate(("sun", "mon", "tue", "wed", "thu", "fri", "sat"))
}


@dataclass(frozen=True, slots=True)
class CronExpr:
    """A parsed five-field cron expression: minute hour day-of-month month day-of-week."""

    minutes: frozenset[int]
    hours: frozenset[int]
    days_of_month: frozenset[int]
    months: frozenset[int]
    days_of_week: frozenset[int]
    dom_restricted: bool
    dow_restricted: bool

    @classmethod
    def parse(cls, expression: str) -> CronExpr:
        parts = expression.split()
        if len(parts) != 5:
            raise CronError(f"cron должен содержать 5 полей, получено {len(parts)}: {expression!r}")
        minute, hour, dom, month, dow = parts
        days_of_week = {value % 7 for value in _parse_field(dow, 0, 7, names=_DOW)}
        return cls(
            minutes=_parse_field(minute, 0, 59),
            hours=_parse_field(hour, 0, 23),
            days_of_month=_parse_field(dom, 1, 31),
            months=_parse_field(month, 1, 12, names=_MONTHS),
            days_of_week=frozenset(days_of_week),
            dom_restricted=dom.strip() != "*",
            dow_restricted=dow.strip() != "*",
        )

    def _day_matches(self, moment: datetime) -> bool:
        # Vixie cron: when both day-of-month and day-of-week are restricted, either
        # matching is enough; otherwise the unrestricted field is ignored.
        dow = moment.isoweekday() % 7  # Sunday -> 0
        dom_ok = moment.day in self.days_of_month
        dow_ok = dow in self.days_of_week
        if self.dom_restricted and self.dow_restricted:
            return dom_ok or dow_ok
        if self.dom_restricted:
            return dom_ok
        if self.dow_restricted:
            return dow_ok
        return True

    def matches(self, moment: datetime) -> bool:
        return (
            moment.minute in self.minutes
            and moment.hour in self.hours
            and moment.month in self.months
            and self._day_matches(moment)
        )

    def next_after(self, after: datetime, tz: tzinfo) -> datetime:
        """The first firing strictly after ``after``, resolved in ``tz``.

        Works on the wall clock in ``tz`` and re-localises each candidate, so a
        firing that lands in a daylight-saving gap is pushed to the next real
        minute and one that lands in a fold happens once.
        """
        local = after.astimezone(tz)
        # Walk wall-clock minutes; jump whole fields when they cannot match, so
        # the loop is bounded by field mismatches, not by real minutes elapsed.
        wall = local.replace(second=0, microsecond=0) + timedelta(minutes=1)
        naive = wall.replace(tzinfo=None)
        for _ in range(_MAX_MINUTES_AHEAD):
            if naive.month not in self.months:
                naive = _first_of_next_month(naive)
                continue
            probe = naive.replace(tzinfo=tz)
            if not self._day_matches(probe):
                naive = (naive + timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if naive.hour not in self.hours:
                naive = _next_hour(naive)
                continue
            if naive.minute not in self.minutes:
                naive += timedelta(minutes=1)
                continue
            return _localize(naive, tz)
        raise CronError(f"cron не имеет ближайшего срабатывания: {self!r}")


def _first_of_next_month(naive: datetime) -> datetime:
    year = naive.year + (1 if naive.month == 12 else 0)
    month = 1 if naive.month == 12 else naive.month + 1
    return naive.replace(year=year, month=month, day=1, hour=0, minute=0)


def _next_hour(naive: datetime) -> datetime:
    following = naive.replace(minute=0) + timedelta(hours=1)
    return following


def _localize(naive: datetime, tz: tzinfo) -> datetime:
    """Attach ``tz`` to a wall-clock time, stepping over a DST gap if needed."""
    aware = naive.replace(tzinfo=tz)
    if aware.utcoffset() is None:  # pragma: no cover - all real zones answer
        return aware
    # A time that does not exist (spring-forward gap) round-trips to a different
    # wall clock; advance until the wall clock is real again.
    guard = 0
    while aware.astimezone(tz).replace(tzinfo=None) != naive and guard < 180:
        naive += timedelta(minutes=1)
        aware = naive.replace(tzinfo=tz)
        guard += 1
    return aware


def cron_daily(hour: int, minute: int = 0) -> str:
    """A cron string for «every day at HH:MM»."""
    return f"{minute} {hour} * * *"


def cron_weekly(weekdays: tuple[int, ...], hour: int, minute: int = 0) -> str:
    """A cron string for the given weekdays (0=Sunday..6=Saturday) at HH:MM."""
    if not weekdays:
        return cron_daily(hour, minute)
    days = ",".join(str(day % 7) for day in sorted(set(weekdays)))
    return f"{minute} {hour} * * {days}"


def next_fire(timer: Timer, *, now: datetime | None = None, tz: tzinfo) -> datetime | None:
    """The next moment ``timer`` should fire, or ``None`` for a spent one-shot.

    A recurring entry (``cron`` set) always has a next moment; a one-shot entry
    returns its ``fire_at`` while it is still in the future, and ``None`` once it
    has passed — the caller decides whether that is a miss.
    """
    moment = now or utc_now()
    if timer.cron:
        return CronExpr.parse(timer.cron).next_after(moment, tz)
    if timer.fire_at is None:
        return None
    fire_at = timer.fire_at if timer.fire_at.tzinfo else timer.fire_at.replace(tzinfo=tz)
    return fire_at if fire_at > moment else None


def remaining(timer: Timer, *, now: datetime | None = None, tz: tzinfo) -> timedelta | None:
    """How long until ``timer`` next fires, or ``None`` when there is nothing due."""
    moment = now or utc_now()
    upcoming = next_fire(timer, now=moment, tz=tz)
    if upcoming is None:
        return None
    return max(timedelta(0), upcoming - moment)


def is_recurring(timer: Timer) -> bool:
    return bool(timer.cron)


def default_sound_for(kind: TimerKind) -> str:
    """The library sound name a kind falls back to when the entry named none."""
    return {
        TimerKind.TIMER: "timer_done",
        TimerKind.REMINDER: "reminder",
        TimerKind.ALARM: "alarm",
    }.get(kind, "timer_done")
