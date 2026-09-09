"""Cron and one-shot command triggers on top of the shared timer scheduler."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final, Literal

from ayris.core.pipeline_states import Scheduler, ThreadScheduler, Timer

MissedPolicy = Literal["run_once", "skip"]
_CHECK_INTERVAL: Final = 60.0
_ON_TIME_GRACE: Final = 90.0


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    trigger_id: int
    fire_at: datetime | None = None
    cron: str = ""
    missed: MissedPolicy = "run_once"


def _field(text: str, minimum: int, maximum: int, *, sunday: bool = False) -> frozenset[int]:
    values: set[int] = set()
    for part in text.split(","):
        base, slash, step_text = part.partition("/")
        step = int(step_text) if slash else 1
        if step <= 0:
            raise ValueError("шаг cron должен быть положительным")
        if base == "*":
            start, end = minimum, maximum
        elif "-" in base:
            left, right = base.split("-", 1)
            start, end = int(left), int(right)
        else:
            start = end = int(base)
        if start < minimum or end > maximum or start > end:
            raise ValueError(f"поле cron {text!r} вне диапазона {minimum}..{maximum}")
        values.update(range(start, end + 1, step))
    if sunday and 7 in values:
        values.remove(7)
        values.add(0)
    return frozenset(values)


class CronExpression:
    """Strict five-field local-time cron expression."""

    def __init__(self, expression: str) -> None:
        fields = expression.split()
        if len(fields) != 5:
            raise ValueError("cron должен содержать пять полей")
        self.minutes = _field(fields[0], 0, 59)
        self.hours = _field(fields[1], 0, 23)
        self.days = _field(fields[2], 1, 31)
        self.months = _field(fields[3], 1, 12)
        self.weekdays = _field(fields[4], 0, 7, sunday=True)
        self._any_day = fields[2] == "*"
        self._any_weekday = fields[4] == "*"

    def matches(self, moment: datetime) -> bool:
        cron_weekday = (moment.weekday() + 1) % 7
        day_match = moment.day in self.days
        weekday_match = cron_weekday in self.weekdays
        if self._any_day:
            calendar_match = weekday_match
        elif self._any_weekday:
            calendar_match = day_match
        else:
            calendar_match = day_match or weekday_match
        return (
            moment.minute in self.minutes
            and moment.hour in self.hours
            and moment.month in self.months
            and calendar_match
        )

    def next_after(self, moment: datetime) -> datetime:
        candidate = moment.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(60 * 24 * 366 * 5):
            if self.matches(candidate):
                return candidate
            candidate += timedelta(minutes=1)
        raise ValueError("cron не срабатывает в пределах пяти лет")


class TriggerSchedule:
    """Maintain all schedules with exactly one pending monotonic timer."""

    def __init__(
        self,
        callback: Callable[[int], None],
        *,
        scheduler: Scheduler | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._callback = callback
        self._scheduler = scheduler or ThreadScheduler()
        self._wall_clock = wall_clock or (lambda: datetime.now().astimezone())
        self._monotonic = monotonic
        self._entries: dict[int, ScheduleEntry] = {}
        self._cron: dict[int, CronExpression] = {}
        self._next: dict[int, datetime] = {}
        self._last_wall_key: dict[int, tuple[int, int, int, int, int]] = {}
        self._timer: Timer | None = None
        self._armed_wall: datetime | None = None
        self._armed_mono = 0.0

    def replace(self, entries: Iterable[ScheduleEntry]) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        now = self._wall_clock()
        self._entries = {entry.trigger_id: entry for entry in entries}
        self._cron = {
            trigger_id: CronExpression(entry.cron)
            for trigger_id, entry in self._entries.items()
            if entry.cron
        }
        self._next.clear()
        for trigger_id, entry in self._entries.items():
            if entry.fire_at is not None:
                self._next[trigger_id] = self._compatible(entry.fire_at, now)
            else:
                self._next[trigger_id] = self._cron[trigger_id].next_after(
                    now.replace(second=0, microsecond=0) - timedelta(minutes=1)
                )
        self._arm()

    @property
    def next_fire(self) -> datetime | None:
        return min(self._next.values(), default=None)

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._entries.clear()
        self._next.clear()

    def _wake(self) -> None:
        self._timer = None
        now = self._wall_clock()
        # A suspend or wall-clock correction invalidates the old delay. Due entries
        # are still decided from wall time; monotonic time is only the wakeup device.
        if self._armed_wall is not None:
            wall_elapsed = (now - self._armed_wall).total_seconds()
            mono_elapsed = self._monotonic() - self._armed_mono
            _clock_shifted = abs(wall_elapsed - mono_elapsed) > 2.0
        for trigger_id, due in tuple(self._next.items()):
            if due > now:
                continue
            entry = self._entries[trigger_id]
            lateness = (now - due).total_seconds()
            key = (due.year, due.month, due.day, due.hour, due.minute)
            should_fire = entry.missed == "run_once" or lateness <= _ON_TIME_GRACE
            if should_fire and self._last_wall_key.get(trigger_id) != key:
                self._last_wall_key[trigger_id] = key
                self._callback(trigger_id)
            if entry.cron:
                self._next[trigger_id] = self._cron[trigger_id].next_after(now)
            else:
                self._next.pop(trigger_id, None)
        self._arm()

    def _arm(self) -> None:
        if not self._next:
            return
        now = self._wall_clock()
        delay = max(0.0, (min(self._next.values()) - now).total_seconds())
        self._armed_wall = now
        self._armed_mono = self._monotonic()
        self._timer = self._scheduler.call_later(min(delay, _CHECK_INTERVAL), self._wake)

    @staticmethod
    def _compatible(value: datetime, reference: datetime) -> datetime:
        if value.tzinfo is None and reference.tzinfo is not None:
            return value.replace(tzinfo=reference.tzinfo)
        if value.tzinfo is not None and reference.tzinfo is None:
            return value.replace(tzinfo=None)
        return value
