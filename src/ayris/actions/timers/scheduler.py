"""The one scheduler for every timer, reminder and alarm.

State lives in the ``timers`` table; the scheduler keeps only the nearest firing
in memory and a single wait, re-armed whenever the set changes. It never spawns a
thread per entry. Firing publishes :class:`~ayris.core.events.TimerFired` and
hands the entry to a notifier; recurring entries are re-pointed at their next
occurrence, one-shot entries are marked done rather than deleted.

The clock is injectable and the real wait is confined to :meth:`start`, so the
whole of the scheduling logic — recovery, missed handling, recurrence — is tested
against a frozen clock with no sleeping.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta, tzinfo
from typing import Final, Protocol

from ayris.actions.timers.schedule import MissedPolicy, is_recurring, next_fire
from ayris.core.events import EventBus, NotificationRequested, TimerFired
from ayris.core.models import Timer, TimerKind, utc_now
from ayris.core.repositories import Repositories
from ayris.utils.logger import get_logger

__all__ = [
    "ActiveTimer",
    "BusTimerNotifier",
    "TimerNotifier",
    "TimerScheduler",
    "active_scheduler",
    "set_active_scheduler",
]

_log = get_logger(__name__)

# Never sleep longer than this in one hop: a longer wait would not survive the
# machine sleeping, so we re-check on waking rather than trust one long timer.
_MAX_WAIT_SECONDS: Final = 60.0


@dataclass(frozen=True, slots=True)
class ActiveTimer:
    """A live entry as the UI needs it: label, kind and time left."""

    id: int
    label: str
    kind: TimerKind
    fire_at: datetime
    remaining: timedelta

    def remaining_seconds(self, _now: datetime | None = None) -> int:
        return max(0, round(self.remaining.total_seconds()))

    @property
    def due(self) -> datetime:
        return self.fire_at


class TimerNotifier(Protocol):
    """Turns a firing into something the user sees and hears."""

    def notify(self, timer: Timer, *, missed: bool = False) -> None: ...


class BusTimerNotifier:
    """Default notifier: a tray/overlay notification over the event bus.

    Sound and speech are left to whoever subscribes to the notification and to
    :class:`~ayris.core.events.TimerFired`; the scheduler stays free of the audio
    subsystem so it remains testable without one.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def notify(self, timer: Timer, *, missed: bool = False) -> None:
        title = _title_for(timer.kind)
        message = timer.label or title
        if missed:
            message = f"Пропущено: {message}"
        self._bus.publish(
            NotificationRequested(
                title=title,
                message=message,
                level="info",
                action="snooze" if timer.kind != TimerKind.TIMER else "",
            )
        )


def _title_for(kind: TimerKind) -> str:
    return {
        TimerKind.TIMER: "Таймер",
        TimerKind.REMINDER: "Напоминание",
        TimerKind.ALARM: "Будильник",
    }.get(kind, "Таймер")


def _system_tz() -> tzinfo:
    return datetime.now().astimezone().tzinfo or UTC


class TimerScheduler:
    """Owns the wait for the nearest entry and fires it when it comes due."""

    def __init__(
        self,
        repositories: Repositories,
        bus: EventBus,
        *,
        notifier: TimerNotifier | None = None,
        tz: tzinfo | None = None,
        clock: Callable[[], datetime] = utc_now,
        missed_policy: MissedPolicy = MissedPolicy.MARK_MISSED,
        missed_grace: timedelta = timedelta(minutes=15),
    ) -> None:
        self._repos = repositories
        self._bus = bus
        self._notifier = notifier or BusTimerNotifier(bus)
        self._tz = tz or _system_tz()
        self._clock = clock
        self._missed_policy = missed_policy
        self._missed_grace = missed_grace
        self._lock = threading.RLock()
        self._wait: threading.Timer | None = None
        self._started = False

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Recover entries left over from last run, then arm the wait."""
        with self._lock:
            self._started = True
        self.recover()
        self._arm()

    def stop(self) -> None:
        with self._lock:
            self._started = False
            if self._wait is not None:
                self._wait.cancel()
                self._wait = None

    # -- mutation -----------------------------------------------------------

    def add(self, timer: Timer) -> Timer:
        # A recurring entry needs a concrete next moment so «due» and «time left»
        # can read one field; compute it from the cron once, up front.
        if timer.cron and timer.fire_at is None:
            upcoming = next_fire(timer, now=self._clock(), tz=self._tz)
            timer = replace(timer, fire_at=upcoming)
        created = self._repos.timers.create(timer)
        self._arm()
        return created

    def cancel(self, timer_id: int) -> bool:
        deleted = self._repos.timers.delete(timer_id)
        self._arm()
        return deleted

    def cancel_by_label(self, text: str) -> list[int]:
        needle = text.strip().casefold()
        if not needle:
            return []
        cancelled: list[int] = []
        for timer in self._repos.timers.list_all(enabled_only=True):
            matches = timer.id is not None and needle in timer.label.casefold()
            if matches and timer.id is not None and self._repos.timers.delete(timer.id):
                cancelled.append(timer.id)
        self._arm()
        return cancelled

    def edit(self, timer: Timer) -> None:
        self._repos.timers.update(timer)
        self._arm()

    def snooze(self, timer_id: int, minutes: int) -> Timer | None:
        original = self._repos.timers.get(timer_id)
        if original is None:
            return None
        fire_at = self._clock().astimezone(UTC) + timedelta(minutes=max(1, minutes))
        again = Timer(
            label=original.label,
            kind=original.kind,
            fire_at=fire_at,
            sound=original.sound,
            payload=original.payload,
        )
        return self.add(again)

    # -- reads --------------------------------------------------------------

    def active(self, *, now: datetime | None = None) -> list[ActiveTimer]:
        moment = now or self._clock()
        result: list[ActiveTimer] = []
        for timer in self._repos.timers.list_all(enabled_only=True):
            due = self._due_at(timer, moment)
            if due is None or timer.id is None:
                continue
            left = max(timedelta(0), due - moment)
            result.append(ActiveTimer(timer.id, timer.label, timer.kind, due, left))
        result.sort(key=lambda item: item.fire_at)
        return result

    def _due_at(self, timer: Timer, now: datetime) -> datetime | None:
        """The moment this entry is next expected to fire, from its stored state."""
        if timer.fire_at is not None:
            return self._aware(timer.fire_at)
        if timer.cron:
            return next_fire(timer, now=now, tz=self._tz)
        return None

    # Structural match for the overlay's TimerProvider protocol.
    def active_timers(self) -> Sequence[ActiveTimer]:
        return self.active()

    def cancel_timer(self, timer_id: int) -> None:
        self.cancel(timer_id)

    # -- firing -------------------------------------------------------------

    def recover(self, *, now: datetime | None = None) -> None:
        """Deal with entries whose moment passed while the app was not running."""
        moment = now or self._clock()
        for timer in self._repos.timers.list_all(enabled_only=True):
            if timer.id is None:
                continue
            if is_recurring(timer):
                # Re-point at the next occurrence without replaying the ones
                # missed while the app was down — no avalanche of firings.
                upcoming = next_fire(timer, now=moment, tz=self._tz)
                if upcoming is not None:
                    self._repos.timers.reschedule(timer.id, upcoming)
                continue
            if timer.fire_at is None:
                continue
            fire_at = self._aware(timer.fire_at)
            if fire_at > moment:
                continue
            self._handle_missed(timer, fire_at, moment)

    def _handle_missed(self, timer: Timer, fire_at: datetime, now: datetime) -> None:
        assert timer.id is not None
        too_old = now - fire_at > self._missed_grace
        if self._missed_policy is MissedPolicy.SKIP or (
            too_old and self._missed_policy is not MissedPolicy.MARK_MISSED
        ):
            self._repos.timers.set_enabled(timer.id, enabled=False)
            return
        missed = self._missed_policy is MissedPolicy.MARK_MISSED or too_old
        self._fire(timer, missed=missed)

    def fire_due(self, *, now: datetime | None = None) -> list[int]:
        """Fire every enabled entry that is due at ``now``. Returns their ids."""
        moment = now or self._clock()
        fired: list[int] = []
        with self._lock:
            for timer in self._repos.timers.list_all(enabled_only=True):
                if timer.id is None:
                    continue
                due = self._due_at(timer, moment)
                if due is None or due > moment:
                    continue
                self._fire(timer, missed=False)
                fired.append(timer.id)
        return fired

    def _fire(self, timer: Timer, *, missed: bool) -> None:
        assert timer.id is not None
        self._bus.publish(TimerFired(timer_id=timer.id, label=timer.label, kind=str(timer.kind)))
        try:
            self._notifier.notify(timer, missed=missed)
        except Exception:
            # A broken notifier must not wedge the scheduler or lose the firing.
            _log.exception("нотификатор таймера упал на записи %s", timer.id)
        if is_recurring(timer):
            # Advance strictly past now so the same occurrence is not fired twice.
            upcoming = next_fire(timer, now=self._clock(), tz=self._tz)
            if upcoming is not None:
                self._repos.timers.reschedule(timer.id, upcoming)
        else:
            self._repos.timers.set_enabled(timer.id, enabled=False)

    # -- the single wait ----------------------------------------------------

    def _next_moment(self, now: datetime) -> datetime | None:
        moments = [
            due
            for timer in self._repos.timers.list_all(enabled_only=True)
            if (due := self._due_at(timer, now)) is not None
        ]
        return min(moments) if moments else None

    def _arm(self) -> None:
        with self._lock:
            if not self._started:
                return
            if self._wait is not None:
                self._wait.cancel()
                self._wait = None
            now = self._clock()
            upcoming = self._next_moment(now)
            if upcoming is None:
                return
            delay = max(0.0, (upcoming - now).total_seconds())
            delay = min(delay, _MAX_WAIT_SECONDS)
            self._wait = threading.Timer(delay, self._wake)
            self._wait.daemon = True
            self._wait.start()

    def _wake(self) -> None:
        self.fire_due()
        self._arm()

    def _aware(self, value: datetime) -> datetime:
        return value if value.tzinfo else value.replace(tzinfo=UTC)


_ACTIVE: TimerScheduler | None = None
_ACTIVE_LOCK: Final = threading.Lock()


def set_active_scheduler(scheduler: TimerScheduler | None) -> None:
    """Register the scheduler that actions reach through :func:`active_scheduler`."""
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = scheduler


def active_scheduler() -> TimerScheduler | None:
    with _ACTIVE_LOCK:
        return _ACTIVE
