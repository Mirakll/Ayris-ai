"""Scheduler, cron maths and timer actions, driven by a frozen clock.

No test sleeps: the clock is a value the test advances by hand, and the single
real wait in :meth:`TimerScheduler.start` is never armed here. Cron is checked
across a DST boundary; missed-firing recovery is checked for every policy.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from ayris.actions.timers.actions import CancelTimer, ListTimers, SetAlarm, SetReminder, SetTimer
from ayris.actions.timers.schedule import (
    CronError,
    CronExpr,
    MissedPolicy,
    cron_daily,
    cron_weekly,
    next_fire,
)
from ayris.actions.timers.scheduler import (
    TimerScheduler,
    active_scheduler,
    set_active_scheduler,
)
from ayris.core.database import Database, reset_database
from ayris.core.events import Event, EventBus, TimerFired
from ayris.core.models import Timer, TimerKind
from ayris.core.repositories import Repositories
from ayris.nlu.timeparse import parse_duration, parse_moment

pytestmark = pytest.mark.unit

MSK = ZoneInfo("Europe/Moscow")  # no DST since 2014
BERLIN = ZoneInfo("Europe/Berlin")


class Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def notify(self, timer: Timer, *, missed: bool = False) -> None:
        assert timer.id is not None
        self.calls.append((timer.id, missed))


@pytest.fixture
def database(tmp_path) -> Iterator[Database]:  # type: ignore[no-untyped-def]
    handle = Database.open(tmp_path / "timers.db")
    yield handle
    handle.close()
    reset_database()


def _make(
    database: Database,
    clock: Clock,
    *,
    policy: MissedPolicy = MissedPolicy.MARK_MISSED,
    grace: timedelta = timedelta(minutes=15),
) -> tuple[TimerScheduler, list[Event], RecordingNotifier]:
    bus = EventBus(thread_id=None)
    events: list[Event] = []
    bus.subscribe(Event, events.append)
    notifier = RecordingNotifier()
    scheduler = TimerScheduler(
        Repositories(database),
        bus,
        notifier=notifier,
        tz=MSK,
        clock=clock,
        missed_policy=policy,
        missed_grace=grace,
    )
    return scheduler, events, notifier


# --------------------------------------------------------------------------- #
# Cron and one-shot maths
# --------------------------------------------------------------------------- #


def test_cron_daily_next_occurrence() -> None:
    start = datetime(2026, 6, 1, 6, 0, tzinfo=MSK)
    got = CronExpr.parse(cron_daily(7, 30)).next_after(start, MSK)
    assert got == datetime(2026, 6, 1, 7, 30, tzinfo=MSK)


def test_cron_rolls_to_next_day_when_time_passed() -> None:
    start = datetime(2026, 6, 1, 8, 0, tzinfo=MSK)
    got = CronExpr.parse(cron_daily(7, 30)).next_after(start, MSK)
    assert got == datetime(2026, 6, 2, 7, 30, tzinfo=MSK)


def test_cron_weekly_picks_the_named_weekday() -> None:
    # 2026-06-01 is a Monday; ask for Wednesday (cron dow 3).
    start = datetime(2026, 6, 1, 9, 0, tzinfo=MSK)
    got = CronExpr.parse(cron_weekly((3,), 9, 0)).next_after(start, MSK)
    assert got.isoweekday() == 3
    assert got == datetime(2026, 6, 3, 9, 0, tzinfo=MSK)


def test_cron_step_and_range_parse() -> None:
    expr = CronExpr.parse("*/15 9-10 * * *")
    assert expr.minutes == frozenset({0, 15, 30, 45})
    assert expr.hours == frozenset({9, 10})


def test_cron_rejects_wrong_field_count() -> None:
    with pytest.raises(CronError):
        CronExpr.parse("* * * *")


def test_cron_survives_dst_and_stays_aware() -> None:
    # Europe/Berlin springs forward 02:00 -> 03:00 on 2026-03-29. A daily noon
    # alarm crossing that night keeps advancing one day and stays aware, and its
    # UTC offset flips from +01:00 (winter) to +02:00 (summer).
    expr = CronExpr.parse(cron_daily(12, 0))
    winter = expr.next_after(datetime(2026, 3, 27, 13, 0, tzinfo=BERLIN), BERLIN)
    summer = expr.next_after(winter, BERLIN)
    assert winter == datetime(2026, 3, 28, 12, 0, tzinfo=BERLIN)
    assert summer == datetime(2026, 3, 29, 12, 0, tzinfo=BERLIN)
    assert winter.utcoffset() != summer.utcoffset()  # DST actually changed


def test_cron_in_dst_gap_returns_a_real_aware_time() -> None:
    # 02:30 does not exist on the spring-forward night; the result must still be a
    # real, timezone-aware moment strictly after the start, not a crash.
    expr = CronExpr.parse("30 2 * * *")
    start = datetime(2026, 3, 29, 0, 0, tzinfo=BERLIN)
    got = expr.next_after(start, BERLIN)
    assert got.tzinfo is not None
    assert got > start
    assert got.astimezone(BERLIN).replace(tzinfo=None) == got.replace(tzinfo=None)


def test_next_fire_oneshot_future_and_past() -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=MSK)
    future = Timer(label="t", fire_at=now + timedelta(minutes=5))
    past = Timer(label="t", fire_at=now - timedelta(minutes=5))
    assert next_fire(future, now=now, tz=MSK) == future.fire_at
    assert next_fire(past, now=now, tz=MSK) is None


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #


def test_active_lists_entries_with_time_left(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    scheduler, _events, _notifier = _make(database, Clock(now))
    scheduler.add(Timer(label="Чай", fire_at=now + timedelta(seconds=300)))
    active = scheduler.active(now=now)
    assert len(active) == 1
    assert active[0].label == "Чай"
    assert active[0].remaining_seconds() == 300


def test_oneshot_fires_once_then_is_disabled(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    scheduler, events, notifier = _make(database, Clock(now))
    scheduler.add(Timer(label="Готово", fire_at=now))
    fired = scheduler.fire_due(now=now)
    assert len(fired) == 1
    assert any(isinstance(event, TimerFired) for event in events)
    assert notifier.calls == [(fired[0], False)]
    assert scheduler.fire_due(now=now) == []
    assert scheduler.active(now=now) == []


def test_recurring_reschedules_and_does_not_double_fire(database: Database) -> None:
    clock = Clock(datetime(2026, 6, 1, 6, 59, tzinfo=MSK))
    scheduler, _events, notifier = _make(database, clock)
    created = scheduler.add(Timer(label="Подъём", kind=TimerKind.ALARM, cron=cron_daily(7, 0)))
    assert created.id is not None
    repo = Repositories(database).timers
    assert repo.get(created.id).fire_at == datetime(2026, 6, 1, 7, 0, tzinfo=MSK)  # type: ignore[union-attr]
    clock.now = datetime(2026, 6, 1, 7, 0, tzinfo=MSK)
    assert scheduler.fire_due(now=clock.now) == [created.id]
    stored = repo.get(created.id)
    assert stored is not None and stored.enabled
    assert stored.fire_at == datetime(2026, 6, 2, 7, 0, tzinfo=MSK)
    assert scheduler.fire_due(now=clock.now) == []  # not due again today
    assert notifier.calls == [(created.id, False)]


def test_recover_mark_missed_fires_with_flag(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    repo = Repositories(database).timers
    created = repo.create(Timer(label="Позвонить", fire_at=now - timedelta(hours=1)))
    scheduler, _events, notifier = _make(database, Clock(now), policy=MissedPolicy.MARK_MISSED)
    scheduler.recover(now=now)
    assert notifier.calls == [(created.id, True)]
    assert repo.get(created.id).enabled is False  # type: ignore[union-attr]


def test_recover_skip_does_not_fire(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    repo = Repositories(database).timers
    created = repo.create(Timer(label="Позвонить", fire_at=now - timedelta(minutes=1)))
    scheduler, _events, notifier = _make(database, Clock(now), policy=MissedPolicy.SKIP)
    scheduler.recover(now=now)
    assert notifier.calls == []
    assert repo.get(created.id).enabled is False  # type: ignore[union-attr]


def test_recover_fire_now_within_grace_but_skips_stale(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    repo = Repositories(database).timers
    fresh = repo.create(Timer(label="Свежий", fire_at=now - timedelta(minutes=1)))
    stale = repo.create(Timer(label="Старый", fire_at=now - timedelta(hours=2)))
    scheduler, _events, notifier = _make(
        database, Clock(now), policy=MissedPolicy.FIRE_NOW, grace=timedelta(minutes=15)
    )
    scheduler.recover(now=now)
    assert notifier.calls == [(fresh.id, False)]
    assert repo.get(stale.id).enabled is False  # type: ignore[union-attr]


def test_snooze_creates_a_new_entry(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    scheduler, _events, _notifier = _make(database, Clock(now))
    first = scheduler.add(Timer(label="Чай", fire_at=now))
    assert first.id is not None
    again = scheduler.snooze(first.id, 5)
    assert again is not None and again.id != first.id
    assert again.fire_at is not None
    assert abs((again.fire_at - (now + timedelta(minutes=5))).total_seconds()) < 1


def test_cancel_by_label(database: Database) -> None:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    scheduler, _events, _notifier = _make(database, Clock(now))
    scheduler.add(Timer(label="таймер на чай", fire_at=now + timedelta(minutes=5)))
    scheduler.add(Timer(label="напоминание купить хлеб", fire_at=now + timedelta(minutes=9)))
    cancelled = scheduler.cancel_by_label("чай")
    assert len(cancelled) == 1
    assert len(scheduler.active(now=now)) == 1


# --------------------------------------------------------------------------- #
# Actions (over a registered scheduler)
# --------------------------------------------------------------------------- #


@pytest.fixture
def active(database: Database) -> Iterator[TimerScheduler]:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    scheduler, _events, _notifier = _make(database, Clock(now))
    set_active_scheduler(scheduler)
    yield scheduler
    set_active_scheduler(None)


def test_set_timer_action_creates_entry(active: TimerScheduler) -> None:
    result = SetTimer().run(SetTimer.Params(seconds=300, label="чай"))
    assert result.ok
    assert active_scheduler() is active
    assert len(active.active()) == 1


def test_set_alarm_action_builds_cron(active: TimerScheduler) -> None:
    result = SetAlarm().run(SetAlarm.Params(hour=7, minute=0, weekdays=(1,)))
    assert result.ok
    assert result.data["cron"] == cron_weekly((1,), 7, 0)


def test_set_reminder_action(active: TimerScheduler) -> None:
    when = datetime(2026, 6, 1, 15, 0, tzinfo=UTC)
    result = SetReminder().run(SetReminder.Params(fire_at=when, text="позвонить маме"))
    assert result.ok
    assert len(active.active()) == 1


def test_list_and_cancel_by_label_actions(active: TimerScheduler) -> None:
    SetTimer().run(SetTimer.Params(seconds=600, label="чай"))
    listed = ListTimers().run(ListTimers.Params())
    assert listed.value == 1
    cancelled = CancelTimer().run(CancelTimer.Params(label="чай"))
    assert cancelled.ok
    assert ListTimers().run(ListTimers.Params()).value == 0


# --------------------------------------------------------------------------- #
# Time-phrase parsing (existing NLU, exercised for the acceptance phrases)
# --------------------------------------------------------------------------- #


def test_duration_and_moment_phrases_parse() -> None:
    duration = parse_duration("5 минут")
    assert duration is not None and duration.seconds == 300
    now = datetime(2026, 6, 1, 6, 0, tzinfo=MSK)
    moment = parse_moment("в 7 утра", now=now)
    assert moment is not None and moment.hour == 7
