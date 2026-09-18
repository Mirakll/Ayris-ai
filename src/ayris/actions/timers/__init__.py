"""Timers, reminders and alarms: one scheduler over the ``timers`` table.

The scheduler keeps only the nearest firing in memory and a single wait; the
actions (``SetTimer``, ``SetReminder``, ``SetAlarm``, ``CancelTimer``,
``ListTimers``) feed it and are registered for the macro engine and NLU. Time
maths (one-shot and cron, DST-aware) lives in :mod:`ayris.actions.timers.schedule`;
optional calendar sync in :mod:`ayris.actions.timers.sync`.
"""

from __future__ import annotations

from ayris.actions.timers.schedule import (
    CronError,
    CronExpr,
    MissedPolicy,
    cron_daily,
    cron_weekly,
    next_fire,
    remaining,
)
from ayris.actions.timers.scheduler import (
    ActiveTimer,
    BusTimerNotifier,
    TimerNotifier,
    TimerScheduler,
    active_scheduler,
    set_active_scheduler,
)

__all__ = [
    "ActiveTimer",
    "BusTimerNotifier",
    "CronError",
    "CronExpr",
    "MissedPolicy",
    "TimerNotifier",
    "TimerScheduler",
    "active_scheduler",
    "cron_daily",
    "cron_weekly",
    "next_fire",
    "remaining",
    "set_active_scheduler",
]
