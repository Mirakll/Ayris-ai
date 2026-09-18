"""The registered actions that create, list and cancel scheduled entries.

Time parsing lives in the NLU slot types (``nlu/timeparse.py``); these actions
take an already-resolved duration, moment or time-of-day and only turn it into a
:class:`~ayris.core.models.Timer`. They go through the active
:class:`~ayris.actions.timers.scheduler.TimerScheduler` when one is running so it
re-arms, and fall back to the database directly otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.timers.schedule import cron_daily, cron_weekly, next_fire, remaining
from ayris.actions.timers.scheduler import active_scheduler
from ayris.core.database import get_database
from ayris.core.models import Timer, TimerKind, utc_now
from ayris.core.repositories import Repositories

__all__ = ["CancelTimer", "ListTimers", "SetAlarm", "SetReminder", "SetTimer"]

_WEEKDAY_RU: tuple[str, ...] = ("вс", "пн", "вт", "ср", "чт", "пт", "сб")


def _create(timer: Timer) -> Timer:
    scheduler = active_scheduler()
    if scheduler is not None:
        return scheduler.add(timer)
    return Repositories(get_database()).timers.create(timer)


def _human_duration(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours} ч")
    if minutes:
        parts.append(f"{minutes} мин")
    if secs and not hours:
        parts.append(f"{secs} с")
    return " ".join(parts) or "0 с"


@register
class SetTimer(Action):
    """Set a one-shot timer a fixed interval from now."""

    meta: ClassVar = ActionMeta(
        name="SetTimer",
        category=ActionCategory.TIMERS,
        title_ru="Поставить таймер",
        description_ru="Одноразовый таймер на заданный интервал",
    )

    class Params(ActionParams):
        seconds: int = Field(ge=1, le=86400 * 30, title="Через сколько", description="В секундах")
        label: str = Field(default="", max_length=200, title="Название")

    def run(self, params: Params) -> ActionResult[int]:
        fire_at = utc_now() + timedelta(seconds=params.seconds)
        label = params.label or f"Таймер на {_human_duration(params.seconds)}"
        created = _create(Timer(label=label, kind=TimerKind.TIMER, fire_at=fire_at))
        return ActionResult.done(
            f"Таймер поставлен на {_human_duration(params.seconds)}",
            value=created.id,
            data={"timer_id": created.id, "fire_at": fire_at.isoformat()},
        )


@register
class SetReminder(Action):
    """Set a reminder at an absolute date and time."""

    meta: ClassVar = ActionMeta(
        name="SetReminder",
        category=ActionCategory.TIMERS,
        title_ru="Напоминание",
        description_ru="Напоминание на конкретные дату и время",
    )

    class Params(ActionParams):
        fire_at: datetime = Field(title="Когда", description="Дата и время срабатывания")
        text: str = Field(min_length=1, max_length=500, title="Текст напоминания")

    def run(self, params: Params) -> ActionResult[int]:
        fire_at = params.fire_at
        if fire_at.tzinfo is None:
            fire_at = fire_at.replace(tzinfo=UTC)
        created = _create(Timer(label=params.text, kind=TimerKind.REMINDER, fire_at=fire_at))
        stamp = fire_at.astimezone().strftime("%d.%m %H:%M")
        return ActionResult.done(
            f"Напомню {stamp}: {params.text}",
            value=created.id,
            data={"timer_id": created.id, "fire_at": fire_at.isoformat()},
        )


@register
class SetAlarm(Action):
    """Set a recurring alarm at a time of day, optionally on given weekdays."""

    meta: ClassVar = ActionMeta(
        name="SetAlarm",
        category=ActionCategory.TIMERS,
        title_ru="Будильник",
        description_ru="Повторяющийся будильник на время суток",
    )

    class Params(ActionParams):
        hour: int = Field(ge=0, le=23, title="Час")
        minute: int = Field(default=0, ge=0, le=59, title="Минута")
        weekdays: tuple[int, ...] = Field(
            default=(),
            title="Дни недели",
            description="0 — воскресенье … 6 — суббота; пусто — каждый день",
        )
        label: str = Field(default="", max_length=200, title="Название")

    def run(self, params: Params) -> ActionResult[int]:
        for day in params.weekdays:
            if not 0 <= day <= 6:
                return ActionResult.failed(f"День недели вне диапазона 0..6: {day}")
        cron = (
            cron_weekly(params.weekdays, params.hour, params.minute)
            if params.weekdays
            else cron_daily(params.hour, params.minute)
        )
        label = params.label or "Будильник"
        created = _create(Timer(label=label, kind=TimerKind.ALARM, cron=cron))
        when = f"{params.hour:02d}:{params.minute:02d}"
        days = (
            "каждый день"
            if not params.weekdays
            else ", ".join(_WEEKDAY_RU[day % 7] for day in sorted(set(params.weekdays)))
        )
        return ActionResult.done(
            f"Будильник на {when} ({days})",
            value=created.id,
            data={"timer_id": created.id, "cron": cron},
        )


@register
class CancelTimer(Action):
    """Cancel a scheduled entry by id or by a fragment of its label."""

    meta: ClassVar = ActionMeta(
        name="CancelTimer",
        category=ActionCategory.TIMERS,
        title_ru="Отменить таймер",
        description_ru="Отмена таймера, напоминания или будильника",
    )

    class Params(ActionParams):
        timer_id: int | None = Field(default=None, title="Идентификатор")
        label: str = Field(default="", max_length=200, title="Название или его часть")

    def run(self, params: Params) -> ActionResult[int]:
        scheduler = active_scheduler()
        if params.timer_id is not None:
            ok = scheduler.cancel(params.timer_id) if scheduler else _repo_cancel(params.timer_id)
            if not ok:
                return ActionResult.failed("Такого таймера нет")
            return ActionResult.done("Отменил", value=1)
        if not params.label.strip():
            return ActionResult.failed("Нужен идентификатор или название таймера")
        cancelled = (
            scheduler.cancel_by_label(params.label)
            if scheduler
            else _repo_cancel_by_label(params.label)
        )
        if not cancelled:
            return ActionResult.failed(f"Не нашёл таймер «{params.label}»")
        word = "таймер" if len(cancelled) == 1 else "таймеров"
        return ActionResult.done(f"Отменил {len(cancelled)} {word}", value=len(cancelled))


@register
class ListTimers(Action):
    """List active scheduled entries with the time left on each."""

    meta: ClassVar = ActionMeta(
        name="ListTimers",
        category=ActionCategory.TIMERS,
        title_ru="Список таймеров",
        description_ru="Активные таймеры, напоминания и будильники",
    )

    class Params(ActionParams):
        pass

    def run(self, _params: Params) -> ActionResult[int]:
        scheduler = active_scheduler()
        entries = scheduler.active() if scheduler else _repo_active()
        if not entries:
            return ActionResult.done("Активных таймеров нет", value=0)
        lines = [f"{entry.label} — {_left(entry.remaining)}" for entry in entries]
        return ActionResult.done(
            "; ".join(lines),
            value=len(entries),
            data={"timers": [entry.id for entry in entries]},
        )


def _left(delta: timedelta) -> str:
    seconds = max(0, round(delta.total_seconds()))
    return _human_duration(seconds)


# -- repository fallbacks used when no scheduler is running --------------------


def _repo_cancel(timer_id: int) -> bool:
    return Repositories(get_database()).timers.delete(timer_id)


def _repo_cancel_by_label(text: str) -> list[int]:
    needle = text.strip().casefold()
    repo = Repositories(get_database()).timers
    cancelled: list[int] = []
    for timer in repo.list_all(enabled_only=True):
        if timer.id is not None and needle in timer.label.casefold() and repo.delete(timer.id):
            cancelled.append(timer.id)
    return cancelled


def _repo_active() -> list[_View]:
    tz = datetime.now().astimezone().tzinfo or UTC
    now = utc_now()
    repo = Repositories(get_database()).timers
    views: list[_View] = []
    for timer in repo.list_all(enabled_only=True):
        upcoming = next_fire(timer, now=now, tz=tz)
        if upcoming is None or timer.id is None:
            continue
        left = remaining(timer, now=now, tz=tz) or timedelta(0)
        views.append(_View(timer.id, timer.label, left))
    views.sort(key=lambda item: item.remaining)
    return views


class _View:
    """A minimal active-entry view for the no-scheduler fallback path."""

    __slots__ = ("id", "label", "remaining")

    def __init__(self, timer_id: int, label: str, left: timedelta) -> None:
        self.id = timer_id
        self.label = label
        self.remaining = left
