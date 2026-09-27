"""Задача 69: сон, гибернация и пробуждение — декодирование и восстановление.

:func:`classify_power_message` — чистое сопоставление ``WM_POWERBROADCAST`` с
переходом питания, поэтому проверяется на любой платформе без Windows.
:class:`PowerCoordinator` собирается на подставной шине с утиными зависимостями:
засыпание объявляет парковку, пробуждение пересоздаёт аудиопоток
(``restart_scope(AUDIO)`` — реальная попытка заново открыть устройство),
перепроверяет связь и пересчитывает таймеры, затем объявляет пробуждение.

Ключевой тест — с НАСТОЯЩИМ :class:`~ayris.actions.timers.scheduler.TimerScheduler`
на замороженных часах: монотонный таймер не идёт во сне, поэтому на пробуждении
просроченный одноразовый таймер должен сработать. Ни одного ``sleep`` — планировщик
не запускается, срабатывание идёт через :meth:`resume`, а не через реальное
ожидание.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from ayris.actions.timers.scheduler import TimerScheduler
from ayris.core.config import RestartScope
from ayris.core.database import Database, reset_database
from ayris.core.events import Event, EventBus
from ayris.core.models import Timer
from ayris.core.power_events import (
    PBT_APMRESUMEAUTOMATIC,
    PBT_APMRESUMECRITICAL,
    PBT_APMRESUMESUSPEND,
    PBT_APMSUSPEND,
    WM_POWERBROADCAST,
    PowerCoordinator,
    PowerTransition,
    SystemResumed,
    SystemSuspending,
    classify_power_message,
)
from ayris.core.repositories import Repositories

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.integration

# ----------------------------------------------------------------------
# Замороженные часы и утиные зависимости координатора.
# ----------------------------------------------------------------------


class Clock:
    """Замороженные часы: тест сам выбирает «сейчас», планировщик не ждёт."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class RecordingNotifier:
    """Нотификатор, который лишь запоминает сработавшие записи."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, bool]] = []

    def notify(self, timer: Timer, *, missed: bool = False) -> None:
        assert timer.id is not None
        self.calls.append((timer.id, missed))


class FakeManager:
    """Утиный менеджер: запоминает, какие области его просили перезапустить."""

    def __init__(self) -> None:
        self.scopes: list[tuple[RestartScope, str]] = []

    def restart_scope(
        self, scope: RestartScope, settings_reason: str = "изменились настройки"
    ) -> int:
        self.scopes.append((scope, settings_reason))
        return 1


class BrokenManager:
    """Менеджер, у которого пересоздание аудио падает — пробуждение не должно встать."""

    def restart_scope(
        self, scope: RestartScope, settings_reason: str = "изменились настройки"
    ) -> int:
        raise RuntimeError("устройство недоступно")


class FakeConnectivity:
    """Монитор связи, считающий немедленные перепроверки."""

    def __init__(self) -> None:
        self.checks = 0

    def check_now(self) -> Any:
        self.checks += 1
        return True


def collect(bus: EventBus, event_type: type[Event]) -> list[Any]:
    """Подписаться жёстко и вернуть список, в который падают события."""
    received: list[Any] = []
    bus.subscribe(event_type, received.append, weak=False)
    return received


@pytest.fixture
def bus() -> EventBus:
    """Шина, доставляющая встроенно на том потоке, что публикует."""
    return EventBus(thread_id=None)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """Пустая БД во временной папке; глобальный дескриптор сбрасывается после."""
    handle = Database.open(tmp_path / "timers.db")
    try:
        yield handle
    finally:
        handle.close()
        reset_database()


class TestDecode:
    """Декодирование ``WM_POWERBROADCAST`` — чистое, без Windows."""

    @pytest.mark.parametrize(
        "wparam",
        [PBT_APMRESUMECRITICAL, PBT_APMRESUMESUSPEND, PBT_APMRESUMEAUTOMATIC],
    )
    def test_every_resume_code_decodes_to_resume(self, wparam: int) -> None:
        assert classify_power_message(WM_POWERBROADCAST, wparam) is PowerTransition.RESUME

    def test_suspend_code_decodes_to_suspend(self) -> None:
        assert classify_power_message(WM_POWERBROADCAST, PBT_APMSUSPEND) is PowerTransition.SUSPEND

    def test_a_foreign_message_is_ignored(self) -> None:
        # WM_CREATE (0x0001) — не про питание, даже с «спящим» wParam.
        assert classify_power_message(0x0001, PBT_APMSUSPEND) is None

    def test_an_uninteresting_wparam_is_ignored(self) -> None:
        # PBT_APMPOWERSTATUSCHANGE (0x000A) — смена питания сети, нам не важна.
        assert classify_power_message(WM_POWERBROADCAST, 0x000A) is None


class TestCoordinator:
    """Переходы питания как события и восстановление на пробуждении."""

    def test_suspend_announces_a_park(self, bus: EventBus) -> None:
        parked = collect(bus, SystemSuspending)
        PowerCoordinator(bus=bus).on_suspend()
        assert len(parked) == 1

    def test_resume_recreates_audio_rechecks_net_then_announces(self, bus: EventBus) -> None:
        resumed = collect(bus, SystemResumed)
        manager = FakeManager()
        net = FakeConnectivity()
        PowerCoordinator(bus=bus, worker_manager=manager, connectivity=net).on_resume()
        assert manager.scopes == [(RestartScope.AUDIO, "пробуждение системы")]
        assert net.checks == 1
        assert len(resumed) == 1

    def test_handle_dispatches_both_transitions(self, bus: EventBus) -> None:
        parked = collect(bus, SystemSuspending)
        resumed = collect(bus, SystemResumed)
        coordinator = PowerCoordinator(bus=bus)
        coordinator.handle(PowerTransition.SUSPEND)
        coordinator.handle(PowerTransition.RESUME)
        assert len(parked) == 1
        assert len(resumed) == 1

    def test_resume_without_dependencies_still_announces(self, bus: EventBus) -> None:
        resumed = collect(bus, SystemResumed)
        PowerCoordinator(bus=bus).on_resume()
        assert len(resumed) == 1

    def test_a_broken_audio_restart_does_not_wedge_resume(self, bus: EventBus) -> None:
        resumed = collect(bus, SystemResumed)
        net = FakeConnectivity()
        PowerCoordinator(bus=bus, worker_manager=BrokenManager(), connectivity=net).on_resume()
        # Сбой пересоздания аудио проглочен: сеть всё равно перепроверена,
        # пробуждение всё равно объявлено — восстановление идёт до конца.
        assert net.checks == 1
        assert len(resumed) == 1

    def test_resume_fires_an_overdue_one_shot_timer(
        self, bus: EventBus, database: Database
    ) -> None:
        # Монотонный таймер не идёт во сне: на пробуждении просроченный
        # одноразовый таймер должен сработать через resume(), а не через ожидание.
        now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        notifier = RecordingNotifier()
        scheduler = TimerScheduler(Repositories(database), bus, notifier=notifier, clock=Clock(now))
        created = Repositories(database).timers.create(
            Timer(label="Чай", fire_at=now - timedelta(minutes=1))
        )
        assert created.id is not None

        try:
            PowerCoordinator(bus=bus, timer_scheduler=scheduler).on_resume()
        finally:
            scheduler.stop()

        assert notifier.calls == [(created.id, True)]
