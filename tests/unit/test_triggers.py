from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ayris.actions.macros.context import TriggerSource
from ayris.core.events import CommandsChanged, EventBus, HotkeyTriggered, IntentMatched
from ayris.core.models import Command, Profile, Trigger, TriggerType
from ayris.core.pipeline_states import ManualScheduler
from ayris.core.profile import ProfileSwitched
from ayris.triggers.dispatcher import TriggerDispatcher
from ayris.triggers.schedule import CronExpression, ScheduleEntry, TriggerSchedule
from ayris.triggers.system_events import (
    ACTIVE_WINDOW_CHANGED,
    DEVICE_CONNECTED,
    DEVICE_DISCONNECTED,
    FULLSCREEN_ENTERED,
    FULLSCREEN_EXITED,
    POWER_CHANGED,
    PROCESS_STARTED,
    PROCESS_STOPPED,
    AdaptiveProcessPoller,
    SystemEvent,
)


class FakeCommands:
    def __init__(self, rows: list[Command]) -> None:
        self.rows = {row.id: row for row in rows}

    def get(self, command_id: int) -> Command | None:
        return self.rows.get(command_id)


class FakeTriggers:
    def __init__(self, rows: list[Trigger]) -> None:
        self.rows = {row.id: row for row in rows}

    def list_for_profile(self, profile_id: int, **_kwargs: Any) -> list[Trigger]:
        return [row for row in self.rows.values() if row.command_id // 10 == profile_id]

    def list_for_command(self, command_id: int) -> list[Trigger]:
        return [row for row in self.rows.values() if row.command_id == command_id]

    def get(self, trigger_id: int) -> Trigger | None:
        return self.rows.get(trigger_id)

    def update(self, trigger: Trigger) -> None:
        self.rows[trigger.id] = trigger


class FakeRepositories:
    def __init__(self, commands: list[Command], triggers: list[Trigger]) -> None:
        self.commands = FakeCommands(commands)
        self.triggers = FakeTriggers(triggers)


class FakeEngine:
    def __init__(self) -> None:
        self.started: list[tuple[int, TriggerSource, dict[str, Any], str]] = []

    def start(
        self,
        command: Any,
        *,
        slots: Any = None,
        trigger: TriggerSource,
        request_id: str = "",
    ) -> None:
        self.started.append((command.id, trigger, dict(slots or {}), request_id))


class FakeMonitor:
    def __init__(self) -> None:
        self.subscriptions: frozenset[str] = frozenset()
        self.stopped = False

    def replace_subscriptions(self, names: Any) -> None:
        self.subscriptions = frozenset(names)

    def stop(self) -> None:
        self.stopped = True


def command(command_id: int, name: str = "Команда") -> Command:
    return Command(id=command_id, profile_id=command_id // 10, name=name)


def trigger(
    trigger_id: int, command_id: int, kind: TriggerType, payload: dict[str, Any]
) -> Trigger:
    return Trigger(id=trigger_id, command_id=command_id, type=kind, payload=payload)


def make_dispatcher(
    triggers: list[Trigger],
    *,
    clock: list[float] | None = None,
    launches_per_second: int = 10,
    profile_changed: Any = None,
) -> tuple[TriggerDispatcher, EventBus, FakeEngine, FakeRepositories, FakeMonitor]:
    command_ids = sorted({item.command_id for item in triggers})
    repos = FakeRepositories([command(item) for item in command_ids], triggers)
    bus = EventBus(thread_id=None)
    engine = FakeEngine()
    monitor = FakeMonitor()
    monotonic = (lambda: clock[0]) if clock is not None else (lambda: 0.0)
    dispatcher = TriggerDispatcher(
        bus,
        repos,
        engine,
        1,
        monitor=monitor,
        monotonic=monotonic,
        launches_per_second=launches_per_second,
        profile_changed=profile_changed,
    )
    return dispatcher, bus, engine, repos, monitor


def test_voice_hotkey_and_every_system_event_use_one_dispatcher() -> None:
    kinds = [
        PROCESS_STARTED,
        PROCESS_STOPPED,
        ACTIVE_WINDOW_CHANGED,
        DEVICE_CONNECTED,
        DEVICE_DISCONNECTED,
        POWER_CHANGED,
        FULLSCREEN_ENTERED,
        FULLSCREEN_EXITED,
    ]
    rows = [
        trigger(1, 10, TriggerType.VOICE, {"phrase": "запуск"}),
        trigger(2, 10, TriggerType.HOTKEY, {"combo": "ctrl+k"}),
        *(
            trigger(index + 3, 10, TriggerType.EVENT, {"event_name": kind})
            for index, kind in enumerate(kinds)
        ),
    ]
    dispatcher, bus, engine, _repos, _monitor = make_dispatcher(rows)
    try:
        bus.publish(IntentMatched("запуск", command_id=10, slots={"x": 1}, request_id="r"))
        bus.publish(HotkeyTriggered(10))
        for kind in kinds:
            bus.publish(SystemEvent(kind))
        assert [item[1] for item in engine.started] == [
            TriggerSource.VOICE,
            TriggerSource.HOTKEY,
            *([TriggerSource.EVENT] * len(kinds)),
        ]
        assert engine.started[0][2:] == ({"x": 1}, "r")
    finally:
        dispatcher.close()


def test_event_masks_and_debounce() -> None:
    clock = [0.0]
    rows = [
        trigger(
            1,
            10,
            TriggerType.EVENT,
            {
                "event_name": ACTIVE_WINDOW_CHANGED,
                "filter_json": {"process": "code*.exe", "title": "*Ayris*"},
                "debounce_ms": 500,
            },
        )
    ]
    dispatcher, bus, engine, _repos, _monitor = make_dispatcher(rows, clock=clock)
    try:
        matching = SystemEvent(ACTIVE_WINDOW_CHANGED, process="Code.exe", title="AYRIS.py")
        bus.publish(SystemEvent(ACTIVE_WINDOW_CHANGED, process="other.exe", title="Ayris"))
        bus.publish(matching)
        bus.publish(matching)
        clock[0] = 0.6
        bus.publish(matching)
        assert len(engine.started) == 2
    finally:
        dispatcher.close()


def test_command_rate_limit_stops_a_trigger_storm() -> None:
    clock = [0.0]
    rows = [trigger(1, 10, TriggerType.HOTKEY, {"combo": "ctrl+k"})]
    dispatcher, bus, engine, _repos, _monitor = make_dispatcher(
        rows, clock=clock, launches_per_second=2
    )
    try:
        bus.publish(HotkeyTriggered(10))
        bus.publish(HotkeyTriggered(10))
        bus.publish(HotkeyTriggered(10))
        clock[0] = 1.0
        bus.publish(HotkeyTriggered(10))
        assert len(engine.started) == 3
    finally:
        dispatcher.close()


def test_disabled_enable_disable_and_hot_reload_replace_subscriptions() -> None:
    old = trigger(1, 10, TriggerType.EVENT, {"event_name": PROCESS_STARTED, "enabled": False})
    dispatcher, bus, engine, repos, monitor = make_dispatcher([old])
    try:
        assert not dispatcher.active_trigger_ids
        assert not monitor.subscriptions
        dispatcher.enable(1)
        assert dispatcher.active_trigger_ids == frozenset({1})
        bus.publish(SystemEvent(PROCESS_STARTED, process="one.exe"))
        dispatcher.disable(1)
        assert repos.triggers.rows[1].payload["enabled"] is False
        repos.triggers.rows[2] = trigger(2, 10, TriggerType.EVENT, {"event_name": DEVICE_CONNECTED})
        repos.triggers.rows.pop(1)
        bus.publish(CommandsChanged())
        assert monitor.subscriptions == frozenset({DEVICE_CONNECTED})
        bus.publish(SystemEvent(PROCESS_STARTED, process="old.exe"))
        bus.publish(SystemEvent(DEVICE_CONNECTED, device_type="audio"))
        assert len(engine.started) == 2
    finally:
        dispatcher.close()


def test_profile_switch_rebuilds_index() -> None:
    rows = [
        trigger(1, 10, TriggerType.HOTKEY, {"combo": "ctrl+1"}),
        trigger(2, 20, TriggerType.HOTKEY, {"combo": "ctrl+2"}),
    ]
    switched: list[int] = []
    dispatcher, bus, engine, _repos, _monitor = make_dispatcher(
        rows, profile_changed=switched.append
    )
    try:
        bus.publish(ProfileSwitched(Profile(id=2, name="Другой")))
        bus.publish(HotkeyTriggered(10))
        bus.publish(HotkeyTriggered(20))
        assert [item[0] for item in engine.started] == [20]
        assert switched == [2]
    finally:
        dispatcher.close()


def test_cron_and_missed_policy_with_fake_clocks_keep_one_timer() -> None:
    wall = [datetime(2026, 9, 9, 12, 0, 30, tzinfo=UTC)]
    mono = [0.0]
    scheduler = ManualScheduler()
    fired: list[int] = []
    schedule = TriggerSchedule(
        fired.append,
        scheduler=scheduler,
        wall_clock=lambda: wall[0],
        monotonic=lambda: mono[0],
    )
    schedule.replace([ScheduleEntry(1, cron="1 * * * *")])
    assert scheduler.pending == (30.0,)
    wall[0] += timedelta(seconds=30)
    mono[0] += 30
    scheduler.fire_all()
    assert fired == [1]
    assert len(scheduler.pending) == 1

    schedule.replace([ScheduleEntry(2, fire_at=wall[0] + timedelta(minutes=1), missed="skip")])
    wall[0] += timedelta(hours=2)
    mono[0] += 5  # simulated sleep / wall-clock jump
    scheduler.fire_all()
    assert fired == [1]
    schedule.close()


def test_cron_parser_ranges_steps_and_sunday() -> None:
    cron = CronExpression("*/15 9-10 * * 1-5")
    assert cron.matches(datetime(2026, 9, 9, 9, 30))
    assert not cron.matches(datetime(2026, 9, 12, 9, 30))


def test_adaptive_process_polling_measures_cost_and_backs_off() -> None:
    snapshots = iter([[(1, "a.exe")], [(1, "a.exe")], [(2, "b.exe")]])
    moments = iter([0.0, 0.01, 1.0, 1.02, 2.0, 2.03])
    events: list[SystemEvent] = []
    poller = AdaptiveProcessPoller(
        events.append, lambda: next(snapshots), monotonic=lambda: next(moments)
    )
    poller.scan()
    poller.scan()
    assert poller.interval > poller.minimum
    poller.scan()
    assert poller.interval == poller.minimum
    assert [event.kind for event in events] == [PROCESS_STARTED, PROCESS_STOPPED]
    assert poller.stats.process_scans == 3
    assert poller.stats.average_scan_seconds > 0
