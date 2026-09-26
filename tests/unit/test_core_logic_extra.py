"""Дополнительное покрытие чистой логики шести модулей ядра.

Только детерминированные проверки без сети, железа, звука и моделей: часы
инъектируются, планировщики заменяются ручными, потоки заводятся лишь там, где
их гарантированно останавливают и присоединяют. Модуль трогает диспетчер
триггеров, монитор системных событий, менеджер подтверждений, планировщик
таймеров, переносимый профиль и контекст диалога.
"""

from __future__ import annotations

import io
import json
import sys
import threading
import zipfile
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ayris.actions.macros.context import TriggerSource
from ayris.actions.registry import ConfirmationRequest
from ayris.actions.timers.schedule import MissedPolicy, cron_daily
from ayris.actions.timers.scheduler import ActiveTimer, BusTimerNotifier, TimerScheduler
from ayris.core import portable_profile as pp
from ayris.core.config import PrivacyConfig
from ayris.core.errors import ProfileError
from ayris.core.events import (
    CommandsChanged,
    EventBus,
    HotkeyTriggered,
    IntentMatched,
    NotificationRequested,
    TimerFired,
)
from ayris.core.models import (
    Command,
    Profile,
    Timer,
    TimerKind,
    Trigger,
    TriggerType,
    to_db_timestamp,
)
from ayris.core.pipeline_states import ManualScheduler
from ayris.core.profile import ProfileSwitched
from ayris.nlu import context as ctx
from ayris.security import confirmation as conf
from ayris.security.hello import WindowsHello
from ayris.security.pin import PinManager
from ayris.triggers import dispatcher as disp
from ayris.triggers import system_events as se

pytestmark = pytest.mark.unit

# ----------------------------------------------------------------------
# диспетчер триггеров
# ----------------------------------------------------------------------


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


def _command(command_id: int, name: str = "Команда", *, enabled: bool = True) -> Command:
    return Command(id=command_id, profile_id=command_id // 10, name=name, enabled=enabled)


def _trigger(
    trigger_id: int, command_id: int, kind: TriggerType, payload: dict[str, Any]
) -> Trigger:
    return Trigger(id=trigger_id, command_id=command_id, type=kind, payload=payload)


def test_dispatcher_reload_skips_missing_command_and_bad_event_name() -> None:
    repos = FakeRepositories(
        [_command(10)],
        [
            _trigger(1, 10, TriggerType.EVENT, {"event_name": se.ACTIVE_WINDOW_CHANGED}),
            # serialises fine (the writer keeps the name under "name"), but the
            # dispatcher indexes by "event_name" and so skips this one.
            _trigger(2, 10, TriggerType.EVENT, {"name": se.PROCESS_STARTED}),
            _trigger(3, 15, TriggerType.VOICE, {"phrase": "x"}),
        ],
    )
    bus = EventBus(thread_id=None)
    monitor = FakeMonitor()
    dispatcher = disp.TriggerDispatcher(
        bus, repos, FakeEngine(), 1, monitor=monitor, monotonic=lambda: 0.0
    )
    assert dispatcher.profile_id == 1
    assert dispatcher.active_trigger_ids == frozenset({1, 2, 3})
    assert monitor.subscriptions == frozenset({se.ACTIVE_WINDOW_CHANGED})


def test_dispatcher_disabled_command_is_not_indexed() -> None:
    repos = FakeRepositories(
        [_command(10, enabled=False)],
        [_trigger(1, 10, TriggerType.VOICE, {"phrase": "x"})],
    )
    bus = EventBus(thread_id=None)
    engine = FakeEngine()
    disp.TriggerDispatcher(bus, repos, engine, 1, monitor=FakeMonitor(), monotonic=lambda: 0.0)
    bus.publish(IntentMatched("x", command_id=10))
    assert engine.started == []


def test_dispatcher_voice_and_hotkey_launch_and_reload() -> None:
    repos = FakeRepositories(
        [_command(10)],
        [
            _trigger(1, 10, TriggerType.VOICE, {"phrase": "привет"}),
            _trigger(2, 10, TriggerType.HOTKEY, {"combo": "ctrl+a"}),
        ],
    )
    bus = EventBus(thread_id=None)
    engine = FakeEngine()
    disp.TriggerDispatcher(bus, repos, engine, 1, monitor=FakeMonitor(), monotonic=lambda: 0.0)
    bus.publish(IntentMatched("привет", command_id=10, slots={"x": 1}, request_id="r1"))
    bus.publish(HotkeyTriggered(command_id=10))
    assert [row[1] for row in engine.started] == [TriggerSource.VOICE, TriggerSource.HOTKEY]
    assert engine.started[0][2] == {"x": 1}
    assert engine.started[0][3] == "r1"
    bus.publish(CommandsChanged(command_id=10))
    assert len(engine.started) == 2
    bus.publish(IntentMatched("привет", command_id=None))
    assert len(engine.started) == 2


def test_dispatcher_system_event_launch_with_filter_and_debounce() -> None:
    clock = [0.0]
    repos = FakeRepositories(
        [_command(10)],
        [
            _trigger(
                1,
                10,
                TriggerType.EVENT,
                {"event_name": se.ACTIVE_WINDOW_CHANGED, "filter_json": {"process": "code*"}},
            )
        ],
    )
    bus = EventBus(thread_id=None)
    engine = FakeEngine()
    disp.TriggerDispatcher(bus, repos, engine, 1, monitor=FakeMonitor(), monotonic=lambda: clock[0])
    matching = se.SystemEvent(se.ACTIVE_WINDOW_CHANGED, process="Code.exe", title="файл")
    bus.publish(matching)
    bus.publish(matching)
    assert len(engine.started) == 1
    clock[0] += 10.0
    bus.publish(matching)
    assert len(engine.started) == 2
    bus.publish(se.SystemEvent(se.ACTIVE_WINDOW_CHANGED, process="notepad.exe"))
    assert len(engine.started) == 2


def test_dispatcher_timer_reload_and_on_timer() -> None:
    now_dt = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    future = now_dt + timedelta(hours=1)
    repos = FakeRepositories(
        [_command(10)],
        [_trigger(7, 10, TriggerType.TIMER, {"fire_at": to_db_timestamp(future)})],
    )
    bus = EventBus(thread_id=None)
    engine = FakeEngine()
    scheduler = ManualScheduler()
    dispatcher = disp.TriggerDispatcher(
        bus,
        repos,
        engine,
        1,
        monitor=FakeMonitor(),
        scheduler=scheduler,
        wall_clock=lambda: now_dt,
        monotonic=lambda: 0.0,
    )
    assert scheduler.pending  # a wait was armed on the manual scheduler, no real timer
    dispatcher._on_timer(7)
    dispatcher._on_timer(9999)
    assert [row[1] for row in engine.started] == [TriggerSource.TIMER]


def test_dispatcher_enable_unknown_raises_and_profile_switch() -> None:
    repos = FakeRepositories([_command(10)], [_trigger(1, 10, TriggerType.VOICE, {"phrase": "x"})])
    bus = EventBus(thread_id=None)
    dispatcher = disp.TriggerDispatcher(
        bus, repos, FakeEngine(), 1, monitor=FakeMonitor(), monotonic=lambda: 0.0
    )
    with pytest.raises(KeyError):
        dispatcher.enable(9999)
    bus.publish(ProfileSwitched(profile=Profile(id=None, name="без id")))
    assert dispatcher.profile_id == 1
    bus.publish(ProfileSwitched(profile=Profile(id=2, name="второй")))
    assert dispatcher.profile_id == 2


def test_dispatcher_enable_disable_and_close() -> None:
    repos = FakeRepositories([_command(10)], [_trigger(1, 10, TriggerType.VOICE, {"phrase": "x"})])
    bus = EventBus(thread_id=None)
    changed: list[CommandsChanged] = []
    bus.subscribe(CommandsChanged, changed.append, weak=False)
    monitor = FakeMonitor()
    dispatcher = disp.TriggerDispatcher(
        bus, repos, FakeEngine(), 1, monitor=monitor, monotonic=lambda: 0.0
    )
    dispatcher.disable(1)
    disabled = repos.triggers.get(1)
    assert disabled is not None
    assert disabled.payload["enabled"] is False
    assert dispatcher.active_trigger_ids == frozenset()  # reload drops the disabled trigger
    dispatcher.enable(1)
    enabled = repos.triggers.get(1)
    assert enabled is not None
    assert enabled.payload["enabled"] is True
    assert dispatcher.active_trigger_ids == frozenset({1})
    assert len(changed) == 2  # each toggle republishes CommandsChanged
    dispatcher.close()
    assert monitor.stopped is True
    bus.publish(IntentMatched("x", command_id=10))  # unsubscribed: no further reloads
    assert len(changed) == 2


# ----------------------------------------------------------------------
# монитор системных событий
# ----------------------------------------------------------------------


def test_monitor_stats_average() -> None:
    empty = se.MonitorStats(
        process_scans=0, scan_seconds=0.0, max_scan_seconds=0.0, poll_interval=1.0
    )
    assert empty.average_scan_seconds == 0.0
    filled = se.MonitorStats(
        process_scans=4, scan_seconds=2.0, max_scan_seconds=1.0, poll_interval=1.5
    )
    assert filled.average_scan_seconds == 0.5


def test_adaptive_process_poller_scan_diffs_and_interval() -> None:
    ticks = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.0])
    events: list[se.SystemEvent] = []
    snapshots = iter(
        [
            [(1, "a.exe"), (2, "b.exe")],
            [(2, "b.exe"), (3, "c.exe")],
            [(2, "b.exe"), (3, "c.exe")],
        ]
    )
    poller = se.AdaptiveProcessPoller(
        events.append,
        lambda: next(snapshots),
        minimum=1.0,
        maximum=15.0,
        monotonic=lambda: next(ticks),
    )
    assert poller.scan() is False  # first scan only primes the baseline
    assert poller.interval == 1.0
    assert poller.scan() is True  # 3 started, 1 stopped
    kinds = {(event.kind, event.process) for event in events}
    assert (se.PROCESS_STARTED, "c.exe") in kinds
    assert (se.PROCESS_STOPPED, "a.exe") in kinds
    assert poller.interval == 1.0  # change resets to the minimum
    assert poller.scan() is False  # nothing changed
    assert poller.interval == 1.5  # idle backs off by *1.5
    assert poller.stats.process_scans == 3
    assert poller.stats.poll_interval == 1.5


def test_process_snapshot_linux_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(se.sys, "platform", "linux")
    snapshot = list(se._process_snapshot())
    assert snapshot == [(0, Path(sys.executable).name)]


def test_windows_message_loop_init() -> None:
    stop = threading.Event()
    seen: list[se.SystemEvent] = []
    loop = se._WindowsMessageLoop(seen.append, lambda: frozenset({se.POWER_CHANGED}), stop)
    assert loop.emit == seen.append
    assert loop.wanted() == frozenset({se.POWER_CHANGED})
    assert loop.stop is stop
    assert loop._last_fullscreen is False
    assert loop._callbacks == []


def test_system_event_monitor_filters_and_emit() -> None:
    bus = EventBus(thread_id=None)
    seen: list[se.SystemEvent] = []
    bus.subscribe(se.SystemEvent, seen.append, weak=False)
    monitor = se.SystemEventMonitor(bus, process_snapshot=lambda: ())
    # unknown names filter down to nothing, so no worker thread is started
    monitor.replace_subscriptions({"not_a_real_event"})
    assert monitor.subscriptions == frozenset()
    # emit publishes only the currently wanted kinds
    monitor._wanted = frozenset({se.POWER_CHANGED})
    monitor.emit(se.SystemEvent(se.POWER_CHANGED, on_ac=True))
    monitor.emit(se.SystemEvent(se.ACTIVE_WINDOW_CHANGED, process="code.exe"))
    assert [event.kind for event in seen] == [se.POWER_CHANGED]


def test_system_event_monitor_start_run_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(se.sys, "platform", "linux")  # skip the real Windows loop thread
    bus = EventBus(thread_id=None)
    scanned = threading.Event()

    def snapshot() -> list[tuple[int, str]]:
        scanned.set()
        return [(1, "proc.exe")]

    monitor = se.SystemEventMonitor(bus, process_snapshot=snapshot)
    monitor.replace_subscriptions({se.PROCESS_STARTED})  # wanted -> start()
    worker = monitor._thread
    assert worker is not None
    assert scanned.wait(2.0)
    monitor.replace_subscriptions(set())  # not wanted -> stop()
    assert monitor._thread is None
    assert monitor._window_thread is None
    assert not worker.is_alive()


# ----------------------------------------------------------------------
# менеджер подтверждений
# ----------------------------------------------------------------------


class _MemoryPinStore:
    def __init__(self) -> None:
        self._value: str | None = None

    def get_password(self, service_name: str, username: str) -> str | None:
        return self._value

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self._value = password

    def delete_password(self, service_name: str, username: str) -> None:
        self._value = None


class _Pause:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False

    def __enter__(self) -> None:
        self.entered = True

    def __exit__(self, *_exc: object) -> None:
        self.exited = True


class _Named:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeOp:
    def __init__(self, value: Any = None, *, error: BaseException | None = None) -> None:
        self._value = value
        self._error = error

    def get(self) -> Any:
        if self._error is not None:
            raise self._error
        return self._value


class _FakeHelloBackend:
    def __init__(self, availability: _FakeOp, verification: _FakeOp | None = None) -> None:
        self._availability = availability
        self._verification = verification

    def check_availability_async(self) -> _FakeOp:
        return self._availability

    def request_verification_async(self, message: str) -> _FakeOp | None:
        return self._verification


def _privacy(**overrides: Any) -> PrivacyConfig:
    data: dict[str, Any] = {"require_confirmation": True}
    data.update(overrides)
    return PrivacyConfig.model_validate(data)


def _request() -> ConfirmationRequest:
    return ConfirmationRequest(action="power.shutdown", title_ru="Выключить компьютер")


def test_bounded_completes_fails_and_times_out() -> None:
    assert conf._bounded(lambda: 42, 5.0) == (True, 42)

    def boom() -> int:
        raise RuntimeError("boom")

    assert conf._bounded(boom, 5.0) == (True, None)
    release = threading.Event()

    def wedged() -> str:
        release.wait(2.0)
        return "late"

    completed, answer = conf._bounded(wedged, 0.0)
    assert completed is False
    assert answer is None
    release.set()
    for worker in threading.enumerate():
        if worker.name == "ayris-confirmation":
            worker.join(2.0)


def test_voice_decision_yes_no_and_unclear() -> None:
    assert conf._voice_decision("да", 0.82) is True
    assert conf._voice_decision("подтверждаю", 0.82) is True
    assert conf._voice_decision("нет", 0.82) is False
    assert conf._voice_decision("123 ###", 0.82) is None  # no cyrillic letters
    assert conf._voice_decision("абракадабра", 0.99) is None  # below threshold


def test_confirmation_not_required_returns_yes() -> None:
    mgr = conf.ConfirmationManager(_privacy(require_confirmation=False))
    assert mgr(_request()).confirmed is True


def test_confirmation_voice_paths() -> None:
    priv = _privacy(confirmation_method="voice")

    def verdict(answer: str | None) -> str:
        mgr = conf.ConfirmationManager(priv, voice=lambda _q, _t: answer)
        return mgr(_request()).reason

    yes = conf.ConfirmationManager(priv, voice=lambda _q, _t: "да")(_request())
    assert yes.confirmed is True
    rejected = conf.ConfirmationManager(priv, voice=lambda _q, _t: "нет")(_request())
    assert rejected.confirmed is False
    assert rejected.reason == "rejected"
    assert verdict("мяу") == "unclear"
    assert verdict(None) == "timeout"
    assert conf.ConfirmationManager(priv, voice=None)(_request()).reason == "unavailable"


def test_confirmation_voice_uses_pause_pipeline() -> None:
    pause = _Pause()
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="voice"),
        voice=lambda _q, _t: "да",
        pause_pipeline=lambda: pause,
    )
    assert mgr(_request()).confirmed is True
    assert pause.entered and pause.exited


def test_confirmation_dialog_paths() -> None:
    priv = _privacy(confirmation_method="dialog")
    assert conf.ConfirmationManager(priv, dialog=lambda _r, _t: True)(_request()).confirmed is True
    rejected = conf.ConfirmationManager(priv, dialog=lambda _r, _t: False)(_request())
    assert rejected.confirmed is False and rejected.reason == "rejected"
    assert (
        conf.ConfirmationManager(priv, dialog=lambda _r, _t: None)(_request()).reason == "timeout"
    )
    assert conf.ConfirmationManager(priv, dialog=None)(_request()).reason == "unavailable"


def test_confirmation_pin_paths() -> None:
    priv = _privacy(
        confirmation_method="pin", confirmation_pin_attempts=1, confirmation_pin_delay_sec=0.0
    )
    pin = PinManager(_MemoryPinStore(), sleeper=lambda _s: None)
    pin.set_pin("2468")
    assert (
        conf.ConfirmationManager(priv, pin=pin, ask_pin=lambda _n: "2468")(_request()).confirmed
        is True
    )
    rejected = conf.ConfirmationManager(priv, pin=pin, ask_pin=lambda _n: "0000")(_request())
    assert rejected.confirmed is False and rejected.reason == "pin rejected"
    empty = PinManager(_MemoryPinStore(), sleeper=lambda _s: None)
    assert (
        conf.ConfirmationManager(priv, pin=empty, ask_pin=lambda _n: "2468")(_request()).reason
        == "unavailable"
    )
    assert conf.ConfirmationManager(priv, pin=pin, ask_pin=None)(_request()).reason == "unavailable"


def test_confirmation_both_voice_then_dialog() -> None:
    priv = _privacy(confirmation_method="both")
    first_yes = conf.ConfirmationManager(
        priv, voice=lambda _q, _t: "да", dialog=lambda _r, _t: False
    )
    assert first_yes(_request()).confirmed is True
    fall_through = conf.ConfirmationManager(priv, voice=None, dialog=lambda _r, _t: True)
    assert fall_through(_request()).confirmed is True
    voice_rejects = conf.ConfirmationManager(
        priv, voice=lambda _q, _t: "нет", dialog=lambda _r, _t: True
    )
    verdict = voice_rejects(_request())
    assert verdict.confirmed is False and verdict.reason == "rejected"


def test_confirmation_requires_confirmation_reads_actions() -> None:
    mgr = conf.ConfirmationManager(_privacy(confirmation_actions=("power.shutdown",)))
    assert mgr.requires_confirmation(SimpleNamespace(meta=SimpleNamespace(name="power.shutdown")))
    assert not mgr.requires_confirmation(SimpleNamespace(meta=SimpleNamespace(name="music.play")))


def test_confirmation_hello_confirmed() -> None:
    backend = _FakeHelloBackend(_FakeOp(_Named("available")), _FakeOp(_Named("verified")))
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="hello"), hello=WindowsHello(backend)
    )
    assert mgr(_request()).confirmed is True


def test_confirmation_hello_rejected() -> None:
    backend = _FakeHelloBackend(_FakeOp(_Named("available")), _FakeOp(_Named("denied")))
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="hello"), hello=WindowsHello(backend)
    )
    verdict = mgr(_request())
    assert verdict.confirmed is False
    assert "отклонено" in verdict.user_message


def test_confirmation_hello_timeout() -> None:
    backend = _FakeHelloBackend(_FakeOp(_Named("available")), _FakeOp(error=TimeoutError("slow")))
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="hello"), hello=WindowsHello(backend)
    )
    verdict = mgr(_request())
    assert verdict.confirmed is False
    assert "Время ожидания" in verdict.user_message


def test_confirmation_hello_unavailable_falls_back_to_dialog() -> None:
    backend = _FakeHelloBackend(_FakeOp(_Named("device not present")))
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="hello", confirmation_fallback="dialog"),
        hello=WindowsHello(backend),
        dialog=lambda _r, _t: True,
    )
    assert mgr(_request()).confirmed is True


def test_confirmation_hello_and_fallback_unavailable() -> None:
    backend = _FakeHelloBackend(_FakeOp(_Named("not available")))
    mgr = conf.ConfirmationManager(
        _privacy(confirmation_method="hello", confirmation_fallback="dialog"),
        hello=WindowsHello(backend),
        dialog=None,
    )
    verdict = mgr(_request())
    assert verdict.confirmed is False
    assert verdict.reason == "hello and fallback unavailable"


# ----------------------------------------------------------------------
# планировщик таймеров
# ----------------------------------------------------------------------


class _FakeTimerRepo:
    def __init__(self, rows: tuple[Timer, ...] = (), *, bare: tuple[Timer, ...] = ()) -> None:
        self.rows: dict[int, Timer] = {}
        self._next = 1
        self.bare = list(bare)
        for row in rows:
            self.create(row)

    def create(self, timer: Timer) -> Timer:
        tid = timer.id if timer.id is not None else self._next
        self._next = max(self._next, tid + 1)
        stored = replace(timer, id=tid)
        self.rows[tid] = stored
        return stored

    def delete(self, timer_id: int) -> bool:
        return self.rows.pop(timer_id, None) is not None

    def list_all(self, *, enabled_only: bool = False) -> list[Timer]:
        rows = [row for row in self.rows.values() if not enabled_only or row.enabled]
        return rows + self.bare

    def get(self, timer_id: int) -> Timer | None:
        return self.rows.get(timer_id)

    def update(self, timer: Timer) -> None:
        if timer.id is not None:
            self.rows[timer.id] = timer

    def reschedule(self, timer_id: int, upcoming: datetime) -> None:
        row = self.rows.get(timer_id)
        if row is not None:
            self.rows[timer_id] = replace(row, fire_at=upcoming)

    def set_enabled(self, timer_id: int, *, enabled: bool) -> None:
        row = self.rows.get(timer_id)
        if row is not None:
            self.rows[timer_id] = replace(row, enabled=enabled)


class _SchedRepos:
    def __init__(self, timers: _FakeTimerRepo) -> None:
        self.timers = timers


class _BoomNotifier:
    def notify(self, timer: Timer, *, missed: bool = False) -> None:
        raise RuntimeError("нотификатор упал")


def _scheduler(repo: _FakeTimerRepo, bus: EventBus, **kwargs: Any) -> TimerScheduler:
    kwargs.setdefault("tz", UTC)
    return TimerScheduler(_SchedRepos(repo), bus, **kwargs)  # type: ignore[arg-type]


def test_active_timer_helpers() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    item = ActiveTimer(1, "Чай", TimerKind.TIMER, moment, timedelta(seconds=90))
    assert item.remaining_seconds() == 90
    assert item.due == moment
    overdue = ActiveTimer(2, "x", TimerKind.TIMER, moment, timedelta(seconds=-5))
    assert overdue.remaining_seconds() == 0


def test_bus_timer_notifier_variants() -> None:
    bus = EventBus(thread_id=None)
    seen: list[NotificationRequested] = []
    bus.subscribe(NotificationRequested, seen.append, weak=False)
    notifier = BusTimerNotifier(bus)
    notifier.notify(Timer(label="Чай", kind=TimerKind.TIMER))
    notifier.notify(Timer(label="", kind=TimerKind.TIMER))
    notifier.notify(Timer(label="Встреча", kind=TimerKind.REMINDER), missed=True)
    notifier.notify(Timer(label="Подъём", kind=TimerKind.ALARM))
    assert seen[0].title == "Таймер"
    assert seen[0].message == "Чай"
    assert seen[0].action == ""
    assert seen[1].message == "Таймер"
    assert seen[2].message == "Пропущено: Встреча"
    assert seen[2].title == "Напоминание"
    assert seen[2].action == "snooze"
    assert seen[3].title == "Будильник"


def test_scheduler_add_cron_and_active_and_snooze() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo()
    bus = EventBus(thread_id=None)
    sched = _scheduler(repo, bus, clock=lambda: moment)
    created = sched.add(Timer(label="Утро", kind=TimerKind.ALARM, cron=cron_daily(8, 0)))
    assert created.fire_at is not None
    assert created.fire_at > moment
    active = sched.active()
    assert [item.id for item in active] == [created.id]
    assert active[0].fire_at == created.fire_at
    assert created.id is not None
    again = sched.snooze(created.id, 5)
    assert again is not None
    assert again.fire_at == moment + timedelta(minutes=5)
    zero = sched.snooze(created.id, 0)
    assert zero is not None
    assert zero.fire_at == moment + timedelta(minutes=1)
    assert sched.snooze(999, 5) is None


def test_scheduler_cancel_by_label() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(
        rows=(
            Timer(id=1, label="Чай зелёный", fire_at=moment + timedelta(minutes=5)),
            Timer(id=2, label="Кофе", fire_at=moment + timedelta(minutes=5)),
        ),
        bare=(Timer(id=None, label="чай холодный"),),
    )
    bus = EventBus(thread_id=None)
    sched = _scheduler(repo, bus, clock=lambda: moment)
    assert sched.cancel_by_label("  ЧАЙ ") == [1]
    assert 1 not in repo.rows
    assert sched.cancel_by_label("   ") == []
    assert sched.cancel_by_label("нет такого") == []


def test_scheduler_cancel_edit_and_active_timers() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(
        rows=(Timer(id=1, label="Чай", fire_at=moment + timedelta(minutes=5)),),
        bare=(Timer(id=None, label="без id"),),
    )
    bus = EventBus(thread_id=None)
    sched = _scheduler(repo, bus, clock=lambda: moment)
    assert [item.id for item in sched.active_timers()] == [1]
    sched.edit(Timer(id=1, label="Чай чёрный", fire_at=moment + timedelta(minutes=6)))
    assert repo.rows[1].label == "Чай чёрный"
    assert sched.cancel(1) is True
    assert sched.cancel(1) is False
    sched.cancel_timer(999)


def test_scheduler_fire_due_fires_and_reschedules() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(
        rows=(
            Timer(
                id=2,
                label="Утро",
                kind=TimerKind.REMINDER,
                cron=cron_daily(8, 0),
                fire_at=moment - timedelta(minutes=1),
            ),
        )
    )
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    notes: list[NotificationRequested] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    bus.subscribe(NotificationRequested, notes.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment)
    assert sched.fire_due() == [2]
    assert [event.timer_id for event in fired] == [2]
    assert notes[0].action == "snooze"
    assert repo.rows[2].fire_at is not None
    assert repo.rows[2].fire_at > moment
    assert repo.rows[2].fire_at.hour == 8


def test_scheduler_recover_marks_missed_and_reschedules() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(
        rows=(
            Timer(id=1, label="Утро", cron=cron_daily(8, 0), fire_at=moment - timedelta(hours=1)),
            Timer(id=2, label="Чай", fire_at=moment - timedelta(minutes=5)),
        ),
        bare=(Timer(id=None, label="без id"),),
    )
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment)
    sched.recover(now=moment)
    assert [event.timer_id for event in fired] == [2]
    assert repo.rows[2].enabled is False
    assert repo.rows[1].fire_at is not None
    assert repo.rows[1].fire_at > moment


def test_scheduler_recover_skip_policy_disables() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(rows=(Timer(id=2, label="Чай", fire_at=moment - timedelta(minutes=5)),))
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment, missed_policy=MissedPolicy.SKIP)
    sched.recover(now=moment)
    assert fired == []
    assert repo.rows[2].enabled is False


def test_scheduler_recover_too_old_fire_now_disables() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(rows=(Timer(id=2, label="Чай", fire_at=moment - timedelta(minutes=30)),))
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment, missed_policy=MissedPolicy.FIRE_NOW)
    sched.recover(now=moment)
    assert fired == []
    assert repo.rows[2].enabled is False


def test_scheduler_recover_fire_now_within_grace_fires_not_missed() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(rows=(Timer(id=2, label="Чай", fire_at=moment - timedelta(minutes=5)),))
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    notes: list[NotificationRequested] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    bus.subscribe(NotificationRequested, notes.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment, missed_policy=MissedPolicy.FIRE_NOW)
    sched.recover(now=moment)
    assert [event.timer_id for event in fired] == [2]
    assert notes[0].message == "Чай"
    assert repo.rows[2].enabled is False


def test_scheduler_fire_swallows_broken_notifier() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(rows=(Timer(id=2, label="Чай", fire_at=moment - timedelta(minutes=1)),))
    bus = EventBus(thread_id=None)
    fired: list[TimerFired] = []
    bus.subscribe(TimerFired, fired.append, weak=False)
    sched = _scheduler(repo, bus, clock=lambda: moment, notifier=_BoomNotifier())
    assert sched.fire_due() == [2]
    assert [event.timer_id for event in fired] == [2]
    assert repo.rows[2].enabled is False


def test_scheduler_start_arms_and_stop_cancels() -> None:
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    repo = _FakeTimerRepo(rows=(Timer(id=1, label="Чай", fire_at=moment + timedelta(hours=1)),))
    bus = EventBus(thread_id=None)
    # tz=None on purpose: exercises _system_tz().
    sched = TimerScheduler(_SchedRepos(repo), bus, clock=lambda: moment)  # type: ignore[arg-type]
    sched.start()
    first = sched._wait
    assert first is not None
    assert first.is_alive()
    sched._wake()
    second = sched._wait
    assert second is not None
    sched.stop()
    assert sched._wait is None
    first.join(1.0)
    second.join(1.0)
    assert not first.is_alive()
    assert not second.is_alive()


# ----------------------------------------------------------------------
# переносимый профиль
# ----------------------------------------------------------------------


def _zip_bytes(items: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in items.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_manifest_compatibility_warnings() -> None:
    newer = pp.BundleManifest(
        profile_name="Дом",
        db_schema_version=pp.DB_SCHEMA_VERSION + 5,
        config_schema_version=pp.CONFIG_SCHEMA_VERSION + 5,
    )
    warnings = newer.compatibility_warnings()
    assert len(warnings) == 2
    assert any("базы данных" in note for note in warnings)
    assert any("Настройки" in note for note in warnings)
    same = pp.BundleManifest(profile_name="Дом")
    assert same.compatibility_warnings() == ()


def test_bundle_preview_describe() -> None:
    manifest = pp.BundleManifest(
        profile_name="Дом",
        app_version="1.2.3",
        created_at=datetime(2026, 1, 2, 3, 4, tzinfo=UTC),
    )
    preview = pp.BundlePreview(
        manifest=manifest,
        commands=("Свет", "Музыка"),
        folders=("Дом / Кухня",),
        variables=("гость",),
        models=("stt/vosk 1",),
        sounds=("beep.wav",),
        conflicts=("Свет",),
        warnings=("Внимание",),
    )
    text = preview.describe()
    assert "Профиль «Дом»" in text
    assert "Команд: 2, папок: 1, переменных: 1, звуков: 1" in text
    assert "Ожидаемые модели: 1" in text
    assert "Совпадают имена команд: 1" in text
    assert "Внимание" in text
    bare = pp.BundlePreview(manifest=manifest)
    bare_text = bare.describe()
    assert "Ожидаемые модели" not in bare_text
    assert "Совпадают имена команд" not in bare_text


def test_import_report_totals_and_describe() -> None:
    report = pp.ImportReport(
        profile_name="Дом",
        added_commands=("a", "b"),
        replaced_commands=("c",),
        renamed_commands=(("d", "d (2)"),),
        skipped_commands=("e",),
        added_folders=("Кухня",),
        added_variables=("гость",),
        added_sounds=("s.wav",),
        missing_models=("stt/vosk 1",),
        dropped_settings=("audio.mic",),
    )
    assert report.total_commands == 4
    text = report.describe()
    assert "Импортировано из профиля «Дом»." in text
    assert "Добавлено команд: 2" in text
    assert "Перезаписано команд: 1" in text
    assert "Переименовано команд: 1 (d → d (2))" in text
    assert "Пропущено команд: 1" in text
    assert "Создано папок: 1" in text
    assert "Добавлено переменных: 1" in text
    assert "Добавлено звуков: 1" in text
    assert "Не хватает моделей: stt/vosk 1" in text
    assert "Настройки, которые не удалось применить: audio.mic" in text
    with_backup = pp.ImportReport(profile_name="Дом", backup=Path("backup.zip"))
    assert "Резервная копия: backup.zip" in with_backup.describe()


def test_scalar_coercions() -> None:
    assert pp._as_text("hi") == "hi"
    assert pp._as_text(5) == ""
    assert pp._as_int(True) is None
    assert pp._as_int(5) == 5
    assert pp._as_int(" 7 ") == 7
    assert pp._as_int("x") is None
    assert pp._as_int(1.5) is None
    assert pp._as_bool(True, default=False) is True
    assert pp._as_bool("x", default=True) is True
    assert pp._as_bool(None, default=False) is False
    assert pp._as_datetime("2026-01-02T03:04:05") == datetime(2026, 1, 2, 3, 4, 5)
    fallback = pp._as_datetime("не дата")
    assert isinstance(fallback, datetime)
    assert fallback.tzinfo is not None
    assert isinstance(pp._as_datetime(123), datetime)


def test_folder_label_and_unique_name() -> None:
    assert pp._folder_label(["Дом", "Кухня"]) == "Дом / Кухня"
    assert pp._unique_name("Свет", []) == "Свет"
    assert pp._unique_name("Свет", ["Свет"]) == "Свет (2)"
    assert pp._unique_name("Свет", ["Свет", "Свет (2)"]) == "Свет (3)"


def test_reserved_and_unsafe_names() -> None:
    assert pp._is_reserved_device("CON") is True
    assert pp._is_reserved_device("CON.wav") is True
    assert pp._is_reserved_device("concert.wav") is False
    assert pp._is_reserved_device("COM1") is True
    assert pp._is_reserved_device("com.txt") is False
    assert pp._is_unsafe_name("") is True
    assert pp._is_unsafe_name("/etc/passwd") is True
    assert pp._is_unsafe_name("a:b") is True
    assert pp._is_unsafe_name("a\\b") is True
    assert pp._is_unsafe_name("../x") is True
    assert pp._is_unsafe_name("a/./b") is True
    assert pp._is_unsafe_name("sounds/CON.wav") is True
    assert pp._is_unsafe_name("sounds/beep.wav") is False


def test_model_label() -> None:
    assert pp._model_label({"kind": "stt", "name": "vosk", "version": "1.0"}) == "stt/vosk 1.0"
    assert pp._model_label({}) == "?/?"


def test_resolve_conflict() -> None:
    assert pp._resolve_conflict("Свет", exists=False, policy=pp.ConflictPolicy.SKIP, taken=[]) == (
        "Свет"
    )
    assert (
        pp._resolve_conflict("Свет", exists=True, policy=pp.ConflictPolicy.SKIP, taken=["Свет"])
        is None
    )
    assert (
        pp._resolve_conflict("Свет", exists=True, policy=pp.ConflictPolicy.RENAME, taken=["Свет"])
        == "Свет (2)"
    )
    assert (
        pp._resolve_conflict(
            "Свет", exists=True, policy=pp.ConflictPolicy.OVERWRITE, taken=["Свет"]
        )
        == "Свет"
    )


def test_trigger_from_json_valid_and_fallback() -> None:
    valid = pp._trigger_from_json(
        {"type": "hotkey", "payload": {"combo": "ctrl+a"}, "fuzzy": False, "priority": 3},
        command_id=7,
    )
    assert valid.command_id == 7
    assert valid.type is TriggerType.HOTKEY
    assert valid.payload == {"combo": "ctrl+a"}
    assert valid.fuzzy is False
    assert valid.priority == 3
    fallback = pp._trigger_from_json({"type": "чепуха"}, command_id=8)
    assert fallback.type is TriggerType.VOICE
    assert fallback.payload == {}
    assert fallback.fuzzy is True
    assert fallback.priority == 0


def test_entries_reads_manifest_json() -> None:
    manifest = json.dumps({"format": "ayris-profile", "schema_version": 1}).encode("utf-8")
    data = _zip_bytes({"manifest.json": manifest, "sounds/beep.wav": b"RIFF"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        entries = pp._entries(bundle, Path("bundle.zip"))
        assert "manifest.json" in entries
        parsed = pp._read_json(bundle, "manifest.json", Path("bundle.zip"))
        assert parsed["format"] == "ayris-profile"


def test_entries_rejects_unsafe_entry() -> None:
    data = _zip_bytes({"../evil.wav": b"x"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle, pytest.raises(ProfileError):
        pp._entries(bundle, Path("bundle.zip"))


def test_entries_rejects_too_many(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "MAX_ENTRIES", 1)
    data = _zip_bytes({"a.bin": b"1", "b.bin": b"2"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle, pytest.raises(ProfileError):
        pp._entries(bundle, Path("bundle.zip"))


def test_entries_rejects_large_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "MAX_ENTRY_BYTES", 3)
    data = _zip_bytes({"a.bin": b"0123456789"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle, pytest.raises(ProfileError):
        pp._entries(bundle, Path("bundle.zip"))


def test_entries_rejects_total_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pp, "MAX_TOTAL_BYTES", 5)
    monkeypatch.setattr(pp, "MAX_ENTRY_BYTES", 100)
    data = _zip_bytes({"a.bin": b"1234", "b.bin": b"5678"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle, pytest.raises(ProfileError):
        pp._entries(bundle, Path("bundle.zip"))


def test_read_bytes_missing_raises() -> None:
    data = _zip_bytes({"present.bin": b"x"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle, pytest.raises(ProfileError):
        pp._read_bytes(bundle, "absent.bin", Path("bundle.zip"))


def test_read_json_non_dict_and_bad_json_raise() -> None:
    data = _zip_bytes({"array.json": b"[1, 2, 3]", "broken.json": b"{not json"})
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        with pytest.raises(ProfileError):
            pp._read_json(bundle, "array.json", Path("bundle.zip"))
        with pytest.raises(ProfileError):
            pp._read_json(bundle, "broken.json", Path("bundle.zip"))


# ----------------------------------------------------------------------
# контекст диалога
# ----------------------------------------------------------------------


def test_guess_gender() -> None:
    assert ctx.guess_gender("") is ctx.Gender.MASCULINE
    assert ctx.guess_gender("книга") is ctx.Gender.FEMININE
    assert ctx.guess_gender("окно") is ctx.Gender.NEUTER
    assert ctx.guess_gender("часы") is ctx.Gender.PLURAL
    assert ctx.guess_gender("дверь") is ctx.Gender.FEMININE
    assert ctx.guess_gender("стол") is ctx.Gender.MASCULINE
    assert ctx.guess_gender("Chrome") is ctx.Gender.MASCULINE


@pytest.mark.skipif(sys.platform != "win32", reason="WinAPI только на Windows")
def test_win_function_and_process_name() -> None:
    assert sys.platform == "win32"
    assert callable(ctx._win_function("user32", "GetForegroundWindow"))
    assert ctx._win_function("user32", "NoSuchEntryPointXYZ") is None
    assert ctx._win_function("no_such_library_xyz", "X") is None
    assert ctx._process_name(0) == ""
    assert ctx._process_name(-5) == ""


def test_context_object_from_json_roundtrip() -> None:
    original = ctx.ContextObject(
        kind=ctx.ObjectKind.APP,
        name="Chrome",
        value="chrome.exe",
        gender=ctx.Gender.MASCULINE,
        at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert ctx.ContextObject.from_json(original.as_json()) == original
    assert ctx.ContextObject.from_json({"name": "x", "at": "2026-01-01T00:00:00+00:00"}) is None
    assert ctx.ContextObject.from_json({"kind": "app", "at": "2026-01-01T00:00:00+00:00"}) is None


def test_last_command_from_json_roundtrip() -> None:
    original = ctx.LastCommand(
        command_id=5,
        intent="open",
        action="app.launch",
        phrase="открой браузер",
        slots={"n": 2},
        result="ok",
        dangerous=True,
        at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC),
    )
    assert ctx.LastCommand.from_json(original.as_json()) == original
    assert ctx.LastCommand.from_json({}) is None


def test_last_answer_from_json_roundtrip() -> None:
    original = ctx.LastAnswer(text="готово", at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
    assert ctx.LastAnswer.from_json(original.as_json()) == original
    assert ctx.LastAnswer.from_json({"text": "", "at": "2026-01-01T00:00:00+00:00"}) is None
    assert ctx.LastAnswer.from_json({"text": "x"}) is None


def test_jsonable_variants() -> None:
    assert ctx._jsonable(None) is None
    assert ctx._jsonable(True) is True
    assert ctx._jsonable(3) == 3
    assert ctx._jsonable("x") == "x"
    moment = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    assert ctx._jsonable(moment) == moment.isoformat()
    assert ctx._jsonable({"a": 1}) == {"a": 1}
    assert ctx._jsonable([1, 2]) == [1, 2]
    assert sorted(ctx._jsonable(frozenset({1, 2}))) == [1, 2]
    assert isinstance(ctx._jsonable(object()), str)


def test_as_time_variants() -> None:
    assert ctx._as_time("") is None
    assert ctx._as_time(123) is None
    assert ctx._as_time("не дата") is None
    naive = ctx._as_time("2026-01-01T12:00:00")
    assert naive is not None
    assert naive.tzinfo is UTC
    aware = ctx._as_time("2026-01-01T12:00:00+03:00")
    assert aware is not None
    assert aware.utcoffset() == timedelta(hours=3)


def test_enum_member_variants() -> None:
    assert ctx._enum_member(ctx.ObjectKind, "app") is ctx.ObjectKind.APP
    assert ctx._enum_member(ctx.ObjectKind, "нет") is None
    assert ctx._enum_member(ctx.ObjectKind, 5) is None


def test_context_snapshot_variable_lookup() -> None:
    snap = ctx.ContextSnapshot(variables={"Свет": "вкл", "Гость": "Аня"})
    assert snap.variable("Свет") == "вкл"
    assert snap.variable("гость") == "Аня"
    assert snap.variable("нет", default="d") == "d"


def test_dialog_context_window_probe_success_and_failure() -> None:
    ok = ctx.DialogContext(
        window_probe=lambda: ctx.WindowInfo(title="Блокнот", process="notepad.exe"),
        autosave=False,
    )
    window = ok.snapshot().window
    assert window is not None
    assert window.process == "notepad.exe"

    def boom() -> ctx.WindowInfo | None:
        raise RuntimeError("нет окна")

    broken = ctx.DialogContext(window_probe=boom, autosave=False)
    assert broken.snapshot().window is None
