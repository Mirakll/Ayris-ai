"""Task 04: the event bus, the state machine and the application lifecycle.

No Qt anywhere. The bus deliberately knows nothing about Qt, so "the UI thread"
is simulated the same way the real entry point wires it: the bus is created on
this thread, a background thread publishes, the wake-up callback fires, and this
thread drains. If that works here it works with a queued Qt signal, because the
signal only replaces the wake-up.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path

import pytest

from ayris.core.app import (
    AlreadyRunningError,
    AppOptions,
    AyrisApp,
    Component,
    LifecycleStage,
)
from ayris.core.config import RestartScope
from ayris.core.events import (
    AudioLevelChanged,
    CancelRequested,
    ConfigChanged,
    Event,
    EventBus,
    MicToggled,
    ModeChanged,
    NotificationRequested,
    TranscriptReady,
    WakeWordDetected,
)
from ayris.core.state import AssistantState, MicMode, StateMachine, StatusSnapshot

pytestmark = pytest.mark.unit


class Collector:
    """A subscriber that is an object, so weak references have something to hold."""

    def __init__(self) -> None:
        self.seen: list[Event] = []

    def handle(self, event: Event) -> None:
        self.seen.append(event)


def _options(tmp_path: Path, **overrides: object) -> AppOptions:
    """Application options pointed at an isolated profile."""
    defaults: dict[str, object] = {
        "profile": tmp_path / "profile",
        "log_level": "DEBUG",
        "console_log": False,
        "watch_config": False,
        "single_instance": False,
    }
    defaults.update(overrides)
    return AppOptions(**defaults)  # type: ignore[arg-type]


class TestSubscribe:
    def test_publish_reaches_the_subscriber(self) -> None:
        bus = EventBus()
        collector = Collector()
        bus.subscribe(WakeWordDetected, collector.handle)

        bus.publish(WakeWordDetected(phrase="айрис", confidence=0.9))

        assert len(collector.seen) == 1
        event = collector.seen[0]
        assert isinstance(event, WakeWordDetected)
        assert event.phrase == "айрис"
        assert event.at.tzinfo is not None, "события помечаются временем в UTC"

    def test_other_event_types_are_not_delivered(self) -> None:
        bus = EventBus()
        collector = Collector()
        bus.subscribe(WakeWordDetected, collector.handle)

        bus.publish(CancelRequested(reason="стоп"))

        assert collector.seen == []

    def test_subscribing_to_the_base_class_sees_everything(self) -> None:
        """What the DevTools log view and the pipeline tracer rely on."""
        bus = EventBus()
        everything = Collector()
        specific = Collector()
        bus.subscribe(Event, everything.handle)
        bus.subscribe(TranscriptReady, specific.handle)

        bus.publish(TranscriptReady(text="привет"))
        bus.publish(CancelRequested())

        assert len(everything.seen) == 2
        assert len(specific.seen) == 1

    def test_handlers_run_in_subscription_order(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe(CancelRequested, lambda _event: order.append("first"))
        bus.subscribe(CancelRequested, lambda _event: order.append("second"))

        bus.publish(CancelRequested())

        assert order == ["first", "second"]

    def test_a_failing_handler_does_not_stop_the_others(self) -> None:
        bus = EventBus()
        collector = Collector()

        def explode(_event: Event) -> None:
            raise RuntimeError("подписчик сломался")

        bus.subscribe(CancelRequested, explode)
        bus.subscribe(CancelRequested, collector.handle)

        bus.publish(CancelRequested())

        assert len(collector.seen) == 1
        assert bus.failed == 1

    def test_nested_publish_stays_flat_and_ordered(self) -> None:
        """A handler that publishes must not recurse into a second dispatch."""
        bus = EventBus()
        order: list[str] = []

        def on_wake(_event: Event) -> None:
            order.append("wake")
            bus.publish(CancelRequested())
            order.append("wake-done")

        bus.subscribe(WakeWordDetected, on_wake)
        bus.subscribe(CancelRequested, lambda _event: order.append("cancel"))

        bus.publish(WakeWordDetected(phrase="айрис"))

        assert order == ["wake", "wake-done", "cancel"]


class TestUnsubscribe:
    def test_returned_callable_removes_the_subscription(self) -> None:
        bus = EventBus()
        collector = Collector()
        unsubscribe = bus.subscribe(CancelRequested, collector.handle)

        unsubscribe()
        bus.publish(CancelRequested())

        assert collector.seen == []
        assert bus.subscriber_count() == 0

    def test_unsubscribing_twice_is_harmless(self) -> None:
        bus = EventBus()
        collector = Collector()
        unsubscribe = bus.subscribe(CancelRequested, collector.handle)

        unsubscribe()
        unsubscribe()

        assert bus.subscriber_count() == 0

    def test_explicit_unsubscribe_by_handler(self) -> None:
        bus = EventBus()
        collector = Collector()
        bus.subscribe(CancelRequested, collector.handle)

        assert bus.unsubscribe(CancelRequested, collector.handle) is True
        assert bus.unsubscribe(CancelRequested, collector.handle) is False

        bus.publish(CancelRequested())
        assert collector.seen == []

    def test_a_collected_subscriber_leaks_nothing(self) -> None:
        """A closed window that forgot to unsubscribe must still be collectable."""
        bus = EventBus()
        collector = Collector()
        reference = __import__("weakref").ref(collector)
        bus.subscribe(AudioLevelChanged, collector.handle)
        assert bus.subscriber_count() == 1

        del collector
        gc.collect()

        assert reference() is None, "шина удержала подписчика сильной ссылкой"
        assert bus.subscriber_count() == 0
        # Publishing must not resurrect or crash on the dead subscription.
        bus.publish(AudioLevelChanged(rms=0.1))
        assert bus.failed == 0

    def test_a_lambda_is_held_strongly(self) -> None:
        """A weak reference to a closure would die before the first event."""
        bus = EventBus()
        seen: list[Event] = []
        bus.subscribe(CancelRequested, lambda event: seen.append(event))

        gc.collect()
        bus.publish(CancelRequested())

        assert len(seen) == 1

    def test_clear_drops_subscriptions_and_queue(self) -> None:
        bus = EventBus(thread_id=None)
        collector = Collector()
        bus.subscribe(CancelRequested, collector.handle)

        bus.clear()
        bus.publish(CancelRequested())

        assert bus.subscriber_count() == 0
        assert collector.seen == []


class TestThreadDelivery:
    def test_an_event_from_a_worker_reaches_the_ui_thread(self) -> None:
        woken = threading.Event()
        bus = EventBus(wakeup=woken.set)
        collector = Collector()
        bus.subscribe(TranscriptReady, collector.handle)

        ui_thread = threading.get_ident()
        handled_on: list[int] = []
        bus.subscribe(TranscriptReady, lambda _event: handled_on.append(threading.get_ident()))

        def worker() -> None:
            bus.publish(TranscriptReady(text="из фонового потока"))

        thread = threading.Thread(target=worker, name="probe-worker")
        thread.start()
        thread.join(5)

        # Nothing is delivered until the UI thread drains, which is the whole
        # point: a handler may touch widgets.
        assert collector.seen == []
        assert woken.wait(5), "шина не разбудила поток доставки"
        assert bus.pending == 1

        assert bus.drain() == 1
        assert len(collector.seen) == 1
        assert handled_on == [ui_thread]

    def test_publishing_on_the_delivery_thread_is_inline(self) -> None:
        wakeups: list[int] = []
        bus = EventBus(wakeup=lambda: wakeups.append(1))
        collector = Collector()
        bus.subscribe(CancelRequested, collector.handle)

        bus.publish(CancelRequested())

        assert len(collector.seen) == 1
        assert wakeups == [], "локальная публикация не должна будить поток"

    def test_drain_respects_its_batch_limit_and_asks_again(self) -> None:
        """A burst of audio levels must not monopolise the Qt event loop."""
        wakeups: list[int] = []
        bus = EventBus(wakeup=lambda: wakeups.append(1))
        collector = Collector()
        bus.subscribe(AudioLevelChanged, collector.handle)

        def worker() -> None:
            for index in range(10):
                bus.publish(AudioLevelChanged(rms=index / 10))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(5)

        assert bus.drain(limit=4) == 4
        assert bus.pending == 6
        # The bus re-armed the wake-up, so the loop knows to come back.
        assert len(wakeups) >= 2

        assert bus.drain(limit=0) == 6
        assert len(collector.seen) == 10

    def test_an_unbound_bus_delivers_on_the_publishing_thread(self) -> None:
        """What a worker process gets: no marshalling, no queue, no wake-up."""
        bus = EventBus(thread_id=None)
        handled_on: list[int] = []
        bus.subscribe(CancelRequested, lambda _event: handled_on.append(threading.get_ident()))

        thread = threading.Thread(target=lambda: bus.publish(CancelRequested()))
        thread.start()
        thread.join(5)

        assert handled_on and handled_on[0] != threading.get_ident()

    def test_concurrent_publishers_lose_nothing(self) -> None:
        bus = EventBus()
        collector = Collector()
        bus.subscribe(AudioLevelChanged, collector.handle)

        def worker(offset: int) -> None:
            for index in range(50):
                bus.publish(AudioLevelChanged(rms=(offset + index) / 100))

        threads = [threading.Thread(target=worker, args=(base,)) for base in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)

        assert bus.drain(limit=0) == 200
        assert len(collector.seen) == 200


class TestStateMachine:
    def test_a_state_change_publishes_an_event(self) -> None:
        bus = EventBus()
        state = StateMachine(bus)
        collector = Collector()
        bus.subscribe(ModeChanged, collector.handle)

        assert state.set_state(AssistantState.LISTENING) is True

        assert state.state is AssistantState.LISTENING
        assert len(collector.seen) == 1
        event = collector.seen[0]
        assert isinstance(event, ModeChanged)
        assert event.previous is AssistantState.IDLE
        assert event.state is AssistantState.LISTENING

    def test_repeating_a_state_publishes_nothing(self) -> None:
        bus = EventBus()
        state = StateMachine(bus)
        collector = Collector()
        bus.subscribe(ModeChanged, collector.handle)

        state.set_state(AssistantState.LISTENING)
        assert state.set_state(AssistantState.LISTENING) is False

        assert len(collector.seen) == 1

    def test_an_illegal_transition_is_refused(self) -> None:
        bus = EventBus()
        state = StateMachine(bus)
        state.fail("микрофон недоступен")

        assert state.state is AssistantState.ERROR
        # Only идle clears an error: a subsystem must not quietly resume.
        assert state.set_state(AssistantState.LISTENING) is False
        assert state.to_idle() is True
        assert state.state is AssistantState.IDLE

    def test_anything_can_fail(self) -> None:
        bus = EventBus()
        state = StateMachine(bus)
        state.set_state(AssistantState.SPEAKING)

        assert state.fail("синтез упал") is True
        assert state.snapshot.detail == "синтез упал"

    def test_microphone_changes_publish_mic_toggled(self) -> None:
        bus = EventBus()
        state = StateMachine(bus, initial=StatusSnapshot(mic_mode=MicMode.ALWAYS))
        collector = Collector()
        bus.subscribe(MicToggled, collector.handle)

        assert state.toggle_mic() is False
        assert state.set_mic_mode(MicMode.PTT) is True
        assert state.set_mic_mode(MicMode.PTT) is False

        assert len(collector.seen) == 2
        first = collector.seen[0]
        assert isinstance(first, MicToggled)
        assert first.enabled is False

    def test_wake_word_is_off_in_push_to_talk(self) -> None:
        snapshot = StatusSnapshot(mic_mode=MicMode.PTT)
        assert snapshot.listens_for_wake_word is False
        assert StatusSnapshot(mic_mode=MicMode.HYBRID).listens_for_wake_word is True
        assert StatusSnapshot(mic_enabled=False, mic_mode=MicMode.ALWAYS).listens_for_wake_word is (
            False
        )

    def test_online_flag_publishes_once(self) -> None:
        bus = EventBus()
        state = StateMachine(bus)
        seen: list[Event] = []
        bus.subscribe(Event, seen.append)

        assert state.set_online(online=True) is True
        assert state.set_online(online=True) is False

        assert len(seen) == 1
        assert state.online is True

    def test_a_worker_thread_may_drive_the_state(self) -> None:
        woken = threading.Event()
        bus = EventBus(wakeup=woken.set)
        state = StateMachine(bus)
        collector = Collector()
        bus.subscribe(ModeChanged, collector.handle)

        thread = threading.Thread(target=lambda: state.set_state(AssistantState.THINKING))
        thread.start()
        thread.join(5)

        assert state.state is AssistantState.THINKING, "состояние меняется сразу"
        assert collector.seen == [], "а событие ждёт UI-поток"
        assert woken.wait(5)
        bus.drain()
        assert len(collector.seen) == 1

    def test_snapshot_describes_itself_in_russian(self) -> None:
        snapshot = StatusSnapshot(state=AssistantState.LISTENING, mic_enabled=False)
        assert "слушаю" in snapshot.describe()
        assert "микрофон выключен" in snapshot.describe()
        assert "офлайн" in snapshot.describe()


class TestLifecycle:
    def test_startup_walks_every_stage_in_order(self, tmp_path: Path) -> None:
        with AyrisApp(_options(tmp_path)).startup() as app:
            assert app.running is True
            assert app.stages_started == tuple(LifecycleStage)
            assert app.paths.root == (tmp_path / "profile").resolve()
            assert app.database.is_closed is False
            assert app.profile.is_active is True
            assert app.state.state is AssistantState.IDLE
            assert app.bus.thread_id == threading.get_ident()

        assert app.running is False

    def test_the_log_records_the_order(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        logs_dir = app.paths.logs_dir
        app.shutdown()

        text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(logs_dir.glob("ayris_*.log"))
        )
        assert "Ayris запущен" in text
        assert "Ayris остановлен" in text
        for stage in LifecycleStage:
            assert stage.value in text, stage

    def test_components_stop_in_reverse_order(self, tmp_path: Path) -> None:
        order: list[str] = []

        def probe(name: str, stage: LifecycleStage) -> Component:
            return Component(
                name=name,
                stage=stage,
                start=lambda: order.append(f"start:{name}"),
                stop=lambda: order.append(f"stop:{name}"),
            )

        app = AyrisApp(_options(tmp_path))
        app.add_component(probe("actions", LifecycleStage.ACTIONS))
        app.add_component(probe("worker", LifecycleStage.WORKERS))
        app.add_component(probe("gui", LifecycleStage.GUI))

        app.startup()
        assert order == ["start:actions", "start:worker", "start:gui"]

        order.clear()
        app.shutdown()
        assert order == ["stop:gui", "stop:worker", "stop:actions"]

    def test_two_components_in_one_stage_keep_registration_order(self, tmp_path: Path) -> None:
        order: list[str] = []
        app = AyrisApp(_options(tmp_path))
        for name in ("first", "second"):
            app.add_component(
                Component(
                    name=name,
                    stage=LifecycleStage.WORKERS,
                    start=lambda name=name: order.append(f"start:{name}"),  # type: ignore[misc]
                    stop=lambda name=name: order.append(f"stop:{name}"),  # type: ignore[misc]
                )
            )

        app.startup()
        app.shutdown()

        assert order == ["start:first", "start:second", "stop:second", "stop:first"]

    def test_a_component_that_hangs_is_killed(self, tmp_path: Path) -> None:
        release = threading.Event()
        killed = threading.Event()

        app = AyrisApp(_options(tmp_path))
        app.add_component(
            Component(
                name="stuck",
                stage=LifecycleStage.WORKERS,
                stop=lambda: release.wait(30),
                kill=killed.set,
                stop_timeout=0.2,
            )
        )
        app.startup()
        app.shutdown()

        assert killed.is_set(), "зависшую подсистему должны снять принудительно"
        release.set()

    def test_a_failing_stop_does_not_abort_the_rest(self, tmp_path: Path) -> None:
        stopped: list[str] = []

        def explode() -> None:
            raise RuntimeError("не смог остановиться")

        app = AyrisApp(_options(tmp_path))
        app.add_component(Component(name="bad", stage=LifecycleStage.GUI, stop=explode))
        app.add_component(
            Component(
                name="good",
                stage=LifecycleStage.WORKERS,
                stop=lambda: stopped.append("good"),
            )
        )
        app.startup()
        app.shutdown()

        assert stopped == ["good"]
        assert app.running is False

    def test_a_failing_start_unwinds_what_already_came_up(self, tmp_path: Path) -> None:
        stopped: list[str] = []

        def explode() -> None:
            raise RuntimeError("подсистема не поднялась")

        app = AyrisApp(_options(tmp_path))
        app.add_component(
            Component(
                name="early",
                stage=LifecycleStage.ACTIONS,
                stop=lambda: stopped.append("early"),
            )
        )
        app.add_component(Component(name="late", stage=LifecycleStage.NLU, start=explode))

        with pytest.raises(RuntimeError):
            app.startup()

        assert stopped == ["early"]
        assert app.running is False
        assert app.stages_started == ()

    def test_shutdown_is_idempotent(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        app.shutdown()
        app.shutdown()

        assert app.running is False

    def test_starting_twice_is_refused(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        try:
            with pytest.raises(Exception, match="twice"):
                app.startup()
        finally:
            app.shutdown()

    def test_asking_for_a_subsystem_too_early_explains_itself(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path))
        with pytest.raises(Exception, match="not initialised") as caught:
            _ = app.database
        assert "Ayris" in getattr(caught.value, "user_message", "")

    def test_the_database_is_migrated_and_closed(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        database = app.database
        assert database.query_value("PRAGMA user_version") > 0

        app.shutdown()
        assert database.is_closed is True

    def test_transient_variables_do_not_survive_shutdown(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        variables = app.repositories.variables
        variables.set("постоянная", "да", persistent=True)
        variables.set("временная", "нет", persistent=False)
        app.shutdown()

        again = AyrisApp(_options(tmp_path)).startup()
        try:
            assert again.repositories.variables.get_value("постоянная") == "да"
            assert again.repositories.variables.get_value("временная") is None
        finally:
            again.shutdown()

    def test_a_component_added_after_startup_starts_at_once(self, tmp_path: Path) -> None:
        started: list[str] = []
        app = AyrisApp(_options(tmp_path)).startup()
        app.add_component(
            Component(
                name="plugin",
                stage=LifecycleStage.PLUGINS,
                start=lambda: started.append("plugin"),
                stop=lambda: started.append("stopped"),
            )
        )

        assert started == ["plugin"]
        app.shutdown()
        assert started == ["plugin", "stopped"]


class TestSingleInstance:
    def test_a_second_instance_is_refused(self, tmp_path: Path) -> None:
        first = AyrisApp(_options(tmp_path, single_instance=True)).startup()
        second = AyrisApp(_options(tmp_path, single_instance=True))
        try:
            with pytest.raises(AlreadyRunningError) as caught:
                second.startup()
            assert "уже запущен" in caught.value.user_message
            assert caught.value.recoverable is False
        finally:
            first.shutdown()

    def test_the_token_is_released_on_shutdown(self, tmp_path: Path) -> None:
        first = AyrisApp(_options(tmp_path, single_instance=True)).startup()
        first.shutdown()

        second = AyrisApp(_options(tmp_path, single_instance=True)).startup()
        try:
            assert second.running is True
        finally:
            second.shutdown()

    def test_the_guard_can_be_switched_off(self, tmp_path: Path) -> None:
        first = AyrisApp(_options(tmp_path, single_instance=False)).startup()
        second = AyrisApp(_options(tmp_path, single_instance=False))
        try:
            second.startup()
            assert second.running is True
        finally:
            second.shutdown()
            first.shutdown()


class TestConfigReaction:
    def test_a_settings_change_becomes_an_event(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        collector = Collector()
        app.bus.subscribe(ConfigChanged, collector.handle)
        try:
            app.config.apply({"voice.tts.speed": 1.4})

            assert len(collector.seen) == 1
            event = collector.seen[0]
            assert isinstance(event, ConfigChanged)
            assert event.touches("voice.tts")
            assert event.settings.voice.tts.speed == pytest.approx(1.4)
        finally:
            app.shutdown()

    def test_a_live_setting_is_applied_without_a_restart(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path, log_level=None)).startup()
        try:
            app.config.apply({"voice.wake.mic_mode": "ptt"})

            assert app.state.mic_mode is MicMode.PTT
            assert app.pending_restarts == frozenset()
        finally:
            app.shutdown()

    def test_a_change_that_needs_a_restart_calls_its_handler(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        restarted: list[str] = []
        app.register_restart_handler(RestartScope.STT, lambda _settings: restarted.append("stt"))
        try:
            app.config.apply({"performance.stt_threads": 3})

            assert restarted == ["stt"]
            assert RestartScope.STT not in app.pending_restarts
        finally:
            app.shutdown()

    def test_a_scope_nobody_handles_stays_pending(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        try:
            app.config.apply({"performance.stt_threads": 3})
            assert RestartScope.STT in app.pending_restarts
        finally:
            app.shutdown()

    def test_a_failing_restart_handler_keeps_the_scope_pending(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()

        def explode(_settings: object) -> None:
            raise RuntimeError("воркер не перезапустился")

        app.register_restart_handler(RestartScope.STT, explode)
        try:
            app.config.apply({"performance.stt_threads": 3})
            assert RestartScope.STT in app.pending_restarts
        finally:
            app.shutdown()


class TestExceptionHandling:
    def test_an_unhandled_exception_is_logged_and_announced(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        collector = Collector()
        app.bus.subscribe(NotificationRequested, collector.handle)
        logs_dir = app.paths.logs_dir

        import sys

        try:
            error = RuntimeError("что-то пошло не так")
            sys.excepthook(type(error), error, None)

            assert len(collector.seen) == 1
            event = collector.seen[0]
            assert isinstance(event, NotificationRequested)
            assert event.level == "error"
        finally:
            app.shutdown()

        text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(logs_dir.glob("ayris_*.log"))
        )
        assert "необработанное исключение" in text

    def test_an_exception_in_a_thread_is_logged(self, tmp_path: Path) -> None:
        app = AyrisApp(_options(tmp_path)).startup()
        logs_dir = app.paths.logs_dir
        try:

            def explode() -> None:
                raise RuntimeError("поток упал")

            thread = threading.Thread(target=explode, name="probe-crash")
            thread.start()
            thread.join(5)
        finally:
            app.shutdown()

        text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted(logs_dir.glob("ayris_*.log"))
        )
        assert "необработанное исключение в потоке probe-crash" in text

    def test_the_hooks_are_restored_on_shutdown(self, tmp_path: Path) -> None:
        import sys

        original = sys.excepthook
        original_thread_hook = threading.excepthook

        app = AyrisApp(_options(tmp_path)).startup()
        assert sys.excepthook is not original
        app.shutdown()

        assert sys.excepthook is original
        assert threading.excepthook is original_thread_hook
