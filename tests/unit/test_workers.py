"""Tests for the worker protocol, the base worker and the supervisor.

These start real processes. That is the point: the failures this infrastructure
exists to survive — a killed process, a wedged one, a pipe that closes mid-call —
cannot be reproduced with a mock, and a mocked worker would prove only that the
mock behaves.

Every test tears its manager down through :meth:`WorkerManager.shutdown`, and
:func:`test_shutdown_leaves_no_processes` asserts the thing the specification
actually asks for: nothing left running afterwards.
"""

from __future__ import annotations

import multiprocessing
import pickle
import struct
import sys
import threading
import time
from pathlib import Path

import pytest

from ayris.core.config import Settings
from ayris.core.errors import AyrisError, SttError
from ayris.core.events import EventBus, LogLine, WorkerCrashed, WorkerRestarted
from ayris.workers.manager import WorkerManager, WorkerStatus
from ayris.workers.protocol import (
    MAGIC,
    PROTOCOL_VERSION,
    AudioChunk,
    Control,
    ControlKind,
    ErrorInfo,
    Heartbeat,
    Ready,
    Request,
    Response,
    SharedAudioBlock,
    WorkerCrashError,
    WorkerError,
    WorkerEvent,
    WorkerProtocolError,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
    decode,
    encode,
    iter_frames,
    open_audio,
)
from ayris.workers.registry import WorkerKind, WorkerSpec, plan_workers

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ECHO = "echo_worker:EchoWorker"

#: Long enough that a loaded CI machine does not fail a spawn, short enough that a
#: genuine hang does not stall the suite.
START_TIMEOUT = 30.0


def echo_spec(name: str = "echo", **overrides: object) -> WorkerSpec:
    """A spec pointing at the echo worker in ``tests/fixtures``."""
    values: dict[str, object] = {
        "name": name,
        "kind": WorkerKind.TEST,
        "entrypoint": ECHO,
        "python_path": (str(FIXTURES),),
        "start_timeout": START_TIMEOUT,
        "call_timeout": 15.0,
        "stop_timeout": 5.0,
        "heartbeat_interval": 0.1,
        "restart_delay": 0.05,
        "max_restart_delay": 0.2,
    }
    values.update(overrides)
    return WorkerSpec(**values)  # type: ignore[arg-type]


@pytest.fixture
def bus() -> EventBus:
    """A bus that delivers on ``drain``, like the application's does."""
    return EventBus()


@pytest.fixture
def manager(bus: EventBus) -> object:
    """A manager that is always shut down, even when the test fails."""
    instance = WorkerManager(bus)
    try:
        yield instance
    finally:
        instance.shutdown()


def collect(bus: EventBus, event_type: type) -> list[object]:
    """Subscribe strongly and return the list the handler appends to."""
    received: list[object] = []
    bus.subscribe(event_type, received.append, weak=False)
    return received


def wait_for(predicate: object, timeout: float = 5.0, interval: float = 0.02) -> bool:
    """Poll ``predicate`` until it is true or the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(interval)
    return False


def drain_until(bus: EventBus, predicate: object, timeout: float = 5.0) -> bool:
    """Deliver bus events until ``predicate`` holds. Events arrive from threads."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        bus.drain()
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.02)
    bus.drain()
    return bool(predicate())  # type: ignore[operator]


# ----------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------


class TestProtocol:
    """Framing, versioning and error translation, with no processes involved."""

    @pytest.mark.parametrize(
        "message",
        [
            Request(id=1, method="echo", params={"a": 1}),
            Response(id=1, ok=True, payload={"b": [1, 2, 3]}),
            Response(id=2, ok=False, error=ErrorInfo(error_type="SttError", message="нет")),
            WorkerEvent(kind="level", payload={"rms": 0.5}),
            Heartbeat(seq=3, at=1.5, busy=True, handled=7),
            Control(kind=ControlKind.CANCEL, request_id=4),
            Ready(name="echo", pid=1, protocol_version=PROTOCOL_VERSION, methods=("echo",)),
        ],
    )
    def test_round_trip(self, message):
        assert decode(encode(message)) == message

    def test_rejects_foreign_bytes(self):
        with pytest.raises(WorkerProtocolError, match="magic"):
            decode(b"nonsense")

    def test_rejects_other_protocol_version(self):
        raw = encode(Request(id=1, method="echo"))
        forged = MAGIC + bytes((PROTOCOL_VERSION + 1,)) + raw[len(MAGIC) + 1 :]
        with pytest.raises(WorkerProtocolError, match="version"):
            decode(forged)

    def test_rejects_unknown_payload_type(self):
        forged = MAGIC + bytes((PROTOCOL_VERSION,)) + pickle.dumps({"not": "a message"})
        with pytest.raises(WorkerProtocolError):
            decode(forged)

    def test_error_info_keeps_the_ayris_type(self):
        info = ErrorInfo.from_exception(
            SttError("модель не загрузилась", user_message="Речь не распознана.")
        )
        restored = info.to_exception()
        assert isinstance(restored, SttError)
        assert restored.user_message == "Речь не распознана."
        assert "модель не загрузилась" in str(restored)

    def test_error_info_degrades_unknown_types(self):
        info = ErrorInfo(error_type="SomeLibraryError", message="boom")
        restored = info.to_exception()
        assert isinstance(restored, AyrisError)
        assert "SomeLibraryError" in str(restored)

    def test_error_info_from_plain_exception(self):
        info = ErrorInfo.from_exception(ValueError("bad value"))
        assert info.error_type == "ValueError"
        assert info.traceback == "" or "ValueError" in info.traceback


class TestSharedAudio:
    """Shared memory: descriptors on the pipe, samples out of band."""

    def test_block_round_trip_in_process(self):
        pcm = struct.pack("<4h", 1, -2, 3, -4)
        with SharedAudioBlock.create(pcm, sample_rate=16000) as block:
            params = block.chunk.to_params({"language": "ru"})
            assert params["language"] == "ru"
            chunk = AudioChunk.from_params(params)
            assert chunk is not None
            with open_audio(chunk) as view:
                assert bytes(view) == pcm

    def test_chunk_reports_duration(self):
        block = SharedAudioBlock.create(b"\x00" * (16000 * 2), sample_rate=16000)
        try:
            assert block.chunk.frames == 16000
            assert block.chunk.duration_ms == pytest.approx(1000.0)
        finally:
            block.close()

    def test_from_params_without_audio(self):
        assert AudioChunk.from_params({"language": "ru"}) is None

    def test_iter_frames_splits_evenly(self):
        pcm = b"\x00\x01" * 100
        chunk = AudioChunk(block="unused", nbytes=len(pcm))
        frames = list(iter_frames(pcm, chunk, 25))
        assert [len(frame) for frame in frames] == [50, 50, 50, 50]


# ----------------------------------------------------------------------
# the happy path
# ----------------------------------------------------------------------


class TestWorkerLifecycle:
    def test_start_call_stop(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")

        assert manager.is_ready("echo")
        assert manager.call_sync("echo", "add", {"a": 2, "b": 3}) == 5
        assert manager.call_sync("echo", "say_name") == "echo"

        summary = manager.status()[0]
        assert summary.status is WorkerStatus.READY
        assert summary.pid not in (None, 0)
        assert "echo" in summary.methods

        manager.stop("echo")
        assert manager.worker_status("echo") is WorkerStatus.STOPPED
        assert manager.status()[0].pid is None

    def test_runs_in_another_process(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        answer = manager.call_sync("echo", "echo", {"x": 1})
        assert answer["echo"] == {"x": 1}
        assert answer["pid"] != multiprocessing.current_process().pid

    def test_spawn_inherits_nothing(self, manager: WorkerManager):
        sys.path.insert(0, str(FIXTURES))
        try:
            import echo_worker

            echo_worker.poison()
            manager.register(echo_spec(params={"model": "tiny"}))
            manager.start("echo")
            state = manager.call_sync("echo", "inherited")
        finally:
            sys.path.remove(str(FIXTURES))
        assert state["poisoned"] is False, "worker inherited parent state"
        assert state["params"] == {"model": "tiny"}
        assert state["has_qt"] is False, "worker imported Qt"

    def test_builtin_methods(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        assert manager.call_sync("echo", "ping") is True
        stats = manager.call_sync("echo", "stats")
        assert stats["name"] == "echo"
        assert stats["handled"] >= 1
        assert manager.call_sync("echo", "configure", {"volume": 3}) is True
        assert manager.call_sync("echo", "inherited")["params"]["volume"] == 3

    def test_heartbeat_keeps_arriving(self, manager: WorkerManager):
        manager.register(echo_spec(heartbeat_interval=0.05))
        manager.start("echo")
        assert wait_for(lambda: (manager.status()[0].last_heartbeat_age or 9) < 0.5)

    def test_unknown_worker(self, manager: WorkerManager):
        with pytest.raises(WorkerUnavailableError):
            manager.call("nope", "echo")

    def test_duplicate_name_while_running(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(WorkerStartError, match="already running"):
            manager.register(echo_spec())


class TestErrors:
    def test_typed_error_crosses_the_pipe(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(SttError, match="recognition exploded") as caught:
            manager.call_sync("echo", "boom", {"typed": True})
        assert caught.value.user_message == "Распознавание сломалось."
        assert manager.is_ready("echo"), "a handler error must not kill the worker"

    def test_plain_error_becomes_an_ayris_error(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(AyrisError, match="something entirely unexpected"):
            manager.call_sync("echo", "boom", {"typed": False})

    def test_unknown_method(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(AyrisError, match="unknown worker method"):
            manager.call_sync("echo", "no_such_method")
        assert manager.is_ready("echo")

    def test_unpicklable_result_reports_instead_of_hanging(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(AyrisError):
            manager.call_sync("echo", "unpicklable")
        assert manager.is_ready("echo")

    def test_failure_during_start_is_reported(self, manager: WorkerManager):
        manager.register(echo_spec(params={"fail_on_start": True}))
        with pytest.raises(WorkerStartError, match="was told to fail"):
            manager.start("echo")
        assert manager.worker_status("echo") is WorkerStatus.FAILED

    def test_missing_entrypoint_is_reported(self, manager: WorkerManager):
        manager.register(echo_spec(entrypoint="echo_worker:NoSuchWorker"))
        with pytest.raises(WorkerStartError):
            manager.start("echo")

    def test_autostart_off_refuses_to_wake_a_worker(self, manager: WorkerManager):
        manager.register(echo_spec())
        with pytest.raises(WorkerUnavailableError, match="autostart"):
            manager.call("echo", "ping", autostart=False)


class TestTimeoutAndCancel:
    def test_call_timeout_leaves_the_worker_alive(self, manager: WorkerManager):
        manager.register(echo_spec(call_timeout=0.3))
        manager.start("echo")
        with pytest.raises(WorkerTimeoutError):
            manager.call_sync("echo", "sleep", {"seconds": 5.0})
        assert manager.is_ready("echo")
        assert manager.call_sync("echo", "add", {"a": 1, "b": 1}, timeout=10.0) == 2

    def test_per_call_timeout_overrides_the_spec(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(WorkerTimeoutError):
            manager.call_sync("echo", "sleep", {"seconds": 5.0}, timeout=0.3)

    def test_cancel_stops_a_cooperative_handler(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        future = manager.call("echo", "sleep", {"seconds": 10.0})
        assert wait_for(lambda: manager.status()[0].pending_calls == 1)
        manager.cancel_all()
        with pytest.raises(AyrisError):
            future.result(10.0)
        assert manager.is_ready("echo")

    def test_futures_are_independent(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        futures = [manager.call("echo", "add", {"a": index, "b": 1}) for index in range(20)]
        assert [future.result(15.0) for future in futures] == list(range(1, 21))


class TestEventsAndLogs:
    def test_worker_events_reach_the_bus(self, bus: EventBus, manager: WorkerManager):
        seen: list[tuple[str, object]] = []
        manager.register(echo_spec())
        manager.set_event_translator(
            "echo",
            lambda kind, payload: (
                seen.append((kind, payload)) or LogLine(level="INFO", message=kind)
            ),
        )
        manager.start("echo")
        assert manager.call_sync("echo", "emit", {"count": 3, "kind": "tick"}) == 3
        assert wait_for(lambda: len(seen) == 3)
        assert [kind for kind, _ in seen] == ["tick", "tick", "tick"]

    def test_untranslated_events_are_dropped(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        assert manager.call_sync("echo", "emit", {"count": 2}) == 2
        assert manager.is_ready("echo")

    def test_worker_log_is_forwarded(self, bus: EventBus, manager: WorkerManager):
        lines = collect(bus, LogLine)
        manager.register(echo_spec(log_level="INFO"))
        manager.start("echo")
        manager.call_sync("echo", "warn", {"message": "тестовое предупреждение"})
        assert drain_until(
            bus,
            lambda: any(
                "тестовое предупреждение" in getattr(line, "message", "") for line in lines
            ),
        )


class TestAudioTransfer:
    def test_ten_seconds_of_audio_through_shared_memory(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        pcm = bytes(range(256)) * 1250  # 320 000 bytes: 10 s of 16 kHz mono int16
        assert len(pcm) == 16000 * 2 * 10

        started = time.monotonic()
        summary = manager.call_sync("echo", "audio_summary", {"audio_hint": True}, audio=pcm)
        elapsed = time.monotonic() - started

        assert summary["received"] is True
        assert summary["nbytes"] == len(pcm)
        assert summary["duration_ms"] == pytest.approx(10_000.0)
        assert summary["checksum"] == sum(pcm) % 1_000_003
        assert elapsed < 2.0, f"shared memory transfer took {elapsed:.2f} s"

    def test_audio_block_is_released_after_the_call(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        for _ in range(5):
            assert manager.call_sync("echo", "audio_summary", audio=b"\x01\x02" * 8000)["received"]
        assert manager.status()[0].pending_calls == 0

    def test_audio_block_is_released_after_a_timeout(self, manager: WorkerManager):
        manager.register(echo_spec(call_timeout=0.3))
        manager.start("echo")
        with pytest.raises(WorkerTimeoutError):
            manager.call_sync("echo", "sleep", {"seconds": 5.0}, audio=b"\x00" * 32000)
        assert wait_for(lambda: manager.status()[0].pending_calls == 0)


class TestCrashAndRestart:
    def test_kill_leads_to_restart(self, bus: EventBus, manager: WorkerManager):
        crashes = collect(bus, WorkerCrashed)
        restarts = collect(bus, WorkerRestarted)
        manager.register(echo_spec())
        manager.start("echo")
        first_pid = manager.status()[0].pid

        with pytest.raises(AyrisError):
            manager.call_sync("echo", "die", {"code": 7})

        assert wait_for(lambda: manager.is_ready("echo"), timeout=10.0), "worker did not come back"
        second_pid = manager.status()[0].pid
        assert second_pid != first_pid
        assert manager.call_sync("echo", "add", {"a": 1, "b": 2}) == 3

        assert drain_until(bus, lambda: crashes and restarts)
        assert crashes[0].worker == "echo"
        assert restarts[0].worker == "echo"

    def test_pending_calls_fail_when_the_worker_dies(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        slow = manager.call("echo", "sleep", {"seconds": 30.0})
        assert wait_for(lambda: manager.status()[0].pending_calls >= 1)
        manager.call("echo", "die", {"code": 3})
        with pytest.raises(AyrisError):
            slow.result(10.0)

    def test_hung_worker_is_restarted(self, bus: EventBus, manager: WorkerManager):
        crashes = collect(bus, WorkerCrashed)
        manager.register(echo_spec(heartbeat_interval=0.1, heartbeat_misses=3, call_timeout=20.0))
        manager.start("echo")
        first_pid = manager.status()[0].pid

        manager.call("echo", "wedge", {"seconds": 30.0})
        assert wait_for(
            lambda: manager.status()[0].pid not in (None, first_pid),
            timeout=15.0,
        ), "a worker that stopped beating was not replaced"
        assert drain_until(bus, lambda: any("отклик" in event.error for event in crashes))

    def test_restart_limit_gives_up(self, bus: EventBus, manager: WorkerManager):
        manager.register(echo_spec(params={"fail_on_start": True}, max_restarts=2))
        with pytest.raises(WorkerStartError):
            manager.start("echo")
        assert manager.worker_status("echo") is WorkerStatus.FAILED
        # A failed worker stays failed: nothing keeps hammering a broken model.
        time.sleep(0.5)
        assert manager.worker_status("echo") is WorkerStatus.FAILED
        with pytest.raises(WorkerUnavailableError):
            manager.call("echo", "ping")

    def test_manual_restart_keeps_the_registration(self, bus: EventBus, manager: WorkerManager):
        restarts = collect(bus, WorkerRestarted)
        manager.register(echo_spec())
        manager.start("echo")
        first_pid = manager.status()[0].pid
        manager.restart("echo", reason="проверка")
        assert manager.is_ready("echo")
        assert manager.status()[0].pid != first_pid
        assert drain_until(bus, lambda: any(event.reason == "проверка" for event in restarts))


class TestShutdown:
    def test_shutdown_leaves_no_processes(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec("one"))
        instance.register(echo_spec("two"))
        instance.start("one")
        instance.start("two")
        children = [
            child
            for child in multiprocessing.active_children()
            if child.name.startswith("ayris-")
        ]
        assert len(children) == 2

        instance.shutdown()

        assert wait_for(lambda: not any(child.is_alive() for child in children), timeout=10.0)
        assert instance.status() == ()
        assert not [
            child
            for child in multiprocessing.active_children()
            if child.name.startswith("ayris-")
        ]

    def test_shutdown_fails_pending_calls(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        slow = instance.call("echo", "sleep", {"seconds": 30.0})
        assert wait_for(lambda: instance.status()[0].pending_calls == 1)
        instance.shutdown()
        with pytest.raises(AyrisError):
            slow.result(10.0)

    def test_shutdown_is_idempotent(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        instance.shutdown()
        instance.shutdown()

    def test_context_manager_shuts_down(self, bus: EventBus):
        with WorkerManager(bus) as instance:
            instance.register(echo_spec())
            instance.start("echo")
            process_name = f"ayris-{instance.spec('echo').name}"
        assert wait_for(
            lambda: all(
                child.name != process_name for child in multiprocessing.active_children()
            ),
            timeout=10.0,
        )

    def test_stopping_an_unstarted_worker_is_harmless(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.stop("echo")
        manager.stop("nothing-like-this")
        assert manager.worker_status("echo") is WorkerStatus.STOPPED

    def test_no_stray_threads(self, bus: EventBus):
        before = threading.active_count()
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        instance.shutdown()
        assert wait_for(lambda: threading.active_count() <= before + 1, timeout=10.0)


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


class TestRegistry:
    def test_offline_plan_starts_local_workers(self):
        settings = Settings()
        plan = plan_workers(settings)
        names = {planned.spec.name for planned in plan}
        assert {"audio", "stt", "tts"} <= names
        assert plan.autostart_names
        assert "audio" in plan.autostart_names

    def test_eco_mode_defers_unused_workers(self):
        eager = plan_workers(Settings().model_copy(update={}))
        settings = Settings()
        settings.performance.eco_mode = True
        thrifty = plan_workers(settings)
        assert len(thrifty.autostart_names) <= len(eager.autostart_names)
        assert "audio" in thrifty.autostart_names, "the microphone must still come up"

    def test_online_stt_does_not_load_the_local_model(self):
        settings = Settings()
        settings.performance.eco_mode = True
        settings.voice.stt.mode = "online"
        plan = plan_workers(settings)
        stt = next(planned for planned in plan if planned.spec.name == "stt")
        assert stt.autostart is False
        assert stt.reason

    def test_llm_is_deferred_unless_asked_for(self):
        settings = Settings()
        settings.ai.fallback_to_llm = False
        settings.ai.llm_understanding = False
        settings.ai.free_chat = False
        plan = plan_workers(settings)
        assert "llm" not in {planned.spec.name for planned in plan}

    def test_llm_appears_when_free_chat_is_on(self):
        settings = Settings()
        settings.ai.free_chat = True
        plan = plan_workers(settings)
        assert "llm" in {planned.spec.name for planned in plan}

    def test_specs_carry_the_settings_they_need(self):
        settings = Settings()
        settings.voice.stt.offline_engine = "whisper"
        plan = plan_workers(settings, log_dir=Path("logs"))
        stt = next(planned for planned in plan if planned.spec.name == "stt")
        assert stt.spec.params["engine"] == "whisper"
        assert stt.spec.log_dir == Path("logs")
        assert stt.spec.restart_scope == "stt"

    def test_plan_describe_is_readable(self):
        plan = plan_workers(Settings())
        described = plan.describe()
        assert "audio" in described


class TestApplyPlan:
    def test_apply_plan_starts_and_stops(self, manager: WorkerManager):
        from ayris.workers.registry import PlannedWorker, WorkerPlan

        first = WorkerPlan(
            workers=(
                PlannedWorker(spec=echo_spec("one"), autostart=True),
                PlannedWorker(spec=echo_spec("two"), autostart=False, reason="отложен"),
            )
        )
        manager.apply_plan(first)
        assert manager.is_ready("one")
        assert manager.worker_status("two") is WorkerStatus.REGISTERED

        second = WorkerPlan(workers=(PlannedWorker(spec=echo_spec("two"), autostart=True),))
        manager.apply_plan(second)
        assert manager.names == ("two",)
        assert manager.is_ready("two")

    def test_deferred_worker_starts_on_first_call(self, manager: WorkerManager):
        from ayris.workers.registry import PlannedWorker, WorkerPlan

        manager.apply_plan(
            WorkerPlan(workers=(PlannedWorker(spec=echo_spec(), autostart=False, reason="эко"),))
        )
        assert manager.worker_status("echo") is WorkerStatus.REGISTERED
        assert manager.call_sync("echo", "add", {"a": 4, "b": 4}) == 8
        assert manager.is_ready("echo")

    def test_changed_params_restart_the_worker(self, manager: WorkerManager):
        from ayris.workers.registry import PlannedWorker, WorkerPlan

        manager.apply_plan(
            WorkerPlan(workers=(PlannedWorker(spec=echo_spec(params={"v": 1}), autostart=True),))
        )
        first_pid = manager.status()[0].pid
        manager.apply_plan(
            WorkerPlan(workers=(PlannedWorker(spec=echo_spec(params={"v": 2}), autostart=True),))
        )
        assert manager.status()[0].pid != first_pid
        assert manager.call_sync("echo", "inherited")["params"] == {"v": 2}

    def test_unchanged_plan_keeps_the_process(self, manager: WorkerManager):
        from ayris.workers.registry import PlannedWorker, WorkerPlan

        plan = WorkerPlan(workers=(PlannedWorker(spec=echo_spec(), autostart=True),))
        manager.apply_plan(plan)
        first_pid = manager.status()[0].pid
        manager.apply_plan(plan)
        assert manager.status()[0].pid == first_pid

    def test_restart_scope_recycles_matching_workers(self, manager: WorkerManager):
        manager.register(echo_spec("a", restart_scope="stt"))
        manager.register(echo_spec("b", restart_scope="tts"))
        manager.start("a")
        manager.start("b")
        pids = {summary.name: summary.pid for summary in manager.status()}
        assert manager.restart_scope("stt") == 1
        after = {summary.name: summary.pid for summary in manager.status()}
        assert after["a"] != pids["a"]
        assert after["b"] == pids["b"]
