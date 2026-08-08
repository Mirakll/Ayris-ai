"""Tests for the worker protocol, the base worker and the supervisor.

These start real processes. That is the point: a killed worker, a wedged one, a
pipe that closes mid-call — none of it can be reproduced with a mock, and a
mocked worker would only prove that the mock behaves.

Two things keep that affordable. The worker classes live in
``tests/fixtures/echo_worker.py`` rather than here, because under ``spawn`` a
child re-imports the module holding its class and a worker defined in this file
would drag pytest's collection machinery into every child. And every manager is
torn down through :meth:`WorkerManager.shutdown`, with
:meth:`TestShutdown.test_shutdown_leaves_no_processes` asserting what the
specification actually asks for: nothing left running afterwards.

Run the suite as ``python -m pytest``. Started through the bare ``pytest``
console script, :mod:`multiprocessing` hands children the script path as their
``__main__`` and each spawned worker re-runs pytest.
"""

from __future__ import annotations

import multiprocessing
import pickle
import struct
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayris.core.config import RestartScope, Settings
from ayris.core.errors import AyrisError, SttError
from ayris.core.events import Event, EventBus, LogLine, WorkerCrashed, WorkerRestarted
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
    WorkerCancelledError,
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
from ayris.workers.registry import (
    WORKER_TYPES,
    PlannedWorker,
    WorkerKind,
    WorkerPlan,
    WorkerSpec,
    plan_workers,
    worker_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ayris.core.models import JsonObject

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ECHO = "echo_worker:EchoWorker"

#: Spawning an interpreter and importing Ayris takes a moment on a cold cache;
#: this only has to be shorter than the point where a hang stalls the suite.
START_TIMEOUT = 60.0


def echo_spec(name: str = "echo", **overrides: Any) -> WorkerSpec:
    """A spec pointing at the echo worker in ``tests/fixtures``.

    The heartbeat is deliberately slower than the tests need. A short one would
    make every test a liveness test, and a loaded machine that stalls a worker
    thread for a moment would look like a hang.
    """
    base = WorkerSpec(
        name=name,
        kind="test",
        entrypoint=ECHO,
        python_path=(str(FIXTURES),),
        start_timeout=START_TIMEOUT,
        call_timeout=20.0,
        stop_timeout=5.0,
        heartbeat_interval=0.5,
        heartbeat_misses=6,
        restart_delay=0.05,
        max_restart_delay=0.2,
    )
    return replace(base, **overrides)


@pytest.fixture
def bus() -> EventBus:
    """A bus that delivers inline on whichever thread publishes.

    The application marshals everything onto the UI thread; a test has no event
    loop to marshal onto, and ``thread_id=None`` is the mode the bus already
    provides for exactly that.
    """
    return EventBus(thread_id=None)


@pytest.fixture
def manager(bus: EventBus) -> Iterator[WorkerManager]:
    """A manager that is shut down even when the test fails."""
    instance = WorkerManager(bus)
    try:
        yield instance
    finally:
        instance.shutdown()


def collect(bus: EventBus, event_type: type[Event]) -> list[Any]:
    """Subscribe strongly and return the list events land in."""
    received: list[Any] = []
    bus.subscribe(event_type, received.append, weak=False)
    return received


def wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    """Poll ``predicate`` until it is true or the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def child_process(name: str = "echo") -> multiprocessing.process.BaseProcess | None:
    """The live worker process for ``name``, as multiprocessing sees it.

    Used to kill a worker from the outside, which is what a crash looks like: the
    supervisor gets no warning and no exit path of its own.
    """
    for child in multiprocessing.active_children():
        if child.name == f"ayris-{name}":
            return child
    return None


def worker_children() -> list[multiprocessing.process.BaseProcess]:
    """Every live Ayris worker process."""
    return [child for child in multiprocessing.active_children() if child.name.startswith("ayris-")]


def settings_with(values: JsonObject) -> Settings:
    """Settings built from a nested mapping, since every section is frozen."""
    return Settings.model_validate(values)


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
            Response(
                id=2,
                ok=False,
                error=ErrorInfo(error_type="SttError", message="нет", user_message="Не вышло."),
            ),
            WorkerEvent(kind="level", payload={"rms": 0.5}),
            Heartbeat(seq=3, at=1.5, busy=True, handled=7),
            Control(kind=ControlKind.CANCEL, request_id=4),
            Ready(name="echo", pid=1, methods=("echo",)),
        ],
    )
    def test_round_trip(self, message):
        assert decode(encode(message)) == message

    def test_rejects_foreign_bytes(self):
        with pytest.raises(WorkerProtocolError, match="not a worker frame"):
            decode(b"nonsense")

    def test_rejects_a_truncated_frame(self):
        with pytest.raises(WorkerProtocolError):
            decode(MAGIC[:2])

    def test_rejects_another_protocol_version(self):
        raw = encode(Request(id=1, method="echo"))
        forged = MAGIC + bytes((PROTOCOL_VERSION + 1,)) + raw[len(MAGIC) + 1 :]
        with pytest.raises(WorkerProtocolError, match="protocol version"):
            decode(forged)

    def test_rejects_an_unknown_payload_type(self):
        forged = MAGIC + bytes((PROTOCOL_VERSION,)) + pickle.dumps({"not": "a message"})
        with pytest.raises(WorkerProtocolError, match="unexpected worker payload"):
            decode(forged)

    def test_error_info_keeps_the_ayris_type(self):
        info = ErrorInfo.from_exception(
            SttError("модель не загрузилась", user_message="Речь не распознана.")
        )
        restored = info.to_exception()
        assert isinstance(restored, SttError)
        assert restored.user_message == "Речь не распознана."
        assert "модель не загрузилась" in str(restored)

    def test_error_info_degrades_an_unknown_type(self):
        info = ErrorInfo(error_type="SomeLibraryError", message="boom", user_message="Сбой.")
        restored = info.to_exception()
        assert isinstance(restored, AyrisError)
        assert getattr(restored, "remote_type", "") == "SomeLibraryError"

    def test_error_info_from_a_plain_exception(self):
        info = ErrorInfo.from_exception(ValueError("bad value"))
        assert info.error_type == "ValueError"
        assert info.message == "bad value"


class TestSharedAudio:
    """Descriptors travel on the pipe; samples never do."""

    def test_block_round_trip_in_one_process(self):
        pcm = struct.pack("<4h", 1, -2, 3, -4)
        with SharedAudioBlock.create(pcm, sample_rate=16000) as block:
            params = block.chunk.to_params({"language": "ru"})
            assert params["language"] == "ru"
            chunk = AudioChunk.from_params(params)
            assert chunk is not None
            with open_audio(chunk) as view:
                assert bytes(view) == pcm

    def test_chunk_reports_its_duration(self):
        with SharedAudioBlock.create(b"\x00" * (16000 * 2), sample_rate=16000) as block:
            assert block.chunk.frames == 16000
            assert block.chunk.duration_ms == pytest.approx(1000.0)

    def test_empty_buffer_is_refused(self):
        with pytest.raises(AyrisError, match="empty audio buffer"):
            SharedAudioBlock.create(b"")

    def test_from_params_without_audio(self):
        assert AudioChunk.from_params({"language": "ru"}) is None

    def test_reading_a_released_block_fails_cleanly(self):
        block = SharedAudioBlock.create(b"\x01\x02" * 100)
        chunk = block.chunk
        block.close()
        with pytest.raises(AyrisError), open_audio(chunk):
            pass

    def test_iter_frames_splits_evenly(self):
        pcm = b"\x00\x01" * 100
        chunk = AudioChunk(block="unused", nbytes=len(pcm))
        frames = list(iter_frames(memoryview(pcm), chunk, 25))
        assert [frame.nbytes for frame in frames] == [50, 50, 50, 50]


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
        assert summary.alive
        assert summary.pid
        assert "echo" in summary.methods

        manager.stop("echo")
        assert manager.worker_status("echo") is WorkerStatus.STOPPED
        assert manager.status()[0].pid is None
        assert wait_for(lambda: child_process() is None)

    def test_the_work_happens_in_another_process(self, manager: WorkerManager):
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
        assert state["poisoned"] is False, "the worker inherited parent state"
        assert state["params"] == {"model": "tiny"}
        assert state["has_qt"] is False, "the worker imported Qt"

    def test_builtin_methods(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")

        assert manager.call_sync("echo", "ping", {"n": 1})["pong"] is True
        stats = manager.call_sync("echo", "stats")
        assert stats["name"] == "echo"
        assert stats["kind"] == "test"
        assert "audio_summary" in stats["methods"]

        assert manager.call_sync("echo", "configure", {"volume": 3})["configured"] is True
        assert manager.call_sync("echo", "inherited")["params"] == {"volume": 3}

    def test_heartbeats_keep_arriving(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        assert wait_for(lambda: (manager.status()[0].last_heartbeat_age or 99) < 2.0)

    def test_a_slow_handler_does_not_look_like_a_hang(self, manager: WorkerManager):
        """The beat comes from its own thread, so a busy worker still answers."""
        manager.register(echo_spec())
        manager.start("echo")
        future = manager.call("echo", "freeze", {"seconds": 2.0})
        assert wait_for(lambda: (manager.status()[0].last_heartbeat_age or 99) < 2.0, timeout=3.0)
        future.result(15.0)
        assert manager.is_ready("echo")
        assert manager.status()[0].restarts == 0

    def test_unknown_worker(self, manager: WorkerManager):
        with pytest.raises(WorkerUnavailableError, match="not registered"):
            manager.call("nope", "echo")

    def test_a_second_registration_of_a_running_name_is_refused(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(WorkerStartError, match="already running"):
            manager.register(echo_spec())

    def test_registering_over_a_stopped_worker_replaces_the_spec(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        manager.stop("echo")
        manager.register(echo_spec(params={"replaced": True}))
        manager.start("echo")
        assert manager.call_sync("echo", "inherited")["params"] == {"replaced": True}


class TestErrors:
    def test_a_typed_error_crosses_the_pipe(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(SttError, match="recognition exploded") as caught:
            manager.call_sync("echo", "boom", {"typed": True})
        assert caught.value.user_message == "Распознавание сломалось."
        assert manager.is_ready("echo"), "a handler error must not cost the process"

    def test_a_plain_error_becomes_an_ayris_error(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(AyrisError, match="something entirely unexpected") as caught:
            manager.call_sync("echo", "boom", {"typed": False})
        assert getattr(caught.value, "remote_type", "") == "ValueError"

    def test_an_unknown_method_is_reported(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(WorkerProtocolError, match="unknown worker method"):
            manager.call_sync("echo", "no_such_method")
        assert manager.is_ready("echo")

    def test_an_unsendable_result_reports_instead_of_hanging(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        with pytest.raises(AyrisError, match="cannot be sent"):
            manager.call_sync("echo", "unpicklable")
        assert manager.is_ready("echo")

    def test_a_failure_during_start_is_reported(self, manager: WorkerManager):
        manager.register(echo_spec(params={"fail_on_start": True}))
        with pytest.raises(WorkerStartError, match="was told to fail"):
            manager.start("echo")
        assert manager.worker_status("echo") is WorkerStatus.FAILED
        assert wait_for(lambda: child_process() is None)

    def test_an_unresolvable_entrypoint_is_reported(self, manager: WorkerManager):
        manager.register(echo_spec(entrypoint="echo_worker:NoSuchWorker"))
        with pytest.raises(WorkerStartError):
            manager.start("echo")
        assert manager.worker_status("echo") is WorkerStatus.FAILED

    def test_autostart_off_refuses_to_wake_a_worker(self, manager: WorkerManager):
        manager.register(echo_spec())
        with pytest.raises(WorkerUnavailableError, match="autostart is off"):
            manager.call("echo", "ping", autostart=False)
        assert child_process() is None


class TestTimeoutAndCancel:
    def test_a_timeout_leaves_the_worker_alive(self, manager: WorkerManager):
        manager.register(echo_spec(call_timeout=0.5))
        manager.start("echo")
        with pytest.raises(WorkerTimeoutError, match="did not answer in time"):
            manager.call_sync("echo", "sleep", {"seconds": 10.0})
        assert manager.is_ready("echo")
        assert manager.status()[0].restarts == 0
        assert manager.call_sync("echo", "add", {"a": 1, "b": 1}, timeout=15.0) == 2

    def test_a_per_call_timeout_overrides_the_spec(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        started = time.monotonic()
        with pytest.raises(WorkerTimeoutError):
            manager.call_sync("echo", "sleep", {"seconds": 10.0}, timeout=0.5)
        assert time.monotonic() - started < 5.0

    def test_cancel_stops_a_cooperative_handler(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        future = manager.call("echo", "sleep", {"seconds": 30.0})
        assert wait_for(lambda: manager.status()[0].pending_calls == 1)
        manager.cancel_all()
        with pytest.raises(WorkerCancelledError):
            future.result(15.0)
        assert manager.is_ready("echo")

    def test_futures_are_independent(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        futures = [manager.call("echo", "add", {"a": index, "b": 1}) for index in range(20)]
        assert [future.result(30.0) for future in futures] == list(range(1, 21))


class TestEventsAndLogs:
    def test_worker_events_reach_the_bus(self, bus: EventBus, manager: WorkerManager):
        seen: list[tuple[str, JsonObject]] = []

        def translate(kind: str, payload: JsonObject) -> Event | None:
            seen.append((kind, payload))
            return LogLine(level="INFO", message=kind)

        lines = collect(bus, LogLine)
        manager.register(echo_spec())
        manager.set_event_translator("echo", translate)
        manager.start("echo")

        assert manager.call_sync("echo", "emit", {"count": 3, "kind": "tick"}) == 3
        assert wait_for(lambda: len(seen) == 3)
        assert [kind for kind, _ in seen] == ["tick"] * 3
        assert [payload["index"] for _, payload in seen] == [0, 1, 2]
        assert any(line.message == "tick" for line in lines)

    def test_an_untranslated_event_is_dropped(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        assert manager.call_sync("echo", "emit", {"count": 2}) == 2
        assert manager.is_ready("echo")

    def test_a_worker_log_record_is_forwarded(self, bus: EventBus, manager: WorkerManager):
        lines = collect(bus, LogLine)
        manager.register(echo_spec())
        manager.start("echo")
        manager.call_sync("echo", "warn", {"message": "тестовое предупреждение"})
        assert wait_for(lambda: any("тестовое предупреждение" in line.message for line in lines))
        forwarded = next(line for line in lines if "тестовое" in line.message)
        assert forwarded.level == "WARNING"
        assert "echo" in forwarded.logger


class TestAudioTransfer:
    def test_ten_seconds_of_audio_through_shared_memory(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        pcm = bytes(range(256)) * 1250
        assert len(pcm) == 16000 * 2 * 10, "ten seconds of 16 kHz mono int16"

        started = time.monotonic()
        summary = manager.call_sync("echo", "audio_summary", {"language": "ru"}, audio=pcm)
        elapsed = time.monotonic() - started

        assert summary["received"] is True
        assert summary["nbytes"] == len(pcm)
        assert summary["sample_rate"] == 16000
        assert summary["duration_ms"] == pytest.approx(10_000.0)
        assert summary["checksum"] == sum(pcm) % 1_000_003, "samples arrived altered"
        assert summary["head"] == [0, 1, 2, 3]
        # A pickled copy of the same buffer would cost several hundred
        # milliseconds; the descriptor is forty bytes.
        assert elapsed < 5.0, f"transfer took {elapsed:.2f} s"

    def test_a_block_is_released_after_every_call(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        for _ in range(5):
            assert manager.call_sync("echo", "audio_summary", audio=b"\x01\x02" * 8000)["received"]
        assert manager.status()[0].pending_calls == 0

    def test_a_block_is_released_after_a_timeout(self, manager: WorkerManager):
        manager.register(echo_spec(call_timeout=0.5))
        manager.start("echo")
        with pytest.raises(WorkerTimeoutError):
            manager.call_sync("echo", "sleep", {"seconds": 10.0}, audio=b"\x00" * 32000)
        assert wait_for(lambda: manager.status()[0].pending_calls == 0)


class TestCrashAndRestart:
    def test_a_killed_worker_comes_back(self, bus: EventBus, manager: WorkerManager):
        crashes = collect(bus, WorkerCrashed)
        restarts = collect(bus, WorkerRestarted)
        manager.register(echo_spec())
        manager.start("echo")
        first_pid = manager.status()[0].pid

        with pytest.raises(WorkerCrashError):
            manager.call_sync("echo", "die", {"code": 7})

        assert wait_for(lambda: manager.is_ready("echo"), timeout=30.0), "worker never came back"
        assert manager.status()[0].pid not in (None, first_pid)
        assert manager.call_sync("echo", "add", {"a": 1, "b": 2}) == 3

        assert wait_for(lambda: crashes and restarts)
        assert crashes[0].worker == "echo"
        assert crashes[0].exit_code == 7
        assert restarts[0].worker == "echo"
        assert restarts[0].attempt == 1

    def test_pending_calls_fail_when_the_process_is_killed(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.start("echo")
        slow = manager.call("echo", "sleep", {"seconds": 60.0})
        assert wait_for(lambda: manager.status()[0].pending_calls == 1)

        process = child_process()
        assert process is not None
        process.kill()

        with pytest.raises(WorkerCrashError, match="died"):
            slow.result(30.0)

    def test_a_wedged_worker_is_replaced(self, bus: EventBus, manager: WorkerManager):
        """A worker that stops beating is killed, even though it is still alive.

        ``wedge`` starves every thread in the child the way a native decoder call
        that never returns does. The operating system sees a healthy process; only
        the missing heartbeat gives it away.
        """
        crashes = collect(bus, WorkerCrashed)
        manager.register(echo_spec(heartbeat_interval=0.2, heartbeat_misses=5, call_timeout=60.0))
        manager.start("echo")
        first_pid = manager.status()[0].pid

        manager.call("echo", "wedge", {"seconds": 60.0})

        assert wait_for(
            lambda: manager.status()[0].pid not in (None, first_pid),
            timeout=30.0,
        ), "a worker that stopped beating was not replaced"
        assert wait_for(lambda: any("отклик" in event.error for event in crashes))
        assert manager.call_sync("echo", "add", {"a": 2, "b": 2}, timeout=20.0) == 4

    def test_the_restart_limit_gives_up(self, manager: WorkerManager):
        manager.register(echo_spec(max_restarts=1))
        manager.start("echo")

        for _ in range(2):
            process = child_process()
            assert process is not None
            process.kill()
            time.sleep(0.5)
            wait_for(lambda: manager.worker_status("echo") is not WorkerStatus.RESTARTING)

        assert wait_for(lambda: manager.worker_status("echo") is WorkerStatus.FAILED, timeout=30.0)
        assert child_process() is None

        future = manager.call("echo", "ping")
        with pytest.raises(WorkerUnavailableError, match="failed to start"):
            future.result(10.0)

    def test_a_manual_restart_keeps_the_registration(self, bus: EventBus, manager: WorkerManager):
        restarts = collect(bus, WorkerRestarted)
        manager.register(echo_spec())
        manager.start("echo")
        first_pid = manager.status()[0].pid

        manager.restart("echo", reason="проверка")

        assert manager.is_ready("echo")
        assert manager.status()[0].pid != first_pid
        assert manager.names == ("echo",)
        assert any(event.reason == "проверка" for event in restarts)


class TestShutdown:
    def test_shutdown_leaves_no_processes(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec("one"))
        instance.register(echo_spec("two"))
        instance.start("one")
        instance.start("two")
        children = worker_children()
        assert len(children) == 2

        instance.shutdown()

        assert wait_for(lambda: not any(child.is_alive() for child in children), timeout=30.0)
        assert instance.status() == ()
        assert worker_children() == []

    def test_shutdown_fails_pending_calls(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        slow = instance.call("echo", "sleep", {"seconds": 60.0})
        assert wait_for(lambda: instance.status()[0].pending_calls == 1)

        instance.shutdown()

        # Either the worker notices the stop and reports the call cancelled, or
        # the manager fails it while tearing the channel down. Which one wins is
        # a race; what must never happen is the caller waiting on a future that
        # no process will ever answer.
        with pytest.raises(WorkerError):
            slow.result(15.0)

    def test_shutdown_is_idempotent(self, bus: EventBus):
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        instance.shutdown()
        instance.shutdown()
        with pytest.raises(WorkerStartError, match="shutting down"):
            instance.register(echo_spec())

    def test_the_context_manager_shuts_down(self, bus: EventBus):
        with WorkerManager(bus) as instance:
            instance.register(echo_spec())
            instance.start("echo")
            assert child_process() is not None
        assert wait_for(lambda: child_process() is None, timeout=30.0)

    def test_stopping_a_worker_that_never_ran_is_harmless(self, manager: WorkerManager):
        manager.register(echo_spec())
        manager.stop("echo")
        manager.stop("nothing-like-this")
        assert manager.worker_status("echo") is WorkerStatus.STOPPED

    def test_no_threads_are_left_behind(self, bus: EventBus):
        before = threading.active_count()
        instance = WorkerManager(bus)
        instance.register(echo_spec())
        instance.start("echo")
        instance.shutdown()
        assert wait_for(lambda: threading.active_count() <= before, timeout=30.0)


# ----------------------------------------------------------------------
# registry
# ----------------------------------------------------------------------


class TestRegistry:
    """Planning rules only. The worker modules themselves arrive in later tasks,
    so every plan here is built with ``include_unavailable``."""

    def test_every_kind_has_a_type_and_a_label(self):
        assert {entry.kind for entry in WORKER_TYPES} == set(WorkerKind)
        for kind in WorkerKind:
            assert worker_type(kind) is not None
            assert kind.label

    def test_a_missing_module_is_skipped_rather_than_fatal(self):
        """Only the audio worker exists so far, and launch must survive the rest."""
        assert [planned.spec.name for planned in plan_workers(Settings())] == ["audio"]

    def test_the_default_plan_starts_the_local_workers(self):
        plan = plan_workers(Settings(), include_unavailable=True)
        assert [planned.spec.name for planned in plan] == ["audio", "stt", "tts", "llm"]
        assert plan.eco_mode is False
        assert plan.deferred == (), "with eco mode off everything needed starts up front"
        assert plan.by_kind(WorkerKind.AUDIO) is not None
        assert "audio" in plan.describe()

    def test_eco_mode_defers_what_is_not_on_the_primary_path(self):
        eager = plan_workers(Settings(), include_unavailable=True)
        thrifty = plan_workers(
            settings_with({"performance": {"eco_mode": True}}), include_unavailable=True
        )
        assert thrifty.eco_mode is True
        assert len(thrifty.autostart) < len(eager.autostart)
        started = {planned.spec.name for planned in thrifty.autostart}
        assert "audio" in started, "the microphone still has to come up"
        assert "llm" not in started, "plain fallback waits for its first use"
        assert all(planned.reason for planned in thrifty.deferred)
        assert "по требованию" in thrifty.describe()

    def test_online_recognition_does_not_preload_the_local_model(self):
        plan = plan_workers(
            settings_with(
                {"performance": {"eco_mode": True}, "voice": {"stt": {"mode": "online"}}}
            ),
            include_unavailable=True,
        )
        stt = plan.by_kind(WorkerKind.STT)
        assert stt is not None
        assert stt.needed is True, "the worker still holds the cloud client"
        assert stt.autostart is False
        assert "экономии" in stt.reason

    def test_a_cloud_voice_does_not_preload_a_synthesiser(self):
        plan = plan_workers(
            settings_with(
                {"performance": {"eco_mode": True}, "voice": {"tts": {"engine": "yandex"}}}
            ),
            include_unavailable=True,
        )
        tts = plan.by_kind(WorkerKind.TTS)
        assert tts is not None
        assert tts.autostart is False

    def test_no_llm_worker_when_nothing_asks_a_model(self):
        plan = plan_workers(
            settings_with(
                {
                    "ai": {
                        "fallback_to_llm": False,
                        "llm_understanding": False,
                        "free_chat": False,
                    }
                }
            ),
            include_unavailable=True,
        )
        assert plan.by_kind(WorkerKind.LLM) is None

    def test_free_chat_keeps_the_model_warm_even_in_eco_mode(self):
        plan = plan_workers(
            settings_with({"performance": {"eco_mode": True}, "ai": {"free_chat": True}}),
            include_unavailable=True,
        )
        llm = plan.by_kind(WorkerKind.LLM)
        assert llm is not None
        assert llm.autostart is True

    def test_specs_carry_the_settings_their_worker_needs(self):
        plan = plan_workers(
            settings_with({"voice": {"stt": {"offline_engine": "whisper"}}}),
            log_dir=Path("logs"),
            include_unavailable=True,
        )
        stt = plan.by_kind(WorkerKind.STT)
        assert stt is not None
        assert stt.spec.params["offline_engine"] == "whisper"
        assert stt.spec.params["log_level"], "every worker gets the common fields"
        assert stt.spec.log_dir == Path("logs")
        assert stt.spec.restart_scope is RestartScope.STT

        audio = plan.by_kind(WorkerKind.AUDIO)
        assert audio is not None
        assert audio.spec.heartbeat_timeout == pytest.approx(3.0)
        assert audio.spec.restart_scope is RestartScope.AUDIO

    def test_no_credentials_are_copied_into_a_spec(self):
        """Only the name of a credential travels; the key stays in Windows."""
        plan = plan_workers(Settings(), include_unavailable=True)
        for planned in plan:
            for key, value in planned.spec.params.items():
                assert "key" not in key or key.endswith("_ref"), key
                assert not isinstance(value, bytes)


class TestApplyPlan:
    def test_apply_plan_starts_registers_and_retires(self, manager: WorkerManager):
        manager.apply_plan(
            WorkerPlan(
                workers=(
                    PlannedWorker(spec=echo_spec("one"), needed=True, autostart=True),
                    PlannedWorker(
                        spec=echo_spec("two"), needed=True, autostart=False, reason="отложен"
                    ),
                )
            )
        )
        assert manager.is_ready("one")
        assert manager.worker_status("two") is WorkerStatus.REGISTERED
        assert child_process("two") is None

        manager.apply_plan(
            WorkerPlan(workers=(PlannedWorker(spec=echo_spec("two"), needed=True, autostart=True),))
        )
        assert manager.names == ("two",)
        assert manager.is_ready("two")
        assert wait_for(lambda: child_process("one") is None)

    def test_a_deferred_worker_starts_on_first_call(self, manager: WorkerManager):
        manager.apply_plan(
            WorkerPlan(
                workers=(
                    PlannedWorker(spec=echo_spec(), needed=True, autostart=False, reason="эко"),
                )
            )
        )
        assert manager.worker_status("echo") is WorkerStatus.REGISTERED
        assert manager.call_sync("echo", "add", {"a": 4, "b": 4}) == 8
        assert manager.is_ready("echo")

    def test_changed_parameters_restart_the_worker(self, manager: WorkerManager):
        manager.apply_plan(
            WorkerPlan(
                workers=(
                    PlannedWorker(spec=echo_spec(params={"v": 1}), needed=True, autostart=True),
                )
            )
        )
        first_pid = manager.status()[0].pid
        manager.apply_plan(
            WorkerPlan(
                workers=(
                    PlannedWorker(spec=echo_spec(params={"v": 2}), needed=True, autostart=True),
                )
            )
        )
        assert manager.status()[0].pid != first_pid
        assert manager.call_sync("echo", "inherited")["params"] == {"v": 2}

    def test_an_unchanged_plan_keeps_the_process(self, manager: WorkerManager):
        plan = WorkerPlan(workers=(PlannedWorker(spec=echo_spec(), needed=True, autostart=True),))
        manager.apply_plan(plan)
        first_pid = manager.status()[0].pid
        manager.apply_plan(plan)
        assert manager.status()[0].pid == first_pid

    def test_a_scope_restarts_only_its_own_workers(self, manager: WorkerManager):
        manager.register(echo_spec("a", restart_scope=RestartScope.STT))
        manager.register(echo_spec("b", restart_scope=RestartScope.TTS))
        manager.start("a")
        manager.start("b")
        before = {summary.name: summary.pid for summary in manager.status()}

        assert manager.restart_scope(RestartScope.STT) == 1

        after = {summary.name: summary.pid for summary in manager.status()}
        assert after["a"] != before["a"]
        assert after["b"] == before["b"]
