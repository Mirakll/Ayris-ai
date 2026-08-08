"""Supervisor for worker processes: start, call, watch, restart, stop.

:class:`WorkerManager` is the only thing in the main process that owns a child
process, and the only thing that touches a :class:`~ayris.workers.protocol.Channel`
on the parent side. Everything else — the pipeline, the actions, the GUI — talks
to workers through :meth:`WorkerManager.call`, which hands back a
:class:`~concurrent.futures.Future` and never blocks the caller.

**Threads.** One reader thread per worker owns that worker's channel, resolves
futures and republishes worker events onto the bus. One monitor thread, shared by
all workers, watches heartbeats, reaps dead processes, enforces call deadlines and
performs restarts. Neither thread ever calls into Qt: they publish onto the event
bus, which marshals delivery to the GUI thread.

**Restarts.** A worker that dies, wedges or exits unasked is restarted with
exponential backoff — one second, then two, then four, capped — and
:class:`~ayris.core.events.WorkerCrashed` and
:class:`~ayris.core.events.WorkerRestarted` are published so the UI can say so out
loud. Too many restarts inside the window and the worker is marked
:attr:`WorkerStatus.FAILED` and left alone: a model file that no longer parses
will not start on the tenth attempt either, and a restart loop is worse than a
missing subsystem.

**Audio.** A ten-second utterance is roughly 320 KB at 16 kHz mono, and pushing
that through a pipe costs two copies plus a pickle round-trip. Passing ``audio=``
to :meth:`WorkerManager.call` puts the samples in shared memory and sends only the
descriptor; the block is released when the call settles, whichever way it settles.

**No orphans.** Three independent mechanisms, because on Windows each one has a
hole: the child's pipe end closes when the parent dies, the child polls the
parent's liveness, and every child is assigned to a job object created with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` so that even a hard-killed parent takes its
workers with it.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import multiprocessing
import os
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from ayris.core.events import Event, EventBus, LogLine, WorkerCrashed, WorkerRestarted
from ayris.workers.base import WorkerBootstrap, windows_dll, worker_entrypoint
from ayris.workers.protocol import (
    Channel,
    Control,
    ControlKind,
    ErrorInfo,
    Heartbeat,
    Ready,
    Request,
    Response,
    SharedAudioBlock,
    Stopped,
    WorkerCrashError,
    WorkerError,
    WorkerEvent,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from ayris.workers.registry import WorkerPlan, WorkerSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from multiprocessing.context import SpawnContext
    from multiprocessing.process import BaseProcess
    from pathlib import Path

    from ayris.core.app import AyrisApp
    from ayris.core.config import RestartScope, Settings
    from ayris.core.models import JsonObject

    #: What a worker event becomes on the bus. ``None`` drops the event.
    EventTranslator = Callable[[str, JsonObject], Event | None]

__all__ = [
    "WorkerManager",
    "WorkerStatus",
    "WorkerSummary",
    "install_workers",
]

_log = logging.getLogger("ayris.workers.manager")

#: How often the monitor thread wakes. Short enough that a crashed worker is back
#: inside the two seconds section 14 asks for, long enough to be invisible.
MONITOR_TICK: Final = 0.1

#: Grace period after a polite stop before the process is terminated.
TERMINATE_GRACE: Final = 1.0

#: Log records forwarded from workers are re-emitted under this logger prefix.
_WORKER_LOGGER: Final = "ayris.workers"


class WorkerStatus(StrEnum):
    """Where a worker is in its life.

    ``REGISTERED`` is the resting state of a worker the plan deferred: known, not
    running, startable on demand. ``FAILED`` is terminal until something changes
    the settings or calls :meth:`WorkerManager.restart` explicitly.
    """

    REGISTERED = "registered"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    STOPPED = "stopped"
    RESTARTING = "restarting"
    FAILED = "failed"

    @property
    def label(self) -> str:
        """Russian wording for DevTools."""
        return _STATUS_LABELS[self]

    @property
    def is_live(self) -> bool:
        """Whether a process should exist right now."""
        return self in {WorkerStatus.STARTING, WorkerStatus.READY, WorkerStatus.STOPPING}


_STATUS_LABELS: Final[Mapping[WorkerStatus, str]] = {
    WorkerStatus.REGISTERED: "не запущен",
    WorkerStatus.STARTING: "запускается",
    WorkerStatus.READY: "работает",
    WorkerStatus.STOPPING: "останавливается",
    WorkerStatus.STOPPED: "остановлен",
    WorkerStatus.RESTARTING: "перезапускается",
    WorkerStatus.FAILED: "не удалось запустить",
}


@dataclass(frozen=True, slots=True)
class WorkerSummary:
    """Snapshot of one worker, for DevTools and for tests.

    Deliberately inert: a copy of the numbers at one instant, holding no reference
    to the process or the channel.
    """

    name: str
    kind: str
    status: WorkerStatus
    pid: int | None = None
    restarts: int = 0
    pending_calls: int = 0
    handled: int = 0
    last_heartbeat_age: float | None = None
    methods: tuple[str, ...] = ()
    error: str = ""

    @property
    def alive(self) -> bool:
        """Whether the worker is up and answering."""
        return self.status is WorkerStatus.READY


@dataclass(slots=True)
class _PendingCall:
    """One in-flight request and everything that has to be cleaned up after it."""

    id: int
    worker: str
    method: str
    future: Future[Any]
    deadline: float
    audio: SharedAudioBlock | None = None
    generation: int = 0

    def release(self) -> None:
        """Free the shared audio block, if this call carried one."""
        if self.audio is not None:
            self.audio.close()
            self.audio = None


@dataclass(slots=True)
class _Handle:
    """The manager's mutable bookkeeping for one worker.

    Guarded by :attr:`lock`, which is held only for field access — never across a
    ``send``, a ``join`` or a bus publish, so a slow worker cannot stall the
    monitor or another caller.
    """

    spec: WorkerSpec
    status: WorkerStatus = WorkerStatus.REGISTERED
    lock: threading.RLock = field(default_factory=threading.RLock)
    process: BaseProcess | None = None
    channel: Channel | None = None
    reader: threading.Thread | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    start_error: ErrorInfo | None = None
    pending: dict[int, _PendingCall] = field(default_factory=dict)
    methods: tuple[str, ...] = ()
    last_beat: float = 0.0
    handled: int = 0
    generation: int = 0
    restarts: int = 0
    restart_history: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    restart_at: float | None = None
    restart_reason: str = ""
    stopping: bool = False
    error: str = ""

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None


# ----------------------------------------------------------------------
# Windows job object
# ----------------------------------------------------------------------

#: ``JobObjectExtendedLimitInformation``.
_JOB_EXTENDED_LIMIT: Final = 9
#: ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` — the whole point of the job.
_JOB_KILL_ON_CLOSE: Final = 0x00002000
#: ``PROCESS_SET_QUOTA | PROCESS_TERMINATE``, what assignment needs.
_PROCESS_ASSIGN_ACCESS: Final = 0x0100 | 0x0001


class _IoCounters(ctypes.Structure):
    """``IO_COUNTERS``. Declared only because the extended struct embeds it."""

    _fields_ = (
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    )


class _BasicLimitInformation(ctypes.Structure):
    """``JOBOBJECT_BASIC_LIMIT_INFORMATION``."""

    _fields_ = (
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    )


class _ExtendedLimitInformation(ctypes.Structure):
    """``JOBOBJECT_EXTENDED_LIMIT_INFORMATION``."""

    _fields_ = (
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    )


class _JobObject:
    """A Windows job that kills its members when the last handle closes.

    This is the backstop for the one case the other two mechanisms miss: Ayris
    killed from Task Manager, which runs no shutdown code and closes no pipes on
    purpose. Windows closes the job handle as part of tearing the process down,
    and the workers go with it.

    Off Windows :meth:`create` returns ``None`` and the caller carries on; POSIX
    has no equivalent that is worth the complexity here, and the parent-liveness
    watchdog in the child covers it.
    """

    __slots__ = ("_handle", "_kernel32")

    def __init__(self, kernel32: Any, handle: int) -> None:
        self._kernel32 = kernel32
        self._handle = handle

    @classmethod
    def create(cls) -> _JobObject | None:
        """Create the job, or ``None`` if this platform or call cannot."""
        kernel32 = windows_dll("kernel32")
        if kernel32 is None:
            return None
        try:
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return None
            info = _ExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = _JOB_KILL_ON_CLOSE
            kernel32.SetInformationJobObject.argtypes = (
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_void_p,
                ctypes.c_uint32,
            )
            ok = kernel32.SetInformationJobObject(
                ctypes.c_void_p(handle),
                _JOB_EXTENDED_LIMIT,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not ok:
                kernel32.CloseHandle(ctypes.c_void_p(handle))
                return None
        except (AttributeError, OSError):
            _log.debug("объект задания Windows недоступен")
            return None
        return cls(kernel32, int(handle))

    def assign(self, pid: int) -> bool:
        """Put the process into the job. Best effort; failure is only logged."""
        try:
            self._kernel32.OpenProcess.restype = ctypes.c_void_p
            self._kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32)
            process = self._kernel32.OpenProcess(_PROCESS_ASSIGN_ACCESS, False, pid)
            if not process:
                return False
            try:
                self._kernel32.AssignProcessToJobObject.argtypes = (
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                )
                assigned = self._kernel32.AssignProcessToJobObject(
                    ctypes.c_void_p(self._handle),
                    ctypes.c_void_p(process),
                )
            finally:
                self._kernel32.CloseHandle(ctypes.c_void_p(process))
        except (AttributeError, OSError):
            return False
        return bool(assigned)

    def close(self) -> None:
        """Close the job handle, which terminates everything still in it."""
        if not self._handle:
            return
        with contextlib.suppress(AttributeError, OSError):
            self._kernel32.CloseHandle(ctypes.c_void_p(self._handle))
        self._handle = 0


# ----------------------------------------------------------------------
# the supervisor
# ----------------------------------------------------------------------


class WorkerManager:
    """Starts, supervises and stops the worker processes.

    Args:
        bus: Where crashes, restarts and worker events are published.
        log_dir: Passed to each worker so its own log file lands beside the main
            one. ``None`` keeps workers logging only through the channel.
        context: Multiprocessing context. Defaults to ``spawn`` on every platform
            — Windows has nothing else, and matching it elsewhere means a worker
            that inherits state by accident fails in CI rather than in the wild.

    Thread-safe: :meth:`call`, :meth:`start` and :meth:`stop` may be used from
    the GUI thread, the pipeline thread or a hotkey callback.
    """

    __slots__ = (
        "_bus",
        "_handles",
        "_ids",
        "_job",
        "_lock",
        "_log_dir",
        "_monitor",
        "_mp",
        "_shutdown",
        "_stop",
        "_translators",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        log_dir: Path | None = None,
        context: SpawnContext | None = None,
    ) -> None:
        self._bus = bus
        self._log_dir = log_dir
        self._mp: SpawnContext = context if context is not None else multiprocessing.get_context(
            "spawn"
        )
        self._handles: dict[str, _Handle] = {}
        self._translators: dict[str, EventTranslator] = {}
        self._lock = threading.RLock()
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._shutdown = False
        self._job = _JobObject.create()

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------

    def register(self, spec: WorkerSpec) -> None:
        """Make a worker known without starting it.

        Registering is what makes a worker callable: :meth:`call` will start a
        registered worker on demand, which is how eco mode defers the expensive
        ones.

        Raises:
            WorkerStartError: A different worker already uses this name.
        """
        with self._lock:
            if self._shutdown:
                raise WorkerStartError("worker manager is shutting down")
            existing = self._handles.get(spec.name)
            if existing is None:
                self._handles[spec.name] = _Handle(spec=spec)
            elif existing.status.is_live:
                raise WorkerStartError(f"worker {spec.name!r} is already running")
            else:
                existing.spec = spec
            self._ensure_monitor()

    def unregister(self, name: str) -> None:
        """Stop the worker if needed and forget it."""
        handle = self._handles.get(name)
        if handle is None:
            return
        self.stop(name)
        with self._lock:
            self._handles.pop(name, None)
            self._translators.pop(name, None)

    def set_event_translator(self, worker: str, translator: EventTranslator) -> None:
        """Decide what a worker's events become on the bus.

        The manager cannot know that the audio worker's ``level`` event is an
        :class:`~ayris.core.events.AudioLevelChanged`, and hard-coding that here
        would make this module depend on every subsystem. Instead each subsystem
        registers a translator when it registers its worker; returning ``None``
        drops the event.
        """
        self._translators[worker] = translator

    @property
    def names(self) -> tuple[str, ...]:
        """Registered worker names, in registration order."""
        with self._lock:
            return tuple(self._handles)

    def spec(self, name: str) -> WorkerSpec:
        """The spec a worker was registered with."""
        return self._require(name).spec

    def _require(self, name: str) -> _Handle:
        with self._lock:
            handle = self._handles.get(name)
        if handle is None:
            raise WorkerUnavailableError(f"worker {name!r} is not registered")
        return handle

    # ------------------------------------------------------------------
    # starting and stopping
    # ------------------------------------------------------------------

    def start(self, name: str, *, timeout: float | None = None) -> None:
        """Start a registered worker and wait until it reports ready.

        Blocking by design: a caller that needs the worker cannot proceed without
        it, and a model takes as long as it takes. Start-up runs off the GUI
        thread — during launch on the lifecycle thread, later on whichever thread
        called.

        Raises:
            WorkerStartError: The process died during start-up, its entrypoint
                does not resolve, or it did not report ready in time.
        """
        handle = self._require(name)
        with handle.lock:
            if handle.status is WorkerStatus.READY:
                return
            if handle.status.is_live:
                raise WorkerStartError(f"worker {name!r} is already {handle.status.value}")
        self._spawn(handle, timeout=timeout)

    def ensure_started(self, name: str, *, timeout: float | None = None) -> None:
        """Start the worker unless it is already up. Never raises for a race."""
        handle = self._require(name)
        with handle.lock:
            if handle.status is WorkerStatus.READY:
                return
            if handle.status is WorkerStatus.FAILED:
                raise WorkerUnavailableError(
                    f"worker {name!r} failed to start: {handle.error or 'unknown reason'}"
                )
            if handle.status.is_live:
                pass  # A concurrent start is in flight; wait for it below.
            else:
                self._spawn(handle, timeout=timeout)
                return
        deadline = time.monotonic() + (timeout or handle.spec.start_timeout)
        while time.monotonic() < deadline:
            if handle.status is WorkerStatus.READY:
                return
            time.sleep(0.02)
        raise WorkerStartError(f"worker {name!r} did not become ready")

    def _spawn(self, handle: _Handle, *, timeout: float | None = None) -> None:
        """Create the process, hook up the channel and wait for :class:`Ready`."""
        spec = handle.spec
        parent_end, child_end = self._mp.Pipe(duplex=True)
        bootstrap = WorkerBootstrap(
            name=spec.name,
            entrypoint=spec.entrypoint,
            kind=str(spec.kind),
            params=dict(spec.params),
            parent_pid=os.getpid(),
            heartbeat_interval=spec.heartbeat_interval,
            priority=spec.priority,
            log_level=spec.log_level,
            log_dir=spec.log_dir if spec.log_dir is not None else self._log_dir,
            python_path=spec.python_path,
            protocol_version=spec.protocol_version,
        )
        # daemon=True is a fourth layer of orphan protection: multiprocessing
        # terminates daemonic children when the parent interpreter exits normally.
        process = self._mp.Process(
            target=worker_entrypoint,
            args=(bootstrap, child_end),
            name=f"ayris-{spec.name}",
            daemon=True,
        )

        with handle.lock:
            handle.generation += 1
            generation = handle.generation
            handle.status = WorkerStatus.STARTING
            handle.stopping = False
            handle.ready.clear()
            handle.start_error = None
            handle.error = ""
            handle.methods = ()
            handle.last_beat = time.monotonic()

        try:
            process.start()
        except Exception as exc:
            parent_end.close()
            child_end.close()
            with handle.lock:
                handle.status = WorkerStatus.FAILED
                handle.error = str(exc)
            raise WorkerStartError(f"cannot spawn worker {spec.name!r}: {exc}") from exc
        finally:
            # The parent's copy of the child's end must go, or the child's death
            # would never surface as EOF on our side.
            child_end.close()

        if self._job is not None and process.pid is not None:
            self._job.assign(process.pid)

        channel = Channel(parent_end)
        reader = threading.Thread(
            target=self._read_loop,
            args=(handle, channel, generation),
            name=f"ayris-{spec.name}-pipe",
            daemon=True,
        )
        with handle.lock:
            handle.process = process
            handle.channel = channel
            handle.reader = reader
        reader.start()

        self._await_ready(handle, process, timeout or spec.start_timeout)

    def _await_ready(self, handle: _Handle, process: BaseProcess, timeout: float) -> None:
        """Block until the worker reports ready, dies, or runs out of time."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if handle.ready.wait(0.05):
                break
            if not process.is_alive():
                break

        failure: ErrorInfo | None = None
        with handle.lock:
            if handle.status is WorkerStatus.READY:
                _log.info("воркер %s запущен (pid %s)", handle.spec.name, handle.pid)
                return
            failure = handle.start_error
            handle.status = WorkerStatus.FAILED
            handle.error = failure.message if failure else "worker did not report ready"

        self._terminate(handle)
        if failure is not None:
            if failure.traceback:
                _log.debug("трассировка запуска %s:\n%s", handle.spec.name, failure.traceback)
            raise WorkerStartError(
                f"worker {handle.spec.name!r} failed to start: {failure.message}"
            )
        raise WorkerStartError(
            f"worker {handle.spec.name!r} did not report ready within {timeout:g} s"
        )

    def stop(self, name: str, *, timeout: float | None = None) -> None:
        """Ask a worker to finish, then make sure it did.

        Politeness first: a ``stop`` control lets the worker close its device and
        flush its model. Windows cannot deliver a signal, so a worker that ignores
        the request is terminated after :attr:`WorkerSpec.stop_timeout`, and killed
        after another second.
        """
        handle = self._handles.get(name)
        if handle is None:
            return
        with handle.lock:
            if handle.process is None or not handle.status.is_live:
                handle.status = (
                    WorkerStatus.FAILED
                    if handle.status is WorkerStatus.FAILED
                    else WorkerStatus.STOPPED
                )
                handle.restart_at = None
                return
            handle.stopping = True
            handle.status = WorkerStatus.STOPPING
            handle.restart_at = None
            process = handle.process

        self._send(handle, Control(kind=ControlKind.STOP))
        process.join(timeout if timeout is not None else handle.spec.stop_timeout)
        if process.is_alive():
            _log.warning("воркер %s не остановился сам, завершаем принудительно", name)
            self._terminate(handle)
        else:
            self._close_channel(handle)

        self._fail_pending(handle, lambda: WorkerUnavailableError(f"worker {name!r} stopped"))
        with handle.lock:
            handle.status = WorkerStatus.STOPPED
            handle.process = None
            handle.stopping = False
            handle.methods = ()
        _log.info("воркер %s остановлен", name)

    def restart(self, name: str, *, reason: str = "") -> None:
        """Stop and start a worker, keeping its registration and its spec."""
        handle = self._require(name)
        self.stop(name)
        with handle.lock:
            handle.status = WorkerStatus.REGISTERED
            handle.restarts += 1
            attempt = handle.restarts
        self._spawn(handle)
        self._bus.publish(
            WorkerRestarted(worker=name, attempt=attempt, reason=reason or "по запросу")
        )

    def _terminate(self, handle: _Handle) -> None:
        """Terminate, then kill, then close everything. Never raises."""
        with handle.lock:
            process = handle.process
        if process is not None and process.is_alive():
            with contextlib.suppress(OSError, ValueError, AttributeError):
                process.terminate()
            process.join(TERMINATE_GRACE)
        if process is not None and process.is_alive():
            killer = getattr(process, "kill", None)
            if killer is not None:
                with contextlib.suppress(OSError, ValueError, AttributeError):
                    killer()
                process.join(TERMINATE_GRACE)
        self._close_channel(handle)

    def _close_channel(self, handle: _Handle) -> None:
        with handle.lock:
            channel = handle.channel
            reader = handle.reader
            handle.channel = None
            handle.reader = None
        if channel is not None:
            channel.close()
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1.0)

    # ------------------------------------------------------------------
    # calling
    # ------------------------------------------------------------------

    def call(
        self,
        worker: str,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
        audio: bytes | bytearray | memoryview | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_format: str = "int16",
        autostart: bool = True,
    ) -> Future[Any]:
        """Invoke ``method`` on ``worker`` and return a future for its result.

        Never blocks on the worker itself; the only wait is starting a deferred
        worker when ``autostart`` is on, which is the price of eco mode.

        Args:
            worker: Registered worker name.
            method: Name the worker exposed with :func:`~ayris.workers.base.method`.
            params: Picklable arguments. Keep them small — bulk data belongs in
                ``audio`` or on disk.
            timeout: Deadline in seconds. Defaults to the spec's ``call_timeout``.
            audio: PCM samples to pass through shared memory instead of the pipe.
            sample_rate: Sample rate of ``audio``.
            channels: Channel count of ``audio``.
            sample_format: Sample format of ``audio``, e.g. ``"int16"``.
            autostart: Start a registered-but-cold worker. Turn off for a probe
                that must not wake anything.

        Returns:
            A future resolving to the handler's return value, or failing with the
            exception the handler raised — reconstructed as the same
            :class:`~ayris.core.errors.AyrisError` subclass where possible.

        Raises:
            WorkerUnavailableError: No such worker, or it is not running and
                ``autostart`` is off.
        """
        handle = self._require(worker)
        future: Future[Any] = Future()

        if handle.status is not WorkerStatus.READY:
            if not autostart:
                raise WorkerUnavailableError(
                    f"worker {worker!r} is {handle.status.label} and autostart is off"
                )
            try:
                self.ensure_started(worker)
            except WorkerError as exc:
                future.set_exception(exc)
                return future

        block: SharedAudioBlock | None = None
        call_params = dict(params) if params else {}
        if audio is not None:
            try:
                block = SharedAudioBlock.create(
                    audio,
                    sample_rate=sample_rate,
                    channels=channels,
                    sample_format=sample_format,
                )
            except WorkerError as exc:
                future.set_exception(exc)
                return future
            call_params = block.chunk.to_params(call_params)

        request_id = next(self._ids)
        limit = timeout if timeout is not None else handle.spec.call_timeout
        pending = _PendingCall(
            id=request_id,
            worker=worker,
            method=method,
            future=future,
            deadline=time.monotonic() + limit,
            audio=block,
            generation=handle.generation,
        )
        with handle.lock:
            handle.pending[request_id] = pending
        future.set_running_or_notify_cancel()

        if not self._send(handle, Request(id=request_id, method=method, params=call_params)):
            self._settle(handle, request_id, WorkerCrashError(f"worker {worker!r} is not reachable"))
        return future

    def call_sync(
        self,
        worker: str,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Blocking :meth:`call`, for scripts, tests and the DevTools console.

        Not for the GUI thread: the whole point of workers is that the interface
        keeps painting while they work.
        """
        handle = self._require(worker)
        limit = timeout if timeout is not None else handle.spec.call_timeout
        future = self.call(worker, method, params, timeout=limit, **kwargs)
        # The monitor enforces the real deadline; the margin only keeps this call
        # from hanging forever if the monitor thread itself is wedged.
        return future.result(limit + MONITOR_TICK * 20)

    def cancel(self, worker: str, request_id: int | None = None) -> None:
        """Ask a worker to abandon a request it is running.

        Cooperative: the handler notices at its next
        :meth:`~ayris.workers.base.WorkerContext.check_cancelled`. A handler that
        never checks runs to completion and its answer is discarded.
        """
        handle = self._handles.get(worker)
        if handle is None:
            return
        self._send(handle, Control(kind=ControlKind.CANCEL, request_id=request_id))

    def cancel_all(self) -> None:
        """Cancel everything in flight everywhere. The stop word ends up here."""
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            if handle.status is WorkerStatus.READY:
                self._send(handle, Control(kind=ControlKind.CANCEL))

    def _send(self, handle: _Handle, message: Request | Control) -> bool:
        """Write to a worker, reporting a dead channel rather than raising."""
        with handle.lock:
            channel = handle.channel
        if channel is None:
            return False
        try:
            channel.send(message)
        except (OSError, EOFError):
            return False
        except Exception as exc:
            _log.exception("не удалось отправить сообщение воркеру %s", handle.spec.name)
            if isinstance(message, Request):
                self._settle(
                    handle,
                    message.id,
                    WorkerError(f"cannot send {message.method} to {handle.spec.name}: {exc}"),
                )
            return False
        return True

    def _settle(self, handle: _Handle, request_id: int, error: BaseException) -> None:
        """Fail one pending call and release whatever it was holding."""
        with handle.lock:
            pending = handle.pending.pop(request_id, None)
        if pending is None:
            return
        pending.release()
        if not pending.future.done():
            pending.future.set_exception(error)

    def _fail_pending(self, handle: _Handle, error: Callable[[], BaseException]) -> int:
        """Fail every pending call on a worker. Returns how many there were."""
        with handle.lock:
            pending = tuple(handle.pending.values())
            handle.pending.clear()
        for call in pending:
            call.release()
            if not call.future.done():
                call.future.set_exception(error())
        return len(pending)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def _read_loop(self, handle: _Handle, channel: Channel, generation: int) -> None:
        """Own one worker's channel until it closes."""
        while True:
            try:
                message = channel.recv()
            except (EOFError, OSError):
                break
            except Exception:
                _log.exception("нечитаемое сообщение от воркера %s", handle.spec.name)
                continue
            try:
                self._on_message(handle, message, generation)
            except Exception:
                _log.exception("ошибка обработки сообщения воркера %s", handle.spec.name)
        # Unblock a start that is still waiting: a channel that closed before the
        # ready message means the worker died on its way up.
        handle.ready.set()

    def _on_message(
        self,
        handle: _Handle,
        message: Request | Response | WorkerEvent | Heartbeat | Ready | Stopped | Control,
        generation: int,
    ) -> None:
        if isinstance(message, Heartbeat):
            with handle.lock:
                handle.last_beat = time.monotonic()
                handle.handled = message.handled
            return
        if isinstance(message, Response):
            self._on_response(handle, message)
            return
        if isinstance(message, Ready):
            with handle.lock:
                if handle.generation != generation:
                    return
                handle.methods = message.methods
                handle.last_beat = time.monotonic()
                handle.status = WorkerStatus.READY
            handle.ready.set()
            return
        if isinstance(message, WorkerEvent):
            self._on_event(handle, message)
            return
        if isinstance(message, Stopped):
            _log.debug("воркер %s сообщил об остановке: %s", handle.spec.name, message.reason)
            with handle.lock:
                handle.stopping = True
            return
        _log.warning("супервизор получил неожиданное сообщение %s", type(message).__name__)

    def _on_response(self, handle: _Handle, response: Response) -> None:
        with handle.lock:
            pending = handle.pending.pop(response.id, None)
        if pending is None:
            # A late answer to a call the monitor already timed out. Dropping it
            # is correct; the caller has moved on.
            _log.debug("поздний ответ %d от воркера %s", response.id, handle.spec.name)
            return
        pending.release()
        if pending.future.done():
            return
        if response.ok:
            pending.future.set_result(response.payload)
            return
        error = response.error or ErrorInfo(
            error_type="WorkerError",
            message=f"{pending.method} failed without an error",
            user_message="Фоновый процесс Ayris не смог выполнить операцию.",
        )
        if error.traceback:
            _log.debug("трассировка из воркера %s:\n%s", handle.spec.name, error.traceback)
        pending.future.set_exception(error.to_exception())

    def _on_event(self, handle: _Handle, event: WorkerEvent) -> None:
        """Turn a worker event into a bus event, a log record, or nothing."""
        name = handle.spec.name
        if event.kind == "log":
            self._forward_log(name, event.payload)
            return
        if event.kind == "start_failed":
            with handle.lock:
                handle.start_error = ErrorInfo(
                    error_type=str(event.payload.get("error_type", "WorkerStartError")),
                    message=str(event.payload.get("message", "")),
                    user_message=str(event.payload.get("user_message", "")),
                    traceback=str(event.payload.get("traceback", "")),
                )
            handle.ready.set()
            return
        translator = self._translators.get(name)
        if translator is None:
            _log.debug("событие %s от воркера %s без переводчика", event.kind, name)
            return
        translated = translator(event.kind, event.payload)
        if translated is not None:
            self._bus.publish(translated)

    def _forward_log(self, worker: str, payload: JsonObject) -> None:
        """Re-emit a worker's log record in this process.

        One process writes the log file; the worker's records arrive here and go
        through the normal handlers under ``ayris.workers.<name>``, so a warning
        from the recognition process reads like any other warning.
        """
        level_name = str(payload.get("level", "INFO"))
        level = logging.getLevelNamesMapping().get(level_name, logging.INFO)
        message = str(payload.get("message", ""))
        origin = str(payload.get("logger", "")) or f"{_WORKER_LOGGER}.{worker}"
        logging.getLogger(origin).log(level, "%s", message)
        traceback_text = payload.get("traceback")
        if isinstance(traceback_text, str) and traceback_text:
            logging.getLogger(origin).log(level, "%s", traceback_text)
        self._bus.publish(LogLine(level=level_name, message=message, logger=origin))

    # ------------------------------------------------------------------
    # monitoring
    # ------------------------------------------------------------------

    def _ensure_monitor(self) -> None:
        if self._monitor is not None and self._monitor.is_alive():
            return
        self._stop.clear()
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="ayris-workers-monitor",
            daemon=True,
        )
        self._monitor.start()

    def _monitor_loop(self) -> None:
        while not self._stop.wait(MONITOR_TICK):
            now = time.monotonic()
            with self._lock:
                handles = tuple(self._handles.values())
            for handle in handles:
                try:
                    self._expire_calls(handle, now)
                    self._check_health(handle, now)
                    self._maybe_restart(handle, now)
                except Exception:
                    _log.exception("сбой наблюдения за воркером %s", handle.spec.name)

    def _expire_calls(self, handle: _Handle, now: float) -> None:
        """Fail calls past their deadline, leaving the worker alone.

        A slow call is not a broken worker: the heartbeat keeps arriving while a
        handler works, so a timeout means "this answer is no longer useful", not
        "restart everything". The worker is told to abandon the request.
        """
        with handle.lock:
            expired = [call for call in handle.pending.values() if call.deadline <= now]
            for call in expired:
                handle.pending.pop(call.id, None)
        for call in expired:
            call.release()
            if not call.future.done():
                call.future.set_exception(
                    WorkerTimeoutError(
                        f"{call.worker}.{call.method} did not answer in time",
                    )
                )
            self._send(handle, Control(kind=ControlKind.CANCEL, request_id=call.id))
            _log.warning("вызов %s.%s не уложился в таймаут", call.worker, call.method)

    def _check_health(self, handle: _Handle, now: float) -> None:
        """Notice a worker that died or stopped answering its heartbeat."""
        with handle.lock:
            if handle.stopping or handle.status not in {
                WorkerStatus.READY,
                WorkerStatus.STARTING,
            }:
                return
            process = handle.process
            last_beat = handle.last_beat
            ready = handle.status is WorkerStatus.READY
            timeout = handle.spec.heartbeat_timeout

        if process is None:
            return
        if not process.is_alive():
            self._worker_died(handle, exit_code=process.exitcode, reason="процесс завершился")
            return
        if ready and last_beat and now - last_beat > timeout:
            _log.warning(
                "воркер %s молчит %.1f с, считаем зависшим",
                handle.spec.name,
                now - last_beat,
            )
            self._terminate(handle)
            self._worker_died(handle, exit_code=process.exitcode, reason="нет отклика")

    def _worker_died(self, handle: _Handle, *, exit_code: int | None, reason: str) -> None:
        """Common path for a crash, a hang and an unrequested exit."""
        name = handle.spec.name
        spec = handle.spec
        now = time.monotonic()
        with handle.lock:
            if handle.status in {WorkerStatus.RESTARTING, WorkerStatus.STOPPED}:
                return
            handle.restart_history.append(now)
            recent = [
                stamp for stamp in handle.restart_history if now - stamp <= spec.restart_window
            ]
            handle.restart_history = deque(recent, maxlen=32)
            attempt = len(recent)
            handle.restarts += 1
            handle.error = reason
            handle.status = WorkerStatus.RESTARTING

        lost = self._fail_pending(
            handle,
            lambda: WorkerCrashError(f"worker {name!r} died: {reason}"),
        )
        self._terminate(handle)
        with handle.lock:
            handle.process = None
            handle.methods = ()
        _log.error(
            "воркер %s аварийно завершился (%s, код %s), потеряно вызовов: %d",
            name,
            reason,
            exit_code,
            lost,
        )
        self._bus.publish(
            WorkerCrashed(worker=name, exit_code=exit_code, error=reason, restarts=attempt)
        )

        if attempt > spec.max_restarts:
            with handle.lock:
                handle.status = WorkerStatus.FAILED
                handle.restart_at = None
            _log.error(
                "воркер %s перезапускался %d раз за %.0f с — больше не пробуем",
                name,
                attempt,
                spec.restart_window,
            )
            return

        delay = min(spec.restart_delay * (2 ** (attempt - 1)), spec.max_restart_delay)
        with handle.lock:
            handle.restart_at = now + delay
            handle.restart_reason = reason
        _log.info("воркер %s будет перезапущен через %.1f с (попытка %d)", name, delay, attempt)

    def _maybe_restart(self, handle: _Handle, now: float) -> None:
        """Perform a restart that came due."""
        with handle.lock:
            if handle.status is not WorkerStatus.RESTARTING:
                return
            due = handle.restart_at
            if due is None or now < due:
                return
            handle.restart_at = None
            attempt = len(handle.restart_history)
            reason = handle.restart_reason
            name = handle.spec.name

        try:
            self._spawn(handle)
        except WorkerStartError as exc:
            _log.error("перезапуск воркера %s не удался: %s", name, exc)
            delay = min(
                handle.spec.restart_delay * (2**attempt),
                handle.spec.max_restart_delay,
            )
            with handle.lock:
                if attempt >= handle.spec.max_restarts:
                    handle.status = WorkerStatus.FAILED
                    handle.error = str(exc)
                else:
                    handle.status = WorkerStatus.RESTARTING
                    handle.restart_at = time.monotonic() + delay
            return
        self._bus.publish(WorkerRestarted(worker=name, attempt=attempt, reason=reason))
        _log.info("воркер %s перезапущен (попытка %d)", name, attempt)

    # ------------------------------------------------------------------
    # plans, status, shutdown
    # ------------------------------------------------------------------

    def apply_plan(self, plan: WorkerPlan) -> None:
        """Bring the running set in line with ``plan``.

        Idempotent, so it doubles as the settings-changed path: workers that
        vanished from the plan are stopped, new ones are registered, ones whose
        parameters changed are restarted, and the rest are left running.
        """
        wanted = {planned.spec.name: planned for planned in plan}
        for name in self.names:
            if name not in wanted:
                _log.info("воркер %s больше не нужен", name)
                self.unregister(name)

        for planned in plan:
            spec = planned.spec
            handle = self._handles.get(spec.name)
            if handle is None:
                self.register(spec)
                if planned.autostart:
                    self._start_quietly(spec.name)
                else:
                    _log.info("воркер %s: %s", spec.name, planned.reason)
                continue

            changed = handle.spec != spec
            with handle.lock:
                handle.spec = spec
                running = handle.status is WorkerStatus.READY
            if running and changed:
                _log.info("настройки воркера %s изменились, перезапуск", spec.name)
                with contextlib.suppress(WorkerError):
                    self.restart(spec.name, reason="изменились настройки")
            elif not running and planned.autostart and handle.status is not WorkerStatus.FAILED:
                self._start_quietly(spec.name)

    def _start_quietly(self, name: str) -> None:
        """Start a worker, logging a failure instead of aborting the launch.

        One subsystem that cannot come up must not stop Ayris: the specification
        asks for a warning and a degraded assistant, not a refusal to run.
        """
        try:
            self.start(name)
        except WorkerError as exc:
            _log.error("воркер %s не запустился: %s", name, exc)

    def restart_scope(self, scope: RestartScope, settings_reason: str = "изменились настройки") -> int:
        """Restart every worker whose settings scope changed. Returns how many."""
        restarted = 0
        for name in self.names:
            handle = self._handles[name]
            if handle.spec.restart_scope != scope:
                continue
            if handle.status is WorkerStatus.READY:
                with contextlib.suppress(WorkerError):
                    self.restart(name, reason=settings_reason)
                    restarted += 1
        return restarted

    def status(self) -> tuple[WorkerSummary, ...]:
        """A snapshot of every registered worker, for DevTools and tests."""
        now = time.monotonic()
        summaries: list[WorkerSummary] = []
        with self._lock:
            handles = tuple(self._handles.values())
        for handle in handles:
            with handle.lock:
                summaries.append(
                    WorkerSummary(
                        name=handle.spec.name,
                        kind=str(handle.spec.kind),
                        status=handle.status,
                        pid=handle.pid,
                        restarts=handle.restarts,
                        pending_calls=len(handle.pending),
                        handled=handle.handled,
                        last_heartbeat_age=(now - handle.last_beat) if handle.last_beat else None,
                        methods=handle.methods,
                        error=handle.error,
                    )
                )
        return tuple(summaries)

    def worker_status(self, name: str) -> WorkerStatus:
        """Status of one worker."""
        return self._require(name).status

    def is_ready(self, name: str) -> bool:
        """Whether a worker is up and answering, without raising for unknown ones."""
        handle = self._handles.get(name)
        return handle is not None and handle.status is WorkerStatus.READY

    def shutdown(self, timeout: float | None = None) -> None:
        """Stop every worker and release the job object. Idempotent.

        After this returns, Task Manager shows no ``ayris-*`` processes: each
        worker was asked, then terminated, then killed, and whatever survived all
        three dies with the job handle.
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            names = tuple(self._handles)
        self._stop.set()
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=2.0)
        self._monitor = None

        for name in reversed(names):
            try:
                self.stop(name, timeout=timeout)
            except Exception:
                _log.exception("ошибка остановки воркера %s", name)
        with self._lock:
            self._handles.clear()
            self._translators.clear()
        if self._job is not None:
            self._job.close()
            self._job = None
        _log.info("все воркеры остановлены")

    def __enter__(self) -> WorkerManager:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return f"WorkerManager(workers={len(self._handles)})"


# ----------------------------------------------------------------------
# lifecycle wiring
# ----------------------------------------------------------------------


def install_workers(app: AyrisApp) -> WorkerManager:
    """Attach a :class:`WorkerManager` to the application lifecycle.

    Registers the manager under :attr:`~ayris.core.app.LifecycleStage.WORKERS`, so
    workers start after the database and stop before it, and hooks one restart
    handler per settings scope so that changing the recognition model recycles the
    STT worker and nothing else.

    Returns:
        The manager, for the caller to keep on the application object.
    """
    # Imported here rather than at module scope: the application imports workers,
    # and a worker child must not drag the whole lifecycle module into its process.
    from ayris.core.app import Component, LifecycleStage
    from ayris.core.config import RestartScope
    from ayris.workers.registry import plan_workers

    manager = WorkerManager(app.bus, log_dir=app.paths.logs_dir)

    def start() -> None:
        plan = plan_workers(app.settings, log_dir=app.paths.logs_dir)
        _log.info("план воркеров: %s", plan.describe())
        manager.apply_plan(plan)

    def stop() -> None:
        manager.shutdown()

    app.add_component(
        Component(
            name="воркеры",
            stage=LifecycleStage.WORKERS,
            start=start,
            stop=stop,
            kill=stop,
            stop_timeout=10.0,
        )
    )

    for scope in (
        RestartScope.AUDIO,
        RestartScope.WAKE,
        RestartScope.STT,
        RestartScope.TTS,
        RestartScope.LLM,
    ):
        app.register_restart_handler(scope, _scope_handler(manager, scope))
    return manager


def _scope_handler(manager: WorkerManager, scope: RestartScope) -> Callable[[Settings], None]:
    """Build the restart handler for one settings scope."""

    def handler(settings: Settings) -> None:
        del settings  # The plan is rebuilt from the application's settings.
        manager.restart_scope(scope)

    return handler


