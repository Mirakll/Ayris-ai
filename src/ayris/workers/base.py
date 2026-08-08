"""Base class and process entrypoint for every Ayris worker.

A worker is a plain Python object with a few methods marked by :func:`method`.
Everything around it — the pipe, the dispatch loop, the heartbeat, the log
bridge, the parent watchdog, the process priority — belongs to
:class:`_WorkerRuntime` and is set up by :func:`worker_entrypoint`, so a worker
implementation contains only its own subject matter::

    class SttWorker(Worker):
        kind = "stt"

        def on_start(self) -> None:
            self._model = load_model(self.params["model"])

        @method()
        def transcribe(self, params: JsonObject) -> JsonObject:
            chunk = AudioChunk.from_params(params)
            with open_audio(chunk) as pcm:
                return {"text": self._model.decode(pcm)}

**Spawn, not fork.** Windows has no ``fork``, and Ayris uses the ``spawn`` start
method everywhere so that behaviour matches on the developer's machine and in
CI. A worker therefore inherits *nothing*: no open database handle, no loaded
settings, no module-level globals the parent happened to set. Everything it needs
arrives in :class:`WorkerBootstrap`, and anything it wants to say goes back
through the channel.

**No Qt.** A worker process has no ``QApplication`` and must never import
PySide6: importing Qt in a child would allocate a second graphics context and,
on Windows, occasionally deadlock against the parent's message loop.

**Three threads, one dispatch.** A reader thread owns the pipe and answers
control messages immediately, so a stop or a cancel is honoured while a long
call is still running. A heartbeat thread proves liveness on a fixed interval,
independent of how busy dispatch is. A watchdog thread notices a parent that
died without closing the pipe. Requests themselves are dispatched one at a time
on the main thread: an engine that is not thread-safe — which is most of them —
would otherwise need a lock in every implementation.

**Nothing outlives the parent.** The pipe closing is the primary signal: a
supervisor that dies takes its end with it, the worker's ``recv`` raises
``EOFError`` and the process exits. The watchdog covers the rarer case of a
parent killed while a duplicated handle keeps the pipe open, and the supervisor
adds a Windows job object on top.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeVar

from ayris.core.models import JsonObject
from ayris.workers.protocol import (
    PROTOCOL_VERSION,
    Channel,
    Control,
    ControlKind,
    ErrorInfo,
    Heartbeat,
    Ready,
    Request,
    Response,
    Stopped,
    WorkerCancelledError,
    WorkerEvent,
    WorkerProtocolError,
    WorkerStartError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from multiprocessing.connection import Connection
    from pathlib import Path

__all__ = [
    "PRIORITY_CLASSES",
    "Worker",
    "WorkerBootstrap",
    "WorkerContext",
    "apply_process_priority",
    "method",
    "parent_alive",
    "resolve_worker_class",
    "windows_dll",
    "worker_entrypoint",
    "worker_methods",
]

_log = logging.getLogger("ayris.workers.base")

#: Attribute :func:`method` leaves on a function. Read by :func:`worker_methods`.
_METHOD_ATTR: Final = "_ayris_worker_method"

#: Windows priority classes, keyed by the values ``performance.audio_priority``
#: and ``performance.process_priority`` accept.
PRIORITY_CLASSES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "idle": 0x00000040,
        "below_normal": 0x00004000,
        "normal": 0x00000020,
        "above_normal": 0x00008000,
        "high": 0x00000080,
        "realtime": 0x00000100,
    }
)

#: POSIX equivalent, used when the suite runs on Linux. Negative values need
#: privileges the test runner does not have, and failing to get them is fine.
_NICE_VALUES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "idle": 10,
        "below_normal": 5,
        "normal": 0,
        "above_normal": -5,
        "high": -10,
        "realtime": -15,
    }
)

#: How long dispatch waits for a request before re-checking the stop flag.
_QUEUE_POLL: Final = 0.2

#: Exit code used when the worker could not even be constructed.
EXIT_START_FAILED: Final = 3

F = TypeVar("F", bound="Callable[..., Any]")


# ----------------------------------------------------------------------
# platform helpers
# ----------------------------------------------------------------------


def windows_dll(name: str) -> Any | None:
    """Open a Windows DLL with last-error tracking, or ``None`` off Windows.

    Shared with :mod:`ayris.workers.manager`, which needs ``kernel32`` for the job
    object the same way this module needs it for priorities and parent liveness.
    """
    factory = getattr(ctypes, "WinDLL", None)
    if factory is None:
        return None
    try:
        return factory(name, use_last_error=True)
    except OSError:
        return None


def apply_process_priority(priority: str) -> bool:
    """Raise or lower this process's scheduling priority.

    Section 12 of the specification exposes this for the audio worker, where a
    dropped buffer is audible. ``realtime`` needs a privilege a normal user
    account does not have; it is attempted and then quietly downgraded rather
    than refused, because the alternative is a worker that will not start.

    Returns:
        Whether the requested priority was applied.
    """
    if sys.platform != "win32":
        wanted = _NICE_VALUES.get(priority, 0)
        if wanted == 0:
            return True
        try:
            os.nice(wanted)
        except (OSError, PermissionError):
            _log.debug("не удалось изменить nice до %d", wanted)
            return False
        return True

    kernel32 = windows_dll("kernel32")
    wanted_class = PRIORITY_CLASSES.get(priority)
    if kernel32 is None or wanted_class is None:
        return False
    for candidate in (wanted_class, PRIORITY_CLASSES["high"]):
        try:
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            handle = kernel32.GetCurrentProcess()
            if kernel32.SetPriorityClass(ctypes.c_void_p(handle), candidate):
                return candidate == wanted_class
        except (AttributeError, OSError):
            break
    _log.warning("приоритет процесса «%s» не применён", priority)
    return False


def parent_alive(parent_pid: int) -> bool:
    """Whether the process that spawned this worker is still running.

    On POSIX a reparented child sees its ``getppid`` change, which is the cheapest
    possible check. On Windows the parent is opened for synchronisation and polled
    with a zero timeout: ``WAIT_TIMEOUT`` means still running.
    """
    if sys.platform != "win32":
        return os.getppid() == parent_pid

    kernel32 = windows_dll("kernel32")
    if kernel32 is None:
        return True
    synchronize = 0x00100000
    wait_timeout = 0x00000102
    try:
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32)
        handle = kernel32.OpenProcess(synchronize, False, parent_pid)
        if not handle:
            return False
        try:
            kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            status = int(kernel32.WaitForSingleObject(ctypes.c_void_p(handle), 0))
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except (AttributeError, OSError):
        # Cannot tell; assume alive rather than kill a healthy worker.
        return True
    return status == wait_timeout


# ----------------------------------------------------------------------
# bootstrap payload
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerBootstrap:
    """Everything a spawned worker needs, because it inherits nothing.

    Args:
        name: Identifier the supervisor uses in events and log lines.
        kind: Worker type, e.g. ``"stt"``. Informational.
        entrypoint: ``"package.module:ClassName"`` of the :class:`Worker` subclass.
        params: Configuration for the worker, already flattened out of
            :class:`~ayris.core.config.Settings` by the caller. Must be picklable.
        parent_pid: Process the watchdog watches.
        heartbeat_interval: Seconds between heartbeats. The supervisor's timeout
            is a multiple of this.
        parent_check_interval: Seconds between watchdog polls.
        priority: Value for :func:`apply_process_priority`.
        log_level: Threshold for this worker's own logger.
        log_dir: Where to write the worker's log file. ``None`` keeps logging in
            memory and relies on the bridge to the parent.
        log_forward_level: Records at or above this level are forwarded to the
            supervisor over the channel, which is how a worker's warnings reach
            the main log without two processes writing one file.
        python_path: Extra ``sys.path`` entries added before the entrypoint is
            imported. Used for plugin workers, and by the test suite for its
            fixture worker.
        protocol_version: Checked against the worker's own; a mismatch means a
            stale process survived an update.
    """

    name: str
    entrypoint: str
    kind: str = "generic"
    params: JsonObject = field(default_factory=dict)
    parent_pid: int = 0
    heartbeat_interval: float = 2.0
    parent_check_interval: float = 2.0
    priority: str = "normal"
    log_level: str = "INFO"
    log_dir: Path | None = None
    log_forward_level: str = "WARNING"
    python_path: tuple[str, ...] = ()
    protocol_version: int = PROTOCOL_VERSION


# ----------------------------------------------------------------------
# worker API
# ----------------------------------------------------------------------


def method(name: str | None = None) -> Callable[[F], F]:
    """Expose a worker method to :meth:`WorkerManager.call`.

    Args:
        name: Wire name. Defaults to the function name, which is what almost
            every worker wants; pass it explicitly to keep the wire name stable
            while renaming the Python method.

    The decorated function takes the request's ``params`` dict and returns
    anything picklable. Raising is a normal outcome: the exception is flattened
    into the response and re-raised in the caller.
    """

    def decorate(func: F) -> F:
        setattr(func, _METHOD_ATTR, name or func.__name__)
        return func

    return decorate


@cache
def worker_methods(worker_class: type[Worker]) -> Mapping[str, str]:
    """Wire name to attribute name for every :func:`method` on ``worker_class``.

    Walks the MRO from the base down, so a subclass may override an inherited
    method under the same wire name. Cached per class — the answer cannot change
    once the class object exists.
    """
    found: dict[str, str] = {}
    for klass in reversed(worker_class.__mro__):
        for attribute, value in vars(klass).items():
            marker = getattr(value, _METHOD_ATTR, None)
            if isinstance(marker, str):
                found[marker] = attribute
    return MappingProxyType(found)


class WorkerContext:
    """The outside world, as a worker sees it.

    Handed to the worker's constructor. Everything a worker needs from its
    supervisor is here, and nothing else is reachable — a worker cannot publish
    onto the event bus, touch the database or read the settings file, because
    none of those exist in its process.
    """

    __slots__ = ("_runtime", "kind", "name")

    def __init__(self, runtime: _WorkerRuntime, name: str, kind: str) -> None:
        self._runtime = runtime
        self.name = name
        self.kind = kind

    @property
    def params(self) -> JsonObject:
        """Current configuration. Replaced wholesale by ``configure``."""
        return self._runtime.params

    @property
    def stopping(self) -> bool:
        """Whether a stop was requested. A long loop should check this."""
        return self._runtime.stopping

    @property
    def cancelled(self) -> bool:
        """Whether the request being handled right now was cancelled."""
        return self._runtime.current_cancelled

    def check_cancelled(self) -> None:
        """Abort the current handler if it was cancelled or a stop is pending.

        Raises:
            WorkerCancelledError: Cancellation was requested. The supervisor
                reports it as a cancelled call rather than a failure.
        """
        if self.cancelled:
            raise WorkerCancelledError("call cancelled by the supervisor")
        if self.stopping:
            raise WorkerCancelledError("worker is shutting down")

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        """Send an unsolicited event to the supervisor.

        Cheap, but not free — it crosses a pipe and ends up on the event bus. The
        audio worker throttles its level events rather than sending one per frame.
        """
        self._runtime.emit(kind, payload)

    def logger(self, suffix: str = "") -> logging.Logger:
        """A logger whose records reach the main process's log file."""
        base = f"ayris.workers.{self.name}"
        return logging.getLogger(f"{base}.{suffix}" if suffix else base)

    def __repr__(self) -> str:
        return f"WorkerContext({self.name}, kind={self.kind})"


class Worker:
    """Base class for the process-side half of a subsystem.

    Subclasses override :meth:`on_start` and :meth:`on_stop` for setup and
    teardown, and mark their callable surface with :func:`method`. Three methods
    come for free — ``ping``, ``stats`` and ``configure`` — so the supervisor can
    probe and reconfigure any worker without knowing what it does.

    Args:
        context: Supplied by the runtime; a subclass forwards it unchanged.
    """

    #: Worker type, mirrored into log lines and events. Subclasses set it.
    kind: ClassVar[str] = "generic"

    def __init__(self, context: WorkerContext) -> None:
        self.context = context
        self.log = context.logger()
        self._started_at = time.monotonic()

    # -- lifecycle hooks ------------------------------------------------

    def on_start(self) -> None:
        """Load models, open devices, connect clients.

        Runs before the supervisor is told the worker is ready, so anything
        raised here fails the start rather than the first call. Slow by design:
        a speech model takes seconds, which is exactly why this happens in
        another process.
        """

    def on_stop(self) -> None:
        """Release whatever :meth:`on_start` acquired. Must not raise."""

    def on_configure(self, params: JsonObject) -> None:
        """React to new settings that did not need a restart.

        The default swaps :attr:`params` and does nothing else, which is right for
        a worker whose parameters are read per call.
        """

    # -- conveniences ---------------------------------------------------

    @property
    def params(self) -> JsonObject:
        """Current configuration, as passed in :class:`WorkerBootstrap`."""
        return self.context.params

    @property
    def uptime(self) -> float:
        """Seconds since the worker object was created."""
        return time.monotonic() - self._started_at

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        """Shorthand for :meth:`WorkerContext.emit`."""
        self.context.emit(kind, payload)

    # -- built-in methods -----------------------------------------------

    @method()
    def ping(self, params: JsonObject) -> JsonObject:
        """Round-trip probe. Answers with whatever it was given, plus identity."""
        return {"pong": True, "pid": os.getpid(), "name": self.context.name, "echo": params}

    @method()
    def stats(self, _params: JsonObject) -> JsonObject:
        """Uptime, dispatched calls and the method surface, for DevTools."""
        return {
            "name": self.context.name,
            "kind": self.kind,
            "pid": os.getpid(),
            "uptime_sec": round(self.uptime, 3),
            "methods": sorted(worker_methods(type(self))),
        }

    @method()
    def configure(self, params: JsonObject) -> JsonObject:
        """Replace the worker's parameters without restarting the process."""
        self.context._runtime.replace_params(params)
        self.on_configure(params)
        return {"configured": True}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.context.name!r}, kind={self.kind!r})"


def resolve_worker_class(entrypoint: str) -> type[Worker]:
    """Import ``"package.module:ClassName"`` and return the class.

    Raises:
        WorkerStartError: The string is malformed, the module does not import, or
            the attribute is not a :class:`Worker` subclass. The message names the
            entrypoint, because this is what a typo in the registry looks like.
    """
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise WorkerStartError(f"malformed worker entrypoint {entrypoint!r}, expected module:Class")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise WorkerStartError(f"cannot import worker module {module_name!r}: {exc}") from exc
    candidate = getattr(module, attribute, None)
    if not isinstance(candidate, type) or not issubclass(candidate, Worker):
        raise WorkerStartError(f"{entrypoint} is not a Worker subclass")
    return candidate


# ----------------------------------------------------------------------
# runtime
# ----------------------------------------------------------------------


class _PipeLogHandler(logging.Handler):
    """Forward log records to the supervisor instead of to a second log file.

    Two processes appending to one rotating file interleave records and race on
    rollover, and on Windows the rename fails outright while another process holds
    the handle. Forwarding keeps one writer, and the supervisor re-emits each
    record under ``ayris.workers.<name>`` so it lands in the normal log with the
    worker's identity intact.
    """

    def __init__(self, runtime: _WorkerRuntime, level: int) -> None:
        super().__init__(level)
        self._runtime = runtime
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        # A failure inside the channel logs, which would re-enter this handler.
        if getattr(self._local, "sending", False):
            return
        self._local.sending = True
        try:
            payload: JsonObject = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
            if record.exc_info is not None:
                payload["traceback"] = self.format(record)
            self._runtime.emit("log", payload)
        except Exception:
            # A broken log line must not take the worker down with it.
            pass
        finally:
            self._local.sending = False


class _WorkerRuntime:
    """Owns the process: channel, threads, dispatch and teardown.

    Split from :class:`Worker` so that a worker implementation never has to think
    about any of it, and so the loop can be exercised without a real subsystem.
    """

    __slots__ = (
        "_bootstrap",
        "_cancelled",
        "_channel",
        "_current_id",
        "_handled",
        "_heartbeat",
        "_lock",
        "_params",
        "_queue",
        "_reader",
        "_seq",
        "_stop",
        "_stop_reason",
        "_watchdog",
        "_worker",
    )

    def __init__(self, bootstrap: WorkerBootstrap, channel: Channel) -> None:
        self._bootstrap = bootstrap
        self._channel = channel
        self._params: JsonObject = dict(bootstrap.params)
        self._queue: queue.Queue[Request | None] = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._cancelled: set[int] = set()
        self._current_id: int | None = None
        self._handled = 0
        self._seq = 0
        self._stop_reason = ""
        self._worker: Worker | None = None
        self._reader: threading.Thread | None = None
        self._heartbeat: threading.Thread | None = None
        self._watchdog: threading.Thread | None = None

    # -- state seen by the worker ---------------------------------------

    @property
    def params(self) -> JsonObject:
        with self._lock:
            return self._params

    def replace_params(self, params: JsonObject) -> None:
        with self._lock:
            self._params = dict(params)

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    @property
    def current_cancelled(self) -> bool:
        with self._lock:
            return self._current_id is not None and self._current_id in self._cancelled

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        self._send(WorkerEvent(kind=kind, payload=dict(payload) if payload else {}))

    # -- plumbing -------------------------------------------------------

    def _send(self, message: Request | Response | WorkerEvent | Heartbeat | Ready | Stopped) -> None:
        """Write to the channel, treating a dead pipe as a stop signal."""
        try:
            self._channel.send(message)
        except (OSError, EOFError, BrokenPipeError):
            self.request_stop("channel closed")
        except Exception:
            # Most often an unpicklable payload. The caller decides what to do;
            # here the only job is to not take the process down with it.
            raise

    def request_stop(self, reason: str) -> None:
        """Ask the loop to finish. Safe from any thread, idempotent."""
        if self._stop.is_set():
            return
        self._stop_reason = reason
        self._stop.set()
        # Unblock dispatch even when no request is queued.
        self._queue.put(None)

    # -- threads --------------------------------------------------------

    def _read_loop(self) -> None:
        """Own the pipe. Control messages are handled here, requests are queued."""
        while not self._stop.is_set():
            try:
                message = self._channel.recv()
            except EOFError:
                self.request_stop("parent closed the channel")
                return
            except OSError:
                self.request_stop("channel error")
                return
            except Exception:
                _log.exception("нечитаемое сообщение от супервизора")
                continue
            if isinstance(message, Control):
                self._handle_control(message)
            elif isinstance(message, Request):
                self._queue.put(message)
            else:
                _log.warning("воркер получил неожиданное сообщение %s", type(message).__name__)

    def _handle_control(self, control: Control) -> None:
        if control.kind is ControlKind.STOP:
            self.request_stop("stop requested")
        elif control.kind is ControlKind.PING:
            self._send(self._beat())
        elif control.kind is ControlKind.CANCEL:
            if control.request_id is None:
                # A bare cancel aborts whatever is running, which is what the
                # stop word and the cancel hotkey mean.
                with self._lock:
                    if self._current_id is not None:
                        self._cancelled.add(self._current_id)
            else:
                with self._lock:
                    self._cancelled.add(control.request_id)

    def _beat(self) -> Heartbeat:
        with self._lock:
            self._seq += 1
            return Heartbeat(
                seq=self._seq,
                at=time.time(),
                busy=self._current_id is not None,
                handled=self._handled,
            )

    def _heartbeat_loop(self) -> None:
        """Prove liveness on a fixed interval, however busy dispatch is."""
        interval = max(self._bootstrap.heartbeat_interval, 0.05)
        while not self._stop.wait(interval):
            self._send(self._beat())

    def _watchdog_loop(self) -> None:
        """Exit if the parent vanished without closing the pipe.

        Rare but real: a parent killed with ``TerminateProcess`` while a third
        process holds a duplicate of the pipe handle leaves the worker blocked in
        ``recv`` forever, which is precisely the orphan the specification forbids.
        """
        parent = self._bootstrap.parent_pid
        if not parent:
            return
        interval = max(self._bootstrap.parent_check_interval, 0.1)
        while not self._stop.wait(interval):
            if not parent_alive(parent):
                _log.warning("родительский процесс %d исчез, воркер завершается", parent)
                self.request_stop("parent process gone")
                # Nothing will ever answer on the channel again; closing it
                # unblocks the reader thread.
                self._channel.close()
                return

    # -- dispatch -------------------------------------------------------

    def _dispatch(self, request: Request, worker: Worker) -> None:
        with self._lock:
            self._current_id = request.id
        started = time.perf_counter()
        try:
            handler = self._resolve(request.method, worker)
            payload = handler(request.params)
        except Exception as exc:
            duration = (time.perf_counter() - started) * 1000.0
            if not isinstance(exc, WorkerCancelledError):
                _log.warning("метод %s завершился ошибкой: %s", request.method, exc)
            response = Response(
                id=request.id,
                ok=False,
                error=ErrorInfo.from_exception(exc),
                duration_ms=duration,
            )
        else:
            duration = (time.perf_counter() - started) * 1000.0
            response = Response(id=request.id, ok=True, payload=payload, duration_ms=duration)
        finally:
            with self._lock:
                self._current_id = None
                self._cancelled.discard(request.id)
                self._handled += 1

        self._reply(request, response)

    def _reply(self, request: Request, response: Response) -> None:
        """Send a response, degrading to an error response if it will not pickle."""
        try:
            self._send(response)
        except Exception as exc:
            _log.exception("не удалось отправить ответ метода %s", request.method)
            self._send(
                Response(
                    id=request.id,
                    ok=False,
                    error=ErrorInfo(
                        error_type=type(exc).__name__,
                        message=f"result of {request.method} cannot be sent: {exc}",
                        user_message="Фоновый процесс вернул результат, который не удалось передать.",
                    ),
                    duration_ms=response.duration_ms,
                )
            )

    @staticmethod
    def _resolve(name: str, worker: Worker) -> Callable[[JsonObject], Any]:
        attribute = worker_methods(type(worker)).get(name)
        if attribute is None:
            known = ", ".join(sorted(worker_methods(type(worker))))
            raise WorkerProtocolError(f"unknown worker method {name!r}; known methods: {known}")
        handler: Callable[[JsonObject], Any] = getattr(worker, attribute)
        return handler

    # -- entry ----------------------------------------------------------

    def run(self) -> int:
        """Bring the process up, serve requests, tear it down. Never raises."""
        self._configure_logging()
        self._install_signal_handlers()
        applied = apply_process_priority(self._bootstrap.priority)
        if not applied and self._bootstrap.priority != "normal":
            _log.info("приоритет «%s» недоступен, оставлен обычный", self._bootstrap.priority)

        self._reader = self._spawn_thread(self._read_loop, "reader")

        worker = self._build_worker()
        if worker is None:
            return EXIT_START_FAILED
        self._worker = worker

        self._heartbeat = self._spawn_thread(self._heartbeat_loop, "heartbeat")
        self._watchdog = self._spawn_thread(self._watchdog_loop, "watchdog")
        self._send(
            Ready(
                name=self._bootstrap.name,
                pid=os.getpid(),
                protocol_version=PROTOCOL_VERSION,
                methods=tuple(sorted(worker_methods(type(worker)))),
            )
        )
        _log.info("воркер %s готов (pid %d)", self._bootstrap.name, os.getpid())

        self._serve(worker)
        return self._teardown(worker)

    def _serve(self, worker: Worker) -> None:
        while not self._stop.is_set():
            try:
                request = self._queue.get(timeout=_QUEUE_POLL)
            except queue.Empty:
                continue
            if request is None:
                break
            self._dispatch(request, worker)

    def _teardown(self, worker: Worker) -> int:
        reason = self._stop_reason or "stopped"
        _log.info("воркер %s останавливается: %s", self._bootstrap.name, reason)
        self._stop.set()
        try:
            worker.on_stop()
        except Exception:
            _log.exception("ошибка в on_stop воркера %s", self._bootstrap.name)

        # Best effort: the parent may already be gone, in which case send() has
        # nowhere to go and quietly turns into a no-op.
        self._send(Stopped(reason=reason))
        for thread in (self._heartbeat, self._watchdog):
            if thread is not None:
                thread.join(timeout=1.0)
        self._channel.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        logging.shutdown()
        return 0

    def _build_worker(self) -> Worker | None:
        """Import and construct the worker, reporting failure over the channel."""
        for entry in self._bootstrap.python_path:
            if entry not in sys.path:
                sys.path.insert(0, entry)
        context = WorkerContext(self, self._bootstrap.name, self._bootstrap.kind)
        try:
            worker_class = resolve_worker_class(self._bootstrap.entrypoint)
            worker = worker_class(context)
            worker.on_start()
        except Exception as exc:
            _log.exception("воркер %s не запустился", self._bootstrap.name)
            self._send(WorkerEvent(kind="start_failed", payload=_error_payload(exc)))
            self._channel.close()
            return None
        return worker

    def _spawn_thread(self, target: Callable[[], None], suffix: str) -> threading.Thread:
        thread = threading.Thread(
            target=target,
            name=f"ayris-{self._bootstrap.name}-{suffix}",
            daemon=True,
        )
        thread.start()
        return thread

    def _configure_logging(self) -> None:
        """Set up this process's logging: a file when asked, plus the bridge."""
        root = logging.getLogger("ayris")
        root.propagate = False
        level = logging.getLevelNamesMapping().get(self._bootstrap.log_level.upper(), logging.INFO)
        root.setLevel(level)
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        if self._bootstrap.log_dir is not None:
            # Imported here, not at module scope: a worker that logs only through
            # the bridge should not pay for the handler machinery.
            from ayris.utils.logger import DailySizedRotatingFileHandler

            try:
                file_handler = DailySizedRotatingFileHandler(
                    self._bootstrap.log_dir,
                    f"worker_{self._bootstrap.name}",
                )
            except OSError:
                pass
            else:
                file_handler.setLevel(level)
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                root.addHandler(file_handler)

        forward = logging.getLevelNamesMapping().get(
            self._bootstrap.log_forward_level.upper(), logging.WARNING
        )
        bridge = _PipeLogHandler(self, forward)
        bridge.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(bridge)

    def _install_signal_handlers(self) -> None:
        """Turn a signal into a graceful stop where the platform sends one.

        Windows does not deliver ``SIGTERM``: ``Process.terminate`` calls
        ``TerminateProcess``, and there is nothing to catch. The supervisor
        therefore always asks over the channel first and only terminates a worker
        that ignored it.
        """
        for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
            handler_signal = getattr(signal, name, None)
            if handler_signal is None:
                continue
            try:
                signal.signal(handler_signal, self._on_signal)
            except (OSError, ValueError):
                continue

    def _on_signal(self, signum: int, _frame: object) -> None:
        self.request_stop(f"signal {signum}")


def _error_payload(exc: BaseException) -> JsonObject:
    """Flatten a start-up failure into an event payload."""
    info = ErrorInfo.from_exception(exc)
    return {
        "error_type": info.error_type,
        "message": info.message,
        "user_message": info.user_message,
        "traceback": info.traceback,
    }


def worker_entrypoint(bootstrap: WorkerBootstrap, connection: Connection) -> None:
    """Process entrypoint. Target of the spawned :class:`multiprocessing.Process`.

    Module-level and picklable by reference, which ``spawn`` requires: the child
    interpreter re-imports this module and looks the function up by name rather
    than receiving any of the parent's state.
    """
    channel = Channel(connection)
    if bootstrap.protocol_version != PROTOCOL_VERSION:
        # Nothing can be trusted across a version gap, not even the shape of the
        # error frame, so this is reported by exit code alone.
        channel.close()
        raise SystemExit(EXIT_START_FAILED)
    runtime = _WorkerRuntime(bootstrap, channel)
    raise SystemExit(runtime.run())
