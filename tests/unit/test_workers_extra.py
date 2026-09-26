"""Дополнительные модульные тесты воркеров: платформенные помощники, рантайм,
реестр и супервизор — синхронно, без реальных процессов, устройств и сети.

Файл только добавляет покрытие к :mod:`ayris.workers.base`,
:mod:`ayris.workers.registry` и :mod:`ayris.workers.manager`; он не дублирует
``test_workers.py`` и не поднимает настоящих дочерних процессов. Любой поток,
который здесь стартует, детерминированно останавливается и join-ится в самом
тесте, поэтому висящих потоков и таймеров не остаётся.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import types
from typing import TYPE_CHECKING, Any

import pytest

from ayris.core.events import EventBus
from ayris.workers import base, registry
from ayris.workers import manager as manager_mod
from ayris.workers.base import (
    Worker,
    WorkerBootstrap,
    WorkerContext,
    apply_process_priority,
    parent_alive,
    resolve_worker_class,
    windows_dll,
)
from ayris.workers.manager import WorkerManager, WorkerStatus

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

# ----------------------------------------------------------------------
# fakes — no real Qt, no real devices, no real child processes
# ----------------------------------------------------------------------


class _FakeCFunc:
    """A stand-in for a ctypes foreign function: callable and attribute-settable.

    Real ctypes functions accept ``argtypes``/``restype`` assignment, so the
    WinAPI helpers set them on the fly; the fake has to tolerate that.
    """

    def __init__(self, result: object = 0, *, raises: BaseException | None = None) -> None:
        self._result = result
        self._raises = raises
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *_args: object) -> object:
        if self._raises is not None:
            raise self._raises
        return self._result


def _fake_kernel32(**funcs: _FakeCFunc) -> types.SimpleNamespace:
    """A namespace whose attributes are :class:`_FakeCFunc` instances."""
    return types.SimpleNamespace(**funcs)


class _ScriptedChannel:
    """A :class:`~ayris.workers.protocol.Channel` look-alike for the runtime.

    ``recv`` replays a script — messages are returned, exception instances are
    raised — and an exhausted script raises ``EOFError``. ``send`` records.
    """

    def __init__(self, script: list[object] | None = None) -> None:
        self._script = list(script or [])
        self.sent: list[object] = []
        self.closed = False

    def recv(self) -> object:
        if not self._script:
            raise EOFError
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def send(self, message: object) -> None:
        self.sent.append(message)

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self._script)

    def close(self) -> None:
        self.closed = True


class _SendChannel:
    """Channel whose ``send`` raises a chosen error; used for the manager's ``_send``."""

    def __init__(self, error: BaseException | None) -> None:
        self._error = error
        self.sent: list[object] = []

    def send(self, message: object) -> None:
        if self._error is not None:
            raise self._error
        self.sent.append(message)

    def close(self) -> None: ...


class _FakePipeEnd:
    """One end of a fake duplex pipe: only ``close`` is exercised by ``_spawn``."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    """A ``multiprocessing`` process stand-in with scriptable liveness."""

    def __init__(
        self,
        *,
        start_error: BaseException | None = None,
        alive: bool = False,
        terminate_stops: bool = True,
        kill_stops: bool = True,
        pid: int | None = 4321,
    ) -> None:
        self._start_error = start_error
        self._alive = alive
        self._terminate_stops = terminate_stops
        self._kill_stops = kill_stops
        self.pid = pid
        self.exitcode: int | None = 1
        self.started = False

    def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def join(self, _timeout: float | None = None) -> None: ...

    def terminate(self) -> None:
        if self._terminate_stops:
            self._alive = False

    def kill(self) -> None:
        if self._kill_stops:
            self._alive = False


class _FakeContext:
    """A ``SpawnContext`` stand-in returning fake pipe ends and one fake process."""

    def __init__(self, process: _FakeProcess) -> None:
        self._process = process

    def Pipe(self, duplex: bool = True) -> tuple[_FakePipeEnd, _FakePipeEnd]:  # noqa: N802
        del duplex
        return _FakePipeEnd(), _FakePipeEnd()

    def Process(self, **_kw: object) -> _FakeProcess:  # noqa: N802
        return self._process


class _AliveRaises:
    """A process whose ``is_alive`` always raises — to trip error handling."""

    def __init__(self) -> None:
        self.pid = 11
        self.exitcode: int | None = None
        self.calls = 0

    def is_alive(self) -> bool:
        self.calls += 1
        raise RuntimeError("boom")

    def join(self, _timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class _HandlerRuntime:
    """Minimal runtime for :class:`_PipeLogHandler`: records or fails ``emit``."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail
        self.events: list[tuple[str, dict[str, Any]]] = []

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._fail:
            raise RuntimeError("channel down")
        self.events.append((kind, payload))


class _FakeWorker:
    """A worker object for ``run``/``_teardown`` that never touches a subsystem."""

    def __init__(self, *, stop_raises: bool = False) -> None:
        self._stop_raises = stop_raises
        self.stopped = False

    def on_stop(self) -> None:
        self.stopped = True
        if self._stop_raises:
            raise RuntimeError("on_stop failed")


class _TameRuntime(base._WorkerRuntime):
    """A runtime whose global side effects (logging, signals, worker) are inert.

    The subclass carries a ``__dict__``, so a test sets ``_fake_worker`` on the
    instance; leaving it unset makes ``_build_worker`` report a start failure.
    """

    def _configure_logging(self) -> None: ...

    def _install_signal_handlers(self) -> None: ...

    def _build_worker(self) -> _FakeWorker | None:
        return getattr(self, "_fake_worker", None)


class _OneShotStop:
    """A ``threading.Event`` look-alike that lets ``_monitor_loop`` run once.

    ``wait`` returns ``False`` on the first call and ``True`` afterwards, so the
    supervisor's ``while not self._stop.wait(...)`` executes exactly one body.
    """

    def __init__(self) -> None:
        self._calls = 0
        self._set = False

    def wait(self, _timeout: float | None = None) -> bool:
        if self._set:
            return True
        self._calls += 1
        return self._calls > 1

    def is_set(self) -> bool:
        return self._set

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False


# ----------------------------------------------------------------------
# helpers and fixtures
# ----------------------------------------------------------------------


def _spec(
    name: str = "w", *, entrypoint: str = "ayris.workers.base:Worker", **kw: object
) -> registry.WorkerSpec:
    return registry.WorkerSpec(name=name, entrypoint=entrypoint, **kw)


def _bootstrap(**kw: object) -> WorkerBootstrap:
    defaults: dict[str, object] = {"name": "w", "entrypoint": "ayris.workers.base:Worker"}
    defaults.update(kw)
    return WorkerBootstrap(**defaults)


def _insert(
    manager: WorkerManager,
    name: str = "w",
    *,
    status: WorkerStatus = WorkerStatus.REGISTERED,
    process: object = None,
    channel: object = None,
    **speckw: object,
) -> manager_mod._Handle:
    """Register a handle directly, bypassing the monitor thread ``register`` starts."""
    handle = manager_mod._Handle(spec=_spec(name, **speckw))
    handle.status = status
    handle.process = process  # type: ignore[assignment]
    handle.channel = channel  # type: ignore[assignment]
    manager._handles[name] = handle
    return handle


@pytest.fixture
def bus() -> EventBus:
    return EventBus(thread_id=None)


@pytest.fixture
def manager(bus: EventBus) -> Iterator[WorkerManager]:
    instance = WorkerManager(bus)
    try:
        yield instance
    finally:
        instance.shutdown()


class TestPlatformHelpers:
    """``windows_dll``, ``apply_process_priority`` and ``parent_alive``."""

    def test_windows_dll_opens_and_reports_failure(self) -> None:
        """A real library opens; a missing one is swallowed into ``None``."""
        assert windows_dll("kernel32") is not None
        assert windows_dll("no_such_library_xyz_123") is None

    def test_windows_dll_without_factory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Off Windows there is no ``WinDLL``; the helper returns ``None``."""
        monkeypatch.delattr(ctypes, "WinDLL", raising=False)
        assert windows_dll("kernel32") is None

    def test_priority_posix_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The POSIX path: zero nice is a no-op, a change is attempted, failure is soft."""
        monkeypatch.setattr(base.sys, "platform", "linux")
        assert apply_process_priority("normal") is True

        applied: list[int] = []
        monkeypatch.setattr(base.os, "nice", applied.append, raising=False)
        assert apply_process_priority("idle") is True
        assert applied == [10]

        def _boom(_n: int) -> int:
            raise OSError("no permission")

        monkeypatch.setattr(base.os, "nice", _boom, raising=False)
        assert apply_process_priority("high") is False

    def test_priority_windows_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Windows path applies the requested class through kernel32."""
        monkeypatch.setattr(base.sys, "platform", "win32")
        kernel32 = _fake_kernel32(
            SetPriorityClass=_FakeCFunc(1),
            GetCurrentProcess=_FakeCFunc(7),
        )
        monkeypatch.setattr(base, "windows_dll", lambda _name: kernel32)
        assert apply_process_priority("normal") is True

    def test_priority_windows_missing_pieces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No kernel32 or an unknown class means the priority is not applied."""
        monkeypatch.setattr(base.sys, "platform", "win32")
        monkeypatch.setattr(base, "windows_dll", lambda _name: None)
        assert apply_process_priority("normal") is False

        kernel32 = _fake_kernel32(
            SetPriorityClass=_FakeCFunc(1),
            GetCurrentProcess=_FakeCFunc(7),
        )
        monkeypatch.setattr(base, "windows_dll", lambda _name: kernel32)
        assert apply_process_priority("bogus-class") is False

    def test_priority_windows_syscall_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A raising ``SetPriorityClass`` is caught and reported as failure."""
        monkeypatch.setattr(base.sys, "platform", "win32")
        kernel32 = _fake_kernel32(
            SetPriorityClass=_FakeCFunc(raises=OSError("denied")),
            GetCurrentProcess=_FakeCFunc(7),
        )
        monkeypatch.setattr(base, "windows_dll", lambda _name: kernel32)
        assert apply_process_priority("high") is False

    def test_parent_alive_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On POSIX the check is a ``getppid`` comparison."""
        monkeypatch.setattr(base.sys, "platform", "linux")
        monkeypatch.setattr(base.os, "getppid", lambda: 999, raising=False)
        assert parent_alive(999) is True
        assert parent_alive(1000) is False

    def test_parent_alive_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Windows path opens the parent and reads ``WaitForSingleObject``."""
        monkeypatch.setattr(base.sys, "platform", "win32")
        wait_timeout = 0x00000102

        kernel32 = _fake_kernel32(
            OpenProcess=_FakeCFunc(123),
            WaitForSingleObject=_FakeCFunc(wait_timeout),
            CloseHandle=_FakeCFunc(0),
        )
        monkeypatch.setattr(base, "windows_dll", lambda _name: kernel32)
        assert parent_alive(555) is True

        kernel32.WaitForSingleObject = _FakeCFunc(0)
        assert parent_alive(555) is False

    def test_parent_alive_windows_edge_cases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No kernel32, an unopenable handle and a raising call all resolve safely."""
        monkeypatch.setattr(base.sys, "platform", "win32")
        monkeypatch.setattr(base, "windows_dll", lambda _name: None)
        assert parent_alive(1) is True

        closed = _fake_kernel32(OpenProcess=_FakeCFunc(0))
        monkeypatch.setattr(base, "windows_dll", lambda _name: closed)
        assert parent_alive(1) is False

        raiser = _fake_kernel32(OpenProcess=_FakeCFunc(raises=OSError("no access")))
        monkeypatch.setattr(base, "windows_dll", lambda _name: raiser)
        assert parent_alive(1) is True


class TestResolveWorkerClass:
    """``resolve_worker_class`` maps ``module:Class`` to a class or fails loudly."""

    def test_returns_worker_subclass(self) -> None:
        assert resolve_worker_class("ayris.workers.base:Worker") is Worker

    def test_malformed_entrypoint(self) -> None:
        with pytest.raises(base.WorkerStartError):
            resolve_worker_class("no-colon-here")

    def test_missing_module(self) -> None:
        with pytest.raises(base.WorkerStartError):
            resolve_worker_class("no.such.module.zzz:Klass")

    def test_not_a_worker_subclass(self) -> None:
        with pytest.raises(base.WorkerStartError):
            resolve_worker_class("ayris.workers.base:WorkerContext")


class TestPipeLogHandler:
    """The bridge that ships a worker's log records to the supervisor."""

    def _record(
        self, message: str = "hi %s", args: tuple[object, ...] = ("there",)
    ) -> logging.LogRecord:
        return logging.LogRecord("ayris.test", logging.WARNING, __file__, 1, message, args, None)

    def test_emit_forwards_record(self) -> None:
        runtime = _HandlerRuntime()
        handler = base._PipeLogHandler(runtime, logging.WARNING)
        handler.emit(self._record())
        assert runtime.events[0][0] == "log"
        assert runtime.events[0][1]["message"] == "hi there"

    def test_emit_includes_traceback(self) -> None:
        runtime = _HandlerRuntime()
        handler = base._PipeLogHandler(runtime, logging.ERROR)
        handler.setFormatter(logging.Formatter("%(message)s"))
        try:
            raise ValueError("nope")
        except ValueError:
            record = logging.LogRecord(
                "ayris.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
            )
        handler.emit(record)
        assert "traceback" in runtime.events[0][1]

    def test_emit_reentrancy_guard(self) -> None:
        runtime = _HandlerRuntime()
        handler = base._PipeLogHandler(runtime, logging.WARNING)
        handler._local.sending = True
        handler.emit(self._record())
        assert runtime.events == []

    def test_emit_swallows_channel_failure(self) -> None:
        runtime = _HandlerRuntime(fail=True)
        handler = base._PipeLogHandler(runtime, logging.WARNING)
        handler.emit(self._record())  # must not raise
        assert handler._local.sending is False


class TestContextAndWorkerRepr:
    """The ``__repr__`` helpers on the context and the worker base class."""

    def test_context_repr(self) -> None:
        context = WorkerContext(object(), "voice", "stt")  # type: ignore[arg-type]
        assert "voice" in repr(context)

    def test_worker_repr(self) -> None:
        context = WorkerContext(object(), "voice", "stt")  # type: ignore[arg-type]
        worker = Worker(context)
        text = repr(worker)
        assert "Worker" in text
        assert "voice" in text


# PLACEHOLDER_TESTS
