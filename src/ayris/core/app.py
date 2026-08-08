"""Composition root: every subsystem, brought up and taken down in order.

:class:`AyrisApp` owns the objects the rest of Ayris needs — paths, settings,
database, repositories, event bus, state machine — and hands them out through
properties. Nothing here reaches for a module-level singleton, so a test can
build a whole application inside ``tmp_path`` and throw it away.

Startup walks :class:`LifecycleStage` in declaration order and shutdown walks it
backwards, which is the only ordering guarantee the later tasks need: a worker
registered under :attr:`LifecycleStage.WORKERS` is always started after the
database and stopped before it. Subsystems attach themselves with
:meth:`AyrisApp.add_component` rather than being wired in here, so tasks 05
(workers), 19 (actions), 43 (GUI) and 69 (plugins) extend the lifecycle without
touching this file.

Paths come before logging even though the specification lists the logger first:
the log directory lives inside the profile, so there is nowhere to write until
the root is resolved. Everything between those two points either raises
:class:`~ayris.core.errors.ConfigError` with a Russian message or writes to
stderr.

Qt is deliberately absent. The event bus is created here and bound to whichever
thread calls :meth:`AyrisApp.startup`; the entry point installs the wake-up that
drains it inside the Qt event loop.
"""

from __future__ import annotations

import ctypes
import faulthandler
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Final, Self, TypeVar

from ayris import __app_name__, __version__
from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.config import ConfigManager, RestartScope, init_config, reset_config_manager
from ayris.core.database import Database, init_database, reset_database
from ayris.core.errors import AyrisError
from ayris.core.events import ConfigChanged, EventBus, NotificationRequested
from ayris.core.migrations import apply_migrations
from ayris.core.paths import AppPaths, init_paths, reset_paths
from ayris.core.repositories import Repositories
from ayris.core.state import StateMachine
from ayris.utils.logger import get_logger, setup_logging, shutdown_logging

if TYPE_CHECKING:
    from typing import TextIO

    from ayris.core.config import Settings
    from ayris.core.models import Profile

__all__ = [
    "FAULT_LOG_NAME",
    "MUTEX_NAME",
    "STOP_TIMEOUT",
    "AlreadyRunningError",
    "AppOptions",
    "AyrisApp",
    "Component",
    "LifecycleStage",
    "RestartHandler",
    "signal_existing_instance",
]

_log = get_logger(__name__)

T = TypeVar("T")

#: How long a component gets to stop before it is killed.
STOP_TIMEOUT: Final = 5.0

#: Session-local, so two Windows users may each run their own Ayris. A
#: ``Global\`` prefix would make the second one fail instead.
MUTEX_NAME: Final = "AyrisSingleInstanceMutex"

#: Registered window message the second launch broadcasts to raise the first
#: instance's window. The first instance answers it from task 43 onwards.
SHOW_WINDOW_MESSAGE: Final = "AyrisShowWindow"

#: Where faulthandler writes a native crash dump, inside the logs directory.
FAULT_LOG_NAME: Final = "crash.log"

#: Lock file used where there is no Windows mutex, i.e. the test suite.
LOCK_FILE_NAME: Final = "ayris.lock"

#: Name of the profile created on first run.
DEFAULT_PROFILE_NAME: Final = "По умолчанию"

_ERROR_ALREADY_EXISTS: Final = 183
_HWND_BROADCAST: Final = 0xFFFF

#: Signatures of the hooks saved and restored around startup.
ExceptHook = Callable[[type[BaseException], BaseException, TracebackType | None], None]
ThreadExceptHook = Callable[[threading.ExceptHookArgs], None]


class AlreadyRunningError(AyrisError):
    """Another instance holds the lock. Raised only when the user asked for one.

    Defined here rather than in :mod:`ayris.core.errors` because nothing outside
    the lifecycle can produce it.
    """

    default_user_message = "Ayris уже запущен. Окно открыто из значка в трее."

    def __init__(self, technical: str = "another instance is already running") -> None:
        super().__init__(technical, recoverable=False)


class LifecycleStage(StrEnum):
    """Startup order. Shutdown is exactly this, reversed.

    Declaration order is the contract; ``tuple(LifecycleStage)`` preserves it.
    """

    PATHS = "paths"
    LOGGING = "logging"
    CONFIG = "config"
    SINGLE_INSTANCE = "single_instance"
    DATABASE = "database"
    MIGRATIONS = "migrations"
    PROFILE = "profile"
    EVENT_BUS = "event_bus"
    STATE = "state"
    ACTIONS = "actions"
    WORKERS = "workers"
    NLU = "nlu"
    GUI = "gui"
    PLUGINS = "plugins"


#: Called when a settings change needs a worker recycled. Receives the new
#: settings; anything it raises is logged and the scope stays pending.
RestartHandler = Callable[["Settings"], None]


@dataclass(frozen=True, slots=True)
class Component:
    """A subsystem attached to a lifecycle stage.

    Args:
        name: Shown in the startup and shutdown log lines.
        stage: Where in the order it belongs.
        start: Called during startup. May raise; the failure aborts startup and
            unwinds whatever already came up.
        stop: Called during shutdown, with ``stop_timeout`` to finish.
        kill: Called when ``stop`` ran out of time. A worker process kills its
            child here; anything without a forcible path leaves it ``None``.
        stop_timeout: Seconds allowed for ``stop``. ``0`` waits forever, which is
            only sensible for something that cannot block.
    """

    name: str
    stage: LifecycleStage
    start: Callable[[], None] | None = None
    stop: Callable[[], None] | None = None
    kill: Callable[[], None] | None = None
    stop_timeout: float = STOP_TIMEOUT


@dataclass(frozen=True, slots=True)
class AppOptions:
    """What the entry point knows before anything is initialised.

    Mirrors the command line. The two ``None`` fields mean "no opinion": the
    setting wins unless a flag overrode it, which is how ``--log-level`` can
    beat ``devtools.log_level`` without permanently changing it.
    """

    profile: Path | None = None
    portable: bool = False
    log_level: str | None = None
    console_log: bool | None = None
    minimized: bool = False
    watch_config: bool = True
    single_instance: bool | None = None


class _SingleInstanceGuard:
    """Holds the system-wide token that says "Ayris is running".

    On Windows this is a named mutex, which the OS releases even if the process
    is killed. Everywhere else — that is, in the test suite — it is an
    :func:`~fcntl.flock` on a file inside the profile. ``flock`` locks belong to
    the open file description rather than to the process, so a second
    :class:`AyrisApp` built inside the same interpreter is refused just like a
    second process would be.
    """

    __slots__ = ("_handle", "_lock_file", "_lock_path")

    def __init__(self, lock_path: Path) -> None:
        self._lock_path = lock_path
        self._handle: int | None = None
        self._lock_file: TextIO | None = None

    @property
    def acquired(self) -> bool:
        """Whether this guard currently owns the token."""
        return self._handle is not None or self._lock_file is not None

    def acquire(self) -> bool:
        """Take the token.

        Returns:
            ``False`` when another instance already holds it.
        """
        through_mutex = self._acquire_mutex()
        if through_mutex is not None:
            return through_mutex
        return self._acquire_lock_file()

    def release(self) -> None:
        """Give the token back. Safe to call when it was never taken."""
        handle, self._handle = self._handle, None
        if handle is not None:
            kernel32 = _win_dll("kernel32")
            if kernel32 is not None:
                kernel32.CloseHandle(ctypes.c_void_p(handle))

        lock_file, self._lock_file = self._lock_file, None
        if lock_file is not None:
            self._release_lock_file(lock_file)

    def _acquire_mutex(self) -> bool | None:
        """Windows named mutex.

        Returns:
            ``None`` when there is no Windows API to use, which sends the caller
            to the lock file instead.
        """
        kernel32 = _win_dll("kernel32")
        if kernel32 is None:
            return None
        try:
            kernel32.CreateMutexW.restype = ctypes.c_void_p
            kernel32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p)
            handle = kernel32.CreateMutexW(None, True, MUTEX_NAME)
            # Read through ctypes rather than kernel32.GetLastError: the DLL was
            # opened with use_last_error, so this is the value from that call and
            # not from whatever ran in between.
            last_error = int(ctypes.get_last_error())  # type: ignore[attr-defined,unused-ignore]
        except (AttributeError, OSError):
            _log.warning("не удалось создать мьютекс единственного экземпляра")
            return None

        if not handle:
            _log.warning("CreateMutexW вернул пустой дескриптор (код %d)", last_error)
            return None
        if last_error == _ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
            return False
        self._handle = int(handle)
        return True

    def _acquire_lock_file(self) -> bool:
        """Advisory lock on a file in the profile. Used off Windows."""
        # Written as a platform guard so mypy (which type-checks with
        # platform = "win32") skips a branch that cannot run there.
        if sys.platform != "win32":
            import fcntl

            self._lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = self._lock_path.open("w", encoding="utf-8")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                handle.close()
                return False
            handle.write(str(os.getpid()))
            handle.flush()
            self._lock_file = handle
            return True
        return True

    def _release_lock_file(self, handle: TextIO) -> None:
        if sys.platform != "win32":
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                _log.debug("не удалось снять блокировку %s", self._lock_path)
            handle.close()
            self._lock_path.unlink(missing_ok=True)


def _win_dll(name: str) -> Any | None:
    """Open a Windows DLL with last-error tracking, or ``None`` off Windows."""
    factory = getattr(ctypes, "WinDLL", None)
    if factory is None:
        return None
    try:
        return factory(name, use_last_error=True)
    except OSError:
        return None


def signal_existing_instance() -> bool:
    """Ask an already running Ayris to show its window.

    Broadcasts a registered window message; the main window subscribes to it from
    task 43 onwards. Until then the second launch simply exits quietly.

    Returns:
        Whether the message was posted.
    """
    user32 = _win_dll("user32")
    if user32 is None:
        return False
    try:
        message = int(user32.RegisterWindowMessageW(SHOW_WINDOW_MESSAGE))
        if not message:
            return False
        user32.PostMessageW(_HWND_BROADCAST, message, 0, 0)
    except (AttributeError, OSError):
        _log.debug("не удалось отправить сообщение работающему экземпляру")
        return False
    return True


class AyrisApp:
    """Every subsystem of Ayris, brought up and taken down as one unit.

    Args:
        options: What the command line decided. Defaults are what a plain
            ``python -m ayris`` gets.

    Typical use from the entry point::

        with AyrisApp(options).startup() as app:
            app.bus.set_wakeup(bridge.request_drain)
            return qt_application.exec()

    :meth:`shutdown` runs on the way out of the ``with`` block, and never raises:
    a subsystem that fails to stop is logged and the rest still get their turn.
    """

    __slots__ = (
        # Explicit, so the event bus can hold the application weakly: without it
        # __slots__ removes __weakref__ and every weak subscription fails.
        "__weakref__",
        "_bus",
        "_components",
        "_config",
        "_database",
        "_fault_log",
        "_guard",
        "_options",
        "_paths",
        "_pending_restarts",
        "_previous_excepthook",
        "_previous_thread_excepthook",
        "_profile",
        "_repositories",
        "_restart_handlers",
        "_running",
        "_stages_started",
        "_started_components",
        "_state",
        "_unsubscribe_config",
        "_unsubscribe_event",
    )

    def __init__(self, options: AppOptions | None = None) -> None:
        self._options = options if options is not None else AppOptions()
        self._components: list[Component] = []
        self._started_components: list[Component] = []
        self._stages_started: list[LifecycleStage] = []
        self._restart_handlers: dict[RestartScope, list[RestartHandler]] = {}
        self._pending_restarts: set[RestartScope] = set()
        self._running = False

        # The bus exists before startup so components can subscribe while they
        # are being registered; startup only binds it to the delivery thread.
        self._bus = EventBus()
        self._unsubscribe_event = self._bus.subscribe(ConfigChanged, self._on_config_event)

        self._paths: AppPaths | None = None
        self._config: ConfigManager | None = None
        self._database: Database | None = None
        self._repositories: Repositories | None = None
        self._state: StateMachine | None = None
        self._profile: Profile | None = None
        self._guard: _SingleInstanceGuard | None = None
        self._fault_log: TextIO | None = None
        self._unsubscribe_config: Callable[[], None] | None = None
        self._previous_excepthook: ExceptHook | None = None
        self._previous_thread_excepthook: ThreadExceptHook | None = None

    # ------------------------------------------------------------------
    # container
    # ------------------------------------------------------------------

    @property
    def options(self) -> AppOptions:
        return self._options

    @property
    def bus(self) -> EventBus:
        """The event bus. Available before startup, so wiring can happen early."""
        return self._bus

    @property
    def running(self) -> bool:
        return self._running

    @property
    def stages_started(self) -> tuple[LifecycleStage, ...]:
        """Stages that came up, in the order they did. Shutdown reverses it."""
        return tuple(self._stages_started)

    @property
    def paths(self) -> AppPaths:
        """Profile layout.

        Raises:
            AyrisError: Startup has not reached :attr:`LifecycleStage.PATHS`.
        """
        return _require(self._paths, "пути профиля")

    @property
    def config(self) -> ConfigManager:
        """Settings manager. Use :attr:`settings` to read a value."""
        return _require(self._config, "конфигурация")

    @property
    def settings(self) -> Settings:
        """Current settings."""
        return self.config.settings

    @property
    def database(self) -> Database:
        return _require(self._database, "база данных")

    @property
    def repositories(self) -> Repositories:
        """Storage for commands, history, variables and the rest."""
        return _require(self._repositories, "репозитории")

    @property
    def state(self) -> StateMachine:
        """Assistant state, microphone mode and connectivity."""
        return _require(self._state, "состояние помощника")

    @property
    def profile(self) -> Profile:
        """The active profile."""
        return _require(self._profile, "активный профиль")

    @property
    def pending_restarts(self) -> frozenset[RestartScope]:
        """Workers a settings change asked for, that nothing has restarted yet."""
        return frozenset(self._pending_restarts)

    def add_component(self, component: Component) -> None:
        """Attach a subsystem to its lifecycle stage.

        Registration order decides the order within a stage; shutdown reverses
        it. A component added while the application is already past its stage is
        started immediately, which is what a plugin loaded at runtime needs.
        """
        self._components.append(component)
        if self._running and component.stage in self._stages_started:
            self._start_component(component)

    def register_restart_handler(
        self,
        scope: RestartScope,
        handler: RestartHandler,
    ) -> Callable[[], None]:
        """Register what to do when a settings change needs ``scope`` recycled.

        Task 05 registers one per worker. Until then a change that needs a
        restart is only logged and left in :attr:`pending_restarts`.

        Returns:
            A callable that removes the handler.
        """
        self._restart_handlers.setdefault(scope, []).append(handler)

        def unregister() -> None:
            handlers = self._restart_handlers.get(scope)
            if handlers is not None and handler in handlers:
                handlers.remove(handler)

        return unregister

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> Self:
        """Bring every stage up in order.

        The calling thread becomes the event bus delivery thread, so this must be
        the UI thread in the main process.

        Raises:
            AlreadyRunningError: Another instance holds the single-instance token.
            AyrisError: A stage failed. Whatever already came up is shut down
                again before the error propagates.
        """
        if self._running:
            raise AyrisError("startup() called twice", user_message="Ayris уже запущен.")
        self._running = True
        try:
            for stage in LifecycleStage:
                self._start_stage(stage)
        except BaseException:
            _log.error("запуск прерван на этапе %s", self._current_stage())
            self.shutdown()
            raise
        _log.info("Ayris запущен: %s", ", ".join(stage.value for stage in self._stages_started))
        return self

    def shutdown(self) -> None:
        """Take every started stage down in reverse order. Never raises."""
        if not self._running and not self._stages_started:
            return
        self._running = False
        _log.info("завершение работы Ayris")

        for stage in reversed(self._stages_started):
            _log.debug("этап %s: остановка", stage.value)
            for component in self._components_of(stage):
                self._stop_component(component)
            stopper = _STAGE_STOPPERS.get(stage)
            if stopper is not None:
                self._guarded(f"этап {stage.value}", partial(stopper, self))

        self._started_components.clear()
        self._stages_started.clear()
        _log.info("Ayris остановлен")
        self._guarded("журнал", shutdown_logging)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------

    def _current_stage(self) -> str:
        return self._stages_started[-1].value if self._stages_started else "—"

    def _components_of(self, stage: LifecycleStage) -> list[Component]:
        """Started components of ``stage``, newest first — shutdown order."""
        return [item for item in reversed(self._started_components) if item.stage is stage]

    def _start_stage(self, stage: LifecycleStage) -> None:
        # Logged per stage so the startup order is readable in the log file
        # rather than only summarised at the end.
        _log.debug("этап %s: запуск", stage.value)
        starter = _STAGE_STARTERS.get(stage)
        if starter is not None:
            starter(self)
        self._stages_started.append(stage)
        for component in self._components:
            if component.stage is stage:
                self._start_component(component)

    def _start_component(self, component: Component) -> None:
        _log.debug("запуск подсистемы %s (%s)", component.name, component.stage.value)
        if component.start is not None:
            component.start()
        self._started_components.append(component)

    def _stop_component(self, component: Component) -> None:
        """Stop one component, killing it if it overruns its timeout."""
        if component in self._started_components:
            self._started_components.remove(component)
        if component.stop is None:
            return
        _log.debug("остановка подсистемы %s", component.name)

        if component.stop_timeout <= 0:
            self._guarded(component.name, component.stop)
            return

        worker = threading.Thread(
            target=self._guarded,
            args=(component.name, component.stop),
            name=f"ayris-stop-{component.name}",
            daemon=True,
        )
        worker.start()
        worker.join(component.stop_timeout)
        if not worker.is_alive():
            return

        _log.error(
            "подсистема %s не остановилась за %.1f с, снимаем принудительно",
            component.name,
            component.stop_timeout,
        )
        if component.kill is not None:
            self._guarded(f"{component.name} (kill)", component.kill)

    @staticmethod
    def _guarded(what: str, action: Callable[[], None]) -> None:
        """Run a teardown step, logging instead of letting it abort the rest."""
        try:
            action()
        except Exception:
            _log.exception("ошибка при остановке: %s", what)

    # -- paths ---------------------------------------------------------

    def _start_paths(self) -> None:
        self._paths = init_paths(portable=self._options.portable, profile=self._options.profile)

    def _stop_paths(self) -> None:
        reset_paths()
        self._paths = None

    # -- logging -------------------------------------------------------

    def _start_logging(self) -> None:
        paths = self.paths
        setup_logging(
            self._options.log_level or "INFO",
            console=self._options.console_log is not False,
            log_dir=paths.logs_dir,
        )
        _log.info("%s %s запускается", __app_name__, __version__)
        _log.info("python %s, платформа %s", sys.version.split()[0], sys.platform)
        _log.info("профиль: %s", paths.root)
        self._enable_faulthandler(paths)
        self._install_exception_hooks()

    def _stop_logging(self) -> None:
        self._restore_exception_hooks()
        self._disable_faulthandler()

    def _enable_faulthandler(self, paths: AppPaths) -> None:
        """Dump native tracebacks to a file — a C-level crash leaves no traceback.

        Audio, STT and TTS all run native code; without this a segfault inside a
        model would end the process with nothing in the log at all.
        """
        try:
            handle = (paths.logs_dir / FAULT_LOG_NAME).open("a", encoding="utf-8")
            faulthandler.enable(file=handle, all_threads=True)
        except (OSError, RuntimeError, ValueError):
            _log.warning("faulthandler не включён, аварийные дампы писаться не будут")
            return
        self._fault_log = handle

    def _disable_faulthandler(self) -> None:
        handle, self._fault_log = self._fault_log, None
        if handle is None:
            return
        # Disable before closing: faulthandler keeps the descriptor.
        faulthandler.disable()
        handle.close()

    def _install_exception_hooks(self) -> None:
        self._previous_excepthook = sys.excepthook
        self._previous_thread_excepthook = threading.excepthook
        sys.excepthook = self._handle_exception
        threading.excepthook = self._handle_thread_exception

    def _restore_exception_hooks(self) -> None:
        if self._previous_excepthook is not None:
            sys.excepthook = self._previous_excepthook
            self._previous_excepthook = None
        if self._previous_thread_excepthook is not None:
            threading.excepthook = self._previous_thread_excepthook
            self._previous_thread_excepthook = None

    def _handle_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        """Last stop for an unhandled exception on any non-worker thread.

        The windowed build has no console, so without this the traceback would be
        lost entirely and Ayris would appear to freeze.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        _log.critical("необработанное исключение", exc_info=(exc_type, exc_value, exc_tb))
        self._notify_crash(exc_value)

    def _handle_thread_exception(self, args: threading.ExceptHookArgs) -> None:
        """Same, for an exception that escaped a thread's ``run``."""
        if args.exc_type is SystemExit or args.exc_value is None:
            return
        _log.critical(
            "необработанное исключение в потоке %s",
            args.thread.name if args.thread is not None else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        self._notify_crash(args.exc_value)

    def _notify_crash(self, exc: BaseException) -> None:
        """Tell the user something broke, without letting the telling break too."""
        if isinstance(exc, AyrisError):
            message = exc.user_message
        else:
            message = str(exc) or type(exc).__name__
        try:
            self._bus.publish(
                NotificationRequested(
                    title="Внутренняя ошибка Ayris",
                    message=message,
                    level="error",
                )
            )
        except Exception:
            _log.exception("не удалось показать уведомление об ошибке")

    # -- configuration -------------------------------------------------

    def _start_config(self) -> None:
        self._config = init_config(self.paths.config_file, watch=self._options.watch_config)
        self._unsubscribe_config = self._config.subscribe(self._on_settings_changed)
        self._apply_live_settings(self._config.settings, initial=True)

    def _stop_config(self) -> None:
        if self._unsubscribe_config is not None:
            self._unsubscribe_config()
            self._unsubscribe_config = None
        reset_config_manager()
        self._config = None

    def _on_settings_changed(self, diff: SettingsDiff) -> None:
        """Config listener. Runs on the watcher thread, so it only publishes."""
        self._bus.publish(ConfigChanged(diff=diff))

    def _on_config_event(self, event: ConfigChanged) -> None:
        """Apply a settings change on the delivery thread."""
        diff = event.diff
        _log.info("настройки изменились: %s", diff.summary())
        self._apply_live_settings(diff.settings)
        scopes = diff.restart_scopes - {RestartScope.NONE}
        if scopes:
            self._pending_restarts.update(scopes)
            self._run_restart_handlers(diff.settings)

    def _apply_live_settings(self, settings: Settings, *, initial: bool = False) -> None:
        """Adopt the settings that take effect without restarting anything."""
        if self._options.log_level is None:
            setup_logging(
                settings.devtools.log_level,
                console=(
                    settings.devtools.console_log
                    if self._options.console_log is None
                    else self._options.console_log
                ),
                log_dir=self.paths.logs_dir,
            )
        if self._state is not None:
            self._state.apply_settings(settings)
        if not initial and self._repositories is not None:
            self._guarded(
                "очистка истории",
                partial(self._trim_history, settings.privacy.history_limit),
            )

    def _run_restart_handlers(self, settings: Settings) -> None:
        """Recycle whatever a settings change invalidated.

        A scope with no handler stays in :attr:`pending_restarts`, which the
        settings window reads to show «требуется перезапуск».
        """
        for scope in sorted(self._pending_restarts):
            handlers = self._restart_handlers.get(scope)
            if not handlers:
                _log.info("перезапуск %s отложен: некому обработать", scope.value)
                continue
            failed = False
            for handler in handlers:
                try:
                    handler(settings)
                except Exception:
                    failed = True
                    _log.exception("не удалось перезапустить %s", scope.value)
            if failed:
                continue
            self._pending_restarts.discard(scope)
            if self._config is not None:
                self._config.acknowledge_restart(scope)
            _log.info("перезапущено: %s", scope.value)

    # -- single instance -----------------------------------------------

    def _start_single_instance(self) -> None:
        wanted = self._options.single_instance
        if wanted is None:
            wanted = self.settings.general.single_instance
        if not wanted:
            _log.info("защита от второго экземпляра выключена")
            return

        guard = _SingleInstanceGuard(self.paths.root / LOCK_FILE_NAME)
        if not guard.acquire():
            _log.warning("Ayris уже запущен, показываем окно работающего экземпляра")
            signal_existing_instance()
            raise AlreadyRunningError
        self._guard = guard

    def _stop_single_instance(self) -> None:
        if self._guard is not None:
            self._guard.release()
            self._guard = None

    # -- database ------------------------------------------------------

    def _start_database(self) -> None:
        # migrate=False: migrations are their own stage, so the log shows them
        # separately and a schema failure is distinguishable from a locked file.
        self._database = init_database(self.paths.database_file, migrate=False)
        _log.info("база данных: %s", self._database.path)

    def _stop_database(self) -> None:
        if self._database is None:
            return
        # TRUNCATE folds the write-ahead log back into the file, so a copied
        # profile directory is complete without the -wal sidecar.
        self._guarded("контрольная точка БД", self._database.checkpoint)
        reset_database()
        self._database = None

    def _start_migrations(self) -> None:
        version = apply_migrations(self.database)
        _log.info("схема базы данных: версия %d", version)

    # -- profile -------------------------------------------------------

    def _start_profile(self) -> None:
        repositories = Repositories(self.database)
        profile = repositories.profiles.active()
        if profile is None:
            profile = repositories.profiles.create(DEFAULT_PROFILE_NAME, activate=True)
            _log.info("создан профиль «%s»", profile.name)
        self._repositories = repositories
        self._profile = profile
        _log.info("активный профиль: %s", profile.name)

        # Variables from a previous run that were never meant to outlive it. A
        # crash leaves them behind, so they are cleared on the way in as well as
        # on the way out.
        dropped = repositories.variables.clear_transient()
        if dropped:
            _log.debug("удалено временных переменных: %d", dropped)

    def _stop_profile(self) -> None:
        """Keep the persistent variables, drop the rest.

        Values written with ``persistent=True`` are already on disk — the work
        here is removing everything that was not.
        """
        if self._repositories is None:
            return
        dropped = self._repositories.variables.clear_transient()
        _log.info("сохранены постоянные переменные, удалено временных: %d", dropped)
        limit = self._config.settings.privacy.history_limit if self._config is not None else 0
        if limit:
            self._guarded("очистка истории", partial(self._trim_history, limit))
        self._repositories = None
        self._profile = None

    def _trim_history(self, limit: int) -> None:
        if self._repositories is not None:
            self._repositories.maintenance.apply_retention(history_limit=limit)

    # -- event bus and state -------------------------------------------

    def _start_event_bus(self) -> None:
        # Whoever starts the application owns the UI thread; from here on every
        # handler runs there, whatever thread published the event.
        self._bus.bind_to_thread()
        _log.debug("шина событий привязана к потоку %s", self._bus.thread_id)

    def _stop_event_bus(self) -> None:
        # One last pass so a shutdown notification published by a component that
        # already stopped still reaches the tray before it goes away.
        delivered = self._bus.drain(limit=0)
        if delivered:
            _log.debug("доставлено событий при остановке: %d", delivered)
        self._bus.set_wakeup(None)
        if self._unsubscribe_event is not None:
            self._unsubscribe_event()
        self._bus.clear()

    def _start_state(self) -> None:
        self._state = StateMachine(self._bus)
        self._state.apply_settings(self.settings)

    def _stop_state(self) -> None:
        self._state = None

    def __repr__(self) -> str:
        return f"AyrisApp(running={self._running}, stages={len(self._stages_started)})"


#: Built-in work per stage. Components registered by the subsystems run after
#: the starter and before the stopper of their own stage.
_STAGE_STARTERS: Final[dict[LifecycleStage, Callable[[AyrisApp], None]]] = {
    LifecycleStage.PATHS: AyrisApp._start_paths,
    LifecycleStage.LOGGING: AyrisApp._start_logging,
    LifecycleStage.CONFIG: AyrisApp._start_config,
    LifecycleStage.SINGLE_INSTANCE: AyrisApp._start_single_instance,
    LifecycleStage.DATABASE: AyrisApp._start_database,
    LifecycleStage.MIGRATIONS: AyrisApp._start_migrations,
    LifecycleStage.PROFILE: AyrisApp._start_profile,
    LifecycleStage.EVENT_BUS: AyrisApp._start_event_bus,
    LifecycleStage.STATE: AyrisApp._start_state,
}

_STAGE_STOPPERS: Final[dict[LifecycleStage, Callable[[AyrisApp], None]]] = {
    LifecycleStage.STATE: AyrisApp._stop_state,
    LifecycleStage.EVENT_BUS: AyrisApp._stop_event_bus,
    LifecycleStage.PROFILE: AyrisApp._stop_profile,
    LifecycleStage.DATABASE: AyrisApp._stop_database,
    LifecycleStage.SINGLE_INSTANCE: AyrisApp._stop_single_instance,
    LifecycleStage.CONFIG: AyrisApp._stop_config,
    LifecycleStage.LOGGING: AyrisApp._stop_logging,
    LifecycleStage.PATHS: AyrisApp._stop_paths,
}


def _require(value: T | None, what: str) -> T:
    """Return ``value``, or explain which stage has not run yet."""
    if value is None:
        raise AyrisError(
            f"{what} is not initialised yet",
            user_message="Ayris ещё не завершил запуск.",
            recoverable=False,
        )
    return value
