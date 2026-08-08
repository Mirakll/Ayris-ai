"""Entry point: ``python -m ayris`` and the ``ayris`` console script.

Startup order matters and is not negotiable:

1. Parse arguments — a bad ``--profile`` must fail before anything touches disk.
2. Set per-monitor DPI awareness — Qt latches this when ``QApplication`` is
   constructed, so it cannot move any later.
3. Set the high-DPI rounding policy, also read at construction time.
4. Create ``QApplication``.
5. Build :class:`~ayris.core.app.AyrisApp` — paths, logging, config, database,
   migrations, profile, event bus, state machine — and start it. This is where
   the single-instance check happens: a second launch fails before any UI exists
   and the first instance is asked to show its window.
6. Install the Qt bridge: the bus wakes the UI thread through a queued signal,
   the UI thread drains the bus. From here on every event handler runs inside
   the Qt event loop.
7. Run the event loop. Shutdown happens in reverse, as one ``with`` block.

PySide6 is imported lazily inside :func:`_run_application` so that ``--version``
and argument errors do not pay for loading Qt.
"""

from __future__ import annotations

import argparse
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import FrameType, TracebackType

from ayris import __app_name__, __version__
from ayris.core.app import (
    AlreadyRunningError,
    AppOptions,
    AyrisApp,
    Component,
    LifecycleStage,
)
from ayris.core.errors import AyrisError
from ayris.core.paths import AppPaths, init_paths
from ayris.utils.dpi import enable_per_monitor_dpi_awareness
from ayris.utils.logger import LOG_LEVELS, get_logger, setup_logging, shutdown_logging

__all__ = ["CliOptions", "build_parser", "main", "parse_args"]

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_STARTUP_ERROR = 2
EXIT_ALREADY_RUNNING = 3

_SIGNAL_POLL_MS = 200

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CliOptions:
    """Validated command line options."""

    minimized: bool
    profile: Path | None
    log_level: str
    portable: bool
    no_console_log: bool
    log_level_from_cli: bool


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser. Exposed separately so tests can exercise it."""
    parser = argparse.ArgumentParser(
        prog="ayris",
        description="Ayris — голосовой помощник для Windows 11.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{__app_name__} {__version__}",
    )
    parser.add_argument(
        "--minimized",
        action="store_true",
        help="запустить свёрнутым, без показа окна настроек",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        metavar="PATH",
        help="путь к папке профиля вместо %%APPDATA%%\\Ayris",
    )
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="уровень подробности логов (по умолчанию берётся из настроек)",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="хранить профиль рядом с исполняемым файлом",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="не дублировать логи в консоль",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CliOptions:
    """Parse ``argv`` into typed options.

    Raises:
        SystemExit: argparse rejected the arguments or handled ``--version``.
    """
    parser = build_parser()
    namespace = parser.parse_args(argv)

    profile: Path | None = namespace.profile
    portable = bool(namespace.portable)
    if profile is not None and portable:
        parser.error("--profile и --portable нельзя использовать одновременно")

    # Whether the user actually passed --log-level. When they did not, the
    # setting from config wins once it is loaded; when they did, the command
    # line beats the file for this run only.
    given = list(argv) if argv is not None else sys.argv[1:]
    log_level_from_cli = "--log-level" in given

    return CliOptions(
        minimized=bool(namespace.minimized),
        profile=profile,
        log_level=str(namespace.log_level),
        portable=portable,
        no_console_log=bool(namespace.no_console_log),
        log_level_from_cli=log_level_from_cli,
    )


def _install_excepthook() -> None:
    """Route unhandled exceptions into the log instead of a silent crash.

    In the windowed build there is no console, so an uncaught traceback would
    otherwise vanish entirely. :class:`AyrisApp` installs its own, richer hook
    at startup; this one is the safety net for the boot sequence itself.
    """

    def hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        _log.critical("unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = hook


def _bootstrap(options: CliOptions) -> AppPaths:
    """Resolve paths and configure logging before Qt exists."""
    paths = init_paths(portable=options.portable, profile=options.profile)
    setup_logging(options.log_level, console=not options.no_console_log, log_dir=paths.logs_dir)
    _install_excepthook()

    _log.info("%s %s starting", __app_name__, __version__)
    _log.info("python %s on %s", sys.version.split()[0], sys.platform)
    _log.info("profile root: %s", paths.root)
    return paths


def _install_signal_handlers(quit_callback: Callable[[], None]) -> None:
    """Make Ctrl+C and SIGTERM quit the application."""

    def handler(signum: int, _frame: FrameType | None) -> None:
        _log.info("received signal %d, shutting down", signum)
        quit_callback()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


class _QtBridge:
    """Move queued bus events onto the Qt event loop.

    A worker thread publishes an event; the bus calls :meth:`request_drain`,
    which emits a signal. Qt guarantees a queued connection is delivered on the
    thread that owns the receiver — the UI thread — and the slot drains the bus
    there. The bus itself never imports Qt; only this class touches both.
    """

    def __init__(self, app: AyrisApp) -> None:
        from PySide6.QtCore import QObject, Qt, Signal

        bus = app.bus

        class _Waker(QObject):
            """Owned by the UI thread, so its queued slot runs there."""

            fired = Signal()

            def __init__(self) -> None:
                super().__init__()
                self.fired.connect(self._drain, Qt.ConnectionType.QueuedConnection)

            def _drain(self) -> None:
                delivered = bus.drain()
                # drain() stops at DRAIN_BATCH so a burst cannot freeze the UI;
                # the bus re-arms the wake-up itself when it left events behind.
                if delivered:
                    _log.log(5, "delivered %d event(s) on the UI thread", delivered)

        self._waker = _Waker()
        self._bus = bus
        bus.set_wakeup(self.request_drain)

    def request_drain(self) -> None:
        """Called from any thread; returns immediately."""
        self._waker.fired.emit()

    def disconnect(self) -> None:
        """Stop waking the UI thread. Called before the event loop is gone."""
        self._bus.set_wakeup(None)
        self._waker.fired.disconnect()


def _run_application(options: CliOptions) -> int:
    """Create the Qt application and run the event loop."""
    dpi_result = enable_per_monitor_dpi_awareness()
    _log.info("dpi awareness: %s (per-monitor: %s)", dpi_result.value, dpi_result.is_per_monitor)

    # Imported here so that --version and argument errors do not load Qt.
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from ayris.gui.main_window import MainWindow

    # PassThrough keeps fractional scaling (125%, 150%) instead of rounding it,
    # which the overlay needs to stay pixel-aligned. Must precede QApplication.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Only the program name is handed to Qt; our own flags are already parsed.
    app = QApplication(sys.argv[:1])
    app.setApplicationName(__app_name__)
    app.setApplicationDisplayName(__app_name__)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(__app_name__)
    # Task 44 adds the tray icon; until then the window is the only way to quit,
    # so the process must exit when it closes.
    app.setQuitOnLastWindowClosed(True)

    # Everything below the window belongs to the lifecycle. Startup binds the
    # event bus to this thread, so a handler that publishes from the UI thread
    # is delivered inline and one published from a worker is queued.
    ayris = AyrisApp(
        AppOptions(
            profile=options.profile,
            portable=options.portable,
            log_level=options.log_level if options.log_level_from_cli else None,
            console_log=False if options.no_console_log else None,
            minimized=options.minimized,
        )
    )
    try:
        ayris.startup()
    except AlreadyRunningError as exc:
        # The running instance was already asked to show its window.
        _log.info("%s", exc.technical)
        sys.stderr.write(f"{exc.user_message}\n")
        return EXIT_ALREADY_RUNNING

    with ayris:
        bridge = _QtBridge(ayris)
        window = MainWindow()

        def close_window() -> None:
            # QWidget.close returns a bool; the lifecycle wants None.
            window.close()

        ayris.add_component(
            Component(
                name="main_window",
                stage=LifecycleStage.GUI,
                stop=close_window,
            )
        )

        if options.minimized:
            _log.info(
                "started minimized; no tray icon exists yet (task 44), "
                "so the only way out is Ctrl+C or terminating the process"
            )
        else:
            window.show()

        _install_signal_handlers(app.quit)
        # Python runs signal handlers only between bytecode instructions, and
        # QApplication.exec blocks inside C++. A periodic no-op timer hands
        # control back to the interpreter often enough for Ctrl+C to be noticed.
        signal_timer = QTimer(app)
        signal_timer.timeout.connect(lambda: None)
        signal_timer.start(_SIGNAL_POLL_MS)

        _log.info("entering Qt event loop")
        try:
            exit_code = app.exec()
        finally:
            bridge.disconnect()

    _log.info("Qt event loop finished with code %d", exit_code)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    """Process entry point. Returns the process exit code."""
    options = parse_args(argv)

    try:
        paths = _bootstrap(options)
    except AyrisError as exc:
        # Logging may not exist yet, so this one message also goes to stderr.
        sys.stderr.write(f"{exc.user_message}\n")
        return EXIT_STARTUP_ERROR

    try:
        return _run_application(options)
    except KeyboardInterrupt:
        _log.info("interrupted by user")
        return EXIT_OK
    except AyrisError as exc:
        _log.critical("fatal: %s", exc.technical, exc_info=exc)
        sys.stderr.write(f"{exc.user_message}\n")
        return EXIT_FAILURE
    except Exception:
        _log.critical("unexpected fatal error in profile %s", paths.root, exc_info=True)
        return EXIT_FAILURE
    finally:
        shutdown_logging()


if __name__ == "__main__":
    sys.exit(main())
