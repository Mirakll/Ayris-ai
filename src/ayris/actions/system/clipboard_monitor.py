"""The clipboard monitor: a listener window that turns copies into history.

Windows offers three ways to watch the clipboard, and only one of them is safe.
The old ``SetClipboardViewer`` chain breaks for everyone downstream the moment a
single program in it exits without repairing the chain, and polling
``GetClipboardSequenceNumber`` on a timer either misses a copy or wakes the CPU
forever. ``AddClipboardFormatListener`` is the supported one: Windows posts
``WM_CLIPBOARDUPDATE`` to a window, once per change, with no chain to maintain.

The catch is the word *window*. A listener needs an HWND and a message loop, and
Ayris' own loop belongs to Qt, on the GUI thread, where a clipboard read that
blocks for tens of milliseconds is felt. So this module runs its own
message-only window on its own thread — see :class:`ayris.utils.winapi.MessageWindow`
— and touches nothing but the repository.

Policy lives next door in :func:`ayris.actions.system.clipboard.record_clipboard`:
deduplication, limits, eviction that keeps pinned entries, and the refusal to
store anything a password manager marked. This module is the thread and the
window, and the decisions it makes on its own are only about the thread and the
window.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from ayris.actions.system.clipboard import (
    ClipboardBusy,
    clipboard_settings,
    get_clipboard,
    get_clipboard_store,
    record_clipboard,
    windows_history_enabled,
)
from ayris.core.errors import ActionError
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.system.clipboard import ClipboardBackend
    from ayris.core.config import ClipboardActionsConfig
    from ayris.core.repositories import ClipboardRepository

__all__ = [
    "ClipboardMonitor",
    "MonitorStats",
    "get_clipboard_monitor",
    "reset_clipboard_monitor",
    "set_clipboard_monitor",
    "start_clipboard_monitor",
    "stop_clipboard_monitor",
]

_log = get_logger(__name__)

#: Window class name. Unique per process id so two Ayris instances — a running one
#: and a developer's second copy — do not collide on ``RegisterClassW``.
_CLASS_PREFIX: Final = "AyrisClipboardListener"

#: How long :meth:`ClipboardMonitor.stop` waits for the thread to leave its pump.
#: Generous: the thread only has to notice one posted message, and if it somehow
#: cannot, hanging shutdown is worse than leaking a daemon thread.
_STOP_TIMEOUT_S: Final = 2.0


@dataclass(slots=True)
class MonitorStats:
    """Counters for the settings tab and for tests.

    Not just diagnostics: «I copied something and the history stayed empty» is the
    one complaint here that cannot be investigated by looking at the data, since
    the data is what is missing. ``skipped`` broken down by reason answers it.
    """

    received: int = 0
    stored: int = 0
    skipped: int = 0
    errors: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


class ClipboardMonitor:
    """Watches the clipboard on its own thread and writes text to the history.

    Start it once at application startup and stop it on exit; both are idempotent.
    ``handle_update`` is deliberately public and does not need the thread or the
    window at all, which is how everything interesting about the monitor gets
    tested on a machine with no clipboard.
    """

    def __init__(
        self,
        *,
        backend: ClipboardBackend | None = None,
        store: ClipboardRepository | None = None,
        settings: ClipboardActionsConfig | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._settings = settings
        self._window: winapi.MessageWindow | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self.stats = MonitorStats()

    # ---------------------------------------------------------------- state

    @property
    def running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _config(self) -> ClipboardActionsConfig:
        return self._settings if self._settings is not None else clipboard_settings()

    def _clipboard(self) -> ClipboardBackend:
        return self._backend if self._backend is not None else get_clipboard()

    def _repository(self) -> ClipboardRepository:
        return self._store if self._store is not None else get_clipboard_store()

    # -------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Begin listening. ``False`` if the setting is off or it already runs.

        Returns only after the window exists and the listener is registered, so a
        copy made right after ``start()`` returned is not lost to a race — and so a
        failure to register surfaces here instead of in a dead thread.
        """
        with self._lock:
            if self.running:
                return False
            if not self._config().monitor:
                _log.debug("clipboard monitor disabled by settings")
                return False
            if not winapi.available():
                _log.info("clipboard monitor unavailable: not Windows")
                return False

            self._ready.clear()
            thread = threading.Thread(target=self._run, name="ayris-clipboard-monitor", daemon=True)
            self._thread = thread
            thread.start()

        self._ready.wait(_STOP_TIMEOUT_S)
        if self._failure is not None:
            self._thread = None
            _log.warning("clipboard monitor failed to start: %s", self._failure)
            return False
        if self._config().warn_windows_history and windows_history_enabled():
            _log.info(
                "Windows clipboard history (Win+V) is on: copies are stored twice, "
                "and cloud sync may put them outside this machine"
            )
        _log.info("clipboard monitor started")
        return True

    def stop(self, timeout: float = _STOP_TIMEOUT_S) -> None:
        """Stop listening and join the thread. Safe to call when not running."""
        with self._lock:
            thread, window = self._thread, self._window
            self._thread = None
        if window is not None:
            window.stop()
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                _log.warning("clipboard monitor thread did not stop within %.1fs", timeout)
        _log.debug("clipboard monitor stopped")

    def _run(self) -> None:
        """Thread body: one window, one listener, one message loop.

        Everything WinAPI here happens on this thread on purpose. A window belongs
        to the thread that created it — messages are delivered to that thread's
        queue and nowhere else — so creating it in ``start()`` and pumping it here
        would deliver ``WM_CLIPBOARDUPDATE`` to whoever called ``start()``, which
        is the GUI thread, which is the thing this design exists to avoid.
        """
        window = winapi.MessageWindow(f"{_CLASS_PREFIX}_{os.getpid()}", self._on_message)
        listening = False
        # Cleared here rather than in start(): whatever happens below, this runs
        # before _ready is set, so the caller never reads a failure from last time.
        self._failure = None
        try:
            window.create()
            winapi.add_clipboard_format_listener(window.hwnd)
            listening = True
            self._window = window
        except winapi.WinApiError as error:
            self._failure = error
            _close_quietly(window)
            self._ready.set()
            return
        finally:
            self._ready.set()

        try:
            window.pump()
        except Exception as error:  # a dead listener must not be a silent one
            self.stats.errors += 1
            _log.exception("clipboard monitor loop crashed: %s", error)
        finally:
            self._window = None
            if listening:
                winapi.remove_clipboard_format_listener(window.hwnd)
            _close_quietly(window)

    # ---------------------------------------------------------------- messages

    def _on_message(self, code: int) -> None:
        if code == winapi.WM_CLIPBOARDUPDATE:
            self.handle_update()

    def handle_update(self) -> None:
        """One clipboard change: read it, decide, record it.

        Never raises. This runs inside a message loop, and an exception escaping
        here would take down the listener for the rest of the session — a monitor
        that quietly stops recording is worse than one that skips a single copy.
        """
        self.stats.received += 1
        try:
            snapshot = self._clipboard().read()
        except ClipboardBusy as error:
            self.stats.errors += 1
            _log.debug("clipboard busy on update, skipping this change: %s", error)
            return
        except ActionError as error:
            self.stats.errors += 1
            _log.warning("clipboard read failed on update: %s", error)
            return

        try:
            outcome = record_clipboard(snapshot, store=self._repository(), settings=self._config())
        except Exception as error:  # a broken database must not kill the listener
            self.stats.errors += 1
            _log.exception("failed to record a clipboard change: %s", error)
            return

        if outcome.stored:
            self.stats.stored += 1
            _log.debug("clipboard history entry %s stored", outcome.entry_id)
        else:
            self.stats.note(outcome.reason)
            _log.debug("clipboard change not stored: %s", outcome.reason)


def _close_quietly(window: winapi.MessageWindow) -> None:
    """Destroy the listener window, ignoring failures.

    Teardown is best effort by definition: the thread is already going away, and an
    exception out of ``DestroyWindow`` would replace a real error with a cosmetic
    one — or hide it, if the real one is what brought us here.
    """
    with contextlib.suppress(Exception):
        window.close()


# --------------------------------------------------------------------------- #
# Process-wide monitor
# --------------------------------------------------------------------------- #

_monitor: ClipboardMonitor | None = None
_monitor_lock = threading.Lock()


def get_clipboard_monitor() -> ClipboardMonitor:
    """The monitor for this process, created on first use but not started."""
    global _monitor
    with _monitor_lock:
        if _monitor is None:
            _monitor = ClipboardMonitor()
        return _monitor


def set_clipboard_monitor(monitor: ClipboardMonitor | None) -> None:
    """Install a monitor, or drop the cached one with ``None``. Test seam."""
    global _monitor
    with _monitor_lock:
        _monitor = monitor


def reset_clipboard_monitor() -> None:
    """Stop the cached monitor and forget it."""
    with _monitor_lock:
        monitor = _monitor
    if monitor is not None:
        monitor.stop()
    set_clipboard_monitor(None)


def start_clipboard_monitor() -> bool:
    """Start the process monitor. What application startup calls."""
    return get_clipboard_monitor().start()


def stop_clipboard_monitor() -> None:
    """Stop the process monitor. What shutdown calls."""
    with _monitor_lock:
        monitor = _monitor
    if monitor is not None:
        monitor.stop()
