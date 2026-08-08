"""Per-monitor DPI awareness for Windows.

Qt reads the process DPI awareness at ``QGuiApplication`` construction time, so
:func:`enable_per_monitor_dpi_awareness` must run *before* the application
object exists. Without it the overlay is bitmap-stretched (blurry) when dragged
between monitors with different scaling — see section 18 of the specification.

Three APIs exist, newest first:

* ``SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)`` — Windows 10 1703+.
  The only one that also scales non-client areas and sends ``WM_DPICHANGED``.
* ``SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)`` — Windows 8.1+.
* ``SetProcessDPIAware()`` — system-wide awareness only, last resort.

All three fail if awareness is already set (by a manifest, or because Python was
started by a host that set it). That is a success for our purposes, not an error.
``ctypes.windll`` is reached through :func:`getattr` because the attribute does
not exist on non-Windows platforms, where the test suite and linters run.
"""

from __future__ import annotations

import ctypes
import sys
from enum import Enum
from typing import Any, Final

from ayris.utils.logger import get_logger

__all__ = ["DpiAwarenessResult", "enable_per_monitor_dpi_awareness"]

_log = get_logger(__name__)

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, passed as a handle-sized value.
_PER_MONITOR_AWARE_V2: Final = -4
# PROCESS_DPI_AWARENESS.PROCESS_PER_MONITOR_DPI_AWARE
_PROCESS_PER_MONITOR_DPI_AWARE: Final = 2
# E_ACCESSDENIED — awareness already set for this process.
_E_ACCESSDENIED: Final = -2147024891


class DpiAwarenessResult(Enum):
    """Which API ended up providing DPI awareness."""

    PER_MONITOR_V2 = "per_monitor_v2"
    PER_MONITOR = "per_monitor"
    SYSTEM = "system"
    ALREADY_SET = "already_set"
    UNSUPPORTED = "unsupported"

    @property
    def is_per_monitor(self) -> bool:
        """True when the overlay will re-scale correctly across monitors."""
        return self in (
            DpiAwarenessResult.PER_MONITOR_V2,
            DpiAwarenessResult.PER_MONITOR,
            DpiAwarenessResult.ALREADY_SET,
        )


def _win_function(library: str, function: str) -> Any | None:
    """Look up a WinAPI entry point, or ``None`` when it is unavailable."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        return getattr(getattr(windll, library), function)
    except (AttributeError, OSError):
        return None


def _try_context_v2() -> bool:
    """``user32.SetProcessDpiAwarenessContext`` — Windows 10 1703 and newer."""
    set_context = _win_function("user32", "SetProcessDpiAwarenessContext")
    if set_context is None:
        return False
    set_context.restype = ctypes.c_bool
    set_context.argtypes = [ctypes.c_void_p]
    return bool(set_context(ctypes.c_void_p(_PER_MONITOR_AWARE_V2)))


def _try_shcore() -> tuple[bool, bool]:
    """``shcore.SetProcessDpiAwareness`` — Windows 8.1 and newer.

    Returns:
        ``(succeeded, already_set)``.
    """
    set_awareness = _win_function("shcore", "SetProcessDpiAwareness")
    if set_awareness is None:
        return False, False
    set_awareness.restype = ctypes.c_long
    set_awareness.argtypes = [ctypes.c_int]
    hresult = int(set_awareness(_PROCESS_PER_MONITOR_DPI_AWARE))
    if hresult == 0:
        return True, False
    return False, hresult == _E_ACCESSDENIED


def _try_legacy() -> bool:
    """``user32.SetProcessDPIAware`` — system DPI awareness, Vista and newer."""
    set_aware = _win_function("user32", "SetProcessDPIAware")
    if set_aware is None:
        return False
    return bool(set_aware())


def enable_per_monitor_dpi_awareness() -> DpiAwarenessResult:
    """Make this process per-monitor DPI aware.

    Must be called before any ``QGuiApplication``/``QApplication`` is created.
    Never raises: on an unsupported platform the application simply runs with
    whatever awareness the OS assigned.
    """
    # sys.platform rather than os.name: mypy treats it as a platform guard and
    # will not report the other branch as unreachable under warn_unreachable.
    if sys.platform != "win32":
        _log.debug("DPI awareness skipped: not running on Windows")
        return DpiAwarenessResult.UNSUPPORTED

    if _try_context_v2():
        _log.debug("DPI awareness: per-monitor v2")
        return DpiAwarenessResult.PER_MONITOR_V2

    succeeded, already_set = _try_shcore()
    if succeeded:
        _log.debug("DPI awareness: per-monitor v1")
        return DpiAwarenessResult.PER_MONITOR
    if already_set:
        _log.debug("DPI awareness already set for this process")
        return DpiAwarenessResult.ALREADY_SET

    if _try_legacy():
        _log.warning(
            "DPI awareness: system-wide only; the overlay may look blurry on a "
            "secondary monitor with different scaling"
        )
        return DpiAwarenessResult.SYSTEM

    _log.warning("could not set DPI awareness through any Windows API")
    return DpiAwarenessResult.UNSUPPORTED
