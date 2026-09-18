"""How the overlay stays a helper window: on top, off the taskbar, no focus theft.

Two concerns live here, kept apart from each other and from Qt:

* the pure bit arithmetic on a window's extended style — testable without a
  window at all;
* a thin native backend that applies those bits and re-asserts topmost, isolated
  behind :class:`NativeWindow` so the controller can be driven by a fake.

The main overlay is *interactive*, so ``WS_EX_NOACTIVATE`` is never held
permanently: a programmatic show uses ``SW_SHOWNOACTIVATE`` instead, which keeps
the window from stealing focus while still letting a click focus it.
"""

from __future__ import annotations

import sys
from typing import Protocol

from ayris.utils import winapi
from ayris.utils.logger import get_logger

__all__ = [
    "NativeWindow",
    "WindowFlags",
    "native_window",
    "overlay_qt_flags",
    "tool_window_bits",
    "with_no_activate",
]

_log = get_logger(__name__)

# HWND_TOPMOST for SetWindowPos: place (and keep) the window above non-topmost
# windows. Re-asserting with SWP_NOACTIVATE|NOMOVE|NOSIZE neither moves the
# window nor takes focus, so it does not flicker.
_HWND_TOPMOST = -1


def overlay_qt_flags() -> int:
    """The Qt window-flag combination for the frameless, always-on-top panel."""
    from PySide6.QtCore import Qt

    return int(
        Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint
    )


def tool_window_bits(current: int) -> int:
    """Add ``WS_EX_TOOLWINDOW`` and clear ``WS_EX_APPWINDOW``.

    Together these keep the window out of the taskbar and the Alt+Tab list.
    """
    return (current | winapi.WS_EX_TOOLWINDOW) & ~winapi.WS_EX_APPWINDOW


def with_no_activate(current: int, *, enabled: bool) -> int:
    """Set or clear ``WS_EX_NOACTIVATE``. Only ever set transiently."""
    if enabled:
        return current | winapi.WS_EX_NOACTIVATE
    return current & ~winapi.WS_EX_NOACTIVATE


class NativeWindow(Protocol):
    """The minimal native surface :class:`WindowFlags` drives."""

    def ex_style(self) -> int: ...

    def set_ex_style(self, value: int) -> None: ...

    def assert_topmost(self) -> None: ...


class WindowFlags:
    """Apply the helper-window style and re-assert topmost, over a backend."""

    def __init__(self, native: NativeWindow | None) -> None:
        self._native = native

    @property
    def available(self) -> bool:
        return self._native is not None

    def apply_tool_window(self) -> None:
        """Make the window a tool window: no taskbar button, no Alt+Tab entry."""
        native = self._native
        if native is None:
            return
        try:
            native.set_ex_style(tool_window_bits(native.ex_style()))
        except OSError as exc:  # pragma: no cover - live WinAPI only
            _log.warning("не удалось выставить стиль tool-window: %s", exc)

    def set_no_activate(self, *, enabled: bool) -> None:
        native = self._native
        if native is None:
            return
        try:
            native.set_ex_style(with_no_activate(native.ex_style(), enabled=enabled))
        except OSError as exc:  # pragma: no cover - live WinAPI only
            _log.warning("не удалось изменить WS_EX_NOACTIVATE: %s", exc)

    def confirm_topmost(self) -> None:
        """Cheap, focus-safe re-assertion of the topmost z-order."""
        native = self._native
        if native is None:
            return
        try:
            native.assert_topmost()
        except OSError as exc:  # pragma: no cover - live WinAPI only
            _log.debug("подтверждение topmost не удалось: %s", exc)


class _Win32Window:
    """Live backend over ``user32``. Exercised on Windows and in CI, not here."""

    def __init__(self, hwnd: int) -> None:  # pragma: no cover - live WinAPI only
        self._hwnd = hwnd

    def ex_style(self) -> int:  # pragma: no cover - live WinAPI only
        return winapi.window_ex_style(self._hwnd)

    def set_ex_style(self, value: int) -> None:  # pragma: no cover - live WinAPI only
        import ctypes

        set_long = getattr(ctypes.windll.user32, "SetWindowLongPtrW", None)
        if set_long is None:
            set_long = ctypes.windll.user32.SetWindowLongW
        set_long.restype = ctypes.c_ssize_t
        set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        set_long(ctypes.c_void_p(self._hwnd), winapi.GWL_EXSTYLE, value)

    def assert_topmost(self) -> None:  # pragma: no cover - live WinAPI only
        import ctypes

        set_pos = ctypes.windll.user32.SetWindowPos
        set_pos.restype = ctypes.c_bool
        set_pos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        flags = winapi.SWP_NOMOVE | winapi.SWP_NOSIZE | winapi.SWP_NOACTIVATE
        set_pos(
            ctypes.c_void_p(self._hwnd),
            ctypes.c_void_p(_HWND_TOPMOST),
            0,
            0,
            0,
            0,
            flags,
        )


def native_window(hwnd: int) -> NativeWindow | None:
    """A live backend for ``hwnd`` on Windows, or ``None`` where WinAPI is absent."""
    if sys.platform != "win32" or not hwnd:
        return None
    return _Win32Window(hwnd)
