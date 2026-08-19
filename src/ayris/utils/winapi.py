"""The WinAPI calls the system actions need, and nothing else.

This module is deliberately dumb: one Python function per Windows entry point,
arguments converted, result checked, :class:`WinApiError` raised on failure. All
the decisions — which window to focus, which show command a spoken «сверни»
means, what to do when the shell refuses to raise a window — live in
:mod:`ayris.actions.system`, where they can be tested on any platform against a
fake of this layer.

Three things this layer takes care of so callers do not have to:

* ``ctypes.windll`` is reached through :func:`getattr`, because the attribute
  does not exist on Linux, where the test suite and the linters run. Every entry
  point is looked up lazily for the same reason.
* ``ctypes.wintypes`` is never imported — on non-Windows it raises at import
  time. Handles are :class:`ctypes.c_void_p`, and every value that crosses back
  into Python is cast explicitly, so nothing leaks ``Any`` upwards.
* Windows reports failure by returning zero and setting the thread's last error.
  Each wrapper turns that into an exception carrying the numeric code and the
  system's own description, which is what ends up in the log.

Nothing here is imported from :mod:`ayris.core`: utilities sit at the bottom of
the dependency graph, so :class:`WinApiError` is a plain ``RuntimeError`` and the
action layer is the one that translates it into a Russian
:class:`ayris.core.errors.ActionError`.
"""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "GW_OWNER",
    "KEYEVENTF_EXTENDEDKEY",
    "KEYEVENTF_KEYUP",
    "KEYEVENTF_SCANCODE",
    "KEYEVENTF_UNICODE",
    "MAPVK_VK_TO_VSC_EX",
    "MONITOR_DEFAULTTONEAREST",
    "MOUSEEVENTF_ABSOLUTE",
    "MOUSEEVENTF_HWHEEL",
    "MOUSEEVENTF_LEFTDOWN",
    "MOUSEEVENTF_LEFTUP",
    "MOUSEEVENTF_MIDDLEDOWN",
    "MOUSEEVENTF_MIDDLEUP",
    "MOUSEEVENTF_MOVE",
    "MOUSEEVENTF_RIGHTDOWN",
    "MOUSEEVENTF_RIGHTUP",
    "MOUSEEVENTF_VIRTUALDESK",
    "MOUSEEVENTF_WHEEL",
    "MOUSEEVENTF_XDOWN",
    "MOUSEEVENTF_XUP",
    "SW_MAXIMIZE",
    "SW_MINIMIZE",
    "SW_RESTORE",
    "SW_SHOW",
    "SW_SHOWMINNOACTIVE",
    "SW_SHOWNORMAL",
    "VK_CONTROL",
    "VK_LEFT",
    "VK_LWIN",
    "VK_MENU",
    "VK_RIGHT",
    "WHEEL_DELTA",
    "WS_EX_APPWINDOW",
    "WS_EX_NOACTIVATE",
    "WS_EX_TOOLWINDOW",
    "MonitorInfo",
    "Rect",
    "WinApiError",
    "attach_thread_input",
    "available",
    "bring_window_to_top",
    "current_thread_id",
    "cursor_position",
    "dpi_for_monitor",
    "enum_display_monitors",
    "enum_windows",
    "foreground_window",
    "is_cloaked",
    "is_iconic",
    "is_window",
    "is_window_visible",
    "is_zoomed",
    "map_virtual_key",
    "monitor_from_point",
    "monitor_from_window",
    "monitor_info",
    "post_close",
    "press_chord",
    "process_image_name",
    "process_running",
    "send_close",
    "send_key_events",
    "send_mouse_event",
    "send_unicode_text",
    "set_foreground_window",
    "set_window_position",
    "shell_execute",
    "show_window",
    "switch_to_this_window",
    "terminate_process",
    "virtual_screen_rect",
    "vk_key_scan",
    "window_class_name",
    "window_ex_style",
    "window_owner",
    "window_pid",
    "window_rect",
    "window_thread_id",
    "window_title",
    "windows_build",
]

_log = get_logger(__name__)

# --- ShowWindow commands. Only the ones a voice command can ask for. ---
SW_HIDE: Final = 0
SW_SHOWNORMAL: Final = 1
SW_SHOWMINIMIZED: Final = 2
SW_MAXIMIZE: Final = 3
SW_SHOWNOACTIVATE: Final = 4
SW_SHOW: Final = 5
SW_MINIMIZE: Final = 6
SW_SHOWMINNOACTIVE: Final = 7
SW_RESTORE: Final = 9

# --- Extended window styles used to tell a real window from a helper one. ---
WS_EX_TOOLWINDOW: Final = 0x00000080
WS_EX_APPWINDOW: Final = 0x00040000
WS_EX_NOACTIVATE: Final = 0x08000000

GWL_EXSTYLE: Final = -20
GW_OWNER: Final = 4

WM_CLOSE: Final = 0x0010
SMTO_ABORTIFHUNG: Final = 0x0002

MONITOR_DEFAULTTONEAREST: Final = 0x00000002
MONITORINFOF_PRIMARY: Final = 0x00000001

SWP_NOSIZE: Final = 0x0001
SWP_NOMOVE: Final = 0x0002
SWP_NOZORDER: Final = 0x0004
SWP_NOACTIVATE: Final = 0x0010

# DWMWA_CLOAKED: set for windows the compositor hides — minimised Store apps and
# everything that lives on another virtual desktop.
DWMWA_CLOAKED: Final = 14

SEE_MASK_NOCLOSEPROCESS: Final = 0x00000040
SEE_MASK_FLAG_NO_UI: Final = 0x00000400

PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
PROCESS_TERMINATE: Final = 0x0001
SYNCHRONIZE: Final = 0x00100000
STILL_ACTIVE: Final = 259

# --- Virtual key codes for the chords the desktop switcher emulates. ---
VK_LWIN: Final = 0x5B
VK_CONTROL: Final = 0x11
VK_MENU: Final = 0x12
VK_LEFT: Final = 0x25
VK_RIGHT: Final = 0x27
KEYEVENTF_KEYUP: Final = 0x0002

# --- SendInput: the flags of one synthesised keystroke. ---
#: Send the scan code and ignore the virtual key. What games read.
KEYEVENTF_SCANCODE: Final = 0x0008
#: ``wScan`` holds a UTF-16 code unit, not a scan code. Layout-independent.
KEYEVENTF_UNICODE: Final = 0x0004
#: Prefix the scan code with ``E0`` — the right Alt/Ctrl, the arrows, Insert.
KEYEVENTF_EXTENDEDKEY: Final = 0x0001

#: ``MapVirtualKeyW``: virtual key to scan code, with the extended prefix kept.
MAPVK_VK_TO_VSC_EX: Final = 4

# --- SendInput: the flags of one synthesised mouse event. ---
MOUSEEVENTF_MOVE: Final = 0x0001
MOUSEEVENTF_LEFTDOWN: Final = 0x0002
MOUSEEVENTF_LEFTUP: Final = 0x0004
MOUSEEVENTF_RIGHTDOWN: Final = 0x0008
MOUSEEVENTF_RIGHTUP: Final = 0x0010
MOUSEEVENTF_MIDDLEDOWN: Final = 0x0020
MOUSEEVENTF_MIDDLEUP: Final = 0x0040
MOUSEEVENTF_XDOWN: Final = 0x0080
MOUSEEVENTF_XUP: Final = 0x0100
MOUSEEVENTF_WHEEL: Final = 0x0800
MOUSEEVENTF_HWHEEL: Final = 0x1000
#: ``dx``/``dy`` are normalised absolute coordinates, not a delta.
MOUSEEVENTF_ABSOLUTE: Final = 0x8000
#: Normalise against the whole virtual desktop instead of the primary display.
#: Without it a click meant for the second monitor lands on the first one.
MOUSEEVENTF_VIRTUALDESK: Final = 0x4000

#: One notch of the wheel, as ``WM_MOUSEWHEEL`` counts them.
WHEEL_DELTA: Final = 120

#: ``INPUT.type`` discriminators. Only the two kinds Ayris injects.
_INPUT_MOUSE: Final = 0
_INPUT_KEYBOARD: Final = 1

#: ``GetSystemMetrics`` indices of the virtual desktop's bounding box.
SM_XVIRTUALSCREEN: Final = 76
SM_YVIRTUALSCREEN: Final = 77
SM_CXVIRTUALSCREEN: Final = 78
SM_CYVIRTUALSCREEN: Final = 79

#: ``MONITOR_DPI_TYPE.MDT_EFFECTIVE_DPI`` — the scale the user actually set.
_MDT_EFFECTIVE_DPI: Final = 0
#: DPI of an unscaled display. Everything else is a multiple of it.
_DEFAULT_DPI: Final = 96

#: Pause between the key-down and key-up halves of an emulated chord. Without it
#: the shell occasionally sees the modifier as still held and swallows the arrow.
_CHORD_HOLD_S: Final = 0.02
#: Longest title Windows will hand back. Titles are short; the cap only exists so
#: a hostile window cannot make us allocate megabytes.
_MAX_TEXT: Final = 1024


class WinApiError(RuntimeError):
    """A WinAPI call failed, or this is not Windows at all.

    Carries the English text for the log. The action layer catches it and raises
    an :class:`ayris.core.errors.ActionError` with something a user can act on.
    """


@dataclass(frozen=True, slots=True)
class Rect:
    """A screen rectangle in physical pixels, as Windows reports one."""

    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def is_empty(self) -> bool:
        """Whether the rectangle has no area — an unmapped or zero-size window."""
        return self.width <= 0 or self.height <= 0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def as_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True)
class MonitorInfo:
    """One display: its full rectangle, the part not covered by the taskbar, and its name."""

    handle: int
    rect: Rect
    work: Rect
    device: str = ""
    primary: bool = False


class _RECT(ctypes.Structure):
    """``RECT``: the shape every WinAPI geometry call speaks in."""

    _fields_ = (
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    )


class _POINT(ctypes.Structure):
    """``POINT``: a single screen coordinate pair."""

    _fields_ = (
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    )


class _MONITORINFOEXW(ctypes.Structure):
    """``MONITORINFOEXW``: the plain struct plus the device name."""

    _fields_ = (
        ("cbSize", ctypes.c_ulong),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", ctypes.c_ulong),
        ("szDevice", ctypes.c_wchar * 32),
    )


class _SHELLEXECUTEINFOW(ctypes.Structure):
    """``SHELLEXECUTEINFOW``. All fifteen fields: the shell checks ``cbSize``."""

    _fields_ = (
        ("cbSize", ctypes.c_ulong),
        ("fMask", ctypes.c_ulong),
        ("hwnd", ctypes.c_void_p),
        ("lpVerb", ctypes.c_wchar_p),
        ("lpFile", ctypes.c_wchar_p),
        ("lpParameters", ctypes.c_wchar_p),
        ("lpDirectory", ctypes.c_wchar_p),
        ("nShow", ctypes.c_int),
        ("hInstApp", ctypes.c_void_p),
        ("lpIDList", ctypes.c_void_p),
        ("lpClass", ctypes.c_wchar_p),
        ("hkeyClass", ctypes.c_void_p),
        ("dwHotKey", ctypes.c_ulong),
        ("hIcon", ctypes.c_void_p),
        ("hProcess", ctypes.c_void_p),
    )


class _MOUSEINPUT(ctypes.Structure):
    """``MOUSEINPUT``: one synthesised move, click or wheel notch."""

    _fields_ = (
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    )


class _KEYBDINPUT(ctypes.Structure):
    """``KEYBDINPUT``: one synthesised half of a keystroke."""

    _fields_ = (
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.c_void_p),
    )


class _HARDWAREINPUT(ctypes.Structure):
    """``HARDWAREINPUT``. Never sent — it exists to size the union correctly."""

    _fields_ = (
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_ushort),
        ("wParamH", ctypes.c_ushort),
    )


class _INPUTUNION(ctypes.Union):
    _fields_ = (
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    )


class _INPUT(ctypes.Structure):
    """``INPUT``: the tagged union ``SendInput`` takes an array of.

    ``_anonymous_`` is deliberately not used — naming the union member keeps the
    assignment visible at the call site, and that is the part that goes wrong.
    """

    _fields_ = (
        ("type", ctypes.c_ulong),
        ("union", _INPUTUNION),
    )


def available() -> bool:
    """Whether WinAPI can be called at all in this process."""
    return sys.platform == "win32" and getattr(ctypes, "windll", None) is not None


def _win_function(library: str, function: str) -> Any | None:
    """Look up an entry point, or ``None`` when it is unavailable."""
    windll = getattr(ctypes, "windll", None)
    if windll is None:
        return None
    try:
        return getattr(getattr(windll, library), function)
    except (AttributeError, OSError):
        return None


def _require(library: str, function: str) -> Any:
    """Look up an entry point or raise. Every wrapper below starts here."""
    if sys.platform != "win32":
        raise WinApiError(f"{library}.{function} is unavailable: not running on Windows")
    entry = _win_function(library, function)
    if entry is None:
        raise WinApiError(f"{library}.{function} is not present in this Windows build")
    return entry


def _last_error(call: str) -> WinApiError:
    """Turn the thread's last error code into an exception worth logging."""
    if sys.platform != "win32":  # pragma: no cover - guarded by _require
        return WinApiError(f"{call} failed")
    code = int(ctypes.GetLastError())
    if not code:
        return WinApiError(f"{call} failed with no error code")
    return WinApiError(f"{call} failed: [{code}] {ctypes.FormatError(code).strip()}")


def _handle(hwnd: int) -> Any:
    """Window handle as the pointer-sized argument every user32 call expects."""
    return ctypes.c_void_p(hwnd)


# --------------------------------------------------------------------------- #
# Enumeration and window properties
# --------------------------------------------------------------------------- #


def enum_windows() -> list[int]:
    """Every top-level window, in z-order: the front-most one first.

    The order matters — it is what lets «переключись на браузер» pick the window
    the user saw last when several match.
    """
    enum_proc = _require("user32", "EnumWindows")
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def _collect(hwnd: Any, _param: Any) -> bool:
        if hwnd:
            handles.append(int(hwnd))
        return True

    enum_proc.restype = ctypes.c_bool
    enum_proc.argtypes = [callback_type, ctypes.c_void_p]
    if not enum_proc(callback_type(_collect), None) and not handles:
        raise _last_error("EnumWindows")
    return handles


def window_title(hwnd: int) -> str:
    """Window caption, or ``""`` for a window without one."""
    length_of = _require("user32", "GetWindowTextLengthW")
    read = _require("user32", "GetWindowTextW")
    length_of.restype = ctypes.c_int
    length_of.argtypes = [ctypes.c_void_p]
    size = int(length_of(_handle(hwnd)))
    if size <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(min(size, _MAX_TEXT) + 1)
    read.restype = ctypes.c_int
    read.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    read(_handle(hwnd), buffer, len(buffer))
    return str(buffer.value)


def window_class_name(hwnd: int) -> str:
    """Window class, e.g. ``Chrome_WidgetWin_1``. Stable across localisations."""
    get_class = _require("user32", "GetClassNameW")
    buffer = ctypes.create_unicode_buffer(256)
    get_class.restype = ctypes.c_int
    get_class.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
    get_class(_handle(hwnd), buffer, len(buffer))
    return str(buffer.value)


def window_pid(hwnd: int) -> int:
    """Process id owning the window, or ``0`` when the handle is gone."""
    get_thread = _require("user32", "GetWindowThreadProcessId")
    pid = ctypes.c_ulong(0)
    get_thread.restype = ctypes.c_ulong
    get_thread.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    get_thread(_handle(hwnd), ctypes.byref(pid))
    return int(pid.value)


def window_thread_id(hwnd: int) -> int:
    """Id of the thread that created the window. Needed by ``AttachThreadInput``."""
    get_thread = _require("user32", "GetWindowThreadProcessId")
    get_thread.restype = ctypes.c_ulong
    get_thread.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return int(get_thread(_handle(hwnd), None))


def is_window(hwnd: int) -> bool:
    """Whether the handle still refers to a window."""
    check = _require("user32", "IsWindow")
    check.restype = ctypes.c_bool
    check.argtypes = [ctypes.c_void_p]
    return bool(check(_handle(hwnd)))


def is_window_visible(hwnd: int) -> bool:
    check = _require("user32", "IsWindowVisible")
    check.restype = ctypes.c_bool
    check.argtypes = [ctypes.c_void_p]
    return bool(check(_handle(hwnd)))


def is_iconic(hwnd: int) -> bool:
    """Whether the window is minimised."""
    check = _require("user32", "IsIconic")
    check.restype = ctypes.c_bool
    check.argtypes = [ctypes.c_void_p]
    return bool(check(_handle(hwnd)))


def is_zoomed(hwnd: int) -> bool:
    """Whether the window is maximised."""
    check = _require("user32", "IsZoomed")
    check.restype = ctypes.c_bool
    check.argtypes = [ctypes.c_void_p]
    return bool(check(_handle(hwnd)))


def is_cloaked(hwnd: int) -> bool:
    """Whether the compositor is hiding the window.

    True for Store apps that are suspended and for every window sitting on
    another virtual desktop. Such windows are visible by ``IsWindowVisible`` but
    have no business in a list the user is offered.
    """
    get_attribute = _win_function("dwmapi", "DwmGetWindowAttribute")
    if get_attribute is None:
        return False
    value = ctypes.c_int(0)
    get_attribute.restype = ctypes.c_long
    get_attribute.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    hresult = int(
        get_attribute(
            _handle(hwnd),
            DWMWA_CLOAKED,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
    )
    return hresult == 0 and value.value != 0


def window_ex_style(hwnd: int) -> int:
    """Extended style bits — ``WS_EX_TOOLWINDOW`` and friends."""
    get_long = _win_function("user32", "GetWindowLongPtrW") or _require("user32", "GetWindowLongW")
    get_long.restype = ctypes.c_ssize_t
    get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return int(get_long(_handle(hwnd), GWL_EXSTYLE))


def window_owner(hwnd: int) -> int:
    """Owner window handle, or ``0`` for a top-level window of its own."""
    get_window = _require("user32", "GetWindow")
    get_window.restype = ctypes.c_void_p
    get_window.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    owner = get_window(_handle(hwnd), GW_OWNER)
    return int(owner or 0)


def window_rect(hwnd: int) -> Rect:
    """Outer bounds of the window in screen coordinates."""
    get_rect = _require("user32", "GetWindowRect")
    box = _RECT()
    get_rect.restype = ctypes.c_bool
    get_rect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RECT)]
    if not get_rect(_handle(hwnd), ctypes.byref(box)):
        raise _last_error("GetWindowRect")
    return Rect(int(box.left), int(box.top), int(box.right), int(box.bottom))


def foreground_window() -> int:
    """Handle of the window with the keyboard focus, or ``0``."""
    get_foreground = _require("user32", "GetForegroundWindow")
    get_foreground.restype = ctypes.c_void_p
    get_foreground.argtypes = []
    return int(get_foreground() or 0)


# --------------------------------------------------------------------------- #
# Changing a window's state
# --------------------------------------------------------------------------- #


def show_window(hwnd: int, command: int) -> bool:
    """``ShowWindow``. Returns the previous visibility, not whether it worked."""
    show = _require("user32", "ShowWindow")
    show.restype = ctypes.c_bool
    show.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return bool(show(_handle(hwnd), command))


def set_window_position(hwnd: int, rect: Rect) -> None:
    """Move and resize without touching the z-order or the focus."""
    set_pos = _require("user32", "SetWindowPos")
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
    ok = set_pos(
        _handle(hwnd),
        None,
        rect.left,
        rect.top,
        rect.width,
        rect.height,
        SWP_NOZORDER | SWP_NOACTIVATE,
    )
    if not ok:
        raise _last_error("SetWindowPos")


def set_foreground_window(hwnd: int) -> bool:
    """``SetForegroundWindow``. Fails silently when we lack foreground rights.

    Windows only grants the right to a process the user has just interacted with,
    so a voice command routinely gets ``False`` here. The caller is expected to
    try the workaround in :mod:`ayris.actions.system.windows` and, if that also
    fails, to say so out loud rather than pretend it worked.
    """
    set_fg = _require("user32", "SetForegroundWindow")
    set_fg.restype = ctypes.c_bool
    set_fg.argtypes = [ctypes.c_void_p]
    return bool(set_fg(_handle(hwnd)))


def bring_window_to_top(hwnd: int) -> bool:
    """Raise the window in the z-order. Does not give it the focus."""
    bring = _require("user32", "BringWindowToTop")
    bring.restype = ctypes.c_bool
    bring.argtypes = [ctypes.c_void_p]
    return bool(bring(_handle(hwnd)))


def switch_to_this_window(hwnd: int) -> bool:
    """Undocumented ``SwitchToThisWindow`` — the Alt+Tab code path.

    Present since Windows XP and still exported by Windows 11. Ignores the
    foreground lock, which is exactly why it is the last resort here.
    """
    switch = _win_function("user32", "SwitchToThisWindow")
    if switch is None:
        return False
    switch.restype = None
    switch.argtypes = [ctypes.c_void_p, ctypes.c_bool]
    switch(_handle(hwnd), True)
    return True


def current_thread_id() -> int:
    """Id of the calling thread."""
    get_id = _require("kernel32", "GetCurrentThreadId")
    get_id.restype = ctypes.c_ulong
    get_id.argtypes = []
    return int(get_id())


def attach_thread_input(target_thread: int, *, attach: bool) -> bool:
    """Share the input queue with another thread, or stop sharing it.

    Attaching to the thread that currently owns the foreground makes Windows
    treat our ``SetForegroundWindow`` as coming from that thread. It must always
    be undone: a stale attachment freezes both input queues together.
    """
    attach_input = _require("user32", "AttachThreadInput")
    attach_input.restype = ctypes.c_bool
    attach_input.argtypes = [ctypes.c_ulong, ctypes.c_ulong, ctypes.c_bool]
    return bool(attach_input(current_thread_id(), target_thread, attach))


def press_chord(keys: Sequence[int]) -> None:
    """Tap a chord: press the keys in order, release them in reverse.

    Used for the two shell shortcuts that have no API — Win+Ctrl+Left/Right for
    virtual desktops — and for the lone Alt tap that unsticks the foreground lock.
    """
    keybd_event = _require("user32", "keybd_event")
    keybd_event.restype = None
    keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_void_p]
    for key in keys:
        keybd_event(key, 0, 0, None)
    time.sleep(_CHORD_HOLD_S)
    for key in reversed(tuple(keys)):
        keybd_event(key, 0, KEYEVENTF_KEYUP, None)


def post_close(hwnd: int) -> bool:
    """Ask the window to close and return immediately.

    ``PostMessage`` never blocks, so a hung application cannot stall the action.
    """
    post = _require("user32", "PostMessageW")
    post.restype = ctypes.c_bool
    post.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    return bool(post(_handle(hwnd), WM_CLOSE, None, None))


def send_close(hwnd: int, timeout_ms: int = 2000) -> bool:
    """Ask the window to close and wait up to ``timeout_ms`` for it to answer.

    ``SMTO_ABORTIFHUNG`` means a frozen application costs us the timeout and
    nothing more. A ``False`` here is the signal that only ``force`` will help.
    """
    send = _require("user32", "SendMessageTimeoutW")
    result = ctypes.c_size_t(0)
    send.restype = ctypes.c_ssize_t
    send.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    answered = int(
        send(
            _handle(hwnd),
            WM_CLOSE,
            None,
            None,
            SMTO_ABORTIFHUNG,
            max(1, timeout_ms),
            ctypes.byref(result),
        )
    )
    return answered != 0


# --------------------------------------------------------------------------- #
# Launching and processes
# --------------------------------------------------------------------------- #


def shell_execute(
    file: str,
    *,
    arguments: str = "",
    directory: str = "",
    verb: str = "",
    show: int = SW_SHOWNORMAL,
) -> int:
    """Launch ``file`` the way a double click would, and return the new pid.

    ``ShellExecuteExW`` rather than ``ShellExecuteW``: the extended form is the
    only one that can hand back a process handle, and knowing the pid is what
    later lets :class:`~ayris.actions.system.apps.CloseApp` find the windows it
    should close. Returns ``0`` when the shell handled the request without
    starting a process — opening a document in a running editor, for one.
    """
    execute = _require("shell32", "ShellExecuteExW")
    get_pid = _require("kernel32", "GetProcessId")
    close_handle = _require("kernel32", "CloseHandle")

    info = _SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(_SHELLEXECUTEINFOW)
    info.fMask = SEE_MASK_NOCLOSEPROCESS | SEE_MASK_FLAG_NO_UI
    info.lpVerb = verb or None
    info.lpFile = file
    info.lpParameters = arguments or None
    info.lpDirectory = directory or None
    info.nShow = show

    execute.restype = ctypes.c_bool
    execute.argtypes = [ctypes.POINTER(_SHELLEXECUTEINFOW)]
    if not execute(ctypes.byref(info)):
        raise _last_error(f"ShellExecuteExW({file})")

    handle = info.hProcess
    if not handle:
        return 0
    get_pid.restype = ctypes.c_ulong
    get_pid.argtypes = [ctypes.c_void_p]
    pid = int(get_pid(handle))
    close_handle.restype = ctypes.c_bool
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle(handle)
    return pid


def _open_process(pid: int, access: int) -> Any | None:
    """Process handle with the least access that will do, or ``None``."""
    open_process = _require("kernel32", "OpenProcess")
    open_process.restype = ctypes.c_void_p
    open_process.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
    handle = open_process(access, False, pid)
    return handle or None


def _close_handle(handle: Any) -> None:
    close_handle = _require("kernel32", "CloseHandle")
    close_handle.restype = ctypes.c_bool
    close_handle.argtypes = [ctypes.c_void_p]
    close_handle(handle)


def process_image_name(pid: int) -> str:
    """Executable file name of a process, e.g. ``chrome.exe``.

    ``QueryFullProcessImageNameW`` with ``PROCESS_QUERY_LIMITED_INFORMATION``
    works for processes of other integrity levels, which the older
    ``GetModuleFileNameExW`` does not. Returns ``""`` when the process is gone or
    protected rather than raising: a name is nice to have, not essential.
    """
    if pid <= 0:
        return ""
    handle = _open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if handle is None:
        return ""
    try:
        query = _require("kernel32", "QueryFullProcessImageNameW")
        buffer = ctypes.create_unicode_buffer(_MAX_TEXT)
        size = ctypes.c_ulong(len(buffer))
        query.restype = ctypes.c_bool
        query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_wchar_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        if not query(handle, 0, buffer, ctypes.byref(size)):
            return ""
        path = str(buffer.value)
    finally:
        _close_handle(handle)
    return path.rsplit("\\", 1)[-1]


def process_running(pid: int) -> bool:
    """Whether the process still exists and has not exited."""
    if pid <= 0:
        return False
    handle = _open_process(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if handle is None:
        return False
    try:
        get_code = _require("kernel32", "GetExitCodeProcess")
        code = ctypes.c_ulong(0)
        get_code.restype = ctypes.c_bool
        get_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        if not get_code(handle, ctypes.byref(code)):
            return False
        return int(code.value) == STILL_ACTIVE
    finally:
        _close_handle(handle)


def terminate_process(pid: int) -> None:
    """Kill a process outright. Only ever called with the user's explicit consent."""
    handle = _open_process(pid, PROCESS_TERMINATE | SYNCHRONIZE)
    if handle is None:
        raise _last_error(f"OpenProcess({pid})")
    try:
        terminate = _require("kernel32", "TerminateProcess")
        terminate.restype = ctypes.c_bool
        terminate.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        if not terminate(handle, 1):
            raise _last_error(f"TerminateProcess({pid})")
    finally:
        _close_handle(handle)


# --------------------------------------------------------------------------- #
# Monitors
# --------------------------------------------------------------------------- #


def enum_display_monitors() -> list[int]:
    """Handles of every attached display, in the order Windows reports them."""
    enum_monitors = _require("user32", "EnumDisplayMonitors")
    handles: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_RECT),
        ctypes.c_void_p,
    )

    def _collect(monitor: Any, _dc: Any, _rect: Any, _param: Any) -> bool:
        if monitor:
            handles.append(int(monitor))
        return True

    enum_monitors.restype = ctypes.c_bool
    enum_monitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p, callback_type, ctypes.c_void_p]
    if not enum_monitors(None, None, callback_type(_collect), None) and not handles:
        raise _last_error("EnumDisplayMonitors")
    return handles


def monitor_info(handle: int) -> MonitorInfo:
    """Geometry and name of one display."""
    get_info = _require("user32", "GetMonitorInfoW")
    info = _MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(_MONITORINFOEXW)
    get_info.restype = ctypes.c_bool
    get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MONITORINFOEXW)]
    if not get_info(ctypes.c_void_p(handle), ctypes.byref(info)):
        raise _last_error("GetMonitorInfoW")
    full = info.rcMonitor
    work = info.rcWork
    return MonitorInfo(
        handle=handle,
        rect=Rect(int(full.left), int(full.top), int(full.right), int(full.bottom)),
        work=Rect(int(work.left), int(work.top), int(work.right), int(work.bottom)),
        device=str(info.szDevice),
        primary=bool(int(info.dwFlags) & MONITORINFOF_PRIMARY),
    )


def monitor_from_window(hwnd: int, flags: int = MONITOR_DEFAULTTONEAREST) -> int:
    """Handle of the display the window is mostly on."""
    from_window = _require("user32", "MonitorFromWindow")
    from_window.restype = ctypes.c_void_p
    from_window.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    return int(from_window(_handle(hwnd), flags) or 0)


def windows_build() -> int:
    """Build number of the running Windows, or ``0`` elsewhere.

    Virtual desktops arrived in build 10240 and their internals moved twice
    since; the desktop actions refuse to guess below that line.
    """
    if sys.platform != "win32":
        return 0
    version = sys.getwindowsversion()
    return int(version.build)


def virtual_screen_rect() -> Rect:
    """Bounding box of every display together, in physical pixels.

    This is the coordinate space ``SendInput`` normalises absolute mouse moves
    against, and its origin is not ``(0, 0)``: a second monitor placed to the
    left of the primary one gives the virtual desktop a negative ``left``. That
    offset is the whole reason a click aimed at the second screen lands wrong
    when the normalisation forgets it.
    """
    metrics = _require("user32", "GetSystemMetrics")
    metrics.restype = ctypes.c_int
    metrics.argtypes = [ctypes.c_int]
    left = int(metrics(SM_XVIRTUALSCREEN))
    top = int(metrics(SM_YVIRTUALSCREEN))
    width = int(metrics(SM_CXVIRTUALSCREEN))
    height = int(metrics(SM_CYVIRTUALSCREEN))
    return Rect(left, top, left + width, top + height)


def cursor_position() -> tuple[int, int]:
    """Where the mouse pointer is now, in virtual-desktop coordinates."""
    get_pos = _require("user32", "GetCursorPos")
    point = _POINT()
    get_pos.restype = ctypes.c_bool
    get_pos.argtypes = [ctypes.POINTER(_POINT)]
    if not get_pos(ctypes.byref(point)):
        raise _last_error("GetCursorPos")
    return int(point.x), int(point.y)


def monitor_from_point(x: int, y: int, flags: int = MONITOR_DEFAULTTONEAREST) -> int:
    """Handle of the display a point falls on, or the nearest one."""
    from_point = _require("user32", "MonitorFromPoint")
    from_point.restype = ctypes.c_void_p
    from_point.argtypes = [_POINT, ctypes.c_ulong]
    return int(from_point(_POINT(x, y), flags) or 0)


def dpi_for_monitor(handle: int) -> int:
    """Effective DPI of one display, or ``96`` when it cannot be asked.

    ``GetDpiForMonitor`` lives in shcore and arrived in Windows 8.1; on an older
    build, or in a process that is not per-monitor DPI aware, the answer is the
    unscaled 96 and callers treat that as "no scaling to undo" rather than as an
    error. A wrong DPI moves the cursor to the wrong place, so this never raises
    for the sake of the caller's arithmetic.
    """
    get_dpi = _win_function("shcore", "GetDpiForMonitor")
    if get_dpi is None:
        return _DEFAULT_DPI
    horizontal = ctypes.c_uint(0)
    vertical = ctypes.c_uint(0)
    get_dpi.restype = ctypes.c_long
    get_dpi.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
    ]
    status = int(
        get_dpi(
            ctypes.c_void_p(handle),
            _MDT_EFFECTIVE_DPI,
            ctypes.byref(horizontal),
            ctypes.byref(vertical),
        )
    )
    if status != 0 or not horizontal.value:
        return _DEFAULT_DPI
    return int(horizontal.value)


# --------------------------------------------------------------------------- #
# Synthesised input
# --------------------------------------------------------------------------- #


def map_virtual_key(vk: int) -> int:
    """Scan code of a virtual key, or ``0`` when the key has none.

    ``MAPVK_VK_TO_VSC_EX`` rather than the plain form: it keeps the ``E0`` prefix
    of the extended keys, which is what tells the right Ctrl from the left one.
    Only the low byte is the scan code; the prefix is reported separately through
    :data:`KEYEVENTF_EXTENDEDKEY`.
    """
    mapper = _require("user32", "MapVirtualKeyW")
    mapper.restype = ctypes.c_uint
    mapper.argtypes = [ctypes.c_uint, ctypes.c_uint]
    return int(mapper(vk, MAPVK_VK_TO_VSC_EX)) & 0xFF


def vk_key_scan(char: str) -> int:
    """``VkKeyScanW``: which key and modifiers type ``char`` on the current layout.

    Returns ``-1`` when the active layout cannot produce the character at all —
    a Cyrillic letter under a US layout, which is precisely the case that forces
    :data:`KEYEVENTF_UNICODE` instead of scan codes.
    """
    scan = _require("user32", "VkKeyScanW")
    scan.restype = ctypes.c_short
    scan.argtypes = [ctypes.c_ushort]
    return int(scan(ord(char[0])))


def _send_input(events: Sequence[_INPUT], *, call: str) -> int:
    """Hand an array of ``INPUT`` to ``SendInput`` and return what it accepted.

    A return below ``len(events)`` means the injection was blocked, and the usual
    reason is UIPI: a process at medium integrity cannot send input to a window
    of an elevated one. The code is reported as-is — the input actions turn it
    into the sentence about administrator rights.
    """
    if not events:
        return 0
    send = _require("user32", "SendInput")
    send.restype = ctypes.c_uint
    send.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_int]
    array = (_INPUT * len(events))(*events)
    sent = int(send(len(events), ctypes.byref(array), ctypes.sizeof(_INPUT)))
    if sent == 0:
        raise _last_error(call)
    return sent


def _key_event(*, vk: int, scan: int, flags: int) -> _INPUT:
    """One ``INPUT`` holding half a keystroke."""
    event = _INPUT()
    event.type = _INPUT_KEYBOARD
    event.union.ki = _KEYBDINPUT(
        wVk=vk,
        wScan=scan,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    return event


def send_key_events(events: Sequence[tuple[int, int, int]]) -> int:
    """Inject keystrokes as ``(virtual key, scan code, flags)`` triples.

    Sent as one array rather than one call per event: ``SendInput`` guarantees
    that a batch is not interleaved with real typing, which is what keeps a
    modifier from being reported as released in the middle of a chord.
    """
    batch = [_key_event(vk=vk, scan=scan, flags=flags) for vk, scan, flags in events]
    return _send_input(batch, call="SendInput(keyboard)")


def send_unicode_text(text: str) -> int:
    """Type ``text`` character by character through :data:`KEYEVENTF_UNICODE`.

    Layout-independent by construction — the character travels as a UTF-16 code
    unit, so Cyrillic arrives with a US layout active. Characters outside the
    BMP (emoji) are two surrogates, and both halves are sent, which is what the
    documentation requires and what makes them arrive as one glyph.
    """
    encoded = text.encode("utf-16-le")
    events: list[tuple[int, int, int]] = []
    for offset in range(0, len(encoded), 2):
        unit = int.from_bytes(encoded[offset : offset + 2], "little")
        events.append((0, unit, KEYEVENTF_UNICODE))
        events.append((0, unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP))
    return send_key_events(events)


def send_mouse_event(
    *,
    flags: int,
    dx: int = 0,
    dy: int = 0,
    data: int = 0,
) -> int:
    """Inject one mouse event: a move, a button change or a wheel notch."""
    event = _INPUT()
    event.type = _INPUT_MOUSE
    event.union.mi = _MOUSEINPUT(
        dx=dx,
        dy=dy,
        mouseData=ctypes.c_ulong(data & 0xFFFFFFFF).value,
        dwFlags=flags,
        time=0,
        dwExtraInfo=None,
    )
    return _send_input([event], call="SendInput(mouse)")
