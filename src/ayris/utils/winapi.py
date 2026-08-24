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

import contextlib
import ctypes
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

__all__ = [
    "CF_BITMAP",
    "CF_DIB",
    "CF_DIBV5",
    "CF_HDROP",
    "CF_TIFF",
    "CF_UNICODETEXT",
    "DWMWA_EXTENDED_FRAME_BOUNDS",
    "ELEVATION_TYPE_DEFAULT",
    "ELEVATION_TYPE_FULL",
    "ELEVATION_TYPE_LIMITED",
    "ERROR_CANCELLED",
    "EWX_FORCEIFHUNG",
    "EWX_LOGOFF",
    "EWX_POWEROFF",
    "EWX_REBOOT",
    "EWX_SHUTDOWN",
    "GW_OWNER",
    "IMAGE_FORMATS",
    "INTEGRITY_HIGH",
    "INTEGRITY_LOW",
    "INTEGRITY_MEDIUM",
    "INTEGRITY_SYSTEM",
    "INTEGRITY_UNTRUSTED",
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
    "SE_SHUTDOWN_NAME",
    "SHUTDOWN_REASON_PLANNED",
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
    "WM_CLIPBOARDUPDATE",
    "WS_EX_APPWINDOW",
    "WS_EX_NOACTIVATE",
    "WS_EX_TOOLWINDOW",
    "ClipboardData",
    "ElevationInfo",
    "MessageWindow",
    "MonitorInfo",
    "ProcessRun",
    "Rect",
    "WinApiError",
    "abort_shutdown",
    "add_clipboard_format_listener",
    "attach_thread_input",
    "available",
    "bring_window_to_top",
    "clipboard_clear",
    "clipboard_sequence_number",
    "clipboard_set_binary",
    "clipboard_set_text",
    "console_output_codepage",
    "current_thread_id",
    "cursor_position",
    "display_device",
    "dpi_for_monitor",
    "enable_privilege",
    "enum_display_monitors",
    "enum_windows",
    "exit_windows",
    "extended_frame_bounds",
    "foreground_window",
    "initiate_shutdown",
    "is_cloaked",
    "is_iconic",
    "is_window",
    "is_window_visible",
    "is_zoomed",
    "lock_workstation",
    "map_virtual_key",
    "monitor_from_point",
    "monitor_from_window",
    "monitor_info",
    "oem_codepage",
    "post_close",
    "press_chord",
    "process_elevation",
    "process_image_name",
    "process_running",
    "read_clipboard",
    "register_clipboard_format",
    "remove_clipboard_format_listener",
    "send_close",
    "send_key_events",
    "send_mouse_event",
    "send_unicode_text",
    "set_foreground_window",
    "set_suspend_state",
    "set_window_position",
    "shell_execute",
    "shell_execute_ex",
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
#: DWMWA_EXTENDED_FRAME_BOUNDS: what the window actually covers on screen.
#: ``GetWindowRect`` adds the invisible resize margins the compositor draws the
#: drop shadow into — around 7 px per side on Windows 11 — so a screenshot cropped
#: to it carries a border of whatever was behind the window.
DWMWA_EXTENDED_FRAME_BOUNDS: Final = 9

#: Clipboard formats a screenshot is offered in. ``CF_DIB`` is what every program
#: back to Paint understands; the registered ``"PNG"`` format keeps the alpha
#: channel for the ones that prefer it (browsers, Office, Telegram).
CF_DIB: Final = 8

#: The other standard formats Ayris only ever *reads*, to tell what is on the
#: clipboard. Text is the only kind the history stores; a picture or a set of
#: dragged files is recognised and reported, never copied into the database.
CF_BITMAP: Final = 2
CF_TIFF: Final = 6
CF_UNICODETEXT: Final = 13
CF_HDROP: Final = 15
CF_DIBV5: Final = 17

#: Formats that mean «there is a picture on the clipboard», in the order Windows
#: itself prefers them.
IMAGE_FORMATS: Final = (CF_DIBV5, CF_DIB, CF_BITMAP, CF_TIFF)

#: ``WM_CLIPBOARDUPDATE`` — the one message a clipboard listener window gets. It
#: carries no data: the window is told the clipboard changed and reads it itself.
WM_CLIPBOARDUPDATE: Final = 0x031D

#: ``GlobalAlloc`` flags: moveable memory, which is the only kind the clipboard
#: accepts ownership of.
GMEM_MOVEABLE: Final = 0x0002

SEE_MASK_NOCLOSEPROCESS: Final = 0x00000040
SEE_MASK_FLAG_NO_UI: Final = 0x00000400

PROCESS_QUERY_LIMITED_INFORMATION: Final = 0x1000
PROCESS_TERMINATE: Final = 0x0001
SYNCHRONIZE: Final = 0x00100000
STILL_ACTIVE: Final = 259

#: The user clicked «Нет» in the UAC prompt. Not a failure of the call.
ERROR_CANCELLED: Final = 1223

#: ``WaitForSingleObject`` gave up before the process ended.
WAIT_TIMEOUT: Final = 0x00000102
_INFINITE: Final = 0xFFFFFFFF

#: Access rights for the process token: reading the elevation flag needs
#: ``TOKEN_QUERY``, enabling ``SE_SHUTDOWN_NAME`` also needs ``ADJUST_PRIVILEGES``.
TOKEN_QUERY: Final = 0x0008
TOKEN_ADJUST_PRIVILEGES: Final = 0x0020

#: ``TOKEN_INFORMATION_CLASS`` members this module reads.
_TOKEN_ELEVATION_TYPE: Final = 18
_TOKEN_ELEVATION: Final = 20
_TOKEN_INTEGRITY_LEVEL: Final = 25

#: Integrity levels, as the RID of the label SID. A process below ``HIGH`` cannot
#: shut the machine down however many privileges it asks for.
INTEGRITY_UNTRUSTED: Final = 0x0000
INTEGRITY_LOW: Final = 0x1000
INTEGRITY_MEDIUM: Final = 0x2000
INTEGRITY_HIGH: Final = 0x3000
INTEGRITY_SYSTEM: Final = 0x4000

#: ``TOKEN_ELEVATION_TYPE``: default (UAC off, or a plain user), full (elevated),
#: limited (an administrator running without elevation — a prompt would work).
ELEVATION_TYPE_DEFAULT: Final = 1
ELEVATION_TYPE_FULL: Final = 2
ELEVATION_TYPE_LIMITED: Final = 3

SE_PRIVILEGE_ENABLED: Final = 0x00000002

#: The privilege ``ExitWindowsEx`` and ``InitiateSystemShutdownExW`` demand. Held
#: by administrators but disabled in the token until asked for, which is why
#: shutdown fails with «access denied» in code that never calls
#: :func:`enable_privilege`.
SE_SHUTDOWN_NAME: Final = "SeShutdownPrivilege"

#: ``ExitWindowsEx`` flags. ``EWX_FORCEIFHUNG`` is deliberately the only forcing
#: variant offered: ``EWX_FORCE`` skips the «сохранить изменения?» prompts of every
#: running program, which loses work.
EWX_LOGOFF: Final = 0x00000000
EWX_SHUTDOWN: Final = 0x00000001
EWX_REBOOT: Final = 0x00000002
EWX_POWEROFF: Final = 0x00000008
EWX_FORCEIFHUNG: Final = 0x00000010

#: ``SHTDN_REASON_*``: a planned change made by software at the user's request.
SHTDN_REASON_MAJOR_OTHER: Final = 0x00000000
SHTDN_REASON_MINOR_OTHER: Final = 0x00000000
SHTDN_REASON_FLAG_PLANNED: Final = 0x80000000
SHUTDOWN_REASON_PLANNED: Final = (
    SHTDN_REASON_MAJOR_OTHER | SHTDN_REASON_MINOR_OTHER | SHTDN_REASON_FLAG_PLANNED
)

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
#: ``OpenClipboard`` fails outright while another process holds the clipboard, and
#: something always does right after a Ctrl+C. Retry a few times over ~0.15 s.
_CLIPBOARD_TRIES: Final = 6
_CLIPBOARD_RETRY_S: Final = 0.03
#: Caps on what one clipboard read may pull into memory. Text longer than this is
#: not something a person copied on purpose, and the history limit is far lower
#: anyway; the blob cap only guards a caller that asked for a picture format.
_MAX_CLIPBOARD_TEXT: Final = 1_000_000
_MAX_CLIPBOARD_BLOB: Final = 4096
_MAX_CLIPBOARD_FORMATS: Final = 64
_MAX_DROPPED_FILES: Final = 256
#: ``HWND_MESSAGE``: parent of a window that exists only to receive messages.
_HWND_MESSAGE: Final = -3
#: ``WM_APP + 1``: the message :meth:`MessageWindow.stop` posts to itself. Windows
#: promises the ``WM_APP`` range is never used by the system.
_WM_AYRIS_STOP: Final = 0x8001
_ERROR_CLASS_ALREADY_EXISTS: Final = 1410
#: Longest title Windows will hand back. Titles are short; the cap only exists so
#: a hostile window cannot make us allocate megabytes.
_MAX_TEXT: Final = 1024


class WinApiError(RuntimeError):
    """A WinAPI call failed, or this is not Windows at all.

    Carries the English text for the log. The action layer catches it and raises
    an :class:`ayris.core.errors.ActionError` with something a user can act on.

    ``code`` is the Windows error code when there was one, ``0`` otherwise. It
    exists because a few codes mean something a caller must act on rather than
    just report: :data:`ERROR_CANCELLED` from an elevation request is the user
    clicking «Нет» in the UAC prompt, not a broken call.
    """

    def __init__(self, message: str, *, code: int = 0) -> None:
        super().__init__(message)
        self.code = code


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


class _DISPLAYDEVICEW(ctypes.Structure):
    """``DISPLAY_DEVICEW``: the adapter or the panel behind one display name."""

    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("DeviceName", ctypes.c_wchar * 32),
        ("DeviceString", ctypes.c_wchar * 128),
        ("StateFlags", ctypes.c_ulong),
        ("DeviceID", ctypes.c_wchar * 128),
        ("DeviceKey", ctypes.c_wchar * 128),
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


class _LUID(ctypes.Structure):
    """``LUID``: the machine-local id a privilege name resolves to."""

    _fields_ = (
        ("LowPart", ctypes.c_ulong),
        ("HighPart", ctypes.c_long),
    )


class _LuidAndAttributes(ctypes.Structure):
    _fields_ = (
        ("Luid", _LUID),
        ("Attributes", ctypes.c_ulong),
    )


class _TokenPrivileges(ctypes.Structure):
    """``TOKEN_PRIVILEGES`` sized for exactly one privilege.

    The real structure ends in a variable-length array; one entry is all
    :func:`enable_privilege` ever adjusts, and a fixed-size declaration keeps the
    ``byref`` call honest about how many bytes Windows may read.
    """

    _fields_ = (
        ("PrivilegeCount", ctypes.c_ulong),
        ("Privileges", _LuidAndAttributes * 1),
    )


@dataclass(frozen=True, slots=True)
class ElevationInfo:
    """What the process token says about this process's rights.

    Args:
        elevated: ``TokenElevation`` — the flag that decides whether a privileged
            call will go through at all.
        elevation_type: ``TokenElevationType``. ``ELEVATION_TYPE_LIMITED`` is the
            interesting one: an administrator who is not elevated *yet*, so a UAC
            prompt would succeed.
        integrity_level: RID of the token's integrity label,
            ``INTEGRITY_MEDIUM`` for a normal process.
    """

    elevated: bool = False
    elevation_type: int = ELEVATION_TYPE_DEFAULT
    integrity_level: int = INTEGRITY_MEDIUM

    @property
    def can_elevate(self) -> bool:
        """Whether asking for elevation could plausibly succeed."""
        return self.elevation_type == ELEVATION_TYPE_LIMITED

    @property
    def high_integrity(self) -> bool:
        """Whether the token runs at high integrity or above."""
        return self.integrity_level >= INTEGRITY_HIGH


@dataclass(frozen=True, slots=True)
class ProcessRun:
    """A process the shell started for us.

    ``exit_code`` is ``None`` while the process is still running, and also when it
    ended before we could ask — the shell hands back a handle, not a promise that
    the process outlives the next statement.
    """

    pid: int = 0
    exit_code: int | None = None
    timed_out: bool = False

    @property
    def finished(self) -> bool:
        """Whether the process is known to have ended."""
        return not self.timed_out and self.exit_code is not None


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
    return WinApiError(f"{call} failed: [{code}] {ctypes.FormatError(code).strip()}", code=code)


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


def extended_frame_bounds(hwnd: int) -> Rect:
    """What the window visibly covers, without the shadow margins.

    The rectangle to crop a screenshot to. :func:`window_rect` answers something
    larger: since Aero the resize border is drawn outside the visible frame and
    is transparent, so on Windows 11 ``GetWindowRect`` overshoots by roughly
    7 px on the left, right and bottom, and a capture of that region has a strip
    of the desktop — or of the window behind — along three of its edges.

    Falls back to :func:`window_rect` when the compositor refuses to answer:
    dwmapi is missing, composition is off, or the handle is already gone. A
    slightly too large screenshot beats no screenshot.
    """
    get_attribute = _win_function("dwmapi", "DwmGetWindowAttribute")
    if get_attribute is not None:
        box = _RECT()
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
                DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.byref(box),
                ctypes.sizeof(box),
            )
        )
        bounds = Rect(int(box.left), int(box.top), int(box.right), int(box.bottom))
        if hresult == 0 and not bounds.is_empty:
            return bounds
        _log.debug("EXTENDED_FRAME_BOUNDS refused: hresult=0x%08x", hresult & 0xFFFFFFFF)
    return window_rect(hwnd)


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
    return shell_execute_ex(
        file,
        arguments=arguments,
        directory=directory,
        verb=verb,
        show=show,
    ).pid


def shell_execute_ex(
    file: str,
    *,
    arguments: str = "",
    directory: str = "",
    verb: str = "",
    show: int = SW_SHOWNORMAL,
    wait_ms: int = 0,
) -> ProcessRun:
    """Launch through the shell and optionally wait for the result.

    The waiting form exists for the elevation helper: a process started with
    ``verb="runas"`` runs in its own security context, so the only thing the
    unelevated caller learns about it is its exit code. ``wait_ms`` of ``0`` does
    not wait at all; a negative value waits without a deadline.

    Raises:
        WinApiError: The shell refused. ``code`` is :data:`ERROR_CANCELLED` when
            the user dismissed the UAC prompt.
    """
    execute = _require("shell32", "ShellExecuteExW")

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
        return ProcessRun()
    try:
        return _process_run(handle, wait_ms)
    finally:
        _close_handle(handle)


def _process_run(handle: Any, wait_ms: int) -> ProcessRun:
    """Pid, and the exit code once the process ended, for an open handle."""
    get_pid = _require("kernel32", "GetProcessId")
    get_pid.restype = ctypes.c_ulong
    get_pid.argtypes = [ctypes.c_void_p]
    pid = int(get_pid(handle))
    if wait_ms == 0:
        return ProcessRun(pid=pid)

    wait = _require("kernel32", "WaitForSingleObject")
    wait.restype = ctypes.c_ulong
    wait.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    timeout = _INFINITE if wait_ms < 0 else ctypes.c_ulong(wait_ms).value
    if int(wait(handle, timeout)) == WAIT_TIMEOUT:
        return ProcessRun(pid=pid, timed_out=True)

    get_code = _require("kernel32", "GetExitCodeProcess")
    code = ctypes.c_ulong(0)
    get_code.restype = ctypes.c_bool
    get_code.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    if not get_code(handle, ctypes.byref(code)):
        return ProcessRun(pid=pid)
    return ProcessRun(pid=pid, exit_code=int(code.value))


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
# Process token: rights and privileges
# --------------------------------------------------------------------------- #


def _current_process_token(access: int) -> Any:
    """Open this process's token.

    Raises:
        WinApiError: Not Windows, or the token could not be opened.
    """
    open_token = _require("advapi32", "OpenProcessToken")
    current = _require("kernel32", "GetCurrentProcess")
    current.restype = ctypes.c_void_p
    current.argtypes = []
    token = ctypes.c_void_p()
    open_token.restype = ctypes.c_bool
    open_token.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p)]
    if not open_token(current(), access, ctypes.byref(token)):
        raise _last_error("OpenProcessToken")
    return token


def _token_information(token: Any, info_class: int, size: int) -> bytes:
    """Raw ``GetTokenInformation`` bytes, the buffer grown once if too small."""
    query = _require("advapi32", "GetTokenInformation")
    query.restype = ctypes.c_bool
    query.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    buffer = ctypes.create_string_buffer(size)
    needed = ctypes.c_ulong(0)
    if query(token, info_class, buffer, len(buffer), ctypes.byref(needed)):
        return bytes(buffer.raw[: needed.value or size])
    wanted = int(needed.value)
    if wanted <= size:
        raise _last_error(f"GetTokenInformation({info_class})")
    buffer = ctypes.create_string_buffer(wanted)
    if not query(token, info_class, buffer, len(buffer), ctypes.byref(needed)):
        raise _last_error(f"GetTokenInformation({info_class})")
    return bytes(buffer.raw[: needed.value or wanted])


def _token_dword(token: Any, info_class: int) -> int:
    """One ``DWORD`` out of the token."""
    raw = _token_information(token, info_class, ctypes.sizeof(ctypes.c_ulong))
    return int.from_bytes(raw[:4], "little")


def _token_integrity_level(token: Any) -> int:
    """RID of the token's integrity label.

    ``TokenIntegrityLevel`` answers with a ``TOKEN_MANDATORY_LABEL``: a pointer to
    a SID whose last sub-authority is the level. The pointer is read out of the
    buffer rather than described with a structure, because the SID itself lives at
    the far end of it and its length is not known in advance.
    """
    raw = _token_information(token, _TOKEN_INTEGRITY_LEVEL, 64)
    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    if len(raw) < pointer_size:
        raise WinApiError("TokenIntegrityLevel returned a short buffer")
    address = int.from_bytes(raw[:pointer_size], "little")
    if not address:
        raise WinApiError("TokenIntegrityLevel returned a null SID")
    count_of = _require("advapi32", "GetSidSubAuthorityCount")
    authority_of = _require("advapi32", "GetSidSubAuthority")
    count_of.restype = ctypes.POINTER(ctypes.c_ubyte)
    count_of.argtypes = [ctypes.c_void_p]
    authority_of.restype = ctypes.POINTER(ctypes.c_ulong)
    authority_of.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    sid = ctypes.c_void_p(address)
    count = int(count_of(sid).contents.value)
    if count <= 0:
        raise WinApiError("integrity SID has no sub-authorities")
    return int(authority_of(sid, count - 1).contents.value)


def process_elevation() -> ElevationInfo:
    """What this process's token says about its rights.

    Raises:
        WinApiError: Not Windows, or the token refused to answer.
    """
    token = _current_process_token(TOKEN_QUERY)
    try:
        elevated = bool(_token_dword(token, _TOKEN_ELEVATION))
        kind = _token_dword(token, _TOKEN_ELEVATION_TYPE)
        try:
            integrity = _token_integrity_level(token)
        except WinApiError as exc:
            # An unreadable label is not worth failing the whole check over: the
            # elevation flag above already decided the question.
            _log.debug("integrity level unavailable: %s", exc)
            integrity = INTEGRITY_HIGH if elevated else INTEGRITY_MEDIUM
    finally:
        _close_handle(token)
    return ElevationInfo(elevated=elevated, elevation_type=kind, integrity_level=integrity)


def enable_privilege(name: str) -> bool:
    """Enable one privilege in this process's token, returning whether it stuck.

    Administrators hold ``SeShutdownPrivilege`` but Windows keeps it *disabled* in
    the token, so ``ExitWindowsEx`` in a process that never asked for it fails with
    «access denied» — the single most common way shutdown code goes wrong.
    ``AdjustTokenPrivileges`` also reports success when it enabled only some of
    what was asked for, so the last error is checked even on a true return.

    Raises:
        WinApiError: Not Windows, or the token could not be opened or adjusted.
    """
    lookup = _require("advapi32", "LookupPrivilegeValueW")
    adjust = _require("advapi32", "AdjustTokenPrivileges")
    luid = _LUID()
    lookup.restype = ctypes.c_bool
    lookup.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.POINTER(_LUID)]
    if not lookup(None, name, ctypes.byref(luid)):
        raise _last_error(f"LookupPrivilegeValueW({name})")

    token = _current_process_token(TOKEN_QUERY | TOKEN_ADJUST_PRIVILEGES)
    try:
        privileges = _TokenPrivileges()
        privileges.PrivilegeCount = 1
        privileges.Privileges[0].Luid = luid
        privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED
        adjust.restype = ctypes.c_bool
        adjust.argtypes = [
            ctypes.c_void_p,
            ctypes.c_bool,
            ctypes.POINTER(_TokenPrivileges),
            ctypes.c_ulong,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        ctypes.set_last_error(0)
        if not adjust(token, False, ctypes.byref(privileges), 0, None, None):
            raise _last_error(f"AdjustTokenPrivileges({name})")
        return int(ctypes.GetLastError()) == 0
    finally:
        _close_handle(token)


# --------------------------------------------------------------------------- #
# Power: suspend, shut down, lock
# --------------------------------------------------------------------------- #


def set_suspend_state(*, hibernate: bool, force: bool = False, wake_events: bool = False) -> None:
    """Put the machine to sleep or into hibernation.

    ``force`` is passed through as Windows documents it — ignored on anything
    modern — and left at ``False`` so applications still get their
    ``WM_POWERBROADCAST`` and can save what they were doing.

    Raises:
        WinApiError: Not Windows, or the request was refused. Hibernation refused
            here usually means it is turned off in the system, which
            ``powercfg /a`` explains and the action layer checks first.
    """
    suspend = _require("powrprof", "SetSuspendState")
    suspend.restype = ctypes.c_bool
    suspend.argtypes = [ctypes.c_bool, ctypes.c_bool, ctypes.c_bool]
    ctypes.set_last_error(0)
    if not suspend(hibernate, force, not wake_events):
        raise _last_error(f"SetSuspendState(hibernate={hibernate})")


def exit_windows(flags: int, *, reason: int = SHUTDOWN_REASON_PLANNED) -> None:
    """``ExitWindowsEx``: log off, shut down or reboot, right now.

    The privilege must already be enabled for anything but a log-off; see
    :func:`enable_privilege`.

    Raises:
        WinApiError: Not Windows, or Windows refused — a program vetoing the
            shutdown, or the missing privilege.
    """
    exit_ex = _require("user32", "ExitWindowsEx")
    exit_ex.restype = ctypes.c_bool
    exit_ex.argtypes = [ctypes.c_uint, ctypes.c_ulong]
    if not exit_ex(flags, reason):
        raise _last_error(f"ExitWindowsEx({flags:#x})")


def initiate_shutdown(
    *,
    delay_s: int,
    reboot: bool,
    message: str = "",
    force_apps: bool = False,
    reason: int = SHUTDOWN_REASON_PLANNED,
) -> None:
    """``InitiateSystemShutdownExW``: schedule a shutdown or reboot.

    The scheduled form, unlike :func:`exit_windows`, is the one that can be taken
    back — see :func:`abort_shutdown`. It also shows Windows' own countdown
    dialog, which is a better warning than anything a voice assistant could say.

    Raises:
        WinApiError: Not Windows, or Windows refused.
    """
    initiate = _require("advapi32", "InitiateSystemShutdownExW")
    initiate.restype = ctypes.c_bool
    initiate.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_ulong,
    ]
    if not initiate(None, message or None, max(0, delay_s), force_apps, reboot, reason):
        raise _last_error(f"InitiateSystemShutdownExW(delay={delay_s}, reboot={reboot})")


def abort_shutdown() -> None:
    """``AbortSystemShutdownW``: cancel a shutdown scheduled on this machine.

    Raises:
        WinApiError: Not Windows, or there was nothing to cancel.
    """
    abort = _require("advapi32", "AbortSystemShutdownW")
    abort.restype = ctypes.c_bool
    abort.argtypes = [ctypes.c_wchar_p]
    if not abort(None):
        raise _last_error("AbortSystemShutdownW")


def lock_workstation() -> None:
    """``LockWorkStation``: the Win+L screen.

    Returns as soon as the request is queued, not once the screen is locked.

    Raises:
        WinApiError: Not Windows, or the session cannot be locked — a Remote
            Desktop session, for one.
    """
    lock = _require("user32", "LockWorkStation")
    lock.restype = ctypes.c_bool
    lock.argtypes = []
    if not lock():
        raise _last_error("LockWorkStation")


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


def display_device(device: str, index: int = 0) -> tuple[str, str]:
    """Model name and device id of the panel plugged into a display adapter.

    ``device`` is what :func:`monitor_info` reports — ``\\\\.\\DISPLAY1``. The
    answer is ``(name, id)``, for instance ``("LG ULTRAGEAR",
    "MONITOR\\GSM5B09\\{4d36e96e-...}\\0002")``, or two empty strings when
    Windows has nothing to say. Both halves are best-effort: the point is a
    string the user might actually say out loud, since ``DISPLAY1`` is not one.
    """
    enum_devices = _win_function("user32", "EnumDisplayDevicesW")
    if enum_devices is None:
        return ("", "")
    info = _DISPLAYDEVICEW()
    info.cb = ctypes.sizeof(_DISPLAYDEVICEW)
    enum_devices.restype = ctypes.c_bool
    enum_devices.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.POINTER(_DISPLAYDEVICEW),
        ctypes.c_ulong,
    ]
    if not enum_devices(device or None, index, ctypes.byref(info), 0):
        return ("", "")
    return (str(info.DeviceString).strip(), str(info.DeviceID).strip())


def windows_build() -> int:
    """Build number of the running Windows, or ``0`` elsewhere.

    Virtual desktops arrived in build 10240 and their internals moved twice
    since; the desktop actions refuse to guess below that line.
    """
    if sys.platform != "win32":
        return 0
    version = sys.getwindowsversion()
    return int(version.build)


def console_output_codepage() -> int:
    """Code page a console program writes its output in, or ``0`` elsewhere.

    Needed because the console keeps the OEM code page — 866 on a Russian
    Windows — while :mod:`locale` reports the ANSI one, 1251. Decoding ``netsh``
    output with the wrong one of those turns every SSID with a Cyrillic letter
    into mojibake, and the user cannot connect to a network they cannot name.

    Returns ``0`` rather than raising when there is no console attached: a
    windowed build has none, and the caller falls back to :func:`oem_codepage`.
    """
    entry = _win_function("kernel32", "GetConsoleOutputCP")
    if entry is None:
        return 0
    entry.restype = ctypes.c_uint
    entry.argtypes = []
    return int(entry())


def oem_codepage() -> int:
    """The system OEM code page, or ``0`` off Windows.

    What :func:`console_output_codepage` falls back to. A windowed build has no
    console of its own, so ``GetConsoleOutputCP`` answers nothing — but a console
    child it starts still gets a console, and that console gets this code page. The
    ANSI one that :mod:`locale` reports is a different number on the same machine
    (1251 against 866 in Russian), so guessing from the locale is exactly the
    mojibake this avoids.
    """
    entry = _win_function("kernel32", "GetOEMCP")
    if entry is None:
        return 0
    entry.restype = ctypes.c_uint
    entry.argtypes = []
    return int(entry())


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
# Clipboard
# --------------------------------------------------------------------------- #


def register_clipboard_format(name: str) -> int:
    """Numeric id of a named clipboard format, or ``0`` when it cannot be had.

    Registering an already registered name returns the same id, so every process
    that asks for ``"PNG"`` agrees on the number without anyone owning it.
    """
    register = _win_function("user32", "RegisterClipboardFormatW")
    if register is None:
        return 0
    register.restype = ctypes.c_uint
    register.argtypes = [ctypes.c_wchar_p]
    return int(register(name))


@contextlib.contextmanager
def _clipboard_open() -> Iterator[None]:
    """Hold the clipboard open for the body, retrying while someone else has it.

    The clipboard is one global lock for the whole desktop, and right after a
    Ctrl+C something always holds it — the source program, a manager, Windows'
    own history service. It nearly always lets go within a frame, so a refusal is
    worth retrying for ~0.15 s before it becomes an error. Without this every
    clipboard action would fail at random, which is exactly what it looks like to
    the user: «иногда работает».
    """
    open_clipboard = _require("user32", "OpenClipboard")
    close_clipboard = _require("user32", "CloseClipboard")
    open_clipboard.restype = ctypes.c_bool
    open_clipboard.argtypes = [ctypes.c_void_p]
    close_clipboard.restype = ctypes.c_bool
    close_clipboard.argtypes = []

    for attempt in range(_CLIPBOARD_TRIES):
        if open_clipboard(None):
            break
        if attempt == _CLIPBOARD_TRIES - 1:
            raise _last_error("OpenClipboard")
        time.sleep(_CLIPBOARD_RETRY_S)
    try:
        yield
    finally:
        close_clipboard()


def clipboard_set_binary(payloads: Sequence[tuple[int, bytes]]) -> None:
    """Put the same picture on the clipboard in several formats at once.

    ``payloads`` is ``(format, bytes)`` pairs, best format first — Windows keeps
    the order and a pasting program takes the first one it understands. Offering
    both :data:`CF_DIB` and the registered ``"PNG"`` is the difference between a
    screenshot that pastes into Paint and one that keeps its alpha channel in a
    browser; neither format alone covers both.

    Memory handed to ``SetClipboardData`` belongs to the system afterwards and
    must not be freed; memory it rejected still belongs to us and must be.
    """
    if not payloads:
        return

    empty_clipboard = _require("user32", "EmptyClipboard")
    set_data = _require("user32", "SetClipboardData")
    global_alloc = _require("kernel32", "GlobalAlloc")
    global_lock = _require("kernel32", "GlobalLock")
    global_unlock = _require("kernel32", "GlobalUnlock")
    global_free = _require("kernel32", "GlobalFree")

    empty_clipboard.restype = ctypes.c_bool
    empty_clipboard.argtypes = []
    set_data.restype = ctypes.c_void_p
    set_data.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    global_alloc.restype = ctypes.c_void_p
    global_alloc.argtypes = [ctypes.c_uint, ctypes.c_size_t]
    global_lock.restype = ctypes.c_void_p
    global_lock.argtypes = [ctypes.c_void_p]
    global_unlock.restype = ctypes.c_bool
    global_unlock.argtypes = [ctypes.c_void_p]
    global_free.restype = ctypes.c_void_p
    global_free.argtypes = [ctypes.c_void_p]

    with _clipboard_open():
        if not empty_clipboard():
            raise _last_error("EmptyClipboard")
        for fmt, blob in payloads:
            if not fmt or not blob:
                continue
            block = global_alloc(GMEM_MOVEABLE, len(blob))
            if not block:
                raise _last_error("GlobalAlloc")
            address = global_lock(block)
            if not address:
                global_free(block)
                raise _last_error("GlobalLock")
            try:
                ctypes.memmove(address, blob, len(blob))
            finally:
                global_unlock(block)
            if not set_data(fmt, block):
                error = _last_error("SetClipboardData")
                global_free(block)
                raise error


def clipboard_set_text(text: str) -> None:
    """Put plain text on the clipboard as :data:`CF_UNICODETEXT`.

    Windows expects the buffer NUL-terminated and counted in bytes, so the
    terminator is two of them.
    """
    clipboard_set_binary([(CF_UNICODETEXT, text.encode("utf-16-le") + b"\x00\x00")])


def clipboard_clear() -> None:
    """Empty the clipboard. What a secret paste has to do the moment it is done."""
    empty_clipboard = _require("user32", "EmptyClipboard")
    empty_clipboard.restype = ctypes.c_bool
    empty_clipboard.argtypes = []
    with _clipboard_open():
        if not empty_clipboard():
            raise _last_error("EmptyClipboard")


def clipboard_sequence_number() -> int:
    """Counter Windows bumps on every clipboard change. ``0`` when unavailable.

    Cheaper than reading the clipboard and needs no lock, so a monitor can tell
    «nothing happened» from «read it again» without fighting for the handle.
    """
    counter = _win_function("user32", "GetClipboardSequenceNumber")
    if counter is None:
        return 0
    counter.restype = ctypes.c_uint
    counter.argtypes = []
    return int(counter())


@dataclass(frozen=True, slots=True)
class ClipboardData:
    """One consistent look at the clipboard, taken while it was held open.

    Everything comes from a single open/close cycle on purpose: asking for the
    formats, then the text, then the pictures would be three separate locks with
    a different clipboard possibly behind each one.
    """

    formats: tuple[int, ...] = ()
    text: str = ""
    files: tuple[str, ...] = ()
    blobs: Mapping[int, bytes] = field(default_factory=dict)
    sequence: int = 0

    def has(self, fmt: int) -> bool:
        """Whether the clipboard offered that format."""
        return fmt in self.formats


def _read_blob(handle: Any, *, limit: int = 0) -> bytes:
    """Copy a global memory block out of the clipboard, optionally truncated."""
    global_lock = _require("kernel32", "GlobalLock")
    global_unlock = _require("kernel32", "GlobalUnlock")
    global_size = _require("kernel32", "GlobalSize")
    global_lock.restype = ctypes.c_void_p
    global_lock.argtypes = [ctypes.c_void_p]
    global_unlock.restype = ctypes.c_bool
    global_unlock.argtypes = [ctypes.c_void_p]
    global_size.restype = ctypes.c_size_t
    global_size.argtypes = [ctypes.c_void_p]

    size = int(global_size(handle))
    if size <= 0:
        return b""
    if limit and size > limit:
        size = limit
    address = global_lock(handle)
    if not address:
        return b""
    try:
        return ctypes.string_at(address, size)
    finally:
        global_unlock(handle)


def _read_dropped_files(handle: Any) -> tuple[str, ...]:
    """Names behind a :data:`CF_HDROP` handle, or ``()`` when shell32 refuses."""
    query = _win_function("shell32", "DragQueryFileW")
    if query is None:
        return ()
    query.restype = ctypes.c_uint
    query.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_wchar_p, ctypes.c_uint]

    count = int(query(handle, 0xFFFFFFFF, None, 0))
    names: list[str] = []
    for index in range(min(count, _MAX_DROPPED_FILES)):
        length = int(query(handle, index, None, 0))
        if length <= 0:
            continue
        buffer = ctypes.create_unicode_buffer(length + 1)
        if query(handle, index, buffer, length + 1):
            names.append(buffer.value)
    return tuple(names)


def read_clipboard(blobs: Sequence[int] = ()) -> ClipboardData:
    """Read the clipboard once: which formats it holds, its text, its files.

    ``blobs`` names extra formats whose raw bytes the caller wants — the markers
    password managers set to ask monitors to look away are DWORD-sized formats
    read this way. Their payload is capped: a caller asking for a picture format
    by mistake should not pull megabytes into memory.
    """
    enum_formats = _require("user32", "EnumClipboardFormats")
    get_data = _require("user32", "GetClipboardData")
    enum_formats.restype = ctypes.c_uint
    enum_formats.argtypes = [ctypes.c_uint]
    get_data.restype = ctypes.c_void_p
    get_data.argtypes = [ctypes.c_uint]

    wanted = {fmt for fmt in blobs if fmt}
    formats: list[int] = []
    payloads: dict[int, bytes] = {}
    text = ""
    files: tuple[str, ...] = ()

    with _clipboard_open():
        current = int(enum_formats(0))
        while current and len(formats) < _MAX_CLIPBOARD_FORMATS:
            formats.append(current)
            current = int(enum_formats(current))
        if CF_UNICODETEXT in formats:
            handle = get_data(CF_UNICODETEXT)
            if handle:
                raw = _read_blob(handle, limit=_MAX_CLIPBOARD_TEXT * 2)
                text = raw.decode("utf-16-le", "replace").split("\0", 1)[0]
        if CF_HDROP in formats:
            handle = get_data(CF_HDROP)
            if handle:
                files = _read_dropped_files(handle)
        for fmt in sorted(wanted & set(formats)):
            handle = get_data(fmt)
            if handle:
                payloads[fmt] = _read_blob(handle, limit=_MAX_CLIPBOARD_BLOB)

    return ClipboardData(
        formats=tuple(formats),
        text=text,
        files=files,
        blobs=payloads,
        sequence=clipboard_sequence_number(),
    )


def add_clipboard_format_listener(hwnd: int) -> None:
    """Ask Windows to post :data:`WM_CLIPBOARDUPDATE` to ``hwnd``.

    The modern replacement for the clipboard viewer chain: no chain to join, no
    neighbour to forward to, and nothing breaks when another listener crashes.
    """
    listen = _require("user32", "AddClipboardFormatListener")
    listen.restype = ctypes.c_bool
    listen.argtypes = [ctypes.c_void_p]
    if not listen(_handle(hwnd)):
        raise _last_error("AddClipboardFormatListener")


def remove_clipboard_format_listener(hwnd: int) -> None:
    """Stop the notifications :func:`add_clipboard_format_listener` started."""
    stop = _win_function("user32", "RemoveClipboardFormatListener")
    if stop is None:
        return
    stop.restype = ctypes.c_bool
    stop.argtypes = [ctypes.c_void_p]
    stop(_handle(hwnd))


class _MSG(ctypes.Structure):
    _fields_ = (
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("wParam", ctypes.c_void_p),
        ("lParam", ctypes.c_void_p),
        ("time", ctypes.c_ulong),
        ("pt", _POINT),
    )


class _WNDCLASSW(ctypes.Structure):
    _fields_ = (
        ("style", ctypes.c_uint),
        ("lpfnWndProc", ctypes.c_void_p),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", ctypes.c_void_p),
        ("hIcon", ctypes.c_void_p),
        ("hCursor", ctypes.c_void_p),
        ("hbrBackground", ctypes.c_void_p),
        ("lpszMenuName", ctypes.c_wchar_p),
        ("lpszClassName", ctypes.c_wchar_p),
    )


class MessageWindow:
    """A window with no pixels, existing only to be sent messages.

    ``AddClipboardFormatListener`` needs a window handle, and a background thread
    has none — so it makes one whose parent is ``HWND_MESSAGE``. Such a window
    never appears on screen, in the taskbar or in Alt+Tab, and it is not painted,
    which is why it costs nothing to keep alive for the whole session.

    The window procedure is Windows' own ``DefWindowProcW``: every message Ayris
    cares about is *posted*, so it arrives in the thread's queue and :meth:`pump`
    reads it there. Nothing has to travel through a Python callback, and a bug in
    the handler therefore cannot corrupt the stack of a WinAPI call.

    :meth:`create` and :meth:`pump` must run on the same thread. :meth:`stop` is
    the exception and may be called from any of them.
    """

    def __init__(self, class_name: str, on_message: Callable[[int], None]) -> None:
        self._class_name = class_name
        self._on_message = on_message
        self._hwnd = 0
        self._atom = 0

    @property
    def hwnd(self) -> int:
        """Handle of the live window, or ``0`` before :meth:`create`."""
        return self._hwnd

    def create(self) -> None:
        """Register the class and make the window. Idempotent."""
        if self._hwnd:
            return
        register = _require("user32", "RegisterClassW")
        create = _require("user32", "CreateWindowExW")
        default_proc = _require("user32", "DefWindowProcW")
        module_handle = _require("kernel32", "GetModuleHandleW")
        register.restype = ctypes.c_ushort
        register.argtypes = [ctypes.c_void_p]
        create.restype = ctypes.c_void_p
        create.argtypes = [
            ctypes.c_uint,
            ctypes.c_wchar_p,
            ctypes.c_wchar_p,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        module_handle.restype = ctypes.c_void_p
        module_handle.argtypes = [ctypes.c_wchar_p]

        instance = module_handle(None)
        window_class = _WNDCLASSW()
        window_class.lpfnWndProc = ctypes.cast(default_proc, ctypes.c_void_p)
        window_class.hInstance = instance
        window_class.lpszClassName = self._class_name
        atom = int(register(ctypes.byref(window_class)))
        if not atom:
            error = _last_error("RegisterClassW")
            # 1410 is ERROR_CLASS_ALREADY_EXISTS: a previous window of ours was
            # torn down without unregistering, and the class is still usable.
            if error.code != _ERROR_CLASS_ALREADY_EXISTS:
                raise error
        else:
            self._atom = atom

        hwnd = create(
            0,
            self._class_name,
            self._class_name,
            0,
            0,
            0,
            0,
            0,
            _handle(_HWND_MESSAGE),
            None,
            instance,
            None,
        )
        if not hwnd:
            raise _last_error("CreateWindowExW")
        self._hwnd = int(hwnd)

    def pump(self) -> None:
        """Read messages until :meth:`stop`. Blocks; call on the owning thread."""
        if not self._hwnd:
            raise WinApiError("MessageWindow.pump called before create")
        get_message = _require("user32", "GetMessageW")
        translate = _require("user32", "TranslateMessage")
        dispatch = _require("user32", "DispatchMessageW")
        get_message.restype = ctypes.c_int
        get_message.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
        translate.restype = ctypes.c_bool
        translate.argtypes = [ctypes.c_void_p]
        dispatch.restype = ctypes.c_void_p
        dispatch.argtypes = [ctypes.c_void_p]

        message = _MSG()
        pointer = ctypes.byref(message)
        while True:
            got = int(get_message(pointer, _handle(self._hwnd), 0, 0))
            if got <= 0:
                # 0 is WM_QUIT, -1 an invalid handle — the window is gone either way.
                return
            code = int(message.message)
            if code == _WM_AYRIS_STOP:
                return
            translate(pointer)
            dispatch(pointer)
            try:
                self._on_message(code)
            except Exception:
                _log.exception("message window handler failed on 0x%04X", code)

    def stop(self) -> None:
        """Make :meth:`pump` return. Safe from another thread."""
        if not self._hwnd:
            return
        post = _win_function("user32", "PostMessageW")
        if post is None:
            return
        post.restype = ctypes.c_bool
        post.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
        post(_handle(self._hwnd), _WM_AYRIS_STOP, None, None)

    def close(self) -> None:
        """Destroy the window and forget the class. Idempotent."""
        if self._hwnd:
            destroy = _win_function("user32", "DestroyWindow")
            if destroy is not None:
                destroy.restype = ctypes.c_bool
                destroy.argtypes = [ctypes.c_void_p]
                destroy(_handle(self._hwnd))
            self._hwnd = 0
        if self._atom:
            unregister = _win_function("user32", "UnregisterClassW")
            if unregister is not None:
                unregister.restype = ctypes.c_bool
                unregister.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p]
                unregister(self._class_name, None)
            self._atom = 0


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
