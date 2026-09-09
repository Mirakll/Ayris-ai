"""Thin, mockable Win32 global-hotkey and keyboard-capture wrapper.

All RegisterHotKey/UnregisterHotKey calls happen on the thread that owns this
object and its message loop.  The manager above it contains policy only, which
keeps the behaviour testable on non-interactive CI runners.
"""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
from collections.abc import Callable
from ctypes import wintypes
from typing import Final

from ayris.actions.input.keys import KEYS
from ayris.core.errors import HotkeyError
from ayris.utils.hotkeys import Hotkey

__all__ = [
    "ERROR_HOTKEY_ALREADY_REGISTERED",
    "HotkeyBackendUnavailable",
    "HotkeyRegistrationError",
    "WinApiBackend",
]

WM_HOTKEY: Final = 0x0312
WM_APP_COMMAND: Final = 0x8001
WM_QUIT: Final = 0x0012
WM_KEYDOWN: Final = 0x0100
WM_KEYUP: Final = 0x0101
WM_SYSKEYDOWN: Final = 0x0104
WM_SYSKEYUP: Final = 0x0105
WM_CLOSE: Final = 0x0010
WM_DESTROY: Final = 0x0002
WH_KEYBOARD_LL: Final = 13
HC_ACTION: Final = 0
ERROR_HOTKEY_ALREADY_REGISTERED: Final = 1409
MOD_ALT: Final = 0x0001
MOD_CONTROL: Final = 0x0002
MOD_SHIFT: Final = 0x0004
MOD_WIN: Final = 0x0008
MOD_NOREPEAT: Final = 0x4000
VK_ESCAPE: Final = 0x1B
HWND_MESSAGE: Final = -3


class HotkeyBackendUnavailable(HotkeyError):  # noqa: N818 - public capability state
    """The selected backend cannot run on this machine."""

    default_user_message = "Глобальные горячие клавиши сейчас недоступны."


class HotkeyRegistrationError(HotkeyError):
    """Windows refused one registration."""

    def __init__(self, hotkey: Hotkey, owner: str, error_code: int) -> None:
        if error_code == ERROR_HOTKEY_ALREADY_REGISTERED:
            message = (
                f"Сочетание {hotkey.label_ru} уже занято Windows или другой программой. "
                "Выберите другое сочетание."
            )
        else:
            message = (
                f"Не удалось зарегистрировать {hotkey.label_ru} для «{owner}» "
                f"(ошибка Windows {error_code})."
            )
        super().__init__(
            f"RegisterHotKey({hotkey.canonical}, owner={owner!r}) failed: {error_code}",
            user_message=message,
        )
        self.hotkey = hotkey
        self.owner = owner
        self.error_code = error_code


if sys.platform == "win32":
    ULONG_PTR = wintypes.WPARAM

    class KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - ctypes requires this exact class attribute
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        ]

    LowLevelKeyboardProc = ctypes.WINFUNCTYPE(
        wintypes.LPARAM, wintypes.INT, wintypes.WPARAM, wintypes.LPARAM
    )
    WindowProc = ctypes.WINFUNCTYPE(
        wintypes.LPARAM, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
    )

    class WNDCLASSEXW(ctypes.Structure):
        _fields_ = [  # noqa: RUF012 - ctypes requires this exact class attribute
            ("cbSize", wintypes.UINT),
            ("style", wintypes.UINT),
            ("lpfnWndProc", WindowProc),
            ("cbClsExtra", ctypes.c_int),
            ("cbWndExtra", ctypes.c_int),
            ("hInstance", wintypes.HINSTANCE),
            ("hIcon", wintypes.HICON),
            ("hCursor", wintypes.HANDLE),
            ("hbrBackground", wintypes.HBRUSH),
            ("lpszMenuName", wintypes.LPCWSTR),
            ("lpszClassName", wintypes.LPCWSTR),
            ("hIconSm", wintypes.HICON),
        ]


class WinApiBackend:
    """RegisterHotKey message pump plus a temporary low-level capture hook."""

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise HotkeyBackendUnavailable(
                "WinAPI hotkeys requested off Windows",
                user_message="Глобальные горячие клавиши работают только в Windows.",
            )
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._thread_id = 0
        self._window = 0
        self._window_class = f"AyrisHotkeys-{id(self):x}"
        self._window_proc: object | None = None
        self._callback: Callable[[int], None] | None = None
        self._capture_callback: Callable[[int, bool], bool] | None = None
        self._hook = 0
        self._hook_proc: object | None = None
        self._commands: queue.SimpleQueue[Callable[[], None]] = queue.SimpleQueue()
        self._ready = threading.Event()
        self._startup_error: HotkeyBackendUnavailable | None = None

    def run(self, callback: Callable[[int], None]) -> None:
        """Own a message-only window and its queue until :meth:`stop`."""
        self._startup_error = None
        self._callback = callback
        self._thread_id = int(self._kernel32.GetCurrentThreadId())
        message = wintypes.MSG()
        try:
            self._create_message_window()
        except HotkeyBackendUnavailable as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        while self._user32.GetMessageW(ctypes.byref(message), None, 0, 0) > 0:
            self._user32.TranslateMessage(ctypes.byref(message))
            self._user32.DispatchMessageW(ctypes.byref(message))
        self._remove_capture_hook()
        self._window = 0
        self._window_proc = None

    def _create_message_window(self) -> None:
        hinstance = self._kernel32.GetModuleHandleW(None)

        def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
            if message == WM_HOTKEY and self._callback is not None:
                self._callback(int(wparam))
                return 0
            if message == WM_APP_COMMAND:
                while not self._commands.empty():
                    self._commands.get()()
                return 0
            if message == WM_CLOSE:
                self._user32.DestroyWindow(hwnd)
                return 0
            if message == WM_DESTROY:
                self._user32.PostQuitMessage(0)
                return 0
            return int(self._user32.DefWindowProcW(hwnd, message, wparam, lparam))

        self._window_proc = WindowProc(window_proc)
        window_class = WNDCLASSEXW(
            cbSize=ctypes.sizeof(WNDCLASSEXW),
            lpfnWndProc=self._window_proc,
            hInstance=hinstance,
            lpszClassName=self._window_class,
        )
        if not self._user32.RegisterClassExW(ctypes.byref(window_class)):
            raise HotkeyBackendUnavailable(
                f"RegisterClassExW failed: {ctypes.get_last_error()}",
                user_message="Не удалось создать обработчик глобальных горячих клавиш.",
            )
        self._user32.CreateWindowExW.restype = wintypes.HWND
        window = self._user32.CreateWindowExW(
            0,
            self._window_class,
            "Ayris global hotkeys",
            0,
            0,
            0,
            0,
            0,
            ctypes.c_void_p(HWND_MESSAGE),
            None,
            hinstance,
            None,
        )
        if not window:
            raise HotkeyBackendUnavailable(
                f"CreateWindowExW failed: {ctypes.get_last_error()}",
                user_message="Не удалось создать обработчик глобальных горячих клавиш.",
            )
        self._window = int(window)

    def wait_ready(self, timeout: float) -> bool:
        ready = self._ready.wait(timeout)
        if self._startup_error is not None:
            raise self._startup_error
        return ready and bool(self._window)

    def invoke(self, command: Callable[[], None]) -> bool:
        """Queue callable execution onto the message-loop thread."""
        if not self._window:
            return False
        self._commands.put(command)
        return bool(self._user32.PostMessageW(self._window, WM_APP_COMMAND, 0, 0))

    def register(self, identifier: int, hotkey: Hotkey, owner: str) -> None:
        modifiers = MOD_NOREPEAT
        modifiers |= MOD_CONTROL if hotkey.ctrl else 0
        modifiers |= MOD_ALT if hotkey.alt else 0
        modifiers |= MOD_SHIFT if hotkey.shift else 0
        modifiers |= MOD_WIN if hotkey.win else 0
        if not self._user32.RegisterHotKey(
            self._window, identifier, modifiers, KEYS[hotkey.key].vk
        ):
            raise HotkeyRegistrationError(hotkey, owner, int(ctypes.get_last_error()))

    def unregister(self, identifier: int) -> None:
        self._user32.UnregisterHotKey(self._window, identifier)

    def key_down(self, hotkey: Hotkey) -> bool:
        return bool(self._user32.GetAsyncKeyState(KEYS[hotkey.key].vk) & 0x8000)

    def start_capture(self, callback: Callable[[int, bool], bool]) -> None:
        self._capture_callback = callback

        def hook(code: int, wparam: int, lparam: int) -> int:
            if code == HC_ACTION and self._capture_callback is not None:
                pressed = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
                released = wparam in (WM_KEYUP, WM_SYSKEYUP)
                if pressed or released:
                    data = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                    if self._capture_callback(int(data.vkCode), pressed):
                        return 1
            return int(self._user32.CallNextHookEx(self._hook, code, wparam, lparam))

        self._hook_proc = LowLevelKeyboardProc(hook)
        self._hook = int(
            self._user32.SetWindowsHookExW(
                WH_KEYBOARD_LL, self._hook_proc, self._kernel32.GetModuleHandleW(None), 0
            )
        )
        if not self._hook:
            self._hook_proc = None
            raise HotkeyBackendUnavailable(
                f"SetWindowsHookExW failed: {ctypes.get_last_error()}",
                user_message="Не удалось начать захват сочетания клавиш.",
            )

    def stop_capture(self) -> None:
        self._remove_capture_hook()

    def _remove_capture_hook(self) -> None:
        if self._hook:
            self._user32.UnhookWindowsHookEx(self._hook)
        self._hook = 0
        self._hook_proc = None
        self._capture_callback = None

    def stop(self) -> None:
        if self._window:
            self._user32.PostMessageW(self._window, WM_CLOSE, 0, 0)
        elif self._thread_id:
            self._user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
