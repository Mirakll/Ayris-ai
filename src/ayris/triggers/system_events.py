"""Low-overhead Windows event monitor plus platform-neutral event values."""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from ayris.core.events import Event, EventBus
from ayris.utils.logger import get_logger

_log = get_logger(__name__)

PROCESS_STARTED: Final = "process_started"
PROCESS_STOPPED: Final = "process_stopped"
ACTIVE_WINDOW_CHANGED: Final = "active_window_changed"
DEVICE_CONNECTED: Final = "device_connected"
DEVICE_DISCONNECTED: Final = "device_disconnected"
POWER_CHANGED: Final = "power_changed"
FULLSCREEN_ENTERED: Final = "fullscreen_entered"
FULLSCREEN_EXITED: Final = "fullscreen_exited"
SYSTEM_EVENT_NAMES: Final = frozenset(
    {
        PROCESS_STARTED,
        PROCESS_STOPPED,
        ACTIVE_WINDOW_CHANGED,
        DEVICE_CONNECTED,
        DEVICE_DISCONNECTED,
        POWER_CHANGED,
        FULLSCREEN_ENTERED,
        FULLSCREEN_EXITED,
    }
)


@dataclass(frozen=True, slots=True)
class SystemEvent(Event):
    kind: str
    process: str = ""
    title: str = ""
    device_type: str = ""
    device_id: str = ""
    on_ac: bool | None = None
    fullscreen: bool | None = None


@dataclass(frozen=True, slots=True)
class MonitorStats:
    process_scans: int
    scan_seconds: float
    max_scan_seconds: float
    poll_interval: float

    @property
    def average_scan_seconds(self) -> float:
        return self.scan_seconds / self.process_scans if self.process_scans else 0.0


class AdaptiveProcessPoller:
    """Diff process snapshots, slowing down while the machine is idle."""

    def __init__(
        self,
        callback: Callable[[SystemEvent], None],
        snapshot: Callable[[], Iterable[tuple[int, str]]],
        *,
        minimum: float = 1.0,
        maximum: float = 15.0,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._callback = callback
        self._snapshot = snapshot
        self.minimum = minimum
        self.maximum = maximum
        self.interval = minimum
        self._clock = monotonic
        self._known: dict[int, str] | None = None
        self.scans = 0
        self.scan_seconds = 0.0
        self.max_scan_seconds = 0.0

    def scan(self) -> bool:
        started = self._clock()
        current = dict(self._snapshot())
        elapsed = max(0.0, self._clock() - started)
        self.scans += 1
        self.scan_seconds += elapsed
        self.max_scan_seconds = max(self.max_scan_seconds, elapsed)
        if self._known is None:
            self._known = current
            return False
        started_ids = current.keys() - self._known.keys()
        stopped_ids = self._known.keys() - current.keys()
        changed = bool(started_ids or stopped_ids)
        for process_id in started_ids:
            self._callback(SystemEvent(PROCESS_STARTED, process=current[process_id]))
        for process_id in stopped_ids:
            self._callback(SystemEvent(PROCESS_STOPPED, process=self._known[process_id]))
        self._known = current
        self.interval = self.minimum if changed else min(self.maximum, self.interval * 1.5)
        return changed

    @property
    def stats(self) -> MonitorStats:
        return MonitorStats(self.scans, self.scan_seconds, self.max_scan_seconds, self.interval)


class SystemEventMonitor:
    """Own one Windows message loop and monitor only currently needed sources."""

    def __init__(
        self,
        bus: EventBus,
        *,
        process_snapshot: Callable[[], Iterable[tuple[int, str]]] | None = None,
    ) -> None:
        self._bus = bus
        self._wanted: frozenset[str] = frozenset()
        self._process_snapshot = process_snapshot or _process_snapshot
        self._poller = AdaptiveProcessPoller(bus.publish, self._process_snapshot)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._window_thread: threading.Thread | None = None

    @property
    def stats(self) -> MonitorStats:
        return self._poller.stats

    @property
    def subscriptions(self) -> frozenset[str]:
        return self._wanted

    def replace_subscriptions(self, names: Iterable[str]) -> None:
        wanted = frozenset(names) & SYSTEM_EVENT_NAMES
        self._wanted = wanted
        if wanted and self._thread is None:
            self.start()
        elif not wanted:
            self.stop()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ayris-system-events", daemon=True)
        self._thread.start()
        if sys.platform == "win32":
            self._window_thread = threading.Thread(
                target=self._windows_loop, name="ayris-win-events", daemon=True
            )
            self._window_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._thread, self._window_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(2.0)
        self._thread = None
        self._window_thread = None

    def emit(self, event: SystemEvent) -> None:
        """Test seam and WinAPI callback endpoint."""
        if event.kind in self._wanted:
            self._bus.publish(event)

    def _run(self) -> None:
        while not self._stop.is_set():
            if {PROCESS_STARTED, PROCESS_STOPPED} & self._wanted:
                try:
                    self._poller.scan()
                except Exception:
                    _log.exception("не удалось получить список процессов")
            self._stop.wait(self._poller.interval)

    def _windows_loop(self) -> None:  # pragma: no cover - exercised on Windows CI
        try:
            _WindowsMessageLoop(self.emit, lambda: self._wanted, self._stop).run()
        except Exception:
            _log.exception("цикл системных событий Windows завершился с ошибкой")


def _process_snapshot() -> Iterable[tuple[int, str]]:
    """No dependency process snapshot; opening Query handles is unnecessary."""
    if sys.platform != "win32":
        return ((0, Path(sys.executable).name),)
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == wintypes.HANDLE(-1).value:
        return ()

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    result: list[tuple[int, str]] = []
    try:
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            result.append((entry.th32ProcessID, entry.szExeFile))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return result


class _WindowsMessageLoop:
    """Hidden window for device/power messages and a foreground WinEvent hook."""

    def __init__(
        self,
        emit: Callable[[SystemEvent], None],
        wanted: Callable[[], frozenset[str]],
        stop: threading.Event,
    ) -> None:
        self.emit = emit
        self.wanted = wanted
        self.stop = stop
        self._last_fullscreen = False
        self._callbacks: list[Any] = []

    def run(self) -> None:  # pragma: no cover - Windows only
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ole32 = ctypes.OleDLL("ole32")
        ole32.CoInitializeEx(None, 0x2)
        lresult = ctypes.c_ssize_t
        window_proc_type = ctypes.WINFUNCTYPE(
            lresult, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )
        event_proc_type = ctypes.WINFUNCTYPE(
            None,
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.HWND,
            wintypes.LONG,
            wintypes.LONG,
            wintypes.DWORD,
            wintypes.DWORD,
        )

        def foreground(
            _hook: Any,
            _event: int,
            hwnd: int,
            _object: int,
            _child: int,
            _thread: int,
            _time: int,
        ) -> None:
            self._foreground(user32, hwnd)

        callback = event_proc_type(foreground)

        def window_proc(hwnd: int, message: int, wparam: int, lparam: int) -> int:
            if message == 0x0219:  # WM_DEVICECHANGE
                if wparam in {0x8000, 0x8004}:  # arrival / removal complete
                    kind = DEVICE_CONNECTED if wparam == 0x8000 else DEVICE_DISCONNECTED
                    device_type, device_id = _device_payload(lparam)
                    self.emit(SystemEvent(kind, device_type=device_type, device_id=device_id))
                return 1
            if message == 0x0218:  # WM_POWERBROADCAST
                self.emit(SystemEvent(POWER_CHANGED, on_ac=_on_ac_power()))
                return 1
            return int(user32.DefWindowProcW(hwnd, message, wparam, lparam))

        window_callback = window_proc_type(window_proc)
        self._callbacks.extend((callback, window_callback))

        class WindowClass(ctypes.Structure):
            _fields_: ClassVar[list[tuple[str, Any]]] = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", window_proc_type),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
        kernel32.GetModuleHandleW.restype = wintypes.HMODULE
        user32.RegisterClassW.argtypes = [ctypes.POINTER(WindowClass)]
        user32.RegisterClassW.restype = wintypes.ATOM
        user32.CreateWindowExW.argtypes = [
            wintypes.DWORD,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.DWORD,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.HWND,
            wintypes.HMENU,
            wintypes.HINSTANCE,
            wintypes.LPVOID,
        ]
        user32.CreateWindowExW.restype = wintypes.HWND
        user32.DefWindowProcW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.DefWindowProcW.restype = lresult
        user32.SetWinEventHook.argtypes = [
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HMODULE,
            event_proc_type,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        user32.SetWinEventHook.restype = wintypes.HANDLE

        class_name = f"AyrisSystemEvents{threading.get_ident()}"
        module = kernel32.GetModuleHandleW(None)
        window_class = WindowClass(0, window_callback, 0, 0, module, 0, 0, 0, None, class_name)
        atom = user32.RegisterClassW(ctypes.byref(window_class))
        if not atom:
            raise ctypes.WinError(ctypes.get_last_error())
        hwnd = user32.CreateWindowExW(
            0, class_name, "Ayris system events", 0x80000000, 0, 0, 0, 0, 0, 0, module, None
        )
        if not hwnd:
            user32.UnregisterClassW(class_name, module)
            raise ctypes.WinError(ctypes.get_last_error())
        notification = _register_device_notifications(user32, hwnd)
        hook = user32.SetWinEventHook(3, 3, None, callback, 0, 0, 0x0000 | 0x0002)
        try:
            message = wintypes.MSG()
            while not self.stop.is_set():
                while user32.PeekMessageW(ctypes.byref(message), 0, 0, 0, 1):
                    user32.TranslateMessage(ctypes.byref(message))
                    user32.DispatchMessageW(ctypes.byref(message))
                self.stop.wait(0.05)
        finally:
            if hook:
                user32.UnhookWinEvent(hook)
            if notification:
                user32.UnregisterDeviceNotification(notification)
            user32.DestroyWindow(hwnd)
            user32.UnregisterClassW(class_name, module)
            ole32.CoUninitialize()

    def _foreground(self, user32: Any, hwnd: int) -> None:  # pragma: no cover - Windows only
        import ctypes

        length = user32.GetWindowTextLengthW(hwnd)
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        process = _window_process_name(user32, hwnd)
        self.emit(SystemEvent(ACTIVE_WINDOW_CHANGED, process=process, title=buffer.value))
        fullscreen = _is_fullscreen(user32, hwnd)
        if fullscreen != self._last_fullscreen:
            self._last_fullscreen = fullscreen
            self.emit(
                SystemEvent(
                    FULLSCREEN_ENTERED if fullscreen else FULLSCREEN_EXITED,
                    process=process,
                    title=buffer.value,
                    fullscreen=fullscreen,
                )
            )


def _register_device_notifications(user32: Any, hwnd: int) -> int:
    """Ask Windows for every device-interface arrival/removal notification."""
    import ctypes
    from ctypes import wintypes

    class DeviceInterface(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("size", wintypes.DWORD),
            ("device_type", wintypes.DWORD),
            ("reserved", wintypes.DWORD),
            ("class_guid", ctypes.c_byte * 16),
            ("name", wintypes.WCHAR),
        ]

    interface = DeviceInterface()
    interface.size = ctypes.sizeof(interface)
    interface.device_type = 5  # DBT_DEVTYP_DEVICEINTERFACE
    user32.RegisterDeviceNotificationW.argtypes = [wintypes.HANDLE, wintypes.LPVOID, wintypes.DWORD]
    user32.RegisterDeviceNotificationW.restype = wintypes.HANDLE
    handle = user32.RegisterDeviceNotificationW(
        hwnd, ctypes.byref(interface), 0x00000004  # DEVICE_NOTIFY_ALL_INTERFACE_CLASSES
    )
    return int(handle or 0)


def _on_ac_power() -> bool | None:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    class Status(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("ACLineStatus", wintypes.BYTE),
            ("rest", wintypes.BYTE * 11),
        ]

    status = Status()
    if not ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return None if status.ACLineStatus == 255 else status.ACLineStatus == 1


def _device_payload(lparam: int) -> tuple[str, str]:  # pragma: no cover - Windows only
    if not lparam:
        return "device", ""
    import ctypes
    from ctypes import wintypes

    class Header(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("size", wintypes.DWORD),
            ("device_type", wintypes.DWORD),
            ("reserved", wintypes.DWORD),
        ]

    header = ctypes.cast(lparam, ctypes.POINTER(Header)).contents
    names = {1: "oem", 2: "volume", 3: "port", 5: "device_interface", 6: "handle"}
    device_type = names.get(header.device_type, f"type_{header.device_type}")
    if header.device_type != 5 or header.size <= ctypes.sizeof(Header) + 16:
        return device_type, ""
    name_offset = ctypes.sizeof(Header) + 16  # DEV_BROADCAST_DEVICEINTERFACE class GUID
    name_pointer = lparam + name_offset
    return device_type, ctypes.wstring_at(name_pointer)


def _window_process_name(user32: Any, hwnd: int) -> str:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    process_id = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
    for pid, name in _process_snapshot():
        if pid == process_id.value:
            return name
    return ""


def _is_fullscreen(user32: Any, hwnd: int) -> bool:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    if not hwnd or user32.IsIconic(hwnd):
        return False
    window = wintypes.RECT()
    monitor = user32.MonitorFromWindow(hwnd, 2)

    class MonitorInfo(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    info = MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    return bool(
        user32.GetWindowRect(hwnd, ctypes.byref(window))
        and user32.GetMonitorInfoW(monitor, ctypes.byref(info))
        and window.left <= info.rcMonitor.left
        and window.top <= info.rcMonitor.top
        and window.right >= info.rcMonitor.right
        and window.bottom >= info.rcMonitor.bottom
    )
