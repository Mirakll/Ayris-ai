"""Задача 63: доп. покрытие ``ayris.utils.winapi`` и ``ayris.utils.monitors``.

Тесты не делают ни одного настоящего вызова WinAPI: точка входа каждой обёртки
подменяется фейком через ``_require``/``_win_function``, а функции ``monitors``
подменой соседних функций ``winapi``. Ни окон, ни мониторов, ни процессов, ни
буфера обмена — всё живёт в памяти теста, поэтому набор мгновенный и без
предупреждений.
"""

from __future__ import annotations

import ctypes

import pytest

from ayris.utils import monitors, winapi

pytestmark = pytest.mark.unit


class FakeWinFn:
    """Фейковая точка входа: держит ``restype``/``argtypes`` и зовёт ``side_effect``."""

    def __init__(self, result: object = 0, side_effect: object = None) -> None:
        self.result = result
        self.side_effect = side_effect
        self.calls: list[tuple[object, ...]] = []
        self.restype: object = None
        self.argtypes: object = None

    def __call__(self, *args: object) -> object:
        self.calls.append(args)
        if self.side_effect is not None:
            return self.side_effect(*args)  # type: ignore[operator]
        return self.result


class FakeWinApi:
    """Диспетчер точек входа с ключом ``"library.function"``."""

    def __init__(self) -> None:
        self._entries: dict[str, FakeWinFn] = {}
        self._absent: set[str] = set()

    def stub(self, name: str, result: object = 0, side_effect: object = None) -> FakeWinFn:
        entry = FakeWinFn(result, side_effect)
        self._entries[name] = entry
        return entry

    def absent(self, *names: str) -> None:
        self._absent.update(names)

    def require(self, library: str, function: str) -> FakeWinFn:
        name = f"{library}.{function}"
        if name in self._absent:
            raise winapi.WinApiError(f"{name} absent in test")
        return self._entries.setdefault(name, FakeWinFn())

    def win_function(self, library: str, function: str) -> FakeWinFn | None:
        name = f"{library}.{function}"
        if name in self._absent:
            return None
        return self._entries.setdefault(name, FakeWinFn())


@pytest.fixture
def api(monkeypatch):
    """Подменяет оба сита поиска точек входа фейковым диспетчером."""
    fake = FakeWinApi()
    monkeypatch.setattr(winapi, "_require", fake.require)
    monkeypatch.setattr(winapi, "_win_function", fake.win_function)
    return fake


def _out(byref_obj: object, struct_type: object) -> object:
    """Достать из ``byref``-аргумента структуру, чтобы записать в неё ответ."""
    return ctypes.cast(byref_obj, ctypes.POINTER(struct_type)).contents  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Точки входа и последняя ошибка
# --------------------------------------------------------------------------- #


class TestLookupSeam:
    def test_win_function_none_without_windll(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Без ``ctypes.windll`` точка входа не находится, а не падает."""
        monkeypatch.setattr(winapi.ctypes, "windll", None, raising=False)
        assert winapi._win_function("user32", "IsWindow") is None

    def test_win_function_none_when_lookup_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Отсутствующая библиотека — тоже ``None`` (getattr бросает)."""
        monkeypatch.setattr(winapi.ctypes, "windll", object(), raising=False)
        assert winapi._win_function("nope", "AlsoNope") is None

    def test_require_off_windows_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.sys, "platform", "linux")
        with pytest.raises(winapi.WinApiError, match="not running on Windows"):
            winapi._require("user32", "IsWindow")

    def test_require_raises_when_entry_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "_win_function", lambda _lib, _fn: None)
        with pytest.raises(winapi.WinApiError, match="not present"):
            winapi._require("user32", "Ghost")

    def test_last_error_without_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.sys, "platform", "win32")
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 0)
        error = winapi._last_error("DoThing")
        assert "no error code" in str(error)
        assert error.code == 0

    def test_last_error_with_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.sys, "platform", "win32")
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 2)
        monkeypatch.setattr(winapi.ctypes, "FormatError", lambda _code: "Не найдено")
        error = winapi._last_error("DoThing")
        assert error.code == 2
        assert "[2]" in str(error)


class TestDataclasses:
    def test_elevation_info_can_elevate(self) -> None:
        limited = winapi.ElevationInfo(elevation_type=winapi.ELEVATION_TYPE_LIMITED)
        full = winapi.ElevationInfo(elevation_type=winapi.ELEVATION_TYPE_FULL)
        assert limited.can_elevate is True
        assert full.can_elevate is False

    def test_elevation_info_high_integrity(self) -> None:
        high = winapi.ElevationInfo(integrity_level=winapi.INTEGRITY_HIGH)
        medium = winapi.ElevationInfo(integrity_level=winapi.INTEGRITY_MEDIUM)
        assert high.high_integrity is True
        assert medium.high_integrity is False

    def test_process_run_finished(self) -> None:
        assert winapi.ProcessRun(pid=1, exit_code=0).finished is True
        assert winapi.ProcessRun(pid=1, exit_code=None).finished is False
        assert winapi.ProcessRun(pid=1, exit_code=0, timed_out=True).finished is False


# --------------------------------------------------------------------------- #
# Перечисление окон и их свойства
# --------------------------------------------------------------------------- #


class TestWindowProperties:
    def test_enum_windows_skips_null_and_collects(self, api: FakeWinApi) -> None:
        def sweep(callback: object, _param: object) -> bool:
            callback(0, None)  # type: ignore[operator]
            callback(0x3210, None)  # type: ignore[operator]
            return True

        api.stub("user32.EnumWindows", side_effect=sweep)
        assert winapi.enum_windows() == [0x3210]

    def test_enum_windows_raises_when_empty(self, api: FakeWinApi) -> None:
        api.stub("user32.EnumWindows", result=False)
        with pytest.raises(winapi.WinApiError, match="EnumWindows"):
            winapi.enum_windows()

    def test_window_title_empty_for_zero_length(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindowTextLengthW", result=0)
        assert winapi.window_title(1) == ""

    def test_window_title_reads_buffer(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindowTextLengthW", result=5)

        def fill(_hwnd: object, buffer: object, _size: object) -> int:
            buffer.value = "Окно"  # type: ignore[attr-defined]
            return 4

        api.stub("user32.GetWindowTextW", side_effect=fill)
        assert winapi.window_title(1) == "Окно"

    def test_window_class_name_reads_buffer(self, api: FakeWinApi) -> None:
        def fill(_hwnd: object, buffer: object, _size: object) -> int:
            buffer.value = "Chrome_WidgetWin_1"  # type: ignore[attr-defined]
            return 18

        api.stub("user32.GetClassNameW", side_effect=fill)
        assert winapi.window_class_name(1) == "Chrome_WidgetWin_1"

    def test_window_pid_reads_out_param(self, api: FakeWinApi) -> None:
        def fill(_hwnd: object, pid_ref: object) -> int:
            _out(pid_ref, ctypes.c_ulong).value = 4242
            return 7

        api.stub("user32.GetWindowThreadProcessId", side_effect=fill)
        assert winapi.window_pid(1) == 4242

    def test_window_thread_id(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindowThreadProcessId", result=99)
        assert winapi.window_thread_id(1) == 99

    def test_boolean_window_probes(self, api: FakeWinApi) -> None:
        api.stub("user32.IsWindow", result=1)
        api.stub("user32.IsWindowVisible", result=0)
        api.stub("user32.IsIconic", result=1)
        api.stub("user32.IsZoomed", result=0)
        assert winapi.is_window(1) is True
        assert winapi.is_window_visible(1) is False
        assert winapi.is_iconic(1) is True
        assert winapi.is_zoomed(1) is False

    def test_is_cloaked_false_without_dwmapi(self, api: FakeWinApi) -> None:
        api.absent("dwmapi.DwmGetWindowAttribute")
        assert winapi.is_cloaked(1) is False

    def test_is_cloaked_reads_attribute(self, api: FakeWinApi) -> None:
        def fill(_hwnd: object, _attr: object, value_ref: object, _size: object) -> int:
            _out(value_ref, ctypes.c_int).value = 1
            return 0

        api.stub("dwmapi.DwmGetWindowAttribute", side_effect=fill)
        assert winapi.is_cloaked(1) is True

    def test_window_ex_style_prefers_ptr_variant(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindowLongPtrW", result=winapi.WS_EX_TOOLWINDOW)
        assert winapi.window_ex_style(1) == winapi.WS_EX_TOOLWINDOW

    def test_window_ex_style_falls_back_to_long(self, api: FakeWinApi) -> None:
        api.absent("user32.GetWindowLongPtrW")
        api.stub("user32.GetWindowLongW", result=winapi.WS_EX_APPWINDOW)
        assert winapi.window_ex_style(1) == winapi.WS_EX_APPWINDOW

    def test_window_owner(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindow", result=0x555)
        assert winapi.window_owner(1) == 0x555

    def test_window_rect_reads_box(self, api: FakeWinApi) -> None:
        def fill(_hwnd: object, box_ref: object) -> int:
            box = _out(box_ref, winapi._RECT)
            box.left, box.top, box.right, box.bottom = 1, 2, 101, 82
            return 1

        api.stub("user32.GetWindowRect", side_effect=fill)
        assert winapi.window_rect(1) == winapi.Rect(1, 2, 101, 82)

    def test_window_rect_raises_on_failure(self, api: FakeWinApi) -> None:
        api.stub("user32.GetWindowRect", result=0)
        with pytest.raises(winapi.WinApiError, match="GetWindowRect"):
            winapi.window_rect(1)

    def test_extended_frame_bounds_uses_dwm(self, api: FakeWinApi) -> None:
        def fill(_hwnd: object, _attr: object, box_ref: object, _size: object) -> int:
            box = _out(box_ref, winapi._RECT)
            box.left, box.top, box.right, box.bottom = 0, 0, 100, 80
            return 0

        api.stub("dwmapi.DwmGetWindowAttribute", side_effect=fill)
        assert winapi.extended_frame_bounds(1) == winapi.Rect(0, 0, 100, 80)

    def test_extended_frame_bounds_falls_back_when_dwm_refuses(self, api: FakeWinApi) -> None:
        api.stub("dwmapi.DwmGetWindowAttribute", result=1)  # ненулевой hresult

        def rect(_hwnd: object, box_ref: object) -> int:
            box = _out(box_ref, winapi._RECT)
            box.left, box.top, box.right, box.bottom = 5, 5, 55, 45
            return 1

        api.stub("user32.GetWindowRect", side_effect=rect)
        assert winapi.extended_frame_bounds(1) == winapi.Rect(5, 5, 55, 45)

    def test_extended_frame_bounds_falls_back_without_dwm(self, api: FakeWinApi) -> None:
        api.absent("dwmapi.DwmGetWindowAttribute")

        def rect(_hwnd: object, box_ref: object) -> int:
            box = _out(box_ref, winapi._RECT)
            box.left, box.top, box.right, box.bottom = 0, 0, 10, 10
            return 1

        api.stub("user32.GetWindowRect", side_effect=rect)
        assert winapi.extended_frame_bounds(1) == winapi.Rect(0, 0, 10, 10)

    def test_foreground_window(self, api: FakeWinApi) -> None:
        api.stub("user32.GetForegroundWindow", result=0x1234)
        assert winapi.foreground_window() == 0x1234


# --------------------------------------------------------------------------- #
# Изменение состояния окна
# --------------------------------------------------------------------------- #


class TestWindowState:
    def test_show_window(self, api: FakeWinApi) -> None:
        api.stub("user32.ShowWindow", result=1)
        assert winapi.show_window(1, winapi.SW_RESTORE) is True

    def test_set_window_position_ok(self, api: FakeWinApi) -> None:
        api.stub("user32.SetWindowPos", result=1)
        winapi.set_window_position(1, winapi.Rect(0, 0, 800, 600))

    def test_set_window_position_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.SetWindowPos", result=0)
        with pytest.raises(winapi.WinApiError, match="SetWindowPos"):
            winapi.set_window_position(1, winapi.Rect(0, 0, 800, 600))

    def test_set_foreground_window(self, api: FakeWinApi) -> None:
        api.stub("user32.SetForegroundWindow", result=0)
        assert winapi.set_foreground_window(1) is False

    def test_bring_window_to_top(self, api: FakeWinApi) -> None:
        api.stub("user32.BringWindowToTop", result=1)
        assert winapi.bring_window_to_top(1) is True

    def test_switch_to_this_window_present(self, api: FakeWinApi) -> None:
        entry = api.stub("user32.SwitchToThisWindow")
        assert winapi.switch_to_this_window(1) is True
        assert entry.calls

    def test_switch_to_this_window_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.SwitchToThisWindow")
        assert winapi.switch_to_this_window(1) is False

    def test_current_thread_id(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetCurrentThreadId", result=1234)
        assert winapi.current_thread_id() == 1234

    def test_attach_thread_input(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetCurrentThreadId", result=1)
        api.stub("user32.AttachThreadInput", result=1)
        assert winapi.attach_thread_input(2, attach=True) is True

    def test_press_chord_presses_then_releases(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.time, "sleep", lambda _s: None)
        entry = api.stub("user32.keybd_event")
        winapi.press_chord([0x11, 0x25])
        pressed = [call[0] for call in entry.calls if call[2] == 0]
        released = [call[0] for call in entry.calls if call[2] == winapi.KEYEVENTF_KEYUP]
        assert pressed == [0x11, 0x25]
        assert released == [0x25, 0x11]

    def test_post_close(self, api: FakeWinApi) -> None:
        api.stub("user32.PostMessageW", result=1)
        assert winapi.post_close(1) is True

    def test_send_close_answered(self, api: FakeWinApi) -> None:
        api.stub("user32.SendMessageTimeoutW", result=1)
        assert winapi.send_close(1, timeout_ms=500) is True

    def test_send_close_hung(self, api: FakeWinApi) -> None:
        api.stub("user32.SendMessageTimeoutW", result=0)
        assert winapi.send_close(1) is False


# --------------------------------------------------------------------------- #
# Запуск и процессы
# --------------------------------------------------------------------------- #


class TestProcesses:
    def test_shell_execute_ex_raises(self, api: FakeWinApi) -> None:
        api.stub("shell32.ShellExecuteExW", result=0)
        with pytest.raises(winapi.WinApiError, match="ShellExecuteExW"):
            winapi.shell_execute_ex("app.exe")

    def test_shell_execute_ex_without_handle(self, api: FakeWinApi) -> None:
        api.stub("shell32.ShellExecuteExW", result=1)  # hProcess остаётся нулём
        assert winapi.shell_execute_ex("app.exe") == winapi.ProcessRun()

    def test_shell_execute_returns_pid(self, api: FakeWinApi) -> None:
        api.stub("shell32.ShellExecuteExW", result=1)
        assert winapi.shell_execute("app.exe") == 0

    def test_process_run_no_wait(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetProcessId", result=555)
        run = winapi._process_run(ctypes.c_void_p(1), 0)
        assert run == winapi.ProcessRun(pid=555)

    def test_process_run_times_out(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetProcessId", result=555)
        api.stub("kernel32.WaitForSingleObject", result=winapi.WAIT_TIMEOUT)
        run = winapi._process_run(ctypes.c_void_p(1), 1000)
        assert run == winapi.ProcessRun(pid=555, timed_out=True)

    def test_process_run_reads_exit_code(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetProcessId", result=555)
        api.stub("kernel32.WaitForSingleObject", result=0)

        def fill(_handle: object, code_ref: object) -> int:
            _out(code_ref, ctypes.c_ulong).value = 3
            return 1

        api.stub("kernel32.GetExitCodeProcess", side_effect=fill)
        run = winapi._process_run(ctypes.c_void_p(1), -1)
        assert run == winapi.ProcessRun(pid=555, exit_code=3)

    def test_process_run_without_exit_code(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetProcessId", result=555)
        api.stub("kernel32.WaitForSingleObject", result=0)
        api.stub("kernel32.GetExitCodeProcess", result=0)
        run = winapi._process_run(ctypes.c_void_p(1), 100)
        assert run == winapi.ProcessRun(pid=555)

    def test_open_process_none_when_zero(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0)
        assert winapi._open_process(1, 0) is None

    def test_process_image_name_guards(self, api: FakeWinApi) -> None:
        assert winapi.process_image_name(0) == ""
        api.stub("kernel32.OpenProcess", result=0)
        assert winapi.process_image_name(10) == ""

    def test_process_image_name_query_fails(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)
        api.stub("kernel32.QueryFullProcessImageNameW", result=0)
        assert winapi.process_image_name(10) == ""

    def test_process_image_name_returns_basename(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)

        def fill(_handle: object, _flags: object, buffer: object, _size: object) -> int:
            buffer.value = r"C:\Program Files\App\chrome.exe"  # type: ignore[attr-defined]
            return 1

        api.stub("kernel32.QueryFullProcessImageNameW", side_effect=fill)
        assert winapi.process_image_name(10) == "chrome.exe"

    def test_process_running_guards(self, api: FakeWinApi) -> None:
        assert winapi.process_running(0) is False
        api.stub("kernel32.OpenProcess", result=0)
        assert winapi.process_running(10) is False

    def test_process_running_get_code_fails(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)
        api.stub("kernel32.GetExitCodeProcess", result=0)
        assert winapi.process_running(10) is False

    def test_process_running_still_active(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)

        def fill(_handle: object, code_ref: object) -> int:
            _out(code_ref, ctypes.c_ulong).value = winapi.STILL_ACTIVE
            return 1

        api.stub("kernel32.GetExitCodeProcess", side_effect=fill)
        assert winapi.process_running(10) is True

    def test_terminate_process_no_handle(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0)
        with pytest.raises(winapi.WinApiError, match="OpenProcess"):
            winapi.terminate_process(10)

    def test_terminate_process_fails(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)
        api.stub("kernel32.TerminateProcess", result=0)
        with pytest.raises(winapi.WinApiError, match="TerminateProcess"):
            winapi.terminate_process(10)

    def test_terminate_process_ok(self, api: FakeWinApi) -> None:
        api.stub("kernel32.OpenProcess", result=0x40)
        api.stub("kernel32.TerminateProcess", result=1)
        winapi.terminate_process(10)


# --------------------------------------------------------------------------- #
# Токен процесса: права и привилегии
# --------------------------------------------------------------------------- #


class TestToken:
    def test_current_process_token_raises(self, api: FakeWinApi) -> None:
        api.stub("advapi32.OpenProcessToken", result=0)
        with pytest.raises(winapi.WinApiError, match="OpenProcessToken"):
            winapi._current_process_token(winapi.TOKEN_QUERY)

    def test_token_information_short_buffer_raises(self, api: FakeWinApi) -> None:
        def fill(_token: object, _cls: object, _buf: object, _len: object, need_ref: object) -> int:
            _out(need_ref, ctypes.c_ulong).value = 4  # <= size, значит не «мал буфер»
            return 0

        api.stub("advapi32.GetTokenInformation", side_effect=fill)
        with pytest.raises(winapi.WinApiError, match="GetTokenInformation"):
            winapi._token_information(ctypes.c_void_p(1), 25, 16)

    def test_token_information_grows_then_fails(self, api: FakeWinApi) -> None:
        def fill(_token: object, _cls: object, _buf: object, _len: object, need_ref: object) -> int:
            _out(need_ref, ctypes.c_ulong).value = 128  # больше size -> вырасти
            return 0

        api.stub("advapi32.GetTokenInformation", side_effect=fill)
        with pytest.raises(winapi.WinApiError, match="GetTokenInformation"):
            winapi._token_information(ctypes.c_void_p(1), 25, 16)

    def test_token_information_grows_then_succeeds(self, api: FakeWinApi) -> None:
        state = {"first": True}

        def fill(_token: object, _cls: object, _buf: object, _len: object, need_ref: object) -> int:
            if state["first"]:
                state["first"] = False
                _out(need_ref, ctypes.c_ulong).value = 32
                return 0
            _out(need_ref, ctypes.c_ulong).value = 32
            return 1

        api.stub("advapi32.GetTokenInformation", side_effect=fill)
        assert len(winapi._token_information(ctypes.c_void_p(1), 25, 16)) == 32

    def test_token_information_first_call_succeeds(self, api: FakeWinApi) -> None:
        def fill(_token: object, _cls: object, _buf: object, _len: object, need_ref: object) -> int:
            _out(need_ref, ctypes.c_ulong).value = 8
            return 1

        api.stub("advapi32.GetTokenInformation", side_effect=fill)
        assert winapi._token_information(ctypes.c_void_p(1), 25, 16) == b"\x00" * 8

    def test_token_dword(self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi, "_token_information", lambda *_a: b"\x05\x00\x00\x00")
        assert winapi._token_dword(ctypes.c_void_p(1), 20) == 5

    def test_token_integrity_level_short_buffer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi, "_token_information", lambda *_a: b"\x00\x00")
        with pytest.raises(winapi.WinApiError, match="short buffer"):
            winapi._token_integrity_level(ctypes.c_void_p(1))

    def test_token_integrity_level_null_sid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pointer_size = ctypes.sizeof(ctypes.c_void_p)
        monkeypatch.setattr(winapi, "_token_information", lambda *_a: b"\x00" * pointer_size)
        with pytest.raises(winapi.WinApiError, match="null SID"):
            winapi._token_integrity_level(ctypes.c_void_p(1))

    def test_current_user_is_admin_sid_fails(self, api: FakeWinApi) -> None:
        api.stub("advapi32.CreateWellKnownSid", result=0)
        api.stub("advapi32.CheckTokenMembership", result=1)
        with pytest.raises(winapi.WinApiError, match="CreateWellKnownSid"):
            winapi.current_user_is_admin()

    def test_current_user_is_admin_membership_fails(self, api: FakeWinApi) -> None:
        api.stub("advapi32.CreateWellKnownSid", result=1)
        api.stub("advapi32.CheckTokenMembership", result=0)
        with pytest.raises(winapi.WinApiError, match="CheckTokenMembership"):
            winapi.current_user_is_admin()

    def test_enable_privilege_lookup_fails(self, api: FakeWinApi) -> None:
        api.stub("advapi32.LookupPrivilegeValueW", result=0)
        with pytest.raises(winapi.WinApiError, match="LookupPrivilegeValueW"):
            winapi.enable_privilege(winapi.SE_SHUTDOWN_NAME)

    def test_enable_privilege_ok(self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 0)
        api.stub("advapi32.LookupPrivilegeValueW", result=1)
        api.stub("advapi32.OpenProcessToken", result=1)
        api.stub("advapi32.AdjustTokenPrivileges", result=1)
        assert winapi.enable_privilege(winapi.SE_SHUTDOWN_NAME) is True

    def test_enable_privilege_partial_returns_false(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 1300)
        api.stub("advapi32.LookupPrivilegeValueW", result=1)
        api.stub("advapi32.OpenProcessToken", result=1)
        api.stub("advapi32.AdjustTokenPrivileges", result=1)
        assert winapi.enable_privilege(winapi.SE_SHUTDOWN_NAME) is False

    def test_enable_privilege_adjust_fails(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 5)
        monkeypatch.setattr(winapi.ctypes, "FormatError", lambda _c: "Отказано")
        api.stub("advapi32.LookupPrivilegeValueW", result=1)
        api.stub("advapi32.OpenProcessToken", result=1)
        api.stub("advapi32.AdjustTokenPrivileges", result=0)
        with pytest.raises(winapi.WinApiError, match="AdjustTokenPrivileges"):
            winapi.enable_privilege(winapi.SE_SHUTDOWN_NAME)


# --------------------------------------------------------------------------- #
# Питание: сон, выключение, блокировка
# --------------------------------------------------------------------------- #


class TestPower:
    def test_set_suspend_state_ok(self, api: FakeWinApi) -> None:
        api.stub("powrprof.SetSuspendState", result=1)
        winapi.set_suspend_state(hibernate=False)

    def test_set_suspend_state_raises(self, api: FakeWinApi) -> None:
        api.stub("powrprof.SetSuspendState", result=0)
        with pytest.raises(winapi.WinApiError, match="SetSuspendState"):
            winapi.set_suspend_state(hibernate=True)

    def test_exit_windows_ok(self, api: FakeWinApi) -> None:
        api.stub("user32.ExitWindowsEx", result=1)
        winapi.exit_windows(winapi.EWX_LOGOFF)

    def test_exit_windows_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.ExitWindowsEx", result=0)
        with pytest.raises(winapi.WinApiError, match="ExitWindowsEx"):
            winapi.exit_windows(winapi.EWX_SHUTDOWN)

    def test_initiate_shutdown_ok(self, api: FakeWinApi) -> None:
        api.stub("advapi32.InitiateSystemShutdownExW", result=1)
        winapi.initiate_shutdown(delay_s=30, reboot=False)

    def test_initiate_shutdown_raises(self, api: FakeWinApi) -> None:
        api.stub("advapi32.InitiateSystemShutdownExW", result=0)
        with pytest.raises(winapi.WinApiError, match="InitiateSystemShutdownExW"):
            winapi.initiate_shutdown(delay_s=0, reboot=True, message="скоро")

    def test_abort_shutdown_ok(self, api: FakeWinApi) -> None:
        api.stub("advapi32.AbortSystemShutdownW", result=1)
        winapi.abort_shutdown()

    def test_abort_shutdown_raises(self, api: FakeWinApi) -> None:
        api.stub("advapi32.AbortSystemShutdownW", result=0)
        with pytest.raises(winapi.WinApiError, match="AbortSystemShutdownW"):
            winapi.abort_shutdown()

    def test_lock_workstation_ok(self, api: FakeWinApi) -> None:
        api.stub("user32.LockWorkStation", result=1)
        winapi.lock_workstation()

    def test_lock_workstation_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.LockWorkStation", result=0)
        with pytest.raises(winapi.WinApiError, match="LockWorkStation"):
            winapi.lock_workstation()


# --------------------------------------------------------------------------- #
# Дисплеи: перечисление, геометрия, яркость, DPI, кодовые страницы
# --------------------------------------------------------------------------- #


class TestDisplays:
    def test_enum_display_monitors_collects_and_skips(self, api: FakeWinApi) -> None:
        def enum(_hdc: object, _clip: object, cb: object, _param: object) -> int:
            cb(0, None, None, None)  # falsy -> пропуск
            cb(7, None, None, None)  # добавляется
            return 1

        api.stub("user32.EnumDisplayMonitors", side_effect=enum)
        assert winapi.enum_display_monitors() == [7]

    def test_enum_display_monitors_raises_when_empty(self, api: FakeWinApi) -> None:
        api.stub("user32.EnumDisplayMonitors", result=0)
        with pytest.raises(winapi.WinApiError, match="EnumDisplayMonitors"):
            winapi.enum_display_monitors()

    def test_monitor_info_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.GetMonitorInfoW", result=0)
        with pytest.raises(winapi.WinApiError, match="GetMonitorInfoW"):
            winapi.monitor_info(1)

    def test_monitor_info_success(self, api: FakeWinApi) -> None:
        def fill(_h: object, ref: object) -> int:
            info = _out(ref, winapi._MONITORINFOEXW)
            info.rcMonitor.left = 0
            info.rcMonitor.top = 0
            info.rcMonitor.right = 1920
            info.rcMonitor.bottom = 1080
            info.rcWork.left = 0
            info.rcWork.top = 0
            info.rcWork.right = 1920
            info.rcWork.bottom = 1040
            info.dwFlags = winapi.MONITORINFOF_PRIMARY
            info.szDevice = "\\\\.\\DISPLAY1"
            return 1

        api.stub("user32.GetMonitorInfoW", side_effect=fill)
        got = winapi.monitor_info(3)
        assert got.handle == 3
        assert got.rect.right == 1920
        assert got.work.bottom == 1040
        assert got.device == "\\\\.\\DISPLAY1"
        assert got.primary is True

    def test_physical_monitors_count_fails(self, api: FakeWinApi) -> None:
        api.stub("dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR", result=0)
        with pytest.raises(winapi.WinApiError, match="GetNumberOfPhysicalMonitors"):
            winapi.physical_monitors(1)

    def test_physical_monitors_zero(self, api: FakeWinApi) -> None:
        def count(_h: object, ref: object) -> int:
            _out(ref, ctypes.c_ulong).value = 0
            return 1

        api.stub("dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR", side_effect=count)
        assert winapi.physical_monitors(1) == []

    def test_physical_monitors_items_fail(self, api: FakeWinApi) -> None:
        def count(_h: object, ref: object) -> int:
            _out(ref, ctypes.c_ulong).value = 1
            return 1

        api.stub("dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR", side_effect=count)
        api.stub("dxva2.GetPhysicalMonitorsFromHMONITOR", result=0)
        with pytest.raises(winapi.WinApiError, match="GetPhysicalMonitorsFromHMONITOR"):
            winapi.physical_monitors(1)

    def test_physical_monitors_success(self, api: FakeWinApi) -> None:
        def count(_h: object, ref: object) -> int:
            _out(ref, ctypes.c_ulong).value = 1
            return 1

        def items(_h: object, _count: object, array: object) -> int:
            array[0].hPhysicalMonitor = 42  # type: ignore[index]
            return 1

        api.stub("dxva2.GetNumberOfPhysicalMonitorsFromHMONITOR", side_effect=count)
        api.stub("dxva2.GetPhysicalMonitorsFromHMONITOR", side_effect=items)
        result = winapi.physical_monitors(1)
        assert len(result) == 1
        assert result[0].handle == 42

    def test_destroy_physical_monitors_empty(self, api: FakeWinApi) -> None:
        winapi.destroy_physical_monitors([])  # без обращения к WinAPI

    def test_destroy_physical_monitors_raises(self, api: FakeWinApi) -> None:
        api.stub("dxva2.DestroyPhysicalMonitors", result=0)
        with pytest.raises(winapi.WinApiError, match="DestroyPhysicalMonitors"):
            winapi.destroy_physical_monitors([winapi.PhysicalMonitor(1, "экран")])

    def test_destroy_physical_monitors_ok(self, api: FakeWinApi) -> None:
        api.stub("dxva2.DestroyPhysicalMonitors", result=1)
        winapi.destroy_physical_monitors([winapi.PhysicalMonitor(1, "экран")])

    def test_monitor_brightness_raises(self, api: FakeWinApi) -> None:
        api.stub("dxva2.GetMonitorBrightness", result=0)
        with pytest.raises(winapi.WinApiError, match="GetMonitorBrightness"):
            winapi.monitor_brightness(1)

    def test_monitor_brightness_success(self, api: FakeWinApi) -> None:
        def get(_h: object, lo: object, cur: object, hi: object) -> int:
            _out(lo, ctypes.c_ulong).value = 0
            _out(cur, ctypes.c_ulong).value = 50
            _out(hi, ctypes.c_ulong).value = 100
            return 1

        api.stub("dxva2.GetMonitorBrightness", side_effect=get)
        assert winapi.monitor_brightness(1) == (0, 50, 100)

    def test_set_monitor_brightness_raises(self, api: FakeWinApi) -> None:
        api.stub("dxva2.SetMonitorBrightness", result=0)
        with pytest.raises(winapi.WinApiError, match="SetMonitorBrightness"):
            winapi.set_monitor_brightness(1, 40)

    def test_set_monitor_brightness_ok(self, api: FakeWinApi) -> None:
        api.stub("dxva2.SetMonitorBrightness", result=1)
        winapi.set_monitor_brightness(1, -5)  # клампится к 0

    def test_monitor_capabilities_raises(self, api: FakeWinApi) -> None:
        api.stub("dxva2.GetMonitorCapabilities", result=0)
        with pytest.raises(winapi.WinApiError, match="GetMonitorCapabilities"):
            winapi.monitor_capabilities(1)

    def test_monitor_capabilities_success(self, api: FakeWinApi) -> None:
        def get(_h: object, cap: object, temp: object) -> int:
            _out(cap, ctypes.c_ulong).value = 3
            _out(temp, ctypes.c_ulong).value = 1
            return 1

        api.stub("dxva2.GetMonitorCapabilities", side_effect=get)
        assert winapi.monitor_capabilities(1) == (3, 1)

    def test_monitor_from_window(self, api: FakeWinApi) -> None:
        api.stub("user32.MonitorFromWindow", result=99)
        assert winapi.monitor_from_window(1) == 99

    def test_display_device_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.EnumDisplayDevicesW")
        assert winapi.display_device("\\\\.\\DISPLAY1") == ("", "")

    def test_display_device_fail(self, api: FakeWinApi) -> None:
        api.stub("user32.EnumDisplayDevicesW", result=0)
        assert winapi.display_device("\\\\.\\DISPLAY1") == ("", "")

    def test_display_device_success(self, api: FakeWinApi) -> None:
        def fill(_dev: object, _idx: object, ref: object, _flags: object) -> int:
            info = _out(ref, winapi._DISPLAYDEVICEW)
            info.DeviceString = "LG ULTRAGEAR "
            info.DeviceID = " MONITOR\\GSM5B09 "
            return 1

        api.stub("user32.EnumDisplayDevicesW", side_effect=fill)
        assert winapi.display_device("\\\\.\\DISPLAY1") == ("LG ULTRAGEAR", "MONITOR\\GSM5B09")

    def test_windows_build_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi.sys, "platform", "linux")
        assert winapi.windows_build() == 0

    def test_windows_build_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Version:
            build = 22000

        monkeypatch.setattr(winapi.sys, "platform", "win32")
        monkeypatch.setattr(winapi.sys, "getwindowsversion", lambda: _Version, raising=False)
        assert winapi.windows_build() == 22000

    def test_console_output_codepage_absent(self, api: FakeWinApi) -> None:
        api.absent("kernel32.GetConsoleOutputCP")
        assert winapi.console_output_codepage() == 0

    def test_console_output_codepage_present(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetConsoleOutputCP", result=866)
        assert winapi.console_output_codepage() == 866

    def test_oem_codepage_absent(self, api: FakeWinApi) -> None:
        api.absent("kernel32.GetOEMCP")
        assert winapi.oem_codepage() == 0

    def test_oem_codepage_present(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GetOEMCP", result=1251)
        assert winapi.oem_codepage() == 1251

    def test_virtual_screen_rect(self, api: FakeWinApi) -> None:
        table = {
            winapi.SM_XVIRTUALSCREEN: -100,
            winapi.SM_YVIRTUALSCREEN: 0,
            winapi.SM_CXVIRTUALSCREEN: 2020,
            winapi.SM_CYVIRTUALSCREEN: 1080,
        }
        api.stub("user32.GetSystemMetrics", side_effect=lambda flag: table[flag])
        rect = winapi.virtual_screen_rect()
        assert (rect.left, rect.top, rect.right, rect.bottom) == (-100, 0, 1920, 1080)

    def test_cursor_position_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.GetCursorPos", result=0)
        with pytest.raises(winapi.WinApiError, match="GetCursorPos"):
            winapi.cursor_position()

    def test_cursor_position_success(self, api: FakeWinApi) -> None:
        def fill(ref: object) -> int:
            point = _out(ref, winapi._POINT)
            point.x = 10
            point.y = 20
            return 1

        api.stub("user32.GetCursorPos", side_effect=fill)
        assert winapi.cursor_position() == (10, 20)

    def test_monitor_from_point(self, api: FakeWinApi) -> None:
        api.stub("user32.MonitorFromPoint", result=55)
        assert winapi.monitor_from_point(1, 2) == 55

    def test_dpi_for_monitor_absent(self, api: FakeWinApi) -> None:
        api.absent("shcore.GetDpiForMonitor")
        assert winapi.dpi_for_monitor(1) == 96

    def test_dpi_for_monitor_bad_status(self, api: FakeWinApi) -> None:
        api.stub("shcore.GetDpiForMonitor", result=1)  # status != 0
        assert winapi.dpi_for_monitor(1) == 96

    def test_dpi_for_monitor_zero_horizontal(self, api: FakeWinApi) -> None:
        def get(_h: object, _mdt: object, horiz: object, _vert: object) -> int:
            _out(horiz, ctypes.c_uint).value = 0
            return 0

        api.stub("shcore.GetDpiForMonitor", side_effect=get)
        assert winapi.dpi_for_monitor(1) == 96

    def test_dpi_for_monitor_success(self, api: FakeWinApi) -> None:
        def get(_h: object, _mdt: object, horiz: object, _vert: object) -> int:
            _out(horiz, ctypes.c_uint).value = 144
            return 0

        api.stub("shcore.GetDpiForMonitor", side_effect=get)
        assert winapi.dpi_for_monitor(1) == 144


# --------------------------------------------------------------------------- #
# Буфер обмена: удержание, запись, чтение, слушатели
# --------------------------------------------------------------------------- #


class TestClipboard:
    def test_clipboard_holder_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.GetOpenClipboardWindow")
        assert winapi.clipboard_holder() == ""

    def test_clipboard_holder_no_window(self, api: FakeWinApi) -> None:
        api.stub("user32.GetOpenClipboardWindow", result=0)
        assert winapi.clipboard_holder() == ""

    def test_clipboard_holder_names_process(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api.stub("user32.GetOpenClipboardWindow", result=321)
        monkeypatch.setattr(winapi, "window_pid", lambda _h: 42)
        monkeypatch.setattr(winapi, "process_image_name", lambda _p: "notepad.exe")
        assert winapi.clipboard_holder() == "notepad.exe"

    def test_clipboard_open_retries_then_succeeds(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.time, "sleep", lambda _s: None)
        opens = iter([0, 1])
        api.stub("user32.OpenClipboard", side_effect=lambda _h: next(opens))
        api.stub("user32.CloseClipboard", result=1)
        with winapi._clipboard_open():
            pass

    def test_clipboard_open_fails_without_holder(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.time, "sleep", lambda _s: None)
        monkeypatch.setattr(winapi, "clipboard_holder", lambda: "")
        api.stub("user32.OpenClipboard", result=0)
        with (
            pytest.raises(winapi.WinApiError, match="OpenClipboard"),
            winapi._clipboard_open(),
        ):
            pass

    def test_clipboard_open_fails_with_holder(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.time, "sleep", lambda _s: None)
        monkeypatch.setattr(winapi, "clipboard_holder", lambda: "manager.exe")
        api.stub("user32.OpenClipboard", result=0)
        with (
            pytest.raises(winapi.WinApiError, match="clipboard held by manager.exe"),
            winapi._clipboard_open(),
        ):
            pass

    def test_clipboard_set_binary_empty(self, api: FakeWinApi) -> None:
        winapi.clipboard_set_binary([])  # без обращения к WinAPI

    def _open_ok(self, api: FakeWinApi) -> None:
        api.stub("user32.OpenClipboard", result=1)
        api.stub("user32.CloseClipboard", result=1)

    def test_clipboard_set_binary_skips_blank(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        set_data = api.stub("user32.SetClipboardData", result=1)
        winapi.clipboard_set_binary([(0, b"x"), (13, b"")])
        assert set_data.calls == []

    def test_clipboard_set_binary_empty_clipboard_fails(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=0)
        with pytest.raises(winapi.WinApiError, match="EmptyClipboard"):
            winapi.clipboard_set_binary([(13, b"data")])

    def test_clipboard_set_binary_alloc_fails(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        api.stub("kernel32.GlobalAlloc", result=0)
        with pytest.raises(winapi.WinApiError, match="GlobalAlloc"):
            winapi.clipboard_set_binary([(13, b"data")])

    def test_clipboard_set_binary_lock_fails(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        api.stub("kernel32.GlobalAlloc", result=1000)
        api.stub("kernel32.GlobalLock", result=0)
        api.stub("kernel32.GlobalFree", result=0)
        with pytest.raises(winapi.WinApiError, match="GlobalLock"):
            winapi.clipboard_set_binary([(13, b"data")])

    def test_clipboard_set_binary_set_data_fails(self, api: FakeWinApi) -> None:
        buffer = ctypes.create_string_buffer(16)
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        api.stub("kernel32.GlobalAlloc", result=1000)
        api.stub("kernel32.GlobalLock", result=ctypes.addressof(buffer))
        api.stub("kernel32.GlobalUnlock", result=1)
        api.stub("kernel32.GlobalFree", result=0)
        api.stub("user32.SetClipboardData", result=0)
        with pytest.raises(winapi.WinApiError, match="SetClipboardData"):
            winapi.clipboard_set_binary([(13, b"data")])

    def test_clipboard_set_binary_success(self, api: FakeWinApi) -> None:
        buffer = ctypes.create_string_buffer(16)
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        api.stub("kernel32.GlobalAlloc", result=1000)
        api.stub("kernel32.GlobalLock", result=ctypes.addressof(buffer))
        api.stub("kernel32.GlobalUnlock", result=1)
        api.stub("user32.SetClipboardData", result=1)
        winapi.clipboard_set_binary([(13, b"data")])
        assert buffer.raw[:4] == b"data"

    def test_clipboard_set_text(self, api: FakeWinApi) -> None:
        buffer = ctypes.create_string_buffer(64)
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        api.stub("kernel32.GlobalAlloc", result=1000)
        api.stub("kernel32.GlobalLock", result=ctypes.addressof(buffer))
        api.stub("kernel32.GlobalUnlock", result=1)
        api.stub("user32.SetClipboardData", result=1)
        winapi.clipboard_set_text("хи")
        assert buffer.raw[:4] == "хи".encode("utf-16-le")

    def test_clipboard_clear_ok(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=1)
        winapi.clipboard_clear()

    def test_clipboard_clear_raises(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        api.stub("user32.EmptyClipboard", result=0)
        with pytest.raises(winapi.WinApiError, match="EmptyClipboard"):
            winapi.clipboard_clear()

    def test_clipboard_sequence_number_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.GetClipboardSequenceNumber")
        assert winapi.clipboard_sequence_number() == 0

    def test_clipboard_sequence_number_present(self, api: FakeWinApi) -> None:
        api.stub("user32.GetClipboardSequenceNumber", result=7)
        assert winapi.clipboard_sequence_number() == 7

    def test_read_blob_empty(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GlobalSize", result=0)
        assert winapi._read_blob(ctypes.c_void_p(1)) == b""

    def test_read_blob_lock_fails(self, api: FakeWinApi) -> None:
        api.stub("kernel32.GlobalSize", result=4)
        api.stub("kernel32.GlobalLock", result=0)
        assert winapi._read_blob(ctypes.c_void_p(1)) == b""

    def test_read_blob_truncates_to_limit(self, api: FakeWinApi) -> None:
        buffer = ctypes.create_string_buffer(b"abcdefghij", 10)
        api.stub("kernel32.GlobalSize", result=10)
        api.stub("kernel32.GlobalLock", result=ctypes.addressof(buffer))
        api.stub("kernel32.GlobalUnlock", result=1)
        assert winapi._read_blob(ctypes.c_void_p(1), limit=4) == b"abcd"

    def test_read_dropped_files_absent(self, api: FakeWinApi) -> None:
        api.absent("shell32.DragQueryFileW")
        assert winapi._read_dropped_files(ctypes.c_void_p(1)) == ()

    def test_read_dropped_files_reads_names(self, api: FakeWinApi) -> None:
        lengths = {0: 5, 1: 0, 2: 5}
        names = {0: "a.txt", 2: "b.txt"}

        def drag(_handle: object, index: int, buf: object, _size: object) -> int:
            if index == 0xFFFFFFFF:
                return 3
            if buf is None:
                return lengths[index]
            buf.value = names[index]  # type: ignore[attr-defined]
            return len(names[index])

        api.stub("shell32.DragQueryFileW", side_effect=drag)
        assert winapi._read_dropped_files(ctypes.c_void_p(1)) == ("a.txt", "b.txt")

    def test_read_clipboard_full(self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch) -> None:
        self._open_ok(api)
        formats = iter([13, 15, 49999, 0])
        api.stub("user32.EnumClipboardFormats", side_effect=lambda _c: next(formats))
        api.stub("user32.GetClipboardData", result=1)
        monkeypatch.setattr(winapi, "_read_blob", lambda _h, **_kw: "hi".encode("utf-16-le"))
        monkeypatch.setattr(winapi, "_read_dropped_files", lambda _h: ("c:\\x.txt",))
        data = winapi.read_clipboard(blobs=[49999])
        assert data.formats == (13, 15, 49999)
        assert data.text == "hi"
        assert data.files == ("c:\\x.txt",)
        assert 49999 in data.blobs
        assert data.has(13) is True

    def test_read_clipboard_text_handle_missing(self, api: FakeWinApi) -> None:
        self._open_ok(api)
        formats = iter([13, 0])
        api.stub("user32.EnumClipboardFormats", side_effect=lambda _c: next(formats))
        api.stub("user32.GetClipboardData", result=0)
        data = winapi.read_clipboard()
        assert data.formats == (13,)
        assert data.text == ""

    def test_add_clipboard_format_listener_ok(self, api: FakeWinApi) -> None:
        api.stub("user32.AddClipboardFormatListener", result=1)
        winapi.add_clipboard_format_listener(1)

    def test_add_clipboard_format_listener_raises(self, api: FakeWinApi) -> None:
        api.stub("user32.AddClipboardFormatListener", result=0)
        with pytest.raises(winapi.WinApiError, match="AddClipboardFormatListener"):
            winapi.add_clipboard_format_listener(1)

    def test_remove_clipboard_format_listener_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.RemoveClipboardFormatListener")
        winapi.remove_clipboard_format_listener(1)  # тихо выходит

    def test_remove_clipboard_format_listener_present(self, api: FakeWinApi) -> None:
        stop = api.stub("user32.RemoveClipboardFormatListener", result=1)
        winapi.remove_clipboard_format_listener(1)
        assert len(stop.calls) == 1


# --------------------------------------------------------------------------- #
# Невидимое окно сообщений
# --------------------------------------------------------------------------- #


def _wndproc_seam(api: FakeWinApi) -> None:
    """``DefWindowProcW`` только приводится к ``c_void_p`` — подсунуть кастуемое."""
    api._entries["user32.DefWindowProcW"] = ctypes.c_void_p(0)  # type: ignore[assignment]


class TestMessageWindow:
    def test_create_success_and_idempotent(self, api: FakeWinApi) -> None:
        _wndproc_seam(api)
        api.stub("user32.RegisterClassW", result=1)
        api.stub("user32.CreateWindowExW", result=555)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window.create()
        assert window.hwnd == 555
        window.create()  # второй раз — no-op
        assert window.hwnd == 555

    def test_create_register_fails_hard(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 8)
        monkeypatch.setattr(winapi.ctypes, "FormatError", lambda _c: "нет памяти")
        _wndproc_seam(api)
        api.stub("user32.RegisterClassW", result=0)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        with pytest.raises(winapi.WinApiError, match="RegisterClassW"):
            window.create()

    def test_create_class_already_exists_then_creates(
        self, api: FakeWinApi, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(winapi.ctypes, "GetLastError", lambda: 1410)
        monkeypatch.setattr(winapi.ctypes, "FormatError", lambda _c: "уже есть")
        _wndproc_seam(api)
        api.stub("user32.RegisterClassW", result=0)
        api.stub("user32.CreateWindowExW", result=777)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window.create()
        assert window.hwnd == 777

    def test_create_window_fails(self, api: FakeWinApi) -> None:
        _wndproc_seam(api)
        api.stub("user32.RegisterClassW", result=1)
        api.stub("user32.CreateWindowExW", result=0)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        with pytest.raises(winapi.WinApiError, match="CreateWindowExW"):
            window.create()

    def test_pump_before_create_raises(self) -> None:
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        with pytest.raises(winapi.WinApiError, match="before create"):
            window.pump()

    def test_pump_returns_on_quit(self, api: FakeWinApi) -> None:
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]
        api.stub("user32.GetMessageW", result=0)
        window.pump()

    def test_pump_returns_on_stop_message(self, api: FakeWinApi) -> None:
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]

        def get(ref: object, _hwnd: object, _min: object, _max: object) -> int:
            _out(ref, winapi._MSG).message = winapi._WM_AYRIS_STOP
            return 1

        api.stub("user32.GetMessageW", side_effect=get)
        window.pump()

    def test_pump_swallows_handler_error(self, api: FakeWinApi) -> None:
        def handler(_code: int) -> None:
            raise RuntimeError("boom")

        window = winapi.MessageWindow("AyrisTest", handler)
        window._hwnd = 9  # type: ignore[attr-defined]
        counter = iter([1, 0])

        def get(ref: object, _hwnd: object, _min: object, _max: object) -> int:
            step = next(counter)
            if step:
                _out(ref, winapi._MSG).message = 0x0400
            return step

        api.stub("user32.GetMessageW", side_effect=get)
        window.pump()  # исключение обработчика гасится и не роняет цикл

    def test_stop_without_window(self, api: FakeWinApi) -> None:
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window.stop()  # нет окна — тихий выход

    def test_stop_when_post_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.PostMessageW")
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]
        window.stop()

    def test_stop_posts(self, api: FakeWinApi) -> None:
        post = api.stub("user32.PostMessageW", result=1)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]
        window.stop()
        assert len(post.calls) == 1

    def test_close_with_present_helpers(self, api: FakeWinApi) -> None:
        destroy = api.stub("user32.DestroyWindow", result=1)
        unregister = api.stub("user32.UnregisterClassW", result=1)
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]
        window._atom = 1  # type: ignore[attr-defined]
        window.close()
        assert window.hwnd == 0
        assert len(destroy.calls) == 1
        assert len(unregister.calls) == 1

    def test_close_when_helpers_absent(self, api: FakeWinApi) -> None:
        api.absent("user32.DestroyWindow", "user32.UnregisterClassW")
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window._hwnd = 9  # type: ignore[attr-defined]
        window._atom = 1  # type: ignore[attr-defined]
        window.close()
        assert window.hwnd == 0

    def test_close_when_nothing_to_do(self, api: FakeWinApi) -> None:
        window = winapi.MessageWindow("AyrisTest", lambda _code: None)
        window.close()  # ни окна, ни атома
        assert window.hwnd == 0


# --------------------------------------------------------------------------- #
# Синтезированный ввод
# --------------------------------------------------------------------------- #


class TestInput:
    def test_map_virtual_key_keeps_low_byte(self, api: FakeWinApi) -> None:
        api.stub("user32.MapVirtualKeyW", result=0x11E)
        assert winapi.map_virtual_key(0x11) == 0x1E

    def test_vk_key_scan(self, api: FakeWinApi) -> None:
        api.stub("user32.VkKeyScanW", result=0x41)
        assert winapi.vk_key_scan("A") == 0x41

    def test_vk_key_scan_unmapped(self, api: FakeWinApi) -> None:
        api.stub("user32.VkKeyScanW", result=-1)
        assert winapi.vk_key_scan("й") == -1

    def test_send_input_empty(self, api: FakeWinApi) -> None:
        assert winapi._send_input([], call="SendInput(test)") == 0

    def test_send_key_events_raises_when_blocked(self, api: FakeWinApi) -> None:
        api.stub("user32.SendInput", result=0)
        with pytest.raises(winapi.WinApiError, match="SendInput"):
            winapi.send_key_events([(0x11, 0x1D, 0)])

    def test_send_key_events_success(self, api: FakeWinApi) -> None:
        api.stub("user32.SendInput", result=1)
        assert winapi.send_key_events([(0x11, 0x1D, 0)]) == 1

    def test_send_unicode_text(self, api: FakeWinApi) -> None:
        api.stub("user32.SendInput", result=4)
        assert winapi.send_unicode_text("hi") == 4

    def test_send_mouse_event(self, api: FakeWinApi) -> None:
        api.stub("user32.SendInput", result=1)
        assert winapi.send_mouse_event(flags=0x0001, dx=5, dy=6) == 1


# --------------------------------------------------------------------------- #
# ayris.utils.monitors: раскладка, разрешение адреса дисплея
# --------------------------------------------------------------------------- #


def _make_monitor(
    *,
    index: int,
    left: int,
    right: int,
    top: int = 0,
    bottom: int = 1080,
    primary: bool = False,
    external_index: int = -1,
    name: str = "",
    device: str = "",
    device_id: str = "",
    dpi: int = 96,
    handle: int | None = None,
) -> monitors.MonitorInfo:
    rect = winapi.Rect(left, top, right, bottom)
    return monitors.MonitorInfo(
        handle=index + 1 if handle is None else handle,
        index=index,
        rect=rect,
        work=rect,
        device=device,
        name=name,
        device_id=device_id,
        dpi=dpi,
        primary=primary,
        external_index=external_index,
    )


class TestMonitorInfo:
    def test_geometry_properties(self) -> None:
        monitor = _make_monitor(index=0, left=0, right=1920, top=0, bottom=1080)
        assert monitor.width == 1920
        assert monitor.height == 1080
        assert monitor.resolution == (1920, 1080)
        assert monitor.contains(10, 10) is True
        assert monitor.contains(1920, 0) is False
        assert monitor.contains(-1, 0) is False

    def test_title_ru_primary_unscaled(self) -> None:
        monitor = _make_monitor(index=0, left=0, right=1920, primary=True)
        assert monitor.title_ru == "Основной — 1920×1080"

    def test_title_ru_named_scaled(self) -> None:
        monitor = _make_monitor(index=1, left=1920, right=3840, name="Dell", dpi=144)
        assert monitor.title_ru == "Dell — 1920×1080 (150%)"


class TestOrderMonitors:
    def test_keeps_already_placed_and_renumbers_the_rest(self) -> None:
        primary = _make_monitor(index=0, left=0, right=1920, primary=True, handle=1)
        external = _make_monitor(index=5, left=1920, right=3840, external_index=9, handle=2)
        result = monitors.order_monitors([primary, external])
        assert result[0] is primary  # уже на месте — тот же объект
        assert result[1].index == 1
        assert result[1].external_index == 0
        assert result[1].handle == 2


class TestListMonitors:
    def test_off_windows_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "linux")
        assert monitors.list_monitors() == []

    def test_enum_failure_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")

        def boom() -> list[int]:
            raise winapi.WinApiError("no monitors")

        monkeypatch.setattr(winapi, "enum_display_monitors", boom)
        assert monitors.list_monitors() == []

    def test_skips_monitor_that_cannot_be_described(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "enum_display_monitors", lambda: [1, 2])

        def info(handle: int) -> winapi.MonitorInfo:
            if handle == 1:
                raise winapi.WinApiError("gone")
            return winapi.MonitorInfo(
                handle=2,
                rect=winapi.Rect(0, 0, 1920, 1080),
                work=winapi.Rect(0, 0, 1920, 1040),
                device="\\\\.\\DISPLAY1",
                primary=True,
            )

        monkeypatch.setattr(winapi, "monitor_info", info)
        monkeypatch.setattr(winapi, "display_device", lambda _dev: ("Dell", "MONITOR\\X"))
        monkeypatch.setattr(winapi, "dpi_for_monitor", lambda _h: 144)
        result = monitors.list_monitors()
        assert len(result) == 1
        assert result[0].handle == 2
        assert result[0].name == "Dell"
        assert result[0].dpi == 144


class TestVirtualBounds:
    def test_off_windows_none_is_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "linux")
        assert monitors.virtual_bounds() == winapi.Rect()

    def test_uses_system_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "virtual_screen_rect", lambda: winapi.Rect(-100, 0, 1920, 1080))
        assert monitors.virtual_bounds() == winapi.Rect(-100, 0, 1920, 1080)

    def test_falls_back_to_union_when_metrics_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")

        def boom() -> winapi.Rect:
            raise winapi.WinApiError("metrics failed")

        monkeypatch.setattr(winapi, "virtual_screen_rect", boom)
        monkeypatch.setattr(
            monitors,
            "list_monitors",
            lambda: [_make_monitor(index=0, left=0, right=1920)],
        )
        assert monitors.virtual_bounds() == winapi.Rect(0, 0, 1920, 1080)

    def test_empty_explicit_list_is_empty(self) -> None:
        assert monitors.virtual_bounds([]) == winapi.Rect()

    def test_union_of_explicit_list(self) -> None:
        left_monitor = _make_monitor(index=0, left=-1920, right=0)
        right_monitor = _make_monitor(index=1, left=0, right=1920, external_index=0)
        bounds = monitors.virtual_bounds([left_monitor, right_monitor])
        assert (bounds.left, bounds.top, bounds.right, bounds.bottom) == (-1920, 0, 1920, 1080)


class TestMonitorForPoint:
    def test_no_monitors_returns_none(self) -> None:
        assert monitors.monitor_for_point(0, 0, []) is None

    def test_point_inside(self) -> None:
        monitor = _make_monitor(index=0, left=0, right=1920)
        assert monitors.monitor_for_point(10, 10, [monitor]) is monitor

    def test_nearest_when_in_the_gap(self) -> None:
        left_monitor = _make_monitor(index=0, left=0, right=100, bottom=100)
        right_monitor = _make_monitor(index=1, left=200, right=300, bottom=100, external_index=0)
        got = monitors.monitor_for_point(150, 50, [left_monitor, right_monitor])
        assert got is right_monitor


class TestMonitorForWindow:
    def test_no_monitors_returns_none(self) -> None:
        assert monitors.monitor_for_window(1, []) is None

    def test_matches_by_handle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "monitor_from_window", lambda _h: 7)
        primary = _make_monitor(index=0, left=0, right=1920, primary=True, handle=5)
        second = _make_monitor(index=1, left=1920, right=3840, external_index=0, handle=7)
        assert monitors.monitor_for_window(1, [primary, second]) is second

    def test_frame_centre_when_handle_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")

        def gone(_h: int) -> int:
            raise winapi.WinApiError("window closed")

        monkeypatch.setattr(winapi, "monitor_from_window", gone)
        monkeypatch.setattr(
            winapi, "extended_frame_bounds", lambda _h: winapi.Rect(2000, 0, 2100, 100)
        )
        primary = _make_monitor(index=0, left=0, right=1920, primary=True, handle=5)
        second = _make_monitor(index=1, left=1920, right=3840, external_index=0, handle=7)
        assert monitors.monitor_for_window(1, [primary, second]) is second

    def test_frame_unavailable_falls_back_to_primary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "monitor_from_window", lambda _h: 0)

        def no_dwm(_h: int) -> winapi.Rect:
            raise winapi.WinApiError("no frame")

        monkeypatch.setattr(winapi, "extended_frame_bounds", no_dwm)
        primary = _make_monitor(index=0, left=0, right=1920, primary=True, handle=5)
        second = _make_monitor(index=1, left=1920, right=3840, external_index=0, handle=7)
        assert monitors.monitor_for_window(1, [second, primary]) is primary

    def test_off_windows_uses_frame(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "linux")
        monkeypatch.setattr(winapi, "extended_frame_bounds", lambda _h: winapi.Rect(0, 0, 100, 100))
        primary = _make_monitor(index=0, left=0, right=1920, primary=True, handle=5)
        assert monitors.monitor_for_window(1, [primary]) is primary


class TestResolveMonitor:
    def _prim(self, **kwargs: object) -> monitors.MonitorInfo:
        return _make_monitor(index=0, left=0, right=1920, primary=True, handle=1, **kwargs)  # type: ignore[arg-type]

    def _ext1(self, **kwargs: object) -> monitors.MonitorInfo:
        return _make_monitor(
            index=1, left=1920, right=3840, external_index=0, handle=2, **kwargs  # type: ignore[arg-type]
        )

    def _ext2(self, **kwargs: object) -> monitors.MonitorInfo:
        return _make_monitor(
            index=2, left=3840, right=5760, external_index=1, handle=3, **kwargs  # type: ignore[arg-type]
        )

    def test_no_monitors_raises(self) -> None:
        with pytest.raises(monitors.MonitorNotFound):
            monitors.resolve_monitor("primary", [])

    def test_none_and_primary_word(self) -> None:
        prim, ext1 = self._prim(), self._ext1()
        assert monitors.resolve_monitor(None, [prim, ext1]) is prim
        assert monitors.resolve_monitor("основной", [prim, ext1]) is prim

    def test_current_word_uses_cursor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "cursor_position", lambda: (2500, 50))
        prim, ext1 = self._prim(), self._ext1()
        assert monitors.resolve_monitor("текущий", [prim, ext1]) is ext1

    def test_current_word_falls_back_to_primary(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")

        def no_cursor() -> tuple[int, int]:
            raise winapi.WinApiError("no cursor")

        monkeypatch.setattr(winapi, "cursor_position", no_cursor)
        prim, ext1 = self._prim(), self._ext1()
        assert monitors.resolve_monitor("current", [prim, ext1]) is prim

    def test_external_ordinal(self) -> None:
        prim, ext1, ext2 = self._prim(), self._ext1(), self._ext2()
        assert monitors.resolve_monitor("external_1", [prim, ext1, ext2]) is ext1
        assert monitors.resolve_monitor("внешний 2", [prim, ext1, ext2]) is ext2

    def test_external_bare_means_first(self) -> None:
        prim, ext1 = self._prim(), self._ext1()
        assert monitors.resolve_monitor("external", [prim, ext1]) is ext1

    def test_external_without_any_raises(self) -> None:
        with pytest.raises(monitors.MonitorNotFound):
            monitors.resolve_monitor("external", [self._prim()])

    def test_external_out_of_range_raises(self) -> None:
        with pytest.raises(monitors.MonitorNotFound):
            monitors.resolve_monitor("external_5", [self._prim(), self._ext1()])

    def test_indexed(self) -> None:
        prim, ext1 = self._prim(), self._ext1()
        assert monitors.resolve_monitor("2", [prim, ext1]) is ext1
        assert monitors.resolve_monitor("монитор 1", [prim, ext1]) is prim

    def test_indexed_out_of_range_raises(self) -> None:
        with pytest.raises(monitors.MonitorNotFound):
            monitors.resolve_monitor("монитор 3", [self._prim(), self._ext1()])

    def test_name_match_single(self) -> None:
        prim, ext1 = self._prim(name="Dell"), self._ext1(name="LG")
        assert monitors.resolve_monitor("dell", [prim, ext1]) is prim

    def test_name_match_none_raises(self) -> None:
        with pytest.raises(monitors.MonitorNotFound):
            monitors.resolve_monitor("sony", [self._prim(name="Dell"), self._ext1(name="LG")])

    def test_name_match_multiple_takes_leftmost(self) -> None:
        prim, ext1 = self._prim(name="Acme A"), self._ext1(name="Acme B")
        assert monitors.resolve_monitor("acme", [prim, ext1]) is prim


class TestMonitorUnderCursor:
    def test_off_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "linux")
        assert monitors._monitor_under_cursor([_make_monitor(index=0, left=0, right=1920)]) is None

    def test_cursor_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")

        def no_cursor() -> tuple[int, int]:
            raise winapi.WinApiError("no cursor")

        monkeypatch.setattr(winapi, "cursor_position", no_cursor)
        assert monitors._monitor_under_cursor([_make_monitor(index=0, left=0, right=1920)]) is None

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(monitors.sys, "platform", "win32")
        monkeypatch.setattr(winapi, "cursor_position", lambda: (10, 10))
        monitor = _make_monitor(index=0, left=0, right=1920)
        assert monitors._monitor_under_cursor([monitor]) is monitor
