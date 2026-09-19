from __future__ import annotations

import ctypes
import json
from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from ayris.actions.base import ActionCategory
from ayris.actions.system import brightness
from ayris.actions.system.brightness import (
    AdjustBrightness,
    BrightnessCapability,
    BrightnessMethod,
    BrightnessState,
    ListMonitors,
    SetBrightness,
)
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.slot_types import Direction
from ayris.utils import winapi
from ayris.utils.monitors import MonitorInfo


def monitor(index: int, *, handle: int | None = None) -> MonitorInfo:
    left = index * 1920
    return MonitorInfo(
        handle=index + 10 if handle is None else handle,
        index=index,
        rect=winapi.Rect(left, 0, left + 1920, 1080),
        work=winapi.Rect(left, 0, left + 1920, 1040),
        device=rf"\\.\DISPLAY{index + 1}",
        name=f"Panel {index + 1}",
        device_id=rf"MONITOR\PANEL{index + 1}",
        primary=index == 0,
        external_index=index - 1,
    )


class FakeBackend:
    def __init__(self, monitors: list[MonitorInfo], levels: list[int]) -> None:
        self.levels = dict(zip((item.handle for item in monitors), levels, strict=True))
        self.unsupported: set[int] = set()
        self.fail_write: set[int] = set()
        self.writes: list[tuple[int, int]] = []

    def capabilities(self, item: MonitorInfo) -> BrightnessCapability:
        if item.handle in self.unsupported:
            return BrightnessCapability(False, None, reason="unsupported")
        return BrightnessCapability(True, BrightnessMethod.DDC)

    def read(self, item: MonitorInfo) -> BrightnessState:
        if item.handle in self.unsupported:
            raise ActionUnavailable(
                "unsupported",
                user_message=f"Монитор {item.label} не поддерживает управление яркостью.",
            )
        return BrightnessState(item, self.levels[item.handle], BrightnessMethod.DDC)

    def write(self, item: MonitorInfo, percent: int) -> None:
        if item.handle in self.fail_write:
            raise winapi.WinApiError("write failed")
        self.writes.append((item.handle, percent))
        self.levels[item.handle] = percent


@pytest.fixture
def displays(monkeypatch: pytest.MonkeyPatch) -> tuple[list[MonitorInfo], FakeBackend]:
    items = [monitor(0), monitor(1), monitor(2)]
    backend = FakeBackend(items, [95, 5, 40])
    monkeypatch.setattr(brightness, "list_monitors", lambda: items)
    brightness.set_brightness_backend(backend)
    yield items, backend
    brightness.set_brightness_backend(None)


def test_adjust_clamps_and_uses_configured_step(
    displays: tuple[list[MonitorInfo], FakeBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    _items, backend = displays
    monkeypatch.setattr(brightness, "brightness_step", lambda: 10)
    up = AdjustBrightness().run(AdjustBrightness.Params(monitor="1", direction=Direction.UP))
    down = AdjustBrightness().run(AdjustBrightness.Params(monitor="2", direction=Direction.DOWN))
    assert up.ok and down.ok
    assert backend.writes == [(10, 100), (11, 0)]


def test_explicit_amount_and_monitor_resolution(
    displays: tuple[list[MonitorInfo], FakeBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    _items, backend = displays
    monkeypatch.setattr(brightness, "brightness_step", lambda: 50)
    result = AdjustBrightness().run(
        AdjustBrightness.Params(monitor="Panel 3", direction=Direction.DOWN, amount=15)
    )
    assert result.ok
    assert backend.writes == [(12, 25)]


def test_unsupported_single_monitor_raises(displays: tuple[list[MonitorInfo], FakeBackend]) -> None:
    _items, backend = displays
    backend.unsupported.add(10)
    with pytest.raises(ActionUnavailable, match="не поддерживает"):
        SetBrightness().run(SetBrightness.Params(monitor="1", level=50))


def test_group_is_best_effort_and_undo_restores_changed_monitors(
    displays: tuple[list[MonitorInfo], FakeBackend],
) -> None:
    _items, backend = displays
    backend.fail_write.add(11)
    result = SetBrightness().run(SetBrightness.Params(level=60))
    assert result.ok
    assert result.message_ru == "Яркость изменена на 2 из 3 мониторов."
    assert result.value is not None and [item.ok for item in result.value] == [True, False, True]
    assert result.undo_token is not None
    payload = json.loads(result.undo_token)
    assert [item["level"] for item in payload["monitors"]] == [95, 40]
    backend.fail_write.clear()
    undone = SetBrightness().undo(result.undo_token)
    assert undone.ok
    assert backend.levels == {10: 95, 11: 5, 12: 40}


def test_single_undo_prefers_device_after_handle_change(
    displays: tuple[list[MonitorInfo], FakeBackend], monkeypatch: pytest.MonkeyPatch
) -> None:
    items, backend = displays
    result = SetBrightness().run(SetBrightness.Params(monitor="1", level=20))
    assert result.undo_token
    moved = replace(items[0], handle=99)
    backend.levels[99] = backend.levels.pop(10)
    monkeypatch.setattr(brightness, "list_monitors", lambda: [moved, *items[1:]])
    assert SetBrightness().undo(result.undo_token).ok
    assert backend.levels[99] == 95


def test_malformed_undo_token_has_russian_error() -> None:
    with pytest.raises(ActionError, match="malformed") as caught:
        SetBrightness().undo("not json")
    assert caught.value.user_message == "Не помню, какая яркость была до этого."


def test_list_monitors_returns_models(displays: tuple[list[MonitorInfo], FakeBackend]) -> None:
    items, _backend = displays
    result = ListMonitors().run(ListMonitors.Params())
    assert result.value == items
    assert all(isinstance(item, MonitorInfo) for item in result.value or [])


def test_action_metadata_and_parameter_bounds() -> None:
    assert SetBrightness.meta.category is ActionCategory.DISPLAY
    assert SetBrightness.meta.supports_undo
    assert AdjustBrightness.meta.supports_undo
    with pytest.raises(ValidationError):
        SetBrightness.Params(level=101)
    with pytest.raises(ValidationError):
        AdjustBrightness.Params(direction=Direction.UP, amount=0)


def test_capability_cache_and_topology_invalidation(monkeypatch: pytest.MonkeyPatch) -> None:
    first = monitor(0)
    attached = [first]
    backend = brightness.WindowsBrightnessBackend()
    calls = 0

    def probe(item: MonitorInfo) -> Any:
        nonlocal calls
        calls += 1
        return brightness._CachedCapability(BrightnessCapability(True, BrightnessMethod.DDC))

    monkeypatch.setattr(brightness, "list_monitors", lambda: attached)
    monkeypatch.setattr(backend, "_probe", probe)
    backend.capabilities(first)
    backend.capabilities(first)
    assert calls == 1
    changed = replace(first, handle=99)
    attached[:] = [changed]
    backend.capabilities(changed)
    assert calls == 2


def test_dxva2_wrappers_and_destroy(monkeypatch: pytest.MonkeyPatch) -> None:
    destroyed: list[int] = []

    def count(_monitor: object, target: object) -> int:
        ctypes.cast(target, ctypes.POINTER(ctypes.c_ulong))[0] = 1
        return 1

    def enumerate_physical(_monitor: object, _count: object, target: object) -> int:
        array = ctypes.cast(target, ctypes.POINTER(winapi._PhysicalMonitorStruct))
        array[0].hPhysicalMonitor = 123
        array[0].szPhysicalMonitorDescription = "Test panel"
        return 1

    def get_brightness(_handle: object, minimum: object, current: object, maximum: object) -> int:
        ctypes.cast(minimum, ctypes.POINTER(ctypes.c_ulong))[0] = 10
        ctypes.cast(current, ctypes.POINTER(ctypes.c_ulong))[0] = 55
        ctypes.cast(maximum, ctypes.POINTER(ctypes.c_ulong))[0] = 90
        return 1

    def destroy(count_value: int, target: object) -> int:
        array = ctypes.cast(target, ctypes.POINTER(winapi._PhysicalMonitorStruct))
        destroyed.extend(int(array[index].hPhysicalMonitor or 0) for index in range(count_value))
        return 1

    functions = {
        "GetNumberOfPhysicalMonitorsFromHMONITOR": count,
        "GetPhysicalMonitorsFromHMONITOR": enumerate_physical,
        "GetMonitorBrightness": get_brightness,
        "DestroyPhysicalMonitors": destroy,
    }
    monkeypatch.setattr(winapi, "_require", lambda _library, function: functions[function])
    physical = winapi.physical_monitors(7)
    assert physical == [winapi.PhysicalMonitor(123, "Test panel")]
    assert winapi.monitor_brightness(123) == (10, 55, 90)
    winapi.destroy_physical_monitors(physical)
    assert destroyed == [123]
