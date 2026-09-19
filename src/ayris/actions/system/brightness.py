"""Brightness control for built-in (WMI) and external (DDC/CI) displays."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Any, ClassVar, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.slot_types import Direction, RelativeUnit, RelativeValue
from ayris.utils import winapi
from ayris.utils.monitors import MonitorInfo, MonitorNotFound, list_monitors, resolve_monitor

__all__ = [
    "AdjustBrightness",
    "BrightnessBackend",
    "BrightnessCapability",
    "BrightnessMethod",
    "BrightnessOutcome",
    "BrightnessState",
    "ListMonitors",
    "SetBrightness",
    "WindowsBrightnessBackend",
    "brightness_step",
    "get_brightness_backend",
    "set_brightness_backend",
]


class BrightnessMethod(StrEnum):
    WMI = "wmi"
    DDC = "ddc"


@dataclass(frozen=True, slots=True)
class BrightnessCapability:
    supported: bool
    method: BrightnessMethod | None
    minimum: int = 0
    maximum: int = 100
    latency_ms: int = 0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class BrightnessState:
    monitor: MonitorInfo
    level: int
    method: BrightnessMethod

    def as_dict(self) -> dict[str, object]:
        return {"monitor": self.monitor.as_dict(), "level": self.level, "method": self.method}


@dataclass(frozen=True, slots=True)
class BrightnessOutcome:
    monitor: MonitorInfo
    ok: bool
    state: BrightnessState | None = None
    message_ru: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "monitor": self.monitor.as_dict(),
            "ok": self.ok,
            "state": None if self.state is None else self.state.as_dict(),
            "message_ru": self.message_ru,
        }


class BrightnessBackend(Protocol):
    def capabilities(self, monitor: MonitorInfo) -> BrightnessCapability: ...
    def read(self, monitor: MonitorInfo) -> BrightnessState: ...
    def write(self, monitor: MonitorInfo, percent: int) -> None: ...


@dataclass(frozen=True, slots=True)
class _CachedCapability:
    capability: BrightnessCapability
    physical_index: int = -1
    wmi_instance: str = ""


def _identity(monitor: MonitorInfo) -> tuple[int, str, tuple[int, int, int, int]]:
    return monitor.handle, monitor.device, monitor.rect.as_tuple()


def _normalise_device(value: str) -> str:
    return "".join(character for character in value.casefold() if character.isalnum())


class WindowsBrightnessBackend:
    """Lazy Windows backend: WMI first, then DDC/CI, with topology-aware caching."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str, tuple[int, int, int, int]], _CachedCapability] = {}
        self._topology: tuple[tuple[int, str, tuple[int, int, int, int]], ...] = ()
        self._local = threading.local()
        self._lock = threading.RLock()

    def _refresh_topology(self) -> None:
        topology = tuple(_identity(monitor) for monitor in list_monitors())
        if topology != self._topology:
            self._cache.clear()
            self._topology = topology

    def capabilities(self, monitor: MonitorInfo) -> BrightnessCapability:
        with self._lock:
            self._refresh_topology()
            key = _identity(monitor)
            cached = self._cache.get(key)
            if cached is None:
                cached = self._probe(monitor)
                self._cache[key] = cached
            return cached.capability

    def read(self, monitor: MonitorInfo) -> BrightnessState:
        cached = self._cached(monitor)
        if cached.capability.method is BrightnessMethod.WMI:
            level = self._wmi_read(cached.wmi_instance)
        elif cached.capability.method is BrightnessMethod.DDC:
            level = self._ddc_read(monitor, cached)
        else:
            raise self._unsupported(monitor, cached.capability.reason)
        return BrightnessState(monitor, level, cached.capability.method)

    def write(self, monitor: MonitorInfo, percent: int) -> None:
        cached = self._cached(monitor)
        level = max(0, min(100, int(percent)))
        if cached.capability.method is BrightnessMethod.WMI:
            self._wmi_write(cached.wmi_instance, level)
        elif cached.capability.method is BrightnessMethod.DDC:
            self._ddc_write(monitor, cached, level)
        else:
            raise self._unsupported(monitor, cached.capability.reason)

    def _cached(self, monitor: MonitorInfo) -> _CachedCapability:
        self.capabilities(monitor)
        return self._cache[_identity(monitor)]

    def _probe(self, monitor: MonitorInfo) -> _CachedCapability:
        started = time.perf_counter()
        try:
            instance = self._wmi_instance(monitor)
            if instance:
                return _CachedCapability(
                    BrightnessCapability(
                        True, BrightnessMethod.WMI, latency_ms=self._elapsed(started)
                    ),
                    wmi_instance=instance,
                )
        except Exception as exc:
            if not isinstance(exc, (*winapi.COM_ERRORS, ImportError, AttributeError)):
                raise
            wmi_reason = str(exc)
        else:
            wmi_reason = "WMI monitor not matched"
        try:
            physical = winapi.physical_monitors(monitor.handle)
            try:
                for index, item in enumerate(physical):
                    try:
                        minimum, _current, maximum = winapi.monitor_brightness(item.handle)
                    except winapi.WinApiError:
                        continue
                    return _CachedCapability(
                        BrightnessCapability(
                            True, BrightnessMethod.DDC, minimum, maximum, self._elapsed(started)
                        ),
                        physical_index=index,
                    )
            finally:
                winapi.destroy_physical_monitors(physical)
        except winapi.WinApiError as exc:
            ddc_reason = str(exc)
        else:
            ddc_reason = "DDC/CI brightness is unavailable"
        reason = f"{wmi_reason}; {ddc_reason}"
        return _CachedCapability(
            BrightnessCapability(False, None, latency_ms=self._elapsed(started), reason=reason)
        )

    @staticmethod
    def _elapsed(started: float) -> int:
        return max(0, round((time.perf_counter() - started) * 1000))

    def _wmi_service(self) -> Any:
        service = getattr(self._local, "wmi_service", None)
        if service is None:
            from comtypes.client import CreateObject

            locator = CreateObject("WbemScripting.SWbemLocator")
            service = locator.ConnectServer(".", r"root\wmi")
            self._local.wmi_service = service
        return service

    def _wmi_instance(self, monitor: MonitorInfo) -> str:
        needle = _normalise_device(monitor.device_id)
        if not needle:
            return ""
        matches = []
        for item in self._wmi_service().ExecQuery("SELECT * FROM WmiMonitorBrightness"):
            instance = str(item.InstanceName)
            normalised = _normalise_device(instance)
            if needle in normalised or normalised in needle:
                matches.append(instance)
        return matches[0] if len(matches) == 1 else ""

    def _wmi_read(self, instance: str) -> int:
        for item in self._wmi_service().ExecQuery("SELECT * FROM WmiMonitorBrightness"):
            if str(item.InstanceName) == instance:
                return max(0, min(100, int(item.CurrentBrightness)))
        raise ActionUnavailable(
            "cached WMI monitor disappeared", user_message="Монитор больше не отвечает."
        )

    def _wmi_write(self, instance: str, level: int) -> None:
        matches = [
            item
            for item in self._wmi_service().ExecQuery("SELECT * FROM WmiMonitorBrightnessMethods")
            if str(item.InstanceName) == instance
        ]
        if len(matches) != 1:
            raise ActionUnavailable(
                "WMI brightness method disappeared", user_message="Монитор больше не отвечает."
            )
        matches[0].WmiSetBrightness(0, level)

    @staticmethod
    def _native(percent: int, minimum: int, maximum: int) -> int:
        return round(minimum + max(0, min(100, percent)) * (maximum - minimum) / 100)

    @staticmethod
    def _percent(value: int, minimum: int, maximum: int) -> int:
        if maximum <= minimum:
            return 0
        return max(0, min(100, round((value - minimum) * 100 / (maximum - minimum))))

    def _ddc_read(self, monitor: MonitorInfo, cached: _CachedCapability) -> int:
        physical = winapi.physical_monitors(monitor.handle)
        try:
            item = physical[cached.physical_index]
            minimum, current, maximum = winapi.monitor_brightness(item.handle)
            return self._percent(current, minimum, maximum)
        finally:
            winapi.destroy_physical_monitors(physical)

    def _ddc_write(self, monitor: MonitorInfo, cached: _CachedCapability, level: int) -> None:
        physical = winapi.physical_monitors(monitor.handle)
        try:
            item = physical[cached.physical_index]
            native = self._native(level, cached.capability.minimum, cached.capability.maximum)
            winapi.set_monitor_brightness(item.handle, native)
        finally:
            winapi.destroy_physical_monitors(physical)

    @staticmethod
    def _unsupported(monitor: MonitorInfo, reason: str) -> ActionUnavailable:
        return ActionUnavailable(
            f"brightness unsupported for {monitor.label}: {reason}",
            user_message=f"Монитор {monitor.label} не поддерживает управление яркостью.",
        )


_backend: BrightnessBackend | None = None
_real_backend: WindowsBrightnessBackend | None = None
_backend_lock = threading.Lock()


def get_brightness_backend() -> BrightnessBackend:
    if _backend is not None:
        return _backend
    global _real_backend
    with _backend_lock:
        if _real_backend is None:
            _real_backend = WindowsBrightnessBackend()
        return _real_backend


def set_brightness_backend(backend: BrightnessBackend | None) -> None:
    global _backend
    _backend = backend


def brightness_step() -> int:
    from ayris.core.config import get_settings

    return get_settings().actions.display.brightness_step


class _MonitorParams(ActionParams):
    monitor: str = Field(
        default="", max_length=160, title="Монитор", description="Пусто — все мониторы"
    )


def _targets(address: str) -> tuple[list[MonitorInfo], bool]:
    monitors = list_monitors()
    if not monitors:
        raise ActionUnavailable(
            "no attached monitors", user_message="Не нашла подключённых мониторов."
        )
    if not address.strip():
        return monitors, len(monitors) > 1
    try:
        return [resolve_monitor(address, monitors)], False
    except MonitorNotFound as exc:
        raise ActionError(str(exc), user_message=f"Монитор «{address.strip()}» не найден.") from exc


def _token(states: list[BrightnessState]) -> str | None:
    if not states:
        return None
    return json.dumps(
        {
            "v": 1,
            "monitors": [
                {
                    "handle": state.monitor.handle,
                    "device": state.monitor.device,
                    "level": state.level,
                }
                for state in states
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _change(
    monitor: MonitorInfo,
    backend: BrightnessBackend,
    level: int,
    current: BrightnessState | None = None,
) -> tuple[BrightnessOutcome, BrightnessState | None]:
    try:
        capability = backend.capabilities(monitor)
        if not capability.supported:
            raise WindowsBrightnessBackend._unsupported(monitor, capability.reason)
        before = backend.read(monitor) if current is None else current
        target = max(0, min(100, level))
        if target == before.level:
            return BrightnessOutcome(monitor, True, before, f"Яркость уже {target}%."), None
        backend.write(monitor, target)
        state = BrightnessState(monitor, target, before.method)
        return BrightnessOutcome(monitor, True, state, f"Яркость {target}%."), before
    except Exception as exc:
        if not isinstance(exc, (ActionError, winapi.WinApiError, *winapi.COM_ERRORS)):
            raise
        message = (
            exc.user_message
            if isinstance(exc, ActionError)
            else f"Не смогла изменить яркость монитора {monitor.label}."
        )
        return BrightnessOutcome(monitor, False, message_ru=message), None


def _result(
    outcomes: list[BrightnessOutcome], before: list[BrightnessState], grouped: bool
) -> ActionResult[list[BrightnessOutcome]]:
    changed = len(before)
    succeeded = sum(outcome.ok for outcome in outcomes)
    if grouped:
        message = f"Яркость изменена на {changed} из {len(outcomes)} мониторов."
    else:
        message = outcomes[0].message_ru
    data = {"monitors": [outcome.as_dict() for outcome in outcomes]}
    if not grouped and not succeeded:
        raise ActionUnavailable(outcomes[0].message_ru, user_message=outcomes[0].message_ru)
    if (grouped and changed) or (not grouped and succeeded):
        return ActionResult.done(message, value=outcomes, undo_token=_token(before), data=data)
    return ActionResult.failed(message, value=outcomes, data=data)


def _undo(token: str) -> ActionResult[list[BrightnessOutcome]]:
    try:
        payload = json.loads(token)
        items = payload["monitors"]
        if payload.get("v") != 1 or not isinstance(items, list):
            raise ValueError
        targets = list_monitors()
        backend = get_brightness_backend()
        outcomes: list[BrightnessOutcome] = []
        for item in items:
            monitor = next(
                (m for m in targets if m.device and m.device == str(item["device"])), None
            )
            monitor = monitor or next((m for m in targets if m.handle == int(item["handle"])), None)
            if monitor is None:
                continue
            outcome, _before = _change(monitor, backend, int(item["level"]))
            outcomes.append(outcome)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ActionError(
            "malformed brightness undo token", user_message="Не помню, какая яркость была до этого."
        ) from exc
    if not outcomes or not any(outcome.ok for outcome in outcomes):
        return ActionResult.failed("Не удалось вернуть прежнюю яркость.", value=outcomes)
    return ActionResult.done("Вернула прежнюю яркость.", value=outcomes)


@register
class SetBrightness(Action):
    meta: ClassVar = ActionMeta(
        name="SetBrightness",
        category=ActionCategory.DISPLAY,
        title_ru="Установить яркость",
        description_ru="Задать яркость одного или всех мониторов",
        supports_undo=True,
        timeout_ms=15_000,
    )

    class Params(_MonitorParams):
        level: int = Field(ge=0, le=100, title="Яркость", json_schema_extra={"unit_ru": "%"})

    def run(self, params: Params) -> ActionResult[list[BrightnessOutcome]]:
        targets, grouped = _targets(params.monitor)
        backend = get_brightness_backend()
        pairs = [_change(monitor, backend, params.level) for monitor in targets]
        return _result([pair[0] for pair in pairs], [pair[1] for pair in pairs if pair[1]], grouped)

    def undo(self, token: str) -> ActionResult[list[BrightnessOutcome]]:
        return _undo(token)


@register
class AdjustBrightness(Action):
    meta: ClassVar = ActionMeta(
        name="AdjustBrightness",
        category=ActionCategory.DISPLAY,
        title_ru="Изменить яркость",
        description_ru="Сделать один или все мониторы ярче либо темнее",
        supports_undo=True,
        timeout_ms=15_000,
    )

    class Params(_MonitorParams):
        direction: Direction = Field(title="Куда")
        amount: int | None = Field(
            default=None, ge=1, le=100, title="На сколько", json_schema_extra={"unit_ru": "%"}
        )

    def run(self, params: Params) -> ActionResult[list[BrightnessOutcome]]:
        targets, grouped = _targets(params.monitor)
        backend = get_brightness_backend()
        outcomes: list[BrightnessOutcome] = []
        previous: list[BrightnessState] = []
        change = RelativeValue(
            direction=params.direction,
            amount=None if params.amount is None else Decimal(params.amount),
            unit=RelativeUnit.STEP if params.amount is None else RelativeUnit.PERCENT,
        )
        for monitor in targets:
            try:
                current = backend.read(monitor)
                level = change.resolve(current.level, step=brightness_step())
                outcome, before = _change(monitor, backend, level, current)
            except Exception as exc:
                if not isinstance(exc, (ActionError, winapi.WinApiError, *winapi.COM_ERRORS)):
                    raise
                message = (
                    exc.user_message
                    if isinstance(exc, ActionError)
                    else f"Не смогла изменить яркость монитора {monitor.label}."
                )
                outcome, before = BrightnessOutcome(monitor, False, message_ru=message), None
            outcomes.append(outcome)
            if before is not None:
                previous.append(before)
        return _result(outcomes, previous, grouped)

    def undo(self, token: str) -> ActionResult[list[BrightnessOutcome]]:
        return _undo(token)


@register
class ListMonitors(Action):
    meta: ClassVar = ActionMeta(
        name="ListMonitors",
        category=ActionCategory.DISPLAY,
        title_ru="Список мониторов",
        description_ru="Получить подключённые мониторы",
        timeout_ms=5_000,
    )

    class Params(ActionParams):
        pass

    def run(self, _params: Params) -> ActionResult[list[MonitorInfo]]:
        monitors = list_monitors()
        if not monitors:
            raise ActionUnavailable(
                "no attached monitors", user_message="Не нашла подключённых мониторов."
            )
        return ActionResult.done(f"Подключено мониторов: {len(monitors)}.", value=monitors)
