"""Задача 69: сон, гибернация и пробуждение системы.

Windows замораживает весь процесс на время сна. Часы монотонного времени при
этом не идут, поэтому таймеры, отсчитанные :class:`threading.Timer`, на пробуждении
оказываются просрочены; открытый аудиопоток держит уже недействительный дескриптор
устройства; связь с облаком, скорее всего, была разорвана. Здесь — обработка этих
переходов.

* :class:`PowerCoordinator` — переносимая, тестируемая логика: на засыпание
  распускает активные потоки (событие на шине), на пробуждение ПЕРЕСОЗДАЁТ
  аудиопоток (перезапуск аудио-воркера — это реальная попытка заново открыть
  устройство, а не проверка его наличия в списке), перепроверяет сеть и
  пересчитывает просроченные таймеры.
* :func:`classify_power_message` — чистое сопоставление ``WM_POWERBROADCAST`` →
  переход, чтобы декодирование сообщения проверялось без Windows.
* Нативный фильтр событий Qt (только Windows) ловит ``WM_POWERBROADCAST`` и зовёт
  координатор. Он собирается лениво в :func:`install_power_events`, поэтому модуль
  импортируется без Qt.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, cast

from ayris.core.config import RestartScope
from ayris.core.events import Event

if TYPE_CHECKING:

    from PySide6.QtCore import QByteArray
    from PySide6.QtWidgets import QApplication

    from ayris.core.app import AyrisApp
    from ayris.core.events import EventBus

_log = logging.getLogger("ayris.power")

_WINDOWS_EVENT_TYPE = b"windows_generic_MSG"

#: ``WM_POWERBROADCAST`` и его коды в ``wParam`` (winuser.h).
WM_POWERBROADCAST = 0x0218
PBT_APMSUSPEND = 0x0004
PBT_APMRESUMECRITICAL = 0x0006
PBT_APMRESUMESUSPEND = 0x0007
PBT_APMRESUMEAUTOMATIC = 0x0012
_RESUME_EVENTS = frozenset({PBT_APMRESUMECRITICAL, PBT_APMRESUMESUSPEND, PBT_APMRESUMEAUTOMATIC})

__all__ = [
    "PowerCoordinator",
    "PowerTransition",
    "SystemResumed",
    "SystemSuspending",
    "classify_power_message",
    "install_power_events",
]


# ----------------------------------------------------------------------
# События питания на шине.
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SystemSuspending(Event):
    """Система уходит в сон или гибернацию — пора распустить активные потоки."""


@dataclass(frozen=True, slots=True)
class SystemResumed(Event):
    """Система проснулась — аудиопоток, сеть и таймеры уже пересобраны."""


# ----------------------------------------------------------------------
# Чистое декодирование сообщения питания.
# ----------------------------------------------------------------------


class PowerTransition(StrEnum):
    """Переход питания, который нас интересует."""

    SUSPEND = "suspend"
    RESUME = "resume"


def classify_power_message(message: int, wparam: int) -> PowerTransition | None:
    """Сопоставить ``WM_POWERBROADCAST`` с переходом питания.

    Возвращает ``None`` для чужих сообщений и незначимых кодов ``wParam``
    (например ``PBT_APMPOWERSTATUSCHANGE``), чтобы вызывающий их пропустил.
    Чистая функция — проверяется без Windows.
    """
    if message != WM_POWERBROADCAST:
        return None
    if wparam == PBT_APMSUSPEND:
        return PowerTransition.SUSPEND
    if wparam in _RESUME_EVENTS:
        return PowerTransition.RESUME
    return None


# ----------------------------------------------------------------------
# Координатор переходов питания.
# ----------------------------------------------------------------------


class _Restartable(Protocol):
    """Минимум от :class:`~ayris.workers.manager.WorkerManager` для пробуждения."""

    def restart_scope(self, scope: RestartScope, settings_reason: str = ...) -> int: ...


class _Connectivity(Protocol):
    """Монитор связи умеет перепроверить состояние немедленно."""

    def check_now(self) -> Any: ...


class _Resumable(Protocol):
    """Планировщик таймеров умеет пересчитать просроченные срабатывания."""

    def resume(self) -> None: ...


class PowerCoordinator:
    """Переносимая логика переходов сна и пробуждения.

    На засыпание — только мягкая парковка: событие :class:`SystemSuspending` на
    шине, чтобы подсистемы сами свернули активность (открытый аудиопоток всё равно
    станет недействительным). На пробуждение — активное восстановление в жёстком
    порядке: пересоздать аудиопоток (это РЕАЛЬНАЯ попытка заново открыть
    устройство, а не сверка со списком), перепроверить связь и пересчитать
    просроченные таймеры, затем объявить :class:`SystemResumed`.

    Все зависимости опциональны и утиные: без них соответствующий шаг молча
    пропускается, а координатор остаётся вызываемым программно и из тестов.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        worker_manager: _Restartable | None = None,
        timer_scheduler: _Resumable | None = None,
        connectivity: _Connectivity | None = None,
    ) -> None:
        self._bus = bus
        self._worker_manager = worker_manager
        self._timer_scheduler = timer_scheduler
        self._connectivity = connectivity

    def handle(self, transition: PowerTransition) -> None:
        """Разобрать переход питания в соответствующее действие."""
        if transition is PowerTransition.SUSPEND:
            self.on_suspend()
        else:
            self.on_resume()

    def on_suspend(self) -> None:
        """Засыпание: объявить парковку. Дескрипторы всё равно протухнут во сне."""
        _log.info("система уходит в сон — распускаю активные потоки")
        self._bus.publish(SystemSuspending())

    def on_resume(self) -> None:
        """Пробуждение: пересоздать аудио, перепроверить сеть, пересчитать таймеры."""
        _log.info("система проснулась — пересоздаю аудиопоток, проверяю сеть и таймеры")
        self._recreate_audio()
        self._recheck_connectivity()
        self._recompute_timers()
        self._bus.publish(SystemResumed())

    # -- шаги пробуждения ---------------------------------------------------

    def _recreate_audio(self) -> None:
        if self._worker_manager is None:
            return
        try:
            restarted = self._worker_manager.restart_scope(
                RestartScope.AUDIO, "пробуждение системы"
            )
        except Exception:
            _log.exception("не удалось пересоздать аудиопоток после пробуждения")
            return
        _log.info("пробуждение: перезапущено аудио-воркеров — %d", restarted)

    def _recheck_connectivity(self) -> None:
        if self._connectivity is None:
            return
        with contextlib.suppress(Exception):
            self._connectivity.check_now()

    def _recompute_timers(self) -> None:
        if self._timer_scheduler is None:
            return
        try:
            self._timer_scheduler.resume()
        except Exception:
            _log.exception("не удалось пересчитать таймеры после пробуждения")


# ----------------------------------------------------------------------
# Нативный фильтр событий Qt (только Windows).
# ----------------------------------------------------------------------


class _PowerMessage(ctypes.Structure):
    """Префикс WinAPI ``MSG`` на 64-битной Windows: ``hwnd``, ``message``, ``wParam``.

    Между ``message`` (``UINT``, 4 байта) и ``wParam`` (``WPARAM``, указательного
    размера) компилятор вставляет 4 байта выравнивания — ``_pad`` их занимает,
    иначе ``wParam`` читается со смещения 12 вместо 16 и ломается.
    """

    _fields_ = (
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
        ("_pad", ctypes.c_uint),
        ("wparam", ctypes.c_size_t),
    )


class _NativeMessagePointer(Protocol):
    def __int__(self) -> int: ...


def _make_power_filter(coordinator: PowerCoordinator) -> Any:
    """Собрать нативный фильтр событий Qt. Лениво — ради импорта без Qt."""
    from PySide6.QtCore import QAbstractNativeEventFilter, QByteArray

    class _PowerEventFilter(QAbstractNativeEventFilter):
        def nativeEventFilter(  # noqa: N802
            self,
            event_type: QByteArray | bytes | bytearray | memoryview,
            message: int | None,
        ) -> object:
            name = (
                bytes(event_type.data())
                if isinstance(event_type, QByteArray)
                else bytes(event_type)
            )
            if name != _WINDOWS_EVENT_TYPE or message is None:
                return False, 0
            address = int(cast("_NativeMessagePointer", message))
            native = ctypes.cast(address, ctypes.POINTER(_PowerMessage)).contents
            transition = classify_power_message(int(native.message), int(native.wparam))
            if transition is not None:
                coordinator.handle(transition)
            return False, 0

    return _PowerEventFilter()


def install_power_events(
    app: AyrisApp,
    worker_manager: _Restartable | None = None,
    *,
    qapp: QApplication | None = None,
    timer_scheduler: _Resumable | None = None,
    connectivity: _Connectivity | None = None,
) -> PowerCoordinator:
    """Смонтировать обработку сна и пробуждения в жизненный цикл приложения.

    На Windows с переданным ``qapp`` регистрирует нативный фильтр событий,
    который ловит ``WM_POWERBROADCAST`` и зовёт координатор. Без Windows или без
    ``QApplication`` фильтр не ставится, но координатор всё равно возвращается —
    его можно дёрнуть программно и он покрыт тестами.
    """
    from ayris.core.app import Component, LifecycleStage

    coordinator = PowerCoordinator(
        bus=app.bus,
        worker_manager=worker_manager,
        timer_scheduler=timer_scheduler,
        connectivity=connectivity,
    )

    if sys.platform != "win32" or qapp is None:
        return coordinator

    filters: list[Any] = []

    def start() -> None:
        native_filter = _make_power_filter(coordinator)
        qapp.installNativeEventFilter(native_filter)
        filters.append(native_filter)

    def stop() -> None:
        for native_filter in filters:
            with contextlib.suppress(Exception):
                qapp.removeNativeEventFilter(native_filter)
        filters.clear()

    app.add_component(
        Component(name="события питания", stage=LifecycleStage.GUI, start=start, stop=stop)
    )
    return coordinator
