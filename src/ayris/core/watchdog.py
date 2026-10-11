"""Задача 69: watchdog главного процесса.

Supervisor воркеров (:class:`ayris.workers.manager.WorkerManager`) стережёт
дочерние процессы, но не видит собственный главный процесс: если подвис цикл
событий Qt, воркеры продолжают слать heartbeat в пустоту. Этот watchdog закрывает
брешь.

Он делает две вещи из фонового потока-демона:

* **Отзывчивость цикла событий.** UI-поток тикает :meth:`Watchdog.beat` по
  таймеру. Если удары прекратились дольше ``ui_timeout`` — цикл подвис (мягкая
  тревога), дольше ``ui_fatal_timeout`` — считаем это дедлоком и идём на
  контролируемое завершение.
* **Heartbeat воркеров как страховка.** Поверх собственного контроля менеджера
  (у watchdog порог заведомо больше): если воркер числится живым, но давно молчит,
  watchdog просит менеджер его перезапустить — мягкое восстановление подсистемы.

Мягкое восстановление ограничено числом попыток в окне: заклинивший узел не
перезапускается вечно. Когда цикл событий не воскресить, watchdog записывает
причину для следующего запуска (рядом с ``crash.log``) и передаёт управление
контролируемому завершению, которое сохраняет состояние.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PySide6.QtWidgets import QApplication

    from ayris.core.app import AyrisApp
    from ayris.core.paths import AppPaths
    from ayris.workers.manager import WorkerManager, WorkerSummary

_log = logging.getLogger("ayris.watchdog")

#: Причина последнего аварийного завершения, читается при следующем запуске.
SHUTDOWN_REASON_NAME = "last_exit_reason.json"

#: Код выхода, которым watchdog добивает зависший процесс.
EXIT_WATCHDOG = 70

__all__ = [
    "EXIT_WATCHDOG",
    "SHUTDOWN_REASON_NAME",
    "Watchdog",
    "WatchdogAction",
    "WatchdogVerdict",
    "install_watchdog",
    "read_shutdown_reason",
    "record_shutdown_reason",
]


class WatchdogAction(StrEnum):
    """Что watchdog решил сделать по итогам одной проверки."""

    HEALTHY = "healthy"
    RECOVER = "recover"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class WatchdogVerdict:
    """Итог одной проверки: действие, причина и затронутая подсистема."""

    action: WatchdogAction
    reason: str = ""
    subsystem: str = ""


# ----------------------------------------------------------------------
# Причина завершения для следующего запуска.
# ----------------------------------------------------------------------


def record_shutdown_reason(
    paths: AppPaths, reason: str, *, kind: str = "watchdog", at: str = ""
) -> None:
    """Записать причину аварийного завершения рядом с ``crash.log``.

    Никогда не бросает: сбой записи не должен мешать самому завершению.
    """
    payload = {"kind": kind, "reason": reason, "at": at}
    try:
        target = paths.logs_dir / SHUTDOWN_REASON_NAME
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except OSError:
        _log.exception("не удалось записать причину завершения")


def read_shutdown_reason(paths: AppPaths, *, consume: bool = True) -> dict[str, Any] | None:
    """Прочитать причину прошлого аварийного завершения, если она есть.

    По умолчанию удаляет файл (``consume``), чтобы причина всплыла один раз.
    """
    target = paths.logs_dir / SHUTDOWN_REASON_NAME
    if not target.exists():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _log.warning("файл причины завершения нечитаем, игнорирую")
        data = None
    if consume:
        with contextlib.suppress(OSError):
            target.unlink()
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------
# Сам watchdog.
# ----------------------------------------------------------------------


class Watchdog:
    """Страж главного процесса: отзывчивость цикла событий и heartbeat воркеров.

    :meth:`check` — чистая функция от впрыснутых часов и поставщика состояния
    воркеров, поэтому тестируется без потоков и без реального сна. :meth:`start`
    запускает поток-демон, который вызывает :meth:`check` каждые ``interval``
    секунд и действует по вердикту.
    """

    def __init__(
        self,
        *,
        health: Callable[[], Sequence[WorkerSummary]],
        on_recover: Callable[[WatchdogVerdict], None],
        on_shutdown: Callable[[WatchdogVerdict], None],
        clock: Callable[[], float] = time.monotonic,
        interval: float = 2.0,
        ui_timeout: float = 8.0,
        ui_fatal_timeout: float = 30.0,
        worker_hang_grace: float = 30.0,
        max_recover_attempts: int = 3,
        recover_window: float = 120.0,
    ) -> None:
        self._health = health
        self._on_recover = on_recover
        self._on_shutdown = on_shutdown
        self._clock = clock
        self._interval = interval
        self._ui_timeout = ui_timeout
        self._ui_fatal_timeout = ui_fatal_timeout
        self._worker_hang_grace = worker_hang_grace
        self._max_recover_attempts = max_recover_attempts
        self._recover_window = recover_window
        self._lock = threading.Lock()
        self._last_ui_beat = clock()
        self._recover_history: dict[str, list[float]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- жизненный цикл -----------------------------------------------------

    def beat(self) -> None:
        """Отметка живости цикла событий. Вызывается UI-потоком по таймеру."""
        with self._lock:
            self._last_ui_beat = self._clock()

    def start(self) -> None:
        """Запустить фоновый поток проверки. Повторный вызов ничего не делает."""
        if self._thread is not None:
            return
        self.beat()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="ayris-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Остановить поток проверки и дождаться его завершения."""
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=self._interval * 2)

    # -- логика -------------------------------------------------------------

    def check(self) -> WatchdogVerdict:
        """Одна проверка. Детерминирована при впрыснутых часах и поставщике."""
        now = self._clock()
        with self._lock:
            stall = now - self._last_ui_beat
        if stall >= self._ui_fatal_timeout:
            return WatchdogVerdict(
                WatchdogAction.SHUTDOWN,
                reason=f"цикл интерфейса не отвечает {stall:.0f} с — вероятен дедлок",
                subsystem="gui",
            )
        if stall >= self._ui_timeout:
            return WatchdogVerdict(
                WatchdogAction.RECOVER,
                reason=f"цикл интерфейса подвис на {stall:.0f} с",
                subsystem="gui",
            )
        for summary in self._health():
            age = summary.last_heartbeat_age
            if summary.alive and age is not None and age >= self._worker_hang_grace:
                return WatchdogVerdict(
                    WatchdogAction.RECOVER,
                    reason=f"воркер «{summary.name}» молчит {age:.0f} с",
                    subsystem=summary.name,
                )
        return WatchdogVerdict(WatchdogAction.HEALTHY)

    def _allow_recover(self, key: str) -> bool:
        now = self._clock()
        history = [t for t in self._recover_history.get(key, []) if now - t < self._recover_window]
        if len(history) >= self._max_recover_attempts:
            self._recover_history[key] = history
            return False
        history.append(now)
        self._recover_history[key] = history
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                verdict = self.check()
            except Exception:
                _log.exception("watchdog: проверка упала")
                continue
            if verdict.action is WatchdogAction.HEALTHY:
                continue
            if verdict.action is WatchdogAction.RECOVER:
                self._handle_recover(verdict)
            else:
                self._handle_shutdown(verdict)
                return

    def _handle_recover(self, verdict: WatchdogVerdict) -> None:
        key = verdict.subsystem or "gui"
        if not self._allow_recover(key):
            _log.error(
                "watchdog: «%s» не восстановить за %d попыток — оставляю supervisor'у",
                key,
                self._max_recover_attempts,
            )
            return
        _log.warning("watchdog: мягкое восстановление «%s» (%s)", key, verdict.reason)
        try:
            self._on_recover(verdict)
        except Exception:
            _log.exception("watchdog: обработчик восстановления упал")

    def _handle_shutdown(self, verdict: WatchdogVerdict) -> None:
        _log.critical("watchdog: контролируемое завершение — %s", verdict.reason)
        try:
            self._on_shutdown(verdict)
        except Exception:
            _log.exception("watchdog: обработчик завершения упал")


# ----------------------------------------------------------------------
# Монтаж в жизненный цикл приложения.
# ----------------------------------------------------------------------

#: Как часто UI-поток отмечается живым.
_BEAT_INTERVAL_MS = 1000


def _install_beat_timer(qapp: QApplication, dog: Watchdog, holder: list[Any]) -> None:
    from PySide6.QtCore import QTimer

    timer = QTimer(qapp)
    timer.timeout.connect(dog.beat)
    timer.start(_BEAT_INTERVAL_MS)
    holder.append(timer)


def _request_qt_shutdown(qapp: QApplication, grace: float) -> None:
    def _hard() -> None:
        _log.critical("watchdog: цикл не ответил на выход — принудительное завершение")
        os._exit(EXIT_WATCHDOG)

    # Мягкий quit постится в UI-поток; если он в дедлоке, событие не разберётся,
    # поэтому запасной таймер из независимого потока добивает процесс.
    threading.Timer(grace, _hard).start()
    with contextlib.suppress(Exception):
        qapp.quit()


def install_watchdog(
    app: AyrisApp,
    manager: WorkerManager,
    *,
    qapp: QApplication | None = None,
    request_shutdown: Callable[[WatchdogVerdict], None] | None = None,
    hard_exit_grace: float = 5.0,
) -> Watchdog:
    """Смонтировать watchdog в жизненный цикл приложения.

    При переданном ``qapp`` watchdog сам заводит QTimer, который тикает
    :meth:`Watchdog.beat` из UI-потока, и завершает процесс через ``qapp.quit``
    с принудительным добиванием, если цикл событий не ответил. ``request_shutdown``
    позволяет подменить это поведение (например, сначала сохранить состояние).
    """
    from ayris.core.app import Component, LifecycleStage

    def on_recover(verdict: WatchdogVerdict) -> None:
        if verdict.subsystem in ("", "gui"):
            # Зависший из другого потока цикл интерфейса не расшевелить — это
            # забота пути завершения, а не мягкого восстановления.
            return
        with contextlib.suppress(Exception):
            manager.restart(verdict.subsystem, reason="watchdog: подвис")

    def on_shutdown(verdict: WatchdogVerdict) -> None:
        record_shutdown_reason(app.paths, verdict.reason)
        if request_shutdown is not None:
            request_shutdown(verdict)
        elif qapp is not None:
            _request_qt_shutdown(qapp, hard_exit_grace)
        else:
            os._exit(EXIT_WATCHDOG)

    dog = Watchdog(health=manager.status, on_recover=on_recover, on_shutdown=on_shutdown)
    timers: list[Any] = []

    def start() -> None:
        dog.start()
        if qapp is not None:
            _install_beat_timer(qapp, dog, timers)

    def stop() -> None:
        dog.stop()
        for timer in timers:
            with contextlib.suppress(Exception):
                timer.stop()
        timers.clear()

    app.add_component(Component(name="watchdog", stage=LifecycleStage.GUI, start=start, stop=stop))
    return dog
