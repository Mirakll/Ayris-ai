"""Задача 69: матрица сценариев отказа и единый слой сообщений о них.

Этот модуль ничего не изобретает заново. Устойчивость Ayris уже собрана из
готовых частей — supervisor воркеров с бэкоффом и heartbeat
(:class:`ayris.workers.manager.WorkerManager`), монитор связи с гистерезисом
(:class:`ayris.core.connectivity.ConnectivityMonitor`), машина состояний с
переходом в ERROR (:class:`ayris.core.state.StateMachine`) и типизированные
ошибки (:mod:`ayris.core.errors`). Здесь два артефакта, которых не хватало:

* **Матрица отказов** (:data:`FAILURE_MATRIX`) — декларативная таблица, по одной
  строке на обработанный сценарий, с явным ожидаемым поведением: что ломается,
  какой модуль это ловит, какая типизированная ошибка, состояние интерфейса,
  единственное сообщение пользователю, ожидается ли автовосстановление. Она
  проверяется тестами и служит чек-листом, а не только документацией.
* **Единый слой сообщений** (:class:`FailureNotifier`) — подписывается на события
  отказов на шине и превращает их в НЕ дублирующиеся
  :class:`~ayris.core.events.NotificationRequested`: одна формулировка на класс
  проблемы, с гашением повторов и сбросом ключа после восстановления.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ayris.core.errors import (
    ActionTimeout,
    AudioError,
    ConfigError,
    LlmError,
    ModelError,
    SttError,
    TtsError,
)
from ayris.core.events import (
    ActionFailed,
    MacroFailed,
    ModelDownloadFailed,
    NotificationRequested,
    OnlineStatusChanged,
    WorkerCrashed,
    WorkerRestarted,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayris.core.app import AyrisApp
    from ayris.core.errors import AyrisError
    from ayris.core.events import EventBus
    from ayris.workers.manager import WorkerManager

_log = logging.getLogger("ayris.resilience")

__all__ = [
    "FAILURE_MATRIX",
    "FailureNotifier",
    "FailureScenario",
    "ProblemClass",
    "install_resilience",
    "scenario",
]


class ProblemClass(StrEnum):
    """Класс проблемы — единица дедупликации и одна формулировка на класс."""

    WORKER = "worker"
    OFFLINE = "offline"
    MODEL = "model"
    ACTION = "action"
    DISK = "disk"
    INTERNAL = "internal"


@dataclass(frozen=True, slots=True)
class FailureScenario:
    """Одна строка матрицы отказов: сценарий и его контракт обработки.

    Args:
        key: Стабильный идентификатор строки; на него ссылаются тесты.
        title: Человекочитаемое название сценария.
        trigger: Что именно ломается.
        handled_by: Модуль или класс, который отвечает за обработку.
        problem_class: К какому классу проблемы относится для слоя сообщений.
        ui_state: Ожидаемое состояние интерфейса ПОСЛЕ обработки — оверлей и трей
            никогда не остаются висеть в «слушаю».
        user_message: Единственное сообщение, показываемое пользователю (без
            трейсбека), одно на класс проблемы.
        recovers: Ожидается ли автоматическое восстановление подсистемы.
        error_type: Типизированная ошибка из :mod:`ayris.core.errors`, в которую
            заворачивается внешний вызов, либо ``None`` для инфраструктурных
            сбоев на уровне процесса.
        note: Оговорки контракта — бэкофф, предел попыток, гистерезис.
    """

    key: str
    title: str
    trigger: str
    handled_by: str
    problem_class: ProblemClass
    ui_state: str
    user_message: str
    recovers: bool
    error_type: type[AyrisError] | None = None
    note: str = ""


# ----------------------------------------------------------------------
# Матрица отказов. Порядок — от воркеров к внешним ресурсам и действиям.
# ----------------------------------------------------------------------

FAILURE_MATRIX: tuple[FailureScenario, ...] = (
    FailureScenario(
        key="worker_crash_audio",
        title="Аварийное завершение аудио-воркера",
        trigger="Процесс захвата звука упал: исключение, нативный сбой или убит извне.",
        handled_by="workers.manager.WorkerManager",
        problem_class=ProblemClass.WORKER,
        ui_state="Оверлей выходит из «слушаю» в покой; уровень сбрасывается в ноль.",
        user_message="Подсистема «звук» перезапускается после сбоя.",
        recovers=True,
        error_type=AudioError,
        note=(
            "Экспоненциальный бэкофф и предел попыток; при исчерпании "
            "статус FAILED и сообщение «отключена»."
        ),
    ),
    FailureScenario(
        key="worker_crash_stt",
        title="Аварийное завершение воркера распознавания",
        trigger="Процесс STT упал во время или между распознаваниями.",
        handled_by="workers.manager.WorkerManager",
        problem_class=ProblemClass.WORKER,
        ui_state="Оверлей возвращается в покой; текущий запрос завершается ошибкой, не зависает.",
        user_message="Подсистема «распознавание речи» перезапускается после сбоя.",
        recovers=True,
        error_type=SttError,
        note="Ожидающие вызовы падают с WorkerCrashError; движок поднимается заново с бэкоффом.",
    ),
    FailureScenario(
        key="worker_crash_tts",
        title="Аварийное завершение воркера синтеза",
        trigger="Процесс TTS упал во время озвучивания или простоя.",
        handled_by="workers.manager.WorkerManager",
        problem_class=ProblemClass.WORKER,
        ui_state="Оверлей выходит из «говорю» в покой; воспроизведение обрывается чисто.",
        user_message="Подсистема «синтез речи» перезапускается после сбоя.",
        recovers=True,
        error_type=TtsError,
        note="Роутер TTS откатывается на офлайн-движок; повторный сбой не зацикливается.",
    ),
    FailureScenario(
        key="worker_crash_llm",
        title="Аварийное завершение воркера языковой модели",
        trigger="Процесс LLM упал во время генерации ответа.",
        handled_by="workers.manager.WorkerManager",
        problem_class=ProblemClass.WORKER,
        ui_state="Оверлей выходит из «думаю» в покой; поток ответа закрывается.",
        user_message="Подсистема «языковая модель» перезапускается после сбоя.",
        recovers=True,
        error_type=LlmError,
        note="См. также «обрыв LLM-стрима»: частичный ответ не теряет управление.",
    ),
    FailureScenario(
        key="internet_lost",
        title="Потеря интернета в онлайн-запросе",
        trigger="Облачный STT/TTS/LLM-запрос не доходит: обрыв связи или таймаут.",
        handled_by="core.connectivity.ConnectivityMonitor + роутеры STT/TTS",
        problem_class=ProblemClass.OFFLINE,
        ui_state="Индикатор связи гаснет; движки переключаются на локальные, оверлей не виснет.",
        user_message=(
            "Облачные сервисы недоступны — перехожу на локальные движки, " "пока связь не вернётся."
        ),
        recovers=True,
        error_type=None,
        note=(
            "Гистерезис (RECOVERY_CONFIRMATIONS=2) гасит дребезг online↔offline; "
            "откат не зацикливается."
        ),
    ),
    FailureScenario(
        key="audio_device_lost",
        title="Исчезновение аудио-устройства при прослушивании",
        trigger="Микрофон отключён или перехвачен, поток обрывается на лету.",
        handled_by="workers.audio_worker (переоткрытие) + WorkerManager",
        problem_class=ProblemClass.WORKER,
        ui_state=(
            "Оверлей выходит из «слушаю» в покой; при возврате устройства "
            "прослушивание продолжается."
        ),
        user_message="Подсистема «звук» перезапускается после сбоя.",
        recovers=True,
        error_type=AudioError,
        note="Проверяется РЕАЛЬНОЕ открытие потока, а не только наличие устройства в списке.",
    ),
    FailureScenario(
        key="model_missing",
        title="Модель недоступна",
        trigger="Файл модели отсутствует на диске или путь не читается.",
        handled_by="core.paths + workers.*_worker (загрузка модели)",
        problem_class=ProblemClass.MODEL,
        ui_state="Соответствующий движок отключён; оверлей в покое, а не в ложном «готов».",
        user_message="Не удалось загрузить или проверить модель.",
        recovers=False,
        error_type=ModelError,
        note="Воркер завершается в FAILED; пользователь ставит модель во вкладке «Обновления».",
    ),
    FailureScenario(
        key="model_corrupt",
        title="Модель повреждена",
        trigger="Несовпадение SHA256 или битый .onnx/.gguf при загрузке.",
        handled_by="core.models_download (проверка) + workers.*_worker",
        problem_class=ProblemClass.MODEL,
        ui_state="Движок не поднимается; оверлей в покое; предлагается перекачать модель.",
        user_message="Не удалось загрузить или проверить модель.",
        recovers=False,
        error_type=ModelError,
        note="Битый файл не запускается втихую; сбой виден в логе и в сообщении.",
    ),
    FailureScenario(
        key="disk_full",
        title="Переполнение диска",
        trigger="Нет места при загрузке модели или записи лога.",
        handled_by="core.paths.AppPaths.ensure_directories + загрузчик моделей",
        problem_class=ProblemClass.DISK,
        ui_state="Операция отменяется; оверлей в покое; частичный файл убирается.",
        user_message="Недостаточно места на диске. Освободите место и повторите.",
        recovers=False,
        error_type=ConfigError,
        note="Запись лога с деградацией: сбой самого лога не должен ронять приложение.",
    ),
    FailureScenario(
        key="action_timeout",
        title="Зависание действия",
        trigger="Обработчик действия не завершается за отведённое время.",
        handled_by="actions.registry.ActionRegistry (таймаут)",
        problem_class=ProblemClass.ACTION,
        ui_state="Действие снимается по таймауту; оверлей возвращается в покой.",
        user_message="Действие не успело выполниться.",
        recovers=True,
        error_type=ActionTimeout,
        note="Таймаут заворачивается в типизированную ActionTimeout, не в голый Exception.",
    ),
    FailureScenario(
        key="llm_stream_dropped",
        title="Обрыв LLM-стрима на середине",
        trigger="Соединение с моделью рвётся посреди генерации ответа.",
        handled_by="nlu.llm.* (стрим) + core.pipeline",
        problem_class=ProblemClass.ACTION,
        ui_state="Частичный ответ фиксируется; оверлей выходит из «думаю» в покой.",
        user_message="Языковая модель недоступна.",
        recovers=True,
        error_type=LlmError,
        note="Отмена пользователем — это LlmCancelledError и НЕ уведомление.",
    ),
)

_BY_KEY: dict[str, FailureScenario] = {row.key: row for row in FAILURE_MATRIX}


def scenario(key: str) -> FailureScenario:
    """Строка матрицы по ключу. Бросает ``KeyError``, если ключ неизвестен."""
    return _BY_KEY[key]


# ----------------------------------------------------------------------
# Единый слой сообщений об отказах.
# ----------------------------------------------------------------------

_SUBSYSTEM_LABELS: dict[str, str] = {
    "audio": "звук",
    "stt": "распознавание речи",
    "tts": "синтез речи",
    "llm": "языковая модель",
}


def _subsystem(name: str) -> str:
    return _SUBSYSTEM_LABELS.get(name, name)


class FailureNotifier:
    """Единый слой сообщений об отказах.

    Слушает события отказов на шине и превращает их в НЕ дублирующиеся
    :class:`~ayris.core.events.NotificationRequested`. Одна формулировка на класс
    проблемы; повтор того же ключа в пределах ``cooldown`` секунд гасится, чтобы
    буря одинаковых сбоев не выдала десять всплывающих подсказок подряд.

    Восстановление связи выдаёт короткое сообщение и сбрасывает ключ, так что
    следующий обрыв снова будет замечен. Штатный перезапуск воркера намеренно
    молчит: пользователь уже получил одно предупреждение о сбое, а мелькание
    «упал/поднялся» на дребезжащем движке — это шум, а не информация.
    """

    def __init__(
        self,
        bus: EventBus,
        *,
        manager: WorkerManager | None = None,
        clock: Callable[[], float] = time.monotonic,
        cooldown: float = 30.0,
    ) -> None:
        self._bus = bus
        self._manager = manager
        self._clock = clock
        self._cooldown = cooldown
        self._last_shown: dict[str, float] = {}
        self._unsubscribe: list[Callable[[], None]] = []

    # -- жизненный цикл -----------------------------------------------------

    def install(self) -> None:
        """Подписаться на события отказов. Повторный вызов ничего не делает."""
        if self._unsubscribe:
            return
        self._unsubscribe.extend(
            (
                self._bus.subscribe(WorkerCrashed, self._on_worker_crashed, weak=False),
                self._bus.subscribe(WorkerRestarted, self._on_worker_restarted, weak=False),
                self._bus.subscribe(OnlineStatusChanged, self._on_online, weak=False),
                self._bus.subscribe(ModelDownloadFailed, self._on_model_failed, weak=False),
                self._bus.subscribe(ActionFailed, self._on_action_failed, weak=False),
                self._bus.subscribe(MacroFailed, self._on_macro_failed, weak=False),
            )
        )

    def close(self) -> None:
        """Отписаться и забыть историю показов."""
        for unsubscribe in self._unsubscribe:
            unsubscribe()
        self._unsubscribe.clear()
        self._last_shown.clear()

    # -- дедупликация -------------------------------------------------------

    def _should_show(self, key: str) -> bool:
        now = self._clock()
        last = self._last_shown.get(key)
        if last is not None and (now - last) < self._cooldown:
            return False
        self._last_shown[key] = now
        return True

    def _clear(self, key: str) -> None:
        self._last_shown.pop(key, None)

    def _emit(self, key: str, *, title: str, message: str, level: str) -> None:
        if not self._should_show(key):
            _log.debug("подавляю повтор уведомления %s", key)
            return
        _log.log(
            logging.ERROR if level == "error" else logging.WARNING,
            "отказ [%s]: %s",
            key,
            message,
        )
        self._bus.publish(NotificationRequested(title=title, message=message, level=level))

    # -- обработчики событий отказов ---------------------------------------

    def _is_terminal(self, event: WorkerCrashed) -> bool:
        if self._manager is None:
            return False
        try:
            return event.restarts > self._manager.spec(event.worker).max_restarts
        except Exception:
            # Незнакомый воркер или снятая регистрация — считаем не терминальным.
            return False

    def _on_worker_crashed(self, event: WorkerCrashed) -> None:
        subsystem = _subsystem(event.worker)
        if self._is_terminal(event):
            self._emit(
                f"worker:{event.worker}:failed",
                title="Подсистема отключена",
                message=(
                    f"Подсистема «{subsystem}» отключена после повторных сбоев. "
                    "Проверьте её настройки и модели."
                ),
                level="error",
            )
        else:
            self._emit(
                f"worker:{event.worker}",
                title="Сбой подсистемы",
                message=f"Подсистема «{subsystem}» перезапускается после сбоя.",
                level="warning",
            )

    def _on_worker_restarted(self, event: WorkerRestarted) -> None:
        # Ручной подъём отключённого движка снимает терминальный ключ, чтобы
        # новый отказ снова уведомил. Ключ предупреждения не трогаем: иначе
        # дребезжащий воркер (упал→поднялся→упал) заспамит уведомлениями.
        self._clear(f"worker:{event.worker}:failed")

    def _on_online(self, event: OnlineStatusChanged) -> None:
        if event.online:
            self._clear("offline")
            self._emit(
                "online",
                title="Соединение восстановлено",
                message="Облачные сервисы снова доступны.",
                level="info",
            )
        else:
            self._clear("online")
            self._emit(
                "offline",
                title="Нет доступа к сети",
                message=(
                    "Облачные сервисы недоступны — перехожу на локальные движки, "
                    "пока связь не вернётся."
                ),
                level="warning",
            )

    def _on_model_failed(self, event: ModelDownloadFailed) -> None:
        if event.cancelled:
            # Пользователь сам отменил загрузку — это не отказ.
            return
        self._emit(
            f"model:{event.model_id}",
            title="Проблема с моделью",
            message=event.user_message or "Не удалось загрузить или проверить модель.",
            level="error",
        )

    def _on_action_failed(self, event: ActionFailed) -> None:
        self._emit(
            f"action:{event.action}",
            title="Команда не выполнена",
            message=event.user_message or "Не удалось выполнить команду.",
            level="warning",
        )

    def _on_macro_failed(self, event: MacroFailed) -> None:
        self._emit(
            f"macro:{event.path}",
            title="Макрос остановлен",
            message=event.user_message or "Ошибка при выполнении макроса.",
            level="warning",
        )


def install_resilience(app: AyrisApp, manager: WorkerManager | None = None) -> FailureNotifier:
    """Смонтировать единый слой сообщений об отказах в жизненный цикл приложения.

    Слой подписывается на шину на этапе ``STATE`` — раньше воркеров, чтобы поймать
    и записать в лог даже ранние сбои, — и отписывается при остановке.
    """
    from ayris.core.app import Component, LifecycleStage

    notifier = FailureNotifier(app.bus, manager=manager)
    app.add_component(
        Component(
            name="слой сообщений об отказах",
            stage=LifecycleStage.STATE,
            start=notifier.install,
            stop=notifier.close,
        )
    )
    return notifier
