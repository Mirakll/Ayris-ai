"""Краш воркера посреди макроса → сбой всплывает, движок и подсистема выживают.

``tests/unit/test_workers.py`` проверяет, что убитый воркер поднимается сам и что
``call_sync`` отдаёт :class:`WorkerCrashError`; ``tests/stress/test_leaks.py`` гоняет
макрос-движок на реестре-заглушке. Никто не проверяет их ВМЕСТЕ: что когда действие
макроса уводит работу в настоящий дочерний процесс, а тот падает прямо во время
вызова, крах доходит до движка как провал шага — не зависанием и не проглоченным
исключением, — движок отдаёт провальный отчёт и остаётся живым, а воркер-менеджер
перезапускает процесс. Это и есть шов «краш воркера посреди макроса» из задачи 70.

Процесс настоящий: класс воркера живёт в ``tests/fixtures/echo_worker.py`` (под
``spawn`` ребёнок реимпортирует модуль своего класса — класс в этом файле затащил
бы pytest в каждого ребёнка), метод ``die`` выходит из процесса без раскрутки, а
запускать набор надо через ``python -m pytest``.

Мост между макросом и воркером — :class:`WorkerBackedRunner`: он повторяет контракт
``ActionRegistry`` (``has`` + ``execute``, «наружу не улетает ничего голого»), только
работа за ним — вызов воркера. Реестр-заглушка ``_NullRegistry`` из ``test_leaks``
показывает, что раннер-двойник для движка каноничен; здесь двойник не молчит, а
падает вместе с процессом.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.errors import MacroBlockError
from ayris.actions.macros.schema import CommandModel
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError
from ayris.core.events import EventBus, MacroFailed, WorkerCrashed, WorkerRestarted
from ayris.workers.manager import WorkerManager
from ayris.workers.protocol import WorkerCrashError, WorkerError
from ayris.workers.registry import WorkerSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration

#: Где лежит спавн-безопасный класс воркера (не в этом файле — см. модульный докстринг).
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Точка входа эхо-воркера: ``модуль:класс`` внутри :data:`FIXTURES`.
ECHO = "echo_worker:EchoWorker"

#: Имя синтетического действия, за которым прячется вызов воркера. Логическим блоком
#: движка оно не является, поэтому доходит до раннера через ``has``/``execute``.
OFFLOAD = "OffloadToWorker"

#: Спавн интерпретатора и импорт Ayris на холодном кэше не мгновенны; порог лишь короче
#: точки, где зависание застопорит набор.
START_TIMEOUT = 60.0

#: Код выхода, которым ``die`` роняет процесс, — он же уезжает в событие краха.
EXIT_CODE = 7


def wait_for(predicate: Callable[[], object], timeout: float = 30.0) -> bool:
    """Опрашивать ``predicate`` до истины или до конца времени."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def echo_spec() -> WorkerSpec:
    """Спека эхо-воркера: быстрый перезапуск, чтобы крах восстанавливался в тесте."""
    return WorkerSpec(
        name="echo",
        kind="test",
        entrypoint=ECHO,
        python_path=(str(FIXTURES),),
        start_timeout=START_TIMEOUT,
        call_timeout=20.0,
        stop_timeout=5.0,
        heartbeat_interval=0.5,
        heartbeat_misses=6,
        restart_delay=0.05,
        max_restart_delay=0.2,
    )


class WorkerBackedRunner:
    """``ActionRunner``, чьё единственное действие уводит работу в настоящий воркер.

    Повторяет контракт ``ActionRegistry``, на который опирается интерпретатор: ``has``
    отвечает за тип блока, ``execute`` выполняет работу и любой сбой превращает в
    типизированную ``ActionError``, которую движок ловит. Работа здесь — вызов воркера,
    так что смерть процесса приходит как :class:`WorkerCrashError`; раннер её ловит,
    запоминает (чтобы тест увидел настоящий крах у источника) и всплывает наверх ошибкой
    действия — ровно как это сделал бы реестр.
    """

    def __init__(self, manager: WorkerManager, *, worker: str = "echo") -> None:
        self._manager = manager
        self._worker = worker
        self.crashes: list[WorkerError] = []

    def has(self, name: str) -> bool:
        return name == OFFLOAD

    def execute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        call = dict(params or {})
        method = str(call.get("method", "die"))
        payload = call.get("params", {})
        try:
            value = self._manager.call_sync(self._worker, method, payload)
        except WorkerError as exc:
            self.crashes.append(exc)
            raise ActionError(
                f"worker {self._worker!r} failed during {method!r}: {exc}",
                user_message="Фоновая задача прервалась.",
            ) from exc
        return ActionResult.done(value=value)


def offload_command() -> CommandModel:
    """Макрос из одного действия: оно уводит работу в воркер, который выйдет с кодом."""
    return CommandModel.model_validate(
        {
            "name": "offload",
            "actions": [
                {"type": OFFLOAD, "params": {"method": "die", "params": {"code": EXIT_CODE}}}
            ],
        }
    )


def clean_command() -> CommandModel:
    """Тривиальный макрос без воркера: доказывает, что движок пережил краш."""
    return CommandModel.model_validate(
        {"name": "clean", "actions": [{"type": "SetVar", "params": {"name": "n", "value": "1"}}]}
    )


@pytest.fixture
def bus() -> EventBus:
    """Шина, доставляющая инлайново на потоке-издателе, — как в тестах воркеров."""
    return EventBus(thread_id=None)


@pytest.fixture
def manager(bus: EventBus) -> Iterator[WorkerManager]:
    """Менеджер, который гасится даже при падении теста."""
    instance = WorkerManager(bus)
    try:
        yield instance
    finally:
        instance.shutdown()


def _collect(bus: EventBus, event_type: type[Any]) -> list[Any]:
    """Подписаться сильной ссылкой и вернуть список, в который падают события."""
    received: list[Any] = []
    bus.subscribe(event_type, received.append, weak=False)
    return received


def test_worker_crash_mid_macro_surfaces_and_engine_survives(
    bus: EventBus, manager: WorkerManager
) -> None:
    """Воркер падает посреди макроса: сбой всплывает, движок жив, процесс вернулся.

    Один живой стык от начала до конца: команда с действием-мостом, настоящий дочерний
    процесс, его смерть во время вызова. Проверяется, что крах дошёл до движка провалом
    шага (а не зависанием), что у источника лежит настоящий :class:`WorkerCrashError`,
    что событие :class:`MacroFailed` ушло на шину, что следующая команда всё ещё
    проходит, и что менеджер сам поднял процесс и он снова считает.
    """
    crashes = _collect(bus, WorkerCrashed)
    restarts = _collect(bus, WorkerRestarted)
    failures = _collect(bus, MacroFailed)

    manager.register(echo_spec())
    manager.start("echo")
    assert manager.is_ready("echo")
    first_pid = manager.status()[0].pid

    runner = WorkerBackedRunner(manager)
    with MacroEngine(runner, bus=bus) as engine:
        report = engine.run(offload_command())

        # Крах всплыл — не зависанием: движок отдал провальный отчёт с типизированной
        # ошибкой, указывающей на упавший шаг.
        assert report.failed
        assert isinstance(report.error, MacroBlockError)
        assert report.error.block == OFFLOAD
        assert report.error.path == "actions[0]"

        # У источника сбоя — настоящий WorkerCrashError: процесс действительно умер.
        assert len(runner.crashes) == 1
        assert isinstance(runner.crashes[0], WorkerCrashError)

        # Событие сбоя макроса ушло на шину — для оверлея, истории и звука ошибки.
        assert wait_for(lambda: bool(failures))
        assert failures[0].command == "offload"

        # Сам движок жив: следующая команда (без воркера) проходит до конца.
        assert engine.run(clean_command()).ok

    # Воркер вернулся сам и снова считает — краш посреди макроса не убил подсистему.
    assert wait_for(lambda: manager.is_ready("echo")), "воркер не перезапустился"
    assert manager.status()[0].pid not in (None, first_pid)
    assert manager.call_sync("echo", "add", {"a": 1, "b": 2}) == 3

    # А события менеджера подтверждают именно тот крах, что устроил макрос.
    assert wait_for(lambda: bool(crashes and restarts))
    assert crashes[0].worker == "echo"
    assert crashes[0].exit_code == EXIT_CODE
    assert restarts[0].worker == "echo"
