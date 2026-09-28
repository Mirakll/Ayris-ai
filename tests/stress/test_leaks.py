"""Нагрузочные тесты раздела 24: утечки памяти, объектов и хендлов под маркером slow.

Три сценария, которые в обычном прогоне не гоняют, а держат для `nightly.yml`:
макрос-движок под циклом, шина событий под залпом, воркер-менеджер под серией
перезапусков. Каждый меряет не «быстро ли», а «не растёт ли»: RSS процесса,
число живых объектов (:func:`gc.get_objects`), а на Windows — число хендлов
(:meth:`psutil.Process.num_handles`) и живых дочерних процессов. Замер идёт по
одной схеме — прогреть, снять базовую точку, отработать измеряемый цикл, снять
вторую точку — потому что первый прогон всегда аллоцирует служебное (пул нитей,
кэши pydantic), и без прогрева дельта мерила бы разовую инициализацию, а не течь.

Пороги нарочно с запасом: аллокатор держит арены, поэтому RSS никогда не
возвращается ровно в ноль — миллиметровый порог мигал бы. Течь же копит
монотонно, её видно на любом разумном пороге. Маркер `slow` не исключён в
`addopts`, но обычный CI гоняет `-m "not slow"`; сюда цикл заходит только ночью.

Процессы настоящие: класс воркера живёт в ``tests/fixtures/echo_worker.py``
(под ``spawn`` ребёнок реимпортирует модуль своего класса — класс в этом файле
затащил бы pytest в каждого ребёнка), а запускать надо через ``python -m pytest``.
"""

from __future__ import annotations

import gc
import multiprocessing
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import psutil
import pytest

from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.schema import CommandModel
from ayris.actions.result import ActionResult
from ayris.core.events import EventBus, WorkerRestarted
from ayris.workers.manager import WorkerManager
from ayris.workers.registry import WorkerSpec

pytestmark = pytest.mark.slow

#: Где лежит спавн-безопасный класс воркера (не в этом файле — см. модульный докстринг).
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Точка входа эхо-воркера: ``модуль:класс`` внутри :data:`FIXTURES`.
ECHO = "echo_worker:EchoWorker"

#: Прогрев макрос-движка: разовые аллокации пула и кэшей до базовой точки.
MACRO_WARMUP = 100

#: Измеряемый цикл макроса — на нём копилась бы течь, если бы была.
MACRO_ITERS = 400

#: Залп по шине: столько событий один подписчик обязан получить без потерь.
EVENT_ITERS = 1000

#: Прогрев и измеряемая серия перезапусков воркера (каждый — новый интерпретатор).
RESTART_WARMUP = 5
RESTART_ITERS = 25

#: Запас RSS: аллокатор держит арены, точного нуля не бывает.
RSS_SLACK_BYTES = 24 * 1024 * 1024

#: Запас по числу живых объектов между двумя точками замера.
OBJECT_SLACK = 2000

#: Запас по числу хендлов процесса (Windows) за серию перезапусков.
HANDLE_SLACK = 64


def _objects() -> int:
    """Число живых объектов после сборки мусора — снимок для дельты."""
    gc.collect()
    return len(gc.get_objects())


def _rss() -> int:
    """Resident set size процесса в байтах."""
    return psutil.Process().memory_info().rss


def _handles() -> int:
    """Число хендлов процесса. Только Windows — иначе вызывающий не спрашивает."""
    return psutil.Process().num_handles()  # type: ignore[no-any-return,attr-defined]


def worker_children() -> list[multiprocessing.process.BaseProcess]:
    """Все живые процессы воркеров Ayris — их не должно копиться."""
    return [child for child in multiprocessing.active_children() if child.name.startswith("ayris-")]


def wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    """Опрашивать ``predicate`` до истины или до конца времени."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


class _NullRegistry:
    """Реестр-заглушка: чистый блок логики (``SetVar``) к нему не обращается."""

    def has(self, name: str) -> bool:
        return False

    def execute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        return ActionResult.done()


def _leak_macro() -> CommandModel:
    """Тривиальный макрос из одного логического блока — без реестра и без звука."""
    return CommandModel.model_validate(
        {"name": "leak", "actions": [{"type": "SetVar", "params": {"name": "n", "value": "1"}}]}
    )


def test_macro_run_loop_does_not_leak_objects() -> None:
    """Сотни прогонов одного макроса не копят объекты и почти не двигают RSS."""
    command = _leak_macro()
    with MacroEngine(_NullRegistry()) as engine:
        for _ in range(MACRO_WARMUP):
            assert engine.run(command).ok

        objects_before = _objects()
        rss_before = _rss()

        for _ in range(MACRO_ITERS):
            assert engine.run(command).ok

        objects_after = _objects()
        rss_after = _rss()

    assert objects_after - objects_before < OBJECT_SLACK
    assert rss_after - rss_before < RSS_SLACK_BYTES


def test_event_bus_blast_delivers_all_without_subscriber_growth() -> None:
    """Тысяча событий доходит до подписчика целиком, число подписок не растёт."""
    bus = EventBus(thread_id=None)
    received: list[WorkerRestarted] = []
    # weak=False обязателен: слабая ссылка на список-приёмник умерла бы сразу,
    # и обработчик молча выпал бы — вышла бы ложная «нулевая потеря».
    bus.subscribe(WorkerRestarted, received.append, weak=False)

    subscribers_before = bus.subscriber_count()
    for _ in range(MACRO_WARMUP):
        bus.publish(WorkerRestarted(worker="warmup"))
    received.clear()

    objects_before = _objects()
    for _ in range(EVENT_ITERS):
        bus.publish(WorkerRestarted(worker="blast"))
    objects_after = _objects()

    assert len(received) == EVENT_ITERS
    assert bus.subscriber_count() == subscribers_before
    assert bus.delivered >= EVENT_ITERS
    assert bus.failed == 0
    # События заморожены и приёмник их держит списком, так что рост объектов
    # примерно равен числу событий; течь была бы кратно больше.
    assert objects_after - objects_before < EVENT_ITERS + OBJECT_SLACK


@pytest.mark.skipif(sys.platform != "win32", reason="num_handles есть только на Windows")
def test_worker_restart_loop_leaks_no_handles_or_processes() -> None:
    """Серия перезапусков воркера не копит ни хендлы, ни процессы, ни нити."""
    spec = WorkerSpec(
        name="echo",
        kind="test",
        entrypoint=ECHO,
        python_path=(str(FIXTURES),),
        start_timeout=60.0,
        call_timeout=20.0,
        stop_timeout=5.0,
        heartbeat_interval=0.5,
        heartbeat_misses=6,
        restart_delay=0.05,
        max_restart_delay=0.2,
    )
    bus = EventBus(thread_id=None)
    manager = WorkerManager(bus)
    try:
        manager.register(spec)
        manager.start("echo")
        for _ in range(RESTART_WARMUP):
            manager.restart("echo")
        assert manager.is_ready("echo")

        children_before = len(worker_children())
        threads_before = threading.active_count()
        handles_before = _handles()

        for _ in range(RESTART_ITERS):
            manager.restart("echo")
        assert manager.is_ready("echo")

        assert len(worker_children()) == children_before
        assert threading.active_count() <= threads_before + 1
        assert _handles() - handles_before < HANDLE_SLACK
    finally:
        manager.shutdown()

    assert wait_for(lambda: not worker_children())
