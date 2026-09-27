"""Задача 69: матрица отказов, слой сообщений и внесение сбоев в живой supervisor.

Три группы. :class:`TestMatrix` проверяет, что декларативная матрица отказов
самосогласована и покрывает все сценарии из спецификации — она чек-лист, а не
украшение. :class:`TestFailureNotifier` гоняет единый слой сообщений на подставной
шине с замороженными часами: одна формулировка на класс проблемы, повтор в
пределах cooldown гасится, восстановление сбрасывает ключ. Наконец
:class:`TestFaultInjection` (только Windows) убивает НАСТОЯЩИЙ процесс воркера и
проверяет контракт: подсистема поднимается, число процессов возвращается к
исходному, а «шторм» падений подряд не рождает лавину уведомлений и не оставляет
сирот.

Тесты внесения сбоев поднимают реальные процессы, поэтому у них тот же приём, что
у ``tests/unit/test_workers.py``: класс воркера живёт в ``tests/fixtures``, а
менеджер гасится в ``finally`` даже на упавшем тесте.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from ayris.core.errors import AyrisError
from ayris.core.events import (
    ActionFailed,
    Event,
    EventBus,
    MacroFailed,
    ModelDownloadFailed,
    NotificationRequested,
    OnlineStatusChanged,
    WorkerCrashed,
    WorkerRestarted,
)
from ayris.core.resilience import FAILURE_MATRIX, FailureNotifier, ProblemClass, scenario
from ayris.workers.manager import WorkerManager, WorkerStatus
from ayris.workers.registry import WorkerSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ECHO = "echo_worker:EchoWorker"
START_TIMEOUT = 60.0

#: Сценарии, которых спецификация Задачи 69 требует явно.
REQUIRED_KEYS = frozenset(
    {
        "worker_crash_audio",
        "worker_crash_stt",
        "worker_crash_tts",
        "worker_crash_llm",
        "internet_lost",
        "audio_device_lost",
        "model_missing",
        "model_corrupt",
        "disk_full",
        "action_timeout",
        "llm_stream_dropped",
    }
)


def echo_spec(name: str = "echo", **overrides: Any) -> WorkerSpec:
    """Спецификация тестового echo-воркера из ``tests/fixtures``."""
    base = WorkerSpec(
        name=name,
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
    return replace(base, **overrides)


def wait_for(predicate: Callable[[], object], timeout: float = 10.0) -> bool:
    """Опрашивать ``predicate`` до истины или до истечения времени."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def worker_children() -> list[multiprocessing.process.BaseProcess]:
    """Все живые процессы воркеров Ayris."""
    return [child for child in multiprocessing.active_children() if child.name.startswith("ayris-")]


def child_process(name: str = "echo") -> multiprocessing.process.BaseProcess | None:
    """Живой процесс воркера ``name``, как его видит multiprocessing."""
    for child in multiprocessing.active_children():
        if child.name == f"ayris-{name}":
            return child
    return None


def collect(bus: EventBus, event_type: type[Event]) -> list[Any]:
    """Подписаться жёстко и вернуть список, в который падают события."""
    received: list[Any] = []
    bus.subscribe(event_type, received.append, weak=False)
    return received


class Clock:
    """Впрыснутые монотонные часы: тест двигает их сам, без ожидания."""

    def __init__(self, value: float = 1000.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeSpec:
    def __init__(self, max_restarts: int) -> None:
        self.max_restarts = max_restarts


class _FakeManager:
    """Ровно столько от менеджера, сколько нужно notifier для терминального отказа."""

    def __init__(self, max_restarts: int = 1) -> None:
        self._max_restarts = max_restarts

    def spec(self, _name: str) -> _FakeSpec:
        return _FakeSpec(self._max_restarts)


@pytest.fixture
def bus() -> EventBus:
    """Шина, доставляющая встроенно на том потоке, что публикует."""
    return EventBus(thread_id=None)


class TestMatrix:
    """Матрица отказов как чек-лист: самосогласована и покрывает спецификацию."""

    def test_keys_are_unique(self) -> None:
        keys = [row.key for row in FAILURE_MATRIX]
        assert len(keys) == len(set(keys))

    def test_required_scenarios_are_covered(self) -> None:
        present = {row.key for row in FAILURE_MATRIX}
        assert present >= REQUIRED_KEYS

    def test_lookup_roundtrips_and_rejects_unknown(self) -> None:
        for row in FAILURE_MATRIX:
            assert scenario(row.key) is row
        with pytest.raises(KeyError):
            scenario("не такой ключ")

    def test_every_row_states_a_contract(self) -> None:
        for row in FAILURE_MATRIX:
            assert row.title and row.trigger and row.handled_by, row.key
            assert row.ui_state, row.key
            assert row.user_message, row.key
            assert "Traceback" not in row.user_message, row.key
            assert isinstance(row.problem_class, ProblemClass)

    def test_error_types_are_typed_ayris_errors(self) -> None:
        for row in FAILURE_MATRIX:
            if row.error_type is not None:
                assert issubclass(row.error_type, AyrisError), row.key

    def test_models_and_disk_do_not_claim_auto_recovery(self) -> None:
        for key in ("model_missing", "model_corrupt", "disk_full"):
            assert scenario(key).recovers is False, key

    def test_every_worker_crash_recovers(self) -> None:
        for key in (
            "worker_crash_audio",
            "worker_crash_stt",
            "worker_crash_tts",
            "worker_crash_llm",
        ):
            assert scenario(key).recovers is True, key


class TestFailureNotifier:
    """Единый слой сообщений: одна формулировка на класс, повторы гасятся."""

    def test_a_storm_of_one_worker_yields_one_message(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, clock=Clock()).install()
        for _ in range(10):
            bus.publish(WorkerCrashed(worker="audio", exit_code=1, error="boom", restarts=1))
        assert len(notes) == 1
        assert "звук" in notes[0].message

    def test_cooldown_lets_it_speak_again(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        clock = Clock()
        FailureNotifier(bus, clock=clock, cooldown=30.0).install()
        bus.publish(WorkerCrashed(worker="stt"))
        clock.advance(31.0)
        bus.publish(WorkerCrashed(worker="stt"))
        assert len(notes) == 2

    def test_distinct_problem_classes_each_speak(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, clock=Clock()).install()
        bus.publish(WorkerCrashed(worker="tts"))
        bus.publish(OnlineStatusChanged(online=False))
        bus.publish(ModelDownloadFailed(model_id="vosk-ru", error="sha"))
        bus.publish(ActionFailed(action="open", error="boom"))
        bus.publish(MacroFailed(run_id="r1", path="0", error="boom"))
        assert len(notes) == 5
        assert len({note.title for note in notes}) >= 4

    def test_terminal_crash_reports_disabled(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        manager = cast("WorkerManager", _FakeManager(max_restarts=1))
        FailureNotifier(bus, manager=manager, clock=Clock()).install()
        bus.publish(WorkerCrashed(worker="llm", restarts=2))
        assert len(notes) == 1
        assert notes[0].level == "error"
        assert "отключена" in notes[0].message

    def test_manual_restart_rearms_the_terminal_message(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        manager = cast("WorkerManager", _FakeManager(max_restarts=1))
        FailureNotifier(bus, manager=manager, clock=Clock()).install()
        bus.publish(WorkerCrashed(worker="llm", restarts=2))
        bus.publish(WorkerRestarted(worker="llm"))
        bus.publish(WorkerCrashed(worker="llm", restarts=2))
        assert len(notes) == 2

    def test_a_cancelled_download_is_silent(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, clock=Clock()).install()
        bus.publish(ModelDownloadFailed(model_id="x", cancelled=True))
        assert notes == []

    def test_reconnecting_resets_the_offline_message(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, clock=Clock()).install()
        bus.publish(OnlineStatusChanged(online=False))
        bus.publish(OnlineStatusChanged(online=True))
        bus.publish(OnlineStatusChanged(online=False))
        assert len(notes) == 3
        assert any("снова доступны" in note.message for note in notes)

    def test_close_stops_delivery(self, bus: EventBus) -> None:
        notes = collect(bus, NotificationRequested)
        notifier = FailureNotifier(bus, clock=Clock())
        notifier.install()
        bus.publish(WorkerCrashed(worker="audio"))
        notifier.close()
        bus.publish(WorkerCrashed(worker="stt"))
        assert len(notes) == 1


@pytest.mark.skipif(
    sys.platform != "win32", reason="убийство процесса воркера — только Windows-раннер"
)
class TestFaultInjection:
    """Намеренные сбои в живом supervisor: восстановление без сирот и лавины."""

    @pytest.fixture
    def manager(self, bus: EventBus) -> Iterator[WorkerManager]:
        """Менеджер, который гасится даже на упавшем тесте."""
        instance = WorkerManager(bus)
        try:
            yield instance
        finally:
            instance.shutdown()

    def test_a_killed_worker_recovers_and_is_reported(
        self, bus: EventBus, manager: WorkerManager
    ) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, manager=manager).install()
        manager.register(echo_spec())
        manager.start("echo")
        assert manager.is_ready("echo")
        baseline = len(worker_children())
        first_pid = manager.status()[0].pid

        process = child_process()
        assert process is not None
        process.kill()

        assert wait_for(
            lambda: manager.is_ready("echo") and manager.status()[0].pid not in (None, first_pid),
            timeout=30.0,
        ), "воркер не поднялся после убийства"
        assert wait_for(lambda: len(notes) >= 1), "об отказе не сообщили"
        assert notes[0].level == "warning"
        assert "перезапускается" in notes[0].message
        assert len(worker_children()) == baseline, "остался осиротевший процесс"

    def test_a_storm_of_crashes_settles_without_an_avalanche(
        self, bus: EventBus, manager: WorkerManager
    ) -> None:
        notes = collect(bus, NotificationRequested)
        FailureNotifier(bus, manager=manager, cooldown=30.0).install()
        manager.register(echo_spec(max_restarts=10))
        manager.start("echo")
        assert manager.is_ready("echo")
        baseline = len(worker_children())

        for _ in range(4):
            assert wait_for(lambda: child_process() is not None, timeout=30.0)
            process = child_process()
            assert process is not None
            process.kill()
            assert wait_for(
                lambda: manager.is_ready("echo") and child_process() is not None,
                timeout=30.0,
            ), "воркер не вернулся между ударами шторма"

        assert manager.worker_status("echo") is WorkerStatus.READY
        assert len(worker_children()) == baseline, "шторм расплодил процессы"
        # Сообщение приходит с потока-монитора менеджера — дождаться доставки,
        # прежде чем считать: синхронное чтение сразу после шторма гонится с ней.
        assert wait_for(lambda: len(notes) >= 1, timeout=30.0), "об отказе не сообщили"
        # Одна формулировка на класс: буря падений в пределах cooldown — одно сообщение.
        assert 1 <= len(notes) <= 2
