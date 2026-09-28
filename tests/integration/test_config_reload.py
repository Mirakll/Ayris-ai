"""Горячая перезагрузка конфига → перезапуск воркеров по областям (стык подсистем).

Отдельные части этого шва уже покрыты: ``tests/unit/test_config.py`` проверяет,
что смена поля даёт правильный набор :class:`RestartScope`, а
``tests/unit/test_workers.py`` — что ``WorkerManager.restart`` действительно
пересоздаёт процесс. Но никто не проверяет их вместе: что вызов
``config.apply(...)`` у РЕАЛЬНОГО приложения доедет по шине до обработчиков
перезапуска и поднимет заново ровно те воркеры, чью область задело, не тронув
остальных, а область без обработчика останется «висеть» в ``pending_restarts``.

Поэтому здесь мы поднимаем настоящий :class:`AyrisApp` (в этом приложении стадии
WORKERS/NLU/GUI не имеют стартеров, так что реальные движки не порождаются),
навешиваем настоящий :class:`WorkerManager` с двумя echo-воркерами, помеченными
областями STT и TTS, и связываем их через продакшн-``_scope_handler``. Затем
двигаем конфиг обычным ``apply`` и смотрим на PID: сменился ли нужный воркер,
устоял ли соседний, квитировалась ли область.

Echo-воркер живёт в ``tests/fixtures/echo_worker.py`` (под ``spawn`` ребёнок
переимпортирует модуль с классом — воркер в самом тесте затащил бы pytest в
каждый процесс). Запускать набор как ``python -m pytest``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ayris.core.app import AppOptions, AyrisApp
from ayris.core.config import RestartScope
from ayris.core.events import EventBus
from ayris.workers.manager import WorkerManager, _scope_handler
from ayris.workers.registry import WorkerSpec

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ECHO = "echo_worker:EchoWorker"
#: Холодный интерпретатор с импортом Ayris заводится не мгновенно; порог должен
#: быть лишь короче точки, где зависание застопорит набор.
START_TIMEOUT = 60.0


def _spec(name: str, scope: RestartScope) -> WorkerSpec:
    """Спека echo-воркера из ``tests/fixtures`` с нужной областью перезапуска."""
    return WorkerSpec(
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
        restart_scope=scope,
    )


class _Rig:
    """Собранный стенд: приложение + менеджер с двумя размеченными воркерами."""

    def __init__(self, app: AyrisApp, manager: WorkerManager) -> None:
        self.app = app
        self.manager = manager

    def pid(self, name: str) -> int | None:
        """PID живого воркера ``name`` из снимка менеджера."""
        for summary in self.manager.status():
            if summary.name == name:
                return summary.pid
        raise AssertionError(f"нет воркера {name!r}")


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[_Rig]:
    """Настоящее приложение и менеджер; и то, и другое гасится даже при падении.

    Обработчик AUDIO намеренно НЕ регистрируется — на нём проверяется, что
    область без обработчика оседает в ``pending_restarts``.
    """
    options = AppOptions(
        profile=tmp_path / "profile",
        watch_config=False,
        single_instance=False,
        log_level="DEBUG",
    )
    app = AyrisApp(options).startup()
    manager = WorkerManager(EventBus(thread_id=None))
    try:
        manager.register(_spec("stt_echo", RestartScope.STT))
        manager.register(_spec("tts_echo", RestartScope.TTS))
        manager.start("stt_echo")
        manager.start("tts_echo")
        app.register_restart_handler(RestartScope.STT, _scope_handler(manager, RestartScope.STT))
        app.register_restart_handler(RestartScope.TTS, _scope_handler(manager, RestartScope.TTS))
        yield _Rig(app, manager)
    finally:
        manager.shutdown()
        app.shutdown()


def test_tts_change_restarts_only_tts_worker(rig: _Rig) -> None:
    """Смена TTS-поля поднимает заново TTS-воркер и не трогает STT-воркер."""
    stt_pid = rig.pid("stt_echo")
    tts_pid = rig.pid("tts_echo")

    diff = rig.app.config.apply({"voice.tts.expressiveness": 0.5})

    assert RestartScope.TTS in diff.restart_scopes
    assert rig.pid("tts_echo") != tts_pid  # пересоздан
    assert rig.pid("stt_echo") == stt_pid  # соседняя область цела
    assert rig.manager.is_ready("tts_echo")
    assert not rig.app.pending_restarts  # область квитирована


def test_stt_change_restarts_only_stt_worker(rig: _Rig) -> None:
    """Симметрично: смена STT-поля поднимает STT-воркер, TTS устаивает."""
    stt_pid = rig.pid("stt_echo")
    tts_pid = rig.pid("tts_echo")

    diff = rig.app.config.apply({"voice.stt.offline_model": "vosk-small"})

    assert RestartScope.STT in diff.restart_scopes
    assert rig.pid("stt_echo") != stt_pid
    assert rig.pid("tts_echo") == tts_pid
    assert rig.manager.is_ready("stt_echo")
    assert not rig.app.pending_restarts


def test_live_change_restarts_nothing(rig: _Rig) -> None:
    """«Живое» поле (скорость речи) применяется без перезапуска чего-либо."""
    stt_pid = rig.pid("stt_echo")
    tts_pid = rig.pid("tts_echo")

    diff = rig.app.config.apply({"voice.tts.speed": 1.2})

    assert diff  # изменение действительно применилось
    assert not (diff.restart_scopes - {RestartScope.NONE})  # перезапускать нечего
    assert rig.pid("stt_echo") == stt_pid
    assert rig.pid("tts_echo") == tts_pid
    assert not rig.app.pending_restarts


def test_unhandled_scope_stays_pending(rig: _Rig) -> None:
    """Область без обработчика (AUDIO) оседает в ``pending_restarts``."""
    diff = rig.app.config.apply({"voice.audio_input.device": "Микрофон (USB)"})

    assert RestartScope.AUDIO in diff.restart_scopes
    assert RestartScope.AUDIO in rig.app.pending_restarts  # ждёт перезапуска
    # Чужая область не задела уже поднятые воркеры.
    assert rig.manager.is_ready("stt_echo")
    assert rig.manager.is_ready("tts_echo")
