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

import time
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


def test_live_change_reaches_running_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """«Живое» поле доезжает до УЖЕ запущенного воркера через ``configure``.

    Регрессия на «сколько ни крути, ничего не меняется»: смена поля без области
    перезапуска (скорость речи) обязана дойти до работающего воркера БЕЗ
    пересоздания процесса — через подписку, которую вешает ``install_workers``:
    ``refresh_spec`` принимает свежую спеку, затем воркеру шлётся ``configure``.
    Это тот самый шов, которого раньше не было: воркер читал параметры только на
    старте. Планировщик подменён на echo-воркера, чтобы шов проверялся без
    реальных движков, а менять ему нужно одно поле ``speed``.
    """
    from dataclasses import replace as _replace

    from ayris.workers import registry
    from ayris.workers.manager import install_workers
    from ayris.workers.registry import PlannedWorker, WorkerPlan

    def fake_plan(settings, *, log_dir=None, include_unavailable=False):  # noqa: ANN202
        """Один echo-воркер, чья единственная настройка — текущая скорость речи."""
        spec = _replace(
            _spec("tts_echo", RestartScope.NONE),
            params={"speed": settings.voice.tts.speed},
        )
        return WorkerPlan(workers=(PlannedWorker(spec=spec, needed=True, autostart=True),))

    monkeypatch.setattr(registry, "plan_workers", fake_plan)

    options = AppOptions(
        profile=tmp_path / "profile",
        watch_config=False,
        single_instance=False,
        log_level="DEBUG",
    )
    app = AyrisApp(options).startup()
    manager = install_workers(app)  # сам поднимет echo-воркера на фоне
    try:
        deadline = time.monotonic() + START_TIMEOUT
        while not manager.is_ready("tts_echo"):
            if time.monotonic() > deadline:
                raise AssertionError("echo-воркер не поднялся за отведённое время")
            time.sleep(0.05)

        pid = next(s.pid for s in manager.status() if s.name == "tts_echo")
        before = manager.call_sync("tts_echo", "inherited", timeout=20.0)
        assert before["params"].get("speed") != 1.2  # есть куда меняться

        diff = app.config.apply({"voice.tts.speed": 1.2})
        assert diff  # поле действительно сменилось
        assert not (diff.restart_scopes - {RestartScope.NONE})  # и это «живое» поле

        after = manager.call_sync("tts_echo", "inherited", timeout=20.0)
        assert after["params"].get("speed") == 1.2  # настройка доехала на лету
        same_pid = next(s.pid for s in manager.status() if s.name == "tts_echo")
        assert same_pid == pid  # и БЕЗ перезапуска процесса
    finally:
        manager.shutdown()
        app.shutdown()
