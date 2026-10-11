"""Переключение профиля → голосовые триггеры перепривязываются (стык подсистем).

``tests/unit/test_profile.py`` проверяет, что ``ProfileManager.switch`` меняет
активную запись в БД и шлёт :class:`ProfileSwitched`, а ``tests/unit/test_matcher``
— что матчер находит команду по фразе. Никто не проверяет их вместе: что живой
пайплайн, собранный ``install_pipeline`` (это и есть шов переключения — его
обработчик ``on_profile`` пересобирает индекс триггеров), после смены профиля
начинает узнавать команды НОВОГО профиля и перестаёт узнавать команды старого.

Поэтому здесь мы поднимаем настоящий :class:`AyrisApp` (без воркеров и без
движков — текстовый путь ``run_text`` не требует ни микрофона, ни STT), заводим
в БД два профиля с разными голосовыми командами, подключаем пайплайн и убеждаемся,
что до переключения матчится команда профиля A, после ``ProfileManager.switch`` —
команда профиля B, а команда A становится неузнаваемой. Именно так «команды и
триггеры перепривязываются» на живом стыке.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ayris.audio.tts.app_router import set_active_tts_router
from ayris.core.app import AppOptions, AyrisApp
from ayris.core.models import Command
from ayris.core.pipeline_app import install_pipeline
from ayris.core.profile import ProfileManager

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ayris.core.pipeline import Pipeline

pytestmark = pytest.mark.integration

PHRASE_A = "привет альфа"
PHRASE_B = "привет бета"


class _Rig:
    """Приложение с двумя профилями и подключённым текстовым пайплайном."""

    def __init__(self, app: AyrisApp, pipeline: Pipeline, cmd_a: int, cmd_b: int) -> None:
        self.app = app
        self.pipeline = pipeline
        self.cmd_a = cmd_a
        self.cmd_b = cmd_b


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[_Rig]:
    """Настоящее приложение; профиль A активен, профиль B ждёт в БД.

    Глобальный TTS-роутер сбрасывается: ``install_pipeline`` берёт его из
    общего слота, и роутер, протёкший из другого теста, здесь не нужен.
    """
    set_active_tts_router(None)
    options = AppOptions(
        profile=tmp_path / "profile",
        watch_config=False,
        single_instance=False,
        log_level="DEBUG",
    )
    app = AyrisApp(options).startup()
    try:
        repos = app.repositories
        profile_a = app.profile
        assert profile_a.id is not None
        profile_b = repos.profiles.create("Профиль Б")
        assert profile_b.id is not None

        cmd_a = repos.commands.create(Command(name="Альфа", profile_id=profile_a.id))
        repos.triggers.add_voice(cmd_a.id, PHRASE_A)
        cmd_b = repos.commands.create(Command(name="Бета", profile_id=profile_b.id))
        repos.triggers.add_voice(cmd_b.id, PHRASE_B)

        # Индекс строится из активного профиля A при подключении.
        pipeline = install_pipeline(app)
        yield _Rig(app, pipeline, cmd_a.id, cmd_b.id)
    finally:
        app.shutdown()
        set_active_tts_router(None)


def test_switch_rebinds_voice_triggers(rig: _Rig) -> None:
    """До переключения узнаётся команда A; после — команда B, а A забывается.

    Признак привязки — ``command_id``: в hybrid-режиме по умолчанию непопадание
    уходит в LLM-фолбэк (не настроен → без команды), поэтому чужая фраза даёт
    ``command_id is None``, а своя — id ровно нужной команды.
    """
    # Профиль A активен: его фраза даёт команду A, чужая — ни одной.
    before_a = rig.pipeline.run_text(PHRASE_A)
    assert before_a.matched
    assert before_a.command_id == rig.cmd_a
    assert rig.pipeline.run_text(PHRASE_B).command_id is None

    # Переключаемся на профиль B настоящим менеджером — он шлёт ProfileSwitched,
    # обработчик пайплайна пересобирает индекс на лету (шина инлайновая в тесте).
    repos = rig.app.repositories
    manager = ProfileManager(repos, paths=rig.app.paths, bus=rig.app.bus)
    target = next(p for p in repos.profiles.list_all() if p.name == "Профиль Б")
    manager.switch(target)

    # Теперь фраза B даёт команду B, а фраза A больше не даёт ни одной команды.
    after_b = rig.pipeline.run_text(PHRASE_B)
    assert after_b.matched
    assert after_b.command_id == rig.cmd_b
    assert rig.pipeline.run_text(PHRASE_A).command_id is None
