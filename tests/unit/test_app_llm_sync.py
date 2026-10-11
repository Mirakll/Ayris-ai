"""Синхронизация активной локальной модели в конфиг (пункт 8).

Менеджер моделей (вкладка «Обновления») — второе место, где выбирается локальная
модель. Когда активный GGUF меняется и движок — встроенный ``llamacpp``, выбор
зеркалится в ``ai.model_path`` (тег :class:`RestartScope.LLM`), чтобы воркер
перезапустился на новом файле. Проверяем чистый помощник без запуска приложения.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.core.app import _sync_llm_model_path
from ayris.core.config import ConfigManager
from ayris.core.events import ActiveModelChanged

pytestmark = pytest.mark.unit


@pytest.fixture
def config(tmp_path: Path) -> ConfigManager:
    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    return manager


def test_active_llm_model_writes_path_under_llamacpp(config: ConfigManager) -> None:
    config.apply({"ai.provider": "llamacpp"})
    _sync_llm_model_path(
        config,
        ActiveModelChanged(kind="llm", name="Qwen 2.5 3B", path="C:/models/q3.gguf", model_id=7),
    )
    ai = config.settings.ai
    assert ai.model_path == "C:/models/q3.gguf"
    assert ai.model == "Qwen 2.5 3B"


def test_cloud_provider_is_not_clobbered(config: ConfigManager) -> None:
    config.apply({"ai.provider": "openai", "ai.model": "gpt-4o-mini"})
    _sync_llm_model_path(
        config,
        ActiveModelChanged(kind="llm", name="Qwen", path="C:/models/q.gguf", model_id=1),
    )
    ai = config.settings.ai
    assert ai.model == "gpt-4o-mini"
    assert ai.model_path == ""


def test_a_cleared_slot_writes_nothing(config: ConfigManager) -> None:
    config.apply({"ai.provider": "llamacpp", "ai.model_path": "C:/models/keep.gguf"})
    _sync_llm_model_path(config, ActiveModelChanged(kind="llm", name="", path="", model_id=None))
    assert config.settings.ai.model_path == "C:/models/keep.gguf"


def test_a_non_llm_kind_is_ignored(config: ConfigManager) -> None:
    config.apply({"ai.provider": "llamacpp"})
    _sync_llm_model_path(
        config,
        ActiveModelChanged(kind="stt", name="Vosk", path="C:/models/vosk", model_id=2),
    )
    assert config.settings.ai.model_path == ""
