"""Шаг мастера «Языковая модель» (llama.cpp): выбор, пропуск, запись конфига.

Offscreen, без сети и без нативного колеса llama.cpp: бэкенд менеджера моделей
подменён фейком, события загрузки дёргаются напрямую. Проверяем контракт шага
(необязательный, пропускаемый) и то, что выбор пишет провайдера ``llamacpp``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.core.config import ConfigManager
from ayris.core.events import ModelDownloadFinished
from ayris.core.models import ModelRecord
from ayris.gui.theme import ThemeManager
from ayris.onboarding.steps.llm import LlmStep, downloadable_specs

pytestmark = pytest.mark.unit


class _FakeBackend:
    """Записывает вызовы загрузки вместо похода в реестр и сеть."""

    def __init__(self, installed: Sequence[ModelRecord] = ()) -> None:
        self._installed = list(installed)
        self.started: list[str] = []

    def installed(self) -> list[ModelRecord]:
        return list(self._installed)

    def start_download(self, model_id: str) -> None:
        self.started.append(model_id)


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def config(tmp_path: Path) -> ConfigManager:
    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    return manager


def _step(app: QApplication, config: ConfigManager, backend: _FakeBackend) -> LlmStep:
    return LlmStep(ThemeManager(app), config, backend, None, ram_mb=16384)


def test_lists_only_downloadable_specs(app: QApplication, config: ConfigManager) -> None:
    step = _step(app, config, _FakeBackend())
    assert step._radios  # хотя бы одна модель с GGUF
    assert set(step._radios) == {spec.gguf_catalog_id for spec in downloadable_specs()}
    step.teardown()
    step.close()


def test_step_is_skippable_and_later_is_default(app: QApplication, config: ConfigManager) -> None:
    step = _step(app, config, _FakeBackend())
    assert step.can_skip() is True
    assert step._later.isChecked()
    assert step._selected is None
    assert not step._download_button.isEnabled()
    step.teardown()
    step.close()


def test_later_apply_leaves_provider_untouched(app: QApplication, config: ConfigManager) -> None:
    config.apply({"ai.provider": "openai"})
    step = _step(app, config, _FakeBackend())
    step.apply()  # выбран «Позже» — ничего не пишем
    assert config.settings.ai.provider == "openai"
    step.teardown()
    step.close()


def test_selecting_a_model_and_apply_switches_to_llamacpp(
    app: QApplication, config: ConfigManager
) -> None:
    step = _step(app, config, _FakeBackend())
    catalog_id = next(iter(step._radios))
    radio, spec = step._radios[catalog_id]
    radio.setChecked(True)
    assert step._selected is spec
    assert step._download_button.isEnabled()
    step.apply()
    ai = config.settings.ai
    assert ai.provider == "llamacpp"
    assert ai.model == spec.name
    step.teardown()
    step.close()


def test_start_download_calls_backend(app: QApplication, config: ConfigManager) -> None:
    backend = _FakeBackend()
    step = _step(app, config, backend)
    catalog_id = next(iter(step._radios))
    step._radios[catalog_id][0].setChecked(True)
    step._start_download()
    assert backend.started == [catalog_id]
    assert step._downloading_id == catalog_id
    step.teardown()
    step.close()


def test_finished_event_writes_model_path(app: QApplication, config: ConfigManager) -> None:
    step = _step(app, config, _FakeBackend())
    catalog_id = next(iter(step._radios))
    _radio, spec = step._radios[catalog_id]
    step._selected = spec
    step._downloading_id = catalog_id
    step._on_finished(
        ModelDownloadFinished(model_id=catalog_id, kind="llm", path="C:/models/y.gguf")
    )
    ai = config.settings.ai
    assert ai.provider == "llamacpp"
    assert ai.model == spec.name
    assert ai.model_path == "C:/models/y.gguf"
    assert step._downloading_id == ""
    step.teardown()
    step.close()
