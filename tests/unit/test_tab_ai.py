"""Вкладка «ИИ / LLM» (задача 64): связывание, режимы, провайдеры, проба.

Offscreen, всё замокано: ни один тест не ходит в сеть и не поднимает живых
потоков. Клиенты провайдеров подменяются фейками, сигналы стрима и загрузки
дёргаются напрямую, а ключ проверяется на то, что остаётся только в хранилище и
не протекает в конфиг. Каждый виджет закрывается в конце — незакрытая страница
держит подписку на конфиг и подвешивает CI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.core.config import ConfigManager, RestartScope, dump_settings
from ayris.core.pipeline import NluMode, mode_from_config
from ayris.core.secrets import SecretsStore
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.ai import AiServices, AiTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import set_active_worker_control
from ayris.gui.widgets.llm_test_panel import TestRequest as LlmTestRequest
from ayris.nlu.llm.base import (
    CredentialCheck,
    FinishReason,
    LlmClient,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmResponse,
    LlmRole,
    LlmTextDelta,
    LlmTool,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.nlu.llm.ollama_client import OllamaPullProgress

pytestmark = pytest.mark.unit


class _FakeKeyring:
    """An in-memory credential store, so no test touches Windows."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.entries.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.entries[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.entries.pop((service, username), None)


class _FakeControl:
    """Supervisor stand-in that records restarts instead of performing them."""

    def __init__(self) -> None:
        self.restarted: list[tuple[RestartScope, str]] = []

    def restart_scope(self, scope: RestartScope, settings_reason: str = "") -> int:
        self.restarted.append((scope, settings_reason))
        return 1


class _FakeStreamClient(LlmClient):
    """A configured client that streams a fixed answer without any network."""

    name = "openai"
    supports_streaming = True

    def __init__(
        self,
        *,
        chunks: tuple[str, ...] = ("Прив", "ет"),
        prompt_tokens: int = 5,
        completion_tokens: int = 3,
        models: tuple[str, ...] = ("gpt-4o-mini", "gpt-4o"),
    ) -> None:
        self._chunks = chunks
        self._usage = LlmUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        self._models = models
        self.closed = False

    @property
    def configured(self) -> bool:
        return True

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        return LlmResponse(text="".join(self._chunks), usage=self._usage)

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        for chunk in self._chunks:
            yield LlmTextDelta(chunk)
        yield LlmUsageDelta(self._usage)
        yield LlmDoneDelta(FinishReason.STOP)

    def list_models(self) -> tuple[str, ...]:
        return self._models

    def check_credentials(self) -> CredentialCheck:
        return CredentialCheck(ok=True, detail="Ключ рабочий.", models=self._models)

    def close(self) -> None:
        self.closed = True


def _fake_factory(provider: str, **kwargs: object) -> LlmClient:
    return _FakeStreamClient()


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


@pytest.fixture(autouse=True)
def _clear_control() -> Iterator[None]:
    yield
    set_active_worker_control(None)


def _services(**overrides: object) -> AiServices:
    defaults: dict[str, object] = {
        "secrets": SecretsStore(backend=_FakeKeyring()),
        "client_factory": _fake_factory,
        "total_ram_mb": 16384,
    }
    defaults.update(overrides)
    return AiServices(**defaults)


def _make_tab(manager: ConfigManager, theme: ThemeManager, **overrides: object) -> AiTab:
    tab = AiTab(manager, theme, None, services=_services(**overrides))
    tab.load_from_config()
    return tab


def test_tab_registers_itself_as_the_ai_factory() -> None:
    assert tab_spec("ai").factory is AiTab


def test_content_fits_without_a_horizontal_scrollbar(
    app: QApplication, manager: ConfigManager
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    tab = _make_tab(manager, ThemeManager(app))
    scroll = tab.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    content = scroll.widget()
    assert content is not None
    assert content.minimumSizeHint().width() <= 600
    tab.dispose()
    tab.close()


def test_tab_assembles_and_loads_defaults(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    assert tab._provider_combo.currentData() == "ollama"
    assert tab._model_edit.text() == "qwen2.5:7b-instruct"
    assert mode_from_config(manager.settings) is NluMode.HYBRID
    assert tab._mode_buttons[NluMode.HYBRID].isChecked()
    tab.dispose()
    tab.close()


def test_mode_toggles_collapse_or_reveal_the_model_section(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    # HYBRID by default: the language-model section is present.
    assert any(not widget.isHidden() for widget in tab._llm_widgets)

    tab._mode_buttons[NluMode.COMMANDS].setChecked(True)
    app.processEvents()
    assert mode_from_config(manager.settings) is NluMode.COMMANDS
    assert all(widget.isHidden() for widget in tab._llm_widgets)
    assert tab._mode_warn.isHidden()

    tab._mode_buttons[NluMode.AI].setChecked(True)
    app.processEvents()
    assert mode_from_config(manager.settings) is NluMode.AI
    assert any(not widget.isHidden() for widget in tab._llm_widgets)
    assert not tab._mode_warn.isHidden()
    tab.dispose()
    tab.close()


def test_generation_sliders_reach_config(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._temp_slider.setValue(150)  # 1.5
    tab._tokens_slider.setValue(2048)
    tab._timeout_slider.setValue(60)  # 60 s
    tab.flush_pending()
    ai = manager.settings.ai
    assert ai.temperature == pytest.approx(1.5)
    assert ai.max_tokens == 2048
    assert ai.request_timeout_sec == pytest.approx(60.0)
    tab.dispose()
    tab.close()


def test_history_sliders_reach_config(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._history_slider.setValue(20)
    tab._summary_slider.setValue(60)
    tab.flush_pending()
    ai = manager.settings.ai
    assert ai.history_turns == 20
    assert ai.summarize_after_turns == 60
    tab.dispose()
    tab.close()


def test_provider_switch_toggles_cloud_and_local_blocks(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    combo = tab._provider_combo

    # Default Ollama: local host and catalogue, no cloud key.
    assert not tab._host_card.isHidden()
    assert not tab._catalog_block.isHidden()
    assert tab._key_block.isHidden()
    assert tab._probe_button.text() == "Проверить соединение"

    combo.setCurrentIndex(combo.findData("openai"))
    assert not tab._key_block.isHidden()
    assert tab._catalog_block.isHidden()
    assert tab._host_card.isHidden()
    assert tab._probe_button.text() == "Проверить ключ"

    combo.setCurrentIndex(combo.findData("custom"))
    assert not tab._key_block.isHidden()  # custom is a cloud provider → key shown
    assert not tab._host_card.isHidden()  # custom keeps a base-URL field
    assert tab._catalog_block.isHidden()
    tab.dispose()
    tab.close()


def test_cloud_key_never_reaches_the_config(
    app: QApplication, manager: ConfigManager, tmp_path: Path
) -> None:
    keyring = _FakeKeyring()
    tab = _make_tab(manager, ThemeManager(app), secrets=SecretsStore(backend=keyring))
    key_field = tab._key_field

    key = "supersecretcloudkey1234567890"
    key_field.edit.setText(key)
    key_field._save()

    # Stored in the credential manager, and only there.
    assert keyring.entries[("Ayris", "openai")] == key
    assert key_field.edit.text() == ""
    assert manager.settings.ai.credential_ref == "openai"
    assert key not in str(dump_settings(manager.settings))
    assert key not in (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert key not in repr(tab.services.secrets)
    tab.dispose()
    tab.close()


def test_probe_populates_the_model_combo(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._on_probe_done(CredentialCheck(ok=True, detail="Ключ рабочий.", models=("m-a", "m-b")))
    assert tab._probe_status.text() == "Ключ рабочий."
    assert tab._model_combo.isEnabled()
    assert tab._model_combo.count() == 2
    assert tab._model_combo.itemData(0) == "m-a"
    tab.dispose()
    tab.close()


def test_pull_progress_updates_the_download(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._on_pull_progress(OllamaPullProgress("pulling manifest", completed=40, total=100))
    assert tab._catalog_status.text() == "pulling manifest"
    assert tab._download.updates_applied >= 1
    tab.dispose()
    tab.close()


def test_prompt_editor_edit_reaches_config_and_preview(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    editor = tab._chat_editor
    editor.editor.setPlainText("Будь краток.")
    tab.flush_pending()
    assert manager.settings.ai.chat_system_prompt == "Будь краток."
    assert "Будь краток." in editor.preview.toPlainText()
    tab.dispose()
    tab.close()


def test_build_test_request_uses_only_system_and_user(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    request = tab._build_test_request("который час")
    assert len(request.messages) == 2
    assert request.messages[0].role is LlmRole.SYSTEM
    assert request.messages[1].role is LlmRole.USER
    assert request.messages[1].content == "который час"
    assert request.provider == manager.settings.ai.provider
    request.client.close()
    tab.dispose()
    tab.close()


def test_test_panel_streams_answer_and_counts_tokens(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    client = _FakeStreamClient(chunks=("Прив", "ет"), prompt_tokens=5, completion_tokens=3)
    request = LlmTestRequest(
        client=client,
        messages=(LlmMessage.system("s"), LlmMessage.user("привет")),
        provider="openai",
        model="gpt-4o-mini",
    )
    tab._test_panel._runner._run(request)
    assert tab._test_panel.answer.toPlainText() == "Привет"
    assert "токенов" in tab._test_panel.stats.text()
    assert client.closed
    tab.dispose()
    tab.close()


def test_restart_plaque_appears_then_clears(app: QApplication, manager: ConfigManager) -> None:
    control = _FakeControl()
    set_active_worker_control(control)
    tab = _make_tab(manager, ThemeManager(app))
    bar = tab._restart_bars[RestartScope.LLM]
    assert bar.isHidden()

    manager.apply({"ai.provider": "openai"})
    app.processEvents()
    assert not bar.isHidden()
    assert bar._button.isEnabled()

    bar._button.click()
    assert control.restarted and control.restarted[0][0] is RestartScope.LLM
    assert RestartScope.LLM not in manager.pending_restarts
    assert bar.isHidden()
    tab.dispose()
    tab.close()


def test_clear_history_button_enabled_only_with_a_service(
    app: QApplication, manager: ConfigManager
) -> None:
    without = _make_tab(manager, ThemeManager(app))
    assert not without._clear_button.isEnabled()
    without.dispose()
    without.close()

    with_service = _make_tab(manager, ThemeManager(app), clear_history=lambda: None)
    assert with_service._clear_button.isEnabled()
    with_service.dispose()
    with_service.close()
