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
from ayris.nlu.llm.catalog import RECOMMENDED, LlmModelSpec
from ayris.nlu.llm.local_models import LocalModel
from ayris.nlu.llm.ollama_client import OllamaPullProgress

pytestmark = pytest.mark.unit


def _gguf_spec() -> LlmModelSpec:
    """Первая рекомендованная модель со встроенным GGUF для llama.cpp."""
    for spec in RECOMMENDED:
        if spec.gguf_catalog_id and spec.runs_on("llamacpp"):
            return spec
    raise AssertionError("в каталоге нет модели с GGUF для llama.cpp")


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


def _no_local_models(_current_path: str) -> tuple[LocalModel, ...]:
    """Пустой диск по умолчанию: тест не должен зависеть от профиля машины."""
    return ()


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
        "local_models": _no_local_models,
    }
    defaults.update(overrides)
    return AiServices(**defaults)


def _make_tab(manager: ConfigManager, theme: ThemeManager, **overrides: object) -> AiTab:
    tab = AiTab(manager, theme, None, services=_services(**overrides))
    tab.load_from_config()
    return tab


def _cloud_tab(manager: ConfigManager, theme: ThemeManager, **overrides: object) -> AiTab:
    """Вкладка на облачном поставщике: у него список моделей приходит по сети.

    По умолчанию в конфиге стоит llama.cpp, а у встроенного движка список — это
    файлы на диске, и сетевых веток там просто нет.
    """
    manager.apply({"ai.provider": "openai"})
    return _make_tab(manager, theme, **overrides)


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
    assert tab._provider_combo.currentData() == "llamacpp"
    assert tab._model_picker.current_model() == "qwen2.5:7b-instruct"
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

    # Default llama.cpp (in-process): catalogue and the .gguf picker, no host or key.
    assert not tab._catalog_block.isHidden()
    assert not tab._gguf_pick_button.isHidden()
    assert tab._host_card.isHidden()
    assert tab._key_block.isHidden()
    assert tab._probe_block.isHidden()

    # Ollama: local host and catalogue, no cloud key, and no .gguf picker.
    combo.setCurrentIndex(combo.findData("ollama"))
    assert not tab._host_card.isHidden()
    assert not tab._catalog_block.isHidden()
    assert tab._gguf_pick_button.isHidden()
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


def test_probe_warms_the_model_picker(app: QApplication, manager: ConfigManager) -> None:
    tab = _cloud_tab(manager, ThemeManager(app))
    tab._on_probe_done(CredentialCheck(ok=True, detail="Ключ рабочий.", models=("m-a", "m-b")))
    assert tab._probe_status.text() == "Ключ рабочий."
    # Список пикера подтянулся с сервера, текущая модель осталась приколотой.
    assert "m-a" in tab._model_picker.models()
    assert "m-b" in tab._model_picker.models()
    assert tab._model_picker.current_model() == "qwen2.5:7b-instruct"
    tab.dispose()
    tab.close()


def test_choosing_a_model_writes_config_and_history(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    provider = manager.settings.ai.provider
    tab._on_model_chosen("some-model:latest")
    assert manager.settings.ai.model == "some-model:latest"
    assert manager.settings.ai.model_history[provider] == ["some-model:latest"]
    assert tab._model_picker.current_model() == "some-model:latest"
    tab.dispose()
    tab.close()


def test_add_new_reveals_input_and_applies_typed_name(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    assert tab._new_model_row.isHidden()

    tab._on_add_new_requested()
    assert not tab._new_model_row.isHidden()

    tab._new_model_edit.setText("gpt-4o")
    tab._commit_new_model()
    assert manager.settings.ai.model == "gpt-4o"
    assert tab._new_model_row.isHidden()
    assert tab._new_model_edit.text() == ""
    tab.dispose()
    tab.close()


def test_history_is_most_recent_first_and_deduped(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    provider = manager.settings.ai.provider
    tab._on_model_chosen("a")
    tab._on_model_chosen("b")
    tab._on_model_chosen("a")
    assert manager.settings.ai.model_history[provider] == ["a", "b"]
    tab.dispose()
    tab.close()


def test_opening_the_picker_fetches_and_populates(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _cloud_tab(manager, ThemeManager(app))
    # Прогоняем «фон» синхронно, чтобы не зависеть от живого потока.
    tab._models_runner.run = lambda work: tab._models_runner.finished.emit(work())
    tab._model_picker.list_requested.emit()
    # Фейковый клиент отдаёт ("gpt-4o-mini", "gpt-4o").
    assert "gpt-4o-mini" in tab._model_picker.models()
    assert not tab._models_loading
    # Повторное открытие берёт из кэша — новый прогон не запускается.
    tab._models_runner.run = lambda _work: pytest.fail("не должно ходить в сеть повторно")
    tab._model_picker.list_requested.emit()
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


def test_llamacpp_download_finish_writes_model_path_and_model(
    app: QApplication, manager: ConfigManager
) -> None:
    from ayris.core.events import ModelDownloadFinished

    tab = _make_tab(manager, ThemeManager(app))  # provider llamacpp по умолчанию
    spec = _gguf_spec()
    tab._pulling_spec = spec
    tab._pulling_catalog_id = spec.gguf_catalog_id
    tab._on_gguf_finished(
        ModelDownloadFinished(model_id=spec.gguf_catalog_id, kind="llm", path="C:/models/x.gguf")
    )
    ai = manager.settings.ai
    assert ai.model_path == "C:/models/x.gguf"
    assert ai.model == spec.name
    assert tab._pulling_catalog_id == ""
    assert f"«{spec.name}»" in tab._catalog_status.text()
    tab.dispose()
    tab.close()


def test_pick_local_gguf_writes_model_path(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._set_local_model("C:/weights/custom.gguf")
    assert manager.settings.ai.model_path == str(Path("C:/weights/custom.gguf"))
    assert "custom.gguf" in tab._model_path_label.text()
    # Имя модели идёт за файлом: иначе в отчётах осталась бы прежняя модель.
    assert manager.settings.ai.model == "custom"
    tab.dispose()
    tab.close()


# -- список моделей встроенного llama.cpp -----------------------------------


def _disk(*models: tuple[str, str]) -> Callable[[str], tuple[LocalModel, ...]]:
    """Подменить «что лежит на диске» парами «имя → путь»."""

    def scan(_current_path: str) -> tuple[LocalModel, ...]:
        return tuple(LocalModel(name, Path(path)) for name, path in models)

    return scan


def test_llamacpp_picker_lists_the_gguf_files_on_disk(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(
        manager,
        ThemeManager(app),
        local_models=_disk(("Vikhr-Gemma-2B", "D:/llm/Vikhr-Gemma-2B.gguf")),
    )
    assert "Vikhr-Gemma-2B" in tab._model_picker.models()
    tab.dispose()
    tab.close()


def test_llamacpp_picker_marks_the_entry_whose_file_is_configured(
    app: QApplication, manager: ConfigManager
) -> None:
    manager.apply({"ai.model_path": "D:/llm/vikhr.gguf", "ai.model": "совсем другое имя"})
    tab = _make_tab(
        manager,
        ThemeManager(app),
        local_models=_disk(("vikhr", "D:/llm/vikhr.gguf"), ("gemma", "D:/llm/gemma.gguf")),
    )
    # Отметка идёт по файлу, а не по записанному имени: движок работает с файлом.
    assert tab._model_picker.current_model() == "vikhr"
    tab.dispose()
    tab.close()


def test_choosing_a_local_model_switches_both_the_name_and_the_file(
    app: QApplication, manager: ConfigManager
) -> None:
    manager.apply({"ai.model_path": "D:/llm/vikhr.gguf", "ai.model": "vikhr"})
    tab = _make_tab(
        manager,
        ThemeManager(app),
        local_models=_disk(("vikhr", "D:/llm/vikhr.gguf"), ("gemma", "D:/llm/gemma.gguf")),
    )

    tab._on_model_chosen("gemma")

    ai = manager.settings.ai
    assert ai.model == "gemma"
    assert ai.model_path == str(Path("D:/llm/gemma.gguf"))
    # Обе правки под областью LLM — воркер перезапустится и выгрузит старую модель.
    assert RestartScope.LLM in manager.pending_restarts
    assert tab._model_picker.current_model() == "gemma"
    tab.dispose()
    tab.close()


def test_switching_back_restores_the_first_file(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(
        manager,
        ThemeManager(app),
        local_models=_disk(("vikhr", "D:/llm/vikhr.gguf"), ("gemma", "D:/llm/gemma.gguf")),
    )

    tab._on_model_chosen("gemma")
    tab._on_model_chosen("vikhr")

    assert manager.settings.ai.model_path == str(Path("D:/llm/vikhr.gguf"))
    assert manager.settings.ai.model == "vikhr"
    tab.dispose()
    tab.close()


def test_a_hand_typed_name_does_not_touch_the_model_file(
    app: QApplication, manager: ConfigManager
) -> None:
    manager.apply({"ai.model_path": "D:/llm/vikhr.gguf"})
    tab = _make_tab(manager, ThemeManager(app), local_models=_disk(("vikhr", "D:/llm/vikhr.gguf")))

    tab._new_model_edit.setText("какая-то-своя")
    tab._commit_new_model()

    assert manager.settings.ai.model == "какая-то-своя"
    assert manager.settings.ai.model_path == "D:/llm/vikhr.gguf"
    tab.dispose()
    tab.close()


def test_opening_the_llamacpp_picker_rescans_the_disk_without_network(
    app: QApplication, manager: ConfigManager
) -> None:
    seen: list[str] = []
    files: list[tuple[str, str]] = []

    def scan(current_path: str) -> tuple[LocalModel, ...]:
        seen.append(current_path)
        return tuple(LocalModel(name, Path(path)) for name, path in files)

    tab = _make_tab(manager, ThemeManager(app), local_models=scan)
    tab._models_runner.run = lambda _work: pytest.fail("встроенный движок в сеть не ходит")
    assert tab._model_picker.models() == ()

    files.append(("новая", "D:/llm/новая.gguf"))
    tab._model_picker.list_requested.emit()

    assert "новая" in tab._model_picker.models()
    assert len(seen) >= 2  # скан на загрузке вкладки и ещё раз на открытии списка
    tab.dispose()
    tab.close()


def test_a_failing_disk_scan_keeps_the_previous_list(
    app: QApplication, manager: ConfigManager
) -> None:
    broken = False

    def scan(_current_path: str) -> tuple[LocalModel, ...]:
        if broken:
            raise OSError("том отвалился")
        return (LocalModel("vikhr", Path("D:/llm/vikhr.gguf")),)

    tab = _make_tab(manager, ThemeManager(app), local_models=scan)
    assert "vikhr" in tab._model_picker.models()

    broken = True
    tab._model_picker.list_requested.emit()

    assert "vikhr" in tab._model_picker.models()
    tab.dispose()
    tab.close()


def test_gguf_download_without_backend_reports_unavailable(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tab._backend_tried = True  # менеджер моделей недоступен и уже пробовали
    tab._start_gguf_download(_gguf_spec())
    assert "Менеджер моделей недоступен" in tab._catalog_status.text()
    assert tab._pulling_catalog_id == ""
    tab.dispose()
    tab.close()
