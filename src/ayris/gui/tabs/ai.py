"""Вкладка «ИИ / LLM»: режим распознавания, провайдеры, промпты, память, проба.

Задача 64. Вкладка строится на :class:`~ayris.gui.tabs.base.SettingsTab`, как и
«Голос», и повторяет её приёмы: отложенное двустороннее связывание с конфигом,
плашки перезапуска воркера и вынос сетевых вызовов из UI-потока. Своё здесь —
три взаимоисключающих режима (:func:`~ayris.core.pipeline.mode_from_config`),
которые сворачивают все настройки модели в режиме «только команды», и разговор с
провайдерами: проверка ключа/соединения, список моделей с сервера и загрузка
рекомендованной модели с прогрессом — всё в фоне, с отменой.

Ключ облачного сервиса, как и на «Голосе», не касается конфига: в ``config.toml``
живёт только ``credential_ref``, а сам ключ — в диспетчере учётных данных Windows
через :class:`~ayris.core.secrets.SecretsStore`. Проба, загрузка и тестовый
запрос строят клиента ТОЛЬКО после :meth:`flush_pending`, чтобы читать свежие
настройки, а не значения до дебаунса.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import AiConfig, ConfigManager, RestartScope
from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.errors import AyrisError, ConfigError, SecretsError
from ayris.core.events import (
    EventBus,
    ModelDownloadFailed,
    ModelDownloadFinished,
    ModelDownloadProgress,
)
from ayris.core.pipeline import NluMode, mode_from_config
from ayris.core.secrets import SecretsStore, get_secrets, is_valid_ref
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.tabs.voice import AsyncRunner, combo_options
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    BusyIndicator,
    ConfirmDialog,
    DownloadProgress,
    ModelPicker,
    SettingCard,
    SliderField,
    ThemedComboBox,
    ToggleSwitch,
    active_worker_control,
)
from ayris.gui.widgets.llm_test_panel import LlmTestPanel, TestRequest
from ayris.gui.widgets.prompt_editor import PromptEditor
from ayris.nlu.llm.base import CredentialCheck, LlmClient, LlmMessage
from ayris.nlu.llm.catalog import (
    RECOMMENDED,
    LlmModelSpec,
    detect_total_ram_mb,
    verdict_for,
    verdict_label,
)
from ayris.nlu.llm.factory import (
    HOST_PROVIDERS,
    LLAMACPP_PROVIDER,
    create_llm_client,
    is_cloud_provider,
    is_local_provider,
)
from ayris.nlu.llm.local_models import LocalModel, discover_local_models, local_model_dirs
from ayris.nlu.llm.modes import describe_mode
from ayris.nlu.llm.ollama_client import OllamaPullProgress
from ayris.nlu.llm.prompts import build_chat_prompt, build_nlu_prompt
from ayris.nlu.llm.tools import CommandCard, render_catalog
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.models import ModelRecord
    from ayris.gui.widgets.model_manager import ModelManagerBackend

__all__ = ["AiServices", "AiTab"]

_log = get_logger(__name__)

#: Подписи провайдеров для выпадающего списка. Значения — литералы ``AiConfig.provider``.
_PROVIDER_LABELS: dict[str, str] = {
    "ollama": "Ollama (локально)",
    "lmstudio": "LM Studio (локально)",
    "llamacpp": "llama.cpp (в процессе)",
    "openai": "OpenAI",
    "anthropic": "Anthropic · Claude",
    "openrouter": "OpenRouter",
    "deepseek": "DeepSeek",
    "gigachat": "GigaChat · Сбер",
    "yandex": "YandexGPT",
    "gemini": "Gemini · Google",
    "custom": "Свой OpenAI-совместимый",
}

#: Флаги ``AiConfig``, которыми режим кодируется в конфиге (зеркало ``mode_from_config``).
_MODE_FLAGS: dict[NluMode, dict[str, bool]] = {
    NluMode.COMMANDS: {
        "ai.fallback_to_llm": False,
        "ai.llm_understanding": False,
        "ai.free_chat": False,
    },
    NluMode.HYBRID: {
        "ai.fallback_to_llm": True,
        "ai.llm_understanding": False,
        "ai.free_chat": False,
    },
    NluMode.AI: {
        "ai.fallback_to_llm": False,
        "ai.llm_understanding": False,
        "ai.free_chat": True,
    },
}


def _no_cards() -> tuple[CommandCard, ...]:
    """Пустой каталог команд — дефолт, когда вкладке не дали доступ к библиотеке."""
    return ()


def _scan_local_models(current_path: str) -> tuple[LocalModel, ...]:
    """Файлы ``.gguf`` для пикера: папка профиля плюс папка уже выбранного файла.

    Отдельной функцией, а не методом, чтобы тест подменил её полем
    :attr:`AiServices.local_models` и не зависел ни от профиля, ни от диска.
    Недоступные пути — это пустой список, а не исключение во вкладке.
    """
    models_dir: Path | None = None
    try:
        from ayris.core.paths import get_paths

        models_dir = get_paths().llm_models_dir
    except Exception:
        _log.debug("папка моделей профиля недоступна", exc_info=True)
    return discover_local_models(local_model_dirs(models_dir=models_dir, current_path=current_path))


@dataclass(slots=True)
class AiServices:
    """Внешние зависимости вкладки: секреты, фабрика клиентов, команды, память, ОЗУ.

    Всё — с дефолтами, чтобы вкладка поднималась и без приложения (в тестах и
    предпросмотре). ``clear_history`` приходит от пайплайна: без него кнопка сброса
    истории заблокирована. ``total_ram_mb`` позволяет тестам не звать ``psutil``.
    ``backend`` нужен только пути llama.cpp: через него скачивается GGUF из каталога
    моделей. Строится лениво при первой загрузке; ``None`` — менеджер моделей
    недоступен, и загрузка из каталога выключена (свой файл выбрать всё равно можно).
    ``local_models`` отдаёт пикеру файлы ``.gguf`` с диска — это весь список моделей
    встроенного движка, спрашивать их не у кого.
    """

    secrets: SecretsStore = field(default_factory=get_secrets)
    client_factory: Callable[..., LlmClient] = create_llm_client
    command_cards: Callable[[], Sequence[CommandCard]] = _no_cards
    clear_history: Callable[[], None] | None = None
    total_ram_mb: int = 0
    backend: ModelManagerBackend | None = None
    local_models: Callable[[str], Sequence[LocalModel]] = _scan_local_models


def _default_backend(bus: EventBus | None) -> ModelManagerBackend | None:
    """Собрать бэкенд менеджера моделей для загрузки GGUF, или ``None``, если нельзя.

    Разделяет процессную базу и пути с остальным приложением, чтобы скачанная
    модель попала в тот же реестр, что видит вкладка «Обновления». В отличие от
    менеджера моделей, вкладке «ИИ» пустой каталог не нужен: при недоступной базе
    возвращаем ``None`` и просто гасим загрузку из каталога, не роняя вкладку.
    """
    try:
        from ayris.core.database import get_database
        from ayris.core.paths import get_paths
        from ayris.core.repositories import Repositories
        from ayris.gui.tabs.updates import RegistryBackend
        from ayris.models.registry import ModelRegistry

        repositories = Repositories(get_database())
        registry = ModelRegistry(repositories.models, get_paths(), bus=bus)
        return RegistryBackend(registry)
    except Exception:
        _log.exception("менеджер моделей недоступен — загрузка GGUF из каталога отключена")
        return None


class _DownloadRelay(QObject):
    """Переносит события загрузки модели с потока-загрузчика в поток вкладки.

    Прогресс публикуется из демон-потока координатора; трогать виджеты оттуда
    нельзя. Сигналы эмитятся из подписчика шины и доставляются очередью в поток,
    которому принадлежит реле, — тот же приём, что у менеджера моделей.
    """

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(object)


class _RestartBar(QFrame):
    """Плашка «нужен перезапуск воркера»: видима, только когда есть отложенная область.

    Копия приёма из вкладки «Голос» (там ``_RestartBar`` приватна и не
    экспортируется, поэтому дублируем, а не тянем через модуль). Кнопка активна
    лишь при живом управляющем воркером; иначе плашка честно сообщает, что
    перезапускать нечего.
    """

    restart_requested = Signal()

    def __init__(self, scope_hint: str, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self.setProperty("status", "warning")
        layout = QHBoxLayout(self)
        margin = theme.metric("spacing_md")
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(theme.metric("spacing_sm"))
        self._label = QLabel(f"Изменения вступят в силу после перезапуска: {scope_hint}.")
        self._label.setWordWrap(True)
        layout.addWidget(self._label, 1)
        self._button = QPushButton("Перезапустить")
        self._button.setProperty("kind", "primary")
        self._button.clicked.connect(self.restart_requested)
        layout.addWidget(self._button)
        self.hide()

    def set_pending(self, pending: bool, *, has_control: bool) -> None:
        """Показать плашку при отложенном перезапуске; кнопку — если есть кому его сделать."""
        self.setVisible(pending)
        self._button.setEnabled(has_control)
        if pending and not has_control:
            self._button.setToolTip("Перезапуск станет доступен после запуска воркеров.")
        else:
            self._button.setToolTip("")


class _KeyField(QWidget):
    """Поле ключа облачного сервиса поверх диспетчера учётных данных Windows.

    Как :class:`_SecretField` на «Голосе», но без локальной кнопки «Проверить»:
    сетевую проверку ключа делает блок пробы провайдера, а здесь — только запись,
    удаление и статус. Ключ никогда не показывается: после сохранения виджет
    сообщает лишь «ключ сохранён», а значение уходит в хранилище и стирается из поля.
    """

    def __init__(
        self,
        theme: ThemeManager,
        secrets: SecretsStore,
        ref_getter: Callable[[], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._secrets = secrets
        self._ref_getter = ref_getter
        self.setProperty("transparent", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))
        row = QHBoxLayout()
        row.setSpacing(theme.metric("spacing_sm"))
        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText("Вставьте ключ — он сохранится в хранилище Windows")
        self.edit.setAccessibleName("Ключ облачного сервиса")
        self.edit.returnPressed.connect(self._save)
        row.addWidget(self.edit, 1)
        self.save_button = QPushButton("Сохранить")
        self.save_button.setProperty("kind", "primary")
        self.save_button.clicked.connect(self._save)
        row.addWidget(self.save_button)
        self.delete_button = QPushButton("Удалить")
        self.delete_button.clicked.connect(self._delete)
        row.addWidget(self.delete_button)
        layout.addLayout(row)
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

    def refresh(self) -> None:
        """Свериться с хранилищем: доступно ли оно, корректна ли запись, есть ли ключ."""
        if not self._secrets.is_available():
            self.edit.setEnabled(False)
            self.save_button.setEnabled(False)
            self.delete_button.setEnabled(False)
            self.status.setText("Хранилище ключей Windows недоступно.")
            return
        ref = self._ref_getter().strip()
        self.edit.setEnabled(True)
        self.save_button.setEnabled(True)
        if not is_valid_ref(ref):
            self.delete_button.setEnabled(False)
            self.status.setText("Задайте имя записи ключа (латиница, например openai).")
            return
        stored = self._secrets.status(ref).stored
        self.delete_button.setEnabled(stored)
        if stored:
            self.status.setText(f"Запись «{ref}»: ключ сохранён.")
        else:
            self.status.setText(f"Запись «{ref}»: ключ не задан.")

    def _save(self) -> None:
        value = self.edit.text().strip()
        if not value:
            self.status.setText("Поле пустое — вставьте ключ и повторите.")
            return
        ref = self._ref_getter().strip()
        if not is_valid_ref(ref):
            self.status.setText("Сначала задайте корректное имя записи.")
            return
        try:
            self._secrets.save(ref, value)
        except SecretsError as exc:
            self.status.setText(exc.user_message)
            return
        self.edit.clear()
        self.refresh()

    def _delete(self) -> None:
        ref = self._ref_getter().strip()
        if not is_valid_ref(ref):
            return
        try:
            self._secrets.delete(ref)
        except SecretsError as exc:
            self.status.setText(exc.user_message)
            return
        self.edit.clear()
        self.refresh()


class _PullRunner(QObject):
    """Тянет модель Ollama вне UI-потока и шлёт прогресс/итог сигналами."""

    progress = Signal(object)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        """Взвести отмену — цикл загрузки опрашивает её между кусками."""
        self._cancel.set()

    def reset(self) -> None:
        """Свежее событие отмены перед новой загрузкой."""
        self._cancel = threading.Event()

    def run(self, client: LlmClient, model: str) -> None:
        """Запустить загрузку на демоне; прогресс придёт сигналами в поток вкладки."""
        threading.Thread(target=self._run, args=(client, model), daemon=True).start()

    def _run(self, client: LlmClient, model: str) -> None:
        pull = getattr(client, "pull", None)
        if pull is None:
            client.close()
            self.failed.emit("Этот провайдер не умеет загружать модели.")
            return
        try:
            for progress in pull(model):
                if self._cancel.is_set():
                    break
                self.progress.emit(progress)
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
            return
        except Exception as exc:  # клиент может кинуть что угодно — UI не должен упасть
            _log.exception("загрузка модели через Ollama упала")
            self.failed.emit(str(exc))
            return
        finally:
            client.close()
        if self._cancel.is_set():
            return
        self.finished.emit(model)


class _CatalogRow(QFrame):
    """Строка каталога: имя модели, размер, вердикт по памяти и кнопка загрузки."""

    download_requested = Signal(object)

    def __init__(
        self, spec: LlmModelSpec, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._spec = spec
        self.setProperty("transparent", True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))
        text = QVBoxLayout()
        text.setSpacing(theme.metric("spacing_xs"))
        self._title = QLabel(f"{spec.name}  ·  {spec.human_size}")
        self._title.setProperty("role", "secondary")
        self._title.setWordWrap(True)
        text.addWidget(self._title)
        self._verdict = QLabel("")
        self._verdict.setProperty("role", "muted")
        self._verdict.setWordWrap(True)
        text.addWidget(self._verdict)
        layout.addLayout(text, 1)
        self.button = QPushButton("Загрузить")
        self.button.clicked.connect(lambda: self.download_requested.emit(self._spec))
        layout.addWidget(self.button)

    @property
    def spec(self) -> LlmModelSpec:
        return self._spec

    def set_verdict(self, total_ram_mb: int) -> None:
        """Обновить строку вердикта: описание модели плюс оценка «влезет/не влезет»."""
        label = verdict_label(verdict_for(self._spec, total_ram_mb))
        desc = self._spec.description
        self._verdict.setText(f"{desc}  ·  {label}" if desc else label)

    def set_downloadable(self, downloadable: bool) -> None:
        """Показать кнопку загрузки, только когда провайдер умеет тянуть модель."""
        self.button.setVisible(downloadable)
        self.button.setEnabled(downloadable)


class _AllowedDirsEditor(QWidget):
    """Белый список папок для файловых действий (``actions.files.allowed_dirs``).

    Структурный список, как варианты слова активации: не через ``bind_*``, а
    прямой записью в конфиг. Пустой список = «файловым действиям доступ закрыт»,
    поэтому заполняет его только пользователь — нативным выбором папки.
    ``refresh()`` перечитывает конфиг, чтобы внешняя правка (импорт профиля,
    сброс) перестроила строки, а ``_last_written`` гасит перестроение в ответ на
    собственную запись.
    """

    _CONFIG_PATH = "actions.files.allowed_dirs"

    def __init__(
        self, manager: ConfigManager, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._theme = theme
        self._last_written: tuple[str, ...] = ()
        self.setProperty("transparent", True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))

        self._rows_container = QWidget()
        self._rows_container.setProperty("transparent", True)
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(theme.metric("spacing_xs"))
        layout.addWidget(self._rows_container)

        add_row = QHBoxLayout()
        add_row.setSpacing(theme.metric("spacing_sm"))
        self._add_button = QPushButton("Добавить папку…")
        self._add_button.setProperty("kind", "primary")
        self._add_button.clicked.connect(self._add_dir)
        add_row.addWidget(self._add_button)
        add_row.addStretch(1)
        layout.addLayout(add_row)

        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._rebuild_rows()

    def _current(self) -> list[str]:
        return list(self._manager.settings.actions.files.allowed_dirs)

    # <<<EDITOR>>>

    def _rebuild_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        dirs = self._current()
        if not dirs:
            empty = QLabel("Папки не добавлены — файловым действиям доступ закрыт.")
            empty.setProperty("role", "muted")
            empty.setWordWrap(True)
            self._rows_layout.addWidget(empty)
        for path in dirs:
            self._rows_layout.addWidget(self._make_row(path))
        self._last_written = tuple(dirs)

    def _make_row(self, path: str) -> QWidget:
        row = QWidget()
        row.setProperty("transparent", True)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(self._theme.metric("spacing_sm"))
        label = QLabel(path)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row_layout.addWidget(label, 1)
        remove = QPushButton("Удалить")
        remove.clicked.connect(lambda: self._remove_dir(path))
        row_layout.addWidget(remove)
        return row

    def _apply(self, dirs: list[str]) -> bool:
        try:
            self._manager.apply({self._CONFIG_PATH: dirs})
        except ConfigError as exc:
            self._status.setText(exc.user_message)
            return False
        self._status.setText("")
        self._rebuild_rows()
        return True

    def _add_dir(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Выберите папку, доступную файловым действиям", str(Path.home())
        )
        if not chosen:
            return
        resolved = str(Path(chosen))
        if resolved in self._current():
            self._status.setText("Эта папка уже в списке.")
            return
        self._apply([*self._current(), resolved])

    def _remove_dir(self, path: str) -> None:
        self._apply([item for item in self._current() if item != path])

    def refresh(self) -> None:
        if tuple(self._current()) != self._last_written:
            self._rebuild_rows()


class AiTab(SettingsTab):
    """Настройки языковой модели: режим, провайдеры, генерация, память, промпты, проба.

    На :class:`SettingsTab`, как «Голос»: связывание дебаунсится, сеть уходит в
    фон, а динамика (режим ↔ видимость секций, провайдер ↔ форма) пересобирается в
    :meth:`_refresh_dynamic`. Секции модели живут в ``_llm_widgets`` и целиком
    прячутся в режиме «только команды»; внутри провайдерской секции их доводит
    :meth:`_sync_provider`, читая свежий выбор из комбобокса, а не из конфига до дебаунса.
    """

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: AiServices | None = None,
    ) -> None:
        super().__init__("ai", "ИИ / LLM", ("ai",), manager, theme, bus)
        self.services = services if services is not None else AiServices()
        self.event_bus = bus
        self._restart_bars: dict[RestartScope, _RestartBar] = {}
        self._refreshers: list[Callable[[], None]] = []
        self._teardown: list[Callable[[], None]] = []
        self._llm_widgets: list[QWidget] = []
        self._catalog_rows: list[_CatalogRow] = []
        self._syncing_mode = False
        self._pulling_spec: LlmModelSpec | None = None
        self._pulling_catalog_id = ""
        self._backend_tried = False
        self._ram_mb = self.services.total_ram_mb or detect_total_ram_mb()
        self._command_cards = self.services.command_cards
        self._mode_buttons: dict[NluMode, QRadioButton] = {}

        self._probe_runner = AsyncRunner(self)
        self._probe_runner.finished.connect(self._on_probe_done)
        self._probe_runner.failed.connect(self._on_probe_failed)
        # Список моделей для пикера тянется в фоне при открытии выпадашки и
        # кэшируется по «поставщик|адрес|ключ»: пока сигнатура та же, повторное
        # открытие мгновенно и без сети.
        self._models_runner = AsyncRunner(self)
        self._models_runner.finished.connect(self._on_models_loaded)
        self._models_runner.failed.connect(self._on_models_failed)
        self._model_cache: dict[str, tuple[str, ...]] = {}
        self._models_loading = False
        self._models_signature = ""
        #: Для встроенного llama.cpp список моделей — это файлы на диске:
        #: «имя пункта → абсолютный путь». Пересобирается при каждом обновлении
        #: вкладки и при открытии выпадашки, чтобы свежескачанный файл появился
        #: без перезапуска.
        self._local_models: dict[str, str] = {}
        self._pull_runner = _PullRunner(self)
        self._pull_runner.progress.connect(self._on_pull_progress)
        self._pull_runner.finished.connect(self._on_pull_finished)
        self._pull_runner.failed.connect(self._on_pull_failed)
        # GGUF downloads (the llama.cpp path) run in the model registry's own
        # coordinator thread and report over the bus; a relay marshals those
        # events onto this thread, exactly as the model-manager widget does.
        self._dl_relay = _DownloadRelay(self)
        self._dl_relay.progress.connect(self._on_gguf_progress)
        self._dl_relay.finished.connect(self._on_gguf_finished)
        self._dl_relay.failed.connect(self._on_gguf_failed)
        if bus is not None:
            unsubscribers = [
                bus.subscribe(ModelDownloadProgress, self._dl_relay.progress.emit),
                bus.subscribe(ModelDownloadFinished, self._dl_relay.finished.emit),
                bus.subscribe(ModelDownloadFailed, self._dl_relay.failed.emit),
            ]

            def _drop_download_subs() -> None:
                for unsubscribe in unsubscribers:
                    unsubscribe()

            self.add_teardown(_drop_download_subs)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self.content = QVBoxLayout(container)
        self.content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._build_mode_section()
        self._build_provider_section()
        self._build_generation_section()
        self._build_history_section()
        self._build_agent_section()
        self._build_prompts_section()
        self._build_test_section()
        self.content.addStretch(1)

        self.register_refresh(self._sync_mode)
        self.register_refresh(self._sync_provider)
        self.register_refresh(self._sync_model_picker)
        self.register_refresh(self._chat_editor.refresh_preview)
        self.register_refresh(self._nlu_editor.refresh_preview)

    # -- section-facing helpers (зеркало «Голоса», AiTab строится на SettingsTab) --

    @property
    def manager(self) -> ConfigManager:
        """Менеджер конфига: секции применяют через него структурные значения."""
        return self._manager

    @property
    def theme(self) -> ThemeManager:
        """Активная тема — для виджетов, которые рисуют себя сами."""
        return self._theme

    def add_header(self, text: str) -> QLabel:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self.content.addWidget(header)
        return header

    def add_caption(self, text: str) -> QLabel:
        caption = QLabel(text)
        caption.setProperty("role", "muted")
        caption.setWordWrap(True)
        self.content.addWidget(caption)
        return caption

    def add_card(self, title: str, description: str, control: QWidget) -> SettingCard:
        if isinstance(control, QComboBox):
            self.tame_combo(control)
        card = SettingCard(title, description, control, self._theme)
        self.content.addWidget(card)
        return card

    def tame_combo(self, combo: QComboBox) -> None:
        """Не дать длинному пункту раздвинуть страницу: фикс. длина и эллипсис."""
        combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def add_widget(self, widget: QWidget) -> None:
        self.content.addWidget(widget)

    def panel(self) -> QWidget:
        """Прозрачный контейнер-только-под-layout, чтобы не заливать карточку тёмным."""
        container = QWidget()
        container.setProperty("transparent", True)
        return container

    def add_block(self, title: str, description: str, body: QWidget) -> QFrame:
        """Карточка во всю ширину: заголовок и описание — над телом, а не сбоку."""
        frame = QFrame()
        frame.setProperty("card", True)
        frame.setAccessibleName(title)
        pad = self._theme.metric("spacing_lg")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        if description:
            caption = QLabel(description)
            caption.setProperty("role", "secondary")
            caption.setWordWrap(True)
            layout.addWidget(caption)
        layout.addWidget(body)
        self.content.addWidget(frame)
        return frame

    def add_restart_bar(self, scope: RestartScope) -> _RestartBar:
        bar = _RestartBar(scope.label, self._theme)
        bar.restart_requested.connect(lambda: self._restart_scope(scope))
        self._restart_bars[scope] = bar
        self.content.addWidget(bar)
        return bar

    def bind_int_slider(self, field_widget: SliderField, path: str, label: str) -> None:
        self._bind(
            field_widget,
            path,
            label,
            field_widget.value,
            field_widget.setValue,
            field_widget.value_changed,
        )

    def bind_scaled_slider(
        self, field_widget: SliderField, path: str, label: str, *, factor: float
    ) -> None:
        """Связать целочисленный слайдер с дробным полем конфига, масштабируя туда-обратно."""
        self._bind(
            field_widget,
            path,
            label,
            lambda: round(field_widget.value() / factor, 3),
            lambda value: field_widget.setValue(round(float(value) * factor)),
            field_widget.value_changed,
        )

    def register_refresh(self, refresher: Callable[[], None]) -> None:
        """Звать ``refresher`` при внешнем изменении настроек и на загрузке."""
        self._refreshers.append(refresher)

    def add_teardown(self, callback: Callable[[], None]) -> None:
        """Зарегистрировать очистку (отписку, остановку таймера) на dispose."""
        self._teardown.append(callback)

    # -- обёртки «секция модели»: то, что прячется в режиме «только команды» ------

    def _model_header(self, text: str) -> QLabel:
        header = self.add_header(text)
        self._llm_widgets.append(header)
        return header

    def _model_caption(self, text: str) -> QLabel:
        caption = self.add_caption(text)
        self._llm_widgets.append(caption)
        return caption

    def _model_card(self, title: str, description: str, control: QWidget) -> SettingCard:
        card = self.add_card(title, description, control)
        self._llm_widgets.append(card)
        return card

    def _model_block(self, title: str, description: str, body: QWidget) -> QFrame:
        block = self.add_block(title, description, body)
        self._llm_widgets.append(block)
        return block

    # -- жизненный цикл -----------------------------------------------------

    def load_from_config(self) -> None:
        super().load_from_config()
        self._refresh_dynamic()

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        super()._on_config_changed(diff)
        self._refresh_dynamic()

    def _restart_scope(self, scope: RestartScope) -> None:
        control = active_worker_control()
        if control is None:
            return
        try:
            control.restart_scope(scope, "перезапуск из настроек")
        except Exception:
            _log.exception("не удалось перезапустить воркеры области %s", scope.value)
            return
        self._manager.acknowledge_restart(scope)
        self._refresh_dynamic()

    def _refresh_dynamic(self) -> None:
        pending = self._manager.pending_restarts
        has_control = active_worker_control() is not None
        for scope, bar in self._restart_bars.items():
            bar.set_pending(scope in pending, has_control=has_control)
        for refresher in self._refreshers:
            try:
                refresher()
            except Exception:
                _log.exception("сбой обновления секции вкладки «ИИ»")

    def dispose(self) -> None:
        self._pull_runner.cancel()
        for callback in self._teardown:
            try:
                callback()
            except Exception:
                _log.exception("сбой очистки вкладки «ИИ»")
        self._teardown.clear()
        super().dispose()

    # -- секция режима ------------------------------------------------------

    def _build_mode_section(self) -> None:
        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_md"))
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        for mode in (NluMode.COMMANDS, NluMode.HYBRID, NluMode.AI):
            info = describe_mode(mode)
            entry = self.panel()
            entry_layout = QVBoxLayout(entry)
            entry_layout.setContentsMargins(0, 0, 0, 0)
            entry_layout.setSpacing(self._theme.metric("spacing_xs"))
            radio = QRadioButton(info.label_ru)
            radio.setAccessibleName(f"Режим: {info.label_ru}")
            radio.toggled.connect(lambda checked, m=mode: self._on_mode_toggled(m, checked))
            self._mode_group.addButton(radio)
            self._mode_buttons[mode] = radio
            entry_layout.addWidget(radio)
            note = QLabel(info.note_ru)
            note.setProperty("role", "muted")
            note.setWordWrap(True)
            entry_layout.addWidget(note)
            layout.addWidget(entry)
        self._mode_warn = QLabel(describe_mode(NluMode.AI).warn_ru)
        self._mode_warn.setProperty("status", "warning")
        self._mode_warn.setWordWrap(True)
        self._mode_warn.hide()
        layout.addWidget(self._mode_warn)
        self.add_block(
            "Режим распознавания",
            "Как Айрис понимает фразы. От режима зависит, участвует ли языковая модель.",
            body,
        )

    def _on_mode_toggled(self, mode: NluMode, checked: bool) -> None:
        if not checked or self._syncing_mode:
            return
        self._manager.apply(dict(_MODE_FLAGS[mode]))

    def _sync_mode(self) -> None:
        mode = mode_from_config(self._manager.settings)
        radio = self._mode_buttons.get(mode)
        if radio is not None and not radio.isChecked():
            self._syncing_mode = True
            try:
                radio.setChecked(True)
            finally:
                self._syncing_mode = False
        self._mode_warn.setVisible(mode is NluMode.AI)
        for widget in self._llm_widgets:
            widget.setVisible(mode.uses_llm)

    # -- секция провайдера --------------------------------------------------

    def _build_provider_section(self) -> None:
        self._model_header("Языковая модель")

        self._provider_combo = ThemedComboBox()
        for value, label in combo_options(AiConfig, "provider", _PROVIDER_LABELS):
            self._provider_combo.addItem(label, value)
        self.bind_combo(self._provider_combo, "ai.provider", "Поставщик модели")
        self._provider_combo.currentIndexChanged.connect(lambda _i: self._sync_provider())
        self._model_card(
            "Поставщик",
            "Облачный сервис или локальный движок. Ключи облака хранятся в диспетчере Windows.",
            self._provider_combo,
        )

        self._host_edit = QLineEdit()
        self._host_edit.setPlaceholderText("http://127.0.0.1:11434")
        self.bind_line_edit(self._host_edit, "ai.host", "Адрес сервера моделей")
        self._host_card = self._model_card(
            "Адрес сервера",
            "Локальный сервер (Ollama, LM Studio) или базовый URL OpenAI-совместимого API.",
            self._host_edit,
        )

        self._model_picker = ModelPicker()
        self.tame_combo(self._model_picker)
        self._model_picker.list_requested.connect(self._request_model_list)
        self._model_picker.model_chosen.connect(self._on_model_chosen)
        self._model_picker.add_new_requested.connect(self._on_add_new_requested)

        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        layout.addWidget(self._model_picker)

        # Строка ручного ввода: скрыта, пока не выбрали «➕ Добавить свою модель…».
        self._new_model_row = self.panel()
        add_layout = QHBoxLayout(self._new_model_row)
        add_layout.setContentsMargins(0, 0, 0, 0)
        add_layout.setSpacing(self._theme.metric("spacing_sm"))
        self._new_model_edit = QLineEdit()
        self._new_model_edit.setPlaceholderText("например, gpt-4o или qwen2.5:7b-instruct")
        self._new_model_edit.returnPressed.connect(self._commit_new_model)
        add_layout.addWidget(self._new_model_edit, 1)
        self._new_model_add = QPushButton("Добавить")
        self._new_model_add.setProperty("kind", "primary")
        self._new_model_add.clicked.connect(self._commit_new_model)
        add_layout.addWidget(self._new_model_add)
        self._new_model_cancel = QPushButton("Отмена")
        self._new_model_cancel.clicked.connect(self._cancel_new_model)
        add_layout.addWidget(self._new_model_cancel)
        self._new_model_row.hide()
        layout.addWidget(self._new_model_row)

        self._model_block(
            "Модель",
            "Откройте список — покажутся установленные и недавние модели. "
            "«Добавить свою» — чтобы ввести имя облачной модели вручную.",
            body,
        )

        self._build_key_block()
        self._build_probe_block()
        self._build_catalog_block()
        self.add_restart_bar(RestartScope.LLM)

    def _build_key_block(self) -> None:
        self._key_ref_edit = QLineEdit()
        self._key_ref_edit.setPlaceholderText("openai")
        self.bind_line_edit(self._key_ref_edit, "ai.credential_ref", "Имя записи ключа")
        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        ref_row = QHBoxLayout()
        ref_row.setSpacing(self._theme.metric("spacing_sm"))
        ref_label = QLabel("Имя записи:")
        ref_label.setProperty("role", "secondary")
        ref_row.addWidget(ref_label)
        ref_row.addWidget(self._key_ref_edit, 1)
        layout.addLayout(ref_row)
        self._key_field = _KeyField(self._theme, self.services.secrets, self._key_ref_edit.text)
        self._key_ref_edit.textChanged.connect(lambda _t: self._key_field.refresh())
        layout.addWidget(self._key_field)
        self._key_block = self._model_block(
            "Ключ облачного сервиса",
            "Ключ хранится в диспетчере учётных данных Windows, а не в config.toml.",
            body,
        )

    def _build_probe_block(self) -> None:
        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        row = QHBoxLayout()
        row.setSpacing(self._theme.metric("spacing_sm"))
        self._probe_button = QPushButton("Проверить")
        self._probe_button.setProperty("kind", "primary")
        self._probe_button.clicked.connect(self._run_probe)
        row.addWidget(self._probe_button)
        self._probe_busy = BusyIndicator(self._theme, active=False)
        self._probe_busy.hide()
        row.addWidget(self._probe_busy)
        row.addStretch(1)
        layout.addLayout(row)
        self._probe_status = QLabel("")
        self._probe_status.setProperty("role", "muted")
        self._probe_status.setWordWrap(True)
        layout.addWidget(self._probe_status)
        self._probe_block = self._model_block(
            "Проверка",
            "Проверяет ключ или соединение и подтягивает список моделей. Сеть — в фоне.",
            body,
        )

    def _build_catalog_block(self) -> None:
        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        for index, spec in enumerate(RECOMMENDED):
            # Разделительная линия между моделями: без неё кнопка «Загрузить»
            # (её видит не каждая строка) читается как висящая между двумя
            # соседними моделями, а не принадлежащая своей. Высоту и цвет задаём
            # явно — на min/max-height из QSS в вертикальном лэйауте полагаться
            # нельзя, линия схлопывается в ноль.
            if index:
                layout.addWidget(self._catalog_divider())
            row = _CatalogRow(spec, self._theme)
            row.download_requested.connect(self._start_pull)
            layout.addWidget(row)
            self._catalog_rows.append(row)
        self._download = DownloadProgress(self._theme)
        self._download.cancel_requested.connect(self._cancel_pull)
        self._download.hide()
        layout.addWidget(self._download)
        self._catalog_status = QLabel("")
        self._catalog_status.setProperty("role", "muted")
        self._catalog_status.setWordWrap(True)
        layout.addWidget(self._catalog_status)
        # Свой файл модели (только llama.cpp): путь к .gguf минует каталог и реестр —
        # воркер грузит модель по абсолютному пути. Кнопка и строка пути видны лишь
        # под llama.cpp, их показывает _sync_provider.
        pick_row = QHBoxLayout()
        pick_row.setSpacing(self._theme.metric("spacing_sm"))
        self._gguf_pick_button = QPushButton("Выбрать файл .gguf…")
        self._gguf_pick_button.clicked.connect(self._pick_gguf)
        pick_row.addWidget(self._gguf_pick_button)
        pick_row.addStretch(1)
        layout.addLayout(pick_row)
        self._model_path_label = QLabel("")
        self._model_path_label.setProperty("role", "muted")
        self._model_path_label.setWordWrap(True)
        layout.addWidget(self._model_path_label)
        self._catalog_block = self._model_block(
            "Рекомендованные модели",
            "Оценка по вашей памяти. Ollama тянет модель по имени; "
            "для llama.cpp GGUF скачивается в менеджер моделей.",
            body,
        )

    def _catalog_divider(self) -> QFrame:
        """Тонкая горизонтальная линия цветом границы между строками каталога."""
        line = QFrame()
        line.setFixedHeight(max(1, self._theme.metric("border_width")))
        line.setStyleSheet(f"background-color: {self._theme.theme.color('border')}; border: none;")
        return line

    # -- выбор модели через пикер -------------------------------------------

    #: Сколько недавних моделей помнить на каждого поставщика.
    _MAX_HISTORY = 12

    def _provider_signature(self) -> str:
        """Ключ кэша списка моделей: у разных ключей/адресов свой список."""
        ai = self._manager.settings.ai
        return f"{ai.provider}|{ai.host}|{ai.credential_ref}"

    def _merge_history(self, provider: str, model: str) -> dict[str, list[str]]:
        """Новый словарь истории с ``model`` в начале списка ``provider``."""
        history = {k: list(v) for k, v in self._manager.settings.ai.model_history.items()}
        recent = [model, *(m for m in history.get(provider, ()) if m != model)]
        history[provider] = recent[: self._MAX_HISTORY]
        return history

    def _apply_model(self, model: str) -> None:
        """Записать выбранную модель в конфиг и запомнить её для поставщика.

        Для встроенного llama.cpp одного имени мало: движок открывает файл, а не
        имя, поэтому вместе с ``ai.model`` пишется и ``ai.model_path`` выбранного
        пункта — обе правки под :class:`RestartScope.LLM`, так что воркер
        перезапустится и старая модель выгрузится вместе с его процессом. Имя,
        которого нет среди найденных файлов (введено вручную), путь не трогает.
        """
        model = model.strip()
        if not model:
            return
        provider = self._manager.settings.ai.provider
        updates: dict[str, object] = {
            "ai.model": model,
            "ai.model_history": self._merge_history(provider, model),
        }
        path = self._local_models.get(model, "")
        if provider == LLAMACPP_PROVIDER and path:
            updates["ai.model_path"] = path
        self._manager.apply(updates)
        if provider == LLAMACPP_PROVIDER:
            self._refresh_model_path_label()

    def _on_model_chosen(self, model: str) -> None:
        self._cancel_new_model()
        self._apply_model(model)

    def _on_add_new_requested(self) -> None:
        self._new_model_row.show()
        self._new_model_edit.setFocus()

    def _commit_new_model(self) -> None:
        model = self._new_model_edit.text().strip()
        if not model:
            return
        self._new_model_edit.clear()
        self._new_model_row.hide()
        self._apply_model(model)

    def _cancel_new_model(self) -> None:
        self._new_model_edit.clear()
        self._new_model_row.hide()

    def _refresh_local_models(self) -> None:
        """Пересобрать карту «имя пункта → файл .gguf» для встроенного движка.

        Сбой чтения диска оставляет прежнюю карту, а не обнуляет её: пропавший на
        секунду сетевой том не должен стирать из списка модель, которой человек
        пользуется.
        """
        ai = self._manager.settings.ai
        if ai.provider != LLAMACPP_PROVIDER:
            self._local_models = {}
            return
        try:
            found = self.services.local_models(ai.model_path)
        except Exception:
            _log.exception("не удалось просмотреть папку с моделями")
            return
        self._local_models = {model.name: str(model.path) for model in found}

    def _current_local_name(self) -> str:
        """Какой пункт отметить активным: тот, чей файл совпал с ``ai.model_path``.

        В конфиге имя модели и имя файла расходятся (например «Qwen2.5 3B
        Instruct» против стема скачанного GGUF), а движок работает с файлом, —
        поэтому отметка идёт по пути, и только если такого файла в списке нет,
        показываем то имя, что записано в конфиге.
        """
        ai = self._manager.settings.ai
        if ai.model_path:
            wanted = Path(ai.model_path)
            for name, candidate in self._local_models.items():
                if Path(candidate) == wanted:
                    return name
        return ai.model

    def _sync_model_picker(self) -> None:
        """Свести список пикера из кэша сервера и истории, отметить текущую модель."""
        ai = self._manager.settings.ai
        if ai.provider == LLAMACPP_PROVIDER:
            # У встроенного движка список спрашивать не у кого: это файлы на
            # диске. История чужих поставщиков (теги Ollama, имена облачных
            # моделей) в нём только мешала бы — её не подставить в model_path.
            self._refresh_local_models()
            self._model_picker.set_models(list(self._local_models), loading=False)
            self._model_picker.set_current_model(self._current_local_name())
            return
        remembered = list(ai.model_history.get(ai.provider, ()))
        cached = list(self._model_cache.get(self._provider_signature(), ()))
        self._model_picker.set_models([*cached, *remembered], loading=self._models_loading)
        self._model_picker.set_current_model(ai.model)

    def _request_model_list(self) -> None:
        """Подтянуть модели с сервера при открытии списка — один раз на сигнатуру."""
        if not mode_from_config(self._manager.settings).uses_llm:
            return
        if self._manager.settings.ai.provider == LLAMACPP_PROVIDER:
            # Пересматриваем диск на каждое открытие: только что скачанный файл
            # должен появиться сразу, а стоит это один glob по двум папкам.
            self._sync_model_picker()
            self._model_picker.reload_open()
            return
        signature = self._provider_signature()
        if self._models_loading or signature in self._model_cache:
            return
        self.flush_pending()
        client = self._build_client()
        self._models_loading = True
        self._models_signature = signature
        self._model_picker.set_loading(True)

        def work() -> tuple[str, ...]:
            try:
                return tuple(client.list_models())
            finally:
                client.close()

        self._models_runner.run(work)

    def _on_models_loaded(self, result: object) -> None:
        self._models_loading = False
        models = tuple(result) if isinstance(result, tuple | list) else ()
        self._model_cache[self._models_signature] = models
        self._sync_model_picker()
        self._model_picker.reload_open()

    def _on_models_failed(self, _message: str) -> None:
        # Не кэшируем неудачу — повторное открытие попробует снова; текст ошибки
        # пользователь увидит в блоке «Проверка», список молча остаётся прежним.
        self._models_loading = False
        self._model_picker.set_loading(False)

    def _sync_provider(self) -> None:
        settings = self._manager.settings
        if not mode_from_config(settings).uses_llm:
            return
        provider = str(self._provider_combo.currentData() or settings.ai.provider)
        cloud = is_cloud_provider(provider)
        local = is_local_provider(provider)
        is_llamacpp = provider == LLAMACPP_PROVIDER
        self._host_card.setVisible(provider in HOST_PROVIDERS)
        self._key_block.setVisible(cloud)
        self._catalog_block.setVisible(local)
        self._probe_block.setVisible(cloud or (local and provider in HOST_PROVIDERS))
        self._probe_button.setText("Проверить ключ" if cloud else "Проверить соединение")
        for row in self._catalog_rows:
            row.set_verdict(self._ram_mb)
            row.set_downloadable(self._row_downloadable(row, provider))
        # Свой файл и его путь имеют смысл только для встроенного llama.cpp.
        self._gguf_pick_button.setVisible(is_llamacpp)
        self._model_path_label.setVisible(is_llamacpp)
        if is_llamacpp:
            self._refresh_model_path_label()
        self._key_field.refresh()

    def _row_downloadable(self, row: _CatalogRow, provider: str) -> bool:
        """Можно ли тянуть эту модель текущим провайдером.

        Ollama тянет любую по имени-тегу; llama.cpp — только те, для которых Айрис
        поставляет GGUF (``gguf_catalog_id``); у LM Studio свой загрузчик.
        """
        if provider == "ollama":
            return row.spec.runs_on("ollama")
        if provider == LLAMACPP_PROVIDER:
            return bool(row.spec.gguf_catalog_id) and row.spec.runs_on(LLAMACPP_PROVIDER)
        return False

    def _refresh_model_path_label(self) -> None:
        """Показать выбранный файл модели, пропажу файла или подсказку «не выбран»."""
        path = self._manager.settings.ai.model_path
        if not path:
            self._model_path_label.setText("Файл модели не выбран — скачайте или укажите .gguf.")
            return
        file = Path(path)
        try:
            exists = file.is_file()
        except OSError:  # недоступный том — честнее сказать «не найден», чем промолчать
            exists = False
        if exists:
            self._model_path_label.setText(f"Выбран файл: {file.name}")
        else:
            self._model_path_label.setText(f"Файл не найден: {path}")

    # -- проба соединения и списка моделей -----------------------------------

    def _build_client(self) -> LlmClient:
        ai = self._manager.settings.ai
        extra: dict[str, object] | None = None
        if ai.provider == LLAMACPP_PROVIDER:
            # Встроенный движок открывает файл, а не адрес: без `extra` тестовая
            # панель строила бы клиента без модели и всегда отвечала «не настроено».
            extra = {
                "model_path": ai.model_path,
                "n_ctx": ai.n_ctx,
                "n_gpu_layers": ai.n_gpu_layers,
            }
        return self.services.client_factory(
            ai.provider,
            model=ai.model,
            base_url=ai.host,
            credential_ref=ai.credential_ref,
            temperature=ai.temperature,
            max_tokens=ai.max_tokens,
            read_timeout_sec=ai.request_timeout_sec,
            store=self.services.secrets,
            extra=extra,
        )

    def _run_probe(self) -> None:
        self.flush_pending()
        provider = self._manager.settings.ai.provider
        cloud = is_cloud_provider(provider)
        client = self._build_client()
        self._probe_button.setEnabled(False)
        self._probe_busy.show()
        self._probe_busy.setActive(True)
        self._probe_status.setText("Проверяю…")

        def work() -> CredentialCheck:
            try:
                if cloud:
                    return client.check_credentials()
                return CredentialCheck(ok=True, models=tuple(client.list_models()))
            finally:
                client.close()

        self._probe_runner.run(work)

    def _finish_probe(self) -> None:
        self._probe_busy.setActive(False)
        self._probe_busy.hide()
        self._probe_button.setEnabled(True)

    def _on_probe_done(self, result: object) -> None:
        self._finish_probe()
        if not isinstance(result, CredentialCheck):
            return
        if result.ok:
            self._probe_status.setText(result.detail or "Проверка прошла успешно.")
        else:
            self._probe_status.setText(result.detail or "Проверка не прошла.")
        self._populate_models(result.models)

    def _on_probe_failed(self, message: str) -> None:
        self._finish_probe()
        self._probe_status.setText(f"Не удалось проверить: {message}")

    def _populate_models(self, models: Sequence[str]) -> None:
        """Проверка вернула список — прогреть кэш пикера, чтобы открылся мгновенно."""
        self._model_cache[self._provider_signature()] = tuple(models)
        self._sync_model_picker()

    # -- загрузка модели через Ollama ---------------------------------------

    def _start_pull(self, spec: object) -> None:
        if not isinstance(spec, LlmModelSpec) or self._pulling_spec is not None:
            return
        self.flush_pending()
        provider = self._manager.settings.ai.provider
        if provider == LLAMACPP_PROVIDER:
            self._start_gguf_download(spec)
            return
        if provider != "ollama":
            self._catalog_status.setText("Загрузка доступна только для Ollama и llama.cpp.")
            return
        client = self._build_client()
        self._pulling_spec = spec
        self._download.reset()
        self._download.set_cancellable(True)
        self._download.show()
        self._catalog_status.setText(f"Загружаю «{spec.name}»…")
        for row in self._catalog_rows:
            row.button.setEnabled(False)
        self._pull_runner.reset()
        self._pull_runner.run(client, spec.ollama_tag)

    def _cancel_pull(self) -> None:
        if self._pulling_catalog_id:
            backend = self.services.backend
            if backend is not None:
                backend.cancel_download(self._pulling_catalog_id)
            self._download.set_cancellable(False)
            self._catalog_status.setText("Отменяю загрузку…")
            return
        self._pull_runner.cancel()
        self._download.set_cancellable(False)
        self._catalog_status.setText("Отменяю загрузку…")

    def _finish_pull(self) -> None:
        self._pulling_spec = None
        self._pulling_catalog_id = ""
        self._download.set_cancellable(False)
        self._sync_provider()

    def _on_pull_progress(self, progress: object) -> None:
        if not isinstance(progress, OllamaPullProgress):
            return
        self._download.set_progress(progress.completed, progress.total, 0.0, 0.0)
        if progress.status:
            self._catalog_status.setText(progress.status)

    def _on_pull_finished(self, model: str) -> None:
        self._download.flush()
        self._finish_pull()
        self._catalog_status.setText(f"Модель «{model}» загружена.")

    def _on_pull_failed(self, message: str) -> None:
        self._finish_pull()
        self._catalog_status.setText(f"Не удалось загрузить: {message}")

    # -- загрузка GGUF через менеджер моделей (llama.cpp) --------------------

    def _ensure_backend(self) -> ModelManagerBackend | None:
        """Лениво собрать бэкенд менеджера моделей; ``None`` — загрузка недоступна.

        Попытка сборки одна: при недоступной базе ``_default_backend`` возвращает
        ``None`` и логирует причину, повторять на каждый клик незачем.
        """
        if self.services.backend is None and not self._backend_tried:
            self._backend_tried = True
            self.services.backend = _default_backend(self.event_bus)
        return self.services.backend

    def _start_gguf_download(self, spec: LlmModelSpec) -> None:
        catalog_id = spec.gguf_catalog_id
        if not catalog_id:
            self._catalog_status.setText(f"Для «{spec.name}» нет встроенного GGUF.")
            return
        backend = self._ensure_backend()
        if backend is None:
            self._catalog_status.setText(
                "Менеджер моделей недоступен — выберите файл .gguf вручную."
            )
            return
        self._pulling_spec = spec
        self._pulling_catalog_id = catalog_id
        self._download.reset()
        self._download.set_cancellable(True)
        self._download.show()
        self._catalog_status.setText(f"Скачиваю «{spec.name}»…")
        for row in self._catalog_rows:
            row.button.setEnabled(False)
        try:
            backend.start_download(catalog_id)
        except AyrisError as exc:
            self._finish_pull()
            self._download.hide()
            self._catalog_status.setText(f"Не удалось начать загрузку: {exc.user_message}")
        except Exception as exc:  # бэкенд может кинуть что угодно — вкладка не падает
            _log.exception("не удалось начать загрузку GGUF")
            self._finish_pull()
            self._download.hide()
            self._catalog_status.setText(f"Не удалось начать загрузку: {exc}")

    def _on_gguf_progress(self, event: object) -> None:
        if not isinstance(event, ModelDownloadProgress):
            return
        if event.model_id != self._pulling_catalog_id:
            return
        self._download.set_progress(event.downloaded, event.total, event.speed_bps, event.eta_s)

    def _on_gguf_finished(self, event: object) -> None:
        if not isinstance(event, ModelDownloadFinished):
            return
        if event.model_id != self._pulling_catalog_id:
            return
        spec = self._pulling_spec
        self._download.flush()
        self._activate_downloaded(event.model_id, event.path, spec)
        self._finish_pull()
        name = spec.name if spec is not None else event.model_id
        self._catalog_status.setText(f"Модель «{name}» загружена и выбрана.")

    def _on_gguf_failed(self, event: object) -> None:
        if not isinstance(event, ModelDownloadFailed):
            return
        if event.model_id != self._pulling_catalog_id:
            return
        self._finish_pull()
        self._download.hide()
        if event.cancelled:
            self._catalog_status.setText("Загрузка отменена.")
        else:
            reason = event.user_message or event.error
            self._catalog_status.setText(f"Не удалось загрузить: {reason}")

    def _activate_downloaded(self, catalog_id: str, path: str, spec: LlmModelSpec | None) -> None:
        """Записать путь в конфиг и сделать модель активной в реестре.

        Конфиг пишется напрямую (это работает и без шины, в тестах), а активная
        запись в реестре держит выбор в согласии со вкладкой «Обновления». Правка
        ``ai.model_path`` под :class:`RestartScope.LLM` перезапустит воркер с новой
        моделью и прогреет её на следующем старте.
        """
        record = self._find_record(catalog_id)
        resolved = path or (record.path if record is not None else "")
        updates: dict[str, object] = {"ai.model_path": resolved}
        if spec is not None:
            updates["ai.model"] = spec.name
        self._manager.apply(updates)
        self._refresh_model_path_label()
        self._sync_model_picker()
        backend = self.services.backend
        if backend is not None and record is not None:
            try:
                backend.set_active(record)
            except Exception:  # активация — «на всякий», конфиг уже записан
                _log.exception("не удалось отметить модель активной в реестре")

    def _find_record(self, catalog_id: str) -> ModelRecord | None:
        backend = self.services.backend
        if backend is None:
            return None
        try:
            for record in backend.installed():
                if record.kind == "llm" and record.catalog_id == catalog_id:
                    return record
        except Exception:  # чтение реестра не должно ронять обработчик события
            _log.exception("не удалось найти скачанную запись в реестре")
        return None

    def _pick_gguf(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл модели (.gguf)",
            "",
            "Модели GGUF (*.gguf);;Все файлы (*)",
        )
        if not path:
            return
        self._set_local_model(path)

    def _set_local_model(self, path: str) -> None:
        """Указать свой файл модели: пишем абсолютный путь в конфиг, минуя реестр.

        Вместе с путём обновляем и ``ai.model``: имя модели — это то, что видно в
        пикере и в отчётах об использовании, и оставлять там прежнее значение,
        когда файл уже другой, значит врать про то, чем Айрис думает.
        """
        resolved = str(Path(path))
        name = Path(resolved).stem
        self._local_models[name] = resolved
        self._manager.apply(
            {
                "ai.model_path": resolved,
                "ai.model": name,
                "ai.model_history": self._merge_history(LLAMACPP_PROVIDER, name),
            }
        )
        self._refresh_model_path_label()
        self._sync_model_picker()
        self._catalog_status.setText(f"Выбран файл: {Path(resolved).name}")

    # -- секция параметров генерации ----------------------------------------

    def _build_generation_section(self) -> None:
        self._model_header("Параметры генерации")

        self._temp_slider = SliderField(
            self._theme, minimum=0, maximum=200, value=70, label="Температура"
        )
        self.bind_scaled_slider(self._temp_slider, "ai.temperature", "Температура", factor=100)
        self._model_card(
            "Температура",
            "Разброс ответов: 0 — предсказуемо, 200 — максимально разнообразно (шкала 0–2 ×100).",
            self._temp_slider,
        )

        self._tokens_slider = SliderField(
            self._theme, minimum=64, maximum=32768, value=1024, unit="ток.", label="Лимит ответа"
        )
        self.bind_int_slider(self._tokens_slider, "ai.max_tokens", "Лимит токенов ответа")
        self._model_card(
            "Лимит ответа",
            "Максимум токенов в ответе модели. Больше — длиннее и дороже.",
            self._tokens_slider,
        )

        self._timeout_slider = SliderField(
            self._theme, minimum=2, maximum=300, value=30, unit="с", label="Таймаут запроса"
        )
        self.bind_scaled_slider(
            self._timeout_slider, "ai.request_timeout_sec", "Таймаут запроса", factor=1.0
        )
        self._model_card(
            "Таймаут",
            "Сколько секунд ждать ответ модели, прежде чем прервать запрос.",
            self._timeout_slider,
        )

    # -- секция истории диалога ---------------------------------------------

    def _build_history_section(self) -> None:
        self._model_header("История диалога")

        self._memory_mode_combo = ThemedComboBox()
        self._memory_mode_combo.addItem("Скользящее окно (по умолчанию)", "sliding_window")
        self._memory_mode_combo.addItem("Долговременная память MemGPT (факты + архив)", "memgpt")
        self.bind_combo(self._memory_mode_combo, "ai.memory_mode", "Режим памяти")
        self._model_card(
            "Режим памяти",
            "Тип хранения диалога: простое скользящее окно последних реплик или "
            "долговременная память MemGPT с фактами о пользователе и поиском по архиву.",
            self._memory_mode_combo,
        )

        self._history_slider = SliderField(
            self._theme, minimum=0, maximum=100, value=10, unit="реплик", label="Длина истории"
        )
        self.bind_int_slider(self._history_slider, "ai.history_turns", "Длина истории")
        self._model_card(
            "Сколько помнить",
            "Сколько последних реплик держать в контексте. 0 — не помнить ничего.",
            self._history_slider,
        )

        self._summary_slider = SliderField(
            self._theme, minimum=0, maximum=500, value=30, unit="реплик", label="Порог сжатия"
        )
        self.bind_int_slider(self._summary_slider, "ai.summarize_after_turns", "Порог суммаризации")
        self._model_card(
            "Когда сворачивать",
            "После скольких реплик сворачивать историю в краткий пересказ. 0 — никогда; "
            "иначе не меньше длины истории.",
            self._summary_slider,
        )

        self._recall_slider = SliderField(
            self._theme, minimum=0, maximum=10, value=3, unit="записей", label="Архивных фактов"
        )
        self.bind_int_slider(self._recall_slider, "ai.long_term_recall_limit", "Лимит воспоминаний")
        self._model_card(
            "Подмешивание из архива",
            "Сколько релевантных реплик из архива прошлых дней подмешивать "
            "в запрос к модели (в режиме MemGPT).",
            self._recall_slider,
        )

        body = self.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        row = QHBoxLayout()
        self._clear_button = QPushButton("Очистить историю диалога")
        self._clear_button.setProperty("kind", "danger")
        self._clear_button.clicked.connect(self._confirm_clear_history)
        can_clear = self.services.clear_history is not None
        self._clear_button.setEnabled(can_clear)
        if not can_clear:
            self._clear_button.setToolTip("История станет доступна после запуска модели.")
        row.addWidget(self._clear_button)
        row.addStretch(1)
        layout.addLayout(row)
        self._clear_status = QLabel("")
        self._clear_status.setProperty("role", "muted")
        self._clear_status.setWordWrap(True)
        layout.addWidget(self._clear_status)
        self._model_block(
            "Сброс истории",
            "Забыть текущий разговор с моделью. На заготовленные команды не влияет.",
            body,
        )

    def _confirm_clear_history(self) -> None:
        clear = self.services.clear_history
        if clear is None:
            return
        dialog = ConfirmDialog(
            "Очистить историю диалога?",
            "Айрис забудет текущий разговор с моделью. Команды и настройки останутся.",
            self._theme,
            confirm_text="Очистить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            clear()
        except Exception:
            _log.exception("не удалось очистить историю диалога")
            self._clear_status.setText("Не удалось очистить историю.")
            return
        self._clear_status.setText("История диалога очищена.")

    # -- секция промптов ----------------------------------------------------

    def _build_agent_section(self) -> None:
        self.add_header("Агентный режим")

        self._agent_toggle = ToggleSwitch(self._theme, label="Агентный режим")
        self.bind_toggle(self._agent_toggle, "ai.agent_mode", "Агентный режим")
        self.add_card(
            "Самостоятельные действия",
            "Модель сама вызывает действия в несколько шагов, пока не выполнит "
            "задачу. Нужна облачная модель. Разрешённые для файловых действий "
            "папки — ниже.",
            self._agent_toggle,
        )

        self._allowed_dirs = _AllowedDirsEditor(self._manager, self._theme)
        self.register_refresh(self._allowed_dirs.refresh)
        self.add_block(
            "Папки для файловых действий",
            "Белый список: читать, писать и удалять файлы Айрис может только "
            "внутри этих папок. Пусто — файловые действия отключены. Относится "
            "и к агенту, и к обычным командам.",
            self._allowed_dirs,
        )

    def _build_prompts_section(self) -> None:
        self._model_header("Промпты")
        defaults = AiConfig()

        self._chat_editor = PromptEditor(
            self._theme,
            default_text=defaults.chat_system_prompt,
            build_preview=self._preview_chat,
        )
        self._bind(
            self._chat_editor,
            "ai.chat_system_prompt",
            "Промпт свободного разговора",
            self._chat_editor.text,
            self._chat_editor.setText,
            self._chat_editor.changed,
        )
        self._model_block(
            "Промпт свободного разговора",
            "Ваша «персона» для чата. Ниже — итоговый промпт с шаблоном поведения.",
            self._chat_editor,
        )

        self._nlu_editor = PromptEditor(
            self._theme,
            default_text=defaults.nlu_system_prompt,
            build_preview=self._preview_nlu,
        )
        self._bind(
            self._nlu_editor,
            "ai.nlu_system_prompt",
            "Промпт разбора команд",
            self._nlu_editor.text,
            self._nlu_editor.setText,
            self._nlu_editor.changed,
        )
        self._model_block(
            "Промпт разбора команд",
            "Инструкция для распознавания команд. В предпросмотр подставляется список команд.",
            self._nlu_editor,
        )

    def _preview_chat(self, persona: str) -> str:
        return build_chat_prompt(persona)

    def _preview_nlu(self, persona: str) -> str:
        cards = tuple(self._command_cards())
        return build_nlu_prompt(persona, render_catalog(cards, ""))

    # -- секция пробного запроса --------------------------------------------

    def _build_test_section(self) -> None:
        self._model_header("Спросить у модели")
        self._model_caption(
            "Разовый запрос с текущими настройками: ответ, время и токены. "
            "Историю диалога не затрагивает."
        )
        self._test_panel = LlmTestPanel(self._theme, build_request=self._build_test_request)
        self._model_block(
            "Пробный запрос",
            "Проверьте, что модель отвечает так, как вы настроили.",
            self._test_panel,
        )

    def _build_test_request(self, question: str) -> TestRequest:
        self.flush_pending()
        ai = self._manager.settings.ai
        messages = (
            LlmMessage.system(build_chat_prompt(ai.chat_system_prompt)),
            LlmMessage.user(question),
        )
        return TestRequest(
            client=self._build_client(),
            messages=messages,
            provider=ai.provider,
            model=ai.model,
            temperature=ai.temperature,
            max_tokens=ai.max_tokens,
        )


register_tab("ai", AiTab)
