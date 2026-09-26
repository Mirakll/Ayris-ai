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

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
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
from ayris.core.errors import AyrisError, SecretsError
from ayris.core.events import EventBus
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
    SettingCard,
    SliderField,
    ThemedComboBox,
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
    create_llm_client,
    is_cloud_provider,
    is_local_provider,
)
from ayris.nlu.llm.modes import describe_mode
from ayris.nlu.llm.ollama_client import OllamaPullProgress
from ayris.nlu.llm.prompts import build_chat_prompt, build_nlu_prompt
from ayris.nlu.llm.tools import CommandCard, render_catalog
from ayris.utils.logger import get_logger

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


@dataclass(slots=True)
class AiServices:
    """Внешние зависимости вкладки: секреты, фабрика клиентов, команды, память, ОЗУ.

    Всё — с дефолтами, чтобы вкладка поднималась и без приложения (в тестах и
    предпросмотре). ``clear_history`` приходит от пайплайна: без него кнопка сброса
    истории заблокирована. ``total_ram_mb`` позволяет тестам не звать ``psutil``.
    """

    secrets: SecretsStore = field(default_factory=get_secrets)
    client_factory: Callable[..., LlmClient] = create_llm_client
    command_cards: Callable[[], Sequence[CommandCard]] = _no_cards
    clear_history: Callable[[], None] | None = None
    total_ram_mb: int = 0


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

    def __init__(self, spec: LlmModelSpec, theme: ThemeManager, parent: QWidget | None = None) -> None:
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
        self._ram_mb = self.services.total_ram_mb or detect_total_ram_mb()
        self._command_cards = self.services.command_cards
        self._mode_buttons: dict[NluMode, QRadioButton] = {}

        self._probe_runner = AsyncRunner(self)
        self._probe_runner.finished.connect(self._on_probe_done)
        self._probe_runner.failed.connect(self._on_probe_failed)
        self._pull_runner = _PullRunner(self)
        self._pull_runner.progress.connect(self._on_pull_progress)
        self._pull_runner.finished.connect(self._on_pull_finished)
        self._pull_runner.failed.connect(self._on_pull_failed)

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
        self._build_prompts_section()
        self._build_test_section()
        self.content.addStretch(1)

        self.register_refresh(self._sync_mode)
        self.register_refresh(self._sync_provider)
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
            radio.toggled.connect(
                lambda checked, m=mode: self._on_mode_toggled(m, checked)
            )
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

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("qwen2.5:7b-instruct")
        self.bind_line_edit(self._model_edit, "ai.model", "Модель")
        self._model_card(
            "Модель",
            "Название модели у поставщика. Проверка ниже подтянет список прямо с сервера.",
            self._model_edit,
        )

        self._model_combo = ThemedComboBox()
        self._model_combo.setEnabled(False)
        self._model_combo.addItem("Список появится после проверки", "")
        self._model_combo.activated.connect(self._pick_model)
        self._models_card = self._model_card(
            "Модели с сервера",
            "Выберите из того, что вернул сервис при проверке.",
            self._model_combo,
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
        for spec in RECOMMENDED:
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
        self._catalog_block = self._model_block(
            "Рекомендованные модели",
            "Оценка по вашей памяти. Загрузка доступна для Ollama — модель тянется по имени.",
            body,
        )

    def _pick_model(self, _index: int) -> None:
        model = self._model_combo.currentData()
        if not model:
            return
        self._manager.apply({"ai.model": str(model)})

    def _sync_provider(self) -> None:
        settings = self._manager.settings
        if not mode_from_config(settings).uses_llm:
            return
        provider = str(self._provider_combo.currentData() or settings.ai.provider)
        cloud = is_cloud_provider(provider)
        local = is_local_provider(provider)
        self._host_card.setVisible(provider in HOST_PROVIDERS)
        self._key_block.setVisible(cloud)
        self._catalog_block.setVisible(local)
        self._probe_block.setVisible(cloud or (local and provider in HOST_PROVIDERS))
        self._probe_button.setText("Проверить ключ" if cloud else "Проверить соединение")
        for row in self._catalog_rows:
            row.set_verdict(self._ram_mb)
            row.set_downloadable(provider == "ollama" and row.spec.runs_on(provider))
        self._key_field.refresh()

    # -- проба соединения и списка моделей -----------------------------------

    def _build_client(self) -> LlmClient:
        ai = self._manager.settings.ai
        return self.services.client_factory(
            ai.provider,
            model=ai.model,
            base_url=ai.host,
            credential_ref=ai.credential_ref,
            temperature=ai.temperature,
            max_tokens=ai.max_tokens,
            read_timeout_sec=ai.request_timeout_sec,
            store=self.services.secrets,
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
        self._model_combo.clear()
        if not models:
            self._model_combo.addItem("Сервис не вернул список моделей", "")
            self._model_combo.setEnabled(False)
            return
        for name in models:
            self._model_combo.addItem(name, name)
        self._model_combo.setEnabled(True)
        index = self._model_combo.findData(self._manager.settings.ai.model)
        if index >= 0:
            self._model_combo.setCurrentIndex(index)

    # -- загрузка модели через Ollama ---------------------------------------

    def _start_pull(self, spec: object) -> None:
        if not isinstance(spec, LlmModelSpec) or self._pulling_spec is not None:
            return
        self.flush_pending()
        if self._manager.settings.ai.provider != "ollama":
            self._catalog_status.setText("Загрузка доступна только для Ollama.")
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
        self._pull_runner.cancel()
        self._download.set_cancellable(False)
        self._catalog_status.setText("Отменяю загрузку…")

    def _finish_pull(self) -> None:
        self._pulling_spec = None
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

