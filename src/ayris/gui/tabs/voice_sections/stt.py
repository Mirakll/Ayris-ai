"""«Распознавание речи (STT)»: engine, model, mode, cloud key, self-test."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker, Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import RestartScope, SttConfig
from ayris.core.errors import SecretsError
from ayris.core.secrets import SecretsStore, is_valid_ref
from ayris.gui.tabs.voice import AsyncRunner, combo_options
from ayris.gui.widgets import BusyIndicator, InlineNotice, ThemedComboBox

if TYPE_CHECKING:
    from ayris.gui.tabs.voice import VoiceTab

__all__ = ["SttSection"]

_MODE_LABELS = {
    "offline": "Только офлайн",
    "online": "Только облако",
    "auto": "Авто: облако с откатом на офлайн",
}
_OFFLINE_ENGINES = {
    "gigaam": "GigaAM (локально)",
    "vosk": "Vosk (локально)",
    "whisper": "Whisper.cpp (локально)",
}
_PROVIDERS = {
    "yandex": "Яндекс SpeechKit",
    "google": "Google Cloud Speech-to-Text",
    "azure": "Azure Speech",
    "openai": "OpenAI-совместимый (Whisper)",
}


class _SecretField(QWidget):
    """A masked key field over the Windows Credential Manager, never the config.

    The key goes to :class:`~ayris.core.secrets.SecretsStore` under the reference
    named in the config; ``config.toml`` keeps only the reference. Nothing here
    logs the value or writes it to a bound field.
    """

    def __init__(
        self,
        tab: VoiceTab,
        secrets: SecretsStore,
        ref_getter: Callable[[], str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._tab = tab
        self._secrets = secrets
        self._ref_getter = ref_getter
        self.setProperty("transparent", True)
        theme = tab.theme

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))

        self.edit = QLineEdit()
        self.edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.edit.setPlaceholderText("Вставьте ключ — он не попадёт в файл настроек")
        self.edit.setAccessibleName("Ключ облачного сервиса")
        layout.addWidget(self.edit)

        buttons = QHBoxLayout()
        buttons.setSpacing(theme.metric("spacing_sm"))
        self.save_button = QPushButton("Сохранить")
        self.save_button.setProperty("kind", "primary")
        self.save_button.clicked.connect(self._save)
        self.check_button = QPushButton("Проверить")
        self.check_button.clicked.connect(self._check)
        self.delete_button = QPushButton("Удалить")
        self.delete_button.clicked.connect(self._delete)
        for button in (self.save_button, self.check_button, self.delete_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self.status = QLabel()
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def refresh(self) -> None:
        ref = self._ref_getter()
        if not self._secrets.is_available():
            self._set_enabled(False)
            self.status.setText("Хранилище ключей Windows недоступно.")
            return
        self._set_enabled(True)
        if not is_valid_ref(ref):
            self.status.setText("Задайте короткое латинское имя записи, например yandex.")
            return
        stored = self._secrets.status(ref).stored
        self.delete_button.setEnabled(stored)
        self.status.setText(
            f"Запись «{ref}»: ключ сохранён." if stored else f"Запись «{ref}»: ключ не задан."
        )

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.edit, self.save_button, self.check_button, self.delete_button):
            widget.setEnabled(enabled)

    def _save(self) -> None:
        value = self.edit.text().strip()
        ref = self._ref_getter()
        if not value:
            self.status.setText("Введите ключ, затем нажмите «Сохранить».")
            return
        try:
            self._secrets.save(ref, value)
        except SecretsError as exc:
            self.status.setText(exc.user_message)
            return
        self.edit.clear()  # the value never lingers in a widget after it is stored
        self.refresh()

    def _delete(self) -> None:
        ref = self._ref_getter()
        try:
            removed = self._secrets.delete(ref)
        except SecretsError as exc:
            self.status.setText(exc.user_message)
            return
        self.edit.clear()
        self.refresh()
        if not removed:
            self.status.setText(f"Запись «{ref}»: ключа не было.")

    def _check(self) -> None:
        ref = self._ref_getter()
        typed = self.edit.text().strip()
        if typed:
            if is_valid_ref(typed):
                self.status.setText("Это похоже на имя записи, а не на ключ. Вставьте сам ключ.")
            else:
                self.status.setText("Ключ выглядит правдоподобно. Нажмите «Сохранить».")
            return
        if self._secrets.status(ref).stored:
            self.status.setText(f"Запись «{ref}»: ключ на месте. Проверка в облаке — при команде.")
        else:
            self.status.setText(f"Запись «{ref}»: ключ не задан — облако работать не будет.")


class SttSection:
    """Builds the recognition part of the «Голос» tab."""

    def __init__(self, tab: VoiceTab) -> None:
        self._tab = tab
        self._runner = AsyncRunner(tab)
        self._runner.finished.connect(self._on_test_done)
        self._runner.failed.connect(self._on_test_failed)
        self._build()
        tab.register_refresh(self.refresh)

    def _build(self) -> None:
        tab = self._tab
        tab.add_header("Распознавание речи (STT)")

        self._mode_combo = ThemedComboBox()
        for value, label in combo_options(SttConfig, "mode", _MODE_LABELS):
            self._mode_combo.addItem(label, value)
        tab.bind_combo(self._mode_combo, "voice.stt.mode", "Режим распознавания")
        tab.add_card(
            "Режим",
            "Онлайн точнее и без моделей на диске, офлайн работает без сети. "
            "Авто начинает с облака и откатывается на офлайн при потере связи.",
            self._mode_combo,
        )

        self._engine_combo = ThemedComboBox()
        for value, label in combo_options(SttConfig, "offline_engine", _OFFLINE_ENGINES):
            self._engine_combo.addItem(label, value)
        tab.bind_combo(self._engine_combo, "voice.stt.offline_engine", "Локальный движок")
        self._engine_combo.currentIndexChanged.connect(self._reload_models)
        tab.add_card(
            "Локальный движок",
            "Движок распознавания для офлайн-режима. GigaAM точнее на русском.",
            self._engine_combo,
        )

        self._model_combo = ThemedComboBox()
        tab.bind_combo(self._model_combo, "voice.stt.offline_model", "Локальная модель")
        tab.add_card(
            "Модель",
            "Файл модели в папке моделей. Модели скачиваются в менеджере моделей.",
            self._model_combo,
        )

        self._model_notice = InlineNotice(
            "Модель для выбранного движка не скачана.",
            tab.theme,
            kind="warning",
        )
        self._model_notice.close_button.setText("Менеджер моделей")
        self._model_notice.close_button.clicked.disconnect()
        self._model_notice.close_button.clicked.connect(self._open_model_manager)
        self._model_notice.hide()
        tab.add_widget(self._model_notice)

        self._provider_combo = ThemedComboBox()
        for value, label in combo_options(SttConfig, "online_provider", _PROVIDERS):
            self._provider_combo.addItem(label, value)
        tab.bind_combo(self._provider_combo, "voice.stt.online_provider", "Облачный провайдер")
        self._provider_combo.currentIndexChanged.connect(
            lambda _index: self._update_cloud_visibility()
        )
        tab.add_card(
            "Облачный провайдер",
            "Сервис распознавания для онлайн-режима.",
            self._provider_combo,
        )

        # Endpoint and model are the paste-your-own-service fields: the OpenAI
        # provider POSTs to whatever URL is set here, so any OpenAI-compatible
        # transcription API (a self-hosted Whisper, Groq, a regional proxy) works
        # without a code change. Hidden until that provider is chosen — the other
        # three have their address and model fixed for them.
        self._endpoint_edit = QLineEdit()
        self._endpoint_edit.setPlaceholderText("https://api.openai.com/v1/audio/transcriptions")
        tab.bind_line_edit(self._endpoint_edit, "voice.stt.online_endpoint", "Адрес сервиса")
        self._endpoint_card = tab.add_card(
            "Адрес сервиса",
            "Полный URL метода транскрипции OpenAI-совместимого API. "
            "Пусто — адрес OpenAI по умолчанию.",
            self._endpoint_edit,
        )

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("whisper-1")
        tab.bind_line_edit(self._model_edit, "voice.stt.online_model", "Модель распознавания")
        self._model_card = tab.add_card(
            "Модель",
            "Идентификатор модели распознавания у сервиса. "
            "Пусто — модель по умолчанию (whisper-1).",
            self._model_edit,
        )

        for card in (self._endpoint_card, self._model_card):
            card.setVisible(False)

        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("yandex")
        tab.bind_line_edit(self._ref_edit, "voice.stt.credential_ref", "Имя записи ключа")
        key_card_body = tab.panel()
        key_layout = QVBoxLayout(key_card_body)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.setSpacing(tab.theme.metric("spacing_sm"))
        ref_row = QHBoxLayout()
        ref_row.setSpacing(tab.theme.metric("spacing_sm"))
        ref_label = QLabel("Имя записи:")
        ref_label.setProperty("role", "secondary")
        ref_row.addWidget(ref_label)
        ref_row.addWidget(self._ref_edit, 1)
        key_layout.addLayout(ref_row)
        self._secret = _SecretField(tab, tab.services.secrets, self._ref_edit.text)
        self._ref_edit.textChanged.connect(lambda _text: self._secret.refresh())
        key_layout.addWidget(self._secret)
        tab.add_block(
            "Ключ облачного сервиса",
            "Ключ хранится в диспетчере учётных данных Windows, а не в config.toml.",
            key_card_body,
        )

        self._cloud_notice = InlineNotice(
            "Облачный движок выбран, но ключ не задан — команды будут молча отклоняться.",
            tab.theme,
            kind="warning",
        )
        self._cloud_notice.hide()
        tab.add_widget(self._cloud_notice)

        self._build_test_card()
        tab.add_restart_bar(RestartScope.STT)

    def _build_test_card(self) -> None:
        tab = self._tab
        body = tab.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))
        row = QHBoxLayout()
        row.setSpacing(tab.theme.metric("spacing_sm"))
        self._test_button = QPushButton("Записать фразу")
        self._test_button.clicked.connect(self._run_test)
        row.addWidget(self._test_button)
        self._busy = BusyIndicator(tab.theme, active=False)
        self._busy.hide()
        row.addWidget(self._busy)
        row.addStretch(1)
        layout.addLayout(row)
        self._test_result = QLabel("Скажите фразу — распознанный текст появится здесь.")
        self._test_result.setProperty("role", "secondary")
        self._test_result.setWordWrap(True)
        self._test_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._test_result)
        tab.add_block(
            "Тест распознавания",
            "Проверьте микрофон и движок: запишите короткую фразу и сверьте текст.",
            body,
        )

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        self._reload_models()
        self._update_notices()
        self._update_cloud_visibility()
        ready = self._tab.services.transcribe_once is not None
        self._test_button.setEnabled(ready)
        self._test_button.setToolTip(
            "" if ready else "Тест станет доступен после запуска распознавания"
        )
        self._secret.refresh()

    def _update_cloud_visibility(self) -> None:
        """Show the endpoint and model fields only for the OpenAI-compatible provider.

        They are the paste-your-own-service settings: the OpenAI recogniser sends its
        request to whatever address is set, so any OpenAI-compatible transcription API
        works. Yandex, Google and Azure have their address and model fixed for them, so
        the fields would only confuse there.
        """
        provider = str(
            self._provider_combo.currentData()
            or self._tab.manager.settings.voice.stt.online_provider
        )
        is_openai = provider == "openai"
        self._endpoint_card.setVisible(is_openai)
        self._model_card.setVisible(is_openai)

    def _reload_models(self) -> None:
        settings = self._tab.manager.settings.voice.stt
        engine = self._engine_combo.currentData() or settings.offline_engine
        catalog = self._tab.model_catalog()
        entries = catalog.for_engine("stt", str(engine))
        current = settings.offline_model
        with QSignalBlocker(self._model_combo):
            self._model_combo.clear()
            seen: set[str] = set()
            for entry in entries:
                self._model_combo.addItem(entry.name, entry.install_name)
                seen.add(entry.install_name)
            if current and current not in seen:
                # Keep whatever the config points at, even if the catalog lost it.
                self._model_combo.addItem(f"{current} (нет в каталоге)", current)
            index = self._model_combo.findData(current)
            if index >= 0:
                self._model_combo.setCurrentIndex(index)

    def _update_notices(self) -> None:
        settings = self._tab.manager.settings.voice.stt
        uses_offline = settings.mode in ("offline", "auto")
        installed = self._tab.installed_models("stt")
        missing = bool(installed) and settings.offline_model not in installed
        self._model_notice.setVisible(uses_offline and missing)

        uses_online = settings.mode in ("online", "auto")
        has_key = self._tab.services.secrets.status(settings.credential_ref).stored
        self._cloud_notice.setVisible(uses_online and not has_key)

    def _open_model_manager(self) -> None:
        opener = self._tab.services.open_model_manager
        if opener is not None:
            opener()

    # -- self-test ----------------------------------------------------------

    def _run_test(self) -> None:
        service = self._tab.services.transcribe_once
        if service is None:
            return
        self._test_button.setEnabled(False)
        self._busy.show()
        self._busy.setActive(True)
        self._test_result.setText("Слушаю…")
        self._runner.run(service)

    def _on_test_done(self, result: object) -> None:
        self._finish_test()
        if isinstance(result, tuple) and len(result) == 2:
            text, elapsed_ms = result
            self._test_result.setText(f"«{text}»  ·  {int(elapsed_ms)} мс")
        else:
            self._test_result.setText(str(result))

    def _on_test_failed(self, message: str) -> None:
        self._finish_test()
        self._test_result.setText(f"Не удалось распознать: {message}")

    def _finish_test(self) -> None:
        self._busy.setActive(False)
        self._busy.hide()
        self._test_button.setEnabled(self._tab.services.transcribe_once is not None)
