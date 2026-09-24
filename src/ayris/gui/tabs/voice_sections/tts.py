"""«Синтез речи (TTS)»: engine, voice, speed/pitch/volume, listen, custom model."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSignalBlocker
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from ayris.audio.tts.cloud_base import is_cloud_engine
from ayris.core.config import RestartScope, TtsConfig
from ayris.core.paths import get_paths
from ayris.gui.tabs.voice import AsyncRunner, combo_options
from ayris.gui.tabs.voice_sections.stt import _SecretField
from ayris.gui.widgets import BusyIndicator, SliderField, ThemedComboBox, ToggleSwitch
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.gui.tabs.voice import VoiceTab

__all__ = ["TtsSection"]

_log = get_logger(__name__)

_ENGINES = {
    "piper": "Piper (локально)",
    "silero": "Silero (локально)",
    "xtts": "Coqui XTTS (локально)",
    "yandex": "Яндекс SpeechKit",
    "google": "Google Cloud TTS",
    "azure": "Azure Speech",
    "elevenlabs": "ElevenLabs",
    "openai": "OpenAI-совместимый (OpenRouter…)",
}
#: Which engines accept a user-supplied model file, and what extension it wants.
_CUSTOM_MODEL_EXT = {
    "piper": ".onnx",
    "xtts": ".pth",
}
#: Suggested voices for a cloud engine, shown in the editable voice box so a user can
#: pick without typing. Not exhaustive — the box stays editable, so any voice the
#: service accepts can be typed — and only the OpenAI-compatible family has a stable
#: public set worth seeding.
_CLOUD_VOICES = {
    "openai": (
        ("Alloy", "alloy"),
        ("Echo", "echo"),
        ("Fable", "fable"),
        ("Onyx", "onyx"),
        ("Nova", "nova"),
        ("Shimmer", "shimmer"),
    ),
}


class TtsSection:
    """Builds the synthesis part of the «Голос» tab."""

    def __init__(self, tab: VoiceTab) -> None:
        self._tab = tab
        self._runner = AsyncRunner(tab)
        self._runner.finished.connect(lambda _result: self._finish_listen())
        self._runner.failed.connect(self._on_listen_failed)
        self._build()
        tab.register_refresh(self.refresh)

    def _build(self) -> None:
        tab = self._tab
        tab.add_header("Синтез речи (TTS)")

        self._engine_combo = ThemedComboBox()
        for value, label in combo_options(TtsConfig, "engine", _ENGINES):
            self._engine_combo.addItem(label, value)
        tab.bind_combo(self._engine_combo, "voice.tts.engine", "Движок синтеза")
        self._engine_combo.currentIndexChanged.connect(self._reload_voices)
        tab.add_card(
            "Движок", "Чем озвучивать ответы. Piper — быстрый локальный голос.", self._engine_combo
        )

        self._voice_combo = ThemedComboBox()
        tab.bind_combo(self._voice_combo, "voice.tts.voice", "Голос")
        tab.add_card(
            "Голос", "Голос выбранного движка. Свои модели можно загрузить ниже.", self._voice_combo
        )

        self._build_cloud_cards()

        self._speed = SliderField(
            tab.theme, minimum=50, maximum=200, value=100, unit="%", label="Скорость"
        )
        tab.bind_scaled_slider(self._speed, "voice.tts.speed", "Скорость речи", factor=100)
        tab.add_card("Скорость", "1.0× — обычный темп речи.", self._speed)

        self._pitch = SliderField(
            tab.theme, minimum=50, maximum=200, value=100, unit="%", label="Тон"
        )
        tab.bind_scaled_slider(self._pitch, "voice.tts.pitch", "Тон голоса", factor=100)
        tab.add_card("Тон", "Высота голоса относительно обычной.", self._pitch)

        self._volume = SliderField(
            tab.theme, minimum=0, maximum=100, value=80, unit="%", label="Громкость"
        )
        tab.bind_int_slider(self._volume, "voice.tts.volume", "Громкость озвучки")
        tab.add_card("Громкость", "Громкость озвучки помощника.", self._volume)

        self._build_listen_card()
        self._build_custom_model_card()

        self._fallback = ToggleSwitch(tab.theme, label="Облачный запасной синтез")
        tab.bind_toggle(self._fallback, "voice.tts.cloud_fallback", "Облачный запасной синтез")
        self._fallback.toggled.connect(lambda _checked: self._update_cloud_visibility())
        tab.add_card(
            "Запасной облачный голос",
            "Если локальный синтез не справился, озвучить ответ через облако.",
            self._fallback,
        )

        tab.add_restart_bar(RestartScope.TTS)

    def _build_cloud_cards(self) -> None:
        """Endpoint, model and key — the paste-your-own-cloud-TTS fields.

        Hidden until a cloud engine is chosen (or the cloud fallback is on): endpoint
        and model belong to the generic OpenAI-compatible engine, so they show only
        for it; the key is needed by every cloud engine and by the fallback, so it
        shows whenever either wants a cloud voice. The key itself never touches the
        config — :class:`~ayris.gui.tabs.voice_sections.stt._SecretField` puts it in
        the Windows credential store under the reference named here.
        """
        tab = self._tab

        self._endpoint_edit = QLineEdit()
        self._endpoint_edit.setPlaceholderText("https://openrouter.ai/api/v1")
        tab.bind_line_edit(self._endpoint_edit, "voice.tts.endpoint", "Адрес сервиса")
        self._endpoint_card = tab.add_card(
            "Адрес сервиса",
            "Базовый URL OpenAI-совместимого API. Пусто — адрес по умолчанию (OpenRouter).",
            self._endpoint_edit,
        )

        self._model_edit = QLineEdit()
        self._model_edit.setPlaceholderText("например, openai/gpt-4o-mini-tts")
        tab.bind_line_edit(self._model_edit, "voice.tts.model", "Модель синтеза")
        self._model_card = tab.add_card(
            "Модель",
            "Идентификатор модели синтеза у сервиса — обязателен для облачного голоса.",
            self._model_edit,
        )

        self._ref_edit = QLineEdit()
        self._ref_edit.setPlaceholderText("openai")
        tab.bind_line_edit(self._ref_edit, "voice.tts.credential_ref", "Имя записи ключа")
        body = tab.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))
        ref_row = QHBoxLayout()
        ref_row.setSpacing(tab.theme.metric("spacing_sm"))
        ref_label = QLabel("Имя записи:")
        ref_label.setProperty("role", "secondary")
        ref_row.addWidget(ref_label)
        ref_row.addWidget(self._ref_edit, 1)
        layout.addLayout(ref_row)
        self._secret = _SecretField(tab, tab.services.secrets, self._ref_edit.text)
        self._ref_edit.textChanged.connect(lambda _text: self._secret.refresh())
        layout.addWidget(self._secret)
        self._cloud_card = tab.add_block(
            "Ключ облачного сервиса",
            "Ключ хранится в диспетчере учётных данных Windows, а не в config.toml.",
            body,
        )

        for card in (self._endpoint_card, self._model_card, self._cloud_card):
            card.setVisible(False)

    def _build_listen_card(self) -> None:
        tab = self._tab
        row = tab.panel()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))
        self._listen_button = QPushButton("Прослушать")
        self._listen_button.setProperty("kind", "primary")
        self._listen_button.clicked.connect(self._listen)
        layout.addWidget(self._listen_button)
        self._busy = BusyIndicator(tab.theme, active=False)
        self._busy.hide()
        layout.addWidget(self._busy)
        layout.addStretch(1)
        tab.add_block(
            "Проба голоса",
            "Произнести тестовую фразу выбранным голосом с текущими скоростью, тоном и громкостью.",
            row,
        )

    def _build_custom_model_card(self) -> None:
        tab = self._tab
        body = tab.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))
        self._load_button = QPushButton("Выбрать файл модели…")
        self._load_button.clicked.connect(self._load_model)
        layout.addWidget(self._load_button)
        self._load_status = QLabel("")
        self._load_status.setProperty("role", "muted")
        self._load_status.setWordWrap(True)
        layout.addWidget(self._load_status)
        tab.add_block(
            "Своя модель",
            "Piper — файл .onnx, Coqui — файл .pth. Файл копируется в папку голосов.",
            body,
        )

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        self._reload_voices()
        can_listen = self._tab.services.speak_sample is not None
        self._listen_button.setEnabled(can_listen)
        self._listen_button.setToolTip(
            "" if can_listen else "Проба станет доступна после запуска синтеза"
        )
        engine = str(self._tab.manager.settings.voice.tts.engine)
        supports_custom = engine in _CUSTOM_MODEL_EXT
        self._load_button.setEnabled(supports_custom)
        self._load_button.setToolTip(
            "" if supports_custom else "Загрузка своей модели доступна для Piper и Coqui"
        )

    def _reload_voices(self) -> None:
        settings = self._tab.manager.settings.voice.tts
        engine = str(self._engine_combo.currentData() or settings.engine)
        if is_cloud_engine(engine):
            self._reload_cloud_voices(engine, settings.voice)
        else:
            self._reload_local_voices(engine, settings.voice)
        self._update_cloud_visibility()

    def _reload_local_voices(self, engine: str, current: str) -> None:
        """The installed voices of a local engine, chosen from a fixed list."""
        self._set_voice_editable(False)
        catalog = self._tab.model_catalog()
        entries = catalog.for_engine("tts", engine)
        with QSignalBlocker(self._voice_combo):
            self._voice_combo.clear()
            seen: set[str] = set()
            for entry in entries:
                self._voice_combo.addItem(entry.label, entry.install_name)
                seen.add(entry.install_name)
            for name in self._local_voice_files(engine):
                if name not in seen:
                    self._voice_combo.addItem(name, name)
                    seen.add(name)
            if current and current not in seen:
                self._voice_combo.addItem(current, current)
            index = self._voice_combo.findData(current)
            if index >= 0:
                self._voice_combo.setCurrentIndex(index)

    def _reload_cloud_voices(self, engine: str, current: str) -> None:
        """Seed a cloud engine's suggested voices, keep the box editable.

        The service accepts far more voices than any list can hold, so the box stays
        editable: the suggestions are a convenience, and a voice not among them is
        typed straight in (see :meth:`_commit_typed_voice`).
        """
        self._set_voice_editable(True)
        with QSignalBlocker(self._voice_combo):
            self._voice_combo.clear()
            seen: set[str] = set()
            for label, voice_id in _CLOUD_VOICES.get(engine, ()):
                self._voice_combo.addItem(label, voice_id)
                seen.add(voice_id)
            if current and current not in seen:
                self._voice_combo.addItem(current, current)
                seen.add(current)
            index = self._voice_combo.findData(current)
            if index >= 0:
                self._voice_combo.setCurrentIndex(index)
            else:
                line = self._voice_combo.lineEdit()
                if line is not None:
                    line.setText(current)

    def _set_voice_editable(self, editable: bool) -> None:
        """Flip the voice box between a fixed picker and a free-text field.

        Toggling recreates the internal line edit, so the ``editingFinished`` hook is
        (re)connected each time the box becomes editable; the guard keeps a no-op call
        from dropping text the user is mid-way through typing.
        """
        combo = self._voice_combo
        if combo.isEditable() == editable:
            return
        combo.setEditable(editable)
        if editable:
            combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            line = combo.lineEdit()
            if line is not None:
                line.setPlaceholderText("Например alloy — или впишите голос сервиса")
                line.editingFinished.connect(self._commit_typed_voice)

    def _commit_typed_voice(self) -> None:
        """Turn a typed voice name into the combo's selected value.

        With ``NoInsert`` Qt never adds the text itself, so a name the user typed is
        added here as its own data and selected, which fires ``currentIndexChanged``
        and lets the ordinary combo binding save it. A name matching a suggestion's
        id or label selects that entry instead of duplicating it.
        """
        combo = self._voice_combo
        line = combo.lineEdit()
        if line is None:
            return
        text = line.text().strip()
        if not text:
            return
        index = combo.findData(text)
        if index < 0:
            index = combo.findText(text)
        if index < 0:
            combo.addItem(text, text)
            index = combo.findData(text)
        if index >= 0 and index != combo.currentIndex():
            combo.setCurrentIndex(index)

    def _update_cloud_visibility(self) -> None:
        """Show the cloud fields the current engine and fallback actually need.

        Endpoint and model are the generic OpenAI-compatible engine's own settings, so
        they appear only for it; the key is wanted by any cloud engine and by the cloud
        fallback, so it appears whenever either does.
        """
        settings = self._tab.manager.settings.voice.tts
        engine = str(self._engine_combo.currentData() or settings.engine)
        is_openai = engine == "openai"
        want_key = is_cloud_engine(engine) or self._fallback.isChecked()
        self._endpoint_card.setVisible(is_openai)
        self._model_card.setVisible(is_openai)
        self._cloud_card.setVisible(want_key)
        if want_key:
            self._secret.refresh()

    def _local_voice_files(self, engine: str) -> list[str]:
        """Voice files already sitting in the profile's TTS folder."""
        ext = _CUSTOM_MODEL_EXT.get(engine)
        if ext is None:
            return []
        try:
            directory = get_paths().tts_models_dir
            return sorted(path.name for path in directory.glob(f"*{ext}"))
        except OSError:
            return []

    # -- listen -------------------------------------------------------------

    def _listen(self) -> None:
        service = self._tab.services.speak_sample
        if service is None:
            return
        self._listen_button.setEnabled(False)
        self._busy.show()
        self._busy.setActive(True)
        self._runner.run(lambda: (service(), None)[1])

    def _finish_listen(self) -> None:
        self._busy.setActive(False)
        self._busy.hide()
        self._listen_button.setEnabled(self._tab.services.speak_sample is not None)

    def _on_listen_failed(self, message: str) -> None:
        self._finish_listen()
        self._load_status.setText(f"Не удалось озвучить: {message}")

    # -- custom model -------------------------------------------------------

    def _load_model(self) -> None:
        engine = str(self._tab.manager.settings.voice.tts.engine)
        ext = _CUSTOM_MODEL_EXT.get(engine)
        if ext is None:
            return
        chosen, _filter = QFileDialog.getOpenFileName(
            self._tab,
            "Выберите файл модели голоса",
            "",
            f"Модель голоса (*{ext})",
        )
        if not chosen:
            return
        error = self._accept_model_file(Path(chosen), engine)
        if error is not None:
            self._load_status.setText(error)
            return
        name = Path(chosen).name
        self._reload_voices()
        index = self._voice_combo.findData(name)
        if index >= 0:
            self._voice_combo.setCurrentIndex(index)  # emits the bound change
        self._load_status.setText(f"Голос «{name}» добавлен.")

    def _accept_model_file(self, source: Path, engine: str) -> str | None:
        """Validate and copy a voice model. Returns a Russian error, or ``None``.

        Split out from :meth:`_load_model` so the validation can be tested without
        a file dialog.
        """
        ext = _CUSTOM_MODEL_EXT.get(engine)
        if ext is None:
            return "Для этого движка нельзя загрузить свою модель."
        if source.suffix.lower() != ext:
            return f"Ожидается файл {ext} для выбранного движка."
        try:
            target_dir = get_paths().tts_models_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            destination = target_dir / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        except OSError as exc:
            _log.exception("не удалось скопировать модель голоса")
            return f"Не удалось скопировать файл: {exc}"
        return None
