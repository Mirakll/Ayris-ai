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
    QPushButton,
    QVBoxLayout,
)

from ayris.core.config import RestartScope, TtsConfig
from ayris.core.paths import get_paths
from ayris.gui.tabs.voice import AsyncRunner, combo_options
from ayris.gui.widgets import BusyIndicator, SliderField, ToggleSwitch
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.gui.tabs.voice import VoiceTab

__all__ = ["TtsSection"]

_log = get_logger(__name__)

_ENGINES = {
    "piper": "Piper (локально)",
    "silero": "Silero (локально)",
    "xtts": "Coqui XTTS (локально)",
    "sapi": "Windows SAPI",
    "yandex": "Яндекс SpeechKit",
    "elevenlabs": "ElevenLabs",
}
#: Which engines accept a user-supplied model file, and what extension it wants.
_CUSTOM_MODEL_EXT = {
    "piper": ".onnx",
    "xtts": ".pth",
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

        self._engine_combo = QComboBox()
        for value, label in combo_options(TtsConfig, "engine", _ENGINES):
            self._engine_combo.addItem(label, value)
        tab.bind_combo(self._engine_combo, "voice.tts.engine", "Движок синтеза")
        self._engine_combo.currentIndexChanged.connect(self._reload_voices)
        tab.add_card(
            "Движок", "Чем озвучивать ответы. Piper — быстрый локальный голос.", self._engine_combo
        )

        self._voice_combo = QComboBox()
        tab.bind_combo(self._voice_combo, "voice.tts.voice", "Голос")
        tab.add_card(
            "Голос", "Голос выбранного движка. Свои модели можно загрузить ниже.", self._voice_combo
        )

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
        tab.add_card(
            "Запасной облачный голос",
            "Если локальный синтез не справился, озвучить ответ через облако.",
            self._fallback,
        )

        tab.add_restart_bar(RestartScope.TTS)

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
        engine = self._engine_combo.currentData() or settings.engine
        catalog = self._tab.model_catalog()
        entries = catalog.for_engine("tts", str(engine))
        current = settings.voice
        with QSignalBlocker(self._voice_combo):
            self._voice_combo.clear()
            seen: set[str] = set()
            for entry in entries:
                self._voice_combo.addItem(entry.label, entry.install_name)
                seen.add(entry.install_name)
            for name in self._local_voice_files(str(engine)):
                if name not in seen:
                    self._voice_combo.addItem(name, name)
                    seen.add(name)
            if current and current not in seen:
                self._voice_combo.addItem(current, current)
            index = self._voice_combo.findData(current)
            if index >= 0:
                self._voice_combo.setCurrentIndex(index)

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
