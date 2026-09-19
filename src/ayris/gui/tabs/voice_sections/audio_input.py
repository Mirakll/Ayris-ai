"""«Аудио вход»: device with hot-plug, gain, level meter, VAD, RNNoise, calibrate."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ayris.audio.devices import DeviceDirection, list_devices
from ayris.core.config import AudioInputConfig, RestartScope
from ayris.core.errors import AudioError
from ayris.core.events import AudioLevelChanged
from ayris.gui.tabs.voice import AsyncRunner, combo_options
from ayris.gui.widgets import ConfirmDialog, LevelMeter, SliderField, ThemedComboBox
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.audio.calibration import CalibrationReport, Recommendation
    from ayris.audio.devices import AudioDevice, DeviceEnumerator
    from ayris.gui.tabs.voice import VoiceTab

__all__ = ["AudioInputSection"]

_log = get_logger(__name__)

_GAIN_FACTOR = 100
_VAD_FACTOR = 100
_DENOISE = {
    "off": "Выключено",
    "rnnoise": "RNNoise (лучшее качество, +~10 мс)",
    "spectral": "Спектральное (лёгкое)",
}
#: The identifier the config stores for «system default».
_DEFAULT_DEVICE = ""


class _AudioSignals(QObject):
    """Marshals the microphone level from the worker thread onto the GUI thread."""

    level = Signal(float)


class AudioInputSection:
    """Builds the microphone part of the «Голос» tab."""

    def __init__(self, tab: VoiceTab) -> None:
        self._tab = tab
        self._enumerator_cache: DeviceEnumerator | None = tab.services.devices
        self._enumerator_tried = tab.services.devices is not None
        self._runner = AsyncRunner(tab)
        self._runner.finished.connect(self._on_calibrated)
        self._runner.failed.connect(self._on_calibration_failed)

        self._signals = _AudioSignals(tab)
        self._signals.level.connect(self._on_level)

        self._build()
        tab.register_refresh(self.refresh)
        self._subscribe()

    def _build(self) -> None:
        tab = self._tab
        tab.add_header("Аудио вход")

        self._device_combo = ThemedComboBox()
        tab.tame_combo(self._device_combo)
        self._device_notice = QLabel("")
        self._device_notice.setProperty("role", "muted")
        self._device_notice.setWordWrap(True)
        self._populate_devices()
        tab.bind_combo(self._device_combo, "voice.audio_input.device", "Устройство записи")
        device_body = tab.panel()
        device_layout = QHBoxLayout(device_body)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(tab.theme.metric("spacing_sm"))
        device_layout.addWidget(self._device_combo, 1)
        refresh_button = QPushButton("Обновить")
        refresh_button.setToolTip("Перечитать список устройств после подключения микрофона")
        refresh_button.clicked.connect(self._rescan_devices)
        device_layout.addWidget(refresh_button)
        # A block (stacked), not a card (side-by-side): the body is a combo + a
        # button in a row, which side by side with the text column doubles the
        # card's minimum width and forces a horizontal scrollbar on a narrow page.
        tab.add_block(
            "Устройство записи",
            "Микрофон для распознавания. «Обновить» перечитывает список после hot-plug.",
            device_body,
        )
        tab.add_widget(self._device_notice)

        self._gain = SliderField(
            tab.theme, minimum=10, maximum=1000, value=100, unit="%", label="Усиление"
        )
        tab.bind_scaled_slider(
            self._gain, "voice.audio_input.gain", "Усиление сигнала", factor=_GAIN_FACTOR
        )
        tab.add_card(
            "Усиление", "Программное усиление входного сигнала. 100 % — без изменения.", self._gain
        )

        self._meter = LevelMeter(tab.theme)
        self._meter.setProperty("transparent", True)
        tab.add_block(
            "Уровень сигнала",
            "Полоса — текущий уровень, вертикальная черта — порог речи. "
            "Обновляется, только пока вкладка открыта.",
            self._meter,
        )

        self._vad = SliderField(
            tab.theme, minimum=0, maximum=100, value=50, unit="%", label="Порог речи"
        )
        self._vad.value_changed.connect(self._on_vad_changed)
        tab.bind_scaled_slider(
            self._vad, "voice.audio_input.vad_threshold", "Порог VAD", factor=_VAD_FACTOR
        )
        tab.add_card(
            "Порог речи (VAD)",
            "Ниже порога сигнал считается тишиной. Черта на шкале выше показывает порог.",
            self._vad,
        )

        self._denoise_combo = ThemedComboBox()
        for value, label in combo_options(AudioInputConfig, "denoise", _DENOISE):
            self._denoise_combo.addItem(label, value)
        tab.bind_combo(self._denoise_combo, "voice.audio_input.denoise", "Шумоподавление")
        tab.add_card(
            "Шумоподавление",
            "RNNoise добавляет небольшую задержку, но заметно чище в шумной комнате.",
            self._denoise_combo,
        )

        self._build_calibrate_card()
        tab.add_restart_bar(RestartScope.AUDIO)

    def _build_calibrate_card(self) -> None:
        tab = self._tab
        body = tab.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))
        self._calibrate_button = QPushButton("Калибровать")
        self._calibrate_button.setProperty("kind", "primary")
        self._calibrate_button.clicked.connect(self._calibrate)
        layout.addWidget(self._calibrate_button)
        self._calibrate_status = QLabel("")
        self._calibrate_status.setProperty("role", "muted")
        self._calibrate_status.setWordWrap(True)
        layout.addWidget(self._calibrate_status)
        tab.add_block(
            "Калибровка микрофона",
            "Измерит шум комнаты и вашу речь, затем предложит усиление, порог и шумоподавление.",
            body,
        )

    # -- devices ------------------------------------------------------------

    def _enumerator(self) -> DeviceEnumerator | None:
        """The device source, built from PortAudio on first use if none was given."""
        if self._enumerator_cache is not None or self._enumerator_tried:
            return self._enumerator_cache
        self._enumerator_tried = True
        try:
            from ayris.audio.devices import SoundDeviceBackend

            self._enumerator_cache = SoundDeviceBackend()
        except Exception:
            _log.exception("не удалось создать источник аудиоустройств")
            self._enumerator_cache = None
        self._tab.services.devices = self._enumerator_cache
        return self._enumerator_cache

    def _populate_devices(self) -> None:
        current = self._tab.manager.settings.voice.audio_input.device
        with QSignalBlocker(self._device_combo):
            self._device_combo.clear()
            self._device_combo.addItem("Системное по умолчанию", _DEFAULT_DEVICE)
            seen = {_DEFAULT_DEVICE}
            for device in self._devices():
                self._device_combo.addItem(device.label, device.id)
                seen.add(device.id)
            missing = bool(current) and current not in seen
            if missing:
                self._device_combo.addItem(f"{current} (недоступно)", current)
            index = self._device_combo.findData(current)
            if index >= 0:
                self._device_combo.setCurrentIndex(index)
        self._update_device_notice(missing=missing)

    def _devices(self) -> tuple[AudioDevice, ...]:
        enumerator = self._enumerator()
        if enumerator is None:
            return ()
        try:
            return list_devices(enumerator, DeviceDirection.INPUT)
        except AudioError:
            _log.warning("не удалось перечислить устройства записи")
            return ()

    def _rescan_devices(self) -> None:
        enumerator = self._enumerator()
        if enumerator is not None:
            try:
                enumerator.refresh()
            except Exception:
                _log.exception("не удалось перечитать устройства")
        self._populate_devices()

    def _update_device_notice(self, *, missing: bool) -> None:
        if missing:
            self._device_notice.setText(
                "Выбранное устройство сейчас недоступно — подключите его или выберите другое."
            )
        else:
            self._device_notice.setText("")

    # -- level meter --------------------------------------------------------

    def _subscribe(self) -> None:
        bus = self._tab.event_bus
        if bus is None:
            return
        unsubscribe = bus.subscribe(AudioLevelChanged, self._relay_level)
        self._tab.add_teardown(unsubscribe)

    def _relay_level(self, event: AudioLevelChanged) -> None:
        # Worker thread → GUI thread via a queued signal.
        self._signals.level.emit(event.rms)

    def _on_level(self, rms: float) -> None:
        if not self._tab.isVisible():
            return  # a hidden tab does not need to redraw the meter
        self._meter.set_level(rms)

    def _on_vad_changed(self, value: int) -> None:
        self._meter.set_threshold(value / _VAD_FACTOR)

    # -- calibration --------------------------------------------------------

    def _calibrate(self) -> None:
        factory = self._tab.services.audio_source
        if factory is None:
            return
        self._calibrate_button.setEnabled(False)
        self._calibrate_status.setText(
            "Помолчите 3 секунды, затем произнесите: «айрис открой браузер»…"
        )

        def work() -> CalibrationReport:
            from ayris.audio.calibration import run_calibration

            settings = self._tab.manager.settings.voice.audio_input
            return run_calibration(factory(), base_gain=settings.gain)

        self._runner.run(work)

    def _on_calibrated(self, result: object) -> None:
        self._calibrate_button.setEnabled(self._tab.services.audio_source is not None)
        from ayris.audio.calibration import CalibrationReport

        if not isinstance(result, CalibrationReport):
            return
        self._calibrate_status.setText(result.summary)
        dialog = ConfirmDialog(
            "Применить рекомендованные значения?",
            result.summary + "\n\n" + "\n".join(result.messages),
            self._tab.theme,
            confirm_text="Применить",
            parent=self._tab,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.apply_recommendation(result.recommended)

    def _on_calibration_failed(self, message: str) -> None:
        self._calibrate_button.setEnabled(self._tab.services.audio_source is not None)
        self._calibrate_status.setText(f"Калибровка не удалась: {message}")

    def apply_recommendation(self, recommendation: Recommendation) -> None:
        """Write calibration results into the audio-input fields."""
        self._tab.manager.apply(
            {
                "voice.audio_input.gain": recommendation.gain,
                "voice.audio_input.vad_threshold": recommendation.vad_threshold,
                "voice.audio_input.noise_floor_db": recommendation.noise_floor_db,
                "voice.audio_input.silence_ms": recommendation.silence_ms,
                "voice.audio_input.denoise": recommendation.denoise.value,
            }
        )

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        self._populate_devices()
        self._meter.set_threshold(self._tab.manager.settings.voice.audio_input.vad_threshold)
        can_calibrate = self._tab.services.audio_source is not None
        self._calibrate_button.setEnabled(can_calibrate)
        self._calibrate_button.setToolTip(
            "" if can_calibrate else "Калибровка станет доступна после запуска захвата звука"
        )
