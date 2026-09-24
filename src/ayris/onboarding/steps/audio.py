"""Шаг «Микрофон»: выбор устройства, усиление, живой уровень и калибровка.

Шаг ничего не знает ни о PortAudio, ни о калибровке напрямую — он получает
узкий :class:`AudioProbe` (перечислить устройства и, если есть источник звука,
снять калибровку) и шину для живого уровня. Это повторяет вкладку «Голос»
(:mod:`ayris.gui.tabs.voice_sections.audio_input`), но без её хелперов вкладки,
и позволяет тестам подставить фейковый пробник без звуковой карты.

Микрофона может не быть вовсе: тогда список пуст, показывается предупреждение, а
шаг остаётся пропускаемым — мастер не должен упираться в железо.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from PySide6.QtCore import QObject, QSignalBlocker, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.core.config import ConfigManager
from ayris.core.events import AudioLevelChanged, EventBus
from ayris.gui.tabs.voice import AsyncRunner
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import LevelMeter, SliderField, ThemedComboBox
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.audio.calibration import CalibrationReport

__all__ = ["AudioProbe", "AudioStep"]

_log = get_logger(__name__)

_GAIN_FACTOR = 100
#: Что config хранит для «системного по умолчанию».
_DEFAULT_DEVICE = ""


class AudioProbe(Protocol):
    """Узкий контракт железа, чтобы шаг не зависел от PortAudio и калибровки."""

    def input_devices(self) -> list[tuple[str, str]]:
        """Пары ``(id, подпись)`` устройств записи; пусто — микрофона нет."""

    def can_calibrate(self) -> bool:
        """Доступна ли калибровка (нужен живой источник звука)."""

    def calibrate(self, *, base_gain: float) -> CalibrationReport:
        """Снять шум и фразу, вернуть отчёт. Блокирующая — звать в потоке."""


class _AudioSignals(QObject):
    """Переносит уровень микрофона с потока воркера на поток GUI."""

    level = Signal(float)


class AudioStep(WizardStep):
    """Настройка микрофона: устройство, усиление, уровень, калибровка."""

    def __init__(
        self,
        theme: ThemeManager,
        config: ConfigManager,
        probe: AudioProbe | None,
        bus: EventBus | None,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "audio"
        self.title = "Микрофон"
        self._config = config
        self._probe = probe
        self._bus = bus
        self._unsub: Callable[[], None] | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Микрофон"))
        layout.addWidget(
            caption(
                "Выберите устройство записи и проверьте уровень. Калибровка измерит "
                "шум комнаты и вашу речь и подберёт усиление и порог."
            )
        )

        self._device_combo = ThemedComboBox()
        self._populate_devices()
        layout.addWidget(self._device_combo)

        self._device_notice = QLabel("")
        self._device_notice.setProperty("role", "muted")
        self._device_notice.setWordWrap(True)
        layout.addWidget(self._device_notice)

        current_gain = config.settings.voice.audio_input.gain
        self._gain = SliderField(
            theme,
            minimum=10,
            maximum=1000,
            value=int(round(current_gain * _GAIN_FACTOR)),
            unit="%",
            label="Усиление",
        )
        layout.addWidget(self._gain)

        self._meter = LevelMeter(theme)
        self._meter.setProperty("transparent", True)
        self._meter.set_threshold(config.settings.voice.audio_input.vad_threshold)
        layout.addWidget(self._meter)
        layout.addWidget(
            caption("Полоса — текущий уровень. Обновляется, только пока идёт захват звука.")
        )

        row = QHBoxLayout()
        self._calibrate_button = QPushButton("Калибровать")
        self._calibrate_button.setProperty("kind", "primary")
        self._calibrate_button.clicked.connect(self._calibrate)
        row.addWidget(self._calibrate_button)
        row.addStretch(1)
        layout.addLayout(row)

        self._calibrate_status = QLabel("")
        self._calibrate_status.setProperty("role", "muted")
        self._calibrate_status.setWordWrap(True)
        layout.addWidget(self._calibrate_status)
        layout.addStretch(1)

        self._signals = _AudioSignals(self)
        self._signals.level.connect(self._on_level)
        self._runner = AsyncRunner(self)
        self._runner.finished.connect(self._on_calibrated)
        self._runner.failed.connect(self._on_calibration_failed)
        self._sync_calibrate_enabled()

    # -- устройства --------------------------------------------------------

    def _populate_devices(self) -> None:
        current = self._config.settings.voice.audio_input.device
        with QSignalBlocker(self._device_combo):
            self._device_combo.clear()
            self._device_combo.addItem("Системное по умолчанию", _DEFAULT_DEVICE)
            seen = {_DEFAULT_DEVICE}
            for device_id, label in self._devices():
                self._device_combo.addItem(label, device_id)
                seen.add(device_id)
            missing = bool(current) and current not in seen
            if missing:
                self._device_combo.addItem(f"{current} (недоступно)", current)
            index = self._device_combo.findData(current)
            if index >= 0:
                self._device_combo.setCurrentIndex(index)

    def _devices(self) -> list[tuple[str, str]]:
        if self._probe is None:
            return []
        try:
            return self._probe.input_devices()
        except Exception:
            _log.exception("не удалось перечислить устройства записи")
            return []

    def _sync_calibrate_enabled(self) -> None:
        can = self._probe is not None and self._probe.can_calibrate()
        has_mic = self._probe is not None
        self._calibrate_button.setEnabled(can)
        if not has_mic:
            self._device_notice.setText(
                "Микрофон не найден. Можно продолжить без него и настроить позже "
                "на вкладке «Голос»."
            )
        elif not can:
            self._device_notice.setText("Калибровка станет доступна после запуска захвата звука.")
        else:
            self._device_notice.setText("")

    # -- уровень -----------------------------------------------------------

    def _on_level(self, rms: float) -> None:
        if self.isVisible():
            self._meter.set_level(rms)

    def _relay_level(self, event: AudioLevelChanged) -> None:
        self._signals.level.emit(event.rms)

    # -- калибровка --------------------------------------------------------

    def _calibrate(self) -> None:
        if self._probe is None or not self._probe.can_calibrate():
            return
        self._calibrate_button.setEnabled(False)
        self._calibrate_status.setText(
            "Помолчите 3 секунды, затем произнесите: «айрис открой браузер»…"
        )
        probe = self._probe
        base_gain = self._config.settings.voice.audio_input.gain

        def work() -> CalibrationReport:
            return probe.calibrate(base_gain=base_gain)

        self._runner.run(work)

    def _on_calibrated(self, result: object) -> None:
        self._calibrate_button.setEnabled(True)
        from ayris.audio.calibration import CalibrationReport

        if not isinstance(result, CalibrationReport):
            return
        self._calibrate_status.setText(result.summary)
        rec = result.recommended
        try:
            self._config.apply(
                {
                    "voice.audio_input.gain": rec.gain,
                    "voice.audio_input.vad_threshold": rec.vad_threshold,
                    "voice.audio_input.noise_floor_db": rec.noise_floor_db,
                    "voice.audio_input.silence_ms": rec.silence_ms,
                    "voice.audio_input.denoise": rec.denoise.value,
                }
            )
        except Exception:
            _log.exception("не удалось применить рекомендации калибровки")
            return
        self._gain.setValue(int(round(rec.gain * _GAIN_FACTOR)))
        self._meter.set_threshold(rec.vad_threshold)

    def _on_calibration_failed(self, message: str) -> None:
        self._calibrate_button.setEnabled(True)
        self._calibrate_status.setText(f"Калибровка не удалась: {message}")

    # -- контракт шага -----------------------------------------------------

    def activate(self) -> None:
        self._populate_devices()
        self._sync_calibrate_enabled()
        if self._bus is not None and self._unsub is None:
            self._unsub = self._bus.subscribe(AudioLevelChanged, self._relay_level)

    def deactivate(self) -> None:
        self._drop_subscription()

    def apply(self) -> None:
        device = self._device_combo.currentData()
        self._config.apply(
            {
                "voice.audio_input.device": device if isinstance(device, str) else _DEFAULT_DEVICE,
                "voice.audio_input.gain": self._gain.value() / _GAIN_FACTOR,
            }
        )

    def teardown(self) -> None:
        self._drop_subscription()

    def _drop_subscription(self) -> None:
        if self._unsub is not None:
            try:
                self._unsub()
            except Exception:
                _log.exception("не удалось отписаться от уровня микрофона")
            self._unsub = None
