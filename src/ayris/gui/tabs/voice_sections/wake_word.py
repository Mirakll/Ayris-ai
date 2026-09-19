"""«Слово активации (Wake Word)»: engine, editable variants, debounce, test mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import RestartScope, WakeConfig
from ayris.core.errors import ConfigError
from ayris.core.events import WakeWordDetected
from ayris.gui.tabs.voice import combo_options
from ayris.gui.widgets import SliderField, ToggleSwitch
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.gui.tabs.voice import VoiceTab

__all__ = ["WakeWordSection"]

_log = get_logger(__name__)

_ENGINES = {
    "openwakeword": "openWakeWord",
    "porcupine": "Porcupine",
    "vosk": "Vosk KWS",
}
_MIC_MODES = {
    "always": "Всегда слушать",
    "ptt": "По клавише (Push-to-Talk)",
    "hybrid": "И то и другое",
}

#: Sensitivity slider works in whole percent; the config field is 0.0-1.0.
_SENS_FACTOR = 100


class _PhraseRow(QWidget):
    """One wake-word variant: its text, a sensitivity slider, a switch, a remove button."""

    changed = Signal()
    removed = Signal(str)

    def __init__(self, phrase: str, sensitivity: float, enabled: bool, tab: VoiceTab) -> None:
        super().__init__(parent=tab)
        self.phrase = phrase
        self.setProperty("transparent", True)
        theme = tab.theme
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))

        name = QLabel(phrase)
        name.setMinimumWidth(theme.metric("spacing_2xl") * 3)
        layout.addWidget(name)

        self.sensitivity = SliderField(
            theme,
            minimum=0,
            maximum=100,
            value=round(sensitivity * _SENS_FACTOR),
            unit="%",
            label=f"Чувствительность «{phrase}»",
        )
        self.sensitivity.value_changed.connect(lambda _value: self.changed.emit())
        layout.addWidget(self.sensitivity, 1)

        self.enabled = ToggleSwitch(theme, checked=enabled, label=f"Вариант «{phrase}»")
        self.enabled.toggled.connect(lambda _checked: self.changed.emit())
        layout.addWidget(self.enabled)

        remove = QPushButton("Удалить")
        remove.clicked.connect(lambda: self.removed.emit(self.phrase))
        layout.addWidget(remove)

    def value(self) -> tuple[str, float, bool]:
        return (self.phrase, self.sensitivity.value() / _SENS_FACTOR, self.enabled.isChecked())


class _WakeSignals(QObject):
    """Marshals wake detections from the worker thread onto the GUI thread."""

    detected = Signal(str, float)


class WakeWordSection:
    """Builds the wake-word part of the «Голос» tab."""

    def __init__(self, tab: VoiceTab) -> None:
        self._tab = tab
        self._last_written: tuple[tuple[str, float, bool], ...] = ()
        self._rows: list[_PhraseRow] = []
        self._test_active = False
        self._detections = 0
        self._false_positives = 0

        self._debounce = QTimer(tab)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(350)
        self._debounce.timeout.connect(self._flush_live)

        self._signals = _WakeSignals(tab)
        self._signals.detected.connect(self._on_detected)

        self._build()
        tab.register_refresh(self.refresh)
        self._subscribe()

    def _build(self) -> None:
        tab = self._tab
        tab.add_header("Слово активации (Wake Word)")

        self._enabled = ToggleSwitch(tab.theme, label="Реагировать на слово активации")
        tab.bind_toggle(self._enabled, "voice.wake.enabled", "Слово активации включено")
        tab.add_card(
            "Активация голосом",
            "Реагировать на «Айрис» и его варианты без нажатия клавиш.",
            self._enabled,
        )

        self._engine_combo = QComboBox()
        for value, label in combo_options(WakeConfig, "engine", _ENGINES):
            self._engine_combo.addItem(label, value)
        tab.bind_combo(self._engine_combo, "voice.wake.engine", "Движок слова активации")
        tab.add_card("Движок", "Чем распознавать слово активации.", self._engine_combo)

        self._mode_combo = QComboBox()
        for value, label in combo_options(WakeConfig, "mic_mode", _MIC_MODES):
            self._mode_combo.addItem(label, value)
        tab.bind_combo(self._mode_combo, "voice.wake.mic_mode", "Режим микрофона")
        tab.add_card(
            "Режим микрофона",
            "Всегда слушать, только по клавише или оба способа сразу.",
            self._mode_combo,
        )

        self._build_phrase_editor()

        self._debounce_field = SliderField(
            tab.theme,
            minimum=200,
            maximum=10000,
            value=1500,
            unit="мс",
            label="Пауза после срабатывания",
        )
        tab.bind_int_slider(self._debounce_field, "voice.wake.debounce_ms", "Дебаунс активации")
        tab.add_card(
            "Пауза после срабатывания",
            "Не срабатывать повторно в течение этого времени.",
            self._debounce_field,
        )

        self._build_test_card()
        tab.add_restart_bar(RestartScope.WAKE)

    def _build_phrase_editor(self) -> None:
        tab = self._tab
        body = tab.panel()
        self._editor_layout = QVBoxLayout(body)
        self._editor_layout.setContentsMargins(0, 0, 0, 0)
        self._editor_layout.setSpacing(tab.theme.metric("spacing_sm"))

        self._rows_container = tab.panel()
        self._rows_layout = QVBoxLayout(self._rows_container)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(tab.theme.metric("spacing_xs"))
        self._editor_layout.addWidget(self._rows_container)

        add_row = QHBoxLayout()
        add_row.setSpacing(tab.theme.metric("spacing_sm"))
        self._new_phrase = QLineEdit()
        self._new_phrase.setPlaceholderText("Новый вариант, например «слушай айрис»")
        self._new_phrase.returnPressed.connect(self._add_phrase)
        add_row.addWidget(self._new_phrase, 1)
        add_button = QPushButton("Добавить")
        add_button.setProperty("kind", "primary")
        add_button.clicked.connect(self._add_phrase)
        add_row.addWidget(add_button)
        self._editor_layout.addLayout(add_row)

        self._editor_status = QLabel("")
        self._editor_status.setProperty("role", "muted")
        self._editor_status.setWordWrap(True)
        self._editor_layout.addWidget(self._editor_status)

        tab.add_block(
            "Варианты слова активации",
            "Чем больше вариантов, тем терпимее к произношению; чувствительность выше — "
            "чаще ложные срабатывания.",
            body,
        )
        self._rebuild_rows()

    def _build_test_card(self) -> None:
        tab = self._tab
        body = tab.panel()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(tab.theme.metric("spacing_sm"))

        top = QHBoxLayout()
        top.setSpacing(tab.theme.metric("spacing_sm"))
        self._test_toggle = ToggleSwitch(tab.theme, label="Режим теста активации")
        self._test_toggle.toggled.connect(self._set_test_active)
        top.addWidget(self._test_toggle)
        toggle_label = QLabel("Режим теста")
        toggle_label.setProperty("role", "secondary")
        top.addWidget(toggle_label)
        top.addStretch(1)
        layout.addLayout(top)

        buttons = QHBoxLayout()
        buttons.setSpacing(tab.theme.metric("spacing_sm"))
        self._reset_button = QPushButton("Сбросить счётчик")
        self._reset_button.clicked.connect(self._reset_counts)
        buttons.addWidget(self._reset_button)
        self._false_button = QPushButton("Отметить ложное")
        self._false_button.clicked.connect(self._mark_false)
        buttons.addWidget(self._false_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._test_status = QLabel()
        self._test_status.setProperty("role", "secondary")
        self._test_status.setWordWrap(True)
        layout.addWidget(self._test_status)

        tab.add_block(
            "Проверка слова активации",
            "Включите режим и произносите слово: здесь видно каждое срабатывание и уверенность.",
            body,
        )
        self._update_test_status()

    # -- phrases ------------------------------------------------------------

    def _rebuild_rows(self) -> None:
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._rows = []
        phrases = self._tab.manager.settings.voice.wake.phrases
        for phrase in phrases:
            row = _PhraseRow(phrase.phrase, phrase.sensitivity, phrase.enabled, self._tab)
            row.changed.connect(self._debounce.start)
            row.removed.connect(self._remove_phrase)
            self._rows_layout.addWidget(row)
            self._rows.append(row)
        self._last_written = tuple(row.value() for row in self._rows)

    def _current_snapshot(self) -> tuple[tuple[str, float, bool], ...]:
        return tuple(row.value() for row in self._rows)

    def _apply(self, snapshot: tuple[tuple[str, float, bool], ...], *, rebuild: bool) -> bool:
        payload = [
            {"phrase": phrase, "sensitivity": round(sensitivity, 3), "enabled": enabled}
            for phrase, sensitivity, enabled in snapshot
        ]
        try:
            self._tab.manager.apply({"voice.wake.phrases": payload})
        except ConfigError as exc:
            self._editor_status.setText(exc.user_message)
            return False
        # Read back the validated, de-duplicated tuple so refresh() can tell a
        # self-triggered change from a real external edit.
        self._last_written = tuple(
            (phrase.phrase, phrase.sensitivity, phrase.enabled)
            for phrase in self._tab.manager.settings.voice.wake.phrases
        )
        self._editor_status.setText("")
        if rebuild:
            self._rebuild_rows()
        return True

    def _flush_live(self) -> None:
        # Sensitivity/enabled edits: no rebuild, so a slider drag is not reset.
        self._apply(self._current_snapshot(), rebuild=False)

    def _add_phrase(self) -> None:
        text = self._new_phrase.text().strip()
        if len(text) < 2:
            self._editor_status.setText("Слово активации должно быть не короче двух символов.")
            return
        snapshot = (*self._current_snapshot(), (text, 0.5, True))
        if self._apply(snapshot, rebuild=True):
            self._new_phrase.clear()

    def _remove_phrase(self, phrase: str) -> None:
        snapshot = tuple(item for item in self._current_snapshot() if item[0] != phrase)
        self._apply(snapshot, rebuild=True)

    # -- test mode ----------------------------------------------------------

    def _subscribe(self) -> None:
        bus = self._tab.event_bus
        if bus is None:
            return
        unsubscribe = bus.subscribe(WakeWordDetected, self._relay_detection)
        self._tab.add_teardown(unsubscribe)

    def _relay_detection(self, event: WakeWordDetected) -> None:
        # Runs on the worker thread; hand off to the GUI thread via a queued signal.
        self._signals.detected.emit(event.phrase, event.confidence)

    def _on_detected(self, phrase: str, confidence: float) -> None:
        if not self._test_active or not self._tab.isVisible():
            return
        self._detections += 1
        self._update_test_status(last=f"«{phrase}» ({confidence:.0%})")

    def _set_test_active(self, active: bool) -> None:
        self._test_active = active
        self._update_test_status()

    def _reset_counts(self) -> None:
        self._detections = 0
        self._false_positives = 0
        self._update_test_status()

    def _mark_false(self) -> None:
        self._false_positives += 1
        self._update_test_status()

    def _update_test_status(self, *, last: str = "") -> None:
        state = "включён" if self._test_active else "выключен"
        tail = f" Последнее: {last}." if last else ""
        self._test_status.setText(
            f"Режим теста {state}. Срабатываний: {self._detections}, "
            f"отмечено ложных: {self._false_positives}.{tail}"
        )

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        current = tuple(
            (phrase.phrase, phrase.sensitivity, phrase.enabled)
            for phrase in self._tab.manager.settings.voice.wake.phrases
        )
        if current != self._last_written:
            self._rebuild_rows()
