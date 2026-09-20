"""What makes a command fire: voice phrases, hotkeys, events and schedules.

Each trigger of ``model.triggers`` is one editable card, added by type. The four types
are the four :class:`~ayris.actions.macros.schema.TriggerModel` variants:

* **voice** — a phrase with ``fuzzy`` and ``regex`` toggles, a fuzzy threshold and a
  priority. The regex is compiled as the user types; a bad pattern is flagged and its
  named groups are shown as the slots they become. The named groups of a template
  phrase (``{name}``) are shown the same way.
* **hotkey** — a combination captured through the task-37
  :class:`~ayris.utils.hotkey_manager.HotkeyManager`, injected as a capture callable so
  the widget never touches the keyboard hook itself and a test can supply a fixed combo.
* **event** — a system event name with a filter, filtered from an injected list.
* **schedule** — a one-off moment or a cron expression, with a Russian reading of the
  schedule under the field.

Conflicts are shown as they are typed: a voice phrase or a hotkey combo already used by
another command lights the card and names the other command. The set of taken
combinations is injected (the store computes it), so this widget stays a view.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.schema import (
    CommandModel,
    EventTrigger,
    HotkeyTrigger,
    TimerTrigger,
    TriggerModel,
    VoiceTrigger,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.slider_field import SliderField
from ayris.gui.widgets.toggle import ToggleSwitch

__all__ = ["HotkeyCapture", "TriggerList", "describe_cron"]

#: A callable that captures one hotkey combo (canonical string) or returns ``None``.
HotkeyCapture = Callable[[], str | None]

_MONTHS = ("янв", "фев", "мар", "апр", "мая", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
_WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


def describe_cron(expression: str) -> str:
    """A short Russian reading of a five-field cron, or the raw text if it is odd.

    Covers the shapes the schedule editor produces — every N minutes, daily at a time,
    weekly on a weekday — and falls back to the expression itself for anything hand
    written, so an unusual cron is shown, not mis-described.
    """
    parts = expression.split()
    if len(parts) != 5:
        return expression
    minute, hour, dom, month, dow = parts
    if minute.startswith("*/") and (hour, dom, month, dow) == ("*", "*", "*", "*"):
        return f"каждые {minute[2:]} мин"
    if minute.isdigit() and hour.isdigit() and (dom, month) == ("*", "*"):
        clock = f"{int(hour):02d}:{int(minute):02d}"
        if dow == "*":
            return f"ежедневно в {clock}"
        if dow.isdigit():
            index = int(dow) % 7
            return f"по {_WEEKDAYS[(index - 1) % 7] if index else 'вс'} в {clock}"
    return expression


class _TriggerCard(QFrame):
    """Base card: a heading with a remove button and a body the type fills."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, theme: ThemeManager, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("card", True)
        self._theme = theme
        self._loading = False
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(*(theme.metric("spacing_md"),) * 4)
        self._outer.setSpacing(theme.metric("spacing_sm"))

        head = QHBoxLayout()
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        head.addWidget(heading)
        head.addStretch(1)
        remove = QPushButton("Удалить")
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        head.addWidget(remove)
        self._outer.addLayout(head)

        self._conflict = QLabel("")
        self._conflict.setProperty("role", "muted")
        self._conflict.setProperty("badge", "warning")
        self._conflict.setWordWrap(True)
        self._conflict.hide()
        self._outer.addWidget(self._conflict)

    def add_body(self, widget: QWidget) -> None:
        self._outer.addWidget(widget)

    def add_row(self, *widgets: QWidget) -> None:
        row = QHBoxLayout()
        row.setSpacing(self._theme.metric("spacing_sm"))
        for widget in widgets:
            row.addWidget(widget)
        row.addStretch(1)
        self._outer.addLayout(row)

    def show_conflict(self, names: Sequence[str]) -> None:
        if names:
            listed = ", ".join(names)
            self._conflict.setText(f"Уже используется командой: {listed}")
            self._conflict.show()
            self.setProperty("status", "warning")
        else:
            self._conflict.hide()
            self.setProperty("status", "")
        _repolish(self)

    def trigger(self) -> TriggerModel | None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _emit(self, *_args: object) -> None:
        if not self._loading:
            self.changed.emit()


class _VoiceCard(_TriggerCard):
    def __init__(self, theme: ThemeManager, trigger: VoiceTrigger) -> None:
        super().__init__(theme, "Голосовая фраза")
        self._phrase = QLineEdit(trigger.phrase)
        self._phrase.setPlaceholderText("Фраза или регэксп, например: включи {device}")
        self._phrase.textChanged.connect(self._on_phrase)
        self.add_body(self._phrase)

        self._slots = QLabel("")
        self._slots.setProperty("role", "muted")
        self._slots.setWordWrap(True)
        self.add_body(self._slots)

        self._fuzzy = ToggleSwitch(theme, label="Нечёткое совпадение", checked=trigger.fuzzy)
        self._fuzzy.toggled.connect(self._emit)
        self._regex = ToggleSwitch(theme, label="Регулярное выражение", checked=trigger.regex)
        self._regex.toggled.connect(self._on_phrase)
        self.add_row(self._fuzzy, self._regex)

        self._threshold = SliderField(
            theme,
            minimum=0,
            maximum=100,
            value=int((trigger.fuzzy_threshold or 0.8) * 100),
            unit="%",
            label="Порог нечёткости",
        )
        self._threshold.value_changed.connect(self._emit)
        self._priority = QSpinBox()
        self._priority.setRange(-1000, 1000)
        self._priority.setValue(trigger.priority)
        self._priority.setPrefix("приоритет ")
        self._priority.valueChanged.connect(self._emit)
        self.add_row(self._threshold, self._priority)
        self._loading = False
        self._refresh_slots()

    def _on_phrase(self, *_args: object) -> None:
        self._refresh_slots()
        self._emit()

    def _refresh_slots(self) -> None:
        phrase = self._phrase.text()
        if self._regex.isChecked():
            try:
                compiled = re.compile(phrase)
            except re.error as exc:
                self._slots.setText(f"Ошибка регэкспа: {exc}")
                self._slots.setProperty("badge", "error")
                _repolish(self._slots)
                return
            groups = list(compiled.groupindex)
            text = f"Слоты: {', '.join(groups)}" if groups else "Именованных групп нет"
        else:
            names = re.findall(r"\{(\w+)\}", phrase)
            text = f"Слоты: {', '.join(names)}" if names else "Слотов нет"
        self._slots.setText(text)
        self._slots.setProperty("badge", "muted")
        _repolish(self._slots)

    def trigger(self) -> TriggerModel | None:
        phrase = self._phrase.text().strip()
        if not phrase:
            return None
        try:
            return VoiceTrigger(
                phrase=phrase,
                fuzzy=self._fuzzy.isChecked(),
                regex=self._regex.isChecked(),
                fuzzy_threshold=round(self._threshold.value() / 100, 2),
                priority=self._priority.value(),
            )
        except ValueError:
            return None


class _HotkeyCard(_TriggerCard):
    def __init__(
        self, theme: ThemeManager, trigger: HotkeyTrigger, capture: HotkeyCapture | None
    ) -> None:
        super().__init__(theme, "Горячая клавиша")
        self._combo = trigger.combo
        self._capture = capture
        self._label = QLineEdit(trigger.combo)
        self._label.setReadOnly(True)
        self._label.setPlaceholderText("Комбинация не задана")
        self._button = QPushButton("Записать")
        self._button.setEnabled(capture is not None)
        self._button.clicked.connect(self._on_capture)
        self.add_row(self._label, self._button)

    def _on_capture(self) -> None:
        if self._capture is None:
            return
        combo = self._capture()
        if combo:
            self._combo = combo
            self._label.setText(combo)
            self._emit()

    def trigger(self) -> TriggerModel | None:
        if not self._combo:
            return None
        try:
            return HotkeyTrigger(combo=self._combo)
        except ValueError:
            return None


class _EventCard(_TriggerCard):
    def __init__(self, theme: ThemeManager, trigger: EventTrigger, events: Sequence[str]) -> None:
        super().__init__(theme, "Системное событие")
        self._combo = ThemedComboBox()
        self._combo.setEditable(True)
        for name in events:
            self._combo.addItem(name)
        if trigger.event_name:
            self._combo.setCurrentText(trigger.event_name)
        self._combo.currentTextChanged.connect(self._emit)
        self.add_body(self._combo)

    def trigger(self) -> TriggerModel | None:
        name = self._combo.currentText().strip()
        if not name:
            return None
        try:
            return EventTrigger(event_name=name, filter_json={})
        except ValueError:
            return None


class _ScheduleCard(_TriggerCard):
    def __init__(self, theme: ThemeManager, trigger: TimerTrigger) -> None:
        super().__init__(theme, "Расписание")
        self._mode = ThemedComboBox()
        self._mode.addItem("По расписанию (cron)", "cron")
        self._mode.addItem("Один раз", "once")
        self._mode.setCurrentIndex(1 if trigger.cron is None and trigger.fire_at else 0)
        self._mode.currentIndexChanged.connect(self._on_mode)
        self.add_body(self._mode)

        self._value = QLineEdit()
        self._value.textChanged.connect(self._on_value)
        self.add_body(self._value)

        self._reading = QLabel("")
        self._reading.setProperty("role", "muted")
        self.add_body(self._reading)

        if trigger.cron is not None:
            self._value.setText(trigger.cron)
        elif trigger.fire_at is not None:
            self._value.setText(trigger.fire_at.strftime("%Y-%m-%d %H:%M"))
        self._loading = False
        self._on_mode()

    def _on_mode(self, *_args: object) -> None:
        is_cron = self._mode.currentData() == "cron"
        self._value.setPlaceholderText(
            "минута час день месяц день-недели, например 0 9 * * 1-5"
            if is_cron
            else "ГГГГ-ММ-ДД ЧЧ:ММ"
        )
        self._on_value()

    def _on_value(self, *_args: object) -> None:
        if self._mode.currentData() == "cron":
            self._reading.setText(describe_cron(self._value.text().strip()))
        else:
            self._reading.setText("")
        self._emit()

    def trigger(self) -> TriggerModel | None:
        text = self._value.text().strip()
        if not text:
            return None
        try:
            if self._mode.currentData() == "cron":
                return TimerTrigger(cron=text)
            return TimerTrigger(fire_at=datetime.strptime(text, "%Y-%m-%d %H:%M").astimezone())
        except ValueError:
            return None


class TriggerList(QWidget):
    """Editable list of a command's triggers, writing ``model.triggers`` in place."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        hotkey_capture: HotkeyCapture | None = None,
        event_names: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._hotkey_capture = hotkey_capture
        self._event_names = tuple(event_names)
        self._model: CommandModel | None = None
        self._conflicts: Mapping[tuple[str, str], Sequence[str]] = {}
        self._cards: list[_TriggerCard] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_md"))

        self._cards_host = QWidget()
        self._cards_host.setProperty("transparent", True)
        self._cards_layout = QVBoxLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(theme.metric("spacing_md"))
        outer.addWidget(self._cards_host)

        add_bar = QHBoxLayout()
        add_bar.addWidget(QLabel("Добавить триггер:"))
        for label, factory in (
            ("Фраза", self._new_voice),
            ("Клавиша", self._new_hotkey),
            ("Событие", self._new_event),
            ("Расписание", self._new_schedule),
        ):
            button = QPushButton(label)
            button.clicked.connect(factory)
            add_bar.addWidget(button)
        add_bar.addStretch(1)
        outer.addLayout(add_bar)

    # -- public API ---------------------------------------------------------

    def set_command(self, model: CommandModel) -> None:
        self._model = model
        self._rebuild()

    def set_conflicts(self, conflicts: Mapping[tuple[str, str], Sequence[str]]) -> None:
        """Taken voice phrases / hotkey combos and the commands that own them."""
        self._conflicts = conflicts
        self._paint_conflicts()

    def cards(self) -> Sequence[_TriggerCard]:
        return tuple(self._cards)

    # -- building -----------------------------------------------------------

    def _rebuild(self) -> None:
        self._clear()
        if self._model is None:
            return
        for trigger in self._model.triggers:
            self._add_card(self._card_for(trigger), commit=False)
        self._paint_conflicts()

    def _card_for(self, trigger: TriggerModel) -> _TriggerCard:
        if isinstance(trigger, VoiceTrigger):
            return _VoiceCard(self._theme, trigger)
        if isinstance(trigger, HotkeyTrigger):
            return _HotkeyCard(self._theme, trigger, self._hotkey_capture)
        if isinstance(trigger, EventTrigger):
            return _EventCard(self._theme, trigger, self._event_names)
        return _ScheduleCard(self._theme, trigger)

    def _add_card(self, card: _TriggerCard, *, commit: bool) -> None:
        card.changed.connect(self._commit)
        card.remove_requested.connect(self._remove_card)
        self._cards.append(card)
        self._cards_layout.addWidget(card)
        if commit:
            self._commit()

    def _remove_card(self, card: object) -> None:
        if isinstance(card, _TriggerCard) and card in self._cards:
            self._cards.remove(card)
            card.setParent(None)
            card.deleteLater()
            self._commit()

    def _clear(self) -> None:
        for card in self._cards:
            card.setParent(None)
            card.deleteLater()
        self._cards.clear()

    # -- adding new ---------------------------------------------------------

    def _new_voice(self) -> None:
        self._add_card(_VoiceCard(self._theme, VoiceTrigger(phrase="новая фраза")), commit=True)

    def _new_hotkey(self) -> None:
        self._add_card(
            _HotkeyCard(self._theme, HotkeyTrigger(combo="ctrl+alt+a"), self._hotkey_capture),
            commit=True,
        )

    def _new_event(self) -> None:
        default = self._event_names[0] if self._event_names else "system.event"
        self._add_card(
            _EventCard(
                self._theme, EventTrigger(event_name=default, filter_json={}), self._event_names
            ),
            commit=True,
        )

    def _new_schedule(self) -> None:
        self._add_card(_ScheduleCard(self._theme, TimerTrigger(cron="0 9 * * *")), commit=True)

    # -- writing back -------------------------------------------------------

    def _commit(self) -> None:
        if self._model is None:
            return
        triggers = [card.trigger() for card in self._cards]
        self._model.triggers = [trigger for trigger in triggers if trigger is not None]
        self._paint_conflicts()
        self.changed.emit()

    def _paint_conflicts(self) -> None:
        for card in self._cards:
            trigger = card.trigger()
            key = _conflict_key(trigger)
            card.show_conflict(list(self._conflicts.get(key, ())) if key is not None else [])


def _conflict_key(trigger: TriggerModel | None) -> tuple[str, str] | None:
    """The ``(kind, text)`` identity used to look a trigger up in the conflict map."""
    if isinstance(trigger, VoiceTrigger):
        text = trigger.phrase.casefold().strip()
        return ("voice", text) if text else None
    if isinstance(trigger, HotkeyTrigger):
        combo = trigger.combo.casefold().strip()
        return ("hotkey", combo) if combo else None
    return None


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
