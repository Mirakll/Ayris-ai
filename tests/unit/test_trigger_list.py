"""Редактор триггеров команды (задача 52), offscreen.

Ничего не рисуется и не разглядывается: каждая проверка — про состояние модели
(``model.triggers``) и про сигналы. Чистая функция :func:`describe_cron` покрыта по
веткам; карточки Voice/Hotkey/Event/Schedule проверяются через публичный API
:class:`TriggerList` (``set_command``, кнопки добавления, ``set_conflicts``) и через
их собственные поля. Железо не трогается: захват горячей клавиши — инжектируемый
фейк, а не настоящий хук клавиатуры.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from ayris.actions.macros.schema import (
    CommandModel,
    EventTrigger,
    HotkeyTrigger,
    TimerTrigger,
    VoiceTrigger,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.trigger_list import (
    TriggerList,
    _EventCard,
    _HotkeyCard,
    _ScheduleCard,
    _VoiceCard,
    describe_cron,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    manager = ThemeManager(app)
    manager.apply()
    return manager


class _FakeCapture:
    """Заглушка захвата горячей клавиши: возвращает заранее заданную комбинацию."""

    def __init__(self, result: str | None) -> None:
        self.result = result
        self.calls = 0

    def __call__(self) -> str | None:
        self.calls += 1
        return self.result


def _command(*triggers: object) -> CommandModel:
    return CommandModel(name="Тест", triggers=list(triggers))  # type: ignore[arg-type]


def _add_buttons(widget: TriggerList) -> dict[str, QPushButton]:
    """Кнопки панели «Добавить триггер», пойманные по подписи до появления карточек."""
    return {button.text(): button for button in widget.findChildren(QPushButton)}


def _remove_button(card: object) -> QPushButton:
    buttons = [b for b in card.findChildren(QPushButton) if b.text() == "Удалить"]  # type: ignore[attr-defined]
    return buttons[0]


# ----------------------------------------------------------------------
# describe_cron — чистая функция, все ветки
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("*/5 * * * *", "каждые 5 мин"),
        ("*/15 * * * *", "каждые 15 мин"),
        ("0 9 * * *", "ежедневно в 09:00"),
        ("5 0 * * *", "ежедневно в 00:05"),
        ("30 8 * * 1", "по пн в 08:30"),
        ("0 9 * * 3", "по ср в 09:00"),
        ("0 9 * * 6", "по сб в 09:00"),
        ("0 9 * * 0", "по вс в 09:00"),
        ("0 9 * * 7", "по вс в 09:00"),
    ],
)
def test_describe_cron_known_shapes(expression: str, expected: str) -> None:
    assert describe_cron(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "bad",  # не пять полей
        "a b c d e",  # пять полей, но минута не число и не */N
        "0 9 1 * *",  # с числом дня месяца это не «ежедневно»
        "*/5 9 * * *",  # */N минут, но час не *
        "0 9 * * 1-5",  # день недели — диапазон, не одно число
        "0 9 * 6 *",  # задан месяц
    ],
)
def test_describe_cron_falls_back_to_raw(expression: str) -> None:
    assert describe_cron(expression) == expression


# ----------------------------------------------------------------------
# set_command читает model.triggers и строит карточки по типам
# ----------------------------------------------------------------------


def test_set_command_builds_one_card_per_trigger(app: QApplication, theme: ThemeManager) -> None:
    model = _command(
        VoiceTrigger(phrase="свет"),
        HotkeyTrigger(combo="ctrl+alt+a"),
        EventTrigger(event_name="system.event"),
        TimerTrigger(cron="0 9 * * *"),
    )
    widget = TriggerList(theme)
    widget.set_command(model)
    cards = widget.cards()
    assert [type(card) for card in cards] == [
        _VoiceCard,
        _HotkeyCard,
        _EventCard,
        _ScheduleCard,
    ]


def test_set_command_does_not_emit_changed(app: QApplication, theme: ThemeManager) -> None:
    seen: list[int] = []
    widget = TriggerList(theme)
    widget.changed.connect(lambda: seen.append(1))
    widget.set_command(_command(VoiceTrigger(phrase="свет")))
    # Построение из модели — это не правка: сигнал не летит.
    assert seen == []


def test_set_command_twice_rebuilds_from_scratch(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="один"), VoiceTrigger(phrase="два")))
    assert len(widget.cards()) == 2
    widget.set_command(_command(HotkeyTrigger(combo="ctrl+alt+a")))
    assert [type(card) for card in widget.cards()] == [_HotkeyCard]


# ----------------------------------------------------------------------
# добавление триггеров кнопками пишет в модель и шлёт changed
# ----------------------------------------------------------------------


def test_add_voice_writes_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = _command()
    widget = TriggerList(theme)
    buttons = _add_buttons(widget)
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))
    buttons["Фраза"].click()
    assert len(widget.cards()) == 1
    assert isinstance(model.triggers[0], VoiceTrigger)
    assert seen != []


def test_add_hotkey_uses_default_combo(app: QApplication, theme: ThemeManager) -> None:
    model = _command()
    widget = TriggerList(theme)
    buttons = _add_buttons(widget)
    widget.set_command(model)
    buttons["Клавиша"].click()
    assert isinstance(model.triggers[0], HotkeyTrigger)
    assert model.triggers[0].combo == "ctrl+alt+a"


def test_add_event_default_without_names(app: QApplication, theme: ThemeManager) -> None:
    model = _command()
    widget = TriggerList(theme)
    buttons = _add_buttons(widget)
    widget.set_command(model)
    buttons["Событие"].click()
    assert isinstance(model.triggers[0], EventTrigger)
    assert model.triggers[0].event_name == "system.event"


def test_add_event_uses_first_injected_name(app: QApplication, theme: ThemeManager) -> None:
    model = _command()
    widget = TriggerList(theme, event_names=("app.launched", "system.event"))
    buttons = _add_buttons(widget)
    widget.set_command(model)
    buttons["Событие"].click()
    card = widget.cards()[0]
    assert isinstance(card, _EventCard)
    assert card._combo.currentText() == "app.launched"
    assert card._combo.count() == 2
    assert model.triggers[0].event_name == "app.launched"


def test_add_schedule_writes_cron(app: QApplication, theme: ThemeManager) -> None:
    model = _command()
    widget = TriggerList(theme)
    buttons = _add_buttons(widget)
    widget.set_command(model)
    buttons["Расписание"].click()
    assert isinstance(model.triggers[0], TimerTrigger)
    assert model.triggers[0].cron == "0 9 * * *"


def test_add_before_set_command_is_a_noop_on_model(app: QApplication, theme: ThemeManager) -> None:
    # Без модели _commit просто выходит: карточка строится, но писать некуда.
    seen: list[int] = []
    widget = TriggerList(theme)
    widget.changed.connect(lambda: seen.append(1))
    _add_buttons(widget)["Фраза"].click()
    assert len(widget.cards()) == 1
    assert seen == []


# ----------------------------------------------------------------------
# удаление карточки
# ----------------------------------------------------------------------


def test_remove_card_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = _command(VoiceTrigger(phrase="свет"))
    widget = TriggerList(theme)
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))
    card = widget.cards()[0]
    _remove_button(card).click()
    assert widget.cards() == ()
    assert model.triggers == []
    assert seen != []


def test_remove_card_ignores_foreign_object(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="свет")))
    # Не-карточка и повторное удаление уже удалённой карточки — тихий no-op.
    widget._remove_card("не карточка")
    card = widget.cards()[0]
    widget._remove_card(card)
    widget._remove_card(card)
    assert widget.cards() == ()


# ----------------------------------------------------------------------
# голосовая карточка: слоты, регэксп, правки летят в модель
# ----------------------------------------------------------------------


def test_voice_edit_phrase_writes_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = _command(VoiceTrigger(phrase="старая"))
    widget = TriggerList(theme)
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    card._phrase.setText("привет")
    assert model.triggers[0].phrase == "привет"
    assert seen != []


def test_voice_template_slots_are_listed(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="включи {device}")))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    assert card._slots.text() == "Слоты: device"


def test_voice_without_slots_says_so(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="просто фраза")))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    assert card._slots.text() == "Слотов нет"


def test_voice_regex_named_groups_shown_as_slots(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase=r"(?P<device>\w+) on", regex=True)))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    assert card._slots.text() == "Слоты: device"


def test_voice_regex_without_groups_says_none(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase=r"hello.*", regex=True)))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    assert card._slots.text() == "Именованных групп нет"


def test_voice_bad_regex_flags_error_and_drops_trigger(
    app: QApplication, theme: ThemeManager
) -> None:
    model = _command(VoiceTrigger(phrase="valid", regex=True))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    card._phrase.setText("(")
    # Не компилируется: слот-строка сообщает ошибку, а битый триггер выпадает из модели.
    assert card._slots.text().startswith("Ошибка регэкспа")
    assert model.triggers == []


def test_voice_empty_phrase_gives_no_trigger(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="что-то")))
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    card._phrase.setText("   ")
    assert card.trigger() is None


def test_voice_toggles_and_sliders_write_model(app: QApplication, theme: ThemeManager) -> None:
    model = _command(VoiceTrigger(phrase="тест", fuzzy=True))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    card._fuzzy.setChecked(False)
    card._threshold.setValue(50)
    card._priority.setValue(10)
    trigger = model.triggers[0]
    assert isinstance(trigger, VoiceTrigger)
    assert trigger.fuzzy is False
    assert trigger.fuzzy_threshold == 0.5
    assert trigger.priority == 10


# ----------------------------------------------------------------------
# конфликты подсвечиваются по мере ввода
# ----------------------------------------------------------------------


def test_voice_conflict_shows_owning_command(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="Свет")))
    widget.set_conflicts({("voice", "свет"): ["Другая команда"]})
    card = widget.cards()[0]
    assert not card._conflict.isHidden()
    assert "Другая команда" in card._conflict.text()
    assert card.property("status") == "warning"


def test_conflict_clears_when_map_empty(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="свет")))
    widget.set_conflicts({("voice", "свет"): ["X"]})
    widget.set_conflicts({})
    card = widget.cards()[0]
    assert card._conflict.isHidden()
    assert card.property("status") == ""


def test_conflict_appears_as_phrase_is_typed(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(VoiceTrigger(phrase="новая")))
    widget.set_conflicts({("voice", "свет"): ["Кухня"]})
    card = widget.cards()[0]
    assert isinstance(card, _VoiceCard)
    assert card._conflict.isHidden()
    card._phrase.setText("свет")
    assert not card._conflict.isHidden()
    assert "Кухня" in card._conflict.text()


def test_hotkey_conflict_shows(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(HotkeyTrigger(combo="ctrl+alt+a")))
    widget.set_conflicts({("hotkey", "ctrl+alt+a"): ["Команда Б"]})
    card = widget.cards()[0]
    assert not card._conflict.isHidden()
    assert "Команда Б" in card._conflict.text()


def test_schedule_has_no_conflict_key(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(TimerTrigger(cron="0 9 * * *")))
    # У расписания нет ключа конфликта: карта конфликтов его не касается.
    widget.set_conflicts({("voice", "свет"): ["X"]})
    card = widget.cards()[0]
    assert card._conflict.isHidden()


# ----------------------------------------------------------------------
# горячая клавиша: захват — фейк, без железа
# ----------------------------------------------------------------------


def test_hotkey_capture_updates_label_and_model(app: QApplication, theme: ThemeManager) -> None:
    capture = _FakeCapture("ctrl+shift+b")
    model = _command(HotkeyTrigger(combo="ctrl+alt+a"))
    widget = TriggerList(theme, hotkey_capture=capture)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _HotkeyCard)
    card._button.click()
    assert capture.calls == 1
    assert card._label.text() == "ctrl+shift+b"
    assert model.triggers[0].combo == "ctrl+shift+b"


def test_hotkey_capture_none_keeps_combo(app: QApplication, theme: ThemeManager) -> None:
    capture = _FakeCapture(None)
    model = _command(HotkeyTrigger(combo="ctrl+alt+a"))
    widget = TriggerList(theme, hotkey_capture=capture)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _HotkeyCard)
    card._button.click()
    assert capture.calls == 1
    assert card._label.text() == "ctrl+alt+a"
    assert model.triggers[0].combo == "ctrl+alt+a"


def test_hotkey_button_disabled_without_capture(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)  # без hotkey_capture
    widget.set_command(_command(HotkeyTrigger(combo="ctrl+alt+a")))
    card = widget.cards()[0]
    assert isinstance(card, _HotkeyCard)
    assert card._button.isEnabled() is False
    # Прямой вызов при отсутствии захвата — тихий выход, ничего не меняет.
    card._on_capture()
    assert card._label.text() == "ctrl+alt+a"


def test_hotkey_invalid_capture_drops_trigger(app: QApplication, theme: ThemeManager) -> None:
    capture = _FakeCapture("неизвестнаяклавиша")
    model = _command(HotkeyTrigger(combo="ctrl+alt+a"))
    widget = TriggerList(theme, hotkey_capture=capture)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _HotkeyCard)
    card._button.click()
    # Комбинация не разбирается — HotkeyTrigger падает, триггер выпадает из модели.
    assert model.triggers == []


def test_hotkey_empty_combo_gives_no_trigger(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(HotkeyTrigger(combo="ctrl+alt+a")))
    card = widget.cards()[0]
    assert isinstance(card, _HotkeyCard)
    card._combo = ""
    assert card.trigger() is None


# ----------------------------------------------------------------------
# карточка события
# ----------------------------------------------------------------------


def test_event_edit_writes_model(app: QApplication, theme: ThemeManager) -> None:
    model = _command(EventTrigger(event_name="system.event"))
    widget = TriggerList(theme, event_names=("system.event",))
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _EventCard)
    card._combo.setCurrentText("app.closed")
    assert isinstance(model.triggers[0], EventTrigger)
    assert model.triggers[0].event_name == "app.closed"


def test_event_empty_name_gives_no_trigger(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(EventTrigger(event_name="system.event")))
    card = widget.cards()[0]
    assert isinstance(card, _EventCard)
    card._combo.setCurrentText("   ")
    assert card.trigger() is None


def test_event_invalid_name_drops_trigger(app: QApplication, theme: ThemeManager) -> None:
    model = _command(EventTrigger(event_name="system.event"))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _EventCard)
    card._combo.setCurrentText("123")  # имя события не может начинаться с цифры
    assert model.triggers == []


# ----------------------------------------------------------------------
# карточка расписания: cron и «один раз»
# ----------------------------------------------------------------------


def test_schedule_cron_reads_and_describes(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(TimerTrigger(cron="0 9 * * *")))
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    assert card._mode.currentData() == "cron"
    assert card._value.text() == "0 9 * * *"
    assert card._reading.text() == "ежедневно в 09:00"


def test_schedule_once_mode_from_fire_at(app: QApplication, theme: ThemeManager) -> None:
    moment = datetime(2026, 9, 20, 9, 30)
    widget = TriggerList(theme)
    widget.set_command(_command(TimerTrigger(fire_at=moment)))
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    assert card._mode.currentData() == "once"
    assert card._value.text() == "2026-09-20 09:30"
    assert card._reading.text() == ""


def test_schedule_switch_to_once_and_set_datetime(app: QApplication, theme: ThemeManager) -> None:
    model = _command(TimerTrigger(cron="0 9 * * *"))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    card._mode.setCurrentIndex(1)  # «Один раз»
    assert card._reading.text() == ""
    card._value.setText("2026-09-20 09:30")
    trigger = model.triggers[0]
    assert isinstance(trigger, TimerTrigger)
    assert trigger.fire_at is not None
    assert trigger.cron is None


def test_schedule_bad_datetime_drops_trigger(app: QApplication, theme: ThemeManager) -> None:
    model = _command(TimerTrigger(fire_at=datetime(2026, 9, 20, 9, 30)))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    card._value.setText("не дата")
    assert model.triggers == []


def test_schedule_bad_cron_drops_trigger(app: QApplication, theme: ThemeManager) -> None:
    model = _command(TimerTrigger(cron="0 9 * * *"))
    widget = TriggerList(theme)
    widget.set_command(model)
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    card._value.setText("привет")
    assert card._reading.text() == "привет"  # не пять полей — сырой текст
    assert model.triggers == []


def test_schedule_empty_value_gives_no_trigger(app: QApplication, theme: ThemeManager) -> None:
    widget = TriggerList(theme)
    widget.set_command(_command(TimerTrigger(cron="0 9 * * *")))
    card = widget.cards()[0]
    assert isinstance(card, _ScheduleCard)
    card._value.setText("")
    assert card.trigger() is None
