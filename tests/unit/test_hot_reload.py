"""Горячее применение сохранённой команды без перезапуска (задача 54).

:class:`~ayris.actions.macros.hot_reload.HotReloader` владеет порядком вокруг
сохранения: проверка → запись строки, триггеров и версии в одной транзакции →
публикация ``CommandsChanged`` (подсистемы триггеров перерегистрируют команду) →
публикация ``CommandReloaded`` (дерево и оверлей). Ключевые гарантии, которые
здесь и проверяются:

* при успехе оба события уходят, и ``CommandsChanged`` — раньше ``CommandReloaded``,
  чтобы к приходу второго команда уже была живой в новом виде;
* ошибка проверки не пишет и не публикует ничего — прежняя версия остаётся
  зарегистрированной, команда не выключается как побочный эффект;
* перерегистрация идемпотентна: подсистема, слушающая ``CommandsChanged``, получает
  ровно одно событие на сохранение и перестраивает свой взгляд на команду по нему,
  так что двойное сохранение не даёт двойной подписки;
* снятие команды (``retire_command``) публикует те же два события с пометкой
  удаления.

Здесь нет ни окна, ни живого воркера: сейвер — фейк, шина — настоящая, подписчик —
список. Индекс NLU представляет фейковая подсистема, которая, как настоящий
``TriggerIndex``, перестраивает свой взгляд на команду из события.
"""

from __future__ import annotations

import pytest

from ayris.actions.macros.hot_reload import HotReloader, ReloadResult
from ayris.actions.macros.schema import ActionBlock, CommandModel, VoiceTrigger
from ayris.actions.macros.validator import MacroValidationError
from ayris.core.events import (
    COMMANDS_CHANGE_DELETED,
    COMMANDS_CHANGE_SAVED,
    CommandReloaded,
    CommandsChanged,
    EventBus,
)

pytestmark = pytest.mark.unit


class FakeSaver:
    """Стоит за протоколом :class:`CommandSaver`: помнит вызовы, отдаёт модель с id.

    Настоящий store пишет строку, триггеры, версию и подрезает историю в одной
    транзакции; фейку это не нужно — он лишь возвращает команду с проставленным id,
    как её прочитали бы обратно, чтобы reloader было что публиковать.
    """

    def __init__(self, *, assign_id: int | None = 7) -> None:
        self.calls: list[CommandModel] = []
        self._assign_id = assign_id

    def save_command(self, model: CommandModel) -> CommandModel:
        self.calls.append(model)
        return model.model_copy(update={"id": model.id or self._assign_id})


class FakeIndex:
    """Подсистема, как NLU-индекс: перестраивает фразы одной команды из события.

    Слушает ``CommandsChanged`` и по ``command_id`` заменяет свой набор фраз тем,
    что читает у сейвера, — ровно как настоящий индекс перечитывает строки. Считает,
    сколько раз перестраивалась, чтобы тест увидел идемпотентность.
    """

    def __init__(self, bus: EventBus, phrases_by_id: dict[int, list[str]]) -> None:
        self.phrases: dict[int, list[str]] = {}
        self.rebuilds = 0
        self._source = phrases_by_id
        self._unsub = bus.subscribe(CommandsChanged, self._on_changed)

    def _on_changed(self, event: CommandsChanged) -> None:
        self.rebuilds += 1
        if event.command_id is None:
            return
        if event.change == COMMANDS_CHANGE_DELETED:
            self.phrases.pop(event.command_id, None)
            return
        # Идемпотентно: набор фраз заменяется целиком, а не дополняется.
        self.phrases[event.command_id] = list(self._source.get(event.command_id, ()))


def _voice_command(command_id: int | None, phrase: str) -> CommandModel:
    return CommandModel(
        id=command_id,
        name="Свет",
        triggers=[VoiceTrigger(phrase=phrase)],
        actions=[ActionBlock(type="Say", params={"text": "готово"})],
    )


# ----------------------------------------------------------------------
# успешное применение: перерегистрация без перезапуска, порядок событий
# ----------------------------------------------------------------------


def test_apply_reregisters_command_and_updates_index() -> None:
    bus = EventBus()
    saver = FakeSaver(assign_id=7)
    index = FakeIndex(bus, phrases_by_id={7: ["включи свет"]})
    reloader = HotReloader(saver, bus)

    result = reloader.apply_command(_voice_command(7, "включи свет"))

    assert isinstance(result, ReloadResult)
    assert saver.calls  # запись произошла
    # Индекс перестроил свой взгляд на команду из события — фраза на месте, без рестарта.
    assert index.phrases == {7: ["включи свет"]}
    assert index.rebuilds == 1


def test_apply_publishes_changed_before_reloaded() -> None:
    bus = EventBus()
    order: list[str] = []
    bus.subscribe(CommandsChanged, lambda _e: order.append("changed"))
    bus.subscribe(CommandReloaded, lambda _e: order.append("reloaded"))
    reloader = HotReloader(FakeSaver(assign_id=7), bus)

    reloader.apply_command(_voice_command(7, "свет"))

    # CommandsChanged раньше CommandReloaded: к приходу второго команда уже живая.
    assert order == ["changed", "reloaded"]


def test_reloaded_names_the_command_and_change() -> None:
    bus = EventBus()
    seen: list[CommandReloaded] = []
    bus.subscribe(CommandReloaded, seen.append)
    reloader = HotReloader(FakeSaver(assign_id=42), bus)

    reloader.apply_command(_voice_command(None, "свет"))

    assert len(seen) == 1
    assert seen[0].command_id == 42
    assert seen[0].change == COMMANDS_CHANGE_SAVED


def test_double_save_keeps_one_subscription_worth_of_phrases() -> None:
    # Идемпотентность: два сохранения подряд — индекс перестраивается дважды, но
    # набор фраз остаётся одинарным, без задвоения подписки/фразы.
    bus = EventBus()
    saver = FakeSaver(assign_id=7)
    index = FakeIndex(bus, phrases_by_id={7: ["свет"]})
    reloader = HotReloader(saver, bus)

    reloader.apply_command(_voice_command(7, "свет"))
    reloader.apply_command(_voice_command(7, "свет"))

    assert index.rebuilds == 2
    assert index.phrases == {7: ["свет"]}  # одна фраза, не две


# ----------------------------------------------------------------------
# ошибка проверки: прежняя версия остаётся живой, ничего не опубликовано
# ----------------------------------------------------------------------


def test_validation_error_publishes_nothing_and_does_not_save() -> None:
    bus = EventBus()
    saver = FakeSaver()
    events: list[object] = []
    bus.subscribe(CommandsChanged, events.append)
    bus.subscribe(CommandReloaded, events.append)
    reloader = HotReloader(saver, bus)

    # Ссылка на необъявленную переменную — ошибка проверки (без реестра).
    invalid = CommandModel(
        id=7,
        name="Свет",
        actions=[ActionBlock(type="Say", params={"text": "привет, {никто}"})],
    )
    with pytest.raises(MacroValidationError):
        reloader.apply_command(invalid)

    assert saver.calls == []  # ничего не записано
    assert events == []  # ничего не опубликовано — прежняя версия жива


def test_validation_error_keeps_previous_index_intact() -> None:
    # Сначала успешно применяем валидную версию — индекс держит её фразу.
    bus = EventBus()
    saver = FakeSaver(assign_id=7)
    index = FakeIndex(bus, phrases_by_id={7: ["включи свет"]})
    reloader = HotReloader(saver, bus)
    reloader.apply_command(_voice_command(7, "включи свет"))
    assert index.phrases == {7: ["включи свет"]}
    rebuilds_before = index.rebuilds

    # Теперь пытаемся применить невалидную — индекс не должен шелохнуться.
    invalid = CommandModel(
        id=7,
        name="Свет",
        actions=[ActionBlock(type="Say", params={"text": "{нет_такой}"})],
    )
    with pytest.raises(MacroValidationError):
        reloader.apply_command(invalid)

    assert index.phrases == {7: ["включи свет"]}  # прежняя версия зарегистрирована
    assert index.rebuilds == rebuilds_before  # индекс не перестраивался


# ----------------------------------------------------------------------
# снятие команды
# ----------------------------------------------------------------------


def test_retire_deleted_publishes_deleted_change_and_drops_phrases() -> None:
    bus = EventBus()
    saver = FakeSaver(assign_id=7)
    index = FakeIndex(bus, phrases_by_id={7: ["свет"]})
    reloader = HotReloader(saver, bus)
    reloader.apply_command(_voice_command(7, "свет"))
    assert index.phrases == {7: ["свет"]}

    changes: list[str] = []
    bus.subscribe(CommandReloaded, lambda e: changes.append(e.change))
    reloader.retire_command(7, deleted=True)

    assert changes == [COMMANDS_CHANGE_DELETED]
    assert index.phrases == {}  # фраза снята без рестарта


def test_retire_without_delete_uses_saved_change() -> None:
    bus = EventBus()
    changes: list[str] = []
    bus.subscribe(CommandReloaded, lambda e: changes.append(e.change))
    reloader = HotReloader(FakeSaver(), bus)

    reloader.retire_command(7)

    assert changes == [COMMANDS_CHANGE_SAVED]
