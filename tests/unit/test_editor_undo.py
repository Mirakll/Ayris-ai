"""Стек отмены/повтора редактора команд (задача 54), чистый слой.

Стек — над моделью, не над виджетом: и список (52), и ноды (53) — два вида одной
:class:`CommandModel`, поэтому шаг отмены — снимок «до/после», а не обратная
операция, и переключение видов несёт ту же историю. Здесь проверяется чистый
:class:`~ayris.gui.widgets.editor_undo.UndoStack` без Qt: отмена/повтор
восстанавливают модель по шагам, описание берётся из диффа, а быстрые правки
одного поля схлопываются в один шаг по таймауту (часы впрыснуты, без сна).
"""

from __future__ import annotations

import pytest

from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.gui.widgets.editor_undo import UndoStack

pytestmark = pytest.mark.unit


class FakeClock:
    """Впрыснутые часы: тест двигает время сам, чтобы проверить схлопывание."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _cmd(name: str, *texts: str) -> CommandModel:
    return CommandModel(
        name=name,
        actions=[ActionBlock(type="Say", params={"text": t}) for t in texts],
    )


# ----------------------------------------------------------------------
# базовые инварианты
# ----------------------------------------------------------------------


def test_fresh_stack_cannot_undo_or_redo() -> None:
    stack = UndoStack()
    stack.reset(_cmd("Свет"))
    assert stack.can_undo is False
    assert stack.can_redo is False
    assert stack.undo() is None
    assert stack.redo() is None


def test_record_empty_diff_pushes_nothing() -> None:
    stack = UndoStack()
    model = _cmd("Свет")
    stack.reset(model)
    # Та же модель — дифф пуст, шаг не записывается.
    assert stack.record(model.model_copy(deep=True)) is False
    assert stack.can_undo is False


# ----------------------------------------------------------------------
# отмена/повтор восстанавливают модель по шагам
# ----------------------------------------------------------------------


def test_undo_redo_restores_model_step_by_step() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))

    stack.record(_cmd("Свет", "раз"))  # шаг 1: добавлен блок
    stack.record(_cmd("Свет", "раз", "два"))  # шаг 2: добавлен ещё блок
    assert stack.can_undo is True

    back1 = stack.undo()  # к состоянию после шага 1
    assert back1 is not None and [b.params["text"] for b in back1.actions] == ["раз"]

    back0 = stack.undo()  # к базовому состоянию
    assert back0 is not None and back0.actions == []
    assert stack.can_undo is False

    fwd1 = stack.redo()  # снова шаг 1
    assert fwd1 is not None and [b.params["text"] for b in fwd1.actions] == ["раз"]
    assert stack.can_redo is True


def test_record_after_undo_clears_redo() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))
    stack.record(_cmd("Свет", "раз"))
    stack.undo()
    assert stack.can_redo is True
    # Новая правка после отмены обрубает повтор — классическое поведение стека.
    stack.record(_cmd("Свет", "иначе"))
    assert stack.can_redo is False


# ----------------------------------------------------------------------
# описание шага берётся из диффа
# ----------------------------------------------------------------------


def test_step_description_names_single_change() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))
    stack.record(_cmd("Свет", "раз"))
    # Одно изменение — его собственная подпись (добавлен блок Say).
    assert "Say" in (stack.undo_description() or "")


def test_multi_change_step_is_counted() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))
    # Сразу два изменения: имя и добавленный блок.
    stack.record(_cmd("Тьма", "раз"))
    assert stack.undo_description() == "Изменений: 2"


def test_descriptions_lists_steps_oldest_first() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))
    stack.record(_cmd("Свет", "раз"))
    stack.record(_cmd("Свет", "раз", "два"))
    descriptions = stack.descriptions()
    assert len(descriptions) == 2


# ----------------------------------------------------------------------
# схлопывание быстрых правок одного поля
# ----------------------------------------------------------------------


def test_rapid_same_field_edits_coalesce() -> None:
    clock = FakeClock()
    stack = UndoStack(coalesce_seconds=0.8, clock=clock)
    stack.reset(_cmd("Свет", "п"))

    stack.record(_cmd("Свет", "пр"))  # шаг 1
    clock.now += 0.1
    stack.record(_cmd("Свет", "при"))  # в окне — схлопывается в шаг 1
    clock.now += 0.1
    stack.record(_cmd("Свет", "привет"))  # ещё в окне — тоже в шаг 1

    # Один шаг на всю серию правок одного параметра.
    assert len(stack.descriptions()) == 1
    back = stack.undo()
    assert back is not None and back.actions[0].params["text"] == "п"  # к базовому


def test_edits_outside_window_do_not_coalesce() -> None:
    clock = FakeClock()
    stack = UndoStack(coalesce_seconds=0.8, clock=clock)
    stack.reset(_cmd("Свет", "п"))
    stack.record(_cmd("Свет", "пр"))
    clock.now += 2.0  # за окном
    stack.record(_cmd("Свет", "при"))
    assert len(stack.descriptions()) == 2


def test_block_add_does_not_coalesce_with_next() -> None:
    clock = FakeClock()
    stack = UndoStack(coalesce_seconds=10.0, clock=clock)
    stack.reset(_cmd("Свет"))
    stack.record(_cmd("Свет", "раз"))  # добавление блока — пустая подпись
    stack.record(_cmd("Свет", "раз", "два"))  # другое добавление — отдельный шаг
    # Добавления не схлопываются даже в одном окне: подпись пуста.
    assert len(stack.descriptions()) == 2


# ----------------------------------------------------------------------
# лимит и сброс
# ----------------------------------------------------------------------


def test_limit_evicts_oldest_steps() -> None:
    clock = FakeClock()
    stack = UndoStack(limit=3, coalesce_seconds=0.0, clock=clock)
    stack.reset(_cmd("Свет"))
    for i in range(6):
        clock.now += 1.0
        stack.record(_cmd(f"Имя{i}"))
    # Не больше лимита шагов в истории.
    assert len(stack.descriptions()) == 3


def test_reset_clears_history() -> None:
    stack = UndoStack(clock=FakeClock())
    stack.reset(_cmd("Свет"))
    stack.record(_cmd("Свет", "раз"))
    assert stack.can_undo is True
    stack.reset(_cmd("Другая"))
    assert stack.can_undo is False
    assert stack.can_redo is False
