"""Структурный дифф двух команд (задача 54).

Дифф считается на моделях, а не на JSON-тексте: переотступ файла или переставленный
ключ — не изменение, а перенесённый блок — одно перемещение, а не стена шума. Здесь
проверяется, что дифф находит добавленный, удалённый и изменённый блок с правильным
путём (``actions[1].then[0]``), различает поля, триггеры, переменные и звуки, и что
две одинаковые команды дают пустой дифф. Тесты чистые — ни Qt, ни базы.
"""

from __future__ import annotations

import pytest

from ayris.actions.macros.diff import (
    CHANGE_ADDED,
    CHANGE_CHANGED,
    CHANGE_REMOVED,
    diff_commands,
)
from ayris.actions.macros.schema import (
    ActionBlock,
    CommandModel,
    SoundBinding,
    SoundStage,
    VariableModel,
    VoiceTrigger,
)

pytestmark = pytest.mark.unit


def _say(text: str) -> ActionBlock:
    return ActionBlock(type="Say", params={"text": text})


# ----------------------------------------------------------------------
# ничего не изменилось
# ----------------------------------------------------------------------


def test_identical_commands_have_empty_diff() -> None:
    command = CommandModel(name="Свет", actions=[_say("привет")])
    result = diff_commands(command, command.model_copy(deep=True))
    assert result.is_empty
    assert result.summary() == "без изменений"


def test_reordered_json_keys_are_not_a_change() -> None:
    # Параметры в разном порядке — та же модель, дифф пуст.
    old = CommandModel(name="К", actions=[ActionBlock(type="X", params={"a": "1", "b": "2"})])
    new = CommandModel(name="К", actions=[ActionBlock(type="X", params={"b": "2", "a": "1"})])
    assert diff_commands(old, new).is_empty


# ----------------------------------------------------------------------
# блоки: добавлен, удалён, изменён — с правильным путём
# ----------------------------------------------------------------------


def test_added_block_reported_at_its_path() -> None:
    old = CommandModel(name="К", actions=[_say("раз")])
    new = CommandModel(name="К", actions=[_say("раз"), _say("два")])
    result = diff_commands(old, new)
    blocks = result.of_category("block")
    assert len(blocks) == 1
    assert blocks[0].kind == CHANGE_ADDED
    assert blocks[0].path == "actions[1]"
    assert "Say" in blocks[0].label


def test_removed_block_reported_at_its_path() -> None:
    old = CommandModel(name="К", actions=[_say("раз"), _say("два")])
    new = CommandModel(name="К", actions=[_say("раз")])
    result = diff_commands(old, new)
    blocks = result.of_category("block")
    assert len(blocks) == 1
    assert blocks[0].kind == CHANGE_REMOVED
    assert blocks[0].path == "actions[1]"


def test_changed_param_reported_with_old_and_new() -> None:
    old = CommandModel(name="К", actions=[_say("раз")])
    new = CommandModel(name="К", actions=[_say("два")])
    result = diff_commands(old, new)
    changed = [c for c in result.of_category("block") if c.kind == CHANGE_CHANGED]
    assert len(changed) == 1
    assert changed[0].path == "actions[0]"
    assert "text" in changed[0].label
    assert changed[0].old == "раз"
    assert changed[0].new == "два"


def test_nested_block_change_carries_full_path() -> None:
    # Изменение внутри ветки then блока If: путь actions[0].then[0].
    old = CommandModel(
        name="К",
        actions=[ActionBlock(type="If", params={"condition": "{x}"}, then=[_say("раз")])],
    )
    new = CommandModel(
        name="К",
        actions=[ActionBlock(type="If", params={"condition": "{x}"}, then=[_say("два")])],
    )
    result = diff_commands(old, new)
    changed = [c for c in result.of_category("block") if c.kind == CHANGE_CHANGED]
    assert len(changed) == 1
    assert changed[0].path == "actions[0].then[0]"


def test_insert_in_middle_does_not_shift_following_blocks() -> None:
    # Вставка в середину: выравнивание по типам не превращает хвост в изменения.
    old = CommandModel(name="К", actions=[ActionBlock(type="A"), ActionBlock(type="C")])
    new = CommandModel(
        name="К",
        actions=[ActionBlock(type="A"), ActionBlock(type="B"), ActionBlock(type="C")],
    )
    result = diff_commands(old, new)
    blocks = result.of_category("block")
    assert len(blocks) == 1
    assert blocks[0].kind == CHANGE_ADDED
    assert blocks[0].path == "actions[1]"


# ----------------------------------------------------------------------
# поля, триггеры, переменные, звуки
# ----------------------------------------------------------------------


def test_field_change_is_reported() -> None:
    old = CommandModel(name="Свет", priority=0)
    new = CommandModel(name="Тьма", priority=5)
    fields = diff_commands(old, new).of_category("field")
    labels = {c.path for c in fields}
    assert "Имя" in labels
    assert "Приоритет" in labels


def test_added_and_removed_trigger() -> None:
    old = CommandModel(name="К")
    new = CommandModel(name="К", triggers=[VoiceTrigger(phrase="свет")])
    added = diff_commands(old, new).of_category("trigger")
    assert added and added[0].kind == CHANGE_ADDED
    removed = diff_commands(new, old).of_category("trigger")
    assert removed and removed[0].kind == CHANGE_REMOVED


def test_variable_change_is_reported() -> None:
    old = CommandModel(name="К", variables=[VariableModel(name="v", default="1")])
    new = CommandModel(name="К", variables=[VariableModel(name="v", default="2")])
    changed = diff_commands(old, new).of_category("variable")
    assert changed and changed[0].kind == CHANGE_CHANGED


def test_sound_change_is_reported() -> None:
    old = CommandModel(
        name="К",
        sounds=[SoundBinding(stage=SoundStage.ON_SUCCESS, value="builtin:done")],
    )
    new = CommandModel(
        name="К",
        sounds=[SoundBinding(stage=SoundStage.ON_SUCCESS, value="builtin:done", volume=50)],
    )
    changed = diff_commands(old, new).of_category("sound")
    assert changed and changed[0].kind == CHANGE_CHANGED


# ----------------------------------------------------------------------
# сводка
# ----------------------------------------------------------------------


def test_summary_counts_kinds() -> None:
    old = CommandModel(name="Свет", actions=[_say("раз"), _say("хвост")])
    new = CommandModel(name="Тьма", actions=[_say("два"), _say("хвост"), _say("три")])
    result = diff_commands(old, new)
    # Имя изменено (~), текст первого блока изменён (~), добавлен блок (+).
    summary = result.summary()
    assert "+1" in summary
    assert "~2" in summary
