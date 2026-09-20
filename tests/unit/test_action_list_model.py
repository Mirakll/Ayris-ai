"""The block-list model of the command editor (task 52), without a widget.

:class:`~ayris.gui.widgets.action_list.ActionListModel` is the pure half of the action
list: it owns the working :class:`~ayris.actions.macros.schema.CommandModel` and edits
``model.actions`` in place. These tests drive it directly — no Qt, no view — and check
the two things a view cannot: that the tree flattens into the rows the view will draw,
and that every mutation lands at the right block path and leaves a model the schema
still accepts. Moving a block into and out of an ``If`` branch, the depth ceiling, the
self-nest refusal, undo, duplicate and the clipboard are all asserted against the model
state, never a rendered row.
"""

from __future__ import annotations

import pytest

from ayris.actions.macros.schema import MAX_BLOCK_DEPTH, ActionBlock, CommandModel
from ayris.gui.widgets.action_list import ActionListModel

pytestmark = pytest.mark.unit


def _command(*blocks: ActionBlock) -> CommandModel:
    return CommandModel(name="Команда", actions=list(blocks))


def _model(*blocks: ActionBlock) -> ActionListModel:
    model = ActionListModel()
    model.set_command(_command(*blocks))
    return model


# ----------------------------------------------------------------------
# reading
# ----------------------------------------------------------------------


def test_empty_model_has_no_rows() -> None:
    assert ActionListModel().rows() == []


def test_rows_flatten_depth_first_parents_before_children() -> None:
    model = _model(
        ActionBlock(type="Say", params={"text": "раз"}),
        ActionBlock(
            type="If",
            params={"condition": "{x}"},
            then=[ActionBlock(type="Say", params={"text": "да"})],
            else_=[ActionBlock(type="Say", params={"text": "нет"})],
        ),
    )
    rows = model.rows()
    # Say, If, then/Say, else/Say — parent before either branch, then before else.
    assert [r.block.type for r in rows] == ["Say", "If", "Say", "Say"]
    assert [r.path_text for r in rows] == [
        "actions[0]",
        "actions[1]",
        "actions[1].then[0]",
        "actions[1].else[0]",
    ]
    assert [r.depth for r in rows] == [0, 0, 1, 1]
    # A branch's first child is marked so the view can head it.
    assert rows[2].branch == "then" and rows[2].branch_head
    assert rows[3].branch == "else" and rows[3].branch_head


def test_branches_of_reads_the_schema() -> None:
    model = _model()
    assert model.branches_of(ActionBlock(type="If")) == ("then", "else")
    assert model.branches_of(ActionBlock(type="Try")) == ("body", "catch")
    assert model.branches_of(ActionBlock(type="Say")) == ()


# ----------------------------------------------------------------------
# insert
# ----------------------------------------------------------------------


def test_insert_at_root_returns_path_and_mutates_model() -> None:
    model = _model(ActionBlock(type="Say"))
    path = model.insert(ActionBlock(type="Break"), ("actions",), 0)
    assert path == ("actions", 0)
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["Break", "Say"]


def test_insert_clamps_past_the_end() -> None:
    model = _model(ActionBlock(type="Say"))
    path = model.insert(ActionBlock(type="Break"), ("actions",), 99)
    assert path == ("actions", 1)


def test_insert_type_makes_a_default_block() -> None:
    model = _model()
    path = model.insert_type("Say", ("actions",), 0)
    assert path == ("actions", 0)
    assert model.block_at(("actions", 0)) is not None
    block = model.block_at(("actions", 0))
    assert block is not None and block.type == "Say"


def test_insert_into_a_branch() -> None:
    model = _model(ActionBlock(type="If", then=[ActionBlock(type="Say")]))
    path = model.insert_type("Break", ("actions", 0, "then"), 1)
    assert path == ("actions", 0, "then", 1)
    outer = model.block_at(("actions", 0))
    assert outer is not None
    assert [b.type for b in outer.then] == ["Say", "Break"]


def test_insert_refused_past_max_depth() -> None:
    # A chain of Ifs nested to the deepest the schema allows — the innermost If sits at
    # depth MAX_BLOCK_DEPTH - 1. Inserting into *that* If's own branch would put a block
    # at MAX_BLOCK_DEPTH, too deep, so the model refuses instead of building a tree that
    # could never be saved (the guard the schema would raise on at the next edit).
    node = ActionBlock(type="If")
    for _ in range(MAX_BLOCK_DEPTH - 1):
        node = ActionBlock(type="If", then=[node])
    model = _model(node)
    command = model.command
    assert command is not None  # the nested command is itself valid
    # The container of the innermost If's then-branch, one level past the deepest block.
    container: tuple[object, ...] = ("actions",)
    for _ in range(MAX_BLOCK_DEPTH):
        container = (*container, 0, "then")
    assert model.insert_type("Break", container, 0) is None
    # A childless block one level shallower is still allowed (it sits at the last legal
    # depth), proving the guard is at the boundary and not one level too eager.
    legal: tuple[object, ...] = ("actions",)
    for _ in range(MAX_BLOCK_DEPTH - 1):
        legal = (*legal, 0, "then")
    assert model.insert_type("Break", legal, 0) == (*legal, 0)


# ----------------------------------------------------------------------
# move
# ----------------------------------------------------------------------


def test_move_reorders_within_root() -> None:
    model = _model(
        ActionBlock(type="Say", params={"text": "раз"}),
        ActionBlock(type="Say", params={"text": "два"}),
        ActionBlock(type="Say", params={"text": "три"}),
    )
    # Move the first block to the end: popping it shifts the target index down by one.
    path = model.move(("actions",), 0, ("actions",), 3)
    assert path == ("actions", 2)
    command = model.command
    assert command is not None
    assert [b.params.get("text") for b in command.actions] == ["два", "три", "раз"]


def test_move_into_a_branch() -> None:
    model = _model(
        ActionBlock(type="If", then=[ActionBlock(type="Say", params={"text": "тело"})]),
        ActionBlock(type="Break"),
    )
    path = model.move(("actions",), 1, ("actions", 0, "then"), 0)
    assert path == ("actions", 0, "then", 0)
    outer = model.block_at(("actions", 0))
    assert outer is not None
    assert [b.type for b in outer.then] == ["Break", "Say"]
    command = model.command
    assert command is not None
    assert len(command.actions) == 1  # the Break left the root


def test_move_out_of_a_branch_to_root() -> None:
    model = _model(
        ActionBlock(type="If", then=[ActionBlock(type="Break")]),
    )
    path = model.move(("actions", 0, "then"), 0, ("actions",), 1)
    assert path == ("actions", 1)
    outer = model.block_at(("actions", 0))
    assert outer is not None
    assert outer.then == []
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["If", "Break"]


def test_move_refuses_to_drop_a_block_inside_itself() -> None:
    model = _model(ActionBlock(type="If", then=[ActionBlock(type="Say")]))
    # Dragging the If into its own then-branch would detach it from the tree.
    refused = model.move(("actions",), 0, ("actions", 0, "then"), 1)
    assert refused is None
    # The tree is untouched.
    assert [r.path_text for r in model.rows()] == ["actions[0]", "actions[0].then[0]"]


def test_move_corrects_destination_below_the_source() -> None:
    # Destination path descends through a sibling that sits after the source: popping
    # the source shifts that sibling up, and the model must follow it.
    model = _model(
        ActionBlock(type="Break"),
        ActionBlock(type="If", then=[ActionBlock(type="Say")]),
    )
    path = model.move(("actions",), 0, ("actions", 1, "then"), 1)
    # The If is now at index 0; the Break landed after the Say in its then-branch.
    assert path == ("actions", 0, "then", 1)
    outer = model.block_at(("actions", 0))
    assert outer is not None
    assert [b.type for b in outer.then] == ["Say", "Break"]


# ----------------------------------------------------------------------
# remove / undo
# ----------------------------------------------------------------------


def test_remove_returns_the_block_and_offers_undo() -> None:
    model = _model(ActionBlock(type="Say"), ActionBlock(type="Break"))
    assert not model.can_undo()
    removed = model.remove(("actions",), 0)
    assert removed is not None and removed.type == "Say"
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["Break"]
    assert model.can_undo()


def test_undo_restores_at_the_same_place() -> None:
    model = _model(ActionBlock(type="Say"), ActionBlock(type="Break"))
    model.remove(("actions",), 0)
    assert model.undo_remove()
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["Say", "Break"]
    assert not model.can_undo()


def test_remove_out_of_range_is_a_no_op() -> None:
    model = _model(ActionBlock(type="Say"))
    assert model.remove(("actions",), 5) is None
    assert not model.can_undo()


# ----------------------------------------------------------------------
# duplicate / clipboard
# ----------------------------------------------------------------------


def test_duplicate_deep_copies_after_the_original() -> None:
    model = _model(
        ActionBlock(type="If", then=[ActionBlock(type="Say", params={"text": "тело"})]),
    )
    path = model.duplicate(("actions",), 0)
    assert path == ("actions", 1)
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["If", "If"]
    # The copy is deep: editing the copy's branch does not touch the original.
    command.actions[1].then[0].params["text"] = "иное"
    assert command.actions[0].then[0].params["text"] == "тело"


def test_copy_paste_between_positions() -> None:
    model = _model(ActionBlock(type="Say", params={"text": "образец"}))
    assert not model.has_clipboard()
    assert model.copy(("actions",), 0)
    assert model.has_clipboard()
    path = model.paste(("actions",), 1)
    assert path == ("actions", 1)
    command = model.command
    assert command is not None
    assert [b.params.get("text") for b in command.actions] == ["образец", "образец"]


def test_paste_without_clipboard_is_none() -> None:
    model = _model()
    assert model.paste(("actions",), 0) is None


def test_paste_survives_a_new_command() -> None:
    # The clipboard is the copy-paste-between-commands path: it must outlive set_command.
    source = _model(ActionBlock(type="Say", params={"text": "перенос"}))
    source.copy(("actions",), 0)
    source.set_command(_command(ActionBlock(type="Break")))
    path = source.paste(("actions",), 1)
    assert path == ("actions", 1)
    command = source.command
    assert command is not None
    assert [b.type for b in command.actions] == ["Break", "Say"]


# ----------------------------------------------------------------------
# enable / comment
# ----------------------------------------------------------------------


def test_set_enabled_toggles_on_the_model() -> None:
    model = _model(ActionBlock(type="Say"))
    model.set_enabled(("actions",), 0, enabled=False)
    block = model.block_at(("actions", 0))
    assert block is not None and block.enabled is False


def test_set_comment_trims_and_stores() -> None:
    model = _model(ActionBlock(type="Say"))
    model.set_comment(("actions",), 0, "  заметка  ")
    block = model.block_at(("actions", 0))
    assert block is not None and block.comment == "заметка"


def test_block_at_out_of_range_is_none() -> None:
    model = _model(ActionBlock(type="Say"))
    assert model.block_at(("actions", 3)) is None
    assert model.block_at(()) is None
