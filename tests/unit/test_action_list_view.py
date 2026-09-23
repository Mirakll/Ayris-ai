"""The Qt view of the action list (task 52), offscreen.

The model :class:`~ayris.gui.widgets.action_list.ActionListModel` is covered by
``test_action_list_model``; here only the thin view
:class:`~ayris.gui.widgets.action_list.ActionListView` is exercised — the mapping from
model rows to tree items, the signals a selection and a checkbox raise, the labels a
row shows (disabled, branch heading, comment, dangerous tooltip), where a drop lands,
and the per-row menu handlers. No pixels are inspected: every assertion is about the
tree's item data or the model state a handler changed. Widgets are closed in the app
fixture; the tree holds no timers of its own.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QMimeData, Qt
from PySide6.QtWidgets import QAbstractItemView, QApplication

from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import action_list as al
from ayris.gui.widgets.action_list import ActionListModel, ActionListView
from ayris.gui.widgets.block_palette import BLOCK_MIME

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


def _command(*blocks: ActionBlock) -> CommandModel:
    return CommandModel(name="Команда", actions=list(blocks))


def _view(theme: ThemeManager, *blocks: ActionBlock) -> tuple[ActionListView, ActionListModel]:
    model = ActionListModel()
    model.set_command(_command(*blocks))
    view = ActionListView(model, theme)
    return view, model


class _FakeMime:
    """Just enough of a drop event's mime for the accept-path tests."""

    def __init__(self, block_type: str | None) -> None:
        self._data = QMimeData()
        if block_type is not None:
            self._data.setData(BLOCK_MIME, block_type.encode("utf-8"))

    def mimeData(self) -> QMimeData:  # noqa: N802 — mirrors the Qt event API.
        return self._data


class _FakeDropEvent:
    """A stand-in for QDropEvent carrying a block-type mime, recording the verdict."""

    def __init__(self, block_type: str | None) -> None:
        self._mime = _FakeMime(block_type)
        self.accepted = False
        self.ignored = False

    def mimeData(self) -> QMimeData:  # noqa: N802 — mirrors the Qt event API.
        return self._mime.mimeData()

    def acceptProposedAction(self) -> None:  # noqa: N802 — mirrors the Qt event API.
        self.accepted = True

    def ignore(self) -> None:
        self.ignored = True


# ----------------------------------------------------------------------
# building the tree
# ----------------------------------------------------------------------


def test_rebuild_lays_out_rows_as_tree(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(
        theme,
        ActionBlock(type="Say"),
        ActionBlock(type="If", then=[ActionBlock(type="Say")]),
    )
    # Two top-level rows; the If carries its then-child as a nested item.
    assert view.topLevelItemCount() == 2
    if_item = view.topLevelItem(1)
    assert if_item is not None
    assert if_item.childCount() == 1


def test_disabled_and_comment_show_in_label(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say", enabled=False, comment="привет"))
    item = view.topLevelItem(0)
    assert item is not None
    label = item.text(0)
    assert "(выкл.)" in label
    assert "привет" in label


def test_detached_shows_in_label(app: QApplication, theme: ThemeManager) -> None:
    # Свободная (неподключённая) нода видна в списке и помечена «(не подключено)»,
    # тем же способом, что и выключенный блок, — список показывает все блоки по порядку.
    view, _ = _view(theme, ActionBlock(type="Say", detached=True))
    item = view.topLevelItem(0)
    assert item is not None
    assert "(не подключено)" in item.text(0)


def test_branch_head_prefixes_label(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="If", then=[ActionBlock(type="Say")]))
    if_item = view.topLevelItem(0)
    assert if_item is not None
    child = if_item.child(0)
    assert child is not None
    assert child.text(0).startswith("[Тогда]")


def test_dangerous_block_marked_in_tooltip(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="RunShell"))
    item = view.topLevelItem(0)
    assert item is not None
    assert "⚠" in item.toolTip(0)


# ----------------------------------------------------------------------
# selection and the enable checkbox
# ----------------------------------------------------------------------


def test_selecting_a_row_emits_its_path(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"), ActionBlock(type="Break"))
    seen: list[tuple[object, ...]] = []
    view.block_selected.connect(seen.append)
    item = view.topLevelItem(1)
    assert item is not None
    view.setCurrentItem(item)
    assert seen and seen[-1] == ("actions", 1)


def test_selected_path_reports_current(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"))
    assert view.selected_path() == ()
    view.select_path(("actions", 0))
    assert view.selected_path() == ("actions", 0)


def test_unchecking_a_row_disables_the_block(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say"))
    changed: list[int] = []
    view.changed.connect(lambda: changed.append(1))
    item = view.topLevelItem(0)
    assert item is not None
    item.setCheckState(0, Qt.CheckState.Unchecked)
    block = model.block_at(("actions", 0))
    assert block is not None and block.enabled is False
    assert changed


# ----------------------------------------------------------------------
# drop targets and drag acceptance
# ----------------------------------------------------------------------


def test_drop_on_empty_area_targets_root_end(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"))
    container, index = view._drop_target(None)
    assert container == ("actions",)
    assert index == 1


def test_drop_onto_logic_block_targets_first_branch(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="If", then=[ActionBlock(type="Say")]))
    view.dropIndicatorPosition = (  # type: ignore[method-assign]
        lambda: QAbstractItemView.DropIndicatorPosition.OnItem
    )
    if_item = view.topLevelItem(0)
    container, index = view._drop_target(if_item)
    assert container == ("actions", 0, "then")
    assert index == 0


def test_drop_below_a_row_lands_after_it(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"), ActionBlock(type="Break"))
    view.dropIndicatorPosition = (  # type: ignore[method-assign]
        lambda: QAbstractItemView.DropIndicatorPosition.BelowItem
    )
    item = view.topLevelItem(0)
    container, index = view._drop_target(item)
    assert container == ("actions",)
    assert index == 1


def test_drag_enter_accepts_block_mime(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"))
    event = _FakeDropEvent("Say")
    view.dragEnterEvent(event)  # type: ignore[arg-type]
    assert event.accepted


def test_drag_move_accepts_block_mime(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"))
    event = _FakeDropEvent("Break")
    view.dragMoveEvent(event)  # type: ignore[arg-type]
    assert event.accepted


def test_drop_block_mime_inserts_a_block(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say"))
    view.itemAt = lambda _point: None  # type: ignore[method-assign] — force the root target.
    event = _FakeDropEvent("Break")
    view.dropEvent(event)  # type: ignore[arg-type]
    assert event.accepted
    command = model.command
    assert command is not None
    assert [b.type for b in command.actions] == ["Say", "Break"]


# ----------------------------------------------------------------------
# the per-row menu handlers
# ----------------------------------------------------------------------


def test_edit_comment_writes_through(
    app: QApplication, theme: ThemeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, model = _view(theme, ActionBlock(type="Say"))
    monkeypatch.setattr(al.QInputDialog, "getText", staticmethod(lambda *_a, **_k: ("шаг", True)))
    view._edit_comment(("actions",), 0)
    block = model.block_at(("actions", 0))
    assert block is not None and block.comment == "шаг"


def test_edit_comment_cancelled_keeps_value(
    app: QApplication, theme: ThemeManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    view, model = _view(theme, ActionBlock(type="Say", comment="старое"))
    monkeypatch.setattr(
        al.QInputDialog, "getText", staticmethod(lambda *_a, **_k: ("новое", False))
    )
    view._edit_comment(("actions",), 0)
    block = model.block_at(("actions", 0))
    assert block is not None and block.comment == "старое"


def test_toggle_enabled_flips_the_block(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say"))
    block = model.block_at(("actions", 0))
    view._toggle_enabled(("actions",), 0, block)
    assert block is not None and block.enabled is False


def test_delete_and_undo_round_trip(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say"), ActionBlock(type="Break"))
    view._delete(("actions",), 0)
    command = model.command
    assert command is not None and [b.type for b in command.actions] == ["Break"]
    view._undo()
    assert [b.type for b in command.actions] == ["Say", "Break"]


def test_duplicate_through_act_helper(app: QApplication, theme: ThemeManager) -> None:
    view, model = _view(theme, ActionBlock(type="Say"))
    view._act(model.duplicate, ("actions",), 0)
    command = model.command
    assert command is not None and [b.type for b in command.actions] == ["Say", "Say"]


def test_root_len_counts_top_level(app: QApplication, theme: ThemeManager) -> None:
    view, _ = _view(theme, ActionBlock(type="Say"), ActionBlock(type="Break"))
    assert view._root_len() == 2
