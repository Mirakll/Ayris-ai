"""The command library tree (task 51): the model on a fixture, the widget offscreen.

No window is shown. Most assertions run against :class:`CommandTreeModel` and
:class:`CommandTreeStore` over an in-memory database — the tree is built from a
fixture and checked by row counts, folder counters, the result of a move in the
database, filter results, conflict marks and an export→import round-trip. The parts
that need the widget (drag-and-drop, the context-menu operations, multi-selection)
are exercised through the model's own drop and the store's bulk methods and verified
by state, not by a screenshot. Every widget is closed in teardown so CI does not hang
on an open view.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pytest
from PySide6.QtCore import QMimeData, QModelIndex, Qt
from PySide6.QtWidgets import QApplication

from ayris.core.database import Database
from ayris.core.models import Command, CommandFolder
from ayris.core.repositories import Repositories
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.command_tree import CommandTree
from ayris.gui.widgets.command_tree_model import (
    CONFLICT_ROLE,
    COUNT_ROLE,
    ENABLED_ROLE,
    ENTITY_ID_ROLE,
    KIND_ROLE,
    CommandTreeModel,
    CommandTreeStore,
    ConflictStrategy,
    NodeKind,
    StatusFilter,
    TreeFilter,
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
def repos() -> Iterator[Repositories]:
    database = Database.open(":memory:")
    yield Repositories(database)
    database.close()


@pytest.fixture
def store(repos: Repositories) -> CommandTreeStore:
    profile = repos.profiles.create("Основной", activate=True)
    assert profile.id is not None
    work = repos.folders.create(CommandFolder(name="Работа", profile_id=profile.id))
    games = repos.folders.create(CommandFolder(name="Игры", profile_id=profile.id, sort_order=1))
    assert work.id is not None and games.id is not None
    sub = repos.folders.create(
        CommandFolder(name="Проекты", profile_id=profile.id, parent_id=work.id)
    )
    assert sub.id is not None
    light = repos.commands.create(
        Command(name="Свет", profile_id=profile.id, folder_id=work.id, priority=10)
    )
    repos.commands.create(Command(name="Тьма", profile_id=profile.id, folder_id=sub.id))
    repos.commands.create(
        Command(name="Игра", profile_id=profile.id, folder_id=games.id, enabled=False)
    )
    root = repos.commands.create(Command(name="Корень", profile_id=profile.id))
    assert light.id is not None and root.id is not None
    repos.commands.get(light.id)
    return CommandTreeStore(repos, profile.id)


def _folder_index(model: CommandTreeModel, name: str) -> QModelIndex:
    for row in range(model.rowCount(QModelIndex())):
        index = model.index(row, 0, QModelIndex())
        if index.data(KIND_ROLE) == str(NodeKind.FOLDER) and index.data(
            Qt.ItemDataRole.DisplayRole
        ).startswith(name):
            return index
    raise AssertionError(f"папка {name!r} не найдена")


def _folder_id(store: CommandTreeStore, name: str) -> int:
    for folder in store.folders():
        if folder.name == name and folder.id is not None:
            return folder.id
    raise AssertionError(name)


def _command_id(store: CommandTreeStore, name: str) -> int:
    for command in store.commands():
        if command.name == name and command.id is not None:
            return command.id
    raise AssertionError(name)


# ----------------------------------------------------------------------
# building and counters
# ----------------------------------------------------------------------


def test_tree_lists_folders_and_root_commands(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    # Two root folders (Работа, Игры) and one root command (Корень).
    assert model.rowCount(QModelIndex()) == 3
    work = _folder_index(model, "Работа")
    assert work.data(KIND_ROLE) == str(NodeKind.FOLDER)


def test_folder_counter_includes_nested(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    work = _folder_index(model, "Работа")
    # «Работа» holds «Свет» and, in «Проекты», «Тьма» — two in the subtree.
    assert work.data(COUNT_ROLE) == 2
    assert work.data(Qt.ItemDataRole.DisplayRole) == "Работа (2)"


def test_children_order_folders_before_commands(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    work = _folder_index(model, "Работа")
    first = model.index(0, 0, work)
    assert first.data(KIND_ROLE) == str(NodeKind.FOLDER)  # «Проекты» sorts before «Свет»


# ----------------------------------------------------------------------
# moves persist in the database
# ----------------------------------------------------------------------


def test_move_command_between_folders_persists(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    games = _folder_id(store, "Игры")
    light = _command_id(store, "Свет")
    store.move_command(light, games)
    model.reload()
    moved = store._repos.commands.get(light)
    assert moved is not None and moved.folder_id == games


def test_reorder_folders_persists(app: QApplication, store: CommandTreeStore) -> None:
    work = _folder_id(store, "Работа")
    games = _folder_id(store, "Игры")
    store.reorder_folders([games, work])
    by_id = {f.id: f for f in store.folders()}
    assert by_id[games].sort_order < by_id[work].sort_order


def test_folder_cannot_move_into_itself(app: QApplication, store: CommandTreeStore) -> None:
    work = _folder_id(store, "Работа")
    sub = _folder_id(store, "Проекты")
    from ayris.core.errors import AyrisError

    with pytest.raises(AyrisError):
        store.move_folder(work, sub)  # «Работа» under its own child «Проекты»


def test_drop_command_onto_folder_moves_it(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    root_command = _command_id(store, "Корень")
    root_index = QModelIndex()
    command_index = next(
        model.index(row, 0, root_index)
        for row in range(model.rowCount(root_index))
        if model.index(row, 0, root_index).data(ENTITY_ID_ROLE) == root_command
        and model.index(row, 0, root_index).data(KIND_ROLE) == str(NodeKind.COMMAND)
    )
    mime = model.mimeData([command_index])
    games_index = _folder_index(model, "Игры")
    assert model.dropMimeData(mime, Qt.DropAction.MoveAction, -1, 0, games_index)
    moved = store._repos.commands.get(root_command)
    assert moved is not None and moved.folder_id == _folder_id(store, "Игры")


def test_drop_folder_into_own_subtree_is_rejected(
    app: QApplication, store: CommandTreeStore
) -> None:
    model = CommandTreeModel(store)
    rejected: list[str] = []
    model.drop_rejected.connect(rejected.append)
    work_index = _folder_index(model, "Работа")
    work_id = int(work_index.data(ENTITY_ID_ROLE))
    mime = QMimeData()
    from PySide6.QtCore import QByteArray, QDataStream, QIODevice

    payload = QByteArray()
    stream = QDataStream(payload, QIODevice.OpenModeFlag.WriteOnly)
    stream.writeInt32(1)
    stream.writeQString(str(NodeKind.FOLDER))
    stream.writeInt64(work_id)
    from ayris.gui.widgets.command_tree_model import TREE_MIME_TYPE

    mime.setData(TREE_MIME_TYPE, payload)
    sub_index = model.index(0, 0, work_index)  # «Проекты»
    assert not model.dropMimeData(mime, Qt.DropAction.MoveAction, -1, 0, sub_index)
    assert rejected  # a Russian message reached the widget


# ----------------------------------------------------------------------
# search and filters
# ----------------------------------------------------------------------


def test_filter_by_name_prunes_to_matches(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(text="тьма"))
    # Only «Работа» survives (it holds «Проекты/Тьма»); «Игры» and root drop out.
    assert model.rowCount(QModelIndex()) == 1
    work = model.index(0, 0, QModelIndex())
    sub = model.index(0, 0, work)
    assert sub.data(Qt.ItemDataRole.DisplayRole).startswith("Проекты")


def test_filter_by_trigger_phrase(app: QApplication, store: CommandTreeStore) -> None:
    repos = store._repos
    repos.triggers.add_voice(_command_id(store, "Свет"), "включи освещение")
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(text="освещение"))
    assert model.rowCount(QModelIndex()) == 1  # «Работа» kept for «Свет»


def test_filter_by_status_disabled(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(status=StatusFilter.DISABLED))
    # Only «Игра» is disabled; only «Игры» survives.
    assert model.rowCount(QModelIndex()) == 1
    assert model.index(0, 0, QModelIndex()).data(Qt.ItemDataRole.DisplayRole).startswith("Игры")


def test_filter_by_tag(app: QApplication, store: CommandTreeStore) -> None:
    from dataclasses import replace

    repos = store._repos
    command = repos.commands.get(_command_id(store, "Корень"))
    assert command is not None
    repos.commands.update(replace(command, tags=("важное",)))
    model = CommandTreeModel(store)
    assert "важное" in model.all_tags()
    model.set_filter(TreeFilter(tag="важное"))
    assert model.rowCount(QModelIndex()) == 1


def test_filter_only_conflicts(app: QApplication, store: CommandTreeStore) -> None:
    repos = store._repos
    repos.triggers.add_voice(_command_id(store, "Свет"), "айрис старт")
    repos.triggers.add_voice(_command_id(store, "Корень"), "айрис старт")
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(only_conflicts=True))
    # «Свет» (in «Работа») and «Корень» conflict; both survive their paths.
    names = _visible_command_names(model)
    assert names == {"Свет", "Корень"}


# ----------------------------------------------------------------------
# conflicts and disabled indication
# ----------------------------------------------------------------------


def test_conflict_role_and_tooltip(app: QApplication, store: CommandTreeStore) -> None:
    repos = store._repos
    repos.triggers.add_voice(_command_id(store, "Свет"), "айрис один")
    repos.triggers.add_voice(_command_id(store, "Корень"), "айрис один")
    model = CommandTreeModel(store)
    root_index = _root_command_index(model, "Корень")
    conflicts = root_index.data(CONFLICT_ROLE)
    assert conflicts == ("Свет",)
    tooltip = root_index.data(Qt.ItemDataRole.ToolTipRole)
    assert "Свет" in tooltip


def test_hotkey_conflict_detected(app: QApplication, store: CommandTreeStore) -> None:
    from ayris.core.models import Trigger, TriggerType

    repos = store._repos
    for name in ("Свет", "Корень"):
        repos.triggers.add(
            Trigger(
                command_id=_command_id(store, name),
                type=TriggerType.HOTKEY,
                payload={"combo": "ctrl+alt+k"},
            )
        )
    conflicts = store.conflicts()
    assert set(conflicts) == {_command_id(store, "Свет"), _command_id(store, "Корень")}


def test_disabled_command_reported_by_role(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    games = _folder_index(model, "Игры")
    game = model.index(0, 0, games)
    assert game.data(ENABLED_ROLE) is False


def test_set_enabled_refreshes_without_reset(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    resets = []
    model.modelReset.connect(lambda: resets.append(1))
    game = model.index(0, 0, _folder_index(model, "Игры"))
    model.set_enabled(int(game.data(ENTITY_ID_ROLE)), enabled=True)
    assert game.data(ENABLED_ROLE) is True
    assert not resets  # a single toggle is a dataChanged, not a whole rebuild


# ----------------------------------------------------------------------
# rename, duplicate, delete
# ----------------------------------------------------------------------


def test_rename_via_setdata(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    index = _root_command_index(model, "Корень")
    assert model.setData(index, "Основа", Qt.ItemDataRole.EditRole)
    assert store._repos.commands.get(_command_id(store, "Основа")) is not None


def test_duplicate_names_and_drops_hotkey(app: QApplication, store: CommandTreeStore) -> None:
    from ayris.core.models import Trigger, TriggerType

    repos = store._repos
    light = _command_id(store, "Свет")
    repos.triggers.add_voice(light, "айрис свет")
    repos.triggers.add(
        Trigger(command_id=light, type=TriggerType.HOTKEY, payload={"combo": "ctrl+l"})
    )
    copy = store.duplicate_command(light)
    assert copy is not None and copy.name == "Свет — копия"
    assert copy.id is not None
    kinds = {t.type for t in repos.triggers.list_for_command(copy.id)}
    assert TriggerType.VOICE in kinds
    assert TriggerType.HOTKEY not in kinds  # busy hotkey not duplicated


def test_delete_folder_moves_commands_to_root(app: QApplication, store: CommandTreeStore) -> None:
    work = _folder_id(store, "Работа")
    inside = {c.name for c in store.commands_in_subtree(work)}
    assert inside == {"Свет", "Тьма"}
    store.delete_folder(work)
    light = store._repos.commands.get_by_name(store.profile_id, "Свет")
    dark = store._repos.commands.get_by_name(store.profile_id, "Тьма")
    assert light is not None and light.folder_id is None
    assert dark is not None and dark.folder_id is None  # subtree command kept, moved to root


def test_bulk_assign_tag(app: QApplication, store: CommandTreeStore) -> None:
    ids = [_command_id(store, "Свет"), _command_id(store, "Корень")]
    assert store.assign_tag(ids, "набор") == 2
    assert store.assign_tag(ids, "набор") == 0  # idempotent, already tagged


# ----------------------------------------------------------------------
# export / import round-trip
# ----------------------------------------------------------------------


def test_export_import_round_trip(app: QApplication, store: CommandTreeStore) -> None:
    repos = store._repos
    repos.triggers.add_voice(_command_id(store, "Свет"), "айрис свет")
    text = store.export_command(_command_id(store, "Свет"))
    games = _folder_id(store, "Игры")
    outcome = store.apply_import(text, target_folder_id=games, strategy=ConflictStrategy.RENAME)
    assert outcome.imported == 1
    assert outcome.renamed == 1  # «Свет» exists, imported copy is renamed
    imported = repos.commands.get_by_name(store.profile_id, "Свет 2")
    assert imported is not None
    # The command's own path («Работа») is recreated under the chosen folder «Игры».
    new_folder = repos.folders.get(imported.folder_id) if imported.folder_id else None
    assert new_folder is not None and new_folder.name == "Работа" and new_folder.parent_id == games


def test_import_skip_strategy(app: QApplication, store: CommandTreeStore) -> None:
    text = store.export_command(_command_id(store, "Свет"))
    outcome = store.apply_import(text, target_folder_id=None, strategy=ConflictStrategy.SKIP)
    assert outcome.imported == 0 and outcome.skipped == 1


def test_import_replace_strategy(app: QApplication, store: CommandTreeStore) -> None:
    before = _command_id(store, "Свет")
    text = store.export_command(before)
    outcome = store.apply_import(text, target_folder_id=None, strategy=ConflictStrategy.REPLACE)
    assert outcome.replaced == 1 and outcome.imported == 1
    again = store._repos.commands.get_by_name(store.profile_id, "Свет")
    assert again is not None and again.id != before  # a fresh row replaced the old one


def test_export_folder_carries_subtree(app: QApplication, store: CommandTreeStore) -> None:
    from ayris.actions.macros.serializer import load_document

    text = store.export_folder(_folder_id(store, "Работа"))
    document = load_document(text)
    names = {c.name for c in document.commands}
    assert names == {"Свет", "Тьма"}


# ----------------------------------------------------------------------
# performance on a large tree
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_filter_on_large_tree_is_fast(app: QApplication, repos: Repositories) -> None:
    profile = repos.profiles.create("Большой", activate=True)
    assert profile.id is not None
    folder = repos.folders.create(CommandFolder(name="Все", profile_id=profile.id))
    assert folder.id is not None
    for number in range(1200):
        repos.commands.create(
            Command(name=f"Команда {number}", profile_id=profile.id, folder_id=folder.id)
        )
    store = CommandTreeStore(repos, profile.id)
    model = CommandTreeModel(store)
    start = time.perf_counter()
    model.set_filter(TreeFilter(text="Команда 999"))
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    assert _visible_command_names(model) == {"Команда 999"}


# ----------------------------------------------------------------------
# the widget: signals and filter wiring
# ----------------------------------------------------------------------


def test_widget_emits_command_activated(app: QApplication, store: CommandTreeStore) -> None:
    tree = CommandTree(store, ThemeManager(app))
    seen: list[int] = []
    tree.command_activated.connect(seen.append)
    index = _root_command_index(tree.model, "Корень")
    tree._view.setCurrentIndex(index)
    assert seen and seen[-1] == _command_id(store, "Корень")
    tree.close()


def test_widget_search_filters_model(app: QApplication, store: CommandTreeStore) -> None:
    tree = CommandTree(store, ThemeManager(app))
    tree._search.setText("тьма")
    assert tree.model.rowCount(QModelIndex()) == 1
    tree.close()


def test_widget_tree_changed_after_store_op(app: QApplication, store: CommandTreeStore) -> None:
    tree = CommandTree(store, ThemeManager(app))
    fired: list[int] = []
    tree.tree_changed.connect(lambda: fired.append(1))
    tree._bulk_enable([_command_id(store, "Свет")], False)
    assert fired
    tree.close()


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _visible_command_names(model: CommandTreeModel) -> set[str]:
    names: set[str] = set()

    def walk(parent: QModelIndex) -> None:
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            if index.data(KIND_ROLE) == str(NodeKind.COMMAND):
                names.add(str(index.data(Qt.ItemDataRole.DisplayRole)))
            else:
                walk(index)

    walk(QModelIndex())
    return names


def _root_command_index(model: CommandTreeModel, name: str) -> QModelIndex:
    for row in range(model.rowCount(QModelIndex())):
        index = model.index(row, 0, QModelIndex())
        if (
            index.data(KIND_ROLE) == str(NodeKind.COMMAND)
            and index.data(Qt.ItemDataRole.DisplayRole) == name
        ):
            return index
    raise AssertionError(f"команда {name!r} не найдена в корне")
