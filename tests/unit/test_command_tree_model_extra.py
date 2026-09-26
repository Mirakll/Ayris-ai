"""Дополнительное покрытие логики :mod:`command_tree_model` без окон и вьюх.

Здесь добираются ветки :class:`CommandTreeModel` и его хранилища
:class:`CommandTreeStore`, которые основной набор ``test_command_tree`` не
трогает: пустые/невалидные индексы модели, роли data/setData, помощники drag &
drop и одиночное обновление строки, а со стороны хранилища — резюме импорта,
редкие ветки конфликтов, переименований, дублирования, экспорта, импорта и
уникальных имён. Ни одного настоящего виджета-вьюхи, ни ``exec``, ни живых
таймеров: модель строится над базой в памяти и проверяется по состоянию.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QMimeData, QModelIndex, Qt
from PySide6.QtWidgets import QApplication

from ayris.core.database import Database
from ayris.core.errors import AyrisError
from ayris.core.models import Command, CommandFolder, Trigger, TriggerType, VariableScope
from ayris.core.repositories import Repositories
from ayris.gui.widgets.command_tree_model import (
    KIND_ROLE,
    TREE_MIME_TYPE,
    CommandTreeModel,
    CommandTreeStore,
    ConflictStrategy,
    ImportOutcome,
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
    repos.commands.create(
        Command(name="Свет", profile_id=profile.id, folder_id=work.id, priority=10)
    )
    repos.commands.create(Command(name="Тьма", profile_id=profile.id, folder_id=sub.id))
    repos.commands.create(
        Command(name="Игра", profile_id=profile.id, folder_id=games.id, enabled=False)
    )
    repos.commands.create(Command(name="Корень", profile_id=profile.id))
    return CommandTreeStore(repos, profile.id)


def _folder_index(model: CommandTreeModel, name: str) -> QModelIndex:
    for row in range(model.rowCount(QModelIndex())):
        index = model.index(row, 0, QModelIndex())
        display = index.data(Qt.ItemDataRole.DisplayRole)
        if index.data(KIND_ROLE) == str(NodeKind.FOLDER) and display.startswith(name):
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


def _root_command_index(model: CommandTreeModel, name: str) -> QModelIndex:
    for row in range(model.rowCount(QModelIndex())):
        index = model.index(row, 0, QModelIndex())
        if (
            index.data(KIND_ROLE) == str(NodeKind.COMMAND)
            and index.data(Qt.ItemDataRole.DisplayRole) == name
        ):
            return index
    raise AssertionError(f"команда {name!r} не найдена в корне")


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


# ----------------------------------------------------------------------
# store: сводка импорта, версии, редкие ветки конфликтов и имён
# ----------------------------------------------------------------------


def test_import_outcome_summary_lists_every_change() -> None:
    full = ImportOutcome(imported=2, renamed=1, replaced=3, skipped=4, folders=5)
    summary = full.summary
    assert "добавлено команд: 2" in summary
    assert "переименовано: 1" in summary
    assert "заменено: 3" in summary
    assert "пропущено: 4" in summary
    assert "новых папок: 5" in summary
    # Ничего кроме добавленных — все прочие ветки не срабатывают.
    assert ImportOutcome(imported=3).summary == "добавлено команд: 3"


def test_version_limit_reports_configured_value(repos: Repositories) -> None:
    profile = repos.profiles.create("Лимит", activate=True)
    assert profile.id is not None
    tight = CommandTreeStore(repos, profile.id, version_limit=7)
    assert tight.version_limit == 7


def test_event_trigger_is_neither_conflict_nor_phrase(store: CommandTreeStore) -> None:
    repos = store._repos
    light = _command_id(store, "Свет")
    # Событийный триггер не сравнивается — ключ у него отсутствует.
    repos.triggers.add(
        Trigger(command_id=light, type=TriggerType.EVENT, payload={"event": "запуск"})
    )
    assert store.conflicts() == {}
    assert light not in store.phrases_by_command()


def test_rename_missing_command_is_silent(store: CommandTreeStore) -> None:
    before = {c.name for c in store.commands()}
    store.rename_command(999_999, "Никого")  # get() вернёт None → тихий выход
    assert {c.name for c in store.commands()} == before


def test_is_descendant_handles_self_and_deep_walk(store: CommandTreeStore) -> None:
    work = _folder_id(store, "Работа")
    sub = _folder_id(store, "Проекты")
    games = _folder_id(store, "Игры")
    assert store.is_descendant(work, work) is True  # папка — свой же потомок
    # «Проекты» не лежит внутри «Игры»: обход поднимается до корня и возвращает False.
    assert store.is_descendant(sub, games) is False


def test_move_folder_reparents_and_reorders(store: CommandTreeStore) -> None:
    sub = _folder_id(store, "Проекты")
    games = _folder_id(store, "Игры")
    store.move_folder(sub, games)
    moved = next(f for f in store.folders() if f.id == sub)
    assert moved.parent_id == games


def test_duplicate_missing_command_returns_none(store: CommandTreeStore) -> None:
    assert store.duplicate_command(999_999) is None


def test_assign_tag_ignores_blank(store: CommandTreeStore) -> None:
    assert store.assign_tag([_command_id(store, "Свет")], "   ") == 0


def test_sibling_names_of_missing_command_is_empty(store: CommandTreeStore) -> None:
    assert store.sibling_names(999_999) == set()


def test_trigger_conflicts_names_other_owners(store: CommandTreeStore) -> None:
    repos = store._repos
    light = _command_id(store, "Свет")
    root = _command_id(store, "Корень")
    repos.triggers.add_voice(light, "общая фраза")
    repos.triggers.add_voice(root, "общая фраза")
    # Для «Свет» перечисляются владельцы того же ключа, кроме себя.
    conflicts = store.trigger_conflicts(light)
    assert conflicts[("voice", "общая фраза")] == ("Корень",)


def test_save_command_syncs_scoped_declarations(app: QApplication, store: CommandTreeStore) -> None:
    from ayris.actions.macros.schema import VariableModel
    from ayris.core.models import VariableType

    repos = store._repos
    # Уже существующую профильную переменную сохранение не трогает; новую заводит.
    repos.variables.set(
        "существует",
        "старое",
        scope=VariableScope.PROFILE,
        profile_id=store.profile_id,
        var_type=VariableType.STRING,
        persistent=False,
    )
    model = store.command_model(_command_id(store, "Свет"))
    edited = model.model_copy(
        update={
            "variables": [
                VariableModel(name="локальная", scope=VariableScope.LOCAL),
                VariableModel(name="существует", scope=VariableScope.PROFILE, default="новое"),
                VariableModel(name="новая", scope=VariableScope.PROFILE, default="x"),
            ]
        }
    )
    store.save_command(edited)
    created = repos.variables.get("новая", scope=VariableScope.PROFILE, profile_id=store.profile_id)
    kept = repos.variables.get(
        "существует", scope=VariableScope.PROFILE, profile_id=store.profile_id
    )
    assert created is not None and created.value == "x"
    assert kept is not None and kept.value == "старое"


def test_version_model_missing_raises(store: CommandTreeStore) -> None:
    with pytest.raises(AyrisError):
        store.version_model(_command_id(store, "Свет"), 999)


def test_export_commands_bundles_many(store: CommandTreeStore) -> None:
    text = store.export_commands([_command_id(store, "Свет"), _command_id(store, "Тьма")])
    assert "Свет" in text and "Тьма" in text


def test_apply_import_new_command_without_conflict(store: CommandTreeStore) -> None:
    root_id = _command_id(store, "Корень")
    text = store.export_command(root_id)
    store.delete_command(root_id)
    outcome = store.apply_import(text, target_folder_id=None, strategy=ConflictStrategy.RENAME)
    assert outcome.imported == 1
    assert outcome.renamed == 0 and outcome.replaced == 0
    assert store._repos.commands.get_by_name(store.profile_id, "Корень") is not None


def test_apply_import_reuses_existing_folder(store: CommandTreeStore) -> None:
    # Путь «Игры» уже есть: обход перебирает «Работа» (мимо) и находит «Игры».
    text = store.export_command(_command_id(store, "Игра"))
    outcome = store.apply_import(text, target_folder_id=None, strategy=ConflictStrategy.RENAME)
    assert outcome.folders == 0  # ни одной новой папки — переиспользована старая
    assert outcome.renamed == 1
    imported = store._repos.commands.get_by_name(store.profile_id, "Игра 2")
    assert imported is not None and imported.folder_id == _folder_id(store, "Игры")


def test_unique_name_increments_past_taken(store: CommandTreeStore) -> None:
    assert store.create_command(None, "Свет").name == "Свет 2"
    # «Свет» и «Свет 2» заняты — цикл доходит до «Свет 3».
    assert store.create_command(None, "Свет").name == "Свет 3"


# ----------------------------------------------------------------------
# model: построение, фильтр, роли и невалидные индексы
# ----------------------------------------------------------------------


class _StubStore:
    """Хранилище-двойник: отдаёт сущности без id, чтобы проверить их отсев."""

    def folders(self) -> list[CommandFolder]:
        return [CommandFolder(name="призрак"), CommandFolder(name="реальная", id=7)]

    def commands(self) -> list[Command]:
        return [
            Command(name="ничья", profile_id=1),
            Command(name="есть", profile_id=1, id=3, folder_id=7),
        ]

    def conflicts(self) -> dict[int, tuple[str, ...]]:
        return {}

    def phrases_by_command(self) -> dict[int, tuple[str, ...]]:
        return {}


def test_build_skips_entities_without_id(app: QApplication) -> None:
    model = CommandTreeModel(_StubStore())  # type: ignore[arg-type]
    # Папка и команда без id пропущены: в корне остаётся одна настоящая папка.
    assert model.rowCount(QModelIndex()) == 1
    only = model.index(0, 0, QModelIndex())
    assert only.data(KIND_ROLE) == str(NodeKind.FOLDER)


def test_filter_enabled_hides_disabled(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(status=StatusFilter.ENABLED))
    names = _visible_command_names(model)
    assert "Игра" not in names  # единственная выключенная отсеяна
    assert "Свет" in names


def test_invalid_index_returns_defaults(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    assert not model.parent(QModelIndex()).isValid()
    assert model.flags(QModelIndex()) == Qt.ItemFlag.ItemIsDropEnabled
    assert model.data(QModelIndex()) is None
    assert model.setData(QModelIndex(), "x", int(Qt.ItemDataRole.EditRole)) is False


def test_data_edit_role_and_empty_tooltip(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    index = _root_command_index(model, "Корень")
    assert model.data(index, int(Qt.ItemDataRole.EditRole)) == "Корень"
    # Без конфликтов подсказки нет.
    assert model.data(index, int(Qt.ItemDataRole.ToolTipRole)) is None


def test_dirty_bullet_marks_and_clears(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    root_id = _command_id(store, "Корень")
    index = model.index_for(NodeKind.COMMAND, root_id)
    assert index.isValid()
    model.set_dirty_command(root_id)
    assert model.data(index, int(Qt.ItemDataRole.DisplayRole)) == "● Корень"
    model.set_dirty_command(root_id)  # тот же id — ранний выход, без изменений
    model.set_dirty_command(None)  # снятие метки
    assert model.data(index, int(Qt.ItemDataRole.DisplayRole)) == "Корень"


# ----------------------------------------------------------------------
# model: setData
# ----------------------------------------------------------------------


def test_setdata_rejects_empty_same_and_wrong_role(
    app: QApplication, store: CommandTreeStore
) -> None:
    model = CommandTreeModel(store)
    index = _root_command_index(model, "Корень")
    assert model.setData(index, "   ", int(Qt.ItemDataRole.EditRole)) is False  # пусто
    assert model.setData(index, "Корень", int(Qt.ItemDataRole.EditRole)) is False  # то же имя
    assert model.setData(index, "Другое", KIND_ROLE) is False  # не EditRole


def test_setdata_renames_folder(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    folder = _folder_index(model, "Работа")
    assert model.setData(folder, "Работа2", int(Qt.ItemDataRole.EditRole)) is True
    assert any(f.name == "Работа2" for f in store.folders())


def test_setdata_swallows_ayris_error(
    app: QApplication, store: CommandTreeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(_command_id: int, _name: str) -> None:
        raise AyrisError("не вышло", user_message="Не удалось.")

    monkeypatch.setattr(store, "rename_command", _boom)
    model = CommandTreeModel(store)
    index = _root_command_index(model, "Корень")
    assert model.setData(index, "Новое имя", int(Qt.ItemDataRole.EditRole)) is False


# ----------------------------------------------------------------------
# model: drag & drop
# ----------------------------------------------------------------------


def test_mime_type_and_supported_actions(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    assert model.supportedDropActions() == Qt.DropAction.MoveAction
    assert model.mimeTypes() == [TREE_MIME_TYPE]


def test_drop_ignore_and_bad_format(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    assert model.dropMimeData(QMimeData(), Qt.DropAction.IgnoreAction, -1, 0, QModelIndex()) is True
    # Move без нужного mime-типа — отказ.
    assert model.dropMimeData(QMimeData(), Qt.DropAction.MoveAction, -1, 0, QModelIndex()) is False


def test_empty_drop_makes_no_change(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    mime = model.mimeData([])  # ноль строк
    assert model.dropMimeData(mime, Qt.DropAction.MoveAction, -1, 0, QModelIndex()) is False


def test_drop_reorders_root_folders(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    mime = model.mimeData([_folder_index(model, "Работа")])
    # Ставим «Работа» после «Игры» в корне.
    assert model.dropMimeData(mime, Qt.DropAction.MoveAction, 1, 0, QModelIndex())
    by_id = {f.id: f for f in store.folders()}
    games = _folder_id(store, "Игры")
    work = _folder_id(store, "Работа")
    assert by_id[games].sort_order < by_id[work].sort_order


def test_drop_moves_folder_into_folder(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    work_index = _folder_index(model, "Работа")
    sub_index = model.index(0, 0, work_index)  # «Проекты»
    assert sub_index.data(KIND_ROLE) == str(NodeKind.FOLDER)
    mime = model.mimeData([sub_index])
    assert model.dropMimeData(mime, Qt.DropAction.MoveAction, -1, 0, _folder_index(model, "Игры"))
    sub = _folder_id(store, "Проекты")
    games = _folder_id(store, "Игры")
    moved = next(f for f in store.folders() if f.id == sub)
    assert moved.parent_id == games


# ----------------------------------------------------------------------
# model: точечное обновление строки
# ----------------------------------------------------------------------


def test_refresh_unknown_command_reloads(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    resets: list[int] = []
    model.modelReset.connect(lambda: resets.append(1))
    model.refresh_command(999_999)  # узла нет → полный reload
    assert resets


def test_refresh_deleted_command_reloads(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    command_id = _command_id(store, "Корень")
    store.delete_command(command_id)  # строка исчезла из базы, но узел ещё в модели
    resets: list[int] = []
    model.modelReset.connect(lambda: resets.append(1))
    model.refresh_command(command_id)  # get() вернёт None → reload
    assert resets
    assert store._repos.commands.get_by_name(store.profile_id, "Корень") is None


def test_refresh_filtered_out_command_skips_emit(
    app: QApplication, store: CommandTreeStore
) -> None:
    model = CommandTreeModel(store)
    model.set_filter(TreeFilter(status=StatusFilter.ENABLED))  # прячет «Игра»
    changed: list[int] = []
    model.dataChanged.connect(lambda *_: changed.append(1))
    model.refresh_command(_command_id(store, "Игра"))  # индекс невалиден → без dataChanged
    assert not changed


def test_set_store_switches_profile(app: QApplication, store: CommandTreeStore) -> None:
    model = CommandTreeModel(store)
    other = store._repos.profiles.create("Второй")
    assert other.id is not None
    store._repos.commands.create(Command(name="Одна", profile_id=other.id))
    replacement = CommandTreeStore(store._repos, other.id)
    model.set_store(replacement)
    assert model.store is replacement
    assert model.rowCount(QModelIndex()) == 1  # единственная команда нового профиля
