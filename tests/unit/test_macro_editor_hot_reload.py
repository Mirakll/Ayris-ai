"""Интеграция задачи 54 в редактор команд, offscreen.

Здесь проверяется склейка, а не чистые слои (для них — ``test_editor_undo``,
``test_command_diff``, ``test_hot_reload``, ``test_draft_store``): отмена/повтор
через сам редактор применяют модель ко всем секциям; сохранение с шиной проходит
через :class:`HotReloader` и публикует ``CommandReloaded``; ошибка проверки при
сохранении оставляет прежнюю версию, не выключая команду; черновик после «краха»
предлагается к восстановлению; откат из истории сохраняется как новая версия;
страж несохранённого возвращает выбор. Таймер автосейва останавливается в фикстуре,
иначе он пережил бы тест и подвесил CI.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import ActionBlock
from ayris.core.database import Database
from ayris.core.events import CommandReloaded, CommandsChanged, EventBus
from ayris.core.models import Command
from ayris.core.repositories import Repositories
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.draft_store import DraftStore
from ayris.gui.widgets.macro_editor import (
    MacroEditor,
    MacroEditorServices,
    UnsavedChoice,
    _UnsavedDialog,
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


@pytest.fixture
def repos() -> Iterator[Repositories]:
    database = Database.open(":memory:")
    yield Repositories(database)
    database.close()


@pytest.fixture
def store(repos: Repositories) -> CommandTreeStore:
    profile = repos.profiles.create("Основной", activate=True)
    assert profile.id is not None
    repos.commands.create(Command(name="Свет", profile_id=profile.id))
    repos.commands.create(Command(name="Тьма", profile_id=profile.id))
    return CommandTreeStore(repos, profile.id, version_limit=5)


@pytest.fixture(scope="module")
def catalog() -> BlockCatalog:
    return BlockCatalog()


def _command_id(store: CommandTreeStore, name: str) -> int:
    for command in store.commands():
        if command.name == name and command.id is not None:
            return command.id
    raise AssertionError(name)


def _editor(
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    *,
    bus: EventBus | None = None,
    drafts: DraftStore | None = None,
    autosave_s: float = 0.0,
) -> MacroEditor:
    services = MacroEditorServices(
        catalog=catalog,
        bus=bus,
        draft_store=drafts,
        draft_autosave_s=autosave_s,
    )
    editor = MacroEditor(store, theme, services=services)
    return editor


# ----------------------------------------------------------------------
# отмена/повтор через сам редактор
# ----------------------------------------------------------------------


def test_editor_undo_redo_applies_model_to_sections(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._header._name.setText("Свет новый")
    assert editor._undo.can_undo is True

    editor.undo()
    # Модель вернулась к исходной и это видно в секции-заголовке.
    assert editor._model is not None and editor._model.name == "Свет"
    assert editor._header._name.text() == "Свет"

    editor.redo()
    assert editor._model is not None and editor._model.name == "Свет новый"
    assert editor._header._name.text() == "Свет новый"
    editor.stop_autosave()


def test_undo_reset_on_command_switch(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    editor._header._name.setText("Свет 2")
    assert editor._undo.can_undo is True
    # Переключение на другую команду сбрасывает историю (требование задачи 54).
    editor.load_command(_command_id(store, "Тьма"))
    assert editor._undo.can_undo is False
    editor.stop_autosave()


# ----------------------------------------------------------------------
# сохранение с шиной: горячая перерегистрация
# ----------------------------------------------------------------------


def test_save_with_bus_publishes_reloaded(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    bus = EventBus()
    reloaded: list[CommandReloaded] = []
    changed: list[CommandsChanged] = []
    bus.subscribe(CommandReloaded, reloaded.append)
    bus.subscribe(CommandsChanged, changed.append)
    editor = _editor(store, theme, catalog, bus=bus)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    editor._header._name.setText("Свет 2")
    editor.save()

    assert editor._status.text() == "Команда сохранена."
    assert editor.is_dirty is False
    assert reloaded and reloaded[-1].command_id == cid
    assert changed and changed[-1].command_id == cid
    editor.stop_autosave()


def test_invalid_canvas_save_stays_dirty_and_shows_the_bad_block(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    """A block wired in on the canvas but left invalid must not save silently.

    Regression for the «кружок не гаснет» report: an in-flow block with an empty
    required parameter makes ``apply_command`` raise, so ``save`` returns ``False``
    and the command stays dirty (its old version is still what runs). The failure
    must be visible — the editor jumps to «Действия» and selects the offending
    block — not hidden behind an unresponsive «Сохранить».
    """
    from ayris.actions.macros.schema import ActionBlock
    from ayris.gui.widgets.node_editor.bridge import MAIN_PORT, graph_from_command

    bus = EventBus()
    editor = _editor(store, theme, catalog, bus=bus)
    cid = _command_id(store, "Свет")
    seed = store.command_model(cid)
    seed.actions = [ActionBlock(type="Say", params={"text": "привет"})]
    store.save_command(seed)
    editor.load_command(cid)

    assert editor._model is not None
    root_id = graph_from_command(editor._model, catalog=catalog).root_id
    assert root_id is not None
    editor._node_editor._insert_block("SetVar")  # actions[1], empty required params
    assert editor._node_editor._scene.request_connect(root_id, MAIN_PORT, "actions[1]") is True

    saved = editor.save()
    assert saved is False
    assert editor.is_dirty is True  # not saved — the mark legitimately stays lit
    assert editor._tabs.tabText(editor._tabs.currentIndex()) == "Действия"
    assert editor._current_action_view().selected_path() == ("actions", 1)
    assert "SetVar" in editor._status.text()
    # The old version is untouched in the database.
    assert [b.type for b in store.command_model(cid).actions] == ["Say"]

    # Filling the parameters lets the save go through and clears the mark.
    editor._on_block_selected(("actions", 1))
    editor._param_form.set_values({"name": "x", "value": "1"})
    editor._on_params_changed()
    assert editor.save() is True
    assert editor.is_dirty is False
    editor.stop_autosave()


def test_save_validation_error_keeps_previous_version(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    bus = EventBus()
    reloaded: list[CommandReloaded] = []
    bus.subscribe(CommandReloaded, reloaded.append)
    editor = _editor(store, theme, catalog, bus=bus)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    # Ссылка на необъявленную переменную — ошибка проверки при сохранении.
    editor._model is not None and editor._action_model.set_command(
        editor._model.model_copy(
            update={"actions": [ActionBlock(type="Say", params={"text": "{нет}"})]}
        )
    )
    editor._model.actions = [ActionBlock(type="Say", params={"text": "{нет}"})]
    editor.save()

    # Ничего не перезагрузилось, команда осталась в редакторе для правки.
    assert reloaded == []
    assert editor._model is not None  # модель не сброшена
    # Прежняя запись в БД нетронута.
    assert store.command_model(cid).actions == []
    editor.stop_autosave()


# ----------------------------------------------------------------------
# восстановление черновика
# ----------------------------------------------------------------------


def test_draft_restored_on_open(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    drafts = DraftStore(tmp_path / "drafts")
    cid = _command_id(store, "Свет")
    # «Крах»: черновик с несохранённым текстом лежит на диске.
    draft_model = store.command_model(cid)
    draft_model.actions = [ActionBlock(type="Say", params={"text": "восстановлено"})]
    drafts.save(draft_model)

    editor = _editor(store, theme, catalog, drafts=drafts)
    # Принять восстановление без модального окна.
    from ayris.gui.widgets import confirm_dialog as cd

    monkeypatch.setattr(cd.ConfirmDialog, "exec", lambda _self: cd.QDialog.DialogCode.Accepted)
    editor.load_command(cid)

    assert editor._model is not None
    assert editor._model.actions[0].params["text"] == "восстановлено"
    assert editor.is_dirty is True  # восстановленный черновик отличается от сохранённого
    editor.stop_autosave()


def test_draft_declined_opens_saved(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    drafts = DraftStore(tmp_path / "drafts")
    cid = _command_id(store, "Свет")
    draft_model = store.command_model(cid)
    draft_model.actions = [ActionBlock(type="Say", params={"text": "черновик"})]
    drafts.save(draft_model)

    editor = _editor(store, theme, catalog, drafts=drafts)
    from ayris.gui.widgets import confirm_dialog as cd

    monkeypatch.setattr(cd.ConfirmDialog, "exec", lambda _self: cd.QDialog.DialogCode.Rejected)
    editor.load_command(cid)

    # Отказ — открыта сохранённая (пустая) версия, черновик удалён.
    assert editor._model is not None and editor._model.actions == []
    assert editor.is_dirty is False
    assert drafts.has_draft(cid) is False
    editor.stop_autosave()


def test_autosave_writes_draft_only_when_dirty(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    tmp_path: object,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    drafts = DraftStore(tmp_path / "drafts")
    editor = _editor(store, theme, catalog, drafts=drafts)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    # Чисто — тик автосейва ничего не пишет.
    editor._autosave_draft()
    assert drafts.has_draft(cid) is False
    # Правка — теперь тик пишет черновик.
    editor._header._name.setText("Свет 2")
    editor._autosave_draft()
    assert drafts.has_draft(cid) is True
    editor.stop_autosave()


def test_successful_save_discards_draft(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    tmp_path: object,
) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    drafts = DraftStore(tmp_path / "drafts")
    editor = _editor(store, theme, catalog, drafts=drafts)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    editor._header._name.setText("Свет 2")
    editor._autosave_draft()
    assert drafts.has_draft(cid) is True
    editor.save()
    # После успешного сохранения черновик больше не нужен.
    assert drafts.has_draft(cid) is False
    editor.stop_autosave()


# ----------------------------------------------------------------------
# откат из истории сохраняется как новая версия
# ----------------------------------------------------------------------


def test_rollback_request_saves_as_new_version(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    # История получает версию: меняем и сохраняем.
    editor._header._name.setText("Свет 2")
    editor.save()
    versions_before = len(store.versions(cid))

    # Откат к содержимому старой версии, поданный виджетом истории.
    old = store.version_model(cid, store.versions(cid)[-1].version)
    rolled = old.model_copy(update={"id": cid})
    editor._on_rollback_requested(rolled)

    assert editor._status.text() == "Команда сохранена."
    # Откат применён как обычное сохранение — в истории новая версия, текущее не потеряно.
    assert len(store.versions(cid)) > versions_before
    assert store.command_model(cid).name == old.name
    editor.stop_autosave()


# ----------------------------------------------------------------------
# страж несохранённого
# ----------------------------------------------------------------------


def test_guard_true_when_not_dirty(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = _editor(store, theme, catalog)
    editor.load_command(_command_id(store, "Свет"))
    assert editor.guard_unsaved() is True  # чисто — уходить можно сразу
    editor.stop_autosave()


def test_guard_save_choice_persists_and_allows_leave(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = _editor(store, theme, catalog)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    editor._header._name.setText("Свет 2")
    # Пользователь выбрал «Сохранить».
    monkeypatch.setattr(_UnsavedDialog, "exec", _accept_with(UnsavedChoice.SAVE))
    assert editor.guard_unsaved() is True
    assert editor.is_dirty is False
    assert store.command_model(cid).name == "Свет 2"
    editor.stop_autosave()


def test_guard_cancel_choice_keeps_command(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = _editor(store, theme, catalog)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    editor._header._name.setText("Свет 2")
    monkeypatch.setattr(_UnsavedDialog, "exec", _accept_with(UnsavedChoice.CANCEL))
    assert editor.guard_unsaved() is False  # отмена — остаёмся
    assert editor.is_dirty is True
    editor.stop_autosave()


def test_guard_discard_choice_allows_leave_without_save(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    catalog: BlockCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    editor = _editor(store, theme, catalog)
    cid = _command_id(store, "Свет")
    editor.load_command(cid)
    editor._header._name.setText("Свет 2")
    monkeypatch.setattr(_UnsavedDialog, "exec", _accept_with(UnsavedChoice.DISCARD))
    assert editor.guard_unsaved() is True  # не сохранять — уходим
    assert store.command_model(cid).name == "Свет"  # в БД ничего не записано
    editor.stop_autosave()


def _accept_with(choice: str) -> Callable[[_UnsavedDialog], int]:
    """Фабрика: подменяет ``exec`` диалога, выставляя нужный выбор."""

    def _exec(dialog: _UnsavedDialog) -> int:
        dialog.choice = choice
        return 0

    return _exec
