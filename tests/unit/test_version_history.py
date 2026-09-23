"""Виджет истории версий команды (задача 54), offscreen.

Слева — таблица версий (номер, дата, автор, правка, размер, важность), справа —
структурный дифф выбранной версии против текущей команды, посчитанный на моделях.
Ни один пиксель не проверяется: утверждения — про строки таблицы, содержимое
диффа, сигнал ``rollback_requested`` с пересобранной моделью, закрепление важной
версии и экспорт в ``.ayris``. Модальный диалог отката подменён на «принять», иначе
offscreen-прогон завис бы на ``exec``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.core.database import Database
from ayris.core.models import Command
from ayris.core.repositories import Repositories
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import confirm_dialog as confirm_dialog_module
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.version_history import VersionHistory

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
    return CommandTreeStore(repos, profile.id, version_limit=5)


def _command_id(store: CommandTreeStore, name: str) -> int:
    for command in store.commands():
        if command.name == name and command.id is not None:
            return command.id
    raise AssertionError(name)


def _make_versions(store: CommandTreeStore, command_id: int) -> CommandModel:
    """Дважды сохранить команду, накопив версии; вернуть текущую модель."""
    model = store.command_model(command_id)
    model.name = "Свет 2"
    model.actions = [ActionBlock(type="Say", params={"text": "раз"})]
    store.save_command(model)
    current = store.command_model(command_id)
    current.name = "Свет 3"
    current.actions = [ActionBlock(type="Say", params={"text": "два"})]
    return store.save_command(current)


# ----------------------------------------------------------------------
# наполнение таблицы
# ----------------------------------------------------------------------


def test_load_fills_version_table(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    assert widget._table.rowCount() == len(store.versions(cid))
    assert widget._table.rowCount() >= 2


def test_empty_when_no_command(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager
) -> None:
    widget = VersionHistory(store, theme)
    widget.refresh()  # команда не открыта — тихий выход
    assert widget._table.rowCount() == 0


# ----------------------------------------------------------------------
# дифф выбранной версии против текущей
# ----------------------------------------------------------------------


def test_selecting_version_shows_diff_against_current(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    # Выбрать самую старую версию (последняя строка): она заметно отличается.
    widget._table.selectRow(widget._table.rowCount() - 1)
    app.processEvents()
    assert widget._diff.topLevelItemCount() > 0
    assert "Отличий" in widget._diff_summary.text()


# ----------------------------------------------------------------------
# откат
# ----------------------------------------------------------------------


def test_rollback_emits_rebuilt_model_with_live_id(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    # Подтвердить откат без модального окна.
    monkeypatch.setattr(
        ConfirmDialog, "exec", lambda _self: confirm_dialog_module.QDialog.DialogCode.Accepted
    )
    emitted: list[object] = []
    widget.rollback_requested.connect(emitted.append)

    widget._table.selectRow(widget._table.rowCount() - 1)  # старая версия
    widget._rollback()

    assert len(emitted) == 1
    model = emitted[0]
    assert isinstance(model, CommandModel)
    assert model.id == cid  # живой id, не устаревший
    assert model.name == "Свет"  # содержимое самой старой версии


def test_rollback_cancelled_emits_nothing(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    monkeypatch.setattr(
        ConfirmDialog, "exec", lambda _self: confirm_dialog_module.QDialog.DialogCode.Rejected
    )
    emitted: list[object] = []
    widget.rollback_requested.connect(emitted.append)
    widget._table.selectRow(widget._table.rowCount() - 1)
    widget._rollback()
    assert emitted == []  # отменили — модель не уходит


# ----------------------------------------------------------------------
# важные версии
# ----------------------------------------------------------------------


def test_toggle_important_pins_version(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    widget._table.selectRow(0)
    widget._toggle_important()
    # Версия закреплена — читается из хранилища.
    version = store.versions(cid)[0]
    assert version.important is True


# ----------------------------------------------------------------------
# экспорт
# ----------------------------------------------------------------------


def test_export_writes_ayris_file(
    app: QApplication,
    store: CommandTreeStore,
    theme: ThemeManager,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    target = tmp_path / "снимок.ayris"
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), ""))
    statuses: list[str] = []
    widget.status.connect(statuses.append)

    widget._table.selectRow(0)
    widget._export()

    assert target.exists()
    assert '"kind"' in target.read_text(encoding="utf-8")
    assert statuses and "экспортирована" in statuses[-1]


# ----------------------------------------------------------------------
# смена профиля
# ----------------------------------------------------------------------


def test_set_store_clears_view(
    app: QApplication, store: CommandTreeStore, theme: ThemeManager, repos: Repositories
) -> None:
    cid = _command_id(store, "Свет")
    current = _make_versions(store, cid)
    widget = VersionHistory(store, theme)
    widget.load(cid, current)
    assert widget._table.rowCount() > 0

    other = repos.profiles.create("Второй", activate=False)
    assert other.id is not None
    widget.set_store(CommandTreeStore(repos, other.id))
    assert widget._table.rowCount() == 0
