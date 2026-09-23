"""Version history of one command, with diff, rollback and export — task 54.

Left, a table of the stored versions of the open command: number, date, author
(the editor, or an import), the save comment and the snapshot's size. Right, the
structural diff of the selected version against the current command, computed on
the models by :mod:`ayris.actions.macros.diff` — never on the JSON text. From here
the user can pin a version so the prune never evicts it, roll one back (confirmed
through the task-42 dialog, applied as a fresh save so nothing is lost), and export
one to a ``.ayris`` file.

The widget reads and previews; it does not write the library. A rollback is handed
back to the editor through :attr:`rollback_requested` with the rebuilt model, so the
one save path — validate, version, re-register — stays the editor's, and the tree
and overlay update from the same event a normal save fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.diff import (
    CHANGE_ADDED,
    CHANGE_REMOVED,
    Change,
    diff_commands,
)
from ayris.core.errors import AyrisError
from ayris.gui.widgets.confirm_dialog import ConfirmDialog

if TYPE_CHECKING:
    from ayris.actions.macros.schema import CommandModel
    from ayris.core.models import CommandVersion
    from ayris.gui.theme import ThemeManager
    from ayris.gui.widgets.command_tree_model import CommandTreeStore

__all__ = ["VersionHistory"]


class VersionHistory(QWidget):
    """The version list of one command, its diff, and the rollback / export actions."""

    #: A version was chosen for rollback; carries the rebuilt model for the editor
    #: to save through its own validate → version → re-register path.
    rollback_requested = Signal(object)
    #: A short status line for the host to show (export result, an error).
    status = Signal(str)

    def __init__(
        self,
        store: CommandTreeStore,
        theme: ThemeManager,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._store = store
        self._theme = theme
        self._command_id: int | None = None
        self._current: CommandModel | None = None
        self._versions: list[CommandVersion] = []

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_md"))

        outer.addLayout(self._build_list(), 2)
        outer.addLayout(self._build_diff(), 3)

    # -- construction -------------------------------------------------------

    def _build_list(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(self._theme.metric("spacing_sm"))

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(_COL_COMMENT, QHeaderView.ResizeMode.Stretch)
        for column_index in (_COL_VERSION, _COL_DATE, _COL_AUTHOR, _COL_SIZE, _COL_PIN):
            header.setSectionResizeMode(column_index, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemSelectionChanged.connect(self._on_selection)
        column.addWidget(self._table)

        buttons = QHBoxLayout()
        self._pin_button = QPushButton("Пометить важной")
        self._pin_button.clicked.connect(self._toggle_important)
        self._rollback_button = QPushButton("Откатить…")
        self._rollback_button.clicked.connect(self._rollback)
        self._export_button = QPushButton("Экспорт…")
        self._export_button.clicked.connect(self._export)
        for button in (self._pin_button, self._rollback_button, self._export_button):
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addStretch(1)
        column.addLayout(buttons)
        return column

    def _build_diff(self) -> QVBoxLayout:
        column = QVBoxLayout()
        column.setSpacing(self._theme.metric("spacing_sm"))
        title = QLabel("Отличия от текущей команды")
        title.setProperty("role", "h2")
        column.addWidget(title)
        self._diff = QTreeWidget()
        self._diff.setColumnCount(3)
        self._diff.setHeaderLabels(("Изменение", "Было", "Стало"))
        self._diff.setRootIsDecorated(False)
        self._diff.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        column.addWidget(self._diff, 1)
        self._diff_summary = QLabel("")
        self._diff_summary.setProperty("role", "muted")
        column.addWidget(self._diff_summary)
        return column

    # -- public API ---------------------------------------------------------

    def set_store(self, store: CommandTreeStore) -> None:
        """Point at a new profile's library and clear the view."""
        self._store = store
        self.clear()

    def clear(self) -> None:
        """Empty the widget — no command is open."""
        self._command_id = None
        self._current = None
        self._versions = []
        self._table.setRowCount(0)
        self._diff.clear()
        self._diff_summary.clear()
        self._update_buttons()

    def load(self, command_id: int, current: CommandModel) -> None:
        """Show the versions of a command, diffed against its current model."""
        self._command_id = command_id
        self._current = current
        self.refresh()

    def refresh(self) -> None:
        """Re-read the version list — after a save added one, or a pin changed."""
        if self._command_id is None:
            return
        try:
            self._versions = self._store.versions(self._command_id)
        except AyrisError:
            self._versions = []
        self._fill_table()
        self._on_selection()

    # -- table --------------------------------------------------------------

    def _fill_table(self) -> None:
        self._table.setRowCount(len(self._versions))
        for row, version in enumerate(self._versions):
            self._set_cell(row, _COL_VERSION, str(version.version))
            self._set_cell(row, _COL_DATE, _format_date(version))
            self._set_cell(row, _COL_AUTHOR, _author_of(version))
            self._set_cell(row, _COL_COMMENT, version.comment or "—")
            self._set_cell(row, _COL_SIZE, _format_size(version.size_bytes))
            self._set_cell(row, _COL_PIN, "★" if version.important else "")

    def _set_cell(self, row: int, column: int, text: str) -> None:
        item = QTableWidgetItem(text)
        if column in (_COL_VERSION, _COL_SIZE, _COL_PIN):
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._table.setItem(row, column, item)

    def _selected_version(self) -> CommandVersion | None:
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if not rows:
            return None
        index = rows[0].row()
        return self._versions[index] if 0 <= index < len(self._versions) else None

    # -- diff ---------------------------------------------------------------

    def _on_selection(self) -> None:
        self._update_buttons()
        self._diff.clear()
        self._diff_summary.clear()
        version = self._selected_version()
        if version is None or self._command_id is None or self._current is None:
            return
        try:
            old = self._store.version_model(self._command_id, version.version)
        except AyrisError as exc:
            self._diff_summary.setText(exc.user_message)
            return
        result = diff_commands(old, self._current)
        for change in result.changes:
            self._diff.addTopLevelItem(_diff_item(change))
        if result.is_empty:
            self._diff_summary.setText("Эта версия совпадает с текущей командой.")
        else:
            self._diff_summary.setText(f"Отличий: {result.summary()}")

    # -- actions ------------------------------------------------------------

    def _toggle_important(self) -> None:
        version = self._selected_version()
        if version is None or self._command_id is None:
            return
        self._store.mark_version_important(
            self._command_id, version.version, important=not version.important
        )
        self.refresh()

    def _rollback(self) -> None:
        version = self._selected_version()
        if version is None or self._command_id is None:
            return
        try:
            model = self._store.version_model(self._command_id, version.version)
        except AyrisError as exc:
            self.status.emit(exc.user_message)
            return
        dialog = ConfirmDialog(
            "Откат к версии",
            f"Команда будет заменена содержимым версии {version.version}. "
            "Текущее состояние сохранится в истории как новая версия.",
            self._theme,
            confirm_text="Откатить",
            parent=self,
        )
        if dialog.exec() != ConfirmDialog.DialogCode.Accepted:
            return
        # The editor applies it as a normal save: it keeps the live id and folder,
        # so the rebuilt model carries the current command's id, not a stale one.
        restored = model.model_copy(update={"id": self._command_id})
        self.rollback_requested.emit(restored)

    def _export(self) -> None:
        version = self._selected_version()
        if version is None or self._command_id is None:
            return
        from pathlib import Path

        from PySide6.QtWidgets import QFileDialog

        from ayris.actions.macros.serializer import AYRIS_SUFFIX

        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Экспорт версии",
            f"version_{version.version}{AYRIS_SUFFIX}",
            f"Команды Ayris (*{AYRIS_SUFFIX})",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix != AYRIS_SUFFIX:
            path = path.with_suffix(AYRIS_SUFFIX)
        try:
            path.write_text(
                self._store.export_version(self._command_id, version.version), encoding="utf-8"
            )
        except (AyrisError, OSError) as exc:
            self.status.emit(f"Не удалось экспортировать: {exc}")
            return
        self.status.emit(f"Версия {version.version} экспортирована в {path.name}.")

    def _update_buttons(self) -> None:
        version = self._selected_version()
        enabled = version is not None
        self._rollback_button.setEnabled(enabled)
        self._export_button.setEnabled(enabled)
        self._pin_button.setEnabled(enabled)
        if version is not None:
            self._pin_button.setText("Снять важность" if version.important else "Пометить важной")


# ----------------------------------------------------------------------
# columns and formatting
# ----------------------------------------------------------------------

_COL_VERSION = 0
_COL_DATE = 1
_COL_AUTHOR = 2
_COL_COMMENT = 3
_COL_SIZE = 4
_COL_PIN = 5
_COLUMNS = ("№", "Дата", "Автор", "Правка", "Размер", "Важно")


def _diff_item(change: Change) -> QTreeWidgetItem:
    old = "" if change.kind == CHANGE_ADDED else (change.old or "")
    new = "" if change.kind == CHANGE_REMOVED else (change.new or "")
    label = (
        change.label
        if not change.path or change.path in change.label
        else f"{change.label} ({change.path})"
    )
    return QTreeWidgetItem((label, old, new))


def _author_of(version: CommandVersion) -> str:
    """Who wrote the version, read from the save comment.

    The comment is the only signal the row carries: «редактор» for an editor save,
    «импорт …» for an import, «откат к версии N» for a rollback. Mapped to a short
    word for the column; anything else shows «правка».
    """
    comment = version.comment.lower()
    if "импорт" in comment:
        return "импорт"
    if "откат" in comment:
        return "откат"
    if "редактор" in comment:
        return "редактор"
    return "правка"


def _format_date(version: CommandVersion) -> str:
    if version.created_at is None:
        return "—"
    return version.created_at.astimezone().strftime("%d.%m.%Y %H:%M")


def _format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "—"
    if size_bytes < 1024:
        return f"{size_bytes} Б"
    return f"{size_bytes / 1024:.1f} КБ"
