"""The command library's left panel: a tree with search, drag-and-drop and menus.

:class:`CommandTree` wraps a :class:`~ayris.gui.widgets.command_tree_model.CommandTreeModel`
in a :class:`QTreeView` and adds everything around it — a search field and status,
tag and conflict filters; a context menu that creates, renames, duplicates, toggles,
deletes and exports on one item or a whole multi-selection; internal drag-and-drop
that moves commands between folders and reorders folders; and import/export of a
command, a folder or the selected set through the task-30 ``.ayris`` serializer.

It draws disabled commands muted and marks a trigger conflict with a warning glyph
whose tooltip lists the clashing commands. The panel emits :attr:`command_activated`
with the selected command's id, which the editor of task 52 consumes, and
:attr:`tree_changed` after any change, which the tab turns into a ``CommandsChanged``
event so the rest of Ayris re-reads the library.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QModelIndex, QPersistentModelIndex, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.serializer import AYRIS_SUFFIX
from ayris.core.errors import AyrisError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.command_import_dialog import CommandImportDialog
from ayris.gui.widgets.command_tree_model import (
    CONFLICT_ROLE,
    ENABLED_ROLE,
    ENTITY_ID_ROLE,
    KIND_ROLE,
    CommandTreeModel,
    CommandTreeStore,
    NodeKind,
    StatusFilter,
    TreeFilter,
)
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.search_field import SearchField
from ayris.gui.widgets.toggle import ToggleSwitch
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.models import CommandFolder

__all__ = ["CommandTree"]

_log = get_logger(__name__)

_STATUS_OPTIONS: tuple[tuple[StatusFilter, str], ...] = (
    (StatusFilter.ALL, "Все"),
    (StatusFilter.ENABLED, "Включённые"),
    (StatusFilter.DISABLED, "Выключенные"),
)
_ANY_TAG = "\x00"


class CommandTree(QWidget):
    """The tree view plus its search, filters, menu and file operations."""

    command_activated = Signal(int)
    selection_changed = Signal()
    tree_changed = Signal()

    def __init__(
        self, store: CommandTreeStore, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._model = CommandTreeModel(store)
        self._model.drop_rejected.connect(lambda msg: self._notify(msg, "error"))
        self._model.changed.connect(self.tree_changed)

        self._layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self._search = SearchField(placeholder="Поиск команд", theme=theme)
        self._search.search_changed.connect(self._on_filter_changed)
        controls.addWidget(self._search, 1)
        self._status_combo = ThemedComboBox()
        for status, label in _STATUS_OPTIONS:
            self._status_combo.addItem(label, str(status))
        self._status_combo.currentIndexChanged.connect(self._on_filter_changed)
        controls.addWidget(self._status_combo)
        self._tag_combo = ThemedComboBox()
        self._tag_combo.currentIndexChanged.connect(self._on_filter_changed)
        controls.addWidget(self._tag_combo)
        self._layout.addLayout(controls)

        second = QHBoxLayout()
        self._conflicts_toggle = ToggleSwitch(theme, label="Только с конфликтами")
        self._conflicts_toggle.toggled.connect(self._on_filter_changed)
        conflicts_label = QLabel("Только с конфликтами")
        conflicts_label.setProperty("role", "secondary")
        second.addWidget(self._conflicts_toggle)
        second.addWidget(conflicts_label)
        second.addStretch(1)
        self._new_command_button = QPushButton("＋ Команда")
        self._new_command_button.clicked.connect(
            lambda: self._create_command(self._current_folder())
        )
        self._new_folder_button = QPushButton("＋ Папка")
        self._new_folder_button.clicked.connect(lambda: self._create_folder(self._current_folder()))
        self._import_button = QPushButton("Импорт…")
        self._import_button.clicked.connect(self._import_file)
        second.addWidget(self._new_command_button)
        second.addWidget(self._new_folder_button)
        second.addWidget(self._import_button)
        self._layout.addLayout(second)

        self._view = QTreeView()
        self._view.setModel(self._model)
        self._view.setHeaderHidden(True)
        self._view.setUniformRowHeights(True)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._view.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._view.setDropIndicatorShown(True)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        self._view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._view.customContextMenuRequested.connect(self._show_menu)
        self._delegate = _TreeDelegate(theme, self._view)
        self._view.setItemDelegate(self._delegate)
        selection = self._view.selectionModel()
        if selection is not None:
            selection.currentChanged.connect(self._on_current_changed)
            selection.selectionChanged.connect(lambda *_: self.selection_changed.emit())
        self._layout.addWidget(self._view, 1)

        self._status = QLabel("")
        self._status.setProperty("badge", "info")
        self._status.setWordWrap(True)
        self._status.hide()
        self._layout.addWidget(self._status)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self._reload_tags()

    # -- public -------------------------------------------------------------

    @property
    def model(self) -> CommandTreeModel:
        return self._model

    def set_store(self, store: CommandTreeStore) -> None:
        """Point the tree at a different profile's library and rebuild it."""
        self._model.set_store(store)
        self._reload_tags()

    def refresh(self) -> None:
        """Rebuild from storage — for a ``CommandsChanged`` from elsewhere."""
        self._model.reload()
        self._reload_tags()

    # -- filtering ----------------------------------------------------------

    def _on_filter_changed(self, *_args: object) -> None:
        tag_data = self._tag_combo.currentData()
        tree_filter = TreeFilter(
            text=self._search.text(),
            status=StatusFilter(self._status_combo.currentData() or str(StatusFilter.ALL)),
            tag="" if tag_data in (None, _ANY_TAG) else str(tag_data),
            only_conflicts=self._conflicts_toggle.isChecked(),
        )
        self._model.set_filter(tree_filter)
        self._delegate.set_query(tree_filter.text.strip())
        self._view.expandAll()

    def _reload_tags(self) -> None:
        current = self._tag_combo.currentData()
        self._tag_combo.blockSignals(True)
        self._tag_combo.clear()
        self._tag_combo.addItem("Все теги", _ANY_TAG)
        for tag in self._model.all_tags():
            self._tag_combo.addItem(tag, tag)
        index = self._tag_combo.findData(current)
        self._tag_combo.setCurrentIndex(index if index >= 0 else 0)
        self._tag_combo.blockSignals(False)

    # -- selection ----------------------------------------------------------

    def _on_current_changed(self, current: QModelIndex, _previous: QModelIndex) -> None:
        if current.isValid() and current.data(KIND_ROLE) == str(NodeKind.COMMAND):
            self.command_activated.emit(int(current.data(ENTITY_ID_ROLE)))

    def _current_folder(self) -> int | None:
        index = self._view.currentIndex()
        if not index.isValid():
            return None
        if index.data(KIND_ROLE) == str(NodeKind.FOLDER):
            return int(index.data(ENTITY_ID_ROLE))
        parent = index.parent()
        if parent.isValid():
            return int(parent.data(ENTITY_ID_ROLE))
        return None

    def _selected_commands(self) -> list[int]:
        return [
            int(idx.data(ENTITY_ID_ROLE))
            for idx in self._view.selectionModel().selectedIndexes()
            if idx.column() == 0 and idx.data(KIND_ROLE) == str(NodeKind.COMMAND)
        ]

    # -- context menu -------------------------------------------------------

    def _show_menu(self, point: QPoint) -> None:
        index = self._view.indexAt(point)
        menu = QMenu(self)
        commands = self._selected_commands()
        is_folder = index.isValid() and index.data(KIND_ROLE) == str(NodeKind.FOLDER)
        is_command = index.isValid() and index.data(KIND_ROLE) == str(NodeKind.COMMAND)
        multi = len(commands) > 1

        folder_for_new = self._current_folder()
        _add(menu, "Новая команда", lambda: self._create_command(folder_for_new))
        _add(menu, "Новая папка", lambda: self._create_folder(folder_for_new))

        if multi:
            menu.addSeparator()
            _add(menu, f"Включить ({len(commands)})", lambda: self._bulk_enable(commands, True))
            _add(menu, f"Выключить ({len(commands)})", lambda: self._bulk_enable(commands, False))
            _add(menu, "Назначить тег…", lambda: self._assign_tag(commands))
            _add(menu, "Экспортировать выбранное…", lambda: self._export_commands(commands))
            _add(menu, f"Удалить ({len(commands)})", lambda: self._delete_commands(commands))
        elif is_command:
            command_id = int(index.data(ENTITY_ID_ROLE))
            enabled = bool(index.data(ENABLED_ROLE))
            menu.addSeparator()
            _add(menu, "Переименовать", lambda: self._view.edit(index))
            _add(menu, "Дублировать", lambda: self._duplicate(command_id))
            label = "Выключить" if enabled else "Включить"
            _add(menu, label, lambda: self._toggle(command_id, not enabled))
            _add(menu, "Экспортировать…", lambda: self._export_commands([command_id]))
            menu.addSeparator()
            _add(menu, "Удалить", lambda: self._delete_commands([command_id]))
        elif is_folder:
            folder_id = int(index.data(ENTITY_ID_ROLE))
            menu.addSeparator()
            _add(menu, "Переименовать", lambda: self._view.edit(index))
            _add(menu, "Экспортировать папку…", lambda: self._export_folder(folder_id))
            menu.addSeparator()
            _add(menu, "Удалить папку", lambda: self._delete_folder(folder_id))

        viewport = self._view.viewport()
        if viewport is not None:
            menu.exec(viewport.mapToGlobal(point))

    # -- create -------------------------------------------------------------

    def _create_command(self, folder_id: int | None) -> None:
        name, ok = QInputDialog.getText(self, "Новая команда", "Имя команды:")
        if not ok or not name.strip():
            return
        try:
            created = self._model.store.create_command(folder_id, name.strip())
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()
        if created.id is not None:
            self._select(NodeKind.COMMAND, created.id)

    def _create_folder(self, parent_id: int | None) -> None:
        name, ok = QInputDialog.getText(self, "Новая папка", "Имя папки:")
        if not ok or not name.strip():
            return
        try:
            self._model.store.create_folder(parent_id, name.strip())
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()

    # -- single-command operations -----------------------------------------

    def _duplicate(self, command_id: int) -> None:
        try:
            created = self._model.store.duplicate_command(command_id)
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()
        if created is not None and created.id is not None:
            self._select(NodeKind.COMMAND, created.id)

    def _toggle(self, command_id: int, enabled: bool) -> None:
        try:
            self._model.set_enabled(command_id, enabled=enabled)
        except AyrisError as exc:
            self._notify(exc.user_message, "error")

    # -- bulk operations ----------------------------------------------------

    def _bulk_enable(self, command_ids: list[int], enabled: bool) -> None:
        done = 0
        for command_id in command_ids:
            try:
                self._model.store.set_enabled(command_id, enabled=enabled)
                done += 1
            except AyrisError:
                _log.exception("не удалось переключить команду %s", command_id)
        self._after_change()
        verb = "включено" if enabled else "выключено"
        self._notify(f"Готово: {verb} команд — {done} из {len(command_ids)}.", "success")

    def _assign_tag(self, command_ids: list[int]) -> None:
        tag, ok = QInputDialog.getText(self, "Назначить тег", "Тег:")
        if not ok or not tag.strip():
            return
        try:
            changed = self._model.store.assign_tag(command_ids, tag.strip())
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()
        self._notify(f"Тег «{tag.strip()}» добавлен командам: {changed}.", "success")

    def _delete_commands(self, command_ids: list[int]) -> None:
        count = len(command_ids)
        title = "Удалить команду?" if count == 1 else "Удалить команды?"
        text = (
            "Команда будет удалена без возможности восстановления."
            if count == 1
            else f"Будут удалены команды: {count}. Действие необратимо."
        )
        if not self._confirm(title, text):
            return
        done = 0
        for command_id in command_ids:
            try:
                self._model.store.delete_command(command_id)
                done += 1
            except AyrisError:
                _log.exception("не удалось удалить команду %s", command_id)
        self._after_change()
        self._notify(f"Удалено команд: {done} из {count}.", "success")

    def _delete_folder(self, folder_id: int) -> None:
        inside = len(self._model.store.commands_in_subtree(folder_id))
        detail = (
            "Папка пуста и будет удалена."
            if inside == 0
            else f"Внутри команд: {inside}. Они переедут в корень, а папка будет удалена."
        )
        if not self._confirm("Удалить папку?", detail):
            return
        try:
            self._model.store.delete_folder(folder_id)
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()

    # -- import / export ----------------------------------------------------

    def _export_commands(self, command_ids: list[int]) -> None:
        if not command_ids:
            return
        default = "command" if len(command_ids) == 1 else "commands"
        path = self._save_path(default)
        if path is None:
            return
        try:
            if len(command_ids) == 1:
                text = self._model.store.export_command(command_ids[0])
            else:
                text = self._model.store.export_commands(command_ids)
            path.write_text(text, encoding="utf-8")
        except (AyrisError, OSError) as exc:
            self._notify(f"Не удалось экспортировать: {exc}", "error")
            return
        self._notify(f"Экспортировано в {path.name}.", "success")

    def _export_folder(self, folder_id: int) -> None:
        path = self._save_path("folder")
        if path is None:
            return
        try:
            path.write_text(self._model.store.export_folder(folder_id), encoding="utf-8")
        except (AyrisError, OSError) as exc:
            self._notify(f"Не удалось экспортировать: {exc}", "error")
            return
        self._notify(f"Папка экспортирована в {path.name}.", "success")

    def _import_file(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл команд", "", f"Команды Ayris (*{AYRIS_SUFFIX})"
        )
        if not chosen:
            return
        try:
            text = _read_text(chosen)
        except OSError as exc:
            self._notify(f"Не удалось открыть файл: {exc}", "error")
            return
        dialog = CommandImportDialog(text, self._theme, folders=self._folder_choices(), parent=self)
        if not dialog.is_valid or dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            outcome = self._model.store.apply_import(
                text, target_folder_id=dialog.target_folder_id, strategy=dialog.strategy
            )
        except AyrisError as exc:
            self._notify(exc.user_message, "error")
            return
        self._after_change()
        self._notify(outcome.summary, "success")

    def _folder_choices(self) -> list[tuple[int | None, str]]:
        choices: list[tuple[int | None, str]] = [(None, "Корень")]
        folders = self._model.store.folders()
        for folder in sorted(folders, key=lambda f: _folder_label(f, folders)):
            if folder.id is not None:
                choices.append((folder.id, _folder_label(folder, folders)))
        return choices

    def _save_path(self, stem: str) -> Path | None:
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Сохранить как", f"{stem}{AYRIS_SUFFIX}", f"Команды Ayris (*{AYRIS_SUFFIX})"
        )
        if not chosen:
            return None
        path = Path(chosen)
        if path.suffix != AYRIS_SUFFIX:
            path = path.with_suffix(AYRIS_SUFFIX)
        return path

    # -- helpers ------------------------------------------------------------

    def _select(self, kind: NodeKind, entity_id: int) -> None:
        index = self._model.index_for(kind, entity_id)
        if index.isValid():
            self._view.setCurrentIndex(index)
            self._view.scrollTo(index)

    def _after_change(self) -> None:
        self._model.reload()
        self._reload_tags()
        self._view.expandAll()
        self.tree_changed.emit()

    def _confirm(self, title: str, text: str) -> bool:
        dialog = ConfirmDialog(
            title, text, self._theme, confirm_text="Удалить", dangerous=True, parent=self
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _notify(self, message: str, kind: str) -> None:
        self._status.setText(message)
        self._status.setProperty("badge", kind)
        style = self._status.style()
        if style is not None:
            style.unpolish(self._status)
            style.polish(self._status)
        self._status.setVisible(bool(message))

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        gap = self._theme.metric("spacing_sm")
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(gap)


class _TreeDelegate(QStyledItemDelegate):
    """Mutes disabled commands, bolds a search hit and flags a trigger conflict."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._query = ""

    def set_query(self, query: str) -> None:
        self._query = query.casefold()

    def initStyleOption(  # noqa: N802
        self, option: QStyleOptionViewItem, index: QModelIndex | QPersistentModelIndex
    ) -> None:
        super().initStyleOption(option, index)
        if index.data(KIND_ROLE) == str(NodeKind.COMMAND) and not bool(index.data(ENABLED_ROLE)):
            muted = QColor(self._theme.theme.color("text_muted"))
            palette = option.palette  # type: ignore[attr-defined]
            palette.setColor(palette.ColorRole.Text, muted)
            palette.setColor(palette.ColorRole.WindowText, muted)
        display = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        if self._query and self._query in display.casefold():
            font = QFont(option.font)  # type: ignore[attr-defined]
            font.setBold(True)
            option.font = font  # type: ignore[attr-defined]

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex | QPersistentModelIndex,
    ) -> None:
        super().paint(painter, option, index)
        conflicts = index.data(CONFLICT_ROLE)
        if not conflicts:
            return
        size = self._theme.metric("icon_sm")
        rect = option.rect  # type: ignore[attr-defined]
        box = QRect(rect.right() - size - 4, rect.center().y() - size // 2, size, size)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = QColor(self._theme.theme.color("warning"))
        pen = QPen(colour)
        pen.setWidth(max(1, self._theme.metric("border_width")))
        painter.setPen(pen)
        top = QPoint(box.center().x(), box.top())
        painter.drawLine(top, box.bottomLeft())
        painter.drawLine(top, box.bottomRight())
        painter.drawLine(box.bottomLeft(), box.bottomRight())
        mid_x = box.center().x()
        painter.drawLine(mid_x, box.top() + size // 3, mid_x, box.bottom() - size // 3)
        painter.drawPoint(mid_x, box.bottom() - size // 4)
        painter.restore()


def _add(menu: QMenu, text: str, handler: Callable[..., object]) -> QAction:
    action = QAction(text, menu)
    action.triggered.connect(handler)
    menu.addAction(action)
    return action


def _folder_label(folder: CommandFolder, folders: list[CommandFolder]) -> str:
    by_id = {f.id: f for f in folders}
    parts: list[str] = []
    current: CommandFolder | None = folder
    guard = 0
    while current is not None and guard < 100:
        parts.append(current.name)
        current = by_id.get(current.parent_id) if current.parent_id is not None else None
        guard += 1
    parts.reverse()
    return " / ".join(parts)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
