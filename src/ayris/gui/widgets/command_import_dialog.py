"""Preview and options for importing a ``.ayris`` file into the command library.

The dialog parses the file through the task-30 serializer, shows what it holds —
commands and the folders they carry — and collects the two decisions the store
needs to apply it: which folder to import under, and how to resolve a name that
already exists (rename, replace or skip). It applies nothing itself; the caller
reads :attr:`target_folder_id` and :attr:`strategy` and hands them to
:meth:`CommandTreeStore.apply_import`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.format_migrations import MacroFormatError
from ayris.actions.macros.serializer import load_document
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.command_tree_model import ConflictStrategy

if TYPE_CHECKING:
    from ayris.actions.macros.schema import CommandModel

__all__ = ["CommandImportDialog"]

_STRATEGIES: tuple[tuple[ConflictStrategy, str], ...] = (
    (ConflictStrategy.RENAME, "Переименовать новые"),
    (ConflictStrategy.REPLACE, "Заменить существующие"),
    (ConflictStrategy.SKIP, "Пропустить существующие"),
)


class CommandImportDialog(QDialog):
    """Import preview: destination folder and conflict resolution."""

    def __init__(
        self,
        text: str,
        theme: ThemeManager,
        *,
        folders: list[tuple[int | None, str]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowTitle("Импорт команд")
        self.setAccessibleName("Импорт команд")
        self.setModal(True)
        self._commands: tuple[CommandModel, ...] = ()

        self._layout = QVBoxLayout(self)
        heading = QLabel("Импорт команд")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)

        self._summary = QLabel("")
        self._summary.setProperty("role", "secondary")
        self._summary.setWordWrap(True)
        self._layout.addWidget(self._summary)

        self._list = QListWidget()
        self._list.setAccessibleName("Содержимое файла")
        self._layout.addWidget(self._list, 1)

        form = QFormLayout()
        self._folder_combo = ThemedComboBox()
        for folder_id, label in folders:
            self._folder_combo.addItem(label, folder_id)
        self._strategy_combo = ThemedComboBox()
        for strategy, label in _STRATEGIES:
            self._strategy_combo.addItem(label, str(strategy))
        form.addRow("Импортировать в папку:", self._folder_combo)
        form.addRow("При совпадении имени:", self._strategy_combo)
        self._layout.addLayout(form)

        buttons = QDialogButtonBox()
        self._import_button = QPushButton("Импортировать")
        self._import_button.setProperty("kind", "primary")
        cancel = QPushButton("Отмена")
        self._import_button.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(self._import_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self._layout.addWidget(buttons)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self._load(text)

    @property
    def target_folder_id(self) -> int | None:
        data = self._folder_combo.currentData()
        return data if isinstance(data, int) else None

    @property
    def strategy(self) -> ConflictStrategy:
        return ConflictStrategy(self._strategy_combo.currentData())

    @property
    def is_valid(self) -> bool:
        return bool(self._commands)

    def _load(self, text: str) -> None:
        try:
            document = load_document(text)
        except (MacroFormatError, ValueError) as exc:
            self._summary.setText(f"Не удалось прочитать файл: {exc}")
            self._import_button.setEnabled(False)
            return
        self._commands = tuple(document.commands)
        if not self._commands:
            self._summary.setText("В файле нет команд.")
            self._import_button.setEnabled(False)
            return
        folder_count = len({tuple(f.path) for f in document.folders})
        self._summary.setText(f"Команд в файле: {len(self._commands)}. Папок: {folder_count}.")
        for command in self._commands:
            path = " / ".join(command.folder)
            label = f"{command.name} — {path}" if path else command.name
            self._list.addItem(label)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_md"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))
        self._list.setMinimumHeight(self._theme.metric("control_height") * 4)
