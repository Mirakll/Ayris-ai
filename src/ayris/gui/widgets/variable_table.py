"""The command's variable declarations as a table, editing ``model.variables``.

Columns are name / type / scope / default / persistent, the five fields of
:class:`~ayris.actions.macros.schema.VariableModel`. Editing a cell writes straight
back into the working :class:`CommandModel`, so the editor keeps one representation of
the command and the node view of task 53 sees the same declarations.

Two hints come from the task-30 validator, not from this widget: a declared variable
nobody references, and a ``{name}`` used in the blocks that was never declared. The
widget only paints them — :meth:`set_diagnostics` takes the two name sets the editor
computed off the UI thread and colours the matching rows, so a heavy walk of the block
tree never happens here.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.schema import CommandModel, VariableModel
from ayris.core.models import VariableScope, VariableType
from ayris.gui.theme import ThemeManager

__all__ = ["VariableTable"]

_TYPE_LABELS: dict[VariableType, str] = {
    VariableType.STRING: "строка",
    VariableType.INT: "целое",
    VariableType.FLOAT: "дробное",
    VariableType.BOOL: "да/нет",
    VariableType.ARRAY: "список",
    VariableType.DICT: "словарь",
}

_SCOPE_LABELS: dict[VariableScope, str] = {
    VariableScope.LOCAL: "локальная",
    VariableScope.PROFILE: "профиль",
    VariableScope.GLOBAL: "глобальная",
}


class VariableTable(QWidget):
    """Editable table of a command's variable declarations."""

    changed = Signal()

    _COLUMNS = ("Имя", "Тип", "Область", "По умолчанию", "Хранить")

    def __init__(self, theme: ThemeManager, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._model: CommandModel | None = None
        self._loading = False
        self._unused: set[str] = set()
        self._undeclared: set[str] = set()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_sm"))

        self._table = QTableWidget(0, len(self._COLUMNS))
        self._table.setHorizontalHeaderLabels(self._COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self._table.itemChanged.connect(self._on_item_changed)
        outer.addWidget(self._table)

        self._diagnostics = QLabel("")
        self._diagnostics.setProperty("role", "muted")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.hide()
        outer.addWidget(self._diagnostics)

        buttons = QHBoxLayout()
        add = QPushButton("Добавить")
        add.clicked.connect(self._add_row)
        self._remove = QPushButton("Удалить")
        self._remove.clicked.connect(self._remove_selected)
        buttons.addWidget(add)
        buttons.addWidget(self._remove)
        buttons.addStretch(1)
        outer.addLayout(buttons)

    # -- public API ---------------------------------------------------------

    def set_command(self, model: CommandModel) -> None:
        self._model = model
        self._rebuild()

    def declared_names(self) -> list[str]:
        """The names currently declared, for the parameter form's completions."""
        return [] if self._model is None else [v.name for v in self._model.variables]

    def set_diagnostics(self, *, unused: Iterable[str], undeclared: Iterable[str]) -> None:
        """Colour the rows the validator flagged and list undeclared references."""
        self._unused = set(unused)
        self._undeclared = set(undeclared)
        self._paint_diagnostics()

    # -- building -----------------------------------------------------------

    def _rebuild(self) -> None:
        self._loading = True
        try:
            self._table.setRowCount(0)
            if self._model is not None:
                for declared in self._model.variables:
                    self._append_row(declared)
        finally:
            self._loading = False
        self._paint_diagnostics()

    def _append_row(self, declared: VariableModel) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        name_item = QTableWidgetItem(declared.name)
        self._table.setItem(row, 0, name_item)

        type_combo = QComboBox()
        for kind, label in _TYPE_LABELS.items():
            type_combo.addItem(label, kind)
        type_combo.setCurrentIndex(type_combo.findData(declared.type))
        type_combo.currentIndexChanged.connect(self._commit)
        self._table.setCellWidget(row, 1, type_combo)

        scope_combo = QComboBox()
        for scope, label in _SCOPE_LABELS.items():
            scope_combo.addItem(label, scope)
        scope_combo.setCurrentIndex(scope_combo.findData(declared.scope))
        scope_combo.currentIndexChanged.connect(self._commit)
        self._table.setCellWidget(row, 2, scope_combo)

        default_item = QTableWidgetItem("" if declared.default is None else str(declared.default))
        self._table.setItem(row, 3, default_item)

        persist_item = QTableWidgetItem()
        persist_item.setFlags(
            Qt.ItemFlag.ItemIsUserCheckable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
        )
        persist_item.setCheckState(
            Qt.CheckState.Checked if declared.persistent else Qt.CheckState.Unchecked
        )
        self._table.setItem(row, 4, persist_item)

    def _add_row(self) -> None:
        if self._model is None:
            return
        base = "переменная"
        existing = {v.name for v in self._model.variables}
        name = base
        index = 1
        while name in existing:
            index += 1
            name = f"{base}{index}"
        self._append_row(
            VariableModel(name=name, type=VariableType.STRING, scope=VariableScope.LOCAL)
        )
        self._commit()

    def _remove_selected(self) -> None:
        rows = {index.row() for index in self._table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self._table.removeRow(row)
        self._commit()

    # -- writing back -------------------------------------------------------

    def _on_item_changed(self, _item: QTableWidgetItem) -> None:
        if not self._loading:
            self._commit()

    def _commit(self, *_args: object) -> None:
        if self._model is None or self._loading:
            return
        declared: list[VariableModel] = []
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            name = name_item.text().strip() if name_item is not None else ""
            if not name:
                continue
            type_combo = self._table.cellWidget(row, 1)
            scope_combo = self._table.cellWidget(row, 2)
            default_item = self._table.item(row, 3)
            persist_item = self._table.item(row, 4)
            assert isinstance(type_combo, QComboBox)
            assert isinstance(scope_combo, QComboBox)
            var_type = type_combo.currentData()
            raw_default = default_item.text() if default_item is not None else ""
            persistent = (
                persist_item is not None and persist_item.checkState() == Qt.CheckState.Checked
            )
            try:
                declared.append(
                    VariableModel(
                        name=name,
                        type=var_type,
                        scope=scope_combo.currentData(),
                        default=_coerce_default(raw_default, var_type),
                        persistent=persistent,
                    )
                )
            except ValueError:
                self._mark_row_invalid(row)
                continue
            self._mark_row_valid(row)
        self._model.variables = declared
        self.changed.emit()
        self._paint_diagnostics()

    # -- diagnostics --------------------------------------------------------

    def _paint_diagnostics(self) -> None:
        clear = QColor(Qt.GlobalColor.transparent)
        warn = _tint(self._theme.theme.color("warning"))
        for row in range(self._table.rowCount()):
            name_item = self._table.item(row, 0)
            if name_item is None:
                continue
            unused = name_item.text().strip() in self._unused
            name_item.setBackground(warn if unused else clear)
            name_item.setToolTip("На эту переменную никто не ссылается." if unused else "")
        if self._undeclared:
            listed = ", ".join(sorted(self._undeclared))
            self._diagnostics.setText(f"Не объявлены, но используются: {listed}")
            self._diagnostics.show()
        else:
            self._diagnostics.hide()

    def _mark_row_invalid(self, row: int) -> None:
        item = self._table.item(row, 3)
        if item is not None:
            item.setBackground(_tint(self._theme.theme.color("error")))
            item.setToolTip("Значение не соответствует типу переменной.")

    def _mark_row_valid(self, row: int) -> None:
        item = self._table.item(row, 3)
        if item is not None:
            item.setBackground(QColor(Qt.GlobalColor.transparent))
            item.setToolTip("")


def _tint(hex_color: str) -> QColor:
    """A translucent wash of a token colour, so the text on the cell stays legible."""
    color = QColor(hex_color)
    color.setAlpha(64)
    return color


def _coerce_default(text: str, var_type: VariableType) -> object:
    """Parse the default cell to the declared type, raising on a mismatch."""
    text = text.strip()
    if text == "":
        return None
    if var_type is VariableType.INT:
        return int(text)
    if var_type is VariableType.FLOAT:
        return float(text)
    if var_type is VariableType.BOOL:
        lowered = text.casefold()
        if lowered in ("да", "true", "1", "yes"):
            return True
        if lowered in ("нет", "false", "0", "no"):
            return False
        raise ValueError(text)
    if var_type in (VariableType.ARRAY, VariableType.DICT):
        import json

        value = json.loads(text)
        if var_type is VariableType.ARRAY and not isinstance(value, list):
            raise ValueError(text)
        if var_type is VariableType.DICT and not isinstance(value, dict):
            raise ValueError(text)
        return value
    return text
