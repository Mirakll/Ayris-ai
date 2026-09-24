"""Task 59: read-only audit journal with filters, search and export.

A thin view over :class:`~ayris.core.audit.AuditReader`. The registry is the only
writer and masks secrets before a row ever reaches the database, so the values
shown here are already masked — the view never re-masks and never writes.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ayris.core.audit import AuditFilter, AuditReader
from ayris.core.models import AuditEntry, ExecutionResult, utc_now
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.search_field import SearchField

__all__ = ["AuditView", "serialize_audit_rows"]

#: Period presets: label → age in days (0 means «за всё время»).
_PERIODS: tuple[tuple[str, int], ...] = (
    ("За всё время", 0),
    ("За сутки", 1),
    ("За неделю", 7),
    ("За месяц", 30),
)

#: Result presets: label → value (None means «любой»).
_RESULTS: tuple[tuple[str, ExecutionResult | None], ...] = (
    ("Любой результат", None),
    ("Успех", ExecutionResult.OK),
    ("Ошибка", ExecutionResult.ERROR),
    ("Отменено", ExecutionResult.CANCELLED),
    ("Таймаут", ExecutionResult.TIMEOUT),
    ("Отказано", ExecutionResult.DENIED),
    ("Не распознано", ExecutionResult.UNMATCHED),
)

_COLUMNS: tuple[str, ...] = (
    "Время (UTC)",
    "Команда",
    "Параметры",
    "Результат",
    "Админ",
    "С повышением",
    "Подтверждено",
)

_PAGE_SIZE = 500

#: Column order for CSV/JSON export, shared by the writer header and each row.
_EXPORT_FIELDS: tuple[str, ...] = (
    "ts",
    "command",
    "params",
    "result",
    "require_admin",
    "elevated",
    "confirmed",
)


_RESULT_LABELS: dict[ExecutionResult, str] = {value: label for label, value in _RESULTS if value}


def _yes_no(value: bool) -> str:
    return "да" if value else "нет"


def _format_ts(ts: datetime | None) -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S") if ts is not None else ""


def _params_text(entry: AuditEntry) -> str:
    """Compact one-line rendering of the already-masked parameters."""
    if not entry.params:
        return ""
    return json.dumps(entry.params, ensure_ascii=False, sort_keys=True)


def serialize_audit_rows(rows: Sequence[AuditEntry], fmt: str) -> str:
    """Render audit rows as CSV or JSON. Pure — used by the export buttons and tests.

    The parameters are written exactly as stored (masked at the source); this never
    reaches back to the raw values.
    """
    records = [
        {
            "ts": entry.ts.isoformat() if entry.ts is not None else "",
            "command": entry.command_name,
            "params": entry.params,
            "result": entry.result.value,
            "require_admin": entry.require_admin,
            "elevated": entry.elevated,
            "confirmed": entry.confirmed,
        }
        for entry in rows
    ]
    if fmt == "json":
        return json.dumps(records, ensure_ascii=False, indent=2)
    if fmt == "csv":
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(_EXPORT_FIELDS)
        for record in records:
            writer.writerow(
                [
                    record["ts"],
                    record["command"],
                    json.dumps(record["params"], ensure_ascii=False, sort_keys=True),
                    record["result"],
                    record["require_admin"],
                    record["elevated"],
                    record["confirmed"],
                ]
            )
        return buffer.getvalue()
    raise ValueError(f"неизвестный формат экспорта: {fmt}")


class AuditView(QWidget):
    """Filterable, searchable, exportable table over the audit journal."""

    def __init__(
        self,
        reader: AuditReader,
        theme: ThemeManager,
        parent: QWidget | None = None,
        *,
        save_path: Callable[[str, str], str | None] | None = None,
    ) -> None:
        super().__init__(parent)
        self._reader = reader
        self._theme = theme
        self._rows: tuple[AuditEntry, ...] = ()
        self._total = 0
        # Injected so tests exercise export without a modal file dialog.
        self._save_path = save_path if save_path is not None else self._ask_save_path

        self._root = QVBoxLayout(self)
        filters = QHBoxLayout()
        self._period = ThemedComboBox()
        for label, days in _PERIODS:
            self._period.addItem(label, days)
        self._result = ThemedComboBox()
        for label, value in _RESULTS:
            self._result.addItem(label, value)
        self._admin = ThemedComboBox()
        self._admin.addItem("Любые команды", None)
        self._admin.addItem("Только требующие админ", True)
        self._admin.addItem("Только обычные", False)
        self._search = SearchField(placeholder="Поиск по команде", theme=theme)
        self._period.currentIndexChanged.connect(self._refresh)
        self._result.currentIndexChanged.connect(self._refresh)
        self._admin.currentIndexChanged.connect(self._refresh)
        self._search.search_changed.connect(self._refresh)
        filters.addWidget(self._period)
        filters.addWidget(self._result)
        filters.addWidget(self._admin)
        filters.addWidget(self._search, 1)
        self._root.addLayout(filters)

        self._table = QTableWidget(0, len(_COLUMNS))
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setAccessibleName("Журнал аудита")
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self._root.addWidget(self._table, 1)

        footer = QHBoxLayout()
        self._summary = QLabel("")
        self._summary.setProperty("role", "secondary")
        self._export_csv = QPushButton("Экспорт в CSV")
        self._export_json = QPushButton("Экспорт в JSON")
        self._export_csv.clicked.connect(lambda: self._export("csv"))
        self._export_json.clicked.connect(lambda: self._export("json"))
        footer.addWidget(self._summary, 1)
        footer.addWidget(self._export_csv)
        footer.addWidget(self._export_json)
        self._root.addLayout(footer)

        self._refresh()

    # -- filters and query ------------------------------------------------

    def current_filter(self) -> AuditFilter:
        days = int(self._period.currentData() or 0)
        since = utc_now() - timedelta(days=days) if days > 0 else None
        result = self._result.currentData()
        admin = self._admin.currentData()
        return AuditFilter(
            since=since,
            command=self._search.text().strip(),
            result=result if isinstance(result, ExecutionResult) else None,
            require_admin=admin if isinstance(admin, bool) else None,
        )

    def _refresh(self, *_args: object) -> None:
        page = self._reader.page(self.current_filter(), page=1, page_size=_PAGE_SIZE)
        self._rows = page.items
        self._total = page.total
        self._fill_table()
        shown = len(self._rows)
        self._summary.setText(f"Показано {shown} из {self._total}")

    def _fill_table(self) -> None:
        self._table.setRowCount(len(self._rows))
        for row, entry in enumerate(self._rows):
            cells = (
                _format_ts(entry.ts),
                entry.command_name,
                _params_text(entry),
                _RESULT_LABELS.get(entry.result, entry.result.value),
                _yes_no(entry.require_admin),
                _yes_no(entry.elevated),
                _yes_no(entry.confirmed),
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row, column, item)

    # -- export -----------------------------------------------------------

    def refresh(self) -> None:
        """Re-run the current query. Called after a cleanup empties the journal."""
        self._refresh()

    def selected_rows(self) -> tuple[AuditEntry, ...]:
        """The highlighted rows, or every shown row when nothing is selected."""
        chosen = sorted({index.row() for index in self._table.selectionModel().selectedRows()})
        if not chosen:
            return self._rows
        return tuple(self._rows[row] for row in chosen if 0 <= row < len(self._rows))

    def export_to(self, path: str, fmt: str) -> int:
        """Write the selected rows to ``path``. Returns the number of rows written."""
        rows = self.selected_rows()
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            handle.write(serialize_audit_rows(rows, fmt))
        return len(rows)

    def _export(self, fmt: str) -> None:
        path = self._save_path(fmt, f"audit.{fmt}")
        if path:
            self.export_to(path, fmt)

    def _ask_save_path(self, fmt: str, suggested: str) -> str | None:
        from PySide6.QtWidgets import QFileDialog

        caption = "Экспорт аудита"
        mask = "CSV (*.csv)" if fmt == "csv" else "JSON (*.json)"
        path, _ = QFileDialog.getSaveFileName(self, caption, suggested, mask)
        return path or None
