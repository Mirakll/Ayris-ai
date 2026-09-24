"""Live log viewer for the «Логи / DevTools» tab (task 58).

The widget mirrors the task-41 ring buffer and the ``LogLine`` events published
onto the bus. Two facts from the task shape it.

*It shows already-redacted data.* Every line — from the ring buffer or the bus —
went through the task-41 secret filter before it got here. The view never
re-masks and never sees a raw secret.

*It updates in batches.* A DEBUG session is thousands of lines a minute. Records
arrive on any thread, land in a bounded queue, and one timer on the GUI thread
drains them into the widget a few times a second. Appending per line would wedge
the event loop; a batch keeps the interface answering.

:class:`LogViewModel` holds all of that logic without a widget, so the tests read
it directly: the cap and its eviction, the level and module filters, the search.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.events import EventBus, LogLine
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.search_field import SearchField
from ayris.utils.logger import LOG_LEVELS, LogBufferEntry, get_log_buffer

__all__ = ["LogRow", "LogView", "LogViewModel", "level_color_token"]

#: How often the queue is drained into the widget, in milliseconds.
_DRAIN_MS = 200
_ALL_LEVELS = "все уровни"
_ALL_MODULES = "все модули"
_ROOT_PREFIX = "ayris."


def _level_value(level: str) -> int:
    """Numeric threshold for a level name, or ``NOTSET`` if it is unknown."""
    return logging.getLevelNamesMapping().get(level.upper(), logging.NOTSET)


def level_color_token(level: str) -> str:
    """Theme colour token for a level, so the model and the widget agree."""
    up = level.upper()
    if up in ("ERROR", "CRITICAL"):
        return "error"
    if up == "WARNING":
        return "warning"
    if up == "DEBUG":
        return "text_muted"
    return "text_secondary"


@dataclass(frozen=True, slots=True)
class LogRow:
    """One line the view holds: a ring-buffer entry and a bus event, unified."""

    created: float
    level: str
    logger: str
    message: str
    request_id: str = ""

    @classmethod
    def from_buffer(cls, entry: LogBufferEntry) -> LogRow:
        return cls(entry.created, entry.level, entry.logger, entry.message, entry.request_id)

    @classmethod
    def from_event(cls, event: LogLine) -> LogRow:
        # A bus LogLine carries no timestamp; stamp arrival so the row can be
        # ordered and shown with a time like every other line.
        return cls(time.time(), event.level, event.logger, event.message, event.request_id)

    @property
    def time_text(self) -> str:
        return datetime.fromtimestamp(self.created).strftime("%H:%M:%S")

    @property
    def short_logger(self) -> str:
        """Logger name without the ``ayris.`` prefix every logger shares."""
        name = self.logger
        return name[len(_ROOT_PREFIX) :] if name.startswith(_ROOT_PREFIX) else name

    def format_line(self) -> str:
        rid = f" [{self.request_id}]" if self.request_id else ""
        return f"{self.time_text}  {self.level:<8} {self.short_logger}{rid}  {self.message}"


class LogViewModel:
    """The log the view shows, without a widget: capped, filtered, searchable.

    A plain object, not a Qt model — the list is rebuilt into a text widget
    wholesale on a filter change and appended to line by line otherwise, and the
    tests want to assert the cap, the eviction and every filter with no window in
    the picture.
    """

    def __init__(self, *, capacity: int = 1000) -> None:
        self._rows: deque[LogRow] = deque(maxlen=max(1, capacity))
        # logger name -> how many live rows carry it, for the module combo
        self._modules: dict[str, int] = {}
        self._level = ""
        self._module = ""
        self._search = ""

    # -- ingest -------------------------------------------------------------

    def set_capacity(self, capacity: int) -> None:
        self._rows = deque(self._rows, maxlen=max(1, capacity))
        self._modules.clear()
        for row in self._rows:
            self._modules[row.logger] = self._modules.get(row.logger, 0) + 1

    def add(self, row: LogRow) -> None:
        if self._rows.maxlen is not None and len(self._rows) == self._rows.maxlen:
            self._forget_module(self._rows[0].logger)
        self._rows.append(row)
        self._modules[row.logger] = self._modules.get(row.logger, 0) + 1

    def extend(self, rows: Iterable[LogRow]) -> None:
        for row in rows:
            self.add(row)

    def clear(self) -> None:
        self._rows.clear()
        self._modules.clear()

    def _forget_module(self, logger: str) -> None:
        remaining = self._modules.get(logger, 0) - 1
        if remaining <= 0:
            self._modules.pop(logger, None)
        else:
            self._modules[logger] = remaining

    # -- filters ------------------------------------------------------------

    @property
    def known_modules(self) -> list[str]:
        return sorted(self._modules)

    def set_level_filter(self, level: str) -> None:
        self._level = "" if not level or level == _ALL_LEVELS else level.upper()

    def set_module_filter(self, module: str) -> None:
        self._module = "" if not module or module == _ALL_MODULES else module

    def set_search(self, text: str) -> None:
        self._search = text.strip().casefold()

    def matches(self, row: LogRow) -> bool:
        if self._level and _level_value(row.level) < _level_value(self._level):
            return False
        if self._module and not (
            row.logger == self._module or row.logger.startswith(self._module + ".")
        ):
            return False
        return not (
            self._search
            and self._search not in row.message.casefold()
            and self._search not in row.logger.casefold()
            and self._search not in row.request_id.casefold()
        )

    def visible_rows(self) -> list[LogRow]:
        return [row for row in self._rows if self.matches(row)]

    def count(self) -> int:
        return len(self._rows)


class LogView(QFrame):
    """The live log: filter controls above a batched, colour-coded text view.

    Args:
        theme: Active theme, for metrics and the per-level colours.
        bus: Where ``LogLine`` events arrive; ``None`` leaves the view static
            (still filled once from the ring buffer).
        capacity: How many lines to keep, matched by the widget's own block cap.
    """

    def __init__(
        self,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        capacity: int = 1000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._bus = bus
        self._model = LogViewModel(capacity=capacity)
        self._queue: deque[LogRow] = deque()
        self._queue_lock = threading.Lock()
        self._paused = False
        self._autoscroll = True
        self._syncing = False
        self._line_count = 0
        self._unsubscribe: Callable[[], None] | None = None

        self._outer = QVBoxLayout(self)
        self._build_controls()
        self._build_view()

        # Fill once from the ring buffer so the view is not empty on open, then
        # follow the bus for everything after.
        self._model.extend(LogRow.from_buffer(entry) for entry in get_log_buffer())
        self._refresh_modules()
        self._render_all()

        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(_DRAIN_MS)
        self._drain_timer.timeout.connect(self._drain)
        self._drain_timer.start()

        if bus is not None:
            self._unsubscribe = bus.subscribe(LogLine, self._on_log_line, weak=False)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> None:
        row = QHBoxLayout()
        self._level_combo = ThemedComboBox()
        self._level_combo.addItem(_ALL_LEVELS)
        for level in LOG_LEVELS:
            self._level_combo.addItem(level)
        self._level_combo.currentTextChanged.connect(self._on_level_filter)
        row.addWidget(QLabel("Уровень:"))
        row.addWidget(self._level_combo)

        self._module_combo = ThemedComboBox()
        self._module_combo.setAccessibleName("Фильтр по модулю")
        self._module_combo.currentTextChanged.connect(self._on_module_filter)
        row.addWidget(QLabel("Модуль:"))
        row.addWidget(self._module_combo, 1)

        self._search = SearchField(placeholder="Поиск по логу", theme=self._theme)
        self._search.search_changed.connect(self._on_search)
        row.addWidget(self._search, 2)
        self._outer.addLayout(row)

        actions = QHBoxLayout()
        self._pause_box = QCheckBox("Пауза")
        self._pause_box.toggled.connect(self._on_pause)
        actions.addWidget(self._pause_box)
        self._autoscroll_box = QCheckBox("Автопрокрутка")
        self._autoscroll_box.setChecked(True)
        self._autoscroll_box.toggled.connect(self._on_autoscroll_box)
        actions.addWidget(self._autoscroll_box)
        actions.addStretch(1)
        self._copy_button = QPushButton("Скопировать видимое")
        self._copy_button.clicked.connect(self._copy_visible)
        actions.addWidget(self._copy_button)
        self._clear_button = QPushButton("Очистить")
        self._clear_button.clicked.connect(self._clear)
        actions.addWidget(self._clear_button)
        self._outer.addLayout(actions)

    def _build_view(self) -> None:
        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setAccessibleName("Журнал")
        self._text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._text.setMaximumBlockCount(self._model._rows.maxlen or 1000)
        self._text.setFont(QFont("Consolas"))
        scrollbar = self._text.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.valueChanged.connect(self._on_scroll)
        self._outer.addWidget(self._text, 1)

    # -- ingest -------------------------------------------------------------

    def _on_log_line(self, event: LogLine) -> None:
        """Bus handler. Runs on any thread — only touch the queue, never Qt."""
        with self._queue_lock:
            self._queue.append(LogRow.from_event(event))

    def _drain(self) -> None:
        """Move a batch of queued rows into the model and the view, on the GUI thread."""
        with self._queue_lock:
            if not self._queue:
                return
            batch = list(self._queue)
            self._queue.clear()
        before = set(self._model.known_modules)
        self._model.extend(batch)
        if set(self._model.known_modules) != before:
            self._refresh_modules()
        if self._paused:
            return
        appended = False
        for row in batch:
            if self._model.matches(row):
                self._append_row(row)
                appended = True
        if appended and self._autoscroll:
            self._scroll_to_bottom()

    # -- rendering ----------------------------------------------------------

    def _append_row(self, row: LogRow) -> None:
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(self._theme.theme.color(level_color_token(row.level))))
        if self._line_count:
            cursor.insertBlock()
        cursor.insertText(row.format_line(), fmt)
        self._line_count += 1

    def _render_all(self) -> None:
        self._syncing = True
        self._text.setUpdatesEnabled(False)
        self._text.clear()
        self._line_count = 0
        for row in self._model.visible_rows():
            self._append_row(row)
        self._text.setUpdatesEnabled(True)
        self._syncing = False
        if self._autoscroll:
            self._scroll_to_bottom()

    def _refresh_modules(self) -> None:
        current = self._module_combo.currentText()
        self._module_combo.blockSignals(True)
        self._module_combo.clear()
        self._module_combo.addItem(_ALL_MODULES)
        for module in self._model.known_modules:
            self._module_combo.addItem(module)
        index = self._module_combo.findText(current)
        if index >= 0:
            self._module_combo.setCurrentIndex(index)
        else:
            self._module_combo.setCurrentIndex(0)
            self._model.set_module_filter("")
        self._module_combo.blockSignals(False)

    # -- filter handlers ----------------------------------------------------

    def _on_level_filter(self, text: str) -> None:
        self._model.set_level_filter(text)
        self._render_all()

    def _on_module_filter(self, text: str) -> None:
        self._model.set_module_filter(text)
        self._render_all()

    def _on_search(self, text: str) -> None:
        self._model.set_search(text)
        self._render_all()

    def _on_pause(self, checked: bool) -> None:
        self._paused = checked
        if not checked:
            self._render_all()

    def _on_autoscroll_box(self, checked: bool) -> None:
        self._autoscroll = checked
        if checked:
            self._scroll_to_bottom()

    def _on_scroll(self, value: int) -> None:
        if self._syncing:
            return
        scrollbar = self._text.verticalScrollBar()
        if scrollbar is not None:
            at_bottom = value >= scrollbar.maximum() - 2
            if at_bottom != self._autoscroll:
                self._autoscroll_box.setChecked(at_bottom)

    def _scroll_to_bottom(self) -> None:
        scrollbar = self._text.verticalScrollBar()
        if scrollbar is not None:
            self._syncing = True
            scrollbar.setValue(scrollbar.maximum())
            self._syncing = False

    # -- actions ------------------------------------------------------------

    def _copy_visible(self) -> None:
        text = "\n".join(row.format_line() for row in self._model.visible_rows())
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    def _clear(self) -> None:
        self._model.clear()
        self._text.clear()
        self._line_count = 0
        self._refresh_modules()

    def jump_to_request(self, request_id: str) -> None:
        """Show everything for one ``request_id`` — the pipeline table's «к логу»."""
        if not request_id:
            return
        self._level_combo.setCurrentIndex(0)
        module_all = self._module_combo.findText(_ALL_MODULES)
        if module_all >= 0:
            self._module_combo.setCurrentIndex(module_all)
        self._search.setText(request_id)

    # -- lifecycle ----------------------------------------------------------

    @property
    def model(self) -> LogViewModel:
        return self._model

    @property
    def is_paused(self) -> bool:
        return self._paused

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(self._theme.metric("spacing_sm"))

    def stop(self) -> None:
        """Stop the batch timer. A live timer keeps the event loop from ending."""
        if self._drain_timer.isActive():
            self._drain_timer.stop()

    def dispose(self) -> None:
        self.stop()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
