"""The overlay's short dialogue log: what was heard, what Ayris answered, errors.

The panel keeps only the last N lines in memory; the full history lives in the
database. Autoscroll follows the newest line until the user scrolls up, then it
stops so a scrollback is not yanked away; it resumes once the view is back at the
bottom. Any line can be copied.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Final

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QMenu, QVBoxLayout, QWidget

from ayris.core.models import utc_now
from ayris.gui.theme import ThemeManager

__all__ = ["DialogEntry", "DialogKind", "DialogLog"]

_DEFAULT_MAX_LINES: Final = 50


class DialogKind(StrEnum):
    HEARD = "heard"
    ANSWER = "answer"
    ERROR = "error"


_PREFIX: Final[dict[DialogKind, str]] = {
    DialogKind.HEARD: "Вы",
    DialogKind.ANSWER: "Айрис",
    DialogKind.ERROR: "Ошибка",
}

_COLOR_TOKEN: Final[dict[DialogKind, str]] = {
    DialogKind.HEARD: "text_secondary",
    DialogKind.ANSWER: "text_primary",
    DialogKind.ERROR: "error",
}


@dataclass(frozen=True, slots=True)
class DialogEntry:
    kind: DialogKind
    text: str
    at: datetime

    def render(self) -> str:
        stamp = self.at.astimezone().strftime("%H:%M")
        return f"{stamp}  {_PREFIX[self.kind]}: {self.text}"


class DialogLog(QWidget):
    """A bounded, autoscrolling list of dialogue lines."""

    def __init__(
        self,
        theme: ThemeManager,
        *,
        max_lines: int = _DEFAULT_MAX_LINES,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._max_lines = max(1, max_lines)
        self._entries: deque[DialogEntry] = deque(maxlen=self._max_lines)
        self._stick_to_bottom = True
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._list = QListWidget(self)
        self._list.setAccessibleName("Лог диалога")
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setWordWrap(True)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_menu)
        scrollbar = self._list.verticalScrollBar()
        scrollbar.valueChanged.connect(self._scrolled)
        layout.addWidget(self._list)
        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    @property
    def entries(self) -> tuple[DialogEntry, ...]:
        return tuple(self._entries)

    @property
    def max_lines(self) -> int:
        return self._max_lines

    def set_max_lines(self, value: int) -> None:
        self._max_lines = max(1, value)
        trimmed = deque(self._entries, maxlen=self._max_lines)
        self._entries = trimmed
        self._rebuild()

    def add_entry(self, kind: DialogKind, text: str, *, at: datetime | None = None) -> None:
        clean = text.strip()
        if not clean:
            return
        entry = DialogEntry(kind, clean, at or utc_now())
        overflowing = len(self._entries) == self._max_lines
        self._entries.append(entry)
        if overflowing:
            first = self._list.takeItem(0)
            del first
        self._append_item(entry)
        if self._stick_to_bottom:
            self._list.scrollToBottom()

    def clear(self) -> None:
        self._entries.clear()
        self._list.clear()
        self._stick_to_bottom = True

    def copy_selected(self) -> bool:
        selected = self._list.selectedItems()
        if not selected:
            return False
        QGuiApplication.clipboard().setText(selected[0].text())
        return True

    def _append_item(self, entry: DialogEntry) -> None:
        item = QListWidgetItem(entry.render())
        item.setForeground(QBrush(QColor(self._theme.theme.color(_COLOR_TOKEN[entry.kind]))))
        item.setData(Qt.ItemDataRole.AccessibleTextRole, entry.render())
        self._list.addItem(item)

    def _rebuild(self) -> None:
        self._list.clear()
        for entry in self._entries:
            self._append_item(entry)
        if self._stick_to_bottom:
            self._list.scrollToBottom()

    def _scrolled(self, value: int) -> None:
        scrollbar = self._list.verticalScrollBar()
        self._stick_to_bottom = value >= scrollbar.maximum()

    def _show_menu(self, point: QPoint) -> None:
        if not self._list.selectedItems():
            return
        menu = QMenu(self._list)
        copy = menu.addAction("Копировать строку")
        copy.triggered.connect(lambda _checked: self.copy_selected())
        menu.exec(self._list.mapToGlobal(point))

    def _refresh_theme(self, _theme: object | None = None) -> None:
        spacing = self._theme.metric("spacing_xs")
        self._list.setSpacing(spacing)
        self._rebuild()
