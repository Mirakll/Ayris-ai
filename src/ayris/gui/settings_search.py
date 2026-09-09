"""Search index and popup for navigating directly to a settings control."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from ayris.gui.tabs.base import SearchEntry
from ayris.gui.theme import ThemeManager

__all__ = ["SettingsSearch", "SettingsSearchIndex"]


class SettingsSearchIndex:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], SearchEntry] = {}

    @property
    def entries(self) -> tuple[SearchEntry, ...]:
        return tuple(self._entries.values())

    def add(self, entry: SearchEntry) -> None:
        self._entries[(entry.section_key, entry.path)] = entry

    def find(self, query: str) -> tuple[SearchEntry, ...]:
        words = tuple(part for part in query.casefold().split() if part)
        if not words:
            return ()
        matches = [
            entry
            for entry in self._entries.values()
            if all(word in entry.search_text for word in words)
        ]
        return tuple(sorted(matches, key=lambda entry: (entry.section_title, entry.label)))


class SettingsSearch(QListWidget):
    chosen = Signal(object)

    def __init__(self, index: SettingsSearchIndex, theme: ThemeManager, parent: QWidget) -> None:
        super().__init__(parent)
        self._index = index
        self._theme = theme
        self.setWindowFlags(Qt.WindowType.Popup)
        self.setAccessibleName("Результаты поиска настроек")
        self.itemActivated.connect(self._activate)
        self.itemClicked.connect(self._activate)

    def update_query(self, query: str, anchor: QWidget) -> None:
        self.clear()
        for entry in self._index.find(query):
            item = QListWidgetItem(f"{entry.label}  ·  {entry.section_title}")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.addItem(item)
        if not query.strip() or not self.count():
            self.hide()
            return
        self.setCurrentRow(0)
        width = max(anchor.width(), self.sizeHintForColumn(0) + self._theme.metric("spacing_2xl"))
        height = min(
            self.count() * max(1, self.sizeHintForRow(0)) + self._theme.metric("spacing_sm"),
            self._theme.metric("control_height_lg") * 7,
        )
        self.resize(width, height)
        self.move(anchor.mapToGlobal(QPoint(0, anchor.height())))
        self.show()

    def _activate(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, SearchEntry):
            self.hide()
            self.chosen.emit(entry)

    def activate_current(self) -> None:
        item = self.currentItem()
        if item is not None:
            self._activate(item)


def highlight_widget(widget: QWidget, theme: ThemeManager, *, duration_ms: int = 2000) -> None:
    previous = widget.styleSheet()
    widget.setProperty("searchHighlight", True)
    widget.setStyleSheet(
        previous + f"; border: {theme.metric('focus_width')}px solid {theme.theme.color('focus')};"
    )
    widget.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def clear() -> None:
        if widget.isWidgetType():
            widget.setProperty("searchHighlight", False)
            widget.setStyleSheet(previous)

    timer = QTimer(widget)
    timer.setSingleShot(True)
    timer.timeout.connect(clear)
    timer.timeout.connect(timer.deleteLater)
    timer.start(duration_ms)
