"""Accessible keyboard-navigable sidebar for settings sections."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QListWidget, QListWidgetItem, QWidget

from ayris.gui.tabs.registry import TabSpec
from ayris.gui.theme import ThemeManager

__all__ = ["NavSidebar"]


class NavSidebar(QListWidget):
    section_selected = Signal(str)

    def __init__(
        self,
        sections: tuple[TabSpec, ...],
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._sections = sections
        self.setAccessibleName("Разделы настроек")
        self.setObjectName("settingsSidebar")
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setIconSize(QSize(theme.metric("icon_md"), theme.metric("icon_md")))
        self.setSpacing(theme.metric("spacing_xs"))
        self.setFixedWidth(theme.metric("content_width") // 3)
        for spec in sections:
            item = QListWidgetItem(self.style().standardIcon(spec.icon), spec.title)
            item.setData(Qt.ItemDataRole.UserRole, spec.key)
            item.setSizeHint(QSize(0, theme.metric("control_height_lg")))
            self.addItem(item)
        self.currentItemChanged.connect(self._selected)

    def select_section(self, key: str) -> bool:
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == key:
                self.setCurrentRow(row)
                return True
        return False

    def _selected(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current is not None:
            self.section_selected.emit(str(current.data(Qt.ItemDataRole.UserRole)))
