"""Compact icon button with mandatory accessible text."""

from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QPushButton, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["IconButton"]


class IconButton(QPushButton):
    def __init__(
        self,
        icon: QIcon,
        tooltip: str,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        if not tooltip.strip():
            raise ValueError("Кнопке-иконке нужна непустая подсказка")
        super().__init__(parent)
        self._theme = theme
        self.setIcon(icon)
        self.setToolTip(tooltip)
        self.setAccessibleName(tooltip)
        self.setProperty("iconButton", True)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        size = self._theme.metric("icon_md")
        self.setIconSize(QSize(size, size))
