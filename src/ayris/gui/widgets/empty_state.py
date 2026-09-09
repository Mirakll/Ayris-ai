"""Centred empty state with optional recovery action."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["EmptyState"]


class EmptyState(QWidget):
    def __init__(
        self,
        title: str,
        text: str,
        theme: ThemeManager,
        *,
        icon: QIcon | None = None,
        action_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._layout = QVBoxLayout(self)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon_label = QLabel()
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if icon is not None:
            self.icon_label.setPixmap(icon.pixmap(theme.metric("icon_lg")))
        self.title_label = QLabel(title)
        self.title_label.setProperty("role", "h2")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label = QLabel(text)
        self.text_label.setProperty("role", "secondary")
        self.text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.text_label.setWordWrap(True)
        self._layout.addWidget(self.icon_label)
        self._layout.addWidget(self.title_label)
        self._layout.addWidget(self.text_label)
        self.action_button: QPushButton | None = None
        if action_text is not None:
            self.action_button = QPushButton(action_text)
            self.action_button.setProperty("kind", "primary")
            self._layout.addWidget(self.action_button, 0, Qt.AlignmentFlag.AlignCenter)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_sm"))
