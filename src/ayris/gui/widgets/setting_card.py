"""A labelled settings row with an arbitrary control on the right."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["SettingCard"]


class SettingCard(QFrame):
    def __init__(
        self,
        title: str,
        description: str,
        control: QWidget,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("card", True)
        self.setAccessibleName(title)
        self._layout = QHBoxLayout(self)
        self._texts = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setProperty("role", "h2")
        self.description_label = QLabel(description)
        self.description_label.setProperty("role", "secondary")
        self.description_label.setWordWrap(True)
        self._texts.addWidget(self.title_label)
        self._texts.addWidget(self.description_label)
        self._layout.addLayout(self._texts, 1)
        self._layout.addWidget(
            control, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_lg"))
        self._texts.setSpacing(self._theme.metric("spacing_xs"))
