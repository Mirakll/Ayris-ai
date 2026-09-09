"""Confirmation dialog with an explicit destructive variant."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager

__all__ = ["ConfirmDialog"]


class ConfirmDialog(QDialog):
    def __init__(
        self,
        title: str,
        text: str,
        theme: ThemeManager,
        *,
        confirm_text: str = "Подтвердить",
        dangerous: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowTitle(title)
        self.setAccessibleName(title)
        self.setModal(True)
        self._layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        body = QLabel(text)
        body.setProperty("role", "secondary")
        body.setWordWrap(True)
        self._layout.addWidget(heading)
        self._layout.addWidget(body)
        buttons = QDialogButtonBox()
        self.confirm_button = QPushButton(confirm_text)
        self.confirm_button.setProperty("kind", "danger" if dangerous else "primary")
        self.cancel_button = QPushButton("Отмена")
        self.confirm_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        buttons.addButton(self.cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(self.confirm_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self._layout.addWidget(buttons)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_lg"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))
