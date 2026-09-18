"""The overlay's text command field: type a command, press Enter, same pipeline.

A typed phrase goes through the exact path a spoken one does — the caller wires
:attr:`CommandInput.submitted` to the pipeline's text entry point. The field
keeps a small history browsable with the up/down arrows, and gains focus the
normal way when the panel is opened to type.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLineEdit, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["CommandInput"]

_MAX_HISTORY: Final = 50


class CommandInput(QLineEdit):
    """A single-line command field with arrow-key history."""

    submitted = Signal(str)

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._history: list[str] = []
        self._browse_index: int | None = None
        self._draft = ""
        self.setPlaceholderText("Введите команду…")
        self.setAccessibleName("Текстовая команда")
        self.setClearButtonEnabled(True)
        self.returnPressed.connect(self._submit)
        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    @property
    def history(self) -> tuple[str, ...]:
        return tuple(self._history)

    def _submit(self) -> None:
        text = self.text().strip()
        if not text:
            return
        if not self._history or self._history[-1] != text:
            self._history.append(text)
            del self._history[:-_MAX_HISTORY]
        self._browse_index = None
        self._draft = ""
        self.clear()
        self.submitted.emit(text)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Up:
            self._browse(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Down:
            self._browse(1)
            event.accept()
            return
        super().keyPressEvent(event)

    def _browse(self, direction: int) -> None:
        if not self._history:
            return
        if self._browse_index is None:
            if direction > 0:
                return
            self._draft = self.text()
            self._browse_index = len(self._history) - 1
        else:
            self._browse_index += direction
        if self._browse_index < 0:
            self._browse_index = 0
        elif self._browse_index >= len(self._history):
            self._browse_index = None
            self.setText(self._draft)
            return
        self.setText(self._history[self._browse_index])
        self.end(False)

    def _refresh_theme(self, _theme: object | None = None) -> None:
        height = self._theme.metric("control_height")
        self.setMinimumHeight(height)
