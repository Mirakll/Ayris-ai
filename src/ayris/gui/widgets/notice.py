"""Inline and floating notifications with optional auto-hide."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["InlineNotice", "Toast"]

NoticeKind = Literal["info", "warning", "error", "success"]


class InlineNotice(QFrame):
    closed = Signal()

    def __init__(
        self,
        text: str,
        theme: ThemeManager,
        *,
        kind: NoticeKind = "info",
        auto_hide_ms: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("notice", True)
        self.setProperty("status", kind)
        self.setAccessibleName(text)
        self._layout = QHBoxLayout(self)
        self.label = QLabel(text)
        self.label.setWordWrap(True)
        self._layout.addWidget(self.label, 1)
        self.close_button = QPushButton("Закрыть")
        self.close_button.setAccessibleName("Закрыть уведомление")
        self.close_button.clicked.connect(self.dismiss)
        self._layout.addWidget(self.close_button)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        if auto_hide_ms > 0:
            QTimer.singleShot(auto_hide_ms, self.dismiss)

    def dismiss(self) -> None:
        if self.isHidden():
            return
        self.hide()
        self.closed.emit()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_sm")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(pad)
        self.setMinimumHeight(self._theme.metric("notice_min_height"))


class Toast(QFrame):
    closed = Signal()

    def __init__(
        self,
        title: str,
        text: str,
        theme: ThemeManager,
        *,
        kind: NoticeKind = "info",
        auto_hide_ms: int = 4000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("notice", True)
        self.setProperty("status", kind)
        self.setWindowFlags(Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAccessibleName(f"{title}. {text}")
        self._layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        body = QLabel(text)
        body.setWordWrap(True)
        body.setProperty("role", "secondary")
        self._layout.addWidget(heading)
        self._layout.addWidget(body)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        if auto_hide_ms > 0:
            QTimer.singleShot(auto_hide_ms, self.dismiss)

    def dismiss(self) -> None:
        if self.isHidden():
            return
        self.hide()
        self.closed.emit()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_xs"))
        self.setFixedWidth(self._theme.metric("toast_width"))
