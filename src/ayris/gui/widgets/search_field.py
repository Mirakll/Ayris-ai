"""Search field with a semantic icon and clear action."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLineEdit, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["SearchField"]


class SearchField(QLineEdit):
    search_changed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        placeholder: str = "Поиск",
        theme: ThemeManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setPlaceholderText(placeholder)
        self.setAccessibleName("Поиск")
        self.setClearButtonEnabled(True)
        if theme is not None:
            theme.theme_changed.connect(self._refresh_icon)
            self._refresh_icon()
        self.textChanged.connect(self.search_changed)

    def _refresh_icon(self, _theme: object | None = None) -> None:
        if self._theme is None:
            return
        size = self._theme.metric("icon_sm")
        stroke = self._theme.metric("border_width")
        inset = self._theme.metric("spacing_xs")
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(self._theme.theme.color("text_secondary")))
        pen.setWidth(stroke)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        lens = max(stroke, size - inset * 2)
        painter.drawEllipse(QRectF(inset, inset, lens, lens))
        handle_start = inset + lens
        handle_end = size - inset
        painter.drawLine(handle_start, handle_start, handle_end, handle_end)
        painter.end()
        self.addAction(QIcon(pixmap), QLineEdit.ActionPosition.LeadingPosition)
