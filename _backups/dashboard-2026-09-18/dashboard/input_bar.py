"""The dashboard command bar: type or speak, in one rounded field.

A rounded container holds a code-mode glyph, the reused :class:`CommandInput`, a
microphone toggle and an accent send button. Enter or the send button submit the
text down the same path a spoken phrase takes; the microphone toggles voice
input. Everything is drawn from theme tokens, so it re-colours with the theme.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.theme import ThemeManager

__all__ = ["InputBar"]


def _mic_icon(color: str, size: int) -> QIcon:
    """A simple microphone glyph tinted with ``color``."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.4, size * 0.09))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    unit = size / 24.0
    # Capsule head.
    head = QRectF(9 * unit, 3 * unit, 6 * unit, 11 * unit)
    painter.drawRoundedRect(head, 3 * unit, 3 * unit)
    # Cradle arc.
    painter.drawArc(QRectF(6 * unit, 8 * unit, 12 * unit, 10 * unit), 200 * 16, 140 * 16)
    # Stand and base.
    painter.drawLine(int(12 * unit), int(17 * unit), int(12 * unit), int(20 * unit))
    painter.drawLine(int(8.5 * unit), int(20 * unit), int(15.5 * unit), int(20 * unit))
    painter.end()
    return QIcon(pixmap)


def _arrow_up_icon(color: str, size: int) -> QIcon:
    """An upward arrow for the send button."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.6, size * 0.11))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    unit = size / 24.0
    path = QPainterPath()
    path.moveTo(12 * unit, 19 * unit)
    path.lineTo(12 * unit, 6 * unit)
    path.moveTo(6 * unit, 12 * unit)
    path.lineTo(12 * unit, 6 * unit)
    path.lineTo(18 * unit, 12 * unit)
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


class InputBar(QFrame):
    """Rounded input container with code, microphone and send controls."""

    submitted = Signal(str)
    voice_requested = Signal()
    code_requested = Signal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._voice_active = False
        self.setObjectName("inputBar")

        self._layout = QHBoxLayout(self)

        self.code_button = QPushButton("</>", self)
        self.code_button.setObjectName("inputGhost")
        self.code_button.setAccessibleName("Режим команды или кода")
        self.code_button.setToolTip("Режим команды или кода")
        self.code_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.code_button.clicked.connect(self.code_requested.emit)

        self.command = CommandInput(theme, self)
        self.command.setObjectName("inputField")
        self.command.setPlaceholderText("Чем могу помочь сегодня?")
        self.command.setClearButtonEnabled(False)
        self.command.setFrame(False)
        self.command.submitted.connect(self.submitted.emit)

        self.mic_button = QPushButton(self)
        self.mic_button.setObjectName("inputGhost")
        self.mic_button.setAccessibleName("Голосовой ввод")
        self.mic_button.setToolTip("Голосовой ввод")
        self.mic_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_button.clicked.connect(self.voice_requested.emit)

        self.send_button = QPushButton(self)
        self.send_button.setObjectName("inputSend")
        self.send_button.setAccessibleName("Отправить команду")
        self.send_button.setToolTip("Отправить")
        self.send_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_button.clicked.connect(self.command.submit)

        self._layout.addWidget(self.code_button)
        self._layout.addWidget(self.command, 1)
        self._layout.addWidget(self.mic_button)
        self._layout.addWidget(self.send_button)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    def focus_command(self) -> None:
        self.command.setFocus(Qt.FocusReason.OtherFocusReason)

    def set_voice_active(self, active: bool) -> None:
        """Highlight the microphone while voice input is live."""
        if active == self._voice_active:
            return
        self._voice_active = active
        self._refresh_theme()

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        icon = metric("icon_md")
        control = metric("control_height")
        pad = metric("spacing_sm")
        radius_lg = metric("radius_lg")
        radius_md = metric("radius_md")
        focus_width = metric("focus_width")

        surface = color("surface_highlight")
        border = color("border")
        muted = color("text_muted")
        secondary = color("text_secondary")
        text = color("text_primary")
        accent = color("accent")
        accent_hover = color("accent_hover")
        focus = color("focus")
        on_accent = color("on_accent")

        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(metric("spacing_xs"))

        self.setStyleSheet(
            f"#inputBar {{ background: {surface}; border: 1px solid {border};"
            f" border-radius: {radius_lg}px; }}"
            f"#inputBar:focus-within {{ border: {focus_width}px solid {focus}; }}"
            f"#inputField {{ background: transparent; border: none; color: {text};"
            f" selection-background-color: {accent}; }}"
            f"#inputGhost {{ background: transparent; border: none; color: {secondary};"
            f" font-size: {metric('icon_sm')}px; font-weight: 600;"
            f" border-radius: {radius_md}px; }}"
            f"#inputGhost:hover {{ color: {text}; }}"
            f"#inputSend {{ background: {accent}; border: none; border-radius: {radius_md}px; }}"
            f"#inputSend:hover {{ background: {accent_hover}; }}"
        )

        self.command.setPlaceholderText("Чем могу помочь сегодня?")
        self.command.setStyleSheet(f"QLineEdit {{ color: {text}; }}")

        square = QSize(control, control)
        for button in (self.code_button, self.mic_button, self.send_button):
            button.setFixedSize(square)
            button.setIconSize(QSize(icon, icon))

        mic_color = accent if self._voice_active else secondary
        self.mic_button.setIcon(_mic_icon(mic_color, icon))
        self.send_button.setIcon(_arrow_up_icon(on_accent, icon))
        # Nudge the placeholder colour without a dedicated token line.
        palette = self.command.palette()
        palette.setColor(palette.ColorRole.PlaceholderText, QColor(muted))
        self.command.setPalette(palette)
