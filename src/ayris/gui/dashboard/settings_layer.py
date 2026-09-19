"""Settings as a full-window layer that slides down over the dashboard.

The hamburger opens this; the same button, the close control or Esc send it back
up. While open it covers the whole window with a dimmed backdrop, so there is
never a second window. The layer only provides the chrome and the slide — the
search field, section sidebar and page stack are put into :attr:`body_layout` by
:class:`~ayris.gui.main_window.MainWindow`, which still owns that logic.
"""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["SettingsLayer"]


class SettingsLayer(QWidget):
    """A dimmed, slide-down container for the settings UI."""

    closed = Signal()

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._open = False
        self.setObjectName("settingsBackdrop")
        self.setAutoFillBackground(True)
        self.hide()

        self._content = QFrame(self)
        self._content.setObjectName("settingsLayer")
        content_layout = QVBoxLayout(self._content)

        header = QHBoxLayout()
        self._title = QLabel("Настройки", self._content)
        self._title.setObjectName("settingsTitle")
        self.close_button = QPushButton("✕", self._content)  # ✕
        self.close_button.setObjectName("settingsClose")
        self.close_button.setAccessibleName("Закрыть настройки")
        self.close_button.setToolTip("Закрыть настройки")
        self.close_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_button.clicked.connect(self.close_layer)
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self.close_button)
        content_layout.addLayout(header)

        self.body_layout = QVBoxLayout()
        content_layout.addLayout(self.body_layout, 1)
        self._content_layout = content_layout

        self._anim = QPropertyAnimation(self._content, b"pos", self)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    @property
    def is_open(self) -> bool:
        return self._open

    def open_layer(self) -> None:
        if self._open:
            return
        self._open = True
        self._sync_content_geometry(offscreen=True)
        self.show()
        self.raise_()
        self._animate_to(0)
        self.close_button.setFocus(Qt.FocusReason.OtherFocusReason)

    def close_layer(self) -> None:
        if not self._open:
            return
        self._open = False
        self._animate_to(-self.height())

    def toggle(self) -> None:
        self.close_layer() if self._open else self.open_layer()

    def _animate_to(self, target_y: int) -> None:
        # Выезд слоя настроек чуть медленнее общего animation_normal (200 мс),
        # чтобы движение читалось спокойнее — только для этого слоя.
        duration = round(self._theme.metric("animation_normal") * 1.6)
        self._anim.stop()
        self._anim.setDuration(duration)
        self._anim.setStartValue(self._content.pos())
        self._anim.setEndValue(QPoint(0, target_y))
        self._anim.start()

    def _on_anim_finished(self) -> None:
        if not self._open:
            self.hide()
            self.closed.emit()

    def _sync_content_geometry(self, *, offscreen: bool) -> None:
        y = -self.height() if offscreen else (0 if self._open else -self.height())
        self._content.setGeometry(0, y, self.width(), self.height())

    def resizeEvent(self, event: object) -> None:  # noqa: N802, ARG002
        self._content.resize(self.width(), self.height())
        if self._anim.state() != QPropertyAnimation.State.Running:
            self._content.move(0, 0 if self._open else -self.height())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape and self._open:
            self.close_layer()
            event.accept()
            return
        super().keyPressEvent(event)

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        typography = self._theme.theme.typography
        pad = metric("spacing_lg")

        self._content_layout.setContentsMargins(pad, pad, pad, pad)
        self._content_layout.setSpacing(metric("spacing_lg"))
        self.setStyleSheet(
            f"#settingsBackdrop {{ background: {color('overlay')}; }}"
            f"#settingsLayer {{ background: {color('background')}; }}"
            f"#settingsTitle {{ color: {color('text_primary')};"
            f" font-size: {typography.h2_size}px; font-weight: {typography.weight_bold}; }}"
            f"#settingsClose {{ background: transparent; border: none;"
            f" color: {color('text_secondary')}; font-size: {typography.h2_size}px;"
            f" border-radius: {metric('radius_md')}px; }}"
            f"#settingsClose:hover {{ color: {color('accent')};"
            f" background: {color('surface_highlight')}; }}"
        )
        self.close_button.setFixedSize(metric("control_height"), metric("control_height"))
