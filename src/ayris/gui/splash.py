"""Заставка с лого и сферой на время инициализации.

Кадр без рамки: по центру экрана сфера, вордмарк «Айрис» и строка этапа. Она
только показывает то, что и так происходит при старте, и не должна задерживать
запуск — поэтому вся тяжёлая работа идёт в вызывающем коде, а заставка лишь
рисует состояние и меняет подпись через :meth:`set_stage`. Сфера анимируется,
пока кадр виден; :meth:`finish` гасит таймер и закрывает окно.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import QApplication, QFrame, QLabel, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere import PainterSphereWidget, SphereState

__all__ = ["SplashScreen"]


class SplashScreen(QWidget):
    """Безрамочная заставка запуска."""

    def __init__(self, theme: ThemeManager, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setWindowFlags(Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        card = QFrame(self)
        card.setObjectName("splashCard")
        card.setStyleSheet(
            f"#splashCard {{ background: {theme.theme.color('surface')};"
            f" border: 1px solid {theme.theme.color('border')};"
            f" border-radius: {theme.metric('spacing_md')}px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(card)

        inner = QVBoxLayout(card)
        margin = theme.metric("spacing_2xl")
        inner.setContentsMargins(margin, margin, margin, margin)
        inner.setSpacing(theme.metric("spacing_lg"))

        self._sphere = PainterSphereWidget(theme, point_count=520, parent=card)
        self._sphere.setFixedSize(200, 200)
        self._sphere.set_animations_enabled(False)
        inner.addWidget(self._sphere, 0, Qt.AlignmentFlag.AlignHCenter)

        wordmark = QLabel("Айрис")
        wordmark.setProperty("role", "h1")
        wordmark.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        inner.addWidget(wordmark)

        self._stage = QLabel("Запуск…")
        self._stage.setProperty("role", "secondary")
        self._stage.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        inner.addWidget(self._stage)

        self.setFixedSize(360, 360)
        self._centre_on_screen()

    def _centre_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is not None:
            centre = screen.availableGeometry().center()
            self.move(centre.x() - self.width() // 2, centre.y() - self.height() // 2)

    def set_stage(self, text: str) -> None:
        """Обновить подпись этапа (например «Поднимаем воркеры…»)."""
        self._stage.setText(text)

    def finish(self) -> None:
        """Погасить анимацию и закрыть заставку."""
        self._sphere.set_animations_enabled(False)
        self.close()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 — Qt-хук
        super().showEvent(event)
        self._sphere.set_animations_enabled(True)
        self._sphere.set_state(SphereState.THINKING)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 — Qt-хук
        self._sphere.set_animations_enabled(False)
        super().closeEvent(event)
