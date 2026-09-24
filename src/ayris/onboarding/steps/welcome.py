"""Шаг «Приветствие»: сфера, вордмарк и короткое введение.

Ничего не пишет в конфиг — только рассказывает, что будет дальше (микрофон и
модели), и напоминает, что Айрис пока только на русском. Сферу анимируем лишь
пока шаг виден: таймер запускается в ``activate`` и гасится в ``deactivate`` и
``teardown``, чтобы не крутить кадры на скрытом шаге и не держать таймер в тестах.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere import PainterSphereWidget, SphereState
from ayris.onboarding.steps._common import caption
from ayris.onboarding.wizard import WizardStep

__all__ = ["WelcomeStep"]


class WelcomeStep(WizardStep):
    """Первый экран мастера: приветствие и обзор дальнейших шагов."""

    def __init__(self, theme: ThemeManager, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = "welcome"
        self.title = "Приветствие"
        self._theme = theme

        self._sphere = PainterSphereWidget(theme, point_count=520, parent=self)
        self._sphere.setFixedSize(220, 220)
        self._sphere.set_animations_enabled(False)

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addStretch(1)
        layout.addWidget(self._sphere, 0, Qt.AlignmentFlag.AlignHCenter)

        wordmark = QLabel("Айрис")
        wordmark.setProperty("role", "h1")
        wordmark.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(wordmark)

        intro = caption(
            "Голосовой помощник, который работает на вашем компьютере и говорит "
            "по-русски. Сейчас коротко настроим микрофон и загрузим базовые модели — "
            "это всё, что нужно для первого запуска."
        )
        intro.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(intro)

        note = caption("Пока Айрис понимает и отвечает только на русском языке.")
        note.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(note)
        layout.addStretch(1)

    def can_skip(self) -> bool:
        return False

    def activate(self) -> None:
        self._sphere.set_animations_enabled(True)
        self._sphere.set_state(SphereState.LISTENING)

    def deactivate(self) -> None:
        self._sphere.set_animations_enabled(False)

    def teardown(self) -> None:
        self._sphere.set_animations_enabled(False)
