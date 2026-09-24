"""Шаг «Режим»: онлайн / офлайн / авто с честными последствиями.

Пишет ``voice.stt.mode``. Текст под каждым вариантом прямо говорит, что уходит в
интернет и что придётся загрузить, чтобы выбор был осознанным. Режим влияет на
то, как шаг моделей подаёт локальную модель распознавания (в онлайне она
необязательна), но сам набор моделей шаг не трогает.
"""

from __future__ import annotations

from PySide6.QtWidgets import QButtonGroup, QRadioButton, QVBoxLayout, QWidget

from ayris.core.config import ConfigManager
from ayris.gui.theme import ThemeManager
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep

__all__ = ["ModeStep"]


class ModeStep(WizardStep):
    """Выбор режима распознавания речи."""

    #: (значение конфига, заголовок, пояснение)
    _OPTIONS: tuple[tuple[str, str, str], ...] = (
        (
            "auto",
            "Авто (рекомендуется)",
            "Сначала пробует облако, а без сети или ключа переключается на локальные "
            "модели. Нужен интернет, а на случай отката — базовые локальные модели.",
        ),
        (
            "offline",
            "Только офлайн",
            "Всё распознавание идёт на вашем компьютере, звук не уходит в интернет. "
            "Потребуется загрузить локальные модели (около 280 МБ).",
        ),
        (
            "online",
            "Только онлайн",
            "Распознавание в облаке выбранного провайдера: быстро и без крупных "
            "загрузок, но аудио уходит в интернет и нужен ключ доступа.",
        ),
    )

    def __init__(
        self,
        theme: ThemeManager,
        config: ConfigManager,
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.key = "mode"
        self.title = "Режим"
        self._config = config
        self._buttons: dict[str, QRadioButton] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Как Айрис будет распознавать речь"))
        layout.addWidget(caption("Это можно изменить позже на вкладке «Голос»."))

        group = QButtonGroup(self)
        current = self._config.settings.voice.stt.mode
        for value, title, desc in self._OPTIONS:
            button = QRadioButton(title)
            button.setChecked(value == current)
            group.addButton(button)
            self._buttons[value] = button
            layout.addWidget(button)
            layout.addWidget(caption(desc))
        layout.addStretch(1)

    def _selected(self) -> str:
        for value, button in self._buttons.items():
            if button.isChecked():
                return value
        return "auto"

    def apply(self) -> None:
        self._config.apply({"voice.stt.mode": self._selected()})
