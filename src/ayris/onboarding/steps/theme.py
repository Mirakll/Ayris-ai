"""Шаг «Тема»: выбор оформления с живым предпросмотром.

Переключатель применяет тему сразу (через :class:`ThemeManager`), чтобы было
видно результат, но в конфиг (``general.theme``) пишет только ``apply()`` на
«Далее». Если шаг покинули, не применив (Назад/Пропустить), предпросмотр
откатывается к сохранённому значению — так брошенный шаг не оставляет тему,
которую пользователь на самом деле не выбрал.
"""

from __future__ import annotations

from PySide6.QtWidgets import QButtonGroup, QRadioButton, QVBoxLayout, QWidget

from ayris.core.config import ConfigManager
from ayris.gui.theme import ThemeManager
from ayris.onboarding.steps._common import caption, heading
from ayris.onboarding.wizard import WizardStep

__all__ = ["ThemeStep"]


class ThemeStep(WizardStep):
    """Выбор светлой/тёмной/системной темы с мгновенным предпросмотром."""

    #: (значение конфига, заголовок, пояснение)
    _OPTIONS: tuple[tuple[str, str, str], ...] = (
        (
            "dark_purple",
            "Тёмная (фиолетовая)",
            "Оформление по умолчанию: глубокий фиолетовый фон, мягкий контраст.",
        ),
        ("light", "Светлая", "Светлый фон — удобно в ярко освещённой комнате."),
        (
            "system",
            "Как в системе",
            "Следовать светлой или тёмной теме Windows и переключаться автоматически.",
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
        self.key = "theme"
        self.title = "Тема"
        self._theme = theme
        self._config = config
        self._applied = False
        self._buttons: dict[str, QRadioButton] = {}

        layout = QVBoxLayout(self)
        layout.setSpacing(theme.metric("spacing_lg"))
        layout.addWidget(heading("Выберите оформление"))
        layout.addWidget(
            caption("Тему можно сменить в любой момент в настройках. Результат виден сразу.")
        )

        group = QButtonGroup(self)
        current = self._config.settings.general.theme
        for value, title, desc in self._OPTIONS:
            button = QRadioButton(title)
            selected = value == current or (value == "dark_purple" and current == "dark")
            button.setChecked(selected)
            group.addButton(button)
            self._buttons[value] = button
            layout.addWidget(button)
            layout.addWidget(caption(desc))
        layout.addStretch(1)

        # Сигналы подключаем ПОСЛЕ начальной установки, чтобы setChecked выше не
        # дёргал set_mode во время построения.
        for value, button in self._buttons.items():
            button.toggled.connect(
                lambda checked, v=value: self._set_live_mode(v) if checked else None
            )

    def _selected(self) -> str:
        for value, button in self._buttons.items():
            if button.isChecked():
                return value
        return "dark_purple"

    def _set_live_mode(self, value: str) -> None:
        if value == "light":
            self._theme.set_mode("light")
        elif value == "system":
            self._theme.set_mode("system")
        else:
            self._theme.set_mode("dark")

    def activate(self) -> None:
        # Каждый визит оценивается заново: применили на этом визите или нет.
        self._applied = False

    def apply(self) -> None:
        self._applied = True
        self._config.apply({"general.theme": self._selected()})

    def deactivate(self) -> None:
        if not self._applied:
            self._set_live_mode(self._config.settings.general.theme)
