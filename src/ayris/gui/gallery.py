"""Visual gallery for every shared theme token and widget."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal, cast

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris import __app_name__
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    BusyIndicator,
    ConfirmDialog,
    EmptyState,
    IconButton,
    InlineNotice,
    SearchField,
    SettingCard,
    SliderField,
    Toast,
    ToggleSwitch,
)
from ayris.utils.dpi import enable_per_monitor_dpi_awareness

_THEMES = {"Тёмная": "dark", "Светлая": "light", "Как в системе": "system"}
_SCALES = {"100 %": 1.0, "125 %": 1.25, "150 %": 1.5, "200 %": 2.0}


class GalleryWindow(QMainWindow):
    def __init__(self, manager: ThemeManager) -> None:
        super().__init__()
        self._theme = manager
        self._toast: Toast | None = None
        self.setWindowTitle("Ayris — галерея интерфейса")
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        self._layout = QVBoxLayout(content)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(content)
        self.setCentralWidget(scroll)
        self._build_toolbar()
        self._build_tokens()
        self._build_widgets()
        manager.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _build_toolbar(self) -> None:
        row = QHBoxLayout()
        heading = QLabel("Галерея компонентов")
        heading.setProperty("role", "h1")
        row.addWidget(heading, 1)
        theme_box = QComboBox()
        theme_box.addItems(list(_THEMES))
        theme_box.setAccessibleName("Тема")
        theme_box.currentTextChanged.connect(
            lambda text: self._theme.set_mode(
                cast(Literal["dark", "light", "system"], _THEMES[text])
            )
        )
        scale_box = QComboBox()
        scale_box.addItems(list(_SCALES))
        scale_box.setAccessibleName("Масштаб предпросмотра")
        scale_box.currentTextChanged.connect(lambda text: self._theme.set_scale(_SCALES[text]))
        row.addWidget(QLabel("Тема"))
        row.addWidget(theme_box)
        row.addWidget(QLabel("Масштаб"))
        row.addWidget(scale_box)
        self._layout.addLayout(row)

    def _build_tokens(self) -> None:
        heading = QLabel("Цветовые токены")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)
        self._tokens = QWidget()
        self._token_grid = QGridLayout(self._tokens)
        self._layout.addWidget(self._tokens)
        self._populate_tokens()

    def _populate_tokens(self) -> None:
        while self._token_grid.count():
            item = self._token_grid.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        for index, (name, value) in enumerate(self._theme.theme.colors.model_dump().items()):
            sample = QLabel(f"{name}\n{value}")
            sample.setProperty("card", True)
            sample.setAlignment(Qt.AlignmentFlag.AlignCenter)
            sample.setStyleSheet(
                "background-color: "
                + value
                + "; color: "
                + self._theme.theme.color(
                    "on_accent" if name.startswith("accent") else "text_primary"
                )
                + ";"
            )
            self._token_grid.addWidget(sample, index // 4, index % 4)

    def _build_widgets(self) -> None:
        heading = QLabel("Общие виджеты")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)
        toggle = ToggleSwitch(self._theme, checked=True, label="Голосовые ответы")
        self._layout.addWidget(
            SettingCard(
                "Голосовые ответы",
                "Айрис будет озвучивать результат команды.",
                toggle,
                self._theme,
            )
        )
        self._layout.addWidget(
            SliderField(self._theme, value=65, unit="%", label="Громкость ответа")
        )
        self._layout.addWidget(SearchField(placeholder="Найти команду", theme=self._theme))
        style = self.style()
        icon = (
            QIcon() if style is None else style.standardIcon(style.StandardPixmap.SP_BrowserReload)
        )
        icon_button = IconButton(icon, "Обновить список", self._theme)
        self._layout.addWidget(icon_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._layout.addWidget(InlineNotice("Настройки сохранены.", self._theme, kind="success"))
        self._layout.addWidget(
            InlineNotice(
                "Микрофон недоступен. Проверьте разрешение Windows.",
                self._theme,
                kind="warning",
            )
        )
        self._layout.addWidget(
            EmptyState(
                "Команд пока нет",
                "Создайте первую команду или импортируйте готовый профиль.",
                self._theme,
                action_text="Создать команду",
            )
        )
        busy_row = QHBoxLayout()
        busy_row.addWidget(BusyIndicator(self._theme))
        busy_row.addWidget(QLabel("Загружаем модели…"), 1)
        self._layout.addLayout(busy_row)
        toast_button = QPushButton("Показать тост")
        toast_button.clicked.connect(self._show_toast)
        confirm_button = QPushButton("Открыть подтверждение")
        confirm_button.setProperty("kind", "danger")
        confirm_button.clicked.connect(self._show_confirm)
        actions = QHBoxLayout()
        actions.addWidget(toast_button)
        actions.addWidget(confirm_button)
        actions.addStretch()
        self._layout.addLayout(actions)

    def _show_toast(self) -> None:
        if self._toast is not None:
            self._toast.close()
        self._toast = Toast(
            "Готово", "Новая тема применена без перезапуска.", self._theme, kind="success"
        )
        self._toast.show()

    def _show_confirm(self) -> None:
        dialog = ConfirmDialog(
            "Удалить профиль?",
            "Команды и настройки профиля будут удалены без возможности восстановления.",
            self._theme,
            confirm_text="Удалить",
            dangerous=True,
            parent=self,
        )
        dialog.exec()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_lg"))
        self._token_grid.setSpacing(self._theme.metric("spacing_sm"))
        self.setMinimumSize(
            self._theme.metric("window_min_width"),
            self._theme.metric("window_min_height"),
        )
        self._populate_tokens()


def main(argv: Sequence[str] | None = None) -> int:
    enable_per_monitor_dpi_awareness()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    arguments = list(argv) if argv is not None else sys.argv
    application = QApplication(arguments)
    application.setApplicationName(f"{__app_name__} — галерея")
    manager = ThemeManager(application)
    manager.apply()
    window = GalleryWindow(manager)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
