"""QSS generation and live theme application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

from ayris.gui.theme.system import SystemThemeWatcher
from ayris.gui.theme.tokens import Theme, bundled_theme_path, load_theme

__all__ = [
    "DEFAULT_QSS_TEMPLATE",
    "ThemeManager",
    "render_qss",
    "resolve_font_family",
]

ThemeMode = Literal["dark", "light", "system"]
_TOKEN = re.compile(r"{{\s*(color|metric|typography)\.([a-zA-Z0-9_]+)\s*}}")

DEFAULT_QSS_TEMPLATE = """
* {
    color: {{color.text_primary}};
    font-family: "{{typography.family}}";
    font-size: {{typography.body_size}}px;
}
QWidget { background-color: {{color.background}}; }
QLabel, QSlider, QRadioButton, QCheckBox, SliderField, ToggleSwitch, DownloadProgress,
ModelManager { background: transparent; }
/* Layout-only containers opt out of the window fill so they don't paint a dark
   rectangle on top of a card. */
QWidget[transparent="true"], QFrame[transparent="true"] { background: transparent; }
QWidget[card="true"], QFrame[card="true"] {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
}
QLabel[role="secondary"] { color: {{color.text_secondary}}; }
QLabel[role="muted"] {
    color: {{color.text_muted}};
    font-size: {{typography.caption_size}}px;
}
QLabel[role="h1"] {
    font-size: {{typography.h1_size}}px;
    font-weight: {{typography.weight_bold}};
}
QLabel[role="h2"] {
    font-size: {{typography.h2_size}}px;
    font-weight: {{typography.weight_bold}};
}
QLabel[status="warning"] { color: {{color.warning}}; }
QListWidget#settingsSidebar, QListWidget {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
    outline: none;
    padding: {{metric.spacing_xs}}px;
}
QListWidget#settingsSidebar::item {
    border-radius: {{metric.radius_md}}px;
    padding-left: {{metric.spacing_md}}px;
}
QListWidget#settingsSidebar::item:hover { background-color: {{color.surface_highlight}}; }
QListWidget#settingsSidebar::item:selected {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
}
QPushButton, QLineEdit, QSpinBox, QComboBox {
    min-height: {{metric.control_height}}px;
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
    background-color: {{color.surface}};
    padding-left: {{metric.spacing_md}}px;
    padding-right: {{metric.spacing_md}}px;
}
QPushButton:hover, QLineEdit:hover, QSpinBox:hover, QComboBox:hover {
    border-color: {{color.accent_hover}};
    background-color: {{color.surface_highlight}};
}
QPushButton:pressed { border-color: {{color.accent_pressed}}; }
QPushButton:focus, QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: {{metric.focus_width}}px solid {{color.focus}};
}
QPushButton:disabled, QLineEdit:disabled, QSpinBox:disabled {
    color: {{color.text_muted}};
    border-color: {{color.accent_disabled}};
}
/* Combobox: шеврон рисует сам ThemedComboBox (widgets/combo_box.py) в чипе
   справа. Стилю запрещаем рисовать любую стрелку и дроп-даун — иначе на дробном
   DPI (125 %, 150 %) QStyleSheetStyle всё равно нарисует свою стрелку по центру
   поля поверх нашей: те самые «две стрелки». Ноль ширины и image: none гасят её
   для любого QComboBox, в том числе на будущих страницах. */
QComboBox {
    min-height: {{metric.control_height_lg}}px;
    padding-right: {{metric.spacing_xs}}px;
}
QComboBox::drop-down {
    width: 0;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
/* ThemedComboBox — единственный, кто рисует стрелку; кормим ему цвета и размеры
   Qt-свойствами, чтобы он оставался темозависимым. */
ThemedComboBox {
    qproperty-chipColor: {{color.surface_highlight}};
    qproperty-chipColorActive: {{color.accent}};
    qproperty-arrowColor: {{color.text_secondary}};
    qproperty-arrowColorActive: {{color.on_accent}};
    qproperty-arrowColorDisabled: {{color.text_muted}};
    qproperty-chipSize: {{metric.control_height}};
    qproperty-chipInset: {{metric.spacing_xs}};
    qproperty-chipRadius: {{metric.radius_sm}};
    qproperty-arrowSize: {{metric.icon_sm}};
}
QComboBox QAbstractItemView {
    background: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
    padding: {{metric.spacing_xs}}px;
    outline: none;
    selection-background-color: {{color.accent}};
    selection-color: {{color.on_accent}};
}
QPushButton[kind="primary"] {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
    border-color: {{color.accent}};
    font-weight: {{typography.weight_medium}};
}
QPushButton[kind="primary"]:hover {
    background-color: {{color.accent_hover}};
    border-color: {{color.accent_hover}};
}
QPushButton[kind="primary"]:pressed {
    background-color: {{color.accent_pressed}};
    border-color: {{color.accent_pressed}};
}
QPushButton[kind="danger"] {
    color: {{color.on_accent}};
    background-color: {{color.error}};
    border-color: {{color.error}};
    font-weight: {{typography.weight_medium}};
}
QPushButton[iconButton="true"] {
    min-width: {{metric.control_height}}px;
    max-width: {{metric.control_height}}px;
    padding: 0;
}
QFrame[notice="true"] {
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
}
QFrame[status="info"] { border-color: {{color.info}}; }
QFrame[status="warning"] { border-color: {{color.warning}}; }
QFrame[status="error"] { border-color: {{color.error}}; }
QFrame[status="success"] { border-color: {{color.success}}; }
/* Прогресс-бар загрузки модели: тема, не системный серый квадрат. */
QProgressBar {
    min-height: {{metric.spacing_md}}px;
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_sm}}px;
    background: {{color.surface_highlight}};
    text-align: center;
    color: {{color.text_primary}};
}
QProgressBar::chunk {
    background: {{color.accent}};
    border-radius: {{metric.radius_sm}}px;
}
/* Пилюля-статус модели: цвет несёт смысл, фон приглушён, без тёмного квадрата. */
QLabel[badge="true"] {
    padding: {{metric.spacing_xs}}px {{metric.spacing_sm}}px;
    border-radius: {{metric.radius_sm}}px;
    background: {{color.surface_highlight}};
    color: {{color.text_secondary}};
    font-size: {{typography.caption_size}}px;
    font-weight: {{typography.weight_medium}};
}
QLabel[badge="success"] { color: {{color.success}}; }
QLabel[badge="warning"] { color: {{color.warning}}; }
QLabel[badge="error"] { color: {{color.error}}; }
QLabel[badge="info"] { color: {{color.info}}; }
QLabel[badge="muted"] { color: {{color.text_muted}}; }
QSlider::groove:horizontal {
    height: {{metric.spacing_xs}}px;
    background: {{color.surface_highlight}};
    border-radius: {{metric.radius_sm}}px;
}
QSlider::sub-page:horizontal {
    background: {{color.accent}};
    border-radius: {{metric.radius_sm}}px;
}
QSlider::handle:horizontal {
    width: {{metric.spacing_md}}px;
    margin: -6px 0;
    border-radius: 6px;
    background: {{color.accent}};
}
QScrollBar:vertical {
    width: {{metric.spacing_md}}px;
    background: {{color.background}};
    border-radius: 6px;
    margin: 0;
}
QScrollBar:horizontal {
    height: {{metric.spacing_md}}px;
    background: {{color.background}};
    border-radius: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    min-height: {{metric.control_height_lg}}px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent_hover}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::handle:horizontal {
    min-width: {{metric.control_height_lg}}px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent_hover}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
    background: transparent;
    border: none;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: transparent;
}
QToolTip {
    color: {{color.text_primary}};
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    padding: {{metric.spacing_sm}}px;
}
QMenu {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
    padding: {{metric.spacing_xs}}px;
    font-weight: {{typography.weight_medium}};
}
QMenu::item {
    padding: 9px 18px 9px 12px;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    border-radius: {{metric.radius_sm}}px;
    color: {{color.text_primary}};
}
QMenu::item:selected {
    background-color: {{color.surface_highlight}};
}
QMenu::item:checked {
    color: {{color.accent}};
    border-left: 3px solid {{color.accent}};
}
QMenu::item:checked:selected {
    background-color: {{color.surface_highlight}};
}
QMenu::item:disabled {
    color: {{color.text_muted}};
}
QMenu::icon {
    padding-left: {{metric.spacing_sm}}px;
}
QMenu::indicator {
    width: 0px;
    height: 0px;
}
QMenu::separator {
    height: {{metric.border_width}}px;
    background: {{color.border}};
    margin: {{metric.spacing_sm}}px {{metric.spacing_sm}}px;
}
QMenu::right-arrow {
    width: {{metric.spacing_sm}}px;
    height: {{metric.spacing_sm}}px;
    margin-right: {{metric.spacing_md}}px;
}
""".strip()


def resolve_font_family(theme: Theme) -> str:
    """Choose the first installed family, always ending at a system fallback."""
    installed = {family.casefold(): family for family in QFontDatabase.families()}
    for candidate in (theme.typography.family, *theme.typography.fallbacks, "Segoe UI"):
        if candidate.casefold() in installed:
            return installed[candidate.casefold()]
    return QFont().defaultFamily()


def render_qss(theme: Theme, template: str = DEFAULT_QSS_TEMPLATE, *, scale: float = 1.0) -> str:
    """Substitute semantic tokens into QSS and reject unknown placeholders."""
    font_family = resolve_font_family(theme)

    def replace(match: re.Match[str]) -> str:
        group, name = match.groups()
        if group == "color":
            return theme.color(name)
        if group == "metric":
            return str(theme.metric(name, scale=scale))
        value = theme.type_value(name)
        if name == "family":
            return font_family
        if isinstance(value, tuple):
            return ", ".join(value)
        if isinstance(value, int | float) and name.endswith("_size"):
            return str(max(1, round(value * scale)))
        return str(value)

    rendered = _TOKEN.sub(replace, template)
    unresolved = _TOKEN.search(rendered)
    if unresolved is not None:
        raise KeyError(f"Не удалось подставить токен {unresolved.group(0)}")
    return rendered


class ThemeManager(QObject):
    """Own the current theme, stylesheet, system mode and DPI preview scale."""

    theme_changed = Signal(object)

    def __init__(
        self,
        application: QApplication,
        *,
        theme_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._application = application
        # Base every widget on Fusion. The native Windows 11 style paints its own
        # combobox arrow that QSS cannot suppress, so a themed chevron leaves the
        # native one behind — two arrows on every combo. Fusion is a QSS-friendly
        # base the stylesheet fully owns, so the chevron is the only arrow and the
        # rest of the dark theme applies cleanly too.
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            application.setStyle(fusion)
        self._theme_dir = theme_dir
        self._mode: ThemeMode = "dark"
        self._scale = 1.0
        self._theme = self._load_named("dark_purple")
        self._watcher = SystemThemeWatcher(parent=self)
        self._watcher.changed.connect(self._system_changed)

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def scale(self) -> float:
        return self._scale

    def metric(self, name: str) -> int:
        return self._theme.metric(name, scale=self._scale)

    def set_scale(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError("Масштаб интерфейса должен быть больше нуля")
        if scale == self._scale:
            return
        self._scale = scale
        self._apply()

    def set_mode(self, mode: ThemeMode) -> None:
        if mode not in ("dark", "light", "system"):
            raise ValueError(f"Неизвестный режим темы: {mode}")
        self._mode = mode
        if mode == "system":
            self._watcher.start()
            selected = self._watcher.current
        else:
            self._watcher.stop()
            selected = mode
        self._theme = self._load_named("light" if selected == "light" else "dark_purple")
        self._apply()

    def apply(self) -> None:
        self._apply()

    def _load_named(self, name: str) -> Theme:
        path = self._theme_dir / f"{name}.json" if self._theme_dir else bundled_theme_path(name)
        return load_theme(path)

    def _system_changed(self, mode: str) -> None:
        if self._mode != "system":
            return
        self._theme = self._load_named("light" if mode == "light" else "dark_purple")
        self._apply()

    def _apply(self) -> None:
        qss = render_qss(self._theme, scale=self._scale)
        self._application.setStyleSheet(qss)
        font = QFont(resolve_font_family(self._theme))
        font.setPixelSize(max(1, round(self._theme.typography.body_size * self._scale)))
        self._application.setFont(font)
        style = self._application.style()
        if style is not None:
            for widget in self._application.allWidgets():
                style.unpolish(widget)
                style.polish(widget)
                widget.repaint()
        self.theme_changed.emit(self._theme)


def screen_scale(widget: QWidget) -> float:
    """DPI ratio for manual painting; Qt scales QSS logical units itself."""
    screen = widget.screen()
    return screen.logicalDotsPerInch() / 96.0 if screen is not None else 1.0
