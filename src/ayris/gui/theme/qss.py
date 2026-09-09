"""QSS generation and live theme application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget

from ayris.gui.theme.system import SystemThemeWatcher
from ayris.gui.theme.tokens import Theme, bundled_theme_path, load_theme

__all__ = ["DEFAULT_QSS_TEMPLATE", "ThemeManager", "render_qss", "resolve_font_family"]

ThemeMode = Literal["dark", "light", "system"]
_TOKEN = re.compile(r"{{\s*(color|metric|typography)\.([a-zA-Z0-9_]+)\s*}}")

DEFAULT_QSS_TEMPLATE = """
* {
    color: {{color.text_primary}};
    font-family: "{{typography.family}}";
    font-size: {{typography.body_size}}px;
}
QWidget { background-color: {{color.background}}; }
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
    width: {{metric.icon_sm}}px;
    margin: -{{metric.radius_sm}}px 0;
    border-radius: {{metric.spacing_sm}}px;
    background: {{color.accent}};
}
QToolTip {
    color: {{color.text_primary}};
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    padding: {{metric.spacing_sm}}px;
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
        self._application.setStyleSheet(render_qss(self._theme, scale=self._scale))
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
