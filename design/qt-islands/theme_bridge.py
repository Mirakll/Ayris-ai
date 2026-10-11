"""Мост «дизайн-токены → QML».

Единый источник токенов у приложения — `resources/themes/*.json`
(схема `ayris.gui.theme.tokens.Theme`). Здесь мы отдаём те же самые токены в
QML как Qt-свойства одного контекстного объекта `theme`, чтобы QML-острова
биндились на них нативно:

    color: theme.accent
    Behavior on color { ColorAnimation { duration: theme.animNormal } }

Смена темы = `set_theme(...)` → сигнал `changed` → QML перепривязывает всё сам.
Это и есть то, ради чего в вебе тянутся к «CSS-переменным», только без эмуляции.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Property, QObject, Signal
from PySide6.QtGui import QColor

# Тот же загрузчик/схема, что и у приложения — токены не расходятся с продом.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from ayris.gui.theme.tokens import Theme, load_theme  # noqa: E402

_THEMES_DIR = _REPO / "resources" / "themes"


def available_themes() -> list[str]:
    return sorted(p.stem for p in _THEMES_DIR.glob("*.json"))


class ThemeBridge(QObject):
    """QObject с токенами темы в виде свойств, читаемых из QML."""

    changed = Signal()

    def __init__(self, theme_name: str = "dark_purple") -> None:
        super().__init__()
        self._theme: Theme = load_theme(_THEMES_DIR / f"{theme_name}.json")
        self._name = theme_name

    # --- управление ---------------------------------------------------
    def set_theme(self, name: str) -> None:
        self._theme = load_theme(_THEMES_DIR / f"{name}.json")
        self._name = name
        self.changed.emit()  # одного сигнала хватает: все свойства notify на него

    def _c(self, key: str) -> QColor:
        return QColor(self._theme.color(key))

    # --- цвета (QColor, чтобы QML принимал как color) ------------------
    name = Property(str, lambda s: s._name, notify=changed)
    mode = Property(str, lambda s: s._theme.mode, notify=changed)
    background = Property(QColor, lambda s: s._c("background"), notify=changed)
    surface = Property(QColor, lambda s: s._c("surface"), notify=changed)
    surfaceHi = Property(QColor, lambda s: s._c("surface_highlight"), notify=changed)
    border = Property(QColor, lambda s: s._c("border"), notify=changed)
    accent = Property(QColor, lambda s: s._c("accent"), notify=changed)
    accentHover = Property(QColor, lambda s: s._c("accent_hover"), notify=changed)
    accentPressed = Property(QColor, lambda s: s._c("accent_pressed"), notify=changed)
    textPrimary = Property(QColor, lambda s: s._c("text_primary"), notify=changed)
    textSecondary = Property(QColor, lambda s: s._c("text_secondary"), notify=changed)
    textMuted = Property(QColor, lambda s: s._c("text_muted"), notify=changed)
    success = Property(QColor, lambda s: s._c("success"), notify=changed)
    warning = Property(QColor, lambda s: s._c("warning"), notify=changed)
    error = Property(QColor, lambda s: s._c("error"), notify=changed)
    info = Property(QColor, lambda s: s._c("info"), notify=changed)
    roleAction = Property(QColor, lambda s: s._c("role_action"), notify=changed)
    roleSound = Property(QColor, lambda s: s._c("role_sound"), notify=changed)
    onAccent = Property(QColor, lambda s: s._c("on_accent"), notify=changed)

    # --- метрики ------------------------------------------------------
    radiusMd = Property(int, lambda s: s._theme.metric("radius_md"), notify=changed)
    radiusLg = Property(int, lambda s: s._theme.metric("radius_lg"), notify=changed)
    radiusXl = Property(int, lambda s: s._theme.metric("radius_xl"), notify=changed)
    animFast = Property(int, lambda s: s._theme.metric("animation_fast"), notify=changed)
    animNormal = Property(int, lambda s: s._theme.metric("animation_normal"), notify=changed)
    animSlow = Property(int, lambda s: s._theme.metric("animation_slow"), notify=changed)
