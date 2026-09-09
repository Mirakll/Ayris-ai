"""Windows colour-scheme detection and runtime change polling."""

from __future__ import annotations

import sys
from typing import Literal

from PySide6.QtCore import QObject, QTimer, Signal

__all__ = ["SystemThemeWatcher", "detect_system_theme"]

SystemTheme = Literal["dark", "light"]
_KEY = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"


def detect_system_theme() -> SystemTheme:
    """Return the Windows app colour scheme; dark is the safe fallback."""
    if sys.platform != "win32":
        return "dark"
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "AppsUseLightTheme")
        return "light" if int(value) else "dark"
    except (OSError, TypeError, ValueError):
        return "dark"


class SystemThemeWatcher(QObject):
    """Poll the tiny registry value without blocking the GUI thread."""

    changed = Signal(str)

    def __init__(self, *, interval_ms: int = 1000, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current = detect_system_theme()
        self._timer = QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self.check_now)

    @property
    def current(self) -> SystemTheme:
        return self._current

    def start(self) -> None:
        self.check_now()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def check_now(self) -> None:
        detected = detect_system_theme()
        if detected == self._current:
            return
        self._current = detected
        self.changed.emit(detected)
