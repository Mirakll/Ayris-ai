"""Windows per-user autostart backed by the actual HKCU Run value."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from ayris.core.config import ConfigManager

__all__ = ["command", "disable", "enable", "is_enabled", "reconcile"]

RUN_KEY: Final = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE_NAME: Final = "Ayris"


def command() -> str:
    """Exact command Windows should run for this build or source checkout."""
    if getattr(sys, "frozen", False):
        arguments = [str(Path(sys.executable).resolve()), "--minimized"]
    else:
        arguments = [str(Path(sys.executable).resolve()), "-m", "ayris", "--minimized"]
    return subprocess.list2cmdline(arguments)


def _read() -> str | None:
    if sys.platform != "win32":
        return None
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, VALUE_NAME)
    except FileNotFoundError:
        return None
    return str(value)


def is_enabled() -> bool:
    """Whether the registry contains the current, non-stale command."""
    return _read() == command()


def enable() -> None:
    """Create or repair the per-user Run value."""
    if sys.platform != "win32":
        return
    import winreg

    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, VALUE_NAME, 0, winreg.REG_SZ, command())


def disable() -> None:
    """Remove the per-user Run value if present."""
    if sys.platform != "win32":
        return
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, VALUE_NAME)
    except FileNotFoundError:
        return


def reconcile(manager: ConfigManager) -> bool:
    """Repair a stale enabled path, then make config reflect the real key."""
    configured = manager.settings.general.autostart
    raw = _read()
    if configured and raw is not None and raw != command():
        enable()
    actual = is_enabled()
    if configured != actual:
        manager.apply({"general.autostart": actual})
    return actual
