"""Windows HKCU autostart uses and repairs the command actually installed."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ayris.core.config import ConfigManager
from ayris.utils import autostart

pytestmark = [pytest.mark.unit, pytest.mark.skipif(sys.platform != "win32", reason="HKCU Run")]


def test_enable_disable_and_stale_path_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(autostart, "VALUE_NAME", f"AyrisTest-{tmp_path.name}")
    manager = ConfigManager(tmp_path / "config.toml")
    manager.load()
    try:
        autostart.disable()
        assert not autostart.is_enabled()
        autostart.enable()
        assert autostart.is_enabled()
        manager.apply({"general.autostart": True})

        import winreg

        with winreg.CreateKeyEx(
            winreg.HKEY_CURRENT_USER, autostart.RUN_KEY, 0, winreg.KEY_SET_VALUE
        ) as key:
            winreg.SetValueEx(key, autostart.VALUE_NAME, 0, winreg.REG_SZ, "old.exe --minimized")
        assert not autostart.is_enabled()
        assert autostart.reconcile(manager) is True
        assert autostart.is_enabled()
    finally:
        autostart.disable()


def test_source_command_is_quoted_and_minimized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    value = autostart.command()
    assert " -m ayris --minimized" in value
    assert str(Path(sys.executable).resolve()) in value
