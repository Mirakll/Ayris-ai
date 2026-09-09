"""Backends used by the global hotkey manager."""

from ayris.utils.hotkey_backends.interception import InterceptionBackend
from ayris.utils.hotkey_backends.winapi import WinApiBackend

__all__ = ["InterceptionBackend", "WinApiBackend"]
