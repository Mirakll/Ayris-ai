"""Optional Interception hotkey source used by full-screen games."""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable
from typing import Any

from ayris.actions.input.keys import KEYS, MODIFIERS
from ayris.utils.hotkey_backends.winapi import HotkeyBackendUnavailable
from ayris.utils.hotkeys import Hotkey

__all__ = ["InterceptionBackend"]


class InterceptionBackend:
    """Translate Interception key strokes into manager press/release events.

    The package and kernel driver are deliberately imported only here.  Machines
    without the optional ``games`` extra therefore start normally and the manager
    can fall back to RegisterHotKey.
    """

    def __init__(self) -> None:
        try:
            module = importlib.import_module("interception")
        except (ImportError, OSError) as exc:
            raise HotkeyBackendUnavailable(
                f"interception is unavailable: {exc}",
                user_message=(
                    "Драйвер Interception не установлен или недоступен. "
                    "Использую обычные горячие клавиши Windows."
                ),
            ) from exc
        context_class = getattr(module, "Interception", None)
        key_stroke = getattr(module, "KeyStroke", None)
        if not callable(context_class) or key_stroke is None:
            raise HotkeyBackendUnavailable(
                "interception package has no input context API",
                user_message="Драйвер Interception недоступен. Использую WinAPI.",
            )
        try:
            context = context_class()
            if not bool(getattr(context, "valid", False)):
                raise OSError("driver returned no devices")
        except Exception as exc:
            raise HotkeyBackendUnavailable(
                f"interception driver failed: {exc}",
                user_message="Драйвер Interception не запущен. Использую WinAPI.",
            ) from exc
        self._module: Any = module
        self._context: Any = context
        self._key_stroke: Any = key_stroke
        self._stop = threading.Event()
        self._bindings: dict[int, Hotkey] = {}
        self._callback: Callable[[int, bool], None] | None = None
        self._pressed: set[str] = set()
        self._scan_codes = _scan_codes()

    def set_bindings(self, bindings: dict[int, Hotkey]) -> None:
        self._bindings = dict(bindings)

    def run(self, callback: Callable[[int, bool], None]) -> None:
        self._callback = callback
        constants = importlib.import_module("interception.constants")
        filter_key = constants.FilterKeyFlag
        key_flag = constants.KeyFlag
        self._context.set_filter(
            self._context.is_keyboard,
            int(filter_key.FILTER_KEY_DOWN | filter_key.FILTER_KEY_UP),
        )
        try:
            while not self._stop.is_set():
                device = self._context.await_input(50)
                if device is None:
                    continue
                stroke = self._context.devices[device].receive()
                if stroke is None:
                    continue
                if isinstance(stroke, self._key_stroke):
                    pressed = not bool(int(stroke.flags) & int(key_flag.KEY_UP))
                    self._on_stroke(int(stroke.code), pressed)
                # Never swallow input: the same key must still reach the game.
                self._context.send(device, stroke)
        finally:
            self._context.destroy()

    def _on_stroke(self, scan_code: int, pressed: bool) -> None:
        name = self._scan_codes.get(scan_code, "")
        if not name:
            return
        modifier = _modifier_name(name)
        if pressed:
            self._pressed.add(name)
        else:
            self._pressed.discard(name)
        modifiers = frozenset(found for key in self._pressed if (found := _modifier_name(key)))
        for identifier, hotkey in tuple(self._bindings.items()):
            if modifier or name != hotkey.key:
                continue
            if frozenset(hotkey.modifiers) == modifiers and self._callback is not None:
                self._callback(identifier, pressed)

    def stop(self) -> None:
        self._stop.set()


def _scan_codes() -> dict[int, str]:
    keycodes = importlib.import_module("interception._keycodes")
    result: dict[int, str] = {}
    for name in KEYS:
        try:
            data = keycodes.get_key_information(name)
        except Exception:
            continue
        if int(data.scan_code) >= 0:
            result.setdefault(int(data.scan_code), name)
    return result


def _modifier_name(name: str) -> str:
    if name not in MODIFIERS:
        return ""
    for modifier in ("ctrl", "alt", "shift", "win"):
        if modifier in name or (modifier == "win" and name in {"lwin", "rwin"}):
            return modifier
    return ""
