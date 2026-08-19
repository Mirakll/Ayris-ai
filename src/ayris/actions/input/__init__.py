"""Keyboard and mouse: everything Ayris presses, types, clicks and drags.

Three modules, split by what each has to know:

* :mod:`~ayris.actions.input.backend` — the seam. One interface, three
  implementations (``SendInput``, the ``interception`` driver, and a recorder that
  writes events down instead of sending them), and no knowledge of displays or key
  names.
* :mod:`~ayris.actions.input.keys` — the name table, the combination parser, and
  the four keyboard blocks. Also the registry of keys a ``KeyDown`` left held, so
  that a stopped macro cannot leave Ctrl down on a real desktop.
* :mod:`~ayris.actions.input.mouse` — the coordinate arithmetic (virtual desktop,
  multiple monitors, per-monitor DPI) and the four pointer blocks.

Importing this package registers all eight actions, which is what
:meth:`ActionRegistry.discover <ayris.actions.registry.ActionRegistry.discover>`
walks the package for.
"""

from __future__ import annotations

from ayris.actions.input.backend import (
    BackendKind,
    InputBackend,
    InputBlocked,
    InputDriverMissing,
    InputEvent,
    InterceptionBackend,
    KeyStroke,
    MouseButton,
    RecordingBackend,
    SendInputBackend,
    get_input_backend,
    reset_input_backend,
    set_input_backend,
    set_input_bus,
)
from ayris.actions.input.keys import (
    ALIASES,
    KEYS,
    MODIFIERS,
    KeyDown,
    KeyPress,
    KeyUp,
    TypeMode,
    TypeText,
    UnknownKey,
    held_keys,
    parse_combo,
    release_held_keys,
    resolve_key,
)
from ayris.actions.input.mouse import (
    MouseClick,
    MouseDrag,
    MouseMove,
    MousePoint,
    MouseWheel,
    Origin,
    ScreenBackend,
    ScreenLayout,
    WinApiScreen,
    drag_path,
    get_screen_backend,
    normalize_point,
    set_screen_backend,
)

__all__ = [
    "ALIASES",
    "KEYS",
    "MODIFIERS",
    "BackendKind",
    "InputBackend",
    "InputBlocked",
    "InputDriverMissing",
    "InputEvent",
    "InterceptionBackend",
    "KeyDown",
    "KeyPress",
    "KeyStroke",
    "KeyUp",
    "MouseButton",
    "MouseClick",
    "MouseDrag",
    "MouseMove",
    "MousePoint",
    "MouseWheel",
    "Origin",
    "RecordingBackend",
    "ScreenBackend",
    "ScreenLayout",
    "SendInputBackend",
    "TypeMode",
    "TypeText",
    "UnknownKey",
    "WinApiScreen",
    "drag_path",
    "get_input_backend",
    "get_screen_backend",
    "held_keys",
    "normalize_point",
    "parse_combo",
    "release_held_keys",
    "reset_input_backend",
    "resolve_key",
    "set_input_backend",
    "set_input_bus",
    "set_screen_backend",
]
