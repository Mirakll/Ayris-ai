"""VoiceAttack action names and fields translated to Ayris blocks."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from ayris.actions.macros.schema import ActionBlock
from ayris.utils.hotkeys import HotkeyNotationError, canonical_hotkey


def _clean_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _value(fields: Mapping[str, str], *names: str, default: str = "") -> str:
    lowered = {_clean_tag(key): value.strip() for key, value in fields.items()}
    for name in names:
        value = lowered.get(name.casefold())
        if value:
            return value
    return default


def _number(fields: Mapping[str, str], *names: str, default: int = 0) -> int:
    raw = _value(fields, *names)
    try:
        return int(float(raw.replace(",", ".")))
    except ValueError:
        return default


def _keys(fields: Mapping[str, str], block_type: str = "KeyPress") -> ActionBlock:
    raw = _value(fields, "keys", "key", "keycodes", "keycode", "value")
    try:
        combo = canonical_hotkey(raw)
    except HotkeyNotationError:
        combo = raw.casefold().replace(" ", "+")
    return ActionBlock(type=block_type, params={"combo": combo})


def _text(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(type="TypeText", params={"text": _value(fields, "text", "value")})


def _run(fields: Mapping[str, str]) -> ActionBlock:
    target = _value(fields, "path", "application", "target", "value", "command")
    args = _value(fields, "parameters", "arguments", "args")
    if re.search(r"(?:^|[\\/])(cmd|powershell|pwsh)(?:\.exe)?$", target, re.I):
        command = " ".join(part for part in (target, args) if part)
        return ActionBlock(type="RunShell", params={"command": command}, enabled=False)
    return ActionBlock(
        type="RunApp",
        params={"app": target, "arguments": args},
    )


def _pause(fields: Mapping[str, str]) -> ActionBlock:
    milliseconds = _number(fields, "milliseconds", "durationms", "value")
    if not milliseconds:
        milliseconds = int(_number(fields, "seconds", "duration") * 1000)
    return ActionBlock(type="Wait", params={"ms": milliseconds})


def _say(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(type="Say", params={"text": _value(fields, "text", "value")})


def _sound(fields: Mapping[str, str]) -> ActionBlock:
    path = _value(fields, "file", "path", "sound", "value")
    return ActionBlock(type="PlaySound", params={"sound": path})


def _set_var(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="SetVar",
        params={"name": _value(fields, "name", "variable"), "value": _value(fields, "value")},
    )


def _condition(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="If",
        params={"condition": _value(fields, "condition", "expression", "value") or "false"},
        then=[ActionBlock(type="OverlayLog", params={"message": "Условие импортировано без тела"})],
    )


def _call(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="CallCommand", params={"command": _value(fields, "command", "name", "value")}
    )


def _mouse_click(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="MouseClick",
        params={
            "button": _value(fields, "button", default="left").casefold(),
            "clicks": max(1, _number(fields, "clicks", "count", default=1)),
        },
    )


def _mouse_move(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="MouseMove",
        params={"x": _number(fields, "x"), "y": _number(fields, "y")},
    )


def _mouse_wheel(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(type="MouseWheel", params={"clicks": _number(fields, "clicks", "value")})


def _window(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(type="FocusWindow", params={"title": _value(fields, "title", "value")})


def _window_state(fields: Mapping[str, str]) -> ActionBlock:
    return ActionBlock(
        type="WindowState",
        params={
            "title": _value(fields, "title"),
            "command": _value(fields, "state", "command", "value", default="minimize").casefold(),
        },
    )


@dataclass(frozen=True, slots=True)
class VapActionMapping:
    block: str
    convert: Callable[[Mapping[str, str]], ActionBlock]


VAP_ACTIONS: Final[dict[str, VapActionMapping]] = {
    "keypress": VapActionMapping("KeyPress", _keys),
    "presskey": VapActionMapping("KeyPress", _keys),
    "keydown": VapActionMapping("KeyDown", lambda fields: _keys(fields, "KeyDown")),
    "keyup": VapActionMapping("KeyUp", lambda fields: _keys(fields, "KeyUp")),
    "type": VapActionMapping("TypeText", _text),
    "typetext": VapActionMapping("TypeText", _text),
    "write": VapActionMapping("TypeText", _text),
    "mouseclick": VapActionMapping("MouseClick", _mouse_click),
    "mousemove": VapActionMapping("MouseMove", _mouse_move),
    "mousewheel": VapActionMapping("MouseWheel", _mouse_wheel),
    "runapplication": VapActionMapping("RunApp", _run),
    "runprogram": VapActionMapping("RunApp", _run),
    "execute": VapActionMapping("RunApp", _run),
    "activatewindow": VapActionMapping("FocusWindow", _window),
    "focuswindow": VapActionMapping("FocusWindow", _window),
    "windowstate": VapActionMapping("WindowState", _window_state),
    "pause": VapActionMapping("Wait", _pause),
    "wait": VapActionMapping("Wait", _pause),
    "say": VapActionMapping("Say", _say),
    "texttospeech": VapActionMapping("Say", _say),
    "playsound": VapActionMapping("PlaySound", _sound),
    "setvariable": VapActionMapping("SetVar", _set_var),
    "settext": VapActionMapping("SetVar", _set_var),
    "condition": VapActionMapping("If", _condition),
    "if": VapActionMapping("If", _condition),
    "executecommand": VapActionMapping("CallCommand", _call),
    "callcommand": VapActionMapping("CallCommand", _call),
}


def normalize_action_name(name: str) -> str:
    """Discard punctuation and common VoiceAttack type suffixes."""
    cleaned = re.sub(r"[^a-z0-9]", "", name.casefold())
    for suffix in ("action", "commandaction", "event"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    return cleaned


def map_vap_action(kind: str, fields: Mapping[str, str]) -> ActionBlock | None:
    mapping = VAP_ACTIONS.get(normalize_action_name(kind))
    return mapping.convert(fields) if mapping is not None else None
