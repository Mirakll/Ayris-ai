"""Key names, combinations, and the four keyboard blocks.

Three separate problems live here, and the middle one is the reason the other two
are in the same module.

**Names.** A macro says ``ctrl+shift+f5``, not ``0x11, 0x10, 0x74``. :data:`KEYS`
is that dictionary, and it carries one flag beyond the virtual key: ``extended``,
for the keys whose scan code needs the ``E0`` prefix. Get that wrong and Home
arrives as Numpad-7 in anything reading scan codes — which is exactly the class of
application the scan-code mode exists for.

**Aliases, including Russian ones.** The user dictates «контрол шифт эф пять» and
the recogniser hands over something spelled the way it was heard. :data:`ALIASES`
absorbs the common spellings — ``ктрл``, ``шифт``, ``альт``, ``вин``, ``ввод``,
``пробел``, ``вверх`` — so that a macro written by voice and one typed by hand mean
the same thing. This is a synonym table, not a transliterator: an unknown name is
an error with the list of what was expected, because silently pressing the wrong key
is worse than refusing.

**Pressing, as three blocks instead of one.** :class:`KeyPress` is down-and-up.
:class:`KeyDown` and :class:`KeyUp` are the halves, and they exist because a macro
needs to hold Shift across several other blocks — select, move, release. The cost of
that is a key that stays down after the macro stops, which on a real keyboard would
mean a stuck modifier and an unusable desktop. So every key a :class:`KeyDown`
leaves down is written into a module-level registry, and
:func:`release_held_keys` — called by the macro runner when a run ends, is
cancelled, or fails — puts them all back up in reverse order. :class:`KeyPress`
does not rely on that: it releases in a ``finally``, so an exception mid-combo
still frees Ctrl.
"""

from __future__ import annotations

import time
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.input.backend import InputBackend, KeyStroke, get_input_backend
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "ALIASES",
    "KEYS",
    "MODIFIERS",
    "KeyDown",
    "KeyPress",
    "KeyUp",
    "TypeMode",
    "TypeText",
    "UnknownKey",
    "char_key",
    "char_shift_state",
    "held_keys",
    "parse_combo",
    "press_combo",
    "release_held_keys",
    "resolve_key",
]

_log = get_logger(__name__)


class UnknownKey(ActionError):
    """A combination named a key the table does not have."""

    default_user_message = "Не понимаю, какую клавишу нажать."


def _stroke(name: str, vk: int, *, extended: bool = False) -> KeyStroke:
    return KeyStroke(name=name, vk=vk, extended=extended)


def _build_keys() -> dict[str, KeyStroke]:
    """The name table.

    Virtual keys are written as literals rather than pulled from
    :mod:`ayris.utils.winapi`: the module keeps the handful of codes its own
    functions need, and duplicating three hundred ``VK_`` constants there to be
    read once each would put the table one indirection away from the names it
    describes. The values are from ``winuser.h`` and do not change.
    """
    keys: dict[str, KeyStroke] = {}

    # Modifiers. The bare names are the "either side" virtual keys, which is what
    # a shortcut means; the sided ones matter when an application distinguishes
    # them (AltGr is right Alt, some games bind left Shift only).
    keys["ctrl"] = _stroke("ctrl", 0x11)
    keys["shift"] = _stroke("shift", 0x10)
    keys["alt"] = _stroke("alt", 0x12)
    keys["win"] = _stroke("win", 0x5B, extended=True)
    keys["lctrl"] = _stroke("lctrl", 0xA2)
    keys["rctrl"] = _stroke("rctrl", 0xA3, extended=True)
    keys["lshift"] = _stroke("lshift", 0xA0)
    keys["rshift"] = _stroke("rshift", 0xA1)
    keys["lalt"] = _stroke("lalt", 0xA4)
    keys["ralt"] = _stroke("ralt", 0xA5, extended=True)
    keys["lwin"] = _stroke("lwin", 0x5B, extended=True)
    keys["rwin"] = _stroke("rwin", 0x5C, extended=True)

    # Typing and editing.
    keys["backspace"] = _stroke("backspace", 0x08)
    keys["tab"] = _stroke("tab", 0x09)
    keys["enter"] = _stroke("enter", 0x0D)
    keys["escape"] = _stroke("escape", 0x1B)
    keys["space"] = _stroke("space", 0x20)
    keys["capslock"] = _stroke("capslock", 0x14)
    keys["apps"] = _stroke("apps", 0x5D, extended=True)

    # Navigation. Every one of these is extended.
    for name, vk in (
        ("pageup", 0x21),
        ("pagedown", 0x22),
        ("end", 0x23),
        ("home", 0x24),
        ("left", 0x25),
        ("up", 0x26),
        ("right", 0x27),
        ("down", 0x28),
        ("insert", 0x2D),
        ("delete", 0x2E),
    ):
        keys[name] = _stroke(name, vk, extended=True)

    # System keys. Print Screen and Num Lock carry the E0 prefix; Pause does not,
    # it is the one key with an E1 prefix and Windows reports its scan code whole.
    keys["printscreen"] = _stroke("printscreen", 0x2C, extended=True)
    keys["numlock"] = _stroke("numlock", 0x90, extended=True)
    keys["scrolllock"] = _stroke("scrolllock", 0x91)
    keys["pause"] = _stroke("pause", 0x13)

    # Digits and letters. ``vk`` equals the ASCII code of the uppercase character,
    # which is why the row of digits needs no table of its own.
    for digit in range(10):
        keys[str(digit)] = _stroke(str(digit), 0x30 + digit)
    for index in range(26):
        letter = chr(ord("a") + index)
        keys[letter] = _stroke(letter, 0x41 + index)

    # Function keys, all twenty-four of them.
    for number in range(1, 25):
        name = f"f{number}"
        keys[name] = _stroke(name, 0x6F + number)

    # Numpad. ``num*`` names are deliberate: a macro that wants the numeric keypad
    # must be able to say so, because a game bound to Numpad-4 does not accept 4.
    for digit in range(10):
        name = f"num{digit}"
        keys[name] = _stroke(name, 0x60 + digit)
    keys["nummultiply"] = _stroke("nummultiply", 0x6A)
    keys["numadd"] = _stroke("numadd", 0x6B)
    keys["numsubtract"] = _stroke("numsubtract", 0x6D)
    keys["numdecimal"] = _stroke("numdecimal", 0x6E)
    keys["numdivide"] = _stroke("numdivide", 0x6F, extended=True)
    keys["numenter"] = _stroke("numenter", 0x0D, extended=True)

    # Punctuation, as the OEM keys. These are *positions* on a US keyboard: on a
    # Russian layout the key named ``semicolon`` types «ж». Which is correct for a
    # shortcut — Ctrl+«;» is one physical key regardless of layout — and is why
    # text is typed by :class:`TypeText` and not by naming keys.
    for name, vk in (
        ("semicolon", 0xBA),
        ("equal", 0xBB),
        ("comma", 0xBC),
        ("minus", 0xBD),
        ("period", 0xBE),
        ("slash", 0xBF),
        ("backquote", 0xC0),
        ("bracketleft", 0xDB),
        ("backslash", 0xDC),
        ("bracketright", 0xDD),
        ("quote", 0xDE),
    ):
        keys[name] = _stroke(name, vk)

    # Media and browser keys. These have no scan code at all, so the scan-code
    # mode cannot express them; the virtual key is sent instead.
    for name, vk in (
        ("volumemute", 0xAD),
        ("volumedown", 0xAE),
        ("volumeup", 0xAF),
        ("medianext", 0xB0),
        ("mediaprev", 0xB1),
        ("mediastop", 0xB2),
        ("mediaplay", 0xB3),
        ("browserback", 0xA6),
        ("browserforward", 0xA7),
        ("browserrefresh", 0xA8),
        ("browserstop", 0xA9),
        ("browsersearch", 0xAA),
        ("browserhome", 0xAC),
    ):
        keys[name] = _stroke(name, vk, extended=True)

    return keys


#: Every key Ayris can name, by its canonical name.
KEYS: Final[dict[str, KeyStroke]] = _build_keys()

#: Names that mean a modifier. Pressed first and released last in a combination.
MODIFIERS: Final[frozenset[str]] = frozenset(
    {
        "ctrl",
        "shift",
        "alt",
        "win",
        "lctrl",
        "rctrl",
        "lshift",
        "rshift",
        "lalt",
        "ralt",
        "lwin",
        "rwin",
    }
)

#: Alternative spellings, Russian and otherwise, mapped onto canonical names.
#:
#: The Russian half is what dictation produces; the Latin half is what people who
#: have used other macro tools type from muscle memory.
ALIASES: Final[dict[str, str]] = {
    # Modifiers.
    "control": "ctrl",
    "ctl": "ctrl",
    "контрол": "ctrl",
    "контрл": "ctrl",
    "ктрл": "ctrl",
    "шифт": "shift",
    "сдвиг": "shift",
    "альт": "alt",
    "меню": "alt",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "вин": "win",
    "винда": "win",
    "виндовс": "win",
    "пуск": "win",
    # Typing and editing.
    "return": "enter",
    "ввод": "enter",
    "энтер": "enter",
    "esc": "escape",
    "эск": "escape",
    "эскейп": "escape",
    "пробел": "space",
    "таб": "tab",
    "табуляция": "tab",
    "бэкспейс": "backspace",
    "забой": "backspace",
    "del": "delete",
    "удалить": "delete",
    "ins": "insert",
    "вставка": "insert",
    "капс": "capslock",
    "контекстное": "apps",
    # Navigation.
    "pgup": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "вверх": "up",
    "вниз": "down",
    "влево": "left",
    "вправо": "right",
    "домой": "home",
    "начало": "home",
    "конец": "end",
    "страницавверх": "pageup",
    "страницавниз": "pagedown",
    # Punctuation, by the character it sits on.
    ";": "semicolon",
    "=": "equal",
    "+": "equal",
    "plus": "equal",
    "плюс": "equal",
    ",": "comma",
    "-": "minus",
    "_": "minus",
    "минус": "minus",
    ".": "period",
    "/": "slash",
    "`": "backquote",
    "~": "backquote",
    "[": "bracketleft",
    "\\": "backslash",
    "]": "bracketright",
    "'": "quote",
    '"': "quote",
    # Media.
    "prtsc": "printscreen",
    "printscr": "printscreen",
    "play": "mediaplay",
    "pause_media": "mediaplay",
    "mute": "volumemute",
}


def resolve_key(name: str) -> KeyStroke:
    """One key by name, alias, or single character.

    A bare character is resolved through the table as well, so ``"a"`` and
    ``"7"`` work and ``"й"`` does not — a letter that is not a key *position* has
    no place in a shortcut, and text belongs in :class:`TypeText`.

    Raises:
        UnknownKey: no key goes by that name.
    """
    cleaned = name.strip().lower().replace("_", "") if len(name.strip()) > 1 else name.strip()
    if not cleaned:
        raise UnknownKey("empty key name")
    canonical = ALIASES.get(cleaned, cleaned)
    key = KEYS.get(canonical)
    if key is None:
        raise UnknownKey(
            f"unknown key {name!r}",
            user_message=f"Клавиша «{name.strip()}» мне неизвестна.",
        )
    return key


def _split_combo(text: str) -> tuple[list[str], bool]:
    """Split on ``+`` while letting ``+`` itself be a key.

    ``ctrl++`` is Ctrl and the plus key: the second separator has nothing before
    it, so it is the key. ``ctrl+`` is a mistake, and the second element of the
    return says so rather than quietly yielding just Ctrl.
    """
    parts: list[str] = []
    current = ""
    after_separator = False
    for char in text:
        if char != "+":
            current += char
            if not char.isspace():
                after_separator = False
            continue
        if current.strip():
            parts.append(current.strip())
            current = ""
            after_separator = True
        elif after_separator or not parts:
            current = "+"
            after_separator = False
        else:
            after_separator = True
    if current.strip():
        parts.append(current.strip())
        return parts, False
    return parts, after_separator


def parse_combo(combo: str) -> tuple[KeyStroke, ...]:
    """``"ctrl+shift+f5"`` into the keys to press, modifiers first.

    Order is normalised rather than trusted: ``tab+alt`` is the same intent as
    ``alt+tab``, and pressing Tab before Alt produces a Tab character instead of
    a window switch. Duplicates collapse, so ``ctrl+ctrl+c`` is harmless.

    Raises:
        UnknownKey: the string is malformed or names a key that does not exist.
    """
    parts, dangling = _split_combo(combo)
    if dangling:
        raise UnknownKey(
            f"combo {combo!r} ends with a separator",
            user_message=f"В комбинации «{combo.strip()}» не хватает клавиши после «+».",
        )
    if not parts:
        raise UnknownKey(
            f"combo {combo!r} names no keys",
            user_message="Комбинация клавиш не указана.",
        )
    modifiers: list[KeyStroke] = []
    rest: list[KeyStroke] = []
    seen: set[str] = set()
    for part in parts:
        key = resolve_key(part)
        if key.name in seen:
            continue
        seen.add(key.name)
        (modifiers if key.name in MODIFIERS else rest).append(key)
    return tuple(modifiers) + tuple(rest)


def char_key(char: str) -> KeyStroke | None:
    """The key that types ``char`` on the *current* layout, or ``None``.

    Used only by the scan-code typing mode. ``VkKeyScanW`` answers with the
    virtual key in the low byte and the required modifier state in the high one;
    ``None`` means the active layout cannot produce this character at all, which
    is the normal answer for Cyrillic under a US layout and the reason the Unicode
    mode is the default.
    """
    if len(char) != 1:
        return None
    try:
        packed = winapi.vk_key_scan(char)
    except winapi.WinApiError:
        return None
    if packed < 0:
        return None
    vk = packed & 0xFF
    try:
        scan = winapi.map_virtual_key(vk)
    except winapi.WinApiError:
        scan = 0
    if not scan:
        return None
    return KeyStroke(name=char, vk=vk, scan=scan)


def char_shift_state(char: str) -> tuple[bool, bool, bool]:
    """Which modifiers the layout wants held to type ``char``: shift, ctrl, alt."""
    if len(char) != 1:
        return (False, False, False)
    try:
        packed = winapi.vk_key_scan(char)
    except winapi.WinApiError:
        return (False, False, False)
    if packed < 0:
        return (False, False, False)
    state = (packed >> 8) & 0xFF
    return (bool(state & 1), bool(state & 2), bool(state & 4))


# --------------------------------------------------------------------------- #
# Keys currently held down by a KeyDown block.
# --------------------------------------------------------------------------- #

#: Insertion-ordered, so release happens in the reverse of press order.
_held: dict[str, tuple[KeyStroke, bool]] = {}


def held_keys() -> tuple[str, ...]:
    """Names of the keys a :class:`KeyDown` left down, in the order pressed."""
    return tuple(_held)


def release_held_keys(*, backend: InputBackend | None = None) -> tuple[str, ...]:
    """Put every held key back up. Returns what was released.

    Called when a macro finishes, is stopped, or raises. Errors on individual
    keys are logged and skipped rather than raised: this is cleanup, and a
    failure to release Shift must not prevent the attempt to release Ctrl.
    """
    if not _held:
        return ()
    sender = backend or get_input_backend()
    released: list[str] = []
    for name, (key, scancode) in reversed(list(_held.items())):
        try:
            sender.key_up(key, scancode=scancode)
        except Exception:
            _log.warning("не удалось отпустить клавишу %s", name, exc_info=True)
        else:
            released.append(name)
    _held.clear()
    return tuple(released)


def _remember_held(key: KeyStroke, *, scancode: bool) -> None:
    _held[key.name] = (key, scancode)


def _forget_held(key: KeyStroke) -> None:
    _held.pop(key.name, None)


def _pause(ms: int) -> None:
    """Wait between events. Zero means no wait, and the config forbids it as a default."""
    if ms > 0:
        time.sleep(ms / 1000.0)


def _timings() -> tuple[int, int, int]:
    """``(key_delay_ms, key_hold_ms, char_delay_ms)`` from the config, read fresh.

    Read on every call rather than cached, so a change in the settings applies to
    the next block instead of the next launch.
    """
    from ayris.core.config import get_settings

    section = get_settings().actions.input
    return (section.key_delay_ms, section.key_hold_ms, section.char_delay_ms)


def press_combo(
    keys: Sequence[KeyStroke],
    *,
    backend: InputBackend,
    hold_ms: int,
    scancode: bool = False,
) -> None:
    """Down in order, hold, up in reverse — with the release guaranteed.

    The ``finally`` is the whole point. If the third key of a four-key combination
    fails, the two already down have to come back up, or the user is left holding
    Ctrl+Shift with no keyboard of their own to release it.
    """
    pressed: list[KeyStroke] = []
    try:
        for key in keys:
            backend.key_down(key, scancode=scancode)
            pressed.append(key)
        _pause(hold_ms)
    finally:
        for key in reversed(pressed):
            try:
                backend.key_up(key, scancode=scancode)
            except Exception:
                _log.warning("не удалось отпустить клавишу %s", key.name, exc_info=True)


def _combo_title(keys: Iterable[KeyStroke]) -> str:
    return "+".join(key.name for key in keys)


@register
class KeyPress(Action):
    """Press a combination and release it."""

    meta: ClassVar = ActionMeta(
        name="KeyPress",
        category=ActionCategory.INPUT,
        title_ru="Нажать клавиши",
        description_ru="Нажимает комбинацию, например ctrl+shift+f5, и отпускает её.",
    )

    class Params(ActionParams):
        combo: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Комбинация клавиш: ctrl+c, alt+tab, win+left",
        )
        times: int = Field(
            default=1,
            ge=1,
            le=100,
            description="Сколько раз нажать",
        )
        hold_ms: int | None = Field(
            default=None,
            ge=0,
            le=5000,
            description="Сколько держать нажатой; пусто — из настроек",
            json_schema_extra={"unit_ru": "мс"},
        )
        delay_ms: int | None = Field(
            default=None,
            ge=0,
            le=2000,
            description="Пауза между повторами; пусто — из настроек",
            json_schema_extra={"unit_ru": "мс"},
        )
        scancode: bool = Field(
            default=False,
            description="Отправлять скан-коды — нужно играм, которые читают DirectInput",
        )

    def run(self, params: Params) -> ActionResult[str]:
        keys = parse_combo(params.combo)
        delay_default, hold_default, _ = _timings()
        hold = hold_default if params.hold_ms is None else params.hold_ms
        delay = delay_default if params.delay_ms is None else params.delay_ms
        backend = get_input_backend()
        title = _combo_title(keys)
        for index in range(params.times):
            if index:
                _pause(delay)
            press_combo(keys, backend=backend, hold_ms=hold, scancode=params.scancode)
        suffix = f" ×{params.times}" if params.times > 1 else ""
        return ActionResult.done(
            f"Нажал {title}{suffix}.",
            value=title,
            data={"keys": [key.name for key in keys], "times": params.times},
        )


@register
class KeyDown(Action):
    """Hold a key down until a later block releases it.

    Every key pressed here is remembered, and :func:`release_held_keys` is what
    the macro runner calls when the run ends by any route. Nothing is left down
    because a macro forgot its :class:`KeyUp`.
    """

    meta: ClassVar = ActionMeta(
        name="KeyDown",
        category=ActionCategory.INPUT,
        title_ru="Зажать клавиши",
        description_ru="Нажимает и удерживает клавиши до блока «Отпустить клавиши».",
    )

    class Params(ActionParams):
        combo: str = Field(
            ...,
            min_length=1,
            max_length=200,
            description="Что зажать: shift, ctrl+alt, w",
        )
        scancode: bool = Field(
            default=False,
            description="Отправлять скан-коды — нужно играм, которые читают DirectInput",
        )

    def run(self, params: Params) -> ActionResult[str]:
        keys = parse_combo(params.combo)
        backend = get_input_backend()
        for key in keys:
            backend.key_down(key, scancode=params.scancode)
            _remember_held(key, scancode=params.scancode)
        title = _combo_title(keys)
        return ActionResult.done(
            f"Держу {title}.",
            value=title,
            data={"held": list(held_keys())},
        )


@register
class KeyUp(Action):
    """Release keys held by :class:`KeyDown`, or everything at once."""

    meta: ClassVar = ActionMeta(
        name="KeyUp",
        category=ActionCategory.INPUT,
        title_ru="Отпустить клавиши",
        description_ru="Отпускает указанные клавиши или все зажатые, если ничего не указано.",
    )

    class Params(ActionParams):
        combo: str = Field(
            default="",
            max_length=200,
            description="Что отпустить; пусто — все зажатые клавиши",
        )
        scancode: bool = Field(
            default=False,
            description="Отправлять скан-коды — нужно играм, которые читают DirectInput",
        )

    def run(self, params: Params) -> ActionResult[str]:
        backend = get_input_backend()
        if not params.combo.strip():
            released = release_held_keys(backend=backend)
            if not released:
                return ActionResult.done("Зажатых клавиш не было.", value="")
            return ActionResult.done(
                f"Отпустил {'+'.join(released)}.",
                value="+".join(released),
                data={"released": list(released)},
            )
        keys = parse_combo(params.combo)
        for key in reversed(keys):
            backend.key_up(key, scancode=params.scancode)
            _forget_held(key)
        title = _combo_title(keys)
        return ActionResult.done(
            f"Отпустил {title}.",
            value=title,
            data={"held": list(held_keys())},
        )


#: Modes of :class:`TypeText`, with the labels the editor shows.
class TypeMode(StrEnum):
    """Which of the three typing routes :class:`TypeText` takes."""

    AUTO = "auto"
    UNICODE = "unicode"
    LAYOUT = "layout"
    CLIPBOARD = "clipboard"

    @property
    def title_ru(self) -> str:
        return _TYPE_MODE_TITLES[self]


_TYPE_MODE_TITLES: Final[dict[TypeMode, str]] = {
    TypeMode.AUTO: "автоматически",
    TypeMode.UNICODE: "как символы (кириллица, эмодзи)",
    TypeMode.LAYOUT: "через раскладку (для игр)",
    TypeMode.CLIPBOARD: "через буфер обмена",
}


@register
class TypeText(Action):
    """Type text, by whichever of three routes suits the receiver.

    ``unicode`` — the default — sends each character as its UTF-16 code units.
    The active layout is irrelevant, so Cyrillic arrives with a US layout in front
    and nothing has to be switched behind the user's back. Emoji work too, as a
    surrogate pair.

    ``layout`` sends scan codes through the current layout, for applications that
    read the keyboard through DirectInput and discard anything without a scan
    code — that is, games. It can only type what the active layout can produce;
    characters it cannot are sent as Unicode rather than dropped.

    ``clipboard`` puts the text on the clipboard and presses Ctrl+V. For a long
    passage this is the difference between instant and a visible half-minute of
    typing. The previous clipboard contents are put back afterwards.

    ``auto`` is ``clipboard`` above the configured length and ``unicode`` below
    it, which is what a person means by "just type this".
    """

    meta: ClassVar = ActionMeta(
        name="TypeText",
        category=ActionCategory.INPUT,
        title_ru="Напечатать текст",
        description_ru="Печатает текст: кириллица и эмодзи независимо от раскладки.",
    )

    class Params(ActionParams):
        text: str = Field(
            ...,
            max_length=100_000,
            description="Что напечатать",
            json_schema_extra={"multiline": True},
        )
        mode: TypeMode = Field(
            default=TypeMode.AUTO,
            description="Как печатать",
            json_schema_extra={"choices_ru": {str(m): m.title_ru for m in TypeMode}},
        )
        char_delay_ms: int | None = Field(
            default=None,
            ge=0,
            le=1000,
            description="Пауза между символами; пусто — из настроек",
            json_schema_extra={"unit_ru": "мс"},
        )

    def run(self, params: Params) -> ActionResult[int]:
        if not params.text:
            return ActionResult.done("Печатать нечего.", value=0)
        _, _, char_default = _timings()
        delay = char_default if params.char_delay_ms is None else params.char_delay_ms
        mode = self._resolve_mode(params.mode, params.text)
        backend = get_input_backend()
        if mode is TypeMode.CLIPBOARD:
            self._paste(params.text, backend=backend)
        elif mode is TypeMode.LAYOUT:
            self._type_through_layout(params.text, backend=backend, delay=delay)
        else:
            self._type_unicode(params.text, backend=backend, delay=delay)
        return ActionResult.done(
            f"Напечатал {len(params.text)} символов.",
            value=len(params.text),
            data={"mode": str(mode), "length": len(params.text)},
        )

    def _resolve_mode(self, mode: TypeMode, text: str) -> TypeMode:
        """``auto`` decided against the configured threshold."""
        if mode is not TypeMode.AUTO:
            return mode
        from ayris.core.config import get_settings

        threshold = get_settings().actions.input.clipboard_threshold
        if threshold and len(text) >= threshold:
            return TypeMode.CLIPBOARD
        return TypeMode.UNICODE

    def _type_unicode(self, text: str, *, backend: InputBackend, delay: int) -> None:
        """Character by character, so the pause between them is real.

        Handing the whole string to one ``SendInput`` call would be faster and
        would defeat the point: an application that debounces input, or one whose
        edit field cannot keep up, needs the gap the user configured.
        """
        if delay <= 0:
            backend.type_text(text)
            return
        for index, char in enumerate(text):
            if index:
                _pause(delay)
            backend.type_text(char)

    def _type_through_layout(self, text: str, *, backend: InputBackend, delay: int) -> None:
        """Scan codes, with Shift held where the layout requires it."""
        shift = KEYS["shift"]
        unsupported = 0
        for index, char in enumerate(text):
            if index:
                _pause(delay)
            key = char_key(char)
            if key is None:
                unsupported += 1
                backend.type_text(char)
                continue
            needs_shift, _, _ = char_shift_state(char)
            if needs_shift:
                backend.key_down(shift, scancode=True)
            try:
                backend.key_down(key, scancode=True)
                backend.key_up(key, scancode=True)
            finally:
                if needs_shift:
                    backend.key_up(shift, scancode=True)
        if unsupported:
            _log.info(
                "раскладка не набирает %d символов из %d, они ушли как Unicode",
                unsupported,
                len(text),
            )

    def _paste(self, text: str, *, backend: InputBackend) -> None:
        """Clipboard and Ctrl+V, with the old contents restored.

        Goes through :class:`~ayris.actions.system.clipboard.ClipboardBackend`, the
        one clipboard wrapper in the project: it retries a clipboard held open by
        another program, which is exactly what happens right after a Ctrl+C. Both
        writes are hidden from the history monitor — the user copied neither of
        them, the clipboard is only being borrowed as a transport.

        Restoring is best-effort by nature: the receiving application reads the
        clipboard on its own schedule, so the old value goes back after the hold
        delay rather than immediately. Losing what the user had copied would be a
        worse failure than a paste that lands a moment late.
        """
        # Imported here rather than at module level: clipboard.py synthesises its
        # own Ctrl+V through press_combo from this module, so one of the two
        # directions has to be late.
        from ayris.actions.system.clipboard import get_clipboard, suppress_record

        clipboard = get_clipboard()
        try:
            snapshot = clipboard.read()
        except ActionError:
            # Nothing readable is not a reason to refuse the paste; it only means
            # there is nothing worth putting back afterwards.
            previous = ""
        else:
            previous = snapshot.text if snapshot.is_text else ""
        suppress_record(text)
        clipboard.write_text(text)
        _, hold, _ = _timings()
        try:
            press_combo(
                parse_combo("ctrl+v"),
                backend=backend,
                hold_ms=hold,
                scancode=False,
            )
            _pause(max(hold, 20))
        finally:
            try:
                suppress_record(previous)
                clipboard.write_text(previous)
            except ActionError:
                _log.warning("не удалось вернуть буфер обмена", exc_info=True)
