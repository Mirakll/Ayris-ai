"""Key combinations written as text: parsing, canonical form, Russian label.

This is the only place in Ayris that reads a combination *spelled out*. Three
notations arrive from three directions and have to end up meaning the same thing:

* what a person types — ``ctrl+alt+v``, ``Ctrl + Alt + V``, ``CTRL-ALT-V``;
* what VoiceAttack profiles carry — ``[LCONTROL][LMENU][V]``, with Win32 ``VK_``
  spellings inside the brackets (``PRIOR``, ``CAPITAL``, ``OEM_PLUS``);
* what AutoHotkey scripts carry — ``^!v``, ``+{F5}``, ``~$#z``, ``<^>!a``, and the
  ``::`` that ends a hotkey line.

All three collapse onto :class:`Hotkey` and one canonical string, so the importers
of task 36, the hotkey layer of task 37 and the macro editor compare combinations
by value instead of by spelling.

**Sides collapse.** ``LCONTROL`` and ``<^`` both become plain ``ctrl``: a shortcut
means "either Ctrl", ``RegisterHotKey`` cannot tell the halves apart anyway, and
without the folding VoiceAttack's ``[LCONTROL][LMENU][V]`` and a hand-typed
``ctrl+alt+v`` would be two different hotkeys. Key *identity* is kept in full —
``num5`` is not ``5``, because an application bound to the numeric keypad does not
accept the digit row.

**No WinAPI here.** Virtual keys, scan codes and registration live elsewhere
(:mod:`ayris.utils.winapi`, :mod:`ayris.actions.input.keys`, task 37); this module
is text in, value object out, and imports nothing platform-specific — the macro
schema and the ``.vap``/``.ahk`` importers have to parse combinations in the Linux
CI job too. The canonical vocabulary is deliberately the same as
:data:`ayris.actions.input.keys.KEYS`, so a canonical combination can be handed
straight to :func:`ayris.actions.input.keys.parse_combo` for synthesis;
``tests/unit/test_hotkeys.py`` holds the two tables to that promise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from ayris.core.errors import HotkeyError

__all__ = [
    "KEY_NAMES",
    "MODIFIER_NAMES",
    "Hotkey",
    "HotkeyNotationError",
    "canonical_hotkey",
    "parse_hotkey",
    "try_parse_hotkey",
]


class HotkeyNotationError(HotkeyError):
    """A combination could not be read: unknown key, no key, or two of them.

    A subclass rather than a bare :class:`~ayris.core.errors.HotkeyError` because
    "this text is not a combination" and "the system refused to register it" are
    different problems with different fixes; code that does not care catches the
    parent.
    """

    default_user_message = "Не могу разобрать комбинацию клавиш."


#: Modifier names in canonical order — the same order AutoHotkey gives its symbols
#: (``^!+#``), which is also the order Windows shortcuts are written in.
MODIFIER_NAMES: Final[tuple[str, ...]] = ("ctrl", "alt", "shift", "win")

_SPECIAL_KEYS: Final[tuple[str, ...]] = (
    "backspace",
    "tab",
    "enter",
    "escape",
    "space",
    "capslock",
    "apps",
    "pageup",
    "pagedown",
    "end",
    "home",
    "left",
    "up",
    "right",
    "down",
    "insert",
    "delete",
    "printscreen",
    "numlock",
    "scrolllock",
    "pause",
    "nummultiply",
    "numadd",
    "numsubtract",
    "numdecimal",
    "numdivide",
    "numenter",
    "semicolon",
    "equal",
    "comma",
    "minus",
    "period",
    "slash",
    "backquote",
    "bracketleft",
    "backslash",
    "bracketright",
    "quote",
    "volumemute",
    "volumedown",
    "volumeup",
    "medianext",
    "mediaprev",
    "mediastop",
    "mediaplay",
    "browserback",
    "browserforward",
    "browserrefresh",
    "browserstop",
    "browsersearch",
    "browserhome",
)


def _build_key_names() -> frozenset[str]:
    """Every name a combination may end with, generated the way ``keys.py`` does.

    Written as rules instead of three hundred literals so the two vocabularies
    cannot silently drift; the test module asserts this set is a subset of
    :data:`ayris.actions.input.keys.KEYS`.
    """
    names: set[str] = set(MODIFIER_NAMES)
    names.update(_SPECIAL_KEYS)
    names.update(str(digit) for digit in range(10))
    names.update(chr(ord("a") + index) for index in range(26))
    names.update(f"f{number}" for number in range(1, 25))
    names.update(f"num{digit}" for digit in range(10))
    return frozenset(names)


#: Canonical key names, modifiers included.
KEY_NAMES: Final[frozenset[str]] = _build_key_names()

_ALIAS_LITERALS: Final[dict[str, str]] = {
    # Modifiers, with the sided spellings of VoiceAttack and AutoHotkey folded onto
    # the plain ones. ``menu`` is Alt because that is what ``VK_MENU`` means; the
    # context-menu key is ``apps``, as in Win32 and AutoHotkey.
    "control": "ctrl",
    "ctl": "ctrl",
    "lctrl": "ctrl",
    "rctrl": "ctrl",
    "lcontrol": "ctrl",
    "rcontrol": "ctrl",
    "leftcontrol": "ctrl",
    "rightcontrol": "ctrl",
    "контрол": "ctrl",
    "контрл": "ctrl",
    "ктрл": "ctrl",
    "menu": "alt",
    "lmenu": "alt",
    "rmenu": "alt",
    "lalt": "alt",
    "ralt": "alt",
    "leftalt": "alt",
    "rightalt": "alt",
    "altgr": "alt",
    "альт": "alt",
    "меню": "alt",
    "lshift": "shift",
    "rshift": "shift",
    "leftshift": "shift",
    "rightshift": "shift",
    "шифт": "shift",
    "сдвиг": "shift",
    "windows": "win",
    "super": "win",
    "meta": "win",
    "cmd": "win",
    "lwin": "win",
    "rwin": "win",
    "lwindows": "win",
    "rwindows": "win",
    "leftwin": "win",
    "rightwin": "win",
    "вин": "win",
    "винда": "win",
    "виндовс": "win",
    "пуск": "win",
    # Typing and editing. ``return``, ``back`` and ``capital`` are the Win32 names
    # VoiceAttack writes; ``bs``, ``del``, ``ins`` are AutoHotkey shorthand.
    "return": "enter",
    "ввод": "enter",
    "энтер": "enter",
    "esc": "escape",
    "эск": "escape",
    "эскейп": "escape",
    "пробел": "space",
    "таб": "tab",
    "табуляция": "tab",
    "bs": "backspace",
    "back": "backspace",
    "бэкспейс": "backspace",
    "забой": "backspace",
    "del": "delete",
    "удалить": "delete",
    "ins": "insert",
    "вставка": "insert",
    "capital": "capslock",
    "капс": "capslock",
    "appskey": "apps",
    "контекстное": "apps",
    # Navigation. ``prior``/``next`` are Win32 for Page Up and Page Down.
    "pgup": "pageup",
    "prior": "pageup",
    "pgdn": "pagedown",
    "pgdown": "pagedown",
    "next": "pagedown",
    "uparrow": "up",
    "downarrow": "down",
    "leftarrow": "left",
    "rightarrow": "right",
    "вверх": "up",
    "вниз": "down",
    "влево": "left",
    "вправо": "right",
    "домой": "home",
    "начало": "home",
    "конец": "end",
    "страницавверх": "pageup",
    "страницавниз": "pagedown",
    # Punctuation, named by the character sitting on the key. ``oem*`` is how Win32
    # and VoiceAttack spell the same positions.
    ";": "semicolon",
    "oem1": "semicolon",
    "=": "equal",
    "+": "equal",
    "plus": "equal",
    "плюс": "equal",
    "oemplus": "equal",
    ",": "comma",
    "oemcomma": "comma",
    "-": "minus",
    "_": "minus",
    "минус": "minus",
    "oemminus": "minus",
    ".": "period",
    "oemperiod": "period",
    "/": "slash",
    "oem2": "slash",
    "`": "backquote",
    "~": "backquote",
    "oem3": "backquote",
    "[": "bracketleft",
    "oem4": "bracketleft",
    "\\": "backslash",
    "oem5": "backslash",
    "]": "bracketright",
    "oem6": "bracketright",
    "'": "quote",
    '"': "quote",
    "oem7": "quote",
    # Numpad, as AutoHotkey and Win32 spell it. The digits are generated below.
    "numpadmult": "nummultiply",
    "numpadmultiply": "nummultiply",
    "multiply": "nummultiply",
    "numpadadd": "numadd",
    "add": "numadd",
    "numpadsub": "numsubtract",
    "numpadsubtract": "numsubtract",
    "subtract": "numsubtract",
    "numpaddot": "numdecimal",
    "numpaddecimal": "numdecimal",
    "decimal": "numdecimal",
    "numpaddiv": "numdivide",
    "numpaddivide": "numdivide",
    "divide": "numdivide",
    "numpadenter": "numenter",
    "numpadreturn": "numenter",
    # System and media. Underscores are dropped before the lookup, so AutoHotkey's
    # ``Media_Play_Pause`` and Win32's ``VK_VOLUME_UP`` land here as one word.
    "prtsc": "printscreen",
    "printscr": "printscreen",
    "snapshot": "printscreen",
    "capslockstate": "capslock",
    "play": "mediaplay",
    "mediaplaypause": "mediaplay",
    "mediaprevious": "mediaprev",
    "mediaprevtrack": "mediaprev",
    "medianexttrack": "medianext",
    "mute": "volumemute",
    "browserrefreshkey": "browserrefresh",
}


def _build_aliases() -> dict[str, str]:
    """Alias table with the numpad digits generated rather than typed out."""
    aliases = dict(_ALIAS_LITERALS)
    aliases.update({f"numpad{digit}": f"num{digit}" for digit in range(10)})
    return aliases


_ALIASES: Final[dict[str, str]] = _build_aliases()

#: Russian captions for the keys whose name is not already what a person reads.
#: Letters, digits and function keys fall through to upper case, punctuation to the
#: character itself.
_LABELS_RU: Final[dict[str, str]] = {
    "ctrl": "Ctrl",
    "alt": "Alt",
    "shift": "Shift",
    "win": "Win",
    "backspace": "Backspace",
    "tab": "Tab",
    "enter": "Ввод",
    "escape": "Esc",
    "space": "Пробел",
    "capslock": "Caps Lock",
    "apps": "Контекстное меню",
    "pageup": "Page Up",
    "pagedown": "Page Down",
    "end": "End",
    "home": "Home",
    "left": "Стрелка влево",
    "up": "Стрелка вверх",
    "right": "Стрелка вправо",
    "down": "Стрелка вниз",
    "insert": "Insert",
    "delete": "Delete",
    "printscreen": "Print Screen",
    "numlock": "Num Lock",
    "scrolllock": "Scroll Lock",
    "pause": "Pause",
    "nummultiply": "Num *",
    "numadd": "Num +",
    "numsubtract": "Num -",
    "numdecimal": "Num .",
    "numdivide": "Num /",
    "numenter": "Num Ввод",
    "semicolon": ";",
    "equal": "=",
    "comma": ",",
    "minus": "-",
    "period": ".",
    "slash": "/",
    "backquote": "`",
    "bracketleft": "[",
    "backslash": "\\",
    "bracketright": "]",
    "quote": "'",
    "volumemute": "Без звука",
    "volumedown": "Тише",
    "volumeup": "Громче",
    "medianext": "Следующий трек",
    "mediaprev": "Предыдущий трек",
    "mediastop": "Стоп",
    "mediaplay": "Плей/пауза",
    "browserback": "Браузер: назад",
    "browserforward": "Браузер: вперёд",
    "browserrefresh": "Браузер: обновить",
    "browserstop": "Браузер: стоп",
    "browsersearch": "Браузер: поиск",
    "browserhome": "Браузер: домой",
}

#: AutoHotkey modifier symbols. ``<`` and ``>`` narrow the next symbol to one side
#: of the keyboard and are read but dropped — see the note about sides above.
_AHK_MODIFIERS: Final[dict[str, str]] = {"^": "ctrl", "!": "alt", "+": "shift", "#": "win"}

#: AutoHotkey prefixes that change how a hotkey is hooked, not what it is:
#: ``~`` passes the key through, ``$`` forces the keyboard hook, ``*`` ignores the
#: other modifiers.
_AHK_PREFIXES: Final[str] = "~$*"

_SIDES: Final[str] = "<>"
_SEPARATORS: Final[str] = "+-"


def _key_label(name: str) -> str:
    label = _LABELS_RU.get(name)
    if label is not None:
        return label
    if name.startswith("num") and name[3:].isdigit():
        return f"Num {name[3:]}"
    return name.upper()


@dataclass(frozen=True, slots=True)
class Hotkey:
    """One combination: a key plus the modifiers held with it.

    Frozen and hashable on purpose — task 37 detects conflicting registrations by
    comparing values, and a set of :class:`Hotkey` is the cheapest way to do that.
    Constructing one directly validates the key name; parsing text is
    :func:`parse_hotkey`.
    """

    key: str
    ctrl: bool = False
    alt: bool = False
    shift: bool = False
    win: bool = False

    def __post_init__(self) -> None:
        if self.key not in KEY_NAMES:
            raise HotkeyNotationError(
                f"unknown key {self.key!r}",
                user_message=f"Клавиша «{self.key}» мне неизвестна.",
            )

    @property
    def modifiers(self) -> tuple[str, ...]:
        """Held modifiers in canonical order."""
        held = ((self.ctrl, "ctrl"), (self.alt, "alt"), (self.shift, "shift"), (self.win, "win"))
        return tuple(name for on, name in held if on)

    @property
    def canonical(self) -> str:
        """The one spelling everything else stores and compares: ``ctrl+alt+v``."""
        return "+".join((*self.modifiers, self.key))

    @property
    def label_ru(self) -> str:
        """Caption for the UI: ``Ctrl + Alt + V``, ``Win + Стрелка вверх``."""
        return " + ".join([_LABELS_RU[name] for name in self.modifiers] + [_key_label(self.key)])

    def __str__(self) -> str:
        return self.canonical


#: One bracketed segment of VoiceAttack notation, with ``+`` and ``,`` between
#: segments tolerated: profiles have been seen written both ways.
_BRACKETED = re.compile(r"[\s+,]*\[\s*([^\[\]]+?)\s*\][\s+,]*")

#: Nothing but bracketed segments — the whole string, or it is not this notation.
#: A typed ``ctrl+[`` also contains a bracket, and there the bracket is the key.
_ALL_BRACKETED = re.compile(rf"(?:{_BRACKETED.pattern})+")


def _normalize_token(token: str) -> str:
    cleaned = token.strip()
    if len(cleaned) > 2 and cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1].strip()
    cleaned = cleaned.lower()
    if len(cleaned) > 1:
        cleaned = cleaned.replace("_", "").replace(" ", "")
    return cleaned


def _resolve(token: str, raw: str) -> str:
    """One token to a canonical name, folding sides and every known spelling."""
    name = _normalize_token(token)
    canonical = _ALIASES.get(name, name)
    if canonical not in KEY_NAMES:
        raise HotkeyNotationError(
            f"unknown key {token!r} in {raw!r}",
            user_message=f"Не понимаю «{token.strip()}» в комбинации «{raw}».",
        )
    return canonical


def _tokenize_brackets(text: str) -> list[str]:
    """Every bracketed segment in order; the caller has checked there is nothing else."""
    return [match.group(1) for match in _BRACKETED.finditer(text)]


def _looks_like_ahk(text: str) -> bool:
    """True when the text is symbol notation rather than named keys.

    ``^`` and ``#`` appear in no other notation, and a leading ``!``, ``+`` or side
    marker followed by something else can only be AutoHotkey — a bare ``+`` is the
    plus key and stays out of here.
    """
    body = text.lstrip(_AHK_PREFIXES)
    if not body or (body[0] not in _AHK_MODIFIERS and body[0] not in _SIDES):
        return False
    return bool(body.lstrip("".join(_AHK_MODIFIERS) + _SIDES))


def _tokenize_ahk(text: str) -> list[str]:
    """``~$*`` prefixes off, symbols to modifier names, the rest is the key.

    The loop stops one character before the end, so the final character is always
    left for the key and a string of nothing but symbols cannot eat its own:
    ``^+`` would be Ctrl and Shift with nothing to press, which is why
    :func:`_looks_like_ahk` refuses it before this is reached.
    """
    body = text.lstrip(_AHK_PREFIXES).strip()
    tokens: list[str] = []
    index = 0
    while index < len(body) - 1:
        char = body[index]
        if char in _SIDES:
            index += 1
            continue
        modifier = _AHK_MODIFIERS.get(char)
        if modifier is None:
            break
        tokens.append(modifier)
        index += 1
    rest = body[index:].strip()
    if rest.startswith("{") and rest.endswith("}"):
        rest = rest[1:-1].strip()
    if rest:
        tokens.append(rest)
    return tokens


def _tokenize_plain(text: str) -> list[str]:
    """Split named keys on ``+``, ``-`` and whitespace, letting both be keys.

    A separator right after another separator has nothing before it, so it is the
    key itself: ``ctrl++`` is Ctrl and plus, ``ctrl+-`` is Ctrl and minus. A
    separator after a completed token is a separator, which is what makes
    ``Ctrl + Alt + V`` and ``ctrl+alt+v`` the same three tokens.
    """
    tokens: list[str] = []
    current = ""
    after_token = False
    for char in text:
        if char.isspace():
            if current:
                tokens.append(current)
                current = ""
                after_token = True
            continue
        if char in _SEPARATORS:
            if current:
                tokens.append(current)
                current = ""
            elif not after_token:
                current = char
            after_token = False
            continue
        current += char
    if current:
        tokens.append(current)
    return tokens


def _tokenize(text: str) -> list[str]:
    """Pick the notation and split the text into key tokens.

    Bracketed notation is claimed only when the text is *entirely* bracketed
    segments: a bracket alone is a key a person can type, and ``ctrl+[`` has to
    reach the named-key reader.
    """
    if _ALL_BRACKETED.fullmatch(text):
        return _tokenize_brackets(text)
    if _looks_like_ahk(text):
        return _tokenize_ahk(text)
    return _tokenize_plain(text)


def parse_hotkey(text: str) -> Hotkey:
    """Read a combination in any supported notation.

    Accepts ``ctrl+alt+v``, ``Ctrl + Alt + V``, ``CTRL-ALT-V``, VoiceAttack's
    ``[LCONTROL][LMENU][V]``, AutoHotkey's ``^!v`` and ``~$<^>!{F5}``, with or
    without the trailing ``::`` of a hotkey line.

    Raises:
        HotkeyNotationError: unknown key, no key at all, or more than one.
    """
    raw = text.strip()
    if not raw:
        raise HotkeyNotationError(
            "empty combination",
            user_message="Комбинация клавиш не указана.",
        )
    body = raw.removesuffix("::").strip()
    if not body:
        raise HotkeyNotationError(
            f"no combination in {raw!r}",
            user_message=f"В строке «{raw}» нет комбинации клавиш.",
        )
    held: dict[str, bool] = dict.fromkeys(MODIFIER_NAMES, False)
    keys: list[str] = []
    for token in _tokenize(body):
        name = _resolve(token, raw)
        if name in held:
            held[name] = True
        else:
            keys.append(name)
    if not keys:
        raise HotkeyNotationError(
            f"no key in {raw!r}",
            user_message=f"В комбинации «{raw}» нет клавиши — только модификаторы.",
        )
    if len(keys) > 1:
        raise HotkeyNotationError(
            f"more than one key in {raw!r}: {keys}",
            user_message=f"В комбинации «{raw}» больше одной клавиши, а нужна одна.",
        )
    return Hotkey(
        key=keys[0],
        ctrl=held["ctrl"],
        alt=held["alt"],
        shift=held["shift"],
        win=held["win"],
    )


def try_parse_hotkey(text: str) -> Hotkey | None:
    """Same as :func:`parse_hotkey`, but unreadable text is ``None``.

    For places that check whether something *is* a combination — an importer
    guessing at a field, a settings form validating as the user types.
    """
    try:
        return parse_hotkey(text)
    except HotkeyNotationError:
        return None


def canonical_hotkey(text: str) -> str:
    """Canonical spelling of a combination written any which way."""
    return parse_hotkey(text).canonical
