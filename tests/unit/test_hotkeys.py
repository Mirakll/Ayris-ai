"""Задача 30: комбинации клавиш как текст — три нотации, один канонический вид.

The module under test parses strings and returns a value object; there is no
keyboard anywhere in it, which is exactly why it can be pinned this thoroughly.
Three things carry the weight here.

*Every notation collapses onto one spelling.* ``ctrl+alt+v``, ``CTRL-ALT-V``,
``[LCONTROL][LMENU][V]`` and ``^!v`` are the same shortcut written by a person, by
VoiceAttack and by AutoHotkey. :class:`TestOneCanonicalForm` is the table that says
so, and it is the assertion task 36's importers and task 37's registry both lean
on: they compare combinations by value, so any spelling that does not fold produces
a duplicate hotkey nobody can find.

*Sides fold, key identity does not.* ``LCONTROL`` and ``<^`` become plain ``ctrl``
because ``RegisterHotKey`` cannot tell the halves apart — but ``num5`` stays
``num5``, because an application bound to the keypad does not accept the digit row.

*The vocabulary is shared with the synthesiser.* A canonical combination is meant to
go straight into :func:`~ayris.actions.input.keys.parse_combo`, so
:class:`TestSharedVocabulary` walks every name in :data:`KEY_NAMES` through it. Two
tables drifting apart would mean a hotkey that reads back fine and cannot be pressed.

Groups:

* :class:`TestTypedNotation` — what a person types, separators and punctuation keys.
* :class:`TestVoiceAttackNotation` — bracketed ``VK_`` spellings out of ``.vap``.
* :class:`TestAutoHotkeyNotation` — symbol modifiers, braces, hook prefixes, ``::``.
* :class:`TestOneCanonicalForm` — the same shortcut written six ways.
* :class:`TestRefusals` — no key, two keys, unknown key, nothing at all.
* :class:`TestLabels` — the Russian caption the settings window shows.
* :class:`TestValueObject` — frozen, hashable, validated on construction.
* :class:`TestSharedVocabulary` — every canonical name is a key ``keys.py`` knows.
"""

from __future__ import annotations

import pytest

from ayris.actions.input.keys import KEYS, parse_combo
from ayris.core.errors import HotkeyError
from ayris.utils.hotkeys import (
    KEY_NAMES,
    MODIFIER_NAMES,
    Hotkey,
    HotkeyNotationError,
    canonical_hotkey,
    parse_hotkey,
    try_parse_hotkey,
)

pytestmark = pytest.mark.unit


class TestTypedNotation:
    """What a person types into the settings window."""

    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("ctrl+alt+v", "ctrl+alt+v"),
            ("Ctrl + Alt + V", "ctrl+alt+v"),
            ("CTRL-ALT-V", "ctrl+alt+v"),
            ("ctrl alt v", "ctrl+alt+v"),
            ("v+alt+ctrl", "ctrl+alt+v"),
            ("shift+f5", "shift+f5"),
            ("win+up", "win+up"),
            ("ctrl+shift+escape", "ctrl+shift+escape"),
            ("контрол+альт+v", "ctrl+alt+v"),
            ("вин+пробел", "win+space"),
        ],
    )
    def test_folds(self, text: str, canonical: str) -> None:
        assert canonical_hotkey(text) == canonical

    @pytest.mark.parametrize(
        ("text", "modifier", "key"),
        [
            ("ctrl++", "ctrl", "equal"),
            ("ctrl+-", "ctrl", "minus"),
            ("ctrl+plus", "ctrl", "equal"),
            ("alt+.", "alt", "period"),
            ("ctrl+/", "ctrl", "slash"),
            ("ctrl+`", "ctrl", "backquote"),
            ("ctrl+[", "ctrl", "bracketleft"),
            ("ctrl+]", "ctrl", "bracketright"),
            ("ctrl+\\", "ctrl", "backslash"),
        ],
    )
    def test_punctuation_is_a_key_not_a_separator(self, text: str, modifier: str, key: str) -> None:
        """``ctrl++`` is Ctrl and the plus key: a separator with nothing before it is the key."""
        hotkey = parse_hotkey(text)
        assert hotkey.key == key
        assert hotkey.modifiers == (modifier,)

    def test_the_keypad_is_not_the_digit_row(self) -> None:
        """Key identity is kept in full — a macro bound to Num 5 must not get plain 5."""
        assert parse_hotkey("num5") != parse_hotkey("5")
        assert canonical_hotkey("numpad5") == "num5"

    def test_modifiers_come_out_in_one_order(self) -> None:
        assert canonical_hotkey("shift+win+alt+ctrl+a") == "ctrl+alt+shift+win+a"


class TestVoiceAttackNotation:
    """``[LCONTROL][LMENU][V]`` — what a ``.vap`` profile of task 36 carries."""

    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("[LCONTROL][LMENU][V]", "ctrl+alt+v"),
            ("[RCONTROL][RMENU][V]", "ctrl+alt+v"),
            ("[LCONTROL]+[V]", "ctrl+v"),
            ("[LCONTROL], [V]", "ctrl+v"),
            ("[ LSHIFT ][ F5 ]", "shift+f5"),
            ("[PRIOR]", "pageup"),
            ("[NEXT]", "pagedown"),
            ("[CAPITAL]", "capslock"),
            ("[SNAPSHOT]", "printscreen"),
            ("[OEM_PLUS]", "equal"),
            ("[NUMPAD5]", "num5"),
            ("[LWIN][D]", "win+d"),
            ("[VOLUME_UP]", "volumeup"),
        ],
    )
    def test_folds(self, text: str, canonical: str) -> None:
        assert canonical_hotkey(text) == canonical

    @pytest.mark.parametrize("text", ["[LCONTROL]V", "[LCONTROL", "LCONTROL]", "[]"])
    def test_malformed_brackets_are_refused(self, text: str) -> None:
        with pytest.raises(HotkeyNotationError):
            parse_hotkey(text)


class TestAutoHotkeyNotation:
    """``^!v`` — what an ``.ahk`` script carries, hook prefixes and all."""

    @pytest.mark.parametrize(
        ("text", "canonical"),
        [
            ("^!v", "ctrl+alt+v"),
            ("^!V", "ctrl+alt+v"),
            ("+{F5}", "shift+f5"),
            ("~$#z", "win+z"),
            ("<^>!a", "ctrl+alt+a"),
            ("*^c", "ctrl+c"),
            ("#{Up}", "win+up"),
            ("^!v::", "ctrl+alt+v"),
            ("^{NumpadEnter}", "ctrl+numenter"),
            ("!{Media_Play_Pause}", "alt+mediaplay"),
        ],
    )
    def test_folds(self, text: str, canonical: str) -> None:
        assert canonical_hotkey(text) == canonical

    def test_symbols_alone_are_not_a_combination(self) -> None:
        """``^+`` is Ctrl and Shift with nothing to press, so it is not a binding."""
        assert try_parse_hotkey("^+") is None

    def test_a_bare_plus_is_the_plus_key(self) -> None:
        """A single ``+`` is not AutoHotkey's Shift — there would be no key left."""
        assert canonical_hotkey("+") == "equal"


class TestOneCanonicalForm:
    """The promise the whole module exists for: one shortcut, one spelling."""

    @pytest.mark.parametrize(
        "text",
        [
            "ctrl+alt+v",
            "Ctrl + Alt + V",
            "CTRL-ALT-V",
            "ctrl alt v",
            "[LCONTROL][LMENU][V]",
            "[RCONTROL][RMENU][V]",
            "^!v",
            "~$<^>!v::",
            "контрол+альт+V",
        ],
    )
    def test_every_notation_gives_the_same_hotkey(self, text: str) -> None:
        assert parse_hotkey(text) == Hotkey(key="v", ctrl=True, alt=True)
        assert canonical_hotkey(text) == "ctrl+alt+v"

    def test_canonical_text_parses_back_to_itself(self) -> None:
        """Whatever came in, its canonical form is a fixed point."""
        for text in ("[LWIN][UP]", "^+{F12}", "Ctrl - Numpad5", "alt+pgup"):
            once = canonical_hotkey(text)
            assert canonical_hotkey(once) == once


class TestRefusals:
    """Text that is not a combination has to say so in Russian."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", "не указана"),
            ("   ", "не указана"),
            ("::", "нет комбинации"),
            ("ctrl+alt", "только модификаторы"),
            ("ctrl", "только модификаторы"),
            ("a+b", "больше одной клавиши"),
            ("ctrl+бубубо", "Не понимаю"),
            ("[NOSUCHKEY]", "Не понимаю"),
        ],
    )
    def test_message_says_what_is_wrong(self, text: str, expected: str) -> None:
        with pytest.raises(HotkeyNotationError) as raised:
            parse_hotkey(text)
        assert expected in raised.value.user_message

    def test_the_error_is_a_hotkey_error(self) -> None:
        """Code that does not care which kind of hotkey trouble it is catches the parent."""
        with pytest.raises(HotkeyError):
            parse_hotkey("ctrl+бубубо")

    @pytest.mark.parametrize("text", ["", "ctrl", "a+b", "не комбинация"])
    def test_try_parse_returns_none_instead(self, text: str) -> None:
        assert try_parse_hotkey(text) is None

    def test_try_parse_returns_the_hotkey_when_it_reads(self) -> None:
        assert try_parse_hotkey("ctrl+alt+v") == Hotkey(key="v", ctrl=True, alt=True)


class TestLabels:
    """The caption the settings window and the command list show."""

    @pytest.mark.parametrize(
        ("text", "label"),
        [
            ("ctrl+alt+v", "Ctrl + Alt + V"),
            ("win+up", "Win + Стрелка вверх"),
            ("ctrl+shift+escape", "Ctrl + Shift + Esc"),
            ("f5", "F5"),
            ("num5", "Num 5"),
            ("ctrl+space", "Ctrl + Пробел"),
            ("alt+enter", "Alt + Ввод"),
            ("ctrl+;", "Ctrl + ;"),
            ("volumeup", "Громче"),
        ],
    )
    def test_label_ru(self, text: str, label: str) -> None:
        assert parse_hotkey(text).label_ru == label

    def test_every_key_has_a_caption_that_is_not_its_name(self) -> None:
        """Nothing falls through to a bare internal name a person would not recognise."""
        for name in sorted(KEY_NAMES):
            label = Hotkey(key=name).label_ru
            assert label, f"у клавиши «{name}» нет подписи"
            assert label == label.strip()
            assert "_" not in label, f"подпись «{label}» выглядит как имя из кода"


class TestValueObject:
    """:class:`Hotkey` itself: task 37 keeps these in sets to find conflicts."""

    def test_it_is_hashable_and_equal_by_value(self) -> None:
        registered = {parse_hotkey("ctrl+alt+v"), parse_hotkey("[LCONTROL][LMENU][V]")}
        assert len(registered) == 1
        assert parse_hotkey("^!v") in registered

    def test_it_is_frozen(self) -> None:
        hotkey = parse_hotkey("ctrl+v")
        with pytest.raises(AttributeError):
            hotkey.key = "c"  # type: ignore[misc]

    def test_construction_checks_the_key_name(self) -> None:
        """A canonical name, not a spelling: the aliases are the parser's job."""
        with pytest.raises(HotkeyNotationError):
            Hotkey(key="V")
        with pytest.raises(HotkeyNotationError):
            Hotkey(key="LCONTROL")

    def test_str_is_the_canonical_form(self) -> None:
        assert str(parse_hotkey("Ctrl + Alt + V")) == "ctrl+alt+v"

    def test_a_bare_modifier_is_not_a_combination(self) -> None:
        """``[LSHIFT]`` alone has nothing to press with it, so the parser says so.

        The value object still allows it — task 37 may hold a bare modifier — but text
        that names only modifiers is a half-written binding and is refused.
        """
        assert Hotkey(key="shift").canonical == "shift"
        assert try_parse_hotkey("[LSHIFT]") is None


class TestSharedVocabulary:
    """The names here and the names in ``keys.py`` are one vocabulary, or nothing works."""

    def test_every_name_is_a_key_the_synthesiser_knows(self) -> None:
        missing = sorted(KEY_NAMES - set(KEYS))
        assert not missing, f"клавиш нет в keys.py: {missing}"

    def test_a_canonical_combination_can_be_pressed(self) -> None:
        """What the schema stores is what :func:`parse_combo` takes, without translation."""
        for name in sorted(KEY_NAMES - set(MODIFIER_NAMES)):
            combo = Hotkey(key=name, ctrl=True, shift=True).canonical
            strokes = parse_combo(combo)
            assert len(strokes) == 3, f"«{combo}» разобралась в {len(strokes)} клавиш"

    def test_the_modifier_names_are_the_same_three_letters(self) -> None:
        assert set(MODIFIER_NAMES) <= set(KEYS)
        assert parse_combo("ctrl+alt+shift+win+a")
