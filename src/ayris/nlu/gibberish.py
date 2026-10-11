"""Spotting a keyboard mash — «фывп», «asdf» — before it reaches the model.

A chat turn with no command behind it goes to the language model, which on a
small local model answers a random run of keys with a long, earnest «вы не задали
конкретного вопроса…». The user typed noise and got a lecture. This module names
that noise so the pipeline can answer it with one short line instead of paying for
a model call that cannot do better.

**Only an obvious mash, never a real short word.** The test is narrow on purpose:
a single run of letters, long enough to be unambiguous, that is either almost
vowel-less («ждлкпр») or a slide along one keyboard row («фывп», «qwerty»). A real
reaction — «нишево», «блин», «ахах» — has ordinary letter shape and is left for
the model under the chat prompt, which is told to answer it briefly and in
character. Classifying *those* is the model's job; this is only for text that no
language could pronounce.

The decision is deterministic and dictionary-free: it works on a Cyrillic
(ЙЦУКЕН) and a Latin (QWERTY) keyboard map, so it needs no word list to maintain
and the same call classifies the same text every time.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

__all__ = ["looks_like_gibberish"]

#: Shorter than this and a mash is indistinguishable from a real short word
#: («как», «что», «вот»), so the gate stays out and the model answers.
_MIN_LETTERS: Final = 4

#: Longer than this and keyboard-adjacency says little — a long string slides in
#: and out of rows by chance. A mash the user actually types is short.
_MAX_LETTERS: Final = 14

#: Below this share of vowels the run cannot be pronounced as any word.
_MIN_VOWEL_RATIO: Final = 0.2

#: At or above this share of neighbouring keys the run is a slide along the
#: keyboard, not a word that happens to touch a few adjacent letters.
_MIN_ADJACENCY_RATIO: Final = 0.6

_VOWELS: Final[frozenset[str]] = frozenset("аеёиоуыэюяaeiouy")

# Keyboard rows, Cyrillic then Latin. The two layouts are given disjoint row
# numbers so a Cyrillic letter is never «adjacent» to a Latin one — a token that
# mixes scripts is odd in its own right and must not be called a slide by
# accident. Row/column is the physical key; «adjacent» is a king-move away.
_CYRILLIC_ROWS: Final = ("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")
_LATIN_ROWS: Final = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _build_keyboard() -> dict[str, tuple[int, int]]:
    keys: dict[str, tuple[int, int]] = {}
    for row, letters in enumerate(_CYRILLIC_ROWS):
        for col, letter in enumerate(letters):
            keys[letter] = (row, col)
    for offset, letters in enumerate(_LATIN_ROWS):
        for col, letter in enumerate(letters):
            keys[letter] = (offset + 10, col)
    return keys


_KEYBOARD: Final[dict[str, tuple[int, int]]] = _build_keyboard()


def _adjacency_ratio(letters: str) -> float:
    """Share of neighbouring letter pairs that sit on adjacent keyboard keys.

    A repeated key (``дд``) is not a slide, so a zero move does not count as
    adjacent; an unknown character counts as a pair that is simply not adjacent,
    which only lowers the ratio.
    """
    pairs = list(pairwise(letters))
    if not pairs:
        return 0.0
    adjacent = 0
    for first, second in pairs:
        here = _KEYBOARD.get(first)
        there = _KEYBOARD.get(second)
        if here is None or there is None:
            continue
        step = max(abs(here[0] - there[0]), abs(here[1] - there[1]))
        if 0 < step <= 1:
            adjacent += 1
    return adjacent / len(pairs)


def looks_like_gibberish(text: str) -> bool:
    """Whether ``text`` is an unpronounceable keyboard mash, not words.

    Conservative by design — see the module docstring. Returns ``False`` for
    anything with a space (more than one token is almost always real), anything
    too short or too long to judge, a single repeated letter (an expression like
    «ааааа», not a mash), and anything with an ordinary vowel shape that does not
    slide along the keyboard.
    """
    token = text.strip().casefold()
    if not token or " " in token:
        return False
    if any(not char.isalpha() for char in token):
        return False
    if not (_MIN_LETTERS <= len(token) <= _MAX_LETTERS):
        return False
    if len(set(token)) == 1:
        return False
    vowel_ratio = sum(char in _VOWELS for char in token) / len(token)
    if vowel_ratio < _MIN_VOWEL_RATIO:
        return True
    return _adjacency_ratio(token) >= _MIN_ADJACENCY_RATIO
