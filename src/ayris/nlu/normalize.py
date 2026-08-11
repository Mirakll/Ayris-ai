"""Bringing spoken text and stored triggers to the same shape.

Matching «Айрис, открой браузер!» against a trigger stored as ``открой браузер``
is only possible if both sides go through one function, and that is the whole
purpose of this module: :func:`normalize` is applied to the recogniser's output
and to every trigger phrase at index time, so a difference that survives it is a
real difference and not a comma.

Five things happen, each because of a way recognisers and users actually differ:

**Case and ``ё``.** Recognisers are inconsistent about both — Vosk writes
``еще``, faster-whisper writes ``Ещё`` — so text is lowercased and ``ё`` is
folded onto ``е``. The fold is one-directional and lossless for matching: no
Russian pair of words differs only by that letter in a way a command cares
about.

**Punctuation.** Cloud recognisers punctuate, offline ones do not. Everything
that is not a letter, a digit or a space is dropped, and runs of whitespace
collapse to one.

**The address.** A user says «Айрис, открой браузер», and the trigger is
``открой браузер``; but a trigger can also be written *with* the address, and
both have to end up the same. A leading address form — and an optional filler
before it, ``эй``/``окей``/``слушай`` — is therefore stripped from the front.
It is stripped only when something is left afterwards, so the phrase «Айрис»
alone still matches a trigger that consists of the address itself.

**Numerals.** «громкость пятьдесят» and «громкость 50» are the same command, so
number words become digits: units, teens, tens, hundreds and the scale words
``тысяча``/``миллион``/``миллиард`` in the cases people actually speak. Words
combine the way Russian composes them — ``сто двадцать три`` → ``123``,
``две тысячи двадцать пять`` → ``2025`` — but only while the magnitudes keep
decreasing, so ``три четыре`` stays ``3 4`` instead of collapsing into ``7``.

**Both variants are kept.** The digit conversion is exactly what a regex trigger
written against words must not see, so :class:`NormalizedText` carries the text
twice: :attr:`~NormalizedText.text` with digits for exact and fuzzy matching,
:attr:`~NormalizedText.spoken` with the number words left alone for regex. They
are the same object when the phrase has no numerals in it, which is the common
case, and :attr:`~NormalizedText.has_numerals` says so without comparing.

The module holds no state and touches nothing outside itself: normalising is a
pure function of its argument, which is what makes it usable from the index
builder, from the matcher and from the settings window alike.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

__all__ = [
    "CARDINAL_WORDS",
    "DEFAULT_ADDRESS_FORMS",
    "FILLER_WORDS",
    "SCALE_WORDS",
    "NormalizedText",
    "fold_letters",
    "fold_yo",
    "normalize",
    "normalize_text",
    "numeral_rank",
    "spoken_to_digits",
    "strip_address",
]

#: How the assistant is addressed. Mirrors the wake word variants shipped in
#: :class:`ayris.core.config.WakeConfig`, because the recogniser transcribes
#: whichever one was said and the phrase behind it is the command either way.
DEFAULT_ADDRESS_FORMS: Final[tuple[str, ...]] = ("айрис", "аирис", "эйрис", "ирис")

#: Words that may precede the address and carry no meaning of their own. Only
#: removed together with an address form: «слушай» on its own is a command.
FILLER_WORDS: Final[frozenset[str]] = frozenset({"эй", "хэй", "хей", "окей", "ок", "слушай"})

#: Everything that is not a letter, a digit or a space. ``\w`` would keep the
#: underscore, which is punctuation as far as speech is concerned.
_PUNCTUATION: Final = re.compile(r"[^\w\s]|_", re.UNICODE)

#: The percent sign, spelled out before punctuation is stripped. A user typing
#: «на 10% тише» into the command library and a recogniser hearing «на десять
#: процентов тише» mean the same thing, and dropping the sign as punctuation
#: would leave the first one as a bare number — the one shape
#: :class:`ayris.nlu.slot_types.VolumeType` cannot tell from an absolute step.
_PERCENT_SIGN: Final = re.compile(r"%")

_WHITESPACE: Final = re.compile(r"\s+")

#: Numerals below one hundred and the hundreds, in the nominative plus the
#: oblique forms that survive dictation: «на пятьдесят процентов», «двух минут».
#: Feminine and neuter variants matter — «две минуты» is as common as «два часа».
#:
#: Public because :mod:`ayris.nlu.numbers` parses the same words in a context
#: where they must become an ``int`` rather than a substring, and two tables
#: would drift apart the first time a case form is added to one of them.
CARDINAL_WORDS: Final[dict[str, int]] = {
    "ноль": 0,
    "нуль": 0,
    "один": 1,
    "одна": 1,
    "одно": 1,
    "одну": 1,
    "одного": 1,
    "два": 2,
    "две": 2,
    "двух": 2,
    "двум": 2,
    "три": 3,
    "трех": 3,
    "трём": 3,
    "трем": 3,
    "четыре": 4,
    "четырех": 4,
    "пять": 5,
    "пяти": 5,
    "шесть": 6,
    "шести": 6,
    "семь": 7,
    "семи": 7,
    "восемь": 8,
    "восьми": 8,
    "девять": 9,
    "девяти": 9,
    "десять": 10,
    "десяти": 10,
    "одиннадцать": 11,
    "одиннадцати": 11,
    "двенадцать": 12,
    "двенадцати": 12,
    "тринадцать": 13,
    "тринадцати": 13,
    "четырнадцать": 14,
    "четырнадцати": 14,
    "пятнадцать": 15,
    "пятнадцати": 15,
    "шестнадцать": 16,
    "шестнадцати": 16,
    "семнадцать": 17,
    "семнадцати": 17,
    "восемнадцать": 18,
    "восемнадцати": 18,
    "девятнадцать": 19,
    "девятнадцати": 19,
    "двадцать": 20,
    "двадцати": 20,
    "тридцать": 30,
    "тридцати": 30,
    "сорок": 40,
    "сорока": 40,
    "пятьдесят": 50,
    "пятидесяти": 50,
    "шестьдесят": 60,
    "шестидесяти": 60,
    "семьдесят": 70,
    "семидесяти": 70,
    "восемьдесят": 80,
    "восьмидесяти": 80,
    "девяносто": 90,
    "девяноста": 90,
    "сто": 100,
    "ста": 100,
    "двести": 200,
    "двухсот": 200,
    "триста": 300,
    "трехсот": 300,
    "четыреста": 400,
    "четырехсот": 400,
    "пятьсот": 500,
    "пятисот": 500,
    "шестьсот": 600,
    "шестисот": 600,
    "семьсот": 700,
    "семисот": 700,
    "восемьсот": 800,
    "восьмисот": 800,
    "девятьсот": 900,
    "девятисот": 900,
}

#: Multipliers. ``полтора`` is deliberately absent: it is not an integer and the
#: slots of task 16 expect one. :mod:`ayris.nlu.numbers` handles it, on the side
#: where a fraction is a valid answer.
SCALE_WORDS: Final[dict[str, int]] = {
    "тысяча": 1_000,
    "тысячи": 1_000,
    "тысяч": 1_000,
    "тысяче": 1_000,
    "тысячу": 1_000,
    "миллион": 1_000_000,
    "миллиона": 1_000_000,
    "миллионов": 1_000_000,
    "миллиард": 1_000_000_000,
    "миллиарда": 1_000_000_000,
    "миллиардов": 1_000_000_000,
}


def numeral_rank(value: int) -> int:
    """Magnitude class of a numeral, used to decide whether words compose.

    Russian builds a number from strictly decreasing parts — hundreds, then
    tens, then units. Teens share the rank of units because nothing may follow
    them: ``пятнадцать пять`` is two numbers, not seventeen point something.

    Public for :mod:`ayris.nlu.numbers`, which composes the same words under the
    same rule while keeping a fraction and a sign that this module has no use
    for.
    """
    if value >= 100:
        return 2
    if value >= 20:
        return 1
    return 0


@dataclass(frozen=True, slots=True)
class NormalizedText:
    """One phrase, in the shapes the matcher needs it in.

    ``raw`` is kept for the log and the history table: when a command does not
    fire, what the user actually said is the only thing worth showing.
    """

    raw: str
    text: str
    spoken: str
    words: tuple[str, ...]

    @property
    def has_numerals(self) -> bool:
        """Whether number words were rewritten, i.e. the two texts differ."""
        return self.text != self.spoken

    @property
    def is_empty(self) -> bool:
        """Whether nothing survived normalisation — silence, or pure punctuation."""
        return not self.text

    def __len__(self) -> int:
        """Length of the matched form, so ``len(phrase)`` needs no attribute."""
        return len(self.text)


def fold_yo(text: str) -> str:
    """Fold ``ё`` onto ``е``, keeping the case of everything else.

    The half of :func:`fold_letters` that a regex pattern may safely be put
    through: lowercasing a pattern would turn ``\\S`` into ``\\s`` and
    ``(?P<app>...)`` into ``(?p<app>...)``, which does not parse.
    """
    return text.replace("ё", "е").replace("Ё", "Е")


def fold_letters(text: str) -> str:
    """Lowercase and fold ``ё`` onto ``е``.

    Split out from :func:`normalize` because a regex pattern needs this half and
    must not get the rest: dropping punctuation from ``(?P<app>.+)`` would leave
    a pattern that no longer parses.
    """
    return fold_yo(text.lower())


def strip_address(
    words: tuple[str, ...], forms: tuple[str, ...] = DEFAULT_ADDRESS_FORMS
) -> tuple[str, ...]:
    """Drop a leading address, with any filler in front of it.

    Returns ``words`` unchanged when the address is all there is: «айрис» has to
    stay matchable as a phrase, otherwise a trigger consisting of the wake word
    could never fire.
    """
    index = 0
    while index < len(words) and words[index] in FILLER_WORDS:
        index += 1
    if index >= len(words) or words[index] not in forms:
        return words
    rest = words[index + 1 :]
    return rest if rest else words


def spoken_to_digits(words: tuple[str, ...]) -> tuple[str, ...]:
    """Rewrite runs of number words as digits, leaving everything else alone.

    The accumulator follows the way the language composes numbers: parts add up
    while their magnitude decreases, a scale word multiplies what came before
    it, and anything else closes the run. A word that is not a numeral flushes
    the run and is copied through untouched.
    """
    out: list[str] = []
    total = 0
    group = 0
    last_rank = 3
    last_scale = 0
    pending = False

    def flush() -> None:
        nonlocal total, group, last_rank, last_scale, pending
        if pending:
            out.append(str(total + group))
        total = 0
        group = 0
        last_rank = 3
        last_scale = 0
        pending = False

    for word in words:
        value = CARDINAL_WORDS.get(word)
        if value is not None:
            rank = numeral_rank(value)
            if rank >= last_rank:
                flush()
            group += value
            last_rank = numeral_rank(value)
            pending = True
            continue

        scale = SCALE_WORDS.get(word)
        if scale is not None:
            # ``тысяча тысяча`` is not a number; neither is ``миллион миллиард``.
            # A scale must be smaller than the previous one to keep composing.
            if last_scale and scale >= last_scale:
                flush()
            total += (group or 1) * scale
            group = 0
            last_rank = 3
            last_scale = scale
            pending = True
            continue

        flush()
        out.append(word)

    flush()
    return tuple(out)


def normalize(
    text: str,
    *,
    address_forms: tuple[str, ...] = DEFAULT_ADDRESS_FORMS,
    strip_wake_word: bool = True,
    numerals: bool = True,
) -> NormalizedText:
    """Normalise a phrase for matching.

    Args:
        text: Raw text — recogniser output, or a trigger phrase from the library.
        address_forms: How the assistant may be addressed. Comes from the wake
            word settings when the caller has them; the default mirrors the
            variants Ayris ships with.
        strip_wake_word: Whether a leading address is removed.
        numerals: Whether number words become digits. Turning it off yields a
            :class:`NormalizedText` whose two texts are identical, which is what
            the index does when it needs the spoken form on its own.

    Returns:
        The phrase folded, stripped, and — in :attr:`NormalizedText.text` —
        with its numerals rewritten as digits.
    """
    folded = _PERCENT_SIGN.sub(" процентов ", fold_letters(text))
    cleaned = _WHITESPACE.sub(" ", _PUNCTUATION.sub(" ", folded)).strip()
    words = tuple(cleaned.split())
    if strip_wake_word and words:
        words = strip_address(words, address_forms)
    spoken = " ".join(words)
    if not numerals:
        return NormalizedText(raw=text, text=spoken, spoken=spoken, words=words)
    digit_words = spoken_to_digits(words)
    return NormalizedText(
        raw=text,
        text=" ".join(digit_words),
        spoken=spoken,
        words=digit_words,
    )


def normalize_text(text: str) -> str:
    """Shorthand for ``normalize(text).text``, for callers that want the string."""
    return normalize(text).text
