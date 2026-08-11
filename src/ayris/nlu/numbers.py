"""Russian numerals, in both directions.

:mod:`ayris.nlu.normalize` already rewrites number words as digits before a
phrase reaches the matcher, and this module deliberately does not repeat that
work: it imports the same :data:`~ayris.nlu.normalize.CARDINAL_WORDS` and
:data:`~ayris.nlu.normalize.SCALE_WORDS` tables and the same composition rule.
What it adds is everything a *slot* needs and matching does not.

**A slot may receive either spelling.** By the time a slot is filled the text
has usually been through normalisation, so ``{volume}`` sees ``"50"``; but a
regex trigger matches against the spoken form, so the same slot may just as well
see ``"пятьдесят"``. :func:`parse_number` accepts both, which is what makes
«громкость пятьдесят» and «громкость 50» produce the identical slot value.

**Not every number is an integer.** Normalisation stops at integers on purpose —
a matcher only needs the digits to line up. A slot has to deal with what people
actually say: ``полтора часа`` is 1.5, ``пол-литра`` is 0.5, ``двадцать пять и
пять`` is 25.5, and «минус пять» is negative. So the accumulator here carries a
sign and a fraction that :func:`~ayris.nlu.normalize.spoken_to_digits` has no
use for.

**Ordinals are a separate question.** ``второй`` is 2 for a slot that wants an
index, but ``полвторого`` is half past *one*, and that difference belongs to
:mod:`ayris.nlu.timeparse`. This module answers only «which number is this
word», through :func:`parse_ordinal`, and leaves the interpretation upstream.

**Percent is a unit, not a number.** ``на десять процентов`` is a
:class:`Percent` — a bare 10 loses the fact that a step was requested rather
than a level, and the volume action needs to tell those apart. The parse is
here rather than in the slot layer because the ``процент`` suffix declines
(``процент``/``процента``/``процентов``) and the stem test belongs next to the
number that precedes it.

**The reverse direction exists for speech.** :func:`number_to_words` renders a
number back into Russian so TTS says «сорок два процента» instead of spelling
digits, and it agrees the noun with the number — :func:`plural_form` picks
between «минута», «минуты» and «минут», which is the difference between an
assistant that sounds Russian and one that does not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Final

from ayris.nlu.normalize import CARDINAL_WORDS, SCALE_WORDS, fold_letters, numeral_rank

__all__ = [
    "FRACTION_WORDS",
    "HALF_WORDS",
    "ORDINAL_WORDS",
    "ParsedNumber",
    "Percent",
    "number_to_words",
    "parse_number",
    "parse_ordinal",
    "parse_percent",
    "plural_form",
    "tokenize",
]

#: Ordinals people say to a voice assistant. Only the low ones: nobody asks for
#: the «сто сорок третий» window, and every form past twenty would double the
#: table for cases that never arrive. Masculine, feminine and neuter, plus the
#: genitive that «полвторого» and «до третьего» are built from.
ORDINAL_WORDS: Final[dict[str, int]] = {
    "первый": 1,
    "первая": 1,
    "первое": 1,
    "первого": 1,
    "второй": 2,
    "вторая": 2,
    "второе": 2,
    "второго": 2,
    "третий": 3,
    "третья": 3,
    "третье": 3,
    "третьего": 3,
    "четвертый": 4,
    "четвертая": 4,
    "четвертое": 4,
    "четвертого": 4,
    "пятый": 5,
    "пятая": 5,
    "пятое": 5,
    "пятого": 5,
    "шестой": 6,
    "шестая": 6,
    "шестое": 6,
    "шестого": 6,
    "седьмой": 7,
    "седьмая": 7,
    "седьмое": 7,
    "седьмого": 7,
    "восьмой": 8,
    "восьмая": 8,
    "восьмое": 8,
    "восьмого": 8,
    "девятый": 9,
    "девятая": 9,
    "девятое": 9,
    "девятого": 9,
    "десятый": 10,
    "десятая": 10,
    "десятое": 10,
    "десятого": 10,
    "одиннадцатый": 11,
    "одиннадцатого": 11,
    "двенадцатый": 12,
    "двенадцатого": 12,
    "тринадцатый": 13,
    "тринадцатого": 13,
    "четырнадцатый": 14,
    "четырнадцатого": 14,
    "пятнадцатый": 15,
    "пятнадцатого": 15,
    "шестнадцатый": 16,
    "шестнадцатого": 16,
    "семнадцатый": 17,
    "семнадцатого": 17,
    "восемнадцатый": 18,
    "восемнадцатого": 18,
    "девятнадцатый": 19,
    "девятнадцатого": 19,
    "двадцатый": 20,
    "двадцатого": 20,
}

#: Words that mean «one and a half» on their own. ``полтора`` and ``полторы``
#: differ only by the gender of what follows, which the number does not care
#: about: «полтора часа» and «полторы минуты» are both 1.5.
HALF_WORDS: Final[frozenset[str]] = frozenset({"полтора", "полторы", "полутора"})

#: Standalone fractions. ``пол`` is here as a whole word — the prefixed form
#: («полчаса», «пол-литра») is split off by :func:`_split_half_prefix`, because
#: the recogniser writes it both ways and the user means the same thing.
FRACTION_WORDS: Final[dict[str, Decimal]] = {
    "пол": Decimal("0.5"),
    "половина": Decimal("0.5"),
    "половины": Decimal("0.5"),
    "половину": Decimal("0.5"),
    # Instrumental, because the «с половиной» form is the common one in speech:
    # «два с половиной часа».
    "половиной": Decimal("0.5"),
    "четверть": Decimal("0.25"),
    "четверти": Decimal("0.25"),
    "треть": Decimal(1) / Decimal(3),
    "трети": Decimal(1) / Decimal(3),
}

#: Words that negate what follows. «минус пять» arrives from a user adjusting a
#: value, and dropping the sign would move the setting the wrong way.
_NEGATIVE_WORDS: Final[frozenset[str]] = frozenset({"минус", "-"})

#: Joins a whole part to a fractional one: «двадцать пять и пять» → 25.5. Only
#: honoured between two numbers, so «пять и десять» stays two numbers unless the
#: second one is short enough to read as a fraction.
_FRACTION_JOINERS: Final[frozenset[str]] = frozenset({"и", "целых", "целые", "целая", "запятая"})

#: Stems of «процент» in the cases dictation produces. Matched by prefix rather
#: than by an exhaustive list because the ending is what varies.
_PERCENT_STEM: Final = "процент"

#: A bare number as digits, with an optional sign and decimal separator. Both
#: separators are accepted: recognisers write «10.5» and «10,5» interchangeably.
_DIGITS: Final = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")

#: How the prefixed half attaches: «полчаса», «пол-литра», «полминуты».
_HALF_PREFIX: Final = re.compile(r"^пол-?(?P<rest>.+)$")

#: Units, teens and tens as words, for the reverse direction. Indexed by value,
#: masculine by default; :func:`number_to_words` swaps in the feminine forms
#: when the noun that follows asks for them.
_ONES_MASCULINE: Final[tuple[str, ...]] = (
    "ноль",
    "один",
    "два",
    "три",
    "четыре",
    "пять",
    "шесть",
    "семь",
    "восемь",
    "девять",
)
_ONES_FEMININE: Final[tuple[str, ...]] = ("ноль", "одна", "две", *_ONES_MASCULINE[3:])
_TEENS: Final[tuple[str, ...]] = (
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
)
_TENS: Final[tuple[str, ...]] = (
    "",
    "десять",
    "двадцать",
    "тридцать",
    "сорок",
    "пятьдесят",
    "шестьдесят",
    "семьдесят",
    "восемьдесят",
    "девяносто",
)
_HUNDREDS: Final[tuple[str, ...]] = (
    "",
    "сто",
    "двести",
    "триста",
    "четыреста",
    "пятьсот",
    "шестьсот",
    "семьсот",
    "восемьсот",
    "девятьсот",
)

#: Scale words with their three plural forms, largest first. The tuple is the
#: argument order of :func:`plural_form`: one, two, five.
_SCALE_FORMS: Final[tuple[tuple[int, tuple[str, str, str], bool], ...]] = (
    (1_000_000_000, ("миллиард", "миллиарда", "миллиардов"), False),
    (1_000_000, ("миллион", "миллиона", "миллионов"), False),
    (1_000, ("тысяча", "тысячи", "тысяч"), True),
)


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    """A number lifted out of text, with what it cost to read it.

    ``value`` is a :class:`~decimal.Decimal` rather than a float because the
    numbers here are decimal by origin — a percentage, a volume level, a count
    of minutes — and ``0.1 + 0.2`` producing ``0.30000000000000004`` in a
    setting the user reads back is a bug report waiting to happen. Callers that
    want an ``int`` ask :attr:`as_int`, which refuses a fractional value rather
    than silently truncating it.

    ``words`` is how many tokens were consumed, so a caller scanning a phrase
    knows where the number ended; :attr:`is_spelled` says whether it arrived as
    words or as digits, which is the difference between a confident parse and
    one that survived a recogniser's spelling.
    """

    value: Decimal
    words: int
    is_spelled: bool = False

    @property
    def as_int(self) -> int | None:
        """The value as an ``int``, or ``None`` when it is fractional."""
        if self.value != self.value.to_integral_value():
            return None
        return int(self.value)

    @property
    def as_float(self) -> float:
        """The value as a ``float``, for callers that do arithmetic with it."""
        return float(self.value)


@dataclass(frozen=True, slots=True)
class Percent:
    """A number the user expressed as a percentage.

    Kept apart from a bare number because «на десять процентов» and «десять» ask
    for different things — a step and a level — and the volume action has to be
    able to tell them apart. ``relative`` records the preposition that said so:
    «на 10 процентов» is a delta, «10 процентов» is a target.
    """

    value: Decimal
    relative: bool = False

    @property
    def as_float(self) -> float:
        """The percentage as a ``float``."""
        return float(self.value)


def _split_half_prefix(word: str) -> tuple[str, ...]:
    """Split «полчаса» into ``("пол", "часа")``, leaving other words alone.

    The recogniser writes the prefixed half three ways — ``полчаса``,
    ``пол часа``, ``пол-часа`` — and the user means one thing by all of them. A
    split is only accepted when the remainder is a word in its own right and not
    itself a numeral, so ``полтора`` and ``полсотни`` survive intact.
    """
    if word in HALF_WORDS or word in FRACTION_WORDS or word in CARDINAL_WORDS:
        return (word,)
    found = _HALF_PREFIX.match(word)
    if found is None:
        return (word,)
    rest = found.group("rest")
    if len(rest) < 3:
        return (word,)
    return ("пол", rest)


def tokenize(text: str) -> list[str]:
    """Fold, split, and separate a prefixed «пол» from the noun behind it.

    Public because :attr:`ParsedNumber.words` counts tokens of *this*
    tokenisation: a caller that scans a phrase word by word — the time parser
    reading «полтора часа тридцать минут» — has to index the same list, or the
    offset the number reports lands on the wrong word.
    """
    words: list[str] = []
    for word in fold_letters(text).replace("-", " - ").split():
        words.extend(_split_half_prefix(word) if word != "-" else (word,))
    return words


def _digits_value(word: str) -> Decimal | None:
    """Read a token that is already written in digits, or return ``None``."""
    if not _DIGITS.match(word):
        return None
    try:
        return Decimal(word.replace(",", "."))
    except InvalidOperation:  # pragma: no cover - guarded by the regex above
        return None


def parse_number(text: str, *, allow_ordinal: bool = False) -> ParsedNumber | None:
    """Read the number at the start of ``text``, or return ``None``.

    Accepts what a slot actually receives: digits (``"50"``, ``"10,5"``), number
    words (``"пятьдесят"``, ``"сто двадцать три"``), the mixed forms dictation
    produces (``"полтора"``, ``"два с половиной"``), and a leading sign.

    Args:
        text: The fragment to read. Only the leading number is consumed; what
            follows is ignored, and :attr:`ParsedNumber.words` says how much was
            taken.
        allow_ordinal: Whether ``второй`` counts as 2. Off by default, because a
            slot expecting a quantity should not accept «второй» as one; the
            time parser turns it on.

    Returns:
        The number and how many tokens it took, or ``None`` when the text does
        not start with one.
    """
    words = tokenize(text)
    if not words:
        return None

    index = 0
    negative = False
    if words[index] in _NEGATIVE_WORDS:
        negative = True
        index += 1
        if index >= len(words):
            return None

    total = Decimal(0)
    group = Decimal(0)
    fraction = Decimal(0)
    last_rank = 3
    last_scale = 0
    spelled = False
    seen = False

    while index < len(words):
        word = words[index]

        digits = _digits_value(word)
        if digits is not None:
            # A digit run is a whole number by itself: «50 20» is two numbers,
            # not seventy, and composing them would turn a misheard pause into
            # a wrong volume.
            if seen:
                break
            group = digits
            seen = True
            index += 1
            continue

        cardinal = CARDINAL_WORDS.get(word)
        if cardinal is not None:
            rank = numeral_rank(cardinal)
            if seen and rank >= last_rank:
                break
            group += cardinal
            last_rank = rank
            spelled = True
            seen = True
            index += 1
            continue

        scale = SCALE_WORDS.get(word)
        if scale is not None:
            if last_scale and scale >= last_scale:
                break
            total += (group or Decimal(1)) * scale
            group = Decimal(0)
            last_rank = 3
            last_scale = scale
            spelled = True
            seen = True
            index += 1
            continue

        if word in HALF_WORDS:
            if seen:
                break
            group = Decimal(1)
            fraction = Decimal("0.5")
            spelled = True
            seen = True
            index += 1
            continue

        if allow_ordinal and not seen:
            ordinal = ORDINAL_WORDS.get(word)
            if ordinal is not None:
                return ParsedNumber(
                    value=Decimal(-ordinal if negative else ordinal),
                    words=index + 1,
                    is_spelled=True,
                )

        consumed = _read_fraction_tail(words, index, seen=seen)
        if consumed is not None:
            fraction, index = consumed
            spelled = True
            seen = True
            break

        # «три четверти», «две трети» — a bare fraction word straight after a
        # small whole number multiplies it instead of adding to it. The additive
        # form always has a joiner («два с половиной»), so the absence of one is
        # what tells them apart. Capped below ten because «двадцать четверти» is
        # not Russian, and the cap keeps a misheard pause from turning a count
        # into a fraction of itself.
        if seen and total == 0 and fraction == 0 and Decimal(1) < group < 10:
            share = FRACTION_WORDS.get(word)
            if share is not None and share < 1:
                group *= share
                spelled = True
                index += 1
            break

        break

    if not seen:
        return None

    value = total + group + fraction
    return ParsedNumber(
        value=-value if negative else value,
        words=index,
        is_spelled=spelled,
    )


def _read_fraction_tail(words: list[str], index: int, *, seen: bool) -> tuple[Decimal, int] | None:
    """Read a fractional tail at ``index``: «с половиной», «и пять», «пол».

    Returns the fraction and the index after it, or ``None`` when the words at
    that position are not one. Three shapes, all of them things people say:
    ``с половиной`` after a whole number, a joiner plus a short number
    (``двадцать пять и пять``), and a bare fraction word standing on its own
    (``пол``, ``четверть``).
    """
    word = words[index]

    if seen and word == "с" and index + 1 < len(words):
        value = FRACTION_WORDS.get(words[index + 1])
        if value is not None:
            return value, index + 2

    if seen and word in _FRACTION_JOINERS and index + 1 < len(words):
        tail = parse_number(" ".join(words[index + 1 :]))
        # Only a value below one reads as a fraction: «пять и десять» is two
        # numbers, «пять и пять» is 5.5. The digits after the joiner are read
        # as tenths, hundredths and so on, the way the phrase is spoken.
        if tail is not None and tail.value > 0:
            scaled = tail.value / (10 ** len(str(int(tail.value))))
            return scaled, index + 1 + tail.words

    if not seen:
        value = FRACTION_WORDS.get(word)
        if value is not None:
            return value, index + 1

    return None


def parse_ordinal(text: str) -> int | None:
    """Read an ordinal — «второй», «третьего» — as its number.

    Digits are accepted too, so a trigger written as ``{index:int}`` still works
    when the user says «два» and normalisation has already rewritten it.
    """
    words = tokenize(text)
    if not words:
        return None
    ordinal = ORDINAL_WORDS.get(words[0])
    if ordinal is not None:
        return ordinal
    digits = _digits_value(words[0])
    if digits is not None and digits == digits.to_integral_value():
        return int(digits)
    return None


def parse_percent(text: str) -> Percent | None:
    """Read «на десять процентов», «10%», «двадцать процентов».

    ``relative`` comes out ``True`` for the ``на`` form, which is the whole
    reason this returns a type instead of a number: «на 10 процентов тише» asks
    for a step and «громкость 10 процентов» asks for a level, and only the
    preposition distinguishes them.
    """
    words = tokenize(text.replace("%", " % "))
    if not words:
        return None

    relative = False
    start = 0
    if words[0] == "на":
        relative = True
        start = 1
    if start >= len(words):
        return None

    number = parse_number(" ".join(words[start:]))
    if number is None:
        return None

    tail = words[start + number.words :]
    if not tail:
        return None
    unit = tail[0]
    if unit != "%" and not unit.startswith(_PERCENT_STEM):
        return None
    return Percent(value=number.value, relative=relative)


def plural_form(count: int | Decimal, one: str, few: str, many: str) -> str:
    """Pick the Russian plural form for ``count``.

    «1 минута», «2 минуты», «5 минут», and — the part that catches every naive
    implementation — «11 минут» rather than «11 минута», because the teens all
    take the ``many`` form regardless of their last digit.

    A fractional count takes ``few`` whatever its integer part: «полторы минуты»,
    «две с половиной минуты», «0.5 минуты». Truncating to the integer first would
    say «полторы минута», and it is the fraction rather than the whole that the
    noun agrees with.
    """
    if count != int(count):
        return few
    number = abs(int(count))
    if number % 100 // 10 == 1:
        return many
    last = number % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def _under_thousand(value: int, *, feminine: bool) -> list[str]:
    """Spell a number in ``1..999`` as words."""
    words: list[str] = []
    hundreds, rest = divmod(value, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
        return words
    tens, ones = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if ones:
        words.append((_ONES_FEMININE if feminine else _ONES_MASCULINE)[ones])
    return words


def number_to_words(value: int | Decimal | float, *, feminine: bool = False) -> str:
    """Render a number in Russian words, for TTS to read aloud.

    The reverse of :func:`parse_number`, and the reason it exists: an assistant
    that answers «громкость установлена на 40» has the recogniser spell digits
    back at the user, which sounds wrong in Russian.

    Args:
        value: The number. A fractional value is read as «целых» and the
            fraction spelled out, which is how the numbers this application
            produces — a volume, a percentage, a count of minutes — are said.
        feminine: Whether the unit that follows is feminine, so «одна минута»
            and «две минуты» come out right. Scale words carry their own gender:
            «тысяча» is feminine no matter what it counts.

    Returns:
        The number as words, lowercase, without the unit.
    """
    number = Decimal(str(value)) if isinstance(value, float) else Decimal(value)
    words: list[str] = []
    if number < 0:
        words.append("минус")
        number = -number

    whole = int(number.to_integral_value(rounding="ROUND_DOWN"))
    fraction = number - whole

    if whole == 0 and not fraction:
        return " ".join([*words, "ноль"])

    remaining = whole
    for scale, forms, scale_feminine in _SCALE_FORMS:
        count, remaining = divmod(remaining, scale)
        if not count:
            continue
        words.extend(_under_thousand(count, feminine=scale_feminine))
        words.append(plural_form(count, *forms))

    if remaining:
        # A fraction makes the whole part agree with «целая», which is feminine
        # regardless of the unit: «одна целая пять десятых секунды».
        words.extend(_under_thousand(remaining, feminine=feminine or bool(fraction)))
    elif not words:
        # «ноль целых пять десятых»: the whole part is spoken even when it is
        # zero, otherwise the fraction has nothing to hang off.
        words.append("ноль")

    if fraction:
        digits = format(fraction.normalize(), "f").partition(".")[2]
        tenths = int(digits)
        scale_forms = _FRACTION_SCALES.get(len(digits))
        words.append(plural_form(whole, "целая", "целых", "целых"))
        words.extend(_under_thousand(tenths, feminine=True))
        if scale_forms is not None:
            words.append(plural_form(tenths, *scale_forms))

    return " ".join(words)


#: Names of the decimal places, by how many digits the fraction has. Beyond
#: three the number is no longer something a voice assistant reads aloud, and a
#: fraction that long is spelled digit by digit instead.
_FRACTION_SCALES: Final[dict[int, tuple[str, str, str]]] = {
    1: ("десятая", "десятых", "десятых"),
    2: ("сотая", "сотых", "сотых"),
    3: ("тысячная", "тысячных", "тысячных"),
}
