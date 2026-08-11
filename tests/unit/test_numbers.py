"""Task 16: Russian numerals, in both directions.

Numbers are where a Russian voice assistant either works or does not, and the
reason is grammar rather than arithmetic. «пятьдесят» is one word, «сорок пять»
is two, «две тысячи двадцать пять» is four and composes by multiplication in the
middle; and every one of them arrives in whatever case the sentence around it
demanded — «двух минут», «на пятидесяти процентах». A parser that only knows the
nominative fails on most of what people actually say.

Two boundaries are asserted hard, because getting either wrong is silent.

*Composition must stop.* «три четыре» is two numbers, not seven; «пятнадцать
пять» is not twenty. Russian builds a number from parts of strictly decreasing
magnitude, and the moment the magnitude stops decreasing the number has ended.
Without that rule a recogniser's stutter turns into arithmetic.

*A fraction is not an integer.* «полтора» is 1.5 and :attr:`ParsedNumber.as_int`
has to refuse it rather than round, because the caller asking for an ``int`` is
about to set a volume or an index and 1.5 hours truncated to 1 is a wrong alarm
rather than an error.

Everything works in :class:`~decimal.Decimal`. Binary floats cannot hold 0.1, and
a user who said «на 0.1 громче» twice would drift; the values here are the user's
own numbers, so they are kept exactly as spoken.

Groups:

* :class:`TestTokenize` — folding, splitting, and the «пол-» prefix.
* :class:`TestCardinals` — units, teens, tens, hundreds, scales, cases.
* :class:`TestComposition` — where a number ends and the next one starts.
* :class:`TestFractions` — «полтора», «с половиной», decimal separators.
* :class:`TestOrdinals` — «второй», «третьего», and their genders.
* :class:`TestPercent` — «на десять процентов», «10%», relative forms.
* :class:`TestPlural` — the three-way plural agreement.
* :class:`TestToWords` — rendering back out, for TTS to read aloud.
* :class:`TestRoundTrip` — words → number → words over a wide range.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from ayris.nlu.numbers import (
    ParsedNumber,
    number_to_words,
    parse_number,
    parse_ordinal,
    parse_percent,
    plural_form,
    tokenize,
)


class TestTokenize:
    """Folding and splitting, including the one prefix that is a word."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("пять минут", ["пять", "минут"]),
            ("Пять  Минут", ["пять", "минут"]),
            ("трёх часов", ["трех", "часов"]),
            ("", []),
            ("   ", []),
            # Punctuation stays attached: «50%» and «9:30» are single tokens that
            # the percent and clock readers take apart themselves. Splitting them
            # here would leave a bare «%» for every caller to step over.
            ("50%", ["50%"]),
            ("9:30", ["9:30"]),
        ],
    )
    def test_splits_and_folds(self, text: str, expected: list[str]) -> None:
        assert tokenize(text) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("полтора", ["полтора"]),
            ("полчаса", ["пол", "часа"]),
            ("полминуты", ["пол", "минуты"]),
            ("полвторого", ["пол", "второго"]),
            ("пол седьмого", ["пол", "седьмого"]),
        ],
    )
    def test_half_prefix_is_separated(self, text: str, expected: list[str]) -> None:
        """«полчаса» is «пол» + «часа», but «полтора» is one word and stays whole."""
        assert tokenize(text) == expected


class TestCardinals:
    """Every magnitude and the cases that survive dictation."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("ноль", 0),
            ("один", 1),
            ("одна", 1),
            ("одну", 1),
            ("два", 2),
            ("две", 2),
            ("двух", 2),
            ("три", 3),
            ("трех", 3),
            ("трёх", 3),
            ("четыре", 4),
            ("пять", 5),
            ("пяти", 5),
            ("шесть", 6),
            ("семь", 7),
            ("восемь", 8),
            ("девять", 9),
            ("десять", 10),
            ("одиннадцать", 11),
            ("двенадцать", 12),
            ("пятнадцать", 15),
            ("девятнадцать", 19),
            ("двадцать", 20),
            ("тридцать", 30),
            ("сорок", 40),
            ("пятьдесят", 50),
            ("шестьдесят", 60),
            ("семьдесят", 70),
            ("восемьдесят", 80),
            ("девяносто", 90),
            ("сто", 100),
            ("двести", 200),
            ("пятьсот", 500),
            ("девятьсот", 900),
        ],
    )
    def test_single_words(self, text: str, expected: int) -> None:
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected
        assert parsed.is_spelled is True

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("сорок пять", 45),
            ("двадцать один", 21),
            ("сто двадцать три", 123),
            ("сто", 100),
            ("двести пятьдесят", 250),
            ("девятьсот девяносто девять", 999),
            ("тысяча", 1_000),
            ("две тысячи", 2_000),
            ("две тысячи двадцать пять", 2_025),
            ("сто тысяч", 100_000),
            ("миллион", 1_000_000),
            ("два миллиона пятьсот тысяч", 2_500_000),
        ],
    )
    def test_composed(self, text: str, expected: int) -> None:
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("пятидесяти", 50),
            ("двадцати", 20),
            ("девяноста", 90),
            ("трехсот", 300),
            ("восьмидесяти", 80),
        ],
    )
    def test_oblique_cases(self, text: str, expected: int) -> None:
        """«на пятидесяти процентах» has to read as fifty, not as nothing."""
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("50", 50),
            ("0", 0),
            ("100", 100),
            ("2025", 2_025),
            ("-5", -5),
        ],
    )
    def test_digits(self, text: str, expected: int) -> None:
        """Task 15 normalisation may already have turned words into digits."""
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected
        assert parsed.is_spelled is False

    @pytest.mark.parametrize("text", ["", "   ", "бубубу", "громче", "и", "%"])
    def test_not_a_number(self, text: str) -> None:
        assert parse_number(text) is None

    def test_word_count_is_reported(self) -> None:
        """The caller has to know how much of the phrase the number ate."""
        parsed = parse_number("сто двадцать три градуса")
        assert parsed is not None
        assert parsed.as_int == 123
        assert parsed.words == 3


class TestComposition:
    """Where one number ends and the next one begins."""

    @pytest.mark.parametrize(
        ("text", "expected", "words"),
        [
            ("три четыре", 3, 1),
            ("пятнадцать пять", 15, 1),
            ("двадцать двадцать", 20, 1),
            ("сто сто", 100, 1),
            ("пять пятьдесят", 5, 1),
        ],
    )
    def test_stops_when_magnitude_stops_decreasing(
        self, text: str, expected: int, words: int
    ) -> None:
        """Two numbers in a row are two numbers. «три четыре» is not seven."""
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected
        assert parsed.words == words

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("сорок пять", 45),
            ("сто пятьдесят шесть", 156),
            ("девяносто девять", 99),
        ],
    )
    def test_composes_while_it_decreases(self, text: str, expected: int) -> None:
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.as_int == expected

    def test_number_must_start_the_text(self) -> None:
        """This reads a number *at the start*; finding one anywhere is another job."""
        assert parse_number("громкость пятьдесят") is None


class TestFractions:
    """The half of the parser that cannot return an ``int``."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("полтора", Decimal("1.5")),
            ("полторы", Decimal("1.5")),
            ("половина", Decimal("0.5")),
            ("половину", Decimal("0.5")),
            ("две с половиной", Decimal("2.5")),
            ("пять с половиной", Decimal("5.5")),
            ("четверть", Decimal("0.25")),
            ("три четверти", Decimal("0.75")),
        ],
    )
    def test_spelled_fractions(self, text: str, expected: Decimal) -> None:
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.value == expected

    def test_thirds(self) -> None:
        """A third has no exact decimal form, so it is compared as a ratio."""
        parsed = parse_number("две трети")
        assert parsed is not None
        assert parsed.value * 3 == pytest.approx(Decimal(2))

    def test_a_joiner_adds_and_its_absence_multiplies(self) -> None:
        """«два с половиной» is 2.5; «три четверти» is 0.75, not 3.25."""
        additive = parse_number("два с половиной")
        multiplicative = parse_number("три четверти")
        assert additive is not None
        assert multiplicative is not None
        assert additive.value == Decimal("2.5")
        assert multiplicative.value == Decimal("0.75")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("1.5", Decimal("1.5")),
            ("1,5", Decimal("1.5")),
            ("0.25", Decimal("0.25")),
            ("2.75", Decimal("2.75")),
        ],
    )
    def test_written_fractions(self, text: str, expected: Decimal) -> None:
        """A comma is a decimal separator in Russian and a recogniser may emit it."""
        parsed = parse_number(text)
        assert parsed is not None
        assert parsed.value == expected

    def test_as_int_refuses_a_fraction(self) -> None:
        """Truncating 1.5 hours to 1 is a wrong alarm, not a rounding error."""
        parsed = parse_number("полтора")
        assert parsed is not None
        assert parsed.as_int is None
        assert parsed.as_float == pytest.approx(1.5)

    def test_exact_decimal_arithmetic(self) -> None:
        """The value is the user's own number, so it is kept exactly as spoken."""
        parsed = parse_number("0.1")
        assert parsed is not None
        assert parsed.value * 3 == Decimal("0.3")


class TestOrdinals:
    """«полвторого» needs these, and so does «третий пункт»."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("первый", 1),
            ("первая", 1),
            ("первого", 1),
            ("второй", 2),
            ("второго", 2),
            ("третий", 3),
            ("третьего", 3),
            ("четвертый", 4),
            ("пятый", 5),
            ("седьмого", 7),
            ("двенадцатого", 12),
        ],
    )
    def test_ordinals(self, text: str, expected: int) -> None:
        assert parse_ordinal(text) == expected

    @pytest.mark.parametrize("text", ["пять", "бубубу", "", "полтора"])
    def test_not_an_ordinal(self, text: str) -> None:
        assert parse_ordinal(text) is None

    def test_cardinal_parser_ignores_ordinals_by_default(self) -> None:
        """«второго» is not a quantity, and reading it as 2 would misfill a count."""
        assert parse_number("второго") is None
        allowed = parse_number("второго", allow_ordinal=True)
        assert allowed is not None
        assert allowed.as_int == 2


class TestPercent:
    """The form «на N процентов», and the relative reading of it."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("десять процентов", Decimal(10)),
            ("10 процентов", Decimal(10)),
            ("10%", Decimal(10)),
            ("пятьдесят процентов", Decimal(50)),
            ("сто процентов", Decimal(100)),
            ("двадцать пять процентов", Decimal(25)),
            ("полтора процента", Decimal("1.5")),
        ],
    )
    def test_absolute(self, text: str, expected: Decimal) -> None:
        percent = parse_percent(text)
        assert percent is not None
        assert percent.value == expected
        assert percent.relative is False

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("на десять процентов", Decimal(10)),
            ("на 10%", Decimal(10)),
            ("на 30 процентов", Decimal(30)),
        ],
    )
    def test_relative(self, text: str, expected: Decimal) -> None:
        """«на» makes it a change to what it is now, not the value it becomes."""
        percent = parse_percent(text)
        assert percent is not None
        assert percent.value == expected
        assert percent.relative is True

    @pytest.mark.parametrize("text", ["", "громче", "пятьдесят", "процентов", "50"])
    def test_not_a_percentage(self, text: str) -> None:
        """A bare number is not a percentage — the unit is the whole point."""
        assert parse_percent(text) is None

    def test_as_float(self) -> None:
        percent = parse_percent("на 12.5%")
        assert percent is not None
        assert percent.as_float == pytest.approx(12.5)


class TestPlural:
    """The three-way agreement, which every spoken answer needs."""

    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (1, "минута"),
            (2, "минуты"),
            (3, "минуты"),
            (4, "минуты"),
            (5, "минут"),
            (10, "минут"),
            (11, "минут"),
            (12, "минут"),
            (14, "минут"),
            (15, "минут"),
            (21, "минута"),
            (22, "минуты"),
            (25, "минут"),
            (100, "минут"),
            (101, "минута"),
            (111, "минут"),
            (0, "минут"),
        ],
    )
    def test_forms(self, count: int, expected: str) -> None:
        assert plural_form(count, "минута", "минуты", "минут") == expected

    def test_fraction_takes_the_few_form(self) -> None:
        """«полторы минуты», not «полторы минута»."""
        assert plural_form(Decimal("1.5"), "минута", "минуты", "минут") == "минуты"


class TestToWords:
    """Rendering out, for a TTS engine that must not read «1» as «один» wrongly."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "ноль"),
            (1, "один"),
            (2, "два"),
            (11, "одиннадцать"),
            (21, "двадцать один"),
            (45, "сорок пять"),
            (100, "сто"),
            (123, "сто двадцать три"),
            (1_000, "одна тысяча"),
            (2_025, "две тысячи двадцать пять"),
            (1_000_000, "один миллион"),
        ],
    )
    def test_integers(self, value: int, expected: str) -> None:
        assert number_to_words(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1, "одна"),
            (2, "две"),
            (21, "двадцать одна"),
            (22, "двадцать две"),
        ],
    )
    def test_feminine_agreement(self, value: int, expected: str) -> None:
        """«две минуты», not «два минуты» — the noun decides, so the caller says."""
        assert number_to_words(value, feminine=True) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (Decimal("1.5"), "одна целая пять десятых"),
            (Decimal("0.5"), "ноль целых пять десятых"),
        ],
    )
    def test_fractions(self, value: Decimal, expected: str) -> None:
        assert number_to_words(value) == expected

    def test_negative(self) -> None:
        assert number_to_words(-5) == "минус пять"


class TestRoundTrip:
    """Rendering out and reading back in has to be the identity."""

    @pytest.mark.parametrize(
        "value",
        [0, 1, 2, 5, 11, 15, 19, 20, 21, 45, 99, 100, 101, 123, 500, 999, 1_000, 2_025, 100_000],
    )
    def test_words_survive_a_round_trip(self, value: int) -> None:
        words = number_to_words(value)
        parsed = parse_number(words)
        assert parsed is not None, words
        assert parsed.as_int == value, words

    @pytest.mark.parametrize("value", [1, 2, 21, 22, 45])
    def test_feminine_round_trip(self, value: int) -> None:
        parsed = parse_number(number_to_words(value, feminine=True))
        assert parsed is not None
        assert parsed.as_int == value

    def test_parsed_number_is_immutable(self) -> None:
        """Everything in the NLU layer is frozen; a parse result is shared freely."""
        parsed = ParsedNumber(value=Decimal(5), words=1)
        with pytest.raises(AttributeError):
            parsed.value = Decimal(6)  # type: ignore[misc]
