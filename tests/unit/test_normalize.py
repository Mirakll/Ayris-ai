"""Task 15: text normalisation — case, ``ё``, punctuation, numerals, address.

The matcher can only be as good as the shape both sides are brought into, and
this module is that shape. The recogniser and the stored trigger go through the
same :func:`~ayris.nlu.normalize.normalize`, so a difference that survives it is
a real difference and not a comma.

Two details are load-bearing and worth spelling out. The digit rewrite keeps
both forms of a phrase: :attr:`~ayris.nlu.normalize.NormalizedText.text` with
the digits is what exact and fuzzy matching see, while
:attr:`~ayris.nlu.normalize.NormalizedText.spoken` keeps the number words so a
regex written against words is not broken. And numerals compose the way Russian
does — ``сто двадцать три`` → ``123`` — but only while magnitudes decrease, so
``три четыре`` stays ``3 4`` instead of collapsing into ``7``.

The address is stripped only when something is left behind it: the phrase
«Айрис» on its own must still match a trigger that consists of the address
itself. A filler in front (``эй``/``окей``/``слушай``) goes only together with
an address form, so «слушай музыку» stays a command.

Groups:

* :class:`TestFoldLetters` — case, ``ё``, and the variant that leaves a regex alone.
* :class:`TestAddress` — wake-word stripping, fillers, the address-only phrase.
* :class:`TestPunctuation` — what is dropped, what survives, empty phrases.
* :class:`TestNumerals` — words to digits, scale composition, non-composition.
* :class:`TestNormalizedText` — the two forms, ``has_numerals``, ``is_empty``.
* :class:`TestNormalize` — the full pipeline and its switches.
"""

from __future__ import annotations

import pytest

from ayris.nlu.normalize import (
    DEFAULT_ADDRESS_FORMS,
    FILLER_WORDS,
    NormalizedText,
    fold_letters,
    fold_yo,
    normalize,
    normalize_text,
    spoken_to_digits,
    strip_address,
)

pytestmark = pytest.mark.unit


class TestFoldLetters:
    def test_lowercase_and_yo(self) -> None:
        assert fold_letters("Ещё Раз ЕЛКА") == "еще раз елка"

    def test_yo_variants_match(self) -> None:
        assert fold_letters("ещё") == fold_letters("еще")

    def test_fold_yo_leaves_case_alone(self) -> None:
        # Lowercasing a regex pattern would turn \S into \s and (?P<app>...)
        # into (?p<app>...), which does not parse. The regex path uses this half.
        assert fold_yo(r"Ёлка \S (?P<A>x)") == r"Елка \S (?P<A>x)"


class TestAddress:
    def test_leading_address_stripped(self) -> None:
        assert normalize("Айрис, открой браузер").text == "открой браузер"

    def test_filler_before_address_stripped(self) -> None:
        assert normalize("эй, айрис, открой почту").text == "открой почту"

    def test_any_address_form(self) -> None:
        for form in DEFAULT_ADDRESS_FORMS:
            assert normalize(f"{form}, включи свет").text == "включи свет"

    def test_address_alone_survives(self) -> None:
        # «Айрис» alone has to stay matchable: a trigger consisting of the wake
        # word itself must still fire.
        assert normalize("Айрис").text == "айрис"

    def test_filler_alone_is_a_command(self) -> None:
        # A filler is dropped only together with an address form: «слушай» on
        # its own is a command.
        assert normalize("слушай музыку").text == "слушай музыку"

    def test_address_in_the_middle_survives(self) -> None:
        assert normalize("открой айрис браузер").text == "открой айрис браузер"

    def test_strip_wake_word_off(self) -> None:
        assert normalize("Айрис, открой", strip_wake_word=False).text == "айрис открой"

    def test_custom_address_forms(self) -> None:
        assert normalize("компьютер, стоп", address_forms=("компьютер",)).text == "стоп"

    def test_strip_address_is_a_noop_without_a_form(self) -> None:
        assert strip_address(("открой", "браузер")) == ("открой", "браузер")


class TestPunctuation:
    def test_punctuation_dropped(self) -> None:
        assert normalize("Открой, браузер!").text == "открой браузер"

    def test_whitespace_collapsed(self) -> None:
        assert normalize("открой    браузер").text == "открой браузер"

    def test_underscore_is_punctuation(self) -> None:
        # \w would keep the underscore, which is punctuation as far as speech
        # is concerned. «файл_один» becomes «файл 1», so a trigger stored as
        # «файл один» still matches it.
        assert normalize("файл_один").text == "файл 1"

    def test_digits_survive(self) -> None:
        assert normalize("открой файл 42").text == "открой файл 42"

    def test_pure_punctuation_is_empty(self) -> None:
        assert normalize("!!!").is_empty
        assert normalize("   ").is_empty


class TestNumerals:
    def test_simple_units(self) -> None:
        assert normalize("громкость пять").text == "громкость 5"

    def test_composition_decreasing(self) -> None:
        assert normalize("сто двадцать три").text == "123"

    def test_composition_with_scale(self) -> None:
        assert normalize("две тысячи двадцать пять").text == "2025"

    def test_scale_alone(self) -> None:
        assert normalize("тысяча").text == "1000"

    def test_scale_multiplies_group(self) -> None:
        assert normalize("миллион двести тридцать четыре").text == "1000234"

    def test_non_composition_stays_separate(self) -> None:
        # Magnitudes must keep decreasing; «три четыре» is two numbers.
        assert normalize("три четыре").text == "3 4"
        assert normalize("пятнадцать пять").text == "15 5"
        assert normalize("сто двадцать три сорок").text == "123 40"

    def test_repeated_scale_breaks(self) -> None:
        assert normalize("тысяча тысяча").text == "1000 1000"

    def test_oblique_forms(self) -> None:
        assert normalize("таймер на двадцать одну минуту").text == "таймер на 21 минуту"
        assert normalize("через пять минут").text == "через 5 минут"

    def test_numerals_off(self) -> None:
        n = normalize("сто двадцать три", numerals=False)
        assert n.text == n.spoken == "сто двадцать три"

    def test_spoken_form_keeps_words(self) -> None:
        n = normalize("поставь таймер на пять минут")
        assert n.text == "поставь таймер на 5 минут"
        assert n.spoken == "поставь таймер на пять минут"

    def test_zero_variants(self) -> None:
        assert normalize("ноль").text == "0"
        assert normalize("нуль").text == "0"

    def test_scale_word_is_a_number(self) -> None:
        # A trigger that is itself a scale word normalises to a digit, so it
        # matches «тысяча» spoken aloud.
        assert normalize("миллион").text == "1000000"

    def test_plain_words_pass_through(self) -> None:
        assert normalize("как дела").text == "как дела"


class TestNormalizedText:
    def test_has_numerals(self) -> None:
        assert normalize("громкость пять").has_numerals
        assert not normalize("громкость").has_numerals

    def test_len_is_matched_form(self) -> None:
        assert len(normalize("Айрис, открой браузер")) == len("открой браузер")

    def test_is_empty(self) -> None:
        assert normalize("").is_empty
        assert not normalize("привет").is_empty

    def test_raw_is_kept(self) -> None:
        assert normalize("Айрис, привет!").raw == "Айрис, привет!"

    def test_words(self) -> None:
        assert normalize("включи свет").words == ("включи", "свет")

    def test_constructed_directly(self) -> None:
        # The index builds one of these per trigger, so the constructor has to
        # be usable without going through ``normalize``.
        phrase = NormalizedText(raw="Пять", text="5", spoken="пять", words=("5",))
        assert phrase.has_numerals
        assert not phrase.is_empty
        assert len(phrase) == 1


class TestNormalize:
    def test_normalize_text(self) -> None:
        assert normalize_text("ЁЛКА, ёж!") == "елка еж"

    def test_filler_vocabulary(self) -> None:
        assert {"эй", "окей", "слушай"} <= FILLER_WORDS

    def test_spoken_to_digits_directly(self) -> None:
        assert spoken_to_digits(("сорок", "пять", "минут")) == ("45", "минут")
        assert spoken_to_digits(("открой", "браузер")) == ("открой", "браузер")

    def test_normalizing_twice_changes_nothing(self) -> None:
        # The index normalises trigger phrases and the matcher normalises the
        # recognised text; a phrase that came from either side must be stable.
        once = normalize("Айрис, поставь таймер на пять минут").text
        assert normalize(once).text == once

    @pytest.mark.parametrize(
        ("said", "stored"),
        [
            ("Айрис, открой браузер!", "открой браузер"),
            ("ОТКРОЙ БРАУЗЕР", "открой браузер"),
            ("Ещё раз", "еще раз"),
            ("громкость на пятьдесят процентов", "громкость на 50 процентов"),
        ],
    )
    def test_both_sides_meet(self, said: str, stored: str) -> None:
        assert normalize(said).text == normalize(stored).text
