"""Task 16: templates with holes, and what comes out of the holes.

Three ideas are worth the file, and each has its group below.

*An unparsed slot is a result, not an error.* «поставь громкость на бубубу» fits
the template — every literal word around the hole was said — and only the value
failed. So :meth:`~ayris.nlu.slots.SlotTemplate.extract` returns a
:class:`~ayris.nlu.slots.SlotSet` with ``value=None`` and ``complete`` false,
rather than ``None`` or an exception. ``None`` is reserved for «this is not this
command», which is the only case that justifies trying the next template, and the
difference is what lets a command ask a follow-up question instead of dying.

*A bad template fails when it is written, not when it is said.* A greedy
``{query}`` in the middle of a template eats every slot after it, and a stray
brace compiles into a literal that never matches. Both would otherwise become a
command that silently never fires — the worst possible bug report — so
:func:`~ayris.nlu.slots.compile_slots` refuses them and
:func:`~ayris.nlu.slots.validate_template` turns the refusal into the Russian
sentence the settings window shows.

*The registry is the plugin seam.* A type is looked up by name at compile time,
so a plugin that registers ``устройство`` before its commands compile gets
``{что:устройство}`` for free. The tests here pin the seam from both ends: the
registry itself, and a template going through :class:`~ayris.nlu.index.TriggerIndex`
with a custom registry attached.

Everything is pinned to an explicit ``now`` and to the shipped app dictionary, for
the same reason as in ``test_timeparse``: the CI runners are UTC and «в семь»
means different things on either side of noon.

Groups:

* :class:`TestVolume` — the four shapes of a volume slot, absolute and relative.
* :class:`TestRelativeValues` — «громче» applied to a current value, and clamping.
* :class:`TestPercent` — a percentage, with and without a direction.
* :class:`TestTime` — a moment, and the reading when no ``now`` was given.
* :class:`TestDuration` — «пять минут» in a slot.
* :class:`TestApp` — a program name, exact and fuzzy, and no resolver at all.
* :class:`TestSite` — spoken names and typed addresses the normaliser took apart.
* :class:`TestQuery` — the greedy slot, and what it may not sit before.
* :class:`TestPlainTypes` — ``int``, ``float`` and the ``str`` fallback.
* :class:`TestMatching` — what fits a template and what does not.
* :class:`TestSlotSetApi` — the surface a command handler actually reads.
* :class:`TestCompile` — every way a template can be refused.
* :class:`TestValidateTemplate` — the messages the settings window shows.
* :class:`TestRegistry` — registration, replacement, copies, plugin conflicts.
* :class:`TestSafeParse` — a parser that raises costs its slot and nothing more.
* :class:`TestIndexSeam` — a template trigger through the matcher and the index.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from ayris.nlu.apps import AppMatch, AppResolver, load_apps
from ayris.nlu.index import TriggerIndex
from ayris.nlu.matcher import Matcher, Trigger, TriggerKind
from ayris.nlu.slot_types import (
    MAX_VOLUME,
    MIN_VOLUME,
    BuiltinSlotType,
    Direction,
    DurationType,
    IntType,
    RelativeUnit,
    RelativeValue,
    SlotContext,
    SlotType,
    SlotTypeRegistry,
    default_registry,
)
from ayris.nlu.slots import (
    MAX_SLOTS,
    Slot,
    SlotSet,
    SlotTemplateError,
    compile_slots,
    extract_slots,
    template_slot_names,
    validate_template,
)
from ayris.nlu.timeparse import ClockTime
from ayris.utils.logger import ROOT_LOGGER_NAME

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: Moscow, and midday. Same reasoning as ``test_timeparse``: no DST, and both
#: readings of a bare hour available so the resolver's rules are exercised. As a
#: plain offset, not a named zone — Windows has no system tz database.
MSK = timezone(timedelta(hours=3))
NOON = datetime(2026, 8, 11, 12, 0, tzinfo=MSK)

#: The shipped dictionary, read once. Every ``{app}`` test resolves against the
#: real thing rather than a hand-made entry, so a phrase that stops working
#: because an alias moved shows up here.
APPS = AppResolver.from_apps(load_apps().apps)

#: What a running Ayris passes down: a clock to resolve against and a dictionary
#: to resolve names in.
CTX = SlotContext(now=NOON, apps=APPS)


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` on that logger and ``caplog``
    listens on the interpreter root, so a plain ``caplog.at_level`` sees nothing
    once any earlier test in the run has configured logging.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


def fill(template: str, text: str, context: SlotContext | None = CTX) -> SlotSet:
    """Extract slots and insist the phrase fitted the template.

    Most tests are about the *value*, and a ``None`` from a template that did not
    match would otherwise surface as an attribute error three lines later.
    """
    result = extract_slots(template, text, context=context)
    assert result is not None, f"фраза {text!r} не подошла к шаблону {template!r}"
    return result


class TestVolume:
    """«поставь громкость на …» — the checklist's four shapes, and the refusals."""

    @pytest.mark.parametrize(
        ("phrase", "value"),
        [
            ("поставь громкость на пятьдесят", 50),
            ("поставь громкость на 50", 50),
            ("поставь громкость на ноль", 0),
            ("поставь громкость на сто", 100),
            ("поставь громкость на семьдесят пять", 75),
            ("поставь громкость на двадцать", 20),
        ],
    )
    def test_absolute(self, phrase: str, value: int) -> None:
        slots = fill("поставь громкость на {volume}", phrase)
        assert slots.value("volume") == value
        assert slots.complete

    @pytest.mark.parametrize(
        ("phrase", "direction", "amount", "unit"),
        [
            ("поставь громкость на громче", Direction.UP, None, RelativeUnit.STEP),
            ("поставь громкость на тише", Direction.DOWN, None, RelativeUnit.STEP),
            ("поставь громкость на погромче", Direction.UP, None, RelativeUnit.STEP),
            (
                "поставь громкость на 10 процентов тише",
                Direction.DOWN,
                Decimal("10"),
                RelativeUnit.PERCENT,
            ),
            (
                "поставь громкость на десять процентов громче",
                Direction.UP,
                Decimal("10"),
                RelativeUnit.PERCENT,
            ),
            (
                "поставь громкость на пять громче",
                Direction.UP,
                Decimal("5"),
                RelativeUnit.ABSOLUTE,
            ),
        ],
    )
    def test_relative(
        self,
        phrase: str,
        direction: Direction,
        amount: Decimal | None,
        unit: RelativeUnit,
    ) -> None:
        """A direction word makes the slot a change rather than a value."""
        value = fill("поставь громкость на {volume}", phrase).value("volume")
        assert value == RelativeValue(direction, amount, unit)

    @pytest.mark.parametrize("phrase", ["на 300", "на бубубу", "на минус десять", "на много"])
    def test_out_of_range_and_nonsense(self, phrase: str) -> None:
        """The template matched, the value did not — an incomplete set, not ``None``."""
        slots = fill("поставь громкость {volume}", f"поставь громкость {phrase}")
        assert slots.value("volume") is None
        assert not slots.complete
        assert slots.unparsed == ("volume",)

    def test_raw_survives_a_failure(self) -> None:
        """A command that cannot proceed still has to say what it heard."""
        slots = fill("поставь громкость на {volume}", "поставь громкость на бубубу")
        assert slots.raw("volume") == "бубубу"

    def test_the_raw_text_is_the_normalised_one(self) -> None:
        """A numeral is folded to digits before the slot ever sees it."""
        slots = fill("поставь громкость на {volume}", "Поставь громкость на пятьдесят!")
        assert slots.raw("volume") == "50"


class TestRelativeValues:
    """Turning «громче» into a number, which only the caller can do."""

    @pytest.mark.parametrize(
        ("value", "current", "expected"),
        [
            (RelativeValue(Direction.UP), 50, 60),
            (RelativeValue(Direction.DOWN), 50, 40),
            (RelativeValue(Direction.DOWN, Decimal("10"), RelativeUnit.PERCENT), 20, 10),
            (RelativeValue(Direction.UP, Decimal("15"), RelativeUnit.ABSOLUTE), 20, 35),
        ],
    )
    def test_resolve(self, value: RelativeValue, current: int, expected: int) -> None:
        assert value.resolve(current) == expected

    @pytest.mark.parametrize(
        ("value", "current", "expected"),
        [
            (RelativeValue(Direction.UP), 95, MAX_VOLUME),
            (RelativeValue(Direction.DOWN), 5, MIN_VOLUME),
            (RelativeValue(Direction.UP, Decimal("80"), RelativeUnit.PERCENT), 50, MAX_VOLUME),
        ],
    )
    def test_clamped(self, value: RelativeValue, current: int, expected: int) -> None:
        """«громче» at 95 is «as loud as it goes», not an error to refuse."""
        assert value.resolve(current) == expected

    def test_percent_counts_against_the_scale(self) -> None:
        """«на 10% тише» twice from 20 reaches zero, not 16.2 — a slider, not a ratio."""
        step = RelativeValue(Direction.DOWN, Decimal("10"), RelativeUnit.PERCENT)
        assert step.resolve(step.resolve(20)) == 0

    def test_step_is_the_callers_business(self) -> None:
        assert RelativeValue(Direction.UP).resolve(50, step=25) == 75

    def test_bounds_are_arguments(self) -> None:
        """Brightness shares the class and does not have to run 0..100."""
        assert RelativeValue(Direction.UP).resolve(250, minimum=0, maximum=255) == 255

    def test_sign(self) -> None:
        assert RelativeValue(Direction.UP).sign == 1
        assert RelativeValue(Direction.DOWN).sign == -1


class TestPercent:
    """A percentage is a value until a direction word turns it into a change."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("яркость на тридцать процентов", Decimal("30")),
            ("яркость на 30%", Decimal("30")),
            ("яркость на сто процентов", Decimal("100")),
        ],
    )
    def test_absolute(self, phrase: str, expected: Decimal) -> None:
        assert fill("яркость на {percent}", phrase).value("percent") == expected

    def test_relative(self) -> None:
        value = fill("яркость на {percent}", "яркость на 10 процентов тише").value("percent")
        assert value == RelativeValue(Direction.DOWN, Decimal("10"), RelativeUnit.PERCENT)

    def test_nonsense(self) -> None:
        assert fill("яркость на {percent}", "яркость на бубубу").value("percent") is None


class TestTime:
    """A moment when the context has a clock, a reading when it has not."""

    @pytest.mark.parametrize(
        ("template", "phrase", "expected"),
        [
            ("разбуди в {time}", "разбуди в семь", datetime(2026, 8, 11, 19, 0, tzinfo=MSK)),
            (
                "разбуди в {time}",
                "разбуди в полвторого",
                datetime(2026, 8, 11, 13, 30, tzinfo=MSK),
            ),
            ("разбуди {time}", "разбуди в 19:30", datetime(2026, 8, 11, 19, 30, tzinfo=MSK)),
            (
                "разбуди {time}",
                "разбуди завтра в девять",
                datetime(2026, 8, 12, 9, 0, tzinfo=MSK),
            ),
            (
                "разбуди {time}",
                "разбуди в семь утра",
                datetime(2026, 8, 12, 7, 0, tzinfo=MSK),
            ),
        ],
    )
    def test_moment(self, template: str, phrase: str, expected: datetime) -> None:
        assert fill(template, phrase).value("time") == expected

    def test_a_bare_hour_is_accepted_inside_a_slot(self) -> None:
        """The preposition lives in the template's literal, outside the brace.

        This is why :class:`~ayris.nlu.slot_types.TimeType` passes ``expected=True``:
        the obvious way to write the template hands the parser a bare «семь», and
        refusing it would make the obvious way the broken one.
        """
        assert fill("разбуди в {time}", "разбуди в семь").value("time") is not None

    def test_without_a_clock_the_reading_comes_out(self) -> None:
        """A settings field storing «когда не беспокоить» wants no date at all."""
        value = fill("разбуди в {time}", "разбуди в семь", context=SlotContext()).value("time")
        assert value == ClockTime(hour=7, words=1)

    def test_nonsense(self) -> None:
        assert fill("разбуди в {time}", "разбуди в бубубу").value("time") is None


class TestDuration:
    """A length of time, which a timer needs and a reminder does not."""

    @pytest.mark.parametrize(
        ("phrase", "seconds"),
        [
            ("таймер на пять минут", 300),
            ("таймер на 5 минут", 300),
            ("таймер на полтора часа", 5_400),
            ("таймер на полчаса", 1_800),
            ("таймер на две минуты тридцать секунд", 150),
        ],
    )
    def test_lengths(self, phrase: str, seconds: int) -> None:
        assert fill("таймер на {duration}", phrase).value("duration") == timedelta(seconds=seconds)

    def test_out_of_range(self) -> None:
        """The range is a property of the type, so it fails identically everywhere."""
        registry = default_registry()
        registry.register("короткий", DurationType(maximum=3_600))
        slots = extract_slots("таймер на {d:короткий}", "таймер на два часа", registry=registry)
        assert slots is not None
        assert slots.value("d") is None

    def test_nonsense(self) -> None:
        assert fill("таймер на {duration}", "таймер на бубубу").value("duration") is None


class TestApp:
    """A program name, and the confidence that came with it."""

    @pytest.mark.parametrize(
        ("phrase", "app_id"),
        [
            ("открой гугл хром", "chrome"),
            ("открой хром", "chrome"),
            ("открой телеграм", "telegram"),
            ("открой блокнот", "notepad"),
            ("открой яндекс браузер", "yandex-browser"),
            ("открой фотошоп", "photoshop"),
        ],
    )
    def test_exact(self, phrase: str, app_id: str) -> None:
        value = fill("открой {app}", phrase).value("app")
        assert isinstance(value, AppMatch)
        assert value.app_id == app_id

    def test_confidence_reaches_the_slot(self) -> None:
        """A fuzzy match must not read like an exact one further down the pipeline."""
        exact = fill("открой {app}", "открой хром")
        fuzzy = fill("открой {app}", "открой хрм")
        assert exact.confidence == 1.0
        assert 0.75 <= fuzzy.confidence < 1.0
        assert fuzzy.value("app") is not None

    def test_below_the_floor_is_unparsed(self) -> None:
        """A command that launches programs must not fire on a name nobody knows."""
        slots = fill("открой {app}", "открой абракадабру")
        assert slots.value("app") is None
        assert slots.unparsed == ("app",)

    def test_without_a_resolver_nothing_resolves(self) -> None:
        slots = fill("открой {app}", "открой хром", context=SlotContext(now=NOON))
        assert slots.value("app") is None


class TestSite:
    """Spoken names, and typed addresses the normaliser took apart."""

    @pytest.mark.parametrize(
        ("phrase", "url"),
        [
            ("открой ютуб", "https://youtube.com"),
            ("открой вк", "https://vk.com"),
            ("открой на ютубе", "https://youtube.com"),
            ("открой википедию", "https://ru.wikipedia.org"),
        ],
    )
    def test_spoken_names(self, phrase: str, url: str) -> None:
        assert fill("открой {site}", phrase).value("site") == url

    @pytest.mark.parametrize(
        ("phrase", "url"),
        [
            ("открой youtube.com", "https://youtube.com"),
            ("открой ya.ru", "https://ya.ru"),
            ("открой docs.google.com", "https://docs.google.com"),
            ("открой https://habr.com/ru", "https://habr.com/ru"),
            ("открой президент.рф", "https://президент.рф"),
        ],
    )
    def test_typed_addresses(self, phrase: str, url: str) -> None:
        """The dot is gone by the time a slot sees it — normalisation ate it.

        Without putting the address back together, the only sites that would work
        are the handful named by word, and «открой youtube.com» — the most literal
        thing a user can say — would quietly do nothing.
        """
        assert fill("открой {site}", phrase).value("site") == url

    @pytest.mark.parametrize("phrase", ["открой погоду", "открой два часа", "открой мой e-mail"])
    def test_not_an_address(self, phrase: str) -> None:
        """«погода» is a search. Opening a browser at ``погода`` helps nobody."""
        assert fill("открой {site}", phrase).value("site") is None


class TestQuery:
    """The greedy slot: everything to the end of the phrase, and nothing after it."""

    def test_takes_the_rest(self) -> None:
        slots = fill("найди {query}", "найди как приготовить плов")
        assert slots.value("query") == "как приготовить плов"

    def test_after_another_slot(self) -> None:
        slots = fill("напиши {app} сообщение {query}", "напиши телеграм сообщение привет как дела")
        value = slots.value("app")
        assert isinstance(value, AppMatch)
        assert value.app_id == "telegram"
        assert slots.value("query") == "привет как дела"

    def test_cannot_precede_a_slot(self) -> None:
        """Caught at compile time, because at match time it half-works."""
        with pytest.raises(SlotTemplateError):
            compile_slots("найди {query} в {site}")


class TestPlainTypes:
    """``int``, ``float``, and a slot whose name is not a type at all."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [("шаг двадцать", 20), ("шаг 20", 20), ("шаг минус пять", -5)],
    )
    def test_int(self, phrase: str, expected: int) -> None:
        assert fill("шаг {n:int}", phrase).value("n") == expected

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [("вес полтора", 1.5), ("вес две с половиной", 2.5), ("вес 5", 5.0)],
    )
    def test_float(self, phrase: str, expected: float) -> None:
        """Spoken fractions only: a typed «2.5» loses its dot to the normaliser."""
        assert fill("вес {n:float}", phrase).value("n") == pytest.approx(expected)

    def test_int_refuses_a_fraction(self) -> None:
        assert fill("шаг {n:int}", "шаг полтора").value("n") is None

    def test_an_unknown_name_falls_back_to_text(self) -> None:
        """«включи {что}» is a reasonable thing to write, and text is what it means."""
        slots = fill("включи {что}", "включи любимый плейлист")
        assert slots.value("что") == "любимый плейлист"
        assert slots["что"].type == BuiltinSlotType.STR

    def test_ranged_type(self) -> None:
        registry = default_registry()
        registry.register("процент", IntType(minimum=0, maximum=100))
        assert extract_slots("на {p:процент}", "на 50", registry=registry) is not None
        slots = extract_slots("на {p:процент}", "на 500", registry=registry)
        assert slots is not None
        assert slots.value("p") is None


class TestMatching:
    """Whether a phrase fits the template at all — the ``None`` case."""

    @pytest.mark.parametrize(
        "phrase",
        [
            "закрой хром",
            "открой",
            "просто открой хром пожалуйста",
            "",
        ],
    )
    def test_does_not_fit(self, phrase: str) -> None:
        assert extract_slots("открой {app}", phrase, context=CTX) is None

    @pytest.mark.parametrize(
        "phrase",
        [
            "открой хром",
            "Открой хром!",
            "открой   хром",
            "открой, хром",
        ],
    )
    def test_fits_however_it_was_said(self, phrase: str) -> None:
        """Punctuation, case and spacing are the recogniser's business, not a template's."""
        assert extract_slots("открой {app}", phrase, context=CTX) is not None

    def test_a_template_with_no_slots_still_matches(self) -> None:
        slots = fill("выключи звук", "выключи звук")
        assert len(slots) == 0
        assert slots.complete

    def test_a_slot_needs_something_in_it(self) -> None:
        """An empty capture would report a slot that is there and blank."""
        assert extract_slots("открой {app}", "открой ", context=CTX) is None

    def test_extract_is_the_same_as_compile_then_extract(self) -> None:
        template = compile_slots("открой {app}")
        once = template.extract("открой хром", CTX)
        again = extract_slots("открой {app}", "открой хром", context=CTX)
        assert once is not None
        assert again is not None
        assert once.as_dict() == again.as_dict()

    def test_bind_takes_groups_the_matcher_already_has(self) -> None:
        """The matcher ran the regex; matching twice is a second chance to disagree."""
        template = compile_slots("поставь громкость на {volume}")
        assert template.bind({"volume": "50"}, CTX).value("volume") == 50

    def test_bind_tolerates_a_missing_group(self) -> None:
        template = compile_slots("поставь громкость на {volume}")
        slots = template.bind({}, CTX)
        assert slots.value("volume") is None
        assert slots.raw("volume") == ""

    def test_match_returns_the_regex_match(self) -> None:
        template = compile_slots("открой {app}")
        found = template.match("Открой хром!")
        assert found is not None
        assert found["app"] == "хром"
        assert template.match("закрой хром") is None


class TestSlotSetApi:
    """The surface a command handler reads, including the empty and broken cases."""

    @pytest.fixture
    def mixed(self) -> SlotSet:
        """One slot that parsed and one that did not — the interesting shape."""
        return fill("напиши {app} сообщение {query}", "напиши бубубу сообщение привет")

    def test_as_dict(self, mixed: SlotSet) -> None:
        assert mixed.as_dict() == {"app": None, "query": "привет"}

    def test_incomplete(self, mixed: SlotSet) -> None:
        assert not mixed.complete
        assert mixed.unparsed == ("app",)

    def test_confidence_is_the_weakest_slot(self, mixed: SlotSet) -> None:
        """The minimum, not the mean: averaging lets certain slots hide a guess."""
        assert mixed.confidence == 0.0

    def test_confidence_of_nothing_is_certain(self) -> None:
        assert SlotSet().confidence == 1.0
        assert SlotSet().complete

    def test_value_default(self, mixed: SlotSet) -> None:
        assert mixed.value("app", "нет") == "нет"
        assert mixed.value("отсутствует", 7) == 7

    def test_raw_default(self, mixed: SlotSet) -> None:
        assert mixed.raw("app") == "бубубу"
        assert mixed.raw("отсутствует", "пусто") == "пусто"

    def test_get_and_getitem(self, mixed: SlotSet) -> None:
        assert mixed.get("отсутствует") is None
        assert mixed["query"].value == "привет"
        with pytest.raises(KeyError):
            mixed["отсутствует"]

    def test_contains(self, mixed: SlotSet) -> None:
        assert "app" in mixed
        assert "отсутствует" not in mixed
        assert 5 not in mixed

    def test_iteration_follows_the_template(self, mixed: SlotSet) -> None:
        assert tuple(slot.name for slot in mixed) == ("app", "query")
        assert len(mixed) == 2

    def test_str_falls_back_to_what_was_said(self, mixed: SlotSet) -> None:
        """A message about a failure has only the raw text to show."""
        assert str(mixed["app"]) == "бубубу"
        assert str(mixed["query"]) == "привет"

    def test_parsed_flag(self) -> None:
        assert Slot(name="a", type="int", raw="5", value=5).parsed
        assert not Slot(name="a", type="int", raw="бубубу").parsed


class TestCompile:
    """Every way a template can be refused, each with its own reason."""

    def test_names_keep_the_order_of_the_template(self) -> None:
        assert compile_slots("напиши {app} сообщение {query}").names == ("app", "query")

    def test_names_without_compiling(self) -> None:
        """The settings window wants this while the template is still invalid."""
        assert template_slot_names("напиши {app} сообщение {query} в {site") == ("app", "query")
        assert template_slot_names("нет слотов") == ()

    def test_empty(self) -> None:
        with pytest.raises(SlotTemplateError):
            compile_slots("   ")

    def test_unknown_type(self) -> None:
        with pytest.raises(SlotTemplateError):
            compile_slots("открой {app:браузер}")

    def test_repeated_name(self) -> None:
        with pytest.raises(SlotTemplateError):
            compile_slots("громкость {volume} и {volume}")

    @pytest.mark.parametrize("template", ["открой {app", "открой app}", "включи {1234}", "{}"])
    def test_stray_brace(self, template: str) -> None:
        """Left alone it compiles to a literal and the command never fires."""
        with pytest.raises(SlotTemplateError):
            compile_slots(template)

    def test_too_many_slots(self) -> None:
        template = " ".join(f"{{s{index}}}" for index in range(MAX_SLOTS + 1))
        with pytest.raises(SlotTemplateError):
            compile_slots(template)

    def test_the_limit_itself_is_allowed(self) -> None:
        template = " ".join(f"{{s{index}}}" for index in range(MAX_SLOTS))
        assert len(compile_slots(template).names) == MAX_SLOTS

    def test_the_template_is_stored_stripped(self) -> None:
        assert compile_slots("  открой {app}  ").template == "открой {app}"

    def test_cyrillic_slot_and_type_names(self) -> None:
        """A plugin may register «устройство», so the pattern has to be Unicode."""
        registry = default_registry()
        registry.register("устройство", lambda raw, _context: raw.upper())
        slots = extract_slots("включи {что:устройство}", "включи свет", registry=registry)
        assert slots is not None
        assert slots.value("что") == "СВЕТ"


class TestValidateTemplate:
    """The Russian sentences the command editor shows before saving."""

    def test_valid_is_empty(self) -> None:
        assert validate_template("поставь громкость на {volume}") == ""

    @pytest.mark.parametrize(
        ("template", "fragment"),
        [
            ("", "пуст"),
            ("открой {app:браузер}", "Неизвестный тип слота"),
            ("найди {query} в {site}", "никогда не заполнится"),
            ("громкость {volume} и {volume}", "дважды"),
            ("открой {app", "Фигурная скобка"),
            (" ".join(f"{{s{index}}}" for index in range(MAX_SLOTS + 1)), "больше 12 слотов"),
        ],
    )
    def test_messages(self, template: str, fragment: str) -> None:
        message = validate_template(template)
        assert fragment in message

    def test_the_unknown_type_message_lists_the_alternatives(self) -> None:
        """A typo in «volumе» with a Cyrillic е is only fixable if the list is there."""
        message = validate_template("громкость {v:volumе}")
        assert "volume" in message
        assert "query" in message

    def test_a_custom_registry_changes_the_answer(self) -> None:
        registry = default_registry()
        assert validate_template("включи {что:устройство}", registry) != ""
        registry.register("устройство", lambda raw, _context: raw)
        assert validate_template("включи {что:устройство}", registry) == ""


class TestRegistry:
    """Registration, conflicts, and the copy a plugin cannot break."""

    def test_the_builtins_are_all_there(self) -> None:
        registry = default_registry()
        assert set(registry.names()) == {str(name) for name in BuiltinSlotType}
        assert len(registry) == len(BuiltinSlotType)

    def test_names_are_sorted(self) -> None:
        """For an error message that reads like a list rather than a dict dump."""
        names = default_registry().names()
        assert list(names) == sorted(names)

    def test_default_registry_is_fresh_every_time(self) -> None:
        """A shared singleton would leak a plugin's type into every other library."""
        first = default_registry()
        first.register("устройство", lambda raw, _context: raw)
        assert "устройство" not in default_registry()

    def test_register_a_function(self) -> None:
        registry = SlotTypeRegistry()
        slot_type = registry.register("свой", lambda raw, _context: len(raw))
        assert registry.get("свой") is slot_type
        assert slot_type.parse("абвг", SlotContext()) == 4

    def test_register_an_instance_takes_the_name(self) -> None:
        """A type usually needs data, and an instance is where the data lives."""
        registry = SlotTypeRegistry()
        slot_type = registry.register("процент", IntType(minimum=0, maximum=100))
        assert slot_type.name == "процент"
        assert registry.get("процент") is slot_type

    def test_a_conflict_is_an_error(self) -> None:
        """Two plugins calling their type ``устройство`` is not a silent overwrite."""
        registry = SlotTypeRegistry()
        registry.register("устройство", lambda raw, _context: raw)
        with pytest.raises(ValueError, match="уже зарегистрирован"):
            registry.register("устройство", lambda _raw, _context: None)

    def test_replace_is_explicit(self) -> None:
        registry = SlotTypeRegistry()
        registry.register("устройство", lambda _raw, _context: "первый")
        registry.register("устройство", lambda _raw, _context: "второй", replace=True)
        slot_type = registry.get("устройство")
        assert slot_type is not None
        assert slot_type.parse("что-то", SlotContext()) == "второй"

    @pytest.mark.parametrize("name", ["", "   ", "1abc", "a-b", "два слова"])
    def test_bad_names(self, name: str) -> None:
        with pytest.raises(ValueError, match="недопустимое имя"):
            SlotTypeRegistry().register(name, lambda raw, _context: raw)

    def test_unregister(self) -> None:
        """A plugin unloading has to take its types with it."""
        registry = default_registry()
        registry.register("устройство", lambda raw, _context: raw)
        assert registry.unregister("устройство") is True
        assert registry.unregister("устройство") is False
        assert "устройство" not in registry

    def test_lookup_is_case_insensitive_and_trimmed(self) -> None:
        registry = default_registry()
        assert registry.get(" VOLUME ") is not None
        assert "VOLUME" in registry
        assert 5 not in registry

    def test_copy_is_independent(self) -> None:
        registry = default_registry()
        copy = registry.copy()
        copy.register("свой", lambda raw, _context: raw)
        assert "свой" in copy
        assert "свой" not in registry

    def test_iteration_is_sorted(self) -> None:
        assert list(default_registry()) == sorted(default_registry())

    def test_a_greedy_custom_type_obeys_the_same_rule(self) -> None:
        registry = default_registry()
        registry.register("хвост", lambda raw, _context: raw, pattern=r".+", greedy=True)
        assert validate_template("скажи {t:хвост} про {app}", registry) != ""
        assert validate_template("скажи про {app} текст {t:хвост}", registry) == ""


class TestSafeParse:
    """A parser that raises costs its own slot and nothing else.

    A plugin's parser is arbitrary code, and a traceback out of one type would
    take down a command that happened to sit next to it in the same template.
    """

    def test_the_slot_goes_unparsed(self) -> None:
        registry = default_registry()
        registry.register("бомба", _Boom())
        slots = extract_slots("тест {x:бомба} и {y}", "тест значение и второе", registry=registry)
        assert slots is not None
        assert slots.value("x") is None
        assert slots.value("y") == "второе"
        assert slots.unparsed == ("x",)

    def test_the_failure_is_logged(self, ayris_log: pytest.LogCaptureFixture) -> None:
        """Silently swallowing it would make a broken plugin undiagnosable."""
        registry = default_registry()
        registry.register("бомба", _Boom())
        extract_slots("тест {x:бомба}", "тест значение", registry=registry)
        assert any("бомба" in record.getMessage() for record in ayris_log.records)

    def test_the_base_class_has_no_parser(self) -> None:
        with pytest.raises(NotImplementedError):
            SlotType().parse("что-то", SlotContext())

    def test_empty_text_never_reaches_the_parser(self) -> None:
        assert _Boom().safe_parse("   ", SlotContext()) is None


class _Boom(SlotType):
    """A type whose parser always raises, for the tests above."""

    name = "бомба"
    #: One word, so that the slot after it in the template gets the rest. The
    #: default pattern crosses spaces and would make this test about capturing.
    pattern = r"[^\s]+"

    def parse(self, raw: str, context: SlotContext) -> object | None:
        raise RuntimeError("тестовое падение парсера")


class TestIndexSeam:
    """A template trigger through the matcher and back out as typed values.

    The two halves are deliberately separate — the matcher runs the regex, the
    index keeps the compiled template — so this is the join that has to be tested
    end to end rather than in either module alone.
    """

    def test_a_template_trigger_matches_and_binds(self) -> None:
        index = TriggerIndex()
        index.replace_all(
            [
                Trigger(
                    id=1,
                    command_id=7,
                    pattern="поставь громкость на {volume}",
                    kind=TriggerKind.TEMPLATE,
                )
            ]
        )
        found = Matcher(index).match("поставь громкость на пятьдесят")
        assert found is not None
        assert found.command_id == 7
        snapshot = index.snapshot()
        slots = snapshot.bind_slots(found, CTX)
        assert slots is not None
        assert slots.value("volume") == 50

    def test_the_matcher_hands_over_the_groups_it_captured(self) -> None:
        index = TriggerIndex()
        index.replace_all(
            [Trigger(id=1, command_id=7, pattern="открой {app}", kind=TriggerKind.TEMPLATE)]
        )
        found = Matcher(index).match("открой гугл хром")
        assert found is not None
        assert found.raw_groups == {"app": "гугл хром"}

    def test_a_phrase_trigger_has_no_slots(self) -> None:
        index = TriggerIndex()
        index.replace_all([Trigger(id=1, command_id=7, pattern="выключи звук")])
        found = Matcher(index).match("выключи звук")
        assert found is not None
        assert index.snapshot().slot_template(1) is None
        assert index.snapshot().bind_slots(found) is None

    def test_a_bad_template_costs_its_trigger_and_no_more(self) -> None:
        """The index never raises on a template — it drops it and logs."""
        index = TriggerIndex()
        index.replace_all(
            [
                Trigger(
                    id=1, command_id=7, pattern="найди {query} в {site}", kind=TriggerKind.TEMPLATE
                ),
                Trigger(id=2, command_id=7, pattern="открой {app}", kind=TriggerKind.TEMPLATE),
            ]
        )
        snapshot = index.snapshot()
        assert snapshot.slot_template(1) is None
        assert snapshot.slot_template(2) is not None

    def test_a_custom_registry_reaches_the_index(self) -> None:
        """The reason the registry is a constructor argument and not a global."""
        registry = default_registry()
        registry.register("устройство", lambda raw, _context: raw.upper())
        index = TriggerIndex(registry)
        index.replace_all(
            [
                Trigger(
                    id=1,
                    command_id=1,
                    pattern="включи {что:устройство}",
                    kind=TriggerKind.TEMPLATE,
                )
            ]
        )
        found = Matcher(index).match("включи свет")
        assert found is not None
        slots = index.snapshot().bind_slots(found, CTX)
        assert slots is not None
        assert slots.value("что") == "СВЕТ"
