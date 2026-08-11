"""Task 15: the matcher and the trigger index — exact, regex, fuzzy, conflicts.

Everything here runs against the in-memory index. That is not an economy of the
tests, it is the contract: the matcher runs on every recognised phrase and may
not touch the database while doing it, so a test that needed a connection would
be testing the wrong module.

Three things are easy to get wrong and are therefore pinned down hard.

*Fuzzy is a last resort.* It only runs when exact and regex found nothing, which
is both what keeps the millisecond budget and what stops «открой браузер» from
also reporting three near misses. On short phrases the ratio stops meaning
anything — «да» is one edit from half a library — so there is a floor below which
nothing is fuzzy-matched at all and a ramp above it.

*A broken pattern must not take voice control down.* One user typo in one regex
is refused at index time, logged, and the rest of the library goes on matching.
:func:`~ayris.nlu.matcher.validate_pattern` is the editor's side of that, so the
user hears about it while they can still fix it.

*The order is total.* Several triggers matching one phrase is normal — that is
what synonyms are — so the sort has to be deterministic down to the last key.
The tests assert on the whole ranking, not just on the head, and check that a
rebuilt index gives the same answer.

The incremental path is asserted through ``rebuilt_entries``: editing one command
of a thousand must re-normalise that command's phrases and nobody else's. A
counter is the only way to tell that apart from a full rebuild that happens to
produce the same snapshot.

Groups:

* :class:`TestLevenshtein` — distance, the ceiling, and the ratio built on it.
* :class:`TestSettings` — validation, config mapping, the short-phrase ramp.
* :class:`TestPatterns` — validation and compilation of a user's regex.
* :class:`TestExact` — the dictionary path, through normalisation.
* :class:`TestFuzzy` — typos, thresholds, the guards, and the off switch.
* :class:`TestRegex` — named groups, the spoken form, coverage as the score.
* :class:`TestConflicts` — priority, weight, kind, specificity, determinism.
* :class:`TestIndex` — building, stats, disabled triggers, single-trigger edits.
* :class:`TestIncremental` — one command changes, the rest of the index does not.
* :class:`TestBinding` — following :class:`CommandsChanged` on the bus.
* :class:`TestFromDb` — stored rows into the shape the matcher works with.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest

from ayris.core.events import (
    COMMANDS_CHANGE_DELETED,
    COMMANDS_CHANGE_RELOADED,
    COMMANDS_CHANGE_SAVED,
    CommandsChanged,
    EventBus,
)
from ayris.core.models import Trigger as DbTrigger
from ayris.core.models import TriggerType
from ayris.nlu.index import GRAM_SIZE, TriggerIndex, text_grams
from ayris.nlu.matcher import (
    DEFAULT_THRESHOLD,
    Matcher,
    MatcherSettings,
    MatchKind,
    Trigger,
    TriggerKind,
    compile_pattern,
    levenshtein,
    similarity,
    trigger_from_db,
    validate_pattern,
)
from ayris.nlu.normalize import normalize
from ayris.utils.logger import ROOT_LOGGER_NAME

pytestmark = pytest.mark.unit


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` on that logger and ``caplog``
    listens on the interpreter root, so a plain ``caplog.at_level`` sees nothing
    once any earlier test in the run has configured logging. Attaching the
    handler directly makes the assertion independent of test order.
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


def phrase(trigger_id: int, command_id: int, text: str, **kwargs: object) -> Trigger:
    """A voice trigger spelled out as a phrase."""
    return Trigger(id=trigger_id, command_id=command_id, pattern=text, **kwargs)  # type: ignore[arg-type]


def pattern(trigger_id: int, command_id: int, source: str, **kwargs: object) -> Trigger:
    """A voice trigger written as a regex."""
    return Trigger(
        id=trigger_id,
        command_id=command_id,
        pattern=source,
        kind=TriggerKind.REGEX,
        **kwargs,  # type: ignore[arg-type]
    )


def matcher(*triggers: Trigger, settings: MatcherSettings | None = None) -> Matcher:
    """A matcher over exactly these triggers."""
    return Matcher.from_triggers(triggers, settings)


def library(count: int) -> list[Trigger]:
    """``count`` distinct trigger phrases, three to a command."""
    return [phrase(i, i // 3, f"выполни процедуру номер {i} сейчас") for i in range(1, count + 1)]


class TestLevenshtein:
    def test_identical(self) -> None:
        assert levenshtein("открой браузер", "открой браузер") == 0

    def test_substitution(self) -> None:
        assert levenshtein("кот", "кит") == 1

    def test_insert_and_delete(self) -> None:
        assert levenshtein("кот", "кот ") == 1
        assert levenshtein("скот", "кот") == 1

    def test_empty_side(self) -> None:
        assert levenshtein("", "кот") == 3
        assert levenshtein("кот", "") == 3
        assert levenshtein("", "") == 0

    def test_symmetric(self) -> None:
        assert levenshtein("браузер", "барузер") == levenshtein("барузер", "браузер")

    def test_ceiling_reports_over_budget(self) -> None:
        # Not the real distance, but it compares correctly against the ceiling,
        # which is all the fuzzy sweep needs.
        assert levenshtein("абвгде", "жзиклм", max_distance=2) == 3

    def test_ceiling_does_not_change_a_fitting_answer(self) -> None:
        assert levenshtein("кот", "кит", max_distance=3) == 1

    def test_ceiling_of_zero(self) -> None:
        assert levenshtein("кот", "кот", max_distance=0) == 0
        assert levenshtein("кот", "кит", max_distance=0) == 1

    def test_length_difference_short_circuits(self) -> None:
        assert levenshtein("а", "абвгдеж", max_distance=2) == 3

    def test_similarity_ratio(self) -> None:
        assert similarity("открой браузер", "открой браузир") == pytest.approx(13 / 14)

    def test_similarity_of_equal_strings(self) -> None:
        assert similarity("да", "да") == 1.0
        assert similarity("", "") == 1.0

    def test_similarity_floor_abandons_early(self) -> None:
        # The number below the floor is not guaranteed to be the true ratio, only
        # to be below it.
        assert similarity("открой браузер", "закрой окно", 0.9) < 0.9


class TestSettings:
    def test_defaults(self) -> None:
        settings = MatcherSettings()
        assert settings.threshold == DEFAULT_THRESHOLD
        assert settings.fuzzy_enabled

    def test_ramp_between_the_two_thresholds(self) -> None:
        settings = MatcherSettings()
        assert settings.effective_threshold(settings.min_length) == settings.short_threshold
        assert settings.effective_threshold(settings.short_length) == settings.threshold
        middle = settings.effective_threshold(10)
        assert settings.threshold < middle < settings.short_threshold

    def test_ramp_is_monotonic(self) -> None:
        settings = MatcherSettings()
        values = [settings.effective_threshold(length) for length in range(1, 25)]
        assert values == sorted(values, reverse=True)

    def test_long_phrase_gets_the_plain_threshold(self) -> None:
        assert MatcherSettings().effective_threshold(200) == DEFAULT_THRESHOLD

    def test_from_config(self) -> None:
        from ayris.core.config import CommandsConfig

        settings = MatcherSettings.from_config(CommandsConfig(fuzzy_threshold=0.9))
        assert settings.threshold == pytest.approx(0.9)
        assert settings.short_threshold == MatcherSettings().short_threshold

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0.0},
            {"threshold": 1.5},
            {"short_threshold": 0.0},
            {"min_length": 0},
            {"short_length": 2},
        ],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError, match="."):
            MatcherSettings(**kwargs)  # type: ignore[arg-type]

    def test_settings_apply_to_the_next_phrase(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        assert engine.match("открой браузир") is not None
        engine.settings = MatcherSettings(fuzzy_enabled=False)
        assert engine.match("открой браузир") is None


class TestPatterns:
    def test_valid_pattern(self) -> None:
        assert validate_pattern(r"открой (?P<app>\w+)") == ""

    def test_empty_pattern(self) -> None:
        assert validate_pattern("   ") == "шаблон не может быть пустым"

    def test_broken_pattern_explains_itself(self) -> None:
        message = validate_pattern("(открой")
        assert message.startswith("ошибка в шаблоне:")

    def test_compile_keeps_named_groups(self) -> None:
        compiled = compile_pattern(r"(?:включи|выключи) (?P<device>\w+)")
        assert compiled is not None
        assert compiled.groupindex == {"device": 1}

    def test_compile_folds_yo_but_not_case(self) -> None:
        compiled = compile_pattern(r"свет на ёлке \S+")
        assert compiled is not None
        # ``ё`` is folded because normalisation removed it from the input side;
        # ``\S`` must survive, which lowercasing would not have allowed.
        assert compiled.pattern == r"свет на елке \S+"

    def test_compiled_pattern_is_case_insensitive(self) -> None:
        compiled = compile_pattern("ОТКРОЙ браузер")
        assert compiled is not None
        assert compiled.search("открой браузер") is not None

    def test_compile_drops_a_leading_address(self) -> None:
        compiled = compile_pattern(r"айрис, открой (?P<app>\w+)")
        assert compiled is not None
        assert compiled.search("открой браузер") is not None

    def test_compile_keeps_the_anchor_when_dropping_the_address(self) -> None:
        compiled = compile_pattern(r"^айрис открой (?P<app>\w+)")
        assert compiled is not None
        assert compiled.pattern.startswith("^открой")

    def test_compile_refuses_a_broken_pattern(self) -> None:
        assert compile_pattern("(открой") is None

    def test_broken_pattern_is_logged(self, ayris_log: pytest.LogCaptureFixture) -> None:
        assert compile_pattern("(открой") is None
        assert "не компилируется" in ayris_log.text


class TestExact:
    def test_stored_phrase_matches_spoken_phrase(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        result = engine.match("Айрис, открой браузер!")
        assert result is not None
        assert result.command_id == 10
        assert result.kind is MatchKind.EXACT
        assert result.score == 1.0

    def test_trigger_written_with_the_address(self) -> None:
        engine = matcher(phrase(1, 10, "Айрис, открой браузер"))
        assert engine.match("открой браузер") is not None

    def test_numerals_meet_in_the_middle(self) -> None:
        engine = matcher(phrase(1, 10, "громкость 50 процентов"))
        assert engine.match("громкость пятьдесят процентов") is not None

    def test_nothing_matches(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        assert engine.match("расскажи анекдот про программиста") is None

    def test_empty_phrase(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        assert engine.match("...") is None
        assert engine.match_all("") == []

    def test_accepts_an_already_normalised_phrase(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        result = engine.match(normalize("Айрис, открой браузер"))
        assert result is not None
        assert result.normalized_text == "открой браузер"

    def test_synonyms_of_one_command(self) -> None:
        engine = matcher(
            phrase(1, 10, "открой браузер"),
            phrase(2, 10, "запусти браузер"),
        )
        assert engine.match("открой браузер") is not None
        assert engine.match("запусти браузер") is not None

    def test_two_commands_share_a_phrase(self) -> None:
        engine = matcher(phrase(1, 10, "поехали"), phrase(2, 11, "поехали"))
        results = engine.match_all("поехали")
        assert {item.command_id for item in results} == {10, 11}

    def test_result_carries_the_context(self) -> None:
        engine = matcher(phrase(1, 10, "громкость 50", weight=0.7, priority=3))
        result = engine.match("громкость пятьдесят")
        assert result is not None
        assert result.trigger_id == 1
        assert result.pattern == "громкость 50"
        assert result.matched_text == "громкость 50"
        assert result.spoken_text == "громкость пятьдесят"
        assert result.weight == pytest.approx(0.7)
        assert result.priority == 3


class TestFuzzy:
    def test_recogniser_typo(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        result = engine.match("открой браузир")
        assert result is not None
        assert result.kind is MatchKind.FUZZY
        assert result.score < 1.0

    def test_too_far_is_not_a_match(self) -> None:
        engine = matcher(phrase(1, 10, "открой браузер"))
        assert engine.match("закрой все окна") is None

    def test_threshold_from_settings(self) -> None:
        triggers = [phrase(1, 10, "поставь напоминание")]
        strict = Matcher.from_triggers(triggers, MatcherSettings(threshold=0.99))
        lenient = Matcher.from_triggers(triggers, MatcherSettings(threshold=0.6))
        assert strict.match("поставь напоминанте") is None
        assert lenient.match("поставь напоминанте") is not None

    def test_per_trigger_threshold_overrides_settings(self) -> None:
        engine = matcher(phrase(1, 10, "поставь напоминание", threshold=1.0))
        assert engine.match("поставь напоминанте") is None
        assert engine.match("поставь напоминание") is not None

    def test_short_phrases_are_never_fuzzy_matched(self) -> None:
        # Below ``min_length`` the ratio means nothing: «да» is one edit from
        # half the library.
        engine = matcher(phrase(1, 10, "да"))
        assert engine.match("на") is None
        assert engine.match("да") is not None

    def test_the_ramp_protects_medium_phrases(self) -> None:
        settings = MatcherSettings()
        engine = matcher(phrase(1, 10, "стоп игра"), settings=settings)
        # One edit in nine characters is 0.889 — enough at the plain threshold,
        # not enough at the ramped one this length demands.
        assert settings.effective_threshold(len("стоп игра")) > 8 / 9
        assert engine.match("стоп игрп") is None

    def test_fuzzy_disabled_on_a_trigger(self) -> None:
        engine = matcher(phrase(1, 10, "удали все файлы", fuzzy=False))
        assert engine.match("удали все файды") is None
        assert engine.match("удали все файлы") is not None

    def test_fuzzy_disabled_globally(self) -> None:
        engine = matcher(
            phrase(1, 10, "открой браузер"),
            settings=MatcherSettings(fuzzy_enabled=False),
        )
        assert engine.match("открой браузир") is None

    def test_exact_hit_suppresses_the_fuzzy_sweep(self) -> None:
        # The near miss exists and would match on its own; an exact hit means the
        # sweep is never run, which is what keeps the millisecond budget.
        engine = matcher(
            phrase(1, 10, "открой браузер"),
            phrase(2, 11, "открой браузеры"),
        )
        results = engine.match_all("открой браузер")
        assert [item.kind for item in results] == [MatchKind.EXACT]
        assert engine.match_all("открой браузерx")[0].kind is MatchKind.FUZZY

    def test_prefilter_keeps_a_reachable_candidate(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "открой браузер")])
        candidates = index.snapshot().fuzzy_candidates("открой браузир", DEFAULT_THRESHOLD)
        assert [item.trigger.id for item in candidates] == [1]

    def test_prefilter_drops_an_unreachable_candidate(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "открой браузер")])
        assert index.snapshot().fuzzy_candidates("сверни все окна", DEFAULT_THRESHOLD) == []

    def test_prefilter_sorts_longest_first(self) -> None:
        index = TriggerIndex()
        index.replace_all(
            [
                phrase(1, 10, "открой браузер"),
                phrase(2, 11, "открой браузер быстро"),
            ]
        )
        candidates = index.snapshot().fuzzy_candidates("открой браузер", 0.5)
        assert [item.length for item in candidates] == sorted(
            (item.length for item in candidates), reverse=True
        )

    def test_grams_pad_short_words(self) -> None:
        assert GRAM_SIZE == 3
        assert text_grams("абв") == frozenset({"  а", " аб", "абв", "бв ", "в  "})
        assert text_grams("") == frozenset()


class TestRegex:
    def test_named_groups_reach_the_result(self) -> None:
        engine = matcher(pattern(1, 10, r"(?:включи|выключи) (?P<device>\w+)"))
        result = engine.match("включи свет")
        assert result is not None
        assert result.kind is MatchKind.REGEX
        assert result.raw_groups == {"device": "свет"}

    def test_unmatched_optional_group_is_left_out(self) -> None:
        engine = matcher(pattern(1, 10, r"открой (?P<app>\w+)(?: в (?P<mode>\w+))?"))
        result = engine.match("открой браузер")
        assert result is not None
        assert result.raw_groups == {"app": "браузер"}

    def test_pattern_written_against_words_sees_them(self) -> None:
        # The digit rewrite must not break a pattern written the way people speak.
        engine = matcher(pattern(1, 10, r"таймер на (?P<minutes>пять|десять) минут"))
        result = engine.match("поставь таймер на пять минут")
        assert result is not None
        assert result.raw_groups == {"minutes": "пять"}

    def test_pattern_written_against_digits_also_works(self) -> None:
        engine = matcher(pattern(1, 10, r"таймер на (?P<minutes>\d+) минут"))
        result = engine.match("поставь таймер на пять минут")
        assert result is not None
        assert result.raw_groups == {"minutes": "5"}

    def test_score_is_the_share_covered(self) -> None:
        engine = matcher(pattern(1, 10, r"включи (?P<device>\w+)"))
        full = engine.match("включи свет")
        partial = engine.match("включи свет и закрой дверь")
        assert full is not None and partial is not None
        assert full.score == 1.0
        assert partial.score < full.score

    def test_address_in_front_of_the_input(self) -> None:
        engine = matcher(pattern(1, 10, r"включи (?P<device>\w+)"))
        assert engine.match("Айрис, включи свет") is not None

    def test_yo_in_the_pattern(self) -> None:
        engine = matcher(pattern(1, 10, r"включи свет на ёлке"))
        assert engine.match("включи свет на елке") is not None

    def test_broken_pattern_is_skipped_not_raised(self) -> None:
        engine = matcher(
            pattern(1, 10, "(включи"),
            phrase(2, 11, "включи свет"),
        )
        result = engine.match("включи свет")
        assert result is not None
        assert result.trigger_id == 2

    def test_a_regex_trigger_is_never_fuzzy_matched(self) -> None:
        engine = matcher(pattern(1, 10, r"включи (?P<device>\w+)"))
        assert engine.match("вклюци свет") is None

    def test_regex_triggers_are_not_in_the_exact_map(self) -> None:
        index = TriggerIndex()
        index.replace_all([pattern(1, 10, r"включи (?P<device>\w+)")])
        stats = index.stats()
        assert stats.regex_triggers == 1
        assert stats.exact_phrases == 0
        assert stats.fuzzy_triggers == 0


class TestConflicts:
    def test_priority_wins(self) -> None:
        engine = matcher(
            phrase(1, 10, "поехали", priority=1),
            phrase(2, 11, "поехали", priority=5),
        )
        result = engine.match("поехали")
        assert result is not None
        assert result.command_id == 11

    def test_weight_breaks_a_priority_tie(self) -> None:
        engine = matcher(
            phrase(1, 10, "поехали", weight=0.4),
            phrase(2, 11, "поехали", weight=0.9),
        )
        result = engine.match("поехали")
        assert result is not None
        assert result.trigger_id == 2

    def test_score_breaks_a_weight_tie(self) -> None:
        engine = matcher(
            phrase(1, 10, "открой браузер"),
            phrase(2, 11, "открой браузеры"),
        )
        results = engine.match_all("открой браузерx")
        assert [item.trigger_id for item in results] == [2, 1]

    def test_exact_beats_regex_at_the_same_score(self) -> None:
        engine = matcher(
            pattern(1, 10, r"открой (?P<app>браузер)"),
            phrase(2, 11, "открой браузер"),
        )
        results = engine.match_all("открой браузер")
        assert [item.kind for item in results] == [MatchKind.EXACT, MatchKind.REGEX]

    def test_the_more_specific_pattern_wins(self) -> None:
        # Two patterns cover the phrase equally; the longer source is the one
        # that spells out what it accepts, so it goes first.
        engine = matcher(
            pattern(1, 10, r"^включи (?P<device>.+)$"),
            pattern(2, 11, r"^включи (?P<device>свет|лампу)$"),
        )
        results = engine.match_all("включи свет")
        assert [item.score for item in results] == [1.0, 1.0]
        assert [item.trigger_id for item in results] == [2, 1]

    def test_order_does_not_depend_on_insertion_order(self) -> None:
        triggers = [
            phrase(1, 10, "поехали"),
            phrase(2, 11, "поехали"),
            phrase(3, 12, "поехали"),
        ]
        forward = matcher(*triggers).match_all("поехали")
        backward = matcher(*reversed(triggers)).match_all("поехали")
        assert [item.trigger_id for item in forward] == [item.trigger_id for item in backward]

    def test_the_last_key_is_the_trigger_id(self) -> None:
        engine = matcher(phrase(7, 10, "поехали"), phrase(3, 11, "поехали"))
        assert [item.trigger_id for item in engine.match_all("поехали")] == [3, 7]

    def test_match_is_the_head_of_match_all(self) -> None:
        engine = matcher(
            phrase(1, 10, "поехали", priority=2),
            phrase(2, 11, "поехали", priority=9),
        )
        results = engine.match_all("поехали")
        best = engine.match("поехали")
        assert best is not None
        assert best.trigger_id == results[0].trigger_id

    def test_limit_and_max_results(self) -> None:
        triggers = [phrase(i, i, "поехали") for i in range(1, 8)]
        engine = Matcher.from_triggers(triggers, MatcherSettings(max_results=3))
        assert len(engine.match_all("поехали")) == 3
        assert len(engine.match_all("поехали", limit=2)) == 2

    def test_sort_key_shape(self) -> None:
        engine = matcher(phrase(1, 10, "поехали", weight=0.5, priority=2))
        result = engine.match("поехали")
        assert result is not None
        assert result.sort_key[:2] == (2, 0.5)


class TestIndex:
    def test_building_and_stats(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(30))
        stats = index.stats()
        assert stats.triggers == 30
        assert stats.exact_phrases == 30
        assert stats.fuzzy_triggers == 30
        assert stats.grams > 0
        assert len(index) == 30
        assert index.generation == stats.generation == 1

    def test_replace_all_drops_the_previous_library(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(9))
        index.replace_all([phrase(100, 50, "новая единственная команда")])
        assert len(index) == 1
        assert index.triggers_for(0) == ()

    def test_a_phrase_that_normalises_to_nothing_is_refused(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "!!!"), phrase(2, 10, "открой браузер")])
        assert len(index) == 1

    def test_triggers_for_returns_them_as_stored(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "Открой Браузер!"), phrase(2, 11, "закрой окно")])
        assert [item.pattern for item in index.triggers_for(10)] == ["Открой Браузер!"]

    def test_add_and_remove_a_single_trigger(self) -> None:
        index = TriggerIndex()
        index.add(phrase(1, 10, "открой браузер"))
        assert len(index) == 1
        index.remove(1)
        assert len(index) == 0

    def test_removing_an_unknown_trigger_changes_nothing(self) -> None:
        index = TriggerIndex()
        index.add(phrase(1, 10, "открой браузер"))
        generation = index.generation
        index.remove(999)
        assert index.generation == generation

    def test_adding_the_same_id_replaces_it(self) -> None:
        index = TriggerIndex()
        index.add(phrase(1, 10, "открой браузер"))
        index.add(phrase(1, 10, "запусти браузер"))
        assert [item.pattern for item in index.triggers_for(10)] == ["запусти браузер"]

    def test_moving_a_trigger_to_another_command(self) -> None:
        index = TriggerIndex()
        index.add(phrase(1, 10, "открой браузер"))
        index.add(phrase(1, 11, "открой браузер"))
        assert index.triggers_for(10) == ()
        assert len(index.triggers_for(11)) == 1

    def test_disabled_triggers_stay_indexed_but_cannot_match(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "открой браузер"), phrase(2, 11, "закрой окно")])
        index.set_enabled(10, False)
        engine = Matcher(index)
        assert engine.match("открой браузер") is None
        assert len(index) == 2
        index.set_enabled(10, True)
        assert engine.match("открой браузер") is not None

    def test_set_priority_reaches_the_results(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "поехали"), phrase(2, 11, "поехали")])
        index.set_priority(11, 5)
        result = Matcher(index).match("поехали")
        assert result is not None
        assert result.command_id == 11

    def test_retagging_does_not_recompile(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(30))
        before = index.stats().rebuilt_entries
        index.set_priority(3, 4)
        index.set_enabled(3, False)
        assert index.stats().rebuilt_entries == before

    def test_retagging_to_the_same_value_is_a_no_op(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "поехали", priority=3)])
        generation = index.generation
        index.set_priority(10, 3)
        assert index.generation == generation

    def test_retagging_an_unknown_command_is_a_no_op(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "поехали")])
        generation = index.generation
        index.set_priority(999, 3)
        index.set_enabled(999, False)
        assert index.generation == generation

    def test_snapshot_is_immutable_from_the_reader_side(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "открой браузер")])
        taken = index.snapshot()
        index.replace_all([phrase(2, 11, "закрой окно")])
        # The old snapshot still describes the old library: a match in flight is
        # not disturbed by an edit.
        assert [item.trigger.id for item in taken.all_triggers()] == [1]
        assert taken.generation < index.generation

    def test_matcher_sees_the_index_it_was_given(self) -> None:
        index = TriggerIndex()
        engine = Matcher(index)
        assert engine.index is index
        assert engine.match("открой браузер") is None
        index.add(phrase(1, 10, "открой браузер"))
        assert engine.match("открой браузер") is not None

    def test_exact_candidates_and_all_triggers(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "открой браузер"), pattern(2, 11, r"включи (\w+)")])
        snapshot = index.snapshot()
        assert len(snapshot.exact_candidates("открой браузер")) == 1
        assert snapshot.exact_candidates("нет такой фразы") == ()
        assert len(snapshot.regex_triggers()) == 1
        assert {item.trigger.id for item in snapshot.all_triggers()} == {1, 2}

    def test_indexed_trigger_reports_its_shape(self) -> None:
        index = TriggerIndex()
        index.replace_all([phrase(1, 10, "Открой Браузер"), pattern(2, 11, r"включи (\w+)")])
        by_id = {item.trigger.id: item for item in index.snapshot().all_triggers()}
        assert by_id[1].text == "открой браузер"
        assert by_id[1].length == len("открой браузер")
        assert not by_id[1].is_regex
        assert by_id[2].is_regex
        assert isinstance(by_id[2].regex, re.Pattern)


class TestIncremental:
    def test_one_command_edited_rebuilds_only_it(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(300))
        before = index.stats()
        index.update_command(5, [phrase(15, 5, "совершенно другая фраза команды")])
        after = index.stats()
        # Three triggers went in, one came out; a full rebuild would have moved
        # the counter by three hundred.
        assert after.rebuilt_entries - before.rebuilt_entries == 1
        assert after.triggers == before.triggers - 2
        assert after.generation == before.generation + 1

    def test_the_rest_of_the_library_still_matches(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(30))
        engine = Matcher(index)
        index.update_command(5, [phrase(15, 5, "совершенно другая фраза команды")])
        assert engine.match("выполни процедуру номер 1 сейчас") is not None
        assert engine.match("совершенно другая фраза команды") is not None
        # The two phrases command 5 used to own left the postings with it: their
        # own text no longer resolves to command 5.
        indexed = {item.trigger.id for item in index.snapshot().all_triggers()}
        assert 15 in indexed
        assert {16, 17}.isdisjoint(indexed)
        for gone in ("выполни процедуру номер 16 сейчас", "выполни процедуру номер 17 сейчас"):
            assert all(item.command_id != 5 for item in engine.match_all(gone))

    def test_update_reassigns_the_command_id(self) -> None:
        index = TriggerIndex()
        index.update_command(7, [phrase(1, 999, "открой браузер")])
        assert [item.command_id for item in index.triggers_for(7)] == [7]

    def test_update_with_nothing_empties_the_command(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(9))
        index.update_command(1, [])
        assert index.triggers_for(1) == ()
        assert len(index) == 6

    def test_remove_command(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(9))
        index.remove_command(1)
        assert len(index) == 6
        assert index.stats().commands == 3

    def test_removing_an_unknown_command_is_a_no_op(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(9))
        generation = index.generation
        index.remove_command(999)
        assert index.generation == generation

    def test_generation_moves_once_per_change(self) -> None:
        index = TriggerIndex()
        index.replace_all(library(9))
        generation = index.generation
        index.update_command(1, [phrase(4, 1, "новая фраза команды один")])
        index.remove_command(2)
        assert index.generation == generation + 2


class TestBinding:
    def test_saved_command_is_reloaded_incrementally(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        shelf = {1: [phrase(1, 1, "открой браузер")]}
        index.bind(bus, lambda cid: shelf[cid] if cid is not None else _flatten(shelf))
        bus.publish(CommandsChanged(command_id=1, change=COMMANDS_CHANGE_SAVED))
        assert Matcher(index).match("открой браузер") is not None

    def test_reloaded_replaces_the_whole_library(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        shelf = {1: [phrase(1, 1, "открой браузер")], 2: [phrase(2, 2, "закрой окно")]}
        index.bind(bus, lambda cid: shelf[cid] if cid is not None else _flatten(shelf))
        bus.publish(CommandsChanged(change=COMMANDS_CHANGE_RELOADED))
        assert len(index) == 2
        del shelf[2]
        bus.publish(CommandsChanged(change=COMMANDS_CHANGE_RELOADED))
        assert len(index) == 1

    def test_a_change_without_a_command_id_reloads_everything(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        shelf = {1: [phrase(1, 1, "открой браузер")]}
        index.bind(bus, lambda cid: shelf[cid] if cid is not None else _flatten(shelf))
        bus.publish(CommandsChanged(change=COMMANDS_CHANGE_SAVED))
        assert len(index) == 1

    def test_deleted_command_does_not_ask_the_loader(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        index.replace_all([phrase(1, 1, "открой браузер"), phrase(2, 2, "закрой окно")])
        asked: list[int | None] = []

        def loader(command_id: int | None) -> list[Trigger]:
            asked.append(command_id)
            return []

        index.bind(bus, loader)
        bus.publish(CommandsChanged(command_id=2, change=COMMANDS_CHANGE_DELETED))
        assert len(index) == 1
        assert asked == []

    def test_a_failing_loader_leaves_the_previous_snapshot(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        index.replace_all([phrase(1, 1, "открой браузер")])
        generation = index.generation

        def loader(command_id: int | None) -> list[Trigger]:
            raise RuntimeError("база недоступна")

        index.bind(bus, loader)
        bus.publish(CommandsChanged(command_id=1, change=COMMANDS_CHANGE_SAVED))
        assert index.generation == generation
        assert Matcher(index).match("открой браузер") is not None

    def test_unsubscribing_stops_the_updates(self) -> None:
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        shelf = {1: [phrase(1, 1, "открой браузер")]}
        unsubscribe = index.bind(
            bus, lambda cid: shelf[cid] if cid is not None else _flatten(shelf)
        )
        unsubscribe()
        bus.publish(CommandsChanged(command_id=1, change=COMMANDS_CHANGE_SAVED))
        assert len(index) == 0
        assert bus.subscriber_count(CommandsChanged) == 0

    def test_the_subscription_survives_the_call_that_made_it(self) -> None:
        # ``bind`` subscribes a closure, which a weak subscription would drop the
        # moment it returned.
        bus = EventBus(thread_id=None)
        index = TriggerIndex()
        index.bind(bus, lambda _cid: [phrase(1, 1, "открой браузер")])
        assert bus.subscriber_count(CommandsChanged) == 1


def _flatten(shelf: dict[int, list[Trigger]]) -> list[Trigger]:
    """Every trigger on the shelf, the way a full reload would read them."""
    return [trigger for triggers in shelf.values() for trigger in triggers]


class TestFromDb:
    def test_voice_phrase(self) -> None:
        row = DbTrigger(
            id=4,
            command_id=10,
            type=TriggerType.VOICE,
            payload={"phrase": "открой почту"},
            priority=2,
        )
        trigger = trigger_from_db(row, command_priority=7)
        assert trigger is not None
        assert trigger.id == 4
        assert trigger.command_id == 10
        assert trigger.pattern == "открой почту"
        assert trigger.kind is TriggerKind.PHRASE
        # The database keeps the trigger's weight in its ``priority`` column; the
        # command's own priority is the one passed in.
        assert trigger.weight == pytest.approx(2.0)
        assert trigger.priority == 7

    def test_voice_regex(self) -> None:
        row = DbTrigger(
            id=1,
            command_id=10,
            type=TriggerType.VOICE,
            payload={"regex": r"открой (?P<app>\w+)"},
        )
        trigger = trigger_from_db(row)
        assert trigger is not None
        assert trigger.kind is TriggerKind.REGEX

    def test_payload_weight_and_threshold_win(self) -> None:
        row = DbTrigger(
            command_id=10,
            type=TriggerType.VOICE,
            payload={"phrase": "открой почту", "weight": 0.5, "threshold": 0.95},
            priority=9,
        )
        trigger = trigger_from_db(row)
        assert trigger is not None
        assert trigger.weight == pytest.approx(0.5)
        assert trigger.threshold == pytest.approx(0.95)

    def test_disabled_in_the_payload(self) -> None:
        row = DbTrigger(
            command_id=10,
            type=TriggerType.VOICE,
            payload={"phrase": "открой почту", "enabled": False},
        )
        trigger = trigger_from_db(row)
        assert trigger is not None
        assert not trigger.enabled

    def test_disabled_command(self) -> None:
        row = DbTrigger(command_id=10, type=TriggerType.VOICE, payload={"phrase": "открой почту"})
        trigger = trigger_from_db(row, enabled=False)
        assert trigger is not None
        assert not trigger.enabled

    def test_other_trigger_types_are_not_ours(self) -> None:
        for kind in TriggerType:
            if kind is TriggerType.VOICE:
                continue
            row = DbTrigger(command_id=10, type=kind, payload={"phrase": "открой почту"})
            assert trigger_from_db(row) is None

    def test_a_voice_trigger_with_nothing_to_match(self) -> None:
        assert trigger_from_db(DbTrigger(command_id=10, type=TriggerType.VOICE)) is None
        row = DbTrigger(command_id=10, type=TriggerType.VOICE, payload={"phrase": "   "})
        assert trigger_from_db(row) is None

    def test_rows_go_straight_into_the_index(self) -> None:
        rows = [
            DbTrigger(
                id=1, command_id=10, type=TriggerType.VOICE, payload={"phrase": "открой почту"}
            ),
            DbTrigger(id=2, command_id=10, type=TriggerType.HOTKEY, payload={"keys": "ctrl+m"}),
        ]
        triggers = [item for item in (trigger_from_db(row) for row in rows) if item is not None]
        engine = Matcher.from_triggers(triggers)
        result = engine.match("Айрис, открой почту")
        assert result is not None
        assert result.command_id == 10
