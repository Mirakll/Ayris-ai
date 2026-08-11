"""Task 15: the matcher's time budget on a library that is bigger than real.

The acceptance criterion is single-digit milliseconds on 1000+ triggers, and the
only honest way to check it is to build such a library and time it. Two thousand
triggers is deliberately twice the number in the criterion: a user with two
thousand voice phrases does not exist, so anything that passes here has room.

The numbers are what the sandbox measures; the ceilings asserted below are far
above them, because a CI runner is shared hardware and a tight bound would turn
a green build red for reasons that have nothing to do with this code. The point
of the assertion is to catch an algorithmic regression — a linear sweep creeping
back into the fuzzy path, a regex recompiled per phrase, the prefilter stopping
to filter — not to measure a machine.

What makes it fast is worth stating, because these are the properties a future
edit could quietly remove: exact matching is a dictionary lookup, the regexes
are compiled once when the index is built, the fuzzy sweep only runs when the
first two found nothing and only against candidates that share enough character
trigrams to be reachable within the threshold, and the distance itself is
bit-parallel rather than a Python loop over a matrix.

Marked ``slow`` so the ordinary run stays quick; ``-m slow`` or a full run has
it.

Groups:

* :class:`TestMatchBudget` — a miss, a hit, and a typo against 2000 triggers.
* :class:`TestPrefilter` — the sweep looks at a small share of the library, and
  loses nothing the scorer would have accepted.
* :class:`TestIndexBudget` — building once, and editing one command of many.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import pytest

from ayris.nlu.index import TriggerIndex
from ayris.nlu.matcher import (
    DEFAULT_THRESHOLD,
    Matcher,
    MatcherSettings,
    MatchKind,
    Trigger,
    TriggerKind,
    similarity,
)
from ayris.nlu.normalize import normalize

pytestmark = [pytest.mark.unit, pytest.mark.slow]

TRIGGER_COUNT = 2000
TRIGGERS_PER_COMMAND = 3

#: One letter away from the first phrase in the library, so the fuzzy path is
#: the one that has to answer and the answer is known.
_TYPO = "открой браузер на экрне вариант 0"

_VERBS = ("открой", "закрой", "запусти", "останови", "сверни", "разверни", "покажи", "спрячь")
_OBJECTS = ("браузер", "почту", "музыку", "календарь", "заметки", "терминал", "проводник")
_PLACES = ("на экране", "в фоне", "справа", "слева", "сверху", "внизу", "в углу")


def _phrases(count: int) -> list[str]:
    """``count`` distinct Russian phrases of a realistic length."""
    phrases = []
    for i in range(count):
        verb = _VERBS[i % len(_VERBS)]
        obj = _OBJECTS[(i // len(_VERBS)) % len(_OBJECTS)]
        place = _PLACES[(i // (len(_VERBS) * len(_OBJECTS))) % len(_PLACES)]
        phrases.append(f"{verb} {obj} {place} вариант {i}")
    return phrases


@pytest.fixture(scope="module")
def big_index() -> TriggerIndex:
    """An index over :data:`TRIGGER_COUNT` phrase triggers and 200 regexes."""
    triggers: list[Trigger] = [
        Trigger(id=i + 1, command_id=i // TRIGGERS_PER_COMMAND, pattern=text)
        for i, text in enumerate(_phrases(TRIGGER_COUNT))
    ]
    triggers += [
        Trigger(
            id=TRIGGER_COUNT + i + 1,
            command_id=10_000 + i,
            pattern=rf"поставь таймер на (?P<minutes>\d+) минут вариант {i}",
            kind=TriggerKind.REGEX,
        )
        for i in range(200)
    ]
    index = TriggerIndex()
    index.replace_all(triggers)
    return index


def _milliseconds(call: Callable[[], object], repeats: int) -> float:
    """Best-of wall time per call, in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - started)
    return best * 1000.0


class TestMatchBudget:
    def test_the_library_is_the_size_it_claims(self, big_index: TriggerIndex) -> None:
        assert len(big_index) >= 1000
        stats = big_index.stats()
        assert stats.exact_phrases == TRIGGER_COUNT
        assert stats.regex_triggers == 200

    def test_exact_hit(self, big_index: TriggerIndex) -> None:
        engine = Matcher(big_index)
        said = "Айрис, открой браузер на экране вариант 0"
        assert engine.match(said) is not None
        assert _milliseconds(lambda: engine.match(said), 20) < 10.0

    def test_regex_hit(self, big_index: TriggerIndex) -> None:
        engine = Matcher(big_index)
        said = "поставь таймер на пять минут вариант 199"
        result = engine.match(said)
        assert result is not None
        assert result.kind is MatchKind.REGEX
        # Every regex is tried in turn, so this is the widest of the three paths;
        # it is also the one that would explode if patterns were recompiled here.
        assert _milliseconds(lambda: engine.match(said), 20) < 20.0

    def test_fuzzy_hit(self, big_index: TriggerIndex) -> None:
        engine = Matcher(big_index)
        said = "открой браузер на экрне вариант 0"
        result = engine.match(said)
        assert result is not None
        assert result.kind is MatchKind.FUZZY
        # The expensive path, and the one the acceptance criterion is about: a
        # single-digit number of milliseconds here is the prefilter and the
        # bit-parallel distance working; a linear sweep would be twenty times it.
        assert _milliseconds(lambda: engine.match(said), 20) < 30.0

    def test_a_phrase_nothing_matches(self, big_index: TriggerIndex) -> None:
        engine = Matcher(big_index)
        said = "расскажи пожалуйста длинный анекдот про сисадмина и его кота"
        assert engine.match(said) is None
        assert _milliseconds(lambda: engine.match(said), 20) < 20.0

    def test_a_stream_of_phrases_averages_well_under_the_budget(
        self, big_index: TriggerIndex
    ) -> None:
        engine = Matcher(big_index)
        said = [
            "открой браузер на экране вариант 0",
            "закрой почту на экране вариант 1",
            "поставь таймер на десять минут вариант 5",
            "открой браузер на экрне вариант 0",
            "расскажи анекдот про сисадмина и его кота",
        ]
        elapsed = _milliseconds(lambda: [engine.match(item) for item in said], 5)
        assert elapsed / len(said) < 20.0


class TestPrefilter:
    def test_the_sweep_sees_a_small_share_of_the_library(self, big_index: TriggerIndex) -> None:
        snapshot = big_index.snapshot()
        candidates = snapshot.fuzzy_candidates(_TYPO, DEFAULT_THRESHOLD)
        assert candidates
        # A linear sweep would be the whole library; the trigram postings and the
        # length band have to cut that down by an order of magnitude.
        assert len(candidates) < TRIGGER_COUNT // 10

    def test_the_reachable_trigger_survives_the_prefilter(self, big_index: TriggerIndex) -> None:
        snapshot = big_index.snapshot()
        candidates = snapshot.fuzzy_candidates(_TYPO, DEFAULT_THRESHOLD)
        assert any(item.text == "открой браузер на экране вариант 0" for item in candidates)

    def test_prefiltering_is_itself_cheap(self, big_index: TriggerIndex) -> None:
        snapshot = big_index.snapshot()
        elapsed = _milliseconds(
            lambda: snapshot.fuzzy_candidates(_TYPO, DEFAULT_THRESHOLD),
            20,
        )
        assert elapsed < 10.0

    def test_the_prefilter_loses_nothing_the_scorer_would_have_taken(
        self, big_index: TriggerIndex
    ) -> None:
        # The bound is only worth having if it is a bound. Against the whole
        # library, scored the slow way, the prefilter must not have dropped a
        # single trigger that would have cleared its threshold.
        snapshot = big_index.snapshot()
        settings = MatcherSettings()
        said = normalize(_TYPO).text
        reachable = {
            entry.trigger.id
            for entry in snapshot.entries.values()
            if similarity(said, entry.text)
            >= settings.effective_threshold(min(len(said), entry.length))
        }
        assert reachable
        survived = {
            entry.trigger.id for entry in snapshot.fuzzy_candidates(said, DEFAULT_THRESHOLD)
        }
        assert reachable <= survived


class TestIndexBudget:
    def test_building_the_whole_library(self) -> None:
        triggers: Sequence[Trigger] = [
            Trigger(id=i + 1, command_id=i // TRIGGERS_PER_COMMAND, pattern=text)
            for i, text in enumerate(_phrases(TRIGGER_COUNT))
        ]
        index = TriggerIndex()
        elapsed = _milliseconds(lambda: index.replace_all(triggers), 3)
        # A full rebuild happens on startup and on an import, not per phrase, so
        # this budget is a second, not a millisecond.
        assert elapsed < 5000.0
        assert len(index) == TRIGGER_COUNT

    def test_editing_one_command_is_far_cheaper_than_rebuilding(
        self, big_index: TriggerIndex
    ) -> None:
        replacement = [Trigger(id=1, command_id=0, pattern="совсем другая фраза для команды ноль")]
        original = list(big_index.triggers_for(0))
        try:
            elapsed = _milliseconds(lambda: big_index.update_command(0, replacement), 5)
            assert elapsed < 20.0
        finally:
            big_index.update_command(0, original)

    def test_the_edit_touches_only_that_command(self, big_index: TriggerIndex) -> None:
        original = list(big_index.triggers_for(0))
        before = big_index.stats().rebuilt_entries
        try:
            big_index.update_command(
                0, [Trigger(id=1, command_id=0, pattern="совсем другая фраза для команды ноль")]
            )
            assert big_index.stats().rebuilt_entries - before == 1
        finally:
            big_index.update_command(0, original)
