"""Deciding which command a phrase means: exact, then regex, then fuzzy.

The matcher is the narrow part of the pipeline. It runs on every recognised
phrase, it must answer in well under a millisecond with a library of a thousand
triggers, and it may not touch the database while doing it — everything it reads
is the in-memory index of :mod:`ayris.nlu.index`.

**Three strategies, in order of how much they can be trusted.**

*Exact* is a dictionary lookup on the normalised phrase. After
:func:`ayris.nlu.normalize.normalize` has folded case, ``ё``, punctuation, the
address and the numerals, most utterances land here, and they cost one hash.

*Regex* runs the patterns the user wrote, with their named groups going into
:attr:`MatchResult.raw_groups` for the slot extraction of task 16. A pattern is
tried against the *spoken* form first — the one where ``пятьдесят`` is still a
word — because a pattern written against words must not be broken by the digit
rewrite; if that finds nothing, the digit form is tried, so ``(?P<level>\\d+)``
keeps working for a user who says the number out loud. The score is the share of
the phrase the match covers, which makes an anchored pattern beat a fragment
without any special-casing.

*Fuzzy* is a Levenshtein ratio against the phrases the prefilter let through.
It exists for one reason — recognisers mishear a single sound, «открой браузир»
— and it is dangerous for exactly the opposite reason: on short phrases the
ratio stops meaning anything, because «да» is one edit away from half the
library. Two guards, both in :class:`MatcherSettings`: nothing shorter than
:attr:`~MatcherSettings.min_length` is fuzzy-matched at all, and between there
and :attr:`~MatcherSettings.short_length` the required similarity ramps up to
:attr:`~MatcherSettings.short_threshold`, so a short phrase has to be nearly
perfect to win.

**Conflicts are resolved, not avoided.** Several triggers matching one phrase is
normal — that is what synonyms are. :meth:`Matcher.match_all` sorts them by
command priority, then trigger weight, then score, then the kind of match
(exact over regex over fuzzy), then the length of the pattern that matched, so
the more specific of two equally good triggers wins. The last key is the trigger
id, which means the order never depends on dictionary iteration or on the order
triggers were added: two runs over the same library give the same answer.

:meth:`Matcher.match` returns the head of that list, or ``None``. The full list
is what the DevTools tab of task 58 shows when a user asks why the wrong command
fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.nlu.normalize import DEFAULT_ADDRESS_FORMS, NormalizedText, fold_yo, normalize
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ayris.core.config import CommandsConfig
    from ayris.core.models import Trigger as DbTrigger
    from ayris.nlu.index import IndexedTrigger, IndexSnapshot, TriggerIndex

__all__ = [
    "DEFAULT_THRESHOLD",
    "MatchKind",
    "MatchResult",
    "Matcher",
    "MatcherSettings",
    "Trigger",
    "TriggerKind",
    "compile_pattern",
    "distance_ceiling",
    "levenshtein",
    "similarity",
    "trigger_from_db",
    "validate_pattern",
]

_log = get_logger(__name__)

#: Default similarity a fuzzy candidate needs. Mirrors
#: :attr:`ayris.core.config.CommandsConfig.fuzzy_threshold`, which is where the
#: user changes it; the constant is here so the matcher works without settings.
DEFAULT_THRESHOLD: Final = 0.82

#: Anchors and the address in front of a user's regex. A pattern copied from a
#: voice phrase usually starts with «айрис», which normalisation has already
#: removed from the input, so the pattern would never match anything.
_PATTERN_ADDRESS: Final = re.compile(
    r"^\s*(\^)?\s*(?:{})\b[\s,]*".format("|".join(DEFAULT_ADDRESS_FORMS)),
    re.IGNORECASE,
)


class TriggerKind(StrEnum):
    """How a trigger's pattern is meant to be read."""

    PHRASE = "phrase"
    REGEX = "regex"


class MatchKind(StrEnum):
    """How a phrase ended up matching. Ordered by trust in :data:`_KIND_RANK`."""

    EXACT = "exact"
    REGEX = "regex"
    FUZZY = "fuzzy"


#: Trust order for the tie-break. An exact hit beats a regex that covers the
#: whole phrase, which beats a fuzzy ratio of the same number.
_KIND_RANK: Final[dict[MatchKind, int]] = {
    MatchKind.EXACT: 2,
    MatchKind.REGEX: 1,
    MatchKind.FUZZY: 0,
}


@dataclass(frozen=True, slots=True)
class Trigger:
    """One way of saying one command.

    A command owns several of these — synonyms, a regex with slots, a terser
    phrase for when the user is in a hurry — and they compete with each other as
    well as with other commands' triggers.

    ``priority`` is the *command's* priority, copied in when the index is built.
    It has to live here because :meth:`Matcher.match` may not open the database
    to look it up, and it is the first sorting key: a high-priority command wins
    over a lower one even when the lower one matched better.

    ``weight`` ranks triggers *within* that priority — the phrase the user
    considers canonical against the sloppy synonym they added later.

    ``threshold`` overrides :attr:`MatcherSettings.threshold` for this trigger
    alone: a phrase full of proper nouns can afford to be strict, a long one can
    afford not to be.
    """

    id: int
    command_id: int
    pattern: str
    kind: TriggerKind = TriggerKind.PHRASE
    fuzzy: bool = True
    threshold: float | None = None
    weight: float = 1.0
    priority: int = 0
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class MatchResult:
    """A trigger that fired, and everything needed to rank or explain it.

    ``raw_groups`` holds the named groups of a regex exactly as they were
    captured — untrimmed, unconverted — because turning ``"50"`` into an integer
    or resolving ``"браузер"`` to an executable is the job of the slot layer in
    task 16, and it needs the original text to do it.

    Both forms of the phrase are kept: ``normalized_text`` with digits, which is
    what exact and fuzzy matched on, and ``spoken_text`` with the number words
    intact, which is what a regex most likely saw.
    """

    command_id: int
    trigger_id: int
    score: float
    kind: MatchKind
    raw_groups: dict[str, str] = field(default_factory=dict)
    normalized_text: str = ""
    spoken_text: str = ""
    matched_text: str = ""
    pattern: str = ""
    weight: float = 1.0
    priority: int = 0

    @property
    def sort_key(self) -> tuple[int, float, float, int, int, int]:
        """Ranking key, descending on every field except the id.

        The trigger id at the end is what makes the order total: without it two
        identical triggers would be ordered by whatever the index happened to
        iterate first, and the same phrase could fire different commands on two
        runs of the same program.
        """
        return (
            self.priority,
            self.weight,
            round(self.score, 6),
            _KIND_RANK[self.kind],
            len(self.pattern),
            -self.trigger_id,
        )


@dataclass(frozen=True, slots=True)
class MatcherSettings:
    """Thresholds and guards, all of them tunable from the settings window.

    The defaults are the ones the tab «Команды» ships with; see
    :meth:`from_config` for the two that the user can actually reach.
    """

    #: Similarity a fuzzy candidate needs on a phrase of normal length.
    threshold: float = DEFAULT_THRESHOLD
    #: Similarity required at :attr:`min_length`. Higher, because a short phrase
    #: reaches a high ratio by accident.
    short_threshold: float = 0.95
    #: Below this many characters a phrase is never fuzzy-matched.
    min_length: int = 6
    #: At and above this length the plain :attr:`threshold` applies; in between
    #: the requirement is interpolated between the two.
    short_length: int = 14
    #: Whether fuzzy matching runs at all. The user who turned it off gets exact
    #: and regex only, and a much quieter false-positive rate.
    fuzzy_enabled: bool = True
    #: Cap on results :meth:`Matcher.match_all` returns; the tail of a fuzzy
    #: sweep is noise and the DevTools list has to end somewhere.
    max_results: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("порог схожести задаётся в диапазоне 0..1")
        if not 0.0 < self.short_threshold <= 1.0:
            raise ValueError("порог для коротких фраз задаётся в диапазоне 0..1")
        if self.min_length < 1:
            raise ValueError("минимальная длина фразы для fuzzy — от одного символа")
        if self.short_length < self.min_length:
            raise ValueError("короткой считается фраза не длиннее обычной")

    @classmethod
    def from_config(cls, commands: CommandsConfig) -> MatcherSettings:
        """Build settings from the «Команды» tab, keeping the rest at defaults."""
        return replace(cls(), threshold=commands.fuzzy_threshold)

    def effective_threshold(self, length: int) -> float:
        """Similarity required of a phrase this long.

        The length taken is the shorter of the two strings being compared: it is
        the short side that makes a ratio meaningless, no matter what it is
        being compared against.
        """
        if length >= self.short_length:
            return self.threshold
        if length <= self.min_length:
            return self.short_threshold
        span = self.short_length - self.min_length
        travelled = (length - self.min_length) / span
        return self.short_threshold + (self.threshold - self.short_threshold) * travelled


def levenshtein(left: str, right: str, max_distance: int | None = None) -> int:
    """Edit distance between two strings, with an optional ceiling.

    Myers' bit-parallel algorithm rather than the textbook matrix: the whole
    column of the dynamic programming table is carried in two integers, and one
    character of the longer string costs a handful of machine words of shifting
    instead of a Python loop over the shorter one. On the phrase lengths this
    module sees it is some seven times faster than two-row DP, which is the
    difference between the fuzzy sweep fitting in the budget and not.

    ``max_distance`` gives an early exit. The distance to the text read so far
    can still fall by at most one per remaining character, so once it is further
    than that above the ceiling the answer cannot come back under it and the
    scan stops.

    Returns ``max_distance + 1`` when the ceiling is exceeded, which is not the
    real distance but compares correctly against it.
    """
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    if max_distance is not None:
        if max_distance < 0:
            return 1
        if abs(len(left) - len(right)) > max_distance:
            return max_distance + 1

    # The bit vectors are as wide as the first string, and the scan is over the
    # second; keeping the shorter one first costs a swap and saves a word of
    # arithmetic per character.
    if len(left) > len(right):
        left, right = right, left

    width = len(left)
    # ``peq[c]`` marks the positions of ``c`` in ``left``: the equality column
    # of the matrix, precomputed once instead of compared per cell.
    peq: dict[str, int] = {}
    for position, char in enumerate(left):
        peq[char] = peq.get(char, 0) | (1 << position)

    full = (1 << width) - 1
    last = 1 << (width - 1)
    positive = full  # cells where the column rises by one
    negative = 0  # cells where it falls
    distance = width
    remaining = len(right)

    for char in right:
        remaining -= 1
        equal = peq.get(char, 0)
        carry = equal | negative
        delta = (((carry & positive) + positive) & full) ^ positive | carry
        rise = negative | ~(delta | positive)
        fall = delta & positive
        if rise & last:
            distance += 1
        elif fall & last:
            distance -= 1
        if max_distance is not None and distance - remaining > max_distance:
            return max_distance + 1
        shifted = ((rise << 1) | 1) & full
        positive = ((fall << 1) & full) | ~(delta | shifted) & full
        negative = delta & shifted

    if max_distance is not None and distance > max_distance:
        return max_distance + 1
    return distance


def distance_ceiling(floor: float, length: int) -> int:
    """Edits a string of ``length`` may take and still score at least ``floor``.

    The one place the arithmetic lives, because two callers have to agree on it
    exactly: :func:`similarity` uses it to abandon a comparison, and the
    prefilter in :mod:`ayris.nlu.index` uses it to decide which comparisons to
    make at all. If the prefilter were even one edit stricter it would drop
    candidates the scorer would have accepted.
    """
    return int(length * (1.0 - floor))


def similarity(left: str, right: str, floor: float = 0.0) -> float:
    """Levenshtein ratio in ``0.0..1.0``: ``1 - distance / longer``.

    ``floor`` turns into a distance ceiling, so a pair that cannot reach the
    caller's threshold is abandoned mid-scan. The number returned in that case
    is only guaranteed to be below ``floor``.
    """
    if not left and not right:
        return 1.0
    longest = max(len(left), len(right))
    ceiling = distance_ceiling(floor, longest) if floor > 0.0 else None
    distance = levenshtein(left, right, ceiling)
    return 1.0 - distance / longest


def validate_pattern(pattern: str) -> str:
    """Check a user's regex, returning a Russian error message or ``""``.

    Called by the command editor before a trigger is saved. The matcher itself
    never raises on a broken pattern — it skips it — but a user who typed one
    deserves to hear about it while they can still fix it.
    """
    if not pattern.strip():
        return "шаблон не может быть пустым"
    try:
        re.compile(pattern)
    except re.error as exc:
        return f"ошибка в шаблоне: {exc.msg}"
    return ""


def compile_pattern(pattern: str) -> re.Pattern[str] | None:
    """Compile a trigger's regex against normalised text, or return ``None``.

    Two adjustments before compiling. ``ё`` is folded, because normalisation
    already removed it from the input side. A leading address is dropped for the
    same reason, with the ``^`` anchor put back if there was one. Case folding is
    left to :data:`re.IGNORECASE`: lowercasing the source would quietly turn
    ``\\S`` into ``\\s``, ``(?P<app>...)`` into ``(?p<app>...)`` — which does not
    parse at all — and invert half the pattern.

    A pattern that does not compile is logged and refused. Matching then carries
    on without it, which is the whole point — one bad trigger in the library must
    not take voice control down with it.
    """
    folded = fold_yo(pattern)
    stripped = _PATTERN_ADDRESS.sub(lambda m: m.group(1) or "", folded)
    source = stripped or folded
    try:
        return re.compile(source, re.IGNORECASE)
    except re.error as exc:
        _log.warning("Триггер пропущен, шаблон не компилируется: %s (%s)", pattern, exc.msg)
        return None


def trigger_from_db(
    row: DbTrigger,
    *,
    command_priority: int = 0,
    enabled: bool = True,
) -> Trigger | None:
    """Convert a stored trigger into the shape the matcher works with.

    Returns ``None`` for anything that is not a voice trigger — hotkeys, events
    and schedules fire commands too, but not through this module — and for a
    voice trigger with nothing to match against.

    The database keeps a trigger's weight in its ``priority`` column, which is
    what section 5.2 of the specification calls «приоритет (вес)»; the command's
    own priority is the one that outranks it, and it is passed in.
    """
    from ayris.core.models import TriggerType

    if row.type is not TriggerType.VOICE:
        return None
    payload = row.payload
    raw_pattern = payload.get("regex") or payload.get("phrase") or ""
    if not isinstance(raw_pattern, str) or not raw_pattern.strip():
        return None
    kind = TriggerKind.REGEX if payload.get("regex") else TriggerKind.PHRASE
    threshold = payload.get("threshold")
    weight = payload.get("weight")
    return Trigger(
        id=row.id if row.id is not None else 0,
        command_id=row.command_id,
        pattern=raw_pattern,
        kind=kind,
        fuzzy=row.fuzzy,
        threshold=float(threshold) if isinstance(threshold, int | float) else None,
        weight=float(weight) if isinstance(weight, int | float) else float(row.priority),
        priority=command_priority,
        enabled=enabled and bool(payload.get("enabled", True)),
    )


class Matcher:
    """Turns a phrase into the command it means.

    Holds no data of its own: the triggers live in the index, which can be
    rebuilt underneath a running matcher without locking anything, because every
    call takes one immutable snapshot and works on that.
    """

    __slots__ = ("_index", "_settings")

    def __init__(self, index: TriggerIndex, settings: MatcherSettings | None = None) -> None:
        self._index = index
        self._settings = settings or MatcherSettings()

    @classmethod
    def from_triggers(
        cls,
        triggers: Iterable[Trigger],
        settings: MatcherSettings | None = None,
    ) -> Matcher:
        """Build a matcher over a fixed set of triggers.

        Convenience for tests and for the plugin SDK. The import is local
        because the index imports this module for its data model, and a module
        level import here would close that circle.
        """
        from ayris.nlu.index import TriggerIndex

        index = TriggerIndex()
        index.replace_all(triggers)
        return cls(index, settings)

    @property
    def settings(self) -> MatcherSettings:
        """Current thresholds."""
        return self._settings

    @settings.setter
    def settings(self, value: MatcherSettings) -> None:
        """Apply new thresholds. Takes effect on the next phrase, not this one."""
        self._settings = value

    @property
    def index(self) -> TriggerIndex:
        """The index being matched against."""
        return self._index

    def match(self, text: str | NormalizedText) -> MatchResult | None:
        """Best match for a phrase, or ``None`` when nothing is close enough."""
        results = self.match_all(text, limit=1)
        return results[0] if results else None

    def match_all(self, text: str | NormalizedText, limit: int | None = None) -> list[MatchResult]:
        """Every trigger that fired, best first.

        Used by the DevTools tab and by the tests; the pipeline wants
        :meth:`match`. Fuzzy candidates are only collected when nothing matched
        exactly — an exact hit is the answer, and sweeping the library after
        finding one would spend milliseconds to change nothing.
        """
        phrase = normalize(text) if isinstance(text, str) else text
        if phrase.is_empty:
            return []
        snapshot = self._index.snapshot()
        results: list[MatchResult] = []
        results.extend(self._match_exact(snapshot, phrase))
        results.extend(self._match_regex(snapshot, phrase))
        if not results and self._settings.fuzzy_enabled:
            results.extend(self._match_fuzzy(snapshot, phrase, skip=results))
        results.sort(key=lambda item: item.sort_key, reverse=True)
        cap = self._settings.max_results if limit is None else limit
        return results[:cap]

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------

    def _match_exact(self, snapshot: IndexSnapshot, phrase: NormalizedText) -> list[MatchResult]:
        return [
            self._result(entry, 1.0, MatchKind.EXACT, phrase, matched=entry.text)
            for entry in snapshot.exact_candidates(phrase.text)
        ]

    def _match_regex(self, snapshot: IndexSnapshot, phrase: NormalizedText) -> list[MatchResult]:
        results: list[MatchResult] = []
        # The spoken form first: a pattern written with number words has to see
        # them. The digit form is a fallback, so ``\d+`` also works for a user
        # who says the number out loud.
        subjects = (phrase.spoken, phrase.text) if phrase.has_numerals else (phrase.text,)
        for entry in snapshot.regex_triggers():
            pattern = entry.regex
            if pattern is None:
                continue
            for subject in subjects:
                found = pattern.search(subject)
                if found is None:
                    continue
                covered = found.end() - found.start()
                score = covered / len(subject) if subject else 0.0
                groups = {
                    name: value for name, value in found.groupdict().items() if value is not None
                }
                results.append(
                    self._result(
                        entry,
                        min(score, 1.0),
                        MatchKind.REGEX,
                        phrase,
                        matched=subject[found.start() : found.end()],
                        groups=groups,
                    )
                )
                break
        return results

    def _match_fuzzy(
        self,
        snapshot: IndexSnapshot,
        phrase: NormalizedText,
        skip: Sequence[MatchResult],
    ) -> list[MatchResult]:
        settings = self._settings
        if not settings.fuzzy_enabled:
            return []
        query = phrase.text
        if len(query) < settings.min_length:
            return []
        seen = {result.trigger_id for result in skip}
        results: list[MatchResult] = []
        for entry in snapshot.fuzzy_candidates(query, settings.threshold):
            if entry.trigger.id in seen:
                continue
            required = entry.trigger.threshold
            if required is None:
                required = settings.effective_threshold(min(len(query), entry.length))
            score = similarity(query, entry.text, required)
            if score >= required:
                results.append(
                    self._result(entry, score, MatchKind.FUZZY, phrase, matched=entry.text)
                )
        return results

    @staticmethod
    def _result(
        entry: IndexedTrigger,
        score: float,
        kind: MatchKind,
        phrase: NormalizedText,
        *,
        matched: str,
        groups: dict[str, str] | None = None,
    ) -> MatchResult:
        trigger = entry.trigger
        return MatchResult(
            command_id=trigger.command_id,
            trigger_id=trigger.id,
            score=score,
            kind=kind,
            raw_groups=groups or {},
            normalized_text=phrase.text,
            spoken_text=phrase.spoken,
            matched_text=matched,
            pattern=trigger.pattern,
            weight=trigger.weight,
            priority=trigger.priority,
        )
