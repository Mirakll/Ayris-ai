"""The in-memory trigger index: exact map, regex list, fuzzy prefilter.

Matching cannot read the database. A phrase arrives every few seconds, the
answer is needed inside a millisecond, and SQLite on a disk under a Windows
antivirus is not that. So the whole trigger library lives here, in memory, and
storage is read only when something changes.

**What the index is.** Three structures over the same triggers, one per
strategy:

* a dict from normalised phrase to the triggers spelled that way — synonyms
  collide there, which is correct: they compete in the matcher;
* a list of compiled regexes, because a pattern cannot be prefiltered;
* a character-3-gram posting list, which is the only reason fuzzy matching fits
  in the budget.

**Why the prefilter matters.** Comparing a phrase against a thousand triggers is
a thousand Levenshtein scans — milliseconds, on the heels of the audio thread.
Two strings within a small edit distance must share most of their 3-grams, so
:meth:`IndexSnapshot.fuzzy_candidates` counts shared grams and hands over only
the triggers that could still clear the threshold. The count is a sound bound
rather than a guess: one edit destroys at most three grams, so a candidate
sharing fewer than ``grams - 3 × edits`` of them cannot reach the threshold and
is dropped without a comparison. Length settles the rest — a 40-character phrase
is not a near-miss of a 6-character one — and the edit budget itself is computed
per candidate from :func:`ayris.nlu.matcher.distance_ceiling`, the same function
the scorer uses, so the filter is exactly as strict as the score and not one
edit more.

**Why snapshots.** The command editor rebuilds the index while the pipeline
matches against it. Rather than lock, every mutation publishes a new immutable
:class:`IndexSnapshot`, and a matcher takes one at the top of the call and works
on it to the end. A phrase that arrived a microsecond before a rebuild is
matched against the library as it was, which is the behaviour to want, and the
read path takes no lock at all.

**Incremental rebuilds.** :meth:`TriggerIndex.replace_all` is the cold path, for
startup. When one command is edited — the common case, and the one the user is
watching — :meth:`TriggerIndex.update_command` re-normalises only that command's
phrases and recompiles only its patterns; :meth:`TriggerIndex.set_priority` and
:meth:`TriggerIndex.set_enabled` recompile nothing at all. :meth:`bind` wires
that to :class:`~ayris.core.events.CommandsChanged`, so saving a command in the
library tab updates the index and nothing else has to know it happened.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final

from ayris.core.events import COMMANDS_CHANGE_DELETED, COMMANDS_CHANGE_RELOADED, CommandsChanged
from ayris.nlu.matcher import Trigger, TriggerKind, compile_pattern, distance_ceiling
from ayris.nlu.normalize import normalize
from ayris.nlu.slots import SlotTemplate, SlotTemplateError, compile_slots
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    import re
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ayris.core.events import EventBus, Unsubscribe
    from ayris.nlu.matcher import MatchResult
    from ayris.nlu.slot_types import SlotContext, SlotTypeRegistry
    from ayris.nlu.slots import SlotSet
    from ayris.nlu.trigger_filters import TriggerConditions, TriggerPredicate

__all__ = [
    "GRAM_SIZE",
    "IndexSnapshot",
    "IndexStats",
    "IndexedTrigger",
    "TriggerIndex",
    "TriggerLoader",
    "text_grams",
]

_log = get_logger(__name__)

#: Character n-gram width for the prefilter. Three is the usual compromise: two
#: is so common that nothing gets filtered, four does not survive a single typo
#: in a short word — precisely the case fuzzy matching exists for.
GRAM_SIZE: Final = 3

#: Called by :meth:`TriggerIndex.bind` to fetch triggers after a change.
#: The argument is the command that changed, or ``None`` for «reload the lot».
TriggerLoader = "Callable[[int | None], Sequence[Trigger]]"


def text_grams(text: str) -> frozenset[str]:
    """Character 3-grams of a phrase, padded so short words still produce some.

    Padding matters: without it a two-character phrase has no 3-grams at all and
    would be invisible to the prefilter. Spaces are kept — a gram spanning a word
    boundary carries word order, which is what separates «браузер открой» from
    «открой браузер».
    """
    if not text:
        return frozenset()
    padded = f"  {text}  "
    return frozenset(padded[i : i + GRAM_SIZE] for i in range(len(padded) - GRAM_SIZE + 1))


@dataclass(frozen=True, slots=True)
class IndexedTrigger:
    """A trigger with everything precomputed that matching would otherwise redo.

    Normalising a phrase and compiling a regex are cheap once and ruinous a
    thousand times per utterance, so both happen here, at index time.

    ``regex`` is ``None`` for a phrase trigger and for a pattern that failed to
    compile: :func:`ayris.nlu.matcher.compile_pattern` logs that one and the
    entry is left out of the snapshot, so a broken pattern costs its own trigger
    and nothing else.

    ``slots`` is set only for a template trigger and is what turns the groups a
    match captured into typed values. It lives here rather than being looked up
    later because compiling a template validates it, and validating it once per
    utterance is exactly the work this class exists to avoid.
    """

    trigger: Trigger
    text: str
    spoken: str
    grams: frozenset[str] = field(default_factory=frozenset)
    regex: re.Pattern[str] | None = None
    slots: SlotTemplate | None = None

    @property
    def conditions(self) -> TriggerConditions:
        """When this trigger may fire, as parsed when it was loaded.

        Forwarded from the trigger rather than stored a second time: the matcher's
        predicate reads it per candidate, and a property on a slotted dataclass is
        an attribute lookup either way. Keeping one copy also means
        :meth:`TriggerIndex.set_priority` and friends, which rewrite the trigger in
        place, cannot leave the conditions pointing at a stale one.
        """
        return self.trigger.conditions

    @property
    def is_conditional(self) -> bool:
        """Whether anything has to hold for this trigger to be a candidate."""
        return not self.trigger.conditions.is_empty

    @property
    def length(self) -> int:
        """Length of the normalised phrase, for the fuzzy threshold ramp."""
        return len(self.text)

    @property
    def is_regex(self) -> bool:
        """Whether this entry is matched by pattern rather than by text.

        True for a template as well: a template *is* a regex by the time it
        reaches the matcher, and the whole point of compiling it at index time is
        that nothing downstream has to know the difference.
        """
        return self.trigger.kind in (TriggerKind.REGEX, TriggerKind.TEMPLATE)

    @property
    def has_slots(self) -> bool:
        """Whether a match on this entry can be turned into typed slot values."""
        return self.slots is not None


@dataclass(frozen=True, slots=True)
class IndexStats:
    """What the index holds. Shown in DevTools, asserted on in the tests."""

    triggers: int
    commands: int
    exact_phrases: int
    regex_triggers: int
    fuzzy_triggers: int
    grams: int
    generation: int
    rebuilt_entries: int


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    """An immutable view of the library as it was at one moment.

    Handed out by :meth:`TriggerIndex.snapshot` and never mutated afterwards,
    which is what lets the matcher read while the editor writes.
    """

    exact: dict[str, tuple[IndexedTrigger, ...]]
    regexes: tuple[IndexedTrigger, ...]
    postings: dict[str, tuple[int, ...]]
    entries: dict[int, IndexedTrigger]
    generation: int
    templates: dict[int, IndexedTrigger] = field(default_factory=dict)

    def exact_candidates(
        self,
        text: str,
        predicate: TriggerPredicate | None = None,
    ) -> tuple[IndexedTrigger, ...]:
        """Triggers spelled exactly like this. One hash, usually one result.

        ``predicate`` drops the ones whose conditions do not hold — the trigger
        written for Photoshop when the foreground window is a text editor. It is
        applied here rather than by the caller so that «сохрани» with two spellings
        behaves the same way whichever of them the user said.
        """
        found = self.exact.get(text, ())
        if predicate is None or not found:
            return found
        return tuple(entry for entry in found if predicate(entry))

    def conditional_triggers(self) -> tuple[IndexedTrigger, ...]:
        """Every published entry carrying a condition, in trigger-id order.

        For DevTools and for the tests: «why did this trigger never fire» is
        answered by the conditions on it, and finding them by walking the whole
        library is what this saves.
        """
        return tuple(entry for entry in self.all_triggers() if entry.is_conditional)

    def slot_template(self, trigger_id: int) -> SlotTemplate | None:
        """The template a trigger was compiled from, or ``None`` when it has none.

        ``entries`` cannot answer this: it holds only the phrase triggers, since
        its purpose is the fuzzy prefilter. Templates therefore get their own map,
        keyed the same way — a dict lookup per matched phrase, not a scan of every
        pattern in the library.
        """
        entry = self.templates.get(trigger_id)
        return None if entry is None else entry.slots

    def bind_slots(
        self,
        result: MatchResult,
        context: SlotContext | None = None,
    ) -> SlotSet | None:
        """Typed slot values for a match, or ``None`` when the trigger has no slots.

        The seam between matching and dispatch. The matcher has already run the
        regex and left its named groups in
        :attr:`~ayris.nlu.matcher.MatchResult.raw_groups`, so this only parses —
        matching a second time would be wasted work and a second chance to
        disagree with the result being dispatched.
        """
        template = self.slot_template(result.trigger_id)
        return None if template is None else template.bind(result.raw_groups, context)

    def regex_triggers(
        self,
        predicate: TriggerPredicate | None = None,
    ) -> tuple[IndexedTrigger, ...]:
        """Every compiled pattern, in index order, minus the inactive ones.

        Filtering here rather than after the search is what saves the work: a
        pattern that cannot fire should not be run, and the regex sweep is the
        second most expensive thing the matcher does.
        """
        if predicate is None:
            return self.regexes
        return tuple(entry for entry in self.regexes if predicate(entry))

    def fuzzy_candidates(
        self,
        text: str,
        threshold: float,
        predicate: TriggerPredicate | None = None,
    ) -> list[IndexedTrigger]:
        """Triggers that could still clear ``threshold``, cheapest test first.

        Both tests share one number: ``edits``, the distance the pair may be
        apart and still score ``threshold``, taken from
        :func:`~ayris.nlu.matcher.distance_ceiling` — the very function the
        scorer uses, so the filter is exactly as strict as the score and not one
        edit more.

        Length first, because it settles on integers: a length difference is
        already that many insertions, so a candidate further than ``edits`` from
        the query in length is impossible regardless of its content.

        Then shared 3-grams. An edit destroys at most :data:`GRAM_SIZE` grams,
        so a pair sharing fewer than ``grams - GRAM_SIZE × edits`` cannot reach
        the threshold. The count is taken against the larger of the two gram
        sets, which is what makes it bite on a library of phrases that differ
        only in a tail. That is a bound, not a heuristic: nothing matchable is
        lost, and on a real library it removes the great majority of the
        candidates before any distance is computed.

        ``threshold`` must be the *lowest* any trigger could ask for, since a
        per-trigger override may be looser than the global setting; the matcher
        passes its base threshold, and per-trigger overrides are checked after.

        ``predicate`` drops the triggers whose conditions do not hold, and it runs
        last of the three tests on purpose: it is a Python call, while the other
        two are integer comparisons, so it is cheapest to ask it only about the
        handful of candidates that survived the arithmetic.
        """
        grams = text_grams(text)
        if not grams:
            return []
        query_length = len(text)
        query_grams = len(grams)

        # Counter.update over a tuple of ids runs in C; the same loop written in
        # Python over a defaultdict is several times slower, and this is the hot
        # path of the whole module.
        counts: Counter[int] = Counter()
        for gram in grams:
            ids = self.postings.get(gram)
            if ids:
                counts.update(ids)

        entries = self.entries
        candidates: list[IndexedTrigger] = []
        for trigger_id, count in counts.items():
            entry = entries[trigger_id]
            length = entry.length
            edits = distance_ceiling(threshold, length if length > query_length else query_length)
            if length - query_length > edits or query_length - length > edits:
                continue
            if count < max(query_grams, len(entry.grams)) - GRAM_SIZE * edits:
                continue
            if predicate is not None and not predicate(entry):
                continue
            candidates.append(entry)
        # Longest first: a long trigger is the likelier intent behind a long
        # phrase, and it lets the caller's early exit tighten sooner.
        candidates.sort(key=lambda item: (-item.length, item.trigger.id))
        return candidates

    def all_triggers(self) -> Iterator[IndexedTrigger]:
        """Every live entry, exact and regex alike, in trigger-id order."""
        for trigger_id in sorted(self.entries):
            yield self.entries[trigger_id]
        yield from self.regexes


class TriggerIndex:
    """The mutable side: builds entries, publishes snapshots.

    Writes take a lock, reads do not need one — :meth:`snapshot` hands back an
    object no writer will touch again. Callers are the command library (on every
    change), the plugin loader (when a plugin registers commands) and the tests.
    """

    __slots__ = (
        "_by_command",
        "_entries",
        "_generation",
        "_lock",
        "_rebuilt",
        "_slot_types",
        "_snapshot",
    )

    def __init__(self, slot_types: SlotTypeRegistry | None = None) -> None:
        """Build an empty index.

        Args:
            slot_types: Registry the templates are compiled against. A plugin that
                registered its own slot type passes the registry holding it;
                ``None`` means the shipped types. Fixed at construction rather
                than read per compile, because a template that compiled against
                one registry and matches against another is a bug with no good
                error message.
        """
        self._lock = threading.RLock()
        self._entries: dict[int, IndexedTrigger] = {}
        self._by_command: defaultdict[int, set[int]] = defaultdict(set)
        self._generation = 0
        self._rebuilt = 0
        self._slot_types = slot_types
        self._snapshot = IndexSnapshot(exact={}, regexes=(), postings={}, entries={}, generation=0)

    # ------------------------------------------------------------------
    # reading
    # ------------------------------------------------------------------

    def snapshot(self) -> IndexSnapshot:
        """The current view. Cheap: the object already exists."""
        return self._snapshot

    def stats(self) -> IndexStats:
        """Counters for DevTools and for the tests.

        ``rebuilt_entries`` counts the triggers this index has normalised or
        compiled since it was created. It is how a test tells an incremental
        update from a full rebuild: editing one command of a thousand must move
        it by the size of that command, not by a thousand.
        """
        with self._lock:
            snapshot = self._snapshot
            fuzzy = sum(
                1 for entry in self._entries.values() if entry.trigger.fuzzy and not entry.is_regex
            )
            return IndexStats(
                triggers=len(self._entries),
                commands=sum(1 for ids in self._by_command.values() if ids),
                exact_phrases=len(snapshot.exact),
                regex_triggers=len(snapshot.regexes),
                fuzzy_triggers=fuzzy,
                grams=len(snapshot.postings),
                generation=snapshot.generation,
                rebuilt_entries=self._rebuilt,
            )

    @property
    def generation(self) -> int:
        """Bumped on every change. Lets a caller tell whether it must re-read."""
        return self._snapshot.generation

    def __len__(self) -> int:
        return len(self._entries)

    def triggers_for(self, command_id: int) -> tuple[Trigger, ...]:
        """The triggers of one command, as stored."""
        with self._lock:
            ids = sorted(self._by_command.get(command_id, ()))
            return tuple(self._entries[tid].trigger for tid in ids if tid in self._entries)

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def replace_all(self, triggers: Iterable[Trigger]) -> None:
        """Drop everything and rebuild. The startup path, and the import path."""
        with self._lock:
            self._entries.clear()
            self._by_command.clear()
            for trigger in triggers:
                self._insert(trigger)
            self._publish()

    def update_command(self, command_id: int, triggers: Iterable[Trigger]) -> None:
        """Replace one command's triggers, leaving the rest of the index alone.

        The incremental path, and the one that runs while the user watches: only
        this command's phrases are re-normalised, only its patterns recompiled.
        """
        with self._lock:
            self._forget_command(command_id)
            for trigger in triggers:
                self._insert(replace(trigger, command_id=command_id))
            self._publish()

    def remove_command(self, command_id: int) -> None:
        """Forget a command entirely — deleted, or moved to another profile."""
        with self._lock:
            if not self._by_command.get(command_id):
                return
            self._forget_command(command_id)
            self._publish()

    def add(self, trigger: Trigger) -> None:
        """Add or replace a single trigger by id."""
        with self._lock:
            self._insert(trigger)
            self._publish()

    def remove(self, trigger_id: int) -> None:
        """Drop a single trigger by id. A no-op when it is not there."""
        with self._lock:
            entry = self._entries.pop(trigger_id, None)
            if entry is None:
                return
            self._by_command[entry.trigger.command_id].discard(trigger_id)
            self._publish()

    def set_priority(self, command_id: int, priority: int) -> None:
        """Repoint a command's priority without recompiling its triggers.

        Reordering commands is a drag in a list, so it happens repeatedly and
        must stay free. Nothing derived depends on the priority — it is only read
        while sorting results — so the entries are rewritten in place.
        """
        self._retag(command_id, priority=priority)

    def set_enabled(self, command_id: int, enabled: bool) -> None:
        """Enable or disable a command's triggers, keeping them indexed.

        A disabled trigger stays in the index so re-enabling it costs nothing,
        but it is left out of the published snapshot and cannot match.
        """
        self._retag(command_id, enabled=enabled)

    # ------------------------------------------------------------------
    # staying in step with the library
    # ------------------------------------------------------------------

    def bind(self, bus: EventBus, loader: Callable[[int | None], Sequence[Trigger]]) -> Unsubscribe:
        """Follow :class:`~ayris.core.events.CommandsChanged` on ``bus``.

        ``loader`` is asked for the triggers of the command that changed, or —
        when the event names no command, as a profile switch or an import does
        not — for the whole library. It is the only place the index touches
        storage, and it runs on the delivery thread, never inside a match.

        A loader that raises is logged and ignored: a failed reload leaves the
        previous snapshot in place, which still matches the commands the user
        had a second ago. Returns the unsubscribe callable from the bus.
        """

        def on_change(event: CommandsChanged) -> None:
            try:
                if event.command_id is None or event.change == COMMANDS_CHANGE_RELOADED:
                    self.replace_all(loader(None))
                elif event.change == COMMANDS_CHANGE_DELETED:
                    self.remove_command(event.command_id)
                else:
                    self.update_command(event.command_id, loader(event.command_id))
            except Exception:
                _log.exception(
                    "Не удалось обновить индекс триггеров после изменения команд (%s, id=%s)",
                    event.change,
                    event.command_id,
                )

        # Held strongly: ``on_change`` is a closure, and a weak subscription to
        # one dies the moment ``bind`` returns.
        return bus.subscribe(CommandsChanged, on_change, weak=False)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _retag(
        self,
        command_id: int,
        *,
        priority: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        with self._lock:
            ids = self._by_command.get(command_id)
            if not ids:
                return
            changed = False
            for trigger_id in tuple(ids):
                entry = self._entries.get(trigger_id)
                if entry is None:
                    continue
                trigger = entry.trigger
                new_priority = trigger.priority if priority is None else priority
                new_enabled = trigger.enabled if enabled is None else enabled
                if new_priority == trigger.priority and new_enabled == trigger.enabled:
                    continue
                self._entries[trigger_id] = replace(
                    entry,
                    trigger=replace(trigger, priority=new_priority, enabled=new_enabled),
                )
                changed = True
            if changed:
                self._publish()

    def _forget_command(self, command_id: int) -> None:
        for trigger_id in self._by_command.pop(command_id, set()):
            self._entries.pop(trigger_id, None)

    def _insert(self, trigger: Trigger) -> None:
        entry = self._build(trigger)
        self._rebuilt += 1
        if entry is None:
            return
        previous = self._entries.get(trigger.id)
        if previous is not None and previous.trigger.command_id != trigger.command_id:
            self._by_command[previous.trigger.command_id].discard(trigger.id)
        self._entries[trigger.id] = entry
        self._by_command[trigger.command_id].add(trigger.id)

    def _build(self, trigger: Trigger) -> IndexedTrigger | None:
        """Precompute a trigger's matchable forms, or reject it.

        A regex is not normalised — its punctuation is syntax — but a phrase is,
        so that the trigger and the recognised text meet in the same shape. A
        template is normalised too, and by the compiler rather than here: the
        literal stretches between its slots have to be folded the same way the
        phrase was, and only :func:`~ayris.nlu.slots.compile_slots` knows which
        stretches those are.
        """
        if trigger.kind is TriggerKind.TEMPLATE:
            return self._build_template(trigger)
        if trigger.kind is TriggerKind.REGEX:
            return IndexedTrigger(
                trigger=trigger,
                text=trigger.pattern,
                spoken=trigger.pattern,
                regex=compile_pattern(trigger.pattern),
            )
        phrase = normalize(trigger.pattern)
        if phrase.is_empty:
            return None
        return IndexedTrigger(
            trigger=trigger,
            text=phrase.text,
            spoken=phrase.spoken,
            grams=text_grams(phrase.text) if trigger.fuzzy else frozenset(),
        )

    def _build_template(self, trigger: Trigger) -> IndexedTrigger | None:
        """Compile a template trigger, or drop it with a line in the log.

        A template the user wrote badly costs its own trigger and nothing else —
        the same contract :func:`~ayris.nlu.matcher.compile_pattern` has for a
        broken regex. Raising here would take down the whole index rebuild, and
        the rebuild is triggered by saving *some other* command.
        """
        try:
            template = compile_slots(trigger.pattern, self._slot_types)
        except SlotTemplateError as exc:
            _log.warning(
                "trigger %d has an invalid slot template %r: %s",
                trigger.id,
                trigger.pattern,
                exc,
            )
            return None
        return IndexedTrigger(
            trigger=trigger,
            text=trigger.pattern,
            spoken=trigger.pattern,
            regex=template.regex,
            slots=template,
        )

    def _publish(self) -> None:
        """Rebuild the read-side structures and swap them in atomically.

        Called once per mutation rather than once per trigger: rebuilding the
        maps for a thousand triggers is a few milliseconds of dictionary work —
        no normalising, no compiling — and doing it inside the insert loop would
        make :meth:`replace_all` quadratic.
        """
        exact: defaultdict[str, list[IndexedTrigger]] = defaultdict(list)
        regexes: list[IndexedTrigger] = []
        postings: defaultdict[str, list[int]] = defaultdict(list)
        entries: dict[int, IndexedTrigger] = {}
        templates: dict[int, IndexedTrigger] = {}

        # Sorted by id, so the published order — and every tie-break that falls
        # through to it — is the same after every rebuild.
        for trigger_id in sorted(self._entries):
            entry = self._entries[trigger_id]
            if not entry.trigger.enabled:
                continue
            if entry.has_slots:
                templates[trigger_id] = entry
            if entry.is_regex:
                if entry.regex is not None:
                    regexes.append(entry)
                continue
            entries[trigger_id] = entry
            exact[entry.text].append(entry)
            if not entry.trigger.fuzzy:
                continue
            for gram in entry.grams:
                postings[gram].append(trigger_id)

        self._generation += 1
        self._snapshot = IndexSnapshot(
            exact={text: tuple(items) for text, items in exact.items()},
            regexes=tuple(regexes),
            postings={gram: tuple(items) for gram, items in postings.items()},
            entries=entries,
            generation=self._generation,
            templates=templates,
        )
