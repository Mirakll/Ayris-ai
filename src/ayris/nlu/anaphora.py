"""Turning «закрой его» into «закрой гугл хром».

A pronoun is a promise that both sides remember the same thing. The remembering
is :mod:`ayris.nlu.context`; this module is the other half — finding the word
that stands in for something, deciding what it stands for, and putting the name
back into the phrase so that the rest of the pipeline never has to know a pronoun
was involved.

**Rewriting, not a special path.** The obvious design is to match «закрой его»
against a trigger of its own and then fill the slot from the context. That means
every command that can be referred to needs a second, pronoun-shaped trigger, and
the slot layer needs a branch for values that come from memory rather than from
the phrase. Rewriting the utterance instead — «закрой его» → «закрой гугл хром» —
costs one string substitution and makes the existing trigger «закрой {app}» and
the existing :class:`~ayris.nlu.slot_types.AppType` do the work unchanged. What
comes out is normal text, so it goes through :func:`~ayris.nlu.normalize.normalize`
and the matcher exactly like anything the user actually said.

**The name goes in, not the id.** :attr:`~ayris.nlu.context.ContextObject.value`
holds the machine-readable half — ``chrome``, a path, a URL — and substituting it
would produce text no trigger is written against. The *name* is what the user
said in the first place, so putting it back yields a phrase they could have said
themselves, and the slot parser resolves it the same way it resolved the original.

**Nothing is resolved silently wrong.** Three outcomes, and the caller has to
handle the third: there was no pronoun (:attr:`AnaphoraStatus.ABSENT`), there was
one and the context could answer it (:attr:`~AnaphoraStatus.RESOLVED`), or there
was one and it could not (:attr:`~AnaphoraStatus.UNRESOLVED`) — the TTL expired,
or nothing of a suitable kind was ever mentioned. The third case carries a Russian
:attr:`Anaphora.question` to ask out loud, because an assistant that closes a
random window when «его» meant nothing is worse than one that asks.

**Gender is a hint, never a filter.** «его» prefers a masculine name and «её» a
feminine one, which is what disambiguates two remembered objects, but a mismatch
never rejects a candidate: people say «закрой его» about «программа», and
:func:`~ayris.nlu.context.guess_gender` is guessing from a word ending anyway.

**Order matters at the call site.** «сделай то же самое» is a request to repeat
the last *command*, and «открой то же самое» a reference to the last *object*.
This module reads the second meaning, so :mod:`ayris.nlu.followup` gets the
phrase first: a repeat is recognised by the whole utterance, an anaphora by one
word inside it, and the broader reading has to be tried before the narrower one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.nlu.context import Gender, ObjectKind
from ayris.nlu.normalize import NormalizedText, fold_letters, normalize

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ayris.nlu.context import ContextObject, ContextSnapshot

__all__ = [
    "PLACE_KINDS",
    "PRONOUNS",
    "Anaphora",
    "AnaphoraStatus",
    "Pronoun",
    "PronounMention",
    "find_mention",
    "is_pronoun",
    "resolve_anaphora",
    "resolve_reference",
]

#: What «там», «туда» and «оттуда» can point at. A place is somewhere one can be
#: sent — a window, a file, an address — and never a volume level or a piece of
#: dictated text, which is the one restriction worth encoding: «сохрани туда»
#: resolving to the last thing the user dictated would write a file nobody asked
#: for.
PLACE_KINDS: Final[tuple[ObjectKind, ...]] = (
    ObjectKind.URL,
    ObjectKind.FILE,
    ObjectKind.WINDOW,
    ObjectKind.APP,
)

#: Asked when a pronoun has nothing to point at. Deliberately not «что закрыть?»:
#: this module does not know the verb, and a question that guesses it wrongly is
#: worse than one that is merely general.
QUESTION_OBJECT: Final = "Не понял, о чём речь. Уточни, что именно."
#: The same, for a pronoun of place.
QUESTION_PLACE: Final = "Не понял, куда именно. Уточни, пожалуйста."


class AnaphoraStatus(StrEnum):
    """How :func:`resolve_anaphora` got on."""

    #: No pronoun in the phrase. The text comes back untouched.
    ABSENT = "absent"
    #: A pronoun was found and the context could say what it meant.
    RESOLVED = "resolved"
    #: A pronoun was found and nothing suitable is remembered. Ask.
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class Pronoun:
    """One word (or short phrase) that stands in for something remembered.

    ``gender`` is the agreement the word suggests, or ``None`` when it suggests
    none — «им» is dative singular masculine and dative plural at once, and
    pretending otherwise would make the hint actively misleading.

    ``kinds`` narrows what the word can reach. Empty means «anything», which is
    right for «его» and wrong for «туда»; see :data:`PLACE_KINDS`.
    """

    words: tuple[str, ...]
    gender: Gender | None = None
    kinds: tuple[ObjectKind, ...] = ()
    place: bool = False

    @property
    def text(self) -> str:
        """The pronoun as written, for logs and for the question."""
        return " ".join(self.words)

    @property
    def question(self) -> str:
        """What to ask when this pronoun cannot be resolved."""
        return QUESTION_PLACE if self.place else QUESTION_OBJECT


def _pronoun(
    text: str,
    gender: Gender | None = None,
    kinds: tuple[ObjectKind, ...] = (),
    *,
    place: bool = False,
) -> Pronoun:
    return Pronoun(words=tuple(text.split()), gender=gender, kinds=kinds, place=place)


#: Every word that can carry a reference, with the agreement it suggests.
#:
#: The oblique forms are here because speech is oblique: «закрой его», «сделай с
#: ним», «открой в ней». The demonstratives are the risky half — «это» is a very
#: common word that is usually not a reference at all — and they are safe only
#: because of when this module runs: the pipeline resolves an anaphora after the
#: phrase failed to match anything on its own, so «что это такое» has already
#: been answered by the trigger it hit.
PRONOUNS: Final[tuple[Pronoun, ...]] = (
    # masculine / neuter
    _pronoun("его", Gender.MASCULINE),
    _pronoun("него", Gender.MASCULINE),
    _pronoun("ему", Gender.MASCULINE),
    _pronoun("нему", Gender.MASCULINE),
    _pronoun("нем", Gender.MASCULINE),
    # feminine
    _pronoun("ее", Gender.FEMININE),
    _pronoun("нее", Gender.FEMININE),
    _pronoun("ей", Gender.FEMININE),
    _pronoun("ней", Gender.FEMININE),
    # plural
    _pronoun("их", Gender.PLURAL),
    _pronoun("них", Gender.PLURAL),
    # ambiguous number and gender: no hint at all
    _pronoun("им"),
    _pronoun("ним"),
    _pronoun("ими"),
    _pronoun("ними"),
    # demonstratives
    _pronoun("это", Gender.NEUTER),
    _pronoun("этот", Gender.MASCULINE),
    _pronoun("этого", Gender.MASCULINE),
    _pronoun("эту", Gender.FEMININE),
    _pronoun("эта", Gender.FEMININE),
    _pronoun("эти", Gender.PLURAL),
    _pronoun("этих", Gender.PLURAL),
    _pronoun("тот", Gender.MASCULINE),
    _pronoun("того", Gender.MASCULINE),
    _pronoun("та", Gender.FEMININE),
    _pronoun("ту", Gender.FEMININE),
    _pronoun("те", Gender.PLURAL),
    _pronoun("то же самое", Gender.NEUTER),
    _pronoun("то же", Gender.NEUTER),
    # place
    _pronoun("там", None, PLACE_KINDS, place=True),
    _pronoun("туда", None, PLACE_KINDS, place=True),
    _pronoun("оттуда", None, PLACE_KINDS, place=True),
)

#: Pronouns by first word, longest phrase first.
#:
#: Built once and keyed on the first word so that a phrase of ordinary words
#: costs one dictionary miss per word instead of a scan of the table. The
#: ordering inside a bucket is what makes «то же самое» win over «то же»: a
#: shorter match that is a prefix of a longer one would otherwise leave «самое»
#: dangling in the rewritten phrase.
_BY_FIRST_WORD: Final[Mapping[str, tuple[Pronoun, ...]]] = {
    first: tuple(
        sorted(
            (item for item in PRONOUNS if item.words[0] == first),
            key=lambda item: len(item.words),
            reverse=True,
        )
    )
    for first in {item.words[0] for item in PRONOUNS}
}


def is_pronoun(word: str) -> bool:
    """Whether a single word can carry a reference on its own.

    Used by the slot layer: a slot that captured «его» captured a pronoun, not
    the name of a program, and asking :class:`~ayris.nlu.slot_types.AppType` to
    resolve it would produce a confident match on whatever is closest.
    """
    folded = fold_letters(word).strip()
    return any(len(candidate.words) == 1 for candidate in _BY_FIRST_WORD.get(folded, ()))


@dataclass(frozen=True, slots=True)
class PronounMention:
    """Where a pronoun sits in a phrase, in words."""

    pronoun: Pronoun
    start: int
    length: int

    @property
    def end(self) -> int:
        """Index one past the last word of the mention."""
        return self.start + self.length


def find_mention(text: str | NormalizedText | Sequence[str]) -> PronounMention | None:
    """The first pronoun in a phrase, or ``None``.

    The first rather than the best: a phrase with two references in it —
    «перенеси его туда» — is beyond what one remembered object can answer, and
    resolving the first is both what the user most likely meant and what leaves
    the second word visibly unresolved instead of silently wrong.
    """
    words = _words(text)
    for position, word in enumerate(words):
        for candidate in _BY_FIRST_WORD.get(word, ()):
            length = len(candidate.words)
            if tuple(words[position : position + length]) == candidate.words:
                return PronounMention(pronoun=candidate, start=position, length=length)
    return None


@dataclass(frozen=True, slots=True)
class Anaphora:
    """What became of a phrase that may have contained a pronoun."""

    status: AnaphoraStatus
    text: str
    original: str
    mention: PronounMention | None = None
    object: ContextObject | None = None
    replacement: str = ""

    @property
    def resolved(self) -> bool:
        """Whether a pronoun was replaced. ``False`` also when there was none."""
        return self.status is AnaphoraStatus.RESOLVED

    @property
    def needs_question(self) -> bool:
        """Whether the user has to be asked what they meant."""
        return self.status is AnaphoraStatus.UNRESOLVED

    @property
    def question(self) -> str:
        """What to ask, or ``""`` when nothing needs asking."""
        if self.status is not AnaphoraStatus.UNRESOLVED or self.mention is None:
            return ""
        return self.mention.pronoun.question


def resolve_anaphora(
    text: str | NormalizedText,
    snapshot: ContextSnapshot,
    *,
    kinds: Iterable[ObjectKind] | None = None,
) -> Anaphora:
    """Replace the first pronoun in ``text`` with what the context says it means.

    Args:
        text: The utterance, raw or already normalised. Normalisation happens
            here when it has not happened yet, because the pronoun table is
            written in folded lower case and the substitution has to land on word
            boundaries that survive it.
        snapshot: The context, taken by
            :meth:`~ayris.nlu.context.DialogContext.snapshot`. A snapshot rather
            than the live object on purpose: resolving reads the object list
            twice, and it must be the same list both times.
        kinds: Narrows what the pronoun may resolve to, on top of what the
            pronoun itself allows. The caller that already knows the phrase wants
            a program — because the only trigger left is «закрой {app}» — passes
            ``(ObjectKind.APP,)`` and stops «закрой его» from resolving to the
            last URL.

    Returns:
        An :class:`Anaphora`. Its ``text`` is the rewritten phrase when something
        was resolved, and the input otherwise, so a caller that ignores the
        status still has a usable string.
    """
    normalized = normalize(text) if isinstance(text, str) else text
    words = list(normalized.words)
    original = " ".join(words)
    mention = find_mention(words)
    if mention is None:
        return Anaphora(status=AnaphoraStatus.ABSENT, text=original, original=original)
    target = _pick(snapshot, mention.pronoun, kinds)
    if target is None:
        return Anaphora(
            status=AnaphoraStatus.UNRESOLVED,
            text=original,
            original=original,
            mention=mention,
        )
    rewritten = [*words[: mention.start], target.name, *words[mention.end :]]
    return Anaphora(
        status=AnaphoraStatus.RESOLVED,
        text=" ".join(word for word in rewritten if word),
        original=original,
        mention=mention,
        object=target,
        replacement=target.name,
    )


def resolve_reference(
    raw: str,
    snapshot: ContextSnapshot,
    *,
    kinds: Iterable[ObjectKind] | None = None,
) -> ContextObject | None:
    """The object a captured slot value refers to, or ``None``.

    The other way in, for the case where matching already happened and a slot
    came back holding a pronoun: «закрой {app}» matches «закрой его» perfectly
    well, and it is the value that needs the context rather than the phrase.
    Returns ``None`` both for a value that is not a pronoun and for a pronoun
    with nothing to point at — the caller cannot act on either, and the
    difference is available from :func:`find_mention` when it matters.
    """
    mention = find_mention(raw)
    if mention is None:
        return None
    return _pick(snapshot, mention.pronoun, kinds)


def _pick(
    snapshot: ContextSnapshot,
    pronoun: Pronoun,
    kinds: Iterable[ObjectKind] | None,
) -> ContextObject | None:
    """The remembered object a pronoun points at, honouring both restrictions."""
    allowed = _intersect(pronoun.kinds, kinds)
    if allowed is None:
        return None
    return snapshot.object_of(allowed, gender=pronoun.gender)


def _intersect(
    pronoun_kinds: tuple[ObjectKind, ...],
    caller_kinds: Iterable[ObjectKind] | None,
) -> tuple[ObjectKind, ...] | None:
    """Combine the two kind restrictions.

    Either side being empty means «no restriction from here», so the result is
    the other one. ``None`` comes back when both sides restrict and they have
    nothing in common — «сохрани туда» in a context where the caller will only
    accept a volume level — and that is a request that cannot be satisfied, not
    one to satisfy with the newest object of the wrong kind.
    """
    if caller_kinds is None:
        return pronoun_kinds
    wanted = tuple(caller_kinds)
    if not wanted:
        return pronoun_kinds
    if not pronoun_kinds:
        return wanted
    common = tuple(kind for kind in wanted if kind in pronoun_kinds)
    return common or None


def _words(text: str | NormalizedText | Sequence[str]) -> tuple[str, ...]:
    """The phrase as folded words, whatever shape it arrived in."""
    if isinstance(text, str):
        return normalize(text).words
    if isinstance(text, NormalizedText):
        return text.words
    return tuple(fold_letters(word) for word in text)
