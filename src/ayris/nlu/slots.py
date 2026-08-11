"""Templates with holes in them, and what comes out of the holes.

A command in the library is written «поставь громкость на {volume}», not as a
regex. That is deliberate: the person filling in the settings window is not going
to write ``(?P<volume>[^\\s]+)``, and a template they can read is a template they
will fix themselves when it does not fire. This module is the two halves of
making that work — turning the template into a pattern, and turning what the
pattern captured into values.

**Compilation happens once, extraction happens per phrase.** :func:`compile_slots`
is the expensive half: it parses the braces, checks the type names against the
registry, refuses a greedy slot that is not last, and hands back a
:class:`SlotTemplate` holding a compiled regex. That object is built when the
command library loads and reused for every phrase after. :meth:`SlotTemplate.extract`
is the cheap half and does no parsing of the template at all.

**An unparsed slot is a result, not an error.** «поставь громкость на бубубу»
matched the template — the words around the hole were all there — and only the
value failed. So the slot comes back with ``value=None`` and
:attr:`Slot.parsed` false, the :class:`SlotSet` reports itself incomplete, and
the command decides whether to ask a follow-up question or refuse. Nothing in
here raises on user input; the only exceptions come from a malformed *template*,
which is a mistake in configuration and surfaces when it is saved.

**The greedy slot goes last, and the compiler enforces it.** ``{query}`` matches
to the end of the phrase by nature, so «найди {query} в {site}» can never fill
``site``. Catching that at compile time turns a command that mysteriously half-works
into an error message at the moment the template was written.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

from ayris.core.errors import AyrisError
from ayris.nlu.normalize import normalize, normalize_text
from ayris.nlu.slot_types import SlotContext, SlotType, SlotTypeRegistry, default_registry
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "MAX_SLOTS",
    "SLOT_PATTERN",
    "Slot",
    "SlotSet",
    "SlotTemplate",
    "SlotTemplateError",
    "compile_slots",
    "extract_slots",
    "template_slot_names",
    "validate_template",
]

_log = get_logger(__name__)

#: ``{name}`` or ``{name:type}``. Both halves are matched with Unicode ``\w`` and
#: checked with :meth:`str.isidentifier` afterwards, so «включи {что}» works and
#: so does a plugin that registered its type as «устройство»:
#: :meth:`~ayris.nlu.slot_types.SlotTypeRegistry.register` accepts any identifier,
#: and an ASCII-only pattern here would turn one of those into a stray brace with
#: an error message about the wrong thing.
SLOT_PATTERN: Final = re.compile(
    r"\{(?P<name>[^\W\d]\w*)(?::(?P<type>[^\W\d]\w*))?\}",
    re.UNICODE,
)

#: A brace that :data:`SLOT_PATTERN` did not claim. Left alone it would compile
#: into the pattern as a literal and the command would simply never fire, which
#: is the worst way to report «{volume!} is not a slot».
_STRAY_BRACE: Final = re.compile(r"[{}]")

#: Ceiling on slots in one template. A phrase with more holes than this is not a
#: command any more, and the limit keeps a pathological pattern out of the regex
#: engine's backtracking.
MAX_SLOTS: Final = 12

#: Type used when a slot names none. ``{query}`` and ``{volume}`` work without a
#: colon because the name itself is a registered type — that is the common case,
#: and ``{query:query}`` would be noise.
_FALLBACK_TYPE: Final = "str"


class SlotTemplateError(AyrisError):
    """A template cannot be compiled. Raised when it is saved, not when matched."""

    default_user_message = "Шаблон команды написан неверно."


@dataclass(frozen=True, slots=True)
class Slot:
    """One filled hole: what was said, and what it turned out to mean.

    ``raw`` is kept alongside ``value`` because the two answer different
    questions. A command that failed on «на бубубу» wants to tell the user what
    it heard, and a value of ``None`` cannot. ``confidence`` comes from the type
    when the type has an opinion — an app matched fuzzily is less certain than a
    number read exactly — and is 1.0 otherwise.
    """

    name: str
    type: str
    raw: str
    value: object | None = None
    confidence: float = 1.0

    @property
    def parsed(self) -> bool:
        """Whether the value came out. ``False`` means the raw text is all there is."""
        return self.value is not None

    def __str__(self) -> str:
        return self.raw if self.value is None else str(self.value)


@dataclass(frozen=True, slots=True)
class SlotSet:
    """Every slot a phrase filled, addressable by name.

    Behaves like a read-only mapping over the names, because that is how a
    command handler wants to read it — ``slots["volume"]``. :meth:`value` is the
    one to reach for in a handler: it returns the parsed value or a default, and
    saves every call site from checking :attr:`Slot.parsed` first.
    """

    slots: tuple[Slot, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether every slot parsed. A command needing all of them checks this."""
        return all(slot.parsed for slot in self.slots)

    @property
    def unparsed(self) -> tuple[str, ...]:
        """Names of the slots that did not parse, for the message asking again."""
        return tuple(slot.name for slot in self.slots if not slot.parsed)

    @property
    def confidence(self) -> float:
        """The least confident slot's confidence, or 1.0 when there are none.

        The minimum rather than the mean: a command is as trustworthy as its
        shakiest argument, and averaging lets four certain slots hide one guess.
        """
        return min((slot.confidence for slot in self.slots), default=1.0)

    def get(self, name: str) -> Slot | None:
        """The slot called ``name``, or ``None``."""
        return next((slot for slot in self.slots if slot.name == name), None)

    def value(self, name: str, default: object | None = None) -> object | None:
        """Parsed value of a slot, or ``default`` when it is missing or unparsed."""
        slot = self.get(name)
        return default if slot is None or slot.value is None else slot.value

    def raw(self, name: str, default: str = "") -> str:
        """Text a slot captured, parsed or not."""
        slot = self.get(name)
        return default if slot is None else slot.raw

    def as_dict(self) -> dict[str, object | None]:
        """Parsed values by name — what a plugin's handler is handed."""
        return {slot.name: slot.value for slot in self.slots}

    def __getitem__(self, name: str) -> Slot:
        slot = self.get(name)
        if slot is None:
            raise KeyError(name)
        return slot

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterator[Slot]:
        return iter(self.slots)

    def __len__(self) -> int:
        return len(self.slots)


@dataclass(frozen=True, slots=True)
class SlotTemplate:
    """A template with its regex already built and its types already looked up.

    Built by :func:`compile_slots` and held for the lifetime of the command.
    ``regex`` matches normalised text, because that is what the matcher works in
    and a template compiled against raw text would miss «Поставь громкость!».
    """

    template: str
    regex: re.Pattern[str]
    types: Mapping[str, SlotType] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        """Slot names, in the order the template writes them."""
        return tuple(self.types)

    def match(self, text: str) -> re.Match[str] | None:
        """The regex match for a phrase, or ``None``. Normalises first."""
        return self.regex.match(normalize_text(text))

    def extract(self, text: str, context: SlotContext | None = None) -> SlotSet | None:
        """Slots a phrase fills, or ``None`` when it does not fit the template.

        The distinction matters: ``None`` means «this is not this command», an
        empty-valued :class:`SlotSet` means «this is this command, said badly».
        Only the first justifies trying the next template.
        """
        found = self.match(text)
        if found is None:
            return None
        return self.bind(found.groupdict(), context)

    def bind(
        self,
        groups: Mapping[str, str | None],
        context: SlotContext | None = None,
    ) -> SlotSet:
        """Turn named groups into parsed slots.

        Separate from :meth:`extract` because the matcher has already run the
        regex by the time slots are wanted — :attr:`~ayris.nlu.matcher.MatchResult.raw_groups`
        is exactly this argument, and matching a second time would be wasted work
        and a second chance to disagree.
        """
        ctx = context if context is not None else SlotContext()
        return SlotSet(tuple(self._bind_one(name, groups.get(name), ctx) for name in self.types))

    def _bind_one(self, name: str, raw: str | None, context: SlotContext) -> Slot:
        """One slot, parsed as far as its type can take it."""
        slot_type = self.types[name]
        text = (raw or "").strip()
        value = slot_type.safe_parse(text, context) if text else None
        return Slot(
            name=name,
            type=slot_type.name,
            raw=text,
            value=value,
            confidence=_confidence_of(value),
        )


def _confidence_of(value: object | None) -> float:
    """How sure a parsed value is, asking the value itself when it knows.

    An :class:`~ayris.nlu.apps.AppMatch` carries the confidence of the alias that
    resolved it, and that number has to survive into the slot or a fuzzy app
    match becomes indistinguishable from an exact one.
    """
    if value is None:
        return 0.0
    confidence = getattr(value, "confidence", None)
    return float(confidence) if isinstance(confidence, int | float) else 1.0


def template_slot_names(template: str) -> tuple[str, ...]:
    """Slot names a template mentions, without compiling anything.

    For the settings window, which wants to show what a command will capture
    while the user is still typing and the template is not yet valid.
    """
    return tuple(found["name"] for found in SLOT_PATTERN.finditer(template))


def compile_slots(
    template: str,
    registry: SlotTypeRegistry | None = None,
) -> SlotTemplate:
    """Turn a template into something that can match and extract.

    Raises:
        SlotTemplateError: The template is empty, names an unknown type, repeats
            a slot name, has more than :data:`MAX_SLOTS` slots, or puts a greedy
            slot anywhere but last. Every one of these is a mistake in a template
            the user typed, so the message says which slot and what to do.
    """
    text = template.strip()
    if not text:
        raise SlotTemplateError("empty slot template", user_message="Шаблон команды пуст.")

    types = registry if registry is not None else default_registry()
    resolved: dict[str, SlotType] = {}
    parts: list[str] = []
    position = 0
    greedy_at: str = ""

    for found in SLOT_PATTERN.finditer(text):
        literal = text[position : found.start()]
        _reject_stray_brace(literal, template)
        parts.append(_literal(literal))
        position = found.end()
        name = found["name"]
        type_name = found["type"] or name
        if not name.isidentifier():
            raise SlotTemplateError(
                f"slot name {name!r} is not an identifier in template {template!r}",
                user_message=f"«{name}» не может быть именем слота.",
            )
        slot_type = _lookup(types, name, type_name)
        if name in resolved:
            raise SlotTemplateError(
                f"slot {name!r} repeats in template {template!r}",
                user_message=f"Слот «{name}» указан в шаблоне дважды.",
            )
        if greedy_at:
            raise SlotTemplateError(
                f"slot {greedy_at!r} is greedy and must be last in {template!r}",
                user_message=(
                    f"Слот «{greedy_at}» забирает остаток фразы, "
                    f"поэтому «{name}» после него никогда не заполнится."
                ),
            )
        if slot_type.greedy:
            greedy_at = name
        resolved[name] = slot_type
        parts.append(f"(?P<{name}>{slot_type.pattern})")

    _reject_stray_brace(text[position:], template)
    parts.append(_literal(text[position:]))
    if len(resolved) > MAX_SLOTS:
        raise SlotTemplateError(
            f"template {template!r} has {len(resolved)} slots, over the limit of {MAX_SLOTS}",
            user_message=f"В шаблоне больше {MAX_SLOTS} слотов — столько не поддерживается.",
        )

    pattern = "".join(parts)
    try:
        regex = re.compile(f"^{pattern}$", re.IGNORECASE | re.UNICODE)
    except re.error as exc:
        raise SlotTemplateError(
            f"template {template!r} compiled to invalid regex: {exc}",
            user_message=f"Не удалось разобрать шаблон:\n{template}",
        ) from exc
    return SlotTemplate(template=text, regex=regex, types=resolved)


def _reject_stray_brace(literal: str, template: str) -> None:
    """Refuse a brace outside a well-formed slot.

    Raises:
        SlotTemplateError: The literal text holds ``{`` or ``}``. The message
            names the whole template, because the interesting question is which
            brace was meant to be a slot and is not one.
    """
    if _STRAY_BRACE.search(literal):
        raise SlotTemplateError(
            f"unbalanced or malformed slot braces in template {template!r}",
            user_message=(
                f"Фигурная скобка не образует слот:\n{template}\n"
                "Слот пишется как {имя} или {имя:тип}."
            ),
        )


def _lookup(registry: SlotTypeRegistry, name: str, type_name: str) -> SlotType:
    """The type a slot asks for, with a message naming the alternatives.

    A slot whose name is not itself a type falls back to ``str`` rather than
    failing: «включи {что}» is a reasonable thing to write, and treating an
    unnamed type as text is what the writer meant.
    """
    slot_type = registry.get(type_name)
    if slot_type is not None:
        return slot_type
    if type_name == name:
        fallback = registry.get(_FALLBACK_TYPE)
        if fallback is not None:
            return fallback
    raise SlotTemplateError(
        f"unknown slot type {type_name!r} for slot {name!r}",
        user_message=(
            f"Неизвестный тип слота «{type_name}».\n" f"Доступные: {', '.join(registry.names())}"
        ),
    )


def _literal(text: str) -> str:
    """A literal stretch of template, as a pattern tolerant of extra spaces.

    Normalised first, and this is the part that is easy to get wrong: the phrase
    being matched has already been folded, depunctuated and had its numerals
    turned into digits, so a template written «поставь громкость на пятьдесят»
    has to be folded the same way or it will never meet the phrase it was written
    for. Wake-word stripping is off — a template is not addressed to anyone.

    Whitespace between words then becomes ``\\s+``, because a recogniser's
    spacing is not something a template author should have to predict.
    """
    words = normalize(text, strip_wake_word=False).words
    if not words:
        return r"\s*" if text else ""
    joined = r"\s+".join(re.escape(word) for word in words)
    lead = r"\s*" if text[:1].isspace() else ""
    tail = r"\s*" if text[-1:].isspace() else ""
    return f"{lead}{joined}{tail}"


def extract_slots(
    template: str,
    text: str,
    *,
    registry: SlotTypeRegistry | None = None,
    context: SlotContext | None = None,
) -> SlotSet | None:
    """Compile a template and extract in one call, for a one-off.

    Convenience for tests and for a settings window previewing what a template
    will capture. Anything in the hot path compiles once with
    :func:`compile_slots` and keeps the result.
    """
    return compile_slots(template, registry).extract(text, context)


def validate_template(template: str, registry: SlotTypeRegistry | None = None) -> str:
    """Check a template, returning a Russian message or ``""`` when it is fine.

    The counterpart of :func:`ayris.nlu.matcher.validate_pattern`, and the same
    contract: the command editor calls it before saving so the user hears about a
    mistake while they can still fix it. The index never raises on a bad
    template — it drops the trigger and logs — but a silently dead command is the
    worst way to learn that ``{query}`` cannot go in the middle.
    """
    try:
        compile_slots(template, registry)
    except SlotTemplateError as exc:
        return exc.user_message
    return ""
