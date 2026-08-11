"""What a captured piece of a phrase turns into.

A template says «поставь громкость на {volume}» and the matcher hands back the
string «пятьдесят». Something has to decide that this means the integer 50, that
50 is inside the range a volume may take, and that «на десять процентов тише»
means something else entirely — a change, not a value. That decision is what a
slot type is.

**A slot type is a parser plus a range, and nothing else.** :class:`SlotType`
holds a name, a :meth:`~SlotType.parse` that returns a value or ``None``, and the
knowledge of what values are acceptable. It does not know which command asked, it
does not touch the database, and it never raises on bad input: a phrase the user
half-said is the normal case, and a parser that throws would take the whole
pipeline down with it. ``None`` means «this did not parse», the slot is marked
unparsed, and the command decides whether it can proceed without it.

**Relative values are a separate shape, not a number.** «громче» and «на 10%
тише» carry a direction and possibly an amount, and collapsing them into an
integer loses the only thing that matters — that this is applied to the current
value rather than replacing it. :class:`RelativeValue` keeps the three parts, and
:class:`VolumeType` returns it from the same :meth:`~SlotType.parse` an absolute
value comes out of, because the caller has to handle both anyway.

**The registry is how a plugin joins in.** Types are looked up by name at
template-compile time, so a plugin registering ``device`` before its commands are
compiled gets ``{name:device}`` in a template for free. Registration is by
instance rather than by class: a type usually needs data — the app resolver, a
config step size — and an instance is where that data lives.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol

from ayris.nlu.apps import AppMatch, AppResolver
from ayris.nlu.numbers import parse_number, parse_percent
from ayris.nlu.timeparse import ClockTime, parse_clock, parse_duration, parse_moment
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from datetime import timedelta

__all__ = [
    "DEFAULT_RELATIVE_STEP",
    "DOWN_WORDS",
    "MAX_VOLUME",
    "MIN_VOLUME",
    "UP_WORDS",
    "AppType",
    "BuiltinSlotType",
    "Direction",
    "DurationType",
    "FloatType",
    "IntType",
    "PercentType",
    "QueryType",
    "RelativeUnit",
    "RelativeValue",
    "SiteType",
    "SlotContext",
    "SlotParser",
    "SlotType",
    "SlotTypeRegistry",
    "StringType",
    "TimeType",
    "VolumeType",
    "default_registry",
]

_log = get_logger(__name__)

#: Volume is a percentage, and the settings window shows it as one.
MIN_VOLUME: Final = 0
MAX_VOLUME: Final = 100

#: How much a bare «громче» moves the volume when the config says nothing.
#: Five is too little to hear over a video, twenty overshoots; ten is what the
#: media keys on most keyboards do.
DEFAULT_RELATIVE_STEP: Final = 10

#: Words that mean «more», in every form a user says them in.
UP_WORDS: Final = frozenset(
    {
        "громче",
        "погромче",
        "выше",
        "повыше",
        "больше",
        "побольше",
        "увеличь",
        "увеличить",
        "прибавь",
        "прибавить",
        "добавь",
        "поднять",
        "подними",
        "вверх",
    }
)

#: Words that mean «less». Kept separate rather than derived, because «тише» is
#: not «громче» with a sign flipped in any way a table could express.
DOWN_WORDS: Final = frozenset(
    {
        "тише",
        "потише",
        "ниже",
        "пониже",
        "меньше",
        "поменьше",
        "уменьши",
        "уменьшить",
        "убавь",
        "убавить",
        "убери",
        "опусти",
        "опустить",
        "снизь",
        "снизить",
        "вниз",
    }
)


class Direction(StrEnum):
    """Which way a relative change goes."""

    UP = "up"
    DOWN = "down"


class RelativeUnit(StrEnum):
    """What a relative amount is counted in.

    ``STEP`` is «громче» with no amount at all — the size of one step is the
    caller's business, and a slot type has no opinion on how loud is loud.
    """

    STEP = "step"
    PERCENT = "percent"
    ABSOLUTE = "absolute"


@dataclass(frozen=True, slots=True)
class RelativeValue:
    """A change to apply to whatever the current value is.

    ``amount`` is ``None`` for a bare «громче»: the phrase genuinely did not say
    how much, and inventing a number here would hide the difference from a
    caller that has a configured step to apply. :meth:`resolve` is where a
    concrete step turns it into one.
    """

    direction: Direction
    amount: Decimal | None = None
    unit: RelativeUnit = RelativeUnit.STEP

    @property
    def sign(self) -> int:
        """``+1`` for up, ``-1`` for down."""
        return 1 if self.direction is Direction.UP else -1

    def resolve(
        self,
        current: int,
        *,
        step: int = DEFAULT_RELATIVE_STEP,
        minimum: int = MIN_VOLUME,
        maximum: int = MAX_VOLUME,
    ) -> int:
        """Apply the change to ``current``, clamped and rounded to a whole number.

        A percent amount is a percentage *of the scale*, not of the current
        value: «на 10% тише» at volume 20 gives 10, not 18. That is what a user
        watching a 0..100 slider means, and it is the only reading under which
        «на 10% тише» twice from 20 reaches zero rather than 16.2.

        Clamped for the same reason the scale is what percentages count against:
        «громче» at 95 is a request for as loud as it goes, not an error, and an
        unclamped 105 would be refused further down with nothing happening at
        all. The bounds are arguments because brightness and volume share this
        class and a future setting may not run 0 to 100.
        """
        amount = Decimal(step) if self.amount is None else self.amount
        value = int(Decimal(current) + Decimal(self.sign) * amount)
        return max(minimum, min(maximum, value))


@dataclass(frozen=True, slots=True)
class SlotContext:
    """Everything a parser may need that is not in the captured text.

    Passed to every :meth:`SlotType.parse`, whether the type looks at it or not,
    so that adding a field later does not touch a single existing type. ``now``
    is here rather than read from the clock inside a time parser, which is what
    makes «в семь» testable against a pinned moment; when it is ``None`` a time
    slot yields the clock reading itself and leaves the date to the caller.
    """

    now: datetime | None = None
    apps: AppResolver | None = None
    step: int = DEFAULT_RELATIVE_STEP
    extra: Mapping[str, object] = field(default_factory=dict)


class SlotParser(Protocol):
    """The one method a plugin has to supply to register a type."""

    def __call__(self, raw: str, context: SlotContext) -> object | None: ...


class BuiltinSlotType(StrEnum):
    """Names of the types that ship with Ayris.

    An enum rather than loose strings because these names appear in templates
    the user types, and a typo in «volumе» with a Cyrillic е should fail at
    compile time with a list of what was meant.
    """

    INT = "int"
    FLOAT = "float"
    STR = "str"
    APP = "app"
    TIME = "time"
    DURATION = "duration"
    VOLUME = "volume"
    PERCENT = "percent"
    SITE = "site"
    QUERY = "query"


class SlotType:
    """Base class: a name, a parser, and the pattern that captures it.

    Subclasses override :meth:`parse`. They do not override :meth:`safe_parse`,
    which is what the extractor calls — the wrapper is where a parser that
    raises in spite of everything is turned back into an unparsed slot, so one
    bad plugin cannot take down a command that happened to sit next to it.
    """

    #: Name used in a template as ``{name:type}``.
    name: str = ""

    #: Regex fragment that captures a value of this type. The default is
    #: non-greedy and stops at nothing in particular, which is right for a short
    #: word and wrong for a free-text query — hence :class:`QueryType`.
    pattern: str = r"[^\s].*?"

    #: Whether this type may swallow the rest of the phrase. A template is
    #: rejected at compile time if a greedy slot is not the last one.
    greedy: bool = False

    def parse(self, raw: str, context: SlotContext) -> object | None:
        """Turn captured text into a value, or ``None`` if it is not one."""
        raise NotImplementedError

    def safe_parse(self, raw: str, context: SlotContext) -> object | None:
        """:meth:`parse`, with any exception logged and turned into ``None``.

        A slot value the user mumbled is an everyday event, and the pipeline's
        answer to it is a slot marked unparsed — never a traceback that loses
        the command. Broad by design: a plugin's parser is arbitrary code.
        """
        text = raw.strip()
        if not text:
            return None
        try:
            return self.parse(text, context)
        except Exception:
            _log.exception("парсер слота %r упал на значении %r", self.name, text)
            return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"


class IntType(SlotType):
    """A whole number, spoken or written, optionally range-checked.

    ``minimum`` and ``maximum`` are what make a range a property of the *type*
    rather than a check every command repeats: «поставь громкость на 300» has to
    fail somewhere, and failing here means it fails identically everywhere.
    """

    name = BuiltinSlotType.INT
    pattern = r"[^\s]+(?:\s+[^\s]+){0,5}?"

    def __init__(self, minimum: int | None = None, maximum: int | None = None) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def parse(self, raw: str, context: SlotContext) -> int | None:
        number = parse_number(raw)
        if number is None:
            return None
        value = number.as_int
        if value is None:
            return None
        return value if self._in_range(value) else None

    def _in_range(self, value: int) -> bool:
        """Whether ``value`` is inside the configured bounds."""
        if self.minimum is not None and value < self.minimum:
            return False
        return not (self.maximum is not None and value > self.maximum)


class FloatType(SlotType):
    """A number that may have a fractional part: «полтора», «две с половиной»."""

    name = BuiltinSlotType.FLOAT
    pattern = r"[^\s]+(?:\s+[^\s]+){0,5}?"

    def __init__(self, minimum: float | None = None, maximum: float | None = None) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def parse(self, raw: str, context: SlotContext) -> float | None:
        number = parse_number(raw)
        if number is None:
            return None
        value = number.as_float
        if self.minimum is not None and value < self.minimum:
            return None
        return None if self.maximum is not None and value > self.maximum else value


class StringType(SlotType):
    """Whatever was captured, with the whitespace tidied and nothing else done.

    The escape hatch: a command that wants the words the user said, unparsed and
    unjudged. Empty after stripping counts as no value, because a template that
    matched nothing should not report a slot that is there but blank.
    """

    name = BuiltinSlotType.STR

    def parse(self, raw: str, context: SlotContext) -> str | None:
        value = " ".join(raw.split())
        return value or None


class QueryType(StringType):
    """Free text meant to be handed on whole: a search, a message, a note.

    Greedy, and that is the point — «найди в интернете как приготовить плов»
    ends where the phrase ends. Being greedy is also why the compiler refuses to
    place it anywhere but last: a query in the middle eats every slot after it.
    """

    name = BuiltinSlotType.QUERY
    pattern = r".+"
    greedy = True


class PercentType(SlotType):
    """A percentage, absolute or relative: «на 30 процентов», «на 10% тише».

    Returns a :class:`RelativeValue` when the phrase named a direction and a
    plain :class:`~decimal.Decimal` when it did not. One type covers both
    because the two are said with the same words in a different order, and a
    template author should not have to know which one the user will pick.
    """

    name = BuiltinSlotType.PERCENT
    pattern = r"[^\s]+(?:\s+[^\s]+){0,4}?"

    def parse(self, raw: str, context: SlotContext) -> Decimal | RelativeValue | None:
        direction = _direction_of(raw)
        percent = parse_percent(raw)
        if percent is None:
            number = parse_number(_strip_direction(raw))
            if number is None:
                return None
            percent_value = number.value
        else:
            percent_value = percent.value
        if direction is None:
            return percent_value
        return RelativeValue(direction, percent_value, RelativeUnit.PERCENT)


class VolumeType(SlotType):
    """Loudness: an absolute 0..100, or a change to whatever it is now.

    Four shapes reach this parser and all four have to work: «пятьдесят», «50»,
    «на 10 процентов тише» and a bare «громче». The last is why the return type
    is a union — a step with no number attached is still a complete instruction,
    and the caller resolves it against :attr:`SlotContext.step`.
    """

    name = BuiltinSlotType.VOLUME
    pattern = r"[^\s]+(?:\s+[^\s]+){0,4}?"

    def __init__(self, minimum: int = MIN_VOLUME, maximum: int = MAX_VOLUME) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def parse(self, raw: str, context: SlotContext) -> int | RelativeValue | None:
        direction = _direction_of(raw)
        if direction is not None:
            return self._relative(raw, direction)
        number = parse_number(raw)
        if number is None or number.as_int is None:
            return None
        value = number.as_int
        return value if self.minimum <= value <= self.maximum else None

    def _relative(self, raw: str, direction: Direction) -> RelativeValue:
        """A change, with the amount the phrase gave or none at all."""
        percent = parse_percent(raw)
        if percent is not None:
            return RelativeValue(direction, percent.value, RelativeUnit.PERCENT)
        number = parse_number(_strip_direction(raw))
        if number is not None:
            return RelativeValue(direction, number.value, RelativeUnit.ABSOLUTE)
        return RelativeValue(direction)


class DurationType(SlotType):
    """A length of time: «пять минут», «полтора часа», «через 30 секунд»."""

    name = BuiltinSlotType.DURATION
    pattern = r"[^\s]+(?:\s+[^\s]+){0,6}?"

    def __init__(self, minimum: float = 0.0, maximum: float = 365 * 24 * 3600) -> None:
        self.minimum = minimum
        self.maximum = maximum

    def parse(self, raw: str, context: SlotContext) -> timedelta | None:
        parsed = parse_duration(raw)
        if parsed is None:
            return None
        seconds = parsed.seconds
        return None if not self.minimum <= seconds <= self.maximum else parsed.value


class TimeType(SlotType):
    """A moment on the clock: «в семь», «полвторого», «завтра в 9:30».

    Yields a :class:`~datetime.datetime` when the context pinned a ``now`` to
    resolve against, and the bare :class:`~ayris.nlu.timeparse.ClockTime`
    otherwise. Both are useful: a reminder needs the moment, a settings field
    that stores «когда не беспокоить» needs the reading and no date at all.
    """

    name = BuiltinSlotType.TIME
    pattern = r"[^\s]+(?:\s+[^\s]+){0,4}?"

    def parse(self, raw: str, context: SlotContext) -> datetime | ClockTime | None:
        # ``expected=True``: being inside a ``{time}`` slot is the marker
        # :func:`parse_clock` otherwise demands. «разбуди в {time}» leaves the
        # preposition in the template and hands this a bare «семь», and refusing
        # that would make the obvious way to write the template the broken one.
        if context.now is not None:
            return parse_moment(raw, now=context.now, expected=True)
        return parse_clock(raw, expected=True)


class AppType(SlotType):
    """A program, resolved through :class:`~ayris.nlu.apps.AppResolver`.

    Without a resolver in the context there is nothing to resolve against, so
    the slot goes unparsed rather than guessing — a command that launches things
    should not fire on a name nobody confirmed exists.
    """

    name = BuiltinSlotType.APP
    pattern = r"[^\s]+(?:\s+[^\s]+){0,3}?"

    def __init__(self, minimum_confidence: float = 0.75) -> None:
        self.minimum_confidence = minimum_confidence

    def parse(self, raw: str, context: SlotContext) -> AppMatch | None:
        if context.apps is None:
            _log.debug("слот приложения без словаря: %r", raw)
            return None
        match = context.apps.resolve(raw)
        if match is None or match.confidence < self.minimum_confidence:
            return None
        return match


class SiteType(SlotType):
    """A web address, spoken or typed: «ютуб», «youtube.com», «https://…».

    Returns a URL because that is what a caller opens. A bare name is turned
    into one only if it looks like a domain — «погода» is a search, not a site,
    and quietly resolving it to ``погода`` would open a browser at nothing.
    """

    name = BuiltinSlotType.SITE
    pattern = r"[^\s]+(?:\s+[^\s]+){0,3}?"

    def parse(self, raw: str, context: SlotContext) -> str | None:
        text = " ".join(raw.split()).strip(" .,")
        if not text:
            return None
        folded = _fold_site(text)
        known = _SITE_ALIASES.get(folded)
        if known is not None:
            return f"https://{known}"
        if _URL_RE.match(text):
            return text if "://" in text else f"https://{text}"
        candidate = _SITE_ALIASES.get(folded.replace(" ", ""))
        if candidate is not None:
            return f"https://{candidate}"
        return _rejoin_address(folded)


#: Anything with a dot and a plausible suffix, or an explicit scheme. Deliberately
#: loose: judging whether a domain exists is the browser's job, not a slot's.
_URL_RE: Final = re.compile(
    r"^(?:https?://)?(?:[\w-]+\.)+[a-z]{2,}(?:[/:?#][^\s]*)?$",
    re.IGNORECASE | re.UNICODE,
)

#: Sites named by word rather than by address. Small on purpose — this is the set
#: users say out loud, and everything else is typed and matches :data:`_URL_RE` or
#: is put back together by :func:`_rejoin_address`. The oblique forms are listed
#: rather than stemmed: «на ютубе» and «в википедии» is how the names are actually
#: said, there are a dozen of them in total, and a stemmer here would be a second
#: place where «озон» could turn into «озеро».
_SITE_ALIASES: Final[Mapping[str, str]] = {
    "ютуб": "youtube.com",
    "ютубе": "youtube.com",
    "ютьюб": "youtube.com",
    "ютьюбе": "youtube.com",
    "youtube": "youtube.com",
    "гугл": "google.com",
    "гугле": "google.com",
    "google": "google.com",
    "яндекс": "ya.ru",
    "яндексе": "ya.ru",
    "yandex": "ya.ru",
    "вк": "vk.com",
    "вконтакте": "vk.com",
    "vk": "vk.com",
    "гитхаб": "github.com",
    "гитхабе": "github.com",
    "github": "github.com",
    "википедия": "ru.wikipedia.org",
    "википедию": "ru.wikipedia.org",
    "википедии": "ru.wikipedia.org",
    "вики": "ru.wikipedia.org",
    "wikipedia": "ru.wikipedia.org",
    "хабр": "habr.com",
    "хабре": "habr.com",
    "habr": "habr.com",
    "телеграм": "web.telegram.org",
    "почта": "mail.google.com",
    "почту": "mail.google.com",
    "гмайл": "mail.google.com",
    "gmail": "mail.google.com",
    "твич": "twitch.tv",
    "твиче": "twitch.tv",
    "twitch": "twitch.tv",
    "авито": "avito.ru",
    "озон": "ozon.ru",
    "озоне": "ozon.ru",
    "вайлдберриз": "wildberries.ru",
    "кинопоиск": "kinopoisk.ru",
    "кинопоиске": "kinopoisk.ru",
}


def _fold_site(text: str) -> str:
    """Lowercase a spoken site name and drop the words around it."""
    words = [word for word in text.lower().split() if word not in _SITE_NOISE]
    return " ".join(words)


#: Words that surround a site name without being part of it.
_SITE_NOISE: Final = frozenset({"сайт", "сайте", "на", "в", "открой", "зайди", "перейди"})

#: Suffixes that turn a run of words back into an address. Short on purpose: every
#: entry is also a word a recogniser can produce from speech, and a list with
#: «is» or «am» in it would start reading ordinary phrases as websites.
_TLDS: Final = frozenset(
    {
        "com",
        "ru",
        "org",
        "net",
        "io",
        "dev",
        "tv",
        "info",
        "app",
        "su",
        "рф",
        "edu",
        "gov",
        "biz",
        "online",
        "site",
        "store",
        "kz",
        "by",
        "ua",
    }
)

#: One label of a domain, or one segment of a path. Unicode, because ``рф`` names
#: exist and «президент.рф» is a real thing a user may say.
_LABEL_RE: Final = re.compile(r"^[\w-]+$", re.UNICODE)


def _rejoin_address(folded: str) -> str | None:
    """A domain the normaliser took apart, put back together, or ``None``.

    «youtube.com» never reaches a slot with its dot: the phrase has been through
    :func:`~ayris.nlu.normalize.normalize_text` by then, which replaces every
    punctuation mark with a space, so a typed address arrives as a run of words.
    Without this the only sites that worked would be the ones in
    :data:`_SITE_ALIASES`, and «открой youtube.com» — the most literal thing a
    user can say — would silently do nothing.

    The suffix is what marks the run as an address, and the refusal when there is
    no suffix is the whole safety of it: «сделай погромче» has to stay a phrase.
    The first suffix wins rather than the last, so «habr com ru» is ``habr.com/ru``
    and not the nonexistent ``habr.com.ru``.
    """
    words = folded.split()
    scheme = "https"
    if words and words[0] in ("http", "https"):
        scheme = words[0]
        words = words[1:]
    if len(words) < 2 or not all(_LABEL_RE.match(word) for word in words):
        return None
    for position in range(1, len(words)):
        if words[position] in _TLDS:
            host = ".".join(words[: position + 1])
            path = "/".join(words[position + 1 :])
            return f"{scheme}://{host}/{path}" if path else f"{scheme}://{host}"
    return None


def _direction_of(raw: str) -> Direction | None:
    """Which way a phrase points, or ``None`` when it names a value instead.

    Word by word rather than by substring: «поменьше» is a direction and
    «поменяй» is not, and a substring test cannot tell them apart.
    """
    for word in raw.lower().replace("%", " ").split():
        cleaned = word.strip(".,!?")
        if cleaned in UP_WORDS:
            return Direction.UP
        if cleaned in DOWN_WORDS:
            return Direction.DOWN
    return None


def _strip_direction(raw: str) -> str:
    """The phrase without its direction words, so a number can be read from it."""
    words = [
        word
        for word in raw.split()
        if word.strip(".,!?").lower() not in UP_WORDS | DOWN_WORDS | _AMOUNT_NOISE
    ]
    return " ".join(words)


#: Words between a direction and its amount. «на» in «на 10 процентов тише»
#: carries nothing the parser needs and would otherwise break the numeral scan.
_AMOUNT_NOISE: Final = frozenset({"на", "по", "ещё", "еще", "чуть", "немного", "слегка"})


class _FunctionType(SlotType):
    """Adapter that dresses a plain function as a :class:`SlotType`.

    So that a plugin can register a lambda and still get everything a type
    gets — the safe wrapper, the greedy check, a name in an error message —
    without subclassing anything.
    """

    def __init__(
        self,
        name: str,
        parser: SlotParser,
        *,
        pattern: str = SlotType.pattern,
        greedy: bool = False,
    ) -> None:
        self.name = name
        self.pattern = pattern
        self.greedy = greedy
        self._parser = parser

    def parse(self, raw: str, context: SlotContext) -> object | None:
        return self._parser(raw, context)


class SlotTypeRegistry:
    """The set of type names a template may use.

    Mutable, unlike almost everything else here, because plugins register into
    it as they load and the templates they bring are compiled afterwards. What
    is *not* allowed is replacing a name already taken: two plugins both calling
    their type ``device`` is a conflict the second one has to hear about, not a
    silent overwrite that changes what the first one's commands do.
    """

    def __init__(self, types: Mapping[str, SlotType] | None = None) -> None:
        self._types: dict[str, SlotType] = dict(types or {})

    def register(
        self,
        name: str,
        parser: SlotParser | SlotType,
        *,
        pattern: str | None = None,
        greedy: bool = False,
        replace: bool = False,
    ) -> SlotType:
        """Add a type under ``name`` and return it.

        Raises:
            ValueError: The name is empty, is not a valid identifier, or is
                already registered and ``replace`` was not asked for.
        """
        key = name.strip().lower()
        if not key.isidentifier():
            raise ValueError(f"недопустимое имя типа слота: {name!r}")
        if key in self._types and not replace:
            raise ValueError(f"тип слота {key!r} уже зарегистрирован")
        if isinstance(parser, SlotType):
            slot_type = parser
            slot_type.name = key
        else:
            slot_type = _FunctionType(
                key,
                parser,
                pattern=pattern if pattern is not None else SlotType.pattern,
                greedy=greedy,
            )
        self._types[key] = slot_type
        return slot_type

    def unregister(self, name: str) -> bool:
        """Drop a type. ``True`` if it was there — a plugin unloading calls this."""
        return self._types.pop(name.strip().lower(), None) is not None

    def get(self, name: str) -> SlotType | None:
        """The type registered under ``name``, or ``None``."""
        return self._types.get(name.strip().lower())

    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted — for an error message that helps."""
        return tuple(sorted(self._types))

    def copy(self) -> SlotTypeRegistry:
        """An independent registry with the same types.

        A plugin experimenting with registrations should not be able to break
        the shipped set, and one line here is cheaper than making it immutable.
        """
        return SlotTypeRegistry(self._types)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name.strip().lower() in self._types

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._types))

    def __len__(self) -> int:
        return len(self._types)


def default_registry(
    *,
    apps: AppResolver | None = None,
    step: int = DEFAULT_RELATIVE_STEP,
) -> SlotTypeRegistry:
    """A registry holding the ten built-in types.

    A function rather than a module-level constant so that every caller gets its
    own: a plugin registering ``device`` into a shared singleton would leak the
    name into every other command library in the process, including the ones a
    test built two lines earlier.

    ``apps`` and ``step`` are accepted for symmetry with :class:`SlotContext` and
    are not stored — a resolver rebuilt by a background scan has to reach the
    parser through the context, or a registry compiled at startup would pin the
    empty index it saw then.
    """
    del apps, step
    return SlotTypeRegistry(
        {
            BuiltinSlotType.INT: IntType(),
            BuiltinSlotType.FLOAT: FloatType(),
            BuiltinSlotType.STR: StringType(),
            BuiltinSlotType.APP: AppType(),
            BuiltinSlotType.TIME: TimeType(),
            BuiltinSlotType.DURATION: DurationType(),
            BuiltinSlotType.VOLUME: VolumeType(),
            BuiltinSlotType.PERCENT: PercentType(),
            BuiltinSlotType.SITE: SiteType(),
            BuiltinSlotType.QUERY: QueryType(),
        }
    )
