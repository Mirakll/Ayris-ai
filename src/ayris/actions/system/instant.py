"""Instant answers: a question in, one spoken sentence out.

Two actions. :class:`InstantAnswer` takes a question — any question — and reads the
answer out. :class:`SiteSummary` does the same for an address the user already has,
«что на этой странице», using the summary the page publishes about itself, so a
wiki, a documentation page or a news article can be listened to instead of merely
opened.

**There is no list of supported questions.** The providers in
:mod:`ayris.actions.system.providers` that declare trigger words get the questions
they are good at — a forecast, a rate, a clock, an article by name — and
everything else goes to
:class:`~ayris.actions.system.providers.lookup.LookupProvider`, which searches
first and answers from whatever article the question turns out to be about. So
«какая погода» is a forecast, «столица австралии» is an encyclopedia, and neither
of them is «не поняла, что именно узнать».

**Answers unfold rather than repeat.** An article's whole lead paragraph is stored
with the answer, and ``sentences`` decides how much of it is spoken. «Расскажи
подробнее» is the same request with a bigger number: no second call to the network,
and it works with the connection gone.

**The cache is what makes this usable.** Every free API has a rate limit, and
«какая погода» gets asked four times an evening. An answer inside its time to live
never leaves the machine. An answer past it is still there, and that is the whole
offline story: with no connection — or with offline mode on, which Ayris treats as
the same thing — it reads the stale answer and says outright that there is no
network, and when nothing is cached it says only that.

**Nothing here runs on its own.** Every request in this module is one thing the
user just said. No refresh timer, no prefetch, no warming of the cache at
startup — the zero-telemetry rule is not only about what Ayris sends, it is about
when it opens a connection at all.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.browser import normalize_url
from ayris.actions.system.providers import (
    FALLBACK_KIND,
    AnswerCache,
    HttpFetcher,
    InstantNotFound,
    InstantOffline,
    InstantProvider,
    PageProvider,
    providers,
)
from ayris.actions.system.providers.base import OFFLINE_MESSAGE
from ayris.actions.system.providers.base import InstantAnswer as Answer
from ayris.actions.system.providers.page import MAX_SENTENCES, PageSummary, split_sentences
from ayris.core.config import get_settings
from ayris.core.errors import ActionError, ActionParamsInvalid, ParamProblem
from ayris.nlu.numbers import plural_form
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

__all__ = [
    "InstantAnswer",
    "SiteSummary",
    "answer_now",
    "at_length",
    "detect_kind",
    "get_cache",
    "get_transport",
    "set_cache",
    "set_transport",
    "strip_subject",
    "ttl_for",
]

_log = get_logger(__name__)

#: Words that introduce a question without being part of it.
#:
#: Stripped before routing so «а какая сейчас погода в питере» and «погода питер»
#: reach the same provider with the same subject. Kept short on purpose: a long
#: list starts eating words that matter, and «сколько» has to survive because
#: «сколько времени» is what routes to the clock.
_FILLER: Final[frozenset[str]] = frozenset(
    {
        "а",
        "и",
        "ли",
        "ну",
        "скажи",
        "подскажи",
        "пожалуйста",
        "мне",
        "сейчас",
        "там",
        "какой",
        "какая",
        "какое",
        "какие",
        "сегодня",
        "текущий",
        "текущая",
    }
)

#: Prepositions and link words left over once the trigger and subject are apart.
#:
#: «сколько» is here rather than in :data:`_FILLER` because routing needs it and
#: the subject never does: «сколько времени в Токио» matches the clock on
#: «времени», and what reaches the geocoder has to be «токио» and not «сколько
#: токио». Same for «сколько стоит доллар» — the currency provider keeps its own
#: trigger words, and «сколько» is not one of them.
_SUBJECT_NOISE: Final[frozenset[str]] = frozenset(
    {"в", "во", "на", "у", "по", "для", "о", "об", "сколько"}
)


def _clean(text: str) -> list[str]:
    """A phrase as lowercase words, punctuation and filler removed."""
    words = (word.strip(".,!?;:«»\"'()") for word in text.casefold().split())
    return [word for word in words if word and word not in _FILLER]


def detect_kind(query: str, table: Mapping[str, InstantProvider]) -> str:
    """Which provider answers ``query``, or ``""`` when none claims it.

    Matches the longest trigger first, so a two-word trigger beats a one-word one
    that happens to be inside it: «сколько времени» must route to the clock even
    though «сколько» alone routes nowhere, and «что такое рубль» must be a fact
    rather than a rate because «что такое» is the longer match.
    """
    words = _clean(query)
    if not words:
        return ""
    phrase = " ".join(words)
    best_kind = ""
    best_length = 0
    for kind, provider in table.items():
        for trigger in provider.triggers:
            if len(trigger) <= best_length:
                continue
            if _contains(phrase, words, trigger):
                best_kind, best_length = kind, len(trigger)
    return best_kind


def _contains(phrase: str, words: list[str], trigger: str) -> bool:
    """Whether ``trigger`` appears in the phrase as whole words."""
    if " " in trigger:
        return trigger in phrase
    return trigger in words


def strip_subject(query: str, provider: InstantProvider) -> str:
    """What is left of ``query`` once the trigger words are out of it.

    «какая погода в питере» becomes «питере», which is what the geocoder wants —
    it resolves inflected Russian city names, so declining it back is neither
    needed nor safe to attempt. An empty result is meaningful: it says the user
    named no subject, and the caller substitutes the configured default city.

    A provider whose triggers are also its subject
    (:attr:`~ayris.actions.system.providers.base.InstantProvider.keeps_triggers`)
    keeps them: «сколько стоит доллар» has to reach the rates provider with the
    word «доллар» still in it. A provider that searches
    (:attr:`~ayris.actions.system.providers.base.InstantProvider.wants_phrase`)
    keeps the prepositions too, because it is given a question and not a subject.
    """
    words = _clean(query)
    if provider.wants_phrase:
        return " ".join(words)
    if not provider.keeps_triggers:
        for trigger in sorted(provider.triggers, key=len, reverse=True):
            words = _without(words, trigger.split())
    kept = [word for word in words if word not in _SUBJECT_NOISE]
    return " ".join(kept)


def _without(words: list[str], parts: list[str]) -> list[str]:
    """``words`` with the first run equal to ``parts`` removed."""
    span = len(parts)
    for start in range(len(words) - span + 1):
        if words[start : start + span] == parts:
            return words[:start] + words[start + span :]
    return words


def ttl_for(kind: str) -> float:
    """Seconds an answer of this kind stays fresh, from the settings.

    The clock is the one kind with no cache window worth having: a reading is
    wrong the moment after it is taken, and
    :meth:`~ayris.actions.system.providers.worldtime.WorldTimeProvider.refresh`
    recomputes it from the cached time zone anyway, without the network. So this
    returns zero for it — the entry is still stored, for its time zone.
    """
    instant = get_settings().actions.instant
    minutes = {
        "weather": instant.weather_ttl_min,
        "rates": instant.rates_ttl_min,
        "fact": instant.facts_ttl_min,
        "lookup": instant.facts_ttl_min,
        "page": instant.facts_ttl_min,
        "time": 0,
    }.get(kind, instant.facts_ttl_min)
    return float(minutes) * 60.0


_cache: AnswerCache | None = None
_transport: httpx.BaseTransport | None = None


def get_cache() -> AnswerCache:
    """The answer cache in effect. One file under the profile's cache folder."""
    global _cache
    if _cache is None:
        _cache = AnswerCache()
    return _cache


def set_cache(cache: AnswerCache | None) -> None:
    """Install a cache, or ``None`` to go back to the profile's own."""
    global _cache
    _cache = cache


def get_transport() -> httpx.BaseTransport | None:
    """The HTTP transport the providers use, or ``None`` for the real one."""
    return _transport


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Install an httpx transport, which is how the tests stay off the network."""
    global _transport
    _transport = transport


def _fetcher() -> HttpFetcher:
    """A client configured from the settings, with the installed transport."""
    instant = get_settings().actions.instant
    return HttpFetcher(
        timeout_s=instant.timeout_s,
        retries=instant.retries,
        transport=get_transport(),
    )


@dataclass(frozen=True, slots=True)
class Resolved:
    """One answer and where it came from, for the caller to phrase."""

    answer: Answer
    cached: bool
    provider: InstantProvider


def answer_now(
    provider: InstantProvider,
    subject: str,
    *,
    fresh: bool = False,
    now: float | None = None,
) -> Resolved:
    """An answer from the cache when it is fresh, from the network otherwise.

    The order is the whole behaviour of this module, so it is in one place:

    1. A cached answer inside its time to live is returned untouched — no request.
    2. A provider that can rebuild an answer locally is asked to
       (:meth:`~ayris.actions.system.providers.base.InstantProvider.refresh`),
       which is how the clock works with no connection.
    3. Otherwise the provider goes to the network, and the result is cached.
    4. If that fails for a network reason, the stale entry is served with
       :attr:`~ayris.actions.system.providers.base.InstantAnswer.stale` set — but
       only while it is inside ``stale_hours``, because a forecast from last week
       is not an answer even with a caveat.

    Args:
        provider: Which source to use.
        subject: The subject alone, already stripped of the trigger words.
        fresh: Skip step 1 and ask anyway. What «обнови погоду» sets.
        now: The moment to age the cache against. Injected by the tests.

    Raises:
        InstantOffline: No network and nothing usable cached.
        InstantNotFound: The service has no answer for this subject.
        InstantProviderError: Anything else the service did.
    """
    moment = time.time() if now is None else now
    cache = get_cache()
    key = provider.cache_key(subject)
    stored = cache.peek(key)
    ttl = ttl_for(provider.kind)
    if not fresh and stored is not None and ttl > 0 and stored.age_sec(now=moment) < ttl:
        _log.debug("мгновенный ответ из кэша: %s", key)
        return Resolved(answer=stored, cached=True, provider=provider)
    if stored is not None:
        rebuilt = provider.refresh(stored)
        if rebuilt is not None:
            cache.put(key, rebuilt.at(moment))
            return Resolved(answer=rebuilt.at(moment), cached=True, provider=provider)
    try:
        answer = provider.fetch(subject).at(moment)
    except InstantOffline:
        usable = _usable_stale(stored, now=moment)
        if usable is None:
            raise
        _log.warning("сеть недоступна, отдаю кэш от %s: %s", usable.fetched_at, key)
        return Resolved(answer=usable.aged(now=moment), cached=True, provider=provider)
    cache.put(key, answer)
    return Resolved(answer=answer, cached=False, provider=provider)


def _usable_stale(stored: Answer | None, *, now: float) -> Answer | None:
    """A stale answer young enough to read out, or ``None``."""
    if stored is None:
        return None
    limit = float(get_settings().actions.instant.stale_hours) * 3600.0
    return stored if stored.age_sec(now=now) <= limit else None


def at_length(answer: Answer, sentences: int) -> str:
    """The answer's sentence, re-cut to ``sentences`` when there is more of it.

    A forecast or a rate is one sentence and has no longer version — «доллар — 91
    рубль 50 копеек» does not unfold. An article does: the whole lead paragraph
    came back with the first request and is stored with the answer, so «расскажи
    подробнее» is a wider slice of a string that is already on the machine. No
    second request, and it works with the connection gone.
    """
    extract = answer.data.get("extract")
    if sentences <= 0 or not isinstance(extract, str) or not extract.strip():
        return answer.message_ru
    title = answer.data.get("title")
    summary = PageSummary(
        title=title.strip() if isinstance(title, str) else "",
        extract=extract,
        url=str(answer.data.get("url") or ""),
        source=answer.source,
    )
    return summary.spoken(sentences=sentences)


def _sentences_available(answer: Answer) -> int:
    """How many sentences the stored answer could be unfolded to, at most."""
    extract = answer.data.get("extract")
    return len(split_sentences(extract)) if isinstance(extract, str) else 0


def _age_ru(seconds: float) -> str:
    """How old an answer is, in the words a caveat needs.

    Rounds to the coarsest unit that is still true, because the point of the
    caveat is «this may have changed», not the exact minute it was taken.
    """
    minutes = int(seconds // 60)
    if minutes < 1:
        return "только что"
    if minutes < 60:
        return f"{minutes} {plural_form(minutes, 'минуту', 'минуты', 'минут')} назад"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} {plural_form(hours, 'час', 'часа', 'часов')} назад"
    days = hours // 24
    return f"{days} {plural_form(days, 'день', 'дня', 'дней')} назад"


def _spoken(resolved: Resolved, *, now: float, sentences: int = MAX_SENTENCES) -> str:
    """The final sentence: the answer, and the state of the network before it.

    A stale answer starts with «Нет подключения к сети.» rather than ending with a
    softer version of it. That is the wording the user asked for and the order they
    asked for it in: the fact that Ayris could not check is the first thing to say,
    and the old reading follows as something better than silence.
    """
    answer = resolved.answer
    message = at_length(answer, sentences)
    if not answer.stale:
        return message
    return f"{OFFLINE_MESSAGE} {message} Данные {_age_ru(answer.age_sec(now=now))}."


@register
class InstantAnswer(Action):
    """Answer a question out loud — any question, not one off a list.

    The trigger words in the providers route what they are good at: a forecast, a
    rate, a clock, an article asked for by name. Everything else falls through to
    :data:`~ayris.actions.system.providers.base.FALLBACK_KIND`, which searches and
    answers from the article it lands on. So an unrecognised question is a search,
    not a refusal, and ``kind`` stays a way to *force* a provider rather than the
    only way to reach one.
    """

    meta: ClassVar = ActionMeta(
        name="InstantAnswer",
        category=ActionCategory.WEB,
        title_ru="Мгновенный ответ",
        description_ru="Короткий ответ на вопрос: погода, курс, время, справка, поиск",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        query: str = Field(
            min_length=1,
            max_length=500,
            description="Вопрос целиком, например «какая погода в Питере»",
        )
        kind: str = Field(
            default="",
            max_length=32,
            description="Тип ответа: weather, rates, time, fact, lookup; пусто — по фразе",
        )
        fresh: bool = Field(
            default=False,
            title="Игнорировать кэш",
            description="Запросить заново, не беря ответ из кэша",
        )
        sentences: int = Field(
            default=MAX_SENTENCES,
            ge=1,
            le=8,
            description="Сколько предложений озвучить; больше — подробнее",
        )

    def run(self, params: Params) -> ActionResult[dict[str, object]]:
        table = providers(_fetcher())
        kind = params.kind.strip().casefold() or detect_kind(params.query, table) or FALLBACK_KIND
        provider = table.get(kind)
        if provider is None:
            listed = ", ".join(sorted(table))
            raise ActionParamsInvalid(
                f"no instant provider for {params.query!r} (kind={kind!r}), have {listed}",
                problems=[ParamProblem("kind", f"неизвестный тип {kind or '—'}")],
                user_message="Не знаю такого вида ответа. Просто спросите словами.",
            )
        subject = strip_subject(params.query, provider) if not params.kind.strip() else params.query
        if not subject and kind in _NEEDS_CITY:
            subject = get_settings().actions.instant.city
        now = time.time()
        try:
            resolved = answer_now(provider, subject, fresh=params.fresh, now=now)
        except InstantNotFound as exc:
            return ActionResult.failed(exc.user_message, detail=exc.technical)
        message = _spoken(resolved, now=now, sentences=params.sentences)
        source = "кэш" if resolved.cached else "сеть"
        _log.info("мгновенный ответ (%s, %s): %s", kind, source, subject)
        return ActionResult.done(
            message,
            value=resolved.answer.as_dict(),
            detail=f"{kind} for {subject!r} from {'cache' if resolved.cached else 'network'}",
            data={
                "kind": kind,
                "subject": subject,
                "cached": resolved.cached,
                "stale": resolved.answer.stale,
                "source": resolved.answer.source,
                "age_sec": round(resolved.answer.age_sec(now=now), 1),
                "sentences_total": _sentences_available(resolved.answer),
            },
        )


#: Kinds that answer about a place, so an empty subject means «здесь».
_NEEDS_CITY: Final[frozenset[str]] = frozenset({"weather", "time"})


@register
class SiteSummary(Action):
    """Read out what a page says about itself — any site, not just a wiki.

    The companion to :class:`~ayris.actions.system.browser.OpenURL`: the same
    addresses open there, and the ones worth listening to instead of looking at
    are summarised here. It reads only the summary a page publishes for exactly
    this purpose, which is why it works on a wiki, a documentation page and a news
    article alike without knowing anything about their markup.
    """

    meta: ClassVar = ActionMeta(
        name="SiteSummary",
        category=ActionCategory.WEB,
        title_ru="Выжимка со страницы",
        description_ru="Прочитать краткое описание страницы вслух",
        timeout_ms=30_000,
    )

    class Params(ActionParams):
        url: str = Field(
            min_length=1,
            max_length=2_000,
            description="Адрес страницы или название сайта",
        )
        sentences: int = Field(
            default=2,
            ge=1,
            le=6,
            description="Сколько предложений озвучить",
        )
        fresh: bool = Field(
            default=False,
            title="Игнорировать кэш",
            description="Запросить заново, не беря ответ из кэша",
        )

    def run(self, params: Params) -> ActionResult[dict[str, object]]:
        url = normalize_url(params.url)
        provider = PageProvider(_fetcher(), sentences=params.sentences)
        now = time.time()
        try:
            resolved = answer_now(provider, url, fresh=params.fresh, now=now)
        except InstantNotFound as exc:
            return ActionResult.failed(exc.user_message, detail=exc.technical)
        except InstantOffline as exc:
            raise ActionError(exc.technical, user_message=exc.user_message) from exc
        message = _spoken(resolved, now=now, sentences=params.sentences)
        _log.info("выжимка со страницы %s (%s)", url, "кэш" if resolved.cached else "сеть")
        return ActionResult.done(
            message,
            value=resolved.answer.as_dict(),
            detail=f"summary of {url} from {'cache' if resolved.cached else 'network'}",
            data={
                "url": url,
                "cached": resolved.cached,
                "stale": resolved.answer.stale,
                "source": resolved.answer.source,
            },
        )
