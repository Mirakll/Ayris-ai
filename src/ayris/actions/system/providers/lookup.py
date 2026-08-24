"""Any question at all: find the article first, then read it out.

The other providers answer the questions Ayris knows the shape of — a forecast, a
rate, a clock, an article asked for by name. This one answers the rest, and it is
the reason a user never has to learn what Ayris «supports»: whatever was asked,
it goes and looks, and says the short version of what it found.

**Why this is a separate provider and not a smarter**
:class:`~ayris.actions.system.providers.page.FactProvider`. That one asks
Wikipedia's summary endpoint for a *title*, which is right for «что такое кварк»
and useless for «столица австралии» — a question is not the name of an article.
So this provider does the step in between: it asks Wikipedia's own OpenSearch
endpoint which article best matches the phrase, and only then asks for that
article's summary. Two requests instead of one, both official and keyless, and
the second one is the same endpoint the fact provider already uses.

**What it will not do is read a search engine's result page.** Those pages are
markup that changes without notice, scraping them is against the terms of every
engine that has any, and the answer would be a guess about a layout rather than
something a source published. When Wikipedia has nothing, Ayris says so and
offers to open the search in a browser — which is what
:class:`~ayris.actions.system.browser.SearchWeb` is for, and where a human reads
the results the way they are meant to be read.

**The full lead paragraph is kept, not just the sentences that get spoken.** That
is what makes «расскажи подробнее» free: the answer is already on the machine, and
unfolding it further is a slice of a string rather than another request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Final
from urllib.parse import quote

from ayris.actions.system.providers.base import (
    InstantAnswer,
    InstantNotFound,
    InstantProvider,
    InstantProviderError,
)
from ayris.actions.system.providers.page import (
    MAX_SENTENCES,
    SUMMARY_URL,
    PageSummary,
    parse_summary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.actions.system.providers.base import HttpFetcher

__all__ = [
    "OPENSEARCH_URL",
    "LookupProvider",
    "parse_opensearch",
]

#: Wikipedia's OpenSearch endpoint: a phrase in, matching titles out. Official,
#: keyless, and documented as the search suggestion API — no result page involved.
OPENSEARCH_URL: Final = "https://{lang}.wikipedia.org/w/api.php"

#: Languages tried in order, same reasoning as the fact provider: the question was
#: asked in Russian, and English has an article about ten times as many things.
_LANGUAGES: Final[tuple[str, ...]] = ("ru", "en")

#: How many candidate titles to consider per language.
#:
#: Three. The first hit is usually right; the reason to keep two more is
#: disambiguation pages, which are a valid search hit and not an answer — when the
#: best match turns out to be one, the next candidate is tried instead of giving up.
_CANDIDATES: Final = 3


def parse_opensearch(payload: Any, query: str) -> list[str]:
    """Article titles out of an OpenSearch response, best match first.

    The response is a four-element array — the query, the titles, the
    descriptions, the links — which is an awkward shape to read and the reason
    this is a function with a name rather than three subscripts at the call site.

    Raises:
        InstantProviderError: Not the documented shape at all.
    """
    if not isinstance(payload, list) or len(payload) < 2:
        raise InstantProviderError(
            f"opensearch for {query!r} answered {type(payload).__name__}, not the documented array",
            user_message="Поиск ответил непонятно.",
        )
    titles = payload[1]
    if not isinstance(titles, list):
        raise InstantProviderError(
            f"opensearch for {query!r} has no titles array",
            user_message="Поиск ответил непонятно.",
        )
    return [title.strip() for title in titles if isinstance(title, str) and title.strip()]


class LookupProvider(InstantProvider):
    """Anything the other providers do not claim: «столица австралии», «кто написал Онегина».

    Has no triggers on purpose. Nothing routes *to* it by keyword — everything
    that routed nowhere else lands here, which is what turns «не поняла, что
    именно узнать» into an answer.
    """

    kind: ClassVar = "lookup"
    title_ru: ClassVar = "ответ из интернета"
    triggers: ClassVar = ()
    #: The question is the query, so not a word of it may be dropped on the way in.
    wants_phrase: ClassVar = True

    def __init__(
        self,
        fetcher: HttpFetcher,
        *,
        languages: Sequence[str] = _LANGUAGES,
        sentences: int = MAX_SENTENCES,
    ) -> None:
        super().__init__(fetcher)
        self._languages = tuple(languages)
        self._sentences = sentences

    def fetch(self, query: str) -> InstantAnswer:
        """Search for ``query``, then read the best article's summary.

        Raises:
            InstantNotFound: Nothing matched in any of the languages.
            InstantOffline: Wikipedia could not be reached.
            InstantProviderError: A response that is not the documented shape.
        """
        question = " ".join(query.split()).strip()
        if not question:
            raise InstantNotFound("empty question", user_message="Не поняла вопрос.")
        for language in self._languages:
            answer = self._in_language(language, question)
            if answer is not None:
                return answer
        raise InstantNotFound(
            f"nothing found for {question!r}",
            user_message=(
                f"Не нашла короткого ответа про «{question}». Могу открыть поиск в браузере."
            ),
        )

    def _in_language(self, language: str, question: str) -> InstantAnswer | None:
        """The first candidate in one language that has a summary, or ``None``."""
        titles = parse_opensearch(
            self.fetcher.get_json(
                OPENSEARCH_URL.format(lang=language),
                params={
                    "action": "opensearch",
                    "format": "json",
                    "search": question,
                    "limit": _CANDIDATES,
                    "namespace": 0,
                },
            ),
            question,
        )
        for title in titles[:_CANDIDATES]:
            summary = self._summary(language, title)
            if summary is None:
                continue
            return InstantAnswer(
                kind=self.kind,
                message_ru=summary.spoken(sentences=self._sentences),
                data={
                    **summary.as_dict(),
                    "language": language,
                    "question": question,
                    "matched": title,
                },
                source=summary.source,
            )
        return None

    def _summary(self, language: str, title: str) -> PageSummary | None:
        """One article's summary, or ``None`` when it is not an answer.

        A 404 and a disambiguation page both arrive here as
        :class:`~ayris.actions.system.providers.base.InstantNotFound`, and both
        mean «try the next candidate» rather than «tell the user nothing was
        found» — the search did match something, it just matched a signpost.
        """
        url = SUMMARY_URL.format(lang=language, title=quote(title.replace(" ", "_"), safe=""))
        try:
            return parse_summary(self.fetcher.get_json(url), title)
        except InstantNotFound:
            return None

    def cache_key(self, query: str) -> str:
        """Key on the question as asked, so a repeat costs nothing.

        Not on the article that was found: two different questions can land on the
        same article, and each of them has to be answerable from the cache on its
        own words.
        """
        return f"{self.kind}:{' '.join(query.split()).casefold()}"
