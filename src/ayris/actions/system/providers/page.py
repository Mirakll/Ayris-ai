"""Short facts: an encyclopedia article, or a summary of any site at all.

Two things live here because they are the same job done at two levels of luck.

:class:`FactProvider` asks Wikipedia's REST summary endpoint, which exists
precisely to answer «что такое …» in one paragraph — it is an official API, it
needs no key, and it returns a clean extract rather than a page to scrape. It
tries Russian first and falls back to English, because a question asked in
Russian about something with no Russian article still has an answer.

:class:`PageProvider` handles everything Wikipedia does not: a link the user
already has, a documentation page, a news article, an internal wiki. It fetches
the page and reads the summary **the page publishes about itself** — the
``og:description`` a site puts there for exactly this purpose, then
``<meta name="description">``, then the first real paragraph. That is a
deliberate line: no attempt to interpret a layout, no scraping of search results,
nothing that depends on one site's markup surviving a redesign. If a page
publishes no summary, Ayris says so and offers to open it instead of inventing
one.

The HTML reader is a :class:`~html.parser.HTMLParser` subclass and not a regular
expression. Not for elegance — because a regular expression over ``<meta
content="…">`` gets attribute order, single quotes and entity escapes wrong in
about that order, and the standard library already has the state machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any, ClassVar, Final
from urllib.parse import quote, urlsplit

from ayris.actions.system.providers.base import (
    InstantAnswer,
    InstantNotFound,
    InstantProvider,
    InstantProviderError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "MAX_SENTENCES",
    "SUMMARY_URL",
    "FactProvider",
    "PageProvider",
    "PageSummary",
    "parse_page",
    "parse_summary",
    "shorten",
]

#: Wikipedia's REST summary endpoint, per language. Official, keyless, and
#: documented as the way to get one paragraph about a title.
SUMMARY_URL: Final = "https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}"

#: Languages tried, in order. Russian first because the question was in Russian;
#: English second because it has an article about ten times as many things.
_LANGUAGES: Final[tuple[str, ...]] = ("ru", "en")

#: How many sentences of an extract are worth reading out.
#:
#: Two. A voice answer that runs past about fifteen seconds stops being an answer
#: and becomes a lecture the user has to wait out, and the full text is in
#: ``data`` for the interface to show anyway.
MAX_SENTENCES: Final = 2

#: How much of a page body to look at. The head is where the summary is declared,
#: and a megabyte of JavaScript below it has nothing to contribute.
_MAX_BODY_BYTES: Final = 512 * 1024

#: Sentence end: a full stop, question or exclamation mark followed by a space and
#: a capital. The lookahead is what keeps «т. е.» and «1. Вступление» from
#: splitting a sentence in half.
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+(?=[«\"(]?[A-ZА-ЯЁ0-9])")

#: Runs of whitespace, including the non-breaking spaces Wikipedia is full of.
_SPACES: Final = re.compile(r"[\s   ]+")


def shorten(text: str, *, sentences: int = MAX_SENTENCES) -> str:
    """The first few sentences of ``text``, whitespace normalised.

    Ends on a sentence boundary rather than a character count: a summary cut
    mid-word is worse than a long one, and a synthesiser reading an ellipsis
    aloud is worse than both.
    """
    clean = _SPACES.sub(" ", unescape(text)).strip()
    if not clean:
        return ""
    parts = _SENTENCE_END.split(clean)
    return " ".join(parts[:sentences]).strip()


@dataclass(frozen=True, slots=True)
class PageSummary:
    """What one page says about itself."""

    title: str
    extract: str
    url: str = ""
    source: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe mapping, for the cache and the interface."""
        return {
            "title": self.title,
            "extract": self.extract,
            "url": self.url,
            "source": self.source,
        }

    def spoken(self, *, sentences: int = MAX_SENTENCES) -> str:
        """The sentence Ayris reads out: the title, then the short extract."""
        short = shorten(self.extract, sentences=sentences)
        if not short:
            return f"«{self.title}» — нашла страницу, но краткого описания на ней нет."
        if self.title and not short.casefold().startswith(self.title.casefold()[:12]):
            return f"{self.title}. {short}"
        return short


def parse_summary(payload: Any, query: str) -> PageSummary:
    """Read a Wikipedia REST summary response.

    Raises:
        InstantNotFound: A disambiguation page or an empty extract — both mean
            «I have not answered your question», and the second one silently.
        InstantProviderError: A body that is not the documented shape.
    """
    if not isinstance(payload, dict):
        raise InstantProviderError(f"summary is {type(payload).__name__}, not an object")
    if payload.get("type") == "disambiguation":
        raise InstantNotFound(
            f"{query!r} is a disambiguation page",
            user_message=(
                f"«{query}» — так называется несколько разных вещей. Уточните, пожалуйста."
            ),
        )
    extract = payload.get("extract")
    title = payload.get("title")
    if not isinstance(extract, str) or not extract.strip():
        raise InstantNotFound(
            f"no extract for {query!r}",
            user_message=f"Не нашла краткой справки по «{query}».",
        )
    return PageSummary(
        title=title.strip() if isinstance(title, str) else query,
        extract=extract.strip(),
        url=_summary_link(payload),
        source="wikipedia.org",
    )


def _summary_link(payload: Mapping[str, Any]) -> str:
    """The human-readable link out of a summary response, or ``""``."""
    urls = payload.get("content_urls")
    if isinstance(urls, dict):
        desktop = urls.get("desktop")
        if isinstance(desktop, dict):
            page = desktop.get("page")
            if isinstance(page, str):
                return page
    return ""


class _MetaReader(HTMLParser):
    """Collects the summary a page declares, and its first paragraph.

    Stops paying attention after ``</head>`` unless it still has nothing, which
    is what keeps a long page from being parsed in full for a description that
    was in the first twenty lines.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.og_title = ""
        self.description = ""
        self.og_description = ""
        self.paragraph = ""
        self._in_title = False
        self._in_paragraph = False
        self._paragraph_parts: list[str] = []

    @property
    def best_title(self) -> str:
        """The page's own name: its Open Graph title, or its ``<title>``."""
        return self.og_title or self.title

    @property
    def best_extract(self) -> str:
        """The summary the page publishes, in the order worth trusting."""
        return self.og_description or self.description or self.paragraph

    def handle_starttag(self, tag: str, attrs: Sequence[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag == "meta":
            self._read_meta(dict(attrs))
            return
        if tag == "p" and not self.paragraph:
            self._in_paragraph = True
            self._paragraph_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "p" and self._in_paragraph:
            self._in_paragraph = False
            text = _SPACES.sub(" ", "".join(self._paragraph_parts)).strip()
            # A paragraph of two words is a caption or a cookie notice, not a
            # summary. Forty characters is where a real sentence starts.
            if len(text) >= 40:
                self.paragraph = text

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title + data).strip()
        elif self._in_paragraph:
            self._paragraph_parts.append(data)

    def _read_meta(self, attrs: Mapping[str, str | None]) -> None:
        """Pick the two meta tags worth having out of the dozens a page has."""
        content = (attrs.get("content") or "").strip()
        if not content:
            return
        key = (attrs.get("property") or attrs.get("name") or "").strip().casefold()
        if key == "og:description" and not self.og_description:
            self.og_description = content
        elif key == "description" and not self.description:
            self.description = content
        elif key == "og:title" and not self.og_title:
            self.og_title = content


def parse_page(html: str, url: str) -> PageSummary:
    """Read the summary a page publishes about itself.

    Raises:
        InstantNotFound: The page declares no summary and has no first paragraph
            worth reading. Saying so is the honest outcome — the alternative is
            guessing at a layout, which is the thing this module refuses to do.
    """
    reader = _MetaReader()
    try:
        reader.feed(html)
        reader.close()
    except AssertionError as exc:  # pragma: no cover - malformed beyond the parser
        raise InstantProviderError(f"{url}: cannot parse HTML: {exc}") from exc
    extract = reader.best_extract
    host = urlsplit(url).hostname or url
    if not extract:
        raise InstantNotFound(
            f"{url}: page publishes no summary",
            user_message=f"На {host} нет краткого описания — могу просто открыть страницу.",
        )
    return PageSummary(
        title=reader.best_title or host,
        extract=extract,
        url=url,
        source=host,
    )


class FactProvider(InstantProvider):
    """«что такое кварк», «кто такой Тьюринг» — one paragraph from Wikipedia."""

    kind: ClassVar = "fact"
    title_ru: ClassVar = "справку"
    triggers: ClassVar = (
        "что такое",
        "кто такой",
        "кто такая",
        "кто это",
        "что значит",
        "расскажи про",
        "расскажи о",
        "определение",
        "справка",
    )

    def __init__(
        self,
        fetcher: Any,
        *,
        languages: Sequence[str] = _LANGUAGES,
        sentences: int = MAX_SENTENCES,
    ) -> None:
        super().__init__(fetcher)
        self._languages = tuple(languages)
        self._sentences = sentences

    def fetch(self, query: str) -> InstantAnswer:
        """A short article summary for ``query``.

        Tries each configured language in turn, and reports the first refusal
        rather than the last: «нет такой статьи» in Russian is the message the
        user needs, and an English 404 after it says nothing new.

        Raises:
            InstantNotFound: No article in any of the languages.
            InstantOffline: Wikipedia could not be reached.
        """
        subject = " ".join(query.split()).strip()
        if not subject:
            raise InstantNotFound("empty subject", user_message="Не поняла, о чём рассказать.")
        first: InstantNotFound | None = None
        for language in self._languages:
            url = SUMMARY_URL.format(lang=language, title=quote(subject.replace(" ", "_"), safe=""))
            try:
                summary = parse_summary(self.fetcher.get_json(url), subject)
            except InstantNotFound as exc:
                first = first or exc
                continue
            return InstantAnswer(
                kind=self.kind,
                message_ru=summary.spoken(sentences=self._sentences),
                data={**summary.as_dict(), "language": language},
                source=summary.source,
            )
        raise first or InstantNotFound(
            f"no article for {subject!r}",
            user_message=f"Не нашла справки по «{subject}».",
        )


class PageProvider(InstantProvider):
    """A summary of any page, read from what the page itself declares.

    Not part of :func:`~ayris.actions.system.providers.base.providers` because it
    answers a different question: the others are asked «какая погода», this one is
    asked «что на этой странице» and needs an address rather than a subject. The
    :class:`~ayris.actions.system.instant.SiteSummary` action is what routes to it.
    """

    kind: ClassVar = "page"
    title_ru: ClassVar = "выжимку со страницы"
    triggers: ClassVar = ()

    def __init__(self, fetcher: Any, *, sentences: int = MAX_SENTENCES) -> None:
        super().__init__(fetcher)
        self._sentences = sentences

    def fetch(self, query: str) -> InstantAnswer:
        """The summary of the page at ``query``, which is a URL.

        Raises:
            InstantNotFound: The page publishes no summary.
            InstantOffline: The page could not be fetched.
            InstantProviderError: The server answered with something that is not
                a web page.
        """
        url = query.strip()
        response = self.fetcher.get(url)
        kind = response.headers.get("content-type", "")
        if kind and "html" not in kind.casefold() and "xml" not in kind.casefold():
            raise InstantProviderError(
                f"{url}: content type {kind!r} is not a page",
                user_message="По этой ссылке не страница, а файл — прочитать не смогу.",
            )
        summary = parse_page(response.text[:_MAX_BODY_BYTES], str(response.url) or url)
        return InstantAnswer(
            kind=self.kind,
            message_ru=summary.spoken(sentences=self._sentences),
            data=summary.as_dict(),
            source=summary.source,
        )

    def cache_key(self, query: str) -> str:
        """Key on the address *and* the length, because both change the answer.

        The sentence count is part of the request, not of the page: «прочитай
        покороче» after «что там на странице» has to give a shorter answer, and a
        key built from the address alone would replay the long one for a day.
        """
        return f"{self.kind}:{self._sentences}:{' '.join(query.split())}"
