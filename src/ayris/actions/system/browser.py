"""Opening links and searches in a browser.

Two actions: :class:`OpenURL` takes an address, :class:`SearchWeb` takes words
and turns them into an address. Everything else in this module is the plumbing
between the two — normalising what a person said into a URL, and normalising
what a person called a browser into a command line.

**Any site opens, not a list of them.** «открой ютуб», «открой habr.com»,
«открой https://ya.ru/pogoda» and «открой президент.рф» all arrive here as text
that has been through the phrase normaliser, which replaced every dot with a
space. Putting the address back together is
:class:`~ayris.nlu.slot_types.SiteType`'s job and it already does it, aliases
included, so this module calls it instead of keeping a second table that would
drift from the first one. What is added here is the part a slot type must not
do: refusing ``javascript:`` and ``file:`` unless the caller said so out loud.

**Browsers do not share a command line.** Chromium takes ``--incognito`` and
``--profile-directory=…``, Firefox takes ``-private-window`` and ``-P``, and
Internet Explorer's descendants take neither. So the flags live in a table keyed
by browser family — :data:`BROWSER_FLAGS` — and a browser Ayris has never heard
of gets the URL and nothing else, which is the one thing every browser
understands. Asking for a private window in a browser with no known flag is an
error rather than a silent ordinary window: a user who asked for privacy and did
not get it has been actively misled.

**The system call is one function.** :func:`OpenURL.run` never touches
``ShellExecuteW`` itself — it builds an :class:`OpenRequest` and hands it to the
:class:`BrowserOpener` behind :func:`get_opener`, which the tests replace with a
recorder. That way the interesting half (which URL, which browser, which flags)
is checked on every platform, and only the last inch needs Windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Protocol
from urllib.parse import quote, urlsplit

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.app_index import AppNotFound, get_app_index
from ayris.actions.system.apps import LaunchRequest
from ayris.core.config import DEFAULT_SEARCH_PROVIDERS, get_settings
from ayris.core.errors import ActionError, ActionParamsInvalid, ActionUnavailable, ParamProblem
from ayris.nlu.slot_types import SiteType, SlotContext
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BROWSER_FLAGS",
    "BrowserFlags",
    "BrowserOpener",
    "OpenRequest",
    "OpenURL",
    "RecordingOpener",
    "SafeScheme",
    "SearchWeb",
    "browser_family",
    "build_open_request",
    "get_opener",
    "normalize_url",
    "search_url",
    "set_opener",
    "split_provider",
]

_log = get_logger(__name__)

#: Schemes a link may use without the caller saying anything special.
#:
#: ``http`` is here because plenty of the web still is, and a browser upgrades
#: what it can on its own. ``mailto`` and ``tel`` are here because «напиши на
#: почту …» is a link like any other and the shell knows what to do with them.
SafeScheme: Final[frozenset[str]] = frozenset({"http", "https", "mailto", "tel"})

#: Schemes that open only when ``allow_local`` was passed.
#:
#: ``file:`` reads the disk and ``javascript:`` runs code in whatever page is in
#: front — both are legitimate in a macro somebody wrote on purpose, and neither
#: should ever be reachable from a misheard phrase. Everything not in this set
#: and not in :data:`SafeScheme` is refused outright rather than listed here:
#: a new scheme appearing in a URL Ayris did not build is not a thing to guess
#: about.
_LOCAL_SCHEMES: Final[frozenset[str]] = frozenset({"file", "javascript", "data", "vbscript"})

#: Schemes that carry no host, so ``//`` must not be inserted before the body.
_HOSTLESS_SCHEMES: Final[frozenset[str]] = frozenset(
    {"mailto", "tel", "javascript", "data", "vbscript"}
)


@dataclass(frozen=True, slots=True)
class BrowserFlags:
    """How one family of browsers is told «new window» and «private window».

    ``profile`` is a format string rather than a flag because the two families
    disagree about the shape as well as the name: Chromium wants
    ``--profile-directory=Default`` in one argument, Firefox wants ``-P name``
    in two. An empty string in any field means the family has no such switch,
    and asking for it is then an error the caller sees.
    """

    new_window: str = ""
    private: str = ""
    profile: str = ""
    title_ru: str = ""


#: Command-line switches per browser family, keyed by the stem of the executable.
#:
#: Keyed by executable stem and not by spoken name because the spoken name is
#: already the application index's problem: «яндекс браузер» resolves to
#: ``browser.exe``, and that is what tells us it is a Chromium. The Chromium
#: entries are identical on purpose — they are separate keys so that a family
#: which diverges later can be changed without touching the others.
BROWSER_FLAGS: Final[Mapping[str, BrowserFlags]] = {
    "chrome": BrowserFlags(
        new_window="--new-window",
        private="--incognito",
        profile="--profile-directory={profile}",
        title_ru="Chrome",
    ),
    "msedge": BrowserFlags(
        new_window="--new-window",
        private="--inprivate",
        profile="--profile-directory={profile}",
        title_ru="Edge",
    ),
    "browser": BrowserFlags(
        new_window="--new-window",
        private="--incognito",
        profile="--profile-directory={profile}",
        title_ru="Яндекс Браузер",
    ),
    "opera": BrowserFlags(
        new_window="--new-window",
        private="--private",
        profile="--profile-directory={profile}",
        title_ru="Opera",
    ),
    "brave": BrowserFlags(
        new_window="--new-window",
        private="--incognito",
        profile="--profile-directory={profile}",
        title_ru="Brave",
    ),
    "vivaldi": BrowserFlags(
        new_window="--new-window",
        private="--incognito",
        profile="--profile-directory={profile}",
        title_ru="Vivaldi",
    ),
    "firefox": BrowserFlags(
        new_window="-new-window",
        private="-private-window",
        profile="-P {profile}",
        title_ru="Firefox",
    ),
}

#: What a browser with no entry in :data:`BROWSER_FLAGS` supports: the URL only.
_NO_FLAGS: Final = BrowserFlags()


def browser_family(target: str) -> str:
    """The :data:`BROWSER_FLAGS` key for an executable path, or ``""``.

    Takes a path rather than a name because that is what the application index
    answers with, and the path is the reliable half: a shortcut may be called
    «Google Chrome (рабочий)» and still be ``chrome.exe``.
    """
    stem = target.replace("\\", "/").rsplit("/", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return stem.casefold() if stem.casefold() in BROWSER_FLAGS else ""


def normalize_url(raw: str, *, allow_local: bool = False) -> str:
    """Turn what the user said into a URL, or refuse to.

    Accepts three shapes, in this order: an address with a scheme, which is
    checked and passed through; a site name or bare domain, which
    :class:`~ayris.nlu.slot_types.SiteType` resolves — that covers the alias
    table («вики», «ютуб»), a typed domain, and a domain the phrase normaliser
    took apart into words; and nothing else. A phrase that is not an address does
    not become one: «погода в москве» is a search, and turning it into
    ``https://погода`` would open a browser at an error page.

    Args:
        raw: The address as it arrived — spoken, typed, or already a URL.
        allow_local: Permit ``file:``, ``javascript:`` and friends. Off by
            default, and only ever turned on by a caller that said so.

    Raises:
        ActionParamsInvalid: Empty, not an address, or a scheme that needs
            ``allow_local`` and did not get it.
    """
    text = " ".join(raw.split()).strip()
    if not text:
        raise ActionParamsInvalid(
            "empty url",
            problems=[ParamProblem("url", "адрес не указан")],
            user_message="Не поняла, какой адрес открыть.",
        )
    scheme = _scheme_of(text)
    if scheme:
        return _checked_scheme(text, scheme, allow_local=allow_local)
    resolved = SiteType().parse(text, SlotContext())
    if resolved is not None:
        return resolved
    with_port = _host_with_port(text)
    if with_port:
        return with_port
    raise ActionParamsInvalid(
        f"not an address: {text!r}",
        problems=[ParamProblem("url", f"«{text}» не похоже на адрес сайта")],
        user_message=f"«{text}» не похоже на адрес сайта.",
    )


def _scheme_of(text: str) -> str:
    """The scheme of a URL as written, lowercased, or ``""`` when there is none.

    Hand-rolled rather than left to :func:`~urllib.parse.urlsplit`, which reads
    ``localhost:8080`` as the scheme ``localhost`` — correct by the grammar and
    useless here, because that string is an address a developer types every day.
    So a scheme is only a scheme when it is one Ayris knows, or when ``//``
    follows it: ``ftp://files.example.com`` is then refused as an unsupported
    protocol, while ``localhost:8080`` falls through to :func:`_host_with_port`.
    """
    head, separator, rest = text.partition(":")
    if not separator or not head or not head[0].isalpha():
        return ""
    if not all(character.isalnum() or character in "+-." for character in head):
        return ""
    folded = head.casefold()
    if rest.startswith("//") or folded in SafeScheme or folded in _LOCAL_SCHEMES:
        return folded
    return ""


def _host_with_port(text: str) -> str:
    """``localhost:8080`` and the like, given a scheme, or ``""``.

    The one address shape :class:`~ayris.nlu.slot_types.SiteType` refuses, and it
    has to be refused there: a slot type that accepted ``word:number`` would
    swallow half the phrases in the language. Here the port is the evidence —
    nothing a person says by accident ends in a colon and a number.

    ``http`` and not ``https`` for a local host, because that is what a
    development server speaks and an https guess costs a failed handshake and a
    browser warning before the user gets to see anything.
    """
    match = _HOST_PORT_RE.match(text)
    if match is None:
        return ""
    host = match["host"].casefold()
    scheme = "http" if host in _LOCAL_HOSTS else "https"
    return f"{scheme}://{text}"


#: A host with an explicit port, and whatever path follows it.
_HOST_PORT_RE: Final = re.compile(
    r"^(?P<host>[A-Za-z0-9._-]+|\[[0-9A-Fa-f:]+])" r":(?P<port>\d{1,5})(?P<rest>[/?#].*)?$"
)

#: Hosts that mean this machine, where a development server speaks plain http.
_LOCAL_HOSTS: Final[frozenset[str]] = frozenset(
    {"localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"}
)


def _checked_scheme(text: str, scheme: str, *, allow_local: bool) -> str:
    """Validate an address that came with a scheme, and tidy up ``//``.

    Raises:
        ActionParamsInvalid: The scheme is dangerous and not allowed, or unknown.
    """
    if scheme in _LOCAL_SCHEMES:
        if not allow_local:
            raise ActionParamsInvalid(
                f"scheme {scheme!r} refused without allow_local",
                problems=[
                    ParamProblem("url", f"схема {scheme}: не открываю без явного разрешения")
                ],
                user_message=f"Не открываю ссылки вида «{scheme}:» — это небезопасно.",
            )
        return text
    if scheme not in SafeScheme:
        raise ActionParamsInvalid(
            f"unsupported scheme {scheme!r}",
            problems=[ParamProblem("url", f"неизвестная схема {scheme}")],
            user_message=f"Не знаю, чем открыть ссылку вида «{scheme}:».",
        )
    if scheme in _HOSTLESS_SCHEMES:
        return text
    rest = text[len(scheme) + 1 :]
    # «https:ya.ru» is what a normaliser leaves behind when it eats the slashes,
    # and every browser refuses it. The scheme is unambiguous, so put them back.
    return text if rest.startswith("//") else f"{scheme}://{rest.lstrip('/')}"


def _providers() -> Mapping[str, str]:
    """The search templates in effect: the shipped ones under the user's."""
    configured = get_settings().actions.browser.providers
    return {**DEFAULT_SEARCH_PROVIDERS, **configured}


def split_provider(query: str) -> tuple[str, str]:
    """Split «в ютубе смешные коты» into the provider name and the query.

    Returns ``("", query)`` when the phrase names no provider, so the caller
    falls back to the configured default. Only a provider that actually exists
    in the settings is recognised: «в москве погода» has to stay a query, and it
    does, because there is no template called ``москва`` — while a user who adds
    one called ``рутрекер`` gets «найди в рутрекере …» working without touching
    this function.
    """
    words = query.split()
    if len(words) < 2:
        return "", query
    start = 1 if words[0].casefold() in _PROVIDER_PREPOSITIONS else 0
    if start >= len(words) - 1:
        return "", query
    known = _providers()
    candidate = words[start].casefold()
    for name in (candidate, _PROVIDER_FORMS.get(candidate, "")):
        if name and name in known:
            return name, " ".join(words[start + 1 :])
    return "", query


#: Words that introduce the provider and are not part of it.
_PROVIDER_PREPOSITIONS: Final[frozenset[str]] = frozenset({"в", "во", "на", "через", "по"})

#: Spoken and inflected forms of the shipped provider names.
#:
#: Only the five that come with Ayris are listed, and only in the forms a
#: Russian sentence puts them in — «найди в ютубе», not «найди в youtube». A
#: provider the user adds is matched by its own name as written in the settings,
#: which is the name they will say, because they chose it.
_PROVIDER_FORMS: Final[Mapping[str, str]] = {
    "гугле": "google",
    "гугл": "google",
    "гугле,": "google",
    "google": "google",
    "яндексе": "yandex",
    "яндекс": "yandex",
    "yandex": "yandex",
    "ютубе": "youtube",
    "ютуб": "youtube",
    "ютьюбе": "youtube",
    "ютьюб": "youtube",
    "youtube": "youtube",
    "дакдакго": "duckduckgo",
    "утке": "duckduckgo",
    "duckduckgo": "duckduckgo",
    "вики": "wikipedia",
    "википедии": "wikipedia",
    "википедия": "wikipedia",
    "wikipedia": "wikipedia",
}


def search_url(query: str, provider: str = "") -> tuple[str, str]:
    """Build a search URL. Returns the provider actually used and the URL.

    Args:
        query: What to search for, already free of the provider name.
        provider: Which template to use; empty means the configured default.

    Raises:
        ActionParamsInvalid: The query is empty, or no such provider exists.
    """
    text = " ".join(query.split()).strip()
    if not text:
        raise ActionParamsInvalid(
            "empty search query",
            problems=[ParamProblem("query", "запрос пустой")],
            user_message="Не поняла, что искать.",
        )
    known = _providers()
    name = (provider or get_settings().actions.browser.default_provider).casefold()
    template = known.get(name)
    if template is None:
        listed = ", ".join(sorted(known)) or "нет ни одного"
        raise ActionParamsInvalid(
            f"unknown search provider {name!r}",
            problems=[ParamProblem("provider", f"нет провайдера {name}")],
            user_message=f"Не знаю поисковик «{name}». Настроены: {listed}.",
        )
    # ``quote`` and not ``quote_plus``: the templates put the placeholder in a
    # path as often as in a query string, and a literal ``+`` in a path means a
    # plus. ``%20`` is correct in both places, which is the point of a table of
    # templates the user can extend without being told which one they wrote.
    return name, template.replace("{query}", quote(text, safe=""))


@dataclass(frozen=True, slots=True)
class OpenRequest:
    """One link, and how to open it.

    ``browser`` empty means the system default, which is the common case and the
    only one that works without an installed browser being found first.
    """

    url: str
    browser: str = ""
    browser_name: str = ""
    arguments: tuple[str, ...] = ()

    @property
    def command_ru(self) -> str:
        """The command as one line, for the log and the audit trail."""
        parts = [self.browser or "<браузер по умолчанию>", *self.arguments, self.url]
        return " ".join(parts)

    @property
    def shell_arguments(self) -> str:
        """What ``ShellExecuteW`` gets as its argument string."""
        return " ".join((*self.arguments, _quoted(self.url)))


def _quoted(url: str) -> str:
    """Quote a URL for a Windows command line if it needs it.

    A percent-encoded URL has no spaces in it, but a URL from a macro or from
    the clipboard may, and an unquoted one arrives at the browser as two
    arguments — the first of which it opens.
    """
    return f'"{url}"' if any(character.isspace() for character in url) else url


class BrowserOpener(Protocol):
    """The operating system, as far as opening a link goes."""

    def open(self, request: OpenRequest) -> int:
        """Open the link and return a process id, or ``0`` when unknown."""
        ...


class WinApiOpener:
    """The real thing: ``ShellExecuteW``, either on the URL or on the browser."""

    def open(self, request: OpenRequest) -> int:
        """Open the link.

        With no browser named this hands the URL itself to the shell, which is
        what a click on a link does and what respects the user's default. With a
        browser named it runs that executable instead, because flags and profiles
        only exist on a command line.

        Raises:
            ActionError: The shell refused.
        """
        target = request.browser or request.url
        arguments = request.shell_arguments if request.browser else ""
        try:
            return winapi.shell_execute(target, arguments=arguments)
        except winapi.WinApiError as exc:
            raise ActionError(
                f"failed to open {request.command_ru!r}: {exc}",
                user_message="Не смогла открыть ссылку в браузере.",
            ) from exc


@dataclass(slots=True)
class RecordingOpener:
    """Test double that keeps the requests instead of opening anything.

    Lives here rather than in the tests because the arguments Ayris builds — the
    final URL, the browser path, the private-window flag — are the whole of what
    can be checked away from Windows, and both the unit tests and a plugin
    author debugging a template need the same recorder to look at them.
    """

    requests: list[OpenRequest]
    pid: int = 4242

    def __init__(self, pid: int = 4242) -> None:
        self.requests = []
        self.pid = pid

    def open(self, request: OpenRequest) -> int:
        """Record the request and report the canned process id."""
        self.requests.append(request)
        return self.pid

    @property
    def last(self) -> OpenRequest:
        """The most recent request, for a test that only opened one thing."""
        if not self.requests:
            raise AssertionError("no link was opened")
        return self.requests[-1]


_opener: BrowserOpener | None = None


def get_opener() -> BrowserOpener:
    """The opener in effect: the installed one, or ``ShellExecuteW``.

    Raises:
        ActionUnavailable: Not Windows and nothing installed.
    """
    if _opener is not None:
        return _opener
    if not winapi.available():
        raise ActionUnavailable(
            "browser actions require Windows",
            user_message="Открытие ссылок работает только в Windows.",
        )
    return WinApiOpener()


def set_opener(opener: BrowserOpener | None) -> None:
    """Install an opener, or ``None`` to go back to ``ShellExecuteW``."""
    global _opener
    _opener = opener


def _resolve_browser(name: str) -> tuple[str, str]:
    """Path and display name of a browser named the way a person says it.

    Returns ``("", "")`` for an empty name, which means the system default.

    Raises:
        ActionError: The name matches nothing installed.
    """
    if not name.strip():
        return "", ""
    index = get_app_index()
    try:
        candidate = index.resolve(name)
    except AppNotFound as exc:
        raise ActionError(
            f"browser {name!r} not found: {exc}",
            user_message=f"Не нашла браузер «{name}» на этом компьютере.",
        ) from exc
    request = LaunchRequest.for_candidate(candidate)
    return request.target, candidate.name


def _browser_arguments(
    target: str,
    *,
    new_window: bool,
    private: bool,
    profile: str,
) -> tuple[str, ...]:
    """The switches for one browser, in the order browsers expect them.

    Raises:
        ActionParamsInvalid: A switch was asked for that this browser has no
            known form of. Silence would be worse: an «открой приватно», that
            quietly opened an ordinary window has told the user something false
            about where their history is going.
    """
    flags = BROWSER_FLAGS.get(browser_family(target), _NO_FLAGS)
    arguments: list[str] = []
    if profile:
        if not flags.profile:
            raise ActionParamsInvalid(
                f"no profile switch known for {target!r}",
                problems=[ParamProblem("profile", "этот браузер не умеет выбирать профиль")],
                user_message="Не знаю, как выбрать профиль в этом браузере.",
            )
        arguments.extend(flags.profile.format(profile=_quoted(profile)).split(" "))
    if private:
        if not flags.private:
            raise ActionParamsInvalid(
                f"no private-window switch known for {target!r}",
                problems=[ParamProblem("private", "этот браузер не умеет приватное окно")],
                user_message="Не знаю, как открыть приватное окно в этом браузере.",
            )
        arguments.append(flags.private)
    elif new_window and flags.new_window:
        # A private window is already a new window in every family here, so the
        # two are never passed together — Firefox opens two windows if they are.
        arguments.append(flags.new_window)
    return tuple(arguments)


def build_open_request(
    url: str,
    *,
    browser: str = "",
    new_window: bool = False,
    private: bool | None = None,
    profile: str = "",
    allow_local: bool = False,
) -> OpenRequest:
    """Everything :class:`OpenURL` decides, without opening anything.

    Public because :class:`SearchWeb` needs the same decisions, and because a
    test that checks «what would Ayris do with this phrase» should not have to
    install an opener to find out.

    Args:
        url: The address, in any of the shapes :func:`normalize_url` accepts.
        browser: Spoken name of a browser, or empty for the configured one.
        new_window: Ask for a separate window.
        private: Ask for a private window; ``None`` takes the configured default.
        profile: Browser profile to use.
        allow_local: Permit ``file:`` and other local schemes.

    Raises:
        ActionParamsInvalid: The URL or a switch was refused.
        ActionError: The named browser is not installed.
    """
    settings = get_settings().actions.browser
    incognito = settings.private_by_default if private is None else private
    target = url.strip() if url.strip() else ""
    final = normalize_url(target, allow_local=allow_local)
    name = browser.strip() or settings.browser
    path, display = _resolve_browser(name)
    if not path and (incognito or profile or new_window):
        # The shell has no way to pass a switch to whatever the default browser
        # is, so a request that needs one has to know which browser it is.
        raise ActionParamsInvalid(
            "window options need a named browser",
            problems=[ParamProblem("browser", "нужен конкретный браузер")],
            user_message=(
                "Чтобы открыть в новом или приватном окне, укажите браузер "
                "в настройках или в команде."
            ),
        )
    arguments = _browser_arguments(
        path,
        new_window=new_window,
        private=incognito,
        profile=profile.strip(),
    )
    return OpenRequest(url=final, browser=path, browser_name=display, arguments=arguments)


@register
class OpenURL(Action):
    """Open a link — any site, in the default browser or a named one."""

    meta: ClassVar = ActionMeta(
        name="OpenURL",
        category=ActionCategory.WEB,
        title_ru="Открыть ссылку",
        description_ru="Открыть сайт или адрес в браузере",
        timeout_ms=15_000,
    )

    class Params(ActionParams):
        url: str = Field(
            min_length=1,
            max_length=2_000,
            description="Адрес или название сайта, например «ютуб» или «habr.com»",
        )
        browser: str = Field(
            default="",
            max_length=120,
            description="Браузер: пусто — по умолчанию, иначе «хром», «фаерфокс», «edge»",
        )
        new_window: bool = Field(
            default=False,
            title="Новое окно",
            description="Открыть в отдельном окне, а не во вкладке",
        )
        private: bool | None = Field(
            default=None,
            title="Приватное окно",
            description="Приватный режим; не задано — как в настройках",
        )
        profile: str = Field(
            default="",
            max_length=120,
            description="Профиль браузера",
        )
        allow_local: bool = Field(
            default=False,
            title="Разрешить локальные схемы",
            description="Разрешить file: и подобные схемы — только для своих макросов",
        )

    def run(self, params: Params) -> ActionResult[str]:
        request = build_open_request(
            params.url,
            browser=params.browser,
            new_window=params.new_window,
            private=params.private,
            profile=params.profile,
            allow_local=params.allow_local,
        )
        pid = get_opener().open(request)
        _log.info("открываю %s: %s (pid %s)", request.url, request.command_ru, pid)
        where = f" в {request.browser_name}" if request.browser_name else ""
        private = " в приватном окне" if _is_private(request) else ""
        return ActionResult.done(
            f"Открываю {_spoken_host(request.url)}{where}{private}.",
            value=request.url,
            detail=f"opened {request.command_ru} as pid {pid}",
            data={
                "url": request.url,
                "browser": request.browser,
                "browser_name": request.browser_name,
                "arguments": list(request.arguments),
                "pid": pid,
            },
        )


def _is_private(request: OpenRequest) -> bool:
    """Whether the built command line asks for a private window."""
    flags = BROWSER_FLAGS.get(browser_family(request.browser), _NO_FLAGS)
    return bool(flags.private) and flags.private in request.arguments


def _spoken_host(url: str) -> str:
    """The part of a URL worth saying out loud: its host, or the URL itself."""
    host = urlsplit(url).hostname or ""
    if not host:
        return url
    return host[4:] if host.startswith("www.") else host


@register
class SearchWeb(Action):
    """Search the web — «найди в ютубе …», «найди погоду»."""

    meta: ClassVar = ActionMeta(
        name="SearchWeb",
        category=ActionCategory.WEB,
        title_ru="Найти в интернете",
        description_ru="Открыть поиск по запросу в выбранном поисковике",
        timeout_ms=15_000,
    )

    class Params(ActionParams):
        query: str = Field(
            min_length=1,
            max_length=1_000,
            description="Что искать; можно с поисковиком — «в ютубе смешные коты»",
        )
        provider: str = Field(
            default="",
            max_length=64,
            description="Поисковик: пусто — из настроек, иначе google, yandex, youtube…",
        )
        browser: str = Field(
            default="",
            max_length=120,
            description="Браузер, в котором открыть выдачу",
        )
        new_window: bool = Field(
            default=False,
            title="Новое окно",
            description="Открыть в отдельном окне",
        )
        private: bool | None = Field(
            default=None,
            title="Приватное окно",
            description="Приватный режим; не задано — как в настройках",
        )

    def run(self, params: Params) -> ActionResult[str]:
        provider = params.provider.strip()
        query = params.query
        if not provider:
            provider, query = split_provider(query)
        name, url = search_url(query, provider)
        request = build_open_request(
            url,
            browser=params.browser,
            new_window=params.new_window,
            private=params.private,
        )
        pid = get_opener().open(request)
        _log.info("поиск «%s» в %s: %s", query.strip(), name, request.url)
        return ActionResult.done(
            f"Ищу «{query.strip()}» в {_provider_ru(name)}.",
            value=request.url,
            detail=f"searched {name}: {request.command_ru} as pid {pid}",
            data={
                "provider": name,
                "query": query.strip(),
                "url": request.url,
                "browser": request.browser,
                "pid": pid,
            },
        )


def _provider_ru(name: str) -> str:
    """A provider name in the case a Russian sentence needs after «в»."""
    return _PROVIDER_NAMES_RU.get(name, name)


#: Provider names as they are said, for the one sentence Ayris reads out.
_PROVIDER_NAMES_RU: Final[Mapping[str, str]] = {
    "google": "гугле",
    "yandex": "яндексе",
    "youtube": "ютубе",
    "duckduckgo": "DuckDuckGo",
    "wikipedia": "википедии",
}
