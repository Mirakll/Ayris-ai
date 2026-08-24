"""Task 25, first half: links and searches.

The module splits into a part that decides and a part that calls Windows, and so
do these tests. Everything about the decision — which URL, which browser, which
switches — runs on any platform against
:class:`~ayris.actions.system.browser.RecordingOpener`, which keeps the request
instead of opening it. Only one test actually opens a browser, and it lives
behind a ``skipif``: on the windows runner it opens a blank page written to a
temporary file, waits for the window and closes it again.

Three things are asserted directly, because each of them is a way to mislead a
user rather than merely a way to fail.

*A refused scheme stays refused.* ``javascript:`` and ``file:`` open only when
the caller passed ``allow_local``, and a misheard phrase can never pass it.

*A private window is private, or an error.* Asking for one in a browser with no
known switch raises instead of quietly opening an ordinary window — a user who
asked for privacy and silently did not get it has been told something false
about where their history is going.

*A query is escaped once and correctly.* Cyrillic, spaces, ``&``, ``#``, ``+``
and ``%`` all go through :func:`~urllib.parse.quote`, and the tests read the
resulting URL rather than trusting the call.

Nothing here touches the network: a search action builds a URL and hands it to
the opener, which is the recorder.

Groups:

* :class:`TestNormalizeUrl` — aliases, bare domains, ports, refused schemes.
* :class:`TestProviders` — templates from the settings, escaping, defaults.
* :class:`TestProviderInPhrase` — «найди в ютубе …», and what must not match.
* :class:`TestBrowserFlags` — the per-browser switch table.
* :class:`TestOpenURL` — the action end to end, against the recorder.
* :class:`TestSearchWeb` — the same for the search action.
* :class:`TestRealBrowser` — opens a blank local page, on Windows only.
"""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import pytest

from ayris.actions.system.app_index import AppCandidate, AppNotFound, IndexedApp, IndexSource
from ayris.actions.system.browser import (
    BROWSER_FLAGS,
    OpenURL,
    RecordingOpener,
    SearchWeb,
    browser_family,
    build_open_request,
    normalize_url,
    search_url,
    set_opener,
    split_provider,
)
from ayris.core import config as config_module
from ayris.core.errors import ActionError, ActionParamsInvalid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit

#: Where the fake index says the browsers are. Real-looking paths, because
#: :func:`~ayris.actions.system.browser.browser_family` reads the executable name
#: out of them and a placeholder would not exercise that.
BROWSER_PATHS = {
    "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
    "msedge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "browser": r"C:\Users\Test\AppData\Local\Yandex\YandexBrowser\Application\browser.exe",
    "iexplore": r"C:\Program Files\Internet Explorer\iexplore.exe",
}

#: Spoken name to the executable stem the index resolves it to.
SPOKEN = {
    "хром": "chrome",
    "chrome": "chrome",
    "фаерфокс": "firefox",
    "firefox": "firefox",
    "edge": "msedge",
    "яндекс браузер": "browser",
    "ie": "iexplore",
}


class FakeIndex:
    """The application index, reduced to what a browser lookup needs.

    Stands in for :class:`~ayris.actions.system.app_index.AppIndex`, which would
    otherwise scan the registry and the Start menu. What it returns is a real
    :class:`~ayris.actions.system.app_index.AppCandidate` around a real
    :class:`~ayris.actions.system.app_index.IndexedApp`, so the production code
    that turns one into a launch target is the code under test.
    """

    def __init__(self, *, installed: bool = True) -> None:
        self.installed = installed
        self.asked: list[str] = []

    def resolve(self, phrase: str) -> AppCandidate:
        """Answer a spoken browser name, or refuse the way the index refuses."""
        self.asked.append(phrase)
        stem = SPOKEN.get(phrase.strip().casefold(), "")
        if not stem or not self.installed:
            raise AppNotFound(
                f"no application for {phrase!r}",
                user_message=f"Не нашла «{phrase}».",
            )
        flags = BROWSER_FLAGS.get(stem)
        return AppCandidate(
            app_id=stem,
            name=flags.title_ru if flags else "Internet Explorer",
            confidence=1.0,
            app=IndexedApp(
                name=stem,
                target=BROWSER_PATHS[stem],
                source=IndexSource.APP_PATHS,
                executable=BROWSER_PATHS[stem],
            ),
        )


@pytest.fixture
def index(monkeypatch: pytest.MonkeyPatch) -> FakeIndex:
    """A browser lookup that answers without reading the registry."""
    fake = FakeIndex()
    monkeypatch.setattr("ayris.actions.system.browser.get_app_index", lambda: fake)
    return fake


@pytest.fixture
def opener() -> Iterator[RecordingOpener]:
    """Installed instead of ``ShellExecuteW``; removed again afterwards."""
    recorder = RecordingOpener()
    set_opener(recorder)
    yield recorder
    set_opener(None)


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Change ``[actions.browser]`` the way the rest of the suite does it.

    Through the environment and a config reset rather than by patching
    :func:`~ayris.core.config.get_settings`, so the values go through pydantic
    validation — a template dict that only worked because nothing checked it
    would be a test passing for the wrong reason.
    """

    def apply(**values: str) -> None:
        for field, value in values.items():
            monkeypatch.setenv(f"AYRIS_ACTIONS__BROWSER__{field.upper()}", value)
        config_module.reset_config_manager()

    return apply


class TestNormalizeUrl:
    """What counts as an address, and what is refused."""

    @pytest.mark.parametrize(
        ("spoken", "expected"),
        [
            ("ютуб", "https://youtube.com"),
            ("вики", "https://ru.wikipedia.org"),
            ("habr.com", "https://habr.com"),
            ("habr com", "https://habr.com"),
            ("президент рф", "https://президент.рф"),
            ("https://ya.ru/pogoda", "https://ya.ru/pogoda"),
            ("http://example.org/a?b=1#c", "http://example.org/a?b=1#c"),
            ("mailto:ivan@example.ru", "mailto:ivan@example.ru"),
        ],
    )
    def test_accepts(self, spoken: str, expected: str) -> None:
        """Aliases, bare domains and full URLs all normalise to an address."""
        assert normalize_url(spoken) == expected

    def test_puts_back_eaten_slashes(self) -> None:
        """A normaliser that ate ``//`` leaves a URL no browser accepts."""
        assert normalize_url("https:ya.ru") == "https://ya.ru"

    @pytest.mark.parametrize(
        ("spoken", "expected"),
        [
            ("localhost:8080", "http://localhost:8080"),
            ("localhost:3000/api/v1", "http://localhost:3000/api/v1"),
            ("127.0.0.1:9000", "http://127.0.0.1:9000"),
            ("example.com:8443/x", "https://example.com:8443/x"),
        ],
    )
    def test_host_with_port(self, spoken: str, expected: str) -> None:
        """A port is evidence of an address; a local host speaks plain http."""
        assert normalize_url(spoken) == expected

    @pytest.mark.parametrize("phrase", ["погода в москве", "открой мне что нибудь", "  ", "и"])
    def test_refuses_a_phrase(self, phrase: str) -> None:
        """Words that are not an address do not become one."""
        with pytest.raises(ActionParamsInvalid):
            normalize_url(phrase)

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(1)",
            "file:///C:/Windows/win.ini",
            "data:text/html,<script>x()</script>",
            "vbscript:msgbox",
        ],
    )
    def test_refuses_dangerous_schemes(self, url: str) -> None:
        """The schemes that read the disk or run code need an explicit flag."""
        with pytest.raises(ActionParamsInvalid):
            normalize_url(url)

    @pytest.mark.parametrize("url", ["javascript:alert(1)", "file:///C:/Windows/win.ini"])
    def test_allows_dangerous_schemes_when_told(self, url: str) -> None:
        """A macro written on purpose may still use them."""
        assert normalize_url(url, allow_local=True) == url

    def test_refuses_unknown_scheme(self) -> None:
        """A protocol Ayris has no opener for is an error, not a guess."""
        with pytest.raises(ActionParamsInvalid, match="unsupported scheme"):
            normalize_url("ftp://files.example.com/pub")

    def test_names_the_field(self) -> None:
        """The rejection says which parameter was wrong, for the editor."""
        with pytest.raises(ActionParamsInvalid) as raised:
            normalize_url("совершенно не адрес")
        assert raised.value.fields == ("url",)


class TestProviders:
    """Search templates: where they come from, and how the query is escaped."""

    def test_default_provider_from_settings(self) -> None:
        """With no provider named, the configured one is used."""
        name, url = search_url("тест")
        assert name == "yandex"
        assert url.startswith("https://yandex.ru/search/?text=")

    @pytest.mark.parametrize(
        ("provider", "prefix"),
        [
            ("google", "https://www.google.com/search?q="),
            ("yandex", "https://yandex.ru/search/?text="),
            ("youtube", "https://www.youtube.com/results?search_query="),
            ("duckduckgo", "https://duckduckgo.com/?q="),
            ("wikipedia", "https://ru.wikipedia.org/w/index.php?search="),
        ],
    )
    def test_every_shipped_provider(self, provider: str, prefix: str) -> None:
        """All five templates named in the task build the URL they should."""
        name, url = search_url("кот", provider)
        assert name == provider
        assert url == f"{prefix}%D0%BA%D0%BE%D1%82"

    def test_cyrillic_is_percent_encoded(self) -> None:
        """A Russian query survives as UTF-8 percent escapes."""
        _name, url = search_url("рецепт борща", "google")
        assert url == (
            "https://www.google.com/search?q="
            "%D1%80%D0%B5%D1%86%D0%B5%D0%BF%D1%82%20%D0%B1%D0%BE%D1%80%D1%89%D0%B0"
        )

    @pytest.mark.parametrize(
        ("query", "encoded"),
        [
            ("c++", "c%2B%2B"),
            ("a & b", "a%20%26%20b"),
            ("#hash", "%23hash"),
            ("100% cok", "100%25%20cok"),
            ("что?", "%D1%87%D1%82%D0%BE%3F"),
            ("a/b", "a%2Fb"),
            ("x=1", "x%3D1"),
        ],
    )
    def test_special_characters(self, query: str, encoded: str) -> None:
        """Every character that means something in a URL is escaped, ``+`` too."""
        _name, url = search_url(query, "google")
        assert url == f"https://www.google.com/search?q={encoded}"

    def test_collapses_whitespace(self) -> None:
        """A recogniser's double spaces do not reach the URL."""
        _name, url = search_url("  two   words  ", "duckduckgo")
        assert url == "https://duckduckgo.com/?q=two%20words"

    def test_empty_query_refused(self) -> None:
        """There is nothing to search for."""
        with pytest.raises(ActionParamsInvalid):
            search_url("   ")

    def test_unknown_provider_lists_the_known_ones(self) -> None:
        """The message tells the user what they may say instead."""
        with pytest.raises(ActionParamsInvalid) as raised:
            search_url("кот", "рамблер")
        assert "google" in raised.value.user_message

    def test_a_user_template_needs_no_code(self, configure: Callable[..., None]) -> None:
        """A provider added to the settings works, and one may be overridden.

        The acceptance criterion of the task, spelled out: the templates are
        data, so «найди в дока …» is a line in ``config.toml`` and not a patch to
        this module.
        """
        configure(
            default_provider="дока",
            providers=(
                '{"дока": "https://docs.example.com/search/{query}",'
                ' "google": "https://google.de/search?q={query}"}'
            ),
        )
        assert search_url("кот") == ("дока", "https://docs.example.com/search/%D0%BA%D0%BE%D1%82")
        assert search_url("кот", "google")[1].startswith("https://google.de/")
        # Merged, not replaced: the four templates not mentioned still work.
        assert search_url("кот", "youtube")[1].startswith("https://www.youtube.com/")


class TestProviderInPhrase:
    """«найди в ютубе смешные коты» — the provider inside the sentence."""

    @pytest.mark.parametrize(
        ("phrase", "provider", "query"),
        [
            ("в ютубе смешные коты", "youtube", "смешные коты"),
            ("в гугле погода", "google", "погода"),
            ("в яндексе купить билет", "yandex", "купить билет"),
            ("в википедии кварк", "wikipedia", "кварк"),
            ("на youtube lofi", "youtube", "lofi"),
            ("гугл как варить рис", "google", "как варить рис"),
        ],
    )
    def test_recognises(self, phrase: str, provider: str, query: str) -> None:
        """The provider is taken off the front and the rest stays the query."""
        assert split_provider(phrase) == (provider, query)

    @pytest.mark.parametrize(
        "phrase",
        ["в москве погода", "в чём разница", "погода", "в", "как дела в яндексе"],
    )
    def test_leaves_a_plain_query_alone(self, phrase: str) -> None:
        """A preposition is not a provider, and a provider named late is a query.

        «как дела в яндексе» asks about the company, not for a search there: only
        the first word may name the engine, because that is where a command puts
        it.
        """
        assert split_provider(phrase) == ("", phrase)

    def test_a_user_provider_is_recognised_by_its_own_name(
        self, configure: Callable[..., None]
    ) -> None:
        """A template the user added is matched without a table of inflections."""
        configure(providers='{"рутрекер": "https://rutracker.org/forum/?q={query}"}')
        assert split_provider("в рутрекер линукс") == ("рутрекер", "линукс")


class TestBrowserFlags:
    """The switch table, which is the reason a browser is ever named."""

    def test_family_from_a_path(self) -> None:
        """The executable identifies the family, not the shortcut's label."""
        assert browser_family(BROWSER_PATHS["chrome"]) == "chrome"
        assert browser_family(BROWSER_PATHS["firefox"]) == "firefox"
        assert browser_family(BROWSER_PATHS["browser"]) == "browser"

    def test_unknown_browser_has_no_family(self) -> None:
        """Something Ayris has no table entry for gets the URL and nothing else."""
        assert browser_family(BROWSER_PATHS["iexplore"]) == ""
        assert browser_family("") == ""

    def test_chromium_and_firefox_disagree(self) -> None:
        """The whole point of the table: the families spell it differently."""
        assert BROWSER_FLAGS["chrome"].private == "--incognito"
        assert BROWSER_FLAGS["firefox"].private == "-private-window"
        assert BROWSER_FLAGS["msedge"].private == "--inprivate"

    def test_private_window_uses_the_family_switch(self, index: FakeIndex) -> None:
        """Chrome gets ``--incognito``, Firefox gets ``-private-window``."""
        chrome = build_open_request("ютуб", browser="хром", private=True)
        firefox = build_open_request("ютуб", browser="фаерфокс", private=True)
        assert chrome.arguments == ("--incognito",)
        assert firefox.arguments == ("-private-window",)

    def test_new_window_is_not_added_to_a_private_one(self, index: FakeIndex) -> None:
        """Firefox opens two windows when told both, so it is told one thing."""
        request = build_open_request("ютуб", browser="фаерфокс", new_window=True, private=True)
        assert request.arguments == ("-private-window",)

    def test_new_window_alone(self, index: FakeIndex) -> None:
        """Without privacy, «в новом окне» is the family's own switch."""
        request = build_open_request("ютуб", browser="хром", new_window=True)
        assert request.arguments == ("--new-window",)

    def test_profile_shape_per_family(self, index: FakeIndex) -> None:
        """Chromium takes one argument, Firefox takes two."""
        chrome = build_open_request("ютуб", browser="хром", profile="Work")
        firefox = build_open_request("ютуб", browser="фаерфокс", profile="Work")
        assert chrome.arguments == ("--profile-directory=Work",)
        assert firefox.arguments == ("-P", "Work")

    def test_unknown_browser_refuses_a_private_window(self, index: FakeIndex) -> None:
        """Silence would tell the user their history is private when it is not."""
        with pytest.raises(ActionParamsInvalid, match="private-window"):
            build_open_request("ютуб", browser="ie", private=True)

    def test_unknown_browser_refuses_a_profile(self, index: FakeIndex) -> None:
        """The same for a profile: the wrong one holds somebody else's mail."""
        with pytest.raises(ActionParamsInvalid, match="profile switch"):
            build_open_request("ютуб", browser="ie", profile="Work")

    def test_unknown_browser_still_opens_plainly(self, index: FakeIndex) -> None:
        """Without switches, any executable can open a URL."""
        request = build_open_request("ютуб", browser="ie")
        assert request.arguments == ()
        assert request.browser == BROWSER_PATHS["iexplore"]

    def test_default_browser_cannot_take_switches(self) -> None:
        """The shell has no way to pass a flag to «whatever is default»."""
        with pytest.raises(ActionParamsInvalid, match="named browser"):
            build_open_request("ютуб", private=True)


class TestOpenURL:
    """The action, against the recorder — no browser is started."""

    def test_opens_the_default_browser(self, opener: RecordingOpener) -> None:
        """No browser named: the URL goes to the shell, which owns the choice."""
        result = OpenURL().run(OpenURL.Params(url="ютуб"))
        assert result.ok
        assert opener.last.url == "https://youtube.com"
        assert opener.last.browser == ""
        assert opener.last.arguments == ()
        assert result.value == "https://youtube.com"

    def test_says_which_site(self, opener: RecordingOpener) -> None:
        """The spoken confirmation names the host, not the whole URL."""
        result = OpenURL().run(OpenURL.Params(url="https://www.wikipedia.org/wiki/Кварк"))
        assert result.message_ru == "Открываю wikipedia.org."

    def test_opens_a_named_browser(self, index: FakeIndex, opener: RecordingOpener) -> None:
        """A browser by spoken name is resolved through the application index."""
        result = OpenURL().run(OpenURL.Params(url="habr.com", browser="хром"))
        assert opener.last.browser == BROWSER_PATHS["chrome"]
        assert opener.last.url == "https://habr.com"
        assert result.message_ru == "Открываю habr.com в Chrome."
        assert index.asked == ["хром"]

    def test_private_window_is_said_out_loud(
        self, index: FakeIndex, opener: RecordingOpener
    ) -> None:
        """A user who asked for privacy hears that they got it."""
        result = OpenURL().run(OpenURL.Params(url="habr.com", browser="фаерфокс", private=True))
        assert opener.last.arguments == ("-private-window",)
        assert "в приватном окне" in result.message_ru

    def test_configured_browser_is_used_without_being_named(
        self,
        index: FakeIndex,
        opener: RecordingOpener,
        configure: Callable[..., None],
    ) -> None:
        """The settings answer «which browser» when the command does not."""
        configure(browser="фаерфокс")
        OpenURL().run(OpenURL.Params(url="ютуб"))
        assert opener.last.browser == BROWSER_PATHS["firefox"]

    def test_private_by_default(
        self,
        index: FakeIndex,
        opener: RecordingOpener,
        configure: Callable[..., None],
    ) -> None:
        """A user who always wants incognito sets it once."""
        configure(browser="хром", private_by_default="true")
        OpenURL().run(OpenURL.Params(url="ютуб"))
        assert opener.last.arguments == ("--incognito",)

    def test_explicit_false_beats_the_setting(
        self,
        index: FakeIndex,
        opener: RecordingOpener,
        configure: Callable[..., None],
    ) -> None:
        """«открой обычным окном» has to be able to say so."""
        configure(browser="хром", private_by_default="true")
        OpenURL().run(OpenURL.Params(url="ютуб", private=False))
        assert opener.last.arguments == ()

    def test_missing_browser_is_a_clear_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        opener: RecordingOpener,
    ) -> None:
        """A browser that is not installed says so, and opens nothing."""
        monkeypatch.setattr(
            "ayris.actions.system.browser.get_app_index",
            lambda: FakeIndex(installed=False),
        )
        with pytest.raises(ActionError, match="not found") as raised:
            OpenURL().run(OpenURL.Params(url="ютуб", browser="хром"))
        assert "Не нашла браузер «хром»" in raised.value.user_message
        assert not opener.requests

    def test_refuses_javascript(self, opener: RecordingOpener) -> None:
        """A misheard phrase cannot reach the dangerous schemes."""
        with pytest.raises(ActionParamsInvalid):
            OpenURL().run(OpenURL.Params(url="javascript:alert(1)"))
        assert not opener.requests

    def test_allows_a_local_file_when_told(self, opener: RecordingOpener) -> None:
        """A macro somebody wrote may open a document on disk."""
        result = OpenURL().run(OpenURL.Params(url="file:///C:/report.pdf", allow_local=True))
        assert result.ok
        assert opener.last.url == "file:///C:/report.pdf"

    def test_quotes_a_url_with_spaces(self, index: FakeIndex) -> None:
        """An unquoted URL with a space arrives at the browser as two arguments."""
        request = build_open_request("https://example.com/a b", browser="хром")
        assert request.shell_arguments == '"https://example.com/a b"'

    def test_reports_what_it_did(self, index: FakeIndex, opener: RecordingOpener) -> None:
        """The audit trail holds the whole command line and the pid."""
        result = OpenURL().run(OpenURL.Params(url="ютуб", browser="хром", private=True))
        assert result.data["arguments"] == ["--incognito"]
        assert result.data["pid"] == opener.pid
        assert BROWSER_PATHS["chrome"] in str(result.detail)


class TestSearchWeb:
    """The search action: a phrase in, a URL opened."""

    def test_default_provider(self, opener: RecordingOpener) -> None:
        """«найди погоду» goes to the configured engine."""
        result = SearchWeb().run(SearchWeb.Params(query="погода"))
        assert result.ok
        assert opener.last.url.startswith("https://yandex.ru/search/?text=")
        assert result.data["provider"] == "yandex"

    def test_provider_in_the_phrase(self, opener: RecordingOpener) -> None:
        """«найди в ютубе смешные коты» searches YouTube for the rest of it."""
        result = SearchWeb().run(SearchWeb.Params(query="в ютубе смешные коты"))
        assert opener.last.url == (
            "https://www.youtube.com/results?search_query="
            "%D1%81%D0%BC%D0%B5%D1%88%D0%BD%D1%8B%D0%B5%20%D0%BA%D0%BE%D1%82%D1%8B"
        )
        assert result.data["query"] == "смешные коты"
        assert result.message_ru == "Ищу «смешные коты» в ютубе."

    def test_explicit_provider_beats_the_phrase(self, opener: RecordingOpener) -> None:
        """A macro that sets the provider means the whole phrase is the query."""
        SearchWeb().run(SearchWeb.Params(query="в ютубе коты", provider="google"))
        assert opener.last.url.startswith("https://www.google.com/search?q=%D0%B2%20")

    def test_search_in_a_named_browser(self, index: FakeIndex, opener: RecordingOpener) -> None:
        """The browser options are the same ones :class:`OpenURL` takes."""
        SearchWeb().run(
            SearchWeb.Params(query="кварк", provider="wikipedia", browser="хром", private=True)
        )
        assert opener.last.browser == BROWSER_PATHS["chrome"]
        assert opener.last.arguments == ("--incognito",)
        assert opener.last.url.startswith("https://ru.wikipedia.org/w/index.php?search=")

    def test_unknown_provider_opens_nothing(self, opener: RecordingOpener) -> None:
        """An error before the browser starts, not a page of nonsense."""
        with pytest.raises(ActionParamsInvalid):
            SearchWeb().run(SearchWeb.Params(query="кот", provider="рамблер"))
        assert not opener.requests


@pytest.mark.skipif(sys.platform != "win32", reason="открывает настоящий браузер")
class TestRealBrowser:
    """The one inch of this that cannot be checked anywhere else.

    Everything above stops at :class:`RecordingOpener`, so nothing above proves
    that ``ShellExecuteW`` accepts what Ayris hands it. This does, by opening a
    blank page for real.

    A local file and not ``about:blank``: the shell resolves a URL through the
    registered protocol handlers, and ``about:`` has none — the browsers claim it
    for themselves, on their own command lines. An empty ``.html`` does have one,
    it is the user's default browser, and opening it loads no page and sends no
    request, which is the whole point of the exercise.

    What is asserted is the shell call: the string arrived, Windows accepted it
    and started something. The window is waited for and closed if it turns up, but
    not asserted on — a browser showing its own first-run screen ahead of the file
    is that browser's behaviour, not Ayris's contract, and a test failing on it
    would be reporting the state of the runner rather than a bug.
    """

    def test_the_shell_opens_a_blank_page(self, tmp_path: Path) -> None:
        """``ShellExecuteW`` really runs, and whatever it opened is closed again."""
        from pathlib import Path as RealPath

        from ayris.actions.system.apps import get_launcher
        from ayris.actions.system.windows import WindowQuery, list_windows
        from ayris.core.paths import native_path
        from ayris.utils import winapi

        if not winapi.available():  # pragma: no cover - only on a non-Windows runner
            pytest.skip("нет WinAPI")
        # ``ShellExecuteW`` does not undo the percent-escapes in a file URL, so a
        # temporary folder under a Cyrillic profile name has to be spelled the 8.3
        # way — the same trick as :func:`~ayris.core.paths.native_path` exists for.
        # On CI the path is Latin and this hands it back unchanged.
        folder = native_path(tmp_path) or str(tmp_path)
        page = RealPath(folder) / "ayris-blank.html"
        page.write_text("<!doctype html><title>Ayris blank</title>", encoding="utf-8")

        set_opener(None)
        before = {record.hwnd for record in list_windows()}
        try:
            result = OpenURL().run(OpenURL.Params(url=page.as_uri(), allow_local=True))
        except ActionError as exc:  # pragma: no cover - runner with no browser at all
            pytest.skip(f"на этой машине нечем открыть ссылку: {exc}")
        assert result.ok
        assert result.value == page.as_uri()

        query = WindowQuery(title="Ayris blank")
        opened: list[int] = []
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not opened:
            opened = [record.hwnd for record in list_windows(query) if record.hwnd not in before]
            time.sleep(0.5)
        launcher = get_launcher()
        for hwnd in opened:
            launcher.close_window(hwnd)
