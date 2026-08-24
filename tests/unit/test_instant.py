"""Task 25, second half: the instant answers.

Not one test here opens a socket. Every provider is handed an
:class:`httpx.MockTransport` that answers out of ``tests/fixtures/api``, the cache
is a file under ``tmp_path``, and :func:`~ayris.core.connectivity.link_up` is
pinned in an autouse fixture — so «нет интернета» is a decision the suite makes
rather than something it inherits from the machine it happens to run on.

The recorded bodies are the real ones, oddities included: the Central Bank
answers XML in windows-1251 with a nominal of 100 on the yen, Open-Meteo splits
one forecast between ``current`` and ``daily``, and Wikipedia returns a
disambiguation page with HTTP 200 and a perfectly good-looking extract. Each of
those is a way to be confidently wrong out loud, which is why the parsing is
checked against the recorded bytes and not against a dictionary written by hand.

Three groups matter more than the rest.

*The phrasing.* :func:`~ayris.actions.system.providers.weather.degrees_ru`,
:func:`~ayris.actions.system.providers.currency.roubles_ru`,
:func:`~ayris.actions.system.providers.currency.rate_ru` and
:func:`~ayris.actions.system.providers.worldtime.city_time_ru` are what the user
hears. A wrong plural or a bare ``-3`` is what a synthesiser reads as «дефис
три», so they are pure functions and they are tested as such.

*The cache.* An answer inside its time to live has to cost nothing, and these
tests assert on the request log rather than on a flag: a cache that reports a hit
and still calls the service is the bug worth catching.

*Offline.* No link and nothing stored is «нет интернета»; no link and a stale
answer is that answer plus how old it is; past ``stale_hours`` it is «нет
интернета» again. The clock is the exception — a time zone does not expire, so it
is recomputed locally and says nothing about being stale.

Groups:

* :class:`TestGeocode` — one lookup for coordinates and a time zone.
* :class:`TestForecast` — Open-Meteo's body, and the sentence built from it.
* :class:`TestRates` — windows-1251, nominals, cross rates, roubles out loud.
* :class:`TestClock` — zones, the reading, and the offline recomputation.
* :class:`TestFacts` — Wikipedia summaries and the language fallback.
* :class:`TestPage` — the summary a page publishes about itself.
* :class:`TestRouting` — which provider a phrase reaches, and with what subject.
* :class:`TestCache` — time to live, eviction, a damaged file.
* :class:`TestOffline` — the outcomes of having no connection.
* :class:`TestRetries` — what is retried, and what must never be.
* :class:`TestActions` — both actions end to end.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import httpx
import pytest

from ayris.actions.registry import registered_actions
from ayris.actions.system.instant import (
    InstantAnswer as AnswerAction,
)
from ayris.actions.system.instant import (
    SiteSummary,
    _age_ru,
    answer_now,
    detect_kind,
    set_cache,
    set_transport,
    strip_subject,
    ttl_for,
)
from ayris.actions.system.providers.base import (
    AnswerCache,
    HttpFetcher,
    InstantNotFound,
    InstantOffline,
    InstantProviderError,
    providers,
)
from ayris.actions.system.providers.base import (
    InstantAnswer as Answer,
)
from ayris.actions.system.providers.currency import (
    CurrencyProvider,
    parse_cbr,
    rate_ru,
    roubles_ru,
    split_pair,
)
from ayris.actions.system.providers.geocode import Place, geocode, parse_place
from ayris.actions.system.providers.page import (
    FactProvider,
    PageProvider,
    PageSummary,
    parse_page,
    parse_summary,
    shorten,
)
from ayris.actions.system.providers.weather import (
    Forecast,
    WeatherProvider,
    degrees_ru,
    forecast_ru,
    parse_forecast,
)
from ayris.actions.system.providers.worldtime import (
    CityTime,
    WorldTimeProvider,
    city_time_ru,
    resolve_zone,
)
from ayris.core import config as config_module
from ayris.core.errors import ActionError, ActionParamsInvalid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pytestmark = pytest.mark.unit

#: Where the recorded bodies live.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "api"

#: The sentence the recorded Moscow forecast has to produce, word for word.
#:
#: Every clause in it is a decision: 21.4 rounds to «плюс 21 градус», code 3 is
#: «облачно», an apparent temperature 0.6° away is left out, 4.6 m/s is worth
#: mentioning and rounds to five metres, and today's range comes last.
MOSCOW_SENTENCE = (
    "В Москве плюс 21 градус, облачно, ветер 5 метров в секунду. "
    "Днём от плюс 13 градусов до плюс 25 градусов."
)

#: What the recorded article page reads as: its own title, then its own summary,
#: cut after two sentences.
ARTICLE_SENTENCE = (
    "Как устроен ускоритель частиц. Ускоритель разгоняет заряженные частицы "
    "электрическим полем и удерживает их магнитным. Кольцевые машины возвращают "
    "пучок в ту же секцию много раз."
)

#: Moscow as the geocoder returns it, for the phrasing tests that need a place.
MOSCOW = Place(
    name="Москва",
    latitude=55.75222,
    longitude=37.61556,
    timezone="Europe/Moscow",
    country="Россия",
    region="Москва",
)

#: Tokyo, for the half of the tests that need somewhere abroad.
TOKYO = Place(
    name="Токио",
    latitude=35.6895,
    longitude=139.69171,
    timezone="Asia/Tokyo",
    country="Япония",
    region="Токио",
)


def _json_fixture(name: str) -> Any:
    """One recorded JSON body."""
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def cut_the_link(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report no network at all, the way an unplugged cable does.

    Patched where :class:`~ayris.actions.system.providers.base.HttpFetcher` reads
    it, so the refusal happens before the first attempt — which is the point of
    the check: asking a service with no link spends the whole timeout budget to
    learn what the OS already knew.
    """
    monkeypatch.setattr("ayris.actions.system.providers.base.link_up", lambda: False)


class Server:
    """Every service the providers know, answered from a recorded file.

    One transport for the whole module, routed by host: the geocoder, the
    forecast, the Central Bank, Wikipedia per language, and any other address at
    all — which is what :class:`~ayris.actions.system.providers.page.PageProvider`
    is pointed at.

    The knobs are what the failure tests need. ``fail`` answers one status code
    forever, ``flaky`` answers a queued status per attempt until the queue runs
    out, and ``boom`` raises a transport error the way a dropped Wi-Fi does.
    ``calls`` is the request log the cache tests assert on.
    """

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.geocode: Any = _json_fixture("geocode_moscow.json")
        self.forecast: Any = _json_fixture("forecast_moscow.json")
        self.rates: bytes = (FIXTURES / "cbr_daily.xml").read_bytes()
        self.wiki: dict[str, Any] = {"ru": _json_fixture("wikipedia_quark.json")}
        self.page: str = (FIXTURES / "page_article.html").read_text(encoding="utf-8")
        self.page_type: str = "text/html; charset=utf-8"
        self.fail: dict[str, int] = {}
        self.flaky: dict[str, list[int]] = {}
        self.boom: dict[str, int] = {}

    @property
    def urls(self) -> list[str]:
        """The addresses asked for, in order."""
        return [str(request.url) for request in self.calls]

    def transport(self) -> httpx.MockTransport:
        """A transport that answers out of the fixtures."""
        return httpx.MockTransport(self.handle)

    def fetcher(self, *, retries: int = 0, timeout_s: float = 0.5) -> HttpFetcher:
        """A client wired to this server, for driving one provider directly."""
        return HttpFetcher(timeout_s=timeout_s, retries=retries, transport=self.transport())

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Answer one request, or fail it the way the knobs say."""
        self.calls.append(request)
        host = request.url.host
        if self.boom.get(host, 0):
            self.boom[host] -= 1
            raise httpx.ConnectTimeout("сеть пропала", request=request)
        queued = self.flaky.get(host)
        if queued:
            return httpx.Response(queued.pop(0), text="служба занята")
        refusal = self.fail.get(host, 0)
        if refusal:
            return httpx.Response(refusal, text="отказ")
        return self._answer(host)

    def _answer(self, host: str) -> httpx.Response:
        """The recorded body for one host."""
        if host == "geocoding-api.open-meteo.com":
            return httpx.Response(200, json=self.geocode)
        if host == "api.open-meteo.com":
            return httpx.Response(200, json=self.forecast)
        if host == "www.cbr.ru":
            return httpx.Response(
                200,
                content=self.rates,
                headers={"content-type": "application/xml"},
            )
        if host.endswith("wikipedia.org"):
            payload = self.wiki.get(host.split(".", 1)[0])
            if payload is None:
                return httpx.Response(404, json={"title": "Not found"})
            return httpx.Response(200, json=payload)
        return httpx.Response(200, text=self.page, headers={"content-type": self.page_type})


@pytest.fixture(autouse=True)
def _linked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the OS reporting a link, whatever the runner has."""
    monkeypatch.setattr("ayris.actions.system.providers.base.link_up", lambda: True)


@pytest.fixture(autouse=True)
def server() -> Iterator[Server]:
    """The recorded services, installed for the actions and removed after.

    Autouse so no test can reach the real network even by forgetting to ask for
    a transport: the actions build their client through
    :func:`~ayris.actions.system.instant.get_transport`, and this is what it
    returns for the length of a test.
    """
    built = Server()
    set_transport(built.transport())
    yield built
    set_transport(None)


@pytest.fixture(autouse=True)
def cache(tmp_path: Path) -> Iterator[AnswerCache]:
    """A cache file per test. Nothing here touches the profile's own."""
    store = AnswerCache(tmp_path / "instant.json")
    set_cache(store)
    yield store
    set_cache(None)


@pytest.fixture
def configure(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Override ``[actions.instant]`` the way the user's settings would."""

    def apply(**values: str) -> None:
        for field, value in values.items():
            monkeypatch.setenv(f"AYRIS_ACTIONS__INSTANT__{field.upper()}", value)
        config_module.reset_config_manager()

    return apply


class TestGeocode:
    """One lookup answers where a city is and what time zone it keeps."""

    def test_a_city_becomes_coordinates_and_a_zone(self) -> None:
        place = parse_place(_json_fixture("geocode_moscow.json"), "москва")
        assert place.name == "Москва"
        assert place.latitude == pytest.approx(55.75222)
        assert place.longitude == pytest.approx(37.61556)
        assert place.timezone == "Europe/Moscow"
        assert place.country == "Россия"
        assert place.region == "Москва"

    def test_the_country_is_said_only_when_it_is_not_ours(self) -> None:
        assert parse_place(_json_fixture("geocode_moscow.json"), "москва").spoken == "Москва"
        assert parse_place(_json_fixture("geocode_tokyo.json"), "токио").spoken == "Токио, Япония"

    def test_no_results_is_a_missing_city(self) -> None:
        with pytest.raises(InstantNotFound) as raised:
            parse_place({"results": []}, "хогвартс")
        assert raised.value.user_message == "Не нашла город «хогвартс»."

    def test_a_result_without_coordinates_is_not_an_answer(self) -> None:
        with pytest.raises(InstantProviderError, match="no coordinates"):
            parse_place({"results": [{"name": "Нигде"}]}, "нигде")

    def test_a_body_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(InstantNotFound):
            parse_place("не json", "москва")

    def test_an_empty_name_never_reaches_the_service(self, server: Server) -> None:
        with pytest.raises(InstantNotFound, match="empty city"):
            geocode(server.fetcher(), "   ")
        assert server.calls == []

    def test_the_lookup_asks_for_one_result_in_russian(self, server: Server) -> None:
        geocode(server.fetcher(), "  москва  ")
        assert len(server.calls) == 1
        asked = server.calls[0].url
        assert asked.params["name"] == "москва"
        assert asked.params["count"] == "1"
        assert asked.params["language"] == "ru"


class TestForecast:
    """Open-Meteo's body, and the one sentence built out of it."""

    def test_the_recorded_body_becomes_a_forecast(self) -> None:
        forecast = parse_forecast(_json_fixture("forecast_moscow.json"), MOSCOW)
        assert forecast.temperature_c == pytest.approx(21.4)
        assert forecast.feels_like_c == pytest.approx(20.8)
        assert forecast.humidity_pct == 57
        assert forecast.wind_ms == pytest.approx(4.6)
        assert forecast.code == 3
        assert forecast.condition_ru == "облачно"
        assert forecast.high_c == pytest.approx(24.7)
        assert forecast.low_c == pytest.approx(13.1)
        assert forecast.observed_at == "2026-08-21T09:15"

    def test_the_recorded_forecast_reads_as_one_sentence(self) -> None:
        forecast = parse_forecast(_json_fixture("forecast_moscow.json"), MOSCOW)
        assert forecast_ru(forecast) == MOSCOW_SENTENCE

    def test_a_missing_temperature_is_no_answer(self) -> None:
        with pytest.raises(InstantProviderError, match="no temperature"):
            parse_forecast({"current": {"weather_code": 3}}, MOSCOW)

    def test_a_body_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(InstantProviderError, match="not an object"):
            parse_forecast(["ой"], MOSCOW)

    def test_missing_fields_shorten_the_sentence(self) -> None:
        forecast = parse_forecast({"current": {"temperature_2m": 21.4}}, MOSCOW)
        assert forecast.wind_ms is None
        assert forecast.condition_ru == ""
        assert forecast_ru(forecast) == "В Москве плюс 21 градус."

    def test_feels_like_is_said_only_when_it_differs(self) -> None:
        near = Forecast(place=MOSCOW, temperature_c=21.0, feels_like_c=20.0)
        far = Forecast(place=MOSCOW, temperature_c=21.0, feels_like_c=15.0)
        assert "ощущается" not in forecast_ru(near)
        assert "ощущается как плюс 15 градусов" in forecast_ru(far)

    def test_a_calm_wind_is_not_mentioned(self) -> None:
        calm = Forecast(place=MOSCOW, temperature_c=21.0, wind_ms=2.0)
        windy = Forecast(place=MOSCOW, temperature_c=21.0, wind_ms=9.4)
        assert "ветер" not in forecast_ru(calm)
        assert "ветер 9 метров в секунду" in forecast_ru(windy)

    def test_the_range_needs_both_ends(self) -> None:
        half = Forecast(place=MOSCOW, temperature_c=21.0, high_c=25.0)
        whole = Forecast(place=MOSCOW, temperature_c=21.0, high_c=25.0, low_c=-3.0)
        assert "Днём" not in forecast_ru(half)
        assert forecast_ru(whole).endswith("Днём от минус 3 градуса до плюс 25 градусов.")

    def test_an_unknown_code_drops_the_sky(self) -> None:
        odd = Forecast(place=MOSCOW, temperature_c=21.0, code=4)
        assert odd.condition_ru == ""
        assert forecast_ru(odd) == "В Москве плюс 21 градус."

    @pytest.mark.parametrize(
        ("value", "said"),
        [
            (0, "ноль градусов"),
            (1, "плюс 1 градус"),
            (2, "плюс 2 градуса"),
            (5, "плюс 5 градусов"),
            (11, "плюс 11 градусов"),
            (21, "плюс 21 градус"),
            (-1, "минус 1 градус"),
            (-3, "минус 3 градуса"),
            (-17, "минус 17 градусов"),
        ],
    )
    def test_degrees_are_words_and_not_signs(self, value: int, said: str) -> None:
        assert degrees_ru(value) == said

    @pytest.mark.parametrize(
        ("name", "said"),
        [
            ("Москва", "В Москве"),
            ("Санкт-Петербург", "В Санкт-Петербурге"),
            ("Волгоград", "В Волгограде"),
            ("Пермь", "В Пермь"),
            ("Нижний Новгород", "В Нижний Новгород"),
        ],
    )
    def test_the_city_is_declined_where_an_ending_is_known(self, name: str, said: str) -> None:
        """A shallow table, and a name it cannot handle is left alone."""
        place = Place(name=name, latitude=0.0, longitude=0.0, country="Россия")
        assert forecast_ru(Forecast(place=place, temperature_c=1.0)).startswith(said)

    def test_as_dict_rounds_for_the_interface(self) -> None:
        forecast = parse_forecast(_json_fixture("forecast_moscow.json"), MOSCOW)
        shown = forecast.as_dict()
        assert shown["temperature_c"] == 21.4
        assert shown["condition_ru"] == "облачно"
        assert shown["place"]["timezone"] == "Europe/Moscow"
        assert json.loads(json.dumps(shown))["humidity_pct"] == 57

    def test_the_provider_asks_two_services_and_answers_in_russian(self, server: Server) -> None:
        answer = WeatherProvider(server.fetcher()).fetch("москва")
        assert answer.kind == "weather"
        assert answer.source == "open-meteo.com"
        assert answer.message_ru == MOSCOW_SENTENCE
        assert len(server.calls) == 2
        assert server.calls[1].url.params["latitude"] == "55.75222"
        assert server.calls[1].url.params["wind_speed_unit"] == "ms"

    def test_an_unknown_city_is_not_a_forecast(self, server: Server) -> None:
        server.geocode = {"results": []}
        with pytest.raises(InstantNotFound):
            WeatherProvider(server.fetcher()).fetch("хогвартс")
        assert len(server.calls) == 1


class TestRates:
    """The Central Bank's table: its encoding, its nominals, its wording."""

    @pytest.fixture
    def table(self, server: Server) -> Any:
        return parse_cbr(server.rates)

    def test_windows_1251_names_survive(self, table: Any) -> None:
        """The bytes go to the XML parser, which honours the prologue."""
        assert table.rates["USD"].name_ru == "Доллар США"
        assert table.rates["AUD"].name_ru == "Австралийский доллар"

    def test_the_date_comes_from_the_attribute(self, table: Any) -> None:
        assert table.date == "21.08.2026"

    def test_the_nominal_is_divided_out(self, table: Any) -> None:
        """100 yen for 62.41 roubles is not one yen for 62.41 roubles."""
        assert table.rates["JPY"].nominal == 100
        assert table.rates["JPY"].value == Decimal("62.4130")
        assert table.rates["JPY"].per_unit == Decimal("62.4130") / 100
        assert table.per_unit("KZT") == Decimal("19.0455") / 100

    def test_the_rouble_is_the_unit(self, table: Any) -> None:
        assert table.per_unit("RUB") == Decimal(1)
        assert table.per_unit("usd") == Decimal("91.5043")

    def test_a_currency_the_bank_omits(self, table: Any) -> None:
        with pytest.raises(InstantNotFound) as raised:
            table.get("XXX")
        assert raised.value.user_message == "Центробанк не публикует курс XXX."

    def test_a_cross_rate_is_two_rows_divided(self, table: Any) -> None:
        assert table.cross("USD", "CNY") == Decimal("91.5043") / Decimal("12.7318")
        assert table.cross("USD", "RUB") == Decimal("91.5043")
        assert table.cross("RUB", "USD") == Decimal(1) / Decimal("91.5043")

    def test_the_whole_table_is_json_safe(self, table: Any) -> None:
        shown = json.loads(json.dumps(table.as_dict()))
        assert shown["date"] == "21.08.2026"
        assert shown["rates"]["JPY"]["nominal"] == 100
        assert shown["rates"]["JPY"]["per_unit"].startswith("0.6241")

    def test_not_xml_is_not_an_answer(self) -> None:
        with pytest.raises(InstantProviderError, match="not XML"):
            parse_cbr(b"<html><body>503</body>")

    def test_xml_without_a_single_rate_is_not_an_answer(self) -> None:
        with pytest.raises(InstantProviderError, match="no rates"):
            parse_cbr(b'<?xml version="1.0"?><ValCurs Date="21.08.2026"/>')

    @pytest.mark.parametrize(
        ("amount", "said"),
        [
            ("91.5043", "91 рубль 50 копеек"),
            ("99.1274", "99 рублей 13 копеек"),
            ("100", "100 рублей"),
            ("21", "21 рубль"),
            ("2.02", "2 рубля 2 копейки"),
            ("7.12", "7 рублей 12 копеек"),
            ("0.19", "0 рублей 19 копеек"),
            ("-1.05", "минус 1 рубль 5 копеек"),
        ],
    )
    def test_roubles_are_said_with_their_kopecks(self, amount: str, said: str) -> None:
        assert roubles_ru(Decimal(amount)) == said

    def test_a_rate_against_the_rouble_reads_as_money(self, table: Any) -> None:
        assert rate_ru(table, "USD") == (
            "Доллар — 91 рубль 50 копеек. Курс Центробанка на 21.08.2026."
        )
        assert rate_ru(table, "JPY").startswith("Иена — 0 рублей 62 копейки.")

    def test_a_cross_rate_reads_as_a_ratio(self, table: Any) -> None:
        """«евро стоит 1 доллар 8 копеек» would be nonsense, so it is a number."""
        assert rate_ru(table, "USD", "CNY") == (
            "Доллар — 7,19 юаня. По курсу Центробанка на 21.08.2026."
        )

    @pytest.mark.parametrize(
        ("phrase", "pair"),
        [
            ("доллара", ("USD", "RUB")),
            ("курс доллара", ("USD", "RUB")),
            ("сколько стоит доллар", ("USD", "RUB")),
            ("доллар к евро", ("USD", "EUR")),
            ("рубль к евро", ("RUB", "EUR")),
            ("доллар доллара", ("USD", "RUB")),
            ("евро, юань", ("EUR", "CNY")),
            ("погода в москве", ("", "RUB")),
        ],
    )
    def test_split_pair_reads_what_was_named(self, phrase: str, pair: tuple[str, str]) -> None:
        assert split_pair(phrase) == pair

    def test_the_key_is_the_pair_and_not_the_wording(self, server: Server) -> None:
        provider = CurrencyProvider(server.fetcher())
        assert provider.cache_key("курс доллара") == "rates:USD:RUB"
        assert provider.cache_key("сколько стоит доллар") == "rates:USD:RUB"
        assert provider.cache_key("доллар к юаню") == "rates:USD:CNY"

    def test_the_provider_answers_from_the_recorded_table(self, server: Server) -> None:
        answer = CurrencyProvider(server.fetcher()).fetch("курс доллара")
        assert answer.kind == "rates"
        assert answer.source == "cbr.ru"
        assert answer.message_ru.startswith("Доллар — 91 рубль 50 копеек.")
        assert answer.data["code"] == "USD"
        assert answer.data["into"] == "RUB"
        assert answer.data["nominal"] == 1
        assert answer.data["name_ru"] == "Доллар США"
        assert answer.data["date"] == "21.08.2026"

    def test_the_rouble_itself_is_an_answer(self, server: Server) -> None:
        answer = CurrencyProvider(server.fetcher()).fetch("рубль к юаню")
        assert answer.data["name_ru"] == "Российский рубль"
        assert answer.message_ru.startswith("Рубль — 0,08 юаня.")

    def test_no_currency_in_the_phrase(self, server: Server) -> None:
        with pytest.raises(InstantNotFound) as raised:
            CurrencyProvider(server.fetcher()).fetch("что-нибудь")
        assert raised.value.user_message == "Не поняла, курс какой валюты нужен."
        assert server.calls == []


class TestClock:
    """What time it is somewhere else, and why it works with no connection."""

    def test_a_zone_by_name(self) -> None:
        assert resolve_zone("Asia/Tokyo") is not None

    def test_a_place_with_no_zone_is_refused(self) -> None:
        with pytest.raises(InstantProviderError, match="no timezone"):
            resolve_zone("")

    def test_an_unknown_zone_is_refused_in_russian(self) -> None:
        with pytest.raises(InstantProviderError) as raised:
            resolve_zone("Nowhere/Nothing")
        assert raised.value.user_message == "Не смогла определить часовой пояс."

    def test_the_reading_says_how_far_ahead_it_is(self) -> None:
        local = datetime(2026, 8, 21, 21, 5, tzinfo=ZoneInfo("Europe/Moscow"))
        moment = local.astimezone(ZoneInfo("Asia/Tokyo"))
        reading = CityTime(place=TOKYO, moment=moment, local=local)
        assert reading.offset_hours == pytest.approx(6.0)
        assert city_time_ru(reading) == ("В Токио, Япония 3:05, это на 6 часов больше, чем здесь.")

    def test_the_reading_says_how_far_behind_it_is(self) -> None:
        local = datetime(2026, 8, 21, 21, 5, tzinfo=ZoneInfo("Europe/Moscow"))
        place = Place(name="Нью-Йорк", latitude=40.7, longitude=-74.0, country="США")
        moment = local.astimezone(ZoneInfo("America/New_York"))
        assert city_time_ru(CityTime(place=place, moment=moment, local=local)) == (
            "В Нью-Йорк, США 14:05, это на 7 часов меньше, чем здесь."
        )

    def test_a_half_hour_zone_is_said_in_full(self) -> None:
        local = datetime(2026, 8, 21, 21, 5, tzinfo=ZoneInfo("Europe/Moscow"))
        place = Place(name="Мумбаи", latitude=19.0, longitude=72.8, country="Индия")
        moment = local.astimezone(ZoneInfo("Asia/Kolkata"))
        assert city_time_ru(CityTime(place=place, moment=moment, local=local)) == (
            "В Мумбаи, Индия 23:35, это на 2 часа 30 минут больше, чем здесь."
        )

    def test_the_same_zone_says_so(self) -> None:
        local = datetime(2026, 8, 21, 21, 5, tzinfo=ZoneInfo("Europe/Moscow"))
        reading = CityTime(place=MOSCOW, moment=local, local=local)
        assert city_time_ru(reading) == "В Москве 21:05, столько же, сколько здесь."

    def test_the_provider_reads_the_zone_out_of_one_lookup(self, server: Server) -> None:
        server.geocode = _json_fixture("geocode_tokyo.json")
        answer = WorldTimeProvider(server.fetcher()).fetch("токио")
        zone = ZoneInfo("Asia/Tokyo")
        there = datetime.now(zone)
        here = datetime.now().astimezone()
        expected = (there.utcoffset() - here.utcoffset()).total_seconds() / 3600.0
        assert len(server.calls) == 1
        assert answer.kind == "time"
        assert answer.data["timezone"] == "Asia/Tokyo"
        assert answer.data["hour"] == there.hour
        assert answer.data["offset_hours"] == pytest.approx(expected)
        assert answer.message_ru.startswith("В Токио, Япония ")

    def test_the_clock_needs_no_network_when_the_zone_is_known(
        self,
        server: Server,
        cache: AnswerCache,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A time zone is a property of a place, so a cached one is enough."""
        provider = WorldTimeProvider(server.fetcher())
        stored = Answer(
            kind="time",
            message_ru="В Токио, Япония 3:05, это на 6 часов больше, чем здесь.",
            data={"timezone": "Asia/Tokyo", "place": TOKYO.as_dict()},
            source="open-meteo.com",
        )
        cache.put(provider.cache_key("токио"), stored.at(1_000_000.0))
        cut_the_link(monkeypatch)
        resolved = answer_now(provider, "токио", now=1_000_000.0 + 90 * 86400)
        assert server.calls == []
        assert resolved.cached is True
        assert resolved.answer.stale is False
        assert resolved.answer.data["hour"] == datetime.now(ZoneInfo("Asia/Tokyo")).hour
        assert resolved.answer.message_ru.startswith("В Токио, Япония ")

    def test_a_cached_entry_without_a_zone_falls_back_to_asking(self, server: Server) -> None:
        provider = WorldTimeProvider(server.fetcher())
        assert provider.refresh(Answer(kind="time", message_ru="когда-то", data={})) is None

    def test_a_cached_entry_with_a_broken_zone_falls_back_to_asking(self, server: Server) -> None:
        provider = WorldTimeProvider(server.fetcher())
        stored = Answer(kind="time", message_ru="когда-то", data={"timezone": "Nowhere/Nothing"})
        assert provider.refresh(stored) is None


class TestFacts:
    """Wikipedia's summary endpoint, and what to do when it has nothing."""

    def test_the_recorded_article_becomes_a_summary(self) -> None:
        summary = parse_summary(_json_fixture("wikipedia_quark.json"), "Кварк")
        assert summary.title == "Кварк"
        assert summary.source == "wikipedia.org"
        assert summary.url.startswith("https://ru.wikipedia.org/wiki/")
        assert summary.extract.startswith("Кварк — фундаментальная частица")

    def test_two_sentences_are_read_out_and_the_third_is_not(self) -> None:
        summary = parse_summary(_json_fixture("wikipedia_quark.json"), "Кварк")
        said = summary.spoken()
        assert said.startswith("Кварк — фундаментальная частица")
        assert said.endswith("конфайнмента.")
        assert "Всего известно" not in said
        assert "Всего известно" in summary.extract

    def test_a_disambiguation_page_is_not_an_answer(self) -> None:
        """It arrives with HTTP 200 and a plausible extract, and answers nothing."""
        with pytest.raises(InstantNotFound) as raised:
            parse_summary(_json_fixture("wikipedia_disambiguation.json"), "Ключ")
        assert raised.value.user_message == (
            "«Ключ» — так называется несколько разных вещей. Уточните, пожалуйста."
        )

    def test_an_empty_extract_is_not_an_answer(self) -> None:
        with pytest.raises(InstantNotFound) as raised:
            parse_summary({"title": "Пусто", "extract": "   "}, "пусто")
        assert raised.value.user_message == "Не нашла краткой справки по «пусто»."

    def test_a_body_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(InstantProviderError, match="not an object"):
            parse_summary(["ой"], "кварк")

    @pytest.mark.parametrize(
        ("text", "sentences", "said"),
        [
            ("Раз.  Два.\nТри.", 2, "Раз. Два."),
            ("Это т. е. одно предложение. Второе.", 1, "Это т. е. одно предложение."),
            ("Один — 1. Два.", 1, "Один — 1."),
            ("Вопрос? Ответ. Ещё.", 2, "Вопрос? Ответ."),
            ("   ", 2, ""),
            ("Без точки в конце", 2, "Без точки в конце"),
        ],
    )
    def test_shorten_ends_on_a_sentence(self, text: str, sentences: int, said: str) -> None:
        assert shorten(text, sentences=sentences) == said

    def test_the_title_is_prefixed_only_when_the_extract_lacks_it(self) -> None:
        same = PageSummary(title="Кварк", extract="Кварк — это частица.")
        other = PageSummary(title="Ускоритель", extract="Машина разгоняет частицы.")
        assert same.spoken() == "Кварк — это частица."
        assert other.spoken() == "Ускоритель. Машина разгоняет частицы."

    def test_a_page_with_no_summary_says_so(self) -> None:
        assert PageSummary(title="Ключ", extract="").spoken() == (
            "«Ключ» — нашла страницу, но краткого описания на ней нет."
        )

    def test_the_title_is_percent_encoded_with_underscores(self, server: Server) -> None:
        FactProvider(server.fetcher()).fetch("Большой взрыв")
        asked = server.urls[0]
        assert "/api/rest_v1/page/summary/" in asked
        assert "_" in asked.rsplit("/", 1)[-1]
        assert "%D0%91" in asked

    def test_russian_is_tried_first(self, server: Server) -> None:
        answer = FactProvider(server.fetcher()).fetch("Кварк")
        assert len(server.calls) == 1
        assert server.calls[0].url.host == "ru.wikipedia.org"
        assert answer.data["language"] == "ru"
        assert answer.source == "wikipedia.org"

    def test_english_answers_what_russian_cannot(self, server: Server) -> None:
        server.wiki = {
            "ru": _json_fixture("wikipedia_disambiguation.json"),
            "en": _json_fixture("wikipedia_quark.json"),
        }
        answer = FactProvider(server.fetcher()).fetch("Ключ")
        assert [request.url.host for request in server.calls] == [
            "ru.wikipedia.org",
            "en.wikipedia.org",
        ]
        assert answer.data["language"] == "en"

    def test_the_russian_refusal_is_the_one_reported(self, server: Server) -> None:
        """An English 404 after a Russian «уточните» says nothing new."""
        server.wiki = {"ru": _json_fixture("wikipedia_disambiguation.json")}
        with pytest.raises(InstantNotFound) as raised:
            FactProvider(server.fetcher()).fetch("Ключ")
        assert "несколько разных вещей" in raised.value.user_message
        assert len(server.calls) == 2

    def test_no_article_anywhere(self, server: Server) -> None:
        server.wiki = {}
        with pytest.raises(InstantNotFound):
            FactProvider(server.fetcher()).fetch("Кварк")
        assert len(server.calls) == 2

    def test_an_empty_subject_never_reaches_the_service(self, server: Server) -> None:
        with pytest.raises(InstantNotFound) as raised:
            FactProvider(server.fetcher()).fetch("  ")
        assert raised.value.user_message == "Не поняла, о чём рассказать."
        assert server.calls == []


class TestPage:
    """The summary a page publishes about itself — any page, not just a wiki."""

    def test_open_graph_wins_over_the_plain_description(self) -> None:
        html = (FIXTURES / "page_article.html").read_text(encoding="utf-8")
        summary = parse_page(html, "https://nkj.ru/archive/1.html")
        assert summary.title == "Как устроен ускоритель частиц"
        assert summary.extract.startswith("Ускоритель разгоняет заряженные частицы")
        assert "meta name=description" not in summary.extract
        assert summary.source == "nkj.ru"
        assert summary.url == "https://nkj.ru/archive/1.html"

    def test_the_plain_description_is_used_when_there_is_no_open_graph(self) -> None:
        html = (
            "<html><head><title>Док</title>"
            '<meta name="description" content="Описание из meta.">'
            "</head><body></body></html>"
        )
        summary = parse_page(html, "https://docs.example.org/x")
        assert summary.title == "Док"
        assert summary.extract == "Описание из meta."

    def test_the_first_real_paragraph_is_the_last_resort(self) -> None:
        html = (
            "<html><head><title>Заметка</title></head><body>"
            "<p>Меню</p>"
            "<p>Первый настоящий абзац, достаточно длинный, чтобы быть текстом.</p>"
            "</body></html>"
        )
        summary = parse_page(html, "https://example.org/note")
        assert summary.extract == "Первый настоящий абзац, достаточно длинный, чтобы быть текстом."

    def test_a_page_with_nothing_to_say_is_refused(self) -> None:
        html = (FIXTURES / "page_bare.html").read_text(encoding="utf-8")
        with pytest.raises(InstantNotFound) as raised:
            parse_page(html, "https://example.org/bare")
        assert raised.value.user_message == (
            "На example.org нет краткого описания — могу просто открыть страницу."
        )

    def test_attribute_order_and_single_quotes_do_not_matter(self) -> None:
        """The reason this is an HTML parser and not a regular expression."""
        html = "<html><head><meta content='Так тоже бывает.' property='og:description'></head>"
        assert parse_page(html, "https://example.org/").extract == "Так тоже бывает."

    def test_entities_are_unescaped_before_they_are_read_out(self) -> None:
        html = (
            "<html><head><title>Кавычки</title>"
            '<meta property="og:description" content="Он сказал &quot;да&quot; и ушёл.">'
            "</head>"
        )
        assert parse_page(html, "https://example.org/").spoken().endswith('"да" и ушёл.')

    def test_the_provider_reads_the_recorded_article(self, server: Server) -> None:
        answer = PageProvider(server.fetcher()).fetch("https://nkj.ru/archive/1.html")
        assert answer.kind == "page"
        assert answer.source == "nkj.ru"
        assert answer.message_ru == ARTICLE_SENTENCE

    def test_the_number_of_sentences_is_the_caller_s(self, server: Server) -> None:
        answer = PageProvider(server.fetcher(), sentences=1).fetch("https://nkj.ru/1.html")
        assert answer.message_ru == (
            "Как устроен ускоритель частиц. Ускоритель разгоняет заряженные частицы "
            "электрическим полем и удерживает их магнитным."
        )

    def test_a_file_is_not_a_page(self, server: Server) -> None:
        server.page_type = "application/pdf"
        with pytest.raises(InstantProviderError) as raised:
            PageProvider(server.fetcher()).fetch("https://example.org/manual.pdf")
        assert raised.value.user_message == (
            "По этой ссылке не страница, а файл — прочитать не смогу."
        )


class TestRouting:
    """Which provider a phrase reaches, and what subject arrives with it."""

    @pytest.fixture
    def table(self, server: Server) -> Any:
        return providers(server.fetcher())

    @pytest.mark.parametrize(
        ("phrase", "kind"),
        [
            ("какая погода в москве", "weather"),
            ("погода", "weather"),
            ("прогноз на день", "weather"),
            ("жарко ли сегодня", "weather"),
            ("какая температура на улице", "weather"),
            ("курс доллара", "rates"),
            ("сколько стоит доллар", "rates"),
            ("доллар к юаню", "rates"),
            ("сколько стоит юань", "rates"),
            ("сколько времени в токио", "time"),
            ("который час в лондоне", "time"),
            ("время", "time"),
            ("что такое кварк", "fact"),
            ("кто такой тьюринг", "fact"),
            ("расскажи про квантовую механику", "fact"),
            ("что такое доллар", "fact"),
            ("что такое рубль", "fact"),
            ("сколько", ""),
            ("привет", ""),
            ("", ""),
        ],
    )
    def test_a_phrase_finds_its_provider(self, table: Any, phrase: str, kind: str) -> None:
        """The longest trigger wins, which is why «что такое доллар» is a fact."""
        assert detect_kind(phrase, table) == kind

    @pytest.mark.parametrize(
        ("phrase", "kind", "subject"),
        [
            ("какая погода в питере", "weather", "питере"),
            ("а подскажи пожалуйста погоду в сочи", "weather", "сочи"),
            ("погода", "weather", ""),
            ("жарко ли сегодня", "weather", ""),
            ("сколько времени в токио", "time", "токио"),
            ("который час в нью-йорке", "time", "нью-йорке"),
            ("что такое кварк", "fact", "кварк"),
            ("расскажи про квантовую механику", "fact", "квантовую механику"),
        ],
    )
    def test_the_subject_is_what_is_left(
        self,
        table: Any,
        phrase: str,
        kind: str,
        subject: str,
    ) -> None:
        assert strip_subject(phrase, table[kind]) == subject

    def test_the_rates_provider_keeps_its_own_trigger_words(self, table: Any) -> None:
        """«доллар» both routes the phrase and names the currency."""
        assert table["rates"].keeps_triggers is True
        assert split_pair(strip_subject("сколько стоит доллар", table["rates"])) == ("USD", "RUB")
        assert split_pair(strip_subject("курс доллара к евро", table["rates"])) == ("USD", "EUR")

    @pytest.mark.parametrize(
        ("kind", "seconds"),
        [("weather", 600.0), ("rates", 3600.0), ("fact", 86400.0), ("page", 86400.0)],
    )
    def test_each_kind_has_its_own_time_to_live(self, kind: str, seconds: float) -> None:
        assert ttl_for(kind) == seconds

    def test_the_clock_is_never_cached_as_a_reading(self) -> None:
        """It is stored for its time zone, and recomputed every time."""
        assert ttl_for("time") == 0.0

    def test_an_unknown_kind_gets_the_slowest_window(self) -> None:
        assert ttl_for("что-то") == 86400.0

    def test_the_settings_move_the_window(self, configure: Callable[..., None]) -> None:
        configure(weather_ttl_min="1", rates_ttl_min="120")
        assert ttl_for("weather") == 60.0
        assert ttl_for("rates") == 7200.0


class TestCache:
    """The one thing that keeps a free API inside its rate limit."""

    def test_an_answer_goes_in_and_comes_back(self, cache: AnswerCache) -> None:
        answer = Answer(kind="fact", message_ru="ответ", source="x").at(1000.0)
        cache.put("fact:кварк", answer)
        stored = cache.get("fact:кварк", ttl_sec=60.0, now=1030.0)
        assert stored is not None
        assert stored.message_ru == "ответ"
        assert stored.fetched_at == 1000.0

    def test_past_the_window_it_is_a_miss(self, cache: AnswerCache) -> None:
        cache.put("fact:кварк", Answer(kind="fact", message_ru="ответ").at(1000.0))
        assert cache.get("fact:кварк", ttl_sec=60.0, now=1100.0) is None
        assert cache.peek("fact:кварк") is not None

    def test_staleness_is_decided_when_read_and_not_when_stored(self, cache: AnswerCache) -> None:
        aged = Answer(kind="fact", message_ru="ответ").at(1000.0).aged(now=9000.0)
        cache.put("fact:кварк", aged)
        stored = cache.peek("fact:кварк")
        assert stored is not None
        assert stored.stale is False

    def test_a_damaged_file_is_a_miss_and_not_an_error(self, cache: AnswerCache) -> None:
        cache.path.parent.mkdir(parents=True, exist_ok=True)
        cache.path.write_text("{это не json", encoding="utf-8")
        assert cache.peek("fact:кварк") is None
        cache.put("fact:кварк", Answer(kind="fact", message_ru="ответ").at(1.0))
        assert cache.peek("fact:кварк") is not None

    def test_an_unusable_entry_is_a_miss(self) -> None:
        assert Answer.from_dict({"kind": "fact"}) is None
        assert Answer.from_dict({"kind": "fact", "message_ru": ""}) is None
        assert Answer.from_dict({}) is None

    def test_the_oldest_entries_go_when_the_file_grows(self, cache: AnswerCache) -> None:
        for index in range(AnswerCache.limit + 6):
            answer = Answer(kind="fact", message_ru=f"ответ {index}").at(1000.0 + index)
            cache.put(f"fact:{index}", answer)
        assert cache.peek("fact:0") is None
        assert cache.peek("fact:5") is None
        assert cache.peek(f"fact:{AnswerCache.limit + 5}") is not None
        assert len(json.loads(cache.path.read_text(encoding="utf-8"))) == AnswerCache.limit

    def test_the_key_ignores_spacing_and_case(self, server: Server) -> None:
        provider = WeatherProvider(server.fetcher())
        assert provider.cache_key("  Москва ") == "weather:москва"
        assert provider.cache_key("нижний   новгород") == "weather:нижний новгород"

    def test_a_fresh_answer_costs_no_request(self, server: Server, cache: AnswerCache) -> None:
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.", source="меteo")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        resolved = answer_now(provider, "москва", now=1060.0)
        assert server.calls == []
        assert resolved.cached is True
        assert resolved.answer.message_ru == "В Москве плюс 5 градусов."

    def test_past_the_window_it_asks_again(self, server: Server, cache: AnswerCache) -> None:
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        resolved = answer_now(provider, "москва", now=1000.0 + 3600.0)
        assert len(server.calls) == 2
        assert resolved.cached is False
        assert resolved.answer.message_ru == MOSCOW_SENTENCE

    def test_fresh_asks_anyway(self, server: Server, cache: AnswerCache) -> None:
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        resolved = answer_now(provider, "москва", fresh=True, now=1060.0)
        assert len(server.calls) == 2
        assert resolved.cached is False

    def test_what_came_from_the_network_is_stored(self, server: Server, cache: AnswerCache) -> None:
        provider = WeatherProvider(server.fetcher())
        answer_now(provider, "москва", now=1000.0)
        stored = cache.peek(provider.cache_key("москва"))
        assert stored is not None
        assert stored.fetched_at == 1000.0
        assert stored.message_ru == MOSCOW_SENTENCE


class TestOffline:
    """No connection: the answer, the caveat, or the honest refusal."""

    def test_nothing_cached_is_nothing_to_say(
        self,
        server: Server,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cut_the_link(monkeypatch)
        with pytest.raises(InstantOffline) as raised:
            answer_now(WeatherProvider(server.fetcher()), "москва", now=1000.0)
        assert raised.value.user_message == "Нет интернета."
        assert server.calls == []

    def test_a_stale_answer_is_read_out_with_its_age(
        self,
        server: Server,
        cache: AnswerCache,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        cut_the_link(monkeypatch)
        resolved = answer_now(provider, "москва", now=1000.0 + 2 * 3600.0)
        assert resolved.cached is True
        assert resolved.answer.stale is True
        assert resolved.answer.age_sec(now=1000.0 + 2 * 3600.0) == pytest.approx(7200.0)

    def test_too_old_is_not_an_answer_even_with_a_caveat(
        self,
        server: Server,
        cache: AnswerCache,
        configure: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        configure(stale_hours="1")
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        cut_the_link(monkeypatch)
        with pytest.raises(InstantOffline):
            answer_now(provider, "москва", now=1000.0 + 3 * 3600.0)

    def test_a_service_that_never_answers_falls_back_to_the_cache(
        self,
        server: Server,
        cache: AnswerCache,
    ) -> None:
        """The link is up and the service is down, which is the same for a user."""
        server.fail["geocoding-api.open-meteo.com"] = 503
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("москва"), stored.at(1000.0))
        resolved = answer_now(provider, "москва", now=1000.0 + 3600.0)
        assert resolved.answer.stale is True
        assert resolved.answer.message_ru == "В Москве плюс 5 градусов."

    def test_a_missing_answer_is_not_hidden_behind_the_cache(
        self,
        server: Server,
        cache: AnswerCache,
    ) -> None:
        """«такого города нет» is an answer, and it must not serve yesterday's."""
        server.geocode = {"results": []}
        provider = WeatherProvider(server.fetcher())
        stored = Answer(kind="weather", message_ru="В Москве плюс 5 градусов.")
        cache.put(provider.cache_key("хогвартс"), stored.at(1000.0))
        with pytest.raises(InstantNotFound):
            answer_now(provider, "хогвартс", now=1000.0 + 3600.0)


class TestRetries:
    """What a failing service earns: another attempt, or none at all."""

    def test_no_link_means_no_attempt(
        self,
        server: Server,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cut_the_link(monkeypatch)
        with pytest.raises(InstantOffline, match="no network link"):
            server.fetcher(retries=2).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert server.calls == []

    def test_too_many_requests_is_never_retried(self, server: Server) -> None:
        """429 means the opposite of «try again»."""
        server.fail["www.cbr.ru"] = 429
        with pytest.raises(InstantProviderError) as raised:
            server.fetcher(retries=2).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert raised.value.user_message == ("Слишком много запросов к сервису, попробуйте позже.")
        assert len(server.calls) == 1

    def test_a_missing_page_is_not_retried_either(self, server: Server) -> None:
        server.wiki = {}
        with pytest.raises(InstantNotFound):
            server.fetcher(retries=2).get_json("https://ru.wikipedia.org/api/rest_v1/x")
        assert len(server.calls) == 1

    def test_a_refusal_is_not_retried(self, server: Server) -> None:
        server.fail["example.org"] = 403
        with pytest.raises(InstantProviderError) as raised:
            server.fetcher(retries=2).get("https://example.org/")
        assert raised.value.user_message == "Сервис отклонил запрос."
        assert len(server.calls) == 1

    def test_a_broken_service_is_retried_and_then_works(self, server: Server) -> None:
        server.flaky["www.cbr.ru"] = [500]
        response = server.fetcher(retries=1).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert response.status_code == 200
        assert len(server.calls) == 2

    def test_a_broken_service_that_stays_broken_is_offline(self, server: Server) -> None:
        server.fail["www.cbr.ru"] = 502
        with pytest.raises(InstantOffline) as raised:
            server.fetcher(retries=1).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert raised.value.user_message == (
            "Сервис не отвечает. Проверьте подключение к интернету."
        )
        assert len(server.calls) == 2

    def test_a_dropped_connection_is_retried(self, server: Server) -> None:
        server.boom["www.cbr.ru"] = 1
        response = server.fetcher(retries=1).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert response.status_code == 200
        assert len(server.calls) == 2

    def test_a_connection_that_stays_dropped_is_offline(self, server: Server) -> None:
        server.boom["www.cbr.ru"] = 5
        with pytest.raises(InstantOffline, match="ConnectTimeout"):
            server.fetcher(retries=1).get("https://www.cbr.ru/scripts/XML_daily.asp")
        assert len(server.calls) == 2

    def test_a_body_that_is_not_json_is_not_a_retry(self, server: Server) -> None:
        with pytest.raises(InstantProviderError) as raised:
            server.fetcher(retries=2).get_json("https://example.org/page.html")
        assert raised.value.user_message == "Сервис ответил непонятно."
        assert len(server.calls) == 1

    def test_the_deadline_covers_every_attempt(self) -> None:
        assert HttpFetcher(timeout_s=2.0, retries=2).deadline_sec == pytest.approx(8.0)
        assert HttpFetcher(timeout_s=6.0, retries=0).deadline_sec == pytest.approx(8.0)


class TestActions:
    """Both actions end to end, against the recorded services."""

    def test_both_are_in_the_registry(self) -> None:
        names = {item.name for item in registered_actions()}
        assert {"InstantAnswer", "SiteSummary"} <= names

    def test_the_weather_is_one_sentence_and_the_facts_beside_it(self, server: Server) -> None:
        result = AnswerAction().run(AnswerAction.Params(query="какая погода в москве"))
        assert result.ok is True
        assert result.message_ru == MOSCOW_SENTENCE
        assert result.data["kind"] == "weather"
        assert result.data["subject"] == "москве"
        assert result.data["cached"] is False
        assert result.data["stale"] is False
        assert result.data["source"] == "open-meteo.com"
        assert result.value is not None
        assert result.value["data"]["temperature_c"] == 21.4
        assert len(server.calls) == 2

    def test_the_second_question_costs_nothing(self, server: Server) -> None:
        AnswerAction().run(AnswerAction.Params(query="какая погода в москве"))
        again = AnswerAction().run(AnswerAction.Params(query="а какая сейчас погода в москве"))
        assert again.data["cached"] is True
        assert again.message_ru == MOSCOW_SENTENCE
        assert len(server.calls) == 2

    def test_fresh_asks_again(self, server: Server) -> None:
        AnswerAction().run(AnswerAction.Params(query="какая погода в москве"))
        again = AnswerAction().run(AnswerAction.Params(query="какая погода в москве", fresh=True))
        assert again.data["cached"] is False
        assert len(server.calls) == 4

    def test_no_city_named_means_the_configured_one(
        self,
        server: Server,
        configure: Callable[..., None],
    ) -> None:
        configure(city="Казань")
        result = AnswerAction().run(AnswerAction.Params(query="какая погода"))
        assert result.data["subject"] == "Казань"
        assert server.calls[0].url.params["name"] == "Казань"

    def test_an_explicit_kind_takes_the_query_as_it_is(self, server: Server) -> None:
        """«погода» in the phrase would otherwise be stripped out of a city name."""
        result = AnswerAction().run(AnswerAction.Params(query="Санкт-Петербург", kind="weather"))
        assert result.data["subject"] == "Санкт-Петербург"
        assert server.calls[0].url.params["name"] == "Санкт-Петербург"

    def test_a_phrase_nobody_claims_is_a_parameter_problem(self) -> None:
        with pytest.raises(ActionParamsInvalid) as raised:
            AnswerAction().run(AnswerAction.Params(query="спой мне песню"))
        assert raised.value.user_message == (
            "Не поняла, что именно узнать. Могу сказать погоду, курс, время или справку."
        )
        assert [problem.field for problem in raised.value.problems] == ["kind"]

    def test_an_unknown_kind_is_a_parameter_problem(self) -> None:
        with pytest.raises(ActionParamsInvalid, match="kind='гороскоп'"):
            AnswerAction().run(AnswerAction.Params(query="что там", kind="гороскоп"))

    def test_a_missing_city_is_a_handled_failure(self, server: Server) -> None:
        server.geocode = {"results": []}
        result = AnswerAction().run(AnswerAction.Params(query="погода в хогвартсе"))
        assert result.ok is False
        assert result.message_ru == "Не нашла город «хогвартсе»."

    def test_the_rate_is_read_as_money(self) -> None:
        result = AnswerAction().run(AnswerAction.Params(query="какой сегодня курс доллара"))
        assert result.ok is True
        assert result.message_ru == ("Доллар — 91 рубль 50 копеек. Курс Центробанка на 21.08.2026.")
        assert result.data["kind"] == "rates"

    def test_a_fact_is_two_sentences(self) -> None:
        result = AnswerAction().run(AnswerAction.Params(query="что такое кварк"))
        assert result.ok is True
        assert result.data["kind"] == "fact"
        assert result.message_ru.startswith("Кварк — фундаментальная частица")
        assert "Всего известно" not in result.message_ru

    def test_the_time_somewhere_else(self, server: Server) -> None:
        server.geocode = _json_fixture("geocode_tokyo.json")
        result = AnswerAction().run(AnswerAction.Params(query="сколько времени в токио"))
        assert result.data["kind"] == "time"
        assert result.data["subject"] == "токио"
        assert result.message_ru.startswith("В Токио, Япония ")

    @pytest.mark.parametrize(
        ("age_sec", "said"),
        [
            (40 * 60.0, "40 минут назад"),
            (2 * 3600.0, "2 часа назад"),
            (25 * 3600.0, "1 день назад"),
            (3 * 86400.0, "3 дня назад"),
        ],
    )
    def test_a_stale_answer_says_how_old_it_is(
        self,
        server: Server,
        cache: AnswerCache,
        configure: Callable[..., None],
        monkeypatch: pytest.MonkeyPatch,
        age_sec: float,
        said: str,
    ) -> None:
        configure(stale_hours="720")
        cache.put(
            "weather:москве",
            Answer(kind="weather", message_ru="В Москве плюс 5 градусов.").at(
                time.time() - age_sec
            ),
        )
        cut_the_link(monkeypatch)
        result = AnswerAction().run(AnswerAction.Params(query="какая погода в москве"))
        assert result.ok is True
        assert result.data["stale"] is True
        assert result.message_ru == (
            f"В Москве плюс 5 градусов. Данные {said}, интернета сейчас нет."
        )
        assert server.calls == []

    def test_an_answer_a_moment_old_is_not_dated(self) -> None:
        """Unreachable through the action, which is the point of checking it here.

        The shortest time to live the settings accept is a minute, so an answer
        under a minute old is always a fresh cache hit and never carries the
        caveat. The branch still exists, because ``stale_hours`` is what decides
        it and a provider may be given a window of its own later.
        """
        assert _age_ru(30.0) == "только что"
        assert _age_ru(0.0) == "только что"

    def test_no_connection_and_nothing_stored_is_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cut_the_link(monkeypatch)
        with pytest.raises(InstantOffline) as raised:
            AnswerAction().run(AnswerAction.Params(query="какая погода в москве"))
        assert raised.value.user_message == "Нет интернета."

    def test_a_page_is_summarised_out_loud(self, server: Server) -> None:
        result = SiteSummary().run(SiteSummary.Params(url="nkj.ru/archive/1.html"))
        assert result.ok is True
        assert result.message_ru == ARTICLE_SENTENCE
        assert result.data["url"] == "https://nkj.ru/archive/1.html"
        assert result.data["source"] == "nkj.ru"
        assert result.data["cached"] is False

    def test_the_same_page_twice_is_one_request(self, server: Server) -> None:
        SiteSummary().run(SiteSummary.Params(url="https://nkj.ru/archive/1.html"))
        again = SiteSummary().run(SiteSummary.Params(url="nkj.ru/archive/1.html"))
        assert again.data["cached"] is True
        assert len(server.calls) == 1

    def test_fewer_sentences_when_asked(self) -> None:
        result = SiteSummary().run(
            SiteSummary.Params(url="https://nkj.ru/archive/1.html", sentences=1)
        )
        assert result.message_ru.endswith("удерживает их магнитным.")

    def test_a_page_with_no_summary_is_a_handled_failure(self, server: Server) -> None:
        server.page = (FIXTURES / "page_bare.html").read_text(encoding="utf-8")
        result = SiteSummary().run(SiteSummary.Params(url="example.org/bare"))
        assert result.ok is False
        assert result.message_ru == (
            "На example.org нет краткого описания — могу просто открыть страницу."
        )

    def test_a_phrase_is_not_an_address(self) -> None:
        with pytest.raises(ActionParamsInvalid):
            SiteSummary().run(SiteSummary.Params(url="расскажи что там на сайте"))

    def test_no_connection_is_said_and_not_raised_as_a_traceback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cut_the_link(monkeypatch)
        with pytest.raises(ActionError) as raised:
            SiteSummary().run(SiteSummary.Params(url="nkj.ru/archive/1.html"))
        assert raised.value.user_message == "Нет интернета."
