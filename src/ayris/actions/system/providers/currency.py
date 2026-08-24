"""Exchange rates, from the Central Bank of Russia.

``https://www.cbr.ru/scripts/XML_daily.asp`` is the official daily rate, it needs
no key, and it is the number a Russian speaker means by «курс доллара» — a
commercial aggregator would answer with a market quote that differs from what
every bank and every price list in the country uses.

Two things about that endpoint shape this module. It answers **XML in
windows-1251**, declared in the prologue, so the bytes are handed to
:mod:`xml.etree` and not to a string that was already decoded wrongly. And its
numbers are **European-formatted with a nominal**: the yen arrives as
``<Nominal>100</Nominal><Value>52,3412</Value>``, meaning one hundred yen cost
52.34 roubles, and reading `Value` alone would tell the user a yen costs fifty
roubles.

Cross rates are computed here rather than requested: «сколько евро в долларе»
is two rows of the same table divided, and there is no second call to make.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, ClassVar, Final
from xml.etree import ElementTree

from ayris.actions.system.providers.base import (
    InstantAnswer,
    InstantNotFound,
    InstantProvider,
    InstantProviderError,
)
from ayris.nlu.numbers import plural_form

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CBR_URL",
    "CURRENCY_ALIASES",
    "CurrencyProvider",
    "Rate",
    "RateTable",
    "parse_cbr",
    "rate_ru",
    "roubles_ru",
    "split_pair",
]

#: The Central Bank's daily rates. XML, windows-1251, updated once a business day.
CBR_URL: Final = "https://www.cbr.ru/scripts/XML_daily.asp"

#: Spoken currency names to ISO codes.
#:
#: Every case a user says the name in: the nominative for a typed query, the
#: genitive for «курс доллара», which is how the question is actually asked, and
#: the dative for the second half of «доллар к юаню». The rouble is here as well
#: even though the table has no row for it — :class:`RateTable` answers it as 1,
#: so «курс рубля к евро» works.
CURRENCY_ALIASES: Final[Mapping[str, str]] = {
    "доллар": "USD",
    "доллара": "USD",
    "доллару": "USD",
    "доллары": "USD",
    "долларов": "USD",
    "бакс": "USD",
    "бакса": "USD",
    "usd": "USD",
    "евро": "EUR",
    "eur": "EUR",
    "рубль": "RUB",
    "рубля": "RUB",
    "рублю": "RUB",
    "рублей": "RUB",
    "руб": "RUB",
    "rub": "RUB",
    "юань": "CNY",
    "юаня": "CNY",
    "юаню": "CNY",
    "юаней": "CNY",
    "cny": "CNY",
    "фунт": "GBP",
    "фунта": "GBP",
    "фунту": "GBP",
    "стерлинг": "GBP",
    "стерлингов": "GBP",
    "gbp": "GBP",
    "иена": "JPY",
    "иены": "JPY",
    "иене": "JPY",
    "йена": "JPY",
    "йены": "JPY",
    "йене": "JPY",
    "jpy": "JPY",
    "франк": "CHF",
    "франка": "CHF",
    "франку": "CHF",
    "chf": "CHF",
    "тенге": "KZT",
    "kzt": "KZT",
    "гривна": "UAH",
    "гривны": "UAH",
    "гривне": "UAH",
    "гривен": "UAH",
    "uah": "UAH",
    "лира": "TRY",
    "лиры": "TRY",
    "лире": "TRY",
    "try": "TRY",
    "дирхам": "AED",
    "дирхама": "AED",
    "дирхаму": "AED",
    "aed": "AED",
    "белорусский": "BYN",
    "byn": "BYN",
    "драм": "AMD",
    "драма": "AMD",
    "драму": "AMD",
    "amd": "AMD",
    "лари": "GEL",
    "gel": "GEL",
    "рупия": "INR",
    "рупии": "INR",
    "inr": "INR",
    "вона": "KRW",
    "воны": "KRW",
    "воне": "KRW",
    "krw": "KRW",
}

#: Currency codes in the genitive, for «курс доллара».
_CODE_NAMES_RU: Final[Mapping[str, tuple[str, str, str]]] = {
    "USD": ("доллар", "доллара", "долларов"),
    "EUR": ("евро", "евро", "евро"),
    "RUB": ("рубль", "рубля", "рублей"),
    "CNY": ("юань", "юаня", "юаней"),
    "GBP": ("фунт", "фунта", "фунтов"),
    "JPY": ("иена", "иены", "иен"),
    "CHF": ("франк", "франка", "франков"),
    "KZT": ("тенге", "тенге", "тенге"),
    "UAH": ("гривна", "гривны", "гривен"),
    "TRY": ("лира", "лиры", "лир"),
    "AED": ("дирхам", "дирхама", "дирхамов"),
    "BYN": ("белорусский рубль", "белорусских рубля", "белорусских рублей"),
    "AMD": ("драм", "драма", "драмов"),
    "GEL": ("лари", "лари", "лари"),
    "INR": ("рупия", "рупии", "рупий"),
    "KRW": ("вона", "воны", "вон"),
}

#: The rouble, which the table never lists because it is the unit.
_BASE: Final = "RUB"


@dataclass(frozen=True, slots=True)
class Rate:
    """One row of the Central Bank's table."""

    code: str
    nominal: int
    value: Decimal
    name_ru: str = ""

    @property
    def per_unit(self) -> Decimal:
        """Roubles for one unit, the nominal already divided out."""
        return self.value / Decimal(self.nominal or 1)


@dataclass(frozen=True, slots=True)
class RateTable:
    """A whole publication: every rate, and the date it is for."""

    date: str
    rates: Mapping[str, Rate]

    def get(self, code: str) -> Rate:
        """One currency by ISO code.

        Raises:
            InstantNotFound: The Central Bank does not publish that currency.
        """
        rate = self.rates.get(code.upper())
        if rate is None:
            raise InstantNotFound(
                f"cbr publishes no rate for {code!r}",
                user_message=f"Центробанк не публикует курс {code}.",
            )
        return rate

    def per_unit(self, code: str) -> Decimal:
        """Roubles for one unit of ``code``; the rouble itself is 1."""
        return Decimal(1) if code.upper() == _BASE else self.get(code).per_unit

    def cross(self, code: str, into: str) -> Decimal:
        """How many ``into`` one ``code`` buys, through the rouble.

        Raises:
            InstantNotFound: Either side is not published.
            InstantProviderError: A published rate of zero, which would divide by it.
        """
        target = self.per_unit(into)
        if not target:
            raise InstantProviderError(f"cbr rate for {into!r} is zero")
        return self.per_unit(code) / target

    def as_dict(self) -> dict[str, object]:
        """JSON-safe mapping of the whole table, for the cache and the interface."""
        return {
            "date": self.date,
            "rates": {
                code: {
                    "nominal": rate.nominal,
                    "value": str(rate.value),
                    "per_unit": str(rate.per_unit),
                    "name_ru": rate.name_ru,
                }
                for code, rate in sorted(self.rates.items())
            },
        }


def parse_cbr(body: bytes) -> RateTable:
    """Parse the Central Bank's XML into a table.

    Takes bytes rather than text because the prologue declares windows-1251 and
    :mod:`xml.etree` honours it; handing it a string that ``httpx`` decoded by
    guessing is how «Австралийский доллар» becomes mojibake.

    Raises:
        InstantProviderError: Not XML, or XML without a single readable rate.
    """
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError as exc:
        raise InstantProviderError(
            f"cbr response is not XML: {exc}",
            user_message="Сервис курсов ответил непонятно.",
        ) from exc
    rates: dict[str, Rate] = {}
    for node in root.iter("Valute"):
        code = (node.findtext("CharCode") or "").strip().upper()
        value = _decimal(node.findtext("Value"))
        nominal = _int(node.findtext("Nominal"))
        if not code or value is None:
            continue
        rates[code] = Rate(
            code=code,
            nominal=nominal or 1,
            value=value,
            name_ru=(node.findtext("Name") or "").strip(),
        )
    if not rates:
        raise InstantProviderError(
            "cbr response has no rates",
            user_message="Сервис курсов ответил непонятно.",
        )
    return RateTable(date=(root.get("Date") or "").strip(), rates=rates)


def _decimal(raw: str | None) -> Decimal | None:
    """A ``Decimal`` from the Central Bank's comma-decimal notation, or ``None``."""
    if not raw:
        return None
    try:
        return Decimal(raw.strip().replace(",", ".").replace(" ", ""))
    except InvalidOperation:
        return None


def _int(raw: str | None) -> int | None:
    """An int from a text node, or ``None``."""
    try:
        return int((raw or "").strip())
    except ValueError:
        return None


def roubles_ru(amount: Decimal) -> str:
    """Roubles as they are said: «91 рубль 50 копеек», «7 рублей 12 копеек».

    Kopecks are dropped when they are zero and kept otherwise, because a rate is
    quoted to the kopeck and rounding it to the rouble loses the part that moved.
    """
    cents = int((amount * 100).to_integral_value())
    whole, fraction = divmod(abs(cents), 100)
    sign = "минус " if cents < 0 else ""
    head = f"{whole} {plural_form(whole, 'рубль', 'рубля', 'рублей')}"
    if not fraction:
        return f"{sign}{head}"
    tail = f"{fraction} {plural_form(fraction, 'копейка', 'копейки', 'копеек')}"
    return f"{sign}{head} {tail}"


def _currency_ru(code: str, count: int = 1) -> str:
    """A currency in the form a sentence needs, or the code when unknown."""
    forms = _CODE_NAMES_RU.get(code.upper())
    return plural_form(count, *forms) if forms else code.upper()


def rate_ru(table: RateTable, code: str, into: str = _BASE) -> str:
    """One sentence about a rate, for the synthesiser.

    Against the rouble it reads as money — «Доллар — 91 рубль 50 копеек» — and
    against another currency as a plain number, because «евро стоит 1 доллар 8
    копеек» would be nonsense.
    """
    name = _currency_ru(code).capitalize()
    if into.upper() == _BASE:
        return f"{name} — {roubles_ru(table.per_unit(code))}. Курс Центробанка на {table.date}."
    ratio = table.cross(code, into)
    quantised = ratio.quantize(Decimal("0.01"))
    spoken = f"{quantised.normalize():f}".replace(".", ",")
    return f"{name} — {spoken} {_currency_ru(into, 2)}. По курсу Центробанка на {table.date}."


def split_pair(query: str) -> tuple[str, str]:
    """Read «доллар к евро» or «доллара» into a pair of ISO codes.

    Returns the rouble as the second half when the query names one currency,
    which is the ordinary «курс доллара». An unrecognised word yields ``""`` and
    the caller reports it, rather than a guess that quotes the wrong money.
    """
    words = [word.strip(".,!?").casefold() for word in query.split()]
    codes = [CURRENCY_ALIASES[word] for word in words if word in CURRENCY_ALIASES]
    unique: list[str] = []
    for code in codes:
        if code not in unique:
            unique.append(code)
    if not unique:
        return "", _BASE
    if len(unique) == 1:
        return unique[0], _BASE
    return unique[0], unique[1]


class CurrencyProvider(InstantProvider):
    """«курс доллара», «сколько стоит евро», «доллар к юаню»."""

    kind: ClassVar = "rates"
    title_ru: ClassVar = "курс валют"
    triggers: ClassVar = ("курс", "валюта", "валюты", "доллар", "евро", "юань", "биржа")
    #: The currency names are the subject, so the routing must not eat them:
    #: «сколько стоит доллар» routes here *on* «доллар» and is answered *by* it.
    keeps_triggers: ClassVar = True

    def fetch(self, query: str) -> InstantAnswer:
        """The rate for whatever currency ``query`` names.

        Raises:
            InstantNotFound: No currency in the query, or one the bank omits.
            InstantOffline: The Central Bank could not be reached.
            InstantProviderError: A response that cannot be parsed.
        """
        code, into = split_pair(query)
        if not code:
            raise InstantNotFound(
                f"no currency recognised in {query!r}",
                user_message="Не поняла, курс какой валюты нужен.",
            )
        response = self.fetcher.get(CBR_URL)
        table = parse_cbr(response.content)
        rate = table.get(code) if code != _BASE else None
        return InstantAnswer(
            kind=self.kind,
            message_ru=rate_ru(table, code, into),
            data={
                "date": table.date,
                "code": code,
                "into": into,
                "per_unit": str(table.per_unit(code)),
                "cross": str(table.cross(code, into)),
                "nominal": rate.nominal if rate is not None else 1,
                "name_ru": rate.name_ru if rate is not None else "Российский рубль",
            },
            source="cbr.ru",
        )

    def cache_key(self, query: str) -> str:
        """Key on the pair rather than the wording.

        «курс доллара» and «сколько стоит доллар» are the same request, and one
        publication a day is the same answer either way — a key built from the
        phrase would fetch the whole table twice for the same number.
        """
        code, into = split_pair(query)
        return f"{self.kind}:{code or '?'}:{into}"
