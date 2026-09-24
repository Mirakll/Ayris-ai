"""Counting what a request cost, in tokens and in money.

The «ИИ» tab shows two numbers after every answer: how many tokens the request
spent and roughly what that was worth. This module produces both. Tokens come
from the provider when it reports them and from a heuristic when it does not;
cost comes from a local price table, so no figure here ever leaves the machine
and none of it depends on a billing API.

**The cost is an estimate and says so.** Prices change, a model family covers
many exact names, and the rouble-billed Russian providers have no public
per-token dollar price at all — so :class:`LlmUsageReported` carries an
``estimated`` flag and the numbers are for orientation, not accounting. A missing
price is zero, never a crash.

**Tokens are approximate when the provider is silent.** :func:`estimate_tokens`
weights Cyrillic higher than Latin because byte-pair encoders split it into more
pieces; it is close enough to size a prompt and to fill in a completion count the
stream never delivered, and it is clearly labelled as a guess when used.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil
from types import MappingProxyType
from typing import Final

from ayris.nlu.llm.base import LlmMessage, LlmUsage

__all__ = [
    "ModelPrice",
    "UsageMeter",
    "UsageRecord",
    "estimate_prompt_tokens",
    "estimate_tokens",
    "price_for",
]

#: Chars per token for Latin and most scripts under a typical BPE vocabulary.
_LATIN_CHARS_PER_TOKEN: Final = 4.0

#: Chars per token for Cyrillic: BPE breaks it into more pieces, so a Russian
#: sentence costs more tokens than a Latin one of the same length.
_CYRILLIC_CHARS_PER_TOKEN: Final = 2.5

#: Rough per-message framing overhead (role markers, separators) the wire adds.
_MESSAGE_OVERHEAD_TOKENS: Final = 4


def _is_cyrillic(char: str) -> bool:
    lowered = char.lower()
    return "а" <= lowered <= "я" or lowered == "ё"


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text`` without a tokenizer.

    Deliberately cheap and offline: Cyrillic characters are weighted heavier than
    Latin ones because encoders split them more finely. The result is a guess,
    used to size a prompt and to stand in for a completion count a stream did not
    report — never presented as exact.
    """
    if not text:
        return 0
    cyrillic = sum(1 for char in text if _is_cyrillic(char))
    other = len(text) - cyrillic
    tokens = cyrillic / _CYRILLIC_CHARS_PER_TOKEN + other / _LATIN_CHARS_PER_TOKEN
    return max(1, ceil(tokens))


def estimate_prompt_tokens(messages: Sequence[LlmMessage]) -> int:
    """Approximate how many tokens a whole conversation will cost to send."""
    return sum(estimate_tokens(message.content) + _MESSAGE_OVERHEAD_TOKENS for message in messages)


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """What one model charges, in US dollars per million tokens.

    Per *million* rather than per thousand because that is how every provider
    quotes it now, and because a per-request cost then reads as a small decimal
    the settings tab can show to four places without rounding to zero.
    """

    prompt_per_1m: float = 0.0
    completion_per_1m: float = 0.0

    def cost(self, usage: LlmUsage) -> float:
        """Dollar cost of one request at this price. Zero when the price is unknown."""
        return (
            usage.prompt_tokens * self.prompt_per_1m
            + usage.completion_tokens * self.completion_per_1m
        ) / 1_000_000.0

    @property
    def known(self) -> bool:
        """Whether this is a real price rather than the zero fallback."""
        return self.prompt_per_1m > 0.0 or self.completion_per_1m > 0.0


#: Fallback for a provider we bill in dollars but whose exact model is unlisted.
_UNKNOWN_PRICE: Final = ModelPrice()

#: Prices in USD per million tokens, best-effort and approximate (see the module
#: docstring). Keyed by provider, then by a *substring* of the model name so one
#: entry covers a family: "gpt-4o-mini" matches "gpt-4o-mini-2024-07-18" too. The
#: empty-string key is the provider's default when nothing else matches. The
#: rouble-billed providers (GigaChat, YandexGPT) and the local runtimes have no
#: dollar price, so they resolve to zero and the estimate reads as such.
_PRICES: Final[MappingProxyType[str, MappingProxyType[str, ModelPrice]]] = MappingProxyType(
    {
        "openai": MappingProxyType(
            {
                "gpt-4o-mini": ModelPrice(0.15, 0.60),
                "gpt-4o": ModelPrice(2.50, 10.00),
                "gpt-4.1-mini": ModelPrice(0.40, 1.60),
                "gpt-4.1": ModelPrice(2.00, 8.00),
                "o1-mini": ModelPrice(1.10, 4.40),
                "o1": ModelPrice(15.00, 60.00),
                "": ModelPrice(0.15, 0.60),
            }
        ),
        "anthropic": MappingProxyType(
            {
                "claude-3-5-haiku": ModelPrice(0.80, 4.00),
                "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
                "claude-3-opus": ModelPrice(15.00, 75.00),
                "claude-3-haiku": ModelPrice(0.25, 1.25),
                "": ModelPrice(3.00, 15.00),
            }
        ),
        "deepseek": MappingProxyType(
            {
                "deepseek-reasoner": ModelPrice(0.55, 2.19),
                "deepseek-chat": ModelPrice(0.27, 1.10),
                "": ModelPrice(0.27, 1.10),
            }
        ),
        "openrouter": MappingProxyType(
            {
                "gpt-4o-mini": ModelPrice(0.15, 0.60),
                "claude-3-5-sonnet": ModelPrice(3.00, 15.00),
                "": _UNKNOWN_PRICE,
            }
        ),
    }
)


def price_for(provider: str, model: str) -> ModelPrice:
    """The price to use for ``provider``/``model``, or a zero fallback.

    Matches the exact model first, then any family whose name is a substring of
    it, then the provider default, then zero. Unknown providers and the local
    runtimes fall straight through to zero, which is the truthful answer: they
    cost nothing to the wallet.
    """
    table = _PRICES.get(provider.lower())
    if table is None:
        return _UNKNOWN_PRICE
    exact = table.get(model)
    if exact is not None:
        return exact
    for family, price in table.items():
        if family and family in model:
            return price
    return table.get("", _UNKNOWN_PRICE)


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """One request's tokens and cost, plus the running session total.

    ``estimated`` is true when the token counts are a heuristic rather than the
    provider's own figures; the settings tab shows the cost with a «≈» in front
    when it is. :meth:`as_payload` is the dict the worker emits, which the bus
    translator turns into :class:`~ayris.core.events.LlmUsageReported`.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    session_cost_usd: float
    estimated: bool = False
    request_id: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_payload(self) -> dict[str, object]:
        """The event payload, matching :class:`~ayris.core.events.LlmUsageReported`."""
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "session_cost_usd": round(self.session_cost_usd, 6),
            "estimated": self.estimated,
            "request_id": self.request_id,
        }


class UsageMeter:
    """Accumulates token and cost totals across one session.

    Lives in the LLM worker, one per process, so «за сессию» means "since this
    worker started" — which is what the user sees as one run of Ayris. Not
    thread-safe: the worker dispatches one request at a time.
    """

    __slots__ = ("_completion_tokens", "_cost_usd", "_prompt_tokens", "_requests")

    def __init__(self) -> None:
        self._requests = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._cost_usd = 0.0

    def record(
        self,
        provider: str,
        model: str,
        usage: LlmUsage,
        *,
        estimated: bool = False,
        request_id: str = "",
    ) -> UsageRecord:
        """Fold one request into the totals and return its record."""
        cost = price_for(provider, model).cost(usage)
        self._requests += 1
        self._prompt_tokens += usage.prompt_tokens
        self._completion_tokens += usage.completion_tokens
        self._cost_usd += cost
        return UsageRecord(
            provider=provider,
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cost_usd=cost,
            session_cost_usd=self._cost_usd,
            estimated=estimated,
            request_id=request_id,
        )

    @property
    def requests(self) -> int:
        """How many requests have been folded in this session."""
        return self._requests

    @property
    def session_cost_usd(self) -> float:
        """Best-effort dollar total for the session so far."""
        return self._cost_usd

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens across the session."""
        return self._prompt_tokens + self._completion_tokens

    def snapshot(self) -> dict[str, object]:
        """Session totals for :meth:`LlmWorker.status`."""
        return {
            "requests": self._requests,
            "prompt_tokens": self._prompt_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self._cost_usd, 6),
        }


def resolve_usage(
    provider: str,
    reported: LlmUsage,
    messages: Iterable[LlmMessage],
    completion_text: str,
) -> tuple[LlmUsage, bool]:
    """Return the usage to record and whether it was estimated.

    A provider that reported both counts is trusted as authoritative. When either
    is missing — some providers omit usage entirely, and a stream cut short never
    sends it — the gap is filled from :func:`estimate_tokens`, and the second
    element of the tuple is ``True`` so the caller can flag the cost as a guess.
    """
    del provider  # kept for a future per-provider correction factor.
    prompt = reported.prompt_tokens
    completion = reported.completion_tokens
    estimated = False
    if prompt <= 0:
        prompt = estimate_prompt_tokens(tuple(messages))
        estimated = True
    if completion <= 0:
        completion = estimate_tokens(completion_text)
        estimated = estimated or completion > 0
    return LlmUsage(prompt_tokens=prompt, completion_tokens=completion), estimated
