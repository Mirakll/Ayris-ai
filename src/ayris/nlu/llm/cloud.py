"""What the cloud LLM providers have in common.

Every online model speaks HTTP and streams its answer back token by token, and
they differ only in three places: the shape of the request, the name of the auth
header, and how a chunk of the stream is framed. They also fail in the same few
ways — the key is wrong, the quota is spent, the context is too long, the service
is down — and the rest of Ayris has to tell those apart, because only some are
worth retrying and only some the user can fix. This module holds all of that, so
:mod:`~ayris.nlu.llm.openai_client` and its siblings are a handful of short
methods each.

**A key never reaches a log, an event or an error message.**
:func:`_scrub_headers` masks any header whose name looks like a secret before the
request line is written, and the error text carries the status code and the
provider's own explanation, never the key.

**Retries are only ever before the first token.** A 429/5xx/network failure while
*opening* the stream is retried with capped, jittered backoff. Once a single
delta has been yielded, a broken connection is a terminal :class:`LlmError`
instead — replaying a stream would speak the same sentence twice, which is worse
than a clean error the pipeline can report.

**Cancellation closes the socket.** The predicate is polled between chunks; when
it flips, the ``with`` around the stream exits, httpx closes the connection, and
the terminal delta carries :attr:`FinishReason.CANCELLED` so TTS stops too.

**httpx is imported at module level.** A cloud client is only constructed after
the user picked an online provider, so the import cost is paid by someone who has
already decided to talk to the network.
"""

from __future__ import annotations

import json
import secrets as random_secrets
import ssl
from abc import abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, ClassVar, Final, NoReturn

import httpx

from ayris.core.errors import LlmAuthError, LlmContextOverflowError, LlmError, LlmQuotaError
from ayris.core.models import JsonObject
from ayris.nlu.llm.base import (
    CredentialCheck,
    FinishReason,
    LlmClient,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmResponse,
    LlmTextDelta,
    LlmTool,
    LlmToolCall,
    LlmToolCallDelta,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.utils.logger import get_logger

__all__ = [
    "BACKOFF_BASE_SEC",
    "BACKOFF_MAX_SEC",
    "CONNECT_TIMEOUT_SEC",
    "MAX_RETRIES",
    "READ_TIMEOUT_SEC",
    "CloudLlmClient",
    "CloudOptions",
    "OpenAiCompatibleClient",
]

_log = get_logger(__name__)

#: Connect timeout: DNS + TCP + TLS. Separate so a stalled resolver fails fast
#: instead of waiting out the read timeout.
CONNECT_TIMEOUT_SEC: Final = 5.0

#: Read timeout: the longest gap *between* chunks of a stream, not the whole
#: answer. httpx resets it on every received byte, so a slow-but-steady stream
#: never trips it; a provider that went quiet for this long has stalled.
READ_TIMEOUT_SEC: Final = 60.0

#: Write timeout: sending the request body. The bodies here are small.
WRITE_TIMEOUT_SEC: Final = 10.0

#: Retry attempts after a transient failure while opening the stream. Never
#: applied once tokens have started arriving.
MAX_RETRIES: Final = 2

#: Base delay for exponential backoff, in seconds.
BACKOFF_BASE_SEC: Final = 0.5

#: Maximum backoff delay, in seconds.
BACKOFF_MAX_SEC: Final = 8.0

#: Cap on an honoured ``Retry-After`` header: past this, waiting is worse than
#: telling the user the quota is spent.
_MAX_RETRY_AFTER_SEC: Final = 30.0

#: Sent with every request. Version-free on purpose: a provider that rate-limits
#: by client string must not see a new client on every Ayris update.
_USER_AGENT: Final = "Ayris"

#: Header names whose value is a secret. Matched as lowercased substrings, so a
#: bare "key" also covers ``x-api-key`` and ``api-key``.
_SECRET_HEADERS: Final = ("authorization", "key", "token", "secret", "auth")

#: How much of an error body goes into the technical message.
_ERROR_SNIPPET: Final = 400

#: Substrings that mark a 4xx as "the prompt was too long" rather than a generic
#: bad request. Lowercased; matched against the error body. Best-effort: the
#: providers phrase it differently and none of them use a dedicated status.
_CONTEXT_MARKERS: Final = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
    "too many tokens",
    "maximum number of tokens",
    "reduce the length",
    "prompt is too long",
    "input is too long",
)


def _never() -> bool:
    """The default ``cancel`` predicate: nothing ever cancels."""
    return False


def _scrub_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Replace every auth header value with a placeholder before it is logged."""
    return {
        key: ("***" if any(marker in key.lower() for marker in _SECRET_HEADERS) else value)
        for key, value in headers.items()
    }


def _error_detail(data: bytes) -> str:
    """The provider's own explanation of a failure, short and key-free."""
    if not data:
        return ""
    text = data[:_ERROR_SNIPPET].decode("utf-8", errors="replace").strip()
    return f": {text}" if text else ""


def _looks_like_overflow(data: bytes) -> bool:
    """Whether an error body reads like a context-window overflow."""
    if not data:
        return False
    text = data[:_ERROR_SNIPPET].decode("utf-8", errors="replace").lower()
    return any(marker in text for marker in _CONTEXT_MARKERS)


def _parse_retry_after(value: str | None) -> float | None:
    """Read a ``Retry-After`` header expressed in whole seconds.

    The HTTP-date form is ignored on purpose: parsing it against the local clock
    is a source of surprises, and the backoff schedule is a fine fallback.
    """
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds > 0.0 else None


def _loads_object(payload: str) -> JsonObject | None:
    """Parse one JSON object out of a stream line, or ``None`` if it is not one."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _decode_tool_call(fragment: Mapping[str, str]) -> LlmToolCall:
    """Turn the concatenated fragments of one tool call into a decoded call."""
    raw = fragment.get("args", "")
    arguments: JsonObject = {}
    if raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            arguments = parsed
    return LlmToolCall(
        name=fragment.get("name", ""),
        arguments=arguments,
        call_id=fragment.get("id", ""),
    )


class _Retryable(Exception):  # noqa: N818  # приватный маркер повтора, не «...Error»
    """Internal marker: a failure that may be retried *before the first token*.

    Carries enough to both schedule the next attempt and, once the budget is
    spent, become the right typed :class:`LlmError`. Never leaves this module.
    """

    def __init__(
        self,
        *,
        status: int | None,
        detail: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(detail or (f"HTTP {status}" if status else "network error"))
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class CloudOptions:
    """Everything a cloud client needs, resolved once before it is built.

    ``api_key`` is the plain secret, already read from the credential store by the
    worker; it never round-trips through the config file. ``transport`` is the
    seam the tests use to answer with :class:`httpx.MockTransport` instead of the
    network — production leaves it ``None``.
    """

    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float | None = None
    max_tokens: int | None = None
    connect_timeout_sec: float = CONNECT_TIMEOUT_SEC
    read_timeout_sec: float = READ_TIMEOUT_SEC
    max_retries: int = MAX_RETRIES
    verify: bool = True
    transport: httpx.BaseTransport | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


class CloudLlmClient(LlmClient):
    """Base for cloud LLM providers: HTTP client, auth, retries, error mapping.

    Subclasses fill in four hooks — :meth:`_endpoint`, :meth:`_auth_headers`,
    :meth:`_build_payload` and :meth:`_read_stream` — and get streaming,
    cancellation, retry/backoff and typed errors for free. :meth:`complete`
    drains :meth:`stream`, so a provider writes the streaming path once and the
    non-streaming shape falls out of it.
    """

    #: These clients stream natively; the whole module exists for that.
    supports_streaming: ClassVar[bool] = True

    #: Human name of the service, for log lines and the sentence the user reads.
    title: ClassVar[str] = ""

    #: Endpoint host used when :attr:`CloudOptions.base_url` is empty.
    default_base_url: ClassVar[str] = ""

    #: Model used when :attr:`CloudOptions.model` is empty.
    default_model: ClassVar[str] = ""

    def __init__(self, options: CloudOptions) -> None:
        self._options = options
        self._model = options.model or self.default_model
        self._api_key = options.api_key
        self._base_url = (options.base_url or self.default_base_url).rstrip("/")
        self._client: httpx.Client | None = None

    @property
    def model(self) -> str:
        """The model this client will ask for."""
        return self._model

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    @property
    def _provider_name(self) -> str:
        return self.title or self.name

    def _build_client(self) -> httpx.Client:
        """Open the HTTP client. The transport seam is what the tests inject."""
        options = self._options
        # A mock transport never touches TLS, so skip resolving a trust anchor for
        # it — the tests get the plain flag and no PEM is read off disk.
        verify: bool | ssl.SSLContext = (
            options.verify if options.transport is not None else self._resolve_verify()
        )
        return httpx.Client(
            timeout=httpx.Timeout(
                connect=options.connect_timeout_sec,
                read=options.read_timeout_sec,
                write=WRITE_TIMEOUT_SEC,
                pool=options.connect_timeout_sec,
            ),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
            transport=options.transport,
            verify=verify,
        )

    def _extra_ca_path(self) -> Path | None:
        """A provider-specific CA bundle to trust *in addition* to the system roots.

        The default is ``None`` — a branded provider verifies against certifi like
        any other HTTPS client. A provider whose chain is signed by a CA that ships
        in no default trust store (GigaChat, on the Russian National CA) overrides
        this to point at the PEM bundled under ``resources/certs``; see
        :meth:`_resolve_verify` for how it is turned into an SSL context.
        """
        return None

    def _resolve_verify(self) -> bool | ssl.SSLContext:
        """Turn :attr:`CloudOptions.verify` + :meth:`_extra_ca_path` into a verify arg.

        ``verify=False`` is honoured verbatim (the escape hatch stays an escape
        hatch). Otherwise, when the provider names an extra CA bundle that exists,
        the PEM is read *in Python* and fed to the context as ``cadata`` — never as
        a file path — so OpenSSL is not asked to open a name under the Cyrillic
        install directory it cannot handle. The context still loads the system
        roots first, so the bundle is purely additive: every ordinary site keeps
        verifying, and GigaChat's chain now validates too. A missing or unreadable
        bundle degrades to plain ``True`` with a warning rather than silently
        disabling TLS.
        """
        options = self._options
        if options.verify is False:
            return False
        ca_path = self._extra_ca_path()
        if ca_path is None:
            return True
        try:
            pem = ca_path.read_text(encoding="utf-8")
        except OSError as exc:
            _log.warning(
                "%s: не удалось прочитать корневой сертификат %s (%s); "
                "проверка TLS по системным корням",
                self._provider_name,
                ca_path,
                exc,
            )
            return True
        context = ssl.create_default_context()
        try:
            context.load_verify_locations(cadata=pem)
        except ssl.SSLError as exc:
            _log.warning(
                "%s: сертификат %s не разобран (%s); проверка TLS по системным корням",
                self._provider_name,
                ca_path,
                exc,
            )
            return True
        return context

    def _require_client(self) -> httpx.Client:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def close(self) -> None:
        """Close the HTTP client. Safe to call twice, and never raises."""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - httpx does not raise here
                _log.debug("%s: closing the client failed: %s", self._provider_name, exc)

    # ------------------------------------------------------------------
    # the streaming contract
    # ------------------------------------------------------------------

    def _pick_temperature(self, override: float | None) -> float | None:
        return override if override is not None else self._options.temperature

    def _pick_max_tokens(self, override: int | None) -> int | None:
        return override if override is not None else self._options.max_tokens

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        headers.update(self._auth_headers())
        return headers

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        predicate = cancel or _never
        payload = self._build_payload(
            messages,
            tuple(tools),
            temperature=self._pick_temperature(temperature),
            max_tokens=self._pick_max_tokens(max_tokens),
            stream=True,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._request_headers()
        url = self._endpoint()
        # Exactly one terminal delta reaches the caller: a provider Done, a
        # cancel-synthesised one, or a fallback if the body ended abruptly.
        for delta in self._stream_with_retries(url, headers, body, predicate):
            if predicate():
                yield LlmDoneDelta(finish_reason=FinishReason.CANCELLED)
                return
            if isinstance(delta, LlmDoneDelta):
                yield delta
                return
            yield delta
        yield LlmDoneDelta(
            finish_reason=FinishReason.CANCELLED if predicate() else FinishReason.STOP
        )

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        started = perf_counter()
        text_parts: list[str] = []
        fragments: dict[int, dict[str, str]] = {}
        usage = LlmUsage()
        finish = FinishReason.STOP
        for delta in self.stream(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            cancel=cancel,
        ):
            if isinstance(delta, LlmTextDelta):
                text_parts.append(delta.text)
            elif isinstance(delta, LlmToolCallDelta):
                fragment = fragments.setdefault(delta.index, {"id": "", "name": "", "args": ""})
                if delta.call_id:
                    fragment["id"] = delta.call_id
                if delta.name:
                    fragment["name"] = delta.name
                fragment["args"] += delta.arguments
            elif isinstance(delta, LlmUsageDelta):
                usage = delta.usage
            else:
                finish = delta.finish_reason
        tool_calls = tuple(
            _decode_tool_call(fragment)
            for _, fragment in sorted(fragments.items())
            if fragment["name"]
        )
        return LlmResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            model=self._model,
            engine=self.name,
            finish_reason=finish,
            usage=usage,
            duration_ms=int((perf_counter() - started) * 1000.0),
        )

    # ------------------------------------------------------------------
    # HTTP: opening the stream, retries, error mapping
    # ------------------------------------------------------------------

    def _stream_with_retries(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        attempt = 0
        while True:
            attempt += 1
            if cancel():
                yield LlmDoneDelta(finish_reason=FinishReason.CANCELLED)
                return
            try:
                yield from self._open_and_read(url, headers, body, cancel)
                return
            except _Retryable as exc:
                delay = self._backoff_delay(attempt, exc.retry_after)
                if attempt > self._options.max_retries:
                    raise self._retryable_to_error(exc) from exc
                _log.warning(
                    "%s: %s on attempt %d, retrying in %.1fs",
                    self._provider_name,
                    exc,
                    attempt,
                    delay,
                )
                sleep(delay)

    def _open_and_read(
        self,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        client = self._require_client()
        _log.debug("%s: POST %s headers=%s", self.name, url, _scrub_headers(headers))
        emitted = False
        try:
            with client.stream("POST", url, headers=dict(headers), content=body) as response:
                if not (200 <= response.status_code < 300):
                    self._raise_for_status(response)
                for delta in self._read_stream(response, cancel):
                    emitted = True
                    yield delta
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if emitted:
                # Retrying now would replay tokens the user has already heard.
                raise LlmError(
                    f"{self.name}: stream broke mid-answer: {exc}",
                    user_message=(
                        f"Соединение с «{self._provider_name}» прервалось во время ответа."
                    ),
                ) from exc
            raise _Retryable(status=None, detail=f"{type(exc).__name__}: {exc}") from exc

    def _raise_for_status(self, response: httpx.Response) -> NoReturn:
        """Map a non-2xx onto a retry marker or the typed error the caller acts on."""
        status = response.status_code
        data = response.read()
        detail = _error_detail(data)
        if status == 429:
            raise _Retryable(
                status=429,
                detail=f"quota/rate limited{detail}",
                retry_after=_parse_retry_after(response.headers.get("retry-after")),
            )
        if status >= 500:
            raise _Retryable(status=status, detail=f"server error {status}{detail}")
        if status in (401, 403):
            raise LlmAuthError(f"{self.name}: authentication rejected ({status}){detail}")
        if _looks_like_overflow(data):
            raise LlmContextOverflowError(f"{self.name}: context overflow ({status}){detail}")
        raise LlmError(
            f"{self.name}: HTTP {status}{detail}",
            user_message=f"«{self._provider_name}» отклонил запрос (ошибка {status}).",
        )

    def _retryable_to_error(self, exc: _Retryable) -> LlmError:
        """The typed error a retry marker becomes once the budget is spent."""
        if exc.status == 429:
            return LlmQuotaError(f"{self.name}: quota exhausted after retries: {exc.detail}")
        return LlmError(
            f"{self.name}: unavailable after retries: {exc.detail}",
            user_message=(
                f"«{self._provider_name}» сейчас недоступен. Проверьте подключение к интернету."
            ),
        )

    def _backoff_delay(self, attempt: int, retry_after: float | None = None) -> float:
        """Delay before a retry: an honoured ``Retry-After``, else jittered backoff."""
        if retry_after is not None:
            return min(retry_after, _MAX_RETRY_AFTER_SEC)
        capped: float = min(BACKOFF_BASE_SEC * float(2 ** (attempt - 1)), BACKOFF_MAX_SEC)
        jitter: float = random_secrets.randbelow(1000) / 1000.0
        return capped * 0.5 + capped * 0.5 * jitter

    # ------------------------------------------------------------------
    # catalogue and key check — only ever on explicit user action
    # ------------------------------------------------------------------

    def list_models(self) -> tuple[str, ...]:
        endpoint = self._models_endpoint()
        if not endpoint:
            return ()
        client = self._require_client()
        try:
            response = client.get(endpoint, headers=dict(self._request_headers()))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LlmError(
                f"{self.name}: listing models failed: {exc}",
                user_message=f"Не удалось получить список моделей «{self._provider_name}».",
            ) from exc
        if not (200 <= response.status_code < 300):
            try:
                self._raise_for_status(response)
            except _Retryable as exc:
                raise self._retryable_to_error(exc) from exc
        return self._parse_models(response.read())

    def check_credentials(self) -> CredentialCheck:
        if not self.configured:
            return CredentialCheck(ok=False, detail="Ключ доступа не задан.")
        try:
            models = self.list_models()
        except LlmError as exc:
            return CredentialCheck(ok=False, detail=exc.user_message)
        return CredentialCheck(ok=True, models=models)

    def _models_endpoint(self) -> str:
        """The GET endpoint that lists models, or empty when there is none."""
        return ""

    def _parse_models(self, data: bytes) -> tuple[str, ...]:
        """Read model ids out of an OpenAI-shaped ``{"data": [{"id": ...}]}``."""
        payload = _loads_object(data.decode("utf-8", errors="replace"))
        if payload is None:
            return ()
        items = payload.get("data")
        if not isinstance(items, list):
            return ()
        names = [
            item["id"]
            for item in items
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        return tuple(names)

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    @abstractmethod
    def _endpoint(self) -> str:
        """The full URL the streaming request is POSTed to."""

    @abstractmethod
    def _auth_headers(self) -> Mapping[str, str]:
        """The provider's auth header(s). Values are masked before any log line."""

    @abstractmethod
    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> JsonObject:
        """Build the request body for this provider's chat endpoint."""

    @abstractmethod
    def _read_stream(
        self,
        response: httpx.Response,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        """Turn the streamed body into deltas.

        Polls ``cancel`` between chunks and returns when it flips — the caller's
        ``with`` then closes the connection. May yield a terminal
        :class:`LlmDoneDelta` when the provider signals the end; if it returns
        without one, the base synthesises the terminal delta.

        Never raises :class:`_Retryable`: by the time this runs the stream is
        open and any failure is terminal, because a retry would duplicate tokens.
        """


class OpenAiCompatibleClient(CloudLlmClient):
    """Providers that speak the OpenAI ``/chat/completions`` SSE schema.

    OpenAI, OpenRouter, DeepSeek and GigaChat all frame a stream the same way —
    ``data: {json}`` lines carrying ``choices[0].delta`` fragments, a ``usage``
    block when asked, and a ``data: [DONE]`` terminator — so they differ only in
    base URL, auth header and default model, which are three class attributes.
    """

    supports_tools: ClassVar[bool] = True

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _models_endpoint(self) -> str:
        return f"{self._base_url}/models"

    def _auth_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> JsonObject:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.as_payload() for message in messages],
            "stream": stream,
        }
        if stream:
            # Ask for the final usage block; without it the count is a guess.
            payload["stream_options"] = {"include_usage": True}
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = [tool.as_payload() for tool in tools]
        return payload

    def _read_stream(
        self,
        response: httpx.Response,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        finish_reason: str | None = None
        for line in response.iter_lines():
            if cancel():
                return
            stripped = line.strip()
            if not stripped or not stripped.startswith("data:"):
                continue
            payload = stripped[len("data:") :].strip()
            if payload == "[DONE]":
                yield LlmDoneDelta(finish_reason=_map_openai_finish(finish_reason))
                return
            chunk = _loads_object(payload)
            if chunk is None:
                continue
            usage = _openai_usage(chunk.get("usage"))
            if usage is not None:
                yield usage
            choices = chunk.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            reason = choice.get("finish_reason")
            if isinstance(reason, str):
                finish_reason = reason
            yield from _openai_choice_deltas(choice)
        # No [DONE] arrived; the base will synthesise the terminal delta.


def _map_openai_finish(reason: str | None) -> FinishReason:
    """Map OpenAI's ``finish_reason`` string onto :class:`FinishReason`."""
    if reason == "length":
        return FinishReason.LENGTH
    if reason == "tool_calls":
        return FinishReason.TOOL_CALLS
    if reason == "content_filter":
        return FinishReason.ERROR
    return FinishReason.STOP


def _openai_usage(raw: object) -> LlmUsageDelta | None:
    """Read an OpenAI ``usage`` block into a delta, or ``None`` if absent."""
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    return LlmUsageDelta(
        usage=LlmUsage(
            prompt_tokens=prompt if isinstance(prompt, int) else 0,
            completion_tokens=completion if isinstance(completion, int) else 0,
        )
    )


def _openai_choice_deltas(choice: Mapping[str, Any]) -> Iterator[LlmDelta]:
    """Yield the text and tool-call fragments carried by one streamed choice."""
    delta = choice.get("delta")
    if not isinstance(delta, dict):
        return
    content = delta.get("content")
    if isinstance(content, str) and content:
        yield LlmTextDelta(text=content)
    tool_calls = delta.get("tool_calls")
    if not isinstance(tool_calls, list):
        return
    for entry in tool_calls:
        if not isinstance(entry, dict):
            continue
        index = entry.get("index")
        function = entry.get("function")
        function = function if isinstance(function, dict) else {}
        name = function.get("name")
        arguments = function.get("arguments")
        call_id = entry.get("id")
        yield LlmToolCallDelta(
            index=index if isinstance(index, int) else 0,
            call_id=call_id if isinstance(call_id, str) else "",
            name=name if isinstance(name, str) else "",
            arguments=arguments if isinstance(arguments, str) else "",
        )
