"""What the four cloud recognisers have in common.

Every online provider speaks HTTP and JSON, and differs only in the shape of the
request, the name of the auth header and where the text sits in the answer. They
also fail in the same three ways - the key is wrong, the quota is spent, the
service is down - and the rest of Ayris needs to tell those apart, because only
one of them is worth retrying and only one of them the user can fix. This module
holds all of that, so :mod:`~ayris.audio.stt.google_engine`,
:mod:`~ayris.audio.stt.yandex_engine`, :mod:`~ayris.audio.stt.azure_engine` and
:mod:`~ayris.audio.stt.openai_engine` are three short methods each.

**A key never reaches a log.** :func:`_scrub_headers` replaces the value of any
header whose name looks like an auth header before the request line is written
out, and the bodies are audio - too large to log and pointless to read. The
error messages carry the status code and the provider's own explanation, which
is what makes a failure diagnosable without the key ever appearing.

**Timeouts are separate and the deadline is absolute.** A stalled DNS lookup
fails after :data:`CONNECT_TIMEOUT_SEC` rather than waiting out the read
timeout, and the whole call - every retry included - is capped at
:data:`DEADLINE_SEC`, because past that the router should have fallen back to
the local model already. Retries back off exponentially with jitter; a 429 is
not retried at all, since the next attempt hits the same wall.

**httpx is imported at module level.** These classes are only ever constructed
after the user picks online mode, so the import cost is paid by someone who has
already decided to talk to the network.
"""

from __future__ import annotations

import importlib
import io
import json
import secrets as random_secrets
import wave
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter, sleep
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, NoReturn

import httpx

from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.audio.stt.base import STT_SAMPLE_RATE, SttEngine, TranscriptResult, TranscriptSegment
from ayris.core.errors import AyrisError, SttError
from ayris.core.models import JsonObject
from ayris.core.secrets import get_secrets, mask
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ayris.audio.stt.base import AudioBuffer, SttOptions

__all__ = [
    "ASSUMED_CONFIDENCE",
    "BACKOFF_BASE_SEC",
    "BACKOFF_MAX_SEC",
    "CLOUD_ENGINE_ENTRYPOINTS",
    "CONNECT_TIMEOUT_SEC",
    "DEADLINE_SEC",
    "MAX_RETRIES",
    "READ_TIMEOUT_SEC",
    "CloudSttEngine",
    "NetworkError",
    "QuotaError",
    "as_wav",
    "cloud_engine_names",
    "create_cloud_engine",
]

_log = get_logger(__name__)

#: Online providers, kept apart from
#: :data:`~ayris.audio.stt.base.ENGINE_ENTRYPOINTS` because the two lists answer
#: different questions: that one is "which model can run on this machine", this
#: one is "which service can this account reach". The settings window shows both,
#: and the router picks one from each.
CLOUD_ENGINE_ENTRYPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "yandex": "ayris.audio.stt.yandex_engine:YandexSttEngine",
        "google": "ayris.audio.stt.google_engine:GoogleSttEngine",
        "azure": "ayris.audio.stt.azure_engine:AzureSttEngine",
        "openai": "ayris.audio.stt.openai_engine:OpenAiSttEngine",
    }
)

#: Connect timeout: DNS + TCP handshake. Separate so a stalled resolver does not
#: wait through the whole read timeout.
CONNECT_TIMEOUT_SEC: Final = 3.0

#: Read timeout: first byte to last. Covers the provider's queue time, model
#: inference and response serialisation. Whisper API on a cold model can take 15s.
READ_TIMEOUT_SEC: Final = 20.0

#: Absolute deadline for one transcription attempt including all retries. Beyond
#: this the router gives up and falls back to offline, so no point retrying.
DEADLINE_SEC: Final = 25.0

#: Maximum retry attempts after transient errors (5xx, network). Does not apply
#: to 4xx: those are permanent.
MAX_RETRIES: Final = 2

#: Base delay for exponential backoff, in seconds.
BACKOFF_BASE_SEC: Final = 0.5

#: Maximum backoff delay, in seconds.
BACKOFF_MAX_SEC: Final = 4.0

#: Sent with every request. Version-free on purpose: a provider that rate-limits
#: by client string must not see a new client on every Ayris update.
_USER_AGENT: Final = "Ayris"

#: Confidence reported when a provider recognised something but said nothing
#: about how sure it is. Above the default ``min_confidence`` of 0.4, because
#: dropping a transcript the provider was happy with would be the wrong default.
ASSUMED_CONFIDENCE: Final = 0.9

#: Header names whose value is a secret. Matched as substrings, lowercased, so a
#: bare "key" also covers ``X-Goog-Api-Key`` and ``Ocp-Apim-Subscription-Key``.
_SECRET_HEADERS: Final = ("authorization", "key", "token", "secret", "auth")

#: How much of an error body goes into the technical message.
_ERROR_SNIPPET: Final = 300


def _as_positive(value: object, fallback: float) -> float:
    """Read a positive number out of :attr:`SttOptions.extra`, or fall back."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value) if value > 0.0 else fallback


def _as_retries(value: object) -> int:
    """Read the retry budget. Zero is legitimate: try once, then fall back."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MAX_RETRIES
    return max(0, int(value))


def _scrub_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Replace every auth header value with a placeholder.

    Matched case-insensitively and by substring: the four providers between them
    use ``Authorization``, ``Api-Key``, ``X-Goog-Api-Key`` and
    ``Ocp-Apim-Subscription-Key``, and a provider added later will most likely
    also have "key", "token" or "auth" in the name.
    """
    return {
        key: ("***" if any(marker in key.lower() for marker in _SECRET_HEADERS) else value)
        for key, value in headers.items()
    }


def _error_detail(data: bytes) -> str:
    """The provider's own explanation of a failure, for the technical message.

    Kept short and kept out of the user-facing sentence: the useful part of an
    error body is a code and a phrase, and the rest is a stack trace from a
    machine the user does not own.
    """
    if not data:
        return ""
    text = data[:_ERROR_SNIPPET].decode("utf-8", errors="replace").strip()
    return f": {text}" if text else ""


def as_wav(audio: AudioBuffer) -> bytes:
    """Wrap raw PCM in a WAV container.

    Three of the four providers accept a bare stream and would rather have the
    header anyway; OpenAI insists on a named file with a recognised extension.
    Built in memory: writing a temporary file for every phrase would put the
    user's speech on disk, which ``privacy.store_audio`` says not to do.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(STT_SAMPLE_RATE)
        handle.writeframes(audio.pcm)
    return buffer.getvalue()


@dataclass(frozen=True, slots=True)
class _Request:
    """One prepared HTTP request. Never logged as a whole: the body is audio."""

    url: str
    headers: Mapping[str, str]
    body: bytes


class NetworkError(SttError):
    """Transient network or server error: connection refused, timeout, 5xx."""

    pass


class QuotaError(SttError):
    """Quota exceeded (429). Not retried: the next attempt would hit the same cap."""

    pass


class CloudSttEngine(SttEngine):
    """Base for cloud STT providers: manages HTTP client, auth, retries.

    Subclasses implement :meth:`_provider_name`, :meth:`_load_credential`,
    :meth:`_build_request` and :meth:`_parse_response`. Everything else —
    timeout configuration, retry logic, error code mapping — is here.
    """

    #: Credential entry this provider looks for when the configuration does not
    #: name one. Equal to the :data:`~ayris.core.secrets.KNOWN_SLOTS` reference,
    #: so a user who filled in the provider's field in the settings window gets a
    #: working engine without also having to set ``credential_ref``.
    default_ref: ClassVar[str] = ""

    #: Endpoint used when ``voice.stt.online_endpoint`` is empty. Overridable so
    #: that a private deployment or a regional host can be pointed at without a
    #: code change.
    default_endpoint: ClassVar[str] = ""

    #: Human name of the service, for log lines and for the sentence the user
    #: reads when it fails.
    title: ClassVar[str] = ""

    #: Cloud engines are not offered in the offline engine list and have no
    #: vendor package to check for.
    package: ClassVar[str] = "httpx"
    module: ClassVar[str] = "httpx"

    __slots__ = ("_client", "_credential")

    def __init__(self) -> None:
        super().__init__()
        self._client: httpx.Client | None = None
        self._credential: str = ""

    @property
    def _provider_name(self) -> str:
        """Service name for messages. ``title`` with the engine name as a fallback."""
        return self.title or self.name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Empty: every provider here recognises far more languages than Ayris uses."""
        return ()

    @property
    def device(self) -> str:
        """Not this machine. Reported so the pipeline log says where the time went."""
        return "cloud"

    def load(self, model_path: Path, options: SttOptions) -> None:
        """Read the credential and open the HTTP client.

        Args:
            model_path: Ignored; a cloud engine has no model on disk. Part of the
                signature because the worker calls every engine the same way.
            options: Language, punctuation and the provider-specific extras.

        Raises:
            SttError: No credential, or the credential store refused the read.
        """
        del model_path
        self._credential = self._load_credential(options)
        self._client = self._build_client(options)
        self._options = options
        _log.info(
            "%s: cloud recogniser ready, credential %s",
            self._provider_name,
            mask(self._credential),
        )

    def _build_client(self, options: SttOptions) -> httpx.Client:
        """Open the client. Overridden in tests to inject a mock transport."""
        transport = options.extra.get("transport")
        return httpx.Client(
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SEC,
                read=_as_positive(options.extra.get("read_timeout_sec"), READ_TIMEOUT_SEC),
                write=CONNECT_TIMEOUT_SEC,
                pool=CONNECT_TIMEOUT_SEC,
            ),
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
            transport=transport if isinstance(transport, httpx.BaseTransport) else None,
        )

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise audio via the cloud provider.

        Raises:
            NetworkError: Connection failed, timeout, or 5xx response.
            QuotaError: Quota exceeded (429).
            SttError: Invalid credential (401/403), malformed response, or other
                permanent error.
        """
        options = self._require_loaded()
        prepared = self._prepare(audio)
        if prepared.is_silent() or prepared.duration_ms < options.min_speech_ms:
            return TranscriptResult.empty(
                engine=self.name,
                duration_ms=prepared.duration_ms,
                model="cloud",
            )

        started = perf_counter()
        deadline = _as_positive(options.extra.get("deadline_sec"), DEADLINE_SEC)
        retries = _as_retries(options.extra.get("max_retries"))
        attempt = 0
        last_error: Exception | None = None

        while True:
            attempt += 1
            try:
                result = self._attempt_transcribe(prepared)
            except (httpx.TimeoutException, httpx.TransportError, NetworkError) as exc:
                last_error = exc
                spent = perf_counter() - started
                delay = self._backoff_delay(attempt)
                if attempt > retries or (spent + delay) >= deadline:
                    break
                _log.warning(
                    "%s: %s on attempt %d, retrying in %.1fs",
                    self._provider_name,
                    type(exc).__name__,
                    attempt,
                    delay,
                )
                sleep(delay)
                continue
            return result.with_timing(
                inference_ms=(perf_counter() - started) * 1000.0,
                duration_ms=prepared.duration_ms,
            )

        if isinstance(last_error, NetworkError):
            # The service answered, with a 5xx. It already said what went wrong,
            # and blaming the user's connection instead would send them to the
            # wrong place.
            _log.warning(
                "%s: giving up after %d attempt(s): %s",
                self._provider_name,
                attempt,
                last_error.technical,
            )
            raise last_error
        raise NetworkError(
            f"{self._provider_name}: {type(last_error).__name__} after {attempt} "
            f"attempt(s): {last_error}",
            user_message=(f"{self._provider_name} не отвечает. Проверьте подключение к интернету."),
        ) from last_error

    def unload(self) -> None:
        """Close the HTTP client. Safe to call twice, and never raises."""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - httpx does not raise here
                _log.debug("%s: closing the client failed: %s", self._provider_name, exc)
        self._credential = ""
        self._options = None

    # ------------------------------------------------------------------
    # hooks for subclasses
    # ------------------------------------------------------------------

    def _load_credential(self, options: SttOptions) -> str:
        """Read the key out of the Windows credential store.

        Three places are tried, in order: the key handed over directly (the
        router does this when the worker already resolved the ``stt`` slot), the
        entry named by ``voice.stt.credential_ref``, and this provider's own
        :attr:`default_ref`.

        Raises:
            SttError: Nothing is stored. The message names the entry to fill in,
                because "no key" is a settings problem and the user is the only
                one who can fix it.
        """
        explicit = options.extra.get("credential")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

        wanted: list[str] = []
        configured = options.option("credential_ref")
        if configured:
            wanted.append(configured)
        if self.default_ref and self.default_ref not in wanted:
            wanted.append(self.default_ref)

        store = get_secrets()
        for ref in wanted:
            try:
                value = store.get(ref)
            except AyrisError as exc:
                # A locked or missing store is not this engine's problem to
                # explain twice: log the reason, then report the missing key.
                _log.warning("%s: credential %r unavailable: %s", self.name, ref, exc.technical)
                continue
            if value:
                return value

        named = ", ".join(f"«{ref}»" for ref in wanted) or "ключ"
        raise SttError(
            f"{self.name}: no credential in {wanted or ['<none>']}",
            user_message=(
                f"Ключ для {self._provider_name} не найден. "
                f"Сохраните запись {named} в настройках голоса, "
                f"или переключите распознавание на офлайн."
            ),
        )

    def _endpoint(self, options: SttOptions) -> str:
        """The base URL: the configured one, or :attr:`default_endpoint`."""
        return options.option("endpoint") or self.default_endpoint

    @abstractmethod
    def _build_request(self, audio: AudioBuffer, options: SttOptions) -> _Request:
        """Build the one request this provider needs.

        Args:
            audio: Mono 16 kHz PCM, already prepared.
            options: Language, punctuation and the provider's extras.

        Returns:
            URL, headers and body as bytes, so nothing has to be re-serialised
            later and no body reaches a log line.
        """

    @abstractmethod
    def _parse_response(self, data: bytes, options: SttOptions) -> TranscriptResult:
        """Turn a 2xx body into a result.

        Error statuses never reach here - :meth:`_attempt_transcribe` maps them
        first - so a subclass only deals with the shape of a successful answer.

        Raises:
            SttError: The body is JSON but not the documented shape, which means
                the provider changed its API.
        """

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _attempt_transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """One HTTP round trip: build, send, map the status, parse.

        Raises:
            QuotaError: 429. The router falls back without retrying.
            NetworkError: 5xx. Retriable, and a reason to re-probe connectivity.
            SttError: 401/403, any other non-2xx, or a body that does not parse.
            httpx.TimeoutException, httpx.TransportError: left to the retry loop.
        """
        options = self._require_loaded()
        client = self._client
        if client is None:  # pragma: no cover - load() always sets it
            raise SttError(
                f"{self.name}: HTTP client is gone",
                user_message="Внутренняя ошибка облачного распознавания.",
            )

        request = self._build_request(audio, options)
        _log.debug(
            "%s: POST %s, headers=%s, %d bytes of audio",
            self.name,
            request.url,
            _scrub_headers(request.headers),
            len(request.body),
        )

        response = client.post(request.url, headers=dict(request.headers), content=request.body)
        status = response.status_code
        data = response.content
        _log.debug("%s: HTTP %d, %d bytes back", self.name, status, len(data))

        if 200 <= status < 300:
            return self._parse_response(data, options)
        self._raise_for_status(status, data)

    def _decode_json(self, data: bytes) -> JsonObject:
        """Parse a provider response body.

        Raises:
            SttError: The body is not a JSON object. Truncated to 200 characters
                in the technical message: an HTML error page from a proxy is
                useful to see the start of and pointless to log in full.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            snippet = data[:200].decode("utf-8", errors="replace")
            raise SttError(
                f"{self._provider_name}: response is not JSON: {snippet!r}",
                user_message=f"{self._provider_name} вернул непонятный ответ.",
            ) from exc
        if not isinstance(payload, dict):
            raise SttError(
                f"{self._provider_name}: response is {type(payload).__name__}, not an object",
                user_message=f"{self._provider_name} вернул непонятный ответ.",
            )
        return payload

    def _result(
        self,
        *,
        text: str,
        confidence: float,
        segments: tuple[TranscriptSegment, ...] = (),
        language: str,
        model: str,
    ) -> TranscriptResult:
        """Assemble a result with the fields every provider fills the same way."""
        return TranscriptResult(
            text=text.strip(),
            confidence=max(0.0, min(1.0, confidence)),
            segments=segments,
            language=language,
            engine=self.name,
            device="cloud",
            model=model,
        )

    def _raise_for_status(self, status: int, data: bytes) -> NoReturn:
        """Map a non-2xx status onto the exception the router acts on.

        The three cases are not interchangeable. 429 means the account is out of
        budget, and retrying spends the next minute hitting the same wall; 5xx
        means the service is having a bad day and the same request may well work
        in a second; 401 means the key is wrong and nothing but the user can fix
        it.

        Raises:
            QuotaError, NetworkError, SttError: One of them, always.
        """
        detail = _error_detail(data)
        if status == 429:
            raise QuotaError(
                f"{self.name}: quota exceeded (429){detail}",
                user_message=(
                    f"Превышен лимит запросов {self._provider_name}. "
                    f"Проверьте квоту в личном кабинете сервиса."
                ),
            )
        if status >= 500:
            raise NetworkError(
                f"{self.name}: server error {status}{detail}",
                user_message=(f"Сервис {self._provider_name} сейчас недоступен (ошибка {status})."),
            )
        if status in (401, 403):
            raise SttError(
                f"{self.name}: authentication rejected ({status}){detail}",
                user_message=(
                    f"{self._provider_name} не принял ключ. " f"Проверьте ключ в настройках голоса."
                ),
            )
        raise SttError(
            f"{self.name}: HTTP {status}{detail}",
            user_message=f"{self._provider_name} отклонил запрос (ошибка {status}).",
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Delay before retry ``attempt``: exponential, capped, with jitter.

        The jitter matters more than it looks. Ayris runs on many machines that
        may lose the same Wi-Fi at the same moment, and a fixed backoff would
        send all of them back at the provider in the same millisecond.
        """
        capped: float = min(BACKOFF_BASE_SEC * float(2 ** (attempt - 1)), BACKOFF_MAX_SEC)
        # Half fixed, half random: never longer than the cap, never zero.
        jitter: float = random_secrets.randbelow(1000) / 1000.0
        return capped * 0.5 + capped * 0.5 * jitter


def cloud_engine_names() -> tuple[str, ...]:
    """Online provider names the settings window may offer.

    Unlike the offline list there is nothing to check availability of: all four
    need only httpx, which is a hard dependency. Whether a provider actually
    works depends on a key and a network, and neither can be established without
    spending a request.
    """
    return tuple(CLOUD_ENGINE_ENTRYPOINTS)


def create_cloud_engine(name: str) -> CloudSttEngine:
    """Build the online engine ``voice.stt.online_engine`` names.

    Imported on resolve, like the offline engines, so one broken provider module
    cannot take the other three with it.

    Raises:
        SttError: The name is not a known provider, or its module is broken. Not
            quietly substituted: a user who configured Azure and silently got
            Yandex would be debugging the wrong account.
    """
    entrypoint = CLOUD_ENGINE_ENTRYPOINTS.get(name)
    if entrypoint is None:
        known = ", ".join(sorted(CLOUD_ENGINE_ENTRYPOINTS))
        raise SttError(
            f"unknown cloud stt engine {name!r}, expected one of {known}",
            user_message=f"Неизвестный сервис распознавания речи: {name}.",
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only a broken checkout
        raise SttError(
            f"cannot import cloud stt engine {name!r}: {exc}",
            user_message=f"Не удалось загрузить движок распознавания «{name}».",
        ) from exc
    factory: type[CloudSttEngine] = getattr(module, attribute)
    return factory()
