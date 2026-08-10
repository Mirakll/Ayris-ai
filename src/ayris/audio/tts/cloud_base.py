"""What the four cloud voices have in common.

ElevenLabs, SpeechKit, Google Cloud and Azure all take text over HTTPS and give
back audio, and differ only in the shape of the request, the name of the auth
header, the container the audio arrives in and the units the speed knob is
measured in. They also fail in the same four ways - the key is wrong, the quota
is spent, the service is down, the connection dropped - and the router acts on
each of those differently. All of that lives here, so
:mod:`~ayris.audio.tts.elevenlabs_engine`,
:mod:`~ayris.audio.tts.yandex_tts_engine`,
:mod:`~ayris.audio.tts.google_tts_engine` and
:mod:`~ayris.audio.tts.azure_tts_engine` are a request builder and a format
declaration each.

**Everyone is asked for raw PCM.** All four can return uncompressed audio, and
:func:`decode_audio` therefore only ever has to pass bytes through or unwrap a
RIFF header with :mod:`wave` from the standard library. That is a deliberate
choice over the richer formats: an mp3 or ogg decoder would be a new pinned
dependency in two files for audio that is going straight into a player which
wants ``int16`` anyway. A provider that answers in a compressed format despite
being asked not to gets one attempt through an optional ``soundfile`` and
otherwise a clear error - see :func:`decode_audio`.

**Scales are converted, not guessed.** Ayris speaks in multipliers: speed 0.5–2.0,
pitch 0.5–2.0, volume 0–100. Every provider measures differently - ElevenLabs
allows 0.7–1.2 and nothing else, SpeechKit takes 0.1–3.0, Google wants semitones
for pitch, Azure wants percentages in SSML. :func:`map_range`,
:func:`semitones`, :func:`percent_offset` and :func:`decibels` do the arithmetic
in one place with tests on the endpoints, because a pitch that is off by an
octave sounds like a bug in the voice rather than a bug in a conversion.

**A key never reaches a log.** :func:`scrub_headers` masks the value of any
header whose name looks like auth, and request bodies are never logged: they
contain the text the user asked to have spoken, which
``privacy`` says is theirs. What is logged is the provider, the status and the
provider's own error phrase, which is enough to diagnose a failure without any
of it.

**Timeouts are separate and the deadline is absolute.** A stalled DNS lookup
fails after :data:`CONNECT_TIMEOUT_SEC` instead of waiting out the read timeout,
and one synthesis - retries included - is capped at :data:`DEADLINE_SEC`, past
which the router should already be speaking locally. 402/429 is never retried:
the next attempt hits the same wall and spends another second doing it.
"""

from __future__ import annotations

import importlib
import io
import json
import secrets as random_secrets
import wave
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter, sleep
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, NoReturn, Protocol, runtime_checkable

import httpx

from ayris.audio.tts.base import (
    MAX_PITCH,
    MAX_SPEED,
    MIN_PITCH,
    MIN_SPEED,
    SAMPLE_WIDTH,
    AudioChunk,
    TtsEngine,
    TtsOptions,
    VoiceSpec,
)
from ayris.core.errors import AyrisError, TtsError
from ayris.core.models import JsonObject
from ayris.core.secrets import get_secrets, mask
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "BACKOFF_BASE_SEC",
    "BACKOFF_MAX_SEC",
    "CLOUD_ENGINE_ENTRYPOINTS",
    "CONNECT_TIMEOUT_SEC",
    "DEADLINE_SEC",
    "MAX_RETRIES",
    "READ_TIMEOUT_SEC",
    "STREAM_CHUNK_BYTES",
    "AudioFormat",
    "CloudTtsEngine",
    "TtsAuthError",
    "TtsNetworkError",
    "TtsQuotaError",
    "UsageRecorder",
    "cloud_engine_names",
    "create_cloud_engine",
    "decibels",
    "decode_audio",
    "is_cloud_engine",
    "map_range",
    "percent_offset",
    "scrub_headers",
    "semitones",
]

_log = get_logger(__name__)

#: Cloud voices, kept apart from
#: :data:`~ayris.audio.tts.base.ENGINE_ENTRYPOINTS` for the same reason the
#: recognisers are: that list answers "what can run on this machine", this one
#: answers "what can this account reach". The settings window shows both and the
#: router picks one from each.
CLOUD_ENGINE_ENTRYPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "elevenlabs": "ayris.audio.tts.elevenlabs_engine:ElevenLabsTtsEngine",
        "yandex": "ayris.audio.tts.yandex_tts_engine:YandexTtsEngine",
        "google": "ayris.audio.tts.google_tts_engine:GoogleTtsEngine",
        "azure": "ayris.audio.tts.azure_tts_engine:AzureTtsEngine",
    }
)

#: Connect timeout: DNS plus handshake. Short, because the fallback to a local
#: voice is cheap and a user waiting in silence is not.
CONNECT_TIMEOUT_SEC: Final = 3.0

#: Read timeout, first byte to last. Synthesis of a sentence is fast everywhere;
#: what this really covers is a provider queueing the request behind others.
READ_TIMEOUT_SEC: Final = 15.0

#: Absolute cap on one synthesis, retries included. Past this the router speaks
#: locally, so a retry that lands afterwards would only be talking over it.
DEADLINE_SEC: Final = 20.0

#: Retries after a transient failure. Not applied to 4xx, which are permanent.
MAX_RETRIES: Final = 2

#: Base delay for exponential backoff, in seconds.
BACKOFF_BASE_SEC: Final = 0.4

#: Maximum backoff delay, in seconds.
BACKOFF_MAX_SEC: Final = 3.0

#: Bytes pulled from a streaming response before a chunk is handed on. At 22 kHz
#: mono ``int16`` this is about 180 ms of speech - long enough that the player is
#: never starved by per-chunk overhead, short enough that the first sound leaves
#: the speakers well inside the 1.5 s the specification allows.
STREAM_CHUNK_BYTES: Final = 8192

#: Sent with every request. Version-free on purpose: a provider that rate-limits
#: by client string must not see a new client on every Ayris update.
_USER_AGENT: Final = "Ayris"

#: Header names whose value is a secret. Substring match, lowercased, so a bare
#: "key" also covers ``xi-api-key`` and ``Ocp-Apim-Subscription-Key``.
_SECRET_HEADERS: Final = ("authorization", "key", "token", "secret", "auth")

#: How much of an error body goes into the technical message.
_ERROR_SNIPPET: Final = 300

#: How long a provider's voice list stays fresh. The settings window may reopen
#: several times in a session and the list does not change between reopenings;
#: an hour is short enough that a voice added on the provider's site shows up
#: the same day without a restart.
_VOICE_CACHE_TTL_SEC: Final = 3600.0


class AudioFormat(StrEnum):
    """Container a provider's answer arrives in.

    Only the two uncompressed ones are asked for; :attr:`COMPRESSED` exists so
    that a provider which ignores the format parameter fails with a sentence
    rather than by handing mp3 frames to the sound card.
    """

    #: Headerless little-endian ``int16``. The rate is whatever was requested.
    PCM = "pcm"

    #: RIFF/WAVE. Google returns this even when asked for LINEAR16.
    WAV = "wav"

    #: Anything else: mp3, ogg-opus, flac. Decoded only if ``soundfile`` happens
    #: to be installed.
    COMPRESSED = "compressed"


class TtsNetworkError(TtsError):
    """The service could not be reached, or answered 5xx. Worth retrying."""


class TtsQuotaError(TtsError):
    """Out of budget: 402 or 429. Never retried - the wall does not move."""


class TtsAuthError(TtsError):
    """The key was rejected: 401 or 403. Only the user can fix this."""


@runtime_checkable
class UsageRecorder(Protocol):
    """Where an engine reports what it just spent.

    A Protocol rather than an import so that an engine never reaches into the
    database layer: :class:`~ayris.audio.tts.quota.QuotaTracker` satisfies this,
    and so does a list in a test.
    """

    def record(self, provider: str, characters: int) -> None:
        """Note that ``characters`` were billed to ``provider``."""


@dataclass(frozen=True, slots=True)
class CloudRequest:
    """One prepared HTTP request.

    Attributes:
        url: Absolute URL, region already substituted.
        headers: Auth included. Only ever logged through :func:`scrub_headers`.
        body: Serialised payload. Never logged: it contains the user's text.
        audio_format: What the answer will be, so the decoder does not have to
            sniff bytes.
        sample_rate: Rate the provider was asked for, used when the format
            carries no header of its own.
        json_field: Dotted path to a base64 audio field, for providers that
            answer in JSON. Empty when the body is the audio.
        method: HTTP verb. ``GET`` for the voice list.
    """

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    body: bytes = b""
    audio_format: AudioFormat = AudioFormat.PCM
    sample_rate: int = 0
    json_field: str = ""
    method: str = "POST"


# --------------------------------------------------------------------- scales


def map_range(
    value: float,
    source: tuple[float, float],
    target: tuple[float, float],
    *,
    neutral: float = 1.0,
) -> float:
    """Move a value from Ayris's scale onto a provider's, keeping 1.0 neutral.

    A plain linear interpolation would be wrong here. Ayris's speed runs
    0.5–1.0–2.0 with 1.0 in the middle by meaning but not by arithmetic, so
    interpolating end to end would map "normal speed" onto 0.75 of ElevenLabs'
    range and make every unmodified phrase drawl. Each half is mapped separately
    instead: ``neutral`` lands on the provider's own neutral, below it stretches
    down to their minimum, above it up to their maximum.

    Args:
        value: The Ayris-side multiplier.
        source: ``(min, max)`` of the Ayris scale.
        target: ``(min, max)`` the provider accepts.
        neutral: The Ayris value that means "unchanged", 1.0 for both knobs.

    Returns:
        A value inside ``target``, clamped at both ends.
    """
    low, high = source
    out_low, out_high = target
    # The provider's neutral: where 1.0 sits in their range, or their midpoint
    # when they are symmetric around it.
    out_neutral = min(max(neutral, out_low), out_high)
    if value <= neutral:
        span = neutral - low
        share = 0.0 if span <= 0.0 else (neutral - min(max(value, low), neutral)) / span
        return out_neutral - share * (out_neutral - out_low)
    span = high - neutral
    share = 0.0 if span <= 0.0 else (min(max(value, neutral), high) - neutral) / span
    return out_neutral + share * (out_high - out_neutral)


def percent_offset(value: float, limits: tuple[float, float], *, neutral: float = 1.0) -> int:
    """A multiplier as a signed percentage, the way SSML writes it.

    ``1.0`` is ``0``, ``1.5`` is ``+50``, ``0.5`` is ``-50``. Rounded to a whole
    percent because SSML takes no more precision than that anyway, and clamped
    to what the provider documents.
    """
    low, high = limits
    percent = (value - neutral) * 100.0
    return int(round(min(max(percent, low), high)))


def semitones(value: float, limits: tuple[float, float]) -> float:
    """A pitch multiplier in semitones, which is what Google's API takes.

    A multiplier *is* a frequency ratio, so the conversion is the real one -
    twelve semitones per octave - rather than a linear guess. 2.0 comes out at
    +12, 0.5 at -12, both well inside Google's ±20.
    """
    import math

    if value <= 0.0:
        return limits[0]
    tones = 12.0 * math.log2(value)
    return min(max(tones, limits[0]), limits[1])


def decibels(volume: int, limits: tuple[float, float]) -> float:
    """A 0–100 volume as a gain in dB.

    100 is 0 dB - the provider's own level, unchanged - and quieter settings
    scale down logarithmically, since loudness does not follow a linear slider.
    0 is the provider's floor, which every one of them treats as silence.
    """
    import math

    ratio = min(max(volume, 0), 100) / 100.0
    if ratio <= 0.0:
        return limits[0]
    gain = 20.0 * math.log10(ratio)
    return min(max(gain, limits[0]), limits[1])


# -------------------------------------------------------------------- decoding


def decode_audio(
    data: bytes,
    audio_format: AudioFormat,
    sample_rate: int,
    *,
    provider: str = "",
) -> AudioChunk:
    """Turn a provider's bytes into the one format the player understands.

    The whole point of this function existing is that it is the only place in
    Ayris where a provider's container is known about. Everything downstream -
    the player, the cache, the shared-memory transport - sees little-endian
    ``int16``, and adding a fifth provider that answers in a fifth container
    means changing this and nothing else.

    Args:
        data: The response body.
        audio_format: What it is, as declared by the request that asked for it.
        sample_rate: Rate to attach when the bytes carry no header.
        provider: Named in the error, so a user knows which service to look at.

    Returns:
        An empty chunk for an empty body - a provider answering 200 with nothing
        is odd but not fatal, and the router treats "no audio" as "say it
        locally".

    Raises:
        TtsError: The bytes are compressed and nothing can decode them, or a WAV
            arrives at a sample width Ayris does not carry.
    """
    if not data:
        return AudioChunk(b"", sample_rate or 1)
    if audio_format is AudioFormat.PCM:
        # An odd byte count means a truncated frame, and playing it would shift
        # every sample after it by one byte - white noise, not speech.
        return AudioChunk(data[: len(data) - len(data) % SAMPLE_WIDTH], sample_rate or 1)
    if audio_format is AudioFormat.WAV or data[:4] == b"RIFF":
        return _decode_wav(data, provider)
    return _decode_compressed(data, provider)


def _decode_wav(data: bytes, provider: str) -> AudioChunk:
    """Unwrap a RIFF container with the standard library.

    Raises:
        TtsError: Unreadable, or not 16-bit. Converting 8- or 24-bit samples by
            hand is possible and would be the wrong thing to add here: no
            provider Ayris talks to returns them when asked for LINEAR16, so
            seeing one means the request was built wrong.
    """
    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            width = source.getsampwidth()
            if width != SAMPLE_WIDTH:
                raise TtsError(
                    f"{provider or 'cloud'}: WAV is {width * 8}-bit, expected 16-bit",
                    user_message="Сервис синтеза вернул звук в неподдерживаемом формате.",
                )
            frames = source.readframes(source.getnframes())
            return AudioChunk(
                pcm=frames,
                sample_rate=source.getframerate(),
                channels=max(1, source.getnchannels()),
            )
    except (wave.Error, EOFError, OSError) as exc:
        raise TtsError(
            f"{provider or 'cloud'}: cannot read the WAV response: {exc}",
            user_message="Сервис синтеза вернул повреждённый звук.",
        ) from exc


def _decode_compressed(data: bytes, provider: str) -> AudioChunk:
    """Last resort for audio that arrived compressed anyway.

    Every engine here asks for uncompressed audio, so reaching this means the
    provider ignored the format parameter or changed its default. ``soundfile``
    is tried because it may be present for other reasons, and is imported inside
    the function so that it is never a requirement.

    Raises:
        TtsError: Nothing on this machine can decode it. The message names the
            package, because installing it is a real fix the user can apply
            without waiting for a release.
    """
    try:
        soundfile = importlib.import_module("soundfile")
    except ImportError as exc:
        raise TtsError(
            f"{provider or 'cloud'}: response is compressed audio and soundfile is missing",
            user_message=(
                "Сервис синтеза вернул сжатый звук (mp3/ogg), который Ayris не умеет "
                "распаковать. Установите пакет soundfile или выберите другой сервис."
            ),
        ) from exc
    try:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="int16", always_2d=False)
        pcm = bytes(samples.tobytes())
        channels = 1 if samples.ndim == 1 else int(samples.shape[1])
    except Exception as exc:  # soundfile raises its own error types
        raise TtsError(
            f"{provider or 'cloud'}: cannot decode the compressed response: {exc}",
            user_message="Сервис синтеза вернул звук в неподдерживаемом формате.",
        ) from exc
    return AudioChunk(pcm=pcm, sample_rate=int(rate), channels=channels)


# ------------------------------------------------------------------- logging


def scrub_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Replace every auth header value with a placeholder.

    Matched case-insensitively and by substring, because the four providers
    between them use ``xi-api-key``, ``Authorization``, ``X-Goog-Api-Key`` and
    ``Ocp-Apim-Subscription-Key``, and the next one will most likely also have
    "key", "token" or "auth" somewhere in the name.
    """
    return {
        key: ("***" if any(marker in key.lower() for marker in _SECRET_HEADERS) else value)
        for key, value in headers.items()
    }


def _error_detail(data: bytes) -> str:
    """The provider's own explanation, trimmed, for the technical message."""
    if not data:
        return ""
    text = data[:_ERROR_SNIPPET].decode("utf-8", errors="replace").strip()
    return f": {text}" if text else ""


def _as_positive(value: object, fallback: float) -> float:
    """Read a positive number out of :attr:`TtsOptions.extra`, or fall back."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value) if value > 0.0 else fallback


def _as_retries(value: object) -> int:
    """Read the retry budget. Zero is legitimate: try once, then fall back."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return MAX_RETRIES
    return max(0, int(value))


# --------------------------------------------------------------------- engine


class CloudTtsEngine(TtsEngine):
    """Base for cloud voices: credentials, HTTP, retries, decoding.

    A subclass declares its scales and implements :meth:`_build_request`.
    Everything else - reading the key out of the credential store, opening the
    client, mapping a status onto one of the three typed errors, backing off,
    decoding the container, counting characters - happens here, so that four
    providers cannot disagree about any of it.

    Instances are not thread-safe, matching :class:`~ayris.audio.tts.base.TtsEngine`:
    the router serialises access, and the streaming generator is consumed by one
    thread at a time.
    """

    #: Credential entry to look for when the configuration names none. Equal to
    #: the :data:`~ayris.core.secrets.KNOWN_SLOTS` reference, so filling in the
    #: provider's field in the settings is enough.
    default_ref: ClassVar[str] = ""

    #: Base URL used when ``endpoint`` is not overridden.
    default_endpoint: ClassVar[str] = ""

    #: Human name for log lines and for the sentence the user reads.
    title: ClassVar[str] = ""

    #: Whether the provider has a streaming endpoint worth using. When ``False``
    #: the base class still streams by sentence, which is the ordinary contract.
    supports_streaming: ClassVar[bool] = False

    #: A cloud engine needs nothing but httpx, which is a hard dependency.
    package: ClassVar[str] = "httpx"
    module: ClassVar[str] = "httpx"

    #: Nothing is resident, so the worker's RAM check has nothing to reserve.
    memory_factor: ClassVar[float] = 0.0

    #: Rate asked of the provider, and therefore the rate the answer arrives at.
    #: Higher than the local default because these services synthesize at 24 kHz
    #: natively and downsampling on their side would cost quality for nothing.
    native_sample_rate: ClassVar[int] = 24000

    #: Provider speed range. ``(min, max)``, mapped from Ayris's 0.5–2.0.
    speed_limits: ClassVar[tuple[float, float]] = (0.5, 2.0)

    #: Provider pitch range, in the provider's own units.
    pitch_limits: ClassVar[tuple[float, float]] = (0.5, 2.0)

    __slots__ = ("_client", "_credential", "_usage", "_voice_cache", "_voice_cache_at")

    def __init__(self) -> None:
        super().__init__()
        self._client: httpx.Client | None = None
        self._credential: str = ""
        self._usage: UsageRecorder | None = None
        self._voice_cache: tuple[VoiceSpec, ...] = ()
        self._voice_cache_at: float = 0.0

    # ---------------------------------------------------------- description

    @property
    def provider_name(self) -> str:
        """Service name for messages: :attr:`title`, or the engine name."""
        return self.title or self.name

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Empty: every provider here speaks far more languages than Ayris uses."""
        return ()

    @property
    def device(self) -> str:
        """Not this machine. Logged so the pipeline says where the time went."""
        return "cloud"

    # ------------------------------------------------------------ lifecycle

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Read the credential and open the HTTP client.

        No model is fetched and nothing is resident afterwards, so this is fast
        and can be repeated. It can still fail, and failing here rather than at
        the first phrase is the point: the router learns there is no key before
        it has promised the user any sound.

        Raises:
            TtsAuthError: No credential is stored under any of the names tried.
        """
        self._credential = self._load_credential(options)
        self._client = self._build_client(options)
        self._voice = voice
        self._options = options
        usage = options.extra.get("usage")
        self._usage = usage if isinstance(usage, UsageRecorder) else None
        _log.info(
            "%s: облачный голос готов, ключ %s",
            self.provider_name,
            mask(self._credential),
        )

    def unload(self) -> None:
        """Close the client. Safe to call twice and never raises."""
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - httpx does not raise here
                _log.debug("%s: не удалось закрыть клиент: %s", self.provider_name, exc)
        self._credential = ""
        self._voice = None
        self._usage = None

    def _build_client(self, options: TtsOptions) -> httpx.Client:
        """Open the client. The transport is injectable, which is how tests work."""
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

    # ------------------------------------------------------------ synthesis

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        """One phrase, in one request.

        Raises:
            TtsNetworkError: Unreachable, timed out, or 5xx after the retries.
            TtsQuotaError: 402 or 429.
            TtsAuthError: 401 or 403.
            TtsError: Any other status, or an undecodable body.
        """
        request = self._build_request(text, speed, pitch, stream=False)
        data = self._send(request, text)
        return decode_audio(
            data,
            request.audio_format,
            request.sample_rate or self.native_sample_rate,
            provider=self.provider_name,
        )

    def synthesize_stream(
        self,
        text: str,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield audio as it arrives, when the provider streams.

        For a provider with a streaming endpoint this is what keeps a long
        answer inside the 1.5 s budget: the first :data:`STREAM_CHUNK_BYTES` are
        handed to the player while the rest of the sentence is still being
        synthesized on the other side of the connection. For the others the base
        class's sentence-by-sentence loop is already the right behaviour and is
        used unchanged.

        Streaming is only useful for headerless PCM: a WAV or a JSON envelope
        cannot be interpreted until the whole body has arrived, and pretending
        otherwise would hand the player a RIFF header as if it were samples.

        Raises:
            The same as :meth:`_synthesize`. Raised from the generator, which
            means a failure on the third chunk arrives after two good ones -
            the router's fallback generator is written for exactly that.
        """
        from ayris.audio.tts.sentence_split import is_speakable

        if not self.supports_streaming:
            yield from super().synthesize_stream(text, voice, speed, pitch)
            return
        if not is_speakable(text):
            return
        self._ensure_voice(voice)
        request = self._build_request(
            text,
            self._resolved_speed(speed),
            self._resolved_pitch(pitch),
            stream=True,
        )
        if request.audio_format is not AudioFormat.PCM:  # pragma: no cover - defensive
            yield from super().synthesize_stream(text, voice, speed, pitch)
            return
        rate = request.sample_rate or self.native_sample_rate
        for block in self._send_stream(request, text):
            yield AudioChunk(block, rate)

    def _resolved_speed(self, speed: float | None) -> float:
        """The rate this call should use, clamped."""
        from ayris.audio.tts.base import clamp_speed

        return clamp_speed(self._options.speed if speed is None else speed)

    def _resolved_pitch(self, pitch: float | None) -> float:
        """The pitch this call should use, clamped."""
        from ayris.audio.tts.base import clamp_pitch

        return clamp_pitch(self._options.pitch if pitch is None else pitch)

    # ------------------------------------------------------- subclass hooks

    @abstractmethod
    def _build_request(
        self,
        text: str,
        speed: float,
        pitch: float,
        *,
        stream: bool,
    ) -> CloudRequest:
        """Build the one request this provider needs.

        Args:
            text: The phrase, already checked to be speakable.
            speed: Ayris-side multiplier, 0.5–2.0, already clamped.
            pitch: Ayris-side multiplier, 0.5–2.0, already clamped.
            stream: Whether the streaming endpoint was asked for. Providers
                without one ignore it.

        Returns:
            URL, headers, body and the shape of the answer.
        """

    def _voice_request(self) -> CloudRequest | None:
        """The request that lists the provider's voices, if it has such an API.

        Returns ``None`` by default: a provider whose voices are a documented
        constant answers :meth:`voices` from a table instead, and asking the
        network for something that cannot change is a request the user pays for.
        """
        return None

    def _parse_voices(self, data: bytes) -> tuple[VoiceSpec, ...]:
        """Turn the voice-list body into specs. Only called when there is one."""
        del data
        return ()

    # ------------------------------------------------------------ voice list

    def list_voices(self, *, refresh: bool = False) -> tuple[VoiceSpec, ...]:
        """Voices this account may use, cached for an hour.

        The settings window calls this every time it opens its combo box, and
        without the cache that would be a billed request per open on the
        providers that charge for API calls rather than for characters.

        Returns:
            Empty when the provider has no list endpoint or the call failed. A
            settings window that showed an error box because a voice list did
            not load would be worse than one that lets the user type a name.
        """
        now = perf_counter()
        fresh = self._voice_cache and (now - self._voice_cache_at) < _VOICE_CACHE_TTL_SEC
        if fresh and not refresh:
            return self._voice_cache
        request = self._voice_request()
        if request is None:
            return ()
        try:
            data = self._send(request, "", count_usage=False)
            voices = self._parse_voices(data)
        except (AyrisError, httpx.HTTPError) as exc:
            _log.warning("%s: не удалось получить список голосов: %s", self.provider_name, exc)
            return self._voice_cache
        self._voice_cache = voices
        self._voice_cache_at = now
        return voices

    # ------------------------------------------------------------------ HTTP

    def _send(self, request: CloudRequest, text: str, *, count_usage: bool = True) -> bytes:
        """Send one request with retries, and return the audio bytes.

        The retry loop is bounded twice - by the attempt count and by the
        absolute deadline - because the two protect against different failures.
        A provider that refuses fast would otherwise burn all its retries in
        20 ms and give up before the network had a chance to recover, and a
        provider that hangs would otherwise hold the user in silence for three
        full read timeouts.

        Raises:
            TtsNetworkError, TtsQuotaError, TtsAuthError, TtsError: as mapped by
            :meth:`_raise_for_status`, or wrapped from a transport failure.
        """
        client = self._require_client()
        started = perf_counter()
        deadline = _as_positive(self._options.extra.get("deadline_sec"), DEADLINE_SEC)
        retries = _as_retries(self._options.extra.get("max_retries"))
        attempt = 0
        last_error: Exception | None = None

        while True:
            attempt += 1
            try:
                data = self._attempt(client, request)
            except (httpx.TimeoutException, httpx.TransportError, TtsNetworkError) as exc:
                last_error = exc
                delay = self._backoff_delay(attempt)
                if attempt > retries or (perf_counter() - started + delay) >= deadline:
                    break
                _log.warning(
                    "%s: %s на попытке %d, повтор через %.1f с",
                    self.provider_name,
                    type(exc).__name__,
                    attempt,
                    delay,
                )
                sleep(delay)
                continue
            if count_usage:
                self._count(text)
            return data

        raise self._give_up(last_error, attempt)

    def _attempt(self, client: httpx.Client, request: CloudRequest) -> bytes:
        """One round trip: send, map the status, unwrap a JSON envelope."""
        response = self._roundtrip(client, request)
        data = response.content
        _log.debug("%s: HTTP %d, %d байт", self.name, response.status_code, len(data))
        if not 200 <= response.status_code < 300:
            self._raise_for_status(response.status_code, data)
        return self._unwrap(data, request)

    def _send_stream(self, request: CloudRequest, text: str) -> Iterator[bytes]:
        """Stream one response, yielding blocks of PCM as they arrive.

        Not retried. A retry would mean replaying the phrase from the start, and
        by the time a stream fails the player is usually already speaking - the
        router's fallback resumes from the next sentence instead, which is the
        difference between a stutter and the user hearing the first half twice.

        Raises:
            The same as :meth:`_send`, but on the first failure.
        """
        client = self._require_client()
        headers = dict(request.headers)
        _log.debug(
            "%s: %s %s (поток), заголовки=%s",
            self.name,
            request.method,
            request.url,
            scrub_headers(headers),
        )
        try:
            with client.stream(
                request.method,
                request.url,
                headers=headers,
                content=request.body or None,
            ) as response:
                if not 200 <= response.status_code < 300:
                    response.read()
                    self._raise_for_status(response.status_code, response.content)
                pending = b""
                for block in response.iter_bytes(STREAM_CHUNK_BYTES):
                    pending += block
                    usable = len(pending) - len(pending) % SAMPLE_WIDTH
                    if usable:
                        yield pending[:usable]
                        pending = pending[usable:]
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise self._give_up(exc, 1) from exc
        self._count(text)

    def _roundtrip(self, client: httpx.Client, request: CloudRequest) -> httpx.Response:
        """Send and return the response, logging everything but the body."""
        headers = dict(request.headers)
        _log.debug(
            "%s: %s %s, заголовки=%s, тело %d байт",
            self.name,
            request.method,
            request.url,
            scrub_headers(headers),
            len(request.body),
        )
        return client.request(
            request.method,
            request.url,
            headers=headers,
            content=request.body or None,
        )

    def _unwrap(self, data: bytes, request: CloudRequest) -> bytes:
        """Pull audio out of a JSON envelope, when the provider uses one.

        Google is the reason this exists: it answers ``{"audioContent": "<base64>"}``
        even for LINEAR16.

        Raises:
            TtsError: The field is missing or is not base64, which means the API
                changed shape.
        """
        if not request.json_field:
            return data
        import base64
        import binascii

        payload = self._decode_json(data)
        value: object = payload
        for part in request.json_field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if not isinstance(value, str) or not value:
            raise TtsError(
                f"{self.provider_name}: no {request.json_field!r} in the response",
                user_message=f"{self.provider_name} вернул ответ без звука.",
            )
        try:
            return base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise TtsError(
                f"{self.provider_name}: {request.json_field} is not valid base64",
                user_message=f"{self.provider_name} вернул повреждённый звук.",
            ) from exc

    def _decode_json(self, data: bytes) -> JsonObject:
        """Parse a response body as a JSON object.

        Raises:
            TtsError: Not JSON, or not an object. The snippet is truncated: an
                HTML error page from a proxy is useful to see the start of and
                pointless to log in full.
        """
        try:
            payload = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            snippet = data[:200].decode("utf-8", errors="replace")
            raise TtsError(
                f"{self.provider_name}: response is not JSON: {snippet!r}",
                user_message=f"{self.provider_name} вернул непонятный ответ.",
            ) from exc
        if not isinstance(payload, dict):
            raise TtsError(
                f"{self.provider_name}: response is {type(payload).__name__}, not an object",
                user_message=f"{self.provider_name} вернул непонятный ответ.",
            )
        return payload

    def _raise_for_status(self, status: int, data: bytes) -> NoReturn:
        """Map a non-2xx onto the exception the router acts on.

        The four cases are not interchangeable, and the router does something
        different with each: 402/429 means the account is out of budget and the
        next attempt spends a second hitting the same wall; 5xx means the
        service is having a bad minute and the same request may well work in
        one; 401 means the key is wrong and only the user can fix it; anything
        else is a request Ayris built wrong.

        Raises:
            TtsQuotaError, TtsNetworkError, TtsAuthError, TtsError: one of them,
            always.
        """
        detail = _error_detail(data)
        if status in (402, 429):
            raise TtsQuotaError(
                f"{self.name}: quota exceeded ({status}){detail}",
                user_message=(
                    f"Исчерпан лимит {self.provider_name}. "
                    f"Проверьте квоту в личном кабинете сервиса."
                ),
            )
        if status >= 500:
            raise TtsNetworkError(
                f"{self.name}: server error {status}{detail}",
                user_message=f"Сервис {self.provider_name} сейчас недоступен (ошибка {status}).",
            )
        if status in (401, 403):
            raise TtsAuthError(
                f"{self.name}: authentication rejected ({status}){detail}",
                user_message=(
                    f"{self.provider_name} не принял ключ. Проверьте ключ в настройках голоса."
                ),
            )
        raise TtsError(
            f"{self.name}: HTTP {status}{detail}",
            user_message=f"{self.provider_name} отклонил запрос (ошибка {status}).",
        )

    def _give_up(self, error: Exception | None, attempts: int) -> TtsError:
        """The exception to raise once the retries are spent.

        A 5xx keeps its own message: the service said what went wrong, and
        replacing that with "check your connection" would send the user to look
        at a router that is working fine.
        """
        if isinstance(error, TtsNetworkError):
            _log.warning(
                "%s: сдаёмся после %d попыт(ок): %s",
                self.provider_name,
                attempts,
                error.technical,
            )
            return error
        return TtsNetworkError(
            f"{self.name}: {type(error).__name__} after {attempts} attempt(s): {error}",
            user_message=(f"{self.provider_name} не отвечает. Проверьте подключение к интернету."),
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Delay before retry ``attempt``: exponential, capped, with jitter.

        The jitter is not decoration. Ayris runs on many machines that may lose
        the same Wi-Fi at the same moment, and a fixed backoff would send all of
        them back at the provider in the same millisecond.
        """
        capped: float = min(BACKOFF_BASE_SEC * float(2 ** (attempt - 1)), BACKOFF_MAX_SEC)
        jitter: float = random_secrets.randbelow(1000) / 1000.0
        return capped * 0.5 + capped * 0.5 * jitter

    # --------------------------------------------------------------- helpers

    def _require_client(self) -> httpx.Client:
        """The open client.

        Raises:
            TtsError: :meth:`load` has not run.
        """
        client = self._client
        if client is None:
            raise TtsError(
                f"{self.name}: HTTP client is gone",
                user_message="Облачный синтез речи не инициализирован.",
            )
        return client

    def _count(self, text: str) -> None:
        """Report the characters just billed, if anyone is listening.

        Length only - the text itself never leaves this frame. A recorder that
        raises is logged and ignored: a failure to write a counter must not
        swallow audio the user already paid for.
        """
        recorder = self._usage
        if recorder is None or not text:
            return
        try:
            recorder.record(self.name, len(text))
        except Exception as exc:  # pragma: no cover - a tracker should not raise
            _log.debug("%s: не удалось записать расход: %s", self.provider_name, exc)

    def _endpoint(self) -> str:
        """The base URL: the configured override, or :attr:`default_endpoint`."""
        return self._options.option("endpoint") or self.default_endpoint

    def _load_credential(self, options: TtsOptions) -> str:
        """Read the key out of the credential store.

        Three places are tried in order: a key handed over directly (which the
        router does when it already resolved the slot), the entry named by
        ``credential_ref``, and this provider's :attr:`default_ref`.

        Raises:
            TtsAuthError: Nothing is stored. The message names the entry to
                fill in, because a missing key is a settings problem and the
                user is the only one who can fix it.
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
                # explain twice: log why, then report the missing key.
                _log.warning("%s: запись %r недоступна: %s", self.name, ref, exc.technical)
                continue
            if value:
                return value

        named = ", ".join(f"«{ref}»" for ref in wanted) or "ключ"
        raise TtsAuthError(
            f"{self.name}: no credential in {wanted or ['<none>']}",
            user_message=(
                f"Ключ для {self.provider_name} не найден. "
                f"Сохраните запись {named} в настройках голоса, "
                f"или переключите синтез речи на локальный движок."
            ),
        )

    def _speed_for_provider(self, speed: float) -> float:
        """Ayris's 0.5–2.0 on this provider's own scale."""
        return map_range(speed, (MIN_SPEED, MAX_SPEED), self.speed_limits)

    def _pitch_for_provider(self, pitch: float) -> float:
        """Ayris's 0.5–2.0 pitch on this provider's own scale."""
        return map_range(pitch, (MIN_PITCH, MAX_PITCH), self.pitch_limits)


# ------------------------------------------------------------------ registry


def cloud_engine_names() -> tuple[str, ...]:
    """Cloud provider names the settings window may offer.

    Unlike the local list there is nothing to check availability of: all four
    need only httpx, which is a hard dependency. Whether one actually works
    depends on a key and a network, and neither can be established without
    spending a request.
    """
    return tuple(CLOUD_ENGINE_ENTRYPOINTS)


def is_cloud_engine(name: str) -> bool:
    """Whether ``name`` is one of the cloud providers."""
    return name in CLOUD_ENGINE_ENTRYPOINTS


def create_cloud_engine(name: str) -> CloudTtsEngine:
    """Build the cloud engine the settings name.

    Imported on resolve, like the local engines, so one broken provider module
    cannot take the other three with it.

    Raises:
        TtsError: Unknown provider, or a broken module. Never quietly
            substituted: a user who configured Azure and silently got Yandex
            would be debugging the wrong account.
    """
    entrypoint = CLOUD_ENGINE_ENTRYPOINTS.get(name)
    if entrypoint is None:
        known = ", ".join(sorted(CLOUD_ENGINE_ENTRYPOINTS))
        raise TtsError(
            f"unknown cloud tts engine {name!r}, expected one of {known}",
            user_message=f"Неизвестный сервис синтеза речи: {name}.",
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only a broken checkout
        raise TtsError(
            f"cannot import cloud tts engine {name!r}: {exc}",
            user_message=f"Не удалось загрузить синтез «{name}».",
        ) from exc
    factory: type[CloudTtsEngine] = getattr(module, attribute)
    return factory()
