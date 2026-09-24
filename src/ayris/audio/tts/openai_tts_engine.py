"""OpenAI-compatible cloud TTS: OpenRouter, OpenAI, and anything speaking /audio/speech.

Unlike the four named providers this one has no fixed home. It is the escape hatch
for "any cloud voice with an OpenAI-compatible endpoint": the user pastes a base URL
(``https://openrouter.ai/api/v1``, ``https://api.openai.com/v1``, a self-hosted
server, …), a model id and a key, and Ayris speaks through it. The wire format is
OpenAI's ``POST /audio/speech`` — ``{model, input, voice, response_format}`` with a
``Bearer`` key — which OpenRouter and most hosted TTS services implement verbatim, so
one engine covers a whole family of services instead of a class per vendor.

``response_format=wav`` is asked for by default because WAV carries its own sample
rate: :func:`~ayris.audio.tts.cloud_base.decode_audio` reads the rate from the RIFF
header, so a service synthesising at 22, 24 or 48 kHz all decode correctly without the
engine having to know which. ``pcm`` and ``mp3`` (and other compressed formats) are
available through the ``audio_format`` option for services that do not offer WAV; the
compressed ones decode only where the optional ``soundfile`` package is installed, as
everywhere else in this package.

There is no pitch control — the OpenAI schema has none — and ``speed`` is sent on the
provider's own 0.25–4.0 scale, and only when the user moved it off 1.0, so the default
request is exactly the minimal shape every compatible service documents. The model is
required: there is no universal default, so :meth:`load` refuses rather than send an
empty ``model`` the service would reject on every phrase.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.tts.base import TtsOptions, VoiceSpec
from ayris.audio.tts.cloud_base import AudioFormat, CloudRequest, CloudTtsEngine
from ayris.core.errors import TtsError

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["OpenAiTtsEngine"]

#: Voice used when the settings name none. ``alloy`` is OpenAI's account-independent
#: default and is accepted by the OpenAI-compatible services that mirror its catalogue.
_DEFAULT_VOICE: Final = "alloy"

#: Base URL used when ``endpoint`` is not overridden. OpenRouter rather than OpenAI
#: because the free voices the user found live there; any base URL replaces it.
_DEFAULT_ENDPOINT: Final = "https://openrouter.ai/api/v1"

#: What ``speed`` accepts in the OpenAI schema. Ayris's 0.5–2.0 maps onto it with 1.0
#: landing on 1.0, so an unchanged slider produces unchanged speech.
_SPEED_LIMITS: Final = (0.25, 4.0)

#: Ayris speed values this close to 1.0 are treated as "unchanged" and the field is
#: dropped, keeping the default request to the four keys every service expects.
_SPEED_EPSILON: Final = 0.01

#: ``response_format`` name → the container the decoder should expect.
_FORMATS: Final = {
    "wav": AudioFormat.WAV,
    "pcm": AudioFormat.PCM,
    "mp3": AudioFormat.COMPRESSED,
    "opus": AudioFormat.COMPRESSED,
    "aac": AudioFormat.COMPRESSED,
    "flac": AudioFormat.COMPRESSED,
}

#: Rate OpenAI's ``pcm`` answer arrives at; it carries no header, so the engine must
#: name it. WAV ignores this and reads its own header instead.
_PCM_RATE: Final = 24000


class OpenAiTtsEngine(CloudTtsEngine):
    """Any OpenAI-compatible ``/audio/speech`` service; endpoint and model from settings."""

    name: ClassVar[str] = "openai"
    title: ClassVar[str] = "OpenAI-совместимый"
    default_ref: ClassVar[str] = "openai"
    default_endpoint: ClassVar[str] = _DEFAULT_ENDPOINT
    supports_streaming: ClassVar[bool] = False
    speed_limits: ClassVar[tuple[float, float]] = _SPEED_LIMITS

    __slots__ = ()

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Read the key and open the client, then insist on a model.

        A missing model is the one misconfiguration that cannot be papered over: the
        endpoint and voice have defaults, but no service has a default model and an
        empty one is rejected on every phrase. Failing here means the router learns it
        before promising the user any sound and speaks through the local net instead.
        """
        super().load(voice, options)
        if not options.option("model"):
            self.unload()
            raise TtsError(
                f"{self.name}: model not configured",
                user_message=(
                    "Укажите модель облачного сервиса в настройках голоса — "
                    "без неё синтез не запустится."
                ),
            )

    def _build_request(
        self,
        text: str,
        speed: float,
        pitch: float,
        *,
        stream: bool,
    ) -> CloudRequest:
        """POST the phrase to ``{endpoint}/audio/speech`` in OpenAI's shape.

        ``pitch`` is accepted and dropped: the schema has no field for it. ``stream``
        is ignored — the base class already streams by sentence, which is the right
        granularity for a format that must arrive whole to be decoded.
        """
        del stream, pitch
        voice = self._require_loaded()
        audio_format, response_format = self._format()
        payload: JsonObject = {
            "model": self._options.option("model"),
            "input": text,
            "voice": voice.voice_id or _DEFAULT_VOICE,
            "response_format": response_format,
        }
        provider_speed = round(self._speed_for_provider(speed), 2)
        if abs(provider_speed - 1.0) > _SPEED_EPSILON:
            payload["speed"] = provider_speed
        rate = _PCM_RATE if audio_format is AudioFormat.PCM else 0
        return CloudRequest(
            url=f"{self._endpoint()}/audio/speech",
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
                "Accept": "application/octet-stream",
            },
            body=json.dumps(payload).encode("utf-8"),
            audio_format=audio_format,
            sample_rate=rate,
        )

    def _format(self) -> tuple[AudioFormat, str]:
        """The ``(decoder, wire name)`` for the configured ``audio_format``.

        WAV by default because it carries its own sample rate: a service that answers
        at any rate decodes correctly without the engine guessing. An unknown name
        falls back to WAV rather than sending a format no decoder here understands.
        """
        requested = self._options.option("audio_format", "wav").strip().lower()
        if requested in _FORMATS:
            return _FORMATS[requested], requested
        return AudioFormat.WAV, "wav"
