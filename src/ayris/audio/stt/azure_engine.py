"""Azure Speech, short-audio REST recognition.

**The region is part of the hostname, so it is not optional.** Azure has no
global endpoint for this API: a key issued for ``westeurope`` only works against
``westeurope.stt.speech.microsoft.com``, and pointing it at another region
answers 401 - which reads exactly like a bad key and sends the user looking in
the wrong place. So a missing region is caught here, and the built URL is
overridable through the ``endpoint`` option for a private or sovereign cloud.

**It wants a container, not raw samples.** The body is a WAV file, which
:func:`~ayris.audio.stt.cloud_base.as_wav` wraps around the PCM buffer; the
header is 44 bytes and carries the sample rate Azure would otherwise have to be
told.

**``format=detailed`` is what makes the answer usable.** The simple format
returns text only; detailed adds ``NBest`` with a confidence per hypothesis,
plus the punctuated ``Display`` form next to the raw ``Lexical`` one. The status
lives in its own field: ``RecognitionStatus`` is ``NoMatch`` for "heard nothing"
on a perfectly successful 200.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Final
from urllib.parse import urlencode

from ayris.audio.stt.base import TranscriptResult
from ayris.audio.stt.cloud_base import ASSUMED_CONFIDENCE, CloudSttEngine, _Request, as_wav
from ayris.core.errors import SttError

if TYPE_CHECKING:
    from ayris.audio.stt.base import AudioBuffer, SttOptions
    from ayris.core.models import JsonObject

__all__ = ["AzureSttEngine"]

_HOST_TEMPLATE: Final = "https://{region}.stt.speech.microsoft.com"
_PATH: Final = "/speech/recognition/conversation/cognitiveservices/v1"

_LOCALES: Final = {
    "ru": "ru-RU",
    "en": "en-US",
    "tr": "tr-TR",
    "kk": "kk-KZ",
    "uz": "uz-UZ",
}

#: Statuses that mean "the request worked, there was just nothing to hear".
#: ``InitialSilenceTimeout`` and ``BabbleTimeout`` arrive with HTTP 200 and are
#: the normal answer to a mis-fired wake word.
_EMPTY_STATUSES: Final = frozenset(
    {"NoMatch", "InitialSilenceTimeout", "BabbleTimeout", "EndOfDictation"}
)


class AzureSttEngine(CloudSttEngine):
    """Speech recognition through Azure Cognitive Services Speech."""

    name: ClassVar[str] = "azure"
    title: ClassVar[str] = "Azure Speech"
    default_ref: ClassVar[str] = "azure"

    __slots__ = ()

    def _build_request(self, audio: AudioBuffer, options: SttOptions) -> _Request:
        """POST a WAV body to the region's recognition endpoint.

        Raises:
            SttError: Neither ``endpoint`` nor ``region`` is configured, so there
                is no host to send to.
        """
        query = {
            "language": _LOCALES.get(options.language, options.language or "ru-RU"),
            "format": "detailed",
            "profanity": "raw",
        }
        url = f"{self._base_url(options)}{_PATH}?{urlencode(query)}"
        headers = {
            "Ocp-Apim-Subscription-Key": self._credential,
            "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={self.sample_rate}",
            "Accept": "application/json",
        }
        return _Request(url=url, headers=headers, body=as_wav(audio))

    def _base_url(self, options: SttOptions) -> str:
        """Explicit endpoint if set, otherwise the host built from the region."""
        endpoint = options.option("endpoint")
        if endpoint:
            return endpoint.rstrip("/")
        region = options.option("region")
        if not region:
            raise SttError(
                f"{self.name}: neither endpoint nor region is configured",
                user_message=(
                    "Для Azure Speech нужно указать регион (например, westeurope) "
                    "в настройках голоса: у сервиса нет общего адреса."
                ),
            )
        return _HOST_TEMPLATE.format(region=region.strip().lower())

    def _parse_response(self, data: bytes, options: SttOptions) -> TranscriptResult:
        """Read ``RecognitionStatus``, then the best ``NBest`` hypothesis."""
        payload = self._decode_json(data)
        model = options.option("model", "conversation")
        status = str(payload.get("RecognitionStatus", "")).strip()
        if status in _EMPTY_STATUSES:
            return TranscriptResult.empty(
                engine=self.name,
                language=options.language,
                device="cloud",
                model=model,
            )
        if status and status != "Success":
            raise SttError(
                f"{self.name}: RecognitionStatus={status!r}",
                user_message=(
                    f"{self._provider_name} не смог распознать запись (статус {status})."
                ),
            )

        best = self._best_hypothesis(payload)
        text = self._display_text(payload, best, punctuation=options.punctuation)
        if not text:
            return TranscriptResult.empty(
                engine=self.name,
                language=options.language,
                device="cloud",
                model=model,
            )
        confidence = ASSUMED_CONFIDENCE
        if best is not None:
            raw = best.get("Confidence")
            if isinstance(raw, int | float) and not isinstance(raw, bool):
                confidence = float(raw)
        return self._result(
            text=text,
            confidence=confidence,
            language=options.language,
            model=model,
        )

    @staticmethod
    def _best_hypothesis(payload: JsonObject) -> JsonObject | None:
        """First entry of ``NBest``, which Azure returns already sorted."""
        candidates = payload.get("NBest")
        if not isinstance(candidates, list):
            return None
        for candidate in candidates:
            if isinstance(candidate, dict):
                return candidate
        return None

    @staticmethod
    def _display_text(payload: JsonObject, best: JsonObject | None, *, punctuation: bool) -> str:
        """The transcript, in the form the punctuation setting asks for.

        ``Lexical`` is lower-case and unpunctuated, ``Display`` is capitalised
        and punctuated, and ``DisplayText`` at the top level is the same string
        as the best hypothesis' ``Display``. Falling through the three keeps the
        engine working if a future API version drops one of them.
        """
        keys = ("Display", "ITN", "Lexical") if punctuation else ("Lexical", "Display", "ITN")
        if best is not None:
            for key in keys:
                value = best.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        top = payload.get("DisplayText")
        return top.strip() if isinstance(top, str) else ""
