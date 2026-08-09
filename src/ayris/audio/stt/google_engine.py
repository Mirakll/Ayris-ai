"""Google Cloud Speech-to-Text, synchronous recognition.

The most talkative of the four: it reports a per-alternative confidence and, when
asked, the start and end of every word, so this is the one engine whose
:class:`~ayris.audio.stt.base.TranscriptSegment` list is real rather than empty.

**The audio travels inside the JSON.** ``speech:recognize`` takes base64 in
``audio.content``, which costs a third of the payload in encoding overhead and
caps the request at about a minute of speech. That is the right trade for a
spoken command; anything longer belongs in ``longrunningrecognize``, which
answers with an operation name and needs polling - a different task.

**Auth is an API key in a header, not in the query string.** Both work, but a
key in a URL ends up in every proxy log on the way, and :meth:`_scrub_headers`
in the base class can only mask a header.

**Word times come back as strings** like ``"1.100s"``, not numbers, so they are
parsed rather than cast.
"""

from __future__ import annotations

import json
from base64 import b64encode
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.stt.base import TranscriptResult, TranscriptSegment
from ayris.audio.stt.cloud_base import ASSUMED_CONFIDENCE, CloudSttEngine, _Request
from ayris.core.errors import SttError

if TYPE_CHECKING:
    from ayris.audio.stt.base import AudioBuffer, SttOptions
    from ayris.core.models import JsonObject

__all__ = ["GoogleSttEngine"]

_ENDPOINT: Final = "https://speech.googleapis.com/v1/speech:recognize"

#: Google wants a BCP-47 tag, and rejects a bare two-letter code.
_LOCALES: Final = {
    "ru": "ru-RU",
    "en": "en-US",
    "tr": "tr-TR",
    "kk": "kk-KZ",
    "uz": "uz-UZ",
}


def _duration_ms(raw: object) -> float:
    """Milliseconds out of a protobuf duration string such as ``"1.100s"``.

    Anything unparseable becomes 0.0: a missing word offset is worth less than
    the transcript it belongs to, and is not a reason to fail the recognition.
    """
    if isinstance(raw, int | float) and not isinstance(raw, bool):
        return float(raw) * 1000.0
    if not isinstance(raw, str):
        return 0.0
    try:
        return float(raw.removesuffix("s")) * 1000.0
    except ValueError:
        return 0.0


class GoogleSttEngine(CloudSttEngine):
    """Speech recognition through Google Cloud Speech-to-Text."""

    name: ClassVar[str] = "google"
    title: ClassVar[str] = "Google Cloud Speech-to-Text"
    default_ref: ClassVar[str] = "google"
    default_endpoint: ClassVar[str] = _ENDPOINT

    __slots__ = ()

    def _build_request(self, audio: AudioBuffer, options: SttOptions) -> _Request:
        """POST the recognition config and the base64 audio as one JSON object."""
        config: dict[str, object] = {
            "encoding": "LINEAR16",
            "sampleRateHertz": self.sample_rate,
            "audioChannelCount": 1,
            "languageCode": _LOCALES.get(options.language, options.language or "ru-RU"),
            "enableAutomaticPunctuation": options.punctuation,
            "enableWordTimeOffsets": True,
            "profanityFilter": False,
            "maxAlternatives": 1,
        }
        model = options.option("model")
        if model:
            config["model"] = model
        body = json.dumps(
            {"config": config, "audio": {"content": b64encode(audio.pcm).decode("ascii")}}
        ).encode("utf-8")
        headers = {
            "X-Goog-Api-Key": self._credential,
            "Content-Type": "application/json",
        }
        return _Request(url=self._endpoint(options), headers=headers, body=body)

    def _parse_response(self, data: bytes, options: SttOptions) -> TranscriptResult:
        """Join the alternatives across ``results`` into one transcript.

        Google splits a long utterance into several ``results``, each with its own
        best alternative, and expects the caller to concatenate them. An answer
        with no ``results`` at all is how it says it heard nothing.
        """
        payload = self._decode_json(data)
        model = options.option("model", "default")
        raw_results = payload.get("results")
        if raw_results is None:
            return TranscriptResult.empty(
                engine=self.name,
                language=options.language,
                device="cloud",
                model=model,
            )
        if not isinstance(raw_results, list):
            raise SttError(
                f"{self.name}: 'results' is {type(raw_results).__name__}, not a list",
                user_message=f"{self._provider_name} вернул ответ в неизвестном формате.",
            )

        parts: list[str] = []
        segments: list[TranscriptSegment] = []
        confidences: list[float] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            best = self._best_alternative(entry)
            if best is None:
                continue
            text = str(best.get("transcript", "")).strip()
            if not text:
                continue
            parts.append(text)
            confidence = best.get("confidence")
            if isinstance(confidence, int | float) and not isinstance(confidence, bool):
                confidences.append(float(confidence))
            segments.extend(self._word_segments(best))

        if not parts:
            return TranscriptResult.empty(
                engine=self.name,
                language=options.language,
                device="cloud",
                model=model,
            )
        return self._result(
            text=" ".join(parts),
            # No confidence at all is possible even on a 200: the short-audio
            # path sometimes omits it. Averaging what came back keeps a partial
            # answer usable instead of scoring the whole phrase zero.
            confidence=(sum(confidences) / len(confidences) if confidences else ASSUMED_CONFIDENCE),
            segments=tuple(segments),
            language=options.language,
            model=model,
        )

    @staticmethod
    def _best_alternative(entry: JsonObject) -> JsonObject | None:
        """The first alternative of one result, or ``None`` if there is none.

        ``maxAlternatives`` is 1, so the first is the best; the field is still a
        list because the API allows more.
        """
        alternatives = entry.get("alternatives")
        if not isinstance(alternatives, list):
            return None
        for alternative in alternatives:
            if isinstance(alternative, dict):
                return alternative
        return None

    @staticmethod
    def _word_segments(alternative: JsonObject) -> list[TranscriptSegment]:
        """Word timings, when ``enableWordTimeOffsets`` produced any."""
        words = alternative.get("words")
        if not isinstance(words, list):
            return []
        segments: list[TranscriptSegment] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            start = _duration_ms(word.get("startTime"))
            end = _duration_ms(word.get("endTime"))
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=start,
                    # Google rounds both ends to 100 ms, and a very short word
                    # can come back with end == start; clamping keeps the
                    # interval from running backwards, which the segment forbids.
                    end_ms=max(start, end),
                )
            )
        return segments
