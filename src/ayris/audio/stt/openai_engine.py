"""OpenAI Whisper through the transcriptions API.

The most accurate of the four on Russian speech, and the slowest: the whole
buffer is uploaded and transcribed in one pass, so latency scales with the length
of the phrase rather than arriving as you speak. For a short command that is
still well inside the deadline.

**The request is multipart, and it is built by hand.** httpx can assemble
``files=`` itself, but the base class hands the transport a body that is already
bytes - that is what keeps audio out of any log line and out of a second
serialisation pass. One file part and a few text parts is little enough
machinery to be worth writing out.

**Confidence has to be derived.** Whisper reports ``avg_logprob`` per segment -
the mean log probability of the tokens - and no confidence field at all.
``exp(avg_logprob)`` turns that back into a probability, which is not the same
quantity another provider calls confidence but ranks hypotheses the same way.
``no_speech_prob`` is left alone: the segmenter upstream already decided there
was speech, and second-guessing it here would drop quiet but real phrases.
"""

from __future__ import annotations

from math import exp
from secrets import token_hex
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.stt.base import TranscriptResult, TranscriptSegment
from ayris.audio.stt.cloud_base import (
    ASSUMED_CONFIDENCE,
    CloudSttEngine,
    _Request,
    as_wav,
)
from ayris.core.errors import SttError

if TYPE_CHECKING:
    from ayris.audio.stt.base import AudioBuffer, SttOptions
    from ayris.core.models import JsonObject

__all__ = ["OpenAiSttEngine"]

_ENDPOINT: Final = "https://api.openai.com/v1/audio/transcriptions"
_DEFAULT_MODEL: Final = "whisper-1"
_FILENAME: Final = "speech.wav"


def _multipart(fields: dict[str, str], audio: bytes) -> tuple[str, bytes]:
    """Assemble a ``multipart/form-data`` body.

    Returns:
        The content type, boundary included, and the encoded body. The boundary
        is random per request so it cannot collide with the WAV bytes.
    """
    boundary = f"----AyrisBoundary{token_hex(16)}"
    marker = f"--{boundary}\r\n".encode("ascii")
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(marker)
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(value.encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(marker)
    parts.append(
        f'Content-Disposition: form-data; name="file"; filename="{_FILENAME}"\r\n'.encode("ascii")
    )
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio)
    parts.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def _confidence(avg_logprob: object) -> float:
    """A 0-1 score from a mean log probability."""
    if isinstance(avg_logprob, bool) or not isinstance(avg_logprob, int | float):
        return 0.0
    return max(0.0, min(1.0, exp(float(avg_logprob))))


class OpenAiSttEngine(CloudSttEngine):
    """Speech recognition through the OpenAI Whisper API."""

    name: ClassVar[str] = "openai"
    title: ClassVar[str] = "OpenAI Whisper"
    default_ref: ClassVar[str] = "openai"
    default_endpoint: ClassVar[str] = _ENDPOINT

    __slots__ = ()

    def _build_request(self, audio: AudioBuffer, options: SttOptions) -> _Request:
        """POST the WAV file and the model name as multipart form data."""
        fields = {
            "model": options.option("model") or _DEFAULT_MODEL,
            # verbose_json is what carries the segments and their logprobs; plain
            # json would give the text and nothing to score it with.
            "response_format": "verbose_json",
            # Zero temperature: a command is not creative writing, and Whisper
            # left to itself will happily invent a plausible ending.
            "temperature": "0",
        }
        if options.language:
            # A two-letter code, unlike every other provider here. Passing it
            # saves Whisper the language-detection pass and stops it answering a
            # Russian phrase in transliterated English.
            fields["language"] = options.language
        prompt = options.option("prompt")
        if prompt:
            fields["prompt"] = prompt

        content_type, body = _multipart(fields, as_wav(audio))
        headers = {
            "Authorization": f"Bearer {self._credential}",
            "Content-Type": content_type,
        }
        organisation = options.option("organization")
        if organisation:
            headers["OpenAI-Organization"] = organisation
        return _Request(url=self._endpoint(options), headers=headers, body=body)

    def _parse_response(self, data: bytes, options: SttOptions) -> TranscriptResult:
        """Read ``text`` and the ``segments`` that back it."""
        payload = self._decode_json(data)
        model = options.option("model") or _DEFAULT_MODEL
        raw_text = payload.get("text")
        if raw_text is None:
            raise SttError(
                f"{self.name}: response has no 'text' field: {sorted(payload)}",
                user_message=f"{self._provider_name} вернул ответ в неизвестном формате.",
            )
        text = str(raw_text).strip()
        # Whisper is asked for the language and reports back the one it used,
        # which can differ when detection was left on.
        detected = payload.get("language")
        language = detected if isinstance(detected, str) and detected else options.language
        if not text:
            return TranscriptResult.empty(
                engine=self.name,
                language=language,
                device="cloud",
                model=model,
            )
        segments = self._segments(payload)
        scores = [segment.confidence for segment in segments if segment.confidence > 0.0]
        return self._result(
            text=text,
            # No segments at all is what ``response_format=json`` answers, and a
            # future API version may drop the logprobs from the verbose form too.
            # Scoring that zero would let ``min_confidence`` throw away a
            # transcript Whisper was perfectly happy with.
            confidence=sum(scores) / len(scores) if scores else ASSUMED_CONFIDENCE,
            segments=segments,
            language=language,
            model=model,
        )

    @staticmethod
    def _segments(payload: JsonObject) -> tuple[TranscriptSegment, ...]:
        """Clause-level segments with times in milliseconds.

        Whisper reports seconds as floats. An entry without usable text is
        skipped rather than kept as an empty interval.
        """
        raw = payload.get("segments")
        if not isinstance(raw, list):
            return ()
        segments: list[TranscriptSegment] = []
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get("text", "")).strip()
            if not text:
                continue
            start = entry.get("start")
            end = entry.get("end")
            start_ms = float(start) * 1000.0 if isinstance(start, int | float) else 0.0
            end_ms = float(end) * 1000.0 if isinstance(end, int | float) else start_ms
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=start_ms,
                    end_ms=max(start_ms, end_ms),
                    confidence=_confidence(entry.get("avg_logprob")),
                )
            )
        return tuple(segments)
