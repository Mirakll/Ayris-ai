"""Vosk offline recogniser: small models, CPU-friendly, streaming.

Vosk sits between the high-accuracy/high-cost Whisper tier and cloud APIs: its
small models (~50 MB Russian) run on a single core fast enough to keep up with
speech, they work offline, and while the quality trails Whisper's it beats every
phone-dictation baseline from five years ago.  Ayris treats Vosk as the
eco-mode or low-memory fallback: on a machine with 4 GB available, a Whisper
tiny model plus its runtime plus the LLM do not fit, and Vosk does.

**Streaming and confidence.**  A KaldiRecognizer decodes incrementally: each
block of PCM advances the lattice and may produce a partial result, and the
final result does not appear until the caller signals end-of-phrase.  That is
exactly what :attr:`~ayris.audio.stt.base.SttEngine.supports_streaming` means,
and why the manager layer above calls :meth:`start_stream` / :meth:`accept_audio`
/ :meth:`finish_stream` rather than handing in a complete buffer.  The final
result carries per-word confidences; partial results do not, so their confidence
is estimated.

**Model location.**  Models live in ``<profile>/models/stt/vosk`` or are shared
with the wake-word engine when that one is also Vosk - in which case the STT
model directory is ``<profile>/models/wake/vosk`` or ``.../kws`` or ``.../stt``,
all checked.  A model unpacked straight into the STT models folder is also
recognised by its ``am`` marker.  This overlapping search is why one downloaded
Vosk model serves both tasks.

**Resampling.**  Vosk wants exactly 16 kHz; handing it 48 kHz PCM makes it
decode silence.  The worker resamples before calling in, but a test that builds
an :class:`~ayris.audio.stt.base.AudioBuffer` directly must call
:meth:`~ayris.audio.stt.base.AudioBuffer.prepared_for` or the result will be
wrong and the reason will not be obvious.
"""

from __future__ import annotations

import json
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.stt.base import (
    DEFAULT_LANGUAGE,
    STT_SAMPLE_RATE,
    SttEngine,
    TranscriptResult,
    TranscriptSegment,
)
from ayris.core.errors import SttError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ayris.audio.stt.base import AudioBuffer, SttOptions

__all__ = ["VoskSttEngine"]

_log: Final = get_logger(__name__)

#: Confidence assigned to a partial result.  Vosk emits partials with no
#: per-word scores, and waiting for the final would make the overlay lag by the
#: length of the end-of-phrase pause.  Deliberately below 1.0 so that a caller
#: filtering on ``min_confidence`` drops partials first.
_PARTIAL_CONFIDENCE: Final = 0.65

#: Directory names looked at inside the given model folder before falling back
#: to treating the folder itself as the model.  ``kws`` and ``stt`` are here so
#: that one downloaded Vosk model can serve both the wake word and recognition.
_MODEL_DIRS: Final = ("vosk", "stt", "kws", "model")

#: Marker directory that identifies a Vosk model.  Every Vosk model contains one.
_AM_MARKER: Final = "am"


class VoskSttEngine(SttEngine):
    """Offline recogniser built on Vosk."""

    name: ClassVar[str] = "vosk"
    package: ClassVar[str] = "vosk"
    module: ClassVar[str] = "vosk"
    supports_streaming: ClassVar[bool] = True
    memory_factor: ClassVar[float] = 1.0

    __slots__ = ("_language", "_model", "_recognizer")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._recognizer: Any = None
        self._language: str = DEFAULT_LANGUAGE

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Whatever the loaded model was trained for.

        Vosk models are single-language by construction, and the model directory
        name sometimes names it - ``vosk-model-small-ru-0.22`` - but not always.
        Reporting the configured language is the best this engine can do without
        opening a manifest that may not exist.
        """
        return (self._language,) if self._language else ()

    def load(self, model_path: Path, options: SttOptions) -> None:
        """Open the Vosk model and create a recogniser.

        Args:
            model_path: Directory containing the model, or a directory
                containing one under a well-known name.
            options: Threads are ignored - Vosk decodes on one - and so is the
                GPU choice, because Vosk is CPU-only.  The language is
                remembered and reported in results.

        Raises:
            SttError: If ``vosk`` is missing, no model directory is found, or
                the directory is not a Vosk model.
        """
        module = self._import()
        directory = self._resolve_model(self._model_dir(model_path), markers=(_AM_MARKER,))

        # Vosk logs the whole model load to stderr at level 0.  A worker process
        # that writes to stderr directly ends up interleaved with everything
        # else in the pipe, and Ayris routes its own logging through the bridge.
        set_log_level = getattr(module, "SetLogLevel", None)
        if callable(set_log_level):
            set_log_level(-1)

        started = perf_counter()
        try:
            self._model = module.Model(str(directory))
        except Exception as exc:
            self._model = None
            raise SttError(
                f"vosk: cannot open model at {directory}: {exc}",
                user_message=(
                    f"Не удалось открыть модель Vosk в папке «{directory.name}». "
                    f"Скачайте модель или выберите другой движок."
                ),
            ) from exc

        try:
            self._recognizer = module.KaldiRecognizer(self._model, float(STT_SAMPLE_RATE))
        except Exception as exc:
            self._model = None
            self._recognizer = None
            raise SttError(
                f"vosk: cannot create recognizer: {exc}",
                user_message="Не удалось запустить распознавание через Vosk.",
            ) from exc

        set_words = getattr(self._recognizer, "SetWords", None)
        if callable(set_words):
            # Per-word confidences and timings.  Without them a result carries
            # no segments at all, and section 6's word highlighting has nothing
            # to highlight.
            set_words(True)

        self._language = options.language or DEFAULT_LANGUAGE
        self._options = options
        self._model_path = directory
        _log.info(
            "vosk: модель %s загружена за %.0f мс, язык %s",
            directory.name,
            (perf_counter() - started) * 1000.0,
            self._language,
        )

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise a complete buffer.

        Not the usual path for Vosk - :meth:`start_stream` through
        :meth:`finish_stream` is - but the worker uses it for a phrase the
        segmenter already closed, and a test can feed a WAV file in without
        managing the three-call sequence.

        Raises:
            SttError: If no model is loaded or the decoder fails.
        """
        options = self._require_loaded()
        prepared = self._prepare(audio)
        if prepared.is_silent() or prepared.duration_ms < options.min_speech_ms:
            return TranscriptResult.empty(
                engine=self.name,
                language=self._language,
                duration_ms=prepared.duration_ms,
                model=self.model_name,
            )
        started = perf_counter()
        self.start_stream()
        self.accept_audio(prepared.pcm)
        result = self.finish_stream()
        return result.with_timing(
            inference_ms=(perf_counter() - started) * 1000.0,
            duration_ms=prepared.duration_ms,
        )

    def unload(self) -> None:
        """Drop the recogniser and the model.  Safe to call twice."""
        self._recognizer = None
        self._model = None
        self._language = DEFAULT_LANGUAGE
        self._options = None
        self._model_path = None

    def start_stream(self) -> None:
        """Begin an incremental recognition."""
        self._require_loaded()
        recognizer = self._recognizer
        if recognizer is None:
            return
        reset = getattr(recognizer, "Reset", None)
        if callable(reset):
            reset()

    def accept_audio(self, pcm: bytes) -> str:
        """Feed the next chunk of PCM and return the partial text so far.

        Args:
            pcm: Mono int16 at 16 kHz.  Any length.

        Returns:
            The latest partial hypothesis, which may be empty or may repeat what
            the previous call returned.  Only :meth:`finish_stream` is final.

        Raises:
            SttError: If no model is loaded or the decoder fails.
        """
        self._require_loaded()
        recognizer = self._recognizer
        if recognizer is None:
            return ""
        try:
            recognizer.AcceptWaveform(pcm)
            payload = recognizer.PartialResult()
        except Exception as exc:
            raise SttError(
                f"vosk: decoding failed: {exc}",
                user_message="Ошибка распознавания речи.",
            ) from exc
        parsed = self._parse(payload)
        return str(parsed.get("partial", "")).strip()

    def finish_stream(self) -> TranscriptResult:
        """Close the stream and return the final transcript.

        Raises:
            SttError: If no model is loaded or the final result cannot be read.
        """
        self._require_loaded()
        recognizer = self._recognizer
        if recognizer is None:
            return TranscriptResult.empty(
                engine=self.name, language=self._language, model=self.model_name
            )
        try:
            payload = recognizer.FinalResult()
        except Exception as exc:
            raise SttError(
                f"vosk: cannot read final result: {exc}",
                user_message="Ошибка получения результата распознавания.",
            ) from exc
        parsed = self._parse(payload)
        text = str(parsed.get("text", "")).strip()
        if not text:
            return TranscriptResult.empty(
                engine=self.name, language=self._language, model=self.model_name
            )

        words = parsed.get("result")
        segments: tuple[TranscriptSegment, ...] = ()
        confidence = _PARTIAL_CONFIDENCE
        if isinstance(words, list) and words:
            segments = self._segments(words)
            confidence = self._confidence(words)

        return TranscriptResult(
            text=text,
            confidence=confidence,
            segments=segments,
            language=self._language,
            engine=self.name,
            model=self.model_name,
        )

    @staticmethod
    def _parse(payload: str) -> Mapping[str, Any]:
        """Decode one of Vosk's JSON blobs, tolerating a malformed one."""
        try:
            parsed: Any = json.loads(payload)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _segments(words: list[Any]) -> tuple[TranscriptSegment, ...]:
        """Per-word segments from a final result's ``result`` array."""
        built: list[TranscriptSegment] = []
        for item in words:
            if not isinstance(item, dict):
                continue
            word = str(item.get("word", "")).strip()
            if not word:
                continue
            start = item.get("start", 0.0)
            end = item.get("end", 0.0)
            conf = item.get("conf", 0.0)
            if not isinstance(start, int | float):
                start = 0.0
            if not isinstance(end, int | float):
                end = 0.0
            if not isinstance(conf, int | float):
                conf = 0.0
            built.append(
                TranscriptSegment(
                    text=word,
                    start_ms=float(start) * 1000.0,
                    end_ms=float(end) * 1000.0,
                    confidence=max(0.0, min(1.0, float(conf))),
                )
            )
        return tuple(built)

    @staticmethod
    def _confidence(words: list[Any]) -> float:
        """Lowest per-word confidence in a final result.

        The weakest word is the right summary for a phrase: one that was half-
        guessed is not fully heard, however confident the rest was.
        """
        values = [
            float(word["conf"])
            for word in words
            if isinstance(word, dict) and isinstance(word.get("conf"), int | float)
        ]
        if not values:
            return _PARTIAL_CONFIDENCE
        return max(0.0, min(1.0, min(values)))

    @staticmethod
    def _model_dir(model_path: Path) -> Path:
        """The actual model directory inside what the settings named.

        ``voice.stt.offline_model`` names a folder under ``models/stt``, and what
        is inside it depends on how the archive was unpacked: sometimes the model
        directly, sometimes one level down.  Both are accepted rather than making
        the user reorganise files after a download.  A path that is already a
        model, or that holds none of the known names, is returned unchanged so
        that :meth:`_resolve_model` produces the error instead of this.
        """
        if not model_path.is_dir() or (model_path / _AM_MARKER).is_dir():
            return model_path
        for name in _MODEL_DIRS:
            candidate = model_path / name
            if (candidate / _AM_MARKER).is_dir():
                return candidate
        return model_path
