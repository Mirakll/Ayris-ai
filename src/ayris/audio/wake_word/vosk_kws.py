"""Vosk keyword spotting: the fallback that reuses a model already on disk.

This is not a wake word engine in the sense the other two are.  Vosk is a
speech recogniser, and what it does here is recognise continuously against a
grammar containing nothing but the wake word variants and ``[unk]`` - so the
only thing it can ever output is one of those variants or "something else".
Restricting the grammar this way makes a general recogniser behave like a
spotter: the search space collapses, the decoder gets fast enough to run on
every frame, and words outside the list are absorbed by ``[unk]`` instead of
being forced onto the nearest phrase.

Why keep it at all.  Ayris already ships a Vosk model for offline
speech-to-text, so this engine costs no extra download, no account and no
training - and it accepts any phrase the user types, including one nobody has
ever trained a model for.  That is exactly what a user gets when openWakeWord
has no model for their chosen word.

Why it is not the default.  A recogniser scores a whole word after hearing it,
so a hit arrives a beat later than openWakeWord's, it burns noticeably more CPU
running all the time, and its confidences are coarse - a phone-level average
rather than a calibrated probability.  The sensitivity slider still works, but
it has fewer usable positions than it does on the other engines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.wake_word.base import (
    MIN_THRESHOLD,
    ModelSpec,
    WakeDetection,
    WakeWordEngine,
    normalise_phrase,
)
from ayris.core.errors import WakeWordError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["FRAME_SAMPLES", "VoskKwsEngine"]

_log: Final = get_logger(__name__)

#: 80 ms at 16 kHz.  Vosk accepts any size; this one matches openWakeWord so
#: that switching engines does not change the manager's framing behaviour.
FRAME_SAMPLES: Final = 1280

#: The grammar token that swallows everything that is not a wake word.  Without
#: it the decoder has to map every noise onto some phrase from the list, and the
#: engine fires on coughs.
_UNKNOWN: Final = "[unk]"

#: Confidence assigned to a phrase seen in a partial result.  Partials carry no
#: per-word confidence, and waiting for the final result would add the length of
#: an end-of-phrase pause to every activation.  Deliberately below 1.0 so that a
#: user who drags sensitivity to the bottom stops trusting partials first.
_PARTIAL_CONFIDENCE: Final = 0.7

#: Directory names looked at inside the wake model folder before giving up.
_MODEL_DIRS: Final = ("vosk", "kws", "stt")


class VoskKwsEngine(WakeWordEngine):
    """Keyword spotter built from a Vosk recogniser under a closed grammar."""

    name: ClassVar[str] = "vosk"
    package: ClassVar[str] = "vosk"
    module: ClassVar[str] = "vosk"

    __slots__ = ("_model", "_recognizer", "_texts")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._recognizer: Any = None
        self._texts: frozenset[str] = frozenset()

    @property
    def frame_samples(self) -> int:
        """80 ms, matching the other engines."""
        return FRAME_SAMPLES

    def load(self, spec: ModelSpec) -> None:
        """Open the Vosk model and arm a recogniser restricted to the phrases.

        Raises:
            WakeWordError: If ``vosk`` is missing, no model directory is found,
                or the model cannot be opened.
        """
        phrases = self._require_phrases(spec)
        module = self._import()
        directory = self._model_dir(spec)

        # Vosk logs the whole model load to stderr at level 0.  Ayris routes its
        # own logging through utils.logger, and a worker process that writes to
        # stderr directly ends up interleaved with everything else in the pipe.
        set_log_level = getattr(module, "SetLogLevel", None)
        if callable(set_log_level):
            set_log_level(-1)

        try:
            self._model = module.Model(str(directory))
        except Exception as exc:
            self._model = None
            raise WakeWordError(
                f"vosk: cannot open the model at {directory}: {exc}",
                user_message=(
                    f"Не удалось открыть модель Vosk в папке {directory}. "
                    f"Скачайте модель распознавания речи или выберите другой движок."
                ),
            ) from exc

        self._texts = frozenset(phrase.text for phrase in phrases)
        self._recognizer = self._build_recognizer(module, sorted(self._texts))
        self._spec = spec
        _log.info("vosk kws loaded from %s with %d phrase(s)", directory, len(self._texts))

    def process(self, frame: bytes) -> WakeDetection | None:
        """Feed one frame to the decoder and look at what it is hearing.

        Final results are preferred - they carry per-word confidences - but the
        partial is checked on every frame so that activation does not wait for
        the end-of-phrase pause.

        Raises:
            WakeWordError: If no model is loaded or the frame is the wrong size.
        """
        self._require_loaded()
        self._check_frame(frame)
        recognizer = self._recognizer
        try:
            final = bool(recognizer.AcceptWaveform(frame))
            payload = recognizer.Result() if final else recognizer.PartialResult()
        except Exception as exc:
            raise WakeWordError(
                f"vosk: decoding failed: {exc}",
                user_message="Ошибка распознавания слова активации.",
            ) from exc

        parsed = self._parse(payload)
        text = normalise_phrase(str(parsed.get("text") or parsed.get("partial") or ""))
        if not text:
            return None
        phrase = self._match(text)
        if phrase is None:
            return None

        score = self._confidence(parsed) if final else _PARTIAL_CONFIDENCE
        if score < MIN_THRESHOLD:
            return None
        if final:
            # The decoder would otherwise start the next phrase carrying the
            # tail of this one, and a second frame of the same utterance would
            # be reported all over again.
            self.reset()
        return WakeDetection(
            phrase=phrase,
            score=score,
            engine=self.name,
            scores={phrase: score},
        )

    def unload(self) -> None:
        """Drop the recogniser and the model.  Safe to call twice."""
        self._recognizer = None
        self._model = None
        self._texts = frozenset()
        self._spec = None

    def reset(self) -> None:
        """Clear the decoder state without reopening the model."""
        recognizer = self._recognizer
        if recognizer is None:
            return
        reset = getattr(recognizer, "Reset", None)
        if callable(reset):
            reset()

    # ------------------------------------------------------------------ helpers

    def _build_recognizer(self, module: Any, texts: Sequence[str]) -> Any:
        """Create a recogniser locked to the phrase grammar.

        Falls back to an unrestricted recogniser on a build without grammar
        support: matching still works, it is only less accurate, and refusing to
        start would be a worse answer than a slightly noisier spotter.

        Raises:
            WakeWordError: If even the plain recogniser cannot be created.
        """
        grammar = json.dumps([*texts, _UNKNOWN], ensure_ascii=False)
        try:
            recognizer = module.KaldiRecognizer(self._model, float(self.sample_rate), grammar)
        except Exception as exc:
            _log.warning("vosk: grammar mode unavailable (%s), falling back to free decoding", exc)
            try:
                recognizer = module.KaldiRecognizer(self._model, float(self.sample_rate))
            except Exception as inner:
                raise WakeWordError(
                    f"vosk: cannot create a recognizer: {inner}",
                    user_message="Не удалось запустить распознавание слова активации через Vosk.",
                ) from inner
        set_words = getattr(recognizer, "SetWords", None)
        if callable(set_words):
            # Per-word confidences: without them a final result scores the same
            # as a partial and the sensitivity slider stops meaning anything.
            set_words(True)
        return recognizer

    def _match(self, text: str) -> str | None:
        """The configured variant this text represents, if any.

        Exact match first.  Then containment, because the decoder emits the
        whole utterance and a user who says "айрис, открой браузер" has still
        said the wake word.
        """
        if text in self._texts:
            return text
        for phrase in self._texts:
            if phrase in text:
                return phrase
        return None

    @staticmethod
    def _parse(payload: str) -> Mapping[str, Any]:
        """Decode one of Vosk's JSON blobs, tolerating a malformed one."""
        try:
            parsed: Any = json.loads(payload)
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _confidence(parsed: Mapping[str, Any]) -> float:
        """Lowest per-word confidence in a final result.

        The weakest word is the right summary for a wake word: a phrase whose
        first syllable was guessed is not a phrase that was heard, however
        confident the decoder was about the rest.
        """
        words = parsed.get("result")
        if not isinstance(words, list) or not words:
            return _PARTIAL_CONFIDENCE
        values = [
            float(word["conf"])
            for word in words
            if isinstance(word, dict) and isinstance(word.get("conf"), int | float)
        ]
        if not values:
            return _PARTIAL_CONFIDENCE
        return max(0.0, min(1.0, min(values)))

    def _model_dir(self, spec: ModelSpec) -> Path:
        """Locate the Vosk model directory.

        Raises:
            WakeWordError: If none of the usual places holds one.
        """
        explicit = spec.option("model_path")
        if explicit:
            candidate = Path(explicit)
            if candidate.is_dir():
                return candidate
            raise WakeWordError(
                f"vosk: {explicit} is not a model directory",
                user_message=f"Папка модели Vosk «{explicit}» не найдена.",
            )
        base = spec.models_dir
        if base is not None:
            for name in _MODEL_DIRS:
                candidate = base / name
                if candidate.is_dir():
                    return candidate
            # A model unpacked straight into the wake folder: Vosk models always
            # contain an "am" directory, which is a cheap way to recognise one.
            if (base / "am").is_dir():
                return base
        where = str(base) if base is not None else "папке моделей"
        raise WakeWordError(
            f"vosk: no model directory under {where}",
            user_message=(
                f"Модель Vosk не найдена в {where}. "
                f"Скачайте модель распознавания речи или выберите другой движок."
            ),
        )
