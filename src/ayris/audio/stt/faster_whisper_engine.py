"""faster-whisper: the accurate tier, on the GPU when there is one.

Whisper is what makes Ayris understand a sentence rather than a command.  It
punctuates, it survives an accent and background noise, and on Russian it is in
a different class from a small Kaldi model.  It also costs about twenty times
the memory and, on a CPU, more time than the phrase took to say - which is why
this engine exists twice over: once as a CUDA path that answers in a few hundred
milliseconds, and once as an ``int8`` CPU path that keeps working on a laptop
with integrated graphics.

**The device decision is made here, once, at load time.**  ``performance.gpu``
offers ``auto``, ``cuda`` and ``cpu``.  ``auto`` means: ask CTranslate2 how many
CUDA devices it can see, and if the answer is more than zero, try to load onto
one.  Everything about that probe is defensive.  A machine with a driver that
does not match its runtime raises from inside a C++ library; one with an
old card returns a device that then fails on the first ``float16`` kernel; one
where another process took the memory raises ``cudaErrorMemoryAllocation`` two
seconds into the load.  All three end in the same place - CPU, ``int8``, and a
line in the log saying which of them happened - because a voice assistant that
refuses to start because of a graphics driver is a broken voice assistant.
:attr:`FasterWhisperEngine.fallback_reason` keeps that sentence for DevTools and
for the log the user attaches to a bug report.

**Hallucination is the other half of the work.**  Whisper was trained on
subtitles, and when it is handed silence it does not return an empty string, it
returns the credits: "Продолжение следует...", "Субтитры сделал DimaTorzok".  A
voice assistant that acts on those is worse than one that mishears.  Three
filters run in order: a buffer shorter than
:attr:`~ayris.audio.stt.base.SttOptions.min_speech_ms` or quieter than
:data:`~ayris.audio.stt.base.SILENCE_DBFS` never reaches the model at all; a
segment whose ``no_speech_prob`` is above the configured threshold is dropped;
and what remains is checked against :data:`_HALLUCINATIONS`.  The first filter
is the one that matters most, because it is also the cheapest.

**The model is a directory, not a name.**  ``faster_whisper.WhisperModel``
happily accepts ``"small"`` and downloads a gigabyte from Hugging Face, which is
not something a desktop assistant should do behind the user's back - section 14
puts model downloads in the model manager, with a progress bar and a cancel
button.  So this engine only ever loads from a path, and a settings value that
names a size instead of a folder produces an error telling the user to download
it first.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.stt.base import (
    DEFAULT_LANGUAGE,
    SttEngine,
    TranscriptResult,
    TranscriptSegment,
)
from ayris.core.errors import SttError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ayris.audio.stt.base import AudioBuffer, SttOptions

__all__ = ["CPU_COMPUTE_TYPE", "CUDA_COMPUTE_TYPE", "FasterWhisperEngine", "cuda_available"]

_log: Final = get_logger(__name__)

#: What a CTranslate2 model directory always contains.  Used to tell a real
#: model from a folder that holds a Vosk model or a half-finished download.
_MODEL_MARKERS: Final = ("model.bin",)

#: Quantisation on the GPU.  ``float16`` is the point of having one: half the
#: memory of ``float32`` and roughly twice the throughput on any card with
#: tensor cores, with no measurable accuracy loss on speech.
CUDA_COMPUTE_TYPE: Final = "float16"

#: Quantisation on the CPU.  ``int8`` is two to three times faster than
#: ``float32`` on the same core and fits a ``small`` model into about 500 MB.
#: The accuracy cost on Russian is small; being slower than real time is not.
CPU_COMPUTE_TYPE: Final = "int8"

#: Whisper's own languages, abridged to what Ayris might plausibly be asked for.
#: The full list is ninety-nine entries and lives in the vendor package; this one
#: exists so the settings window can answer without importing it.
_LANGUAGES: Final = ("ru", "en", "uk", "be", "kk", "de", "fr", "es", "it", "pl", "tr", "zh")

#: Phrases Whisper produces when it hears nothing.  Matched case-insensitively
#: against the whole transcript, not against individual segments: "спасибо за
#: просмотр" is a real thing a user may say, but never as the entire utterance
#: after a wake word.  Kept short on purpose - an over-eager list would start
#: swallowing real commands, and the duration and no-speech filters catch most
#: of these first.
_HALLUCINATIONS: Final = frozenset(
    {
        "продолжение следует...",
        "продолжение следует",
        "субтитры сделал dimatorzok",
        "субтитры создавал dimatorzok",
        "редактор субтитров а.синецкая корректор а.егорова",
        "спасибо за просмотр!",
        "спасибо за внимание!",
        "подписывайтесь на канал",
        "thank you for watching",
        "thanks for watching!",
        "you",
        "продолжение в следующей серии",
    }
)

#: Segments shorter than this are dropped even when Whisper was confident.  A
#: real word takes longer than 80 ms to say; anything below is a click or a
#: breath that the model attached a token to.
_MIN_SEGMENT_MS: Final = 80.0


def cuda_available() -> tuple[bool, str]:
    """Whether a CUDA device can be used, and why not when it cannot.

    Asks CTranslate2 rather than torch: faster-whisper runs on CTranslate2, and
    a machine can have a working torch CUDA build while CTranslate2 was compiled
    without one - or, more often, the other way round, since Ayris does not
    depend on torch at all.

    Every failure mode is swallowed.  A mismatched driver raises ``RuntimeError``
    from C++; a missing ``nvcuda.dll`` raises ``OSError``; a CTranslate2 built
    without CUDA has no ``get_cuda_device_count`` at all.  None of those are
    reasons to fail a start, so each one becomes a sentence and a ``False``.

    Returns:
        ``(True, "")`` when a device is visible, otherwise ``(False, reason)``
        with a Russian sentence for the log and DevTools.
    """
    try:
        import ctranslate2
    except ImportError as exc:
        return False, f"ctranslate2 недоступен ({exc})"
    except Exception as exc:  # pragma: no cover - a broken install
        return False, f"ctranslate2 не импортируется ({exc})"

    counter = getattr(ctranslate2, "get_cuda_device_count", None)
    if not callable(counter):
        return False, "сборка ctranslate2 без поддержки CUDA"
    try:
        count = int(counter())
    except Exception as exc:
        # Driver/runtime mismatch, a card in a bad state, a container without
        # /dev/nvidia*.  The library raises from native code, so this catches
        # everything on purpose.
        return False, f"CUDA недоступна ({exc})"
    if count <= 0:
        return False, "видеокарта с поддержкой CUDA не найдена"
    return True, ""


class FasterWhisperEngine(SttEngine):
    """Whisper via CTranslate2, on CUDA when the machine has it."""

    name: ClassVar[str] = "whisper"
    package: ClassVar[str] = "faster-whisper"
    module: ClassVar[str] = "faster_whisper"
    supports_streaming: ClassVar[bool] = False
    #: A CTranslate2 model unpacks its weights and keeps a decoding cache, so
    #: resident memory runs about half again the size on disk.
    memory_factor: ClassVar[float] = 1.6

    __slots__ = ("_compute_type", "_device", "_fallback_reason", "_language", "_model")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._device = "cpu"
        self._compute_type = CPU_COMPUTE_TYPE
        self._language = DEFAULT_LANGUAGE
        self._fallback_reason = ""

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """The subset of Whisper's languages Ayris offers."""
        return _LANGUAGES

    @property
    def device(self) -> str:
        """``cuda`` or ``cpu`` - what the model actually loaded onto."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Quantisation in use: ``float16`` on CUDA, ``int8`` on the CPU."""
        return self._compute_type

    @property
    def fallback_reason(self) -> str:
        """Why the GPU was not used, empty when it was or was never asked for.

        Shown in DevTools next to the device, because "recognition got slow after
        I updated my drivers" is otherwise an unanswerable question.
        """
        return self._fallback_reason

    def load(self, model_path: Path, options: SttOptions) -> None:
        """Load a CTranslate2 Whisper model, preferring CUDA when allowed.

        Args:
            model_path: Directory holding ``model.bin`` and the tokenizer.  Never
                a bare size name - see the module docstring.
            options: ``gpu`` picks the device policy, ``threads`` bounds CPU
                decoding, ``language`` is passed to every transcription.

        Raises:
            SttError: The library is missing, the directory is not a
                CTranslate2 model, or both the GPU *and* the CPU refused it.
                Only the last case is fatal: a GPU failure alone falls back.
        """
        module = self._import()
        directory = self._resolve_model(model_path, markers=_MODEL_MARKERS)
        device, compute_type, reason = self._choose_device(options.gpu)

        started = perf_counter()
        model, device, compute_type, reason = self._open(
            module, directory, options, device, compute_type, reason
        )
        self._model = model
        self._device = device
        self._compute_type = compute_type
        self._fallback_reason = reason
        self._language = options.language or DEFAULT_LANGUAGE
        self._options = options
        self._model_path = directory
        _log.info(
            "whisper: модель %s загружена за %.0f мс на %s (%s)%s",
            directory.name,
            (perf_counter() - started) * 1000.0,
            device,
            compute_type,
            f", причина отката: {reason}" if reason else "",
        )

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise a complete buffer.

        Raises:
            SttError: If no model is loaded or the decoder failed.
        """
        options = self._require_loaded()
        prepared = self._prepare(audio)
        if prepared.duration_ms < options.min_speech_ms or prepared.is_silent():
            # The cheapest of the three hallucination filters, and the one that
            # catches the most: a key released too early never reaches the model.
            return self._empty(prepared.duration_ms)

        started = perf_counter()
        try:
            segments, info = self._model.transcribe(
                prepared.floats(),
                language=self._language or None,
                beam_size=options.beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
                no_speech_threshold=options.no_speech_threshold,
                without_timestamps=False,
            )
            collected = self._collect(segments, options.no_speech_threshold)
        except Exception as exc:
            raise SttError(
                f"whisper: transcription failed on {self._device}: {exc}",
                user_message="Ошибка распознавания речи.",
            ) from exc
        inference_ms = (perf_counter() - started) * 1000.0

        text = " ".join(segment.text for segment in collected).strip()
        if not text or _is_hallucination(text):
            if text:
                _log.debug("whisper: отброшена галлюцинация %r", text)
            return self._empty(prepared.duration_ms, inference_ms=inference_ms)

        return TranscriptResult(
            text=text,
            confidence=_mean_confidence(collected),
            segments=collected,
            language=_detected_language(info, self._language),
            duration_ms=prepared.duration_ms,
            engine=self.name,
            device=self._device,
            model=self.model_name,
            inference_ms=inference_ms,
        )

    def unload(self) -> None:
        """Drop the model.  Safe to call twice.

        Releasing the Python reference is all that can be done from here: the
        VRAM comes back when CTranslate2's destructor runs, which happens on the
        next collection.  The worker calls :func:`gc.collect` after this for
        exactly that reason - an eco-mode unload that does not actually return
        the memory would be pointless.
        """
        self._model = None
        self._options = None
        self._model_path = None
        self._fallback_reason = ""

    # ---------------------------------------------------------------- helpers

    def _empty(self, duration_ms: float, *, inference_ms: float = 0.0) -> TranscriptResult:
        """A "nothing was said" result carrying this engine's identity."""
        return TranscriptResult.empty(
            engine=self.name,
            language=self._language,
            duration_ms=duration_ms,
            device=self._device,
            model=self.model_name,
            inference_ms=inference_ms,
        )

    @staticmethod
    def _choose_device(preference: str) -> tuple[str, str, str]:
        """Turn ``performance.gpu`` into a device, a compute type and a reason.

        ``cuda`` is honoured without probing: a user who forced it wants the
        error if it is not there, not a silent downgrade that makes recognition
        four times slower for reasons they cannot see.
        """
        if preference == "cpu":
            return "cpu", CPU_COMPUTE_TYPE, ""
        if preference == "cuda":
            return "cuda", CUDA_COMPUTE_TYPE, ""
        usable, reason = cuda_available()
        if usable:
            return "cuda", CUDA_COMPUTE_TYPE, ""
        return "cpu", CPU_COMPUTE_TYPE, reason

    def _open(
        self,
        module: Any,
        directory: Path,
        options: SttOptions,
        device: str,
        compute_type: str,
        reason: str,
    ) -> tuple[Any, str, str, str]:
        """Construct the model, falling back from CUDA to the CPU once.

        Returns:
            The model, the device it is on, its compute type and the reason for
            any fallback.

        Raises:
            SttError: If the CPU attempt fails too, or if CUDA was explicitly
                demanded and refused.
        """
        try:
            model = module.WhisperModel(
                str(directory),
                device=device,
                compute_type=compute_type,
                cpu_threads=max(1, options.threads),
                num_workers=1,
            )
        except Exception as exc:
            if device != "cuda":
                raise SttError(
                    f"whisper: cannot load model at {directory}: {exc}",
                    user_message=(
                        f"Не удалось загрузить модель распознавания «{directory.name}». "
                        f"Проверьте её в настройках голоса."
                    ),
                ) from exc
            if options.gpu == "cuda":
                # Explicitly demanded, so say so rather than quietly halving the
                # speed of everything that follows.
                raise SttError(
                    f"whisper: cuda initialisation failed: {exc}",
                    user_message=(
                        "Не удалось запустить распознавание на видеокарте. "
                        "Выберите «Ускорение: авто» или «процессор» в настройках."
                    ),
                ) from exc
            reason = f"инициализация CUDA не удалась ({exc})"
            _log.warning("whisper: %s, переключаюсь на процессор", reason)
            return self._open(module, directory, options, "cpu", CPU_COMPUTE_TYPE, reason)
        return model, device, compute_type, reason

    def _collect(
        self, segments: Iterable[Any], no_speech_threshold: float
    ) -> tuple[TranscriptSegment, ...]:
        """Drain the generator faster-whisper returns, filtering as it goes.

        The generator is lazy - nothing is decoded until it is iterated - which
        is why the timing measurement wraps this call and not the one above it.
        """
        collected: list[TranscriptSegment] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            no_speech = _as_float(getattr(segment, "no_speech_prob", 0.0))
            if no_speech >= no_speech_threshold:
                _log.debug("whisper: сегмент %r отброшен, no_speech=%.2f", text, no_speech)
                continue
            start_ms = _as_float(getattr(segment, "start", 0.0)) * 1000.0
            end_ms = max(start_ms, _as_float(getattr(segment, "end", 0.0)) * 1000.0)
            if end_ms - start_ms < _MIN_SEGMENT_MS:
                continue
            collected.append(
                TranscriptSegment(
                    text=text,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=_logprob_to_confidence(
                        _as_float(getattr(segment, "avg_logprob", 0.0))
                    ),
                )
            )
        return tuple(collected)


def _logprob_to_confidence(avg_logprob: float) -> float:
    """Turn Whisper's mean token log-probability into a 0..1 confidence.

    ``exp(avg_logprob)`` is the geometric mean probability of the tokens, which
    is the closest thing Whisper has to a confidence and lands in a usable range:
    a clean phrase scores around 0.85, a guessed one around 0.4.  It is not
    calibrated against Vosk's word confidences and does not need to be - both
    feed the same slider, and a user tunes the slider to whichever engine they
    use.
    """
    if avg_logprob >= 0.0:
        return 1.0
    try:
        return max(0.0, min(1.0, math.exp(avg_logprob)))
    except OverflowError:  # pragma: no cover - only for absurd inputs
        return 0.0


def _mean_confidence(segments: Sequence[TranscriptSegment]) -> float:
    """Duration-weighted mean confidence over the kept segments.

    Weighted, because an unweighted mean lets a 200 ms interjection outvote a
    four-second sentence.  Falls back to a plain mean when no segment has a
    length, which happens on a model built without timestamps.
    """
    if not segments:
        return 0.0
    total = sum(segment.duration_ms for segment in segments)
    if total <= 0.0:
        return sum(segment.confidence for segment in segments) / len(segments)
    return sum(segment.confidence * segment.duration_ms for segment in segments) / total


def _is_hallucination(text: str) -> bool:
    """Whether the whole transcript is one of Whisper's silence artefacts."""
    return text.strip().casefold().rstrip(".!") in {
        phrase.rstrip(".!") for phrase in _HALLUCINATIONS
    }


def _detected_language(info: Any, fallback: str) -> str:
    """Language from faster-whisper's ``TranscriptionInfo``, or the configured one."""
    detected = getattr(info, "language", None)
    return str(detected) if detected else fallback


def _as_float(value: Any) -> float:
    """Read a vendor attribute as a float without trusting its type."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
