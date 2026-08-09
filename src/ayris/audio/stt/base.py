"""The contract every speech recogniser implements.

An STT engine answers one question - "what was said in this buffer" - and owns
exactly one thing while doing it: its model.  It does not read the microphone, it
does not decide when a phrase ended, and it does not know whether the text will
go to the LLM or into a text field.  Segmentation lives in
:mod:`ayris.audio.segmenter`, the microphone in :mod:`ayris.audio.capture`, and
the lifecycle - when to load, when to give the memory back - in
:mod:`ayris.workers.stt_worker`.

**Why an ABC and not three functions.**  Section 6 of the specification asks for
Vosk *and* faster-whisper *and* an optional Whisper.cpp, chosen in the settings
window, plus cloud providers later.  They agree on almost nothing else: Vosk
streams and returns per-word confidences, faster-whisper works on a whole buffer
and returns per-segment log-probabilities, Whisper.cpp returns whatever its
Python binding felt like exposing.  What they do agree on is
:meth:`SttEngine.transcribe`, and :class:`TranscriptResult` is the shape the rest
of Ayris is allowed to know about.

**Confidence is normalised here, not by callers.**  ``voice.stt.min_confidence``
is one number in the settings window, and a user who raises it expects fewer
wrong transcripts from whichever engine is selected.  So every engine maps its
own notion of certainty onto 0.0-1.0 - Vosk averages its word confidences,
faster-whisper turns ``avg_logprob`` into a probability - and nothing downstream
has to know which engine produced the number.

**Silence is an empty result, not an exception.**  A push-to-talk key released
too early, a wake word that fired on a door closing, a phrase the segmenter
accepted on a noise spike: all of them reach an engine as a buffer with nothing
in it.  Whisper answers those with hallucinated subtitle credits, which is worse
than useless, so :meth:`SttEngine.transcribe` returns
:meth:`TranscriptResult.empty` and lets the caller decide whether to tell the
user anything at all.

**Missing libraries are a message, not a crash.**  Every engine imports its
vendor package inside :meth:`SttEngine.load`, never at module import time.  A
user on Vosk must not need ``ctranslate2`` on disk, and the test suite has to
stay collectable on a runner where the wheel was never installed.
:meth:`SttEngine.available` answers the settings window's question without
importing anything at all.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import wave
from abc import ABC, abstractmethod
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, NoReturn, Self

from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.core.errors import SttError
from ayris.core.models import JsonObject

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "DEFAULT_LANGUAGE",
    "ENGINE_ENTRYPOINTS",
    "MIN_SPEECH_MS",
    "SILENCE_DBFS",
    "STT_SAMPLE_RATE",
    "AudioBuffer",
    "SttEngine",
    "SttOptions",
    "TranscriptResult",
    "TranscriptSegment",
    "create_engine",
    "engine_class",
    "engine_names",
    "estimate_model_bytes",
]

#: What every offline model here is trained at, and what
#: :mod:`ayris.audio.capture` normalises to.  Vosk refuses anything else
#: outright; Whisper resamples internally and loses quality doing it.
STT_SAMPLE_RATE: Final = 16000

#: Language every engine is configured for unless the caller says otherwise.
#: Ayris is a Russian assistant; a model that cannot do Russian is misconfigured,
#: not a fallback.
DEFAULT_LANGUAGE: Final = "ru"

#: Shorter than this and there is no phrase to recognise.  Whisper is the reason
#: the constant exists: on a 90 ms buffer it does not return an empty string, it
#: invents one, usually "Продолжение следует..." from the subtitle corpus it was
#: trained on.  Below this length the answer is empty by construction.
MIN_SPEECH_MS: Final = 200

#: RMS level below which a buffer is treated as room tone.  Not -60: a real quiet
#: room with dither sits around -65 dBFS, and a phrase spoken from across the
#: room still clears -45.  Measured on the whole buffer, so leading and trailing
#: silence pulls the number down - which is why the check is a cheap
#: pre-filter and the engine's own no-speech probability is the real test.
SILENCE_DBFS: Final = -55.0

#: Full-scale amplitude of a 16-bit sample, for the dBFS conversion.
_FULL_SCALE: Final = 32768.0

#: Engine name (the value of ``voice.stt.offline_engine``) to ``module:Class``.
#: Resolved lazily by :func:`create_engine`, so importing this module costs
#: nothing and an engine module that fails to import cannot take the others down
#: with it.
ENGINE_ENTRYPOINTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "vosk": "ayris.audio.stt.vosk_engine:VoskSttEngine",
        "whisper": "ayris.audio.stt.faster_whisper_engine:FasterWhisperEngine",
        "whispercpp": "ayris.audio.stt.whispercpp_engine:WhisperCppEngine",
    }
)


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    """Mono or stereo ``int16`` PCM with the metadata needed to interpret it.

    Deliberately not the same type as
    :class:`~ayris.workers.protocol.AudioChunk`: that one describes a block in
    shared memory and is what crosses the process boundary, this one is the bytes
    themselves after the worker has attached to the block and normalised the
    rate.  Keeping them apart means an engine can be unit-tested from a WAV file
    without a supervisor, a shared-memory segment or a second process.

    Attributes:
        pcm: Interleaved little-endian ``int16`` samples.
        sample_rate: Frames per second.
        channels: 1 or 2.  Engines want mono; :meth:`prepared_for` gets there.

    Raises:
        SttError: If the rate or channel count is impossible, or the byte count
            is not a whole number of frames.  All three mean the caller
            mis-assembled the buffer, and a truncated final frame would shift
            every timestamp in the result.
    """

    pcm: bytes
    sample_rate: int = STT_SAMPLE_RATE
    channels: int = 1

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise SttError(
                f"audio buffer sample rate must be positive, got {self.sample_rate}",
                user_message="Внутренняя ошибка распознавания речи.",
            )
        if self.channels not in (1, 2):
            raise SttError(
                f"audio buffer must be mono or stereo, got {self.channels} channels",
                user_message="Внутренняя ошибка распознавания речи.",
            )
        stride = SAMPLE_WIDTH * self.channels
        if len(self.pcm) % stride:
            raise SttError(
                f"audio buffer of {len(self.pcm)} bytes is not a whole number of "
                f"{self.channels}-channel int16 frames",
                user_message="Внутренняя ошибка распознавания речи.",
            )

    @property
    def frames(self) -> int:
        """Sample frames, counting all channels of one instant as one frame."""
        return len(self.pcm) // (SAMPLE_WIDTH * self.channels)

    @property
    def duration_ms(self) -> float:
        """Length of the buffer in milliseconds."""
        return self.frames * 1000.0 / self.sample_rate

    @property
    def is_empty(self) -> bool:
        """Whether there is nothing to recognise at all."""
        return self.frames == 0

    def samples(self) -> array[int]:
        """The buffer as an ``array('h')``.

        A copy, because the caller usually wants to keep it past the lifetime of
        a shared-memory view.
        """
        block = array("h")
        block.frombytes(self.pcm)
        return block

    def rms_dbfs(self) -> float:
        """Level of the whole buffer, in dBFS.  ``-inf`` for digital silence."""
        block = self.samples()
        if not block:
            return -math.inf
        total = 0.0
        for value in block:
            total += float(value) * float(value)
        mean = total / len(block)
        if mean <= 0.0:
            return -math.inf
        return 20.0 * math.log10(math.sqrt(mean) / _FULL_SCALE)

    def is_silent(self, threshold_dbfs: float = SILENCE_DBFS) -> bool:
        """Whether the buffer is too quiet or too short to hold a phrase.

        Both halves matter.  The length test catches a key released the instant
        it was pressed; the level test catches an open microphone in an empty
        room.  Either way the honest answer is "nothing was said", and asking
        Whisper for it costs a second of GPU time and returns fiction.
        """
        return self.duration_ms < MIN_SPEECH_MS or self.rms_dbfs() < threshold_dbfs

    def to_mono(self) -> AudioBuffer:
        """This buffer with the channels averaged.  A no-op when already mono."""
        if self.channels == 1:
            return self
        block = self.samples()
        mixed = array("h", bytes(len(self.pcm) // 2))
        for index in range(len(mixed)):
            mixed[index] = (block[2 * index] + block[2 * index + 1]) // 2
        return AudioBuffer(mixed.tobytes(), sample_rate=self.sample_rate, channels=1)

    def resampled_to(self, sample_rate: int) -> AudioBuffer:
        """This buffer at another rate.  Mono only; a no-op when rates match.

        Uses :class:`~ayris.audio.capture.Resampler`, imported here rather than
        at module level so that a process which only ever builds an
        :class:`AudioBuffer` does not pay for the capture module and the device
        enumeration behind it.

        Raises:
            SttError: If the buffer is not mono.  Downmixing first is the
                caller's decision, and :meth:`prepared_for` makes it.
        """
        if self.channels != 1:
            raise SttError(
                "resample a mono buffer; call to_mono() first",
                user_message="Внутренняя ошибка распознавания речи.",
            )
        if sample_rate == self.sample_rate:
            return self
        from ayris.audio.capture import Resampler

        resampler = Resampler(self.sample_rate, sample_rate)
        return AudioBuffer(resampler.process(self.pcm), sample_rate=sample_rate, channels=1)

    def prepared_for(self, sample_rate: int = STT_SAMPLE_RATE) -> AudioBuffer:
        """Mono at ``sample_rate`` - the only shape an engine accepts."""
        return self.to_mono().resampled_to(sample_rate)

    def floats(self) -> array[float]:
        """The buffer as ``float32`` in -1.0..1.0, which is what Whisper wants.

        Built with :mod:`array` rather than NumPy: faster-whisper accepts any
        buffer protocol object, and Ayris does not make NumPy a hard dependency
        of the recognition path.
        """
        return array("f", (value / _FULL_SCALE for value in self.samples()))

    @classmethod
    def from_wav(cls, path: Path) -> Self:
        """Read a mono or stereo 16-bit WAV file.

        Used by the tests and by the ``hardware`` runs that feed a real
        recording through a real model.

        Raises:
            SttError: If the file is not 16-bit PCM.  Converting silently would
                hide a fixture that was written with the wrong parameters.
        """
        try:
            with wave.open(str(path), "rb") as handle:
                width = handle.getsampwidth()
                channels = handle.getnchannels()
                rate = handle.getframerate()
                pcm = handle.readframes(handle.getnframes())
        except (OSError, wave.Error) as exc:
            raise SttError(
                f"cannot read WAV file {path}: {exc}",
                user_message="Не удалось прочитать звуковой файл.",
            ) from exc
        if width != SAMPLE_WIDTH:
            raise SttError(
                f"{path} is {width * 8}-bit; only 16-bit PCM is supported",
                user_message="Звуковой файл должен быть 16-битным.",
            )
        return cls(pcm, sample_rate=rate, channels=channels)


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timed piece of the transcript.

    What a segment *is* depends on the engine: a word for Vosk, a clause for
    Whisper.  Callers must not assume either.  What they can assume is that the
    segments are in order, that the times are milliseconds from the start of the
    buffer that was passed in, and that concatenating :attr:`text` with spaces
    reproduces something very close to :attr:`TranscriptResult.text`.

    Timings are what makes a segment worth keeping: section 6 wants the overlay
    to highlight words as they are confirmed, and the correction dialog needs to
    know which part of the audio to replay.

    Attributes:
        text: The words themselves, without surrounding whitespace.
        start_ms: Offset from the beginning of the recognised buffer.
        end_ms: End of the segment.  Never before :attr:`start_ms`.
        confidence: 0.0-1.0, normalised the same way as
            :attr:`TranscriptResult.confidence`.  Zero means the engine did not
            say, which is not the same as "certainly wrong".

    Raises:
        SttError: If the interval runs backwards, which would break every
            consumer that sorts or seeks by it.
    """

    text: str
    start_ms: float = 0.0
    end_ms: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.end_ms < self.start_ms:
            raise SttError(
                f"segment {self.text!r} ends at {self.end_ms} before it starts "
                f"at {self.start_ms}",
                user_message="Внутренняя ошибка распознавания речи.",
            )

    @property
    def duration_ms(self) -> float:
        """Length of the segment."""
        return self.end_ms - self.start_ms

    def to_params(self) -> JsonObject:
        """Flatten for the worker pipe and for the database."""
        return {
            "text": self.text,
            "start_ms": round(self.start_ms, 1),
            "end_ms": round(self.end_ms, 1),
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_params`, tolerating missing keys."""
        return cls(
            text=str(params.get("text", "")),
            start_ms=_as_float(params.get("start_ms"), 0.0),
            end_ms=_as_float(params.get("end_ms"), 0.0),
            confidence=_as_float(params.get("confidence"), 0.0),
        )


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """What an engine heard, plus how long it took to hear it.

    The timings are not decoration.  Section 12 of the specification puts a
    budget on the whole pipeline, and the only way to know which stage spent it
    is to measure each one: :attr:`inference_ms` against
    :attr:`duration_ms` gives :attr:`real_time_factor`, which is the single
    number that says whether the selected model fits the machine it is running
    on.  :mod:`ayris.workers.stt_worker` writes all three into the pipeline log
    on every call.

    Attributes:
        text: The transcript, stripped.  Empty when nothing was said - see
            :meth:`empty`.
        confidence: 0.0-1.0 for the whole phrase.  Compared against
            ``voice.stt.min_confidence`` by the caller, not here: an engine
            reports, the pipeline decides.
        segments: Timed pieces, in order.  May be empty even when :attr:`text`
            is not; not every engine can be made to give timings.
        language: What the model thinks was spoken, as a two-letter code.  Whisper
            detects it, Vosk repeats what it was configured with.
        duration_ms: Length of the audio, so the ratio below can be recomputed
            without keeping the buffer.
        engine: :attr:`SttEngine.name` of whoever produced this.
        device: ``cpu`` or ``cuda`` - which one actually ran, not which one was
            asked for.  The distinction is the whole point of the fallback in
            :mod:`ayris.audio.stt.faster_whisper_engine`.
        model: Name of the loaded model, for the log and the DevTools panel.
        inference_ms: Wall-clock time inside the engine.  Excludes model load,
            which is reported separately because it happens once.
        partial: Whether more audio is still coming.  Streaming engines emit
            partial results so the overlay can show text while the user talks.
    """

    text: str = ""
    confidence: float = 0.0
    segments: tuple[TranscriptSegment, ...] = ()
    language: str = DEFAULT_LANGUAGE
    duration_ms: float = 0.0
    engine: str = ""
    device: str = "cpu"
    model: str = ""
    inference_ms: float = 0.0
    partial: bool = False

    @property
    def is_empty(self) -> bool:
        """Whether nothing was recognised.

        Stripped before the test: Vosk answers a silent buffer with a single
        space often enough that a caller checking ``not result.text`` would act
        on it, and " " is not a phrase.
        """
        return not self.text.strip()

    @property
    def real_time_factor(self) -> float:
        """Inference time over audio time.

        Below 1.0 means the engine is faster than real time, which is what a
        voice assistant needs; 0.3 is a healthy Vosk small model on a laptop
        core.  Zero when the audio length is unknown, never a division error.
        """
        if self.duration_ms <= 0.0:
            return 0.0
        return self.inference_ms / self.duration_ms

    @property
    def word_count(self) -> int:
        """Words in :attr:`text`, for the log line."""
        return len(self.text.split())

    def with_timing(self, *, inference_ms: float, duration_ms: float | None = None) -> Self:
        """A copy carrying measured times.

        Engines build their result before they can know the total, so the timing
        is stamped on afterwards rather than threaded through every branch.
        """
        return type(self)(
            text=self.text,
            confidence=self.confidence,
            segments=self.segments,
            language=self.language,
            duration_ms=self.duration_ms if duration_ms is None else duration_ms,
            engine=self.engine,
            device=self.device,
            model=self.model,
            inference_ms=inference_ms,
            partial=self.partial,
        )

    def to_params(self) -> JsonObject:
        """Flatten for the worker pipe.

        A dataclass would pickle fine, but the response then only means anything
        to a process that can import this module.  Plain keys let DevTools show a
        transcript and let the supervisor build a
        :class:`~ayris.core.events.TranscriptReady` without importing the STT
        package into the main process.
        """
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "segments": [segment.to_params() for segment in self.segments],
            "language": self.language,
            "duration_ms": round(self.duration_ms, 1),
            "engine": self.engine,
            "device": self.device,
            "model": self.model,
            "inference_ms": round(self.inference_ms, 1),
            "real_time_factor": round(self.real_time_factor, 4),
            "partial": self.partial,
        }

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> Self:
        """Rebuild from :meth:`to_params`, tolerating missing keys."""
        raw = params.get("segments")
        segments: tuple[TranscriptSegment, ...] = ()
        if isinstance(raw, Sequence) and not isinstance(raw, str | bytes):
            segments = tuple(
                TranscriptSegment.from_params(item)
                for item in raw
                if isinstance(item, Mapping)  # a malformed entry is dropped, not fatal
            )
        return cls(
            text=str(params.get("text", "")),
            confidence=_as_float(params.get("confidence"), 0.0),
            segments=segments,
            language=str(params.get("language", DEFAULT_LANGUAGE)),
            duration_ms=_as_float(params.get("duration_ms"), 0.0),
            engine=str(params.get("engine", "")),
            device=str(params.get("device", "cpu")),
            model=str(params.get("model", "")),
            inference_ms=_as_float(params.get("inference_ms"), 0.0),
            partial=bool(params.get("partial", False)),
        )

    @classmethod
    def empty(
        cls,
        *,
        engine: str = "",
        duration_ms: float = 0.0,
        language: str = DEFAULT_LANGUAGE,
        device: str = "cpu",
        model: str = "",
        inference_ms: float = 0.0,
    ) -> Self:
        """Nothing was said.

        A first-class outcome, not an error: it is what silence, a mis-fired wake
        word and a half-pressed hotkey all produce, and all three are normal.
        The caller shows nothing and keeps listening.
        """
        return cls(
            text="",
            confidence=0.0,
            language=language,
            duration_ms=duration_ms,
            engine=engine,
            device=device,
            model=model,
            inference_ms=inference_ms,
        )


@dataclass(frozen=True, slots=True)
class SttOptions:
    """Everything an engine needs beyond the model directory.

    One flat object rather than a signature per engine, so that
    :mod:`ayris.workers.stt_worker` can build it once from its parameters and
    hand it to whichever engine the settings named.  Anything genuinely specific
    to one vendor goes in :attr:`extra`, which keeps this from growing a field
    per engine.

    Attributes:
        language: Two-letter code the model is asked for.  Whisper accepts an
            empty string as "detect it", Vosk ignores it - its model is
            single-language by construction.
        threads: CPU threads the engine may use, from
            ``performance.stt_threads``.  Bounded there at 4: past that a small
            model spends more time synchronising than decoding.
        gpu: ``auto``, ``cuda`` or ``cpu`` from ``performance.gpu``.  Only
            faster-whisper and Whisper.cpp look at it.
        punctuation: Whether to ask the engine to punctuate, when it can.
        min_confidence: Threshold the *caller* applies.  Passed in so an engine
            that can push it down into its own decoder (Vosk's alternatives) may
            do so; nobody is required to.
        partial_results: Whether the streaming path should emit intermediate
            text.  Costs a little CPU and makes the overlay feel instant.
        beam_size: Decoder width for Whisper.  1 is greedy and about twice as
            fast; 5 is the library default and noticeably better on Russian.
        min_speech_ms: Buffers shorter than this are answered empty without
            running the model.  See :data:`MIN_SPEECH_MS`.
        no_speech_threshold: Whisper's own probability that a segment is silence.
            Above it, the segment is dropped - this is the hallucination filter.
        extra: Vendor-specific extras with no home in the settings model yet.
    """

    language: str = DEFAULT_LANGUAGE
    threads: int = 2
    gpu: str = "auto"
    punctuation: bool = True
    min_confidence: float = 0.4
    partial_results: bool = True
    beam_size: int = 5
    min_speech_ms: int = MIN_SPEECH_MS
    no_speech_threshold: float = 0.6
    extra: Mapping[str, Any] = field(default_factory=dict)

    def option(self, name: str, fallback: str = "") -> str:
        """Read an extra as a string, tolerating a missing key."""
        value = self.extra.get(name)
        return fallback if value is None else str(value)

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> Self:
        """Build from worker parameters.

        The worker receives plain JSON-ish values over the pipe - a pydantic
        model cannot be pickled into a spawned process without importing the
        settings module there - so every field is coerced and every absence falls
        back to the default.  An older supervisor can start a newer worker.
        """
        defaults = cls()
        return cls(
            language=str(params.get("language", defaults.language)) or defaults.language,
            threads=max(1, _as_int(params.get("threads"), defaults.threads)),
            gpu=str(params.get("gpu", defaults.gpu)) or defaults.gpu,
            punctuation=bool(params.get("punctuation", defaults.punctuation)),
            min_confidence=_as_float(params.get("min_confidence"), defaults.min_confidence),
            partial_results=bool(params.get("partial_results", defaults.partial_results)),
            beam_size=max(1, _as_int(params.get("beam_size"), defaults.beam_size)),
            min_speech_ms=max(0, _as_int(params.get("min_speech_ms"), defaults.min_speech_ms)),
            no_speech_threshold=_as_float(
                params.get("no_speech_threshold"), defaults.no_speech_threshold
            ),
            extra=dict(params.get("extra") or {}),
        )


class SttEngine(ABC):
    """Base class for a speech recogniser.

    Subclasses implement three methods and may override the streaming trio.
    Everything else - when to load, when to unload, how to get the audio here,
    what to do with a low confidence - belongs to
    :mod:`ayris.workers.stt_worker`.

    Instances are not thread-safe.  One engine belongs to one worker process, and
    the worker dispatches calls one at a time.
    """

    #: Value of ``voice.stt.offline_engine`` this class implements.
    name: ClassVar[str] = "stt"

    #: Distribution to install, named in the error the user sees when it is
    #: missing.  Usually equal to :attr:`module`, but not always - the package on
    #: PyPI is ``faster-whisper`` and the module is ``faster_whisper``.
    package: ClassVar[str] = ""

    #: Top-level module imported at load time.  Checked by :meth:`available`
    #: without importing it.
    module: ClassVar[str] = ""

    #: Whether the engine is offered only when its library is present.  The two
    #: engines Ayris ships with are always listed, so that a user who has not
    #: installed one yet still sees why; Whisper.cpp is a build-it-yourself
    #: binding and is hidden until it exists.
    optional: ClassVar[bool] = False

    #: Whether :meth:`accept_audio` does anything.  Vosk decodes incrementally
    #: and can hand back text while the user is still speaking; Whisper needs the
    #: whole buffer before it starts.
    supports_streaming: ClassVar[bool] = False

    #: Resident memory per byte of model on disk.  Vosk mmaps its graph and sits
    #: at roughly its file size; a CTranslate2 model unpacks its weights and
    #: needs about half again.  Used by the worker's RAM check, where being
    #: approximately right early beats being exactly right after the OOM.
    memory_factor: ClassVar[float] = 1.0

    __slots__ = ("_model_path", "_options")

    def __init__(self) -> None:
        self._options: SttOptions | None = None
        self._model_path: Path | None = None

    # ------------------------------------------------------------ description

    @property
    def sample_rate(self) -> int:
        """Rate the buffers handed to :meth:`transcribe` must be at.

        16 kHz for every offline engine here.  Declared as a property rather
        than a constant because a cloud provider added later may well want 8 or
        48, and the worker resamples to whatever this says.
        """
        return STT_SAMPLE_RATE

    @property
    @abstractmethod
    def supported_languages(self) -> tuple[str, ...]:
        """Language codes this engine can be asked for.

        A single-element tuple for a Vosk model - the language is baked into the
        files on disk - and a long list for Whisper.  An empty tuple means "any",
        which is what a model that was not loaded yet has to say.
        """

    @property
    def device(self) -> str:
        """What is actually running the model: ``cpu`` or ``cuda``.

        Not what the settings asked for.  An engine that fell back to the CPU
        because the GPU had no memory left reports ``cpu`` here, and that string
        goes into :attr:`TranscriptResult.device` and from there into the log the
        user sends when they ask why recognition got slow.
        """
        return "cpu"

    @property
    def loaded(self) -> bool:
        """Whether a model is in memory and :meth:`transcribe` will do anything."""
        return self._options is not None

    @property
    def options(self) -> SttOptions | None:
        """The options the engine was loaded with."""
        return self._options

    @property
    def model_path(self) -> Path | None:
        """Directory or file the loaded model came from."""
        return self._model_path

    @property
    def model_name(self) -> str:
        """Name of the loaded model, for logs and results."""
        return self._model_path.name if self._model_path is not None else ""

    # -------------------------------------------------------------- lifecycle

    @abstractmethod
    def load(self, model_path: Path, options: SttOptions) -> None:
        """Bring the model into memory.

        The slowest thing in Ayris' startup: a second for a Vosk small model on a
        warm cache, twenty for a Whisper small on a cold disk.  Which is exactly
        why the worker calls it in a background thread and reports progress as an
        event rather than blocking a request.

        Args:
            model_path: Directory (Vosk, CTranslate2) or file (GGML) to load.
            options: Language, threads, device preference and the rest.

        Raises:
            SttError: The library is missing, the path is not there, the model is
                the wrong kind, or the device refused.  The Russian message says
                which, because all four are things the user can act on.
        """

    @abstractmethod
    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise one complete buffer.

        The buffer must already be mono at :attr:`sample_rate`; the worker
        guarantees that with :meth:`AudioBuffer.prepared_for`.  Implementations
        call :meth:`_prepare` anyway, so a direct caller in a test does not have
        to remember.

        Returns:
            A filled :class:`TranscriptResult`, or :meth:`TranscriptResult.empty`
            when there was nothing to hear.  Silence is never an exception.

        Raises:
            SttError: No model is loaded, or the engine itself failed.  A failure
                here is recoverable - the caller may retry, or fall back to the
                cloud - so it must not leave the engine half-unloaded.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the model.  Must be safe to call twice and must not raise.

        Called on shutdown, when the settings change the model, and by the
        worker's idle timer.  The last one is the reason it has to be cheap and
        total: ``performance.eco_mode`` exists so that a machine with 8 GB does
        not keep a Whisper model resident between two questions an hour apart.
        """

    # -------------------------------------------------------------- streaming

    def start_stream(self) -> None:
        """Begin an incremental recognition.

        Only meaningful when :attr:`supports_streaming`.  The default raises,
        because silently accepting audio that will never be decoded is worse than
        a clear refusal.

        Raises:
            SttError: Always, for an engine that cannot stream.
        """
        self._no_streaming()

    def accept_audio(self, pcm: bytes) -> str:
        """Feed the next piece of a stream.

        Args:
            pcm: Mono ``int16`` at :attr:`sample_rate`.  Any length; engines
                buffer internally.

        Returns:
            The partial text so far, or an empty string when the engine has
            nothing new to say.  Partial text is a preview for the overlay and
            may change completely on the next call - only
            :meth:`finish_stream` is final.

        Raises:
            SttError: Always, for an engine that cannot stream.
        """
        del pcm
        self._no_streaming()

    def finish_stream(self) -> TranscriptResult:
        """Close the stream and return the final transcript.

        Raises:
            SttError: Always, for an engine that cannot stream.
        """
        self._no_streaming()

    # ---------------------------------------------------------------- helpers

    @classmethod
    def available(cls) -> bool:
        """Whether the vendor library can be imported.

        Uses :func:`importlib.util.find_spec`, so asking does not pay for the
        answer: the settings window can grey out an engine without loading a
        machine-learning runtime into the GUI process.
        """
        if not cls.module:
            return False
        try:
            return importlib.util.find_spec(cls.module) is not None
        except (ImportError, ValueError):
            # A namespace package shadowed by a broken install raises instead of
            # returning None.  Either way the import would fail.
            return False

    @classmethod
    def _import(cls, module: str | None = None) -> Any:
        """Import the vendor library.

        Raises:
            SttError: If it is not installed, naming the package to install.
        """
        target = module or cls.module
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            package = cls.package or target
            raise SttError(
                f"{cls.name}: {target} is not installed: {exc}",
                user_message=(
                    f"Движок распознавания «{cls.name}» не установлен. "
                    f"Установите пакет {package} или выберите другой движок "
                    f"в настройках голоса."
                ),
            ) from exc

    def _no_streaming(self) -> NoReturn:
        """Refuse a streaming call on an engine that cannot do it.

        Typed ``NoReturn`` so that the three streaming stubs above type-check
        without a dead ``return`` after the call.

        Raises:
            SttError: Always.
        """
        raise SttError(
            f"{self.name}: streaming recognition is not supported",
            user_message=(
                f"Движок «{self.name}» не умеет распознавать речь на ходу. "
                f"Промежуточный текст будет недоступен."
            ),
        )

    def _require_loaded(self) -> SttOptions:
        """The options the engine was loaded with.

        Raises:
            SttError: If :meth:`load` has not run.  The worker loads lazily, so
                seeing this means something bypassed it.
        """
        options = self._options
        if options is None:
            raise SttError(
                f"{self.name}: no model is loaded",
                user_message="Модель распознавания речи ещё не загружена.",
            )
        return options

    def _prepare(self, audio: AudioBuffer) -> AudioBuffer:
        """Coerce a buffer into the shape this engine accepts.

        Cheap and idempotent when the worker already did it - which it always
        does, because resampling twice is both a waste and a second round of
        interpolation error.
        """
        return audio.prepared_for(self.sample_rate)

    def _resolve_model(self, model_path: Path, *, markers: Sequence[str] = ()) -> Path:
        """Check that a model directory exists and looks like the right kind.

        Args:
            model_path: What the settings named.
            markers: Files or directories that must be inside.  Vosk has ``am``,
                a CTranslate2 model has ``model.bin``.  Empty skips the check.

        Raises:
            SttError: The path is missing or is not a model of this kind.  Two
                separate messages: "download the model" and "this is a model for
                another engine" send the user to different places.
        """
        if not model_path.exists():
            raise SttError(
                f"{self.name}: model path {model_path} does not exist",
                user_message=(
                    f"Модель распознавания «{model_path.name}» не найдена. "
                    f"Скачайте её в настройках голоса."
                ),
            )
        if (
            markers
            and model_path.is_dir()
            and not any((model_path / marker).exists() for marker in markers)
        ):
            expected = ", ".join(markers)
            raise SttError(
                f"{self.name}: {model_path} has none of {expected}",
                user_message=(
                    f"Папка «{model_path.name}» не похожа на модель для движка "
                    f"«{self.name}». Проверьте выбранную модель в настройках."
                ),
            )
        return model_path

    def __repr__(self) -> str:
        return f"{type(self).__name__}(loaded={self.loaded}, model={self.model_name!r})"


def engine_names(*, available_only: bool = False) -> tuple[str, ...]:
    """Engine names the settings window may offer.

    Args:
        available_only: Drop the engines whose library is not installed.  The
            settings window passes ``False`` and greys the missing ones out, so a
            user can see that Whisper exists; a caller picking a default passes
            ``True``.

    An engine marked :attr:`SttEngine.optional` is left out either way unless its
    library is present - Whisper.cpp has no wheel to install, so offering it on a
    machine without one would be offering a dead end.
    """
    names: list[str] = []
    for name in ENGINE_ENTRYPOINTS:
        try:
            engine = engine_class(name)
        except SttError:  # pragma: no cover - only a broken checkout
            continue
        if engine.optional and not engine.available():
            continue
        if available_only and not engine.available():
            continue
        names.append(name)
    return tuple(names)


def engine_class(name: str) -> type[SttEngine]:
    """Resolve an engine name to its class without constructing anything.

    The module is imported here and not at the top of this file, so that a
    vendor library with an expensive import costs nothing until somebody selects
    it, and so that an engine module which fails to import cannot stop the others
    from working.

    Raises:
        SttError: If the name is unknown or its module is broken.  Not silently
            downgraded to another engine: a user who picked Whisper and got Vosk
            would be debugging the wrong component.
    """
    entrypoint = ENGINE_ENTRYPOINTS.get(name)
    if entrypoint is None:
        known = ", ".join(sorted(ENGINE_ENTRYPOINTS))
        raise SttError(
            f"unknown stt engine {name!r}, expected one of {known}",
            user_message=f"Неизвестный движок распознавания речи: {name}.",
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - only a broken checkout
        raise SttError(
            f"cannot import stt engine {name!r}: {exc}",
            user_message=f"Не удалось загрузить движок распознавания «{name}».",
        ) from exc
    factory: type[SttEngine] = getattr(module, attribute)
    return factory


def create_engine(name: str) -> SttEngine:
    """Build the engine ``voice.stt.offline_engine`` names.

    Raises:
        SttError: Same as :func:`engine_class`.
    """
    return engine_class(name)()


def estimate_model_bytes(model_path: Path) -> int:
    """Size of a model on disk, following into directories.

    The input to the RAM check in :mod:`ayris.workers.stt_worker`.  Approximate
    on purpose: the alternative is loading the model to find out how much memory
    loading it needs.  Returns zero for a path that is not there, which the
    caller reads as "cannot tell" and lets through - refusing to load a model
    because its size could not be measured would be a worse failure than the OOM
    it was trying to prevent.
    """
    try:
        if model_path.is_file():
            return model_path.stat().st_size
        if not model_path.is_dir():
            return 0
        return sum(item.stat().st_size for item in model_path.rglob("*") if item.is_file())
    except OSError:
        return 0


def _as_float(value: object, fallback: float) -> float:
    """Coerce a wire value to a float without trusting it.

    Worker parameters arrive from a pickled dict that a supervisor of a different
    version may have built; ``bool`` is excluded because ``True`` as a duration
    is a bug, not a value.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value)


def _as_int(value: object, fallback: int) -> int:
    """Coerce a wire value to an int without trusting it."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return int(value)
