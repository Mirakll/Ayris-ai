"""GigaAM v3 through onnx-asr: the offline recogniser for Russian.

Measured on 42 files of live Piper speech (118 s, three conditions: clean, 10 dB
SNR, and ~0 dB SNR), one CPU thread:

===========================  =======  ======  =======  ========
движок                        WER %    RTFx   ср. мс   пик RAM
===========================  =======  ======  =======  ========
GigaAM v3 CTC int8               3.7    24.5      112    328 МБ
Vosk small ru 0.22              11.8     4.0      692    241 МБ
faster-whisper small int8       26.0     1.4     1920    738 МБ
===========================  =======  ======  =======  ========

That is why this is the default: seventeen times faster than Whisper *and* seven
times more accurate on the same audio, on a CPU, without a graphics card. Vosk is
still shipped, but for two things onnx-asr cannot do at all - partial results
while the user is still speaking, and the keyword grammar ``vosk_kws`` needs.

**The model is a folder, and the file names in it are load-bearing.** ``onnx-asr``
does not take a path to a weights file, it takes a directory and globs it:
``v?_ctc*.onnx`` and ``v?_vocab.txt`` for the plain CTC export, ``v3_e2e_ctc*``
for the end-to-end one, three files for an RNN-T. So the engine reads the folder
to decide *which* GigaAM it was given rather than trusting the setting to say -
one glob, one variant, and a folder holding the wrong export produces a sentence
naming what was actually in it. The catalog puts each variant in its own
subdirectory for the same reason (see
:attr:`~ayris.models.catalog.ModelEntry.directory`).

**Long audio is cut up, and not because of accuracy.** The ONNX export bakes the
self-attention mask at 5000 frames — 200 s of audio — and past that it does not
degrade, it raises ``BroadcastIterator::Append axis == 1``. Memory grows linearly
along the way: 20 s costs 328 MB, 60 s costs 559 MB, 120 s costs 1.2 GB. So a
buffer longer than :data:`_MAX_CHUNK_MS` is cut into pieces of about
:data:`_CHUNK_MS`. The cut lands on the quietest 100 ms frame within
:data:`_SEARCH_MS` of the target rather than on the target itself: cutting blind
splits a word in two and costs 1.6 % WER against the same audio recognised whole,
while cutting at the quiet point costs nothing measurable.

**Confidence is the geometric mean of the per-character probabilities.** The CTC
decoder hands back a log-probability per character, and ``exp(mean(logprobs))``
turns them into the 0..1 number ``voice.stt.min_confidence`` compares against.
It discriminates about as well as anything here can: clean and noisy speech score
0.94 and up, while the ~0 dB condition — the only one this model gets wrong at
all — drops to 0.60.

**CPU only, on purpose.** ``onnxruntime`` on PyPI is the CPU build; the CUDA one
is a different package with its own toolkit requirements, and at 112 ms per
command there is nothing here for a GPU to fix. ``performance.gpu`` is therefore
ignored by this engine instead of being quietly half-honoured.
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

__all__ = ["GIGAAM_VARIANTS", "GigaAmEngine"]

_log: Final = get_logger(__name__)

#: Glob of the file that identifies a variant, and the name ``onnx-asr`` knows it
#: by. Ordered from specific to general: ``v3_e2e_ctc.int8.onnx`` also matches the
#: pattern for the plain CTC export, so the end-to-end ones have to be tried
#: first. An RNN-T export is three files; its encoder is the one that names it.
GIGAAM_VARIANTS: Final[tuple[tuple[str, str], ...]] = (
    ("v3_e2e_ctc*.onnx", "gigaam-v3-e2e-ctc"),
    ("v3_e2e_rnnt_encoder*.onnx", "gigaam-v3-e2e-rnnt"),
    ("v3_ctc*.onnx", "gigaam-v3-ctc"),
    ("v3_rnnt_encoder*.onnx", "gigaam-v3-rnnt"),
    ("v2_ctc*.onnx", "gigaam-v2-ctc"),
    ("v2_rnnt_encoder*.onnx", "gigaam-v2-rnnt"),
)

#: Quantisations the exports are published in, as they appear in a file name
#: (``v3_ctc.int8.onnx``). ``onnx-asr`` needs to be told which one to glob for,
#: and the folder is the only place that knows.
_QUANTIZATIONS: Final = ("int8", "fp16")

#: Language of every GigaAM export Ayris ships. The multilingual checkpoints exist
#: and are deliberately not offered: Russian is what this engine is here for, and
#: the multilingual ones are both bigger and worse at it.
_LANGUAGES: Final = ("ru",)

#: Longest buffer recognised in one pass. Well under the export's 200 s ceiling,
#: because the ceiling is a crash and memory has already tripled by then.
_MAX_CHUNK_MS: Final = 30_000.0

#: Target length of a piece when a buffer has to be cut.
_CHUNK_MS: Final = 20_000.0

#: How far from the target cut the quietest frame is looked for, either way.
_SEARCH_MS: Final = 2_000.0

#: Frame the search measures the level of. 100 ms is shorter than any word and
#: long enough that one glottal pulse does not look like a pause.
_FRAME_MS: Final = 100.0

#: Shortest buffer the mel front-end accepts at all: below one window it raises
#: ``ValueError: window shape cannot be larger than input array shape``. The
#: ``min_speech_ms`` filter normally keeps such buffers away, but it is a setting
#: and can be turned down to zero.
_MIN_AUDIO_MS: Final = 50.0

#: How long one output character covers, for the end of the last word in a
#: segment: the encoder strides 40 ms, so a timestamp is the start of a 40 ms cell.
_CELL_MS: Final = 40.0

#: Reported when a variant returns text without per-character probabilities. Above
#: ``min_confidence``'s default of 0.4 so nothing is silently discarded, below 1.0
#: because it is not measured. Only the RNN-T exports can land here.
_ASSUMED_CONFIDENCE: Final = 0.75


class GigaAmEngine(SttEngine):
    """GigaAM v3 via onnx-asr, on the CPU."""

    name: ClassVar[str] = "gigaam"
    package: ClassVar[str] = "onnx-asr"
    module: ClassVar[str] = "onnx_asr"
    supports_streaming: ClassVar[bool] = False
    #: 225 MB on disk against a 328 MB peak: the weights plus onnxruntime's
    #: arenas, which are allocated on the first inference and not on load.
    memory_factor: ClassVar[float] = 1.5

    __slots__ = ("_language", "_model", "_numpy", "_variant")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._numpy: Any = None
        self._variant = ""
        self._language = DEFAULT_LANGUAGE

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Russian, and only Russian - see :data:`_LANGUAGES`."""
        return _LANGUAGES

    @property
    def variant(self) -> str:
        """Which GigaAM was found in the folder, e.g. ``gigaam-v3-ctc``.

        Shown in DevTools next to the model name, because "why is it writing
        Telegram in Latin" has exactly one answer and it is the end-to-end export.
        """
        return self._variant

    def load(self, model_path: Path, options: SttOptions) -> None:
        """Open a GigaAM export, deciding the variant from the folder.

        Args:
            model_path: Directory holding the weights and the vocabulary, as
                installed by the model manager.
            options: ``threads`` bounds onnxruntime's intra-op pool; ``language``
                is recorded on the results. ``gpu`` is ignored - see the module
                docstring.

        Raises:
            SttError: ``onnx-asr`` is missing, the path is not a GigaAM folder, or
                onnxruntime refused the model.
        """
        module = self._import()
        numpy = self._import("numpy")
        runtime = self._import("onnxruntime")
        directory = self._resolve_model(model_path)
        variant, quantization = _detect_variant(directory, engine=self.name)

        session = runtime.SessionOptions()
        session.intra_op_num_threads = max(1, options.threads)
        # One graph, one branch: the parallel pool costs threads and buys nothing
        # on a model that is a single chain of operators.
        session.inter_op_num_threads = 1

        started = perf_counter()
        try:
            model = module.load_model(
                variant,
                directory,
                quantization=quantization,
                sess_options=session,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            raise SttError(
                f"gigaam: cannot load {variant} from {directory}: {exc}",
                user_message=(
                    f"Не удалось загрузить модель распознавания «{directory.name}». "
                    f"Проверьте её в настройках голоса."
                ),
            ) from exc

        self._model = model.with_timestamps()
        self._numpy = numpy
        self._variant = variant
        self._language = options.language or DEFAULT_LANGUAGE
        self._options = options
        self._model_path = directory
        _log.info(
            "gigaam: модель %s (%s%s) загружена за %.0f мс, потоков %d",
            directory.name,
            variant,
            f", {quantization}" if quantization else "",
            (perf_counter() - started) * 1000.0,
            session.intra_op_num_threads,
        )

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise a complete buffer, cutting it up first when it is long.

        Raises:
            SttError: If no model is loaded or the decoder failed.
        """
        options = self._require_loaded()
        prepared = self._prepare(audio)
        floor = max(float(options.min_speech_ms), _MIN_AUDIO_MS)
        if prepared.duration_ms < floor or prepared.is_silent():
            return self._empty(prepared.duration_ms)

        waveform = self._numpy.frombuffer(prepared.pcm, dtype="<i2")
        waveform = waveform.astype(self._numpy.float32) / 32768.0
        rate = prepared.sample_rate

        started = perf_counter()
        segments: list[TranscriptSegment] = []
        try:
            for offset in _cut_points(waveform, rate, numpy=self._numpy):
                piece = waveform[offset[0] : offset[1]]
                if piece.size / rate * 1000.0 < _MIN_AUDIO_MS:
                    continue
                result = self._model.recognize(piece, sample_rate=rate)
                segments.extend(_words(result, offset[0] / rate * 1000.0))
        except Exception as exc:
            raise SttError(
                f"gigaam: transcription failed on {self._variant}: {exc}",
                user_message="Ошибка распознавания речи.",
            ) from exc
        inference_ms = (perf_counter() - started) * 1000.0

        text = " ".join(segment.text for segment in segments).strip()
        if not text:
            return self._empty(prepared.duration_ms, inference_ms=inference_ms)
        return TranscriptResult(
            text=text,
            confidence=_confidence(segments),
            segments=tuple(segments),
            language=self._language,
            duration_ms=prepared.duration_ms,
            engine=self.name,
            device=self.device,
            model=self.model_name,
            inference_ms=inference_ms,
        )

    def unload(self) -> None:
        """Drop the model. Safe to call twice.

        Releasing the reference is all there is to do: onnxruntime frees its
        arenas in the session's destructor, which runs on the next collection —
        the worker calls :func:`gc.collect` right after this for that reason.
        """
        self._model = None
        self._numpy = None
        self._variant = ""
        self._options = None
        self._model_path = None

    # ---------------------------------------------------------------- helpers

    def _empty(self, duration_ms: float, *, inference_ms: float = 0.0) -> TranscriptResult:
        """A "nothing was said" result carrying this engine's identity."""
        return TranscriptResult.empty(
            engine=self.name,
            language=self._language,
            duration_ms=duration_ms,
            device=self.device,
            model=self.model_name,
            inference_ms=inference_ms,
        )


def _detect_variant(directory: Path, *, engine: str) -> tuple[str, str | None]:
    """Which GigaAM is in ``directory``, and in which quantisation.

    Args:
        directory: Folder the settings named.
        engine: Engine name, for the error message.

    Returns:
        The name ``onnx-asr`` loads this export by, and the quantisation to ask
        for - ``None`` for an unquantised one.

    Raises:
        SttError: The folder holds no GigaAM export. The message lists what is
            actually in it, because the usual cause is a model for another engine
            selected in the settings.
    """
    if not directory.is_dir():
        raise SttError(
            f"{engine}: {directory} is not a directory",
            user_message=(
                f"Модель «{directory.name}» должна быть папкой с файлами GigaAM. "
                f"Выберите модель в настройках голоса."
            ),
        )
    for pattern, variant in GIGAAM_VARIANTS:
        found = sorted(directory.glob(pattern))
        if found:
            return variant, _quantization(found[0].name)
    contents = ", ".join(sorted(item.name for item in directory.iterdir())[:5]) or "пусто"
    raise SttError(
        f"{engine}: {directory} holds no gigaam export ({contents})",
        user_message=(
            f"Папка «{directory.name}» не похожа на модель GigaAM. "
            f"Проверьте выбранную модель в настройках голоса."
        ),
    )


def _quantization(file_name: str) -> str | None:
    """Quantisation named in a weights file name, or ``None`` for a plain export."""
    parts = file_name.split(".")
    for part in parts:
        if part in _QUANTIZATIONS:
            return part
    return None


def _cut_points(waveform: Any, sample_rate: int, *, numpy: Any) -> tuple[tuple[int, int], ...]:
    """Sample ranges to recognise separately, cut where the audio is quietest.

    One range for anything up to :data:`_MAX_CHUNK_MS`, which is the normal case
    and costs nothing to check. Beyond that, each cut is placed on the quietest
    :data:`_FRAME_MS` frame within :data:`_SEARCH_MS` of the target length, so it
    lands in a pause instead of through a word.
    """
    total = int(waveform.size)
    per_ms = sample_rate / 1000.0
    if total <= int(_MAX_CHUNK_MS * per_ms):
        return ((0, total),)

    chunk = int(_CHUNK_MS * per_ms)
    search = int(_SEARCH_MS * per_ms)
    frame = max(1, int(_FRAME_MS * per_ms))
    ranges: list[tuple[int, int]] = []
    start = 0
    while total - start > chunk + search:
        low = max(start + frame, start + chunk - search)
        high = min(total - frame, start + chunk + search)
        cut = _quietest(waveform, low, high, frame, numpy=numpy) if high > low else start + chunk
        ranges.append((start, cut))
        start = cut
    ranges.append((start, total))
    return tuple(ranges)


def _quietest(waveform: Any, low: int, high: int, frame: int, *, numpy: Any) -> int:
    """Middle of the quietest ``frame``-long window in ``waveform[low:high]``.

    Root mean square rather than peak: a single click in an otherwise silent pause
    would make it look like the loudest place in the window.
    """
    quietest = low
    lowest = float("inf")
    for offset in range(low, high, frame):
        window = waveform[offset : offset + frame]
        level = float(numpy.sqrt(numpy.mean(numpy.square(window))))
        if level < lowest:
            lowest = level
            quietest = offset
    return quietest + frame // 2


def _words(result: Any, offset_ms: float) -> tuple[TranscriptSegment, ...]:
    """Group a timestamped result into one segment per word.

    ``onnx-asr`` returns a token, a start time and a log-probability per *output
    cell* - which for every GigaAM CTC export is a character, spaces included. So
    a word is the run of characters between two spaces, its start is the first
    character's timestamp and its confidence the geometric mean of their
    probabilities. One segment per word is what the Vosk engine produces too, so
    everything downstream sees the same shape from both.

    A variant that returns text without timestamps - an RNN-T export can - gets a
    single segment for the whole piece, which is honest rather than invented.
    """
    text = str(getattr(result, "text", "") or "").strip()
    tokens = _as_list(getattr(result, "tokens", None))
    times = _as_list(getattr(result, "timestamps", None))
    scores = _as_list(getattr(result, "logprobs", None))
    if not text:
        return ()
    if not tokens or len(times) != len(tokens):
        return (
            TranscriptSegment(
                text=text,
                start_ms=offset_ms,
                end_ms=offset_ms,
                confidence=_ASSUMED_CONFIDENCE,
            ),
        )
    if len(scores) != len(tokens):
        scores = []

    built: list[TranscriptSegment] = []
    letters: list[str] = []
    start_ms = 0.0
    end_ms = 0.0
    collected: list[float] = []

    def flush() -> None:
        if not letters:
            return
        built.append(
            TranscriptSegment(
                text="".join(letters),
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=_geometric_mean(collected) if collected else _ASSUMED_CONFIDENCE,
            )
        )
        letters.clear()
        collected.clear()

    for index, token in enumerate(tokens):
        letter = str(token)
        if not letter.strip():
            flush()
            continue
        if letter[0].isspace() or letter.startswith("▁"):
            # A sub-word export marks a word start on the token itself.
            flush()
            letter = letter.lstrip().lstrip("▁")
        at_ms = offset_ms + float(times[index]) * 1000.0
        if not letters:
            start_ms = at_ms
        end_ms = max(end_ms, at_ms + _CELL_MS)
        letters.append(letter)
        if scores:
            collected.append(float(scores[index]))
    flush()
    return tuple(built)


def _as_list(value: Any) -> list[Any]:
    """A vendor attribute as a list, empty when it is absent or is not one."""
    return list(value) if isinstance(value, list) else []


def _geometric_mean(logprobs: Sequence[float]) -> float:
    """Turn per-character log-probabilities into a 0..1 confidence."""
    if not logprobs:
        return 0.0
    mean = sum(logprobs) / len(logprobs)
    if mean >= 0.0:
        return 1.0
    try:
        return max(0.0, min(1.0, math.exp(mean)))
    except OverflowError:  # pragma: no cover - only for absurd inputs
        return 0.0


def _confidence(segments: Iterable[TranscriptSegment]) -> float:
    """Length-weighted mean confidence over the words.

    Weighted by how much was said, so a one-letter interjection cannot outvote a
    sentence. Vosk reports the lowest word confidence instead; both feed the same
    slider, and the mean is the one that does not turn a whole dictation into a
    rejection because of a single mumbled preposition.
    """
    words = list(segments)
    if not words:
        return 0.0
    total = sum(len(word.text) for word in words)
    if total <= 0:  # pragma: no cover - a segment with empty text
        return sum(word.confidence for word in words) / len(words)
    return sum(word.confidence * len(word.text) for word in words) / total
