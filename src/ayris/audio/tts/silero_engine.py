"""Silero: the fallback voice, and the one that handles stress marks.

Silero v4 ships five Russian speakers in a single 60 MB TorchScript package and
runs on the CPU at several times real time. It is here for two reasons Piper does
not cover: it accepts SSML-ish stress marks, which matters for the Russian
homographs an assistant runs into constantly («за́мок» / «замо́к»), and it needs no
espeak-ng, so it works when ``piper-phonemize`` will not install.

**The model is one file.** ``v4_ru.pt``, holding all five speakers - so switching
voices is a speaker-name change with no reload, which :meth:`SileroTtsEngine.load`
takes advantage of: it only rebuilds the TorchScript object when the *file*
changed.

**Where the model comes from.** ``models/tts/v4_ru.pt`` in the profile, loaded
with :func:`torch.package`. This engine never downloads: the model manager in
section 14 owns fetching files, and a synthesis engine that reached for the
network mid-sentence would turn a missing file into a thirty-second stall instead
of a sentence telling the user what to download.

**Rate and pitch.** Silero returns a float32 tensor at 8, 24 or 48 kHz, converted
to ``int16`` here because that is what the rest of Ayris carries. Speed and pitch
have no model parameters, so both are done on the samples: resampling by
``1/speed`` and replaying at the original rate stretches time, and resampling
without changing the playback rate shifts pitch. Both use
:class:`~ayris.audio.capture.Resampler`, already in the project and already
tested, rather than a second interpolation routine.

``torch`` is imported inside :meth:`load` and nowhere else. It is 200 MB of wheel
that a Piper user has no reason to have, and it is not on the CI runner.
"""

from __future__ import annotations

import logging
import sys
from array import array
from pathlib import Path
from typing import Any, ClassVar, Final

from ayris.audio.capture import Resampler
from ayris.audio.tts.base import SAMPLE_WIDTH, AudioChunk, TtsEngine, TtsOptions, VoiceSpec
from ayris.core.errors import TtsError
from ayris.core.paths import get_paths

__all__ = [
    "DEFAULT_MODEL_FILE",
    "SILERO_RATES",
    "SILERO_SAMPLE_RATE",
    "SILERO_VOICES",
    "SileroTtsEngine",
]

_log = logging.getLogger(__name__)

#: Rate Silero v4 is asked for. 48 kHz is its best quality and the one every
#: speaker supports; the player resamples down if the device wants less.
SILERO_SAMPLE_RATE: Final = 48000

#: Rates the model accepts. Anything else has to be resampled after the fact.
SILERO_RATES: Final = (8000, 24000, 48000)

#: Model file the engine looks for in ``models/tts``.
DEFAULT_MODEL_FILE: Final = "v4_ru.pt"

#: Speakers in the Russian package, with the names the settings show. ``random``
#: is deliberately left out: an assistant whose voice changes between sentences
#: reads as a bug.
SILERO_VOICES: Final[tuple[tuple[str, str], ...]] = (
    ("aidar", "Айдар (мужской)"),
    ("baya", "Бая (женский)"),
    ("kseniya", "Ксения (женский)"),
    ("xenia", "Ксения v4 (женский)"),
    ("eugene", "Евгений (мужской)"),
)

#: Full scale for ``int16``, for the float conversion.
_INT16_SCALE: Final = 32767.0
_INT16_MAX: Final = 32767
_INT16_MIN: Final = -32768

#: Speed and pitch closer to neutral than this are treated as neutral: the
#: resampling pass costs real time on a long phrase and 1% is inaudible.
_NEUTRAL_EPSILON: Final = 0.02


class SileroTtsEngine(TtsEngine):
    """Silero v4 Russian synthesis on the CPU."""

    name: ClassVar[str] = "silero"
    package: ClassVar[str] = "torch"
    module: ClassVar[str] = "torch"
    optional: ClassVar[bool] = False
    #: TorchScript keeps the weights plus its own intermediate buffers; measured
    #: near 4x the 60 MB package during the first synthesis.
    memory_factor: ClassVar[float] = 4.0
    native_sample_rate: ClassVar[int] = SILERO_SAMPLE_RATE

    __slots__ = ("_model", "_model_path", "_speaker", "_threads", "_torch")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any | None = None
        self._torch: Any | None = None
        self._model_path = ""
        self._speaker = ""
        self._threads = 0

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Russian only: this is the ``v4_ru`` package."""
        return ("ru",)

    # ----------------------------------------------------------- enumeration

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        """The five built-in speakers, pointed at the model file.

        Returns them even when the file is absent: the settings should show what
        this engine offers, and the download prompt belongs to the model manager,
        not to an empty combo box.
        """
        root = directory if directory is not None else get_paths().tts_models_dir
        model = root / DEFAULT_MODEL_FILE
        return tuple(
            VoiceSpec(
                engine=cls.name,
                voice_id=speaker,
                path=str(model),
                language="ru",
                display_name=label,
                sample_rate=SILERO_SAMPLE_RATE,
            )
            for speaker, label in SILERO_VOICES
        )

    # ------------------------------------------------------------- lifecycle

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Load the TorchScript package, or just switch speaker.

        Raises:
            TtsError: ``torch`` is missing, the model file is absent, the
                speaker is unknown, or TorchScript refused the file.
        """
        speaker = voice.voice_id or SILERO_VOICES[0][0]
        known = {name for name, _ in SILERO_VOICES}
        if speaker not in known:
            raise TtsError(
                f"silero: unknown speaker {speaker!r}",
                user_message=(
                    f"Голос «{speaker}» не входит в набор Silero. "
                    f"Доступны: {', '.join(sorted(known))}."
                ),
            )

        model_path = _resolve_model_path(voice)
        if not model_path.is_file():
            raise TtsError(
                f"silero: model {model_path} not found",
                user_message=(
                    f"Модель Silero не найдена:\n{model_path}\n"
                    "Скачайте её в настройках, раздел «Модели»."
                ),
            )

        threads = max(1, options.threads)
        if self._model is not None and self._model_path == str(model_path):
            # Same file, different speaker: nothing to reload.
            self._apply_threads(threads)
            self._speaker = speaker
            self._voice = _spec_for(speaker, model_path)
            self._options = options
            return

        torch = self._import()
        self.unload()
        try:
            torch.set_num_threads(threads)
            model = torch.package.PackageImporter(str(model_path)).load_pickle(
                "tts_models", "model"
            )
            model.to(torch.device("cpu"))
        except Exception as exc:  # torch raises many unrelated types
            raise TtsError(
                f"silero: cannot load {model_path.name}: {exc}",
                user_message=(
                    f"Не удалось загрузить модель Silero «{model_path.name}». "
                    "Возможно, файл повреждён — скачайте его заново."
                ),
            ) from exc

        self._torch = torch
        self._model = model
        self._model_path = str(model_path)
        self._threads = threads
        self._speaker = speaker
        self._voice = _spec_for(speaker, model_path)
        self._options = options
        _log.info("silero: модель загружена, голос %s, %d Гц", speaker, SILERO_SAMPLE_RATE)

    def unload(self) -> None:
        """Drop the model. Safe to call twice; never raises."""
        self._model = None
        self._torch = None
        self._model_path = ""
        self._speaker = ""
        self._threads = 0
        self._voice = None

    # ------------------------------------------------------------- synthesis

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        """Run one phrase through the model and shape it.

        Raises:
            TtsError: No model loaded, or TorchScript failed. A phrase Silero
                cannot phonemize at all - punctuation only - comes back empty
                rather than as an error.
        """
        model = self._model
        if model is None:
            self._require_loaded()  # raises with the right message
            return AudioChunk(b"", SILERO_SAMPLE_RATE)

        try:
            tensor = model.apply_tts(
                text=text,
                speaker=self._speaker,
                sample_rate=SILERO_SAMPLE_RATE,
            )
        except ValueError:
            # Raised for input with no pronounceable content. Not an error: the
            # caller gets silence and moves to the next sentence.
            _log.debug("silero: нечего произносить в %r", text[:40])
            return AudioChunk(b"", SILERO_SAMPLE_RATE)
        except Exception as exc:
            raise TtsError(
                f"silero: synthesis failed: {exc}",
                user_message="Не удалось озвучить ответ: движок Silero вернул ошибку.",
            ) from exc

        pcm = _tensor_to_pcm(tensor)
        return _shape(AudioChunk(pcm, SILERO_SAMPLE_RATE, 1), speed, pitch)

    def _apply_threads(self, threads: int) -> None:
        """Re-apply the thread count when only the speaker changed."""
        torch = self._torch
        if torch is None or threads == self._threads:
            return
        try:
            torch.set_num_threads(threads)
        except Exception as exc:  # a thread-count change must never fail speech
            _log.debug("silero: не удалось сменить число потоков: %s", exc)
            return
        self._threads = threads


# ----------------------------------------------------------------- shaping


def _shape(chunk: AudioChunk, speed: float, pitch: float) -> AudioChunk:
    """Apply speed and pitch to finished samples.

    Order matters. Pitch first: resample by ``pitch`` and keep the declared rate,
    which shifts frequency and duration together. Then speed, which corrects the
    duration back by resampling by ``1/speed`` - so a pitch change alone does not
    also change how long the phrase takes, and a speed change alone does not
    detune it.
    """
    if chunk.empty:
        return chunk
    shaped = chunk
    if abs(pitch - 1.0) > _NEUTRAL_EPSILON:
        shaped = shaped.with_pcm(_resample(shaped.pcm, shaped.sample_rate, pitch))
    if abs(speed - 1.0) > _NEUTRAL_EPSILON:
        shaped = shaped.with_pcm(_resample(shaped.pcm, shaped.sample_rate, 1.0 / speed))
    return shaped


def _resample(pcm: bytes, rate: int, factor: float) -> bytes:
    """Resample by ``factor`` while claiming the same rate.

    ``factor`` above 1.0 produces fewer samples - shorter and higher when played
    at ``rate``. Reuses :class:`~ayris.audio.capture.Resampler` so there is one
    interpolation routine in the project rather than two.
    """
    source = max(1, int(round(rate * factor)))
    if source == rate:
        return pcm
    return Resampler(source, rate).process(pcm)


def _tensor_to_pcm(tensor: Any) -> bytes:
    """Convert a float32 torch tensor in -1..1 to little-endian ``int16``.

    Done with :mod:`array` rather than NumPy: ``tensor.tolist()`` is already a
    Python list by the time it is here, and adding NumPy to this path would mean
    a Silero user who never touches Piper still pays for it.
    """
    try:
        values = tensor.squeeze().tolist()
    except Exception as exc:  # not a tensor: a bug, but not one worth a crash
        raise TtsError(
            f"silero: unexpected model output {type(tensor).__name__}: {exc}",
            user_message="Не удалось озвучить ответ: движок Silero вернул неожиданный формат.",
        ) from exc
    if isinstance(values, float):
        values = [values]

    samples: array[int] = array("h", bytes(len(values) * SAMPLE_WIDTH))
    for index, value in enumerate(values):
        scaled = int(value * _INT16_SCALE)
        samples[index] = min(max(scaled, _INT16_MIN), _INT16_MAX)
    return _to_little_endian(samples)


def _to_little_endian(samples: array[int]) -> bytes:
    """``array('h')`` is native-endian; the wire format is not."""
    if sys.byteorder != "little":
        samples = array("h", samples)
        samples.byteswap()
    return samples.tobytes()


def _resolve_model_path(voice: VoiceSpec) -> Path:
    """Where the ``.pt`` package should be.

    The spec's ``path`` may point at the model file or at the directory holding
    it - a config written by hand does both - and an empty path means the
    profile's default location.
    """
    if not voice.path:
        return get_paths().tts_models_dir / DEFAULT_MODEL_FILE
    candidate = Path(voice.path)
    if candidate.is_dir():
        return candidate / DEFAULT_MODEL_FILE
    return candidate


def _spec_for(speaker: str, model_path: Path) -> VoiceSpec:
    """Canonical spec for a loaded speaker, with its display name filled in."""
    label = next((name for key, name in SILERO_VOICES if key == speaker), speaker)
    return VoiceSpec(
        engine=SileroTtsEngine.name,
        voice_id=speaker,
        path=str(model_path),
        language="ru",
        display_name=label,
        sample_rate=SILERO_SAMPLE_RATE,
    )
