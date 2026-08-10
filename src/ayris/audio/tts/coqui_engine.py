"""Coqui XTTS v2: voice cloning, for the user who asked for it and has the GPU.

XTTS v2 clones a voice from six seconds of reference audio and speaks Russian in
it. That is the one thing neither Piper nor Silero can do, and it is why this
engine exists. What it costs: a 1.8 GB model, ``coqui-tts`` with its whole
dependency tree, and a GPU - on a CPU it synthesizes at roughly real time, which
means a two-sentence answer takes as long to produce as to hear and the
assistant feels broken.

**So it is optional in the strict sense.** ``optional = True`` keeps it out of
the settings list until ``coqui-tts`` is actually installed, rather than offering
a choice that fails when selected. It lives in ``[project.optional-dependencies]
tts-extra``, not in the base install, and it is not in ``requirements-ci.txt`` -
CI would spend ten minutes installing PyTorch to run tests that skip.

**A voice is a WAV file.** Six to thirty seconds of clean speech in
``models/tts/speakers/``; :meth:`CoquiTtsEngine.voices` lists what is there. The
model itself is a directory ``models/tts/xtts_v2`` holding the checkpoint and its
config, downloaded by the model manager, never by this engine.

**The licence is the user's problem to accept, and Ayris says so.** XTTS v2 ships
under the Coqui Public Model License, which restricts commercial use, and cloning
a voice from a recording of a person who did not agree to it is a decision the
software should not make quietly. The log line on load names both.

Everything from ``TTS`` is imported inside :meth:`load`.
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
    "MODEL_DIR_NAME",
    "SPEAKERS_DIR_NAME",
    "XTTS_SAMPLE_RATE",
    "CoquiTtsEngine",
]

_log = logging.getLogger(__name__)

#: What XTTS v2 produces.
XTTS_SAMPLE_RATE: Final = 24000

#: Model directory inside ``models/tts``.
MODEL_DIR_NAME: Final = "xtts_v2"

#: Reference-sample directory inside ``models/tts``.
SPEAKERS_DIR_NAME: Final = "speakers"

#: Reference clips the engine will consider.
_SAMPLE_SUFFIXES: Final = (".wav", ".flac", ".mp3", ".ogg")

_INT16_SCALE: Final = 32767.0
_INT16_MAX: Final = 32767
_INT16_MIN: Final = -32768

#: Below this the resampling pass is not worth its cost.
_NEUTRAL_EPSILON: Final = 0.02


class CoquiTtsEngine(TtsEngine):
    """XTTS v2 voice cloning through ``coqui-tts``."""

    name: ClassVar[str] = "xtts"
    package: ClassVar[str] = "coqui-tts"
    module: ClassVar[str] = "TTS"
    #: Hidden from the settings until the dependency is installed - see the
    #: module docstring.
    optional: ClassVar[bool] = True
    #: 1.8 GB of weights plus the inference graph.
    memory_factor: ClassVar[float] = 2.0
    native_sample_rate: ClassVar[int] = XTTS_SAMPLE_RATE

    __slots__ = ("_device", "_model", "_reference")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any | None = None
        self._reference = ""
        self._device = "cpu"

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """XTTS v2 is multilingual; these are the ones Ayris cares about."""
        return ("ru", "en")

    @property
    def device(self) -> str:
        """Where inference is running."""
        return self._device

    # ----------------------------------------------------------- enumeration

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        """Reference clips found in ``models/tts/speakers``."""
        root = directory if directory is not None else get_paths().tts_models_dir
        speakers = root / SPEAKERS_DIR_NAME
        try:
            samples = sorted(
                item
                for item in speakers.iterdir()
                if item.is_file() and item.suffix.lower() in _SAMPLE_SUFFIXES
            )
        except OSError:
            # A missing directory is the normal case for a user who never set
            # this up; there is nothing to report and nothing to fix.
            return ()
        return tuple(
            VoiceSpec(
                engine=cls.name,
                voice_id=sample.stem,
                path=str(sample),
                language="ru",
                display_name=f"{sample.stem} (клон)",
                sample_rate=XTTS_SAMPLE_RATE,
            )
            for sample in samples
        )

    # ------------------------------------------------------------- lifecycle

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Load the checkpoint and remember the reference clip.

        Raises:
            TtsError: ``coqui-tts`` is missing, the model directory or the
                reference clip is absent, or the checkpoint failed to load.
        """
        model_dir = _resolve_model_dir()
        if not model_dir.is_dir():
            raise TtsError(
                f"xtts: model directory {model_dir} not found",
                user_message=(
                    f"Модель XTTS не найдена:\n{model_dir}\n"
                    "Скачайте её в настройках, раздел «Модели»."
                ),
            )
        reference = _resolve_reference(voice)
        if not reference.is_file():
            raise TtsError(
                f"xtts: reference sample {reference} not found",
                user_message=(
                    f"Образец голоса не найден:\n{reference}\n"
                    "Положите запись 6–30 секунд в папку models/tts/speakers."
                ),
            )

        api = self._import("TTS.api")
        self.unload()
        device = _pick_device(options.gpu)
        try:
            model = api.TTS(model_path=str(model_dir), config_path=str(model_dir / "config.json"))
            model.to(device)
        except Exception as exc:  # coqui raises many unrelated types
            raise TtsError(
                f"xtts: cannot load model from {model_dir}: {exc}",
                user_message=(
                    "Не удалось загрузить модель XTTS. "
                    "Проверьте, что файлы скачаны целиком и хватает видеопамяти."
                ),
            ) from exc

        self._model = model
        self._device = device
        self._reference = str(reference)
        self._voice = VoiceSpec(
            engine=self.name,
            voice_id=voice.voice_id or reference.stem,
            path=str(reference),
            language=voice.language or "ru",
            display_name=voice.display_name or f"{reference.stem} (клон)",
            sample_rate=XTTS_SAMPLE_RATE,
        )
        self._options = options
        _log.info(
            "xtts: модель загружена на %s, образец голоса %s. "
            "Модель распространяется по Coqui Public Model License: "
            "коммерческое использование ограничено, а клонировать чужой голос "
            "можно только с согласия его владельца",
            device,
            reference.name,
        )

    def unload(self) -> None:
        """Drop the model. Safe to call twice; never raises."""
        self._model = None
        self._reference = ""
        self._device = "cpu"
        self._voice = None

    # ------------------------------------------------------------- synthesis

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        """Clone one phrase.

        ``speed`` goes to the model, which has a real speed parameter; ``pitch``
        is done on the samples, as in :mod:`ayris.audio.tts.silero_engine`.

        Raises:
            TtsError: No model loaded, or inference failed.
        """
        model = self._model
        if model is None:
            self._require_loaded()  # raises with the right message
            return AudioChunk(b"", XTTS_SAMPLE_RATE)

        voice = self._require_loaded()
        try:
            samples = model.tts(
                text=text,
                speaker_wav=self._reference,
                language=voice.language or "ru",
                speed=speed,
            )
        except Exception as exc:
            raise TtsError(
                f"xtts: synthesis failed: {exc}",
                user_message="Не удалось озвучить ответ: движок XTTS вернул ошибку.",
            ) from exc

        chunk = AudioChunk(_floats_to_pcm(samples), XTTS_SAMPLE_RATE, 1)
        if chunk.empty or abs(pitch - 1.0) <= _NEUTRAL_EPSILON:
            return chunk
        source = max(1, int(round(XTTS_SAMPLE_RATE * pitch)))
        return chunk.with_pcm(Resampler(source, XTTS_SAMPLE_RATE).process(chunk.pcm))


# ------------------------------------------------------------------ helpers


def _resolve_model_dir() -> Path:
    """Where the checkpoint lives."""
    return get_paths().tts_models_dir / MODEL_DIR_NAME


def _resolve_reference(voice: VoiceSpec) -> Path:
    """The reference clip: an explicit path, or a name under ``speakers``."""
    if voice.path:
        return Path(voice.path)
    speakers = get_paths().tts_models_dir / SPEAKERS_DIR_NAME
    named = speakers / voice.voice_id
    if named.is_file():
        return named
    return speakers / f"{voice.voice_id}.wav"


def _pick_device(preference: str) -> str:
    """Resolve ``auto``/``cuda``/``cpu`` against what the machine has.

    ``auto`` means CUDA when torch reports a card. The import is guarded because
    ``coqui-tts`` can in principle be present without a working torch build, and
    a failure here should mean "run on the CPU", not "cannot speak".
    """
    if preference == "cpu":
        return "cpu"
    try:
        import torch  # deferred: see the module docstring

        if torch.cuda.is_available():
            return "cuda"
    except Exception as exc:  # любой сбой здесь означает «видеокарты нет»
        _log.debug("xtts: CUDA недоступна: %s", exc)
    if preference == "cuda":
        _log.warning("xtts: запрошена CUDA, но видеокарта недоступна — синтез пойдёт на CPU")
    return "cpu"


def _floats_to_pcm(samples: Any) -> bytes:
    """Convert Coqui's float list in -1..1 to little-endian ``int16``.

    ``TTS.api.TTS.tts`` returns a plain Python list, so no NumPy is needed here
    even though the library itself uses it internally.
    """
    try:
        values = list(samples)
    except TypeError as exc:
        raise TtsError(
            f"xtts: unexpected model output {type(samples).__name__}: {exc}",
            user_message="Не удалось озвучить ответ: движок XTTS вернул неожиданный формат.",
        ) from exc

    out: array[int] = array("h", bytes(len(values) * SAMPLE_WIDTH))
    for index, value in enumerate(values):
        scaled = int(float(value) * _INT16_SCALE)
        out[index] = min(max(scaled, _INT16_MIN), _INT16_MAX)
    if sys.byteorder != "little":
        out.byteswap()
    return out.tobytes()
