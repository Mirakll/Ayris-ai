"""openWakeWord: the engine Ayris uses out of the box.

Chosen as the default for one reason - it is the only one of the three that
costs nothing, needs no registration, and can be trained on a new phrase by the
user.  A model is a small ONNX file that scores one 80 ms block at a time
against a phrase it was trained on; running four of them at once is a few
percent of a core.

Two things about the library shape the code below.

It insists on 1280-sample blocks.  Not "at least" and not "about": the
melspectrogram front-end and the shared embedding model have that length baked
into their graphs.  The manager already owns a :class:`~ayris.audio.vad.FrameSplitter`
for exactly this, so the engine only declares the number and checks it.

It carries state across blocks.  A wake word is longer than 80 ms, so the model
keeps a rolling buffer of embeddings; the score for a block depends on the ones
before it.  That is why :meth:`OpenWakeWordEngine.process` must be fed every
block in order, and why the model is reset when the stream restarts rather than
rebuilt.

**Weights are not in the repository.**  They are tens of megabytes of binary and
they belong to the profile, not to the source tree - so this engine can raise at
load time on a perfectly healthy install, and the message has to be the one that
tells the user to download the model rather than a stack trace about a missing
file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.wake_word.base import (
    MIN_THRESHOLD,
    ModelSpec,
    WakeDetection,
    WakePhrase,
    WakeWordEngine,
)
from ayris.core.errors import WakeWordError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["FRAME_SAMPLES", "OpenWakeWordEngine"]

_log: Final = get_logger(__name__)

#: 80 ms at 16 kHz.  Fixed by the model graph, not a tuning choice.
FRAME_SAMPLES: Final = 1280

#: Extensions a model file can have, in the order they are looked for.  ONNX
#: first: on Windows the ONNX runtime ships a wheel for every supported Python,
#: while ``tflite-runtime`` has not been published for Windows in years.
_MODEL_SUFFIXES: Final = (".onnx", ".tflite")

#: Runtime used when the profile does not ask for another one.
_DEFAULT_FRAMEWORK: Final = "onnx"


class OpenWakeWordEngine(WakeWordEngine):
    """Keyword spotter backed by ``openwakeword``.

    One model per phrase variant, all scored on every block.  Scores are raw:
    the thresholds live in :class:`~ayris.audio.wake_word.manager.WakeWordDetector`,
    which is what lets a user change a variant's sensitivity without the model
    being reloaded.
    """

    name: ClassVar[str] = "openwakeword"
    package: ClassVar[str] = "openwakeword"
    module: ClassVar[str] = "openwakeword"

    __slots__ = ("_keys", "_model", "_numpy")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._numpy: Any = None
        # Model key as the library reports it -> the phrase the user typed.
        self._keys: dict[str, str] = {}

    @property
    def frame_samples(self) -> int:
        """1280, and no other value works."""
        return FRAME_SAMPLES

    def load(self, spec: ModelSpec) -> None:
        """Build one model per enabled phrase.

        Raises:
            WakeWordError: If the library is missing, no model file matches a
                phrase, or the shared feature extractors have not been
                downloaded.  Each case gets its own Russian message: they need
                different things from the user.
        """
        phrases = self._require_phrases(spec)
        module = self._import("openwakeword.model")
        self._numpy = self._import("numpy")

        paths = {phrase.text: self._resolve(phrase, spec.models_dir) for phrase in phrases}
        framework = spec.option("inference_framework", _DEFAULT_FRAMEWORK)
        try:
            self._model = module.Model(
                wakeword_models=list(paths.values()),
                inference_framework=framework,
            )
        except Exception as exc:
            # The library raises whatever its runtime raises - a bare OSError
            # from onnxruntime, a ValueError from its own path handling.  All of
            # them mean the same thing to the user.
            self._model = None
            raise WakeWordError(
                f"openwakeword: cannot build the model set {list(paths.values())}: {exc}",
                user_message=(
                    "Не удалось загрузить модель слова активации. "
                    "Проверьте, что файлы модели скачаны в папку профиля, "
                    "или выберите другой движок в настройках."
                ),
            ) from exc

        self._keys = self._map_keys(paths)
        self._spec = spec
        _log.info(
            "openwakeword loaded %d phrase model(s) via %s",
            len(self._keys),
            framework,
        )

    def process(self, frame: bytes) -> WakeDetection | None:
        """Score one 1280-sample block against every loaded phrase.

        Raises:
            WakeWordError: If no model is loaded or the block is the wrong size.
        """
        self._require_loaded()
        self._check_frame(frame)
        samples = self._numpy.frombuffer(frame, dtype=self._numpy.int16)
        raw: Mapping[str, float] = self._model.predict(samples)

        scores: dict[str, float] = {}
        best_phrase = ""
        best_score = 0.0
        for key, value in raw.items():
            phrase = self._keys.get(key)
            if phrase is None:
                # A model the user added to the directory by hand, or one of the
                # library's bundled extras.  Scoring it costs nothing, acting on
                # it would fire on a word nobody configured.
                continue
            score = float(value)
            scores[phrase] = score
            if score > best_score:
                best_phrase, best_score = phrase, score

        if not best_phrase or best_score < MIN_THRESHOLD:
            return None
        return WakeDetection(
            phrase=best_phrase,
            score=best_score,
            engine=self.name,
            scores=scores,
        )

    def unload(self) -> None:
        """Drop the models.  Safe to call on an engine that never loaded."""
        self._model = None
        self._keys = {}
        self._spec = None

    def reset(self) -> None:
        """Forget the rolling embedding buffer.

        Called when capture restarts.  Without it the first blocks of the new
        stream are scored against the tail of the old one, which is how a
        wake word "fires" the moment the microphone is switched.
        """
        model = self._model
        if model is None:
            return
        reset = getattr(model, "reset", None)
        if callable(reset):
            reset()

    # ------------------------------------------------------------------ helpers

    def _resolve(self, phrase: WakePhrase, models_dir: Path | None) -> str:
        """Find the model file for one phrase.

        Three ways, in order: an explicit :attr:`WakePhrase.engine_model` that
        is a path; a bare name, handed to the library so its bundled models
        (``alexa``, ``hey_jarvis``) keep working; or a file next to the other
        wake models named after the phrase.

        Raises:
            WakeWordError: If nothing matches, naming the phrase and the folder
                that was searched - a user who trained a model has to be told
                where to put it.
        """
        explicit = phrase.engine_model.strip()
        if explicit:
            candidate = self._as_path(explicit, models_dir)
            if candidate is not None:
                return str(candidate)
            if not explicit.endswith(_MODEL_SUFFIXES) and "/" not in explicit:
                # A bundled model name.  The library resolves it itself and
                # raises a clear error of its own if it is unknown.
                return explicit
            raise WakeWordError(
                f"openwakeword: model {explicit!r} for phrase {phrase.text!r} not found",
                user_message=(
                    f"Файл модели «{explicit}» для слова «{phrase.text}» не найден. "
                    f"Укажите правильный путь в настройках слова активации."
                ),
            )

        if models_dir is not None:
            stem = phrase.text.replace(" ", "_")
            for suffix in _MODEL_SUFFIXES:
                candidate = models_dir / f"{stem}{suffix}"
                if candidate.is_file():
                    return str(candidate)
        where = str(models_dir) if models_dir is not None else "папке моделей"
        raise WakeWordError(
            f"openwakeword: no model for phrase {phrase.text!r} in {where}",
            user_message=(
                f"Не найдена модель для слова «{phrase.text}». "
                f"Скачайте или обучите модель и положите её в {where}."
            ),
        )

    @staticmethod
    def _as_path(value: str, models_dir: Path | None) -> Path | None:
        """Interpret ``value`` as a file, absolute or relative to the profile."""
        direct = Path(value)
        if direct.is_file():
            return direct
        if models_dir is not None:
            relative = models_dir / value
            if relative.is_file():
                return relative
        return None

    @staticmethod
    def _map_keys(paths: Mapping[str, str]) -> dict[str, str]:
        """Map the library's model keys back to the phrases they came from.

        openWakeWord keys its predictions by the file stem, or by the name in
        the model's own metadata.  The stem is what it uses for a model loaded
        from a path, which is every model Ayris loads.
        """
        return {Path(path).stem: phrase for phrase, path in paths.items()}
