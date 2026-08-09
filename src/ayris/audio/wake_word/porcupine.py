"""Porcupine: the accurate one, with strings attached.

Picovoice's spotter is measurably better than the free alternatives - lower
false-accept rate at the same miss rate, and a fraction of the CPU - which is
why it is offered at all.  It is not the default, and this module exists in the
shape it does, because of the terms: an account is required, the access key is
per-user, and the free tier covers personal use only.  Nothing here may run
unless the user has explicitly chosen the engine and put a key in the Windows
credential store; the key never lands in a settings file.

Three details differ from every other engine in this package.

**The frame size comes from the library, not from us.**  It is 512 samples in
every build shipped so far, but Porcupine validates it and raises on anything
else, so :attr:`PorcupineEngine.frame_samples` reports what the loaded instance
says rather than a constant.  Before a model is loaded it reports the usual 512
so the manager can size its buffers.

**There are no scores.**  ``process`` returns the index of the keyword that
fired, or -1.  The comparison against a threshold has already happened inside
the library, against the sensitivities handed to ``create``.  So this engine
reports a score of 1.0 on a hit and lets the manager's threshold pass it - and
sets :attr:`WakeWordEngine.reload_on_sensitivity`, because moving a slider here
means rebuilding the instance rather than comparing a number differently.

**Russian needs two files.**  A ``.ppn`` keyword file per phrase *and* a
language parameter file (``porcupine_params_ru.pv``); the built-in English
keywords work with neither.  Both are looked for in the profile's wake model
directory.
"""

from __future__ import annotations

from array import array
from pathlib import Path
from typing import Any, ClassVar, Final

from ayris.audio.wake_word.base import ModelSpec, WakeDetection, WakePhrase, WakeWordEngine
from ayris.core.errors import WakeWordError
from ayris.utils.logger import get_logger

__all__ = ["DEFAULT_FRAME_SAMPLES", "PorcupineEngine"]

_log: Final = get_logger(__name__)

#: What every published build uses, and what the manager assumes until a model
#: is loaded and can be asked.
DEFAULT_FRAME_SAMPLES: Final = 512

#: Keyword file extension.
_KEYWORD_SUFFIX: Final = ".ppn"

#: Language model shipped for Russian.  Without it a Russian ``.ppn`` loads but
#: recognises nothing.
_RU_PARAMS: Final = "porcupine_params_ru.pv"

#: Score reported for a hit.  Porcupine has already decided; anything below the
#: loosest configurable threshold would throw its answer away.
_HIT_SCORE: Final = 1.0


class PorcupineEngine(WakeWordEngine):
    """Keyword spotter backed by ``pvporcupine``.

    Optional in every sense: the package is an extra, the key is the user's, and
    a profile that never selects the engine never imports it.
    """

    name: ClassVar[str] = "porcupine"
    package: ClassVar[str] = "pvporcupine"
    module: ClassVar[str] = "pvporcupine"
    reload_on_sensitivity: ClassVar[bool] = True
    needs_credential: ClassVar[bool] = True

    __slots__ = ("_frame_samples", "_order", "_porcupine")

    def __init__(self) -> None:
        super().__init__()
        self._porcupine: Any = None
        self._frame_samples = DEFAULT_FRAME_SAMPLES
        # Keyword index -> phrase text, in the order handed to create().
        self._order: tuple[str, ...] = ()

    @property
    def frame_samples(self) -> int:
        """What the loaded instance demands; 512 before anything is loaded."""
        return self._frame_samples

    @property
    def sample_rate(self) -> int:
        """What the loaded instance demands; 16 kHz before anything is loaded."""
        porcupine = self._porcupine
        if porcupine is None:
            return super().sample_rate
        rate: int = int(porcupine.sample_rate)
        return rate

    def load(self, spec: ModelSpec) -> None:
        """Create the Porcupine instance for the enabled phrases.

        Raises:
            WakeWordError: If ``pvporcupine`` is not installed, the access key
                is missing or rejected, a keyword file cannot be found, or the
                library refuses the combination.  The user can act on all four,
                so each gets its own message.
        """
        phrases = self._require_phrases(spec)
        if not spec.access_key.strip():
            raise WakeWordError(
                "porcupine: no access key",
                user_message=(
                    "Для движка Porcupine нужен ключ доступа Picovoice. "
                    "Добавьте его в настройках слова активации "
                    "или выберите бесплатный движок openWakeWord."
                ),
            )
        module = self._import()

        keywords = [self._resolve(phrase, spec.models_dir) for phrase in phrases]
        sensitivities = [phrase.sensitivity for phrase in phrases]
        model_path = self._model_path(spec)
        try:
            self._porcupine = module.create(
                access_key=spec.access_key,
                keyword_paths=keywords,
                sensitivities=sensitivities,
                model_path=model_path,
            )
        except Exception as exc:
            self._porcupine = None
            raise WakeWordError(
                f"porcupine: create failed: {exc}",
                user_message=(
                    "Не удалось запустить Porcupine. Проверьте ключ доступа "
                    "и файлы моделей в папке профиля."
                ),
            ) from exc

        self._frame_samples = int(self._porcupine.frame_length)
        self._order = tuple(phrase.text for phrase in phrases)
        self._spec = spec
        _log.info(
            "porcupine loaded %d keyword(s), frame_length=%d",
            len(self._order),
            self._frame_samples,
        )

    def process(self, frame: bytes) -> WakeDetection | None:
        """Run one frame through the library.

        Raises:
            WakeWordError: If no model is loaded, the frame is the wrong size,
                or the library fails mid-stream.
        """
        self._require_loaded()
        self._check_frame(frame)
        samples = self._to_samples(frame)
        try:
            index = int(self._porcupine.process(samples))
        except Exception as exc:
            raise WakeWordError(
                f"porcupine: process failed: {exc}",
                user_message="Ошибка распознавания слова активации.",
            ) from exc
        if index < 0 or index >= len(self._order):
            return None
        phrase = self._order[index]
        return WakeDetection(
            phrase=phrase,
            score=_HIT_SCORE,
            engine=self.name,
            scores={phrase: _HIT_SCORE},
        )

    def unload(self) -> None:
        """Release the native handle.  Safe to call twice."""
        porcupine = self._porcupine
        self._porcupine = None
        self._order = ()
        self._spec = None
        if porcupine is None:
            return
        try:
            porcupine.delete()
        except Exception as exc:  # pragma: no cover - a failing free() is fatal anyway
            # Never propagate: unload runs while the worker is already shutting
            # down, and raising here would mask whatever sent it there.
            _log.warning("porcupine: delete failed: %s", exc)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _to_samples(frame: bytes) -> list[int]:
        """Unpack a frame into the ``int16`` list the library expects.

        ``array`` rather than ``numpy``: the C binding iterates the sequence
        either way, and this keeps the engine usable on a profile where the
        scientific stack was never installed.
        """
        samples = array("h")
        samples.frombytes(frame)
        return samples.tolist()

    def _resolve(self, phrase: WakePhrase, models_dir: Path | None) -> str:
        """Find the ``.ppn`` file for one phrase.

        Raises:
            WakeWordError: If it is not there.  Porcupine cannot be trained
                locally - the file comes from the Picovoice console - so the
                message says where to put it rather than how to make it.
        """
        explicit = phrase.engine_model.strip()
        candidates: list[Path] = []
        if explicit:
            candidates.append(Path(explicit))
            if models_dir is not None:
                candidates.append(models_dir / explicit)
        elif models_dir is not None:
            stem = phrase.text.replace(" ", "_")
            candidates.append(models_dir / f"{stem}{_KEYWORD_SUFFIX}")
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        where = str(models_dir) if models_dir is not None else "папке моделей"
        raise WakeWordError(
            f"porcupine: no keyword file for phrase {phrase.text!r} in {where}",
            user_message=(
                f"Не найден файл ключевого слова (.ppn) для «{phrase.text}». "
                f"Скачайте его в консоли Picovoice и положите в {where}."
            ),
        )

    @staticmethod
    def _model_path(spec: ModelSpec) -> str | None:
        """The language parameter file, or ``None`` for the built-in English one.

        Ayris is a Russian assistant, so the Russian file is looked for by name
        in the profile.  Its absence is not an error here: a user experimenting
        with an English keyword should get Porcupine's own message about the
        keyword's language, which is clearer than a guess made in advance.
        """
        explicit = spec.option("model_path")
        if explicit:
            return explicit
        if spec.models_dir is not None:
            candidate = spec.models_dir / _RU_PARAMS
            if candidate.is_file():
                return str(candidate)
        return None
