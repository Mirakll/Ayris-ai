"""Whisper.cpp: the same models as faster-whisper, without CTranslate2.

Optional, and hidden entirely until its binding is installed.  The reason it
exists is disk and dependency weight: ``faster-whisper`` pulls CTranslate2,
several hundred megabytes of it, and on a machine that already has a GGML model
and a compiled ``whisper.cpp`` there is no reason to install a second inference
runtime to use it.  Users who build their own binaries also get Metal, Vulkan and
OpenBLAS backends that CTranslate2 does not offer.

It is not the default and never will be.  ``whisper-cpp-python`` has no wheel for
Windows on PyPI - it is a build-it-yourself binding around a C library - and the
bindings that do exist disagree with each other about the API.  So this engine
loads a model through whichever of the two common shapes the installed package
turns out to have, and both are probed with ``getattr`` rather than assumed:
``Whisper(model_path=...)`` with an OpenAI-shaped ``transcribe`` returning a
dict, and ``Model(...)`` with a ``transcribe`` returning segment objects.  A
binding that matches neither raises a Russian error naming the package rather
than an ``AttributeError`` from inside a worker process.

Because :attr:`~ayris.audio.stt.SttEngine.optional` is set,
:func:`~ayris.audio.stt.base.engine_names` leaves this out of the settings window
on a machine without the library - unlike Vosk and Whisper, which are listed
greyed out so the user can see what they would be installing.  Offering a
build-it-yourself binding as a menu item would be offering a dead end.
"""

from __future__ import annotations

import io
import wave
from time import perf_counter
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.ring_buffer import SAMPLE_WIDTH
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
    from collections.abc import Iterable
    from pathlib import Path

    from ayris.audio.stt.base import AudioBuffer, SttOptions

__all__ = ["WhisperCppEngine"]

_log: Final = get_logger(__name__)

#: Languages, same as the CTranslate2 engine: the models are the same models.
_LANGUAGES: Final = ("ru", "en", "uk", "be", "kk", "de", "fr", "es", "it", "pl", "tr", "zh")

#: Class names the two common bindings expose, tried in order.
_MODEL_FACTORIES: Final = ("Whisper", "Model")

#: Suffix of a GGML/GGUF model file, for the case where the settings name a
#: directory holding one rather than the file itself.
_MODEL_SUFFIXES: Final = (".bin", ".gguf")

#: Confidence reported for a segment, because neither binding exposes a
#: per-token probability.  Above ``min_confidence``'s default of 0.4 so that
#: results are not silently discarded, below 1.0 because it is not measured.
_ASSUMED_CONFIDENCE: Final = 0.75


class WhisperCppEngine(SttEngine):
    """Whisper through a ``whisper.cpp`` binding."""

    name: ClassVar[str] = "whispercpp"
    package: ClassVar[str] = "whisper-cpp-python"
    module: ClassVar[str] = "whisper_cpp_python"
    optional: ClassVar[bool] = True
    supports_streaming: ClassVar[bool] = False
    #: GGML weights are memory-mapped and used in place, so resident memory is
    #: close to the file size plus a decoding buffer.
    memory_factor: ClassVar[float] = 1.2

    __slots__ = ("_language", "_model")

    def __init__(self) -> None:
        super().__init__()
        self._model: Any = None
        self._language = DEFAULT_LANGUAGE

    @property
    def supported_languages(self) -> tuple[str, ...]:
        """Whisper's languages, abridged the same way as the CTranslate2 engine."""
        return _LANGUAGES

    def load(self, model_path: Path, options: SttOptions) -> None:
        """Open a GGML model file.

        Args:
            model_path: The ``.bin``/``.gguf`` file, or a directory holding
                exactly one.
            options: ``threads`` is passed through when the binding accepts it;
                ``gpu`` is not, because which backend a ``whisper.cpp`` build
                uses was decided when it was compiled.

        Raises:
            SttError: The binding is missing, the file is not there, or the
                installed package exposes neither known API.
        """
        module = self._import()
        target = self._resolve_model(self._model_file(model_path))
        factory = self._factory(module)

        started = perf_counter()
        try:
            self._model = factory(model_path=str(target))
        except TypeError:
            # The other binding takes the path positionally.
            try:
                self._model = factory(str(target))
            except Exception as exc:
                self._model = None
                raise SttError(
                    f"whispercpp: cannot open model {target}: {exc}",
                    user_message=(
                        f"Не удалось открыть модель «{target.name}». "
                        f"Проверьте файл модели в настройках голоса."
                    ),
                ) from exc
        except Exception as exc:
            self._model = None
            raise SttError(
                f"whispercpp: cannot open model {target}: {exc}",
                user_message=(
                    f"Не удалось открыть модель «{target.name}». "
                    f"Проверьте файл модели в настройках голоса."
                ),
            ) from exc

        self._language = options.language or DEFAULT_LANGUAGE
        self._options = options
        self._model_path = target
        _log.info(
            "whispercpp: модель %s загружена за %.0f мс",
            target.name,
            (perf_counter() - started) * 1000.0,
        )

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise a complete buffer.

        Raises:
            SttError: If no model is loaded or the binding failed.
        """
        options = self._require_loaded()
        prepared = self._prepare(audio)
        if prepared.duration_ms < options.min_speech_ms or prepared.is_silent():
            return self._empty(prepared.duration_ms)

        started = perf_counter()
        try:
            raw = self._model.transcribe(_as_wav(prepared.pcm), language=self._language)
        except TypeError:
            # A binding that takes samples rather than a file, and no language.
            try:
                raw = self._model.transcribe(prepared.floats())
            except Exception as exc:
                raise SttError(
                    f"whispercpp: transcription failed: {exc}",
                    user_message="Ошибка распознавания речи.",
                ) from exc
        except Exception as exc:
            raise SttError(
                f"whispercpp: transcription failed: {exc}",
                user_message="Ошибка распознавания речи.",
            ) from exc
        inference_ms = (perf_counter() - started) * 1000.0

        segments = _segments(raw)
        text = _text(raw, segments)
        if not text:
            return self._empty(prepared.duration_ms, inference_ms=inference_ms)
        return TranscriptResult(
            text=text,
            confidence=_ASSUMED_CONFIDENCE,
            segments=segments,
            language=self._language,
            duration_ms=prepared.duration_ms,
            engine=self.name,
            model=self.model_name,
            inference_ms=inference_ms,
        )

    def unload(self) -> None:
        """Drop the model.  Safe to call twice."""
        self._model = None
        self._options = None
        self._model_path = None

    # ---------------------------------------------------------------- helpers

    def _empty(self, duration_ms: float, *, inference_ms: float = 0.0) -> TranscriptResult:
        """A "nothing was said" result carrying this engine's identity."""
        return TranscriptResult.empty(
            engine=self.name,
            language=self._language,
            duration_ms=duration_ms,
            model=self.model_name,
            inference_ms=inference_ms,
        )

    def _factory(self, module: Any) -> Any:
        """The model class of whichever binding is installed.

        Raises:
            SttError: If the module exposes neither name, which means the wheel
                that got installed is not one of the two this engine knows.
        """
        for attribute in _MODEL_FACTORIES:
            candidate = getattr(module, attribute, None)
            if callable(candidate):
                return candidate
        raise SttError(
            f"whispercpp: {self.module} exposes none of {', '.join(_MODEL_FACTORIES)}",
            user_message=(
                f"Библиотека {self.package} установлена, но её интерфейс "
                f"не поддерживается. Выберите другой движок распознавания."
            ),
        )

    @staticmethod
    def _model_file(model_path: Path) -> Path:
        """The GGML file inside what the settings named.

        A directory with exactly one model file in it is accepted, because that
        is what unpacking a downloaded archive produces.  Two or more is left
        alone so that :meth:`_resolve_model` reports the directory rather than
        this picking one at random.
        """
        if not model_path.is_dir():
            return model_path
        found = sorted(
            item
            for item in model_path.iterdir()
            if item.is_file() and item.suffix.lower() in _MODEL_SUFFIXES
        )
        return found[0] if len(found) == 1 else model_path


def _as_wav(pcm: bytes) -> io.BytesIO:
    """Wrap mono 16 kHz PCM in a WAV container in memory.

    The OpenAI-shaped binding takes a file object and sniffs the header, so the
    forty-four byte container is the price of not writing a temporary file for
    every phrase.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(STT_SAMPLE_RATE)
        handle.writeframes(pcm)
    buffer.seek(0)
    return buffer


def _segments(raw: Any) -> tuple[TranscriptSegment, ...]:
    """Timed segments out of whichever shape the binding returned.

    Three shapes are handled: a dict with a ``segments`` list of dicts (seconds
    in ``start``/``end``), a list of objects with ``t0``/``t1`` centisecond
    attributes, and anything else, which yields no segments at all - the text is
    still usable, only the word highlighting is not.
    """
    items: Iterable[Any]
    if isinstance(raw, dict):
        candidate = raw.get("segments")
        items = candidate if isinstance(candidate, list) else ()
    elif isinstance(raw, list):
        items = raw
    else:
        items = ()

    built: list[TranscriptSegment] = []
    for item in items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            start_ms = _seconds(item.get("start")) * 1000.0
            end_ms = _seconds(item.get("end")) * 1000.0
        else:
            text = str(getattr(item, "text", "")).strip()
            # whisper.cpp counts in centiseconds; a binding that hands back its
            # own structs passes them through unconverted.
            start_ms = _seconds(getattr(item, "t0", 0)) * 10.0
            end_ms = _seconds(getattr(item, "t1", 0)) * 10.0
        if not text:
            continue
        built.append(
            TranscriptSegment(
                text=text,
                start_ms=start_ms,
                end_ms=max(start_ms, end_ms),
                confidence=_ASSUMED_CONFIDENCE,
            )
        )
    return tuple(built)


def _text(raw: Any, segments: tuple[TranscriptSegment, ...]) -> str:
    """The transcript, preferring the binding's own joined text."""
    if isinstance(raw, dict):
        text = str(raw.get("text", "")).strip()
        if text:
            return text
    elif isinstance(raw, str):
        return raw.strip()
    return " ".join(segment.text for segment in segments).strip()


def _seconds(value: Any) -> float:
    """Read a timestamp as a float without trusting its type."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)
