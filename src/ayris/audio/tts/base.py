"""The contract every TTS engine implements.

A TTS engine answers one question - "how does this text sound" - and owns exactly
one thing while doing it: its model or API connection. It does not decide when to
speak, it does not manage the audio queue, and it does not know whether the user
is listening. The lifecycle - when to load, when to cache, when to give the memory
back - lives in :mod:`ayris.workers.tts_worker`.

**Why an ABC and not three functions.** Section 4 of the specification asks for
Piper *and* Silero *and* an optional Coqui XTTS, chosen in the settings, plus
cloud providers later. They agree on almost nothing else: Piper streams ONNX
frames, Silero returns a complete tensor, XTTS clones voices from a sample, cloud
APIs measure in characters billed. What they do agree on is
:meth:`TtsEngine.synthesize`, and :class:`AudioChunk` is the shape the rest of
Ayris is allowed to know about.

**Voice, speed and pitch are per call.** :meth:`TtsEngine.synthesize` takes all
three, because the caller that knows the text is the caller that knows how to say
it: a confirmation is spoken at the configured rate, a dictated read-back slower.
The base class resolves them - reloading only when the voice actually changed -
and hands the resolved values to :meth:`TtsEngine._synthesize`, so an engine
implements one method and gets voice switching for free.

**Streaming by sentence.** A long answer should start sounding before the last
word is synthesized. :meth:`synthesize_stream` splits text into sentences and
yields one :class:`AudioChunk` per sentence, so the player can start the first
chunk while the second is still being generated. Engines that can stream more
finely than that override it.

**Voice is a spec, not a name.** Different engines identify voices differently:
Piper uses ONNX file names, Silero has a fixed set of IDs, Coqui needs a speaker
sample. :class:`VoiceSpec` unifies them, and :meth:`TtsEngine.voices` enumerates
what a directory offers, so the settings window can fill its combo box without
knowing which engine is selected.

**Missing libraries are a message, not a crash.** Every engine imports its vendor
package inside :meth:`TtsEngine.load`, never at module import time. A user on
Piper must not need ``torch`` on disk, and the test suite has to stay collectable
on a runner where XTTS was never installed. :meth:`TtsEngine.available` answers
the settings window's question without importing anything at all.
"""

from __future__ import annotations

import importlib
import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.core.errors import TtsError
from ayris.core.models import JsonObject

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

__all__ = [
    "DEFAULT_PITCH",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SPEED",
    "ENGINE_ENTRYPOINTS",
    "MAX_PITCH",
    "MAX_SPEED",
    "MIN_PITCH",
    "MIN_SPEED",
    "SAMPLE_WIDTH",
    "AudioChunk",
    "TtsEngine",
    "TtsOptions",
    "VoiceSpec",
    "clamp_pitch",
    "clamp_speed",
    "concat_chunks",
    "create_engine",
    "engine_class",
    "engine_names",
    "estimate_voice_bytes",
]

#: Sample rate most local engines produce. 22.05 kHz for Piper, 48 kHz for
#: Silero, 24 kHz for XTTS. The player resamples to whatever the device wants.
DEFAULT_SAMPLE_RATE: Final = 22050

#: Bytes per ``int16`` sample. Every engine here is asked for ``int16`` because
#: that is what PortAudio takes and what the shared-memory transport carries.
SAMPLE_WIDTH: Final = 2

#: Default speech rate multiplier.
DEFAULT_SPEED: Final = 1.0

#: Default pitch multiplier.
DEFAULT_PITCH: Final = 1.0

#: Range the settings allow, mirrored here so an engine called directly - from a
#: test, or from the model manager's preview button - cannot be handed a rate
#: that makes the voice unintelligible.
MIN_SPEED: Final = 0.5
MAX_SPEED: Final = 2.0
MIN_PITCH: Final = 0.5
MAX_PITCH: Final = 2.0

#: Engine name to ``module:Class`` entrypoint. A mapping proxy so a plugin that
#: wants another engine has to say so through the registry rather than by
#: mutating this at import time from wherever it happens to be loaded.
ENGINE_ENTRYPOINTS: Final[MappingProxyType[str, str]] = MappingProxyType(
    {
        "piper": "ayris.audio.tts.piper_engine:PiperTtsEngine",
        "silero": "ayris.audio.tts.silero_engine:SileroTtsEngine",
        "xtts": "ayris.audio.tts.coqui_engine:CoquiTtsEngine",
    }
)


@dataclass(frozen=True, slots=True)
class VoiceSpec:
    """Description of one voice, independent of the engine that produces it.

    Attributes:
        engine: Engine name (``piper``, ``silero``, ``xtts``).
        voice_id: Engine-specific identifier: a model file stem for Piper, a
            speaker ID for Silero, a sample name for XTTS.
        path: Absolute path to the model or the reference sample, when the voice
            is loaded from disk. Empty for voices the engine downloads or ships.
        language: Two-letter language code.
        display_name: What the settings window shows. Falls back to
            :attr:`voice_id` when the engine has nothing prettier.
        sample_rate: Native rate of this voice, when the engine can tell before
            loading - Piper's ``.json`` states it. ``0`` means "ask after load".
    """

    engine: str
    voice_id: str
    path: str = ""
    language: str = "ru"
    display_name: str = ""
    sample_rate: int = 0

    @property
    def label(self) -> str:
        """Text for the settings combo box."""
        return self.display_name or self.voice_id

    @property
    def key(self) -> str:
        """Identity for the cache and for "did the voice change" checks.

        The path is part of it: two user models can share a stem in different
        folders, and speaking one while the cache serves the other would be a
        bug nobody would think to look for.
        """
        return f"{self.engine}:{self.voice_id}:{self.path}"

    def same_voice(self, other: VoiceSpec | None) -> bool:
        """Whether ``other`` names the same voice, ignoring cosmetics."""
        return other is not None and other.key == self.key

    def to_params(self) -> JsonObject:
        """Flatten for the worker pipe."""
        return {
            "engine": self.engine,
            "voice_id": self.voice_id,
            "path": self.path,
            "language": self.language,
            "display_name": self.display_name,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_params(cls, params: JsonObject) -> VoiceSpec:
        """Rebuild from :meth:`to_params`."""
        return cls(
            engine=str(params.get("engine", "")),
            voice_id=str(params.get("voice_id", "")),
            path=str(params.get("path", "")),
            language=str(params.get("language", "ru")),
            display_name=str(params.get("display_name", "")),
            sample_rate=max(0, _as_int(params.get("sample_rate"), 0)),
        )


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """A piece of synthesized speech: little-endian ``int16`` PCM.

    One format for every engine, decided here rather than negotiated per call.
    Piper already produces ``int16``; Silero and XTTS produce float tensors and
    convert before returning. The alternative - carrying a format tag and
    branching in the player, the cache and the shared-memory transport - buys
    nothing, since the output device wants ``int16`` at the end of it anyway.

    Attributes:
        pcm: Raw samples, ``channels`` interleaved.
        sample_rate: Frames per second, as the engine produced them.
        channels: 1 for mono. Every local engine here is mono; the field exists
            because a cloud provider may not be.
    """

    pcm: bytes
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = 1

    @property
    def frames(self) -> int:
        """Sample frames in the chunk."""
        return len(self.pcm) // (SAMPLE_WIDTH * max(self.channels, 1))

    @property
    def duration_ms(self) -> float:
        """Length in milliseconds."""
        return self.frames * 1000.0 / self.sample_rate if self.sample_rate else 0.0

    @property
    def empty(self) -> bool:
        """Whether there is anything to play."""
        return not self.pcm

    def with_pcm(self, pcm: bytes) -> AudioChunk:
        """Same format, different samples. Used by the resampling step."""
        return replace(self, pcm=pcm)

    def metadata(self) -> JsonObject:
        """Format fields for the worker reply, without the bytes.

        The PCM travels through shared memory; this is what accompanies the
        descriptor so the caller knows how to interpret it.
        """
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass(frozen=True, slots=True)
class TtsOptions:
    """Everything an engine needs beyond the text and the voice.

    Attributes:
        speed: Speech rate multiplier. 1.0 is normal, 2.0 is double speed.
        pitch: Pitch multiplier. 1.0 is unchanged. Piper has no pitch control
            and ignores it; the field is still here so the settings window does
            not have to know that.
        threads: CPU threads the engine may use.
        gpu: ``auto``, ``cuda`` or ``cpu``.
        sample_rate: Rate to deliver. ``0`` - the default - means "whatever the
            voice is native at", which is the cheap path: resampling 22 kHz
            speech to 48 kHz in Python costs more than the player's own
            conversion, which happens once per device rather than once per
            sentence.
    """

    speed: float = DEFAULT_SPEED
    pitch: float = DEFAULT_PITCH
    threads: int = 2
    gpu: str = "auto"
    sample_rate: int = 0

    def to_params(self) -> JsonObject:
        """Flatten for the worker pipe."""
        return {
            "speed": self.speed,
            "pitch": self.pitch,
            "threads": self.threads,
            "gpu": self.gpu,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_params(cls, params: JsonObject) -> TtsOptions:
        """Build from worker parameters, clamping anything out of range."""
        return cls(
            speed=clamp_speed(_as_float(params.get("speed"), DEFAULT_SPEED)),
            pitch=clamp_pitch(_as_float(params.get("pitch"), DEFAULT_PITCH)),
            threads=max(1, _as_int(params.get("threads"), 2)),
            gpu=str(params.get("gpu", "auto")) or "auto",
            sample_rate=max(0, _as_int(params.get("sample_rate"), 0)),
        )


class TtsEngine(ABC):
    """Base class for a speech synthesizer.

    A subclass implements four methods - :meth:`load`, :meth:`_synthesize`,
    :meth:`unload` and :attr:`supported_languages` - and optionally
    :meth:`voices` and :meth:`synthesize_stream`. Everything else, including
    switching voices mid-session and clamping the rate, is handled here so that
    three engines cannot disagree about it.

    Instances are not thread-safe. One engine belongs to one worker process, and
    :class:`~ayris.workers.tts_worker.TtsWorker` holds a lock around it.
    """

    #: Value of ``voice.tts.engine`` this class implements.
    name: ClassVar[str] = "tts"

    #: Distribution to install, named in the error when it is missing.
    package: ClassVar[str] = ""

    #: Top-level module imported at load time.
    module: ClassVar[str] = ""

    #: Whether the engine is offered only when its library is present. XTTS is;
    #: Piper and Silero stay in the list even uninstalled, so the settings can
    #: explain what to install instead of silently hiding the option.
    optional: ClassVar[bool] = False

    #: Resident memory per byte of model on disk, for the worker's RAM check.
    memory_factor: ClassVar[float] = 1.5

    #: Rate this engine produces when the voice does not say otherwise.
    native_sample_rate: ClassVar[int] = DEFAULT_SAMPLE_RATE

    __slots__ = ("_options", "_voice")

    def __init__(self) -> None:
        self._voice: VoiceSpec | None = None
        self._options: TtsOptions = TtsOptions()

    # ------------------------------------------------------------ description

    @property
    @abstractmethod
    def supported_languages(self) -> tuple[str, ...]:
        """Language codes this engine can be asked for.

        An empty tuple means the loaded voice does not declare its language,
        which is not an error - Piper voices are named by locale and XTTS is
        multilingual by construction.
        """

    @property
    def device(self) -> str:
        """What is actually running: ``cpu`` or ``cuda``."""
        return "cpu"

    @property
    def loaded(self) -> bool:
        """Whether a voice is in memory and :meth:`synthesize` will do anything."""
        return self._voice is not None

    @property
    def voice(self) -> VoiceSpec | None:
        """The voice the engine was loaded with."""
        return self._voice

    @property
    def options(self) -> TtsOptions:
        """The options the engine was loaded with."""
        return self._options

    @property
    def sample_rate(self) -> int:
        """Rate the loaded voice produces."""
        voice = self._voice
        if voice is not None and voice.sample_rate > 0:
            return voice.sample_rate
        return self.native_sample_rate

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        """Voices this engine can offer, for the settings combo box.

        Args:
            directory: Where user-supplied voices live, usually
                ``models/tts``. Engines with a fixed built-in set ignore it.

        Returns:
            Empty by default: an engine that cannot enumerate is not broken, it
            just leaves the user to type a voice name. Never raises - a settings
            window that crashed because one folder was unreadable would be worse
            than one showing a short list.
        """
        del directory
        return ()

    # -------------------------------------------------------------- lifecycle

    @abstractmethod
    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        """Bring the voice into memory.

        Implementations set ``self._voice`` and ``self._options`` on success and
        leave them untouched on failure, so a failed voice switch keeps the
        previous voice usable.

        Args:
            voice: Which voice to load. The engine interprets ``voice_id`` and
                ``path`` by its own conventions.
            options: Speed, pitch, threads, device preference.

        Raises:
            TtsError: The library is missing, the voice is not found, or the
                device refused.
        """

    @abstractmethod
    def unload(self) -> None:
        """Release the voice. Must be safe to call twice and must not raise."""

    # ------------------------------------------------------------- synthesis

    def synthesize(
        self,
        text: str,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> AudioChunk:
        """Synthesize one phrase.

        Args:
            text: Plain text. The engine applies its own normalization; what
                arrives here has already been split into speakable pieces by
                :mod:`ayris.audio.tts.sentence_split`.
            voice: Speak in this voice instead of the loaded one. Triggers a
                reload only when it is a different voice.
            speed: Rate multiplier for this call, clamped to
                :data:`MIN_SPEED`..:data:`MAX_SPEED`. ``None`` uses the loaded
                options.
            pitch: Pitch multiplier for this call, clamped the same way.

        Returns:
            PCM audio. Empty when ``text`` has nothing speakable in it - which
            is a normal outcome for "..." or a bare emoji, not an error.

        Raises:
            TtsError: No voice is loaded and none was passed, or synthesis
                failed.
        """
        from ayris.audio.tts.sentence_split import is_speakable

        if not is_speakable(text):
            return AudioChunk(b"", self.sample_rate)
        self._ensure_voice(voice)
        return self._synthesize(
            text,
            clamp_speed(self._options.speed if speed is None else speed),
            clamp_pitch(self._options.pitch if pitch is None else pitch),
        )

    def synthesize_stream(
        self,
        text: str,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> Iterator[AudioChunk]:
        """Synthesize sentence by sentence, yielding each as it is ready.

        This is what keeps the first sound inside the 500 ms budget: the caller
        starts playing chunk one while the generator is still working on chunk
        two. Engines that can stream more finely than a sentence override it.

        Args:
            text: Plain text, possibly several sentences.
            voice: As in :meth:`synthesize`.
            speed: As in :meth:`synthesize`.
            pitch: As in :meth:`synthesize`.

        Yields:
            One chunk per sentence, in order, skipping the empty ones.

        Raises:
            TtsError: Same as :meth:`synthesize`. Raised from the generator, so
                a failure on sentence three arrives after two good chunks - the
                caller has already played them and that is the point.
        """
        from ayris.audio.tts.sentence_split import split_sentences

        for sentence in split_sentences(text):
            chunk = self.synthesize(sentence, voice, speed, pitch)
            if not chunk.empty:
                yield chunk

    @abstractmethod
    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        """Engine-specific synthesis of one non-empty phrase.

        Called with a voice already loaded and ``speed``/``pitch`` already
        clamped, so implementations do no validation.

        Raises:
            TtsError: Synthesis failed.
        """

    # ---------------------------------------------------------------- helpers

    def _ensure_voice(self, voice: VoiceSpec | None) -> VoiceSpec:
        """Load ``voice`` if it differs from the current one, and return it.

        Raises:
            TtsError: Nothing is loaded and ``voice`` is ``None``, or the load
                failed.
        """
        if voice is not None and not voice.same_voice(self._voice):
            self.load(voice, self._options)
        return self._require_loaded()

    def _require_loaded(self) -> VoiceSpec:
        """The voice the engine was loaded with.

        Raises:
            TtsError: If :meth:`load` has not run.
        """
        if self._voice is None:
            raise TtsError(
                f"{self.name}: no voice is loaded",
                user_message="Голос синтеза речи ещё не загружен.",
            )
        return self._voice

    @classmethod
    def available(cls) -> bool:
        """Whether the vendor library can be imported, without importing it."""
        if not cls.module:
            return False
        try:
            return importlib.util.find_spec(cls.module) is not None
        except (ImportError, ValueError):
            return False

    @classmethod
    def _import(cls, module: str | None = None) -> Any:
        """Import the vendor library.

        Raises:
            TtsError: If it is not installed, naming the package to install.
        """
        target = module or cls.module
        try:
            return importlib.import_module(target)
        except ImportError as exc:
            package = cls.package or target
            raise TtsError(
                f"{cls.name}: {target} is not installed: {exc}",
                user_message=(
                    f"Движок синтеза «{cls.name}» не установлен. "
                    f"Установите пакет {package} или выберите другой движок в настройках."
                ),
            ) from exc

    def __repr__(self) -> str:
        voice = self._voice.voice_id if self._voice else ""
        return f"{type(self).__name__}(loaded={self.loaded}, voice={voice!r})"


# ------------------------------------------------------------------ registry


def engine_names(*, available_only: bool = False) -> tuple[str, ...]:
    """Engine names the settings window may offer.

    Args:
        available_only: Drop engines whose library is not installed. Off by
            default, because "Piper (не установлен)" is more useful to a user
            than an option that silently is not there.

    Returns:
        Names in registration order. An engine whose module fails to import at
        all is skipped rather than raised on: a broken optional dependency must
        not empty the combo box.
    """
    names: list[str] = []
    for name in ENGINE_ENTRYPOINTS:
        try:
            engine = engine_class(name)
        except TtsError:
            continue
        if engine.optional and not engine.available():
            continue
        if available_only and not engine.available():
            continue
        names.append(name)
    return tuple(names)


def engine_class(name: str) -> type[TtsEngine]:
    """Resolve an engine name to its class without constructing anything.

    Raises:
        TtsError: If the name is unknown or its module is broken.
    """
    entrypoint = ENGINE_ENTRYPOINTS.get(name)
    if entrypoint is None:
        known = ", ".join(sorted(ENGINE_ENTRYPOINTS))
        raise TtsError(
            f"unknown tts engine {name!r}, expected one of {known}",
            user_message=f"Неизвестный движок синтеза речи: {name}.",
        )
    module_name, _, attribute = entrypoint.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise TtsError(
            f"cannot import tts engine {name!r}: {exc}",
            user_message=f"Не удалось загрузить движок синтеза «{name}».",
        ) from exc
    factory: type[TtsEngine] = getattr(module, attribute)
    return factory


def create_engine(name: str) -> TtsEngine:
    """Build the engine ``voice.tts.engine`` names.

    Raises:
        TtsError: Same as :func:`engine_class`.
    """
    return engine_class(name)()


# ------------------------------------------------------------------- helpers


def concat_chunks(chunks: Sequence[AudioChunk]) -> AudioChunk:
    """Join chunks that share a format into one.

    Used by the cache, which stores a whole phrase, and by callers that asked
    for a stream but want the result in one piece.

    Raises:
        TtsError: The chunks disagree about rate or channel count. Joining them
            anyway would play the second half at the wrong speed.
    """
    present = [chunk for chunk in chunks if not chunk.empty]
    if not present:
        return AudioChunk(b"", DEFAULT_SAMPLE_RATE)
    first = present[0]
    for chunk in present[1:]:
        if chunk.sample_rate != first.sample_rate or chunk.channels != first.channels:
            raise TtsError(
                f"cannot join {first.sample_rate} Hz/{first.channels}ch with "
                f"{chunk.sample_rate} Hz/{chunk.channels}ch audio",
                user_message="Внутренняя ошибка синтеза: куски речи в разном формате.",
            )
    return first.with_pcm(b"".join(chunk.pcm for chunk in present))


def estimate_voice_bytes(path: Path) -> int:
    """Size of a voice on disk, for the worker's RAM check.

    A file is its own size; a directory is the sum of the files in it, which is
    what XTTS and a downloaded Silero package look like. Unreadable paths
    measure as zero, which the caller reads as "cannot tell, let it through" -
    refusing to speak because a stat call failed would be the wrong trade.
    """
    try:
        if path.is_file():
            return path.stat().st_size
        if path.is_dir():
            return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    except OSError:
        return 0
    return 0


def clamp_speed(value: float) -> float:
    """Hold the rate inside the range the settings allow."""
    return min(max(value, MIN_SPEED), MAX_SPEED)


def clamp_pitch(value: float) -> float:
    """Hold the pitch inside the range the settings allow."""
    return min(max(value, MIN_PITCH), MAX_PITCH)


def _as_float(value: object, fallback: float) -> float:
    """Coerce a wire value to a float without trusting it."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return float(value)


def _as_int(value: object, fallback: int) -> int:
    """Coerce a wire value to an int without trusting it."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return fallback
    return int(value)
