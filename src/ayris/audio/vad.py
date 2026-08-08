"""Voice activity detection: deciding which frames of the stream carry speech.

The detector answers one question per frame - "is somebody talking right now" -
and :mod:`ayris.audio.segmenter` turns that stream of yes/no answers into whole
phrases.  Keeping the two apart matters because they fail differently: a VAD
that flickers is a tuning problem, a segmenter that cuts a phrase in half is a
logic problem, and debugging them together is miserable.

Two engines implement :class:`Vad`.

:class:`WebRtcVad`
    The GMM classifier from WebRTC, through ``webrtcvad-wheels``.  It is the
    default because it recognises the *shape* of speech rather than its
    loudness, so a fan or a keyboard does not open the gate.  It is strict about
    its input: mono ``int16`` at 8/16/32/48 kHz, in frames of exactly 10, 20 or
    30 ms - hence :class:`FrameSplitter`.
:class:`EnergyVad`
    A level gate with an adaptive noise floor.  Used when the wheel is missing
    (the import is deferred, so a broken install degrades instead of refusing to
    start) and when a user with an unusual microphone turns the classifier off.

Both are wrapped by a level gate derived from calibration:
``noise_floor_db + margin``, where the margin comes from
``voice.audio_input.vad_threshold``.  WebRTC's classifier has no notion of how
loud a room is, and in a noisy one it happily reports speech for background
chatter from a television; the gate is what makes the calibration procedure in
:mod:`ayris.audio.calibration` worth running.

Everything is stdlib-only and works on ``bytes``, matching
:mod:`ayris.audio.capture`: the audio worker should not import NumPy just to
compare two numbers per frame.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.capture import MIN_DBFS, TARGET_SAMPLE_RATE, pcm_level
from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.core.errors import AudioError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DEFAULT_AGGRESSIVENESS",
    "DEFAULT_NOISE_FLOOR_DB",
    "DEFAULT_THRESHOLD",
    "MAX_MARGIN_DB",
    "MIN_MARGIN_DB",
    "VAD_FRAME_MS",
    "VAD_SAMPLE_RATES",
    "EnergyVad",
    "FrameSplitter",
    "Vad",
    "VadEngine",
    "VadFrame",
    "VadSettings",
    "VadStream",
    "WebRtcVad",
    "create_vad",
    "frames_of",
    "webrtcvad_available",
]

_log = logging.getLogger(__name__)

#: Rates the WebRTC classifier accepts.  Anything else has to be resampled
#: first, which capture already does.
VAD_SAMPLE_RATES: Final[frozenset[int]] = frozenset({8000, 16000, 32000, 48000})

#: Frame lengths the WebRTC classifier accepts, in milliseconds.
VAD_FRAME_MS: Final[tuple[int, ...]] = (10, 20, 30)

#: Middle of the road: 0 lets a lot of noise through, 3 clips quiet speech.
DEFAULT_AGGRESSIVENESS: Final = 2

#: Neutral sensitivity, matching ``voice.audio_input.vad_threshold``.
DEFAULT_THRESHOLD: Final = 0.5

#: Room tone of a quiet room with a decent microphone.  Calibration replaces it.
DEFAULT_NOISE_FLOOR_DB: Final = -45.0

#: How far above the noise floor a frame must sit at ``threshold == 0``.
MIN_MARGIN_DB: Final = 3.0

#: The same at ``threshold == 1``.  Above ~18 dB only shouting gets through.
MAX_MARGIN_DB: Final = 18.0

#: Once the gate is open it stays open for slightly quieter frames, so that the
#: dip between two syllables does not read as the end of the phrase.
_HYSTERESIS_DB: Final = 4.0

#: Exponential trackers for :class:`EnergyVad`'s noise floor.  Falling fast and
#: rising slowly means a door slamming raises the floor for a moment while a
#: fan that switches on is followed within a couple of seconds.
_FLOOR_FALL: Final = 0.35
_FLOOR_RISE: Final = 0.01

#: The floor tracker is not allowed to climb into speech territory: past this
#: point every frame would look like background and the gate would never open.
_FLOOR_CEILING_DB: Final = -20.0


class VadEngine(StrEnum):
    """Which classifier to use.

    ``AUTO`` prefers WebRTC and falls back to the energy gate with a warning in
    the log; the explicit values fail loudly instead, because a user who picked
    an engine in the settings deserves to know it is unavailable.
    """

    AUTO = "auto"
    WEBRTC = "webrtc"
    ENERGY = "energy"


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Everything the detector needs to know.

    Attributes:
        sample_rate: Rate of the frames pushed in.  Must be one of
            :data:`VAD_SAMPLE_RATES`.
        frame_ms: Frame length, one of :data:`VAD_FRAME_MS`.
        aggressiveness: WebRTC mode, 0 (permissive) to 3 (strict).
        threshold: Sensitivity of the level gate, 0.0-1.0, from
            ``voice.audio_input.vad_threshold``.  Higher means the speaker has
            to be louder relative to the room.
        noise_floor_db: Measured room tone, from calibration.
        engine: Which classifier to build.

    Raises:
        AudioError: If a value is outside the range its engine accepts.  The
            settings model already validates the same ranges; this catches a
            worker started with hand-written parameters.
    """

    sample_rate: int = TARGET_SAMPLE_RATE
    frame_ms: int = 20
    aggressiveness: int = DEFAULT_AGGRESSIVENESS
    threshold: float = DEFAULT_THRESHOLD
    noise_floor_db: float = DEFAULT_NOISE_FLOOR_DB
    engine: VadEngine = VadEngine.AUTO

    def __post_init__(self) -> None:
        if self.sample_rate not in VAD_SAMPLE_RATES:
            raise AudioError(
                f"VAD does not support {self.sample_rate} Hz",
                user_message="Частота дискретизации не поддерживается детектором речи.",
            )
        if self.frame_ms not in VAD_FRAME_MS:
            raise AudioError(
                f"VAD frame must be one of {VAD_FRAME_MS} ms, got {self.frame_ms}",
                user_message="Длина кадра VAD должна быть 10, 20 или 30 мс.",
            )
        if not 0 <= self.aggressiveness <= 3:
            raise AudioError(
                f"VAD aggressiveness must be 0-3, got {self.aggressiveness}",
                user_message="Агрессивность VAD задаётся числом от 0 до 3.",
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise AudioError(
                f"VAD threshold must be 0.0-1.0, got {self.threshold}",
                user_message="Порог VAD задаётся числом от 0 до 1.",
            )

    @property
    def frame_samples(self) -> int:
        """Samples in one frame."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        """Bytes in one frame."""
        return self.frame_samples * SAMPLE_WIDTH

    @property
    def margin_db(self) -> float:
        """How far above the noise floor speech has to be."""
        return MIN_MARGIN_DB + (MAX_MARGIN_DB - MIN_MARGIN_DB) * self.threshold

    @property
    def gate_db(self) -> float:
        """Absolute level below which a frame is never speech.

        This is the number the microphone meter draws its cut-off line at.
        """
        return max(MIN_DBFS, self.noise_floor_db + self.margin_db)


@dataclass(frozen=True, slots=True)
class VadFrame:
    """One classified frame.

    Attributes:
        pcm: The frame itself, little-endian ``int16`` mono.
        index: Absolute frame counter since the stream was reset, so a consumer
            can convert to a position without tracking time.
        speech: Final verdict, after the level gate.
        voiced: What the classifier said before the gate.  The two differ
            exactly when the room is louder than calibration expected, which is
            the thing to look at when the VAD "does not work".
        rms_db: Level of the frame in dBFS.
    """

    pcm: bytes
    index: int
    speech: bool
    voiced: bool
    rms_db: float

    @property
    def samples(self) -> int:
        """Samples in this frame."""
        return len(self.pcm) // SAMPLE_WIDTH


class Vad(ABC):
    """Base class for a frame classifier.

    Args:
        settings: Configuration; the engine keeps it and re-reads what it needs
            per frame, so :meth:`configure` is cheap.
    """

    #: Value of ``voice.audio_input`` engine selection this class implements.
    name: ClassVar[str] = "vad"

    #: Whether :class:`VadStream` should apply the calibrated level gate on top
    #: of this engine's answer.  False for engines that already measure level.
    needs_level_gate: ClassVar[bool] = True

    __slots__ = ("_settings",)

    def __init__(self, settings: VadSettings | None = None) -> None:
        self._settings = settings or VadSettings()

    @property
    def settings(self) -> VadSettings:
        """Configuration in force."""
        return self._settings

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Classify one frame of exactly :attr:`VadSettings.frame_bytes` bytes.

        Raises:
            AudioError: If the frame has the wrong length for the settings.
        """

    def configure(self, settings: VadSettings) -> None:
        """Apply new settings.  Subclasses extend this to re-arm the engine."""
        self._settings = settings

    @abstractmethod
    def reset(self) -> None:
        """Forget per-stream state.  Called when capture restarts."""

    def _check_frame(self, frame: bytes) -> None:
        """Reject a frame the engine cannot classify."""
        expected = self._settings.frame_bytes
        if len(frame) != expected:
            raise AudioError(
                f"VAD frame must be {expected} bytes, got {len(frame)}",
                user_message="Внутренняя ошибка обработки звука: неверный размер кадра.",
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self._settings.aggressiveness})"


class WebRtcVad(Vad):
    """The WebRTC classifier.

    The ``webrtcvad`` import is deferred into the constructor rather than done
    at module level: the wheel is a compiled extension, and a user whose install
    is missing it should end up on :class:`EnergyVad` with a warning instead of
    an application that will not start.  It is also what keeps the test suite
    collectable on a runner where the wheel failed to build.

    Raises:
        AudioError: If the extension is not installed or rejects the mode.
    """

    name: ClassVar[str] = "webrtc"
    needs_level_gate: ClassVar[bool] = True

    __slots__ = ("_vad",)

    def __init__(self, settings: VadSettings | None = None) -> None:
        super().__init__(settings)
        self._vad = _new_webrtc_vad(self._settings.aggressiveness)

    def is_speech(self, frame: bytes) -> bool:
        """Classify one frame."""
        self._check_frame(frame)
        try:
            return bool(self._vad.is_speech(frame, self._settings.sample_rate))
        except Exception as exc:  # pragma: no cover - the extension only raises on bad input
            raise AudioError(
                f"webrtcvad rejected a frame: {exc}",
                user_message="Детектор речи не смог обработать звук.",
            ) from exc

    def configure(self, settings: VadSettings) -> None:
        """Apply new settings, rebuilding the classifier if the mode changed."""
        previous = self._settings
        super().configure(settings)
        if settings.aggressiveness != previous.aggressiveness:
            self._vad = _new_webrtc_vad(settings.aggressiveness)

    def reset(self) -> None:
        """Nothing to do: the classifier carries no state between frames."""


class EnergyVad(Vad):
    """Level gate with an adaptive noise floor.

    Deliberately simple: RMS above ``floor + margin`` is speech, with hysteresis
    so that the gap between two syllables does not close the gate.  The floor
    itself follows the quiet parts of the signal, which is what makes this
    usable at all - a fixed threshold is wrong the moment the user moves the
    microphone or the air conditioning starts.

    It cannot tell speech from a slammed door, so it is a fallback rather than a
    peer of :class:`WebRtcVad`.  :class:`~ayris.audio.segmenter.Segmenter`
    covers part of that gap by requiring several frames in a row before it
    believes a phrase started.
    """

    name: ClassVar[str] = "energy"
    #: It *is* the level gate; applying the static one on top would fight the
    #: adaptive floor every time the room turns out quieter than calibration.
    needs_level_gate: ClassVar[bool] = False

    __slots__ = ("_floor_db", "_open")

    def __init__(self, settings: VadSettings | None = None) -> None:
        super().__init__(settings)
        self._floor_db = self._settings.noise_floor_db
        self._open = False

    @property
    def floor_db(self) -> float:
        """Noise floor the tracker currently believes in."""
        return self._floor_db

    def is_speech(self, frame: bytes) -> bool:
        """Classify one frame and update the noise floor."""
        self._check_frame(frame)
        level = pcm_level(frame).rms_db
        threshold = self._floor_db + self._settings.margin_db
        self._open = level >= (threshold - _HYSTERESIS_DB if self._open else threshold)
        if not self._open:
            # Track only while closed: adapting during speech would raise the
            # floor to the speaker's own level and shut the gate mid-phrase.
            rate = _FLOOR_FALL if level < self._floor_db else _FLOOR_RISE
            self._floor_db = min(
                _FLOOR_CEILING_DB, max(MIN_DBFS, self._floor_db + (level - self._floor_db) * rate)
            )
        return self._open

    def configure(self, settings: VadSettings) -> None:
        """Apply new settings, re-seeding the floor when calibration changed it."""
        previous = self._settings
        super().configure(settings)
        if settings.noise_floor_db != previous.noise_floor_db:
            self._floor_db = settings.noise_floor_db

    def reset(self) -> None:
        """Return to the calibrated floor with the gate closed."""
        self._floor_db = self._settings.noise_floor_db
        self._open = False


class FrameSplitter:
    """Cuts an arbitrary byte stream into fixed-size frames.

    Capture delivers whatever the driver hands over - 20 ms blocks normally, but
    a resampled 44.1 kHz stream produces 319 or 321 samples per block, and a
    recovered device produces a short first block.  WebRTC's classifier accepts
    exactly one of three sizes and raises otherwise, so every frame reaching it
    goes through here first.

    Args:
        frame_bytes: Size of the frames to produce.

    Raises:
        ValueError: If ``frame_bytes`` is not a positive whole number of frames.
    """

    __slots__ = ("_buffer", "_frame_bytes")

    def __init__(self, frame_bytes: int) -> None:
        if frame_bytes <= 0 or frame_bytes % SAMPLE_WIDTH:
            raise ValueError(f"frame_bytes must be positive and even, got {frame_bytes}")
        self._frame_bytes = frame_bytes
        self._buffer = bytearray()

    @property
    def frame_bytes(self) -> int:
        """Size of the frames produced."""
        return self._frame_bytes

    @property
    def pending(self) -> int:
        """Bytes held back because they do not fill a frame yet."""
        return len(self._buffer)

    def push(self, pcm: bytes | bytearray | memoryview) -> list[bytes]:
        """Add audio and return every complete frame it produced."""
        self._buffer += bytes(pcm)
        size = self._frame_bytes
        count = len(self._buffer) // size
        if count == 0:
            return []
        frames = [bytes(self._buffer[index * size : (index + 1) * size]) for index in range(count)]
        del self._buffer[: count * size]
        return frames

    def flush(self) -> bytes:
        """Return the remainder padded with silence, or ``b""`` if empty.

        Used when a phrase ends: the last few milliseconds would otherwise be
        dropped, and for a maximum-length cut that is audible.
        """
        if not self._buffer:
            return b""
        frame = bytes(self._buffer) + bytes(self._frame_bytes - len(self._buffer))
        self._buffer.clear()
        return frame

    def reset(self) -> None:
        """Drop the partial frame."""
        self._buffer.clear()


class VadStream:
    """A :class:`Vad` plus framing, gating and frame numbering.

    This is what consumers use.  Push whatever capture produced, get back a
    tuple of classified frames:

        >>> stream = VadStream(VadSettings(frame_ms=20))
        >>> frames = stream.push(block)
        >>> any(frame.speech for frame in frames)
        True

    Args:
        settings: Detector configuration.
        vad: Pre-built engine, for tests and for reusing one across a
            reconfiguration.  Built from ``settings`` when omitted.
    """

    __slots__ = ("_gate_db", "_index", "_settings", "_speech", "_splitter", "_vad")

    def __init__(self, settings: VadSettings | None = None, *, vad: Vad | None = None) -> None:
        self._settings = settings or (vad.settings if vad is not None else VadSettings())
        self._vad = vad if vad is not None else create_vad(self._settings)
        self._splitter = FrameSplitter(self._settings.frame_bytes)
        self._gate_db = self._settings.gate_db
        self._index = 0
        self._speech = False

    @property
    def settings(self) -> VadSettings:
        """Configuration in force."""
        return self._settings

    @property
    def vad(self) -> Vad:
        """The engine doing the classification."""
        return self._vad

    @property
    def engine(self) -> str:
        """Name of the engine, for the status panel."""
        return self._vad.name

    @property
    def gate_db(self) -> float:
        """Level cut-off currently applied, in dBFS.  Drawn by the UI meter."""
        return self._gate_db

    @property
    def is_speech(self) -> bool:
        """Verdict of the most recent frame."""
        return self._speech

    @property
    def index(self) -> int:
        """Frames classified since the last :meth:`reset`."""
        return self._index

    def push(self, pcm: bytes | bytearray | memoryview) -> tuple[VadFrame, ...]:
        """Classify everything in ``pcm`` that fills a whole frame."""
        return tuple(self._classify(frame) for frame in self._splitter.push(pcm))

    def flush(self) -> tuple[VadFrame, ...]:
        """Classify the padded remainder, if any."""
        tail = self._splitter.flush()
        return (self._classify(tail),) if tail else ()

    def configure(self, settings: VadSettings) -> None:
        """Apply new settings without losing the frame counter.

        A changed frame size or sample rate resets the splitter - the bytes it
        holds belong to the old framing - but the absolute index keeps counting
        so that positions handed out earlier stay comparable.
        """
        previous = self._settings
        self._settings = settings
        self._gate_db = settings.gate_db
        if settings.engine != previous.engine:
            self._vad = create_vad(settings)
        else:
            self._vad.configure(settings)
        if settings.frame_bytes != previous.frame_bytes:
            self._splitter = FrameSplitter(settings.frame_bytes)

    def reset(self) -> None:
        """Drop buffered audio and per-stream state."""
        self._splitter.reset()
        self._vad.reset()
        self._index = 0
        self._speech = False

    def _classify(self, frame: bytes) -> VadFrame:
        """Run one frame through the engine and the level gate."""
        voiced = self._vad.is_speech(frame)
        rms_db = pcm_level(frame).rms_db
        speech = voiced and (not self._vad.needs_level_gate or rms_db >= self._gate_db)
        index = self._index
        self._index += 1
        self._speech = speech
        return VadFrame(pcm=frame, index=index, speech=speech, voiced=voiced, rms_db=rms_db)

    def __repr__(self) -> str:
        return f"VadStream(engine={self.engine}, gate={self._gate_db:.1f} dBFS)"


def webrtcvad_available() -> bool:
    """Whether the compiled WebRTC classifier can be imported."""
    try:
        _import_webrtcvad()
    except AudioError:
        return False
    return True


def create_vad(settings: VadSettings | None = None) -> Vad:
    """Build the engine ``settings`` asks for.

    Args:
        settings: Detector configuration.

    Returns:
        A :class:`WebRtcVad` when possible.  Under
        :attr:`VadEngine.AUTO` a missing wheel is logged and downgraded to
        :class:`EnergyVad`; under :attr:`VadEngine.WEBRTC` it raises, because
        silently ignoring an explicit choice is how a user ends up debugging
        the wrong component.

    Raises:
        AudioError: If the requested engine is unavailable.
    """
    resolved = settings or VadSettings()
    if resolved.engine is VadEngine.ENERGY:
        return EnergyVad(resolved)
    if resolved.engine is VadEngine.WEBRTC:
        return WebRtcVad(resolved)
    try:
        return WebRtcVad(resolved)
    except AudioError as exc:
        _log.warning("webrtcvad unavailable (%s), falling back to the energy gate", exc.technical)
        return EnergyVad(replace(resolved, engine=VadEngine.ENERGY))


def frames_of(
    pcm: bytes | bytearray | memoryview,
    settings: VadSettings | None = None,
) -> Iterable[bytes]:
    """Split ``pcm`` into whole frames, dropping an incomplete tail.

    A convenience for offline analysis - calibration and the tests - where the
    audio is already in memory and the framing has no history.
    """
    resolved = settings or VadSettings()
    size = resolved.frame_bytes
    view = memoryview(pcm).cast("B")
    for start in range(0, view.nbytes - size + 1, size):
        yield bytes(view[start : start + size])


def _import_webrtcvad() -> Any:
    """Import the compiled extension.

    Raises:
        AudioError: If it is not installed.
    """
    try:
        # Deferred on purpose - see the class docstring of WebRtcVad.
        import webrtcvad
    except ImportError as exc:
        raise AudioError(
            f"webrtcvad is not installed: {exc}",
            user_message=(
                "Не установлен детектор речи webrtcvad. "
                "Используется упрощённое определение по уровню сигнала."
            ),
        ) from exc
    return webrtcvad


def _new_webrtc_vad(aggressiveness: int) -> Any:
    """Construct the extension's ``Vad`` object.

    Raises:
        AudioError: If the extension is missing or rejects the mode.
    """
    module = _import_webrtcvad()
    try:
        return module.Vad(aggressiveness)
    except Exception as exc:  # pragma: no cover - only a bad mode gets here
        raise AudioError(
            f"webrtcvad rejected mode {aggressiveness}: {exc}",
            user_message="Не удалось настроить детектор речи.",
        ) from exc
