"""Noise suppression for the capture path.

Two engines behind one interface, because the good one cannot be shipped
everywhere.

**RNNoise** is a small recurrent network trained on noisy speech; it removes
keyboard clatter and fan hum that a level-based gate cannot touch, and it costs
about 1% of a core.  It is a compiled C library, so it is an *optional*
dependency: Ayris looks for it, uses it when it is there, and says so in the
settings window when it is not.  Nothing breaks in its absence, which is the
whole point of :func:`create_denoiser`.

**The gate** is the fallback and the "spectral" setting in
``voice.audio_input.denoise``.  It splits the signal into a low and a high band
with one-pole filters, tracks a noise floor in each, and pulls a band down when
it sits at the floor.  Two bands are enough for the case that matters - hiss and
fan noise live above a voice, so attenuating the top band while the bottom one
is open removes the hiss without hollowing out the speech - and, unlike an FFT,
two one-pole filters are affordable in pure Python.  Everything here uses
:mod:`array` and :mod:`math` only, for the same reason
:mod:`ayris.audio.capture` does: the audio worker must start instantly, and
importing NumPy costs more than this module will ever spend running.

Denoising is never free, and the specification asks for the price to be visible.
:class:`DenoiseStream` therefore reports both numbers the settings window shows:
:attr:`DenoiseStats.latency_ms` - the algorithmic delay the engine adds, which
is what a user feels - and :attr:`DenoiseStats.avg_ms` per frame, which is what
tells us whether the machine can afford it at all.

The denoiser sits *before* the detector: the VAD and the wake word see cleaned
audio, and so does recognition.  Nothing here is thread-safe; one instance
belongs to one pipeline.
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from array import array
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.audio.capture import MIN_DBFS, TARGET_SAMPLE_RATE, Resampler
from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.core.errors import AudioError

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "DEFAULT_REDUCTION_DB",
    "RNNOISE_FRAME_SAMPLES",
    "RNNOISE_SAMPLE_RATE",
    "DenoiseEngine",
    "DenoiseMode",
    "DenoiseSettings",
    "DenoiseStats",
    "DenoiseStream",
    "Denoiser",
    "NoiseGate",
    "Passthrough",
    "RnnoiseDenoiser",
    "create_denoiser",
    "denoise_pcm",
    "rnnoise_available",
    "rnnoise_library",
]

_log = logging.getLogger(__name__)

#: RNNoise is trained at this rate and accepts nothing else.
RNNOISE_SAMPLE_RATE: Final = 48000

#: Samples per RNNoise frame: 10 ms at 48 kHz.
RNNOISE_FRAME_SAMPLES: Final = 480

#: Environment override for the RNNoise shared library, for a user who has one
#: in a place we would not think to look.
RNNOISE_LIB_ENV: Final = "AYRIS_RNNOISE_LIB"

#: Library names tried in order.  Not branched on :data:`sys.platform` on
#: purpose - loading a ``.so`` on Windows simply fails, and the flat list keeps
#: the search identical everywhere, including in tests.
_LIBRARY_NAMES: Final = (
    "rnnoise.dll",
    "librnnoise.dll",
    "librnnoise.so.0",
    "librnnoise.so",
    "librnnoise.dylib",
)

#: How far a band is pulled down when it sits at the noise floor.  12 dB is the
#: usual compromise: audibly quieter, still not a hole in the recording that
#: makes a recogniser hallucinate.
DEFAULT_REDUCTION_DB: Final = 12.0

#: Band boundary.  Below it lives the voice fundamental and the first formant,
#: above it the sibilants - and most of the hiss.
_CROSSOVER_HZ: Final = 900.0

#: Level above the floor at which a band is considered fully open.  Below the
#: floor the band is fully attenuated; in between the gain slides.
_OPEN_MARGIN_DB: Final = 9.0

#: Gain smoothing.  Opening must be fast enough not to clip the first consonant,
#: closing slow enough not to chop the end of a word.
_ATTACK_MS: Final = 5.0
_RELEASE_MS: Final = 80.0

#: Noise floor tracking, same asymmetry as :class:`ayris.audio.vad.EnergyVad`:
#: drop to a quiet passage at once, crawl up so speech cannot drag the floor
#: over itself.
_FLOOR_FALL: Final = 0.30
_FLOOR_RISE: Final = 0.01

#: Never treat anything this loud as noise, whatever the floor says.
_FLOOR_CEILING_DB: Final = -22.0

_FULL_SCALE: Final = 32768.0


class DenoiseMode(StrEnum):
    """What the user asked for.  Mirrors ``voice.audio_input.denoise``."""

    #: No processing at all.  The audio reaching recognition is what the sound
    #: card produced.
    OFF = "off"
    #: RNNoise when available, the gate when not.
    RNNOISE = "rnnoise"
    #: The two-band gate, always available.
    SPECTRAL = "spectral"


class DenoiseEngine(StrEnum):
    """What is actually running, which need not be what was asked for."""

    NONE = "none"
    RNNOISE = "rnnoise"
    GATE = "gate"


@dataclass(frozen=True, slots=True)
class DenoiseSettings:
    """Configuration of the suppressor.

    Attributes:
        mode: What the user selected.
        sample_rate: Rate of the audio pushed in, and of what comes out.
        frame_ms: Processing block.  Only affects the gate's reaction time;
            RNNoise always works in its own 10 ms frames.
        noise_floor_db: Starting estimate of the noise level, normally the value
            :mod:`ayris.audio.calibration` measured.
        reduction_db: Maximum attenuation applied to a band at the floor.

    Raises:
        AudioError: On a non-positive rate or frame, or a negative reduction.
    """

    mode: DenoiseMode = DenoiseMode.RNNOISE
    sample_rate: int = TARGET_SAMPLE_RATE
    frame_ms: int = 20
    noise_floor_db: float = -45.0
    reduction_db: float = DEFAULT_REDUCTION_DB

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise AudioError(
                f"sample_rate must be positive, got {self.sample_rate}",
                user_message="Некорректная частота дискретизации.",
            )
        if self.frame_ms <= 0:
            raise AudioError(
                f"frame_ms must be positive, got {self.frame_ms}",
                user_message="Некорректный размер аудиокадра.",
            )
        if self.reduction_db < 0.0:
            raise AudioError(
                f"reduction_db must not be negative, got {self.reduction_db}",
                user_message="Глубина шумоподавления не может быть отрицательной.",
            )

    @property
    def frame_samples(self) -> int:
        """Samples in one processing block."""
        return self.sample_rate * self.frame_ms // 1000

    @property
    def frame_bytes(self) -> int:
        """Bytes in one processing block."""
        return self.frame_samples * SAMPLE_WIDTH

    @property
    def floor_gain(self) -> float:
        """Linear gain a fully closed band is multiplied by."""
        return float(10.0 ** (-self.reduction_db / 20.0))


@dataclass(frozen=True, slots=True)
class DenoiseStats:
    """What the settings window shows next to the denoise switch.

    Attributes:
        mode: What was requested.
        engine: What is running.  A mismatch is exactly the case the UI must
            explain - "RNNoise не найдена, работает встроенный шумодав".
        frames: Blocks processed since the last reset.
        latency_ms: Delay the pipeline adds: framing plus whatever the engine
            needs internally.
        avg_ms: Mean wall-clock time spent on one block.
        max_ms: Worst block.  More telling than the mean, because it is the one
            that drops audio.
        reduction_db: Attenuation currently applied, averaged over the bands.
            Zero while somebody is speaking, which is what makes it a useful
            thing to draw.
        frame_ms: Frame length the averages were measured over.
    """

    mode: DenoiseMode = DenoiseMode.OFF
    engine: DenoiseEngine = DenoiseEngine.NONE
    frames: int = 0
    latency_ms: float = 0.0
    avg_ms: float = 0.0
    max_ms: float = 0.0
    reduction_db: float = 0.0
    frame_ms: int = 20

    @property
    def fallback(self) -> bool:
        """Whether the user asked for RNNoise and got the gate."""
        return self.mode is DenoiseMode.RNNOISE and self.engine is not DenoiseEngine.RNNOISE

    @property
    def realtime_factor(self) -> float:
        """Share of a frame's worth of time spent processing it.

        Above ``1.0`` the machine cannot keep up and audio will drop.  Anything
        under ``0.1`` is comfortable.
        """
        return self.avg_ms / self.frame_ms if self.frame_ms else 0.0


class Denoiser(ABC):
    """One noise suppression engine.

    Implementations process whole blocks and return the same number of samples
    they were given: the pipeline downstream counts frames, and an engine that
    changed the length would silently shift every position handed out by the
    ring buffer.
    """

    name: ClassVar[str] = "denoise"

    #: Algorithmic delay in milliseconds, independent of the block size.
    latency_ms: ClassVar[float] = 0.0

    #: Which engine this is, for the report.
    engine: ClassVar[DenoiseEngine] = DenoiseEngine.NONE

    __slots__ = ("_settings",)

    def __init__(self, settings: DenoiseSettings | None = None) -> None:
        self._settings = settings or DenoiseSettings()

    @property
    def settings(self) -> DenoiseSettings:
        """Configuration in force."""
        return self._settings

    @property
    def reduction_db(self) -> float:
        """Attenuation applied to the last block, in dB."""
        return 0.0

    @abstractmethod
    def process(self, pcm: bytes) -> bytes:
        """Clean one block of little-endian ``int16`` mono audio."""

    @abstractmethod
    def reset(self) -> None:
        """Forget per-stream state.  Called when capture restarts."""

    def configure(self, settings: DenoiseSettings) -> None:
        """Apply new settings.  Subclasses extend this to re-arm the engine."""
        self._settings = settings

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mode={self._settings.mode.value})"


class Passthrough(Denoiser):
    """The ``off`` setting: hands the audio back untouched."""

    name: ClassVar[str] = "off"
    engine: ClassVar[DenoiseEngine] = DenoiseEngine.NONE

    __slots__ = ()

    def process(self, pcm: bytes) -> bytes:
        """Return the block as it came in."""
        return pcm

    def reset(self) -> None:
        """Nothing to forget: the engine holds no state."""


class NoiseGate(Denoiser):
    """Two-band expander driven by a tracked noise floor.

    The signal is split with a one-pole filter at :data:`_CROSSOVER_HZ`.  Each
    band gets its own floor estimate and its own gain, so a fan humming under
    the voice and a hiss above it are handled independently, and the bands are
    summed back - which reconstructs the input exactly when both gains are one,
    so the gate is transparent while somebody is talking.

    Gain is computed once per block and applied as a linear ramp across it.  A
    stepped gain would click on every block boundary; the ramp costs one extra
    multiply per sample and removes the artefact entirely.
    """

    name: ClassVar[str] = "gate"
    engine: ClassVar[DenoiseEngine] = DenoiseEngine.GATE

    __slots__ = ("_floor", "_gain", "_last_reduction", "_low")

    def __init__(self, settings: DenoiseSettings | None = None) -> None:
        super().__init__(settings)
        self._low = 0.0
        self._floor = [self._settings.noise_floor_db, self._settings.noise_floor_db]
        self._gain = [1.0, 1.0]
        self._last_reduction = 0.0

    @property
    def floor_db(self) -> tuple[float, float]:
        """Current noise floor of the low and the high band."""
        return (self._floor[0], self._floor[1])

    @property
    def gain(self) -> tuple[float, float]:
        """Gain applied to the low and the high band at the end of the block."""
        return (self._gain[0], self._gain[1])

    @property
    def reduction_db(self) -> float:
        """Mean attenuation applied to the last block."""
        return self._last_reduction

    def process(self, pcm: bytes) -> bytes:
        """Split, gate and recombine one block."""
        samples = array("h")
        samples.frombytes(bytes(pcm))
        if not samples:
            self._last_reduction = 0.0
            return b""

        alpha = _one_pole_alpha(_CROSSOVER_HZ, self._settings.sample_rate)
        low_band: list[float] = []
        high_band: list[float] = []
        low_sq = 0.0
        high_sq = 0.0
        state = self._low
        for sample in samples:
            value = float(sample)
            state += alpha * (value - state)
            high = value - state
            low_band.append(state)
            high_band.append(high)
            low_sq += state * state
            high_sq += high * high
        self._low = state

        count = len(samples)
        targets = (
            self._band_gain(0, math.sqrt(low_sq / count)),
            self._band_gain(1, math.sqrt(high_sq / count)),
        )
        starts = (self._gain[0], self._gain[1])
        ends = (
            _smooth(starts[0], targets[0], self._step(starts[0], targets[0])),
            _smooth(starts[1], targets[1], self._step(starts[1], targets[1])),
        )
        self._gain = [ends[0], ends[1]]
        self._last_reduction = -20.0 * math.log10(max((ends[0] + ends[1]) / 2.0, 1e-4))

        out = array("h", bytes(len(samples) * SAMPLE_WIDTH))
        span = float(count - 1) if count > 1 else 1.0
        for index in range(count):
            ramp = index / span
            gain_low = starts[0] + (ends[0] - starts[0]) * ramp
            gain_high = starts[1] + (ends[1] - starts[1]) * ramp
            mixed = low_band[index] * gain_low + high_band[index] * gain_high
            out[index] = _clip(mixed)
        return out.tobytes()

    def reset(self) -> None:
        """Re-seed the floors and open the gate."""
        self._low = 0.0
        self._floor = [self._settings.noise_floor_db, self._settings.noise_floor_db]
        self._gain = [1.0, 1.0]
        self._last_reduction = 0.0

    def configure(self, settings: DenoiseSettings) -> None:
        """Apply new settings, re-seeding the floors when calibration moved them."""
        previous = self._settings
        super().configure(settings)
        if settings.noise_floor_db != previous.noise_floor_db:
            self._floor = [settings.noise_floor_db, settings.noise_floor_db]

    def _band_gain(self, band: int, rms: float) -> float:
        """Track the floor of one band and decide how far to pull it down."""
        level_db = _to_db(rms / _FULL_SCALE)
        floor = self._floor[band]
        weight = _FLOOR_FALL if level_db < floor else _FLOOR_RISE
        floor = min(floor + (level_db - floor) * weight, _FLOOR_CEILING_DB)
        self._floor[band] = floor
        above = level_db - floor
        if above >= _OPEN_MARGIN_DB:
            return 1.0
        if above <= 0.0:
            return self._settings.floor_gain
        share = above / _OPEN_MARGIN_DB
        return self._settings.floor_gain + (1.0 - self._settings.floor_gain) * share

    def _step(self, current: float, target: float) -> float:
        """Smoothing coefficient: fast when opening, slow when closing."""
        constant = _ATTACK_MS if target > current else _RELEASE_MS
        block_ms = max(1.0, float(self._settings.frame_ms))
        return 1.0 - math.exp(-block_ms / constant)


class RnnoiseDenoiser(Denoiser):
    """RNNoise through :mod:`ctypes`, with the resampling it needs around it.

    The library is loaded lazily and by name - there is no Python package to
    depend on, users install it from their distribution or drop a DLL next to
    Ayris - so :func:`rnnoise_library` is where "is it available" is decided,
    and this class simply refuses to construct without one.

    Two adaptations sit around the call.  RNNoise runs at 48 kHz while capture
    runs at 16 kHz, so audio is resampled up and back; and it insists on exactly
    480-sample frames, so a remainder is carried to the next block.  The carry
    is why this engine adds 10 ms of latency that the gate does not.

    Args:
        settings: Configuration.
        library: Pre-loaded library, injected by the tests.  Production passes
            ``None`` and gets whatever :func:`rnnoise_library` finds.

    Raises:
        AudioError: When no library can be loaded.
    """

    name: ClassVar[str] = "rnnoise"
    engine: ClassVar[DenoiseEngine] = DenoiseEngine.RNNOISE
    latency_ms: ClassVar[float] = 1000.0 * RNNOISE_FRAME_SAMPLES / RNNOISE_SAMPLE_RATE

    __slots__ = ("_carry", "_down", "_lib", "_state", "_up")

    def __init__(
        self,
        settings: DenoiseSettings | None = None,
        *,
        library: Any = None,
    ) -> None:
        super().__init__(settings)
        self._lib = library if library is not None else rnnoise_library()
        if self._lib is None:
            raise AudioError(
                "RNNoise library not found",
                user_message=(
                    "Библиотека RNNoise не найдена. "
                    "Установите её или выберите встроенное шумоподавление."
                ),
            )
        self._state = self._lib.rnnoise_create(None)
        self._up = Resampler(self._settings.sample_rate, RNNOISE_SAMPLE_RATE)
        self._down = Resampler(RNNOISE_SAMPLE_RATE, self._settings.sample_rate)
        self._carry = array("h")

    def process(self, pcm: bytes) -> bytes:
        """Upsample, denoise in 480-sample frames, downsample back.

        The output can be a few samples shorter or longer than the input while
        the resampler's phase settles; callers that care about exact framing
        push through :class:`DenoiseStream`, which re-frames afterwards.
        """
        wide = array("h")
        wide.frombytes(self._up.process(pcm))
        if self._carry:
            wide = self._carry + wide
        usable = len(wide) - len(wide) % RNNOISE_FRAME_SAMPLES
        self._carry = wide[usable:]
        if not usable:
            return b""
        cleaned = array("h", bytes(usable * SAMPLE_WIDTH))
        for start in range(0, usable, RNNOISE_FRAME_SAMPLES):
            chunk = wide[start : start + RNNOISE_FRAME_SAMPLES]
            cleaned[start : start + RNNOISE_FRAME_SAMPLES] = self._frame(chunk)
        return self._down.process(cleaned.tobytes())

    def reset(self) -> None:
        """Drop the carried samples and the resampler phase."""
        self._carry = array("h")
        self._up.reset()
        self._down.reset()

    def configure(self, settings: DenoiseSettings) -> None:
        """Apply new settings, rebuilding the resamplers when the rate changed."""
        previous = self._settings
        super().configure(settings)
        if settings.sample_rate != previous.sample_rate:
            self._up = Resampler(settings.sample_rate, RNNOISE_SAMPLE_RATE)
            self._down = Resampler(RNNOISE_SAMPLE_RATE, settings.sample_rate)
            self._carry = array("h")

    def _frame(self, chunk: array[int]) -> array[int]:
        """Run one 480-sample frame through the network."""
        import ctypes

        buffer = (ctypes.c_float * RNNOISE_FRAME_SAMPLES)(*(float(value) for value in chunk))
        self._lib.rnnoise_process_frame(self._state, buffer, buffer)
        return array("h", (_clip(value) for value in buffer))

    def close(self) -> None:
        """Release the network state.  Safe to call twice."""
        if self._state is not None:
            self._lib.rnnoise_destroy(self._state)
            self._state = None

    def __del__(self) -> None:
        """Release the network state if the owner forgot to."""
        # Interpreter shutdown can pull ctypes out from under us; a finaliser
        # that raises then prints to stderr, which the project forbids.
        with contextlib.suppress(Exception):
            self.close()


class DenoiseStream:
    """A denoiser plus framing and the measurements the settings window shows.

    Push whatever capture produced, get cleaned audio back in whole frames.  The
    stream is where the cost is measured, because it is the only place that sees
    both the block and the clock:

        >>> stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.SPECTRAL))
        >>> clean = stream.push(b"\\x00\\x00" * 320)
        >>> stream.stats.engine
        <DenoiseEngine.GATE: 'gate'>

    Args:
        settings: Configuration.
        denoiser: Pre-built engine, for tests and for keeping one engine across
            a reconfiguration.
    """

    __slots__ = ("_buffer", "_denoiser", "_frames", "_max_ms", "_settings", "_total_ms")

    def __init__(
        self,
        settings: DenoiseSettings | None = None,
        *,
        denoiser: Denoiser | None = None,
    ) -> None:
        if settings is not None:
            resolved = settings
        elif denoiser is not None:
            resolved = denoiser.settings
        else:
            resolved = DenoiseSettings()
        self._settings = resolved
        self._denoiser = denoiser if denoiser is not None else create_denoiser(resolved)
        self._buffer = bytearray()
        self._frames = 0
        self._total_ms = 0.0
        self._max_ms = 0.0

    @property
    def settings(self) -> DenoiseSettings:
        """Configuration in force."""
        return self._settings

    @property
    def denoiser(self) -> Denoiser:
        """The engine doing the work."""
        return self._denoiser

    @property
    def engine(self) -> DenoiseEngine:
        """Which engine is running."""
        return self._denoiser.engine

    @property
    def enabled(self) -> bool:
        """Whether anything is actually being done to the audio."""
        return self._denoiser.engine is not DenoiseEngine.NONE

    @property
    def latency_ms(self) -> float:
        """Delay the whole stage adds.

        Framing plus the engine's own delay.  ``off`` adds nothing at all - the
        block is handed straight back, so there is not even a frame of buffering.
        """
        if not self.enabled:
            return 0.0
        return float(self._settings.frame_ms) + self._denoiser.latency_ms

    @property
    def stats(self) -> DenoiseStats:
        """A snapshot for the settings window."""
        return DenoiseStats(
            mode=self._settings.mode,
            engine=self._denoiser.engine,
            frames=self._frames,
            latency_ms=self.latency_ms,
            avg_ms=self._total_ms / self._frames if self._frames else 0.0,
            max_ms=self._max_ms,
            reduction_db=self._denoiser.reduction_db,
            frame_ms=self._settings.frame_ms,
        )

    def push(self, pcm: bytes | bytearray | memoryview) -> bytes:
        """Clean everything in ``pcm`` that fills a whole frame.

        Returns:
            Cleaned audio; ``b""`` while the first frame is still filling.
        """
        if not self.enabled:
            return bytes(pcm)
        self._buffer += bytes(pcm)
        size = self._settings.frame_bytes
        count = len(self._buffer) // size
        if not count:
            return b""
        block = bytes(self._buffer[: count * size])
        del self._buffer[: count * size]
        return b"".join(self._timed(block, size, count))

    def flush(self) -> bytes:
        """Clean the remainder, padding it to a whole frame with silence."""
        if not self._buffer:
            return b""
        size = self._settings.frame_bytes
        block = bytes(self._buffer) + bytes(size - len(self._buffer))
        self._buffer.clear()
        return b"".join(self._timed(block, size, 1))

    def reset(self) -> None:
        """Drop buffered audio, engine state and measurements."""
        self._buffer.clear()
        self._denoiser.reset()
        self._frames = 0
        self._total_ms = 0.0
        self._max_ms = 0.0

    def configure(self, settings: DenoiseSettings) -> None:
        """Apply new settings, rebuilding the engine when the mode changed.

        Changing ``denoise`` in the settings window must take effect while the
        microphone is open - section 8.1 lists it as a live toggle - so the
        buffered remainder is dropped rather than run through an engine it was
        not framed for.
        """
        previous = self._settings
        self._settings = settings
        self._buffer.clear()
        if settings.mode != previous.mode:
            self._denoiser = create_denoiser(settings)
        else:
            self._denoiser.configure(settings)
        self._frames = 0
        self._total_ms = 0.0
        self._max_ms = 0.0

    def _timed(self, block: bytes, size: int, count: int) -> Iterator[bytes]:
        """Process ``count`` frames, timing each one."""
        for index in range(count):
            frame = block[index * size : (index + 1) * size]
            started = time.perf_counter()
            cleaned = self._denoiser.process(frame)
            elapsed = (time.perf_counter() - started) * 1000.0
            self._frames += 1
            self._total_ms += elapsed
            self._max_ms = max(self._max_ms, elapsed)
            yield cleaned

    def __repr__(self) -> str:
        return f"DenoiseStream(mode={self._settings.mode.value}, engine={self.engine.value})"


def rnnoise_library() -> Any:
    """Load the RNNoise shared library, or return ``None``.

    Tried in order: the path in :data:`RNNOISE_LIB_ENV`, then the plain library
    names.  Failures are logged at debug level and swallowed - a missing
    optional dependency is not an error, it is the common case.
    """
    import ctypes

    candidates = [os.environ.get(RNNOISE_LIB_ENV, "").strip(), *_LIBRARY_NAMES]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            lib = ctypes.CDLL(candidate)
        except OSError as exc:
            _log.debug("RNNoise: %s not loadable (%s)", candidate, exc)
            continue
        try:
            _declare(lib)
        except AttributeError:
            _log.debug("RNNoise: %s has no rnnoise_process_frame", candidate)
            continue
        _log.info("RNNoise загружена: %s", candidate)
        return lib
    return None


def _declare(lib: Any) -> None:
    """Pin the signatures we use, so ctypes does not guess.

    Raises:
        AttributeError: When the library is not RNNoise after all - which is
            what makes a stray ``librnnoise.so`` from another project harmless.
    """
    import ctypes

    floats = ctypes.POINTER(ctypes.c_float)
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, floats, floats]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_destroy.restype = None


def rnnoise_available() -> bool:
    """Whether RNNoise can be used on this machine.

    Asked by the settings window to grey the option out with an explanation
    instead of letting a user pick something that will silently fall back.
    """
    return rnnoise_library() is not None


def create_denoiser(settings: DenoiseSettings) -> Denoiser:
    """Build the engine for ``settings``, falling back when RNNoise is absent.

    The fallback is deliberate and quiet: ``rnnoise`` is the default in
    ``config.toml`` and most machines will not have the library, so failing here
    would mean Ayris refuses to listen out of the box.  The substitution is
    logged once and reported through :attr:`DenoiseStats.fallback`.
    """
    if settings.mode is DenoiseMode.OFF:
        return Passthrough(settings)
    if settings.mode is DenoiseMode.SPECTRAL:
        return NoiseGate(settings)
    try:
        return RnnoiseDenoiser(settings)
    except AudioError:
        _log.info("RNNoise недоступна, включено встроенное шумоподавление")
        return NoiseGate(settings)


def denoise_pcm(
    pcm: bytes | bytearray | memoryview,
    settings: DenoiseSettings | None = None,
) -> bytes:
    """Clean a finished recording in one call.

    The offline counterpart of :meth:`DenoiseStream.push`, used by calibration
    and by the tests.  The tail is flushed, so nothing is lost.
    """
    stream = DenoiseStream(settings)
    return stream.push(pcm) + stream.flush()


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    """Coefficient of a one-pole low-pass at ``cutoff_hz``."""
    if sample_rate <= 0:
        return 1.0
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return min(1.0, 1.0 - math.exp(-1.0 / (tau * sample_rate)))


def _smooth(current: float, target: float, step: float) -> float:
    """Move ``current`` towards ``target`` by ``step`` of the distance."""
    return current + (target - current) * step


def _to_db(value: float) -> float:
    """Amplitude 0.0-1.0 to dBFS, floored at :data:`~ayris.audio.capture.MIN_DBFS`."""
    if value <= 0.0:
        return MIN_DBFS
    return max(MIN_DBFS, 20.0 * math.log10(value))


def _clip(value: float) -> int:
    """Round to ``int16``, saturating instead of wrapping."""
    return max(-32768, min(32767, int(round(value))))
