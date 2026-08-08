"""Microphone capture: device stream in, normalised 16 kHz mono PCM out.

The pipeline is split across three threads because the first of them is not
really ours:

audio callback (PortAudio's real-time thread)
    Copies the driver's block into a deque and returns.  Nothing else - no
    logging, no resampling, no locks that a slow consumer could hold.  A
    callback that overruns its deadline produces a click in the recording and,
    on WASAPI, can drop the stream entirely.
processing thread
    Drains the deque and does the actual work: downmix to mono, resample to
    16 kHz, apply gain through a soft limiter, push into the ring buffer, feed
    the frame sink, and publish the level at a fixed rate.
monitor thread
    Watches for a device that stopped delivering audio, re-scans the hardware
    while the stream is closed, and brings capture back up when the device
    returns.

Everything numeric here is deliberately written against the standard library
(:mod:`array`, :mod:`math`) rather than NumPy.  A 16 kHz mono stream is 320
samples per 20 ms block; the pure-Python loops cost well under a percent of one
core, and in exchange the audio worker starts without a heavyweight import and
stays testable anywhere.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
import wave
from array import array
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Final

from ayris.audio.devices import (
    AudioBackend,
    AudioDevice,
    AudioStream,
    DeviceChange,
    DeviceDirection,
    SoundDeviceBackend,
    StreamRequest,
    diff_devices,
    list_devices,
    resolve_device,
)
from ayris.audio.ring_buffer import DEFAULT_PRE_ROLL_MS, SAMPLE_WIDTH, BufferRead, RingBuffer
from ayris.core.errors import AudioError

__all__ = [
    "AudioCapture",
    "AudioLevel",
    "CaptureCallbacks",
    "CaptureSettings",
    "CaptureState",
    "CaptureStats",
    "Resampler",
    "apply_gain",
    "downmix",
    "pcm_level",
    "write_wav",
]

_log = logging.getLogger(__name__)

#: Everything downstream of capture - VAD, wake word, Vosk, Whisper - expects
#: 16 kHz mono.
TARGET_SAMPLE_RATE: Final = 16000

#: Full scale for signed 16-bit samples.
_FULL_SCALE: Final = 32768.0
_INT16_MAX: Final = 32767
_INT16_MIN: Final = -32768

#: Level events are published at 20 Hz.  Faster than that is invisible in the
#: overlay and would put a message on the IPC channel every other audio block.
DEFAULT_LEVEL_INTERVAL_MS: Final = 50

#: Silence floor reported instead of ``-inf`` dBFS.
MIN_DBFS: Final = -100.0

#: Above this fraction of full scale the limiter starts bending the signal.
_LIMITER_KNEE: Final = 0.9

#: ``array('h')`` is native-endian; PCM on the wire is little-endian.
_NEEDS_SWAP: Final = sys.byteorder != "little"


class CaptureState(StrEnum):
    """What the capture pipeline is doing."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    #: The device stopped responding or disappeared.  The monitor thread keeps
    #: looking for it and resumes on its own.
    DEVICE_LOST = "device_lost"


@dataclass(frozen=True, slots=True)
class AudioLevel:
    """Loudness of one measurement window, normalised to 0.0-1.0."""

    rms: float = 0.0
    peak: float = 0.0
    clipped: bool = False

    @property
    def rms_db(self) -> float:
        """RMS in dBFS, floored at :data:`MIN_DBFS`."""
        return _to_db(self.rms)

    @property
    def peak_db(self) -> float:
        """Peak in dBFS, floored at :data:`MIN_DBFS`."""
        return _to_db(self.peak)


@dataclass(frozen=True, slots=True)
class CaptureSettings:
    """Everything the pipeline needs to know before opening a stream.

    Attributes:
        device: Identifier or name from the settings; empty means the system
            default.
        sample_rate: Rate delivered to consumers, after resampling.
        frame_ms: Block size requested from the driver, and the granularity at
            which audio reaches the ring buffer.
        gain: Software gain applied before the level is measured, so the meter
            shows what the recognizer will hear.
        buffer_seconds: Size of the rolling window kept for pre-roll.
        pre_roll_ms: Default amount of audio :meth:`AudioCapture.pre_roll`
            returns.
        level_interval_ms: Minimum spacing between level notifications.
        mute_stops_stream: Whether muting closes the device (the microphone
            indicator in the tray goes out, other applications can grab it) or
            merely zeroes the samples (unmuting is instant).
        device_poll_sec: How often the monitor thread checks on the device.
        limiter: Whether gain above unity is passed through a soft limiter.
    """

    device: str = ""
    sample_rate: int = TARGET_SAMPLE_RATE
    frame_ms: int = 20
    gain: float = 1.0
    buffer_seconds: float = 30.0
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS
    level_interval_ms: int = DEFAULT_LEVEL_INTERVAL_MS
    mute_stops_stream: bool = False
    device_poll_sec: float = 1.5
    limiter: bool = True

    @property
    def frame_samples(self) -> int:
        """Frames per block at the target rate."""
        return max(1, self.sample_rate * self.frame_ms // 1000)


@dataclass(frozen=True, slots=True)
class CaptureStats:
    """Snapshot of the pipeline, for ``status`` calls and the DevTools panel."""

    state: CaptureState
    device_id: str = ""
    device_name: str = ""
    device_sample_rate: int = 0
    device_channels: int = 0
    muted: bool = False
    recording: bool = False
    frames: int = 0
    dropped_blocks: int = 0
    overflows: int = 0
    buffered_ms: int = 0
    level: AudioLevel = field(default_factory=AudioLevel)


def _ignore_level(_level: AudioLevel) -> None:
    """Default :attr:`CaptureCallbacks.on_level`."""


def _ignore_state(_state: CaptureState, _detail: str) -> None:
    """Default :attr:`CaptureCallbacks.on_state`."""


def _ignore_devices(_change: DeviceChange) -> None:
    """Default :attr:`CaptureCallbacks.on_devices`."""


def _ignore_frames(_pcm: bytes) -> None:
    """Default :attr:`CaptureCallbacks.on_frames`."""


@dataclass(frozen=True, slots=True)
class CaptureCallbacks:
    """Where the pipeline reports to.

    Every callback runs on the processing or monitor thread, never on the audio
    callback, so they may allocate and log.  They must still return quickly:
    time spent here is time not spent draining the block queue.
    """

    #: Throttled to :attr:`CaptureSettings.level_interval_ms`.
    on_level: Callable[[AudioLevel], None] = _ignore_level
    #: State transitions, with a short Russian explanation.
    on_state: Callable[[CaptureState, str], None] = _ignore_state
    #: Devices appeared or disappeared.
    on_devices: Callable[[DeviceChange], None] = _ignore_devices
    #: Every normalised block, for VAD and wake word (tasks 08 and 09).
    on_frames: Callable[[bytes], None] = _ignore_frames


# ------------------------------------------------------------------------ dsp


def _to_samples(pcm: bytes | bytearray | memoryview) -> array[int]:
    """Decode little-endian ``int16`` PCM into an ``array``."""
    samples: array[int] = array("h")
    samples.frombytes(bytes(pcm))
    if _NEEDS_SWAP:
        samples.byteswap()
    return samples


def _to_bytes(samples: array[int]) -> bytes:
    """Encode an ``array`` back into little-endian ``int16`` PCM."""
    if _NEEDS_SWAP:
        samples = samples[:]
        samples.byteswap()
    return samples.tobytes()


def _to_db(value: float) -> float:
    """Convert a 0.0-1.0 amplitude to dBFS."""
    if value <= 0.0:
        return MIN_DBFS
    return max(MIN_DBFS, 20.0 * math.log10(value))


def pcm_level(pcm: bytes | bytearray | memoryview) -> AudioLevel:
    """Measure RMS and peak of a PCM block.

    Returns:
        Amplitudes normalised to 0.0-1.0, with ``clipped`` set when at least one
        sample sits at the rail - the cue that the gain is too high.
    """
    samples = _to_samples(pcm)
    count = len(samples)
    if count == 0:
        return AudioLevel()
    high = max(samples)
    low = min(samples)
    peak = max(high, -low)
    energy = sum(sample * sample for sample in samples)
    return AudioLevel(
        rms=min(1.0, math.sqrt(energy / count) / _FULL_SCALE),
        peak=min(1.0, peak / _FULL_SCALE),
        clipped=high >= _INT16_MAX or low <= _INT16_MIN,
    )


def apply_gain(
    pcm: bytes | bytearray | memoryview,
    gain: float,
    *,
    limiter: bool = True,
) -> bytes:
    """Scale a block by ``gain``.

    Amplifying a quiet microphone is the difference between usable and useless
    recognition, but naive scaling turns every loud syllable into square-wave
    clipping.  Above :data:`_LIMITER_KNEE` the curve bends through ``tanh``
    instead: monotonic, smooth, and audibly a compressor rather than distortion.

    Args:
        pcm: Little-endian ``int16`` samples.
        gain: Linear factor.  ``1.0`` returns the input untouched.
        limiter: Set to ``False`` to hard-clip instead, for tests that need an
            exactly predictable curve.

    Returns:
        A new PCM block of the same length.
    """
    if gain == 1.0:
        return bytes(pcm)
    samples = _to_samples(pcm)
    knee = _LIMITER_KNEE * _FULL_SCALE
    headroom = _INT16_MAX - knee
    for index, sample in enumerate(samples):
        value = sample * gain
        magnitude = abs(value)
        if magnitude > knee:
            if limiter:
                magnitude = knee + headroom * math.tanh((magnitude - knee) / headroom)
            else:
                magnitude = min(magnitude, float(_INT16_MAX))
            value = math.copysign(magnitude, value)
        samples[index] = max(_INT16_MIN, min(_INT16_MAX, int(value)))
    return _to_bytes(samples)


def downmix(pcm: bytes | bytearray | memoryview, channels: int) -> bytes:
    """Average interleaved channels down to mono.

    Averaging rather than picking channel 0: on a stereo headset the microphone
    is sometimes wired to the right channel only, and taking the left would give
    silence.
    """
    if channels <= 1:
        return bytes(pcm)
    samples = _to_samples(pcm)
    frames = len(samples) // channels
    mono: array[int] = array("h", bytes(frames * SAMPLE_WIDTH))
    for frame in range(frames):
        base = frame * channels
        mono[frame] = sum(samples[base : base + channels]) // channels
    return _to_bytes(mono)


class Resampler:
    """Convert a mono ``int16`` stream between sample rates.

    Two paths, both stateful so that block boundaries stay seamless:

    * an exact integer ratio (48000 -> 16000, the common case on Windows) uses a
      box average, which both decimates and low-passes;
    * anything else (44100 -> 16000) uses linear interpolation with the phase
      carried across blocks.

    Neither is a polyphase filter - linear interpolation aliases above ~6 kHz.
    That is acceptable here: speech recognition models are trained on telephone
    and headset audio, and the alternative costs a NumPy dependency in the audio
    worker.
    """

    __slots__ = ("_carry", "_factor", "_phase", "_ratio", "_source_rate", "_tail", "_target_rate")

    def __init__(self, source_rate: int, target_rate: int) -> None:
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError(f"invalid rates {source_rate} -> {target_rate}")
        self._source_rate = source_rate
        self._target_rate = target_rate
        self._ratio = source_rate / target_rate
        self._factor = source_rate // target_rate if source_rate % target_rate == 0 else 0
        self._carry: array[int] = array("h")
        self._tail: array[int] = array("h")
        self._phase = 0.0

    @property
    def source_rate(self) -> int:
        """Input rate."""
        return self._source_rate

    @property
    def target_rate(self) -> int:
        """Output rate."""
        return self._target_rate

    @property
    def passthrough(self) -> bool:
        """Whether the rates match and no work is needed."""
        return self._source_rate == self._target_rate

    def reset(self) -> None:
        """Forget carried samples.  Call when the stream restarts."""
        self._carry = array("h")
        self._tail = array("h")
        self._phase = 0.0

    def process(self, pcm: bytes | bytearray | memoryview) -> bytes:
        """Resample one block."""
        if self.passthrough:
            return bytes(pcm)
        samples = _to_samples(pcm)
        if not samples:
            return b""
        if self._factor > 1:
            return _to_bytes(self._decimate(samples))
        return _to_bytes(self._interpolate(samples))

    def _decimate(self, samples: array[int]) -> array[int]:
        """Box-average an exact integer ratio."""
        factor = self._factor
        buffer = self._carry + samples if self._carry else samples
        count = len(buffer) // factor
        out: array[int] = array("h", bytes(count * SAMPLE_WIDTH))
        for index in range(count):
            base = index * factor
            out[index] = sum(buffer[base : base + factor]) // factor
        self._carry = buffer[count * factor :]
        return out

    def _interpolate(self, samples: array[int]) -> array[int]:
        """Linearly interpolate a fractional ratio, carrying the phase."""
        buffer = self._tail + samples if self._tail else samples
        size = len(buffer)
        ratio = self._ratio
        phase = self._phase
        out: array[int] = array("h")
        while phase + 1.0 < size:
            index = int(phase)
            fraction = phase - index
            first = buffer[index]
            out.append(int(first + (buffer[index + 1] - first) * fraction))
            phase += ratio
        # Keep the last sample the next block will interpolate from.
        base = min(int(phase), size - 1)
        self._tail = buffer[base:]
        self._phase = phase - base
        return out


def write_wav(path: Path, pcm: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Write mono ``int16`` PCM to a WAV file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


class _WavRecorder:
    """Incremental WAV writer used for the debug dump."""

    __slots__ = ("_frames", "_handle", "_path", "_sample_rate")

    def __init__(self, path: Path, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._sample_rate = sample_rate
        self._frames = 0
        self._handle = wave.open(str(path), "wb")  # noqa: SIM115 - closed by ``close``
        self._handle.setnchannels(1)
        self._handle.setsampwidth(SAMPLE_WIDTH)
        self._handle.setframerate(sample_rate)

    @property
    def path(self) -> Path:
        """File being written."""
        return self._path

    @property
    def duration_ms(self) -> int:
        """How much audio has been written so far."""
        return self._frames * 1000 // self._sample_rate

    def write(self, pcm: bytes) -> None:
        """Append a block."""
        self._handle.writeframes(pcm)
        self._frames += len(pcm) // SAMPLE_WIDTH

    def close(self) -> Path:
        """Finish the file and return its path."""
        self._handle.close()
        return self._path


# -------------------------------------------------------------------- capture


@dataclass(slots=True)
class _LevelAccumulator:
    """Aggregates levels between two notifications.

    Emitting only the last block's level would let a 20 ms clap fall between
    two notifications and never show up.  Peak is taken as the maximum and RMS
    as the true RMS over the whole window.
    """

    energy: float = 0.0
    samples: int = 0
    peak: float = 0.0
    clipped: bool = False
    deadline: float = 0.0

    def add(self, level: AudioLevel, count: int) -> None:
        """Fold one block's measurement in."""
        self.energy += level.rms * level.rms * count
        self.samples += count
        self.peak = max(self.peak, level.peak)
        self.clipped = self.clipped or level.clipped

    def take(self) -> AudioLevel:
        """Return the aggregate and start a new window."""
        level = AudioLevel(
            rms=math.sqrt(self.energy / self.samples) if self.samples else 0.0,
            peak=self.peak,
            clipped=self.clipped,
        )
        self.energy = 0.0
        self.samples = 0
        self.peak = 0.0
        self.clipped = False
        return level


@dataclass(slots=True)
class _StreamState:
    """The open device and the conversion it needs."""

    stream: AudioStream
    device: AudioDevice
    channels: int
    resampler: Resampler = field(default_factory=lambda: Resampler(1, 1))


class AudioCapture:
    """Owns the microphone stream and the normalised audio it produces.

    Args:
        settings: Initial configuration; replace it later with
            :meth:`configure`.
        backend: Audio backend.  Defaults to
            :class:`~ayris.audio.devices.SoundDeviceBackend`; tests pass a fake
            so that none of this needs a sound card.
        callbacks: Where levels, state changes and frames are reported.

    All public methods are safe to call from any thread.
    """

    __slots__ = (
        "_accumulator",
        "_backend",
        "_blocks",
        "_callbacks",
        "_devices",
        "_dropped",
        "_last_level",
        "_lock",
        "_max_blocks",
        "_monitor",
        "_muted",
        "_overflows",
        "_pending",
        "_processor",
        "_recorder",
        "_ring",
        "_seen_blocks",
        "_settings",
        "_state",
        "_stopping",
        "_stream",
        "_watchdog",
    )

    def __init__(
        self,
        settings: CaptureSettings | None = None,
        *,
        backend: AudioBackend | None = None,
        callbacks: CaptureCallbacks | None = None,
    ) -> None:
        self._settings = settings or CaptureSettings()
        self._backend: AudioBackend = backend if backend is not None else SoundDeviceBackend()
        self._callbacks = callbacks or CaptureCallbacks()
        self._ring = RingBuffer(
            seconds=self._settings.buffer_seconds,
            sample_rate=self._settings.sample_rate,
        )
        # Two seconds of slack: enough to ride out a garbage-collection pause in
        # the processing thread, short enough that recovery does not replay
        # stale audio.
        self._max_blocks = max(8, 2000 // max(1, self._settings.frame_ms))
        self._blocks: deque[bytes] = deque()
        self._lock = threading.RLock()
        self._pending = threading.Event()
        self._stopping = threading.Event()
        self._state = CaptureState.STOPPED
        self._stream: _StreamState | None = None
        self._processor: threading.Thread | None = None
        self._monitor: threading.Thread | None = None
        self._muted = False
        self._recorder: _WavRecorder | None = None
        self._devices: tuple[AudioDevice, ...] = ()
        self._accumulator = _LevelAccumulator()
        self._last_level = AudioLevel()
        self._dropped = 0
        self._overflows = 0
        self._seen_blocks = 0
        self._watchdog = -1

    # ----------------------------------------------------------------- state

    @property
    def state(self) -> CaptureState:
        """Current pipeline state."""
        with self._lock:
            return self._state

    @property
    def muted(self) -> bool:
        """Whether the microphone is muted in software."""
        with self._lock:
            return self._muted

    @property
    def settings(self) -> CaptureSettings:
        """Configuration in force."""
        with self._lock:
            return self._settings

    @property
    def device(self) -> AudioDevice | None:
        """Device currently open, if any."""
        with self._lock:
            return self._stream.device if self._stream else None

    @property
    def buffer(self) -> RingBuffer:
        """The rolling window of captured audio."""
        return self._ring

    @property
    def level(self) -> AudioLevel:
        """Most recently published level."""
        with self._lock:
            return self._last_level

    def stats(self) -> CaptureStats:
        """Snapshot of the pipeline."""
        with self._lock:
            stream = self._stream
            return CaptureStats(
                state=self._state,
                device_id=stream.device.id if stream else "",
                device_name=stream.device.name if stream else "",
                device_sample_rate=stream.stream.sample_rate if stream else 0,
                device_channels=stream.channels if stream else 0,
                muted=self._muted,
                recording=self._recorder is not None,
                frames=self._ring.position,
                dropped_blocks=self._dropped,
                overflows=self._overflows,
                buffered_ms=self._ring.available_ms,
                level=self._last_level,
            )

    # --------------------------------------------------------------- control

    def start(self) -> None:
        """Open the configured device and begin capturing.

        Raises:
            AudioError: If the device cannot be opened.  The pipeline still
                enters :attr:`CaptureState.DEVICE_LOST` and keeps retrying, so
                plugging a microphone in afterwards is enough to recover.
        """
        with self._lock:
            if self._state is CaptureState.RUNNING:
                return
            self._stopping.clear()
            self._ensure_threads()
            try:
                self._open(fallback=True)
            except AudioError as exc:
                self._set_state(CaptureState.DEVICE_LOST, exc.user_message)
                raise
            self._set_state(CaptureState.RUNNING, "")

    def stop(self) -> None:
        """Close the device and stop the worker threads."""
        with self._lock:
            self._close()
            self._set_state(CaptureState.STOPPED, "")
        self._stopping.set()
        self._pending.set()
        current = threading.current_thread()
        for thread in (self._processor, self._monitor):
            if thread is not None and thread.is_alive() and thread is not current:
                thread.join(timeout=2.0)
        self._processor = None
        self._monitor = None
        with self._lock:
            self._blocks.clear()
            self.stop_recording()

    def pause(self, reason: str = "") -> None:
        """Release the device while keeping the buffered audio.

        The stream is closed rather than merely ignored so that the microphone
        indicator goes out and other applications can use the device.
        """
        with self._lock:
            if self._state in {CaptureState.STOPPED, CaptureState.PAUSED}:
                return
            self._close()
            self._set_state(CaptureState.PAUSED, reason)

    def resume(self) -> None:
        """Re-open the device after :meth:`pause`.

        Raises:
            AudioError: If the device cannot be opened.
        """
        with self._lock:
            if self._state is CaptureState.RUNNING:
                return
            self._stopping.clear()
            self._ensure_threads()
            try:
                self._open(fallback=True)
            except AudioError as exc:
                self._set_state(CaptureState.DEVICE_LOST, exc.user_message)
                raise
            self._set_state(CaptureState.RUNNING, "")

    def mute(self, muted: bool = True) -> None:
        """Turn the microphone off in software.

        Muting keeps the pipeline in :attr:`CaptureState.RUNNING` and keeps
        publishing levels - a flat meter tells the user the microphone is off,
        while a frozen one looks like a crash.  Whether the device is actually
        released depends on :attr:`CaptureSettings.mute_stops_stream`.
        """
        with self._lock:
            if self._muted == muted:
                return
            self._muted = muted
            if not self._settings.mute_stops_stream or self._state is not CaptureState.RUNNING:
                return
            if muted:
                self._close()
            else:
                try:
                    self._open(fallback=True)
                except AudioError as exc:
                    self._set_state(CaptureState.DEVICE_LOST, exc.user_message)
                    raise

    def set_device(self, spec: str) -> AudioDevice | None:
        """Switch to another device without restarting the worker.

        Args:
            spec: Identifier or name; empty selects the system default.

        Returns:
            The device now in use, or ``None`` while capture is stopped.

        Raises:
            AudioError: If the requested device does not exist or cannot be
                opened.  A device named explicitly by the user is not silently
                replaced by the default one.
        """
        with self._lock:
            self._settings = replace(self._settings, device=spec)
            if self._state in {CaptureState.STOPPED, CaptureState.PAUSED}:
                return None
            self._close()
            self._ring.clear()
            try:
                self._open(fallback=False)
            except AudioError as exc:
                self._set_state(CaptureState.DEVICE_LOST, exc.user_message)
                raise
            self._set_state(CaptureState.RUNNING, "")
            return self._stream.device if self._stream else None

    def set_gain(self, gain: float) -> None:
        """Change the software gain, effective from the next block."""
        with self._lock:
            self._settings = replace(self._settings, gain=gain)

    def configure(self, settings: CaptureSettings) -> None:
        """Apply a new configuration, re-opening the stream only if needed."""
        with self._lock:
            previous = self._settings
            self._settings = settings
            if settings.buffer_seconds != previous.buffer_seconds or (
                settings.sample_rate != previous.sample_rate
            ):
                self._ring = RingBuffer(
                    seconds=settings.buffer_seconds,
                    sample_rate=settings.sample_rate,
                )
            self._max_blocks = max(8, 2000 // max(1, settings.frame_ms))
            reopen = (
                settings.device != previous.device
                or settings.sample_rate != previous.sample_rate
                or settings.frame_ms != previous.frame_ms
            )
            if reopen and self._state is CaptureState.RUNNING:
                self._close()
                self._open(fallback=True)

    # ------------------------------------------------------------------ read

    @property
    def position(self) -> int:
        """Absolute frame index of the newest audio."""
        return self._ring.position

    def pre_roll(self, ms: int | None = None) -> bytes:
        """Return the audio that came just before now.

        Used when a wake word fires: the phrase starts before the detector
        reacts, so the recognizer is fed this block followed by the live stream.
        """
        return self._ring.read_last(ms=ms if ms is not None else self._settings.pre_roll_ms)

    def read_recent(self, ms: float) -> bytes:
        """Return the last ``ms`` milliseconds of audio."""
        return self._ring.read_last(ms=ms)

    def read_from(self, position: int, *, max_frames: int = 0) -> BufferRead:
        """Return everything captured since ``position``."""
        return self._ring.read_from(position, max_frames=max_frames)

    def devices(self, *, rescan: bool = False) -> tuple[AudioDevice, ...]:
        """List input devices.

        Args:
            rescan: Ask the backend to re-scan the hardware first.  Ignored
                while a stream is open, because re-initialising PortAudio would
                kill it.

        Raises:
            AudioError: If the backend cannot enumerate at all.
        """
        with self._lock:
            if rescan and self._stream is None:
                self._backend.refresh()
            devices = list_devices(self._backend, DeviceDirection.INPUT)
            self._devices = devices
            return devices

    # ------------------------------------------------------------- recording

    def dump_wav(self, path: Path, ms: int | None = None) -> Path:
        """Write the tail of the ring buffer to a WAV file.

        The quickest way to answer "what did the microphone actually hear" when
        wake word or recognition misbehaves.

        Args:
            path: Destination file.
            ms: How much audio to write; the whole window by default.

        Returns:
            The path written.
        """
        pcm = self._ring.read_last(ms=ms) if ms is not None else self._ring.snapshot()
        return write_wav(path, pcm, self._settings.sample_rate)

    def start_recording(self, path: Path, *, pre_roll_ms: int | None = None) -> Path:
        """Record everything captured from now on into a WAV file.

        Args:
            path: Destination file.
            pre_roll_ms: Audio from the ring buffer to prepend, so a recording
                started by hand still contains the phrase that prompted it.

        Returns:
            The path being written.
        """
        with self._lock:
            self.stop_recording()
            recorder = _WavRecorder(path, self._settings.sample_rate)
            if pre_roll_ms:
                recorder.write(self._ring.read_last(ms=pre_roll_ms))
            self._recorder = recorder
            return recorder.path

    def stop_recording(self) -> Path | None:
        """Finish the WAV file started by :meth:`start_recording`."""
        with self._lock:
            recorder = self._recorder
            self._recorder = None
        return recorder.close() if recorder is not None else None

    @property
    def recording(self) -> Path | None:
        """Path being recorded, if any."""
        with self._lock:
            return self._recorder.path if self._recorder else None

    # -------------------------------------------------------------- internal

    def _ensure_threads(self) -> None:
        """Start the processing and monitor threads once.  Lock held."""
        if self._processor is None or not self._processor.is_alive():
            self._processor = threading.Thread(
                target=self._process_loop, name="ayris-audio-process", daemon=True
            )
            self._processor.start()
        if self._monitor is None or not self._monitor.is_alive():
            self._monitor = threading.Thread(
                target=self._monitor_loop, name="ayris-audio-monitor", daemon=True
            )
            self._monitor.start()

    def _set_state(self, state: CaptureState, detail: str) -> None:
        """Record a state transition and notify.  Lock held."""
        if self._state is state:
            return
        self._state = state
        _log.info("audio capture is %s%s", state.value, f": {detail}" if detail else "")
        self._callbacks.on_state(state, detail)

    def _open(self, *, fallback: bool) -> None:
        """Resolve the configured device and open a stream.  Lock held."""
        settings = self._settings
        device = resolve_device(
            self._backend,
            settings.device,
            direction=DeviceDirection.INPUT,
            fallback=fallback,
        )
        rate, channels = self._negotiate(device, settings)
        request = StreamRequest(
            device_index=device.index,
            sample_rate=rate,
            channels=channels,
            block_frames=max(1, rate * settings.frame_ms // 1000),
        )
        self._blocks.clear()
        stream = self._backend.open_input_stream(request, self._on_block)
        state = _StreamState(
            stream=stream,
            device=device,
            channels=channels,
            resampler=Resampler(stream.sample_rate, settings.sample_rate),
        )
        try:
            stream.start()
        except AudioError:
            stream.close()
            raise
        self._stream = state
        self._seen_blocks = 0
        self._watchdog = -1
        _log.info(
            "capturing from %s at %d Hz / %dch -> %d Hz mono",
            device.label,
            stream.sample_rate,
            channels,
            settings.sample_rate,
        )

    def _negotiate(self, device: AudioDevice, settings: CaptureSettings) -> tuple[int, int]:
        """Pick a device format, preferring the one that needs no conversion.

        16 kHz mono is asked for first.  Two things commonly refuse it: WASAPI
        in shared mode accepts only the rate configured in Windows, and a few
        drivers insist on being opened with all their channels.  Both are
        recoverable - the extra channels are averaged down and the rate is
        resampled - so the candidates are tried in order of how little work they
        leave for the processing thread.
        """
        native = int(device.default_sample_rate) or settings.sample_rate
        channels = max(1, device.channels)
        candidates = [
            (settings.sample_rate, 1),
            (settings.sample_rate, channels),
            (native, 1),
            (native, channels),
        ]
        for rate, count in candidates:
            request = StreamRequest(
                device_index=device.index,
                sample_rate=rate,
                channels=count,
                block_frames=max(1, rate * settings.frame_ms // 1000),
            )
            if self._backend.supports_rate(request):
                if rate != settings.sample_rate or count != 1:
                    _log.info(
                        "%s does not accept %d Hz mono, using %d Hz / %dch with conversion",
                        device.label,
                        settings.sample_rate,
                        rate,
                        count,
                    )
                return rate, count
        # Nothing was accepted: open at the driver's own format and let the
        # backend report the real failure.
        return native, channels

    def _close(self) -> None:
        """Close the open stream, if any.  Lock held."""
        state = self._stream
        self._stream = None
        if state is None:
            return
        state.stream.stop()
        state.stream.close()

    def _on_block(self, pcm: bytes, overflowed: bool) -> None:
        """Receive one block from the device.

        Runs on PortAudio's real-time thread.  Everything in here is O(1) and
        lock-free: ``deque.append`` and ``deque.__len__`` are atomic under the
        GIL, and the counters have a single writer.  No logging, no allocation
        beyond the block itself, no IPC - see the module docstring.
        """
        if overflowed:
            self._overflows += 1
        if len(self._blocks) >= self._max_blocks:
            self._dropped += 1
            return
        self._blocks.append(pcm)
        self._seen_blocks += 1
        self._pending.set()

    def _process_loop(self) -> None:
        """Drain captured blocks and publish levels."""
        interval = max(0.005, self._settings.level_interval_ms / 1000.0)
        while not self._stopping.is_set():
            self._pending.wait(interval)
            self._pending.clear()
            try:
                self._drain()
            except Exception:
                _log.exception("audio processing failed")
            self._publish_level()

    def _drain(self) -> None:
        """Process every queued block."""
        while True:
            try:
                raw = self._blocks.popleft()
            except IndexError:
                return
            self._handle_block(raw)

    def _handle_block(self, raw: bytes) -> None:
        """Normalise one device block and hand it to the consumers."""
        with self._lock:
            stream = self._stream
            settings = self._settings
            muted = self._muted
            recorder = self._recorder
        if stream is None:
            return

        mono = downmix(raw, stream.channels)
        pcm = stream.resampler.process(mono)
        if not pcm:
            return
        if muted:
            pcm = bytes(len(pcm))
            level = AudioLevel()
        else:
            pcm = apply_gain(pcm, settings.gain, limiter=settings.limiter)
            level = pcm_level(pcm)

        self._ring.write(pcm)
        if recorder is not None:
            recorder.write(pcm)
        with self._lock:
            self._accumulator.add(level, len(pcm) // SAMPLE_WIDTH)
        self._callbacks.on_frames(pcm)

    def _publish_level(self) -> None:
        """Emit the aggregated level, at most once per interval."""
        now = time.monotonic()
        with self._lock:
            if self._state is CaptureState.STOPPED:
                return
            interval = self._settings.level_interval_ms / 1000.0
            if now < self._accumulator.deadline:
                return
            self._accumulator.deadline = now + interval
            level = self._accumulator.take()
            if level == self._last_level and level.rms == 0.0 and level.peak == 0.0:
                # Do not spam the bus with silence once the meter has settled.
                return
            self._last_level = level
        self._callbacks.on_level(level)

    def _monitor_loop(self) -> None:
        """Watch the device: notice it leaving, bring it back when it returns."""
        while not self._stopping.wait(self._settings.device_poll_sec):
            try:
                self._check_device()
            except AudioError as exc:
                _log.warning("device check failed: %s", exc.technical)
            except Exception:
                _log.exception("device monitor failed")

    def _check_device(self) -> None:
        """One monitor pass."""
        with self._lock:
            state = self._state
            stream = self._stream

        if state is CaptureState.RUNNING and stream is not None:
            self._check_alive(stream)
        elif state is CaptureState.DEVICE_LOST:
            self._try_recover()

    def _check_alive(self, stream: _StreamState) -> None:
        """Detect a device that stopped delivering audio.

        Unplugging a USB microphone does not always raise: PortAudio may simply
        stop calling back.  Comparing the block counter across two monitor ticks
        catches both that and a stream the driver killed outright.
        """
        seen = self._seen_blocks
        stalled = seen == self._watchdog
        self._watchdog = seen
        if not stalled and stream.stream.active:
            return
        _log.warning("device %s stopped delivering audio", stream.device.label)
        with self._lock:
            self._close()
            self._set_state(
                CaptureState.DEVICE_LOST,
                f"Устройство «{stream.device.name}» отключено.",
            )
        self._notify_devices()

    def _try_recover(self) -> None:
        """Re-scan and re-open once the device is back."""
        previous = self._devices
        self._backend.refresh()
        devices = self.devices()
        change = diff_devices(previous, devices)
        if change:
            self._callbacks.on_devices(change)
        with self._lock:
            if self._state is not CaptureState.DEVICE_LOST:
                return
            try:
                self._open(fallback=not self._settings.device)
            except AudioError as exc:
                _log.debug("device still unavailable: %s", exc.technical)
                return
            device = self._stream.device if self._stream else None
            self._set_state(
                CaptureState.RUNNING,
                f"Устройство «{device.name}» снова доступно." if device else "",
            )

    def _notify_devices(self) -> None:
        """Re-enumerate and report the difference."""
        previous = self._devices
        try:
            devices = self.devices(rescan=True)
        except AudioError as exc:
            _log.debug("cannot enumerate after device loss: %s", exc.technical)
            return
        change = diff_devices(previous, devices)
        if change:
            self._callbacks.on_devices(change)
