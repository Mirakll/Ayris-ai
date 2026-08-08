"""Fixed-capacity ring of mono PCM, sized in seconds of audio.

The capture pipeline always has more recent audio than any consumer has asked
for.  Wake word detection is the reason: by the time the detector reports a
trigger the user has already started the phrase, so the recognizer needs the
few hundred milliseconds that came *before* the trigger.  Keeping a rolling
window of the last N seconds makes that pre-roll a plain read instead of a
speculative recording.

The buffer stores raw little-endian ``int16`` frames and knows nothing about
who writes them; :mod:`ayris.audio.capture` writes already normalised audio
(mono, target sample rate, gain applied) so that every reader sees the same
stream.

Two reading styles are supported:

``read_last``
    "give me the last 400 ms" - used for pre-roll, WAV dumps and level meters.
``read_from``
    "give me everything since position P" - used by consumers that follow the
    stream continuously (VAD, streaming recognition).  Positions are absolute
    frame counters that never wrap, so a consumer that falls behind learns how
    many frames it missed instead of silently reading garbage.

All public methods are safe to call from any thread.  Writes hold the lock only
for a ``memoryview`` copy, so a reader can never stall the capture pipeline for
a meaningful amount of time.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Final

#: Bytes per sample frame.  The pipeline is mono ``int16`` end to end.
SAMPLE_WIDTH: Final = 2

#: Default window kept in memory.  30 s of 16 kHz mono costs ~960 KiB.
DEFAULT_CAPACITY_SEC: Final = 30.0

#: Audio kept before a wake word trigger.  Speech onset consistently precedes
#: the detector by 200-400 ms; 400 ms leaves headroom without dragging in the
#: previous sentence.
DEFAULT_PRE_ROLL_MS: Final = 400

#: Smallest sane capacity - one 10 ms frame would make every read empty.
_MIN_CAPACITY_MS: Final = 100


@dataclass(frozen=True, slots=True)
class BufferRead:
    """Result of :meth:`RingBuffer.read_from`.

    Attributes:
        pcm: The frames that were available, oldest first.
        position: Absolute frame index to pass to the next ``read_from`` call.
        dropped: Frames that were overwritten before this reader got to them.
            Anything above zero means the consumer is too slow for the buffer
            size and audio was genuinely lost.
    """

    pcm: bytes
    position: int
    dropped: int = 0

    @property
    def frames(self) -> int:
        """Number of sample frames in :attr:`pcm`."""
        return len(self.pcm) // SAMPLE_WIDTH

    def __bool__(self) -> bool:
        """True when the read returned audio."""
        return bool(self.pcm)


class RingBuffer:
    """Thread-safe ring of ``int16`` mono PCM with absolute read positions.

    Args:
        seconds: Window to keep.  Rounded up to whole frames and clamped to at
            least 100 ms.
        sample_rate: Sample rate of the frames that will be written.  Only used
            to convert between milliseconds and frames.

    Raises:
        ValueError: If ``sample_rate`` or ``seconds`` is not positive.
    """

    __slots__ = (
        "_available",
        "_capacity",
        "_data",
        "_lock",
        "_overwritten",
        "_sample_rate",
        "_total",
        "_write",
    )

    def __init__(
        self,
        *,
        seconds: float = DEFAULT_CAPACITY_SEC,
        sample_rate: int = 16000,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if seconds <= 0:
            raise ValueError(f"seconds must be positive, got {seconds}")
        floor = max(1, sample_rate * _MIN_CAPACITY_MS // 1000)
        frames = max(floor, int(round(seconds * sample_rate)))
        self._sample_rate = sample_rate
        self._capacity = frames * SAMPLE_WIDTH
        self._data = bytearray(self._capacity)
        self._write = 0
        self._available = 0
        self._total = 0
        self._overwritten = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ info

    @property
    def sample_rate(self) -> int:
        """Sample rate the buffer converts durations with."""
        return self._sample_rate

    @property
    def capacity_frames(self) -> int:
        """Window size in sample frames."""
        return self._capacity // SAMPLE_WIDTH

    @property
    def capacity_ms(self) -> int:
        """Window size in milliseconds."""
        return self.capacity_frames * 1000 // self._sample_rate

    @property
    def available_frames(self) -> int:
        """Frames currently readable."""
        with self._lock:
            return self._available

    @property
    def available_ms(self) -> int:
        """Readable audio in milliseconds."""
        return self.available_frames * 1000 // self._sample_rate

    @property
    def position(self) -> int:
        """Absolute index just past the newest frame.

        A consumer that wants only fresh audio records this value once and
        passes it to :meth:`read_from` from then on.
        """
        with self._lock:
            return self._total

    @property
    def overwritten_frames(self) -> int:
        """Frames evicted by newer audio over the buffer's lifetime."""
        with self._lock:
            return self._overwritten

    def frames_for_ms(self, ms: float) -> int:
        """Convert a duration to a whole number of frames."""
        return max(0, int(round(ms * self._sample_rate / 1000.0)))

    # ----------------------------------------------------------------- write

    def write(self, pcm: bytes | bytearray | memoryview) -> int:
        """Append frames, evicting the oldest audio when the window is full.

        Args:
            pcm: Little-endian ``int16`` mono samples.

        Returns:
            Number of frames accepted.  A block larger than the whole window is
            truncated to its tail, and the truncated part counts as
            overwritten.

        Raises:
            ValueError: If the payload does not contain whole sample frames.
        """
        view = memoryview(pcm).cast("B")
        size = view.nbytes
        if size == 0:
            return 0
        if size % SAMPLE_WIDTH:
            raise ValueError(f"payload of {size} bytes is not whole int16 frames")
        frames = size // SAMPLE_WIDTH

        with self._lock:
            if size >= self._capacity:
                # The block alone outlives the window: keep only its tail.
                self._data[:] = view[size - self._capacity :]
                self._write = 0
                evicted = self._available + frames - self.capacity_frames
                self._available = self.capacity_frames
            else:
                end = self._write + size
                if end <= self._capacity:
                    self._data[self._write : end] = view
                else:
                    head = self._capacity - self._write
                    self._data[self._write :] = view[:head]
                    self._data[: size - head] = view[head:]
                self._write = end % self._capacity
                evicted = max(0, self._available + frames - self.capacity_frames)
                self._available = min(self.capacity_frames, self._available + frames)
            self._total += frames
            self._overwritten += evicted
        return frames

    def clear(self) -> None:
        """Drop the stored audio while keeping positions monotonic.

        Called when the stream restarts on another device: the old samples are
        unrelated to the new ones, but consumers must not see their position
        jump backwards.
        """
        with self._lock:
            self._write = 0
            self._overwritten += self._available
            self._available = 0

    # ------------------------------------------------------------------ read

    def read_last(self, *, frames: int | None = None, ms: float | None = None) -> bytes:
        """Return the newest audio, oldest frame first.

        Args:
            frames: How many frames to return.  Takes precedence over ``ms``.
            ms: Duration to return.  Defaults to the whole window when neither
                argument is given.

        Returns:
            Up to the requested amount - fewer bytes when the buffer has not
            filled yet.
        """
        if frames is None:
            wanted = self.capacity_frames if ms is None else self.frames_for_ms(ms)
        else:
            wanted = frames
        if wanted <= 0:
            return b""
        with self._lock:
            count = min(wanted, self._available)
            if count == 0:
                return b""
            start = (self._write - count * SAMPLE_WIDTH) % self._capacity
            return self._slice(start, count * SAMPLE_WIDTH)

    def read_from(self, position: int, *, max_frames: int = 0) -> BufferRead:
        """Return everything written since ``position``.

        Args:
            position: Absolute frame index, normally :attr:`BufferRead.position`
                from the previous call or :attr:`position` when starting.
            max_frames: Optional cap, so a consumer can process the backlog in
                bounded chunks.

        Returns:
            The available frames plus the position to continue from.  When the
            consumer fell behind, ``dropped`` reports the lost frames and the
            read resumes at the oldest frame still held.
        """
        with self._lock:
            oldest = self._total - self._available
            dropped = 0
            if position < oldest:
                dropped = oldest - position
                position = oldest
            elif position > self._total:
                # A reader from a previous stream, or a bug: never read ahead.
                position = self._total
            count = self._total - position
            if max_frames > 0:
                count = min(count, max_frames)
            if count == 0:
                return BufferRead(b"", position, dropped)
            offset = (self._total - position) * SAMPLE_WIDTH
            start = (self._write - offset) % self._capacity
            pcm = self._slice(start, count * SAMPLE_WIDTH)
            return BufferRead(pcm, position + count, dropped)

    def snapshot(self) -> bytes:
        """Return the whole window, oldest frame first."""
        return self.read_last(frames=self.capacity_frames)

    # -------------------------------------------------------------- internal

    def _slice(self, start: int, length: int) -> bytes:
        """Copy ``length`` bytes from ``start``, following the wrap.

        The caller must hold the lock.
        """
        end = start + length
        if end <= self._capacity:
            return bytes(self._data[start:end])
        head = self._capacity - start
        return bytes(self._data[start:]) + bytes(self._data[: length - head])
