"""The thing that actually makes sound: a queue, a thread and one device.

Synthesis produces :class:`~ayris.audio.tts.base.AudioChunk` objects; something
has to hand them to PortAudio in order, at the right rate, and stop instantly
when the user says «Айрис, стоп». That is all this module does. It knows nothing
about engines, models or text - it takes PCM and a request identifier.

**Why a thread and not the event loop.** ``stream.write`` blocks until the driver
has room, by design: that is what paces playback without a timer. Doing it on the
Qt thread would freeze the overlay for the length of every phrase, and doing it
in a PortAudio callback would mean allocating in a real-time context. So the
player owns one worker thread, and every public method just moves an item between
queues - :meth:`TtsPlayer.speak` returns in microseconds.

**Priority is two queues, not a sorted one.** A system message («Не расслышала»)
must not wait behind a three-paragraph answer, but two answers must keep their
order. Two deques with the urgent one drained first give exactly that, and
neither needs a comparison key that would have to break ties by arrival time.

**Cancellation is a flag plus an abort.** Setting the flag stops the writer
between blocks; :meth:`OutputStream.stop` maps to PortAudio's ``abort``, which
throws away what the driver has already buffered. Both are needed: without the
flag the writer would push the rest of the phrase, without the abort the user
would hear up to a bufferful after the assistant was told to be quiet. Phrases
are written in small blocks so the check happens often enough to feel immediate.

**Rate changes reopen the stream.** Piper is 22.05 kHz, Silero 48 kHz, XTTS
24 kHz, and a device that took one may refuse another. The player opens at the
first chunk's rate and resamples anything that disagrees, reopening only when the
device itself will not take it - a switch costs tens of milliseconds and a
resample costs a fraction of one.

**A vanished device is not a crash.** Unplugging headphones mid-sentence raises
from ``write``; the player closes the stream, reopens on the current default and
carries on with the next phrase. Losing a sentence is bad. Taking the assistant
down with it is worse.

Everything here talks to PortAudio through :class:`PlaybackBackend` from
:mod:`ayris.audio.devices`, so the tests drive a fake device and never open one.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayris.audio.capture import Resampler, apply_gain
from ayris.audio.devices import (
    DeviceDirection,
    PlaybackRequest,
    resolve_device,
)
from ayris.audio.tts.base import SAMPLE_WIDTH, AudioChunk
from ayris.core.errors import AudioError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from ayris.audio.devices import OutputStream, PlaybackBackend

__all__ = [
    "BLOCK_MS",
    "PlaybackReason",
    "PlayerStats",
    "SpeechRequest",
    "TtsPlayer",
]

_log = get_logger(__name__)

#: Audio handed to the driver in one write. 20 ms is the granularity of a stop:
#: shorter would mean more syscalls for a latency nobody can hear, longer would
#: leave an audible tail after «стоп».
BLOCK_MS: Final = 20

#: How long :meth:`TtsPlayer.stop` waits for the writer thread to notice.
_JOIN_TIMEOUT_S: Final = 2.0

#: Idle poll interval of the writer thread. It normally blocks on the queue
#: condition and wakes immediately on a new phrase; this only bounds the delay
#: when a spurious wakeup races a shutdown.
_IDLE_WAIT_S: Final = 0.2

#: Device rates tried, in order, when the native rate of the audio is refused.
#: 48 kHz first because every modern device takes it.
_FALLBACK_RATES: Final = (48000, 44100, 24000, 22050, 16000)


class PlaybackReason:
    """Why a phrase stopped, mirroring :attr:`TtsFinished.reason`.

    A plain namespace rather than an enum: the same strings cross the worker pipe
    and end up in :class:`ayris.core.events.TtsFinished`, and converting back and
    forth would only add a place to get it wrong.
    """

    COMPLETED: Final = "completed"
    CANCELLED: Final = "cancelled"
    ERROR: Final = "error"


@dataclass(slots=True)
class SpeechRequest:
    """One thing to say, and everything needed to say it.

    Attributes:
        text: What is being spoken, carried for the events and the log only -
            the player never synthesizes.
        chunks: Audio, or an iterator producing it sentence by sentence. An
            iterator is the streaming path: the player plays chunk one while the
            engine is still working on chunk two, which is what keeps the first
            sound under the deadline.
        request_id: Correlates this phrase with the transcript and the reply that
            produced it.
        engine: Name of the engine, for :class:`~ayris.core.events.TtsStarted`.
        priority: Urgent phrases jump the queue but never interrupt what is
            already sounding - cutting a sentence in half to say «Готово» would
            read as a glitch.
        duration_estimate_ms: Known length when the audio is already complete,
            ``0`` while it is still streaming.
    """

    text: str = ""
    chunks: Iterable[AudioChunk] = ()
    request_id: str = ""
    engine: str = ""
    priority: bool = False
    duration_estimate_ms: int = 0


@dataclass(frozen=True, slots=True)
class PlayerStats:
    """Snapshot for DevTools and the tests.

    Attributes:
        queued: Phrases waiting, both queues together.
        speaking: Whether something is sounding right now.
        played: Phrases finished normally since start.
        cancelled: Phrases stopped by :meth:`TtsPlayer.cancel`.
        failed: Phrases dropped because the device or the engine failed.
        device_id: Identifier of the open device, empty when closed.
        sample_rate: Rate the device is open at, ``0`` when closed.
        underruns: Times the stream had to be reopened after a device error.
    """

    queued: int = 0
    speaking: bool = False
    played: int = 0
    cancelled: int = 0
    failed: int = 0
    device_id: str = ""
    sample_rate: int = 0
    underruns: int = 0


@dataclass(slots=True)
class _Stream:
    """The open device, with the conversion the current audio needs."""

    stream: OutputStream
    device_id: str
    sample_rate: int
    channels: int
    resampler: Resampler | None = None
    source_rate: int = 0

    def converter_for(self, chunk: AudioChunk) -> Resampler | None:
        """Resampler taking ``chunk`` to the device rate, or ``None``.

        Cached across chunks of one phrase: :class:`Resampler` carries state
        between blocks, and rebuilding it per sentence would put a click at every
        boundary.
        """
        if chunk.sample_rate == self.sample_rate:
            return None
        if self.resampler is None or self.source_rate != chunk.sample_rate:
            self.resampler = Resampler(chunk.sample_rate, self.sample_rate)
            self.source_rate = chunk.sample_rate
        return self.resampler


def _ignore_started(_request: SpeechRequest, _duration_ms: int) -> None:
    """Default :attr:`TtsPlayer` start callback."""


def _ignore_finished(_request: SpeechRequest, _reason: str) -> None:
    """Default :attr:`TtsPlayer` finish callback."""


class TtsPlayer:
    """Plays synthesized speech in order, and stops on demand.

    Args:
        backend: Where playback streams come from. Defaults to PortAudio through
            :class:`~ayris.audio.devices.SoundDeviceBackend`; the tests pass a
            fake, which is the only reason this module has no import of
            ``sounddevice`` in it.
        device: Identifier or name from the settings. Empty means the system
            default.
        volume: 0.0-1.0, applied to the samples. Not the system mixer: changing
            that would follow the user into their next application.
        on_started: Called on the writer thread when a phrase starts sounding,
            with the estimated duration in milliseconds. The caller publishes
            :class:`~ayris.core.events.TtsStarted` from it.
        on_finished: Called when a phrase stops, with a
            :class:`PlaybackReason` value.

    The callbacks run on the writer thread, so they must not touch Qt widgets
    directly - publishing on the event bus, which is thread-safe by design, is
    what they are for.
    """

    __slots__ = (
        "_backend",
        "_cancel",
        "_condition",
        "_current",
        "_device_spec",
        "_normal",
        "_on_finished",
        "_on_started",
        "_running",
        "_stats",
        "_stream",
        "_thread",
        "_urgent",
        "_volume",
    )

    def __init__(
        self,
        backend: PlaybackBackend | None = None,
        *,
        device: str = "",
        volume: float = 0.8,
        on_started: Callable[[SpeechRequest, int], None] = _ignore_started,
        on_finished: Callable[[SpeechRequest, str], None] = _ignore_finished,
    ) -> None:
        self._backend = backend if backend is not None else _default_backend()
        self._device_spec = device
        self._volume = _clamp_volume(volume)
        self._on_started = on_started
        self._on_finished = on_finished
        self._condition = threading.Condition()
        self._normal: deque[SpeechRequest] = deque()
        self._urgent: deque[SpeechRequest] = deque()
        self._current: SpeechRequest | None = None
        self._cancel = threading.Event()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stream: _Stream | None = None
        self._stats = _Counters()

    # ------------------------------------------------------------- properties

    @property
    def volume(self) -> float:
        """Current gain, 0.0-1.0."""
        return self._volume

    @property
    def speaking(self) -> bool:
        """Whether a phrase is sounding right now."""
        with self._condition:
            return self._current is not None

    @property
    def queued(self) -> int:
        """Phrases waiting to be spoken."""
        with self._condition:
            return len(self._normal) + len(self._urgent)

    @property
    def running(self) -> bool:
        """Whether the writer thread is alive."""
        return self._running

    # ---------------------------------------------------------------- control

    def start(self) -> None:
        """Start the writer thread. Idempotent.

        The device is not opened here: opening it at the first phrase means an
        assistant that is not speaking holds nothing, which matters on machines
        where an exclusive-mode application would otherwise be locked out.
        """
        with self._condition:
            if self._running:
                return
            self._running = True
            self._cancel.clear()
            self._thread = threading.Thread(target=self._run, name="ayris-tts-player", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop speaking, drop the queue and release the device.

        Safe to call from any thread and safe to call twice. Blocks briefly for
        the writer to unwind - up to :data:`_JOIN_TIMEOUT_S`, after which the
        thread is left to die on its own rather than hanging shutdown.
        """
        with self._condition:
            if not self._running:
                return
            self._running = False
            self._clear_locked()
            self._condition.notify_all()
        self._cancel.set()
        self._abort_stream()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=_JOIN_TIMEOUT_S)
            if thread.is_alive():
                _log.warning("tts player thread did not stop in %.1f s", _JOIN_TIMEOUT_S)
        self._thread = None
        self._close_stream()

    def speak(self, request: SpeechRequest) -> None:
        """Queue a phrase.

        Returns immediately: the audio is written on the player's own thread.
        Starts the player if it is not running, so a caller that only ever says
        one thing does not have to remember to.
        """
        if not self._running:
            self.start()
        with self._condition:
            queue = self._urgent if request.priority else self._normal
            queue.append(request)
            self._condition.notify_all()

    def cancel(self, request_id: str = "") -> bool:
        """Stop speaking now and forget what was queued.

        What :class:`~ayris.core.events.CancelRequested` maps to. The current
        phrase is aborted mid-word - PortAudio's ``abort`` discards its buffer,
        so nothing plays out after this returns.

        Args:
            request_id: Cancel only if this is the phrase being spoken. Empty
                cancels whatever is sounding. Used by a caller that wants to
                interrupt *its own* answer without silencing a later one that
                has already replaced it.

        Returns:
            Whether anything was actually stopped or dropped.
        """
        with self._condition:
            current = self._current
            if request_id and (current is None or current.request_id != request_id):
                dropped = self._drop_locked(request_id)
                return dropped > 0
            pending = len(self._normal) + len(self._urgent)
            self._clear_locked()
            speaking = current is not None
        if speaking:
            self._cancel.set()
            self._abort_stream()
        return speaking or pending > 0

    def set_volume(self, volume: float) -> None:
        """Change the gain. Takes effect from the next block, not the next phrase."""
        self._volume = _clamp_volume(volume)

    def set_device(self, device: str) -> None:
        """Point at another output device.

        The current phrase finishes on the old one: the settings window changing
        a device is not a reason to cut a sentence in half, and the next phrase
        is at most a second away.
        """
        if device == self._device_spec:
            return
        self._device_spec = device
        with self._condition:
            stream = self._stream
        if stream is not None:
            _log.info("tts: устройство вывода изменено на %r", device or "по умолчанию")

    def set_observers(
        self,
        *,
        on_started: Callable[[SpeechRequest, int], None] | None = None,
        on_finished: Callable[[SpeechRequest, str], None] | None = None,
    ) -> None:
        """Replace the start/finish callbacks after construction.

        Exists because the bus bridge is built around the player rather than
        before it - see :mod:`ayris.audio.tts.service`. Passing ``None`` leaves
        that callback alone; there is no way to unset one, because a player whose
        events go nowhere is a bug rather than a configuration.
        """
        if on_started is not None:
            self._on_started = on_started
        if on_finished is not None:
            self._on_finished = on_finished

    def stats(self) -> PlayerStats:
        """Snapshot for DevTools and the tests."""
        with self._condition:
            stream = self._stream
            return PlayerStats(
                queued=len(self._normal) + len(self._urgent),
                speaking=self._current is not None,
                played=self._stats.played,
                cancelled=self._stats.cancelled,
                failed=self._stats.failed,
                device_id=stream.device_id if stream is not None else "",
                sample_rate=stream.sample_rate if stream is not None else 0,
                underruns=self._stats.underruns,
            )

    # ----------------------------------------------------------- writer thread

    def _run(self) -> None:
        """Take phrases off the queues and write them, until stopped."""
        while True:
            request = self._next_request()
            if request is None:
                break
            reason = self._play(request)
            self._finish(request, reason)
        self._close_stream()

    def _next_request(self) -> SpeechRequest | None:
        """Block until there is something to say, or the player stops.

        Returns:
            The next phrase, or ``None`` when :meth:`stop` was called. Urgent
            first, then in arrival order.
        """
        with self._condition:
            while self._running and not self._urgent and not self._normal:
                self._condition.wait(_IDLE_WAIT_S)
            if not self._running:
                self._current = None
                return None
            queue = self._urgent if self._urgent else self._normal
            request = queue.popleft()
            self._current = request
            self._cancel.clear()
            return request

    def _play(self, request: SpeechRequest) -> str:
        """Write one phrase to the device.

        The iterator is consumed lazily, so a streaming request has its next
        sentence synthesized while this one sounds. Failure of the *source* -
        the engine raising on sentence three - ends the phrase as an error, with
        the sentences already written left audible.

        Returns:
            A :class:`PlaybackReason` value.
        """
        started = False
        try:
            for chunk in request.chunks:
                if self._cancel.is_set():
                    return PlaybackReason.CANCELLED
                if chunk.empty:
                    continue
                if not started:
                    started = True
                    self._announce(request, chunk)
                if not self._write_chunk(chunk):
                    return PlaybackReason.CANCELLED
        except AudioError as exc:
            _log.warning("tts: воспроизведение прервано: %s", exc)
            return PlaybackReason.ERROR
        except Exception:
            _log.exception("tts: источник аудио отказал")
            return PlaybackReason.ERROR
        if self._cancel.is_set():
            return PlaybackReason.CANCELLED
        if not started:
            # Nothing speakable: an empty answer or a text of only punctuation.
            self._announce(request, None)
        return PlaybackReason.COMPLETED

    def _announce(self, request: SpeechRequest, first: AudioChunk | None) -> None:
        """Report that sound is starting, once per phrase."""
        duration = request.duration_estimate_ms
        if duration <= 0 and first is not None:
            duration = int(first.duration_ms)
        try:
            self._on_started(request, max(0, duration))
        except Exception:
            # A failing observer must not silence the assistant.
            _log.exception("tts: обработчик начала озвучки упал")

    def _finish(self, request: SpeechRequest, reason: str) -> None:
        """Clear the current phrase and report how it ended."""
        with self._condition:
            self._current = None
            if reason == PlaybackReason.COMPLETED:
                self._stats.played += 1
            elif reason == PlaybackReason.CANCELLED:
                self._stats.cancelled += 1
            else:
                self._stats.failed += 1
        self._cancel.clear()
        try:
            self._on_finished(request, reason)
        except Exception:
            _log.exception("tts: обработчик конца озвучки упал")

    def _write_chunk(self, chunk: AudioChunk) -> bool:
        """Write one sentence, in blocks, checking for cancel between them.

        Returns:
            ``False`` when the phrase was cancelled part-way. A device that
            vanished is retried once on the current default; if that fails too
            the :class:`~ayris.core.errors.AudioError` propagates and the phrase
            ends as an error.

        Raises:
            AudioError: The device could not be opened at all.
        """
        stream = self._ensure_stream(chunk)
        converter = stream.converter_for(chunk)
        pcm = converter.process(chunk.pcm) if converter is not None else chunk.pcm
        block = _block_bytes(stream.sample_rate, stream.channels)
        for offset in range(0, len(pcm), block):
            if self._cancel.is_set():
                return False
            piece = pcm[offset : offset + block]
            try:
                stream.stream.write(_with_volume(piece, self._volume))
            except AudioError as exc:
                if self._cancel.is_set():
                    # Writing into a stream that cancel() just aborted. Expected.
                    return False
                _log.warning("tts: устройство вывода потеряно: %s", exc)
                self._stats.underruns += 1
                self._close_stream()
                # One retry: an unplugged headset usually means the audio should
                # continue on the built-in speakers, and the user would rather
                # hear the rest of the answer there than lose it.
                stream = self._ensure_stream(chunk)
                stream.stream.write(_with_volume(piece, self._volume))
        return True

    # ------------------------------------------------------------------ device

    def _ensure_stream(self, chunk: AudioChunk) -> _Stream:
        """Return an open stream that can take ``chunk``, opening one if needed.

        Raises:
            AudioError: No device could be opened.
        """
        with self._condition:
            stream = self._stream
        if stream is not None and stream.channels == chunk.channels and stream.stream.active:
            return stream
        if stream is not None:
            self._close_stream()
        opened = self._open_stream(chunk)
        with self._condition:
            self._stream = opened
        return opened

    def _open_stream(self, chunk: AudioChunk) -> _Stream:
        """Open the device, at the chunk's rate if it will take it.

        Falls back through :data:`_FALLBACK_RATES` and resamples, because a
        device refusing 22.05 kHz is common enough on Windows - shared-mode
        WASAPI usually wants 48 kHz - that failing there would make Piper
        unusable on ordinary hardware.

        Raises:
            AudioError: Every rate was refused, or the device is gone.
        """
        device = resolve_device(
            self._backend,
            self._device_spec,
            direction=DeviceDirection.OUTPUT,
            fallback=True,
        )
        channels = max(1, min(chunk.channels, max(1, device.channels)))
        errors: list[str] = []
        for rate in _rate_candidates(chunk.sample_rate):
            request = PlaybackRequest(
                device_index=device.index,
                sample_rate=rate,
                channels=channels,
                block_frames=max(1, rate * BLOCK_MS // 1000),
            )
            try:
                stream = self._backend.open_output_stream(request)
                stream.start()
            except AudioError as exc:
                errors.append(f"{rate} Hz: {exc}")
                continue
            _log.info(
                "tts: вывод открыт на %s, %d Гц, %d кан.",
                device.label,
                stream.sample_rate,
                stream.channels,
            )
            return _Stream(
                stream=stream,
                device_id=device.id,
                sample_rate=stream.sample_rate,
                channels=stream.channels,
            )
        raise AudioError(
            f"cannot open output device {device.id!r}: {'; '.join(errors)}",
            user_message=(
                f"Не удалось открыть устройство вывода «{device.label}». "
                "Выберите другое в настройках."
            ),
        )

    def _abort_stream(self) -> None:
        """Throw away what the driver still holds, without closing the device."""
        with self._condition:
            stream = self._stream
        if stream is not None:
            stream.stream.stop()

    def _close_stream(self) -> None:
        """Release the device if one is open."""
        with self._condition:
            stream = self._stream
            self._stream = None
        if stream is not None:
            stream.stream.stop()
            stream.stream.close()

    # ------------------------------------------------------------------ queues

    def _clear_locked(self) -> None:
        """Drop every pending phrase. Caller holds the condition."""
        for request in (*self._urgent, *self._normal):
            self._stats.cancelled += 1
            self._report_dropped(request)
        self._urgent.clear()
        self._normal.clear()

    def _drop_locked(self, request_id: str) -> int:
        """Remove queued phrases with this identifier. Caller holds the condition."""
        removed = 0
        for queue in (self._urgent, self._normal):
            kept = [item for item in queue if item.request_id != request_id]
            removed += len(queue) - len(kept)
            for item in queue:
                if item.request_id == request_id:
                    self._report_dropped(item)
            queue.clear()
            queue.extend(kept)
        self._stats.cancelled += removed
        return removed

    def _report_dropped(self, request: SpeechRequest) -> None:
        """Tell the observer a queued phrase will never be spoken.

        Called with the condition held, so the callback must not call back into
        the player. It publishes an event, which is exactly that.
        """
        try:
            self._on_finished(request, PlaybackReason.CANCELLED)
        except Exception:
            _log.exception("tts: обработчик конца озвучки упал")


@dataclass(slots=True)
class _Counters:
    """Mutable counters behind :class:`PlayerStats`."""

    played: int = 0
    cancelled: int = 0
    failed: int = 0
    underruns: int = 0


# ------------------------------------------------------------------- helpers


def chunks_from(audio: AudioChunk) -> Iterator[AudioChunk]:
    """Wrap a finished chunk as a one-item stream.

    Sugar for the common case: a cached phrase is already whole, and
    :attr:`SpeechRequest.chunks` wants something iterable either way.
    """
    yield audio


def _rate_candidates(native: int) -> tuple[int, ...]:
    """Rates to try, native first and then the usual device rates."""
    seen = [native] if native > 0 else []
    seen.extend(rate for rate in _FALLBACK_RATES if rate != native)
    return tuple(seen)


def _block_bytes(sample_rate: int, channels: int) -> int:
    """Bytes in one :data:`BLOCK_MS` write."""
    frames = max(1, sample_rate * BLOCK_MS // 1000)
    return frames * channels * SAMPLE_WIDTH


def _with_volume(pcm: bytes, volume: float) -> bytes:
    """Scale samples, skipping the pass when the gain is 1.0.

    Reuses :func:`ayris.audio.capture.apply_gain`, which already clips and has
    tests, rather than a second scaling loop in this file. The limiter is off:
    speech from an engine is never near full scale, and a limiter's knee would
    audibly compress a loud sentence.
    """
    if volume >= 0.999:
        return pcm
    if volume <= 0.001:
        return bytes(len(pcm))
    return apply_gain(pcm, volume, limiter=False)


def _clamp_volume(volume: float) -> float:
    """Hold the gain in 0.0-1.0."""
    return min(max(volume, 0.0), 1.0)


def _default_backend() -> PlaybackBackend:
    """PortAudio, imported here so the tests can avoid it entirely."""
    from ayris.audio.devices import SoundDeviceBackend

    return SoundDeviceBackend()
