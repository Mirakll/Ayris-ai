"""Task 12: the queue, the order, the volume and the stop.

The runner has no output device, and the developer machine that does must not
start making noise during a test run.  So every test here drives a
:class:`FakeBackend` implementing :class:`~ayris.audio.devices.PlaybackBackend`,
which records the bytes it was given instead of handing them to PortAudio.  That
is not a compromise: what task 12 asks about playback - phrases in order, system
messages first, silence within a block of «Айрис, стоп», a vanished device that
does not take the assistant down - is all observable in what was *written*, and
none of it is observable through a speaker.

The player runs a writer thread, so nothing here sleeps in the hope of winning a
race.  :attr:`FakeBackend.start_blocked` parks the writer inside its first
``write``, which is what makes "the player is speaking right now" an assertable
state rather than a guess, and :func:`wait_until` polls with a deadline and fails
readably instead of hanging the suite.

The one thing a fake cannot show is that PortAudio's ``abort`` really discards
the buffer the driver already holds.  That is marked ``hardware`` and left to the
user's Windows.

Groups:

* :class:`TestPlayerLifecycle` — start, stop, idempotence, no device until needed.
* :class:`TestPlaybackOrder` — FIFO, and system messages jumping the queue.
* :class:`TestStreaming` — chunks pulled lazily, so sentence two is synthesized
  while sentence one sounds.
* :class:`TestCancellation` — what «Айрис, стоп» silences, and how quickly.
* :class:`TestEvents` — what ``TtsStarted``/``TtsFinished`` are built from.
* :class:`TestVolume` — the gain, applied without touching the system mixer.
* :class:`TestDeviceHandling` — rate negotiation, resampling, hot-plug.
* :class:`TestStats` — what DevTools shows.
"""

from __future__ import annotations

import threading
from array import array
from time import monotonic, sleep
from typing import TYPE_CHECKING, Final

import pytest

from ayris.audio.devices import DeviceDirection, PlaybackRequest, RawDevice
from ayris.audio.tts.base import SAMPLE_WIDTH, AudioChunk
from ayris.audio.tts.player import (
    BLOCK_MS,
    PlaybackReason,
    SpeechRequest,
    TtsPlayer,
    chunks_from,
)
from ayris.core.errors import AudioError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pytestmark = pytest.mark.unit

#: Rate the fake device prefers, matching what Windows shared-mode WASAPI does.
DEVICE_RATE: Final = 48000

#: Long enough to span several :data:`BLOCK_MS` writes, so a cancel has somewhere
#: to land in the middle of a phrase.
PHRASE_MS: Final = 400

#: Ceiling for :func:`wait_until`. Generous on purpose: a loaded CI runner is
#: slow, and a test that reaches this has genuinely hung rather than been unlucky.
TIMEOUT_S: Final = 5.0


# ----------------------------------------------------------------------
# the fake device
# ----------------------------------------------------------------------


def tone(ms: int, sample_rate: int = DEVICE_RATE, *, level: int = 8000) -> bytes:
    """``ms`` milliseconds of a constant int16 level.

    Constant rather than a sine because these tests ask about *amplitude* - what
    the volume did, whether a block was silence - and a flat level makes the
    expected value a single number instead of an envelope.
    """
    frames = max(0, int(sample_rate * ms / 1000))
    return array("h", [level] * frames).tobytes()


def peak(pcm: bytes) -> int:
    """Largest absolute sample in ``pcm``; ``0`` for silence or for nothing."""
    if not pcm:
        return 0
    samples = array("h")
    samples.frombytes(pcm)
    return max(abs(value) for value in samples)


class FakeStream:
    """An output stream that appends to a list.

    :attr:`block` holds the writer thread inside a ``write``, which is the only
    way to observe the player mid-phrase: without it a 400 ms phrase is written
    to a list in microseconds and every "while it is speaking" test becomes a
    race against a thread that has already finished.
    """

    def __init__(self, request: PlaybackRequest) -> None:
        self._request = request
        self._active = False
        self.writes: list[bytes] = []
        self.stops = 0
        self.closes = 0
        #: Writes before this one raise nothing; this one raises. ``-1`` never.
        self.fail_after = -1
        self.block = threading.Event()
        self.block.set()
        self.wrote_once = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self._request.sample_rate

    @property
    def channels(self) -> int:
        return self._request.channels

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True

    def write(self, pcm: bytes) -> None:
        if 0 <= self.fail_after <= len(self.writes):
            raise AudioError("устройство исчезло", user_message="Устройство пропало.")
        self.writes.append(bytes(pcm))
        self.wrote_once.set()
        # Parked after recording, so a waiting test can see that the phrase
        # started before it decides what to do next.
        self.block.wait(TIMEOUT_S)

    def stop(self) -> None:
        self.stops += 1
        self._active = False
        # An abort releases a blocked writer, exactly as PortAudio's does.
        self.block.set()

    def close(self) -> None:
        self.closes += 1
        self._active = False
        self.block.set()

    @property
    def written(self) -> bytes:
        """Everything written to this stream, joined."""
        return b"".join(self.writes)


class FakeBackend:
    """A :class:`~ayris.audio.devices.PlaybackBackend` with no PortAudio in it."""

    def __init__(self, *, rates: tuple[int, ...] = (), channels: int = 2) -> None:
        #: Rates the device accepts. Empty means "anything", which is the kind
        #: case; a tuple reproduces a WASAPI device that only takes 48 kHz.
        self.rates = rates
        self.channels = channels
        self.streams: list[FakeStream] = []
        self.refused: list[int] = []
        self.fail_open = False
        #: New streams start with :attr:`FakeStream.block` cleared, parking the
        #: writer at its first write.
        self.start_blocked = False
        #: Makes only the *first* stream fail on write. The retry has to succeed
        #: or the test would be about giving up rather than about recovering.
        self.fail_first_write = False

    def raw_devices(self) -> tuple[RawDevice, ...]:
        return (
            RawDevice(
                index=0,
                name="Динамики",
                host_api="WASAPI",
                max_output_channels=self.channels,
                default_sample_rate=float(DEVICE_RATE),
                default_output=True,
            ),
            RawDevice(
                index=1,
                name="Наушники",
                host_api="WASAPI",
                max_output_channels=2,
                default_sample_rate=float(DEVICE_RATE),
            ),
        )

    def open_output_stream(self, request: PlaybackRequest) -> FakeStream:
        if self.fail_open:
            raise AudioError("нет устройства", user_message="Нет устройства вывода.")
        if self.rates and request.sample_rate not in self.rates:
            self.refused.append(request.sample_rate)
            raise AudioError(f"{request.sample_rate} Гц не поддерживается")
        stream = FakeStream(request)
        if self.start_blocked:
            stream.block.clear()
        if self.fail_first_write and not self.streams:
            stream.fail_after = 0
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeStream:
        """The most recently opened stream, asserted to exist."""
        assert self.streams, "плеер не открыл устройство"
        return self.streams[-1]

    @property
    def written(self) -> bytes:
        """Everything written to every stream this backend handed out."""
        return b"".join(stream.written for stream in self.streams)

    def release(self) -> None:
        """Let every parked writer go, so a test can reach its assertions."""
        self.start_blocked = False
        for stream in self.streams:
            stream.block.set()


class Observer:
    """Records the player's callbacks, which is what becomes bus events."""

    def __init__(self) -> None:
        self.started: list[tuple[str, int]] = []
        self.finished: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def on_started(self, request: SpeechRequest, duration_ms: int) -> None:
        with self._lock:
            self.started.append((request.request_id, duration_ms))

    def on_finished(self, request: SpeechRequest, reason: str) -> None:
        with self._lock:
            self.finished.append((request.request_id, reason))

    def reasons(self, request_id: str) -> list[str]:
        """Every reason reported for one phrase."""
        with self._lock:
            return [reason for name, reason in self.finished if name == request_id]

    @property
    def started_ids(self) -> list[str]:
        with self._lock:
            return [name for name, _duration in self.started]

    @property
    def finished_ids(self) -> list[str]:
        with self._lock:
            return [name for name, _reason in self.finished]


def wait_until(predicate: Callable[[], bool], message: str, timeout: float = TIMEOUT_S) -> None:
    """Poll ``predicate`` until it holds, or fail with ``message``.

    The player works on its own thread, so most assertions here are about
    something that becomes true shortly. A deadline says that in one line and
    fails readably; sleeping long enough to be safe would add seconds to every
    test in the file.
    """
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.005)
    raise AssertionError(f"не дождались: {message}")


def speech(
    request_id: str = "req",
    *,
    ms: int = PHRASE_MS,
    rate: int = DEVICE_RATE,
    priority: bool = False,
    level: int = 8000,
) -> SpeechRequest:
    """One finished phrase, ready to queue."""
    return SpeechRequest(
        text=f"фраза {request_id}",
        chunks=chunks_from(AudioChunk(tone(ms, rate, level=level), rate)),
        request_id=request_id,
        engine="stub",
        priority=priority,
    )


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def observer() -> Observer:
    return Observer()


@pytest.fixture
def player(backend: FakeBackend, observer: Observer) -> Iterator[TtsPlayer]:
    """A started player on the fake device, at full volume.

    Volume 1.0 so that a test asserting "this was not silence" is not quietly
    asserting the default gain instead.
    """
    instance = TtsPlayer(
        backend,
        volume=1.0,
        on_started=observer.on_started,
        on_finished=observer.on_finished,
    )
    instance.start()
    yield instance
    backend.release()
    instance.stop()


def build_player(backend: FakeBackend, observer: Observer, **kwargs: object) -> TtsPlayer:
    """A started player with the observer attached, for tests that need their own."""
    instance = TtsPlayer(
        backend,
        volume=1.0,
        on_started=observer.on_started,
        on_finished=observer.on_finished,
        **kwargs,
    )
    instance.start()
    return instance


def wait_for_finish(observer: Observer, request_id: str) -> str:
    """Block until ``request_id`` is reported finished, and return its reason."""
    wait_until(
        lambda: bool(observer.reasons(request_id)),
        f"фраза {request_id} не завершилась",
    )
    return observer.reasons(request_id)[0]


def wait_for_speaking(backend: FakeBackend) -> None:
    """Block until the writer thread is parked inside a write.

    Only meaningful with :attr:`FakeBackend.start_blocked` set before the phrase
    was queued.
    """
    wait_until(lambda: bool(backend.streams), "устройство не открылось")
    wait_until(backend.stream.wrote_once.is_set, "первый блок не записан")


# ----------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------


class TestPlayerLifecycle:
    """Start, stop, and the device that is not held while nothing is spoken."""

    def test_a_fresh_player_is_not_running(self, backend: FakeBackend):
        assert not TtsPlayer(backend).running

    def test_starting_opens_no_device(self, player: TtsPlayer, backend: FakeBackend):
        """An assistant that is silent must not hold an exclusive-mode device."""
        assert player.running
        assert backend.streams == []

    def test_starting_twice_is_safe(self, player: TtsPlayer):
        player.start()
        assert player.running

    def test_speaking_starts_the_player_by_itself(self, backend: FakeBackend, observer: Observer):
        """A caller that says one thing should not have to remember to start."""
        instance = TtsPlayer(backend, volume=1.0, on_finished=observer.on_finished)
        try:
            instance.speak(speech("req-1", ms=40))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        finally:
            instance.stop()

    def test_stopping_releases_the_device(self, backend: FakeBackend, observer: Observer):
        instance = build_player(backend, observer)
        instance.speak(speech("req-1", ms=40))
        wait_for_finish(observer, "req-1")
        instance.stop()
        assert backend.stream.closes >= 1

    def test_stopping_twice_is_safe(self, backend: FakeBackend):
        instance = TtsPlayer(backend)
        instance.start()
        instance.stop()
        instance.stop()
        assert not instance.running

    def test_stopping_a_player_that_never_started_is_safe(self, backend: FakeBackend):
        TtsPlayer(backend).stop()

    def test_nothing_is_speaking_after_a_stop(self, backend: FakeBackend):
        instance = TtsPlayer(backend)
        instance.start()
        instance.stop()
        assert not instance.running
        assert not instance.speaking


# ----------------------------------------------------------------------
# order
# ----------------------------------------------------------------------


class TestPlaybackOrder:
    """FIFO, except for the system messages that may not wait."""

    def test_one_phrase_reaches_the_device(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        player.speak(speech("req-1", ms=40))
        wait_for_finish(observer, "req-1")
        assert len(backend.written) > 0

    def test_the_whole_phrase_is_written(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """A phrase cut short by the block loop would clip the last word."""
        player.speak(speech("req-1", ms=100))
        wait_for_finish(observer, "req-1")
        assert len(backend.written) == len(tone(100))

    def test_a_phrase_is_written_in_small_blocks(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Block size is what bounds how long «стоп» takes to be heard."""
        player.speak(speech("req-1", ms=200))
        wait_for_finish(observer, "req-1")
        block = DEVICE_RATE * BLOCK_MS // 1000 * backend.stream.channels * SAMPLE_WIDTH
        assert backend.stream.writes
        assert all(len(write) <= block for write in backend.stream.writes)

    def test_three_phrases_are_spoken_in_order(self, player: TtsPlayer, observer: Observer):
        for index in range(3):
            player.speak(speech(f"req-{index}", ms=30))
        wait_until(lambda: len(observer.finished_ids) == 3, "три фразы не отзвучали")
        assert observer.finished_ids == ["req-0", "req-1", "req-2"]

    def test_a_system_message_jumps_the_queue(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """«Не расслышала» must not wait behind a three-paragraph answer."""
        backend.start_blocked = True
        player.speak(speech("first", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("answer", ms=30))
        player.speak(speech("system", ms=30, priority=True))
        backend.release()
        wait_until(lambda: len(observer.finished_ids) == 3, "очередь не разошлась")
        assert observer.finished_ids.index("system") < observer.finished_ids.index("answer")

    def test_a_system_message_does_not_interrupt_what_is_sounding(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Cutting a sentence in half to say «Готово» reads as a glitch."""
        backend.start_blocked = True
        player.speak(speech("answer", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("system", ms=30, priority=True))
        backend.release()
        wait_until(lambda: len(observer.finished_ids) == 2, "очередь не разошлась")
        assert observer.finished_ids == ["answer", "system"]

    def test_two_urgent_messages_keep_their_order(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("blocker", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("first", ms=30, priority=True))
        player.speak(speech("second", ms=30, priority=True))
        backend.release()
        wait_until(lambda: len(observer.finished_ids) == 3, "срочные не отзвучали")
        assert observer.finished_ids == ["blocker", "first", "second"]

    def test_the_queue_length_is_visible(self, player: TtsPlayer, backend: FakeBackend):
        backend.start_blocked = True
        player.speak(speech("one", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("two", ms=30))
        player.speak(speech("three", ms=30))
        assert player.queued == 2

    def test_an_empty_phrase_is_not_an_error(self, player: TtsPlayer, observer: Observer):
        """A punctuation-only answer arrives here as no chunks at all."""
        player.speak(SpeechRequest(text="...", chunks=(), request_id="req-empty"))
        assert wait_for_finish(observer, "req-empty") == PlaybackReason.COMPLETED

    def test_an_empty_phrase_opens_no_device(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        player.speak(SpeechRequest(text="...", chunks=(), request_id="req-empty"))
        wait_for_finish(observer, "req-empty")
        assert backend.streams == []

    def test_an_empty_phrase_still_announces_itself(self, player: TtsPlayer, observer: Observer):
        """The overlay opened on the answer and has to be told it is over."""
        player.speak(SpeechRequest(text="...", chunks=(), request_id="req-empty"))
        wait_for_finish(observer, "req-empty")
        assert observer.started_ids == ["req-empty"]


# ----------------------------------------------------------------------
# streaming
# ----------------------------------------------------------------------


class TestStreaming:
    """Chunks are pulled lazily, which is what the latency budget rests on."""

    def test_the_chunks_are_played_in_order(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        chunks = [
            AudioChunk(tone(30, level=1000), DEVICE_RATE),
            AudioChunk(tone(30, level=2000), DEVICE_RATE),
        ]
        player.speak(SpeechRequest(chunks=iter(chunks), request_id="req-1"))
        wait_for_finish(observer, "req-1")
        written = backend.written
        half = len(written) // 2
        assert peak(written[:half]) == 1000
        assert peak(written[half:]) == 2000

    def test_the_source_is_consumed_lazily(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Sentence two is synthesized while sentence one is sounding.

        This is the whole point of the streaming path. Draining the iterator up
        front would mean waiting for the *last* sentence before the first sound,
        which is exactly the delay task 12 sets a 500 ms budget against.
        """
        gate = threading.Event()
        pulled: list[int] = []

        def source() -> Iterator[AudioChunk]:
            pulled.append(0)
            yield AudioChunk(tone(60), DEVICE_RATE)
            gate.wait(TIMEOUT_S)
            pulled.append(1)
            yield AudioChunk(tone(60), DEVICE_RATE)

        player.speak(SpeechRequest(chunks=source(), request_id="req-1"))
        wait_until(lambda: len(backend.written) >= len(tone(60)), "первый чанк не проигран")
        assert pulled == [0]
        gate.set()
        wait_for_finish(observer, "req-1")
        assert pulled == [0, 1]

    def test_an_empty_chunk_in_the_middle_is_skipped(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        chunks = [
            AudioChunk(tone(30), DEVICE_RATE),
            AudioChunk(b"", DEVICE_RATE),
            AudioChunk(tone(30), DEVICE_RATE),
        ]
        player.speak(SpeechRequest(chunks=iter(chunks), request_id="req-1"))
        wait_for_finish(observer, "req-1")
        assert len(backend.written) == len(tone(60))

    def test_a_failing_source_ends_the_phrase_as_an_error(
        self, player: TtsPlayer, observer: Observer
    ):
        """The engine raising on sentence three must not kill the player."""

        def source() -> Iterator[AudioChunk]:
            yield AudioChunk(tone(30), DEVICE_RATE)
            raise RuntimeError("движок отказал")

        player.speak(SpeechRequest(chunks=source(), request_id="req-1"))
        assert wait_for_finish(observer, "req-1") == PlaybackReason.ERROR

    def test_what_was_already_written_stays_audible(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Half an answer beats none when the failure is the engine's, not the user's."""

        def source() -> Iterator[AudioChunk]:
            yield AudioChunk(tone(30), DEVICE_RATE)
            raise RuntimeError("движок отказал")

        player.speak(SpeechRequest(chunks=source(), request_id="req-1"))
        wait_for_finish(observer, "req-1")
        assert len(backend.written) == len(tone(30))

    def test_the_player_survives_a_failing_source(self, player: TtsPlayer, observer: Observer):
        def source() -> Iterator[AudioChunk]:
            raise RuntimeError("движок отказал")
            yield  # pragma: no cover - unreachable, but keeps this a generator

        player.speak(SpeechRequest(chunks=source(), request_id="req-bad"))
        wait_for_finish(observer, "req-bad")
        player.speak(speech("req-good", ms=30))
        assert wait_for_finish(observer, "req-good") == PlaybackReason.COMPLETED


# ----------------------------------------------------------------------
# cancellation
# ----------------------------------------------------------------------


class TestCancellation:
    """What «Айрис, стоп» silences."""

    def test_cancelling_nothing_reports_nothing(self, player: TtsPlayer):
        assert player.cancel() is False

    def test_a_queued_phrase_is_dropped(self, player: TtsPlayer, backend: FakeBackend):
        backend.start_blocked = True
        player.speak(speech("first", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("second", ms=40))
        assert player.cancel() is True
        assert player.queued == 0

    def test_a_dropped_phrase_is_reported_cancelled(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """The overlay waits for a ``TtsFinished`` for every phrase it opened."""
        backend.start_blocked = True
        player.speak(speech("first", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("second", ms=40))
        player.cancel()
        assert wait_for_finish(observer, "second") == PlaybackReason.CANCELLED

    def test_the_current_phrase_stops_mid_word(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """The audible requirement: not "after this sentence", but now."""
        backend.start_blocked = True
        player.speak(speech("req-1", ms=2000))
        wait_for_speaking(backend)
        player.cancel()
        assert wait_for_finish(observer, "req-1") == PlaybackReason.CANCELLED
        assert len(backend.written) < len(tone(2000))

    def test_the_driver_buffer_is_thrown_away(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Without the abort the user hears up to a bufferful after «стоп»."""
        backend.start_blocked = True
        player.speak(speech("req-1", ms=2000))
        wait_for_speaking(backend)
        player.cancel()
        wait_for_finish(observer, "req-1")
        assert backend.stream.stops >= 1

    def test_cancelling_by_id_stops_that_phrase(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("mine", ms=2000))
        wait_for_speaking(backend)
        assert player.cancel("mine") is True
        assert wait_for_finish(observer, "mine") == PlaybackReason.CANCELLED

    def test_cancelling_another_id_leaves_the_current_phrase_alone(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """A caller interrupting its own answer must not silence a newer one."""
        backend.start_blocked = True
        player.speak(speech("current", ms=40))
        wait_for_speaking(backend)
        assert player.cancel("старый ответ") is False
        backend.release()
        assert wait_for_finish(observer, "current") == PlaybackReason.COMPLETED

    def test_cancelling_by_id_drops_it_from_the_queue(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("current", ms=40))
        wait_for_speaking(backend)
        player.speak(speech("doomed", ms=40))
        assert player.cancel("doomed") is True
        backend.release()
        assert wait_for_finish(observer, "doomed") == PlaybackReason.CANCELLED
        assert wait_for_finish(observer, "current") == PlaybackReason.COMPLETED

    def test_the_player_speaks_again_after_a_cancel(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """A cancel silences one answer, not the assistant."""
        backend.start_blocked = True
        player.speak(speech("req-1", ms=2000))
        wait_for_speaking(backend)
        player.cancel()
        wait_for_finish(observer, "req-1")
        backend.release()
        player.speak(speech("req-2", ms=40))
        assert wait_for_finish(observer, "req-2") == PlaybackReason.COMPLETED

    def test_stopping_the_player_cancels_the_queue(self, backend: FakeBackend, observer: Observer):
        instance = build_player(backend, observer)
        backend.start_blocked = True
        instance.speak(speech("first", ms=40))
        wait_for_speaking(backend)
        instance.speak(speech("second", ms=40))
        instance.stop()
        assert observer.reasons("second") == [PlaybackReason.CANCELLED]

    @pytest.mark.hardware
    def test_a_real_device_goes_silent_immediately(self):  # pragma: no cover - needs speakers
        pytest.skip("нужны настоящие колонки — проверяется на машине пользователя")


# ----------------------------------------------------------------------
# events
# ----------------------------------------------------------------------


class TestEvents:
    """What ``TtsStarted`` and ``TtsFinished`` are built from."""

    def test_a_phrase_announces_its_start(self, player: TtsPlayer, observer: Observer):
        player.speak(speech("req-1", ms=40))
        wait_for_finish(observer, "req-1")
        assert observer.started_ids == ["req-1"]

    def test_the_start_carries_a_duration(self, player: TtsPlayer, observer: Observer):
        """The overlay sizes its progress line with it."""
        player.speak(speech("req-1", ms=200))
        wait_for_finish(observer, "req-1")
        assert observer.started[0][1] > 0

    def test_an_explicit_estimate_wins(self, player: TtsPlayer, observer: Observer):
        """A streaming phrase knows its total length before the first chunk does."""
        request = speech("req-1", ms=40)
        request.duration_estimate_ms = 1234
        player.speak(request)
        wait_for_finish(observer, "req-1")
        assert observer.started[0][1] == 1234

    def test_the_start_fires_once_per_phrase(self, player: TtsPlayer, observer: Observer):
        """Three sentences are one answer, and the overlay opens once."""
        chunks = [AudioChunk(tone(30), DEVICE_RATE) for _ in range(3)]
        player.speak(SpeechRequest(chunks=iter(chunks), request_id="req-1"))
        wait_for_finish(observer, "req-1")
        assert observer.started_ids == ["req-1"]

    def test_a_completed_phrase_says_so(self, player: TtsPlayer, observer: Observer):
        player.speak(speech("req-1", ms=40))
        assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED

    def test_every_phrase_finishes_exactly_once(self, player: TtsPlayer, observer: Observer):
        """A phrase reported twice leaks a speaking indicator in the overlay."""
        for index in range(3):
            player.speak(speech(f"req-{index}", ms=30))
        wait_until(lambda: len(observer.finished_ids) == 3, "не все завершились")
        assert sorted(observer.finished_ids) == ["req-0", "req-1", "req-2"]

    def test_a_failing_start_observer_does_not_silence_the_assistant(
        self, backend: FakeBackend, observer: Observer
    ):
        def broken(_request: SpeechRequest, _duration: int) -> None:
            raise RuntimeError("подписчик упал")

        instance = TtsPlayer(
            backend, volume=1.0, on_started=broken, on_finished=observer.on_finished
        )
        instance.start()
        try:
            instance.speak(speech("req-1", ms=40))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        finally:
            instance.stop()

    def test_a_failing_finish_observer_does_not_stop_the_queue(self, backend: FakeBackend):
        seen: list[str] = []

        def broken(request: SpeechRequest, _reason: str) -> None:
            seen.append(request.request_id)
            raise RuntimeError("подписчик упал")

        instance = TtsPlayer(backend, volume=1.0, on_finished=broken)
        instance.start()
        try:
            instance.speak(speech("req-1", ms=30))
            instance.speak(speech("req-2", ms=30))
            wait_until(lambda: len(seen) == 2, "вторая фраза не отзвучала")
        finally:
            instance.stop()

    def test_the_observers_can_be_replaced_after_construction(
        self, backend: FakeBackend, observer: Observer
    ):
        """The bus bridge is built around the player, not before it."""
        instance = TtsPlayer(backend, volume=1.0)
        instance.set_observers(on_started=observer.on_started, on_finished=observer.on_finished)
        instance.start()
        try:
            instance.speak(speech("req-1", ms=30))
            wait_for_finish(observer, "req-1")
        finally:
            instance.stop()
        assert observer.started_ids == ["req-1"]


# ----------------------------------------------------------------------
# volume
# ----------------------------------------------------------------------


class TestVolume:
    """The gain, applied to the samples rather than to the system mixer."""

    def test_the_default_is_audible(self, backend: FakeBackend):
        assert TtsPlayer(backend).volume > 0.0

    def test_it_is_clamped_on_construction(self, backend: FakeBackend):
        assert TtsPlayer(backend, volume=5.0).volume == 1.0
        assert TtsPlayer(backend, volume=-1.0).volume == 0.0

    def test_it_is_clamped_when_set(self, player: TtsPlayer):
        player.set_volume(99.0)
        assert player.volume == 1.0

    def test_full_volume_passes_the_samples_through(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        player.speak(speech("req-1", ms=40, level=8000))
        wait_for_finish(observer, "req-1")
        assert peak(backend.written) == 8000

    def test_half_volume_halves_the_samples(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        player.set_volume(0.5)
        player.speak(speech("req-1", ms=40, level=8000))
        wait_for_finish(observer, "req-1")
        assert peak(backend.written) == pytest.approx(4000, abs=2)

    def test_zero_volume_is_silence_not_a_skipped_phrase(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Muted still fires the events: the overlay shows the text either way."""
        player.set_volume(0.0)
        player.speak(speech("req-1", ms=40))
        assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        assert peak(backend.written) == 0

    def test_a_volume_change_applies_within_a_phrase(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """The settings slider must not look stuck until the next sentence."""
        backend.start_blocked = True
        player.speak(speech("req-1", ms=200, level=8000))
        wait_for_speaking(backend)
        player.set_volume(0.0)
        backend.release()
        wait_for_finish(observer, "req-1")
        assert peak(backend.stream.writes[-1]) == 0


# ----------------------------------------------------------------------
# the device
# ----------------------------------------------------------------------


class TestDeviceHandling:
    """Rate negotiation, resampling, and a device that vanishes."""

    def test_the_native_rate_is_tried_first(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Resampling Piper's 22.05 kHz when the device would take it is waste."""
        player.speak(speech("req-1", ms=40, rate=22050))
        wait_for_finish(observer, "req-1")
        assert backend.stream.sample_rate == 22050

    def test_a_refused_rate_falls_back(self, observer: Observer):
        """Shared-mode WASAPI usually wants 48 kHz; Piper would be unusable."""
        backend = FakeBackend(rates=(DEVICE_RATE,))
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40, rate=22050))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        finally:
            instance.stop()
        assert backend.stream.sample_rate == DEVICE_RATE
        assert 22050 in backend.refused

    def test_the_fallback_resamples_rather_than_playing_it_fast(self, observer: Observer):
        """The tell is duration: the same speech, at the device's rate."""
        backend = FakeBackend(rates=(DEVICE_RATE,))
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=100, rate=22050))
            wait_for_finish(observer, "req-1")
        finally:
            instance.stop()
        frames = len(backend.written) // (SAMPLE_WIDTH * backend.stream.channels)
        assert frames / DEVICE_RATE == pytest.approx(0.1, abs=0.02)

    def test_a_device_that_cannot_be_opened_is_an_error_not_a_crash(
        self, backend: FakeBackend, observer: Observer
    ):
        backend.fail_open = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.ERROR
        finally:
            instance.stop()

    def test_the_player_recovers_when_the_device_comes_back(
        self, backend: FakeBackend, observer: Observer
    ):
        """A wrong device in the settings must not need a restart to fix."""
        backend.fail_open = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            wait_for_finish(observer, "req-1")
            backend.fail_open = False
            instance.speak(speech("req-2", ms=40))
            assert wait_for_finish(observer, "req-2") == PlaybackReason.COMPLETED
        finally:
            instance.stop()

    def test_an_unplugged_device_is_reopened(self, backend: FakeBackend, observer: Observer):
        """Headphones out mid-sentence: continue on the speakers, do not die."""
        backend.fail_first_write = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            wait_for_finish(observer, "req-1")
        finally:
            instance.stop()
        assert len(backend.streams) == 2

    def test_the_phrase_survives_the_hot_plug(self, backend: FakeBackend, observer: Observer):
        backend.fail_first_write = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        finally:
            instance.stop()

    def test_a_lost_device_is_closed_not_leaked(self, backend: FakeBackend, observer: Observer):
        backend.fail_first_write = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            wait_for_finish(observer, "req-1")
        finally:
            instance.stop()
        assert backend.streams[0].closes >= 1

    def test_setting_the_same_device_changes_nothing(self, player: TtsPlayer):
        player.set_device("")
        assert player.stats().device_id == ""

    def test_a_device_change_is_accepted(self, player: TtsPlayer, observer: Observer):
        """The current phrase finishes on the old device; the next one moves."""
        player.set_device("output:wasapi:наушники")
        player.speak(speech("req-1", ms=40))
        assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED

    def test_an_unknown_device_falls_back_to_the_default(
        self, backend: FakeBackend, observer: Observer
    ):
        """A stale entry in the settings must not leave the assistant mute."""
        instance = build_player(backend, observer, device="output:wasapi:которого-нет")
        try:
            instance.speak(speech("req-1", ms=40))
            assert wait_for_finish(observer, "req-1") == PlaybackReason.COMPLETED
        finally:
            instance.stop()

    def test_mono_audio_opens_a_mono_stream(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        """Every engine here is mono; opening stereo would duplicate nothing."""
        player.speak(speech("req-1", ms=40))
        wait_for_finish(observer, "req-1")
        assert backend.stream.channels == 1

    def test_only_output_devices_are_considered(self, backend: FakeBackend):
        """Guard against picking the microphone: it has no ``write``."""
        from ayris.audio.devices import list_devices

        devices = list_devices(backend, DeviceDirection.OUTPUT)
        assert devices
        assert all(device.direction is DeviceDirection.OUTPUT for device in devices)


# ----------------------------------------------------------------------
# stats
# ----------------------------------------------------------------------


class TestStats:
    """What DevTools shows on the playback row."""

    def test_a_fresh_player_has_empty_stats(self, backend: FakeBackend):
        stats = TtsPlayer(backend).stats()
        assert stats.queued == 0
        assert stats.speaking is False
        assert stats.played == 0

    def test_played_phrases_are_counted(self, player: TtsPlayer, observer: Observer):
        for index in range(2):
            player.speak(speech(f"req-{index}", ms=30))
        wait_until(lambda: len(observer.finished_ids) == 2, "фразы не отзвучали")
        assert player.stats().played == 2

    def test_cancelled_phrases_are_counted(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("req-1", ms=2000))
        wait_for_speaking(backend)
        player.cancel()
        wait_for_finish(observer, "req-1")
        assert player.stats().cancelled == 1

    def test_failed_phrases_are_counted(self, backend: FakeBackend, observer: Observer):
        backend.fail_open = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            wait_for_finish(observer, "req-1")
            assert instance.stats().failed == 1
        finally:
            instance.stop()

    def test_the_open_device_is_reported(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("req-1", ms=40))
        wait_for_speaking(backend)
        stats = player.stats()
        backend.release()
        wait_for_finish(observer, "req-1")
        assert stats.device_id
        assert stats.sample_rate == DEVICE_RATE

    def test_a_hot_plug_is_counted(self, backend: FakeBackend, observer: Observer):
        backend.fail_first_write = True
        instance = build_player(backend, observer)
        try:
            instance.speak(speech("req-1", ms=40))
            wait_for_finish(observer, "req-1")
            assert instance.stats().underruns >= 1
        finally:
            instance.stop()

    def test_speaking_is_visible_while_a_phrase_sounds(
        self, player: TtsPlayer, backend: FakeBackend, observer: Observer
    ):
        backend.start_blocked = True
        player.speak(speech("req-1", ms=40))
        wait_for_speaking(backend)
        assert player.speaking
        backend.release()
        wait_for_finish(observer, "req-1")
        assert not player.speaking
