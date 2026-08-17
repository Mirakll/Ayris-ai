"""Task 07: microphone capture — devices, ring buffer, DSP, the worker's shell.

Not one test here opens a sound card. Everything that would touch PortAudio goes
through :class:`FakeBackend`, which is the whole reason
:class:`~ayris.audio.devices.AudioBackend` exists as a protocol: unplugging a
microphone mid-phrase, a driver that refuses 16 kHz, a machine with no devices at
all — none of those can be arranged on a CI runner, and all of them are the cases
that actually break capture in the field.

Groups:

* :class:`TestRingBuffer` — writes, wrap-around, pre-roll, absolute positions.
* :class:`TestLevels` — RMS/peak/clipping on synthetic waveforms.
* :class:`TestGain` — the soft limiter, and hard clipping when it is off.
* :class:`TestDownmix` — interleaved channels averaged to mono.
* :class:`TestResampler` — 48k→16k decimation, 44.1k→16k interpolation, seams.
* :class:`TestDeviceIdentity` — identifiers that survive a hot-plug.
* :class:`TestEnumeration` — listing, defaults, resolution, diffing.
* :class:`TestCapture` — the pipeline against the fake backend.
* :class:`TestHotPlug` — device loss, notification, automatic recovery.
* :class:`TestWav` — the debug dump.
* :class:`TestWorker` — the worker's method surface and its bus translation.
* :class:`TestDetection` — task 08's VAD, segment and calibration methods.
"""

from __future__ import annotations

import math
import struct
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayris.audio.capture import (
    TARGET_SAMPLE_RATE,
    AudioCapture,
    AudioLevel,
    CaptureCallbacks,
    CaptureSettings,
    CaptureState,
    Resampler,
    apply_gain,
    downmix,
    pcm_level,
    write_wav,
)
from ayris.audio.devices import (
    DeviceChange,
    DeviceDirection,
    RawDevice,
    StreamCallback,
    StreamRequest,
    clean_device_name,
    default_device,
    describe_devices,
    device_id,
    diff_devices,
    find_device,
    list_devices,
    resolve_device,
)
from ayris.audio.ring_buffer import SAMPLE_WIDTH, RingBuffer
from ayris.core.errors import AudioError
from ayris.core.events import AudioLevelChanged, NotificationRequested
from ayris.workers.audio_worker import AudioWorker, translate_audio_event
from ayris.workers.registry import WorkerKind, event_translator

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ayris.core.models import JsonObject

pytestmark = pytest.mark.unit

#: Long enough that a loaded runner does not fail a test about audio, short
#: enough that a genuine hang does not stall the suite.
WAIT = 5.0


def wait_for(predicate: Callable[[], object], timeout: float = WAIT) -> bool:
    """Poll ``predicate`` until it is true or the time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def tone(
    frames: int, amplitude: int = 8000, rate: int = TARGET_SAMPLE_RATE, hz: float = 440.0
) -> bytes:
    """A mono ``int16`` sine, as PCM bytes."""
    samples = array("h", bytes(frames * SAMPLE_WIDTH))
    for index in range(frames):
        samples[index] = int(amplitude * math.sin(2.0 * math.pi * hz * index / rate))
    return samples.tobytes()


def ramp(count: int, start: int = 0) -> bytes:
    """``count`` frames whose sample value is their own index."""
    return array("h", range(start, start + count)).tobytes()


def samples_of(pcm: bytes) -> list[int]:
    """Decode PCM back into a list of ints."""
    decoded = array("h")
    decoded.frombytes(pcm)
    return list(decoded)


# ----------------------------------------------------------------------
# the fake sound card
# ----------------------------------------------------------------------


class FakeStream:
    """A capture stream that produces exactly the audio a test feeds it."""

    def __init__(self, request: StreamRequest, callback: StreamCallback) -> None:
        self.request = request
        self.callback = callback
        self.started = False
        self.closed = False
        #: Set by a test to simulate a device the driver killed without telling us.
        self.dead = False

    @property
    def sample_rate(self) -> int:
        return self.request.sample_rate

    @property
    def channels(self) -> int:
        return self.request.channels

    @property
    def active(self) -> bool:
        return self.started and not self.closed and not self.dead

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def feed(self, pcm: bytes, *, overflowed: bool = False) -> None:
        """Deliver one block the way PortAudio's real-time thread would."""
        self.callback(pcm, overflowed)


class FakeBackend:
    """An :class:`~ayris.audio.devices.AudioBackend` with no hardware behind it.

    ``devices`` is writable so a test can unplug a microphone between two
    monitor ticks, and ``accepts`` decides which formats the imaginary driver
    admits to supporting — the knob that exercises resampling and downmixing.
    """

    def __init__(
        self,
        devices: Sequence[RawDevice] | None = None,
        *,
        accepts: Callable[[StreamRequest], bool] | None = None,
    ) -> None:
        self.devices: list[RawDevice] = list(devices if devices is not None else [usb_mic()])
        self.accepts = (
            accepts
            if accepts is not None
            else (
                lambda request: request.sample_rate == TARGET_SAMPLE_RATE and request.channels == 1
            )
        )
        self.streams: list[FakeStream] = []
        self.refreshes = 0
        #: Raise on the next open, the way a device grabbed by another
        #: application does.
        self.open_fails = False

    def raw_devices(self) -> Sequence[RawDevice]:
        return tuple(self.devices)

    def refresh(self) -> None:
        self.refreshes += 1

    def open_input_stream(self, request: StreamRequest, callback: StreamCallback) -> FakeStream:
        if self.open_fails:
            raise AudioError("device is busy")
        if not any(device.index == request.device_index for device in self.devices):
            raise AudioError(f"device {request.device_index} is gone")
        stream = FakeStream(request, callback)
        self.streams.append(stream)
        return stream

    def supports_rate(self, request: StreamRequest) -> bool:
        return self.accepts(request)

    @property
    def stream(self) -> FakeStream:
        """The stream opened most recently."""
        return self.streams[-1]


def usb_mic(index: int = 0, name: str = "Микрофон (USB Audio Device)") -> RawDevice:
    """The default input device."""
    return RawDevice(
        index=index,
        name=name,
        host_api="MME",
        max_input_channels=1,
        default_sample_rate=16000.0,
        default_input=True,
    )


def webcam_mic(index: int = 1) -> RawDevice:
    """A second input, 48 kHz stereo, so conversion is needed."""
    return RawDevice(
        index=index,
        name="Микрофон (Webcam)",
        host_api="MME",
        max_input_channels=2,
        default_sample_rate=48000.0,
    )


def speakers(index: int = 2) -> RawDevice:
    """Output-only, to prove enumeration filters by direction."""
    return RawDevice(
        index=index,
        name="Динамики (Realtek)",
        host_api="MME",
        max_output_channels=2,
        default_output=True,
    )


@pytest.fixture
def backend() -> FakeBackend:
    """One 16 kHz mono microphone and a pair of speakers."""
    return FakeBackend([usb_mic(), speakers()])


@pytest.fixture
def capture(backend: FakeBackend) -> Any:
    """A started pipeline, torn down however the test leaves it."""
    instance = AudioCapture(
        CaptureSettings(frame_ms=20, device_poll_sec=0.05, level_interval_ms=10),
        backend=backend,
    )
    instance.start()
    yield instance
    instance.stop()


# ----------------------------------------------------------------------
# ring buffer
# ----------------------------------------------------------------------


class TestRingBuffer:
    """Everything a consumer of the rolling window relies on."""

    def test_a_short_write_reads_back_verbatim(self):
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        ring.write(ramp(100))
        assert ring.available_frames == 100
        assert ring.position == 100
        assert samples_of(ring.read_last(frames=100)) == list(range(100))

    def test_the_oldest_audio_is_evicted_once_the_window_is_full(self):
        ring = RingBuffer(seconds=0.1, sample_rate=16000)  # 1600 frames
        ring.write(ramp(1000))
        ring.write(ramp(1000, start=1000))
        assert ring.available_frames == 1600
        assert ring.overwritten_frames == 400
        # The window holds the *newest* 1600 of the 2000 frames written.
        assert samples_of(ring.read_last(frames=1600)) == list(range(400, 2000))

    def test_a_read_follows_the_wrap(self):
        ring = RingBuffer(seconds=0.1, sample_rate=16000)
        for _ in range(5):
            ring.write(ramp(1000))
        tail = samples_of(ring.read_last(frames=10))
        assert tail == list(range(990, 1000)), "a wrapped read must not tear at the seam"

    def test_a_block_larger_than_the_window_keeps_its_tail(self):
        ring = RingBuffer(seconds=0.1, sample_rate=16000)
        ring.write(ramp(5000))
        assert ring.available_frames == 1600
        assert samples_of(ring.read_last(frames=1600)) == list(range(3400, 5000))
        assert ring.position == 5000

    def test_pre_roll_returns_the_audio_that_came_before_now(self):
        ring = RingBuffer(seconds=5.0, sample_rate=16000)
        ring.write(ramp(16000))
        pre_roll = ring.read_last(ms=400)
        assert len(pre_roll) // SAMPLE_WIDTH == 6400
        assert samples_of(pre_roll)[0] == 16000 - 6400

    def test_pre_roll_is_short_rather_than_padded_on_a_cold_buffer(self):
        ring = RingBuffer(seconds=5.0, sample_rate=16000)
        ring.write(ramp(100))
        assert len(ring.read_last(ms=400)) // SAMPLE_WIDTH == 100

    def test_reading_from_a_position_returns_only_what_is_new(self):
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        ring.write(ramp(100))
        start = ring.position
        assert not ring.read_from(start)
        ring.write(ramp(50, start=100))
        read = ring.read_from(start)
        assert read.frames == 50
        assert samples_of(read.pcm) == list(range(100, 150))
        assert read.dropped == 0
        assert not ring.read_from(read.position), "a second read must not repeat frames"

    def test_a_slow_reader_is_told_how_much_it_missed(self):
        ring = RingBuffer(seconds=0.1, sample_rate=16000)
        ring.write(ramp(1000))
        stale = ring.position - 1000
        ring.write(ramp(2000, start=1000))
        read = ring.read_from(stale)
        assert read.dropped == 1400, "3000 frames written into a 1600-frame window"
        assert read.frames == 1600
        assert read.position == 3000

    def test_a_position_from_the_future_is_clamped_rather_than_read_ahead(self):
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        ring.write(ramp(100))
        assert ring.read_from(10_000).frames == 0

    def test_clear_drops_audio_without_rewinding_positions(self):
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        ring.write(ramp(100))
        ring.clear()
        assert ring.available_frames == 0
        assert ring.position == 100, "consumers must never see their position go backwards"

    def test_a_half_sample_write_is_rejected(self):
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        with pytest.raises(ValueError, match="whole int16 frames"):
            ring.write(b"\x00\x01\x02")

    @pytest.mark.parametrize(("seconds", "rate"), [(0.0, 16000), (1.0, 0), (-1.0, 16000)])
    def test_a_nonsensical_size_is_refused(self, seconds: float, rate: int):
        with pytest.raises(ValueError):
            RingBuffer(seconds=seconds, sample_rate=rate)

    def test_a_tiny_window_is_raised_to_something_usable(self):
        ring = RingBuffer(seconds=0.001, sample_rate=16000)
        assert ring.capacity_ms >= 100

    def test_concurrent_writes_and_reads_stay_consistent(self):
        """The capture thread writes while consumers read; neither may tear."""
        ring = RingBuffer(seconds=1.0, sample_rate=16000)
        stop = threading.Event()
        errors: list[BaseException] = []

        def writer() -> None:
            try:
                while not stop.is_set():
                    ring.write(ramp(320))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                assert len(ring.read_last(ms=100)) % SAMPLE_WIDTH == 0
                start = ring.position
                read = ring.read_from(start)
                # The writer keeps running underneath, so the amount returned is
                # unpredictable; what must hold is that the answer is internally
                # consistent and never over-reports.
                assert read.frames == len(read.pcm) // SAMPLE_WIDTH
                assert read.position == start + read.frames + read.dropped
                assert read.frames <= ring.capacity_frames
        finally:
            stop.set()
            thread.join(timeout=WAIT)
        assert not errors


# ----------------------------------------------------------------------
# dsp
# ----------------------------------------------------------------------


class TestLevels:
    """The number the overlay's sphere is drawn from."""

    def test_silence_measures_zero(self):
        level = pcm_level(bytes(640))
        assert level == AudioLevel()
        assert level.rms_db == pytest.approx(-100.0)

    def test_an_empty_block_is_not_a_division_by_zero(self):
        assert pcm_level(b"") == AudioLevel()

    def test_full_scale_square_wave_measures_one(self):
        pcm = array("h", [32767, -32767] * 100).tobytes()
        level = pcm_level(pcm)
        assert level.rms == pytest.approx(1.0, abs=0.001)
        assert level.peak == pytest.approx(1.0, abs=0.001)

    def test_a_sine_measures_its_own_rms(self):
        level = pcm_level(tone(1600, amplitude=16384))
        assert level.peak == pytest.approx(0.5, abs=0.01)
        # RMS of a sine is peak / sqrt(2).
        assert level.rms == pytest.approx(0.5 / math.sqrt(2), abs=0.01)

    def test_clipping_is_reported(self):
        assert pcm_level(array("h", [32767] * 10).tobytes()).clipped
        assert pcm_level(array("h", [-32768] * 10).tobytes()).clipped
        assert not pcm_level(array("h", [32000] * 10).tobytes()).clipped

    def test_decibels_are_floored_rather_than_infinite(self):
        assert AudioLevel(rms=0.0).rms_db == pytest.approx(-100.0)
        assert AudioLevel(rms=1.0).rms_db == pytest.approx(0.0)
        assert AudioLevel(peak=0.5).peak_db == pytest.approx(-6.02, abs=0.01)


class TestGain:
    """Amplifying a quiet microphone without turning speech into a square wave."""

    def test_unity_gain_returns_the_input(self):
        pcm = tone(320)
        assert apply_gain(pcm, 1.0) == pcm

    def test_quiet_audio_scales_linearly_below_the_knee(self):
        pcm = array("h", [1000, -1000]).tobytes()
        assert samples_of(apply_gain(pcm, 2.0)) == [2000, -2000]

    def test_the_limiter_bends_instead_of_clipping(self):
        pcm = array("h", [20000, -20000]).tobytes()
        limited = samples_of(apply_gain(pcm, 4.0))
        assert all(abs(value) < 32767 for value in limited), "a limiter must leave headroom"
        assert all(abs(value) > 29000 for value in limited), "and still be loud"
        assert limited[0] == -limited[1], "the curve must stay symmetric"

    def test_the_limiter_never_overflows_int16(self):
        pcm = array("h", [32767, -32768]).tobytes()
        for value in samples_of(apply_gain(pcm, 50.0)):
            assert -32768 <= value <= 32767

    def test_hard_clipping_is_available_for_a_predictable_curve(self):
        pcm = array("h", [20000]).tobytes()
        assert samples_of(apply_gain(pcm, 4.0, limiter=False)) == [32767]

    def test_attenuation_needs_no_limiter_at_all(self):
        pcm = array("h", [30000, -30000]).tobytes()
        assert samples_of(apply_gain(pcm, 0.5)) == [15000, -15000]


class TestDownmix:
    """Interleaved channels averaged to mono."""

    def test_mono_passes_through_untouched(self):
        pcm = tone(320)
        assert downmix(pcm, 1) == pcm

    def test_stereo_is_averaged(self):
        pcm = array("h", [100, 300, -200, 0]).tobytes()
        assert samples_of(downmix(pcm, 2)) == [200, -100]

    def test_a_microphone_wired_to_one_channel_is_not_lost(self):
        """Taking channel 0 would return silence for a right-wired headset."""
        pcm = array("h", [0, 8000] * 10).tobytes()
        assert all(value == 4000 for value in samples_of(downmix(pcm, 2)))

    def test_four_channels_collapse_too(self):
        pcm = array("h", [100, 200, 300, 400]).tobytes()
        assert samples_of(downmix(pcm, 4)) == [250]


class TestResampler:
    """Whatever the driver hands over has to end up at 16 kHz."""

    def test_matching_rates_are_a_passthrough(self):
        resampler = Resampler(16000, 16000)
        assert resampler.passthrough
        pcm = tone(320)
        assert resampler.process(pcm) == pcm

    def test_an_integer_ratio_decimates_by_averaging(self):
        resampler = Resampler(48000, 16000)
        assert not resampler.passthrough
        out = samples_of(resampler.process(array("h", [0, 3, 6, 9, 12, 15]).tobytes()))
        assert out == [3, 12], "each output frame is the mean of three input frames"

    def test_a_48k_block_becomes_a_third_as_many_frames(self):
        resampler = Resampler(48000, 16000)
        pcm = resampler.process(tone(960, rate=48000))
        assert len(pcm) // SAMPLE_WIDTH == 320

    def test_a_fractional_ratio_lands_within_a_frame_of_the_ideal_count(self):
        resampler = Resampler(44100, 16000)
        pcm = resampler.process(tone(4410, rate=44100))
        assert abs(len(pcm) // SAMPLE_WIDTH - 1600) <= 2

    def test_a_resampled_sine_stays_a_sine(self):
        """Level survives the conversion; a broken resampler shows up as silence."""
        resampler = Resampler(48000, 16000)
        pcm = resampler.process(tone(4800, amplitude=16384, rate=48000, hz=200.0))
        assert pcm_level(pcm).peak == pytest.approx(0.5, abs=0.05)

    @pytest.mark.parametrize("source", [48000, 44100, 32000, 22050])
    def test_block_boundaries_do_not_lose_or_duplicate_audio(self, source: int):
        """Feeding one long block and many short ones must agree.

        A resampler that forgets its phase between calls drifts, and the drift
        only shows up as a slowly growing offset like this.
        """
        frames = source // 10  # 100 ms
        pcm = tone(frames, rate=source, hz=300.0)
        whole = len(Resampler(source, 16000).process(pcm)) // SAMPLE_WIDTH

        chunked = Resampler(source, 16000)
        total = 0
        step = frames // 10
        for offset in range(0, frames, step):
            total += len(
                chunked.process(pcm[offset * SAMPLE_WIDTH : (offset + step) * SAMPLE_WIDTH])
            )
        assert abs(total // SAMPLE_WIDTH - whole) <= 2

    def test_reset_forgets_the_carried_samples(self):
        resampler = Resampler(48000, 16000)
        resampler.process(array("h", [1, 2]).tobytes())  # two frames, none emitted yet
        resampler.reset()
        assert samples_of(resampler.process(array("h", [0, 3, 6]).tobytes())) == [3]

    def test_an_impossible_rate_is_refused(self):
        with pytest.raises(ValueError):
            Resampler(0, 16000)


# ----------------------------------------------------------------------
# devices
# ----------------------------------------------------------------------


class TestDeviceIdentity:
    """Identifiers that survive replugging, which indices do not."""

    def test_the_positional_counter_is_not_part_of_the_name(self):
        assert clean_device_name("Микрофон (2- USB Audio Device)") == "Микрофон (USB Audio Device)"
        assert clean_device_name("Микрофон  (USB)") == "Микрофон (USB)"

    def test_the_same_microphone_in_another_port_keeps_its_identifier(self):
        first = device_id("Микрофон (USB Audio Device)", "MME", DeviceDirection.INPUT)
        second = device_id("Микрофон (3- USB Audio Device)", "MME", DeviceDirection.INPUT)
        assert first == second

    def test_input_and_output_of_one_device_are_different_identifiers(self):
        assert device_id("Realtek", "MME", DeviceDirection.INPUT) != device_id(
            "Realtek", "MME", DeviceDirection.OUTPUT
        )

    def test_the_host_api_separates_two_views_of_one_device(self):
        assert device_id("Микрофон", "MME", DeviceDirection.INPUT) != device_id(
            "Микрофон", "Windows WASAPI", DeviceDirection.INPUT
        )

    def test_the_identifier_survives_a_shifting_index(self):
        backend = FakeBackend([webcam_mic(index=0), usb_mic(index=1)])
        before = list_devices(backend)
        backend.devices = [usb_mic(index=0)]
        after = list_devices(backend)
        assert before[1].index == 1
        assert after[0].index == 0
        assert before[1].id == after[0].id, "unplugging a neighbour must not rename a device"

    def test_two_identical_devices_are_told_apart(self):
        backend = FakeBackend([usb_mic(index=0), usb_mic(index=1)])
        devices = list_devices(backend)
        assert devices[0].id != devices[1].id
        assert devices[1].id.endswith("#2")

    def test_a_device_answers_to_its_id_name_and_label(self):
        device = list_devices(FakeBackend([usb_mic()]))[0]
        assert device.matches(device.id)
        assert device.matches("Микрофон (USB Audio Device)")
        assert device.matches(device.label)
        assert device.matches("МИКРОФОН (usb audio device)"), "case must not matter"
        assert not device.matches("")
        assert not device.matches("Webcam")


class TestEnumeration:
    """Listing, defaults, resolution and the difference between two scans."""

    def test_only_devices_with_channels_in_the_direction_are_listed(self, backend: FakeBackend):
        inputs = list_devices(backend, DeviceDirection.INPUT)
        outputs = list_devices(backend, DeviceDirection.OUTPUT)
        assert [device.name for device in inputs] == ["Микрофон (USB Audio Device)"]
        assert [device.name for device in outputs] == ["Динамики (Realtek)"]

    def test_a_machine_with_no_sound_card_lists_nothing(self):
        assert list_devices(FakeBackend([])) == ()
        assert default_device(FakeBackend([])) is None

    def test_the_system_default_is_preferred_over_the_first_device(self):
        backend = FakeBackend([webcam_mic(index=0), usb_mic(index=1)])
        assert default_device(backend).name == "Микрофон (USB Audio Device)"

    def test_the_first_device_stands_in_when_nothing_is_marked_default(self):
        backend = FakeBackend([webcam_mic(index=0)])
        assert default_device(backend).index == 0

    def test_an_empty_specification_resolves_to_the_default(self, backend: FakeBackend):
        assert resolve_device(backend).is_default

    def test_a_deviceless_machine_gets_a_russian_error_not_a_portaudio_one(self):
        with pytest.raises(AudioError) as info:
            resolve_device(FakeBackend([]))
        assert "микрофон" in info.value.user_message.lower()
        assert info.value.technical == "no input devices available"

    def test_a_missing_device_is_an_error_by_default(self, backend: FakeBackend):
        with pytest.raises(AudioError, match="not found"):
            resolve_device(backend, "Гарнитура")

    def test_a_missing_device_falls_back_when_asked(self, backend: FakeBackend):
        assert resolve_device(backend, "Гарнитура", fallback=True).is_default

    def test_find_returns_none_rather_than_raising(self, backend: FakeBackend):
        devices = list_devices(backend)
        assert find_device(devices, "нет такого") is None
        assert find_device(devices, "") is None

    def test_a_diff_reports_what_appeared_and_what_left(self):
        before = list_devices(FakeBackend([usb_mic(index=0)]))
        after = list_devices(FakeBackend([usb_mic(index=0), webcam_mic(index=1)]))
        change = diff_devices(before, after)
        assert change
        assert [device.name for device in change.added] == ["Микрофон (Webcam)"]
        assert change.removed == ()

        gone = diff_devices(after, before)
        assert [device.name for device in gone.removed] == ["Микрофон (Webcam)"]

    def test_an_unchanged_list_is_falsy(self):
        devices = list_devices(FakeBackend([usb_mic()]))
        assert not diff_devices(devices, devices)
        assert not DeviceChange()

    def test_a_new_default_counts_as_a_change(self):
        before = list_devices(FakeBackend([usb_mic(index=0), webcam_mic(index=1)]))
        after = list_devices(
            FakeBackend(
                [
                    RawDevice(
                        index=0,
                        name="Микрофон (USB Audio Device)",
                        host_api="MME",
                        max_input_channels=1,
                    ),
                    RawDevice(
                        index=1,
                        name="Микрофон (Webcam)",
                        host_api="MME",
                        max_input_channels=2,
                        default_input=True,
                    ),
                ]
            )
        )
        assert diff_devices(before, after).default_changed

    def test_the_first_enumeration_does_not_claim_the_default_changed(self):
        """Everything is "added" against an empty list, but nothing switched.

        Capture seeds its cached list during ``start`` and only diffs against
        it afterwards, so this only guards the degenerate call.
        """
        change = diff_devices((), list_devices(FakeBackend([usb_mic()])))
        assert len(change.added) == 1
        assert not change.default_changed

    def test_devices_are_describable_for_the_log(self, backend: FakeBackend):
        assert describe_devices(()) == "<none>"
        assert describe_devices(list_devices(backend)) == "Микрофон (USB Audio Device) (MME)*"


# ----------------------------------------------------------------------
# the pipeline
# ----------------------------------------------------------------------


class TestCapture:
    """The three-thread pipeline, driven by hand through the fake backend."""

    def test_starting_opens_the_default_device(self, capture: AudioCapture, backend: FakeBackend):
        assert capture.state is CaptureState.RUNNING
        assert capture.device.name == "Микрофон (USB Audio Device)"
        assert backend.stream.started

    def test_captured_audio_reaches_the_ring_buffer(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        backend.stream.feed(tone(320))
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        assert capture.stats().buffered_ms == 20

    def test_the_audio_callback_only_queues(self, backend: FakeBackend):
        """The real-time thread must not resample, lock or allocate a buffer.

        Driving ``_on_block`` without a processing thread running proves it:
        the ring buffer stays empty, and an over-full queue drops rather than
        growing without bound.
        """
        idle = AudioCapture(CaptureSettings(frame_ms=20), backend=backend)
        for _ in range(500):
            idle._on_block(tone(320), False)
        assert idle.buffer.available_frames == 0, "no processing may happen in the callback"
        stats = idle.stats()
        assert stats.dropped_blocks > 0, "the queue is bounded"
        assert len(idle._blocks) <= idle._max_blocks

    def test_an_overflow_reported_by_the_driver_is_counted(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        backend.stream.feed(tone(320), overflowed=True)
        assert wait_for(lambda: capture.stats().overflows == 1)

    def test_gain_is_applied_before_the_level_is_measured(self, backend: FakeBackend):
        instance = AudioCapture(
            CaptureSettings(frame_ms=20, gain=4.0, level_interval_ms=1), backend=backend
        )
        instance.start()
        try:
            backend.stream.feed(array("h", [2000] * 320).tobytes())
            assert wait_for(lambda: instance.buffer.available_frames == 320)
            assert samples_of(instance.buffer.read_last(frames=1)) == [8000]
        finally:
            instance.stop()

    def test_levels_are_published_and_throttled(self, backend: FakeBackend):
        seen: list[AudioLevel] = []
        instance = AudioCapture(
            CaptureSettings(frame_ms=10, level_interval_ms=50),
            backend=backend,
            callbacks=CaptureCallbacks(on_level=seen.append),
        )
        instance.start()
        try:
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                backend.stream.feed(tone(160, amplitude=16384))
                time.sleep(0.005)
            assert seen, "a loud signal must produce a level"
            assert seen[0].rms > 0.1
            # ~500 ms of audio in 10 ms blocks is ~50 blocks but at most ~10
            # notifications; the margin keeps a slow runner from failing.
            assert len(seen) <= 20, f"20 Hz throttle broken: {len(seen)} events"
        finally:
            instance.stop()

    def test_a_loud_moment_between_two_notifications_is_not_lost(self, backend: FakeBackend):
        """Peak is aggregated over the window, not sampled from the last block."""
        seen: list[AudioLevel] = []
        instance = AudioCapture(
            CaptureSettings(frame_ms=20, level_interval_ms=200),
            backend=backend,
            callbacks=CaptureCallbacks(on_level=seen.append),
        )
        instance.start()
        try:
            backend.stream.feed(array("h", [30000] * 320).tobytes())
            for _ in range(5):
                backend.stream.feed(bytes(640))
            assert wait_for(lambda: bool(seen), timeout=2.0)
            assert seen[0].peak > 0.9, "the clap must survive the aggregation window"
        finally:
            instance.stop()

    def test_muting_zeroes_the_audio_without_stopping_the_stream(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        capture.mute()
        assert capture.muted
        backend.stream.feed(array("h", [12000] * 320).tobytes())
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        assert set(samples_of(capture.buffer.read_last(frames=320))) == {0}
        assert capture.state is CaptureState.RUNNING, "a muted microphone is not a broken one"
        assert not backend.stream.closed

    def test_muting_can_release_the_device_instead(self, backend: FakeBackend):
        instance = AudioCapture(
            CaptureSettings(frame_ms=20, mute_stops_stream=True), backend=backend
        )
        instance.start()
        try:
            stream = backend.stream
            instance.mute()
            assert stream.closed, "the tray indicator only goes out if the device is released"
            instance.mute(False)
            assert backend.stream is not stream
        finally:
            instance.stop()

    def test_unmuting_restores_the_signal(self, capture: AudioCapture, backend: FakeBackend):
        capture.mute()
        capture.mute(False)
        backend.stream.feed(array("h", [12000] * 320).tobytes())
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        assert samples_of(capture.buffer.read_last(frames=1)) == [12000]

    def test_pause_releases_the_device_and_resume_takes_it_back(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        stream = backend.stream
        capture.pause("настройки")
        assert capture.state is CaptureState.PAUSED
        assert stream.closed
        capture.resume()
        assert capture.state is CaptureState.RUNNING
        assert backend.stream is not stream

    def test_pausing_keeps_the_buffered_audio(self, capture: AudioCapture, backend: FakeBackend):
        backend.stream.feed(tone(320))
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        capture.pause()
        assert capture.buffer.available_frames == 320, "pre-roll must survive a pause"

    def test_switching_device_does_not_restart_the_pipeline(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        backend.devices.append(webcam_mic(index=3))
        backend.accepts = lambda _request: True
        threads_before = (capture._processor, capture._monitor)

        device = capture.set_device("Микрофон (Webcam)")

        assert device.name == "Микрофон (Webcam)"
        assert capture.state is CaptureState.RUNNING
        assert (capture._processor, capture._monitor) == threads_before
        assert backend.stream.request.device_index == 3

    def test_switching_to_a_device_that_is_not_there_is_an_error(self, capture: AudioCapture):
        with pytest.raises(AudioError, match="not found"):
            capture.set_device("Гарнитура Bluetooth")
        assert capture.state is CaptureState.DEVICE_LOST

    def test_a_48k_stereo_driver_is_converted_to_16k_mono(self, backend: FakeBackend):
        backend.devices = [webcam_mic(index=0)]
        backend.accepts = lambda request: request.sample_rate == 48000 and request.channels == 2
        instance = AudioCapture(CaptureSettings(frame_ms=20), backend=backend)
        instance.start()
        try:
            assert backend.stream.request.sample_rate == 48000
            assert backend.stream.request.channels == 2
            # 20 ms at 48 kHz stereo -> 20 ms at 16 kHz mono.
            interleaved = array("h", [8000, 8000] * 960).tobytes()
            backend.stream.feed(interleaved)
            assert wait_for(lambda: instance.buffer.available_frames == 320)
            assert samples_of(instance.buffer.read_last(frames=1)) == [8000]
            assert instance.stats().device_sample_rate == 48000
        finally:
            instance.stop()

    def test_the_least_conversion_is_preferred(self, backend: FakeBackend):
        """A driver that takes 16 kHz mono must not be opened at 48 kHz."""
        backend.devices = [webcam_mic(index=0)]
        backend.accepts = lambda _request: True
        instance = AudioCapture(CaptureSettings(frame_ms=20), backend=backend)
        instance.start()
        try:
            assert backend.stream.request.sample_rate == TARGET_SAMPLE_RATE
            assert backend.stream.request.channels == 1
        finally:
            instance.stop()

    def test_a_busy_device_leaves_the_pipeline_looking_for_it(self, backend: FakeBackend):
        backend.open_fails = True
        instance = AudioCapture(CaptureSettings(device_poll_sec=0.05), backend=backend)
        with pytest.raises(AudioError, match="busy"):
            instance.start()
        try:
            assert instance.state is CaptureState.DEVICE_LOST
        finally:
            instance.stop()

    def test_starting_without_a_microphone_says_so_in_russian(self):
        instance = AudioCapture(backend=FakeBackend([]))
        with pytest.raises(AudioError) as info:
            instance.start()
        try:
            assert "микрофон" in info.value.user_message.lower()
        finally:
            instance.stop()

    def test_a_consumer_can_follow_the_stream_by_position(
        self, capture: AudioCapture, backend: FakeBackend
    ):
        start = capture.position
        backend.stream.feed(ramp(320))
        assert wait_for(lambda: capture.position == start + 320)
        read = capture.read_from(start)
        assert read.frames == 320
        assert read.dropped == 0

    def test_stopping_twice_is_harmless(self, backend: FakeBackend):
        instance = AudioCapture(backend=backend)
        instance.start()
        instance.stop()
        instance.stop()
        assert instance.state is CaptureState.STOPPED

    def test_reconfiguring_the_buffer_resizes_it(self, capture: AudioCapture):
        capture.configure(replace_settings(capture.settings, buffer_seconds=5.0))
        assert capture.buffer.capacity_ms == pytest.approx(5000, abs=1)

    def test_gain_can_be_changed_while_running(self, capture: AudioCapture, backend: FakeBackend):
        capture.set_gain(2.0)
        backend.stream.feed(array("h", [1000] * 320).tobytes())
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        assert samples_of(capture.buffer.read_last(frames=1)) == [2000]


def replace_settings(settings: CaptureSettings, **changes: Any) -> CaptureSettings:
    """``dataclasses.replace`` without importing it into every test."""
    from dataclasses import replace

    return replace(settings, **changes)


class TestHotPlug:
    """The USB microphone that leaves in the middle of a sentence."""

    def test_a_device_that_stops_delivering_audio_is_noticed(self, backend: FakeBackend):
        """Unplugging often does not raise; PortAudio simply stops calling back."""
        states: list[tuple[CaptureState, str]] = []
        instance = AudioCapture(
            CaptureSettings(device_poll_sec=0.02),
            backend=backend,
            callbacks=CaptureCallbacks(
                on_state=lambda state, detail: states.append((state, detail))
            ),
        )
        instance.start()
        try:
            backend.devices.clear()
            backend.stream.dead = True
            assert wait_for(lambda: instance.state is CaptureState.DEVICE_LOST)
            lost = [detail for state, detail in states if state is CaptureState.DEVICE_LOST]
            assert lost and "отключено" in lost[0]
        finally:
            instance.stop()

    def test_a_lost_device_does_not_kill_the_pipeline(self, backend: FakeBackend):
        instance = AudioCapture(CaptureSettings(device_poll_sec=0.02), backend=backend)
        instance.start()
        try:
            backend.devices.clear()
            backend.stream.dead = True
            assert wait_for(lambda: instance.state is CaptureState.DEVICE_LOST)
            assert instance._processor.is_alive()
            assert instance._monitor.is_alive()
            assert instance.stats().state is CaptureState.DEVICE_LOST
        finally:
            instance.stop()

    def test_the_device_coming_back_resumes_capture_on_its_own(self, backend: FakeBackend):
        changes: list[DeviceChange] = []
        instance = AudioCapture(
            CaptureSettings(device_poll_sec=0.02),
            backend=backend,
            callbacks=CaptureCallbacks(on_devices=changes.append),
        )
        instance.start()
        try:
            instance.devices()  # remember what was there before
            backend.devices.clear()
            backend.stream.dead = True
            assert wait_for(lambda: instance.state is CaptureState.DEVICE_LOST)

            backend.devices.append(usb_mic())
            assert wait_for(lambda: instance.state is CaptureState.RUNNING)
            assert backend.refreshes > 0, "PortAudio has to be re-scanned to see it"
            assert any(change.added for change in changes)

            backend.stream.feed(tone(320))
            assert wait_for(lambda: instance.buffer.available_frames >= 320)
        finally:
            instance.stop()

    def test_recovery_prefers_the_configured_device_over_any_device(self, backend: FakeBackend):
        """A user who chose a headset must not be moved to the webcam silently."""
        backend.devices = [usb_mic(index=0), webcam_mic(index=1)]
        backend.accepts = lambda _request: True
        instance = AudioCapture(
            CaptureSettings(device="Микрофон (Webcam)", device_poll_sec=0.02), backend=backend
        )
        instance.start()
        try:
            backend.devices = [usb_mic(index=0)]
            backend.stream.dead = True
            assert wait_for(lambda: instance.state is CaptureState.DEVICE_LOST)
            # The other microphone is present the whole time and must not be used.
            time.sleep(0.1)
            assert instance.state is CaptureState.DEVICE_LOST

            backend.devices.append(webcam_mic(index=1))
            assert wait_for(lambda: instance.state is CaptureState.RUNNING)
            assert instance.device.name == "Микрофон (Webcam)"
        finally:
            instance.stop()

    def test_enumeration_survives_a_backend_that_fails_mid_scan(self, backend: FakeBackend):
        instance = AudioCapture(CaptureSettings(device_poll_sec=0.02), backend=backend)
        instance.start()
        try:

            def explode() -> Sequence[RawDevice]:
                raise AudioError("PortAudio host error")

            backend.raw_devices = explode  # type: ignore[method-assign]
            backend.stream.dead = True
            time.sleep(0.1)
            assert instance._monitor.is_alive(), "a failing scan must not kill the monitor"
        finally:
            backend.raw_devices = FakeBackend.raw_devices.__get__(backend)  # type: ignore[method-assign]
            instance.stop()


class TestWav:
    """The debug dump behind the DevTools button."""

    def test_a_dump_is_a_readable_16k_mono_file(
        self, capture: AudioCapture, backend: FakeBackend, tmp_path: Path
    ):
        backend.stream.feed(tone(1600))
        assert wait_for(lambda: capture.buffer.available_frames == 1600)
        path = capture.dump_wav(tmp_path / "dump.wav")
        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1
            assert handle.getsampwidth() == SAMPLE_WIDTH
            assert handle.getframerate() == TARGET_SAMPLE_RATE
            assert handle.getnframes() == 1600

    def test_a_dump_can_be_limited_to_the_last_moments(
        self, capture: AudioCapture, backend: FakeBackend, tmp_path: Path
    ):
        backend.stream.feed(tone(1600))
        assert wait_for(lambda: capture.buffer.available_frames == 1600)
        path = capture.dump_wav(tmp_path / "tail.wav", ms=50)
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == 800

    def test_recording_follows_the_stream(
        self, capture: AudioCapture, backend: FakeBackend, tmp_path: Path
    ):
        backend.stream.feed(tone(320))
        assert wait_for(lambda: capture.buffer.available_frames == 320)
        capture.start_recording(tmp_path / "session.wav", pre_roll_ms=20)
        backend.stream.feed(tone(320))
        assert wait_for(lambda: capture.buffer.available_frames == 640)
        assert capture.recording is not None
        path = capture.stop_recording()
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == 640, "pre-roll plus everything captured since"
        assert capture.recording is None
        assert capture.stop_recording() is None

    def test_writing_creates_the_directory(self, tmp_path: Path):
        path = write_wav(tmp_path / "nested" / "deeper" / "out.wav", tone(160))
        assert path.is_file()

    def test_an_empty_dump_is_still_a_valid_file(self, capture: AudioCapture, tmp_path: Path):
        path = capture.dump_wav(tmp_path / "silence.wav", ms=100)
        with wave.open(str(path), "rb") as handle:
            assert handle.getnframes() == 0


# ----------------------------------------------------------------------
# the worker
# ----------------------------------------------------------------------


class FakeContext:
    """Enough of :class:`~ayris.workers.base.WorkerContext` to drive the worker.

    Built here rather than spawning a process: this file's subject is capture,
    and the process machinery already has :mod:`tests.unit.test_workers`.
    """

    def __init__(self, params: JsonObject | None = None) -> None:
        self.name = "audio"
        self.kind = "audio"
        self._params: JsonObject = dict(params or {})
        self.events: list[tuple[str, JsonObject]] = []

    @property
    def params(self) -> JsonObject:
        return self._params

    @property
    def stopping(self) -> bool:
        return False

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        self.events.append((kind, dict(payload or {})))

    def logger(self, suffix: str = "") -> Any:
        import logging

        return logging.getLogger(
            f"ayris.workers.audio.{suffix}" if suffix else "ayris.workers.audio"
        )


@pytest.fixture
def worker(backend: FakeBackend) -> Any:
    """A started :class:`AudioWorker` wired to the fake sound card."""
    context = FakeContext({"frame_ms": 20, "level_interval_ms": 10})
    instance = AudioWorker(context)  # type: ignore[arg-type]
    instance.build_backend = lambda: backend  # type: ignore[method-assign]
    instance.on_start()
    yield instance
    instance.on_stop()


class TestWorker:
    """The method surface the supervisor and DevTools call."""

    def test_starting_brings_capture_up(self, worker: AudioWorker):
        assert worker.status({})["running"]
        assert worker.status({})["sample_rate"] == TARGET_SAMPLE_RATE

    def test_a_missing_microphone_does_not_fail_the_start(self):
        """Restarting the process cannot conjure a device; capture keeps looking."""
        context = FakeContext()
        instance = AudioWorker(context)  # type: ignore[arg-type]
        instance.build_backend = lambda: FakeBackend([])  # type: ignore[method-assign]
        instance.on_start()
        try:
            assert instance.status({})["state"] == CaptureState.DEVICE_LOST.value
        finally:
            instance.on_stop()

    def test_calling_before_start_is_a_typed_error(self):
        instance = AudioWorker(FakeContext())  # type: ignore[arg-type]
        with pytest.raises(AudioError, match="not initialised"):
            instance.status({})  # status tolerates it...
            instance.read({})  # ...but reading does not

    def test_devices_are_listed_for_the_settings_window(self, worker: AudioWorker):
        answer = worker.devices({})
        assert [entry["name"] for entry in answer["devices"]] == ["Микрофон (USB Audio Device)"]
        assert answer["devices"][0]["default"] is True
        assert answer["current"] == answer["devices"][0]["id"]

    def test_reading_returns_the_last_milliseconds(self, worker: AudioWorker, backend: FakeBackend):
        backend.stream.feed(tone(1600))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 100)
        answer = worker.read({"ms": 50})
        assert answer["frames"] == 800
        assert answer["sample_rate"] == TARGET_SAMPLE_RATE
        assert len(answer["pcm"]) == 1600

    def test_reading_defaults_to_the_pre_roll(self, worker: AudioWorker, backend: FakeBackend):
        backend.stream.feed(tone(16000))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 1000)
        assert worker.read({})["frames"] == 6400

    def test_a_streaming_consumer_reads_by_position(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        start = worker.status({})["position"]
        backend.stream.feed(tone(320))
        assert wait_for(lambda: worker.status({})["position"] == start + 320)
        answer = worker.read({"position": start})
        assert answer["frames"] == 320
        assert answer["dropped"] == 0
        assert not worker.read({"position": answer["position"]})["frames"]

    def test_a_read_cannot_put_an_unbounded_blob_on_the_pipe(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        # Двенадцать блоков по секунде: столько влезает и в приёмную очередь
        # (она роняет блок после сотни), и в тридцатисекундный ринг, а мелкая
        # нарезка под нагрузкой теряла четыре пятых поданного. Дальше ждём
        # разбора, подкидывая по кадру на каждый опрос: пауза без новых блоков
        # длиннее тика монитора выглядит как отключённый микрофон, и сторож
        # закрывает поток вместе с буфером. Ожидание щедрое — через DSP надо
        # протолкнуть 160 000 отсчётов чистым питоном, и под покрытием в восемь
        # процессов это ощутимо дольше секунды.
        second = tone(TARGET_SAMPLE_RATE)
        for _ in range(12):
            backend.stream.feed(second)

        def drained() -> bool:
            backend.stream.feed(tone(1))
            return bool(worker.status({})["buffered_ms"] >= 10_000)

        assert wait_for(drained, timeout=30.0)
        assert worker.read({"ms": 999_999})["frames"] <= TARGET_SAMPLE_RATE * 10

    def test_control_methods_report_the_new_state(self, worker: AudioWorker):
        assert worker.pause({"reason": "тест"})["state"] == CaptureState.PAUSED.value
        assert worker.resume({})["running"]
        assert worker.mute({})["muted"]
        assert not worker.mute({"muted": False})["muted"]
        assert worker.stop({})["state"] == CaptureState.STOPPED.value

    def test_the_device_can_be_switched_over_the_wire(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        backend.devices.append(webcam_mic(index=5))
        backend.accepts = lambda _request: True
        answer = worker.set_device({"device": "Микрофон (Webcam)"})
        assert answer["device_name"] == "Микрофон (Webcam)"

    def test_gain_arrives_as_a_string_from_devtools_and_still_works(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        worker.set_gain({"gain": "2.0"})
        backend.stream.feed(array("h", [1000] * 320).tobytes())
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 20)
        assert worker.read({"ms": 20})["pcm"][:2] == struct.pack("<h", 2000)

    def test_nonsense_parameters_fall_back_instead_of_raising(self, worker: AudioWorker):
        worker.set_gain({"gain": "громко"})
        worker.set_gain({"gain": None})
        assert worker.status({})["running"]

    def test_reconfiguring_does_not_drop_the_stream(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        stream = backend.stream
        worker.on_configure({"frame_ms": 20, "gain": 3.0})
        assert backend.stream is stream
        backend.stream.feed(array("h", [1000] * 320).tobytes())
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 20)
        assert worker.read({"ms": 20})["pcm"][:2] == struct.pack("<h", 3000)

    def test_a_level_is_emitted_as_a_worker_event(self, worker: AudioWorker, backend: FakeBackend):
        for _ in range(20):
            backend.stream.feed(tone(320, amplitude=16384))
            time.sleep(0.005)
        assert wait_for(lambda: any(kind == "level" for kind, _ in worker.context.events))
        payload = next(payload for kind, payload in worker.context.events if kind == "level")
        assert payload["rms"] > 0.1
        # is_speech and gate_db joined in task 08: the overlay animates on the
        # first, and the level meter draws the second as the line a phrase has
        # to cross.
        assert set(payload) == {"rms", "peak", "clipped", "is_speech", "gate_db"}

    def test_a_dump_lands_in_the_profile_cache_by_default(
        self, worker: AudioWorker, backend: FakeBackend, profile_paths: Any
    ):
        backend.stream.feed(tone(1600))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 100)
        answer = worker.dump_wav({"ms": 100})
        written = answer["path"]
        assert str(profile_paths.cache_dir) in written
        assert answer["bytes"] > 0
        # A second dump must not overwrite the first.
        assert worker.dump_wav({"ms": 100})["path"] != written

    def test_recording_can_be_started_and_stopped(
        self, worker: AudioWorker, backend: FakeBackend, tmp_path: Path
    ):
        target = tmp_path / "wake_samples.wav"
        assert worker.record({"path": str(target)})["recording"]
        backend.stream.feed(tone(320))
        assert wait_for(lambda: worker.status({})["recording"])
        answer = worker.record({"enabled": False})
        assert not answer["recording"]
        assert answer["path"] == str(target)
        assert target.is_file()


class TestDetection:
    """Task 08's methods, which need a running capture to say anything.

    The detection logic itself is covered in :mod:`tests.unit.test_vad`; what is
    tested here is the wiring — that captured audio reaches the segmenter, that
    a finished phrase becomes an event and stays available for the recogniser,
    and that calibration reads backwards from the ring buffer instead of
    blocking the pipe for eight seconds.
    """

    def test_detection_reports_itself_to_the_settings_window(self, worker: AudioWorker):
        answer = worker.vad({})
        assert answer["running"]
        assert answer["engine"] in {"webrtc", "energy"}
        assert answer["gate_db"] < 0.0
        assert answer["state"] == "idle"
        assert answer["denoise"]["mode"] in {"off", "rnnoise", "spectral"}

    def test_a_phrase_arrives_as_a_pair_of_events(self, worker: AudioWorker, backend: FakeBackend):
        """The overlay needs both halves: one to start listening, one to stop."""
        _speak(worker, backend)
        kinds = [kind for kind, _ in worker.context.events]
        assert "speech_started" in kinds
        assert "speech_ended" in kinds
        payload = next(load for kind, load in worker.context.events if kind == "speech_ended")
        assert payload["accepted"]
        assert payload["reason"] == "silence"
        assert payload["duration_ms"] > 0
        assert "pcm" not in payload, "звук не должен ехать по трубе в событии"

    def test_the_phrase_audio_is_fetched_separately_and_consumed_once(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        """A recogniser that restarts must not pick up a phrase from before it."""
        _speak(worker, backend)
        answer = worker.segment({"keep": True})
        assert answer["available"]
        assert answer["duration_ms"] > 0
        assert len(answer["pcm"]) == answer["frames"] * SAMPLE_WIDTH
        assert worker.segment({})["available"]
        assert not worker.segment({})["available"]

    def test_nothing_detected_yet_is_an_answer_not_an_error(self, worker: AudioWorker):
        assert not worker.segment({})["available"]

    def test_calibration_measures_the_room_without_blocking(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        """Every stage returns straight away: the audio is already buffered."""
        backend.stream.feed(bytes(TARGET_SAMPLE_RATE * SAMPLE_WIDTH))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 900)
        answer = worker.calibrate({"stage": "silence", "seconds": 0.5})
        assert answer["ready"]
        assert answer["recommended"]["noise_floor_db"] < 0.0
        assert not answer["phrase"]["checked"]

    def test_the_phrase_stage_needs_the_silence_stage_first(self, worker: AudioWorker):
        """Guessing the floor would make the whole report a fiction."""
        with pytest.raises(AudioError, match="silence stage"):
            worker.calibrate({"stage": "phrase"})

    def test_a_calibration_can_be_restarted_after_a_cough(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        backend.stream.feed(bytes(TARGET_SAMPLE_RATE * SAMPLE_WIDTH))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 900)
        worker.calibrate({"stage": "silence", "seconds": 0.5})
        assert not worker.calibrate({"stage": "reset"})["ready"]
        with pytest.raises(AudioError, match="silence stage"):
            worker.calibrate({"stage": "report"})

    def test_an_unknown_stage_is_refused(self, worker: AudioWorker, backend: FakeBackend):
        backend.stream.feed(bytes(TARGET_SAMPLE_RATE * SAMPLE_WIDTH))
        assert wait_for(lambda: worker.status({})["buffered_ms"] >= 900)
        worker.calibrate({"stage": "silence", "seconds": 0.5})
        with pytest.raises(AudioError, match="stage"):
            worker.calibrate({"stage": "потом"})

    def test_stopping_capture_hands_over_the_phrase_in_progress(
        self, worker: AudioWorker, backend: FakeBackend
    ):
        """Muting mid-sentence should not silently eat what was already said."""

        def collected_ms() -> int:
            answer = worker.vad({})
            return int(answer["speech_frames"]) * int(answer["frame_ms"])

        backend.stream.feed(_phrase()[: TARGET_SAMPLE_RATE * SAMPLE_WIDTH])
        assert wait_for(lambda: worker.vad({})["state"] == "speech")
        # The state flips after start_frames — 60 ms — and a phrase that short is
        # rejected as too_short whatever closed it. So wait until there is a
        # phrase worth handing over, otherwise the reason under test depends on
        # who wins the race between this thread and the capture thread.
        floor = int(worker.vad({})["min_speech_ms"]) + 3 * int(worker.vad({})["frame_ms"])
        assert wait_for(lambda: collected_ms() >= floor)
        worker.pause({"reason": "тест"})
        assert wait_for(lambda: any(kind == "speech_ended" for kind, _ in worker.context.events))
        payload = next(load for kind, load in worker.context.events if kind == "speech_ended")
        assert payload["reason"] == "flush"


def _phrase() -> bytes:
    """The committed phrase fixture, as raw PCM."""
    path = Path(__file__).resolve().parents[1] / "fixtures" / "audio" / "phrase.wav"
    with wave.open(str(path), "rb") as handle:
        return handle.readframes(handle.getnframes())


def _speak(worker: AudioWorker, backend: FakeBackend) -> None:
    """Play the phrase fixture through the fake microphone and wait for the end."""
    backend.stream.feed(_phrase())
    assert wait_for(lambda: any(kind == "speech_ended" for kind, _ in worker.context.events))


class TestBusTranslation:
    """What the worker's events become in the main process."""

    def test_a_level_becomes_the_overlay_event(self):
        event = translate_audio_event("level", {"rms": 0.4, "peak": 0.9, "clipped": False})
        assert isinstance(event, AudioLevelChanged)
        assert event.rms == pytest.approx(0.4)
        assert event.peak == pytest.approx(0.9)

    def test_losing_the_device_warns_the_user_in_russian(self):
        event = translate_audio_event(
            "state", {"state": "device_lost", "detail": "Устройство «Микрофон» отключено."}
        )
        assert isinstance(event, NotificationRequested)
        assert event.level == "warning"
        assert "Микрофон" in event.message

    def test_an_ordinary_start_is_not_worth_a_notification(self):
        assert translate_audio_event("state", {"state": "running", "detail": ""}) is None
        assert translate_audio_event("state", {"state": "paused", "detail": "настройки"}) is None

    def test_recovery_is_announced(self):
        event = translate_audio_event("state", {"state": "running", "detail": "Снова доступно."})
        assert isinstance(event, NotificationRequested)
        assert event.level == "info"

    def test_a_new_microphone_is_announced_and_a_lost_one_is_not(self):
        added = translate_audio_event("devices", {"added": ["Гарнитура (USB)"], "removed": []})
        assert isinstance(added, NotificationRequested)
        assert "Гарнитура (USB)" in added.message
        assert translate_audio_event("devices", {"added": [], "removed": ["Гарнитура"]}) is None

    def test_an_unknown_event_is_dropped_rather_than_crashing_the_bus(self):
        assert translate_audio_event("what", {}) is None
        assert translate_audio_event("devices", {}) is None

    def test_the_registry_finds_the_translator_by_worker_type(self):
        assert event_translator(WorkerKind.AUDIO) is translate_audio_event
        assert event_translator("test") is None
