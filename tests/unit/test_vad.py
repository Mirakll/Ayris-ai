"""Task 08: speech detection — VAD, phrase segmentation, denoising, calibration.

Everything here runs on the WAV fixtures in ``tests/fixtures/audio``, which are
synthesised by the ``make_fixtures.py`` sitting next to them. Nothing opens a
microphone: the one test that needs a live device carries the ``hardware``
marker and CI skips it.

The fixtures are the point. A segmenter is easy to write so that it passes on
hand-built frame sequences and then clips the first syllable off every real
phrase, because the interesting behaviour — pre-roll, an onset that has to be
confirmed, a breath in the middle of a sentence — only shows up on audio with a
shape. So the assertions here are about milliseconds of a known waveform, not
about frames of a mock.

The worker methods that need a running capture (``vad``, ``segment``,
``calibrate``) are tested in :mod:`tests.unit.test_audio_capture`, where the
fake sound card lives; this file covers the parts of the worker that are pure
functions of their parameters.

Groups:

* :class:`TestFixtures` — the WAV files really are what the rest assumes.
* :class:`TestVadSettings` — validation, and the threshold-to-gate arithmetic.
* :class:`TestVad` — both engines on real waveforms, framing, the stream.
* :class:`TestSegmentation` — the acceptance criteria of the task.
* :class:`TestSegmenterControl` — stats, reset, reconfiguration, forced close.
* :class:`TestDenoise` — passthrough, the gate, the RNNoise fallback, timing.
* :class:`TestRnnoiseLookup` — where the library is searched for, and what a
  missing one costs.
* :class:`TestRnnoise` — the RNNoise path itself; skipped without the library.
* :class:`TestCalibration` — noise profile, verdicts, recommendations, report.
* :class:`TestWorkerWiring` — parameters in, bus events out.
* :class:`TestLiveMicrophone` — the one test that wants a real device.
"""

from __future__ import annotations

import importlib.util
import json
import math
import wave
from array import array
from pathlib import Path

import pytest

from ayris.audio.calibration import (
    CALIBRATION_PHRASE,
    MAX_GAIN,
    MIN_GAIN,
    CalibrationReport,
    Verdict,
    analyse_noise,
    analyse_phrase,
    calibrate_pcm,
    recommend,
    run_calibration,
)
from ayris.audio.capture import TARGET_SAMPLE_RATE, pcm_level
from ayris.audio.denoise import (
    RNNOISE_LIB_ENV,
    DenoiseEngine,
    DenoiseMode,
    DenoiseSettings,
    DenoiseStream,
    NoiseGate,
    Passthrough,
    RnnoiseDenoiser,
    _bundled_libraries,
    create_denoiser,
    denoise_pcm,
    rnnoise_available,
    rnnoise_library,
)
from ayris.audio.segmenter import (
    EndReason,
    Segmenter,
    SegmenterCallbacks,
    SegmenterSettings,
    SegmentState,
    SpeechSegment,
    SpeechStart,
    segment_pcm,
)
from ayris.audio.vad import (
    VAD_FRAME_MS,
    EnergyVad,
    FrameSplitter,
    VadEngine,
    VadFrame,
    VadSettings,
    VadStream,
    WebRtcVad,
    create_vad,
    frames_of,
    webrtcvad_available,
)
from ayris.core.errors import AudioError
from ayris.workers.audio_worker import (
    _denoise_settings_from_params,
    _segmenter_settings_from_params,
    _vad_settings_from_params,
    translate_audio_event,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: How long each fixture is, in milliseconds, by construction.
DURATIONS = {
    "silence.wav": 2000,
    "noise.wav": 2000,
    "phrase.wav": 2200,
    "phrase_with_pause.wav": 2700,
    "two_utterances.wav": 3500,
    "click.wav": 1360,
    "phrase_in_noise.wav": 2200,
    "clipped.wav": 2000,
    "quiet_phrase.wav": 2200,
}


def wav(name: str) -> bytes:
    """Read a fixture as raw PCM, checking it is the format the code expects."""
    with wave.open(str(FIXTURES / name), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == TARGET_SAMPLE_RATE
        return handle.readframes(handle.getnframes())


def tone(ms: int, amplitude: int = 8000, hz: float = 440.0) -> bytes:
    """A sine, for the cases where a waveform's shape does not matter."""
    count = TARGET_SAMPLE_RATE * ms // 1000
    samples = array(
        "h",
        (
            int(amplitude * math.sin(2.0 * math.pi * hz * index / TARGET_SAMPLE_RATE))
            for index in range(count)
        ),
    )
    return samples.tobytes()


def only(segments: tuple[SpeechSegment, ...]) -> SpeechSegment:
    """Assert there is exactly one segment and return it."""
    assert len(segments) == 1, [
        (segment.frame_index, segment.duration_ms, segment.reason.value) for segment in segments
    ]
    return segments[0]


def _starts_at(segment: SpeechSegment, frame_ms: int = 20) -> int:
    """Where a segment begins in the recording, in milliseconds.

    ``frame_index`` counts detector frames while ``frames`` counts samples, so
    the two are not comparable without this.
    """
    return segment.frame_index * frame_ms


def _ends_at(segment: SpeechSegment, frame_ms: int = 20) -> int:
    """Where a segment ends in the recording, in milliseconds."""
    return _starts_at(segment, frame_ms) + segment.duration_ms


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


class TestFixtures:
    """The committed WAVs are the premise of every other test in this file."""

    @pytest.mark.parametrize(("name", "duration_ms"), sorted(DURATIONS.items()))
    def test_every_fixture_has_the_expected_length(self, name: str, duration_ms: int):
        pcm = wav(name)
        assert len(pcm) * 1000 // (TARGET_SAMPLE_RATE * 2) == duration_ms

    def test_the_fixtures_are_small_enough_to_live_in_git(self):
        """A repository is not an audio archive.

        The budget covers every task's fixtures together, not this file's, so it
        moves up when a task adds a few - task 10 added the three ``stt_*``
        files.  A megabyte and a half is still nothing next to a single model.
        """
        total = sum(path.stat().st_size for path in FIXTURES.glob("*.wav"))
        assert total < 1_500_000, f"фикстуры распухли до {total} байт"

    def test_the_generator_is_committed_next_to_its_output(self):
        """Otherwise nobody can tell what the bytes are, let alone regenerate them."""
        assert (FIXTURES / "make_fixtures.py").is_file()

    def test_silence_is_room_tone_and_not_digital_zero(self):
        """A file of zeros would let a broken noise-floor estimate pass."""
        level = pcm_level(wav("silence.wav"))
        assert level.rms > 0.0
        assert level.rms_db < -60.0

    def test_the_loud_fixtures_differ_by_level_not_by_content(self):
        quiet = pcm_level(wav("quiet_phrase.wav"))
        normal = pcm_level(wav("phrase.wav"))
        clipped = pcm_level(wav("clipped.wav"))
        assert quiet.rms_db < normal.rms_db < clipped.rms_db
        assert clipped.clipped
        assert not normal.clipped


# ----------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------


class TestVadSettings:
    """Refusals happen at construction, where the value came from a file."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sample_rate": 44100}, "44100"),
            ({"frame_ms": 25}, "frame"),
            ({"aggressiveness": 9}, "aggressiveness"),
            ({"threshold": 5.0}, "threshold"),
        ],
    )
    def test_impossible_settings_are_refused(self, kwargs: dict[str, object], message: str):
        with pytest.raises(AudioError, match=message):
            VadSettings(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("frame_ms", VAD_FRAME_MS)
    def test_frame_size_follows_the_sample_rate(self, frame_ms: int):
        settings = VadSettings(frame_ms=frame_ms)
        assert settings.frame_samples == TARGET_SAMPLE_RATE * frame_ms // 1000
        assert settings.frame_bytes == settings.frame_samples * 2

    def test_the_threshold_slides_the_gate_above_the_noise_floor(self):
        """One user-facing 0..1 knob, expressed as decibels over the measured floor."""
        low = VadSettings(threshold=0.0, noise_floor_db=-60.0)
        high = VadSettings(threshold=1.0, noise_floor_db=-60.0)
        assert low.gate_db < high.gate_db
        assert low.gate_db == pytest.approx(-57.0)
        assert high.gate_db == pytest.approx(-42.0)

    def test_the_gate_never_drops_below_the_dynamic_range(self):
        """A floor at digital silence must not put the gate at minus infinity."""
        assert VadSettings(threshold=0.0, noise_floor_db=-200.0).gate_db > -200.0


# ----------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------


class TestVad:
    """The classifiers, on audio rather than on hand-built frame sequences."""

    def test_webrtcvad_is_installed(self):
        """It is pinned in requirements-ci.txt; if it vanished, say so here."""
        assert webrtcvad_available(), "webrtcvad-wheels не установлен — проверь окружение"

    @pytest.mark.parametrize("engine", [VadEngine.WEBRTC, VadEngine.ENERGY])
    def test_speech_is_speech_and_room_tone_is_not(self, engine: VadEngine):
        """The property the fixtures exist to guarantee, for both detectors."""
        settings = VadSettings(engine=engine)
        speech = [frame.speech for frame in _classify("phrase.wav", settings)]
        quiet = [frame.speech for frame in _classify("silence.wav", settings)]
        assert sum(speech) == 45, "речь в phrase.wav длится ровно 900 мс"
        assert not any(quiet)

    def test_the_energy_detector_needs_no_compiled_extension(self):
        """The fallback has to work on a machine where the wheel would not build."""
        vad = EnergyVad(VadSettings(noise_floor_db=-70.0))
        assert vad.is_speech(tone(20))
        assert not vad.is_speech(bytes(VadSettings().frame_bytes))

    def test_aggressiveness_only_narrows_what_counts_as_speech(self):
        """Higher settings may reject more, never accept more."""
        counts = [
            sum(frame.speech for frame in _classify("phrase_in_noise.wav", VadSettings(**kwargs)))
            for kwargs in ({"aggressiveness": level} for level in range(4))
        ]
        assert counts == sorted(counts, reverse=True)

    def test_the_engine_is_chosen_automatically_and_can_be_forced(self):
        assert isinstance(create_vad(VadSettings(engine=VadEngine.ENERGY)), EnergyVad)
        assert isinstance(create_vad(VadSettings(engine=VadEngine.WEBRTC)), WebRtcVad)
        assert isinstance(create_vad(VadSettings()), WebRtcVad)

    def test_a_frame_of_the_wrong_size_is_a_typed_error(self):
        """Handing webrtcvad a short frame is a segfault-adjacent mistake."""
        with pytest.raises(AudioError, match="frame"):
            create_vad().is_speech(bytes(100))

    def test_framing_is_independent_of_how_the_audio_arrives(self):
        """Capture delivers whatever the driver hands it; the detector must not care."""
        pcm = wav("phrase.wav")
        whole = [frame.speech for frame in _classify_pcm(pcm)]
        stream = VadStream(VadSettings())
        piecemeal: list[bool] = []
        for start in range(0, len(pcm), 777):
            piecemeal.extend(frame.speech for frame in stream.push(pcm[start : start + 777]))
        piecemeal.extend(frame.speech for frame in stream.flush())
        assert piecemeal[: len(whole)] == whole

    def test_the_splitter_holds_a_partial_frame_and_pads_it_on_flush(self):
        splitter = FrameSplitter(640)
        assert len(splitter.push(bytes(1000))) == 1
        assert splitter.pending == 360
        assert len(splitter.flush()) == 640
        assert not splitter.pending

    def test_the_stream_reports_what_the_meter_needs(self):
        stream = VadStream(VadSettings())
        stream.push(wav("phrase.wav"))
        assert stream.engine == "webrtc"
        assert stream.index == 110
        assert stream.gate_db < 0.0
        assert not stream.is_speech, "фикстура заканчивается тишиной"

    def test_frames_of_is_the_one_shot_form_of_the_stream(self):
        assert len(list(frames_of(wav("phrase.wav")))) == 110

    def test_reconfiguring_keeps_the_stream_usable(self):
        """The absolute index keeps counting, so positions handed out stay valid."""
        stream = VadStream(VadSettings())
        stream.push(wav("phrase.wav")[:6400])
        before = stream.index
        stream.configure(VadSettings(frame_ms=30, threshold=0.2))
        assert stream.settings.frame_ms == 30
        assert stream.index == before
        assert stream.push(wav("phrase.wav"))
        assert stream.index > before


def _classify(name: str, settings: VadSettings | None = None) -> list[VadFrame]:
    """Every frame of a fixture, classified."""
    return _classify_pcm(wav(name), settings)


def _classify_pcm(pcm: bytes, settings: VadSettings | None = None) -> list[VadFrame]:
    stream = VadStream(settings or VadSettings())
    return [*stream.push(pcm), *stream.flush()]


# ----------------------------------------------------------------------
# segmentation
# ----------------------------------------------------------------------


class TestSegmentation:
    """The acceptance criteria of task 08, one test each."""

    def test_a_phrase_comes_back_whole_with_its_pre_roll(self):
        """«айрис открой браузер» must arrive as one segment, start intact.

        The fixture puts 400 ms of room tone before the voice, and the onset is
        only declared three frames in. Without pre-roll the segment would start
        60 ms after the speech does — which is exactly how a recogniser loses
        the first syllable of a wake word.
        """
        segment = only(segment_pcm(wav("phrase.wav")))
        assert segment.accepted
        assert segment.reason is EndReason.SILENCE
        assert segment.speech_ms == 900
        assert segment.pre_roll_ms == 340
        assert segment.frame_index == 3, "сегмент начинается за 340 мс до речи"
        assert segment.duration_ms == 1440
        assert segment.sample_rate == TARGET_SAMPLE_RATE
        assert len(segment.pcm) == segment.frames * 2

    def test_the_pre_roll_carries_audio_from_before_the_onset(self):
        """Not just leading zeros: the ring buffer's contents have to be in there."""
        segment = only(segment_pcm(wav("phrase.wav")))
        pre_roll_bytes = segment.pre_roll_ms * TARGET_SAMPLE_RATE * 2 // 1000
        assert pcm_level(segment.pcm[:pre_roll_bytes]).rms > 0.0

    def test_a_pause_inside_a_sentence_does_not_split_the_phrase(self):
        """300 ms of breath is under the 700 ms end-of-phrase threshold."""
        segment = only(segment_pcm(wav("phrase_with_pause.wav")))
        assert segment.accepted
        assert segment.speech_ms == 1100, "обе половины фразы попали в один сегмент"
        assert segment.duration_ms == 1940

    def test_a_real_gap_between_phrases_does_split_them(self):
        first, second = segment_pcm(wav("two_utterances.wav"))
        assert first.accepted and second.accepted
        assert first.speech_ms == second.speech_ms == 600
        assert _starts_at(second) > _ends_at(first), "второй сегмент начинается после первого"

    def test_a_click_is_rejected_instead_of_being_recognised(self):
        """A 60 ms burst confirms an onset and then has to be thrown away."""
        segment = only(segment_pcm(wav("click.wav")))
        assert not segment.accepted
        assert segment.reason is EndReason.TOO_SHORT
        assert segment.speech_ms == 60

    def test_silence_produces_nothing_at_all(self):
        assert not segment_pcm(wav("silence.wav"))
        assert not segment_pcm(wav("noise.wav"))

    def test_speech_over_noise_is_still_one_phrase(self):
        """Fan noise at -36 dBFS must not break the boundaries."""
        segment = only(segment_pcm(wav("phrase_in_noise.wav")))
        assert segment.accepted
        assert segment.speech_ms == 900

    @pytest.mark.parametrize("engine", [VadEngine.WEBRTC, VadEngine.ENERGY])
    def test_both_engines_agree_on_the_fixtures(self, engine: VadEngine):
        """The energy fallback is not a second-class path; on clean audio it ties."""
        for name in ("phrase.wav", "phrase_with_pause.wav", "two_utterances.wav", "silence.wav"):
            expected = segment_pcm(wav(name))
            actual = segment_pcm(wav(name), vad_settings=VadSettings(engine=engine))
            assert [segment.duration_ms for segment in actual] == [
                segment.duration_ms for segment in expected
            ], name

    def test_a_phrase_too_quiet_for_the_default_gate_is_heard_once_calibrated(self):
        """The whole reason calibration exists, in one assertion."""
        pcm = wav("quiet_phrase.wav")
        assert not segment_pcm(pcm), "с порогом по умолчанию тихая речь не проходит"
        report = calibrate_pcm(wav("silence.wav"), pcm)
        assert only(segment_pcm(pcm, vad_settings=report.recommended.vad_settings())).accepted


class TestSegmenterControl:
    """State, statistics and the ways a phrase gets closed from outside."""

    def test_the_state_machine_reports_where_it_is(self):
        segmenter = Segmenter()
        assert segmenter.state is SegmentState.IDLE
        segmenter.push(wav("phrase.wav")[: 16000 * 2])
        assert segmenter.state is SegmentState.SPEECH
        assert segmenter.is_speech
        assert segmenter.stats.current_ms > 0

    def test_stopping_mid_phrase_hands_over_what_was_collected(self):
        """Muting the microphone should not silently eat the audio."""
        segmenter = Segmenter()
        segmenter.push(wav("phrase.wav")[: 16000 * 2])
        segment = only(segmenter.flush())
        assert segment.reason is EndReason.FLUSH
        assert segment.accepted
        assert segmenter.state is SegmentState.IDLE

    def test_flushing_an_idle_segmenter_produces_nothing(self):
        assert not Segmenter().flush()
        assert not Segmenter().push(wav("silence.wav"))

    def test_an_endless_phrase_is_cut_and_collection_continues(self):
        """Somebody leaves the microphone next to a television."""
        settings = SegmenterSettings(max_utterance_ms=600, min_speech_ms=200)
        segments = segment_pcm(wav("phrase_with_pause.wav"), settings=settings)
        cut = [segment for segment in segments if segment.reason is EndReason.MAX_LENGTH]
        assert len(cut) >= 2
        assert all(segment.duration_ms == 600 for segment in cut)
        assert _starts_at(cut[1]) == _ends_at(cut[0]), "между отрезками нет разрыва"

    def test_cutting_never_emits_an_empty_segment(self):
        """The trailing silence after the last cut is not a phrase."""
        settings = SegmenterSettings(max_utterance_ms=600, min_speech_ms=200)
        segments = segment_pcm(wav("phrase_with_pause.wav"), settings=settings)
        assert all(segment.frames for segment in segments)

    def test_statistics_add_up(self):
        segmenter = Segmenter()
        segmenter.push(wav("two_utterances.wav"))
        segmenter.push(wav("click.wav"))
        segmenter.flush()
        stats = segmenter.stats
        assert stats.segments == 2
        assert stats.rejected == 1
        assert not stats.truncated
        assert 0.0 < stats.speech_ratio < 1.0

    def test_resetting_forgets_the_phrase_and_the_counters(self):
        segmenter = Segmenter()
        segmenter.push(wav("phrase.wav"))
        segmenter.reset()
        assert segmenter.state is SegmentState.IDLE
        assert not segmenter.stats.segments
        assert not segmenter.stats.frames

    def test_reconfiguring_closes_the_phrase_rather_than_mixing_thresholds(self):
        """Half a phrase under one silence window and half under another is nonsense."""
        closed: list[SpeechSegment] = []
        segmenter = Segmenter(callbacks=SegmenterCallbacks(on_speech_ended=closed.append))
        segmenter.push(wav("phrase.wav")[: 16000 * 2])
        segmenter.configure(SegmenterSettings(silence_ms=300))
        assert len(closed) == 1
        assert closed[0].reason is EndReason.FLUSH
        assert segmenter.state is SegmentState.IDLE

    def test_the_callbacks_fire_in_the_order_the_overlay_expects(self):
        """Onset first, then the finished phrase — never one without the other."""
        events: list[str] = []
        callbacks = SegmenterCallbacks(
            on_speech_started=lambda start: events.append(f"start:{start.frame_index}"),
            on_speech_ended=lambda segment: events.append(f"end:{segment.reason.value}"),
        )
        segmenter = Segmenter(callbacks=callbacks)
        segmenter.push(wav("two_utterances.wav"))
        segmenter.flush()
        assert events == ["start:0", "end:silence", "start:83", "end:silence"]

    def test_a_rejected_click_still_closes_the_onset_it_opened(self):
        """Otherwise the overlay sits in its listening animation forever."""
        events: list[str] = []
        callbacks = SegmenterCallbacks(
            on_speech_started=lambda _start: events.append("start"),
            on_speech_ended=lambda segment: events.append(segment.reason.value),
        )
        Segmenter(callbacks=callbacks).push(wav("click.wav"))
        assert events == ["start", "too_short"]

    def test_the_onset_needs_several_frames_in_a_row(self):
        """One stray frame of noise is not somebody starting to talk."""
        starts: list[SpeechStart] = []
        callbacks = SegmenterCallbacks(on_speech_started=starts.append)
        segmenter = Segmenter(
            SegmenterSettings(start_frames=10),
            callbacks=callbacks,
        )
        segmenter.push(wav("click.wav"))
        assert not starts, "60 мс — это три кадра, порог не взят"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"start_frames": 0},
            {"silence_ms": 0},
            {"min_speech_ms": -1},
            {"pre_roll_ms": -5},
            {"tail_ms": -1},
            {"max_utterance_ms": 100},
        ],
    )
    def test_settings_that_could_never_emit_a_segment_are_refused(self, kwargs: dict[str, int]):
        with pytest.raises(AudioError):
            SegmenterSettings(**kwargs)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# denoising
# ----------------------------------------------------------------------


class TestDenoise:
    """Suppression, and — mostly — its behaviour when RNNoise is not there."""

    def test_switching_it_off_returns_the_bytes_untouched(self):
        pcm = wav("phrase.wav")
        assert denoise_pcm(pcm, DenoiseSettings(mode=DenoiseMode.OFF)) == pcm

    def test_off_costs_nothing_and_adds_no_latency(self):
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.OFF))
        assert isinstance(stream.denoiser, Passthrough)
        assert stream.engine is DenoiseEngine.NONE
        assert not stream.enabled
        assert not stream.latency_ms

    def test_rnnoise_falls_back_to_the_gate_instead_of_failing(self):
        """CI has no RNNoise, and neither does a fresh Windows install."""
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.RNNOISE))
        if rnnoise_available():
            assert stream.engine is DenoiseEngine.RNNOISE
            assert not stream.stats.fallback
        else:
            assert stream.engine is DenoiseEngine.GATE
            assert isinstance(stream.denoiser, NoiseGate)
            stream.push(wav("noise.wav"))
            assert stream.stats.fallback, "UI должен показать, что работает запасной путь"

    def test_the_gate_pushes_the_noise_floor_down(self):
        settings = DenoiseSettings(mode=DenoiseMode.SPECTRAL, noise_floor_db=-36.0)
        before = pcm_level(wav("noise.wav")).rms_db
        after = pcm_level(denoise_pcm(wav("noise.wav"), settings)).rms_db
        assert before - after > 6.0

    def test_the_gate_leaves_the_phrase_alone(self):
        """Suppression that eats the speech is worse than no suppression."""
        settings = DenoiseSettings(mode=DenoiseMode.SPECTRAL, noise_floor_db=-36.0)
        cleaned = denoise_pcm(wav("phrase_in_noise.wav"), settings)
        segment = only(segment_pcm(cleaned))
        assert segment.accepted
        assert segment.speech_ms == 900

    def test_denoising_does_not_change_the_length_of_the_audio(self):
        for mode in DenoiseMode:
            pcm = wav("phrase_in_noise.wav")
            assert len(denoise_pcm(pcm, DenoiseSettings(mode=mode))) == len(pcm)

    def test_the_stream_reports_the_cost_the_settings_window_shows(self):
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.SPECTRAL))
        stream.push(wav("phrase.wav"))
        stats = stream.stats
        assert stats.frames == 110
        assert stats.latency_ms == 20.0, "задержка равна одному кадру"
        assert stats.max_ms >= stats.avg_ms > 0.0
        assert stats.realtime_factor < 0.5, "иначе подавление не успевает за захватом"

    def test_a_partial_frame_waits_for_the_rest(self):
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.SPECTRAL))
        assert not stream.push(bytes(100))
        assert len(stream.flush()) == stream.settings.frame_bytes

    def test_the_engine_can_be_swapped_while_running(self):
        """Task 08 asks for a toggle that works without restarting capture."""
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.SPECTRAL))
        stream.push(wav("phrase.wav"))
        stream.configure(DenoiseSettings(mode=DenoiseMode.OFF))
        assert stream.engine is DenoiseEngine.NONE
        assert stream.push(wav("phrase.wav")) == wav("phrase.wav")

    def test_creating_a_denoiser_maps_mode_to_engine(self):
        assert create_denoiser(DenoiseSettings(mode=DenoiseMode.OFF)).engine is DenoiseEngine.NONE
        assert (
            create_denoiser(DenoiseSettings(mode=DenoiseMode.SPECTRAL)).engine is DenoiseEngine.GATE
        )

    def test_the_reduction_is_configurable(self):
        strong = DenoiseSettings(mode=DenoiseMode.SPECTRAL, noise_floor_db=-36.0, reduction_db=24.0)
        weak = DenoiseSettings(mode=DenoiseMode.SPECTRAL, noise_floor_db=-36.0, reduction_db=3.0)
        assert (
            pcm_level(denoise_pcm(wav("noise.wav"), strong)).rms_db
            < pcm_level(denoise_pcm(wav("noise.wav"), weak)).rms_db
        )


# ----------------------------------------------------------------------
# rnnoise, when the machine actually has it
# ----------------------------------------------------------------------


class TestRnnoiseLookup:
    """Where the shared library is looked for.

    This is what decides whether the group below runs at all, so it is tested
    apart from it: until ``pyrnnoise`` went into ``requirements-ci-nodeps.txt``
    the search only knew plain library names, the loader never looks inside
    site-packages, and every RNNoise assertion was skipped on all three runners.
    """

    def test_the_wheel_ships_the_library_and_it_is_found(self):
        """``pip install pyrnnoise`` must be enough — no environment variable."""
        if importlib.util.find_spec("pyrnnoise") is None:  # pragma: no cover - it is pinned
            pytest.skip("pyrnnoise не установлен")
        found = _bundled_libraries()
        assert found, "колесо кладёт библиотеку рядом со своим кодом"
        for path in found:
            assert path.is_file()
            assert path.parent.name == "pyrnnoise"

    def test_it_loads_without_the_environment_override(self, monkeypatch):
        if importlib.util.find_spec("pyrnnoise") is None:  # pragma: no cover - it is pinned
            pytest.skip("pyrnnoise не установлен")
        monkeypatch.delenv(RNNOISE_LIB_ENV, raising=False)
        assert rnnoise_library() is not None

    def test_a_wrong_override_does_not_hide_the_wheel(self, monkeypatch):
        """A stale path in the variable must not switch RNNoise off."""
        if importlib.util.find_spec("pyrnnoise") is None:  # pragma: no cover - it is pinned
            pytest.skip("pyrnnoise не установлен")
        monkeypatch.setenv(RNNOISE_LIB_ENV, "/no/such/librnnoise.so")
        assert rnnoise_library() is not None

    def test_without_the_package_the_search_is_empty_rather_than_loud(self, monkeypatch):
        def absent(name: str) -> None:
            return None

        monkeypatch.setattr(importlib.util, "find_spec", absent)
        assert _bundled_libraries() == []

    def test_a_package_without_a_directory_is_ignored(self, monkeypatch):
        """A namespace package, or a module that is a single file."""
        spec = importlib.util.spec_from_loader("pyrnnoise", loader=None)

        def flat(name: str) -> object:
            return spec

        monkeypatch.setattr(importlib.util, "find_spec", flat)
        assert _bundled_libraries() == []

    def test_a_broken_install_is_not_an_exception_either(self, monkeypatch):
        def explode(name: str) -> None:
            raise ValueError(f"{name}.__spec__ is not set")

        monkeypatch.setattr(importlib.util, "find_spec", explode)
        assert _bundled_libraries() == []


@pytest.mark.skipif(not rnnoise_available(), reason="RNNoise не установлена на этой машине")
class TestRnnoise:
    """The RNNoise path itself, not the fallback around it.

    Runs wherever the library can be loaded, which since ``pyrnnoise`` landed in
    ``requirements-ci-nodeps.txt`` means all three runners as well as the sandbox
    — so these assertions are about what RNNoise does when it is really loaded,
    and can be strict about it. Still skipped rather than failed on a machine
    without it: the dependency stays optional on purpose. To point at a library
    elsewhere, use :data:`RNNOISE_LIB_ENV`.
    """

    def test_the_mode_selects_rnnoise_and_reports_no_fallback(self):
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.RNNOISE))
        stream.push(wav("noise.wav"))
        assert stream.engine is DenoiseEngine.RNNOISE
        assert isinstance(stream.denoiser, RnnoiseDenoiser)
        assert not stream.stats.fallback

    def test_it_suppresses_far_more_than_the_gate(self):
        """The reason for the dependency: the gate is a compromise, this is not."""
        noise = wav("noise.wav")
        gated = pcm_level(denoise_pcm(noise, DenoiseSettings(mode=DenoiseMode.SPECTRAL))).rms_db
        cleaned = pcm_level(denoise_pcm(noise, DenoiseSettings(mode=DenoiseMode.RNNOISE))).rms_db
        assert pcm_level(noise).rms_db - cleaned > 20.0
        assert cleaned < gated

    def test_the_phrase_survives_and_stays_one_segment(self):
        """Suppression that eats the wake word is a regression, not a feature."""
        dirty = wav("phrase_in_noise.wav")
        cleaned = denoise_pcm(dirty, DenoiseSettings(mode=DenoiseMode.RNNOISE))
        segment = only(segment_pcm(cleaned))
        assert segment.accepted
        assert segment.speech_ms >= 800, "речь не должна укоротиться после подавления"

    def test_the_resampling_round_trip_keeps_the_length(self):
        """16 kHz in, 48 kHz inside, 16 kHz out — off-by-one here shifts everything."""
        pcm = wav("phrase.wav")
        assert len(denoise_pcm(pcm, DenoiseSettings(mode=DenoiseMode.RNNOISE))) == len(pcm)

    def test_it_can_be_switched_on_and_off_while_capture_runs(self):
        """Task 08 wants the toggle live: no restart, no gap, no leaked state."""
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.OFF))
        pcm = wav("phrase_in_noise.wav")
        assert stream.push(pcm) == pcm

        for mode, engine in (
            (DenoiseMode.RNNOISE, DenoiseEngine.RNNOISE),
            (DenoiseMode.SPECTRAL, DenoiseEngine.GATE),
            (DenoiseMode.RNNOISE, DenoiseEngine.RNNOISE),
            (DenoiseMode.OFF, DenoiseEngine.NONE),
        ):
            stream.configure(DenoiseSettings(mode=mode))
            assert stream.engine is engine
            assert len(stream.push(pcm) + stream.flush()) == len(pcm)

    def test_it_keeps_up_with_the_microphone(self):
        stream = DenoiseStream(DenoiseSettings(mode=DenoiseMode.RNNOISE))
        stream.push(wav("phrase.wav"))
        stats = stream.stats
        assert stats.latency_ms == 30.0, "кадр RNNoise — 480 сэмплов при 48 кГц"
        assert stats.realtime_factor < 0.5, "иначе подавление не успевает за захватом"


# ----------------------------------------------------------------------
# calibration
# ----------------------------------------------------------------------


class TestCalibration:
    """The report the settings window shows after «помолчите три секунды»."""

    def test_a_quiet_room_measures_as_a_quiet_room(self):
        profile = analyse_noise(wav("silence.wav"), TARGET_SAMPLE_RATE)
        assert profile.quiet
        assert not profile.noisy
        assert profile.duration_ms == 2000
        assert profile.rms_db < -60.0

    def test_a_fan_measures_as_noise(self):
        profile = analyse_noise(wav("noise.wav"), TARGET_SAMPLE_RATE)
        assert profile.noisy
        assert not profile.quiet

    def test_the_test_phrase_is_checked_for_being_caught_whole(self):
        check = analyse_phrase(wav("phrase.wav"), vad_settings=VadSettings(noise_floor_db=-73.0))
        assert check.detected
        assert check.whole
        assert check.segments == 1
        assert check.speech_ms == 900
        assert check.lead_ms > 0, "речь не упирается в начало записи"

    @pytest.mark.parametrize(
        ("silence_name", "phrase_name", "verdict", "ok"),
        [
            ("silence.wav", "phrase.wav", Verdict.GOOD, True),
            ("noise.wav", "phrase_in_noise.wav", Verdict.NOISY, True),
            ("silence.wav", "quiet_phrase.wav", Verdict.QUIET, False),
            ("silence.wav", "clipped.wav", Verdict.CLIPPING, False),
            ("silence.wav", "silence.wav", Verdict.NO_SPEECH, False),
        ],
    )
    def test_each_way_a_microphone_can_be_wrong_gets_its_own_verdict(
        self, silence_name: str, phrase_name: str, verdict: Verdict, ok: bool
    ):
        report = calibrate_pcm(wav(silence_name), wav(phrase_name))
        assert report.verdict is verdict
        assert report.ok is ok
        assert report.summary

    def test_the_advice_is_in_russian_and_says_what_to_do(self):
        """These strings go straight into a dialog; an English traceback will not do."""
        report = calibrate_pcm(wav("silence.wav"), wav("quiet_phrase.wav"))
        assert report.messages
        joined = " ".join(report.messages)
        assert "микрофон" in joined.lower()
        assert all(any("а" <= char.lower() <= "я" for char in text) for text in report.messages)

    def test_a_voice_that_is_too_quiet_gets_all_the_gain_there_is(self):
        report = calibrate_pcm(wav("silence.wav"), wav("quiet_phrase.wav"))
        assert report.recommended.gain == MAX_GAIN

    def test_a_voice_that_is_too_loud_gets_the_gain_taken_away(self):
        report = calibrate_pcm(wav("silence.wav"), wav("clipped.wav"))
        assert report.recommended.gain == MIN_GAIN

    def test_a_normal_voice_is_left_roughly_alone(self):
        report = calibrate_pcm(wav("silence.wav"), wav("phrase.wav"))
        assert 0.8 < report.recommended.gain < 1.5

    def test_the_recommended_threshold_sits_above_the_measured_floor(self):
        for silence_name in ("silence.wav", "noise.wav"):
            report = calibrate_pcm(wav(silence_name), wav("phrase.wav"))
            recommended = report.recommended
            assert 0.0 <= recommended.vad_threshold <= 1.0
            assert recommended.gate_db > recommended.noise_floor_db
            assert recommended.noise_floor_db == pytest.approx(
                pcm_level(wav(silence_name)).rms_db, abs=1.0
            )

    def test_the_recommendation_can_be_applied_to_a_detector(self):
        report = calibrate_pcm(wav("noise.wav"), wav("phrase_in_noise.wav"))
        settings = report.recommended.vad_settings(VadSettings(frame_ms=30))
        assert settings.frame_ms == 30, "чужие поля не затираются"
        assert settings.noise_floor_db == report.recommended.noise_floor_db

    def test_skipping_the_phrase_still_produces_a_usable_report(self):
        """Somebody presses «пропустить» — the floor alone is enough for a threshold."""
        report = calibrate_pcm(wav("silence.wav"))
        assert not report.phrase_checked
        assert report.verdict is Verdict.GOOD
        assert report.recommended.vad_threshold >= 0.0

    def test_the_report_survives_the_worker_pipe(self):
        """It is JSON on the way to the settings window, so it has to be JSON-able."""
        report = calibrate_pcm(wav("noise.wav"), wav("phrase_in_noise.wav"))
        restored = json.loads(json.dumps(report.as_dict()))
        assert restored["verdict"] == report.verdict.value
        assert restored["recommended"]["gain"] == pytest.approx(report.recommended.gain, abs=0.01)
        assert restored["phrase"]["checked"] is True
        assert isinstance(restored["messages"], list)

    def test_recording_happens_in_the_order_the_prompts_are_shown(self):
        """Silence first, then the phrase; the caller draws the prompts between them."""
        source = _FakeSource(["silence.wav", "phrase.wav"])
        report = run_calibration(source, silence_sec=2.0, phrase_sec=2.2)
        assert source.requested == [2.0, 2.2]
        assert isinstance(report, CalibrationReport)
        assert report.verdict is Verdict.GOOD

    def test_the_phrase_recording_can_be_skipped_by_the_caller(self):
        source = _FakeSource(["silence.wav"])
        report = run_calibration(source, silence_sec=2.0, phrase_sec=None)
        assert source.requested == [2.0]
        assert not report.phrase_checked

    def test_the_phrase_the_user_is_asked_to_say_is_the_wake_word(self):
        """A calibration phrase without «айрис» would not exercise the wake word."""
        assert CALIBRATION_PHRASE.startswith("айрис")

    def test_recommending_from_a_noisy_room_turns_denoising_on(self):
        noise = analyse_noise(wav("noise.wav"), TARGET_SAMPLE_RATE)
        phrase = analyse_phrase(
            wav("phrase_in_noise.wav"), vad_settings=VadSettings(noise_floor_db=noise.rms_db)
        )
        assert recommend(noise, phrase).denoise is not DenoiseMode.OFF


class _FakeSource:
    """An :class:`~ayris.audio.calibration.AudioSource` that returns fixtures."""

    sample_rate = TARGET_SAMPLE_RATE

    def __init__(self, names: list[str]) -> None:
        self._names = list(names)
        self.requested: list[float] = []

    def record(self, seconds: float) -> bytes:
        self.requested.append(seconds)
        return wav(self._names[len(self.requested) - 1])


# ----------------------------------------------------------------------
# the worker
# ----------------------------------------------------------------------


class TestWorkerWiring:
    """Configuration in, bus events out — no capture involved."""

    def test_settings_come_straight_out_of_the_configuration_section(self):
        params = {
            "sample_rate": 16000,
            "frame_ms": 30,
            "vad_aggressiveness": 3,
            "vad_threshold": 0.7,
            "noise_floor_db": -50.0,
        }
        settings = _vad_settings_from_params(params)
        assert settings.frame_ms == 30
        assert settings.aggressiveness == 3
        assert settings.threshold == pytest.approx(0.7)

    def test_seconds_in_the_file_become_milliseconds_below_it(self):
        settings = _segmenter_settings_from_params({"max_utterance_sec": 12, "silence_ms": 500})
        assert settings.max_utterance_ms == 12_000
        assert settings.silence_ms == 500

    def test_numbers_that_arrived_as_strings_still_work(self):
        """DevTools and TOML both hand over strings often enough to matter."""
        settings = _vad_settings_from_params({"vad_threshold": "0.25", "frame_ms": "10"})
        assert settings.threshold == pytest.approx(0.25)
        assert settings.frame_ms == 10

    def test_a_typo_in_the_configuration_does_not_stop_the_microphone(self):
        """Refusing to capture over an unknown denoise mode would be a poor trade."""
        assert _denoise_settings_from_params({"denoise": "магия"}).mode is DenoiseMode.RNNOISE
        assert _denoise_settings_from_params({"denoise": "off"}).mode is DenoiseMode.OFF
        assert _vad_settings_from_params({"vad_threshold": "громко"}).threshold == pytest.approx(
            0.5
        )

    def test_the_level_event_carries_the_speech_flag_the_overlay_animates(self):
        from ayris.core.events import AudioLevelChanged

        event = translate_audio_event("level", {"rms": 0.2, "peak": 0.5, "is_speech": True})
        assert isinstance(event, AudioLevelChanged)
        assert event.is_speech
        assert event.rms == pytest.approx(0.2)

    def test_an_onset_becomes_a_speech_started_event(self):
        from ayris.core.events import SpeechStarted

        event = translate_audio_event("speech_started", {"frame_index": 3, "pre_roll_ms": 340})
        assert isinstance(event, SpeechStarted)
        assert event.source == "vad"

    def test_a_finished_phrase_becomes_a_speech_ended_event(self):
        from ayris.core.events import SpeechEnded

        event = translate_audio_event(
            "speech_ended", {"duration_ms": 1440, "reason": "silence", "accepted": True}
        )
        assert isinstance(event, SpeechEnded)
        assert event.duration_ms == 1440
        assert event.reason == "silence"

    def test_a_rejected_phrase_is_published_too(self):
        """The onset already went out; something has to take the overlay back."""
        from ayris.core.events import SpeechEnded

        event = translate_audio_event(
            "speech_ended", {"duration_ms": 600, "reason": "too_short", "accepted": False}
        )
        assert isinstance(event, SpeechEnded)
        assert event.reason == "too_short"

    def test_an_unknown_event_is_dropped_rather_than_crashing_the_bus(self):
        assert translate_audio_event("что-то новое", {}) is None


# ----------------------------------------------------------------------
# hardware
# ----------------------------------------------------------------------


class TestLiveMicrophone:
    """Excluded from CI: a runner has no microphone, and the sandbox has no timing.

    Runs on every ``check.sh`` on a machine that has one, in the quiet serial pass
    — but only reports a failure when the microphone answered and the numbers came
    out wrong. A missing, muted or busy device is a skip with the reason printed:
    otherwise the red would mean «выдерни наушники», which is not a defect.
    """

    @pytest.mark.hardware
    def test_calibration_against_a_real_room(self):
        """Stay quiet for two seconds while this one runs."""
        import time

        from ayris.audio.capture import AudioCapture, CaptureSettings

        capture = AudioCapture(CaptureSettings())
        try:
            capture.start()
        except AudioError as exc:
            pytest.skip(f"микрофон не открылся: {exc}")
        try:
            time.sleep(2.0)
            silence = capture.read_recent(2000.0)
        finally:
            capture.stop()
        if not bytes(silence):
            pytest.skip("микрофон открылся, но кадров не прислал — выключен или замьючен")
        report = calibrate_pcm(silence)
        assert report.recommended.noise_floor_db < 0.0
        assert 0.0 <= report.recommended.vad_threshold <= 1.0
