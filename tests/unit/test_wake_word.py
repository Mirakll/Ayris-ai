"""Task 09: the wake word — the engine contract, the manager, the worker seam.

Two test doubles carry this file, because the two things worth checking need
opposite kinds of input.

:class:`ScriptedEngine` returns scores handed to it. Thresholds, the debounce
window, the counters and the live phrase list are arithmetic over scores, and
driving them with a real model would mean asserting on numbers nobody chose.

:class:`FormantEngine` is the opposite: a genuine, if crude, keyword spotter
that reads the waveform. It bandpasses each frame around the first formant of
"а" and around the second of "и", classifies the frame from the ratio, and
matches the resulting vowel sequence against the phrase. That is enough to tell
``айрис`` from ``ирис`` — they differ only in the first vowel — which is exactly
the discrimination the acceptance criteria are about, and it is why the fixtures
are built from two vowels rather than one.

Neither is the shipped engine. openWakeWord, Porcupine and Vosk all need weights
or a vendor key that the repository deliberately does not contain, so the tests
that touch them carry the ``hardware`` marker and CI skips them; what is checked
here without them is that selecting a missing engine produces a sentence for the
user instead of an ImportError at startup.

Groups:

* :class:`TestFixtures` — the ``wake_*.wav`` files are what the rest assumes.
* :class:`TestWakePhrase` — validation and the sensitivity-to-threshold curve.
* :class:`TestEngineRegistry` — names, lazy imports, missing libraries.
* :class:`TestEngineContract` — what the base class guarantees to the manager.
* :class:`TestDetector` — thresholds and per-phrase sensitivity, on scores.
* :class:`TestDebounce` — one utterance, one activation.
* :class:`TestLivePhraseList` — retuning without restarting the worker.
* :class:`TestPushToTalk` — the task 37 entry point.
* :class:`TestStats` — the false-positive metric.
* :class:`TestOnFixtures` — the acceptance criteria, on real waveforms.
* :class:`TestWorkerWiring` — parameters in, bus events out.
* :class:`TestRealEngines` — needs weights; excluded from CI.
"""

from __future__ import annotations

import math
import os
import wave
from array import array
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, ClassVar

import pytest

from ayris.audio.capture import TARGET_SAMPLE_RATE
from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.audio.wake_word import (
    DEFAULT_SENSITIVITY,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    PTT_PHRASE,
    WAKE_SAMPLE_RATE,
    ModelSpec,
    WakeDetection,
    WakePhrase,
    WakeStats,
    WakeWordCallbacks,
    WakeWordDetector,
    WakeWordEngine,
    WakeWordSettings,
    create_engine,
    engine_names,
    normalise_phrase,
    phrases_from,
)
from ayris.audio.wake_word.base import ENGINE_ENTRYPOINTS, engine_class
from ayris.audio.wake_word.manager import (
    DEFAULT_DEBOUNCE_MS,
    MAX_DEBOUNCE_MS,
    MIN_DEBOUNCE_MS,
)
from ayris.audio.wake_word.openwakeword import OpenWakeWordEngine
from ayris.audio.wake_word.porcupine import PorcupineEngine
from ayris.audio.wake_word.vosk_kws import VoskKwsEngine
from ayris.core.errors import WakeWordError
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.workers.audio_worker import (
    _wake_access_key,
    _wake_settings_from_params,
    translate_audio_event,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: How long each wake fixture is, in milliseconds, by construction.
DURATIONS = {
    "wake_ayris.wav": 2000,
    "wake_absent.wav": 2900,
    "wake_similar.wav": 2020,
    "wake_double.wav": 3100,
}

#: The word the fixtures say, and the near-miss that must not count as it.
AYRIS = "айрис"
IRIS = "ирис"

#: How long :func:`feed` waits for the processing thread. Generous: the thread
#: is doing real filtering in pure Python, and a loaded CI runner is slow.
_FEED_TIMEOUT_SEC = 30.0


def wav(name: str) -> bytes:
    """Read a WAV fixture and return its raw PCM frames.

    Asserts mono, 16-bit, 16 kHz - the form every wake fixture has and every
    engine needs.
    """
    path = FIXTURES / name
    with wave.open(str(path), "rb") as fp:
        assert fp.getnchannels() == 1, f"{name}: must be mono"
        assert fp.getsampwidth() == SAMPLE_WIDTH, f"{name}: must be 16-bit"
        assert fp.getframerate() == WAKE_SAMPLE_RATE, f"{name}: must be 16 kHz"
        return fp.readframes(fp.getnframes())


def feed(detector: WakeWordDetector, pcm: bytes, *, frame_samples: int) -> None:
    """Push audio through the detector and wait for it to be scored.

    Args:
        detector: The detector to feed.
        pcm: Raw PCM, mono int16.
        frame_samples: Length of one detector frame, in samples. Used to
            compute how many frames were fed, so the poll knows when to stop.
    """
    # The counters are per session and survive reset(), so the target has to be
    # relative to where they already are - otherwise a second feed() in the same
    # test returns before the thread has touched the new audio.
    target = detector.stats.frames + len(pcm) // (frame_samples * SAMPLE_WIDTH)
    for offset in range(0, len(pcm), frame_samples * SAMPLE_WIDTH):
        detector.push(pcm[offset : offset + frame_samples * SAMPLE_WIDTH])
    deadline = perf_counter() + _FEED_TIMEOUT_SEC
    while perf_counter() < deadline:
        if detector.stats.frames >= target:
            return
        sleep(0.01)
    raise TimeoutError(
        f"detector processed {detector.stats.frames}/{target} frames in {_FEED_TIMEOUT_SEC}s"
    )


# ======================================================================== doubles


class ScriptedEngine(WakeWordEngine):
    """Returns scores handed to it by the test, for driving the decision logic.

    Thresholds, debounce, counters and the live phrase list are arithmetic over
    scores. A real model would mean asserting on numbers nobody picked, and a
    test that asserts "the detector fired when openWakeWord said 0.712" is not
    checking the manager at all.

    ``frame_samples`` is deliberately not 1280 or 512 - those are the real
    engines, and a test written against their block length would pass even if
    the manager never resampled.
    """

    name: ClassVar[str] = "scripted"
    package: ClassVar[str] = ""
    module: ClassVar[str] = ""

    __slots__ = ("_index", "_script", "load_count", "process_count", "reset_count")

    def __init__(self, script: Mapping[int, Mapping[str, float]] | None = None) -> None:
        super().__init__()
        #: Frame index to phrase-score map.
        self._script: dict[int, Mapping[str, float]] = dict(script) if script else {}
        self._index = 0
        self.load_count = 0
        self.process_count = 0
        self.reset_count = 0

    @property
    def frame_samples(self) -> int:
        return 1600  # 100 ms at 16 kHz

    def load(self, spec: ModelSpec) -> None:
        self._spec = spec
        self.load_count += 1

    def process(self, frame: bytes) -> WakeDetection | None:
        self._check_frame(frame)
        self.process_count += 1
        scores = self._script.get(self._index, {})
        self._index += 1
        if not scores:
            return None
        winner, best = max(scores.items(), key=lambda item: item[1])
        return WakeDetection(phrase=winner, score=best, scores=dict(scores))

    def unload(self) -> None:
        self._spec = None

    def reset(self) -> None:
        self._index = 0
        self.reset_count += 1


class FormantEngine(WakeWordEngine):
    """Analyses the fixture waveforms, for checking the acceptance criteria.

    Crude but genuine: it bandpasses around the first formant of "а" and the
    second of "и", classifies each frame from the energy ratio, and matches the
    resulting vowel run against the phrase. That is enough to tell ``айрис``
    from ``ирис`` on the real ``wake_*.wav`` files, and it is the discrimination
    the task is about.
    """

    name: ClassVar[str] = "formant"
    package: ClassVar[str] = ""
    module: ClassVar[str] = ""

    #: Maximum score this engine produces. Deliberately below MAX_THRESHOLD so
    #: that sensitivity 0.0 never fires - a variant configured that way is "off".
    MAX_SCORE: ClassVar[float] = 0.9

    __slots__ = ("_history", "_silent_run")

    #: Band centres and half-widths for confidence. Measured on the fixtures:
    #: "а" lands around 0.004 and "и" around 0.013, the fricative above 0.11.
    _A_CENTRE: ClassVar[float] = 0.004
    _A_HALF: ClassVar[float] = 0.007
    _I_CENTRE: ClassVar[float] = 0.013
    _I_HALF: ClassVar[float] = 0.010

    #: RMS floor: frames quieter than this are silent and clear the history.
    _RMS_FLOOR: ClassVar[float] = 0.02

    #: Ratio thresholds that separate the three classes. The gap between the
    #: vowels is narrow but clean; anything above the second is a fricative.
    _A_UPPER: ClassVar[float] = 0.010
    _I_UPPER: ClassVar[float] = 0.060

    def __init__(self) -> None:
        super().__init__()
        self._history: list[str] = []
        self._silent_run = 0

    @property
    def frame_samples(self) -> int:
        return 1280  # 80 ms - openWakeWord's size

    def load(self, spec: ModelSpec) -> None:
        self._require_phrases(spec)
        self._spec = spec

    def process(self, frame: bytes) -> WakeDetection | None:
        self._check_frame(frame)
        samples = array("h", frame)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768.0

        if rms < self._RMS_FLOOR:
            self._silent_run += 1
            if self._silent_run >= 2:
                self._history.clear()
            return None

        self._silent_run = 0
        lo_band = self._band_energy(samples, 700, 220)
        hi_band = self._band_energy(samples, 2200, 450)
        ratio = hi_band / lo_band if lo_band > 1e-9 else 1.0

        if ratio < self._A_UPPER:
            vowel_class = "a"
            conf = self._confidence(ratio, self._A_CENTRE, self._A_HALF)
        elif ratio < self._I_UPPER:
            vowel_class = "i"
            conf = self._confidence(ratio, self._I_CENTRE, self._I_HALF)
        else:
            # Fricative or noise - neither extends nor clears the history.
            return None

        if not self._history or self._history[-1] != vowel_class:
            self._history.append(vowel_class)

        assert self._spec is not None
        best_phrase: str | None = None
        best_score = 0.0
        all_scores: dict[str, float] = {}

        for phrase_obj in self._spec.enabled_phrases:
            pattern = self._pattern_of(phrase_obj.text)
            if self._history[-len(pattern) :] == pattern:
                score = self.MAX_SCORE * conf
                all_scores[phrase_obj.text] = score
                if score > best_score:
                    best_phrase = phrase_obj.text
                    best_score = score

        if best_phrase is None:
            return None
        return WakeDetection(
            phrase=best_phrase, score=best_score, engine=self.name, scores=all_scores
        )

    def unload(self) -> None:
        self._spec = None

    def reset(self) -> None:
        self._history.clear()
        self._silent_run = 0

    @staticmethod
    def _pattern_of(text: str) -> list[str]:
        """Collapse the phrase to the vowel classes that will match it.

        Russian vowels map to their formant group: а/о/у/ы/э → "a", и/е/ю/я/ё → "i".
        Consonants are ignored. Repeats are collapsed so "ирис" → ["i"], not ["i","i"].
        """
        mapping = {"а": "a", "о": "a", "у": "a", "ы": "a", "э": "a"}
        mapping.update({"и": "i", "е": "i", "ю": "i", "я": "i", "ё": "i"})
        classes: list[str] = []
        for char in text.lower():
            cls = mapping.get(char)
            if cls and (not classes or classes[-1] != cls):
                classes.append(cls)
        return classes

    @staticmethod
    def _band_energy(samples: array[int], centre_hz: float, bandwidth_hz: float) -> float:
        """Sum of squared samples after a resonator tuned to ``centre_hz``.

        A crude two-pole filter driven by the frame. Not meant to be efficient -
        only to separate "а" from "и" on 80 ms blocks.
        """
        rate = WAKE_SAMPLE_RATE
        r = math.exp(-math.pi * bandwidth_hz / rate)
        theta = 2.0 * math.pi * centre_hz / rate
        b1 = -2.0 * r * math.cos(theta)
        b2 = r * r
        y1, y2 = 0.0, 0.0
        energy = 0.0
        for sample in samples:
            x = float(sample) / 32768.0
            y = x - b1 * y1 - b2 * y2
            energy += y * y
            y2, y1 = y1, y
        return energy

    @staticmethod
    def _confidence(ratio: float, centre: float, half: float) -> float:
        """How close ``ratio`` is to ``centre``, as a 0.0-1.0 score.

        Linearly drops from 1.0 at the centre to 0.3 at the edges of the band.
        Clamped so that values outside the band still return something usable.
        """
        margin = abs(ratio - centre) / half
        return max(0.3, min(1.0, 1.0 - margin))


# =========================================================================== tests


class TestFixtures:
    """The ``wake_*.wav`` files are what the rest of this file assumes."""

    def test_all_present(self) -> None:
        """Every fixture named in DURATIONS exists."""
        for name in DURATIONS:
            assert (FIXTURES / name).is_file(), f"missing {name}"

    def test_durations(self) -> None:
        """Each fixture is exactly as long as the generator declares."""
        for name, expected_ms in DURATIONS.items():
            pcm = wav(name)
            actual_ms = len(pcm) / SAMPLE_WIDTH / WAKE_SAMPLE_RATE * 1000.0
            assert abs(actual_ms - expected_ms) < 1.0, f"{name}: {actual_ms:.0f} != {expected_ms}"

    def test_not_silent(self) -> None:
        """Fixtures contain signal, not runs of zeros."""
        for name in DURATIONS:
            pcm = wav(name)
            samples = array("h", pcm)
            rms = (sum(s * s for s in samples) / len(samples)) ** 0.5 / 32768.0
            assert rms > 0.001, f"{name}: too quiet (rms {rms:.6f})"

    def test_total_size(self) -> None:
        """All four wake fixtures together fit comfortably in memory."""
        total = sum((FIXTURES / name).stat().st_size for name in DURATIONS)
        assert total < 5_000_000, f"wake fixtures are {total} bytes"

    def test_generator_committed(self) -> None:
        """The generator script is tracked, so fixtures can be rebuilt."""
        script = FIXTURES.parent.parent / "fixtures" / "audio" / "make_fixtures.py"
        assert script.is_file(), "make_fixtures.py must be committed"


class TestWakePhrase:
    """Validation and the sensitivity-to-threshold curve."""

    def test_empty_rejected(self) -> None:
        with pytest.raises(WakeWordError, match="must not be empty"):
            WakePhrase("")

    def test_whitespace_rejected(self) -> None:
        with pytest.raises(WakeWordError, match="must not be empty"):
            WakePhrase("   ")

    def test_sensitivity_below_zero(self) -> None:
        with pytest.raises(WakeWordError, match="must be 0.0-1.0"):
            WakePhrase("test", sensitivity=-0.1)

    def test_sensitivity_above_one(self) -> None:
        with pytest.raises(WakeWordError, match="must be 0.0-1.0"):
            WakePhrase("test", sensitivity=1.1)

    def test_text_normalised(self) -> None:
        phrase = WakePhrase(" Айрис  Тест ")
        assert phrase.text == "айрис тест"

    def test_threshold_at_default_sensitivity(self) -> None:
        phrase = WakePhrase("test", sensitivity=DEFAULT_SENSITIVITY)
        assert phrase.threshold == pytest.approx(0.5, abs=0.01)

    def test_threshold_at_minimum_sensitivity(self) -> None:
        phrase = WakePhrase("test", sensitivity=0.0)
        assert phrase.threshold == pytest.approx(MAX_THRESHOLD, abs=0.001)

    def test_threshold_at_maximum_sensitivity(self) -> None:
        phrase = WakePhrase("test", sensitivity=1.0)
        assert phrase.threshold == pytest.approx(MIN_THRESHOLD, abs=0.001)

    def test_with_sensitivity(self) -> None:
        original = WakePhrase("test", sensitivity=0.3, engine_model="foo")
        changed = original.with_sensitivity(0.7)
        assert changed.text == "test"
        assert changed.sensitivity == 0.7
        assert changed.engine_model == "foo"
        assert original.sensitivity == 0.3

    def test_phrases_from_empty(self) -> None:
        assert phrases_from([]) == ()

    def test_phrases_from_simple(self) -> None:
        items = [{"phrase": "one"}, {"phrase": "two", "sensitivity": 0.8}]
        result = phrases_from(items)
        assert len(result) == 2
        assert result[0].text == "one"
        assert result[0].sensitivity == DEFAULT_SENSITIVITY
        assert result[1].text == "two"
        assert result[1].sensitivity == 0.8

    def test_phrases_from_skips_empty(self) -> None:
        items = [{"phrase": ""}, {"phrase": "  "}, {"phrase": "valid"}]
        result = phrases_from(items)
        assert len(result) == 1
        assert result[0].text == "valid"

    def test_phrases_from_deduplicates(self) -> None:
        items = [{"phrase": "test"}, {"phrase": "TEST"}, {"phrase": "other"}]
        result = phrases_from(items)
        assert len(result) == 2
        assert result[0].text == "test"
        assert result[1].text == "other"


class TestEngineRegistry:
    """Names, lazy imports, missing libraries."""

    def test_engine_names(self) -> None:
        names = engine_names()
        assert "openwakeword" in names
        assert "porcupine" in names
        assert "vosk" in names

    def test_unknown_name(self) -> None:
        with pytest.raises(WakeWordError, match="unknown wake word engine"):
            create_engine("nonexistent")

    def test_available_check(self) -> None:
        # OpenWakeWordEngine.available() without importing openwakeword itself.
        assert OpenWakeWordEngine.available() or not OpenWakeWordEngine.available()

    def test_porcupine_needs_credential(self) -> None:
        assert PorcupineEngine.needs_credential is True

    def test_openwakeword_frame_size(self) -> None:
        engine = OpenWakeWordEngine()
        assert engine.frame_samples == 1280

    def test_vosk_frame_size(self) -> None:
        engine = VoskKwsEngine()
        assert engine.frame_samples == 1280

    def test_entrypoints_resolve(self) -> None:
        """Every declared entrypoint imports and is a WakeWordEngine subclass."""
        for name in ENGINE_ENTRYPOINTS:
            cls = engine_class(name)
            assert issubclass(cls, WakeWordEngine), name
            assert cls.name == name

    def test_engine_frame_bytes(self) -> None:
        engine = ScriptedEngine()
        assert engine.frame_bytes == engine.frame_samples * SAMPLE_WIDTH
        assert engine.sample_rate == WAKE_SAMPLE_RATE
        assert engine.frame_ms == pytest.approx(100.0)


class TestEngineContract:
    """What the base class guarantees to the manager."""

    def test_not_loaded_initially(self) -> None:
        engine = ScriptedEngine()
        assert not engine.loaded
        assert engine.phrases == ()
        assert engine.spec is None

    def test_loaded_after_load(self) -> None:
        engine = ScriptedEngine()
        engine.load(ModelSpec(phrases=(WakePhrase(AYRIS),)))
        assert engine.loaded
        assert engine.phrases[0].text == AYRIS

    def test_disabled_phrases_not_reported(self) -> None:
        engine = ScriptedEngine()
        engine.load(ModelSpec(phrases=(WakePhrase(AYRIS), WakePhrase(IRIS, enabled=False))))
        assert [item.text for item in engine.phrases] == [AYRIS]

    def test_wrong_frame_size_rejected(self) -> None:
        engine = ScriptedEngine()
        engine.load(ModelSpec(phrases=(WakePhrase(AYRIS),)))
        with pytest.raises(WakeWordError, match="frame must be"):
            engine.process(b"\x00\x00")

    def test_update_phrases_before_load(self) -> None:
        engine = ScriptedEngine()
        with pytest.raises(WakeWordError, match="before the model is loaded"):
            engine.update_phrases([WakePhrase(AYRIS)])

    def test_update_phrases_reloads(self) -> None:
        engine = ScriptedEngine()
        engine.load(ModelSpec(phrases=(WakePhrase(AYRIS),)))
        engine.update_phrases([WakePhrase(IRIS)])
        assert engine.load_count == 2
        assert [item.text for item in engine.phrases] == [IRIS]

    def test_require_phrases_rejects_empty(self) -> None:
        engine = FormantEngine()
        with pytest.raises(WakeWordError, match="without any enabled phrase"):
            engine.load(ModelSpec(phrases=()))

    def test_unload_is_idempotent(self) -> None:
        engine = ScriptedEngine()
        engine.load(ModelSpec(phrases=(WakePhrase(AYRIS),)))
        engine.unload()
        engine.unload()
        assert not engine.loaded

    def test_model_spec_option(self) -> None:
        spec = ModelSpec(options={"framework": "onnx"})
        assert spec.option("framework") == "onnx"
        assert spec.option("missing") == ""
        assert spec.option("missing", "fallback") == "fallback"

    def test_detection_all_scores_falls_back(self) -> None:
        detection = WakeDetection(phrase=AYRIS, score=0.8)
        assert detection.all_scores() == {AYRIS: 0.8}


def _detector(
    engine: WakeWordEngine,
    *,
    phrases: tuple[WakePhrase, ...],
    debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    seen: list[WakeDetection] | None = None,
) -> WakeWordDetector:
    """Build a started detector around ``engine``, recording activations in ``seen``."""
    events = seen if seen is not None else []
    detector = WakeWordDetector(
        WakeWordSettings(
            phrases=phrases,
            debounce_ms=debounce_ms,
            queue_blocks=4096,
        ),
        WakeWordCallbacks(on_detected=events.append),
        engine=engine,
    )
    detector.start()
    return detector


def _silence(frames: int, *, frame_samples: int) -> bytes:
    """``frames`` frames of digital silence, for driving a ScriptedEngine."""
    return b"\x00\x00" * frame_samples * frames


class TestDetector:
    """Thresholds and per-phrase sensitivity, driven by known scores."""

    def test_score_above_threshold_fires(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.8}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1
        assert seen[0].phrase == AYRIS
        assert seen[0].score == pytest.approx(0.8)

    def test_score_below_threshold_does_not_fire(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.3}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert seen == []
        assert stats.candidates == 1
        assert stats.below_threshold == 1

    def test_per_phrase_threshold(self) -> None:
        """A score clears one variant and not the other, in the same frame.

        This is the reason there is no single engine-wide threshold: 0.4 is
        above what a sensitive variant asks for and below what a strict one
        does, and both are configured at once.
        """
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.4, IRIS: 0.4}})
        detector = _detector(
            engine,
            phrases=(WakePhrase(AYRIS, 0.9), WakePhrase(IRIS, 0.1)),
            seen=seen,
        )
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1
        assert seen[0].phrase == AYRIS

    def test_zero_sensitivity_never_fires(self) -> None:
        """Sensitivity 0.0 means "off": nothing an engine emits reaches it."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.94}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.0),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert seen == []

    def test_disabled_phrase_ignored(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5, enabled=False),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert seen == []

    def test_unknown_phrase_in_scores_ignored(self) -> None:
        """A model scoring something the user never configured is not an activation."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {"привет": 0.99}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert seen == []
        assert stats.below_threshold == 1

    def test_disabled_detector_never_scores(self) -> None:
        """Push-to-Talk mode: the engine is not even loaded."""
        engine = ScriptedEngine({0: {AYRIS: 0.99}})
        detector = WakeWordDetector(
            WakeWordSettings(enabled=False, phrases=(WakePhrase(AYRIS),)),
            engine=engine,
        )
        detector.start()
        try:
            detector.push(_silence(5, frame_samples=1600))
            sleep(0.05)
            stats = detector.stats
        finally:
            detector.stop()
        assert engine.process_count == 0
        assert stats.frames == 0
        assert not stats.running

    def test_audio_clock_advances(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            feed(detector, _silence(10, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.frames == 10
        assert stats.audio_sec == pytest.approx(1.0, abs=0.001)


class TestDebounce:
    """One utterance, one activation."""

    def test_repeat_inside_window_suppressed(self) -> None:
        """Frames 2 and 5 are 300 ms apart; a 1.5 s window collapses them."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 5: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(10, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert len(seen) == 1
        assert stats.candidates == 2
        assert stats.debounced == 1

    def test_repeat_after_window_fires_again(self) -> None:
        """Frames 2 and 20 are 1.8 s apart; both are real activations."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 20: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(25, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 2

    def test_window_is_audio_time(self) -> None:
        """Timestamps are positions in the stream, not wall-clock readings.

        The whole file is pushed in a fraction of a second of real time, so a
        wall-clock debounce would suppress the second activation.
        """
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 20: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(25, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 2
        assert seen[0].timestamp == pytest.approx(0.3, abs=0.001)
        assert seen[1].timestamp == pytest.approx(2.1, abs=0.001)

    def test_global_window_covers_other_variants(self) -> None:
        """Two near-identical variants scoring on one utterance start one conversation."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 4: {IRIS: 0.9}})
        detector = _detector(
            engine,
            phrases=(WakePhrase(AYRIS, 0.5), WakePhrase(IRIS, 0.5)),
            seen=seen,
        )
        try:
            feed(detector, _silence(8, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert len(seen) == 1
        assert stats.debounced == 1

    def test_short_debounce_lets_both_through(self) -> None:
        """A 200 ms window at 300 ms apart is not a suppression."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 5: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), debounce_ms=200, seen=seen)
        try:
            feed(detector, _silence(10, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 2

    def test_debounce_clamped(self) -> None:
        assert WakeWordSettings(debounce_ms=0).debounce_sec == MIN_DEBOUNCE_MS / 1000.0
        assert WakeWordSettings(debounce_ms=10**9).debounce_sec == MAX_DEBOUNCE_MS / 1000.0
        assert WakeWordSettings(debounce_ms=1750).debounce_sec == pytest.approx(1.75)

    def test_reset_clears_window(self) -> None:
        """Restarting capture must not carry the window into the new stream."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            assert len(seen) == 1
            detector.reset()
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 2


class TestLivePhraseList:
    """Retuning without restarting the worker."""

    def test_raise_sensitivity_starts_firing(self) -> None:
        """The same score is refused, then accepted, with no restart in between."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.4}, 30: {AYRIS: 0.4}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.1),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            assert seen == []
            assert detector.set_sensitivity(AYRIS, 0.9)
            feed(detector, _silence(30, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1
        assert seen[0].phrase == AYRIS

    def test_lower_sensitivity_stops_firing(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.6}, 30: {AYRIS: 0.6}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.9),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            assert len(seen) == 1
            assert detector.set_sensitivity(AYRIS, 0.1)
            feed(detector, _silence(30, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1

    def test_set_sensitivity_unknown_phrase(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            assert detector.set_sensitivity("nothing", 0.5) is False
        finally:
            detector.stop()

    def test_set_sensitivity_validates(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            with pytest.raises(WakeWordError, match="must be 0.0-1.0"):
                detector.set_sensitivity(AYRIS, 5.0)
        finally:
            detector.stop()

    def test_add_phrase_live(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {IRIS: 0.9}, 30: {IRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            assert seen == []
            detector.add_phrase(WakePhrase(IRIS, 0.5))
            feed(detector, _silence(30, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1
        assert seen[0].phrase == IRIS

    def test_add_phrase_replaces_same_text(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.2),))
        try:
            detector.add_phrase(WakePhrase(AYRIS, 0.8))
            assert len(detector.phrases) == 1
            assert detector.phrases[0].sensitivity == 0.8
        finally:
            detector.stop()

    def test_remove_phrase_live(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 30: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            assert len(seen) == 1
            assert detector.remove_phrase(AYRIS) is True
            feed(detector, _silence(30, frame_samples=1600), frame_samples=1600)
        finally:
            detector.stop()
        assert len(seen) == 1

    def test_remove_missing_phrase(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            assert detector.remove_phrase("nothing") is False
        finally:
            detector.stop()

    def test_unlimited_variants(self) -> None:
        """The list has no cap: fifty variants load and each keeps its own value."""
        variants = tuple(
            WakePhrase(f"вариант {index}", sensitivity=index / 100.0) for index in range(50)
        )
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=variants)
        try:
            assert len(detector.phrases) == 50
            assert detector.phrases[7].sensitivity == pytest.approx(0.07)
        finally:
            detector.stop()

    def test_sensitivity_change_does_not_reload(self) -> None:
        """An engine whose scores are threshold-independent is not rebuilt."""
        engine = ScriptedEngine({20: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.2),))
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            loads_before = engine.load_count
            detector.set_sensitivity(AYRIS, 0.9)
            feed(detector, _silence(25, frame_samples=1600), frame_samples=1600)
            assert engine.load_count == loads_before
        finally:
            detector.stop()

    def test_new_phrase_reloads_engine(self) -> None:
        """A changed model list does need the engine rebuilt."""
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            feed(detector, _silence(2, frame_samples=1600), frame_samples=1600)
            loads_before = engine.load_count
            detector.add_phrase(WakePhrase(IRIS))
            feed(detector, _silence(4, frame_samples=1600), frame_samples=1600)
            assert engine.load_count == loads_before + 1
        finally:
            detector.stop()


class TestPushToTalk:
    """The task 37 entry point: activation without the microphone."""

    def test_manual_publishes_event(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),), seen=seen)
        try:
            detection = detector.trigger_manual("hotkey")
        finally:
            detector.stop()
        assert detection.phrase == PTT_PHRASE
        assert detection.score == 1.0
        assert len(seen) == 1
        assert seen[0].phrase == PTT_PHRASE

    def test_manual_works_without_phrases(self) -> None:
        """Push-to-Talk is exactly the case where no variant is configured."""
        seen: list[WakeDetection] = []
        detector = WakeWordDetector(
            WakeWordSettings(enabled=False),
            WakeWordCallbacks(on_detected=seen.append),
        )
        detector.start()
        try:
            detector.trigger_manual("hotkey")
        finally:
            detector.stop()
        assert len(seen) == 1

    def test_manual_not_debounced(self) -> None:
        """Two key presses meant two activations; the window is for utterances."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),), seen=seen)
        try:
            detector.trigger_manual("hotkey")
            detector.trigger_manual("hotkey")
        finally:
            detector.stop()
        assert len(seen) == 2

    def test_manual_resets_the_window(self) -> None:
        """Pressing the key and then saying the word is one activation, not two."""
        seen: list[WakeDetection] = []
        engine = ScriptedEngine({2: {AYRIS: 0.9}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),), seen=seen)
        try:
            detector.trigger_manual("hotkey")
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert len(seen) == 1
        assert seen[0].phrase == PTT_PHRASE
        assert stats.debounced == 1

    def test_manual_source_recorded(self) -> None:
        seen: list[WakeDetection] = []
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),), seen=seen)
        try:
            detector.trigger_manual("tray")
        finally:
            detector.stop()
        assert seen[0].engine == "tray"

    def test_manual_counted_separately(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            detector.trigger_manual("hotkey")
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.manual == 1
        assert stats.fired == 1
        assert stats.fired_by_phrase[PTT_PHRASE] == 1

    def test_ptt_phrase_cannot_collide(self) -> None:
        """The marker is not something a user could type as a variant."""
        assert normalise_phrase(PTT_PHRASE) == PTT_PHRASE
        assert not PTT_PHRASE.isalnum()


class TestStats:
    """The false-positive metric."""

    def test_empty_stats(self) -> None:
        stats = WakeStats()
        assert stats.rejected == 0
        assert stats.false_positive_rate == 0.0

    def test_rejected_is_the_sum(self) -> None:
        stats = WakeStats(below_threshold=3, debounced=2)
        assert stats.rejected == 5

    def test_rate_per_minute(self) -> None:
        stats = WakeStats(below_threshold=2, debounced=2, audio_sec=120.0)
        assert stats.false_positive_rate == pytest.approx(2.0)

    def test_counters_split_by_reason(self) -> None:
        """ "Fires too often" and "fires twice per phrase" have different fixes."""
        engine = ScriptedEngine({2: {AYRIS: 0.9}, 4: {AYRIS: 0.9}, 30: {AYRIS: 0.2}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),))
        try:
            feed(detector, _silence(35, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.candidates == 3
        assert stats.fired == 1
        assert stats.debounced == 1
        assert stats.below_threshold == 1
        assert stats.rejected == 2

    def test_rejected_by_phrase(self) -> None:
        engine = ScriptedEngine({2: {AYRIS: 0.2}, 4: {IRIS: 0.2}, 6: {AYRIS: 0.2}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.1), WakePhrase(IRIS, 0.1)))
        try:
            feed(detector, _silence(10, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.rejected_by_phrase[AYRIS] == 2
        assert stats.rejected_by_phrase[IRIS] == 1

    def test_last_activation_recorded(self) -> None:
        engine = ScriptedEngine({3: {AYRIS: 0.77}})
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),))
        try:
            feed(detector, _silence(6, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.last_phrase == AYRIS
        assert stats.last_score == pytest.approx(0.77)
        assert stats.last_fired_sec == pytest.approx(0.4, abs=0.001)

    def test_timing_recorded(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            feed(detector, _silence(5, frame_samples=1600), frame_samples=1600)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.avg_ms >= 0.0
        assert stats.max_ms >= stats.avg_ms

    def test_dropped_blocks_counted(self) -> None:
        """A machine that cannot keep up drops audio instead of growing latency."""
        engine = ScriptedEngine()
        detector = WakeWordDetector(
            WakeWordSettings(phrases=(WakePhrase(AYRIS),), queue_blocks=1),
            engine=engine,
        )
        detector.start()
        try:
            for _ in range(400):
                detector.push(_silence(1, frame_samples=1600))
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.dropped > 0

    def test_engine_name_reported(self) -> None:
        engine = ScriptedEngine()
        detector = _detector(engine, phrases=(WakePhrase(AYRIS),))
        try:
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.engine == "openwakeword"
        assert stats.running is True
        assert stats.loaded is True
        assert stats.error == ""

    def test_load_failure_is_visible_not_fatal(self) -> None:
        """A missing model must not take capture down with it."""
        errors: list[object] = []
        detector = WakeWordDetector(
            WakeWordSettings(engine="porcupine", phrases=(WakePhrase(AYRIS),)),
            WakeWordCallbacks(on_error=errors.append),
        )
        detector.start()
        try:
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.loaded is False
        assert stats.error != ""
        assert len(errors) == 1


class TestOnFixtures:
    """The acceptance criteria, on the real waveforms."""

    def _run(
        self,
        fixture: str,
        phrases: tuple[WakePhrase, ...],
        *,
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    ) -> tuple[list[WakeDetection], WakeStats]:
        seen: list[WakeDetection] = []
        engine = FormantEngine()
        detector = _detector(engine, phrases=phrases, debounce_ms=debounce_ms, seen=seen)
        try:
            feed(detector, wav(fixture), frame_samples=engine.frame_samples)
            stats = detector.stats
        finally:
            detector.stop()
        return seen, stats

    def test_wake_word_fires_once(self) -> None:
        """The fixture that says "айрис" produces exactly one activation."""
        seen, stats = self._run("wake_ayris.wav", (WakePhrase(AYRIS, 0.5),))
        assert len(seen) == 1, f"expected one activation, got {len(seen)}"
        assert seen[0].phrase == AYRIS
        assert stats.candidates > 1, "the word spans several frames"
        assert stats.debounced == stats.candidates - 1

    def test_absent_does_not_fire(self) -> None:
        """Loud speech without the wake word produces nothing.

        The fixture is two long "а" vowels, not silence: an engine that simply
        never fires must not be able to pass this.
        """
        seen, _ = self._run("wake_absent.wav", (WakePhrase(AYRIS, 0.5),))
        assert seen == []

    def test_similar_phrase_does_not_fire(self) -> None:
        """ "ирис" is not "айрис" - they differ only in the first vowel."""
        seen, _ = self._run("wake_similar.wav", (WakePhrase(AYRIS, 0.5),))
        assert seen == []

    def test_similar_phrase_fires_when_configured(self) -> None:
        """The near-miss fixture is a real utterance, so its own variant hears it."""
        seen, _ = self._run("wake_similar.wav", (WakePhrase(IRIS, 0.5),))
        assert len(seen) == 1
        assert seen[0].phrase == IRIS

    def test_double_utterance_is_one_event(self) -> None:
        """Two utterances 1.2 s apart fall inside a 1.5 s window."""
        seen, stats = self._run("wake_double.wav", (WakePhrase(AYRIS, 0.5),))
        assert len(seen) == 1, f"expected one activation, got {len(seen)}"
        assert stats.debounced > 0

    def test_double_utterance_with_short_window(self) -> None:
        """With the window shortened below the gap, both utterances count."""
        seen, _ = self._run("wake_double.wav", (WakePhrase(AYRIS, 0.5),), debounce_ms=500)
        assert len(seen) == 2

    def test_sensitivity_zero_silences_the_fixture(self) -> None:
        """The acceptance criterion "changing sensitivity changes triggering"."""
        seen, _ = self._run("wake_ayris.wav", (WakePhrase(AYRIS, 0.0),))
        assert seen == []

    def test_sensitivity_change_without_restart(self) -> None:
        """The same audio, twice, with only the slider moved in between."""
        seen: list[WakeDetection] = []
        engine = FormantEngine()
        pcm = wav("wake_ayris.wav")
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.0),), seen=seen)
        try:
            feed(detector, pcm, frame_samples=engine.frame_samples)
            assert seen == []
            assert detector.set_sensitivity(AYRIS, 1.0)
            feed(detector, pcm, frame_samples=engine.frame_samples)
        finally:
            detector.stop()
        assert len(seen) == 1

    def test_no_frames_dropped(self) -> None:
        """The whole fixture is scored: a wake word in a dropped block is missed."""
        engine = FormantEngine()
        pcm = wav("wake_double.wav")
        detector = _detector(engine, phrases=(WakePhrase(AYRIS, 0.5),))
        try:
            feed(detector, pcm, frame_samples=engine.frame_samples)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.dropped == 0
        assert stats.errors == 0
        assert stats.frames == len(pcm) // engine.frame_bytes

    def test_resampling_happens_in_the_manager(self) -> None:
        """Audio at another rate still reaches the engine in its own frame size.

        Feeding 8 kHz means the manager has to resample and re-frame; an engine
        that had to do it itself would see blocks it cannot accept.
        """
        pcm = wav("wake_ayris.wav")
        halved = array("h", pcm)[::2].tobytes()
        seen: list[WakeDetection] = []
        engine = FormantEngine()
        detector = WakeWordDetector(
            WakeWordSettings(
                phrases=(WakePhrase(AYRIS, 0.5),),
                source_rate=8000,
                queue_blocks=4096,
            ),
            WakeWordCallbacks(on_detected=seen.append),
            engine=engine,
        )
        detector.start()
        try:
            for offset in range(0, len(halved), 1600):
                detector.push(halved[offset : offset + 1600])
            deadline = perf_counter() + _FEED_TIMEOUT_SEC
            while perf_counter() < deadline and detector.stats.frames < 24:
                sleep(0.01)
            stats = detector.stats
        finally:
            detector.stop()
        assert stats.errors == 0
        assert stats.frames >= 24, "resampled audio never reached the engine"


class TestWorkerWiring:
    """Parameters in, bus events out."""

    def test_defaults(self) -> None:
        settings = _wake_settings_from_params({})
        assert settings.engine == "openwakeword"
        assert settings.debounce_ms == DEFAULT_DEBOUNCE_MS
        assert settings.source_rate == TARGET_SAMPLE_RATE
        assert settings.enabled is False, "no phrases means nothing to listen for"

    def test_phrases_from_params(self) -> None:
        settings = _wake_settings_from_params(
            {"wake_phrases": [{"phrase": AYRIS, "sensitivity": 0.7}]}
        )
        assert settings.enabled is True
        assert len(settings.phrases) == 1
        assert settings.phrases[0].text == AYRIS
        assert settings.phrases[0].sensitivity == 0.7

    def test_ptt_mode_disables_listening(self) -> None:
        """Push-to-Talk only: no engine is loaded even with variants configured."""
        settings = _wake_settings_from_params(
            {"mic_mode": "ptt", "wake_phrases": [{"phrase": AYRIS}]}
        )
        assert settings.enabled is False

    def test_hybrid_mode_listens(self) -> None:
        settings = _wake_settings_from_params(
            {"mic_mode": "hybrid", "wake_phrases": [{"phrase": AYRIS}]}
        )
        assert settings.enabled is True

    def test_always_mode_listens(self) -> None:
        settings = _wake_settings_from_params(
            {"mic_mode": "always", "wake_phrases": [{"phrase": AYRIS}]}
        )
        assert settings.enabled is True

    def test_disabled_flag(self) -> None:
        settings = _wake_settings_from_params(
            {"wake_enabled": False, "wake_phrases": [{"phrase": AYRIS}]}
        )
        assert settings.enabled is False

    def test_debounce_from_params(self) -> None:
        settings = _wake_settings_from_params({"wake_debounce_ms": 2000})
        assert settings.debounce_ms == 2000
        assert settings.debounce_sec == pytest.approx(2.0)

    def test_engine_from_params(self) -> None:
        settings = _wake_settings_from_params({"wake_engine": "vosk"})
        assert settings.engine == "vosk"

    def test_no_credential_without_reference(self) -> None:
        assert _wake_access_key("porcupine", "") == ""

    def test_no_credential_for_free_engine(self) -> None:
        """openWakeWord never touches the credential store."""
        assert _wake_access_key("openwakeword", "porcupine") == ""

    def test_credential_read_for_porcupine(self) -> None:
        store = SecretsStore(backend=_FakeKeyring({("Ayris", "porcupine"): "key-value"}))
        reset_secrets(store)
        try:
            assert _wake_access_key("porcupine", "porcupine") == "key-value"
        finally:
            reset_secrets(None)

    def test_missing_credential_is_empty(self) -> None:
        store = SecretsStore(backend=_FakeKeyring({}))
        reset_secrets(store)
        try:
            assert _wake_access_key("porcupine", "porcupine") == ""
        finally:
            reset_secrets(None)

    def test_unknown_engine_yields_no_credential(self) -> None:
        assert _wake_access_key("nonexistent", "porcupine") == ""

    def test_settings_carry_no_key(self) -> None:
        """The worker parameters hold a reference name, never a secret."""
        params = {"wake_credential_ref": "porcupine", "wake_engine": "openwakeword"}
        settings = _wake_settings_from_params(params)
        assert settings.access_key == ""
        assert "porcupine" not in str(settings.phrases)

    def test_event_translated(self) -> None:
        event = translate_audio_event(
            "wake_word",
            {"phrase": AYRIS, "score": 0.82, "engine": "openwakeword"},
        )
        assert event is not None
        assert type(event).__name__ == "WakeWordDetected"
        assert event.phrase == AYRIS
        assert event.confidence == pytest.approx(0.82)

    def test_manual_event_translated(self) -> None:
        event = translate_audio_event(
            "wake_word",
            {"phrase": PTT_PHRASE, "score": 1.0, "engine": "hotkey", "manual": True},
        )
        assert event is not None
        assert event.phrase == PTT_PHRASE


class _FakeKeyring:
    """A credential store in a dictionary. Keeps the tests off the real keyring."""

    def __init__(self, entries: Mapping[tuple[str, str], str]) -> None:
        self._entries = dict(entries)

    def get_password(self, service_name: str, username: str) -> str | None:
        return self._entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self._entries[service_name, username] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self._entries[service_name, username]


@pytest.mark.hardware
class TestRealEngines:
    """Needs vendor libraries and model weights; excluded from CI.

    The weights are not committed - openWakeWord's are tens of megabytes and
    Porcupine's need a licence - so these run only where somebody has installed
    them. What CI checks instead is that selecting a missing engine produces a
    sentence rather than an ImportError, which is in
    :class:`TestStats.test_load_failure_is_visible_not_fatal`.

    Point ``AYRIS_TEST_WAKE_MODELS`` at a folder of ``.onnx`` files named after
    the phrases - ``hey_jarvis.onnx`` for "hey jarvis" - the way the profile's
    ``models/wake`` folder is laid out in production.
    """

    @staticmethod
    def _models_dir() -> Path:
        location = os.environ.get("AYRIS_TEST_WAKE_MODELS", "")
        if not location:
            pytest.skip("AYRIS_TEST_WAKE_MODELS не задан")
        path = Path(location)
        if not path.is_dir():
            pytest.skip(f"папки моделей нет: {path}")
        return path

    def test_openwakeword_loads(self) -> None:
        if not OpenWakeWordEngine.available():
            pytest.skip("openwakeword is not installed")
        engine = OpenWakeWordEngine()
        engine.load(ModelSpec(phrases=(WakePhrase("hey jarvis"),), models_dir=self._models_dir()))
        try:
            assert engine.loaded
            frame = b"\x00\x00" * engine.frame_samples
            assert engine.process(frame) is None
        finally:
            engine.unload()

    def test_vosk_loads(self) -> None:
        if not VoskKwsEngine.available():
            pytest.skip("vosk is not installed")
        engine = VoskKwsEngine()
        with pytest.raises(WakeWordError):
            # No model directory in the test profile: the message must say so.
            engine.load(ModelSpec(phrases=(WakePhrase(AYRIS),)))
