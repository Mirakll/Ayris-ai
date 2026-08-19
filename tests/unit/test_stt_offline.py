"""Task 10: offline recognition — the engine contract, and the worker around it.

There is no model on the runner and there never will be: a Vosk model is fifty
megabytes and a Whisper one is a gigabyte, so nothing here loads a real one.
That is not the compromise it sounds like.  Almost everything task 10 asks for
lives *outside* the model: the shape of a result, resampling at the worker
boundary, lazy loading, the idle timeout, the RAM cap, the timing numbers.  All
of that is exercised here on a stub engine that returns a fixed transcript, and a
stub is in fact the stricter test — it lets an assertion say "the worker called
``load`` exactly once", which no real model would let us see.

What genuinely needs a model carries ``@pytest.mark.hardware`` and is skipped
everywhere except a developer's machine that has one.

The audio comes from ``tests/fixtures/audio/stt_*.wav``, synthesised by
``make_fixtures.py`` next to them: a command, a sentence and an empty room.  The
recogniser tests care about their *lengths* — under the minimum, around a
command, long enough for a real-time factor to mean something.

Groups:

* :class:`TestFixtures` — the three ``stt_*`` files are present and the right length.
* :class:`TestAudioBuffer` — the container every engine is fed through.
* :class:`TestTranscriptResult` — the wire format, and RTF arithmetic.
* :class:`TestSttOptions` — worker params to engine options, tolerantly.
* :class:`TestEngineRegistry` — names, lazy resolution, the optional engine.
* :class:`TestModelSize` — measuring a directory on disk, recursively.
* :class:`TestEngineContract` — what :class:`SttEngine` guarantees a subclass.
* :class:`TestVoskEngine` / :class:`TestWhisperEngine` — the parts of the two
  real engines that are pure functions of their inputs, not needing a model.
* :class:`TestWorkerAudio` — shared memory in, resampled mono out.
* :class:`TestWorkerLifecycle` — lazy load, idle unload, reload, RAM cap.
* :class:`TestIdleTimeout` — the timeout itself, without sleeping through it.
* :class:`TestWorkerMetrics` — the numbers DevTools and the pipeline log show.
* :class:`TestEventTranslation` — what a worker event becomes on the bus.
* :class:`TestRegistryWiring` — the supervisor can actually find this worker.
"""

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import types
import wave
from array import array
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from ayris.audio.stt import faster_whisper_engine
from ayris.audio.stt.base import (
    DEFAULT_LANGUAGE,
    MIN_SPEECH_MS,
    STT_SAMPLE_RATE,
    AudioBuffer,
    SttEngine,
    SttOptions,
    TranscriptResult,
    TranscriptSegment,
    create_engine,
    engine_class,
    engine_names,
    estimate_model_bytes,
)
from ayris.audio.stt.faster_whisper_engine import (
    CPU_COMPUTE_TYPE,
    CUDA_COMPUTE_TYPE,
    FasterWhisperEngine,
    _is_hallucination,
    _logprob_to_confidence,
    _mean_confidence,
    cuda_available,
)
from ayris.audio.stt.vosk_engine import VoskSttEngine
from ayris.core.config import Settings
from ayris.core.errors import SttError
from ayris.workers.base import worker_methods
from ayris.workers.protocol import SharedAudioBlock
from ayris.workers.registry import WorkerKind, event_translator, worker_type
from ayris.workers.stt_worker import SttWorker, translate_stt_event

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ayris.core.models import JsonObject

    #: The ``ascii_weights`` fixture: a model directory in, a path a native
    #: library can actually open out.
    _Weights = Callable[[Path], Path]

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: ``pythonpath`` from ``pyproject.toml`` is a pytest setting, so a subprocess
#: that has to import ``ayris`` needs the directory spelled out.
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

#: Length of each task 10 fixture, in milliseconds.  Asserted rather than
#: assumed: every test below reasons about durations.
STT_DURATIONS = {
    "stt_command.wav": 1020,
    "stt_phrase.wav": 2400,
    "stt_silence.wav": 1000,
}


#: A final result the way Vosk actually shapes one: the text, and a ``result``
#: array of per-word timings and confidences.
_VOSK_FINAL = (
    '{"text": "привет", "result": ['
    '{"word": "привет", "start": 0.10, "end": 0.40, "conf": 0.95}]}'
)


def wav(name: str) -> AudioBuffer:
    """Load a fixture as an :class:`AudioBuffer`."""
    return AudioBuffer.from_wav(FIXTURES / name)


def _downloaded(variable: str) -> Path:
    """The model directory named by an ``AYRIS_TEST_*`` variable.

    Skips rather than fails when the variable is unset or points nowhere: the
    ``hardware`` tests are for a machine that has the weights, and on every
    other one their absence is the expected state, not a broken run.
    """
    location = os.environ.get(variable, "")
    if not location:
        pytest.skip(f"{variable} не задан")
    path = Path(location)
    if not path.exists():
        pytest.skip(f"модели нет: {path}")
    return path


def tone(ms: int, amplitude: int = 9000, *, rate: int = STT_SAMPLE_RATE) -> AudioBuffer:
    """A loud sine, for tests that need audible audio and not a fixture."""
    count = rate * ms // 1000
    samples = array(
        "h",
        (int(amplitude * math.sin(2.0 * math.pi * 220.0 * index / rate)) for index in range(count)),
    )
    return AudioBuffer(samples.tobytes(), sample_rate=rate, channels=1)


def stereo(mono: AudioBuffer) -> AudioBuffer:
    """The same signal on both channels, for tests about the downmix.

    Both channels carry it rather than one, because averaging a signal against
    its own inverse produces digital silence and the test would then be about
    the silence check instead of about resampling.
    """
    samples = array("h")
    samples.frombytes(mono.pcm)
    doubled = array("h")
    for sample in samples:
        doubled.append(sample)
        doubled.append(sample)
    return AudioBuffer(doubled.tobytes(), sample_rate=mono.sample_rate, channels=2)


class StubEngine(SttEngine):
    """An engine that loads instantly and always says the same thing.

    Counts its own calls, which is the point: the worker's laziness and its idle
    unload are claims about *how many times* ``load`` and ``unload`` ran, and
    only a stub can answer that.
    """

    name: ClassVar[str] = "stub"
    supports_streaming: ClassVar[bool] = False
    memory_factor: ClassVar[float] = 1.0

    __slots__ = ("loads", "seen", "text", "unloads")

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0
        self.unloads = 0
        self.text = "привет"
        self.seen: list[AudioBuffer] = []

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("ru",)

    @property
    def loaded(self) -> bool:
        return self._options is not None

    def load(self, model_path: Path, options: SttOptions) -> None:
        self.loads += 1
        self._model_path = model_path
        self._options = options

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        options = self._require_loaded()
        prepared = self._prepare(audio)
        self.seen.append(prepared)
        if prepared.duration_ms < options.min_speech_ms or prepared.is_silent():
            return TranscriptResult.empty(
                engine=self.name, duration_ms=prepared.duration_ms, model=self.model_name
            )
        return TranscriptResult(
            text=self.text,
            confidence=0.9,
            segments=(
                TranscriptSegment(text=self.text, start_ms=0.0, end_ms=prepared.duration_ms),
            ),
            duration_ms=prepared.duration_ms,
            engine=self.name,
            model=self.model_name,
            inference_ms=12.5,
        )

    def unload(self) -> None:
        self.unloads += 1
        self._options = None
        self._model_path = None


class FakeContext:
    """Enough of :class:`~ayris.workers.base.WorkerContext` to drive the worker.

    Same shape as the one in :mod:`tests.unit.test_audio_capture` and for the
    same reason: this file's subject is recognition, and the process machinery
    has its own tests in :mod:`tests.unit.test_workers`.
    """

    def __init__(self, params: JsonObject | None = None) -> None:
        self.name = "stt"
        self.kind = "stt"
        self._params: JsonObject = dict(params or {})
        self.events: list[tuple[str, JsonObject]] = []
        self.cancelled = False

    @property
    def params(self) -> JsonObject:
        return self._params

    @property
    def stopping(self) -> bool:
        return False

    def check_cancelled(self) -> None:
        return None

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        self.events.append((kind, dict(payload or {})))

    def logger(self, suffix: str = "") -> Any:
        import logging

        return logging.getLogger(f"ayris.workers.stt.{suffix}" if suffix else "ayris.workers.stt")

    def events_of(self, kind: str) -> list[JsonObject]:
        """Every payload emitted under ``kind``, in order."""
        return [payload for name, payload in self.events if name == kind]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A directory that looks like a model and weighs a known amount."""
    directory = tmp_path / "models" / "stub-ru"
    (directory / "am").mkdir(parents=True)
    (directory / "am" / "final.mdl").write_bytes(b"\0" * 4096)
    return directory


@pytest.fixture
def worker(model_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[SttWorker]:
    """A started worker wired to :class:`StubEngine` and a fake model on disk.

    The engine is injected by patching ``create_engine`` in the worker's
    namespace rather than by adding a "stub" entry to ``ENGINE_ENTRYPOINTS``:
    the registry is production data and a test that mutates it leaks into every
    test that reads :func:`engine_names` afterwards.
    """
    engine = StubEngine()
    monkeypatch.setattr("ayris.workers.stt_worker.create_engine", lambda _name: engine)
    context = FakeContext(
        {
            "offline_engine": "stub",
            "offline_model": str(model_dir),
            "language": "ru",
            "model_idle_sec": 0.0,
            "ram_limit_mb": 0,
        }
    )
    instance = SttWorker(context)  # type: ignore[arg-type]
    instance.on_start()
    yield instance
    instance.on_stop()


def engine_of(instance: SttWorker) -> StubEngine:
    """The stub the worker loaded, for assertions about its call counts."""
    loaded = instance._engine
    assert isinstance(loaded, StubEngine)
    return loaded


def transcribe(instance: SttWorker, audio: AudioBuffer, **params: object) -> JsonObject:
    """Call ``transcribe`` the way the supervisor does: audio in shared memory."""
    with SharedAudioBlock.create(
        audio.pcm, sample_rate=audio.sample_rate, channels=audio.channels
    ) as block:
        return instance.transcribe(block.chunk.to_params(dict(params)))


# ----------------------------------------------------------------------
# fixtures on disk
# ----------------------------------------------------------------------


class TestFixtures:
    """The committed WAVs are the premise of everything below."""

    @pytest.mark.parametrize(("name", "duration_ms"), sorted(STT_DURATIONS.items()))
    def test_every_fixture_has_the_expected_length(self, name: str, duration_ms: int):
        assert round(wav(name).duration_ms) == duration_ms

    def test_the_command_is_long_enough_to_reach_a_model(self):
        """Below ``min_speech_ms`` an engine answers empty without decoding."""
        assert wav("stt_command.wav").duration_ms > MIN_SPEECH_MS

    def test_speech_and_silence_differ_by_level(self):
        speech = wav("stt_phrase.wav")
        room = wav("stt_silence.wav")
        assert speech.rms_dbfs() > room.rms_dbfs() + 20.0
        assert not speech.is_silent()
        assert room.is_silent()

    def test_every_fixture_is_mono_at_the_recognition_rate(self):
        for name in STT_DURATIONS:
            buffer = wav(name)
            assert buffer.channels == 1
            assert buffer.sample_rate == STT_SAMPLE_RATE


# ----------------------------------------------------------------------
# the audio container
# ----------------------------------------------------------------------


class TestAudioBuffer:
    """What every engine is fed through, and the checks that keep it honest."""

    def test_duration_and_frames_agree(self):
        buffer = tone(500)
        assert buffer.frames == STT_SAMPLE_RATE // 2
        assert buffer.duration_ms == pytest.approx(500.0)

    def test_an_empty_buffer_is_valid_and_says_so(self):
        buffer = AudioBuffer(b"")
        assert buffer.is_empty
        assert buffer.duration_ms == 0.0
        assert buffer.rms_dbfs() == -math.inf

    def test_a_truncated_frame_is_rejected(self):
        """A half sample would shift every timestamp in the result."""
        with pytest.raises(SttError, match="whole number"):
            AudioBuffer(b"\x00\x01\x02")

    def test_an_impossible_rate_is_rejected(self):
        with pytest.raises(SttError, match="sample rate"):
            AudioBuffer(b"", sample_rate=0)

    def test_more_than_two_channels_is_rejected(self):
        with pytest.raises(SttError, match="mono or stereo"):
            AudioBuffer(b"", channels=6)

    def test_stereo_is_averaged_into_mono(self):
        stereo = AudioBuffer(array("h", [1000, 2000, -1000, -2000]).tobytes(), channels=2)
        mono = stereo.to_mono()
        assert mono.channels == 1
        assert list(mono.samples()) == [1500, -1500]

    def test_to_mono_on_mono_returns_the_same_object(self):
        """No copy for the common case: every phrase goes through this."""
        buffer = tone(50)
        assert buffer.to_mono() is buffer

    def test_resampling_preserves_duration(self):
        """The point of resampling: the same speech, at another rate."""
        original = tone(400, rate=48000)
        converted = original.resampled_to(STT_SAMPLE_RATE)
        assert converted.sample_rate == STT_SAMPLE_RATE
        assert converted.duration_ms == pytest.approx(400.0, abs=2.0)

    def test_resampling_to_the_same_rate_is_a_no_op(self):
        buffer = tone(50)
        assert buffer.resampled_to(STT_SAMPLE_RATE) is buffer

    def test_resampling_stereo_is_refused_rather_than_guessed(self):
        stereo = AudioBuffer(b"\x00" * 8, sample_rate=48000, channels=2)
        with pytest.raises(SttError, match="mono"):
            stereo.resampled_to(STT_SAMPLE_RATE)

    def test_prepared_for_does_both_steps(self):
        stereo = AudioBuffer(
            array("h", [1000, 1000] * 4800).tobytes(), sample_rate=48000, channels=2
        )
        prepared = stereo.prepared_for(STT_SAMPLE_RATE)
        assert prepared.channels == 1
        assert prepared.sample_rate == STT_SAMPLE_RATE

    def test_floats_land_in_the_range_whisper_wants(self):
        floats = tone(20, amplitude=32000).floats()
        assert len(floats) == STT_SAMPLE_RATE * 20 // 1000
        assert all(-1.0 <= value <= 1.0 for value in floats)

    def test_silence_detection_catches_a_quiet_room(self):
        assert wav("stt_silence.wav").is_silent()

    def test_silence_detection_catches_a_buffer_that_is_merely_short(self):
        """A key released the instant it was pressed is not a phrase."""
        assert tone(MIN_SPEECH_MS - 50).is_silent()

    def test_a_loud_short_buffer_is_still_rejected_by_length(self):
        assert tone(10, amplitude=30000).is_silent()

    def test_reading_a_wav_that_is_not_16_bit_is_a_typed_error(self, tmp_path: Path):
        path = tmp_path / "eight.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(1)
            handle.setframerate(STT_SAMPLE_RATE)
            handle.writeframes(b"\x80" * 100)
        with pytest.raises(SttError, match="16-bit"):
            AudioBuffer.from_wav(path)

    def test_reading_a_missing_wav_is_a_typed_error(self, tmp_path: Path):
        with pytest.raises(SttError, match="cannot read"):
            AudioBuffer.from_wav(tmp_path / "nope.wav")


# ----------------------------------------------------------------------
# the result
# ----------------------------------------------------------------------


class TestTranscriptResult:
    """The wire format, and the one number that says whether this machine copes."""

    def test_an_empty_result_is_empty(self):
        result = TranscriptResult.empty(engine="stub", duration_ms=500.0)
        assert result.is_empty
        assert result.text == ""
        assert result.confidence == 0.0
        assert result.duration_ms == 500.0

    def test_whitespace_only_text_counts_as_empty(self):
        assert TranscriptResult(text="   ").is_empty

    def test_real_time_factor_is_inference_over_audio(self):
        result = TranscriptResult(text="да", duration_ms=1000.0, inference_ms=250.0)
        assert result.real_time_factor == pytest.approx(0.25)

    def test_real_time_factor_without_audio_is_zero_not_an_exception(self):
        assert TranscriptResult(text="да", inference_ms=10.0).real_time_factor == 0.0

    def test_word_count_reads_the_text(self):
        assert TranscriptResult(text="открой  браузер").word_count == 2

    def test_with_timing_returns_a_new_result(self):
        """Frozen: the engine builds a result, the caller stamps the clock on it."""
        original = TranscriptResult(text="да")
        timed = original.with_timing(inference_ms=30.0, duration_ms=600.0)
        assert original.inference_ms == 0.0
        assert timed.inference_ms == 30.0
        assert timed.duration_ms == 600.0
        assert timed.text == "да"

    def test_a_round_trip_through_params_preserves_everything(self):
        """The worker answers with ``to_params``; the manager rebuilds from it."""
        result = TranscriptResult(
            text="открой браузер",
            confidence=0.82,
            segments=(
                TranscriptSegment(text="открой", start_ms=0.0, end_ms=400.0, confidence=0.9),
                TranscriptSegment(text="браузер", start_ms=400.0, end_ms=900.0, confidence=0.7),
            ),
            language="ru",
            duration_ms=1000.0,
            engine="vosk",
            device="cpu",
            model="vosk-model-small-ru-0.22",
            inference_ms=140.0,
        )
        restored = TranscriptResult.from_params(result.to_params())
        assert restored == result

    def test_params_are_plain_json_types(self):
        """A dataclass over a pipe is a version-compatibility liability."""
        params = TranscriptResult(
            text="да", segments=(TranscriptSegment(text="да", start_ms=0.0, end_ms=100.0),)
        ).to_params()
        assert isinstance(params["segments"], list)
        assert all(isinstance(item, dict) for item in params["segments"])

    def test_from_params_tolerates_an_older_supervisor(self):
        """Missing keys fall back rather than raising: versions drift."""
        restored = TranscriptResult.from_params({"text": "да"})
        assert restored.text == "да"
        assert restored.language == DEFAULT_LANGUAGE
        assert restored.segments == ()

    def test_from_params_ignores_a_malformed_segment_list(self):
        restored = TranscriptResult.from_params({"text": "да", "segments": "не список"})
        assert restored.segments == ()

    def test_segment_duration(self):
        segment = TranscriptSegment(text="да", start_ms=100.0, end_ms=450.0)
        assert segment.duration_ms == pytest.approx(350.0)


# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


class TestSttOptions:
    """Worker params in, engine options out — tolerantly, because versions drift."""

    def test_defaults_are_the_settings_defaults(self):
        options = SttOptions()
        assert options.language == DEFAULT_LANGUAGE
        assert options.min_speech_ms == MIN_SPEECH_MS
        assert options.gpu == "auto"

    def test_params_are_read(self):
        options = SttOptions.from_params(
            {"language": "en", "threads": 4, "gpu": "cuda", "min_confidence": 0.7}
        )
        assert options.language == "en"
        assert options.threads == 4
        assert options.gpu == "cuda"
        assert options.min_confidence == pytest.approx(0.7)

    def test_absent_params_fall_back(self):
        options = SttOptions.from_params({})
        assert options == SttOptions()

    def test_a_string_where_a_number_was_expected_falls_back(self):
        """An older supervisor sending a string must not crash a worker start."""
        options = SttOptions.from_params({"threads": "много", "beam_size": None})
        assert options.threads == SttOptions().threads
        assert options.beam_size == SttOptions().beam_size

    def test_booleans_are_not_accepted_as_numbers(self):
        """``True`` as a thread count is a bug, not a value of one."""
        assert SttOptions.from_params({"threads": True}).threads == SttOptions().threads

    def test_threads_are_clamped_to_at_least_one(self):
        assert SttOptions.from_params({"threads": 0}).threads == 1

    def test_an_empty_language_falls_back_rather_than_disabling_it(self):
        assert SttOptions.from_params({"language": ""}).language == DEFAULT_LANGUAGE

    def test_extras_survive(self):
        options = SttOptions.from_params({"extra": {"vendor_flag": "on"}})
        assert options.option("vendor_flag") == "on"
        assert options.option("missing", "по умолчанию") == "по умолчанию"


# ----------------------------------------------------------------------
# the registry
# ----------------------------------------------------------------------


class TestEngineRegistry:
    """Names in, classes out — without importing a vendor library to find out."""

    def test_the_two_first_class_engines_are_always_offered(self):
        """Greyed out when missing, but present: a user can see what exists."""
        names = engine_names()
        assert "vosk" in names
        assert "whisper" in names

    def test_the_optional_engine_is_hidden_until_it_is_installed(self):
        """whisper.cpp has no wheel; offering it would be offering a dead end."""
        if engine_class("whispercpp").available():  # pragma: no cover - not on CI
            pytest.skip("whisper-cpp-python установлен")
        assert "whispercpp" not in engine_names()

    def test_available_only_drops_what_cannot_run(self):
        for name in engine_names(available_only=True):
            assert engine_class(name).available()

    def test_an_unknown_name_is_a_typed_error_and_not_a_downgrade(self):
        """A user who picked Whisper and silently got Vosk debugs the wrong thing."""
        with pytest.raises(SttError, match="unknown stt engine"):
            engine_class("нет такого")

    def test_resolving_a_class_does_not_construct_it(self):
        assert isinstance(engine_class("vosk"), type)

    def test_create_engine_builds_an_instance_without_loading_a_model(self):
        engine = create_engine("vosk")
        assert isinstance(engine, SttEngine)
        assert not engine.loaded
        assert engine.model_path is None

    def test_every_registered_engine_declares_its_vendor_package(self):
        """``available()`` and the error messages both need it."""
        for name in engine_names():
            klass = engine_class(name)
            assert klass.package
            assert klass.module

    def test_availability_does_not_import_the_library(self):
        """The settings window asks this; importing CTranslate2 to answer would hang it."""
        import sys

        engine_class("whisper").available()
        assert "faster_whisper" not in sys.modules


class TestModelSize:
    """The input to the RAM check."""

    def test_a_file_is_measured(self, tmp_path: Path):
        path = tmp_path / "model.bin"
        path.write_bytes(b"\0" * 2048)
        assert estimate_model_bytes(path) == 2048

    def test_a_directory_is_measured_recursively(self, tmp_path: Path):
        (tmp_path / "am").mkdir()
        (tmp_path / "am" / "final.mdl").write_bytes(b"\0" * 1000)
        (tmp_path / "conf.txt").write_bytes(b"\0" * 24)
        assert estimate_model_bytes(tmp_path) == 1024

    def test_a_missing_path_measures_zero_rather_than_raising(self, tmp_path: Path):
        """Zero means "cannot tell", and the caller lets it through."""
        assert estimate_model_bytes(tmp_path / "nope") == 0


# ----------------------------------------------------------------------
# the contract
# ----------------------------------------------------------------------


class TestEngineContract:
    """What :class:`SttEngine` guarantees, checked on the stub that implements it."""

    def test_the_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            SttEngine()  # type: ignore[abstract]

    def test_a_fresh_engine_is_not_loaded(self):
        engine = StubEngine()
        assert not engine.loaded
        assert engine.options is None
        assert engine.model_name == ""

    def test_transcribing_before_loading_is_a_typed_error(self):
        """Not an AttributeError from inside a worker process."""
        with pytest.raises(SttError, match="no model is loaded"):
            StubEngine().transcribe(tone(500))

    def test_loading_records_the_model_and_the_options(self, model_dir: Path):
        engine = StubEngine()
        engine.load(model_dir, SttOptions(language="ru"))
        assert engine.loaded
        assert engine.model_path == model_dir
        assert engine.model_name == model_dir.name
        assert engine.options is not None

    def test_unloading_twice_is_safe(self, model_dir: Path):
        """The idle timer and an explicit unload can race; neither may raise."""
        engine = StubEngine()
        engine.load(model_dir, SttOptions())
        engine.unload()
        engine.unload()
        assert not engine.loaded

    def test_the_declared_sample_rate_is_the_recognition_rate(self):
        assert StubEngine().sample_rate == STT_SAMPLE_RATE

    def test_a_non_streaming_engine_refuses_streaming_rather_than_pretending(self):
        engine = StubEngine()
        with pytest.raises(SttError, match="streaming"):
            engine.start_stream()
        with pytest.raises(SttError, match="streaming"):
            engine.accept_audio(b"")
        with pytest.raises(SttError, match="streaming"):
            engine.finish_stream()

    def test_silence_comes_back_empty_and_not_as_an_exception(self, model_dir: Path):
        """The single most important thing an engine does with a silent buffer."""
        engine = StubEngine()
        engine.load(model_dir, SttOptions())
        result = engine.transcribe(wav("stt_silence.wav"))
        assert result.is_empty
        assert result.engine == "stub"
        assert result.duration_ms == pytest.approx(1000.0)

    def test_a_short_buffer_never_reaches_the_model(self, model_dir: Path):
        engine = StubEngine()
        engine.load(model_dir, SttOptions(min_speech_ms=400))
        assert engine.transcribe(tone(200)).is_empty

    def test_a_real_phrase_comes_back_with_text_and_timings(self, model_dir: Path):
        engine = StubEngine()
        engine.load(model_dir, SttOptions())
        result = engine.transcribe(wav("stt_command.wav"))
        assert result.text == "привет"
        assert result.confidence > 0.0
        assert result.duration_ms == pytest.approx(1020.0, abs=1.0)
        assert result.segments

    def test_input_is_normalised_before_the_engine_sees_it(self, model_dir: Path):
        """``_prepare`` is why an engine may assume 16 kHz mono."""
        engine = StubEngine()
        engine.load(model_dir, SttOptions())
        stereo = AudioBuffer(
            array("h", [6000, 6000] * 24000).tobytes(), sample_rate=48000, channels=2
        )
        engine.transcribe(stereo)
        seen = engine.seen[-1]
        assert seen.channels == 1
        assert seen.sample_rate == STT_SAMPLE_RATE

    def test_a_model_path_that_is_not_there_is_a_russian_error(self, tmp_path: Path):
        engine = create_engine("vosk")
        with pytest.raises(SttError) as excinfo:
            engine.load(tmp_path / "missing", SttOptions())
        assert excinfo.value.user_message
        assert excinfo.value.user_message != str(excinfo.value)

    def test_a_directory_without_the_marker_is_refused(self, tmp_path: Path):
        """A folder holding a Whisper model is not a Vosk model."""
        (tmp_path / "not-a-model").mkdir()
        (tmp_path / "not-a-model" / "model.bin").write_bytes(b"\0")
        engine = create_engine("vosk")
        with pytest.raises(SttError):
            engine.load(tmp_path / "not-a-model", SttOptions())


# ----------------------------------------------------------------------
# the two real engines, without their vendor libraries
# ----------------------------------------------------------------------
#
# Neither engine is *loaded* here, because loading is what needs the wheel and
# the model.  Both are driven by assigning the one private attribute ``load``
# would have set - the recogniser, the model handle - and letting the real
# ``transcribe`` run on top.  That covers every line between the audio arriving
# and the ``TranscriptResult`` leaving, which is where all the engine-specific
# behaviour lives: Vosk's JSON, Whisper's log-probabilities and its
# hallucinations.  Setting ``_options`` is what makes ``_require_loaded`` pass.


class FakeRecognizer:
    """Enough of ``vosk.KaldiRecognizer`` to run a stream through."""

    def __init__(self, final: str) -> None:
        self.final = final
        self.accepted: list[bytes] = []
        self.resets = 0

    def Reset(self) -> None:  # noqa: N802 - vendor spelling
        self.resets += 1

    def AcceptWaveform(self, pcm: bytes) -> bool:  # noqa: N802 - vendor spelling
        self.accepted.append(bytes(pcm))
        return True

    def PartialResult(self) -> str:  # noqa: N802 - vendor spelling
        return '{"partial": "при"}'

    def FinalResult(self) -> str:  # noqa: N802 - vendor spelling
        return self.final


def vosk_ready(final: str = _VOSK_FINAL, **options: object) -> VoskSttEngine:
    """A Vosk engine with a fake recogniser in place of a loaded model."""
    engine = VoskSttEngine()
    engine._options = SttOptions(**options)  # type: ignore[arg-type]
    engine._language = "ru"
    engine._recognizer = FakeRecognizer(final)
    return engine


class TestVoskEngine:
    """Vosk's own logic: where its model lives, and what its JSON means."""

    def test_load_refuses_a_path_the_library_cannot_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A Cyrillic model folder is rejected before Kaldi is handed the bytes.

        Vosk encodes the path as UTF-8 and Kaldi reads it back through the ANSI
        code page, so what actually happens is "Failed to create a model" - a
        message that sends the user looking for a corrupt download.  When Windows
        has no 8.3 spelling to offer, the engine has to say what is really wrong,
        and it has to say it without calling in.
        """
        model = tmp_path / "модель"
        (model / "am").mkdir(parents=True)

        def refuse(_path: object) -> None:
            raise AssertionError("Model must not be constructed with an unopenable path")

        def stub_import(_self: object, _name: str | None = None) -> types.SimpleNamespace:
            return types.SimpleNamespace(Model=refuse)

        monkeypatch.setattr(VoskSttEngine, "_import", stub_import)
        monkeypatch.setattr("ayris.audio.stt.vosk_engine.native_path", lambda _path: None)

        engine = VoskSttEngine()
        with pytest.raises(SttError) as caught:
            engine.load(model, SttOptions(language="ru"))

        assert "модель" in caught.value.user_message
        assert "латинские" in caught.value.user_message

    def test_model_dir_finds_a_model_one_level_down(self, tmp_path: Path):
        """An archive unpacked with its folder kept must still work."""
        inner = tmp_path / "outer" / "model"
        (inner / "am").mkdir(parents=True)
        assert VoskSttEngine._model_dir(tmp_path / "outer") == inner

    def test_model_dir_leaves_a_flat_model_alone(self, tmp_path: Path):
        (tmp_path / "am").mkdir()
        assert VoskSttEngine._model_dir(tmp_path) == tmp_path

    def test_model_dir_leaves_a_folder_with_no_model_alone(self, tmp_path: Path):
        """So that ``load`` produces the error, not this helper."""
        assert VoskSttEngine._model_dir(tmp_path) == tmp_path

    def test_parse_ignores_garbage(self):
        """Vosk is a C++ library behind a thin binding; assume nothing."""
        assert VoskSttEngine._parse("not json at all") == {}
        assert VoskSttEngine._parse('"a bare string"') == {}
        assert VoskSttEngine._parse("[1, 2, 3]") == {}
        assert VoskSttEngine._parse("") == {}

    def test_parse_reads_a_final_result(self):
        assert VoskSttEngine._parse('{"text": "привет"}')["text"] == "привет"

    def test_segments_skip_words_without_text(self):
        words = [{"word": "  ", "conf": 0.9}, {"word": "да", "conf": 0.9}, "не словарь"]
        segments = VoskSttEngine._segments(words)
        assert [segment.text for segment in segments] == ["да"]

    def test_segments_convert_seconds_to_milliseconds(self):
        segment = VoskSttEngine._segments([{"word": "а", "start": 0.5, "end": 1.5}])[0]
        assert segment.start_ms == 500.0
        assert segment.end_ms == 1500.0

    def test_segments_clamp_a_confidence_out_of_range(self):
        assert VoskSttEngine._segments([{"word": "а", "conf": 1.4}])[0].confidence == 1.0
        assert VoskSttEngine._segments([{"word": "а", "conf": -2.0}])[0].confidence == 0.0

    def test_segments_survive_a_non_numeric_timing(self):
        segment = VoskSttEngine._segments([{"word": "а", "start": "нет", "conf": None}])[0]
        assert segment.start_ms == 0.0
        assert segment.confidence == 0.0

    def test_confidence_is_the_weakest_word(self):
        """A half-guessed word means the phrase was not fully heard."""
        words = [{"word": "а", "conf": 0.95}, {"word": "б", "conf": 0.4}]
        assert VoskSttEngine._confidence(words) == pytest.approx(0.4)

    def test_confidence_without_numbers_falls_back(self):
        assert VoskSttEngine._confidence([]) == pytest.approx(0.65)
        assert VoskSttEngine._confidence([{"word": "а"}]) == pytest.approx(0.65)

    def test_transcribe_hands_the_decoder_prepared_audio(self):
        """The bytes the decoder saw are exactly the prepared 16 kHz mono ones."""
        engine = vosk_ready()
        audio = wav("stt_command.wav")

        result = engine.transcribe(audio)

        recognizer = engine._recognizer
        assert recognizer.accepted == [audio.pcm]
        assert recognizer.resets == 1
        assert result.text == "привет"
        assert result.language == "ru"
        assert result.engine == "vosk"
        assert result.confidence == pytest.approx(0.95)
        assert result.segments[0].start_ms == pytest.approx(100.0)

    def test_transcribe_reports_the_duration_of_what_it_was_given(self):
        engine = vosk_ready()
        result = engine.transcribe(wav("stt_command.wav"))
        assert result.duration_ms == pytest.approx(1020.0, abs=1.0)
        assert result.inference_ms > 0.0

    def test_transcribe_resamples_before_the_decoder(self):
        """Vosk needs exactly 16 kHz; nothing downstream of here checks again."""
        engine = vosk_ready()
        engine.transcribe(stereo(tone(1000, rate=48000)))
        accepted = engine._recognizer.accepted[0]
        assert len(accepted) == 2 * STT_SAMPLE_RATE  # one second, mono int16

    def test_transcribe_answers_silence_without_waking_the_decoder(self):
        engine = vosk_ready()
        result = engine.transcribe(wav("stt_silence.wav"))
        assert engine._recognizer.accepted == []
        assert result.is_empty
        assert result.engine == "vosk"
        assert result.language == "ru"

    def test_transcribe_answers_a_too_short_buffer_the_same_way(self):
        engine = vosk_ready(min_speech_ms=500)
        assert engine.transcribe(tone(200)).is_empty
        assert engine._recognizer.accepted == []

    def test_an_empty_final_result_is_an_empty_transcript(self):
        """Vosk answers a buffer it made nothing of with ``{}``."""
        engine = vosk_ready(final="{}")
        assert engine.transcribe(wav("stt_command.wav")).is_empty

    def test_a_whitespace_final_result_is_an_empty_transcript(self):
        engine = vosk_ready(final='{"text": "   "}')
        assert engine.transcribe(wav("stt_command.wav")).is_empty

    def test_a_final_result_without_words_still_yields_text(self):
        """Some models are built without word timings.  The text is still good."""
        engine = vosk_ready(final='{"text": "привет"}')
        result = engine.transcribe(wav("stt_command.wav"))
        assert result.text == "привет"
        assert result.segments == ()
        assert result.confidence == pytest.approx(0.65)

    def test_a_decoder_that_raises_becomes_a_typed_error(self):
        engine = vosk_ready()

        def explode(_pcm: bytes) -> bool:
            raise RuntimeError("kaldi упал")

        engine._recognizer.AcceptWaveform = explode  # type: ignore[method-assign]
        with pytest.raises(SttError) as excinfo:
            engine.transcribe(wav("stt_command.wav"))
        assert excinfo.value.user_message == "Ошибка распознавания речи."

    def test_partial_text_comes_back_from_accept_audio(self):
        engine = vosk_ready()
        assert engine.accept_audio(b"\0" * 640) == "при"

    def test_unloading_twice_is_safe(self):
        engine = vosk_ready()
        engine.unload()
        engine.unload()
        assert not engine.loaded
        assert engine._recognizer is None


class FakeSegment:
    """One item of what ``WhisperModel.transcribe`` yields.

    Attributes are read with :func:`getattr` by the engine, so a plain object
    with the right names is indistinguishable from the vendor's namedtuple.
    """

    __slots__ = ("avg_logprob", "end", "no_speech_prob", "start", "text")

    def __init__(
        self,
        text: str,
        start: float,
        end: float,
        avg_logprob: float = -0.16,
        no_speech_prob: float = 0.05,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class FakeInfo:
    """``TranscriptionInfo``, of which the engine reads one attribute."""

    __slots__ = ("language",)

    def __init__(self, language: str) -> None:
        self.language = language


class FakeWhisperModel:
    """A loaded ``WhisperModel``: records its kwargs, returns fixed segments."""

    def __init__(self, segments: list[FakeSegment], language: str = "ru") -> None:
        self.segments = segments
        self.info = FakeInfo(language)
        self.calls: list[JsonObject] = []
        self.audio: list[int] = []

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[list[FakeSegment], FakeInfo]:
        self.calls.append(dict(kwargs))
        self.audio.append(len(list(audio)))
        return self.segments, self.info


def whisper_ready(
    segments: list[FakeSegment] | None = None,
    *,
    language: str = "ru",
    **options: object,
) -> FasterWhisperEngine:
    """A Whisper engine with a fake model in place of a loaded one."""
    engine = FasterWhisperEngine()
    engine._options = SttOptions(**options)  # type: ignore[arg-type]
    engine._language = "ru"
    engine._device = "cpu"
    engine._model = FakeWhisperModel(segments or [], language)
    return engine


class TestWhisperEngine:
    """Device selection, confidence arithmetic, and the hallucination filters."""

    def test_an_explicit_cpu_is_honoured_without_probing(self):
        assert FasterWhisperEngine._choose_device("cpu") == ("cpu", CPU_COMPUTE_TYPE, "")

    def test_an_explicit_cuda_is_honoured_without_probing(self):
        """A user who forced CUDA wants the error, not a silent four-times slowdown."""
        assert FasterWhisperEngine._choose_device("cuda") == ("cuda", CUDA_COMPUTE_TYPE, "")

    def test_auto_takes_cuda_when_it_is_there(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(faster_whisper_engine, "cuda_available", lambda: (True, ""))
        assert FasterWhisperEngine._choose_device("auto") == ("cuda", CUDA_COMPUTE_TYPE, "")

    def test_auto_falls_back_to_cpu_and_keeps_the_reason(self, monkeypatch):
        """The reason is the whole point: DevTools shows why it is slow."""
        monkeypatch.setattr(faster_whisper_engine, "cuda_available", lambda: (False, "нет карты"))
        assert FasterWhisperEngine._choose_device("auto") == ("cpu", CPU_COMPUTE_TYPE, "нет карты")

    def test_the_cuda_probe_never_raises(self):
        """Whatever ctranslate2 does on this machine, a start must survive it."""
        usable, reason = cuda_available()
        assert isinstance(usable, bool)
        assert usable or reason

    def test_the_cuda_probe_survives_a_build_without_cuda(self, monkeypatch):
        module = types.SimpleNamespace()
        monkeypatch.setitem(sys.modules, "ctranslate2", module)
        usable, reason = cuda_available()
        assert not usable
        assert "CUDA" in reason

    def test_the_cuda_probe_survives_a_driver_that_explodes(self, monkeypatch):
        def explode() -> int:
            raise RuntimeError("cuda driver version is insufficient")

        monkeypatch.setitem(
            sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=explode)
        )
        usable, reason = cuda_available()
        assert not usable
        assert "CUDA недоступна" in reason

    def test_the_cuda_probe_reports_an_empty_machine(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 0)
        )
        usable, reason = cuda_available()
        assert not usable
        assert "не найдена" in reason

    def test_the_cuda_probe_accepts_a_card(self, monkeypatch):
        monkeypatch.setitem(
            sys.modules, "ctranslate2", types.SimpleNamespace(get_cuda_device_count=lambda: 1)
        )
        assert cuda_available() == (True, "")

    def test_a_log_probability_becomes_a_usable_confidence(self):
        """``exp(avg_logprob)`` - a clean phrase lands near 0.85."""
        assert _logprob_to_confidence(-0.1625) == pytest.approx(0.85, abs=0.01)
        assert _logprob_to_confidence(-0.9) == pytest.approx(0.41, abs=0.01)
        assert _logprob_to_confidence(-40.0) == pytest.approx(0.0, abs=0.001)

    def test_a_non_negative_log_probability_is_full_confidence(self):
        assert _logprob_to_confidence(0.0) == 1.0
        assert _logprob_to_confidence(0.5) == 1.0

    def test_mean_confidence_weights_by_duration(self):
        """An unweighted mean lets a 200 ms interjection outvote a sentence."""
        segments = (
            TranscriptSegment(text="длинная фраза", start_ms=0.0, end_ms=800.0, confidence=0.9),
            TranscriptSegment(text="ах", start_ms=800.0, end_ms=1000.0, confidence=0.1),
        )
        assert _mean_confidence(segments) == pytest.approx(0.74, abs=0.01)

    def test_mean_confidence_falls_back_to_a_plain_mean(self):
        """A model built without timestamps gives every segment zero length."""
        segments = (
            TranscriptSegment(text="а", start_ms=0.0, end_ms=0.0, confidence=0.6),
            TranscriptSegment(text="б", start_ms=0.0, end_ms=0.0, confidence=0.8),
        )
        assert _mean_confidence(segments) == pytest.approx(0.7)

    def test_mean_confidence_of_nothing_is_zero(self):
        assert _mean_confidence(()) == 0.0

    def test_the_known_hallucinations_are_recognised(self):
        """What Whisper says when handed silence: the credits of a film."""
        assert _is_hallucination("Спасибо за просмотр!")
        assert _is_hallucination("  спасибо за просмотр  ")
        assert _is_hallucination("Продолжение следует...")
        assert _is_hallucination("Субтитры сделал DimaTorzok")

    def test_real_speech_is_not_mistaken_for_one(self):
        """The filter matches whole transcripts, never substrings."""
        assert not _is_hallucination("спасибо за помощь с этим проектом")
        assert not _is_hallucination("открой браузер")
        assert not _is_hallucination("продолжение следует за этим абзацем")

    def test_transcribe_passes_the_options_the_engine_was_configured_with(self):
        engine = whisper_ready([FakeSegment("привет мир", 0.0, 1.0)], beam_size=7)
        engine.transcribe(tone(2000))
        kwargs = engine._model.calls[0]
        assert kwargs["language"] == "ru"
        assert kwargs["beam_size"] == 7
        assert kwargs["vad_filter"] is True
        assert kwargs["condition_on_previous_text"] is False
        assert kwargs["without_timestamps"] is False

    def test_transcribe_joins_the_segments_it_keeps(self):
        engine = whisper_ready(
            [FakeSegment("  открой  ", 0.0, 0.5), FakeSegment(" браузер ", 0.5, 1.2)]
        )
        result = engine.transcribe(tone(2000))
        assert result.text == "открой браузер"
        assert len(result.segments) == 2
        assert result.segments[1].end_ms == pytest.approx(1200.0)
        assert result.confidence == pytest.approx(0.85, abs=0.01)
        assert result.device == "cpu"
        assert result.engine == "whisper"
        assert result.inference_ms > 0.0

    def test_transcribe_reports_the_language_the_model_detected(self):
        engine = whisper_ready([FakeSegment("hello there", 0.0, 1.0)], language="en")
        assert engine.transcribe(tone(2000)).language == "en"

    def test_transcribe_keeps_the_configured_language_when_none_was_detected(self):
        engine = whisper_ready([FakeSegment("привет", 0.0, 1.0)], language="")
        assert engine.transcribe(tone(2000)).language == "ru"

    def test_a_segment_over_the_no_speech_threshold_is_dropped(self):
        engine = whisper_ready(
            [FakeSegment("фоновый шум", 0.0, 1.0, no_speech_prob=0.99)],
            no_speech_threshold=0.6,
        )
        assert engine.transcribe(tone(2000)).is_empty

    def test_a_segment_shorter_than_the_minimum_is_dropped(self):
        engine = whisper_ready([FakeSegment("ы", 0.0, 0.02)])
        assert engine.transcribe(tone(2000)).is_empty

    def test_an_empty_segment_is_dropped(self):
        engine = whisper_ready([FakeSegment("   ", 0.0, 1.0)])
        assert engine.transcribe(tone(2000)).is_empty

    def test_a_hallucinated_transcript_is_discarded_but_still_timed(self):
        """The inference happened; the log should say how long it took."""
        engine = whisper_ready([FakeSegment("Спасибо за просмотр!", 0.0, 2.0)])
        result = engine.transcribe(tone(2000))
        assert result.is_empty
        assert result.inference_ms > 0.0
        assert result.duration_ms == pytest.approx(2000.0)

    def test_silence_never_reaches_the_model(self):
        """The cheapest hallucination filter, and the one that catches the most."""
        engine = whisper_ready([FakeSegment("Продолжение следует...", 0.0, 1.0)])
        result = engine.transcribe(wav("stt_silence.wav"))
        assert engine._model.calls == []
        assert result.is_empty

    def test_a_short_buffer_never_reaches_the_model(self):
        engine = whisper_ready([FakeSegment("привет", 0.0, 1.0)], min_speech_ms=500)
        assert engine.transcribe(tone(200)).is_empty
        assert engine._model.calls == []

    def test_the_model_is_handed_floats_not_int16(self):
        """CTranslate2 wants normalised floats; ``floats()`` is where that happens."""
        engine = whisper_ready([FakeSegment("привет", 0.0, 1.0)])
        engine.transcribe(tone(1000))
        assert engine._model.audio == [STT_SAMPLE_RATE]

    def test_a_model_that_raises_becomes_a_typed_error(self):
        engine = whisper_ready([FakeSegment("привет", 0.0, 1.0)])

        def explode(_audio: object, **_kwargs: object) -> None:
            raise RuntimeError("CUDA out of memory")

        engine._model.transcribe = explode  # type: ignore[method-assign]
        with pytest.raises(SttError) as excinfo:
            engine.transcribe(tone(2000))
        assert excinfo.value.user_message == "Ошибка распознавания речи."
        assert "cpu" in str(excinfo.value)

    def test_whisper_needs_more_memory_than_vosk(self):
        """The RAM check would be wrong for one of them with a shared factor."""
        assert FasterWhisperEngine.memory_factor > VoskSttEngine.memory_factor

    def test_unloading_twice_is_safe(self):
        engine = whisper_ready()
        engine.unload()
        engine.unload()
        assert not engine.loaded
        assert engine._model is None


# ----------------------------------------------------------------------
# the worker
# ----------------------------------------------------------------------


class TestWorkerAudio:
    """Shared memory in, resampled mono out.

    The resampling happens here and exactly once, which is why these assertions
    look at what the *engine* received rather than at the returned text.
    """

    def test_a_phrase_comes_back_as_a_transcript(self, worker: SttWorker):
        reply = transcribe(worker, wav("stt_command.wav"))
        assert reply["text"] == "привет"
        assert reply["engine"] == "stub"
        assert reply["duration_ms"] == pytest.approx(1020.0, abs=1.0)

    def test_the_engine_is_handed_the_recognition_rate(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        seen = engine_of(worker).seen[-1]
        assert seen.sample_rate == STT_SAMPLE_RATE
        assert seen.channels == 1

    def test_forty_eight_kilohertz_stereo_is_downmixed_and_resampled(self, worker: SttWorker):
        """What a real capture device hands over, in the shape Vosk needs."""
        transcribe(worker, stereo(tone(1000, rate=48000)))
        seen = engine_of(worker).seen[-1]
        assert seen.sample_rate == STT_SAMPLE_RATE
        assert seen.channels == 1
        assert seen.duration_ms == pytest.approx(1000.0, abs=1.0)

    def test_a_call_without_audio_is_a_typed_error(self, worker: SttWorker):
        with pytest.raises(SttError) as excinfo:
            worker.transcribe({})
        assert excinfo.value.user_message

    def test_a_stale_block_descriptor_is_a_typed_error(self, worker: SttWorker):
        """The supervisor closed the block before the reply came back."""
        with SharedAudioBlock.create(b"\0" * 640, sample_rate=STT_SAMPLE_RATE, channels=1) as block:
            params = block.chunk.to_params()
        with pytest.raises(SttError):
            worker.transcribe(params)

    def test_silence_comes_back_empty_rather_than_as_an_error(self, worker: SttWorker):
        reply = transcribe(worker, wav("stt_silence.wav"))
        assert reply["text"] == ""
        assert reply["duration_ms"] == pytest.approx(1000.0)

    def test_the_language_travels_with_the_result(self, worker: SttWorker):
        reply = transcribe(worker, wav("stt_command.wav"))
        assert reply["language"] == "ru"

    def test_the_audio_is_copied_out_before_the_block_closes(self, worker: SttWorker):
        """The mapping is only valid inside the ``with``; the engine outlives it."""
        transcribe(worker, wav("stt_command.wav"))
        seen = engine_of(worker).seen[-1]
        assert len(seen.pcm) > 0
        assert seen.duration_ms == pytest.approx(1020.0, abs=1.0)


class TestWorkerLifecycle:
    """Lazy load, idle unload, reload, and the memory cap."""

    def test_starting_the_worker_loads_nothing(self, worker: SttWorker):
        """A start that loads a model is a start the supervisor times out on."""
        assert worker._engine is None
        assert worker.status({})["loaded"] is False

    def test_the_first_transcription_brings_the_model_up(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        assert engine_of(worker).loads == 1
        assert worker.status({})["loaded"] is True

    def test_the_second_transcription_reuses_it(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        transcribe(worker, wav("stt_phrase.wav"))
        assert engine_of(worker).loads == 1

    def test_load_model_returns_before_the_model_is_ready(self, worker: SttWorker):
        """Dispatch is single-threaded: a blocking load stops ``ping`` too."""
        reply = worker.load_model({})
        assert reply["started"] is True
        worker.load_model({"wait": True})
        assert worker.status({})["loaded"] is True

    def test_load_model_twice_does_not_load_twice(self, worker: SttWorker):
        worker.load_model({"wait": True})
        reply = worker.load_model({})
        assert reply["started"] is False
        assert reply["reason"] == "already_loaded"
        assert engine_of(worker).loads == 1

    def test_a_finished_load_announces_itself(self, worker: SttWorker):
        worker.load_model({"wait": True})
        states = [event["state"] for event in worker.context.events_of("model")]
        assert "loaded" in states

    def test_unloading_gives_the_memory_back(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        assert worker.unload({})["unloaded"] is True
        assert engine.unloads == 1
        assert worker.status({})["loaded"] is False

    def test_unloading_when_nothing_is_loaded_is_not_an_error(self, worker: SttWorker):
        assert worker.unload({})["unloaded"] is False

    def test_a_transcription_after_an_unload_loads_again(self, worker: SttWorker):
        """Eco mode dropped the model; the next phrase must still be recognised."""
        transcribe(worker, wav("stt_command.wav"))
        worker.unload({})
        reply = transcribe(worker, wav("stt_command.wav"))
        assert engine_of(worker).loads == 2
        assert reply["text"] == "привет"

    def test_an_unload_is_announced_so_devtools_can_show_it(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        worker.unload({})
        states = [event["state"] for event in worker.context.events_of("model")]
        assert "unloaded" in states

    def test_stopping_the_worker_releases_the_model(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        worker.on_stop()
        assert engine.unloads == 1

    def test_changing_the_engine_drops_the_loaded_model(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        worker.on_configure({"offline_engine": "vosk", "offline_model": "other"})
        assert engine.unloads == 1
        assert worker.status({})["loaded"] is False

    def test_reconfiguring_with_the_same_model_keeps_it(self, worker: SttWorker):
        """A settings save that touched an unrelated slider must not cost 20 seconds."""
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        worker.on_configure({"language": "ru"})
        assert engine.unloads == 0
        assert worker.status({})["loaded"] is True

    def test_a_model_over_the_memory_cap_is_refused_before_loading(
        self, worker: SttWorker, model_dir: Path
    ):
        """An OOM kill looks to the user like the assistant randomly dying."""
        (model_dir / "am" / "final.mdl").write_bytes(b"\0" * (3 * 1024 * 1024))
        worker.context.params["ram_limit_mb"] = 1
        with pytest.raises(SttError) as excinfo:
            transcribe(worker, wav("stt_command.wav"))
        assert excinfo.value.user_message
        assert worker._engine is None or engine_of(worker).loads == 0

    def test_the_refusal_names_both_numbers(self, worker: SttWorker, model_dir: Path):
        """ "Not enough memory" without the two figures is not actionable."""
        (model_dir / "am" / "final.mdl").write_bytes(b"\0" * (3 * 1024 * 1024))
        worker.context.params["ram_limit_mb"] = 1
        with pytest.raises(SttError) as excinfo:
            transcribe(worker, wav("stt_command.wav"))
        message = excinfo.value.user_message
        assert "3" in message
        assert "1 МБ" in message

    def test_a_model_too_small_to_measure_is_let_through(self, worker: SttWorker):
        """Zero means "cannot tell", and guessing wrong must not block a start."""
        worker.context.params["ram_limit_mb"] = 1
        transcribe(worker, wav("stt_command.wav"))
        assert engine_of(worker).loads == 1

    def test_a_cap_of_zero_means_no_cap(self, worker: SttWorker):
        worker.context.params["ram_limit_mb"] = 0
        transcribe(worker, wav("stt_command.wav"))
        assert engine_of(worker).loads == 1

    def test_a_generous_cap_lets_the_model_through(self, worker: SttWorker):
        worker.context.params["ram_limit_mb"] = 4096
        transcribe(worker, wav("stt_command.wav"))
        assert engine_of(worker).loads == 1

    def test_a_missing_model_name_is_a_typed_error(self, worker: SttWorker):
        worker.context.params["offline_model"] = ""
        with pytest.raises(SttError) as excinfo:
            transcribe(worker, wav("stt_command.wav"))
        assert excinfo.value.user_message

    def test_a_failed_load_is_reported_and_not_retried_silently(self, worker: SttWorker):
        worker.context.params["offline_model"] = ""
        worker.load_model({"wait": True})
        failures = [
            event for event in worker.context.events_of("model") if event["state"] == "failed"
        ]
        assert failures
        assert failures[0]["error"]


class TestIdleTimeout:
    """The timeout itself, without waiting for it in real time."""

    def test_the_configured_timeout_is_used(self, worker: SttWorker):
        worker.context.params["model_idle_sec"] = 300.0
        assert worker._idle_timeout() == pytest.approx(300.0)

    def test_eco_mode_halves_it(self, worker: SttWorker):
        """A laptop on battery should give the memory back sooner."""
        worker.context.params["model_idle_sec"] = 300.0
        worker.context.params["eco_mode"] = True
        assert worker._idle_timeout() == pytest.approx(150.0)

    def test_it_never_drops_below_the_floor(self, worker: SttWorker):
        """A five-second timeout against a twenty-second load is not a strategy."""
        worker.context.params["model_idle_sec"] = 10.0
        worker.context.params["eco_mode"] = True
        assert worker._idle_timeout() == pytest.approx(30.0)

    def test_zero_disables_it(self, worker: SttWorker):
        worker.context.params["model_idle_sec"] = 0.0
        assert worker._idle_timeout() == 0.0

    def test_zero_stays_disabled_under_eco_mode(self, worker: SttWorker):
        worker.context.params["model_idle_sec"] = 0.0
        worker.context.params["eco_mode"] = True
        assert worker._idle_timeout() == 0.0

    def test_an_idle_model_is_dropped(self, worker: SttWorker):
        """The unload the timer exists for, triggered without sleeping for it."""
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        worker.context.params["model_idle_sec"] = 30.0
        worker._last_used = monotonic() - 3600.0
        worker._drop_if_idle()
        assert engine.unloads == 1
        assert worker.status({})["loaded"] is False

    def test_a_model_in_use_is_kept(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        engine = engine_of(worker)
        worker.context.params["model_idle_sec"] = 300.0
        worker._drop_if_idle()
        assert engine.unloads == 0

    def test_the_status_reports_the_idle_time(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        assert worker.status({})["idle_sec"] >= 0.0


class TestWorkerMetrics:
    """The numbers the pipeline log and DevTools show."""

    def test_a_call_is_timed(self, worker: SttWorker):
        """Two numbers, not one: what the engine spent, and what the call cost.

        The stub reports a fixed 12.5 ms, which is how this can tell that the
        engine's own figure is the one published rather than the wall clock -
        with a real engine the two are within a millisecond of each other and
        the assertion would prove nothing.
        """
        transcribe(worker, wav("stt_command.wav"))
        metrics = worker.context.events_of("metrics")[-1]
        assert metrics["inference_ms"] == pytest.approx(12.5)
        assert metrics["total_ms"] > 0.0

    def test_the_metrics_carry_the_length_of_the_audio(self, worker: SttWorker):
        transcribe(worker, wav("stt_phrase.wav"))
        metrics = worker.context.events_of("metrics")[-1]
        assert metrics["audio_ms"] == pytest.approx(2400.0, abs=1.0)

    def test_the_real_time_factor_relates_the_two(self, worker: SttWorker):
        """Above 1.0 means the machine cannot keep up with the speaker."""
        transcribe(worker, wav("stt_phrase.wav"))
        metrics = worker.context.events_of("metrics")[-1]
        expected = float(metrics["inference_ms"]) / float(metrics["audio_ms"])
        assert metrics["real_time_factor"] == pytest.approx(expected, abs=0.001)

    def test_the_request_id_travels_through(self, worker: SttWorker):
        """Without it the pipeline log cannot join this line to the phrase."""
        transcribe(worker, wav("stt_command.wav"), request_id="req-42")
        assert worker.context.events_of("metrics")[-1]["request_id"] == "req-42"

    def test_an_empty_result_is_still_measured(self, worker: SttWorker):
        """Time spent deciding there was nothing there is time spent."""
        transcribe(worker, wav("stt_silence.wav"))
        metrics = worker.context.events_of("metrics")[-1]
        assert metrics["empty"] is True
        assert metrics["audio_ms"] == pytest.approx(1000.0)

    def test_the_engine_is_named_in_the_metrics(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        assert worker.context.events_of("metrics")[-1]["engine"] == "stub"

    def test_the_load_time_is_reported_separately(self, worker: SttWorker):
        """A cold start is not a slow machine, and the two must be tellable apart."""
        transcribe(worker, wav("stt_command.wav"))
        assert worker.status({})["load_ms"] >= 0.0

    def test_the_call_count_grows(self, worker: SttWorker):
        assert worker.status({})["calls"] == 0
        transcribe(worker, wav("stt_command.wav"))
        transcribe(worker, wav("stt_phrase.wav"))
        assert worker.status({})["calls"] == 2

    def test_the_average_covers_every_call(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        transcribe(worker, wav("stt_phrase.wav"))
        reported = [float(event["inference_ms"]) for event in worker.context.events_of("metrics")]
        assert worker.status({})["avg_inference_ms"] == pytest.approx(
            sum(reported) / len(reported), abs=0.2
        )

    def test_the_overall_factor_is_over_all_the_audio(self, worker: SttWorker):
        transcribe(worker, wav("stt_command.wav"))
        transcribe(worker, wav("stt_phrase.wav"))
        status = worker.status({})
        assert 0.0 < status["real_time_factor"] < 1.0

    def test_the_averages_start_at_zero(self, worker: SttWorker):
        """Before the first phrase there is nothing to average, not a division by zero."""
        status = worker.status({})
        assert status["avg_inference_ms"] == 0.0
        assert status["real_time_factor"] == 0.0

    def test_the_status_names_the_configured_engine_before_a_load(self, worker: SttWorker):
        status = worker.status({})
        assert status["engine"] == "stub"
        assert status["loaded"] is False

    def test_the_status_lists_only_installed_engines(self, worker: SttWorker):
        """What the settings window offers; a name here must actually run."""
        for name in worker.status({})["available_engines"]:
            assert engine_class(name).available()

    def test_the_status_repeats_the_limits_it_is_enforcing(self, worker: SttWorker):
        worker.context.params["ram_limit_mb"] = 2048
        worker.context.params["model_idle_sec"] = 300.0
        status = worker.status({})
        assert status["ram_limit_mb"] == 2048
        assert status["idle_timeout_sec"] == pytest.approx(300.0)


class TestEventTranslation:
    """What a worker event becomes on the bus."""

    def test_a_failed_load_reaches_the_user(self):
        """The one STT event worth interrupting somebody for."""
        event = translate_stt_event(
            "model", {"state": "failed", "model": "small", "error": "нет файла"}
        )
        assert event is not None
        assert type(event).__name__ == "NotificationRequested"
        assert "нет файла" in str(getattr(event, "message", ""))

    def test_the_failure_is_shown_as_an_error(self):
        event = translate_stt_event("model", {"state": "failed", "error": "нет файла"})
        assert event is not None
        assert str(getattr(event, "level", "")).endswith("error")

    def test_a_successful_load_is_not_a_notification(self):
        """Nobody needs a toast saying a model loaded - that is the point of lazy."""
        assert translate_stt_event("model", {"state": "loaded", "model": "small"}) is None

    def test_an_idle_unload_is_silent_too(self):
        assert translate_stt_event("model", {"state": "unloaded"}) is None

    def test_metrics_stay_off_the_bus(self):
        """One of these per phrase; the bus is not a metrics pipeline."""
        assert translate_stt_event("metrics", {"inference_ms": 90.0}) is None

    def test_an_unknown_event_is_ignored_rather_than_raising(self):
        assert translate_stt_event("что-то новое", {}) is None


def _spec() -> Any:
    """The STT spec the supervisor would build from default settings."""
    kind = worker_type(WorkerKind.STT.value)
    assert kind is not None
    return kind.build(Settings())


class TestRegistryWiring:
    """The supervisor has to be able to find and start this worker."""

    def test_the_stt_kind_points_at_this_module(self):
        """Task 10 puts the worker here; the registry has to agree."""
        kind = worker_type(WorkerKind.STT.value)
        assert kind is not None
        assert kind.entrypoint == "ayris.workers.stt_worker:SttWorker"

    def test_the_entrypoint_resolves_to_the_worker_class(self):
        kind = worker_type(WorkerKind.STT.value)
        assert kind is not None
        module_name, _, class_name = kind.entrypoint.partition(":")
        module = importlib.import_module(module_name)
        assert getattr(module, class_name) is SttWorker

    def test_the_entrypoint_is_importable_in_this_build(self):
        kind = worker_type(WorkerKind.STT.value)
        assert kind is not None
        assert kind.available

    def test_the_worker_declares_its_kind(self):
        assert SttWorker.kind == WorkerKind.STT.value

    def test_starting_it_is_known_to_cost_memory(self):
        """DevTools greys it out in eco mode on the strength of this flag."""
        kind = worker_type(WorkerKind.STT.value)
        assert kind is not None
        assert kind.loads_local_model

    def test_the_spec_carries_the_limits_the_worker_enforces(self):
        """The worker is the only process that can measure the model on disk."""
        spec = _spec()
        assert "ram_limit_mb" in spec.params
        assert "model_idle_sec" in spec.params

    def test_the_spec_carries_the_engine_and_the_model(self):
        spec = _spec()
        assert spec.params["offline_engine"] == Settings().voice.stt.offline_engine
        assert "offline_model" in spec.params

    def test_the_module_exports_a_translator_where_the_registry_looks(self):
        module = importlib.import_module("ayris.workers.stt_worker")
        assert getattr(module, "EVENT_TRANSLATOR", None) is translate_stt_event

    def test_the_registry_finds_that_translator(self):
        assert event_translator(WorkerKind.STT.value) is translate_stt_event

    def test_the_wire_surface_is_what_the_supervisor_calls(self):
        names = set(worker_methods(SttWorker))
        assert {"load_model", "transcribe", "unload", "status"} <= names

    def test_importing_the_worker_does_not_import_a_vendor_library(self):
        """A worker start must not pay for CTranslate2 before an engine is chosen.

        In a fresh interpreter, which is the only place the question has an
        answer.  ``sys.modules`` in *this* one is shared with every test that ran
        before it, and one of them loads a real Vosk engine to check the wording
        of its error - so asserting on this process made the result depend on the
        order pytest happened to pick, and on whether the vendor library was
        installed at all.
        """
        probe = (
            "import importlib, sys;"
            "importlib.import_module('ayris.workers.stt_worker');"
            "print([m for m in ('faster_whisper', 'vosk') if m in sys.modules])"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
        )
        assert completed.stdout.strip() == "[]", completed.stdout


@pytest.mark.hardware
class TestRealEngines:
    """Needs a vendor library and model weights; excluded from CI.

    A Vosk model is fifty megabytes and a Whisper one is a gigabyte, so neither
    is committed and neither exists on a runner.  These run where somebody has
    downloaded one, and they check the single thing a stub cannot: that the real
    library, on real weights, answers the real fixtures the way the contract
    above says it must.

    Two variables, not one: a Vosk model is a directory with ``am`` inside and a
    faster-whisper one is a CTranslate2 export with ``model.bin``, so a single
    path cannot enable both.  Point ``AYRIS_TEST_STT_MODEL`` at the Vosk model
    and ``AYRIS_TEST_WHISPER_MODEL`` at the Whisper one; each engine's tests
    skip on their own.
    """

    @staticmethod
    def _model(weights: _Weights, variable: str = "AYRIS_TEST_STT_MODEL") -> Path:
        """The Vosk model named by *variable*, spelled so Kaldi can open it.

        Vosk hands the path to a C++ library as UTF-8 bytes that Windows reads
        back through the ANSI code page, so a checkout under a Cyrillic folder
        needs the ``ascii_weights`` copy - see that fixture for why production
        answers the same problem differently.
        """
        return weights(_downloaded(variable))

    @staticmethod
    def _whisper_model() -> Path:
        """The Whisper model, straight from where it was downloaded.

        No ``ascii_weights`` here on purpose: CTranslate2 converts the path to
        wide characters itself, so it opens ``E:\\мистер бит ест рис`` fine, and
        loading from the real location is the stricter test - it would notice if
        that ever stopped being true.
        """
        return _downloaded("AYRIS_TEST_WHISPER_MODEL")

    def test_vosk_transcribes_a_command(self, ascii_weights: _Weights) -> None:
        if not VoskSttEngine.available():
            pytest.skip("vosk не установлен")
        engine = VoskSttEngine()
        engine.load(self._model(ascii_weights), SttOptions(language="ru"))
        try:
            result = engine.transcribe(wav("stt_command.wav"))
            assert result.engine == "vosk"
            assert result.duration_ms == pytest.approx(1020.0, abs=1.0)
            assert 0.0 <= result.confidence <= 1.0
        finally:
            engine.unload()

    def test_vosk_answers_silence_with_nothing(self, ascii_weights: _Weights) -> None:
        """The synthetic fixture is not words, so the text is not asserted.

        What is asserted is that a silent buffer produces an empty result and
        not an exception, which is the contract the pipeline depends on.
        """
        if not VoskSttEngine.available():
            pytest.skip("vosk не установлен")
        engine = VoskSttEngine()
        engine.load(self._model(ascii_weights), SttOptions(language="ru"))
        try:
            assert engine.transcribe(wav("stt_silence.wav")).is_empty
        finally:
            engine.unload()

    def test_whisper_answers_silence_with_nothing(self) -> None:
        """The hallucination filters, on the model that needs them."""
        if not FasterWhisperEngine.available():
            pytest.skip("faster-whisper не установлен")
        engine = FasterWhisperEngine()
        engine.load(self._whisper_model(), SttOptions(language="ru"))
        try:
            assert engine.transcribe(wav("stt_silence.wav")).is_empty
        finally:
            engine.unload()

    def test_whisper_reports_the_device_it_chose(self) -> None:
        if not FasterWhisperEngine.available():
            pytest.skip("faster-whisper не установлен")
        engine = FasterWhisperEngine()
        engine.load(self._whisper_model(), SttOptions(language="ru"))
        try:
            assert engine.device in {"cpu", "cuda"}
            if engine.device == "cpu":
                assert engine.fallback_reason
        finally:
            engine.unload()

    def test_recognition_keeps_up_with_the_speaker(self, ascii_weights: _Weights) -> None:
        """A real-time factor over 1.0 means the assistant falls behind."""
        if not VoskSttEngine.available():
            pytest.skip("vosk не установлен")
        engine = VoskSttEngine()
        engine.load(self._model(ascii_weights), SttOptions(language="ru"))
        try:
            result = engine.transcribe(wav("stt_phrase.wav"))
            assert result.real_time_factor < 1.0
        finally:
            engine.unload()
