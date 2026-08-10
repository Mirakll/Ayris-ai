"""Task 12: local synthesis — splitting, caching, the engines and the worker.

There is no voice model on the runner and no speakers attached to it, which
sounds like it leaves little to test.  It does not.  Almost everything task 12
asks for lives *around* the model: where a sentence ends, what the cache is keyed
on, when a voice is loaded and dropped, what happens to a stream when «Айрис,
стоп» arrives.  All of it is exercised here against a stub engine that returns a
tone of a predictable length, and the stub is the stricter subject — it lets an
assertion say "the worker loaded the voice exactly once", which no real Piper
would let us see.

Anything that needs a real ``.onnx`` on disk carries ``@pytest.mark.hardware``
and is skipped everywhere except a machine that has one.

The player has its own file, :mod:`tests.unit.test_tts_player`, because its
subject is a device rather than a model.

Groups:

* :class:`TestSentenceSplit` — where a sentence ends, and where it only looks
  like it does: abbreviations, initials, decimals, ellipses.
* :class:`TestLongSentences` — an answer with no punctuation still has to start
  sounding quickly.
* :class:`TestVoiceSpec` — voice identity, which the cache key is built on.
* :class:`TestAudioChunk` — duration arithmetic and the reply metadata.
* :class:`TestTtsOptions` — worker params to engine options, tolerantly.
* :class:`TestEngineRegistry` — names, lazy resolution, the optional engine.
* :class:`TestEngineContract` — what :class:`TtsEngine` guarantees a subclass.
* :class:`TestPiperEngine` / :class:`TestSileroEngine` — the parts of the two
  real engines that are pure functions of their inputs.
* :class:`TestPhraseCache` — hit, miss, key sensitivity, LRU eviction, limits.
* :class:`TestWorkerSynthesis` — text in, shared memory out.
* :class:`TestWorkerStreaming` — sentence by sentence, with a stream token.
* :class:`TestWorkerCancel` — what «Айрис, стоп» stops.
* :class:`TestWorkerLifecycle` — lazy load, idle unload, reload, the RAM cap.
* :class:`TestWorkerCacheIntegration` — the worker actually uses the cache.
* :class:`TestEventTranslation` — what a worker event becomes on the bus.
* :class:`TestRegistryWiring` — the supervisor can find this worker.
"""

from __future__ import annotations

import wave
from array import array
from math import pi, sin
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final

import pytest

from ayris.audio.tts.base import (
    DEFAULT_SAMPLE_RATE,
    ENGINE_ENTRYPOINTS,
    MAX_PITCH,
    MAX_SPEED,
    MIN_PITCH,
    MIN_SPEED,
    SAMPLE_WIDTH,
    AudioChunk,
    TtsEngine,
    TtsOptions,
    VoiceSpec,
    clamp_pitch,
    clamp_speed,
    concat_chunks,
    engine_class,
    engine_names,
    estimate_voice_bytes,
)
from ayris.audio.tts.cache import (
    MAX_TEXT_CHARS,
    CacheStats,
    PhraseCache,
    phrase_key,
    voice_key,
)
from ayris.audio.tts.sentence_split import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
    is_speakable,
    normalize_whitespace,
    split_sentences,
)
from ayris.core.errors import TtsError
from ayris.core.events import NotificationRequested
from ayris.workers.protocol import AUDIO_PARAM, open_audio
from ayris.workers.tts_worker import EVENT_TRANSLATOR, TtsWorker, translate_tts_event

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ayris.core.models import JsonObject

pytestmark = pytest.mark.unit

#: Sentences long enough that :data:`MIN_CHUNK_CHARS` leaves them alone. A short
#: *final* piece is deliberately folded into its predecessor, so a test about
#: where a boundary is has to keep its sentences above the fold threshold or it
#: ends up asserting the merge rule by accident.
S1: Final = "Первое предложение."
S2: Final = "Второе предложение."
S3: Final = "Третье предложение."
TWO_SENTENCES: Final = f"{S1} {S2}"
THREE_SENTENCES: Final = f"{S1} {S2} {S3}"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def tone(ms: int, sample_rate: int = DEFAULT_SAMPLE_RATE) -> bytes:
    """``ms`` milliseconds of a quiet 220 Hz sine, as int16 mono.

    A tone rather than silence so that a test can tell "audio arrived" from "the
    buffer was zeroed", and quiet so that a stray real device would not startle
    anyone running the suite with headphones on.
    """
    frames = max(0, int(sample_rate * ms / 1000))
    samples = array(
        "h",
        (int(6000 * sin(2 * pi * 220 * index / sample_rate)) for index in range(frames)),
    )
    return samples.tobytes()


class StubEngine(TtsEngine):
    """A synthesizer with no model: predictable audio, countable calls.

    Duration is proportional to the text so a test can assert that the *second*
    sentence is what came back, and inversely proportional to speed so that the
    speed argument is observably not ignored.
    """

    name: ClassVar[str] = "stub"
    package: ClassVar[str] = ""
    module: ClassVar[str] = ""
    optional: ClassVar[bool] = False
    memory_factor: ClassVar[float] = 2.0
    native_sample_rate: ClassVar[int] = DEFAULT_SAMPLE_RATE

    def __init__(self) -> None:
        super().__init__()
        self.loads = 0
        self.unloads = 0
        self.calls: list[tuple[str, float, float]] = []
        self.fail_with: Exception | None = None
        self.fail_load_with: Exception | None = None
        #: Run after a sentence is synthesized. How a test says «стоп» arrived
        #: while the engine was busy, which is the only moment that matters for
        #: cancellation and the one a test cannot otherwise reach.
        self.on_call: Callable[[], None] | None = None

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("ru",)

    @classmethod
    def voices(cls, directory: Path | None = None) -> tuple[VoiceSpec, ...]:
        del directory
        return (
            VoiceSpec(engine="stub", voice_id="stub-ru", display_name="Заглушка"),
            VoiceSpec(engine="stub", voice_id="stub-alt", display_name="Другая заглушка"),
        )

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        if self.fail_load_with is not None:
            raise self.fail_load_with
        self.loads += 1
        self._voice = voice
        self._options = options

    def unload(self) -> None:
        self.unloads += 1
        self._voice = None

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        if self.fail_with is not None:
            raise self.fail_with
        self.calls.append((text, speed, pitch))
        if self.on_call is not None:
            self.on_call()
        return AudioChunk(tone(int(10 * len(text) / speed)), self.sample_rate)


class FakeContext:
    """Enough of :class:`~ayris.workers.base.WorkerContext` to drive the worker.

    Same shape as the one in :mod:`tests.unit.test_stt_offline`, and for the same
    reason: this file's subject is synthesis, and the process machinery has its
    own tests in :mod:`tests.unit.test_workers`.
    """

    def __init__(self, params: JsonObject | None = None) -> None:
        self.name = "tts"
        self.kind = "tts"
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

        return logging.getLogger(f"ayris.workers.tts.{suffix}" if suffix else "ayris.workers.tts")

    def events_of(self, kind: str) -> list[JsonObject]:
        """Every payload emitted under ``kind``, in order."""
        return [payload for name, payload in self.events if name == kind]


def pcm_of(reply: JsonObject) -> bytes:
    """Read a worker reply's audio out of shared memory.

    Copies: the view is only valid inside the ``with``, and a test that returned
    the memoryview would fail later and somewhere else.
    """
    chunk = reply[AUDIO_PARAM]
    with open_audio(chunk) as buffer:
        return bytes(buffer)


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> StubEngine:
    """The engine the worker will get, injected without touching the registry.

    ``ENGINE_ENTRYPOINTS`` is production data; a test that added a "stub" entry
    would leak into every later test that reads :func:`engine_names`.
    """
    engine = StubEngine()
    monkeypatch.setattr("ayris.workers.tts_worker.create_engine", lambda _name: engine)
    monkeypatch.setattr("ayris.workers.tts_worker.engine_class", lambda _name: StubEngine)
    return engine


@pytest.fixture
def worker(stub_engine: StubEngine, tmp_path: Path) -> Iterator[TtsWorker]:
    """A started worker on the stub engine, with the cache off by default."""
    del stub_engine
    context = FakeContext(
        {
            "engine": "stub",
            "voice": "stub-ru",
            "speed": 1.0,
            "pitch": 1.0,
            "cache_size_mb": 0,
            "model_idle_sec": 0.0,
            "ram_limit_mb": 0,
        }
    )
    instance = TtsWorker(context)  # type: ignore[arg-type]
    instance.on_start()
    yield instance
    instance.on_stop()


@pytest.fixture
def cached_worker(worker: TtsWorker, tmp_path: Path) -> TtsWorker:
    """The same worker with a real, small cache pointed at ``tmp_path``."""
    worker._cache = PhraseCache(tmp_path / "tts-cache", limit_bytes=4 * 1024 * 1024)
    return worker


def engine_of(worker: TtsWorker) -> StubEngine:
    """The stub inside a worker, typed."""
    engine = worker._engine
    assert isinstance(engine, StubEngine)
    return engine


def engine_of_started(worker: TtsWorker) -> StubEngine:
    """The stub, loading the voice first so a test can arm ``on_call``."""
    worker.load_voice({})
    return engine_of(worker)


# ----------------------------------------------------------------------
# sentence splitting
# ----------------------------------------------------------------------


class TestSentenceSplit:
    """Where a sentence ends — and where it only looks like it does."""

    def test_empty_text_yields_nothing(self):
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_punctuation_only_yields_nothing(self):
        """«...» is not a sentence, and synthesizing it would produce a click."""
        assert split_sentences("...") == []
        assert split_sentences("!?") == []
        assert split_sentences("—") == []

    def test_is_speakable_wants_a_letter_or_a_digit(self):
        assert is_speakable("а")
        assert is_speakable("5")
        assert not is_speakable("?!.")
        assert not is_speakable("   ")

    def test_one_sentence_stays_one(self):
        assert split_sentences("Привет, я Айрис.") == ["Привет, я Айрис."]

    def test_a_sentence_without_a_terminator_is_still_a_sentence(self):
        assert split_sentences("Открываю браузер") == ["Открываю браузер"]

    def test_two_sentences_are_split(self):
        assert split_sentences(TWO_SENTENCES) == [S1, S2]

    def test_the_terminator_stays_with_its_sentence(self):
        """The engine needs it: a full stop is what makes the pitch fall."""
        text = "Первое предложение! Второе предложение? Третье предложение."
        for piece in split_sentences(text):
            assert piece[-1] in ".!?"

    def test_question_and_exclamation_split_too(self):
        text = "Первое предложение? Второе предложение! Третье предложение."
        assert len(split_sentences(text)) == 3

    def test_whitespace_is_normalized(self):
        assert split_sentences(f"{S1}\n\n  {S2}") == [S1, S2]

    def test_an_ellipsis_is_one_boundary_not_three(self):
        assert split_sentences(f"Одну секунду… {S2}") == ["Одну секунду…", S2]

    def test_dotted_ellipsis_does_not_split_three_ways(self):
        assert len(split_sentences(f"Одну секунду... {S2}")) == 2

    @pytest.mark.parametrize(
        "text",
        [
            "Это т.е. так же, как раньше.",
            "Стоит 100 руб. и ни копейкой больше.",
            "См. настройки для подробностей.",
            "Например, т.д. и т.п. в одном предложении.",
            "Открой файл на ул. Ленина, д. 5 и посмотри.",
            "Подробности есть на рис. 3 в конце главы.",
        ],
    )
    def test_an_abbreviation_is_not_a_boundary(self, text: str):
        """«т.е.» inside a sentence must not cut it in half mid-thought."""
        assert split_sentences(text) == [text]

    def test_an_abbreviation_before_a_capital_is_still_not_a_boundary(self):
        """This is the case the table exists for: «руб.» then a new word."""
        assert split_sentences("Цена 100 руб. Дальше идут другие товары.") == [
            "Цена 100 руб. Дальше идут другие товары."
        ]

    def test_initials_do_not_split(self):
        """«А. С. Пушкин» is one name, not three sentences."""
        assert split_sentences("Автор — А. С. Пушкин.") == ["Автор — А. С. Пушкин."]

    def test_a_decimal_number_does_not_split(self):
        assert split_sentences("Курс 3.14 на сегодня.") == ["Курс 3.14 на сегодня."]

    def test_a_version_number_does_not_split(self):
        assert split_sentences("Версия 1.2.3 установлена.") == ["Версия 1.2.3 установлена."]

    def test_a_list_marker_does_not_start_a_sentence(self):
        """«1.» is numbering, and speaking it alone turns a list into a stutter."""
        assert len(split_sentences("Шаги установки. 1. Открыть файл настроек.")) == 2

    def test_a_closing_quote_stays_with_its_sentence(self):
        pieces = split_sentences(f"Он сказал: «Готово.» {S2}")
        assert pieces[0].endswith("»")

    def test_a_short_tail_is_folded_into_its_predecessor(self):
        """A one-word chunk of its own would sound clipped after a pause."""
        pieces = split_sentences("Файл сохранён в папке загрузок. Да.")
        assert len(pieces) == 1

    def test_a_short_sentence_in_the_middle_survives(self):
        """«Готово.» between two long ones is a real sentence, not a fragment."""
        pieces = split_sentences(
            "Я нашла три подходящих файла в папке загрузок. Готово. "
            "Скажите, если нужно открыть какой-нибудь из них."
        )
        assert "Готово." in pieces

    def test_normalize_whitespace_collapses_runs(self):
        assert normalize_whitespace("  а\t\tб \n в ") == "а б в"

    def test_nothing_is_lost(self):
        """Every word of the answer must reach the speakers."""
        joined = " ".join(split_sentences(THREE_SENTENCES))
        for word in ("Первое", "Второе", "Третье"):
            assert word in joined


class TestLongSentences:
    """A wall of text still has to start sounding quickly."""

    def test_a_long_sentence_is_broken_up(self):
        text = "слово " * 200
        pieces = split_sentences(text)
        assert len(pieces) > 1

    def test_no_piece_exceeds_the_limit(self):
        text = "слово " * 200
        for piece in split_sentences(text):
            assert len(piece) <= MAX_CHUNK_CHARS

    def test_a_long_sentence_breaks_at_a_comma_when_it_can(self):
        text = ("часть один, " * 40).strip(" ,")
        pieces = split_sentences(text)
        assert any(piece.endswith(",") for piece in pieces)

    def test_a_custom_limit_is_honoured(self):
        """Within the fold threshold: the tail is merged back after the split."""
        pieces = split_sentences("слово " * 50, max_chars=40)
        assert all(len(piece) <= 40 + MIN_CHUNK_CHARS for piece in pieces)

    def test_no_word_is_cut_in_half(self):
        pieces = split_sentences("абвгдежз " * 60, max_chars=50)
        for piece in pieces:
            for word in piece.split():
                assert word == "абвгдежз"

    def test_a_disabled_limit_keeps_whole_sentences(self):
        """``max_chars=0`` is the escape hatch for a caller that splits its own way."""
        text = "слово " * 200
        assert len(split_sentences(text, max_chars=0)) == 1


# ----------------------------------------------------------------------
# the value types
# ----------------------------------------------------------------------


class TestVoiceSpec:
    """Voice identity, which everything downstream is keyed on."""

    def test_label_falls_back_to_the_id(self):
        assert VoiceSpec(engine="piper", voice_id="ru_RU-irina").label == "ru_RU-irina"

    def test_label_prefers_the_display_name(self):
        spec = VoiceSpec(engine="piper", voice_id="ru_RU-irina", display_name="Ирина")
        assert spec.label == "Ирина"

    def test_the_key_includes_the_path(self):
        """Two user models can share a stem in different folders."""
        first = VoiceSpec(engine="piper", voice_id="irina", path="C:/a/irina.onnx")
        second = VoiceSpec(engine="piper", voice_id="irina", path="C:/b/irina.onnx")
        assert first.key != second.key

    def test_same_voice_ignores_cosmetics(self):
        first = VoiceSpec(engine="piper", voice_id="irina", display_name="Ирина")
        second = VoiceSpec(engine="piper", voice_id="irina", display_name="IRINA")
        assert first.same_voice(second)

    def test_same_voice_is_false_for_nothing(self):
        assert not VoiceSpec(engine="piper", voice_id="irina").same_voice(None)

    def test_a_round_trip_through_params_preserves_everything(self):
        spec = VoiceSpec(
            engine="piper",
            voice_id="irina",
            path="C:/models/irina.onnx",
            language="ru",
            display_name="Ирина",
            sample_rate=22050,
        )
        assert VoiceSpec.from_params(spec.to_params()) == spec

    def test_from_params_tolerates_rubbish(self):
        """Params cross a pipe; a wrong type must not crash the worker."""
        spec = VoiceSpec.from_params({"engine": "piper", "sample_rate": "не число"})
        assert spec.sample_rate == 0


class TestAudioChunk:
    """Duration arithmetic, and what accompanies the descriptor in a reply."""

    def test_an_empty_chunk_knows_it(self):
        assert AudioChunk(b"").empty

    def test_frames_count_samples_not_bytes(self):
        chunk = AudioChunk(b"\0" * (SAMPLE_WIDTH * 100), 22050)
        assert chunk.frames == 100

    def test_stereo_frames_count_pairs(self):
        chunk = AudioChunk(b"\0" * (SAMPLE_WIDTH * 200), 22050, channels=2)
        assert chunk.frames == 100

    def test_duration_matches_the_rate(self):
        chunk = AudioChunk(tone(500, 22050), 22050)
        assert chunk.duration_ms == pytest.approx(500, abs=1)

    def test_a_rate_of_zero_does_not_divide_by_zero(self):
        assert AudioChunk(b"\0\0", 0).duration_ms == 0.0

    def test_with_pcm_keeps_the_format(self):
        chunk = AudioChunk(tone(10), 22050, channels=1)
        replaced = chunk.with_pcm(tone(20))
        assert replaced.sample_rate == 22050
        assert replaced.channels == 1

    def test_metadata_carries_the_format_but_not_the_bytes(self):
        meta = AudioChunk(tone(100), 22050).metadata()
        assert meta["sample_rate"] == 22050
        assert meta["channels"] == 1
        assert "pcm" not in meta

    def test_concat_joins_in_order(self):
        first = AudioChunk(b"\x01\x00", 22050)
        second = AudioChunk(b"\x02\x00", 22050)
        assert concat_chunks([first, second]).pcm == b"\x01\x00\x02\x00"

    def test_concat_skips_empty_chunks(self):
        chunks = [AudioChunk(b""), AudioChunk(b"\x01\x00", 22050), AudioChunk(b"")]
        assert concat_chunks(chunks).pcm == b"\x01\x00"

    def test_concat_of_nothing_is_empty(self):
        assert concat_chunks([]).empty

    def test_concat_refuses_mismatched_rates(self):
        """Joining them anyway would play the second half at the wrong speed."""
        with pytest.raises(TtsError):
            concat_chunks([AudioChunk(b"\x01\x00", 22050), AudioChunk(b"\x01\x00", 48000)])


class TestTtsOptions:
    """Worker params to engine options, tolerantly."""

    def test_defaults_are_sane(self):
        options = TtsOptions()
        assert options.speed == 1.0
        assert options.pitch == 1.0

    def test_params_are_read(self):
        options = TtsOptions.from_params({"speed": 1.5, "pitch": 0.9, "threads": 4})
        assert options.speed == 1.5
        assert options.pitch == 0.9
        assert options.threads == 4

    def test_missing_params_fall_back(self):
        assert TtsOptions.from_params({}).speed == 1.0

    def test_an_out_of_range_speed_is_clamped_not_refused(self):
        """A bad settings file must not make the assistant mute."""
        assert TtsOptions.from_params({"speed": 99.0}).speed == MAX_SPEED
        assert TtsOptions.from_params({"speed": 0.01}).speed == MIN_SPEED

    def test_pitch_is_clamped_the_same_way(self):
        assert TtsOptions.from_params({"pitch": 99.0}).pitch == MAX_PITCH
        assert TtsOptions.from_params({"pitch": -1.0}).pitch == MIN_PITCH

    def test_threads_never_drop_below_one(self):
        assert TtsOptions.from_params({"threads": 0}).threads == 1

    def test_clamp_helpers_agree_with_the_options(self):
        assert clamp_speed(99.0) == MAX_SPEED
        assert clamp_pitch(-5.0) == MIN_PITCH

    def test_a_round_trip_preserves_the_options(self):
        options = TtsOptions(speed=1.2, pitch=0.8, threads=3, gpu="cpu", sample_rate=48000)
        assert TtsOptions.from_params(options.to_params()) == options


# ----------------------------------------------------------------------
# the registry and the contract
# ----------------------------------------------------------------------


class TestEngineRegistry:
    """Names, lazy resolution, and the optional engine."""

    def test_the_three_engines_are_registered(self):
        assert set(ENGINE_ENTRYPOINTS) >= {"piper", "silero", "xtts"}

    def test_piper_and_silero_are_offered_even_uninstalled(self):
        """«Piper (не установлен)» is more useful than an option that is not there."""
        names = engine_names()
        assert "piper" in names
        assert "silero" in names

    def test_engine_class_does_not_construct(self):
        assert isinstance(engine_class("piper"), type)

    def test_an_unknown_engine_names_the_alternatives(self):
        with pytest.raises(TtsError) as excinfo:
            engine_class("не существует")
        assert "piper" in str(excinfo.value)

    def test_an_unknown_engine_has_a_russian_message(self):
        with pytest.raises(TtsError) as excinfo:
            engine_class("не существует")
        assert excinfo.value.user_message

    def test_available_only_filters_by_the_library(self):
        """Nothing vendor-specific is installed on the runner."""
        assert set(engine_names(available_only=True)) <= set(engine_names())

    def test_resolving_an_engine_imports_no_vendor_library(self):
        """Collection on a bare runner must not pull in torch."""
        import sys

        if "torch" in sys.modules:  # pragma: no cover - a developer machine with torch
            pytest.skip("torch уже импортирован чем-то другим")
        engine_class("silero")
        assert "torch" not in sys.modules


class TestEngineContract:
    """What :class:`TtsEngine` guarantees a subclass, using the stub."""

    def test_a_fresh_engine_is_not_loaded(self):
        assert not StubEngine().loaded

    def test_synthesizing_without_a_voice_is_an_error_not_a_crash(self):
        with pytest.raises(TtsError):
            StubEngine().synthesize("привет")

    def test_loading_makes_it_loaded(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        assert engine.loaded

    def test_empty_text_returns_empty_audio_without_calling_the_engine(self):
        """ "..." is a normal outcome, not an error."""
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        assert engine.synthesize("   ").empty
        assert engine.calls == []

    def test_a_per_call_voice_triggers_a_reload(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        engine.synthesize("привет", VoiceSpec(engine="stub", voice_id="stub-alt"))
        assert engine.loads == 2

    def test_the_same_voice_does_not_reload(self):
        engine = StubEngine()
        voice = VoiceSpec(engine="stub", voice_id="stub-ru")
        engine.load(voice, TtsOptions())
        engine.synthesize("привет", voice)
        assert engine.loads == 1

    def test_speed_reaches_the_implementation_clamped(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        engine.synthesize("привет", speed=99.0)
        assert engine.calls[-1][1] == MAX_SPEED

    def test_the_loaded_speed_is_used_when_none_is_given(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions(speed=1.4))
        engine.synthesize("привет")
        assert engine.calls[-1][1] == pytest.approx(1.4)

    def test_streaming_yields_one_chunk_per_sentence(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        chunks = list(engine.synthesize_stream(THREE_SENTENCES))
        assert len(chunks) == 3

    def test_streaming_folds_a_short_tail_into_the_previous_chunk(self):
        """Three one-word sentences are two chunks, not three.

        A trailing fragment shorter than :data:`MIN_CHUNK_CHARS` costs a whole
        synthesis call and comes back clipped, so the split merges it. Asserted
        here because the engine contract is where callers meet that rule.
        """
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        assert len(list(engine.synthesize_stream("Раз. Два. Три."))) == 2

    def test_streaming_skips_empty_chunks(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        assert list(engine.synthesize_stream("...")) == []

    def test_unload_is_safe_twice(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="stub-ru"), TtsOptions())
        engine.unload()
        engine.unload()
        assert not engine.loaded

    def test_the_sample_rate_prefers_the_voice(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="v", sample_rate=48000), TtsOptions())
        assert engine.sample_rate == 48000

    def test_the_sample_rate_falls_back_to_the_engine(self):
        engine = StubEngine()
        engine.load(VoiceSpec(engine="stub", voice_id="v"), TtsOptions())
        assert engine.sample_rate == DEFAULT_SAMPLE_RATE


class TestModelSize:
    """Measuring a voice on disk, which the RAM cap is computed from."""

    def test_a_file_is_its_own_size(self, tmp_path: Path):
        model = tmp_path / "voice.onnx"
        model.write_bytes(b"\0" * 2048)
        assert estimate_voice_bytes(model) == 2048

    def test_a_directory_is_summed_recursively(self, tmp_path: Path):
        directory = tmp_path / "xtts"
        (directory / "nested").mkdir(parents=True)
        (directory / "model.pth").write_bytes(b"\0" * 1024)
        (directory / "nested" / "config.json").write_bytes(b"\0" * 512)
        assert estimate_voice_bytes(directory) == 1536

    def test_a_missing_path_measures_zero(self, tmp_path: Path):
        """ "Cannot tell" reads as "let it through" — better than refusing to speak."""
        assert estimate_voice_bytes(tmp_path / "нет.onnx") == 0


class TestPiperEngine:
    """The parts of Piper that are pure functions of their inputs."""

    def test_the_engine_is_registered_under_its_name(self):
        assert engine_class("piper").name == "piper"

    def test_importing_the_module_does_not_import_piper(self):
        import sys

        from ayris.audio.tts import piper_engine

        assert piper_engine.PiperTtsEngine.name == "piper"
        if "piper" in sys.modules:  # pragma: no cover - a developer machine with piper
            pytest.skip("piper уже импортирован чем-то другим")
        assert "piper" not in sys.modules

    def test_voices_are_found_by_their_onnx_files(self, tmp_path: Path):
        (tmp_path / "ru_RU-irina-medium.onnx").write_bytes(b"\0")
        (tmp_path / "ru_RU-irina-medium.onnx.json").write_text("{}", encoding="utf-8")
        found = engine_class("piper").voices(tmp_path)
        assert [spec.voice_id for spec in found] == ["ru_RU-irina-medium"]

    def test_a_voice_without_a_config_is_not_offered(self, tmp_path: Path):
        """Piper cannot speak without the .json, so listing it would be a trap.

        The user would pick the voice, hear nothing, and get an error from
        :meth:`load` instead of from the combo box that made the promise.
        """
        (tmp_path / "ru_RU-denis.onnx").write_bytes(b"\0")
        assert engine_class("piper").voices(tmp_path) == ()

    def test_the_sample_rate_is_read_from_the_config(self, tmp_path: Path):
        (tmp_path / "voice.onnx").write_bytes(b"\0")
        (tmp_path / "voice.onnx.json").write_text(
            '{"audio": {"sample_rate": 16000}}', encoding="utf-8"
        )
        spec = engine_class("piper").voices(tmp_path)[0]
        assert spec.sample_rate == 16000

    def test_a_broken_config_does_not_crash_the_listing(self, tmp_path: Path):
        """One corrupt file must not take the whole settings window down."""
        (tmp_path / "плохой.onnx").write_bytes(b"\0")
        (tmp_path / "плохой.onnx.json").write_text("{не json", encoding="utf-8")
        (tmp_path / "хороший.onnx").write_bytes(b"\0")
        (tmp_path / "хороший.onnx.json").write_text("{}", encoding="utf-8")
        found = engine_class("piper").voices(tmp_path)
        assert [spec.voice_id for spec in found] == ["хороший"]

    def test_a_voice_renamed_to_a_plain_json_is_still_found(self, tmp_path: Path):
        """``voice.json`` is what a user ends up with after tidying a download."""
        (tmp_path / "voice.onnx").write_bytes(b"\0")
        (tmp_path / "voice.json").write_text("{}", encoding="utf-8")
        assert [spec.voice_id for spec in engine_class("piper").voices(tmp_path)] == ["voice"]

    def test_a_voice_in_a_subdirectory_is_found(self, tmp_path: Path):
        """Piper downloads arrive as a folder per voice."""
        nested = tmp_path / "ru" / "irina"
        nested.mkdir(parents=True)
        (nested / "voice.onnx").write_bytes(b"\0")
        (nested / "voice.onnx.json").write_text("{}", encoding="utf-8")
        assert engine_class("piper").voices(tmp_path)

    def test_a_missing_directory_yields_no_voices(self, tmp_path: Path):
        assert engine_class("piper").voices(tmp_path / "нет") == ()

    def test_loading_without_the_library_says_what_to_install(self, tmp_path: Path):
        model = tmp_path / "voice.onnx"
        model.write_bytes(b"\0")
        engine = engine_class("piper")()
        if engine.available():  # pragma: no cover - a developer machine with piper
            pytest.skip("piper установлен, ошибку импорта не воспроизвести")
        with pytest.raises(TtsError) as excinfo:
            engine.load(VoiceSpec(engine="piper", voice_id="voice", path=str(model)), TtsOptions())
        assert excinfo.value.user_message

    def test_loading_a_missing_model_is_a_clear_error(self, tmp_path: Path):
        engine = engine_class("piper")()
        with pytest.raises(TtsError):
            engine.load(
                VoiceSpec(engine="piper", voice_id="нет", path=str(tmp_path / "нет.onnx")),
                TtsOptions(),
            )

    @pytest.mark.hardware
    def test_a_real_voice_speaks(self, tmp_path: Path):  # pragma: no cover - needs a model
        pytest.skip("нужен настоящий .onnx — проверяется на машине пользователя")


class TestSileroEngine:
    """The same, for Silero."""

    def test_the_engine_is_registered_under_its_name(self):
        assert engine_class("silero").name == "silero"

    def test_the_russian_voices_are_listed_without_a_download(self):
        voices = engine_class("silero").voices()
        assert {spec.voice_id for spec in voices} >= {"baya", "kseniya", "xenia", "aidar"}

    def test_every_listed_voice_is_russian(self):
        assert all(spec.language == "ru" for spec in engine_class("silero").voices())

    def test_the_voices_declare_a_rate(self):
        assert all(spec.sample_rate > 0 for spec in engine_class("silero").voices())

    def test_loading_without_torch_says_what_to_install(self):
        engine = engine_class("silero")()
        if engine.available():  # pragma: no cover - a developer machine with torch
            pytest.skip("torch установлен, ошибку импорта не воспроизвести")
        with pytest.raises(TtsError) as excinfo:
            engine.load(VoiceSpec(engine="silero", voice_id="baya"), TtsOptions())
        assert excinfo.value.user_message

    def test_an_unknown_voice_is_refused(self):
        engine = engine_class("silero")()
        with pytest.raises(TtsError):
            engine.load(VoiceSpec(engine="silero", voice_id="нет такого"), TtsOptions())


class TestCoquiEngine:
    """XTTS is optional, and its absence must be quiet."""

    def test_it_is_marked_optional(self):
        assert engine_class("xtts").optional

    def test_it_is_hidden_when_not_installed(self):
        engine = engine_class("xtts")
        if engine.available():  # pragma: no cover - a developer machine with XTTS
            pytest.skip("XTTS установлен")
        assert "xtts" not in engine_names()

    def test_importing_the_module_does_not_import_tts(self):
        import sys

        from ayris.audio.tts import coqui_engine

        assert coqui_engine.CoquiTtsEngine.optional
        if "TTS" in sys.modules:  # pragma: no cover - a developer machine with XTTS
            pytest.skip("TTS уже импортирован чем-то другим")
        assert "TTS" not in sys.modules


# ----------------------------------------------------------------------
# the cache
# ----------------------------------------------------------------------


class TestPhraseCache:
    """Hit, miss, key sensitivity, eviction, and the limit."""

    @pytest.fixture
    def voice(self) -> VoiceSpec:
        return VoiceSpec(engine="stub", voice_id="stub-ru")

    @pytest.fixture
    def cache(self, tmp_path: Path) -> PhraseCache:
        return PhraseCache(tmp_path / "cache", limit_bytes=1024 * 1024)

    def test_a_fresh_cache_misses(self, cache: PhraseCache, voice: VoiceSpec):
        assert cache.get("привет", voice, 1.0, 1.0) is None

    def test_what_was_put_comes_back(self, cache: PhraseCache, voice: VoiceSpec):
        chunk = AudioChunk(tone(100), 22050)
        assert cache.put("привет", voice, 1.0, 1.0, chunk)
        assert cache.get("привет", voice, 1.0, 1.0) is not None

    def test_the_audio_survives_the_round_trip(self, cache: PhraseCache, voice: VoiceSpec):
        chunk = AudioChunk(tone(100), 22050)
        cache.put("привет", voice, 1.0, 1.0, chunk)
        restored = cache.get("привет", voice, 1.0, 1.0)
        assert restored is not None
        assert restored.pcm == chunk.pcm

    def test_the_format_survives_too(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50, 48000), 48000))
        restored = cache.get("привет", voice, 1.0, 1.0)
        assert restored is not None
        assert restored.sample_rate == 48000

    def test_a_different_text_misses(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.get("пока", voice, 1.0, 1.0) is None

    def test_a_different_speed_misses(self, cache: PhraseCache, voice: VoiceSpec):
        """Serving the 1.0x recording at 1.5x would ignore the setting."""
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.get("привет", voice, 1.5, 1.0) is None

    def test_a_different_pitch_misses(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.get("привет", voice, 1.0, 0.8) is None

    def test_a_different_voice_misses(self, cache: PhraseCache, voice: VoiceSpec):
        other = VoiceSpec(engine="stub", voice_id="stub-alt")
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.get("привет", other, 1.0, 1.0) is None

    def test_the_key_is_a_safe_file_name(self, voice: VoiceSpec):
        """Text reaches this from an LLM; a path separator in a key would escape."""
        key = phrase_key("C:\\Windows\\..\\секрет?*", voice, 1.0, 1.0)
        assert not set(key) & set('\\/:*?"<>|')

    def test_the_voice_key_prefixes_the_phrase_key(self, voice: VoiceSpec):
        """Invalidation on a voice change scans by prefix."""
        assert phrase_key("привет", voice, 1.0, 1.0).startswith(voice_key(voice))

    def test_a_disabled_cache_stores_nothing(self, tmp_path: Path, voice: VoiceSpec):
        cache = PhraseCache(tmp_path / "off", limit_bytes=0)
        assert not cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.get("привет", voice, 1.0, 1.0) is None

    def test_an_empty_chunk_is_not_stored(self, cache: PhraseCache, voice: VoiceSpec):
        assert not cache.put("привет", voice, 1.0, 1.0, AudioChunk(b""))

    def test_an_over_long_text_is_not_stored(self, cache: PhraseCache, voice: VoiceSpec):
        """A whole essay would evict every phrase that actually recurs."""
        long_text = "а" * (MAX_TEXT_CHARS + 1)
        assert not cache.put(long_text, voice, 1.0, 1.0, AudioChunk(tone(50), 22050))

    def test_an_entry_larger_than_a_quarter_of_the_budget_is_refused(
        self, tmp_path: Path, voice: VoiceSpec
    ):
        cache = PhraseCache(tmp_path / "small", limit_bytes=8192)
        assert not cache.put("длинная фраза", voice, 1.0, 1.0, AudioChunk(tone(1000), 22050))

    def test_a_corrupt_entry_is_a_miss_and_is_deleted(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        path = cache.directory / phrase_key("привет", voice, 1.0, 1.0)
        path.write_bytes(b"this is not a wav file")
        assert cache.get("привет", voice, 1.0, 1.0) is None
        assert not path.exists()

    def test_the_cache_evicts_the_least_recently_used(self, tmp_path: Path, voice: VoiceSpec):
        """Twelve entries of ~9 KB into a 40 KB budget: most of them have to go."""
        cache = PhraseCache(tmp_path / "lru", limit_bytes=40_000)
        for index in range(12):
            cache.put(f"фраза номер {index}", voice, 1.0, 1.0, AudioChunk(tone(200), 22050))
        assert cache.stats().bytes_used <= cache.limit_bytes

    def test_eviction_is_counted(self, tmp_path: Path, voice: VoiceSpec):
        cache = PhraseCache(tmp_path / "lru", limit_bytes=40_000)
        for index in range(12):
            cache.put(f"фраза номер {index}", voice, 1.0, 1.0, AudioChunk(tone(200), 22050))
        assert cache.stats().evictions > 0

    def test_the_newest_phrase_survives_the_eviction(self, tmp_path: Path, voice: VoiceSpec):
        """Evicting what was just stored would make the cache pure overhead."""
        cache = PhraseCache(tmp_path / "lru", limit_bytes=40_000)
        for index in range(12):
            cache.put(f"фраза номер {index}", voice, 1.0, 1.0, AudioChunk(tone(200), 22050))
        assert cache.get("фраза номер 11", voice, 1.0, 1.0) is not None

    def test_lowering_the_limit_trims_immediately(self, cache: PhraseCache, voice: VoiceSpec):
        """Leaving 200 MB on disk after the user asked for 50 makes the setting a lie."""
        for index in range(4):
            cache.put(f"фраза {index}", voice, 1.0, 1.0, AudioChunk(tone(200), 22050))
        cache.set_limit(20_000)
        assert cache.stats().bytes_used <= 20_000

    def test_setting_the_limit_to_zero_clears_everything(
        self, cache: PhraseCache, voice: VoiceSpec
    ):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        cache.set_limit(0)
        assert cache.stats().entries == 0

    def test_invalidating_a_voice_keeps_only_that_voice(self, cache: PhraseCache, voice: VoiceSpec):
        other = VoiceSpec(engine="stub", voice_id="stub-alt")
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        cache.put("привет", other, 1.0, 1.0, AudioChunk(tone(50), 22050))
        cache.invalidate(other)
        assert cache.get("привет", other, 1.0, 1.0) is not None
        assert cache.get("привет", voice, 1.0, 1.0) is None

    def test_invalidating_everything_leaves_nothing(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.invalidate() == 1
        assert cache.stats().entries == 0

    def test_hits_and_misses_are_counted(self, cache: PhraseCache, voice: VoiceSpec):
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        cache.get("привет", voice, 1.0, 1.0)
        cache.get("пока", voice, 1.0, 1.0)
        stats = cache.stats()
        assert stats.hits == 1
        assert stats.misses == 1

    def test_the_hit_rate_of_an_untouched_cache_is_zero(self):
        assert CacheStats().hit_rate == 0.0

    def test_the_hit_rate_is_a_fraction(self):
        assert CacheStats(hits=3, misses=1).hit_rate == pytest.approx(0.75)

    def test_the_directory_is_created_on_demand(self, tmp_path: Path, voice: VoiceSpec):
        cache = PhraseCache(tmp_path / "не" / "существует", limit_bytes=1024 * 1024)
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        assert cache.directory.is_dir()

    def test_the_stored_file_is_a_readable_wav(self, cache: PhraseCache, voice: VoiceSpec):
        """Debuggable by hand: a developer should be able to play a cache entry."""
        cache.put("привет", voice, 1.0, 1.0, AudioChunk(tone(50), 22050))
        path = cache.directory / phrase_key("привет", voice, 1.0, 1.0)
        with wave.open(str(path), "rb") as handle:
            assert handle.getframerate() == 22050
            assert handle.getsampwidth() == SAMPLE_WIDTH


# ----------------------------------------------------------------------
# the worker
# ----------------------------------------------------------------------


class TestWorkerSynthesis:
    """Text in, shared memory out."""

    def test_starting_the_worker_loads_nothing(self, worker: TtsWorker):
        """A start that loads a voice is a start the supervisor times out on."""
        assert worker._engine is None
        assert worker.status({})["loaded"] is False

    def test_synthesis_returns_audio(self, worker: TtsWorker):
        reply = worker.synthesize({"text": "Привет."})
        assert reply["empty"] is False
        assert len(pcm_of(reply)) > 0

    def test_the_reply_describes_the_format(self, worker: TtsWorker):
        reply = worker.synthesize({"text": "Привет."})
        assert reply["sample_rate"] == DEFAULT_SAMPLE_RATE
        assert reply["channels"] == 1
        assert reply["duration_ms"] > 0

    def test_the_audio_does_not_travel_in_the_reply(self, worker: TtsWorker):
        """A sentence is 100-200 KB; pickling it would copy it twice."""
        reply = worker.synthesize({"text": "Привет."})
        assert "pcm" not in reply

    def test_empty_text_is_not_an_error(self, worker: TtsWorker):
        assert worker.synthesize({"text": "   "})["empty"] is True

    def test_punctuation_only_produces_nothing(self, worker: TtsWorker):
        assert worker.synthesize({"text": "..."})["empty"] is True

    def test_the_request_id_comes_back(self, worker: TtsWorker):
        """The player carries it into ``TtsStarted`` so the overlay can match."""
        reply = worker.synthesize({"text": "Привет.", "request_id": "req-1"})
        assert reply["request_id"] == "req-1"

    def test_several_sentences_arrive_as_one_buffer(self, worker: TtsWorker):
        one = worker.synthesize({"text": S1})
        many = worker.synthesize({"text": THREE_SENTENCES})
        assert len(pcm_of(many)) > len(pcm_of(one))

    def test_every_sentence_reaches_the_engine(self, worker: TtsWorker):
        worker.synthesize({"text": THREE_SENTENCES})
        assert [call[0] for call in engine_of(worker).calls] == [S1, S2, S3]

    def test_speed_from_the_request_wins_over_the_settings(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет.", "speed": 1.5})
        assert engine_of(worker).calls[-1][1] == pytest.approx(1.5)

    def test_an_absurd_speed_is_clamped(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет.", "speed": 99.0})
        assert engine_of(worker).calls[-1][1] == MAX_SPEED

    def test_the_configured_speed_is_used_by_default(self, worker: TtsWorker):
        worker.context._params["speed"] = 1.3  # type: ignore[attr-defined]
        worker.synthesize({"text": "Привет."})
        assert engine_of(worker).calls[-1][1] == pytest.approx(1.3)

    def test_a_failing_engine_raises_a_typed_error(
        self, worker: TtsWorker, stub_engine: StubEngine
    ):
        stub_engine.fail_with = TtsError("движок сломался")
        with pytest.raises(TtsError):
            worker.synthesize({"text": "Привет."})

    def test_a_released_block_is_forgotten(self, worker: TtsWorker):
        reply = worker.synthesize({"text": "Привет."})
        assert worker.release({"block": reply["block"]})["released"] == 1
        assert worker.status({})["open_blocks"] == 0

    def test_releasing_everything_works_without_a_name(self, worker: TtsWorker):
        worker.synthesize({"text": S1})
        worker.synthesize({"text": S2})
        assert worker.release({})["released"] == 2

    def test_stopping_frees_the_blocks(self, worker: TtsWorker):
        """A leaked mapping outlives the process that made it."""
        worker.synthesize({"text": "Привет."})
        worker.on_stop()
        assert worker._blocks == {}

    def test_voices_can_be_listed_without_loading(self, worker: TtsWorker):
        reply = worker.voices({"engine": "stub"})
        assert [entry["voice_id"] for entry in reply["voices"]] == ["stub-ru", "stub-alt"]
        assert worker.status({})["loaded"] is False


class TestWorkerStreaming:
    """Sentence by sentence, so the first sound is not held up by the last."""

    def test_the_first_call_returns_the_first_sentence_only(self, worker: TtsWorker):
        worker.synthesize_stream({"text": THREE_SENTENCES})
        assert engine_of(worker).calls == [(S1, 1.0, 1.0)]

    def test_a_stream_token_is_handed_out(self, worker: TtsWorker):
        assert worker.synthesize_stream({"text": TWO_SENTENCES})["stream"]

    def test_remaining_counts_what_is_left(self, worker: TtsWorker):
        assert worker.synthesize_stream({"text": THREE_SENTENCES})["remaining"] == 2

    def test_a_single_sentence_opens_no_stream(self, worker: TtsWorker):
        """Nothing to come back for; a token would only have to be cleaned up."""
        reply = worker.synthesize_stream({"text": S1})
        assert reply["stream"] == ""
        assert reply["remaining"] == 0

    def test_nothing_speakable_opens_no_stream(self, worker: TtsWorker):
        reply = worker.synthesize_stream({"text": "..."})
        assert reply["empty"] is True
        assert reply["stream"] == ""

    def test_the_sentences_arrive_in_order(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": THREE_SENTENCES})["stream"]
        worker.next_chunk({"stream": token})
        worker.next_chunk({"stream": token})
        assert [call[0] for call in engine_of(worker).calls] == [S1, S2, S3]

    def test_the_stream_closes_on_the_last_sentence(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": TWO_SENTENCES})["stream"]
        reply = worker.next_chunk({"stream": token})
        assert reply["stream"] == ""
        assert reply["remaining"] == 0

    def test_asking_past_the_end_is_empty_not_an_error(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": TWO_SENTENCES})["stream"]
        worker.next_chunk({"stream": token})
        assert worker.next_chunk({"stream": token})["empty"] is True

    def test_an_unknown_token_is_empty_not_an_error(self, worker: TtsWorker):
        """A worker restart between calls must not crash the player."""
        assert worker.next_chunk({"stream": "нет такого"})["empty"] is True

    def test_a_finished_stream_is_forgotten(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": TWO_SENTENCES})["stream"]
        worker.next_chunk({"stream": token})
        assert worker.status({})["open_streams"] == 0

    def test_each_chunk_carries_its_own_audio(self, worker: TtsWorker):
        """The stub's duration follows the text, so length identifies the piece."""
        first = worker.synthesize_stream(
            {"text": "Совсем коротко. Значительно более длинное второе предложение."}
        )
        second = worker.next_chunk({"stream": first["stream"]})
        assert len(pcm_of(second)) > len(pcm_of(first))

    def test_abandoned_streams_are_eventually_dropped(self, worker: TtsWorker, monkeypatch):
        """A caller that crashed mid-answer must not pin sentences forever."""
        from ayris.workers import tts_worker as module

        worker.synthesize_stream({"text": TWO_SENTENCES})
        monkeypatch.setattr(module, "_STREAM_TTL_SEC", -1.0)
        worker.synthesize_stream({"text": TWO_SENTENCES})
        assert worker.status({})["open_streams"] == 1


class TestWorkerCancel:
    """What «Айрис, стоп» stops."""

    def test_cancel_reports_itself(self, worker: TtsWorker):
        assert worker.cancel({})["cancelled"] is True

    def test_cancel_drops_the_open_streams(self, worker: TtsWorker):
        worker.synthesize_stream({"text": THREE_SENTENCES})
        assert worker.cancel({})["streams_dropped"] == 1
        assert worker.status({})["open_streams"] == 0

    def test_no_further_sentence_is_synthesized_after_a_cancel(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": THREE_SENTENCES})["stream"]
        worker.cancel({})
        worker.next_chunk({"stream": token})
        assert [call[0] for call in engine_of(worker).calls] == [S1]

    def test_the_cancelled_stream_answers_empty(self, worker: TtsWorker):
        token = worker.synthesize_stream({"text": TWO_SENTENCES})["stream"]
        worker.cancel({})
        assert worker.next_chunk({"stream": token})["empty"] is True

    def test_a_cancel_mid_answer_stops_the_remaining_sentences(self, worker: TtsWorker):
        """The whole point: «стоп» during a long answer must cut it short.

        The sentence already inside the engine finishes - no CPU synthesizer can
        be interrupted mid-inference - but the ones after it are never started.
        """
        engine = engine_of_started(worker)
        engine.on_call = lambda: worker.cancel({})
        with pytest.raises(TtsError):
            worker.synthesize({"text": THREE_SENTENCES})
        assert [call[0] for call in engine.calls] == [S1]

    def test_a_cancelled_synthesis_produces_no_audio_at_all(self, worker: TtsWorker):
        """Half an answer is worse than none: the user asked for silence."""
        engine = engine_of_started(worker)
        engine.on_call = lambda: worker.cancel({})
        with pytest.raises(TtsError) as excinfo:
            worker.synthesize({"text": THREE_SENTENCES})
        assert excinfo.value.user_message
        assert worker.status({})["open_blocks"] == 0

    def test_cancel_frees_the_shared_blocks(self, worker: TtsWorker):
        worker.synthesize({"text": S1})
        worker.cancel({})
        assert worker.status({})["open_blocks"] == 0

    def test_the_next_request_works_after_a_cancel(self, worker: TtsWorker):
        """A cancel silences one answer, not the assistant."""
        worker.cancel({})
        assert worker.synthesize({"text": S1})["empty"] is False

    def test_a_supervisor_cancel_stops_synthesis_too(self, worker: TtsWorker):
        """``CancelRequested`` is not the only way an answer is abandoned."""
        worker.context.cancelled = True  # type: ignore[attr-defined]
        with pytest.raises(TtsError):
            worker.synthesize({"text": TWO_SENTENCES})

    def test_the_cancel_echoes_the_request_id(self, worker: TtsWorker):
        assert worker.cancel({"request_id": "req-9"})["request_id"] == "req-9"


class TestWorkerLifecycle:
    """Lazy load, idle unload, reload, and the memory cap."""

    def test_the_first_synthesis_brings_the_voice_up(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет."})
        assert engine_of(worker).loads == 1
        assert worker.status({})["loaded"] is True

    def test_the_second_synthesis_reuses_it(self, worker: TtsWorker):
        worker.synthesize({"text": "Раз."})
        worker.synthesize({"text": "Два."})
        assert engine_of(worker).loads == 1

    def test_load_voice_reports_the_details(self, worker: TtsWorker):
        reply = worker.load_voice({})
        assert reply["loaded"] is True
        assert reply["voice"] == "stub-ru"
        assert reply["sample_rate"] == DEFAULT_SAMPLE_RATE

    def test_load_voice_twice_loads_once(self, worker: TtsWorker):
        worker.load_voice({})
        worker.load_voice({})
        assert engine_of(worker).loads == 1

    def test_a_finished_load_announces_itself(self, worker: TtsWorker):
        worker.load_voice({})
        assert "loaded" in [event["state"] for event in worker.context.events_of("voice")]

    def test_a_failed_load_announces_itself(self, worker: TtsWorker, stub_engine: StubEngine):
        """Silence with nothing on screen reads as a hang, so the bus hears it."""
        stub_engine.fail_load_with = TtsError(
            "stub: load failed", user_message="Файл голоса повреждён."
        )
        with pytest.raises(TtsError):
            worker.load_voice({})
        failures = [
            event for event in worker.context.events_of("voice") if event["state"] == "failed"
        ]
        assert failures
        assert failures[-1]["error"] == "Файл голоса повреждён."

    def test_a_failed_load_leaves_nothing_half_loaded(
        self, worker: TtsWorker, stub_engine: StubEngine
    ):
        stub_engine.fail_load_with = TtsError("stub: load failed")
        with pytest.raises(TtsError):
            worker.load_voice({})
        assert worker.status({})["loaded"] is False

    def test_unloading_gives_the_memory_back(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет."})
        engine = engine_of(worker)
        assert worker.unload({})["unloaded"] is True
        assert engine.unloads == 1
        assert worker.status({})["loaded"] is False

    def test_unloading_when_nothing_is_loaded_is_not_an_error(self, worker: TtsWorker):
        assert worker.unload({})["unloaded"] is False

    def test_an_unload_is_announced(self, worker: TtsWorker):
        worker.load_voice({})
        worker.unload({})
        assert "unloaded" in [event["state"] for event in worker.context.events_of("voice")]

    def test_speaking_after_an_unload_loads_again(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет."})
        worker.unload({})
        worker.synthesize({"text": "Привет."})
        assert engine_of(worker).loads == 2

    def test_changing_the_voice_reloads(self, worker: TtsWorker):
        worker.load_voice({})
        worker.load_voice({"voice": "stub-alt"})
        assert engine_of(worker).loads == 2

    def test_a_settings_change_of_voice_unloads(self, worker: TtsWorker):
        worker.load_voice({})
        worker.on_configure({"engine": "stub", "voice": "stub-alt"})
        assert worker.status({})["loaded"] is False

    def test_a_settings_change_of_speed_does_not_unload(self, worker: TtsWorker):
        """Speed is per call; dropping the voice for it would cost seconds."""
        worker.load_voice({})
        worker.on_configure({"engine": "stub", "voice": "stub-ru", "speed": 1.4})
        assert worker.status({})["loaded"] is True

    def test_a_voice_that_does_not_fit_is_refused(self, worker: TtsWorker, tmp_path: Path):
        model = tmp_path / "huge.onnx"
        model.write_bytes(b"\0" * (4 * 1024 * 1024))
        worker.context._params["ram_limit_mb"] = 1  # type: ignore[attr-defined]
        with pytest.raises(TtsError) as excinfo:
            worker.load_voice({"voice": str(model)})
        assert excinfo.value.user_message

    def test_the_refusal_names_a_way_out(self, worker: TtsWorker, tmp_path: Path):
        model = tmp_path / "huge.onnx"
        model.write_bytes(b"\0" * (4 * 1024 * 1024))
        worker.context._params["ram_limit_mb"] = 1  # type: ignore[attr-defined]
        with pytest.raises(TtsError) as excinfo:
            worker.load_voice({"voice": str(model)})
        assert "настройк" in excinfo.value.user_message.lower()

    def test_a_voice_that_fits_is_allowed(self, worker: TtsWorker, tmp_path: Path):
        model = tmp_path / "small.onnx"
        model.write_bytes(b"\0" * 1024)
        worker.context._params["ram_limit_mb"] = 64  # type: ignore[attr-defined]
        assert worker.load_voice({"voice": str(model)})["loaded"] is True

    def test_a_disabled_cap_lets_anything_through(self, worker: TtsWorker, tmp_path: Path):
        model = tmp_path / "huge.onnx"
        model.write_bytes(b"\0" * (4 * 1024 * 1024))
        worker.context._params["ram_limit_mb"] = 0  # type: ignore[attr-defined]
        assert worker.load_voice({"voice": str(model)})["loaded"] is True

    def test_no_voice_configured_is_a_clear_error(self, worker: TtsWorker):
        worker.context._params["voice"] = ""  # type: ignore[attr-defined]
        with pytest.raises(TtsError) as excinfo:
            worker.load_voice({})
        assert excinfo.value.user_message


class TestIdleTimeout:
    """The timeout itself, without sleeping through it."""

    def test_zero_disables_the_unload(self, worker: TtsWorker):
        worker.context._params["model_idle_sec"] = 0  # type: ignore[attr-defined]
        assert worker._idle_timeout() == 0.0

    def test_the_configured_value_is_used(self, worker: TtsWorker):
        worker.context._params["model_idle_sec"] = 300  # type: ignore[attr-defined]
        assert worker._idle_timeout() == pytest.approx(300.0)

    def test_eco_mode_halves_it(self, worker: TtsWorker):
        worker.context._params["model_idle_sec"] = 600  # type: ignore[attr-defined]
        worker.context._params["eco_mode"] = True  # type: ignore[attr-defined]
        assert worker._idle_timeout() == pytest.approx(300.0)

    def test_it_never_drops_below_the_floor(self, worker: TtsWorker):
        """Unloading after five seconds would reload on every second sentence."""
        worker.context._params["model_idle_sec"] = 1  # type: ignore[attr-defined]
        assert worker._idle_timeout() >= 30.0

    def test_an_idle_voice_is_dropped(self, worker: TtsWorker):
        worker.context._params["model_idle_sec"] = 60  # type: ignore[attr-defined]
        worker.load_voice({})
        worker._last_used -= 3600
        worker._drop_if_idle()
        assert worker.status({})["loaded"] is False

    def test_a_recently_used_voice_stays(self, worker: TtsWorker):
        worker.context._params["model_idle_sec"] = 60  # type: ignore[attr-defined]
        worker.load_voice({})
        worker._drop_if_idle()
        assert worker.status({})["loaded"] is True

    def test_an_open_stream_holds_the_voice(self, worker: TtsWorker):
        """Unloading between two sentences of one answer would stall it audibly."""
        worker.context._params["model_idle_sec"] = 60  # type: ignore[attr-defined]
        worker.synthesize_stream({"text": TWO_SENTENCES})
        worker._last_used -= 3600
        worker._drop_if_idle()
        assert worker.status({})["loaded"] is True


class TestWorkerCacheIntegration:
    """The worker actually uses the cache, and drops it when the voice changes."""

    def test_the_second_identical_phrase_is_not_synthesized(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        cached_worker.synthesize({"text": "Готово."})
        assert len(engine_of(cached_worker).calls) == 1

    def test_the_hit_is_reported(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        assert cached_worker.synthesize({"text": "Готово."})["cached"] is True

    def test_a_miss_is_reported_as_such(self, cached_worker: TtsWorker):
        assert cached_worker.synthesize({"text": "Готово."})["cached"] is False

    def test_the_cached_audio_is_the_same_audio(self, cached_worker: TtsWorker):
        first = pcm_of(cached_worker.synthesize({"text": "Готово."}))
        second = pcm_of(cached_worker.synthesize({"text": "Готово."}))
        assert first == second

    def test_a_different_speed_is_synthesized_again(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        cached_worker.synthesize({"text": "Готово.", "speed": 1.5})
        assert len(engine_of(cached_worker).calls) == 2

    def test_streaming_uses_the_cache_per_sentence(self, cached_worker: TtsWorker):
        """The same «Готово.» recurs at the end of many answers."""
        tail = "И ещё кое-что важное здесь."
        cached_worker.synthesize({"text": "Готово."})
        reply = cached_worker.synthesize_stream({"text": f"Готово. {tail}"})
        cached_worker.next_chunk({"stream": reply["stream"]})
        assert [call[0] for call in engine_of(cached_worker).calls] == ["Готово.", tail]

    def test_a_cached_first_sentence_still_returns_audio(self, cached_worker: TtsWorker):
        """A hit must not turn into silence at the front of the answer."""
        cached_worker.synthesize({"text": "Готово."})
        reply = cached_worker.synthesize_stream({"text": "Готово. И ещё кое-что важное."})
        assert len(pcm_of(reply)) > 0

    def test_the_cache_stats_reach_the_status(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        cached_worker.synthesize({"text": "Готово."})
        assert cached_worker.status({})["cache"]["hits"] == 1

    def test_a_disabled_cache_never_hits(self, worker: TtsWorker):
        worker.synthesize({"text": "Готово."})
        worker.synthesize({"text": "Готово."})
        assert len(engine_of(worker).calls) == 2

    def test_changing_the_voice_invalidates_the_cache(self, cached_worker: TtsWorker):
        """Speaking Irina's recording in Denis's voice is the bug nobody looks for."""
        cached_worker.synthesize({"text": "Готово."})
        cached_worker.load_voice({"voice": "stub-alt"})
        assert cached_worker.status({})["cache"]["entries"] == 0

    def test_clear_cache_empties_it(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        assert cached_worker.clear_cache({})["removed"] == 1
        assert cached_worker.status({})["cache"]["entries"] == 0

    def test_the_limit_follows_the_settings(self, worker: TtsWorker):
        worker.context._params["cache_size_mb"] = 8  # type: ignore[attr-defined]
        worker.on_configure(dict(worker.context.params))
        assert worker._cache.limit_bytes == 8 * 1024 * 1024


class TestWorkerMetrics:
    """The numbers DevTools and the pipeline log show."""

    def test_calls_are_counted(self, worker: TtsWorker):
        worker.synthesize({"text": S1})
        worker.synthesize({"text": S2})
        assert worker.status({})["calls"] == 2

    def test_metrics_are_emitted_per_phrase(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет.", "request_id": "req-2"})
        events = worker.context.events_of("metrics")
        assert events[-1]["request_id"] == "req-2"
        assert events[-1]["audio_ms"] > 0

    def test_a_cache_hit_is_marked_in_the_metrics(self, cached_worker: TtsWorker):
        cached_worker.synthesize({"text": "Готово."})
        cached_worker.synthesize({"text": "Готово."})
        assert cached_worker.context.events_of("metrics")[-1]["cached"] is True

    def test_the_real_time_factor_is_reported(self, worker: TtsWorker):
        worker.synthesize({"text": "Привет."})
        assert worker.status({})["real_time_factor"] >= 0.0

    def test_status_works_before_anything_happened(self, worker: TtsWorker):
        """DevTools opens before the first phrase is spoken."""
        status = worker.status({})
        assert status["calls"] == 0
        assert status["loaded"] is False

    def test_status_lists_the_engines(self, worker: TtsWorker):
        assert isinstance(worker.status({})["available_engines"], list)


# ----------------------------------------------------------------------
# wiring
# ----------------------------------------------------------------------


class TestEventTranslation:
    """What a worker event becomes on the bus."""

    def test_a_failed_load_becomes_a_notification(self):
        event = translate_tts_event("voice", {"state": "failed", "error": "нет файла"})
        assert isinstance(event, NotificationRequested)
        assert event.level == "error"

    def test_the_notification_carries_the_reason(self):
        event = translate_tts_event("voice", {"state": "failed", "error": "нет файла"})
        assert isinstance(event, NotificationRequested)
        assert "нет файла" in event.message

    def test_a_failure_without_a_reason_still_says_something(self):
        event = translate_tts_event("voice", {"state": "failed"})
        assert isinstance(event, NotificationRequested)
        assert event.message

    def test_a_successful_load_is_not_announced_to_the_user(self):
        """A toast on every load would fire on every idle unload cycle."""
        assert translate_tts_event("voice", {"state": "loaded"}) is None

    def test_metrics_stay_off_the_bus(self):
        """One event per sentence in front of every subscriber buys nothing."""
        assert translate_tts_event("metrics", {"chars": 10}) is None

    def test_an_unknown_kind_is_ignored(self):
        assert translate_tts_event("что-то новое", {}) is None

    def test_the_module_exports_the_translator(self):
        assert EVENT_TRANSLATOR is translate_tts_event


class TestRegistryWiring:
    """The supervisor can actually find and configure this worker."""

    def test_the_worker_declares_its_kind(self):
        assert TtsWorker.kind == "tts"

    def test_the_registered_entrypoint_names_this_class(self):
        from ayris.workers.registry import WorkerKind, worker_type

        entry = worker_type(WorkerKind.TTS.value)
        assert entry is not None
        assert entry.entrypoint == "ayris.workers.tts_worker:TtsWorker"

    def test_the_spec_carries_what_the_worker_reads(self):
        """Every params key the worker asks for has to be in the spec it gets."""
        from ayris.core.config import Settings
        from ayris.workers.registry import WorkerKind, worker_type

        entry = worker_type(WorkerKind.TTS.value)
        assert entry is not None
        params = entry.build(Settings()).params
        for key in (
            "engine",
            "voice",
            "speed",
            "pitch",
            "volume",
            "output_device",
            "cache_size_mb",
            "model_idle_sec",
            "ram_limit_mb",
        ):
            assert key in params

    def test_the_spec_defaults_to_piper(self):
        from ayris.core.config import Settings
        from ayris.workers.registry import WorkerKind, worker_type

        entry = worker_type(WorkerKind.TTS.value)
        assert entry is not None
        assert entry.build(Settings()).params["engine"] == "piper"

    def test_the_translator_is_registered_for_this_kind(self):
        from ayris.workers.registry import event_translator

        assert event_translator("tts") is translate_tts_event
