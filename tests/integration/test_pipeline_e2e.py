"""Сквозной проход пайплайна на РЕАЛЬНОМ фронтенде и настоящих WAV-фикстурах.

``tests/integration/test_pipeline.py`` (задача 18) проверяет ЛОГИКУ диспетчера
на синтетических буферах ``b"\\x01\\x00"*n`` и на прямых ``Rig.wake()``/``spoke()``:
там нет ни настоящего движка wake, ни сегментатора, ни роутера STT — вход всегда
подставной. Именно поэтому шов «wake + segmenter + router» ничем не покрыт: никто
не проверяет, что реальное решение движка на реальном WAV доезжает до пайплайна,
что нарезанный сегментатором буфер уходит в распознавание, а его результат — в
настоящий матчер и дальше в действие/озвучку.

Поэтому здесь собран настоящий фронтенд по краям пайплайна:

* реальный :class:`FormantEngine` в реальном :class:`WakeWordDetector` слушает
  ``wake_ayris.wav`` и сам решает, что это «айрис»; его детект публикуется в
  пайплайн как :class:`WakeWordDetected`;
* реальный :func:`segment_pcm` нарезает ``phrase.wav`` в один принятый сегмент,
  и этот PCM становится буфером, который отдаёт ``phrase_source``;
* буфер уходит в реальный :class:`SttRouter` (режим OFFLINE, движок — стаб без
  сети и GPU), а его :class:`TranscriptResult` — в реальный :class:`Matcher`.

Подставлены только края, которые в бою и так внешние: действие, озвучка и модель
(их подставляет и задача 18). Всё между ними — настоящее. Проверяются четыре
исхода на этой живой цепочке: полный голосовой проход до действия (COMMANDS),
негатив движка на ``wake_absent.wav``, промах матчера с фолбэком в чат-LLM
(HYBRID) и промах без модели (COMMANDS → «не нашла такую команду»).

Импорты-хелперы скопированы из юнит-тестов намеренно: перекрёстные импорты между
тестовыми модулями запрещены (нет ``__init__.py``, линт режет относительные), так
что двойники и загрузчики фикстур живут в каждом файле своей копией.

Ни ``sleep``, ни настоящих таймеров: шина инлайновая, ``inline_runner`` гонит
проход синхронно на потоке теста, планировщик ручной. Единственный живой поток —
демон детектора wake; он всегда останавливается в ``finally``.
"""

from __future__ import annotations

import math
import wave
from array import array
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter, sleep
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.audio.segmenter import EndReason, SpeechSegment, segment_pcm
from ayris.audio.stt.base import AudioBuffer, TranscriptResult
from ayris.audio.stt.router import SttMode, SttRouter
from ayris.audio.wake_word import (
    WAKE_SAMPLE_RATE,
    ModelSpec,
    WakeDetection,
    WakePhrase,
    WakeWordCallbacks,
    WakeWordDetector,
    WakeWordEngine,
    WakeWordSettings,
)
from ayris.audio.wake_word.manager import DEFAULT_DEBOUNCE_MS
from ayris.core.config import Settings
from ayris.core.errors import ActionError, TtsError
from ayris.core.events import (
    EventBus,
    PipelineStateChanged,
    SpeechEnded,
    WakeWordDetected,
)
from ayris.core.pipeline import (
    NOT_MATCHED_MESSAGE,
    ActionOutcome,
    ActionRequest,
    Pipeline,
    inline_runner,
)
from ayris.core.pipeline_states import ManualScheduler
from ayris.nlu.llm.base import (
    FinishReason,
    LlmClient,
    LlmMessage,
    LlmResponse,
    LlmTool,
)
from ayris.nlu.matcher import Matcher, Trigger, TriggerKind

if TYPE_CHECKING:
    pass

pytestmark = pytest.mark.integration

#: Каталог с закоммиченными WAV (16 кГц, моно, 16 бит). Общий для wake и VAD.
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "audio"

#: Слово активации, которое произносит ``wake_ayris.wav``.
AYRIS = "айрис"

#: Команда, которая ЕСТЬ в :func:`library` — её вернёт стаб STT в позитивном
#: проходе, чтобы матчер нашёл ``command_id=8``.
COMMAND_TEXT = "открой браузер"

#: Фраза, которой в библиотеке нет ни в каком виде: на ней проверяются оба
#: промаха — фолбэк в чат (HYBRID) и «не нашла» (COMMANDS).
UNKNOWN_TEXT = "расскажи что-нибудь про черепах"

#: Как долго :func:`feed` ждёт поток детектора. Щедро: фильтрация идёт на чистом
#: Python, а нагруженный CI-раннер медленный.
_FEED_TIMEOUT_SEC = 30.0


def wake_pcm(name: str) -> bytes:
    """Прочитать WAV-фикстуру и вернуть сырой PCM (моно, 16 бит, 16 кГц)."""
    with wave.open(str(FIXTURES / name), "rb") as fp:
        assert fp.getnchannels() == 1, f"{name}: должен быть моно"
        assert fp.getsampwidth() == SAMPLE_WIDTH, f"{name}: должен быть 16 бит"
        assert fp.getframerate() == WAKE_SAMPLE_RATE, f"{name}: должен быть 16 кГц"
        return fp.readframes(fp.getnframes())


def feed(detector: WakeWordDetector, pcm: bytes, *, frame_samples: int) -> None:
    """Протолкнуть аудио в детектор и дождаться, пока поток его обсчитает."""
    target = detector.stats.frames + len(pcm) // (frame_samples * SAMPLE_WIDTH)
    for offset in range(0, len(pcm), frame_samples * SAMPLE_WIDTH):
        detector.push(pcm[offset : offset + frame_samples * SAMPLE_WIDTH])
    deadline = perf_counter() + _FEED_TIMEOUT_SEC
    while perf_counter() < deadline:
        if detector.stats.frames >= target:
            return
        sleep(0.01)
    raise TimeoutError(
        f"детектор обсчитал {detector.stats.frames}/{target} кадров за {_FEED_TIMEOUT_SEC} с"
    )


def only(segments: tuple[SpeechSegment, ...]) -> SpeechSegment:
    """Убедиться, что сегмент ровно один, и вернуть его."""
    assert len(segments) == 1, [
        (segment.frame_index, segment.duration_ms, segment.reason.value) for segment in segments
    ]
    return segments[0]


class FormantEngine(WakeWordEngine):
    """Разбирает форму волны фикстур, для проверки критериев приёмки.

    Грубый, но настоящий: полосовой фильтр вокруг первой форманты «а» и второй
    форманты «и», классификация кадра по отношению энергий и сверка полученной
    цепочки гласных с фразой. Этого хватает, чтобы отличить ``айрис`` от ``ирис``
    на реальных ``wake_*.wav`` — ровно та дискриминация, о которой задача.

    Копия двойника из ``tests/unit/test_wake_word.py``: перекрёстные импорты между
    тестовыми модулями запрещены, поэтому движок живёт здесь своей копией.
    """

    name: ClassVar[str] = "formant"
    package: ClassVar[str] = ""
    module: ClassVar[str] = ""

    #: Максимум, который выдаёт движок. Ниже MAX_THRESHOLD, чтобы sensitivity 0.0
    #: никогда не срабатывала — вариант с такой настройкой «выключен».
    MAX_SCORE: ClassVar[float] = 0.9

    __slots__ = ("_history", "_silent_run")

    #: Центры полос и полуширины для достоверности. Замерено на фикстурах:
    #: «а» около 0.004, «и» около 0.013, фрикатив выше 0.11.
    _A_CENTRE: ClassVar[float] = 0.004
    _A_HALF: ClassVar[float] = 0.007
    _I_CENTRE: ClassVar[float] = 0.013
    _I_HALF: ClassVar[float] = 0.010

    #: Порог RMS: кадры тише этого — тишина, они чистят историю.
    _RMS_FLOOR: ClassVar[float] = 0.02

    #: Пороги отношения, разделяющие три класса.
    _A_UPPER: ClassVar[float] = 0.010
    _I_UPPER: ClassVar[float] = 0.060

    def __init__(self) -> None:
        super().__init__()
        self._history: list[str] = []
        self._silent_run = 0

    @property
    def frame_samples(self) -> int:
        return 1280  # 80 мс — размер openWakeWord

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
            # Фрикатив или шум — не продлевает и не чистит историю.
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
        """Свернуть фразу к цепочке гласных классов, которые её сматчат.

        Русские гласные → форманта-группа: а/о/у/ы/э → "a", и/е/ю/я/ё → "i".
        Согласные игнорируются, повторы схлопываются: "ирис" → ["i"].
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
        """Сумма квадратов после резонатора, настроенного на ``centre_hz``."""
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
        """Насколько ``ratio`` близко к ``centre``, как оценка 0.0–1.0."""
        margin = abs(ratio - centre) / half
        return max(0.3, min(1.0, 1.0 - margin))


def _detector(
    engine: WakeWordEngine,
    *,
    phrases: tuple[WakePhrase, ...],
    debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    seen: list[WakeDetection] | None = None,
) -> WakeWordDetector:
    """Запущенный детектор вокруг ``engine``; активации пишет в ``seen``."""
    events = seen if seen is not None else []
    detector = WakeWordDetector(
        WakeWordSettings(phrases=phrases, debounce_ms=debounce_ms, queue_blocks=4096),
        WakeWordCallbacks(on_detected=events.append),
        engine=engine,
    )
    detector.start()
    return detector


def wake_decision(fixture: str, phrase: str = AYRIS) -> list[WakeDetection]:
    """Прогнать реальный движок по WAV и вернуть его активации.

    Демон-поток детектора всегда останавливается в ``finally``: инъецированный
    движок при этом не выгружается (принадлежит вызывающему).
    """
    seen: list[WakeDetection] = []
    engine = FormantEngine()
    detector = _detector(engine, phrases=(WakePhrase(phrase, 0.5),), seen=seen)
    try:
        feed(detector, wake_pcm(fixture), frame_samples=engine.frame_samples)
    finally:
        detector.stop()
    return seen


class StubSttEngine:
    """Локальный движок распознавания одной заранее заданной фразы.

    Не подкласс ``SttEngine``: роутеру нужны только ``load``/``transcribe``, а
    достоверность выставляем явно ≥ 0.4 — иначе пайплайн отбросит фразу как
    «не расслышала» на гейте достоверности (``voice.stt.min_confidence`` = 0.4).
    """

    def __init__(self, *, text: str, confidence: float = 0.9, name: str = "stub") -> None:
        self.text = text
        self.confidence = confidence
        self.name = name
        self.loads = 0
        self.calls = 0
        self.heard: list[AudioBuffer] = []

    def load(self, model_path: Any, options: Any) -> None:
        self.loads += 1

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        self.calls += 1
        self.heard.append(audio)
        return TranscriptResult(
            text=self.text,
            confidence=self.confidence,
            engine=self.name,
            device="cpu",
            duration_ms=audio.duration_ms,
        )

    def unload(self) -> None:
        pass


def stt_provider(engine: StubSttEngine) -> Callable[[], StubSttEngine]:
    """Провайдер, который загружает движок, как это делают настоящие."""

    def provide() -> StubSttEngine:
        engine.load(None, None)
        return engine

    return provide


class Handle:
    """Фраза «в динамиках»: помнит, дождались её или сняли."""

    def __init__(self) -> None:
        self.cancelled = False
        self.waited = False
        self._done = False

    @property
    def done(self) -> bool:
        return self._done

    def cancel(self) -> bool:
        self.cancelled = True
        self._done = True
        return True

    def wait(self, timeout: float | None = None) -> bool:
        self.waited = True
        self._done = True
        return True


class FakeTts:
    """Озвучка, которая ничего не произносит, но помнит каждую фразу."""

    def __init__(self, *, error: str = "") -> None:
        self.said: list[str] = []
        self.handles: list[Handle] = []
        self.error = error

    def say(self, text: str) -> Handle:
        self.said.append(text)
        if self.error:
            raise TtsError(self.error)
        handle = Handle()
        self.handles.append(handle)
        return handle

    @property
    def last(self) -> str:
        return self.said[-1] if self.said else ""


class FakeActions:
    """Исполнитель одной команды: помнит запрос, отдаёт заданный итог."""

    def __init__(self, *, outcome: ActionOutcome | None = None, error: str = "") -> None:
        self.outcome = outcome if outcome is not None else ActionOutcome(speak="Готово.")
        self.error = error
        self.seen: list[ActionRequest] = []

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        self.seen.append(request)
        if self.error:
            raise ActionError(self.error)
        return self.outcome

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def last(self) -> ActionRequest:
        return self.seen[-1]


class FakeLlm(LlmClient):
    """Настроенная модель, которая всегда отвечает одним и тем же.

    ``configured=True`` обязателен: иначе в HYBRID/AI пайплайн произнёс бы
    «модель не настроена» вместо ответа.
    """

    name = "fake"

    def __init__(self, text: str = "Сегодня вторник.") -> None:
        self.text = text
        self.prompts: list[tuple[LlmMessage, ...]] = []

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        self.prompts.append(tuple(messages))
        return LlmResponse(text=self.text, engine=self.name, finish_reason=FinishReason.STOP)

    @property
    def asked(self) -> str:
        """Что модель услышала последней репликой пользователя."""
        return self.prompts[-1][-1].content if self.prompts else ""


def library() -> Matcher:
    """Библиотека из двух команд: одна со слотом, одна без."""
    return Matcher.from_triggers(
        [
            Trigger(
                id=1,
                command_id=7,
                pattern="громкость {value:volume}",
                kind=TriggerKind.TEMPLATE,
            ),
            Trigger(id=2, command_id=8, pattern="открой браузер"),
        ]
    )


def settings_with(**sections: object) -> Settings:
    """Настройки с изменённой секцией. Модели заморожены, копия — только так."""
    return Settings.model_validate(sections)


def commands_only() -> Settings:
    """«Только команды»: все три тумблера «ИИ» выключены."""
    return settings_with(ai={"fallback_to_llm": False})


def hybrid() -> Settings:
    """«Гибрид»: сначала библиотека, при промахе — модель."""
    return settings_with(ai={"fallback_to_llm": True})


def phrase_segment(fixture: str = "phrase.wav") -> tuple[SpeechSegment, AudioBuffer]:
    """Реальный сегментатор нарезает ``fixture`` в один сегмент → буфер для STT.

    Возвращает и сам сегмент (чтобы утверждать, что он принят по тишине), и
    :class:`AudioBuffer` из его PCM — ровно то, что в бою отдаёт ``phrase_source``
    к концу фразы.
    """
    seg = only(segment_pcm(wake_pcm(fixture)))
    return seg, AudioBuffer(pcm=seg.pcm, sample_rate=seg.sample_rate)


@dataclass(slots=True)
class Rig:
    """Пайплайн на реальном роутере STG и матчере; края (действие/озвучка) — фейки."""

    bus: EventBus
    pipeline: Pipeline
    stt_engine: StubSttEngine
    router: SttRouter
    tts: FakeTts
    actions: FakeActions
    llm: FakeLlm | None
    buffer: AudioBuffer
    states: list[PipelineStateChanged] = field(default_factory=list)

    def stages(self) -> list[str]:
        """Стадии, о которых пайплайн сообщил на шину, по порядку."""
        return [event.state.value for event in self.states]

    def close(self) -> None:
        """Снять подписки пайплайна и закрыть роутер (его подписку на шину)."""
        self.pipeline.detach()
        self.router.close()


def build_rig(
    *,
    settings: Settings,
    transcript: str,
    buffer: AudioBuffer,
    confidence: float = 0.9,
    llm: FakeLlm | None = None,
) -> Rig:
    """Собрать пайплайн: реальный OFFLINE-роутер со стабом, реальный матчер.

    ``phrase_source`` отдаёт ``buffer`` — настоящий PCM, нарезанный сегментатором;
    его получает роутер и передаёт стабу, который возвращает ``transcript``.
    """
    bus = EventBus()
    states: list[PipelineStateChanged] = []
    bus.subscribe(PipelineStateChanged, states.append, weak=False)

    stt_engine = StubSttEngine(text=transcript, confidence=confidence)
    router = SttRouter(mode=SttMode.OFFLINE, offline=stt_provider(stt_engine))
    tts = FakeTts()
    actions = FakeActions()

    pipeline = Pipeline(
        bus,
        matcher=library(),
        stt=router,
        tts=tts,
        actions=actions,
        llm=llm,
        phrase_source=lambda: buffer,
        settings=settings,
        scheduler=ManualScheduler(),
        runner=inline_runner,
    )
    pipeline.attach()
    return Rig(bus, pipeline, stt_engine, router, tts, actions, llm, buffer, states)


#: Полный проход одной команды: активация → запись → распознавание → разбор →
#: исполнение → озвучка → покой. Тот же порядок утверждает задача 18 на синтетике.
FULL_PASS_STAGES = [
    "listening",
    "recording",
    "transcribing",
    "understanding",
    "executing",
    "responding",
    "idle",
]


def drive(rig: Rig, detection: WakeDetection) -> None:
    """Провести живой проход: активация из детекта + конец фразы по тишине.

    Оба события — настоящие ``WakeWordDetected``/``SpeechEnded`` на шине; под
    инлайновым раннером второй публикует прогоняет весь проход синхронно.
    """
    rig.bus.publish(WakeWordDetected(phrase=detection.phrase, confidence=detection.score))
    rig.bus.publish(SpeechEnded(duration_ms=round(rig.buffer.duration_ms), reason="silence"))


def test_full_voice_pass_wake_to_action() -> None:
    """Живая цепочка целиком: WAV → wake → сегмент → STT → матчер → действие."""
    seen = wake_decision("wake_ayris.wav")
    assert len(seen) == 1, seen
    assert seen[0].phrase == AYRIS

    seg, buffer = phrase_segment()
    assert seg.reason is EndReason.SILENCE

    rig = build_rig(settings=commands_only(), transcript=COMMAND_TEXT, buffer=buffer)
    try:
        drive(rig, seen[0])
    finally:
        rig.close()

    assert rig.stages() == FULL_PASS_STAGES
    assert rig.stt_engine.calls == 1
    assert rig.stt_engine.heard[0].pcm == buffer.pcm
    assert rig.actions.calls == 1
    assert rig.actions.last.command_id == 8
    assert rig.tts.said == ["Готово."]


def test_absent_fixture_never_activates() -> None:
    """Негатив движка: ``wake_absent.wav`` не даёт ни одной активации."""
    assert wake_decision("wake_absent.wav") == []


def test_hybrid_miss_falls_back_to_llm() -> None:
    """Промах матчера в HYBRID уходит в чат-модель, а не в «не нашла»."""
    seen = wake_decision("wake_ayris.wav")
    assert len(seen) == 1

    _seg, buffer = phrase_segment()
    llm = FakeLlm("Сегодня вторник.")
    rig = build_rig(settings=hybrid(), transcript=UNKNOWN_TEXT, buffer=buffer, llm=llm)
    try:
        drive(rig, seen[0])
    finally:
        rig.close()

    assert rig.actions.calls == 0
    assert llm.asked == UNKNOWN_TEXT
    assert rig.tts.last == llm.text


def test_commands_miss_says_not_matched() -> None:
    """Промах матчера без модели (COMMANDS) отвечает «не нашла такую команду»."""
    seen = wake_decision("wake_ayris.wav")
    assert len(seen) == 1

    _seg, buffer = phrase_segment()
    rig = build_rig(settings=commands_only(), transcript=UNKNOWN_TEXT, buffer=buffer)
    try:
        drive(rig, seen[0])
    finally:
        rig.close()

    assert rig.actions.calls == 0
    assert rig.tts.last == NOT_MATCHED_MESSAGE
