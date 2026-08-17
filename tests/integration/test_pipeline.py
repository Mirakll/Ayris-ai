"""Задача 18: пайплайн-диспетчер целиком — на подставных STT, действии и озвучке.

Модуль `core/pipeline.py` сам не делает ничего: он решает, в каком порядке
работают чужие узлы, что происходит, когда один из них падает, и кто побеждает,
когда пользователь заговорил во время ответа. Проверять такое можно только
целиком, поэтому здесь собирается настоящий пайплайн на настоящей шине событий —
подставлены только четыре узла по краям (распознавание, действие, озвучка,
модель) и два «источника времени».

Четыре решения этого файла стоит знать заранее.

**Ни одного `sleep` и ни одной секунды настоящего ожидания.** Дедлайны ждёт
:class:`ManualScheduler` — тест выстреливает их сам; время трейса идёт по
подставленным часам с шагом 250 мс, поэтому в утверждениях стоят точные
миллисекунды, а не «больше нуля». Файл задачи предупреждает про это прямым
текстом: тест на таймаут с настоящим ожиданием первым начнёт мигать на
windows-раннере, где планировщик грубее.

**Проход идёт на потоке теста.** :func:`inline_runner` вместо потока-по-сессию:
к моменту возврата из `submit_audio` проход уже закончен, и тест утверждает
результат, а не ждёт его. Это ровно тот шов, который в бою даёт «не блокировать
UI-поток», а в тесте — детерминизм.

**Все семь стадий видны только на шине.** Пайплайн публикует
``PipelineStateChanged`` на каждом переходе, и именно последовательность этих
событий — а не финальное состояние — проверяет, что цикл прошёл по спецификации.
Поэтому почти каждый тест смотрит на :meth:`Rig.stages`.

**Отмена проверяется изнутри стадии.** Отменить сессию «между стадиями» легко и
неинтересно; настоящий случай — пользователь сказал «стоп», пока STT уже считает
или действие уже выполняется. Подставные узлы поэтому умеют дёрнуть отмену
прямо из своего вызова, и тест проверяет, что следующая граница стадий её
поймала и ничего лишнего не сказала и не выполнила.

Группы:

* :class:`TestVoicePass` — полный голосовой цикл и события одного прохода.
* :class:`TestTextEntry` — ``run_text`` как вход без звука.
* :class:`TestModes` — три режима разбора и заглушка модели.
* :class:`TestSegment` — что делать с плохим или отсутствующим фрагментом.
* :class:`TestCancel` — отмена на каждой стадии, без утечки сессии.
* :class:`TestBargeIn` — активация во время ответа, гейт эха, отказ на занятом.
* :class:`TestTimeouts` — дедлайн стадии и что после него.
* :class:`TestStageErrors` — ошибка каждой стадии: фраза вслух и жизнь дальше.
* :class:`TestTrace` — тайминги, строка раздела 15, история, DevTools.
* :class:`TestWiring` — подписки, настройки на ходу, состояние оверлея.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

import pytest

from ayris.audio.stt.base import AudioBuffer, TranscriptResult
from ayris.core.config import Settings
from ayris.core.errors import ActionError, SttError, TtsError
from ayris.core.events import (
    ActionFailed,
    ActionFinished,
    ActionStarted,
    CancelRequested,
    Event,
    EventBus,
    IntentMatched,
    ModeChanged,
    PipelineStateChanged,
    SpeechEnded,
    TranscriptReady,
    WakeWordDetected,
)
from ayris.core.models import ExecutionResult, HistoryEntry
from ayris.core.pipeline import (
    ACTION_FAILED_MESSAGE,
    CANCEL_REASON_BARGE_IN,
    NOT_HEARD_MESSAGE,
    NOT_MATCHED_MESSAGE,
    NOTHING_SAID_MESSAGE,
    SOURCE_PTT,
    SOURCE_TEXT,
    TIMEOUT_MESSAGE,
    ActionOutcome,
    ActionRequest,
    NluMode,
    Pipeline,
    PipelineResult,
    inline_runner,
    mode_from_config,
)
from ayris.core.pipeline_states import ManualScheduler, PipelineState, PipelineTimeouts
from ayris.core.pipeline_trace import Stage, TraceRecord
from ayris.core.state import AssistantState, StateMachine
from ayris.nlu.context import DialogContext
from ayris.nlu.followup import CANCEL_REASON
from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
    FinishReason,
    LlmClient,
    LlmMessage,
    LlmResponse,
    LlmTool,
)
from ayris.nlu.matcher import Matcher, MatchResult, Trigger, TriggerKind
from ayris.utils.logger import PIPELINE_LOGGER_NAME, ROOT_LOGGER_NAME

pytestmark = pytest.mark.integration

#: Фраза, на которой проверяется всё: она матчится шаблонным триггером и отдаёт
#: слот, то есть проходит и матчер, и разбор слотов, и действие.
PHRASE = "громкость 50"

#: Команда, которой в библиотеке нет ни в каком виде.
UNKNOWN_PHRASE = "расскажи что-нибудь про черепах"

#: Слово активации, каким его сообщает движок wake word.
WAKE_PHRASE = "айрис"

#: …и то, что тот же движок присылает вместо фразы для горячей клавиши.
PTT_PHRASE = "<ptt>"

#: Длина «записанной» фразы. Она же — ожидаемый тайминг стадии записи: пайплайн
#: берёт его из самого буфера, а не по часам.
SEGMENT_MS = 640

#: Один тик подставленных часов, в миллисекундах: сколько показывает любая
#: стадия, время которой считается стопвотчем.
TICK_MS = 250


def speech(ms: int = SEGMENT_MS) -> AudioBuffer:
    """Буфер такой длины, будто в него говорили: 16 кГц, моно, int16."""
    return AudioBuffer(pcm=b"\x01\x00" * (16 * ms))


class Clock:
    """Монотонные секунды, которые двигает тест, а не машина.

    Каждое обращение прибавляет шаг, поэтому любая замеренная стопвотчем стадия
    показывает ровно :data:`TICK_MS` — число, которое видно в утверждении.
    """

    def __init__(self, *, step: float = TICK_MS / 1000.0) -> None:
        self.now = 1000.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value


class Handle:
    """Фраза «в динамиках»: помнит, дождались её или сняли."""

    def __init__(self, *, on_wait: Callable[[], None] | None = None) -> None:
        self.cancelled = False
        self.waited = False
        self._done = False
        self._on_wait = on_wait

    @property
    def done(self) -> bool:
        return self._done

    def cancel(self) -> bool:
        self.cancelled = True
        self._done = True
        return True

    def wait(self, timeout: float | None = None) -> bool:
        self.waited = True
        if self._on_wait is not None:
            self._on_wait()
        self._done = True
        return True


class FakeTts:
    """Озвучка, которая ничего не произносит, но помнит каждую фразу.

    ``on_wait`` вызывается ровно там, где в бою идёт звук: это единственное
    место, куда можно вклиниться с barge-in, потому что только там пайплайн
    действительно ждёт.
    """

    def __init__(self, *, on_wait: Callable[[], None] | None = None, error: str = "") -> None:
        self.said: list[str] = []
        self.handles: list[Handle] = []
        self.on_wait = on_wait
        self.error = error

    def say(self, text: str) -> Handle:
        self.said.append(text)
        if self.error:
            raise TtsError(self.error)
        handle = Handle(on_wait=self.on_wait)
        self.handles.append(handle)
        return handle

    @property
    def last(self) -> str:
        """Последнее сказанное, или ``""`` — если Айрис промолчала."""
        return self.said[-1] if self.said else ""


class FakeStt:
    """Распознавание одной заранее известной фразы.

    ``before`` дёргается внутри :meth:`transcribe`, то есть внутри стадии
    распознавания: так проверяются отмена и таймаут «на середине», а не между
    стадиями.
    """

    def __init__(
        self,
        *,
        text: str = PHRASE,
        confidence: float = 0.9,
        error: str = "",
        before: Callable[[], None] | None = None,
    ) -> None:
        self.text = text
        self.confidence = confidence
        self.error = error
        self.before = before
        self.heard: list[AudioBuffer] = []

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        self.heard.append(audio)
        if self.before is not None:
            self.before()
        if self.error:
            raise SttError(self.error)
        return TranscriptResult(
            text=self.text,
            confidence=self.confidence,
            engine="mock",
            duration_ms=audio.duration_ms,
            inference_ms=7.0,
        )


class FakeActions:
    """Исполнитель одной команды: помнит запрос, отдаёт заданный итог."""

    def __init__(
        self,
        *,
        outcome: ActionOutcome | None = None,
        error: str = "",
        crash: str = "",
        before: Callable[[], None] | None = None,
    ) -> None:
        self.outcome = outcome if outcome is not None else ActionOutcome(speak="Готово.")
        self.error = error
        self.crash = crash
        self.before = before
        self.seen: list[ActionRequest] = []

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        self.seen.append(request)
        if self.before is not None:
            self.before()
        if self.error:
            raise ActionError(self.error)
        if self.crash:
            raise RuntimeError(self.crash)
        return self.outcome

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def last(self) -> ActionRequest:
        return self.seen[-1]


class FakeLlm(LlmClient):
    """Настроенная модель, которая всегда отвечает одним и тем же.

    Задача 63 приносит восемь настоящих провайдеров; пайплайн не должен от этого
    измениться, поэтому здесь подставлен именно :class:`LlmClient`, а не что-то
    похожее по форме.
    """

    name = "fake"

    def __init__(self, text: str = "Сегодня вторник.") -> None:
        self.text = text
        self.prompts: list[tuple[LlmMessage, ...]] = []
        self.cancels: list[Callable[[], bool] | None] = []

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
        self.cancels.append(cancel)
        return LlmResponse(text=self.text, engine=self.name, finish_reason=FinishReason.STOP)

    @property
    def asked(self) -> str:
        """Что модель услышала последней репликой пользователя."""
        return self.prompts[-1][-1].content if self.prompts else ""


class FakeHistory:
    """Таблица ``history`` в списке. ``error`` — сломанный приёмник."""

    def __init__(self, *, error: str = "") -> None:
        self.rows: list[HistoryEntry] = []
        self.error = error

    def add(self, entry: HistoryEntry) -> HistoryEntry:
        if self.error:
            raise RuntimeError(self.error)
        self.rows.append(entry)
        return entry

    @property
    def last(self) -> HistoryEntry:
        return self.rows[-1]


class FakePhrases:
    """Источник PCM: то, что «записал» аудио-воркер к концу фразы."""

    def __init__(self, buffer: AudioBuffer | None = None) -> None:
        self.buffer = buffer if buffer is not None else speech()
        self.calls = 0

    @classmethod
    def empty(cls) -> FakePhrases:
        """Источник, который к концу фразы так ничего и не отдал."""
        source = cls()
        source.buffer = None
        return source

    def __call__(self) -> AudioBuffer | None:
        self.calls += 1
        return self.buffer


class CancellingMatcher(Matcher):
    """Библиотека, которая отменяет сессию прямо во время разбора.

    Единственный способ попасть отменой внутрь стадии ``UNDERSTANDING``: разбор
    синхронный и целиком лежит внутри одного замера.
    """

    def __init__(self, matcher: Matcher, interrupt: Callable[[], None]) -> None:
        super().__init__(matcher.index, matcher.settings)
        self.interrupt = interrupt

    def match(self, text: object, **kwargs: object) -> MatchResult | None:
        self.interrupt()
        return super().match(text, **kwargs)  # type: ignore[arg-type]


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


def ai_only() -> Settings:
    """«Только ИИ»: библиотеку не спрашиваем вообще."""
    return settings_with(ai={"free_chat": True})


@dataclass(slots=True)
class Rig:
    """Собранный пайплайн со всеми подставными узлами и записанными событиями."""

    bus: EventBus
    pipeline: Pipeline
    stt: FakeStt
    tts: FakeTts
    actions: FakeActions
    phrases: FakePhrases
    history: FakeHistory
    scheduler: ManualScheduler
    clock: Clock
    states: list[PipelineStateChanged] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    cancels: list[CancelRequested] = field(default_factory=list)

    def stages(self) -> list[str]:
        """Стадии, о которых пайплайн сообщил на шину, по порядку."""
        return [event.state.value for event in self.states]

    def event_names(self) -> list[str]:
        """Имена всех событий прохода — «партитура» одного запроса."""
        return [event.name for event in self.events]

    def trace(self) -> TraceRecord:
        """Последний закрытый трейс."""
        return self.pipeline.traces()[-1]

    def wake(self, phrase: str = WAKE_PHRASE) -> None:
        """Сказать слово активации так, как это делает движок wake word."""
        self.bus.publish(WakeWordDetected(phrase=phrase, confidence=0.9))

    def spoke(self, *, duration_ms: int = SEGMENT_MS, reason: str = "silence") -> None:
        """Сообщить, что фраза закончилась, как это делает сегментатор."""
        self.bus.publish(SpeechEnded(duration_ms=duration_ms, reason=reason))

    def voice_command(self, phrase: str = WAKE_PHRASE) -> None:
        """Полный голосовой проход: слово активации, фраза, конец фразы."""
        self.wake(phrase)
        self.spoke()


def build(
    *,
    settings: Settings | None = None,
    stt: FakeStt | None = None,
    tts: FakeTts | None = None,
    actions: FakeActions | None = None,
    llm: LlmClient | None = None,
    history: FakeHistory | None = None,
    context: DialogContext | None = None,
    state: StateMachine | None = None,
    matcher: Matcher | None = None,
    phrases: FakePhrases | None = None,
    timeouts: PipelineTimeouts | None = None,
) -> Rig:
    """Пайплайн на подставных узлах, подставных часах и ручном планировщике.

    ``llm``, ``context``, ``state`` и ``settings`` пробрасываются как есть:
    ``None`` для них — это осмысленная конфигурация («модель не настроена»,
    «контекст не подключён»), и половина тестов проверяет именно её.
    """
    bus = EventBus()
    rig = Rig(
        bus=bus,
        pipeline=Pipeline(bus),  # переопределяется ниже, см. комментарий
        stt=stt if stt is not None else FakeStt(),
        tts=tts if tts is not None else FakeTts(),
        actions=actions if actions is not None else FakeActions(),
        phrases=phrases if phrases is not None else FakePhrases(),
        history=history if history is not None else FakeHistory(),
        scheduler=ManualScheduler(),
        clock=Clock(),
    )
    bus.subscribe(PipelineStateChanged, rig.states.append, weak=False)
    bus.subscribe(CancelRequested, rig.cancels.append, weak=False)
    bus.subscribe(Event, rig.events.append, weak=False)
    # Пайплайн создаётся после подписок, чтобы ни одно событие сборки не
    # проскочило мимо записи; поле в Rig перезаписывается, потому что dataclass
    # нужен раньше — на него ссылаются подставные узлы.
    rig.pipeline = Pipeline(
        bus,
        state=state,
        matcher=matcher if matcher is not None else library(),
        context=context,
        stt=rig.stt,
        tts=rig.tts,
        actions=rig.actions,
        llm=llm,
        phrase_source=rig.phrases,
        history=rig.history,
        settings=settings,
        timeouts=timeouts,
        scheduler=rig.scheduler,
        runner=inline_runner,
        clock=rig.clock,
    )
    return rig


def pipeline_of(rig: Rig, **parts: object) -> Pipeline:
    """Второй пайплайн на той же шине, собранный ровно из перечисленных узлов.

    Нужен там, где проверяется отсутствие узла: «нет движка распознавания», «нет
    исполнителя действий». Через :func:`build` такое выражается только
    сентинелом, а через отдельный конструктор — списком того, что есть.
    """
    return Pipeline(
        rig.bus,
        runner=inline_runner,
        scheduler=rig.scheduler,
        clock=rig.clock,
        **parts,  # type: ignore[arg-type]
    )


class Collect(logging.Handler):
    """Обработчик-список: на время теста единственный, поэтому без дублей."""

    def __init__(self, sink: list[logging.LogRecord]) -> None:
        super().__init__(level=logging.DEBUG)
        self.sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self.sink.append(record)


def logs_of(name: str) -> Callable[[], Iterator[list[logging.LogRecord]]]:
    """Фикстура, ловящая записи одного логгера Ayris.

    ``caplog`` слушает корень интерпретатора, а ``setup_logging`` ставит
    ``propagate = False`` и на ``ayris``, и на ``ayris.pipeline`` — то есть
    утверждение о логе зависело бы от того, поднимал ли кто-то логирование
    раньше в прогоне. Свой обработчик прямо на нужном логгере, с выключенным на
    время теста подъёмом наверх, не зависит ни от того, ни от другого.
    """

    def fixture() -> Iterator[list[logging.LogRecord]]:
        logger = logging.getLogger(name)
        records: list[logging.LogRecord] = []
        handler = Collect(records)
        level, propagate = logger.level, logger.propagate
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.addHandler(handler)
        try:
            yield records
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
            logger.propagate = propagate

    return fixture


pipeline_log = pytest.fixture(logs_of(PIPELINE_LOGGER_NAME))
ayris_log = pytest.fixture(logs_of(ROOT_LOGGER_NAME))


def said_in(records: list[logging.LogRecord]) -> list[str]:
    """Собранные сообщения, в порядке появления."""
    return [record.getMessage() for record in records]


class TestVoicePass:
    """Полный цикл: слово активации → запись → STT → NLU → действие → озвучка."""

    def test_state_sequence(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.stages() == [
            PipelineState.LISTENING.value,
            PipelineState.RECORDING.value,
            PipelineState.TRANSCRIBING.value,
            PipelineState.UNDERSTANDING.value,
            PipelineState.EXECUTING.value,
            PipelineState.RESPONDING.value,
            PipelineState.IDLE.value,
        ]

    def test_pass_ends_idle_without_a_session(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.pipeline.session_id == ""
        assert not rig.pipeline.busy

    def test_command_runs_with_its_slots(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.actions.calls == 1
        request = rig.actions.last
        assert request.command_id == 7
        assert request.slots == {"value": 50}
        assert request.phrase == PHRASE
        # Подтверждение просят только у повтора и у ответа на вопрос.
        assert not request.confirmed

    def test_answer_is_spoken_and_waited_for(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.tts.said == ["Готово."]
        assert rig.tts.handles[0].waited

    def test_segment_reaches_recognition(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.phrases.calls == 1
        assert rig.stt.heard == [rig.phrases.buffer]

    def test_events_of_one_pass(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        names = rig.event_names()
        assert names.index(TranscriptReady.__name__) < names.index(IntentMatched.__name__)
        assert names.index(IntentMatched.__name__) < names.index(ActionStarted.__name__)
        assert names.index(ActionStarted.__name__) < names.index(ActionFinished.__name__)
        assert ActionFailed.__name__ not in names

    def test_every_event_carries_the_session(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        session_id = rig.trace().session_id
        tagged = [
            event
            for event in rig.events
            if isinstance(event, TranscriptReady | IntentMatched | ActionStarted | ActionFinished)
        ]
        assert tagged
        assert {event.request_id for event in tagged} == {session_id}

    def test_hotkey_activation_is_its_own_source(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command(PTT_PHRASE)

        assert rig.trace().source == SOURCE_PTT

    def test_two_passes_in_a_row(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()
        rig.voice_command()

        assert rig.actions.calls == 2
        assert len(rig.pipeline.traces()) == 2
        assert rig.pipeline.state is PipelineState.IDLE

    def test_submitted_audio_can_skip_the_bus(self) -> None:
        """Тесты и DevTools отдают фрагмент напрямую, минуя сегментатор."""
        rig = build()

        session_id = rig.pipeline.activate(source=SOURCE_PTT)
        assert rig.pipeline.submit_audio(speech()) == session_id
        assert rig.tts.said == ["Готово."]


class TestTextEntry:
    """``run_text`` — тот же пайплайн, начиная со стадии разбора."""

    def test_command_runs_without_audio(self) -> None:
        rig = build()

        result = rig.pipeline.run_text(PHRASE)

        assert result.ok
        assert result.command_id == 7
        assert result.spoken == "Готово."
        assert rig.stt.heard == []
        assert rig.phrases.calls == 0

    def test_states_start_at_understanding(self) -> None:
        rig = build()

        rig.pipeline.run_text(PHRASE)

        assert rig.stages() == [
            PipelineState.UNDERSTANDING.value,
            PipelineState.EXECUTING.value,
            PipelineState.RESPONDING.value,
            PipelineState.IDLE.value,
        ]

    def test_result_carries_a_frozen_trace(self) -> None:
        rig = build()

        result = rig.pipeline.run_text(PHRASE)

        assert result.trace is not None
        assert result.trace.source == SOURCE_TEXT
        assert result.trace.stt_raw == PHRASE
        assert result.trace.ok

    def test_empty_text_is_refused(self) -> None:
        rig = build()

        result = rig.pipeline.run_text("   ")

        assert result.outcome is ExecutionResult.ERROR
        assert result.error == "empty text"
        assert result.session_id == ""
        assert result.trace is None
        assert rig.tts.said == []
        assert rig.pipeline.traces() == ()

    def test_typed_command_outranks_a_voice_session(self) -> None:
        """Набранное в окне не может быть эхом, поэтому побеждает всегда."""
        rig = build()

        voice = rig.pipeline.activate(source=SOURCE_PTT)
        result = rig.pipeline.run_text(PHRASE)

        assert result.ok
        assert result.session_id != voice
        assert [record.outcome for record in rig.pipeline.traces()] == [
            ExecutionResult.CANCELLED,
            ExecutionResult.OK,
        ]
        assert rig.pipeline.state is PipelineState.IDLE


class TestModes:
    """Три режима разбора из раздела 5.1 и заглушка модели."""

    @pytest.mark.parametrize(
        ("toggles", "expected"),
        [
            ({"fallback_to_llm": False}, NluMode.COMMANDS),
            ({"fallback_to_llm": True}, NluMode.HYBRID),
            ({"llm_understanding": True, "fallback_to_llm": False}, NluMode.HYBRID),
            ({"free_chat": True}, NluMode.AI),
            ({"free_chat": True, "fallback_to_llm": False}, NluMode.AI),
        ],
    )
    def test_mode_from_config(self, toggles: dict[str, bool], expected: NluMode) -> None:
        assert mode_from_config(settings_with(ai=toggles)) is expected

    def test_commands_only_says_the_command_was_not_found(self) -> None:
        rig = build(settings=commands_only(), llm=FakeLlm())

        result = rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert rig.pipeline.mode is NluMode.COMMANDS
        assert result.outcome is ExecutionResult.UNMATCHED
        assert not result.matched
        assert rig.tts.said == [NOT_MATCHED_MESSAGE]
        assert rig.stages() == [PipelineState.UNDERSTANDING.value, PipelineState.IDLE.value]

    def test_commands_only_still_runs_a_known_command(self) -> None:
        rig = build(settings=commands_only())

        assert rig.pipeline.run_text(PHRASE).ok
        assert rig.actions.calls == 1

    def test_hybrid_matches_first(self) -> None:
        llm = FakeLlm()
        rig = build(settings=hybrid(), llm=llm)

        result = rig.pipeline.run_text(PHRASE)

        assert result.command_id == 7
        assert llm.prompts == []

    def test_hybrid_falls_back_to_the_model(self) -> None:
        llm = FakeLlm()
        rig = build(settings=hybrid(), llm=llm)

        result = rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert rig.pipeline.mode is NluMode.HYBRID
        assert result.ok
        assert result.spoken == "Сегодня вторник."
        assert llm.asked == UNKNOWN_PHRASE
        assert rig.actions.calls == 0
        assert rig.trace().duration_of(Stage.LLM) == TICK_MS

    def test_ai_only_ignores_the_library(self) -> None:
        llm = FakeLlm()
        rig = build(settings=ai_only(), llm=llm)

        result = rig.pipeline.run_text(PHRASE)

        assert rig.pipeline.mode is NluMode.AI
        assert result.command_id is None
        assert llm.asked == PHRASE
        assert rig.actions.calls == 0

    @pytest.mark.parametrize("settings", [hybrid(), ai_only()], ids=["hybrid", "ai"])
    def test_a_model_that_is_not_configured_says_so(self, settings: Settings) -> None:
        rig = build(settings=settings)

        result = rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert rig.tts.said == [NOT_CONFIGURED_MESSAGE]
        assert result.outcome is ExecutionResult.ERROR
        assert result.error == "llm not configured"
        assert rig.pipeline.state is PipelineState.IDLE

    def test_an_empty_answer_is_a_miss(self) -> None:
        rig = build(settings=ai_only(), llm=FakeLlm(text="   "))

        result = rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert result.outcome is ExecutionResult.UNMATCHED
        assert rig.tts.said == [NOT_MATCHED_MESSAGE]

    def test_the_model_gets_a_cancel_predicate(self) -> None:
        llm = FakeLlm()
        rig = build(settings=ai_only(), llm=llm)

        rig.pipeline.run_text(UNKNOWN_PHRASE)

        cancel = llm.cancels[-1]
        assert cancel is not None
        assert not cancel()

    def test_mode_changes_without_a_restart(self) -> None:
        rig = build(settings=commands_only(), llm=FakeLlm())
        assert rig.pipeline.mode is NluMode.COMMANDS

        rig.pipeline.apply_settings(ai_only())

        assert rig.pipeline.mode is NluMode.AI
        assert rig.pipeline.run_text(UNKNOWN_PHRASE).ok

    def test_mode_is_written_to_the_trace(self) -> None:
        rig = build(settings=ai_only(), llm=FakeLlm())

        rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert rig.trace().mode == NluMode.AI.value


class TestSegment:
    """Фрагмент, которого нет, или который не стоило распознавать."""

    def test_rejected_segment_ends_the_session(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.wake()
        rig.spoke(duration_ms=90, reason="too_short")

        assert rig.tts.said == [NOT_HEARD_MESSAGE]
        assert rig.stt.heard == []
        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.trace().error == "too_short"

    def test_missing_buffer_ends_the_session(self) -> None:
        rig = build(phrases=FakePhrases.empty())
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.tts.said == [NOT_HEARD_MESSAGE]
        assert rig.trace().error == "empty segment"

    def test_audio_without_a_session_is_dropped(self) -> None:
        rig = build()

        assert rig.pipeline.submit_audio(speech()) == ""
        assert rig.stt.heard == []
        assert rig.pipeline.traces() == ()

    def test_empty_transcript_says_it_was_not_heard(self) -> None:
        rig = build(stt=FakeStt(text="  "))

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.tts.said == [NOT_HEARD_MESSAGE]
        assert rig.actions.calls == 0
        assert rig.trace().outcome is ExecutionResult.ERROR

    def test_a_phrase_below_the_confidence_floor_is_dropped(self) -> None:
        rig = build(
            settings=settings_with(voice={"stt": {"min_confidence": 0.8}}),
            stt=FakeStt(confidence=0.4),
        )

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.tts.said == [NOT_HEARD_MESSAGE]
        assert rig.trace().error == "confidence below threshold"

    def test_confidence_is_kept_when_no_floor_is_configured(self) -> None:
        rig = build(stt=FakeStt(confidence=0.05))

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.actions.calls == 1


class TestCancel:
    """«Отмена» на любой стадии: назад в ожидание, без остатков сессии."""

    def test_cancel_while_listening(self) -> None:
        rig = build()

        rig.pipeline.activate(source=SOURCE_PTT)
        assert rig.pipeline.cancel(reason="кнопка")

        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.pipeline.session_id == ""
        assert rig.tts.said == []
        assert rig.trace().outcome is ExecutionResult.CANCELLED
        assert rig.trace().error == "кнопка"

    def test_cancel_during_recognition(self) -> None:
        rig = build()
        rig.stt.before = lambda: rig.pipeline.cancel(reason="кнопка")

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.stages()[-1] == PipelineState.IDLE.value
        assert rig.actions.calls == 0
        assert rig.tts.said == []
        assert rig.trace().outcome is ExecutionResult.CANCELLED

    def test_cancel_during_understanding(self) -> None:
        rig = build()
        rig.pipeline.set_matcher(
            CancellingMatcher(library(), lambda: rig.pipeline.cancel(reason="кнопка"))
        )

        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.CANCELLED
        assert PipelineState.EXECUTING.value not in rig.stages()
        assert rig.actions.calls == 0

    def test_cancel_during_the_action(self) -> None:
        rig = build(actions=FakeActions())
        rig.actions.before = lambda: rig.pipeline.cancel(reason="кнопка")

        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.CANCELLED
        assert rig.tts.said == []
        assert rig.stages()[-1] == PipelineState.IDLE.value

    def test_cancel_during_speech_stops_the_sound(self) -> None:
        rig = build(tts=FakeTts())
        rig.tts.on_wait = lambda: rig.pipeline.cancel(reason="кнопка")

        result = rig.pipeline.run_text(PHRASE)

        assert rig.tts.handles[0].cancelled
        assert result.outcome is ExecutionResult.CANCELLED

    def test_cancel_arrives_as_an_event(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.bus.publish(CancelRequested(reason="кнопка"))

        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.trace().outcome is ExecutionResult.CANCELLED

    def test_the_pipelines_own_cancel_does_not_come_back(self) -> None:
        """Иначе barge-in закрыл бы сессию, которую сам только что открыл."""
        rig = build()
        rig.pipeline.attach()

        session_id = rig.pipeline.activate(source=SOURCE_PTT)
        rig.bus.publish(CancelRequested(reason=CANCEL_REASON_BARGE_IN))

        assert rig.pipeline.session_id == session_id
        assert rig.pipeline.state is PipelineState.LISTENING

    def test_cancel_with_nothing_in_flight(self) -> None:
        rig = build()

        assert not rig.pipeline.cancel(reason="кнопка")
        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.pipeline.traces() == ()

    def test_a_cancel_phrase_cancels(self) -> None:
        rig = build(context=DialogContext(window_probe=lambda: None, autosave=False))

        result = rig.pipeline.run_text("отмена")

        assert result.outcome is ExecutionResult.CANCELLED
        assert [event.reason for event in rig.cancels] == [CANCEL_REASON]
        assert rig.actions.calls == 0
        assert rig.pipeline.state is PipelineState.IDLE

    def test_no_session_leaks_after_a_cancel(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.cancel(reason="кнопка")
        rig.voice_command()

        assert rig.actions.calls == 1
        assert rig.pipeline.state is PipelineState.IDLE
        assert [record.outcome for record in rig.pipeline.traces()] == [
            ExecutionResult.CANCELLED,
            ExecutionResult.OK,
        ]


class TestBargeIn:
    """Ровно одна живая сессия: новая активация либо перебивает, либо отказ."""

    def test_activation_during_speech_starts_a_new_session(self) -> None:
        rig = build()
        rig.pipeline.attach()
        first = ""
        rig.tts.on_wait = lambda: rig.wake(PTT_PHRASE)

        result = rig.pipeline.run_text(PHRASE)
        first = result.session_id

        assert result.outcome is ExecutionResult.CANCELLED
        assert rig.pipeline.state is PipelineState.LISTENING
        assert rig.pipeline.session_id not in {"", first}
        assert rig.tts.handles[0].cancelled
        assert rig.trace().error == CANCEL_REASON_BARGE_IN

    def test_barge_in_does_not_wait_for_the_answer_to_end(self) -> None:
        rig = build()
        rig.pipeline.attach()
        rig.tts.on_wait = lambda: rig.wake(PTT_PHRASE)

        rig.pipeline.run_text(PHRASE)

        assert rig.stages()[-2:] == [PipelineState.IDLE.value, PipelineState.LISTENING.value]
        assert [event.reason for event in rig.cancels] == [CANCEL_REASON_BARGE_IN]

    def test_the_echo_guard_drops_a_microphone_activation(self) -> None:
        """С выключенным «прерывать озвучку» микрофон во время ответа — это эхо."""
        rig = build(settings=settings_with(voice={"tts": {"interrupt_on_speech": False}}))
        rig.pipeline.attach()
        refused: list[str] = []
        rig.tts.on_wait = lambda: refused.append(rig.pipeline.activate(source=WAKE_PHRASE))

        result = rig.pipeline.run_text(PHRASE)

        assert refused == [""]
        assert result.ok
        assert not rig.tts.handles[0].cancelled

    def test_the_echo_guard_lets_a_hotkey_through(self) -> None:
        rig = build(settings=settings_with(voice={"tts": {"interrupt_on_speech": False}}))
        rig.pipeline.attach()
        started: list[str] = []
        rig.tts.on_wait = lambda: started.append(rig.pipeline.activate(source=SOURCE_PTT))

        rig.pipeline.run_text(PHRASE)

        assert started[0] != ""
        assert rig.pipeline.state is PipelineState.LISTENING

    def test_a_second_activation_while_listening_is_refused(self) -> None:
        rig = build()

        first = rig.pipeline.activate(source=SOURCE_PTT)

        assert rig.pipeline.activate(source=WAKE_PHRASE) == ""
        assert rig.pipeline.session_id == first
        assert rig.stages() == [PipelineState.LISTENING.value]

    def test_an_activation_during_recognition_is_refused(self) -> None:
        rig = build()
        refused: list[str] = []
        rig.stt.before = lambda: refused.append(rig.pipeline.activate(source=WAKE_PHRASE))

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert refused == [""]
        assert rig.actions.calls == 1

    def test_an_activation_during_the_action_is_refused(self) -> None:
        rig = build()
        refused: list[str] = []
        rig.actions.before = lambda: refused.append(rig.pipeline.activate(source=SOURCE_PTT))

        assert rig.pipeline.run_text(PHRASE).ok
        assert refused == [""]


class TestTimeouts:
    """Дедлайн стадии: ручной планировщик, ни одной настоящей секунды."""

    def test_listening_deadline_comes_from_the_settings(self) -> None:
        rig = build(settings=settings_with(voice={"wake": {"listen_window_sec": 4.0}}))

        rig.pipeline.activate(source=SOURCE_PTT)

        assert rig.scheduler.pending == (4.0,)

    def test_a_hotkey_that_heard_nothing_says_so(self) -> None:
        rig = build()

        rig.pipeline.activate(source=SOURCE_PTT)
        assert rig.scheduler.fire_all() == 1

        assert rig.tts.said == [NOTHING_SAID_MESSAGE]
        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.trace().outcome is ExecutionResult.TIMEOUT
        assert rig.trace().error == "timeout in listening"

    def test_a_wake_word_that_heard_nothing_stays_quiet(self) -> None:
        """Слово активации срабатывает само; озвучивать каждый ложный — хуже."""
        rig = build()
        rig.pipeline.attach()

        rig.wake()
        rig.scheduler.fire_all()

        assert rig.tts.said == []
        assert rig.trace().outcome is ExecutionResult.TIMEOUT

    def test_a_deadline_in_the_middle_of_recognition(self) -> None:
        rig = build()
        rig.stt.before = rig.scheduler.fire_all

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.tts.said == [TIMEOUT_MESSAGE]
        assert rig.trace().error == "timeout in transcribing"
        assert rig.actions.calls == 0

    def test_the_next_activation_works_after_a_timeout(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.scheduler.fire_all()
        rig.voice_command()

        assert rig.actions.calls == 1
        assert rig.pipeline.state is PipelineState.IDLE

    def test_no_deadline_is_left_armed_after_a_pass(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        assert rig.scheduler.pending == ()

    def test_new_deadlines_apply_from_the_next_session(self) -> None:
        rig = build(settings=settings_with(voice={"wake": {"listen_window_sec": 4.0}}))

        rig.pipeline.apply_settings(settings_with(voice={"wake": {"listen_window_sec": 9.0}}))
        rig.pipeline.activate(source=SOURCE_PTT)

        assert rig.scheduler.pending == (9.0,)


class TestStageErrors:
    """Ошибка любой стадии — фраза вслух, и пайплайн жив для следующей."""

    def test_recognition_failure_speaks_its_message(
        self, ayris_log: list[logging.LogRecord]
    ) -> None:
        rig = build(stt=FakeStt(error="engine down"))

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.submit_audio(speech())

        assert rig.tts.said == [SttError("").user_message]
        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.trace().error == "engine down"
        # Пользователю — короткая фраза, в лог — техническая причина и трейсбек.
        assert any("engine down" in line for line in said_in(ayris_log))
        assert any(record.exc_info for record in ayris_log)

    def test_a_missing_engine_is_reported(self) -> None:
        rig = build()
        silent = pipeline_of(rig, matcher=library(), tts=rig.tts)

        silent.activate(source=SOURCE_PTT)
        silent.submit_audio(speech())

        assert rig.tts.said == [SttError("").user_message]
        assert silent.state is PipelineState.IDLE

    def test_action_failure_speaks_and_publishes(self) -> None:
        rig = build(actions=FakeActions(error="volume driver missing"))

        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.ERROR
        assert rig.tts.said == [ActionError("").user_message]
        failures = [event for event in rig.events if isinstance(event, ActionFailed)]
        assert [event.error for event in failures] == ["volume driver missing"]
        assert rig.pipeline.state is PipelineState.IDLE

    def test_a_missing_action_runner_is_reported(self) -> None:
        rig = build()
        toothless = pipeline_of(rig, matcher=library(), tts=rig.tts)

        result = toothless.run_text(PHRASE)

        assert result.outcome is ExecutionResult.ERROR
        assert rig.tts.said == [ACTION_FAILED_MESSAGE]

    def test_a_silent_failure_still_gets_a_word(self) -> None:
        """Тишина после команды читается как «сработало»."""
        rig = build(actions=FakeActions(outcome=ActionOutcome(result=ExecutionResult.ERROR)))

        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.ERROR
        assert rig.tts.said == [ACTION_FAILED_MESSAGE]

    def test_a_voice_that_will_not_speak_keeps_the_command(self) -> None:
        rig = build(tts=FakeTts(error="no output device"))

        result = rig.pipeline.run_text(PHRASE)

        assert result.ok
        assert result.spoken == "Готово."
        assert rig.actions.calls == 1

    def test_an_unexpected_exception_is_answered(self) -> None:
        rig = build(actions=FakeActions(crash="boom"))

        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.ERROR
        assert rig.tts.said == [ACTION_FAILED_MESSAGE]
        assert "boom" in rig.trace().error

    def test_one_failed_session_does_not_break_the_next(self) -> None:
        rig = build(actions=FakeActions(error="volume driver missing"))
        rig.pipeline.attach()

        rig.pipeline.run_text(PHRASE)
        rig.actions.error = ""
        rig.voice_command()

        assert rig.pipeline.state is PipelineState.IDLE
        assert [record.outcome for record in rig.pipeline.traces()] == [
            ExecutionResult.ERROR,
            ExecutionResult.OK,
        ]


class TestTrace:
    """Тайминги стадий, строка раздела 15, история и панель DevTools."""

    def test_every_stage_is_timed(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        record = rig.trace()
        # Запись измерена не часами, а самим буфером: воркер сообщает свою длину.
        assert record.duration_of(Stage.RECORD) == SEGMENT_MS
        assert record.duration_of(Stage.STT) == TICK_MS
        assert record.duration_of(Stage.NLU) == TICK_MS
        assert record.duration_of(Stage.ACTION) == TICK_MS
        assert record.duration_of(Stage.TTS) == TICK_MS
        assert [timing.stage for timing in record.stages] == [
            Stage.RECORD,
            Stage.STT,
            Stage.NLU,
            Stage.ACTION,
            Stage.TTS,
        ]

    def test_the_log_line_has_the_section_15_shape(
        self, pipeline_log: list[logging.LogRecord]
    ) -> None:
        rig = build()

        rig.pipeline.run_text(PHRASE)

        record = rig.trace()
        assert said_in(pipeline_log) == [
            f"[{record.session_id}] STT raw: {PHRASE} → NLU intent: command:7 "
            f"→ Action: command:7 → Result: ok — {record.total_ms} мс "
            f"[разбор {TICK_MS} мс | действие {TICK_MS} мс | озвучка {TICK_MS} мс]"
        ]

    def test_a_failed_stage_is_marked_in_the_line(self) -> None:
        rig = build(actions=FakeActions(error="volume driver missing"))

        rig.pipeline.run_text(PHRASE)

        failed = [timing for timing in rig.trace().stages if not timing.ok]
        assert [timing.stage for timing in failed] == [Stage.ACTION]

    def test_the_session_is_written_to_history(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        record = rig.trace()
        entry = rig.history.last
        assert entry.stt_raw == PHRASE
        assert entry.matched_command_id == 7
        assert entry.intent == "command:7"
        assert entry.result is ExecutionResult.OK
        assert entry.duration_ms == record.total_ms
        assert entry.params["value"] == 50
        assert entry.params["session_id"] == record.session_id
        assert entry.params["source"] == WAKE_PHRASE
        assert entry.params["answer"] == "Готово."
        assert entry.params["stages_ms"] == {
            "record": SEGMENT_MS,
            "stt": TICK_MS,
            "nlu": TICK_MS,
            "action": TICK_MS,
            "tts": TICK_MS,
        }

    def test_a_cancelled_session_is_written_too(self) -> None:
        rig = build()

        rig.pipeline.activate(source=SOURCE_PTT)
        rig.pipeline.cancel(reason="кнопка")

        assert rig.history.last.result is ExecutionResult.CANCELLED

    def test_history_can_be_switched_off(self) -> None:
        rig = build(settings=settings_with(privacy={"store_history": False}))

        assert rig.pipeline.run_text(PHRASE).ok
        assert rig.history.rows == []
        # Трейс всё равно есть: DevTools живёт не в базе.
        assert len(rig.pipeline.traces()) == 1

    def test_a_broken_history_sink_does_not_lose_the_answer(
        self, ayris_log: list[logging.LogRecord]
    ) -> None:
        rig = build(history=FakeHistory(error="database is locked"))

        result = rig.pipeline.run_text(PHRASE)

        assert result.ok
        assert rig.tts.said == ["Готово."]
        assert any("database is locked" in line for line in said_in(ayris_log))

    def test_devtools_gets_the_structured_form(self) -> None:
        rig = build()
        rig.pipeline.attach()

        rig.voice_command()

        payload = rig.trace().payload
        assert payload["stt_raw"] == PHRASE
        assert payload["stt_engine"] == "mock"
        assert payload["intent"] == "command:7"
        assert payload["slots"] == {"value": 50}
        assert payload["answer"] == "Готово."
        assert payload["outcome"] == ExecutionResult.OK.value
        assert [stage["stage"] for stage in payload["stages"]] == [  # type: ignore[union-attr]
            Stage.RECORD.value,
            Stage.STT.value,
            Stage.NLU.value,
            Stage.ACTION.value,
            Stage.TTS.value,
        ]

    def test_traces_are_returned_oldest_first(self) -> None:
        rig = build()

        results = [rig.pipeline.run_text(PHRASE) for _ in range(3)]

        assert [record.session_id for record in rig.pipeline.traces()] == [
            result.session_id for result in results
        ]
        assert len(rig.pipeline.traces(limit=2)) == 2
        assert rig.pipeline.traces(limit=2)[-1].session_id == results[-1].session_id


class TestWiring:
    """Подписки, настройки на ходу и вторая машина состояний — для оверлея."""

    def test_attach_is_idempotent(self) -> None:
        rig = build()

        rig.pipeline.attach()
        rig.pipeline.attach()

        assert rig.bus.subscriber_count(WakeWordDetected) == 1

    def test_detach_stops_the_activations(self) -> None:
        rig = build()
        rig.pipeline.attach()
        rig.pipeline.detach()

        rig.wake()

        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.pipeline.traces() == ()

    def test_close_cancels_and_unsubscribes(self) -> None:
        rig = build()
        rig.pipeline.attach()
        rig.pipeline.activate(source=SOURCE_PTT)

        rig.pipeline.close()

        assert rig.pipeline.state is PipelineState.IDLE
        assert rig.bus.subscriber_count(WakeWordDetected) == 0
        assert rig.scheduler.pending == ()
        assert rig.trace().outcome is ExecutionResult.CANCELLED

    def test_the_overlay_follows_the_stages(self) -> None:
        rig = build()
        modes: list[ModeChanged] = []
        rig.bus.subscribe(ModeChanged, modes.append, weak=False)
        overlay = StateMachine(rig.bus)
        pipeline = pipeline_of(
            rig,
            state=overlay,
            matcher=library(),
            stt=rig.stt,
            tts=rig.tts,
            actions=rig.actions,
        )

        pipeline.run_text(PHRASE)

        assert [event.state for event in modes] == [
            AssistantState.THINKING,
            AssistantState.SPEAKING,
            AssistantState.IDLE,
        ]
        assert overlay.state is AssistantState.IDLE

    def test_the_overlay_shows_a_failure(self) -> None:
        rig = build(actions=FakeActions(error="volume driver missing"))
        overlay = StateMachine(rig.bus)
        pipeline = pipeline_of(
            rig,
            state=overlay,
            matcher=library(),
            tts=rig.tts,
            actions=rig.actions,
        )

        pipeline.run_text(PHRASE)

        assert overlay.state is AssistantState.ERROR

    def test_the_library_can_be_swapped(self) -> None:
        rig = build(settings=commands_only())

        rig.pipeline.set_matcher(None)
        result = rig.pipeline.run_text(PHRASE)

        assert result.outcome is ExecutionResult.UNMATCHED
        assert rig.tts.said == [NOT_MATCHED_MESSAGE]

    def test_a_real_model_can_be_installed_later(self) -> None:
        rig = build(settings=ai_only())

        rig.pipeline.set_llm(FakeLlm(text="Готов."))
        first: PipelineResult = rig.pipeline.run_text(UNKNOWN_PHRASE)
        rig.pipeline.set_llm(None)
        second = rig.pipeline.run_text(UNKNOWN_PHRASE)

        assert first.spoken == "Готов."
        assert second.spoken == NOT_CONFIGURED_MESSAGE

    def test_repr_names_the_stage_and_the_mode(self) -> None:
        rig = build(settings=commands_only())

        assert repr(rig.pipeline) == "Pipeline(idle, mode=commands)"
