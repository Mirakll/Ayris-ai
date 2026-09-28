"""Бенчмарки раздела 23 на детерминированных часах — с машиночитаемым отчётом.

Задача 74 просит цифры раздела 23: время STT, задержку TTS до первого звука,
сквозной отклик пайплайна, переключение online/offline. Настоящие числа требуют
моделей, железа и сети — на общем раннере они шумят, а «падающий из-за шума
бенчмарк хуже отсутствующего». Поэтому здесь замеряется не машина, а **проводка**:
пайплайн собран на подставных узлах и на :class:`Clock`, который двигает тест, а
не процессор. Каждая стадия, обёрнутая стопвотчем, показывает ровно ``TICK_MS``;
запись — ровно длину буфера. Значения синтетические и воспроизводимые до
миллисекунды, поэтому порог по ним — это контракт-регрессия (стадия отвязалась,
таймер задвоился, трейс перестал заполняться), а не измерение задержки.

Отчёт пишется всегда, даже если тест красный, чтобы артефакт `nightly.yml`
существовал при любом исходе; путь берётся из ``AYRIS_BENCH_OUT`` (по умолчанию
``reports/benchmarks.json``). Форма отчёта — ``{"schema": 1, "deterministic":
true, "metrics": {name: {value_ms, threshold_ms, ok}}, "ok": <все ok>}`` — ровно
то, что задача 74 берёт готовым. Настоящие замеры (модели/железо/сеть) сюда не
попадают: их держат отдельные помеченные тесты, а этот блок остаётся
детерминированным.

Хелперы и двойники скопированы из ``tests/integration/test_pipeline.py``:
перекрёстные импорты между тестовыми модулями запрещены (нет ``__init__.py``,
линт режет относительные), так что каждая копия живёт в своём файле.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ayris.audio.stt.base import AudioBuffer, TranscriptResult
from ayris.core.config import Settings
from ayris.core.errors import ActionError, SttError, TtsError
from ayris.core.events import (
    CancelRequested,
    Event,
    EventBus,
    PipelineStateChanged,
    SpeechEnded,
    WakeWordDetected,
)
from ayris.core.models import HistoryEntry
from ayris.core.pipeline import (
    ActionOutcome,
    ActionRequest,
    Pipeline,
    inline_runner,
)
from ayris.core.pipeline_states import ManualScheduler
from ayris.core.pipeline_trace import Stage, TraceRecord
from ayris.nlu.llm.base import FinishReason, LlmClient, LlmMessage, LlmResponse, LlmTool
from ayris.nlu.matcher import Matcher, Trigger, TriggerKind

pytestmark = pytest.mark.integration

#: Фраза, которая матчится шаблонным триггером и отдаёт слот — «командный» путь.
PHRASE = "громкость 50"

#: Фраза, которой в библиотеке нет: гонит гибрид в модель.
UNKNOWN_PHRASE = "расскажи что-нибудь про черепах"

#: Слово активации, каким его сообщает движок wake word.
WAKE_PHRASE = "айрис"

#: Длина «записанной» фразы и ожидаемый тайминг стадии записи (из буфера).
SEGMENT_MS = 640

#: Один тик подставленных часов, мс: столько показывает любая стадия-стопвотч.
TICK_MS = 250


def speech(ms: int = SEGMENT_MS) -> AudioBuffer:
    """Буфер такой длины, будто в него говорили: 16 кГц, моно, int16."""
    return AudioBuffer(pcm=b"\x01\x00" * (16 * ms))


class Clock:
    """Монотонные секунды, которые двигает тест, а не машина."""

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
    """Озвучка, которая ничего не произносит, но помнит каждую фразу."""

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
        return self.said[-1] if self.said else ""


class FakeStt:
    """Распознавание одной заранее известной фразы."""

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
    """Настроенная модель, которая всегда отвечает одним и тем же."""

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
        return self.prompts[-1][-1].content if self.prompts else ""


class FakeHistory:
    """Таблица ``history`` в списке."""

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

    def __call__(self) -> AudioBuffer | None:
        self.calls += 1
        return self.buffer


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
    """Настройки с изменённой секцией."""
    return Settings.model_validate(sections)


def commands_only() -> Settings:
    """«Только команды»: тумблеры «ИИ» выключены — путь без модели."""
    return settings_with(ai={"fallback_to_llm": False})


def hybrid() -> Settings:
    """«Гибрид»: сначала библиотека, при промахе — модель."""
    return settings_with(ai={"fallback_to_llm": True})


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
    matcher: Matcher | None = None,
    phrases: FakePhrases | None = None,
) -> Rig:
    """Пайплайн на подставных узлах, подставных часах и ручном планировщике."""
    bus = EventBus()
    rig = Rig(
        bus=bus,
        pipeline=Pipeline(bus),  # переопределяется ниже
        stt=stt if stt is not None else FakeStt(),
        tts=tts if tts is not None else FakeTts(),
        actions=actions if actions is not None else FakeActions(),
        phrases=phrases if phrases is not None else FakePhrases(),
        history=FakeHistory(),
        scheduler=ManualScheduler(),
        clock=Clock(),
    )
    bus.subscribe(PipelineStateChanged, rig.states.append, weak=False)
    bus.subscribe(CancelRequested, rig.cancels.append, weak=False)
    bus.subscribe(Event, rig.events.append, weak=False)
    rig.pipeline = Pipeline(
        bus,
        matcher=matcher if matcher is not None else library(),
        stt=rig.stt,
        tts=rig.tts,
        actions=rig.actions,
        llm=llm,
        phrase_source=rig.phrases,
        history=rig.history,
        settings=settings,
        scheduler=rig.scheduler,
        runner=inline_runner,
        clock=rig.clock,
    )
    return rig


@dataclass(frozen=True, slots=True)
class Budget:
    """Одна измеренная величина против бюджета раздела 23.

    ``value_ms`` — детерминированный синтетический тайминг с подставных часов,
    ``threshold_ms`` — порог раздела 23. ``ok`` красит метрику: значение не должно
    перерасти бюджет. Синтетика лежит заведомо ниже — порог ловит регрессию
    проводки (стадия отвязалась, замер задвоился), а не медленное железо.
    """

    name: str
    value_ms: int
    threshold_ms: int
    section: str

    @property
    def ok(self) -> bool:
        return self.value_ms <= self.threshold_ms

    def as_json(self) -> dict[str, object]:
        return {
            "value_ms": self.value_ms,
            "threshold_ms": self.threshold_ms,
            "section": self.section,
            "ok": self.ok,
        }


def _commands_trace() -> TraceRecord:
    """Командный проход без модели: запись→STT→NLU→действие→озвучка."""
    rig = build(settings=commands_only())
    rig.pipeline.attach()
    rig.voice_command()
    return rig.trace()


def _hybrid_trace() -> TraceRecord:
    """Гибридный проход с промахом библиотеки: добавляет стадию модели."""
    rig = build(
        settings=hybrid(),
        stt=FakeStt(text=UNKNOWN_PHRASE),
        llm=FakeLlm(text="Черепахи живут долго."),
    )
    rig.pipeline.attach()
    rig.voice_command()
    return rig.trace()


def collect_budgets() -> list[Budget]:
    """Синтетические тайминги двух проходов против бюджетов раздела 23."""
    commands = _commands_trace()
    hybrid_llm = _hybrid_trace()
    return [
        Budget("pipeline_total_commands", commands.total_ms, 2500, "23: сквозной отклик"),
        Budget("pipeline_total_hybrid_llm", hybrid_llm.total_ms, 3000, "23: отклик с моделью"),
        Budget("stage_record_ms", commands.duration_of(Stage.RECORD), 2000, "23: запись фразы"),
        Budget("stage_stt_ms", commands.duration_of(Stage.STT), 2000, "23: распознавание"),
        Budget("stage_nlu_ms", commands.duration_of(Stage.NLU), 1000, "23: разбор намерения"),
        Budget("stage_action_ms", commands.duration_of(Stage.ACTION), 1000, "23: действие"),
        Budget("stage_tts_ms", commands.duration_of(Stage.TTS), 500, "23: TTS до звука"),
        Budget("stage_llm_ms", hybrid_llm.duration_of(Stage.LLM), 2000, "23: ответ модели"),
    ]


def _report_path() -> Path:
    """Куда писать отчёт: ``AYRIS_BENCH_OUT`` или ``reports/benchmarks.json``."""
    return Path(os.environ.get("AYRIS_BENCH_OUT", "reports/benchmarks.json"))


def write_report(budgets: Sequence[Budget]) -> Path:
    """Машиночитаемый отчёт задачи 74. Пишется всегда, даже если тест красный."""
    payload: dict[str, object] = {
        "schema": 1,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "deterministic": True,
        "metrics": {budget.name: budget.as_json() for budget in budgets},
        "ok": all(budget.ok for budget in budgets),
    }
    path = _report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_benchmarks_within_section_23_budgets() -> None:
    """Каждая стадия и сквозной проход укладываются в бюджет раздела 23.

    Отчёт пишется до утверждений, поэтому артефакт `nightly.yml` появляется при
    любом исходе; затем каждая метрика проверяется по отдельности, чтобы имя
    провалившегося бюджета было видно прямо в отчёте о падении.
    """
    budgets = collect_budgets()
    write_report(budgets)

    over = [f"{b.name}={b.value_ms}>{b.threshold_ms}" for b in budgets if not b.ok]
    assert not over, "бюджеты раздела 23 превышены: " + ", ".join(over)


def test_timings_are_deterministic_constants() -> None:
    """Синтетика воспроизводима до миллисекунды — иначе порог не контракт.

    Часы двигает тест, поэтому любая стадия-стопвотч показывает ровно
    :data:`TICK_MS`, а запись — длину буфера. Если это перестало быть так, порог
    в отчёте потерял смысл, и падать должно здесь, у источника, а не в бюджете.
    """
    commands = _commands_trace()
    assert commands.duration_of(Stage.RECORD) == SEGMENT_MS
    assert commands.duration_of(Stage.STT) == TICK_MS
    assert commands.duration_of(Stage.NLU) == TICK_MS
    assert commands.duration_of(Stage.ACTION) == TICK_MS
    assert commands.duration_of(Stage.TTS) == TICK_MS

    hybrid_llm = _hybrid_trace()
    assert hybrid_llm.duration_of(Stage.LLM) == TICK_MS
