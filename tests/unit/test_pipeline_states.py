"""Задача 18: машина стадий, её дедлайны и трейс одного прохода.

Здесь проверяются две вещи, которые в интеграционном тесте пайплайна видны
только косвенно: таблица переходов между семью стадиями и стопвотч, который
пишет строку из раздела 15 спецификации.

Три решения этого файла стоит знать заранее.

**Ни одного `sleep`.** Файл задачи предупреждает прямым текстом: тест на таймаут
с настоящим ожиданием первым начнёт мигать на windows-раннере, где планировщик
грубее. Поэтому дедлайны ждёт :class:`ManualScheduler`, а трейс тикает по
подставленным часам — тест утверждает точное число миллисекунд, а не «больше
нуля», и это утверждение про код, а не про загрузку машины.

**Таблица переходов проверяется с двух сторон.** Легальные переходы — половина
требования; вторая половина в том, что нелегальный отклоняется, а не проходит
молча. :class:`TestTransitions` перечисляет обе.

**Устаревший таймер важнее сработавшего.** Дедлайн, который выстрелил, пока
пайплайн уже перешёл дальше, — не «редкий случай», а норма при быстром проходе:
таймер живёт на своём потоке и отменяется не мгновенно. Если такой выстрел не
отбрасывать, он отменит следующую стадию. :class:`TestDeadlines` проверяет обе
причины отбрасывания — и стадию, и сессию.

Группы:

* :class:`TestPipelineState` — семь стадий, их подписи и ``is_active``.
* :class:`TestTransitions` — что разрешено, что отклонено, что делает ``force``.
* :class:`TestStateEvents` — ``PipelineStateChanged`` и его поля.
* :class:`TestDeadlines` — таймер: арм, снятие, устаревший выстрел.
* :class:`TestTimeouts` — вывод дедлайнов из настроек.
* :class:`TestManualScheduler` — сам подставной планировщик.
* :class:`TestStageTiming` — одна замерянная стадия.
* :class:`TestTrace` — стопвотч, строка лога, история, ``freeze``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from ayris.core.config import Settings
from ayris.core.events import EventBus, PipelineStateChanged
from ayris.core.models import ExecutionResult
from ayris.core.pipeline_states import (
    ALLOWED_TRANSITIONS,
    ASSISTANT_STATES,
    BUSY_STATES,
    STT_TIMEOUT_FACTOR,
    STT_TIMEOUT_FLOOR_SEC,
    ManualScheduler,
    PipelineState,
    PipelineStateMachine,
    PipelineTimeouts,
)
from ayris.core.pipeline_trace import PipelineTrace, Stage, StageTiming
from ayris.core.state import AssistantState

if TYPE_CHECKING:
    from collections.abc import Callable

pytestmark = pytest.mark.unit


class Clock:
    """Монотонные секунды, которые двигает тест.

    Тик по умолчанию — 0.25 с, то есть ровно 250 мс на замер: число, которое
    видно в утверждении и не зависит от того, как быстро выполнился блок.
    """

    def __init__(self, *, step: float = 0.25) -> None:
        self.now = 1000.0
        self.step = step

    def __call__(self) -> float:
        value = self.now
        self.now += self.step
        return value

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Deadline:
    """Взведённый дедлайн, который тест выстреливает руками."""

    delay: float
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        """Выстрелить, даже если таймер уже сняли.

        Настоящий ``threading.Timer`` так и делает: ``cancel()`` не догоняет
        поток, который уже начал звать коллбэк. Ровно от этой гонки машина
        защищается перепроверкой стадии и сессии, и проверить защиту можно
        только выстрелив по снятому таймеру.
        """
        self.callback()


class Deadlines:
    """Планировщик, который помнит все дедлайны, включая снятые."""

    def __init__(self) -> None:
        self.armed: list[Deadline] = []

    def call_later(self, delay: float, callback: Callable[[], None]) -> Deadline:
        deadline = Deadline(delay, callback)
        self.armed.append(deadline)
        return deadline


def machine(
    bus: EventBus,
    *,
    timeouts: PipelineTimeouts | None = None,
    on_timeout: Callable[[PipelineState], None] | None = None,
) -> tuple[PipelineStateMachine, ManualScheduler]:
    """Машина стадий с подставным планировщиком: дедлайны не идут сами."""
    scheduler = ManualScheduler()
    return (
        PipelineStateMachine(
            bus,
            timeouts=timeouts,
            on_timeout=on_timeout,
            scheduler=scheduler,
        ),
        scheduler,
    )


def states_of(events: list[PipelineStateChanged]) -> list[str]:
    return [event.state.value for event in events]


@pytest.fixture
def bus() -> EventBus:
    return EventBus()


@pytest.fixture
def seen(bus: EventBus) -> list[PipelineStateChanged]:
    """Все ``PipelineStateChanged`` в порядке публикации."""
    events: list[PipelineStateChanged] = []
    bus.subscribe(PipelineStateChanged, events.append, weak=False)
    return events


class TestPipelineState:
    """Сами стадии: их набор, подписи и отображение на четыре вида сферы."""

    def test_no_error_state(self) -> None:
        """Ошибочной стадии нет — из неё пришлось бы отдельно выбираться."""
        assert "ERROR" not in PipelineState.__members__
        assert len(PipelineState) == 7

    def test_only_idle_is_inactive(self) -> None:
        active = [state for state in PipelineState if state.is_active]
        assert PipelineState.IDLE not in active
        assert len(active) == len(PipelineState) - 1

    def test_every_state_has_a_russian_label(self) -> None:
        for state in PipelineState:
            assert state.label
            assert state.label.isascii() is False

    def test_every_state_maps_onto_the_sphere(self) -> None:
        """Семь стадий на четыре вида, и ни один не остаётся без вида."""
        assert set(ASSISTANT_STATES) == set(PipelineState)
        assert ASSISTANT_STATES[PipelineState.RECORDING] is AssistantState.LISTENING
        assert ASSISTANT_STATES[PipelineState.EXECUTING] is AssistantState.THINKING
        assert ASSISTANT_STATES[PipelineState.RESPONDING] is AssistantState.SPEAKING

    def test_responding_is_not_busy(self) -> None:
        """Иначе barge-in был бы невозможен: озвучку и надо перебивать."""
        assert PipelineState.RESPONDING not in BUSY_STATES
        assert PipelineState.IDLE not in BUSY_STATES
        assert PipelineState.TRANSCRIBING in BUSY_STATES


class TestTransitions:
    """Таблица переходов: и то, что разрешено, и то, что отклоняется."""

    def test_idle_is_reachable_from_everywhere(self) -> None:
        """Отмена, таймаут и упавшая стадия — это всё дорога в ``IDLE``."""
        for state, allowed in ALLOWED_TRANSITIONS.items():
            if state is PipelineState.IDLE:
                continue
            assert PipelineState.IDLE in allowed, state

    def test_no_state_may_transition_to_itself(self) -> None:
        for state, allowed in ALLOWED_TRANSITIONS.items():
            assert state not in allowed

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (PipelineState.IDLE, PipelineState.LISTENING),
            (PipelineState.IDLE, PipelineState.UNDERSTANDING),
            (PipelineState.LISTENING, PipelineState.RECORDING),
            (PipelineState.RECORDING, PipelineState.TRANSCRIBING),
            (PipelineState.TRANSCRIBING, PipelineState.UNDERSTANDING),
            (PipelineState.UNDERSTANDING, PipelineState.EXECUTING),
            (PipelineState.UNDERSTANDING, PipelineState.RESPONDING),
            (PipelineState.EXECUTING, PipelineState.RESPONDING),
            (PipelineState.RESPONDING, PipelineState.LISTENING),
        ],
    )
    def test_shortcuts_the_pipeline_needs_are_legal(
        self, bus: EventBus, source: PipelineState, target: PipelineState
    ) -> None:
        """Текстовая команда, промах матчера, действие без ответа."""
        state_machine, _ = machine(bus)
        state_machine.enter(source, session_id="s", force=True)
        assert state_machine.can_transition(target)
        assert state_machine.enter(target) is True
        assert state_machine.state is target

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (PipelineState.IDLE, PipelineState.EXECUTING),
            (PipelineState.IDLE, PipelineState.RESPONDING),
            (PipelineState.LISTENING, PipelineState.UNDERSTANDING),
            (PipelineState.RECORDING, PipelineState.EXECUTING),
            (PipelineState.EXECUTING, PipelineState.LISTENING),
            (PipelineState.RESPONDING, PipelineState.EXECUTING),
        ],
    )
    def test_illegal_moves_are_refused_not_silently_taken(
        self, bus: EventBus, source: PipelineState, target: PipelineState
    ) -> None:
        state_machine, _ = machine(bus)
        state_machine.enter(source, session_id="s", force=True)
        assert state_machine.can_transition(target) is False
        assert state_machine.enter(target) is False
        assert state_machine.state is source

    def test_entering_the_current_state_is_a_noop(self, bus: EventBus) -> None:
        """Второй вход в ту же стадию не публикует событие и не сбрасывает таймер."""
        state_machine, scheduler = machine(bus, timeouts=PipelineTimeouts(listening=5.0))
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        armed = scheduler.pending
        assert state_machine.enter(PipelineState.LISTENING, session_id="s") is False
        assert scheduler.pending == armed

    def test_force_bypasses_the_table(self, bus: EventBus) -> None:
        state_machine, _ = machine(bus)
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        assert state_machine.enter(PipelineState.EXECUTING, force=True) is True
        assert state_machine.state is PipelineState.EXECUTING

    def test_to_idle_works_from_any_state(self, bus: EventBus) -> None:
        for state in PipelineState:
            if state is PipelineState.IDLE:
                continue
            state_machine, _ = machine(bus)
            state_machine.enter(state, session_id="s", force=True)
            assert state_machine.to_idle() is True
            assert state_machine.state is PipelineState.IDLE


class TestStateEvents:
    """``PipelineStateChanged``: то, чем анимируется сфера и живёт DevTools."""

    def test_event_carries_both_ends_of_the_move(
        self, bus: EventBus, seen: list[PipelineStateChanged]
    ) -> None:
        state_machine, _ = machine(bus)
        state_machine.enter(PipelineState.LISTENING, session_id="abc", detail="жду фразу")
        bus.drain()
        (event,) = seen
        assert event.previous is PipelineState.IDLE
        assert event.state is PipelineState.LISTENING
        assert event.session_id == "abc"
        assert event.detail == "жду фразу"

    def test_session_id_is_kept_across_stages(
        self, bus: EventBus, seen: list[PipelineStateChanged]
    ) -> None:
        """Стадия без явного id остаётся в той же сессии — иначе трейс потеряет её."""
        state_machine, _ = machine(bus)
        state_machine.enter(PipelineState.LISTENING, session_id="abc")
        state_machine.enter(PipelineState.RECORDING)
        state_machine.enter(PipelineState.TRANSCRIBING)
        bus.drain()
        assert {event.session_id for event in seen} == {"abc"}
        assert state_machine.session_id == "abc"

    def test_idle_clears_the_session(self, bus: EventBus, seen: list[PipelineStateChanged]) -> None:
        state_machine, _ = machine(bus)
        state_machine.enter(PipelineState.LISTENING, session_id="abc")
        state_machine.to_idle(detail="отмена")
        bus.drain()
        assert state_machine.session_id == ""
        assert seen[-1].session_id == ""
        assert seen[-1].detail == "отмена"

    def test_happy_path_publishes_the_full_sequence(
        self, bus: EventBus, seen: list[PipelineStateChanged]
    ) -> None:
        state_machine, _ = machine(bus)
        for state in (
            PipelineState.LISTENING,
            PipelineState.RECORDING,
            PipelineState.TRANSCRIBING,
            PipelineState.UNDERSTANDING,
            PipelineState.EXECUTING,
            PipelineState.RESPONDING,
            PipelineState.IDLE,
        ):
            state_machine.enter(state, session_id="abc", force=True)
        bus.drain()
        assert states_of(seen) == [
            "listening",
            "recording",
            "transcribing",
            "understanding",
            "executing",
            "responding",
            "idle",
        ]

    def test_refused_move_publishes_nothing(
        self, bus: EventBus, seen: list[PipelineStateChanged]
    ) -> None:
        state_machine, _ = machine(bus)
        state_machine.enter(PipelineState.EXECUTING)
        bus.drain()
        assert seen == []


class TestDeadlines:
    """Дедлайн стадии: когда он ставится, когда снимается, когда игнорируется."""

    def test_deadline_is_armed_from_the_timeouts(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(bus, timeouts=PipelineTimeouts(listening=0.05))
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        assert scheduler.pending == (0.05,)

    def test_idle_arms_nothing(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(bus)
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        state_machine.to_idle()
        assert scheduler.pending == ()

    def test_zero_means_unbounded(self, bus: EventBus) -> None:
        """Тест не про таймауты выключает их, чтобы медленная машина не мешала."""
        state_machine, scheduler = machine(bus, timeouts=PipelineTimeouts(listening=0.0))
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        assert scheduler.pending == ()

    def test_next_stage_disarms_the_previous_deadline(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(
            bus, timeouts=PipelineTimeouts(listening=0.05, recording=0.09)
        )
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        state_machine.enter(PipelineState.RECORDING)
        assert scheduler.pending == (0.09,)

    def test_timeout_reports_the_state_that_ran_out(self, bus: EventBus) -> None:
        fired: list[PipelineState] = []
        state_machine, scheduler = machine(
            bus, timeouts=PipelineTimeouts(recording=0.01), on_timeout=fired.append
        )
        state_machine.enter(PipelineState.RECORDING, session_id="s", force=True)
        assert scheduler.fire_all() == 1
        assert fired == [PipelineState.RECORDING]

    def test_stale_timer_of_a_past_stage_is_dropped(self, bus: EventBus) -> None:
        """Выстрел таймера прошлой стадии не должен отменять текущую."""
        fired: list[PipelineState] = []
        deadlines = Deadlines()
        state_machine = PipelineStateMachine(
            bus,
            timeouts=PipelineTimeouts(listening=0.01, recording=0.0),
            on_timeout=fired.append,
            scheduler=deadlines,
        )
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        (stale,) = deadlines.armed
        state_machine.enter(PipelineState.RECORDING)
        stale.fire()
        assert fired == []
        assert state_machine.state is PipelineState.RECORDING

    def test_stale_timer_of_a_past_session_is_dropped(self, bus: EventBus) -> None:
        """Та же стадия, но другая сессия: barge-in переоткрыл её мгновенно."""
        fired: list[PipelineState] = []
        deadlines = Deadlines()
        state_machine = PipelineStateMachine(
            bus,
            timeouts=PipelineTimeouts(listening=0.01),
            on_timeout=fired.append,
            scheduler=deadlines,
        )
        state_machine.enter(PipelineState.LISTENING, session_id="first")
        stale = deadlines.armed[0]
        state_machine.to_idle()
        state_machine.enter(PipelineState.LISTENING, session_id="second")
        stale.fire()
        assert fired == []
        assert state_machine.state is PipelineState.LISTENING
        assert state_machine.session_id == "second"

    def test_disarm_keeps_the_stage(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(bus, timeouts=PipelineTimeouts(listening=0.05))
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        state_machine.disarm()
        assert scheduler.pending == ()
        assert state_machine.state is PipelineState.LISTENING

    def test_close_drops_the_deadline(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(bus, timeouts=PipelineTimeouts(listening=0.05))
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        state_machine.close()
        assert scheduler.pending == ()

    def test_new_timeouts_apply_from_the_next_stage(self, bus: EventBus) -> None:
        state_machine, scheduler = machine(
            bus, timeouts=PipelineTimeouts(listening=0.05, recording=0.05)
        )
        state_machine.enter(PipelineState.LISTENING, session_id="s")
        state_machine.apply_timeouts(PipelineTimeouts(listening=9.0, recording=0.02))
        assert scheduler.pending == (0.05,), "текущая стадия не перевзводится"
        state_machine.enter(PipelineState.RECORDING)
        assert scheduler.pending == (0.02,)

    def test_repr_names_the_stage_and_session(self, bus: EventBus) -> None:
        state_machine, _ = machine(bus)
        assert "idle" in repr(state_machine)
        state_machine.enter(PipelineState.LISTENING, session_id="abc")
        assert "listening" in repr(state_machine)
        assert "abc" in repr(state_machine)


class TestTimeouts:
    """Дедлайны выводятся из настроек, которые пользователь уже видит."""

    def test_defaults_are_all_positive(self) -> None:
        timeouts = PipelineTimeouts()
        for state in PipelineState:
            if state is PipelineState.IDLE:
                continue
            assert timeouts.for_state(state) > 0.0, state

    def test_idle_has_no_deadline(self) -> None:
        assert PipelineTimeouts().for_state(PipelineState.IDLE) == 0.0

    def test_listening_follows_the_wake_window(self) -> None:
        settings = Settings.model_validate({"voice": {"wake": {"listen_window_sec": 8.0}}})
        assert PipelineTimeouts.from_settings(settings).listening == 8.0

    def test_recording_leaves_room_for_the_silence_tail(self) -> None:
        """Сегментатор сам закрывает фразу; пайплайн не должен выстрелить раньше."""
        settings = Settings()
        timeouts = PipelineTimeouts.from_settings(settings)
        audio = settings.voice.audio_input
        assert timeouts.recording == audio.max_utterance_sec + audio.silence_ms / 1000.0
        assert timeouts.recording > audio.max_utterance_sec

    def test_stt_deadline_is_a_multiple_of_one_cloud_attempt(self) -> None:
        """Роутер в ``auto`` пробует облако, потом офлайн — одного таймаута мало."""
        settings = Settings()
        expected = max(
            settings.voice.stt.online_timeout_sec * STT_TIMEOUT_FACTOR,
            STT_TIMEOUT_FLOOR_SEC,
        )
        assert PipelineTimeouts.from_settings(settings).transcribing == expected

    def test_stt_deadline_has_a_floor(self) -> None:
        lowered = Settings.model_validate({"voice": {"stt": {"online_timeout_sec": 1.0}}})
        assert PipelineTimeouts.from_settings(lowered).transcribing == STT_TIMEOUT_FLOOR_SEC

    def test_understanding_follows_the_model_timeout(self) -> None:
        settings = Settings.model_validate({"ai": {"request_timeout_sec": 45.0}})
        assert PipelineTimeouts.from_settings(settings).understanding == 45.0

    def test_action_and_answer_keep_their_constants(self) -> None:
        """Ни одно поле настроек их не покрывает — они не про ожидание сети."""
        derived = PipelineTimeouts.from_settings(Settings())
        assert derived.executing == PipelineTimeouts().executing
        assert derived.responding == PipelineTimeouts().responding


class TestManualScheduler:
    """Планировщик, которым тесты заменяют настоящее ожидание."""

    def test_nothing_runs_until_told(self) -> None:
        scheduler = ManualScheduler()
        calls: list[str] = []
        scheduler.call_later(5.0, lambda: calls.append("late"))
        assert calls == []
        assert scheduler.pending == (5.0,)

    def test_fire_all_runs_everything_once(self) -> None:
        scheduler = ManualScheduler()
        calls: list[int] = []
        scheduler.call_later(1.0, lambda: calls.append(1))
        scheduler.call_later(2.0, lambda: calls.append(2))
        assert scheduler.fire_all() == 2
        assert calls == [1, 2]
        assert scheduler.fire_all() == 0
        assert calls == [1, 2]

    def test_cancelled_timer_neither_pends_nor_fires(self) -> None:
        scheduler = ManualScheduler()
        calls: list[str] = []
        timer = scheduler.call_later(1.0, lambda: calls.append("x"))
        timer.cancel()
        assert scheduler.pending == ()
        assert scheduler.fire_all() == 0
        assert calls == []

    def test_cancel_after_firing_is_safe(self) -> None:
        scheduler = ManualScheduler()
        timer = scheduler.call_later(1.0, lambda: None)
        scheduler.fire_all()
        timer.cancel()


class TestStageTiming:
    """Одна замерянная стадия и её вид в строке лога."""

    def test_describe_names_the_stage_in_russian(self) -> None:
        timing = StageTiming(stage=Stage.STT, duration_ms=240)
        assert timing.describe() == "распознавание 240 мс"

    def test_failed_stage_is_marked(self) -> None:
        timing = StageTiming(stage=Stage.ACTION, duration_ms=12, ok=False)
        assert timing.describe() == "действие 12 мс ✗"

    def test_as_json_is_flat(self) -> None:
        timing = StageTiming(stage=Stage.TTS, duration_ms=800, detail="piper")
        assert timing.as_json() == {
            "stage": "tts",
            "duration_ms": 800,
            "ok": True,
            "detail": "piper",
        }

    def test_every_stage_has_a_label(self) -> None:
        for stage in Stage:
            assert stage.label
            assert stage.label.isascii() is False


class TestTrace:
    """Стопвотч прохода: точные миллисекунды, строка раздела 15, история."""

    def test_stage_records_exact_milliseconds(self) -> None:
        clock = Clock(step=0.25)
        trace = PipelineTrace(session_id="s", clock=clock)
        with trace.stage(Stage.STT):
            pass
        assert trace.duration_of(Stage.STT) == 250

    def test_stages_are_kept_in_order(self) -> None:
        clock = Clock(step=0.1)
        trace = PipelineTrace(session_id="s", clock=clock)
        for stage in (Stage.STT, Stage.NLU, Stage.ACTION):
            with trace.stage(stage):
                pass
        assert [timing.stage for timing in trace.stages] == [Stage.STT, Stage.NLU, Stage.ACTION]

    def test_a_stage_that_raised_is_recorded_as_failed(self) -> None:
        """Иначе трейс упавшего прохода не покажет, сколько стоила упавшая стадия."""
        clock = Clock(step=0.05)
        trace = PipelineTrace(session_id="s", clock=clock)
        with pytest.raises(RuntimeError), trace.stage(Stage.ACTION):
            raise RuntimeError("нет такого окна")
        (timing,) = trace.stages
        assert timing.stage is Stage.ACTION
        assert timing.ok is False
        assert timing.duration_ms == 50

    def test_duration_of_a_stage_never_reached_is_zero(self) -> None:
        trace = PipelineTrace(session_id="s", clock=Clock())
        assert trace.duration_of(Stage.LLM) == 0

    def test_record_adds_a_timing_measured_elsewhere(self) -> None:
        """Длительность записи приходит от воркера, а не от стопвотча пайплайна."""
        trace = PipelineTrace(session_id="s", clock=Clock())
        trace.record(Stage.RECORD, 830)
        assert trace.duration_of(Stage.RECORD) == 830

    def test_repeated_stage_sums_up(self) -> None:
        trace = PipelineTrace(session_id="s", clock=Clock())
        trace.record(Stage.TTS, 300)
        trace.record(Stage.TTS, 200)
        assert trace.duration_of(Stage.TTS) == 500

    def test_total_counts_while_the_pass_runs(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="s", clock=clock)
        clock.advance(1.5)
        assert trace.total_ms == 1500
        clock.advance(0.5)
        assert trace.total_ms == 2000

    def test_finish_stops_the_stopwatch_and_is_idempotent(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="s", clock=clock)
        clock.advance(0.4)
        trace.finish()
        clock.advance(10.0)
        trace.finish()
        assert trace.total_ms == 400

    def test_log_line_is_the_section_15_shape(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="abc123", clock=clock)
        trace.stt_raw = "сделай громкость 50"
        trace.intent = "volume.set"
        trace.action = "volume.set"
        trace.record(Stage.STT, 240)
        trace.record(Stage.NLU, 3)
        trace.record(Stage.ACTION, 15)
        clock.advance(0.6)
        trace.finish()
        assert trace.log_line() == (
            "[abc123] STT raw: сделай громкость 50 → NLU intent: volume.set "
            "→ Action: volume.set → Result: ok — 600 мс "
            "[распознавание 240 мс | разбор 3 мс | действие 15 мс]"
        )

    def test_log_line_marks_the_empty_fields(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="abc", clock=clock)
        trace.finish()
        assert trace.log_line() == (
            "[abc] STT raw: — → NLU intent: — → Action: — → Result: ok — 0 мс"
        )

    def test_log_line_carries_the_error(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="abc", clock=clock)
        trace.outcome = ExecutionResult.ERROR
        trace.error = "engine down"
        trace.finish()
        assert "Result: error (engine down)" in trace.log_line()

    def test_as_json_carries_every_stage(self) -> None:
        trace = PipelineTrace(session_id="s", source="text", clock=Clock(step=0.0))
        trace.mode = "hybrid"
        trace.record(Stage.NLU, 4)
        payload = trace.as_json()
        assert payload["session_id"] == "s"
        assert payload["source"] == "text"
        assert payload["mode"] == "hybrid"
        assert payload["stages"] == [{"stage": "nlu", "duration_ms": 4, "ok": True, "detail": ""}]

    def test_as_json_rounds_the_scores(self) -> None:
        trace = PipelineTrace(session_id="s", clock=Clock(step=0.0))
        trace.stt_confidence = 0.918273
        trace.match_score = 0.876543
        payload = trace.as_json()
        assert payload["stt_confidence"] == 0.918
        assert payload["match_score"] == 0.877

    def test_history_row_carries_the_slots_and_the_stage_timings(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="sid", source="ptt", clock=clock)
        trace.mode = "commands"
        trace.stt_raw = "громкость 50"
        trace.intent = "volume.set"
        trace.command_id = 7
        trace.match_source = "exact"
        trace.slots = {"value": 50}
        trace.answer = "Готово."
        trace.record(Stage.STT, 240)
        trace.record(Stage.ACTION, 15)
        clock.advance(0.3)
        trace.finish()

        entry = trace.to_history()
        assert entry.stt_raw == "громкость 50"
        assert entry.matched_command_id == 7
        assert entry.intent == "volume.set"
        assert entry.result is ExecutionResult.OK
        assert entry.duration_ms == 300
        assert entry.params["value"] == 50
        assert entry.params["session_id"] == "sid"
        assert entry.params["source"] == "ptt"
        assert entry.params["mode"] == "commands"
        assert entry.params["answer"] == "Готово."
        assert entry.params["match_source"] == "exact"
        assert entry.params["stages_ms"] == {"stt": 240, "action": 15}

    def test_history_row_of_an_empty_pass_omits_what_it_has_not_got(self) -> None:
        trace = PipelineTrace(session_id="sid", clock=Clock(step=0.0))
        params = trace.to_history().params
        assert params == {"session_id": "sid"}

    def test_history_row_keeps_the_error(self) -> None:
        trace = PipelineTrace(session_id="sid", clock=Clock(step=0.0))
        trace.outcome = ExecutionResult.TIMEOUT
        trace.error = "timeout in transcribing"
        entry = trace.to_history()
        assert entry.result is ExecutionResult.TIMEOUT
        assert entry.error == "timeout in transcribing"

    def test_freeze_is_immutable_and_keeps_the_totals(self) -> None:
        clock = Clock(step=0.0)
        trace = PipelineTrace(session_id="sid", clock=clock)
        trace.record(Stage.STT, 240)
        clock.advance(0.5)
        trace.finish()
        record = trace.freeze()

        assert record.total_ms == 500
        assert record.duration_of(Stage.STT) == 240
        assert record.ok is True
        with pytest.raises((AttributeError, TypeError)):
            record.total_ms = 1  # type: ignore[misc]

    def test_freeze_does_not_follow_later_edits(self) -> None:
        """DevTools держит снимки; проход, изменившийся потом, не должен их трогать."""
        trace = PipelineTrace(session_id="sid", clock=Clock(step=0.0))
        trace.answer = "Готово."
        record = trace.freeze()
        trace.answer = "Что-то другое."
        trace.record(Stage.TTS, 900)
        assert record.answer == "Готово."
        assert record.duration_of(Stage.TTS) == 0

    def test_frozen_record_reports_a_failed_outcome(self) -> None:
        trace = PipelineTrace(session_id="sid", clock=Clock(step=0.0))
        trace.outcome = ExecutionResult.UNMATCHED
        assert trace.freeze().ok is False
