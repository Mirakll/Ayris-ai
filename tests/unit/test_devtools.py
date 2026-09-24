"""Задача 58: модели вкладки «Логи / DevTools» и защита текстового ввода.

Файл задачи предупреждает прямым текстом: тесты идут в offscreen-режиме и
проверяют *модели*, а не окно, — живой таймер пачечного обновления держит цикл
событий и вешает прогон в CI по таймауту. Поэтому здесь не создаётся ни одного
виджета: только модели без Qt-таймеров (:class:`LogViewModel`,
:class:`PipelineViewModel`, :class:`ReplModel`, :class:`WorkerHealthModel`) и один
головной проход пайплайна через :meth:`Pipeline.run_text` для обязательной
проверки безопасности.

Обязательная проверка (раздел «панель разработчика — не обход защит»): опасное
действие, запущенное из текстового поля, приходит в слой действий *без* отметки
подтверждения — ровно как только что распознанная голосовая команда, — и потому
проходит тот же гейт подтверждения из задачи 40. «Режима разработчика», который
снимал бы это подтверждение, нет.

Группы:

* :class:`TestLogViewModel` — ёмкость и вытеснение, фильтры уровня и модуля, поиск.
* :class:`TestPipelineViewModel` — метка времени первого показа, фильтр, порядок.
* :class:`TestReplModel` — выражение против оператора, трейсбек, история, дополнение.
* :class:`TestWorkerHealthModel` — аптайм по pid, RAM через сэмплер, флаг «мигает».
* :class:`TestTextInputSafety` — типизованная команда неподтверждена, сухой прогон не бежит.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from ayris.core.events import EventBus
from ayris.core.models import ExecutionResult
from ayris.core.pipeline import ActionOutcome, ActionRequest, Pipeline
from ayris.core.pipeline_states import ManualScheduler
from ayris.core.pipeline_trace import Stage, StageTiming, TraceRecord
from ayris.core.state import StateMachine
from ayris.gui.widgets.log_view import LogRow, LogViewModel, level_color_token
from ayris.gui.widgets.pipeline_view import PipelineRow, PipelineViewModel, result_label
from ayris.gui.widgets.repl_console import ReplModel, ReplResult
from ayris.gui.widgets.resource_monitor import Sampler
from ayris.gui.widgets.worker_health import WorkerHealthModel
from ayris.nlu.matcher import Matcher, Trigger
from ayris.workers.manager import WorkerStatus, WorkerSummary

pytestmark = pytest.mark.unit


def _row(
    *,
    level: str = "INFO",
    logger: str = "ayris.core.pipeline",
    message: str = "сообщение",
    request_id: str = "",
    created: float = 0.0,
) -> LogRow:
    return LogRow(
        created=created, level=level, logger=logger, message=message, request_id=request_id
    )


class TestLogViewModel:
    def test_capacity_evicts_oldest_and_forgets_its_module(self) -> None:
        model = LogViewModel(capacity=2)
        model.add(_row(logger="ayris.a"))
        model.add(_row(logger="ayris.b"))
        model.add(_row(logger="ayris.c"))
        assert model.count() == 2
        # Самая старая строка вытеснена, вместе с ней — её модуль из списка.
        assert model.known_modules == ["ayris.b", "ayris.c"]

    def test_level_filter_keeps_at_or_above_threshold(self) -> None:
        model = LogViewModel()
        model.set_level_filter("WARNING")
        assert not model.matches(_row(level="INFO"))
        assert model.matches(_row(level="WARNING"))
        assert model.matches(_row(level="ERROR"))
        model.set_level_filter("все уровни")
        assert model.matches(_row(level="DEBUG"))

    def test_module_filter_is_exact_or_dotted_prefix(self) -> None:
        model = LogViewModel()
        model.set_module_filter("ayris.core")
        assert model.matches(_row(logger="ayris.core"))
        assert model.matches(_row(logger="ayris.core.pipeline"))
        assert not model.matches(_row(logger="ayris.cores"))
        assert not model.matches(_row(logger="ayris.gui"))

    def test_search_spans_message_logger_and_request_id(self) -> None:
        model = LogViewModel()
        model.set_search("REQ-7")
        assert model.matches(_row(request_id="req-7"))
        assert model.matches(_row(message="строка req-7 тут"))
        assert model.matches(_row(logger="ayris.req-7"))
        assert not model.matches(_row(message="ничего"))

    def test_visible_rows_and_known_modules_sorted(self) -> None:
        model = LogViewModel()
        model.add(_row(logger="ayris.z", level="INFO"))
        model.add(_row(logger="ayris.a", level="ERROR"))
        model.set_level_filter("ERROR")
        visible = model.visible_rows()
        assert [row.logger for row in visible] == ["ayris.a"]
        assert model.known_modules == ["ayris.a", "ayris.z"]

    def test_log_row_format_and_short_logger(self) -> None:
        row = _row(level="ERROR", logger="ayris.core.pipeline", message="упало", request_id="s1")
        assert row.short_logger == "core.pipeline"
        line = row.format_line()
        assert "ERROR" in line
        assert "core.pipeline" in line
        assert "[s1]" in line
        assert "упало" in line

    def test_level_color_token(self) -> None:
        assert level_color_token("ERROR") == "error"
        assert level_color_token("CRITICAL") == "error"
        assert level_color_token("WARNING") == "warning"
        assert level_color_token("DEBUG") == "text_muted"
        assert level_color_token("INFO") == "text_secondary"


class _DateClock:
    """Datetime-часы, которые двигаются ровно на один тик за вызов."""

    def __init__(self) -> None:
        self._tick = 0

    def __call__(self) -> datetime:
        self._tick += 1
        return datetime(2026, 1, 1, 12, 0, self._tick)


def _trace(
    session_id: str,
    *,
    outcome: ExecutionResult = ExecutionResult.OK,
    intent: str = "",
    stages: tuple[StageTiming, ...] = (),
    slots: dict[str, str] | None = None,
) -> TraceRecord:
    payload: dict[str, object] = {"slots": slots} if slots else {}
    return TraceRecord(
        session_id=session_id, intent=intent, outcome=outcome, stages=stages, payload=payload
    )


class TestPipelineViewModel:
    def test_first_seen_stamp_is_kept_across_ingest(self) -> None:
        clock = _DateClock()
        model = PipelineViewModel(clock=clock)
        record = _trace("s1")
        assert model.ingest([record]) is True
        first_stamp = model.rows[0].seen_at
        # Тот же трейс приходит снова: метка времени не переснимается, а ingest
        # сообщает, что ничего не изменилось.
        assert model.ingest([record]) is False
        assert model.rows[0].seen_at == first_stamp

    def test_dropped_session_is_removed(self) -> None:
        model = PipelineViewModel(clock=_DateClock())
        model.ingest([_trace("s1")])
        assert model.ingest([]) is True
        assert model.count() == 0

    def test_ingest_flags_outcome_change(self) -> None:
        model = PipelineViewModel(clock=_DateClock())
        model.ingest([_trace("s1", outcome=ExecutionResult.OK)])
        assert model.ingest([_trace("s1", outcome=ExecutionResult.ERROR)]) is True

    def test_result_filter(self) -> None:
        model = PipelineViewModel(clock=_DateClock())
        model.ingest([_trace("ok"), _trace("bad", outcome=ExecutionResult.ERROR)])
        model.set_result_filter(ExecutionResult.ERROR.value)
        assert [row.session_id for row in model.visible_rows()] == ["bad"]
        model.set_result_filter("все результаты")
        assert len(model.visible_rows()) == 2

    def test_visible_rows_are_newest_first(self) -> None:
        model = PipelineViewModel(clock=_DateClock())
        model.ingest([_trace("old"), _trace("new")])
        assert [row.session_id for row in model.visible_rows()] == ["new", "old"]

    def test_longest_timing_and_intent_text(self) -> None:
        stages = (
            StageTiming(stage=Stage.NLU, duration_ms=5),
            StageTiming(stage=Stage.ACTION, duration_ms=40),
        )
        record = _trace("s1", intent="open", stages=stages, slots={"app": "браузер"})
        row = PipelineRow(seen_at=datetime(2026, 1, 1, 12, 0, 0), record=record)
        longest = row.longest_timing()
        assert longest is not None
        assert longest.stage is Stage.ACTION
        assert row.intent_text == "open (app=браузер)"

    def test_result_label(self) -> None:
        assert result_label(ExecutionResult.OK) == "успех"
        assert result_label(ExecutionResult.ERROR) == "ошибка"


class TestReplModel:
    def test_expression_echoes_value_and_binds_underscore(self) -> None:
        model = ReplModel({})
        result = model.execute("2 * 3")
        assert result.ok
        assert result.value_repr == "6"
        assert model.namespace["_"] == 6

    def test_statement_runs_without_echo(self) -> None:
        model = ReplModel({})
        assert model.execute("x = 41").value_repr == ""
        assert model.execute("x").value_repr == "41"

    def test_exception_becomes_traceback(self) -> None:
        result = ReplModel({}).execute("1 / 0")
        assert not result.ok
        assert "ZeroDivisionError" in result.error

    def test_stdout_is_captured(self) -> None:
        result = ReplModel({}).execute("print('ping')")
        assert result.output.strip() == "ping"

    def test_is_complete(self) -> None:
        model = ReplModel({})
        assert model.is_complete("1 + 1")
        assert not model.is_complete("for i in range(3):")
        # Синтаксическая ошибка считается «завершённой», чтобы Enter показал её,
        # а не запер пользователя на строке, которая никогда не скомпилируется.
        assert model.is_complete("x === 1")

    def test_completions_from_namespace_and_dir(self) -> None:
        model = ReplModel({"alpha": 1, "alto": 2, "beta": 3})
        # Имена из пространства имён вместе со встроенными (например «all»),
        # отсортированные и без лишнего «beta».
        matches = model.completions("al")
        assert "alpha" in matches
        assert "alto" in matches
        assert "beta" not in matches
        assert matches == sorted(matches)
        assert model.completions("gamma") == []
        assert model.completions("alpha.bit_l") == ["bit_length"]
        assert "__add__" not in model.completions("alpha.")
        assert "__add__" in model.completions("alpha.__")

    def test_history_dedup_and_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        model = ReplModel({}, history_path=path)
        model.remember("a")
        model.remember("a")
        model.remember("b")
        assert model.history == ["a", "b"]
        assert ReplModel({}, history_path=path).history == ["a", "b"]

    def test_history_limit(self) -> None:
        model = ReplModel({}, history_limit=2)
        for source in ("a", "b", "c"):
            model.remember(source)
        assert model.history == ["b", "c"]

    def test_repl_result_ok(self) -> None:
        assert ReplResult(source="x").ok
        assert not ReplResult(source="x", error="boom").ok


class _Clock:
    """Монотонные секунды под управлением теста; вызов их не двигает."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _FakeSampler(Sampler):
    def __init__(self, readings: dict[int, tuple[int, float]]) -> None:
        self._readings = readings

    def sample(self, pid: int) -> tuple[int, float] | None:
        return self._readings.get(pid)


def _summary(
    *,
    name: str = "stt",
    status: WorkerStatus = WorkerStatus.READY,
    pid: int | None = 100,
    restarts: int = 0,
) -> WorkerSummary:
    return WorkerSummary(
        name=name, kind="stt", status=status, pid=pid, restarts=restarts, last_heartbeat_age=1.0
    )


class TestWorkerHealthModel:
    def test_uptime_tracks_pid_across_polls(self) -> None:
        clock = _Clock()
        model = WorkerHealthModel(clock=clock)
        clock.now = 1000.0
        assert model.update([_summary(pid=7)])[0].uptime_s == 0.0
        clock.now = 1005.0
        assert model.update([_summary(pid=7)])[0].uptime_s == 5.0
        # Новый pid — воркер перезапустился, часы аптайма обнуляются.
        clock.now = 1010.0
        assert model.update([_summary(pid=8)])[0].uptime_s == 0.0

    def test_uptime_is_none_when_not_live(self) -> None:
        model = WorkerHealthModel()
        assert model.update([_summary(pid=8, status=WorkerStatus.STOPPED)])[0].uptime_s is None
        assert model.update([_summary(pid=None, status=WorkerStatus.STARTING)])[0].uptime_s is None

    def test_ram_from_sampler(self) -> None:
        sampler = _FakeSampler({7: (2 * 1024 * 1024, 3.5)})
        model = WorkerHealthModel(sampler=sampler)
        assert model.update([_summary(pid=7)])[0].ram_bytes == 2 * 1024 * 1024
        assert model.update([_summary(name="tts", pid=999)])[0].ram_bytes is None

    def test_hot_flag_at_threshold(self) -> None:
        model = WorkerHealthModel(restart_threshold=3)
        assert not model.update([_summary(restarts=2)])[0].hot
        assert model.update([_summary(restarts=3)])[0].hot

    def test_gone_worker_is_pruned(self) -> None:
        clock = _Clock()
        model = WorkerHealthModel(clock=clock)
        clock.now = 1.0
        model.update([_summary(pid=7)])
        assert "stt" in model._since
        model.update([])
        assert "stt" not in model._since


def _matcher_for(command_id: int, phrase: str) -> Matcher:
    return Matcher.from_triggers([Trigger(id=1, command_id=command_id, pattern=phrase)])


class _CapturingRunner:
    """Заменяет реестр действий задачи 19: запоминает запрос и ничего не гейтит."""

    def __init__(self) -> None:
        self.request: ActionRequest | None = None

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        self.request = request
        return ActionOutcome(result=ExecutionResult.OK, dangerous=True)


class TestTextInputSafety:
    """Панель разработчика — не обход защит (обязательная проверка задачи 58)."""

    def _pipeline(self, runner: _CapturingRunner | None) -> Pipeline:
        bus = EventBus(thread_id=None)
        return Pipeline(
            bus,
            state=StateMachine(bus),
            matcher=_matcher_for(42, "удали всё"),
            actions=runner,
            scheduler=ManualScheduler(),
        )

    def test_typed_command_reaches_action_unconfirmed(self) -> None:
        runner = _CapturingRunner()
        pipeline = self._pipeline(runner)
        try:
            result = pipeline.run_text("удали всё")
        finally:
            pipeline.close()
        assert result.ok
        assert runner.request is not None
        assert runner.request.command_id == 42
        # Не «режим разработчика»: команда из текстового поля приходит в слой
        # действий без отметки подтверждения — ровно как свежая голосовая, — и
        # значит проходит тот же гейт опасных действий из задачи 40.
        assert runner.request.confirmed is False

    def test_dry_run_matches_without_executing(self) -> None:
        runner = _CapturingRunner()
        pipeline = self._pipeline(runner)
        try:
            result = pipeline.run_text("удали всё", execute=False)
        finally:
            pipeline.close()
        # «Сухой прогон»: команда распознана, но не выполнена — раннер не вызван.
        assert result.ok
        assert result.command_id == 42
        assert runner.request is None
