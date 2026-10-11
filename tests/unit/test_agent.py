"""Цикл агента: шаги, финальный ответ, лимит и отмена — на подставных функциях.

Модель и исполнение действий приходят в :func:`run_agent` замыканиями, поэтому
весь цикл проверяется без сети, без Windows и без живой модели: сценарный клиент
отдаёт заготовленные ответы, рекордер пишет исполненные вызовы.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ayris.nlu.llm.agent import AgentStep, run_agent
from ayris.nlu.llm.base import LlmMessage, LlmResponse, LlmRole, LlmToolCall
from ayris.nlu.llm.tools import ActionReply

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

pytestmark = pytest.mark.unit


def _seed() -> list[LlmMessage]:
    return [LlmMessage(role=LlmRole.USER, content="сделай что-нибудь")]


def _tool(name: str, **arguments: object) -> LlmResponse:
    return LlmResponse(tool_calls=(LlmToolCall(name=name, arguments=dict(arguments)),))


class _Scripted:
    """Отдаёт заготовленные ответы по очереди; запоминает длину каждого диалога."""

    def __init__(self, *responses: LlmResponse) -> None:
        self._responses = list(responses)
        self.seen: list[int] = []

    def __call__(self, history: Sequence[LlmMessage], tools: Sequence[object]) -> LlmResponse:
        self.seen.append(len(history))
        return self._responses.pop(0)


class _Recorder:
    """Исполнение действий: пишет вызовы, отдаёт заданный ответ."""

    def __init__(self, reply: ActionReply | None = None) -> None:
        self.calls: list[tuple[str, Mapping[str, object]]] = []
        self._reply = reply if reply is not None else ActionReply(ok=True, spoken="готово")

    def __call__(self, name: str, arguments: Mapping[str, object]) -> ActionReply:
        self.calls.append((name, dict(arguments)))
        return self._reply


class TestLoop:
    def test_answers_without_tools(self) -> None:
        model = _Scripted(LlmResponse(text="Готово, сделала."))
        recorder = _Recorder()
        outcome = run_agent(_seed(), (), complete=model, execute=recorder, max_steps=5)
        assert outcome.stop_reason == "answered"
        assert outcome.answer == "Готово, сделала."
        assert outcome.steps == ()
        assert recorder.calls == []

    def test_runs_one_tool_then_answers(self) -> None:
        model = _Scripted(_tool("CreateFolder", path="C:/x"), LlmResponse(text="Создала папку."))
        recorder = _Recorder(ActionReply(ok=True, spoken="Создала папку x."))
        outcome = run_agent(_seed(), (), complete=model, execute=recorder, max_steps=5)
        assert outcome.stop_reason == "answered"
        assert outcome.answer == "Создала папку."
        assert [step.tool for step in outcome.steps] == ["CreateFolder"]
        assert recorder.calls == [("CreateFolder", {"path": "C:/x"})]

    def test_result_is_fed_back_to_the_model(self) -> None:
        model = _Scripted(_tool("ReadTextFile", path="C:/a.txt"), LlmResponse(text="ответ"))
        recorder = _Recorder(ActionReply(ok=True, spoken="Прочитала.", content="СЕКРЕТ-ТЕКСТ"))
        outcome = run_agent(_seed(), (), complete=model, execute=recorder, max_steps=5)
        assert outcome.steps[0].result == "Прочитала. СЕКРЕТ-ТЕКСТ"
        # второй вызов модели увидел более длинный диалог — результат доехал
        assert model.seen == [1, 2]

    def test_stops_at_max_steps(self) -> None:
        model = _Scripted(*[_tool("ListDir", path="C:/x") for _ in range(10)])
        recorder = _Recorder()
        outcome = run_agent(_seed(), (), complete=model, execute=recorder, max_steps=3)
        assert outcome.stop_reason == "max_steps"
        assert len(outcome.steps) == 3
        assert "шаг" in outcome.answer.casefold()

    def test_cancel_stops_before_calling_the_model(self) -> None:
        model = _Scripted(LlmResponse(text="не должно прозвучать"))
        recorder = _Recorder()
        outcome = run_agent(
            _seed(), (), complete=model, execute=recorder, max_steps=5, cancel=lambda: True
        )
        assert outcome.stop_reason == "cancelled"
        assert outcome.steps == ()
        assert model.seen == []

    def test_failed_tool_is_fed_back(self) -> None:
        model = _Scripted(_tool("DeleteFile", path="C:/x"), LlmResponse(text="не вышло"))
        recorder = _Recorder(ActionReply(ok=False, spoken="Не нашла.", error="not found"))
        outcome = run_agent(_seed(), (), complete=model, execute=recorder, max_steps=5)
        assert outcome.steps[0].ok is False
        assert "ошибка" in outcome.steps[0].result
        assert outcome.answer == "не вышло"


class TestStepHooks:
    def test_callbacks_fire_per_step(self) -> None:
        model = _Scripted(_tool("ListDir", path="C:/x"), LlmResponse(text="вот"))
        recorder = _Recorder()
        starts: list[tuple[int, str, Mapping[str, object]]] = []
        steps: list[tuple[int, AgentStep]] = []
        run_agent(
            _seed(),
            (),
            complete=model,
            execute=recorder,
            max_steps=5,
            on_step_start=lambda index, tool, args: starts.append((index, tool, dict(args))),
            on_step=lambda index, step: steps.append((index, step)),
        )
        assert starts == [(1, "ListDir", {"path": "C:/x"})]
        assert [index for index, _ in steps] == [1]
        assert steps[0][1].tool == "ListDir"
