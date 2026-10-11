"""Автономный агент: цикл «модель → инструмент → результат → модель».

Это единственный новый кусок поверх уже готового вызова инструментов (задача 63):
обычный чат делает один шаг и останавливается, а агент продолжает — исполняет
вызванные моделью действия, возвращает их результат в диалог и спрашивает модель
снова, пока она не ответит словами или не кончится лимит шагов.

Модуль чистый: он не знает ни о реестре действий, ни о шине, ни о воркере. Модель
приходит замыканием ``complete``, исполнение действий — замыканием ``execute``, а
отмена и лог шагов — предикатом ``cancel`` и колбэками. Поэтому весь цикл
проверяется на подставных функциях, без сети, без Windows и без живой модели.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayris.nlu.llm.base import LlmMessage, LlmRole

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from ayris.nlu.llm.base import LlmResponse, LlmTool
    from ayris.nlu.llm.tools import ActionReply

__all__ = ["AgentOutcome", "AgentStep", "run_agent"]

#: Сколько символов результата кладём обратно в диалог — чтобы один вызов не съел
#: контекст модели.
_RESULT_CLIP: Final = 2000
_EXHAUSTED_MESSAGE: Final = "Не уложилась в отведённое число шагов."


@dataclass(frozen=True, slots=True)
class AgentStep:
    """Один исполненный шаг: какой инструмент, с чем, чем кончился."""

    tool: str
    arguments: Mapping[str, object]
    ok: bool
    result: str


@dataclass(frozen=True, slots=True)
class AgentOutcome:
    """Итог хода агента: финальный ответ, пройденные шаги и причина остановки."""

    answer: str
    steps: tuple[AgentStep, ...]
    stop_reason: str  # "answered" | "max_steps" | "cancelled"


def run_agent(
    history: list[LlmMessage],
    tools: Sequence[LlmTool],
    *,
    complete: Callable[[Sequence[LlmMessage], Sequence[LlmTool]], LlmResponse],
    execute: Callable[[str, Mapping[str, object]], ActionReply],
    max_steps: int,
    cancel: Callable[[], bool] = lambda: False,
    on_step_start: Callable[[int, str, Mapping[str, object]], None] | None = None,
    on_step: Callable[[int, AgentStep], None] | None = None,
) -> AgentOutcome:
    """Крутить «вызов → исполнение → результат» до ответа словами или лимита.

    ``complete`` зовёт модель (замыкание над клиентом и параметрами), ``execute``
    исполняет действие через реестр (с правами и подтверждением опасных). ``cancel``
    проверяется перед каждым обращением к модели и каждым действием, поэтому «стоп»
    рвёт цикл на ближайшей границе. ``on_step_start``/``on_step`` — крючки лога
    шагов; маскировку аргументов делает вызывающая сторона.
    """
    convo = list(history)
    steps: list[AgentStep] = []
    for _ in range(max(1, max_steps)):
        if cancel():
            return AgentOutcome("", tuple(steps), "cancelled")
        response = complete(convo, tools)
        if not response.has_tool_calls:
            return AgentOutcome(response.text.strip(), tuple(steps), "answered")
        text = response.text.strip()
        if text:
            convo.append(LlmMessage.assistant(text))
        for call in response.tool_calls:
            if cancel():
                return AgentOutcome("", tuple(steps), "cancelled")
            arguments = dict(call.arguments)
            index = len(steps) + 1
            if on_step_start is not None:
                on_step_start(index, call.name, arguments)
            reply = execute(call.name, arguments)
            step = AgentStep(
                tool=call.name, arguments=arguments, ok=reply.ok, result=reply.for_agent
            )
            steps.append(step)
            if on_step is not None:
                on_step(index, step)
            convo.append(_result_message(call.name, step.result))
    return AgentOutcome(_EXHAUSTED_MESSAGE, tuple(steps), "max_steps")


def _result_message(tool: str, result: str) -> LlmMessage:
    """Результат инструмента обратно в диалог обычным ``user``-сообщением.

    Не ``tool``-ролью: шов :class:`LlmMessage` не несёт идентификатор вызова, а
    облачный API отвергает ``tool`` без парного ``assistant``-хода с ``tool_calls``.
    Для первого облачного агента текстовой обратной связи достаточно.
    """
    clipped = result if len(result) <= _RESULT_CLIP else result[:_RESULT_CLIP] + "…"
    return LlmMessage(role=LlmRole.USER, content=f"[результат «{tool}»] {clipped}")
