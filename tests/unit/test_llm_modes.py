"""Задача 63: три режима NLU, LLM-разбор, история и tool calling — на моках.

Модель здесь никогда не настоящая: каждый клиент отдаёт заранее записанные
ответы (в том числе заведомо кривой JSON), поэтому тест детерминирован и ничего
не отправляет в сеть. Проверяется ровно то, что обещает файл задачи:

* матрица трёх режимов (COMMANDS / HYBRID / AI) на совпавшей и на незнакомой
  фразе — кто побеждает и доходит ли дело до модели;
* промах NLU (``{"command": null}``) уводит фразу в свободный чат;
* невалидный JSON даёт ровно один переспрос, после чего — честное «не нашла», а
  не случайное действие; кривой-затем-верный ответ доводит команду до запуска;
* переполнение окна истории сворачивается в резюме и переживает перезапуск, а
  само резюме вкладывается в системный промпт чата;
* и tool call, и мгновенный ответ ходят к действиям ТОЛЬКО через реестр, где
  живут проверки прав и подтверждение опасных команд (задачи 39, 40).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import ClassVar

import pytest

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import (
    ActionRegistry,
    ConfirmationCheck,
    ConfirmationRequest,
    ConfirmationVerdict,
)
from ayris.actions.result import ActionResult
from ayris.core.config import Settings
from ayris.core.events import EventBus
from ayris.core.models import ExecutionResult
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
    LlmToolCall,
)
from ayris.nlu.llm.memory import DialogMemory, InMemoryStore, MemoryState, is_reset
from ayris.nlu.llm.tools import CommandCard, RegistryGateway, SlotSpec
from ayris.nlu.matcher import Matcher, Trigger, TriggerKind

pytestmark = pytest.mark.unit


# --- действия для реестра (исполняются ТОЛЬКО через шлюз) -------------------
#: Куда действия отмечаются, что их запустили, — так тест видит, дошло ли дело
#: до тела. Чистится перед каждым тестом фикстурой ``_clear_ran``.
_RAN: list[str] = []


@pytest.fixture(autouse=True)
def _clear_ran() -> Iterator[None]:
    _RAN.clear()
    yield
    _RAN.clear()


class EchoPing(Action):
    """Безопасное действие: подтверждения не требует, просто отвечает."""

    meta: ClassVar[ActionMeta] = ActionMeta(
        name="EchoPing", category=ActionCategory.SYSTEM, title_ru="Пинг"
    )

    class Params(ActionParams):
        pass

    def run(self, params: EchoPing.Params) -> ActionResult[None]:
        _RAN.append("EchoPing")
        return ActionResult.done("Понг.")


class WipeDisk(Action):
    """Опасное действие: без подтверждения (задача 40) до тела дойти не должно."""

    meta: ClassVar[ActionMeta] = ActionMeta(
        name="WipeDisk",
        category=ActionCategory.SYSTEM,
        title_ru="Стереть диск",
        is_dangerous=True,
    )

    class Params(ActionParams):
        pass

    def run(self, params: WipeDisk.Params) -> ActionResult[None]:
        _RAN.append("WipeDisk")
        return ActionResult.done("Диск стёрт.")


class TimeReading(Action):
    """Мгновенный ответ (задача 25): срабатывает по детектору до модели."""

    meta: ClassVar[ActionMeta] = ActionMeta(
        name="TimeReading", category=ActionCategory.SYSTEM, title_ru="Сколько времени"
    )

    class Params(ActionParams):
        query: str = ""
        kind: str = ""

    def run(self, params: TimeReading.Params) -> ActionResult[None]:
        _RAN.append("TimeReading")
        return ActionResult.done("Сейчас 14:30.")


# --- модель на записанных ответах ------------------------------------------


class ScriptedLlm(LlmClient):
    """Клиент, отдающий подготовленные ответы по очереди и всё про себя помнящий.

    Ничего не считает и никуда не ходит: последний ответ повторяется, если
    вызовов больше, чем заготовок. ``prompts`` и ``tools`` копят всё, что уходило
    бы в сеть, — по ним тест проверяет и текст промпта, и предложенные инструменты.
    """

    name: ClassVar[str] = "scripted"

    def __init__(self, *responses: LlmResponse) -> None:
        self._responses = list(responses) or [_text("Готово.")]
        self.prompts: list[tuple[LlmMessage, ...]] = []
        self.tools: list[tuple[LlmTool, ...]] = []
        self.calls = 0

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
        self.tools.append(tuple(tools))
        response = self._responses[min(self.calls, len(self._responses) - 1)]
        self.calls += 1
        return response

    @property
    def last_prompt(self) -> tuple[LlmMessage, ...]:
        return self.prompts[-1]


class ToolLlm(ScriptedLlm):
    """То же, но с tool calling — иначе пайплайн не предложит инструменты."""

    supports_tools: ClassVar[bool] = True


def _text(text: str) -> LlmResponse:
    return LlmResponse(text=text, engine="scripted", finish_reason=FinishReason.STOP)


def _tool_reply(*calls: LlmToolCall) -> LlmResponse:
    return LlmResponse(tool_calls=calls, engine="scripted", finish_reason=FinishReason.TOOL_CALLS)


# --- окружение пайплайна ----------------------------------------------------


class Handle:
    """Заглушка ручки озвучки: барджа-ина в этих тестах нет."""

    def wait(self, timeout: float | None = None) -> bool:
        return True

    def cancel(self) -> bool:
        return True


class FakeTts:
    """Копит всё сказанное, чтобы тест сверил реплику."""

    def __init__(self) -> None:
        self.said: list[str] = []

    def say(self, text: str) -> Handle:
        self.said.append(text)
        return Handle()

    @property
    def last(self) -> str:
        return self.said[-1] if self.said else ""


class FakeActions:
    """Исполнитель совпавшей команды: запоминает запрос, но ничего не делает."""

    def __init__(self) -> None:
        self.seen: list[ActionRequest] = []

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        self.seen.append(request)
        return ActionOutcome(speak="Готово.")

    @property
    def calls(self) -> int:
        return len(self.seen)

    @property
    def last(self) -> ActionRequest:
        return self.seen[-1]


def _cards() -> list[CommandCard]:
    """Каталог из двух команд: «Громкость» со слотом value и «Открыть браузер»."""
    return [
        CommandCard(
            command_id=7,
            name="Громкость",
            description="громкость звука",
            phrases=("громкость 50",),
            slots=(SlotSpec(name="value", type="int"),),
        ),
        CommandCard(command_id=8, name="Открыть браузер", phrases=("открой браузер",)),
    ]


def _library() -> Matcher:
    """Матчер под те же два command_id, что и каталог, — для режима команд."""
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


def _settings(**ai: object) -> Settings:
    return Settings.model_validate({"ai": ai})


def _commands() -> Settings:
    return _settings(fallback_to_llm=False)


def _hybrid_fallback() -> Settings:
    return _settings(fallback_to_llm=True)


def _hybrid_understanding() -> Settings:
    return _settings(llm_understanding=True, fallback_to_llm=False)


def _hybrid_both() -> Settings:
    return _settings(llm_understanding=True, fallback_to_llm=True)


def _ai() -> Settings:
    return _settings(free_chat=True)


def _build(
    *,
    settings: Settings,
    llm: LlmClient | None = None,
    catalog: Callable[[], Sequence[CommandCard]] | None = None,
    gateway: RegistryGateway | None = None,
    memory: DialogMemory | None = None,
) -> tuple[Pipeline, FakeTts, FakeActions]:
    """Собрать пайплайн задачи 18 с моками и включённым LLM-швом задачи 63."""
    tts = FakeTts()
    actions = FakeActions()
    pipeline = Pipeline(
        EventBus(),
        matcher=_library(),
        tts=tts,
        actions=actions,
        llm=llm,
        settings=settings,
        scheduler=ManualScheduler(),
        runner=inline_runner,
        catalog=catalog,
        gateway=gateway,
        memory=memory,
    )
    return pipeline, tts, actions


@pytest.fixture
def make_gateway() -> Iterator[Callable[..., RegistryGateway]]:
    """Фабрика шлюзов над настоящим реестром; пулы воркеров глушим в конце."""
    registries: list[ActionRegistry] = []

    def build(
        *actions: type[Action],
        confirm: ConfirmationCheck | None = None,
        detect: Callable[[str], str] | None = None,
        instant_action: str = "InstantAnswer",
    ) -> RegistryGateway:
        registry = ActionRegistry(
            is_elevated=lambda: True, audit_enabled=lambda: False, confirm=confirm
        )
        for action_cls in actions:
            registry.add(action_cls)
        registries.append(registry)
        return RegistryGateway(registry, detect=detect, instant_action=instant_action)

    try:
        yield build
    finally:
        for registry in registries:
            registry.shutdown()


# --- матрица режимов на совпавшей фразе -------------------------------------


@pytest.mark.parametrize("settings", [_commands(), _hybrid_fallback()], ids=["commands", "hybrid"])
def test_matched_phrase_runs_command_without_model(settings: Settings) -> None:
    """И в «только командах», и в гибриде матчер побеждает — модель не трогаем."""
    llm = ScriptedLlm(_text("Модель не должна отвечать."))
    pipeline, _tts, actions = _build(settings=settings, llm=llm)
    result = pipeline.run_text("громкость 50")
    assert actions.calls == 1
    assert actions.last.command_id == 7
    assert llm.calls == 0
    assert result.command_id == 7


def test_matched_phrase_in_ai_mode_goes_to_model() -> None:
    """В режиме «только ИИ» матчер выключен: та же фраза уходит к модели."""
    llm = ScriptedLlm(_text("Пятьдесят чего?"))
    pipeline, _tts, actions = _build(settings=_ai(), llm=llm)
    result = pipeline.run_text("громкость 50")
    assert llm.calls == 1
    assert actions.calls == 0
    assert result.spoken == "Пятьдесят чего?"
    assert result.outcome is ExecutionResult.OK


# --- матрица режимов на незнакомой фразе ------------------------------------


def test_unknown_phrase_in_commands_mode_is_honest_miss() -> None:
    """«Только команды»: незнакомую фразу честно не нашли, модель не звали."""
    llm = ScriptedLlm(_text("Модель не должна отвечать."))
    pipeline, _tts, actions = _build(settings=_commands(), llm=llm)
    result = pipeline.run_text("расскажи про черепах")
    assert result.outcome is ExecutionResult.UNMATCHED
    assert result.spoken == NOT_MATCHED_MESSAGE
    assert llm.calls == 0
    assert actions.calls == 0


@pytest.mark.parametrize("settings", [_hybrid_fallback(), _ai()], ids=["hybrid", "ai"])
def test_unknown_phrase_falls_through_to_chat(settings: Settings) -> None:
    """Гибрид с фоллбеком и «только ИИ» уводят незнакомую фразу в свободный чат."""
    llm = ScriptedLlm(_text("Черепахи живут долго."))
    pipeline, _tts, actions = _build(settings=settings, llm=llm)
    result = pipeline.run_text("расскажи про черепах")
    assert llm.calls == 1
    assert result.spoken == "Черепахи живут долго."
    assert result.outcome is ExecutionResult.OK
    assert actions.calls == 0


# --- NLU-разбор: промах, кривой JSON, восстановление ------------------------


def test_nlu_miss_falls_through_to_chat() -> None:
    """Честный ``{"command": null}`` — не команда: фраза уходит в свободный чат."""
    llm = ScriptedLlm(_text('{"command": null}'), _text("Не команда, но вот вам факт."))
    pipeline, _tts, actions = _build(settings=_hybrid_both(), llm=llm, catalog=_cards)
    result = pipeline.run_text("посоветуй фильм")
    assert llm.calls == 2
    assert result.spoken == "Не команда, но вот вам факт."
    assert result.outcome is ExecutionResult.OK
    assert actions.calls == 0


def test_invalid_json_retries_once_then_honestly_misses() -> None:
    """Кривой JSON даёт один переспрос (строгий ретрай), затем честное «не нашла»."""
    llm = ScriptedLlm(_text("совсем не json"), _text("и снова не json"))
    pipeline, _tts, actions = _build(settings=_hybrid_understanding(), llm=llm, catalog=_cards)
    result = pipeline.run_text("посоветуй фильм")
    assert llm.calls == 2
    assert "строго одним JSON" in llm.prompts[1][-1].content
    assert result.outcome is ExecutionResult.UNMATCHED
    assert result.spoken == NOT_MATCHED_MESSAGE
    assert actions.calls == 0


def test_invalid_json_then_valid_routes_command() -> None:
    """Кривой-затем-верный ответ доводит команду до запуска — без случайных действий."""
    good = _text('{"command": "Громкость", "params": {"value": 30}, "confidence": 0.9}')
    llm = ScriptedLlm(_text("мусор"), good)
    pipeline, _tts, actions = _build(settings=_hybrid_understanding(), llm=llm, catalog=_cards)
    result = pipeline.run_text("сделай погромче на тридцать")
    assert llm.calls == 2
    assert actions.calls == 1
    assert actions.last.command_id == 7
    assert actions.last.slots == {"value": 30}
    assert actions.last.confirmed is False
    assert result.command_id == 7


# --- tool calling и мгновенные ответы через реестр --------------------------


def test_command_tool_call_routes_like_a_match() -> None:
    """Tool call ``cmd_<id>`` едет тем же путём, что и совпадение матчера."""
    llm = ToolLlm(_tool_reply(LlmToolCall(name="cmd_7", arguments={"value": 40})))
    pipeline, _tts, actions = _build(settings=_ai(), llm=llm, catalog=_cards)
    result = pipeline.run_text("сделай громче")
    assert actions.calls == 1
    assert actions.last.command_id == 7
    assert actions.last.slots == {"value": 40}
    assert actions.last.confirmed is False
    assert any(tool.name == "cmd_7" for tool in llm.tools[-1])
    assert result.command_id == 7


def test_builtin_action_tool_runs_through_registry(
    make_gateway: Callable[..., RegistryGateway],
) -> None:
    """Встроенное действие исполняется ТОЛЬКО реестром и озвучивает свой итог."""
    gateway = make_gateway(EchoPing)
    llm = ToolLlm(_tool_reply(LlmToolCall(name="EchoPing", arguments={})))
    pipeline, _tts, actions = _build(settings=_ai(), llm=llm, gateway=gateway)
    result = pipeline.run_text("пингани сервер")
    assert _RAN == ["EchoPing"]
    assert result.spoken == "Понг."
    assert result.outcome is ExecutionResult.OK
    assert actions.calls == 0


def test_dangerous_action_tool_is_refused_without_confirmation(
    make_gateway: Callable[..., RegistryGateway],
) -> None:
    """Опасное действие без подтверждения (задача 40) до тела не доходит."""

    def refuse(request: ConfirmationRequest) -> ConfirmationVerdict:
        return ConfirmationVerdict.no("нет подтверждения", user_message="Не стираю без спроса.")

    gateway = make_gateway(WipeDisk, confirm=refuse)
    llm = ToolLlm(_tool_reply(LlmToolCall(name="WipeDisk", arguments={})))
    pipeline, _tts, _actions = _build(settings=_ai(), llm=llm, gateway=gateway)
    result = pipeline.run_text("сотри диск")
    assert _RAN == []
    assert result.spoken == "Не стираю без спроса."


def test_instant_answer_intercepts_before_model(
    make_gateway: Callable[..., RegistryGateway],
) -> None:
    """«Сколько времени» перехватывает действие-провайдер — модель не зовём."""

    def detect(utterance: str) -> str:
        return "time" if "врем" in utterance else ""

    gateway = make_gateway(TimeReading, detect=detect, instant_action="TimeReading")
    llm = ScriptedLlm(_text("Модель не должна отвечать."))
    pipeline, _tts, _actions = _build(settings=_ai(), llm=llm, gateway=gateway)
    result = pipeline.run_text("сколько времени")
    assert _RAN == ["TimeReading"]
    assert result.spoken == "Сейчас 14:30."
    assert llm.calls == 0


# --- история диалога: окно, суммаризация, персистентность -------------------


def test_memory_overflow_summarizes_and_survives_restart() -> None:
    """Вытесненные реплики сворачиваются в резюме, а окно и резюме переживают рестарт."""
    seen: list[int] = []

    def summarize(messages: Sequence[LlmMessage]) -> str:
        seen.append(len(messages))
        return f"Резюме №{len(seen)}"

    store = InMemoryStore()
    memory = DialogMemory(
        store=store, summarizer=summarize, max_turns=4, summarize_after=2, profile_id=1
    )
    memory.remember("привет", "здравствуй")
    memory.remember("как дела", "хорошо")
    memory.remember("а погода", "ясно")

    window = [message.content for message in memory.messages()]
    assert len(window) == 4
    assert "привет" not in window
    assert memory.summary == "Резюме №1"
    assert len(seen) == 1

    reborn = DialogMemory(store=store, profile_id=1)
    assert reborn.summary == "Резюме №1"
    assert [message.content for message in reborn.messages()] == window


def test_summary_is_folded_into_chat_system_prompt() -> None:
    """Бегущее резюме уходит одним system-сообщением как «Ранее в разговоре: …»."""
    store = InMemoryStore()
    store.save(None, MemoryState(summary="Обсуждали поездку в Париж."))
    memory = DialogMemory(store=store)
    llm = ScriptedLlm(_text("Париж прекрасен весной."))
    pipeline, _tts, _actions = _build(settings=_ai(), llm=llm, memory=memory)
    result = pipeline.run_text("что там дальше")
    system = llm.last_prompt[0]
    assert system.role.value == "system"
    assert "Ранее в разговоре: Обсуждали поездку в Париж." in system.content
    assert result.spoken == "Париж прекрасен весной."


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Айрис, забудь всё!", True),
        ("забудь", True),
        ("Начни сначала.", True),
        ("новый разговор", True),
        ("включи музыку", False),
        ("расскажи, что было раньше", False),
        ("", False),
    ],
)
def test_reset_phrases_are_detected_on_normalized_text(text: str, expected: bool) -> None:
    """Просьба забыть разговор ловится по нормализованной форме, а обычная речь — нет."""
    assert is_reset(text) is expected
