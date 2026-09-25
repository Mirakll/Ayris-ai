"""Реестр действий и библиотека команд как инструменты для модели.

Три поставки для LLM-режима NLU (задача 63), собранные из того, что уже описано
типами в других местах, без нового источника правды:

* :class:`CommandCard` — компактная карточка команды пользователя (имя, о чём она,
  пара характерных фраз, слоты и их типы). Из карточек собирается блок для
  промпта (:func:`render_catalog`) с отбором по релевантности и бюджетом
  (:func:`select_cards`), чтобы список не раздувался до всего контекста;
* :func:`command_tools` / :func:`action_tools` — те же команды и встроенные
  действия в виде :class:`~ayris.nlu.llm.base.LlmTool` для провайдеров с
  tool calling. Имя инструмента команды — ``cmd_<id>``: имена команд бывают
  кириллические и с пробелами, а контракт функций OpenAI требует ``[A-Za-z0-9_-]``;
* :class:`RegistryGateway` — исполнение того, что выбрала модель, ТОЛЬКО через
  :class:`~ayris.actions.registry.ActionRegistry`: те же проверки прав и
  подтверждения опасных команд (задачи 39, 40), что и у остального приложения.
  Модель не получает прямого доступа к действиям в обход реестра.

Мгновенные ответы (погода, курс, время — задача 25) перехватываются здесь же, до
обращения к модели: :meth:`RegistryGateway.instant_answer` спрашивает переданный
детектор, claim ли какой-нибудь провайдер эту фразу, и лишь тогда запускает
действие. Голая фраза-вопрос без явного провайдера уходит дальше, к модели.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from ayris.nlu.llm.base import LlmTool

if TYPE_CHECKING:
    from ayris.actions.base import ActionSchema
    from ayris.actions.registry import ActionRegistry

__all__ = [
    "TOOL_PREFIX",
    "ActionReply",
    "CommandCard",
    "InstantDetector",
    "RegistryGateway",
    "SlotSpec",
    "action_tools",
    "command_tool_name",
    "command_tools",
    "parse_command_tool",
    "render_catalog",
    "select_cards",
]

_log = logging.getLogger("ayris.nlu.llm.tools")

#: Префикс имени инструмента, за которым прячется команда пользователя.
TOOL_PREFIX: Final = "cmd_"

#: Сколько команд и сколько символов их блока попадают в промпт по умолчанию.
_DEFAULT_LIMIT: Final = 12
_DEFAULT_BUDGET_CHARS: Final = 1600

_WORD = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)

#: Кто решает, claim ли какой-нибудь мгновенный провайдер эту фразу. Возвращает
#: вид ответа (``weather`` / ``rates`` / ``time`` / ``fact`` …) или пустую строку.
InstantDetector = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """Один слот команды: имя, тип и обязателен ли он — для промпта и схемы."""

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""

    def as_property(self) -> dict[str, Any]:
        """JSON-schema-описание слота для параметров инструмента."""
        prop: dict[str, Any] = {"type": _json_type(self.type)}
        if self.description:
            prop["description"] = self.description
        return prop


@dataclass(frozen=True, slots=True)
class CommandCard:
    """Команда пользователя в том виде, в каком её показывают модели."""

    command_id: int
    name: str
    description: str = ""
    phrases: tuple[str, ...] = ()
    slots: tuple[SlotSpec, ...] = ()

    def render(self) -> str:
        """Одна строка списка команд для NLU-промпта."""
        parts = [f"- {self.name}"]
        if self.description:
            parts.append(f": {self.description}")
        if self.phrases:
            sample = "; ".join(self.phrases[:3])
            parts.append(f" (например: {sample})")
        if self.slots:
            slots = ", ".join(f"{slot.name}:{slot.type}" for slot in self.slots)
            parts.append(f" [слоты: {slots}]")
        return "".join(parts)

    def tokens(self) -> set[str]:
        """Слова имени и фраз — по ним карточка ранжируется под фразу пользователя."""
        haystack = " ".join((self.name, self.description, *self.phrases))
        return _tokens(haystack)


def _tokens(text: str) -> set[str]:
    return {match.group(0).casefold() for match in _WORD.finditer(text)}


def _json_type(slot_type: str) -> str:
    mapping = {
        "int": "integer",
        "integer": "integer",
        "number": "number",
        "float": "number",
        "bool": "boolean",
        "boolean": "boolean",
    }
    return mapping.get(slot_type.casefold(), "string")


def select_cards(
    cards: Sequence[CommandCard],
    utterance: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    budget_chars: int = _DEFAULT_BUDGET_CHARS,
) -> list[CommandCard]:
    """Самые близкие к фразе команды, обрезанные по числу и по бюджету символов.

    Отбор — по доле общих слов фразы и карточки. Команды без единого общего слова
    остаются в хвосте: они попадут в список, только если под бюджет влезли все
    более релевантные, — так модель видит и точные попадания, и запас, но не весь
    справочник целиком.
    """
    wanted = _tokens(utterance)

    def score(card: CommandCard) -> tuple[int, int]:
        overlap = len(wanted & card.tokens())
        # Второй ключ — стабильность: одинаковый вход даёт одинаковый порядок.
        return (overlap, -card.command_id)

    ranked = sorted(cards, key=score, reverse=True)
    chosen: list[CommandCard] = []
    spent = 0
    for card in ranked[: max(limit, 0)]:
        line = card.render()
        spent += len(line) + 1
        if chosen and spent > budget_chars:
            break
        chosen.append(card)
    return chosen


def render_catalog(
    cards: Sequence[CommandCard],
    utterance: str = "",
    *,
    limit: int = _DEFAULT_LIMIT,
    budget_chars: int = _DEFAULT_BUDGET_CHARS,
) -> str:
    """Готовый блок списка команд для подстановки в NLU-промпт."""
    selected = select_cards(cards, utterance, limit=limit, budget_chars=budget_chars)
    return "\n".join(card.render() for card in selected)


def command_tool_name(command_id: int) -> str:
    """Имя инструмента для команды с этим идентификатором."""
    return f"{TOOL_PREFIX}{command_id}"


def parse_command_tool(name: str) -> int | None:
    """Идентификатор команды из имени инструмента, или ``None`` для чужого имени."""
    if not name.startswith(TOOL_PREFIX):
        return None
    try:
        return int(name[len(TOOL_PREFIX) :])
    except ValueError:
        return None


def command_tools(cards: Sequence[CommandCard]) -> list[LlmTool]:
    """Команды пользователя в виде инструментов для провайдеров с tool calling."""
    tools: list[LlmTool] = []
    for card in cards:
        properties = {slot.name: slot.as_property() for slot in card.slots}
        required = [slot.name for slot in card.slots if slot.required]
        parameters: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            parameters["required"] = required
        description = card.description or card.name
        if card.phrases:
            description = f"{description}. Например: {card.phrases[0]}"
        tools.append(
            LlmTool(
                name=command_tool_name(card.command_id),
                description=description,
                parameters=parameters,
            )
        )
    return tools


def action_tools(schemas: Sequence[ActionSchema]) -> list[LlmTool]:
    """Встроенные действия (задача 19) в виде инструментов модели."""
    tools: list[LlmTool] = []
    for schema in schemas:
        tools.append(
            LlmTool(
                name=schema.name,
                description=schema.description_ru or schema.name,
                parameters=dict(schema.json_schema),
            )
        )
    return tools


@dataclass(frozen=True, slots=True)
class ActionReply:
    """Итог исполнения действия, пригодный и для озвучки, и для возврата модели."""

    ok: bool
    spoken: str = ""
    detail: str = ""
    error: str = ""
    dangerous: bool = False

    @property
    def as_tool_result(self) -> str:
        """Что вернуть модели как результат tool call, чтобы она это озвучила."""
        return self.spoken or self.detail or ("ошибка: " + self.error if self.error else "готово")


class RegistryGateway:
    """Исполнение выбора модели через реестр действий — единственная дверь к нему.

    И мгновенные ответы, и tool call ходят только сюда, а отсюда — только в
    :meth:`~ayris.actions.registry.ActionRegistry.execute`, где уже сидят проверки
    прав и подтверждение опасных команд (задачи 39, 40). Модель не получает
    отдельного пути к действиям, поэтому ужесточать защиту в двух местах не надо.
    """

    def __init__(
        self,
        registry: ActionRegistry,
        *,
        detect: InstantDetector | None = None,
        instant_action: str = "InstantAnswer",
    ) -> None:
        self._registry = registry
        self._detect = detect
        self._instant_action = instant_action

    def instant_answer(self, utterance: str, *, request_id: str = "") -> str | None:
        """Мгновенный ответ на фразу, если её claim какой-нибудь провайдер.

        Вопрос без явного провайдера (просто «расскажи что-нибудь») не перехватыва-
        ется здесь: детектор возвращает пустой вид, и фраза уходит к модели. Так
        числа и даты не выдумываются, а болтовня остаётся болтовнёй.
        """
        if self._detect is None:
            return None
        kind = self._detect(utterance)
        if not kind:
            return None
        if not self._registry.has(self._instant_action):
            return None
        reply = self.run_action(
            self._instant_action,
            {"query": utterance, "kind": kind},
            request_id=request_id,
        )
        return reply.spoken or None

    def run_action(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionReply:
        """Выполнить действие и свести его итог к пригодной для озвучки реплике."""
        from ayris.core.errors import ActionError, ActionNotConfirmed

        try:
            result = self._registry.execute(
                name, params, request_id=request_id, command_id=command_id
            )
        except ActionNotConfirmed as exc:
            return ActionReply(
                ok=False, spoken=exc.user_message, error=exc.technical, dangerous=True
            )
        except ActionError as exc:
            _log.info("действие %s не выполнено: %s", name, exc.technical)
            return ActionReply(ok=False, spoken=exc.user_message, error=exc.technical)
        return ActionReply(
            ok=bool(result.ok),
            spoken=result.message_ru,
            detail=result.detail,
        )

    def action_tools(self) -> list[LlmTool]:
        """Все зарегистрированные действия как инструменты модели."""
        return action_tools(self._registry.describe_all())
