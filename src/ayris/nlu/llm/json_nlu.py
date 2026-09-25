"""Разбор и проверка JSON-ответа модели в NLU-режиме.

Модель просят вернуть строго ``{"command", "params", "confidence"}``, но она
регулярно оборачивает его в markdown-забор, дописывает пояснение до или после, а
иногда выдаёт мусор. Этот модуль всё это переживает, но не «догадывается» о
смысле: он вытаскивает первый сбалансированный объект, проверяет форму и
сопоставляет имя команды с каталогом (:mod:`ayris.nlu.llm.tools`). Чего он не
делает — так это не превращает кривой ответ в случайное действие: непонятный
ответ помечается как невалидный (повод для одного уточняющего ретрая), а честное
``command: null`` или незнакомое имя — как промах, по которому пайплайн скажет
«не поняла», а не запустит что попало.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from ayris.nlu.llm.tools import CommandCard, parse_command_tool

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "NluDecision",
    "interpret",
    "parse_json_object",
    "resolve_command",
]

_log = logging.getLogger("ayris.nlu.llm.json_nlu")

#: Порог доверия к выбору модели. Ниже него уверенно выбранная, но сомнительная
#: команда считается промахом — лучше «не поняла», чем не то действие.
DEFAULT_MIN_CONFIDENCE: Final = 0.3


@dataclass(frozen=True, slots=True)
class NluDecision:
    """Что модель поняла из фразы после разбора и проверки.

    ``invalid`` отделяет «ответ не удалось прочитать» (стоит переспросить один
    раз) от честного промаха — ``command: null`` или незнакомое имя с уже
    исчерпанным ретраем, — по которому переспрашивать бесполезно.
    """

    resolved: bool
    command_id: int | None = None
    command_name: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""
    invalid: bool = False


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Первый сбалансированный JSON-объект из ответа модели, или ``None``.

    Сначала пробуем весь текст (сняв markdown-забор), затем — первый ``{...}``,
    считая скобки и уважая строковые литералы, чтобы ``}`` внутри строки не
    оборвал объект раньше времени.
    """
    stripped = _strip_fence(text)
    parsed = _loads(stripped)
    if parsed is not None:
        return parsed
    span = _first_object(stripped)
    if span is None:
        return None
    return _loads(span)


def resolve_command(cards: Sequence[CommandCard], command: Any) -> CommandCard | None:
    """Карточка команды по её имени или идентификатору из ответа модели."""
    if command is None:
        return None
    if isinstance(command, bool):  # bool — подкласс int; это не идентификатор
        return None
    if isinstance(command, int):
        return _by_id(cards, command)
    if not isinstance(command, str):
        return None
    name = command.strip()
    if not name:
        return None
    by_tool = parse_command_tool(name)
    if by_tool is not None:
        return _by_id(cards, by_tool)
    folded = name.casefold()
    for card in cards:
        if card.name.casefold() == folded:
            return card
    return None


def interpret(
    text: str,
    cards: Sequence[CommandCard],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> NluDecision:
    """Разобрать ответ модели и решить, какую команду (если есть) он называет."""
    obj = parse_json_object(text)
    if obj is None:
        return NluDecision(resolved=False, reason="ответ не разобран", invalid=True)
    if "command" not in obj:
        return NluDecision(resolved=False, reason="нет поля command", invalid=True)
    command = obj["command"]
    if command is None or (isinstance(command, str) and not command.strip()):
        return NluDecision(resolved=False, reason="команда не выбрана")
    card = resolve_command(cards, command)
    if card is None:
        # Модель назвала команду не из списка — выдумала. Один переспрос ещё
        # может помочь, поэтому это «невалидно», а не окончательный промах.
        return NluDecision(
            resolved=False, command_name=str(command), reason="неизвестная команда", invalid=True
        )
    confidence = _confidence(obj.get("confidence"))
    if confidence < min_confidence:
        return NluDecision(
            resolved=False,
            command_id=card.command_id,
            command_name=card.name,
            confidence=confidence,
            reason="низкая уверенность",
        )
    return NluDecision(
        resolved=True,
        command_id=card.command_id,
        command_name=card.name,
        params=_clean_params(obj.get("params"), card),
        confidence=confidence,
        reason="ok",
    )


def _by_id(cards: Sequence[CommandCard], command_id: int) -> CommandCard | None:
    for card in cards:
        if card.command_id == command_id:
            return card
    return None


def _clean_params(raw: Any, card: CommandCard) -> dict[str, Any]:
    """Только слоты, объявленные у команды: чужие ключи модель придумала."""
    if not isinstance(raw, Mapping):
        return {}
    allowed = {slot.name for slot in card.slots}
    return {str(key): value for key, value in raw.items() if str(key) in allowed}


def _confidence(raw: Any) -> float:
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, int | float):
        return max(0.0, min(1.0, float(raw)))
    if isinstance(raw, str):
        try:
            return max(0.0, min(1.0, float(raw.strip())))
        except ValueError:
            return 0.0
    # Поле не прислали — считаем, что модель уверена: форма ответа корректна.
    return 1.0 if raw is None else 0.0


def _loads(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    # Убираем первую строку с ``` (и, возможно, языком) и хвостовой забор.
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _first_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
