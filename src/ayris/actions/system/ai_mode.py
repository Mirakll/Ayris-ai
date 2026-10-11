"""Переключение агентного режима ИИ — голосом или из макроса.

Редкий случай действия, которое ПИШЕТ настройку, а не читает: оно задаёт или
инвертирует ``ai.agent_mode`` через :meth:`ConfigManager.apply`, а тот сохраняет
файл и живьём рассылает ``ConfigChanged`` — пайплайн подхватывает новый режим без
перезапуска. Нужно, чтобы агент включался не только тумблером в настройках, но и
голосом: «включи режим агента».
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult

__all__ = ["SetAgentMode", "ToggleAgentMode"]


def _apply_agent_mode(enabled: bool) -> None:
    from ayris.core.config import get_config_manager

    get_config_manager().apply({"ai.agent_mode": enabled})


def _agent_mode() -> bool:
    from ayris.core.config import get_settings

    return bool(get_settings().ai.agent_mode)


@register
class SetAgentMode(Action):
    """Включить или выключить агентный режим ИИ."""

    meta: ClassVar = ActionMeta(
        name="SetAgentMode",
        category=ActionCategory.SYSTEM,
        title_ru="Агентный режим",
        description_ru="Включить или выключить агентный режим ИИ",
    )

    class Params(ActionParams):
        enabled: bool = Field(
            title="Включить",
            description="Включить агентный режим (true) или выключить (false)",
        )

    def run(self, params: Params) -> ActionResult[bool]:
        _apply_agent_mode(params.enabled)
        state = "включён" if params.enabled else "выключен"
        return ActionResult.done(
            f"Агентный режим {state}.",
            value=params.enabled,
            detail=f"ai.agent_mode set to {params.enabled}",
            data={"agent_mode": params.enabled},
        )


@register
class ToggleAgentMode(Action):
    """Переключить агентный режим ИИ: вкл, если выключен, и наоборот."""

    meta: ClassVar = ActionMeta(
        name="ToggleAgentMode",
        category=ActionCategory.SYSTEM,
        title_ru="Переключить агентный режим",
        description_ru="Включить агентный режим, если он выключен, иначе выключить",
    )

    class Params(ActionParams):
        pass

    def run(self, params: Params) -> ActionResult[bool]:
        del params
        new_value = not _agent_mode()
        _apply_agent_mode(new_value)
        state = "включён" if new_value else "выключен"
        return ActionResult.done(
            f"Агентный режим {state}.",
            value=new_value,
            detail=f"ai.agent_mode toggled to {new_value}",
            data={"agent_mode": new_value},
        )
