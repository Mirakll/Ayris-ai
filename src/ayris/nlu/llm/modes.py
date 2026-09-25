"""Три режима распознавания и политика каждого — единая точка для пайплайна и UI.

Сам перечень :class:`~ayris.core.pipeline.NluMode` и чтение его из настроек
(:func:`~ayris.core.pipeline.mode_from_config`) живут в пайплайне задачи 18 —
там они на своём месте, и туда же смотрят его тесты. Этот модуль их
переэкспортирует и добавляет то, чего в пайплайне нет: описание каждого режима
для окна настроек (задача 64) и предикат «нужен ли для него воркер модели»,
которым руководствуется реестр воркеров.

Держать политику в одном месте важно по двум причинам из ТЗ: в режиме «только
ИИ» пользователь остаётся без заготовленных команд — об этом надо предупредить в
интерфейсе, — а в «только команды» воркер модели не поднимается вовсе, и это
решение должно читаться из режима, а не быть зашитым в трёх местах порознь.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayris.core.pipeline import NluMode, mode_from_config

if TYPE_CHECKING:
    from ayris.core.config import Settings

__all__ = [
    "MODE_INFO",
    "ModeInfo",
    "NluMode",
    "describe_mode",
    "mode_from_config",
    "needs_llm",
]


@dataclass(frozen=True, slots=True)
class ModeInfo:
    """Как один режим называется и что о нём стоит сказать пользователю.

    ``warn`` — предупреждение, которое интерфейс показывает при выборе режима:
    для «только ИИ» это потеря заготовленных команд, для остальных пусто.
    """

    mode: NluMode
    label_ru: str
    note_ru: str
    warn_ru: str = ""

    @property
    def uses_matcher(self) -> bool:
        """Обращается ли режим к библиотеке команд."""
        return self.mode.uses_matcher

    @property
    def uses_llm(self) -> bool:
        """Может ли фраза в этом режиме дойти до модели."""
        return self.mode.uses_llm


MODE_INFO: dict[NluMode, ModeInfo] = {
    NluMode.COMMANDS: ModeInfo(
        mode=NluMode.COMMANDS,
        label_ru="Только команды",
        note_ru=(
            "Айрис выполняет только заготовленные команды и честно сообщает, "
            "когда фразу не удалось сопоставить. Модель не используется."
        ),
    ),
    NluMode.HYBRID: ModeInfo(
        mode=NluMode.HYBRID,
        label_ru="Гибрид",
        note_ru=(
            "Сначала подбирается команда из библиотеки; если ничего не совпало, "
            "фраза уходит модели."
        ),
    ),
    NluMode.AI: ModeInfo(
        mode=NluMode.AI,
        label_ru="Только ИИ",
        note_ru="Любая фраза уходит модели; ответ озвучивается.",
        warn_ru=(
            "В этом режиме заготовленные команды не срабатывают напрямую — всё " "решает модель."
        ),
    ),
}


def describe_mode(mode: NluMode) -> ModeInfo:
    """Описание режима для интерфейса настроек."""
    return MODE_INFO[mode]


def needs_llm(settings: Settings) -> bool:
    """Нужно ли для текущих настроек поднимать модель и её воркер.

    В «только команды» — нет: модель не участвует, и держать её в памяти незачем.
    В гибриде и «только ИИ» — да, даже если сейчас ни одна фраза до неё не дошла.
    """
    return mode_from_config(settings).uses_llm
