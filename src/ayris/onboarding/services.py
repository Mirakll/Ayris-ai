"""Сервисы мастера первого запуска — всё, что нужно его шагам, одним объектом.

Мастер и его фабрика не лезут в приложение напрямую: они получают
:class:`WizardServices`, а тесты собирают его из заглушек (фейковый бэкенд
загрузок, подставной импортёр, пробник микрофона без звуковой карты). Все поля,
кроме темы и конфига, необязательны — любой недоступный сервис вырождает свой шаг
в «пропустить», а не роняет мастер.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayris.core.config import ConfigManager
from ayris.core.events import EventBus
from ayris.gui.theme import ThemeManager

if TYPE_CHECKING:
    from ayris.gui.widgets.model_manager import ModelManagerBackend
    from ayris.onboarding.steps.audio import AudioProbe
    from ayris.onboarding.steps.profile import ProfileImporter

__all__ = ["WizardServices"]


@dataclass(frozen=True, slots=True)
class WizardServices:
    """Зависимости шагов мастера, внедряемые извне."""

    theme: ThemeManager
    config: ConfigManager
    bus: EventBus | None = None
    backend: ModelManagerBackend | None = None
    importer: ProfileImporter | None = None
    audio_probe: AudioProbe | None = None
    submit_text: Callable[[str], None] | None = None
