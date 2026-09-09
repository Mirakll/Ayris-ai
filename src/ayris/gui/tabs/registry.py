"""Ordered registry of settings pages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from PySide6.QtWidgets import QStyle

if TYPE_CHECKING:
    from ayris.core.config import ConfigManager
    from ayris.core.events import EventBus
    from ayris.gui.tabs.base import SettingsTab
    from ayris.gui.theme import ThemeManager

__all__ = ["SECTIONS", "TabSpec", "register_tab", "tab_spec"]

TabFactory = Callable[["ConfigManager", "ThemeManager", "EventBus | None"], "SettingsTab"]


@dataclass(frozen=True, slots=True)
class TabSpec:
    key: str
    title: str
    icon: QStyle.StandardPixmap
    task: int
    config_paths: tuple[str, ...]
    factory: TabFactory | None = None


SECTIONS: Final[tuple[TabSpec, ...]] = (
    TabSpec(
        "general",
        "Общие",
        QStyle.StandardPixmap.SP_ComputerIcon,
        48,
        ("general", "performance"),
    ),
    TabSpec("voice", "Голос", QStyle.StandardPixmap.SP_MediaVolume, 49, ("voice",)),
    TabSpec(
        "commands",
        "Команды",
        QStyle.StandardPixmap.SP_FileDialogDetailedView,
        51,
        ("commands", "actions"),
    ),
    TabSpec("ai", "ИИ / LLM", QStyle.StandardPixmap.SP_MessageBoxInformation, 64, ("ai",)),
    TabSpec(
        "hotkeys",
        "Горячие клавиши",
        QStyle.StandardPixmap.SP_CommandLink,
        55,
        ("hotkeys",),
    ),
    TabSpec("overlay", "Оверлей", QStyle.StandardPixmap.SP_DesktopIcon, 56, ("overlay",)),
    TabSpec("plugins", "Плагины", QStyle.StandardPixmap.SP_DriveNetIcon, 67, ("plugins",)),
    TabSpec("profiles", "Профили", QStyle.StandardPixmap.SP_DirHomeIcon, 57, ()),
    TabSpec(
        "updates",
        "Обновления",
        QStyle.StandardPixmap.SP_BrowserReload,
        50,
        ("updates",),
    ),
    TabSpec(
        "devtools",
        "Логи / DevTools",
        QStyle.StandardPixmap.SP_FileDialogInfoView,
        58,
        ("devtools",),
    ),
    TabSpec("privacy", "Приватность", QStyle.StandardPixmap.SP_DialogNoButton, 59, ("privacy",)),
)

_registry = {spec.key: spec for spec in SECTIONS}


def register_tab(key: str, factory: TabFactory) -> None:
    """Replace a placeholder with a real page using one registration call."""
    spec = tab_spec(key)
    _registry[key] = TabSpec(spec.key, spec.title, spec.icon, spec.task, spec.config_paths, factory)


def tab_spec(key: str) -> TabSpec:
    try:
        return _registry[key]
    except KeyError as exc:
        raise KeyError(f"Неизвестный раздел настроек: {key}") from exc
