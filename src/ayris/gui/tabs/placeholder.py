"""Lazy placeholder used until a settings page is implemented."""

from __future__ import annotations

from ayris.core.config import ConfigManager
from ayris.core.events import EventBus
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import TabSpec
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import EmptyState

__all__ = ["PlaceholderTab"]


class PlaceholderTab(SettingsTab):
    def __init__(
        self,
        spec: TabSpec,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
    ) -> None:
        super().__init__(spec.key, spec.title, spec.config_paths, manager, theme, bus)
        self.reset_button.hide()
        self.body.addWidget(
            EmptyState(
                spec.title,
                f"Раздел появится в задаче {spec.task}",
                theme,
                icon=self.style().standardIcon(spec.icon),
            )
        )
