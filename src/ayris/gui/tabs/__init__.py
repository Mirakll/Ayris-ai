"""Settings window pages and their shared registration API."""

from ayris.gui.tabs.base import SearchEntry, SettingsTab
from ayris.gui.tabs.placeholder import PlaceholderTab
from ayris.gui.tabs.registry import SECTIONS, TabSpec, register_tab, tab_spec

__all__ = [
    "SECTIONS",
    "PlaceholderTab",
    "SearchEntry",
    "SettingsTab",
    "TabSpec",
    "register_tab",
    "tab_spec",
]
