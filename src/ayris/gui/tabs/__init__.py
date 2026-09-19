"""Settings window pages and their shared registration API."""

# Importing a page module runs its register_tab() call, swapping the placeholder
# in SECTIONS for the real factory. It imports the registry from its submodule
# directly, so the order relative to the re-exports below does not matter.
from ayris.gui.tabs import general as _general  # noqa: F401
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
