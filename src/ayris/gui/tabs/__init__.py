"""Settings window pages and their shared registration API."""

# Importing a page module runs its register_tab() call, swapping the placeholder
# in SECTIONS for the real factory. It imports the registry from its submodule
# directly, so the order relative to the re-exports below does not matter.
from ayris.gui.tabs import ai as _ai  # noqa: F401
from ayris.gui.tabs import commands as _commands  # noqa: F401
from ayris.gui.tabs import devtools as _devtools  # noqa: F401
from ayris.gui.tabs import general as _general  # noqa: F401
from ayris.gui.tabs import hotkeys as _hotkeys  # noqa: F401
from ayris.gui.tabs import overlay_settings as _overlay  # noqa: F401
from ayris.gui.tabs import privacy as _privacy  # noqa: F401
from ayris.gui.tabs import profiles as _profiles  # noqa: F401
from ayris.gui.tabs import updates as _updates  # noqa: F401
from ayris.gui.tabs import voice as _voice  # noqa: F401
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
