"""The single Ayris dashboard: one frameless window for everything.

The old settings window and the floating overlay are merged here. The window
has two columns — a left «showcase» (logo, state sphere, caption) and a right
«dialogue» (top bar, log, status line, timers, command input) — and settings
slide down as a full-window layer instead of opening a second window.

Assembly point stays :class:`ayris.gui.main_window.MainWindow`; this package
holds the pieces it composes.
"""

from ayris.gui.dashboard.dialog_view import DialogView
from ayris.gui.dashboard.input_bar import InputBar
from ayris.gui.dashboard.settings_layer import SettingsLayer
from ayris.gui.dashboard.showcase import ShowcasePanel
from ayris.gui.dashboard.sphere_host import make_sphere

__all__ = [
    "DialogView",
    "InputBar",
    "SettingsLayer",
    "ShowcasePanel",
    "make_sphere",
]
