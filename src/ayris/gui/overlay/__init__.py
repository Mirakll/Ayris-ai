"""The single main overlay: an interactive, always-on-top assistant panel.

Ayris has one overlay. It is shown or hidden as a whole — there is no separate
compact mode and no size toggle (that is deferred to task 76). The panel carries
the state sphere, network and microphone indicators, a bounded dialogue log,
active timers and a text command field, all driven by the event bus.
"""

from __future__ import annotations

from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.overlay.dialog_log import DialogEntry, DialogKind, DialogLog
from ayris.gui.overlay.indicators import MicIndicator, NetworkIndicator
from ayris.gui.overlay.main import MainOverlay, OverlayController
from ayris.gui.overlay.placement import Geometry, Placement, Position, place, snap_position
from ayris.gui.overlay.timers_panel import ActiveTimer, TimerProvider, TimersPanel
from ayris.gui.overlay.window_flags import WindowFlags, overlay_qt_flags

__all__ = [
    "ActiveTimer",
    "CommandInput",
    "DialogEntry",
    "DialogKind",
    "DialogLog",
    "Geometry",
    "MainOverlay",
    "MicIndicator",
    "NetworkIndicator",
    "OverlayController",
    "Placement",
    "Position",
    "TimerProvider",
    "TimersPanel",
    "WindowFlags",
    "overlay_qt_flags",
    "place",
    "snap_position",
]
