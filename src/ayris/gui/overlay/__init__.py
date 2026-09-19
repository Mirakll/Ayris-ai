"""Reusable dialogue-panel widgets shared by the Ayris dashboard.

These used to live in a floating overlay window; that window is gone — its parts
now sit in the right column of the single main window
(:class:`ayris.gui.main_window.MainWindow`). The widgets here — the command
field, the bounded dialogue log, the network/microphone indicators and the
active-timers panel — are all driven by the event bus and reused as plain
widgets, not as a separate window.
"""

from __future__ import annotations

from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.overlay.dialog_log import DialogEntry, DialogKind, DialogLog
from ayris.gui.overlay.indicators import MicIndicator, NetworkIndicator
from ayris.gui.overlay.timers_panel import ActiveTimer, TimerProvider, TimersPanel

__all__ = [
    "ActiveTimer",
    "CommandInput",
    "DialogEntry",
    "DialogKind",
    "DialogLog",
    "MicIndicator",
    "NetworkIndicator",
    "TimerProvider",
    "TimersPanel",
]
