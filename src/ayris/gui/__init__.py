"""Qt user interface: settings window, overlay, tray icon, widgets.

Runs exclusively on the main thread. No blocking work here — anything heavy goes
to a worker process or a ``QThread``.
"""

from __future__ import annotations
