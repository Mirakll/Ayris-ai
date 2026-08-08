"""Cross-cutting helpers: logging, DPI, monitors, hotkeys, elevation.

Utilities must not import from :mod:`ayris.core` or :mod:`ayris.gui` — they sit
at the bottom of the dependency graph so any layer can use them.
"""

from __future__ import annotations
