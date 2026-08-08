"""Application core: lifecycle, configuration, storage, event bus, profiles.

Owns process-wide state. Nothing in this package may import from
:mod:`ayris.gui` — the dependency direction is GUI -> core, never the reverse.
"""

from __future__ import annotations
