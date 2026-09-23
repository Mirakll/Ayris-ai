"""Colour palettes for the AEGIS point-cloud sphere.

The sphere is a Jarvis-style HUD element whose look is deliberately independent
of the application theme: the specification asks for a specific cool cyan/white
default and a warm gold alternative that can be toggled at runtime.  The error
colour is the one value we still take from :class:`ThemeManager`, so an alarm
reads as an alarm in whatever palette the app is wearing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtGui import QColor


class PaletteName(StrEnum):
    CYAN = "cyan"
    GOLD = "gold"


@dataclass(frozen=True, slots=True)
class SpherePalette:
    """Two self-glowing tints plus a dim core, used as a bokeh gradient.

    ``base`` is the dominant surface tint, ``warm`` the brighter highlight that
    energetic, camera-facing particles blend toward (the soft photon look), and
    ``core`` the faint centre glow behind the cloud.
    """

    base: str
    warm: str
    core: str

    def base_color(self) -> QColor:
        return QColor(self.base)

    def warm_color(self) -> QColor:
        return QColor(self.warm)

    def core_color(self) -> QColor:
        return QColor(self.core)


_PALETTES: dict[PaletteName, SpherePalette] = {
    # Dominant cool cyan / white-blue with a soft warm-white highlight.
    PaletteName.CYAN: SpherePalette(base="#5FD8FF", warm="#EAF7FF", core="#1C8FD0"),
    # Rich amber / gold at the same density and silhouette as the cyan map.
    PaletteName.GOLD: SpherePalette(base="#FFC24D", warm="#FFF3D2", core="#C77A17"),
}


def get_palette(name: PaletteName | str) -> SpherePalette:
    """Return a built-in palette by name (defaults are documented above)."""

    return _PALETTES[PaletteName(name)]
