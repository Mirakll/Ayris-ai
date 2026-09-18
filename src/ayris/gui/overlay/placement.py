"""Where the overlay panel sits, as pure arithmetic over monitors and settings.

The panel remembers a position *relative to one display* — a preset corner or a
logical offset from the display's work-area origin — never a raw pixel on the
virtual desktop. That is what lets it come back to the same visual spot after a
DPI change or a resolution change, and fall back to the primary display when the
one it was on is unplugged.

Everything here is a plain function of numbers and :class:`MonitorInfo` values,
so the placement logic is tested on fabricated screens without a live desktop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from ayris.utils import winapi
from ayris.utils.monitors import MonitorInfo, MonitorNotFound, resolve_monitor

__all__ = [
    "DEFAULT_MARGIN",
    "SNAP_THRESHOLD",
    "Geometry",
    "Placement",
    "Position",
    "logical_offset",
    "place",
    "snap_position",
]

Position = Literal[
    "top_left",
    "top_right",
    "bottom_left",
    "bottom_right",
    "top_center",
    "bottom_center",
    "custom",
]

#: Gap between the panel and the work-area edge for the corner and centre
#: presets, in logical pixels (multiplied by the display scale when placing).
DEFAULT_MARGIN: Final = 16

#: How close, in physical pixels, a dragged panel must come to an edge before it
#: snaps to the matching preset.
SNAP_THRESHOLD: Final = 24


@dataclass(frozen=True, slots=True)
class Geometry:
    """A window rectangle in physical (device) pixels on the virtual desktop."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True, slots=True)
class Placement:
    """The result of placing the panel: geometry plus the display it landed on."""

    geometry: Geometry
    monitor: MonitorInfo
    position: Position
    fell_back: bool


def _clamp(value: int, low: int, high: int) -> int:
    if high < low:
        return low
    return max(low, min(high, value))


def _fit_size(width: int, height: int, work: winapi.Rect) -> tuple[int, int]:
    """Never let the panel be larger than the work area; content scrolls."""
    return min(width, work.width), min(height, work.height)


def place(
    monitors: list[MonitorInfo],
    *,
    address: str | int | None,
    logical_size: tuple[int, int],
    position: Position,
    custom_logical: tuple[int, int] = (0, 0),
    margin: int = DEFAULT_MARGIN,
) -> Placement:
    """Resolve the target display and return the panel's physical geometry.

    ``logical_size`` and ``custom_logical`` are in device-independent pixels; they
    are multiplied by the resolved display's scale so the panel keeps the same
    apparent size and offset on a 96-DPI screen and a 192-DPI one. When the stored
    display is gone, the panel falls back to the primary display.
    """
    monitor, fell_back = _resolve(monitors, address)
    scale = monitor.scale
    width = round(logical_size[0] * scale)
    height = round(logical_size[1] * scale)
    width, height = _fit_size(width, height, monitor.work)
    scaled_margin = round(margin * scale)
    custom_physical = (round(custom_logical[0] * scale), round(custom_logical[1] * scale))
    x, y = _place_anchor(position, monitor.work, (width, height), scaled_margin, custom_physical)
    x = _clamp(x, monitor.work.left, max(monitor.work.left, monitor.work.right - width))
    y = _clamp(y, monitor.work.top, max(monitor.work.top, monitor.work.bottom - height))
    return Placement(Geometry(x, y, width, height), monitor, position, fell_back)


def _place_anchor(
    position: Position,
    work: winapi.Rect,
    size: tuple[int, int],
    margin: int,
    custom_physical: tuple[int, int],
) -> tuple[int, int]:
    width, height = size
    left = work.left + margin
    right = work.right - width - margin
    top = work.top + margin
    bottom = work.bottom - height - margin
    center_x = work.left + (work.width - width) // 2
    anchors: dict[str, tuple[int, int]] = {
        "top_left": (left, top),
        "top_right": (right, top),
        "bottom_left": (left, bottom),
        "bottom_right": (right, bottom),
        "top_center": (center_x, top),
        "bottom_center": (center_x, bottom),
    }
    if position == "custom":
        return work.left + custom_physical[0], work.top + custom_physical[1]
    return anchors[position]


def _resolve(monitors: list[MonitorInfo], address: str | int | None) -> tuple[MonitorInfo, bool]:
    if not monitors:
        raise MonitorNotFound(str(address))
    try:
        return resolve_monitor(address, monitors), False
    except MonitorNotFound:
        for monitor in monitors:
            if monitor.primary:
                return monitor, True
        return monitors[0], True


def logical_offset(monitor: MonitorInfo, geometry: Geometry) -> tuple[int, int]:
    """Physical geometry → the logical offset to persist for «custom».

    The inverse of what :func:`place` does for a custom position: divide the
    physical distance from the work-area origin by the display scale, so the value
    written to the config is DPI-independent.
    """
    scale = monitor.scale or 1.0
    dx = round((geometry.x - monitor.work.left) / scale)
    dy = round((geometry.y - monitor.work.top) / scale)
    return max(0, dx), max(0, dy)


def snap_position(
    monitor: MonitorInfo,
    geometry: Geometry,
    *,
    threshold: int = SNAP_THRESHOLD,
) -> Position:
    """Classify a freely-dragged rectangle into a preset when it hugs an edge.

    Returns ``"custom"`` when the panel is not near enough to any edge to snap.
    """
    work = monitor.work
    near_left = abs(geometry.x - work.left) <= threshold
    near_right = abs(geometry.right - work.right) <= threshold
    near_top = abs(geometry.y - work.top) <= threshold
    near_bottom = abs(geometry.bottom - work.bottom) <= threshold
    center_x = work.left + (work.width - geometry.width) // 2
    near_center_x = abs(geometry.x - center_x) <= threshold
    if near_top and near_left:
        return "top_left"
    if near_top and near_right:
        return "top_right"
    if near_bottom and near_left:
        return "bottom_left"
    if near_bottom and near_right:
        return "bottom_right"
    if near_top and near_center_x:
        return "top_center"
    if near_bottom and near_center_x:
        return "bottom_center"
    return "custom"
