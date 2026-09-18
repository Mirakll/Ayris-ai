"""Pure placement logic for the overlay, tested on fabricated monitors.

No Qt and no live desktop: :func:`place`, :func:`logical_offset` and
:func:`snap_position` are functions of monitors, sizes and settings, so DPI
recomputation and the disconnected-monitor fallback are checked with plain data.
"""

from __future__ import annotations

import pytest

from ayris.gui.overlay.placement import (
    DEFAULT_MARGIN,
    Geometry,
    logical_offset,
    place,
    snap_position,
)
from ayris.utils import winapi
from ayris.utils.monitors import MonitorInfo

pytestmark = pytest.mark.unit


def _monitor(
    *,
    index: int,
    left: int,
    top: int,
    width: int,
    height: int,
    dpi: int = 96,
    primary: bool = False,
) -> MonitorInfo:
    rect = winapi.Rect(left, top, left + width, top + height)
    return MonitorInfo(
        handle=1000 + index,
        index=index,
        rect=rect,
        work=rect,
        name=f"DISPLAY{index}",
        dpi=dpi,
        primary=primary,
    )


def _primary() -> MonitorInfo:
    return _monitor(index=0, left=0, top=0, width=1920, height=1080, primary=True)


def _second() -> MonitorInfo:
    return _monitor(index=1, left=1920, top=0, width=2560, height=1440, dpi=144)


def test_corner_presets_sit_inside_work_area_with_margin() -> None:
    monitors = [_primary()]
    top_left = place(monitors, address=None, logical_size=(360, 420), position="top_left").geometry
    assert top_left.x == DEFAULT_MARGIN
    assert top_left.y == DEFAULT_MARGIN

    bottom_right = place(
        monitors, address=None, logical_size=(360, 420), position="bottom_right"
    ).geometry
    assert bottom_right.right == 1920 - DEFAULT_MARGIN
    assert bottom_right.bottom == 1080 - DEFAULT_MARGIN


def test_center_presets_are_horizontally_centred() -> None:
    monitors = [_primary()]
    top_center = place(
        monitors, address=None, logical_size=(360, 420), position="top_center"
    ).geometry
    assert top_center.x == (1920 - 360) // 2
    assert top_center.y == DEFAULT_MARGIN


def test_custom_offset_is_scaled_by_display_dpi() -> None:
    # Same logical offset lands at twice the physical distance on a 2.0 display.
    one_x = _monitor(index=0, left=0, top=0, width=1920, height=1080, dpi=96, primary=True)
    two_x = _monitor(index=0, left=0, top=0, width=3840, height=2160, dpi=192, primary=True)
    at_1x = place(
        [one_x], address=None, logical_size=(360, 420), position="custom", custom_logical=(100, 50)
    ).geometry
    at_2x = place(
        [two_x], address=None, logical_size=(360, 420), position="custom", custom_logical=(100, 50)
    ).geometry
    assert (at_1x.x, at_1x.y) == (100, 50)
    assert (at_2x.x, at_2x.y) == (200, 100)
    # The panel itself is scaled too, so it keeps its apparent size.
    assert (at_2x.width, at_2x.height) == (720, 840)


def test_panel_never_leaves_the_work_area() -> None:
    monitors = [_primary()]
    # A custom offset that would push the panel off-screen is clamped back in.
    placed = place(
        monitors,
        address=None,
        logical_size=(360, 420),
        position="custom",
        custom_logical=(5000, 5000),
    ).geometry
    assert placed.right <= 1920
    assert placed.bottom <= 1080
    assert placed.x >= 0
    assert placed.y >= 0


def test_oversized_panel_is_capped_to_the_work_area() -> None:
    small = _monitor(index=0, left=0, top=0, width=300, height=300, primary=True)
    placed = place([small], address=None, logical_size=(360, 420), position="top_left").geometry
    assert placed.width <= 300
    assert placed.height <= 300


def test_missing_monitor_falls_back_to_primary() -> None:
    monitors = [_primary()]  # the second display was unplugged
    result = place(monitors, address=2, logical_size=(360, 420), position="top_left")
    assert result.fell_back is True
    assert result.monitor.primary is True


def test_known_monitor_does_not_report_fallback() -> None:
    result = place([_primary()], address=None, logical_size=(360, 420), position="top_left")
    assert result.fell_back is False


def test_logical_offset_inverts_custom_placement() -> None:
    monitor = _second()  # 144 DPI, origin at x=1920
    geometry = place(
        [_primary(), monitor],
        address=2,
        logical_size=(360, 420),
        position="custom",
        custom_logical=(120, 80),
    ).geometry
    # Round-trip: physical geometry back to the logical offset we would persist.
    assert logical_offset(monitor, geometry) == (120, 80)


def test_snap_classifies_edges_and_leaves_the_middle_custom() -> None:
    monitor = _primary()
    corner = Geometry(x=2, y=3, width=360, height=420)
    assert snap_position(monitor, corner) == "top_left"
    centre_bottom = Geometry(x=(1920 - 360) // 2, y=1080 - 420 - 2, width=360, height=420)
    assert snap_position(monitor, centre_bottom) == "bottom_center"
    floating = Geometry(x=800, y=400, width=360, height=420)
    assert snap_position(monitor, floating) == "custom"
