"""Which displays are attached, and what «второй монитор» refers to.

Two things live here, and they are together on purpose:

* :func:`list_monitors` — every attached display with its geometry in
  virtual-desktop coordinates, its DPI and its human name. The one enumeration
  the whole application uses, so the overlay, the screenshots and the brightness
  actions cannot disagree about where the second monitor is.
* :func:`resolve_monitor` — one parser for the address a user or a macro names a
  display by: ``primary``, ``external_1``, a number, or part of the model name.
  ``SetBrightness(monitor="external_1")`` and a screenshot of the same monitor
  must mean the same screen, and they only do while there is a single resolver.

Coordinates are physical pixels in the virtual desktop, exactly as Windows
reports them, and the origin is **not** the top-left corner: a monitor placed to
the left of the primary one has a negative ``left``, and one placed above it a
negative ``top``. Nothing here normalises that away — the capture layer needs the
real numbers to hand to ``mss``, and the overlay needs them to cover the whole
desktop. What callers do get is :attr:`MonitorInfo.index`, an ordering that runs
left to right regardless of which screen Windows happens to enumerate first.

This module sits in :mod:`ayris.utils`, below :mod:`ayris.core`, so a bad address
raises the plain :class:`MonitorNotFound`; the action layer turns it into a
Russian :class:`ayris.core.errors.ActionError`.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "DEFAULT_DPI",
    "MonitorInfo",
    "MonitorNotFound",
    "list_monitors",
    "monitor_for_point",
    "monitor_for_window",
    "order_monitors",
    "resolve_monitor",
    "virtual_bounds",
]

_log = get_logger(__name__)

#: DPI of an unscaled display. Everything else is a multiple of it.
DEFAULT_DPI: Final = 96

#: Addresses that mean «the screen the taskbar and the Start menu are on».
_PRIMARY_WORDS: Final = frozenset(
    {"primary", "main", "основной", "основный", "главный", "первичный"}
)
#: Addresses that mean «the screen the mouse is on right now».
_CURRENT_WORDS: Final = frozenset(
    {"current", "active", "cursor", "текущий", "активный", "под курсором"}
)
#: ``external_2``, ``external 2``, ``внешний-2``. Group 1 is the ordinal.
_EXTERNAL_RE: Final = re.compile(r"^(?:external|внешний)[\s_-]*(\d*)$")
#: ``monitor_2``, ``экран 2``, or a bare ``2``.
_INDEXED_RE: Final = re.compile(r"^(?:monitor|display|screen|монитор|экран|дисплей)?[\s_-]*(\d+)$")


class MonitorNotFound(LookupError):  # noqa: N818 - под стать WindowNotFound из actions
    """No attached display answers to the address that was asked for.

    Carries the address and the list that was searched so the action layer can
    say «Монитор „Dell“ не найден: подключены DISPLAY1, DISPLAY2» rather than
    just failing.
    """

    def __init__(self, address: str, available: Sequence[MonitorInfo] = ()) -> None:
        self.address = address
        self.available = tuple(available)
        names = ", ".join(monitor.label for monitor in self.available) or "none"
        super().__init__(f"no monitor matches {address!r}; attached: {names}")


@dataclass(frozen=True, slots=True)
class MonitorInfo:
    """One attached display.

    ``rect`` is the whole panel, ``work`` the part left over once the taskbar has
    taken its strip — a maximised window fills ``work``, a full-screen screenshot
    covers ``rect``. Both are in virtual-desktop physical pixels and may start at
    a negative coordinate.

    ``index`` orders the displays left to right, so ``index == 1`` is the screen
    the user calls «второй». That is *not* the order ``EnumDisplayMonitors``
    reports, which follows the graphics adapters.
    """

    handle: int
    index: int
    rect: winapi.Rect
    work: winapi.Rect
    device: str = ""
    name: str = ""
    device_id: str = ""
    dpi: int = DEFAULT_DPI
    primary: bool = False
    #: Position among the non-primary displays, ``-1`` for the primary one.
    #: Filled in by :func:`order_monitors`, which every entry point runs through.
    external_index: int = -1

    @property
    def width(self) -> int:
        return self.rect.width

    @property
    def height(self) -> int:
        return self.rect.height

    @property
    def resolution(self) -> tuple[int, int]:
        """Panel size in physical pixels — what a screenshot of it will be."""
        return (self.rect.width, self.rect.height)

    @property
    def scale(self) -> float:
        """Windows scaling factor: 1.0 at 96 DPI, 1.5 at 144, 2.0 at 192."""
        return self.dpi / DEFAULT_DPI

    @property
    def label(self) -> str:
        """Shortest string that still identifies the display to a human."""
        return self.name or self.device or f"#{self.index + 1}"

    @property
    def address(self) -> str:
        """The canonical address :func:`resolve_monitor` accepts for this display.

        ``primary`` for the main screen, ``external_1``… for the rest, numbered
        left to right among the non-primary displays.
        """
        if self.primary:
            return "primary"
        return f"external_{max(self.external_index, 0) + 1}"

    @property
    def title_ru(self) -> str:
        """How the display is named out loud: «Основной — 1920×1080 (150%)»."""
        role = "Основной" if self.primary else self.label
        size = f"{self.rect.width}×{self.rect.height}"
        if self.dpi == DEFAULT_DPI:
            return f"{role} — {size}"
        return f"{role} — {size} ({round(self.scale * 100)}%)"

    def contains(self, x: int, y: int) -> bool:
        """Whether a virtual-desktop point falls on this display."""
        return self.rect.left <= x < self.rect.right and self.rect.top <= y < self.rect.bottom

    def as_dict(self) -> dict[str, object]:
        """Flat form for :class:`ayris.actions.result.ActionResult` payloads."""
        return {
            "index": self.index,
            "address": self.address,
            "name": self.label,
            "device": self.device,
            "device_id": self.device_id,
            "primary": self.primary,
            "dpi": self.dpi,
            "scale": round(self.scale, 4),
            "rect": self.rect.as_dict(),
            "work": self.work.as_dict(),
            "width": self.rect.width,
            "height": self.rect.height,
        }


def order_monitors(monitors: Iterable[MonitorInfo]) -> list[MonitorInfo]:
    """Left to right, top to bottom, with the two orderings renumbered.

    Split out from :func:`list_monitors` so tests can build a layout by hand —
    primary on the right, mixed DPI, negative coordinates — and get the same
    :attr:`MonitorInfo.index` and :attr:`MonitorInfo.external_index` the live
    enumeration would produce.
    """
    ordered = sorted(monitors, key=lambda item: (item.rect.left, item.rect.top))
    result: list[MonitorInfo] = []
    external = 0
    for position, monitor in enumerate(ordered):
        place = -1
        if not monitor.primary:
            place = external
            external += 1
        if monitor.index == position and monitor.external_index == place:
            result.append(monitor)
        else:
            result.append(_placed(monitor, position, place))
    return result


def _placed(monitor: MonitorInfo, index: int, external_index: int) -> MonitorInfo:
    """Copy of the display sitting at a different place in the ordering."""
    return MonitorInfo(
        handle=monitor.handle,
        index=index,
        rect=monitor.rect,
        work=monitor.work,
        device=monitor.device,
        name=monitor.name,
        device_id=monitor.device_id,
        dpi=monitor.dpi,
        primary=monitor.primary,
        external_index=external_index,
    )


def list_monitors() -> list[MonitorInfo]:
    """Every attached display, ordered left to right.

    Returns an empty list off Windows and when the enumeration fails — a caller
    that needs a display has to say so itself, in Russian, and there is nothing
    useful to raise from a utility layer. The names come from
    ``EnumDisplayDevicesW`` and are best-effort: some panels report only
    «Generic PnP Monitor», which is why the numeric addresses exist.
    """
    if sys.platform != "win32":
        return []
    try:
        handles = winapi.enum_display_monitors()
    except winapi.WinApiError:
        _log.warning("EnumDisplayMonitors failed; treating the display list as empty")
        return []

    collected: list[MonitorInfo] = []
    for handle in handles:
        try:
            info = winapi.monitor_info(handle)
        except winapi.WinApiError:
            _log.debug("skipping monitor %#x: GetMonitorInfoW failed", handle)
            continue
        name, device_id = winapi.display_device(info.device)
        collected.append(
            MonitorInfo(
                handle=handle,
                index=len(collected),
                rect=info.rect,
                work=info.work,
                device=info.device,
                name=name,
                device_id=device_id,
                dpi=winapi.dpi_for_monitor(handle),
                primary=info.primary,
            )
        )
    return order_monitors(collected)


def virtual_bounds(monitors: Sequence[MonitorInfo] | None = None) -> winapi.Rect:
    """Bounding box of every display together.

    Taken from ``GetSystemMetrics`` when no list is supplied, because that is the
    authority ``mss`` and ``SendInput`` agree with; computed from the union of the
    rectangles otherwise, which is what tests need. The two answer the same thing
    on a real desktop.
    """
    if monitors is None:
        if sys.platform != "win32":
            return winapi.Rect()
        try:
            return winapi.virtual_screen_rect()
        except winapi.WinApiError:
            _log.debug("GetSystemMetrics(SM_*VIRTUALSCREEN) failed; unioning the monitors")
            monitors = list_monitors()
    if not monitors:
        return winapi.Rect()
    return winapi.Rect(
        left=min(item.rect.left for item in monitors),
        top=min(item.rect.top for item in monitors),
        right=max(item.rect.right for item in monitors),
        bottom=max(item.rect.bottom for item in monitors),
    )


def monitor_for_point(
    x: int,
    y: int,
    monitors: Sequence[MonitorInfo] | None = None,
) -> MonitorInfo | None:
    """Display a virtual-desktop point falls on, or the nearest one.

    «Nearest» matters because the point may sit in the L-shaped gap two monitors
    of different heights leave between them, and a screenshot still has to pick a
    screen.
    """
    available = list_monitors() if monitors is None else list(monitors)
    if not available:
        return None
    for monitor in available:
        if monitor.contains(x, y):
            return monitor
    return min(available, key=lambda item: _distance_squared(item.rect, x, y))


def _distance_squared(rect: winapi.Rect, x: int, y: int) -> int:
    """Squared distance from a point to a rectangle; ``0`` when inside it."""
    dx = max(rect.left - x, 0, x - (rect.right - 1))
    dy = max(rect.top - y, 0, y - (rect.bottom - 1))
    return dx * dx + dy * dy


def monitor_for_window(
    hwnd: int,
    monitors: Sequence[MonitorInfo] | None = None,
) -> MonitorInfo | None:
    """Display the window is mostly on, matched back into :func:`list_monitors`.

    Goes through ``MonitorFromWindow`` so the answer agrees with the one Windows
    would give, and falls back to the centre of the window's frame when the
    handle is already gone.
    """
    available = list_monitors() if monitors is None else list(monitors)
    if not available:
        return None
    if sys.platform == "win32":
        try:
            handle = winapi.monitor_from_window(hwnd)
        except winapi.WinApiError:
            handle = 0
        if handle:
            for monitor in available:
                if monitor.handle == handle:
                    return monitor
    try:
        frame = winapi.extended_frame_bounds(hwnd)
    except winapi.WinApiError:
        return next((item for item in available if item.primary), available[0])
    return monitor_for_point(
        frame.left + frame.width // 2,
        frame.top + frame.height // 2,
        available,
    )


def resolve_monitor(
    address: str | int | None,
    monitors: Sequence[MonitorInfo] | None = None,
) -> MonitorInfo:
    """Turn a spoken or configured display address into one display.

    Understood forms, in the order they are tried:

    ``None`` / empty
        The primary display — what «сделай скриншот монитора» means when the
        user names no monitor.
    ``primary``, ``основной``, ``главный``
        The display Windows flags as primary.
    ``current``, ``текущий``, ``под курсором``
        Whichever display the mouse pointer is on.
    ``external_1``…``external_N``, ``внешний 2``
        The non-primary displays, numbered left to right from one. A bare
        ``external`` means the first of them.
    ``2``, ``монитор 2``, ``screen 3``
        Every display, numbered left to right from one, primary included.
    a substring of the name
        Matched case-insensitively against the model name, the ``\\\\.\\DISPLAY``
        device and the device id. A single hit wins; several hits are ambiguous
        and the leftmost one is taken, with a line in the log.

    Raises:
        MonitorNotFound: nothing matched, or no display is attached at all.
    """
    available = list_monitors() if monitors is None else order_monitors(monitors)
    text = str(address).strip().lower() if address is not None else ""
    if not available:
        raise MonitorNotFound(text or "primary", available)

    primary = next((item for item in available if item.primary), available[0])
    if not text or text in _PRIMARY_WORDS:
        return primary

    if text in _CURRENT_WORDS:
        return _monitor_under_cursor(available) or primary

    external = [item for item in available if not item.primary]
    match = _EXTERNAL_RE.match(text)
    if match is not None:
        if not external:
            raise MonitorNotFound(text, available)
        ordinal = int(match.group(1) or 1)
        if not 1 <= ordinal <= len(external):
            raise MonitorNotFound(text, available)
        return external[ordinal - 1]

    match = _INDEXED_RE.match(text)
    if match is not None:
        ordinal = int(match.group(1))
        if not 1 <= ordinal <= len(available):
            raise MonitorNotFound(text, available)
        return available[ordinal - 1]

    hits = [item for item in available if _name_matches(item, text)]
    if not hits:
        raise MonitorNotFound(text, available)
    if len(hits) > 1:
        _log.debug("address %r matches %d monitors; taking the leftmost", text, len(hits))
    return hits[0]


def _name_matches(monitor: MonitorInfo, text: str) -> bool:
    """Whether a lower-cased fragment appears in any name the display has."""
    return (
        text in monitor.name.lower()
        or text in monitor.device.lower()
        or text in monitor.device_id.lower()
    )


def _monitor_under_cursor(monitors: Sequence[MonitorInfo]) -> MonitorInfo | None:
    """Display the mouse pointer is on, or ``None`` when it cannot be asked."""
    if sys.platform != "win32":
        return None
    try:
        x, y = winapi.cursor_position()
    except winapi.WinApiError:
        return None
    return monitor_for_point(x, y, monitors)
