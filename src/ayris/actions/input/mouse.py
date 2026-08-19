"""Pointer and buttons, and the coordinate arithmetic nobody expects to need.

``SendInput`` will not take pixels. An absolute mouse move is expressed as a
fraction of the virtual desktop scaled to ``0..65535``, and the virtual desktop is
the bounding box of every monitor — which starts at a negative x when a second
display sits to the left of the primary one. Skip that offset and every click on
the left-hand monitor lands somewhere on the right-hand one. :func:`normalize_point`
is the whole of that conversion, and it lives above the backend precisely so a
recording fake can prove it.

The second surprise is DPI. On a mixed-scaling setup — a 150 % laptop panel beside
a 100 % external monitor — a window reports its bounds in physical pixels, but a
coordinate a person reads off a design or a screenshot may be logical. The two
differ by a factor of 1.5 on one monitor and not at all on the other, so
:class:`Origin` makes the frame of reference explicit rather than guessing:
coordinates are relative to the whole desktop, to one monitor, or to the active
window, and :attr:`MousePoint.logical` says which scale they are in.

Everything here reads the display layout through :class:`ScreenBackend`, a seam
with exactly one real implementation, so the tests can describe a two-monitor
mixed-DPI desktop without owning one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.input.backend import ABSOLUTE_MAX, MouseButton, get_input_backend
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.input.backend import InputBackend

__all__ = [
    "MouseClick",
    "MouseDrag",
    "MouseMove",
    "MousePoint",
    "MouseWheel",
    "Origin",
    "ScreenBackend",
    "ScreenLayout",
    "WinApiScreen",
    "drag_path",
    "get_screen_backend",
    "normalize_point",
    "set_screen_backend",
]

_log = get_logger(__name__)

#: DPI that means "no scaling". Windows counts scale as a fraction of this.
BASE_DPI: Final = 96


class Origin(StrEnum):
    """What a coordinate pair is measured from."""

    DESKTOP = "desktop"
    MONITOR = "monitor"
    WINDOW = "window"
    CURSOR = "cursor"

    @property
    def title_ru(self) -> str:
        return _ORIGIN_TITLES[self]


_ORIGIN_TITLES: Final[dict[Origin, str]] = {
    Origin.DESKTOP: "от всего рабочего стола",
    Origin.MONITOR: "от угла монитора",
    Origin.WINDOW: "от угла активного окна",
    Origin.CURSOR: "от текущего положения курсора",
}


@dataclass(frozen=True, slots=True)
class ScreenLayout:
    """The monitors, as one snapshot.

    Taken once per action rather than per event: a drag makes dozens of moves, and
    re-enumerating displays for each of them would be both slow and inconsistent
    if the user unplugged something halfway.
    """

    virtual: winapi.Rect
    monitors: tuple[winapi.MonitorInfo, ...] = ()
    dpi: tuple[int, ...] = ()

    def primary(self) -> winapi.MonitorInfo | None:
        for monitor in self.monitors:
            if monitor.primary:
                return monitor
        return self.monitors[0] if self.monitors else None

    def by_index(self, index: int) -> winapi.MonitorInfo:
        """Monitor by its 1-based number, as the user counts them.

        Raises:
            ActionError: no monitor with that number.
        """
        if index < 1 or index > len(self.monitors):
            raise ActionError(
                f"monitor {index} of {len(self.monitors)} does not exist",
                user_message=(
                    f"Монитора №{index} нет — подключено {len(self.monitors)}."
                    if self.monitors
                    else "Не вижу ни одного монитора."
                ),
            )
        return self.monitors[index - 1]

    def dpi_of(self, monitor: winapi.MonitorInfo) -> int:
        """Effective DPI of one monitor, ``96`` when unknown."""
        for candidate, dpi in zip(self.monitors, self.dpi, strict=False):
            if candidate.handle == monitor.handle:
                return dpi or BASE_DPI
        return BASE_DPI

    def containing(self, x: int, y: int) -> winapi.MonitorInfo | None:
        """The monitor a physical point falls on."""
        for monitor in self.monitors:
            rect = monitor.rect
            if rect.left <= x < rect.right and rect.top <= y < rect.bottom:
                return monitor
        return None


class ScreenBackend(Protocol):
    """Everything the mouse actions need to know about displays and windows."""

    def layout(self) -> ScreenLayout:
        """Snapshot of the monitors and their scaling."""
        ...

    def cursor(self) -> tuple[int, int]:
        """Current pointer position in physical virtual-desktop pixels."""
        ...

    def active_window(self) -> winapi.Rect | None:
        """Bounds of the foreground window, or ``None`` when there is none."""
        ...


class WinApiScreen:
    """The real one, straight over :mod:`ayris.utils.winapi`."""

    def layout(self) -> ScreenLayout:
        handles = winapi.enum_display_monitors()
        monitors = tuple(winapi.monitor_info(handle) for handle in handles)
        dpi = tuple(winapi.dpi_for_monitor(handle) for handle in handles)
        return ScreenLayout(virtual=winapi.virtual_screen_rect(), monitors=monitors, dpi=dpi)

    def cursor(self) -> tuple[int, int]:
        return winapi.cursor_position()

    def active_window(self) -> winapi.Rect | None:
        hwnd = winapi.foreground_window()
        if not hwnd:
            return None
        rect = winapi.window_rect(hwnd)
        return None if rect.is_empty else rect


_screen: ScreenBackend | None = None


def get_screen_backend() -> ScreenBackend:
    """The screen backend in force, creating the real one on first use."""
    global _screen
    if _screen is None:
        _screen = WinApiScreen()
    return _screen


def set_screen_backend(backend: ScreenBackend | None) -> None:
    """Install a screen backend, or restore the real one with ``None``. Test seam."""
    global _screen
    _screen = backend


def normalize_point(x: int, y: int, virtual: winapi.Rect) -> tuple[int, int]:
    """Physical pixels into the ``0..65535`` scale ``SendInput`` wants.

    Dividing by ``span - 1`` rather than by ``span`` is not a rounding fudge.
    Windows maps the normalised range across the desktop *inclusively*: the last
    column of pixels has to reach 65535, and dividing by the width alone leaves it
    one step short, which puts a click on the far right edge one pixel inside the
    wrong monitor.

    ``virtual.left`` and ``virtual.top`` are subtracted first, and they are
    negative whenever a monitor sits above or to the left of the primary one.

    Raises:
        ActionError: the desktop has no area, meaning no display was found.
    """
    if virtual.is_empty:
        raise ActionError(
            "virtual desktop is empty",
            user_message="Не удалось определить размер рабочего стола.",
        )
    span_x = max(virtual.width, 1)
    span_y = max(virtual.height, 1)
    offset_x = min(max(x - virtual.left, 0), span_x - 1)
    offset_y = min(max(y - virtual.top, 0), span_y - 1)
    nx = (offset_x * ABSOLUTE_MAX) // (span_x - 1) if span_x > 1 else 0
    ny = (offset_y * ABSOLUTE_MAX) // (span_y - 1) if span_y > 1 else 0
    return (nx, ny)


def drag_path(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    step_px: int,
) -> tuple[tuple[int, int], ...]:
    """Intermediate points from ``start`` to ``end``, ``end`` included.

    A drag that jumps straight to the destination is ignored by a good half of the
    applications worth dragging in: a list reorder, a canvas selection and a
    scrollbar all track ``WM_MOUSEMOVE`` and need to see the pointer travel. The
    steps are evenly spaced and no longer than ``step_px``; ``start`` itself is not
    repeated because the pointer is already there.
    """
    x0, y0 = start
    x1, y1 = end
    dx = x1 - x0
    dy = y1 - y0
    distance = max(abs(dx), abs(dy))
    if distance == 0:
        return ((x1, y1),)
    steps = max(1, -(-distance // max(step_px, 1)))
    points: list[tuple[int, int]] = []
    for index in range(1, steps + 1):
        points.append((x0 + (dx * index) // steps, y0 + (dy * index) // steps))
    if points[-1] != (x1, y1):
        points.append((x1, y1))
    return tuple(points)


def _pause(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def _mouse_timings() -> tuple[int, int, int]:
    """``(drag_step_px, drag_step_delay_ms, mouse_settle_ms)``, read fresh each call."""
    from ayris.core.config import get_settings

    section = get_settings().actions.input
    return (section.drag_step_px, section.drag_step_delay_ms, section.mouse_settle_ms)


def _key_hold_ms() -> int:
    from ayris.core.config import get_settings

    return get_settings().actions.input.key_hold_ms


class MousePoint(ActionParams):
    """Where a mouse action happens, in whichever frame the macro found convenient.

    Shared by every block below, so ``origin``/``logical``/``monitor`` mean the
    same thing in a click, a move and both ends of a drag.
    """

    x: int = Field(default=0, ge=-100_000, le=100_000, description="X")
    y: int = Field(default=0, ge=-100_000, le=100_000, description="Y")
    origin: Origin = Field(
        default=Origin.DESKTOP,
        description="Откуда считать координаты",
        json_schema_extra={"choices_ru": {item.value: item.title_ru for item in Origin}},
    )
    monitor: int | None = Field(
        default=None,
        ge=1,
        le=16,
        description="Номер монитора для origin=monitor; пусто — основной",
    )
    logical: bool = Field(
        default=False,
        description="Координаты в логических точках (с учётом масштаба), а не в пикселях",
    )

    def resolve(self, screen: ScreenBackend, layout: ScreenLayout) -> tuple[int, int]:
        """This point as physical virtual-desktop pixels.

        Raises:
            ActionError: the frame of reference cannot be established — no such
                monitor, or ``origin=window`` with nothing in the foreground.
        """
        if self.origin is Origin.CURSOR:
            cx, cy = screen.cursor()
            dx, dy = self._scaled(layout, layout.containing(cx, cy))
            return (cx + dx, cy + dy)
        if self.origin is Origin.MONITOR:
            monitor = (
                layout.by_index(self.monitor) if self.monitor is not None else layout.primary()
            )
            if monitor is None:
                raise ActionError(
                    "no monitors found",
                    user_message="Не вижу ни одного монитора.",
                )
            dx, dy = self._scaled(layout, monitor)
            return (monitor.rect.left + dx, monitor.rect.top + dy)
        if self.origin is Origin.WINDOW:
            rect = screen.active_window()
            if rect is None:
                raise ActionError(
                    "no foreground window",
                    user_message="Нет активного окна, от которого считать координаты.",
                )
            anchor = layout.containing(rect.left, rect.top)
            dx, dy = self._scaled(layout, anchor)
            return (rect.left + dx, rect.top + dy)
        anchor = layout.containing(self.x, self.y) or layout.primary()
        return self._scaled(layout, anchor)

    def _scaled(self, layout: ScreenLayout, monitor: winapi.MonitorInfo | None) -> tuple[int, int]:
        """The offset in physical pixels, converting from logical points if asked."""
        if not self.logical or monitor is None:
            return (self.x, self.y)
        dpi = layout.dpi_of(monitor)
        if dpi == BASE_DPI:
            return (self.x, self.y)
        return ((self.x * dpi) // BASE_DPI, (self.y * dpi) // BASE_DPI)


def _move_to(
    point: tuple[int, int],
    *,
    backend: InputBackend,
    layout: ScreenLayout,
) -> tuple[int, int]:
    """Send one absolute move and return the normalised coordinates used."""
    nx, ny = normalize_point(point[0], point[1], layout.virtual)
    backend.mouse_move(nx, ny)
    return (nx, ny)


@register
class MouseMove(Action):
    """Move the pointer, absolutely or by an offset."""

    meta: ClassVar = ActionMeta(
        name="MouseMove",
        category=ActionCategory.INPUT,
        title_ru="Переместить курсор",
        description_ru="Ставит курсор в точку экрана, монитора, окна или сдвигает его.",
    )

    class Params(MousePoint):
        pass

    def run(self, params: Params) -> ActionResult[tuple[int, int]]:
        screen = get_screen_backend()
        layout = screen.layout()
        target = params.resolve(screen, layout)
        backend = get_input_backend()
        _move_to(target, backend=backend, layout=layout)
        return ActionResult.done(
            f"Курсор в точке {target[0]}, {target[1]}.",
            value=target,
            data={"x": target[0], "y": target[1]},
        )


@register
class MouseClick(Action):
    """Click a button, at the pointer or at a given point."""

    meta: ClassVar = ActionMeta(
        name="MouseClick",
        category=ActionCategory.INPUT,
        title_ru="Щёлкнуть мышью",
        description_ru="Одиночный, двойной или тройной щелчок любой кнопкой мыши.",
    )

    class Params(MousePoint):
        button: MouseButton = Field(
            default=MouseButton.LEFT,
            description="Какая кнопка",
            json_schema_extra={"choices_ru": {item.value: item.title_ru for item in MouseButton}},
        )
        clicks: int = Field(default=1, ge=1, le=3, description="Сколько щелчков")
        move: bool = Field(
            default=False,
            description="Сначала переместить курсор в указанную точку",
        )

    def run(self, params: Params) -> ActionResult[str]:
        backend = get_input_backend()
        position: tuple[int, int] | None = None
        if params.move:
            screen = get_screen_backend()
            layout = screen.layout()
            position = params.resolve(screen, layout)
            _move_to(position, backend=backend, layout=layout)
            _pause(_mouse_timings()[2])
        hold = _key_hold_ms()
        for index in range(params.clicks):
            if index:
                _pause(hold)
            backend.mouse_button(params.button, pressed=True)
            _pause(hold)
            backend.mouse_button(params.button, pressed=False)
        kind = {1: "Щёлкнул", 2: "Дважды щёлкнул", 3: "Трижды щёлкнул"}[params.clicks]
        where = f" в точке {position[0]}, {position[1]}" if position else ""
        return ActionResult.done(
            f"{kind} {params.button.instrumental_ru}{where}.",
            value=params.button.value,
            data={"button": params.button.value, "clicks": params.clicks},
        )


@register
class MouseDrag(Action):
    """Press at one point, travel to another, release.

    The travel is interpolated — see :func:`drag_path` for why a straight jump does
    not work — and the button comes up in a ``finally``, because a drag left
    half-finished holds the left button down over the user's desktop.
    """

    meta: ClassVar = ActionMeta(
        name="MouseDrag",
        category=ActionCategory.INPUT,
        title_ru="Перетащить мышью",
        description_ru="Зажимает кнопку, плавно ведёт курсор в другую точку и отпускает.",
    )

    class Params(ActionParams):
        start: MousePoint = Field(default_factory=MousePoint, description="Откуда тащить")
        end: MousePoint = Field(default_factory=MousePoint, description="Куда тащить")
        button: MouseButton = Field(
            default=MouseButton.LEFT,
            description="Какая кнопка",
            json_schema_extra={"choices_ru": {item.value: item.title_ru for item in MouseButton}},
        )
        step_px: int | None = Field(
            default=None,
            ge=1,
            le=500,
            description="Длина шага; пусто — из настроек",
            json_schema_extra={"unit_ru": "px"},
        )
        step_delay_ms: int | None = Field(
            default=None,
            ge=0,
            le=1000,
            description="Пауза между шагами; пусто — из настроек",
            json_schema_extra={"unit_ru": "мс"},
        )

    def run(self, params: Params) -> ActionResult[tuple[int, int]]:
        screen = get_screen_backend()
        layout = screen.layout()
        origin = params.start.resolve(screen, layout)
        target = params.end.resolve(screen, layout)
        step_default, delay_default, settle = _mouse_timings()
        step = step_default if params.step_px is None else params.step_px
        delay = delay_default if params.step_delay_ms is None else params.step_delay_ms
        backend = get_input_backend()

        _move_to(origin, backend=backend, layout=layout)
        _pause(settle)
        path = drag_path(origin, target, step_px=step)
        backend.mouse_button(params.button, pressed=True)
        try:
            _pause(settle)
            for index, point in enumerate(path):
                if index:
                    _pause(delay)
                _move_to(point, backend=backend, layout=layout)
            _pause(settle)
        finally:
            backend.mouse_button(params.button, pressed=False)
        return ActionResult.done(
            f"Перетащил из {origin[0]}, {origin[1]} в {target[0]}, {target[1]}.",
            value=target,
            data={
                "from": list(origin),
                "to": list(target),
                "steps": len(path),
            },
        )


@register
class MouseWheel(Action):
    """Turn the wheel, vertically or horizontally, in notches."""

    meta: ClassVar = ActionMeta(
        name="MouseWheel",
        category=ActionCategory.INPUT,
        title_ru="Прокрутить колесом",
        description_ru="Крутит колесо мыши: вверх-вниз или влево-вправо, по щелчкам.",
    )

    class Params(ActionParams):
        clicks: int = Field(
            ...,
            ge=-1000,
            le=1000,
            description="Щелчков колеса: положительно — вверх или вправо",
        )
        horizontal: bool = Field(
            default=False,
            description="Горизонтальная прокрутка вместо вертикальной",
        )
        delay_ms: int | None = Field(
            default=None,
            ge=0,
            le=1000,
            description="Пауза между щелчками; пусто — из настроек",
            json_schema_extra={"unit_ru": "мс"},
        )

    def run(self, params: Params) -> ActionResult[int]:
        if params.clicks == 0:
            return ActionResult.done("Крутить не на что.", value=0)
        backend = get_input_backend()
        delay = _mouse_timings()[1] if params.delay_ms is None else params.delay_ms
        step = 1 if params.clicks > 0 else -1
        # One notch per event, not one event of N notches: applications with
        # smooth scrolling animate per notch, and a single large delta makes the
        # view jump instead of scroll.
        for index in range(abs(params.clicks)):
            if index:
                _pause(delay)
            backend.mouse_wheel(step, horizontal=params.horizontal)
        axis = "по горизонтали" if params.horizontal else "по вертикали"
        return ActionResult.done(
            f"Прокрутил {abs(params.clicks)} щелчков {axis}.",
            value=params.clicks,
            data={"clicks": params.clicks, "horizontal": params.horizontal},
        )
