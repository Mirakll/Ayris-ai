"""Interactive region selection: dim the screens, let the user drag a rectangle.

One frameless always-on-top widget per monitor, all of them dimmed, the selection
left undimmed with a frame and its size printed next to it. Escape or right-click
cancels, Enter accepts, releasing the mouse accepts. What comes back is a
rectangle in **physical virtual-desktop pixels** — the coordinate space
:mod:`ayris.actions.system.screenshot` captures in — or ``None`` for a
cancellation.

Three things here are not obvious and all three have a reason:

**One widget per screen, not one wide one.** A single widget spanning the whole
virtual desktop is the shorter code, and it renders wrong the moment two monitors
have different scaling: Qt picks one screen's device pixel ratio for the whole
window, so the half of the overlay on the other monitor is drawn at the wrong size
and the dimming stops at a visible seam. Per-screen widgets each get their own
ratio.

**Logical pixels in, physical pixels out.** Qt reports positions in
device-independent pixels, and with per-monitor DPI awareness the logical
positions of the screens are not the physical ones scaled by anything uniform —
Qt lays the logical desktop out itself so that scaled screens do not overlap. The
only reliable conversion is per screen: take the offset from *that screen's*
logical origin, multiply by *that screen's* ratio, add *that monitor's* physical
origin, which :mod:`ayris.utils.monitors` knows. Paired by device name, because
``QScreen.name()`` on Windows is the same ``\\\\.\\DISPLAY1`` that
``MONITORINFOEXW`` reports.

**The overlays stay hidden after the selection is over.** They are dismissed
before this returns, but the assistant's own always-on-top windows are restored
only :data:`RESTORE_DELAY_MS` later — the caller still has to take the screenshot,
and a HUD that reappears in the meantime lands in the picture.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from PySide6.QtCore import QEventLoop, QObject, QPoint, QRect, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QRegion,
    QScreen,
)
from PySide6.QtWidgets import QApplication, QWidget

from ayris.utils import monitors, winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["RegionSelector", "select_region"]

_log = get_logger(__name__)

#: Colour of the selection frame: light blue, visible over both a dark editor and
#: a white document, and not red — red on a screenshot reads as an error.
_FRAME_COLOUR: Final = QColor(0x4F, 0xC3, 0xF7)

#: Frame thickness in logical pixels. One pixel disappears on a 4K monitor.
_FRAME_WIDTH: Final = 2

#: The size hint: white on a dark plate, because it is drawn over the user's own
#: screen content and has to stay readable over anything.
_HINT_BACKGROUND: Final = QColor(0x11, 0x11, 0x11, 0xDD)
_HINT_FOREGROUND: Final = QColor(0xFF, 0xFF, 0xFF)
_HINT_PADDING: Final = 6
_HINT_MARGIN: Final = 8
_HINT_FONT_PT: Final = 10

#: Below this the drag is a click, and the selection is not shown or accepted.
_MIN_DRAG: Final = 4

#: How long our own always-on-top windows stay hidden after the selection ends.
#: Long enough for the caller to grab the frame — :data:`OVERLAY_SETTLE_S` plus the
#: capture itself — and short enough that a user who cancelled does not think the
#: assistant has died.
RESTORE_DELAY_MS: Final = 700

#: Extra time :func:`select_region` waits for the GUI thread on top of the
#: selection's own timeout, so that the two never race: a worker thread that gave
#: up while the overlay was still open would leave the overlay on screen.
_MARSHAL_GRACE_S: Final = 5.0


@dataclass(frozen=True, slots=True)
class _Surface:
    """One monitor as both Qt and Windows see it.

    ``screen`` is where the mouse coordinates come from, ``monitor`` is where they
    have to end up, and ``ratio`` is the number between them.
    """

    screen: QScreen
    monitor: monitors.MonitorInfo
    ratio: float

    def contains(self, point: QPoint) -> bool:
        return self.screen.geometry().contains(point)

    def to_physical(self, point: QPoint) -> tuple[int, int]:
        """A point in Qt's global logical space → physical desktop pixels.

        Clamped to the monitor, because fractional scaling makes the round trip
        inexact by a pixel and a selection must not spill onto the neighbour.
        """
        geometry = self.screen.geometry()
        rect = self.monitor.rect
        x = rect.left + round((point.x() - geometry.x()) * self.ratio)
        y = rect.top + round((point.y() - geometry.y()) * self.ratio)
        return (
            min(max(x, rect.left), rect.right),
            min(max(y, rect.top), rect.bottom),
        )


def _build_surfaces(
    screens: Sequence[QScreen],
    displays: Sequence[monitors.MonitorInfo],
) -> list[_Surface]:
    """Pair every Qt screen with the monitor it is.

    By device name first — ``QScreen.name()`` and ``MONITORINFOEXW.szDevice`` are
    the same string on Windows. When that fails, which it does on the odd driver
    and on any platform that is not Windows, both lists are sorted the same way and
    zipped: :func:`ayris.utils.monitors.order_monitors` already orders left to
    right, and Qt's own order is arbitrary, so sorting is what makes them line up.
    """
    by_name = {display.device: display for display in displays if display.device}
    surfaces: list[_Surface] = []
    unmatched: list[QScreen] = []
    for screen in screens:
        display = by_name.get(screen.name())
        if display is None:
            unmatched.append(screen)
            continue
        surfaces.append(_Surface(screen, display, screen.devicePixelRatio()))

    if unmatched:
        spare = [display for display in displays if display not in {s.monitor for s in surfaces}]
        spare.sort(key=lambda display: (display.rect.left, display.rect.top))
        unmatched.sort(key=lambda screen: (screen.geometry().x(), screen.geometry().y()))
        _log.debug(
            "%d Qt screens did not match by device name, pairing by position", len(unmatched)
        )
        for screen, display in zip(unmatched, spare, strict=False):
            surfaces.append(_Surface(screen, display, screen.devicePixelRatio()))
    return surfaces


class _Overlay(QWidget):
    """The dimming over one monitor. Paints; decides nothing.

    All of the state lives in the :class:`RegionSelector` that owns it, because a
    drag that starts on one monitor and ends on another is one selection drawn by
    two widgets.
    """

    def __init__(self, selector: RegionSelector, surface: _Surface, alpha: int) -> None:
        super().__init__(None)
        self._selector = selector
        self._surface = surface
        self._shade = QColor(0, 0, 0, alpha)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # keeps it out of the taskbar and Alt-Tab
            | Qt.WindowType.BypassWindowManagerHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setScreen(surface.screen)
        self.setGeometry(surface.screen.geometry())
        font = QFont(self.font())
        font.setPointSize(_HINT_FONT_PT)
        self.setFont(font)

    # -- painting ---------------------------------------------------------- #

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002 - Qt override
        painter = QPainter(self)
        selection = self._local_selection()
        if selection is None:
            painter.fillRect(self.rect(), self._shade)
            return
        # Dim everything except the selection, so the user sees the region exactly
        # as it will be captured rather than through a grey film.
        painter.setClipRegion(QRegion(self.rect()) - QRegion(selection))
        painter.fillRect(self.rect(), self._shade)
        painter.setClipping(False)
        painter.setPen(QPen(_FRAME_COLOUR, _FRAME_WIDTH))
        painter.drawRect(selection.adjusted(0, 0, -1, -1))
        self._paint_hint(painter, selection)

    def _local_selection(self) -> QRect | None:
        """The shared selection in this widget's coordinates, if it shows here."""
        selection = self._selector.logical_selection()
        if selection is None:
            return None
        local = selection.translated(-self.geometry().topLeft())
        return local if local.intersects(self.rect()) else None

    def _paint_hint(self, painter: QPainter, selection: QRect) -> None:
        """Print the size the capture will have, in physical pixels.

        Physical and not logical: the number has to match the file the user gets,
        and on a 150% monitor those differ by half again.
        """
        physical = self._selector.physical_selection()
        if physical is None:
            return
        text = f"{physical.width} × {physical.height}"
        metrics = painter.fontMetrics()
        box = metrics.boundingRect(text).adjusted(
            -_HINT_PADDING, -_HINT_PADDING, _HINT_PADDING, _HINT_PADDING
        )
        # Above the selection by default, inside it when there is no room — a
        # selection that starts at the top edge of the screen is the common case
        # for "capture this window".
        top = selection.top() - box.height() - _HINT_MARGIN
        if top < 0:
            top = min(selection.top() + _HINT_MARGIN, self.height() - box.height())
        left = min(max(selection.left(), 0), max(self.width() - box.width(), 0))
        plate = QRect(left, top, box.width(), box.height())
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_HINT_BACKGROUND)
        painter.drawRect(plate)
        painter.setPen(_HINT_FOREGROUND)
        painter.drawText(plate, int(Qt.AlignmentFlag.AlignCenter), text)

    # -- input ------------------------------------------------------------- #

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.RightButton:
            self._selector.cancel()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._selector.begin(event.globalPosition().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        # Qt grabs the mouse on press, so this keeps arriving here even when the
        # pointer has moved onto another monitor. That is what makes a selection
        # spanning two screens work.
        self._selector.extend(event.globalPosition().toPoint())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton:
            self._selector.finish(event.globalPosition().toPoint())

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt override
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._selector.cancel()
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._selector.accept()
        else:
            super().keyPressEvent(event)


class RegionSelector(QObject):
    """Owns the overlays, the selection, and the event loop that waits for it.

    Must be used on the GUI thread; :func:`select_region` is the entry point that
    gets there from wherever the caller is.
    """

    def __init__(self, *, dim: float = 0.45, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._alpha = int(max(0.0, min(dim, 0.9)) * 255)
        self._surfaces: list[_Surface] = []
        self._overlays: list[_Overlay] = []
        self._anchor: QPoint | None = None
        self._cursor: QPoint | None = None
        self._result: winapi.Rect | None = None
        self._loop: QEventLoop | None = None

    # -- state the overlays read ------------------------------------------- #

    def logical_selection(self) -> QRect | None:
        """The dragged rectangle in Qt's global logical coordinates."""
        if self._anchor is None or self._cursor is None:
            return None
        rect = QRect(self._anchor, self._cursor).normalized()
        if rect.width() < _MIN_DRAG or rect.height() < _MIN_DRAG:
            return None
        return rect

    def physical_selection(self) -> winapi.Rect | None:
        """The dragged rectangle in physical virtual-desktop pixels.

        Each corner is converted on the monitor it lands on, so a selection that
        starts on a 100% screen and ends on a 150% one comes out right.
        """
        logical = self.logical_selection()
        if logical is None:
            return None
        left, top = self._to_physical(logical.topLeft())
        # bottomRight() of a QRect is the last pixel *inside* it, and a capture
        # rectangle is half-open, so the far corner gets its one pixel back.
        right, bottom = self._to_physical(logical.bottomRight())
        return winapi.Rect(left, top, right + 1, bottom + 1)

    def _to_physical(self, point: QPoint) -> tuple[int, int]:
        surface = self._surface_at(point)
        if surface is None:
            return (point.x(), point.y())
        return surface.to_physical(point)

    def _surface_at(self, point: QPoint) -> _Surface | None:
        """The monitor a point is on, or the nearest one when it is in a gap.

        Two monitors of different heights leave a strip of virtual desktop that
        belongs to neither, and the pointer can be dragged through it.
        """
        for surface in self._surfaces:
            if surface.contains(point):
                return surface
        if not self._surfaces:
            return None
        return min(
            self._surfaces,
            key=lambda surface: _distance_squared(surface.screen.geometry(), point),
        )

    # -- what the overlays report ------------------------------------------ #

    def begin(self, point: QPoint) -> None:
        self._anchor = point
        self._cursor = point
        self._repaint()

    def extend(self, point: QPoint) -> None:
        if self._anchor is None:
            return
        self._cursor = point
        self._repaint()

    def finish(self, point: QPoint) -> None:
        if self._anchor is None:
            return
        self._cursor = point
        self.accept()

    def accept(self) -> None:
        """End with whatever is selected — or as a cancellation if that is nothing.

        A click without a drag arrives here too, and «выделил ничего» is a
        cancellation rather than a four-pixel screenshot.
        """
        self._result = self.physical_selection()
        self._close()

    def cancel(self) -> None:
        self._result = None
        self._close()

    def _repaint(self) -> None:
        for overlay in self._overlays:
            overlay.update()

    # -- running ----------------------------------------------------------- #

    def run(self, *, timeout_s: float = 60.0) -> winapi.Rect | None:
        """Show the overlays and wait for a rectangle. ``None`` when cancelled.

        A local event loop, so the rest of the application keeps drawing and the
        assistant is still listening while the user drags. It has to be a loop and
        not a callback: the action layer calls this from a worker thread and needs
        an answer to continue with.
        """
        self._surfaces = _build_surfaces(QGuiApplication.screens(), monitors.list_monitors())
        if not self._surfaces:
            _log.warning("no screens to select on")
            return None

        hidden = _hide_own_topmost()
        self._overlays = [_Overlay(self, surface, self._alpha) for surface in self._surfaces]
        for overlay in self._overlays:
            overlay.show()
        self._focus_overlay()

        self._loop = QEventLoop()
        # Cancellation on timeout, not an error: the user walked away, and leaving
        # the whole desktop dimmed until they come back is the worse outcome.
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(self._on_timeout)
        timer.start(max(int(timeout_s * 1000), 1000))
        try:
            self._loop.exec()
        finally:
            timer.stop()
            self._loop = None
            self._dismiss_overlays()
            _restore_later(hidden)
        return self._result

    def _focus_overlay(self) -> None:
        """Give the keyboard to the overlay the pointer is on, so Escape works."""
        pointer = QGuiApplication.screens()[0].geometry().center()
        cursor = QGuiApplication.primaryScreen()
        if cursor is not None:
            pointer = cursor.geometry().center()
        target = self._overlays[0]
        for overlay, surface in zip(self._overlays, self._surfaces, strict=True):
            if surface.contains(pointer):
                target = overlay
                break
        target.raise_()
        target.activateWindow()
        target.setFocus(Qt.FocusReason.OtherFocusReason)

    def _on_timeout(self) -> None:
        _log.info("region selection timed out")
        self.cancel()

    def _close(self) -> None:
        if self._loop is not None:
            self._loop.quit()

    def _dismiss_overlays(self) -> None:
        """Take the overlays off the screen before anyone captures it.

        ``hide`` is a request to the compositor, not a fact; the caller waits
        :data:`ayris.actions.system.screenshot.OVERLAY_SETTLE_S` on top of this.
        Processing the pending paint events here is what makes that wait short
        enough to be unnoticeable.
        """
        for overlay in self._overlays:
            overlay.hide()
            overlay.close()
        self._overlays = []
        application = QApplication.instance()
        if application is not None:
            application.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)


def _distance_squared(rect: QRect, point: QPoint) -> int:
    """How far a point is from a rectangle, squared. Zero when inside."""
    dx = max(rect.left() - point.x(), 0, point.x() - rect.right())
    dy = max(rect.top() - point.y(), 0, point.y() - rect.bottom())
    return dx * dx + dy * dy


def _hide_own_topmost() -> list[QWidget]:
    """Hide the assistant's own always-on-top windows and say which they were.

    The dimming would otherwise be drawn *under* a HUD that declares itself
    topmost, and — worse — that HUD would end up in the screenshot. Ordinary
    windows are left alone on purpose: «сними область» over the settings window is
    a legitimate thing to ask for.
    """
    hidden: list[QWidget] = []
    for widget in QApplication.topLevelWidgets():
        if not widget.isVisible():
            continue
        if widget.windowFlags() & Qt.WindowType.WindowStaysOnTopHint:
            widget.hide()
            hidden.append(widget)
    return hidden


def _restore_later(widgets: Sequence[QWidget]) -> None:
    """Bring the hidden windows back, but not until the capture is done."""
    if not widgets:
        return
    restorable = list(widgets)

    def restore() -> None:
        for widget in restorable:
            # A window closed while the overlay was up must stay closed.
            if not widget.isHidden():
                continue
            widget.show()

    QTimer.singleShot(RESTORE_DELAY_MS, restore)


def select_region(*, timeout_s: float = 60.0, dim: float = 0.45) -> winapi.Rect | None:
    """Ask the user to drag a rectangle. ``None`` when they cancel.

    Callable from any thread. The overlay is built and run on the GUI thread —
    Qt widgets exist nowhere else — and a call from a worker blocks until the
    answer comes back, which is what the action layer wants: it is already running
    on a pool thread with its own timeout.

    Returns:
        The selected rectangle in physical virtual-desktop pixels, or ``None`` if
        the user pressed Escape, right-clicked, clicked without dragging, or let
        the selection time out.

    Raises:
        RuntimeError: there is no ``QApplication``, so there is nothing to show an
            overlay in. The action layer turns this into ActionUnavailable.
    """
    application = QApplication.instance()
    if application is None:
        raise RuntimeError("region selection needs a running QApplication")

    def run() -> winapi.Rect | None:
        return RegionSelector(dim=dim).run(timeout_s=timeout_s)

    if QObject().thread() is application.thread():
        return run()

    answer: list[winapi.Rect | None] = []
    failure: list[BaseException] = []
    finished = threading.Event()

    def on_gui_thread() -> None:
        try:
            answer.append(run())
        except BaseException as exc:  # re-raised in the calling thread below
            failure.append(exc)
        finally:
            finished.set()

    QTimer.singleShot(0, application, on_gui_thread)
    if not finished.wait(timeout_s + _MARSHAL_GRACE_S):
        # The GUI thread is wedged. Reporting a cancellation is the only safe
        # answer: the overlay may still be up, and there is no way to take a
        # meaningful screenshot through it.
        _log.error("the GUI thread did not answer the region selection in time")
        return None
    if failure:
        raise failure[0]
    return answer[0] if answer else None
