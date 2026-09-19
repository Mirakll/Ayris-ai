"""Tray context menu: event commands in, authoritative state out."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QGuiApplication,
    QIcon,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShowEvent,
)
from PySide6.QtWidgets import QMenu

from ayris.core.events import (
    EventBus,
    MicModeRequested,
    MicToggleRequested,
    OverlayVisibilityRequested,
    ProfileSwitchRequested,
)
from ayris.core.models import Profile
from ayris.core.state import MicMode, StatusSnapshot

__all__ = ["TrayMenu"]

# Fallback palette so the menu still draws icons when built without a theme
# (e.g. in headless tests). :meth:`TrayMenu.restyle` swaps these for live tokens.
_DEFAULT_MUTED = "#948AAC"
_DEFAULT_ACCENT = "#9B6DFF"
_ICON_SIZE = 18


def _glyph_pixmap(name: str, color: str, *, ratio: float = 3.0) -> QPixmap:
    """Draw one line glyph on a 24-unit grid, tinted with ``color``.

    Mirrors the painter idiom in :mod:`ayris.gui.dashboard.input_bar`: a
    supersampled, device-pixel-ratio-aware canvas keeps thin strokes crisp at
    any Windows scaling instead of upscaling a tiny bitmap.
    """
    pixmap = QPixmap(round(_ICON_SIZE * ratio), round(_ICON_SIZE * ratio))
    pixmap.setDevicePixelRatio(ratio)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    unit = _ICON_SIZE / 24.0
    pen = QPen(QColor(color))
    pen.setWidthF(max(1.4, _ICON_SIZE * 0.1))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    _draw_glyph(painter, name, unit)
    painter.end()
    return pixmap


def _draw_glyph(painter: QPainter, name: str, u: float) -> None:
    if name == "mic":
        painter.drawRoundedRect(QRectF(9 * u, 2 * u, 6 * u, 11 * u), 3 * u, 3 * u)
        painter.drawArc(QRectF(5 * u, 4 * u, 14 * u, 14 * u), 180 * 16, 180 * 16)
        painter.drawLine(QPointF(12 * u, 18 * u), QPointF(12 * u, 22 * u))
        painter.drawLine(QPointF(8 * u, 22 * u), QPointF(16 * u, 22 * u))
    elif name == "mode":
        for y, cx in ((6, 15), (12, 9), (18, 15)):
            painter.drawLine(QPointF(4 * u, y * u), QPointF(20 * u, y * u))
            painter.drawEllipse(QPointF(cx * u, y * u), 2.2 * u, 2.2 * u)
    elif name == "layers":
        top = QPainterPath()
        top.moveTo(12 * u, 2.5 * u)
        top.lineTo(21 * u, 7 * u)
        top.lineTo(12 * u, 11.5 * u)
        top.lineTo(3 * u, 7 * u)
        top.closeSubpath()
        painter.drawPath(top)
        for dy in (12, 17):
            layer = QPainterPath()
            layer.moveTo(3 * u, dy * u)
            layer.lineTo(12 * u, (dy + 4.5) * u)
            layer.lineTo(21 * u, dy * u)
            painter.drawPath(layer)
    elif name == "gear":
        painter.drawEllipse(QPointF(12 * u, 12 * u), 3 * u, 3 * u)
        painter.drawEllipse(QPointF(12 * u, 12 * u), 6.5 * u, 6.5 * u)
        from math import cos, radians, sin

        for angle in range(0, 360, 45):
            rad = radians(angle)
            painter.drawLine(
                QPointF((12 + 6.5 * cos(rad)) * u, (12 + 6.5 * sin(rad)) * u),
                QPointF((12 + 9 * cos(rad)) * u, (12 + 9 * sin(rad)) * u),
            )
    elif name == "user":
        painter.drawEllipse(QPointF(12 * u, 8 * u), 4 * u, 4 * u)
        painter.drawArc(QRectF(4 * u, 14 * u, 16 * u, 16 * u), 20 * 16, 140 * 16)
    elif name == "power":
        painter.drawArc(QRectF(5 * u, 5 * u, 14 * u, 14 * u), 120 * 16, 300 * 16)
        painter.drawLine(QPointF(12 * u, 3 * u), QPointF(12 * u, 11.5 * u))


def _glyph_icon(name: str, *, muted: str, accent: str) -> QIcon:
    """A menu icon that is muted at rest and accent-coloured on hover/when on.

    Qt asks a :class:`QIcon` for a ``Selected``/``Active`` pixmap when the row
    is highlighted and for the ``On`` state when the action is checked, so a
    single icon recolours itself without any per-state swapping in code.
    """
    muted_pm = _glyph_pixmap(name, muted)
    accent_pm = _glyph_pixmap(name, accent)
    icon = QIcon()
    icon.addPixmap(muted_pm, QIcon.Mode.Normal, QIcon.State.Off)
    # Checked-but-not-hovered rows still read as "on".
    icon.addPixmap(accent_pm, QIcon.Mode.Normal, QIcon.State.On)
    for mode in (QIcon.Mode.Selected, QIcon.Mode.Active):
        icon.addPixmap(accent_pm, mode, QIcon.State.Off)
        icon.addPixmap(accent_pm, mode, QIcon.State.On)
    return icon


class _StickyMenu(QMenu):
    """A menu that stays open when a checkable item is toggled with the mouse.

    Flipping a mode or a switch should let you see the state change — and flip
    again — without the whole menu vanishing; plain commands still close it.
    """

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        action = self.activeAction()
        if action is not None and action.isEnabled() and action.isCheckable():
            action.trigger()
            return
        super().mouseReleaseEvent(event)


class TrayMenu(_StickyMenu):
    """A menu that reflects supplied state and never owns application state."""

    def __init__(
        self,
        bus: EventBus,
        *,
        profiles: Callable[[], Sequence[Profile]],
        active_profile: Callable[[], Profile],
        show_settings: Callable[[], None],
        quit_application: Callable[[], None],
    ) -> None:
        super().__init__()
        self._bus = bus
        self._profiles = profiles
        self._active_profile = active_profile
        self._overlay_visible = True
        self._muted = _DEFAULT_MUTED
        self._accent = _DEFAULT_ACCENT

        self.microphone = self.addAction("Микрофон включён")
        self.microphone.setCheckable(True)
        self.microphone.triggered.connect(lambda _checked: self._bus.publish(MicToggleRequested()))

        self.mode_menu = _StickyMenu("Режим", self)
        self.addMenu(self.mode_menu)
        self.mode_group = QActionGroup(self)
        self.mode_group.setExclusive(True)
        self.always = self.mode_menu.addAction("Always Listening")
        self.ptt = self.mode_menu.addAction("Push-to-Talk")
        for action, mode in ((self.always, MicMode.ALWAYS), (self.ptt, MicMode.PTT)):
            action.setCheckable(True)
            self.mode_group.addAction(action)
            action.triggered.connect(
                lambda _checked, selected=mode: self._bus.publish(MicModeRequested(selected))
            )

        self.overlay = self.addAction("Показывать основной оверлей")
        self.overlay.setCheckable(True)
        self.overlay.triggered.connect(self._request_overlay)
        self.settings = self.addAction("Настройки")
        self.settings.triggered.connect(show_settings)
        self.profile_menu = self.addMenu("Профиль")
        self.profile_menu.aboutToShow.connect(self.reload_profiles)
        self.aboutToShow.connect(self.reload_profiles)
        self.addSeparator()
        self.quit = self.addAction("Выход")
        self.quit.triggered.connect(quit_application)

        # Each row carries a line glyph; the submenu headers reuse their leaf icon.
        self._glyphs: dict[QAction, str] = {
            self.microphone: "mic",
            self.mode_menu.menuAction(): "mode",
            self.overlay: "layers",
            self.settings: "gear",
            self.profile_menu.menuAction(): "user",
            self.quit: "power",
        }
        self.restyle(muted=self._muted, accent=self._accent)

    def restyle(self, *, muted: str, accent: str) -> None:
        """Rebuild every glyph in the given palette; call it on theme changes."""
        self._muted = muted
        self._accent = accent
        for action, name in self._glyphs.items():
            action.setIcon(_glyph_icon(name, muted=muted, accent=accent))

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt override
        """Nudge the popup off the screen edges so it never sits flush to them.

        The reposition is deferred to the next event-loop turn: Qt (and the tray)
        finalise the popup's geometry *after* ``showEvent``, so moving it here
        directly would be overwritten.
        """
        super().showEvent(event)
        QTimer.singleShot(0, self._keep_off_edges)

    def _keep_off_edges(self, gap: int = 8) -> None:
        # A tray popup only shows when a screen exists, so this always resolves.
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        rect = self.frameGeometry()
        x, y = rect.x(), rect.y()
        if rect.right() > area.right() - gap:
            x = area.right() - gap - rect.width()
        if x < area.left() + gap:
            x = area.left() + gap
        if rect.bottom() > area.bottom() - gap:
            y = area.bottom() - gap - rect.height()
        if y < area.top() + gap:
            y = area.top() + gap
        if x != rect.x() or y != rect.y():
            self.move(x, y)

    def sync(self, snapshot: StatusSnapshot, *, overlay_visible: bool) -> None:
        widgets = (self.microphone, self.always, self.ptt, self.overlay)
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self.microphone.setChecked(snapshot.mic_enabled)
            self.microphone.setText(
                "Микрофон включён" if snapshot.mic_enabled else "Микрофон выключен"
            )
            self.always.setChecked(snapshot.mic_mode in (MicMode.ALWAYS, MicMode.HYBRID))
            self.ptt.setChecked(snapshot.mic_mode is MicMode.PTT)
            self._overlay_visible = overlay_visible
            self.overlay.setChecked(overlay_visible)
        finally:
            for widget, blocked in zip(widgets, previous, strict=True):
                widget.blockSignals(blocked)

    def _request_overlay(self, visible: bool) -> None:
        self._bus.publish(OverlayVisibilityRequested(visible=visible))

    def reload_profiles(self) -> None:
        self.profile_menu.clear()
        active = self._active_profile()
        group = QActionGroup(self.profile_menu)
        group.setExclusive(True)
        for profile in self._profiles():
            action = QAction(profile.name, self.profile_menu)
            action.setCheckable(True)
            action.setChecked(profile.id == active.id)
            if profile.id is not None:
                action.triggered.connect(
                    lambda _checked, profile_id=profile.id: self._bus.publish(
                        ProfileSwitchRequested(profile_id)
                    )
                )
            group.addAction(action)
            self.profile_menu.addAction(action)
