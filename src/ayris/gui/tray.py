"""System tray controller, status model and throttled notifications."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from PySide6.QtCore import QObject, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QSystemTrayIcon

from ayris.core.app import AyrisApp
from ayris.core.events import (
    ConfigChanged,
    MicToggled,
    ModeChanged,
    ModelDownloadFinished,
    NotificationRequested,
    OnlineStatusChanged,
    WorkerCrashed,
    WorkerRestarted,
)
from ayris.core.profile import ProfileSwitched
from ayris.core.state import AssistantState, StatusSnapshot
from ayris.gui.main_window import MainWindow
from ayris.gui.theme import ThemeManager
from ayris.gui.tray_menu import TrayMenu

__all__ = ["TrayController", "TrayLevel", "TrayStatus", "status_for"]

_THROTTLE_SECONDS: Final = 30.0


class TrayLevel(StrEnum):
    HEALTHY = "green"
    WARNING = "orange"
    PAUSED = "gray"


@dataclass(frozen=True, slots=True)
class TrayStatus:
    level: TrayLevel
    tooltip: str


def status_for(snapshot: StatusSnapshot, *, problem: str = "") -> TrayStatus:
    if problem or snapshot.state is AssistantState.ERROR:
        detail = problem or snapshot.detail or "требуется внимание"
        return TrayStatus(TrayLevel.WARNING, f"Ayris — проблема: {detail}")
    if not snapshot.mic_enabled:
        return TrayStatus(TrayLevel.PAUSED, "Ayris — микрофон выключен")
    return TrayStatus(TrayLevel.HEALTHY, f"Ayris — {snapshot.describe()}")


# Voice waveform in a 32×32 viewbox: (x, top, height); every bar shares a
# width.  A tall centre column tapering to short outer bars reads at a glance
# as "voice" — the mark stays the same across states, only its colour changes.
_WAVE_VIEWBOX: Final = 32.0
_WAVE_BAR_WIDTH: Final = 2.6
_WAVE_BARS: Final = (
    (5.9, 12.5, 7.0),
    (10.3, 9.5, 13.0),
    (14.7, 6.0, 20.0),
    (19.1, 9.5, 13.0),
    (23.5, 12.5, 7.0),
)
# (dark panel, light panel) per level: the bright tint suits a dark taskbar,
# the deeper shade keeps the same bars legible on a light one where the bright
# green/grey would otherwise wash out.
_WAVE_COLORS: Final = {
    TrayLevel.HEALTHY: ("#22C55E", "#16A34A"),
    TrayLevel.WARNING: ("#F59E0B", "#D97706"),
    TrayLevel.PAUSED: ("#9CA3AF", "#6B7280"),
}
# Draw each tray size natively so Windows picks a crisp pixmap instead of
# downscaling one — thin bars survive 16 px only when rasterised at 16 px.
_TRAY_ICON_SIZES: Final = (16, 20, 24, 32, 48, 64)


def _paint_waveform(painter: QPainter, size: int, color: QColor) -> None:
    unit = size / _WAVE_VIEWBOX
    radius = 1.3 * unit
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(color)
    for x, top, height in _WAVE_BARS:
        painter.drawRoundedRect(
            QRectF(x * unit, top * unit, _WAVE_BAR_WIDTH * unit, height * unit),
            radius,
            radius,
        )


def _icon(level: TrayLevel, *, light_panel: bool) -> QIcon:
    color = QColor(_WAVE_COLORS[level][1 if light_panel else 0])
    icon = QIcon()
    for size in _TRAY_ICON_SIZES:
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        _paint_waveform(painter, size, color)
        painter.end()
        icon.addPixmap(pixmap)
    return icon


class TrayController(QObject):
    def __init__(
        self,
        app: AyrisApp,
        window: MainWindow,
        theme: ThemeManager,
        *,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._app = app
        self._window = window
        self._theme = theme
        self._problem = ""
        self._notified: dict[str, float] = {}
        self.menu = TrayMenu(
            app.bus,
            profiles=app.profile_manager.list_all,
            active_profile=lambda: app.profile_manager.active,
            show_settings=self.show_settings,
            quit_application=self.quit_application,
        )
        self.icon = QSystemTrayIcon(self)
        self.icon.setContextMenu(self.menu)
        self.icon.activated.connect(self._activated)
        self._unsubscribers = [
            app.bus.subscribe(ModeChanged, self._state_changed),
            app.bus.subscribe(MicToggled, self._state_changed),
            app.bus.subscribe(OnlineStatusChanged, self._online_changed),
            app.bus.subscribe(ProfileSwitched, self._profile_changed),
            app.bus.subscribe(ConfigChanged, self._config_changed),
            app.bus.subscribe(NotificationRequested, self._notification),
            app.bus.subscribe(WorkerCrashed, self._worker_crashed),
            app.bus.subscribe(WorkerRestarted, self._worker_restarted),
            app.bus.subscribe(ModelDownloadFinished, self._model_finished),
        ]
        self._theme_changed: Callable[[object], None] = lambda _theme: self.refresh()
        theme.theme_changed.connect(self._theme_changed)
        self.refresh()

    def start(self) -> None:
        if QSystemTrayIcon.isSystemTrayAvailable():
            self.icon.show()

    def close(self) -> None:
        self.icon.hide()
        self._theme.theme_changed.disconnect(self._theme_changed)
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self.menu.close()

    def refresh(self) -> None:
        status = status_for(self._app.state.snapshot, problem=self._problem)
        self.icon.setIcon(_icon(status.level, light_panel=self._theme.theme.mode == "light"))
        self.icon.setToolTip(status.tooltip)
        self.menu.restyle(
            muted=self._theme.theme.color("text_muted"),
            accent=self._theme.theme.color("accent"),
        )
        self.menu.sync(
            self._app.state.snapshot,
            overlay_visible=self._app.settings.overlay.enabled,
        )

    def show_settings(self) -> None:
        self._window.showNormal()
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()

    def quit_application(self) -> None:
        self._window.exit()
        from PySide6.QtWidgets import QApplication

        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.quit()

    def notify(
        self,
        title: str,
        body: str,
        level: str = "info",
        *,
        kind: str | None = None,
        timeout_ms: int = 5000,
    ) -> bool:
        key = kind or f"{level}:{title}"
        now = time.monotonic()
        if now - self._notified.get(key, float("-inf")) < _THROTTLE_SECONDS:
            return False
        self._notified[key] = now
        if not self._app.settings.general.show_tray_notifications:
            return False
        message_icon = {
            "warning": QSystemTrayIcon.MessageIcon.Warning,
            "error": QSystemTrayIcon.MessageIcon.Critical,
        }.get(level, QSystemTrayIcon.MessageIcon.Information)
        self.icon.showMessage(title, body, message_icon, timeout_ms)
        return True

    def _activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason is QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show_settings()

    def _state_changed(self, _event: object) -> None:
        self.refresh()

    def _online_changed(self, event: OnlineStatusChanged) -> None:
        online_required = self._app.settings.voice.stt.mode == "online"
        if not event.online and online_required:
            self._problem = event.detail or "сеть недоступна в онлайн-режиме"
        else:
            self._problem = ""
        if not event.online and event.detail:
            self.notify("Работа офлайн", event.detail, "warning", kind="online-fallback")
        self.refresh()

    def _profile_changed(self, _event: ProfileSwitched) -> None:
        self.menu.reload_profiles()

    def _config_changed(self, event: ConfigChanged) -> None:
        if event.touches("overlay"):
            self.refresh()

    def _notification(self, event: NotificationRequested) -> None:
        self.notify(event.title, event.message, event.level, timeout_ms=event.timeout_ms)

    def _worker_crashed(self, event: WorkerCrashed) -> None:
        self._problem = f"воркер {event.worker} остановлен"
        title = "Потерян микрофон" if event.worker.casefold() == "audio" else "Сбой воркера"
        self.notify(title, self._problem, "error", kind=f"worker-crash:{event.worker}")
        self.refresh()

    def _worker_restarted(self, event: WorkerRestarted) -> None:
        self._problem = ""
        self.notify(
            "Воркер перезапущен",
            f"{event.worker}: работа восстановлена",
            kind=f"worker-restart:{event.worker}",
        )
        self.refresh()

    def _model_finished(self, event: ModelDownloadFinished) -> None:
        self.notify("Модель загружена", event.model_id, kind=f"model:{event.model_id}")
