"""The one main overlay: an interactive, always-on-top panel over other windows.

There is a single overlay and two visibility states — shown or hidden. The
:class:`MainOverlay` widget assembles the sphere, indicators, dialogue log,
timers and command field; :class:`OverlayController` binds it to the event bus,
places it per monitor, and shows it without stealing focus.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QMouseEvent, QShowEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import OverlayConfig
from ayris.core.events import (
    ActionFailed,
    AudioLevelChanged,
    EventBus,
    MacroFailed,
    MicToggled,
    ModeChanged,
    OnlineStatusChanged,
    OverlayToggleRequested,
    OverlayVisibilityRequested,
    PttPressed,
    TranscriptReady,
    TtsStarted,
    WakeWordDetected,
)
from ayris.core.profile import ProfileSwitched
from ayris.core.state import AssistantState, MicMode, StatusSnapshot
from ayris.gui.overlay.command_input import CommandInput
from ayris.gui.overlay.dialog_log import DialogKind, DialogLog
from ayris.gui.overlay.indicators import MicIndicator, NetworkIndicator
from ayris.gui.overlay.placement import Geometry, place
from ayris.gui.overlay.timers_panel import TimerProvider, TimersPanel
from ayris.gui.overlay.window_flags import (
    NativeWindow,
    WindowFlags,
    native_window,
    overlay_qt_flags,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.sphere_widget import SphereWidget
from ayris.utils import winapi
from ayris.utils.logger import get_logger
from ayris.utils.monitors import MonitorInfo, MonitorNotFound, list_monitors

__all__ = ["MainOverlay", "OverlayController"]

_log = get_logger(__name__)

#: A low, measurable cadence for re-asserting topmost — never per frame, and
#: SWP_NOACTIVATE|NOMOVE|NOSIZE so it neither flickers nor takes focus.
_TOPMOST_MS: Final = 4000

_SPHERE_SIDE: Final = 56
_LOGICAL_SIZE: Final = (360, 420)


class MainOverlay(QWidget):
    """The frameless, translucent panel. Owns no state; setters reflect events."""

    command_submitted = Signal(str)
    hide_requested = Signal()
    settings_requested = Signal()
    interaction_started = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        point_count: int = 600,
        target_fps: int = 60,
        animations: bool = True,
        timer_provider: TimerProvider | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType(overlay_qt_flags()))
        self._theme = theme
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAccessibleName("Оверлей Айрис")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._root = QFrame(self)
        self._root.setObjectName("overlayRoot")
        outer.addWidget(self._root)
        root_layout = QVBoxLayout(self._root)

        header = QHBoxLayout()
        self.sphere = SphereWidget(theme, point_count=point_count, target_fps=target_fps)
        self.sphere.setFixedSize(_SPHERE_SIDE, _SPHERE_SIDE)
        self.sphere.set_animations_enabled(animations)
        self.title = QLabel("Ayris")
        self.title.setObjectName("overlayTitle")
        self.profile = QLabel("")
        self.profile.setObjectName("overlayProfile")
        self.network = NetworkIndicator(theme)
        self.mic = MicIndicator(theme)
        self.settings_button = QPushButton("Настройки")
        self.settings_button.setAccessibleName("Открыть настройки")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        self.hide_button = QPushButton("Скрыть")
        self.hide_button.setAccessibleName("Скрыть оверлей")
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header.addWidget(self.sphere)
        header.addWidget(self.title)
        header.addWidget(self.profile)
        header.addStretch(1)
        header.addWidget(self.network)
        header.addWidget(self.mic)
        header.addWidget(self.settings_button)
        header.addWidget(self.hide_button)
        root_layout.addLayout(header)

        self.log = DialogLog(theme)
        root_layout.addWidget(self.log, 1)
        self.timers = TimersPanel(theme, provider=timer_provider)
        root_layout.addWidget(self.timers)
        self.command = CommandInput(theme)
        self.command.submitted.connect(self.command_submitted.emit)
        root_layout.addWidget(self.command)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    # -- state, all reflecting confirmed events -----------------------------

    def set_assistant_state(self, state: AssistantState) -> None:
        self.sphere.set_state(state.value)

    def set_level(self, level: float) -> None:
        self.sphere.set_level(max(0.0, min(1.0, level)))

    def set_online(self, *, online: bool, detail: str = "") -> None:
        self.network.set_online(online=online, detail=detail)

    def set_mic(self, *, enabled: bool, mode: MicMode) -> None:
        self.mic.set_state(enabled=enabled, mode=mode)

    def set_profile(self, name: str) -> None:
        self.profile.setText(name)

    def add_dialog(self, kind: DialogKind, text: str) -> None:
        self.log.add_entry(kind, text)

    def focus_command(self) -> None:
        self.command.setFocus(Qt.FocusReason.OtherFocusReason)

    def apply_config(self, config: OverlayConfig) -> None:
        self.sphere.set_point_count(config.sphere_points)
        self.sphere.set_animations_enabled(config.animations)
        self.setWindowOpacity(config.opacity)

    # -- interaction & lifecycle -------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self.interaction_started.emit()
        super().mousePressEvent(event)

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        # Children start their own animation/timers on show; nothing to force.

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        # A hidden panel spends nothing: the sphere stops its own animation on
        # its hide event; the timers list is stopped explicitly here too.
        self.timers.stop_updates()
        super().hideEvent(event)

    def _refresh_theme(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_md")
        radius = self._theme.metric("radius_lg")
        surface = self._theme.theme.color("surface")
        border = self._theme.theme.color("border")
        text = self._theme.theme.color("text_primary")
        muted = self._theme.theme.color("text_secondary")
        layout = self._root.layout()
        if layout is not None:
            layout.setContentsMargins(pad, pad, pad, pad)
            layout.setSpacing(self._theme.metric("spacing_sm"))
        self._root.setStyleSheet(
            f"#overlayRoot {{ background: {surface}; border: 1px solid {border};"
            f" border-radius: {radius}px; }}"
            f"#overlayTitle {{ color: {text}; font-weight: 600; }}"
            f"#overlayProfile {{ color: {muted}; }}"
        )


class OverlayController(QObject):
    """Bind the single overlay to the event bus, place it, show it politely."""

    def __init__(
        self,
        bus: EventBus,
        theme: ThemeManager,
        settings: Callable[[], OverlayConfig],
        *,
        monitors: Callable[[], Sequence[MonitorInfo]] = list_monitors,
        timer_provider: TimerProvider | None = None,
        submit_text: Callable[[str], None] | None = None,
        show_settings: Callable[[], None] | None = None,
        native_factory: Callable[[int], NativeWindow | None] = native_window,
        snapshot: StatusSnapshot | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bus = bus
        self._settings = settings
        self._monitors = monitors
        self._submit_text = submit_text
        self._show_settings = show_settings
        self._native_factory = native_factory
        config = settings()
        self.overlay = MainOverlay(
            theme,
            point_count=config.sphere_points,
            target_fps=config.target_fps,
            animations=config.animations,
            timer_provider=timer_provider,
        )
        self._flags = WindowFlags(None)
        self._tool_window_applied = False
        self._topmost = QTimer(self)
        self._topmost.setInterval(_TOPMOST_MS)
        self._topmost.timeout.connect(self._reassert_topmost)
        self._idle_hide = QTimer(self)
        self._idle_hide.setSingleShot(True)
        self._idle_hide.timeout.connect(self.hide_overlay)

        self.overlay.command_submitted.connect(self._on_command)
        self.overlay.hide_requested.connect(
            lambda: bus.publish(OverlayVisibilityRequested(visible=False))
        )
        self.overlay.settings_requested.connect(self._on_settings)
        self.overlay.interaction_started.connect(self._allow_focus)

        self._unsubscribers = [
            bus.subscribe(ModeChanged, self._on_mode),
            bus.subscribe(AudioLevelChanged, self._on_level),
            bus.subscribe(MicToggled, self._on_mic),
            bus.subscribe(OnlineStatusChanged, self._on_online),
            bus.subscribe(ProfileSwitched, self._on_profile),
            bus.subscribe(TranscriptReady, self._on_transcript),
            bus.subscribe(TtsStarted, self._on_answer),
            bus.subscribe(ActionFailed, self._on_action_failed),
            bus.subscribe(MacroFailed, self._on_macro_failed),
            bus.subscribe(OverlayToggleRequested, self._on_toggle),
            bus.subscribe(OverlayVisibilityRequested, self._on_visibility),
            bus.subscribe(WakeWordDetected, self._on_activity),
            bus.subscribe(PttPressed, self._on_activity),
        ]
        if snapshot is not None:
            self.sync(snapshot)

    # -- public API ---------------------------------------------------------

    def sync(self, snapshot: StatusSnapshot) -> None:
        self.overlay.set_assistant_state(snapshot.state)
        self.overlay.set_mic(enabled=snapshot.mic_enabled, mode=snapshot.mic_mode)
        self.overlay.set_online(online=snapshot.online, detail=snapshot.detail)

    def start(self) -> None:
        if self._settings().enabled:
            self.show_overlay()

    def close(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._topmost.stop()
        self._idle_hide.stop()
        self.overlay.close()

    def show_overlay(self, *, activate: bool = False) -> None:
        overlay = self.overlay
        overlay.show()
        self._ensure_tool_window()
        self._place()
        self._flags.confirm_topmost()
        if not self._topmost.isActive():
            self._topmost.start()
        if activate:
            self._allow_focus()
            overlay.activateWindow()
            overlay.raise_()
            overlay.focus_command()

    def hide_overlay(self) -> None:
        self._topmost.stop()
        self._idle_hide.stop()
        self.overlay.hide()

    def toggle(self) -> None:
        if self.overlay.isVisible():
            self.hide_overlay()
        else:
            self.show_overlay()

    # -- placement & flags --------------------------------------------------

    def _ensure_tool_window(self) -> None:
        if self._tool_window_applied:
            return
        handle = int(self.overlay.winId())
        self._flags = WindowFlags(self._native_factory(handle))
        self._flags.apply_tool_window()
        self._tool_window_applied = True

    def _place(self) -> None:
        config = self._settings()
        monitors = list(self._monitors())
        if not monitors:
            return
        address: str | int | None = None if config.monitor == 0 else config.monitor
        try:
            placement = place(
                monitors,
                address=address,
                logical_size=_LOGICAL_SIZE,
                position=config.position,
                custom_logical=(config.custom_x, config.custom_y),
            )
        except MonitorNotFound:
            return
        self._apply_geometry(placement.geometry)

    def _apply_geometry(self, geometry: Geometry) -> None:
        handle = int(self.overlay.winId()) if self._tool_window_applied else 0
        if self._flags.available and handle:
            # Physical desktop coordinates: unambiguous across per-monitor DPI.
            rect = winapi.Rect(
                geometry.x, geometry.y, geometry.x + geometry.width, geometry.y + geometry.height
            )
            try:
                winapi.set_window_position(handle, rect)
                return
            except OSError as exc:  # pragma: no cover - live WinAPI only
                _log.debug("set_window_position не удалось: %s", exc)
        self.overlay.setGeometry(*geometry.as_tuple())

    def _allow_focus(self) -> None:
        # The window is shown WA_ShowWithoutActivating and is never WS_EX_NOACTIVATE,
        # so a click already activates it. Clearing the bit here is a no-op guard
        # in case a transient set was ever added.
        self._flags.set_no_activate(enabled=False)

    def _reassert_topmost(self) -> None:
        if self.overlay.isVisible():
            self._flags.confirm_topmost()

    def _arm_idle_hide(self) -> None:
        config = self._settings()
        if config.hide_when_idle and self.overlay.isVisible():
            self._idle_hide.start(int(config.idle_hide_sec * 1000))

    # -- event handlers -----------------------------------------------------

    def _on_command(self, text: str) -> None:
        if self._submit_text is not None:
            self._submit_text(text)
        else:
            _log.info("текстовая команда без подключённого пайплайна: %r", text)

    def _on_settings(self) -> None:
        if self._show_settings is not None:
            self._show_settings()

    def _on_mode(self, event: ModeChanged) -> None:
        self.overlay.set_assistant_state(event.state)
        # ModeChanged carries the mode but not the mute flag, which MicToggled
        # owns; keep the indicator's current enabled value.
        self.overlay.set_mic(enabled=self.overlay.mic.enabled, mode=event.mic_mode)
        if event.state is AssistantState.ERROR:
            self._on_activity(event)
        if event.state is AssistantState.IDLE:
            self._arm_idle_hide()
        else:
            self._idle_hide.stop()

    def _on_level(self, event: AudioLevelChanged) -> None:
        self.overlay.set_level(event.rms)

    def _on_mic(self, event: MicToggled) -> None:
        mode = event.mic_mode if event.mic_mode is not None else self.overlay.mic.mode
        self.overlay.set_mic(enabled=event.enabled, mode=mode)

    def _on_online(self, event: OnlineStatusChanged) -> None:
        self.overlay.set_online(online=event.online, detail=event.detail)

    def _on_profile(self, event: ProfileSwitched) -> None:
        self.overlay.set_profile(event.profile.name)

    def _on_transcript(self, event: TranscriptReady) -> None:
        if event.is_final and self._settings().show_transcript:
            self.overlay.add_dialog(DialogKind.HEARD, event.text)

    def _on_answer(self, event: TtsStarted) -> None:
        self.overlay.add_dialog(DialogKind.ANSWER, event.text)

    def _on_action_failed(self, event: ActionFailed) -> None:
        self.overlay.add_dialog(DialogKind.ERROR, event.user_message or event.error)

    def _on_macro_failed(self, event: MacroFailed) -> None:
        self.overlay.add_dialog(DialogKind.ERROR, event.user_message or event.error)

    def _on_toggle(self, _event: OverlayToggleRequested) -> None:
        self.toggle()

    def _on_visibility(self, event: OverlayVisibilityRequested) -> None:
        if event.visible:
            self.show_overlay()
        else:
            self.hide_overlay()

    def _on_activity(self, _event: object) -> None:
        if self._settings().enabled and not self.overlay.isVisible():
            self.show_overlay()
