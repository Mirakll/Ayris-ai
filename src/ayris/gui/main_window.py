"""The single Ayris window: a frameless dashboard with slide-down settings.

One window does everything. The left column is the showcase (logo, state sphere,
caption); the right column is the dialogue (top bar, conversation or status line,
active timers, command input). Settings are not a second window — the hamburger
slides them down as a full-window layer over the dashboard.

The window reflects the assistant's real state off the event bus and exposes
:meth:`set_state`, :meth:`set_status` and :meth:`add_message` for callers that
drive it directly. Closing only hides it; :meth:`exit` is the real shutdown.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Iterable
from typing import Final, Protocol, cast

from PySide6.QtCore import (
    QAbstractNativeEventFilter,
    QByteArray,
    QEvent,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QCloseEvent,
    QGuiApplication,
    QKeyEvent,
    QMoveEvent,
    QResizeEvent,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QWidget,
)

from ayris import __app_name__
from ayris.core.config import ConfigManager, WindowConfig
from ayris.core.events import (
    ActionFailed,
    AudioLevelChanged,
    EventBus,
    MacroFailed,
    MicToggled,
    MicToggleRequested,
    ModeChanged,
    OnlineStatusChanged,
    OpenCommandRequested,
    OverlayToggleRequested,
    OverlayVisibilityRequested,
    TranscriptReady,
    TtsStarted,
)
from ayris.core.models import Profile
from ayris.core.profile import ProfileSwitched
from ayris.core.state import AssistantState, MicMode, StatusSnapshot
from ayris.gui.dashboard import DialogView, SettingsLayer, ShowcasePanel
from ayris.gui.nav_sidebar import NavSidebar
from ayris.gui.overlay.dialog_log import DialogKind
from ayris.gui.overlay.timers_panel import TimerProvider
from ayris.gui.settings_search import SettingsSearch, SettingsSearchIndex, highlight_widget
from ayris.gui.tabs import SECTIONS, PlaceholderTab, SearchEntry, SettingsTab, tab_spec
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import SearchField
from ayris.gui.widgets.sphere.states import SphereState

__all__ = ["MainWindow", "ShowWindowNativeEventFilter", "restored_geometry"]

_STATE_DEBOUNCE_MS = 500
_WINDOWS_EVENT_TYPE: Final = b"windows_generic_MSG"

#: Status line shown for each assistant state; SPEAKING keeps the spoken line.
_STATE_STATUS: Final[dict[AssistantState, str]] = {
    AssistantState.IDLE: "Привет, чем помочь?",
    AssistantState.LISTENING: "Слушаю…",
    AssistantState.THINKING: "Думаю…",
    AssistantState.ERROR: "Что-то пошло не так",
}

_ROLE_KIND: Final[dict[str, DialogKind]] = {
    "user": DialogKind.HEARD,
    "heard": DialogKind.HEARD,
    "assistant": DialogKind.ANSWER,
    "answer": DialogKind.ANSWER,
    "error": DialogKind.ERROR,
}


class _NativeMessage(ctypes.Structure):
    """Prefix of WinAPI ``MSG``; only the message identifier is inspected."""

    _fields_ = (
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
    )


class _NativeMessagePointer(Protocol):
    def __int__(self) -> int: ...


class ShowWindowNativeEventFilter(QAbstractNativeEventFilter):
    """Restore the window when another Ayris instance is launched."""

    def __init__(self, window: MainWindow, message_id: int) -> None:
        super().__init__()
        self._window = window
        self._message_id = message_id

    def nativeEventFilter(  # noqa: N802
        self,
        event_type: QByteArray | bytes | bytearray | memoryview[int],
        message: int | object,
    ) -> tuple[bool, int]:
        event_name = event_type.data() if isinstance(event_type, QByteArray) else bytes(event_type)
        if not self._message_id or event_name != _WINDOWS_EVENT_TYPE or not message:
            return False, 0
        address = int(cast(_NativeMessagePointer, message))
        native_message = ctypes.cast(address, ctypes.POINTER(_NativeMessage)).contents
        if native_message.message == self._message_id:
            self.show_window()
        return False, 0

    def show_window(self) -> None:
        """Restore, raise and focus the existing window on the Qt UI thread."""
        self._window.showNormal()
        self._window.show()
        self._window.raise_()
        self._window.activateWindow()


def restored_geometry(state: WindowConfig, screens: Iterable[QRect], primary: QRect) -> QRect:
    """Return visible saved geometry, or centre it on the primary screen."""
    size = QSize(state.width, state.height)
    if state.x >= 0 and state.y >= 0:
        saved = QRect(QPoint(state.x, state.y), size)
        if any(saved.intersects(screen) for screen in screens):
            return saved
    fallback = QRect(QPoint(), size)
    fallback.moveCenter(primary.center())
    return fallback


class MainWindow(QMainWindow):
    """Frameless dashboard window; closing hides it, :meth:`exit` shuts down."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        theme: ThemeManager | None = None,
        manager: ConfigManager | None = None,
        bus: EventBus | None = None,
        timer_provider: TimerProvider | None = None,
        submit_text: Callable[[str], None] | None = None,
        profiles: Callable[[], list[Profile]] | None = None,
        switch_profile: Callable[[Profile], object] | None = None,
        snapshot: StatusSnapshot | None = None,
    ) -> None:
        super().__init__(parent)
        application = QApplication.instance()
        if theme is None:
            if not isinstance(application, QApplication):
                raise RuntimeError("MainWindow требует QApplication")
            theme = ThemeManager(application, parent=self)
            theme.apply()
        self._theme = theme
        self._manager = manager or ConfigManager()
        self._bus = bus
        self._submit_text = submit_text
        self._profiles = profiles
        self._switch_profile = switch_profile
        self._pages: dict[str, SettingsTab] = {}
        self._current_section = "general"
        self._allow_close = False
        self._fullscreen = False
        self._pending_fullscreen = False
        self._restoring = True
        self._search_index = SettingsSearchIndex()
        self._unsubscribers: list[Callable[[], None]] = []

        self.setWindowTitle(__app_name__)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setMinimumSize(theme.metric("window_min_width"), theme.metric("window_min_height"))
        self.setCentralWidget(self._build_central_widget(timer_provider))

        self._state_timer = QTimer(self)
        self._state_timer.setSingleShot(True)
        self._state_timer.setInterval(_STATE_DEBOUNCE_MS)
        self._state_timer.timeout.connect(self._save_window_state)

        self._subscribe_to_bus()
        self._reload_profiles()
        if snapshot is not None:
            self.sync(snapshot)

        self._restore_window_state()
        self._restoring = False

    # -- properties kept for compatibility ----------------------------------

    @property
    def created_sections(self) -> tuple[str, ...]:
        return tuple(self._pages)

    @property
    def current_section(self) -> str:
        return self._current_section

    @property
    def search_index(self) -> SettingsSearchIndex:
        return self._search_index

    @property
    def settings_open(self) -> bool:
        return self._settings_layer.is_open

    # -- public dashboard API -----------------------------------------------

    def set_state(self, state: SphereState | AssistantState | str) -> None:
        """Drive the sphere, the microphone highlight and the status line."""
        value = state.value if isinstance(state, SphereState | AssistantState) else str(state)
        self._showcase.set_state(value)
        self._dialog.set_voice_active(value == SphereState.LISTENING.value)
        default = _STATE_STATUS.get(AssistantState(value))
        if default is not None:
            self._dialog.set_status(default)

    def set_status(self, text: str) -> None:
        self._dialog.set_status(text)

    def add_message(self, role: DialogKind | str, text: str) -> None:
        if isinstance(role, DialogKind):
            kind = role
        else:
            kind = _ROLE_KIND.get(str(role), DialogKind.ANSWER)
        self._dialog.add_message(kind, text)

    def set_level(self, level: float) -> None:
        self._showcase.set_level(max(0.0, min(1.0, level)))

    def set_mic(self, *, enabled: bool, mode: MicMode) -> None:
        self._dialog.set_mic(enabled=enabled, mode=mode)

    def set_online(self, *, online: bool, detail: str = "") -> None:
        self._dialog.set_online(online=online, detail=detail)

    def set_profile(self, name: str) -> None:
        self._dialog.set_profile(name)

    def sync(self, snapshot: StatusSnapshot) -> None:
        self.set_state(snapshot.state.value)
        self.set_mic(enabled=snapshot.mic_enabled, mode=snapshot.mic_mode)
        self.set_online(online=snapshot.online, detail=snapshot.detail)

    # -- settings sections --------------------------------------------------

    def open_section(self, key: str) -> SettingsTab:
        # Create a section page only on its first visit.
        spec = tab_spec(key)
        page = self._pages.get(key)
        if page is None:
            page = (
                spec.factory(self._manager, self._theme, self._bus)
                if spec.factory is not None
                else PlaceholderTab(spec, self._manager, self._theme, self._bus)
            )
            self._pages[key] = page
            self._stack.addWidget(page)
            page.search_entries_changed.connect(lambda page=page: self._index_page(page))
            self._index_page(page)
        self._current_section = key
        self._stack.setCurrentWidget(page)
        self._sidebar.select_section(key)
        self._schedule_state_save()
        return page

    def open_settings(self) -> None:
        # Drop the WebGL sphere's native surface so the layer can cover the panel.
        self._showcase.set_sphere_visible(False)
        self._settings_layer.setGeometry(self._root.rect())
        self._settings_layer.open_layer()

    def close_settings(self) -> None:
        self._settings_layer.close_layer()

    def toggle_settings(self) -> None:
        if self._settings_layer.is_open:
            self.close_settings()
        else:
            self.open_settings()

    def _on_settings_closed(self) -> None:
        # Restore the sphere hidden in open_settings, then return focus to input.
        self._showcase.set_sphere_visible(True)
        self._dialog.focus_command()

    # -- full screen --------------------------------------------------------

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self._fullscreen)

    def set_fullscreen(self, active: bool) -> None:
        """Enter or leave full screen, flattening the frameless card's corners.

        The window is frameless with a translucent, rounded background, so a
        plain ``showFullScreen`` would leave the desktop showing through the
        rounded corners. The panels and the root card drop their radius and
        border while full-screen, then restore them on the way back.
        """
        if active == self._fullscreen:
            return
        self._apply_fullscreen_chrome(active)
        if active:
            self.showFullScreen()
        else:
            self.showNormal()
        # A deliberate toggle is worth persisting at once, not on the debounce.
        self._save_window_state()

    def _apply_fullscreen_chrome(self, active: bool) -> None:
        # Flatten (or restore) the frameless card's corners and the panels; the
        # actual show/normal transition is the caller's job.
        self._fullscreen = active
        self._showcase.set_fullscreen(active)
        self._dialog.set_flat(active)
        self._refresh_theme()

    # -- lifecycle ----------------------------------------------------------

    def exit(self) -> None:
        # Closing is accepted only during explicit application shutdown.
        self._allow_close = True
        self._state_timer.stop()
        self._save_window_state()
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        for page in self._pages.values():
            page.dispose()
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self._save_window_state()
        if self._allow_close:
            event.accept()
            return
        self.hide()
        event.ignore()

    def moveEvent(self, event: QMoveEvent) -> None:  # noqa: N802
        super().moveEvent(event)
        self._schedule_state_save()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._settings_layer.setGeometry(self._root.rect())
        self._schedule_state_save()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._pending_fullscreen and not self._fullscreen:
            self._pending_fullscreen = False
            self.set_fullscreen(True)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        super().changeEvent(event)
        # Keep the chrome in sync when something outside set_fullscreen changes
        # the window state — e.g. a tray restore calling showNormal(). Minimising
        # keeps the full-screen intent so restoring from the taskbar returns to it.
        if event.type() == QEvent.Type.WindowStateChange:
            state = self.windowState()
            if state & Qt.WindowState.WindowMinimized:
                return
            is_fullscreen = bool(state & Qt.WindowState.WindowFullScreen)
            if is_fullscreen != self._fullscreen:
                self._apply_fullscreen_chrome(is_fullscreen)
                self._schedule_state_save()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._settings_layer.is_open:
            self.close_settings()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape and self._fullscreen:
            self.set_fullscreen(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self._search_field and event.type() == QEvent.Type.KeyPress:
            key_event = event if isinstance(event, QKeyEvent) else None
            if key_event is not None and self._search_popup.isVisible():
                if key_event.key() in (Qt.Key.Key_Down, Qt.Key.Key_Up):
                    step = 1 if key_event.key() == Qt.Key.Key_Down else -1
                    row = (self._search_popup.currentRow() + step) % self._search_popup.count()
                    self._search_popup.setCurrentRow(row)
                    return True
                if key_event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    item = self._search_popup.currentItem()
                    if item is not None:
                        self._search_popup.activate_current()
                    return True
                if key_event.key() == Qt.Key.Key_Escape:
                    self._search_popup.hide()
                    return True
        return super().eventFilter(watched, event)

    # -- construction -------------------------------------------------------

    def _build_central_widget(self, timer_provider: TimerProvider | None) -> QWidget:
        self._root = QFrame(self)
        self._root.setObjectName("windowRoot")
        outer = QHBoxLayout(self._root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._showcase = ShowcasePanel(self._theme, self._root)
        self._showcase.drag_started.connect(self._start_system_move)
        self._showcase.fullscreen_toggle_requested.connect(self.toggle_fullscreen)

        separator = QFrame(self._root)
        separator.setObjectName("columnSeam")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFixedWidth(1)

        self._dialog = DialogView(self._theme, timer_provider=timer_provider, parent=self._root)
        self._dialog.settings_requested.connect(self.toggle_settings)
        self._dialog.minimize_requested.connect(self.showMinimized)
        self._dialog.close_requested.connect(self.close)
        self._dialog.command_submitted.connect(self._on_command)
        self._dialog.voice_requested.connect(self._on_voice)
        self._dialog.code_requested.connect(self._dialog.focus_command)
        self._dialog.profile_selected.connect(self._on_profile_chosen)
        self._dialog.drag_started.connect(self._start_system_move)

        outer.addWidget(self._showcase, 56)
        outer.addWidget(separator)
        outer.addWidget(self._dialog, 44)

        self._build_settings_layer()
        self._theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()
        return self._root

    def _build_settings_layer(self) -> None:
        self._settings_layer = SettingsLayer(self._theme, self._root)
        self._settings_layer.closed.connect(self._on_settings_closed)

        self._search_field = SearchField(
            self._settings_layer, placeholder="Найти настройку", theme=self._theme
        )
        self._search_field.installEventFilter(self)
        self._settings_layer.body_layout.addWidget(self._search_field)

        content = QHBoxLayout()
        content.setSpacing(self._theme.metric("spacing_lg"))
        self._sidebar = NavSidebar(SECTIONS, self._theme, self._settings_layer)
        self._stack = QStackedWidget(self._settings_layer)
        content.addWidget(self._sidebar)
        content.addWidget(self._stack, 1)
        self._settings_layer.body_layout.addLayout(content, 1)

        self._search_popup = SettingsSearch(self._search_index, self._theme, self)
        self._search_field.search_changed.connect(
            lambda query: self._search_popup.update_query(query, self._search_field)
        )
        self._search_popup.chosen.connect(self._open_search_result)
        self._sidebar.section_selected.connect(self.open_section)

    # -- event bus ----------------------------------------------------------

    def _subscribe_to_bus(self) -> None:
        bus = self._bus
        if bus is None:
            return
        self._unsubscribers = [
            bus.subscribe(ModeChanged, self._on_mode),
            bus.subscribe(AudioLevelChanged, self._on_level),
            bus.subscribe(MicToggled, self._on_mic),
            bus.subscribe(OnlineStatusChanged, self._on_online),
            bus.subscribe(ProfileSwitched, self._on_profile_switched),
            bus.subscribe(TranscriptReady, self._on_transcript),
            bus.subscribe(TtsStarted, self._on_answer),
            bus.subscribe(ActionFailed, self._on_action_failed),
            bus.subscribe(MacroFailed, self._on_macro_failed),
            bus.subscribe(OverlayToggleRequested, self._on_toggle_visibility),
            bus.subscribe(OverlayVisibilityRequested, self._on_visibility),
            bus.subscribe(OpenCommandRequested, self._on_open_command),
        ]

    def _on_mode(self, event: ModeChanged) -> None:
        self.set_state(event.state.value)
        self.set_mic(enabled=self._current_mic_enabled(), mode=event.mic_mode)
        if event.state is AssistantState.ERROR and event.detail:
            self.set_status(event.detail)

    def _on_level(self, event: AudioLevelChanged) -> None:
        self.set_level(event.rms)

    def _on_mic(self, event: MicToggled) -> None:
        mode = event.mic_mode if event.mic_mode is not None else MicMode.HYBRID
        self._mic_enabled = event.enabled
        self.set_mic(enabled=event.enabled, mode=mode)

    def _on_online(self, event: OnlineStatusChanged) -> None:
        self.set_online(online=event.online, detail=event.detail)

    def _on_profile_switched(self, event: ProfileSwitched) -> None:
        self.set_profile(event.profile.name)
        self._reload_profiles()

    def _on_transcript(self, event: TranscriptReady) -> None:
        if event.is_final and self._show_transcript():
            self.add_message(DialogKind.HEARD, event.text)

    def _on_answer(self, event: TtsStarted) -> None:
        self.add_message(DialogKind.ANSWER, event.text)
        self.set_status(event.text)

    def _on_action_failed(self, event: ActionFailed) -> None:
        self.add_message(DialogKind.ERROR, event.user_message or event.error)

    def _on_macro_failed(self, event: MacroFailed) -> None:
        self.add_message(DialogKind.ERROR, event.user_message or event.error)

    def _on_toggle_visibility(self, _event: OverlayToggleRequested) -> None:
        if self.isVisible():
            self.hide()
        else:
            self._raise_and_focus()

    def _on_visibility(self, event: OverlayVisibilityRequested) -> None:
        if event.visible:
            self._raise_and_focus()
        else:
            self.hide()

    def _on_open_command(self, event: OpenCommandRequested) -> None:
        # The «Открыть команду» link in the «Горячие клавиши» tab (task 55): make
        # sure the settings layer is up, switch to «Команды», and select the command.
        if not self._settings_layer.is_open:
            self.open_settings()
        page = self.open_section("commands")
        reveal = getattr(page, "reveal_command", None)
        if callable(reveal):
            reveal(event.command_id)

    # -- interaction --------------------------------------------------------

    def _on_command(self, text: str) -> None:
        self.add_message(DialogKind.HEARD, text)
        if self._submit_text is not None:
            self._submit_text(text)

    def _on_voice(self) -> None:
        if self._bus is not None:
            self._bus.publish(MicToggleRequested())

    def _on_profile_chosen(self, profile: object) -> None:
        if isinstance(profile, Profile) and self._switch_profile is not None:
            self._switch_profile(profile)

    def _reload_profiles(self) -> None:
        if self._profiles is not None:
            self._dialog.set_profiles(tuple(self._profiles()))

    def _start_system_move(self) -> None:
        handle = self.windowHandle()
        if handle is not None:
            handle.startSystemMove()

    def _raise_and_focus(self) -> None:
        self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    # -- helpers ------------------------------------------------------------

    def _current_mic_enabled(self) -> bool:
        return getattr(self, "_mic_enabled", True)

    def _show_transcript(self) -> bool:
        try:
            return bool(self._manager.settings.overlay.show_transcript)
        except Exception:  # pragma: no cover - defensive: settings shape drift
            return True

    def _index_page(self, page: SettingsTab) -> None:
        for entry in page.search_entries:
            self._search_index.add(entry)

    def _open_search_result(self, entry: object) -> None:
        if not isinstance(entry, SearchEntry):
            return
        self.open_section(entry.section_key)
        if entry.scroll_area is not None:
            entry.scroll_area.ensureWidgetVisible(entry.widget)
        highlight_widget(entry.widget, self._theme)
        self._search_field.clear()

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        radius = 0 if self._fullscreen else self._theme.metric("radius_lg")
        border = "none" if self._fullscreen else f"1px solid {color('border')}"
        self._root.setStyleSheet(
            f"#windowRoot {{ background: {color('surface')};"
            f" border: {border}; border-radius: {radius}px; }}"
            f"#columnSeam {{ color: {color('border')}; background: {color('border')};"
            " border: none; }"
        )

    # -- window state -------------------------------------------------------

    def _restore_window_state(self) -> None:
        state = self._manager.settings.window
        screens = tuple(screen.availableGeometry() for screen in QGuiApplication.screens())
        primary_screen = QGuiApplication.primaryScreen()
        if primary_screen is not None:
            primary = primary_screen.availableGeometry()
            self.setGeometry(restored_geometry(state, screens, primary))
        else:  # pragma: no cover - Qt stubs claim this headless branch is impossible
            self.resize(state.width, state.height)  # type: ignore[unreachable]
        key = state.section if any(spec.key == state.section for spec in SECTIONS) else "general"
        self.open_section(key)
        # Defer full screen to the first show: applying it here would force the
        # window visible before the caller (and start-minimized) had their say.
        self._pending_fullscreen = state.fullscreen

    def _schedule_state_save(self) -> None:
        if not self._restoring and hasattr(self, "_state_timer"):
            self._state_timer.start()

    def _save_window_state(self) -> None:
        values: dict[str, object] = {
            "window.section": self._current_section,
            "window.fullscreen": self._fullscreen,
        }
        # A full screen has no meaningful position or size to remember; keep the
        # last normal geometry so leaving full screen returns to it.
        if not self._fullscreen:
            geometry = self.normalGeometry()
            values.update(
                {
                    "window.x": geometry.x(),
                    "window.y": geometry.y(),
                    "window.width": geometry.width(),
                    "window.height": geometry.height(),
                }
            )
        current = self._manager.settings.window
        if all(
            getattr(current, path.removeprefix("window.")) == value
            for path, value in values.items()
        ):
            return
        self._manager.apply(values)
