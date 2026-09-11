"""Main settings window with lazy navigation and persisted state."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable
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
from PySide6.QtGui import QCloseEvent, QGuiApplication, QKeyEvent, QMoveEvent, QResizeEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QMainWindow,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ayris import __app_name__, __version__
from ayris.core.config import ConfigManager, WindowConfig
from ayris.core.events import EventBus
from ayris.gui.nav_sidebar import NavSidebar
from ayris.gui.settings_search import SettingsSearch, SettingsSearchIndex, highlight_widget
from ayris.gui.tabs import SECTIONS, PlaceholderTab, SearchEntry, SettingsTab, tab_spec
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import SearchField

__all__ = ["MainWindow", "ShowWindowNativeEventFilter", "restored_geometry"]

_STATE_DEBOUNCE_MS = 500
_WINDOWS_EVENT_TYPE: Final = b"windows_generic_MSG"


class _NativeMessage(ctypes.Structure):
    """Prefix of WinAPI ``MSG``; only the message identifier is inspected."""

    _fields_ = (
        ("hwnd", ctypes.c_void_p),
        ("message", ctypes.c_uint),
    )


class _NativeMessagePointer(Protocol):
    def __int__(self) -> int: ...


class ShowWindowNativeEventFilter(QAbstractNativeEventFilter):
    """Restore the settings window when another Ayris instance is launched."""

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
    # Settings window whose normal close action only hides it.

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        theme: ThemeManager | None = None,
        manager: ConfigManager | None = None,
        bus: EventBus | None = None,
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
        self._pages: dict[str, SettingsTab] = {}
        self._current_section = "general"
        self._allow_close = False
        self._restoring = True
        self._search_index = SettingsSearchIndex()

        self.setWindowTitle(f"{__app_name__} {__version__} — Настройки")
        self.setMinimumSize(theme.metric("window_min_width"), theme.metric("window_min_height"))
        self.setCentralWidget(self._build_central_widget())

        self._state_timer = QTimer(self)
        self._state_timer.setSingleShot(True)
        self._state_timer.setInterval(_STATE_DEBOUNCE_MS)
        self._state_timer.timeout.connect(self._save_window_state)
        self._restore_window_state()
        self._restoring = False

    @property
    def created_sections(self) -> tuple[str, ...]:
        return tuple(self._pages)

    @property
    def current_section(self) -> str:
        return self._current_section

    @property
    def search_index(self) -> SettingsSearchIndex:
        return self._search_index

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

    def exit(self) -> None:
        # Closing is accepted only during explicit application shutdown.
        self._allow_close = True
        self._state_timer.stop()
        self._save_window_state()
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
        self._schedule_state_save()

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

    def _build_central_widget(self) -> QWidget:
        container = QWidget(self)
        outer = QVBoxLayout(container)
        margin = self._theme.metric("spacing_lg")
        outer.setContentsMargins(margin, margin, margin, margin)
        outer.setSpacing(self._theme.metric("spacing_lg"))

        self._search_field = SearchField(
            container, placeholder="Найти настройку", theme=self._theme
        )
        self._search_field.installEventFilter(self)
        outer.addWidget(self._search_field)

        content = QHBoxLayout()
        content.setSpacing(self._theme.metric("spacing_lg"))
        self._sidebar = NavSidebar(SECTIONS, self._theme, container)
        self._stack = QStackedWidget(container)
        content.addWidget(self._sidebar)
        content.addWidget(self._stack, 1)
        outer.addLayout(content, 1)

        self._search_popup = SettingsSearch(self._search_index, self._theme, self)
        self._search_field.search_changed.connect(
            lambda query: self._search_popup.update_query(query, self._search_field)
        )
        self._search_popup.chosen.connect(self._open_search_result)
        self._sidebar.section_selected.connect(self.open_section)
        return container

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

    def _schedule_state_save(self) -> None:
        if not self._restoring and hasattr(self, "_state_timer"):
            self._state_timer.start()

    def _save_window_state(self) -> None:
        geometry = self.normalGeometry()
        values = {
            "window.x": geometry.x(),
            "window.y": geometry.y(),
            "window.width": geometry.width(),
            "window.height": geometry.height(),
            "window.section": self._current_section,
        }
        current = self._manager.settings.window
        if all(
            getattr(current, path.removeprefix("window.")) == value
            for path, value in values.items()
        ):
            return
        self._manager.apply(values)
