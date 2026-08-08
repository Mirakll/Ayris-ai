"""Main settings window.

Task 01 only needs a window that opens and closes cleanly, so this is the bare
shell: title, icon-less, sensible minimum size, geometry centred on the screen
the cursor is on. Task 43 replaces the body with the sidebar navigation and the
eleven settings sections; the class name and constructor signature stay, so the
entry point does not need to change.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QCursor, QGuiApplication
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from ayris import __app_name__, __version__
from ayris.utils.logger import get_logger

__all__ = ["MainWindow"]

_log = get_logger(__name__)

_MIN_WIDTH = 900
_MIN_HEIGHT = 620

_PLACEHOLDER_TEXT = (
    "Ayris запущен.\n\n"
    "Настройки появятся здесь после выполнения задачи 43.\n"
    "Закройте окно, чтобы завершить работу."
)


class MainWindow(QMainWindow):
    """Settings window shell."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{__app_name__} {__version__} — Настройки")
        self.setMinimumSize(_MIN_WIDTH, _MIN_HEIGHT)
        self.setCentralWidget(self._build_central_widget())
        self._centre_on_current_screen()

    @staticmethod
    def _build_central_widget() -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(32, 32, 32, 32)

        label = QLabel(_PLACEHOLDER_TEXT)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        layout.addWidget(label)

        return container

    def _centre_on_current_screen(self) -> None:
        """Centre on the screen the cursor is currently on.

        On a multi-monitor setup this opens the window where the user is
        looking instead of always on the primary display. Task 43 replaces this
        with geometry restored from the config.
        """
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            # The PySide6 stubs say primaryScreen() always returns a QScreen, so
            # mypy calls this branch dead. At runtime it returns None on a
            # headless session, and reading availableGeometry() would crash.
            _log.warning(  # type: ignore[unreachable]
                "no screen reported by Qt; leaving window at default position"
            )
            return
        available = screen.availableGeometry()

        geometry = self.frameGeometry()
        geometry.setSize(self.minimumSize())
        geometry.moveCenter(available.center())
        self.move(geometry.topLeft())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        """Log the shutdown request.

        Task 44 turns this into "hide to tray" once a tray icon exists to bring
        the window back; until then closing the window really does exit.
        """
        _log.info("main window closed by user")
        super().closeEvent(event)
