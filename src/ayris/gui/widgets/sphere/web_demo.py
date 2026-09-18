"""Offline WebGL sphere showcase: ``python -m ayris.gui.widgets.sphere.web_demo``."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ayris import __app_name__
from ayris.gui.widgets.sphere.states import SphereState
from ayris.gui.widgets.sphere.web_widget import SphereWidget
from ayris.utils.dpi import enable_per_monitor_dpi_awareness


class WebSphereDemo(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Ayris — сфера (WebGL, офлайн)")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.sphere = SphereWidget(show_controls=False)
        self.sphere.setMinimumSize(640, 520)
        layout.addWidget(self.sphere, 1)

        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 8, 12, 8)
        labels = {
            SphereState.IDLE: "Спокойствие",
            SphereState.LISTENING: "Слушает",
            SphereState.THINKING: "Думает",
            SphereState.SPEAKING: "Говорит",
            SphereState.ERROR: "Ошибка",
        }
        for state, label in labels.items():
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=state: self.sphere.set_state(value))
            row.addWidget(button)
        row.addSpacing(16)
        row.addWidget(QLabel("Микрофон"))
        level = QSlider(Qt.Orientation.Horizontal)
        level.setRange(0, 100)
        level.valueChanged.connect(lambda value: self.sphere.set_level(value / 100.0))
        row.addWidget(level, 1)
        layout.addWidget(bar)

        self.setCentralWidget(content)
        self.resize(820, 700)


def main(argv: Sequence[str] | None = None) -> int:
    enable_per_monitor_dpi_awareness()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName(f"{__app_name__} — сфера")
    window = WebSphereDemo()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
