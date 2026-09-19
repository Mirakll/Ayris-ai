"""Interactive showcase: ``python -m ayris.gui.widgets.geosphere.demo``."""

from __future__ import annotations

import sys
from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ayris import __app_name__
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.geosphere import GeoSphereWidget, GeoState, PaletteName
from ayris.utils.dpi import enable_per_monitor_dpi_awareness


class GeoSphereDemoWindow(QMainWindow):
    def __init__(self, theme: ThemeManager) -> None:
        super().__init__()
        self.setWindowTitle("Ayris — AEGIS geo-сфера")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        self.sphere = GeoSphereWidget(theme, point_count=1800, target_fps=60, prefer_opengl=False)
        self.sphere.setMinimumSize(560, 460)
        layout.addWidget(self.sphere, 1)

        state_row = QHBoxLayout()
        labels = {
            GeoState.IDLE: "Покой",
            GeoState.LISTENING: "Слушаю",
            GeoState.THINKING: "Думаю",
            GeoState.SPEAKING: "Говорю",
            GeoState.ERROR: "Ошибка",
        }
        for state, label in labels.items():
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=state: self.sphere.set_state(value))
            state_row.addWidget(button)
        layout.addLayout(state_row)

        palette_row = QHBoxLayout()
        palette_row.addWidget(QLabel("Палитра"))
        palette = QComboBox()
        palette.addItem("Циан / бело-голубая", PaletteName.CYAN)
        palette.addItem("Золото / янтарь", PaletteName.GOLD)
        palette.currentIndexChanged.connect(
            lambda _index: self.sphere.set_palette(palette.currentData())
        )
        palette_row.addWidget(palette, 1)
        layout.addLayout(palette_row)

        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Уровень микрофона"))
        level = QSlider(Qt.Orientation.Horizontal)
        level.setRange(0, 100)
        level.setValue(45)
        level.valueChanged.connect(lambda value: self.sphere.set_level(value / 100.0))
        self.sphere.set_level(0.45)
        level_row.addWidget(level, 1)
        layout.addLayout(level_row)

        points_row = QHBoxLayout()
        points_row.addWidget(QLabel("Точек"))
        points = QSlider(Qt.Orientation.Horizontal)
        points.setRange(300, 4000)
        points.setSingleStep(100)
        points.setValue(1800)
        points.valueChanged.connect(self.sphere.set_point_count)
        self.points_label = QLabel("1800")
        points.valueChanged.connect(lambda value: self.points_label.setText(str(value)))
        points_row.addWidget(points, 1)
        points_row.addWidget(self.points_label)
        layout.addLayout(points_row)

        self.metrics = QLabel()
        self.metrics.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.metrics)
        self.sphere.metrics_changed.connect(self._show_metrics)
        self._show_metrics(0.0, 0.0, self.sphere.rendered_point_count, self.sphere.backend_name)
        self.setCentralWidget(content)
        self.resize(780, 720)

    def _show_metrics(self, fps: float, frame_ms: float, points: int, backend: str) -> None:
        self.metrics.setText(f"{fps:5.1f} FPS · {frame_ms:4.2f} мс · {points} точек · {backend}")


def main(argv: Sequence[str] | None = None) -> int:
    enable_per_monitor_dpi_awareness()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    application = QApplication(list(argv) if argv is not None else sys.argv)
    application.setApplicationName(f"{__app_name__} — geo-сфера")
    theme = ThemeManager(application)
    theme.apply()
    window = GeoSphereDemoWindow(theme)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
