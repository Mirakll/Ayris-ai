"""Slider and numeric field that always share one value."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QSpinBox, QWidget

from ayris.gui.theme import ThemeManager

__all__ = ["SliderField"]


class SliderField(QWidget):
    value_changed = Signal(int)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        minimum: int = 0,
        maximum: int = 100,
        value: int = 0,
        unit: str = "",
        label: str = "Значение",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.spin_box = QSpinBox()
        self.spin_box.setRange(minimum, maximum)
        self.unit_label = QLabel(unit)
        self.unit_label.setProperty("role", "secondary")
        self.slider.setAccessibleName(label)
        self.spin_box.setAccessibleName(label)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self.slider, 1)
        self._layout.addWidget(self.spin_box)
        self._layout.addWidget(self.unit_label)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin_box.valueChanged.connect(self._from_field)
        theme.theme_changed.connect(self._refresh_metrics)
        self.setValue(value)
        self._refresh_metrics()

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:  # noqa: N802
        self.slider.setValue(value)
        self.spin_box.setValue(value)

    def _from_slider(self, value: int) -> None:
        self.spin_box.setValue(value)
        self.value_changed.emit(value)

    def _from_field(self, value: int) -> None:
        self.slider.setValue(value)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self._layout.setSpacing(self._theme.metric("spacing_sm"))
        self.spin_box.setFixedWidth(self._theme.metric("slider_field_width"))
