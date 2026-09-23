"""Встроенный предпросмотр сферы для вкладки «Оверлей» (задача 56).

Показывает ту же сферу задачи 45, что и главное окно, с переключателем всех
пяти состояний и живым замером стоимости отрисовки. Облик приходит через
:meth:`apply_overlay`, которая переиспользует ``apply_overlay_appearance`` —
единственную карту «конфиг → сфера», — поэтому предпросмотр и showcase-сфера
всегда выглядят одинаково.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.dashboard.sphere_host import SphereLike, apply_overlay_appearance, make_sphere
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import SphereState

if TYPE_CHECKING:
    from ayris.core.config import OverlayConfig

__all__ = ["SpherePreview"]

#: Пять состояний сферы и их подписи в порядке слева направо.
_STATES: tuple[tuple[SphereState, str], ...] = (
    (SphereState.IDLE, "Покой"),
    (SphereState.LISTENING, "Слушаю"),
    (SphereState.THINKING, "Думаю"),
    (SphereState.SPEAKING, "Говорю"),
    (SphereState.ERROR, "Ошибка"),
)


class SpherePreview(QWidget):
    """Сфера задачи 45 + переключатель пяти состояний + замер отрисовки."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        # Голый QWidget наследует тёмную заливку окна из QSS; гасим её, чтобы
        # предпросмотр сливался с карточкой, а не рисовал плашку под сферой.
        self.setProperty("transparent", True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.metric("spacing_sm"))

        sphere_widget = make_sphere(theme, self)
        self._sphere: SphereLike = sphere_widget  # type: ignore[assignment]
        sphere_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sphere_widget.setMinimumHeight(theme.metric("control_height_lg") * 5)
        layout.addWidget(sphere_widget, 1)

        self._buttons = QButtonGroup(self)
        self._buttons.setExclusive(True)
        self._state_row = QHBoxLayout()
        self._state_row.setSpacing(theme.metric("spacing_xs"))
        for index, (state, label) in enumerate(_STATES):
            button = QPushButton(label)
            button.setObjectName("spherePreviewState")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAccessibleName(f"Состояние сферы: {label}")
            button.clicked.connect(lambda _checked=False, s=state: self.show_state(s))
            self._buttons.addButton(button, index)
            self._state_row.addWidget(button)
        layout.addLayout(self._state_row)

        self.readout = QLabel("—")
        self.readout.setProperty("role", "muted")
        self.readout.setAccessibleName("Замер отрисовки сферы")
        layout.addWidget(self.readout)

        # Замер есть только у QPainter-сферы (сигнал ``metrics_changed``); у
        # WebGL-сферы его нет — тогда строка остаётся прочерком.
        metrics = getattr(sphere_widget, "metrics_changed", None)
        if metrics is not None:
            metrics.connect(self._update_readout)

        first = self._buttons.button(0)
        if first is not None:
            first.setChecked(True)
        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()
        self.show_state(SphereState.IDLE)

    @property
    def sphere(self) -> SphereLike:
        return self._sphere

    @property
    def state(self) -> SphereState:
        return self._sphere.state

    def show_state(self, state: SphereState) -> None:
        """Переключить сферу в состояние и подсветить его кнопку."""
        self._sphere.set_state(state)
        for index, (candidate, _label) in enumerate(_STATES):
            if candidate is state:
                button = self._buttons.button(index)
                if button is not None and not button.isChecked():
                    button.setChecked(True)
                break

    def apply_overlay(self, overlay: OverlayConfig) -> None:
        """Толкнуть весь облик из ``overlay`` в сферу — та же карта, что у showcase."""
        apply_overlay_appearance(self._sphere, overlay)

    def _update_readout(self, fps: float, frame_ms: float, points: int, backend: str) -> None:
        self.readout.setText(f"{fps:.0f} к/с · {frame_ms:.1f} мс/кадр · {points} точек · {backend}")

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        typography = self._theme.theme.typography
        self._state_row.setSpacing(metric("spacing_xs"))
        # Сегментами: некрупные пилюли, у активной — акцентная заливка.
        self.setStyleSheet(
            f"#spherePreviewState {{ background: {color('surface_highlight')};"
            f" color: {color('text_secondary')}; border: none;"
            f" border-radius: {metric('radius_md')}px;"
            f" padding: {metric('spacing_xs')}px {metric('spacing_sm')}px;"
            f" font-size: {typography.caption_size}px; }}"
            f"#spherePreviewState:hover {{ color: {color('text_primary')}; }}"
            f"#spherePreviewState:checked {{ background: {color('accent')};"
            f" color: {color('on_accent')}; }}"
        )
