"""The public AEGIS point-cloud sphere widget."""

from __future__ import annotations

import math
import time
from typing import Final

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.geosphere.field import animate
from ayris.gui.widgets.geosphere.geometry import CAMERA_TILT, BasePoint, point_cloud, project_dots
from ayris.gui.widgets.geosphere.palette import PaletteName, SpherePalette, get_palette
from ayris.gui.widgets.geosphere.renderer import (
    OpenGLSphereRenderer,
    PainterSphereRenderer,
    RenderScene,
    create_renderer,
)
from ayris.gui.widgets.geosphere.states import (
    AnimationParams,
    GeoState,
    GeoStateMachine,
    PerformanceGovernor,
)

_MIN_POINTS: Final = 60
_MAX_POINTS: Final = 4000


class GeoSphereWidget(QWidget):
    """A dense, self-glowing point cloud projected into a Qt overlay widget."""

    metrics_changed = Signal(float, float, int, str)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        point_count: int = 1800,
        target_fps: int = 60,
        palette: PaletteName | str = PaletteName.CYAN,
        transition_ms: int | None = None,
        prefer_opengl: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        duration = (
            theme.metric("animation_normal") if transition_ms is None else max(0, transition_ms)
        )
        self._machine = GeoStateMachine(transition_seconds=duration / 1000.0)
        self._governor = PerformanceGovernor(target_fps)
        self._palette: SpherePalette = get_palette(palette)
        self._requested_points = 0
        self._base: tuple[BasePoint, ...] = ()
        self._params = self._machine.parameters
        self._angle_x = CAMERA_TILT
        self._angle_y = 0.0
        self._angle_z = 0.0
        self._clock = 0.0
        self._last_tick = time.perf_counter()
        self._fps = 0.0
        self._frame_ms = 0.0
        self._animations_enabled = True
        self._scene = RenderScene((), self._params, self._palette, True)
        self.setAccessibleName("Сфера состояния Айрис")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._renderer = create_renderer(
            self._current_scene, theme, self, prefer_opengl=prefer_opengl
        )
        layout.addWidget(self._renderer)
        if isinstance(self._renderer, OpenGLSphereRenderer):
            self._renderer.context_failed.connect(self._fallback_to_painter)
        self._renderer.frame_rendered.connect(self._frame_rendered)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        self.set_point_count(point_count)
        self._apply_performance_level()
        theme.theme_changed.connect(self._theme_changed)

    @property
    def state(self) -> GeoState:
        return self._machine.state

    @property
    def animation_parameters(self) -> AnimationParams:
        return self._params

    @property
    def palette_name(self) -> str:
        return self._palette.base

    @property
    def point_count(self) -> int:
        return self._requested_points

    @property
    def rendered_point_count(self) -> int:
        return len(self._base)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_time_ms(self) -> float:
        return self._frame_ms

    @property
    def backend_name(self) -> str:
        return self._renderer.backend_name

    def is_animation_running(self) -> bool:
        return self._timer.isActive()

    def set_state(self, state: GeoState | str) -> None:
        self._machine.set_state(state)
        if not self._timer.isActive() and self.isVisible():
            self._params = self._machine.advance(0.0)
            self._rebuild_scene()
            self._renderer.update()

    def set_level(self, level: float) -> None:
        self._machine.set_level(level)

    def set_palette(self, palette: PaletteName | str) -> None:
        self._palette = get_palette(palette)
        self._rebuild_scene()
        self._renderer.update()

    def set_point_count(self, count: int) -> None:
        self._requested_points = max(_MIN_POINTS, min(_MAX_POINTS, int(count)))
        self._rebuild_points()

    def set_animations_enabled(self, enabled: bool) -> None:
        self._animations_enabled = enabled
        if enabled and self.isVisible():
            self._start_timer()
        else:
            self._timer.stop()
            self._renderer.update()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._animations_enabled:
            self._start_timer()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def changeEvent(self, event: QEvent) -> None:  # noqa: N802
        if event.type() == QEvent.Type.WindowStateChange:
            window = self.window()
            if window is not None and window.isMinimized():
                self._timer.stop()
            elif self.isVisible() and self._animations_enabled:
                self._start_timer()
        super().changeEvent(event)

    def record_frame_time(self, frame_ms: float) -> None:
        previous = self._governor.level
        current = self._governor.record_frame(frame_ms)
        if current != previous:
            self._apply_performance_level()

    def _start_timer(self) -> None:
        self._last_tick = time.perf_counter()
        self._timer.start(max(1, round(1000 / self._governor.level.fps)))

    def _tick(self) -> None:
        started = time.perf_counter()
        delta = min(0.1, max(0.0, started - self._last_tick))
        self._last_tick = started
        self._clock += delta
        self._params = self._machine.advance(delta)
        self._advance_angles(delta)
        self._rebuild_scene()
        self._renderer.update()
        if delta > 0.0:
            instant_fps = 1.0 / delta
            self._fps = instant_fps if self._fps == 0.0 else self._fps * 0.9 + instant_fps * 0.1

    def _advance_angles(self, delta: float) -> None:
        # Thinking makes the spin uneven, as if compute were redirected.
        wobble = (
            1.0
            - self._params.irregular * 0.5
            + self._params.irregular * 0.9 * (0.5 + 0.5 * math.sin(self._clock * 0.7))
        )
        self._angle_y = (self._angle_y + delta * self._params.rotation_speed * wobble) % math.tau
        # Keep the elevated, top-forward camera; let it bob almost imperceptibly.
        self._angle_x = CAMERA_TILT + math.sin(self._clock * 0.31) * 0.05
        self._angle_z = math.sin(self._clock * 0.19) * 0.04

    def _rebuild_points(self) -> None:
        effective = max(
            _MIN_POINTS, round(self._requested_points * self._governor.level.point_ratio)
        )
        self._base = point_cloud(effective)
        self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        size = min(self.width(), self.height())
        radius = max(1.0, size * 0.33)
        displaced = animate(
            self._base,
            self._params,
            clock=self._clock,
            level=self._machine.level,
            angle_x=self._angle_x,
            angle_y=self._angle_y,
            angle_z=self._angle_z,
        )
        dots = project_dots(
            displaced,
            self.width(),
            self.height(),
            radius,
            base_point_size=max(1.1, size / 240.0),
        )
        speak_gain = 0.4 + 0.6 * self._machine.level
        core_wave = 0.5 + 0.5 * math.sin(self._clock * math.tau * 2.2)
        core_glow = self._params.core_pulse * core_wave * speak_gain
        shake = self._params.shake * min(8.0, size * 0.02)
        self._scene = RenderScene(
            dots,
            self._params,
            self._palette,
            self._governor.level.glow,
            core_glow,
            math.sin(self._clock * 91.0) * shake,
            math.cos(self._clock * 77.0) * shake,
        )

    def _current_scene(self) -> RenderScene:
        return self._scene

    def _frame_rendered(self, frame_ms: float) -> None:
        self._frame_ms = self._frame_ms * 0.88 + frame_ms * 0.12
        self.record_frame_time(frame_ms)
        self.metrics_changed.emit(
            self._fps, self._frame_ms, len(self._base), self._renderer.backend_name
        )

    def _apply_performance_level(self) -> None:
        self._rebuild_points()
        if self._timer.isActive():
            self._timer.setInterval(max(1, round(1000 / self._governor.level.fps)))

    def _fallback_to_painter(self) -> None:
        if isinstance(self._renderer, PainterSphereRenderer):
            return
        old = self._renderer
        replacement = PainterSphereRenderer(self._current_scene, self._theme, self)
        layout = self.layout()
        if isinstance(layout, QVBoxLayout):
            layout.replaceWidget(old, replacement)
        self._renderer = replacement
        replacement.frame_rendered.connect(self._frame_rendered)
        old.hide()
        old.deleteLater()
        replacement.show()

    def _theme_changed(self, _theme: object | None = None) -> None:
        self._renderer.update()
