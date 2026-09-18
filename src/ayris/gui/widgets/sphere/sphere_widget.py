"""Public animated sphere widget."""

from __future__ import annotations

import math
import time
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.geometry import (
    DeformParams,
    build_wire_grid,
    displace_normals,
    project_array,
    rotate_array,
)
from ayris.gui.widgets.sphere.renderer import (
    OpenGLSphereRenderer,
    PainterSphereRenderer,
    RenderScene,
    create_renderer,
)
from ayris.gui.widgets.sphere.states import (
    AnimationParams,
    PerformanceGovernor,
    SphereState,
    SphereStateMachine,
)

_MIN_POINTS: Final = 24


def _empty_scene(params: AnimationParams, glow: bool) -> RenderScene:
    empty = np.empty((0, 4), dtype=np.float64)
    axis = np.empty((0,), dtype=np.float64)
    return RenderScene(empty, axis, axis, params, (), glow)


class SphereWidget(QWidget):
    """A real 3D point cloud projected into a lightweight Qt overlay widget."""

    metrics_changed = Signal(float, float, int, str)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        point_count: int = 600,
        target_fps: int = 60,
        transition_ms: int | None = None,
        prefer_opengl: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        duration = (
            theme.metric("animation_normal") if transition_ms is None else max(0, transition_ms)
        )
        self._machine = SphereStateMachine(transition_seconds=duration / 1000.0)
        self._governor = PerformanceGovernor(target_fps)
        self._requested_points = 0
        self._rendered_points = 0
        self._grid_resolution: tuple[int, int] = (0, 0)
        self._normals_a: NDArray[np.float64] = np.empty((0, 3))
        self._normals_b: NDArray[np.float64] = np.empty((0, 3))
        self._vertical: NDArray[np.float64] = np.empty((0,))
        self._params = self._machine.parameters
        self._angle_x = -0.18
        self._angle_y = 0.0
        self._angle_z = 0.0
        self._clock = 0.0
        self._last_tick = time.perf_counter()
        self._fps = 0.0
        self._frame_ms = 0.0
        self._animations_enabled = True
        self._scene = _empty_scene(self._params, True)
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
    def state(self) -> SphereState:
        return self._machine.state

    @property
    def animation_parameters(self) -> AnimationParams:
        return self._params

    @property
    def point_count(self) -> int:
        return self._requested_points

    @property
    def rendered_point_count(self) -> int:
        return self._rendered_points

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

    def set_state(self, state: SphereState | str) -> None:
        self._machine.set_state(state)
        if not self._timer.isActive() and self.isVisible():
            self._params = self._machine.advance(0.0)
            self._rebuild_scene()
            self._renderer.update()

    def set_level(self, level: float) -> None:
        self._machine.set_level(level)

    def set_point_count(self, count: int) -> None:
        self._requested_points = max(100, min(3000, int(count)))
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
        self._angle_y = (self._angle_y + delta * self._params.spin) % math.tau
        self._angle_x = -0.18 + math.sin(self._clock * 0.37) * 0.06
        self._angle_z = math.sin(self._clock * 0.23) * 0.04
        self._rebuild_scene()
        self._renderer.update()
        if delta > 0.0:
            instant_fps = 1.0 / delta
            self._fps = instant_fps if self._fps == 0.0 else self._fps * 0.9 + instant_fps * 0.1

    def _rebuild_points(self) -> None:
        effective = max(
            _MIN_POINTS, round(self._requested_points * self._governor.level.point_ratio)
        )
        self._rendered_points = effective
        # A denser point budget maps to a denser wire grid, bounded so the
        # always-on-top overlay stays cheap even on weak GPUs.
        lat = int(min(40, max(12, round(math.sqrt(effective) * 1.15))))
        lon = int(min(64, max(18, round(lat * 1.6))))
        if (lat, lon) != self._grid_resolution:
            self._grid_resolution = (lat, lon)
            # subdiv=1: the lat/long cells are already small, so straight edges
            # read as a clean grid while halving the line count we paint.
            self._normals_a, self._normals_b = build_wire_grid(lat, lon, 1)
            latitude = (self._normals_a[:, 1] + self._normals_b[:, 1]) * 0.5
            self._vertical = np.clip(latitude * 0.5 + 0.5, 0.0, 1.0)
        self._rebuild_scene()

    def _rebuild_scene(self) -> None:
        params = self._params
        if self._normals_a.shape[0] == 0:
            self._scene = _empty_scene(params, self._governor.level.glow)
            return
        size = min(self.width(), self.height())
        radius = max(1.0, size * 0.33)
        breath = (
            math.sin(self._clock * params.breath_hz * math.tau) * params.breath_amp
            if params.breath_hz > 0.0
            else 0.0
        )
        deform = DeformParams(
            amp=params.amp + params.level_amp * self._machine.level,
            freq=params.freq,
            speed=params.speed,
            wave_amp=params.wave_amp,
            wave_freq=params.wave_freq,
            twist=params.twist,
            jitter=params.jitter,
            breath=breath,
        )
        pos_a = rotate_array(
            displace_normals(self._normals_a, self._clock, deform),
            self._angle_x,
            self._angle_y,
            self._angle_z,
        )
        pos_b = rotate_array(
            displace_normals(self._normals_b, self._clock, deform),
            self._angle_x,
            self._angle_y,
            self._angle_z,
        )
        xa, ya, da = project_array(pos_a, self.width(), self.height(), radius)
        xb, yb, db = project_array(pos_b, self.width(), self.height(), radius)
        depth = (da + db) * 0.5
        order = np.argsort(depth)  # far hemisphere first, near lines paint last
        segments = np.stack([xa, ya, xb, yb], axis=1)[order]
        shimmer = (
            (0.5 + 0.5 * math.sin(self._clock * params.shimmer_hz * math.tau)) * params.shimmer_amt
            if params.shimmer_amt > 0.0 and params.shimmer_hz > 0.0
            else 0.0
        )
        self._scene = RenderScene(
            segments,
            depth[order],
            self._vertical[order],
            params,
            self._machine.speaking_wave_progresses(),
            self._governor.level.glow,
            shimmer,
        )

    def _current_scene(self) -> RenderScene:
        return self._scene

    def _frame_rendered(self, frame_ms: float) -> None:
        self._frame_ms = self._frame_ms * 0.88 + frame_ms * 0.12
        self.record_frame_time(frame_ms)
        self.metrics_changed.emit(
            self._fps, self._frame_ms, self._rendered_points, self._renderer.backend_name
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
