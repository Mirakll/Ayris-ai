"""Geometry, state transitions and offscreen lifecycle for the sphere."""

from __future__ import annotations

import math
import os
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.geometry import (
    DeformParams,
    Point3D,
    build_wire_grid,
    displace_normals,
    fibonacci_sphere,
    project_points,
    rotate_point,
    simplex3,
    wireframe_sphere,
)
from ayris.gui.widgets.sphere.painter_demo import SphereDemoWindow
from ayris.gui.widgets.sphere.renderer import RenderScene, paint_scene
from ayris.gui.widgets.sphere.sphere_widget import SphereWidget
from ayris.gui.widgets.sphere.states import (
    PerformanceGovernor,
    SphereState,
    SphereStateMachine,
)


@pytest.fixture(scope="module")
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    application.processEvents()


def test_fibonacci_points_are_uniform_and_on_unit_sphere() -> None:
    points = fibonacci_sphere(800)
    assert len(points) == 800
    radial_error = max(
        abs(math.sqrt(point.x**2 + point.y**2 + point.z**2) - 1.0) for point in points
    )
    centre_error = max(
        abs(sum(getattr(point, axis) for point in points) / len(points)) for axis in "xyz"
    )
    assert radial_error < 1e-12
    assert centre_error < 0.002


def test_rotation_preserves_radius_and_projection_sorts_depth() -> None:
    point = rotate_point(Point3D(1.0, 0.0, 0.0), 0.2, 0.7, -0.3)
    assert math.sqrt(point.x**2 + point.y**2 + point.z**2) == pytest.approx(1.0)
    projected = project_points((Point3D(0.0, 0.0, 1.0), Point3D(0.0, 0.0, -1.0)), 200, 200, 70)
    assert [item.depth for item in projected] == [-1.0, 1.0]
    assert projected[0].size < projected[1].size
    assert projected[0].opacity < projected[1].opacity


def test_wireframe_is_continuous_and_lies_on_unit_sphere() -> None:
    segments = wireframe_sphere(5, 8, 12)
    assert len(segments) == 5 * 12 + 8 * 12
    assert all(
        math.sqrt(point.x**2 + point.y**2 + point.z**2) == pytest.approx(1.0)
        for segment in segments
        for point in segment
    )


def test_simplex_noise_is_bounded_and_deterministic() -> None:
    coords = np.linspace(-4.0, 4.0, 300).reshape(-1, 3)
    values = simplex3(coords)
    assert values.shape == (100,)
    assert float(values.min()) >= -1.05
    assert float(values.max()) <= 1.05
    # deterministic: same input, same output
    assert np.allclose(values, simplex3(coords))


def _ring(latitude_deg: float, longitudes: int = 96) -> np.ndarray:
    lat = math.radians(latitude_deg)
    lons = np.linspace(0.0, math.tau, longitudes, endpoint=False)
    cos_lat = math.cos(lat)
    return np.stack(
        [cos_lat * np.cos(lons), np.full(longitudes, math.sin(lat)), cos_lat * np.sin(lons)],
        axis=1,
    )


def test_wire_grid_is_unit_and_displacement_keeps_poles_stable() -> None:
    a, b = build_wire_grid(20, 32, 2)
    assert a.shape == b.shape
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    # pole_fade damps the poles: a near-pole ring must move far less than the
    # equator, so the converging meridians no longer collapse toward the centre.
    strong = DeformParams(amp=0.2, freq=2.4, wave_amp=0.18, twist=0.3, jitter=0.05)
    near_pole = np.abs(np.linalg.norm(displace_normals(_ring(87.0), 3.0, strong), axis=1) - 1.0)
    equator = np.abs(np.linalg.norm(displace_normals(_ring(0.0), 3.0, strong), axis=1) - 1.0)
    assert near_pole.mean() < equator.mean() * 0.5


def test_interrupted_transition_is_continuous_and_error_returns() -> None:
    machine = SphereStateMachine(transition_seconds=0.4, error_seconds=0.2)
    machine.set_state(SphereState.LISTENING)
    before = machine.advance(0.15)
    machine.set_state(SphereState.THINKING)
    assert machine.parameters == before
    machine.set_state(SphereState.ERROR)
    assert machine.has_pending_error
    machine.advance(0.0)
    assert machine.state is SphereState.ERROR
    machine.set_state(SphereState.SPEAKING)
    machine.advance(0.61)
    assert machine.state is SphereState.SPEAKING
    assert machine.desired_state is SphereState.SPEAKING


def test_listening_level_is_smoothed_and_speaking_waves_stop_at_two_thirds() -> None:
    machine = SphereStateMachine(transition_seconds=0.0)
    machine.set_state(SphereState.LISTENING)
    machine.set_level(1.0)
    machine.advance(0.01)
    early_level = machine.level
    machine.advance(0.2)
    # the mic level ramps smoothly (it drives deformation amplitude, not zoom)
    assert 0.0 < early_level < machine.level < 1.0
    assert machine.parameters.level_amp > 0.0
    machine.set_state(SphereState.SPEAKING)
    machine.advance(0.1)
    waves = machine.speaking_wave_progresses()
    assert len(waves) == 3
    assert all(0.0 <= progress <= 2.0 / 3.0 for progress in waves)


def test_governor_degrades_points_glow_and_frame_rate() -> None:
    governor = PerformanceGovernor(60)
    for _ in range(40):
        level = governor.record_frame(50.0)
    assert level.fps == 30
    assert level.point_ratio < 1.0
    assert not level.glow


def test_hidden_widget_stops_timer_and_degrades_without_gl(app: QApplication) -> None:
    theme = ThemeManager(app)
    widget = SphereWidget(theme, point_count=200, prefer_opengl=False)
    try:
        assert not widget.is_animation_running()
        widget.show()
        app.processEvents()
        assert widget.is_animation_running()
        for _ in range(40):
            widget.record_frame_time(50.0)
        assert widget.rendered_point_count < widget.point_count
        assert widget.backend_name == "QPainter"
        widget.hide()
        app.processEvents()
        assert not widget.is_animation_running()
    finally:
        widget.close()


def test_demo_constructs_and_closes_offscreen(app: QApplication) -> None:
    window = SphereDemoWindow(ThemeManager(app))
    try:
        assert window.sphere.point_count == 1100
        assert window.sphere.backend_name == "QPainter"
    finally:
        window.close()
        app.processEvents()


def test_renderer_clears_previous_frame(app: QApplication) -> None:
    theme = ThemeManager(app)
    widget = SphereWidget(theme, point_count=100, prefer_opengl=False)
    image = QImage(180, 180, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(theme.theme.colors.error)
    empty = np.empty((0, 4))
    axis = np.empty((0,))
    painter = QPainter(image)
    paint_scene(
        painter,
        widget,
        RenderScene(empty, axis, axis, SphereStateMachine().parameters, (), False),
        theme,
    )
    painter.end()
    try:
        assert image.pixelColor(0, 0).alpha() == 0
    finally:
        widget.close()


def test_web_state_mode_map_covers_all_states() -> None:
    from ayris.gui.widgets.sphere.web_widget import _STATE_TO_MODE

    assert set(_STATE_TO_MODE) == set(SphereState)
    assert set(_STATE_TO_MODE.values()) == {"calm", "listening", "thinking", "speaking", "error"}


def test_web_assets_are_bundled_offline() -> None:
    import ayris.gui.widgets.sphere.web_widget as web

    assets = Path(web.__file__).resolve().parent / "assets"
    html = (assets / "sphere.html").read_text(encoding="utf-8")
    # No CDN references: the vendored Three.js is what makes the widget offline.
    assert "unpkg" not in html
    assert "cdn" not in html.lower()
    assert "https://" not in html
    assert (assets / "vendor" / "three.module.js").is_file()
    assert (assets / "vendor" / "jsm" / "postprocessing" / "UnrealBloomPass.js").is_file()


def test_web_assets_served_over_loopback() -> None:
    from ayris.gui.widgets.sphere.web_widget import _ensure_server

    port = _ensure_server()
    for path in ("sphere.html", "vendor/jsm/shaders/CopyShader.js"):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/{path}", timeout=5) as resp:
            assert resp.status == 200
