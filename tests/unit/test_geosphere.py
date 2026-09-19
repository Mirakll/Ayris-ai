"""Geometry, animation field, state machine and lifecycle for the AEGIS sphere."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.geosphere.demo import GeoSphereDemoWindow
from ayris.gui.widgets.geosphere.field import animate
from ayris.gui.widgets.geosphere.geometry import Point3D, point_cloud, project_dots, rotate
from ayris.gui.widgets.geosphere.palette import PaletteName, get_palette
from ayris.gui.widgets.geosphere.renderer import RenderScene, paint_scene
from ayris.gui.widgets.geosphere.states import (
    STATE_PARAMS,
    AnimationParams,
    GeoState,
    GeoStateMachine,
    PerformanceGovernor,
)
from ayris.gui.widgets.geosphere.widget import GeoSphereWidget


@pytest.fixture(scope="module")
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    application.processEvents()


def _norm(point: Point3D) -> float:
    return math.sqrt(point.x**2 + point.y**2 + point.z**2)


def test_point_cloud_is_dense_uniform_and_seeded() -> None:
    points = point_cloud(1200)
    assert len(points) == 1200
    radial_error = max(abs(_norm(Point3D(p.x, p.y, p.z)) - 1.0) for p in points)
    centre_error = max(abs(sum(getattr(p, axis) for p in points) / len(points)) for axis in "xyz")
    assert radial_error < 1e-12
    assert centre_error < 0.002
    # Seeds must differ so the flicker and pulsation do not strobe in lockstep.
    assert len({round(p.phase, 6) for p in points}) > 1000


def test_rotation_preserves_radius_and_projection_sorts_depth() -> None:
    point = rotate(Point3D(1.0, 0.0, 0.0), 0.2, 0.7, -0.3)
    assert _norm(point) == pytest.approx(1.0)
    dots = project_dots(
        ((Point3D(0.0, 0.0, 1.0), 1.0), (Point3D(0.0, 0.0, -1.0), 1.0)), 200, 200, 70
    )
    assert [dot.depth for dot in dots] == [-1.0, 1.0]
    # Near particle (depth +1) is larger and brighter than the far one.
    assert dots[0].size < dots[1].size
    assert dots[0].brightness < dots[1].brightness


def test_idle_breathes_and_thinking_rings_spread_energy() -> None:
    base = point_cloud(400)
    idle = animate(
        base,
        STATE_PARAMS[GeoState.IDLE],
        clock=1.3,
        level=0.0,
        angle_x=-0.32,
        angle_y=0.4,
        angle_z=0.0,
    )
    energies = [energy for _point, energy in idle]
    assert max(energies) - min(energies) > 0.05  # subtle life, not a flat field

    thinking = animate(
        base,
        STATE_PARAMS[GeoState.THINKING],
        clock=1.3,
        level=0.0,
        angle_x=-0.32,
        angle_y=0.4,
        angle_z=0.0,
    )
    ring_energy = max(energy for _point, energy in thinking)
    assert ring_energy > 1.3  # concentric rings brighten a band of particles


def test_speaking_bursts_outward_and_error_flickers() -> None:
    base = point_cloud(500)
    speaking = animate(
        base,
        STATE_PARAMS[GeoState.SPEAKING],
        clock=0.6,
        level=1.0,
        angle_x=-0.32,
        angle_y=0.0,
        angle_z=0.0,
    )
    assert max(_norm(point) for point, _energy in speaking) > 1.05  # radial burst

    error = animate(
        base,
        STATE_PARAMS[GeoState.ERROR],
        clock=0.6,
        level=0.0,
        angle_x=-0.32,
        angle_y=0.0,
        angle_z=0.0,
    )
    energies = [energy for _point, energy in error]
    assert max(energies) - min(energies) > 0.5  # chaotic per-point flicker


def test_interrupted_transition_is_continuous_and_error_returns() -> None:
    machine = GeoStateMachine(transition_seconds=0.4, error_seconds=0.2)
    machine.set_state(GeoState.LISTENING)
    before = machine.advance(0.15)
    machine.set_state(GeoState.THINKING)
    assert machine.parameters == before
    machine.set_state(GeoState.ERROR)
    assert machine.has_pending_error
    machine.advance(0.0)
    assert machine.state is GeoState.ERROR
    machine.set_state(GeoState.SPEAKING)
    machine.advance(0.61)
    assert machine.state is GeoState.SPEAKING
    assert machine.desired_state is GeoState.SPEAKING


def test_listening_level_is_smoothed_into_the_pulse() -> None:
    machine = GeoStateMachine(transition_seconds=0.0)
    machine.set_state(GeoState.LISTENING)
    machine.set_level(1.0)
    first = machine.advance(0.01)
    later = machine.advance(0.2)
    assert 0.0 < machine.level < 1.0
    assert later.radius_scale != first.radius_scale


def test_params_interpolation_covers_every_field() -> None:
    blended = AnimationParams().interpolated(STATE_PARAMS[GeoState.ERROR], 0.5)
    assert blended.error == pytest.approx(0.5)
    assert blended.flash == pytest.approx(0.5)


def test_palettes_are_distinct() -> None:
    assert get_palette(PaletteName.CYAN).base != get_palette(PaletteName.GOLD).base


def test_governor_degrades_points_and_frame_rate() -> None:
    governor = PerformanceGovernor(60)
    level = governor.level
    for _ in range(40):
        level = governor.record_frame(50.0)
    assert level.fps == 30
    assert level.point_ratio < 1.0


def test_hidden_widget_stops_timer_and_switches_palette(app: QApplication) -> None:
    theme = ThemeManager(app)
    widget = GeoSphereWidget(theme, point_count=300, prefer_opengl=False)
    try:
        assert not widget.is_animation_running()
        widget.show()
        app.processEvents()
        assert widget.is_animation_running()
        assert widget.backend_name == "QPainter"
        widget.set_palette(PaletteName.GOLD)
        assert widget.palette_name == get_palette(PaletteName.GOLD).base
        for _ in range(40):
            widget.record_frame_time(50.0)
        assert widget.rendered_point_count < widget.point_count
        widget.hide()
        app.processEvents()
        assert not widget.is_animation_running()
    finally:
        widget.close()


def test_demo_constructs_and_closes_offscreen(app: QApplication) -> None:
    window = GeoSphereDemoWindow(ThemeManager(app))
    try:
        assert window.sphere.point_count == 1800
        assert window.sphere.backend_name == "QPainter"
    finally:
        window.close()
        app.processEvents()


def test_renderer_clears_previous_frame(app: QApplication) -> None:
    theme = ThemeManager(app)
    widget = GeoSphereWidget(theme, point_count=120, prefer_opengl=False)
    image = QImage(180, 180, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(theme.theme.colors.error)
    painter = QPainter(image)
    paint_scene(
        painter,
        widget,
        RenderScene((), GeoStateMachine().parameters, get_palette(PaletteName.CYAN), False),
        theme,
    )
    painter.end()
    try:
        assert image.pixelColor(0, 0).alpha() == 0
    finally:
        widget.close()
