"""Pure 3D geometry for the AEGIS point-cloud sphere.

Nothing here touches Qt: the cloud is built, rotated and perspective-projected
with plain maths so it can be unit-tested without a display.  The renderer only
draws the :class:`ProjectedDot` list this module returns.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

_GOLDEN_ANGLE: Final = math.pi * (3.0 - math.sqrt(5.0))
# Fixed camera tilt (~18°), looking slightly down at the top pole.  The spec
# asks for an elevated, top-forward HUD view rather than a dead-on equator.
CAMERA_TILT: Final = -0.32


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class BasePoint:
    """A unit-sphere particle with a stable per-point animation seed.

    ``phase`` desynchronises the idle pulsation so the cloud breathes instead of
    strobing; ``seed`` drives the chaotic error flicker deterministically.
    """

    x: float
    y: float
    z: float
    phase: float
    seed: float


@dataclass(frozen=True, slots=True)
class ProjectedDot:
    x: float
    y: float
    depth: float  # -1 (far) .. 1 (near) in view space
    size: float
    brightness: float  # 0 .. ~1.4, already folds depth and particle energy
    source_index: int


def point_cloud(count: int) -> tuple[BasePoint, ...]:
    """Return ``count`` near-uniform, densely packed points on a unit sphere."""

    if count < 1:
        raise ValueError("число точек должно быть положительным")
    points: list[BasePoint] = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * _GOLDEN_ANGLE
        # Deterministic, well-spread pseudo-random values from the index.
        phase = (math.sin(index * 12.9898) * 43758.5453) % 1.0
        seed = (math.sin(index * 78.233 + 1.7) * 24634.6345) % 1.0
        points.append(
            BasePoint(
                math.cos(angle) * radial,
                y,
                math.sin(angle) * radial,
                phase * math.tau,
                seed,
            )
        )
    return tuple(points)


def rotate(point: Point3D, angle_x: float, angle_y: float, angle_z: float) -> Point3D:
    """Rotate around X, then Y, then Z."""

    sin_x, cos_x = math.sin(angle_x), math.cos(angle_x)
    sin_y, cos_y = math.sin(angle_y), math.cos(angle_y)
    sin_z, cos_z = math.sin(angle_z), math.cos(angle_z)
    y = point.y * cos_x - point.z * sin_x
    z = point.y * sin_x + point.z * cos_x
    x = point.x * cos_y + z * sin_y
    z = -point.x * sin_y + z * cos_y
    return Point3D(x * cos_z - y * sin_z, x * sin_z + y * cos_z, z)


def project_dots(
    points: Iterable[tuple[Point3D, float]],
    width: float,
    height: float,
    radius: float,
    *,
    camera_distance: float = 3.4,
    base_point_size: float = 2.4,
) -> tuple[ProjectedDot, ...]:
    """Perspective-project ``(point, energy)`` pairs, farthest first.

    Front particles come out larger and brighter (perspective plus a depth
    ramp) so the flat dot field reads as a solid, glowing sphere.  ``energy`` is
    the per-particle multiplier the animation field supplies (pulsation, wave
    boosts, error flicker).
    """

    if camera_distance <= 1.0:
        raise ValueError("камера должна находиться вне единичной сферы")
    centre_x, centre_y = width / 2.0, height / 2.0
    projected: list[ProjectedDot] = []
    for index, (point, energy) in enumerate(points):
        perspective = camera_distance / (camera_distance - point.z)
        depth = max(0.0, min(1.0, (point.z + 1.0) / 2.0))
        brightness = (0.28 + 0.72 * depth) * energy
        projected.append(
            ProjectedDot(
                centre_x + point.x * radius * perspective,
                centre_y + point.y * radius * perspective,
                point.z,
                base_point_size * (0.55 + 0.85 * depth) * perspective,
                brightness,
                index,
            )
        )
    projected.sort(key=lambda dot: dot.depth)
    return tuple(projected)
