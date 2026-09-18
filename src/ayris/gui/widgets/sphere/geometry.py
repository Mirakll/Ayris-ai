"""Pure 3D geometry for the wireframe assistant sphere.

The visible sphere is a latitude/longitude wire grid whose vertices are pushed
along their normal by 3D simplex noise plus a travelling wave — the same maths as
the tuned HTML prototype in ``design/sphere_ref``.  Everything here is plain data
(NumPy arrays / dataclasses) so it can be unit-tested without a Qt context.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray

_GOLDEN_ANGLE: Final = math.pi * (3.0 - math.sqrt(5.0))


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class ProjectedPoint:
    x: float
    y: float
    depth: float
    size: float
    opacity: float
    source_index: int


def fibonacci_sphere(count: int) -> tuple[Point3D, ...]:
    """Return ``count`` near-uniform points on a unit sphere."""

    if count < 1:
        raise ValueError("число точек должно быть положительным")
    points: list[Point3D] = []
    for index in range(count):
        y = 1.0 - 2.0 * (index + 0.5) / count
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        angle = index * _GOLDEN_ANGLE
        points.append(Point3D(math.cos(angle) * radial, y, math.sin(angle) * radial))
    return tuple(points)


def wireframe_sphere(
    latitude_rings: int = 17, longitude_lines: int = 34, steps: int = 48
) -> tuple[tuple[Point3D, Point3D], ...]:
    """Return continuous latitude and longitude strokes on a unit sphere."""

    if latitude_rings < 1 or longitude_lines < 3 or steps < 4:
        raise ValueError("сетка сферы слишком редкая")
    segments: list[tuple[Point3D, Point3D]] = []

    def point(theta: float, phi: float) -> Point3D:
        radial = math.sin(theta)
        return Point3D(radial * math.cos(phi), math.cos(theta), radial * math.sin(phi))

    for ring in range(1, latitude_rings + 1):
        theta = math.pi * ring / (latitude_rings + 1)
        for step in range(steps):
            phi = math.tau * step / steps
            next_phi = math.tau * (step + 1) / steps
            segments.append((point(theta, phi), point(theta, next_phi)))
    for line in range(longitude_lines):
        phi = math.tau * line / longitude_lines
        for step in range(steps):
            theta = math.pi * step / steps
            next_theta = math.pi * (step + 1) / steps
            segments.append((point(theta, phi), point(next_theta, phi)))
    return tuple(segments)


def rotate_point(point: Point3D, angle_x: float, angle_y: float, angle_z: float) -> Point3D:
    """Rotate a point around X, Y and Z, in that order."""

    sin_x, cos_x = math.sin(angle_x), math.cos(angle_x)
    sin_y, cos_y = math.sin(angle_y), math.cos(angle_y)
    sin_z, cos_z = math.sin(angle_z), math.cos(angle_z)
    y = point.y * cos_x - point.z * sin_x
    z = point.y * sin_x + point.z * cos_x
    x = point.x * cos_y + z * sin_y
    z = -point.x * sin_y + z * cos_y
    return Point3D(x * cos_z - y * sin_z, x * sin_z + y * cos_z, z)


def rotate_points(
    points: Iterable[Point3D], angle_x: float, angle_y: float, angle_z: float
) -> tuple[Point3D, ...]:
    return tuple(rotate_point(point, angle_x, angle_y, angle_z) for point in points)


def project_points(
    points: Iterable[Point3D],
    width: float,
    height: float,
    radius: float,
    *,
    camera_distance: float = 3.2,
    base_point_size: float = 2.1,
) -> tuple[ProjectedPoint, ...]:
    """Perspective-project points, farthest first for alpha compositing."""

    if camera_distance <= 1.0:
        raise ValueError("камера должна находиться вне единичной сферы")
    centre_x, centre_y = width / 2.0, height / 2.0
    projected: list[ProjectedPoint] = []
    for index, point in enumerate(points):
        perspective = camera_distance / (camera_distance - point.z)
        depth = max(0.0, min(1.0, (point.z + 1.0) / 2.0))
        projected.append(
            ProjectedPoint(
                centre_x + point.x * radius * perspective,
                centre_y + point.y * radius * perspective,
                point.z,
                base_point_size * (0.55 + 0.75 * depth) * perspective,
                0.20 + 0.80 * depth,
                index,
            )
        )
    projected.sort(key=lambda item: item.depth)
    return tuple(projected)


# ---------------------------------------------------------------------------
# NumPy wireframe path — the hero visual (deforming lat/long grid).
# ---------------------------------------------------------------------------

Float = NDArray[np.float64]
_Vec3 = tuple[float, float, float]


def _permute(x: Float) -> Float:
    return np.mod(((x * 34.0) + 1.0) * x, 289.0)


def _taylor_inv_sqrt(r: Float) -> Float:
    return 1.79284291400159 - 0.85373472095314 * r


def simplex3(coords: Float) -> Float:
    """Vectorised 3D simplex noise (Ashima/Gustavson port), range ~[-1, 1].

    ``coords`` is ``(N, 3)``; returns ``(N,)``.  Same algorithm as the GLSL
    ``snoise`` used in the prototype, so the deformed shape matches.
    """

    v = np.asarray(coords, dtype=np.float64)
    cx, cy = 1.0 / 6.0, 1.0 / 3.0
    i = np.floor(v + (v.sum(axis=1, keepdims=True) * cy))
    x0 = v - i + (i.sum(axis=1, keepdims=True) * cx)

    # order the four simplex corners
    g = (x0[:, [0, 1, 2]] >= x0[:, [1, 2, 0]]).astype(np.float64)
    lc = 1.0 - g
    i1 = np.minimum(g, lc[:, [2, 0, 1]])
    i2 = np.maximum(g, lc[:, [2, 0, 1]])
    x1 = x0 - i1 + cx
    x2 = x0 - i2 + 2.0 * cx
    x3 = x0 - 1.0 + 3.0 * cx

    i = np.mod(i, 289.0)
    zeros, ones = np.zeros(len(v)), np.ones(len(v))

    def corner(axis: int) -> Float:
        return np.stack([zeros, i1[:, axis], i2[:, axis], ones], 1)

    p = _permute(
        _permute(_permute(i[:, 2:3] + corner(2)) + i[:, 1:2] + corner(1)) + i[:, 0:1] + corner(0)
    )

    ns_x, ns_y, ns_z = 2.0 / 7.0, 0.5 / 7.0 - 1.0, 1.0 / 7.0  # n_*D.wyz - D.xzx
    j = p - 49.0 * np.floor(p * ns_z * ns_z)
    x_ = np.floor(j * ns_z)
    y_ = np.floor(j - 7.0 * x_)
    x = x_ * ns_x + ns_y
    y = y_ * ns_x + ns_y
    h = 1.0 - np.abs(x) - np.abs(y)

    b0 = np.concatenate([x[:, :2], y[:, :2]], axis=1)
    b1 = np.concatenate([x[:, 2:], y[:, 2:]], axis=1)
    s0 = np.floor(b0) * 2.0 + 1.0
    s1 = np.floor(b1) * 2.0 + 1.0
    sh = -(h <= 0.0).astype(np.float64)
    a0 = b0[:, [0, 2, 1, 3]] + s0[:, [0, 2, 1, 3]] * sh[:, [0, 0, 1, 1]]
    a1 = b1[:, [0, 2, 1, 3]] + s1[:, [0, 2, 1, 3]] * sh[:, [2, 2, 3, 3]]

    p0 = np.stack([a0[:, 0], a0[:, 1], h[:, 0]], axis=1)
    p1 = np.stack([a0[:, 2], a0[:, 3], h[:, 1]], axis=1)
    p2 = np.stack([a1[:, 0], a1[:, 1], h[:, 2]], axis=1)
    p3 = np.stack([a1[:, 2], a1[:, 3], h[:, 3]], axis=1)

    norm = _taylor_inv_sqrt(
        np.stack([(p0 * p0).sum(1), (p1 * p1).sum(1), (p2 * p2).sum(1), (p3 * p3).sum(1)], axis=1)
    )
    p0 *= norm[:, 0:1]
    p1 *= norm[:, 1:2]
    p2 *= norm[:, 2:3]
    p3 *= norm[:, 3:4]

    m = np.maximum(
        0.6 - np.stack([(x0 * x0).sum(1), (x1 * x1).sum(1), (x2 * x2).sum(1), (x3 * x3).sum(1)], 1),
        0.0,
    )
    m = m * m
    dots = np.stack(
        [(p0 * x0).sum(1), (p1 * x1).sum(1), (p2 * x2).sum(1), (p3 * x3).sum(1)], axis=1
    )
    return cast(Float, 42.0 * (m * m * dots).sum(axis=1))


def build_wire_grid(
    latitude_rings: int = 28, longitude_lines: int = 44, subdivisions: int = 2
) -> tuple[Float, Float]:
    """Build a lat/long wire grid on the unit sphere.

    Returns ``(a, b)`` — two ``(S, 3)`` arrays of unit normals, one per segment
    endpoint.  Each edge is split into ``subdivisions`` pieces so lines bend
    smoothly once the surface is displaced.
    """

    if latitude_rings < 2 or longitude_lines < 3 or subdivisions < 1:
        raise ValueError("сетка сферы слишком редкая")

    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []

    def point(lat: float, lon: float) -> tuple[float, float, float]:
        cos_lat = math.cos(lat)
        return (cos_lat * math.cos(lon), math.sin(lat), cos_lat * math.sin(lon))

    def lerp(a: _Vec3, b: _Vec3, t: float) -> _Vec3:
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)

    def emit(a: _Vec3, b: _Vec3) -> None:
        for step in range(subdivisions):
            starts.append(lerp(a, b, step / subdivisions))
            ends.append(lerp(a, b, (step + 1) / subdivisions))

    for ring in range(1, latitude_rings):
        lat = -math.pi / 2.0 + math.pi * ring / latitude_rings
        for col in range(longitude_lines):
            lon0 = math.tau * col / longitude_lines
            lon1 = math.tau * (col + 1) / longitude_lines
            emit(point(lat, lon0), point(lat, lon1))
    for col in range(longitude_lines):
        lon = math.tau * col / longitude_lines
        for ring in range(latitude_rings):
            lat0 = -math.pi / 2.0 + math.pi * ring / latitude_rings
            lat1 = -math.pi / 2.0 + math.pi * (ring + 1) / latitude_rings
            emit(point(lat0, lon), point(lat1, lon))

    a = np.array(starts, dtype=np.float64)
    b = np.array(ends, dtype=np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a, b


@dataclass(frozen=True, slots=True)
class DeformParams:
    """Per-frame deformation controls (mirror of the HTML prototype uniforms)."""

    amp: float = 0.05
    freq: float = 1.6
    speed: float = 0.16
    wave_amp: float = 0.04
    wave_freq: float = 1.9
    twist: float = 0.0
    jitter: float = 0.0
    breath: float = 0.0


def _smoothstep_arr(edge0: float, edge1: float, x: Float) -> Float:
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def displace_normals(normals: Float, time: float, params: DeformParams) -> Float:
    """Return displaced positions (``(N, 3)``, ~unit radius) for unit normals."""

    n = np.asarray(normals, dtype=np.float64)
    lat = n[:, 1]

    # twist around the vertical axis, stronger toward the poles
    ang = params.twist * lat
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    nx = n[:, 0] * cos_a - n[:, 2] * sin_a
    nz = n[:, 0] * sin_a + n[:, 2] * cos_a
    rotated = np.stack([nx, n[:, 1], nz], axis=1)

    # Smooth, low-frequency flow field (a cheap multi-directional sine sum).  It
    # reads like the prototype's simplex displacement — soft rolling hills — but
    # costs a handful of vectorised sines instead of a full simplex per frame,
    # which is what keeps the always-on-top overlay at 60 FPS.
    t = time * params.speed
    q = rotated * params.freq
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    noise = (
        np.sin(qx + 1.7 * qy - 1.3 * qz + t)
        + np.sin(1.3 * qy + 2.1 * qz - 0.9 * qx + t * 0.8)
        + np.sin(1.1 * qz + 2.3 * qx - 1.5 * qy + t * 1.2)
    ) * (1.0 / 3.0)

    wave = np.sin(lat * params.wave_freq - time * 1.6)
    bottom = _smoothstep_arr(0.2, -0.75, lat)
    disp = params.amp * noise + params.wave_amp * wave * (0.5 + 0.5 * bottom)

    if params.jitter > 0.0:
        rx, ry, rz = rotated[:, 0], rotated[:, 1], rotated[:, 2]
        tremor = np.sin(rx * 17.0 + ry * 23.0 + rz * 19.0 + time * 9.0)
        disp += params.jitter * tremor * 0.6

    pole_fade = 1.0 - np.abs(lat) ** 3
    disp *= 0.4 + 0.6 * pole_fade

    radius = (1.0 + disp) * (1.0 + params.breath)
    return cast(Float, rotated * radius[:, None])


def rotate_array(positions: Float, angle_x: float, angle_y: float, angle_z: float) -> Float:
    """Rotate ``(N, 3)`` points around X, then Y, then Z."""

    p = np.asarray(positions, dtype=np.float64)
    sx, cx = math.sin(angle_x), math.cos(angle_x)
    sy, cy = math.sin(angle_y), math.cos(angle_y)
    sz, cz = math.sin(angle_z), math.cos(angle_z)
    y = p[:, 1] * cx - p[:, 2] * sx
    z = p[:, 1] * sx + p[:, 2] * cx
    x = p[:, 0] * cy + z * sy
    z = -p[:, 0] * sy + z * cy
    return np.stack([x * cz - y * sz, x * sz + y * cz, z], axis=1)


def project_array(
    positions: Float, width: float, height: float, radius: float, *, camera_distance: float = 3.2
) -> tuple[Float, Float, Float]:
    """Perspective-project ``(N, 3)`` points; returns ``(x, y, depth)`` arrays.

    ``depth`` is normalised 0 (far) → 1 (near) for alpha/width falloff.
    """

    p = np.asarray(positions, dtype=np.float64)
    perspective = camera_distance / (camera_distance - p[:, 2])
    x = width / 2.0 + p[:, 0] * radius * perspective
    y = height / 2.0 + p[:, 1] * radius * perspective
    depth = np.clip((p[:, 2] + 1.0) / 2.0, 0.0, 1.0)
    return x, y, depth
