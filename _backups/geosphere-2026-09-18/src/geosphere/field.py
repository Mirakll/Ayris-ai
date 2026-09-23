"""Per-frame particle animation for the AEGIS sphere.

This module owns the *motion*: it takes the static point cloud plus the current
:class:`AnimationParams`, clock and input level and returns, for every particle,
a rotated 3D position and an energy multiplier (brightness/size).  Each state's
signature effect lives here:

* **idle** — a subtle, desynchronised per-point pulsation ("life").
* **listening** — flowing surface turbulence and small hops scaled by the mic.
* **thinking** — concentric, structured calculation rings from the top pole.
* **speaking** — fast radial sound waves from the camera-facing point, with
  particles bursting outward along the front, plus a central-zone pulse.
* **error** — chaotic per-point flicker, a squeezing shell and a darkened core
  that leaves a bright rim.

Every block is gated on its parameter, so idle stays cheap and only the active
effects pay for their trig.
"""

from __future__ import annotations

import math

from ayris.gui.widgets.geosphere.geometry import BasePoint, Point3D, rotate
from ayris.gui.widgets.geosphere.states import AnimationParams

_EPS = 1e-3
_RIPPLE_FRONTS = 3
_RADIAL_FRONTS = 3


def _ring(angle: float, front: float, width: float) -> float:
    """A soft gaussian bump centred on an expanding wavefront angle."""

    delta = (angle - front) / width
    return math.exp(-delta * delta)


def animate(
    base: tuple[BasePoint, ...],
    params: AnimationParams,
    *,
    clock: float,
    level: float,
    angle_x: float,
    angle_y: float,
    angle_z: float,
) -> list[tuple[Point3D, float]]:
    """Return ``(rotated_point, energy)`` for every particle this frame."""

    listen_gain = 0.3 + 0.7 * level
    speak_gain = 0.4 + 0.6 * level

    # Global shell squeeze/expand for the error state (applied to every point).
    contract = 1.0
    if params.contract > _EPS:
        contract = 1.0 + params.contract * 0.06 * math.sin(clock * math.tau * 2.5)

    # Pre-compute the expanding wavefront angles once per frame.
    ripple_fronts: tuple[float, ...] = ()
    if params.ripple > _EPS:
        walk = clock * (0.4 + params.ripple_speed)
        ripple_fronts = tuple(
            ((walk + k / _RIPPLE_FRONTS) % 1.0) * math.pi for k in range(_RIPPLE_FRONTS)
        )
    radial_fronts: tuple[float, ...] = ()
    if params.radial > _EPS:
        walk = clock * 1.15
        radial_fronts = tuple(
            ((walk + k / _RADIAL_FRONTS) % 1.0) * math.pi for k in range(_RADIAL_FRONTS)
        )

    core_wave = 0.5 + 0.5 * math.sin(clock * math.tau * 2.2)
    flick_step = math.floor(clock * 14.0)

    out: list[tuple[Point3D, float]] = []
    append = out.append
    for point in base:
        r = rotate(Point3D(point.x, point.y, point.z), angle_x, angle_y, angle_z)
        factor = params.radius_scale * contract
        energy = 1.0

        # Idle life: always on, keeps the cloud breathing under every state.
        breath = math.sin(clock * 0.9 + point.phase)
        factor += 0.012 * breath
        energy += 0.10 * breath

        if params.boil > _EPS:
            turb = (
                math.sin(point.x * 3.0 + clock * 1.4)
                + math.sin(point.y * 3.5 - clock * 1.1)
                + math.sin(point.z * 2.6 + clock * 1.7)
            ) / 3.0
            factor += params.boil * 0.11 * turb * listen_gain
            energy += params.boil * 0.35 * abs(turb) * listen_gain
        if params.jump > _EPS:
            hop = max(0.0, math.sin(clock * 9.0 + point.seed * math.tau))
            factor += params.jump * 0.05 * hop * listen_gain
            energy += params.jump * 0.5 * hop * listen_gain

        if ripple_fronts:
            # Concentric, deliberate rings from the top pole (object +Y).
            polar = math.acos(max(-1.0, min(1.0, point.y)))
            bump = max(_ring(polar, front, 0.14) for front in ripple_fronts)
            factor += params.ripple * 0.05 * bump
            energy += params.ripple * 0.9 * bump

        if radial_fronts:
            # Fast sound waves from the camera-facing point (view +Z).
            facing = math.acos(max(-1.0, min(1.0, r.z)))
            bump = max(_ring(facing, front, 0.18) for front in radial_fronts)
            factor += params.burst * bump * speak_gain
            energy += params.radial * 0.8 * bump * speak_gain
        if params.core_pulse > _EPS:
            front_mask = max(0.0, r.z)
            front_mask *= front_mask
            factor += params.core_pulse * core_wave * front_mask * speak_gain
            energy += params.core_pulse * 1.4 * core_wave * front_mask * speak_gain

        if params.flicker > _EPS:
            blink = (math.sin(point.seed * 133.7 + flick_step * 8.4) * 43758.0) % 1.0
            energy *= 1.0 - params.flicker * 0.65 + params.flicker * 0.65 * (0.3 + 1.2 * blink)
            jitter = (math.sin(point.seed * 71.3 + clock * 37.0) * 0.5) * params.flicker
            factor += 0.03 * jitter
            # Darken the camera-facing core; keep the silhouette rim aglow.
            front_mask = max(0.0, r.z)
            energy *= 1.0 - params.flicker * 0.7 * front_mask * front_mask
            energy += params.flicker * 0.45 * (1.0 - abs(r.z))

        append((Point3D(r.x * factor, r.y * factor, r.z * factor), max(0.0, energy)))
    return out
