"""Animation state machine and performance policy for the AEGIS sphere.

Pure logic, no Qt: :class:`GeoStateMachine` turns a requested state plus a live
input level into an interpolated :class:`AnimationParams` bundle that the field
and renderer consume.  Error is a priority state that always plays out fully
before returning to whatever was desired underneath it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Final


class GeoState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AnimationParams:
    radius_scale: float = 1.0
    rotation_speed: float = 0.16
    irregular: float = 0.0  # thinking: uneven, redirected spin
    pulse: float = 0.0  # listening: radius throb scaled by level
    boil: float = 0.0  # listening: flowing surface turbulence
    jump: float = 0.0  # listening: per-point radial hop
    ripple: float = 0.0  # thinking: concentric calculation rings
    ripple_speed: float = 0.0
    radial: float = 0.0  # speaking: fast sound waves, front-sourced
    burst: float = 0.0  # speaking: outward displacement along a wave
    core_pulse: float = 0.0  # speaking: equator zone pushes outward
    flash: float = 0.0  # error: crimson radial flash
    shake: float = 0.0  # error: whole-body jitter
    contract: float = 0.0  # error: shell squeeze/expand
    flicker: float = 0.0  # error: chaotic per-point blink
    error: float = 0.0  # colour blend toward the alarm red

    def interpolated(self, other: AnimationParams, amount: float) -> AnimationParams:
        values = {
            spec.name: _lerp(getattr(self, spec.name), getattr(other, spec.name), amount)
            for spec in fields(self)
        }
        return AnimationParams(**values)


STATE_PARAMS: Final[dict[GeoState, AnimationParams]] = {
    GeoState.IDLE: AnimationParams(rotation_speed=0.16),
    GeoState.LISTENING: AnimationParams(rotation_speed=0.30, pulse=0.14, boil=0.10, jump=0.055),
    GeoState.THINKING: AnimationParams(
        rotation_speed=0.9, irregular=1.0, ripple=1.0, ripple_speed=0.34
    ),
    GeoState.SPEAKING: AnimationParams(
        rotation_speed=0.34, radial=1.0, burst=0.09, core_pulse=0.12
    ),
    GeoState.ERROR: AnimationParams(
        radius_scale=1.02,
        rotation_speed=0.06,
        flash=1.0,
        shake=1.0,
        contract=1.0,
        flicker=1.0,
        error=1.0,
    ),
}


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _smoothstep(amount: float) -> float:
    bounded = max(0.0, min(1.0, amount))
    return bounded * bounded * (3.0 - 2.0 * bounded)


class GeoStateMachine:
    """Interruptible transitions with a priority, self-clearing Error state."""

    def __init__(self, *, transition_seconds: float = 0.34, error_seconds: float = 0.7) -> None:
        if transition_seconds < 0.0 or error_seconds < 0.0:
            raise ValueError("длительности анимации не могут быть отрицательными")
        self.transition_seconds = transition_seconds
        self.error_seconds = error_seconds
        self._state = GeoState.IDLE
        self._desired = GeoState.IDLE
        self._start = STATE_PARAMS[self._state]
        self._params = self._start
        self._target = self._start
        self._transition_elapsed = transition_seconds
        self._error_elapsed = 0.0
        self._in_error = False
        self._pending_error = False
        self._level_target = 0.0
        self._level = 0.0
        self._clock = 0.0

    @property
    def state(self) -> GeoState:
        return self._state

    @property
    def desired_state(self) -> GeoState:
        return self._desired

    @property
    def parameters(self) -> AnimationParams:
        return self._animated_parameters()

    @property
    def level(self) -> float:
        return self._level

    @property
    def clock(self) -> float:
        return self._clock

    @property
    def has_pending_error(self) -> bool:
        return self._pending_error

    def set_state(self, state: GeoState | str) -> None:
        requested = GeoState(state)
        if requested is GeoState.ERROR:
            if not self._in_error:
                self._pending_error = True
            return
        self._desired = requested
        if not self._in_error:
            self._begin_transition(requested)

    def set_level(self, level: float) -> None:
        self._level_target = max(0.0, min(1.0, float(level)))

    def advance(self, seconds: float) -> AnimationParams:
        seconds = max(0.0, seconds)
        self._clock += seconds
        smoothing = 1.0 - math.exp(-seconds / 0.09) if seconds else 0.0
        self._level += (self._level_target - self._level) * smoothing
        if self._pending_error and not self._in_error:
            self._pending_error = False
            self._in_error = True
            self._error_elapsed = 0.0
            self._begin_transition(GeoState.ERROR)
        self._advance_transition(seconds)
        if self._in_error:
            self._error_elapsed += seconds
            total = self.transition_seconds + self.error_seconds
            if self._error_elapsed >= total:
                self._in_error = False
                self._begin_transition(self._desired)
        return self._animated_parameters()

    def _begin_transition(self, state: GeoState) -> None:
        live = self._interpolated_base()
        self._start = live
        self._params = live
        self._target = STATE_PARAMS[state]
        self._state = state
        self._transition_elapsed = 0.0
        if self.transition_seconds == 0.0:
            self._params = self._target

    def _advance_transition(self, seconds: float) -> None:
        if self._transition_elapsed >= self.transition_seconds:
            self._params = self._target
            return
        self._transition_elapsed = min(self.transition_seconds, self._transition_elapsed + seconds)
        self._params = self._interpolated_base()

    def _interpolated_base(self) -> AnimationParams:
        if self.transition_seconds == 0.0:
            return self._target
        amount = _smoothstep(self._transition_elapsed / self.transition_seconds)
        return self._start.interpolated(self._target, amount)

    def _animated_parameters(self) -> AnimationParams:
        params = self._params
        if params.pulse <= 0.0:
            return params
        # Listening throb rides the smoothed input level, so a silent mic sits
        # still and a loud one visibly swells.
        throb = (0.6 + 0.4 * math.sin(self._clock * math.tau * 1.6)) * self._level
        return replace(params, radius_scale=params.radius_scale * (1.0 + params.pulse * throb))


@dataclass(frozen=True, slots=True)
class PerformanceLevel:
    fps: int
    point_ratio: float
    glow: bool


class PerformanceGovernor:
    """Degrade conservatively after sustained slow frames and recover slowly."""

    def __init__(self, target_fps: int = 60) -> None:
        self.target_fps = max(15, min(144, target_fps))
        self._stage = 0
        self._slow_frames = 0
        self._fast_frames = 0
        self._average_ms = 1000.0 / self.target_fps

    @property
    def level(self) -> PerformanceLevel:
        levels = (
            PerformanceLevel(self.target_fps, 1.0, True),
            PerformanceLevel(min(30, self.target_fps), 0.68, True),
            PerformanceLevel(min(30, self.target_fps), 0.45, False),
        )
        return levels[self._stage]

    @property
    def average_frame_ms(self) -> float:
        return self._average_ms

    def record_frame(self, frame_ms: float) -> PerformanceLevel:
        frame_ms = max(0.0, frame_ms)
        self._average_ms = self._average_ms * 0.90 + frame_ms * 0.10
        budget = 1000.0 / self.level.fps
        if self._average_ms > budget * 1.30:
            self._slow_frames += 1
            self._fast_frames = 0
            if self._slow_frames >= 18 and self._stage < 2:
                self._stage += 1
                self._slow_frames = 0
        elif self._average_ms < budget * 0.72:
            self._fast_frames += 1
            self._slow_frames = 0
            if self._fast_frames >= 240 and self._stage > 0:
                self._stage -= 1
                self._fast_frames = 0
        else:
            self._slow_frames = 0
            self._fast_frames = 0
        return self.level
