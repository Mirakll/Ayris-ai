"""Pure animation state machine and performance policy for the sphere."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class SphereState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class AnimationParams:
    """Deformation + colour controls for one state (mirrors the HTML prototype).

    All fields are plain floats so interpolation between states is a per-field
    lerp.  Time-based effects (breath, shimmer, mic level) are folded in by the
    widget, which owns the clock; the machine only carries their strengths.
    """

    amp: float  # simplex displacement amplitude
    freq: float  # simplex frequency (surface detail)
    speed: float  # how fast the surface evolves
    wave_amp: float  # travelling bottom-up wave amplitude
    wave_freq: float
    twist: float  # spiral twist around the vertical axis
    jitter: float  # high-frequency per-vertex tremor (error)
    spin: float  # steady rotation speed around Y
    breath_amp: float  # whole-sphere breathing scale
    breath_hz: float
    level_amp: float  # extra amplitude driven by mic level (listening)
    waves: float  # speaking edge-wave intensity (0..1)
    shimmer_amt: float  # colour shimmer strength (listening)
    shimmer_hz: float
    error: float  # red/orange tint weight (error)

    def interpolated(self, other: AnimationParams, amount: float) -> AnimationParams:
        a = amount
        return AnimationParams(
            _lerp(self.amp, other.amp, a),
            _lerp(self.freq, other.freq, a),
            _lerp(self.speed, other.speed, a),
            _lerp(self.wave_amp, other.wave_amp, a),
            _lerp(self.wave_freq, other.wave_freq, a),
            _lerp(self.twist, other.twist, a),
            _lerp(self.jitter, other.jitter, a),
            _lerp(self.spin, other.spin, a),
            _lerp(self.breath_amp, other.breath_amp, a),
            _lerp(self.breath_hz, other.breath_hz, a),
            _lerp(self.level_amp, other.level_amp, a),
            _lerp(self.waves, other.waves, a),
            _lerp(self.shimmer_amt, other.shimmer_amt, a),
            _lerp(self.shimmer_hz, other.shimmer_hz, a),
            _lerp(self.error, other.error, a),
        )


# Tuned in the HTML prototype (design/sphere_ref): calm/listening/thinking/
# speaking/error.  Speaking and error deliberately do NOT scale the sphere —
# only the surface reacts.  Listening shimmers toward violet instead of zooming.
STATE_PARAMS: Final[dict[SphereState, AnimationParams]] = {
    # fields: amp freq speed wave_amp wave_freq twist jitter spin breath_amp
    #         breath_hz level_amp waves shimmer_amt shimmer_hz error
    SphereState.IDLE: AnimationParams(
        0.048, 1.6, 0.16, 0.042, 1.9, 0.12, 0.0, 0.075, 0.018, 0.28, 0.0, 0.0, 0.0, 0.0, 0.0
    ),
    SphereState.LISTENING: AnimationParams(
        0.050, 2.4, 0.50, 0.035, 3.0, 0.05, 0.0, 0.08, 0.0, 0.0, 0.06, 0.0, 0.75, 0.32, 0.0
    ),
    SphereState.THINKING: AnimationParams(
        0.050, 1.9, 0.34, 0.030, 2.3, 0.28, 0.0, 0.22, 0.020, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0
    ),
    SphereState.SPEAKING: AnimationParams(
        0.100, 2.0, 0.95, 0.095, 2.2, 0.06, 0.0, 0.12, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
    ),
    SphereState.ERROR: AnimationParams(
        0.080, 2.0, 0.45, 0.040, 2.4, 0.04, 0.022, 0.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0
    ),
}


def _lerp(start: float, end: float, amount: float) -> float:
    return start + (end - start) * amount


def _smoothstep(amount: float) -> float:
    bounded = max(0.0, min(1.0, amount))
    return bounded * bounded * (3.0 - 2.0 * bounded)


class SphereStateMachine:
    """Interruptible transitions with a priority, self-clearing Error state."""

    def __init__(self, *, transition_seconds: float = 0.32, error_seconds: float = 0.62) -> None:
        if transition_seconds < 0.0 or error_seconds < 0.0:
            raise ValueError("длительности анимации не могут быть отрицательными")
        self.transition_seconds = transition_seconds
        self.error_seconds = error_seconds
        self._state = SphereState.IDLE
        self._desired = SphereState.IDLE
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
    def state(self) -> SphereState:
        return self._state

    @property
    def desired_state(self) -> SphereState:
        return self._desired

    @property
    def parameters(self) -> AnimationParams:
        return self._animated_parameters()

    @property
    def level(self) -> float:
        return self._level

    @property
    def has_pending_error(self) -> bool:
        return self._pending_error

    def set_state(self, state: SphereState | str) -> None:
        requested = SphereState(state)
        if requested is SphereState.ERROR:
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
            self._begin_transition(SphereState.ERROR)
        self._advance_transition(seconds)
        if self._in_error:
            self._error_elapsed += seconds
            total = self.transition_seconds + self.error_seconds
            if self._error_elapsed >= total:
                self._in_error = False
                self._begin_transition(self._desired)
        return self._animated_parameters()

    def speaking_wave_progresses(self, count: int = 3) -> tuple[float, ...]:
        if self._params.waves <= 0.001 or count <= 0:
            return ()
        return tuple(
            min(2.0 / 3.0, ((self._clock * 0.72 + index / count) % 1.0) * (2.0 / 3.0))
            for index in range(count)
        )

    def _begin_transition(self, state: SphereState) -> None:
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
        # Time-based effects (breath, shimmer, mic level) are applied by the
        # widget, which owns the render clock; the machine only interpolates.
        return self._params


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
            PerformanceLevel(min(30, self.target_fps), 0.72, False),
            PerformanceLevel(min(30, self.target_fps), 0.48, False),
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
