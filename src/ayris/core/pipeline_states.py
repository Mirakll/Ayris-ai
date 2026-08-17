"""The stages one request goes through, and the clock that limits them.

:class:`ayris.core.state.AssistantState` is what the overlay animates — four
looks and an error. This module is the finer machine underneath it: the seven
stages of a single pass through the dispatcher, from the wake word to the last
sound of the answer. Two machines rather than one because they answer different
questions. The sphere does not care whether Ayris is waiting for the cloud or
for the matcher, while the trace, DevTools and the timeouts care about nothing
else. :data:`ASSISTANT_STATES` maps one onto the other so the pipeline can drive
both from a single transition.

Every stage carries its own deadline. :class:`PipelineTimeouts` reads them from
the settings the user already sees — the listening window, the maximum utterance
length, the cloud timeout, the model timeout — instead of inventing a second set
of numbers next to them; only the two stages the settings have no field for
(``executing`` and ``responding``) fall back to the constants below.

Waiting is delegated to a :class:`Scheduler`, which exists so that the timeout
tests do not sleep. :class:`ThreadScheduler` is the real one, a
``threading.Timer`` per deadline; a test passes a manual scheduler and fires the
deadline itself, which is the only way the millisecond timeouts of the task file
can be asserted without the windows-latest scheduler making the test flake.

The module imports :mod:`ayris.core.events` and :mod:`ayris.core.state` and
never the other way round: :class:`ayris.core.events.PipelineStateChanged`
annotates :class:`PipelineState` under ``TYPE_CHECKING`` only.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol

from ayris.core.events import EventBus, PipelineStateChanged
from ayris.core.state import AssistantState
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.config import Settings

__all__ = [
    "ALLOWED_TRANSITIONS",
    "ASSISTANT_STATES",
    "BUSY_STATES",
    "EXECUTING_TIMEOUT_SEC",
    "RESPONDING_TIMEOUT_SEC",
    "STATE_LABELS",
    "STT_TIMEOUT_FACTOR",
    "STT_TIMEOUT_FLOOR_SEC",
    "ManualScheduler",
    "PipelineState",
    "PipelineStateMachine",
    "PipelineTimeouts",
    "Scheduler",
    "ThreadScheduler",
    "Timer",
]

_log = get_logger(__name__)


class PipelineState(StrEnum):
    """Where a request is right now.

    The happy path is ``IDLE → LISTENING → RECORDING → TRANSCRIBING →
    UNDERSTANDING → EXECUTING → RESPONDING → IDLE``, and every shortcut through
    it is a legal move in :data:`ALLOWED_TRANSITIONS`: a text command skips the
    audio stages, a phrase that matched nothing skips ``EXECUTING``, an action
    with nothing to say skips ``RESPONDING``.

    There is no ``ERROR`` member on purpose. A failed stage speaks its message
    and returns to ``IDLE``; the error *look* is a property of the overlay and
    comes from :meth:`ayris.core.state.StateMachine.fail`, which keeps the
    pipeline free of a state it would have to be dragged out of before the next
    activation.
    """

    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    UNDERSTANDING = "understanding"
    EXECUTING = "executing"
    RESPONDING = "responding"

    @property
    def label(self) -> str:
        """Russian text for DevTools and the overlay caption."""
        return STATE_LABELS[self]

    @property
    def is_active(self) -> bool:
        """Whether a session is in flight, i.e. anything but :attr:`IDLE`."""
        return self is not PipelineState.IDLE


STATE_LABELS: Final[Mapping[PipelineState, str]] = {
    PipelineState.IDLE: "ожидание",
    PipelineState.LISTENING: "слушаю",
    PipelineState.RECORDING: "записываю фразу",
    PipelineState.TRANSCRIBING: "распознаю",
    PipelineState.UNDERSTANDING: "разбираю",
    PipelineState.EXECUTING: "выполняю",
    PipelineState.RESPONDING: "отвечаю",
}

#: Legal moves. ``IDLE`` is reachable from everywhere — that is cancellation, a
#: timeout and a failed stage all at once — and every stage may also jump
#: forward, because a pass that has nothing to transcribe or nothing to say
#: still has to end up in ``RESPONDING`` or back in ``IDLE``.
ALLOWED_TRANSITIONS: Final[Mapping[PipelineState, frozenset[PipelineState]]] = {
    PipelineState.IDLE: frozenset(
        {
            PipelineState.LISTENING,
            PipelineState.RECORDING,
            PipelineState.TRANSCRIBING,
            PipelineState.UNDERSTANDING,
        }
    ),
    PipelineState.LISTENING: frozenset(
        {PipelineState.IDLE, PipelineState.RECORDING, PipelineState.TRANSCRIBING}
    ),
    PipelineState.RECORDING: frozenset(
        {PipelineState.IDLE, PipelineState.LISTENING, PipelineState.TRANSCRIBING}
    ),
    PipelineState.TRANSCRIBING: frozenset(
        {PipelineState.IDLE, PipelineState.UNDERSTANDING, PipelineState.RESPONDING}
    ),
    PipelineState.UNDERSTANDING: frozenset(
        {PipelineState.IDLE, PipelineState.EXECUTING, PipelineState.RESPONDING}
    ),
    PipelineState.EXECUTING: frozenset({PipelineState.IDLE, PipelineState.RESPONDING}),
    PipelineState.RESPONDING: frozenset({PipelineState.IDLE, PipelineState.LISTENING}),
}

#: What the overlay shows for each stage. Four looks for seven stages: the user
#: has no use for the difference between «распознаю» and «разбираю» while the
#: sphere is pulsing, and giving each stage its own animation would make the
#: overlay flicker three times per request.
ASSISTANT_STATES: Final[Mapping[PipelineState, AssistantState]] = {
    PipelineState.IDLE: AssistantState.IDLE,
    PipelineState.LISTENING: AssistantState.LISTENING,
    PipelineState.RECORDING: AssistantState.LISTENING,
    PipelineState.TRANSCRIBING: AssistantState.THINKING,
    PipelineState.UNDERSTANDING: AssistantState.THINKING,
    PipelineState.EXECUTING: AssistantState.THINKING,
    PipelineState.RESPONDING: AssistantState.SPEAKING,
}

#: States in which a second activation is not a fresh request. ``RESPONDING`` is
#: deliberately absent: interrupting the answer is the whole point of barge-in.
BUSY_STATES: Final[frozenset[PipelineState]] = frozenset(
    {
        PipelineState.LISTENING,
        PipelineState.RECORDING,
        PipelineState.TRANSCRIBING,
        PipelineState.UNDERSTANDING,
        PipelineState.EXECUTING,
    }
)

#: Deadline for an action. No settings field covers it: the timeouts the user
#: can see are about waiting for something remote, while an action runs locally
#: and either returns quickly or is stuck. Generous, because «открой фотошоп»
#: legitimately takes seconds.
EXECUTING_TIMEOUT_SEC: Final = 20.0

#: Deadline for speaking. Long answers exist, and cutting one off mid-sentence
#: because the estimate was wrong is worse than waiting; this is a guard against
#: a wedged player, not a length limit.
RESPONDING_TIMEOUT_SEC: Final = 120.0

#: ``voice.stt.online_timeout_sec`` bounds one cloud attempt, and the router in
#: ``auto`` mode may try the cloud and then fall back to the local model, so the
#: stage deadline has to be a multiple of it — otherwise the pipeline would give
#: up on a recognition that is still going to succeed.
STT_TIMEOUT_FACTOR: Final = 3.0

#: …and a floor, so that lowering the cloud timeout to its minimum does not
#: leave the offline model less time than it needs to start.
STT_TIMEOUT_FLOOR_SEC: Final = 10.0


@dataclass(frozen=True, slots=True)
class PipelineTimeouts:
    """Deadline per stage, in seconds. ``0`` or less means «do not arm one».

    ``IDLE`` has no deadline for the obvious reason, and neither has a stage a
    caller decided to run unbounded — which is what a test that is not about
    timeouts does, so that a slow machine cannot cancel the session under it.
    """

    listening: float = 6.0
    recording: float = 30.0
    transcribing: float = 15.0
    understanding: float = 30.0
    executing: float = EXECUTING_TIMEOUT_SEC
    responding: float = RESPONDING_TIMEOUT_SEC

    @classmethod
    def from_settings(cls, settings: Settings) -> PipelineTimeouts:
        """Derive the deadlines from the settings the user already edits."""
        stt = max(
            settings.voice.stt.online_timeout_sec * STT_TIMEOUT_FACTOR,
            STT_TIMEOUT_FLOOR_SEC,
        )
        return cls(
            listening=settings.voice.wake.listen_window_sec,
            # The segmenter stops the phrase itself at max_utterance_sec; this is
            # the same number plus the silence it waits for, so the pipeline
            # never fires first and orphans a segment that was about to arrive.
            recording=(
                settings.voice.audio_input.max_utterance_sec
                + settings.voice.audio_input.silence_ms / 1000.0
            ),
            transcribing=stt,
            understanding=settings.ai.request_timeout_sec,
        )

    def for_state(self, state: PipelineState) -> float:
        """Deadline for ``state``, or ``0.0`` when the stage is unbounded."""
        return _TIMEOUT_FIELDS.get(state, lambda _: 0.0)(self)


_TIMEOUT_FIELDS: Final[Mapping[PipelineState, Callable[[PipelineTimeouts], float]]] = {
    PipelineState.LISTENING: lambda t: t.listening,
    PipelineState.RECORDING: lambda t: t.recording,
    PipelineState.TRANSCRIBING: lambda t: t.transcribing,
    PipelineState.UNDERSTANDING: lambda t: t.understanding,
    PipelineState.EXECUTING: lambda t: t.executing,
    PipelineState.RESPONDING: lambda t: t.responding,
}


class Timer(Protocol):
    """A pending deadline that can be called off."""

    def cancel(self) -> None:
        """Stop the timer. Must be safe to call after it has already fired."""


class Scheduler(Protocol):
    """Runs a callback once, later. The seam the timeout tests replace."""

    def call_later(self, delay: float, callback: Callable[[], None]) -> Timer:
        """Schedule ``callback`` to run ``delay`` seconds from now."""


class ThreadScheduler:
    """One :class:`threading.Timer` per deadline.

    Daemon threads: a deadline that has not fired must not keep the process
    alive on shutdown, and the pipeline cancels its timer on every transition
    anyway, so a stray one is at most a few seconds of a sleeping thread.
    """

    __slots__ = ()

    def call_later(self, delay: float, callback: Callable[[], None]) -> Timer:
        timer = threading.Timer(delay, callback)
        timer.daemon = True
        timer.start()
        return timer


class ManualScheduler:
    """Scheduler that fires only when told to. For tests, and only for tests.

    Lives here rather than in the test module because both the unit tests of the
    machine and the integration tests of the pipeline need it, and because a
    timeout test that sleeps is the one the task file warns about by name.
    """

    __slots__ = ("_pending",)

    def __init__(self) -> None:
        self._pending: list[_ManualTimer] = []

    def call_later(self, delay: float, callback: Callable[[], None]) -> Timer:
        timer = _ManualTimer(delay, callback)
        self._pending.append(timer)
        return timer

    @property
    def pending(self) -> tuple[float, ...]:
        """Delays of the timers that are still armed, in the order scheduled."""
        return tuple(timer.delay for timer in self._pending if not timer.cancelled)

    def fire_all(self) -> int:
        """Run every armed callback. Returns how many actually ran."""
        armed = [timer for timer in self._pending if not timer.cancelled]
        self._pending.clear()
        for timer in armed:
            timer.fire()
        return len(armed)


@dataclass(slots=True)
class _ManualTimer:
    delay: float
    callback: Callable[[], None]
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            self.callback()


class PipelineStateMachine:
    """Current :class:`PipelineState`, its deadline and the event that announces it.

    Args:
        bus: Where :class:`~ayris.core.events.PipelineStateChanged` goes.
        timeouts: Deadline per stage. Defaults are the schema defaults, so a
            machine built without settings still behaves sensibly.
        on_timeout: Called with the state that ran out of time, on the
            scheduler's thread. The pipeline turns that into a cancelled session
            with a spoken message; the machine itself only reports it.
        scheduler: How waiting is done. Replace it in tests.

    Thread safety: the state is read and written under a lock, events are
    published outside it, and the deadline callback re-checks that the state it
    was armed for is still current — a timer that fires while the pipeline is
    already moving on is dropped rather than cancelling the next stage.
    """

    __slots__ = (
        "_bus",
        "_lock",
        "_on_timeout",
        "_scheduler",
        "_session_id",
        "_state",
        "_timer",
        "timeouts",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        timeouts: PipelineTimeouts | None = None,
        on_timeout: Callable[[PipelineState], None] | None = None,
        scheduler: Scheduler | None = None,
    ) -> None:
        self._bus = bus
        self.timeouts: PipelineTimeouts = timeouts if timeouts is not None else PipelineTimeouts()
        self._on_timeout = on_timeout
        self._scheduler: Scheduler = scheduler if scheduler is not None else ThreadScheduler()
        self._lock = threading.RLock()
        self._state = PipelineState.IDLE
        self._session_id = ""
        self._timer: Timer | None = None

    @property
    def state(self) -> PipelineState:
        """Current stage. Safe to call from any thread."""
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str:
        """Session the current stage belongs to, empty when idle."""
        with self._lock:
            return self._session_id

    @property
    def busy(self) -> bool:
        """Whether a second activation should be ignored rather than started."""
        return self.state in BUSY_STATES

    def can_transition(self, target: PipelineState) -> bool:
        """Whether moving to ``target`` is legal from the current stage."""
        with self._lock:
            current = self._state
        if target is current:
            return False
        return target in ALLOWED_TRANSITIONS[current]

    def enter(
        self,
        target: PipelineState,
        *,
        session_id: str = "",
        detail: str = "",
        force: bool = False,
    ) -> bool:
        """Move to ``target``, arm its deadline and publish the change.

        Args:
            target: Stage to enter.
            session_id: Session the stage belongs to. Kept from the previous
                stage when empty, cleared on the way to :attr:`PipelineState.IDLE`.
            detail: Russian explanation, shown by DevTools next to the stage.
            force: Skip :data:`ALLOWED_TRANSITIONS`. The cancel path uses it to
                get back to idle from wherever the pipeline was.

        Returns:
            Whether the stage actually changed.
        """
        with self._lock:
            previous = self._state
            if target is previous:
                return False
            if not force and target not in ALLOWED_TRANSITIONS[previous]:
                _log.debug("переход %s → %s отклонён", previous.value, target.value)
                return False
            if target is PipelineState.IDLE:
                self._session_id = ""
            elif session_id:
                self._session_id = session_id
            self._state = target
            current_session = self._session_id
            self._disarm_locked()
            deadline = self.timeouts.for_state(target)
            if deadline > 0.0:
                self._timer = self._scheduler.call_later(
                    deadline, lambda: self._fire_timeout(target, current_session)
                )

        _log.debug("пайплайн: %s → %s", previous.value, target.value)
        self._bus.publish(
            PipelineStateChanged(
                state=target,
                previous=previous,
                session_id=current_session,
                detail=detail,
            )
        )
        return True

    def to_idle(self, *, detail: str = "") -> bool:
        """Return to ``IDLE`` from any stage and drop the pending deadline."""
        return self.enter(PipelineState.IDLE, detail=detail, force=True)

    def disarm(self) -> None:
        """Cancel the pending deadline without changing the stage.

        Used while a stage waits for something that cannot be interrupted, and by
        :meth:`close`.
        """
        with self._lock:
            self._disarm_locked()

    def apply_timeouts(self, timeouts: PipelineTimeouts) -> None:
        """Adopt new deadlines. Takes effect from the next stage, not this one."""
        with self._lock:
            self.timeouts = timeouts

    def close(self) -> None:
        """Drop the pending deadline. Called on shutdown."""
        self.disarm()

    def _disarm_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _fire_timeout(self, state: PipelineState, session_id: str) -> None:
        with self._lock:
            if self._state is not state or self._session_id != session_id:
                return
            self._timer = None
            callback = self._on_timeout
        _log.warning("таймаут стадии %s (сессия %s)", state.value, session_id or "—")
        if callback is not None:
            callback(state)

    def __repr__(self) -> str:
        return f"PipelineStateMachine({self.state.value}, session={self.session_id or '—'})"
