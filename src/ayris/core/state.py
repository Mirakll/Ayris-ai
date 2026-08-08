"""What the assistant is doing right now, as a small state machine.

Four independent facts make up the status: what the assistant is doing
(:class:`AssistantState`), how the microphone is armed (:class:`MicMode`),
whether the microphone is muted at all, and whether the cloud is reachable. The
overlay's sphere, the tray icon and the pipeline dispatcher all read exactly
these, so they live in one place with one lock rather than being tracked
separately by each of them.

Every change publishes an event, which is the only way subscribers learn about
it — nothing polls :attr:`StateMachine.snapshot`. Transitions are validated
against :data:`ALLOWED_TRANSITIONS`: a rejected move is logged and ignored rather
than raising, because the caller is usually a worker callback reacting to
hardware that has its own opinion about ordering, and crashing the pipeline over
a stale ``speaking -> speaking`` would be worse than dropping it.

The module imports :mod:`ayris.core.events` and never the other way round.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.core.events import EventBus, MicToggled, ModeChanged, OnlineStatusChanged
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.config import Settings

__all__ = [
    "ALLOWED_TRANSITIONS",
    "MIC_MODE_LABELS",
    "STATE_LABELS",
    "AssistantState",
    "MicMode",
    "StateMachine",
    "StatusSnapshot",
]

_log = get_logger(__name__)


class AssistantState(StrEnum):
    """What Ayris is doing. Drives the sphere animation and the tray icon."""

    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"

    @property
    def label(self) -> str:
        """Russian text for the tray tooltip and the overlay caption."""
        return STATE_LABELS[self]


class MicMode(StrEnum):
    """How the microphone is armed.

    ``HYBRID`` is not a third way of listening but both of the other two at once;
    it exists because the settings schema offers it, and dropping it here would
    make the mapping from ``voice.wake.mic_mode`` lossy.
    """

    ALWAYS = "always"
    PTT = "ptt"
    HYBRID = "hybrid"

    @property
    def label(self) -> str:
        return MIC_MODE_LABELS[self]

    @property
    def wake_word_active(self) -> bool:
        """Whether the wake word engine should be running in this mode."""
        return self is not MicMode.PTT


STATE_LABELS: Final[Mapping[AssistantState, str]] = {
    AssistantState.IDLE: "ожидание",
    AssistantState.LISTENING: "слушаю",
    AssistantState.THINKING: "думаю",
    AssistantState.SPEAKING: "отвечаю",
    AssistantState.ERROR: "ошибка",
}

MIC_MODE_LABELS: Final[Mapping[MicMode, str]] = {
    MicMode.ALWAYS: "микрофон всегда включён",
    MicMode.PTT: "по нажатию клавиши",
    MicMode.HYBRID: "слово активации и клавиша",
}

#: Legal moves. Anything may fail into ``ERROR``, and ``ERROR`` only clears
#: through ``IDLE``, so a subsystem cannot quietly resume mid-pipeline after a
#: failure the user was told about.
ALLOWED_TRANSITIONS: Final[Mapping[AssistantState, frozenset[AssistantState]]] = {
    AssistantState.IDLE: frozenset(
        {AssistantState.LISTENING, AssistantState.THINKING, AssistantState.SPEAKING}
    ),
    AssistantState.LISTENING: frozenset({AssistantState.IDLE, AssistantState.THINKING}),
    AssistantState.THINKING: frozenset(
        {AssistantState.IDLE, AssistantState.SPEAKING, AssistantState.LISTENING}
    ),
    AssistantState.SPEAKING: frozenset(
        {AssistantState.IDLE, AssistantState.LISTENING, AssistantState.THINKING}
    ),
    AssistantState.ERROR: frozenset({AssistantState.IDLE}),
}


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    """Consistent view of the four facts. Frozen, so it is safe to hand around."""

    state: AssistantState = AssistantState.IDLE
    mic_mode: MicMode = MicMode.HYBRID
    mic_enabled: bool = True
    online: bool = False
    detail: str = ""

    @property
    def is_busy(self) -> bool:
        """Whether a request is being worked on right now."""
        return self.state in (AssistantState.THINKING, AssistantState.SPEAKING)

    @property
    def listens_for_wake_word(self) -> bool:
        """Whether the wake word engine should be listening at this moment."""
        return self.mic_enabled and self.mic_mode.wake_word_active

    def describe(self) -> str:
        """One-line Russian summary for the tray tooltip."""
        parts = [self.state.label]
        if not self.mic_enabled:
            parts.append("микрофон выключен")
        else:
            parts.append(self.mic_mode.label)
        parts.append("сеть доступна" if self.online else "офлайн")
        return ", ".join(parts)


class StateMachine:
    """Current :class:`StatusSnapshot`, guarded and event-publishing.

    Args:
        bus: Where changes are announced. Publishing is thread-safe, so a worker
            callback may drive the state directly.
        initial: Starting snapshot. Defaults to idle, hybrid microphone, offline —
            offline until something proves otherwise, so no cloud request is
            attempted before the first connectivity check.

    Thread safety: the snapshot is replaced under a lock and never mutated, so a
    reader always sees a whole one. Events are published outside the lock.
    """

    __slots__ = ("_bus", "_lock", "_snapshot")

    def __init__(self, bus: EventBus, *, initial: StatusSnapshot | None = None) -> None:
        self._bus = bus
        self._lock = threading.RLock()
        self._snapshot = initial if initial is not None else StatusSnapshot()

    @property
    def snapshot(self) -> StatusSnapshot:
        """Current status. Safe to call from any thread."""
        with self._lock:
            return self._snapshot

    @property
    def state(self) -> AssistantState:
        return self.snapshot.state

    @property
    def mic_mode(self) -> MicMode:
        return self.snapshot.mic_mode

    @property
    def mic_enabled(self) -> bool:
        return self.snapshot.mic_enabled

    @property
    def online(self) -> bool:
        return self.snapshot.online

    # ------------------------------------------------------------------
    # transitions
    # ------------------------------------------------------------------

    def can_transition(self, target: AssistantState) -> bool:
        """Whether moving to ``target`` is legal from the current state."""
        current = self.snapshot.state
        if target is current or target is AssistantState.ERROR:
            return True
        return target in ALLOWED_TRANSITIONS[current]

    def set_state(self, target: AssistantState, *, detail: str = "", force: bool = False) -> bool:
        """Move to ``target`` and publish :class:`~ayris.core.events.ModeChanged`.

        Args:
            target: New state.
            detail: Russian explanation shown next to an error, ignored otherwise.
            force: Skip the transition table. Used by the cancel path, which has
                to get back to idle from wherever the pipeline was.

        Returns:
            Whether the state actually changed. A rejected or repeated transition
            returns ``False`` and publishes nothing.
        """
        with self._lock:
            previous = self._snapshot
            if not force and not self.can_transition(target):
                _log.debug("переход %s → %s отклонён", previous.state.value, target.value)
                return False
            if previous.state is target and previous.detail == detail:
                return False
            self._snapshot = replace(previous, state=target, detail=detail)
            current = self._snapshot

        _log.debug("состояние: %s → %s", previous.state.value, target.value)
        self._bus.publish(
            ModeChanged(
                state=current.state,
                previous=previous.state,
                mic_mode=current.mic_mode,
                detail=detail,
            )
        )
        return True

    def to_idle(self, *, detail: str = "") -> bool:
        """Return to idle from anywhere, including from the error state."""
        return self.set_state(AssistantState.IDLE, detail=detail, force=True)

    def fail(self, detail: str) -> bool:
        """Enter the error state with a Russian explanation for the overlay."""
        return self.set_state(AssistantState.ERROR, detail=detail, force=True)

    # ------------------------------------------------------------------
    # microphone and connectivity
    # ------------------------------------------------------------------

    def set_mic_mode(self, mode: MicMode) -> bool:
        """Switch between wake word, push-to-talk and both.

        Publishes :class:`~ayris.core.events.MicToggled` because the audio worker
        and the tray both key off the pair (mode, enabled) rather than either one.
        """
        with self._lock:
            previous = self._snapshot
            if previous.mic_mode is mode:
                return False
            self._snapshot = replace(previous, mic_mode=mode)
            enabled = self._snapshot.mic_enabled

        _log.info("режим микрофона: %s", mode.label)
        self._bus.publish(MicToggled(enabled=enabled, mic_mode=mode))
        return True

    def set_mic_enabled(self, *, enabled: bool) -> bool:
        """Mute or unmute the microphone."""
        with self._lock:
            previous = self._snapshot
            if previous.mic_enabled == enabled:
                return False
            self._snapshot = replace(previous, mic_enabled=enabled)
            mode = self._snapshot.mic_mode

        _log.info("микрофон %s", "включён" if enabled else "выключен")
        self._bus.publish(MicToggled(enabled=enabled, mic_mode=mode))
        return True

    def toggle_mic(self) -> bool:
        """Flip the mute state. Returns the new value."""
        self.set_mic_enabled(enabled=not self.snapshot.mic_enabled)
        return self.snapshot.mic_enabled

    def set_online(self, *, online: bool, detail: str = "") -> bool:
        """Record cloud reachability, flipping the STT and TTS routers."""
        with self._lock:
            previous = self._snapshot
            if previous.online == online:
                return False
            self._snapshot = replace(previous, online=online)

        _log.info("сеть: %s", "доступна" if online else "недоступна")
        self._bus.publish(OnlineStatusChanged(online=online, detail=detail))
        return True

    def apply_settings(self, settings: Settings) -> None:
        """Adopt the microphone mode from the settings. Called on every change."""
        self.set_mic_mode(MicMode(settings.voice.wake.mic_mode))

    def __repr__(self) -> str:
        return f"StateMachine({self.snapshot.describe()})"
