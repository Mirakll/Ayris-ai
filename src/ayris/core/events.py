"""Typed event bus and the event vocabulary of Ayris.

Architecture invariant 3: subsystems talk through events instead of importing
each other. Every event is a frozen dataclass, so a handler gets attribute
access and mypy checks the payload at both ends.

Delivery has one rule: **handlers run on the UI thread.** Audio levels arrive
from a worker reader thread, transcripts from a ``QThread``, timers from a
scheduler thread, and nearly every one of them ends up touching the overlay or
the tray. :meth:`EventBus.publish` is therefore safe to call from any thread: an
event published from the UI thread is delivered inline, an event published from
anywhere else is queued and handed over by the wake-up installed with
:meth:`EventBus.set_wakeup`.

This module does not import Qt — ``ayris.core`` has to stay importable inside
worker processes, which have no Qt event loop. The composition root installs a
wake-up that emits a Qt signal over a queued connection whose slot calls
:meth:`EventBus.drain`; a worker installs nothing and drains the queue in its
own loop.

Publishing is re-entrant. A handler that publishes another event does not
recurse: the nested event joins the queue and is delivered by the loop already
running, so ordering stays FIFO and the stack stays flat.

Subscriptions to bound methods are held weakly, so a closed window stops
receiving events even when it forgot to unsubscribe. Anything else — a module
level function, a lambda, a closure — is held strongly and has to be released
through the callable :meth:`EventBus.subscribe` returns, because a weak
reference to it would die immediately. Subscribing to a base class works: a
handler registered for :class:`Event` sees every event, which is what the
DevTools log view and the pipeline tracer use.
"""

from __future__ import annotations

import threading
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from types import MethodType
from typing import TYPE_CHECKING, Any, Final, TypeVar

from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.models import JsonObject, utc_now
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from datetime import datetime

    # Imported for annotations only: ayris.core.state publishes onto the bus, so
    # a runtime import here would close a cycle. ``from __future__ import
    # annotations`` keeps these names unevaluated, and no dataclass field needs
    # the real class at runtime.
    from ayris.core.paths import ModelKind
    from ayris.core.state import AssistantState, MicMode

__all__ = [
    "COMMANDS_CHANGE_DELETED",
    "COMMANDS_CHANGE_RELOADED",
    "COMMANDS_CHANGE_SAVED",
    "DRAIN_BATCH",
    "TTS_REASON_CANCELLED",
    "TTS_REASON_COMPLETED",
    "TTS_REASON_ERROR",
    "ActionFailed",
    "ActionFinished",
    "ActionStarted",
    "ActiveModelChanged",
    "AudioLevelChanged",
    "CancelRequested",
    "CommandsChanged",
    "ConfigChanged",
    "Event",
    "EventBus",
    "Handler",
    "IntentMatched",
    "LogLine",
    "MicToggled",
    "ModeChanged",
    "ModelDownloadFailed",
    "ModelDownloadFinished",
    "ModelDownloadProgress",
    "ModelDownloadStarted",
    "ModelRemoved",
    "NotificationRequested",
    "OnlineStatusChanged",
    "SpeechEnded",
    "SpeechStarted",
    "TimerFired",
    "TranscriptReady",
    "TtsFinished",
    "TtsStarted",
    "Unsubscribe",
    "WakeUp",
    "WakeWordDetected",
    "WorkerCrashed",
    "WorkerRestarted",
]

_log = get_logger(__name__)

#: Events delivered in one :meth:`EventBus.drain` before the loop yields, so a
#: burst of audio levels cannot starve the Qt event loop of paint events.
DRAIN_BATCH: Final = 256

#: Values :attr:`TtsFinished.reason` takes. Strings rather than an enum because
#: the same value crosses the worker pipe as JSON, and one representation on both
#: sides is worth more here than the type checking an enum would add.
TTS_REASON_COMPLETED: Final = "completed"
TTS_REASON_CANCELLED: Final = "cancelled"
TTS_REASON_ERROR: Final = "error"

#: Values :attr:`CommandsChanged.change` takes. ``SAVED`` covers a new command
#: and an edited one alike — the subscriber reloads that command either way, so
#: telling the two apart would only give it a distinction to ignore.
COMMANDS_CHANGE_SAVED: Final = "saved"
COMMANDS_CHANGE_DELETED: Final = "deleted"
#: The library as a whole moved: a profile switch, an import, an undo of a batch.
#: :attr:`CommandsChanged.command_id` is ``None`` and everything is re-read.
COMMANDS_CHANGE_RELOADED: Final = "reloaded"


@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base class for every event.

    ``at`` is keyword-only so subclasses can still declare positional fields
    despite the inherited default.
    """

    at: datetime = field(default_factory=utc_now)

    @property
    def name(self) -> str:
        """Class name, used by the log and DevTools instead of ``repr``."""
        return type(self).__name__


# ----------------------------------------------------------------------
# audio and speech
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WakeWordDetected(Event):
    """The wake word engine matched one of the configured phrases."""

    phrase: str
    confidence: float = 0.0
    engine: str = ""


@dataclass(frozen=True, slots=True)
class AudioLevelChanged(Event):
    """Input level for the overlay's sphere and the microphone meter.

    Published tens of times per second, so subscribers must stay cheap and the
    capture worker throttles it rather than emitting per frame.
    """

    rms: float
    peak: float = 0.0
    is_speech: bool = False


@dataclass(frozen=True, slots=True)
class SpeechStarted(Event):
    """VAD detected the start of an utterance."""

    source: str = "vad"


@dataclass(frozen=True, slots=True)
class SpeechEnded(Event):
    """VAD detected the end of an utterance."""

    duration_ms: int = 0
    reason: str = "silence"


@dataclass(frozen=True, slots=True)
class TranscriptReady(Event):
    """Recognised text. Partial results arrive with ``is_final=False``."""

    text: str
    is_final: bool = True
    confidence: float = 0.0
    engine: str = ""
    duration_ms: int = 0
    request_id: str = ""


# ----------------------------------------------------------------------
# understanding and execution
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntentMatched(Event):
    """NLU resolved a phrase to a command.

    ``source`` says which stage matched — ``exact``, ``fuzzy``, ``regex`` or
    ``llm`` — which is what the pipeline log and the debugger display.
    """

    intent: str
    command_id: int | None = None
    confidence: float = 0.0
    source: str = "exact"
    slots: JsonObject = field(default_factory=dict)
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class ActionStarted(Event):
    """A registered action began executing."""

    action: str
    command_id: int | None = None
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class ActionFinished(Event):
    """An action completed. ``result`` is the Russian summary for the history."""

    action: str
    result: str = ""
    duration_ms: int = 0
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class ActionFailed(Event):
    """An action raised. ``user_message`` is safe to show; ``error`` is for the log."""

    action: str
    error: str
    user_message: str = ""
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class TtsStarted(Event):
    """Sound began coming out of the speakers.

    Published by :class:`ayris.audio.tts.player.TtsPlayer` when a phrase reaches
    the device, not when synthesis was requested: the overlay's mouth animation
    has to line up with what the user hears, and the gap between the two is the
    whole point of the streaming path.

    ``duration_estimate_ms`` is how long the queued audio will take, so the
    overlay can plan an animation instead of polling. It is exact for a phrase
    already synthesized in full and a lower bound while more sentences are still
    coming, which the overlay treats as "at least this long".
    """

    text: str
    engine: str = ""
    duration_estimate_ms: int = 0
    request_id: str = ""


@dataclass(frozen=True, slots=True)
class TtsFinished(Event):
    """Speaking stopped.

    ``reason`` is one of :data:`TTS_REASON_COMPLETED`,
    :data:`TTS_REASON_CANCELLED` or :data:`TTS_REASON_ERROR`. The overlay needs
    the distinction: a completed phrase fades out, an interrupt snaps back to
    idle, and an error shows the notification.
    """

    reason: str = "completed"
    text: str = ""
    request_id: str = ""

    @property
    def interrupted(self) -> bool:
        """Whether «Айрис, стоп» - or anything else - cut this phrase short."""
        return self.reason == TTS_REASON_CANCELLED


@dataclass(frozen=True, slots=True)
class CancelRequested(Event):
    """The stop word, the cancel hotkey or the overlay asked to abort.

    Every long-running subsystem subscribes: TTS stops speaking, the macro engine
    unwinds, the LLM request is dropped.
    """

    reason: str = ""


# ----------------------------------------------------------------------
# assistant state
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModeChanged(Event):
    """The assistant moved between idle, listening, thinking, speaking or error.

    Published by :class:`ayris.core.state.StateMachine`; the overlay animates the
    sphere from it and the tray swaps its icon.
    """

    state: AssistantState
    previous: AssistantState
    mic_mode: MicMode
    detail: str = ""


@dataclass(frozen=True, slots=True)
class MicToggled(Event):
    """The microphone was muted or unmuted."""

    enabled: bool
    mic_mode: MicMode | None = None


@dataclass(frozen=True, slots=True)
class OnlineStatusChanged(Event):
    """Cloud reachability changed, which flips the STT and TTS routers."""

    online: bool
    detail: str = ""


# ----------------------------------------------------------------------
# models
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelDownloadStarted(Event):
    """A model download began, or resumed after an interruption.

    ``downloaded`` is non-zero when a ``.part`` file was picked up, so the
    progress bar can start where the previous attempt stopped instead of
    snapping back to zero.
    """

    model_id: str
    kind: ModelKind
    name: str = ""
    total: int = 0
    downloaded: int = 0
    resumed: bool = False


@dataclass(frozen=True, slots=True)
class ModelDownloadProgress(Event):
    """How far along a download is.

    Published a few times per second, not per chunk — same treatment as
    :class:`AudioLevelChanged`, and for the same reason: at a megabyte a second
    an unthrottled event per 64 KiB block would be sixteen repaints per second of
    a progress bar nobody can read that fast.

    ``total`` is ``0`` when the server sent no ``Content-Length``, which the UI
    should render as an indeterminate bar rather than as "0 %".
    """

    model_id: str
    downloaded: int
    total: int = 0
    speed_bps: float = 0.0
    eta_s: float = 0.0

    @property
    def fraction(self) -> float:
        """Progress in ``0.0..1.0``. ``0.0`` while the total is unknown."""
        if self.total <= 0:
            return 0.0
        return min(1.0, self.downloaded / self.total)


@dataclass(frozen=True, slots=True)
class ModelDownloadFinished(Event):
    """A model finished downloading, was verified and is installed on disk."""

    model_id: str
    kind: ModelKind
    path: str = ""
    size_bytes: int = 0
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class ModelDownloadFailed(Event):
    """A download stopped short.

    ``cancelled`` separates "the user pressed отмена" from a real failure: the
    first needs no notification, the second does. ``resumable`` says whether the
    partial file was kept, so the UI can offer «продолжить» instead of
    «скачать заново».
    """

    model_id: str
    error: str = ""
    user_message: str = ""
    cancelled: bool = False
    resumable: bool = False


@dataclass(frozen=True, slots=True)
class ActiveModelChanged(Event):
    """The model a subsystem should be using changed.

    Published by :class:`ayris.models.registry.ModelRegistry` whenever the active
    row for a kind moves, including when it is cleared by a deletion — then
    ``name`` is empty. The STT, TTS and wake-word workers subscribe and reload;
    that is the whole point of the event, and it is why it carries ``path`` as
    well as ``name``, so a reload needs no second trip to the database.
    """

    kind: ModelKind
    name: str = ""
    engine: str = ""
    path: str = ""
    model_id: int | None = None
    catalog_id: str = ""

    @property
    def cleared(self) -> bool:
        """Whether this kind now has no active model at all."""
        return self.model_id is None


@dataclass(frozen=True, slots=True)
class ModelRemoved(Event):
    """An installed model was deleted, freeing ``freed_bytes`` on disk."""

    model_id: str
    kind: ModelKind
    name: str = ""
    freed_bytes: int = 0


# ----------------------------------------------------------------------
# workers, configuration, notifications
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WorkerCrashed(Event):
    """A worker process died. Section 14: restart, warn, fall back."""

    worker: str
    exit_code: int | None = None
    error: str = ""
    restarts: int = 0


@dataclass(frozen=True, slots=True)
class WorkerRestarted(Event):
    """A worker came back up after a crash or a settings change."""

    worker: str
    attempt: int = 1
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConfigChanged(Event):
    """Settings changed on disk or through the settings window.

    Wraps the diff from :mod:`ayris.core.config` rather than repeating it, so a
    subscriber can ask ``event.touches("overlay")`` and reload only its own part.
    """

    diff: SettingsDiff

    @property
    def settings(self) -> Any:
        """The new settings (``ayris.core.config.Settings``).

        Typed loosely to keep the event payload import-light.
        """
        return self.diff.settings

    def touches(self, prefix: str) -> bool:
        """Whether anything under ``prefix`` changed, e.g. ``"voice.tts"``."""
        return self.diff.touches(prefix)


@dataclass(frozen=True, slots=True)
class CommandsChanged(Event):
    """The command library changed and anything derived from it is now stale.

    Published by the layer that writes commands, consumed by the trigger index:
    a saved or deleted command names itself in :attr:`command_id` so only its
    triggers get rebuilt, while ``reloaded`` leaves it ``None`` and means the
    whole library moved — a profile switch, an import, an undo of a batch.
    """

    command_id: int | None = None
    change: str = COMMANDS_CHANGE_SAVED


@dataclass(frozen=True, slots=True)
class TimerFired(Event):
    """A timer, reminder or alarm came due."""

    timer_id: int
    label: str = ""
    kind: str = "timer"


@dataclass(frozen=True, slots=True)
class NotificationRequested(Event):
    """Something wants a tray balloon or an overlay toast.

    ``level`` is ``info``, ``warning`` or ``error``; the tray maps it to an icon.
    """

    title: str
    message: str = ""
    level: str = "info"
    timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class LogLine(Event):
    """One log record, mirrored onto the bus for the DevTools log view."""

    level: str
    message: str
    logger: str = ""


# ----------------------------------------------------------------------
# the bus
# ----------------------------------------------------------------------

E = TypeVar("E", bound=Event)

#: A subscriber. Must not raise; the bus logs and carries on if it does.
Handler = Callable[[E], None]

#: Returned by :meth:`EventBus.subscribe`. Idempotent.
Unsubscribe = Callable[[], None]

#: Installed by the composition root to wake the UI thread. Called from the
#: publishing thread, so it must be thread-safe and must not block — emitting a
#: Qt signal over a queued connection is exactly the intended implementation.
WakeUp = Callable[[], None]


class _Subscription:
    """One handler, held weakly when that is safe.

    A bound method is wrapped in :class:`weakref.WeakMethod`: widgets and
    controllers are the things that realistically outlive their subscription, and
    holding them strongly would keep a closed window alive and still call it.
    Everything else is held strongly, because a weak reference to a lambda or a
    local closure would be collected before the first event arrives.
    """

    __slots__ = ("_ref", "_strong", "weak")

    def __init__(self, handler: Callable[[Any], None], *, weak: bool) -> None:
        self._ref: weakref.WeakMethod[Any] | None = None
        self._strong: Callable[[Any], None] | None = handler
        self.weak = False
        if weak and isinstance(handler, MethodType):
            self._ref = weakref.WeakMethod(handler)
            self._strong = None
            self.weak = True

    def resolve(self) -> Callable[[Any], None] | None:
        """The handler, or ``None`` once its owner has been collected."""
        if self._ref is not None:
            return self._ref()
        return self._strong

    def matches(self, handler: Callable[[Any], None]) -> bool:
        """Whether this subscription wraps ``handler``."""
        return self.resolve() == handler


class EventBus:
    """Typed publish/subscribe with delivery on one designated thread.

    Args:
        thread_id: Identifier of the thread handlers run on. Defaults to the
            thread that constructs the bus, which in the main process is the UI
            thread. ``None`` disables marshalling entirely: every publish is
            delivered inline, which is what a worker process wants.
        wakeup: Callable that asks the delivery thread to call :meth:`drain`.
            Without one, a cross-thread publish waits in the queue until
            something drains it — a periodic timer, or the worker's own loop.

    Thread safety: subscriber bookkeeping and the queue are guarded by one lock.
    Handlers run outside it, so a handler may subscribe, unsubscribe or publish
    without deadlocking.
    """

    __slots__ = (
        "_cache",
        "_delivered",
        "_failed",
        "_local",
        "_lock",
        "_queue",
        "_subscriptions",
        "_thread_id",
        "_unbound",
        "_wake_pending",
        "_wakeup",
    )

    def __init__(self, *, thread_id: int | None = -1, wakeup: WakeUp | None = None) -> None:
        self._lock = threading.RLock()
        self._subscriptions: dict[type[Event], list[_Subscription]] = {}
        self._cache: dict[type[Event], tuple[type[Event], ...]] = {}
        self._queue: deque[Event] = deque()
        self._local = threading.local()
        self._wakeup = wakeup
        self._wake_pending = False
        self._delivered = 0
        self._failed = 0
        # -1 is the "not given" marker: None is a meaningful value that means
        # "deliver on whichever thread publishes".
        self._unbound = thread_id is None
        self._thread_id = threading.get_ident() if thread_id == -1 else thread_id

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------

    def bind_to_thread(self, thread_id: int | None = None) -> None:
        """Declare which thread handlers run on. Defaults to the calling thread.

        Called from the UI thread once ``QApplication`` exists, because the bus is
        usually constructed before it.
        """
        with self._lock:
            self._unbound = False
            self._thread_id = threading.get_ident() if thread_id is None else thread_id

    def set_wakeup(self, wakeup: WakeUp | None) -> None:
        """Install (or remove) the callback that asks the UI thread to drain."""
        with self._lock:
            self._wakeup = wakeup
            pending = bool(self._queue)
            self._wake_pending = False
        if pending:
            self._wake()

    @property
    def thread_id(self) -> int | None:
        """Thread handlers run on, or ``None`` when marshalling is off."""
        with self._lock:
            return None if self._unbound else self._thread_id

    def _on_delivery_thread(self) -> bool:
        return self._unbound or threading.get_ident() == self._thread_id

    # ------------------------------------------------------------------
    # subscription
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: type[E],
        handler: Handler[E],
        *,
        weak: bool = True,
    ) -> Unsubscribe:
        """Register ``handler`` for ``event_type`` and its subclasses.

        Args:
            event_type: Event class to listen for. :class:`Event` itself matches
                everything.
            handler: Called on the delivery thread with the event.
            weak: Hold a bound method weakly, so its owner can be collected. Has
                no effect on plain functions, lambdas and closures — see the class
                docstring — and is worth turning off for a bound method on an
                object nothing else keeps alive.

        Returns:
            A callable that removes the subscription. Safe to call twice.
        """
        subscription = _Subscription(handler, weak=weak)
        with self._lock:
            self._subscriptions.setdefault(event_type, []).append(subscription)
            self._cache.clear()

        def unsubscribe() -> None:
            with self._lock:
                bucket = self._subscriptions.get(event_type)
                if bucket is None:
                    return
                if subscription in bucket:
                    bucket.remove(subscription)
                if not bucket:
                    del self._subscriptions[event_type]
                self._cache.clear()

        return unsubscribe

    def unsubscribe(self, event_type: type[E], handler: Handler[E]) -> bool:
        """Remove ``handler`` from ``event_type``.

        Returns:
            Whether anything was removed.
        """
        with self._lock:
            bucket = self._subscriptions.get(event_type)
            if bucket is None:
                return False
            remaining = [item for item in bucket if not item.matches(handler)]
            if len(remaining) == len(bucket):
                return False
            if remaining:
                self._subscriptions[event_type] = remaining
            else:
                del self._subscriptions[event_type]
            self._cache.clear()
            return True

    def subscriber_count(self, event_type: type[Event] | None = None) -> int:
        """Live subscriptions, for tests and the DevTools tab.

        Collected weak subscriptions are not counted, and are dropped on the way.
        """
        with self._lock:
            buckets = (
                list(self._subscriptions.values())
                if event_type is None
                else [self._subscriptions.get(event_type, [])]
            )
            total = 0
            for bucket in buckets:
                total += sum(1 for item in bucket if item.resolve() is not None)
            return total

    def clear(self) -> None:
        """Drop every subscription and every queued event. Used on shutdown."""
        with self._lock:
            self._subscriptions.clear()
            self._cache.clear()
            self._queue.clear()
            self._wake_pending = False

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to its subscribers. Callable from any thread.

        On the delivery thread the handlers run before this returns, unless a
        delivery is already in progress — then the event is appended to the queue
        the running loop is draining, which keeps ordering FIFO and stops a
        handler that publishes from recursing.
        """
        if self._on_delivery_thread() and not getattr(self._local, "dispatching", False):
            self._queue_event(event, wake=False)
            self._run_queue()
            return
        self._queue_event(event, wake=not self._on_delivery_thread())

    def _queue_event(self, event: Event, *, wake: bool) -> None:
        with self._lock:
            self._queue.append(event)
            needs_wake = wake and not self._wake_pending
            if needs_wake:
                self._wake_pending = True
        if needs_wake:
            self._wake()

    def _wake(self) -> None:
        with self._lock:
            wakeup = self._wakeup
        if wakeup is None:
            return
        try:
            wakeup()
        except Exception:
            _log.exception("не удалось разбудить поток доставки событий")

    def drain(self, limit: int = DRAIN_BATCH) -> int:
        """Deliver queued events on the calling thread.

        Args:
            limit: Stop after this many events, leaving the rest for the next
                call. ``0`` drains everything. The default keeps a burst of audio
                levels from starving the Qt event loop.

        Returns:
            How many events were delivered.
        """
        if getattr(self._local, "dispatching", False):
            return 0
        return self._run_queue(limit)

    def _run_queue(self, limit: int = 0) -> int:
        """Pop and dispatch until the queue is empty or ``limit`` is reached."""
        self._local.dispatching = True
        delivered = 0
        try:
            while True:
                with self._lock:
                    if not self._queue or (limit and delivered >= limit):
                        # Cleared inside the lock in both exit cases: a publisher
                        # that arrives after this point must arm a fresh wake-up
                        # rather than assume this loop will pick its event up.
                        self._wake_pending = False
                        break
                    event = self._queue.popleft()
                self._dispatch(event)
                delivered += 1
        finally:
            self._local.dispatching = False
        # Whatever the limit left behind needs another pass.
        self._rearm()
        return delivered

    def _rearm(self) -> None:
        """Ask for another drain if events are still queued."""
        with self._lock:
            if self._wake_pending or not self._queue:
                return
            self._wake_pending = True
        self._wake()

    def _dispatch(self, event: Event) -> None:
        """Call every live subscriber of ``event``, surviving the ones that raise."""
        for subscription in self._subscriptions_for(type(event)):
            handler = subscription.resolve()
            if handler is None:
                continue
            try:
                handler(event)
            except Exception:
                self._failed += 1
                _log.exception("подписчик события %s завершился с ошибкой", event.name)
        self._delivered += 1

    def _subscriptions_for(self, event_type: type[Event]) -> list[_Subscription]:
        """Subscriptions for ``event_type``, walking up to its base classes.

        Dead weak subscriptions are pruned here rather than by a callback, which
        keeps the collection cost on the delivery thread instead of on whatever
        thread happened to run the garbage collector.
        """
        with self._lock:
            keys = self._cache.get(event_type)
            if keys is None:
                keys = tuple(
                    base
                    for base in event_type.__mro__
                    if isinstance(base, type)
                    and issubclass(base, Event)
                    and base in self._subscriptions
                )
                self._cache[event_type] = keys

            matched: list[_Subscription] = []
            for key in keys:
                bucket = self._subscriptions.get(key)
                if bucket is None:
                    continue
                live = [item for item in bucket if item.resolve() is not None]
                if len(live) != len(bucket):
                    if live:
                        self._subscriptions[key] = live
                    else:
                        del self._subscriptions[key]
                    self._cache.clear()
                matched.extend(live)
            return matched

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        """Events waiting for the delivery thread."""
        with self._lock:
            return len(self._queue)

    @property
    def delivered(self) -> int:
        """Events dispatched since the bus was created."""
        return self._delivered

    @property
    def failed(self) -> int:
        """Handler calls that raised. Shown in DevTools; never zero silently."""
        return self._failed

    def __repr__(self) -> str:
        return (
            f"EventBus(subscribers={self.subscriber_count()}, "
            f"pending={self.pending}, delivered={self._delivered})"
        )
