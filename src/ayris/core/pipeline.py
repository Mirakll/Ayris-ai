"""The dispatcher: one utterance, from the wake word to the last sound of the answer.

Everything Ayris can do already exists somewhere else — the audio worker hears,
:class:`~ayris.audio.stt.router.SttRouter` recognises,
:class:`~ayris.nlu.matcher.Matcher` decides what was meant,
:class:`~ayris.audio.tts.router.TtsRouter` answers. This module owns none of
that. What it owns is the order they run in, the session they share, the clock
over each of them, and what happens when one of them fails: exactly the things
that have nowhere to live if every subsystem talks to the next one directly.

Four decisions shape the module.

*One session at a time, and the rule is explicit.* :meth:`Pipeline.activate`
either starts a session, cancels the current one and starts a new one, or refuses
— never «whichever happens first». Idle starts. Speaking is barge-in: the answer
is cut off and the new phrase wins, because a user who talks over Ayris is
correcting it. Anything in between is refused with a log line, because a second
wake word while the first phrase is still being recognised is an echo or a
neighbour, not a request.

*The stages are synchronous, the pipeline is not.* :meth:`Pipeline.run_text` and
the voice path run the same ``_process`` function top to bottom; the difference is
that the voice path hands it to a ``runner`` (a thread by default) and returns
immediately. Nothing here is ever waited on from the UI thread — invariant 2 —
and there is no second implementation of the flow to keep in step with the first.

*Cancellation is cooperative and checked between stages.* A stage that has
started is not killed; :class:`_Session` is flagged, the TTS handle is cancelled,
the LLM's ``cancel`` predicate starts returning true, and the next stage boundary
raises :class:`_Cancelled`. The alternative — killing a thread mid-recognition —
leaves the STT engine in a state nobody has tested.

*Every failure is a sentence.* Each stage is wrapped, the typed exceptions from
:mod:`ayris.core.errors` already carry a Russian ``user_message``, and that is
what gets spoken. A stage that raises something untyped is a bug, and it is
logged as one and still answered with a sentence, because a silent assistant
looks broken in a way that a wrong answer does not.

Nothing concrete is imported: STT, TTS, the action runner and the LLM all arrive
as protocols, which is invariant 4 and also what lets the whole module be tested
in the sandbox with no sound card and no Windows.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final, Protocol

from ayris.audio.stt.base import AudioBuffer
from ayris.core.errors import ActionError, AyrisError, LlmError, SttError, TtsError
from ayris.core.events import (
    ActionFailed,
    ActionFinished,
    ActionStarted,
    CancelRequested,
    EventBus,
    IntentMatched,
    SpeechEnded,
    TranscriptReady,
    WakeWordDetected,
)
from ayris.core.models import ExecutionResult, HistoryEntry, JsonObject
from ayris.core.pipeline_states import (
    ASSISTANT_STATES,
    BUSY_STATES,
    PipelineState,
    PipelineStateMachine,
    PipelineTimeouts,
    Scheduler,
)
from ayris.core.pipeline_trace import PipelineTrace, Stage, TraceRecord
from ayris.core.state import AssistantState, StateMachine
from ayris.nlu.context import DialogContext, ObjectKind
from ayris.nlu.followup import (
    CANCEL_REASON,
    AnswerStatus,
    FollowUpKind,
    PendingAnswer,
    answer_pending,
    publish_cancel,
    resolve_followup,
)
from ayris.nlu.llm.base import LlmClient, LlmMessage, LlmResponse, LlmTool, NullLlmClient
from ayris.nlu.matcher import Matcher, MatchResult
from ayris.nlu.normalize import normalize
from ayris.nlu.slot_types import SlotContext
from ayris.nlu.slots import SlotSet
from ayris.utils.logger import get_logger, get_pipeline_logger

if TYPE_CHECKING:
    from ayris.audio.stt.base import TranscriptResult
    from ayris.core.config import Settings

__all__ = [
    "ACTION_FAILED_MESSAGE",
    "CANCELLED_MESSAGE",
    "CANCEL_REASON_BARGE_IN",
    "CANCEL_REASON_SHUTDOWN",
    "CANCEL_REASON_TIMEOUT",
    "MAX_TRACES",
    "NOTHING_SAID_MESSAGE",
    "NOT_HEARD_MESSAGE",
    "NOT_MATCHED_MESSAGE",
    "SOURCE_PTT",
    "SOURCE_TEXT",
    "SOURCE_WAKE",
    "TIMEOUT_MESSAGE",
    "ActionOutcome",
    "ActionRequest",
    "ActionRunner",
    "HistorySink",
    "NluMode",
    "PhraseSource",
    "Pipeline",
    "PipelineResult",
    "Runner",
    "SpeechHandleLike",
    "SpeechOutput",
    "SttSource",
    "inline_runner",
    "mode_from_config",
    "thread_runner",
]

_log = get_logger(__name__)

#: What Ayris says when recognition came back with nothing. Short on purpose: it
#: is said after the user already waited, and «повторите, пожалуйста, я вас не
#: расслышала» costs another second of that wait.
NOT_HEARD_MESSAGE: Final = "Не расслышала."

#: …when the phrase was recognised but matches no command, in «только команды»
#: mode. Names the failure rather than apologising: the user has to know it is
#: the library that is missing the phrase, not the microphone.
NOT_MATCHED_MESSAGE: Final = "Не нашла такую команду."

#: …when the action failed and did not supply a message of its own.
ACTION_FAILED_MESSAGE: Final = "Не получилось выполнить."

#: …when a stage ran out of time.
TIMEOUT_MESSAGE: Final = "Не успела. Попробуй ещё раз."

#: …when the listening window closed with nobody speaking. Said only for an
#: explicit activation, see :meth:`Pipeline._on_timeout`.
NOTHING_SAID_MESSAGE: Final = "Ничего не услышала."

#: Value of :attr:`ayris.core.pipeline_trace.PipelineTrace.source` for the two
#: activations that are not a wake phrase.
SOURCE_PTT: Final = "ptt"
SOURCE_TEXT: Final = "text"

#: Fallback ``source`` when a wake word arrived without a phrase attached.
SOURCE_WAKE: Final = "wake"

#: How many finished traces DevTools can look back over. A trace is a few hundred
#: bytes and the tab shows a table, not a log.
MAX_TRACES: Final = 100

#: Reasons the pipeline puts on its own :class:`~ayris.core.events.CancelRequested`
#: so that its own subscription can tell them from a cancel the user asked for.
CANCEL_REASON_BARGE_IN: Final = "новая активация"
CANCEL_REASON_TIMEOUT: Final = "таймаут стадии"
CANCEL_REASON_SHUTDOWN: Final = "выключение"


class NluMode(StrEnum):
    """The three ways of understanding an utterance, from section 5.1.

    ``COMMANDS`` matches against the library and says so when nothing fits.
    ``HYBRID`` matches first and asks the model only when the matcher missed —
    the default, and the only mode where a missed match still costs a model
    request. ``AI`` sends everything to the model, library and all.
    """

    COMMANDS = "commands"
    HYBRID = "hybrid"
    AI = "ai"

    @property
    def uses_matcher(self) -> bool:
        """Whether the command library is consulted at all."""
        return self is not NluMode.AI

    @property
    def uses_llm(self) -> bool:
        """Whether a phrase the matcher missed reaches the model."""
        return self is not NluMode.COMMANDS


def mode_from_config(settings: Settings) -> NluMode:
    """Read the mode off the three «ИИ» toggles.

    The settings schema has three booleans where the spec has three modes,
    because each toggle is meaningful on its own in the settings window. The
    mapping lives here so that nothing else has to guess it: free chat wins, then
    either of the two command-related toggles makes it hybrid, and with all three
    off Ayris only knows what it was taught.
    """
    ai = settings.ai
    if ai.free_chat:
        return NluMode.AI
    if ai.fallback_to_llm or ai.llm_understanding:
        return NluMode.HYBRID
    return NluMode.COMMANDS


# ----------------------------------------------------------------------
# what the pipeline needs from everything around it
# ----------------------------------------------------------------------


class SttSource(Protocol):
    """Turns audio into text. :class:`~ayris.audio.stt.router.SttRouter` fits."""

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise one phrase, or raise :class:`~ayris.core.errors.SttError`."""


class SpeechHandleLike(Protocol):
    """A phrase being spoken. :class:`~ayris.audio.tts.router.SpeechHandle` fits."""

    @property
    def done(self) -> bool:
        """Whether the phrase has stopped, for any reason."""

    def cancel(self) -> bool:
        """Stop this phrase. ``True`` when there was something to stop."""

    def wait(self, timeout: float | None = None) -> bool:
        """Block until it stops. ``False`` on timeout."""


class SpeechOutput(Protocol):
    """Says things out loud. :class:`~ayris.audio.tts.router.TtsRouter` fits."""

    def say(self, text: str) -> SpeechHandleLike:
        """Speak ``text`` and return a handle to it."""


@dataclass(frozen=True, slots=True)
class ActionRequest:
    """What the pipeline asks the action layer to do.

    Task 19 brings the registry this is handed to. Keeping the request a
    dataclass rather than a call with six arguments means the registry can grow a
    field without the pipeline changing shape.
    """

    session_id: str
    command_id: int | None = None
    intent: str = ""
    slots: JsonObject = field(default_factory=dict)
    phrase: str = ""
    confirmed: bool = False

    @property
    def name(self) -> str:
        """Identifier for the log and the trace: the intent, or the command id."""
        if self.intent:
            return self.intent
        return f"command:{self.command_id}" if self.command_id is not None else ""


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """What came of running one action.

    ``speak`` is what the user hears. Empty is legitimate and common: turning the
    volume down is its own confirmation, and narrating it would be noise.
    """

    result: ExecutionResult = ExecutionResult.OK
    speak: str = ""
    detail: str = ""
    error: str = ""
    object_kind: ObjectKind | None = None
    object_name: str = ""
    object_value: str = ""
    dangerous: bool = False

    @property
    def ok(self) -> bool:
        """Whether the action did what it was asked."""
        return self.result is ExecutionResult.OK


class ActionRunner(Protocol):
    """Runs one matched command. Implemented by the registry of task 19."""

    def __call__(self, request: ActionRequest) -> ActionOutcome:
        """Execute ``request``, or raise :class:`~ayris.core.errors.ActionError`."""


class PhraseSource(Protocol):
    """Fetches the audio of the phrase that was just spoken.

    The audio worker does not put PCM on the event bus — a few hundred kilobytes
    per phrase would be the wrong thing to marshal through the UI thread — so
    :class:`~ayris.core.events.SpeechEnded` is only a notification and the
    payload is pulled with this. ``None`` means the segment was rejected or
    already consumed, and the session ends with «не расслышала».
    """

    def __call__(self) -> AudioBuffer | None:
        """The last accepted phrase, or ``None`` when there is none."""


class HistorySink(Protocol):
    """Where a finished session is written. ``Repositories.history`` fits."""

    def add(self, entry: HistoryEntry) -> HistoryEntry:
        """Store one pass through the pipeline."""


class Runner(Protocol):
    """Runs a session off the calling thread. The seam the tests replace."""

    def __call__(self, work: Callable[[], None]) -> None:
        """Arrange for ``work`` to run. May run it inline."""


def thread_runner(work: Callable[[], None]) -> None:
    """Default runner: one daemon thread per session.

    A thread rather than a pool, because there is at most one session at a time
    and a pool would only add a queue where a refusal is the correct behaviour.
    """
    threading.Thread(target=work, name="ayris-pipeline", daemon=True).start()


def inline_runner(work: Callable[[], None]) -> None:
    """Runner that runs the session on the calling thread.

    What :meth:`Pipeline.run_text` uses, since it has to return the result, and
    what the tests pass so that a pass is finished by the time the call returns.
    """
    work()


# ----------------------------------------------------------------------
# session
# ----------------------------------------------------------------------


class _Stop(Exception):  # noqa: N818 - not an error, a control-flow signal
    """Unwinds :meth:`Pipeline._process` for a pass that is already finished.

    Raised once the session has been closed and spoken for, so the only thing
    left to do is get out of the flow. Carries the outcome so that
    :meth:`Pipeline.run_text` reports what actually happened rather than the
    outcome of whichever ``except`` clause caught it.
    """

    def __init__(self, outcome: ExecutionResult) -> None:
        super().__init__(outcome.value)
        self.outcome = outcome


class _Cancelled(_Stop):
    """The session is no longer wanted: cancelled, timed out, or superseded."""

    def __init__(self) -> None:
        super().__init__(ExecutionResult.CANCELLED)


@dataclass(slots=True)
class _Session:
    """One pass in flight. Mutable, owned by the thread running it."""

    session_id: str
    trace: PipelineTrace
    mode: NluMode
    text: str = ""
    audio: AudioBuffer | None = None
    cancelled: bool = False
    reason: str = ""
    speech: SpeechHandleLike | None = None
    from_text: bool = False

    def mark_cancelled(self, reason: str) -> SpeechHandleLike | None:
        """Flag the session and hand back the speech to stop, if any."""
        self.cancelled = True
        if not self.reason:
            self.reason = reason
        return self.speech


@dataclass(frozen=True, slots=True)
class _Plan:
    """What the understanding stage decided, before anything acts on it.

    Exists so that :meth:`Pipeline._plan` can be pure: it is the only thing
    inside the ``NLU`` stage timer, and a plan that spoke for itself would put
    the synthesis time inside the parse time.
    """

    #: A command is chosen; go on to execute it.
    resolved: bool = False
    #: The user said «отмена» and nothing else.
    cancelled: bool = False
    #: What to say, when the whole answer is a sentence.
    speak: str = ""
    #: Outcome to close the session with, for everything but ``resolved``.
    outcome: ExecutionResult = ExecutionResult.OK

    @property
    def speak_only(self) -> bool:
        """Whether the answer is a sentence and there is nothing to run."""
        return not self.resolved and not self.cancelled and bool(self.speak)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """What :meth:`Pipeline.run_text` returns.

    Carries a frozen :class:`~ayris.core.pipeline_trace.TraceRecord` rather than
    the live trace, so a caller that keeps the result cannot watch it change.
    """

    session_id: str
    outcome: ExecutionResult = ExecutionResult.OK
    text: str = ""
    intent: str = ""
    command_id: int | None = None
    spoken: str = ""
    error: str = ""
    trace: TraceRecord | None = None

    @property
    def ok(self) -> bool:
        """Whether the pass did what the user asked."""
        return self.outcome is ExecutionResult.OK

    @property
    def matched(self) -> bool:
        """Whether the phrase resolved to something at all."""
        return self.outcome is not ExecutionResult.UNMATCHED


# ----------------------------------------------------------------------
# the pipeline
# ----------------------------------------------------------------------


class Pipeline:
    """Orchestrates one utterance at a time.

    Args:
        bus: The event bus. Subscribed to for activations and cancels, published
            onto at every stage.
        state: The four-state machine the overlay animates. Driven in step with
            the seven-stage one so the sphere follows without knowing the stages.
        matcher: The command library. ``None`` disables matching, which is what
            «только ИИ» amounts to before a library has been loaded.
        context: Dialogue memory. Consulted before the matcher for «отмена»,
            «повтори» and the answer to a pending question, and updated after a
            command runs.
        stt: Recognition. ``None`` means text-only, which is a legitimate
            configuration while no model is installed.
        tts: Speech. ``None`` means Ayris runs silently — every message still
            reaches the trace and the history.
        actions: What runs a matched command. ``None`` reports every match as
            unavailable, which is the state of the world until task 19.
        llm: The model. Defaults to :class:`~ayris.nlu.llm.base.NullLlmClient`,
            whose answer is the «ИИ не настроен» sentence.
        phrase_source: How the PCM of a finished phrase is fetched.
        history: Where finished sessions are written.
        settings: Read for the mode, the deadlines and the privacy flags.
            :meth:`apply_settings` re-reads them when the user changes one.
        timeouts: Overrides the deadlines derived from ``settings``. The timeout
            tests pass milliseconds here.
        scheduler: How deadlines are waited for. Replaced in tests.
        runner: How a voice session gets off the calling thread.
        clock: Monotonic seconds, for the trace.

    Thread safety: the session and the stage machine are guarded by one
    re-entrant lock, held only while deciding — never while a stage runs, or a
    cancel could not arrive during one. Speaking happens after the lock is
    released for the same reason. A stage runs on the runner's thread and touches
    only its own :class:`_Session`.
    """

    __slots__ = (
        "_actions",
        "_bus",
        "_clock",
        "_context",
        "_echo_guard",
        "_history",
        "_llm",
        "_lock",
        "_matcher",
        "_mode",
        "_phrase_source",
        "_pipeline_log",
        "_runner",
        "_session",
        "_settings",
        "_speak_gate",
        "_state",
        "_states",
        "_store_history",
        "_stt",
        "_subscriptions",
        "_traces",
        "_tts",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        state: StateMachine | None = None,
        matcher: Matcher | None = None,
        context: DialogContext | None = None,
        stt: SttSource | None = None,
        tts: SpeechOutput | None = None,
        actions: ActionRunner | None = None,
        llm: LlmClient | None = None,
        phrase_source: PhraseSource | None = None,
        history: HistorySink | None = None,
        settings: Settings | None = None,
        timeouts: PipelineTimeouts | None = None,
        scheduler: Scheduler | None = None,
        runner: Runner | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._bus = bus
        self._state = state
        self._matcher = matcher
        self._context = context
        self._stt = stt
        self._tts = tts
        self._actions = actions
        self._llm: LlmClient = llm if llm is not None else NullLlmClient()
        self._phrase_source = phrase_source
        self._history = history
        self._settings = settings
        self._runner: Runner = runner if runner is not None else thread_runner
        self._clock: Callable[[], float] = clock if clock is not None else perf_counter
        self._lock = threading.RLock()
        self._speak_gate = threading.Lock()
        self._session: _Session | None = None
        self._traces: list[TraceRecord] = []
        self._subscriptions: list[Callable[[], None]] = []
        self._mode = NluMode.HYBRID
        self._store_history = True
        self._pipeline_log = False
        self._echo_guard = False
        if settings is not None:
            self._read_settings(settings)
        self._states = PipelineStateMachine(
            bus,
            timeouts=self._initial_timeouts(settings, timeouts),
            on_timeout=self._on_timeout,
            scheduler=scheduler,
        )

    @staticmethod
    def _initial_timeouts(
        settings: Settings | None, timeouts: PipelineTimeouts | None
    ) -> PipelineTimeouts:
        if timeouts is not None:
            return timeouts
        if settings is not None:
            return PipelineTimeouts.from_settings(settings)
        return PipelineTimeouts()

    def _read_settings(self, settings: Settings) -> None:
        self._mode = mode_from_config(settings)
        self._store_history = settings.privacy.store_history
        self._pipeline_log = settings.devtools.pipeline_log
        # Whether the microphone is trusted while Ayris is speaking. On means an
        # activation during the answer is treated as the assistant's own voice
        # coming back through the speakers and is dropped; a hotkey or a typed
        # command ignores it, since neither can be an echo.
        self._echo_guard = not settings.voice.tts.interrupt_on_speech

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------

    def attach(self) -> None:
        """Subscribe to the events that drive and interrupt the pipeline.

        Not done in ``__init__``: a pipeline built for a test that only calls
        :meth:`run_text` has no business receiving wake words, and the
        composition root wants to choose the moment the assistant goes live.
        """
        if self._subscriptions:
            return
        self._subscriptions = [
            self._bus.subscribe(WakeWordDetected, self._on_wake_word, weak=False),
            self._bus.subscribe(SpeechEnded, self._on_speech_ended, weak=False),
            self._bus.subscribe(CancelRequested, self._on_cancel_requested, weak=False),
        ]

    def detach(self) -> None:
        """Stop listening. Idempotent."""
        for unsubscribe in self._subscriptions:
            unsubscribe()
        self._subscriptions = []

    def close(self) -> None:
        """Cancel whatever is in flight, unsubscribe, drop the deadline."""
        self.cancel(reason=CANCEL_REASON_SHUTDOWN)
        self.detach()
        self._states.close()

    def apply_settings(self, settings: Settings) -> None:
        """Adopt new settings. The mode may change mid-conversation.

        Takes effect from the next session: changing the mode under a phrase that
        is already being understood would make the trace lie about which path it
        took.
        """
        self._settings = settings
        self._read_settings(settings)
        self._states.apply_timeouts(PipelineTimeouts.from_settings(settings))

    def set_matcher(self, matcher: Matcher | None) -> None:
        """Swap the command library, e.g. on a profile switch."""
        self._matcher = matcher

    def set_llm(self, llm: LlmClient | None) -> None:
        """Install a real model, or take it away. Task 63 calls this."""
        self._llm = llm if llm is not None else NullLlmClient()

    # ------------------------------------------------------------------
    # what the outside can ask
    # ------------------------------------------------------------------

    @property
    def state(self) -> PipelineState:
        """Which stage the pipeline is in."""
        return self._states.state

    @property
    def mode(self) -> NluMode:
        """The understanding mode in force."""
        return self._mode

    @property
    def session_id(self) -> str:
        """Session in flight, or ``""`` when there is none."""
        with self._lock:
            return self._session.session_id if self._session is not None else ""

    @property
    def busy(self) -> bool:
        """Whether a microphone activation would be refused right now."""
        return self._states.state in BUSY_STATES

    def traces(self, limit: int = 0) -> tuple[TraceRecord, ...]:
        """Finished sessions, oldest first. What the DevTools tab renders."""
        with self._lock:
            records = tuple(self._traces)
        return records[-limit:] if limit > 0 else records

    def activate(self, *, source: str = "", phrase: str = "") -> str:
        """Start listening for a command. Returns the session id, or ``""``.

        The whole one-session rule is here, in one place:

        * idle — start;
        * speaking — cancel the answer and start (barge-in), unless the echo
          guard is on and the activation came from the microphone;
        * anything else — refuse and log, because an activation in the middle of
          recognising the previous phrase is not a second request.
        """
        explicit = source in _EXPLICIT_SOURCES
        with self._lock:
            current = self._states.state
            if current is PipelineState.RESPONDING:
                if self._echo_guard and not explicit:
                    _log.debug("активация во время озвучки отброшена: гейт эха")
                    return ""
                self._cancel_locked(reason=CANCEL_REASON_BARGE_IN, publish=True)
            elif current in BUSY_STATES:
                _log.info("активация отклонена: пайплайн занят (%s)", current.value)
                return ""
            session = self._begin_locked(source=source or phrase or SOURCE_WAKE)
            self._states.enter(PipelineState.LISTENING, session_id=session.session_id)
            self._sync_assistant_state(PipelineState.LISTENING)
        _log.info("сессия %s открыта (%s)", session.session_id, session.trace.source)
        return session.session_id

    def submit_audio(self, audio: AudioBuffer | None) -> str:
        """Hand the pipeline the phrase it was listening for.

        Called by :meth:`_on_speech_ended` with whatever ``phrase_source``
        returned, and directly by the tests. Without an open session the audio is
        dropped: a phrase nobody asked for is somebody talking in the room.
        """
        message = ""
        with self._lock:
            session = self._session
            if session is None or self._states.state not in _RECORDING_STATES:
                _log.debug("фрагмент без активной сессии — отброшен")
                return ""
            if audio is None or audio.is_empty:
                message = self._finish_locked(
                    session, ExecutionResult.ERROR, speak=NOT_HEARD_MESSAGE, error="empty segment"
                )
            else:
                session.audio = audio
                session.trace.record(Stage.RECORD, round(audio.duration_ms))
                self._states.enter(PipelineState.TRANSCRIBING, session_id=session.session_id)
                self._sync_assistant_state(PipelineState.TRANSCRIBING)
        if message:
            self._speak_detached(message)
            return ""
        self._runner(lambda: self._run_detached(session))
        return session.session_id

    def run_text(self, text: str) -> PipelineResult:
        """Run one phrase through the pipeline without audio.

        The DevTools text field and most of the tests come in here. The stages
        are the ones the voice path runs, minus the wake word and STT, and the
        pass runs on the calling thread so that the result can be returned.

        A typed command outranks whatever was happening: it came from a window,
        so it cannot be an echo or a misheard wake word, and making the user wait
        for an answer they are already reading would be silly.
        """
        phrase = text.strip()
        if not phrase:
            return PipelineResult(session_id="", outcome=ExecutionResult.ERROR, error="empty text")
        with self._lock:
            if self._states.state.is_active:
                self._cancel_locked(reason=CANCEL_REASON_BARGE_IN, publish=True)
            session = self._begin_locked(source=SOURCE_TEXT)
            session.from_text = True
            session.text = phrase
            session.trace.stt_raw = phrase
            self._states.enter(PipelineState.UNDERSTANDING, session_id=session.session_id)
            self._sync_assistant_state(PipelineState.UNDERSTANDING)
        return self._run(session)

    def cancel(self, *, reason: str = "") -> bool:
        """Abort the session in flight and return to idle.

        Returns whether there was anything to abort. Safe from any thread and at
        any stage: the stage that is running finds out at its next boundary.
        """
        with self._lock:
            return self._cancel_locked(reason=reason or CANCEL_REASON, publish=False)

    # ------------------------------------------------------------------
    # event handlers
    # ------------------------------------------------------------------

    def _on_wake_word(self, event: WakeWordDetected) -> None:
        source = SOURCE_PTT if event.phrase == _PTT_PHRASE else event.phrase
        self.activate(source=source, phrase=event.phrase)

    def _on_speech_ended(self, event: SpeechEnded) -> None:
        """A phrase finished. Fetch its audio and start recognising.

        A rejected segment — too little speech to be a phrase — ends the session
        with «не расслышала» rather than leaving it listening: the user has
        stopped talking, so the window would only make them wait for a timeout.
        """
        message = ""
        with self._lock:
            session = self._session
            if session is None:
                return
            if self._states.state is PipelineState.LISTENING:
                self._states.enter(PipelineState.RECORDING, session_id=session.session_id)
                self._sync_assistant_state(PipelineState.RECORDING)
            if event.reason == _REJECTED_REASON:
                message = self._finish_locked(
                    session, ExecutionResult.ERROR, speak=NOT_HEARD_MESSAGE, error=event.reason
                )
        if message:
            self._speak_detached(message)
            return
        source = self._phrase_source
        self.submit_audio(source() if source is not None else None)

    def _on_cancel_requested(self, event: CancelRequested) -> None:
        """Somebody said «отмена» — or the pipeline itself did.

        Its own cancels are recognised by their reason and ignored: the session
        was flagged before the event went out, and re-entering the cancel path
        from inside it would close the session the barge-in has just opened.
        """
        if event.reason in _OWN_CANCEL_REASONS:
            return
        self.cancel(reason=event.reason or CANCEL_REASON)

    def _on_timeout(self, state: PipelineState) -> None:
        """A stage ran out of time. Say so, unless there is nothing to say.

        A listening window that closed with nobody speaking is the one case that
        stays quiet for a wake word and speaks for a hotkey: the wake word fires
        on its own now and then, and an assistant that announces every false
        positive is worse than one that misses a phrase.
        """
        message = ""
        with self._lock:
            session = self._session
            if session is None:
                return
            speak = TIMEOUT_MESSAGE
            if state is PipelineState.LISTENING:
                speak = NOTHING_SAID_MESSAGE if session.trace.source == SOURCE_PTT else ""
            session.mark_cancelled(CANCEL_REASON_TIMEOUT)
            message = self._finish_locked(
                session,
                ExecutionResult.TIMEOUT,
                speak=speak,
                error=f"timeout in {state.value}",
            )
        if message:
            self._speak_detached(message)

    # ------------------------------------------------------------------
    # session bookkeeping, all under the lock
    # ------------------------------------------------------------------

    def _begin_locked(self, *, source: str) -> _Session:
        session_id = uuid.uuid4().hex[:12]
        trace = PipelineTrace(session_id=session_id, source=source, clock=self._clock)
        trace.mode = self._mode.value
        session = _Session(session_id=session_id, trace=trace, mode=self._mode)
        self._session = session
        return session

    def _cancel_locked(self, *, reason: str, publish: bool) -> bool:
        """Close the session in flight. Says nothing: a cancel is not an answer."""
        session = self._session
        if session is None:
            self._states.to_idle()
            return False
        # Flag before publishing: the pipeline subscribes to its own bus, and a
        # handler that ran against a session still marked live would close the
        # session that has just replaced it.
        speech = session.mark_cancelled(reason)
        if speech is not None:
            speech.cancel()
        if publish:
            self._bus.publish(CancelRequested(reason=reason))
        self._finish_locked(session, ExecutionResult.CANCELLED, error=reason)
        return True

    def _finish_locked(
        self,
        session: _Session,
        outcome: ExecutionResult,
        *,
        speak: str = "",
        error: str = "",
    ) -> str:
        """End ``session``: record it, return to idle, and report what to say.

        Speaking is deliberately left to the caller. This runs under the lock,
        and holding it through synthesis would block every cancel for as long as
        the answer lasts.
        """
        if self._session is session:
            self._session = None
        trace = session.trace
        if trace.outcome is ExecutionResult.OK:
            trace.outcome = outcome
        if error and not trace.error:
            trace.error = error
        if speak:
            trace.user_message = speak
            if not trace.answer:
                trace.answer = speak
        trace.finish()
        self._states.to_idle(detail=speak or error)
        self._sync_assistant_state(PipelineState.IDLE, failure=error if speak else "")
        self._record(session)
        return speak

    def _record(self, session: _Session) -> None:
        """Write the trace to the log, the ring buffer and ``history``."""
        trace = session.trace
        line = trace.log_line()
        get_pipeline_logger().info(line, extra={"request_id": trace.session_id})
        if self._pipeline_log:
            _log.info("%s", line)
        self._traces.append(trace.freeze())
        if len(self._traces) > MAX_TRACES:
            del self._traces[:-MAX_TRACES]
        if not self._store_history or self._history is None:
            return
        try:
            self._history.add(trace.to_history())
        except Exception as exc:
            # The user already got their answer; losing the history row is the
            # smaller loss, and a failing sink must not take the session with it.
            _log.warning("сессию %s не записать в историю: %r", trace.session_id, exc)

    def _sync_assistant_state(self, state: PipelineState, *, failure: str = "") -> None:
        """Drive the four-state machine the overlay animates."""
        machine = self._state
        if machine is None:
            return
        if failure:
            machine.fail(failure)
            return
        target = ASSISTANT_STATES[state]
        if target is AssistantState.IDLE:
            machine.to_idle()
        else:
            machine.set_state(target, force=True)

    # ------------------------------------------------------------------
    # the pass itself, off the lock
    # ------------------------------------------------------------------

    def _run_detached(self, session: _Session) -> None:
        """Run a voice session for its side effects. What the runner is handed.

        The result of a voice pass has nowhere to go — the answer was spoken and
        the trace is already in the ring buffer — so it is dropped here rather
        than widening :class:`Runner` to a callable returning anything.
        """
        self._run(session)

    def _run(self, session: _Session) -> PipelineResult:
        """Run every remaining stage, turning any failure into a sentence."""
        try:
            return self._process(session)
        except _Stop as stop:
            return self._result(session, stop.outcome)
        except AyrisError as exc:
            _log.warning("сессия %s: %s", session.session_id, exc.technical, exc_info=True)
            self._fail(session, exc.user_message, error=exc.technical)
            return self._result(session, ExecutionResult.ERROR)
        except Exception as exc:
            _log.exception("сессия %s: непредвиденная ошибка", session.session_id)
            self._fail(session, ACTION_FAILED_MESSAGE, error=repr(exc))
            return self._result(session, ExecutionResult.ERROR)

    def _process(self, session: _Session) -> PipelineResult:
        """The flow. Every ``_check`` and ``_enter`` is a cancel boundary."""
        if not session.from_text:
            self._transcribe(session)
        self._enter(session, PipelineState.UNDERSTANDING)
        self._understand(session)
        outcome = self._execute(session)
        self._check(session)
        if outcome.speak:
            self._respond(session, outcome.speak)
        self._settle(session, outcome)
        return self._result(session, session.trace.outcome)

    # -- stage: recognition --------------------------------------------

    def _transcribe(self, session: _Session) -> None:
        """Audio in, text out. Raises :class:`_Stop` when there is no text."""
        audio = session.audio
        if audio is None:
            raise SttError("no audio to transcribe", user_message=NOT_HEARD_MESSAGE)
        source = self._stt
        if source is None:
            raise SttError("no recognition engine configured")
        trace = session.trace
        with trace.stage(Stage.STT):
            result = source.transcribe(audio)
        trace.stt_engine = result.engine
        trace.stt_confidence = result.confidence
        trace.stt_raw = result.text
        self._check(session)
        if result.is_empty or not result.text.strip():
            self._fail(session, NOT_HEARD_MESSAGE)
            raise _Stop(ExecutionResult.ERROR)
        minimum = self._min_confidence()
        if 0.0 < result.confidence < minimum:
            _log.info(
                "фраза «%s» отброшена: уверенность %.2f < %.2f",
                result.text,
                result.confidence,
                minimum,
            )
            self._fail(session, NOT_HEARD_MESSAGE, error="confidence below threshold")
            raise _Stop(ExecutionResult.ERROR)
        session.text = result.text
        self._bus.publish(
            TranscriptReady(
                text=result.text,
                confidence=result.confidence,
                engine=result.engine,
                duration_ms=round(result.inference_ms),
                request_id=session.session_id,
            )
        )

    def _min_confidence(self) -> float:
        return self._settings.voice.stt.min_confidence if self._settings is not None else 0.0

    # -- stage: understanding ------------------------------------------

    def _understand(self, session: _Session) -> None:
        """Work out what was said. Raises :class:`_Stop` unless there is an action.

        Only a matched command lets the flow continue. A cancel, a repeat, an
        answer to a pending question, a phrase nothing matched and a chat reply
        from the model all have their answer already and nothing to run.
        """
        trace = session.trace
        with trace.stage(Stage.NLU):
            plan = self._plan(session)
        if plan.resolved:
            return
        if plan.cancelled:
            self._done(session, ExecutionResult.CANCELLED, speak=plan.speak)
            self._publish_cancel(session)
            raise _Stop(ExecutionResult.CANCELLED)
        if plan.speak_only:
            self._respond(session, plan.speak)
            self._done(session, plan.outcome)
            raise _Stop(plan.outcome)
        if not session.mode.uses_llm:
            self._fail(session, NOT_MATCHED_MESSAGE, outcome=ExecutionResult.UNMATCHED)
            raise _Stop(ExecutionResult.UNMATCHED)
        self._ask_llm(session)

    def _plan(self, session: _Session) -> _Plan:
        """Decide what the phrase means. Pure: parses and matches, never speaks.

        Order is the one task 17 fixed. A pending answer comes first, because
        «да» means yes only while a question is open; then the follow-ups,
        because «отмена» outranks anything in the library; then the matcher.
        """
        conversational = self._plan_conversational(session)
        if conversational is not None:
            return conversational
        if self._match(session):
            return _Plan(resolved=True)
        return _Plan()

    def _plan_conversational(self, session: _Session) -> _Plan | None:
        """Cancel, repeat, or the answer to a question Ayris asked."""
        context = self._context
        if context is None:
            return None
        normalized = normalize(session.text)
        answer = answer_pending(context.pending(), normalized, context=self._slot_context())
        if answer.handled:
            return self._plan_pending(session, answer)
        follow = resolve_followup(normalized, context.snapshot())
        context.touch()
        if not follow.handled:
            return None
        if follow.kind is FollowUpKind.CANCEL:
            context.cancel(reason=CANCEL_REASON)
            session.trace.intent = follow.kind.value
            return _Plan(cancelled=True, speak=follow.speak, outcome=ExecutionResult.CANCELLED)
        if follow.kind is FollowUpKind.CONFIRM and follow.pending is not None:
            context.set_pending(follow.pending)
            session.trace.intent = follow.kind.value
            return _Plan(speak=follow.pending.question)
        if follow.kind is FollowUpKind.REPEAT_ACTION and follow.command is not None:
            command = follow.command
            trace = session.trace
            trace.intent = command.intent or trace.intent
            trace.command_id = command.command_id
            trace.slots = {name: _plain(value) for name, value in command.slots.items()}
            trace.match_source = _SOURCE_REPEAT
            return _Plan(resolved=True)
        # REPEAT_ANSWER and UNAVAILABLE: the whole answer is the sentence.
        session.trace.intent = follow.kind.value
        return _Plan(speak=follow.speak)

    def _plan_pending(self, session: _Session, answer: PendingAnswer) -> _Plan | None:
        """Fold the answer to a pending question back into the session."""
        context = self._context
        if context is None:
            return None
        trace = session.trace
        status = answer.status
        if status is AnswerStatus.RETRY and answer.retry is not None:
            context.set_pending(answer.retry)
            trace.intent = f"pending:{status.value}"
            return _Plan(speak=answer.speak)
        context.clear_pending()
        if status in _PENDING_END_STATUSES:
            trace.intent = f"pending:{status.value}"
            outcome = (
                ExecutionResult.CANCELLED
                if status is AnswerStatus.CANCELLED
                else ExecutionResult.ERROR
            )
            return _Plan(speak=answer.speak or CANCELLED_MESSAGE, outcome=outcome)
        request = answer.request
        if request is None:
            return None
        trace.intent = request.intent or trace.intent
        trace.command_id = request.command_id
        trace.slots = {name: _plain(value) for name, value in (answer.slots or {}).items()}
        trace.match_source = _SOURCE_PENDING
        return _Plan(resolved=True)

    def _match(self, session: _Session) -> bool:
        """Ask the library. Fills the trace and publishes ``IntentMatched``."""
        trace = session.trace
        matcher = self._matcher
        if matcher is None or not session.mode.uses_matcher:
            return False
        context = self._context
        snapshot = context.snapshot() if context is not None else None
        result = matcher.match(session.text, context=snapshot)
        if result is None:
            return False
        trace.command_id = result.command_id
        trace.match_source = result.kind.value
        trace.match_score = result.score
        trace.intent = trace.intent or f"command:{result.command_id}"
        slots = self._bind_slots(matcher, result)
        if slots is not None:
            trace.slots = {name: _plain(value) for name, value in slots.as_dict().items()}
        self._bus.publish(
            IntentMatched(
                intent=trace.intent,
                command_id=result.command_id,
                confidence=result.score,
                source=result.kind.value,
                slots=dict(trace.slots),
                request_id=session.session_id,
            )
        )
        return True

    def _bind_slots(self, matcher: Matcher, result: MatchResult) -> SlotSet | None:
        try:
            return matcher.index.snapshot().bind_slots(result, self._slot_context())
        except AyrisError as exc:
            _log.warning("слоты триггера %s не разобрались: %s", result.trigger_id, exc)
            return None

    def _slot_context(self) -> SlotContext:
        """Everything a slot parser needs beyond the captured text."""
        return SlotContext()

    def _ask_llm(self, session: _Session) -> None:
        """Hand the phrase to the model. Always raises :class:`_Stop`.

        The «гибрид» fallback and all of «только ИИ». With
        :class:`~ayris.nlu.llm.base.NullLlmClient` this is where the «ИИ не
        настроен» sentence comes from, which is what the task file asks both
        model-using modes to say.
        """
        trace = session.trace
        client = self._llm
        with trace.stage(Stage.LLM):
            response = self._complete(session, client)
        trace.intent = trace.intent or _SOURCE_LLM
        trace.match_source = _SOURCE_LLM
        self._check(session)
        answer = response.text.strip()
        if not answer:
            self._fail(session, NOT_MATCHED_MESSAGE, outcome=ExecutionResult.UNMATCHED)
            raise _Stop(ExecutionResult.UNMATCHED)
        outcome = ExecutionResult.OK if client.configured else ExecutionResult.ERROR
        if not client.configured:
            trace.error = trace.error or "llm not configured"
        self._respond(session, answer)
        self._remember_answer(answer)
        self._done(session, outcome)
        raise _Stop(outcome)

    def _complete(self, session: _Session, client: LlmClient) -> LlmResponse:
        settings = self._settings
        try:
            return client.complete(
                self._prompt(session),
                self._tools(),
                temperature=settings.ai.temperature if settings is not None else None,
                max_tokens=settings.ai.max_tokens if settings is not None else None,
                cancel=lambda: session.cancelled,
            )
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(f"llm request failed: {exc!r}") from exc

    def _prompt(self, session: _Session) -> Sequence[LlmMessage]:
        settings = self._settings
        if settings is None:
            return [LlmMessage.user(session.text)]
        prompt = (
            settings.ai.chat_system_prompt
            if session.mode is NluMode.AI
            else settings.ai.nlu_system_prompt
        )
        messages = [LlmMessage.system(prompt)] if prompt else []
        context = self._context
        if context is not None:
            previous = context.snapshot().answer
            if previous is not None:
                messages.append(LlmMessage.assistant(previous.text))
        messages.append(LlmMessage.user(session.text))
        return messages

    def _tools(self) -> Sequence[LlmTool]:
        """Commands offered to the model as callable tools.

        Empty for now: turning the library into tool declarations needs the
        parameter schemas the action registry brings in task 19, and a tool the
        model cannot actually invoke would be worse than none.
        """
        return ()

    # -- stage: execution ----------------------------------------------

    def _execute(self, session: _Session) -> ActionOutcome:
        """Run the matched command."""
        trace = session.trace
        self._enter(session, PipelineState.EXECUTING)
        request = ActionRequest(
            session_id=session.session_id,
            command_id=trace.command_id,
            intent=trace.intent,
            slots=dict(trace.slots),
            phrase=session.text,
            confirmed=trace.match_source in _PREFILLED_SOURCES,
        )
        trace.action = request.name
        runner = self._actions
        if runner is None:
            raise ActionError(
                f"no action runner for {request.name or 'command'}",
                user_message=ACTION_FAILED_MESSAGE,
            )
        self._bus.publish(
            ActionStarted(
                action=request.name,
                command_id=request.command_id,
                request_id=session.session_id,
            )
        )
        try:
            with trace.stage(Stage.ACTION):
                outcome = runner(request)
        except ActionError as exc:
            trace.error = exc.technical
            self._bus.publish(
                ActionFailed(
                    action=request.name,
                    error=exc.technical,
                    user_message=exc.user_message,
                    request_id=session.session_id,
                )
            )
            raise
        trace.action_result = outcome.detail or outcome.speak
        trace.outcome = outcome.result
        if outcome.error:
            trace.error = outcome.error
        self._bus.publish(
            ActionFinished(
                action=request.name,
                result=trace.action_result,
                duration_ms=trace.duration_of(Stage.ACTION),
                request_id=session.session_id,
            )
        )
        if outcome.ok or outcome.speak:
            return outcome
        # An action that failed without a word to say still owes the user one:
        # silence after a command reads as «it worked».
        return ActionOutcome(
            result=outcome.result,
            speak=ACTION_FAILED_MESSAGE,
            detail=outcome.detail,
            error=outcome.error,
            dangerous=outcome.dangerous,
        )

    # -- stage: answering ----------------------------------------------

    def _respond(self, session: _Session, text: str) -> None:
        """Say ``text`` and wait for it, so barge-in has something to interrupt."""
        if not text:
            return
        trace = session.trace
        trace.answer = text
        self._enter(session, PipelineState.RESPONDING)
        output = self._tts
        if output is None:
            return
        try:
            with trace.stage(Stage.TTS):
                handle = output.say(text)
                session.speech = handle
                handle.wait()
        except TtsError as exc:
            # A voice that will not speak is no reason to lose the command that
            # already ran: log it, keep the answer in the trace, carry on.
            _log.warning("озвучка не удалась: %s", exc.technical)
        finally:
            session.speech = None

    def _speak_detached(self, text: str) -> None:
        """Say something that belongs to no session — a failure, a timeout.

        Fire-and-forget, and serialised by its own gate so that two failures in a
        row do not overlap in the speakers.
        """
        output = self._tts
        if output is None or not text:
            return
        with self._speak_gate:
            try:
                output.say(text)
            except TtsError as exc:
                _log.warning("сообщение «%s» не озвучено: %s", text, exc.technical)

    # -- finishing -----------------------------------------------------

    def _settle(self, session: _Session, outcome: ActionOutcome) -> None:
        """Remember what ran, then close the session."""
        context = self._context
        trace = session.trace
        if context is not None:
            context.remember_command(
                intent=trace.intent,
                action=trace.action,
                phrase=session.text,
                slots=trace.slots,
                result=trace.action_result,
                dangerous=outcome.dangerous,
                command_id=trace.command_id,
            )
            if outcome.object_kind is not None and outcome.object_name:
                context.remember_object(
                    outcome.object_kind, outcome.object_name, outcome.object_value
                )
            if outcome.speak:
                context.remember_answer(outcome.speak)
        self._done(session, outcome.result)

    def _remember_answer(self, text: str) -> None:
        context = self._context
        if context is not None and text:
            context.remember_answer(text)

    def _done(self, session: _Session, outcome: ExecutionResult, *, speak: str = "") -> None:
        """Close ``session`` cleanly. ``speak`` is for what was not said yet."""
        self._close(session, outcome, speak=speak)

    def _fail(
        self,
        session: _Session,
        message: str,
        *,
        outcome: ExecutionResult = ExecutionResult.ERROR,
        error: str = "",
    ) -> None:
        """End the session with a spoken message. Never raises."""
        session.trace.outcome = outcome
        self._close(session, outcome, speak=message, error=error)

    def _close(
        self,
        session: _Session,
        outcome: ExecutionResult,
        *,
        speak: str = "",
        error: str = "",
    ) -> None:
        message = ""
        with self._lock:
            if self._session is not session:
                return
            message = self._finish_locked(session, outcome, speak=speak, error=error)
        if message:
            self._speak_detached(message)

    def _publish_cancel(self, session: _Session) -> None:
        """Announce a voice cancel, once the session is already closed.

        Order matters: publishing first would reach :meth:`_on_cancel_requested`
        and close the session from under this pass.
        """
        session.mark_cancelled(CANCEL_REASON)
        publish_cancel(self._bus)

    def _result(self, session: _Session, outcome: ExecutionResult) -> PipelineResult:
        trace = session.trace
        trace.finish()
        return PipelineResult(
            session_id=session.session_id,
            outcome=outcome,
            text=trace.stt_raw,
            intent=trace.intent,
            command_id=trace.command_id,
            spoken=trace.answer,
            error=trace.error,
            trace=trace.freeze(),
        )

    # -- plumbing ------------------------------------------------------

    def _check(self, session: _Session) -> None:
        """Raise :class:`_Cancelled` when the session is no longer wanted."""
        if session.cancelled:
            raise _Cancelled

    def _enter(self, session: _Session, state: PipelineState) -> None:
        """Move to ``state``, unless the session was cancelled in the meantime.

        Deliberately not forced: every move the flow makes is in
        :data:`~ayris.core.pipeline_states.ALLOWED_TRANSITIONS`, so a refusal
        means the flow and the table disagree, and that is worth a log line
        rather than a silently bypassed invariant.
        """
        self._check(session)
        with self._lock:
            if self._session is not session:
                raise _Cancelled
            current = self._states.state
            if state is not current and not self._states.enter(
                state, session_id=session.session_id
            ):
                _log.warning("стадия %s недостижима из %s", state.value, current.value)
                return
            self._sync_assistant_state(state)

    def __repr__(self) -> str:
        return f"Pipeline({self.state.value}, mode={self._mode.value})"


# ----------------------------------------------------------------------
# module-level tables, kept out of the class body
# ----------------------------------------------------------------------

#: Phrase the audio worker reports for a hotkey activation. Copied rather than
#: imported so that the pipeline does not pull the wake word package — and with
#: it a possible engine import — into the main process.
_PTT_PHRASE: Final = "<ptt>"

#: ``SpeechEnded.reason`` for a segment the segmenter threw away.
_REJECTED_REASON: Final = "too_short"

#: What the history records for a question the user aborted.
CANCELLED_MESSAGE: Final = "Отменено."

_OWN_CANCEL_REASONS: Final[frozenset[str]] = frozenset(
    {CANCEL_REASON_BARGE_IN, CANCEL_REASON_TIMEOUT, CANCEL_REASON_SHUTDOWN}
)

#: Activations that cannot be Ayris hearing itself, and so ignore the echo guard.
_EXPLICIT_SOURCES: Final[frozenset[str]] = frozenset({SOURCE_PTT, SOURCE_TEXT})

#: States in which submitted audio is still expected.
_RECORDING_STATES: Final[frozenset[PipelineState]] = frozenset(
    {PipelineState.LISTENING, PipelineState.RECORDING}
)

#: Pending-answer statuses that end the pass where they are.
_PENDING_END_STATUSES: Final[frozenset[AnswerStatus]] = frozenset(
    {AnswerStatus.CANCELLED, AnswerStatus.DECLINED, AnswerStatus.EXHAUSTED}
)

#: ``trace.match_source`` for a command that was decided before the matcher ran.
_SOURCE_REPEAT: Final = "repeat"
_SOURCE_PENDING: Final = "pending"
_SOURCE_LLM: Final = "llm"

#: …and the two of them together: an action reached this way is already confirmed.
_PREFILLED_SOURCES: Final[frozenset[str]] = frozenset({_SOURCE_REPEAT, _SOURCE_PENDING})


def _plain(value: object) -> Any:
    """Reduce a parsed slot value to something JSON can hold."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
