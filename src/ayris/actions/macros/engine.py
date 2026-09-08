"""The interpreter: one command's block tree, walked on a pool thread.

Task 30 defined what a command *is* and checked that it makes sense; this module
runs it. The shape of a run is fixed by four questions sections 7 and 22 of the
specification ask, and the answers are the whole design.

**Who runs it.** Not the UI thread — invariant 2. :meth:`MacroEngine.start` puts the
run on a :class:`~concurrent.futures.ThreadPoolExecutor` and hands back a
:class:`MacroRun` immediately, so a hotkey handler returns in microseconds and the
overlay keeps animating while ``RunApp`` waits for Chrome to come up.

**What happens when two fire at once.** :class:`ConcurrencyPolicy` decides: ``parallel``
lets both run, ``queue`` makes the second wait, ``preempt`` lets a command with a
higher priority cancel the ones already running. A command fired again inside its own
``cooldown_ms`` does not run at all and says so — in the log and in
:class:`~ayris.core.events.MacroSkipped` — because silence there looks to the user
like a broken hotkey.

**How it stops.** Cooperatively. Every block asks whether its run's
:class:`threading.Event` is set, and ``Wait`` sleeps *on* that event instead of in
:func:`time.sleep`, so a stop word reaches a command sleeping for ten seconds in the
time it takes to schedule a thread. Nothing is killed from outside: a half-finished
``SetVolume`` cannot be undone by killing the thread that called it.

**What it leaves behind.** An :class:`~ayris.actions.macros.report.ExecutionReport`
with one record per block, always — including for the runs that never started. Events
go onto the bus for the overlay and the debugger; the report goes back to the caller,
because it is too big for a bus that also carries audio levels and nobody but the
caller wants all of it.

Three things this module deliberately does not do. It does not evaluate expressions
itself — :mod:`ayris.actions.macros.expressions` does. It does not implement the blocks
that are not actions: :mod:`ayris.actions.macros.blocks` does, behind
:class:`~ayris.actions.macros.blocks.BlockRuntime`, and what is left here is the walk
those blocks run inside — the pool, the gate, the report, the budget. And it does not
play the sound bound to a block: nothing plays sounds yet, and when something does it
will be an action like any other.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol

from ayris.actions.macros.blocks import BLOCK_HANDLERS, Flow, short
from ayris.actions.macros.context import (
    ExecutionContext,
    MemoryVariables,
    RunInfo,
    TriggerSource,
)
from ayris.actions.macros.errors import (
    MacroBlockError,
    MacroCallError,
    MacroCancelledError,
    MacroEngineStoppedError,
    MacroLimitError,
    MacroRuntimeError,
    MacroTimeoutError,
)
from ayris.actions.macros.report import ExecutionReport, ReportBuilder, RunOutcome, StepStatus
from ayris.actions.macros.schema import MAX_BLOCK_DEPTH, MAX_BLOCKS, OnError, SoundStage
from ayris.actions.macros.sounds.binding import SoundBindingPlayer, bindings_for_stage
from ayris.core.errors import AyrisError, MacroError
from ayris.core.events import (
    CancelRequested,
    MacroBlockFinished,
    MacroCancelled,
    MacroFailed,
    MacroFinished,
    MacroSkipped,
    MacroStarted,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.macros.context import VariableStore
    from ayris.actions.macros.report import StepRecord
    from ayris.actions.macros.schema import ActionBlock, CommandModel
    from ayris.actions.result import ActionResult
    from ayris.core.events import Event, EventBus, MacroEnded, Unsubscribe
    from ayris.nlu.slots import SlotSet

__all__ = [
    "ActionRunner",
    "CommandLibrary",
    "ConcurrencyPolicy",
    "ExecutionLimits",
    "MacroEngine",
    "MacroRun",
]

_log = get_logger(__name__)


class ActionRunner(Protocol):
    """The part of :class:`~ayris.actions.registry.ActionRegistry` the interpreter uses.

    A protocol rather than the class itself for two reasons: invariant 5 says macros
    reach actions only through the registry, and a test that wants to know in which
    order ``SetVolume`` and ``PlaySound`` were called should not have to register real
    actions that change the machine's volume to find out.
    """

    def has(self, name: str) -> bool:
        """Whether an action of this name is registered."""
        ...

    def execute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        """Run one action and give back its result."""
        ...


#: How ``CallCommand`` finds the command it names. Task 33 owns the library; the engine
#: only needs one question answered, so it takes the answer as a function instead of
#: importing a store that does not exist yet.
CommandLibrary = Callable[[str], "CommandModel | None"]


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    """The ceilings a run cannot cross. Every one of them has a story.

    ``max_depth`` and ``max_steps`` come from task 30, where the same numbers stop a
    command being *saved*; repeating them here is not duplication, because a command
    can also arrive from a ``.ayris`` file written by hand or by an older version.

    ``max_iterations`` is the one the user meets: a ``While`` whose condition never
    turns false is the commonest way to write a command that hangs, and a ``While``
    block may lower the ceiling for itself but not raise it.

    ``timeout_s`` bounds the run as a whole, ``max_wait_ms`` bounds one ``Wait``, and
    ``max_call_depth`` stops a command that calls itself. Zero means "no limit" for the
    two that are times, because a command driving a long install genuinely takes minutes
    and the user who wrote it knows that better than this module does.
    """

    max_depth: int = MAX_BLOCK_DEPTH
    max_steps: int = MAX_BLOCKS
    max_iterations: int = 1000
    max_call_depth: int = 8
    timeout_s: float = 60.0
    max_wait_ms: int = 300_000

    @property
    def budget_ms(self) -> int:
        """The run's whole allowance in milliseconds, or ``0`` when it has none."""
        return int(self.timeout_s * 1000) if self.timeout_s > 0 else 0


class ConcurrencyPolicy(StrEnum):
    """What the engine does when a command fires while another one is running.

    ``PARALLEL`` is the default because most commands are independent: "громкость 50"
    and "рабочий режим" have no reason to wait for each other. ``QUEUE`` is for a
    machine where they do — two commands both driving the mixer, for instance.
    ``PREEMPT`` is the one section 7 asks for by name: a command with a higher priority
    cancels what is running instead of queueing behind it, so "стоп" or "экстренный
    режим" does not wait out a five-minute loop.
    """

    PARALLEL = "parallel"
    QUEUE = "queue"
    PREEMPT = "preempt"


@dataclass(slots=True)
class MacroRun:
    """One run in flight: how to wait for it, and how to stop it.

    What :meth:`MacroEngine.start` gives back. The caller holds this and nothing else:
    a hotkey handler drops it, the editor's "run once" button keeps it to grey out while
    it runs, and ``CallCommand`` with ``wait: true`` waits on it.

    A run that never started — cooldown, or a switched-off command — comes back with
    :attr:`report` already filled in, so a caller reads the same field either way
    instead of telling a skipped run from a started one.
    """

    run_id: str
    command: str
    command_id: int | None = None
    priority: int = 0
    trigger: TriggerSource = TriggerSource.MANUAL
    request_id: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    context: ExecutionContext | None = None
    future: Future[ExecutionReport] | None = None
    report: ExecutionReport | None = None
    reason: str = ""
    admitted: bool = False

    @property
    def running(self) -> bool:
        """Whether this run is still going."""
        return self.report is None

    @property
    def cancelled(self) -> bool:
        """Whether someone has asked this run to stop."""
        return self.cancel.is_set()

    def stop(self, reason: str = "") -> None:
        """Ask the run to stop at its next block, or right now if it is in a ``Wait``."""
        if not self.cancel.is_set():
            self.reason = reason
        self.cancel.set()

    def wait(self, timeout: float | None = None) -> ExecutionReport:
        """The report, waiting for the run to finish if it has not.

        Raises:
            TimeoutError: ``timeout`` passed and the run is still going. The run keeps
                going: waiting is not cancelling, and a caller that wanted both calls
                :meth:`stop` first.
        """
        if self.report is not None:
            return self.report
        if self.future is None:  # pragma: no cover - set before the run can be seen
            raise MacroEngineStoppedError("run was never scheduled")
        return self.future.result(timeout)


def _run_id() -> str:
    """A short id for one run. Short because it goes into every log line and event."""
    return uuid.uuid4().hex[:12]


class _Runner:
    """One command's tree, walked once, on one thread.

    Separate from :class:`MacroEngine` because the state of a walk — where it is, what
    it has recorded, what a ``Return`` left behind — belongs to one run and not to the
    engine that may be running four of them at once. Everything here happens on the
    pool thread the run was handed to, so nothing in this class locks: the only object
    shared with another thread is :attr:`MacroRun.cancel`.

    A nested ``CallCommand`` builds a second runner over the same cancel event and the
    same variable store, with its own context and its own report — see :meth:`call`.

    The blocks themselves are not here. This class is the
    :class:`~ayris.actions.macros.blocks.BlockRuntime` the blocks of
    :mod:`ayris.actions.macros.blocks` run behind: it satisfies that protocol
    structurally, which is why the ``handler(self, ...)`` call in :meth:`_dispatch`
    type-checks without either module importing the other's implementation.
    """

    __slots__ = ("_builder", "_command", "_context", "_engine", "_limits", "_run", "_value")

    def __init__(
        self,
        engine: MacroEngine,
        run: MacroRun,
        *,
        command: CommandModel,
        context: ExecutionContext,
        builder: ReportBuilder,
        limits: ExecutionLimits,
    ) -> None:
        self._engine = engine
        self._run = run
        self._command = command
        self._context = context
        self._builder = builder
        self._limits = limits
        self._value: Any = None

    @property
    def value(self) -> Any:
        """What a ``Return`` block left behind, or ``None``. Goes into the report."""
        return self._value

    @property
    def context(self) -> ExecutionContext:
        """The variables, slots and expression evaluator of this run."""
        return self._context

    @property
    def report(self) -> ReportBuilder:
        """Where a block writes down what it did, one record per block."""
        return self._builder

    @property
    def limits(self) -> ExecutionLimits:
        """The ceilings this run may not cross."""
        return self._limits

    @property
    def reason(self) -> str:
        """Why the run was stopped, when it was. What a cancellation message says."""
        return self._run.reason

    def execute(self) -> Flow:
        """Walk the command's top-level blocks. The whole run, in one call."""
        return self.walk(self._command.actions, "actions", 0)

    def walk(self, blocks: Iterable[ActionBlock], prefix: str, depth: int) -> Flow:
        """Run a list of blocks in order, stopping at the first one that redirects."""
        for index, block in enumerate(blocks):
            flow = self.run_one(block, f"{prefix}[{index}]", depth)
            if flow is not Flow.NEXT:
                return flow
        return Flow.NEXT

    def check(self) -> None:
        """Ask whether the run may go on. The protocol's name for :meth:`_guard`."""
        self._guard()

    def pause(self, ms: int) -> bool:
        """Sleep up to ``ms`` on the run's cancel event, answering whether it was set.

        The event and not :func:`time.sleep`, so a stop word reaches a command waiting ten
        seconds in the time it takes to schedule a thread.
        """
        return self._run.cancel.wait(ms / 1000)

    def returned(self, value: Any) -> None:
        """Remember what a ``Return`` block left for whoever called this command."""
        self._value = value

    def publish(self, event: object) -> None:
        """Publish an event through the engine's guarded bus boundary."""
        self._engine._publish(event)  # type: ignore[arg-type]

    def _guard(self) -> None:
        """The three questions asked before every block: stopped, too long, too many.

        Raises:
            MacroCancelledError: someone set the run's event.
            MacroLimitError: the run has executed as many blocks as it is allowed to.
            MacroTimeoutError: the run has used up its whole budget.
        """
        if self._run.cancel.is_set():
            raise MacroCancelledError(self._run.reason)
        if self._builder.count >= self._limits.max_steps:
            raise MacroLimitError("steps", self._limits.max_steps)
        budget = self._limits.budget_ms
        if budget and self._builder.elapsed_ms > budget:
            raise MacroTimeoutError(self._limits.timeout_s)

    def run_one(self, block: ActionBlock, path: str, depth: int) -> Flow:
        """Run one block: guard, record, dispatch, and decide what a failure means."""
        self._guard()
        if not block.enabled:
            self._builder.mark(
                path, block.type, StepStatus.SKIPPED, depth=depth, message="выключен"
            )
            return Flow.NEXT
        if depth > self._limits.max_depth:
            raise MacroLimitError(
                "depth", depth, user_message="Блоки команды вложены слишком глубоко."
            )
        try:
            self._engine._play_sounds(
                self._command, SoundStage.ON_START, block=block, owner=self._run.run_id
            )
            with self._builder.measure(path, block.type, depth=depth):
                flow = self._dispatch(block, path, depth)
            self._engine._play_sounds(
                self._command, SoundStage.ON_SUCCESS, block=block, owner=self._run.run_id
            )
            return flow
        except (MacroCancelledError, MacroLimitError):
            raise
        except Exception as exc:
            self._engine._play_sounds(
                self._command, SoundStage.ON_ERROR, block=block, owner=self._run.run_id
            )
            return self._recover(block, path, exc)

    def _dispatch(self, block: ActionBlock, path: str, depth: int) -> Flow:
        """Hand the block to its handler, or to the registry when it has none."""
        handler = BLOCK_HANDLERS.get(block.type)
        if handler is not None:
            return handler(self, block, path, depth)
        return self._action(block, path)

    def _recover(self, block: ActionBlock, path: str, exc: Exception) -> Flow:
        """What a broken block means for the chain: ``continue`` swallows, ``stop`` raises.

        Raises:
            MacroRuntimeError: the block's policy is ``stop``. The error carries the path
                it happened at, for the report, the log and the editor.
        """
        error = _as_block_error(block, path, exc)
        if block.on_error is OnError.CONTINUE:
            _log.warning("макрос %s: %s упал, продолжаю — %s", self._run.command, path, error)
            return Flow.NEXT
        _log.error("макрос %s: %s упал, останавливаю — %s", self._run.command, path, error)
        raise error

    def _action(self, block: ActionBlock, path: str) -> Flow:
        """One action block: the registry runs it, and its result becomes ``last_result``.

        Invariant 5 in one method. The interpreter knows an action's name and its
        parameters and nothing else about it — no import, no class, no guess about what it
        is going to do. Parameters are filled before the call, so ``{volume}`` reaches
        ``SetVolume`` as ``50`` and never as text with braces still in it.

        Raises:
            MacroBlockError: no action of that name is registered, or it refused.
        """
        if not self._engine.registry.has(block.type):
            raise MacroBlockError(
                f"unknown action {block.type!r}",
                path=path,
                block=block.type,
                user_message=f"Действие «{block.type}» не подключено.",
            )
        result = self._engine.registry.execute(
            block.type,
            self._context.fill(block.params),
            request_id=self._run.request_id,
            command_id=self._run.command_id,
        )
        self._context.set_result(result.value, action=result)
        self._builder.note(result.message_ru or short(result.value))
        if not result.ok:
            raise MacroBlockError(
                result.detail or f"action {block.type} refused",
                path=path,
                block=block.type,
                user_message=result.message_ru or None,
            )
        return Flow.NEXT

    def call(self, name: str, args: Mapping[str, Any], *, wait: bool) -> None:
        """``CallCommand``, the half only the engine can do: find the command and run it.

        Inline and on this thread when the caller waits, on purpose. A nested run handed to
        the pool would deadlock the first time the policy is ``queue`` — the caller holding
        a worker while it waits for a run that is waiting for the caller — and would cost a
        second worker for nothing, because the caller has no work to do meanwhile. What the
        nested run shares is this run's cancel event, so one stop word stops the whole
        stack; what it gets of its own is a context, a report and its own pair of events, so
        the history shows two commands and not one.

        ``wait: false`` is the other half: the call goes to the engine as an independent
        run, and the block is finished as soon as that run has been scheduled.

        Raises:
            MacroCallError: no such command, or the command is switched off.
        """
        target = self._engine.find(name)
        if target is None:
            raise MacroCallError(name)
        if not target.enabled:
            raise MacroCallError(name, f"command {name!r} is switched off")
        if wait:
            self._nested(target, args)
            return
        started = self._engine.start(
            target, slots=args, trigger=TriggerSource.CALL, request_id=self._run.request_id
        )
        self._builder.note(f"«{name}» запущена отдельно: {started.run_id}")

    def _nested(self, target: CommandModel, args: Mapping[str, Any]) -> None:
        """Run a called command here and now, and fold its report into this one.

        The called command gets the arguments as its slots: a command reads what it was
        given the same way whether a phrase filled it or a caller did, which is why
        ``{name}`` works in both.

        Raises:
            MacroLimitError: the stack of commands is as deep as it may get.
            MacroTimeoutError: nothing is left of the budget to give the called command.
            MacroCancelledError: the called run was stopped, which stops this one too.
            MacroRuntimeError: the called command failed — its own error, with its own path.
        """
        info = self._context.info
        if info.depth >= self._limits.max_call_depth:
            raise MacroLimitError(
                "call_depth",
                info.depth,
                user_message="Команды вызывают друг друга слишком глубоко.",
            )
        limits = self._limits
        if limits.budget_ms:
            left = limits.budget_ms - self._builder.elapsed_ms
            if left <= 0:
                raise MacroTimeoutError(limits.timeout_s)
            limits = replace(limits, timeout_s=left / 1000)
        nested = info.called(target.name, run_id=_run_id(), command_id=target.id)
        run = MacroRun(
            run_id=nested.run_id,
            command=target.name,
            command_id=target.id,
            priority=target.priority,
            trigger=TriggerSource.CALL,
            request_id=self._run.request_id,
            cancel=self._run.cancel,
            admitted=True,
        )
        context = self._context.child(nested, slots=args, variables=target.variables)
        report = self._engine.perform(run, target, context, limits=limits)
        self._builder.note(f"«{target.name}»: {report.outcome} за {report.duration_ms} мс")
        self._context.set_result(report.value)
        if report.cancelled:
            raise MacroCancelledError(self._run.reason)
        if report.error is not None:
            raise report.error


def _as_block_error(block: ActionBlock, path: str, exc: Exception) -> MacroRuntimeError:
    """One block's failure as an error that knows where it happened.

    An error already carrying a path is given back untouched, so a failure inside a nested
    ``CallCommand`` keeps pointing at the block that really broke instead of at the
    ``CallCommand`` above it.
    """
    if isinstance(exc, MacroBlockError):
        if not exc.path:
            exc.path = path
            exc.block = exc.block or block.type
        return exc
    return MacroBlockError(
        str(exc) or type(exc).__name__,
        path=path,
        block=block.type,
        user_message=exc.user_message if isinstance(exc, AyrisError) else None,
        cause=exc,
    )


def _as_macro_error(exc: AyrisError) -> MacroError:
    """Any Ayris failure as the macro failure a report can carry."""
    if isinstance(exc, MacroError):
        return exc
    return MacroRuntimeError(
        exc.technical, user_message=exc.user_message, recoverable=exc.recoverable
    )


def _ended_event(report: ExecutionReport) -> MacroEnded:
    """The event that says how a run ended: finished, failed, or cancelled.

    Three classes over one base, so the overlay can subscribe to
    :class:`~ayris.core.events.MacroEnded` and hear all three while the history tab
    subscribes to the failure alone.
    """
    if report.cancelled:
        return MacroCancelled(
            run_id=report.run_id,
            command=report.command,
            command_id=report.command_id,
            outcome=str(report.outcome),
            duration_ms=report.duration_ms,
            steps=len(report.steps),
            request_id=report.request_id,
            reason=report.error.reason if isinstance(report.error, MacroCancelledError) else "",
        )
    if report.failed:
        return MacroFailed(
            run_id=report.run_id,
            command=report.command,
            command_id=report.command_id,
            outcome=str(report.outcome),
            duration_ms=report.duration_ms,
            steps=len(report.steps),
            request_id=report.request_id,
            error=str(report.error) if report.error is not None else "",
            user_message=report.user_message,
            path=report.error.path if isinstance(report.error, MacroBlockError) else "",
        )
    return MacroFinished(
        run_id=report.run_id,
        command=report.command,
        command_id=report.command_id,
        outcome=str(report.outcome),
        duration_ms=report.duration_ms,
        steps=len(report.steps),
        request_id=report.request_id,
        message_ru=report.message_ru,
    )


def _default_threads() -> int:
    """How many commands may run at once, from ``performance.macro_threads``."""
    from ayris.core.config import get_settings

    try:
        return max(1, int(get_settings().performance.macro_threads))
    except AyrisError:
        return 4


class MacroEngine:
    """The front door: what runs a command, on which thread, and how many at a time.

    One engine per application, made once and shut down once. Everything the interpreter
    needs that outlives a single run lives here — the pool, the shared variable store, the
    cooldown table, the gate concurrent runs pass through — and everything that does not
    lives in :class:`_Runner`.

    ``registry`` is the only argument without a default, because a command that cannot
    reach an action is not a command. The ``bus`` is optional so a test can run a command
    without listening to anything, and so is the ``library``, which only ``CallCommand``
    needs. Give ``clock`` a fake to test a cooldown without waiting one out.
    """

    def __init__(
        self,
        registry: ActionRunner,
        *,
        bus: EventBus | None = None,
        store: VariableStore | None = None,
        library: CommandLibrary | None = None,
        limits: ExecutionLimits | None = None,
        policy: ConcurrencyPolicy = ConcurrencyPolicy.PARALLEL,
        threads: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        sounds: SoundBindingPlayer | None = None,
    ) -> None:
        self.registry = registry
        self._bus = bus
        self._store: VariableStore = MemoryVariables() if store is None else store
        self._library = library
        self._limits = limits if limits is not None else ExecutionLimits()
        self._policy = policy
        self._clock = clock
        self._sounds = sounds
        self._pool = ThreadPoolExecutor(
            max_workers=threads if threads is not None else _default_threads(),
            thread_name_prefix="macro",
        )
        self._lock = threading.Condition()
        self._active: dict[str, MacroRun] = {}
        self._cooldown: dict[str, float] = {}
        self._running = True
        self._unsubscribe: Unsubscribe | None = None
        if bus is not None:
            self._unsubscribe = bus.subscribe(CancelRequested, self._on_cancel)

    @property
    def limits(self) -> ExecutionLimits:
        """The ceilings every run of this engine is held to."""
        return self._limits

    @property
    def policy(self) -> ConcurrencyPolicy:
        """What this engine does when a command fires while another one is running."""
        return self._policy

    @property
    def variables(self) -> VariableStore:
        """Where ``profile`` and ``global`` variables live, shared by every run."""
        return self._store

    @property
    def running(self) -> bool:
        """Whether the engine still takes new runs."""
        return self._running

    @property
    def active(self) -> tuple[MacroRun, ...]:
        """The runs the engine is holding right now, in the order they arrived."""
        with self._lock:
            return tuple(self._active.values())

    def find(self, name: str) -> CommandModel | None:
        """The command of this name, when the engine was given a library to look in."""
        return None if self._library is None else self._library(name)

    def start(
        self,
        command: CommandModel,
        *,
        slots: SlotSet | Mapping[str, Any] | None = None,
        trigger: TriggerSource = TriggerSource.MANUAL,
        request_id: str = "",
        priority: int | None = None,
    ) -> MacroRun:
        """Put a command on a pool thread and give the handle back at once.

        Invariant 2 lives in the two lines that submit and return: a hotkey handler is back
        in microseconds, and the overlay keeps animating while ``RunApp`` waits for Chrome
        to come up.

        A command that will not run comes back already finished, with a report that says
        why — switched off, or fired again inside its own ``cooldown_ms``. Silence there
        looks to the user like a broken hotkey, so the reason is logged and goes onto the
        bus as :class:`~ayris.core.events.MacroSkipped` too.

        Raises:
            MacroEngineStoppedError: the engine has been shut down.
        """
        if not self._running:
            raise MacroEngineStoppedError(f"engine is stopped, cannot run {command.name!r}")
        run = MacroRun(
            run_id=_run_id(),
            command=command.name,
            command_id=command.id,
            priority=command.priority if priority is None else priority,
            trigger=trigger,
            request_id=request_id,
        )
        if not command.enabled:
            return self._refuse(run, command, RunOutcome.DISABLED, "Команда выключена.")
        left = self._cooldown_left(command)
        if left > 0:
            return self._refuse(
                run,
                command,
                RunOutcome.COOLDOWN,
                f"Команда ещё не остыла — осталось {left} мс.",
                retry_after_ms=left,
            )
        run.context = ExecutionContext(
            info=RunInfo(
                run_id=run.run_id,
                command=command.name,
                command_id=command.id,
                trigger=trigger,
                request_id=request_id,
            ),
            store=self._store,
            slots=slots,
            variables=command.variables,
        )
        return self._submit(run, command)

    def run(
        self,
        command: CommandModel,
        *,
        slots: SlotSet | Mapping[str, Any] | None = None,
        trigger: TriggerSource = TriggerSource.MANUAL,
        request_id: str = "",
        priority: int | None = None,
        timeout: float | None = None,
    ) -> ExecutionReport:
        """Run a command and wait for its report — for a test, a script, or a one-off.

        Not for the UI thread: this one waits. A hotkey calls :meth:`start`.
        """
        started = self.start(
            command, slots=slots, trigger=trigger, request_id=request_id, priority=priority
        )
        return started.wait(timeout)

    def _submit(self, run: MacroRun, command: CommandModel) -> MacroRun:
        """Register the run at the gate, then hand it to the pool.

        Registered before it is submitted rather than when a worker picks it up: a run
        waiting for a free worker is exactly the run the stop word has to be able to reach.

        Raises:
            MacroEngineStoppedError: the pool was shut down between the check and here.
        """
        with self._lock:
            self._active[run.run_id] = run
        try:
            run.future = self._pool.submit(self._task, run, command)
        except RuntimeError as exc:
            self._leave(run)
            raise MacroEngineStoppedError(f"pool refused {command.name!r}: {exc}") from exc
        return run

    def _task(self, run: MacroRun, command: CommandModel) -> ExecutionReport:
        """The pool thread's whole job: pass the gate, run the command, leave the gate."""
        context = run.context
        if context is None:  # pragma: no cover - start fills it in before submitting
            raise MacroEngineStoppedError(f"run {run.run_id} has no context")
        try:
            self._admit(run)
            return self.perform(run, command, context)
        finally:
            self._leave(run)

    def perform(
        self,
        run: MacroRun,
        command: CommandModel,
        context: ExecutionContext,
        *,
        limits: ExecutionLimits | None = None,
    ) -> ExecutionReport:
        """Walk one command's tree to the end and produce its report.

        The thread-facing half of the engine, and the entry point a nested ``CallCommand``
        uses directly: it takes the context and the limits it is given instead of making
        its own, so a called command inherits what is left of the caller's budget rather
        than starting a fresh minute.

        The variable store is flushed once here, when the walk is over — not on every
        ``SetVar``. A ``While`` that counts to a thousand in a ``persistent`` variable would
        otherwise be a thousand writes to SQLite for one number nobody read in between.
        """
        builder = ReportBuilder(
            run_id=run.run_id,
            command=command.name,
            command_id=command.id,
            trigger=str(run.trigger),
            request_id=run.request_id,
            on_step=partial(self._on_step, run),
        )
        self._publish(
            MacroStarted(
                run_id=run.run_id,
                command=command.name,
                command_id=command.id,
                trigger=str(run.trigger),
                request_id=run.request_id,
            )
        )
        self._play_sounds(command, SoundStage.ON_START, owner=run.run_id)
        report = self._walk(run, command, context, builder, limits or self._limits)
        self._store.flush()
        _log.info(
            "макрос «%s»: %s, %d шагов за %d мс",
            command.name,
            report.outcome,
            len(report.steps),
            report.duration_ms,
        )
        # The event goes out before the report is published to whoever is waiting.
        # ``MacroRun.wait`` returns the moment :attr:`MacroRun.report` is set, so the other
        # order lets a caller act on a finished run before the bus has heard it ended — and
        # a subscriber that redraws on the event would be a redraw behind.
        self._publish(_ended_event(report))
        stage = SoundStage.ON_SUCCESS if report.ok else SoundStage.ON_ERROR
        self._play_sounds(command, stage, owner=run.run_id)
        run.report = report
        return report

    def _play_sounds(
        self,
        command: CommandModel,
        stage: SoundStage,
        *,
        block: ActionBlock | None = None,
        owner: str = "",
    ) -> None:
        """Play effective bindings without allowing accompaniment to mask the command."""
        if self._sounds is None:
            return
        for binding in bindings_for_stage(command, stage, block):
            try:
                self._sounds.play_binding(binding, owner=owner)
            except Exception as exc:
                _log.warning(
                    "макрос «%s»: звук %s не проигран — %s",
                    command.name,
                    binding.reference,
                    exc,
                )

    def _walk(
        self,
        run: MacroRun,
        command: CommandModel,
        context: ExecutionContext,
        builder: ReportBuilder,
        limits: ExecutionLimits,
    ) -> ExecutionReport:
        """Run the tree and turn however it ended into one outcome.

        Every exception the interpreter can raise is caught here and nowhere else: the
        report is the engine's whole answer, and a caller who would rather have an
        exception asks the report to re-raise it.

        A bare ``Exception`` reaching this far is a bug in Ayris rather than in the
        command, so it is logged with its traceback and then reported like any other
        failure — one broken action must not take the assistant down with it.
        """
        runner = _Runner(
            self,
            run,
            command=command,
            context=context,
            builder=builder,
            limits=limits,
        )
        try:
            runner.execute()
        except MacroCancelledError as cancelled:
            _log.info("макрос «%s» отменён: %s", command.name, cancelled.reason or "по запросу")
            return builder.finish(RunOutcome.CANCELLED, error=cancelled, message_ru="Отменено.")
        except MacroTimeoutError as expired:
            return builder.finish(
                RunOutcome.TIMEOUT, error=expired, message_ru=expired.user_message
            )
        except AyrisError as broke:
            error = _as_macro_error(broke)
            _log.warning("макрос «%s» не выполнен: %s", command.name, error.technical)
            return builder.finish(RunOutcome.FAILED, error=error, message_ru=error.user_message)
        except Exception as unexpected:
            _log.exception("макрос «%s» упал неожиданно", command.name)
            error = MacroRuntimeError(f"unexpected failure in {command.name!r}: {unexpected}")
            return builder.finish(RunOutcome.FAILED, error=error, message_ru=error.user_message)
        return builder.finish(RunOutcome.SUCCESS, value=runner.value)

    def _refuse(
        self,
        run: MacroRun,
        command: CommandModel,
        outcome: RunOutcome,
        reason: str,
        *,
        retry_after_ms: int = 0,
    ) -> MacroRun:
        """A run that never started: give it a report, write down why, and say so aloud.

        The handle comes back finished rather than ``None``, so a caller reads
        :attr:`MacroRun.report` the same way whether the command ran or not.
        """
        builder = ReportBuilder(
            run_id=run.run_id,
            command=command.name,
            command_id=command.id,
            trigger=str(run.trigger),
            request_id=run.request_id,
        )
        run.reason = reason
        run.report = builder.finish(outcome, message_ru=reason)
        _log.info("макрос «%s» не запущен: %s", command.name, reason)
        self._publish(
            MacroSkipped(
                command=command.name,
                command_id=command.id,
                reason=str(outcome),
                retry_after_ms=retry_after_ms,
                trigger=str(run.trigger),
                request_id=run.request_id,
            )
        )
        return run

    def _leave(self, run: MacroRun) -> None:
        """Take a finished run off the gate and wake whoever is waiting behind it."""
        with self._lock:
            self._active.pop(run.run_id, None)
            self._lock.notify_all()

    def _admit(self, run: MacroRun) -> None:
        """Hold the run at the gate until the policy lets it through.

        ``PARALLEL`` lets everything through and the pool's worker count is the only
        limit. ``QUEUE`` runs one command at a time, in arrival order. ``PREEMPT`` is the
        answer section 22 asks for when two triggers land together: the newcomer cancels
        every peer it strictly outranks, and a newcomer that outranks nobody waits its
        turn like ``QUEUE``.

        A cancelled peer counts as gone. Cancellation is cooperative, so it is already
        winding down and will do nothing else; waiting for it to actually stop would put
        the whole point of preempting behind the run being preempted.

        Never raises. A run cancelled while it waited still reaches
        :meth:`_Runner.execute`, whose first act is to check the cancel event, so there is
        one way out of a run and not two.
        """
        if self._policy is ConcurrencyPolicy.PARALLEL:
            run.admitted = True
            return
        with self._lock:
            while self._running and not run.cancelled:
                peers = [
                    other
                    for other in self._active.values()
                    if other.run_id != run.run_id and other.admitted and not other.cancelled
                ]
                if not peers:
                    break
                if self._policy is ConcurrencyPolicy.PREEMPT:
                    for other in peers:
                        if other.priority < run.priority:
                            other.stop(f"команда «{run.command}» важнее")
                self._lock.wait(0.05)  # a cancel of this very run has to be noticed too
            run.admitted = True

    def _cooldown_left(self, command: CommandModel) -> int:
        """Milliseconds this command still has to cool down; at ``0`` it takes the slot.

        Asking and taking happen under one lock deliberately. Two hotkey presses in the
        same millisecond ask at the same time, and if the answer arrived before the mark
        was made, both would be told to go ahead. A refused run does not push the cooldown
        forward: it never ran, so the clock keeps running from the start that did.
        """
        if command.cooldown_ms <= 0:
            return 0
        key = str(command.id) if command.id is not None else command.name
        now = self._clock()
        with self._lock:
            started = self._cooldown.get(key)
            if started is not None:
                left = command.cooldown_ms - (now - started) * 1000
                if left > 0:
                    return max(1, round(left))
            self._cooldown[key] = now
        return 0

    def _on_step(self, run: MacroRun, record: StepRecord) -> None:
        """Announce one finished block, so a timeline can be drawn as the command runs."""
        self._publish(
            MacroBlockFinished(
                run_id=run.run_id,
                path=record.path,
                block=record.block,
                status=str(record.status),
                duration_ms=record.duration_ms,
                depth=record.depth,
                message=record.error or record.message,
                request_id=run.request_id,
            )
        )

    def _publish(self, event: Event) -> None:
        """Put an event on the bus when there is one, and never let it break a run.

        A handler that raises is a bug in the subscriber. The command it interrupted did
        nothing wrong, so the exception is logged and swallowed here.
        """
        if self._bus is None:
            return
        try:
            self._bus.publish(event)
        except Exception:
            _log.exception("подписчик упал на событии %s", type(event).__name__)

    def cancel(self, run_id: str = "", *, command: str = "", reason: str = "") -> int:
        """Ask the matching runs to stop, and say how many were asked.

        Asking, not stopping: a run notices at its next block, or wakes out of its
        ``Wait``, and its report comes back with :attr:`RunOutcome.CANCELLED`. With neither
        argument it means every run — which is what the stop word means.
        """
        with self._lock:
            targets = [
                run
                for run in self._active.values()
                if (not run_id or run.run_id == run_id) and (not command or run.command == command)
            ]
        for run in targets:
            run.stop(reason)
        with self._lock:
            self._lock.notify_all()
        if targets:
            _log.info("отмена макросов: %d, причина — %s", len(targets), reason or "не указана")
        return len(targets)

    def cancel_all(self, reason: str = "") -> int:
        """Every run at once: the stop word, the panic hotkey, the application closing."""
        return self.cancel(reason=reason)

    def reset_cooldowns(self, command: CommandModel | None = None) -> None:
        """Forget one command's cooldown, or all of them. For the editor and for tests."""
        with self._lock:
            if command is None:
                self._cooldown.clear()
                return
            key = str(command.id) if command.id is not None else command.name
            self._cooldown.pop(key, None)

    def _on_cancel(self, event: CancelRequested) -> None:
        """The stop word and the cancel hotkey reach the engine here, through the bus."""
        self.cancel(reason=event.reason or "по запросу")

    def shutdown(self, *, wait: bool = True, reason: str = "выход") -> None:
        """Stop taking runs, ask the running ones to stop, and let the pool go.

        Idempotent, because closing an application calls things twice. Unsubscribing from
        the bus first matters: a stop word arriving during the shutdown must not queue a
        cancel against runs that are already gone.

        ``wait=False`` is for a caller that cannot afford to block. With the default the
        call returns when the pool's threads are done, which after the cancel is the time
        the slowest action needs to come back — an action already inside a WinAPI call
        cannot be interrupted, only waited out.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        stopped = self.cancel_all(reason)
        self._pool.shutdown(wait=wait)
        _log.info("движок макросов остановлен, прерванных запусков: %d", stopped)

    def __enter__(self) -> MacroEngine:
        """The engine as a context manager, so a test cannot leak a thread pool."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Shut down on the way out, whether the block ended well or badly."""
        self.shutdown()
