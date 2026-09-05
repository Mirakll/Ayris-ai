"""The record a run leaves behind: every step, how long it took, how it ended.

Section 22 of the specification asks for an execution report with per-block
timings, and it is asked for by four different readers: the history tab shows the
outcome, the debugger of task 35 shows the steps in order, the DevTools timeline
shows their durations, and a test asserts on both. One frozen structure serves all
four, and the engine hands it back from every run — including the runs that never
started, where the outcome is the whole answer.

Timings nest rather than tile. A ``While`` block's duration contains the duration
of everything its body did, the same way a span contains its children, because the
useful question about a loop is how long the loop took. Sum
:attr:`ExecutionReport.steps` and you will get more than
:attr:`ExecutionReport.duration_ms`; read :attr:`StepRecord.offset_ms` and you can
draw the timeline that explains why.

Nothing here touches the event bus or the database. The engine publishes and task
33 records; this module only remembers.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from ayris.actions.macros.errors import MacroCancelledError
from ayris.core.models import ExecutionResult, utc_now

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

    from ayris.core.errors import MacroError

__all__ = [
    "ExecutionReport",
    "ReportBuilder",
    "RunOutcome",
    "StepDraft",
    "StepRecord",
    "StepStatus",
]


class StepStatus(StrEnum):
    """How one block ended.

    ``SKIPPED`` is a block that was not run at all — switched off in the editor, or a
    ``Case`` the ``Switch`` did not choose. It is recorded rather than omitted so the
    debugger can show that the block was considered.
    """

    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class RunOutcome(StrEnum):
    """How the whole run ended.

    ``COOLDOWN`` and ``DISABLED`` are the two outcomes of a run that never executed a
    block: the command was fired again inside its ``cooldown_ms``, or it is switched
    off. Both are answers, not errors, which is why they live in this enum and not in
    :mod:`ayris.actions.macros.errors`.
    """

    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


#: How an outcome is written into ``history`` and ``audit`` by task 33. ``COOLDOWN``
#: and ``DISABLED`` become ``DENIED``: from the history's point of view the command
#: was asked for and refused, which is the same row shape as a refused confirmation.
_EXECUTION: Final[dict[RunOutcome, ExecutionResult]] = {
    RunOutcome.SUCCESS: ExecutionResult.OK,
    RunOutcome.FAILED: ExecutionResult.ERROR,
    RunOutcome.CANCELLED: ExecutionResult.CANCELLED,
    RunOutcome.TIMEOUT: ExecutionResult.TIMEOUT,
    RunOutcome.COOLDOWN: ExecutionResult.DENIED,
    RunOutcome.DISABLED: ExecutionResult.DENIED,
}


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One executed block: where it is, what it did, how long it took.

    ``path`` is the reader's spelling of the position in the tree —
    ``actions[1].then[0]`` — so a record can be matched back to a block in the editor
    without carrying the block itself.
    """

    path: str
    block: str
    status: StepStatus = StepStatus.OK
    duration_ms: int = 0
    offset_ms: int = 0
    depth: int = 0
    message: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """Whether this block did what it was asked to."""
        return self.status is StepStatus.OK

    def as_line(self) -> str:
        """One line for the log and the debugger: path, block, status, duration."""
        tail = self.error or self.message
        indent = "  " * self.depth
        head = f"{indent}{self.path} {self.block} [{self.status}] {self.duration_ms} мс"
        return f"{head} — {tail}" if tail else head


@dataclass(slots=True)
class StepDraft:
    """A block being executed right now, before its record is frozen.

    The mutable half of :class:`StepRecord`. A block handler reports what it did into
    the draft — the branch an ``If`` took, the number of turns a ``While`` made — and
    :meth:`ReportBuilder.measure` stamps the timing and freezes it.
    """

    path: str
    block: str
    depth: int = 0
    status: StepStatus = StepStatus.OK
    message: str = ""
    error: str = ""

    def fail(self, error: BaseException | str) -> None:
        """Mark the step failed, taking the message from ``error``."""
        self.status = StepStatus.FAILED
        self.error = str(error) if isinstance(error, str) else _error_text(error)

    def cancel(self) -> None:
        """Mark the step cancelled: it was interrupted, not broken."""
        self.status = StepStatus.CANCELLED


def _error_text(error: BaseException) -> str:
    """Short one-line description of an exception, for a step record."""
    text = str(error).strip() or type(error).__name__
    return text if len(text) <= 200 else f"{text[:197]}..."


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """What one run of one command amounts to.

    Always returned, never optional: a command stopped by its cooldown gets a report
    with no steps and :attr:`RunOutcome.COOLDOWN`, so a caller reads one field to
    learn what happened instead of telling an exception from a ``None``.

    ``error`` is the typed exception that ended the run, kept as an object rather
    than a string so the caller can match on its class and show
    :attr:`~ayris.core.errors.AyrisError.user_message` without parsing anything.
    """

    run_id: str
    command: str = ""
    command_id: int | None = None
    outcome: RunOutcome = RunOutcome.SUCCESS
    duration_ms: int = 0
    steps: tuple[StepRecord, ...] = ()
    value: Any = None
    error: MacroError | None = None
    message_ru: str = ""
    trigger: str = ""
    request_id: str = ""
    started_at: datetime = field(default_factory=utc_now)

    @property
    def ok(self) -> bool:
        """Whether the command ran to the end without a stopping failure."""
        return self.outcome is RunOutcome.SUCCESS

    @property
    def cancelled(self) -> bool:
        """Whether the run was stopped on purpose."""
        return self.outcome is RunOutcome.CANCELLED

    @property
    def failed(self) -> bool:
        """Whether the run ended on a failure or a timeout."""
        return self.outcome in (RunOutcome.FAILED, RunOutcome.TIMEOUT)

    @property
    def ran(self) -> bool:
        """Whether any block was executed at all."""
        return bool(self.steps)

    @property
    def execution(self) -> ExecutionResult:
        """How this run is written into ``history`` and ``audit``."""
        return _EXECUTION[self.outcome]

    @property
    def failures(self) -> tuple[StepRecord, ...]:
        """Steps that failed — more than one when the policy was ``continue``."""
        return tuple(step for step in self.steps if step.status is StepStatus.FAILED)

    @property
    def slowest(self) -> StepRecord | None:
        """The longest step, or ``None`` when nothing ran. What the timeline points at."""
        if not self.steps:
            return None
        return max(self.steps, key=lambda step: step.duration_ms)

    @property
    def user_message(self) -> str:
        """One Russian line for the overlay: the summary, or what the error says."""
        if self.message_ru:
            return self.message_ru
        return self.error.user_message if self.error is not None else ""

    def step(self, path: str) -> StepRecord | None:
        """The record of the block at ``path``, or ``None``.

        The first one when a block ran more than once — a block inside a ``While``
        appears once per turn, and the caller that wants them all filters
        :attr:`steps` itself.
        """
        return next((step for step in self.steps if step.path == path), None)

    def raise_for_status(self) -> None:
        """Re-raise the error that ended the run, for a caller that prefers exceptions.

        Does nothing when the run succeeded, was cancelled, or never started: those
        are outcomes to read, not failures to raise.
        """
        if self.failed and self.error is not None:
            raise self.error

    def as_text(self) -> str:
        """The whole report as lines, for a log entry or a debugger pane."""
        head = f"{self.command or self.run_id}: {self.outcome} за {self.duration_ms} мс"
        return "\n".join([head, *(step.as_line() for step in self.steps)])


def _ms(seconds: float) -> int:
    """Seconds as whole milliseconds, never negative."""
    return max(0, round(seconds * 1000))


class ReportBuilder:
    """Collects steps while a command runs, then freezes them into a report.

    One builder per run, used from the single thread that walks that command's tree,
    so nothing here locks. ``on_step`` is how the engine turns a finished step into a
    :class:`~ayris.core.events.MacroBlockFinished` without this module knowing that
    an event bus exists.

    ``clock`` is injected for the same reason the registry injects one: a test that
    asserts on a duration should not have to wait for real milliseconds to pass.
    """

    def __init__(
        self,
        *,
        run_id: str,
        command: str = "",
        command_id: int | None = None,
        trigger: str = "",
        request_id: str = "",
        clock: Callable[[], float] = time.perf_counter,
        on_step: Callable[[StepRecord], None] | None = None,
    ) -> None:
        self._run_id = run_id
        self._command = command
        self._command_id = command_id
        self._trigger = trigger
        self._request_id = request_id
        self._clock = clock
        self._on_step = on_step
        self._steps: list[StepRecord] = []
        self._open: list[StepDraft] = []
        self._started_at = utc_now()
        self._started = clock()

    @property
    def steps(self) -> tuple[StepRecord, ...]:
        """Everything recorded so far, in the order it happened."""
        return tuple(self._steps)

    @property
    def count(self) -> int:
        """How many blocks have been recorded. The engine's step budget counts these."""
        return len(self._steps)

    @property
    def elapsed_ms(self) -> int:
        """Milliseconds since the builder was made — the run's own clock."""
        return _ms(self._clock() - self._started)

    def note(self, message: str) -> None:
        """Describe what the block being measured did, for its record.

        Silently does nothing outside a :meth:`measure` block, so a handler can report
        without checking whether anyone is listening.
        """
        if self._open:
            self._open[-1].message = message

    def mark(
        self,
        path: str,
        block: str,
        status: StepStatus,
        *,
        depth: int = 0,
        message: str = "",
    ) -> StepRecord:
        """Record a block that took no time: one that was skipped or never entered."""
        draft = StepDraft(path=path, block=block, depth=depth, status=status, message=message)
        return self._append(draft, self._clock() - self._started, 0.0)

    @contextmanager
    def measure(self, path: str, block: str, *, depth: int = 0) -> Iterator[StepDraft]:
        """Time one block and record it, whatever way it ends.

        An exception passing through is recorded as a failure and re-raised: the engine
        decides what a failure means for the chain, and a block that broke has to be in
        the report either way. A cancellation is the exception to that — the block was
        interrupted, not broken, and every block still open above it was interrupted too,
        which is what the reader of a stopped run needs to see.
        """
        draft = StepDraft(path=path, block=block, depth=depth)
        offset = self._clock() - self._started
        self._open.append(draft)
        started = self._clock()
        try:
            yield draft
        except MacroCancelledError:
            if draft.status is StepStatus.OK:
                draft.cancel()
            raise
        except BaseException as exc:
            if draft.status is StepStatus.OK:
                draft.fail(exc)
            raise
        finally:
            self._open.pop()
            self._append(draft, offset, self._clock() - started)

    def finish(
        self,
        outcome: RunOutcome,
        *,
        value: Any = None,
        error: MacroError | None = None,
        message_ru: str = "",
    ) -> ExecutionReport:
        """Freeze everything collected into the report the engine returns."""
        return ExecutionReport(
            run_id=self._run_id,
            command=self._command,
            command_id=self._command_id,
            outcome=outcome,
            duration_ms=self.elapsed_ms,
            steps=tuple(self._steps),
            value=value,
            error=error,
            message_ru=message_ru,
            trigger=self._trigger,
            request_id=self._request_id,
            started_at=self._started_at,
        )

    def _append(self, draft: StepDraft, offset: float, duration: float) -> StepRecord:
        """Freeze one draft, remember it, and tell whoever asked to be told."""
        record = StepRecord(
            path=draft.path,
            block=draft.block,
            status=draft.status,
            duration_ms=_ms(duration),
            offset_ms=_ms(offset),
            depth=draft.depth,
            message=draft.message,
            error=draft.error,
        )
        self._steps.append(record)
        if self._on_step is not None:
            self._on_step(record)
        return record
