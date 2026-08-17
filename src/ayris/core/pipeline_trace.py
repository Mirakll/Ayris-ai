"""What happened during one pass, with a stopwatch on every stage.

Section 15 of the spec asks for one line per request in the form
``STT raw → NLU intent → Action → Result`` with the duration of each stage in
milliseconds. That line is not the whole story though: DevTools wants the same
data structured, and the ``history`` table wants a row. All three come from the
same object, filled in as the pass proceeds — writing the log line separately
from the history row is how the two drift apart.

:class:`PipelineTrace` is therefore mutable, which is unusual for this codebase.
It is owned by exactly one session and touched from the thread running that
session, and :meth:`PipelineTrace.freeze` hands out an immutable
:class:`TraceRecord` for anything that outlives the pass — the DevTools ring
buffer keeps those, not the live traces.

Timing goes through an injectable ``clock`` (``time.perf_counter`` by default) so
a test can assert on exact millisecond figures instead of on «greater than
zero», which is the only assertion a real clock allows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter
from typing import Any, Final

from ayris.core.models import ExecutionResult, HistoryEntry, JsonObject, utc_now

__all__ = [
    "STAGE_LABELS",
    "PipelineTrace",
    "Stage",
    "StageTiming",
    "TraceRecord",
]


class Stage(StrEnum):
    """A timed step of one pass.

    Not the same list as :class:`~ayris.core.pipeline_states.PipelineState`: a
    state is where the pipeline *is*, a stage is what it *spent time on*.
    ``UNDERSTANDING`` covers both the matcher and the model, and the trace has to
    keep those apart — a hybrid pass that fell through to the LLM should show
    where the two seconds went.
    """

    RECORD = "record"
    STT = "stt"
    NLU = "nlu"
    LLM = "llm"
    ACTION = "action"
    TTS = "tts"

    @property
    def label(self) -> str:
        """Russian name for DevTools."""
        return STAGE_LABELS[self]


STAGE_LABELS: Final[Mapping[Stage, str]] = {
    Stage.RECORD: "запись",
    Stage.STT: "распознавание",
    Stage.NLU: "разбор",
    Stage.LLM: "модель",
    Stage.ACTION: "действие",
    Stage.TTS: "озвучка",
}

#: What goes in the log line where a stage produced nothing to show.
_EMPTY: Final = "—"


@dataclass(frozen=True, slots=True)
class StageTiming:
    """How long one stage took and whether it got where it was going."""

    stage: Stage
    duration_ms: int
    ok: bool = True
    detail: str = ""

    def describe(self) -> str:
        """``распознавание 240 мс``, with a marker when the stage failed."""
        mark = "" if self.ok else " ✗"
        return f"{self.stage.label} {self.duration_ms} мс{mark}"

    def as_json(self) -> JsonObject:
        return {
            "stage": self.stage.value,
            "duration_ms": self.duration_ms,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass(slots=True)
class PipelineTrace:
    """Everything one pass through the pipeline produced.

    Args:
        session_id: The identifier every event of this pass carries.
        source: How the pass started — a wake phrase, ``ptt``, ``text``.
        clock: Monotonic seconds. Injected so timeout and timing tests are exact.

    Fields are filled in by the pipeline as each stage completes; anything a pass
    never reached keeps its empty default, which is what makes a half-finished
    trace still loggable.
    """

    session_id: str
    source: str = ""
    clock: Callable[[], float] = perf_counter
    stt_raw: str = ""
    stt_engine: str = ""
    stt_confidence: float = 0.0
    mode: str = ""
    intent: str = ""
    command_id: int | None = None
    match_source: str = ""
    match_score: float = 0.0
    slots: JsonObject = field(default_factory=dict)
    action: str = ""
    action_result: str = ""
    answer: str = ""
    outcome: ExecutionResult = ExecutionResult.OK
    error: str = ""
    user_message: str = ""
    stages: list[StageTiming] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def __post_init__(self) -> None:
        if self.started_at == 0.0:
            self.started_at = self.clock()

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, stage: Stage) -> Iterator[None]:
        """Time the block and record it, whether it returns or raises.

        A stage that raises is recorded with ``ok=False`` before the exception
        continues upwards, so the trace of a failed pass still shows how long the
        failing stage took. That is the number that tells a timeout apart from a
        refusal.
        """
        started = self.clock()
        ok = True
        try:
            yield
        except BaseException:
            ok = False
            raise
        finally:
            self.record(stage, self._elapsed_ms(started), ok=ok)

    def record(self, stage: Stage, duration_ms: int, *, ok: bool = True, detail: str = "") -> None:
        """Add a timing measured elsewhere — a worker reports its own ms."""
        self.stages.append(StageTiming(stage=stage, duration_ms=duration_ms, ok=ok, detail=detail))

    def finish(self) -> None:
        """Stop the total stopwatch. Idempotent, so a cancel path may repeat it."""
        if self.finished_at == 0.0:
            self.finished_at = self.clock()

    @property
    def total_ms(self) -> int:
        """Wall time of the whole pass, still counting while it runs."""
        end = self.finished_at if self.finished_at > 0.0 else self.clock()
        return max(0, round((end - self.started_at) * 1000))

    def duration_of(self, stage: Stage) -> int:
        """Total ms spent in ``stage``, ``0`` when the pass never reached it."""
        return sum(timing.duration_ms for timing in self.stages if timing.stage is stage)

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self.clock() - started) * 1000))

    # ------------------------------------------------------------------
    # output
    # ------------------------------------------------------------------

    def log_line(self) -> str:
        """The section 15 line: ``STT raw → NLU intent → Action → Result``."""
        stages = " | ".join(timing.describe() for timing in self.stages)
        parts = [
            f"[{self.session_id}]",
            f"STT raw: {self.stt_raw or _EMPTY}",
            "→",
            f"NLU intent: {self.intent or _EMPTY}",
            "→",
            f"Action: {self.action or _EMPTY}",
            "→",
            f"Result: {self.outcome.value}",
        ]
        line = " ".join(parts)
        if self.error:
            line = f"{line} ({self.error})"
        return f"{line} — {self.total_ms} мс" + (f" [{stages}]" if stages else "")

    def as_json(self) -> JsonObject:
        """Structured form for the DevTools pipeline tab."""
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "source": self.source,
            "mode": self.mode,
            "stt_raw": self.stt_raw,
            "stt_engine": self.stt_engine,
            "stt_confidence": round(self.stt_confidence, 3),
            "intent": self.intent,
            "command_id": self.command_id,
            "match_source": self.match_source,
            "match_score": round(self.match_score, 3),
            "slots": dict(self.slots),
            "action": self.action,
            "action_result": self.action_result,
            "answer": self.answer,
            "outcome": self.outcome.value,
            "error": self.error,
            "user_message": self.user_message,
            "total_ms": self.total_ms,
            "stages": [timing.as_json() for timing in self.stages],
        }
        return payload

    def to_history(self) -> HistoryEntry:
        """The ``history`` row for this pass.

        ``params`` carries the slots plus the few facts that make a row readable
        on its own — the mode it ran in, the answer that was spoken — because the
        table has no column for them and a history entry that cannot explain
        itself is not worth keeping.
        """
        params: JsonObject = dict(self.slots)
        params["session_id"] = self.session_id
        if self.source:
            params["source"] = self.source
        if self.mode:
            params["mode"] = self.mode
        if self.answer:
            params["answer"] = self.answer
        if self.match_source:
            params["match_source"] = self.match_source
        timings = {timing.stage.value: timing.duration_ms for timing in self.stages}
        if timings:
            params["stages_ms"] = timings
        return HistoryEntry(
            ts=utc_now(),
            stt_raw=self.stt_raw,
            matched_command_id=self.command_id,
            intent=self.intent,
            params=params,
            result=self.outcome,
            error=self.error,
            duration_ms=self.total_ms,
        )

    def freeze(self) -> TraceRecord:
        """Immutable snapshot, safe to keep after the session ended."""
        return TraceRecord(
            session_id=self.session_id,
            source=self.source,
            mode=self.mode,
            stt_raw=self.stt_raw,
            intent=self.intent,
            command_id=self.command_id,
            action=self.action,
            answer=self.answer,
            outcome=self.outcome,
            error=self.error,
            user_message=self.user_message,
            total_ms=self.total_ms,
            stages=tuple(self.stages),
            payload=self.as_json(),
        )


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """A finished :class:`PipelineTrace`. What DevTools keeps in its ring buffer."""

    session_id: str
    source: str = ""
    mode: str = ""
    stt_raw: str = ""
    intent: str = ""
    command_id: int | None = None
    action: str = ""
    answer: str = ""
    outcome: ExecutionResult = ExecutionResult.OK
    error: str = ""
    user_message: str = ""
    total_ms: int = 0
    stages: Sequence[StageTiming] = ()
    payload: JsonObject = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Whether the pass did what the user asked."""
        return self.outcome is ExecutionResult.OK

    def duration_of(self, stage: Stage) -> int:
        """Total ms spent in ``stage``."""
        return sum(timing.duration_ms for timing in self.stages if timing.stage is stage)
