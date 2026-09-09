"""State and SQLite persistence for one macro-debugger session."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ayris.actions.macros.errors import MacroRuntimeError
from ayris.actions.macros.report import ExecutionReport, RunOutcome, StepRecord, StepStatus
from ayris.core.models import VariableScope, VariableType, utc_now

if TYPE_CHECKING:
    from ayris.core.database import Database

__all__ = [
    "Breakpoint",
    "DebugSessionSnapshot",
    "DebugSessionStore",
    "DebugState",
    "VariableView",
    "WatchValue",
]


class DebugState(StrEnum):
    """Lifecycle visible to the future editor UI."""

    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class Breakpoint:
    """A block path and an optional expression that must be true."""

    path: str
    condition: str = ""
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class VariableView:
    """One name exactly as the watch panel displays it."""

    name: str
    value: Any
    type: VariableType | None
    scope: VariableScope | str


@dataclass(frozen=True, slots=True)
class WatchValue:
    """A watch expression and either its value or its evaluation error."""

    expression: str
    value: Any = None
    error: str = ""


@dataclass(slots=True)
class DebugSessionSnapshot:
    """The durable part of a session; live thread state is deliberately absent."""

    command_id: int
    breakpoints: list[Breakpoint] = field(default_factory=list)
    watches: list[str] = field(default_factory=list)
    slots_override: dict[str, Any] = field(default_factory=dict)
    report: ExecutionReport | None = None


class DebugSessionStore:
    """Persist the last debugger state in the application database."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def load(self, command_id: int) -> DebugSessionSnapshot | None:
        row = self._database.query_one(
            "SELECT breakpoints_json, watches_json, slots_json, report_json "
            "FROM macro_debug_sessions WHERE command_id = ?",
            (command_id,),
        )
        if row is None:
            return None
        breakpoints = [Breakpoint(**item) for item in json.loads(str(row[0]))]
        watches = [str(item) for item in json.loads(str(row[1]))]
        slots = dict(json.loads(str(row[2])))
        raw_report = json.loads(str(row[3])) if row[3] else None
        return DebugSessionSnapshot(
            command_id=command_id,
            breakpoints=breakpoints,
            watches=watches,
            slots_override=slots,
            report=_report_from_json(raw_report) if raw_report else None,
        )

    def save(self, snapshot: DebugSessionSnapshot) -> None:
        self._database.execute(
            """
            INSERT INTO macro_debug_sessions (
                command_id, breakpoints_json, watches_json, slots_json, report_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (command_id) DO UPDATE SET
                breakpoints_json = excluded.breakpoints_json,
                watches_json = excluded.watches_json,
                slots_json = excluded.slots_json,
                report_json = excluded.report_json,
                updated_at = excluded.updated_at
            """,
            (
                snapshot.command_id,
                _json([asdict(item) for item in snapshot.breakpoints]),
                _json(snapshot.watches),
                _json(snapshot.slots_override),
                _json(_report_to_json(snapshot.report)) if snapshot.report else "",
                utc_now().isoformat(),
            ),
        )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _report_to_json(report: ExecutionReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "command": report.command,
        "command_id": report.command_id,
        "outcome": str(report.outcome),
        "duration_ms": report.duration_ms,
        "steps": [
            {
                **asdict(step),
                "status": str(step.status),
            }
            for step in report.steps
        ],
        "value": report.value,
        "error": str(report.error) if report.error else "",
        "message_ru": report.message_ru,
        "trigger": report.trigger,
        "request_id": report.request_id,
        "started_at": report.started_at.isoformat(),
    }


def _report_from_json(raw: dict[str, Any]) -> ExecutionReport:
    from datetime import datetime

    error_text = str(raw.get("error", ""))
    return ExecutionReport(
        run_id=str(raw["run_id"]),
        command=str(raw.get("command", "")),
        command_id=raw.get("command_id"),
        outcome=RunOutcome(str(raw.get("outcome", RunOutcome.SUCCESS))),
        duration_ms=int(raw.get("duration_ms", 0)),
        steps=tuple(
            StepRecord(
                path=str(item["path"]),
                block=str(item["block"]),
                status=StepStatus(str(item.get("status", StepStatus.OK))),
                duration_ms=int(item.get("duration_ms", 0)),
                offset_ms=int(item.get("offset_ms", 0)),
                depth=int(item.get("depth", 0)),
                message=str(item.get("message", "")),
                error=str(item.get("error", "")),
                params=dict(item.get("params", {})),
                result=item.get("result"),
            )
            for item in raw.get("steps", [])
        ),
        value=raw.get("value"),
        error=MacroRuntimeError(error_text) if error_text else None,
        message_ru=str(raw.get("message_ru", "")),
        trigger=str(raw.get("trigger", "")),
        request_id=str(raw.get("request_id", "")),
        started_at=datetime.fromisoformat(str(raw["started_at"])),
    )
