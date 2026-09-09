"""Cooperative backend debugger for the macro interpreter."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Mapping
from dataclasses import replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from ayris.actions.macros.context import ExecutionContext, RunInfo, TriggerSource
from ayris.actions.macros.debug_session import (
    Breakpoint,
    DebugSessionSnapshot,
    DebugSessionStore,
    DebugState,
    VariableView,
    WatchValue,
)
from ayris.actions.macros.engine import MacroEngine, MacroRun
from ayris.actions.macros.errors import MacroCancelledError
from ayris.actions.macros.report import ExecutionReport, StepRecord
from ayris.actions.macros.schema import ActionBlock, CommandModel, walk_blocks
from ayris.actions.result import ActionResult
from ayris.core.events import DebugFinished, DebugPaused, DebugStepFinished, Event
from ayris.core.models import VariableScope

if TYPE_CHECKING:
    from concurrent.futures import Future

    from ayris.actions.macros.engine import _Runner
    from ayris.core.events import EventBus

__all__ = ["Breakpoint", "DebugSessionStore", "DebugState", "MacroDebugger"]


class DebugController(Protocol):
    """Hooks the engine calls at block and CallCommand boundaries."""

    def before_block(self, runner: _Runner, block: ActionBlock, path: str, depth: int) -> None: ...

    def should_skip(self, path: str) -> bool: ...

    def step_finished(self, record: StepRecord) -> None: ...

    def execute_action(
        self, runner: _Runner, block: ActionBlock, path: str, params: Mapping[str, Any]
    ) -> ActionResult[Any]: ...

    def finished(self, report: ExecutionReport) -> ExecutionReport: ...

    def enter_call(self, command: str, context: ExecutionContext) -> None: ...

    def leave_call(self) -> None: ...


class _Mode(StrEnum):
    CONTINUE = "continue"
    OVER = "over"
    INTO = "into"
    OUT = "out"


class MacroDebugger:
    """Drive one interpreter run while its worker cooperatively pauses."""

    def __init__(
        self,
        engine: MacroEngine,
        *,
        bus: EventBus | None = None,
        store: DebugSessionStore | None = None,
        dry_run: bool = False,
    ) -> None:
        self._engine = engine
        self._bus = bus
        self._store = store
        self._dry_run = dry_run
        self._condition = threading.Condition()
        self._state = DebugState.FINISHED
        self._mode = _Mode.CONTINUE
        self._resume_depth = 0
        self._force_pause = False
        self._stop = False
        self._command: CommandModel | None = None
        self._run: MacroRun | None = None
        self._future: Future[ExecutionReport] | None = None
        self._context: ExecutionContext | None = None
        self._current_context: ExecutionContext | None = None
        self._path = ""
        self._depth = 0
        self._call_stack: list[str] = []
        self._breakpoints: dict[str, Breakpoint] = {}
        self._watches: list[str] = []
        self._slots: dict[str, Any] = {}
        self._report: ExecutionReport | None = None
        self._steps: list[StepRecord] = []
        self._start_path: str | None = None
        self._only_path: str | None = None
        self._restored_id: int | None = None
        self._warnings: tuple[str, ...] = ()

    @property
    def state(self) -> DebugState:
        with self._condition:
            return self._state

    @property
    def current_path(self) -> str:
        with self._condition:
            return self._path

    @property
    def report(self) -> ExecutionReport | None:
        with self._condition:
            return self._report

    @property
    def call_stack(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._call_stack)

    @property
    def breakpoints(self) -> tuple[Breakpoint, ...]:
        with self._condition:
            return tuple(self._breakpoints.values())

    @property
    def watch_expressions(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(self._watches)

    @property
    def slots_override(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._slots)

    @property
    def warnings(self) -> tuple[str, ...]:
        with self._condition:
            return self._warnings

    def open(self, command: CommandModel) -> DebugSessionSnapshot | None:
        """Restore the last session for an editor without starting execution."""
        with self._condition:
            if self._state is not DebugState.FINISHED:
                raise RuntimeError("debug session is already active")
            self._command = command
            self._restore(command)
            if self._store is None or command.id is None:
                return None
            return self._store.load(command.id)

    def start(
        self, command: CommandModel, slots_override: Mapping[str, Any] | None = None
    ) -> MacroRun:
        """Start asynchronously and pause immediately before the first block."""
        return self._start(command, slots_override=slots_override)

    def _start(
        self,
        command: CommandModel,
        *,
        slots_override: Mapping[str, Any] | None = None,
        variables: Mapping[str, Any] | None = None,
        start_path: str | None = None,
        only_path: str | None = None,
    ) -> MacroRun:
        with self._condition:
            if self._state is not DebugState.FINISHED:
                raise RuntimeError("debug session is already active")
            self._restore(command)
            if slots_override is not None:
                self._slots = dict(slots_override)
            self._command = command
            self._report = None
            self._steps = []
            self._stop = False
            self._force_pause = True
            self._mode = _Mode.CONTINUE
            self._start_path = start_path
            self._only_path = only_path
            self._warnings = (
                (f"Предыдущие блоки не выполнялись; запуск начат с {start_path}.",)
                if start_path is not None
                else ()
            )
            run = MacroRun(
                run_id=uuid.uuid4().hex[:12],
                command=command.name,
                command_id=command.id,
                trigger=TriggerSource.MANUAL,
                admitted=True,
            )
            context = ExecutionContext(
                info=RunInfo(run_id=run.run_id, command=command.name, command_id=command.id),
                store=self._engine.variables,
                slots=self._slots,
                variables=command.variables,
            )
            run.context = context
            for name, value in (variables or {}).items():
                context.set(name, value)
            self._run = run
            self._context = context
            self._current_context = context
            self._state = DebugState.RUNNING
            self._future = self._engine.submit_debug(run, command, context, self)
            run.future = self._future
            self._save()
            return run

    def continue_(self) -> None:
        self._resume(_Mode.CONTINUE)

    def step_over(self) -> None:
        self._resume(_Mode.OVER)

    def step_into(self) -> None:
        self._resume(_Mode.INTO)

    def step_out(self) -> None:
        self._resume(_Mode.OUT)

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            if self._run is not None:
                self._run.stop("остановлено отладчиком")
            self._condition.notify_all()

    def wait(self, timeout: float | None = None) -> ExecutionReport:
        future = self._future
        if future is None:
            raise RuntimeError("debug session has not started")
        return future.result(timeout)

    def wait_paused(self, timeout: float = 2.0) -> str:
        """Wait for a pause; primarily useful to non-Qt clients and tests."""
        import time

        limit = time.monotonic() + timeout
        with self._condition:
            while self._state is not DebugState.PAUSED:
                left = limit - time.monotonic()
                if left <= 0:
                    raise TimeoutError("debugger did not pause")
                self._condition.wait(left)
            return self._path

    def add_breakpoint(self, path: str, condition: str = "", *, enabled: bool = True) -> Breakpoint:
        point = Breakpoint(path, condition, enabled)
        with self._condition:
            self._breakpoints[path] = point
            self._save()
        return point

    def remove_breakpoint(self, path: str) -> bool:
        with self._condition:
            removed = self._breakpoints.pop(path, None) is not None
            self._save()
            return removed

    def toggle_breakpoint(self, path: str) -> Breakpoint:
        with self._condition:
            point = self._breakpoints.get(path)
            point = Breakpoint(path) if point is None else replace(point, enabled=not point.enabled)
            self._breakpoints[path] = point
            self._save()
            return point

    def add_watch(self, expression: str) -> None:
        with self._condition:
            if expression not in self._watches:
                self._watches.append(expression)
                self._save()

    def remove_watch(self, expression: str) -> bool:
        with self._condition:
            if expression not in self._watches:
                return False
            self._watches.remove(expression)
            self._save()
            return True

    def evaluate_watches(self) -> tuple[WatchValue, ...]:
        context = self._paused_context()
        values: list[WatchValue] = []
        for expression in self._watches:
            try:
                values.append(WatchValue(expression, context.evaluate(expression)))
            except Exception as exc:
                values.append(WatchValue(expression, error=str(exc)))
        return tuple(values)

    def get_variables(self) -> tuple[VariableView, ...]:
        context = self._paused_context()
        views: list[VariableView] = []
        for name, value in context.locals.items():
            views.append(VariableView(name, value, context.type_of(name), VariableScope.LOCAL))
        for name, value in context.slots.items():
            if name not in context.locals:
                views.append(VariableView(name, value, None, "slot"))
        hidden = set(context.locals) | set(context.slots)
        for scope in (VariableScope.PROFILE, VariableScope.GLOBAL):
            for name in context.store.names(scope):
                if name not in hidden:
                    views.append(
                        VariableView(
                            name, context.store.read(scope, name), context.type_of(name), scope
                        )
                    )
        return tuple(sorted(views, key=lambda item: (str(item.scope), item.name)))

    def set_variable(self, name: str, value: Any, scope: VariableScope | None = None) -> Any:
        context = self._paused_context()
        if name in context.slots and scope is None:
            context.slots[name] = value
            return value
        stored = context.set(name, value, scope)
        if name == "last_result":
            context.set_result(stored)
        return stored

    def run_from(
        self,
        command: CommandModel,
        block_path: str,
        *,
        variables: Mapping[str, Any] | None = None,
        slots_override: Mapping[str, Any] | None = None,
    ) -> MacroRun:
        self._validate_path(command, block_path)
        return self._start(
            command,
            slots_override=slots_override,
            variables=variables,
            start_path=block_path,
        )

    def run_block(
        self,
        command: CommandModel,
        block_path: str,
        *,
        variables: Mapping[str, Any] | None = None,
        slots_override: Mapping[str, Any] | None = None,
    ) -> MacroRun:
        self._validate_path(command, block_path)
        return self._start(
            command,
            slots_override=slots_override,
            variables=variables,
            start_path=block_path,
            only_path=block_path,
        )

    def should_skip(self, path: str) -> bool:
        with self._condition:
            if self._only_path is None and self._start_path is None:
                return False
            if self._only_path is not None and self._start_path is None:
                return True
            target = self._start_path
            if target is None:
                return False
            return path != target and not target.startswith(f"{path}.")

    def step_finished(self, record: StepRecord) -> None:
        with self._condition:
            self._steps.append(record)
            self._publish_step(self._command.id if self._command else None, record)

    def execute_action(
        self, runner: _Runner, block: ActionBlock, path: str, params: Mapping[str, Any]
    ) -> ActionResult[Any]:
        """Run through the registry, or describe the skipped effect in dry-run mode."""
        if self._dry_run:
            return ActionResult.done(
                "Сухой прогон: действие не выполнено.",
                detail=f"dry-run skipped {block.type} at {path}",
                data={"dry_run": True},
            )
        return self._engine.registry.execute(
            block.type,
            params,
            request_id=runner.run.request_id,
            command_id=runner.run.command_id,
        )

    def before_block(self, runner: _Runner, block: ActionBlock, path: str, depth: int) -> None:
        del block
        with self._condition:
            self._current_context = runner.context
            if self._start_path is not None and path != self._start_path:
                return
            if self._start_path is not None:
                self._start_path = None
            current_depth = len(self._call_stack)
            point = self._breakpoints.get(path)
            hit = bool(
                point
                and point.enabled
                and (not point.condition or runner.context.truth(point.condition))
            )
            stepping = (
                self._force_pause
                or (self._mode is _Mode.OVER and current_depth <= self._resume_depth)
                or (self._mode is _Mode.INTO)
                or (self._mode is _Mode.OUT and current_depth < self._resume_depth)
            )
            if not hit and not stepping:
                return
            self._pause_locked(path, depth)

    def enter_call(self, command: str, context: ExecutionContext) -> None:
        with self._condition:
            self._call_stack.append(command)
            self._current_context = context

    def leave_call(self) -> None:
        with self._condition:
            if self._call_stack:
                self._call_stack.pop()
            self._current_context = self._context

    def _pause_locked(self, path: str, depth: int) -> None:
        self._path = path
        self._depth = depth
        self._force_pause = False
        self._state = DebugState.PAUSED
        self._publish(
            DebugPaused(
                command_id=self._command.id if self._command else None,
                block_path=path,
                call_stack=tuple(self._call_stack),
            )
        )
        self._condition.notify_all()
        while self._state is DebugState.PAUSED and not self._stop:
            self._condition.wait()
        if self._stop and self._run is not None:
            self._run.cancel.set()
            raise MacroCancelledError(self._run.reason)

    def _resume(self, mode: _Mode) -> None:
        with self._condition:
            if self._state is not DebugState.PAUSED:
                raise RuntimeError("debugger is not paused")
            self._mode = mode
            self._resume_depth = len(self._call_stack)
            self._state = DebugState.RUNNING
            self._condition.notify_all()

    def finished(self, report: ExecutionReport) -> ExecutionReport:
        """Freeze streamed nested steps and persist the completed session."""
        with self._condition:
            report = replace(report, steps=tuple(self._steps))
            command = self._command
            run = self._run
            if command is None or run is None:  # pragma: no cover - start owns the worker
                raise RuntimeError("debug session disappeared while running")
            self._report = report
            run.report = report
            self._state = DebugState.FINISHED
            self._publish(
                DebugFinished(
                    command_id=command.id,
                    outcome=str(report.outcome),
                    duration_ms=report.duration_ms,
                    steps=len(report.steps),
                )
            )
            self._save()
            self._condition.notify_all()
        return report

    def _publish_step(self, command_id: int | None, step: StepRecord) -> None:
        self._publish(
            DebugStepFinished(
                command_id=command_id,
                block_path=step.path,
                block=step.block,
                duration_ms=step.duration_ms,
                status=str(step.status),
            )
        )

    def _publish(self, event: Event) -> None:
        if self._bus is not None:
            self._bus.publish(event)

    def _paused_context(self) -> ExecutionContext:
        with self._condition:
            if self._state is not DebugState.PAUSED or self._current_context is None:
                raise RuntimeError("variables are available only while paused")
            return self._current_context

    def _restore(self, command: CommandModel) -> None:
        if command.id is None or command.id == self._restored_id or self._store is None:
            return
        saved = self._store.load(command.id)
        self._restored_id = command.id
        if saved is None:
            return
        self._breakpoints = {point.path: point for point in saved.breakpoints}
        self._watches = list(saved.watches)
        self._slots = dict(saved.slots_override)
        self._report = saved.report

    def _save(self) -> None:
        if self._store is None or self._command is None or self._command.id is None:
            return
        self._store.save(
            DebugSessionSnapshot(
                command_id=self._command.id,
                breakpoints=list(self._breakpoints.values()),
                watches=list(self._watches),
                slots_override=dict(self._slots),
                report=self._report,
            )
        )

    @staticmethod
    def _validate_path(command: CommandModel, path: str) -> None:
        if not any(location.path_text == path for location in walk_blocks(command.actions)):
            raise ValueError(f"unknown block path: {path}")
