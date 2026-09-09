"""Task 35: cooperative macro debugger and durable sessions."""

from __future__ import annotations

from typing import Any

import pytest

from ayris.actions.macros.debug_session import DebugSessionStore, DebugState
from ayris.actions.macros.debugger import MacroDebugger
from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.schema import CommandModel
from ayris.actions.result import ActionResult
from ayris.core.database import Database
from ayris.core.events import DebugStepFinished, EventBus


class Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has(self, name: str) -> bool:
        return name == "Echo"

    def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        del request_id, command_id
        values = dict(params or {})
        self.calls.append((name, values))
        return ActionResult.done(value=values.get("value"))


def command(*actions: dict[str, Any], name: str = "Тест", command_id: int = 1) -> CommandModel:
    return CommandModel(id=command_id, name=name, actions=list(actions))


def echo(value: Any) -> dict[str, Any]:
    return {"type": "Echo", "params": {"value": value}}


def finish(debugger: MacroDebugger) -> None:
    try:
        if debugger.state is DebugState.PAUSED:
            debugger.continue_()
        debugger.wait(2)
    finally:
        debugger.stop()


def test_breakpoint_pauses_before_the_requested_block() -> None:
    registry = Registry()
    with MacroEngine(registry) as engine:
        debugger = MacroDebugger(engine)
        debugger.add_breakpoint("actions[1]")
        debugger.start(command(echo(1), echo(2)))
        assert debugger.wait_paused() == "actions[0]"
        debugger.continue_()
        assert debugger.wait_paused() == "actions[1]"
        finish(debugger)
    assert [params["value"] for _, params in registry.calls] == [1, 2]


def test_step_into_call_and_step_out_to_caller() -> None:
    nested = command(echo("inside"), name="Вложенная", command_id=2)
    top = command(
        {"type": "CallCommand", "params": {"command": "Вложенная"}},
        echo("after"),
    )
    registry = Registry()
    with MacroEngine(
        registry, library=lambda name: nested if name == nested.name else None
    ) as engine:
        debugger = MacroDebugger(engine)
        debugger.start(top)
        assert debugger.wait_paused() == "actions[0]"
        debugger.step_into()
        assert debugger.wait_paused() == "actions[0]"
        assert debugger.call_stack == ("Вложенная",)
        debugger.step_out()
        assert debugger.wait_paused() == "actions[1]"
        assert debugger.call_stack == ()
        finish(debugger)


def test_step_over_call_stays_in_the_caller() -> None:
    nested = command(echo("inside"), name="Вложенная", command_id=2)
    top = command(
        {"type": "CallCommand", "params": {"command": "Вложенная"}},
        echo("after"),
    )
    registry = Registry()
    with MacroEngine(
        registry, library=lambda name: nested if name == nested.name else None
    ) as engine:
        debugger = MacroDebugger(engine)
        debugger.start(top)
        assert debugger.wait_paused() == "actions[0]"
        debugger.step_over()
        assert debugger.wait_paused() == "actions[1]"
        assert debugger.call_stack == ()
        finish(debugger)


def test_conditional_breakpoint_and_variable_change_choose_if_branch() -> None:
    macro = command(
        {
            "type": "If",
            "params": {"condition": "flag"},
            "then": [echo("yes")],
            "else": [echo("no")],
        },
        echo("done"),
    )
    registry = Registry()
    with MacroEngine(registry) as engine:
        debugger = MacroDebugger(engine)
        debugger.add_breakpoint("actions[1]", "flag")
        debugger.start(macro)
        debugger.wait_paused()
        debugger.set_variable("flag", True)
        debugger.add_watch("flag == true")
        assert debugger.evaluate_watches()[0].value is True
        debugger.continue_()
        assert debugger.wait_paused() == "actions[1]"
        finish(debugger)
    assert registry.calls[0][1]["value"] == "yes"


def test_run_from_skips_preceding_blocks_and_records_filled_params() -> None:
    registry = Registry()
    with MacroEngine(registry) as engine:
        debugger = MacroDebugger(engine)
        debugger.run_from(
            command(echo("before"), echo("{value}")),
            "actions[1]",
            slots_override={"value": 42},
        )
        assert debugger.wait_paused() == "actions[1]"
        finish(debugger)
        report = debugger.report
    assert [params["value"] for _, params in registry.calls] == [42]
    assert report is not None
    assert report.steps[0].params == {"value": 42}
    assert report.steps[0].result == 42


def test_session_restores_breakpoints_watches_slots_and_report(tmp_path) -> None:
    path = tmp_path / "ayris.db"
    database = Database.open(path)
    profile_id = database.insert(
        "INSERT INTO profiles (name, created_at, is_active) VALUES (?, ?, 1)",
        ("Основной", "2026-09-08T00:00:00+00:00"),
    )
    command_id = database.insert(
        """
        INSERT INTO commands (profile_id, name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (profile_id, "Тест", "2026-09-08T00:00:00+00:00", "2026-09-08T00:00:00+00:00"),
    )
    macro = command(echo("{slot}"), command_id=command_id)
    registry = Registry()
    store = DebugSessionStore(database)
    with MacroEngine(registry) as engine:
        first = MacroDebugger(engine, store=store)
        first.add_breakpoint("actions[0]")
        first.add_watch("slot")
        first.start(macro, {"slot": "saved"})
        first.wait_paused()
        finish(first)

        second = MacroDebugger(engine, store=store)
        restored = second.open(macro)
        assert restored is not None
        assert restored.report is not None
        assert second.report is not None
        assert second.slots_override == {"slot": "saved"}
        second.start(macro)
        second.wait_paused()
        assert second.breakpoints[0].path == "actions[0]"
        assert second.watch_expressions == ("slot",)
        assert next(item for item in second.get_variables() if item.name == "slot").value == "saved"
        finish(second)
        assert second.report is not None
    database.close()


def test_run_block_executes_only_one_block() -> None:
    registry = Registry()
    with MacroEngine(registry) as engine:
        debugger = MacroDebugger(engine)
        debugger.run_block(command(echo(1), echo(2), echo(3)), "actions[1]")
        debugger.wait_paused()
        finish(debugger)
    assert [params["value"] for _, params in registry.calls] == [2]
    assert debugger.warnings == ("Предыдущие блоки не выполнялись; запуск начат с actions[1].",)


def test_dry_run_records_blocks_without_calling_actions() -> None:
    registry = Registry()
    with MacroEngine(registry) as engine:
        debugger = MacroDebugger(engine, dry_run=True)
        debugger.start(command(echo(1)))
        debugger.wait_paused()
        finish(debugger)
        report = debugger.report
    assert registry.calls == []
    assert report is not None
    assert report.steps[0].params == {"value": 1}
    assert report.steps[0].message == "Сухой прогон: действие не выполнено."


def test_debug_step_events_include_nested_and_skipped_blocks() -> None:
    nested = command(echo("inside"), name="Вложенная", command_id=2)
    macro = command(
        {"type": "CallCommand", "params": {"command": "Вложенная"}},
        {"type": "Echo", "enabled": False, "params": {"value": "off"}},
    )
    events: list[DebugStepFinished] = []
    bus = EventBus()
    bus.subscribe(DebugStepFinished, events.append)
    with MacroEngine(
        Registry(), library=lambda name: nested if name == nested.name else None
    ) as engine:
        debugger = MacroDebugger(engine, bus=bus)
        debugger.start(macro)
        debugger.wait_paused()
        finish(debugger)
    bus.drain()
    assert [(event.block, event.status) for event in events] == [
        ("Echo", "ok"),
        ("CallCommand", "ok"),
        ("Echo", "skipped"),
    ]


def test_variable_access_requires_a_pause() -> None:
    with MacroEngine(Registry()) as engine:
        debugger = MacroDebugger(engine)
        with pytest.raises(RuntimeError, match="only while paused"):
            debugger.get_variables()


def test_false_conditional_breakpoint_is_ignored() -> None:
    with MacroEngine(Registry()) as engine:
        debugger = MacroDebugger(engine)
        debugger.add_breakpoint("actions[1]", "flag")
        debugger.start(command(echo(1), echo(2)))
        debugger.wait_paused()
        debugger.set_variable("flag", False)
        debugger.continue_()
        report = debugger.wait(2)
    assert report.ok


def test_stop_releases_a_paused_worker() -> None:
    with MacroEngine(Registry()) as engine:
        debugger = MacroDebugger(engine)
        debugger.start(command(echo(1)))
        debugger.wait_paused()
        debugger.stop()
        report = debugger.wait(2)
    assert report.cancelled
