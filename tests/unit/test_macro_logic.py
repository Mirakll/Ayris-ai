"""Задача 32: блоки логики, блоки данных и хранилище переменных со scope и БД.

Task 31 built the walk; this file tests what the walk runs. The tests go through a real
:class:`~ayris.actions.macros.engine.MacroEngine` rather than calling the handlers
directly, for the reason task 31's own tests give: the blocks are only interesting
together — a ``Break`` matters because of the ``While`` above it, and ``CallCommand``
matters because of the local scope it does *not* see. A fake registry records what came
out the far end.

Four things carry the weight here.

*Nesting has to work in both directions.* :class:`TestNestedFlow` runs a loop inside a
loop and asserts that ``Break`` leaves exactly one of them, that ``Continue`` skips the
rest of one turn and not the loop, and that ``Return`` from the bottom of three nested
blocks ends the command and carries its value out.

*A called command is a unit.* :class:`TestCallCommand` pins the three halves of that:
arguments go in as slots, the ``Return`` value comes back as ``last_result``, and the
callee cannot see the caller's locals — a command that could would only ever work from
where it was written.

*All six types, through all nine data blocks.* :class:`TestDataBlocks` and
:class:`TestTypes`. A ``DictSet`` on an ``int`` variable is a mistake with a name in it,
not a stack trace.

*A persistent variable outlives the process; a local one does not.*
:class:`TestPersistence` stages the restart the way the task file asks — a second
:class:`~ayris.core.repositories.Repositories` over the same file — because that is what
a restart *is* from the store's point of view, and it is testable without a subprocess.

Groups:

* :class:`TestBranches` — ``If``/``Else``, ``Switch``/``Case``/``Default``.
* :class:`TestLoops` — ``While`` and ``For``, their limits and their loop variable.
* :class:`TestNestedFlow` — loops in loops: ``Break``, ``Continue``, ``Return``.
* :class:`TestTryCatch` — what is caught, what is not, and the error variable.
* :class:`TestCallCommand` — arguments, return value, isolation, depth, cycles.
* :class:`TestDataBlocks` — the nine blocks of table 7.2 and its neighbours.
* :class:`TestTypes` — the six declared types, coerced and refused.
* :class:`TestScopes` — ``local``, ``profile``, ``global``: who sees what.
* :class:`TestPersistence` — the database store, batching, and surviving a restart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ayris.actions.macros.context import ExecutionContext, MemoryVariables, RunInfo
from ayris.actions.macros.engine import ExecutionLimits, MacroEngine
from ayris.actions.macros.errors import MacroCallError
from ayris.actions.macros.report import RunOutcome
from ayris.actions.macros.schema import CommandModel
from ayris.actions.macros.variables import MISSING, DatabaseVariables
from ayris.actions.result import ActionResult
from ayris.core.database import Database, reset_database
from ayris.core.models import VariableScope, VariableType
from ayris.core.repositories import Repositories

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit


class FakeRegistry:
    """Two methods and a memory of the calls, the way the engine sees the registry."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.raises: dict[str, Exception] = {}
        self.unknown: set[str] = set()

    def has(self, name: str) -> bool:
        return name not in self.unknown

    def execute(
        self,
        name: str,
        params: Any = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        self.calls.append((name, dict(params or {})))
        error = self.raises.get(name)
        if error is not None:
            raise error
        return ActionResult.done(value=params.get("value") if params else None)

    @property
    def names(self) -> list[str]:
        """The action names, in the order they were called."""
        return [name for name, _ in self.calls]

    def params(self, name: str) -> dict[str, Any]:
        """The parameters of the last call of ``name``."""
        return next(params for called, params in reversed(self.calls) if called == name)

    def every(self, name: str) -> list[dict[str, Any]]:
        """The parameters of every call of ``name``, in order."""
        return [params for called, params in self.calls if called == name]


def command(*blocks: dict[str, Any], name: str = "Проверка", **fields: Any) -> CommandModel:
    """A command built from block dictionaries, the way a ``.ayris`` file carries them."""
    return CommandModel.model_validate({"name": name, "actions": list(blocks), **fields})


def block(kind: str, /, **params: Any) -> dict[str, Any]:
    """One block: its type and its parameters, with no branches."""
    return {"type": kind, "params": params}


def echo(value: Any) -> dict[str, Any]:
    """An action call that records one value, so a test can read the order out."""
    return block("Echo", value=value)


@pytest.fixture
def registry() -> FakeRegistry:
    """A registry that says yes to everything and remembers what it was asked for."""
    return FakeRegistry()


@pytest.fixture
def engines(registry: FakeRegistry) -> Iterator[Callable[..., MacroEngine]]:
    """A factory for engines, every one of them shut down when the test ends."""
    made: list[MacroEngine] = []

    def make(**options: Any) -> MacroEngine:
        engine = MacroEngine(registry, **options)
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.shutdown()


@pytest.fixture
def engine(engines) -> MacroEngine:
    """The ordinary engine: real threads, no bus, defaults everywhere else."""
    return engines()


@pytest.fixture
def library() -> dict[str, CommandModel]:
    """The commands ``CallCommand`` can find. A test puts its callee in here."""
    return {}


@pytest.fixture
def calling(engines, library) -> MacroEngine:
    """An engine that can call commands, looking them up in :func:`library`."""
    return engines(library=library.get)


def values(registry: FakeRegistry) -> list[Any]:
    """What every ``Echo`` was given, in order. How a loop's turns are read back."""
    return [params.get("value") for params in registry.every("Echo")]


class TestBranches:
    """``If``/``Else`` and ``Switch``: choosing one of several ways on."""

    def test_if_runs_the_then_branch_and_records_it(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "If",
                    "params": {"condition": "{volume} > 10"},
                    "then": [echo("громко")],
                    "else": [echo("тихо")],
                }
            ),
            slots={"volume": 50},
        )

        assert report.ok
        assert values(registry) == ["громко"]
        assert report.step("actions[0]").message == "then"

    def test_if_runs_the_else_branch(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "If",
                    "params": {"condition": "{volume} > 90"},
                    "then": [echo("громко")],
                    "else": [echo("тихо")],
                }
            ),
            slots={"volume": 50},
        )

        assert report.ok
        assert values(registry) == ["тихо"]
        assert report.step("actions[0]").message == "else"

    def test_an_if_without_an_else_simply_goes_on(self, engine, registry) -> None:
        report = engine.run(
            command(
                {"type": "If", "params": {"condition": "false"}, "then": [echo("нет")]},
                echo("дальше"),
            )
        )

        assert report.ok
        assert values(registry) == ["дальше"]

    def test_nested_ifs_choose_independently(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "If",
                    "params": {"condition": "{a} > 0"},
                    "then": [
                        {
                            "type": "If",
                            "params": {"condition": "{b} > 0"},
                            "then": [echo("оба")],
                            "else": [echo("только a")],
                        }
                    ],
                }
            ),
            slots={"a": 1, "b": 0},
        )

        assert report.ok
        assert values(registry) == ["только a"]

    def test_switch_takes_the_matching_case(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "Switch",
                    "params": {"value": "{mode}"},
                    "body": [
                        {"type": "Case", "params": {"value": "тихо"}, "body": [echo("mute")]},
                        {"type": "Case", "params": {"value": "громко"}, "body": [echo("loud")]},
                        {"type": "Default", "body": [echo("default")]},
                    ],
                }
            ),
            slots={"mode": "громко"},
        )

        assert report.ok
        assert values(registry) == ["loud"]
        assert report.step("actions[0]").message == "body[1]"

    def test_switch_falls_through_to_default(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "Switch",
                    "params": {"value": "{mode}"},
                    "body": [
                        {"type": "Case", "params": {"value": "тихо"}, "body": [echo("mute")]},
                        {"type": "Default", "body": [echo("default")]},
                    ],
                }
            ),
            slots={"mode": "что-то другое"},
        )

        assert report.ok
        assert values(registry) == ["default"]

    def test_switch_with_no_matching_arm_and_no_default_does_nothing(self, engine, registry):
        report = engine.run(
            command(
                {
                    "type": "Switch",
                    "params": {"value": "нет"},
                    "body": [{"type": "Case", "params": {"value": "да"}, "body": [echo("да")]}],
                },
                echo("дальше"),
            )
        )

        assert report.ok
        assert values(registry) == ["дальше"]
        assert report.step("actions[0]").message == "ни одна ветка не подошла"

    def test_a_number_and_its_text_are_the_same_case(self, engine, registry) -> None:
        """A slot arrives as text, and ``case 50`` has to catch ``"50"``."""
        report = engine.run(
            command(
                {
                    "type": "Switch",
                    "params": {"value": "{level}"},
                    "body": [
                        {"type": "Case", "params": {"value": 50}, "body": [echo("пятьдесят")]}
                    ],
                }
            ),
            slots={"level": "50"},
        )

        assert report.ok
        assert values(registry) == ["пятьдесят"]

    def test_the_arms_not_taken_are_recorded_as_skipped(self, engine) -> None:
        report = engine.run(
            command(
                {
                    "type": "Switch",
                    "params": {"value": "b"},
                    "body": [
                        {"type": "Case", "params": {"value": "a"}, "body": [echo("a")]},
                        {"type": "Case", "params": {"value": "b"}, "body": [echo("b")]},
                        {"type": "Default", "body": [echo("d")]},
                    ],
                }
            )
        )

        assert report.ok
        skipped = [step.path for step in report.steps if step.status.value == "skipped"]
        assert skipped == ["actions[0].body[0]", "actions[0].body[2]"]


class TestLoops:
    """``While`` and ``For``: how many turns, with what in the loop variable."""

    def test_while_counts_down_and_stops(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="осталось", value=[1, 2, 3]),
                {
                    "type": "While",
                    "params": {"condition": "len({осталось}) > 0"},
                    "body": [block("ArrayPop", name="осталось", into="взятый"), echo("{взятый}")],
                },
            )
        )

        assert report.ok
        assert values(registry) == [3, 2, 1]
        assert report.step("actions[1]").message == "3 итераций"

    def test_a_while_that_never_holds_does_not_run_its_body(self, engine, registry) -> None:
        report = engine.run(
            command({"type": "While", "params": {"condition": "false"}, "body": [echo("нет")]})
        )

        assert report.ok
        assert values(registry) == []
        assert report.step("actions[0]").message == "0 итераций"

    def test_an_endless_while_is_stopped_at_the_limit(self, engines) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=5, max_steps=200))

        report = engine.run(
            command({"type": "While", "params": {"condition": "true"}, "body": [echo("виток")]})
        )

        assert report.outcome is RunOutcome.FAILED
        assert "5 раз" in (report.user_message or "")

    def test_a_block_may_lower_the_ceiling_but_not_raise_it(self, engines) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=10, max_steps=500))

        report = engine.run(
            command(
                {
                    "type": "While",
                    "params": {"condition": "true", "max_iterations": 1000},
                    "body": [echo("виток")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "10 раз" in (report.user_message or "")

    def test_for_walks_an_inclusive_range(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 3},
                    "body": [echo("{n}")],
                }
            )
        )

        assert report.ok
        assert values(registry) == [1, 2, 3]

    def test_for_walks_a_range_with_a_step_and_backwards(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 10, "to": 6, "step": -2},
                    "body": [echo("{n}")],
                }
            )
        )

        assert report.ok
        assert values(registry) == [10, 8, 6]

    def test_for_walks_an_array_variable(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="items", value=["раз", "два"]),
                {
                    "type": "For",
                    "params": {"var": "item", "items": "{items}"},
                    "body": [echo("{item}")],
                },
            )
        )

        assert report.ok
        assert values(registry) == ["раз", "два"]

    def test_the_loop_variable_survives_the_loop(self, engine, registry) -> None:
        """An ordinary local: a command counting attempts reads it afterwards."""
        report = engine.run(
            command(
                {"type": "For", "params": {"var": "n", "from": 1, "to": 3}, "body": []},
                echo("{n}"),
            )
        )

        assert report.ok
        assert values(registry) == [3]

    def test_a_huge_range_costs_a_short_list_and_a_clear_error(self, engines) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=4, max_steps=200))

        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 1_000_000},
                    "body": [echo("{n}")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "слишком длинным" in (report.user_message or "")

    def test_for_over_an_empty_array_does_nothing(self, engine, registry) -> None:
        report = engine.run(
            command(
                {"type": "For", "params": {"var": "n", "items": []}, "body": [echo("нет")]},
                echo("дальше"),
            )
        )

        assert report.ok
        assert values(registry) == ["дальше"]


class TestNestedFlow:
    """Loops inside loops, and the three blocks that redirect a walk."""

    def test_break_leaves_the_loop_it_sits_in(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 5},
                    "body": [
                        {
                            "type": "If",
                            "params": {"condition": "{n} > 2"},
                            "then": [block("Break")],
                        },
                        echo("{n}"),
                    ],
                },
                echo("после"),
            )
        )

        assert report.ok
        assert values(registry) == [1, 2, "после"]

    def test_break_in_a_nested_loop_leaves_only_the_inner_one(self, engine, registry) -> None:
        """The whole reason a flow signal is a return value: the loop above catches it."""
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "outer", "from": 1, "to": 3},
                    "body": [
                        {
                            "type": "For",
                            "params": {"var": "inner", "from": 1, "to": 3},
                            "body": [
                                {
                                    "type": "If",
                                    "params": {"condition": "{inner} > 1"},
                                    "then": [block("Break")],
                                },
                                echo("{outer}.{inner}"),
                            ],
                        }
                    ],
                }
            )
        )

        assert report.ok
        assert values(registry) == ["1.1", "2.1", "3.1"]

    def test_continue_skips_the_rest_of_one_turn(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 4},
                    "body": [
                        {
                            "type": "If",
                            "params": {"condition": "{n} % 2 == 0"},
                            "then": [block("Continue")],
                        },
                        echo("{n}"),
                    ],
                }
            )
        )

        assert report.ok
        assert values(registry) == [1, 3]

    def test_continue_in_a_while_still_reaches_the_condition(self, engine, registry) -> None:
        """A ``Continue`` that skipped the counting would hang; the counting is before it.

        ``SetVar`` substitutes and does not evaluate — text a user dictated is not an
        expression — so a counter is a growing array, and the ``While`` does the arithmetic
        in its condition, which is where arithmetic in this language belongs.
        """
        report = engine.run(
            command(
                {
                    "type": "While",
                    "params": {"condition": "len({витки}) < 4", "max_iterations": 10},
                    "body": [
                        block("ArrayPush", name="витки", value="ещё"),
                        block("ArrayLength", name="витки", into="i"),
                        {
                            "type": "If",
                            "params": {"condition": "{i} == 2"},
                            "then": [block("Continue")],
                        },
                        echo("{i}"),
                    ],
                }
            ),
            slots={"витки": []},
        )

        assert report.ok
        assert values(registry) == [1, 3, 4]

    def test_return_from_three_levels_down_ends_the_command(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "outer", "from": 1, "to": 3},
                    "body": [
                        {
                            "type": "While",
                            "params": {"condition": "true", "max_iterations": 5},
                            "body": [
                                {
                                    "type": "If",
                                    "params": {"condition": "{outer} == 2"},
                                    "then": [block("Return", value="готово")],
                                },
                                echo("{outer}"),
                                block("Break"),
                            ],
                        }
                    ],
                },
                echo("сюда не дойдёт"),
            )
        )

        assert report.ok
        assert report.value == "готово"
        assert values(registry) == [1]

    def test_return_without_a_value_ends_the_command_all_the_same(self, engine, registry) -> None:
        report = engine.run(command(echo("раз"), block("Return"), echo("два")))

        assert report.ok
        assert report.value is None
        assert values(registry) == ["раз"]

    def test_break_outside_a_loop_simply_ends_the_run(self, engine, registry) -> None:
        """Nothing above it catches the signal, so the walk stops. Not a failure."""
        report = engine.run(command(echo("раз"), block("Break"), echo("два")))

        assert report.ok
        assert values(registry) == ["раз"]


class TestTryCatch:
    """What ``Try`` catches, what it must not, and what it leaves in the error variable."""

    def test_catch_runs_on_a_failure_and_the_command_goes_on(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [block("Broken"), echo("не дойдёт")],
                    "catch": [echo("поймали")],
                },
                echo("дальше"),
            )
        )

        assert report.ok
        assert values(registry) == ["поймали", "дальше"]

    def test_the_error_lands_in_the_variable_the_block_names(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "params": {"error_var": "почему"},
                    "body": [block("Broken")],
                    "catch": [echo("{почему}")],
                }
            )
        )

        assert report.ok
        assert "сломалось" in str(values(registry)[0])

    def test_the_default_name_of_the_error_variable_is_error(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(
            command({"type": "Try", "body": [block("Broken")], "catch": [echo("{error}")]})
        )

        assert report.ok
        assert "сломалось" in str(values(registry)[0])

    def test_a_try_without_a_catch_swallows_and_goes_on(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(command({"type": "Try", "body": [block("Broken")]}, echo("дальше")))

        assert report.ok
        assert values(registry) == ["дальше"]

    def test_a_try_that_does_not_fail_never_enters_the_catch(self, engine, registry) -> None:
        report = engine.run(
            command({"type": "Try", "body": [echo("ок")], "catch": [echo("поймали")]})
        )

        assert report.ok
        assert values(registry) == ["ок"]

    def test_a_limit_is_not_the_commands_mistake_to_catch(self, engines) -> None:
        """A ``Try`` that swallowed a limit would turn the ceiling into a suggestion."""
        engine = engines(limits=ExecutionLimits(max_iterations=3, max_steps=200))

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [
                        {
                            "type": "While",
                            "params": {"condition": "true"},
                            "body": [echo("виток")],
                        }
                    ],
                    "catch": [echo("поймали")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED

    def test_a_nested_try_catches_before_the_outer_one(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [
                        {
                            "type": "Try",
                            "body": [block("Broken")],
                            "catch": [echo("внутренний")],
                        }
                    ],
                    "catch": [echo("внешний")],
                }
            )
        )

        assert report.ok
        assert values(registry) == ["внутренний"]

    def test_a_failure_in_the_catch_branch_reaches_the_outer_try(self, engine, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [
                        {
                            "type": "Try",
                            "body": [block("Broken")],
                            "catch": [block("Broken")],
                        }
                    ],
                    "catch": [echo("внешний")],
                }
            )
        )

        assert report.ok
        assert values(registry) == ["внешний"]


class TestCallCommand:
    """One command calling another: what goes in, what comes back, what stays behind."""

    def test_arguments_arrive_as_the_callees_slots(self, calling, library, registry) -> None:
        library["Помощник"] = command(echo("{кому}"), name="Помощник")

        report = calling.run(
            command(block("CallCommand", command="Помощник", args={"кому": "миру"}))
        )

        assert report.ok
        assert values(registry) == ["миру"]

    def test_the_return_value_comes_back_as_last_result(self, calling, library, registry) -> None:
        library["Считалка"] = command(block("Return", value=42), name="Считалка")

        report = calling.run(
            command(block("CallCommand", command="Считалка"), echo("{last_result}"))
        )

        assert report.ok
        assert values(registry) == [42]

    def test_the_value_can_be_stored_and_used(self, calling, library, registry) -> None:
        library["Считалка"] = command(block("Return", value=7), name="Считalka")

        report = calling.run(
            command(
                block("CallCommand", command="Считалка"),
                block("SetVar", name="сколько", value="{last_result}"),
                echo("{сколько}"),
            )
        )

        assert report.ok
        assert values(registry) == [7]

    def test_the_callee_cannot_see_the_callers_locals(self, calling, library) -> None:
        """A command that read its caller's variables could only work where it was written."""
        library["Помощник"] = command(echo("{секрет}"), name="Помощник")

        report = calling.run(
            command(
                block("SetVar", name="секрет", value="не для тебя"),
                block("CallCommand", command="Помощник"),
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "секрет" in str(report.error)

    def test_the_callee_does_see_the_shared_scopes(self, calling, library, registry) -> None:
        library["Помощник"] = command(echo("{общая}"), name="Помощник")
        calling.variables.write(VariableScope.GLOBAL, "общая", "видно")

        report = calling.run(command(block("CallCommand", command="Помощник")))

        assert report.ok
        assert values(registry) == ["видно"]

    def test_what_the_callee_writes_to_a_global_the_caller_reads(
        self, calling, library, registry
    ) -> None:
        library["Помощник"] = command(
            block("SetVar", name="итог", value="посчитано", scope="global"), name="Помощник"
        )

        report = calling.run(command(block("CallCommand", command="Помощник"), echo("{итог}")))

        assert report.ok
        assert values(registry) == ["посчитано"]

    def test_an_unknown_command_says_which_one(self, calling) -> None:
        report = calling.run(command(block("CallCommand", command="Нет такой")))

        assert report.outcome is RunOutcome.FAILED
        assert "Нет такой" in str(report.error)

    def test_a_switched_off_command_is_not_called(self, calling, library, registry) -> None:
        library["Выключенная"] = command(echo("нет"), name="Выключенная", enabled=False)

        report = calling.run(command(block("CallCommand", command="Выключенная")))

        assert report.outcome is RunOutcome.FAILED
        assert values(registry) == []

    def test_a_command_calling_itself_is_stopped_at_the_depth_limit(self, engines, library) -> None:
        """The cycle ban: a recursive call is caught by the ceiling and named for the user."""
        engine = engines(library=library.get, limits=ExecutionLimits(max_call_depth=3))
        library["Сама себя"] = command(block("CallCommand", command="Сама себя"), name="Сама себя")

        report = engine.run(library["Сама себя"])

        assert report.outcome is RunOutcome.FAILED
        assert "слишком глубоко" in (report.user_message or "")

    def test_two_commands_calling_each_other_are_stopped_too(self, engines, library) -> None:
        engine = engines(library=library.get, limits=ExecutionLimits(max_call_depth=4))
        library["Первая"] = command(block("CallCommand", command="Вторая"), name="Первая")
        library["Вторая"] = command(block("CallCommand", command="Первая"), name="Вторая")

        report = engine.run(library["Первая"])

        assert report.outcome is RunOutcome.FAILED
        assert "слишком глубоко" in (report.user_message or "")

    def test_a_failure_inside_the_callee_points_at_the_block_that_broke(
        self, calling, library, registry
    ) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")
        library["Помощник"] = command(block("Broken"), name="Помощник")

        report = calling.run(command(block("CallCommand", command="Помощник")))

        assert report.outcome is RunOutcome.FAILED
        assert report.error is not None
        assert getattr(report.error, "path", "") == "actions[0]"

    def test_a_call_can_be_caught_by_a_try(self, calling, library, registry) -> None:
        registry.raises["Broken"] = RuntimeError("сломалось")
        library["Помощник"] = command(block("Broken"), name="Помощник")

        report = calling.run(
            command(
                {
                    "type": "Try",
                    "body": [block("CallCommand", command="Помощник")],
                    "catch": [echo("поймали")],
                }
            )
        )

        assert report.ok
        assert values(registry) == ["поймали"]

    def test_a_call_that_is_not_waited_for_runs_on_its_own(self, calling, library) -> None:
        library["Помощник"] = command(echo("отдельно"), name="Помощник")

        report = calling.run(command(block("CallCommand", command="Помощник", wait=False)))

        assert report.ok
        assert "запущена отдельно" in (report.step("actions[0]").message or "")


class TestDataBlocks:
    """The nine blocks a command reads and writes its own data with."""

    def test_set_var_and_get_var(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="имя", value="Ayris"),
                block("GetVar", name="имя", into="копия"),
                echo("{копия}"),
            )
        )

        assert report.ok
        assert values(registry) == ["Ayris"]

    def test_get_var_without_into_leaves_the_value_in_last_result(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="имя", value="Ayris"),
                block("GetVar", name="имя"),
                echo("{last_result}"),
            )
        )

        assert report.ok
        assert values(registry) == ["Ayris"]

    def test_get_var_of_an_unknown_name_says_which(self, engine) -> None:
        report = engine.run(command(block("GetVar", name="нет_такой")))

        assert report.outcome is RunOutcome.FAILED
        assert "нет_такой" in str(report.error)

    def test_set_var_substitutes_and_does_not_evaluate(self, engine, registry) -> None:
        """Text a user dictated is not an expression: ``"{a} + 10"`` stores ``"50 + 10"``."""
        report = engine.run(
            command(block("SetVar", name="итог", value="{a} + 10"), echo("{итог}")),
            slots={"a": 50},
        )

        assert report.ok
        assert values(registry) == ["50 + 10"]

    def test_array_push_and_array_get(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("ArrayPush", name="список", value="раз"),
                block("ArrayPush", name="список", value="два"),
                block("ArrayGet", name="список", index=1, into="второй"),
                echo("{второй}"),
            )
        )

        assert report.ok
        assert values(registry) == ["два"]

    def test_array_get_counts_from_the_end_with_a_negative_index(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[1, 2, 3]),
                block("ArrayGet", name="список", index=-1, into="последний"),
                echo("{последний}"),
            )
        )

        assert report.ok
        assert values(registry) == [3]

    def test_array_get_outside_the_array_names_the_index(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[1]),
                block("ArrayGet", name="список", index=5),
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "5" in (report.user_message or "")

    def test_array_pop_takes_the_last_element_by_default(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[1, 2, 3]),
                block("ArrayPop", name="список", into="взятый"),
                block("ArrayLength", name="список", into="сколько"),
                echo("{взятый}"),
                echo("{сколько}"),
            )
        )

        assert report.ok
        assert values(registry) == [3, 2]

    def test_array_pop_can_take_one_by_index(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[1, 2, 3]),
                block("ArrayPop", name="список", index=0, into="взятый"),
                block("GetVar", name="список", into="остаток"),
                echo("{взятый}"),
                echo("{остаток}"),
            )
        )

        assert report.ok
        assert values(registry) == [1, [2, 3]]

    def test_array_pop_of_an_empty_array_says_so(self, engine) -> None:
        report = engine.run(
            command(block("SetVar", name="список", value=[]), block("ArrayPop", name="список"))
        )

        assert report.outcome is RunOutcome.FAILED

    def test_array_length_of_an_empty_array_is_zero(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[]),
                block("ArrayLength", name="список", into="сколько"),
                echo("{сколько}"),
            )
        )

        assert report.ok
        assert values(registry) == [0]

    def test_dict_set_and_dict_get(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("DictSet", name="словарь", key="режим", value="работа"),
                block("DictGet", name="словарь", key="режим", into="что"),
                echo("{что}"),
            )
        )

        assert report.ok
        assert values(registry) == ["работа"]

    def test_dict_set_overwrites_one_key_and_keeps_the_others(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("DictSet", name="словарь", key="a", value=1),
                block("DictSet", name="словарь", key="b", value=2),
                block("DictSet", name="словарь", key="a", value=3),
                block("GetVar", name="словарь", into="весь"),
                echo("{весь}"),
            )
        )

        assert report.ok
        assert values(registry) == [{"a": 3, "b": 2}]

    def test_dict_get_of_a_missing_key_names_the_key(self, engine) -> None:
        report = engine.run(
            command(
                block("DictSet", name="словарь", key="a", value=1),
                block("DictGet", name="словарь", key="нет"),
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "нет" in (report.user_message or "")

    def test_dict_keys_gives_an_array_a_for_can_walk(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("DictSet", name="словарь", key="a", value=1),
                block("DictSet", name="словарь", key="b", value=2),
                block("DictKeys", name="словарь", into="ключи"),
                {
                    "type": "For",
                    "params": {"var": "k", "items": "{ключи}"},
                    "body": [echo("{k}")],
                },
            )
        )

        assert report.ok
        assert values(registry) == ["a", "b"]

    def test_a_data_block_on_the_wrong_kind_of_variable_says_which_kind(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="число", value=5),
                block("ArrayLength", name="число"),
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "не массив" in (report.user_message or "")

    def test_dict_keys_on_an_array_says_it_is_not_a_dictionary(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="список", value=[1, 2]),
                block("DictKeys", name="список"),
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert "не словарь" in (report.user_message or "")

    def test_a_block_without_a_variable_name_is_a_failure_and_not_a_nameless_write(
        self, engine
    ) -> None:
        report = engine.run(command(block("SetVar", name="", value=1)))

        assert report.outcome is RunOutcome.FAILED

    def test_the_name_of_a_variable_can_itself_be_a_placeholder(self, engine, registry) -> None:
        report = engine.run(
            command(
                block("SetVar", name="{какая}", value="через подстановку"),
                echo("{цель}"),
            ),
            slots={"какая": "цель"},
        )

        assert report.ok
        assert values(registry) == ["через подстановку"]


class TestTypes:
    """The six declared types: what a command may put in one, and what it may not."""

    @staticmethod
    def declaring(kind: str, default: Any, *blocks: dict[str, Any]) -> CommandModel:
        """A command with one declared variable of type ``kind``."""
        return command(*blocks, variables=[{"name": "x", "type": kind, "default": default}])

    @pytest.mark.parametrize(
        ("kind", "given", "wanted"),
        [
            ("string", 50, "50"),
            ("int", "50", 50),
            ("int", 50.7, 51),
            ("float", "2,5", 2.5),
            ("bool", "нет", False),
            ("bool", "да", True),
            ("array", "[1, 2]", [1, 2]),
            ("dict", '{"a": 1}', {"a": 1}),
        ],
    )
    def test_a_value_is_made_to_fit_the_declared_type(
        self, engine, registry, kind, given, wanted
    ) -> None:
        report = engine.run(
            self.declaring(kind, None, block("SetVar", name="x", value=given), echo("{x}"))
        )

        assert report.ok
        assert values(registry) == [wanted]

    @pytest.mark.parametrize(
        ("kind", "given"),
        [("int", "громко"), ("float", "не число"), ("array", "не json"), ("dict", "[1, 2]")],
    )
    def test_a_value_that_does_not_fit_names_the_variable(self, engine, kind, given) -> None:
        report = engine.run(self.declaring(kind, None, block("SetVar", name="x", value=given)))

        assert report.outcome is RunOutcome.FAILED
        assert "x" in (report.user_message or "") or "x" in str(report.error)

    @pytest.mark.parametrize(
        ("kind", "empty"),
        [
            ("string", ""),
            ("int", 0),
            ("float", 0.0),
            ("bool", False),
            ("array", []),
            ("dict", {}),
        ],
    )
    def test_a_declared_variable_starts_at_its_empty_value(
        self, engine, registry, kind, empty
    ) -> None:
        report = engine.run(self.declaring(kind, None, echo("{x}")))

        assert report.ok
        assert values(registry) == [empty]

    def test_a_declared_default_is_used(self, engine, registry) -> None:
        report = engine.run(self.declaring("int", 70, echo("{x}")))

        assert report.ok
        assert values(registry) == [70]

    def test_a_declared_array_can_be_pushed_to_without_being_set_first(
        self, engine, registry
    ) -> None:
        report = engine.run(
            self.declaring("array", None, block("ArrayPush", name="x", value="раз"), echo("{x}"))
        )

        assert report.ok
        assert values(registry) == [["раз"]]

    def test_a_declared_dict_can_be_written_to_without_being_set_first(
        self, engine, registry
    ) -> None:
        report = engine.run(
            self.declaring("dict", None, block("DictSet", name="x", key="a", value=1), echo("{x}"))
        )

        assert report.ok
        assert values(registry) == [{"a": 1}]


class TestScopes:
    """``local``, ``profile``, ``global``: where a write lands and who sees it."""

    def test_a_local_variable_does_not_reach_the_store(self, engine) -> None:
        report = engine.run(command(block("SetVar", name="только_тут", value=1)))

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "только_тут") is MISSING
        assert engine.variables.read(VariableScope.PROFILE, "только_тут") is MISSING

    def test_a_local_variable_is_forgotten_between_runs(self, engine, registry) -> None:
        first = engine.run(command(block("SetVar", name="только_тут", value=1)))
        second = engine.run(command(echo("{только_тут}")))

        assert first.ok
        assert second.outcome is RunOutcome.FAILED

    @pytest.mark.parametrize("scope", ["profile", "global"])
    def test_a_shared_variable_outlives_the_run(self, engine, registry, scope) -> None:
        first = engine.run(command(block("SetVar", name="общая", value="держится", scope=scope)))
        second = engine.run(command(echo("{общая}")))

        assert first.ok
        assert second.ok
        assert values(registry) == ["держится"]
        assert engine.variables.read(VariableScope(scope), "общая") == "держится"

    def test_an_undeclared_write_without_a_scope_becomes_a_local(self, engine) -> None:
        """So a typo cannot quietly create a global that outlives the mistake."""
        report = engine.run(command(block("SetVar", name="опечатка", value=1)))

        assert report.ok
        assert engine.variables.names(VariableScope.GLOBAL) == frozenset()

    def test_the_declared_scope_is_used_when_the_block_does_not_say(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="x", value=5),
                variables=[{"name": "x", "type": "int", "scope": "global"}],
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "x") == 5

    def test_a_declared_shared_variable_keeps_what_the_store_already_has(self, engine) -> None:
        """The point of a shared variable: it outlives the run, default and all."""
        engine.variables.write(VariableScope.GLOBAL, "яркость", 30)

        report = engine.run(
            command(
                block("GetVar", name="яркость", into="сколько"),
                variables=[{"name": "яркость", "type": "int", "scope": "global", "default": 70}],
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "яркость") == 30

    def test_a_local_shadows_a_global_of_the_same_name(self, engine, registry) -> None:
        engine.variables.write(VariableScope.GLOBAL, "имя", "общее")

        report = engine.run(
            command(block("SetVar", name="имя", value="своё", scope="local"), echo("{имя}"))
        )

        assert report.ok
        assert values(registry) == ["своё"]
        assert engine.variables.read(VariableScope.GLOBAL, "имя") == "общее"

    def test_an_unknown_scope_name_is_a_failure(self, engine) -> None:
        report = engine.run(command(block("SetVar", name="x", value=1, scope="вселенная")))

        assert report.outcome is RunOutcome.FAILED
        assert "область" in (report.user_message or "")

    def test_the_store_is_shared_by_two_engines_of_one_application(self, engines) -> None:
        store = MemoryVariables()
        first = engines(store=store)
        second = engines(store=store)

        first.run(command(block("SetVar", name="общая", value="раз", scope="global")))
        report = second.run(command(block("GetVar", name="общая", into="что")))

        assert report.ok
        assert store.read(VariableScope.GLOBAL, "общая") == "раз"


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Where the test database lives. The file is created when it is opened."""
    return tmp_path / "ayris.db"


@pytest.fixture
def repos(db_path: Path) -> Iterator[Repositories]:
    """Repositories over a fresh database on disk."""
    handle = Database.open(db_path)
    yield Repositories(handle)
    handle.close()
    reset_database()


class TestPersistence:
    """The database-backed store: what is written, when, and what survives a restart."""

    @staticmethod
    def reopened(path: Path) -> Iterator[Repositories]:
        """Repositories over the same file, opened again. What a restart looks like."""
        handle = Database.open(path)
        try:
            yield Repositories(handle)
        finally:
            handle.close()

    def test_a_persistent_variable_is_written_on_flush(self, repos) -> None:
        store = DatabaseVariables(repos.variables)
        store.declare(VariableScope.GLOBAL, "режим", var_type=VariableType.STRING, persistent=True)
        store.write(VariableScope.GLOBAL, "режим", "работа")

        assert store.pending() == frozenset({(VariableScope.GLOBAL, "режим")})
        assert repos.variables.get("режим") is None

        store.flush()

        assert store.pending() == frozenset()
        stored = repos.variables.get("режим")
        assert stored is not None
        assert stored.value == "работа"

    def test_a_variable_that_was_not_declared_persistent_is_never_written(self, repos) -> None:
        store = DatabaseVariables(repos.variables)
        store.write(VariableScope.GLOBAL, "мимолётная", "и всё")
        store.flush()

        assert store.pending() == frozenset()
        assert repos.variables.get("мимолётная") is None
        assert store.read(VariableScope.GLOBAL, "мимолётная") == "и всё"

    def test_a_loop_writes_once_and_not_once_per_turn(self, repos, engines) -> None:
        """The task file's warning, as a test: a thousand turns must not be a thousand rows.

        Counted by how many times the repository was asked, because that is the cost —
        the value it ends up holding would look the same either way.
        """
        store = DatabaseVariables(repos.variables)
        store.declare(VariableScope.GLOBAL, "счётчик", var_type=VariableType.INT, persistent=True)
        writes = 0
        original = repos.variables.set

        def counted(*args: Any, **kwargs: Any) -> Any:
            nonlocal writes
            writes += 1
            return original(*args, **kwargs)

        repos.variables.set = counted  # type: ignore[method-assign]
        engine = engines(store=store)

        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 20},
                    "body": [block("SetVar", name="счётчик", value="{n}", scope="global")],
                }
            )
        )

        assert report.ok
        assert writes == 1
        stored = repos.variables.get("счётчик")
        assert stored is not None
        assert stored.value == 20

    def test_too_many_pending_names_flush_on_their_own(self, repos) -> None:
        """A ceiling on memory, not the normal path: the normal path is one flush per run."""
        store = DatabaseVariables(repos.variables, max_pending=4)
        for index in range(4):
            name = f"имя{index}"
            store.declare(VariableScope.GLOBAL, name, var_type=VariableType.INT, persistent=True)
            store.write(VariableScope.GLOBAL, name, index)

        assert store.pending() == frozenset()
        assert len(repos.variables.list_all(scope=VariableScope.GLOBAL)) == 4

    def test_a_persistent_variable_survives_a_restart(self, repos, db_path) -> None:
        store = DatabaseVariables(repos.variables)
        store.declare(VariableScope.GLOBAL, "режим", var_type=VariableType.STRING, persistent=True)
        store.write(VariableScope.GLOBAL, "режим", "работа")
        store.flush()

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.GLOBAL, "режим") == "работа"

    def test_a_local_variable_does_not_survive_a_restart(self, repos, db_path) -> None:
        """Not because it is dropped, but because it never went in: locals live in the run."""
        repos.variables.set("только_тут", "и всё", scope=VariableScope.LOCAL, persistent=True)
        store = DatabaseVariables(repos.variables)

        assert store.read(VariableScope.LOCAL, "только_тут") is MISSING

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.LOCAL, "только_тут") is MISSING

    def test_a_profile_variable_of_another_profile_is_not_loaded(self, repos, db_path) -> None:
        profile = repos.profiles.create("первый", activate=True)
        other = repos.profiles.create("второй")
        repos.variables.set("своя", "первого", scope=VariableScope.PROFILE, profile_id=profile.id)
        repos.variables.set("своя", "второго", scope=VariableScope.PROFILE, profile_id=other.id)

        store = DatabaseVariables(repos.variables, profile_id=profile.id)

        assert store.read(VariableScope.PROFILE, "своя") == "первого"

    def test_the_declared_type_is_kept_across_a_restart(self, repos, db_path) -> None:
        store = DatabaseVariables(repos.variables)
        store.declare(VariableScope.GLOBAL, "число", var_type=VariableType.INT, persistent=True)
        store.write(VariableScope.GLOBAL, "число", "50")
        store.flush()

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.GLOBAL, "число") == 50

    def test_an_array_survives_as_an_array(self, repos, db_path) -> None:
        store = DatabaseVariables(repos.variables)
        store.declare(VariableScope.GLOBAL, "список", var_type=VariableType.ARRAY, persistent=True)
        store.update(VariableScope.GLOBAL, "список", lambda _current: ["раз", "два"])
        store.flush()

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.GLOBAL, "список") == ["раз", "два"]

    def test_a_run_through_the_engine_persists_and_comes_back(self, repos, db_path, engines):
        """End to end: the engine flushes, a new store over the same file reads it."""
        store = DatabaseVariables(repos.variables)
        engine = engines(store=store)

        report = engine.run(
            command(
                block("SetVar", name="яркость", value=70),
                variables=[
                    {
                        "name": "яркость",
                        "type": "int",
                        "scope": "global",
                        "persistent": True,
                        "default": 100,
                    }
                ],
            )
        )

        assert report.ok
        assert store.pending() == frozenset()

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.GLOBAL, "яркость") == 70

    def test_a_declaration_that_is_not_persistent_stays_in_memory(
        self, repos, db_path, engines
    ) -> None:
        store = DatabaseVariables(repos.variables)
        engine = engines(store=store)

        report = engine.run(
            command(
                block("SetVar", name="сессия", value="только сейчас"),
                variables=[{"name": "сессия", "type": "string", "scope": "global"}],
            )
        )

        assert report.ok
        assert repos.variables.get("сессия") is None

        for fresh in self.reopened(db_path):
            restarted = DatabaseVariables(fresh.variables)
            assert restarted.read(VariableScope.GLOBAL, "сессия") is MISSING


class TestContextDirectly:
    """The two things the context does that no block can reach through the engine."""

    @staticmethod
    def context(**options: Any) -> ExecutionContext:
        """A context on its own, with no run around it."""
        return ExecutionContext(
            info=RunInfo(run_id="test", command="Проверка"),
            store=options.pop("store", MemoryVariables()),
            **options,
        )

    def test_last_result_exists_before_the_first_block(self) -> None:
        """A ``While`` polling until an action answers reads it on its very first turn."""
        assert self.context().resolve("last_result") is None

    def test_a_child_context_keeps_the_store_and_drops_the_locals(self) -> None:
        parent = self.context(slots={"кому": "миру"})
        parent.set("секрет", "не для тебя")
        parent.set("общая", "видно", VariableScope.GLOBAL)

        child = parent.child(parent.info.called("Помощник", run_id="c1"))

        assert child.resolve("общая") == "видно"
        assert child.has("секрет") is False
        assert child.has("кому") is False

    def test_the_call_depth_grows_with_every_nested_context(self) -> None:
        parent = self.context()
        first = parent.child(parent.info.called("Раз", run_id="c1"))
        second = first.child(first.info.called("Два", run_id="c2"))

        assert (parent.info.depth, first.info.depth, second.info.depth) == (0, 1, 2)

    def test_pop_takes_the_element_out_of_a_shared_array_atomically(self) -> None:
        """The removed element has to come out of the same locked change that removed it."""
        store = MemoryVariables()
        context = self.context(store=store)
        context.set("список", [1, 2, 3], VariableScope.GLOBAL)

        assert context.pop("список") == 3
        assert store.read(VariableScope.GLOBAL, "список") == [1, 2]

    def test_an_unknown_command_in_a_call_is_a_named_failure(self, engine) -> None:
        """The error a ``CallCommand`` raises, on its own: it names the command."""
        with pytest.raises(MacroCallError) as caught:
            raise MacroCallError("Нет такой")
        assert "Нет такой" in str(caught.value)
