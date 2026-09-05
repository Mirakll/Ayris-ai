"""Задача 31: интерпретатор команд — что он делает, когда останавливается, что оставляет.

The engine is the one place in Ayris where a user's own program runs, so these tests are
written from the outside: a fake registry, a real event bus, real threads, and the two
commands of section 22 read off disk. Five things carry the weight.

*The examples of section 22 run.* :class:`TestSectionTwentyTwoExamples` loads the same
fixtures task 30 wrote and checks the block order, the branch an ``If`` took and the
parameters that actually reached the registry — a slot filled with ``50`` has to arrive as
the number ``50``, not as the text ``{volume}``.

*Stopping has to be quick.* :class:`TestCancellation` cancels a command sitting in a five
second ``Wait`` and asserts it is over in well under 200 ms. A ``Wait`` that slept in
:func:`time.sleep` would pass every other test in this file and fail this one.

*A refusal is an answer, not a silence.* A command fired inside its own cooldown, or
switched off, comes back with a report saying so and an event on the bus:
:class:`TestCooldownAndRefusal` pins both, because a hotkey that does nothing and says
nothing is indistinguishable from a broken hotkey.

*A broken block is not a broken assistant.* :class:`TestFailurePolicy` and
:class:`TestLimits` cover the two halves of that: ``on_error`` decides whether the chain
goes on, and the limits decide when an endless loop is stopped for the user.

*The report is the whole answer.* :class:`TestReport` reads the timings, the paths and the
nesting the DevTools timeline and the debugger of task 35 are drawn from.

Groups:

* :class:`TestSectionTwentyTwoExamples` — the two fixtures, end to end.
* :class:`TestBlocks` — the nineteen logic blocks: branches, loops, variables.
* :class:`TestCancellation` — the stop word, the hotkey, and how fast they land.
* :class:`TestCooldownAndRefusal` — cooldown, disabled, and what is published.
* :class:`TestConcurrency` — two triggers at once: parallel, queued, preempted.
* :class:`TestFailurePolicy` — ``continue`` against ``stop``, and ``Try``/``Catch``.
* :class:`TestLimits` — iterations, depth, steps, budget, pause.
* :class:`TestCalls` — one command calling another.
* :class:`TestReport` — timings, order, and the text a log line shows.
* :class:`TestEngineLifecycle` — what the engine holds and how it lets go.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayris.actions.macros.context import MISSING, MemoryVariables, TriggerSource
from ayris.actions.macros.engine import (
    _HANDLERS,
    ConcurrencyPolicy,
    ExecutionLimits,
    MacroEngine,
)
from ayris.actions.macros.errors import (
    MacroBlockError,
    MacroCallError,
    MacroCancelledError,
    MacroEngineStoppedError,
    MacroLimitError,
)
from ayris.actions.macros.report import RunOutcome, StepStatus
from ayris.actions.macros.schema import LOGIC_BLOCKS, CommandModel
from ayris.actions.macros.serializer import read_document
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionUnavailable
from ayris.core.events import (
    CancelRequested,
    EventBus,
    MacroBlockFinished,
    MacroCancelled,
    MacroEnded,
    MacroFailed,
    MacroFinished,
    MacroSkipped,
    MacroStarted,
)
from ayris.core.models import ExecutionResult, VariableScope

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ayris.core.events import Event

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "macros"

#: Long enough that no scheduling hiccup ends it by itself, so a test that sees a
#: cancelled ``Wait`` knows the cancel is what ended it.
FOREVER_MS = 5_000

#: How long a test waits for another thread before calling it a failure. Generous,
#: because it is only ever reached when something is broken.
GIVE_UP_S = 5.0


def until(check: Callable[[], bool], timeout: float = GIVE_UP_S) -> bool:
    """Whether ``check`` became true within ``timeout``, polling in small steps.

    For the assertions about two threads at once, where the alternative is a
    :func:`time.sleep` long enough to be slow and short enough to be flaky.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.005)
    return check()


class FakeRegistry:
    """The registry as the interpreter sees it: two methods and a memory of the calls.

    Deliberately not an :class:`~ayris.actions.registry.ActionRegistry` with test actions
    registered in it. Invariant 5 says the engine reaches actions only through this pair of
    methods, and a fake is how that stays proven rather than assumed — anything the engine
    started doing behind the registry's back would show up as a test that cannot be written.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.results: dict[str, ActionResult[Any]] = {}
        self.answers: dict[str, list[ActionResult[Any]]] = {}
        self.raises: dict[str, Exception] = {}
        self.delays: dict[str, float] = {}
        self.hold: dict[str, threading.Event] = {}
        self.unknown: set[str] = set()
        self.started = threading.Event()
        self._lock = threading.Lock()

    def has(self, name: str) -> bool:
        """Every action exists unless a test says otherwise."""
        return name not in self.unknown

    def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        """Record the call, then do whatever the test asked this action to do."""
        with self._lock:
            self.calls.append((name, dict(params or {})))
        self.started.set()
        gate = self.hold.get(name)
        if gate is not None:
            gate.wait(GIVE_UP_S)
        delay = self.delays.get(name, 0.0)
        if delay:
            time.sleep(delay)
        error = self.raises.get(name)
        if error is not None:
            raise error
        queued = self.answers.get(name)
        if queued:
            return queued.pop(0)
        return self.results.get(name, ActionResult.done("готово", value=name))

    @property
    def names(self) -> list[str]:
        """Just the action names, in call order."""
        with self._lock:
            return [name for name, _ in self.calls]

    def params(self, name: str) -> dict[str, Any]:
        """The parameters of the first call of ``name``."""
        with self._lock:
            return next(params for called, params in self.calls if called == name)


class Recorder:
    """Every macro event the bus delivered, in order.

    Subscribes to :class:`~ayris.core.events.MacroEnded` rather than to its three
    subclasses, which is how the history tab of task 33 will do it — and pins on the way
    that a subscription to the base really does hear a failure.
    """

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        self._keep = [
            bus.subscribe(kind, self.events.append, weak=False)
            for kind in (MacroStarted, MacroBlockFinished, MacroEnded, MacroSkipped)
        ]

    def of(self, kind: type[Event]) -> list[Any]:
        """Only the events of this class, subclasses included."""
        return [event for event in self.events if isinstance(event, kind)]

    def one(self, kind: type[Event]) -> Any:
        """The single event of this class, asserting that there is exactly one."""
        found = self.of(kind)
        assert len(found) == 1, f"{kind.__name__}: ожидалось одно событие, пришло {len(found)}"
        return found[0]

    @property
    def blocks(self) -> list[str]:
        """The paths of the finished blocks, in the order they were announced."""
        return [event.path for event in self.of(MacroBlockFinished)]


def command(*blocks: dict[str, Any], name: str = "Проверка", **fields: Any) -> CommandModel:
    """A command built from block dictionaries, the way a ``.ayris`` file carries them."""
    return CommandModel.model_validate({"name": name, "actions": list(blocks), **fields})


def example(file: str) -> CommandModel:
    """One of the section 22 commands, read from the fixture task 30 wrote."""
    return read_document(FIXTURES / file).commands[0]


def block(kind: str, /, **params: Any) -> dict[str, Any]:
    """One block: its type and its parameters, with no branches."""
    return {"type": kind, "params": params}


def switch_command() -> CommandModel:
    """A ``Switch`` over a slot, with two cases and a default."""
    return command(
        {
            "type": "Switch",
            "params": {"value": "{mode}"},
            "body": [
                {"type": "Case", "params": {"value": "тихо"}, "body": [block("Mute")]},
                {"type": "Case", "params": {"value": "громко"}, "body": [block("Loud")]},
                {"type": "Default", "body": [block("Nothing")]},
            ],
        }
    )


@pytest.fixture
def registry() -> FakeRegistry:
    """A registry that says yes to everything and remembers what it was asked for."""
    return FakeRegistry()


@pytest.fixture
def bus() -> EventBus:
    """A bus that delivers on the publishing thread, so a test sees an event at once."""
    return EventBus(thread_id=None)


@pytest.fixture
def heard(bus: EventBus) -> Recorder:
    """Everything the engine said while the test ran."""
    return Recorder(bus)


@pytest.fixture
def engines(registry: FakeRegistry, bus: EventBus) -> Iterator[Callable[..., MacroEngine]]:
    """A factory for engines, every one of them shut down when the test ends.

    A leaked :class:`~concurrent.futures.ThreadPoolExecutor` outlives the test that made it
    and its threads then fail somewhere else, so the cleanup is the fixture's whole job.
    """
    made: list[MacroEngine] = []

    def make(**options: Any) -> MacroEngine:
        engine = MacroEngine(registry, bus=bus, **options)
        made.append(engine)
        return engine

    yield make
    for engine in made:
        engine.shutdown()


@pytest.fixture
def engine(engines) -> MacroEngine:
    """The ordinary engine: real threads, the test's bus, everything else at its default."""
    return engines()


class TestSectionTwentyTwoExamples:
    """The two commands of section 22, from the file to the registry call."""

    def test_volume_50_fills_the_slot(self, engine, registry) -> None:
        report = engine.run(example("volume_50.ayris"), slots={"volume": 50})

        assert report.ok
        assert registry.calls == [("SetVolume", {"level": 50})]
        assert [step.path for step in report.steps] == ["actions[0]"]
        assert report.execution is ExecutionResult.OK

    def test_work_mode_runs_in_order_and_takes_the_then_branch(self, engine, registry, heard):
        report = engine.run(example("work_mode.ayris"), request_id="req-1")

        assert report.ok
        assert registry.names == ["RunApp", "RunApp", "SetBrightness"]
        assert registry.params("RunApp")["app"] == "code"
        # 70 is the command's own ``profile`` declaration: a local declaration shadows
        # whatever a global of the same name holds, and the ``If`` above it agrees.
        assert registry.params("SetBrightness") == {"monitor": "external_1", "level": 70}
        # Children before the parent — an ``If`` is recorded when its branch is done, which
        # is what makes a step's duration contain everything it caused.
        assert [step.path for step in report.steps] == [
            "actions[0]",
            "actions[1]",
            "actions[2]",
            "actions[3].then[0]",
            "actions[3]",
        ]
        assert report.step("actions[3]").message == "then"
        assert engine.variables.read(VariableScope.GLOBAL, "work_mode") is True
        assert heard.blocks == [step.path for step in report.steps]
        assert heard.one(MacroStarted).request_id == "req-1"
        assert isinstance(heard.one(MacroEnded), MacroFinished)

    def test_work_mode_leaves_brightness_alone_when_the_profile_says_zero(self, engine, registry):
        engine.variables.write(VariableScope.PROFILE, "work_monitor_brightness", 0)

        report = engine.run(example("work_mode.ayris"))

        assert report.ok
        assert registry.names == ["RunApp", "RunApp"]
        assert report.step("actions[3]").message == "else"


class TestBlocks:
    """The nineteen logic blocks: what each one does to the run around it."""

    def test_every_logic_block_of_the_language_has_a_handler(self) -> None:
        """A block added to task 30's language without an interpreter fails here.

        Otherwise it would be treated as the name of an action, and the user would be told
        that ``Repeat`` is not connected instead of that it is not implemented.
        """
        assert set(_HANDLERS) == set(LOGIC_BLOCKS)

    def test_set_var_writes_to_the_scope_the_block_names(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="a", value="раз"),
                block("SetVar", name="b", value="два", scope="profile"),
                block("SetVar", name="both", value="{a} и {b}", scope="global"),
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.PROFILE, "b") == "два"
        assert engine.variables.read(VariableScope.GLOBAL, "both") == "раз и два"
        assert engine.variables.read(VariableScope.GLOBAL, "a") is MISSING
        assert report.step("actions[0]").message == "a = раз"

    def test_set_var_substitutes_and_does_not_calculate(self, engine, registry) -> None:
        """``{n} + 1`` is text; the same arithmetic inside a condition is arithmetic.

        The boundary is deliberate: an expression is only safe to compute because calls and
        attributes are refused, and a condition is the one place section 22 needs one.
        """
        report = engine.run(
            command(
                block("SetVar", name="n", value=1),
                block("SetVar", name="text", value="{n} + 1", scope="global"),
                {"type": "If", "params": {"condition": "{n} + 1 == 2"}, "then": [block("Yes")]},
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "text") == "1 + 1"
        assert registry.names == ["Yes"]

    def test_get_var_reads_into_a_name_and_into_last_result(self, engine) -> None:
        report = engine.run(
            command(
                block("SetVar", name="src", value=7),
                block("GetVar", name="src", into="copy"),
                block("SetVar", name="seen", value="{last_result}/{copy}", scope="global"),
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "seen") == "7/7"

    def test_arrays_and_dictionaries_are_read_and_written_by_their_blocks(self, engine) -> None:
        report = engine.run(
            command(
                block("ArrayPush", name="items", value="раз"),
                block("ArrayPush", name="items", value="два"),
                block("ArrayGet", name="items", index=-1, into="last"),
                block("DictSet", name="map", key="k", value=5),
                block("DictGet", name="map", key="k", into="got"),
                block("SetVar", name="out", value="{last}-{got}", scope="global"),
                variables=[
                    {"name": "items", "type": "array", "scope": "local"},
                    {"name": "map", "type": "dict", "scope": "local"},
                ],
            )
        )

        assert report.ok
        assert engine.variables.read(VariableScope.GLOBAL, "out") == "два-5"
        assert report.step("actions[1]").message == "items: 2 элементов"

    def test_a_missing_array_element_is_a_failure_that_names_the_index(self, engine) -> None:
        report = engine.run(
            command(
                block("ArrayPush", name="items", value="раз"),
                block("ArrayGet", name="items", index=5),
                variables=[{"name": "items", "type": "array"}],
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroBlockError)
        assert report.error.path == "actions[1]"
        assert "нет элемента с номером 5" in report.error.user_message

    def test_switch_runs_the_matching_case_and_records_the_rest_as_skipped(self, engine, registry):
        report = engine.run(switch_command(), slots={"mode": "громко"})

        assert registry.names == ["Loud"]
        assert report.step("actions[0]").message == "body[1]"
        assert report.step("actions[0].body[0]").status is StepStatus.SKIPPED
        assert report.step("actions[0].body[2]").status is StepStatus.SKIPPED

    def test_switch_falls_through_to_default(self, engine, registry) -> None:
        report = engine.run(switch_command(), slots={"mode": "шёпот"})

        assert report.ok
        assert registry.names == ["Nothing"]

    def test_while_repeats_until_its_condition_goes_false(self, engine, registry) -> None:
        """A poll, which is what a ``While`` is for: the world changes, not the counter."""
        registry.answers["Poll"] = [
            ActionResult.done(value="ждём"),
            ActionResult.done(value="ждём"),
            ActionResult.done(value="готово"),
        ]
        report = engine.run(
            command(
                {
                    "type": "While",
                    "params": {"condition": '{last_result} != "готово"'},
                    "body": [block("Poll")],
                }
            )
        )

        assert report.ok
        assert registry.names == ["Poll", "Poll", "Poll"]
        assert report.step("actions[0]").message == "3 итераций"

    def test_for_walks_a_list_and_break_leaves_early(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "item", "items": ["раз", "два", "стоп", "три"]},
                    "body": [
                        {
                            "type": "If",
                            "params": {"condition": '{item} == "стоп"'},
                            "then": [block("Break")],
                        },
                        block("Take", what="{item}"),
                    ],
                }
            )
        )

        assert report.ok
        assert [params["what"] for _, params in registry.calls] == ["раз", "два"]
        assert report.step("actions[0]").message == "3 из 4 итераций"

    def test_for_counts_a_range_and_continue_skips_one_turn(self, engine, registry) -> None:
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 4},
                    "body": [
                        {
                            "type": "If",
                            "params": {"condition": "{n} == 2"},
                            "then": [block("Continue")],
                        },
                        block("Tick", n="{n}"),
                    ],
                }
            )
        )

        assert report.ok
        assert [params["n"] for _, params in registry.calls] == [1, 3, 4]
        assert report.step("actions[0]").message == "4 из 4 итераций"

    def test_return_ends_the_command_and_carries_a_value(self, engine, registry) -> None:
        report = engine.run(command(block("First"), block("Return", value="итог"), block("Never")))

        assert report.ok
        assert report.value == "итог"
        assert registry.names == ["First"]

    def test_a_switched_off_block_is_recorded_and_not_run(self, engine, registry) -> None:
        report = engine.run(command({"type": "Skip", "enabled": False}, block("Run")))

        assert registry.names == ["Run"]
        step = report.step("actions[0]")
        assert step.status is StepStatus.SKIPPED
        assert step.message == "выключен"

    def test_wait_records_the_pause_it_took(self, engine) -> None:
        report = engine.run(command(block("Wait", ms=20)))

        assert report.ok
        assert report.step("actions[0]").message == "20 мс"

    def test_sleep_is_the_same_block_under_the_other_name(self, engine) -> None:
        """Section 22 lists both spellings; a command may use either."""
        report = engine.run(command(block("Sleep", ms=5)))

        assert report.ok
        assert report.step("actions[0]").block == "Sleep"
        assert report.step("actions[0]").message == "5 мс"

    def test_an_if_without_an_else_simply_goes_on(self, engine, registry) -> None:
        report = engine.run(
            command(
                {"type": "If", "params": {"condition": "1 == 2"}, "then": [block("Never")]},
                block("After"),
            )
        )

        assert report.ok
        assert registry.names == ["After"]
        assert report.step("actions[0]").message == "else"

    def test_for_walks_the_comma_separated_line_a_person_types(self, engine, registry) -> None:
        """What arrives from the editor's one-line field, or from a slot."""
        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "app", "items": "{apps}"},
                    "body": [block("RunApp", app="{app}")],
                }
            ),
            slots={"apps": "code, chrome"},
        )

        assert report.ok
        assert [params["app"] for _, params in registry.calls] == ["code", "chrome"]


class TestCancellation:
    """Stopping a command: how it is asked for, and how long it takes to land."""

    @pytest.mark.slow
    def test_cancel_reaches_a_command_sitting_in_a_wait(self, engine, registry, heard) -> None:
        """The whole point of not sleeping in :func:`time.sleep`, in one measurement.

        The command is in a five second ``Wait``; the run has to be over in a fraction of
        that, and the report has to say it was cancelled rather than that it finished.
        """
        run = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS)))
        assert registry.started.wait(GIVE_UP_S)

        asked_at = time.perf_counter()
        assert engine.cancel(run.run_id, reason="стоп-слово") == 1
        report = run.wait(GIVE_UP_S)
        took_ms = (time.perf_counter() - asked_at) * 1000

        assert report.outcome is RunOutcome.CANCELLED
        assert took_ms < 200, f"отмена заняла {took_ms:.0f} мс"
        assert report.duration_ms < FOREVER_MS
        assert report.step("actions[1]").status is StepStatus.CANCELLED
        assert report.user_message == "Отменено."
        assert report.execution is ExecutionResult.CANCELLED
        cancelled = heard.one(MacroCancelled)
        assert cancelled.reason == "стоп-слово"
        assert cancelled.run_id == run.run_id

    def test_the_stop_word_on_the_bus_stops_every_run(self, engine, registry, bus) -> None:
        """What the stop word actually does: one event, and no engine method called.

        The subscription is a bound method, and the bus holds handlers weakly — a run
        that ignored :class:`~ayris.core.events.CancelRequested` because the engine's
        subscription had been garbage collected is exactly the bug this pins.
        """
        run = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS)))
        assert registry.started.wait(GIVE_UP_S)

        bus.publish(CancelRequested(reason="хватит"))

        report = run.wait(GIVE_UP_S)
        assert report.cancelled
        assert isinstance(report.error, MacroCancelledError)
        assert report.error.reason == "хватит"

    def test_a_run_still_waiting_for_a_worker_is_cancelled_too(self, engines, registry) -> None:
        """A queued run is reachable: it is registered before it is submitted."""
        engine = engines(policy=ConcurrencyPolicy.QUEUE, threads=1)
        first = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS), name="Первая"))
        assert registry.started.wait(GIVE_UP_S)
        second = engine.start(command(block("Second"), name="Вторая"))

        assert engine.cancel_all("выход") == 2

        assert first.wait(GIVE_UP_S).cancelled
        queued = second.wait(GIVE_UP_S)
        assert queued.cancelled
        assert not queued.ran
        assert "Second" not in registry.names

    def test_cancelling_by_name_leaves_the_other_command_alone(self, engine, registry) -> None:
        loud = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS), name="Долгая"))
        assert registry.started.wait(GIVE_UP_S)
        quiet = engine.start(command(block("Wait", ms=30), name="Короткая"))

        assert engine.cancel(command="Долгая", reason="не эту") == 1

        assert loud.wait(GIVE_UP_S).cancelled
        assert quiet.wait(GIVE_UP_S).ok

    def test_cancelling_nothing_is_not_an_error(self, engine) -> None:
        assert engine.cancel("нет такого запуска") == 0
        assert engine.cancel_all("тишина") == 0

    def test_a_cancelled_loop_stops_inside_its_body(self, engine, registry) -> None:
        """Every block still open is recorded as cancelled, not as broken."""
        run = engine.start(
            command(
                {
                    "type": "While",
                    "params": {"condition": "1 == 1"},
                    "body": [block("Tick"), block("Wait", ms=FOREVER_MS)],
                }
            )
        )
        assert registry.started.wait(GIVE_UP_S)
        engine.cancel(run.run_id, reason="стоп")

        report = run.wait(GIVE_UP_S)
        assert report.cancelled
        assert report.step("actions[0].body[1]").status is StepStatus.CANCELLED
        assert report.step("actions[0]").status is StepStatus.CANCELLED
        assert report.failures == ()


class TestCooldownAndRefusal:
    """A command that does not run: the answer the user gets instead of silence."""

    def test_a_second_press_inside_the_cooldown_does_not_run_and_says_why(
        self, engine, registry, heard, caplog
    ) -> None:
        hot = command(block("Run"), cooldown_ms=FOREVER_MS)

        assert engine.run(hot).ok
        with caplog.at_level(logging.INFO, logger="ayris.actions.macros.engine"):
            refused = engine.run(hot)

        assert refused.outcome is RunOutcome.COOLDOWN
        assert not refused.ran
        assert registry.names == ["Run"]
        assert "не остыла" in refused.user_message
        assert refused.execution is ExecutionResult.DENIED
        assert "не запущен" in caplog.text
        skipped = heard.one(MacroSkipped)
        assert skipped.reason == "cooldown"
        assert 0 < skipped.retry_after_ms <= FOREVER_MS

    def test_the_cooldown_ends_when_its_milliseconds_have_passed(self, engines, registry) -> None:
        """A fake clock, because a test that waits out a cooldown is a test that is slow."""
        now = [1000.0]
        engine = engines(clock=lambda: now[0])
        hot = command(block("Run"), cooldown_ms=500)

        assert engine.run(hot).ok
        assert engine.run(hot).outcome is RunOutcome.COOLDOWN
        now[0] += 0.6
        assert engine.run(hot).ok

        assert registry.names == ["Run", "Run"]

    def test_a_refused_run_does_not_push_the_cooldown_forward(self, engines) -> None:
        """Otherwise a hotkey held down would keep a command cold for ever."""
        now = [0.0]
        engine = engines(clock=lambda: now[0])
        hot = command(block("Run"), cooldown_ms=1000)

        assert engine.run(hot).ok
        now[0] += 0.5
        assert engine.run(hot).outcome is RunOutcome.COOLDOWN
        now[0] += 0.6
        assert engine.run(hot).ok

    def test_reset_cooldowns_lets_the_command_run_at_once(self, engine, registry) -> None:
        hot = command(block("Run"), cooldown_ms=FOREVER_MS, name="Горячая")

        assert engine.run(hot).ok
        engine.reset_cooldowns(hot)

        assert engine.run(hot).ok
        assert registry.names == ["Run", "Run"]

    def test_a_switched_off_command_is_refused_before_anything_runs(
        self, engine, registry, heard
    ) -> None:
        report = engine.run(command(block("Run"), enabled=False))

        assert report.outcome is RunOutcome.DISABLED
        assert report.user_message == "Команда выключена."
        assert report.execution is ExecutionResult.DENIED
        assert registry.calls == []
        assert heard.one(MacroSkipped).reason == "disabled"
        assert heard.of(MacroStarted) == []

    def test_two_presses_in_the_same_instant_let_exactly_one_through(self, engine) -> None:
        """Asking and marking happen under one lock, so the race has one winner.

        Four threads press the same hotkey at once. If the cooldown were read before it
        were written, more than one of them would be told to go ahead.
        """
        hot = command(block("Run"), cooldown_ms=FOREVER_MS, name="Одна на всех")
        outcomes: list[RunOutcome] = []
        ready = threading.Barrier(4)
        lock = threading.Lock()

        def press() -> None:
            ready.wait(GIVE_UP_S)
            report = engine.run(hot, timeout=GIVE_UP_S)
            with lock:
                outcomes.append(report.outcome)

        threads = [threading.Thread(target=press, name=f"press-{n}") for n in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(GIVE_UP_S)

        assert outcomes.count(RunOutcome.SUCCESS) == 1
        assert outcomes.count(RunOutcome.COOLDOWN) == 3


class TestConcurrency:
    """Two triggers at once: side by side, in a queue, or one cancelling the other."""

    def test_parallel_runs_both_commands_at_the_same_time(self, engine, registry) -> None:
        registry.hold["Held"] = threading.Event()
        first = engine.start(command(block("Held"), name="Первая"))
        second = engine.start(command(block("Other"), name="Вторая"))

        assert until(lambda: sorted(registry.names) == ["Held", "Other"])
        assert first.running
        registry.hold["Held"].set()

        assert first.wait(GIVE_UP_S).ok
        assert second.wait(GIVE_UP_S).ok

    def test_queue_runs_one_command_at_a_time(self, engines, registry) -> None:
        engine = engines(policy=ConcurrencyPolicy.QUEUE)
        registry.hold["Held"] = threading.Event()
        first = engine.start(command(block("Held"), name="Первая"))
        assert registry.started.wait(GIVE_UP_S)
        second = engine.start(command(block("Second"), name="Вторая"))

        assert registry.names == ["Held"], "вторая команда не должна была начаться"
        registry.hold["Held"].set()

        assert first.wait(GIVE_UP_S).ok
        assert second.wait(GIVE_UP_S).ok
        assert registry.names == ["Held", "Second"]

    def test_preempt_cancels_the_command_it_outranks(self, engines, registry) -> None:
        """Section 7's answer to two triggers at once: the important one does not queue."""
        engine = engines(policy=ConcurrencyPolicy.PREEMPT)
        slow = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS), name="Обычная"))
        assert registry.started.wait(GIVE_UP_S)

        urgent = engine.start(command(block("Halt"), name="Экстренная", priority=100))

        assert urgent.wait(GIVE_UP_S).ok
        stopped = slow.wait(GIVE_UP_S)
        assert stopped.cancelled
        assert isinstance(stopped.error, MacroCancelledError)
        assert "важнее" in stopped.error.reason

    def test_preempt_waits_its_turn_when_it_outranks_nobody(self, engines, registry) -> None:
        engine = engines(policy=ConcurrencyPolicy.PREEMPT)
        registry.hold["Held"] = threading.Event()
        first = engine.start(command(block("Held"), name="Первая", priority=50))
        assert registry.started.wait(GIVE_UP_S)
        second = engine.start(command(block("Second"), name="Вторая", priority=50))

        assert registry.names == ["Held"]
        registry.hold["Held"].set()

        assert first.wait(GIVE_UP_S).ok
        assert second.wait(GIVE_UP_S).ok


class TestFailurePolicy:
    """A block that breaks: what it does to the blocks after it."""

    def test_continue_keeps_the_chain_going_and_keeps_the_failure_in_the_report(
        self, engine, registry, heard
    ) -> None:
        registry.raises["Boom"] = RuntimeError("экран не отвечает")

        report = engine.run(command({"type": "Boom", "on_error": "continue"}, block("Next")))

        assert report.ok
        assert registry.names == ["Boom", "Next"]
        assert [step.path for step in report.failures] == ["actions[0]"]
        assert "экран не отвечает" in report.step("actions[0]").error
        assert isinstance(heard.one(MacroEnded), MacroFinished)

    def test_stop_ends_the_run_at_the_block_that_broke(self, engine, registry, heard) -> None:
        registry.raises["Boom"] = RuntimeError("экран не отвечает")

        report = engine.run(command(block("Boom"), block("Never")))

        assert report.outcome is RunOutcome.FAILED
        assert registry.names == ["Boom"]
        assert isinstance(report.error, MacroBlockError)
        assert report.error.path == "actions[0]"
        assert report.error.block == "Boom"
        failed = heard.one(MacroFailed)
        assert failed.path == "actions[0]"
        assert failed.user_message == report.user_message

    def test_an_action_that_refuses_stops_the_chain_with_its_own_russian_message(
        self, engine, registry
    ) -> None:
        """A refusal is not an exception, and the user still has to hear about it."""
        registry.results["Deny"] = ActionResult.failed("Микрофон занят.", detail="device busy")

        report = engine.run(command(block("Deny"), block("Never")))

        assert report.outcome is RunOutcome.FAILED
        assert report.user_message == "Микрофон занят."
        assert registry.names == ["Deny"]
        assert isinstance(report.error, MacroBlockError)
        assert "device busy" in str(report.error)

    def test_an_action_nobody_registered_names_itself(self, engine, registry) -> None:
        registry.unknown.add("Ghost")

        report = engine.run(command(block("Ghost")))

        assert report.outcome is RunOutcome.FAILED
        assert report.user_message == "Действие «Ghost» не подключено."
        assert registry.calls == []

    def test_try_runs_its_catch_with_the_failure_at_hand(self, engine, registry) -> None:
        registry.raises["Boom"] = RuntimeError("нет доступа")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "params": {"error_var": "беда"},
                    "body": [block("Boom")],
                    "catch": [block("Heal", note="{беда}")],
                },
                block("After"),
            )
        )

        assert report.ok
        assert registry.names == ["Boom", "Heal", "After"]
        assert "нет доступа" in registry.params("Heal")["note"]
        assert "поймано" in report.step("actions[0]").message

    def test_try_without_a_catch_simply_swallows(self, engine, registry) -> None:
        registry.raises["Boom"] = RuntimeError("нет доступа")

        report = engine.run(
            command({"type": "Try", "body": [block("Boom"), block("Never")]}, block("After"))
        )

        assert report.ok
        assert registry.names == ["Boom", "After"]

    def test_try_prefers_the_russian_line_when_the_failure_has_one(self, engine, registry) -> None:
        """An action's own typed error already speaks Russian; that is what the ``Catch`` gets."""
        registry.raises["Boom"] = ActionUnavailable("no such monitor")

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [block("Boom")],
                    "catch": [block("Heal", note="{error}")],
                }
            )
        )

        assert report.ok
        assert registry.params("Heal")["note"] == "Это действие сейчас недоступно."

    def test_try_does_not_swallow_the_stop_word(self, engine, registry) -> None:
        """A ``Try`` that caught a cancellation would turn the stop word into a suggestion."""
        run = engine.start(
            command(
                {
                    "type": "Try",
                    "body": [block("Ping"), block("Wait", ms=FOREVER_MS)],
                    "catch": [block("Never")],
                }
            )
        )
        assert registry.started.wait(GIVE_UP_S)
        engine.cancel(run.run_id, reason="стоп")

        report = run.wait(GIVE_UP_S)
        assert report.cancelled
        assert "Never" not in registry.names

    def test_try_does_not_swallow_a_limit(self, engines, registry) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=2))

        report = engine.run(
            command(
                {
                    "type": "Try",
                    "body": [
                        {
                            "type": "While",
                            "params": {"condition": "1 == 1"},
                            "body": [block("Tick")],
                        }
                    ],
                    "catch": [block("Never")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert registry.names == ["Tick", "Tick"]


class TestLimits:
    """The ceilings: what stops a command that would otherwise never end."""

    def test_an_endless_while_is_stopped_by_the_iteration_limit(self, engines, registry) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=4))

        report = engine.run(
            command({"type": "While", "params": {"condition": "1 == 1"}, "body": [block("Tick")]})
        )

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert report.error.limit == "iterations"
        assert registry.names == ["Tick"] * 4
        assert report.user_message == "Цикл в команде повторился 4 раз и был остановлен."

    def test_a_while_block_may_lower_the_ceiling_but_not_raise_it(self, engines, registry) -> None:
        engine = engines(limits=ExecutionLimits(max_iterations=3))

        report = engine.run(
            command(
                {
                    "type": "While",
                    "params": {"condition": "1 == 1", "max_iterations": 100},
                    "body": [block("Tick")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert registry.names == ["Tick"] * 3

    def test_a_for_over_a_huge_range_costs_a_short_list(self, engines, registry) -> None:
        """``from: 1`` ``to: 1000000`` is a typo, not a gigabyte of integers."""
        engine = engines(limits=ExecutionLimits(max_iterations=5))

        report = engine.run(
            command(
                {
                    "type": "For",
                    "params": {"var": "n", "from": 1, "to": 1_000_000},
                    "body": [block("Tick")],
                }
            )
        )

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert len(registry.names) == 5

    def test_blocks_nested_deeper_than_allowed_stop_the_run(self, engines, registry) -> None:
        engine = engines(limits=ExecutionLimits(max_depth=1))
        deep = {
            "type": "If",
            "params": {"condition": "1 == 1"},
            "then": [
                {
                    "type": "If",
                    "params": {"condition": "1 == 1"},
                    "then": [block("Deep")],
                }
            ],
        }

        report = engine.run(command(deep))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert report.error.limit == "depth"
        assert report.user_message == "Блоки команды вложены слишком глубоко."
        assert registry.calls == []

    def test_the_step_budget_stops_a_command_that_does_too_much(self, engines, registry) -> None:
        engine = engines(limits=ExecutionLimits(max_steps=3))

        report = engine.run(command(*[block(f"Step{n}") for n in range(6)]))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert report.error.limit == "steps"
        assert len(report.steps) == 3
        assert registry.names == ["Step0", "Step1", "Step2"]

    @pytest.mark.slow
    def test_the_whole_run_has_a_budget_and_a_wait_is_cut_down_to_it(self, engines, heard) -> None:
        """A ``Wait`` cannot outlive the timeout around it: the pause is what is left."""
        engine = engines(limits=ExecutionLimits(timeout_s=0.05))

        started = time.perf_counter()
        report = engine.run(command(block("Wait", ms=FOREVER_MS)))
        took_ms = (time.perf_counter() - started) * 1000

        assert report.outcome is RunOutcome.TIMEOUT
        assert took_ms < 1000, f"пауза заняла {took_ms:.0f} мс"
        assert report.execution is ExecutionResult.TIMEOUT
        assert isinstance(heard.one(MacroEnded), MacroFailed)

    def test_one_pause_longer_than_a_pause_may_be_is_refused_at_once(self, engines) -> None:
        engine = engines(limits=ExecutionLimits(max_wait_ms=100))

        report = engine.run(command(block("Wait", ms=FOREVER_MS)))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert report.error.limit == "wait_ms"
        assert report.user_message == "Пауза в команде слишком длинная."

    def test_a_run_without_a_timeout_is_allowed_to_take_its_time(self, engines) -> None:
        """Zero means no limit: a command driving an install takes minutes on purpose."""
        engine = engines(limits=ExecutionLimits(timeout_s=0))

        report = engine.run(command(block("Wait", ms=10)))

        assert report.ok
        assert engine.limits.budget_ms == 0


class TestCalls:
    """One command calling another: inline, or as a run of its own."""

    def test_a_called_command_runs_here_and_its_value_becomes_last_result(
        self, engines, registry, heard
    ) -> None:
        table = {
            "Вложенная": command(block("Inner"), block("Return", value="итог"), name="Вложенная")
        }
        engine = engines(library=table.get)
        caller = command(
            {"type": "CallCommand", "params": {"command": "Вложенная"}},
            block("SetVar", name="seen", value="{last_result}", scope="global"),
            name="Внешняя",
        )

        report = engine.run(caller, request_id="req-7")

        assert report.ok
        assert registry.names == ["Inner"]
        assert engine.variables.read(VariableScope.GLOBAL, "seen") == "итог"
        assert "«Вложенная»" in report.step("actions[0]").message
        started = heard.of(MacroStarted)
        assert [event.command for event in started] == ["Внешняя", "Вложенная"]
        assert started[1].trigger == "call"
        assert started[1].request_id == "req-7"
        assert len(heard.of(MacroFinished)) == 2

    def test_the_arguments_of_a_call_arrive_as_the_slots_of_the_called_command(
        self, engines, registry
    ) -> None:
        """A command reads ``{name}`` the same way whether a phrase filled it or a caller."""
        table = {"Приветствие": command(block("Say", text="привет, {who}"), name="Приветствие")}
        engine = engines(library=table.get)

        report = engine.run(
            command(
                {
                    "type": "CallCommand",
                    "params": {"command": "Приветствие", "args": {"who": "Артём"}},
                }
            )
        )

        assert report.ok
        assert registry.params("Say")["text"] == "привет, Артём"

    def test_a_command_that_does_not_exist_is_a_failure_that_names_it(self, engines) -> None:
        """The report carries a block error that knows the path; the cause names the call."""
        engine = engines(library={}.get)

        report = engine.run(command({"type": "CallCommand", "params": {"command": "Пропала"}}))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroBlockError)
        assert report.error.path == "actions[0]"
        assert isinstance(report.error.cause, MacroCallError)
        assert report.user_message == "Не удалось вызвать команду «Пропала»."

    def test_a_switched_off_command_cannot_be_called(self, engines, registry) -> None:
        table = {"Спит": command(block("Inner"), name="Спит", enabled=False)}
        engine = engines(library=table.get)

        report = engine.run(command({"type": "CallCommand", "params": {"command": "Спит"}}))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error.cause, MacroCallError)
        assert report.user_message == "Не удалось вызвать команду «Спит»."
        assert registry.calls == []

    def test_a_failure_inside_a_called_command_keeps_pointing_at_the_block_that_broke(
        self, engines, registry
    ) -> None:
        table = {"Хрупкая": command(block("Ok"), block("Boom"), name="Хрупкая")}
        engine = engines(library=table.get)
        registry.raises["Boom"] = RuntimeError("сломалось внутри")

        report = engine.run(command({"type": "CallCommand", "params": {"command": "Хрупкая"}}))

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroBlockError)
        assert report.error.path == "actions[1]"
        assert report.error.block == "Boom"

    def test_wait_false_starts_the_called_command_as_a_run_of_its_own(
        self, engines, registry
    ) -> None:
        table = {"Отдельная": command(block("Detached"), name="Отдельная")}
        engine = engines(library=table.get)

        report = engine.run(
            command(
                {
                    "type": "CallCommand",
                    "params": {"command": "Отдельная", "wait": False},
                }
            )
        )

        assert report.ok
        assert "запущена отдельно" in report.step("actions[0]").message
        assert until(lambda: "Detached" in registry.names)

    def test_a_command_calling_itself_is_stopped_by_the_call_depth(self, engines, registry) -> None:
        recursive = command(
            block("Tick"),
            {"type": "CallCommand", "params": {"command": "Рекурсия"}},
            name="Рекурсия",
        )
        engine = engines(
            library={"Рекурсия": recursive}.get, limits=ExecutionLimits(max_call_depth=3)
        )

        report = engine.run(recursive)

        assert report.outcome is RunOutcome.FAILED
        assert isinstance(report.error, MacroLimitError)
        assert report.error.limit == "call_depth"
        assert report.user_message == "Команды вызывают друг друга слишком глубоко."
        assert len(registry.names) == 4

    def test_a_stop_reaches_the_command_called_inside_another(
        self, engines, registry, heard
    ) -> None:
        """One cancel, two runs: a nested call shares the cancel event of its caller."""
        table = {"Долгая": command(block("Slow"), block("Wait", ms=FOREVER_MS), name="Долгая")}
        engine = engines(library=table.get)
        run = engine.start(command({"type": "CallCommand", "params": {"command": "Долгая"}}))
        assert registry.started.wait(GIVE_UP_S)

        assert engine.cancel(run.run_id, reason="стоп") == 1

        report = run.wait(GIVE_UP_S)
        assert report.cancelled
        assert report.error.reason == "стоп"
        assert report.step("actions[0]").status is StepStatus.CANCELLED
        # Two commands ran, so the history sees two: the caller and the one it called.
        assert [event.command for event in heard.of(MacroStarted)] == ["Проверка", "Долгая"]
        assert [event.command for event in heard.of(MacroCancelled)] == ["Долгая", "Проверка"]


class TestReport:
    """The report: the timings, the order, and the text every reader of a run gets."""

    def test_a_step_is_timed_and_the_run_contains_it(self, engine, registry) -> None:
        registry.delays["Slow"] = 0.05

        report = engine.run(command(block("Quick"), block("Slow")))

        quick, slow = report.steps
        assert slow.duration_ms >= 40
        assert quick.duration_ms <= slow.duration_ms
        assert report.duration_ms >= slow.duration_ms

    def test_offsets_grow_along_the_run_and_start_at_its_beginning(self, engine, registry) -> None:
        """``offset_ms`` is what a timeline is drawn from, so it counts from the run's start."""
        registry.delays["Middle"] = 0.03

        report = engine.run(command(block("First"), block("Middle"), block("Last")))

        offsets = [step.offset_ms for step in report.steps]
        assert offsets == sorted(offsets)
        assert offsets[0] < 20
        assert offsets[2] >= 25
        assert all(
            step.offset_ms + step.duration_ms <= report.duration_ms + 5 for step in report.steps
        )

    def test_a_loop_contains_the_time_of_its_body(self, engine, registry) -> None:
        """Timings nest rather than tile: three turns of 20 ms are inside the ``While``.

        Sum the steps of a report with a loop in it and the total passes the run's own
        duration, which is the arithmetic proof that a parent's time contains its children's
        instead of sitting next to it.
        """
        registry.delays["Tick"] = 0.02
        registry.answers["Tick"] = [
            ActionResult.done(value="ещё"),
            ActionResult.done(value="ещё"),
            ActionResult.done(value="хватит"),
        ]
        report = engine.run(
            command(
                {
                    "type": "While",
                    "params": {"condition": '{last_result} != "хватит"'},
                    "body": [block("Tick")],
                }
            )
        )

        assert report.ok
        loop = report.step("actions[0]")
        turns = [step for step in report.steps if step.path == "actions[0].body[0]"]
        assert len(turns) == 3
        assert loop.duration_ms >= 3 * 15
        assert all(turn.duration_ms < loop.duration_ms for turn in turns)
        assert sum(step.duration_ms for step in report.steps) > report.duration_ms

    def test_a_line_of_the_report_shows_where_what_and_how_long(self, engine) -> None:
        """One line per block: the indent is the nesting, and the tail is what it did."""
        report = engine.run(
            command({"type": "If", "params": {"condition": "1 == 1"}, "then": [block("Deep")]})
        )

        inner = report.step("actions[0].then[0]")
        assert inner.depth == 1
        assert inner.as_line() == f"  actions[0].then[0] Deep [ok] {inner.duration_ms} мс — готово"
        lines = report.as_text().splitlines()
        assert lines[0] == f"Проверка: success за {report.duration_ms} мс"
        assert lines[1:] == [step.as_line() for step in report.steps]

    def test_a_path_that_never_ran_has_no_step(self, engine) -> None:
        report = engine.run(command(block("Only")))

        assert report.step("actions[0]").ok
        assert report.step("actions[7]") is None

    def test_raise_for_status_raises_a_failure_and_stays_quiet_otherwise(
        self, engine, registry
    ) -> None:
        """A caller who would rather have an exception gets one — for a failure only.

        Cancelled and refused are outcomes to read, not failures to raise: a stop word is
        not an error, and neither is a command pressed twice inside its cooldown.
        """
        registry.raises["Boom"] = RuntimeError("сломалось")

        engine.run(command(block("Ok"))).raise_for_status()
        engine.run(command(block("Run"), enabled=False)).raise_for_status()
        stopped = engine.start(command(block("Wait", ms=FOREVER_MS)))
        engine.cancel(stopped.run_id, reason="стоп")
        cancelled = stopped.wait(GIVE_UP_S)
        assert cancelled.cancelled
        cancelled.raise_for_status()

        with pytest.raises(MacroBlockError):
            engine.run(command(block("Boom"))).raise_for_status()

    def test_the_slowest_step_is_what_the_timeline_points_at(self, engine, registry) -> None:
        registry.delays["Sluggish"] = 0.05

        report = engine.run(command(block("Quick"), block("Sluggish"), block("Brisk")))

        assert report.slowest.block == "Sluggish"
        assert report.slowest.duration_ms >= 40

    def test_a_run_that_never_started_still_has_a_report(self, engine) -> None:
        """The refusal is the whole answer: one outcome, one line, and no steps."""
        report = engine.run(command(block("Run"), enabled=False))

        assert not report.ran
        assert report.steps == ()
        assert report.slowest is None
        assert report.failures == ()
        assert report.as_text() == f"Проверка: disabled за {report.duration_ms} мс"

    def test_the_report_says_which_run_of_which_command_this_was(self, engine) -> None:
        """What task 33 writes into ``history``: the run, the command, and who asked."""
        run = engine.start(
            command(block("Ping"), name="Свет"), trigger=TriggerSource.HOTKEY, request_id="req-7"
        )

        report = run.wait(GIVE_UP_S)

        assert report.run_id == run.run_id
        assert report.command == "Свет"
        assert report.command_id is None
        assert report.trigger == "hotkey"
        assert report.request_id == "req-7"
        assert report.started_at.tzinfo is not None
        assert report.ran
        assert report.failures == ()


class TestEngineLifecycle:
    """What the engine holds while a command runs, and how it lets everything go."""

    def test_active_holds_a_run_while_it_runs_and_lets_it_go_after(self, engine, registry) -> None:
        """The gate needs this list, and so does the panic hotkey: it is the only handle."""
        registry.hold["Held"] = threading.Event()
        run = engine.start(command(block("Held")))
        assert registry.started.wait(GIVE_UP_S)

        assert [waiting.run_id for waiting in engine.active] == [run.run_id]
        assert run.running

        registry.hold["Held"].set()
        assert run.wait(GIVE_UP_S).ok
        assert until(lambda: engine.active == ())
        assert not run.running

    def test_the_engine_refuses_new_runs_after_shutdown(self, engines) -> None:
        """Said out loud, because a command silently dropped on the way out looks like a bug."""
        engine = engines()
        assert engine.running

        engine.shutdown()

        assert not engine.running
        with pytest.raises(MacroEngineStoppedError):
            engine.start(command(block("Late")))
        with pytest.raises(MacroEngineStoppedError):
            engine.run(command(block("Late")))

    def test_shutdown_twice_changes_nothing(self, engines, registry) -> None:
        """The application closing calls it, and so does the fixture that made the engine."""
        engine = engines()
        assert engine.run(command(block("Ping"))).ok

        engine.shutdown()
        engine.shutdown()

        assert registry.names == ["Ping"]

    def test_shutdown_stops_a_command_that_is_still_running(self, engines, registry) -> None:
        """Closing the application does not sit through a five second ``Wait``."""
        engine = engines()
        run = engine.start(command(block("Ping"), block("Wait", ms=FOREVER_MS)))
        assert registry.started.wait(GIVE_UP_S)

        engine.shutdown(reason="выход")

        report = run.wait(GIVE_UP_S)
        assert report.cancelled
        assert report.error.reason == "выход"

    def test_the_engine_is_a_context_manager(self, registry, bus) -> None:
        """A test that forgets to shut an engine down leaks its threads into the next one."""
        with MacroEngine(registry, bus=bus) as engine:
            assert engine.run(command(block("Inside"))).ok

        assert not engine.running
        assert engine.active == ()

    def test_find_looks_only_in_the_library_it_was_given(self, engines) -> None:
        table = {"Есть": command(block("Ok"), name="Есть")}

        assert engines(library=table.get).find("Есть").name == "Есть"
        assert engines(library=table.get).find("Нету") is None
        assert engines().find("Есть") is None

    def test_the_engine_says_what_it_was_configured_with(self, engines) -> None:
        limits = ExecutionLimits(max_iterations=7)

        engine = engines(limits=limits, policy=ConcurrencyPolicy.QUEUE)

        assert engine.limits is limits
        assert engine.policy is ConcurrencyPolicy.QUEUE

    def test_a_shared_store_outlives_the_engine_that_wrote_to_it(self, registry, bus) -> None:
        """``profile`` and ``global`` belong to the session, not to one engine."""
        store = MemoryVariables()

        with MacroEngine(registry, bus=bus, store=store) as first:
            assert first.variables is store
            assert first.run(command(block("SetVar", name="kept", value="да", scope="global"))).ok
        with MacroEngine(registry, bus=bus, store=store) as second:
            assert second.variables.read(VariableScope.GLOBAL, "kept") == "да"
