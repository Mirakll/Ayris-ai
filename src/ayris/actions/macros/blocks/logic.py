"""Control flow: the blocks that decide what runs next, and the interface they run behind.

Section 7.1 lists eleven blocks that are not actions. ``If`` chooses a branch, ``While``
repeats one, ``Break`` leaves it, ``Return`` ends the command with a value, ``Try`` catches
what broke, ``CallCommand`` runs another command as a subroutine. None of them touches the
machine; all of them change where the walk goes next.

**How they say it.** By returning a :class:`Flow`, not by raising. ``Break`` inside two
nested loops has to leave one of them, and the loop that catches it is the one directly
above — which is exactly what "the caller inspects the return value" gives for free. A
private exception would read better in the walk itself, but every exception class in this
project is named ``...Error`` (ruff N818, and it is right: ``_Break`` in a traceback reads
like a failure), and a control-flow signal that has to be called an error to satisfy a
linter is worse than a value.

**Why they are functions and not methods.** The interpreter of task 31 owns a walk: where
it is, what it has recorded, how much of the budget is left. A block needs to reach into
that walk — ``While`` runs its body, ``Wait`` sleeps on the run's cancel event — but it
needs a named handful of things and not the interpreter's insides. :class:`BlockRuntime` is
that handful. The interpreter satisfies it structurally, a test can satisfy it with thirty
lines, and a new block is a function in this file plus a line in
:data:`LOGIC_HANDLERS` — which is the whole point of splitting these out of the engine.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol

from ayris.actions.macros.errors import (
    MacroBlockError,
    MacroCancelledError,
    MacroLimitError,
    MacroTimeoutError,
)
from ayris.actions.macros.expressions import format_value
from ayris.actions.macros.expressions import truthy as _truthy
from ayris.actions.macros.report import StepStatus
from ayris.core.errors import AyrisError
from ayris.core.models import VariableScope

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ayris.actions.macros.context import ExecutionContext
    from ayris.actions.macros.engine import ExecutionLimits
    from ayris.actions.macros.report import ReportBuilder
    from ayris.actions.macros.schema import ActionBlock

__all__ = [
    "LOGIC_HANDLERS",
    "BlockHandler",
    "BlockRuntime",
    "Flow",
    "as_dict",
    "as_int",
    "as_list",
    "error_text",
    "matches",
    "run_break",
    "run_call",
    "run_case",
    "run_continue",
    "run_for",
    "run_if",
    "run_return",
    "run_switch",
    "run_try",
    "run_wait",
    "run_while",
    "short",
]


class Flow(StrEnum):
    """What a block tells the walk above it to do next.

    ``NEXT`` is "go on", and it is what every block that is not one of the four flow
    blocks returns. The other three travel up the walk until something catches them: a
    loop catches ``BREAK`` and ``CONTINUE``, the run itself catches ``RETURN``.
    """

    NEXT = "next"
    BREAK = "break"
    CONTINUE = "continue"
    RETURN = "return"


class BlockRuntime(Protocol):
    """The part of the interpreter a block is allowed to see.

    Five things, and each one is here because a block in this file cannot be written
    without it: the variables to read (:attr:`context`), the place to write down what
    happened (:attr:`report`), the ceilings (:attr:`limits`), a way to run a body
    (:meth:`walk` and :meth:`run_one`), and the two things only the engine can do —
    sleep on the run's cancel event (:meth:`pause`) and start another command
    (:meth:`call`).

    What is deliberately not here: the thread pool, the event bus, the cooldowns, the
    admission policy. A block cannot cancel a run, cannot publish an event and cannot
    reach the engine that owns four other runs, because none of that is a block's
    business.
    """

    @property
    def context(self) -> ExecutionContext:
        """The variables, slots and expression evaluator of this run."""
        ...

    @property
    def report(self) -> ReportBuilder:
        """Where a block writes down what it did, one record per block."""
        ...

    @property
    def limits(self) -> ExecutionLimits:
        """The ceilings this run may not cross."""
        ...

    @property
    def reason(self) -> str:
        """Why the run was stopped, when it was. What a cancellation message says."""
        ...

    def walk(self, blocks: Iterable[ActionBlock], prefix: str, depth: int) -> Flow:
        """Run a list of blocks in order, stopping at the first one that redirects."""
        ...

    def run_one(self, block: ActionBlock, path: str, depth: int) -> Flow:
        """Run one block, with everything :meth:`walk` does around it."""
        ...

    def check(self) -> None:
        """Ask whether the run may go on: not cancelled, not out of steps or of time."""
        ...

    def pause(self, ms: int) -> bool:
        """Sleep up to ``ms``, waking early and answering ``True`` if the run was stopped."""
        ...

    def call(self, name: str, args: Mapping[str, Any], *, wait: bool) -> None:
        """Run another command: inline when ``wait``, otherwise as a run of its own.

        Everything about starting a run belongs to the engine — the pool, the admission
        policy, the call depth, what is left of the budget — so a block asks for a command
        by name and does not learn how it was made to happen.
        """
        ...

    def returned(self, value: Any) -> None:
        """Remember the value a ``Return`` block leaves for whoever called the command."""
        ...


#: One block's implementation: the runtime, the block, where it sits in the tree, and how
#: deep. The path and the depth are passed rather than tracked because a block that runs a
#: body has to build the path of that body — ``actions[1].then[0]`` is what the editor
#: highlights and what the report keeps.
BlockHandler = Callable[["BlockRuntime", "ActionBlock", str, int], Flow]


def short(value: object, limit: int = 60) -> str:
    """A value as one short piece of text, for a step's message and for a log line."""
    text = format_value(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def as_int(value: Any, default: int = 0) -> int:
    """A block parameter as a whole number: ``50``, ``"50"``, ``50.0``, or nothing at all.

    Raises:
        MacroBlockError: the parameter is there and is not a number.
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return int(value)
    try:
        return int(float(format_value(value).strip().replace(",", ".")))
    except ValueError as exc:
        raise MacroBlockError(
            f"expected a whole number, got {value!r}",
            user_message=f"Ожидалось число, а не «{short(value)}».",
            cause=exc,
        ) from exc


def as_list(value: Any) -> list[Any]:
    """What a ``For`` walks when it was given ``items``.

    A list is itself, a mapping is its keys, and text is either the JSON it looks like or
    the comma-separated line a person types into the editor.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return list(value)
    if isinstance(value, str):
        return split_items(value)
    return [] if value is None else [value]


def split_items(text: str) -> list[Any]:
    """A line of items as a list: JSON when it is JSON, comma-separated otherwise."""
    stripped = text.strip()
    if not stripped:
        return []
    if not stripped.startswith("["):
        return [part.strip() for part in stripped.split(",") if part.strip()]
    try:
        loaded = json.loads(stripped)
    except ValueError as exc:
        raise MacroBlockError(f"items is not a list: {stripped!r}", cause=exc) from exc
    return loaded if isinstance(loaded, list) else [loaded]


def as_dict(value: Any) -> dict[str, Any]:
    """A ``CallCommand`` argument bundle as a dictionary of names to values.

    A mapping is taken as it is and anything else — the absent parameter included — as no
    arguments at all: a called command reads what it was given the way it reads slots, and
    a slot set has names.
    """
    if isinstance(value, Mapping):
        return {format_value(key): item for key, item in value.items()}
    return {}


def matches(left: Any, right: Any) -> bool:
    """Whether a ``Case`` value answers a ``Switch`` value.

    Compared as values first and as text second, so ``50`` from a slot matches the ``"50"``
    a file spells, which is the same leniency
    :func:`~ayris.actions.macros.expressions.substitute` gives everywhere else.
    """
    if left == right:
        return True
    return format_value(left) == format_value(right)


def error_text(exc: Exception) -> str:
    """What a ``Try`` writes into its error variable: the Russian line where there is one.

    Where there is not, it is the technical text and not the polite default: a ``Catch``
    branch that shows «Шаг команды не выполнился» tells the person who wrote the command
    nothing at all, while «нет доступа» tells them what to fix.
    """
    if isinstance(exc, MacroBlockError):
        if isinstance(exc.cause, AyrisError):
            return exc.cause.user_message
        if exc.cause is not None:
            return exc.technical
    if isinstance(exc, AyrisError):
        return exc.user_message
    return str(exc) or type(exc).__name__


def run_if(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``If``: evaluate the condition, run one branch, and record which one it was."""
    taken = rt.context.truth(block.params.get("condition", ""))
    name = "then" if taken else "else"
    rt.report.note(name)
    branch = block.then if taken else block.else_
    if not branch:
        return Flow.NEXT
    return rt.walk(branch, f"{path}.{name}", depth + 1)


def run_switch(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``Switch``: run the first ``Case`` that matches, or ``Default``.

    Every arm that is not chosen is recorded as skipped rather than left out of the report:
    the debugger shows that the branch was considered and passed over, which is exactly the
    question a user asks about a ``Switch`` that took the wrong turn. That is why the loop
    runs to the end instead of stopping on the chosen arm — the arms below it are as much a
    part of the answer as the arms above.
    """
    value = rt.context.fill(block.params.get("value", ""))
    chosen = _arm(rt, block, value)
    rt.report.note(f"body[{chosen}]" if chosen >= 0 else "ни одна ветка не подошла")
    flow = Flow.NEXT
    for index, arm in enumerate(block.body):
        if index == chosen:
            flow = rt.run_one(arm, f"{path}.body[{index}]", depth + 1)
            continue
        rt.report.mark(f"{path}.body[{index}]", arm.type, StepStatus.SKIPPED, depth=depth + 1)
    return flow


def _arm(rt: BlockRuntime, block: ActionBlock, value: Any) -> int:
    """Which arm of a ``Switch`` answers ``value``: a ``Case``, a ``Default``, or none."""
    arms = tuple((index, arm) for index, arm in enumerate(block.body) if arm.enabled)
    for index, arm in arms:
        if arm.type == "Default":
            continue
        if matches(rt.context.fill(arm.params.get("value")), value):
            return index
    return next((index for index, arm in arms if arm.type == "Default"), -1)


def run_case(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``Case`` and ``Default``: their body, once, when the ``Switch`` chose them."""
    return rt.walk(block.body, f"{path}.body", depth + 1)


def run_while(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``While``: the body while the condition holds, and never more than the limit.

    The limit is checked before the body and not after, so the error names the turn that
    was refused. A block may lower the ceiling for itself but not raise it: a hand-written
    ``max_iterations: 100000`` is the exact case the ceiling exists for.

    Raises:
        MacroLimitError: the condition was still true after the last allowed turn.
    """
    ceiling = rt.limits.max_iterations
    asked = as_int(rt.context.fill(block.params.get("max_iterations")), ceiling)
    limit = min(asked, ceiling)
    condition = block.params.get("condition", "")
    turns = 0
    while rt.context.truth(condition):
        if turns >= limit:
            rt.report.note(f"{turns} итераций, предел")
            raise MacroLimitError(
                "iterations",
                limit,
                user_message=f"Цикл в команде повторился {limit} раз и был остановлен.",
            )
        turns += 1
        flow = rt.walk(block.body, f"{path}.body", depth + 1)
        if flow is Flow.BREAK:
            break
        if flow is Flow.RETURN:
            rt.report.note(f"{turns} итераций, выход")
            return flow
        rt.check()
    rt.report.note(f"{turns} итераций")
    return Flow.NEXT


def run_for(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``For``: the body once per item, with the loop variable set on every turn.

    The variable is an ordinary local, so a block after the loop still reads what the last
    turn left in it — which is what a command counting attempts expects.

    Raises:
        MacroLimitError: the sequence is longer than the run may iterate.
    """
    name = format_value(rt.context.fill(block.params.get("var", "item")))
    values = sequence(rt, block)
    turns = 0
    for value in values:
        if turns >= rt.limits.max_iterations:
            raise MacroLimitError(
                "iterations",
                rt.limits.max_iterations,
                user_message="Перебор в команде оказался слишком длинным.",
            )
        turns += 1
        rt.context.set(name, value)
        flow = rt.walk(block.body, f"{path}.body", depth + 1)
        if flow is Flow.BREAK:
            break
        if flow is Flow.RETURN:
            rt.report.note(f"{turns} итераций, выход")
            return flow
        rt.check()
    rt.report.note(f"{turns} из {len(values)} итераций")
    return Flow.NEXT


def sequence(rt: BlockRuntime, block: ActionBlock) -> list[Any]:
    """What a ``For`` walks: an explicit list, or an inclusive range of numbers.

    Cut at one item past the iteration limit, so ``from: 1`` ``to: 1000000`` costs a short
    list and a clear error instead of a gigabyte of integers nobody asked for.
    """
    cap = rt.limits.max_iterations + 1
    if "items" in block.params:
        return as_list(rt.context.fill(block.params["items"]))[:cap]
    start = as_int(rt.context.fill(block.params.get("from")))
    stop = as_int(rt.context.fill(block.params.get("to")))
    step = as_int(rt.context.fill(block.params.get("step")), 1) or 1
    end = stop + 1 if step > 0 else stop - 1
    return list(range(start, end, step)[:cap])


def run_try(rt: BlockRuntime, block: ActionBlock, path: str, depth: int) -> Flow:
    """``Try``: the body, and on a failure the ``catch`` branch with the error at hand.

    Catches what a block's own ``on_error`` let through, including the failure of a nested
    ``CallCommand``, and nothing else. A cancellation or a limit is not the command's
    mistake to handle: a ``Try`` that swallowed the stop word would turn the stop word into
    a suggestion.
    """
    name = format_value(rt.context.fill(block.params.get("error_var", "error"))).strip()
    try:
        return rt.walk(block.body, f"{path}.body", depth + 1)
    except (MacroCancelledError, MacroLimitError):
        raise
    except Exception as exc:
        text = error_text(exc)
        rt.context.set(name or "error", text, VariableScope.LOCAL)
        rt.report.note(f"поймано: {short(text)}")
        if not block.catch:
            return Flow.NEXT
        return rt.walk(block.catch, f"{path}.catch", depth + 1)


def run_wait(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``Wait`` and ``Sleep``: pause on the cancel event, never in :func:`time.sleep`.

    Sleeping *on* the event is the whole reason a stop word reaches a command waiting ten
    seconds in the time it takes to schedule a thread. The pause is also cut down to
    whatever is left of the run's budget, so one ``Wait`` cannot outlive the timeout that
    bounds the run around it.

    Raises:
        MacroCancelledError: the event was set while this block was waiting.
        MacroLimitError: the block asks for a longer pause than one block may take.
        MacroTimeoutError: the budget ran out inside the pause.
    """
    asked = as_int(rt.context.fill(block.params.get("ms")))
    if asked > rt.limits.max_wait_ms:
        raise MacroLimitError("wait_ms", asked, user_message="Пауза в команде слишком длинная.")
    budget = rt.limits.budget_ms
    left = budget - rt.report.elapsed_ms if budget else asked
    pause = max(0, min(asked, left))
    rt.report.note(f"{pause} мс")
    if rt.pause(pause):
        raise MacroCancelledError(rt.reason)
    if pause < asked:
        raise MacroTimeoutError(rt.limits.timeout_s)
    return Flow.NEXT


def run_call(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``CallCommand``: another command as a subroutine, with arguments and a return value.

    The arguments become the called command's slots: a command reads what it was given the
    same way whether a phrase filled it or a caller did, which is why ``{name}`` works in
    both. What comes back is the value of its ``Return``, in this run's ``last_result``.

    ``wait: false`` makes it a command that was merely started — an independent run, with
    nothing to return and no failure to inherit.

    Raises:
        MacroCallError: no such command, or the command is switched off.
        MacroLimitError: the stack of commands is as deep as it may get.
        MacroCancelledError: the called run was stopped, which stops this one too.
    """
    name = format_value(rt.context.fill(block.params.get("command", ""))).strip()
    args = as_dict(rt.context.fill(block.params.get("args")))
    wait = _truthy(rt.context.fill(block.params.get("wait", True)))
    rt.call(name, args, wait=wait)
    return Flow.NEXT


def run_return(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``Return``: end the command here, with a value for whoever called it."""
    value = rt.context.fill(block.params.get("value"))
    rt.returned(value)
    rt.context.set_result(value)
    rt.report.note(short(value))
    return Flow.RETURN


def run_break(_rt: BlockRuntime, _block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``Break``: leave the loop this block sits in."""
    return Flow.BREAK


def run_continue(_rt: BlockRuntime, _block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``Continue``: go on to the next turn of the loop this block sits in."""
    return Flow.CONTINUE


#: Every flow block by the name a ``.ayris`` file spells. ``Case`` and ``Default`` share a
#: handler because they differ only in how :func:`_arm` chooses them, and ``Sleep`` is
#: ``Wait`` under the name VoiceAttack uses — task 36 imports profiles full of it.
LOGIC_HANDLERS: Final[dict[str, BlockHandler]] = {
    "If": run_if,
    "Switch": run_switch,
    "Case": run_case,
    "Default": run_case,
    "While": run_while,
    "For": run_for,
    "Try": run_try,
    "Wait": run_wait,
    "Sleep": run_wait,
    "CallCommand": run_call,
    "Return": run_return,
    "Break": run_break,
    "Continue": run_continue,
}
