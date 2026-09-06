"""What a running command can see and change.

The interpreter of task 31 walks a tree; this module is everything that tree reads
from and writes to. Four sources, and the order between them is the contract:

1. **Locals** — declared with ``scope = local`` or written by ``SetVar``. One run,
   one set; a second copy of the same command fired at the same time has its own.
2. **Slots** — what the phrase filled: ``{volume}`` in "айрис громкость 50".
   Read-only, because rewriting what the user said is never what was meant.
3. **Profile** and **global** variables — shared, and therefore locked. Two
   commands fired at once run on two threads of the pool and both may touch
   ``work_mode``, so every write to a shared scope goes through
   :class:`VariableStore`, whose implementation holds the lock.

Two things this module does not do itself. Text becomes a value in
:mod:`~ayris.actions.macros.expressions` — substitution and the ``eval``-less expression
engine — and shared variables live in :mod:`~ayris.actions.macros.variables`, in memory
or in the database. Both are injected: :attr:`ExecutionContext.store` takes any
:class:`~ayris.actions.macros.variables.VariableStore` and ``evaluator`` takes any
"expression plus a lookup" callable, so a command that says
``{work_monitor_brightness} > 0`` works exactly as section 22 of the specification
writes it, whether or not that variable survives a restart.

Names from those two modules are re-exported here because this is the module the
interpreter imports, and moving a helper between files should not move an import.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ayris.actions.macros.errors import MacroIndexError, MacroReferenceError
from ayris.actions.macros.expressions import (
    coerce_value,
    empty_value,
    evaluate_expression,
    format_value,
    substitute,
    to_container,
    truthy,
)
from ayris.actions.macros.variables import (
    MISSING,
    DatabaseVariables,
    MemoryVariables,
    VariableStore,
)
from ayris.core.models import VariableScope, VariableType, utc_now

if TYPE_CHECKING:
    from datetime import datetime

    from ayris.actions.macros.schema import VariableModel
    from ayris.actions.result import ActionResult
    from ayris.nlu.slots import SlotSet

__all__ = [
    "MISSING",
    "DatabaseVariables",
    "ExecutionContext",
    "MemoryVariables",
    "RunInfo",
    "TriggerSource",
    "VariableStore",
    "coerce_value",
    "empty_value",
    "evaluate_expression",
    "format_value",
    "substitute",
    "truthy",
]


class TriggerSource(StrEnum):
    """What set this run going.

    Not the same list as :class:`~ayris.core.models.TriggerType`, which describes what
    a command *has*: a run can also come from the editor's "run once" button, from a
    ``CallCommand`` inside another command, or from a plugin that has no trigger at
    all. The report and the history keep this, so "why did my microphone mute" has an
    answer.
    """

    VOICE = "voice"
    HOTKEY = "hotkey"
    EVENT = "event"
    TIMER = "timer"
    MANUAL = "manual"
    CALL = "call"
    PLUGIN = "plugin"


def _slot_values(slots: SlotSet | None) -> dict[str, Any]:
    """Slots as plain values: the parsed one when it came out, the spoken text otherwise.

    ``{volume}`` in "айрис громкость пятьдесят" is the number 50 when the slot type
    parsed it and the word "пятьдесят" when it did not. A block reading the slot gets
    the better of the two rather than nothing at all.
    """
    if slots is None:
        return {}
    return {slot.name: slot.value if slot.parsed else slot.raw for slot in slots}


@dataclass(frozen=True, slots=True)
class RunInfo:
    """Which run this is: its identity, what started it, and who called it.

    ``request_id`` is the same string the whole chain carries — the intent that matched,
    the actions the registry ran, the rows task 33 writes — so one voice phrase can be
    followed from the microphone to the report even when it fired three commands.

    ``call_stack`` is the names of the commands above this one. It is what stops a
    command that calls itself: the engine compares its length against its call-depth
    limit, and the names make the message say which loop it was.
    """

    run_id: str
    command: str = ""
    command_id: int | None = None
    trigger: TriggerSource = TriggerSource.MANUAL
    request_id: str = ""
    call_stack: tuple[str, ...] = ()
    started_at: datetime = field(default_factory=utc_now)

    @property
    def depth(self) -> int:
        """How many commands are above this one. Zero for a run the user started."""
        return len(self.call_stack)

    @property
    def called_by(self) -> str:
        """The command that called this one, or ``""`` for a run nobody called."""
        return self.call_stack[-1] if self.call_stack else ""

    def called(self, command: str, *, run_id: str, command_id: int | None = None) -> RunInfo:
        """The info a nested ``CallCommand`` runs under: same request, one level deeper."""
        return RunInfo(
            run_id=run_id,
            command=command,
            command_id=command_id,
            trigger=TriggerSource.CALL,
            request_id=self.request_id,
            call_stack=(*self.call_stack, self.command or self.run_id),
        )


class ExecutionContext:
    """Everything one run of one command can read and write.

    Built by the engine before the first block and handed to every block after it.
    :attr:`locals` belong to this run alone; the store behind ``profile`` and
    ``global`` is shared with every run happening at the same time, which is why
    writes to it go through :class:`VariableStore` rather than through a dictionary
    here.

    ``evaluator`` is a seam, and its signature is the entire contract — an expression,
    and a way to look one name up. The debugger of task 35 replaces it to watch what a
    condition read; nothing else needs to.
    """

    __slots__ = (
        "_action",
        "_evaluator",
        "_last",
        "_scopes",
        "_types",
        "info",
        "locals",
        "slots",
        "store",
    )

    def __init__(
        self,
        *,
        info: RunInfo,
        store: VariableStore | None = None,
        slots: SlotSet | Mapping[str, Any] | None = None,
        variables: Iterable[VariableModel] = (),
        evaluator: Callable[[str, Callable[[str], Any]], Any] = evaluate_expression,
    ) -> None:
        self.info = info
        self.store: VariableStore = MemoryVariables() if store is None else store
        self.slots: dict[str, Any] = (
            dict(slots) if isinstance(slots, Mapping) else _slot_values(slots)
        )
        self.locals: dict[str, Any] = {"last_result": None}
        self._types: dict[str, VariableType] = {}
        self._scopes: dict[str, VariableScope] = {}
        self._evaluator = evaluator
        self._last: Any = None
        self._action: ActionResult[Any] | None = None
        self.declare(variables)

    def declare(self, variables: Iterable[VariableModel]) -> None:
        """Register what the command declares and fill in the defaults that are absent.

        A ``local`` variable starts at its default on every run. A ``profile`` or
        ``global`` one is written only when the store does not have it yet: the point of
        a shared variable is that it outlives the run, so section 22's
        ``work_monitor_brightness`` keeps the 30 the user set instead of going back to
        its declared 70 every morning.

        The declaration is passed on to the store as well, because ``persistent`` is a
        property of the name and not of any one write: the store has to know which names
        it owes a row before the first ``SetVar`` touches one.
        """
        for variable in variables:
            name = variable.name
            self._types[name] = variable.type
            self._scopes[name] = variable.scope
            if variable.default is None:
                value = empty_value(variable.type)
            else:
                value = coerce_value(name, variable.default, variable.type)
            if variable.scope is VariableScope.LOCAL:
                self.locals[name] = value
                continue
            self.store.declare(
                variable.scope,
                name,
                var_type=variable.type,
                persistent=variable.persistent,
            )
            if self.store.read(variable.scope, name) is MISSING:
                self.store.write(variable.scope, name, value)

    @property
    def last_result(self) -> Any:
        """What the previous block produced. Also readable as ``{last_result}``."""
        return self._last

    @property
    def last_action(self) -> ActionResult[Any] | None:
        """The whole result of the last action block, when the last block was one.

        The value is in :attr:`last_result`; this is for a caller that wants the rest —
        the Russian message, the duration, the undo token.
        """
        return self._action

    def set_result(self, value: Any, *, action: ActionResult[Any] | None = None) -> None:
        """Remember what a block produced, for the next block and for ``{last_result}``.

        The name exists before the first block, holding ``None``: a ``While`` that polls
        until an action answers reads it on its very first turn, and «ничего ещё не
        произошло» is an answer there, where an unknown name would be a failure.
        """
        self._last = value
        self._action = action
        self.locals["last_result"] = value

    def lookup(self, name: str) -> Any:
        """The value of ``name``, or :data:`MISSING`. The order is this module's contract."""
        if name in self.locals:
            return self.locals[name]
        if name in self.slots:
            return self.slots[name]
        for scope in (VariableScope.PROFILE, VariableScope.GLOBAL):
            value = self.store.read(scope, name)
            if value is not MISSING:
                return value
        return MISSING

    def has(self, name: str) -> bool:
        """Whether anything answers to ``name``."""
        return self.lookup(name) is not MISSING

    def get(self, name: str, default: Any = None) -> Any:
        """The value of ``name``, or ``default`` when there is no such name."""
        value = self.lookup(name)
        return default if value is MISSING else value

    def resolve(self, name: str) -> Any:
        """The value of ``name``, or a failure that says which name.

        What every ``{placeholder}`` and every bare name in a condition goes through.

        Raises:
            MacroReferenceError: nothing answers to ``name``.
        """
        value = self.lookup(name)
        if value is MISSING:
            raise MacroReferenceError(name)
        return value

    def type_of(self, name: str) -> VariableType | None:
        """The declared type of ``name``, or ``None`` when it was never declared."""
        return self._types.get(name)

    def scope_of(self, name: str) -> VariableScope:
        """Where a write to ``name`` goes when the block does not say.

        The declared scope, else the scope the name already lives in, else ``local`` —
        so an undeclared ``SetVar`` cannot quietly create a global.
        """
        if name in self._scopes:
            return self._scopes[name]
        if name in self.locals or name in self.slots:
            return VariableScope.LOCAL
        for scope in (VariableScope.PROFILE, VariableScope.GLOBAL):
            if self.store.read(scope, name) is not MISSING:
                return scope
        return VariableScope.LOCAL

    def set(self, name: str, value: Any, scope: VariableScope | None = None) -> Any:
        """Write ``value`` to ``name``, coerced to its declared type. Gives back what went in.

        Raises:
            MacroValueError: the value does not fit the declared type.
        """
        target = self.scope_of(name) if scope is None else scope
        kind = self._types.get(name)
        stored = value if kind is None else coerce_value(name, value, kind)
        if target is VariableScope.LOCAL:
            self.locals[name] = stored
        else:
            self.store.write(target, name, stored)
            self._scopes.setdefault(name, target)
        return stored

    def append(self, name: str, value: Any) -> Any:
        """``ArrayPush``: add one element without losing an element another run pushed."""

        def change(current: Any) -> list[Any]:
            items = self._container(name, current, VariableType.ARRAY)
            return [*items, value]

        return self._change(name, change)

    def put(self, name: str, key: Any, value: Any) -> Any:
        """``DictSet``: write one key without losing a key another run wrote."""

        def change(current: Any) -> dict[Any, Any]:
            mapping = self._container(name, current, VariableType.DICT)
            return {**mapping, key: value}

        return self._change(name, change)

    def pop(self, name: str, index: int | None = None) -> Any:
        """``ArrayPop``: take one element out and give it back, in one step.

        The removed element cannot be found by reading the array afterwards — another run
        may have pushed to it in between — so the removal and the answer have to happen
        inside the same locked change. The element is carried out through ``taken``.

        Raises:
            MacroValueError: what is under ``name`` is not an array.
            MacroIndexError: the array is empty, or the index is outside it.
        """
        taken: list[Any] = []

        def change(current: Any) -> list[Any]:
            items = list(self._container(name, current, VariableType.ARRAY))
            position = len(items) - 1 if index is None else index
            if not items or not -len(items) <= position < len(items):
                raise MacroIndexError(name, position, len(items))
            taken.append(items.pop(position))
            return items

        self._change(name, change)
        return taken[0]

    @staticmethod
    def _container(name: str, current: Any, kind: VariableType) -> Any:
        """What is already under ``name``, as the container kind, empty when nothing is."""
        if current is MISSING or current is None or current == "":
            return empty_value(kind)
        return to_container(name, current, kind)

    def _change(self, name: str, change: Callable[[Any], Any]) -> Any:
        """Read, change and write one container. Atomic when the scope is a shared one."""
        target = self.scope_of(name)
        if target is VariableScope.LOCAL:
            value = change(self.locals.get(name, MISSING))
            self.locals[name] = value
            return value
        value = self.store.update(target, name, change)
        self._scopes.setdefault(name, target)
        return value

    def fill(self, value: Any) -> Any:
        """Fill the ``{placeholders}`` in a value, in a list of them, or in a params dict.

        Text becomes text with values written into it, a lone placeholder becomes the
        value itself, and anything that is not text comes back untouched — which is what
        lets both ``ms: 500`` and ``ms: "{delay}"`` reach ``Wait``.

        Raises:
            MacroReferenceError: a placeholder names something that does not exist.
        """
        if isinstance(value, str):
            return substitute(value, self.resolve)
        if isinstance(value, Mapping):
            return {key: self.fill(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [self.fill(item) for item in value]
        return value

    def evaluate(self, expression: Any) -> Any:
        """Compute a condition. Anything that is not text is already its own answer."""
        if not isinstance(expression, str):
            return expression
        return self._evaluator(expression, self.resolve)

    def truth(self, expression: Any) -> bool:
        """Whether a condition holds, by :func:`truthy` on whatever it computes to."""
        return truthy(self.evaluate(expression))

    def child(
        self,
        info: RunInfo,
        *,
        slots: Mapping[str, Any] | None = None,
        variables: Iterable[VariableModel] = (),
    ) -> ExecutionContext:
        """The context a ``CallCommand`` runs in: its own locals, the same shared store.

        The nested command does not see the caller's locals or slots. A command is a
        unit, and one that read its caller's variables could not be called from anywhere
        else. What does cross is ``args``, which the engine passes in as ``slots``.
        """
        return ExecutionContext(
            info=info,
            store=self.store,
            slots={} if slots is None else slots,
            variables=variables,
            evaluator=self._evaluator,
        )

    def snapshot(self) -> dict[str, Any]:
        """Everything visible right now, shadowed the way :meth:`lookup` shadows it.

        For the debugger of task 35 and for the variables pane: one dictionary, no
        promise about the scope a name came from.
        """
        values: dict[str, Any] = {}
        for scope in (VariableScope.GLOBAL, VariableScope.PROFILE):
            for name in self.store.names(scope):
                values[name] = self.store.read(scope, name)
        values.update(self.slots)
        values.update(self.locals)
        return values
