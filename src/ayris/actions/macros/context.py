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

Two pieces here are seams rather than final answers. :func:`evaluate_expression`
runs conditions without ``eval`` — literals, names, comparisons, boolean and
arithmetic operators, nothing that can call anything — and :class:`MemoryVariables`
keeps shared variables for the length of the session. Task 32 replaces the first
with the full expression module and the second with the store that persists
``persistent`` variables to the database. Both are injected, both keep this
contract, and until then a command that says ``{work_monitor_brightness} > 0``
works exactly as section 22 of the specification writes it.
"""

from __future__ import annotations

import ast
import json
import operator
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Protocol

from ayris.actions.macros.errors import (
    MacroExpressionError,
    MacroReferenceError,
    MacroValueError,
)
from ayris.core.models import VariableScope, VariableType, utc_now
from ayris.nlu.slots import SLOT_PATTERN

if TYPE_CHECKING:
    from datetime import datetime

    from ayris.actions.macros.schema import VariableModel
    from ayris.actions.result import ActionResult
    from ayris.nlu.slots import SlotSet

__all__ = [
    "MISSING",
    "ExecutionContext",
    "MemoryVariables",
    "RunInfo",
    "TriggerSource",
    "VariableStore",
    "coerce_value",
    "evaluate_expression",
    "format_value",
    "substitute",
    "truthy",
]


class _Missing:
    """Type of :data:`MISSING`."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


#: "There is no such name here", as distinct from "its value is ``None``". A macro
#: variable may legitimately hold ``None``, so a lookup cannot use it as the answer.
MISSING: Final = _Missing()


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


#: Words that mean "no" when a string has to be read as a condition. A ``While``
#: whose variable holds the text ``"false"`` must stop, and ``bool("false")`` is
#: ``True``, which would spin forever. Both languages, because a person writing a
#: command by hand writes either.
_FALSE_WORDS: Final[frozenset[str]] = frozenset(
    {"", "0", "false", "none", "null", "no", "off", "нет", "выкл", "выключено", "ложь"}
)


def truthy(value: object) -> bool:
    """Whether a value counts as "yes" for ``If`` and ``While``.

    Numbers by zero, containers by emptiness, strings by :data:`_FALSE_WORDS` rather
    than by length — see there for why.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().casefold() not in _FALSE_WORDS
    if isinstance(value, int | float):
        return value != 0
    return bool(value)


def format_value(value: object) -> str:
    """A value as it goes into text.

    ``50.0`` becomes ``"50"`` and ``True`` becomes ``"true"``: the first because a
    volume of "50.0 percent" is not what anyone said, the second because that is how
    the same value is spelled in a ``.ayris`` file, so a round trip through text does
    not change meaning.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, Mapping | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _to_number(name: str, value: object, kind: VariableType) -> int | float:
    """A value as a whole number or a fractional one, or a typed failure."""
    if isinstance(value, bool):
        number: int | float = int(value)
    elif isinstance(value, int | float):
        number = value
    else:
        text = format_value(value).strip().replace(",", ".")
        try:
            number = float(text)
        except ValueError as exc:
            raise MacroValueError(name, value, kind.value) from exc
    if kind is VariableType.INT:
        return round(number)
    return float(number)


def _to_container(name: str, value: object, kind: VariableType) -> Any:
    """A value as a list or a dictionary, parsing JSON text when that is what came."""
    wanted: type[Any] = list if kind is VariableType.ARRAY else dict
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError as exc:
            raise MacroValueError(name, value, kind.value) from exc
    if kind is VariableType.ARRAY and isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping) and kind is VariableType.DICT:
        return dict(value)
    if not isinstance(value, wanted):
        raise MacroValueError(name, value, kind.value)
    return value


def coerce_value(name: str, value: object, kind: VariableType) -> Any:
    """A value made to fit the declared type of ``name``.

    Declared types are not decoration: ``SetVolume`` wants a number, and a command
    that puts the text of a slot into an ``int`` variable should fail at the
    assignment, where the name is known, rather than three blocks later inside an
    action's parameter validation.

    Raises:
        MacroValueError: the value cannot be read as ``kind``.
    """
    if kind is VariableType.STRING:
        return format_value(value)
    if kind in (VariableType.INT, VariableType.FLOAT):
        return _to_number(name, value, kind)
    if kind is VariableType.BOOL:
        return truthy(value)
    return _to_container(name, value, kind)


def _empty_value(kind: VariableType) -> Any:
    """What a declared variable holds before anything is written to it."""
    return {
        VariableType.STRING: "",
        VariableType.INT: 0,
        VariableType.FLOAT: 0.0,
        VariableType.BOOL: False,
        VariableType.ARRAY: [],
        VariableType.DICT: {},
    }[kind]


#: A ``{placeholder}``, or a doubled brace standing for a literal one. The
#: placeholder half is :data:`~ayris.nlu.slots.SLOT_PATTERN` itself and not a copy of
#: it: the same spelling has to be understood by the phrase that fills a slot and by
#: the block that reads it, and two regexes would eventually disagree.
_PLACEHOLDER: Final = re.compile(r"\{\{|\}\}|" + SLOT_PATTERN.pattern, re.UNICODE)


def _replace(match: re.Match[str], resolve: Callable[[str], Any]) -> str:
    """One placeholder as text, or the literal brace a doubled one stands for."""
    name = match.group("name")
    if name is None:
        return match.group(0)[0]
    return format_value(resolve(name))


def substitute(text: str, resolve: Callable[[str], Any]) -> Any:
    """Fill the ``{placeholders}`` in ``text``.

    A string that is nothing but one placeholder gives back the value itself rather
    than its text: section 22 writes ``"level": "{volume}"`` and ``SetVolume`` wants
    the number 50 there, which a file cannot spell because the placeholder occupies
    the space where the number would go. Everything else is text with values written
    into it, and ``{{`` is how a command asks for a brace of its own.
    """
    whole = SLOT_PATTERN.fullmatch(text.strip())
    if whole is not None:
        return resolve(whole.group("name"))
    return _PLACEHOLDER.sub(lambda match: _replace(match, resolve), text)


#: Ceilings for one expression. Small on purpose: a condition in a voice command is a
#: comparison, not a program, and ``2 ** 2 ** 40`` should not be able to eat the
#: machine's memory on a thread the user cannot see.
_MAX_EXPRESSION: Final = 500
_MAX_DEPTH: Final = 24
_MAX_POWER: Final = 64
_MAX_REPEAT: Final = 10_000

_BINARY: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_COMPARE: Final[dict[type[ast.cmpop], Callable[[Any, Any], Any]]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}


@dataclass(frozen=True, slots=True)
class _Scope:
    """What the expression walker carries down: the source text and where names live."""

    expression: str
    values: Mapping[str, Any]
    resolve: Callable[[str], Any]


def _guard(scope: _Scope, action: Callable[[], Any]) -> Any:
    """Run one operator, turning "these values do not go together" into a typed error."""
    try:
        return action()
    except (TypeError, ValueError, ZeroDivisionError, KeyError, IndexError, OverflowError) as exc:
        raise MacroExpressionError(scope.expression, f"cannot evaluate ({exc})") from exc


def _is_number(value: object) -> bool:
    """Whether a value is a number to compute with. ``True`` is not one."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _maybe_number(text: str) -> int | float | None:
    """The number a string spells, or ``None``. Accepts a comma for a decimal point."""
    stripped = text.strip().replace(",", ".")
    try:
        number = float(stripped)
    except ValueError:
        return None
    return int(number) if number.is_integer() and "." not in stripped else number


def _same_kind(left: Any, right: Any) -> tuple[Any, Any]:
    """A number and a numeric string made into two numbers.

    An unparsed slot arrives as the text the user said, so ``{volume} > 50`` would
    otherwise fail on comparing ``str`` with ``int`` — a failure about Python types in
    answer to a question about loudness. Text that is not a number is left alone and
    the comparison fails honestly.
    """
    if isinstance(left, str) and _is_number(right):
        number = _maybe_number(left)
        return (number, right) if number is not None else (left, right)
    if isinstance(right, str) and _is_number(left):
        number = _maybe_number(right)
        return (left, number) if number is not None else (left, right)
    return left, right


def _check_power(left: Any, right: Any, scope: _Scope) -> None:
    """Refuse a power that would compute for minutes or fill the memory."""
    if not _is_number(left) or not _is_number(right):
        raise MacroExpressionError(scope.expression, "** wants two numbers")
    if abs(right) > _MAX_POWER or abs(left) > 2**32:
        raise MacroExpressionError(scope.expression, "** operands are too large")


def _check_repeat(left: Any, right: Any, scope: _Scope) -> None:
    """Refuse ``"a" * 10_000_000``, which is a memory bomb rather than a condition."""
    for value, count in ((left, right), (right, left)):
        if isinstance(value, str | list | tuple) and isinstance(count, int):
            if abs(count) > _MAX_REPEAT:
                raise MacroExpressionError(scope.expression, "repetition is too large")
            return


def _boolean(node: ast.BoolOp, scope: _Scope, depth: int) -> Any:
    """``and`` / ``or``, short-circuiting and giving back the operand, as Python does."""
    wanted = isinstance(node.op, ast.And)
    result: Any = wanted
    for operand in node.values:
        result = _evaluate(operand, scope, depth + 1)
        if truthy(result) is not wanted:
            return result
    return result


def _compare(node: ast.Compare, scope: _Scope, depth: int) -> bool:
    """One comparison, or a chain of them: ``0 < {level} <= 100``."""
    left = _evaluate(node.left, scope, depth + 1)
    for op, side in zip(node.ops, node.comparators, strict=True):
        apply = _COMPARE.get(type(op))
        if apply is None:
            raise MacroExpressionError(scope.expression, f"{type(op).__name__} is not allowed")
        right = _evaluate(side, scope, depth + 1)
        first, second = _same_kind(left, right)
        if not _guard(scope, partial(apply, first, second)):
            return False
        left = right
    return True


def _binary(node: ast.BinOp, scope: _Scope, depth: int) -> Any:
    """Arithmetic: the five operators plus ``**``, each with its own ceiling."""
    apply = _BINARY.get(type(node.op))
    if apply is None:
        raise MacroExpressionError(scope.expression, f"{type(node.op).__name__} is not allowed")
    left = _evaluate(node.left, scope, depth + 1)
    right = _evaluate(node.right, scope, depth + 1)
    if isinstance(node.op, ast.Pow):
        _check_power(left, right, scope)
    elif isinstance(node.op, ast.Mult):
        _check_repeat(left, right, scope)
    first, second = _same_kind(left, right)
    return _guard(scope, partial(apply, first, second))


def _unary(node: ast.UnaryOp, scope: _Scope, depth: int) -> Any:
    """``not``, unary minus, unary plus."""
    value = _evaluate(node.operand, scope, depth + 1)
    if isinstance(node.op, ast.Not):
        return not truthy(value)
    if isinstance(node.op, ast.USub):
        return _guard(scope, partial(operator.neg, value))
    if isinstance(node.op, ast.UAdd):
        return value
    raise MacroExpressionError(scope.expression, f"{type(node.op).__name__} is not allowed")


def _constant(node: ast.Constant, scope: _Scope) -> Any:
    """A literal, as long as it is one of the four kinds a command can mean."""
    if isinstance(node.value, str | int | float | None):
        return node.value
    kind = type(node.value).__name__
    raise MacroExpressionError(scope.expression, f"{kind} literals are not allowed")


#: Words that spell a literal without quotes. ``True`` and ``None`` written the
#: Python way are constants to :mod:`ast` and never reach here; ``true`` and ``null``
#: are how the same values are spelled in a ``.ayris`` file, and a condition copied
#: out of one should mean the same thing.
_CONSTANTS: Final[dict[str, Any]] = {"true": True, "false": False, "none": None, "null": None}


def _name(name: str, scope: _Scope) -> Any:
    """A bare name: an alias for a placeholder, a spelled-out literal, or a variable."""
    if name in scope.values:
        return scope.values[name]
    if name in _CONSTANTS:
        return _CONSTANTS[name]
    return scope.resolve(name)


def _mapping(node: ast.Dict, scope: _Scope, depth: int) -> Any:
    """A ``{"mode": "работа"}`` literal. ``{**other}`` is not one."""
    pairs: list[tuple[Any, Any]] = []
    for key, value in zip(node.keys, node.values, strict=True):
        if key is None:
            raise MacroExpressionError(scope.expression, "** in a dict is not allowed")
        pairs.append((_evaluate(key, scope, depth + 1), _evaluate(value, scope, depth + 1)))
    return _guard(scope, partial(dict, pairs))


def _subscript(node: ast.Subscript, scope: _Scope, depth: int) -> Any:
    """``{items}[0]`` and ``{data}["mode"]``: one element out of a container."""
    if isinstance(node.slice, ast.Slice):
        raise MacroExpressionError(scope.expression, "slices are not allowed")
    value = _evaluate(node.value, scope, depth + 1)
    if not isinstance(value, Mapping | list | tuple | str):
        raise MacroExpressionError(scope.expression, "only a list, a dict or text can be indexed")
    key = _evaluate(node.slice, scope, depth + 1)
    return _guard(scope, partial(operator.getitem, value, key))


def _conditional(node: ast.IfExp, scope: _Scope, depth: int) -> Any:
    """``"тихо" if {volume} < 20 else "громко"``."""
    if truthy(_evaluate(node.test, scope, depth + 1)):
        return _evaluate(node.body, scope, depth + 1)
    return _evaluate(node.orelse, scope, depth + 1)


def _evaluate(node: ast.expr, scope: _Scope, depth: int) -> Any:
    """One node of the whitelist. What is not listed here cannot run at all.

    No call, no attribute, no comprehension, no assignment, no f-string: each is
    refused by the name of its node type, because a condition that reaches for
    ``{x}.__class__`` is either a mistake worth a message or an attempt worth a no.
    """
    if depth > _MAX_DEPTH:
        raise MacroExpressionError(scope.expression, "expression is nested too deeply")
    if isinstance(node, ast.Constant):
        return _constant(node, scope)
    if isinstance(node, ast.Name):
        return _name(node.id, scope)
    if isinstance(node, ast.BoolOp):
        return _boolean(node, scope, depth)
    if isinstance(node, ast.Compare):
        return _compare(node, scope, depth)
    if isinstance(node, ast.BinOp):
        return _binary(node, scope, depth)
    if isinstance(node, ast.UnaryOp):
        return _unary(node, scope, depth)
    if isinstance(node, ast.List | ast.Tuple):
        return [_evaluate(item, scope, depth + 1) for item in node.elts]
    if isinstance(node, ast.Dict):
        return _mapping(node, scope, depth)
    if isinstance(node, ast.Subscript):
        return _subscript(node, scope, depth)
    if isinstance(node, ast.IfExp):
        return _conditional(node, scope, depth)
    raise MacroExpressionError(scope.expression, f"{type(node).__name__} is not allowed")


def _alias_placeholders(text: str, resolve: Callable[[str], Any]) -> tuple[str, dict[str, Any]]:
    """``{volume} > 50`` as ``_p0 > 50``, plus what ``_p0`` stands for.

    Values are bound to generated names instead of being written into the text. A slot
    holding ``50) or (1`` has to be a value that fails a comparison, not source that
    parses — this is the whole reason the evaluator exists. One alias per name, so a
    condition cannot see two different values for the same variable.
    """
    values: dict[str, Any] = {}
    aliases: dict[str, str] = {}

    def alias(match: re.Match[str]) -> str:
        name = match.group("name")
        if name is None:
            return match.group(0)[0]
        key = aliases.get(name)
        if key is None:
            key = f"_p{len(aliases)}"
            aliases[name] = key
            values[key] = resolve(name)
        return key

    return _PLACEHOLDER.sub(alias, text), values


def evaluate_expression(expression: str, resolve: Callable[[str], Any]) -> Any:
    """Compute one condition or expression, without ``eval``.

    Parsed by :mod:`ast` and walked by the whitelist in :func:`_evaluate`: literals,
    names, comparisons, boolean and arithmetic operators, indexing, list and dict
    literals, and ``a if c else b``. Placeholders are looked up through ``resolve``
    before parsing and reach the tree as values, never as text.

    Raises:
        MacroExpressionError: it does not parse, it uses something refused, or the
            values do not go together.
        MacroReferenceError: a placeholder names something that does not exist.
    """
    if len(expression) > _MAX_EXPRESSION:
        shown = f"{expression[:60]}..."
        raise MacroExpressionError(shown, f"longer than {_MAX_EXPRESSION} characters")
    text, values = _alias_placeholders(expression, resolve)
    if not text.strip():
        raise MacroExpressionError(expression, "expression is empty")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise MacroExpressionError(expression, "cannot parse") from exc
    return _evaluate(tree.body, _Scope(expression, values, resolve), 0)


def _slot_values(slots: SlotSet | None) -> dict[str, Any]:
    """Slots as plain values: the parsed one when it came out, the spoken text otherwise.

    ``{volume}`` in "айрис громкость пятьдесят" is the number 50 when the slot type
    parsed it and the word "пятьдесят" when it did not. A block reading the slot gets
    the better of the two rather than nothing at all.
    """
    if slots is None:
        return {}
    return {slot.name: slot.value if slot.parsed else slot.raw for slot in slots}


class VariableStore(Protocol):
    """Where ``profile`` and ``global`` variables live.

    Task 32 implements this over the database; :class:`MemoryVariables` implements it
    over a dictionary, and the interpreter never learns which one it got.
    :meth:`update` is here because ``ArrayPush`` on a shared array is a read, a change
    and a write that another thread must not be able to split.
    """

    def read(self, scope: VariableScope, name: str) -> Any:
        """The value, or :data:`MISSING` when this scope has no such name."""
        ...

    def write(self, scope: VariableScope, name: str, value: Any) -> None:
        """Put ``value`` under ``name``, replacing whatever was there."""
        ...

    def update(self, scope: VariableScope, name: str, change: Callable[[Any], Any]) -> Any:
        """Read, change and write as one step, giving back what was written."""
        ...

    def names(self, scope: VariableScope) -> frozenset[str]:
        """Every name this scope holds."""
        ...


class MemoryVariables:
    """Shared variables for the length of the session: a dictionary behind a lock.

    Not a stand-in for something missing. A ``global`` variable that is not
    ``persistent`` lives exactly this long, so task 32 replaces this class for the
    persistent half and keeps it for the rest.

    The lock is an :class:`threading.RLock` because :meth:`update` calls a function
    while holding it and that function may read the same store.
    """

    def __init__(self, initial: Mapping[VariableScope, Mapping[str, Any]] | None = None) -> None:
        self._lock = threading.RLock()
        self._values: dict[VariableScope, dict[str, Any]] = {
            VariableScope.PROFILE: {},
            VariableScope.GLOBAL: {},
        }
        for scope, values in (initial or {}).items():
            self._values.setdefault(scope, {}).update(values)

    def read(self, scope: VariableScope, name: str) -> Any:
        """The value, or :data:`MISSING` when this scope has no such name."""
        with self._lock:
            return self._values.get(scope, {}).get(name, MISSING)

    def write(self, scope: VariableScope, name: str, value: Any) -> None:
        """Put ``value`` under ``name``, replacing whatever was there."""
        with self._lock:
            self._values.setdefault(scope, {})[name] = value

    def update(self, scope: VariableScope, name: str, change: Callable[[Any], Any]) -> Any:
        """Read, change and write as one step, giving back what was written."""
        with self._lock:
            value = change(self.read(scope, name))
            self.write(scope, name, value)
            return value

    def names(self, scope: VariableScope) -> frozenset[str]:
        """Every name this scope holds."""
        with self._lock:
            return frozenset(self._values.get(scope, {}))

    def snapshot(self, scope: VariableScope | None = None) -> dict[str, Any]:
        """A copy of one scope, or of both shadowed the way a lookup shadows them."""
        with self._lock:
            if scope is not None:
                return dict(self._values.get(scope, {}))
            merged: dict[str, Any] = {}
            for key in (VariableScope.GLOBAL, VariableScope.PROFILE):
                merged.update(self._values.get(key, {}))
            return merged

    def clear(self, scope: VariableScope | None = None) -> None:
        """Forget one scope, or every one of them. What switching profiles does."""
        with self._lock:
            for key in [scope] if scope is not None else list(self._values):
                self._values[key] = {}


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

    ``evaluator`` is the seam for task 32: :func:`evaluate_expression` today, the full
    expression module later. Its signature is the entire contract — an expression, and
    a way to look one name up.
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
        """
        for variable in variables:
            name = variable.name
            self._types[name] = variable.type
            self._scopes[name] = variable.scope
            if variable.default is None:
                value = _empty_value(variable.type)
            else:
                value = coerce_value(name, variable.default, variable.type)
            if variable.scope is VariableScope.LOCAL:
                self.locals[name] = value
            elif self.store.read(variable.scope, name) is MISSING:
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

    @staticmethod
    def _container(name: str, current: Any, kind: VariableType) -> Any:
        """What is already under ``name``, as the container kind, empty when nothing is."""
        if current is MISSING or current is None or current == "":
            return _empty_value(kind)
        return _to_container(name, current, kind)

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
