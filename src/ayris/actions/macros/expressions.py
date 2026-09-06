"""Turning text into a value: substitution, and an expression engine with no ``eval``.

Two jobs in one module, because they are the same job seen twice. ``{volume}`` inside
``"громкость {volume}"`` is a value written into text; ``{volume} > 50`` is a value
compared with another one. Both start from a name and a way to look it up, and both
have to survive text a stranger wrote: section 7.1 lets a command carry conditions,
task 36 imports them wholesale out of VoiceAttack profiles, and a plugin may build one
from a string it took off the network.

**Why not ``eval``.** Because ``eval`` cannot be made safe by a list of forbidden
words. Every attribute is a door — ``().__class__.__base__.__subclasses__()`` reaches
the interpreter from a bare tuple — so this module never runs source at all. It parses
with :mod:`ast` and walks the tree itself, and the walk knows a dozen node types.
Everything else is refused by the name of its node type: ``__import__("os")`` is a
call of a name that is not in :data:`FUNCTIONS`, ``{x}.__class__`` is an
:class:`ast.Attribute`, ``[c for c in x]`` is a comprehension. There is no blacklist to
keep up to date, which is the whole point.

**Why a value never becomes source.** A slot holding ``50) or (1`` would parse as an
expression if it were written into the text. :func:`_alias_placeholders` binds every
placeholder to a generated name — ``{volume} > 50`` becomes ``_p0 > 50`` — before
anything is parsed, so what a stranger said is data by construction.

**Why the ceilings.** ``10 ** 10 ** 10`` is nine characters of text and minutes of a
core; ``"a" * 10000000`` is ten megabytes. :data:`MAX_EXPRESSION`, :data:`MAX_DEPTH`,
:data:`MAX_POWER` and :data:`MAX_REPEAT` bound the text, the tree, the exponent and the
repetition, and each of them fails the same typed way as everything else here.

The functions a command may call are a list and not a rule: :data:`FUNCTIONS` holds the
nine of section 7.1 with an arity for each. Adding one is a line in this file; asking
for one that is not there is a message naming the function that was refused.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from functools import partial
from typing import Any, Final

from ayris.actions.macros.errors import MacroExpressionError, MacroValueError
from ayris.core.models import VariableType
from ayris.nlu.slots import SLOT_PATTERN

__all__ = [
    "FUNCTIONS",
    "MAX_DEPTH",
    "MAX_EXPRESSION",
    "MAX_PLACES",
    "MAX_POWER",
    "MAX_REPEAT",
    "coerce_value",
    "empty_value",
    "evaluate_expression",
    "format_value",
    "substitute",
    "to_container",
    "truthy",
]


#: Words that mean "no" when a string has to be read as a condition. A ``While`` whose
#: variable holds the text ``"false"`` must stop, and ``bool("false")`` is ``True``,
#: which would spin forever. Both languages, because a person writing a command by hand
#: writes either.
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

    ``50.0`` becomes ``"50"`` and ``True`` becomes ``"true"``: the first because a volume
    of "50.0 percent" is not what anyone said, the second because that is how the same
    value is spelled in a ``.ayris`` file, so a round trip through text does not change
    meaning.
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


def _half_up(number: int | float, places: int = 0) -> int | float:
    """Round the way a person means it: ``2.5`` up, ``-2.5`` away from zero.

    Not :func:`round`, which rounds halves to the nearest *even* number — ``round(2.5)``
    is 2 and ``round(3.5)`` is 4. That is the right rule for statistics and the wrong one
    for a command that says «округли»: a user who sees 2.5 become 2 reports a bug.
    """
    if isinstance(number, int):
        return number if places >= 0 else int(_decimal_round(number, places))
    return _decimal_round(number, places)


def _decimal_round(number: int | float, places: int) -> int | float:
    """The rounding itself, done in decimal so ``2.675`` is not ``2.67``."""
    step = Decimal(1).scaleb(-places)
    shifted = Decimal(str(number)).quantize(step, rounding=ROUND_HALF_UP)
    return int(shifted) if places <= 0 else float(shifted)


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
        return int(_half_up(number))
    return float(number)


def to_container(name: str, value: object, kind: VariableType) -> Any:
    """A value as a list or a dictionary, parsing JSON text when that is what came.

    Public because ``ArrayPush`` needs exactly this and nothing around it: what is under
    the name has to become a list before an element can be added to it.

    Raises:
        MacroValueError: the value is not that container and cannot be read as one.
    """
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

    Declared types are not decoration: ``SetVolume`` wants a number, and a command that
    puts the text of a slot into an ``int`` variable should fail at the assignment, where
    the name is known, rather than three blocks later inside an action's parameter
    validation.

    Raises:
        MacroValueError: the value cannot be read as ``kind``.
    """
    if kind is VariableType.STRING:
        return format_value(value)
    if kind in (VariableType.INT, VariableType.FLOAT):
        return _to_number(name, value, kind)
    if kind is VariableType.BOOL:
        return truthy(value)
    return to_container(name, value, kind)


def empty_value(kind: VariableType) -> Any:
    """What a declared variable holds before anything is written to it."""
    return {
        VariableType.STRING: "",
        VariableType.INT: 0,
        VariableType.FLOAT: 0.0,
        VariableType.BOOL: False,
        VariableType.ARRAY: [],
        VariableType.DICT: {},
    }[kind]


#: A ``{placeholder}``, or a doubled brace standing for a literal one. The placeholder
#: half is :data:`~ayris.nlu.slots.SLOT_PATTERN` itself and not a copy of it: the same
#: spelling has to be understood by the phrase that fills a slot and by the block that
#: reads it, and two regexes would eventually disagree.
_PLACEHOLDER: Final = re.compile(r"\{\{|\}\}|" + SLOT_PATTERN.pattern, re.UNICODE)


def _replace(match: re.Match[str], resolve: Callable[[str], Any]) -> str:
    """One placeholder as text, or the literal brace a doubled one stands for."""
    name = match.group("name")
    if name is None:
        return match.group(0)[0]
    return format_value(resolve(name))


def substitute(text: str, resolve: Callable[[str], Any]) -> Any:
    """Fill the ``{placeholders}`` in ``text``.

    A string that is nothing but one placeholder gives back the value itself rather than
    its text: section 22 writes ``"level": "{volume}"`` and ``SetVolume`` wants the number
    50 there, which a file cannot spell because the placeholder occupies the space where
    the number would go. Everything else is text with values written into it, and ``{{``
    is how a command asks for a brace of its own.

    Raises:
        MacroReferenceError: ``resolve`` knows no such name — raised by the caller's
            lookup, and left alone here so the message names the variable.
    """
    whole = SLOT_PATTERN.fullmatch(text.strip())
    if whole is not None:
        return resolve(whole.group("name"))
    return _PLACEHOLDER.sub(lambda match: _replace(match, resolve), text)


#: How long an expression may be. Small on purpose: a condition in a voice command is a
#: comparison, not a program.
MAX_EXPRESSION: Final = 500
#: How deeply an expression may nest. ``((((…))))`` past this is refused before it runs.
MAX_DEPTH: Final = 24
#: The largest exponent ``**`` will compute, so ``10 ** 10 ** 10`` cannot eat the memory
#: of a machine on a thread the user cannot see.
MAX_POWER: Final = 64
#: The largest repetition ``*`` will build, so ``"a" * 10000000`` is refused as well.
MAX_REPEAT: Final = 10_000
#: How many decimal places ``round`` will go to. ``round(x, 100000)`` is not a rounding.
MAX_PLACES: Final = 12

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


def _as_number(value: Any) -> int | float:
    """A value as a number for arithmetic, accepting a comma for a decimal point.

    Raises:
        ValueError: the text is not a number. Turned into a typed failure by
            :func:`_guard`, which is the only caller's caller.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float):
        return value
    return float(format_value(value).strip().replace(",", "."))


def _length(value: Any) -> int:
    """``len``, over the four things that have one.

    Raises:
        TypeError: a number has no length, and ``len({volume})`` is a mistake worth a
            message rather than a zero.
    """
    if isinstance(value, str | list | tuple | Mapping):
        return len(value)
    raise TypeError(f"len() wants text, a list or a dict, got {type(value).__name__}")


def _round(value: Any, digits: Any = None) -> Any:
    """``round``: to a whole number by default, to ``digits`` places when asked.

    Half up and computed in decimal — see :func:`_half_up` for why the built-in rule is
    the wrong one here.
    """
    number = _as_number(value)
    places = 0 if digits is None else int(_as_number(digits))
    if abs(places) > MAX_PLACES:
        raise ValueError(f"round() to {places} places makes no sense")
    return _half_up(number, places)


def _pick(choose: Callable[[Sequence[Any]], Any], args: Sequence[Any]) -> Any:
    """``min`` and ``max``, over either one list or several values."""
    if len(args) == 1 and isinstance(args[0], list | tuple):
        items: Sequence[Any] = args[0]
    else:
        items = args
    if not items:
        raise ValueError("min() and max() want at least one value")
    return choose(items)


def _contains(haystack: Any, needle: Any) -> bool:
    """``contains(x, y)``: what ``in`` means, spelled as a function.

    Section 7.1 asks for it by name, and it is more lenient than ``in`` on purpose: an
    array built out of slots holds the text ``"50"`` where a condition says ``50``, and a
    command asking whether the number is in the list means yes.
    """
    if isinstance(haystack, str):
        return format_value(needle) in haystack
    if isinstance(haystack, Mapping):
        return needle in haystack or format_value(needle) in haystack
    if isinstance(haystack, list | tuple):
        if needle in haystack:
            return True
        wanted = format_value(needle)
        return any(format_value(item) == wanted for item in haystack)
    raise TypeError(f"contains() wants text, a list or a dict, got {type(haystack).__name__}")


#: The functions a command may call, with the number of arguments each takes. The nine of
#: section 7.1 and nothing else: a name that is not a key here is refused by name, so
#: ``__import__("os")`` fails the same way as a typo. ``str`` is :func:`format_value` and
#: not ``str`` itself, so ``str(50.0)`` gives ``"50"`` — the same text the value would
#: have inside a phrase.
FUNCTIONS: Final[dict[str, tuple[Callable[..., Any], int, int]]] = {
    "len": (_length, 1, 1),
    "int": (lambda value: int(_as_number(value)), 1, 1),
    "float": (lambda value: float(_as_number(value)), 1, 1),
    "str": (format_value, 1, 1),
    "round": (_round, 1, 2),
    "abs": (lambda value: abs(_as_number(value)), 1, 1),
    "min": (lambda *args: _pick(min, args), 1, 8),
    "max": (lambda *args: _pick(max, args), 1, 8),
    "contains": (_contains, 2, 2),
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
    if abs(right) > MAX_POWER or abs(left) > 2**32:
        raise MacroExpressionError(scope.expression, "** operands are too large")


def _check_repeat(left: Any, right: Any, scope: _Scope) -> None:
    """Refuse ``"a" * 10_000_000``, which is a memory bomb rather than a condition."""
    for value, count in ((left, right), (right, left)):
        if isinstance(value, str | list | tuple) and isinstance(count, int):
            if abs(count) > MAX_REPEAT:
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


def _call(node: ast.Call, scope: _Scope, depth: int) -> Any:
    """A call of one of :data:`FUNCTIONS`, and of nothing else.

    The name has to be a bare :class:`ast.Name`: ``{x}.upper()`` has an
    :class:`ast.Attribute` in the callable position and is refused here rather than
    somewhere deeper, and a name that is not a key of :data:`FUNCTIONS` is refused by the
    name it asked for — which is what happens to ``__import__("os")``.
    """
    if not isinstance(node.func, ast.Name):
        raise MacroExpressionError(scope.expression, "only a plain function name can be called")
    entry = FUNCTIONS.get(node.func.id)
    if entry is None:
        raise MacroExpressionError(scope.expression, f"function {node.func.id}() is not allowed")
    if node.keywords:
        raise MacroExpressionError(scope.expression, "named arguments are not allowed")
    function, least, most = entry
    args = [_evaluate(argument, scope, depth + 1) for argument in node.args]
    if not least <= len(args) <= most:
        wanted = f"{least}" if least == most else f"{least} to {most}"
        raise MacroExpressionError(
            scope.expression, f"{node.func.id}() wants {wanted} arguments, got {len(args)}"
        )
    return _guard(scope, partial(function, *args))


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
    """``{items}[0]`` and ``{data}["mode"]``: one element out of a container.

    The two ways it can fail get their own messages instead of a bare ``KeyError`` inside
    :func:`_guard`: "no key" and "index outside the list" are what the user has to fix,
    and the name of a Python exception is not that.
    """
    if isinstance(node.slice, ast.Slice):
        raise MacroExpressionError(scope.expression, "slices are not allowed")
    value = _evaluate(node.value, scope, depth + 1)
    if not isinstance(value, Mapping | list | tuple | str):
        raise MacroExpressionError(scope.expression, "only a list, a dict or text can be indexed")
    key = _evaluate(node.slice, scope, depth + 1)
    if isinstance(value, Mapping):
        if key not in value:
            raise MacroExpressionError(scope.expression, f"no key {format_value(key)!r}")
        return value[key]
    index = key if isinstance(key, int) and not isinstance(key, bool) else _maybe_number(str(key))
    if not isinstance(index, int):
        raise MacroExpressionError(scope.expression, "an index has to be a whole number")
    if not -len(value) <= index < len(value):
        raise MacroExpressionError(
            scope.expression, f"index {index} is outside a list of {len(value)}"
        )
    return value[index]


def _conditional(node: ast.IfExp, scope: _Scope, depth: int) -> Any:
    """``"тихо" if {volume} < 20 else "громко"``."""
    if truthy(_evaluate(node.test, scope, depth + 1)):
        return _evaluate(node.body, scope, depth + 1)
    return _evaluate(node.orelse, scope, depth + 1)


def _evaluate(node: ast.expr, scope: _Scope, depth: int) -> Any:
    """One node of the whitelist. What is not listed here cannot run at all.

    No attribute, no comprehension, no assignment, no f-string, no ``*args``: each is
    refused by the name of its node type, because a condition that reaches for
    ``{x}.__class__`` is either a mistake worth a message or an attempt worth a no. A
    call is listed, but :func:`_call` lets through only the nine names of
    :data:`FUNCTIONS`.
    """
    if depth > MAX_DEPTH:
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
    if isinstance(node, ast.Call):
        return _call(node, scope, depth)
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
    literals, ``a if c else b``, and a call of one of :data:`FUNCTIONS`. Placeholders are
    looked up through ``resolve`` before parsing and reach the tree as values, never as
    text.

    Raises:
        MacroExpressionError: it does not parse, it uses something refused, or the
            values do not go together.
        MacroReferenceError: a placeholder names something that does not exist.
    """
    if len(expression) > MAX_EXPRESSION:
        shown = f"{expression[:60]}..."
        raise MacroExpressionError(shown, f"longer than {MAX_EXPRESSION} characters")
    text, values = _alias_placeholders(expression, resolve)
    if not text.strip():
        raise MacroExpressionError(expression, "expression is empty")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise MacroExpressionError(expression, "cannot parse") from exc
    return _evaluate(tree.body, _Scope(expression, values, resolve), 0)
