"""Data blocks: the nine ways a command reads and writes its own variables.

Table 7.2 of the specification lists six — ``SetVar``, ``GetVar``, ``ArrayPush``,
``ArrayGet``, ``DictSet``, ``DictGet`` — and this module adds the three that fall out of
the same code: ``ArrayPop``, ``ArrayLength``, ``DictKeys``. A command that pushes onto an
array needs to take from it, and a ``While`` over a queue needs its length; both would
otherwise be written as arithmetic on a condition, which is worse and no safer.

None of the writing happens here. :class:`~ayris.actions.macros.context.ExecutionContext`
owns the scopes, the declared types and the lock behind the shared ones, so these blocks
are a name, a value and a note for the report — the interesting part is which of the
context's methods each one is. ``ArrayPush``, ``DictSet`` and ``ArrayPop`` go through the
read-change-write methods rather than a read and a write of their own, because two runs
pushing onto the same global array at the same moment must both survive.

**Why a value is substituted and not evaluated.** ``SetVar`` with ``value: "{volume} + 10"``
writes the text ``50 + 10``. Section 22 asks for arithmetic inside a condition, where the
expression engine is; a ``SetVar`` that quietly evaluated its value would make every string
a command stores an expression, and text a user dictated is not one.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from ayris.actions.macros.blocks.logic import BlockHandler, Flow, as_int, short
from ayris.actions.macros.errors import MacroBlockError
from ayris.actions.macros.expressions import format_value
from ayris.core.models import VariableScope

if TYPE_CHECKING:
    from ayris.actions.macros.blocks.logic import BlockRuntime
    from ayris.actions.macros.schema import ActionBlock

__all__ = [
    "VARIABLE_HANDLERS",
    "run_array_get",
    "run_array_length",
    "run_array_pop",
    "run_array_push",
    "run_dict_get",
    "run_dict_keys",
    "run_dict_set",
    "run_get_var",
    "run_set_var",
]


def variable_name(rt: BlockRuntime, block: ActionBlock) -> str:
    """The variable a block works on, placeholders filled.

    Raises:
        MacroBlockError: the block names no variable at all.
    """
    name = format_value(rt.context.fill(block.params.get("name", ""))).strip()
    if not name:
        raise MacroBlockError(f"{block.type} without a variable name", block=block.type)
    return name


def store_read(rt: BlockRuntime, block: ActionBlock, value: Any) -> None:
    """Put what a reading block read into ``into``, when it names one, and last_result."""
    into = format_value(rt.context.fill(block.params.get("into", ""))).strip()
    if into:
        rt.context.set(into, value)
    rt.context.set_result(value)


def as_scope(value: Any) -> VariableScope | None:
    """A ``scope`` parameter as a scope, or ``None`` when the block does not say.

    Raises:
        MacroBlockError: the block names a scope that does not exist.
    """
    if value is None or value == "":
        return None
    if isinstance(value, VariableScope):
        return value
    text = format_value(value).strip().lower()
    try:
        return VariableScope(text)
    except ValueError as exc:
        raise MacroBlockError(
            f"unknown variable scope {text!r}",
            user_message=f"Неизвестная область видимости переменной: {text}.",
        ) from exc


def element(name: str, items: Any, index: int) -> Any:
    """One element of an array variable, or a failure that names the index.

    Raises:
        MacroBlockError: the variable is not an array, or the index is outside it.
    """
    if not isinstance(items, list | tuple):
        raise MacroBlockError(
            f"{name!r} is not an array", user_message=f"Переменная {name} — не массив."
        )
    if -len(items) <= index < len(items):
        return items[index]
    raise MacroBlockError(
        f"index {index} is outside {name!r} of {len(items)}",
        user_message=f"В массиве {name} нет элемента с номером {index}.",
    )


def member(name: str, mapping: Any, key: str) -> Any:
    """One key of a dictionary variable, or a failure that names the key.

    Raises:
        MacroBlockError: the variable is not a dictionary, or it has no such key.
    """
    if not isinstance(mapping, Mapping):
        raise MacroBlockError(
            f"{name!r} is not a dictionary", user_message=f"Переменная {name} — не словарь."
        )
    if key in mapping:
        return mapping[key]
    raise MacroBlockError(
        f"{name!r} has no key {key!r}", user_message=f"В словаре {name} нет ключа «{key}»."
    )


def run_set_var(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``SetVar``: fill the placeholders in the value, then write it where it belongs.

    The scope is the block's when it names one and the variable's declared scope otherwise;
    an undeclared name with no scope becomes a local, so a typo cannot quietly create a
    global that outlives the mistake.

    Raises:
        MacroValueError: the value does not fit the declared type of the variable.
    """
    name = variable_name(rt, block)
    scope = as_scope(block.params.get("scope"))
    value = rt.context.set(name, rt.context.fill(block.params.get("value")), scope)
    rt.context.set_result(value)
    rt.report.note(f"{name} = {short(value)}")
    return Flow.NEXT


def run_get_var(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``GetVar``: read a variable into ``last_result``, and into ``into`` when given.

    Raises:
        MacroReferenceError: nothing of that name exists in any scope.
    """
    name = variable_name(rt, block)
    value = rt.context.resolve(name)
    store_read(rt, block, value)
    rt.report.note(f"{name} → {short(value)}")
    return Flow.NEXT


def run_array_push(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``ArrayPush``: add one element, atomically for a shared array."""
    name = variable_name(rt, block)
    items = rt.context.append(name, rt.context.fill(block.params.get("value")))
    rt.context.set_result(items)
    rt.report.note(f"{name}: {len(items)} элементов")
    return Flow.NEXT


def run_array_get(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``ArrayGet``: one element by index, negative indexes counted from the end."""
    name = variable_name(rt, block)
    index = as_int(rt.context.fill(block.params.get("index")))
    value = element(name, rt.context.resolve(name), index)
    store_read(rt, block, value)
    rt.report.note(f"{name}[{index}] → {short(value)}")
    return Flow.NEXT


def run_array_pop(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``ArrayPop``: take one element out and read it, in one step.

    The last element by default, or the one at ``index``. One step and not a read followed
    by a write, because another run may push onto the same array in between and the element
    this block reported would then not be the element it removed.

    Raises:
        MacroIndexError: the array is empty, or the index is outside it.
    """
    name = variable_name(rt, block)
    raw = rt.context.fill(block.params.get("index"))
    index = None if raw is None or raw == "" else as_int(raw)
    value = rt.context.pop(name, index)
    store_read(rt, block, value)
    rt.report.note(f"{name} → {short(value)}")
    return Flow.NEXT


def run_array_length(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``ArrayLength``: how many elements an array holds.

    Raises:
        MacroBlockError: the variable is not an array.
    """
    name = variable_name(rt, block)
    items = rt.context.resolve(name)
    if not isinstance(items, list | tuple):
        raise MacroBlockError(
            f"{name!r} is not an array", user_message=f"Переменная {name} — не массив."
        )
    store_read(rt, block, len(items))
    rt.report.note(f"{name}: {len(items)} элементов")
    return Flow.NEXT


def run_dict_set(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``DictSet``: write one key, atomically for a shared dictionary."""
    name = variable_name(rt, block)
    key = format_value(rt.context.fill(block.params.get("key", "")))
    value = rt.context.fill(block.params.get("value"))
    rt.context.set_result(rt.context.put(name, key, value))
    rt.report.note(f"{name}[{key}] = {short(value)}")
    return Flow.NEXT


def run_dict_get(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``DictGet``: one key of a dictionary variable."""
    name = variable_name(rt, block)
    key = format_value(rt.context.fill(block.params.get("key", "")))
    value = member(name, rt.context.resolve(name), key)
    store_read(rt, block, value)
    rt.report.note(f"{name}[{key}] → {short(value)}")
    return Flow.NEXT


def run_dict_keys(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """``DictKeys``: the keys of a dictionary as an array, for a ``For`` to walk.

    Raises:
        MacroBlockError: the variable is not a dictionary.
    """
    name = variable_name(rt, block)
    mapping = rt.context.resolve(name)
    if not isinstance(mapping, Mapping):
        raise MacroBlockError(
            f"{name!r} is not a dictionary", user_message=f"Переменная {name} — не словарь."
        )
    keys = [format_value(key) for key in mapping]
    store_read(rt, block, keys)
    rt.report.note(f"{name}: {len(keys)} ключей")
    return Flow.NEXT


#: Every data block by the name a ``.ayris`` file spells.
VARIABLE_HANDLERS: Final[dict[str, BlockHandler]] = {
    "SetVar": run_set_var,
    "GetVar": run_get_var,
    "ArrayPush": run_array_push,
    "ArrayGet": run_array_get,
    "ArrayPop": run_array_pop,
    "ArrayLength": run_array_length,
    "DictSet": run_dict_set,
    "DictGet": run_dict_get,
    "DictKeys": run_dict_keys,
}
