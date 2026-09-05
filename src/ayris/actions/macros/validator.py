"""Everything that can be wrong with a command, in one list with paths.

The models of :mod:`ayris.actions.macros.schema` refuse a file that is not a command.
This module answers the next question — whether a command that *loads* will actually
run — and answers it as a list rather than as an exception: the editor of task 33
paints every problem next to its block, and a save that stopped at the first mistake
would make the user find them one at a time.

**Two severities, one rule for telling them apart.** An error means the command
cannot do what it says: a parameter the action will refuse, a variable nobody sets, a
``Break`` with no loop around it, a call that never returns. A warning means it will
run and something is still worth saying out loud — a block this build knows by name
but has not registered yet, a parameter an action ignores, a command with no trigger.
Only errors stop a save.

**What cannot be known is not reported.** A parameter written as ``{volume}`` is
checked for the reference and not for the type, because its type arrives with the
phrase the user speaks. An unregistered block is an error only when a registry was
given, and an undeclared variable is one only when the caller says which variables
already exist: a plugin of task 26 registers its actions when it loads, and a global
another command wrote is not a broken reference. This is deliberate — a validator
that cries wolf gets switched off, and then it protects nothing.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from pydantic import ValidationError

from ayris.actions.macros.schema import (
    DECLARED_BLOCKS,
    BlockLocation,
    CommandModel,
    LogicBlockSpec,
)
from ayris.core.errors import MacroError
from ayris.core.models import VariableScope
from ayris.nlu.slots import SLOT_PATTERN

if TYPE_CHECKING:
    from ayris.actions.registry import ActionRegistry

__all__ = [
    "MacroValidationError",
    "Problem",
    "Severity",
    "ValidationReport",
    "ensure_valid",
    "validate_command",
    "validate_library",
]


class Severity(StrEnum):
    """Whether the command is broken or merely questionable."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing wrong with a command, at one place in it.

    ``path`` is a :attr:`~ayris.actions.macros.schema.BlockLocation.path_text` —
    ``actions[1].then[0]`` — or one of the command's own fields (``triggers``,
    ``sounds[0]``) for a problem no block owns. ``block`` repeats the block's type so
    a list of problems reads without the tree next to it.
    """

    message: str
    path: str = ""
    severity: Severity = Severity.ERROR
    block: str = ""

    @property
    def text(self) -> str:
        """``actions[0]: у действия «SetVolume» нет параметра «lvl»``."""
        return f"{self.path}: {self.message}" if self.path else self.message


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Everything found in one command, in the order the editor draws the blocks."""

    problems: tuple[Problem, ...] = ()

    @property
    def errors(self) -> tuple[Problem, ...]:
        """The problems that stop a save."""
        return tuple(problem for problem in self.problems if problem.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Problem, ...]:
        """The problems worth showing next to a command that does save."""
        return tuple(problem for problem in self.problems if problem.severity is Severity.WARNING)

    @property
    def ok(self) -> bool:
        """Whether the command can be saved. Warnings do not stop it."""
        return not self.errors

    @property
    def user_message(self) -> str:
        """The first error as one Russian line, or ``""`` when there is none.

        The first rather than all of them: a dialog gets one sentence, and the rest
        are already painted on their blocks.
        """
        errors = self.errors
        return errors[0].text if errors else ""


class MacroValidationError(MacroError):
    """A command that cannot be saved, carrying the whole report.

    Raised only by :func:`ensure_valid`, for the callers that want an exception —
    an import, a plugin loading its own commands. The editor asks for the report.
    """

    default_user_message = "Команда не проходит проверку."

    def __init__(self, report: ValidationReport) -> None:
        problems = "; ".join(problem.text for problem in report.errors)
        super().__init__(
            f"command is not valid: {problems}" if problems else "command is not valid",
            user_message=report.user_message or None,
        )
        self.report = report


def validate_command(
    command: CommandModel,
    *,
    registry: ActionRegistry | None = None,
    library: Mapping[str, CommandModel] | None = None,
    known_variables: Iterable[str] = (),
) -> ValidationReport:
    """Check one command and return every problem found, each with its path.

    Args:
        command: the command to check, already loaded — shape problems are
            :class:`pydantic.ValidationError` and never get this far.
        registry: the action registry. Without it block types and action parameters
            are not checked at all, because nothing here can tell an action that does
            not exist from one whose plugin has not loaded yet.
        library: the other commands, by name, for following ``CallCommand``. Without
            it only a call to a name the command does not have is checked, and a cycle
            through two commands cannot be seen.
        known_variables: names that exist outside this command — the globals and
            profile variables the ``variables`` table already holds. A reference to
            one of those is not a broken reference.
    """
    problems: list[Problem] = []
    locations = list(command.blocks())
    if not command.actions:
        problems.append(
            Problem(
                message="в команде нет ни одного блока: она ничего не сделает",
                path="actions",
                severity=Severity.WARNING,
            )
        )
    if not command.triggers:
        problems.append(
            Problem(
                message="у команды нет триггеров: запустить её можно будет только вручную",
                path="triggers",
                severity=Severity.WARNING,
            )
        )
    _check_blocks(locations, registry, problems)
    _check_references(command, locations, frozenset(known_variables), problems)
    _check_calls(command, locations, library, problems)
    return ValidationReport(problems=_ordered(problems, locations))


def _ordered(
    problems: list[Problem],
    locations: Sequence[BlockLocation],
) -> tuple[Problem, ...]:
    """The problems in the order the blocks are drawn, not the order they were found.

    The checks run one after another over the whole tree, so without this a problem
    about the last block's parameters would come before a reference problem in the
    first. The editor paints by path and would not care; a person reading the list
    under the command would.
    """
    order = {location.path_text: index for index, location in enumerate(locations)}
    return tuple(sorted(problems, key=lambda problem: order.get(_block_path(problem.path), -1)))


def _block_path(path: str) -> str:
    """The block's own path out of a problem's path: ``actions[0].params.text`` → ``actions[0]``."""
    for suffix in (".params", ".sound"):
        head, found, _ = path.partition(suffix)
        if found:
            return head
    return path


def ensure_valid(
    command: CommandModel,
    *,
    registry: ActionRegistry | None = None,
    library: Mapping[str, CommandModel] | None = None,
    known_variables: Iterable[str] = (),
) -> ValidationReport:
    """:func:`validate_command`, raising instead of returning errors.

    Returns:
        The report, so warnings survive a successful check and can be shown.

    Raises:
        MacroValidationError: the command has at least one error.
    """
    report = validate_command(
        command,
        registry=registry,
        library=library,
        known_variables=known_variables,
    )
    if not report.ok:
        raise MacroValidationError(report)
    return report


def validate_library(
    commands: Iterable[CommandModel],
    *,
    registry: ActionRegistry | None = None,
) -> dict[str, ValidationReport]:
    """Check commands together: one report per command, by name.

    Together rather than one at a time, because two things only exist between
    commands — a ``CallCommand`` cycle, and a global one command declares and another
    reads. Both are invisible to a check of a single command.
    """
    listed = list(commands)
    library = {command.name: command for command in listed}
    known = _library_variables(listed)
    return {
        command.name: validate_command(
            command,
            registry=registry,
            library=library,
            known_variables=known,
        )
        for command in listed
    }


def _at(location: BlockLocation, message: str, *, severity: Severity = Severity.ERROR) -> Problem:
    """A problem pointing at one block."""
    return Problem(
        message=message,
        path=location.path_text,
        severity=severity,
        block=location.block.type,
    )


#: Blocks a ``Break`` or a ``Continue`` needs above it somewhere.
_LOOPS: Final[frozenset[str]] = frozenset({"While", "For"})

#: Blocks that are the arms of a ``Switch`` and mean nothing anywhere else.
_ARMS: Final[frozenset[str]] = frozenset({"Case", "Default"})

_JUMPS: Final[frozenset[str]] = frozenset({"Break", "Continue"})


def _check_blocks(
    locations: Sequence[BlockLocation],
    registry: ActionRegistry | None,
    problems: list[Problem],
) -> None:
    """Every block: does it exist, do its parameters fit, is it in a legal place."""
    types = {location.path: location.block.type for location in locations}
    for location in locations:
        block = location.block
        spec = block.spec
        if spec is not None:
            _check_logic_params(spec, location, problems)
            _check_branches(spec, location, problems)
            _check_placement(location, _ancestors(location.path, types), problems)
        elif registry is None:
            continue
        elif registry.has(block.type):
            _check_action_params(registry, location, problems)
        elif block.type in DECLARED_BLOCKS:
            problems.append(
                _at(
                    location,
                    f"действие «{block.type}» описано в ТЗ, но эта сборка его ещё не умеет",
                    severity=Severity.WARNING,
                )
            )
        else:
            problems.append(
                _at(location, f"действия «{block.type}» нет ни в реестре, ни среди блоков логики")
            )


def _ancestors(
    path: tuple[str | int, ...],
    types: Mapping[tuple[str | int, ...], str],
) -> tuple[str, ...]:
    """Types of the blocks this one sits inside, nearest first.

    Read off the paths instead of passed down the walk: :func:`walk_blocks` yields
    parents before children, so by the time a block is looked at its whole chain is
    already in the map, and one loop over a flat list does what a second recursion
    would.
    """
    names: list[str] = []
    walk = path[:-2]
    while walk:
        name = types.get(walk)
        if name is None:
            break
        names.append(name)
        walk = walk[:-2]
    return tuple(names)


def _check_placement(
    location: BlockLocation,
    ancestors: tuple[str, ...],
    problems: list[Problem],
) -> None:
    """Blocks that only mean something inside another block."""
    kind = location.block.type
    if kind in _ARMS and (not ancestors or ancestors[0] != "Switch"):
        problems.append(_at(location, f"блок «{kind}» бывает только внутри Switch"))
    if kind in _JUMPS and not _LOOPS.intersection(ancestors):
        problems.append(
            _at(location, f"блок «{kind}» стоит вне цикла: ни While, ни For над ним нет")
        )


def _check_logic_params(
    spec: LogicBlockSpec,
    location: BlockLocation,
    problems: list[Problem],
) -> None:
    """Parameters of a logic block, against the table that describes it.

    A logic block has no ``Params`` model to validate against — its parameters are
    expressions and variable names, not typed values — so the check is presence and
    spelling. Types are the interpreter's problem at run time, where a condition can
    actually be evaluated.
    """
    params = location.block.params
    for name in spec.required_params:
        if _blank(params.get(name)):
            problems.append(_at(location, f"блоку «{spec.name}» нужен параметр «{name}»"))
    for name in sorted(set(params) - spec.params):
        problems.append(
            _at(
                location,
                f"параметр «{name}» блоку «{spec.name}» ничего не говорит",
                severity=Severity.WARNING,
            )
        )
    if spec.name == "For" and not _loop_range(params):
        problems.append(
            _at(location, "блоку «For» нужен либо список items, либо границы from и to")
        )


def _blank(value: object) -> bool:
    """Whether a parameter is absent or an empty string. ``False`` and ``0`` are values."""
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _loop_range(params: Mapping[str, Any]) -> bool:
    """Whether a ``For`` says what to iterate over."""
    if not _blank(params.get("items")):
        return True
    return not _blank(params.get("from")) and not _blank(params.get("to"))


def _check_branches(
    spec: LogicBlockSpec,
    location: BlockLocation,
    problems: list[Problem],
) -> None:
    """Branches the block cannot work without. Which branches exist is the model's job."""
    filled = {name for name, blocks in location.block.branches() if blocks}
    for name in spec.required_branches:
        if name not in filled:
            problems.append(
                _at(location, f"у блока «{spec.name}» пустая ветка {name}: выполнять нечего")
            )


#: Pydantic error codes said in Russian. What is not here keeps the pydantic message
#: behind a Russian prefix: a wrong translation is worse than an English sentence.
_PARAM_HINTS: Final[Mapping[str, str]] = {
    "bool_parsing": "нужно да или нет",
    "bool_type": "нужно да или нет",
    "int_parsing": "нужно целое число",
    "int_type": "нужно целое число",
    "float_parsing": "нужно число",
    "float_type": "нужно число",
    "string_type": "нужна строка",
    "list_type": "нужен список",
    "dict_type": "нужен словарь",
    "greater_than": "значение слишком мало",
    "greater_than_equal": "значение слишком мало",
    "less_than": "значение слишком велико",
    "less_than_equal": "значение слишком велико",
    "string_too_short": "значение пустое",
    "string_too_long": "значение слишком длинное",
    "enum": "значение не из списка допустимых",
    "literal_error": "значение не из списка допустимых",
}


def _check_action_params(
    registry: ActionRegistry,
    location: BlockLocation,
    problems: list[Problem],
) -> None:
    """The block's parameters against the ``Params`` model of its action.

    Parameters holding a placeholder are taken out first: ``{"level": "{volume}"}`` is
    a string where the action wants a number, and it *will* be a number once the
    phrase is matched. Removing them then looks like a parameter left out, so a
    ``missing`` complaint about exactly those names is dropped as well — the
    reference itself is checked by :func:`_check_references`.
    """
    block = location.block
    model = registry.get(block.type).params_model()
    deferred = {name for name, value in block.params.items() if _has_placeholder(value)}
    given = {name: value for name, value in block.params.items() if name not in deferred}
    try:
        model.model_validate(given)
    except ValidationError as exc:
        for error in exc.errors():
            name = str(error["loc"][0]) if error["loc"] else ""
            kind = str(error["type"])
            if kind == "missing" and name in deferred:
                continue
            problems.append(
                _at(location, _param_message(block.type, name, kind, str(error["msg"])))
            )
    except ValueError as exc:  # a Params model with a validator that raises its own
        problems.append(
            _at(location, f"параметры действия «{block.type}» не проходят проверку: {exc}")
        )


def _param_message(block_type: str, name: str, kind: str, detail: str) -> str:
    """One pydantic error as one Russian line about one parameter."""
    if kind == "missing":
        return f"действию «{block_type}» нужен параметр «{name}»"
    if kind == "extra_forbidden":
        return f"у действия «{block_type}» нет параметра «{name}»"
    hint = _PARAM_HINTS.get(kind)
    if hint is not None:
        return f"параметр «{name}»: {hint}"
    return f"параметр «{name}» не подходит: {detail}"


def _has_placeholder(value: Any) -> bool:
    """Whether a parameter mentions a variable anywhere inside it."""
    return any(SLOT_PATTERN.search(text) for _, text in _strings(value, ""))


#: Blocks that create or overwrite a variable, and the parameter holding its name.
#: A macro may set a variable and read it two blocks later, so what these write
#: counts as a declaration exactly as the ``variables`` list does.
_WRITERS: Final[Mapping[str, tuple[str, ...]]] = {
    "SetVar": ("name",),
    "GetVar": ("into",),
    "For": ("var",),
    "Try": ("error_var",),
    "ArrayPush": ("name",),
    "ArrayGet": ("into",),
    "DictSet": ("name",),
    "DictGet": ("into",),
}

#: Blocks that name a variable in a parameter instead of in braces. Reading one
#: nobody wrote is the same mistake as ``{nobody}``, and a search for placeholders
#: cannot see it.
_READERS: Final[Mapping[str, tuple[str, ...]]] = {
    "GetVar": ("name",),
    "ArrayGet": ("name",),
    "DictGet": ("name",),
}


def _check_references(
    command: CommandModel,
    locations: Sequence[BlockLocation],
    known: frozenset[str],
    problems: list[Problem],
) -> None:
    """Every ``{name}`` in the command, against everything that could fill one."""
    available = command.variable_names | command.slot_names | known | _assigned(locations)
    for index, sound in enumerate(command.sounds):
        _check_text(sound.value, f"sounds[{index}]", "", available, problems)
    for location in locations:
        block = location.block
        base = location.path_text
        for path, text in _strings(block.params, f"{base}.params"):
            _check_text(text, path, block.type, available, problems)
        if block.sound is not None:
            _check_text(block.sound.value, f"{base}.sound", block.type, available, problems)
        for name in _READERS.get(block.type, ()):
            value = block.params.get(name)
            if not isinstance(value, str) or not value.strip() or _has_placeholder(value):
                continue
            if value not in available:
                problems.append(
                    _at(
                        location,
                        f"блок «{block.type}» читает переменную «{value}», "
                        "которую никто не объявил и не задал",
                    )
                )


def _check_text(
    text: str,
    path: str,
    block: str,
    available: frozenset[str],
    problems: list[Problem],
) -> None:
    """One string's placeholders, each of which needs somewhere to come from."""
    for found in SLOT_PATTERN.finditer(text):
        name = found["name"]
        if name in available:
            continue
        problems.append(
            Problem(
                message=f"ссылка на «{name}»: нет ни такой переменной, ни такого слота",
                path=path,
                block=block,
            )
        )


def _strings(value: Any, prefix: str) -> Iterator[tuple[str, str]]:
    """Every string inside a parameter value, with the path that leads to it.

    Parameters are not always strings: an action may take a list of keys or a
    mapping of headers, and a reference hiding in one of those is still a reference.
    """
    if isinstance(value, str):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(item, f"{prefix}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            yield from _strings(item, f"{prefix}[{index}]")


def _assigned(locations: Sequence[BlockLocation]) -> frozenset[str]:
    """Names the command writes itself. A computed name is not one of them."""
    names: set[str] = set()
    for location in locations:
        for key in _WRITERS.get(location.block.type, ()):
            value = location.block.params.get(key)
            if isinstance(value, str) and value.strip() and not SLOT_PATTERN.search(value):
                names.add(value)
    return frozenset(names)


def _check_calls(
    command: CommandModel,
    locations: Sequence[BlockLocation],
    library: Mapping[str, CommandModel] | None,
    problems: list[Problem],
) -> None:
    """``CallCommand``: is there something to call, and does the call come back.

    A command calling itself is an error without any library at all — that one needs
    no graph. Everything longer needs the other commands, and until the caller hands
    them over a call to a name is just a name.
    """
    for location in locations:
        block = location.block
        spec = block.spec
        if spec is None or not spec.is_call:
            continue
        target = block.params.get("command")
        if not isinstance(target, str) or not target.strip():
            continue  # already reported as the missing parameter it is
        if _has_placeholder(target):
            continue  # which command runs is only known once the variable has a value
        if target == command.name:
            problems.append(_at(location, f"команда «{target}» вызывает саму себя"))
            continue
        if library is None:
            continue
        if target not in library:
            problems.append(_at(location, f"команды «{target}» нет: вызывать нечего"))
            continue
        chain = _find_cycle(command.name, target, library)
        if chain is not None:
            problems.append(_at(location, f"вызовы ходят по кругу: {' → '.join(chain)}"))


def _find_cycle(
    origin: str,
    target: str,
    library: Mapping[str, CommandModel],
) -> tuple[str, ...] | None:
    """The chain of calls leading from ``target`` back to ``origin``, or ``None``.

    Iterative and with a visited set, because the graph being walked may already be a
    cycle: a recursive walk of one ends in a ``RecursionError`` instead of a line in a
    report, and the report is the whole point.
    """
    stack: list[tuple[str, tuple[str, ...]]] = [(target, (origin, target))]
    seen: set[str] = set()
    while stack:
        name, chain = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        entry = library.get(name)
        if entry is None:
            continue
        for called in _call_targets(entry):
            if called == origin:
                return (*chain, origin)
            stack.append((called, (*chain, called)))
    return None


def _call_targets(command: CommandModel) -> Iterator[str]:
    """Names of the commands this one calls, computed ones left out."""
    for location in command.blocks():
        block = location.block
        spec = block.spec
        if spec is None or not spec.is_call:
            continue
        target = block.params.get("command")
        if isinstance(target, str) and target.strip() and not _has_placeholder(target):
            yield target


def _library_variables(commands: Sequence[CommandModel]) -> frozenset[str]:
    """Variables that outlive one command: declared, or written with a lasting scope.

    A global is a global from wherever it was set, so a command reading one another
    command writes is not reading a name nobody has.
    """
    names: set[str] = set()
    for command in commands:
        names.update(
            declared.name
            for declared in command.variables
            if declared.scope is not VariableScope.LOCAL
        )
        for location in command.blocks():
            block = location.block
            if block.type != "SetVar":
                continue
            scope = block.params.get("scope", VariableScope.LOCAL.value)
            value = block.params.get("name")
            if scope != VariableScope.LOCAL.value and isinstance(value, str) and value.strip():
                names.add(value)
    return frozenset(names)
