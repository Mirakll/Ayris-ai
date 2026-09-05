"""What can go wrong while a command runs, as typed exceptions.

:mod:`ayris.core.errors` keeps one class for this whole subsystem —
:class:`~ayris.core.errors.MacroError` — which is enough to say "the macro broke"
and not enough to say "block ``actions[1].then[0]`` broke because the action
refused its parameters, and its policy says stop the chain". Task 30 split the
definition side off the same way with ``MacroValidationError``; this module is its
run-time twin, and everything the interpreter raises lives here.

Two things are deliberately not errors and are not in this module. A command
skipped by its cooldown, and a command that finished with a failed step under
``on_error = continue``, are both outcomes of an ordinary run and are reported
through :class:`~ayris.actions.macros.report.ExecutionReport`: raising for them
would put a ``try`` around the ordinary case in every caller.

Cancellation is the borderline case. It is an exception here, because unwinding a
tree walk halfway is what exceptions are for, but the engine catches it at the top
and turns it into an outcome — nothing above the engine sees
:class:`MacroCancelledError` unless it asks the report to re-raise.
"""

from __future__ import annotations

from ayris.core.errors import MacroError

__all__ = [
    "MacroBlockError",
    "MacroCallError",
    "MacroCancelledError",
    "MacroEngineStoppedError",
    "MacroExpressionError",
    "MacroLimitError",
    "MacroReferenceError",
    "MacroRuntimeError",
    "MacroTimeoutError",
    "MacroValueError",
]


class MacroRuntimeError(MacroError):
    """Base for every failure raised while a command is being executed."""

    default_user_message = "Не удалось выполнить команду."


class MacroBlockError(MacroRuntimeError):
    """One block failed, together with the place in the tree where it failed.

    ``path`` is the reader's spelling of that place — ``actions[1].then[0]``, the
    string :attr:`~ayris.actions.macros.schema.BlockLocation.path_text` produces —
    so the editor of task 33 can select the block that broke and a log line points
    at it without anyone counting brackets.

    ``cause`` keeps the original exception even after the traceback is gone: the
    report crosses a thread boundary and is usually read later, by the history tab
    or by the debugger, when the stack it came from no longer exists.
    """

    default_user_message = "Шаг команды не выполнился."

    def __init__(
        self,
        technical: str,
        *,
        path: str = "",
        block: str = "",
        user_message: str | None = None,
        recoverable: bool = True,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(technical, user_message=user_message, recoverable=recoverable)
        self.path = path
        self.block = block
        self.cause = cause

    def __str__(self) -> str:
        return f"{self.path}: {self.technical}" if self.path else self.technical


class MacroReferenceError(MacroRuntimeError):
    """A ``{placeholder}`` names something that does not exist.

    A typo in a variable name is the commonest authoring mistake, and the validator
    catches it when the command is saved. This is the same mistake reaching run time
    anyway: a slot the phrase did not fill, a global deleted between two runs, a
    command imported from another machine. The only useful answer is which name.
    """

    default_user_message = "В команде есть ссылка на неизвестное имя."

    def __init__(self, name: str, *, user_message: str | None = None) -> None:
        super().__init__(
            f"unknown name in a placeholder: {name!r}",
            user_message=user_message or f"Неизвестное имя в команде: {name}.",
        )
        self.name = name


class MacroExpressionError(MacroRuntimeError):
    """A condition or an expression cannot be evaluated.

    Covers all three ways that happens: it does not parse, it uses something the
    evaluator refuses to run — a call, an attribute, a comprehension — or the values
    do not go together, as in ``"тихо" > 0``.
    """

    default_user_message = "Не удалось вычислить условие в команде."

    def __init__(self, expression: str, technical: str, *, user_message: str | None = None) -> None:
        super().__init__(f"{technical}: {expression!r}", user_message=user_message)
        self.expression = expression


class MacroValueError(MacroRuntimeError):
    """A value does not fit the declared type of the variable it is written to."""

    default_user_message = "Значение переменной не подходит по типу."

    def __init__(self, name: str, value: object, expected: str) -> None:
        shown = str(value)
        if len(shown) > 60:
            shown = f"{shown[:57]}..."
        super().__init__(
            f"cannot store {value!r} in {name!r} declared as {expected}",
            user_message=f"Значение «{shown}» не подходит переменной {name} ({expected}).",
        )
        self.name = name
        self.value = value
        self.expected = expected


class MacroLimitError(MacroRuntimeError):
    """A safety limit stopped the run: nesting, iterations, steps or call depth.

    An endless ``While`` is a bug in the command rather than in Ayris, and the honest
    answer is to stop and name the limit that hit — a run that never ends looks to
    the user like an assistant that froze.
    """

    default_user_message = "Команда остановлена: превышен предел выполнения."

    def __init__(
        self,
        limit: str,
        value: int,
        *,
        technical: str = "",
        user_message: str | None = None,
    ) -> None:
        super().__init__(
            technical or f"macro limit {limit} exceeded: {value}",
            user_message=user_message,
        )
        self.limit = limit
        self.value = value


class MacroTimeoutError(MacroLimitError):
    """The command as a whole ran longer than the budget it was given.

    A limit like the others, and a subclass so that "stop, a limit hit" needs one
    ``except`` rather than two, while the history can still record ``timeout``
    separately from ``error``.
    """

    default_user_message = "Команда выполнялась слишком долго и была остановлена."

    def __init__(self, timeout_s: float) -> None:
        super().__init__(
            "timeout",
            int(timeout_s * 1000),
            technical=f"command timed out after {timeout_s:g} s",
        )
        self.timeout_s = timeout_s


class MacroCancelledError(MacroRuntimeError):
    """The run was cancelled: the stop word, the hotkey, or a command with priority.

    Not a failure to show the user — cancelling is what they asked for. ``reason`` is
    what :class:`~ayris.core.events.CancelRequested` carried, or the name of the
    command that preempted this one.
    """

    default_user_message = "Команда отменена."

    def __init__(self, reason: str = "") -> None:
        super().__init__(f"macro cancelled: {reason}" if reason else "macro cancelled")
        self.reason = reason


class MacroCallError(MacroRuntimeError):
    """``CallCommand`` cannot be made: no such command, or it is switched off.

    The validator follows calls when it is given the library, so this is the run-time
    remainder: a command renamed or deleted after the caller was saved.
    """

    default_user_message = "Вложенная команда не найдена."

    def __init__(self, command: str, technical: str = "") -> None:
        super().__init__(
            technical or f"cannot call command {command!r}",
            user_message=f"Не удалось вызвать команду «{command}».",
        )
        self.command = command


class MacroEngineStoppedError(MacroRuntimeError):
    """The engine is shut down and will not take a new run.

    Happens while the application is closing, when a hotkey or a timer fires after
    the pool has been told to stop. Raised instead of the bare ``RuntimeError`` a
    dead :class:`concurrent.futures.ThreadPoolExecutor` would give.
    """

    default_user_message = "Движок макросов остановлен."
