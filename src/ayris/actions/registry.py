"""The one way to run an action, and the only place that knows they all exist.

Invariant 5 of the architecture: NLU, macros and plugins call actions *through*
this module, never by importing an action class. That buys three things at once.
A macro exported on one machine runs on another, because it names actions by
string. The editor can list and describe every block without importing WinAPI.
And the crosscutting work — validating parameters, checking rights, enforcing a
timeout, publishing events, writing the audit row, masking secrets — is written
once here instead of seventy times in the actions.

**Registration is a mark, not a side effect.** ``@register`` records the class in
a module-level table; a registry picks classes up from that table when it is
built. So a test gets a fresh :class:`ActionRegistry` over the *real* actions
without a global to reset, and two registries in one process — the app's and a
plugin sandbox's — do not fight over one dictionary.

**Autodiscovery imports, it does not scan.** :meth:`ActionRegistry.discover`
walks ``ayris.actions.system`` with ``pkgutil`` and imports each module; the
decorators inside do the rest. A new action file is therefore picked up with no
edit anywhere else. Modules from tasks 20-29 import WinAPI, pycaw and WinRT, so
discovery tolerates an :exc:`ImportError` from one module — on Linux, or on a
machine without ``pycaw`` — logs it, and keeps the rest of the library usable
instead of taking the whole registry down with it.

**Nothing bare escapes.** :meth:`ActionRegistry.execute` converts every failure
into a subclass of :class:`~ayris.core.errors.ActionError`: a missing name, a
rejected parameter, a refused elevation, an expired timeout, or whatever the
action itself raised. The pipeline already relies on that (task 18) and catches
nothing else.

**A timed-out synchronous action is not forgotten.** ``asyncio.wait_for`` and
``Future.result(timeout=...)`` both give up waiting without stopping the work: a
``SetForegroundWindow`` that wedged holds its thread forever. The registry counts
those threads (:attr:`ActionRegistry.stuck_workers`), keeps them out of the pool's
capacity, and refuses new synchronous calls with
:class:`~ayris.core.errors.ActionUnavailable` once none are left — a loud failure
instead of an ``execute`` that blocks behind a dead thread and looks like a hung
assistant.
"""

from __future__ import annotations

import importlib
import pkgutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Final, TypeVar

from pydantic import ValidationError

from ayris.actions.base import (
    Action,
    ActionCategory,
    ActionSchema,
    build_schema,
    mask_params,
)
from ayris.actions.result import ActionResult
from ayris.core.errors import (
    ActionError,
    ActionNotFound,
    ActionParamsInvalid,
    ActionRequiresAdmin,
    ActionTimeout,
    ActionUnavailable,
    AyrisError,
    ParamProblem,
)
from ayris.core.events import ActionFailed, ActionFinished, ActionStarted
from ayris.core.models import AuditEntry, ExecutionResult
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from ayris.actions.base import ActionParams
    from ayris.core.events import EventBus
    from ayris.core.repositories import AuditRepository

__all__ = [
    "SYSTEM_PACKAGE",
    "ActionRegistry",
    "AuditSink",
    "RegisteredAction",
    "register",
    "registered_actions",
]

_log = get_logger(__name__)

#: Package autodiscovery walks. Every module under it is imported for its side
#: effect of declaring actions.
SYSTEM_PACKAGE: Final = "ayris.actions.system"

#: How many synchronous actions may run at once. Section 12 lets the user size the
#: macro thread pool; this is the floor the registry keeps for itself.
DEFAULT_MAX_WORKERS: Final = 4

ActionT = TypeVar("ActionT", bound=Action)


class AuditSink:
    """What the registry needs from the audit trail.

    A protocol in all but name, satisfied by
    :class:`ayris.core.repositories.AuditRepository`. Declared as a class so the
    dependency points at storage, not at SQLite.
    """

    def add(self, entry: AuditEntry) -> AuditEntry:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class RegisteredAction:
    """One action class as the module-level table holds it."""

    action_class: type[Action]
    plugin: str = ""

    @property
    def name(self) -> str:
        """Registered name, plugin prefix included."""
        return self.action_class.meta.with_prefix(self.plugin).name


_MARKED: dict[str, RegisteredAction] = {}
_MARKED_LOCK: Final = Lock()


def register(action_class: type[ActionT]) -> type[ActionT]:
    """Mark ``action_class`` as an action of the built-in library.

    Validates the declaration at import time — metadata present, name unique, one
    of ``run``/``arun`` implemented, ``Params`` a model — because the alternative
    is discovering a typo the first time a user says the phrase.

    Raises:
        TypeError: The class is not a usable action.
        ValueError: Another action already claims that name.
    """
    _validate_class(action_class)
    entry = RegisteredAction(action_class=action_class)
    with _MARKED_LOCK:
        existing = _MARKED.get(entry.name)
        if existing is not None and existing.action_class is not action_class:
            raise ValueError(
                f"action name {entry.name!r} is already used by "
                f"{existing.action_class.__module__}.{existing.action_class.__qualname__}"
            )
        _MARKED[entry.name] = entry
    return action_class


def registered_actions() -> tuple[RegisteredAction, ...]:
    """Every class marked with :func:`register`, in declaration order."""
    with _MARKED_LOCK:
        return tuple(_MARKED.values())


def _validate_class(action_class: type[Action]) -> None:
    if not isinstance(action_class, type) or not issubclass(action_class, Action):
        raise TypeError(f"{action_class!r} is not a subclass of Action")
    meta = getattr(action_class, "meta", None)
    if meta is None or not isinstance(getattr(meta, "name", None), str):
        raise TypeError(f"{action_class.__qualname__} has no ActionMeta in 'meta'")
    if not (action_class.implements_sync() or action_class.implements_async()):
        raise TypeError(f"{action_class.__qualname__} implements neither run() nor arun()")
    if meta.supports_undo and action_class.undo is Action.undo:
        raise TypeError(
            f"{action_class.__qualname__} declares supports_undo but does not override undo()"
        )


# --------------------------------------------------------------------------- #
# Thread pool that remembers what wedged
# --------------------------------------------------------------------------- #


class _WorkerPool:
    """Thread pool for synchronous actions, with the stuck ones subtracted.

    ``Future.result(timeout=...)`` returning control does not mean the work
    stopped. Every future the registry gives up on is counted here until it
    actually finishes, so capacity reflects the threads that are really free.
    """

    def __init__(self, max_workers: int) -> None:
        self._max = max(1, max_workers)
        self._pool = ThreadPoolExecutor(max_workers=self._max, thread_name_prefix="ayris-action")
        self._lock = Lock()
        self._stuck = 0

    @property
    def stuck(self) -> int:
        """Threads still inside an action the registry stopped waiting for."""
        with self._lock:
            return self._stuck

    @property
    def capacity(self) -> int:
        """Threads that can still take work."""
        with self._lock:
            return self._max - self._stuck

    def run(
        self, work: Callable[[], ActionResult[Any]], timeout_s: float | None
    ) -> ActionResult[Any]:
        """Run ``work`` on a worker thread, waiting at most ``timeout_s``.

        Raises:
            ActionTimeout: The wait expired.
            ActionUnavailable: Every worker thread is wedged in an earlier action.
        """
        if self.capacity <= 0:
            raise ActionUnavailable(
                f"all {self._max} action workers are stuck in earlier calls",
                user_message="Предыдущее действие не отвечает. Перезапустите Ayris.",
            )
        future = self._pool.submit(work)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeout as exc:
            if not future.cancel():
                self._mark_stuck(future)
            raise ActionTimeout("action exceeded its timeout") from exc

    def _mark_stuck(self, future: Future[ActionResult[Any]]) -> None:
        with self._lock:
            self._stuck += 1
            stuck = self._stuck
        _log.warning(
            "action worker still running after timeout; %d of %d workers stuck",
            stuck,
            self._max,
        )
        future.add_done_callback(self._release)

    def _release(self, _future: Future[ActionResult[Any]]) -> None:
        with self._lock:
            self._stuck = max(0, self._stuck - 1)

    def shutdown(self) -> None:
        """Stop accepting work. Stuck threads are left to finish on their own."""
        self._pool.shutdown(wait=False)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


class ActionRegistry:
    """Holds the action library and runs one action at a time on request.

    Args:
        bus: Event bus for ``ActionStarted``/``Finished``/``Failed``. Optional so
            the editor and the tests can build a registry with no application
            around it.
        audit: Where privileged execution is journalled (section 11).
        audit_enabled: Whether to write audit rows at all. Defaults to the
            ``privacy.audit_commands`` setting, read per call because the user can
            switch it off while Ayris is running.
        is_elevated: Whether the process holds administrator rights. The seam
            task 39 replaces with the real check and the UAC prompt.
        max_workers: Synchronous actions running at once.
        clock: Monotonic clock, replaceable in tests.
    """

    def __init__(
        self,
        *,
        bus: EventBus | None = None,
        audit: AuditSink | AuditRepository | None = None,
        audit_enabled: Callable[[], bool] | None = None,
        is_elevated: Callable[[], bool] | None = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._actions: dict[str, Action] = {}
        self._bus = bus
        self._audit = audit
        self._audit_enabled = audit_enabled if audit_enabled is not None else _audit_from_settings
        self._is_elevated = is_elevated if is_elevated is not None else _process_is_elevated
        self._pool = _WorkerPool(max_workers)
        self._clock = clock
        self._discovered: set[str] = set()
        self._lock = Lock()

    # -- population -------------------------------------------------------- #

    def add(self, action_class: type[Action], *, plugin: str = "", replace: bool = False) -> str:
        """Put one action class into this registry and return its name.

        Args:
            action_class: The class to instantiate. Validated the same way
                :func:`register` validates it.
            plugin: Slug to prefix the name with, for plugin-supplied actions.
            replace: Allow shadowing an existing name. Off by default: two blocks
                answering to one name would make a macro's meaning depend on
                import order.

        Raises:
            ValueError: The name is taken and ``replace`` is not set.
        """
        _validate_class(action_class)
        meta = action_class.meta.with_prefix(plugin)
        action = action_class()
        # The instance keeps the prefixed metadata, otherwise a plugin action
        # would report one name to the registry and another to the editor.
        if meta.name != action_class.meta.name:
            action.meta = meta  # type: ignore[misc]
        with self._lock:
            if meta.name in self._actions and not replace:
                raise ValueError(f"action {meta.name!r} is already registered")
            self._actions[meta.name] = action
        return meta.name

    def add_all(self, entries: Iterable[RegisteredAction], *, replace: bool = False) -> int:
        """Add every entry that is not in this registry yet. Returns how many."""
        added = 0
        for entry in entries:
            if not replace and entry.name in self._actions:
                continue
            self.add(entry.action_class, plugin=entry.plugin, replace=replace)
            added += 1
        return added

    def discover(self, package: str = SYSTEM_PACKAGE, *, reload: bool = False) -> int:
        """Import every module under ``package`` and adopt what it declared.

        A module that cannot be imported here — WinAPI on Linux, a missing
        optional dependency — is logged and skipped. Returns the number of actions
        the registry gained.
        """
        if package in self._discovered and not reload:
            return self.add_all(registered_actions())
        self._discovered.add(package)
        try:
            root = importlib.import_module(package)
        except ImportError:
            _log.exception("action package %s is not importable", package)
            return 0
        for module in _walk_modules(root):
            try:
                importlib.import_module(module)
            except ImportError as exc:
                # Expected in the sandbox and in CI's Linux job: pycaw, comtypes
                # and WinRT are Windows-only. Not expected on Windows, so it is a
                # warning with the reason attached, not a debug line.
                _log.warning("skipped action module %s: %s", module, exc)
            except Exception:
                _log.exception("action module %s failed to import", module)
        return self.add_all(registered_actions())

    # -- lookup ------------------------------------------------------------ #

    def get(self, name: str) -> Action:
        """One action by name.

        Raises:
            ActionNotFound: Nothing is registered under that name.
        """
        action = self._actions.get(name)
        if action is None:
            raise ActionNotFound(
                f"no action named {name!r}",
                user_message=f"Не знаю действия «{name}».",
            )
        return action

    def has(self, name: str) -> bool:
        """Whether ``name`` is registered."""
        return name in self._actions

    @property
    def names(self) -> tuple[str, ...]:
        """Every registered name, sorted."""
        return tuple(sorted(self._actions))

    def __len__(self) -> int:
        return len(self._actions)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._actions

    def __iter__(self) -> Iterator[Action]:
        return iter(list(self._actions.values()))

    def find(
        self,
        *,
        category: ActionCategory | None = None,
        query: str = "",
        plugin: str | None = None,
    ) -> list[Action]:
        """Actions matching a category, a substring, or both.

        ``query`` is matched case-insensitively against the name, the Russian
        title and the Russian description — the editor's search box, where a user
        types «громк» and expects ``SetVolume``.
        """
        needle = query.strip().casefold()
        found = [
            action
            for action in self._actions.values()
            if (category is None or action.meta.category is category)
            and (plugin is None or action.meta.plugin == plugin)
            and (not needle or _matches(action, needle))
        ]
        return sorted(found, key=lambda item: (item.meta.category.value, item.meta.name))

    def list_categories(self) -> list[ActionCategory]:
        """Categories that actually have actions, in editor order."""
        present = {action.meta.category for action in self._actions.values()}
        return [category for category in ActionCategory if category in present]

    # -- introspection ----------------------------------------------------- #

    def describe(self, name: str) -> ActionSchema:
        """Schema of one action, for the editor's parameter form."""
        return build_schema(self.get(name))

    def describe_all(self) -> list[ActionSchema]:
        """Schemas of every action, grouped the way the block tree shows them."""
        return [build_schema(action) for action in self.find()]

    # -- execution --------------------------------------------------------- #

    def execute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        """Run one action and return its result.

        Args:
            name: Registered action name.
            params: Raw values, as they come out of a macro or of slot extraction.
            request_id: Session this call belongs to, for the trace and the events.
            command_id: Command that asked, when a macro is driving.

        Raises:
            ActionNotFound: No such action.
            ActionParamsInvalid: ``params`` do not fit the schema.
            ActionRequiresAdmin: Needs elevation Ayris does not have.
            ActionTimeout: Ran past ``meta.timeout_ms``.
            ActionError: Anything the action itself raised, wrapped.
        """
        action, validated = self._prepare(name, params, request_id=request_id)
        meta = action.meta
        started = self._clock()
        self._announce(action, validated, request_id=request_id, command_id=command_id)
        try:
            result = self._pool.run(lambda: action.run(validated), meta.timeout_s)
        except BaseException as exc:
            raise self._fail(action, validated, exc, request_id=request_id) from exc
        return self._settle(action, validated, result, started, request_id=request_id)

    async def aexecute(
        self,
        name: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str = "",
        command_id: int | None = None,
    ) -> ActionResult[Any]:
        """Await one action. Same contract as :meth:`execute`."""
        import asyncio

        action, validated = self._prepare(name, params, request_id=request_id)
        meta = action.meta
        started = self._clock()
        self._announce(action, validated, request_id=request_id, command_id=command_id)
        try:
            result = await asyncio.wait_for(action.arun(validated), meta.timeout_s)
        except TimeoutError as exc:
            error = ActionTimeout("action exceeded its timeout")
            raise self._fail(action, validated, error, request_id=request_id) from exc
        except BaseException as exc:
            raise self._fail(action, validated, exc, request_id=request_id) from exc
        return self._settle(action, validated, result, started, request_id=request_id)

    def undo(self, name: str, token: str, *, request_id: str = "") -> ActionResult[Any]:
        """Reverse an earlier call of ``name`` using the token it returned.

        Raises:
            ActionUnavailable: The action does not support undo.
        """
        action = self.get(name)
        if not action.meta.supports_undo:
            raise ActionUnavailable(
                f"{name} does not support undo",
                user_message=f"«{action.meta.title_ru}» нельзя отменить.",
            )
        started = self._clock()
        try:
            result = action.undo(token)
        except BaseException as exc:
            raise self._fail(action, None, exc, request_id=request_id, undo=True) from exc
        return self._settle(action, None, result, started, request_id=request_id, undo=True)

    @property
    def stuck_workers(self) -> int:
        """Threads wedged inside actions the registry gave up waiting for."""
        return self._pool.stuck

    @property
    def free_workers(self) -> int:
        """Worker threads still able to take a synchronous action."""
        return self._pool.capacity

    def shutdown(self) -> None:
        """Release the worker pool. Called from the application's teardown."""
        self._pool.shutdown()

    # -- internals --------------------------------------------------------- #

    def _prepare(
        self, name: str, params: Mapping[str, Any] | None, *, request_id: str
    ) -> tuple[Action, ActionParams]:
        """Resolve the action, validate parameters, check rights."""
        action = self.get(name)
        validated = self._validate(action, params or {})
        if action.meta.require_admin and not self._elevated():
            error = ActionRequiresAdmin(
                f"{name} requires elevation",
                user_message=(f"«{action.meta.title_ru}» требует прав администратора."),
            )
            self._record(action, validated, ExecutionResult.DENIED)
            self._publish(
                ActionFailed(
                    action=name,
                    error=error.technical,
                    user_message=error.user_message,
                    request_id=request_id,
                    reason="denied",
                )
            )
            _log.warning("action %s denied: not elevated", name)
            raise error
        return action, validated

    def _validate(self, action: Action, params: Mapping[str, Any]) -> ActionParams:
        model = type(action).params_model()
        try:
            return model.model_validate(dict(params))
        except ValidationError as exc:
            problems = _problems_of(exc)
            listed = "; ".join(str(problem) for problem in problems)
            raise ActionParamsInvalid(
                f"{action.meta.name}: invalid parameters ({exc.error_count()})",
                problems=problems,
                user_message=(
                    f"«{action.meta.title_ru}»: не поняла параметры — {listed}."
                    if listed
                    else f"«{action.meta.title_ru}»: параметры не подходят."
                ),
            ) from exc

    def _elevated(self) -> bool:
        try:
            return bool(self._is_elevated())
        except OSError:
            _log.exception("elevation check failed; assuming no rights")
            return False

    def _announce(
        self,
        action: Action,
        params: ActionParams,
        *,
        request_id: str,
        command_id: int | None,
    ) -> None:
        masked = mask_params(params)
        _log.debug("action %s started %s", action.meta.name, masked)
        self._publish(
            ActionStarted(
                action=action.meta.name,
                command_id=command_id,
                request_id=request_id,
                params=masked,
            )
        )

    def _settle(
        self,
        action: Action,
        params: ActionParams | None,
        result: ActionResult[Any],
        started: float,
        *,
        request_id: str,
        undo: bool = False,
    ) -> ActionResult[Any]:
        """Stamp the duration, publish, journal, log. One exit for every success."""
        stamped = result.with_duration(int((self._clock() - started) * 1000))
        self._record(action, params, stamped.execution)
        self._publish(
            ActionFinished(
                action=action.meta.name,
                result=stamped.message_ru or stamped.detail,
                duration_ms=stamped.duration_ms,
                request_id=request_id,
                ok=stamped.ok,
            )
        )
        _log.info(
            "action %s%s finished in %d ms: %s",
            action.meta.name,
            " (undo)" if undo else "",
            stamped.duration_ms,
            "ok" if stamped.ok else stamped.detail or "failed",
        )
        return stamped

    def _fail(
        self,
        action: Action,
        params: ActionParams | None,
        exc: BaseException,
        *,
        request_id: str,
        undo: bool = False,
    ) -> ActionError:
        """Turn whatever went wrong into one typed error. One exit for failure."""
        error = _as_action_error(action, exc, undo=undo)
        outcome = (
            ExecutionResult.TIMEOUT
            if isinstance(error, ActionTimeout)
            else (
                ExecutionResult.DENIED
                if isinstance(error, ActionRequiresAdmin)
                else ExecutionResult.ERROR
            )
        )
        self._record(action, params, outcome)
        self._publish(
            ActionFailed(
                action=action.meta.name,
                error=error.technical,
                user_message=error.user_message,
                request_id=request_id,
                reason=outcome.value,
            )
        )
        if isinstance(exc, ActionError):
            _log.error("action %s failed: %s", action.meta.name, error.technical)
        else:
            _log.exception("action %s raised", action.meta.name)
        return error

    def _record(
        self, action: Action, params: ActionParams | None, outcome: ExecutionResult
    ) -> None:
        """Write one row to the audit trail, secrets masked, failures swallowed."""
        if self._audit is None or not self._audit_allowed():
            return
        entry = AuditEntry(
            command_name=action.meta.name,
            params=mask_params(params) if params is not None else {},
            result=outcome,
            require_admin=action.meta.require_admin,
            elevated=self._elevated(),
        )
        try:
            self._audit.add(entry)
        except AyrisError:
            # A broken journal must not turn a working action into a failed one.
            _log.exception("audit write failed for %s", action.meta.name)

    def _audit_allowed(self) -> bool:
        try:
            return bool(self._audit_enabled())
        except AyrisError:
            _log.exception("could not read the audit setting; journalling anyway")
            return True

    def _publish(self, event: ActionStarted | ActionFinished | ActionFailed) -> None:
        if self._bus is None:
            return
        try:
            self._bus.publish(event)
        except AyrisError:
            _log.exception("could not publish %s", event.name)


def _matches(action: Action, needle: str) -> bool:
    meta = action.meta
    haystack = (meta.name, meta.title_ru, meta.description_ru)
    return any(needle in text.casefold() for text in haystack)


def _walk_modules(root: Any) -> Iterator[str]:
    """Names of every module under ``root``, packages included."""
    paths: Sequence[str] = getattr(root, "__path__", ())
    for info in pkgutil.walk_packages(paths, prefix=f"{root.__name__}."):
        yield info.name


def _as_action_error(action: Action, exc: BaseException, *, undo: bool) -> ActionError:
    """Wrap an arbitrary failure in the typed error the callers expect."""
    if isinstance(exc, ActionError):
        return exc
    what = f"{action.meta.title_ru}: отмена не удалась" if undo else action.meta.title_ru
    if isinstance(exc, TimeoutError):
        return ActionTimeout(f"{action.meta.name} timed out: {exc}")
    if isinstance(exc, NotImplementedError):
        return ActionUnavailable(
            f"{action.meta.name} is not implemented on this platform: {exc}",
            user_message=f"«{what}» здесь недоступно.",
        )
    if isinstance(exc, PermissionError):
        return ActionRequiresAdmin(f"{action.meta.name} was denied: {exc}")
    if isinstance(exc, AyrisError):
        return ActionError(
            f"{action.meta.name} failed: {exc.technical}",
            user_message=exc.user_message,
        )
    if isinstance(exc, Exception):
        return ActionError(
            f"{action.meta.name} raised {type(exc).__name__}: {exc}",
            user_message=f"Не удалось выполнить «{what}».",
        )
    # KeyboardInterrupt and SystemExit are not ours to convert into a message.
    raise exc


_PROBLEM_MESSAGES: Final[Mapping[str, str]] = {
    "missing": "не заполнено",
    "extra_forbidden": "лишний параметр",
    "int_parsing": "нужно целое число",
    "int_type": "нужно целое число",
    "float_parsing": "нужно число",
    "float_type": "нужно число",
    "bool_parsing": "нужно да или нет",
    "bool_type": "нужно да или нет",
    "string_type": "нужен текст",
    "string_too_short": "слишком короткое значение",
    "string_too_long": "слишком длинное значение",
    "string_pattern_mismatch": "значение не того вида",
    "enum": "недопустимое значение",
    "literal_error": "недопустимое значение",
    "greater_than": "значение слишком мало",
    "greater_than_equal": "значение слишком мало",
    "less_than": "значение слишком велико",
    "less_than_equal": "значение слишком велико",
    "too_short": "слишком мало элементов",
    "too_long": "слишком много элементов",
    "list_type": "нужен список",
    "dict_type": "нужен словарь",
}


def _problems_of(exc: ValidationError) -> tuple[ParamProblem, ...]:
    """Translate pydantic's report into per-field Russian complaints."""
    problems = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"])
        text = _PROBLEM_MESSAGES.get(error["type"], "значение не подходит")
        limit = _limit_of(error)
        if limit is not None:
            text = f"{text} ({limit})"
        problems.append(ParamProblem(field=location, message=text))
    return tuple(problems)


def _limit_of(error: Mapping[str, Any]) -> str | None:
    """The bound pydantic rejected against, when it named one."""
    context = error.get("ctx")
    if not isinstance(context, dict):
        return None
    for key in ("ge", "le", "gt", "lt", "min_length", "max_length"):
        if key in context:
            return f"{key} {context[key]}"
    expected = context.get("expected")
    return f"допустимо: {expected}" if isinstance(expected, str) else None


def _audit_from_settings() -> bool:
    """Read ``privacy.audit_commands``, defaulting to on when unreadable."""
    from ayris.core.config import get_settings

    try:
        return bool(get_settings().privacy.audit_commands)
    except AyrisError:
        return True


def _process_is_elevated() -> bool:
    """Whether this process runs with administrator rights.

    The honest answer until task 39 brings the real elevation handling: on Windows
    ``IsUserAnAdmin`` is authoritative enough to gate an action, and anywhere else
    there is no such thing, so the answer is no and ``require_admin`` actions
    refuse instead of half-running.
    """
    import ctypes
    import sys

    if sys.platform != "win32":
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except (AttributeError, OSError):
        _log.warning("IsUserAnAdmin is unavailable; treating the process as unprivileged")
        return False
