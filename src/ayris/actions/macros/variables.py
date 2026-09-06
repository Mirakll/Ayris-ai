"""Where shared variables live: in memory for a session, in the database for good.

A ``local`` variable belongs to one run and is kept by
:class:`~ayris.actions.macros.context.ExecutionContext` itself. ``profile`` and
``global`` ones are shared, which makes them a concurrency problem rather than a
dictionary: two commands fired at once run on two threads of the engine's pool, and both
may touch ``work_mode``. So every shared read and write goes through one object with a
lock, and the interpreter never learns which implementation it got —
:class:`MemoryVariables` for the session, :class:`DatabaseVariables` when
``persistent`` variables have to outlive the process.

**Why writes are batched.** ``While`` counting to a thousand and writing a persistent
counter each turn would be a thousand upserts, each of them a disk sync; section 7.2's
counters exist precisely to be incremented in loops. So a write marks the name dirty and
returns, and :meth:`DatabaseVariables.flush` moves the dirty set into the database once,
at the end of the run. What the process would lose in a crash is the last few seconds of
a counter, which is the right trade for not turning a loop into an I/O storm.

**Why a failed write does not fail the command.** The database may be locked by another
process or the disk may be full, and "яркость выставлена" is still true when the counter
did not get saved. :meth:`~DatabaseVariables.flush` logs the failure and leaves the names
dirty so the next run tries again.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Final, Protocol

from ayris.actions.macros.expressions import coerce_value
from ayris.core.errors import DatabaseError
from ayris.core.models import VariableScope, VariableType
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ayris.core.repositories import VariableRepository

__all__ = [
    "MAX_PENDING",
    "MISSING",
    "DatabaseVariables",
    "MemoryVariables",
    "VariableStore",
]

_log = get_logger(__name__)


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


class VariableStore(Protocol):
    """Where ``profile`` and ``global`` variables live.

    :meth:`update` is on the protocol because ``ArrayPush`` on a shared array is a read,
    a change and a write that another thread must not be able to split. :meth:`declare`
    and :meth:`flush` are here for the persistent half: the first tells the store which
    names have to reach the database and with what type, the second is when they do.
    :class:`MemoryVariables` answers both with a shrug, and the interpreter cannot tell
    the two implementations apart.
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

    def declare(
        self,
        scope: VariableScope,
        name: str,
        *,
        var_type: VariableType = VariableType.STRING,
        persistent: bool = False,
    ) -> None:
        """Say that ``name`` exists in ``scope``, and whether it has to be persisted."""
        ...

    def flush(self) -> None:
        """Write out whatever has been changed since the last call."""
        ...


class MemoryVariables:
    """Shared variables for the length of the session: a dictionary behind a lock.

    Not a stand-in for something missing. A ``global`` variable that is not
    ``persistent`` is supposed to live exactly this long, so this is the whole answer for
    it, and :class:`DatabaseVariables` is the answer for the other half.

    The lock is an :class:`threading.RLock` because :meth:`update` calls a function while
    holding it and that function may read the same store.
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

    def declare(
        self,
        scope: VariableScope,
        name: str,
        *,
        var_type: VariableType = VariableType.STRING,
        persistent: bool = False,
    ) -> None:
        """Nothing to remember: this store has one lifetime and no types of its own."""

    def flush(self) -> None:
        """Nothing to write: the dictionary *is* the storage."""

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


#: How many names may wait for :meth:`DatabaseVariables.flush` before it happens on its
#: own. A run that touches more shared variables than this is not the loop the batching
#: was written for, and holding an unbounded dirty set for it would trade one problem for
#: another.
MAX_PENDING: Final = 64


class DatabaseVariables:
    """Shared variables kept in memory and written to the database when asked.

    Reads never touch the disk: the whole shared set is loaded once at construction and
    kept in the same dictionary :class:`MemoryVariables` uses, because a condition like
    ``{work_mode} == "работа"`` is evaluated on the interpreter's thread and a SELECT per
    comparison would be felt. Writes go to that dictionary too, and only mark the name as
    owing a row.

    Which names owe one is decided by :meth:`declare`, from the ``persistent`` flag of the
    declaration in the command; a name loaded from the database already has it, because a
    row that is there is a promise to keep it there. Everything else lives and dies with
    the session, exactly as :class:`MemoryVariables` would have it.
    """

    def __init__(
        self,
        repository: VariableRepository,
        *,
        profile_id: int | None = None,
        max_pending: int = MAX_PENDING,
    ) -> None:
        self._repository = repository
        self._profile_id = profile_id
        self._max_pending = max(1, max_pending)
        self._lock = threading.RLock()
        self._memory = MemoryVariables()
        self._types: dict[tuple[VariableScope, str], VariableType] = {}
        self._persistent: set[tuple[VariableScope, str]] = set()
        self._dirty: set[tuple[VariableScope, str]] = set()
        self._load()

    def _load(self) -> None:
        """Read the stored rows into the cache. A locked or broken file is not fatal."""
        try:
            rows = self._repository.list_all()
        except DatabaseError:
            _log.exception("не удалось прочитать сохранённые переменные")
            return
        for row in rows:
            if row.scope is VariableScope.LOCAL:
                continue
            if row.scope is VariableScope.PROFILE and row.profile_id != self._profile_id:
                continue
            key = (row.scope, row.name)
            self._memory.write(row.scope, row.name, row.value)
            self._types[key] = row.type
            if row.persistent:
                self._persistent.add(key)

    def read(self, scope: VariableScope, name: str) -> Any:
        """The value, or :data:`MISSING` when this scope has no such name."""
        with self._lock:
            return self._memory.read(scope, name)

    def write(self, scope: VariableScope, name: str, value: Any) -> None:
        """Put ``value`` under ``name``; mark it for the database when it is persistent."""
        with self._lock:
            self._memory.write(scope, name, value)
            self._touch(scope, name)

    def update(self, scope: VariableScope, name: str, change: Callable[[Any], Any]) -> Any:
        """Read, change and write as one step, giving back what was written."""
        with self._lock:
            value = self._memory.update(scope, name, change)
            self._touch(scope, name)
            return value

    def names(self, scope: VariableScope) -> frozenset[str]:
        """Every name this scope holds."""
        with self._lock:
            return self._memory.names(scope)

    def declare(
        self,
        scope: VariableScope,
        name: str,
        *,
        var_type: VariableType = VariableType.STRING,
        persistent: bool = False,
    ) -> None:
        """Remember the declared type of a name, and whether it has to be persisted."""
        if scope is VariableScope.LOCAL:
            return
        with self._lock:
            key = (scope, name)
            self._types[key] = var_type
            if persistent:
                self._persistent.add(key)

    def _touch(self, scope: VariableScope, name: str) -> None:
        """Mark a written name as owing a row, flushing early if too many do.

        Called with the lock held. The early flush is a ceiling on memory, not the normal
        path: the normal path is one :meth:`flush` per run.
        """
        key = (scope, name)
        if key not in self._persistent:
            return
        self._dirty.add(key)
        if len(self._dirty) >= self._max_pending:
            self._write_pending()

    def flush(self) -> None:
        """Write every changed persistent variable. Called once per run by the engine."""
        with self._lock:
            self._write_pending()

    def pending(self) -> frozenset[tuple[VariableScope, str]]:
        """Which names are still waiting to be written. For tests and for the editor."""
        with self._lock:
            return frozenset(self._dirty)

    def _write_pending(self) -> None:
        """The upserts themselves, with the lock held.

        A name whose row fails to write stays dirty, so the next flush tries again rather
        than losing the value silently. The failure is logged once per flush and not once
        per name: a locked database fails for all of them at the same moment, and one line
        per variable would bury the reason.
        """
        if not self._dirty:
            return
        written: set[tuple[VariableScope, str]] = set()
        failures: list[str] = []
        reason: DatabaseError | None = None
        for key in sorted(self._dirty, key=lambda item: (str(item[0]), item[1])):
            scope, name = key
            value = self._memory.read(scope, name)
            if value is MISSING:
                written.add(key)
                continue
            try:
                self._store(scope, name, value)
            except DatabaseError as exc:
                failures.append(f"{scope}.{name}")
                reason = exc
            else:
                written.add(key)
        self._dirty -= written
        if reason is not None:
            _log.error(
                "не удалось сохранить переменные (%s), попробуем позже: %s",
                ", ".join(failures),
                reason,
            )

    def _store(self, scope: VariableScope, name: str, value: Any) -> None:
        """One upsert, with the value made to fit the declared type when there is one."""
        var_type = self._types.get((scope, name))
        stored = value if var_type is None else coerce_value(name, value, var_type)
        self._repository.set(
            name,
            stored,
            scope=scope,
            profile_id=self._profile_id if scope is VariableScope.PROFILE else None,
            var_type=var_type,
            persistent=True,
        )
