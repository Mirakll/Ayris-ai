"""SQLite access layer: connections, PRAGMAs, transactions and maintenance.

Everything that touches ``ayris.db`` goes through :class:`Database`. It owns
three things the rest of the code should not have to think about.

**One connection per thread.** A :class:`sqlite3.Connection` may not be used
concurrently from several threads, so each thread lazily gets its own and the
open connections are tracked centrally in order to close them on shutdown.
Connections are created with ``check_same_thread=False`` — not to share them,
but so the thread that shuts the application down can close a connection a
worker thread left behind.

**WAL and its consequences.** WAL lets readers run while a writer is active,
which is what keeps the GUI responsive while a macro writes history. It also
means writers still serialise, hence ``BEGIN IMMEDIATE`` in :meth:`transaction`:
taking the write lock up front turns a mid-transaction ``database is locked``
into a wait at the start, which ``busy_timeout`` handles. WAL is a property of
the database file, not of the connection, so it survives restarts once set.

**Typed failures.** Every :class:`sqlite3.Error` is re-raised as
:class:`~ayris.core.errors.DatabaseError` with a Russian ``user_message``, so no
caller has to import :mod:`sqlite3` to handle a failure.

Timestamps are ISO-8601 UTC text rather than adapted ``datetime`` objects; see
:mod:`ayris.core.models` for why.
"""

from __future__ import annotations

import shutil
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from ayris.core.errors import DatabaseError
from ayris.core.paths import get_paths
from ayris.utils.logger import get_logger

__all__ = [
    "BUSY_TIMEOUT_MS",
    "Database",
    "SqlParams",
    "get_database",
    "init_database",
    "reset_database",
]

_log = get_logger(__name__)

#: How long a statement waits for the write lock before giving up. Long enough
#: to outlast a competing write, short enough that a deadlocked GUI is noticed.
BUSY_TIMEOUT_MS: Final = 5_000

#: Positional or named statement parameters.
SqlParams = Sequence[Any] | Mapping[str, Any]

_MEMORY_PATHS: Final = frozenset({":memory:", ""})

_database: Database | None = None


class _ThreadState(threading.local):
    """Per-thread connection and transaction depth."""

    connection: sqlite3.Connection | None = None
    depth: int = 0
    savepoints: int = 0


class Database:
    """A single SQLite database file.

    Args:
        path: Database file, or ``":memory:"`` for a throwaway one.
        timeout_ms: Busy timeout applied to every connection.

    Use :meth:`open` rather than constructing directly when the schema should be
    migrated as part of opening.
    """

    def __init__(self, path: Path | str, *, timeout_ms: int = BUSY_TIMEOUT_MS) -> None:
        self._is_memory = str(path) in _MEMORY_PATHS
        self._path = Path(path) if not self._is_memory else Path(":memory:")
        self._timeout_ms = timeout_ms
        self._state = _ThreadState()
        self._lock = threading.RLock()
        self._connections: list[sqlite3.Connection] = []
        self._shared: sqlite3.Connection | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        path: Path | str | None = None,
        *,
        migrate: bool = True,
        timeout_ms: int = BUSY_TIMEOUT_MS,
    ) -> Self:
        """Open the database and, by default, bring the schema up to date.

        Args:
            path: Database file. Defaults to the profile's ``ayris.db``.
            migrate: Apply pending migrations. Only turned off by tools that
                inspect an old file without upgrading it.
        """
        target = path if path is not None else get_paths().database_file
        database = cls(target, timeout_ms=timeout_ms)
        # Force the connection now so a broken path fails here rather than at
        # the first unrelated query.
        database.connect()
        if migrate:
            from ayris.core.migrations import apply_migrations

            apply_migrations(database)
        return database

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Database file. ``:memory:`` for an in-memory database."""
        return self._path

    @property
    def is_memory(self) -> bool:
        return self._is_memory

    @property
    def is_closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------
    # connections
    # ------------------------------------------------------------------

    def connect(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use.

        An in-memory database is the exception: every connection to
        ``":memory:"`` is a *separate* empty database, so one connection is
        shared by all threads instead.
        """
        if self._closed:
            raise DatabaseError(
                f"database {self._path} is closed",
                user_message="База данных уже закрыта.",
                recoverable=False,
            )

        if self._is_memory:
            with self._lock:
                if self._shared is None:
                    self._shared = self._new_connection()
                return self._shared

        existing = self._state.connection
        if existing is not None:
            return existing
        connection = self._new_connection()
        self._state.connection = connection
        return connection

    def _new_connection(self) -> sqlite3.Connection:
        if not self._is_memory:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(
                    f"cannot create directory for {self._path}: {exc}",
                    user_message=(
                        "Не удалось создать папку для базы данных:\n"
                        f"{self._path.parent}\nПроверьте права доступа."
                    ),
                    recoverable=False,
                ) from exc

        try:
            connection = sqlite3.connect(
                str(self._path),
                # Transactions are opened explicitly in transaction(); the
                # driver must not inject its own BEGIN behind our back.
                isolation_level=None,
                check_same_thread=False,
                timeout=self._timeout_ms / 1000,
            )
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"cannot open database {self._path}: {exc}",
                user_message=(
                    f"Не удалось открыть базу данных:\n{self._path}\n"
                    "Возможно, файл повреждён или занят другой программой."
                ),
                recoverable=False,
            ) from exc

        connection.row_factory = sqlite3.Row
        self._register_functions(connection)
        self._apply_pragmas(connection)
        with self._lock:
            self._connections.append(connection)
        return connection

    @staticmethod
    def _register_functions(connection: sqlite3.Connection) -> None:
        """Add the SQL helpers SQLite does not ship with.

        SQLite's built-in ``lower()`` and ``LIKE`` only case-fold ASCII, so
        ``'СВЕТ' LIKE '%свет%'`` is false — which would make search in the
        command library useless for a Russian interface. Python's ``str.lower``
        knows the full Unicode case tables, so it is exposed as ``ulower`` and
        used wherever a query has to ignore case.

        ``deterministic=True`` lets SQLite use the function in an index if a
        later migration needs one.
        """
        connection.create_function("ulower", 1, _ulower, deterministic=True)

    def _apply_pragmas(self, connection: sqlite3.Connection) -> None:
        """Configure a fresh connection.

        ``foreign_keys`` is per-connection and off by default in SQLite, so it
        has to be set here or cascades silently do nothing. ``journal_mode`` is
        stored in the file; setting it repeatedly is harmless.
        """
        connection.execute(f"PRAGMA busy_timeout = {self._timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        # NORMAL is the documented companion to WAL: durable across application
        # crashes, and only at risk of losing the last commits if the machine
        # itself loses power. FULL would fsync on every commit.
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")

        if self._is_memory:
            return
        mode_row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        mode = str(mode_row[0]).lower() if mode_row is not None else "unknown"
        if mode != "wal":
            # Network shares and some virtual filesystems refuse WAL. The
            # application still works, just with less read/write concurrency.
            _log.warning("WAL недоступен для %s, используется режим журнала %r", self._path, mode)

    # ------------------------------------------------------------------
    # statement helpers
    # ------------------------------------------------------------------

    def execute(self, sql: str, params: SqlParams = ()) -> sqlite3.Cursor:
        """Run one statement, translating driver failures."""
        connection = self.connect()
        try:
            return connection.execute(sql, params)
        except sqlite3.Error as exc:
            raise _wrap(exc, sql) from exc

    def executemany(self, sql: str, seq: Sequence[SqlParams]) -> sqlite3.Cursor:
        """Run one statement for every parameter set."""
        connection = self.connect()
        try:
            return connection.executemany(sql, seq)
        except sqlite3.Error as exc:
            raise _wrap(exc, sql) from exc

    def query_all(self, sql: str, params: SqlParams = ()) -> list[sqlite3.Row]:
        """Run a SELECT and return every row."""
        return self.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: SqlParams = ()) -> sqlite3.Row | None:
        """Run a SELECT and return the first row, or ``None``."""
        row: sqlite3.Row | None = self.execute(sql, params).fetchone()
        return row

    def query_value(self, sql: str, params: SqlParams = (), default: Any = None) -> Any:
        """Return the first column of the first row, or ``default``."""
        row = self.query_one(sql, params)
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    def insert(self, sql: str, params: SqlParams = ()) -> int:
        """Run an INSERT and return the new ``rowid``."""
        cursor = self.execute(sql, params)
        rowid = cursor.lastrowid
        if rowid is None:  # pragma: no cover - sqlite always sets it for INSERT
            raise DatabaseError(
                f"insert did not produce a rowid: {sql}",
                user_message="Не удалось сохранить запись в базе данных.",
            )
        return rowid

    # ------------------------------------------------------------------
    # transactions
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(self, *, immediate: bool = True) -> Iterator[sqlite3.Connection]:
        """Run a block atomically.

        Commits on success, rolls back on any exception. Nesting is supported:
        an inner ``with`` becomes a SAVEPOINT, so a repository method can be
        called standalone or as part of a larger unit of work without knowing
        which it is. Only the outermost block commits.

        Args:
            immediate: Take the write lock when the transaction opens. Turn off
                for read-only blocks that only need a consistent snapshot.
        """
        connection = self.connect()
        if self._state.depth == 0:
            yield from self._outer_transaction(connection, immediate=immediate)
        else:
            yield from self._savepoint(connection)

    def _outer_transaction(
        self, connection: sqlite3.Connection, *, immediate: bool
    ) -> Iterator[sqlite3.Connection]:
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        except sqlite3.Error as exc:
            raise _wrap(exc, "BEGIN") from exc

        self._state.depth = 1
        try:
            yield connection
        except BaseException:
            # rollback() is a no-op when SQLite has already unwound the
            # transaction itself, which execute("ROLLBACK") would turn into a
            # second, confusing error.
            self._safe_rollback(connection)
            raise
        else:
            try:
                connection.commit()
            except sqlite3.Error as exc:
                self._safe_rollback(connection)
                raise _wrap(exc, "COMMIT") from exc
        finally:
            self._state.depth = 0

    def _savepoint(self, connection: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
        self._state.savepoints += 1
        name = f"ayris_sp_{self._state.savepoints}"
        try:
            connection.execute(f"SAVEPOINT {name}")
        except sqlite3.Error as exc:
            raise _wrap(exc, f"SAVEPOINT {name}") from exc

        self._state.depth += 1
        try:
            yield connection
        except BaseException:
            try:
                connection.execute(f"ROLLBACK TO {name}")
                connection.execute(f"RELEASE {name}")
            except sqlite3.Error:  # pragma: no cover - outer block will roll back
                _log.debug("не удалось откатить точку сохранения %s", name)
            raise
        else:
            try:
                connection.execute(f"RELEASE {name}")
            except sqlite3.Error as exc:
                raise _wrap(exc, f"RELEASE {name}") from exc
        finally:
            self._state.depth -= 1

    @staticmethod
    def _safe_rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.rollback()
        except sqlite3.Error:  # pragma: no cover - nothing useful to do
            _log.exception("откат транзакции не удался")

    # ------------------------------------------------------------------
    # maintenance
    # ------------------------------------------------------------------

    def vacuum(self) -> None:
        """Rebuild the file, reclaiming space freed by deletes.

        Cannot run inside a transaction, so callers must finish theirs first.
        """
        if self._state.depth:
            raise DatabaseError(
                "VACUUM cannot run inside a transaction",
                user_message="Нельзя сжимать базу данных во время другой операции.",
            )
        self.execute("VACUUM")
        _log.info("база данных сжата: %s", self._path)

    def checkpoint(self, mode: str = "TRUNCATE") -> None:
        """Fold the WAL back into the main file.

        Called before copying the database so the copy is self-contained.
        """
        if self._is_memory:
            return
        self.execute(f"PRAGMA wal_checkpoint({mode})")

    def integrity_check(self) -> bool:
        """Whether SQLite considers the file structurally sound."""
        result = self.query_value("PRAGMA integrity_check", default="unknown")
        return str(result).lower() == "ok"

    def backup(self, destination: Path | str) -> Path:
        """Copy the database to ``destination`` using the online backup API.

        Safe to call while the application is running: SQLite takes a consistent
        snapshot rather than copying a file that may be mid-write. The result is
        a single file with no side WAL, so the user can archive it as is.

        Returns:
            The path written.
        """
        target = Path(destination)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(
                f"cannot create backup directory {target.parent}: {exc}",
                user_message=f"Не удалось создать папку для резервной копии:\n{target.parent}",
            ) from exc

        source = self.connect()
        try:
            # sqlite3's connection context manager commits, it does not close;
            # closing() is what actually releases the destination file.
            with closing(sqlite3.connect(str(target))) as destination_connection:
                source.backup(destination_connection)
        except sqlite3.Error as exc:
            raise DatabaseError(
                f"backup to {target} failed: {exc}",
                user_message=(
                    f"Не удалось создать резервную копию базы данных:\n{target}\n"
                    "Проверьте свободное место и права доступа."
                ),
            ) from exc

        _log.info("резервная копия базы данных: %s", target)
        return target

    def restore(self, source: Path | str) -> None:
        """Replace the current database with a backup file.

        The running connections are closed first: overwriting the file under an
        open connection would leave stale pages in its cache.
        """
        origin = Path(source)
        if not origin.is_file():
            raise DatabaseError(
                f"backup file not found: {origin}",
                user_message=f"Файл резервной копии не найден:\n{origin}",
            )
        if self._is_memory:
            raise DatabaseError(
                "cannot restore into an in-memory database",
                user_message="Восстановление во временную базу данных невозможно.",
            )

        self.close()
        try:
            for suffix in ("-wal", "-shm"):
                Path(f"{self._path}{suffix}").unlink(missing_ok=True)
            shutil.copyfile(origin, self._path)
        except OSError as exc:
            raise DatabaseError(
                f"cannot restore {self._path} from {origin}: {exc}",
                user_message=f"Не удалось восстановить базу данных из файла:\n{origin}",
                recoverable=False,
            ) from exc

        self._closed = False
        _log.info("база данных восстановлена из %s", origin)

    def file_size(self) -> int:
        """Size of the database in bytes, WAL included. ``0`` for in-memory."""
        if self._is_memory:
            return 0
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{self._path}{suffix}")
            if candidate.is_file():
                total += candidate.stat().st_size
        return total

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close every connection. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            connections = list(self._connections)
            self._connections.clear()
            self._shared = None

        for connection in connections:
            try:
                connection.close()
            except sqlite3.Error:  # pragma: no cover - closing must not raise
                _log.exception("не удалось закрыть соединение с базой данных")

        self._state.connection = None
        self._state.depth = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _ulower(value: str | bytes | int | float | None) -> str | bytes | int | float | None:
    """Unicode-aware ``lower()`` for SQL. Non-text values pass through.

    The parameter type lists what SQLite can hand a user function; anything
    other than text is returned untouched so the function is safe to apply to a
    nullable column.
    """
    return value.lower() if isinstance(value, str) else value


def _wrap(exc: sqlite3.Error, sql: str) -> DatabaseError:
    """Translate a driver error into a typed one with a Russian message."""
    statement = " ".join(sql.split())
    if len(statement) > 200:
        statement = f"{statement[:200]}..."

    if isinstance(exc, sqlite3.IntegrityError):
        return DatabaseError(
            f"integrity error on [{statement}]: {exc}",
            user_message=_integrity_message(str(exc)),
        )
    if isinstance(exc, sqlite3.OperationalError) and "locked" in str(exc).lower():
        return DatabaseError(
            f"database is locked on [{statement}]: {exc}",
            user_message="База данных занята другой операцией. Повторите попытку.",
        )
    return DatabaseError(
        f"sqlite error on [{statement}]: {exc}",
        user_message="Ошибка при обращении к базе данных Ayris.",
    )


def _integrity_message(detail: str) -> str:
    lowered = detail.lower()
    if "unique" in lowered:
        return "Запись с таким именем уже существует."
    if "foreign key" in lowered:
        return "Запись ссылается на объект, которого больше нет."
    if "not null" in lowered:
        return "Не заполнено обязательное поле."
    return "Данные не прошли проверку целостности базы."


def init_database(path: Path | str | None = None, *, migrate: bool = True) -> Database:
    """Open and install the process-wide database.

    Called once at startup, next to :func:`ayris.core.paths.init_paths`.
    """
    global _database
    if _database is not None:
        _database.close()
    _database = Database.open(path, migrate=migrate)
    return _database


def get_database() -> Database:
    """Return the process-wide database, opening it on first use."""
    if _database is None:
        return init_database()
    return _database


def reset_database() -> None:
    """Close and forget the process-wide database. Used on shutdown and in tests."""
    global _database
    if _database is not None:
        _database.close()
    _database = None
