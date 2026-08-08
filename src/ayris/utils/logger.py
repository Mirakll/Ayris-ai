"""Logging setup: rotating file handler, console handler, pipeline channel.

Rotation follows section 15 of the specification: one file per day named
``ayris_YYYYMMDD.log``, capped at 10 MB, keeping 7 days of history.

The stdlib handlers cannot express that combination.
``TimedRotatingFileHandler`` renames the *current* file on rollover and keeps
writing to the original name, so a date baked into the base name would go stale
after midnight; it also prunes only files it created itself, so history from
previous runs would live forever. ``RotatingFileHandler`` has no notion of days
at all. Hence :class:`DailySizedRotatingFileHandler`, which owns both limits and
the retention sweep.

``print()`` is banned project-wide (enforced by ruff ``T20``); every diagnostic
goes through :func:`get_logger`.

Task 41 extends this module with a multiprocess queue writer, live per-module
level switching and an in-memory ring buffer. The contract later tasks rely on
is :func:`setup_logging`, :func:`get_logger` and :func:`get_pipeline_logger`.
"""

from __future__ import annotations

import logging
import logging.handlers
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, get_args

from ayris.core.paths import get_paths

__all__ = [
    "BACKUP_DAYS",
    "LOG_LEVELS",
    "MAX_BYTES",
    "PIPELINE_LOGGER_NAME",
    "ROOT_LOGGER_NAME",
    "DailySizedRotatingFileHandler",
    "LogLevel",
    "get_logger",
    "get_pipeline_logger",
    "setup_logging",
    "shutdown_logging",
]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
LOG_LEVELS: Final[tuple[str, ...]] = get_args(LogLevel)

ROOT_LOGGER_NAME: Final = "ayris"
PIPELINE_LOGGER_NAME: Final = "ayris.pipeline"

MAIN_LOG_PREFIX: Final = "ayris"
PIPELINE_LOG_PREFIX: Final = "pipeline"

MAX_BYTES: Final = 10 * 1024 * 1024
BACKUP_DAYS: Final = 7

_STAMP_FORMAT: Final = "%Y%m%d"
_FILE_FORMAT: Final = (
    "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
)
_CONSOLE_FORMAT: Final = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_PIPELINE_FORMAT: Final = "%(asctime)s.%(msecs)03d | %(request_id)s | %(message)s"
_DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

_configured = False


class DailySizedRotatingFileHandler(logging.handlers.BaseRotatingHandler):
    """Write to ``<prefix>_<YYYYMMDD>.log``, rolling on date and on size.

    On a date change the handler switches to a new file rather than renaming the
    old one, so the date in a file name always matches the records inside it.
    When the size cap is hit within a single day the current file is moved aside
    to ``<prefix>_<YYYYMMDD>.<n>.log`` with an increasing ``n``, so repeated
    rollovers on a busy day never overwrite each other.

    Retention is enforced by scanning the directory, which also cleans up files
    left by previous runs of the application.

    Args:
        directory: Where log files live.
        prefix: File name prefix, ``ayris`` or ``pipeline``.
        max_bytes: Size cap; ``0`` disables the size limit.
        retention_days: Delete files whose date is older than this many days.
    """

    def __init__(
        self,
        directory: Path | str,
        prefix: str,
        *,
        max_bytes: int = MAX_BYTES,
        retention_days: int = BACKUP_DAYS,
        encoding: str = "utf-8",
    ) -> None:
        self.directory = Path(directory)
        self.prefix = prefix
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        self._stamp = self._today_stamp()
        self._name_pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})(?:\.\d+)?\.log$")

        self.directory.mkdir(parents=True, exist_ok=True)
        super().__init__(
            self._path_for(self._stamp),
            mode="a",
            encoding=encoding,
            delay=True,
        )
        self.purge_expired()

    @staticmethod
    def _today_stamp() -> str:
        return time.strftime(_STAMP_FORMAT)

    def _path_for(self, stamp: str) -> Path:
        return self.directory / f"{self.prefix}_{stamp}.log"

    def _next_part_path(self) -> Path:
        """First free ``<prefix>_<stamp>.<n>.log`` for the current day."""
        part = 1
        while True:
            candidate = self.directory / f"{self.prefix}_{self._stamp}.{part}.log"
            if not candidate.exists():
                return candidate
            part += 1

    def shouldRollover(self, record: logging.LogRecord) -> int:  # noqa: N802
        if self._today_stamp() != self._stamp:
            return 1
        if self.max_bytes <= 0:
            return 0
        # The size is read from disk instead of the stream because the handler
        # opens lazily (delay=True), so the stream may not exist yet.
        current = Path(self.baseFilename)
        if not current.exists():
            return 0
        message = f"{self.format(record)}\n".encode(self.encoding or "utf-8")
        return int(current.stat().st_size + len(message) >= self.max_bytes)

    def doRollover(self) -> None:  # noqa: N802
        if self.stream is not None:
            self.stream.close()
            # The stdlib's own rotating handlers close and clear the stream the
            # same way, but typeshed declares FileHandler.stream as a plain
            # TextIOWrapper, so the None it accepts at runtime is a type error.
            self.stream = None  # type: ignore[assignment]

        today = self._today_stamp()
        if today != self._stamp:
            self._stamp = today
            self.baseFilename = str(self._path_for(today))
        else:
            current = Path(self.baseFilename)
            if current.exists():
                current.replace(self._next_part_path())

        self.purge_expired()
        if not self.delay:
            self.stream = self._open()

    def purge_expired(self) -> None:
        """Delete log files older than :attr:`retention_days`."""
        if self.retention_days <= 0:
            return
        cutoff = date.today() - timedelta(days=self.retention_days)
        for candidate in self.directory.glob(f"{self.prefix}_*.log"):
            match = self._name_pattern.match(candidate.name)
            if match is None:
                continue
            try:
                stamp = datetime.strptime(match.group(1), _STAMP_FORMAT).date()
            except ValueError:
                continue
            if stamp < cutoff:
                candidate.unlink(missing_ok=True)


class _PipelineFilter(logging.Filter):
    """Guarantee ``request_id`` exists so the pipeline formatter never fails."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            # setattr, not direct assignment: LogRecord has no such attribute in
            # typeshed, and mypy --strict rejects adding one.
            setattr(record, "request_id", "-")  # noqa: B010
        return True


def _build_file_handler(directory: Path, prefix: str, level: int) -> logging.Handler:
    handler = DailySizedRotatingFileHandler(directory, prefix)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _build_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


def _configure_pipeline_logger(directory: Path, level: int) -> None:
    """Separate channel for ``STT -> NLU -> Action -> Result`` timing traces.

    Kept in its own file so DEBUG-level pipeline traces never drown the main log,
    and so DevTools (task 58) can render it as a table of stages.
    """
    pipeline = logging.getLogger(PIPELINE_LOGGER_NAME)
    pipeline.setLevel(level)
    # Written to its own file only, never duplicated into the main log.
    pipeline.propagate = False

    _clear_handlers(pipeline)

    handler = DailySizedRotatingFileHandler(directory, PIPELINE_LOG_PREFIX)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_PIPELINE_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_PipelineFilter())
    pipeline.addHandler(handler)


def _clear_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def _resolve_level(level: str) -> int:
    """Map a level name to its numeric value, falling back to INFO."""
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        logging.getLogger(ROOT_LOGGER_NAME).warning(
            "unknown log level %r, falling back to INFO", level
        )
        return logging.INFO
    return numeric


def setup_logging(
    level: LogLevel | str = "INFO",
    *,
    console: bool = True,
    log_dir: Path | None = None,
) -> logging.Logger:
    """Configure the ``ayris`` logger tree. Idempotent.

    Args:
        level: Threshold for every handler. An unknown name falls back to INFO
            with a warning rather than raising during startup.
        console: Also write to stderr. Disabled in the windowed build, where no
            console is attached.
        log_dir: Override the log directory. Defaults to the profile's
            ``logs/``; tests point it at ``tmp_path``.

    Returns:
        The configured ``ayris`` logger.
    """
    global _configured

    numeric_level = _resolve_level(str(level))
    directory = log_dir if log_dir is not None else get_paths().logs_dir
    directory.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(numeric_level)
    # Ayris owns its subtree; nothing escapes to the interpreter root logger.
    root.propagate = False

    if _configured:
        _apply_level(numeric_level)
        return root

    _clear_handlers(root)
    root.addHandler(_build_file_handler(directory, MAIN_LOG_PREFIX, numeric_level))
    if console:
        root.addHandler(_build_console_handler(numeric_level))

    _configure_pipeline_logger(directory, numeric_level)
    logging.captureWarnings(True)

    _configured = True
    root.debug(
        "logging configured: level=%s dir=%s",
        logging.getLevelName(numeric_level),
        directory,
    )
    return root


def _apply_level(numeric_level: int) -> None:
    """Retarget an already-configured logger tree at a new level."""
    for name in (ROOT_LOGGER_NAME, PIPELINE_LOGGER_NAME):
        logger = logging.getLogger(name)
        logger.setLevel(numeric_level)
        for handler in logger.handlers:
            handler.setLevel(numeric_level)


def get_logger(name: str) -> logging.Logger:
    """Return a child of the ``ayris`` logger.

    Accepts a bare module name or a dotted ``__name__``; both end up inside the
    ``ayris`` subtree, so the configured handlers apply.
    """
    if name == ROOT_LOGGER_NAME or name.startswith(f"{ROOT_LOGGER_NAME}."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")


def get_pipeline_logger() -> logging.Logger:
    """Return the pipeline channel logger.

    Callers pass ``extra={"request_id": ...}`` to correlate the stages of one
    request, from wake word to spoken answer.
    """
    return logging.getLogger(PIPELINE_LOGGER_NAME)


def shutdown_logging() -> None:
    """Flush and detach handlers. Called on exit and between tests."""
    global _configured
    for name in (PIPELINE_LOGGER_NAME, ROOT_LOGGER_NAME):
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.flush()
        _clear_handlers(logger)
    logging.captureWarnings(False)
    _configured = False
