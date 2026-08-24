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
import threading
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
    "SECRET_PLACEHOLDER",
    "DailySizedRotatingFileHandler",
    "LogLevel",
    "SecretFilter",
    "forget_secret",
    "get_logger",
    "get_pipeline_logger",
    "guard_secret",
    "guarded_secret_count",
    "redact",
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


# --------------------------------------------------------------------------- #
# Secret redaction
# --------------------------------------------------------------------------- #

#: What a redacted value looks like in the log. Deliberately not the same string
#: as :data:`ayris.actions.base.SECRET_MASK`: seeing this one in a log file means
#: a value leaked into a message that was never meant to carry it, and the two
#: are worth telling apart when that happens.
SECRET_PLACEHOLDER: Final = "[скрыто]"

#: Shorter than this and a secret is not worth guarding: redacting every «abc»
#: would mangle unrelated messages, and nothing that short is worth protecting.
_MIN_SECRET_LEN: Final = 4
#: A bound on the registry, because forgetting is best-effort — a provider that
#: reads a hundred passwords must not grow this list forever. The oldest value
#: falls out first; it is also the least likely to still be on its way to a log.
_MAX_SECRETS: Final = 64

_secrets_lock = threading.Lock()
#: Insertion-ordered set of the values currently worth hiding.
_secrets: dict[str, None] = {}


def guard_secret(value: str) -> None:
    """Redact ``value`` from every log record from now on.

    Called the moment a password or a card number enters the process — by the
    autofill action and by the password-manager providers — and undone with
    :func:`forget_secret` once the value has been used. Sequence matters: guard
    before the value can reach any code that logs, not after.

    Ayris does not log secrets on purpose anywhere, so in normal operation this
    filter never fires. It exists for the paths nobody audited: an exception
    whose message quotes the argument that failed, a DEBUG dump of a whole
    parameter dict, a third-party library being helpful. One rule catching all of
    them beats trusting that every future call site remembers.
    """
    text = value.strip()
    if len(text) < _MIN_SECRET_LEN:
        return
    with _secrets_lock:
        _secrets.pop(text, None)
        _secrets[text] = None
        while len(_secrets) > _MAX_SECRETS:
            del _secrets[next(iter(_secrets))]


def forget_secret(value: str) -> None:
    """Stop redacting ``value``. Safe to call for something never guarded."""
    text = value.strip()
    if not text:
        return
    with _secrets_lock:
        _secrets.pop(text, None)


def guarded_secret_count() -> int:
    """How many values are currently being redacted. For tests and DevTools."""
    with _secrets_lock:
        return len(_secrets)


def redact(text: str) -> str:
    """Replace every guarded secret in ``text`` with :data:`SECRET_PLACEHOLDER`.

    Longest values first, so a password that happens to contain a shorter one
    does not leave the tail of itself behind.
    """
    if not _secrets or not text:
        return text
    with _secrets_lock:
        values = sorted(_secrets, key=len, reverse=True)
    for value in values:
        if value in text:
            text = text.replace(value, SECRET_PLACEHOLDER)
    return text


class SecretFilter(logging.Filter):
    """Strip known secret values out of records on their way to a handler.

    Installed on handlers rather than on loggers: a filter on the ``ayris`` logger
    would only see records logged through that logger itself, while every record
    a child logger emits passes the *handlers* of its ancestors. Sitting there
    also covers records made by code outside Ayris that :func:`logging.captureWarnings`
    routes our way.

    The record is rewritten rather than dropped — the line is still useful, only
    without the value. When nothing is guarded the filter returns immediately, so
    the ordinary case costs one dictionary emptiness check.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _secrets:
            return True
        message = str(record.msg)
        if record.args:
            try:
                formatted = record.getMessage()
            except (TypeError, ValueError):
                formatted = message + " " + repr(record.args)
            cleaned = redact(formatted)
            if cleaned != formatted:
                record.msg = cleaned
                record.args = None
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True


def _install_filters(handler: logging.Handler, *extra: logging.Filter) -> logging.Handler:
    """Add the secret filter, and whatever else this handler needs, to it."""
    handler.addFilter(SecretFilter())
    for one in extra:
        handler.addFilter(one)
    return handler


def _build_file_handler(directory: Path, prefix: str, level: int) -> logging.Handler:
    handler = DailySizedRotatingFileHandler(directory, prefix)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT))
    return _install_filters(handler)


def _build_console_handler(level: int) -> logging.Handler:
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    return _install_filters(handler)


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
    _install_filters(handler, _PipelineFilter())
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
    """Flush and detach handlers. Called on exit and between tests.

    Propagation goes back on as well. :func:`setup_logging` turns it off because
    Ayris owns its subtree while it has handlers of its own; once they are gone,
    leaving it off would silently drop every later record instead of letting it
    reach whoever is listening — which between tests is ``caplog``.
    """
    global _configured
    for name in (PIPELINE_LOGGER_NAME, ROOT_LOGGER_NAME):
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.flush()
        _clear_handlers(logger)
        logger.propagate = True
    logging.captureWarnings(False)
    _configured = False
