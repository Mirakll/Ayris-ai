"""Smoke tests for task 01: the package imports and the logger works.

No Qt here — GUI construction needs a display and is covered from task 43
onwards with ``pytest-qt``.
"""

from __future__ import annotations

import logging
import time
from datetime import date, timedelta
from pathlib import Path

import pytest

import ayris
from ayris.core.errors import ActionError, AyrisError, ConfigError
from ayris.core.paths import AppPaths, init_paths, reset_paths
from ayris.utils.dpi import DpiAwarenessResult, enable_per_monitor_dpi_awareness
from ayris.utils.logger import (
    PIPELINE_LOGGER_NAME,
    DailySizedRotatingFileHandler,
    get_logger,
    get_pipeline_logger,
    setup_logging,
    shutdown_logging,
)

pytestmark = pytest.mark.unit


def _today_stamp() -> str:
    return time.strftime("%Y%m%d")


def _drain(handler: logging.Handler, name: str) -> logging.Logger:
    """Attach ``handler`` to an isolated logger that writes nowhere else."""
    log = logging.getLogger(f"ayris.{name}")
    log.propagate = False
    log.setLevel(logging.DEBUG)
    for existing in list(log.handlers):
        log.removeHandler(existing)
    log.addHandler(handler)
    return log


def test_package_imports_and_exposes_metadata() -> None:
    assert ayris.__app_name__ == "Ayris"
    assert ayris.__version__.count(".") == 2


def test_entry_point_module_is_importable() -> None:
    """``python -m ayris`` resolves; the parser is built without loading Qt."""
    from ayris.__main__ import build_parser

    options = build_parser().parse_args([])
    assert options.log_level == "INFO"
    assert options.minimized is False


def test_cli_parses_all_flags() -> None:
    from ayris.__main__ import parse_args

    options = parse_args(["--minimized", "--log-level", "DEBUG", "--profile", "C:/tmp/p"])
    assert options.minimized is True
    assert options.log_level == "DEBUG"
    assert options.profile == Path("C:/tmp/p")
    assert options.portable is False


def test_cli_rejects_profile_with_portable() -> None:
    from ayris.__main__ import parse_args

    with pytest.raises(SystemExit):
        parse_args(["--portable", "--profile", "C:/tmp/p"])


class TestPaths:
    def test_init_creates_every_directory(self, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "profile")
        for directory in paths.all_directories():
            assert directory.is_dir(), directory

    def test_layout_is_rooted_in_the_profile(self, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "profile")
        assert paths.config_file.name == "config.toml"
        assert paths.database_file.name == "ayris.db"
        assert paths.log_file("20260808").name == "ayris_20260808.log"
        assert paths.stt_models_dir.parent == paths.models_dir

    def test_explicit_profile_wins_over_portable(self, tmp_path: Path) -> None:
        from ayris.core.paths import resolve_root

        root = resolve_root(portable=True, profile=tmp_path / "chosen")
        assert root == (tmp_path / "chosen").resolve()

    def test_get_paths_is_cached(self, tmp_path: Path) -> None:
        from ayris.core.paths import get_paths

        first = init_paths(profile=tmp_path / "profile")
        assert get_paths() is first

        reset_paths()
        assert get_paths() is not first


class TestLogger:
    def test_setup_creates_log_file_and_writes(self, tmp_path: Path) -> None:
        setup_logging("DEBUG", console=False, log_dir=tmp_path)
        log = get_logger("test.module")
        log.info("hello from the smoke test")

        for handler in logging.getLogger("ayris").handlers:
            handler.flush()

        written = sorted(tmp_path.glob("ayris_*.log"))
        assert len(written) == 1
        assert "hello from the smoke test" in written[0].read_text(encoding="utf-8")

    def test_child_loggers_live_under_the_ayris_root(self, tmp_path: Path) -> None:
        setup_logging("INFO", console=False, log_dir=tmp_path)
        assert get_logger("audio.capture").name == "ayris.audio.capture"
        # An already-qualified name is not prefixed twice.
        assert get_logger("ayris.audio.capture").name == "ayris.audio.capture"

    def test_pipeline_logger_is_a_separate_channel(self, tmp_path: Path) -> None:
        setup_logging("DEBUG", console=False, log_dir=tmp_path)
        pipeline = get_pipeline_logger()
        pipeline.info("STT raw -> NLU intent", extra={"request_id": "req-1"})
        for handler in pipeline.handlers:
            handler.flush()

        files = sorted(tmp_path.glob("pipeline_*.log"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert "req-1" in content
        assert pipeline.name == PIPELINE_LOGGER_NAME
        # Pipeline traces must not leak into the main log.
        main_log = next(iter(tmp_path.glob("ayris_*.log")))
        assert "STT raw -> NLU intent" not in main_log.read_text(encoding="utf-8")

    def test_pipeline_record_without_request_id_does_not_crash(self, tmp_path: Path) -> None:
        setup_logging("DEBUG", console=False, log_dir=tmp_path)
        pipeline = get_pipeline_logger()
        pipeline.info("no request id attached")
        for handler in pipeline.handlers:
            handler.flush()
        content = next(iter(tmp_path.glob("pipeline_*.log"))).read_text(encoding="utf-8")
        assert "no request id attached" in content

    def test_setup_is_idempotent_and_updates_level(self, tmp_path: Path) -> None:
        setup_logging("INFO", console=False, log_dir=tmp_path)
        handler_count = len(logging.getLogger("ayris").handlers)

        root = setup_logging("DEBUG", console=False, log_dir=tmp_path)
        assert len(root.handlers) == handler_count
        assert root.level == logging.DEBUG

    def test_shutdown_removes_handlers(self, tmp_path: Path) -> None:
        setup_logging("INFO", console=False, log_dir=tmp_path)
        shutdown_logging()
        assert logging.getLogger("ayris").handlers == []
        assert logging.getLogger(PIPELINE_LOGGER_NAME).handlers == []


class TestRotation:
    """The three rotation guarantees from section 15 of the specification."""

    def test_size_rollover_keeps_every_part(self, tmp_path: Path) -> None:
        handler = DailySizedRotatingFileHandler(tmp_path, "probe", max_bytes=512)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = _drain(handler, "size_probe")
        try:
            for index in range(60):
                log.info("padding line %03d %s", index, "x" * 40)
            handler.flush()
        finally:
            log.removeHandler(handler)
            handler.close()

        stamp = _today_stamp()
        assert (tmp_path / f"probe_{stamp}.log").exists()
        parts = sorted(tmp_path.glob(f"probe_{stamp}.*.log"))
        # Every rollover gets its own increasing suffix; none overwrites another.
        assert len(parts) >= 2, parts
        assert {part.name for part in parts} >= {
            f"probe_{stamp}.1.log",
            f"probe_{stamp}.2.log",
        }

    def test_date_change_opens_a_new_file_instead_of_renaming(self, tmp_path: Path) -> None:
        """The date in a file name must always match the records inside it."""
        handler = DailySizedRotatingFileHandler(tmp_path, "probe", max_bytes=0)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = _drain(handler, "date_probe")
        today = _today_stamp()
        try:
            log.info("record from today")
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(
                    DailySizedRotatingFileHandler,
                    "_today_stamp",
                    staticmethod(lambda: "20990101"),
                )
                log.info("record from tomorrow")
                handler.flush()
        finally:
            log.removeHandler(handler)
            handler.close()

        today_file = tmp_path / f"probe_{today}.log"
        future_file = tmp_path / "probe_20990101.log"
        assert "record from today" in today_file.read_text(encoding="utf-8")
        assert "record from tomorrow" in future_file.read_text(encoding="utf-8")
        # The old file keeps its own name — no ``.log.2099-01-01`` suffix anywhere.
        assert "record from tomorrow" not in today_file.read_text(encoding="utf-8")
        assert not list(tmp_path.glob("probe_*.log.*"))

    def test_retention_sweeps_files_left_by_previous_runs(self, tmp_path: Path) -> None:
        today = date.today()
        expired = tmp_path / f"probe_{today - timedelta(days=30):%Y%m%d}.log"
        expired_part = tmp_path / f"probe_{today - timedelta(days=30):%Y%m%d}.2.log"
        recent = tmp_path / f"probe_{today - timedelta(days=3):%Y%m%d}.log"
        foreign = tmp_path / "pipeline_20260101.log"
        for stale in (expired, expired_part, recent, foreign):
            stale.write_text("old record\n", encoding="utf-8")

        handler = DailySizedRotatingFileHandler(tmp_path, "probe", retention_days=7)
        handler.close()

        assert not expired.exists()
        assert not expired_part.exists()
        assert recent.exists(), "a file inside the retention window must survive"
        assert foreign.exists(), "another prefix must not be touched"

    def test_retention_disabled_keeps_everything(self, tmp_path: Path) -> None:
        ancient = tmp_path / "probe_19990101.log"
        ancient.write_text("old record\n", encoding="utf-8")

        handler = DailySizedRotatingFileHandler(tmp_path, "probe", retention_days=0)
        handler.close()

        assert ancient.exists()


class TestErrors:
    def test_subclasses_share_the_base(self) -> None:
        assert issubclass(ConfigError, AyrisError)
        assert issubclass(ActionError, AyrisError)

    def test_default_user_message_is_russian_and_specific(self) -> None:
        error = ActionError("SetVolume failed: device busy")
        assert str(error) == "SetVolume failed: device busy"
        assert error.user_message == "Не удалось выполнить команду."
        assert error.recoverable is True

    def test_explicit_user_message_wins(self) -> None:
        error = ConfigError("bad toml at line 3", user_message="Строка 3 повреждена.")
        assert error.user_message == "Строка 3 повреждена."

    def test_unrecoverable_flag_is_carried(self) -> None:
        error = AyrisError("disk full", recoverable=False)
        assert error.recoverable is False


class TestPathsFailure:
    def test_unwritable_root_raises_config_error(self, tmp_path: Path) -> None:
        def deny(*_args: object, **_kwargs: object) -> None:
            raise OSError("access denied")

        paths = AppPaths(root=tmp_path / "denied")
        # A scoped patch rather than the shared ``monkeypatch`` fixture: the
        # autouse teardown in conftest touches disk, and it runs before a
        # fixture-level undo would restore Path.mkdir.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "mkdir", deny)
            with pytest.raises(ConfigError) as caught:
                paths.ensure_directories()

        assert caught.value.recoverable is False
        assert "--profile" in caught.value.user_message


def test_dpi_awareness_never_raises() -> None:
    """On Linux CI this returns UNSUPPORTED; on Windows it must not raise."""
    result = enable_per_monitor_dpi_awareness()
    assert isinstance(result, DpiAwarenessResult)
    assert isinstance(result.is_per_monitor, bool)
