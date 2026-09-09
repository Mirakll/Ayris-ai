"""Task 41: live logging, bus mirroring, redaction and support export."""

from __future__ import annotations

import logging
import threading
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pytest

from ayris.core.database import Database
from ayris.core.events import EventBus, LogLine
from ayris.core.migrations import apply_migrations
from ayris.core.paths import AppPaths
from ayris.utils.bug_report import build_bug_report
from ayris.utils.logger import (
    DailySizedRotatingFileHandler,
    bind_log_bus,
    dropped_log_lines,
    get_level_state,
    get_log_buffer,
    get_logger,
    reset_module_level,
    set_level,
    set_module_level,
    setup_logging,
    shutdown_logging,
)
from ayris.utils.redaction import redact_text

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_logging() -> None:
    shutdown_logging()
    yield
    shutdown_logging()


def test_limits_reach_handlers_and_update_live(tmp_path: Path) -> None:
    root = setup_logging("INFO", console=False, log_dir=tmp_path, max_mb=3, retention_days=12)
    handler = next(
        item for item in root.handlers if isinstance(item, DailySizedRotatingFileHandler)
    )
    assert handler.max_bytes == 3 * 1024 * 1024
    assert handler.retention_days == 12

    setup_logging("INFO", console=False, log_dir=tmp_path, max_mb=4, retention_days=8)
    assert handler.max_bytes == 4 * 1024 * 1024
    assert handler.retention_days == 8


def test_rotation_by_size_date_and_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = DailySizedRotatingFileHandler(tmp_path, "probe", max_bytes=20, retention_days=7)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("rotation-probe")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.info("x" * 30)
    logger.info("y" * 30)
    assert list(tmp_path.glob("probe_*.1.log"))

    old_stamp = handler._stamp
    monkeypatch.setattr(handler, "_today_stamp", lambda: "20990101")
    logger.info("tomorrow")
    assert (tmp_path / f"probe_{old_stamp}.log").exists()
    assert (tmp_path / "probe_20990101.log").exists()
    handler.close()

    expired = tmp_path / f"probe_{date.today() - timedelta(days=20):%Y%m%d}.log"
    expired.write_text("old", encoding="utf-8")
    DailySizedRotatingFileHandler(tmp_path, "probe", retention_days=7).close()
    assert not expired.exists()


def test_live_levels_and_module_override(tmp_path: Path) -> None:
    setup_logging("INFO", console=False, log_dir=tmp_path)
    assert set_level("WARNING") == "WARNING"
    assert set_module_level("audio.vad", "DEBUG") == "DEBUG"
    assert get_level_state() == ("WARNING", {"ayris.audio.vad": "DEBUG"})
    assert get_logger("audio.vad").isEnabledFor(logging.DEBUG)
    get_logger("audio.vad").debug("selected-debug")
    get_logger("audio.other").debug("other-debug")
    for handler in logging.getLogger("ayris").handlers:
        handler.flush()
    contents = next(tmp_path.glob("ayris_*.log")).read_text(encoding="utf-8")
    assert "selected-debug" in contents
    assert "other-debug" not in contents
    assert reset_module_level("audio.vad") is True
    assert not get_logger("audio.vad").isEnabledFor(logging.INFO)


def test_buffer_filters_and_main_bus_request_id(tmp_path: Path) -> None:
    setup_logging("DEBUG", console=False, log_dir=tmp_path, buffer_lines=3)
    bus = EventBus(thread_id=None)
    seen: list[LogLine] = []
    bus.subscribe(LogLine, seen.append, weak=False)
    bind_log_bus(bus, capacity=10)
    get_logger("audio.vad").warning("speech", extra={"request_id": "req-41"})
    get_logger("other").info("other")
    deadline = time.monotonic() + 1
    while not seen and time.monotonic() < deadline:
        time.sleep(0.01)
    assert seen[0].request_id == "req-41"
    assert get_log_buffer(level="WARNING", module="audio.vad")[0].message == "speech"


def test_bus_overflow_counts_dropped_lines(tmp_path: Path) -> None:
    setup_logging("DEBUG", console=False, log_dir=tmp_path)
    entered = threading.Event()
    release = threading.Event()

    class SlowBus:
        def publish(self, _event: LogLine) -> None:
            entered.set()
            release.wait(timeout=2)

    bind_log_bus(SlowBus(), capacity=1)  # type: ignore[arg-type]
    get_logger("overflow").info("first")
    assert entered.wait(timeout=1)
    for index in range(10):
        get_logger("overflow").info("queued-%d", index)
    assert dropped_log_lines() > 0
    release.set()


def test_pattern_redaction() -> None:
    bearer = "Bearer abcdefghijklmnopqrstuvwxyz123456"
    api_key = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    card = "4111 1111 1111 1111"
    cleaned = redact_text(f"{bearer} {api_key} {card}")
    assert bearer not in cleaned
    assert api_key not in cleaned
    assert card not in cleaned


def test_pattern_redaction_reaches_log_file(tmp_path: Path) -> None:
    setup_logging("INFO", console=False, log_dir=tmp_path)
    api_key = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    get_logger("redaction").warning("credential=%s", api_key)
    for handler in logging.getLogger("ayris").handlers:
        handler.flush()
    contents = next(tmp_path.glob("ayris_*.log")).read_text(encoding="utf-8")
    assert api_key not in contents
    assert "[скрыто]" in contents


def test_bug_report_redacts_every_entry(tmp_path: Path) -> None:
    profile = AppPaths(root=tmp_path)
    profile.ensure_directories()
    secret = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    profile.config_file.write_text(f'[provider]\napi_key = "{secret}"\n', encoding="utf-8")
    (profile.logs_dir / "worker_stt_20260909.log").write_text(secret, encoding="utf-8")
    database = Database.open(profile.database_file)
    apply_migrations(database)
    database.execute(
        "INSERT INTO audit (ts, command_name, params_json, result) VALUES (?, ?, ?, ?)",
        ("2026-09-09T00:00:00+00:00", "probe", f'{{"token": "{secret}"}}', "ok"),
    )
    result = build_bug_report(tmp_path / "report.zip", paths=profile, database=database)
    database.close()
    with zipfile.ZipFile(result.path) as archive:
        contents = "\n".join(archive.read(name).decode() for name in archive.namelist())
    assert secret not in contents
    assert "audit.json" in result.included
    assert "logs/worker_stt_20260909.log" in result.included
