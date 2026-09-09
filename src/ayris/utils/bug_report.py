"""Local, redacted support archive builder. Nothing is uploaded."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import tomlkit

from ayris import __app_name__, __version__
from ayris.core.database import Database
from ayris.core.paths import AppPaths, get_paths
from ayris.utils.redaction import redact_text

__all__ = ["BugReportResult", "build_bug_report", "open_report_folder"]

_SECRET_KEYS = ("password", "secret", "token", "api_key", "apikey", "credential")


@dataclass(frozen=True, slots=True)
class BugReportResult:
    path: Path
    included: tuple[str, ...]


def _sanitize(value: Any, key: str = "") -> Any:
    folded = key.casefold()
    if any(marker in folded for marker in _SECRET_KEYS) and not folded.endswith("credential_ref"):
        return "[скрыто]"
    if isinstance(value, dict):
        return {str(name): _sanitize(item, str(name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _config_text(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        parsed = tomlkit.parse(path.read_text(encoding="utf-8-sig")).unwrap()
        return tomlkit.dumps(_sanitize(parsed))
    except (OSError, UnicodeError, ValueError):
        return path.read_text(encoding="utf-8", errors="replace")


def _audio_devices() -> list[dict[str, Any]]:
    try:
        import sounddevice as sd

        return [dict(device) for device in sd.query_devices()]
    except Exception:
        return []


def _system_info() -> str:
    packages: dict[str, str] = {}
    for name in ("PySide6", "pydantic", "sounddevice", "numpy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return json.dumps(
        {
            "application": __app_name__,
            "version": __version__,
            "python": sys.version,
            "os": platform.platform(),
            "packages": packages,
            "audio_devices": _audio_devices(),
        },
        ensure_ascii=False,
        indent=2,
    )


def _recent(path: Path, days: int) -> bool:
    if days <= 0:
        return True
    cutoff = datetime.now(UTC) - timedelta(days=days)
    return datetime.fromtimestamp(path.stat().st_mtime, UTC) >= cutoff


def build_bug_report(
    destination: Path | str,
    *,
    days: int = 7,
    paths: AppPaths | None = None,
    database: Database | None = None,
) -> BugReportResult:
    """Build a local zip containing redacted logs, audit rows and settings."""
    resolved = paths if paths is not None else get_paths()
    target = Path(destination)
    if target.suffix.casefold() != ".zip":
        target /= f"ayris_bug_report_{datetime.now():%Y%m%d_%H%M%S}.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    included: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ayris-report-") as temporary:
        temp = Path(temporary)
        sources: list[tuple[str, str]] = []
        for log_file in sorted(resolved.logs_dir.glob("*.log")):
            if _recent(log_file, days):
                sources.append(
                    (
                        f"logs/{log_file.name}",
                        log_file.read_text(encoding="utf-8", errors="replace"),
                    )
                )
        sources.append(("config.toml", _config_text(resolved.config_file)))
        sources.append(("system.json", _system_info()))

        owned_db = database is None
        db = database
        try:
            if db is None and resolved.database_file.exists():
                db = Database.open(resolved.database_file, migrate=False)
            if db is not None:
                rows = db.query_all("SELECT * FROM audit ORDER BY ts DESC, id DESC")
                sources.append(
                    (
                        "audit.json",
                        json.dumps([dict(row) for row in rows], ensure_ascii=False, indent=2),
                    )
                )
        finally:
            if owned_db and db is not None:
                db.close()

        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in sources:
                safe = redact_text(content)
                staged = temp / Path(name).name
                staged.write_text(safe, encoding="utf-8")
                archive.write(staged, name)
                included.append(name)
    return BugReportResult(target, tuple(included))


def open_report_folder(report: Path | str) -> None:
    """Open the containing folder only when the UI explicitly requests it."""
    folder = str(Path(report).resolve().parent)
    if sys.platform == "win32":
        os.startfile(folder)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", folder])
    else:
        subprocess.Popen(["xdg-open", folder])
