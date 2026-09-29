"""Artifact composition and size report (task 71, §16.1).

A build that quietly grew 200 MB because a transitive dependency started shipping
CUDA wheels is the kind of regression nobody notices until a user downloads it. So
every build writes a manifest — the sorted list of files and their sizes plus the
total — next to the artifact, and, when a previous manifest is handed in, a diff:
what appeared, what vanished, what changed size. The diff is what turns "the exe is
big" into "``onnxruntime_providers_cuda.dll`` (+310 MB) should not be here".

The report is plain data (dataclasses and dicts) so it serialises to JSON for the
next build to diff against, and renders to a short human summary for the build log
and the release notes. No compiler, no filesystem magic — just a walk and some
arithmetic, which is why it is fully unit-tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "FileEntry",
    "Report",
    "ReportDiff",
    "collect_files",
    "diff_reports",
    "human_size",
    "make_report",
    "read_report",
    "render_diff",
    "render_report",
    "write_report",
]

_KIB: Final = 1024
_TOP_FILES: Final = 15


@dataclass(frozen=True, slots=True)
class FileEntry:
    """One file in the artifact: path relative to the artifact root, and its size."""

    path: str
    size: int


@dataclass(frozen=True, slots=True)
class Report:
    """A whole artifact measured: its files, their count and total size."""

    total_bytes: int
    file_count: int
    files: tuple[FileEntry, ...]


@dataclass(frozen=True, slots=True)
class ReportDiff:
    """What changed between two artifacts."""

    added: tuple[FileEntry, ...]
    removed: tuple[FileEntry, ...]
    #: (path, old_size, new_size) for files present in both with a different size.
    changed: tuple[tuple[str, int, int], ...]
    total_delta: int


def human_size(num_bytes: int) -> str:
    """A compact human string: ``1536`` -> ``1.5 KiB``, ``0`` -> ``0 B``."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < _KIB or unit == "TiB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= _KIB
    # unreachable: the loop always returns on the "TiB" iteration
    raise AssertionError


def collect_files(root: Path) -> list[FileEntry]:
    """Every regular file under *root*, sorted by path, sizes in bytes.

    Symlinks are followed only as far as ``stat`` does; the artifact tree Nuitka
    writes has none. Paths use forward slashes so a manifest written on Windows
    diffs cleanly against one read anywhere.
    """
    entries: list[FileEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            entries.append(
                FileEntry(path=path.relative_to(root).as_posix(), size=path.stat().st_size)
            )
    return entries


def make_report(root: Path) -> Report:
    """Measure the artifact rooted at *root*."""
    files = tuple(collect_files(root))
    return Report(
        total_bytes=sum(entry.size for entry in files),
        file_count=len(files),
        files=files,
    )


def _report_to_dict(report: Report) -> dict[str, object]:
    return {
        "total_bytes": report.total_bytes,
        "file_count": report.file_count,
        "files": [{"path": entry.path, "size": entry.size} for entry in report.files],
    }


def write_report(report: Report, path: Path) -> None:
    """Serialise *report* to JSON at *path* for the next build to diff against."""
    path.write_text(
        json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_report(path: Path) -> Report:
    """Load a report previously written by :func:`write_report`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    files = tuple(
        FileEntry(path=str(item["path"]), size=int(item["size"])) for item in data["files"]
    )
    return Report(
        total_bytes=int(data["total_bytes"]),
        file_count=int(data["file_count"]),
        files=files,
    )


def render_report(report: Report) -> str:
    """A short human summary: total, count, and the largest files."""
    lines = [
        f"Размер артефакта: {human_size(report.total_bytes)} ({report.total_bytes} байт)",
        f"Файлов: {report.file_count}",
        f"Крупнейшие {min(_TOP_FILES, report.file_count)}:",
    ]
    largest = sorted(report.files, key=lambda entry: entry.size, reverse=True)[:_TOP_FILES]
    lines.extend(f"  {human_size(entry.size):>10}  {entry.path}" for entry in largest)
    return "\n".join(lines)


def diff_reports(previous: Report, current: Report) -> ReportDiff:
    """Compare two artifacts: what was added, removed, or changed size."""
    old = {entry.path: entry.size for entry in previous.files}
    new = {entry.path: entry.size for entry in current.files}

    added = tuple(FileEntry(path=path, size=new[path]) for path in sorted(new.keys() - old.keys()))
    removed = tuple(
        FileEntry(path=path, size=old[path]) for path in sorted(old.keys() - new.keys())
    )
    changed = tuple(
        (path, old[path], new[path])
        for path in sorted(old.keys() & new.keys())
        if old[path] != new[path]
    )
    return ReportDiff(
        added=added,
        removed=removed,
        changed=changed,
        total_delta=current.total_bytes - previous.total_bytes,
    )


def render_diff(diff: ReportDiff) -> str:
    """A human summary of a diff, or a single reassuring line when nothing moved."""
    sign = "+" if diff.total_delta >= 0 else "−"
    header = f"Изменение размера: {sign}{human_size(abs(diff.total_delta))}"
    if not diff.added and not diff.removed and not diff.changed:
        return f"{header}\nСостав артефакта не изменился."
    lines = [header]
    for entry in diff.added:
        lines.append(f"  + {human_size(entry.size):>10}  {entry.path}")
    for entry in diff.removed:
        lines.append(f"  − {human_size(entry.size):>10}  {entry.path}")
    for path, old_size, new_size in diff.changed:
        delta = new_size - old_size
        mark = "+" if delta >= 0 else "−"
        lines.append(f"  ~ {mark}{human_size(abs(delta)):>9}  {path}")
    return "\n".join(lines)
