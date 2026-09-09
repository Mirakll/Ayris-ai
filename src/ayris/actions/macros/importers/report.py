"""Human-readable import reports."""

from __future__ import annotations

from pathlib import Path

from ayris.actions.macros.importers.base import ApplyReport, ImportResult


def render_report(result: ImportResult, applied: ApplyReport | None = None) -> str:
    """Render a preview or final report suitable for the UI and a text file."""
    summary = applied.summary if applied is not None else result.summary
    lines = [summary, f"Источник: {result.source}"]
    if result.profile_name:
        lines.append(f"Профиль: {result.profile_name}")
    skipped = applied.skipped if applied is not None else dict(result.skipped)
    if skipped:
        lines.extend(
            ["", "Пропущено:"] + [f"- {reason}: {count}" for reason, count in skipped.items()]
        )
    if result.warnings:
        lines.extend(["", "Предупреждения:"] + [f"- {item.text}" for item in result.warnings])
    if result.unsupported:
        lines.extend(["", "Неподдерживаемое:"] + [f"- {item.text}" for item in result.unsupported])
    return "\n".join(lines) + "\n"


def write_report(path: Path, result: ImportResult, applied: ApplyReport | None = None) -> None:
    """Export a UTF-8 report without touching imported source files."""
    path.write_text(render_report(result, applied), encoding="utf-8", newline="\n")
