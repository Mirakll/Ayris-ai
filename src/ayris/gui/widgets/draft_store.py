"""Autosave of unsaved command edits, task 54.

An editor session can be lost — the app crashes, the machine reboots, the user
closes the window on a «потом допишу». The draft store keeps the working command
on disk so the next open of that command can offer it back. It is deliberately
*not* the database: a draft is unsaved by definition, must not show up in a
profile export, and must not cost a write on every keystroke.

One file per command id, under the profile's ``cache/command_drafts``. The file is
the command serialised with the same ``.ayris`` document writer the export uses,
so a draft is a valid command and reading it back is the reader that already
exists. A new, never-saved command has no id yet; its draft is keyed by ``0`` only
if the caller asks, and the common path is a draft for an existing command.

The store does no timing itself — it reads, writes and deletes. The editor owns
the interval timer (from ``commands.draft_autosave_s``) and calls :meth:`save`
when it fires and the model is dirty. Keeping the clock out of here is what lets
the tests write and read a draft without a running event loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ayris.actions.macros.serializer import (
    AYRIS_SUFFIX,
    dump_command,
    load_command,
)
from ayris.core.models import from_db_timestamp, to_db_timestamp, utc_now
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ayris.actions.macros.schema import CommandModel

__all__ = ["DraftRecord", "DraftStore"]

_log = get_logger(__name__)

#: Sidecar suffix for the timestamp of a draft, beside its ``.ayris`` file.
_STAMP_SUFFIX = ".saved_at"


@dataclass(frozen=True, slots=True)
class DraftRecord:
    """A recovered draft: the command as last autosaved, and when."""

    command: CommandModel
    saved_at: datetime


class DraftStore:
    """Reads, writes and clears per-command autosave drafts on disk."""

    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def save(self, command: CommandModel) -> None:
        """Write the working command as its draft. Silent on I/O error.

        A draft that cannot be written is a lost recovery, not a lost command —
        the save button still works — so a full disk logs and moves on rather than
        interrupting the edit.
        """
        command_id = command.id if command.id is not None else 0
        stamp = to_db_timestamp(utc_now())
        assert stamp is not None  # utc_now() всегда даёт дату — None только для None-входа
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            self._file(command_id).write_text(dump_command(command), encoding="utf-8")
            self._stamp(command_id).write_text(stamp, encoding="utf-8")
        except OSError:
            _log.warning("не удалось сохранить черновик команды %s", command_id, exc_info=True)

    def load(self, command_id: int) -> DraftRecord | None:
        """The draft for a command, or ``None`` when there is none or it is unreadable.

        An unreadable draft (a truncated write, a format from a newer build) is
        deleted and reported as absent: a recovery prompt for a file that will not
        load is worse than no prompt.
        """
        path = self._file(command_id)
        if not path.exists():
            return None
        try:
            command = load_command(path.read_text(encoding="utf-8"))
        except Exception:
            _log.warning("черновик команды %s не читается, удаляю", command_id, exc_info=True)
            self.discard(command_id)
            return None
        return DraftRecord(command=command, saved_at=self._read_stamp(command_id))

    def has_draft(self, command_id: int) -> bool:
        """Whether a draft file exists for a command, without parsing it."""
        return self._file(command_id).exists()

    def discard(self, command_id: int) -> None:
        """Delete a command's draft and its timestamp. Safe when absent.

        Called after a successful save (the draft is now redundant) and when the
        user declines to restore one.
        """
        for path in (self._file(command_id), self._stamp(command_id)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                _log.warning("не удалось удалить черновик %s", path, exc_info=True)

    # -- internals ----------------------------------------------------------

    def _file(self, command_id: int) -> Path:
        return self._dir / f"{command_id}{AYRIS_SUFFIX}"

    def _stamp(self, command_id: int) -> Path:
        return self._dir / f"{command_id}{_STAMP_SUFFIX}"

    def _read_stamp(self, command_id: int) -> datetime:
        try:
            raw = self._stamp(command_id).read_text(encoding="utf-8").strip()
            parsed = from_db_timestamp(raw)
            if parsed is not None:
                return parsed
        except (OSError, ValueError):
            pass
        return utc_now()
