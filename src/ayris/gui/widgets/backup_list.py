"""The list of automatic and manual backups, with restore and delete.

A backup is a portable ``.zip`` written by :meth:`ProfileManager.backup` into the
``backups`` folder, named ``{profile}_{reason}_{stamp}.zip``. The list parses that
name back into a date and a reason — «вручную», «перед импортом», «перед сбросом» —
so a row reads as what it is without opening the archive. :func:`parse_backup_name`
is pure and total, so the parsing is tested without a widget.

The widget owns no file logic: «Восстановить» and «Удалить» only emit the path, and
the tab runs the slow, irreversible work in a background thread with a confirmation
first, exactly as the task requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.models.downloader import human_size

__all__ = ["REASON_LABELS", "BackupInfo", "BackupList", "parse_backup_name"]

#: The reasons :meth:`ProfileManager.backup` stamps into a file name, in Russian.
REASON_LABELS: dict[str, str] = {
    "manual": "вручную",
    "import": "перед импортом",
    "reset": "перед сбросом",
}


@dataclass(frozen=True, slots=True)
class BackupInfo:
    """One backup archive, described from its file name and size on disk."""

    path: Path
    created: datetime | None
    reason: str
    size: int

    @property
    def reason_label(self) -> str:
        return REASON_LABELS.get(self.reason, self.reason or "неизвестно")

    @property
    def created_label(self) -> str:
        return self.created.strftime("%d.%m.%Y %H:%M") if self.created else "дата неизвестна"

    @property
    def size_label(self) -> str:
        return human_size(self.size) if self.size else "—"


def parse_backup_name(path: Path) -> BackupInfo:
    """Read ``{profile}_{reason}_{YYYYMMDD}_{HHMMSS}.zip`` into a :class:`BackupInfo`.

    Parsed from the right so a profile name with its own underscores survives: the
    last two parts are the date and time, the one before them is the reason. A name
    that does not fit the shape still yields a record — with an unknown date and an
    empty reason — rather than raising, because a stray file in the folder must not
    take the list down.
    """
    stem = path.stem
    reason = ""
    created: datetime | None = None
    parts = stem.rsplit("_", 3)
    if len(parts) == 4 and parts[2].isdigit() and parts[3].isdigit():
        reason = parts[1]
        try:
            created = datetime.strptime(f"{parts[2]}_{parts[3]}", "%Y%m%d_%H%M%S")
        except ValueError:
            created = None
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return BackupInfo(path=path, created=created, reason=reason, size=size)


class BackupList(QWidget):
    """A vertical list of backup rows; the tab wires the two signals to actions."""

    restore_requested = Signal(Path)
    delete_requested = Signal(Path)

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setAccessibleName("Список резервных копий")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel("Резервных копий пока нет.")
        self._empty.setProperty("role", "secondary")
        self._empty.setWordWrap(True)
        self._layout.addWidget(self._empty)
        self._rows: list[_BackupRow] = []
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def set_backups(self, paths: list[Path]) -> None:
        """Rebuild the list from ``paths`` (newest first, as the manager returns)."""
        for row in self._rows:
            row.setParent(None)
            row.deleteLater()
        self._rows.clear()
        for path in paths:
            row = _BackupRow(parse_backup_name(path), self._theme)
            row.restore_requested.connect(self.restore_requested)
            row.delete_requested.connect(self.delete_requested)
            self._layout.addWidget(row)
            self._rows.append(row)
        self._empty.setVisible(not paths)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self._layout.setSpacing(self._theme.metric("spacing_sm"))


class _BackupRow(QFrame):
    """A single backup: date, reason and size on the left, two actions on the right."""

    restore_requested = Signal(Path)
    delete_requested = Signal(Path)

    def __init__(
        self, info: BackupInfo, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._info = info
        self.setProperty("card", True)
        self.setAccessibleName(f"Копия {info.created_label}, {info.reason_label}")
        self._layout = QHBoxLayout(self)
        texts = QVBoxLayout()
        title = QLabel(f"{info.created_label} — {info.reason_label}")
        title.setProperty("role", "h2")
        detail = QLabel(f"Размер: {info.size_label}")
        detail.setProperty("role", "secondary")
        texts.addWidget(title)
        texts.addWidget(detail)
        self._layout.addLayout(texts, 1)
        restore = QPushButton("Восстановить")
        restore.setProperty("kind", "primary")
        restore.clicked.connect(lambda: self.restore_requested.emit(info.path))
        delete = QPushButton("Удалить")
        delete.clicked.connect(lambda: self.delete_requested.emit(info.path))
        self._layout.addWidget(restore)
        self._layout.addWidget(delete)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_md")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_sm"))
