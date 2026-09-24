"""Task 59: overview of what Ayris stores on disk, by category.

Shows each store's path and on-disk size with a button to open its folder, plus
the database's per-table row counts. Sizes are walked off the UI thread so a large
models directory never freezes the tab; tests use ``background=False`` to compute
synchronously and never leave a thread running.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.paths import AppPaths
from ayris.gui.theme import ThemeManager
from ayris.utils.logger import get_logger

__all__ = ["DataInventory", "compute_sizes", "human_size"]

_log = get_logger(__name__)

#: Database tables worth breaking out under «База данных», label → count key.
_DB_TABLES: tuple[tuple[str, str], ...] = (
    ("Команды", "commands"),
    ("История", "history"),
    ("Аудит", "audit"),
    ("Буфер обмена", "clipboard_history"),
    ("Таймеры", "timers"),
    ("Переменные", "variables"),
    ("Версии команд", "command_versions"),
    ("Модели", "models"),
)


@dataclass(frozen=True, slots=True)
class _Target:
    key: str
    title: str
    root: Path
    kind: str  # "file" | "dir" | "backups"


def human_size(num: int) -> str:
    """Bytes as a short Russian-friendly string: «1,2 МБ»."""
    value = float(num)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            rendered = f"{value:.0f}" if unit == "Б" else f"{value:.1f}".replace(".", ",")
            return f"{rendered} {unit}"
        value /= 1024
    return f"{num} Б"


def _dir_size(root: Path) -> int:
    total = 0
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _target_size(target: _Target) -> int:
    root = target.root
    try:
        if target.kind == "file":
            return root.stat().st_size if root.is_file() else 0
        if target.kind == "dir":
            return _dir_size(root) if root.is_dir() else 0
        if target.kind == "backups":
            parent = root.parent
            if not parent.is_dir():
                return 0
            return sum(
                path.stat().st_size
                for path in parent.glob(f"{root.stem}_backup_*.db")
                if path.is_file()
            )
    except OSError:
        return 0
    return 0


def compute_sizes(targets: Sequence[_Target]) -> dict[str, int]:
    """Resolve the on-disk size of every target. Pure — no Qt, safe off the UI thread."""
    return {target.key: _target_size(target) for target in targets}


class _SizeWorker(QObject):
    """Walks the sizes on a daemon thread and reports back on the UI thread."""

    done = Signal(dict)

    def __init__(self, targets: Sequence[_Target], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._targets = tuple(targets)

    def run(self) -> None:
        try:
            sizes = compute_sizes(self._targets)
        except Exception:  # a broken path must not take the tab down
            _log.exception("не удалось посчитать размеры данных")
            sizes = {}
        self.done.emit(sizes)


class DataInventory(QWidget):
    """Category list with paths, sizes and an «Открыть папку» action each."""

    def __init__(
        self,
        paths: AppPaths,
        theme: ThemeManager,
        parent: QWidget | None = None,
        *,
        statistics: Callable[[], dict[str, int]] | None = None,
        open_folder: Callable[[Path], None] | None = None,
        background: bool = True,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._statistics = statistics
        self._open_folder = open_folder if open_folder is not None else _open_in_explorer
        self._targets: tuple[_Target, ...] = (
            _Target("config", "Настройки", paths.config_file, "file"),
            _Target("database", "База данных", paths.database_file, "file"),
            _Target("model_stt", "Модели · распознавание", paths.stt_models_dir, "dir"),
            _Target("model_tts", "Модели · синтез речи", paths.tts_models_dir, "dir"),
            _Target("model_wake", "Модели · активация", paths.wake_models_dir, "dir"),
            _Target("model_llm", "Модели · языковые", paths.llm_models_dir, "dir"),
            _Target("logs", "Логи", paths.logs_dir, "dir"),
            _Target("backups", "Резервные копии БД", paths.database_file, "backups"),
            _Target("drafts", "Черновики редактора", paths.command_drafts_dir, "dir"),
        )
        self._size_labels: dict[str, QLabel] = {}
        self._thread: threading.Thread | None = None

        self._root = QVBoxLayout(self)
        self._grid = QGridLayout()
        self._root.addLayout(self._grid)
        for row, target in enumerate(self._targets):
            self._build_row(row, target)
        self._db_counts = QLabel("")
        self._db_counts.setProperty("role", "secondary")
        self._db_counts.setWordWrap(True)
        self._root.addWidget(self._db_counts)
        self._total = QLabel("")
        self._total.setProperty("role", "h3")
        self._root.addWidget(self._total)

        self._refresh_counts()
        self.refresh(background=background)

    def _build_row(self, row: int, target: _Target) -> None:
        title = QLabel(target.title)
        title.setProperty("role", "h3")
        path_label = QLabel(str(target.root if target.kind != "backups" else target.root.parent))
        path_label.setProperty("role", "secondary")
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        size_label = QLabel("…")
        size_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        button = QPushButton("Открыть папку")
        button.clicked.connect(lambda _=False, t=target: self._open(t))
        self._size_labels[target.key] = size_label
        self._grid.addWidget(title, row, 0)
        self._grid.addWidget(path_label, row, 1)
        self._grid.addWidget(size_label, row, 2)
        self._grid.addWidget(button, row, 3)

    def _open(self, target: _Target) -> None:
        folder = target.root if target.kind == "dir" else target.root.parent
        try:
            self._open_folder(folder)
        except Exception:
            _log.exception("не удалось открыть папку %s", folder)

    def _refresh_counts(self) -> None:
        if self._statistics is None:
            self._db_counts.setText("")
            return
        try:
            stats = self._statistics()
        except Exception:
            _log.exception("не удалось прочитать статистику БД")
            return
        parts = [f"{label}: {stats.get(key, 0)}" for label, key in _DB_TABLES]
        self._db_counts.setText("Записей в таблицах — " + ", ".join(parts))

    def refresh(self, *, background: bool = True) -> None:
        """Recompute sizes; ``background=False`` runs synchronously (used by tests)."""
        if not background:
            self._apply_sizes(compute_sizes(self._targets))
            return
        worker = _SizeWorker(self._targets, parent=self)
        worker.done.connect(self._apply_sizes)
        self._thread = threading.Thread(target=worker.run, daemon=True)
        self._thread.start()

    def _apply_sizes(self, sizes: dict[str, int]) -> None:
        total = 0
        for key, label in self._size_labels.items():
            value = int(sizes.get(key, 0))
            total += value
            label.setText(human_size(value))
        self._total.setText(f"Всего на диске: {human_size(total)}")

    def reload(self, *, background: bool = True) -> None:
        """Recompute table counts and on-disk sizes. Used after a cleanup."""
        self._refresh_counts()
        self.refresh(background=background)

    def wait_for_sizes(self, timeout: float = 5.0) -> None:
        """Join the background size walk. Call before closing to avoid a hung thread."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def dispose(self) -> None:
        self.wait_for_sizes()


def _open_in_explorer(folder: Path) -> None:
    import subprocess
    import sys

    if not folder.exists():
        return
    if sys.platform == "win32":
        import os

        os.startfile(str(folder))  # opening a known local folder chosen from a fixed list
    else:  # pragma: no cover - Ayris ships on Windows
        subprocess.run(["xdg-open", str(folder)], check=False)
