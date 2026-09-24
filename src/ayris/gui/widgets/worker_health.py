"""Worker health panel for the «Логи / DevTools» tab (task 58).

Task 05 runs each subsystem — audio, STT, TTS, the LLM — in its own process under
a :class:`~ayris.workers.manager.WorkerManager`. This panel is the window into
that supervisor: per worker its kind, status, uptime, restart count, resident
memory and the age of its last heartbeat, with buttons to restart or pause one.

Two facts from the task shape it.

*The supervisor does not report uptime or RAM.* A :class:`WorkerSummary` is a
snapshot of counters, not a process monitor: it has a pid but no start time and
no memory figure. So :class:`WorkerHealthModel` derives uptime by watching a
worker's pid across polls — a new pid resets the clock — and reads RAM through
the same psutil sampler the resource panel uses.

*Restart and pause touch the live pipeline.* Bouncing the audio worker mid-phrase
aborts whatever the user was saying, so both buttons warn first when a session is
active, and only act if the warning is accepted.

:class:`WorkerHealthModel` holds the uptime bookkeeping and the highlight rule
without a widget, so the tests read it directly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.resource_monitor import PsutilSampler, Sampler
from ayris.utils.logger import get_logger
from ayris.workers.manager import WorkerStatus, WorkerSummary

__all__ = [
    "WorkerHealth",
    "WorkerHealthModel",
    "WorkerHealthRow",
    "WorkerSupervisor",
]

_log = get_logger(__name__)

#: How often the panel repolls the supervisor while it is visible.
_POLL_MS: Final = 1500
#: Restart count at or above which a worker is flagged as flapping.
_RESTART_HOT: Final = 3
_COLUMNS: Final = ("Воркер", "Тип", "Статус", "Аптайм", "Перезапуски", "RAM", "Пульс")
#: Row-widget keys, paired with :data:`_COLUMNS` column by column.
_ROW_KEYS: Final = ("name", "kind", "status", "uptime", "restarts", "ram", "heartbeat")


def _format_uptime(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours} ч {minutes} мин"
    if minutes:
        return f"{minutes} мин {secs} с"
    return f"{secs} с"


def _format_ram(rss_bytes: int | None) -> str:
    if rss_bytes is None:
        return "—"
    return f"{rss_bytes / (1024 * 1024):.0f} МБ"


def _format_heartbeat(age: float | None) -> str:
    if age is None:
        return "—"
    return f"{age:.0f} с назад"


@dataclass(frozen=True, slots=True)
class WorkerHealthRow:
    """One worker at one instant, with the figures the summary does not carry."""

    name: str
    kind: str
    status: WorkerStatus
    pid: int | None
    restarts: int
    uptime_s: float | None
    ram_bytes: int | None
    heartbeat_age: float | None
    error: str
    hot: bool

    @property
    def status_text(self) -> str:
        return self.status.label

    @property
    def uptime_text(self) -> str:
        return _format_uptime(self.uptime_s)

    @property
    def ram_text(self) -> str:
        return _format_ram(self.ram_bytes)

    @property
    def heartbeat_text(self) -> str:
        return _format_heartbeat(self.heartbeat_age)

    @property
    def is_live(self) -> bool:
        return self.status.is_live


class WorkerHealthModel:
    """Turns supervisor snapshots into rows, deriving uptime and RAM per worker.

    Args:
        sampler: Reads resident memory for a pid; ``None`` leaves RAM blank, which
            is what tests that only check uptime use.
        clock: Monotonic seconds, injected so uptime assertions are exact.
        restart_threshold: Restart count at which a worker is flagged as flapping.
    """

    def __init__(
        self,
        *,
        sampler: Sampler | None = None,
        clock: Callable[[], float] = time.monotonic,
        restart_threshold: int = _RESTART_HOT,
    ) -> None:
        self._sampler = sampler
        self._clock = clock
        self._threshold = restart_threshold
        # worker name -> (pid seen, monotonic time that pid first appeared)
        self._since: dict[str, tuple[int, float]] = {}
        self._rows: list[WorkerHealthRow] = []

    @property
    def rows(self) -> list[WorkerHealthRow]:
        return list(self._rows)

    def update(self, summaries: Sequence[WorkerSummary]) -> list[WorkerHealthRow]:
        """Fold a fresh snapshot into rows, keeping the uptime bookkeeping current."""
        now = self._clock()
        rows: list[WorkerHealthRow] = []
        seen: set[str] = set()
        for summary in summaries:
            seen.add(summary.name)
            uptime = self._track_uptime(summary, now)
            ram = self._sample_ram(summary.pid)
            rows.append(
                WorkerHealthRow(
                    name=summary.name,
                    kind=summary.kind,
                    status=summary.status,
                    pid=summary.pid,
                    restarts=summary.restarts,
                    uptime_s=uptime,
                    ram_bytes=ram,
                    heartbeat_age=summary.last_heartbeat_age,
                    error=summary.error,
                    hot=summary.restarts >= self._threshold,
                )
            )
        for gone in [name for name in self._since if name not in seen]:
            del self._since[gone]
        self._rows = rows
        return rows

    def _track_uptime(self, summary: WorkerSummary, now: float) -> float | None:
        if summary.pid is None or not summary.status.is_live:
            self._since.pop(summary.name, None)
            return None
        previous = self._since.get(summary.name)
        if previous is None or previous[0] != summary.pid:
            self._since[summary.name] = (summary.pid, now)
            return 0.0
        return now - previous[1]

    def _sample_ram(self, pid: int | None) -> int | None:
        if self._sampler is None or pid is None:
            return None
        reading = self._sampler.sample(pid)
        return reading[0] if reading is not None else None


class WorkerSupervisor(Protocol):
    """What the panel needs from the running :class:`WorkerManager`.

    A ``Protocol`` so the manager satisfies it structurally — the workers layer
    must not import the GUI — while a test double implements it directly.
    """

    def status(self) -> tuple[WorkerSummary, ...]:
        """Snapshot of every registered worker."""
        ...

    def restart(self, name: str, *, reason: str = "") -> None:
        """Restart one worker."""
        ...

    def stop(self, name: str, *, timeout: float | None = None) -> None:
        """Pause (stop) one worker."""
        ...

    def start(self, name: str, *, timeout: float | None = None) -> None:
        """Start a worker that is paused or not yet up."""
        ...


@dataclass(slots=True)
class _RowWidgets:
    """The widgets of one worker row, so a refresh updates them in place."""

    labels: dict[str, QLabel]
    restart_button: QPushButton
    pause_button: QPushButton


class WorkerHealth(QFrame):
    """A card of per-worker health with restart and pause controls.

    Args:
        theme: Active theme, for metrics and the flapping-worker highlight.
        control: The running supervisor; ``None`` shows an empty, inert panel,
            which is what the settings window uses before workers exist.
        sampler: RAM sampler for the model; defaults to a real psutil one.
        is_session_active: Returns whether a pipeline session is running, so the
            buttons can warn before they interrupt it. ``None`` never warns.
    """

    def __init__(
        self,
        theme: ThemeManager,
        *,
        control: WorkerSupervisor | None = None,
        sampler: Sampler | None = None,
        is_session_active: Callable[[], bool] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._control = control
        self._is_session_active = is_session_active
        self._model = WorkerHealthModel(sampler=sampler if sampler is not None else PsutilSampler())
        self._names: list[str] = []
        self._cells: dict[str, _RowWidgets] = {}

        self.setProperty("card", True)
        self.setAccessibleName("Здоровье воркеров")
        self._outer = QVBoxLayout(self)
        self._title = QLabel("Воркеры")
        self._title.setProperty("role", "h2")
        self._outer.addWidget(self._title)
        self._grid = QGridLayout()
        self._grid.setColumnStretch(0, 1)
        self._outer.addLayout(self._grid)
        self._empty = QLabel("Супервизор воркеров не запущен.")
        self._empty.setProperty("role", "secondary")
        self._outer.addWidget(self._empty)

        from PySide6.QtCore import QTimer

        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_MS)
        self._timer.timeout.connect(self.refresh)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    # -- polling lifecycle --------------------------------------------------

    def is_polling(self) -> bool:
        """Whether the refresh timer is running. The visibility contract, for tests."""
        return self._timer.isActive()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        """Poll the supervisor, fold the snapshot into rows and repaint them."""
        if self._control is None:
            return
        rows = self._model.update(self._control.status())
        self._reconcile([row.name for row in rows])
        for row in rows:
            self._update_row(row)

    # -- grid ---------------------------------------------------------------

    def _reconcile(self, names: list[str]) -> None:
        """Rebuild the grid only when the set of workers changes, like task 58's monitor."""
        if names == self._names:
            return
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cells.clear()
        self._names = list(names)
        self._empty.setVisible(not names)
        if not names:
            return
        for col, title in enumerate(_COLUMNS):
            header = QLabel(title)
            header.setProperty("role", "secondary")
            self._grid.addWidget(header, 0, col)
        for row_index, name in enumerate(names, start=1):
            self._build_row(row_index, name)

    def _build_row(self, row_index: int, name: str) -> None:
        labels: dict[str, QLabel] = {}
        for col, key in enumerate(_ROW_KEYS):
            label = QLabel("—")
            labels[key] = label
            self._grid.addWidget(label, row_index, col)
        restart = QPushButton("Перезапустить")
        restart.clicked.connect(lambda: self._on_restart(name))
        self._grid.addWidget(restart, row_index, len(_COLUMNS))
        pause = QPushButton("Пауза")
        pause.clicked.connect(lambda: self._on_pause(name))
        self._grid.addWidget(pause, row_index, len(_COLUMNS) + 1)
        self._cells[name] = _RowWidgets(labels=labels, restart_button=restart, pause_button=pause)

    def _update_row(self, row: WorkerHealthRow) -> None:
        cells = self._cells.get(row.name)
        if cells is None:
            return
        cells.labels["name"].setText(row.name)
        cells.labels["kind"].setText(row.kind)
        cells.labels["status"].setText(row.status_text)
        cells.labels["uptime"].setText(row.uptime_text)
        cells.labels["restarts"].setText(str(row.restarts))
        cells.labels["ram"].setText(row.ram_text)
        cells.labels["heartbeat"].setText(row.heartbeat_text)
        cells.labels["restarts"].setStyleSheet(
            f"color: {self._theme.theme.color('error')};" if row.hot else ""
        )
        cells.pause_button.setText("Пауза" if row.is_live else "Запустить")

    # -- handlers -----------------------------------------------------------

    def _row_for(self, name: str) -> WorkerHealthRow | None:
        for row in self._model.rows:
            if row.name == name:
                return row
        return None

    def _on_restart(self, name: str) -> None:
        if self._control is None:
            return
        if not self._confirm_interrupt("перезапустить", name):
            return
        self._control.restart(name, reason="devtools")
        self.refresh()

    def _on_pause(self, name: str) -> None:
        if self._control is None:
            return
        row = self._row_for(name)
        if row is not None and row.is_live:
            if not self._confirm_interrupt("приостановить", name):
                return
            self._control.stop(name)
        else:
            self._control.start(name)
        self.refresh()

    def _confirm_interrupt(self, verb: str, name: str) -> bool:
        """Warn that a live session dies if the worker is bounced; ``True`` to proceed."""
        if self._is_session_active is None or not self._is_session_active():
            return True
        dialog = ConfirmDialog(
            "Идёт активная сессия",
            f"Сейчас идёт голосовая сессия. Если {verb} воркер «{name}», она будет "
            "прервана. Продолжить?",
            self._theme,
            confirm_text="Прервать и продолжить",
            dangerous=True,
            parent=self,
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    # -- lifecycle ----------------------------------------------------------

    @property
    def model(self) -> WorkerHealthModel:
        return self._model

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        gap = self._theme.metric("spacing_sm")
        pad = self._theme.metric("spacing_md")
        self._outer.setContentsMargins(pad, pad, pad, pad)
        self._outer.setSpacing(gap)
        self._grid.setHorizontalSpacing(pad)
        self._grid.setVerticalSpacing(gap)

    def stop(self) -> None:
        """Stop the poll timer. A live timer keeps the event loop from ending."""
        if self._timer.isActive():
            self._timer.stop()

    def dispose(self) -> None:
        self.stop()
