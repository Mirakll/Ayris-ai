"""Live RAM/CPU panel for the «Общие» tab, per process and per worker.

Two rules shape this widget, both from task 48.

*It polls only while it can be seen.* Sampling every process every second is cheap
but not free, and a settings window spends most of its life behind the dashboard.
The :class:`QTimer` is started in :meth:`showEvent` and stopped in
:meth:`hideEvent`, so a hidden panel costs nothing.

*It never owns a worker.* The panel reads worker pids and statuses through a
:class:`WorkerControl` — the running :class:`~ayris.workers.manager.WorkerManager`
registers itself with :func:`set_active_worker_control`, and when nothing has, the
panel simply shows the main process and no worker rows. That keeps the widget
testable and lets the settings window open long before (or without) a supervisor.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QVBoxLayout, QWidget

from ayris.gui.theme import ThemeManager
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.config import RestartScope
    from ayris.workers.manager import WorkerSummary

__all__ = [
    "ResourceMonitor",
    "ResourceRow",
    "Sampler",
    "WorkerControl",
    "active_worker_control",
    "set_active_worker_control",
]

_log = get_logger(__name__)

#: How often the panel resamples while visible. Slow enough to be invisible on a
#: weak machine, fast enough that a leak shows up while the user is still looking.
_POLL_MS: Final = 1500


@dataclass(frozen=True, slots=True)
class ResourceRow:
    """One line in the panel: a process to sample, or a worker that is not up.

    Args:
        key: Stable identity, so a row keeps its place between refreshes.
        label: Human-readable name shown on the left.
        pid: Process to sample, or ``None`` for a worker that is not running.
        note: Replaces the numbers when there is no process, e.g. «не запущен».
    """

    key: str
    label: str
    pid: int | None
    note: str = ""


class Sampler:
    """Reads RSS and CPU for a pid. A class, not a protocol, so tests subclass it."""

    def sample(self, pid: int) -> tuple[int, float] | None:
        """Resident bytes and CPU percent for ``pid``, or ``None`` if it is gone."""
        raise NotImplementedError


class PsutilSampler(Sampler):
    """The real sampler. Keeps one :class:`psutil.Process` per pid for CPU deltas.

    ``cpu_percent()`` measures the busy fraction since the previous call on the
    same object, so the first reading of any process is ``0.0`` and the cached
    handles are what make the following ones meaningful. The percentage is divided
    by the core count, so a fully busy quad-core reads 100, not 400.
    """

    def __init__(self) -> None:
        try:
            import psutil
        except ImportError:  # pragma: no cover - psutil is a pinned dependency
            _log.warning("psutil недоступен, панель ресурсов работать не будет")
            self._psutil = None
            self._cores = 1
        else:
            self._psutil = psutil
            self._cores = max(1, psutil.cpu_count() or 1)
        self._processes: dict[int, Any] = {}

    def sample(self, pid: int) -> tuple[int, float] | None:
        if self._psutil is None:
            return None
        process = self._processes.get(pid)
        try:
            if process is None:
                process = self._psutil.Process(pid)
                self._processes[pid] = process
                process.cpu_percent(None)  # Prime the delta; this call reads 0.
            with process.oneshot():
                rss = int(process.memory_info().rss)
                cpu = float(process.cpu_percent(None)) / self._cores
        except Exception:  # psutil raises many kinds; a dead pid is normal here
            self._processes.pop(pid, None)
            return None
        return rss, cpu


class ResourceMonitor(QFrame):
    """A card showing memory and CPU for the app and each running worker.

    Args:
        theme: Active theme, for metrics and the over-limit highlight colour.
        sampler: Where the numbers come from. Defaults to a real psutil sampler;
            tests inject a fake one.
        sources: Returns the worker rows to show beneath the main process. The
            main process is always the first row and is not returned here.
        ram_limit_mb: Soft cap the total is compared against; ``0`` disables the
            highlight. Update it live with :meth:`set_ram_limit`.
    """

    def __init__(
        self,
        theme: ThemeManager,
        *,
        sampler: Sampler | None = None,
        sources: Callable[[], Sequence[ResourceRow]] | None = None,
        ram_limit_mb: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._sampler = sampler if sampler is not None else PsutilSampler()
        self._sources = sources
        self._ram_limit_mb = ram_limit_mb
        self._pid = os.getpid()
        self._rows: list[str] = []

        self.setProperty("card", True)
        self.setAccessibleName("Текущее потребление ресурсов")
        self._outer = QVBoxLayout(self)
        self._title = QLabel("Текущее потребление")
        self._title.setProperty("role", "h2")
        self._outer.addWidget(self._title)

        self._grid = QGridLayout()
        self._grid.setColumnStretch(0, 1)
        self._outer.addLayout(self._grid)

        self._summary = QLabel()
        self._summary.setProperty("role", "secondary")
        self._outer.addWidget(self._summary)

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

    # -- configuration ------------------------------------------------------

    def set_ram_limit(self, ram_limit_mb: int) -> None:
        """Change the soft cap the total is checked against."""
        self._ram_limit_mb = ram_limit_mb
        if self._timer.isActive():
            self.refresh()

    # -- refresh ------------------------------------------------------------

    def refresh(self) -> None:
        """Resample every row and update the grid and the total."""
        rows = [ResourceRow(key="__app__", label="Ayris — главный процесс", pid=self._pid)]
        if self._sources is not None:
            rows.extend(self._sources())
        self._reconcile([row.key for row in rows])

        total_rss = 0
        for index, row in enumerate(rows):
            reading = self._sampler.sample(row.pid) if row.pid is not None else None
            name_label, mem_label, cpu_label = self._cells[row.key]
            name_label.setText(row.label)
            if reading is None:
                mem_label.setText(row.note or "нет данных")
                cpu_label.setText("")
            else:
                rss, cpu = reading
                total_rss += rss
                mem_label.setText(_format_mb(rss))
                cpu_label.setText(f"{cpu:.0f}%")
            del index

        self._update_summary(total_rss)

    def _update_summary(self, total_rss: int) -> None:
        limit_bytes = self._ram_limit_mb * 1024 * 1024
        over = self._ram_limit_mb > 0 and total_rss > limit_bytes
        if over:
            self._summary.setText(
                f"Всего: {_format_mb(total_rss)} — превышен лимит {self._ram_limit_mb // 1024} ГБ"
            )
        else:
            self._summary.setText(f"Всего: {_format_mb(total_rss)}")
        self._summary.setProperty("status", "warning" if over else "")
        self._repolish(self._summary)

    def _reconcile(self, keys: list[str]) -> None:
        """Rebuild the grid only when the set of rows actually changed."""
        if keys == self._rows:
            return
        self._rows = list(keys)
        self._cells: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        for column, heading in enumerate(("Процесс", "Память", "ЦП")):
            header = QLabel(heading)
            header.setProperty("role", "muted")
            align = Qt.AlignmentFlag.AlignLeft if column == 0 else Qt.AlignmentFlag.AlignRight
            self._grid.addWidget(header, 0, column, align)
        for offset, key in enumerate(keys, start=1):
            name_label = QLabel()
            mem_label = QLabel()
            mem_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            cpu_label = QLabel()
            cpu_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self._grid.addWidget(name_label, offset, 0)
            self._grid.addWidget(mem_label, offset, 1)
            self._grid.addWidget(cpu_label, offset, 2)
            self._cells[key] = (name_label, mem_label, cpu_label)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_lg")
        self._outer.setContentsMargins(pad, pad, pad, pad)
        self._outer.setSpacing(self._theme.metric("spacing_md"))
        self._grid.setHorizontalSpacing(self._theme.metric("spacing_lg"))
        self._grid.setVerticalSpacing(self._theme.metric("spacing_sm"))

    def _repolish(self, widget: QWidget) -> None:
        style = widget.style()
        if style is not None:
            style.unpolish(widget)
            style.polish(widget)


def _format_mb(rss_bytes: int) -> str:
    """Bytes as whole megabytes, the granularity that matters for a soft cap."""
    return f"{rss_bytes / (1024 * 1024):.0f} МБ"


# ----------------------------------------------------------------------
# worker control: the running supervisor registers itself here
# ----------------------------------------------------------------------


class WorkerControl(Protocol):
    """What the settings tab needs from the supervisor: statuses and restarts.

    A ``Protocol`` so the real :class:`~ayris.workers.manager.WorkerManager`
    satisfies it structurally — the workers layer must not import the GUI, so it
    cannot inherit — while a test double can subclass it explicitly.
    """

    def status(self) -> tuple[WorkerSummary, ...]:
        """Snapshot of every registered worker."""
        ...

    def restart_scope(self, scope: RestartScope, settings_reason: str = "") -> int:
        """Restart every worker whose settings scope changed. Returns how many."""
        ...


_ACTIVE_CONTROL: WorkerControl | None = None
_CONTROL_LOCK: Final = threading.Lock()


def set_active_worker_control(control: WorkerControl | None) -> None:
    """Register (or clear) the supervisor the resource panel and restart buttons use."""
    global _ACTIVE_CONTROL
    with _CONTROL_LOCK:
        _ACTIVE_CONTROL = control


def active_worker_control() -> WorkerControl | None:
    """The supervisor registered with :func:`set_active_worker_control`, if any."""
    with _CONTROL_LOCK:
        return _ACTIVE_CONTROL
