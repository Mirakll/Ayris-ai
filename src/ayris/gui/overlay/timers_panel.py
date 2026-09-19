"""Active timers and reminders in the overlay, with a live countdown and cancel.

The panel does not read the database. It asks a :class:`TimerProvider` — the
interface task 28 exposes — for the current active records and refreshes once a
second, but only while it is visible: a hidden panel stops its timer. Countdown
is computed from each record's absolute due time, so it does not drift and it
survives the machine sleeping.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

from PySide6.QtCore import QTimer
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ayris.core.models import utc_now
from ayris.gui.theme import ThemeManager

__all__ = ["ActiveTimer", "TimerProvider", "TimersPanel"]

_TICK_MS: Final = 1000


@dataclass(frozen=True, slots=True)
class ActiveTimer:
    """One active timer/reminder/alarm, as the overlay needs to show it."""

    id: int
    label: str
    due: datetime
    kind: str = "timer"

    def remaining_seconds(self, now: datetime) -> int:
        return max(0, round((self.due - now).total_seconds()))


class TimerProvider(Protocol):
    """What the overlay needs from the timer subsystem (task 28)."""

    def active_timers(self) -> Sequence[ActiveTimer]: ...

    def cancel_timer(self, timer_id: int) -> None: ...


def format_remaining(seconds: int) -> str:
    """``90`` → ``"01:30"``; hours appear only when there are any."""
    if seconds >= 3600:
        hours, rest = divmod(seconds, 3600)
        minutes, secs = divmod(rest, 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"
    minutes, secs = divmod(seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


class _TimerRow(QWidget):
    def __init__(
        self,
        timer: ActiveTimer,
        *,
        on_cancel: Callable[[int], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._id = timer.id
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(timer.label or "Таймер")
        self.remaining = QLabel("")
        self.remaining.setAccessibleName("Осталось времени")
        cancel = QPushButton("Отменить")
        cancel.setAccessibleName(f"Отменить: {timer.label or 'таймер'}")
        cancel.clicked.connect(lambda _checked: on_cancel(self._id))
        layout.addWidget(self.label, 1)
        layout.addWidget(self.remaining)
        layout.addWidget(cancel)

    def update_remaining(self, seconds: int) -> None:
        self.remaining.setText(format_remaining(seconds))


class TimersPanel(QWidget):
    """List of active timers that updates once a second while it is visible."""

    def __init__(
        self,
        theme: ThemeManager,
        *,
        provider: TimerProvider | None = None,
        clock: Callable[[], datetime] = utc_now,
        show_empty: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._provider = provider
        self._clock = clock
        self._show_empty = show_empty
        self._rows: dict[int, _TimerRow] = {}
        self._timers: tuple[ActiveTimer, ...] = ()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._empty = QLabel("Нет активных таймеров")
        self._empty.setAccessibleName("Нет активных таймеров")
        self._empty.setVisible(self._show_empty)
        self._layout.addWidget(self._empty)
        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self.refresh)

    @property
    def timers(self) -> tuple[ActiveTimer, ...]:
        return self._timers

    def is_updating(self) -> bool:
        return self._tick.isActive()

    def set_provider(self, provider: TimerProvider | None) -> None:
        self._provider = provider
        self.refresh()

    def refresh(self) -> None:
        provider = self._provider
        active = tuple(provider.active_timers()) if provider is not None else ()
        self._timers = active
        self._reconcile(active)
        now = self._clock()
        for timer in active:
            row = self._rows.get(timer.id)
            if row is not None:
                row.update_remaining(timer.remaining_seconds(now))
        self._empty.setVisible(self._show_empty and not active)

    def start_updates(self) -> None:
        if not self._tick.isActive():
            self._tick.start()
        self.refresh()

    def stop_updates(self) -> None:
        self._tick.stop()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        self.start_updates()

    def hideEvent(self, event: QHideEvent) -> None:  # noqa: N802
        self.stop_updates()
        super().hideEvent(event)

    def _reconcile(self, active: Sequence[ActiveTimer]) -> None:
        wanted = {timer.id for timer in active}
        for timer_id in list(self._rows):
            if timer_id not in wanted:
                row = self._rows.pop(timer_id)
                self._layout.removeWidget(row)
                row.deleteLater()
        for timer in active:
            if timer.id not in self._rows:
                row = _TimerRow(timer, on_cancel=self._cancel, parent=self)
                self._rows[timer.id] = row
                self._layout.addWidget(row)

    def _cancel(self, timer_id: int) -> None:
        provider = self._provider
        if provider is not None:
            provider.cancel_timer(timer_id)
        self.refresh()
