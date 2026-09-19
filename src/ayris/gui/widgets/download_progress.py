"""A throttled download row: progress bar, speed, ETA and a cancel button.

Models are gigabytes, so :class:`~ayris.core.events.ModelDownloadProgress` arrives
several times a second even after the downloader's own throttle. Repainting a
progress bar on every event is wasted work the user cannot read, so this widget
coalesces updates to at most :data:`_MAX_UPDATES_PER_SEC` a second: an update
that lands inside the window is remembered, not drawn, and a single-shot timer
flushes the last one so the bar always finishes where the transfer did.

The widget is a pure display driven from the bus. It owns no download and no
thread — it emits :attr:`cancel_requested` and lets the coordinator that owns the
transfer act on it. That is what lets a download outlive the settings window: the
widget can be torn down and rebuilt while the bytes keep coming.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Final

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.models.downloader import human_size

__all__ = ["DownloadProgress", "human_eta"]

#: Ceiling on repaints per second. The task asks for «не чаще ~10 раз в секунду»;
#: 100 ms between applied updates is exactly that.
_MAX_UPDATES_PER_SEC: Final = 10
_MIN_INTERVAL_SEC: Final = 1.0 / _MAX_UPDATES_PER_SEC

#: Progress bar range for a determinate transfer. A wide integer range keeps the
#: bar smooth without floating-point rounding showing as a stuck percentage.
_SCALE: Final = 1000


def human_eta(seconds: float) -> str:
    """``95`` → ``осталось 1 мин 35 с``. Empty when the estimate is not yet known."""
    if seconds <= 0:
        return ""
    total = int(round(seconds))
    if total < 60:
        return f"осталось {total} с"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"осталось {minutes} мин {secs} с" if secs else f"осталось {minutes} мин"
    hours, minutes = divmod(minutes, 60)
    return f"осталось {hours} ч {minutes} мин" if minutes else f"осталось {hours} ч"


class DownloadProgress(QWidget):
    """Progress bar with a speed/ETA caption and a cancel button.

    Args:
        theme: For spacing and the themed bar.
        clock: Monotonic time source. Injected so a test can drive the throttle
            without sleeping; defaults to :func:`time.monotonic`.
    """

    cancel_requested = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        clock: Callable[[], float] = time.monotonic,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._clock = clock
        self.setProperty("transparent", True)
        # How many times the visible widgets were actually touched — the throttle
        # test reads this instead of watching for repaints.
        self.updates_applied = 0
        self._last_emit = float("-inf")
        self._pending: tuple[int, int, float, float] | None = None
        self._applied: tuple[int, int, float, float] | None = None

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self._bar = QProgressBar()
        self._bar.setRange(0, _SCALE)
        self._bar.setTextVisible(True)
        self._bar.setAccessibleName("Прогресс загрузки")
        self._cancel = QPushButton("Отмена")
        self._cancel.setAccessibleName("Отменить загрузку")
        self._cancel.clicked.connect(self.cancel_requested)
        top.addWidget(self._bar, 1)
        top.addWidget(self._cancel)
        self._layout.addLayout(top)

        self._caption = QLabel("")
        self._caption.setProperty("role", "muted")
        self._caption.setWordWrap(False)
        self._layout.addWidget(self._caption)

        # Trailing flush: the last update inside the throttle window is not drawn
        # by set_progress itself, so a lull would leave the bar one step short.
        self._flush_timer = QTimer(self)
        self._flush_timer.setSingleShot(True)
        self._flush_timer.setInterval(int(_MIN_INTERVAL_SEC * 1000))
        self._flush_timer.timeout.connect(self._flush)

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()

    # -- API ----------------------------------------------------------------

    def set_progress(self, downloaded: int, total: int, speed_bps: float, eta_s: float) -> None:
        """Record a progress sample, drawing it only if the throttle window is open."""
        self._pending = (downloaded, total, speed_bps, eta_s)
        now = self._clock()
        if now - self._last_emit >= _MIN_INTERVAL_SEC:
            self._last_emit = now
            self._apply(self._pending)
        elif not self._flush_timer.isActive():
            self._flush_timer.start()

    def flush(self) -> None:
        """Draw the most recent sample immediately — used when a transfer ends."""
        self._flush_timer.stop()
        if self._pending is not None:
            self._last_emit = self._clock()
            self._apply(self._pending)

    def reset(self) -> None:
        """Clear back to an idle bar, forgetting any pending sample."""
        self._flush_timer.stop()
        self._pending = None
        self._applied = None
        self._bar.setRange(0, _SCALE)
        self._bar.setValue(0)
        self._caption.setText("")

    def set_cancellable(self, cancellable: bool) -> None:
        """Enable the cancel button; disabled once the transfer is past cancelling."""
        self._cancel.setEnabled(cancellable)

    # -- internals ----------------------------------------------------------

    def _flush(self) -> None:
        if self._pending is not None and self._pending != self._applied:
            self._last_emit = self._clock()
            self._apply(self._pending)

    def _apply(self, sample: tuple[int, int, float, float]) -> None:
        downloaded, total, speed_bps, eta_s = sample
        self._applied = sample
        self.updates_applied += 1
        if total > 0:
            self._bar.setRange(0, _SCALE)
            self._bar.setValue(min(_SCALE, round(_SCALE * downloaded / total)))
            self._bar.setFormat("%p%")
            done = f"{human_size(downloaded)} из {human_size(total)}"
        else:
            # No Content-Length: an indeterminate bar reads more honestly than 0 %.
            self._bar.setRange(0, 0)
            self._bar.setFormat("")
            done = human_size(downloaded)
        parts = [done]
        if speed_bps > 0:
            parts.append(f"{human_size(speed_bps)}/с")
        eta = human_eta(eta_s)
        if eta:
            parts.append(eta)
        self._caption.setText("   ·   ".join(parts))

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        self._layout.setSpacing(self._theme.metric("spacing_xs"))
