"""Ties the player to the event bus: speech events out, «Айрис, стоп» in.

Kept out of :mod:`ayris.audio.tts.player` on purpose. The player is a device
driver - a queue, a thread and a stream - and it is tested against a fake
backend with no bus in sight. The bus, meanwhile, lives in the main process and
knows nothing about audio. This module is the one place where the two meet, which
is also the only place that has to be reasoned about when the overlay's
animation looks out of step.

Two directions, and they are not symmetrical:

*Out.* :class:`~ayris.core.events.TtsStarted` is published when sound actually
reaches the device, not when synthesis was requested - the overlay's mouth has to
move with what the user hears. :class:`~ayris.core.events.TtsFinished` carries
why speaking stopped, because a completed phrase fades out where an interrupted
one snaps back to idle.

*In.* :class:`~ayris.core.events.CancelRequested` stops playback. Handled here
rather than in the player so that the player keeps working in a test with no bus,
and so that the cancel reaches the worker too: dropping the queue without
cancelling synthesis would leave the next sentence to arrive a moment later and
start speaking again.

The handler is registered with ``weak=False``. The bus holds subscribers weakly
by default, and the closure over the player would otherwise be collected
immediately, leaving a cancel that silently does nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayris.core.events import (
    TTS_REASON_CANCELLED,
    TTS_REASON_COMPLETED,
    TTS_REASON_ERROR,
    CancelRequested,
    TtsFinished,
    TtsStarted,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayris.audio.tts.player import SpeechRequest, TtsPlayer
    from ayris.core.events import EventBus, Unsubscribe

__all__ = ["TtsBridge", "connect_player"]

_log = get_logger(__name__)

#: Maps a :class:`~ayris.audio.tts.player.PlaybackReason` onto the bus vocabulary.
#: They agree today; the table is here so that adding a reason on one side fails
#: loudly rather than publishing an unknown string to the overlay.
_REASONS = {
    "completed": TTS_REASON_COMPLETED,
    "cancelled": TTS_REASON_CANCELLED,
    "error": TTS_REASON_ERROR,
}


class TtsBridge:
    """Publishes what the player does and cancels it when the bus says so.

    Args:
        bus: Where the events go.
        player: The player to observe and to stop.
        on_cancel: Called when a cancel arrives, before the player is stopped.
            The pipeline passes the worker's ``cancel`` here, so that the
            sentence being synthesized is dropped as well as the ones queued.

    Hold on to the instance: it owns the bus subscription, and letting it be
    collected turns «Айрис, стоп» into a no-op. :meth:`close` detaches it.
    """

    __slots__ = ("_bus", "_on_cancel", "_player", "_unsubscribe")

    def __init__(
        self,
        bus: EventBus,
        player: TtsPlayer,
        *,
        on_cancel: Callable[[], None] | None = None,
    ) -> None:
        self._bus = bus
        self._player = player
        self._on_cancel = on_cancel
        self._unsubscribe: Unsubscribe | None = bus.subscribe(
            CancelRequested, self._on_cancel_requested, weak=False
        )

    def started(self, request: SpeechRequest, duration_ms: int) -> None:
        """Player callback: a phrase began sounding."""
        self._bus.publish(
            TtsStarted(
                text=request.text,
                engine=request.engine,
                duration_estimate_ms=max(0, duration_ms),
                request_id=request.request_id,
            )
        )

    def finished(self, request: SpeechRequest, reason: str) -> None:
        """Player callback: a phrase stopped, for whatever reason."""
        self._bus.publish(
            TtsFinished(
                reason=_REASONS.get(reason, TTS_REASON_ERROR),
                text=request.text,
                request_id=request.request_id,
            )
        )

    def close(self) -> None:
        """Stop listening for cancels. Idempotent."""
        unsubscribe = self._unsubscribe
        self._unsubscribe = None
        if unsubscribe is not None:
            unsubscribe()

    def _on_cancel_requested(self, event: CancelRequested) -> None:
        """Silence the assistant now.

        The worker is told first: it is the slower of the two, and a sentence it
        finishes after the queue was cleared would be played by the next request.
        The player's own stop is immediate - it aborts the device rather than
        letting the buffered tail play out.
        """
        if self._on_cancel is not None:
            try:
                self._on_cancel()
            except Exception:
                # A worker that is down cannot be asked to stop, and that is no
                # reason to leave the speakers running.
                _log.exception("tts: не удалось отменить синтез в воркере")
        stopped = self._player.cancel()
        if stopped:
            _log.info("tts: озвучка прервана (%s)", event.reason or "без причины")


def connect_player(
    bus: EventBus,
    player: TtsPlayer,
    *,
    on_cancel: Callable[[], None] | None = None,
) -> TtsBridge:
    """Wire a player to the bus in both directions.

    Returns:
        The bridge, which the caller must keep alive - see :class:`TtsBridge`.
        Its ``started``/``finished`` methods are installed as the player's
        callbacks, replacing whatever was there.
    """
    bridge = TtsBridge(bus, player, on_cancel=on_cancel)
    player.set_observers(on_started=bridge.started, on_finished=bridge.finished)
    return bridge
