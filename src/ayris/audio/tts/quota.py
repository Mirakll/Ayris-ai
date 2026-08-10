"""Counting what the cloud voices cost.

Every provider here bills by the character and every one of them cuts the
account off when the month's allowance runs out - silently, in the middle of a
sentence, with a 429 that says nothing about how close it was. The point of this
module is that the user finds out before that happens rather than by noticing
Ayris has gone quiet.

**What is stored is a number, not a transcript.** One row per provider per
month, holding a character count and a request count. A row per phrase would be
more useful for debugging and would also be a permanent record of everything the
assistant has ever said aloud, which ``privacy`` says is the user's and does not
belong on disk. The text never reaches this module: the engines call
:meth:`QuotaTracker.record` with ``len(text)``.

**Writes are batched.** A sentence-by-sentence stream would otherwise mean a
``COMMIT`` per sentence on the synthesis thread, and SQLite's fsync is the one
thing in this path that can take longer than the speech it is accounting for.
Counts accumulate in memory and are flushed on a size or time threshold, on
:meth:`usage`, and on :meth:`close`. A crash therefore loses at most a few
seconds of counting, which is the right trade for a number that exists to warn
rather than to bill.

**The warning fires once per period.** Crossing the configured share publishes
one :class:`~ayris.core.events.NotificationRequested`, and the next one only
comes after the limit changes or the month rolls over. A warning that repeated
on every phrase would be the thing the user silences, and then it would not be
there when it mattered.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from ayris.core.models import to_db_timestamp, utc_now
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.database import Database
    from ayris.core.events import EventBus

__all__ = [
    "FLUSH_AFTER_CHARS",
    "FLUSH_AFTER_SEC",
    "WARN_RATIO",
    "QuotaTracker",
    "UsageRow",
    "current_period",
]

_log = get_logger(__name__)

#: Characters buffered before a flush. About a page of text; at conversational
#: volume this is a write every few minutes.
FLUSH_AFTER_CHARS: Final = 2000

#: Seconds a pending count may sit unwritten. Bounds the loss from a crash to
#: something the user would not notice in a monthly total.
FLUSH_AFTER_SEC: Final = 30.0

#: Share of the limit at which the user is warned. 0.8 leaves enough headroom to
#: switch to a local voice or top the account up before anything stops working.
WARN_RATIO: Final = 0.8


def current_period(now: datetime | None = None) -> str:
    """The billing month as ``YYYY-MM``.

    In UTC, matching every provider's own reset, and matching the timestamps
    already stored elsewhere in the database. A user in Kamchatka would
    otherwise see the counter reset a day early relative to the invoice.
    """
    moment = now or utc_now()
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime("%Y-%m")


@dataclass(frozen=True, slots=True)
class UsageRow:
    """What one provider spent in one month.

    Attributes:
        provider: Engine name, as in ``elevenlabs`` or ``yandex``.
        period: ``YYYY-MM``.
        characters: Characters sent for synthesis.
        requests: Requests that reached the provider.
    """

    provider: str
    period: str
    characters: int = 0
    requests: int = 0

    def share_of(self, limit: int) -> float:
        """How much of ``limit`` is spent, 0.0 when there is no limit set."""
        return self.characters / limit if limit > 0 else 0.0


class QuotaTracker:
    """Counts characters per provider and warns before the limit.

    Satisfies :class:`~ayris.audio.tts.cloud_base.UsageRecorder`, which is how
    the engines reach it without importing the database layer.

    Args:
        database: Where the counters live. The table is ``tts_usage``, created
            by schema migration 2.
        bus: Where the warning goes. Optional: the tracker is useful in a test
            or in a worker process with no bus, and simply logs there instead.
        limits: Characters per month per provider, from the settings. A provider
            with no entry is counted but never warned about, which is the right
            default - only the user knows what plan they are on.

    Thread-safe: the synthesis thread records while the settings window may ask
    for :meth:`usage`.
    """

    __slots__ = ("_bus", "_database", "_limits", "_lock", "_pending", "_since", "_warned")

    def __init__(
        self,
        database: Database,
        *,
        bus: EventBus | None = None,
        limits: dict[str, int] | None = None,
    ) -> None:
        self._database = database
        self._bus = bus
        self._limits = dict(limits or {})
        self._lock = threading.Lock()
        self._pending: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._since = utc_now()
        self._warned: set[tuple[str, str]] = set()

    # ------------------------------------------------------------- recording

    def record(self, provider: str, characters: int) -> None:
        """Note that ``characters`` were billed to ``provider``.

        Called on the synthesis path, so it does no I/O in the common case: the
        count lands in a dict and is written when the batch is big or old enough.
        """
        if not provider or characters <= 0:
            return
        with self._lock:
            entry = self._pending[provider]
            entry[0] += characters
            entry[1] += 1
            if self._should_flush_locked():
                self._flush_locked()

    def flush(self) -> None:
        """Write the pending counts now. Safe to call when there are none."""
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        """Flush and stop. Idempotent, and never raises."""
        try:
            self.flush()
        except Exception as exc:  # pragma: no cover - a failed flush is not fatal
            _log.debug("tts: не удалось записать расход при закрытии: %s", exc)

    # -------------------------------------------------------------- reading

    def usage(self, provider: str, period: str | None = None) -> UsageRow:
        """What one provider has spent this month, pending counts included."""
        self.flush()
        wanted = period or current_period()
        row = self._database.query_one(
            "SELECT characters, requests FROM tts_usage WHERE provider = ? AND period = ?",
            (provider, wanted),
        )
        if row is None:
            return UsageRow(provider=provider, period=wanted)
        return UsageRow(
            provider=provider,
            period=wanted,
            characters=int(row["characters"]),
            requests=int(row["requests"]),
        )

    def all_usage(self, period: str | None = None) -> tuple[UsageRow, ...]:
        """Every provider's spending in one month, for the settings window."""
        self.flush()
        wanted = period or current_period()
        rows = self._database.query_all(
            "SELECT provider, characters, requests FROM tts_usage "
            "WHERE period = ? ORDER BY characters DESC",
            (wanted,),
        )
        return tuple(
            UsageRow(
                provider=str(row["provider"]),
                period=wanted,
                characters=int(row["characters"]),
                requests=int(row["requests"]),
            )
            for row in rows
        )

    def set_limits(self, limits: dict[str, int]) -> None:
        """Replace the per-provider limits.

        Clears the "already warned" marks: a user who has just raised the limit
        should be warned again when they approach the new one, and one who
        lowered it below what is already spent should hear about it now.
        """
        with self._lock:
            self._limits = dict(limits)
            self._warned.clear()

    def remaining(self, provider: str) -> int:
        """Characters left this month, or ``-1`` when no limit is configured."""
        limit = self._limits.get(provider, 0)
        if limit <= 0:
            return -1
        return max(0, limit - self.usage(provider).characters)

    # -------------------------------------------------------------- internals

    def _should_flush_locked(self) -> bool:
        """Whether the batch is big enough or old enough to write."""
        total = sum(entry[0] for entry in self._pending.values())
        if total >= FLUSH_AFTER_CHARS:
            return True
        return (utc_now() - self._since).total_seconds() >= FLUSH_AFTER_SEC

    def _flush_locked(self) -> None:
        """Write and clear the batch. Caller holds the lock.

        An upsert per provider, in one transaction. A database that refuses the
        write is logged and the counts are dropped: the alternative is retrying
        forever with a growing dict, and a lost character count must never be
        the reason a phrase is not spoken.
        """
        if not self._pending:
            self._since = utc_now()
            return
        period = current_period()
        stamp = to_db_timestamp(utc_now())
        batch = [
            (provider, period, entry[0], entry[1], stamp)
            for provider, entry in self._pending.items()
        ]
        self._pending.clear()
        self._since = utc_now()
        try:
            with self._database.transaction():
                self._database.executemany(
                    "INSERT INTO tts_usage (provider, period, characters, requests, updated_at) "
                    "VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (provider, period) DO UPDATE SET "
                    "characters = characters + excluded.characters, "
                    "requests = requests + excluded.requests, "
                    "updated_at = excluded.updated_at",
                    batch,
                )
        except Exception as exc:
            _log.warning("tts: не удалось записать расход облачного синтеза: %s", exc)
            return
        for provider, _, _, _, _ in batch:
            self._check_limit_locked(provider, period)

    def _check_limit_locked(self, provider: str, period: str) -> None:
        """Warn once when a provider crosses :data:`WARN_RATIO` of its limit."""
        limit = self._limits.get(provider, 0)
        if limit <= 0 or (provider, period) in self._warned:
            return
        row = self._database.query_one(
            "SELECT characters FROM tts_usage WHERE provider = ? AND period = ?",
            (provider, period),
        )
        if row is None:
            return
        used = int(row["characters"])
        if used < limit * WARN_RATIO:
            return
        self._warned.add((provider, period))
        percent = int(used * 100 / limit)
        _log.warning("tts: %s израсходовал %d%% месячного лимита", provider, percent)
        self._notify(provider, percent, used, limit)

    def _notify(self, provider: str, percent: int, used: int, limit: int) -> None:
        """Tell the user, if there is a bus to tell them through."""
        bus = self._bus
        if bus is None:
            return
        from ayris.core.events import NotificationRequested

        exhausted = used >= limit
        message = (
            f"Израсходовано {used:,} из {limit:,} символов ({percent}%). "
            f"{'Синтез переключится на локальный голос.' if exhausted else 'Осталось немного.'}"
        ).replace(",", " ")
        try:
            bus.publish(
                NotificationRequested(
                    title=f"Лимит синтеза речи: {provider}",
                    message=message,
                    level="error" if exhausted else "warning",
                    timeout_ms=8000,
                )
            )
        except Exception as exc:  # pragma: no cover - the bus swallows its own
            _log.debug("tts: не удалось показать предупреждение о лимите: %s", exc)
