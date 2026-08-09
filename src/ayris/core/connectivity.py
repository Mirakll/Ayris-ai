"""Is the network there? Answered from evidence, not from polling.

A voice assistant needs this to decide between a cloud recogniser and the local
model, and the decision has to be made in well under the two seconds section 3.3
allows for a switch. That rules out asking the network at the moment of the
question - a DNS lookup on a dead Wi-Fi can take longer than the whole budget.

So the answer is kept as state and updated from three signals, in order of how
much they cost:

* **A real request failed.** :meth:`ConnectivityMonitor.report_failure` is called
  by the cloud engine that just timed out. Free, immediate, and the only signal
  that matters for latency: the fallback happens on the failure that provoked it.
* **The adapter has no link.** On Windows :func:`link_up` asks wininet whether
  there is a connection at all. Free and local - no packet leaves the machine -
  so it gates the probe below and catches an unplugged cable straight away.
* **A background probe.** One HEAD request to the single address in
  ``voice.stt.probe_url``, and only while offline: while online there is nothing
  to find out, because the next real request will say. This is what brings the
  router back after the network returns, and it is the reason the interval is a
  setting rather than a constant.

**Zero telemetry, and it is checkable.** The probe goes to exactly one URL, the
one in the config; an empty ``probe_url`` disables it entirely; and the monitor
does nothing at all until :meth:`start` is called, which the composition root
only does in online or auto mode. Nothing here reports anything about the user -
the request carries no body, no cookie and no identifier beyond the user agent.

**Flapping is filtered, but only in the optimistic direction.** Going offline
takes one failure, because a user waiting on a transcript should not wait through
a confirmation round trip. Coming back online takes
:data:`RECOVERY_CONFIRMATIONS` consecutive good probes, so a captive portal that
answers every other request does not make the router oscillate.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Final

import httpx

from ayris.core.events import OnlineStatusChanged
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.events import EventBus

__all__ = [
    "PROBE_TIMEOUT_SEC",
    "RECOVERY_CONFIRMATIONS",
    "ConnectivityMonitor",
    "link_up",
]

_log = get_logger(__name__)

#: The probe is a liveness check, not a request whose answer is needed: anything
#: slower than this is indistinguishable from down for our purposes.
PROBE_TIMEOUT_SEC: Final = 4.0

#: Consecutive successful probes before declaring the network back. Two, not one:
#: a hotel portal that answers the first request and then redirects would
#: otherwise flip the router back and forth.
RECOVERY_CONFIRMATIONS: Final = 2

#: Backstop between probes if the configured interval is nonsense.
_MIN_INTERVAL_SEC: Final = 15.0

#: wininet flag set when any connection is up. Only the fact is read; the bit
#: mask distinguishing modem from LAN is not something Ayris should care about.
_INTERNET_CONNECTION_MASK: Final = 0


def link_up() -> bool:
    """Whether the OS thinks there is a network connection at all.

    A local question with a local answer - no traffic - so it is safe to ask
    often and before every probe. ``True`` on any platform that cannot be asked,
    which makes this a filter that never blocks the real signals: a wrong "yes"
    only costs one probe, a wrong "no" would strand the router offline.
    """
    if sys.platform != "win32":  # pragma: no cover - checked on the CI runner
        return True
    try:  # pragma: no cover - exercised on Windows only
        import ctypes

        flags = ctypes.c_ulong(_INTERNET_CONNECTION_MASK)
        return bool(ctypes.windll.wininet.InternetGetConnectedState(ctypes.byref(flags), 0))
    except (OSError, AttributeError, ValueError):  # pragma: no cover - defensive
        # No wininet, no answer. Assume there is a link and let the probe decide.
        return True


class ConnectivityMonitor:
    """Tracks cloud reachability and publishes :class:`OnlineStatusChanged`.

    Args:
        bus: Where the status change is published. The STT router subscribes.
        probe_url: The one address the background probe may contact. Empty
            disables the probe, leaving only the request-failure signal.
        interval_sec: Seconds between probes while offline.
        transport: Injected for tests, so the suite never touches the network.

    Starts optimistic: until something fails there is no reason to assume a
    machine is offline, and the cost of being wrong is one failed request that
    immediately corrects the state.
    """

    __slots__ = (
        "_bus",
        "_client",
        "_interval",
        "_lock",
        "_online",
        "_probe_url",
        "_stop",
        "_streak",
        "_thread",
        "_transport",
    )

    def __init__(
        self,
        bus: EventBus,
        *,
        probe_url: str = "",
        interval_sec: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._bus = bus
        self._probe_url = probe_url.strip()
        self._interval = max(_MIN_INTERVAL_SEC, float(interval_sec))
        self._transport = transport
        self._lock = threading.Lock()
        self._online = True
        self._streak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._client: httpx.Client | None = None

    @property
    def online(self) -> bool:
        """Last known reachability. Read without blocking, from any thread."""
        with self._lock:
            return self._online

    @property
    def probing(self) -> bool:
        """Whether the background probe thread is running."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self) -> None:
        """Begin background probing, if there is an address to probe.

        Idempotent. Does nothing without a ``probe_url``, which is how offline
        mode keeps the network completely quiet.
        """
        if not self._probe_url:
            _log.debug("connectivity: no probe url, running on request failures only")
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="ayris-connectivity", daemon=True
            )
            self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        """Stop probing and close the client. Safe to call twice, never raises."""
        self._stop.set()
        with self._lock:
            thread, self._thread = self._thread, None
            client, self._client = self._client, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        if client is not None:
            try:
                client.close()
            except Exception as exc:  # pragma: no cover - close is best effort
                _log.debug("connectivity: closing probe client failed: %s", exc)

    def report_failure(self, reason: str = "") -> None:
        """A real request just failed with a network error.

        The signal the fallback is built on: called from the cloud engine's
        caller, it flips the state before the next attempt is chosen, and starts
        the probe that will notice the network coming back.
        """
        detail = reason or "сетевая ошибка запроса"
        if self._set_online(value=False, detail=detail):
            _log.info("connectivity: offline (%s)", detail)
        self.start()

    def report_success(self) -> None:
        """A real request just succeeded, which is proof enough to go online."""
        if self._set_online(value=True, detail="запрос прошёл"):
            _log.info("connectivity: online again")

    def check_now(self) -> bool:
        """Probe once, synchronously, and return the resulting state.

        Used by the settings window's "check connection" button and by the tests.
        Without a ``probe_url`` this reports the current state unchanged rather
        than inventing an answer.
        """
        if not self._probe_url:
            return self.online
        reachable = self._probe()
        self._apply_probe(reachable=reachable)
        return self.online

    def _loop(self) -> None:
        """Probe while offline, sleep otherwise, until :meth:`stop`."""
        while not self._stop.wait(self._interval):
            if self.online:
                # Nothing to learn: the next real request is a better signal and
                # costs nothing extra.
                continue
            if not link_up():
                self._apply_probe(reachable=False)
                continue
            self._apply_probe(reachable=self._probe())

    def _probe(self) -> bool:
        """One HEAD request to the configured address."""
        client = self._ensure_client()
        try:
            response = client.head(self._probe_url)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            _log.debug("connectivity: probe failed: %s", type(exc).__name__)
            return False
        # Any answer at all proves the path works. A 404 from a misconfigured
        # probe address still means packets got there and came back.
        return response.status_code < 500

    def _ensure_client(self) -> httpx.Client:
        """The probe client, created on first use and reused after that."""
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(PROBE_TIMEOUT_SEC),
                    follow_redirects=False,
                    headers={"User-Agent": "Ayris"},
                    transport=self._transport,
                )
            return self._client

    def _apply_probe(self, *, reachable: bool) -> None:
        """Fold one probe result into the state, applying the recovery streak."""
        if not reachable:
            with self._lock:
                self._streak = 0
            return
        with self._lock:
            self._streak += 1
            confirmed = self._streak >= RECOVERY_CONFIRMATIONS
        if confirmed and self._set_online(value=True, detail="связь восстановлена"):
            _log.info("connectivity: online again after %d good probes", RECOVERY_CONFIRMATIONS)

    def _set_online(self, *, value: bool, detail: str) -> bool:
        """Store the state and publish if it changed. Returns whether it did."""
        with self._lock:
            if self._online == value:
                if value:
                    self._streak = 0
                return False
            self._online = value
            self._streak = 0
        self._bus.publish(OnlineStatusChanged(online=value, detail=detail))
        return True

    def __enter__(self) -> ConnectivityMonitor:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
