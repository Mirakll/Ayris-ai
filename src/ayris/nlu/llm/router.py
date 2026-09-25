"""One entry point for the language model, three modes behind it.

This is the LLM twin of :class:`~ayris.audio.stt.router.SttRouter`: the rest of
Ayris asks it to :meth:`complete` or :meth:`stream` an answer and never learns
which engine replied. What it adds over a bare client is the decision, and the
decision is mostly about failure.

* **offline** — the local engine only (Ollama or llama.cpp). Nothing here reaches
  the network, which is what a user who chose this mode is asking for.
* **online** — the cloud only. A failure is reported as a failure: quietly
  answering from a local model would be a different privacy answer than the one
  the user gave.
* **auto** — cloud first, local the moment the cloud does not work, back to cloud
  when it recovers.

**Fallback is only clean before the first token.** ``complete`` is atomic, so a
cloud failure simply re-runs on the local engine. ``stream`` is trickier: once a
delta has been spoken, replaying on another engine would say the sentence twice,
so the switch happens only when the cloud fails *before* the first delta; a
mid-stream break propagates as a plain error, exactly as the client already
guarantees. Either way the user's request itself is never dropped.

**Coming back is automatic and quiet.** After a fallback the router stays local
until :class:`~ayris.core.connectivity.ConnectivityMonitor` publishes
:class:`~ayris.core.events.OnlineStatusChanged` with ``online=True``; then the
next request goes to the cloud again. Notifications are gated so a flaky link does
not narrate every switch, matching the STT router's hysteresis.

**Economy mode keeps local memory idle.** With a cloud provider active and
``eco_mode`` on, the router does not warm the local engine and drops it again the
moment the cloud is back, so a cloud-first session is not holding a model's worth
of RAM it is not using.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.core.errors import (
    AyrisError,
    LlmAuthError,
    LlmContextOverflowError,
    LlmError,
    LlmQuotaError,
)
from ayris.core.events import NotificationRequested, OnlineStatusChanged
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from ayris.core.connectivity import ConnectivityMonitor
    from ayris.core.events import EventBus, Unsubscribe
    from ayris.nlu.llm.base import LlmClient, LlmDelta, LlmMessage, LlmResponse, LlmTool

    #: Hands back a ready client, building it on the first call. The router never
    #: constructs a client itself: a cloud client needs a key and a local one a
    #: host or a model path, and the factory already knows how to resolve each.
    ClientProvider = Callable[[], LlmClient]

__all__ = ["LlmMode", "LlmRouter"]

_log = get_logger(__name__)

#: Every LLM failure is a reason to try the local engine. The subclasses (a spent
#: quota, a wrong key, a context overflow) are permanent and the user must fix
#: them, but in the meantime the assistant should keep answering rather than
#: repeat the complaint on every turn.
_FALLBACK_ERRORS: Final = (LlmError,)


def _is_connectivity(exc: LlmError) -> bool:
    """Whether a failure looks like «the network is down» rather than a policy no.

    A spent quota, a rejected key and a context overflow are real answers from a
    reachable service, so they must not be reported to the connectivity monitor as
    an outage — only a plain :class:`LlmError` (the shape a transport failure
    becomes) counts as «offline».
    """
    return not isinstance(exc, LlmAuthError | LlmContextOverflowError | LlmQuotaError)


class LlmMode(StrEnum):
    """Which engine answers, mirroring the STT router's three modes."""

    OFFLINE = "offline"
    ONLINE = "online"
    AUTO = "auto"

    @classmethod
    def parse(cls, value: str) -> LlmMode:
        """Read a mode out of the settings, defaulting to :attr:`AUTO`.

        An unknown string becomes auto rather than an error: the value may come
        from a config a newer Ayris wrote, and refusing to answer at all is worse
        than picking the mode that copes with both cloud and local.
        """
        try:
            return cls(value.strip().lower())
        except ValueError:
            _log.warning("llm router: unknown mode %r, using auto", value)
            return cls.AUTO


class LlmRouter:
    """Routes one request to the cloud or to the local engine.

    Args:
        mode: Which of the three modes to run.
        online: Provider for the cloud client. ``None`` means no cloud client is
            configured, which makes online and auto behave as offline.
        offline: Provider for the local client (Ollama or llama.cpp). ``None``
            means no local engine, so auto has nothing to fall back to.
        monitor: Connectivity state. The router reports connectivity failures to
            it and listens for its recovery event; without one, auto falls back on
            failure and returns to the cloud on the next request.
        bus: Where notifications and the recovery subscription live.
        eco_mode: When set, auto mode does not warm the local engine and drops it
            again the moment the cloud is back, so a cloud-first session is not
            holding a model's worth of RAM it is not using (§12).
    """

    __slots__ = (
        # Explicit, because __slots__ otherwise removes __weakref__ and the bus
        # holds a bound-method subscription weakly: without this, subscribing
        # _on_online_status raises and the router cannot be constructed with a bus.
        "__weakref__",
        "_bus",
        "_eco_mode",
        "_lock",
        "_mode",
        "_monitor",
        "_offline",
        "_offline_provider",
        "_online",
        "_online_provider",
        "_preload",
        "_quota_notified",
        "_unsubscribe",
        "_use_offline",
    )

    def __init__(
        self,
        *,
        mode: LlmMode = LlmMode.AUTO,
        online: ClientProvider | None = None,
        offline: ClientProvider | None = None,
        monitor: ConnectivityMonitor | None = None,
        bus: EventBus | None = None,
        eco_mode: bool = False,
    ) -> None:
        self._mode = mode
        self._online_provider = online
        self._offline_provider = offline
        self._monitor = monitor
        self._bus = bus
        self._eco_mode = eco_mode
        self._lock = threading.RLock()
        self._online: LlmClient | None = None
        self._offline: LlmClient | None = None
        self._use_offline = False
        self._quota_notified = False
        self._preload: threading.Thread | None = None
        self._unsubscribe: Unsubscribe | None = None
        if bus is not None:
            self._unsubscribe = bus.subscribe(OnlineStatusChanged, self._on_online_status)

    @property
    def mode(self) -> LlmMode:
        """The configured mode."""
        return self._mode

    @property
    def using_offline(self) -> bool:
        """Whether auto mode has fallen back to the local engine and not returned."""
        with self._lock:
            return self._use_offline

    def preload(self) -> None:
        """Warm the local engine in the background so a fallback is quick.

        Only useful in auto mode: offline mode loads on its first request anyway,
        and online mode should not warm an engine it will never use. Economy mode
        skips it too — the whole point there is to not hold local memory while the
        cloud is answering. Idempotent, and failures are logged rather than raised.
        """
        if self._mode is not LlmMode.AUTO or self._offline_provider is None or self._eco_mode:
            return
        with self._lock:
            if self._offline is not None:
                return
            if self._preload is not None and self._preload.is_alive():
                return
            self._preload = threading.Thread(
                target=self._preload_offline, name="ayris-llm-preload", daemon=True
            )
            self._preload.start()

    def close(self) -> None:
        """Drop the subscription and release any clients the router itself built.

        A cloud client holds a pooled socket and a llama.cpp client holds a model
        worth gigabytes, so — unlike the STT router, whose engines are cheap — the
        clients this router constructed through its providers are closed here.
        """
        unsubscribe, self._unsubscribe = self._unsubscribe, None
        if unsubscribe is not None:
            unsubscribe()
        with self._lock:
            online, self._online = self._online, None
            offline, self._offline = self._offline, None
        for client in (online, offline):
            if client is not None:
                client.close()

    # ------------------------------------------------------------------
    # the two answer shapes, each routed through the active mode
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        """Answer ``messages`` through whichever engine the mode allows.

        Atomic, so a cloud failure in auto mode simply re-runs the whole request
        on the local engine — nothing has been shown to the user yet.

        Raises:
            LlmError: No engine could answer. The ``user_message`` names the
                reason — no key, no local model, or a cloud outage with nothing
                installed to fall back to.
        """
        if self._mode is LlmMode.OFFLINE:
            return self._offline_client().complete(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
        if self._mode is LlmMode.ONLINE:
            return self._online_client().complete(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
        return self._complete_auto(
            messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
        )

    def _complete_auto(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        cancel: Callable[[], bool] | None,
    ) -> LlmResponse:
        """Cloud first unless the network is known to be down."""
        if self._online_provider is None or self._prefer_offline():
            return self._offline_client().complete(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
        try:
            result = self._online_client().complete(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
        except _FALLBACK_ERRORS as exc:
            return self._begin_fallback(exc).complete(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
        self._note_success()
        return result

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        """Answer ``messages`` in fragments, so TTS can start on the first sentence.

        Fallback is only clean before the first token: once a delta has been
        yielded, a cloud break in auto mode propagates as a plain :class:`LlmError`
        rather than replay the answer on the local engine and say it twice.

        Raises:
            LlmError: Same failures as :meth:`complete`, plus a mid-stream break
                after the first delta.
        """
        if self._mode is LlmMode.OFFLINE:
            yield from self._offline_client().stream(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
            return
        if self._mode is LlmMode.ONLINE:
            yield from self._online_client().stream(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
            return
        yield from self._stream_auto(
            messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
        )

    def _stream_auto(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        cancel: Callable[[], bool] | None,
    ) -> Iterator[LlmDelta]:
        """Stream from the cloud, switching to local only before the first delta."""
        if self._online_provider is None or self._prefer_offline():
            yield from self._offline_client().stream(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
            return
        emitted = False
        try:
            for delta in self._online_client().stream(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            ):
                emitted = True
                yield delta
        except _FALLBACK_ERRORS as exc:
            if emitted:
                # A token is already on its way to the user; replaying on another
                # engine would repeat it, so the break stays a break.
                raise
            yield from self._begin_fallback(exc).stream(
                messages, tools, temperature=temperature, max_tokens=max_tokens, cancel=cancel
            )
            return
        self._note_success()

    def _begin_fallback(self, exc: LlmError) -> LlmClient:
        """Record the switch to local, notify once, and hand back the local client.

        Shared by :meth:`_complete_auto` and :meth:`_stream_auto`. Only a
        connectivity failure is reported to the monitor — a spent quota or a
        rejected key is a real answer from a reachable service, not an outage.

        Raises:
            LlmError: There is no local engine, so the fallback has nowhere to go.
                Re-raised from the cloud failure with both halves in the message.
        """
        quota = isinstance(exc, LlmQuotaError)
        with self._lock:
            first = not self._use_offline
            self._use_offline = True
        if _is_connectivity(exc) and self._monitor is not None:
            self._monitor.report_failure(type(exc).__name__)
        _log.warning("llm router: cloud failed (%s), falling back to local", exc.technical)

        if self._offline_provider is None:
            self._notify(
                "ИИ недоступен",
                f"{exc.user_message} Локальная модель не установлена, поэтому ответить нечем.",
                level="error",
            )
            raise LlmError(
                f"llm router: cloud failed and no local engine is configured: {exc.technical}",
                user_message=(
                    f"{exc.user_message} Локальной модели нет — установите модель в разделе "
                    f"«ИИ», чтобы помощник работал без интернета."
                ),
            ) from exc

        if first or (quota and not self._quota_notified):
            self._quota_notified = quota
            self._notify(
                "ИИ переключён",
                (
                    f"{exc.user_message} Перешёл на локальную модель — "
                    f"вернусь в облако, когда связь восстановится."
                ),
                level="warning",
            )
        return self._offline_client()

    # ------------------------------------------------------------------
    # building and picking clients
    # ------------------------------------------------------------------

    def _online_client(self) -> LlmClient:
        """The cloud client, built on first use and then reused.

        Raises:
            LlmError: No cloud provider is configured. A build failure inside the
                provider surfaces as whatever it raised — usually an ``LlmError``.
        """
        with self._lock:
            client = self._online
            if client is not None:
                return client
            if self._online_provider is None:
                raise LlmError(
                    "llm router: online mode without a cloud client",
                    user_message=(
                        "Облачный ИИ выбран, но провайдер не настроен. "
                        "Выберите сервис и сохраните ключ в настройках, в разделе «ИИ»."
                    ),
                )
            client = self._online_provider()
            self._online = client
            return client

    def _offline_client(self) -> LlmClient:
        """The local client, built on first use and then reused.

        Raises:
            LlmError: No local engine is configured.
        """
        with self._lock:
            client = self._offline
            if client is not None:
                return client
            if self._offline_provider is None:
                raise LlmError(
                    "llm router: no local engine is configured",
                    user_message=(
                        "Локальная модель не настроена. Установите Ollama или выберите "
                        "файл модели для llama.cpp в разделе «ИИ»."
                    ),
                )
            client = self._offline_provider()
            self._offline = client
            return client

    def _prefer_offline(self) -> bool:
        """Whether auto mode should skip the cloud for this request."""
        with self._lock:
            if self._use_offline:
                return True
        return self._monitor is not None and not self._monitor.online

    def _note_success(self) -> None:
        """A cloud answer arrived: the network is up, whatever we thought."""
        self._quota_notified = False
        if self._monitor is not None:
            self._monitor.report_success()

    def _preload_offline(self) -> None:
        """Thread body for :meth:`preload`. Never raises out of the thread."""
        try:
            self._offline_client()
        except AyrisError as exc:
            _log.info("llm router: local preload skipped: %s", exc.technical)
        except Exception:  # pragma: no cover - a bug, not a usage error
            _log.exception("llm router: local preload failed")

    def _on_online_status(self, event: OnlineStatusChanged) -> None:
        """Return to the cloud when connectivity comes back.

        In economy mode the local client is dropped here as well, since the whole
        point of that mode is to not keep a model's memory around once the cloud
        can answer again.
        """
        if not event.online:
            return
        with self._lock:
            if not self._use_offline:
                return
            self._use_offline = False
            drop = self._offline if self._eco_mode else None
            if drop is not None:
                self._offline = None
        self._quota_notified = False
        if drop is not None:
            drop.close()
        _log.info("llm router: back online (%s), next request goes to the cloud", event.detail)
        self._notify(
            "Связь восстановлена",
            "ИИ снова отвечает через облако.",
            level="info",
        )

    def _notify(self, title: str, message: str, *, level: str) -> None:
        """Put a notification on the bus, if there is a bus."""
        if self._bus is None:
            return
        self._bus.publish(
            NotificationRequested(title=title, message=message, level=level, timeout_ms=6000)
        )
