"""One entry point for recognition, three modes behind it.

The rest of Ayris calls :meth:`SttRouter.transcribe` and never learns which
engine answered - that is in :attr:`TranscriptResult.engine`, which is where the
pipeline log reads it from. What the router adds on top of the engines is the
decision, and the decision is mostly about failure:

* **offline** - the local model only. Nothing here touches the network, which is
  what a user who picked this mode is asking for.
* **online** - the cloud only. A failure is reported as a failure: silently
  recognising locally would be a different privacy answer than the one the user
  gave.
* **auto** - cloud first, local model the moment the cloud does not work.

**The fallback has to be fast, so the local model is loaded before it is needed.**
Section 3.3 gives the switch under two seconds, and a cold Vosk load alone is
most of that; a cold Whisper load is ten times that. So in auto mode
:meth:`preload` warms the offline engine in a background thread at startup, and
the fallback path finds it ready. If it is not ready yet the fallback still
works - it just pays the load, which is better than refusing to transcribe.

**Coming back is automatic and cheap.** After a fallback the router stays local
until :class:`~ayris.core.connectivity.ConnectivityMonitor` publishes
:class:`~ayris.core.events.OnlineStatusChanged` with ``online=True``; then the
next phrase goes to the cloud again. No restart, and no retry storm in between,
because the monitor is the only thing probing.

**A missing offline model is said out loud.** Auto mode with no local model
installed and no network is a dead end, and the one thing it must not do is fail
silently: the user gets a notification and the caller gets an
:class:`~ayris.core.errors.SttError` whose ``user_message`` says what to install.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.audio.stt.cloud_base import NetworkError, QuotaError
from ayris.core.errors import AyrisError, SttError
from ayris.core.events import NotificationRequested, OnlineStatusChanged
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayris.audio.stt.base import AudioBuffer, SttEngine, TranscriptResult
    from ayris.core.connectivity import ConnectivityMonitor
    from ayris.core.events import EventBus, Unsubscribe

    #: Hands back a loaded engine, loading it if this is the first call. The
    #: router never constructs an engine itself: what a cloud engine needs (a
    #: credential) and what a local one needs (a model path and a RAM check) have
    #: nothing in common, and both already have a home.
    EngineProvider = Callable[[], SttEngine]

__all__ = ["SttMode", "SttRouter"]

_log = get_logger(__name__)

#: Errors that mean "the cloud did not work, try the local model". A wrong key
#: belongs here too: it is permanent, the user has to fix it, and in the meantime
#: the assistant should keep working rather than repeat the complaint per phrase.
_FALLBACK_ERRORS: Final = (NetworkError, QuotaError)


class SttMode(StrEnum):
    """Recognition mode, matching ``voice.stt.mode`` in the settings."""

    OFFLINE = "offline"
    ONLINE = "online"
    AUTO = "auto"

    @classmethod
    def parse(cls, value: str) -> SttMode:
        """Read a mode out of the settings, defaulting to :attr:`AUTO`.

        An unknown string becomes auto rather than an error: the mode arrives
        from a config file a newer Ayris may have written, and refusing to
        recognise anything would be a worse answer than picking the mode that
        works in both situations.
        """
        try:
            return cls(value.strip().lower())
        except ValueError:
            _log.warning("stt router: unknown mode %r, using auto", value)
            return cls.AUTO


class SttRouter:
    """Routes one transcription to the cloud or to the local model.

    Args:
        mode: Which of the three modes to run.
        online: Provider for the cloud engine. ``None`` means there is no cloud
            engine configured, which makes online and auto behave as offline.
        offline: Provider for the local engine. ``None`` means no local model, so
            auto has nothing to fall back to.
        monitor: Connectivity state. The router reports failures to it and
            listens for its recovery event; without one, auto mode falls back on
            failure and returns to the cloud on the next phrase.
        bus: Where notifications and the recovery subscription live.
    """

    __slots__ = (
        # Explicit, because __slots__ otherwise removes __weakref__ and the bus
        # holds a bound-method subscription weakly: without this, subscribing
        # _on_online_status raises and the router cannot be constructed with a bus.
        "__weakref__",
        "_bus",
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
        mode: SttMode = SttMode.AUTO,
        online: EngineProvider | None = None,
        offline: EngineProvider | None = None,
        monitor: ConnectivityMonitor | None = None,
        bus: EventBus | None = None,
    ) -> None:
        self._mode = mode
        self._online_provider = online
        self._offline_provider = offline
        self._monitor = monitor
        self._bus = bus
        self._lock = threading.RLock()
        self._online: SttEngine | None = None
        self._offline: SttEngine | None = None
        self._use_offline = False
        self._quota_notified = False
        self._preload: threading.Thread | None = None
        self._unsubscribe: Unsubscribe | None = None
        if bus is not None:
            self._unsubscribe = bus.subscribe(OnlineStatusChanged, self._on_online_status)

    @property
    def mode(self) -> SttMode:
        """The configured mode."""
        return self._mode

    @property
    def using_offline(self) -> bool:
        """Whether auto mode has fallen back and not yet returned."""
        with self._lock:
            return self._use_offline

    def preload(self) -> None:
        """Warm the local model in the background so a fallback is instant.

        Only useful in auto mode: offline mode loads on its first phrase anyway,
        and online mode should not be loading a model it will never use.
        Idempotent, and failures are logged rather than raised - a preload that
        did not work costs latency on the first fallback, not correctness.
        """
        if self._mode is not SttMode.AUTO or self._offline_provider is None:
            return
        with self._lock:
            if self._offline is not None:
                return
            if self._preload is not None and self._preload.is_alive():
                return
            self._preload = threading.Thread(
                target=self._preload_offline, name="ayris-stt-preload", daemon=True
            )
            self._preload.start()

    def close(self) -> None:
        """Drop the subscription. Engines belong to whoever provided them."""
        unsubscribe, self._unsubscribe = self._unsubscribe, None
        if unsubscribe is not None:
            unsubscribe()

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Recognise one phrase through whichever engine the mode allows.

        Raises:
            SttError: No engine could produce a result. The ``user_message``
                names the reason - no model, no key, or no network with nothing
                installed to fall back to.
        """
        if self._mode is SttMode.OFFLINE:
            return self._transcribe_offline()(audio)
        if self._mode is SttMode.ONLINE:
            return self._transcribe_online()(audio)
        return self._transcribe_auto(audio)

    def _transcribe_auto(self, audio: AudioBuffer) -> TranscriptResult:
        """Cloud first unless the network is known to be down."""
        if self._online_provider is None or self._prefer_offline():
            return self._transcribe_offline()(audio)
        try:
            result = self._transcribe_online()(audio)
        except _FALLBACK_ERRORS as exc:
            return self._fall_back(audio, exc)
        except SttError as exc:
            # A wrong key, a missing folder id, an answer in an unexpected shape:
            # permanent, and none of them a reason to leave the user without
            # recognition until they open the settings.
            return self._fall_back(audio, exc)
        self._note_success()
        return result

    def _fall_back(self, audio: AudioBuffer, exc: AyrisError) -> TranscriptResult:
        """Switch to the local model for this phrase and the ones after it.

        Raises:
            SttError: There is no local model, so the fallback has nowhere to go.
                Re-raised from the cloud failure with the offline problem in the
                message, because both halves matter to the user.
        """
        quota = isinstance(exc, QuotaError)
        with self._lock:
            first = not self._use_offline
            self._use_offline = True
        if isinstance(exc, NetworkError) and self._monitor is not None:
            self._monitor.report_failure(type(exc).__name__)
        _log.warning("stt router: cloud failed (%s), falling back to offline", exc.technical)

        if self._offline_provider is None:
            self._notify(
                "Распознавание недоступно",
                f"{exc.user_message} Локальная модель не установлена, "
                f"поэтому распознать фразу нечем.",
                level="error",
            )
            raise SttError(
                f"stt router: cloud failed and no offline engine is configured: {exc.technical}",
                user_message=(
                    f"{exc.user_message} Локальной модели нет — установите модель "
                    f"распознавания в настройках, чтобы помощник работал без интернета."
                ),
            ) from exc

        if first or (quota and not self._quota_notified):
            self._quota_notified = quota
            self._notify(
                "Распознавание переключено",
                (
                    f"{exc.user_message} Перешёл на локальную модель — "
                    f"вернусь в облако, когда связь восстановится."
                ),
                level="warning",
            )
        return self._transcribe_offline()(audio)

    def _transcribe_online(self) -> Callable[[AudioBuffer], TranscriptResult]:
        """The cloud engine's ``transcribe``, loading the engine on first use.

        Raises:
            SttError: No cloud provider is configured, or its load failed - a
                missing key being the usual reason.
        """
        with self._lock:
            engine = self._online
            if engine is not None:
                return engine.transcribe
            if self._online_provider is None:
                raise SttError(
                    "stt router: online mode without a cloud engine",
                    user_message=(
                        "Облачное распознавание выбрано, но провайдер не настроен. "
                        "Выберите сервис и сохраните ключ в настройках голоса."
                    ),
                )
            engine = self._online_provider()
            self._online = engine
            return engine.transcribe

    def _transcribe_offline(self) -> Callable[[AudioBuffer], TranscriptResult]:
        """The local engine's ``transcribe``, loading the model on first use.

        Raises:
            SttError: No local model is configured or it failed to load.
        """
        with self._lock:
            engine = self._offline
            if engine is not None:
                return engine.transcribe
            if self._offline_provider is None:
                raise SttError(
                    "stt router: no offline engine is configured",
                    user_message=(
                        "Локальная модель распознавания не установлена. "
                        "Выберите и скачайте модель в настройках голоса."
                    ),
                )
            engine = self._offline_provider()
            self._offline = engine
            return engine.transcribe

    def _prefer_offline(self) -> bool:
        """Whether auto mode should skip the cloud for this phrase."""
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
            self._transcribe_offline()
        except AyrisError as exc:
            _log.info("stt router: offline preload skipped: %s", exc.technical)
        except Exception:  # pragma: no cover - a bug, not a usage error
            _log.exception("stt router: offline preload failed")

    def _on_online_status(self, event: OnlineStatusChanged) -> None:
        """Return to the cloud when connectivity comes back."""
        if not event.online:
            return
        with self._lock:
            if not self._use_offline:
                return
            self._use_offline = False
        self._quota_notified = False
        _log.info("stt router: back online (%s), next phrase goes to the cloud", event.detail)
        self._notify(
            "Связь восстановлена",
            "Распознавание снова работает через облако.",
            level="info",
        )

    def _notify(self, title: str, message: str, *, level: str) -> None:
        """Put a notification on the bus, if there is a bus."""
        if self._bus is None:
            return
        self._bus.publish(
            NotificationRequested(title=title, message=message, level=level, timeout_ms=6000)
        )
