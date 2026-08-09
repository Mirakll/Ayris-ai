"""Task 11: the three-mode router and the connectivity monitor.

Nothing here talks to a real network. The engines are stand-ins with controllable
behaviour, the monitor probes an :class:`httpx.MockTransport`, and the bus is a
plain in-memory one - so the tests are really about the decisions: which engine
gets the phrase, when a failure flips the mode, and how the state comes back.

Groups:

* :class:`TestMode` — ``SttMode`` is parsed from settings strings, tolerantly.
* :class:`TestAuto` — the fallback, its notifications, and the return to the cloud.
* :class:`TestTranscribeModes` — what each of the three modes does with the same
  failing engine.
* :class:`TestMonitor` — the probe state machine: one failure goes offline, two
  good probes come back, and a silent probe_url never starts a thread.
* :class:`TestNoEngine` — the dead ends, each with a message that says what to do.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from ayris.audio.stt.base import AudioBuffer, TranscriptResult
from ayris.audio.stt.cloud_base import NetworkError, QuotaError
from ayris.audio.stt.router import SttMode, SttRouter
from ayris.core.connectivity import ConnectivityMonitor
from ayris.core.errors import SttError
from ayris.core.events import EventBus, NotificationRequested, OnlineStatusChanged

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

#: Head of the path every engine's :attr:`ayris.audio.stt.base.TranscriptResult.engine`
#: is reported under; used only to tell the engines apart in assertions.
LOCAL = "local"
CLOUD = "cloud"


class RecordingBus(EventBus):
    """An event bus that keeps every published event for assertions."""

    def __init__(self) -> None:
        super().__init__(thread_id=None)
        self.events: list[Any] = []

    def publish(self, event: Any) -> None:
        self.events.append(event)
        super().publish(event)

    def of_type(self, kind: type) -> list[Any]:
        return [event for event in self.events if isinstance(event, kind)]


class StubEngine:
    """A configurable recogniser: succeeds, fails, or counts its calls.

    Not an :class:`SttEngine` subclass: the router only needs ``load`` and
    ``transcribe``, and a stub that cannot be loaded is just as useful in a test
    as one that can.
    """

    def __init__(self, name: str = LOCAL, exc: Exception | None = None) -> None:
        self.name = name
        self.exc = exc
        self.loads = 0
        self.calls = 0
        self.unloads = 0

    def load(self, model_path: Any, options: Any) -> None:
        self.loads += 1

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        return TranscriptResult(text="офлайн", engine=self.name, device="cpu")

    def unload(self) -> None:
        self.unloads += 1


class CountingMonitor(ConnectivityMonitor):
    """A monitor that records what the router told it.

    A subclass rather than a mock: the router's contract with the monitor is two
    calls, and asserting on the calls is what says "a wrong key is not reported
    as a network problem" — which no amount of state inspection can, because both
    signals can leave the state unchanged.
    """

    def __init__(self, bus: RecordingBus) -> None:
        super().__init__(bus, probe_url="")
        self.failures = 0
        self.successes = 0

    def report_failure(self, reason: str = "") -> None:
        self.failures += 1
        super().report_failure(reason)

    def report_success(self) -> None:
        self.successes += 1
        super().report_success()


def local(*, exc: Exception | None = None) -> StubEngine:
    return StubEngine(name=LOCAL, exc=exc)


def cloud(*, exc: Exception | None = None) -> StubEngine:
    return StubEngine(name=CLOUD, exc=exc)


def speech(ms: int = 900) -> AudioBuffer:
    from array import array

    rate = 16000
    count = rate * ms // 1000
    samples = array("h", (100 * (index % 17) for index in range(count)))
    return AudioBuffer(samples.tobytes(), sample_rate=rate, channels=1)


def provider(engine: StubEngine) -> Callable[[], StubEngine]:
    """A provider that loads its engine, the way the real ones do.

    The router deliberately does not load anything itself, so ``engine.loads``
    counting up is the only evidence that the router asked for the engine at all.
    """

    def provide() -> StubEngine:
        engine.load(None, None)
        return engine

    return provide


def wait_for(predicate: Callable[[], bool], *, timeout: float = 5.0) -> bool:
    """Poll until ``predicate`` holds. For the preload thread, which is async.

    A deadline rather than a sleep: the sandbox has no timing guarantees, and a
    fixed sleep would either be flaky or slow.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def router(
    bus: RecordingBus,
    *,
    mode: SttMode = SttMode.AUTO,
    offline: StubEngine | None = None,
    online: StubEngine | None = None,
    monitor: ConnectivityMonitor | None = None,
) -> SttRouter:
    return SttRouter(
        mode=mode,
        online=provider(online) if online is not None else None,
        offline=provider(offline) if offline is not None else None,
        monitor=monitor,
        bus=bus,
    )


@pytest.fixture
def bus() -> RecordingBus:
    return RecordingBus()


@pytest.fixture
def probe(bus: RecordingBus) -> Iterator[ConnectivityMonitor]:
    """A monitor on the test's bus, wired to a MockTransport and torn down after.

    Sharing the bus is the point: the router's return to the cloud is driven by
    the monitor's event, and a monitor publishing to its own bus would let a
    broken subscription pass.
    """
    monitor_instance = ConnectivityMonitor(
        bus,
        probe_url="https://probe.example/204",
        interval_sec=60.0,
        transport=httpx.MockTransport(lambda _request: httpx.Response(204)),
    )
    monitor_instance.start()
    yield monitor_instance
    monitor_instance.stop()


class TestMode:
    """«Готово когда»: the mode in the settings is read without being broken by a typo."""

    def test_known_modes_parse_exactly(self) -> None:
        assert SttMode.parse("auto") is SttMode.AUTO
        assert SttMode.parse(" offline ") is SttMode.OFFLINE
        assert SttMode.parse("ONLINE") is SttMode.ONLINE

    def test_an_unknown_mode_becomes_auto(self) -> None:
        assert SttMode.parse("hover") is SttMode.AUTO

    def test_an_empty_string_is_auto(self) -> None:
        assert SttMode.parse("") is SttMode.AUTO


class TestAuto:
    """The fallback: fast, once, notified, and reversed by the monitor."""

    def test_the_cloud_gets_the_phrase_while_it_works(self, bus: RecordingBus) -> None:
        offline_engine = local()
        cloud_engine = cloud()
        instance = router(bus, offline=offline_engine, online=cloud_engine)
        result = instance.transcribe(speech())
        assert result.engine == CLOUD
        assert cloud_engine.calls == 1
        assert offline_engine.calls == 0

    def test_a_network_error_falls_back_for_this_phrase(self, bus: RecordingBus) -> None:
        offline_engine = local()
        instance = router(
            bus,
            offline=offline_engine,
            online=cloud(exc=NetworkError("stub: down", user_message="Сервис недоступен.")),
        )
        result = instance.transcribe(speech())
        assert result.engine == LOCAL
        assert instance.using_offline

    def test_a_quota_error_falls_back_without_retrying(self, bus: RecordingBus) -> None:
        cloud_engine = cloud(exc=QuotaError("stub: 429", user_message="Превышен лимит."))
        instance = router(bus, offline=local(), online=cloud_engine)
        result = instance.transcribe(speech())
        assert result.engine == LOCAL
        assert cloud_engine.calls == 1
        assert instance.using_offline

    def test_a_wrong_key_falls_back_too(self, bus: RecordingBus) -> None:
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=SttError("stub: 401", user_message="Ключ не принят.")),
        )
        result = instance.transcribe(speech())
        assert result.engine == LOCAL

    def test_the_fallback_happens_once_not_per_phrase(self, bus: RecordingBus) -> None:
        offline_engine = local()
        instance = router(
            bus,
            offline=offline_engine,
            online=cloud(exc=NetworkError("stub", user_message="Сервис недоступен.")),
        )
        instance.transcribe(speech())
        instance.transcribe(speech())
        assert instance.using_offline
        assert offline_engine.calls == 2

    def test_the_monitor_is_told_the_network_failed(
        self, bus: RecordingBus, probe: ConnectivityMonitor
    ) -> None:
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=NetworkError("stub", user_message="Сервис недоступен.")),
            monitor=probe,
        )
        assert probe.online
        instance.transcribe(speech())
        assert not probe.online
        assert probe.probing

    def test_a_wrong_key_is_not_reported_as_a_network_failure(
        self, bus: RecordingBus, probe: ConnectivityMonitor
    ) -> None:
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=SttError("stub: 401", user_message="Ключ не принят.")),
            monitor=probe,
        )
        instance.transcribe(speech())
        assert probe.online

    def test_the_return_to_the_cloud_comes_from_the_monitor(
        self, bus: RecordingBus, probe: ConnectivityMonitor
    ) -> None:
        cloud_engine = cloud(exc=NetworkError("stub", user_message="Сервис недоступен."))
        instance = router(
            bus,
            offline=local(),
            online=cloud_engine,
            monitor=probe,
        )
        instance.transcribe(speech())
        assert instance.using_offline

        # The network came back, so the cloud works again.
        cloud_engine.exc = None
        probe.report_success()
        assert not instance.using_offline

        result = instance.transcribe(speech())
        assert result.engine == CLOUD

    def test_notifications_describe_what_happened(self, bus: RecordingBus) -> None:
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=NetworkError("stub", user_message="Сервис недоступен.")),
        )
        instance.transcribe(speech())
        assert bus.of_type(NotificationRequested)
        notification = bus.of_type(NotificationRequested)[0]
        assert notification.level == "warning"
        assert "локальн" in notification.message.lower()

    def test_a_quota_notification_is_distinct(self, bus: RecordingBus) -> None:
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=QuotaError("stub", user_message="Превышен лимит.")),
        )
        instance.transcribe(speech())
        message = bus.of_type(NotificationRequested)[0].message
        assert "лимит" in message.lower()

    def test_a_cloud_answer_is_reported_to_the_monitor(self, bus: RecordingBus) -> None:
        instance_monitor = CountingMonitor(bus)
        instance = router(bus, offline=local(), online=cloud(), monitor=instance_monitor)
        instance.transcribe(speech())
        assert instance_monitor.successes == 1
        assert instance_monitor.failures == 0

    def test_only_a_network_failure_is_reported_to_the_monitor(self, bus: RecordingBus) -> None:
        instance_monitor = CountingMonitor(bus)
        instance = router(
            bus,
            offline=local(),
            online=cloud(exc=NetworkError("stub", user_message="Сервис недоступен.")),
            monitor=instance_monitor,
        )
        instance.transcribe(speech())
        assert instance_monitor.failures == 1
        assert instance_monitor.successes == 0

    def test_preload_warms_the_local_model_in_auto_mode(self, bus: RecordingBus) -> None:
        offline_engine = local()
        instance = router(bus, offline=offline_engine, online=cloud())
        instance.preload()
        instance.preload()
        # The load happens on a thread; wait for it rather than assuming timing.
        assert wait_for(lambda: offline_engine.loads >= 1)
        assert offline_engine.calls == 0  # loaded, not yet asked to recognise

    def test_preload_does_not_touch_the_offline_engine_in_online_mode(
        self, bus: RecordingBus
    ) -> None:
        offline_engine = local()
        instance = router(bus, mode=SttMode.ONLINE, offline=offline_engine)
        instance.preload()
        assert offline_engine.loads == 0

    def test_the_router_unsubscribes_on_close(self, bus: RecordingBus) -> None:
        instance = router(bus, offline=local())
        instance.close()
        assert bus.subscriber_count(OnlineStatusChanged) == 0


class TestTranscribeModes:
    """«Готово когда»: each mode answers the way its name promises."""

    def test_offline_mode_never_touches_the_cloud(self, bus: RecordingBus) -> None:
        cloud_engine = cloud()
        offline_engine = local()
        instance = router(bus, mode=SttMode.OFFLINE, offline=offline_engine, online=cloud_engine)
        result = instance.transcribe(speech())
        assert result.engine == LOCAL
        assert cloud_engine.calls == 0

    def test_online_mode_never_falls_back(self, bus: RecordingBus) -> None:
        cloud_engine = cloud(exc=NetworkError("stub", user_message="Сервис недоступен."))
        instance = router(bus, mode=SttMode.ONLINE, offline=local(), online=cloud_engine)
        with pytest.raises(NetworkError):
            instance.transcribe(speech())
        assert not instance.using_offline

    def test_online_mode_keeps_working_when_the_cloud_is_happy(self, bus: RecordingBus) -> None:
        instance = router(bus, mode=SttMode.ONLINE, offline=local(), online=cloud(exc=None))
        result = instance.transcribe(speech())
        assert result.engine == CLOUD

    def test_auto_without_a_cloud_engine_is_offline(self, bus: RecordingBus) -> None:
        offline_engine = local()
        instance = router(bus, offline=offline_engine, online=None)
        result = instance.transcribe(speech())
        assert result.engine == LOCAL


class TestMonitor:
    """The probe state machine, driven by a mock transport."""

    def test_online_by_default(self) -> None:
        instance = ConnectivityMonitor(RecordingBus(), probe_url="https://probe.example/204")
        assert instance.online

    def test_one_failure_flips_offline_and_publishes(self, bus: RecordingBus) -> None:
        instance = ConnectivityMonitor(bus, probe_url="")
        instance.report_failure("timed out")
        assert not instance.online
        status = bus.of_type(OnlineStatusChanged)[-1]
        assert status.online is False

    def test_two_good_probes_come_back_but_one_is_not_enough(self, bus: RecordingBus) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(204))
        instance = ConnectivityMonitor(
            bus, probe_url="https://probe.example/204", transport=transport
        )
        instance.report_failure("test")
        assert not instance.online

        instance.check_now()
        assert not instance.online
        instance.check_now()
        assert instance.online

    def test_a_failing_probe_keeps_the_state_offline(self, bus: RecordingBus) -> None:
        transport = httpx.MockTransport(
            lambda _request: (_ for _ in ()).throw(httpx.ConnectError("down"))
        )
        instance = ConnectivityMonitor(
            bus, probe_url="https://probe.example/204", transport=transport
        )
        instance.report_failure("test")
        assert not instance.online
        assert not instance.check_now()

    def test_any_answer_counts_as_reachable(self, bus: RecordingBus) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(404))
        instance = ConnectivityMonitor(
            bus, probe_url="https://probe.example/204", transport=transport
        )
        instance.report_failure("test")
        instance.check_now()
        assert not instance.online
        instance.check_now()
        assert instance.online

    def test_no_probe_url_never_probes(self) -> None:
        instance = ConnectivityMonitor(RecordingBus(), probe_url="")
        instance.start()
        assert not instance.probing
        assert instance.check_now()

    def test_a_success_after_a_failure_goes_online_immediately(self, bus: RecordingBus) -> None:
        instance = ConnectivityMonitor(bus, probe_url="")
        instance.report_failure("test")
        assert not instance.online
        instance.report_success()
        assert instance.online

    def test_the_probe_thread_stops_cleanly(self) -> None:
        transport = httpx.MockTransport(lambda _request: httpx.Response(204))
        instance = ConnectivityMonitor(
            RecordingBus(), probe_url="https://probe.example/204", transport=transport
        )
        instance.start()
        assert instance.probing
        instance.stop()
        assert not instance.probing


class TestNoEngine:
    """The dead ends, and the messages that say what to do about them."""

    def test_auto_with_no_offline_model_names_both_problems(self, bus: RecordingBus) -> None:
        instance = router(
            bus,
            online=cloud(exc=NetworkError("stub", user_message="Сервис недоступен.")),
        )
        with pytest.raises(SttError) as info:
            instance.transcribe(speech())
        assert "локальн" in info.value.user_message.lower()

    def test_offline_mode_with_no_model_says_what_is_missing(self, bus: RecordingBus) -> None:
        instance = router(bus, mode=SttMode.OFFLINE)
        with pytest.raises(SttError) as info:
            instance.transcribe(speech())
        assert "не установлена" in info.value.user_message

    def test_online_mode_with_no_provider_says_what_to_configure(self, bus: RecordingBus) -> None:
        instance = router(bus, mode=SttMode.ONLINE)
        with pytest.raises(SttError) as info:
            instance.transcribe(speech())
        assert "провайдер" in info.value.user_message.lower()
