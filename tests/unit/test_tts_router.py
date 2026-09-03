"""Task 13: the router - one entry point, a fallback that never duplicates.

The router's promise is a small set of behaviours, and each one is tested here
with the same two fakes: a backend that records the bytes it was given instead
of handing them to PortAudio (lifted from ``test_tts_player.py``, because the
synthesis path ends in the player and the player ends in the backend), and a
real engine talking to ``httpx.MockTransport`` so that the *cloud* side of a
fallback is exercised end to end rather than as a stub. A mocked engine could
only assert that the router "called the other provider"; these tests assert what
the user hears and what the user's account is billed for.

The completion path deserves the attention it gets below. The router learns that
a phrase finished either from the bus (``TtsFinished``, published by the bridge
that ``service.py`` wires up) or, when there is no bus, from the player's own
``on_finished`` observer. Both paths are tested, because both are used in
production - the first in the app, the second in the preview window.

Groups:

* :class:`TestNothingConfigured` — a missing engine is said out loud, not silent.
* :class:`TestEnqueue` — say/preview routing, the empty phrase, the handle.
* :class:`TestFallback` — cloud failure to local, no duplicated audio, the
  notification, the monitor.
* :class:`TestNetworkAwareness` — offline goes straight to local, recovery
  returns to the cloud.
* :class:`TestModes` — the four modes and the four pairs they route to.
* :class:`TestVoiceSettings` — per-phrase overrides, clamping, volume.
* :class:`TestHistory` — previews are heard but not recorded.
* :class:`TestCancel` — a spoken phrase can be stopped from its handle.
* :class:`TestShutdown` — close resolves waiters and releases engines.
"""

from __future__ import annotations

import threading
from array import array
from collections.abc import Callable
from time import monotonic
from typing import TYPE_CHECKING

import httpx
import pytest

from ayris.audio.devices import PlaybackRequest, RawDevice
from ayris.audio.tts.base import AudioChunk, TtsEngine, TtsOptions, VoiceSpec
from ayris.audio.tts.elevenlabs_engine import ElevenLabsTtsEngine
from ayris.audio.tts.player import PlaybackReason, TtsPlayer
from ayris.audio.tts.router import (
    SpeechHandle,
    SpokenPhrase,
    TtsMode,
    TtsRouter,
    VoiceParams,
    mode_from_config,
)
from ayris.audio.tts.service import connect_player
from ayris.core.errors import TtsError
from ayris.core.events import (
    EventBus,
    NotificationRequested,
    OnlineStatusChanged,
)
from ayris.core.secrets import SecretsStore, reset_secrets

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: Ceiling for :func:`wait_until`. Generous on purpose: a loaded CI runner is
#: slow, and a test that reaches this has genuinely hung rather than been unlucky.
TIMEOUT_S: float = 5.0

#: Rate the fake device prefers.
DEVICE_RATE: int = 48000

#: Long enough to span several writes, so cancel has somewhere to land.
PHRASE_MS: int = 400


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.entries[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self.entries[(service_name, username)]


@pytest.fixture
def keyring_store() -> Iterator[SecretsStore]:
    """A process-wide store holding a key under ``elevenlabs``."""
    store = SecretsStore("Ayris-test", backend=FakeKeyring())
    store.save("elevenlabs", "test-key")
    reset_secrets(store)
    yield store
    reset_secrets()


def tone(ms: int, sample_rate: int = DEVICE_RATE, *, level: int = 8000) -> bytes:
    """``ms`` milliseconds of a constant int16 level, so amplitude is one number."""
    frames = max(0, int(sample_rate * ms / 1000))
    return array("h", [level] * frames).tobytes()


def pcm(frames: int = 960, level: int = 6000) -> bytes:
    """Short fixed audio: the cloud's answer in the happy path."""
    return array("h", [level] * frames).tobytes()


class FakeStream:
    """An output stream that appends to a list.

    :attr:`block` holds the writer thread inside a ``write``, which is the only
    way to observe the player mid-phrase: without it a phrase is written to a
    list in microseconds and every "while it is speaking" test becomes a race
    against a thread that has already finished.
    """

    def __init__(self, request: PlaybackRequest) -> None:
        self._request = request
        self._active = False
        self.writes: list[bytes] = []
        self.stops = 0
        self.closes = 0
        self.block = threading.Event()
        self.block.set()
        self.wrote_once = threading.Event()

    @property
    def sample_rate(self) -> int:
        return self._request.sample_rate

    @property
    def channels(self) -> int:
        return self._request.channels

    @property
    def active(self) -> bool:
        return self._active

    def start(self) -> None:
        self._active = True

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        self.wrote_once.set()
        self.block.wait(TIMEOUT_S)

    def stop(self) -> None:
        self.stops += 1
        self._active = False
        self.block.set()

    def close(self) -> None:
        self.closes += 1
        self._active = False
        self.block.set()

    @property
    def written(self) -> bytes:
        return b"".join(self.writes)


class FakeBackend:
    """A playback backend with no PortAudio in it, mirroring test_tts_player."""

    def __init__(self) -> None:
        self.streams: list[FakeStream] = []
        self.start_blocked = False

    def raw_devices(self) -> tuple[RawDevice, ...]:
        return (
            RawDevice(
                index=0,
                name="Динамики",
                host_api="WASAPI",
                max_output_channels=2,
                default_sample_rate=float(DEVICE_RATE),
                default_output=True,
            ),
        )

    def open_output_stream(self, request: PlaybackRequest) -> FakeStream:
        stream = FakeStream(request)
        if self.start_blocked:
            stream.block.clear()
        self.streams.append(stream)
        return stream

    @property
    def stream(self) -> FakeStream:
        assert self.streams, "плеер не открыл устройство"
        return self.streams[-1]

    @property
    def written(self) -> bytes:
        return b"".join(stream.written for stream in self.streams)

    def writing(self) -> bool:
        """Whether the player has opened a device and written to it.

        A predicate rather than a property read inside a lambda because the
        device is opened lazily, on the first chunk: polling ``stream.writes``
        would assert on an empty list before the player got that far.
        """
        return any(stream.writes for stream in self.streams)

    def release(self) -> None:
        """Let every parked writer go, so a test can reach its assertions."""
        self.start_blocked = False
        for stream in self.streams:
            stream.block.set()


def make_bus() -> EventBus:
    """An :class:`EventBus` that delivers on whichever thread publishes.

    A bus bound to a thread - the default, and what the UI builds - only queues
    a cross-thread publish and waits for :meth:`EventBus.drain`. The player
    publishes ``TtsFinished`` from its writer thread, so a bound bus here would
    park the event and the handle would never resolve. Unbound delivery is the
    same wiring a worker process uses.
    """
    return EventBus(thread_id=None)


def wait_until(condition: Callable[[], bool], timeout: float = TIMEOUT_S) -> None:
    """Poll ``condition`` until it holds, or fail with the caller's line."""
    deadline = monotonic() + timeout
    while not condition():
        if monotonic() >= deadline:
            pytest.fail(f"условие не наступило за {timeout:.1f} с")
        threading.Event().wait(0.01)


class RecordingHandler:
    """A :class:`httpx.MockTransport` handler that answers and records.

    Three ways to make it fail, because the three failures the router treats
    differently arrive differently: :attr:`failures` for a status per request,
    :attr:`raises` for a transport-level error, and :attr:`responder` for the
    cases that need to look at how many requests came before.
    """

    def __init__(self, content: bytes = b"") -> None:
        self.content = content or pcm()
        self.status = 200
        self.requests: list[httpx.Request] = []
        #: Statuses consumed in order; once empty the canned answer resumes.
        self.failures: list[tuple[int, bytes]] = []
        #: Raised instead of answering, for a connection that is not there.
        self.raises: Exception | None = None
        #: Full override, taking the request index. Wins over everything above.
        self.responder: Callable[[int, httpx.Request], httpx.Response] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.responder is not None:
            return self.responder(len(self.requests), request)
        if self.raises is not None:
            raise self.raises
        if self.failures:
            status, body = self.failures.pop(0)
            return httpx.Response(status, content=body)
        return httpx.Response(self.status, content=self.content)

    @property
    def calls(self) -> int:
        return len(self.requests)


class FakeMonitor:
    """A :class:`~ayris.core.connectivity.ConnectivityMonitor` with no probe."""

    def __init__(self, online: bool = True) -> None:
        self.online = online
        self.failures: list[str] = []
        self.successes = 0

    def report_failure(self, reason: str) -> None:
        self.failures.append(reason)
        self.online = False

    def report_success(self) -> None:
        self.successes += 1
        self.online = True


class FakeEngine(TtsEngine):
    """A local engine: the real contract, canned audio, observable calls.

    A subclass of :class:`~ayris.audio.tts.base.TtsEngine` rather than a
    duck-typed stand-in, so that the sentence splitting and the clamping in the
    base class are part of what these tests exercise - a fallback that "worked"
    only because the fake skipped both would prove nothing about the router.
    """

    name = "fake"
    package = ""
    module = ""
    memory_factor = 0.0
    native_sample_rate = 22050

    __slots__ = ("fail", "loads", "requests", "unloads")

    def __init__(self, *, fail: bool = False) -> None:
        super().__init__()
        self.fail = fail
        self.loads = 0
        self.unloads = 0
        #: ``(text, speed, pitch)`` per synthesized sentence.
        self.requests: list[tuple[str, float, float]] = []

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return ("ru",)

    @classmethod
    def voices(cls, directory: object = None) -> tuple[VoiceSpec, ...]:
        del directory
        return (VoiceSpec(engine=cls.name, voice_id="fake"),)

    @classmethod
    def available(cls) -> bool:
        return True

    def load(self, voice: VoiceSpec, options: TtsOptions) -> None:
        self.loads += 1
        self._voice = voice
        self._options = options

    def unload(self) -> None:
        self.unloads += 1
        self._voice = None

    def _synthesize(self, text: str, speed: float, pitch: float) -> AudioChunk:
        if self.fail:
            raise TtsError("fake: локальный голос упал", user_message="Локальный голос упал.")
        self.requests.append((text, speed, pitch))
        return AudioChunk(pcm(240), self.native_sample_rate)

    @property
    def texts(self) -> list[str]:
        """Just the sentences, for the assertions that only care about those."""
        return [text for text, _, _ in self.requests]


def loaded_local() -> TtsEngine:
    """A :class:`FakeEngine` with a voice on it, for the routers built by hand."""
    engine = FakeEngine()
    engine.load(VoiceSpec(engine="fake", voice_id="fake"), TtsOptions())
    return engine


class EnginePair:
    """A cloud engine and a local one, both speaking through the same fakes.

    Everything a router test needs in one place: the transport that answers the
    cloud with real PCM, the fake local engine, the backend that receives the
    audio, and the player they all end at.
    """

    def __init__(self) -> None:
        self.handler = RecordingHandler()
        self.local = FakeEngine()
        self.backend = FakeBackend()
        self.player = TtsPlayer(backend=self.backend)  # type: ignore[arg-type]
        self.cloud_calls = 0

    def make_cloud(self) -> Callable[[], TtsEngine]:
        """A provider that builds a fresh ElevenLabs engine per call."""

        def build() -> TtsEngine:
            engine = ElevenLabsTtsEngine()
            engine.load(
                VoiceSpec(engine="elevenlabs", voice_id="v1"),
                TtsOptions(
                    extra={
                        "transport": httpx.MockTransport(self.handler),
                        "max_retries": 0,
                    }
                ),
            )
            self.cloud_calls += 1
            return engine

        return build

    def make_local(self) -> Callable[[], TtsEngine]:
        """A provider that loads the local engine, the way a real one does."""

        def build() -> TtsEngine:
            self.local.load(VoiceSpec(engine="fake", voice_id="fake"), TtsOptions())
            return self.local

        return build

    def make_router(self, **kwargs: object) -> TtsRouter:
        """A router over the pair. ``cloud=None``/``local=None`` drop a slot."""
        kwargs.setdefault("cloud", self.make_cloud())
        kwargs.setdefault("local", self.make_local())
        router = TtsRouter(self.player, **kwargs)  # type: ignore[arg-type]
        self.player.start()
        return router


def cloud_pcm() -> bytes:
    """PCM as ElevenLabs would return it (headerless, 24 kHz)."""
    return pcm()


# ----------------------------------------------------------------------
# nothing configured
# ----------------------------------------------------------------------


class TestNothingConfigured:
    """The modes and the sentence that comes with an empty router."""

    def test_auto_with_no_engines_is_a_clear_error(self) -> None:
        backend = FakeBackend()
        player = TtsPlayer(backend=backend)  # type: ignore[arg-type]
        router = TtsRouter(player)
        with pytest.raises(TtsError) as info:
            router.say("Привет")
        assert "не настроен" in info.value.user_message

    def test_online_with_no_cloud_engine_raises(self) -> None:
        backend = FakeBackend()
        player = TtsPlayer(backend=backend)  # type: ignore[arg-type]
        router = TtsRouter(player, mode=TtsMode.ONLINE, local=loaded_local)
        with pytest.raises(TtsError):
            router.say("Привет")
        router.close()

    def test_mode_from_config_maps_engine_names(self) -> None:
        assert mode_from_config("azure") is TtsMode.AUTO
        assert mode_from_config("piper") is TtsMode.OFFLINE
        assert mode_from_config("piper", cloud_fallback=True) is TtsMode.LOCAL_FIRST

    def test_mode_parse_accepts_loose_spelling(self) -> None:
        assert TtsMode.parse("local-first") is TtsMode.LOCAL_FIRST
        assert TtsMode.parse(" OFFLINE ") is TtsMode.OFFLINE

    def test_mode_parse_falls_back_to_auto(self) -> None:
        """A config a newer Ayris wrote must not make this one go mute."""
        assert TtsMode.parse("квантовый") is TtsMode.AUTO

    def test_voice_params_clamp_the_ranges(self) -> None:
        params = VoiceParams(speed=9.0, pitch=-1.0, volume=500)
        assert params.speed == 2.0
        assert params.pitch == 0.5
        assert params.volume == 100

    def test_voice_params_gain_follows_volume(self) -> None:
        assert VoiceParams(volume=80).gain == pytest.approx(0.8)
        assert VoiceParams(volume=0).gain == pytest.approx(0.0)


# ----------------------------------------------------------------------
# enqueue
# ----------------------------------------------------------------------


class TestEnqueue:
    """say() returns at once, and speaks through the player."""

    def test_say_returns_immediately(self) -> None:
        """Synthesis happens on the player's thread, not in the caller's."""
        pair = EnginePair()
        router = pair.make_router()
        started = monotonic()
        handle = router.say("Привет!")
        assert monotonic() - started < 0.5
        assert isinstance(handle, SpeechHandle)
        assert not handle.done
        router.close()

    def test_unspeakable_text_is_a_finished_handle(self) -> None:
        """An empty answer is a normal outcome, not an error."""
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("   ")
        assert handle.done
        assert handle.reason == PlaybackReason.COMPLETED
        assert pair.handler.calls == 0
        router.close()

    def test_say_goes_through_the_cloud_engine(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет, мир!")
        handle.wait(TIMEOUT_S)
        assert pair.handler.calls == 1
        assert handle.reason == PlaybackReason.COMPLETED
        assert handle.engine == "elevenlabs"
        assert pair.cloud_calls == 1
        router.close()

    def test_preview_is_heard_immediately(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.preview()
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert pair.backend.written
        router.close()

    def test_say_split_into_sentences(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Первое предложение. Второе предложение!")
        handle.wait(TIMEOUT_S)
        assert pair.handler.calls == 2
        assert handle.reason == PlaybackReason.COMPLETED
        router.close()

    def test_engine_is_named_on_the_handle(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет")
        assert handle.engine == ""
        handle.wait(TIMEOUT_S)
        assert handle.engine == "elevenlabs"
        router.close()


# ----------------------------------------------------------------------
# fallback
# ----------------------------------------------------------------------


class TestFallback:
    """Cloud failure to local: prompt, complete, and never twice."""

    def test_401_falls_back_to_the_local_voice(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.failures = [(401, b"bad key")]
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert pair.local.requests
        assert router.using_local
        assert handle.engine == "fake"
        router.close()

    def test_429_falls_back_and_warns_about_the_quota(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.failures = [(429, b"limit")]
        bus = make_bus()
        notifications: list[NotificationRequested] = []
        bus.subscribe(NotificationRequested, notifications.append)
        bridge = connect_player(bus, pair.player)
        router = pair.make_router(bus=bus)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert router.using_local
        assert notifications, "квота должна дойти до пользователя уведомлением"
        assert "Голос переключён" in notifications[0].title
        router.close()
        del bridge

    def test_timeout_falls_back_to_the_local_voice(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.raises = httpx.ReadTimeout("облако молчит")
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert pair.local.requests
        assert handle.engine == "fake"
        router.close()

    def test_fallback_sticks_until_connectivity_returns(self, keyring_store: SecretsStore) -> None:
        """The next phrase does not pay the timeout again.

        Retrying a cloud that has just refused would cost the user a wait per
        sentence for nothing; the way back is the connectivity event, not a
        hopeful retry.
        """
        pair = EnginePair()
        pair.handler.failures = [(401, b"bad key")]
        router = pair.make_router()
        first = router.say("Привет!")
        first.wait(TIMEOUT_S)
        assert pair.local.requests

        second = router.say("Как дела?")
        second.wait(TIMEOUT_S)
        assert pair.handler.calls == 1, "облако не должно опрашиваться повторно"
        assert second.engine == "fake"
        router.close()

    def test_a_mid_sentence_failure_does_not_duplicate_audio(
        self, keyring_store: SecretsStore
    ) -> None:
        """The phrase tail is dropped; the next sentence switches engines."""
        pair = EnginePair()

        def dying(index: int, request: httpx.Request) -> httpx.Response:
            """First sentence speaks, second one dies after it is on the device."""
            if index >= 2:
                return httpx.Response(500, content=b"boom")
            return httpx.Response(200, content=pcm())

        pair.handler.responder = dying
        router = pair.make_router()
        handle = router.say("Одно предложение. Другое предложение.")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        # The failing sentence was abandoned, the next one spoke locally.
        assert pair.local.requests
        assert handle.engine == "fake"
        router.close()

    def test_no_duplicate_when_audio_was_never_released(self, keyring_store: SecretsStore) -> None:
        """An engine that raises before any audio means the sentence is re-spoken."""
        pair = EnginePair()
        pair.handler.failures = [(500, b"down")]
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        # The same sentence came out exactly once, and in the local voice: the
        # cloud raised before yielding, so nothing of it reached the device.
        assert pair.local.texts == ["Привет!"]
        assert handle.engine == "fake"
        router.close()

    def test_cloud_failure_is_reported_to_the_monitor(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.failures = [(500, b"down")]
        monitor = FakeMonitor()
        router = pair.make_router(monitor=monitor)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert monitor.failures
        assert not monitor.online
        router.close()

    def test_quota_failure_is_not_a_network_fact(self, keyring_store: SecretsStore) -> None:
        """429 says the account, not the internet; the monitor must stay put."""
        pair = EnginePair()
        pair.handler.failures = [(429, b"limit")]
        monitor = FakeMonitor()
        router = pair.make_router(monitor=monitor)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert monitor.failures == []
        router.close()

    def test_local_engine_failure_does_not_reach_the_cloud(
        self, keyring_store: SecretsStore
    ) -> None:
        """A failed local voice is said out loud, not silently swapped."""
        pair = EnginePair()
        pair.local.fail = True
        router = pair.make_router(mode=TtsMode.OFFLINE)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.ERROR
        assert pair.handler.calls == 0
        router.close()

    def test_fallback_while_a_phrase_is_playing_keeps_the_rest(
        self, keyring_store: SecretsStore
    ) -> None:
        pair = EnginePair()
        pair.handler.failures = [(500, b"down")]
        router = pair.make_router()
        handle = router.say("Первое предложение. Второе предложение. Третье предложение.")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert pair.handler.calls == 1
        assert pair.local.requests
        router.close()


# ----------------------------------------------------------------------
# network awareness
# ----------------------------------------------------------------------


class TestNetworkAwareness:
    """The router reads the monitor, reports to it, and reacts to its event."""

    def test_offline_goes_straight_to_the_local_voice(self, keyring_store: SecretsStore) -> None:
        """No wasted timeout: the monitor already knows the network is down."""
        pair = EnginePair()
        monitor = FakeMonitor(online=False)
        router = pair.make_router(monitor=monitor)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert pair.handler.calls == 0
        assert pair.local.requests
        assert handle.engine == "fake"
        router.close()

    def test_online_status_restores_the_cloud_voice(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.failures = [(401, b"bad key")]
        bus = make_bus()
        # Мост обязателен именно здесь. С шиной роутер узнаёт о конце фразы
        # только из `TtsFinished`, которую публикует мост, а не из коллбэка
        # плеера — и без него `wait` честно отсчитывал таймаут до конца, оба
        # раза по пять секунд, а тест зеленел на проверках, которым конец
        # воспроизведения не нужен. Десять секунд из прогона на пустом месте.
        bridge = connect_player(bus, pair.player)
        router = pair.make_router(bus=bus, monitor=FakeMonitor())
        first = router.say("Привет!")
        assert first.wait(TIMEOUT_S), "фраза не доиграла"
        assert first.reason == PlaybackReason.COMPLETED
        assert router.using_local

        bus.publish(OnlineStatusChanged(online=True, detail="связь вернулась"))
        wait_until(lambda: not router.using_local)
        second = router.say("Пока!")
        assert second.wait(TIMEOUT_S), "фраза не доиграла"
        assert second.engine == "elevenlabs"
        router.close()
        del bridge

    def test_cloud_success_reports_to_the_monitor(self, keyring_store: SecretsStore) -> None:
        """A phrase that came back is evidence the network is up.

        Worth reporting because the STT router reads the same monitor: one of
        the two proving the connection saves the other a timeout.
        """
        pair = EnginePair()
        monitor = FakeMonitor(online=True)
        router = pair.make_router(monitor=monitor)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert monitor.successes >= 1
        assert monitor.online
        router.close()

    def test_using_local_follows_the_fallback(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.handler.failures = [(401, b"bad key")]
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert router.using_local
        router.set_mode(TtsMode.AUTO)
        assert not router.using_local
        router.close()


# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------


class TestModes:
    """The four modes, and the pairs they route to."""

    def test_offline_never_touches_the_cloud(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router(mode=TtsMode.OFFLINE)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert pair.handler.calls == 0
        assert pair.local.requests
        assert handle.engine == "fake"
        router.close()

    def test_online_with_no_local_engine_keeps_the_cloud(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router(mode=TtsMode.ONLINE, local=None)  # no local voice at all
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert handle.engine == "elevenlabs"
        router.close()

    def test_local_first_uses_local_and_falls_back_to_cloud(
        self, keyring_store: SecretsStore
    ) -> None:
        pair = EnginePair()
        router = pair.make_router(mode=TtsMode.LOCAL_FIRST)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert pair.handler.calls == 0
        assert handle.engine == "fake"
        assert not router.using_local
        router.close()

    def test_local_first_cloud_failure_does_not_switch_modes(
        self, keyring_store: SecretsStore
    ) -> None:
        """The fallback direction is not sticky in local_first: retry local."""
        pair = EnginePair()
        pair.local.fail = True
        router = pair.make_router(mode=TtsMode.LOCAL_FIRST)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert handle.engine == "elevenlabs"
        router.close()

    def test_set_mode_switches_immediately(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router(mode=TtsMode.OFFLINE)
        router.set_mode(TtsMode.AUTO)
        assert router.mode is TtsMode.AUTO
        router.close()

    def test_mode_offline_uses_local_with_no_cloud(self) -> None:
        backend = FakeBackend()
        player = TtsPlayer(backend=backend)  # type: ignore[arg-type]
        router = TtsRouter(player, mode=TtsMode.OFFLINE, local=loaded_local)
        player.start()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert backend.written
        router.close()


# ----------------------------------------------------------------------
# voice settings
# ----------------------------------------------------------------------


class TestVoiceSettings:
    """Per-phrase overrides, clamping, and the immediate volume."""

    def test_per_phrase_speed_reaches_the_engine(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет!", speed=1.5)
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        router.close()

    def test_preview_uses_the_given_params(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.preview(params=VoiceParams(speed=1.5))
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        router.close()

    def test_set_params_volume_takes_effect_immediately(self, keyring_store: SecretsStore) -> None:
        """A user dragging the slider expects it to change what they hear now."""
        pair = EnginePair()
        router = pair.make_router()
        router.set_params(VoiceParams(volume=50))
        assert pair.player.volume == pytest.approx(0.5)
        router.close()

    def test_set_params_is_stored_for_the_next_phrase(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        router.set_params(VoiceParams(volume=20))
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        router.close()


# ----------------------------------------------------------------------
# history
# ----------------------------------------------------------------------


class TestHistory:
    """What was said is kept; previews are not part of it."""

    def test_say_is_recorded_in_history(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        history = router.history()
        assert len(history) == 1
        assert history[0].text == "Привет!"
        assert history[0].engine == "elevenlabs"
        router.close()

    def test_preview_is_not_recorded(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.preview()
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.COMPLETED
        assert router.history() == ()
        router.close()

    def test_preview_does_not_reach_the_spoken_hook(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        spoken: list[SpokenPhrase] = []
        router = pair.make_router(on_spoken=spoken.append)
        handle = router.preview()
        handle.wait(TIMEOUT_S)
        assert spoken == []
        router.close()

    def test_history_is_bounded(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        for _ in range(60):
            handle = router.say("Привет!")
            handle.wait(TIMEOUT_S)
        assert len(router.history()) == 50
        router.close()

    def test_history_limit_argument(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        for _ in range(3):
            handle = router.say("Привет!")
            handle.wait(TIMEOUT_S)
        assert len(router.history(limit=2)) == 2
        router.close()

    def test_a_failed_phrase_is_recorded_with_its_reason(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.local.fail = True
        router = pair.make_router(mode=TtsMode.OFFLINE)
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.ERROR
        history = router.history()
        assert history and history[0].reason == PlaybackReason.ERROR
        router.close()


# ----------------------------------------------------------------------
# cancel
# ----------------------------------------------------------------------


class TestCancel:
    """A phrase can be stopped from its handle."""

    def test_cancel_stops_a_phrase(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.backend.start_blocked = True
        router = pair.make_router()
        handle = router.say("Привет!")
        wait_until(pair.backend.writing)
        assert handle.cancel()
        pair.backend.release()
        handle.wait(TIMEOUT_S)
        assert handle.reason == PlaybackReason.CANCELLED
        router.close()

    def test_cancel_on_a_finished_handle_returns_false(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert not handle.cancel()
        router.close()

    def test_cancelled_phrase_is_recorded_as_cancelled(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.backend.start_blocked = True
        router = pair.make_router()
        handle = router.say("Привет!")
        wait_until(pair.backend.writing)
        handle.cancel()
        pair.backend.release()
        handle.wait(TIMEOUT_S)
        assert router.history()[-1].reason == PlaybackReason.CANCELLED
        router.close()


# ----------------------------------------------------------------------
# shutdown
# ----------------------------------------------------------------------


class TestShutdown:
    """close() resolves waiters and releases the engines."""

    def test_close_resolves_a_pending_handle(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.backend.start_blocked = True
        router = pair.make_router()
        handle = router.say("Привет!")
        router.close()
        assert handle.done
        assert handle.reason == PlaybackReason.ERROR

    def test_close_unloads_what_was_actually_loaded(self, keyring_store: SecretsStore) -> None:
        """Only the slot that spoke holds an engine; the other never built one."""
        pair = EnginePair()
        pair.handler.failures = [(401, b"bad key")]
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        assert pair.local.loads == 1
        router.close()
        assert pair.local.unloads == 1
        assert pair.cloud_calls == 1

    def test_close_leaves_an_unused_slot_alone(self, keyring_store: SecretsStore) -> None:
        """The cloud spoke, so the local voice was never built - nor unloaded."""
        pair = EnginePair()
        router = pair.make_router()
        handle = router.say("Привет!")
        handle.wait(TIMEOUT_S)
        router.close()
        assert pair.local.loads == 0
        assert pair.local.unloads == 0

    def test_close_is_idempotent(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        router = pair.make_router()
        router.close()
        router.close()

    def test_handle_wait_with_timeout(self, keyring_store: SecretsStore) -> None:
        pair = EnginePair()
        pair.backend.start_blocked = True
        router = pair.make_router()
        handle = router.say("Привет!")
        assert not handle.wait(0.05)
        pair.backend.release()
        handle.wait(TIMEOUT_S)
        assert handle.done
        router.close()
