"""One ``say()`` for the whole assistant, and the fallback behind it.

Everything that speaks - the NLU reply, an action's confirmation, a macro, a
plugin, a system notification - calls :meth:`TtsRouter.say` and never learns
which engine produced the sound. That is the point of the module: the choice
between a local voice and a paid one, the reaction to the network going away,
the character counter and the volume slider all live here, once, instead of in
every caller.

**The fallback is per sentence, and it never speaks anything twice.** The
generator handed to :class:`~ayris.audio.tts.player.SpeechRequest` is consumed
lazily by the player, so a cloud failure is caught while the *next* sentence is
being synthesized. If the failing sentence had not produced a single chunk yet,
it is re-synthesized locally and the user hears one clean phrase. If it had
already put audio on the device, the sentence is abandoned where it broke and
the *rest* of the answer comes from the local engine - repeating the first half
of a sentence would be a worse artefact than losing the second half of one.

**Offline costs nothing.** With a
:class:`~ayris.core.connectivity.ConnectivityMonitor` that already knows the
network is down, the cloud is not tried at all: the user gets a local voice
immediately instead of a connect timeout of silence. After a failure the router
stays local until :class:`~ayris.core.events.OnlineStatusChanged` says the
connection is back, so a dead network costs one failed request rather than one
per phrase.

**The voice settings are the same object for both.** :class:`VoiceParams` holds
voice, speed, pitch and volume in Ayris's own units, clamps them on
construction, and each engine converts them to whatever its provider counts in -
see :mod:`ayris.audio.tts.cloud_base`. Volume is applied by the player rather
than by the engines, which is the only way the slider can mean the same thing
for a local voice and a cloud one.

**Preview is deliberately not history.** :meth:`preview` speaks a fixed short
phrase for the settings window and skips the ``on_spoken`` hook, so trying out
six voices does not leave six entries in what the assistant said.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Final
from uuid import uuid4

from ayris.audio.tts.base import DEFAULT_PITCH, DEFAULT_SPEED, clamp_pitch, clamp_speed
from ayris.audio.tts.cloud_base import TtsAuthError, TtsQuotaError, is_cloud_engine
from ayris.audio.tts.player import PlaybackReason, SpeechRequest
from ayris.audio.tts.sentence_split import is_speakable, split_sentences
from ayris.core.errors import TtsError
from ayris.core.events import NotificationRequested, OnlineStatusChanged, TtsFinished
from ayris.core.models import utc_now
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

    from ayris.audio.tts.base import AudioChunk, TtsEngine, VoiceSpec
    from ayris.audio.tts.player import TtsPlayer
    from ayris.core.connectivity import ConnectivityMonitor
    from ayris.core.events import EventBus, Unsubscribe

    #: Hands back a *loaded* engine, loading it on the first call. The router
    #: never constructs one itself: a cloud engine needs a credential and a local
    #: one needs a model path and a RAM check, and neither belongs here.
    EngineProvider = Callable[[], TtsEngine]

__all__ = [
    "PREVIEW_TEXT",
    "SpeechHandle",
    "SpokenPhrase",
    "TtsMode",
    "TtsRouter",
    "VoiceParams",
    "mode_from_config",
]

_log = get_logger(__name__)

#: What :meth:`TtsRouter.preview` says. Short enough that a user comparing voices
#: is not waiting through it, and long enough to hear intonation rather than one
#: syllable. Contains an exclamation and a comma on purpose: that is where the
#: difference between a good voice and a flat one is audible.
PREVIEW_TEXT: Final = "Привет! Я Айрис, ваш голосовой помощник."

#: Kept in memory for DevTools and for «повтори». Bounded because this is the
#: text of everything the assistant said, and it belongs in RAM rather than
#: growing without limit.
_HISTORY_LIMIT: Final = 50

#: Seconds a background preload may hold the lock before it is considered stuck.
#: Only used for the log line - nothing is killed.
_PRELOAD_WARN_SEC: Final = 30.0


class TtsMode(StrEnum):
    """Which engine speaks first, and what happens when it will not.

    Attributes:
        OFFLINE: Local voice only. Nothing here touches the network.
        ONLINE: Cloud only. A failure is reported as a failure rather than
            quietly answered in a different voice.
        AUTO: Cloud first, local voice the moment the cloud does not work.
        LOCAL_FIRST: Local voice first, cloud when it fails. This is what
            ``voice.tts.cloud_fallback`` means - the local engine is the one the
            user chose, and the paid service is the safety net rather than the
            default.
    """

    OFFLINE = "offline"
    ONLINE = "online"
    AUTO = "auto"
    LOCAL_FIRST = "local_first"

    @classmethod
    def parse(cls, value: str) -> TtsMode:
        """Read a mode out of the settings, defaulting to :attr:`AUTO`.

        An unknown string becomes auto rather than an error: it arrives from a
        config file a newer Ayris may have written, and going mute would be a
        worse answer than picking the mode that works in both situations.
        """
        try:
            return cls(value.strip().lower().replace("-", "_"))
        except ValueError:
            _log.warning("tts: неизвестный режим синтеза %r, работаю в auto", value)
            return cls.AUTO


def mode_from_config(engine: str, *, cloud_fallback: bool = False) -> TtsMode:
    """Turn ``voice.tts.engine`` and ``cloud_fallback`` into a mode.

    The settings do not have a mode field for speech the way they do for
    recognition; they have the engine the user picked and one checkbox. Both
    directions of fallback follow from those two:

    Args:
        engine: Engine name from the settings.
        cloud_fallback: ``voice.tts.cloud_fallback`` - speak through the cloud
            when the local engine fails.

    Returns:
        :attr:`TtsMode.AUTO` for a cloud engine (the cloud is the choice, local
        is the safety net), :attr:`TtsMode.LOCAL_FIRST` for a local engine with
        the checkbox on, and :attr:`TtsMode.OFFLINE` otherwise.
    """
    if is_cloud_engine(engine):
        return TtsMode.AUTO
    return TtsMode.LOCAL_FIRST if cloud_fallback else TtsMode.OFFLINE


@dataclass(frozen=True, slots=True)
class VoiceParams:
    """Voice settings in Ayris's units, valid by construction.

    The same object is passed to a Piper voice and to ElevenLabs; each engine
    converts it to its own scale. Clamping happens in ``__post_init__`` rather
    than in the settings window, so a hand-edited ``config.toml`` or a plugin
    calling :meth:`TtsRouter.say` cannot ask for a speed no engine accepts.

    Attributes:
        voice: Which voice to speak in. ``None`` means whatever the engine was
            loaded with, which is the normal case - the engine provider already
            applied the configured voice.
        speed: Rate multiplier, 0.5–2.0. 1.0 is the voice's own tempo.
        pitch: Pitch multiplier, 0.5–2.0. Engines without pitch control ignore
            it; the settings window does not have to know which those are.
        volume: 0–100, applied by the player. Not the system mixer: changing
            that would follow the user into their next application.
    """

    voice: VoiceSpec | None = None
    speed: float = DEFAULT_SPEED
    pitch: float = DEFAULT_PITCH
    volume: int = 80

    def __post_init__(self) -> None:
        """Clamp every field into range. Frozen, so written the long way."""
        object.__setattr__(self, "speed", clamp_speed(self.speed))
        object.__setattr__(self, "pitch", clamp_pitch(self.pitch))
        object.__setattr__(self, "volume", max(0, min(100, int(self.volume))))

    @property
    def gain(self) -> float:
        """:attr:`volume` as the 0.0–1.0 the player takes."""
        return self.volume / 100.0

    def merged(
        self,
        *,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        pitch: float | None = None,
    ) -> VoiceParams:
        """A copy with the given overrides applied, re-clamped.

        ``None`` keeps the current value, which is what makes
        ``say(text, speed=None)`` mean "as configured" rather than "at 0".
        """
        if voice is None and speed is None and pitch is None:
            return self
        return replace(
            self,
            voice=self.voice if voice is None else voice,
            speed=self.speed if speed is None else speed,
            pitch=self.pitch if pitch is None else pitch,
        )


@dataclass(frozen=True, slots=True)
class SpokenPhrase:
    """One thing the assistant said, for the history hook and DevTools.

    Attributes:
        text: What was spoken.
        engine: Engine that produced the audio, as in ``piper`` or ``azure``.
        request_id: Correlates with :class:`~ayris.core.events.TtsFinished`.
        reason: A :class:`~ayris.audio.tts.player.PlaybackReason` value - a
            phrase the user interrupted is part of the history too, and the
            difference matters when reading it back.
        at: When it finished.
    """

    text: str
    engine: str = ""
    request_id: str = ""
    reason: str = PlaybackReason.COMPLETED
    at: datetime | None = None


class SpeechHandle:
    """What :meth:`TtsRouter.say` hands back: cancel it, or wait for it.

    Returned already finished when there was nothing speakable in the text, so a
    caller can always ``wait()`` without checking.

    Waiting is only safe off the UI thread. With an event bus the completion
    arrives as :class:`~ayris.core.events.TtsFinished`, which the bus delivers on
    the UI thread - waiting there would block the very thread that has to
    deliver it.
    """

    __slots__ = ("_done", "_engine", "_player", "_preview", "_reason", "_request_id", "_text")

    def __init__(
        self,
        request_id: str,
        text: str,
        player: TtsPlayer | None = None,
        *,
        preview: bool = False,
    ) -> None:
        self._request_id = request_id
        self._text = text
        self._player = player
        self._preview = preview
        self._engine = ""
        self._reason = ""
        self._done = threading.Event()

    @property
    def request_id(self) -> str:
        """Identifier this phrase carries through the events and the log."""
        return self._request_id

    @property
    def text(self) -> str:
        """What was asked for, before splitting."""
        return self._text

    @property
    def engine(self) -> str:
        """Engine that spoke, empty until synthesis has started.

        After a fallback this is the engine that spoke *last* - the one the user
        heard the end of the phrase in.
        """
        return self._engine

    @property
    def preview(self) -> bool:
        """Whether this came from :meth:`TtsRouter.preview`."""
        return self._preview

    @property
    def done(self) -> bool:
        """Whether the phrase has stopped, for any reason."""
        return self._done.is_set()

    @property
    def reason(self) -> str:
        """Why it stopped: a :class:`~ayris.audio.tts.player.PlaybackReason`.

        Empty while it is still speaking.
        """
        return self._reason

    def cancel(self) -> bool:
        """Stop this phrase, whether it is sounding or still queued.

        Cancels only this phrase: a later one that has already replaced it keeps
        speaking, which is what an action wanting to retract *its own* answer
        needs.

        Returns:
            Whether anything was actually stopped or dropped.
        """
        if self._player is None or self._done.is_set():
            return False
        return self._player.cancel(self._request_id)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the phrase stops.

        Args:
            timeout: Seconds to wait, ``None`` for forever.

        Returns:
            Whether it finished within the timeout.
        """
        return self._done.wait(timeout)

    def __repr__(self) -> str:
        state = self._reason or "speaking"
        return f"SpeechHandle({self._request_id!r}, engine={self._engine!r}, {state})"

    # ------------------------------------------------------------- internals

    def _note_engine(self, engine: str) -> None:
        """Record which engine produced the audio. Called by the router."""
        self._engine = engine

    def _resolve(self, reason: str) -> bool:
        """Mark the phrase finished. Idempotent; returns whether this call did it."""
        if self._done.is_set():
            return False
        self._reason = reason
        self._done.set()
        return True


class _Slot:
    """One engine the router may use, and how to get hold of it.

    The provider is called at most once; what it returns is kept, because
    loading a Piper voice or opening an HTTPS client is not something to redo
    per sentence.
    """

    __slots__ = ("_engine", "_provider", "cloud")

    def __init__(self, provider: EngineProvider | None, *, cloud: bool) -> None:
        self._provider = provider
        self._engine: TtsEngine | None = None
        self.cloud = cloud

    @property
    def configured(self) -> bool:
        """Whether there is anything to load at all."""
        return self._provider is not None

    @property
    def ready(self) -> bool:
        """Whether the engine is already loaded."""
        return self._engine is not None

    @property
    def label(self) -> str:
        """How to name this engine to the user."""
        return "облачный" if self.cloud else "локальный"

    def get(self) -> TtsEngine:
        """The loaded engine.

        Raises:
            TtsError: Nothing is configured for this slot, or the load failed -
                a missing key or a missing model being the usual reasons.
        """
        engine = self._engine
        if engine is not None:
            return engine
        provider = self._provider
        if provider is None:
            raise TtsError(
                f"tts router: no {'cloud' if self.cloud else 'local'} engine is configured",
                user_message=(
                    "Облачный синтез речи не настроен. "
                    "Выберите сервис и сохраните ключ в настройках голоса."
                    if self.cloud
                    else "Локальный голос не установлен. "
                    "Скачайте голос в настройках, чтобы помощник говорил без интернета."
                ),
            )
        engine = provider()
        self._engine = engine
        return engine

    def close(self) -> None:
        """Release the engine, if one was loaded. Never raises."""
        engine, self._engine = self._engine, None
        if engine is None:
            return
        try:
            engine.unload()
        except Exception as exc:  # pragma: no cover - unload must not raise
            _log.debug("tts: не удалось выгрузить движок: %s", exc)


class TtsRouter:
    """Chooses an engine, speaks through the player, and survives the network.

    Args:
        player: Where the audio goes. The router never opens a device itself.
        mode: Which engine speaks first - see :class:`TtsMode`.
        cloud: Provider for the cloud engine. ``None`` makes online and auto
            behave as offline.
        local: Provider for the local engine. ``None`` leaves auto with nothing
            to fall back to, which is said out loud rather than silently.
        params: Voice, speed, pitch and volume. Applied to every phrase.
        monitor: Connectivity state. The router reports cloud failures to it and
            listens for its recovery event.
        bus: Where notifications go and where completions are read from. **With a
            bus, :func:`~ayris.audio.tts.service.connect_player` must be wired
            up** - the router learns that a phrase finished from
            :class:`~ayris.core.events.TtsFinished`, which the bridge publishes.
            Without a bus it observes the player directly instead.
        on_spoken: Called with every finished phrase except previews. This is the
            history hook: whoever stores what the assistant said passes it here.

    Thread-safe. ``say`` is called from the pipeline, the settings window calls
    ``preview`` and ``set_params``, and the synthesis itself runs on the player's
    writer thread.
    """

    __slots__ = (
        # Explicit, because __slots__ otherwise removes __weakref__ and the bus
        # holds a bound-method subscription weakly: without it, subscribing
        # _on_online_status raises and the router cannot be built with a bus.
        "__weakref__",
        "_bus",
        "_cloud",
        "_history",
        "_local",
        "_lock",
        "_mode",
        "_monitor",
        "_on_spoken",
        "_params",
        "_pending",
        "_player",
        "_preload",
        "_use_secondary",
        "_wired",
    )

    def __init__(
        self,
        player: TtsPlayer,
        *,
        mode: TtsMode = TtsMode.AUTO,
        cloud: EngineProvider | None = None,
        local: EngineProvider | None = None,
        params: VoiceParams | None = None,
        monitor: ConnectivityMonitor | None = None,
        bus: EventBus | None = None,
        on_spoken: Callable[[SpokenPhrase], None] | None = None,
    ) -> None:
        self._player = player
        self._mode = mode
        self._cloud = _Slot(cloud, cloud=True)
        self._local = _Slot(local, cloud=False)
        self._params = params or VoiceParams()
        self._monitor = monitor
        self._bus = bus
        self._on_spoken = on_spoken
        self._lock = threading.RLock()
        self._pending: dict[str, SpeechHandle] = {}
        self._history: deque[SpokenPhrase] = deque(maxlen=_HISTORY_LIMIT)
        self._use_secondary = False
        self._preload: threading.Thread | None = None
        self._wired: tuple[Unsubscribe, ...] = ()
        if bus is not None:
            self._wired = (
                bus.subscribe(TtsFinished, self._on_tts_finished),
                bus.subscribe(OnlineStatusChanged, self._on_online_status),
            )
        else:
            # No bus means no bridge, so nothing else owns the player's
            # callbacks and the router can read completions straight off it.
            player.set_observers(on_finished=self._on_player_finished)

    # ------------------------------------------------------------- properties

    @property
    def mode(self) -> TtsMode:
        """The configured mode."""
        return self._mode

    @property
    def params(self) -> VoiceParams:
        """The voice settings in force."""
        return self._params

    @property
    def using_local(self) -> bool:
        """Whether a cloud failure has pushed the router onto the local voice."""
        with self._lock:
            return self._use_secondary

    # ---------------------------------------------------------------- control

    def set_mode(self, mode: TtsMode) -> None:
        """Switch modes. Takes effect from the next phrase.

        Clears the fallback state: a user who has just picked a mode is asking
        for it to be tried, not for a decision made under the old one.
        """
        with self._lock:
            self._mode = mode
            self._use_secondary = False

    def set_params(self, params: VoiceParams) -> None:
        """Replace the voice settings.

        Speed and pitch apply from the next phrase - changing them mid-sentence
        would be audible as a glitch. Volume applies immediately, because that is
        what a user dragging the slider while listening expects.
        """
        with self._lock:
            self._params = params
        self._player.set_volume(params.gain)

    def preload(self) -> None:
        """Warm the fallback engine in the background so a switch is instant.

        Only useful in auto mode: the other modes load what they need on the
        first phrase anyway. Idempotent, and a failure is logged rather than
        raised - a preload that did not work costs latency on the first
        fallback, not correctness.
        """
        if self._mode is not TtsMode.AUTO or not self._local.configured:
            return
        with self._lock:
            if self._local.ready:
                return
            if self._preload is not None and self._preload.is_alive():
                return
            self._preload = threading.Thread(
                target=self._preload_local, name="ayris-tts-preload", daemon=True
            )
            self._preload.start()

    def close(self) -> None:
        """Drop the subscriptions, release the engines, release the waiters.

        Anything still queued is resolved as an error rather than left waiting:
        a caller blocked in :meth:`SpeechHandle.wait` during shutdown would
        otherwise never wake up.
        """
        wired, self._wired = self._wired, ()
        for unsubscribe in wired:
            unsubscribe()
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for handle in pending:
            handle._resolve(PlaybackReason.ERROR)
        self._cloud.close()
        self._local.close()

    # ------------------------------------------------------------------ speech

    def say(
        self,
        text: str,
        *,
        voice: VoiceSpec | None = None,
        speed: float | None = None,
        priority: bool | None = None,
    ) -> SpeechHandle:
        """Speak one answer. The only way anything in Ayris makes sound.

        Returns immediately: synthesis happens on the player's thread as the
        audio is consumed, so a caller is never blocked by a slow provider.

        Args:
            text: What to say. Split into sentences internally; an empty or
                unspeakable text is a finished handle rather than an error.
            voice: Speak in this voice instead of the configured one.
            speed: Rate for this phrase only, 0.5–2.0, clamped.
            priority: Jump the queue. Urgent phrases never interrupt what is
                already sounding - cutting a sentence in half to say «Готово»
                reads as a glitch.

        Returns:
            A handle that can be cancelled and waited on.

        Raises:
            TtsError: No engine is configured for the current mode at all. A
                failure of an engine that *is* configured is not raised here -
                it becomes a fallback, or a
                :class:`~ayris.core.events.TtsFinished` with an error reason.
        """
        return self._enqueue(
            text,
            params=self._params.merged(voice=voice, speed=speed),
            priority=bool(priority),
            preview=False,
        )

    def preview(
        self,
        voice: VoiceSpec | None = None,
        params: VoiceParams | None = None,
        *,
        text: str = PREVIEW_TEXT,
    ) -> SpeechHandle:
        """Speak a short fixed phrase so the settings window can be listened to.

        Identical to :meth:`say` in every way that affects the sound - same
        routing, same fallback, same conversion to the provider's scales - and
        different in two ways that do not: it is urgent, so it is heard now
        rather than after the answer in progress, and it is **not recorded**.
        Trying six voices must not leave six entries in what the assistant said.

        Args:
            voice: Voice to try. ``None`` uses the one in ``params``.
            params: Settings to try, unsaved. ``None`` uses the current ones.
            text: Phrase to speak. Defaults to :data:`PREVIEW_TEXT`.

        Returns:
            A handle, so the settings window can stop a preview when the user
            immediately clicks another voice.
        """
        wanted = params or self._params
        return self._enqueue(
            text,
            params=wanted.merged(voice=voice),
            priority=True,
            preview=True,
        )

    def history(self, limit: int = 0) -> tuple[SpokenPhrase, ...]:
        """What has been said recently, newest last. Previews are not in it."""
        with self._lock:
            items = tuple(self._history)
        return items[-limit:] if limit > 0 else items

    # ----------------------------------------------------------------- routing

    def _enqueue(
        self,
        text: str,
        *,
        params: VoiceParams,
        priority: bool,
        preview: bool,
    ) -> SpeechHandle:
        """Build the request, hand the player a generator, and return the handle."""
        request_id = uuid4().hex
        handle = SpeechHandle(request_id, text, self._player, preview=preview)
        if not is_speakable(text):
            # «...», a bare emoji, an empty answer: a normal outcome, not an
            # error, and not worth waking the player for.
            handle._resolve(PlaybackReason.COMPLETED)
            return handle

        primary, _ = self._route()
        if primary is None:
            raise self._nothing_configured()

        request = SpeechRequest(
            text=text,
            request_id=request_id,
            engine="",
            priority=priority,
        )
        request.chunks = self._speech(request, handle, params)
        with self._lock:
            self._pending[request_id] = handle
        self._player.set_volume(params.gain)
        self._player.speak(request)
        return handle

    def _route(self) -> tuple[_Slot | None, _Slot | None]:
        """The engine to try and the one to fall back to, in that order.

        Offline is decided here rather than by letting the request time out:
        with a monitor that already knows the network is down, the cloud slot is
        skipped entirely and the user hears a local voice at once.

        Returns:
            ``(None, None)`` when nothing this mode could use is configured, so
            that :meth:`say` refuses in the caller's frame rather than inside the
            generator - where the failure would reach the user as an unexplained
            silence instead of as an error with a sentence in it.
        """
        primary, secondary = self._pair()
        if secondary is not None and not secondary.configured:
            secondary = None
        if primary is not None and not primary.configured:
            primary, secondary = secondary, None
        if primary is not None and secondary is not None and self._skip_primary(primary):
            return secondary, None
        return primary, secondary

    def _pair(self) -> tuple[_Slot | None, _Slot | None]:
        """The two slots the current mode uses, before any state is applied."""
        if self._mode is TtsMode.OFFLINE:
            return self._local, None
        if self._mode is TtsMode.ONLINE:
            return self._cloud, None
        if self._mode is TtsMode.LOCAL_FIRST:
            return self._local, self._cloud
        return self._cloud, self._local

    def _skip_primary(self, primary: _Slot | None) -> bool:
        """Whether to go straight to the fallback for this phrase.

        Only ever true for a cloud primary. The reverse - sticking to the cloud
        because the local engine failed once - would leave a user paying per
        character for a hiccup they never heard about, so ``local_first`` retries
        the local voice on every phrase instead.
        """
        if primary is None or not primary.cloud:
            return False
        with self._lock:
            if self._use_secondary:
                return True
        return self._monitor is not None and not self._monitor.online

    def _nothing_configured(self) -> TtsError:
        """The error for a mode with no engine behind it at all."""
        return TtsError(
            f"tts router: mode {self._mode} has no engine configured",
            user_message=(
                "Синтез речи не настроен: нет ни локального голоса, ни облачного сервиса. "
                "Выберите голос в настройках."
            ),
        )

    # -------------------------------------------------------------- synthesis

    def _speech(
        self,
        request: SpeechRequest,
        handle: SpeechHandle,
        params: VoiceParams,
    ) -> Iterator[AudioChunk]:
        """Yield the whole answer, sentence by sentence.

        Consumed by the player's writer thread, one chunk at a time. Everything
        below this line therefore runs *while the previous chunk is sounding*,
        which is what makes a fallback cheap enough to do mid-answer.
        """
        for sentence in split_sentences(request.text):
            yield from self._sentence(request, handle, sentence, params)

    def _sentence(
        self,
        request: SpeechRequest,
        handle: SpeechHandle,
        sentence: str,
        params: VoiceParams,
    ) -> Iterator[AudioChunk]:
        """One sentence, with the fallback that must not duplicate audio.

        Raises:
            TtsError: The sentence could not be spoken at all. The player turns
                that into :attr:`~ayris.audio.tts.player.PlaybackReason.ERROR`.
        """
        primary, secondary = self._route()
        if primary is None:
            raise self._nothing_configured()

        started = False
        try:
            for chunk in self._chunks(primary, request, handle, sentence, params):
                started = True
                yield chunk
        except TtsError as exc:
            if secondary is None:
                raise
            self._fall_back(exc, primary, secondary)
            if started:
                # Audio from this sentence is already on the device. Saying it
                # again in another voice is worse than losing its tail, so the
                # switch takes effect from the next sentence.
                _log.warning(
                    "tts: %s замолчал посреди фразы, остаток скажет %s",
                    primary.label,
                    secondary.label,
                )
                return
            yield from self._chunks(secondary, request, handle, sentence, params)

    def _chunks(
        self,
        slot: _Slot,
        request: SpeechRequest,
        handle: SpeechHandle,
        sentence: str,
        params: VoiceParams,
    ) -> Iterator[AudioChunk]:
        """Audio for one sentence from one engine, streamed if it can be.

        Raises:
            TtsError: The engine could not be loaded, or synthesis failed.
        """
        with self._lock:
            engine = slot.get()
        self._note_engine(request, handle, engine)
        yield from engine.synthesize_stream(sentence, params.voice, params.speed, params.pitch)
        if slot.cloud:
            self._note_success()

    def _note_engine(self, request: SpeechRequest, handle: SpeechHandle, engine: TtsEngine) -> None:
        """Put the engine that is actually speaking on the request and the handle.

        The request is mutated before its first chunk is yielded, so
        :class:`~ayris.core.events.TtsStarted` names the engine the user is
        hearing rather than the one that was asked first - which is the whole
        value of that field to the pipeline log.
        """
        if request.engine != engine.name:
            request.engine = engine.name
        handle._note_engine(engine.name)

    def _fall_back(self, exc: TtsError, primary: _Slot, secondary: _Slot) -> None:
        """Remember the failure, tell the monitor, and tell the user once."""
        if primary.cloud:
            with self._lock:
                first = not self._use_secondary
                self._use_secondary = True
            if self._monitor is not None and not isinstance(exc, TtsQuotaError | TtsAuthError):
                self._monitor.report_failure(type(exc).__name__)
        else:
            first = True

        _log.warning(
            "tts: %s движок отказал (%s), переключаюсь на %s",
            primary.label,
            exc.technical,
            secondary.label,
        )
        if not first:
            return
        self._notify(
            "Голос переключён",
            f"{exc.user_message} Говорю {secondary.label}м голосом"
            + (" — вернусь в облако, когда связь восстановится." if primary.cloud else "."),
            level="warning",
        )

    def _note_success(self) -> None:
        """A cloud sentence came back: the network is up, whatever we thought."""
        if self._monitor is not None:
            self._monitor.report_success()

    def _preload_local(self) -> None:
        """Thread body for :meth:`preload`. Never raises out of the thread."""
        started = utc_now()
        try:
            with self._lock:
                self._local.get()
        except TtsError as exc:
            _log.info("tts: прогрев локального голоса пропущен: %s", exc.technical)
        except Exception:  # pragma: no cover - a bug, not a usage error
            _log.exception("tts: прогрев локального голоса упал")
        elapsed = (utc_now() - started).total_seconds()
        if elapsed >= _PRELOAD_WARN_SEC:
            _log.warning("tts: прогрев локального голоса занял %.0f с", elapsed)

    # ----------------------------------------------------------- completion

    def _on_tts_finished(self, event: TtsFinished) -> None:
        """Bus path: the bridge says a phrase stopped."""
        self._finish(event.request_id, event.reason)

    def _on_player_finished(self, request: SpeechRequest, reason: str) -> None:
        """Observer path, used when there is no bus to carry the event."""
        self._finish(request.request_id, reason)

    def _finish(self, request_id: str, reason: str) -> None:
        """Resolve the handle and record the phrase. Idempotent.

        Called for every phrase the player finishes, including ones this router
        never queued - a worker replaying a cached answer, say - which is why an
        unknown identifier is simply ignored.
        """
        if not request_id:
            return
        with self._lock:
            handle = self._pending.pop(request_id, None)
        if handle is None or not handle._resolve(reason):
            return
        if handle.preview:
            return
        phrase = SpokenPhrase(
            text=handle.text,
            engine=handle.engine,
            request_id=request_id,
            reason=reason,
            at=utc_now(),
        )
        with self._lock:
            self._history.append(phrase)
        self._report(phrase)

    def _report(self, phrase: SpokenPhrase) -> None:
        """Hand the phrase to the history hook, if there is one."""
        hook = self._on_spoken
        if hook is None:
            return
        try:
            hook(phrase)
        except Exception:
            # A history writer that fails must not be able to stop the assistant
            # from speaking the next thing.
            _log.exception("tts: не удалось записать произнесённую фразу")

    # -------------------------------------------------------------- events

    def _on_online_status(self, event: OnlineStatusChanged) -> None:
        """Return to the cloud voice when connectivity comes back."""
        if not event.online:
            return
        with self._lock:
            if not self._use_secondary:
                return
            self._use_secondary = False
        _log.info("tts: связь восстановлена (%s), следующая фраза уйдёт в облако", event.detail)

    def _notify(self, title: str, message: str, *, level: str) -> None:
        """Put a notification on the bus, if there is a bus."""
        bus = self._bus
        if bus is None:
            return
        bus.publish(
            NotificationRequested(title=title, message=message, level=level, timeout_ms=6000)
        )
