"""Bridge the pipeline's speech protocols onto the workers (task 47 input seam).

Task 47 mounted the pipeline's *output* — a matched command's answer is spoken —
and left its *input* deliberately open: :class:`~ayris.core.pipeline.Pipeline`
declares the :class:`~ayris.core.pipeline.SttSource` and
:class:`~ayris.core.pipeline.PhraseSource` protocols and drives itself off
wake-word and speech-ended events through
:meth:`~ayris.core.pipeline.Pipeline.attach`, but nothing adapted those protocols
onto the running multiprocess workers, so a phrase spoken into the microphone
never reached recognition. This module is that adapter;
:func:`~ayris.core.pipeline_app.install_pipeline` is where it is wired in.

Two seams, two threads. :class:`WorkerPhraseSource` is called synchronously from
:meth:`~ayris.core.pipeline.Pipeline._on_speech_ended`, which runs on the
event-delivery (UI) thread: it does one short, bounded IPC round-trip to pull the
just-finished phrase's PCM out of the audio worker and must never wedge the
interface, so a slow or dead worker degrades to «нет фразы» rather than a frozen
window. :class:`WorkerSttSource` is called from
:meth:`~ayris.core.pipeline.Pipeline._transcribe`, which the pipeline already runs
on its own detached session thread, so the blocking recognition call is safe
there and is allowed the worker's full configured timeout.

Neither adapter loads a model or opens audio hardware in this process — that is
the whole point of the worker split. They speak to the STT and audio workers
through :class:`~ayris.workers.manager.WorkerManager`, which marshals the PCM into
shared memory (``audio=`` on :meth:`~ayris.workers.manager.WorkerManager.call`)
and reconstructs a worker-side :class:`~ayris.core.errors.SttError` on this side
of the pipe. A genuine recognition failure (no model configured, a model that
would not load, a device that refused) is re-raised so the pipeline speaks its
Russian reason; a worker that is merely unavailable, still starting, timed out or
crashed degrades to an empty transcript, which the pipeline reports as «не
расслышала» and keeps listening.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from time import monotonic, perf_counter, sleep
from typing import TYPE_CHECKING, Any, Protocol

from ayris.audio.stt.base import STT_SAMPLE_RATE, AudioBuffer, TranscriptResult
from ayris.core.errors import SttError
from ayris.workers.protocol import WorkerError
from ayris.workers.registry import WorkerKind

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["WorkerCaller", "WorkerPhraseSource", "WorkerSttSource", "make_transcribe_once"]

_log = logging.getLogger("ayris.core.worker_speech")

#: Ceiling on the phrase-fetch IPC, in seconds. The audio worker answers
#: :meth:`~ayris.workers.audio_worker.AudioWorker.segment` from a buffer it
#: already holds, so a healthy round-trip is a few milliseconds; the timeout only
#: exists so a wedged worker cannot freeze the UI thread that
#: :class:`WorkerPhraseSource` runs on. Kept well under a second for that reason —
#: a missed phrase is «не расслышала», a frozen window is a bug.
_PHRASE_TIMEOUT_SEC = 2.0

#: Ceiling on the activation-time model warm-up IPC, in seconds.
#: :meth:`~ayris.workers.stt_worker.SttWorker.load_model` starts the load on a
#: background thread and returns at once, so this round-trip is short; the
#: timeout only bounds a wedged worker so the fire-and-forget warm thread cannot
#: live forever.
_WARM_TIMEOUT_SEC = 2.0


class WorkerCaller(Protocol):
    """The one method the adapters need from the worker supervisor.

    A protocol rather than :class:`~ayris.workers.manager.WorkerManager` itself, so
    the adapters can be unit-tested with a hand-written stub that returns canned
    replies without spawning a worker process, loading a model or opening a
    microphone. :class:`~ayris.workers.manager.WorkerManager` satisfies it.
    """

    def call_sync(
        self,
        worker: str,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
        audio: bytes | None = None,
        sample_rate: int = STT_SAMPLE_RATE,
        channels: int = 1,
        sample_format: str = "int16",
    ) -> Any: ...


class WorkerSttSource:
    """Recognise a phrase by asking the STT worker; the pipeline's ``SttSource``.

    Runs on the pipeline's detached session thread (see
    :meth:`~ayris.core.pipeline.Pipeline._transcribe`), so the blocking
    :meth:`~ayris.workers.manager.WorkerManager.call_sync` is safe and uses the STT
    spec's own call timeout — a cold model that has to load first is allowed the
    seconds it needs. Autostart is left on so the first spoken command after a
    launch, or after the idle timer unloaded the model, warms the worker instead
    of failing.
    """

    __slots__ = ("_manager",)

    def __init__(self, manager: WorkerCaller) -> None:
        self._manager = manager

    def transcribe(self, audio: AudioBuffer) -> TranscriptResult:
        """Text for one phrase, or an empty result when the worker cannot answer.

        Raises:
            SttError: The worker reported a real recognition failure — no model
                configured, a model that would not load, a device that refused.
                Re-raised unchanged so the pipeline speaks its Russian
                ``user_message`` rather than a generic «не расслышала».
        """
        try:
            reply = self._manager.call_sync(
                WorkerKind.STT.value,
                "transcribe",
                audio=audio.pcm,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                sample_format="int16",
            )
        except SttError:
            # A recognition problem with a message worth hearing. The pipeline
            # turns it into speech; do not flatten it to silence.
            raise
        except (WorkerError, FutureTimeoutError) as exc:
            # Unavailable, still starting, timed out or crashed. None of that is
            # the user's phrase being unclear, but the graceful answer is the
            # same: report nothing heard and keep listening.
            _log.warning("STT-воркер недоступен, распознавание пропущено: %s", exc)
            return TranscriptResult.empty(engine=WorkerKind.STT.value)
        return TranscriptResult.from_params(reply if isinstance(reply, Mapping) else {})

    def warm(self) -> None:
        """Start loading the recognition model ahead of the phrase.

        Called when the assistant is activated (wake word or Push-to-Talk), so
        the STT worker's cold model load overlaps the seconds the user spends
        speaking instead of landing on the critical path at ``transcribe`` time.
        :meth:`~ayris.workers.stt_worker.SttWorker.load_model` returns as soon as
        the background load starts and is a no-op once the model is up, so this
        is cheap to call on every activation. Best-effort: a worker that is
        unavailable, still starting or timed out is ignored — the phrase still
        recognises, it just pays the cold load the old way, as before.
        """
        try:
            self._manager.call_sync(
                WorkerKind.STT.value, "load_model", {}, timeout=_WARM_TIMEOUT_SEC
            )
        except (SttError, WorkerError, FutureTimeoutError) as exc:
            _log.debug("STT-прогрев на активации пропущен: %s", exc)


class WorkerPhraseSource:
    """The last accepted phrase's PCM; the pipeline's ``PhraseSource``.

    Called on the event-delivery thread from
    :meth:`~ayris.core.pipeline.Pipeline._on_speech_ended`. The audio worker keeps
    the PCM of the phrase it just finished
    (:meth:`~ayris.workers.audio_worker.AudioWorker.segment`) instead of pushing it
    across the pipe on every utterance, so this is one small, bounded request. The
    default consumes the segment — one activation, one phrase — which is exactly
    what the pipeline wants.
    """

    __slots__ = ("_manager",)

    def __init__(self, manager: WorkerCaller) -> None:
        self._manager = manager

    def __call__(self) -> AudioBuffer | None:
        """PCM of the phrase, or ``None`` when there is nothing to recognise.

        ``None`` is a first-class answer — no phrase buffered, or the worker could
        not be reached in time — and the pipeline reads it as «не расслышала».
        Never raises: a failure here must not take down the event handler that
        drives the whole voice loop.
        """
        try:
            reply = self._manager.call_sync(
                WorkerKind.AUDIO.value, "segment", timeout=_PHRASE_TIMEOUT_SEC
            )
        except (WorkerError, FutureTimeoutError) as exc:
            _log.warning("аудио-воркер не отдал фразу: %s", exc)
            return None
        if not isinstance(reply, Mapping) or not reply.get("available"):
            return None
        pcm = reply.get("pcm", b"")
        if not isinstance(pcm, bytes | bytearray | memoryview) or not pcm:
            return None
        try:
            return AudioBuffer(
                pcm=bytes(pcm),
                sample_rate=int(reply.get("sample_rate", STT_SAMPLE_RATE)),
                channels=1,
            )
        except (SttError, ValueError, TypeError) as exc:
            # A malformed buffer (odd byte count, impossible rate) is the worker's
            # bug, not a phrase; drop it rather than crash the handler.
            _log.warning("аудио-воркер отдал некорректную фразу: %s", exc)
            return None


#: How long the «Тест распознавания» self-test waits for the user to speak a
#: phrase before giving up. The test is a deliberate "press, then say something"
#: action, so this is generous — a user reaching for the microphone should not
#: race a two-second clock — but bounded, so a silent room ends the test with a
#: message instead of leaving the «Записать фразу» button spinning forever.
_SELF_TEST_LISTEN_SEC = 8.0

#: Gap between segment polls while the self-test waits for a phrase. Short enough
#: that the recognised text feels immediate once the phrase closes, long enough
#: that polling is not a busy-loop on the IPC.
_SELF_TEST_POLL_SEC = 0.15

# __MAKE_TRANSCRIBE_ONCE__


def make_transcribe_once(
    manager: WorkerCaller,
    *,
    listen_sec: float = _SELF_TEST_LISTEN_SEC,
    poll_sec: float = _SELF_TEST_POLL_SEC,
) -> Callable[[], tuple[str, int]]:
    """Build the «Тест распознавания» self-test over the workers.

    The returned callable is what
    :attr:`~ayris.gui.tabs.voice.VoiceServices.transcribe_once` wants: it blocks
    (the tab runs it off the UI thread through
    :class:`~ayris.gui.tabs.voice.AsyncRunner`), records the next phrase the user
    speaks and returns its recognised text with the recognition time in
    milliseconds. It is built from the two input adapters above, so a healthy
    test drives the very same audio-worker → STT-worker path a real spoken
    command does.

    Three things keep it from reporting «не услышала» for a lost microphone or a
    mis-set gate — the two failures that made an earlier version look broken:

    *It checks the microphone first.* Before listening it asks the audio worker
    whether capture is actually running; a stopped or ``device_lost`` stream ends
    the test with «Микрофон недоступен», so a dead device is named as one instead
    of hiding behind «не услышала».

    *It drains the stale segment.* Whatever phrase was buffered before the button
    was pressed is discarded, so the test recognises what the user says now, not
    the click or an earlier sentence still sitting in the segment slot.

    *It falls back to the raw window.* If the segmenter's voice-activity gate
    accepts nothing within ``listen_sec`` — a gate set too high, a microphone too
    quiet — the last ``listen_sec`` of audio is read straight from the ring
    buffer and handed to the recogniser, whose own detection then decides. Only
    when that too comes back empty is «не услышала» the honest answer.

    Raises (from the returned callable):
        SttError: the microphone is not capturing, nothing was heard within
            ``listen_sec`` by either path, the recogniser returned an empty
            transcript, or the STT worker reported a real recognition failure.
            The message is Russian and the tab shows it under «Не удалось
            распознать».
    """
    phrase_source = WorkerPhraseSource(manager)
    stt = WorkerSttSource(manager)

    def ensure_capturing() -> None:
        """Fail with a device-specific message when the microphone is not live."""
        try:
            status = manager.call_sync(
                WorkerKind.AUDIO.value, "status", timeout=_PHRASE_TIMEOUT_SEC
            )
        except (WorkerError, FutureTimeoutError) as exc:
            raise SttError(
                f"self-test: audio capture status unavailable: {exc}",
                user_message=(
                    "Захват звука не запущен. " "Откройте вкладку «Голос» и попробуйте ещё раз."
                ),
            ) from exc
        if not (isinstance(status, Mapping) and status.get("running")):
            state = status.get("state") if isinstance(status, Mapping) else None
            raise SttError(
                f"self-test: audio capture is not running (state={state!r})",
                user_message=(
                    "Микрофон недоступен. " "Проверьте устройство записи в настройках «Голос»."
                ),
            )

    def read_window() -> AudioBuffer | None:
        """Last ``listen_sec`` of audio, when the VAD gate accepted no phrase."""
        try:
            reply = manager.call_sync(
                WorkerKind.AUDIO.value,
                "read",
                {"ms": listen_sec * 1000.0},
                timeout=_PHRASE_TIMEOUT_SEC,
            )
        except (WorkerError, FutureTimeoutError) as exc:
            _log.warning("self-test: не удалось прочитать окно аудио: %s", exc)
            return None
        if not isinstance(reply, Mapping):
            return None
        pcm = reply.get("pcm", b"")
        if not isinstance(pcm, bytes | bytearray | memoryview) or not pcm:
            return None
        try:
            return AudioBuffer(
                pcm=bytes(pcm),
                sample_rate=int(reply.get("sample_rate", STT_SAMPLE_RATE)),
                channels=1,
            )
        except (SttError, ValueError, TypeError) as exc:
            _log.warning("self-test: некорректное окно аудио: %s", exc)
            return None

    def transcribe_once() -> tuple[str, int]:
        ensure_capturing()
        # Drop whatever was buffered before the button was pressed.
        phrase_source()
        deadline = monotonic() + listen_sec
        buffer: AudioBuffer | None = None
        while monotonic() < deadline:
            buffer = phrase_source()
            if buffer is not None:
                break
            sleep(poll_sec)
        if buffer is None:
            # The gate accepted nothing — maybe set too high, maybe a quiet
            # microphone. Read the raw window and let the recogniser's own
            # detection decide before declaring the room silent.
            buffer = read_window()
        if buffer is None:
            raise SttError(
                "self-test: no phrase captured before the listen window closed",
                user_message="Не услышала фразу. Проверьте микрофон и попробуйте ещё раз.",
            )
        started = perf_counter()
        result = stt.transcribe(buffer)
        elapsed_ms = int((perf_counter() - started) * 1000)
        text = result.text.strip()
        if not text:
            raise SttError(
                "self-test: recogniser returned an empty transcript",
                user_message="Не разобрала слова. Скажите фразу чуть чётче и попробуйте ещё раз.",
            )
        return text, elapsed_ms

    return transcribe_once
