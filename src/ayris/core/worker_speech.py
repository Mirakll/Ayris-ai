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
from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any, Protocol

from ayris.audio.stt.base import STT_SAMPLE_RATE, AudioBuffer, TranscriptResult
from ayris.core.errors import SttError
from ayris.workers.protocol import WorkerError
from ayris.workers.registry import WorkerKind

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["WorkerCaller", "WorkerPhraseSource", "WorkerSttSource"]

_log = logging.getLogger("ayris.core.worker_speech")

#: Ceiling on the phrase-fetch IPC, in seconds. The audio worker answers
#: :meth:`~ayris.workers.audio_worker.AudioWorker.segment` from a buffer it
#: already holds, so a healthy round-trip is a few milliseconds; the timeout only
#: exists so a wedged worker cannot freeze the UI thread that
#: :class:`WorkerPhraseSource` runs on. Kept well under a second for that reason —
#: a missed phrase is «не расслышала», a frozen window is a bug.
_PHRASE_TIMEOUT_SEC = 2.0


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
