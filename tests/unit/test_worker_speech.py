"""Task 47 input seam: the two adapters that bridge the pipeline onto the workers.

:mod:`ayris.core.worker_speech` turns the pipeline's ``SttSource`` and
``PhraseSource`` protocols into IPC calls on the worker supervisor. These tests
stand in a hand-written :class:`_FakeCaller` for
:class:`~ayris.workers.manager.WorkerManager` — no worker process, no model, no
microphone — and pin the two behaviours that matter: the exact call each adapter
makes, and how each one degrades when the worker cannot answer. A real
recognition failure (``SttError``) must reach the pipeline so it can be spoken; a
worker that is merely unavailable, timed out or crashed must degrade to «nothing
heard», never to a crash.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any

import pytest

from ayris.audio.stt.base import STT_SAMPLE_RATE, AudioBuffer, TranscriptResult
from ayris.core.errors import SttError
from ayris.core.worker_speech import (
    _PHRASE_TIMEOUT_SEC,
    WorkerPhraseSource,
    WorkerSttSource,
)
from ayris.workers.protocol import (
    WorkerCrashError,
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from ayris.workers.registry import WorkerKind

pytestmark = pytest.mark.unit


@dataclass
class _Call:
    """One recorded ``call_sync`` invocation, for asserting the wire contract."""

    worker: str
    method: str
    timeout: float | None
    audio: bytes | None
    sample_rate: int
    channels: int
    sample_format: str


@dataclass
class _FakeCaller:
    """A :class:`~ayris.core.worker_speech.WorkerCaller` that returns canned replies.

    ``reply`` is either the value ``call_sync`` returns or an exception it raises,
    so a single stub covers both the happy path and every degradation branch.
    """

    reply: object = None
    calls: list[_Call] = field(default_factory=list)

    def call_sync(
        self,
        worker: str,
        method: str,
        params: Any = None,
        *,
        timeout: float | None = None,
        audio: bytes | None = None,
        sample_rate: int = STT_SAMPLE_RATE,
        channels: int = 1,
        sample_format: str = "int16",
    ) -> Any:
        self.calls.append(
            _Call(
                worker=worker,
                method=method,
                timeout=timeout,
                audio=audio,
                sample_rate=sample_rate,
                channels=channels,
                sample_format=sample_format,
            )
        )
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def _buffer(pcm: bytes = b"\x01\x02\x03\x04", *, sample_rate: int = STT_SAMPLE_RATE) -> AudioBuffer:
    return AudioBuffer(pcm=pcm, sample_rate=sample_rate, channels=1)


# --------------------------------------------------------------------------- #
# WorkerSttSource
# --------------------------------------------------------------------------- #


def test_stt_source_returns_the_workers_transcript() -> None:
    reply = TranscriptResult(text="открой блокнот", confidence=0.0, engine="vosk").to_params()
    caller = _FakeCaller(reply=reply)

    result = WorkerSttSource(caller).transcribe(_buffer())

    assert isinstance(result, TranscriptResult)
    assert result.text == "открой блокнот"
    assert result.engine == "vosk"
    assert not result.is_empty


def test_stt_source_calls_transcribe_with_the_pcm_and_format() -> None:
    caller = _FakeCaller(reply=TranscriptResult(text="да").to_params())

    WorkerSttSource(caller).transcribe(_buffer(b"\x05\x06\x07\x08", sample_rate=8000))

    assert len(caller.calls) == 1
    call = caller.calls[0]
    assert call.worker == WorkerKind.STT.value
    assert call.method == "transcribe"
    assert call.audio == b"\x05\x06\x07\x08"
    assert call.sample_rate == 8000
    assert call.channels == 1
    assert call.sample_format == "int16"
    # The recognition runs on the pipeline's own session thread, so no bounded
    # UI-thread timeout is imposed here — the worker's spec timeout applies.
    assert call.timeout is None


def test_stt_source_reraises_a_real_recognition_failure() -> None:
    # A recognition problem carries a Russian message the pipeline speaks; it must
    # not be flattened to «не расслышала».
    caller = _FakeCaller(reply=SttError("no model", user_message="Модель не загружена."))

    with pytest.raises(SttError):
        WorkerSttSource(caller).transcribe(_buffer())


@pytest.mark.parametrize(
    "failure",
    [
        WorkerUnavailableError("stt not started"),
        WorkerTimeoutError("no answer in time"),
        WorkerCrashError("worker died"),
        FutureTimeoutError(),
    ],
)
def test_stt_source_degrades_to_empty_when_the_worker_cannot_answer(
    failure: BaseException,
) -> None:
    caller = _FakeCaller(reply=failure)

    result = WorkerSttSource(caller).transcribe(_buffer())

    assert result.is_empty
    assert result.engine == WorkerKind.STT.value


def test_stt_source_tolerates_a_non_mapping_reply() -> None:
    caller = _FakeCaller(reply=None)

    result = WorkerSttSource(caller).transcribe(_buffer())

    assert result.is_empty


# --------------------------------------------------------------------------- #
# WorkerPhraseSource
# --------------------------------------------------------------------------- #


def test_phrase_source_builds_a_buffer_from_the_segment() -> None:
    caller = _FakeCaller(
        reply={"available": True, "pcm": b"\x01\x02\x03\x04", "sample_rate": STT_SAMPLE_RATE}
    )

    buffer = WorkerPhraseSource(caller)()

    assert isinstance(buffer, AudioBuffer)
    assert buffer.pcm == b"\x01\x02\x03\x04"
    assert buffer.sample_rate == STT_SAMPLE_RATE
    assert buffer.channels == 1


def test_phrase_source_asks_the_audio_worker_with_a_bounded_timeout() -> None:
    caller = _FakeCaller(reply={"available": True, "pcm": b"\x00\x00"})

    WorkerPhraseSource(caller)()

    assert len(caller.calls) == 1
    call = caller.calls[0]
    assert call.worker == WorkerKind.AUDIO.value
    assert call.method == "segment"
    # Runs on the UI thread, so the fetch is capped hard rather than left to the
    # worker's own timeout — a missed phrase is fine, a frozen window is not.
    assert call.timeout == _PHRASE_TIMEOUT_SEC


def test_phrase_source_returns_none_when_no_segment_is_available() -> None:
    caller = _FakeCaller(reply={"available": False})

    assert WorkerPhraseSource(caller)() is None


def test_phrase_source_returns_none_on_empty_or_missing_pcm() -> None:
    assert WorkerPhraseSource(_FakeCaller(reply={"available": True, "pcm": b""}))() is None
    assert WorkerPhraseSource(_FakeCaller(reply={"available": True}))() is None


def test_phrase_source_returns_none_for_a_non_mapping_reply() -> None:
    assert WorkerPhraseSource(_FakeCaller(reply=None))() is None


@pytest.mark.parametrize(
    "failure",
    [
        WorkerUnavailableError("audio not started"),
        WorkerCrashError("worker died"),
        FutureTimeoutError(),
    ],
)
def test_phrase_source_returns_none_when_the_worker_cannot_answer(
    failure: BaseException,
) -> None:
    assert WorkerPhraseSource(_FakeCaller(reply=failure))() is None


def test_phrase_source_drops_a_malformed_buffer() -> None:
    # An odd byte count is not a whole number of int16 frames: AudioBuffer rejects
    # it, and the adapter must swallow that rather than crash the event handler.
    caller = _FakeCaller(reply={"available": True, "pcm": b"\x00\x00\x00"})

    assert WorkerPhraseSource(caller)() is None


def test_worker_manager_reply_shape_round_trips() -> None:
    # Guard against drift between the adapter's reader and TranscriptResult's own
    # wire shape: a full round-trip must survive isinstance/Mapping checks.
    params = TranscriptResult(text="привет", engine="vosk", confidence=0.0).to_params()
    assert isinstance(params, Mapping)
    assert WorkerSttSource(_FakeCaller(reply=params)).transcribe(_buffer()).text == "привет"
