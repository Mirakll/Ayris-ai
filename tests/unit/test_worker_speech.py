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
    _WARM_TIMEOUT_SEC,
    WorkerPhraseSource,
    WorkerSttSource,
    make_transcribe_once,
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


def test_stt_source_warm_asks_the_worker_to_load_the_model() -> None:
    caller = _FakeCaller(reply={"started": True})

    WorkerSttSource(caller).warm()

    assert len(caller.calls) == 1
    call = caller.calls[0]
    assert call.worker == WorkerKind.STT.value
    assert call.method == "load_model"
    # Fired on an activation thread so the cold load overlaps the phrase; bounded
    # so a wedged worker cannot pin the warm thread forever.
    assert call.timeout == _WARM_TIMEOUT_SEC


@pytest.mark.parametrize(
    "failure",
    [
        WorkerUnavailableError("stt not started"),
        WorkerTimeoutError("no answer in time"),
        WorkerCrashError("worker died"),
        FutureTimeoutError(),
        SttError("no model", user_message="Модель не загружена."),
    ],
)
def test_stt_source_warm_is_best_effort(failure: BaseException) -> None:
    # Warming is an optimisation: a worker that cannot answer is swallowed, never
    # raised — the phrase still recognises, it just pays the cold load as before.
    caller = _FakeCaller(reply=failure)

    WorkerSttSource(caller).warm()  # must not raise

    assert len(caller.calls) == 1


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


# --------------------------------------------------------------------------- #
# make_transcribe_once — the «Тест распознавания» self-test
# --------------------------------------------------------------------------- #


@dataclass
class _ScriptedCaller:
    """A ``WorkerCaller`` whose answer depends on the method called.

    ``segment_replies`` is consumed one per audio ``segment`` call, so a test can
    script "nothing stale, nothing yet, then a phrase"; the last entry is reused
    once the script runs out. ``stt_reply`` answers every STT ``transcribe`` call,
    and the PCM handed to it is recorded so a test can prove *which* phrase was
    recognised — the fresh one, not a drained stale segment. ``status_reply``
    answers the microphone pre-check (a running capture by default) and
    ``read_reply`` answers the raw-window fallback (nothing by default, so a test
    opts in to it explicitly).
    """

    segment_replies: list[object]
    stt_reply: object = None
    status_reply: object = field(default_factory=lambda: {"running": True, "state": "running"})
    read_reply: object = None
    segment_calls: int = 0
    transcribe_calls: int = 0
    transcribe_audio: bytes | None = None
    status_calls: int = 0
    read_calls: int = 0

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
        if method == "status":
            self.status_calls += 1
            reply = self.status_reply
        elif method == "read":
            self.read_calls += 1
            reply = self.read_reply
        elif method == "segment":
            index = min(self.segment_calls, len(self.segment_replies) - 1)
            self.segment_calls += 1
            reply = self.segment_replies[index]
        else:
            self.transcribe_calls += 1
            self.transcribe_audio = audio
            reply = self.stt_reply
        if isinstance(reply, BaseException):
            raise reply
        return reply


_NO_PHRASE = {"available": False}


def _phrase(pcm: bytes) -> dict[str, object]:
    return {"available": True, "pcm": pcm, "sample_rate": STT_SAMPLE_RATE}


# __SELF_TEST_CASES__


def test_self_test_recognises_the_next_spoken_phrase() -> None:
    caller = _ScriptedCaller(
        segment_replies=[_NO_PHRASE, _NO_PHRASE, _phrase(b"\x01\x02\x03\x04")],
        stt_reply=TranscriptResult(text="открой блокнот", engine="vosk").to_params(),
    )

    text, elapsed_ms = make_transcribe_once(caller, listen_sec=1.0, poll_sec=0.0)()

    assert text == "открой блокнот"
    assert elapsed_ms >= 0
    assert caller.transcribe_calls == 1


def test_self_test_drains_a_stale_segment_before_listening() -> None:
    # A phrase buffered before the button was pressed must be discarded: the test
    # recognises the fresh phrase, not the click or an earlier sentence.
    caller = _ScriptedCaller(
        segment_replies=[_phrase(b"STAL"), _phrase(b"FRES")],
        stt_reply=TranscriptResult(text="да", engine="vosk").to_params(),
    )

    make_transcribe_once(caller, listen_sec=1.0, poll_sec=0.0)()

    assert caller.transcribe_audio == b"FRES"
    assert caller.segment_calls >= 2


def test_self_test_raises_when_nothing_is_heard() -> None:
    # Neither the VAD path nor the raw-window fallback found audio: a genuinely
    # silent window is the one case where «не услышала» is the honest answer.
    caller = _ScriptedCaller(segment_replies=[_NO_PHRASE])

    with pytest.raises(SttError):
        make_transcribe_once(caller, listen_sec=0.02, poll_sec=0.0)()

    assert caller.transcribe_calls == 0
    assert caller.read_calls >= 1  # the window fallback was tried before giving up


def test_self_test_raises_on_an_empty_transcript() -> None:
    # Silence comes back as an empty result; the self-test turns that into a clear
    # message rather than showing «»  ·  N мс.
    caller = _ScriptedCaller(
        segment_replies=[_NO_PHRASE, _phrase(b"\x01\x02\x03\x04")],
        stt_reply=TranscriptResult(text="   ", engine="vosk").to_params(),
    )

    with pytest.raises(SttError):
        make_transcribe_once(caller, listen_sec=1.0, poll_sec=0.0)()


def test_self_test_propagates_a_real_recognition_failure() -> None:
    # A model that would not load reaches the user unchanged, so «Не удалось
    # распознать» names the real reason instead of «не услышала».
    caller = _ScriptedCaller(
        segment_replies=[_NO_PHRASE, _phrase(b"\x01\x02\x03\x04")],
        stt_reply=SttError("no model", user_message="Модель не загружена."),
    )

    with pytest.raises(SttError) as excinfo:
        make_transcribe_once(caller, listen_sec=1.0, poll_sec=0.0)()
    assert excinfo.value.user_message == "Модель не загружена."


def test_self_test_reports_a_dead_microphone_instead_of_no_phrase() -> None:
    # Capture that is not running is named as a device problem, not flattened to
    # «не услышала» — and the test never even reaches the listen loop or STT.
    caller = _ScriptedCaller(
        segment_replies=[_phrase(b"\x01\x02\x03\x04")],
        stt_reply=TranscriptResult(text="да", engine="vosk").to_params(),
        status_reply={"running": False, "state": "device_lost"},
    )

    with pytest.raises(SttError) as excinfo:
        make_transcribe_once(caller, listen_sec=1.0, poll_sec=0.0)()

    assert "Микрофон недоступен" in excinfo.value.user_message
    assert caller.segment_calls == 0
    assert caller.transcribe_calls == 0


def test_self_test_falls_back_to_the_raw_window_when_the_gate_finds_no_phrase() -> None:
    # The VAD gate accepts nothing (a quiet mic, a gate set too high), so the
    # segment poll times out; the raw window is read and recognised instead, and
    # its PCM — not a segment's — reaches STT.
    caller = _ScriptedCaller(
        segment_replies=[_NO_PHRASE],
        stt_reply=TranscriptResult(text="привет", engine="vosk").to_params(),
        read_reply={"pcm": b"WNDW", "sample_rate": STT_SAMPLE_RATE},
    )

    text, _elapsed = make_transcribe_once(caller, listen_sec=0.05, poll_sec=0.0)()

    assert text == "привет"
    assert caller.read_calls >= 1
    assert caller.transcribe_audio == b"WNDW"
