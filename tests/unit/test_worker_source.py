"""Источник калибровки поверх аудио-воркера (:mod:`ayris.audio.worker_source`).

Без звуковой карты и без Qt: фейковый воркер отдаёт заранее заданный PCM, сон
глушится, а разбор идёт настоящий. Проверяем, что фабрика зажигается только на
готовом воркере и что запись снимает два окна — тишину и фразу.
"""

from __future__ import annotations

from typing import Any

import pytest

from ayris.audio.worker_source import (
    WorkerAudioSource,
    worker_calibration_source,
)


class _FakeWorker:
    """Аудио-воркер-заглушка: отдаёт PCM из буфера и считает запросы ``read``."""

    def __init__(
        self,
        *,
        ready: bool = True,
        sample_rate: int = 16000,
        pcm: bytes = b"\x00\x00" * 1600,
    ) -> None:
        self._ready = ready
        self._sample_rate = sample_rate
        self._pcm = pcm
        self.reads: list[float] = []
        self.status_calls = 0

    def is_ready(self, name: str) -> bool:
        return self._ready and name == "audio"

    def call_sync(self, worker: str, method: str, params: Any = None) -> Any:
        assert worker == "audio"
        if method == "status":
            self.status_calls += 1
            return {"sample_rate": self._sample_rate}
        if method == "read":
            self.reads.append(float((params or {})["ms"]))
            return {"pcm": self._pcm, "sample_rate": self._sample_rate}
        raise AssertionError(f"неожиданный метод воркера: {method}")


def test_no_source_without_a_worker() -> None:
    assert worker_calibration_source(None) is None


def test_no_source_for_a_bare_object() -> None:
    # Объект без is_ready/call_sync не проходит isinstance протокола CalibrationWorker.
    assert worker_calibration_source(object()) is None


def test_no_source_when_worker_not_ready() -> None:
    assert worker_calibration_source(_FakeWorker(ready=False)) is None


def test_calibrates_through_the_worker_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    from ayris.audio.calibration import CalibrationReport, run_calibration

    # Ждать реальные 8 секунд записи незачем — глушим сон, оставляя настоящий разбор.
    monkeypatch.setattr("ayris.audio.worker_source.time.sleep", lambda *_a, **_k: None)
    worker = _FakeWorker()
    factory = worker_calibration_source(worker)
    assert factory is not None

    report = run_calibration(factory(), base_gain=1.5)

    assert isinstance(report, CalibrationReport)
    # Два окна: тишина ~3000 мс, затем фраза ~5000 мс.
    assert [round(ms) for ms in worker.reads] == [3000, 5000]


def test_phase_callback_fires_silence_then_phrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ayris.audio.worker_source.time.sleep", lambda *_a, **_k: None)
    worker = _FakeWorker()
    phases: list[str] = []
    source = WorkerAudioSource(worker, on_phase=phases.append)

    source.record(3.0)
    source.record(5.0)

    assert phases == ["silence", "phrase"]


def test_sample_rate_falls_back_when_status_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from ayris.audio.capture import TARGET_SAMPLE_RATE

    class _Broken(_FakeWorker):
        def call_sync(self, worker: str, method: str, params: Any = None) -> Any:
            if method == "status":
                raise RuntimeError("воркер не ответил")
            return super().call_sync(worker, method, params)

    monkeypatch.setattr("ayris.audio.worker_source.time.sleep", lambda *_a, **_k: None)
    source = WorkerAudioSource(_Broken())
    assert source.sample_rate == TARGET_SAMPLE_RATE
