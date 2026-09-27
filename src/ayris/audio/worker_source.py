"""``AudioSource`` поверх аудио-воркера — калибровка без второго владельца микрофона.

Во время работы микрофоном уже владеет аудио-воркер: он непрерывно пишет звук в
свой кольцевой буфер. Открыть устройство второй раз ради калибровки небезопасно (и
раньше именно так и падало), поэтому источник калибровки не трогает PortAudio — он
ждёт нужное окно вживую (пока пользователь молчит, затем говорит) и забирает у
воркера ровно этот хвост записи его методом ``read``.

Один и тот же источник обслуживает и мастер первого запуска (:mod:`ayris.onboarding`),
и вкладку «Голос»: оба зовут :func:`~ayris.audio.calibration.run_calibration` над
ним, не заводя второй поток захвата.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.audio.calibration import AudioSource

__all__ = ["CalibrationWorker", "WorkerAudioSource", "worker_calibration_source"]

_log = get_logger(__name__)


@runtime_checkable
class CalibrationWorker(Protocol):
    """Узкий срез супервизора воркеров, нужный калибровке через буфер.

    Ровно два метода: спросить готовность аудио-воркера и синхронно дёрнуть его
    ``status``/``read``. И :class:`~ayris.workers.manager.WorkerManager`, и фейк в
    тестах подходят структурно, поэтому :func:`worker_calibration_source` проверяет
    их через ``isinstance`` этого протокола, а не конкретный класс.
    """

    def is_ready(self, name: str) -> bool: ...

    def call_sync(self, worker: str, method: str, params: Any = None) -> Any: ...


class WorkerAudioSource:
    """``AudioSource`` поверх аудио-воркера: калибровка через его кольцевой буфер.

    Микрофоном уже владеет воркер, поэтому источник не открывает устройство второй
    раз, а честно ждёт ``seconds`` (пока пользователь молчит, затем говорит) и берёт
    у воркера ровно этот хвост записи методом ``read``.

    ``on_phase`` вызывается прямо перед каждой записью (``"silence"`` для первой,
    ``"phrase"`` для второй) — вызывающий превращает это в подсказку «помолчите» /
    «говорите». Колбэк летит из фонового потока калибровки, маршалить его в UI-поток
    должен сам вызывающий.
    """

    def __init__(
        self,
        worker: CalibrationWorker,
        *,
        on_phase: Callable[[str], None] | None = None,
    ) -> None:
        self._worker = worker
        self._on_phase = on_phase
        self._rate: int | None = None
        self._reads = 0

    @property
    def sample_rate(self) -> int:
        if self._rate is None:
            from ayris.audio.capture import TARGET_SAMPLE_RATE

            try:
                status = self._worker.call_sync("audio", "status")
                self._rate = int(status.get("sample_rate") or TARGET_SAMPLE_RATE)
            except Exception:
                _log.exception("не удалось узнать частоту дискретизации аудио-воркера")
                self._rate = TARGET_SAMPLE_RATE
        return self._rate

    def record(self, seconds: float) -> bytes:
        self._reads += 1
        if self._on_phase is not None:
            self._on_phase("silence" if self._reads == 1 else "phrase")
        # Звук уже пишется воркером; ждём окно вживую, затем берём его хвост из буфера.
        time.sleep(max(0.0, seconds))
        response = self._worker.call_sync("audio", "read", {"ms": seconds * 1000.0})
        pcm = response.get("pcm", b"")
        return bytes(pcm) if pcm else b""


def worker_calibration_source(control: object) -> Callable[[], AudioSource] | None:
    """Фабрика источника калибровки над ``control`` — или ``None``, если рано.

    Принимает узкий ``WorkerControl``, который GUI публикует через
    :func:`~ayris.gui.widgets.active_worker_control` (или любой объект с
    ``is_ready``/``call_sync``). Возвращает фабрику, только когда аудио-воркер уже
    поднят, чтобы кнопка «Калибровать» зажигалась лишь при живом захвате, а не
    падала в попытке прочитать пустой буфер.
    """
    if isinstance(control, CalibrationWorker) and control.is_ready("audio"):
        worker = control
        return lambda: WorkerAudioSource(worker)
    return None
