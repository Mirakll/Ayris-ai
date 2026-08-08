"""Worker processes: the protocol, the base class and the supervisor.

Ayris keeps recognition, synthesis and language models out of the main process so
that a wedged model cannot freeze the interface and a crashed one can be replaced
without restarting the assistant. The main process supervises; the workers do the
work.

Import from the submodules rather than from here inside a worker: a child process
that imports this package pulls in the supervisor as well, and every import in a
spawned process is paid for again on every restart.
"""

from ayris.workers.base import Worker, WorkerContext, method
from ayris.workers.manager import WorkerManager, WorkerStatus, WorkerSummary, install_workers
from ayris.workers.protocol import (
    AudioChunk,
    SharedAudioBlock,
    WorkerCancelledError,
    WorkerCrashError,
    WorkerError,
    WorkerStartError,
    WorkerTimeoutError,
    WorkerUnavailableError,
    open_audio,
)
from ayris.workers.registry import WorkerKind, WorkerPlan, WorkerSpec, plan_workers

__all__ = [
    "AudioChunk",
    "SharedAudioBlock",
    "Worker",
    "WorkerCancelledError",
    "WorkerContext",
    "WorkerCrashError",
    "WorkerError",
    "WorkerKind",
    "WorkerManager",
    "WorkerPlan",
    "WorkerSpec",
    "WorkerStartError",
    "WorkerStatus",
    "WorkerSummary",
    "WorkerTimeoutError",
    "WorkerUnavailableError",
    "install_workers",
    "method",
    "open_audio",
    "plan_workers",
]
