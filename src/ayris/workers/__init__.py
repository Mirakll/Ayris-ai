"""Worker processes: the protocol, the base class and the supervisor.

Ayris keeps recognition, synthesis and language models out of the main process so
that a wedged model cannot freeze the interface and a crashed one can be replaced
without restarting the assistant. The main process supervises; the workers do the
work.

Import from the submodules rather than from here inside a worker: a child process
that imports this package pulls in the supervisor as well, and every import in a
spawned process is paid for again on every restart.

The names re-exported below are therefore bound **lazily** (PEP 562
``__getattr__``). Importing any worker submodule still runs this package's
``__init__``, so keeping it free of the supervisor's imports (``manager`` and
``registry``, which drag in ``config``/``pydantic``/``httpx``/``asyncio``) is what
lets a spawned ``audio``/``stt`` worker — which only needs ``base`` and
``protocol`` — start without paying for them. The symbols load on first attribute
access, which in practice happens only in the main process.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ayris.workers.base import Worker, WorkerContext, method
    from ayris.workers.manager import (
        WorkerManager,
        WorkerStatus,
        WorkerSummary,
        install_workers,
    )
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

#: Public name -> submodule it lives in. Resolved on demand by ``__getattr__`` so
#: that importing this package does not import the submodule (and its transitive
#: dependencies) until the name is actually used.
_LAZY: dict[str, str] = {
    "Worker": "ayris.workers.base",
    "WorkerContext": "ayris.workers.base",
    "method": "ayris.workers.base",
    "WorkerManager": "ayris.workers.manager",
    "WorkerStatus": "ayris.workers.manager",
    "WorkerSummary": "ayris.workers.manager",
    "install_workers": "ayris.workers.manager",
    "AudioChunk": "ayris.workers.protocol",
    "SharedAudioBlock": "ayris.workers.protocol",
    "WorkerCancelledError": "ayris.workers.protocol",
    "WorkerCrashError": "ayris.workers.protocol",
    "WorkerError": "ayris.workers.protocol",
    "WorkerStartError": "ayris.workers.protocol",
    "WorkerTimeoutError": "ayris.workers.protocol",
    "WorkerUnavailableError": "ayris.workers.protocol",
    "open_audio": "ayris.workers.protocol",
    "WorkerKind": "ayris.workers.registry",
    "WorkerPlan": "ayris.workers.registry",
    "WorkerSpec": "ayris.workers.registry",
    "plan_workers": "ayris.workers.registry",
}

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


def __getattr__(name: str) -> object:
    """Import a re-exported symbol on first access (PEP 562)."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so the next access skips __getattr__
    return value


def __dir__() -> list[str]:
    return list(__all__)
