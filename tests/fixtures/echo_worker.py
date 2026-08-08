"""Worker implementations used by ``tests/unit/test_workers.py``.

This module is imported inside spawned child processes, not by the test session,
so it must stay importable on its own: no pytest, no fixtures, and nothing from
the test module that spawns it. The manager puts this directory on the child's
``sys.path`` through :attr:`~ayris.workers.registry.WorkerSpec.python_path`.

Keeping the fixtures here rather than in the test file is not a style choice.
Under ``spawn`` the child re-imports the module that holds the target, and a
worker class defined inside ``test_workers.py`` would drag pytest's collection
machinery into every child.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TYPE_CHECKING

from ayris.core.errors import SttError
from ayris.workers.base import Worker, method
from ayris.workers.protocol import AudioChunk, open_audio

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

#: Set by :func:`poison`, checked by ``inherited``. A spawned child starts with
#: this back at ``False``, which is the assertion that ``spawn`` really is spawn.
STATE: dict[str, bool] = {"poisoned": False}


def poison() -> None:
    """Mutate module state in the parent, so a child can prove it inherits none."""
    STATE["poisoned"] = True


class EchoWorker(Worker):
    """Answers everything, and can be told to misbehave on request."""

    kind = "test"

    def on_start(self) -> None:
        self._events = 0
        if self.params.get("fail_on_start"):
            raise SttError(
                "echo worker was told to fail",
                user_message="Тестовый воркер не смог запуститься.",
            )

    @method()
    def echo(self, params: JsonObject) -> JsonObject:
        """Return the params untouched, plus the pid that handled them."""
        return {"echo": params, "pid": os.getpid()}

    @method()
    def add(self, params: JsonObject) -> int:
        """Sum two numbers, to prove a plain value survives the round trip."""
        return int(params["a"]) + int(params["b"])

    @method("say_name")
    def report_name(self, _params: JsonObject) -> str:
        """Exposed under a wire name that differs from the method name."""
        return self.context.name

    @method()
    def inherited(self, _params: JsonObject) -> JsonObject:
        """Report what leaked in from the parent process."""
        return {
            "poisoned": STATE["poisoned"],
            "params": dict(self.params),
            "has_qt": "PySide6" in sys.modules,
        }

    @method()
    def boom(self, params: JsonObject) -> None:
        """Raise a typed Ayris error, or a plain one, for the caller to catch."""
        if params.get("typed", True):
            raise SttError("recognition exploded", user_message="Распознавание сломалось.")
        raise ValueError("something entirely unexpected")

    @method()
    def unpicklable(self, _params: JsonObject) -> object:
        """Return something that cannot cross the pipe."""
        return lambda: None

    @method()
    def sleep(self, params: JsonObject) -> float:
        """Sleep, checking for cancellation, and report how long it actually slept."""
        seconds = float(params.get("seconds", 1.0))
        step = float(params.get("step", 0.02))
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            self.context.check_cancelled()
            time.sleep(step)
        return time.monotonic() - started

    @method()
    def freeze(self, params: JsonObject) -> None:
        """Block the dispatch thread without ever checking for cancellation.

        Used to show that a stalled handler does not stop the heartbeat: a worker
        is only declared hung when the beat itself stops.
        """
        time.sleep(float(params.get("seconds", 5.0)))

    @method()
    def wedge(self, params: JsonObject) -> None:
        """Freeze every thread in the process, heartbeat included.

        This is what a native decoder call that never returns actually does: it
        holds the interpreter lock, so the worker is alive as far as the operating
        system is concerned while no Python thread of it runs. Raising the switch
        interval and spinning starves the heartbeat thread the same way.
        """
        sys.setswitchinterval(float(params.get("switch_interval", 300.0)))
        deadline = time.monotonic() + float(params.get("seconds", 60.0))
        while time.monotonic() < deadline:
            pass

    @method()
    def die(self, params: JsonObject) -> None:
        """Leave the process immediately, without unwinding anything."""
        os._exit(int(params.get("code", 9)))

    @method()
    def emit(self, params: JsonObject) -> int:
        """Send unsolicited events, to exercise the reader thread and translators."""
        count = int(params.get("count", 1))
        for index in range(count):
            self.context.emit(str(params.get("kind", "tick")), {"index": index})
            self._events += 1
        return self._events

    @method()
    def warn(self, params: JsonObject) -> bool:
        """Log a warning, which the manager should re-emit in its own process."""
        self.log.warning("%s", params.get("message", "внимание"))
        return True

    @method()
    def audio_summary(self, params: JsonObject) -> JsonObject:
        """Read a shared-memory buffer and report what arrived.

        The checksum is what proves the samples came through unaltered without
        sending them back over the pipe.
        """
        chunk = AudioChunk.from_params(params)
        if chunk is None:
            return {"received": False}
        with open_audio(chunk) as pcm:
            data = bytes(pcm)
        return {
            "received": True,
            "nbytes": len(data),
            "duration_ms": round(chunk.duration_ms, 3),
            "sample_rate": chunk.sample_rate,
            "checksum": sum(data) % 1_000_003,
            "head": list(data[:4]),
        }


class SlowStartWorker(Worker):
    """Takes longer to start than the supervisor is willing to wait."""

    kind = "test"

    def on_start(self) -> None:
        time.sleep(float(self.params.get("start_delay", 5.0)))
