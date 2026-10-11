"""Feed the pipeline's chat off the LLM worker, so the local model answers.

Task 47 mounted the pipeline's output and task 63 gave it an :class:`LlmClient`
seam (:meth:`~ayris.core.pipeline.Pipeline.set_llm`), but nothing in the running
application ever called it: the pipeline kept its default
:class:`~ayris.nlu.llm.base.NullLlmClient`, which answers every turn with «модель
не настроена». Meanwhile the :class:`~ayris.workers.llm_worker.LlmWorker` warms a
downloaded GGUF at startup and then sits ready with no consumer. This adapter is
that consumer: it is the :class:`~ayris.nlu.llm.base.LlmClient` the pipeline talks
to, and every call is forwarded to the worker's ``chat`` RPC — so «запускается
вместе с Айрис» finally means the model actually replies.

Why a worker proxy rather than an in-process client. The model is multi-gigabyte
and holds the GIL for the length of every token; keeping it in the worker is the
whole point of the process split (task 05), and the worker already owns its
lifecycle — warm at start, unload when idle, cancel mid-generation. This side
holds no model, only a manager handle, and marshals one ``chat`` per turn.

Two shapes to respect. The worker parses tools *flat* — ``{name, description,
parameters}`` — not the OpenAI ``as_payload`` envelope, so :meth:`complete`
serialises them that way (see :func:`~ayris.workers.llm_worker._tools`). And
cancellation cannot ride the blocking call: the pipeline's ``cancel`` predicate is
polled here while the request runs, and when it flips the supervisor's out-of-band
CANCEL is sent (:meth:`~ayris.workers.manager.WorkerManager.cancel`), which the
worker's stream notices between chunks.

The worker's own ``sentence`` events (bridged to the bus as
:class:`~ayris.core.events.LlmSentenceReady`) are left alone: nothing subscribes to
them, and the pipeline speaks the whole answer through its own ``_respond``, so
forwarding the finished text here does not double up the speech.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from ayris.core.errors import LlmError
from ayris.nlu.llm.base import (
    FinishReason,
    LlmClient,
    LlmMessage,
    LlmResponse,
    LlmTool,
    LlmToolCall,
    LlmUsage,
)
from ayris.workers.protocol import WorkerError
from ayris.workers.registry import WorkerKind

if TYPE_CHECKING:
    from ayris.core.models import JsonObject

__all__ = ["WorkerLlmCaller", "WorkerLlmClient"]

_log = logging.getLogger("ayris.core.worker_llm")

#: How long the ``configured`` probe waits on the worker's ``status``. Short: the
#: worker answers it from memory, and a slow reply must not stall the pipeline's
#: understanding decision.
_STATUS_TIMEOUT_SEC = 2.0

#: Gap between checks of the ``cancel`` predicate while a ``chat`` is in flight.
#: Small enough that «Айрис, стоп» reaches a running generation promptly, large
#: enough not to busy-spin on the future.
_CANCEL_POLL_SEC = 0.05

#: How many times the background prewarm re-probes ``configured`` before giving up.
#: The worker warms a downloaded GGUF at startup; twenty probes a second apart
#: cover a slow cold load without pinning a thread forever when no model is set.
_PREWARM_ATTEMPTS = 20

#: Pause between prewarm probes. The warm-up is off the critical path, so a lazy
#: once-a-second poll is plenty — it only has to beat the user's first phrase.
_PREWARM_INTERVAL_SEC = 1.0


class WorkerLlmCaller(Protocol):
    """The slice of :class:`~ayris.workers.manager.WorkerManager` this proxy needs.

    A protocol so the proxy is unit-testable with a hand-written stub that returns
    canned futures without spawning a process or loading a model;
    :class:`~ayris.workers.manager.WorkerManager` satisfies it.
    """

    def call(
        self,
        worker: str,
        method: str,
        params: JsonObject | None = None,
        *,
        timeout: float | None = None,
        autostart: bool = True,
    ) -> Future[Any]: ...

    def cancel(self, worker: str, request_id: int | None = None) -> None: ...


class WorkerLlmClient(LlmClient):
    """The pipeline's :class:`LlmClient`, backed by the LLM worker's ``chat``.

    Holds no model — only a manager handle. Every :meth:`complete` is one ``chat``
    RPC; :attr:`configured` reflects the worker's own ``status`` (cached once true,
    re-probed while false), and :meth:`invalidate` drops that cache when the
    configuration changes so a freshly chosen model is picked up.
    """

    name: ClassVar[str] = "worker"
    #: The proxy hands back the finished answer in one piece; it exposes no token
    #: deltas, so streaming is off and the base replays :meth:`complete`.
    supports_streaming: ClassVar[bool] = False
    #: Conservative: the shipped default engine (llama.cpp) does not tool-call, and
    #: nothing in the app drove tool calls through the pipeline before this wiring.
    #: Kept off so no tool schema is sent that the local model would ignore.
    supports_tools: ClassVar[bool] = False

    def __init__(self, manager: WorkerLlmCaller, *, worker: str = WorkerKind.LLM.value) -> None:
        self._manager = manager
        self._worker = worker
        self._lock = Lock()
        #: ``True`` once the worker reported itself configured; ``None`` until then
        #: and after :meth:`invalidate`. Only the positive verdict is cached — a
        #: «not yet» is re-probed, so a model that warms after start is noticed.
        self._configured: bool | None = None

    @property
    def configured(self) -> bool:
        """Whether the worker holds a usable model, from its ``status`` (cached)."""
        with self._lock:
            if self._configured:
                return True
            # The probe is held under the lock so a background prewarm and the
            # first phrase can never both probe the same «not yet» window — one
            # waits, then reads the verdict the other cached. Sequential reads are
            # unchanged (the lock is free between them); only concurrent ones dedup.
            ok = self._probe_configured()
            if ok:
                self._configured = True
            return ok

    def invalidate(self) -> None:
        """Forget the cached ``configured`` verdict; the next read re-probes.

        Wired to :class:`~ayris.core.events.ConfigChanged` so choosing or clearing
        a model (which restarts the worker via ``RestartScope.LLM``) is reflected
        without a relaunch.
        """
        with self._lock:
            self._configured = None

    def prewarm(self) -> Thread | None:
        """Warm the ``configured`` verdict in the background, off the first phrase.

        The expensive part of the first hybrid or chat turn is the synchronous
        ``status`` RPC behind :attr:`configured`: the worker loads a multi-gigabyte
        GGUF lazily, so the probe can block for up to :data:`_STATUS_TIMEOUT_SEC`
        on that cold path. By polling the verdict on a daemon thread at startup the
        positive result is cached before the user ever speaks, and the pipeline's
        understanding decision reads it for free.

        Returns the spawned thread (so callers can join it in tests), or ``None``
        when the verdict is already cached — there is nothing to warm.
        """
        with self._lock:
            if self._configured:
                return None
        thread = Thread(target=self._prewarm_loop, name="ayris-llm-prewarm", daemon=True)
        thread.start()
        return thread

    def _prewarm_loop(
        self,
        *,
        attempts: int = _PREWARM_ATTEMPTS,
        interval: float = _PREWARM_INTERVAL_SEC,
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Poll :attr:`configured` until it is true or ``attempts`` run out.

        Reads the public property so a positive verdict is cached the same way the
        first phrase would have cached it; ``autostart`` stays off inside the probe,
        so this never wakes a cold worker — it only notices one that warms itself.
        ``sleep`` is injected so tests drive the loop without real delays. Any
        failure is swallowed: a best-effort warm-up must never crash its thread —
        the worst case is the first phrase pays the probe it would have paid anyway.
        """
        try:
            for _ in range(max(1, attempts)):
                if self.configured:
                    return True
                sleep(interval)
        except Exception as exc:
            _log.debug("llm: прогрев вердикта прерван: %s", exc)
        return False

    def _probe_configured(self) -> bool:
        """Ask the worker's ``status`` whether a model is configured, best-effort.

        ``autostart`` is off: a readiness probe must not wake a cold worker. Any
        failure — unavailable, timed out — reads as «not configured», which sends
        hybrid understanding to plain chat instead of raising.
        """
        try:
            future = self._manager.call(
                self._worker, "status", {}, timeout=_STATUS_TIMEOUT_SEC, autostart=False
            )
            reply = future.result(_STATUS_TIMEOUT_SEC + 1.0)
        except (WorkerError, FutureTimeoutError) as exc:
            _log.debug("llm: статус воркера недоступен: %s", exc)
            return False
        return bool(reply.get("configured")) if isinstance(reply, Mapping) else False

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        """Answer ``messages`` by forwarding one ``chat`` call to the worker.

        Raises:
            LlmError: the worker was unreachable, timed out or the provider refused
                the request. A missing model is *not* raised — the worker answers
                with its «не настроено» sentence, returned as the response text.
        """
        params: JsonObject = {
            "messages": [message.as_payload() for message in messages],
            "tools": [_tool_payload(tool) for tool in tools],
        }
        if temperature is not None:
            params["temperature"] = temperature
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        try:
            future = self._manager.call(self._worker, "chat", params)
        except WorkerError as exc:
            raise LlmError(f"llm worker unavailable: {exc}") from exc
        return _to_response(self._await(future, cancel))

    def _await(self, future: Future[Any], cancel: Callable[[], bool] | None) -> Any:
        """Block on the worker, sending CANCEL if the pipeline's predicate flips.

        The blocking ``chat`` cannot carry the ``cancel`` predicate itself, so it is
        polled between short waits; the first time it is true the out-of-band CANCEL
        goes out once, and the worker ends the generation (returning ``cancelled``).
        """
        cancel_sent = False
        while True:
            try:
                return future.result(_CANCEL_POLL_SEC)
            except FutureTimeoutError:
                if not cancel_sent and cancel is not None and cancel():
                    self._manager.cancel(self._worker)
                    cancel_sent = True
            except LlmError:
                raise
            except WorkerError as exc:
                raise LlmError(f"llm worker failed: {exc}") from exc


def _tool_payload(tool: LlmTool) -> JsonObject:
    """Serialise a tool the way the worker parses it: flat, not the OpenAI envelope."""
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": dict(tool.parameters),
    }


def _to_response(reply: Any) -> LlmResponse:
    """Build an :class:`LlmResponse` from the worker's ``chat`` reply dict."""
    if not isinstance(reply, Mapping):
        raise LlmError(f"llm worker returned {type(reply).__name__}, expected a reply mapping")
    return LlmResponse(
        text=str(reply.get("text", "")),
        tool_calls=_tool_calls(reply.get("tool_calls")),
        model=str(reply.get("model", "")),
        engine=str(reply.get("engine", "")),
        finish_reason=_finish_reason(reply.get("finish_reason")),
        usage=_usage(reply.get("usage")),
        duration_ms=_as_int(reply.get("duration_ms")),
    )


def _tool_calls(raw: Any) -> tuple[LlmToolCall, ...]:
    """Reconstruct decoded tool calls from the reply's ``{name, arguments, call_id}``."""
    if not isinstance(raw, list):
        return ()
    calls: list[LlmToolCall] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        arguments = item.get("arguments")
        calls.append(
            LlmToolCall(
                name=name,
                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                call_id=str(item.get("call_id", "")),
            )
        )
    return tuple(calls)


def _finish_reason(raw: Any) -> FinishReason:
    """Map the reply's ``finish_reason`` onto the enum, defaulting to ``STOP``."""
    try:
        return FinishReason(str(raw))
    except ValueError:
        return FinishReason.STOP


def _usage(raw: Any) -> LlmUsage:
    """Read token counts out of the reply's ``usage`` block; zero when absent."""
    if not isinstance(raw, Mapping):
        return LlmUsage()
    return LlmUsage(
        prompt_tokens=_as_int(raw.get("prompt_tokens")),
        completion_tokens=_as_int(raw.get("completion_tokens")),
    )


def _as_int(value: Any) -> int:
    """A non-negative int from a wire value, or 0 when it is missing or wrong-typed."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return max(0, int(value))
