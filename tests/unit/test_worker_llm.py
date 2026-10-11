"""The pipeline's consumer of the LLM worker (:mod:`ayris.core.worker_llm`).

These pin the proxy's contract without a worker process or a model: a hand-written
manager returns canned futures, so the tests exercise exactly what
:class:`~ayris.core.worker_llm.WorkerLlmClient` marshals — the ``chat`` payload
(messages via ``as_payload``, tools *flat*), the readiness probe off ``status``
(cached, re-probed, invalidated), cancellation (poll → out-of-band CANCEL), and the
mapping of transport failures onto :class:`~ayris.core.errors.LlmError`.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import pytest

from ayris.core.errors import LlmError
from ayris.core.worker_llm import WorkerLlmClient
from ayris.nlu.llm.base import FinishReason, LlmMessage, LlmTool
from ayris.workers.protocol import WorkerError
from ayris.workers.registry import WorkerKind

pytestmark = pytest.mark.unit

_LLM = WorkerKind.LLM.value


def _reply(**over: Any) -> dict[str, Any]:
    """A well-formed ``chat`` reply, overridable field by field."""
    base: dict[str, Any] = {
        "text": "привет",
        "tool_calls": [],
        "finish_reason": "stop",
        "model": "qwen2.5-0.5b",
        "engine": "llamacpp",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "duration_ms": 42,
    }
    base.update(over)
    return base


@dataclass
class _FakeManager:
    """A worker supervisor that resolves ``chat``/``status`` from memory.

    ``chat_result`` and ``status_result`` are either a value to resolve the future
    with or an :class:`Exception` to fail it with — the two ways a real future ends.
    Every call and every ``cancel`` is recorded so the tests can assert the wire.
    """

    chat_result: Any = None
    status_result: Any = None
    calls: list[tuple[str, str, Any]] = field(default_factory=list)
    autostart: list[bool] = field(default_factory=list)
    cancels: list[str] = field(default_factory=list)

    def call(
        self,
        worker: str,
        method: str,
        params: Any = None,
        *,
        timeout: float | None = None,
        autostart: bool = True,
    ) -> Future[Any]:
        self.calls.append((worker, method, params))
        self.autostart.append(autostart)
        future: Future[Any] = Future()
        result = self.status_result if method == "status" else self.chat_result
        if isinstance(result, Exception):
            future.set_exception(result)
        else:
            future.set_result(result)
        return future

    def cancel(self, worker: str, request_id: int | None = None) -> None:
        self.cancels.append(worker)


@dataclass
class _PendingChatManager:
    """A supervisor whose ``chat`` future stays open until ``cancel`` resolves it.

    Models the real cancellation handshake: the blocking call does not return on
    its own; only the out-of-band CANCEL ends it, with a ``cancelled`` reply.
    """

    future: Future[Any] = field(default_factory=Future)
    cancels: list[str] = field(default_factory=list)

    def call(self, worker: str, method: str, params: Any = None, **_kw: Any) -> Future[Any]:
        return self.future

    def cancel(self, worker: str, request_id: int | None = None) -> None:
        self.cancels.append(worker)
        if not self.future.done():
            self.future.set_result(
                {"text": "", "finish_reason": "cancelled", "model": "", "engine": "llamacpp"}
            )


# --------------------------------------------------------------------------- #
# complete(): the chat payload and the response it builds
# --------------------------------------------------------------------------- #


def test_complete_forwards_messages_and_builds_the_response() -> None:
    manager = _FakeManager(chat_result=_reply())
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    response = client.complete(
        [LlmMessage.system("ты Айрис"), LlmMessage.user("привет")],
        temperature=0.4,
        max_tokens=128,
    )

    worker, method, params = manager.calls[0]
    assert (worker, method) == (_LLM, "chat")
    assert params["messages"] == [
        {"role": "system", "content": "ты Айрис"},
        {"role": "user", "content": "привет"},
    ]
    assert params["temperature"] == 0.4
    assert params["max_tokens"] == 128
    # The reply maps field for field onto the response the pipeline speaks.
    assert response.text == "привет"
    assert response.model == "qwen2.5-0.5b"
    assert response.engine == "llamacpp"
    assert response.finish_reason is FinishReason.STOP
    assert response.usage.prompt_tokens == 7
    assert response.usage.completion_tokens == 3
    assert response.duration_ms == 42


def test_complete_serialises_tools_flat_not_the_openai_envelope() -> None:
    manager = _FakeManager(chat_result=_reply())
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    schema = {"type": "object", "properties": {"город": {"type": "string"}}}
    client.complete(
        [LlmMessage.user("погода")],
        tools=[LlmTool(name="weather", description="погода в городе", parameters=schema)],
    )

    _, _, params = manager.calls[0]
    # The worker parses {name, description, parameters}; the OpenAI as_payload
    # envelope ({type, function:{...}}) would be dropped on the floor there.
    assert params["tools"] == [
        {"name": "weather", "description": "погода в городе", "parameters": schema}
    ]


def test_complete_omits_temperature_and_max_tokens_when_unset() -> None:
    manager = _FakeManager(chat_result=_reply())
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    client.complete([LlmMessage.user("привет")])

    _, _, params = manager.calls[0]
    assert "temperature" not in params
    assert "max_tokens" not in params


def test_complete_decodes_tool_calls_from_the_reply() -> None:
    manager = _FakeManager(
        chat_result=_reply(
            finish_reason="tool_calls",
            tool_calls=[{"name": "weather", "arguments": {"город": "Токио"}, "call_id": "c1"}],
        )
    )
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    response = client.complete([LlmMessage.user("погода в Токио")])

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert (call.name, call.call_id) == ("weather", "c1")
    assert call.arguments == {"город": "Токио"}


# --------------------------------------------------------------------------- #
# transport failures → LlmError (and a worker LlmError propagates as-is)
# --------------------------------------------------------------------------- #


def test_a_worker_transport_failure_becomes_an_llm_error() -> None:
    manager = _FakeManager(chat_result=WorkerError("канал закрыт"))
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    with pytest.raises(LlmError):
        client.complete([LlmMessage.user("привет")])


def test_a_worker_llm_error_propagates_unchanged() -> None:
    # The manager reconstructs worker-side AyrisError subclasses; an LlmError the
    # provider raised must reach the pipeline as itself, not be re-wrapped.
    manager = _FakeManager(chat_result=LlmError("провайдер отказал"))
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    with pytest.raises(LlmError, match="провайдер отказал"):
        client.complete([LlmMessage.user("привет")])


def test_a_non_mapping_reply_is_an_llm_error() -> None:
    manager = _FakeManager(chat_result=["не", "словарь"])
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    with pytest.raises(LlmError):
        client.complete([LlmMessage.user("привет")])


# --------------------------------------------------------------------------- #
# cancellation: poll the predicate, send one out-of-band CANCEL
# --------------------------------------------------------------------------- #


def test_cancel_predicate_sends_one_cancel_and_returns_cancelled() -> None:
    manager = _PendingChatManager()
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    response = client.complete([LlmMessage.user("длинный ответ")], cancel=lambda: True)

    assert manager.cancels == [_LLM]
    assert response.finish_reason is FinishReason.CANCELLED


def test_no_cancel_is_sent_when_the_predicate_stays_false() -> None:
    manager = _FakeManager(chat_result=_reply())
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    client.complete([LlmMessage.user("привет")], cancel=lambda: False)

    assert manager.cancels == []


# --------------------------------------------------------------------------- #
# configured: probe status (autostart off), cache true, re-probe false
# --------------------------------------------------------------------------- #


def test_configured_reads_the_worker_status_without_waking_it() -> None:
    manager = _FakeManager(status_result={"configured": True})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is True
    assert manager.calls[0][:2] == (_LLM, "status")
    # A readiness probe must not autostart a cold worker.
    assert manager.autostart[0] is False


def test_configured_caches_a_true_verdict() -> None:
    manager = _FakeManager(status_result={"configured": True})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is True
    assert client.configured is True
    # The second read is served from the cache — the worker is asked only once.
    assert sum(1 for c in manager.calls if c[1] == "status") == 1


def test_configured_reprobes_while_false() -> None:
    manager = _FakeManager(status_result={"configured": False})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is False
    assert client.configured is False
    # A «not yet» is never cached, so a model that warms later is noticed.
    assert sum(1 for c in manager.calls if c[1] == "status") == 2


def test_configured_is_false_when_the_status_probe_fails() -> None:
    manager = _FakeManager(status_result=WorkerError("воркер не поднят"))
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is False


def test_invalidate_drops_a_cached_true_verdict() -> None:
    manager = _FakeManager(status_result={"configured": True})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is True
    client.invalidate()
    assert client.configured is True
    # After invalidation the next read re-probes rather than trusting the cache.
    assert sum(1 for c in manager.calls if c[1] == "status") == 2


# --------------------------------------------------------------------------- #
# prewarm: warm the configured verdict off the first phrase
# --------------------------------------------------------------------------- #


@dataclass
class _FlippingManager:
    """A supervisor whose ``status`` returns ``False`` a few times then ``True``.

    Models a cold worker warming a GGUF: the readiness probe says «not yet» until
    the model is loaded. ``ready_after`` is how many probes report false first.
    """

    ready_after: int = 2
    probes: int = 0

    def call(self, worker: str, method: str, params: Any = None, **_kw: Any) -> Future[Any]:
        future: Future[Any] = Future()
        if method == "status":
            self.probes += 1
            future.set_result({"configured": self.probes > self.ready_after})
        else:
            future.set_result(_reply())
        return future

    def cancel(self, worker: str, request_id: int | None = None) -> None:  # pragma: no cover
        pass


def test_prewarm_caches_the_verdict_once_the_worker_is_ready() -> None:
    manager = _FakeManager(status_result={"configured": True})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    thread = client.prewarm()
    assert thread is not None
    thread.join(2.0)
    assert not thread.is_alive()
    # The verdict is now cached: reading it does not probe the worker again.
    assert client.configured is True
    assert sum(1 for c in manager.calls if c[1] == "status") == 1


def test_prewarm_is_a_noop_when_already_configured() -> None:
    manager = _FakeManager(status_result={"configured": True})
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]

    assert client.configured is True  # caches the verdict
    assert client.prewarm() is None  # nothing left to warm → no thread


def test_prewarm_loop_retries_until_the_verdict_turns_true() -> None:
    manager = _FlippingManager(ready_after=2)
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]
    slept: list[float] = []

    ok = client._prewarm_loop(attempts=10, interval=0.5, sleep=slept.append)

    assert ok is True
    # Two «not yet» probes, each followed by a sleep, then the third succeeds.
    assert manager.probes == 3
    assert slept == [0.5, 0.5]


def test_prewarm_loop_gives_up_after_the_attempt_budget() -> None:
    manager = _FlippingManager(ready_after=99)  # never ready
    client = WorkerLlmClient(manager)  # type: ignore[arg-type]
    slept: list[float] = []

    ok = client._prewarm_loop(attempts=3, interval=0.1, sleep=slept.append)

    assert ok is False
    assert manager.probes == 3
    assert slept == [0.1, 0.1, 0.1]
