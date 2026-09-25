"""Task 62: the local LLM stack — Ollama, LM Studio, llama.cpp, catalogue, router.

The counterpart to :mod:`tests.unit.test_llm_custom`, one level down: where that
file proves the cloud «custom» provider on a mock transport, this proves the three
local engines, the recommended-models catalogue and the Online/Offline/Auto router
— all without a socket, a server or a native wheel.

* :class:`TestOllama` — the native ``/api`` engine on a bespoke NDJSON recorder:
  reachability, the chat stream, the model list, ``pull`` progress and the RAM
  guard that refuses an oversized model before any byte moves.
* :class:`TestLmStudio` — the OpenAI-compatible local server, reusing the SSE
  recorder: the ``/v1`` endpoint rules, the keyless auth header, the model list.
* :class:`TestCatalogue` — pure verdict / limit / lookup calls, no HTTP at all.
* :class:`TestLlamaCpp` — the in-process engine through an injected fake factory:
  streaming, cancellation, temperature, idle unload, the «unconfigured» reasons.
* :class:`TestRouter` — mode dispatch, auto fallback, connectivity wiring and the
  quiet recovery and economy-mode drop, driven by a programmable stub client.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

import httpx
import pytest

from ayris.core.connectivity import ConnectivityMonitor
from ayris.core.errors import LlmAuthError, LlmError, LlmQuotaError
from ayris.core.events import EventBus, NotificationRequested, OnlineStatusChanged
from ayris.core.models import JsonObject
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.nlu.llm import catalog
from ayris.nlu.llm.base import (
    FinishReason,
    LlmClient,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmResponse,
    LlmTextDelta,
    LlmTool,
)
from ayris.nlu.llm.cloud import OpenAiCompatibleClient
from ayris.nlu.llm.factory import create_llm_client, is_local_provider
from ayris.nlu.llm.llamacpp_client import (
    DEFAULT_TEMPERATURE,
    LlamaCppLlmClient,
    LlamaCppOptions,
)
from ayris.nlu.llm.lmstudio_client import LmStudioLlmClient
from ayris.nlu.llm.ollama_client import OllamaLlmClient, OllamaPullProgress
from ayris.nlu.llm.router import LlmMode, LlmRouter

pytestmark = pytest.mark.unit

#: A key value that must never surface in a header when the engine is keyless.
LMS_KEY = "lm-studio-secret"

#: A one-message prompt reused across the router tests.
MESSAGES = (LlmMessage.user("привет"),)


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.entries[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        del self.entries[(service_name, username)]


@pytest.fixture(autouse=True)
def _isolate_secrets() -> Iterator[None]:
    """Point the process store at an empty in-memory keyring, never the real one.

    Both local factories resolve a key through the process store even when none is
    needed (Ollama is keyless, LM Studio usually is), so an empty fake keeps «no
    key» deterministic and touches nothing outside the process.
    """
    reset_secrets(SecretsStore("Ayris-test-empty", backend=FakeKeyring()))
    yield
    reset_secrets()


# ----------------------------------------------------------------------
# Ollama: a bespoke recorder for its native newline-delimited JSON API
# ----------------------------------------------------------------------


def ollama_ndjson(lines: tuple[JsonObject, ...]) -> bytes:
    """Frame objects as Ollama does: one JSON document per line, newline-separated."""
    body = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
    return body.encode("utf-8")


class OllamaRecorder:
    """A :class:`httpx.MockTransport` handler speaking Ollama's ``/api`` endpoints.

    Routes ``/api/version`` (reachability), ``/api/tags`` (model list), ``/api/chat``
    (the NDJSON answer stream) and ``/api/pull`` (download progress), each answered
    from bytes handed in at construction. With ``raise_on_version`` the reachability
    ping fails like a refused connection — that is how «Ollama не запущена» is proven.
    """

    def __init__(
        self,
        *,
        tags: JsonObject | None = None,
        chat: bytes = b"",
        pull: bytes = b"",
        raise_on_version: bool = False,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._tags = tags if tags is not None else {"models": [{"name": "llama3.2:1b"}]}
        self._chat = chat
        self._pull = pull
        self._raise_on_version = raise_on_version

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/api/version"):
            if self._raise_on_version:
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(200, json={"version": "0.4.7"})
        if path.endswith("/api/tags"):
            return httpx.Response(200, json=self._tags)
        ndjson = {"content-type": "application/x-ndjson"}
        if path.endswith("/api/chat"):
            return httpx.Response(200, content=self._chat, headers=ndjson)
        if path.endswith("/api/pull"):
            return httpx.Response(200, content=self._pull, headers=ndjson)
        return httpx.Response(404, json={"error": f"no route for {path}"})

    def request_to(self, suffix: str) -> httpx.Request:
        """The last request whose path ends with ``suffix`` (asserts one exists)."""
        matches = [r for r in self.requests if r.url.path.endswith(suffix)]
        assert matches, f"no request reached {suffix}"
        return matches[-1]

    def body_of(self, request: httpx.Request) -> dict[str, Any]:
        """The JSON body Ayris sent, decoded as an object."""
        decoded: dict[str, Any] = json.loads(request.content)
        return decoded


def ollama_client(
    recorder: OllamaRecorder,
    *,
    base_url: str = "http://127.0.0.1:11434",
    model: str = "llama3.2:1b",
    extra: Mapping[str, Any] | None = None,
) -> LlmClient:
    """An Ollama client wired to ``recorder`` — the factory, no store, no network."""
    return create_llm_client(
        "ollama",
        model=model,
        base_url=base_url,
        transport=httpx.MockTransport(recorder),
        max_retries=0,
        extra=dict(extra or {}),
    )


# ----------------------------------------------------------------------
# LM Studio: the OpenAI-shaped SSE recorder, as in test_llm_custom
# ----------------------------------------------------------------------


def openai_sse(
    text_chunks: tuple[str, ...],
    *,
    finish: str = "stop",
    usage: tuple[int, int] | None = (12, 7),
) -> bytes:
    """Build an OpenAI-shaped SSE body: text frames, a finish frame, usage, DONE."""
    lines: list[str] = []
    for chunk in text_chunks:
        frame = {"choices": [{"index": 0, "delta": {"content": chunk}}]}
        lines.append(f"data: {json.dumps(frame, ensure_ascii=False)}")
    finish_frame = {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
    lines.append(f"data: {json.dumps(finish_frame)}")
    if usage is not None:
        prompt, completion = usage
        block = {
            "choices": [],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
        }
        lines.append(f"data: {json.dumps(block)}")
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


class OpenAiRecorder:
    """A :class:`httpx.MockTransport` handler routing the OpenAI ``/v1`` endpoints."""

    def __init__(
        self,
        *,
        sse: bytes = b"",
        models: dict[str, Any] | None = None,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._sse = sse or openai_sse(("ok",))
        self._models = models if models is not None else {"data": [{"id": "local-model"}]}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            headers = {"content-type": "text/event-stream"}
            return httpx.Response(200, content=self._sse, headers=headers)
        if path.endswith("/models"):
            return httpx.Response(200, json=self._models)
        return httpx.Response(404, json={"error": {"message": f"no route for {path}"}})

    @property
    def last(self) -> httpx.Request:
        """The most recent request the transport saw."""
        assert self.requests, "no request reached the transport"
        return self.requests[-1]

    def body_of(self, request: httpx.Request) -> dict[str, Any]:
        """The JSON body Ayris sent, decoded as an object."""
        decoded: dict[str, Any] = json.loads(request.content)
        return decoded


def lmstudio_client(
    recorder: OpenAiRecorder,
    *,
    base_url: str = "http://127.0.0.1:1234/v1",
    model: str = "local-model",
    api_key: str = "",
) -> LlmClient:
    """An LM Studio client wired to ``recorder`` — the factory, no store, no network."""
    return create_llm_client(
        "lmstudio",
        model=model,
        base_url=base_url,
        api_key=api_key,
        transport=httpx.MockTransport(recorder),
        max_retries=0,
    )


class TestOllama:
    """The native Ollama engine over a mock of its ``/api`` surface."""

    def test_is_a_local_provider(self) -> None:
        assert is_local_provider("ollama")

    def test_configured_needs_only_an_endpoint(self) -> None:
        client = ollama_client(OllamaRecorder())
        assert isinstance(client, OllamaLlmClient)
        assert client.configured is True
        client.close()

    def test_chat_stream_carries_text_usage_and_engine(self) -> None:
        recorder = OllamaRecorder(
            chat=ollama_ndjson(
                (
                    {"message": {"content": "При"}, "done": False},
                    {"message": {"content": "вет"}, "done": False},
                    {"message": {"content": ", мир"}, "done": False},
                    {
                        "done": True,
                        "done_reason": "stop",
                        "prompt_eval_count": 5,
                        "eval_count": 3,
                    },
                )
            )
        )
        client = ollama_client(recorder)
        response = client.complete([LlmMessage.user("привет")])
        assert response.text == "Привет, мир"
        assert response.engine == "ollama"
        assert response.model == "llama3.2:1b"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3
        client.close()

    def test_chat_request_pings_version_then_posts_chat(self) -> None:
        recorder = OllamaRecorder(chat=ollama_ndjson(({"done": True, "done_reason": "stop"},)))
        client = ollama_client(recorder)
        client.complete([LlmMessage.user("привет")])
        paths = [r.url.path for r in recorder.requests]
        assert paths == ["/api/version", "/api/chat"]
        body = recorder.body_of(recorder.request_to("/api/chat"))
        assert body["model"] == "llama3.2:1b"
        assert body["stream"] is True
        assert body["keep_alive"] == "5m"
        client.close()

    def test_down_server_gives_a_clear_message(self) -> None:
        client = ollama_client(OllamaRecorder(raise_on_version=True))
        with pytest.raises(LlmError) as excinfo:
            client.complete([LlmMessage.user("привет")])
        message = excinfo.value.user_message
        assert "не запущена" in message
        assert "ollama serve" in message
        client.close()

    def test_list_models_reads_the_tags(self) -> None:
        recorder = OllamaRecorder(
            tags={"models": [{"name": "llama3.2:1b"}, {"name": "qwen2.5:7b"}]}
        )
        client = ollama_client(recorder)
        assert isinstance(client, OllamaLlmClient)
        assert client.list_models() == ("llama3.2:1b", "qwen2.5:7b")
        client.close()

    def test_pull_streams_progress(self) -> None:
        recorder = OllamaRecorder(
            pull=ollama_ndjson(
                (
                    {"status": "pulling manifest"},
                    {"status": "downloading", "completed": 50, "total": 100},
                    {"status": "downloading", "completed": 100, "total": 100},
                    {"status": "success"},
                )
            )
        )
        client = ollama_client(recorder)
        assert isinstance(client, OllamaLlmClient)
        progress = list(client.pull("llama3.2:1b"))
        assert all(isinstance(item, OllamaPullProgress) for item in progress)
        assert [item.status for item in progress] == [
            "pulling manifest",
            "downloading",
            "downloading",
            "success",
        ]
        assert progress[0].fraction == 0.0
        assert progress[1].fraction == 0.5
        assert progress[2].fraction == 1.0
        client.close()

    def test_pull_refuses_a_model_over_the_ram_limit(self) -> None:
        recorder = OllamaRecorder()
        client = ollama_client(recorder, extra={"ram_limit_mb": 4096})
        assert isinstance(client, OllamaLlmClient)
        with pytest.raises(LlmError) as excinfo:
            list(client.pull("gemma2:9b"))
        assert "Gemma 2 9B" in excinfo.value.user_message
        assert recorder.requests == []
        client.close()


class TestLmStudio:
    """LM Studio, reached as an OpenAI-compatible local server."""

    def test_is_a_local_provider(self) -> None:
        assert is_local_provider("lmstudio")

    def test_is_an_openai_compatible_subclass(self) -> None:
        assert issubclass(LmStudioLlmClient, OpenAiCompatibleClient)

    def test_default_endpoint_when_base_is_empty(self) -> None:
        recorder = OpenAiRecorder()
        client = lmstudio_client(recorder, base_url="")
        assert client.configured is True
        list(client.stream([LlmMessage.user("привет")]))
        assert str(recorder.last.url) == "http://127.0.0.1:1234/v1/chat/completions"
        client.close()

    def test_v1_is_appended_when_missing(self) -> None:
        recorder = OpenAiRecorder()
        client = lmstudio_client(recorder, base_url="http://127.0.0.1:5000")
        list(client.stream([LlmMessage.user("привет")]))
        assert str(recorder.last.url) == "http://127.0.0.1:5000/v1/chat/completions"
        client.close()

    def test_explicit_v1_is_kept(self) -> None:
        recorder = OpenAiRecorder()
        client = lmstudio_client(recorder, base_url="http://127.0.0.1:7000/v1")
        list(client.stream([LlmMessage.user("привет")]))
        assert str(recorder.last.url) == "http://127.0.0.1:7000/v1/chat/completions"
        client.close()

    def test_no_auth_header_without_a_key(self) -> None:
        recorder = OpenAiRecorder()
        client = lmstudio_client(recorder, api_key="")
        list(client.stream([LlmMessage.user("привет")]))
        assert "authorization" not in recorder.last.headers
        client.close()

    def test_bearer_header_with_a_key(self) -> None:
        recorder = OpenAiRecorder()
        client = lmstudio_client(recorder, api_key=LMS_KEY)
        list(client.stream([LlmMessage.user("привет")]))
        assert recorder.last.headers["authorization"] == f"Bearer {LMS_KEY}"
        assert LMS_KEY not in repr(client)
        client.close()

    def test_answer_is_assembled_with_usage(self) -> None:
        recorder = OpenAiRecorder(sse=openai_sse(("Привет, ", "мир."), usage=(5, 3)))
        client = lmstudio_client(recorder)
        response = client.complete([LlmMessage.user("привет")])
        assert response.text == "Привет, мир."
        assert response.engine == "lmstudio"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 3
        client.close()

    def test_list_models_reads_the_catalogue(self) -> None:
        recorder = OpenAiRecorder(models={"data": [{"id": "local-a"}, {"id": "local-b"}]})
        client = lmstudio_client(recorder)
        assert client.list_models() == ("local-a", "local-b")
        assert recorder.last.url.path.endswith("/v1/models")
        client.close()


class TestCatalogue:
    """The recommended-models list and its verdicts — pure calls, no HTTP."""

    def test_recommended_is_ordered_lightest_first(self) -> None:
        rams = [spec.requires_ram_mb for spec in catalog.RECOMMENDED]
        assert rams == sorted(rams)
        assert len(catalog.RECOMMENDED) == 8

    def test_verdict_fits_when_ram_is_ample(self) -> None:
        phi3 = catalog.get_spec("phi3-mini")
        assert phi3 is not None
        assert catalog.verdict_for(phi3, 16384) is catalog.Verdict.FITS

    def test_verdict_tight_when_ram_eats_the_headroom(self) -> None:
        phi3 = catalog.get_spec("phi3-mini")
        assert phi3 is not None
        assert catalog.verdict_for(phi3, 5120) is catalog.Verdict.TIGHT

    def test_verdict_no_fit_when_model_is_larger_than_ram(self) -> None:
        phi3 = catalog.get_spec("phi3-mini")
        assert phi3 is not None
        assert catalog.verdict_for(phi3, 4096) is catalog.Verdict.NO_FIT

    def test_verdict_unknown_when_ram_reading_failed(self) -> None:
        phi3 = catalog.get_spec("phi3-mini")
        assert phi3 is not None
        assert catalog.verdict_for(phi3, 0) is catalog.Verdict.UNKNOWN

    def test_verdict_labels_are_russian(self) -> None:
        assert catalog.verdict_label(catalog.Verdict.FITS) == "влезет"
        assert catalog.verdict_label(catalog.Verdict.NO_FIT) == "не влезет"

    def test_get_spec_by_id_tag_and_whitespace(self) -> None:
        assert catalog.get_spec("llama3.2-1b") is not None
        assert catalog.get_spec("llama3.2:1b") is not None
        assert catalog.get_spec("  llama3.2:1b  ") is not None
        assert catalog.get_spec("does-not-exist") is None

    def test_for_engine_lists_ollama_and_skips_lmstudio(self) -> None:
        assert len(catalog.for_engine("ollama")) == 8
        assert catalog.for_engine("lmstudio") == ()

    def test_runs_on_is_case_insensitive(self) -> None:
        spec = catalog.get_spec("llama3.2-1b")
        assert spec is not None
        assert spec.runs_on("Ollama") is True
        assert spec.runs_on("lmstudio") is False

    def test_human_size_reads_gb_and_mb(self) -> None:
        gemma = catalog.get_spec("gemma2-9b")
        tiny = catalog.get_spec("llama3.2-1b")
        assert gemma is not None
        assert tiny is not None
        assert gemma.human_size == "≈5.6 ГБ"
        assert tiny.human_size == "≈808 МБ"

    def test_guard_ram_limit_allows_no_cap(self) -> None:
        gemma = catalog.get_spec("gemma2-9b")
        assert gemma is not None
        catalog.guard_ram_limit(gemma, ram_limit_mb=0)
        catalog.guard_ram_limit(gemma, ram_limit_mb=-1)

    def test_guard_ram_limit_allows_within_limit(self) -> None:
        gemma = catalog.get_spec("gemma2-9b")
        assert gemma is not None
        catalog.guard_ram_limit(gemma, ram_limit_mb=16384)

    def test_guard_ram_limit_refuses_over_limit(self) -> None:
        gemma = catalog.get_spec("gemma2-9b")
        assert gemma is not None
        with pytest.raises(LlmError) as excinfo:
            catalog.guard_ram_limit(gemma, ram_limit_mb=4096)
        assert "Gemma 2 9B" in excinfo.value.user_message


# ----------------------------------------------------------------------
# llama.cpp: an injected fake handle, so no native wheel or GGUF is needed
# ----------------------------------------------------------------------


class FakeChatModel:
    """Stands in for a loaded ``llama_cpp.Llama``: records calls, yields chunks."""

    def __init__(self, chunks: tuple[Mapping[str, Any], ...]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def create_chat_completion(
        self,
        *,
        messages: Sequence[JsonObject],
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> Iterator[Mapping[str, Any]]:
        self.calls.append(
            {
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
        )
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class RaisingChatModel:
    """A fake handle whose generation blows up, to prove the error is wrapped."""

    def create_chat_completion(
        self,
        *,
        messages: Sequence[JsonObject],
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> Iterator[Mapping[str, Any]]:
        raise RuntimeError("llama kernel died")


def text_chunk(content: str) -> dict[str, Any]:
    """One OpenAI-shaped streaming chunk carrying a text fragment."""
    return {"choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}


def finish_chunk(reason: str = "stop") -> dict[str, Any]:
    """The terminal OpenAI-shaped chunk carrying only a finish reason."""
    return {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}


def make_factory(handle: object) -> Callable[[LlamaCppOptions], object]:
    """A llama.cpp factory handing back ``handle`` instead of loading a GGUF."""

    def factory(options: LlamaCppOptions) -> object:
        return handle

    return factory


class TestLlamaCpp:
    """The in-process engine, driven through an injected fake factory."""

    def _client(
        self,
        handle: object,
        *,
        model_path: str = "model.gguf",
        model: str = "phi-3",
        temperature: float | None = None,
        max_tokens: int | None = None,
        idle_sec: float = 0.0,
        ram_limit_mb: int = 0,
        requires_ram_mb: int = 0,
    ) -> LlamaCppLlmClient:
        options = LlamaCppOptions(
            model_path=model_path,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            idle_sec=idle_sec,
            ram_limit_mb=ram_limit_mb,
            requires_ram_mb=requires_ram_mb,
        )
        return LlamaCppLlmClient(options, factory=make_factory(handle))

    def test_complete_reads_text_finish_and_engine(self) -> None:
        handle = FakeChatModel((text_chunk("При"), text_chunk("вет"), finish_chunk("stop")))
        client = self._client(handle, model="phi-3")
        response = client.complete([LlmMessage.user("привет")])
        assert response.text == "Привет"
        assert response.engine == "llamacpp"
        assert response.model == "phi-3"
        assert response.finish_reason is FinishReason.STOP

    def test_stream_yields_text_then_one_done(self) -> None:
        handle = FakeChatModel((text_chunk("а"), text_chunk("б"), finish_chunk("stop")))
        client = self._client(handle)
        deltas = list(client.stream([LlmMessage.user("привет")]))
        texts = [d.text for d in deltas if isinstance(d, LlmTextDelta)]
        dones = [d for d in deltas if isinstance(d, LlmDoneDelta)]
        assert "".join(texts) == "аб"
        assert len(dones) == 1
        assert dones[0].finish_reason is FinishReason.STOP

    def test_cancel_stops_with_a_cancelled_done(self) -> None:
        handle = FakeChatModel((text_chunk("а"), text_chunk("б"), finish_chunk("stop")))
        client = self._client(handle)
        deltas = list(client.stream([LlmMessage.user("привет")], cancel=lambda: True))
        assert isinstance(deltas[-1], LlmDoneDelta)
        assert deltas[-1].finish_reason is FinishReason.CANCELLED

    def test_temperature_override_beats_options_and_default(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, temperature=0.3)
        client.complete([LlmMessage.user("привет")], temperature=0.9)
        assert handle.calls[-1]["temperature"] == 0.9

    def test_temperature_falls_back_to_options(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, temperature=0.3)
        client.complete([LlmMessage.user("привет")])
        assert handle.calls[-1]["temperature"] == 0.3

    def test_temperature_falls_back_to_the_default(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, temperature=None)
        client.complete([LlmMessage.user("привет")])
        assert handle.calls[-1]["temperature"] == DEFAULT_TEMPERATURE

    def test_generation_failure_is_wrapped(self) -> None:
        client = self._client(RaisingChatModel())
        with pytest.raises(LlmError) as excinfo:
            client.complete([LlmMessage.user("привет")])
        assert "Локальная модель" in excinfo.value.user_message

    def test_unload_if_idle_frees_a_loaded_model(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, idle_sec=1.0)
        client.complete([LlmMessage.user("привет")])
        assert client.unload_if_idle(now=1e12) is True
        assert handle.closed is True

    def test_unload_if_idle_keeps_a_recent_model(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, idle_sec=1e9)
        client.complete([LlmMessage.user("привет")])
        assert client.unload_if_idle() is False
        assert handle.closed is False

    def test_unload_if_idle_without_a_handle_is_a_noop(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, idle_sec=1.0)
        assert client.unload_if_idle(now=1e12) is False

    def test_zero_window_never_unloads(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, idle_sec=0.0)
        client.complete([LlmMessage.user("привет")])
        assert client.unload_if_idle(now=1e12) is False

    def test_close_is_safe_twice_and_sets_closed(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle)
        client.complete([LlmMessage.user("привет")])
        client.close()
        client.close()
        assert handle.closed is True

    def test_message_when_no_model_path(self) -> None:
        options = LlamaCppOptions(model_path="")
        client = LlamaCppLlmClient(options, factory=make_factory(FakeChatModel(())))
        assert client.configured is False
        assert ".gguf" in client.message

    def test_message_when_ram_over_limit(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, ram_limit_mb=2048, requires_ram_mb=8000)
        assert client.configured is False
        assert "лимит памяти" in client.message

    def test_message_when_binding_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("ayris.nlu.llm.llamacpp_client.llama_cpp_available", lambda: False)
        client = LlamaCppLlmClient(LlamaCppOptions(model_path="model.gguf"))
        assert client.available() is False
        assert client.configured is False
        assert "llama-cpp-python" in client.message

    def test_stream_on_an_unconfigured_client_raises(self) -> None:
        handle = FakeChatModel((finish_chunk("stop"),))
        client = self._client(handle, ram_limit_mb=2048, requires_ram_mb=8000)
        with pytest.raises(LlmError):
            list(client.stream([LlmMessage.user("привет")]))


# ----------------------------------------------------------------------
# Router: a programmable stub client and a real bus + connectivity monitor
# ----------------------------------------------------------------------


class StubClient(LlmClient):
    """A programmable in-memory client for exercising the router's decisions."""

    name: ClassVar[str] = "stub"
    supports_streaming: ClassVar[bool] = True

    def __init__(
        self,
        *,
        text: str = "ответ",
        engine: str = "stub",
        error: LlmError | None = None,
        fails: int | None = None,
        error_after_first: bool = False,
    ) -> None:
        self.text = text
        self._engine = engine
        self._error = error
        self._fails = fails
        self._error_after_first = error_after_first
        self.completes = 0
        self.streams = 0
        self.closed = False

    def _should_fail(self) -> bool:
        if self._error is None:
            return False
        if self._fails is None:
            return True
        if self._fails > 0:
            self._fails -= 1
            return True
        return False

    def _raise(self) -> None:
        assert self._error is not None
        raise self._error

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        self.completes += 1
        if self._should_fail():
            self._raise()
        return LlmResponse(
            text=self.text,
            model=self._engine,
            engine=self._engine,
            finish_reason=FinishReason.STOP,
        )

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        self.streams += 1
        if not self._error_after_first and self._should_fail():
            self._raise()
        yield LlmTextDelta(text=self.text)
        if self._error_after_first and self._should_fail():
            self._raise()
        yield LlmDoneDelta(finish_reason=FinishReason.STOP)

    def close(self) -> None:
        self.closed = True


def make_bus() -> tuple[EventBus, list[NotificationRequested]]:
    """A synchronous bus and the list capturing every notification raised on it."""
    bus = EventBus(thread_id=None)
    notes: list[NotificationRequested] = []
    bus.subscribe(NotificationRequested, notes.append)
    return bus, notes


class TestRouter:
    """Mode dispatch, auto fallback, connectivity wiring and quiet recovery."""

    def test_mode_parse_defaults_to_auto(self) -> None:
        assert LlmMode.parse("offline") is LlmMode.OFFLINE
        assert LlmMode.parse("ONLINE") is LlmMode.ONLINE
        assert LlmMode.parse(" auto ") is LlmMode.AUTO
        assert LlmMode.parse("nonsense") is LlmMode.AUTO

    def test_offline_mode_uses_the_local_engine(self) -> None:
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(mode=LlmMode.OFFLINE, offline=lambda: offline)
        response = router.complete(MESSAGES)
        assert response.engine == "local"
        assert offline.completes == 1
        assert router.mode is LlmMode.OFFLINE
        router.close()

    def test_online_mode_uses_the_cloud(self) -> None:
        online = StubClient(text="облако", engine="cloud")
        router = LlmRouter(mode=LlmMode.ONLINE, online=lambda: online)
        assert router.complete(MESSAGES).engine == "cloud"
        router.close()

    def test_online_without_a_provider_raises(self) -> None:
        router = LlmRouter(mode=LlmMode.ONLINE)
        with pytest.raises(LlmError) as excinfo:
            router.complete(MESSAGES)
        assert "провайдер не настроен" in excinfo.value.user_message
        router.close()

    def test_offline_without_a_provider_raises(self) -> None:
        router = LlmRouter(mode=LlmMode.OFFLINE)
        with pytest.raises(LlmError) as excinfo:
            router.complete(MESSAGES)
        assert "Локальная модель не настроена" in excinfo.value.user_message
        router.close()

    def test_auto_prefers_the_cloud(self) -> None:
        bus, _ = make_bus()
        online = StubClient(text="облако", engine="cloud")
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO, online=lambda: online, offline=lambda: offline, bus=bus
        )
        assert router.complete(MESSAGES).engine == "cloud"
        assert offline.completes == 0
        router.close()

    def test_auto_complete_fails_over_to_local(self) -> None:
        bus, notes = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("network down"))
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        response = router.complete(MESSAGES)
        assert response.engine == "local"
        assert online.completes == 1
        assert offline.completes == 1
        assert router.using_offline is True
        assert monitor.online is False
        assert [n.title for n in notes].count("ИИ переключён") == 1
        router.complete(MESSAGES)
        assert offline.completes == 2
        assert [n.title for n in notes].count("ИИ переключён") == 1
        router.close()

    def test_connectivity_error_reports_offline(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("timeout"))
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        assert router.complete(MESSAGES).engine == "local"
        assert monitor.online is False
        router.close()

    def test_auth_error_falls_back_but_keeps_monitor_online(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmAuthError("bad key"))
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        assert router.complete(MESSAGES).engine == "local"
        assert monitor.online is True
        router.close()

    def test_quota_error_falls_back_but_keeps_monitor_online(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmQuotaError("spent"))
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        assert router.complete(MESSAGES).engine == "local"
        assert monitor.online is True
        router.close()

    def test_auto_stream_falls_back_before_the_first_token(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("down"))
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        deltas = list(router.stream(MESSAGES))
        text = "".join(d.text for d in deltas if isinstance(d, LlmTextDelta))
        assert text == "локально"
        assert router.using_offline is True
        assert offline.streams == 1
        router.close()

    def test_auto_stream_break_after_first_token_propagates(self) -> None:
        bus, _ = make_bus()
        online = StubClient(
            text="обл", engine="cloud", error=LlmError("broke"), error_after_first=True
        )
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO, online=lambda: online, offline=lambda: offline, bus=bus
        )
        with pytest.raises(LlmError):
            list(router.stream(MESSAGES))
        assert offline.streams == 0
        router.close()

    def test_auto_returns_to_cloud_after_recovery(self) -> None:
        bus, notes = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(text="облако", engine="cloud", error=LlmError("down"), fails=1)
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        assert router.complete(MESSAGES).engine == "local"
        assert router.using_offline is True
        monitor.report_success()
        assert router.using_offline is False
        assert router.complete(MESSAGES).engine == "cloud"
        titles = [n.title for n in notes]
        assert "ИИ переключён" in titles
        assert "Связь восстановлена" in titles
        router.close()

    def test_eco_mode_drops_the_local_engine_on_recovery(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("down"), fails=1)
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
            eco_mode=True,
        )
        router.complete(MESSAGES)
        assert offline.closed is False
        monitor.report_success()
        assert offline.closed is True
        router.close()

    def test_non_eco_keeps_the_local_engine_on_recovery(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("down"), fails=1)
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        router.complete(MESSAGES)
        monitor.report_success()
        assert offline.closed is False
        router.close()

    def test_auto_without_a_local_engine_raises_and_notifies(self) -> None:
        bus, notes = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("down", user_message="Облако легло."))
        router = LlmRouter(mode=LlmMode.AUTO, online=lambda: online, monitor=monitor, bus=bus)
        with pytest.raises(LlmError) as excinfo:
            router.complete(MESSAGES)
        assert "Локальной модели нет" in excinfo.value.user_message
        assert [n.title for n in notes].count("ИИ недоступен") == 1
        router.close()

    def test_close_unsubscribes_and_closes_built_clients(self) -> None:
        bus, _ = make_bus()
        monitor = ConnectivityMonitor(bus, probe_url="")
        online = StubClient(engine="cloud", error=LlmError("down"), fails=1)
        offline = StubClient(text="локально", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: online,
            offline=lambda: offline,
            monitor=monitor,
            bus=bus,
        )
        assert bus.subscriber_count(OnlineStatusChanged) == 1
        router.complete(MESSAGES)  # builds both clients through the fallback
        router.close()
        assert bus.subscriber_count(OnlineStatusChanged) == 0
        assert online.closed is True
        assert offline.closed is True
