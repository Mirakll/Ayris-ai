"""Задача 63: LLM-воркер и его опоры на моках — без сети, моделей и звука.

Здесь добираются ветки, до которых не доходят соседние наборы: клиент строится и
роняется по смене настроек, поток отдаёт предложения, tool-call и отмену, а разбор
пропущенной статистики, оценка токенов и таблица цен считаются на месте. Ни один
тест не открывает сокет и не грузит модель — клиенты заранее заготовлены, ключи
берутся из пустого хранилища в памяти, а llama.cpp собирается, но не загружается.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, cast

import pytest

from ayris.core.errors import LlmError
from ayris.core.events import (
    EventBus,
    LlmSentenceReady,
    LlmUsageReported,
    NotificationRequested,
    OnlineStatusChanged,
)
from ayris.core.models import JsonObject
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.nlu.llm import catalog
from ayris.nlu.llm.base import (
    CredentialCheck,
    FinishReason,
    LlmClient,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmResponse,
    LlmRole,
    LlmTextDelta,
    LlmTool,
    LlmToolCall,
    LlmToolCallDelta,
    LlmUsage,
    LlmUsageDelta,
    NullLlmClient,
)
from ayris.nlu.llm.factory import LLAMACPP_PROVIDER, create_llm_client
from ayris.nlu.llm.keys import resolve_api_key
from ayris.nlu.llm.llamacpp_client import LlamaCppLlmClient
from ayris.nlu.llm.router import LlmMode, LlmRouter
from ayris.nlu.llm.stream import SentenceAssembler
from ayris.nlu.llm.usage import (
    ModelPrice,
    UsageMeter,
    estimate_prompt_tokens,
    estimate_tokens,
    price_for,
    resolve_usage,
)
from ayris.workers.llm_worker import (
    LlmWorker,
    _as_float,
    _as_opt_float,
    _as_opt_int,
    _decode_tool_calls,
    _echo,
    _messages,
    _role_of,
    _tools,
    translate_llm_event,
)

pytestmark = pytest.mark.unit


# --- хранилище ключей и изоляция от настоящего --------------------------------


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


class RaisingKeyring:
    """A keyring whose reads always fail — the «store is locked» path of keys.py."""

    def get_password(self, service_name: str, username: str) -> str | None:
        raise RuntimeError("credential store is locked")

    def set_password(self, service_name: str, username: str, password: str) -> None:
        raise RuntimeError("credential store is locked")

    def delete_password(self, service_name: str, username: str) -> None:
        raise RuntimeError("credential store is locked")


@pytest.fixture(autouse=True)
def _isolate_secrets() -> Iterator[None]:
    """Point the process store at an empty in-memory keyring, never the real one."""
    reset_secrets(SecretsStore("Ayris-test-empty", backend=FakeKeyring()))
    yield
    reset_secrets()


# --- клиенты-заглушки (ничего не грузят, ничего не шлют) ----------------------


class ScriptedDeltaClient(LlmClient):
    """A client that yields a fixed script of deltas — drives the worker's stream."""

    name: ClassVar[str] = "scripted"
    supports_streaming: ClassVar[bool] = True
    supports_tools: ClassVar[bool] = True

    def __init__(self, deltas: Sequence[LlmDelta], *, configured: bool = True) -> None:
        self._deltas = tuple(deltas)
        self._configured = configured
        self.closed = False

    @property
    def configured(self) -> bool:
        return self._configured

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        del messages, tools, temperature, max_tokens, cancel
        return LlmResponse(finish_reason=FinishReason.STOP)

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        del messages, tools, temperature, max_tokens, cancel
        yield from self._deltas

    def close(self) -> None:
        self.closed = True


class CompleteOnlyClient(LlmClient):
    """A non-streaming client — exercises the base ``stream`` → ``_response_to_deltas``."""

    name: ClassVar[str] = "complete-only"

    def __init__(self, response: LlmResponse) -> None:
        self._response = response

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        del messages, tools, temperature, max_tokens, cancel
        return self._response


class RouterStubClient(LlmClient):
    """A programmable client for the router: answers, streams, or fails on demand."""

    name: ClassVar[str] = "stub"
    supports_streaming: ClassVar[bool] = True

    def __init__(
        self,
        *,
        text: str = "ответ",
        engine: str = "stub",
        error: LlmError | None = None,
    ) -> None:
        self.text = text
        self._engine = engine
        self._error = error
        self.closed = False
        self.completes = 0
        self.streams = 0

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        del messages, tools, temperature, max_tokens, cancel
        self.completes += 1
        if self._error is not None:
            raise self._error
        return LlmResponse(
            text=self.text, model=self._engine, engine=self._engine, finish_reason=FinishReason.STOP
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
        del messages, tools, temperature, max_tokens, cancel
        self.streams += 1
        if self._error is not None:
            raise self._error
        yield LlmTextDelta(text=self.text)
        yield LlmDoneDelta(finish_reason=FinishReason.STOP)

    def close(self) -> None:
        self.closed = True


# --- окружение воркера --------------------------------------------------------


class FakeContext:
    """Lightweight stand-in for :class:`~ayris.workers.base.WorkerContext`."""

    def __init__(self, params: JsonObject | None = None) -> None:
        self.name = "llm"
        self.kind = "llm"
        self._params: JsonObject = dict(params or {})
        self.events: list[tuple[str, JsonObject]] = []
        self.cancelled = False

    @property
    def params(self) -> JsonObject:
        return self._params

    @property
    def stopping(self) -> bool:
        return False

    def check_cancelled(self) -> None:
        return None

    def emit(self, kind: str, payload: JsonObject | None = None) -> None:
        self.events.append((kind, dict(payload or {})))

    def logger(self, suffix: str = "") -> Any:
        import logging

        return logging.getLogger(f"ayris.workers.llm.{suffix}" if suffix else "ayris.workers.llm")

    def events_of(self, kind: str) -> list[JsonObject]:
        """Every payload emitted under ``kind``, in order."""
        return [payload for name, payload in self.events if name == kind]


def make_worker(params: JsonObject, *, client: LlmClient | None = None) -> LlmWorker:
    """A started worker on a :class:`FakeContext`, optionally with a client injected."""
    worker = LlmWorker(FakeContext(params))  # type: ignore[arg-type]
    worker.on_start()
    if client is not None:
        worker._client = client
        worker._client_key = worker._client_identity()
    return worker


def ctx(worker: LlmWorker) -> FakeContext:
    """The worker's :class:`FakeContext`, typed so its recorder is reachable."""
    return cast(FakeContext, worker.context)


def stream_texts(router: LlmRouter) -> list[str]:
    """The text of every :class:`LlmTextDelta` the router streams for a fixed prompt."""
    deltas = router.stream([LlmMessage.user("q")])
    return [d.text for d in deltas if isinstance(d, LlmTextDelta)]


# --- контракт клиента (base.py) ----------------------------------------------


class TestBaseContract:
    """The dataclasses and the base ``LlmClient`` behaviours providers rely on."""

    def test_tool_message_payload_carries_name_and_call_id(self) -> None:
        payload = LlmMessage.tool("результат", tool_call_id="call-1", name="do_thing").as_payload()
        assert payload["role"] == "tool"
        assert payload["name"] == "do_thing"
        assert payload["tool_call_id"] == "call-1"

    def test_plain_message_payload_omits_empty_fields(self) -> None:
        payload = LlmMessage.user("привет").as_payload()
        assert "name" not in payload
        assert "tool_call_id" not in payload

    def test_tool_payload_uses_the_declared_schema(self) -> None:
        schema: JsonObject = {"type": "object", "properties": {"x": {"type": "integer"}}}
        payload = LlmTool(name="f", description="d", parameters=schema).as_payload()
        assert payload["function"]["parameters"] == schema

    def test_tool_payload_defaults_an_empty_schema(self) -> None:
        payload = LlmTool(name="f").as_payload()
        assert payload["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_usage_total_tokens_sums(self) -> None:
        assert LlmUsage(prompt_tokens=3, completion_tokens=4).total_tokens == 7

    def test_response_empty_and_cancelled_flags(self) -> None:
        assert LlmResponse().is_empty is True
        assert LlmResponse(text="   ").is_empty is True
        assert LlmResponse(text="да").is_empty is False
        assert LlmResponse(finish_reason=FinishReason.CANCELLED).cancelled is True
        assert LlmResponse().cancelled is False

    def test_base_stream_replays_a_full_response(self) -> None:
        response = LlmResponse(
            text="Привет.",
            tool_calls=(LlmToolCall(name="f", arguments={"a": 1}, call_id="c1"),),
            usage=LlmUsage(prompt_tokens=3, completion_tokens=4),
            finish_reason=FinishReason.TOOL_CALLS,
        )
        deltas = list(CompleteOnlyClient(response).stream([LlmMessage.user("q")]))
        first = deltas[0]
        assert isinstance(first, LlmTextDelta)
        assert first.text == "Привет."
        tool_deltas = [d for d in deltas if isinstance(d, LlmToolCallDelta)]
        assert len(tool_deltas) == 1
        assert tool_deltas[0].name == "f"
        assert json.loads(tool_deltas[0].arguments) == {"a": 1}
        assert any(isinstance(d, LlmUsageDelta) for d in deltas)
        last = deltas[-1]
        assert isinstance(last, LlmDoneDelta)
        assert last.finish_reason is FinishReason.TOOL_CALLS

    def test_base_stream_of_an_empty_response_is_just_done(self) -> None:
        deltas = list(CompleteOnlyClient(LlmResponse()).stream([LlmMessage.user("q")]))
        assert len(deltas) == 1
        assert isinstance(deltas[0], LlmDoneDelta)

    def test_default_list_models_and_check_credentials(self) -> None:
        client = CompleteOnlyClient(LlmResponse(text="x"))
        assert client.list_models() == ()
        assert client.check_credentials() == CredentialCheck(ok=True)

    def test_null_client_is_unconfigured_and_answers_the_hint(self) -> None:
        client = NullLlmClient()
        assert client.configured is False
        response = client.complete([LlmMessage.user("привет")])
        assert response.finish_reason is FinishReason.ERROR
        assert response.text == client.message

    def test_null_client_keeps_a_custom_message(self) -> None:
        assert NullLlmClient("совсем не настроено").complete([]).text == "совсем не настроено"


# --- ключ из хранилища (keys.py) ---------------------------------------------


class TestKeys:
    """A key handed over wins; a locked store is skipped, not raised."""

    def test_explicit_key_wins_without_touching_the_store(self) -> None:
        assert resolve_api_key("openai", explicit="  sk-abc  ") == "sk-abc"

    def test_locked_store_is_skipped_not_raised(self) -> None:
        store = SecretsStore("Ayris-test-locked", backend=RaisingKeyring())
        assert resolve_api_key("openai", store=store) == ""


# --- токены и цена (usage.py) ------------------------------------------------


class TestUsage:
    """Token estimation, the price table, the meter, and usage resolution."""

    def test_estimate_tokens_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_tokens_weights_cyrillic_heavier(self) -> None:
        assert estimate_tokens("Привет мир ё") >= 1
        assert estimate_tokens("абвгдеёжзи") > estimate_tokens("abcdefghij")

    def test_estimate_prompt_tokens_adds_per_message_overhead(self) -> None:
        messages = [LlmMessage.user("привет"), LlmMessage.assistant("hi")]
        floor = estimate_tokens("привет") + estimate_tokens("hi")
        assert estimate_prompt_tokens(messages) > floor

    def test_model_price_known_flag(self) -> None:
        assert ModelPrice().known is False
        assert ModelPrice(0.15, 0.60).known is True

    def test_price_for_exact_family_default_and_unknown(self) -> None:
        assert price_for("openai", "gpt-4o").known is True
        assert price_for("openai", "gpt-4o-mini-2024-07-18").prompt_per_1m == 0.15
        assert price_for("openai", "brand-new-model").known is True
        assert price_for("gigachat", "GigaChat").known is False
        assert price_for("nonesuch", "whatever").known is False

    def test_usage_meter_accumulates_and_snapshots(self) -> None:
        meter = UsageMeter()
        first = meter.record("openai", "gpt-4o-mini", LlmUsage(1000, 500), request_id="r1")
        second = meter.record("openai", "gpt-4o-mini", LlmUsage(200, 100), estimated=True)
        assert meter.requests == 2
        assert meter.total_tokens == 1800
        assert meter.session_cost_usd > 0.0
        assert second.session_cost_usd >= first.cost_usd
        snapshot = meter.snapshot()
        assert snapshot["requests"] == 2
        assert snapshot["total_tokens"] == 1800
        assert snapshot["prompt_tokens"] == 1200

    def test_resolve_usage_trusts_reported_counts(self) -> None:
        resolved, estimated = resolve_usage(
            "openai", LlmUsage(10, 5), [LlmMessage.user("q")], "answer"
        )
        assert resolved == LlmUsage(10, 5)
        assert estimated is False

    def test_resolve_usage_estimates_when_counts_missing(self) -> None:
        resolved, estimated = resolve_usage(
            "openai", LlmUsage(), [LlmMessage.user("привет")], "ответ"
        )
        assert estimated is True
        assert resolved.prompt_tokens > 0
        assert resolved.completion_tokens > 0


# --- сборка предложений (stream.py) ------------------------------------------


class TestSentenceAssembler:
    """The incremental sentence release TTS starts speaking on."""

    def test_empty_feed_returns_nothing(self) -> None:
        assembler = SentenceAssembler()
        assert assembler.feed("") == []
        assert assembler.emitted_count == 0

    def test_sentences_release_as_they_confirm(self) -> None:
        assembler = SentenceAssembler()
        assert assembler.feed("Привет всем. ") == []
        assert assembler.feed("Как дела сегодня. ") == ["Привет всем."]
        assert assembler.emitted_count == 1
        assert assembler.flush() == ["Как дела сегодня."]


# --- объём памяти (catalog.py) -----------------------------------------------


class TestCatalog:
    """``detect_total_ram_mb`` reports a number, or degrades to zero."""

    def test_detect_total_ram_reports_a_nonnegative_number(self) -> None:
        assert catalog.detect_total_ram_mb() >= 0

    def test_detect_total_ram_degrades_to_zero_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import psutil

        def boom() -> object:
            raise RuntimeError("psutil refused")

        monkeypatch.setattr(psutil, "virtual_memory", boom)
        assert catalog.detect_total_ram_mb() == 0


# --- фабрика клиентов (factory.py) -------------------------------------------


class TestFactory:
    """The llama.cpp dispatch, its option helpers, and the unknown-provider guard."""

    def test_llamacpp_client_is_built_from_extra(self) -> None:
        client = create_llm_client(
            LLAMACPP_PROVIDER,
            model="local-gguf",
            extra={
                "model_path": "C:/models/x.gguf",
                "n_ctx": 2048,
                "n_threads": 4,
                "n_gpu_layers": 10,
                "idle_sec": 30.0,
                "ram_limit_mb": 8000,
                "requires_ram_mb": 4000,
            },
        )
        try:
            assert isinstance(client, LlamaCppLlmClient)
        finally:
            client.close()

    def test_llamacpp_client_tolerates_missing_or_wrong_typed_extra(self) -> None:
        client = create_llm_client(
            LLAMACPP_PROVIDER,
            extra={"n_ctx": "nope", "idle_sec": None, "n_threads": "x"},
        )
        try:
            assert isinstance(client, LlamaCppLlmClient)
        finally:
            client.close()

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(LlmError):
            create_llm_client("definitely-not-a-provider")


# --- чистые помощники воркера (llm_worker.py) --------------------------------


class TestWorkerHelpers:
    """The module-level parsers and translators the worker leans on."""

    def test_messages_skips_malformed_turns_and_maps_roles(self) -> None:
        params: JsonObject = {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "bogus", "content": "b"},
                "not a dict",
                {"role": "tool", "content": "t", "name": "fn", "tool_call_id": "c1"},
            ]
        }
        messages = _messages(params)
        assert [m.role for m in messages] == [LlmRole.SYSTEM, LlmRole.USER, LlmRole.TOOL]
        assert messages[-1].tool_call_id == "c1"

    def test_messages_of_a_non_list_is_empty(self) -> None:
        assert _messages({"messages": "nope"}) == []

    def test_tools_skips_nameless_and_non_dict(self) -> None:
        params: JsonObject = {
            "tools": [
                {"name": "vol", "description": "d", "parameters": {"type": "object"}},
                {"name": ""},
                {"description": "no name"},
                "not a dict",
                {"name": "plain", "parameters": "not-a-dict"},
            ]
        }
        tools = _tools(params)
        assert [t.name for t in tools] == ["vol", "plain"]
        assert tools[0].parameters == {"type": "object"}
        assert tools[1].parameters == {}

    def test_tools_of_a_non_list_is_empty(self) -> None:
        assert _tools({}) == []

    def test_role_of_defaults_to_user(self) -> None:
        assert _role_of("assistant") is LlmRole.ASSISTANT
        assert _role_of("garbage") is LlmRole.USER

    def test_decode_tool_calls_skips_nameless_fragments(self) -> None:
        frags = {
            1: {"id": "c1", "name": "do", "args": '{"x": 1}'},
            0: {"id": "", "name": "", "args": ""},
        }
        calls = _decode_tool_calls(frags)
        assert len(calls) == 1
        assert calls[0]["name"] == "do"
        assert calls[0]["arguments"] == {"x": 1}

    def test_numeric_helpers_read_and_default(self) -> None:
        assert _as_opt_float(1) == 1.0
        assert _as_opt_float(True) is None
        assert _as_opt_float("x") is None
        assert _as_opt_int(2.0) == 2
        assert _as_opt_int(None) is None
        assert _as_float(3, 9.0) == 3.0
        assert _as_float("nope", 9.0) == 9.0

    def test_echo_carries_request_id_only_when_present(self) -> None:
        assert _echo({"request_id": "r1"}) == {"request_id": "r1"}
        assert _echo({}) == {}

    def test_translate_llm_event_maps_sentence_and_usage(self) -> None:
        sentence = translate_llm_event(
            "sentence",
            {"text": "привет", "index": 2, "final": True, "engine": "custom", "request_id": "r1"},
        )
        assert isinstance(sentence, LlmSentenceReady)
        assert sentence.text == "привет"
        assert sentence.final is True
        usage = translate_llm_event(
            "usage",
            {
                "provider": "openai",
                "model": "gpt-4o",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "cost_usd": 0.1,
                "session_cost_usd": 0.2,
                "estimated": True,
                "request_id": "r1",
            },
        )
        assert isinstance(usage, LlmUsageReported)
        assert usage.provider == "openai"
        assert translate_llm_event("other", {}) is None
        assert translate_llm_event("sentence", {"text": ""}) is None


# --- жизненный цикл воркера (llm_worker.py) ----------------------------------


class TestWorkerLifecycle:
    """Configure, close, status, catalogue and build, all without a socket."""

    def test_on_configure_drops_client_when_identity_changes(self) -> None:
        client = ScriptedDeltaClient([LlmDoneDelta()])
        worker = make_worker(
            {"provider": "custom", "model": "a", "host": "http://h"}, client=client
        )
        worker.context.params["model"] = "b"
        worker.on_configure({})
        assert client.closed is True
        assert worker._client is None

    def test_on_configure_keeps_client_when_identity_stable(self) -> None:
        client = ScriptedDeltaClient([LlmDoneDelta()])
        worker = make_worker(
            {"provider": "custom", "model": "a", "host": "http://h"}, client=client
        )
        worker.on_configure({})
        assert client.closed is False
        assert worker._client is client
        worker.on_stop()

    def test_on_configure_without_a_client_is_a_noop(self) -> None:
        worker = make_worker({"provider": "custom", "model": "a"})
        worker.on_configure({})
        assert worker._client is None

    def test_close_client_is_safe_without_one(self) -> None:
        worker = make_worker({"provider": "custom"})
        worker._close_client()
        assert worker._client is None

    def test_close_client_swallows_a_failing_close(self) -> None:
        class BadClient(ScriptedDeltaClient):
            def close(self) -> None:
                raise RuntimeError("close blew up")

        worker = make_worker(
            {"provider": "custom", "model": "a"}, client=BadClient([LlmDoneDelta()])
        )
        worker._close_client()
        assert worker._client is None

    def test_status_without_a_client(self) -> None:
        worker = make_worker({"provider": "openai", "model": "gpt-4o"})
        status = worker.status({})
        assert status["provider"] == "openai"
        assert status["configured"] is False
        assert status["calls"] == 0
        assert status["requests"] == 0

    def test_status_with_a_client(self) -> None:
        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=ScriptedDeltaClient([LlmDoneDelta()])
        )
        assert worker.status({})["configured"] is True
        worker.on_stop()

    def test_list_models_success(self) -> None:
        class Lister(ScriptedDeltaClient):
            def list_models(self) -> tuple[str, ...]:
                return ("m1", "m2")

        worker = make_worker({"provider": "custom", "model": "gw"}, client=Lister([LlmDoneDelta()]))
        reply = worker.list_models({})
        assert reply["models"] == ["m1", "m2"]
        assert reply["provider"] == "custom"
        worker.on_stop()

    def test_list_models_handles_llm_error(self) -> None:
        class Failing(ScriptedDeltaClient):
            def list_models(self) -> tuple[str, ...]:
                raise LlmError("no catalogue", user_message="нет каталога")

        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=Failing([LlmDoneDelta()])
        )
        assert worker.list_models({})["models"] == []
        worker.on_stop()

    def test_check_key_reads_the_client_verdict(self) -> None:
        class Checker(ScriptedDeltaClient):
            def check_credentials(self) -> CredentialCheck:
                return CredentialCheck(ok=True, detail="ok", models=("m1",))

        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=Checker([LlmDoneDelta()])
        )
        reply = worker.check_key({})
        assert reply["ok"] is True
        assert reply["models"] == ["m1"]
        worker.on_stop()

    def test_cancel_sets_the_flag_and_echoes_request_id(self) -> None:
        worker = make_worker({"provider": "custom"})
        reply = worker.cancel({"request_id": "r9"})
        assert reply["cancelled"] is True
        assert reply["request_id"] == "r9"
        assert worker._should_cancel() is True

    def test_build_client_degrades_to_null_on_unknown_provider(self) -> None:
        worker = make_worker({"provider": "definitely-not-real", "model": "x"})
        client = worker._ensure_client()
        assert isinstance(client, NullLlmClient)
        assert client.configured is False
        worker.on_stop()

    def test_build_client_is_unconfigured_without_a_key(self) -> None:
        worker = make_worker(
            {
                "provider": "custom",
                "model": "gw",
                "host": "https://gateway.example/v1",
                "extra": {"folder_id": "x"},
            }
        )
        client = worker._ensure_client()
        assert client.configured is False
        worker.on_stop()


# --- поток ответа воркера (llm_worker.py) ------------------------------------


class TestWorkerStreaming:
    """The ``chat`` stream loop: sentences, tool-call fragments, cancel, estimation."""

    def test_stream_emits_sentences_tool_calls_and_usage(self) -> None:
        deltas: list[LlmDelta] = [
            LlmTextDelta(text="Привет всем. "),
            LlmTextDelta(text="Как дела сегодня. "),
            LlmToolCallDelta(index=0, call_id="c1", name="do_thing", arguments='{"x":'),
            LlmToolCallDelta(index=0, arguments="1}"),
            LlmToolCallDelta(index=1),
            LlmUsageDelta(usage=LlmUsage(prompt_tokens=11, completion_tokens=5)),
            LlmDoneDelta(finish_reason=FinishReason.TOOL_CALLS),
        ]
        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=ScriptedDeltaClient(deltas)
        )
        reply = worker.chat(
            {"messages": [{"role": "user", "content": "давай"}], "request_id": "r1"}
        )
        assert reply["finish_reason"] == FinishReason.TOOL_CALLS.value
        assert reply["cancelled"] is False
        assert reply["text"] == "Привет всем. Как дела сегодня. "
        assert len(reply["tool_calls"]) == 1
        assert reply["tool_calls"][0]["name"] == "do_thing"
        assert reply["tool_calls"][0]["arguments"] == {"x": 1}
        sentences = ctx(worker).events_of("sentence")
        assert [s["text"] for s in sentences] == ["Привет всем.", "Как дела сегодня."]
        assert sentences[-1]["final"] is True
        assert reply["usage"]["prompt_tokens"] == 11
        worker.on_stop()

    def test_cancelled_stream_skips_the_tail(self) -> None:
        deltas: list[LlmDelta] = [
            LlmTextDelta(text="частичный ответ без конца"),
            LlmDoneDelta(finish_reason=FinishReason.CANCELLED),
        ]
        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=ScriptedDeltaClient(deltas)
        )
        reply = worker.chat({"messages": [{"role": "user", "content": "стоп"}], "request_id": "r2"})
        assert reply["cancelled"] is True
        assert ctx(worker).events_of("sentence") == []
        worker.on_stop()

    def test_missing_usage_is_estimated(self) -> None:
        deltas: list[LlmDelta] = [
            LlmTextDelta(text="Короткий ответ."),
            LlmDoneDelta(finish_reason=FinishReason.STOP),
        ]
        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=ScriptedDeltaClient(deltas)
        )
        reply = worker.chat(
            {"messages": [{"role": "user", "content": "вопрос"}], "request_id": "r3"}
        )
        usage_events = ctx(worker).events_of("usage")
        assert usage_events
        assert usage_events[-1]["estimated"] is True
        assert reply["usage"]["prompt_tokens"] > 0
        assert reply["usage"]["completion_tokens"] > 0
        worker.on_stop()

    def test_stream_keeps_draining_after_a_done_delta(self) -> None:
        # A stray delta trailing the done marker is still drained: the loop reads
        # it and the last finish reason wins, so the tail cannot wedge the worker.
        deltas: list[LlmDelta] = [
            LlmDoneDelta(finish_reason=FinishReason.STOP),
            LlmTextDelta(text="хвост после конца"),
            LlmDoneDelta(finish_reason=FinishReason.LENGTH),
        ]
        worker = make_worker(
            {"provider": "custom", "model": "gw"}, client=ScriptedDeltaClient(deltas)
        )
        reply = worker.chat({"messages": [{"role": "user", "content": "ещё"}], "request_id": "r4"})
        assert reply["finish_reason"] == FinishReason.LENGTH.value
        assert reply["text"] == "хвост после конца"
        worker.on_stop()


# --- маршрутизатор (router.py) -----------------------------------------------


class TestRouter:
    """Preload, the three stream modes, and the connectivity callbacks."""

    def test_preload_builds_the_local_client_once(self) -> None:
        builds: list[RouterStubClient] = []

        def offline() -> LlmClient:
            client = RouterStubClient(engine="local")
            builds.append(client)
            return client

        router = LlmRouter(mode=LlmMode.AUTO, offline=offline)
        router.preload()
        if router._preload is not None:
            router._preload.join(timeout=5.0)
        assert len(builds) == 1
        router.preload()
        assert len(builds) == 1
        router.close()

    def test_preload_swallows_a_provider_error(self) -> None:
        def offline() -> LlmClient:
            raise LlmError("no local model", user_message="нет модели")

        router = LlmRouter(mode=LlmMode.AUTO, offline=offline)
        router.preload()
        if router._preload is not None:
            router._preload.join(timeout=5.0)
        assert router.using_offline is False
        router.close()

    def test_preload_is_skipped_outside_auto(self) -> None:
        router = LlmRouter(mode=LlmMode.ONLINE, offline=lambda: RouterStubClient())
        router.preload()
        assert router._preload is None
        router.close()

    def test_preload_does_not_start_a_second_thread_while_one_runs(self) -> None:
        release = threading.Event()
        builds: list[RouterStubClient] = []

        def offline() -> LlmClient:
            release.wait(timeout=5.0)
            client = RouterStubClient(engine="local")
            builds.append(client)
            return client

        router = LlmRouter(mode=LlmMode.AUTO, offline=offline)
        router.preload()
        first = router._preload
        router.preload()  # thread still blocked on the event → no second thread
        assert router._preload is first
        release.set()
        if first is not None:
            first.join(timeout=5.0)
        assert len(builds) == 1
        router.close()

    def test_stream_offline_mode(self) -> None:
        client = RouterStubClient(text="локальный", engine="local")
        router = LlmRouter(mode=LlmMode.OFFLINE, offline=lambda: client)
        texts = stream_texts(router)
        assert texts == ["локальный"]
        assert client.streams == 1
        router.close()

    def test_stream_online_mode(self) -> None:
        router = LlmRouter(mode=LlmMode.ONLINE, online=lambda: RouterStubClient(text="облачный"))
        texts = stream_texts(router)
        assert texts == ["облачный"]
        router.close()

    def test_stream_auto_prefers_offline_after_fallback(self) -> None:
        online = RouterStubClient(error=LlmError("network down"))
        offline = RouterStubClient(text="локальный", engine="local")
        router = LlmRouter(mode=LlmMode.AUTO, online=lambda: online, offline=lambda: offline)
        assert router.complete([LlmMessage.user("q")]).text == "локальный"
        assert router.using_offline is True
        texts = stream_texts(router)
        assert texts == ["локальный"]
        router.close()

    def test_stream_auto_success_notes_the_cloud(self) -> None:
        router = LlmRouter(
            mode=LlmMode.AUTO,
            online=lambda: RouterStubClient(text="облачный"),
            offline=lambda: RouterStubClient(),
        )
        texts = stream_texts(router)
        assert texts == ["облачный"]
        assert router.using_offline is False
        router.close()

    def test_fallback_without_a_bus_does_not_notify(self) -> None:
        online = RouterStubClient(error=LlmError("network down"))
        offline = RouterStubClient(text="локальный", engine="local")
        router = LlmRouter(
            mode=LlmMode.AUTO, online=lambda: online, offline=lambda: offline, bus=None
        )
        assert router.complete([LlmMessage.user("q")]).text == "локальный"
        router.close()

    def test_online_status_ignores_an_offline_event(self) -> None:
        bus = EventBus()
        router = LlmRouter(mode=LlmMode.AUTO, online=lambda: RouterStubClient(), bus=bus)
        bus.publish(OnlineStatusChanged(online=False))
        assert router.using_offline is False
        router.close()

    def test_online_status_when_not_using_offline_returns(self) -> None:
        bus = EventBus()
        router = LlmRouter(mode=LlmMode.AUTO, online=lambda: RouterStubClient(), bus=bus)
        bus.publish(OnlineStatusChanged(online=True))
        assert router.using_offline is False
        router.close()

    def test_fallback_notifies_once_then_stays_quiet(self) -> None:
        # The first switch to local announces itself; a second fallback while
        # already offline must not repeat the banner (the ``378->388`` shortcut).
        bus = EventBus()
        notices: list[NotificationRequested] = []
        bus.subscribe(NotificationRequested, notices.append, weak=False)
        offline = RouterStubClient(text="локальный", engine="local")
        router = LlmRouter(mode=LlmMode.AUTO, offline=lambda: offline, bus=bus)
        first = router._begin_fallback(LlmError("network down"))
        assert router.using_offline is True
        assert len(notices) == 1
        second = router._begin_fallback(LlmError("network down"))
        assert first is second is offline
        assert len(notices) == 1
        router.close()
