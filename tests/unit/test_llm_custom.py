"""Task 61 (extension): the user-supplied OpenAI-compatible «custom» provider.

The six branded providers each pin a host; ``custom`` pins nothing, so any site
that speaks the OpenAI ``/chat/completions`` streaming schema — an OpenRouter-style
broker, Together, Groq, a self-hosted server — works once its endpoint, model and
key are filled in. These tests prove that end to end on ``httpx.MockTransport``:
no socket is opened, and the request the client *would* have sent is inspected as
an object.

Groups:

* :class:`TestRegistry` — ``custom`` is a cloud provider and resolves to its class.
* :class:`TestConfigured` — needs both a key and a base URL, unlike the branded six.
* :class:`TestStream` — a streamed answer, and the request built for it.
* :class:`TestBaseUrl` — the endpoint follows ``base_url``; a branded one ignores it.
* :class:`TestCatalogue` — ``list_models`` / ``check_credentials`` over the mock.
* :class:`TestCredentials` — the key comes from the store, by reference.
* :class:`TestWorker` — the worker rebuilds on an endpoint change and answers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from ayris.core.models import JsonObject
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.nlu.llm.base import FinishReason, LlmClient, LlmMessage
from ayris.nlu.llm.custom_client import CustomOpenAiLlmClient
from ayris.nlu.llm.factory import (
    CLOUD_PROVIDERS,
    CUSTOM_PROVIDER,
    create_llm_client,
    is_cloud_provider,
)
from ayris.workers.llm_worker import LlmWorker

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

#: The key every store-backed test looks for by value in the auth header.
KEY = "test-secret-key"

#: A base URL that is obviously not any branded provider's host.
BASE_URL = "https://gateway.example/api/v1"


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

    A test that passes ``api_key=""`` and no explicit ``store`` still resolves the
    key through :func:`~ayris.core.secrets.get_secrets`; without this it would
    query the real Windows Credential Manager. An empty fake keeps «no key»
    deterministic and touches nothing outside the process.
    """
    reset_secrets(SecretsStore("Ayris-test-empty", backend=FakeKeyring()))
    yield
    reset_secrets()


@pytest.fixture
def keyring_store() -> Iterator[SecretsStore]:
    """A process-wide store holding :data:`KEY` under every provider's ref."""
    store = SecretsStore("Ayris-test", backend=FakeKeyring())
    for ref in CLOUD_PROVIDERS:
        store.save(ref, KEY)
    reset_secrets(store)
    yield store
    reset_secrets()


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
    lines.append(
        f'data: {json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})}'
    )
    if usage is not None:
        prompt, completion = usage
        block = {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
        lines.append(f"data: {json.dumps(block)}")
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


class Recorder:
    """A :class:`httpx.MockTransport` handler that records and routes by path.

    Nothing here opens a socket: every request is answered from the bytes handed
    in at construction, and each request object is kept so a test can inspect the
    URL, headers and body that Ayris *would* have put on the wire.
    """

    def __init__(
        self,
        *,
        sse: bytes = b"",
        models: dict[str, Any] | None = None,
        chat_status: int = 200,
        models_status: int = 200,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._sse = sse or openai_sse(("ok",))
        self._models = models if models is not None else {"data": [{"id": "gw-mini"}]}
        self._chat_status = chat_status
        self._models_status = models_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(
                self._chat_status,
                content=self._sse,
                headers={"content-type": "text/event-stream"},
            )
        if path.endswith("/models"):
            return httpx.Response(self._models_status, json=self._models)
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


def client_for(
    recorder: Recorder,
    *,
    provider: str = CUSTOM_PROVIDER,
    base_url: str = BASE_URL,
    model: str = "gw-mini",
    api_key: str = KEY,
) -> LlmClient:
    """A client wired to ``recorder`` — the factory, no store, no network.

    The explicit ``api_key`` skips the credential store; the ``transport`` is the
    mock, so the client is fully built yet has touched nothing on the network.
    ``max_retries=0`` keeps a mocked failure from sleeping through backoff.
    """
    return create_llm_client(
        provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        transport=httpx.MockTransport(recorder),
        max_retries=0,
    )


class TestRegistry:
    """``custom`` is a cloud provider and resolves to its client class."""

    def test_custom_provider_constant(self) -> None:
        assert CUSTOM_PROVIDER == "custom"

    def test_custom_is_a_cloud_provider(self) -> None:
        assert is_cloud_provider("custom")
        assert is_cloud_provider("  Custom  ")

    def test_custom_is_registered_last(self) -> None:
        assert CUSTOM_PROVIDER in CLOUD_PROVIDERS
        assert list(CLOUD_PROVIDERS)[-1] == CUSTOM_PROVIDER

    def test_factory_builds_the_custom_client(self) -> None:
        client = create_llm_client("custom", base_url=BASE_URL, api_key=KEY)
        assert isinstance(client, CustomOpenAiLlmClient)
        client.close()


class TestConfigured:
    """Unlike the branded six, ``custom`` needs both a key and a base URL."""

    def test_key_without_endpoint_is_unconfigured(self) -> None:
        client = create_llm_client("custom", base_url="", api_key=KEY)
        assert client.configured is False
        client.close()

    def test_endpoint_without_key_is_unconfigured(self) -> None:
        client = create_llm_client("custom", base_url=BASE_URL, api_key="")
        assert client.configured is False
        client.close()

    def test_both_present_is_configured(self) -> None:
        client = create_llm_client("custom", base_url=BASE_URL, api_key=KEY)
        assert client.configured is True
        client.close()

    def test_model_is_not_required_for_configured(self) -> None:
        # list_models needs only key + URL; gating on model would break discovery.
        client = create_llm_client("custom", base_url=BASE_URL, api_key=KEY, model="")
        assert client.configured is True
        client.close()


class TestStream:
    """A streamed answer over the mock, and the request built for it."""

    def test_answer_is_assembled_from_the_stream(self) -> None:
        recorder = Recorder(sse=openai_sse(("При", "вет", ", мир"), usage=(12, 7)))
        client = client_for(recorder)
        response = client.complete([LlmMessage.user("Привет")])
        assert response.text == "Привет, мир"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 7
        client.close()

    def test_request_targets_the_custom_endpoint(self) -> None:
        recorder = Recorder()
        client = client_for(recorder)
        list(client.stream([LlmMessage.user("привет")]))
        request = recorder.last
        assert str(request.url) == f"{BASE_URL}/chat/completions"
        assert request.method == "POST"
        client.close()

    def test_request_carries_the_bearer_key(self) -> None:
        recorder = Recorder()
        client = client_for(recorder)
        list(client.stream([LlmMessage.user("привет")]))
        assert recorder.last.headers["authorization"] == f"Bearer {KEY}"
        client.close()

    def test_request_body_asks_for_the_model_and_usage(self) -> None:
        recorder = Recorder()
        client = client_for(recorder, model="gw-42b")
        list(client.stream([LlmMessage.user("привет")]))
        body = recorder.body_of(recorder.last)
        assert body["model"] == "gw-42b"
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        client.close()

    def test_the_key_never_appears_in_a_repr(self) -> None:
        recorder = Recorder()
        client = client_for(recorder)
        assert KEY not in repr(client)
        client.close()


class TestBaseUrl:
    """The endpoint follows ``base_url`` for custom; a branded one ignores it."""

    def test_custom_follows_the_supplied_base_url(self) -> None:
        other = "https://elsewhere.example/v9"
        recorder = Recorder()
        client = client_for(recorder, base_url=other)
        list(client.stream([LlmMessage.user("привет")]))
        assert str(recorder.last.url) == f"{other}/chat/completions"
        client.close()

    def test_branded_ignores_a_config_base_url(self) -> None:
        # Handing OpenAI a base URL must NOT redirect its traffic off its host.
        recorder = Recorder()
        client = client_for(recorder, provider="openai", base_url="https://evil.example/api")
        list(client.stream([LlmMessage.user("привет")]))
        host = recorder.last.url.host
        assert host == "api.openai.com"
        assert host != "evil.example"
        client.close()


class TestCatalogue:
    """``list_models`` / ``check_credentials`` over the mock, no real network."""

    def test_list_models_reads_the_catalogue(self) -> None:
        recorder = Recorder(models={"data": [{"id": "gw-mini"}, {"id": "gw-large"}]})
        client = client_for(recorder)
        assert client.list_models() == ("gw-mini", "gw-large")
        assert str(recorder.last.url) == f"{BASE_URL}/models"
        client.close()

    def test_check_credentials_ok(self) -> None:
        recorder = Recorder(models={"data": [{"id": "gw-mini"}]})
        client = client_for(recorder)
        check = client.check_credentials()
        assert check.ok is True
        assert check.models == ("gw-mini",)
        client.close()

    def test_check_credentials_without_a_key_makes_no_call(self) -> None:
        recorder = Recorder()
        client = client_for(recorder, api_key="")
        check = client.check_credentials()
        assert check.ok is False
        assert recorder.requests == []
        client.close()


class TestCredentials:
    """The key comes from the store, by reference — never from the config."""

    def test_key_is_resolved_from_the_store(self, keyring_store: SecretsStore) -> None:
        recorder = Recorder()
        client = create_llm_client(
            "custom",
            model="gw-mini",
            base_url=BASE_URL,
            credential_ref="custom",
            store=keyring_store,
            transport=httpx.MockTransport(recorder),
            max_retries=0,
        )
        assert client.configured is True
        list(client.stream([LlmMessage.user("привет")]))
        assert recorder.last.headers["authorization"] == f"Bearer {KEY}"
        client.close()


class FakeContext:
    """Enough of :class:`~ayris.workers.base.WorkerContext` to drive the worker.

    The worker's own process machinery is tested in :mod:`tests.unit.test_workers`;
    here the subject is the ``custom`` provider, so this is the same lightweight
    stand-in :mod:`tests.unit.test_stt_offline` uses — params in, events recorded,
    nothing that touches a pipe.
    """

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
    """A started worker on a :class:`FakeContext`, optionally with a client injected.

    When ``client`` is given it is planted so the worker keeps it instead of
    rebuilding — the key move is setting ``_client_key`` to the current identity,
    exactly as the supervisor's rebuild guard expects, so the mocked transport is
    never swapped out for a real one.
    """
    worker = LlmWorker(FakeContext(params))  # type: ignore[arg-type]
    worker.on_start()
    if client is not None:
        worker._client = client
        worker._client_key = worker._client_identity()
    return worker


class TestWorker:
    """The worker rebuilds on an endpoint change and answers over the mock."""

    def test_identity_includes_the_endpoint_for_custom(self) -> None:
        worker = make_worker(
            {"provider": "custom", "model": "gw-mini", "host": BASE_URL, "credential_ref": ""}
        )
        assert worker._client_identity() == ("custom", "gw-mini", "", BASE_URL)

    def test_identity_omits_the_endpoint_for_branded(self) -> None:
        worker = make_worker(
            {"provider": "openai", "model": "gpt-4o-mini", "host": BASE_URL, "credential_ref": ""}
        )
        assert worker._client_identity() == ("openai", "gpt-4o-mini", "", "")

    def test_changing_the_endpoint_changes_identity(self) -> None:
        worker = make_worker({"provider": "custom", "model": "gw", "host": BASE_URL})
        first = worker._client_identity()
        worker.context.params["host"] = "https://elsewhere.example/v1"
        assert worker._client_identity() != first

    def test_chat_streams_sentences_and_usage(self) -> None:
        recorder = Recorder(sse=openai_sse(("Привет, ", "мир."), usage=(5, 3)))
        worker = make_worker(
            {"provider": "custom", "model": "gw-mini", "host": BASE_URL, "credential_ref": ""},
            client=client_for(recorder),
        )
        reply = worker.chat(
            {"messages": [{"role": "user", "content": "привет"}], "request_id": "r1"}
        )
        assert reply["text"] == "Привет, мир."
        assert reply["engine"] == "custom"
        assert reply["model"] == "gw-mini"
        assert reply["finish_reason"] == FinishReason.STOP.value
        assert reply["cancelled"] is False
        context = worker.context
        sentences = context.events_of("sentence")  # type: ignore[attr-defined]
        assert sentences, "no sentence event was emitted"
        assert "".join(event["text"] for event in sentences) == "Привет, мир."
        assert sentences[-1]["final"] is True
        assert context.events_of("usage")  # type: ignore[attr-defined]
        assert reply["usage"]["prompt_tokens"] == 5
        worker.on_stop()

    def test_unconfigured_custom_answers_without_a_request(self) -> None:
        # No key: the client reports itself unconfigured and the worker answers
        # with the hint, never opening the stream — the zero-telemetry promise.
        recorder = Recorder()
        worker = make_worker(
            {"provider": "custom", "model": "gw-mini", "host": BASE_URL},
            client=client_for(recorder, api_key=""),
        )
        reply = worker.chat(
            {"messages": [{"role": "user", "content": "привет"}], "request_id": "r2"}
        )
        assert reply["configured"] is False
        assert reply["finish_reason"] == FinishReason.ERROR.value
        assert reply["engine"] == "custom"
        assert recorder.requests == [], "an unconfigured provider must make no call"
        worker.on_stop()
