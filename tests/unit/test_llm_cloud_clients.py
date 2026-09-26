"""Брендовые облачные LLM-клиенты на замоканном HTTP, без единого сокета.

Каждый провайдер говорит по HTTP и стримит ответ, но по-своему: OpenAI, DeepSeek
и OpenRouter кадрируют поток одинаково (`data: {...}` c `choices[0].delta`),
Anthropic шлёт типизированные события своего Messages API, а YandexGPT — построчный
JSON, где каждый чанк несёт ответ целиком, а не последний фрагмент. Здесь эти три
формы прогоняются через :class:`httpx.MockTransport`: ни один запрос не уходит в
сеть, а тот, что клиент *собрался* отправить, разбирается как объект.

Группы:

* :class:`TestOpenAiCompatible` — openai/deepseek/openrouter: сборка ответа,
  инструменты, usage, пины хоста и заголовков, каталог и проверка ключа.
* :class:`TestErrorMapping` — статусы и сетевые сбои ложатся на типизированные
  ошибки; повтор до первого токена и его отсутствие после.
* :class:`TestCatalogueErrors` — сбои ``list_models`` / ``check_credentials``.
* :class:`TestAnthropic` — Messages API: система как верхнее поле, инструменты,
  usage из ``message_start``/``message_delta``, заголовок ``x-api-key``.
* :class:`TestYandex` — completion API: ``gpt://``-URI, дифф накопительных чанков,
  ``Api-Key``-заголовок, ``list_models`` без сети, ``check_credentials`` по токенайзеру.
* :class:`TestVerify` — брендовый клиент проверяет TLS по системным корням.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from ayris.core.errors import (
    LlmAuthError,
    LlmContextOverflowError,
    LlmError,
    LlmQuotaError,
)
from ayris.core.models import JsonObject
from ayris.core.secrets import SecretsStore, reset_secrets
from ayris.nlu.llm.base import (
    FinishReason,
    LlmClient,
    LlmDoneDelta,
    LlmMessage,
    LlmTextDelta,
    LlmTool,
)
from ayris.nlu.llm.cloud import CloudOptions
from ayris.nlu.llm.factory import create_llm_client
from ayris.nlu.llm.openai_client import OpenAiLlmClient

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

pytestmark = pytest.mark.unit

#: Ключ, который store-независимые тесты ищут по значению в заголовке авторизации.
KEY = "test-secret-key"


class FakeKeyring:
    """In-memory-замена Windows Credential Manager — пустой и офлайновый."""

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
    """Направить хранилище процесса на пустой фейковый keyring, не на настоящий.

    Тест с ``api_key=""`` резолвит ключ через :func:`get_secrets`; без изоляции он
    бы опросил реальный Credential Manager. Пустой фейк делает «ключа нет»
    детерминированным и не трогает ничего вне процесса.
    """
    reset_secrets(SecretsStore("Ayris-test-empty", backend=FakeKeyring()))
    yield
    reset_secrets()


# ----------------------------------------------------------------------------
# построители тел ответа — байты, точно повторяющие каждый провод
# ----------------------------------------------------------------------------


def _data_lines(events: Iterable[JsonObject], *, done: bool = False) -> bytes:
    """Собрать SSE-тело из ``data: {json}``-строк, при ``done`` добавив терминатор."""
    lines = [f"data: {json.dumps(event, ensure_ascii=False)}" for event in events]
    if done:
        lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode("utf-8")


def openai_sse(
    chunks: tuple[str, ...],
    *,
    finish: str = "stop",
    usage: tuple[int, int] | None = (12, 7),
    done: bool = True,
) -> bytes:
    """OpenAI-кадрирование: фреймы текста, фрейм finish, usage, при ``done`` — DONE."""
    events: list[JsonObject] = [
        {"choices": [{"index": 0, "delta": {"content": chunk}}]} for chunk in chunks
    ]
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage is not None:
        prompt, completion = usage
        events.append(
            {"choices": [], "usage": {"prompt_tokens": prompt, "completion_tokens": completion}}
        )
    return _data_lines(events, done=done)


def openai_tool_sse(
    *,
    name: str,
    call_id: str,
    arg_fragments: tuple[str, ...],
    finish: str = "tool_calls",
) -> bytes:
    """OpenAI-стрим одного tool-call: открытие с id+именем, затем фрагменты аргументов."""
    events: list[JsonObject] = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": call_id, "function": {"name": name, "arguments": ""}}
                        ]
                    },
                }
            ]
        }
    ]
    for fragment in arg_fragments:
        events.append(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": fragment}}]
                        },
                    }
                ]
            }
        )
    events.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    return _data_lines(events, done=True)


def anthropic_sse(
    *,
    text_parts: tuple[str, ...] = (),
    input_tokens: int = 0,
    output_tokens: int = 0,
    stop_reason: str = "end_turn",
    tool: dict[str, Any] | None = None,
) -> bytes:
    """Anthropic Messages: message_start → (tool)/текст-дельты → message_delta → stop."""
    events: list[JsonObject] = [
        {"type": "message_start", "message": {"usage": {"input_tokens": input_tokens}}}
    ]
    if tool is not None:
        events.append(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": tool["id"], "name": tool["name"]},
            }
        )
        for fragment in tool["args"]:
            events.append(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                }
            )
    for part in text_parts:
        events.append(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": part},
            }
        )
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": output_tokens},
        }
    )
    events.append({"type": "message_stop"})
    return _data_lines(events)


def yandex_ndjson(
    steps: Sequence[tuple[str, str]],
    *,
    usage: tuple[int, int] | None = None,
) -> bytes:
    """YandexGPT-стрим: построчный JSON, каждый чанк несёт накопленный текст и статус."""
    lines: list[str] = []
    for index, (text, status) in enumerate(steps):
        result: JsonObject = {"alternatives": [{"message": {"text": text}, "status": status}]}
        if usage is not None and index == len(steps) - 1:
            prompt, completion = usage
            result["usage"] = {"inputTextTokens": str(prompt), "completionTokens": str(completion)}
        lines.append(json.dumps({"result": result}, ensure_ascii=False))
    return ("\n".join(lines) + "\n").encode("utf-8")


# ----------------------------------------------------------------------------
# обработчики MockTransport — записывают запрос и отвечают из памяти
# ----------------------------------------------------------------------------


class Recorder:
    """Обработчик :class:`httpx.MockTransport`: маршрутизация по суффиксу пути.

    Ничего не открывает сокет: chat-подобный путь (``/chat/completions``,
    ``/messages``, ``/completion``) отдаёт ``stream``, ``/models`` — каталог,
    ``/tokenize`` — заглушку. Каждый запрос сохраняется, чтобы тест мог осмотреть
    URL, заголовки и тело, которые Ayris *собрался* положить на провод.
    """

    def __init__(
        self,
        *,
        stream: bytes = b"",
        chat_status: int = 200,
        models: JsonObject | None = None,
        models_raw: bytes | None = None,
        models_status: int = 200,
        tokenize_status: int = 200,
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._stream = stream
        self._chat_status = chat_status
        self._models = models
        self._models_raw = models_raw
        self._models_status = models_status
        self._tokenize_status = tokenize_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/models"):
            if self._models_raw is not None:
                return httpx.Response(self._models_status, content=self._models_raw)
            body = self._models if self._models is not None else {"data": []}
            return httpx.Response(self._models_status, json=body)
        if path.endswith("/tokenize"):
            return httpx.Response(self._tokenize_status, json={"tokens": ["x"]})
        return httpx.Response(
            self._chat_status,
            content=self._stream,
            headers={"content-type": "text/event-stream"},
        )

    @property
    def last(self) -> httpx.Request:
        """Самый свежий запрос, который увидел транспорт."""
        assert self.requests, "ни один запрос не дошёл до транспорта"
        return self.requests[-1]

    def body_of(self, request: httpx.Request) -> JsonObject:
        """JSON-тело, которое Ayris отправил, разобранное как объект."""
        decoded: JsonObject = json.loads(request.content)
        return decoded


class SequenceRecorder:
    """Отдаёт ответы шаг за шагом: для проверок повтора до первого токена."""

    def __init__(self, steps: Sequence[Callable[[], httpx.Response]]) -> None:
        self.requests: list[httpx.Request] = []
        self._steps = list(steps)
        self._index = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        step = self._steps[min(self._index, len(self._steps) - 1)]
        self._index += 1
        return step()


class RaisingRecorder:
    """Возбуждает сетевую ошибку на любом запросе — имитация недоступной сети."""

    def __init__(self, exc: httpx.TransportError) -> None:
        self.requests: list[httpx.Request] = []
        self._exc = exc

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        raise self._exc


@pytest.fixture
def make_client() -> Iterator[Callable[..., LlmClient]]:
    """Фабрика клиентов на замоканном транспорте; все закрываются на teardown.

    Закрытие через фикстуру гарантирует, что httpx не предупредит о незакрытом
    клиенте, даже если тест упал на ассерте, — а ``filterwarnings=error`` делает
    любое предупреждение падением.
    """
    created: list[LlmClient] = []

    def factory(
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        provider: str,
        model: str = "",
        api_key: str = KEY,
        max_retries: int = 0,
        extra: Mapping[str, Any] | None = None,
    ) -> LlmClient:
        client = create_llm_client(
            provider,
            model=model,
            api_key=api_key,
            transport=httpx.MockTransport(handler),
            max_retries=max_retries,
            extra=dict(extra or {}),
        )
        created.append(client)
        return client

    yield factory
    for client in created:
        client.close()


class TestOpenAiCompatible:
    """openai / deepseek / openrouter: общий ``/chat/completions``-провод."""

    def test_complete_assembles_text_usage_and_finish(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("При", "вет", ", мир"), usage=(12, 7)))
        client = make_client(recorder, provider="openai")
        response = client.complete([LlmMessage.user("Привет")])
        assert response.text == "Привет, мир"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 7
        assert response.engine == "openai"

    def test_stream_yields_text_deltas_then_a_done(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("a", "b")))
        client = make_client(recorder, provider="openai")
        deltas = list(client.stream([LlmMessage.user("hi")]))
        texts = [d.text for d in deltas if isinstance(d, LlmTextDelta)]
        assert texts == ["a", "b"]
        assert isinstance(deltas[-1], LlmDoneDelta)
        assert deltas[-1].finish_reason is FinishReason.STOP

    def test_request_pins_openai_host_and_carries_bearer(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="openai", model="gpt-4o-mini")
        list(client.stream([LlmMessage.user("hi")]))
        request = recorder.last
        assert request.url.host == "api.openai.com"
        assert request.url.path == "/v1/chat/completions"
        assert request.method == "POST"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        body = recorder.body_of(request)
        assert body["model"] == "gpt-4o-mini"
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_deepseek_pins_its_own_host_and_default_model(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="deepseek", model="")
        list(client.stream([LlmMessage.user("hi")]))
        request = recorder.last
        assert request.url.host == "api.deepseek.com"
        assert recorder.body_of(request)["model"] == "deepseek-chat"

    def test_openrouter_sends_attribution_headers(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="openrouter")
        list(client.stream([LlmMessage.user("hi")]))
        headers = recorder.last.headers
        assert headers["authorization"] == f"Bearer {KEY}"
        assert headers["http-referer"] == "https://github.com/ayris-assistant"
        assert headers["x-title"] == "Ayris"
        assert recorder.last.url.host == "openrouter.ai"

    def test_tool_call_is_assembled_from_the_stream(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(
            stream=openai_tool_sse(
                name="open_app", call_id="call_1", arg_fragments=('{"app":', '"vscode"}')
            )
        )
        client = make_client(recorder, provider="openai")
        response = client.complete([LlmMessage.user("открой vscode")])
        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.name == "open_app"
        assert call.call_id == "call_1"
        assert call.arguments == {"app": "vscode"}

    def test_tool_call_with_malformed_args_decodes_to_empty(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(
            stream=openai_tool_sse(name="do_it", call_id="c9", arg_fragments=("{bad",))
        )
        client = make_client(recorder, provider="openai")
        response = client.complete([LlmMessage.user("сделай")])
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "do_it"
        assert response.tool_calls[0].arguments == {}

    def test_tools_go_into_the_payload(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="openai")
        tool = LlmTool(name="open_app", description="Открыть", parameters={"type": "object"})
        list(client.stream([LlmMessage.user("hi")], tools=[tool]))
        body = recorder.body_of(recorder.last)
        assert body["tools"][0]["function"]["name"] == "open_app"

    def test_temperature_and_max_tokens_reach_the_payload(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="openai")
        client.complete([LlmMessage.user("hi")], temperature=0.3, max_tokens=64)
        body = recorder.body_of(recorder.last)
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 64

    def test_option_level_temperature_is_used_without_an_override(self) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = create_llm_client(
            "openai",
            api_key=KEY,
            temperature=0.2,
            transport=httpx.MockTransport(recorder),
            max_retries=0,
        )
        client.complete([LlmMessage.user("hi")])
        assert recorder.body_of(recorder.last)["temperature"] == 0.2
        client.close()

    def test_finish_length_maps_from_openai_length(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",), finish="length"))
        client = make_client(recorder, provider="openai")
        assert client.complete([LlmMessage.user("hi")]).finish_reason is FinishReason.LENGTH

    def test_finish_content_filter_maps_to_error(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",), finish="content_filter"))
        client = make_client(recorder, provider="openai")
        assert client.complete([LlmMessage.user("hi")]).finish_reason is FinishReason.ERROR

    def test_stream_without_a_done_still_terminates_with_stop(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("Привет",), done=False))
        client = make_client(recorder, provider="openai")
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "Привет"
        assert response.finish_reason is FinishReason.STOP

    def test_complete_without_a_usage_frame_reports_zero_tokens(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("ok",), usage=None))
        client = make_client(recorder, provider="openai")
        assert client.complete([LlmMessage.user("hi")]).usage.total_tokens == 0

    def test_configured_true_with_key_and_false_without(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        assert make_client(Recorder(), provider="openai").configured is True
        assert make_client(Recorder(), provider="openai", api_key="").configured is False

    def test_list_models_reads_the_catalogue(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(models={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]})
        client = make_client(recorder, provider="openai")
        assert client.list_models() == ("gpt-4o", "gpt-4o-mini")
        assert recorder.last.url.path == "/v1/models"

    def test_check_credentials_ok_over_the_mock(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models={"data": [{"id": "gpt-4o"}]})
        client = make_client(recorder, provider="openai")
        check = client.check_credentials()
        assert check.ok is True
        assert check.models == ("gpt-4o",)

    def test_check_credentials_without_a_key_makes_no_call(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder()
        client = make_client(recorder, provider="openai", api_key="")
        check = client.check_credentials()
        assert check.ok is False
        assert recorder.requests == []

    def test_key_never_appears_in_a_repr(self, make_client: Callable[..., LlmClient]) -> None:
        client = make_client(Recorder(), provider="openai")
        assert KEY not in repr(client)

    def test_close_is_idempotent(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=openai_sse(("ok",)))
        client = make_client(recorder, provider="openai")
        client.complete([LlmMessage.user("hi")])
        client.close()
        client.close()  # второй раз — не должно бросить


class TestCancellation:
    """Отмена: сокет не открывается заранее и закрывается на полуслове."""

    def test_cancel_before_the_first_token_makes_no_request(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_sse(("hi",)))
        client = make_client(recorder, provider="openai")

        def cancel() -> bool:
            return True

        response = client.complete([LlmMessage.user("hi")], cancel=cancel)
        assert response.finish_reason is FinishReason.CANCELLED
        assert recorder.requests == []

    def test_cancel_mid_stream_ends_cancelled(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=openai_sse(("a", "b", "c")))
        client = make_client(recorder, provider="openai")
        polls = {"n": 0}

        def cancel() -> bool:
            polls["n"] += 1
            return polls["n"] >= 2

        response = client.complete([LlmMessage.user("hi")], cancel=cancel)
        assert response.finish_reason is FinishReason.CANCELLED


class TestErrorMapping:
    """Статусы и сетевые сбои ложатся на типизированные ошибки Ayris."""

    def test_401_raises_auth_error(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=b'{"error":"bad key"}', chat_status=401)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmAuthError):
            client.complete([LlmMessage.user("hi")])

    def test_429_becomes_quota_error_after_the_budget(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=b'{"error":"slow down"}', chat_status=429)
        client = make_client(recorder, provider="openai", max_retries=0)
        with pytest.raises(LlmQuotaError):
            client.complete([LlmMessage.user("hi")])

    def test_500_becomes_a_plain_llm_error_after_the_budget(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=b"boom", chat_status=503)
        client = make_client(recorder, provider="openai", max_retries=0)
        with pytest.raises(LlmError) as excinfo:
            client.complete([LlmMessage.user("hi")])
        assert type(excinfo.value) is LlmError

    def test_400_with_a_context_marker_is_overflow(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=b'{"error":"maximum context length exceeded"}', chat_status=400)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmContextOverflowError):
            client.complete([LlmMessage.user("hi")])

    def test_400_without_a_marker_is_a_plain_llm_error(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=b'{"error":"nonsense"}', chat_status=400)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmError) as excinfo:
            client.complete([LlmMessage.user("hi")])
        assert type(excinfo.value) is LlmError

    def test_network_failure_becomes_an_llm_error(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = RaisingRecorder(httpx.ConnectError("no route"))
        client = make_client(recorder, provider="openai", max_retries=0)
        with pytest.raises(LlmError):
            client.complete([LlmMessage.user("hi")])

    def test_retry_then_success_sleeps_with_jittered_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delays: list[float] = []
        monkeypatch.setattr("ayris.nlu.llm.cloud.sleep", delays.append)
        recorder = SequenceRecorder(
            [
                lambda: httpx.Response(503, content=b"down"),
                lambda: httpx.Response(
                    200,
                    content=openai_sse(("готово",)),
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        client = create_llm_client(
            "openai", api_key=KEY, transport=httpx.MockTransport(recorder), max_retries=1
        )
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "готово"
        assert len(recorder.requests) == 2
        assert len(delays) == 1
        assert 0.25 <= delays[0] < 0.5
        client.close()

    def test_retry_after_header_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        delays: list[float] = []
        monkeypatch.setattr("ayris.nlu.llm.cloud.sleep", delays.append)
        recorder = SequenceRecorder(
            [
                lambda: httpx.Response(429, content=b"wait", headers={"retry-after": "1"}),
                lambda: httpx.Response(
                    200,
                    content=openai_sse(("ok",)),
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        client = create_llm_client(
            "openai", api_key=KEY, transport=httpx.MockTransport(recorder), max_retries=1
        )
        client.complete([LlmMessage.user("hi")])
        assert delays == [1.0]
        client.close()


class TestCatalogueErrors:
    """Сбои ``list_models`` / ``check_credentials`` над моком."""

    def test_list_models_auth_error(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(models_raw=b"denied", models_status=401)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmAuthError):
            client.list_models()

    def test_list_models_server_error_is_converted(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models_raw=b"down", models_status=500)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmError):
            client.list_models()

    def test_list_models_network_error(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = RaisingRecorder(httpx.ConnectError("no route"))
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmError):
            client.list_models()

    def test_list_models_ignores_a_non_list_catalogue(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models={"data": "not-a-list"})
        client = make_client(recorder, provider="openai")
        assert client.list_models() == ()

    def test_list_models_ignores_unparseable_json(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models_raw=b"<html>not json</html>")
        client = make_client(recorder, provider="openai")
        assert client.list_models() == ()

    def test_check_credentials_reports_a_failure(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models_raw=b"denied", models_status=401)
        client = make_client(recorder, provider="openai")
        check = client.check_credentials()
        assert check.ok is False
        assert check.detail


class TestAnthropic:
    """Anthropic Messages API — своя схема запроса, потока и заголовков."""

    def test_complete_assembles_text_and_usage(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(
            stream=anthropic_sse(text_parts=("При", "вет"), input_tokens=15, output_tokens=7)
        )
        client = make_client(recorder, provider="anthropic")
        response = client.complete([LlmMessage.user("Привет")])
        assert response.text == "Привет"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 15
        assert response.usage.completion_tokens == 7
        assert response.engine == "anthropic"

    def test_request_targets_messages_endpoint_with_api_key_header(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",)))
        client = make_client(recorder, provider="anthropic")
        list(client.stream([LlmMessage.user("hi")]))
        request = recorder.last
        assert request.url.host == "api.anthropic.com"
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == KEY
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        body = recorder.body_of(request)
        assert body["model"] == "claude-3-5-sonnet-latest"
        assert body["max_tokens"] == 1024
        assert body["stream"] is True

    def test_system_message_becomes_a_top_level_field(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",)))
        client = make_client(recorder, provider="anthropic")
        list(client.stream([LlmMessage.system("Ты — Айрис."), LlmMessage.user("привет")]))
        body = recorder.body_of(recorder.last)
        assert body["system"] == "Ты — Айрис."
        assert [turn["role"] for turn in body["messages"]] == ["user"]

    def test_tool_message_becomes_a_tool_result_turn(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",)))
        client = make_client(recorder, provider="anthropic")
        list(client.stream([LlmMessage.tool("42", tool_call_id="toolu_1", name="calc")]))
        content = recorder.body_of(recorder.last)["messages"][0]["content"][0]
        assert content["type"] == "tool_result"
        assert content["tool_use_id"] == "toolu_1"
        assert content["content"] == "42"

    def test_max_tokens_and_temperature_reach_the_payload(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",)))
        client = make_client(recorder, provider="anthropic")
        client.complete([LlmMessage.user("hi")], max_tokens=256, temperature=0.5)
        body = recorder.body_of(recorder.last)
        assert body["max_tokens"] == 256
        assert body["temperature"] == 0.5

    def test_tools_use_the_anthropic_shape(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",)))
        client = make_client(recorder, provider="anthropic")
        tool = LlmTool(name="open_app", description="Открыть", parameters={"type": "object"})
        list(client.stream([LlmMessage.user("hi")], tools=[tool]))
        declared = recorder.body_of(recorder.last)["tools"][0]
        assert declared["name"] == "open_app"
        assert declared["description"] == "Открыть"
        assert declared["input_schema"] == {"type": "object"}

    def test_tool_call_assembled_from_typed_events(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(
            stream=anthropic_sse(
                input_tokens=15,
                output_tokens=8,
                stop_reason="tool_use",
                tool={"id": "toolu_1", "name": "open_app", "args": ['{"app":"vscode"}']},
            )
        )
        client = make_client(recorder, provider="anthropic")
        response = client.complete([LlmMessage.user("открой vscode")])
        assert response.finish_reason is FinishReason.TOOL_CALLS
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "open_app"
        assert response.tool_calls[0].call_id == "toolu_1"
        assert response.tool_calls[0].arguments == {"app": "vscode"}

    def test_stop_reason_max_tokens_maps_to_length(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("ok",), stop_reason="max_tokens"))
        client = make_client(recorder, provider="anthropic")
        assert client.complete([LlmMessage.user("hi")]).finish_reason is FinishReason.LENGTH

    def test_list_models_reads_an_openai_shaped_catalogue(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(models={"data": [{"id": "claude-3-5-sonnet-latest"}]})
        client = make_client(recorder, provider="anthropic")
        assert client.list_models() == ("claude-3-5-sonnet-latest",)
        assert recorder.last.url.path == "/v1/models"

    def test_check_credentials_ok(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(models={"data": [{"id": "claude-3-5-sonnet-latest"}]})
        client = make_client(recorder, provider="anthropic")
        assert client.check_credentials().ok is True

    def test_configured_flags(self, make_client: Callable[..., LlmClient]) -> None:
        assert make_client(Recorder(), provider="anthropic").configured is True
        assert make_client(Recorder(), provider="anthropic", api_key="").configured is False

    def test_key_never_appears_in_a_repr(self, make_client: Callable[..., LlmClient]) -> None:
        assert KEY not in repr(make_client(Recorder(), provider="anthropic"))


class TestYandex:
    """YandexGPT completion API — ``gpt://``-URI, дифф чанков, ``Api-Key``."""

    def test_complete_diffs_cumulative_chunks(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(
            stream=yandex_ndjson(
                [
                    ("При", "ALTERNATIVE_STATUS_PARTIAL"),
                    ("Привет", "ALTERNATIVE_STATUS_PARTIAL"),
                    ("Привет, мир", "ALTERNATIVE_STATUS_FINAL"),
                ],
                usage=(12, 7),
            )
        )
        client = make_client(recorder, provider="yandex")
        response = client.complete([LlmMessage.user("Привет")])
        assert response.text == "Привет, мир"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.prompt_tokens == 12
        assert response.usage.completion_tokens == 7
        assert response.engine == "yandex"

    def test_request_targets_completion_endpoint_with_api_key_auth(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=yandex_ndjson([("ok", "ALTERNATIVE_STATUS_FINAL")]))
        client = make_client(
            recorder, provider="yandex", model="yandexgpt-lite", extra={"folder_id": "b1gfolder"}
        )
        list(client.stream([LlmMessage.user("привет")]))
        request = recorder.last
        assert request.url.host == "llm.api.cloud.yandex.net"
        assert request.url.path == "/foundationModels/v1/completion"
        assert request.headers["authorization"] == f"Api-Key {KEY}"
        body = recorder.body_of(request)
        assert body["modelUri"] == "gpt://b1gfolder/yandexgpt-lite/latest"
        assert body["completionOptions"]["stream"] is True
        assert body["messages"][0]["text"] == "привет"

    def test_model_uri_is_used_verbatim_when_it_is_a_gpt_scheme(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=yandex_ndjson([("ok", "ALTERNATIVE_STATUS_FINAL")]))
        client = make_client(recorder, provider="yandex", model="gpt://folder/custom/rc")
        list(client.stream([LlmMessage.user("hi")]))
        assert recorder.body_of(recorder.last)["modelUri"] == "gpt://folder/custom/rc"

    def test_model_uri_falls_back_to_the_bare_model_without_a_folder(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=yandex_ndjson([("ok", "ALTERNATIVE_STATUS_FINAL")]))
        client = make_client(recorder, provider="yandex", model="yandexgpt")
        list(client.stream([LlmMessage.user("hi")]))
        assert recorder.body_of(recorder.last)["modelUri"] == "yandexgpt"

    def test_max_tokens_is_serialised_as_a_string(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=yandex_ndjson([("ok", "ALTERNATIVE_STATUS_FINAL")]))
        client = make_client(recorder, provider="yandex")
        client.complete([LlmMessage.user("hi")], max_tokens=100, temperature=0.4)
        options = recorder.body_of(recorder.last)["completionOptions"]
        assert options["maxTokens"] == "100"
        assert options["temperature"] == 0.4

    def test_truncated_final_maps_to_length(self, make_client: Callable[..., LlmClient]) -> None:
        recorder = Recorder(stream=yandex_ndjson([(" well", "ALTERNATIVE_STATUS_TRUNCATED_FINAL")]))
        client = make_client(recorder, provider="yandex")
        assert client.complete([LlmMessage.user("hi")]).finish_reason is FinishReason.LENGTH

    def test_list_models_returns_the_known_set_without_a_request(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder()
        client = make_client(recorder, provider="yandex")
        assert client.list_models() == ("yandexgpt", "yandexgpt-lite", "yandexgpt-32k")
        assert recorder.requests == []

    def test_check_credentials_ok_posts_to_the_tokenizer(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(tokenize_status=200)
        client = make_client(recorder, provider="yandex")
        check = client.check_credentials()
        assert check.ok is True
        assert check.models == ("yandexgpt", "yandexgpt-lite", "yandexgpt-32k")
        assert recorder.last.url.path == "/foundationModels/v1/tokenize"
        assert recorder.last.method == "POST"

    def test_check_credentials_rejects_a_bad_key(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(tokenize_status=401)
        client = make_client(recorder, provider="yandex")
        check = client.check_credentials()
        assert check.ok is False
        assert "не принял" in check.detail

    def test_check_credentials_reports_an_http_error(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(tokenize_status=500)
        client = make_client(recorder, provider="yandex")
        check = client.check_credentials()
        assert check.ok is False
        assert "500" in check.detail

    def test_check_credentials_without_a_key_makes_no_call(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder()
        client = make_client(recorder, provider="yandex", api_key="")
        assert client.check_credentials().ok is False
        assert recorder.requests == []

    def test_check_credentials_network_error_raises(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = RaisingRecorder(httpx.ConnectError("no route"))
        client = make_client(recorder, provider="yandex")
        with pytest.raises(LlmError):
            client.check_credentials()

    def test_configured_flags(self, make_client: Callable[..., LlmClient]) -> None:
        assert make_client(Recorder(), provider="yandex").configured is True
        assert make_client(Recorder(), provider="yandex", api_key="").configured is False

    def test_key_never_appears_in_a_repr(self, make_client: Callable[..., LlmClient]) -> None:
        assert KEY not in repr(make_client(Recorder(), provider="yandex"))


class TestVerify:
    """Брендовый клиент проверяет TLS по системным корням, без своего CA."""

    def test_branded_build_client_verifies_against_system_roots(self) -> None:
        client = OpenAiLlmClient(CloudOptions(api_key="unused"))
        assert client._extra_ca_path() is None
        assert client._resolve_verify() is True
        built = client._build_client()
        assert isinstance(built, httpx.Client)
        built.close()
        client.close()

    def test_verify_false_is_honoured(self) -> None:
        client = OpenAiLlmClient(CloudOptions(api_key=KEY, verify=False))
        assert client._resolve_verify() is False
        client.close()

    def test_model_property_reports_the_requested_model(self) -> None:
        client = OpenAiLlmClient(CloudOptions(model="gpt-4o", api_key=KEY))
        assert client.model == "gpt-4o"
        client.close()


class TestMalformedFrames:
    """Битые кадры не роняют разбор — они молча пропускаются."""

    def test_openai_tolerates_junk_frames(self, make_client: Callable[..., LlmClient]) -> None:
        lines = [
            ": keep-alive",
            "",
            "data: {bad",
            'data: {"choices":"nope"}',
            'data: {"choices":[123]}',
            'data: {"choices":[{"delta":123}]}',
            'data: {"choices":[{"delta":{"tool_calls":[123]}}]}',
            'data: {"choices":[],"usage":{"foo":"bar"}}',
            'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        stream = ("\n".join(lines) + "\n").encode("utf-8")
        client = make_client(Recorder(stream=stream), provider="openai")
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "ok"
        assert response.finish_reason is FinishReason.STOP
        assert response.usage.total_tokens == 0

    def test_tool_call_without_arguments_decodes_to_empty(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=openai_tool_sse(name="ping", call_id="c1", arg_fragments=()))
        client = make_client(recorder, provider="openai")
        response = client.complete([LlmMessage.user("ping")])
        assert response.tool_calls[0].name == "ping"
        assert response.tool_calls[0].arguments == {}

    def test_empty_error_body_maps_to_a_plain_error(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=b"", chat_status=400)
        client = make_client(recorder, provider="openai")
        with pytest.raises(LlmError):
            client.complete([LlmMessage.user("hi")])

    def test_retry_after_non_numeric_falls_back(self) -> None:
        recorder = SequenceRecorder(
            [lambda: httpx.Response(429, content=b"x", headers={"retry-after": "soon"})]
        )
        client = create_llm_client(
            "openai", api_key=KEY, transport=httpx.MockTransport(recorder), max_retries=0
        )
        with pytest.raises(LlmQuotaError):
            client.complete([LlmMessage.user("hi")])
        client.close()

    def test_retry_after_zero_is_ignored(self) -> None:
        recorder = SequenceRecorder(
            [lambda: httpx.Response(429, content=b"x", headers={"retry-after": "0"})]
        )
        client = create_llm_client(
            "openai", api_key=KEY, transport=httpx.MockTransport(recorder), max_retries=0
        )
        with pytest.raises(LlmQuotaError):
            client.complete([LlmMessage.user("hi")])
        client.close()

    def test_anthropic_tolerates_odd_events_and_no_stop(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        lines = [
            "event: ping",
            "data: not json",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text"}}',
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":'
            '{"type":"input_json_delta","partial_json":""}}',
            'data: {"type":"message_delta"}',
            'data: {"type":"content_block_delta","index":0,'
            '"delta":{"type":"text_delta","text":"hi"}}',
        ]
        stream = ("\n".join(lines) + "\n").encode("utf-8")
        client = make_client(Recorder(stream=stream), provider="anthropic")
        response = client.complete([LlmMessage.system(""), LlmMessage.user("hi")])
        assert response.text == "hi"
        assert response.finish_reason is FinishReason.STOP

    def test_anthropic_cancel_mid_stream_ends_cancelled(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(stream=anthropic_sse(text_parts=("a", "b", "c")))
        client = make_client(recorder, provider="anthropic")
        polls = {"n": 0}

        def cancel() -> bool:
            polls["n"] += 1
            return polls["n"] >= 2

        response = client.complete([LlmMessage.user("hi")], cancel=cancel)
        assert response.finish_reason is FinishReason.CANCELLED

    def test_anthropic_message_stop_without_tokens_emits_no_usage(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(
            stream=anthropic_sse(text_parts=("hi",), input_tokens=0, output_tokens=0)
        )
        client = make_client(recorder, provider="anthropic")
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "hi"
        assert response.usage.total_tokens == 0

    def test_yandex_tolerates_odd_chunks_and_no_final(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        lines = [
            "",
            "not json",
            '{"foo":"bar"}',
            '{"result":{"alternatives":[]}}',
            '{"result":{"alternatives":[123]}}',
            '{"result":{"alternatives":[{"message":{"text":"Привет"},"status":"PARTIAL"}]}}',
            '{"result":{"alternatives":[{"message":{"text":"Привет"},"status":"PARTIAL"}]}}',
            '{"result":{"alternatives":[{"message":{"text":"Привет!"},"status":"PARTIAL"}]}}',
        ]
        stream = ("\n".join(lines) + "\n").encode("utf-8")
        client = make_client(Recorder(stream=stream), provider="yandex")
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "Привет!"
        assert response.finish_reason is FinishReason.STOP

    def test_yandex_cancel_mid_stream_ends_cancelled(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        recorder = Recorder(
            stream=yandex_ndjson(
                [("a", "PARTIAL"), ("ab", "PARTIAL"), ("abc", "ALTERNATIVE_STATUS_FINAL")]
            )
        )
        client = make_client(recorder, provider="yandex")
        polls = {"n": 0}

        def cancel() -> bool:
            polls["n"] += 1
            return polls["n"] >= 2

        response = client.complete([LlmMessage.user("hi")], cancel=cancel)
        assert response.finish_reason is FinishReason.CANCELLED

    def test_yandex_drops_unparseable_usage_counts(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        frame: JsonObject = {
            "result": {
                "alternatives": [{"message": {"text": "ok"}, "status": "ALTERNATIVE_STATUS_FINAL"}],
                "usage": {"inputTextTokens": "abc", "completionTokens": True},
            }
        }
        stream = (json.dumps(frame) + "\n").encode("utf-8")
        client = make_client(Recorder(stream=stream), provider="yandex")
        response = client.complete([LlmMessage.user("hi")])
        assert response.text == "ok"
        assert response.usage.total_tokens == 0

    def test_yandex_reads_int_usage_and_ignores_null(
        self, make_client: Callable[..., LlmClient]
    ) -> None:
        frame: JsonObject = {
            "result": {
                "alternatives": [{"message": {"text": "ok"}, "status": "ALTERNATIVE_STATUS_FINAL"}],
                "usage": {"inputTextTokens": 5, "completionTokens": None},
            }
        }
        stream = (json.dumps(frame) + "\n").encode("utf-8")
        client = make_client(Recorder(stream=stream), provider="yandex")
        response = client.complete([LlmMessage.user("hi")])
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 0
