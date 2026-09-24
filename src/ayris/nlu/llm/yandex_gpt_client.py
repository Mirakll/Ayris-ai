"""YandexGPT as a cloud LLM provider.

Yandex has its own API in every respect: the model is a ``gpt://folder/name/tag``
URI, options live under ``completionOptions``, messages carry ``text`` rather than
``content``, and the stream is newline-delimited JSON objects — not SSE — each
carrying the answer *so far* rather than the latest fragment. So this client
diffs each chunk against the previous cumulative text to recover deltas, and
implements the four hooks against Yandex's shape while yielding the same
:class:`~ayris.nlu.llm.base.LlmDelta` stream as every other provider.

The credential is a Yandex Cloud API key; the folder id comes from
``CloudOptions.extra["folder_id"]`` unless the model is already a full ``gpt://``
URI, in which case it is used verbatim.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

import httpx

from ayris.core.errors import LlmError
from ayris.core.models import JsonObject
from ayris.nlu.llm.base import (
    CredentialCheck,
    FinishReason,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmTextDelta,
    LlmTool,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.nlu.llm.cloud import CloudLlmClient, CloudOptions, _loads_object

__all__ = ["YandexGptLlmClient"]

_HOST = "https://llm.api.cloud.yandex.net"

#: Statuses that end a stream, mapped to why it ended.
_FINAL_STATUSES: dict[str, FinishReason] = {
    "ALTERNATIVE_STATUS_FINAL": FinishReason.STOP,
    "ALTERNATIVE_STATUS_TRUNCATED_FINAL": FinishReason.LENGTH,
}

#: Models the picker offers without a network call — Yandex has no cheap list.
_KNOWN_MODELS: tuple[str, ...] = ("yandexgpt", "yandexgpt-lite", "yandexgpt-32k")


class YandexGptLlmClient(CloudLlmClient):
    """YandexGPT over the Foundation Models completion API."""

    name: ClassVar[str] = "yandex"
    title: ClassVar[str] = "YandexGPT"
    default_base_url: ClassVar[str] = _HOST
    default_model: ClassVar[str] = "yandexgpt-lite"

    def __init__(self, options: CloudOptions) -> None:
        super().__init__(options)
        self._folder_id = str(options.extra.get("folder_id") or "")

    def _endpoint(self) -> str:
        return f"{self._base_url}/foundationModels/v1/completion"

    def _auth_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Api-Key {self._api_key}"}

    def _model_uri(self) -> str:
        """Build the ``gpt://`` URI Yandex addresses a model by."""
        if self._model.startswith("gpt://"):
            return self._model
        if self._folder_id:
            return f"gpt://{self._folder_id}/{self._model}/latest"
        # No folder to build a URI from; send what we have and let Yandex reject
        # it with a message the user can act on.
        return self._model

    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> JsonObject:
        del tools  # YandexGPT function calling is not wired up here.
        options: dict[str, Any] = {"stream": stream}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            # Yandex wants maxTokens as a string.
            options["maxTokens"] = str(max_tokens)
        return {
            "modelUri": self._model_uri(),
            "completionOptions": options,
            "messages": [
                {"role": message.role.value, "text": message.content} for message in messages
            ],
        }

    def _read_stream(
        self,
        response: httpx.Response,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        previous = ""
        for line in response.iter_lines():
            if cancel():
                return
            stripped = line.strip()
            if not stripped:
                continue
            chunk = _loads_object(stripped)
            if chunk is None:
                continue
            result = chunk.get("result")
            if not isinstance(result, dict):
                continue
            text, status = _first_alternative(result)
            if len(text) > len(previous):
                yield LlmTextDelta(text=text[len(previous) :])
                previous = text
            elif text:
                previous = text
            if status in _FINAL_STATUSES:
                usage = _yandex_usage(result.get("usage"))
                if usage is not None:
                    yield usage
                yield LlmDoneDelta(finish_reason=_FINAL_STATUSES[status])
                return
        # No final status arrived; the base synthesises the terminal delta.

    def list_models(self) -> tuple[str, ...]:
        return _KNOWN_MODELS

    def check_credentials(self) -> CredentialCheck:
        if not self.configured:
            return CredentialCheck(ok=False, detail="Ключ доступа не задан.")
        client = self._require_client()
        url = f"{self._base_url}/foundationModels/v1/tokenize"
        body: JsonObject = {"modelUri": self._model_uri(), "text": "привет"}
        try:
            response = client.post(
                url,
                headers=dict(self._auth_headers()),
                json=body,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LlmError(
                f"{self.name}: credential check failed: {exc}",
                user_message="YandexGPT недоступен. Проверьте подключение к интернету.",
            ) from exc
        status = response.status_code
        if status in (401, 403):
            return CredentialCheck(ok=False, detail="YandexGPT не принял ключ доступа.")
        if not (200 <= status < 300):
            return CredentialCheck(ok=False, detail=f"YandexGPT ответил ошибкой {status}.")
        return CredentialCheck(ok=True, models=_KNOWN_MODELS)


def _first_alternative(result: Mapping[str, Any]) -> tuple[str, str]:
    """Read the leading alternative's cumulative text and status out of a chunk."""
    alternatives = result.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return "", ""
    first = alternatives[0]
    if not isinstance(first, dict):
        return "", ""
    message = first.get("message")
    text = message.get("text") if isinstance(message, dict) else None
    status = first.get("status")
    return (
        text if isinstance(text, str) else "",
        status if isinstance(status, str) else "",
    )


def _yandex_usage(raw: object) -> LlmUsageDelta | None:
    """Yandex reports token counts as strings; parse the two Ayris cares about."""
    if not isinstance(raw, dict):
        return None
    prompt = _as_int(raw.get("inputTextTokens"))
    completion = _as_int(raw.get("completionTokens"))
    if prompt == 0 and completion == 0:
        return None
    return LlmUsageDelta(usage=LlmUsage(prompt_tokens=prompt, completion_tokens=completion))


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0
