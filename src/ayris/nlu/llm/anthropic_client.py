"""Anthropic (Claude) as a cloud LLM provider.

Anthropic does not speak the OpenAI schema: the system prompt is a top-level
field rather than a message, ``max_tokens`` is required, and the stream is a
sequence of typed events (``message_start``, ``content_block_delta``,
``message_delta``, ``message_stop``) rather than ``choices`` fragments. So this
client subclasses :class:`CloudLlmClient` directly and implements the four hooks
against Claude's own shape, while still yielding the same
:class:`~ayris.nlu.llm.base.LlmDelta` stream everything else consumes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, ClassVar

import httpx

from ayris.core.models import JsonObject
from ayris.nlu.llm.base import (
    FinishReason,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmRole,
    LlmTextDelta,
    LlmTool,
    LlmToolCallDelta,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.nlu.llm.cloud import CloudLlmClient, _loads_object

__all__ = ["AnthropicLlmClient"]

#: Anthropic requires ``max_tokens``; this is used when the caller sets none.
_DEFAULT_MAX_TOKENS = 1024

#: The API version header Anthropic pins its request/response shape to.
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLlmClient(CloudLlmClient):
    """Claude 3.5 Sonnet / Haiku over the Anthropic Messages API."""

    name: ClassVar[str] = "anthropic"
    title: ClassVar[str] = "Anthropic"
    supports_tools: ClassVar[bool] = True
    default_base_url: ClassVar[str] = "https://api.anthropic.com/v1"
    default_model: ClassVar[str] = "claude-3-5-sonnet-latest"

    def _endpoint(self) -> str:
        return f"{self._base_url}/messages"

    def _models_endpoint(self) -> str:
        return f"{self._base_url}/models"

    def _auth_headers(self) -> Mapping[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": _ANTHROPIC_VERSION}

    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> JsonObject:
        system_parts, turns = _split_system(messages)
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": turns,
            "max_tokens": max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS,
            "stream": stream,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [_tool_payload(tool) for tool in tools]
        return payload

    def _read_stream(
        self,
        response: httpx.Response,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        prompt_tokens = 0
        completion_tokens = 0
        stop_reason: str | None = None
        for line in response.iter_lines():
            if cancel():
                return
            stripped = line.strip()
            if not stripped.startswith("data:"):
                continue
            event = _loads_object(stripped[len("data:") :].strip())
            if event is None:
                continue
            kind = event.get("type")
            if kind == "message_start":
                prompt_tokens = _usage_field(event.get("message"), "input_tokens", prompt_tokens)
            elif kind == "content_block_start":
                yield from _tool_block_start(event)
            elif kind == "content_block_delta":
                yield from _block_delta(event)
            elif kind == "message_delta":
                stop_reason = _stop_reason(event) or stop_reason
                completion_tokens = _usage_field(event, "output_tokens", completion_tokens)
            elif kind == "message_stop":
                if prompt_tokens or completion_tokens:
                    yield LlmUsageDelta(
                        usage=LlmUsage(
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                        )
                    )
                yield LlmDoneDelta(finish_reason=_map_stop_reason(stop_reason))
                return
        # The stream ended without message_stop; the base synthesises the tail.


def _split_system(messages: Sequence[LlmMessage]) -> tuple[list[str], list[JsonObject]]:
    """Pull system turns out into Anthropic's top-level ``system`` field."""
    system_parts: list[str] = []
    turns: list[JsonObject] = []
    for message in messages:
        if message.role is LlmRole.SYSTEM:
            if message.content:
                system_parts.append(message.content)
        elif message.role is LlmRole.TOOL:
            turns.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
        else:
            turns.append({"role": message.role.value, "content": message.content})
    return system_parts, turns


def _tool_payload(tool: LlmTool) -> JsonObject:
    """Anthropic's tool shape: name, description and an ``input_schema``."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters or {"type": "object", "properties": {}},
    }


def _usage_field(container: object, field_name: str, current: int) -> int:
    """Read a token count out of a ``usage`` sub-object, keeping the old value if absent."""
    source = container.get("usage") if isinstance(container, dict) else None
    if isinstance(source, dict):
        value = source.get(field_name)
        if isinstance(value, int):
            return value
    return current


def _stop_reason(event: Mapping[str, Any]) -> str | None:
    delta = event.get("delta")
    if isinstance(delta, dict):
        reason = delta.get("stop_reason")
        if isinstance(reason, str):
            return reason
    return None


def _tool_block_start(event: Mapping[str, Any]) -> Iterator[LlmDelta]:
    """A ``content_block_start`` opens a tool call: emit its name and id."""
    block = event.get("content_block")
    if not isinstance(block, dict) or block.get("type") != "tool_use":
        return
    index = event.get("index")
    raw_id = block.get("id")
    raw_name = block.get("name")
    yield LlmToolCallDelta(
        index=index if isinstance(index, int) else 0,
        call_id=raw_id if isinstance(raw_id, str) else "",
        name=raw_name if isinstance(raw_name, str) else "",
        arguments="",
    )


def _block_delta(event: Mapping[str, Any]) -> Iterator[LlmDelta]:
    """A ``content_block_delta`` carries text or a fragment of tool-call JSON."""
    delta = event.get("delta")
    if not isinstance(delta, dict):
        return
    delta_type = delta.get("type")
    if delta_type == "text_delta":
        text = delta.get("text")
        if isinstance(text, str) and text:
            yield LlmTextDelta(text=text)
    elif delta_type == "input_json_delta":
        partial = delta.get("partial_json")
        index = event.get("index")
        if isinstance(partial, str) and partial:
            yield LlmToolCallDelta(
                index=index if isinstance(index, int) else 0,
                arguments=partial,
            )


def _map_stop_reason(reason: str | None) -> FinishReason:
    if reason == "max_tokens":
        return FinishReason.LENGTH
    if reason == "tool_use":
        return FinishReason.TOOL_CALLS
    return FinishReason.STOP
