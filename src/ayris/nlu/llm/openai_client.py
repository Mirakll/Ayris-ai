"""OpenAI as a cloud LLM provider.

OpenAI defines the ``/chat/completions`` streaming schema every OpenAI-compatible
provider borrows, so there is nothing to do here but name the host, the default
model and the ``ai.provider`` value — :class:`OpenAiCompatibleClient` does the
rest. Adding a provider that speaks the same schema is a file exactly this short.
"""

from __future__ import annotations

from typing import ClassVar

from ayris.nlu.llm.cloud import OpenAiCompatibleClient

__all__ = ["OpenAiLlmClient"]


class OpenAiLlmClient(OpenAiCompatibleClient):
    """GPT-4o and friends over the public OpenAI API."""

    name: ClassVar[str] = "openai"
    title: ClassVar[str] = "OpenAI"
    default_base_url: ClassVar[str] = "https://api.openai.com/v1"
    default_model: ClassVar[str] = "gpt-4o-mini"
