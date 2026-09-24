"""OpenRouter as a cloud LLM provider.

OpenRouter is a broker in front of dozens of models but speaks the OpenAI
``/chat/completions`` schema verbatim, so the only additions over
:class:`OpenAiCompatibleClient` are the two headers it asks clients to send for
attribution — neither carries a secret, both are optional, and Ayris sends a
stable value so the dashboard shows one app rather than a new one per update.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ayris.nlu.llm.cloud import OpenAiCompatibleClient

__all__ = ["OpenRouterLlmClient"]


class OpenRouterLlmClient(OpenAiCompatibleClient):
    """Any model in the OpenRouter catalogue, addressed as ``vendor/model``."""

    name: ClassVar[str] = "openrouter"
    title: ClassVar[str] = "OpenRouter"
    default_base_url: ClassVar[str] = "https://openrouter.ai/api/v1"
    default_model: ClassVar[str] = "openai/gpt-4o-mini"

    def _auth_headers(self) -> Mapping[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/ayris-assistant",
            "X-Title": "Ayris",
        }
