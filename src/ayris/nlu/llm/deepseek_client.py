"""DeepSeek as a cloud LLM provider.

DeepSeek's API is OpenAI-compatible down to the streaming frames, so this is the
same three-line provider as :mod:`~ayris.nlu.llm.openai_client` with a different
host and default model.
"""

from __future__ import annotations

from typing import ClassVar

from ayris.nlu.llm.cloud import OpenAiCompatibleClient

__all__ = ["DeepSeekLlmClient"]


class DeepSeekLlmClient(OpenAiCompatibleClient):
    """DeepSeek-Chat and DeepSeek-Reasoner over the public API."""

    name: ClassVar[str] = "deepseek"
    title: ClassVar[str] = "DeepSeek"
    default_base_url: ClassVar[str] = "https://api.deepseek.com/v1"
    default_model: ClassVar[str] = "deepseek-chat"
