"""LLM clients behind a common abstraction: cloud providers and local runtimes.

Used both for free-form Q&A and for turning an utterance into a structured
command in hybrid NLU mode.

:mod:`ayris.nlu.llm.base` holds the contract; the eight providers of
``ai.provider`` arrive in task 63 and plug in behind it without the pipeline
noticing. Until one is configured, :class:`~ayris.nlu.llm.base.NullLlmClient`
answers with a sentence saying so.
"""

from __future__ import annotations

from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
    FinishReason,
    LlmClient,
    LlmMessage,
    LlmResponse,
    LlmRole,
    LlmTool,
    LlmToolCall,
    LlmUsage,
    NullLlmClient,
)

__all__ = [
    "NOT_CONFIGURED_MESSAGE",
    "FinishReason",
    "LlmClient",
    "LlmMessage",
    "LlmResponse",
    "LlmRole",
    "LlmTool",
    "LlmToolCall",
    "LlmUsage",
    "NullLlmClient",
]
