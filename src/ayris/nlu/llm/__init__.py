"""LLM clients behind a common abstraction: cloud providers and local runtimes.

Used both for free-form Q&A and for turning an utterance into a structured
command in hybrid NLU mode.

:mod:`ayris.nlu.llm.base` holds the contract; the six cloud providers of
``ai.provider`` plug in behind it (task 61) and the local runtimes — Ollama, LM
Studio and the in-process llama.cpp (task 62) — plug in the same way, without the
pipeline noticing. :class:`~ayris.nlu.llm.router.LlmRouter` puts the three modes
(offline / online / auto) in front of them with cloud↔local fallback. Until one is
configured, :class:`~ayris.nlu.llm.base.NullLlmClient` answers with a sentence
saying so.

The names re-exported here are the package's public surface: the message and
delta types callers build and read, the :func:`~ayris.nlu.llm.factory.create_llm_client`
factory that maps ``ai.provider`` to a client, the
:class:`~ayris.nlu.llm.router.LlmRouter` that routes between cloud and local, and
the usage accounting the «ИИ» tab shows. The provider client classes are
deliberately *not* here — they are imported lazily by the factory, so importing
this package pulls in no SDK, and the recommended-model
:mod:`~ayris.nlu.llm.catalog` stays a submodule so its optional ``psutil`` probe is
not paid on import.
"""

from __future__ import annotations

from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
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
from ayris.nlu.llm.factory import (
    CLOUD_PROVIDERS,
    LOCAL_PROVIDERS,
    create_llm_client,
    is_cloud_provider,
    is_local_provider,
)
from ayris.nlu.llm.keys import mask, resolve_api_key
from ayris.nlu.llm.router import LlmMode, LlmRouter
from ayris.nlu.llm.stream import SentenceAssembler
from ayris.nlu.llm.usage import (
    ModelPrice,
    UsageMeter,
    UsageRecord,
    price_for,
    resolve_usage,
)

__all__ = [
    "CLOUD_PROVIDERS",
    "LOCAL_PROVIDERS",
    "NOT_CONFIGURED_MESSAGE",
    "CredentialCheck",
    "FinishReason",
    "LlmClient",
    "LlmDelta",
    "LlmDoneDelta",
    "LlmMessage",
    "LlmMode",
    "LlmResponse",
    "LlmRole",
    "LlmRouter",
    "LlmTextDelta",
    "LlmTool",
    "LlmToolCall",
    "LlmToolCallDelta",
    "LlmUsage",
    "LlmUsageDelta",
    "ModelPrice",
    "NullLlmClient",
    "SentenceAssembler",
    "UsageMeter",
    "UsageRecord",
    "create_llm_client",
    "is_cloud_provider",
    "is_local_provider",
    "mask",
    "price_for",
    "resolve_api_key",
    "resolve_usage",
]
