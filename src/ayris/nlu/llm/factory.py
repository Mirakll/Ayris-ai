"""Building the right cloud LLM client from a provider name.

Task 63 will pick a client from ``ai.provider`` for the whole application; this is
the piece it needs first — the mapping from a provider name to a constructed
:class:`~ayris.nlu.llm.base.LlmClient`, with the API key resolved out of the
credential store on the way. It exists now because the LLM worker of task 61
already has to build a client, and doing it here keeps that knowledge in one place
rather than in the worker.

**Adding a provider is a new entry, not a new branch.** :data:`CLOUD_PROVIDERS`
maps each ``ai.provider`` value to its ``"module:Class"`` entrypoint; the classes
are imported lazily, so a machine that never talks to Anthropic never imports its
client and a syntax error in one provider cannot break the other five.

**Only the online providers live here.** The local runtimes (Ollama, LM Studio,
llama.cpp) are a different transport and arrive in task 62; asking this factory
for one is a typed :class:`~ayris.core.errors.LlmError`, which the worker catches
and answers with the «не настроено» sentence rather than crashing on start.

**The ``host`` from config is deliberately ignored for the *branded* providers.**
That field addresses a *local* model server (its default is the Ollama URL);
handing it to OpenAI as a base URL would send its traffic to ``localhost``. Each
branded client uses its own
:attr:`~ayris.nlu.llm.cloud.CloudLlmClient.default_base_url`, and a genuine proxy
override rides in ``extra`` instead.

**The ``custom`` provider is the exception, and the reason ``base_url`` exists
here.** It pins no host of its own, so any OpenAI-compatible service — an
OpenRouter-style broker, Together, Groq, a self-hosted server — works once its
endpoint, model and key are filled in. The worker passes ``ai.host`` as
``base_url`` for that provider only; for every other it passes nothing and the
branded default stands.
"""

from __future__ import annotations

import importlib
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from ayris.core.errors import LlmError
from ayris.nlu.llm.cloud import (
    CONNECT_TIMEOUT_SEC,
    MAX_RETRIES,
    READ_TIMEOUT_SEC,
    CloudOptions,
)
from ayris.nlu.llm.keys import resolve_api_key
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

    from ayris.core.secrets import SecretsStore
    from ayris.nlu.llm.base import LlmClient
    from ayris.nlu.llm.cloud import CloudLlmClient

__all__ = ["CLOUD_PROVIDERS", "CUSTOM_PROVIDER", "create_llm_client", "is_cloud_provider"]

_log = get_logger(__name__)

#: The provider whose endpoint the user supplies, rather than one Ayris pins.
#: The only key in :data:`CLOUD_PROVIDERS` that honours ``base_url``.
CUSTOM_PROVIDER: Final = "custom"

#: Every online provider Ayris speaks to, mapped to the ``"module:Class"`` of its
#: client. Kept in provider-name order for the settings picker; the value is
#: imported only when that provider is actually built. ``custom`` sits last: it is
#: not a named service but a fill-in-the-endpoint entry for anything OpenAI-shaped.
CLOUD_PROVIDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "openai": "ayris.nlu.llm.openai_client:OpenAiLlmClient",
        "anthropic": "ayris.nlu.llm.anthropic_client:AnthropicLlmClient",
        "openrouter": "ayris.nlu.llm.openrouter_client:OpenRouterLlmClient",
        "deepseek": "ayris.nlu.llm.deepseek_client:DeepSeekLlmClient",
        "gigachat": "ayris.nlu.llm.gigachat_client:GigaChatLlmClient",
        "yandex": "ayris.nlu.llm.yandex_gpt_client:YandexGptLlmClient",
        CUSTOM_PROVIDER: "ayris.nlu.llm.custom_client:CustomOpenAiLlmClient",
    }
)


def is_cloud_provider(provider: str) -> bool:
    """Whether ``provider`` names one of the online clients built here."""
    return provider.strip().lower() in CLOUD_PROVIDERS


def _load_client_class(entry: str) -> type[CloudLlmClient]:
    """Import ``"module:Class"`` and return the client class it names.

    Every entry in :data:`CLOUD_PROVIDERS` names a
    :class:`~ayris.nlu.llm.cloud.CloudLlmClient` subclass — one that takes a
    :class:`~ayris.nlu.llm.cloud.CloudOptions` — so the class is typed as such
    rather than the bare :class:`~ayris.nlu.llm.base.LlmClient` (whose ``__init__``
    takes nothing), letting ``client_class(options)`` type-check.
    """
    module_name, _, class_name = entry.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)  # type: ignore[no-any-return]


def create_llm_client(
    provider: str,
    *,
    model: str = "",
    base_url: str = "",
    credential_ref: str = "",
    api_key: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
    connect_timeout_sec: float = CONNECT_TIMEOUT_SEC,
    read_timeout_sec: float = READ_TIMEOUT_SEC,
    max_retries: int = MAX_RETRIES,
    transport: httpx.BaseTransport | None = None,
    store: SecretsStore | None = None,
    extra: Mapping[str, Any] | None = None,
) -> LlmClient:
    """Build the client for ``provider``, key resolved from the credential store.

    Args:
        provider: An ``ai.provider`` value. Only the six online providers in
            :data:`CLOUD_PROVIDERS` are built here.
        model: Model to ask for; empty uses the client's default.
        base_url: Endpoint the request is sent to. Meaningful only for the
            ``custom`` provider, which has no host of its own; for the branded
            providers it is ignored and their pinned host stands. Empty for
            ``custom`` leaves the client unconfigured (nowhere to send to).
        credential_ref: Name of the credential entry to read the key from,
            before falling back to the slot named after the provider.
        api_key: A key handed over directly, skipping the store. Mainly for
            tests; production leaves it empty and lets the store answer.
        temperature: Overrides ``ai.temperature`` when set.
        max_tokens: Overrides ``ai.max_tokens`` when set.
        connect_timeout_sec: Connect timeout for the HTTP client.
        read_timeout_sec: Between-chunk read timeout for the stream.
        max_retries: Retries while *opening* the stream; never mid-stream.
        transport: Injected by the tests to answer with a mock instead of the
            network. Production leaves it ``None``.
        store: Credential store to read from; defaults to the process store.
        extra: Provider-specific settings (Yandex's ``folder_id``, GigaChat's
            ``scope``) passed through verbatim.

    Returns:
        A constructed, unopened client. The socket is not touched until the
        first request, so building one for an unconfigured provider is free and
        makes no network call.

    Raises:
        ayris.core.errors.LlmError: ``provider`` is not one of the online
            providers — a local runtime, an unknown name, or empty.
    """
    key = provider.strip().lower()
    entry = CLOUD_PROVIDERS.get(key)
    if entry is None:
        raise LlmError(
            f"provider {provider!r} is not a cloud LLM provider",
            user_message="Этот провайдер не поддерживается в облачном режиме.",
        )

    resolved_key = resolve_api_key(
        key, credential_ref=credential_ref, explicit=api_key, store=store
    )
    # Only the custom provider takes an endpoint from the caller; a branded one
    # keeps its pinned host so its traffic can never be redirected by config.
    resolved_base_url = base_url.strip() if key == CUSTOM_PROVIDER else ""
    options = CloudOptions(
        model=model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        connect_timeout_sec=connect_timeout_sec,
        read_timeout_sec=read_timeout_sec,
        max_retries=max_retries,
        transport=transport,
        extra=dict(extra or {}),
    )
    client_class = _load_client_class(entry)
    _log.debug(
        "создан клиент %s (%s), ключ %s",
        key,
        client_class.__name__,
        "есть" if resolved_key else "нет",
    )
    return client_class(options)
