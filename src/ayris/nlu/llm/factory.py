"""Building the right LLM client from a provider name.

Task 63 will pick a client from ``ai.provider`` for the whole application; this is
the piece it needs first — the mapping from a provider name to a constructed
:class:`~ayris.nlu.llm.base.LlmClient`, with the API key resolved out of the
credential store on the way. It exists now because the LLM worker of task 61
already has to build a client, and doing it here keeps that knowledge in one place
rather than in the worker.

**Adding a provider is a new entry, not a new branch.** :data:`CLOUD_PROVIDERS`
maps each online ``ai.provider`` value to its ``"module:Class"`` entrypoint and
:data:`LOCAL_PROVIDERS` does the same for the on-device engines; the classes are
imported lazily, so a machine that never talks to Anthropic never imports its
client and a syntax error in one provider cannot break the others.

**The local engines are three, and two of them are HTTP.** Ollama and LM Studio
run a model behind a local server, so they are built through the same
:class:`~ayris.nlu.llm.cloud.CloudOptions` path as the cloud clients — only their
``base_url`` (the ``ai.host`` from config) is honoured, which is what
:data:`HOST_PROVIDERS` marks. llama.cpp is different: it loads a ``.gguf`` file
in-process, has no endpoint or key, and takes its own
:class:`~ayris.nlu.llm.llamacpp_client.LlamaCppOptions` out of ``extra``, so it is
dispatched on its own before the shared path.

**The ``host`` from config is deliberately ignored for the *branded* providers.**
That field addresses a *local* model server (its default is the Ollama URL);
handing it to OpenAI as a base URL would send its traffic to ``localhost``. Each
branded client uses its own
:attr:`~ayris.nlu.llm.cloud.CloudLlmClient.default_base_url`, and a genuine proxy
override rides in ``extra`` instead. Only the providers in :data:`HOST_PROVIDERS`
— ``custom`` and the two local servers — take an endpoint from config.

**The ``custom`` provider is the reason ``base_url`` exists for the cloud side.**
It pins no host of its own, so any OpenAI-compatible service — an OpenRouter-style
broker, Together, Groq, a self-hosted server — works once its endpoint, model and
key are filled in. The worker passes ``ai.host`` as ``base_url`` for the
:data:`HOST_PROVIDERS` only; for every branded provider it passes nothing and the
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

__all__ = [
    "CLOUD_PROVIDERS",
    "CUSTOM_PROVIDER",
    "HOST_PROVIDERS",
    "LLAMACPP_PROVIDER",
    "LOCAL_PROVIDERS",
    "create_llm_client",
    "is_cloud_provider",
    "is_local_provider",
]

_log = get_logger(__name__)

#: The provider whose endpoint the user supplies, rather than one Ayris pins.
CUSTOM_PROVIDER: Final = "custom"

#: The in-process llama.cpp engine, dispatched on its own because it takes a file
#: path and runtime knobs rather than an endpoint and a key.
LLAMACPP_PROVIDER: Final = "llamacpp"

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

#: The local engines, mapped the same way. Ollama and LM Studio are built through
#: the shared :class:`CloudOptions` path (they are HTTP servers); llama.cpp is
#: listed for discovery but constructed by :func:`_create_llamacpp_client`, which
#: gives it its own options type.
LOCAL_PROVIDERS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "ollama": "ayris.nlu.llm.ollama_client:OllamaLlmClient",
        "lmstudio": "ayris.nlu.llm.lmstudio_client:LmStudioLlmClient",
        LLAMACPP_PROVIDER: "ayris.nlu.llm.llamacpp_client:LlamaCppLlmClient",
    }
)

#: Providers whose ``base_url`` comes from ``ai.host``: the fill-in ``custom`` and
#: the two local servers. A branded cloud provider is never in here, so config can
#: never redirect its traffic.
HOST_PROVIDERS: Final = frozenset({CUSTOM_PROVIDER, "ollama", "lmstudio"})


def is_cloud_provider(provider: str) -> bool:
    """Whether ``provider`` names one of the online clients built here."""
    return provider.strip().lower() in CLOUD_PROVIDERS


def is_local_provider(provider: str) -> bool:
    """Whether ``provider`` names an on-device engine (Ollama, LM Studio, llama.cpp)."""
    return provider.strip().lower() in LOCAL_PROVIDERS


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
        provider: An ``ai.provider`` value. The online providers in
            :data:`CLOUD_PROVIDERS` and the local engines in
            :data:`LOCAL_PROVIDERS` are built here.
        model: Model to ask for; empty uses the client's default.
        base_url: Endpoint the request is sent to. Meaningful for the providers in
            :data:`HOST_PROVIDERS` — the ``custom`` provider and the local Ollama
            and LM Studio servers, none of which pin a host of their own; for the
            branded providers it is ignored and their pinned host stands. Empty for
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
        ayris.core.errors.LlmError: ``provider`` is not one of the supported
            providers — an unknown name, or empty.
    """
    key = provider.strip().lower()
    if key == LLAMACPP_PROVIDER:
        return _create_llamacpp_client(
            model=model, temperature=temperature, max_tokens=max_tokens, extra=extra
        )
    entry = CLOUD_PROVIDERS.get(key) or LOCAL_PROVIDERS.get(key)
    if entry is None:
        raise LlmError(
            f"provider {provider!r} is not a supported LLM provider",
            user_message="Этот провайдер языковой модели не поддерживается.",
        )

    resolved_key = resolve_api_key(
        key, credential_ref=credential_ref, explicit=api_key, store=store
    )
    # Only the fill-in custom provider and the local HTTP servers take an endpoint
    # from config; a branded provider keeps its pinned host so its traffic can
    # never be redirected.
    resolved_base_url = base_url.strip() if key in HOST_PROVIDERS else ""
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


def _create_llamacpp_client(
    *,
    model: str,
    temperature: float | None,
    max_tokens: int | None,
    extra: Mapping[str, Any] | None,
) -> LlmClient:
    """Build the in-process llama.cpp client from ``extra``.

    llama.cpp has no endpoint and no key: what it needs is the path to a ``.gguf``
    and a handful of runtime knobs (context size, thread and GPU-layer counts, the
    idle timeout and the RAM figures the §12 guard reads), and those ride in
    ``extra``. Importing the wrapper module is safe even without the native
    ``llama-cpp-python`` wheel — it defers the ``llama_cpp`` import until a model is
    actually loaded, so an unconfigured client is free to construct here.
    """
    from ayris.nlu.llm.llamacpp_client import DEFAULT_N_CTX, LlamaCppLlmClient, LlamaCppOptions

    data = dict(extra or {})
    options = LlamaCppOptions(
        model_path=_as_str(data.get("model_path")),
        model=model,
        n_ctx=_as_int(data.get("n_ctx"), DEFAULT_N_CTX),
        n_threads=_as_opt_int(data.get("n_threads")),
        n_gpu_layers=_as_int(data.get("n_gpu_layers"), 0),
        temperature=temperature,
        max_tokens=max_tokens,
        idle_sec=_as_float(data.get("idle_sec"), 0.0),
        ram_limit_mb=_as_int(data.get("ram_limit_mb"), 0),
        requires_ram_mb=_as_int(data.get("requires_ram_mb"), 0),
    )
    _log.debug("создан клиент llamacpp, модель %r", options.model_path or model)
    return LlamaCppLlmClient(options)


def _as_str(value: object) -> str:
    """A config value read as a string, or empty when it is anything else."""
    return value if isinstance(value, str) else ""


def _as_int(value: object, default: int) -> int:
    """A config value read as an int, or ``default`` when missing or not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return int(value)


def _as_opt_int(value: object) -> int | None:
    """A config value read as an int, or ``None`` when missing or not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _as_float(value: object, default: float) -> float:
    """A config value read as a float, or ``default`` when missing or not numeric."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)
