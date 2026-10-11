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
from pathlib import Path
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
    from collections.abc import Mapping, Sequence

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
        "gemini": "ayris.nlu.llm.gemini_client:GeminiLlmClient",
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
    model_path = _resolve_gguf_path(
        _as_str(data.get("model_path")),
        search_dirs=_as_str_seq(data.get("model_search_dirs")),
        prefer=model,
    )
    options = LlamaCppOptions(
        model_path=model_path,
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


def _as_str_seq(value: object) -> tuple[str, ...]:
    """A config value read as a tuple of strings, dropping anything that is not one."""
    if not isinstance(value, (list | tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _resolve_gguf_path(
    model_path: str, *, search_dirs: Sequence[str] = (), prefer: str = ""
) -> str:
    """The ``.gguf`` to load, falling back to an installed one when the path is bad.

    A stale ``config.toml`` can name a ``.gguf`` that is not there, or carry an
    empty path after a reset (which a transient defaults-load can also produce) -
    either way the in-process engine would report «не выбран файл модели» and the
    assistant would answer with a hint instead of a reply. Rather than fail, this
    searches, in order: the profile's ``models/llm``, then ``search_dirs`` — the
    folders the user has actually kept models in, remembered from their past
    choices. Within a folder a file whose name matches ``prefer`` (the configured
    model) wins, so the fallback does not grab some other, heavier model that
    merely sorts first; otherwise the alphabetically first ``.gguf`` is used. The
    swap is logged. An honest, existing path is returned unchanged; when nothing
    is installed the original (empty or missing) path is handed back so the client
    reports its usual reason.
    """
    if model_path:
        try:
            if Path(model_path).is_file():
                return model_path
        except OSError:
            pass
    # When the bootstrap handed defaults, `prefer` is the schema's default model
    # name, not the user's. Also consult the model named in the persisted config
    # so the self-heal picks their model, not whatever sorts first on disk.
    wanted = [name for name in (prefer, _persisted_model_name()) if name]
    for directory in _gguf_search_dirs(search_dirs):
        picked = _pick_gguf(directory, wanted)
        if picked:
            if picked != model_path:
                _log.warning(
                    "llamacpp: файл модели %r недоступен, беру установленный %s",
                    model_path,
                    Path(picked).name,
                )
            return picked
    return model_path


def _persisted_model_name() -> str:
    """``ai.model`` from ``config.toml`` on disk, or ``""`` on any failure."""
    try:
        from ayris.core.config import get_settings

        return str(get_settings().ai.model)
    except Exception:
        return ""


def _gguf_search_dirs(extra: Sequence[str]) -> list[Path]:
    """Where to look for a replacement ``.gguf``: the profile dir, then the user's.

    De-duplicated, order preserved, non-directories dropped. The profile comes
    first so an installed-via-«ИИ» model wins; the remembered folders cover the
    common case of models kept outside the profile (e.g. ``D:\\ayris-models``).
    The persisted config is read as a last resort too: a worker spawned by a
    parent that momentarily held defaults gets empty ``extra`` here, but the file
    on disk still names the user's real model, so even then the engine recovers
    instead of reporting «не выбран файл модели».
    """
    candidates: list[Path] = []
    try:
        from ayris.core.paths import get_paths

        candidates.append(get_paths().llm_models_dir)
    except OSError:
        pass
    for raw in list(extra) + _dirs_from_persisted_config():
        try:
            candidates.append(Path(raw))
        except (OSError, ValueError):
            continue
    seen: dict[Path, None] = {}
    for candidate in candidates:
        try:
            if candidate.is_dir():
                seen.setdefault(candidate, None)
        except OSError:
            continue
    return list(seen)


def _dirs_from_persisted_config() -> list[str]:
    """Model folders read straight from ``config.toml`` on disk, best-effort.

    The safety net under the bootstrap params: those come from the parent's
    in-memory settings, which a transient defaults-load can empty, but the file
    the user actually edited still names their model. Returns the parent folder
    of the current ``ai.model_path`` and of every full path in ``ai.model_history``.
    Any failure (no file, a mid-write glimpse, a parse error) yields no dirs — the
    caller still has the profile dir and whatever ``extra`` carried.
    """
    try:
        from ayris.core.config import get_settings
        from ayris.nlu.llm.local_models import GGUF_SUFFIX

        ai = get_settings().ai
    except Exception:
        return []
    raw_paths: list[str] = []
    if ai.model_path:
        raw_paths.append(ai.model_path)
    for remembered in ai.model_history.values():
        raw_paths.extend(remembered)
    dirs: list[str] = []
    for raw in raw_paths:
        if GGUF_SUFFIX not in raw.lower():
            continue  # a display name, not a file path
        try:
            parent = str(Path(raw).parent)
        except (OSError, ValueError):
            continue
        if parent and parent != ".":
            dirs.append(parent)
    return dirs


def _pick_gguf(directory: Path, prefer: Sequence[str]) -> str:
    """A ``.gguf`` from ``directory``: one matching a ``prefer`` name, else the first.

    Each ``prefer`` name is matched against the file stem case-insensitively, both
    ways — a config name like ``Vikhr-Qwen-2.5-1.5b-Instruct-Q4_K_M`` is exactly
    the stem, but a looser name still matches when one contains the other. The
    names are tried in order (the bootstrap's, then the persisted config's), so
    the fallback does not grab some other, heavier model that merely sorts first.
    """
    from ayris.nlu.llm.local_models import GGUF_SUFFIX

    try:
        found = sorted(
            (item for item in directory.iterdir() if item.suffix.lower() == GGUF_SUFFIX),
            key=lambda item: item.name.casefold(),
        )
    except OSError:
        return ""
    if not found:
        return ""
    for raw in prefer:
        wanted = raw.strip().casefold()
        if not wanted:
            continue
        for item in found:
            stem = item.stem.casefold()
            if stem == wanted or wanted in stem or stem in wanted:
                return str(item)
    return str(found[0])


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
