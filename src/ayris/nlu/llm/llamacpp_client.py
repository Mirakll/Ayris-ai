"""Running a GGUF file in-process with llama.cpp.

This is the offline path with no server in front of it: the model file the model
manager downloaded (task 14) is loaded straight into Ayris through
``llama-cpp-python``. That binding pulls a native, platform-specific binary and
is the one dependency Nuitka cannot trace, so it is **optional and imported
lazily, in this one module**. A machine without it does not crash on start — the
client simply reports itself unconfigured, with a sentence saying what to install,
and the router falls through to whatever else is available.

**«Configured» is a real, local check.** Unlike a cloud client, there is no key;
instead this asks whether the binding is present, a ``.gguf`` is selected, the
file exists, and the model's RAM need is within the configured limit (§12). Any of
those failing yields a specific :attr:`message` the «ИИ» tab can show, instead of
a load that swaps the machine to a halt or throws deep in native code.

**The model is loaded lazily and can be dropped when idle.** Opening a GGUF is the
slow «первый ответ» the task warns about, so it happens on first use, not at
construction; :meth:`unload_if_idle` lets a supervisor free the handle after the
configured idle timeout so a cloud-first session does not hold gigabytes of a
model it is not using.

**The native handle is reached only through a small Protocol.** The real
``llama_cpp.Llama`` is imported by string so mypy never needs its stubs, and the
tests inject a fake with the same one method — which is how the streaming path is
covered without the native wheel or a real GGUF on disk.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, ClassVar, Final, Protocol

from ayris.core.errors import LlmError
from ayris.core.models import JsonObject
from ayris.nlu.llm.base import (
    FinishReason,
    LlmClient,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmResponse,
    LlmTextDelta,
    LlmTool,
)
from ayris.nlu.llm.cloud import _map_openai_finish
from ayris.utils.logger import get_logger

__all__ = ["LlamaCppLlmClient", "LlamaCppOptions", "LlamaFactory", "llama_cpp_available"]

_log = get_logger(__name__)

#: Context window used when the config leaves it unset.
DEFAULT_N_CTX: Final = 4096

#: Sampling temperature when neither the request nor the config sets one; llama.cpp
#: itself demands a float, so ``None`` is resolved to this rather than passed on.
DEFAULT_TEMPERATURE: Final = 0.7


class _ChatModel(Protocol):
    """The one method this client needs off a loaded ``llama_cpp.Llama``.

    Narrow on purpose: the real handle has a large surface, but everything here
    goes through OpenAI-shaped chat completion, and pinning the Protocol to that
    call is what lets a test inject a fake without the native wheel.
    """

    def create_chat_completion(
        self,
        *,
        messages: Sequence[JsonObject],
        temperature: float,
        max_tokens: int | None,
        stream: bool,
    ) -> Iterator[Mapping[str, Any]]: ...


#: How the client obtains a loaded model. The default imports ``llama_cpp`` and
#: opens the GGUF; a test passes its own to answer with a fake instead.
LlamaFactory = Callable[["LlamaCppOptions"], _ChatModel]

_available: bool | None = None


def _never() -> bool:
    """The default ``cancel`` predicate: nothing ever cancels."""
    return False


def llama_cpp_available() -> bool:
    """Whether the optional ``llama-cpp-python`` binding can be imported.

    Uses :func:`importlib.util.find_spec`, so it answers without importing the
    native library (and paying its load cost) — enough to decide whether the
    engine is offered at all. Cached: the answer does not change within a run.
    """
    global _available
    if _available is None:
        try:
            _available = importlib.util.find_spec("llama_cpp") is not None
        except (ImportError, ValueError):
            _available = False
    return _available


@dataclass(frozen=True, slots=True)
class LlamaCppOptions:
    """Everything needed to open and run one GGUF, resolved before construction."""

    model_path: str = ""
    model: str = ""
    n_ctx: int = DEFAULT_N_CTX
    n_threads: int | None = None
    n_gpu_layers: int = 0
    temperature: float | None = None
    max_tokens: int | None = None
    idle_sec: float = 0.0
    ram_limit_mb: int = 0
    requires_ram_mb: int = 0


def _default_factory(options: LlamaCppOptions) -> _ChatModel:
    """Import ``llama_cpp`` by name and open the model file.

    Imported by string so mypy is never asked for stubs the optional dependency
    does not ship, and only reached once the client has confirmed the binding is
    installed.
    """
    module = importlib.import_module("llama_cpp")
    handle: _ChatModel = module.Llama(
        model_path=options.model_path,
        n_ctx=options.n_ctx,
        n_threads=options.n_threads,
        n_gpu_layers=options.n_gpu_layers,
        verbose=False,
    )
    return handle


class LlamaCppLlmClient(LlmClient):
    """A GGUF model run in-process through the optional llama.cpp binding."""

    name: ClassVar[str] = "llamacpp"
    supports_streaming: ClassVar[bool] = True
    supports_tools: ClassVar[bool] = False

    def __init__(self, options: LlamaCppOptions, *, factory: LlamaFactory | None = None) -> None:
        self._options = options
        self._factory: LlamaFactory = factory or _default_factory
        # An injected factory stands in for the native binding, so «available»
        # and the on-disk file check are both satisfied by its mere presence.
        self._injected = factory is not None
        self._handle: _ChatModel | None = None
        self._last_used = 0.0

    # ------------------------------------------------------------------
    # availability and configuration
    # ------------------------------------------------------------------

    def available(self) -> bool:
        """Whether the runtime can be used at all (binding present or injected)."""
        return self._injected or llama_cpp_available()

    @property
    def configured(self) -> bool:
        return not self._unconfigured_reason()

    @property
    def message(self) -> str:
        """Why the client is unconfigured, or empty when it is ready.

        Read by the worker to answer «только ИИ» with a specific sentence rather
        than the generic «не настроено» when the real reason is a missing binding,
        a missing file or a RAM limit.
        """
        return self._unconfigured_reason()

    def _unconfigured_reason(self) -> str:
        if not self.available():
            return (
                "Движок llama.cpp недоступен: не установлен пакет «llama-cpp-python». "
                "Установите его или выберите Ollama в разделе «ИИ»."
            )
        if not self._options.model_path:
            return "Не выбран файл модели (.gguf) для llama.cpp."
        if not self._injected and not Path(self._options.model_path).exists():
            return f"Файл модели не найден: {self._options.model_path}."
        if not self._ram_fits():
            return (
                f"Модели нужно около {self._options.requires_ram_mb} МБ, "
                f"а лимит памяти — {self._options.ram_limit_mb} МБ. "
                "Выберите модель полегче или поднимите лимит в разделе «Производительность»."
            )
        return ""

    def _ram_fits(self) -> bool:
        limit = self._options.ram_limit_mb
        need = self._options.requires_ram_mb
        if limit <= 0 or need <= 0:
            return True
        return need <= limit

    # ------------------------------------------------------------------
    # the streaming contract
    # ------------------------------------------------------------------

    def _handle_model(self) -> _ChatModel:
        """Load the model on first use, then reuse the held handle."""
        if self._handle is None:
            reason = self._unconfigured_reason()
            if reason:
                raise LlmError(f"llamacpp: not configured: {reason}", user_message=reason)
            try:
                self._handle = self._factory(self._options)
            except Exception as exc:
                raise LlmError(
                    f"llamacpp: failed to load {self._options.model_path!r}: {exc}",
                    user_message="Не удалось загрузить локальную модель.",
                ) from exc
        return self._handle

    def _pick_temperature(self, override: float | None) -> float:
        picked = override if override is not None else self._options.temperature
        return picked if picked is not None else DEFAULT_TEMPERATURE

    def _pick_max_tokens(self, override: int | None) -> int | None:
        return override if override is not None else self._options.max_tokens

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),  # noqa: ARG002 - tools unsupported here
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        predicate = cancel or _never
        model = self._handle_model()
        payload = [message.as_payload() for message in messages]
        self._last_used = monotonic()
        finish_reason: str | None = None
        try:
            chunks = model.create_chat_completion(
                messages=payload,
                temperature=self._pick_temperature(temperature),
                max_tokens=self._pick_max_tokens(max_tokens),
                stream=True,
            )
            for chunk in chunks:
                if predicate():
                    yield LlmDoneDelta(finish_reason=FinishReason.CANCELLED)
                    return
                text, reason = _read_chunk(chunk)
                if text:
                    yield LlmTextDelta(text=text)
                if reason is not None:
                    finish_reason = reason
        except LlmError:
            raise
        except Exception as exc:
            raise LlmError(
                f"llamacpp: generation failed: {exc}",
                user_message="Локальная модель прервала ответ с ошибкой.",
            ) from exc
        finally:
            self._last_used = monotonic()
        yield LlmDoneDelta(finish_reason=_map_openai_finish(finish_reason))

    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        started = perf_counter()
        text_parts: list[str] = []
        finish = FinishReason.STOP
        for delta in self.stream(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            cancel=cancel,
        ):
            if isinstance(delta, LlmTextDelta):
                text_parts.append(delta.text)
            elif isinstance(delta, LlmDoneDelta):
                finish = delta.finish_reason
        return LlmResponse(
            text="".join(text_parts),
            model=self._options.model or Path(self._options.model_path).stem,
            engine=self.name,
            finish_reason=finish,
            duration_ms=int((perf_counter() - started) * 1000.0),
        )

    # ------------------------------------------------------------------
    # idle memory management
    # ------------------------------------------------------------------

    def unload_if_idle(self, *, idle_sec: float | None = None, now: float | None = None) -> bool:
        """Drop the loaded model when it has sat unused past the idle timeout.

        Returns whether anything was freed. ``idle_sec`` defaults to the value the
        client was built with; ``0`` (or less) means «keep loaded», so a session
        that wants the model warm never has it pulled out from under it.
        """
        window = self._options.idle_sec if idle_sec is None else idle_sec
        if self._handle is None or window <= 0.0:
            return False
        current = monotonic() if now is None else now
        if current - self._last_used < window:
            return False
        self.close()
        return True

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        closer = getattr(handle, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:  # pragma: no cover - defensive
                _log.debug("llamacpp: closing the model failed: %s", exc)


def _read_chunk(chunk: Mapping[str, Any]) -> tuple[str, str | None]:
    """Pull the text fragment and any finish reason out of one stream chunk."""
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", None
    choice = choices[0]
    if not isinstance(choice, dict):
        return "", None
    delta = choice.get("delta")
    text = ""
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            text = content
    reason = choice.get("finish_reason")
    return text, reason if isinstance(reason, str) else None
