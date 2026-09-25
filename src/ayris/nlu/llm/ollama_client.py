"""Talking to a local Ollama server over its native HTTP API.

Ollama is the friendliest way to run a model on your own machine: install it, run
``ollama serve``, and it answers on ``127.0.0.1:11434``. It is not OpenAI-shaped —
it streams newline-delimited JSON, not ``data:`` SSE, and it lists and pulls
models over its own endpoints — so this client subclasses
:class:`~ayris.nlu.llm.cloud.CloudLlmClient` for the HTTP plumbing (pooled client,
retry/backoff, the :class:`httpx.MockTransport` seam the tests inject) but frames
the request and reads the stream itself.

**No key, so «configured» means «reachable in principle».** A local server needs
no credential; :attr:`configured` is therefore always true. Whether it is *running
right now* is a different question, answered by :meth:`_ensure_reachable`, which
pings the server before every request and turns a refused connection into a clear
«Ollama не запущена по адресу …» with how to start it — not the branded
«проверьте интернет» sentence, which would be wrong for ``localhost``.

**Pulling a model reports progress.** :meth:`pull` streams Ollama's download
progress as :class:`OllamaPullProgress` records the caller turns into UI events;
before it starts it refuses, with an explanation, any recommended model whose RAM
need is over the configured limit (§12).

**Idle memory is the server's to reclaim, nudged by ``keep_alive``.** The chat
payload carries a ``keep_alive`` the worker derives from the idle timeout and
economy mode: a positive duration keeps the model warm, ``0`` unloads it the
moment the answer is done, so a cloud-first setup does not leave a local model
holding RAM it is not using.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from ayris.core.errors import LlmError
from ayris.core.models import JsonObject
from ayris.nlu.llm import catalog
from ayris.nlu.llm.base import (
    FinishReason,
    LlmDelta,
    LlmDoneDelta,
    LlmMessage,
    LlmTextDelta,
    LlmTool,
    LlmToolCallDelta,
    LlmUsage,
    LlmUsageDelta,
)
from ayris.nlu.llm.cloud import CloudLlmClient, CloudOptions, _loads_object
from ayris.utils.logger import get_logger

__all__ = ["DEFAULT_KEEP_ALIVE", "OllamaLlmClient", "OllamaPullProgress"]

_log = get_logger(__name__)

#: Default ``keep_alive``: keep the model warm for five minutes after an answer.
#: The worker overrides it with ``0`` under economy mode to unload immediately.
DEFAULT_KEEP_ALIVE = "5m"


@dataclass(frozen=True, slots=True)
class OllamaPullProgress:
    """One line of Ollama's ``/api/pull`` progress, ready to become a UI event."""

    status: str
    completed: int = 0
    total: int = 0
    digest: str = ""

    @property
    def fraction(self) -> float:
        """Downloaded share in ``[0, 1]``, or ``0`` while the total is unknown."""
        if self.total <= 0:
            return 0.0
        return max(0.0, min(1.0, self.completed / self.total))


class OllamaLlmClient(CloudLlmClient):
    """A local Ollama server, spoken to over its native ``/api`` endpoints."""

    name: ClassVar[str] = "ollama"
    title: ClassVar[str] = "Ollama"
    default_base_url: ClassVar[str] = "http://127.0.0.1:11434"
    default_model: ClassVar[str] = ""
    #: Ollama forwards tool definitions to models that support them.
    supports_tools: ClassVar[bool] = True

    def __init__(self, options: CloudOptions) -> None:
        super().__init__(options)
        keep_alive = options.extra.get("keep_alive", DEFAULT_KEEP_ALIVE)
        self._keep_alive: str | int = (
            keep_alive if isinstance(keep_alive, str | int) else DEFAULT_KEEP_ALIVE
        )
        ram_limit = options.extra.get("ram_limit_mb", 0)
        self._ram_limit_mb: int = int(ram_limit) if isinstance(ram_limit, int | float) else 0

    @property
    def configured(self) -> bool:
        # A local server needs no key; being *reachable* is checked per request.
        return bool(self._base_url)

    def _down_message(self) -> str:
        """The sentence the user reads when the server is not answering."""
        return (
            f"Ollama не запущена по адресу {self._base_url}. "
            "Запустите её командой «ollama serve» в терминале или откройте приложение "
            "Ollama, затем повторите."
        )

    def _ensure_reachable(self) -> None:
        """Fail fast, and clearly, when the local server is not answering.

        A refused connection on ``localhost`` returns almost instantly, so this
        costs nothing when the server is up and turns «down» into the right
        sentence — not the branded «проверьте интернет» — before the retrying
        chat request would otherwise spin through its backoff schedule.
        """
        client = self._require_client()
        try:
            client.get(
                f"{self._base_url}/api/version",
                headers={"Accept": "application/json"},
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LlmError(
                f"ollama: server unreachable at {self._base_url}: {exc}",
                user_message=self._down_message(),
            ) from exc

    # ------------------------------------------------------------------
    # the streaming contract, wrapped in a reachability ping
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        self._ensure_reachable()
        yield from super().stream(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            cancel=cancel,
        )

    def list_models(self) -> tuple[str, ...]:
        self._ensure_reachable()
        return super().list_models()

    # ------------------------------------------------------------------
    # provider hooks
    # ------------------------------------------------------------------

    def _endpoint(self) -> str:
        return f"{self._base_url}/api/chat"

    def _models_endpoint(self) -> str:
        return f"{self._base_url}/api/tags"

    def _auth_headers(self) -> Mapping[str, str]:
        return {}

    def _build_payload(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> JsonObject:
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            # Ollama caps generation with num_predict, not max_tokens.
            options["num_predict"] = max_tokens
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.as_payload() for message in messages],
            "stream": stream,
            "keep_alive": self._keep_alive,
        }
        if options:
            payload["options"] = options
        if tools:
            payload["tools"] = [tool.as_payload() for tool in tools]
        return payload

    def _read_stream(
        self,
        response: httpx.Response,
        cancel: Callable[[], bool],
    ) -> Iterator[LlmDelta]:
        for line in response.iter_lines():
            if cancel():
                return
            stripped = line.strip()
            if not stripped:
                continue
            chunk = _loads_object(stripped)
            if chunk is None:
                continue
            error = chunk.get("error")
            if isinstance(error, str) and error:
                raise LlmError(
                    f"{self.name}: {error}",
                    user_message=f"Ollama вернула ошибку: {error}",
                )
            message = chunk.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str) and content:
                    yield LlmTextDelta(text=content)
                yield from _ollama_tool_deltas(message.get("tool_calls"))
            if chunk.get("done") is True:
                usage = _ollama_usage(chunk)
                if usage is not None:
                    yield usage
                yield LlmDoneDelta(finish_reason=_map_ollama_finish(chunk.get("done_reason")))
                return

    def _parse_models(self, data: bytes) -> tuple[str, ...]:
        payload = _loads_object(data.decode("utf-8", errors="replace"))
        if payload is None:
            return ()
        items = payload.get("models")
        if not isinstance(items, list):
            return ()
        names = [
            item["name"]
            for item in items
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        return tuple(names)

    # ------------------------------------------------------------------
    # pulling a model, with progress
    # ------------------------------------------------------------------

    def pull(self, model: str = "") -> Iterator[OllamaPullProgress]:
        """Download ``model`` (default: the configured one), yielding progress.

        Refuses, before any bytes move, a recommended model whose RAM need is
        over the configured limit (§12), and turns Ollama's own ``{"error": …}``
        line into a typed :class:`LlmError` so a failed pull reads clearly rather
        than finishing «successfully» on half a file.
        """
        target = model.strip() or self._model
        if not target:
            raise LlmError(
                "ollama: no model to pull",
                user_message="Не указана модель для загрузки.",
            )
        spec = catalog.get_spec(target)
        if spec is not None:
            catalog.guard_ram_limit(spec, ram_limit_mb=self._ram_limit_mb)
        self._ensure_reachable()
        client = self._require_client()
        body = {"model": target, "stream": True}
        with client.stream(
            "POST",
            f"{self._base_url}/api/pull",
            json=body,
            headers={"Accept": "application/x-ndjson"},
        ) as response:
            if not (200 <= response.status_code < 300):
                self._raise_for_status(response)
            for line in response.iter_lines():
                stripped = line.strip()
                if not stripped:
                    continue
                chunk = _loads_object(stripped)
                if chunk is None:
                    continue
                error = chunk.get("error")
                if isinstance(error, str) and error:
                    raise LlmError(
                        f"{self.name}: pull failed: {error}",
                        user_message=f"Не удалось загрузить модель «{target}»: {error}",
                    )
                yield _pull_progress(chunk)


def _pull_progress(chunk: JsonObject) -> OllamaPullProgress:
    """Read one ``/api/pull`` JSON line into a progress record."""
    status = chunk.get("status")
    completed = chunk.get("completed")
    total = chunk.get("total")
    digest = chunk.get("digest")
    return OllamaPullProgress(
        status=status if isinstance(status, str) else "",
        completed=completed if isinstance(completed, int) else 0,
        total=total if isinstance(total, int) else 0,
        digest=digest if isinstance(digest, str) else "",
    )


def _ollama_tool_deltas(raw: object) -> Iterator[LlmToolCallDelta]:
    """Yield tool-call deltas from Ollama's ``message.tool_calls`` list.

    Ollama returns each call whole — ``{"function": {"name", "arguments": {…}}}``
    with the arguments already an object — rather than in string fragments, so the
    object is re-encoded to JSON to match the fragment contract the base
    :meth:`~ayris.nlu.llm.cloud.CloudLlmClient.complete` aggregator expects.
    """
    if not isinstance(raw, list):
        return

    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments")
        encoded = json.dumps(arguments, ensure_ascii=False) if isinstance(arguments, dict) else ""
        call_id = entry.get("id")
        yield LlmToolCallDelta(
            index=index,
            call_id=call_id if isinstance(call_id, str) else "",
            name=name,
            arguments=encoded,
        )


def _ollama_usage(chunk: JsonObject) -> LlmUsageDelta | None:
    """Read Ollama's ``prompt_eval_count`` / ``eval_count`` into a usage delta."""
    prompt = chunk.get("prompt_eval_count")
    completion = chunk.get("eval_count")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    return LlmUsageDelta(
        usage=LlmUsage(
            prompt_tokens=prompt if isinstance(prompt, int) else 0,
            completion_tokens=completion if isinstance(completion, int) else 0,
        )
    )


def _map_ollama_finish(reason: object) -> FinishReason:
    """Map Ollama's ``done_reason`` onto :class:`FinishReason`."""
    if reason == "length":
        return FinishReason.LENGTH
    return FinishReason.STOP
