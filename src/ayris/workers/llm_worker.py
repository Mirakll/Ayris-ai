"""The LLM worker process: streams a model's answer as sentences and usage.

Generation is the slowest thing Ayris does and the one most worth getting off the
GUI thread: a cloud round trip is seconds of waiting, and a local model holds the
GIL for the length of every token. So the language model lives here, in its own
process on the task-05 worker infrastructure, and the parent stays free to paint
the overlay while the answer is still being written.

**The answer leaves as sentences, not as one blob.** :meth:`LlmWorker.chat` is a
single long call — the heartbeat thread proves liveness independently, so it does
not trip the supervisor's watchdog — and while it runs it emits a ``sentence``
event for every sentence the token stream completes. That is what lets TTS start
speaking the first sentence before the model has written the last, which is the
whole reason the LLM path streams. The per-token text deltas themselves never
reach the bus: like the TTS worker's ``metrics``, they would be one event per
subscriber per token and buy nothing. Only assembled sentences and the final
usage report go out.

**Cancellation reaches a call already running.** An ``@method()`` ``cancel`` would
queue behind the blocking ``chat`` on the single dispatch thread, so the cancel
that matters is the supervisor's out-of-band CANCEL control, read here through
:attr:`~ayris.workers.base.WorkerContext.cancelled`. The stream polls it between
chunks, closes the socket when it flips, and stops emitting sentences — «Айрис,
стоп» from the model's side. The ``cancel`` method is kept for the case where no
call is in flight and for symmetry with the other workers.

**The key never enters this module's state.** :func:`~ayris.nlu.llm.factory.create_llm_client`
reads it from the credential store when the client is built and hands the client a
plain string; the worker holds a client, not a key, and nothing here — no log, no
event, no reply — carries it. A provider with no key configured reports itself
unconfigured, and the worker answers with the «модель не настроена» sentence
rather than spending a doomed request, which also keeps the zero-telemetry promise:
no request leaves without the user having set a key.
"""

from __future__ import annotations

import threading
from time import perf_counter
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.core.errors import LlmError
from ayris.core.models import JsonObject
from ayris.nlu.llm.base import (
    NOT_CONFIGURED_MESSAGE,
    FinishReason,
    LlmDoneDelta,
    LlmMessage,
    LlmRole,
    LlmTextDelta,
    LlmTool,
    LlmToolCallDelta,
    LlmUsage,
    LlmUsageDelta,
    NullLlmClient,
)
from ayris.nlu.llm.cloud import READ_TIMEOUT_SEC, _decode_tool_call
from ayris.nlu.llm.factory import HOST_PROVIDERS, LLAMACPP_PROVIDER, create_llm_client
from ayris.nlu.llm.stream import SentenceAssembler
from ayris.nlu.llm.usage import UsageMeter, resolve_usage
from ayris.utils.logger import get_pipeline_logger
from ayris.workers.base import Worker, method

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayris.core.events import Event
    from ayris.nlu.llm.base import LlmClient
    from ayris.workers.base import WorkerContext

__all__ = ["EVENT_TRANSLATOR", "LlmWorker", "translate_llm_event"]


#: How often the idle watcher wakes to check for an unused local model. Mirrors
#: the STT worker: fine granularity for a timeout measured in minutes, cheap, and
#: it notices a timeout shortened by ``configure`` within one tick.
_IDLE_POLL: Final = 5.0

#: Eco mode multiplies the configured idle timeout by this, same as the STT
#: worker — under eco the model is not even warmed at start, but a lazily loaded
#: one is still dropped sooner.
_ECO_IDLE_FACTOR: Final = 0.5

#: Idle unload is never scheduled sooner than this, whatever the settings say:
#: reloading a multi-gigabyte GGUF between two turns of one conversation is not a
#: memory strategy.
_MIN_IDLE_SEC: Final = 30.0


class LlmWorker(Worker):
    """A language model in a worker process, answering as a stream of sentences.

    Args:
        context: Supplied by the runtime.
    """

    kind: ClassVar[str] = "llm"

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self._client: LlmClient | None = None
        #: ``(provider, model, credential_ref, base_url, model_path)`` the current
        #: client was built for. A configure that changes any of these drops the
        #: client so the next call rebuilds it — a model swap, or a new custom
        #: endpoint, must not keep talking to the old one. ``model_path`` is in
        #: there because the in-process engine is bound to a file, not to a name:
        #: two GGUFs can carry the same display model and swapping between them
        #: must still unload the one that was open.
        self._client_key: tuple[str, str, str, str, str] = ("", "", "", "", "")
        self._meter = UsageMeter()
        self._assembler = SentenceAssembler()
        #: Set by :meth:`cancel` and polled by the stream. Distinct from the
        #: supervisor's CANCEL (read through ``context.cancelled``): both stop the
        #: same request, this one covers a cancel with no call in flight yet.
        self._cancelled = threading.Event()
        self._calls = 0
        self._pipeline = get_pipeline_logger()
        #: Serialises building, warming and dropping the client across the
        #: dispatch thread, the startup warm-up thread and the idle watcher.
        #: Reentrant because a warm/configure that holds it also calls
        #: :meth:`_ensure_client`, which takes it too.
        self._model_lock = threading.RLock()
        #: ``True`` for the length of a streaming ``chat``. The idle watcher reads
        #: it so it never frees a local model's native memory mid-generation.
        self._busy = False
        #: The startup warm-up thread, when one is running (llama.cpp, non-eco).
        self._warming: threading.Thread | None = None
        self._idle_stop = threading.Event()
        self._idle_thread: threading.Thread | None = None

    # -- lifecycle ------------------------------------------------------

    def on_start(self) -> None:
        """Announce readiness, start the idle watcher, and warm a local model.

        Cloud and the local HTTP servers stay lazy: a worker that connected on
        start would break the zero-telemetry promise, and a provider with no key
        would fail a launch the user never asked to make a request in. The
        in-process llama.cpp engine is different — «запускается вместе с Айрис»
        means its GGUF is loaded off the request path — so when it is the provider,
        it is configured, and eco mode is off, a background thread opens the file
        now. That load reads the local disk only; no request leaves.
        """
        self._idle_thread = threading.Thread(
            target=self._idle_loop, name="ayris-llm-idle", daemon=True
        )
        self._idle_thread.start()
        self.log.info(
            "llm: воркер готов, провайдер %s, модель %s (клиент по требованию)",
            self._provider() or "—",
            self._model() or "по умолчанию",
        )
        if self._provider() == LLAMACPP_PROVIDER and not self._eco_mode():
            self._warming = threading.Thread(
                target=self._warm_in_background, name="ayris-llm-warm", daemon=True
            )
            self._warming.start()

    def on_stop(self) -> None:
        """Stop the watcher, let the warm-up finish, and close the client."""
        self._idle_stop.set()
        idle = self._idle_thread
        if idle is not None and idle.is_alive():
            idle.join(timeout=_IDLE_POLL + 1.0)
        self._idle_thread = None
        warming = self._warming
        if warming is not None and warming.is_alive():
            warming.join(timeout=1.0)
        self._close_client()

    def on_configure(self, params: JsonObject) -> None:
        """Drop the client when provider, model, endpoint or model file changed.

        Dropping it is what actually unloads a local GGUF: the handle goes with
        the client, and llama.cpp frees the native memory on close. Not while a
        generation streams, though — `chat` does not hold :attr:`_model_lock` for
        the length of the stream, so closing under it would pull native memory
        out from under a running inference. A busy worker is left alone and
        :meth:`_ensure_client` rebuilds on the next call, which runs before any
        stream starts.
        """
        del params
        with self._model_lock:
            if self._client is None or self._client_identity() == self._client_key:
                return
            if self._busy:
                self.log.info("llm: модель сменилась на лету, пересоздам клиент после ответа")
                return
            self.log.info("llm: настройки сменили провайдера или модель, пересоздаю клиент")
            self._close_client()

    # -- wire methods ---------------------------------------------------

    @method()
    def chat(self, params: JsonObject) -> JsonObject:
        """Answer a conversation, emitting a ``sentence`` event per sentence.

        Args:
            params: ``messages`` (a list of ``{role, content, ...}`` dicts),
                optional ``tools``, ``temperature``, ``max_tokens`` and
                ``request_id``.

        Returns:
            The whole answer once the stream ends — text, decoded ``tool_calls``,
            ``finish_reason``, the model actually used, ``cancelled`` and the
            ``usage`` record. The sentences themselves already went out as events
            while this ran; the returned ``text`` is for the caller that wants the
            full answer in one piece.

        Raises:
            ayris.core.errors.LlmError: The provider refused the request or the
                stream broke after the first token. A missing key is not an
                error — the answer is the «не настроено» sentence instead.
        """
        request_id = str(params.get("request_id", ""))
        messages = _messages(params)
        temperature = _as_opt_float(params.get("temperature"))
        max_tokens = _as_opt_int(params.get("max_tokens"))

        self._cancelled.clear()
        self._assembler.reset()
        self._calls += 1
        client = self._ensure_client()
        provider = self._provider()
        model_used = getattr(client, "model", "") or self._model()

        if not client.configured:
            return self._unconfigured_reply(client, request_id, provider, model_used)

        started = perf_counter()
        text_parts: list[str] = []
        frags: dict[int, dict[str, str]] = {}
        reported = LlmUsage()
        finish = FinishReason.STOP
        index = 0

        # Held for the length of the stream so the idle watcher never unloads a
        # local model out from under a generation in flight.
        self._busy = True
        try:
            for delta in client.stream(
                messages,
                _tools(params),
                temperature=temperature,
                max_tokens=max_tokens,
                cancel=self._should_cancel,
            ):
                if isinstance(delta, LlmTextDelta):
                    text_parts.append(delta.text)
                    for sentence in self._assembler.feed(delta.text):
                        self._emit_sentence(sentence, index, provider, request_id, final=False)
                        index += 1
                elif isinstance(delta, LlmToolCallDelta):
                    frag = frags.setdefault(delta.index, {"id": "", "name": "", "args": ""})
                    if delta.call_id:
                        frag["id"] = delta.call_id
                    if delta.name:
                        frag["name"] = delta.name
                    frag["args"] += delta.arguments
                elif isinstance(delta, LlmUsageDelta):
                    reported = delta.usage
                elif isinstance(delta, LlmDoneDelta):
                    finish = delta.finish_reason

            return self._finish_chat(
                provider=provider,
                model_used=model_used,
                request_id=request_id,
                messages=messages,
                text_parts=text_parts,
                frags=frags,
                reported=reported,
                finish=finish,
                index=index,
                started=started,
            )
        finally:
            self._busy = False

    def _finish_chat(
        self,
        *,
        provider: str,
        model_used: str,
        request_id: str,
        messages: list[LlmMessage],
        text_parts: list[str],
        frags: dict[int, dict[str, str]],
        reported: LlmUsage,
        finish: FinishReason,
        index: int,
        started: float,
    ) -> JsonObject:
        """Flush the tail, record usage, and build the reply once the stream ends."""
        cancelled = finish is FinishReason.CANCELLED or self._should_cancel()
        tail = self._assembler.flush()
        if not cancelled:
            last = len(tail) - 1
            for offset, sentence in enumerate(tail):
                self._emit_sentence(sentence, index, provider, request_id, final=offset == last)
                index += 1
        full_text = "".join(text_parts)
        tool_calls = _decode_tool_calls(frags)
        resolved, estimated = resolve_usage(provider, reported, messages, full_text)
        record = self._meter.record(
            provider, model_used, resolved, estimated=estimated, request_id=request_id
        )
        self.emit("usage", record.as_payload())
        duration_ms = int((perf_counter() - started) * 1000.0)
        self._pipeline.info(
            "llm: %s/%s → %d предл., %d ток. (%s), %.0f мс%s",
            provider or "—",
            model_used or "—",
            index,
            record.total_tokens,
            "оценка" if estimated else "точно",
            duration_ms,
            ", отменено" if cancelled else "",
            extra={"request_id": request_id},
        )
        return {
            "text": full_text,
            "tool_calls": tool_calls,
            "finish_reason": finish.value,
            "model": model_used,
            "engine": provider,
            "cancelled": cancelled,
            "sentences": index,
            "usage": record.as_payload(),
            "duration_ms": duration_ms,
            "request_id": request_id,
        }

    @method()
    def cancel(self, params: JsonObject) -> JsonObject:
        """Ask the current answer to stop.

        Sets the flag :meth:`chat` polls between chunks. A ``chat`` already running
        on the dispatch thread is reached by the supervisor's out-of-band CANCEL
        (``context.cancelled``), not by this method, which would queue behind it;
        this covers the case where no call is in flight yet and keeps the worker's
        surface symmetric with the STT and TTS ones. See the module docstring.
        """
        self._cancelled.set()
        self._assembler.reset()
        self.log.info("llm: запрошена отмена ответа")
        return {"cancelled": True, **_echo(params)}

    @method()
    def list_models(self, params: JsonObject) -> JsonObject:
        """The models the configured provider can reach, best-effort.

        Only ever called when the user opens the model picker, which is the
        explicit action that makes the one network call allowed here.
        """
        del params
        client = self._ensure_client()
        try:
            models = list(client.list_models())
        except LlmError as exc:
            self.log.warning("llm: список моделей недоступен: %s", exc.technical)
            models = []
        return {"models": models, "provider": self._provider(), "model": self._model()}

    @method()
    def check_key(self, params: JsonObject) -> JsonObject:
        """Whether the stored key works, probed cheaply and only on request.

        This is «проверить ключ» in settings — the one place a request leaves
        without the user having asked a question, and only because they pressed
        the button. An unconfigured provider answers ``ok: false`` and makes no
        call at all.
        """
        del params
        client = self._ensure_client()
        check = client.check_credentials()
        return {
            "ok": check.ok,
            "detail": check.detail,
            "models": list(check.models),
            "provider": self._provider(),
            "model": self._model(),
        }

    @method()
    def status(self, _params: JsonObject) -> JsonObject:
        """Session totals and identity, for the «ИИ» tab and DevTools."""
        client = self._client
        snapshot = self._meter.snapshot()
        return {
            "provider": self._provider(),
            "model": (getattr(client, "model", "") or self._model()) if client else self._model(),
            "configured": client.configured if client is not None else False,
            "calls": self._calls,
            **snapshot,
        }

    # -- client lifecycle ----------------------------------------------

    def _should_cancel(self) -> bool:
        """Whether the answer in flight should stop.

        Three sources: this worker's own ``cancel`` method, the supervisor's
        out-of-band CANCEL, and a pending stop. The stream polls it between
        chunks and the base client closes the socket when it flips.
        """
        return self._cancelled.is_set() or self.context.cancelled or self.context.stopping

    def _ensure_client(self) -> LlmClient:
        """The client for the current settings, rebuilt when the identity changed.

        Guarded by :attr:`_model_lock` so the startup warm-up thread and the
        dispatch thread cannot each build one: the loser would leak a native
        handle. The lock covers construction only, never a streaming call.
        """
        with self._model_lock:
            identity = self._client_identity()
            if self._client is None or identity != self._client_key:
                self._close_client()
                self._client = self._build_client()
                self._client_key = identity
            return self._client

    def _build_client(self) -> LlmClient:
        """Construct the provider's client, degrading to «не настроено» on failure.

        A key that is absent is not a failure here — the client comes back
        ``configured=False`` and :meth:`chat` answers with the hint sentence. An
        unknown provider name raises :class:`~ayris.core.errors.LlmError`, which
        becomes a :class:`~ayris.nlu.llm.base.NullLlmClient` for the same graceful
        answer.
        """
        provider = self._provider()
        try:
            client = create_llm_client(
                provider,
                model=self._model(),
                base_url=self._base_url(),
                credential_ref=self._credential_ref(),
                temperature=self._config_temperature(),
                max_tokens=self._config_max_tokens(),
                read_timeout_sec=self._read_timeout(),
                extra=self._extra(),
            )
        except LlmError as exc:
            self.log.warning("llm: провайдер «%s» недоступен: %s", provider or "—", exc.technical)
            return NullLlmClient()
        self.log.info(
            "llm: клиент %s готов (модель %s), ключ %s",
            provider or "—",
            getattr(client, "model", "") or self._model() or "по умолчанию",
            "есть" if client.configured else "нет",
        )
        return client

    def _close_client(self) -> None:
        """Close the current client and forget it. Safe when there is none."""
        with self._model_lock:
            client, self._client = self._client, None
            self._client_key = ("", "", "", "", "")
        if client is not None:
            try:
                client.close()
            except Exception:
                self.log.debug("llm: ошибка закрытия клиента", exc_info=True)

    def _warm_in_background(self) -> None:
        """Open the local GGUF now so the first chat does not pay the load.

        The thread body for the startup warm-up. Never raises out of the thread —
        there is no caller to raise at — so a load that fails becomes a log line
        and the client stays lazy, exactly as if the warm-up had not run. A client
        that is not configured (no file chosen yet) is left alone rather than
        forced.
        """
        try:
            client = self._ensure_client()
            warm = getattr(client, "warm", None)
            if not callable(warm):
                return
            if not client.configured:
                self.log.info(
                    "llm: локальная модель не готова к прогреву: %s",
                    getattr(client, "message", "") or "не настроено",
                )
                return
            started = perf_counter()
            if warm():
                self.log.info(
                    "llm: локальная модель прогрета при старте за %.0f мс",
                    (perf_counter() - started) * 1000.0,
                )
        except Exception:  # pragma: no cover - a warm-up bug must not kill the worker
            self.log.exception("llm: прогрев локальной модели не удался")

    def _client_identity(self) -> tuple[str, str, str, str, str]:
        """What a client is bound to: change any of these and it is rebuilt."""
        return (
            self._provider(),
            self._model(),
            self._credential_ref(),
            self._base_url(),
            self._model_file(),
        )

    # -- idle -----------------------------------------------------------

    def _idle_loop(self) -> None:
        """Poll for an idle local model until the worker stops."""
        while not self._idle_stop.wait(_IDLE_POLL):
            self._drop_if_idle()

    def _drop_if_idle(self) -> None:
        """Unload a local model that has sat unused past the idle timeout.

        Only the in-process llama.cpp client owns native memory worth freeing and
        exposes ``unload_if_idle``; cloud and HTTP clients have neither, so the
        duck-typed check simply no-ops for them. Never fires while a ``chat`` is
        streaming (:attr:`_busy`), and the client re-checks its own freshness
        under its load lock, so a generation in flight is safe twice over.
        """
        timeout = self._idle_timeout()
        if timeout <= 0.0:
            return
        with self._model_lock:
            if self._busy:
                return
            client = self._client
            unloader = getattr(client, "unload_if_idle", None)
            if not callable(unloader):
                return
            if unloader(idle_sec=timeout):
                self.log.info("llm: локальная модель выгружена по простою")

    def _idle_timeout(self) -> float:
        """Seconds of idleness before a local model is dropped. ``0`` disables it."""
        configured = _as_float(self.params.get("model_idle_sec"), 0.0)
        if configured <= 0.0:
            return 0.0
        if self._eco_mode():
            configured *= _ECO_IDLE_FACTOR
        return max(_MIN_IDLE_SEC, configured)

    def _eco_mode(self) -> bool:
        """``performance.eco_mode``: skip the startup warm-up and unload sooner."""
        return bool(self.params.get("eco_mode", False))

    def _emit_sentence(
        self,
        text: str,
        index: int,
        provider: str,
        request_id: str,
        *,
        final: bool,
    ) -> None:
        """Publish one finished sentence for TTS to start speaking."""
        self.emit(
            "sentence",
            {
                "text": text,
                "index": index,
                "final": final,
                "engine": provider,
                "request_id": request_id,
            },
        )

    def _unconfigured_reply(
        self,
        client: LlmClient,
        request_id: str,
        provider: str,
        model_used: str,
    ) -> JsonObject:
        """Answer a request when no model is set up, without touching the network."""
        message = getattr(client, "message", NOT_CONFIGURED_MESSAGE)
        self._emit_sentence(message, 0, provider, request_id, final=True)
        self.log.info("llm: модель не настроена, отвечаю подсказкой без запроса")
        return {
            "text": message,
            "tool_calls": [],
            "finish_reason": FinishReason.ERROR.value,
            "model": model_used,
            "engine": provider,
            "cancelled": False,
            "sentences": 1,
            "usage": {},
            "duration_ms": 0,
            "request_id": request_id,
            "configured": False,
        }

    # -- settings -------------------------------------------------------

    def _provider(self) -> str:
        """``ai.provider``, normalised the way the factory keys its table."""
        return str(self.params.get("provider", "")).strip().lower()

    def _model(self) -> str:
        """``ai.model``; empty lets the client pick its default."""
        return str(self.params.get("model", ""))

    def _model_file(self) -> str:
        """``ai.model_path`` as it arrives in ``extra``; empty for every HTTP provider.

        Part of the client identity: the in-process engine is bound to this file,
        so a swap of it has to close the old handle even when the display name
        and the endpoint did not move.
        """
        return str(self._extra().get("model_path", ""))

    def _credential_ref(self) -> str:
        """Name of the credential entry the key is read from. Never the key."""
        return str(self.params.get("credential_ref", ""))

    def _base_url(self) -> str:
        """Endpoint taken from ``ai.host`` for the host-bearing providers; empty else.

        Only the providers in :data:`~ayris.nlu.llm.factory.HOST_PROVIDERS` — the
        fill-in ``custom`` provider and the local Ollama and LM Studio servers —
        have no host of their own, so only they read ``host`` as a base URL.
        Handing a branded provider the local-server ``host`` would point its
        traffic at the wrong place, so it gets nothing and keeps its pinned default.
        """
        if self._provider() in HOST_PROVIDERS:
            return str(self.params.get("host", "")).strip()
        return ""

    def _config_temperature(self) -> float | None:
        """``ai.temperature`` as the client's default, overridable per call."""
        return _as_opt_float(self.params.get("temperature"))

    def _config_max_tokens(self) -> int | None:
        """``ai.max_tokens`` as the client's default, overridable per call."""
        return _as_opt_int(self.params.get("max_tokens"))

    def _read_timeout(self) -> float:
        """Between-chunk read timeout, from ``ai.request_timeout_sec``."""
        return _as_float(self.params.get("request_timeout_sec"), READ_TIMEOUT_SEC)

    def _extra(self) -> dict[str, object]:
        """Provider-specific settings (Yandex ``folder_id``, GigaChat ``scope``).

        Passed through verbatim to the factory. Empty until the config schema
        carries them, at which point they arrive here without another change.
        """
        extra = self.params.get("extra")
        return dict(extra) if isinstance(extra, dict) else {}


def _messages(params: JsonObject) -> list[LlmMessage]:
    """Read the conversation out of a request, skipping malformed turns."""
    raw = params.get("messages")
    if not isinstance(raw, list):
        return []
    messages: list[LlmMessage] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        messages.append(
            LlmMessage(
                role=_role_of(str(item.get("role", "user"))),
                content=str(item.get("content", "")),
                name=str(item.get("name", "")),
                tool_call_id=str(item.get("tool_call_id", "")),
            )
        )
    return messages


def _tools(params: JsonObject) -> list[LlmTool]:
    """Read the tool declarations out of a request, skipping nameless ones."""
    raw = params.get("tools")
    if not isinstance(raw, list):
        return []
    tools: list[LlmTool] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        if not name:
            continue
        parameters = item.get("parameters")
        tools.append(
            LlmTool(
                name=name,
                description=str(item.get("description", "")),
                parameters=parameters if isinstance(parameters, dict) else {},
            )
        )
    return tools


def _role_of(value: str) -> LlmRole:
    """Map a wire role string onto :class:`LlmRole`, defaulting to «user»."""
    try:
        return LlmRole(value.strip().lower())
    except ValueError:
        return LlmRole.USER


def _decode_tool_calls(frags: Mapping[int, Mapping[str, str]]) -> list[JsonObject]:
    """Turn the accumulated tool-call fragments into decoded calls for the reply.

    Uses the same decoder as :meth:`~ayris.nlu.llm.cloud.CloudLlmClient.complete`
    so a malformed argument string becomes an empty object the same way, rather
    than two subtly different behaviours for the streaming and non-streaming paths.
    """
    calls: list[JsonObject] = []
    for _, frag in sorted(frags.items()):
        if not frag.get("name"):
            continue
        call = _decode_tool_call(frag)
        calls.append(
            {"name": call.name, "arguments": dict(call.arguments), "call_id": call.call_id}
        )
    return calls


def _as_opt_float(value: object) -> float | None:
    """Read a params value as a float, or ``None`` when it is absent or wrong-typed."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_opt_int(value: object) -> int | None:
    """Read a params value as an int, or ``None`` when it is absent or wrong-typed."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _as_float(value: object, default: float) -> float:
    """Read a params value as a float, falling back to ``default``."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _echo(params: JsonObject) -> JsonObject:
    """Carry the caller's ``request_id`` into a reply that has nothing else."""
    request_id = str(params.get("request_id", ""))
    return {"request_id": request_id} if request_id else {}


def translate_llm_event(kind: str, payload: JsonObject) -> Event | None:
    """Turn an LLM worker event into a bus event.

    Only two kinds cross to the bus: ``sentence`` (what starts TTS early) and
    ``usage`` (what the «ИИ» tab shows). The token deltas never became events in
    the first place, so there is nothing else to translate.
    """
    if kind == "sentence":
        return _sentence_event(payload)
    if kind == "usage":
        return _usage_event(payload)
    return None


def _sentence_event(payload: JsonObject) -> Event | None:
    """Build :class:`~ayris.core.events.LlmSentenceReady` from a ``sentence`` event."""
    from ayris.core.events import LlmSentenceReady

    text = str(payload.get("text", ""))
    if not text:
        return None
    return LlmSentenceReady(
        text=text,
        index=int(payload.get("index", 0) or 0),
        final=bool(payload.get("final", False)),
        engine=str(payload.get("engine", "")),
        request_id=str(payload.get("request_id", "")),
    )


def _usage_event(payload: JsonObject) -> Event | None:
    """Build :class:`~ayris.core.events.LlmUsageReported` from a ``usage`` event."""
    from ayris.core.events import LlmUsageReported

    return LlmUsageReported(
        provider=str(payload.get("provider", "")),
        model=str(payload.get("model", "")),
        prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
        completion_tokens=int(payload.get("completion_tokens", 0) or 0),
        cost_usd=float(payload.get("cost_usd", 0.0) or 0.0),
        session_cost_usd=float(payload.get("session_cost_usd", 0.0) or 0.0),
        estimated=bool(payload.get("estimated", False)),
        request_id=str(payload.get("request_id", "")),
    )


#: What the supervisor registers for this worker.
EVENT_TRANSLATOR: Final = translate_llm_event
