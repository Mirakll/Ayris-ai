"""The contract every language model client implements.

Task 63 brings the eight providers of ``ai.provider``; this module is what the
pipeline talks to in the meantime and after. The shape is deliberately the one
every chat API already has — a list of messages in, one answer out, optionally
with tool calls — so a provider is a subclass and nothing above it changes.

Two things here exist for the sake of the pipeline rather than of the providers.
:class:`NullLlmClient` stands in while no model is configured and answers with a
short Russian sentence saying exactly that, which is what «только ИИ» and
«гибрид» modes must say out loud instead of failing silently. And ``complete()``
takes a ``cancel`` predicate: a provider is expected to poll it between chunks,
because a model that has started generating is the longest thing the pipeline
ever waits for, and «отмена» has to reach it.

Messages are frozen dataclasses rather than dicts so mypy checks the role at the
call site; :meth:`LlmMessage.as_payload` produces the dict the wire wants.

Two answer shapes live here. ``complete()`` returns the whole :class:`LlmResponse`
at once and is what the pipeline calls today. ``stream()`` yields
:class:`LlmDelta` fragments as they arrive — text, tool-call pieces, a usage
report, a terminal done marker — and is what lets TTS start speaking the first
sentence before the model has finished. A provider implements one and gets the
other free: the base ``stream()`` wraps ``complete()`` and the cloud clients of
task 61 wrap ``complete()`` around a real ``stream()``, so callers pick the shape
they need and no provider has to write both.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Final, Self, TypeAlias

from ayris.core.models import JsonObject

__all__ = [
    "NOT_CONFIGURED_MESSAGE",
    "CredentialCheck",
    "FinishReason",
    "LlmClient",
    "LlmDelta",
    "LlmDoneDelta",
    "LlmMessage",
    "LlmResponse",
    "LlmRole",
    "LlmTextDelta",
    "LlmTool",
    "LlmToolCall",
    "LlmToolCallDelta",
    "LlmUsage",
    "LlmUsageDelta",
    "NullLlmClient",
]

#: What Ayris says when a model is asked for and none is set up. Deliberately
#: names the settings tab: the user who turned «только ИИ» on is one click away
#: from the provider fields, and «ИИ не настроен» alone leaves them guessing.
NOT_CONFIGURED_MESSAGE: Final = (
    "Языковая модель не настроена. Включите её в настройках, в разделе «ИИ»."
)


class LlmRole(StrEnum):
    """Who said a message. The three roles every provider understands."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(StrEnum):
    """Why generation stopped.

    ``CANCELLED`` is Ayris's own: no provider reports it, the client sets it when
    the ``cancel`` predicate returned true mid-stream.
    """

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class LlmMessage:
    """One turn of the conversation."""

    role: LlmRole
    content: str
    name: str = ""
    tool_call_id: str = ""

    @classmethod
    def system(cls, content: str) -> Self:
        return cls(role=LlmRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Self:
        return cls(role=LlmRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> Self:
        return cls(role=LlmRole.ASSISTANT, content=content)

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str, name: str = "") -> Self:
        """The result of a tool call, fed back for the next turn."""
        return cls(role=LlmRole.TOOL, content=content, name=name, tool_call_id=tool_call_id)

    def as_payload(self) -> JsonObject:
        """The dict the provider's API expects, without the empty fields."""
        payload: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class LlmTool:
    """A function the model may ask Ayris to run.

    In hybrid NLU mode the tools are the user's own commands, so ``name`` is a
    command identifier and ``parameters`` is its slot schema.
    """

    name: str
    description: str = ""
    parameters: JsonObject = field(default_factory=dict)

    def as_payload(self) -> JsonObject:
        """OpenAI-style function declaration; the other providers accept it too."""
        schema = self.parameters or {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }


@dataclass(frozen=True, slots=True)
class LlmToolCall:
    """The model's request to run one tool.

    ``arguments`` is already decoded. A provider that returns a JSON string
    parses it before constructing this, so the pipeline never sees the raw text
    and cannot forget to handle a malformed one.
    """

    name: str
    arguments: JsonObject = field(default_factory=dict)
    call_id: str = ""


@dataclass(frozen=True, slots=True)
class LlmUsage:
    """Tokens the request cost. Shown on the «ИИ» tab, never sent anywhere."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LlmResponse:
    """One answer from a model."""

    text: str = ""
    tool_calls: Sequence[LlmToolCall] = ()
    model: str = ""
    engine: str = ""
    finish_reason: FinishReason = FinishReason.STOP
    usage: LlmUsage = field(default_factory=LlmUsage)
    duration_ms: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether there is neither text nor a tool call to act on."""
        return not self.text.strip() and not self.tool_calls

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def cancelled(self) -> bool:
        return self.finish_reason is FinishReason.CANCELLED


@dataclass(frozen=True, slots=True)
class LlmTextDelta:
    """A fragment of the answer's text, in the order it was generated."""

    text: str


@dataclass(frozen=True, slots=True)
class LlmToolCallDelta:
    """A fragment of a tool call the model is assembling.

    Providers stream tool calls the same way they stream text — a bit at a time —
    so ``arguments`` here is a raw JSON *fragment*, not decoded and not
    necessarily complete on its own. ``index`` groups fragments of the same call
    when the model builds several at once; the assembler concatenates by index
    and decodes once the run finishes.
    """

    index: int = 0
    call_id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass(frozen=True, slots=True)
class LlmUsageDelta:
    """Token counts, usually arriving once near the end of the stream."""

    usage: LlmUsage


@dataclass(frozen=True, slots=True)
class LlmDoneDelta:
    """The stream ended; ``finish_reason`` says why.

    Exactly one of these terminates every stream, including a cancelled or failed
    one, so a consumer can flush its sentence buffer and report usage without
    guessing whether more is coming.
    """

    finish_reason: FinishReason = FinishReason.STOP


#: One piece of a streamed answer. A stream is a sequence of these ending in
#: exactly one :class:`LlmDoneDelta`.
LlmDelta: TypeAlias = LlmTextDelta | LlmToolCallDelta | LlmUsageDelta | LlmDoneDelta


@dataclass(frozen=True, slots=True)
class CredentialCheck:
    """The result of :meth:`LlmClient.check_credentials`.

    ``ok`` is the only field the caller must read; ``detail`` is a short Russian
    explanation for the settings dialog when it is ``False``, and ``models`` is
    whatever the check happened to learn, so a successful check can populate the
    model picker without a second round trip.
    """

    ok: bool
    detail: str = ""
    models: tuple[str, ...] = ()


def _response_to_deltas(response: LlmResponse) -> Iterator[LlmDelta]:
    """Replay a finished :class:`LlmResponse` as the stream it could have been.

    The base :meth:`LlmClient.stream` uses this so a provider that only wrote
    ``complete()`` still satisfies the streaming contract — one text chunk, the
    tool calls, a usage report, then the terminal marker.
    """
    if response.text:
        yield LlmTextDelta(text=response.text)
    for index, call in enumerate(response.tool_calls):
        yield LlmToolCallDelta(
            index=index,
            call_id=call.call_id,
            name=call.name,
            arguments=json.dumps(call.arguments, ensure_ascii=False),
        )
    if response.usage.total_tokens:
        yield LlmUsageDelta(usage=response.usage)
    yield LlmDoneDelta(finish_reason=response.finish_reason)


class LlmClient(ABC):
    """What every provider implements.

    Subclasses set :attr:`name` to the value of ``ai.provider`` they serve, so the
    factory of task 63 can pick one by configuration without a chain of
    ``isinstance`` checks.
    """

    #: Value of ``ai.provider`` this client serves.
    name: ClassVar[str] = ""

    #: Whether this client streams token deltas natively. ``False`` means
    #: :meth:`stream` falls back to replaying a finished :meth:`complete`, so the
    #: first sentence only reaches TTS once the whole answer is in.
    supports_streaming: ClassVar[bool] = False

    #: Whether this client can pass ``tools`` to the model and return tool calls.
    supports_tools: ClassVar[bool] = False

    #: Whether this client can reach a model at all. ``False`` on
    #: :class:`NullLlmClient` and on a provider missing its API key, and the
    #: pipeline checks it before building a prompt.
    @property
    def configured(self) -> bool:
        return True

    @abstractmethod
    def complete(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> LlmResponse:
        """Answer ``messages``, optionally choosing one of ``tools``.

        Args:
            messages: The conversation so far, oldest first, system prompt
                included.
            tools: Functions the model may call. Empty means «answer in words».
            temperature: Overrides ``ai.temperature`` for this request.
            max_tokens: Overrides ``ai.max_tokens`` for this request.
            cancel: Polled while generating; returning ``True`` must stop the
                request and return a response with
                :attr:`FinishReason.CANCELLED` rather than raise.

        Returns:
            The answer. Empty text with no tool calls is a valid outcome and the
            caller decides what to say about it.

        Raises:
            ayris.core.errors.LlmError: The provider was unreachable, refused the
                request or returned something unparseable.
        """

    def stream(
        self,
        messages: Sequence[LlmMessage],
        tools: Sequence[LlmTool] = (),
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancel: Callable[[], bool] | None = None,
    ) -> Iterator[LlmDelta]:
        """Answer ``messages``, yielding the answer in pieces as it is generated.

        The arguments mirror :meth:`complete`. The stream is a sequence of
        :class:`LlmDelta` — text fragments, tool-call fragments and an optional
        usage report — terminated by exactly one :class:`LlmDoneDelta`, whose
        :attr:`FinishReason` is :attr:`FinishReason.CANCELLED` when ``cancel``
        returned true. A provider polls ``cancel`` between fragments and closes
        the connection when it flips.

        The base implementation delegates to :meth:`complete` and replays the
        finished answer, so a non-streaming provider still satisfies the
        contract; a real streaming provider overrides this and wraps
        :meth:`complete` around it instead.

        Raises:
            ayris.core.errors.LlmError: Same failures as :meth:`complete`.
        """
        response = self.complete(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            cancel=cancel,
        )
        yield from _response_to_deltas(response)

    def list_models(self) -> tuple[str, ...]:
        """The models this client can reach, best-effort and possibly empty.

        A provider that can enumerate its catalogue overrides this; the default
        returns nothing, which the settings UI shows as «список недоступен»
        rather than an error. Any network call this makes counts as an explicit
        user action — the UI only calls it when the user opens the picker.
        """
        return ()

    def check_credentials(self) -> CredentialCheck:
        """Whether the configured key actually works, without generating.

        The default answers from :attr:`configured` alone and makes no request.
        A provider overrides this to probe cheaply — a models list, a tokenizer
        call — but only ever when the user pressed «проверить», never on start.
        """
        return CredentialCheck(ok=self.configured)

    def close(self) -> None:  # noqa: B027 - optional hook, most providers hold nothing
        """Release sockets and unload a local runtime. Safe to call twice."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, configured={self.configured})"


class NullLlmClient(LlmClient):
    """The client used while no model is set up.

    Answers every request with :data:`NOT_CONFIGURED_MESSAGE` instead of raising,
    and that is the point: in «только ИИ» mode every utterance goes to the model,
    so raising here would turn a configuration gap into an error on every single
    phrase. A plain sentence the user can act on is the better failure.
    """

    name: ClassVar[str] = "none"

    __slots__ = ("message",)

    def __init__(self, message: str = NOT_CONFIGURED_MESSAGE) -> None:
        self.message = message

    @property
    def configured(self) -> bool:
        return False

    def complete(
        self,
        messages: Sequence[LlmMessage],  # noqa: ARG002 - the contract, unused by design
        tools: Sequence[LlmTool] = (),  # noqa: ARG002
        *,
        temperature: float | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        cancel: Callable[[], bool] | None = None,  # noqa: ARG002
    ) -> LlmResponse:
        return LlmResponse(
            text=self.message,
            model="",
            engine=self.name,
            finish_reason=FinishReason.ERROR,
        )
