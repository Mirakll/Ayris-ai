"""What an action gives back, and how the answer reaches the user.

One action, one :class:`ActionResult`. Three audiences read it and they want
different things, which is why it is not just a string:

* the **user** hears :attr:`ActionResult.message_ru` — one short Russian phrase,
  or nothing at all, because a volume change is its own confirmation and
  narrating it would be noise;
* the **caller** — a macro, the pipeline, a plugin — reads :attr:`ActionResult.ok`
  and :attr:`ActionResult.value`, the typed payload of that particular action.
  ``ListWindows`` returns window models, not lines of text, so the next block in
  the macro can filter them;
* the **log and the history** read :attr:`ActionResult.detail` and
  :attr:`ActionResult.duration_ms`.

The payload is typed through the class parameter: ``ActionResult[list[WindowInfo]]``
says what came back, and ``ActionResult[None]`` says the action only did
something. :attr:`ActionResult.duration_ms` is stamped by the registry, not by
the action — the action would have to measure itself, and every one of them would
measure it slightly differently.

:attr:`ActionResult.undo_token` is what makes ``supports_undo`` real. The action
returns an opaque string it can later make sense of (a previous volume, a window
placement, a clipboard entry id); the registry only carries it. Nothing tries to
guess what an undo means for an action that did not offer one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from ayris.core.models import ExecutionResult

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ayris.core.errors import ActionError

__all__ = [
    "ActionResult",
    "ValueT",
]

#: Payload type of a concrete action's result.
ValueT = TypeVar("ValueT")


@dataclass(frozen=True, slots=True)
class ActionResult(Generic[ValueT]):
    """Outcome of one action call.

    Args:
        ok: Whether the action did what it was asked. A failure that the action
            handled itself — «окно не найдено» — is a result with ``ok=False``,
            not an exception; an exception is for a failure it could not describe.
        value: Typed payload. ``None`` when the action only had an effect.
        message_ru: Russian phrase for TTS, the overlay and the history. Empty
            means "say nothing", which is the right default for most actions.
        detail: English one-liner for the log. Never spoken.
        duration_ms: Filled in by the registry around the call.
        undo_token: Opaque state for :meth:`ayris.actions.base.Action.undo`.
            ``None`` when this call cannot be undone, even if the action type can.
        data: Extra scalars for the trace and DevTools — window handle, process
            id, resolved path. Not for payload; that is what ``value`` is.
    """

    ok: bool = True
    value: ValueT | None = None
    message_ru: str = ""
    detail: str = ""
    duration_ms: int = 0
    undo_token: str | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    @property
    def undoable(self) -> bool:
        """Whether this particular call left something to undo."""
        return bool(self.undo_token)

    @property
    def execution(self) -> ExecutionResult:
        """How this call is recorded in ``history`` and ``audit``."""
        return ExecutionResult.OK if self.ok else ExecutionResult.ERROR

    def with_duration(self, duration_ms: int) -> ActionResult[ValueT]:
        """Copy with the measured duration stamped in."""
        return replace(self, duration_ms=max(0, duration_ms))

    @classmethod
    def done(
        cls,
        message_ru: str = "",
        *,
        value: ValueT | None = None,
        detail: str = "",
        undo_token: str | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> ActionResult[ValueT]:
        """A successful result. The shape almost every action returns."""
        return cls(
            ok=True,
            value=value,
            message_ru=message_ru,
            detail=detail,
            undo_token=undo_token,
            data=dict(data or {}),
        )

    @classmethod
    def failed(
        cls,
        message_ru: str,
        *,
        detail: str = "",
        value: ValueT | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> ActionResult[ValueT]:
        """A handled failure: the action knows what went wrong and can say it."""
        return cls(
            ok=False,
            value=value,
            message_ru=message_ru,
            detail=detail or message_ru,
            data=dict(data or {}),
        )

    @classmethod
    def from_error(cls, error: ActionError) -> ActionResult[ValueT]:
        """Turn a typed action error into a result.

        Used where a caller wants one uniform shape instead of a ``try``: the
        registry still raises, and this is the adapter for macros that continue
        the chain after a failed block (section 14).
        """
        return cls(ok=False, message_ru=error.user_message, detail=error.technical)
