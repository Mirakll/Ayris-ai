"""Notification blocks that communicate through the event bus."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ayris.actions.macros.blocks.logic import BlockHandler, BlockRuntime, Flow, as_int
from ayris.core.events import LogLine, NotificationRequested

if TYPE_CHECKING:
    from ayris.actions.macros.schema import ActionBlock

__all__ = ["NOTIFY_HANDLERS", "run_overlay_log", "run_toast_notify"]


def run_toast_notify(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    params = rt.context.fill(block.params)
    rt.publish(
        NotificationRequested(
            title=str(params.get("title", "Ayris")),
            message=str(params.get("message", "")),
            level=str(params.get("level", "info")),
            timeout_ms=as_int(params.get("timeout_ms"), 5000),
            icon=str(params.get("icon", "")),
            action=str(params.get("action", "")),
        )
    )
    rt.report.note(str(params.get("title", "Ayris")))
    return Flow.NEXT


def run_overlay_log(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    params = rt.context.fill(block.params)
    message = str(params.get("message", ""))
    rt.publish(
        LogLine(level=str(params.get("level", "info")), message=message, logger="ayris.macro")
    )
    rt.report.note(message)
    return Flow.NEXT


NOTIFY_HANDLERS: Final[dict[str, BlockHandler]] = {
    "ToastNotify": run_toast_notify,
    "OverlayLog": run_overlay_log,
}
