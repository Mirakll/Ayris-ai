"""Network blocks owned by the macro language."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Final

import httpx

from ayris.actions.macros.blocks.logic import BlockHandler, BlockRuntime, Flow, as_dict, as_int
from ayris.actions.macros.errors import MacroBlockError, MacroLimitError, MacroTimeoutError

if TYPE_CHECKING:
    from ayris.actions.macros.schema import ActionBlock

__all__ = ["WEB_HANDLERS", "get_transport", "run_web_request", "set_transport"]

MAX_RESPONSE_BYTES: Final = 1_000_000
_transport: httpx.BaseTransport | None = None
_PART: Final = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]")


def get_transport() -> httpx.BaseTransport | None:
    """The optional test transport; ``None`` means the real network."""
    return _transport


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Install a transport, principally so tests make no outbound requests."""
    global _transport
    _transport = transport


def _json_path(value: Any, path: str) -> Any:
    text = path.strip()
    if text in ("", "$"):
        return value
    if text.startswith("$"):
        text = text[1:]
    position = 0
    for match in _PART.finditer(text):
        if match.start() != position:
            raise MacroBlockError(
                f"invalid JSON path {path!r}", user_message="Некорректный JSON-path."
            )
        key, index = match.groups()
        try:
            value = value[int(index)] if index is not None else value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise MacroBlockError(
                f"JSON path {path!r} does not exist",
                user_message=f"JSON-path «{path}» не найден в ответе.",
                cause=exc,
            ) from exc
        position = match.end()
    if position != len(text):
        raise MacroBlockError(f"invalid JSON path {path!r}", user_message="Некорректный JSON-path.")
    return value


def run_web_request(rt: BlockRuntime, block: ActionBlock, _path: str, _depth: int) -> Flow:
    """Perform one explicit GET/POST request and optionally store parsed JSON."""
    params = rt.context.fill(block.params)
    method = str(params.get("method", "GET")).upper()
    url = str(params.get("url", "")).strip()
    if method not in {"GET", "POST"}:
        raise MacroBlockError(
            f"method {method!r}", user_message="WebRequest поддерживает только GET и POST."
        )
    if not url:
        raise MacroBlockError("empty URL", user_message="Укажите адрес запроса.")
    timeout_ms = as_int(params.get("timeout_ms"), 10_000)
    max_bytes = as_int(params.get("max_bytes"), MAX_RESPONSE_BYTES)
    if timeout_ms <= 0 or max_bytes <= 0 or max_bytes > MAX_RESPONSE_BYTES:
        raise MacroLimitError(
            "web_response_bytes", max_bytes, user_message="Недопустимые ограничения WebRequest."
        )
    headers = {str(key): str(value) for key, value in as_dict(params.get("headers")).items()}
    body = params.get("body")
    try:
        with httpx.Client(
            transport=get_transport(), timeout=timeout_ms / 1000, follow_redirects=True
        ) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                json=body if isinstance(body, dict | list) else None,
                content=body if isinstance(body, str) else None,
            )
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise MacroTimeoutError(timeout_ms / 1000) from exc
    except httpx.HTTPError as exc:
        raise MacroBlockError(
            str(exc), user_message="Веб-запрос не выполнился.", cause=exc
        ) from exc
    if len(response.content) > max_bytes:
        raise MacroLimitError(
            "web_response_bytes", max_bytes, user_message="Ответ сайта оказался слишком большим."
        )
    parse_json = bool(params.get("json", False) or params.get("json_path"))
    try:
        result: Any = response.json() if parse_json else response.text
    except json.JSONDecodeError as exc:
        raise MacroBlockError(
            "response is not JSON", user_message="Сайт вернул не JSON.", cause=exc
        ) from exc
    if params.get("json_path"):
        result = _json_path(result, str(params["json_path"]))
    into = str(params.get("into", "")).strip()
    if into:
        rt.context.set(into, result)
    rt.context.set_result(result)
    rt.report.note(f"{method} {response.status_code}")
    return Flow.NEXT


WEB_HANDLERS: Final[dict[str, BlockHandler]] = {"WebRequest": run_web_request}
