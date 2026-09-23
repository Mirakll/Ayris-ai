"""Network blocks owned by the macro language.

**Why this block guards its own target.** ``WebRequest`` sends a URL a command author —
or a profile imported off the network — chose, and then hands the answer back into a
macro variable. That is the exact shape of an SSRF: a request that looks outbound but is
aimed at ``169.254.169.254`` to read cloud credentials, or at ``127.0.0.1`` / an RFC-1918
box to reach something the machine can see and the author never should. So every hop is
resolved and checked before a byte leaves: schemes other than http/https, loopback,
link-local, private, reserved, multicast and cloud-metadata addresses are refused with a
typed macro error. Redirects are followed by hand, one at a time, so a ``Location`` header
or a rebound DNS name pointing inward is caught on the hop it appears, not after the fetch.

A smart-home command that must reach the local network opts in explicitly with
``allow_local``; the cloud-metadata endpoints stay shut even then.
"""

from __future__ import annotations

import ipaddress
import json
import re
import socket
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlsplit

import httpx

from ayris.actions.macros.blocks.logic import BlockHandler, BlockRuntime, Flow, as_dict, as_int
from ayris.actions.macros.errors import MacroBlockError, MacroLimitError, MacroTimeoutError
from ayris.actions.macros.expressions import truthy

if TYPE_CHECKING:
    from ayris.actions.macros.schema import ActionBlock

__all__ = [
    "WEB_HANDLERS",
    "get_resolver",
    "get_transport",
    "run_web_request",
    "set_resolver",
    "set_transport",
]

MAX_RESPONSE_BYTES: Final = 1_000_000
#: How many redirects one request may follow before it is stopped: enough for the usual
#: http→https or trailing-slash bounce, few enough that a redirect loop cannot spin.
MAX_REDIRECTS: Final = 10

#: A resolver maps a host name to the addresses it points at. Injectable for the same
#: reason as :data:`_transport`: a test decides what a name resolves to without a real
#: lookup, and the guard is exercised offline.
HostResolver = Callable[[str], "list[str]"]
_IpAddress = ipaddress.IPv4Address | ipaddress.IPv6Address

#: Never reachable from a command, opt-in or not. Every major cloud answers instance
#: credentials on the link-local ``169.254.169.254`` (AWS also on ``fd00:ec2::254``);
#: reading it is the classic SSRF pay-off, so it stays blocked even inside the local net.
_CLOUD_METADATA: Final = frozenset(
    {ipaddress.ip_address("169.254.169.254"), ipaddress.ip_address("fd00:ec2::254")}
)

_transport: httpx.BaseTransport | None = None
_resolver: HostResolver | None = None
_PART: Final = re.compile(r"(?:^|\.)([A-Za-z_][A-Za-z0-9_-]*)|\[(\d+)\]")


def get_transport() -> httpx.BaseTransport | None:
    """The optional test transport; ``None`` means the real network."""
    return _transport


def set_transport(transport: httpx.BaseTransport | None) -> None:
    """Install a transport, principally so tests make no outbound requests."""
    global _transport
    _transport = transport


def get_resolver() -> HostResolver | None:
    """The optional host resolver; ``None`` means real DNS through :mod:`socket`."""
    return _resolver


def set_resolver(resolver: HostResolver | None) -> None:
    """Install a host resolver, so a test decides what a name resolves to offline."""
    global _resolver
    _resolver = resolver


def _host_addresses(host: str) -> list[_IpAddress]:
    """Every address ``host`` stands for: the literal itself, or what the resolver returns.

    A literal is checked as-is; a name is resolved, because a public-looking name that
    answers with ``127.0.0.1`` is the whole trick a raw scheme filter misses.

    Raises:
        MacroBlockError: DNS could not answer, or the name has no usable address.
    """
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        pass
    resolver = get_resolver()
    try:
        if resolver is not None:
            raw = list(resolver(host))
        else:
            raw = [
                str(info[4][0]) for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
            ]
    except OSError as exc:
        raise MacroBlockError(
            f"cannot resolve host {host!r}",
            user_message=f"Не удалось определить адрес узла «{host}».",
            cause=exc,
        ) from exc
    addresses: list[_IpAddress] = []
    for item in raw:
        try:
            addresses.append(ipaddress.ip_address(item))
        except ValueError:
            continue
    if not addresses:
        raise MacroBlockError(
            f"host {host!r} did not resolve to an address",
            user_message=f"Не удалось определить адрес узла «{host}».",
        )
    return addresses


def _is_internal(ip: _IpAddress) -> bool:
    """Whether ``ip`` is anything but a public, globally routable address."""
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _ensure_url_allowed(url: str, *, allow_local: bool) -> None:
    """Refuse a target an SSRF would want, before a single byte leaves the machine.

    Rejects any scheme other than http/https, then resolves the host and checks every
    address it points at. Cloud-metadata endpoints are refused always; every other
    internal address is refused unless ``allow_local`` let the command into the local
    network. Called for the first URL and again for each redirect target, which is what
    turns off DNS-rebind and redirect-to-internal.

    Raises:
        MacroBlockError: the scheme, the host, or one of its addresses is not allowed.
    """
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https"):
        raise MacroBlockError(
            f"unsupported URL scheme {parts.scheme!r}",
            user_message="WebRequest поддерживает только адреса http и https.",
        )
    host = parts.hostname
    if not host:
        raise MacroBlockError(
            f"URL without a host: {url!r}", user_message="В адресе запроса не указан узел."
        )
    for ip in _host_addresses(host):
        target: _IpAddress = ip
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            target = ip.ipv4_mapped
        if target in _CLOUD_METADATA:
            raise MacroBlockError(
                f"blocked cloud metadata address {ip}",
                user_message="Запрос к адресу облачных метаданных заблокирован.",
            )
        if not allow_local and _is_internal(target):
            raise MacroBlockError(
                f"blocked internal address {ip} for host {host!r}",
                user_message=(
                    f"Запрос к локальному адресу «{host}» заблокирован; "
                    "разрешите доступ к локальной сети в блоке, если он нужен."
                ),
            )


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
    """Perform one explicit GET/POST request and optionally store parsed JSON.

    The target is vetted by :func:`_ensure_url_allowed` before the first request and again
    before each redirect it would follow, so a URL, a DNS answer or a ``Location`` header
    pointing at loopback, the private network or a cloud-metadata endpoint is refused with
    a typed error instead of fetched. ``allow_local`` opens the private network for a
    smart-home command; the metadata endpoints stay shut regardless.
    """
    params = rt.context.fill(block.params)
    method = str(params.get("method", "GET")).upper()
    url = str(params.get("url", "")).strip()
    allow_local = truthy(params.get("allow_local", False))
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
            transport=get_transport(), timeout=timeout_ms / 1000, follow_redirects=False
        ) as client:
            request = client.build_request(
                method,
                url,
                headers=headers,
                json=body if isinstance(body, dict | list) else None,
                content=body if isinstance(body, str) else None,
            )
            for _hop in range(MAX_REDIRECTS + 1):
                _ensure_url_allowed(str(request.url), allow_local=allow_local)
                response = client.send(request)
                if response.next_request is None:
                    break
                request = response.next_request
            else:
                raise MacroLimitError(
                    "web_redirects",
                    MAX_REDIRECTS,
                    user_message="Слишком много перенаправлений в запросе.",
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
