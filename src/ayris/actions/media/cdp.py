"""A WebSocket small enough to keep, so CDP costs no dependency.

The advanced half of task 29 drives the Яндекс Музыка desktop app — Electron, so
Chrome — by clicking buttons in its own interface. The way in is the Chrome
DevTools Protocol: one HTTP request to list the pages, then a WebSocket to the one
we want, then ``Runtime.evaluate`` with a line of JavaScript.

The protocol needs no library. What Ayris sends is a text frame with a JSON object
in it and what comes back is the same; that is a hundred lines, all of it here,
against a third-party package in the installer, on the update treadmill, and inside
the signature check. The one part that is easy to get wrong is masking — every
client frame must be XOR-masked with four random bytes, and a server that receives
an unmasked frame is required to hang up — so :func:`encode_frame` and
:func:`decode_frame` are pure functions with a round-trip test.

Two details cost an evening each when this was first proved out against the app:

* **No ``Origin`` header.** Chromium 111 and later reject a DevTools socket that
  arrives with an ``Origin`` it did not expect, and every WebSocket library sends
  one by default (``websocket-client`` needs ``suppress_origin=True``). Handwriting
  the handshake means simply not sending it.
* **No ``permessage-deflate``.** Not offering the extension means never having to
  inflate a frame, and DevTools is happy to speak uncompressed.

What this module does *not* do is decide anything about Яндекс Музыка: no
selectors, no routes, no launching. That is :mod:`ayris.actions.media.yandex_music`.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import secrets
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol
from urllib.parse import urlsplit

from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils.logger import get_logger

__all__ = [
    "CdpClient",
    "CdpError",
    "CdpTransport",
    "CdpUnavailable",
    "Frame",
    "PageTarget",
    "RecordingCdp",
    "accept_token",
    "connect",
    "decode_frame",
    "encode_frame",
    "find_page",
    "is_port_open",
    "list_targets",
    "parse_targets",
    "pick_page",
]

_log = get_logger(__name__)

#: Frame kinds from RFC 6455 that this client can meet.
OP_CONTINUATION: Final = 0x0
OP_TEXT: Final = 0x1
OP_BINARY: Final = 0x2
OP_CLOSE: Final = 0x8
OP_PING: Final = 0x9
OP_PONG: Final = 0xA

#: The magic string every WebSocket server appends to the client's key before
#: hashing it. Fixed by RFC 6455, section 1.3.
_WS_GUID: Final = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

#: Payload-length escapes: 126 means «two more bytes», 127 means «eight more».
_LEN_16: Final = 126
_LEN_64: Final = 127

#: Beyond this a frame is either a bug or an attempt to exhaust memory. A DevTools
#: answer carrying a whole DOM tree is a few megabytes; 64 is room enough.
MAX_FRAME_BYTES: Final = 64 * 1024 * 1024

#: The desktop app's own page. Electron serves the interface from a custom scheme
#: rather than from ``https://``, which makes it trivial to recognise among the
#: service targets DevTools also lists.
MUSIC_PAGE_PREFIX: Final = "music-application://"

#: Seconds to wait for the debug port, and for one ``Runtime.evaluate`` to answer.
#: The port answers in single-digit milliseconds when it is there at all; a click
#: that has to wait for a page to render is slower, and 15 covers a cold render on a
#: loaded machine.
DEFAULT_HTTP_TIMEOUT: Final = 2.0
DEFAULT_WS_TIMEOUT: Final = 15.0


class CdpUnavailable(ActionUnavailable):
    """There is no debug port to talk to, or no page behind it.

    Its own class because this is the condition the media actions degrade on: with
    no CDP, transport control still works through SMTC, and the difference must be
    tellable from an ordinary failure.
    """

    default_user_message = (
        "Яндекс Музыка запущена без отладочного порта, поэтому доступны только "
        "пауза, плей и переключение треков."
    )


class CdpError(ActionError):
    """The protocol answered, and the answer was a failure."""

    default_user_message = "Не получилось управлять Яндекс Музыкой."


@dataclass(frozen=True, slots=True)
class Frame:
    """One WebSocket frame, decoded."""

    opcode: int
    payload: bytes
    final: bool = True


@dataclass(frozen=True, slots=True)
class PageTarget:
    """One entry of ``/json/list``, trimmed to what matters here."""

    id: str
    title: str
    url: str
    ws_url: str

    @property
    def is_music(self) -> bool:
        """Whether this is the app's own interface rather than a service page."""
        return self.url.startswith(MUSIC_PAGE_PREFIX)


class CdpTransport(Protocol):
    """The one thing :mod:`ayris.actions.media.yandex_music` needs from CDP.

    Narrow on purpose: an action builds a line of JavaScript and wants its value
    back. Everything a test needs to stand in for is therefore one method, and
    :class:`RecordingCdp` is that stand-in.
    """

    def evaluate(self, expression: str) -> Any:
        """Run ``expression`` in the page and return its value."""
        ...


@dataclass
class RecordingCdp:
    """Answers from a list, and writes down what it was asked. Test seam."""

    results: list[Any] = field(default_factory=list)
    expressions: list[str] = field(default_factory=list)

    def evaluate(self, expression: str) -> Any:
        self.expressions.append(expression)
        if not self.results:
            return None
        # A single queued answer answers everything: most tests care about one call
        # and should not have to pad the list to match an implementation detail.
        if len(self.results) == 1:
            return self.results[0]
        return self.results.pop(0)


def accept_token(key: str) -> str:
    """The ``Sec-WebSocket-Accept`` a correct server returns for ``key``."""
    digest = hashlib.sha1(f"{key}{_WS_GUID}".encode(), usedforsecurity=False).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, *, opcode: int = OP_TEXT, mask: bytes | None = None) -> bytes:
    """One final, masked frame. ``mask`` is only for tests; production randomises.

    Raises:
        ValueError: ``mask`` is given and is not exactly four bytes.
    """
    if mask is None:
        mask = secrets.token_bytes(4)
    elif len(mask) != 4:
        raise ValueError(f"mask must be 4 bytes, got {len(mask)}")
    header = bytearray()
    header.append(0x80 | opcode)  # FIN set: nothing here is ever fragmented
    size = len(payload)
    if size < _LEN_16:
        header.append(0x80 | size)
    elif size <= 0xFFFF:
        header.append(0x80 | _LEN_16)
        header += size.to_bytes(2, "big")
    else:
        header.append(0x80 | _LEN_64)
        header += size.to_bytes(8, "big")
    header += mask
    masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return bytes(header) + masked


def decode_frame(read: Callable[[int], bytes]) -> Frame:
    """Read one frame through ``read(n)``, which must return exactly ``n`` bytes.

    Server frames are unmasked by the standard, but a masked one is decoded anyway:
    it costs two lines and saves a mystery.

    Raises:
        CdpError: the frame claims a payload larger than :data:`MAX_FRAME_BYTES`.
    """
    first, second = read(2)
    final = bool(first & 0x80)
    opcode = first & 0x0F
    masked = bool(second & 0x80)
    size = second & 0x7F
    if size == _LEN_16:
        size = int.from_bytes(read(2), "big")
    elif size == _LEN_64:
        size = int.from_bytes(read(8), "big")
    if size > MAX_FRAME_BYTES:
        raise CdpError(f"refusing a {size} byte frame")
    mask = read(4) if masked else b""
    payload = read(size) if size else b""
    if mask:
        payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
    return Frame(opcode=opcode, payload=payload, final=final)


def parse_targets(payload: str | bytes) -> tuple[PageTarget, ...]:
    """``/json/list`` into targets. Pure, so a recorded answer is a fixture.

    Entries without a socket URL are dropped: DevTools lists a target that is
    already being debugged by someone else without one, and it is unusable.
    """
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError) as exc:
        raise CdpUnavailable(f"/json/list is not JSON: {exc}") from exc
    if not isinstance(raw, list):
        raise CdpUnavailable(f"/json/list is a {type(raw).__name__}, expected a list")
    targets: list[PageTarget] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("type") != "page":
            continue
        ws_url = str(item.get("webSocketDebuggerUrl") or "")
        if not ws_url:
            continue
        targets.append(
            PageTarget(
                id=str(item.get("id") or ""),
                title=str(item.get("title") or ""),
                url=str(item.get("url") or ""),
                ws_url=ws_url,
            )
        )
    return tuple(targets)


def pick_page(
    targets: Sequence[PageTarget],
    *,
    prefix: str = MUSIC_PAGE_PREFIX,
) -> PageTarget | None:
    """The app's own interface among the listed pages.

    The interface first, by its scheme; failing that the first page at all, which is
    what makes this reusable for a Chrome started with the same flag.
    """
    for target in targets:
        if target.url.startswith(prefix):
            return target
    return targets[0] if targets else None


def is_port_open(port: int, *, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    """Whether anything listens there. No exception, because it is a question.

    Used before launching the app: a running app with the flag needs no launch, and
    a running app without it must not be restarted behind the user's back.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def list_targets(
    port: int,
    *,
    host: str = "127.0.0.1",
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> tuple[PageTarget, ...]:
    """Ask the debug port what pages it has.

    Raises:
        CdpUnavailable: nothing is listening, or the answer was not a target list.
    """
    url = f"http://{host}:{port}/json/list"
    try:
        # Адрес собран здесь же из host и port, схема — литерал http.
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise CdpUnavailable(f"{url} did not answer: {exc}") from exc
    return parse_targets(payload)


def find_page(
    port: int,
    *,
    host: str = "127.0.0.1",
    prefix: str = MUSIC_PAGE_PREFIX,
    timeout: float = DEFAULT_HTTP_TIMEOUT,
) -> PageTarget:
    """The page to talk to.

    Raises:
        CdpUnavailable: no debug port, or it has no page.
    """
    targets = list_targets(port, host=host, timeout=timeout)
    target = pick_page(targets, prefix=prefix)
    if target is None:
        raise CdpUnavailable(f"debug port {port} lists no page")
    return target


class CdpClient:
    """One WebSocket to one DevTools page, and ``Runtime.evaluate`` over it.

    A context manager, and meant to be short-lived: an action opens it, evaluates
    one or two expressions, and closes it. Keeping it open across commands would
    mean handling the app's own reconnects, and there is nothing to gain — the
    handshake is a single round trip on loopback.
    """

    def __init__(self, ws_url: str, *, timeout: float = DEFAULT_WS_TIMEOUT) -> None:
        self._ws_url = ws_url
        self._timeout = timeout
        self._socket: socket.socket | None = None
        self._buffer = bytearray()
        self._next = 0

    def __enter__(self) -> CdpClient:
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        """Open the socket and complete the handshake.

        Raises:
            CdpUnavailable: the page is gone, or refused the upgrade.
        """
        parts = urlsplit(self._ws_url)
        host = parts.hostname or "127.0.0.1"
        port = parts.port or 9222
        path = parts.path or "/"
        if parts.query:
            path = f"{path}?{parts.query}"
        try:
            sock = socket.create_connection((host, port), timeout=self._timeout)
        except OSError as exc:
            raise CdpUnavailable(f"cannot reach {host}:{port}: {exc}") from exc
        sock.settimeout(self._timeout)
        # Nagle would hold a small JSON frame back waiting for company; every
        # message here is small and is followed by a wait for the answer.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self._socket = sock
        self._buffer.clear()
        shaken = False
        try:
            sock.sendall(request.encode("ascii"))
            self._check_handshake(self._read_headers(), key)
            shaken = True
        except OSError as exc:
            raise CdpUnavailable(f"websocket handshake failed: {exc}") from exc
        finally:
            # Рукопожатие рвётся тремя разными исключениями: `OSError` от сокета,
            # `CdpUnavailable` от отказа в апгрейде и `CdpError`, если порт замолчал
            # посреди заголовков. Сокет надо закрыть в любом из трёх случаев.
            if not shaken:
                self.close()
        _log.debug("CDP connected to %s", self._ws_url)

    def close(self) -> None:
        """Say goodbye if we can, hang up either way."""
        sock, self._socket = self._socket, None
        if sock is None:
            return
        # Страница могла уйти первой — это не наша беда, закрываем сокет всё равно.
        with contextlib.suppress(OSError):
            sock.sendall(encode_frame(b"", opcode=OP_CLOSE))
        with contextlib.suppress(OSError):
            sock.close()

    def evaluate(
        self,
        expression: str,
        *,
        await_promise: bool = True,
        user_gesture: bool = True,
    ) -> Any:
        """Run ``expression`` in the page and return its value.

        ``await_promise`` lets an expression be an ``async`` arrow function, which
        is how anything that has to wait for the interface to render is written.
        ``user_gesture`` marks the evaluation as user-initiated: Chromium gates some
        media and clipboard behaviour on that, and a click on a play button is a
        user's wish by definition.

        Raises:
            CdpUnavailable: the socket is not open.
            CdpError: the page raised, or the protocol refused.
        """
        answer = self.call(
            "Runtime.evaluate",
            {
                "expression": expression,
                "returnByValue": True,
                "awaitPromise": await_promise,
                "userGesture": user_gesture,
            },
        )
        details = answer.get("exceptionDetails")
        if details:
            raise CdpError(f"JavaScript raised: {_exception_text(details)}")
        result = answer.get("result") or {}
        return result.get("value")

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One protocol call, waiting for the answer to this one and no other.

        Raises:
            CdpUnavailable: the socket is not open.
            CdpError: the protocol returned an error, or the page hung up.
        """
        if self._socket is None:
            raise CdpUnavailable("CDP client is not connected")
        self._next += 1
        message_id = self._next
        self._send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            message = self._receive()
            if message.get("id") != message_id:
                continue  # an event, or the answer to a call we gave up on
            error = message.get("error")
            if error:
                raise CdpError(f"{method} failed: {error.get('message', error)}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}

    def _send(self, text: str) -> None:
        sock = self._socket
        if sock is None:
            raise CdpError("the debug socket is already closed")
        try:
            sock.sendall(encode_frame(text.encode("utf-8")))
        except OSError as exc:
            raise CdpError(f"could not send: {exc}") from exc

    def _receive(self) -> dict[str, Any]:
        """The next JSON message, reassembling fragments and answering pings."""
        parts: list[bytes] = []
        while True:
            try:
                frame = decode_frame(self._read_exactly)
            except OSError as exc:
                raise CdpError(f"could not read: {exc}") from exc
            if frame.opcode == OP_CLOSE:
                self.close()
                raise CdpError("the page closed the debug socket")
            if frame.opcode == OP_PING:
                self._pong(frame.payload)
                continue
            if frame.opcode == OP_PONG:
                continue
            if frame.opcode not in (OP_TEXT, OP_BINARY, OP_CONTINUATION):
                _log.debug("ignoring websocket opcode %#x", frame.opcode)
                continue
            parts.append(frame.payload)
            if not frame.final:
                continue
            payload = b"".join(parts)
            parts.clear()
            try:
                message = json.loads(payload)
            except ValueError as exc:
                raise CdpError(f"malformed CDP message: {exc}") from exc
            if isinstance(message, dict):
                return message
            raise CdpError(f"CDP sent a {type(message).__name__}, expected an object")

    def _pong(self, payload: bytes) -> None:
        if self._socket is None:
            return
        try:
            self._socket.sendall(encode_frame(payload, opcode=OP_PONG))
        except OSError as exc:  # a ping we cannot answer is a socket already gone
            _log.debug("could not pong: %s", exc)

    def _read_exactly(self, count: int) -> bytes:
        """Exactly ``count`` bytes, buffering whatever else arrives with them."""
        while len(self._buffer) < count:
            self._buffer += self._recv()
        chunk = bytes(self._buffer[:count])
        del self._buffer[:count]
        return chunk

    def _read_headers(self) -> bytes:
        """Everything up to the blank line that ends the HTTP response."""
        while b"\r\n\r\n" not in self._buffer:
            self._buffer += self._recv()
        head, _, rest = bytes(self._buffer).partition(b"\r\n\r\n")
        self._buffer = bytearray(rest)
        return head

    def _recv(self) -> bytes:
        sock = self._socket
        if sock is None:
            raise CdpError("the debug socket is already closed")
        try:
            chunk = sock.recv(65536)
        except TimeoutError as exc:
            raise CdpError(f"the page did not answer in {self._timeout:g} s") from exc
        if not chunk:
            raise CdpError("the debug socket closed")
        return chunk

    @staticmethod
    def _check_handshake(head: bytes, key: str) -> None:
        """A 101 with the right ``Sec-WebSocket-Accept``, or a readable refusal."""
        text = head.decode("latin-1")
        status = text.split("\r\n", 1)[0]
        if " 101 " not in f" {status} ":
            raise CdpUnavailable(f"upgrade refused: {status}")
        expected = accept_token(key).casefold()
        for line in text.split("\r\n")[1:]:
            name, _, value = line.partition(":")
            if name.strip().casefold() == "sec-websocket-accept":
                if value.strip().casefold() == expected:
                    return
                raise CdpUnavailable("Sec-WebSocket-Accept does not match the key")
        raise CdpUnavailable("no Sec-WebSocket-Accept in the upgrade response")


def _exception_text(details: dict[str, Any]) -> str:
    """The readable part of ``exceptionDetails``, whichever field carries it."""
    thrown = details.get("exception") or {}
    return str(
        thrown.get("description") or thrown.get("value") or details.get("text") or "unknown error"
    )


def connect(
    port: int,
    *,
    host: str = "127.0.0.1",
    prefix: str = MUSIC_PAGE_PREFIX,
    timeout: float = DEFAULT_WS_TIMEOUT,
) -> CdpClient:
    """Find the app's page on ``port`` and open a connected client to it.

    Raises:
        CdpUnavailable: no debug port, no page, or the upgrade was refused.
    """
    target = find_page(port, host=host, prefix=prefix)
    client = CdpClient(target.ws_url, timeout=timeout)
    client.connect()
    return client
