"""WebRequest must not become an SSRF hole: the target guard and its per-hop redirect check.

The block sends an author-chosen URL and feeds the answer back into a macro variable, so a
URL aimed at ``169.254.169.254`` or ``127.0.0.1`` is a textbook server-side request forgery.
These tests pin the refusals (loopback, link-local, private, metadata, foreign schemes),
the explicit ``allow_local`` opt-in, and that a redirect to an internal address is caught on
the hop it appears rather than fetched.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from ayris.actions.macros.blocks.web import (
    MAX_REDIRECTS,
    _ensure_url_allowed,
    set_resolver,
    set_transport,
)
from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.errors import MacroBlockError, MacroLimitError
from ayris.actions.macros.report import ExecutionReport
from ayris.actions.macros.schema import CommandModel
from ayris.actions.result import ActionResult

pytestmark = pytest.mark.unit

# A stand-in public address every hostname test resolves to, so nothing touches real DNS.
_PUBLIC = "93.184.216.34"


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has(self, name: str) -> bool:
        return True

    def execute(
        self, name: str, params: dict[str, Any] | None = None, **_options: Any
    ) -> ActionResult[Any]:
        self.calls.append((name, dict(params or {})))
        return ActionResult.done()


def _command(**params: Any) -> CommandModel:
    return CommandModel(name="Сеть", actions=[{"type": "WebRequest", "params": params}])


class _Recorder:
    """A mock transport that remembers each request it was asked to make."""

    def __init__(self, responder: Any) -> None:
        self.requests: list[httpx.Request] = []
        self._responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self._responder(request)


def _ok(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"value": 42})


def _run(
    responder: Any = None, resolver: Any = None, **params: Any
) -> tuple[ExecutionReport, _Recorder]:
    recorder = _Recorder(responder or _ok)
    set_transport(httpx.MockTransport(recorder))
    if resolver is not None:
        set_resolver(resolver)
    with MacroEngine(FakeRegistry()) as engine:
        report = engine.run(_command(**params))
    return report, recorder


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    yield
    set_transport(None)
    set_resolver(None)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",  # loopback
        "http://[::1]/x",  # loopback IPv6
        "http://10.0.0.5/x",  # private RFC-1918
        "http://192.168.1.1/x",  # private RFC-1918
        "http://172.16.0.1/x",  # private RFC-1918
        "http://169.254.169.254/latest/meta-data/",  # link-local + метаданные
        "http://0.0.0.0/x",  # unspecified
        "http://224.0.0.1/x",  # multicast
    ],
)
def test_guard_refuses_internal_targets_by_default(url: str) -> None:
    with pytest.raises(MacroBlockError):
        _ensure_url_allowed(url, allow_local=False)


@pytest.mark.parametrize("scheme", ["file", "ftp", "gopher", "data", "ws"])
def test_guard_refuses_non_http_schemes(scheme: str) -> None:
    with pytest.raises(MacroBlockError):
        _ensure_url_allowed(f"{scheme}://example.com/x", allow_local=True)


def test_guard_refuses_url_without_host() -> None:
    with pytest.raises(MacroBlockError):
        _ensure_url_allowed("http:///no-host", allow_local=True)


def test_metadata_stays_blocked_even_with_local_opt_in() -> None:
    for url in ("http://169.254.169.254/latest/meta-data/", "http://[fd00:ec2::254]/x"):
        with pytest.raises(MacroBlockError):
            _ensure_url_allowed(url, allow_local=True)


def test_ipv4_mapped_metadata_is_blocked() -> None:
    with pytest.raises(MacroBlockError):
        _ensure_url_allowed("http://[::ffff:169.254.169.254]/x", allow_local=True)


def test_local_opt_in_allows_private_but_never_metadata() -> None:
    _ensure_url_allowed("http://192.168.0.10/api", allow_local=True)
    _ensure_url_allowed("http://10.1.2.3/api", allow_local=True)


def test_guard_resolves_a_public_name_pointing_inward() -> None:
    set_resolver(lambda _host: ["10.0.0.9"])  # DNS-rebind: публичное имя → приватный ответ
    with pytest.raises(MacroBlockError):
        _ensure_url_allowed("http://smart-home.example/api", allow_local=False)
    _ensure_url_allowed("http://smart-home.example/api", allow_local=True)  # opt-in пускает


def test_external_get_works_and_reaches_the_network_once() -> None:
    report, recorder = _run(
        resolver=lambda _host: [_PUBLIC],
        url="https://example.test/api",
        json_path="$.value",
        into="answer",
    )
    assert report.ok
    assert len(recorder.requests) == 1
    assert recorder.requests[0].method == "GET"


def test_external_post_works() -> None:
    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"seen": request.method})

    report, recorder = _run(
        responder=echo,
        resolver=lambda _host: [_PUBLIC],
        method="POST",
        url="https://example.test/api",
        body={"a": 1},
        json=True,
        into="answer",
    )
    assert report.ok
    assert recorder.requests[0].method == "POST"


def test_redirect_to_internal_is_refused_and_not_fetched() -> None:
    def redirect(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    report, recorder = _run(
        responder=redirect,
        resolver=lambda _host: [_PUBLIC],
        url="https://public.test/start",
        into="answer",
    )
    assert not report.ok
    assert isinstance(report.error, MacroBlockError)
    assert len(recorder.requests) == 1  # до внутреннего адреса запрос не дошёл


def test_redirect_to_public_is_followed() -> None:
    def hop(request: httpx.Request) -> httpx.Response:
        if request.url.host == "a.test":
            return httpx.Response(302, headers={"location": "https://b.test/final"})
        return httpx.Response(200, json={"value": 7})

    report, recorder = _run(
        responder=hop,
        resolver=lambda _host: [_PUBLIC],
        url="https://a.test/start",
        json_path="$.value",
        into="answer",
    )
    assert report.ok
    assert [request.url.host for request in recorder.requests] == ["a.test", "b.test"]


def test_private_request_refused_without_opt_in() -> None:
    report, recorder = _run(url="http://192.168.1.50/api", into="answer")
    assert not report.ok
    assert isinstance(report.error, MacroBlockError)
    assert recorder.requests == []


def test_local_opt_in_lets_a_private_request_through() -> None:
    report, recorder = _run(url="http://192.168.1.50/api", allow_local=True, into="answer")
    assert report.ok
    assert len(recorder.requests) == 1


def test_a_redirect_loop_stops_at_the_limit() -> None:
    def always_redirect(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "https://loop.test/next"})

    report, recorder = _run(
        responder=always_redirect,
        resolver=lambda _host: [_PUBLIC],
        url="https://loop.test/start",
    )
    assert not report.ok
    assert isinstance(report.error, MacroLimitError)
    assert len(recorder.requests) == MAX_REDIRECTS + 1
