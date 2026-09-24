"""GigaChat's TLS trust anchor: the bundled Russian National CA.

GigaChat's OAuth and API endpoints chain to the Russian National CA, which ships
in no default trust store, so a stock ``verify=True`` client rejects the chain.
:class:`GigaChatLlmClient` fixes this by trusting a PEM bundled under
``resources/certs`` *in addition* to the system roots — TLS stays on, and the
chain validates. These tests prove that wiring without opening a socket:

* :class:`TestBundle` — the shipped PEM is the two certs we vetted, by fingerprint.
* :class:`TestResolveVerify` — the client turns the bundle into an additive
  SSL context, honours ``verify=False``, and degrades to the system store when
  the bundle is missing or unreadable rather than disabling verification.
* :class:`TestBranded` — a provider with no extra CA verifies against certifi.
* :class:`TestTransportShortCircuit` — a mock transport skips the PEM entirely.
"""

from __future__ import annotations

import hashlib
import ssl
from pathlib import Path

import httpx
import pytest

from ayris.core.paths import executable_dir
from ayris.nlu.llm.cloud import CloudOptions
from ayris.nlu.llm.gigachat_client import GigaChatLlmClient
from ayris.nlu.llm.openai_client import OpenAiLlmClient

pytestmark = pytest.mark.unit

#: The two certificates the bundle must contain, by SHA-256 of their DER form.
#: Vetted against gu-st.ru and github.com/koenrh/russian-trusted-root-ca, and
#: functionally against the live Sber endpoints, before the bundle was committed.
ROOT_FP = "D26D2D0231B7C39F92CC738512BA54103519E4405D68B5BD703E9788CA8ECF31"
SUB_FP = "BBBDE2103E790B999EC62BD03CF625A5A2E7C316E10AFE6A490EEDEAD8B3FD9B"

BUNDLE = executable_dir() / "resources" / "certs" / "russian_trusted_ca.crt"


def _der_fingerprints(pem: str) -> set[str]:
    """SHA-256 (upper hex) of every certificate DER in a PEM bundle.

    Slices each block from its ``BEGIN`` marker so the file's header comments do
    not travel into :func:`ssl.PEM_cert_to_DER_cert`, which insists the string
    start at the marker.
    """
    begin, end = "-----BEGIN CERTIFICATE-----", "-----END CERTIFICATE-----"
    out: set[str] = set()
    cursor = 0
    while (start := pem.find(begin, cursor)) != -1:
        stop = pem.find(end, start) + len(end)
        block = pem[start:stop]
        out.add(hashlib.sha256(ssl.PEM_cert_to_DER_cert(block)).hexdigest().upper())
        cursor = stop
    return out


def _ctx_fingerprints(context: ssl.SSLContext) -> set[str]:
    """SHA-256 (upper hex) of every CA loaded into an SSL context."""
    return {
        hashlib.sha256(der).hexdigest().upper() for der in context.get_ca_certs(binary_form=True)
    }


def gigachat(**extra: object) -> GigaChatLlmClient:
    """A GigaChat client with a key but no live token — nothing is fetched here."""
    return GigaChatLlmClient(CloudOptions(api_key="unused", **extra))  # type: ignore[arg-type]


class TestBundle:
    """The shipped PEM is exactly the two certificates we vetted."""

    def test_bundle_is_present(self) -> None:
        assert BUNDLE.is_file(), f"missing CA bundle at {BUNDLE}"

    def test_bundle_holds_the_two_vetted_certs(self) -> None:
        fingerprints = _der_fingerprints(BUNDLE.read_text(encoding="utf-8"))
        assert fingerprints == {ROOT_FP, SUB_FP}

    def test_bundle_is_loadable_as_trust_material(self) -> None:
        context = ssl.create_default_context()
        # Raises ssl.SSLError if the PEM is malformed; that is the assertion.
        context.load_verify_locations(cadata=BUNDLE.read_text(encoding="utf-8"))


class TestResolveVerify:
    """The client turns the bundle into an additive SSL context — or degrades safely."""

    def test_gigachat_points_at_the_bundle(self) -> None:
        assert gigachat()._extra_ca_path() == BUNDLE

    def test_resolves_to_a_context_trusting_the_russian_ca(self) -> None:
        resolved = gigachat()._resolve_verify()
        assert isinstance(resolved, ssl.SSLContext)
        assert {ROOT_FP, SUB_FP} <= _ctx_fingerprints(resolved)

    def test_context_is_additive_not_a_replacement(self) -> None:
        # A plain default context must NOT know the Russian root; ours must — proving
        # the bundle is layered on top of the system store, not swapped in for it.
        plain = _ctx_fingerprints(ssl.create_default_context())
        assert ROOT_FP not in plain
        assert ROOT_FP in _ctx_fingerprints(gigachat()._resolve_verify())  # type: ignore[arg-type]

    def test_verify_false_is_honoured(self) -> None:
        assert gigachat(verify=False)._resolve_verify() is False

    def test_missing_bundle_falls_back_to_system_store(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = gigachat()
        monkeypatch.setattr(client, "_extra_ca_path", lambda: Path("Z:/does/not/exist.pem"))
        assert client._resolve_verify() is True

    def test_unreadable_bundle_falls_back_to_system_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        junk = tmp_path / "garbage.pem"
        junk.write_text("this is not a certificate", encoding="utf-8")
        client = gigachat()
        monkeypatch.setattr(client, "_extra_ca_path", lambda: junk)
        assert client._resolve_verify() is True


class TestBranded:
    """A provider with no extra CA verifies against the default store."""

    def test_openai_has_no_extra_ca(self) -> None:
        client = OpenAiLlmClient(CloudOptions(api_key="unused"))
        assert client._extra_ca_path() is None
        assert client._resolve_verify() is True


class TestTransportShortCircuit:
    """A mock transport skips the PEM entirely — tests never read it off disk."""

    def test_build_client_with_transport_never_resolves_a_ca(self) -> None:
        def boom() -> bool:
            raise AssertionError("_resolve_verify must not run when a transport is injected")

        client = gigachat(transport=httpx.MockTransport(lambda _request: httpx.Response(200)))
        client._resolve_verify = boom  # type: ignore[method-assign]
        built = client._build_client()  # must not raise
        built.close()
        client.close()

    def test_build_client_without_transport_resolves_the_ca(self) -> None:
        seen: list[bool] = []
        client = gigachat()
        original = client._resolve_verify

        def spy() -> bool | ssl.SSLContext:
            seen.append(True)
            return original()

        client._resolve_verify = spy  # type: ignore[method-assign]
        built = client._build_client()
        built.close()
        client.close()
        assert seen == [True]
