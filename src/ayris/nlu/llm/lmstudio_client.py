"""LM Studio, reached through its OpenAI-compatible server.

LM Studio runs a model locally and serves it at ``http://localhost:1234/v1`` in
exactly the OpenAI ``/chat/completions`` shape, so this is the thinnest client in
the package: it *is* :class:`~ayris.nlu.llm.cloud.OpenAiCompatibleClient`, changed
in only two places.

**No key.** LM Studio does not check authorization by default, so the ``Bearer``
header is sent only when the user actually filled one in (some reverse proxies in
front of it want one) and :attr:`configured` asks for the endpoint alone rather
than a credential.

**The endpoint carries the ``/v1``.** LM Studio always serves under ``/v1``, but
people copy the base without it; if the configured host has no ``/v1`` segment we
append one, so ``http://127.0.0.1:1234`` and ``http://127.0.0.1:1234/v1`` both
work and the models list and chat land on the right path.

Everything else — the SSE stream framing, the ``/v1/models`` catalogue, retries,
the mock-transport seam — is inherited unchanged. There is deliberately no copy of
the OpenAI request code here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from ayris.nlu.llm.cloud import CloudOptions, OpenAiCompatibleClient

__all__ = ["LmStudioLlmClient"]


class LmStudioLlmClient(OpenAiCompatibleClient):
    """A local LM Studio server, addressed as an OpenAI-compatible endpoint."""

    name: ClassVar[str] = "lmstudio"
    title: ClassVar[str] = "LM Studio"
    default_base_url: ClassVar[str] = "http://127.0.0.1:1234/v1"
    default_model: ClassVar[str] = ""

    def __init__(self, options: CloudOptions) -> None:
        super().__init__(options)
        # LM Studio's OpenAI surface lives under /v1; forgive a base copied without it.
        if self._base_url and "/v1" not in self._base_url:
            self._base_url = f"{self._base_url}/v1"

    @property
    def configured(self) -> bool:
        # Local and keyless: having somewhere to send to is enough.
        return bool(self._base_url)

    def _auth_headers(self) -> Mapping[str, str]:
        # Only authorize when the user supplied a key; LM Studio needs none.
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
