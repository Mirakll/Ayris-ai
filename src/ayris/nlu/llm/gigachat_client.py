"""GigaChat (Sber) as a cloud LLM provider.

The chat endpoint is OpenAI-shaped, so generation reuses
:class:`OpenAiCompatibleClient` unchanged. The one thing GigaChat does
differently is auth: instead of a static key it wants a short-lived OAuth access
token, obtained by presenting the *authorization key* (a base64 client
credential) to a separate endpoint and refreshed before it expires. That token
never touches the config file or a log — only the authorization key is stored,
in the credential manager, and the access token lives in memory for its half
hour.

The authorization key is what :attr:`CloudOptions.api_key` carries here.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import httpx

from ayris.core.errors import LlmAuthError, LlmError
from ayris.core.paths import executable_dir
from ayris.nlu.llm.cloud import CloudOptions, OpenAiCompatibleClient

__all__ = ["GigaChatLlmClient"]

#: Where the authorization key is exchanged for an access token.
_DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

#: Access scope. ``PERS`` is the individual tier; corporate keys pass a different
#: one through ``CloudOptions.extra["scope"]``.
_DEFAULT_SCOPE = "GIGACHAT_API_PERS"

#: The Russian National CA bundle GigaChat's certificate chain is signed by,
#: shipped under the app's resources and trusted *in addition* to the system
#: roots (see :meth:`CloudLlmClient._resolve_verify`).
_RUSSIAN_CA_RELPATH = ("resources", "certs", "russian_trusted_ca.crt")

#: Refresh this many seconds before the token's stated expiry, so a request never
#: races the clock.
_REFRESH_MARGIN_SEC = 60.0

#: Assumed token lifetime when the response omits ``expires_at``. GigaChat tokens
#: last 30 minutes; this is deliberately shorter.
_ASSUMED_LIFETIME_SEC = 1500.0


class GigaChatLlmClient(OpenAiCompatibleClient):
    """GigaChat over its OpenAI-compatible endpoint, with OAuth token refresh."""

    name: ClassVar[str] = "gigachat"
    title: ClassVar[str] = "GigaChat"
    default_base_url: ClassVar[str] = "https://gigachat.devices.sberbank.ru/api/v1"
    default_model: ClassVar[str] = "GigaChat"

    def __init__(self, options: CloudOptions) -> None:
        super().__init__(options)
        self._oauth_url = str(options.extra.get("oauth_url") or _DEFAULT_OAUTH_URL)
        self._scope = str(options.extra.get("scope") or _DEFAULT_SCOPE)
        self._access_token = ""
        self._token_expiry = 0.0

    def _auth_headers(self) -> Mapping[str, str]:
        return {"Authorization": f"Bearer {self._ensure_token()}"}

    def _extra_ca_path(self) -> Path | None:
        """Trust the bundled Russian National CA GigaChat's chain is signed by.

        GigaChat's OAuth and API endpoints are signed by a CA that ships in no
        default trust store, so verifying against certifi alone fails. The bundle
        rides in the app's resources and is trusted on top of the system roots. If
        it is somehow missing we return ``None`` and the base class falls back to
        the system store — the connection then fails loudly rather than quietly
        skipping verification.
        """
        path = executable_dir().joinpath(*_RUSSIAN_CA_RELPATH)
        return path if path.is_file() else None

    def _ensure_token(self) -> str:
        """Return a live access token, fetching a new one if the old one is stale."""
        if self._access_token and time.time() < self._token_expiry - _REFRESH_MARGIN_SEC:
            return self._access_token

        client = self._require_client()
        headers = {
            "Authorization": f"Basic {self._api_key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        try:
            response = client.post(self._oauth_url, headers=headers, data={"scope": self._scope})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise LlmError(
                f"{self.name}: OAuth request failed: {exc}",
                user_message="GigaChat недоступен. Проверьте подключение к интернету.",
            ) from exc

        status = response.status_code
        if status in (401, 403):
            raise LlmAuthError(f"{self.name}: OAuth rejected the authorization key ({status})")
        if not (200 <= status < 300):
            raise LlmError(
                f"{self.name}: OAuth returned {status}",
                user_message="GigaChat не выдал токен доступа.",
            )

        payload = self._decode_token(response.text)
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise LlmError(
                f"{self.name}: OAuth response has no access_token",
                user_message="GigaChat вернул непонятный ответ на запрос токена.",
            )
        self._access_token = token
        self._token_expiry = self._read_expiry(payload.get("expires_at"))
        return token

    @staticmethod
    def _decode_token(text: str) -> dict[str, Any]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _read_expiry(raw: object) -> float:
        """GigaChat reports ``expires_at`` as a millisecond epoch; convert to seconds."""
        if isinstance(raw, int | float) and raw > 0:
            return float(raw) / 1000.0
        return time.time() + _ASSUMED_LIFETIME_SEC
