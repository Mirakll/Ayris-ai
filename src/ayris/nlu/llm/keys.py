"""Reading a provider's API key out of the Windows Credential Manager.

Section 17 of the specification is firm: the key itself never sits in
``config.toml`` and never reaches a log — the configuration keeps only the *name*
of a credential entry, and the value is fetched from
:class:`~ayris.core.secrets.SecretsStore` at the moment a client is built. This
module is the one place in the LLM layer that turns a reference name into a key,
so the six provider clients receive a plain string and know nothing about where it
came from.

Resolution mirrors :meth:`ayris.audio.stt.cloud_base.CloudSttEngine._load_credential`
so the two subsystems behave the same: a key handed over directly wins, then the
entry named by ``ai.credential_ref``, then the slot named after the provider — a
user who filled in the «OpenAI» field in settings gets a working client without
also having to set ``credential_ref``. A missing key is not an error here: an
empty string comes back, the client reports itself unconfigured, and the worker
answers with the «модель не настроена» sentence rather than spending a doomed
request. :func:`mask` is re-exported so callers log the tail, never the key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayris.core.errors import AyrisError
from ayris.core.secrets import get_secrets, is_valid_ref, mask
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.secrets import SecretsStore

__all__ = ["mask", "resolve_api_key"]

_log = get_logger(__name__)


def resolve_api_key(
    provider: str,
    *,
    credential_ref: str = "",
    explicit: str = "",
    store: SecretsStore | None = None,
) -> str:
    """Return the key for ``provider``, or an empty string when none is stored.

    Three places are tried in order: a key handed over directly (``explicit``,
    used when the worker already resolved the slot), the entry named by
    ``credential_ref``, and the slot named after the provider itself. A locked or
    unreadable store is logged and skipped, never raised — the caller reads the
    empty result as «no key» and shows the settings hint, which is the only thing
    the user can act on.

    The value is never logged; only its reference name and masked tail are.
    """
    if explicit.strip():
        return explicit.strip()

    wanted: list[str] = []
    if credential_ref and is_valid_ref(credential_ref):
        wanted.append(credential_ref)
    if provider and provider not in wanted and is_valid_ref(provider):
        wanted.append(provider)

    secrets = store or get_secrets()
    for ref in wanted:
        try:
            value = secrets.get(ref)
        except AyrisError as exc:
            _log.warning("%s: ключ «%s» недоступен: %s", provider, ref, exc.technical)
            continue
        if value:
            _log.debug("%s: ключ найден в записи «%s» (%s)", provider, ref, mask(value))
            return value
    return ""
