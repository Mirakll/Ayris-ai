"""Where Ayris gets a secret from: the keyring, KeePassXC or Bitwarden.

Three providers behind one ABC (:class:`SecretProvider`), reached through
:func:`get_provider`. The keyring is the default and the only one that needs no
setup — it is the Windows Credential Manager, which is already there and already
unlocked with the user's login. The two vault providers exist because people keep
their passwords in KeePassXC and Bitwarden and will not retype them into a second
place just because an assistant asked.

Providers are built on first use and then kept: an unlocked vault has a session
worth holding on to, and rebuilding the provider per call would throw it away and
ask for the master password again.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

from ayris.actions.system.secrets.base import (
    CliResult,
    CliRunner,
    SecretEntry,
    SecretField,
    SecretProvider,
    SecretProviderError,
    SecretProviderMissing,
    SecretValue,
    SessionCache,
    SubprocessRunner,
    VaultLocked,
    find_cli,
)
from ayris.actions.system.secrets.bitwarden import BitwardenProvider
from ayris.actions.system.secrets.keepass import KeePassProvider
from ayris.actions.system.secrets.keyring_store import KeyringProvider
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "PROVIDERS",
    "BitwardenProvider",
    "CliResult",
    "CliRunner",
    "KeePassProvider",
    "KeyringProvider",
    "SecretEntry",
    "SecretField",
    "SecretProvider",
    "SecretProviderError",
    "SecretProviderMissing",
    "SecretValue",
    "SessionCache",
    "SubprocessRunner",
    "VaultLocked",
    "available_providers",
    "find_cli",
    "get_provider",
    "reset_providers",
    "set_provider",
]

_log = get_logger(__name__)

#: Every provider Ayris knows, by the name that appears in the config.
PROVIDERS: Final[dict[str, Callable[[], SecretProvider]]] = {
    "keyring": KeyringProvider,
    "keepass": KeePassProvider,
    "bitwarden": BitwardenProvider,
}

_instances: dict[str, SecretProvider] = {}
_instances_lock = threading.Lock()


def get_provider(name: str = "") -> SecretProvider:
    """The provider called ``name``, or the configured one when ``name`` is empty.

    Kept between calls, because a vault session must outlive the action that
    unlocked it. An unknown name is an error rather than a silent fall back to the
    keyring: a template pointing at a vault that does not exist should say so.
    """
    key = name.strip().lower()
    if not key:
        from ayris.core.config import get_settings

        key = get_settings().actions.autofill.provider
    factory = PROVIDERS.get(key)
    if factory is None:
        raise SecretProviderError(
            f"unknown secret provider {name!r}",
            user_message=f"Неизвестный менеджер паролей: «{name}».",
        )
    with _instances_lock:
        provider = _instances.get(key)
        if provider is None:
            provider = factory()
            _instances[key] = provider
        return provider


def set_provider(name: str, provider: SecretProvider | None) -> None:
    """Install a provider instance under ``name``. Test seam."""
    key = name.strip().lower()
    with _instances_lock:
        if provider is None:
            _instances.pop(key, None)
        else:
            _instances[key] = provider


def reset_providers() -> None:
    """Lock and forget every provider built so far."""
    with _instances_lock:
        built = list(_instances.values())
        _instances.clear()
    for provider in built:
        try:
            provider.lock()
        except Exception:  # locking is cleanup; a failure here must not propagate
            _log.debug("failed to lock the %s provider on reset", provider.name, exc_info=True)


def available_providers() -> tuple[str, ...]:
    """Names of the providers usable on this machine, in config order.

    Asked by the settings tab to decide what to offer. Each answer costs a
    ``shutil.which`` at most — no vault is opened and no password is needed.
    """
    usable: list[str] = []
    for key in PROVIDERS:
        try:
            if get_provider(key).available():
                usable.append(key)
        except SecretProviderError:
            continue
    return tuple(usable)
