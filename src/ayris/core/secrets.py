"""API keys and other secrets, stored outside ``config.toml``.

Section 17 of the specification is explicit: API keys live in the Windows
Credential Manager, ``config.toml`` holds only the *name* of the entry, and a key
never reaches a log file. This module is the only place in Ayris that talks to
:mod:`keyring`.

The split looks like this::

    config.toml           ai.credential_ref = "openai"     <- name only
    Credential Manager    Ayris / openai = "sk-..."        <- the actual key

Everything that leaves this module is either a reference name, a boolean, or a
masked value from :func:`mask`. :meth:`SecretsStore.__repr__` lists the service
name only, and :class:`~ayris.core.errors.SecretsError` messages are built from
the reference name and the backend error type, never from the value.

``keyring`` is imported lazily. Importing it spins up the Windows backend, which
is slow and pointless for a process that never talks to a cloud provider — and on
a CI machine without a credential store the import itself can fail. Tests inject
a fake through the ``backend`` argument instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast

from ayris.core.errors import SecretsError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

__all__ = [
    "KNOWN_SLOTS",
    "SERVICE_NAME",
    "KeyringBackend",
    "SecretSlot",
    "SecretStatus",
    "SecretsStore",
    "get_secrets",
    "is_valid_ref",
    "mask",
    "reset_secrets",
    "resolve_secrets",
]

_log = get_logger("core.secrets")

#: Service name under which every Ayris credential is filed in the store.
SERVICE_NAME: Final = "Ayris"

#: A reference is a short lowercase identifier, never a key. Keeping it strict
#: means a key pasted into the wrong settings field cannot become a "name" and
#: end up written to ``config.toml`` in clear text.
_REF_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_.-]{0,31}$")

#: Prefixes used by the providers Ayris supports. A "reference" starting with one
#: of these is a key that landed in the wrong field.
_KEY_PREFIXES: Final = ("sk-", "sk_", "pk-", "hf_", "ya29.", "aiza", "gsk_", "xai-")

_MASK_TAIL: Final = 4
_MASK_BODY: Final = "••••••••"
_EMPTY_MASK: Final = "—"


@dataclass(frozen=True, slots=True)
class SecretSlot:
    """One well-known credential: its reference name and how to describe it."""

    ref: str
    title: str
    hint: str


#: Slots the settings UI offers out of the box. A user may still type any other
#: reference — the store itself is not restricted to this table.
KNOWN_SLOTS: Final[MappingProxyType[str, SecretSlot]] = MappingProxyType(
    {
        slot.ref: slot
        for slot in (
            SecretSlot("openai", "OpenAI", "Ключ вида sk-… с platform.openai.com"),
            SecretSlot("anthropic", "Anthropic", "Ключ с console.anthropic.com"),
            SecretSlot("openrouter", "OpenRouter", "Ключ с openrouter.ai"),
            SecretSlot("deepseek", "DeepSeek", "Ключ с platform.deepseek.com"),
            SecretSlot("yandex", "Яндекс SpeechKit", "API-ключ сервисного аккаунта"),
            SecretSlot("gigachat", "GigaChat", "Авторизационные данные Sber"),
            SecretSlot("google", "Google Cloud", "Ключ Speech-to-Text / Text-to-Speech"),
            SecretSlot("azure", "Azure Speech", "Ключ ресурса Speech"),
            SecretSlot("elevenlabs", "ElevenLabs", "Ключ с elevenlabs.io"),
            SecretSlot("porcupine", "Picovoice Porcupine", "AccessKey с console.picovoice.ai"),
        )
    }
)


@dataclass(frozen=True, slots=True)
class SecretStatus:
    """Whether a reference currently holds a value. Safe to log and to display."""

    ref: str
    stored: bool
    title: str = ""

    @property
    def label(self) -> str:
        """Russian one-liner for the settings tab."""
        name = self.title or self.ref
        return f"{name}: ключ сохранён" if self.stored else f"{name}: ключ не задан"


def is_valid_ref(ref: str) -> bool:
    """Whether ``ref`` is a plausible entry name rather than a pasted key."""
    if not _REF_PATTERN.match(ref):
        return False
    return not ref.lower().startswith(_KEY_PREFIXES)


def mask(value: str | None) -> str:
    """Render a secret for the UI and for logs.

    Shows at most the last four characters, and nothing at all for values too
    short for even that to be safe.
    """
    if not value:
        return _EMPTY_MASK
    if len(value) <= _MASK_TAIL * 2:
        return _MASK_BODY
    return _MASK_BODY + value[-_MASK_TAIL:]


class KeyringBackend(Protocol):
    """The three :mod:`keyring` functions Ayris uses. Lets tests inject a fake."""

    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


def _require_ref(ref: str) -> None:
    """Reject anything that is not an entry name.

    Raises:
        SecretsError: ``ref`` is malformed or looks like a key.
    """
    if is_valid_ref(ref):
        return
    raise SecretsError(
        f"invalid credential reference: {len(ref)} chars",
        user_message=(
            "Недопустимое имя записи для ключа. Используйте короткое латинское имя, "
            "например openai. Сам ключ вводится в отдельном поле."
        ),
    )


def _is_missing_entry(exc: BaseException) -> bool:
    """Whether a backend exception means "no such entry" rather than a failure.

    ``keyring.errors.PasswordDeleteError`` covers both a missing entry and a
    locked store, and the Windows backend does not separate them by type.
    Matching on the class name keeps the :mod:`keyring` import lazy.
    """
    if type(exc).__name__ == "PasswordDeleteError":
        return True
    return isinstance(exc, KeyError | FileNotFoundError)


class SecretsStore:
    """Reads and writes credentials in the Windows Credential Manager.

    Args:
        service: Service name in the credential store. Overridden in tests and by
            profiles that want their own namespace.
        backend: Injected :mod:`keyring`-compatible module. ``None`` imports the
            real :mod:`keyring` on first use.
    """

    def __init__(
        self,
        service: str = SERVICE_NAME,
        *,
        backend: KeyringBackend | None = None,
    ) -> None:
        self._service = service
        self._backend = backend
        self._backend_failed = False

    @property
    def service(self) -> str:
        return self._service

    def _keyring(self) -> KeyringBackend:
        """Resolve the backend, importing :mod:`keyring` on first use."""
        if self._backend is not None:
            return self._backend
        try:
            import keyring
        except ImportError as exc:  # pragma: no cover - keyring is a hard dependency
            self._backend_failed = True
            raise SecretsError(
                f"keyring is not installed: {exc}",
                user_message=(
                    "Хранилище ключей Windows недоступно. Облачные сервисы работать не будут."
                ),
            ) from exc
        self._backend = cast(KeyringBackend, keyring)
        return self._backend

    def is_available(self) -> bool:
        """Whether the credential store answers at all.

        The settings tab uses this to disable the key fields up front instead of
        failing on the first save.
        """
        if self._backend_failed:
            return False
        try:
            self._keyring().get_password(self._service, "__probe__")
        except SecretsError:
            return False
        except Exception as exc:
            self._backend_failed = True
            _log.warning("Credential store is unavailable: %s: %s", type(exc).__name__, exc)
            return False
        return True

    def get(self, ref: str) -> str | None:
        """Return the secret stored under ``ref``, or ``None`` if there is none.

        Raises:
            SecretsError: The reference is invalid or the store refused the read.
        """
        _require_ref(ref)
        try:
            value = self._keyring().get_password(self._service, ref)
        except SecretsError:
            raise
        except Exception as exc:
            raise SecretsError(
                f"cannot read credential {ref!r}: {type(exc).__name__}: {exc}",
                user_message=f"Не удалось прочитать ключ «{ref}» из хранилища Windows.",
            ) from exc
        return value or None

    def save(self, ref: str, value: str) -> None:
        """Store ``value`` under ``ref``.

        The value is never logged — only the reference name and the masked tail.

        Raises:
            SecretsError: The reference is invalid, the value is empty, or the
                store refused the write.
        """
        _require_ref(ref)
        if not value.strip():
            raise SecretsError(
                f"refusing to store an empty credential for {ref!r}",
                user_message="Ключ не может быть пустым. Чтобы стереть ключ, нажмите «Удалить».",
            )
        try:
            self._keyring().set_password(self._service, ref, value)
        except SecretsError:
            raise
        except Exception as exc:
            raise SecretsError(
                f"cannot store credential {ref!r}: {type(exc).__name__}: {exc}",
                user_message=f"Не удалось сохранить ключ «{ref}» в хранилище Windows.",
            ) from exc
        _log.info("Credential %r stored (%s)", ref, mask(value))

    def delete(self, ref: str) -> bool:
        """Remove ``ref`` from the store.

        Returns:
            ``True`` if an entry was removed, ``False`` if there was nothing to
            remove.

        Raises:
            SecretsError: The reference is invalid, or the store refused the
                deletion for a reason other than "no such entry".
        """
        _require_ref(ref)
        try:
            self._keyring().delete_password(self._service, ref)
        except SecretsError:
            raise
        except Exception as exc:
            if _is_missing_entry(exc):
                return False
            raise SecretsError(
                f"cannot delete credential {ref!r}: {type(exc).__name__}: {exc}",
                user_message=f"Не удалось удалить ключ «{ref}» из хранилища Windows.",
            ) from exc
        _log.info("Credential %r deleted", ref)
        return True

    def has(self, ref: str) -> bool:
        """Whether ``ref`` currently holds a value. Never raises."""
        try:
            return self.get(ref) is not None
        except SecretsError:
            return False

    def status(self, ref: str) -> SecretStatus:
        """Presence of one credential, in a form safe to show and to log."""
        slot = KNOWN_SLOTS.get(ref)
        return SecretStatus(ref=ref, stored=self.has(ref), title=slot.title if slot else "")

    def statuses(self, refs: Iterable[str] | None = None) -> tuple[SecretStatus, ...]:
        """Presence of several credentials, defaulting to every :data:`KNOWN_SLOTS` entry."""
        names = tuple(refs) if refs is not None else tuple(KNOWN_SLOTS)
        return tuple(self.status(ref) for ref in names)

    def stored_refs(self, refs: Iterable[str] | None = None) -> tuple[str, ...]:
        """References that currently hold a value."""
        return tuple(status.ref for status in self.statuses(refs) if status.stored)

    def prune(self, keep: Iterable[str]) -> tuple[str, ...]:
        """Delete known credentials whose reference is no longer in use.

        Called after the settings are saved, so a provider the user switched away
        from does not leave its key behind.

        Returns:
            The references that were removed.
        """
        retained = set(keep)
        removed: list[str] = []
        for ref in KNOWN_SLOTS:
            if ref in retained:
                continue
            try:
                if self.delete(ref):
                    removed.append(ref)
            except SecretsError as exc:
                _log.warning("Could not prune credential %r: %s", ref, exc.technical)
        return tuple(removed)

    def __repr__(self) -> str:
        """Service name only — a secret must never reach a traceback."""
        return f"SecretsStore(service={self._service!r})"


_store: SecretsStore | None = None


def get_secrets() -> SecretsStore:
    """Process-wide credential store."""
    global _store
    if _store is None:
        _store = SecretsStore()
    return _store


def reset_secrets(store: SecretsStore | None = None) -> None:
    """Replace or drop the process-wide store. Test and profile-switch helper."""
    global _store
    _store = store


def resolve_secrets(refs: Mapping[str, str | None]) -> Iterator[tuple[str, str]]:
    """Yield ``(slot, key)`` for every populated reference in ``refs``.

    Args:
        refs: Logical slot name (``"ai"``, ``"stt"``) mapped to the credential
            reference stored in the configuration.

    Workers use this to receive only the keys they actually need, rather than the
    whole store.
    """
    store = get_secrets()
    for slot, ref in refs.items():
        if not ref:
            continue
        value = store.get(ref)
        if value:
            yield slot, value
