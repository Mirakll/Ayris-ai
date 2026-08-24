"""The Windows Credential Manager as a secret provider.

The default, and the only one that needs nothing installed and nothing unlocked:
Credential Manager is part of Windows and is already open because the user is
logged in. :class:`ayris.core.secrets.SecretsStore` does the actual work — this is
the adapter that makes it look like a password manager, so that
:class:`~ayris.actions.system.autofill.AutoFill` does not care which of the three
a template points at.

«Entries» here are the refs the store already knows: ``autofill.<template>.<field>``
for values Ayris put there itself — transliterated, because Credential Manager refs
are latin and templates are named in Russian — plus any other ref this profile has.
Nothing is enumerable in Credential Manager without a ref to ask about, so the list
comes from :meth:`SecretsStore.stored_refs`, which is exactly as complete as the
profile's list of known refs.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING, Final

from ayris.actions.system.secrets.base import (
    SecretEntry,
    SecretProvider,
    SecretProviderError,
    SecretProviderMissing,
    SecretValue,
)
from ayris.core.secrets import get_secrets, is_valid_ref
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.secrets import SecretsStore

__all__ = ["AUTOFILL_PREFIX", "KeyringProvider", "autofill_ref", "normalise_ref"]

_log = get_logger(__name__)

#: Prefix of every ref :class:`AutoFill` stores on its own. Keeping autofill values
#: under one namespace is what makes «forget everything autofill knows» possible
#: without touching the STT keys or the LLM token sitting in the same store.
AUTOFILL_PREFIX: Final = "autofill"

#: Cyrillic to latin, one letter at a time. Needed because
#: :func:`~ayris.core.secrets.is_valid_ref` accepts lowercase latin only, while
#: every template in this project is named in Russian — «карта», «госуслуги».
#: Without this a card number could not be saved at all.
_TRANSLIT: Final = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "c",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}

#: How much of the readable tail survives when a ref has to be shortened.
_TAIL_CHARS: Final = 10

#: Bytes of digest, hex-encoded, appended when the readable form does not fit.
_DIGEST_BYTES: Final = 4


def _slug(text: str) -> str:
    """``text`` as the lowercase latin identifier a ref may contain.

    Anything that is neither latin nor Cyrillic becomes an underscore rather than
    disappearing, so two different names cannot collapse into one ref by accident.
    """
    letters = [
        _TRANSLIT.get(char, char if char.isascii() and char.isalnum() else "_")
        for char in text.casefold()
    ]
    return re.sub(r"_+", "_", "".join(letters)).strip("_")


def autofill_ref(template: str, field: str) -> str:
    """The keyring ref a template's secret field lives at.

    Dots separate the parts because that is the shape :func:`is_valid_ref` accepts,
    and because the store's own refs (``stt.api_key``) read the same way. Russian
    names are transliterated: «карта» becomes ``autofill.karta.number``, which is
    still recognisable in Credential Manager next to the STT keys.

    A name too long for the 32 characters a ref allows — or one written in an
    alphabet this table does not know — keeps its tail and gains a digest of the
    original, so it stays both unique and stable across launches.
    """
    name, part = _slug(template), _slug(field)
    if name and part:
        readable = f"{AUTOFILL_PREFIX}.{name}.{part}"
        if is_valid_ref(readable):
            return readable
    digest = hashlib.blake2s(
        f"{template}\x00{field}".encode(), digest_size=_DIGEST_BYTES
    ).hexdigest()
    tail = (part or name)[-_TAIL_CHARS:].strip("_")
    return f"{AUTOFILL_PREFIX}.{tail}-{digest}" if tail else f"{AUTOFILL_PREFIX}.{digest}"


def normalise_ref(ref: str) -> str:
    """A ref written by hand, in the shape the store accepts.

    ``keyring:autofill.карта.number`` in a template is a reasonable thing to
    write — it is what the settings tab shows — and it points at the same value as
    the generated ``autofill.karta.number``. Transliterating it here means the
    person gets their password instead of «недопустимое имя записи».
    """
    if is_valid_ref(ref):
        return ref
    candidate = ".".join(part for part in (_slug(one) for one in ref.split(".")) if part)
    return candidate if is_valid_ref(candidate) else ref


class KeyringProvider(SecretProvider):
    """Reads and writes secrets in the Windows Credential Manager."""

    name = "keyring"
    title_ru = "Диспетчер учётных данных Windows"

    def __init__(self, *, store: SecretsStore | None = None) -> None:
        self._store = store

    def _secrets(self) -> SecretsStore:
        return self._store if self._store is not None else get_secrets()

    # --------------------------------------------------------------- lifecycle

    def available(self) -> bool:
        return self._secrets().is_available()

    def unlocked(self) -> bool:
        """Always, when available. The user's login is the unlock."""
        return self.available()

    def unlock(self, master_password: str) -> None:
        """Nothing to unlock. Accepted and ignored so callers can be uniform."""
        del master_password
        self.require_available()

    def lock(self) -> None:
        """Nothing to lock: the store holds no session of its own."""

    def missing_error(self) -> SecretProviderMissing:
        return SecretProviderMissing(
            "keyring backend is unavailable",
            user_message=(
                "Диспетчер учётных данных Windows недоступен, " "секреты сохранить не получится."
            ),
        )

    # ------------------------------------------------------------------ reading

    def list_entries(self) -> tuple[SecretEntry, ...]:
        """Refs this profile has a stored value for."""
        self.require_available()
        return tuple(
            SecretEntry(path=ref, title=ref.rsplit(".", 1)[-1])
            for ref in self._secrets().stored_refs()
        )

    def get_field(self, entry: str, field: str = "password") -> SecretValue:
        """The value at ``entry``, or at ``entry.field`` when ``entry`` is a template.

        Both spellings are accepted because both are natural: a template's schema
        names a field, while the settings tab and the store think in whole refs.
        """
        self.require_available()
        store = self._secrets()
        ref = normalise_ref(entry) if "." in entry else autofill_ref(entry, field)
        value = store.get(ref)
        if value is None and "." in entry:
            value = store.get(autofill_ref(entry, field))
        if value is None:
            raise SecretProviderError(
                f"no secret stored at {ref}",
                user_message=f"Секрет «{field}» для «{entry}» ещё не сохранён.",
            )
        return SecretValue(value)

    # ------------------------------------------------------------------ writing

    def save_field(self, template: str, field: str, value: str) -> str:
        """Store ``value`` for a template's field. Returns the ref it went to.

        The only write path in any provider: KeePassXC and Bitwarden are read-only
        here on purpose — an assistant editing someone's vault is a different
        feature with a different risk, and this one only ever needs to remember what
        the user typed into the autofill settings.
        """
        self.require_available()
        ref = autofill_ref(template, field)
        self._secrets().save(ref, value)
        _log.info("autofill secret saved at %s", ref)
        return ref

    def delete_field(self, template: str, field: str) -> bool:
        """Forget a template's secret field. ``False`` if there was nothing there."""
        self.require_available()
        ref = autofill_ref(template, field)
        removed = self._secrets().delete(ref)
        if removed:
            _log.info("autofill secret deleted at %s", ref)
        return removed
