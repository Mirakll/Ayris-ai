"""Bitwarden through ``bw``.

Bitwarden's model is the opposite of KeePassXC's. There is no local file to open:
``bw`` logs in to a server, then holds a *session key* which every later command
needs. That key is normally handed around in the ``BW_SESSION`` environment
variable, and the vault is «unlocked» exactly as long as something remembers it.

Two consequences shape this module:

* The session key is a secret of the same weight as the master password, and it
  lives in :class:`SessionCache` — in memory, with a deadline. It is passed to
  ``bw`` through the child's environment rather than ``argv``, because arguments
  are readable by every process on the machine while a child's environment is not.
* ``bw`` is a Node program and slow to start — a third of a second before it does
  anything. So the vault listing is fetched once, in full, and cached; asking for
  ten fields of one entry costs one call, not ten.

``bw`` speaks JSON when asked (``--raw``, ``list items``), which is the only sane
way to parse it: entry names contain spaces, quotes and newlines.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Final

from ayris.actions.system.secrets.base import (
    CliResult,
    CliRunner,
    SecretEntry,
    SecretProvider,
    SecretProviderError,
    SecretValue,
    SessionCache,
    SubprocessRunner,
    VaultLocked,
    find_cli,
)
from ayris.core.config import get_settings
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.core.config import AutofillActionsConfig

__all__ = ["BitwardenProvider"]

_log = get_logger(__name__)

#: Executable names to look for. ``bw`` is the official CLI; ``bitwarden-cli`` is
#: what a couple of package managers install it as.
_CLI_NAMES: Final = ("bw", "bitwarden-cli")

#: Where each field Ayris knows lives inside a ``bw`` item object. Bitwarden nests
#: everything interesting under ``login``, and anything else is a custom field.
_ITEM_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    "password": ("login", "password"),
    "username": ("login", "username"),
    "login": ("login", "username"),
    "totp": ("login", "totp"),
    "notes": ("notes",),
    "title": ("name",),
    "url": ("login", "uris", "0", "uri"),
}

#: Item types ``bw`` uses. Only logins and cards carry anything a template wants.
_TYPE_LOGIN: Final = 1
_TYPE_CARD: Final = 3

#: Card fields, which live under ``card`` and are the other half of task 27:
#: a card number is a secret exactly like a password.
_CARD_FIELDS: Final[dict[str, str]] = {
    "card": "number",
    "number": "number",
    "cardholder": "cardholderName",
    "code": "code",
    "cvv": "code",
    "brand": "brand",
    "expiry_month": "expMonth",
    "expiry_year": "expYear",
}

#: Fragments of ``bw`` output that mean «unlock first» rather than «broken».
_LOCKED_MARKERS: Final = (
    "vault is locked",
    "you are not logged in",
    "invalid master password",
    "mac failed",
)


class BitwardenProvider(SecretProvider):
    """Reads a Bitwarden vault by shelling out to ``bw``."""

    name = "bitwarden"
    title_ru = "Bitwarden"

    def __init__(
        self,
        *,
        runner: CliRunner | None = None,
        settings: AutofillActionsConfig | None = None,
    ) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()
        self._settings = settings
        self._session = SessionCache(self._config().session_ttl_s)
        self._items: tuple[dict[str, Any], ...] | None = None

    # ------------------------------------------------------------------ config

    def _config(self) -> AutofillActionsConfig:
        return self._settings if self._settings is not None else get_settings().actions.autofill

    def executable(self) -> str:
        """Path to ``bw``, or ``""``."""
        return find_cli(self._config().bitwarden_cli, *_CLI_NAMES)

    # --------------------------------------------------------------- lifecycle

    def available(self) -> bool:
        return bool(self.executable())

    def unlocked(self) -> bool:
        return bool(self._session.get())

    def unlock(self, master_password: str) -> None:
        """Turn a master password into a session key and keep the key, not the password.

        ``bw unlock --raw`` prints nothing but the key, which is what makes this
        safe to do: the master password is used once, on stdin, and never stored.
        """
        self.require_available()
        if not master_password:
            raise self.locked_error()
        result = self._invoke(
            ("unlock", "--raw", "--passwordenv", "BW_PASSWORD"), password=master_password
        )
        if not result.ok:
            raise self._error_for(result)
        key = result.stdout.strip()
        if not key:
            raise SecretProviderError(
                "bw unlock returned an empty session key",
                user_message="Bitwarden не выдал ключ сессии.",
            )
        self._session.store(key)
        self._items = None
        _log.info("bitwarden vault unlocked")

    def lock(self) -> None:
        """Forget the session key here, and tell ``bw`` to forget it too."""
        key = self._session.get()
        self._session.drop()
        self._items = None
        if key and self.executable():
            result = self._invoke(("lock",), session=key)
            if not result.ok:
                _log.debug("bw lock reported: %s", result.short_error())

    # ------------------------------------------------------------------ reading

    def list_entries(self) -> tuple[SecretEntry, ...]:
        """Every login and card in the vault, without values."""
        return tuple(_entry_of(item) for item in self._load_items())

    def get_field(self, entry: str, field: str = "password") -> SecretValue:
        """One field of one item, found by name.

        Matching is by name, case-insensitively, because that is what a person says
        — «возьми пароль от Госуслуг» — and Bitwarden's own ids are UUIDs nobody
        will ever dictate. An exact match wins over a partial one.
        """
        item = _find_item(self._load_items(), entry)
        if item is None:
            raise SecretProviderError(
                f"bitwarden has no item named {entry!r}",
                user_message=f"В Bitwarden нет записи «{entry}».",
            )
        value = _field_of(item, field)
        if value is None:
            raise SecretProviderError(
                f"bitwarden item {entry!r} has no field {field!r}",
                user_message=f"У записи «{entry}» нет поля «{field}».",
            )
        return SecretValue(value)

    # ------------------------------------------------------------------ plumbing

    def _load_items(self) -> tuple[dict[str, Any], ...]:
        """The whole vault as ``bw`` returns it, fetched once per unlock.

        Values included — unavoidable, ``bw list items`` has no «names only» mode —
        which is why the result stays in this object and is dropped by
        :meth:`lock`, and why nothing here ever logs an item.
        """
        if self._items is not None:
            return self._items
        self.require_available()
        key = self._session.get()
        if not key:
            raise self.locked_error()
        result = self._invoke(("list", "items"), session=key)
        if not result.ok:
            raise self._error_for(result)
        try:
            parsed = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise SecretProviderError(
                f"bw list items returned unparseable output: {exc}",
                user_message="Bitwarden вернул непонятный ответ.",
            ) from exc
        if not isinstance(parsed, list):
            raise SecretProviderError(
                "bw list items did not return a list",
                user_message="Bitwarden вернул непонятный ответ.",
            )
        self._items = tuple(item for item in parsed if isinstance(item, dict))
        _log.debug("bitwarden vault listed: %d items", len(self._items))
        return self._items

    def _invoke(
        self,
        args: Sequence[str],
        *,
        session: str = "",
        password: str = "",
    ) -> CliResult:
        """Run ``bw`` with the secrets in the environment, never in ``argv``.

        ``--nointeraction`` matters: without it a locked vault makes ``bw`` sit and
        wait for a password on a terminal that is not there, and the call dies of
        the timeout instead of returning something explainable.
        """
        executable = self.executable()
        if not executable:
            raise self.missing_error()
        env = _child_env(session=session, password=password)
        return self._runner.run((executable, "--nointeraction", *args), env=env)

    def _error_for(self, result: CliResult) -> SecretProviderError:
        message = result.short_error()
        lowered = message.lower()
        _log.warning("bw failed: %s", message)
        if any(marker in lowered for marker in _LOCKED_MARKERS):
            self._session.drop()
            self._items = None
            return VaultLocked(
                f"bitwarden vault is locked: {message}",
                user_message="Хранилище Bitwarden закрыто, введите мастер-пароль в настройках.",
            )
        return SecretProviderError(
            f"bw failed: {message}",
            user_message="Bitwarden не смог выполнить запрос.",
        )


def _child_env(*, session: str, password: str) -> dict[str, str]:
    """Environment for one ``bw`` call: the parent's, plus what this call needs.

    A copy of :data:`os.environ` rather than a bare pair, because ``bw`` is a Node
    program and needs ``PATH``, ``APPDATA`` and ``HOME`` to find its own data
    directory. The secrets are added to the copy and go no further: a child's
    environment is not readable by other users, unlike its command line.
    """
    env = dict(os.environ)
    if session:
        env["BW_SESSION"] = session
    if password:
        env["BW_PASSWORD"] = password
    else:
        env.pop("BW_PASSWORD", None)
    return env


def _entry_of(item: dict[str, Any]) -> SecretEntry:
    """A ``bw`` item as a :class:`SecretEntry`, with names only."""
    login = item.get("login") if isinstance(item.get("login"), dict) else {}
    uris = login.get("uris") if isinstance(login, dict) else None
    url = ""
    if isinstance(uris, list) and uris and isinstance(uris[0], dict):
        url = str(uris[0].get("uri") or "")
    fields: list[str] = []
    item_type = item.get("type")
    if item_type == _TYPE_LOGIN:
        fields = ["username", "password"]
        if isinstance(login, dict) and login.get("totp"):
            fields.append("totp")
    elif item_type == _TYPE_CARD:
        fields = ["number", "cardholder", "code", "expiry_month", "expiry_year"]
    return SecretEntry(
        path=str(item.get("id") or item.get("name") or ""),
        title=str(item.get("name") or ""),
        username=str(login.get("username") or "") if isinstance(login, dict) else "",
        url=url,
        fields=tuple(fields),
    )


def _find_item(items: Sequence[dict[str, Any]], wanted: str) -> dict[str, Any] | None:
    """The item called ``wanted``: exact name, then id, then a partial name."""
    needle = wanted.strip().casefold()
    if not needle:
        return None
    for item in items:
        if str(item.get("name") or "").casefold() == needle:
            return item
    for item in items:
        if str(item.get("id") or "") == wanted:
            return item
    for item in items:
        if needle in str(item.get("name") or "").casefold():
            return item
    return None


def _field_of(item: dict[str, Any], field: str) -> str | None:
    """One field of an item, or ``None``.

    Looks in the login block, then the card block, then the item's custom fields —
    which is where a bank puts everything a form needs and Bitwarden has no name
    for.
    """
    key = field.strip().casefold()
    path = _ITEM_FIELDS.get(key)
    if path is not None:
        found = _dig(item, path)
        if found:
            return found
    card = item.get("card")
    if isinstance(card, dict):
        card_key = _CARD_FIELDS.get(key)
        if card_key and card.get(card_key):
            return str(card[card_key])
    custom = item.get("fields")
    if isinstance(custom, list):
        for entry in custom:
            if isinstance(entry, dict) and str(entry.get("name") or "").casefold() == key:
                value = entry.get("value")
                if value:
                    return str(value)
    return None


def _dig(item: dict[str, Any], path: Sequence[str]) -> str:
    """Follow ``path`` through nested dicts and lists; ``""`` if it does not lead anywhere."""
    current: Any = item
    for step in path:
        if isinstance(current, dict):
            current = current.get(step)
        elif isinstance(current, list) and step.isdigit():
            index = int(step)
            current = current[index] if index < len(current) else None
        else:
            return ""
        if current is None:
            return ""
    return str(current)
