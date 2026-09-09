"""Salted PIN storage and bounded verification."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from collections.abc import Callable
from typing import Protocol, cast

from ayris.utils.logger import get_logger

__all__ = ["PinManager", "PinStore"]

_log = get_logger(__name__)
_SERVICE = "Ayris"
_ACCOUNT = "confirmation-pin"
_ROUNDS = 600_000


class PinStore(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...

    def delete_password(self, service_name: str, username: str) -> None: ...


class PinManager:
    """Keep only PBKDF2 material in keyring and enforce an attempt budget."""

    def __init__(
        self,
        store: PinStore | None = None,
        *,
        attempts: int = 3,
        delay_sec: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._attempts = max(1, attempts)
        self._delay = max(0.0, delay_sec)
        self._sleep = sleeper

    def _backend(self) -> PinStore:
        if self._store is None:
            import keyring

            self._store = cast(PinStore, keyring)
        return self._store

    def configured(self) -> bool:
        try:
            return bool(self._backend().get_password(_SERVICE, _ACCOUNT))
        except Exception as exc:
            _log.warning("PIN store is unavailable: %s", type(exc).__name__)
            return False

    def set_pin(self, pin: str) -> None:
        """Set or change the PIN. Clear text exists only for this call."""
        if not pin.isdigit() or not 4 <= len(pin) <= 12:
            raise ValueError("ПИН должен содержать от 4 до 12 цифр.")
        salt = secrets.token_bytes(16)
        digest = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, _ROUNDS)
        payload = json.dumps(
            {
                "v": 1,
                "alg": "pbkdf2-sha256",
                "rounds": _ROUNDS,
                "salt": salt.hex(),
                "hash": digest.hex(),
            },
            separators=(",", ":"),
        )
        self._backend().set_password(_SERVICE, _ACCOUNT, payload)

    def clear(self) -> None:
        try:
            self._backend().delete_password(_SERVICE, _ACCOUNT)
        except Exception as exc:
            if type(exc).__name__ != "PasswordDeleteError":
                raise

    def verify(self, pin: str) -> bool:
        payload = self._read()
        if payload is None:
            return False
        try:
            salt = bytes.fromhex(str(payload["salt"]))
            expected = bytes.fromhex(str(payload["hash"]))
            rounds = int(str(payload["rounds"]))
        except (KeyError, TypeError, ValueError):
            return False
        actual = hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, rounds)
        return hmac.compare_digest(actual, expected)

    def prompt(
        self,
        ask: Callable[[int], str | None],
        *,
        attempts: int | None = None,
        delay_sec: float | None = None,
    ) -> bool:
        if not self.configured():
            return False
        limit = max(1, attempts if attempts is not None else self._attempts)
        delay = max(0.0, delay_sec if delay_sec is not None else self._delay)
        for number in range(1, limit + 1):
            candidate = ask(number)
            if candidate is None:
                return False
            if self.verify(candidate):
                return True
            if number < limit and delay:
                self._sleep(delay)
        return False

    def _read(self) -> dict[str, object] | None:
        try:
            raw = self._backend().get_password(_SERVICE, _ACCOUNT)
            value = json.loads(raw) if raw else None
        except Exception:
            return None
        return value if isinstance(value, dict) else None
