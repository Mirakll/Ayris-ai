"""What a password manager has to offer, and how Ayris asks for it.

One ABC, :class:`SecretProvider`, with two questions — «what is in the vault» and
«give me this one field» — and three implementations: the Windows Credential
Manager (:mod:`ayris.core.secrets`, wrapped in :mod:`.keyring_store`), KeePassXC
and Bitwarden. The last two have no library worth using on Windows, so they are
driven through their command-line tools, which is what the rest of this module is
about.

**The master password never appears in a command line.** ``keepassxc-cli`` and
``bw`` both read it from standard input for exactly this reason: an argument
vector is visible to every process on the machine — Task Manager shows it, and so
does any script that walks the process list. :class:`CliRunner` therefore takes
``stdin`` separately and has no way to interpolate a secret into ``argv``.

**Unlocking is remembered in memory and nowhere else.** A vault password typed
once should not be typed again for the next field, and must not survive the
process. :class:`SessionCache` holds it for ``session_ttl_s`` seconds, in this
object, never on disk; :meth:`SecretProvider.lock` throws it away early.

**Discovery is lazy.** ``keepassxc-cli`` and ``bw`` are not Python packages and
are not in ``requirements-ci.txt``; most machines have neither. Nothing here looks
for them until a provider is actually used, and a missing one produces a Russian
sentence naming what to install rather than ``FileNotFoundError``.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.core.errors import ActionError
from ayris.utils.logger import forget_secret, get_logger, guard_secret

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CliResult",
    "CliRunner",
    "SecretEntry",
    "SecretField",
    "SecretProvider",
    "SecretProviderError",
    "SecretProviderMissing",
    "SecretValue",
    "SessionCache",
    "SubprocessRunner",
    "VaultLocked",
    "find_cli",
]

_log = get_logger(__name__)

#: Hard cap on any CLI call. ``bw`` talks to a server and ``keepassxc-cli`` derives
#: a key with a deliberately slow KDF, so this is generous — but not unbounded: a
#: hung vault must not hang the assistant.
_CLI_TIMEOUT_S: Final = 45.0

#: Never let a CLI's own output become an error message verbatim: some tools echo
#: the entry name, and an entry name can be «Visa 4111…». Only this much is kept,
#: and only for the log.
_MAX_STDERR_CHARS: Final = 200


class SecretProviderError(ActionError):
    """The provider is there but could not answer."""


class SecretProviderMissing(SecretProviderError):
    """The command-line tool is not installed, or not where the config says."""


class VaultLocked(SecretProviderError):
    """The vault needs a master password (or a session) that Ayris does not have."""


class SecretField(str):
    """Name of a field inside an entry: ``password``, ``username``, ``totp``…

    A distinct type only so signatures read clearly; providers map these onto
    whatever their own tool calls them.
    """

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class SecretEntry:
    """One vault entry as the picker and the templates see it.

    Deliberately without values. Listing a vault is a browsing operation and must
    not pull a hundred passwords into the process to show a hundred names.
    """

    path: str
    title: str = ""
    username: str = ""
    url: str = ""
    fields: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """What to show in a list: the title, or the path when there is none."""
        base = self.title or self.path
        return f"{base} ({self.username})" if self.username else base


class SecretValue:
    """A secret in transit, redacted from logs for as long as it exists.

    ``with provider.value(entry, "password") as secret:`` guards the value on the
    way in and forgets it on the way out, so the window in which a stray log line
    could carry it is the ``with`` block and not the rest of the session. The value
    is deliberately not exposed through ``__str__`` or ``__repr__`` — those are
    what f-strings and ``%s`` reach for, and they return the mask instead.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value
        guard_secret(value)

    @property
    def value(self) -> str:
        """The secret itself. Every use of this is a place worth reviewing."""
        return self._value

    def __len__(self) -> int:
        return len(self._value)

    def __bool__(self) -> bool:
        return bool(self._value)

    def __str__(self) -> str:
        return "[скрыто]"

    def __repr__(self) -> str:
        return f"SecretValue(len={len(self._value)})"

    def forget(self) -> None:
        """Stop redacting the value and drop the reference to it."""
        forget_secret(self._value)
        self._value = ""

    def __enter__(self) -> SecretValue:
        return self

    def __exit__(self, *_: object) -> None:
        self.forget()


@dataclass(frozen=True, slots=True)
class CliResult:
    """What a command-line tool said."""

    code: int = 0
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    def short_error(self) -> str:
        """First line of stderr, clipped. For the log, never for the user."""
        text = (self.stderr or self.stdout).strip().splitlines()
        first = text[0] if text else ""
        return first[:_MAX_STDERR_CHARS]


class CliRunner(ABC):
    """Runs one command-line tool. The seam the provider tests replace.

    ``stdin`` is a separate argument on purpose: it is where a master password
    goes, and there is no code path that could put it in ``argv`` by accident.
    """

    @abstractmethod
    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float = _CLI_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> CliResult:
        """Run ``argv``, feed ``stdin``, return what came back."""


class SubprocessRunner(CliRunner):
    """The real runner: :func:`subprocess.run` with no shell and no console."""

    def run(
        self,
        argv: Sequence[str],
        *,
        stdin: str | None = None,
        timeout: float = _CLI_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> CliResult:
        if not argv:
            raise SecretProviderError("empty command line")
        try:
            completed = subprocess.run(  # argv is built here, never a shell string
                list(argv),
                input=stdin,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=env,
                # No flashing console window when the assistant asks a vault
                # something while the user is typing somewhere else.
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except FileNotFoundError as exc:
            raise SecretProviderMissing(
                f"{argv[0]} not found",
                user_message=f"Программа {Path(argv[0]).name} не найдена.",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise SecretProviderError(
                f"{argv[0]} timed out after {timeout:.0f}s",
                user_message="Менеджер паролей не ответил вовремя.",
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise SecretProviderError(
                f"{argv[0]} failed to start: {exc}",
                user_message="Не удалось запустить менеджер паролей.",
            ) from exc
        return CliResult(
            code=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def find_cli(configured: str, *names: str) -> str:
    """Where a vault's command-line tool is, or ``""``.

    A configured path wins — a portable KeePassXC unpacked into a folder is the
    normal case, and it is never in ``PATH``. Otherwise every name is tried,
    because the tools are not consistent: Bitwarden's is ``bw``, KeePassXC ships
    ``keepassxc-cli`` and older builds ``keepassxc-cli.exe`` only under the
    installation directory.
    """
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(configured)
        if found:
            return found
        _log.warning("configured password-manager CLI points at nothing: %s", candidate)
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    return ""


class SessionCache:
    """An unlock token — a master password or a session key — kept for a while.

    In memory, with a deadline, and cleared by :meth:`drop`. Nothing here is
    written anywhere: the whole point is that the user types a master password
    once per session instead of once per field, and that closing Ayris ends it.
    """

    __slots__ = ("_deadline", "_lock", "_token", "_ttl_s")

    def __init__(self, ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._token = ""
        self._deadline = 0.0
        self._lock = threading.Lock()

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def store(self, token: str) -> None:
        """Remember ``token`` for :attr:`ttl_s` seconds."""
        if not token:
            return
        with self._lock:
            if self._token and self._token != token:
                forget_secret(self._token)
            self._token = token
            self._deadline = time.monotonic() + self._ttl_s
        guard_secret(token)

    def get(self) -> str:
        """The token if it is still valid, ``""`` once it has expired."""
        with self._lock:
            if not self._token:
                return ""
            if time.monotonic() >= self._deadline:
                expired, self._token, self._deadline = self._token, "", 0.0
                forget_secret(expired)
                _log.debug("vault session expired")
                return ""
            return self._token

    def drop(self) -> None:
        """Forget the token now."""
        with self._lock:
            token, self._token, self._deadline = self._token, "", 0.0
        if token:
            forget_secret(token)

    def __bool__(self) -> bool:
        return bool(self.get())


class SecretProvider(ABC):
    """A source of secrets: a vault, a keyring, anything with named fields.

    Two questions and a lifecycle. ``list_entries`` is for pickers and must not
    fetch values; ``get_field`` returns exactly one field of one entry, wrapped in
    a :class:`SecretValue` that is redacted from the log while it lives.
    """

    #: Stable identifier used in config and in error messages.
    name: str = ""
    #: Russian name for the settings tab and for what Ayris says out loud.
    title_ru: str = ""

    @abstractmethod
    def available(self) -> bool:
        """Whether this provider can be used at all on this machine."""

    @abstractmethod
    def unlocked(self) -> bool:
        """Whether a secret can be fetched right now without a master password."""

    @abstractmethod
    def unlock(self, master_password: str) -> None:
        """Open the vault. Raises :class:`SecretProviderError` if it refuses."""

    @abstractmethod
    def lock(self) -> None:
        """Forget the session. Always safe to call."""

    @abstractmethod
    def list_entries(self) -> tuple[SecretEntry, ...]:
        """Everything in the vault, without values."""

    @abstractmethod
    def get_field(self, entry: str, field: str = "password") -> SecretValue:
        """One field of one entry."""

    def require_available(self) -> None:
        """Raise the «install it» error unless this provider is usable."""
        if not self.available():
            raise self.missing_error()

    def missing_error(self) -> SecretProviderMissing:
        """The error to raise when the provider is not installed."""
        return SecretProviderMissing(
            f"{self.name} provider is unavailable",
            user_message=f"{self.title_ru or self.name} не настроен на этом компьютере.",
        )

    def locked_error(self) -> VaultLocked:
        """The error to raise when the vault is locked."""
        return VaultLocked(
            f"{self.name} vault is locked",
            user_message=(
                f"Хранилище {self.title_ru or self.name} закрыто. "
                "Введите мастер-пароль в настройках."
            ),
        )
