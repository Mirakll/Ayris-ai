"""KeePassXC through ``keepassxc-cli``.

KeePassXC has no stable Python API and no service to talk to: the database is a
file, and the only supported way to read it from outside the application is its
command-line tool. Every call therefore opens the ``.kdbx``, derives the key with
the deliberately slow KDF, answers one question and exits — which is why the
master password is cached for the session (see :class:`SessionCache`) and why
listing is done once rather than per field.

Three details decide whether this works at all:

* the master password goes to **stdin**, never into ``argv`` — ``keepassxc-cli``
  prompts on stderr and reads the answer from standard input for exactly that
  reason;
* ``--quiet`` suppresses the prompt itself, so the first line of stdout is data
  and not «Enter password to unlock…»;
* a wrong password and a missing database both exit non-zero with a message on
  stderr, and they need different Russian sentences, so the text is matched.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.actions.system.secrets.base import (
    CliResult,
    CliRunner,
    SecretEntry,
    SecretProvider,
    SecretProviderError,
    SecretProviderMissing,
    SecretValue,
    SessionCache,
    SubprocessRunner,
    VaultLocked,
    find_cli,
)
from ayris.core.config import get_settings
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.config import AutofillActionsConfig

__all__ = ["KeePassProvider"]

_log = get_logger(__name__)

#: Executable names, in the order they are tried. ``.exe`` is not spelled out:
#: :func:`shutil.which` adds ``PATHEXT`` itself on Windows.
_CLI_NAMES: Final = ("keepassxc-cli",)

#: What ``keepassxc-cli`` calls the fields Ayris uses. ``show -a <attribute>``
#: takes these names, and they are not the ones a person would say.
_ATTRIBUTES: Final[dict[str, str]] = {
    "password": "Password",
    "username": "UserName",
    "login": "UserName",
    "url": "URL",
    "notes": "Notes",
    "title": "Title",
    "totp": "TOTP",
}

#: Fragments of stderr that mean «the password was wrong», as opposed to «the file
#: is not there». Matched case-insensitively; both the English build and the
#: Russian one are covered because the tool follows the system locale.
_WRONG_PASSWORD_MARKERS: Final = (
    "invalid credentials",
    "wrong key",
    "could not decrypt",
    "неверн",
)


class KeePassProvider(SecretProvider):
    """Reads a ``.kdbx`` database by shelling out to ``keepassxc-cli``."""

    name = "keepass"
    title_ru = "KeePassXC"

    def __init__(
        self,
        *,
        runner: CliRunner | None = None,
        settings: AutofillActionsConfig | None = None,
    ) -> None:
        self._runner = runner if runner is not None else SubprocessRunner()
        self._settings = settings
        self._session = SessionCache(self._config().session_ttl_s)
        self._entries: tuple[SecretEntry, ...] | None = None

    # ------------------------------------------------------------------ config

    def _config(self) -> AutofillActionsConfig:
        return self._settings if self._settings is not None else get_settings().actions.autofill

    def executable(self) -> str:
        """Path to ``keepassxc-cli``, or ``""``. Looked up on every call: the user
        may install it while Ayris runs, and caching «not installed» would then be
        wrong until a restart."""
        return find_cli(self._config().keepass_cli, *_CLI_NAMES)

    def database(self) -> str:
        """The configured ``.kdbx`` path.

        Passed through as it is written, Cyrillic and all: ``keepassxc-cli`` is a Qt
        program and receives its arguments as Unicode, unlike the C libraries that
        forced :func:`ayris.core.paths.native_path` on the rest of the project.
        """
        configured = self._config().keepass_database
        if not configured:
            return ""
        return str(Path(configured).expanduser())

    # --------------------------------------------------------------- lifecycle

    def available(self) -> bool:
        return bool(self.executable()) and bool(self._config().keepass_database)

    def unlocked(self) -> bool:
        return bool(self._session.get())

    def unlock(self, master_password: str) -> None:
        """Verify the master password and remember it for the session.

        Verification is a real call — ``ls`` on the database — because a wrong
        password stored silently would fail later, in the middle of filling a form,
        with nothing to point at.
        """
        self.require_available()
        if not master_password:
            raise self.locked_error()
        result = self._run(("ls", "--quiet", "--flatten", self.database()), stdin=master_password)
        if not result.ok:
            raise self._error_for(result)
        self._session.store(master_password)
        self._entries = _parse_listing(result.stdout)
        _log.info("keepass vault unlocked, %d entries", len(self._entries or ()))

    def lock(self) -> None:
        self._session.drop()
        self._entries = None

    # ------------------------------------------------------------------ reading

    def list_entries(self) -> tuple[SecretEntry, ...]:
        """Every entry path in the database, without values.

        Cached after the first listing: each call re-derives the key, which takes
        the better part of a second by design, and a picker that re-listed on every
        keystroke would be unusable.
        """
        if self._entries is not None:
            return self._entries
        self.require_available()
        password = self._require_session()
        result = self._run(("ls", "--quiet", "--flatten", self.database()), stdin=password)
        if not result.ok:
            raise self._error_for(result)
        self._entries = _parse_listing(result.stdout)
        return self._entries

    def get_field(self, entry: str, field: str = "password") -> SecretValue:
        """One attribute of one entry, as a redacted :class:`SecretValue`."""
        self.require_available()
        password = self._require_session()
        attribute = _ATTRIBUTES.get(field.lower(), field)
        result = self._run(
            ("show", "--quiet", "--show-protected", "-a", attribute, self.database(), entry),
            stdin=password,
        )
        if not result.ok:
            raise self._error_for(result, entry=entry, field=field)
        # Only the trailing newline is stripped: a password may legitimately start
        # or end with a space, and .strip() would quietly hand over a different one.
        return SecretValue(result.stdout.rstrip("\r\n"))

    # ------------------------------------------------------------------ plumbing

    def _require_session(self) -> str:
        password = self._session.get()
        if not password:
            raise self.locked_error()
        return password

    def _run(self, args: tuple[str, ...], *, stdin: str) -> CliResult:
        executable = self.executable()
        if not executable:
            raise self.missing_error()
        # The trailing newline matters: keepassxc-cli reads a line, and without it
        # the tool waits for one until the timeout.
        return self._runner.run((executable, *args), stdin=f"{stdin}\n")

    def _error_for(
        self, result: CliResult, *, entry: str = "", field: str = ""
    ) -> SecretProviderError:
        """Turn a non-zero exit into the right Russian sentence.

        Only the entry and field *names* reach the log — never stdout, which is
        where the secret would be.
        """
        message = result.short_error()
        lowered = message.lower()
        where = f" ({entry}/{field})" if entry else ""
        _log.warning("keepassxc-cli failed%s: %s", where, message)
        if any(marker in lowered for marker in _WRONG_PASSWORD_MARKERS):
            self.lock()
            return VaultLocked(
                f"keepassxc-cli rejected the master password: {message}",
                user_message="Мастер-пароль KeePassXC не подошёл.",
            )
        if "no such file" in lowered or "does not exist" in lowered:
            return SecretProviderMissing(
                f"kdbx database not found: {message}",
                user_message="База KeePassXC не найдена, проверьте путь в настройках.",
            )
        if entry:
            return SecretProviderError(
                f"keepassxc-cli could not read {entry}/{field}: {message}",
                user_message=f"В KeePassXC нет записи «{entry}» или поля «{field}».",
            )
        return SecretProviderError(
            f"keepassxc-cli failed: {message}",
            user_message="KeePassXC не смог выполнить запрос.",
        )


def _parse_listing(stdout: str) -> tuple[SecretEntry, ...]:
    """Entry paths out of ``ls --flatten`` output.

    One path per line. Groups end with ``/`` and are dropped — a group is not
    something a template can point at. ``[empty]`` is what the tool prints for an
    empty database.
    """
    entries: list[SecretEntry] = []
    for line in stdout.splitlines():
        path = line.strip()
        if not path or path.endswith("/") or path == "[empty]":
            continue
        title = path.rsplit("/", 1)[-1]
        entries.append(SecretEntry(path=path, title=title))
    return tuple(entries)
