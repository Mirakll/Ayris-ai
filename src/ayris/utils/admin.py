"""Whether this process has administrator rights, and how to borrow them.

Ayris runs unelevated. That is a decision, not an oversight: a voice assistant
that listens all day at high integrity is a much larger target than one that asks
for rights the moment it actually needs them, and a UAC prompt shown at the
moment the user said «выключи Wi-Fi» is a prompt they can connect to something
they asked for.

So there are two halves here.

:func:`is_elevated` and :func:`elevation` answer *do we have rights right now*.
The answer is cached — the token's elevation flag cannot change for the lifetime
of a process, and reading it means opening a token per call, which the action
layer would otherwise do on every audit row. :func:`reset_elevation_cache` exists
for the tests.

:func:`run_elevated` runs *one command* elevated by handing it to
``ShellExecuteW`` with the ``runas`` verb. The new process is elevated; Ayris is
not, and stays that way. The user declining the prompt is an ordinary outcome
here, not an error — :class:`ElevationDeclined` says so as a distinct type, so a
caller can tell «нет прав» from «пользователь отказался» and say something
different.

Nothing in this module imports from :mod:`ayris.core`: like the rest of
``ayris.utils`` it sits at the bottom of the dependency graph, so the errors are
:class:`ayris.utils.winapi.WinApiError` subclasses and the action layer is what
turns them into Russian.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ALWAYS_ADMIN_WARNING_RU",
    "AdminCapability",
    "AdminStatus",
    "ElevatedOutput",
    "ElevationDeclined",
    "ElevationUnavailable",
    "admin_status",
    "can_elevate",
    "configure_always_admin",
    "elevation",
    "format_arguments",
    "is_elevated",
    "requires_elevation",
    "reset_elevation_cache",
    "run_elevated",
    "run_elevated_output",
    "uac_enabled",
    "user_is_admin",
]

_log = get_logger(__name__)

#: How long :func:`run_elevated` waits for the helper by default. Long enough for
#: a ``netsh`` call plus the seconds the prompt itself sits on screen, short
#: enough that a forgotten dialog does not wedge an action past its timeout.
DEFAULT_ELEVATED_TIMEOUT_MS: Final = 60_000

#: The verb that makes the shell show the UAC prompt.
RUNAS_VERB: Final = "runas"

_cached: winapi.ElevationInfo | None = None
_uac_cached: bool | None = None
_admin_cached: bool | None = None

UAC_KEY: Final = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System"
UAC_VALUE: Final = "EnableLUA"
ALWAYS_ADMIN_TASK: Final = r"Ayris\Ayris (administrator)"
ALWAYS_ADMIN_WARNING_RU: Final = (
    "Ayris будет всегда работать с правами администратора. Она сможет перехватывать ввод "
    "над окнами администратора, но drag-and-drop из обычного Проводника перестанет работать, "
    "а каждый загруженный плагин получит те же права. Настройку можно выключить без следов."
)


class AdminCapability(StrEnum):
    ELEVATED = "elevated"
    UAC_DISABLED = "uac_disabled"
    NOT_ADMIN = "not_admin"
    CAN_ELEVATE = "can_elevate"


@dataclass(frozen=True, slots=True)
class AdminStatus:
    state: AdminCapability
    elevated: bool
    uac_enabled: bool
    user_is_admin: bool

    @property
    def message_ru(self) -> str:
        if self.state is AdminCapability.ELEVATED:
            return "Ayris уже запущена с правами администратора."
        if self.state is AdminCapability.UAC_DISABLED:
            return "UAC выключен — повышать права через диалог нечего. Включите UAC в Windows."
        if self.state is AdminCapability.NOT_ADMIN:
            return (
                "Текущий пользователь не входит в группу администраторов — "
                "повышение без учётных данных администратора не поможет."
            )
        return "Повышение возможно: Windows покажет диалог UAC."


@dataclass(frozen=True, slots=True)
class ElevatedOutput:
    run: winapi.ProcessRun
    stdout: str = ""
    stderr: str = ""


class ElevationUnavailable(winapi.WinApiError):
    """Rights cannot be obtained here at all — not Windows, or UAC is off.

    Distinct from a declined prompt: there is nothing for the user to click.
    """


class ElevationDeclined(winapi.WinApiError):
    """The user dismissed the UAC prompt. The command did not run."""


def elevation() -> winapi.ElevationInfo:
    """What the process token says about our rights, read once and remembered.

    Never raises: off Windows, or when the token refuses to answer, the honest
    reply is «no rights», and an action that needs them will refuse rather than
    half-run.
    """
    global _cached
    if _cached is not None:
        return _cached
    if not winapi.available():
        _cached = winapi.ElevationInfo()
        return _cached
    try:
        _cached = winapi.process_elevation()
    except winapi.WinApiError as exc:
        _log.warning("не удалось прочитать токен процесса, считаю права обычными: %s", exc)
        _cached = winapi.ElevationInfo()
    else:
        _log.debug(
            "процесс запущен %s (тип %d, уровень %#x)",
            "с повышением" if _cached.elevated else "без повышения",
            _cached.elevation_type,
            _cached.integrity_level,
        )
    return _cached


def is_elevated() -> bool:
    """Whether this process holds administrator rights."""
    return elevation().elevated


def can_elevate() -> bool:
    """Whether asking for rights could plausibly succeed.

    True for an administrator running with a filtered token — the case a UAC
    prompt was invented for. False when already elevated (nothing to ask for) and
    false for a plain user, where the prompt would demand another account's
    password and there is no point offering it as if it were one click.
    """
    return admin_status().state is AdminCapability.CAN_ELEVATE


def uac_enabled() -> bool:
    """Whether Windows has UAC enabled (``EnableLUA``), cached per process."""
    global _uac_cached
    if _uac_cached is not None:
        return _uac_cached
    if not winapi.available():
        _uac_cached = False
        return _uac_cached
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, UAC_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, UAC_VALUE)
        _uac_cached = bool(int(value))
    except (OSError, TypeError, ValueError) as exc:
        _log.warning("не удалось прочитать EnableLUA, считаю UAC включённым: %s", exc)
        _uac_cached = True
    return _uac_cached


def user_is_admin() -> bool:
    """Whether the current user belongs to the Administrators group."""
    global _admin_cached
    if _admin_cached is not None:
        return _admin_cached
    if not winapi.available():
        _admin_cached = False
        return _admin_cached
    try:
        _admin_cached = winapi.current_user_is_admin()
    except winapi.WinApiError as exc:
        _log.warning("не удалось проверить группу администраторов: %s", exc)
        _admin_cached = False
    return _admin_cached


def admin_status() -> AdminStatus:
    """Describe the four meaningful rights states with a user-facing reason."""
    elevated = is_elevated()
    uac = uac_enabled()
    member = user_is_admin()
    if elevated:
        state = AdminCapability.ELEVATED
    elif not uac:
        state = AdminCapability.UAC_DISABLED
    elif not member:
        state = AdminCapability.NOT_ADMIN
    else:
        state = AdminCapability.CAN_ELEVATE
    return AdminStatus(state, elevated=elevated, uac_enabled=uac, user_is_admin=member)


def reset_elevation_cache() -> None:
    """Forget the cached token answer. Test seam."""
    global _admin_cached, _cached, _uac_cached
    _cached = None
    _uac_cached = None
    _admin_cached = None


def requires_elevation(what: str) -> None:
    """Raise unless this process already has rights, naming what needed them.

    A guard for the places that cannot delegate to a helper process — code that
    must run inside Ayris itself to be of any use.

    Raises:
        ElevationDeclined: We are unelevated but could ask.
        ElevationUnavailable: We are unelevated and cannot ask.
    """
    status = admin_status()
    if status.elevated:
        return
    error = f"{what} requires elevation: {status.state}"
    if status.state is AdminCapability.CAN_ELEVATE:
        raise ElevationDeclined(f"{error}; {status.message_ru}")
    raise ElevationUnavailable(f"{error}; {status.message_ru}")


def format_arguments(arguments: Sequence[str] | str) -> str:
    """Quote arguments into the single string ``ShellExecuteW`` takes.

    The shell wants one command line, not a list, and the naive ``" ".join`` puts
    a network named ``Home Wi-Fi`` into ``netsh`` as two arguments.
    ``shlex.quote`` uses POSIX rules, which are the wrong rules here: Windows
    parses backslashes and double quotes, and single quotes mean nothing to it.
    """
    if isinstance(arguments, str):
        return arguments
    return " ".join(_quote(argument) for argument in arguments)


def _quote(argument: str) -> str:
    """One argument, quoted the way ``CommandLineToArgvW`` will read it back."""
    if argument and not any(character in argument for character in ' \t\n\v"'):
        return argument
    out = ['"']
    backslashes = 0
    for character in argument:
        if character == "\\":
            backslashes += 1
            continue
        if character == '"':
            # Backslashes before a quote are doubled, then the quote is escaped.
            out.append("\\" * (backslashes * 2 + 1))
            out.append('"')
        else:
            out.append("\\" * backslashes)
            out.append(character)
        backslashes = 0
    out.append("\\" * (backslashes * 2))
    out.append('"')
    return "".join(out)


def run_elevated(
    executable: str,
    arguments: Sequence[str] | str = (),
    *,
    directory: str = "",
    wait: bool = True,
    timeout_ms: int = DEFAULT_ELEVATED_TIMEOUT_MS,
    show: int = winapi.SW_HIDE,
) -> winapi.ProcessRun:
    """Run one command with administrator rights, leaving Ayris unelevated.

    Args:
        executable: What to run. A bare name is resolved by the shell.
        arguments: Argument list, quoted by :func:`format_arguments`, or a ready
            command line.
        directory: Working directory for the helper.
        wait: Wait for the helper and report its exit code. On by default: the
            caller almost always needs to know whether the thing it asked for
            happened, and a fire-and-forget ``netsh`` that failed silently is the
            worst of both worlds.
        timeout_ms: How long to wait when ``wait`` is set.
        show: Window state. Hidden by default — a console flashing up is noise,
            and the helpers here have no interface worth seeing.

    Returns:
        The helper's pid, and its exit code once it finished.

    Raises:
        ElevationDeclined: The user dismissed the UAC prompt.
        ElevationUnavailable: This is not Windows.
        winapi.WinApiError: The shell refused for some other reason.
    """
    if not winapi.available():
        raise ElevationUnavailable(f"cannot elevate {executable!r}: not running on Windows")
    line = format_arguments(arguments)
    _log.info("запрашиваю права администратора для %s %s", executable, line)
    try:
        run = winapi.shell_execute_ex(
            executable,
            arguments=line,
            directory=directory,
            verb=RUNAS_VERB,
            show=show,
            wait_ms=timeout_ms if wait else 0,
        )
    except winapi.WinApiError as exc:
        if exc.code == winapi.ERROR_CANCELLED:
            raise ElevationDeclined(f"the elevation prompt for {executable!r} was dismissed") from (
                exc
            )
        raise
    if run.timed_out:
        _log.warning("%s с повышением не завершился за %d мс", executable, timeout_ms)
    return run


def run_elevated_output(
    executable: str,
    arguments: Sequence[str] | str = (),
    *,
    directory: str = "",
    timeout_ms: int = DEFAULT_ELEVATED_TIMEOUT_MS,
) -> ElevatedOutput:
    """Run an elevated helper and collect its output through a profile temp file.

    ``ShellExecuteExW`` cannot redirect handles across the elevation boundary. A
    tiny standard-library helper does the redirect in the elevated process and
    atomically leaves JSON for the unelevated caller to read.
    """
    from ayris.core.paths import get_paths

    output_dir = get_paths().cache_dir / "elevated"
    output_dir.mkdir(parents=True, exist_ok=True)
    fd, raw_path = tempfile.mkstemp(prefix="result-", suffix=".json", dir=output_dir)
    os.close(fd)
    output_path = Path(raw_path)
    try:
        helper = (
            "import json,subprocess,sys;"
            "r=subprocess.run(sys.argv[2:],capture_output=True,text=True,encoding='utf-8',"
            "errors='replace',check=False);"
            "open(sys.argv[1],'w',encoding='utf-8').write(json.dumps("
            "{'stdout':r.stdout,'stderr':r.stderr,'returncode':r.returncode},ensure_ascii=False))"
        )
        run = run_elevated(
            sys.executable,
            ["-c", helper, str(output_path), executable, *(_argument_list(arguments))],
            directory=directory,
            timeout_ms=timeout_ms,
        )
        if run.timed_out or not output_path.stat().st_size:
            return ElevatedOutput(run=run)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        child_run = winapi.ProcessRun(
            pid=run.pid, exit_code=int(payload.get("returncode", 1)), timed_out=False
        )
        return ElevatedOutput(
            run=child_run,
            stdout=str(payload.get("stdout", "")),
            stderr=str(payload.get("stderr", "")),
        )
    finally:
        output_path.unlink(missing_ok=True)


def _argument_list(arguments: Sequence[str] | str) -> list[str]:
    if isinstance(arguments, str):
        # Ready command lines cannot safely be split back into arguments. Run them
        # through cmd, which owns Windows command-line parsing.
        return ["cmd.exe", "/d", "/s", "/c", arguments]
    return list(arguments)


def configure_always_admin(
    enabled: bool,
    *,
    executable: str | None = None,
    arguments: Sequence[str] = (),
) -> winapi.ProcessRun:
    """Create or remove the reversible logon task used for always-admin mode.

    The caller must show :data:`ALWAYS_ADMIN_WARNING_RU` and obtain an explicit
    confirmation before enabling. Disabling removes the task instead of merely
    disabling it, so no scheduler artefact remains.
    """
    if enabled:
        target = executable or sys.executable
        command = format_arguments([target, *arguments])
        task_args = [
            "/Create",
            "/TN",
            ALWAYS_ADMIN_TASK,
            "/SC",
            "ONLOGON",
            "/RL",
            "HIGHEST",
            "/TR",
            command,
            "/F",
        ]
    else:
        task_args = ["/Delete", "/TN", ALWAYS_ADMIN_TASK, "/F"]
    return run_elevated("schtasks.exe", task_args)
