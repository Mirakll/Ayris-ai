"""Starting and stopping programs.

Two actions: :class:`RunApp` and :class:`CloseApp`. Both take a spoken name —
«хром», «вс код», «диспетчер задач» — and hand it to the index from
:mod:`ayris.actions.system.app_index`, which either answers with one program,
refuses, or asks which of two was meant. None of that logic lives here.

**Three ways to start something, and they are not interchangeable.** A plain
``.exe`` is started with its arguments and working directory. A Start-menu ``.lnk``
must go through ``ShellExecuteW``: the shortcut carries its own target, arguments,
icon and «run as administrator» flag, and re-implementing that by parsing the
binary would get the easy cases right and the interesting ones wrong. A Store
application has no path at all — it is a package plus an application id, and the
only supported way in is ``explorer.exe shell:AppsFolder\\<AUMID>``, which asks the
shell to activate the package the same way a click on the tile does.

**Closing is not killing.** ``CloseApp`` posts ``WM_CLOSE`` to every window the
program owns, which is what clicking the cross does: the editor gets to ask about
unsaved changes, the browser gets to write its session. Only when the caller passed
``force`` — and only after the grace period — does it reach for
``TerminateProcess``, and it says so in the result. A voice assistant that discards
an unsaved document because it heard «закрой ворд» is worse than one that reports
«Word не закрылся сам».
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.actions.system.app_index import (
    STORE_PREFIX,
    AppCandidate,
    AppNotFound,
    get_app_index,
)
from ayris.actions.system.windows import WindowQuery, WindowRecord, get_window_backend
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.actions.system.windows import WindowBackend

__all__ = [
    "AppLauncher",
    "CloseApp",
    "CloseOutcome",
    "LaunchKind",
    "LaunchRequest",
    "RunApp",
    "WinApiLauncher",
    "get_launcher",
    "set_launcher",
]

_log = get_logger(__name__)

#: How long ``CloseApp`` waits for a program to close itself, by default.
DEFAULT_CLOSE_TIMEOUT_MS: Final = 5_000

#: The shell binary that activates Store packages. Never a full path: it is
#: resolved through ``App Paths``, and hard-coding ``C:\Windows`` is wrong on a
#: machine whose Windows lives elsewhere.
SHELL_BINARY: Final = "explorer.exe"

#: Pause between polls while waiting for a program to exit.
_POLL_S: Final = 0.05


class LaunchKind(StrEnum):
    """Which of the three launch paths a target needs."""

    EXECUTABLE = "executable"
    SHORTCUT = "shortcut"
    STORE = "store"

    @classmethod
    def of(cls, target: str) -> LaunchKind:
        """Classify a target string the way :class:`RunApp` will launch it."""
        lowered = target.strip().lower()
        if lowered.startswith(STORE_PREFIX.lower()) or lowered.startswith("shell:"):
            return cls.STORE
        if lowered.endswith(".lnk") or lowered.endswith(".url"):
            return cls.SHORTCUT
        return cls.EXECUTABLE


@dataclass(frozen=True, slots=True)
class LaunchRequest:
    """Everything needed to start one program, decided but not yet done.

    A value object rather than four arguments, because it is the whole contract
    between the decision (which program, which path, which arguments) and the
    backend that performs it — which is what lets the tests assert that «открой
    настройки» reaches the shell as ``explorer.exe shell:AppsFolder\\…`` and not as
    a file path.
    """

    target: str
    kind: LaunchKind = LaunchKind.EXECUTABLE
    arguments: str = ""
    working_dir: str = ""
    name: str = ""

    @classmethod
    def for_candidate(
        cls,
        candidate: AppCandidate,
        *,
        arguments: str = "",
        working_dir: str = "",
    ) -> LaunchRequest:
        """Build a request from a resolved application.

        Arguments given in the command win over the ones the shortcut carried: «открой
        хром с профилем» is a deliberate override, and silently appending both would
        produce a command line neither the user nor the shortcut's author meant.

        Raises:
            AppNotFound: the match has nothing launchable behind it — a dictionary
                entry for a program that is not installed and has no bare
                executable name to fall back on.
        """
        target = candidate.target.strip()
        if not target:
            raise AppNotFound(
                f"application {candidate.app_id!r} has no launch target",
                user_message=f"Знаю «{candidate.name}», но не нашла его на этом компьютере.",
            )
        app = candidate.app
        return cls(
            target=target,
            kind=LaunchKind.of(target),
            arguments=arguments.strip() or (app.arguments if app is not None else ""),
            working_dir=working_dir.strip() or (app.working_dir if app is not None else ""),
            name=candidate.name,
        )

    @property
    def shell_file(self) -> str:
        """What ``ShellExecuteW`` gets as its file: the target, or the shell."""
        return SHELL_BINARY if self.kind is LaunchKind.STORE else self.target

    @property
    def shell_arguments(self) -> str:
        """What ``ShellExecuteW`` gets as its arguments.

        For a Store application the moniker *is* the argument, and anything the
        command added follows it — a packaged app receives extra arguments through
        its activation contract, if it declares one.
        """
        if self.kind is not LaunchKind.STORE:
            return self.arguments
        return f"{self.target} {self.arguments}".strip()

    @property
    def command_ru(self) -> str:
        """The command as one line, for the log and the audit trail."""
        parts = [self.shell_file, self.shell_arguments]
        return " ".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class CloseOutcome:
    """What happened while closing a program."""

    windows: int = 0
    closed: bool = False
    killed: tuple[int, ...] = ()
    alive: tuple[int, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the program is gone, however it went."""
        return self.closed or bool(self.killed)


class AppLauncher(Protocol):
    """The operating system, as far as starting and stopping programs goes."""

    def launch(self, request: LaunchRequest) -> int:
        """Start a program and return its process id, or ``0`` when unknown."""
        ...

    def close_window(self, hwnd: int) -> bool:
        """Ask one window to close, the way the title-bar cross does."""
        ...

    def running(self, pid: int) -> bool:
        """Whether a process is still alive."""
        ...

    def wait_exit(self, pids: Sequence[int], timeout_ms: int) -> tuple[int, ...]:
        """Wait for processes to exit; return the ones still running."""
        ...

    def terminate(self, pid: int) -> bool:
        """Kill a process. Only ever called with an explicit ``force``."""
        ...


class WinApiLauncher:
    """The real launcher, over :mod:`ayris.utils.winapi`."""

    def launch(self, request: LaunchRequest) -> int:
        """Start a program through the shell.

        ``ShellExecuteW`` for all three kinds, including plain executables: it
        applies the shortcut's own settings, it triggers the UAC prompt when the
        target asks for elevation, and it does not leave us as the parent of a
        program that will outlive the assistant.

        Raises:
            ActionError: the shell refused. The Russian text is the user's; the
                Windows message goes to the log.
        """
        try:
            return winapi.shell_execute(
                request.shell_file,
                arguments=request.shell_arguments,
                directory=request.working_dir,
            )
        except winapi.WinApiError as exc:
            raise ActionError(
                f"failed to launch {request.command_ru!r}: {exc}",
                user_message=f"Не смогла запустить «{request.name or request.target}».",
            ) from exc

    def close_window(self, hwnd: int) -> bool:
        """``WM_CLOSE``, posted rather than sent.

        Posting does not block on a program that is busy or showing a modal
        «сохранить изменения?» dialog — which is exactly the case where a blocking
        send would stall the action until its timeout.
        """
        return winapi.post_close(hwnd)

    def running(self, pid: int) -> bool:
        return winapi.process_running(pid)

    def wait_exit(self, pids: Sequence[int], timeout_ms: int) -> tuple[int, ...]:
        deadline = time.monotonic() + max(0, timeout_ms) / 1000.0
        pending = tuple(dict.fromkeys(pid for pid in pids if pid))
        while pending:
            pending = tuple(pid for pid in pending if winapi.process_running(pid))
            if not pending or time.monotonic() >= deadline:
                break
            time.sleep(_POLL_S)
        return pending

    def terminate(self, pid: int) -> bool:
        try:
            winapi.terminate_process(pid)
        except winapi.WinApiError as exc:
            _log.warning("не удалось завершить процесс %s: %s", pid, exc)
            return False
        return True


_launcher: AppLauncher | None = None


def get_launcher() -> AppLauncher:
    """The launcher in force. Real WinAPI unless a test replaced it.

    Raises:
        ActionUnavailable: this is not Windows and nothing was injected.
    """
    if _launcher is not None:
        return _launcher
    if not winapi.available():
        raise ActionUnavailable(
            "application actions require Windows",
            user_message="Запуск программ работает только в Windows.",
        )
    return WinApiLauncher()


def set_launcher(launcher: AppLauncher | None) -> None:
    """Install a launcher, or restore the real one with ``None``. Test seam."""
    global _launcher
    _launcher = launcher


def app_windows(
    candidate: AppCandidate,
    backend: WindowBackend,
) -> list[WindowRecord]:
    """Windows belonging to a resolved program.

    By process name when there is an executable to compare against, by caption
    otherwise — a Store application runs inside ``ApplicationFrameHost.exe`` along
    with every other Store application, so its process name identifies nothing.
    """
    stem = _executable_stem(candidate)
    query = WindowQuery(process=stem) if stem else WindowQuery(title=candidate.name)
    return [record for record in backend.list_windows() if query.matches(record)]


def _executable_stem(candidate: AppCandidate) -> str:
    """Process name to look for, without ``.exe``, or ``""`` for Store apps."""
    app = candidate.app
    if app is not None:
        if app.is_store:
            return ""
        source = app.executable or (app.target if not app.is_shortcut else "")
        if source:
            return Path(source).stem.casefold()
    entry = candidate.entry
    if entry is not None and not entry.uwp:
        executable = entry.primary_executable
        if executable:
            return Path(executable).stem.casefold()
    return ""


@register
class RunApp(Action):
    """Start a program by the name a person would say."""

    meta: ClassVar = ActionMeta(
        name="RunApp",
        category=ActionCategory.APPS,
        title_ru="Запустить программу",
        description_ru="Найти программу по названию и запустить её",
        timeout_ms=15_000,
    )

    class Params(ActionParams):
        app: str = Field(
            min_length=1,
            max_length=120,
            description="Название программы, например «хром» или «вс код»",
        )
        arguments: str = Field(
            default="",
            max_length=1_000,
            description="Аргументы командной строки",
        )
        working_dir: str = Field(
            default="",
            max_length=500,
            description="Рабочая папка",
        )

    def run(self, params: Params) -> ActionResult[int]:
        index = get_app_index()
        candidate = index.resolve(params.app)
        request = LaunchRequest.for_candidate(
            candidate,
            arguments=params.arguments,
            working_dir=params.working_dir,
        )
        pid = get_launcher().launch(request)
        launches = index.note_launch(candidate.app_id)
        _log.info("запуск «%s»: %s (pid %s)", candidate.name, request.command_ru, pid)
        return ActionResult.done(
            f"Открываю «{candidate.name}».",
            value=pid,
            detail=f"launched {request.command_ru} as pid {pid}",
            data={
                "app_id": candidate.app_id,
                "name": candidate.name,
                "target": request.target,
                "kind": str(request.kind),
                "pid": pid,
                "confidence": round(candidate.confidence, 3),
                "launches": launches,
            },
        )


@register
class CloseApp(Action):
    """Close a program politely, and kill it only when told to.

    Marked dangerous because of the ``force`` path: an unsaved document is a real
    loss, and the confirmation layer from section 14 exists for exactly this.
    """

    meta: ClassVar = ActionMeta(
        name="CloseApp",
        category=ActionCategory.APPS,
        title_ru="Закрыть программу",
        description_ru="Закрыть окна программы, при необходимости завершить процесс",
        is_dangerous=True,
        timeout_ms=20_000,
    )

    class Params(ActionParams):
        app: str = Field(
            min_length=1,
            max_length=120,
            description="Название программы",
        )
        force: bool = Field(
            default=False,
            title="Принудительно",
            description="Завершить процесс, если окна не закрылись сами",
        )
        timeout_ms: int = Field(
            default=DEFAULT_CLOSE_TIMEOUT_MS,
            ge=200,
            le=30_000,
            description="Сколько ждать закрытия",
            json_schema_extra={"unit_ru": "мс"},
        )

    def run(self, params: Params) -> ActionResult[CloseOutcome]:
        index = get_app_index()
        candidate = index.resolve(params.app)
        backend = get_window_backend()
        launcher = get_launcher()

        records = app_windows(candidate, backend)
        if not records:
            return ActionResult.failed(
                f"«{candidate.name}» и так не запущен.",
                detail=f"no windows for {candidate.app_id}",
                value=CloseOutcome(),
                data={"app_id": candidate.app_id, "name": candidate.name},
            )

        pids = tuple(dict.fromkeys(record.pid for record in records if record.pid))
        for record in records:
            launcher.close_window(record.hwnd)
        alive = launcher.wait_exit(pids, params.timeout_ms)

        killed: tuple[int, ...] = ()
        if alive and params.force:
            killed = tuple(pid for pid in alive if launcher.terminate(pid))
            alive = tuple(pid for pid in alive if pid not in killed)

        outcome = CloseOutcome(
            windows=len(records),
            closed=not alive and not killed,
            killed=killed,
            alive=alive,
        )
        return self._result(candidate.name, candidate.app_id, outcome)

    def _result(
        self,
        name: str,
        app_id: str,
        outcome: CloseOutcome,
    ) -> ActionResult[CloseOutcome]:
        """Turn the outcome into a phrase that matches what actually happened."""
        data = {
            "app_id": app_id,
            "name": name,
            "windows": outcome.windows,
            "killed": list(outcome.killed),
            "alive": list(outcome.alive),
        }
        if outcome.closed:
            return ActionResult.done(f"Закрыла «{name}».", value=outcome, data=data)
        if outcome.killed:
            return ActionResult.done(
                f"«{name}» не закрылся сам, завершила процесс.",
                value=outcome,
                detail=f"terminated pids {outcome.killed}",
                data=data,
            )
        return ActionResult.failed(
            f"«{name}» не закрывается — возможно, спрашивает про несохранённое.",
            detail=f"pids still alive: {outcome.alive}",
            value=outcome,
            data=data,
        )
