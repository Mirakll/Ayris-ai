"""Sleep, hibernation, reboot, shutdown, log-off and the lock screen.

Two actions: :class:`PowerAction` for everything that ends the session, and
:class:`LockWorkstation` for the Win+L screen. They are split because the safety
story is different, not because the code is: locking is instantly reversible by
typing a password, while the other five throw away whatever was not saved.

**Every path here goes through :class:`PowerBackend`, and the default one records
instead of doing.** A test that reaches the real ``ExitWindowsEx`` does not fail —
it reboots the machine it runs on, taking the rest of the suite with it. So the
backend is chosen explicitly: :func:`set_power_backend` installs the real one, and
until something does, :class:`RecordingPowerBackend` writes the call down and
returns. Production installs it once at startup (task 30); the test suite never
does, and :func:`recorded_power_calls` is how a test asserts which flag would have
gone out.

**Delayed and immediate are two different Windows calls, not one with a timer.**
``InitiateSystemShutdownExW`` hands the countdown to Windows, which shows its own
warning dialog and — the part that matters — can be taken back with
``AbortSystemShutdownW``. A ``threading.Timer`` inside Ayris would lose the
schedule when the assistant restarts and would give the user nothing to cancel
from outside it. Sleep and hibernation have no scheduled form at all, so a delay
on those is waited out in the worker thread and cancelled by
:func:`cancel_pending`.

**Hibernation is checked before it is attempted.** ``powercfg /a`` is the only
answer to «is hibernation turned on here», and on a machine where it is not, the
bare ``SetSuspendState`` call quietly puts the machine to sleep instead — which
looks like Ayris misheard «гибернация» as «сон». Being told «гибернация
отключена в системе» is better than that.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar, Final, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.numbers import plural_form
from ayris.utils import winapi
from ayris.utils.logger import get_logger

__all__ = [
    "MAX_DELAY_S",
    "POWERCFG_TIMEOUT_S",
    "CancelPowerAction",
    "LockWorkstation",
    "PowerAction",
    "PowerBackend",
    "PowerCall",
    "PowerOperation",
    "PowerRequest",
    "RecordingPowerBackend",
    "WinApiPowerBackend",
    "cancel_pending",
    "clear_recorded_power_calls",
    "format_delay_ru",
    "get_power_backend",
    "hibernation_from_powercfg",
    "pending_delay",
    "perform",
    "recorded_power_calls",
    "set_power_backend",
]

_log = get_logger(__name__)

#: Longest delay a power action accepts, in seconds. Windows itself allows years;
#: a day is where «через сколько?» stops being a thing a person says out loud, and
#: a scheduled shutdown nobody remembers asking for is worse than a rejected one.
MAX_DELAY_S: Final = 24 * 60 * 60

#: How long to wait for ``powercfg /a``. It reads the power policy and returns in
#: milliseconds; a hang here would stall the action until its own timeout, so the
#: answer is abandoned rather than waited for.
POWERCFG_TIMEOUT_S: Final = 5.0

#: Granularity of the wait for a delayed sleep. Short enough that «отмени» feels
#: immediate, long enough that the worker thread is not spinning.
_TICK_S: Final = 0.25

#: ``CREATE_NO_WINDOW``, so ``powercfg`` does not flash a console window. Spelled
#: out rather than taken from :mod:`subprocess`, where it only exists on Windows.
_CREATE_NO_WINDOW: Final = 0x08000000


class PowerOperation(StrEnum):
    """What :class:`PowerAction` was asked to do.

    Declaration order is least to most destructive, which is also the order the
    editor lists them in.
    """

    SLEEP = "sleep"
    HIBERNATE = "hibernate"
    LOGOFF = "logoff"
    REBOOT = "reboot"
    SHUTDOWN = "shutdown"

    @property
    def title_ru(self) -> str:
        """Name of the operation for the editor and the confirmation prompt."""
        return _TITLES_RU[self]

    @property
    def announce_ru(self) -> str:
        """What Ayris says while doing it: «Засыпаю», «Выключаю компьютер»."""
        return _ANNOUNCE_RU[self]

    @property
    def suspends(self) -> bool:
        """Whether this is a ``SetSuspendState`` call rather than a session end."""
        return self in (PowerOperation.SLEEP, PowerOperation.HIBERNATE)

    @property
    def schedulable(self) -> bool:
        """Whether Windows can hold the countdown itself, and thus cancel it.

        Only shutdown and reboot: ``InitiateSystemShutdownExW`` does nothing else.
        A log-off has no scheduled form either, so it waits like sleep does.
        """
        return self in (PowerOperation.REBOOT, PowerOperation.SHUTDOWN)

    @property
    def exit_flags(self) -> int:
        """Flags for ``ExitWindowsEx``, or ``0`` for the suspend operations.

        ``EWX_POWEROFF`` rather than a bare ``EWX_SHUTDOWN``: the first cuts power
        as well, which is what «выключи компьютер» means, while the second leaves
        the machine at «It is now safe to turn off your computer» on hardware old
        enough to lack ACPI.
        """
        return _EXIT_FLAGS.get(self, 0)


_TITLES_RU: Final[dict[PowerOperation, str]] = {
    PowerOperation.SLEEP: "Сон",
    PowerOperation.HIBERNATE: "Гибернация",
    PowerOperation.LOGOFF: "Выход из системы",
    PowerOperation.REBOOT: "Перезагрузка",
    PowerOperation.SHUTDOWN: "Выключение",
}

_ANNOUNCE_RU: Final[dict[PowerOperation, str]] = {
    PowerOperation.SLEEP: "Отправляю компьютер в сон.",
    PowerOperation.HIBERNATE: "Отправляю компьютер в гибернацию.",
    PowerOperation.LOGOFF: "Выхожу из системы.",
    PowerOperation.REBOOT: "Перезагружаю компьютер.",
    PowerOperation.SHUTDOWN: "Выключаю компьютер.",
}

_EXIT_FLAGS: Final[dict[PowerOperation, int]] = {
    PowerOperation.LOGOFF: winapi.EWX_LOGOFF,
    PowerOperation.REBOOT: winapi.EWX_REBOOT,
    PowerOperation.SHUTDOWN: winapi.EWX_SHUTDOWN | winapi.EWX_POWEROFF,
}


@dataclass(frozen=True, slots=True)
class PowerRequest:
    """One power operation, decided but not yet performed.

    Separating the decision from the call is what makes the whole thing testable:
    a test builds a request, runs it against :class:`RecordingPowerBackend`, and
    reads back exactly which Windows call it would have been.
    """

    operation: PowerOperation
    delay_s: int = 0
    force: bool = False
    message: str = ""

    @property
    def delayed(self) -> bool:
        """Whether anything has to wait before the machine is touched."""
        return self.delay_s > 0

    @property
    def cancellable(self) -> bool:
        """Whether the user still has a window in which «отмени» works.

        True for anything delayed: a scheduled shutdown is cancelled through
        Windows, a delayed sleep by the thread that is waiting it out.
        """
        return self.delayed

    @property
    def delay_ru(self) -> str:
        """The delay as a person would say it: «через 30 секунд», «через 5 минут»."""
        return format_delay_ru(self.delay_s)

    @property
    def summary_ru(self) -> str:
        """One line for the log, the audit trail and the confirmation prompt."""
        title = self.operation.title_ru.lower()
        return f"{title} {self.delay_ru}" if self.delayed else title


def format_delay_ru(delay_s: int) -> str:
    """Render a delay in Russian, choosing the unit a person would use.

    Whole minutes are said as minutes — «через 5 минут», not «через 300 секунд» —
    and the plural form comes from :func:`ayris.nlu.numbers.plural_form`, so 11
    does not come out as «11 минута».
    """
    if delay_s <= 0:
        return "сейчас"
    if delay_s >= 60 and delay_s % 60 == 0:
        minutes = delay_s // 60
        return f"через {minutes} {plural_form(minutes, 'минуту', 'минуты', 'минут')}"
    return f"через {delay_s} {plural_form(delay_s, 'секунду', 'секунды', 'секунд')}"


@dataclass(frozen=True, slots=True)
class PowerCall:
    """One call :class:`RecordingPowerBackend` intercepted.

    ``kind`` is the backend method that was reached, and the rest are the
    arguments that went with it. This is the record a test asserts on — «what flag
    went into ``ExitWindowsEx``», «what delay was scheduled», «that the cancel
    call happened at all».
    """

    kind: str
    flags: int = 0
    delay_s: int = 0
    hibernate: bool = False
    reboot: bool = False
    force: bool = False
    message: str = ""


class PowerBackend(Protocol):
    """The operating system, as far as ending the session goes.

    Small on purpose: one method per Windows entry point, no decisions. Everything
    that decides — which flag a spoken «выключи» means, whether hibernation is
    even on — lives above this line and is tested against a recording backend.
    """

    def hibernation_available(self) -> bool:
        """Whether hibernation is enabled in the system's power policy."""
        ...

    def suspend(self, *, hibernate: bool, force: bool = False) -> None:
        """``SetSuspendState``: sleep, or hibernate when ``hibernate``."""
        ...

    def exit_windows(self, flags: int) -> None:
        """``ExitWindowsEx``: end the session now, with the privilege enabled."""
        ...

    def schedule(
        self, *, delay_s: int, reboot: bool, message: str = "", force: bool = False
    ) -> None:
        """``InitiateSystemShutdownExW``: hand Windows a countdown it can cancel."""
        ...

    def abort(self) -> bool:
        """``AbortSystemShutdownW``. ``False`` when there was nothing to cancel."""
        ...

    def lock(self) -> None:
        """``LockWorkStation``: the Win+L screen."""
        ...


class WinApiPowerBackend:
    """The real one, over :mod:`ayris.utils.winapi`.

    Nothing constructs this by accident: :func:`get_power_backend` never returns
    it unless :func:`set_power_backend` was called, and the only caller that does
    so is application startup. Every method turns :class:`winapi.WinApiError` into
    a Russian :class:`~ayris.core.errors.ActionError`, because the message reaches
    the user and «[1314] A required privilege is not held by the client» does not.
    """

    def hibernation_available(self) -> bool:
        """Ask ``powercfg /a`` and parse its answer.

        A failure to run it is reported as «available»: refusing hibernation
        because a diagnostic tool is missing would be worse than attempting it and
        letting Windows say no.
        """
        try:
            completed = subprocess.run(
                ["powercfg", "/a"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=POWERCFG_TIMEOUT_S,
                creationflags=_CREATE_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            _log.warning("не удалось спросить powercfg /a: %s", exc)
            return True
        return hibernation_from_powercfg(completed.stdout)

    def suspend(self, *, hibernate: bool, force: bool = False) -> None:
        try:
            winapi.set_suspend_state(hibernate=hibernate, force=force)
        except winapi.WinApiError as exc:
            raise _refused(
                PowerOperation.HIBERNATE if hibernate else PowerOperation.SLEEP, exc
            ) from exc

    def exit_windows(self, flags: int) -> None:
        """End the session, enabling ``SeShutdownPrivilege`` first.

        The privilege is off by default in every token, and ``ExitWindowsEx``
        answers a missing one with ``ERROR_PRIVILEGE_NOT_HELD`` rather than a UAC
        prompt. It is enabled per call rather than once at startup so the
        assistant does not sit there holding the right to reboot the machine.
        """
        _enable_shutdown_privilege(logoff=flags == winapi.EWX_LOGOFF)
        try:
            winapi.exit_windows(flags)
        except winapi.WinApiError as exc:
            raise _refused(_operation_for_flags(flags), exc) from exc

    def schedule(
        self, *, delay_s: int, reboot: bool, message: str = "", force: bool = False
    ) -> None:
        _enable_shutdown_privilege(logoff=False)
        try:
            winapi.initiate_shutdown(
                delay_s=delay_s, reboot=reboot, message=message, force_apps=force
            )
        except winapi.WinApiError as exc:
            raise _refused(
                PowerOperation.REBOOT if reboot else PowerOperation.SHUTDOWN, exc
            ) from exc

    def abort(self) -> bool:
        _enable_shutdown_privilege(logoff=False)
        try:
            winapi.abort_shutdown()
        except winapi.WinApiError as exc:
            _log.info("отменять нечего: %s", exc)
            return False
        return True

    def lock(self) -> None:
        try:
            winapi.lock_workstation()
        except winapi.WinApiError as exc:
            raise ActionError(
                f"LockWorkStation failed: {exc}",
                user_message="Не смогла заблокировать компьютер.",
            ) from exc


class RecordingPowerBackend:
    """Writes the call down instead of making it. The default, and the dry run.

    Two jobs in one class. It is what the test suite asserts against, and it is
    what stands in for the real backend until something installs one — so a build
    that forgets to wire up power actions says «сделала» without doing anything,
    rather than rebooting on the first misheard word.
    """

    def __init__(self, *, hibernation: bool = True) -> None:
        self.calls: list[PowerCall] = []
        self.hibernation = hibernation
        self.scheduled = False

    def _record(self, call: PowerCall) -> None:
        self.calls.append(call)
        _log.info("dry-run питания: %s", call)

    def hibernation_available(self) -> bool:
        return self.hibernation

    def suspend(self, *, hibernate: bool, force: bool = False) -> None:
        self._record(PowerCall(kind="suspend", hibernate=hibernate, force=force))

    def exit_windows(self, flags: int) -> None:
        self._record(PowerCall(kind="exit_windows", flags=flags))

    def schedule(
        self, *, delay_s: int, reboot: bool, message: str = "", force: bool = False
    ) -> None:
        self.scheduled = True
        self._record(
            PowerCall(kind="schedule", delay_s=delay_s, reboot=reboot, message=message, force=force)
        )

    def abort(self) -> bool:
        """Report a cancellation only when there was a schedule to cancel.

        Answering ``True`` unconditionally would be the easier fake and the wrong
        one: :func:`cancel_pending` reads this to decide what to tell the user, and
        a backend that always claims success would make «нечего отменять»
        untestable.
        """
        self._record(PowerCall(kind="abort"))
        aborted, self.scheduled = self.scheduled, False
        return aborted

    def lock(self) -> None:
        self._record(PowerCall(kind="lock"))


_backend: PowerBackend | None = None
_fallback = RecordingPowerBackend()


def get_power_backend() -> PowerBackend:
    """The backend in force — the recording one unless something installed a real one."""
    return _backend if _backend is not None else _fallback


def set_power_backend(backend: PowerBackend | None) -> None:
    """Install a backend, or go back to recording with ``None``.

    ``None`` does *not* mean «the real WinAPI one», unlike the other system
    modules. Falling back to the real thing on a missing argument is the failure
    mode that reboots a developer's machine, so the real backend is only ever
    reached by naming it.
    """
    global _backend
    _backend = backend


def recorded_power_calls() -> tuple[PowerCall, ...]:
    """Calls the default recording backend has intercepted so far."""
    return tuple(_fallback.calls)


def clear_recorded_power_calls() -> None:
    """Forget the recorded calls. Test seam."""
    _fallback.calls.clear()


def hibernation_from_powercfg(output: str) -> bool:
    """Whether ``powercfg /a`` says hibernation is one of the available states.

    The output is two lists — the states that are available and the states that
    are not — under localized headings, with localized state names and, in the
    second list, an indented sentence explaining each refusal. Neither the
    headings nor the sentences can be matched on.

    What is stable is the structure: a heading is the only kind of line that ends
    in a colon, so the first one opens the available list and the second closes
    it. Inside that list only the name of the state itself has to be recognised,
    and :data:`_HIBERNATE_WORDS` covers the two spellings Windows ships. An output
    with no headings at all — a locale nobody anticipated, or an error message —
    reads as available, and Windows gets to refuse the call itself rather than
    Ayris refusing on a parse it is not sure of.
    """
    available: list[str] = []
    section = 0
    for raw in output.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            section += 1
            continue
        if section == 1:
            available.append(line.casefold())
    if section == 0:
        return True
    return any(_hibernate_named(state) for state in available)


#: Spellings of the hibernation state across the locales Windows ships. The
#: Russian one is cut to a stem so «Гибернация» and «Гибернировать» both match.
_HIBERNATE_WORDS: Final = ("hibernate", "иберна")


def _hibernate_named(state: str) -> bool:
    """Whether a ``powercfg`` state name is the hibernation one."""
    return any(word in state for word in _HIBERNATE_WORDS)


def _enable_shutdown_privilege(*, logoff: bool) -> None:
    """Turn on ``SeShutdownPrivilege``, unless this is only a log-off.

    A log-off needs no privilege at all, and asking for one Ayris might not have
    would turn «выйди из системы» into a rights error for no reason.

    Raises:
        ActionError: the privilege could not be enabled, which means the operation
            would fail with an access error a moment later.
    """
    if logoff:
        return
    try:
        enabled = winapi.enable_privilege(winapi.SE_SHUTDOWN_NAME)
    except winapi.WinApiError as exc:
        raise ActionError(
            f"could not enable {winapi.SE_SHUTDOWN_NAME}: {exc}",
            user_message="Windows не дала права на завершение работы.",
        ) from exc
    if not enabled:
        raise ActionError(
            f"{winapi.SE_SHUTDOWN_NAME} is not held by this process",
            user_message="У Ayris нет права выключать компьютер в этой системе.",
        )


def _operation_for_flags(flags: int) -> PowerOperation:
    """Which operation a set of ``ExitWindowsEx`` flags stands for."""
    if flags & winapi.EWX_REBOOT:
        return PowerOperation.REBOOT
    if flags & (winapi.EWX_SHUTDOWN | winapi.EWX_POWEROFF):
        return PowerOperation.SHUTDOWN
    return PowerOperation.LOGOFF


def _refused(operation: PowerOperation, exc: winapi.WinApiError) -> ActionError:
    """Turn a refused Windows call into something worth saying out loud.

    A program vetoing the shutdown is the common case and the one worth naming:
    the user can close the unsaved document and try again, which they cannot do if
    all they hear is «не получилось».
    """
    return ActionError(
        f"{operation.value} refused: {exc}",
        user_message=(
            f"{operation.title_ru} не удалась — Windows отказала. "
            "Возможно, какая-то программа не даёт завершить работу."
        ),
    )


@dataclass(slots=True)
class _Pending:
    """A delayed operation Ayris itself is waiting out, and its cancel switch."""

    request: PowerRequest
    cancel: threading.Event = field(default_factory=threading.Event)


_pending: _Pending | None = None
_pending_lock = threading.Lock()


def pending_delay() -> PowerRequest | None:
    """The delayed operation currently waiting, if any.

    Only covers the ones Ayris waits out itself — sleep, hibernation, log-off. A
    scheduled shutdown lives in Windows and is not visible here, which is why
    :func:`cancel_pending` tries both.
    """
    with _pending_lock:
        return _pending.request if _pending is not None else None


def _claim(request: PowerRequest) -> _Pending:
    """Register a delayed operation, replacing and cancelling any earlier one."""
    global _pending
    with _pending_lock:
        if _pending is not None:
            _pending.cancel.set()
        _pending = _Pending(request=request)
        return _pending


def _release(claim: _Pending) -> None:
    """Deregister a wait, unless a newer one already took its place."""
    global _pending
    with _pending_lock:
        if _pending is claim:
            _pending = None


def cancel_pending() -> str:
    """Cancel whatever power operation is scheduled, wherever it is scheduled.

    Two places to look, and both are tried: a shutdown or reboot handed to
    Windows, and a sleep or log-off this process is waiting out. Trying both is
    the point — the user says «отмени», not «отмени тот, который в Windows».

    Returns:
        Russian text describing what was cancelled, or an empty string when
        nothing was pending.
    """
    global _pending
    with _pending_lock:
        claim, _pending = _pending, None
    cancelled_here = claim is not None
    if claim is not None:
        claim.cancel.set()
    cancelled_windows = get_power_backend().abort()
    if cancelled_windows:
        return "Отменила запланированное выключение."
    if cancelled_here and claim is not None:
        return f"Отменила: {claim.request.operation.title_ru.lower()}."
    return ""


def _wait_out(claim: _Pending) -> bool:
    """Wait for a delayed operation's turn. ``False`` when it was cancelled.

    Polls rather than blocking on the event for the whole delay so a cancellation
    is noticed within :data:`_TICK_S` even if the wait was for hours, and so the
    worker thread the registry gave us stays interruptible.
    """
    deadline = time.monotonic() + claim.request.delay_s
    while time.monotonic() < deadline:
        if claim.cancel.wait(_TICK_S):
            return False
    return not claim.cancel.is_set()


def perform(request: PowerRequest) -> ActionResult[PowerCall | None]:
    """Carry out one power request, delay and all.

    The four shapes, in the order they are decided:

    * scheduled shutdown or reboot — handed to Windows, returns immediately;
    * delayed sleep, hibernation or log-off — waited out here, cancellable;
    * immediate — straight through to the backend;
    * hibernation on a machine where it is off — refused before anything happens.

    Raises:
        ActionUnavailable: hibernation is disabled in the system.
        ActionError: Windows refused the operation.
    """
    backend = get_power_backend()
    if request.operation is PowerOperation.HIBERNATE and not backend.hibernation_available():
        raise ActionUnavailable(
            "hibernation is disabled in the system power policy",
            user_message=(
                "Гибернация отключена в системе. "
                "Включите её командой «powercfg /hibernate on» от администратора."
            ),
        )
    if request.delayed and request.operation.schedulable:
        backend.schedule(
            delay_s=request.delay_s,
            reboot=request.operation is PowerOperation.REBOOT,
            message=request.message,
            force=request.force,
        )
        return ActionResult.done(
            f"{request.operation.title_ru} {request.delay_ru}. Скажите «отмени», если передумали.",
            detail=request.summary_ru,
            data={"operation": str(request.operation), "delay_s": request.delay_s},
        )
    if request.delayed:
        claim = _claim(request)
        try:
            if not _wait_out(claim):
                return ActionResult.failed(
                    f"Отменила: {request.operation.title_ru.lower()}.",
                    detail=f"{request.operation.value} cancelled before it ran",
                )
        finally:
            _release(claim)
    _apply(request, backend)
    return ActionResult.done(
        request.operation.announce_ru,
        detail=request.summary_ru,
        data={"operation": str(request.operation), "delay_s": request.delay_s},
    )


def _apply(request: PowerRequest, backend: PowerBackend) -> None:
    """The call itself, once every decision has been made."""
    if request.operation.suspends:
        backend.suspend(
            hibernate=request.operation is PowerOperation.HIBERNATE, force=request.force
        )
        return
    backend.exit_windows(request.operation.exit_flags)


@register
class PowerAction(Action):
    """Sleep, hibernate, log off, reboot or shut down.

    ``is_dangerous`` is what makes the registry ask before any of this runs, and
    it covers the whole action rather than the destructive members of the enum:
    the parameter is chosen by a voice pipeline that can mishear «в сон» as
    «выключи», so «which member is it really» is exactly the question the
    confirmation exists to answer.
    """

    meta: ClassVar = ActionMeta(
        name="PowerAction",
        category=ActionCategory.SYSTEM,
        title_ru="Питание",
        description_ru="Сон, гибернация, выход из системы, перезагрузка или выключение",
        is_dangerous=True,
        timeout_ms=0,
    )

    class Params(ActionParams):
        operation: PowerOperation = Field(
            title="Что сделать",
            description="Сон, гибернация, выход из системы, перезагрузка или выключение",
            json_schema_extra={
                "choices_ru": {str(op): op.title_ru for op in PowerOperation},
            },
        )
        delay_s: int = Field(
            default=0,
            ge=0,
            le=MAX_DELAY_S,
            title="Задержка",
            description="Через сколько секунд выполнить; ноль — сразу",
            json_schema_extra={"unit_ru": "с"},
        )
        force: bool = Field(
            default=False,
            title="Принудительно",
            description="Не ждать программы с несохранёнными изменениями",
        )
        message: str = Field(
            default="",
            max_length=512,
            title="Предупреждение",
            description="Текст в окне отсчёта Windows при отложенном выключении",
        )

    def run(self, params: Params) -> ActionResult[PowerCall | None]:
        request = PowerRequest(
            operation=params.operation,
            delay_s=params.delay_s,
            force=params.force,
            message=params.message,
        )
        _log.info("питание: %s", request.summary_ru)
        return perform(request)


@register
class LockWorkstation(Action):
    """Lock the screen, the way Win+L does.

    Not ``is_dangerous``: nothing is lost, and a confirmation prompt in front of
    «заблокируй компьютер» would defeat the point of saying it while walking away
    from the desk.
    """

    meta: ClassVar = ActionMeta(
        name="LockWorkstation",
        category=ActionCategory.SYSTEM,
        title_ru="Заблокировать компьютер",
        description_ru="Экран блокировки, как по Win+L",
        timeout_ms=5000,
    )

    def run(self, params: ActionParams) -> ActionResult[None]:
        del params
        get_power_backend().lock()
        return ActionResult.done("Блокирую компьютер.")


@register
class CancelPowerAction(Action):
    """Take back a scheduled shutdown, reboot or sleep.

    Registered as an action rather than left as a function so «отмени
    выключение» is a command like any other and lands in the same history and
    audit trail as the shutdown it cancels.
    """

    meta: ClassVar = ActionMeta(
        name="CancelPowerAction",
        category=ActionCategory.SYSTEM,
        title_ru="Отменить выключение",
        description_ru="Снять запланированное выключение, перезагрузку или сон",
        timeout_ms=5000,
    )

    def run(self, params: ActionParams) -> ActionResult[None]:
        del params
        cancelled = cancel_pending()
        if not cancelled:
            return ActionResult.failed(
                "Нечего отменять — выключение не запланировано.",
                detail="no pending power operation",
            )
        return ActionResult.done(cancelled)
