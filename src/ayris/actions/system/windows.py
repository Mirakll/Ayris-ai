"""Windows: finding them, focusing them, and moving them around.

Three actions live here — :class:`ListWindows`, :class:`FocusWindow` and
:class:`WindowState` — and one piece of Windows folklore.

**The foreground lock.** ``SetForegroundWindow`` does not fail loudly when it is
refused; it returns ``FALSE`` and nothing happens. Windows only grants the right
to raise a window to the process the user last interacted with, and a voice
command is by definition not that process. The workaround has four steps, tried in
order and each one uglier than the last: attach our input queue to the thread that
currently owns the foreground, so the shell thinks the request comes from there;
tap Alt, which ends the shell's «no activation» window; ask for the z-order change
alone; and finally admit out loud that it did not work, which is the one thing this
code must never skip — a silent failure here looks to the user like the assistant
ignored them.

**The interesting half is a pure function of a fake.** :class:`WindowBackend` is
the whole surface this module needs from :mod:`ayris.utils.winapi`: ten small
methods. Everything else — which window a phrase means, which ``ShowWindow``
constant «сверни» is, where the left half of a monitor is, in what order the
foreground workaround escalates — is written against that protocol and therefore
tested on Linux, with assertions on the exact handle and the exact constant handed
down. The real implementation is a thin adapter, and the Windows runner in CI is
where it is exercised.

**What counts as a window.** ``EnumWindows`` hands back several hundred handles on
an idle machine: message-only windows, tooltips, IME helpers, the invisible shells
of Store apps parked on another virtual desktop. A window makes the list only if
it is visible, has a caption, is not a tool window, is not owned by another window
and is not cloaked by the compositor. That last check is the one that keeps
«переключись на телеграм» from focusing a window sitting on virtual desktop 3.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from pydantic import Field, model_validator

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.normalize import fold_letters
from ayris.utils import winapi
from ayris.utils.logger import get_logger
from ayris.utils.winapi import Rect

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "FocusOutcome",
    "FocusStep",
    "FocusWindow",
    "ListWindows",
    "SnapSide",
    "WinApiWindows",
    "WindowBackend",
    "WindowCommand",
    "WindowNotFound",
    "WindowPlacement",
    "WindowQuery",
    "WindowRecord",
    "WindowState",
    "focus_window",
    "get_window_backend",
    "list_windows",
    "select_window",
    "set_window_backend",
    "snap_rect",
]

_log = get_logger(__name__)

#: Longest window title kept in a record. Some applications put a whole document
#: path in the caption, and the title is read out loud.
MAX_TITLE: Final = 200

#: Most windows one :class:`ListWindows` call will report.
MAX_LISTED: Final = 100


class WindowNotFound(ActionError):
    """No window on this desktop matches what was asked for."""

    default_user_message = "Не нашла такое окно."


class WindowPlacement(StrEnum):
    """How a window sits right now."""

    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"

    @property
    def title_ru(self) -> str:
        return _PLACEMENT_TITLES[self]


_PLACEMENT_TITLES: Final[dict[WindowPlacement, str]] = {
    WindowPlacement.NORMAL: "обычное",
    WindowPlacement.MINIMIZED: "свёрнуто",
    WindowPlacement.MAXIMIZED: "развёрнуто",
}


class SnapSide(StrEnum):
    """Half of the work area a window can be snapped to."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


class WindowCommand(StrEnum):
    """What :class:`WindowState` was asked to do.

    Snapping is computed from the monitor's work area rather than emulated with
    Win+Left: the hotkey depends on Snap Assist being enabled, opens the layout
    picker on Windows 11 and cannot be aimed at a window that is not in front.
    Arithmetic on the work area has none of those problems and lands the window on
    the exact pixel.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    RESTORE = "restore"
    SNAP_LEFT = "snap_left"
    SNAP_RIGHT = "snap_right"
    SNAP_TOP = "snap_top"
    SNAP_BOTTOM = "snap_bottom"

    @property
    def title_ru(self) -> str:
        return _COMMAND_TITLES[self]

    @property
    def snap_side(self) -> SnapSide | None:
        """Which half this command snaps to, or ``None`` for the plain three."""
        return _SNAP_SIDES.get(self)

    @property
    def show_command(self) -> int:
        """The ``ShowWindow`` constant this command needs.

        Snapping restores first: a maximised window ignores ``SetWindowPos``, and a
        minimised one has no meaningful position to change.
        """
        if self is WindowCommand.MINIMIZE:
            return winapi.SW_MINIMIZE
        if self is WindowCommand.MAXIMIZE:
            return winapi.SW_MAXIMIZE
        return winapi.SW_RESTORE

    @property
    def done_ru(self) -> str:
        """Past tense for the spoken confirmation: «Свернула окно …»."""
        return _COMMAND_DONE[self]


_COMMAND_TITLES: Final[dict[WindowCommand, str]] = {
    WindowCommand.MINIMIZE: "Свернуть",
    WindowCommand.MAXIMIZE: "Развернуть",
    WindowCommand.RESTORE: "Восстановить",
    WindowCommand.SNAP_LEFT: "Прижать влево",
    WindowCommand.SNAP_RIGHT: "Прижать вправо",
    WindowCommand.SNAP_TOP: "Прижать вверх",
    WindowCommand.SNAP_BOTTOM: "Прижать вниз",
}

_COMMAND_DONE: Final[dict[WindowCommand, str]] = {
    WindowCommand.MINIMIZE: "Свернула",
    WindowCommand.MAXIMIZE: "Развернула",
    WindowCommand.RESTORE: "Восстановила",
    WindowCommand.SNAP_LEFT: "Прижала влево",
    WindowCommand.SNAP_RIGHT: "Прижала вправо",
    WindowCommand.SNAP_TOP: "Прижала вверх",
    WindowCommand.SNAP_BOTTOM: "Прижала вниз",
}

_SNAP_SIDES: Final[dict[WindowCommand, SnapSide]] = {
    WindowCommand.SNAP_LEFT: SnapSide.LEFT,
    WindowCommand.SNAP_RIGHT: SnapSide.RIGHT,
    WindowCommand.SNAP_TOP: SnapSide.TOP,
    WindowCommand.SNAP_BOTTOM: SnapSide.BOTTOM,
}


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """One window, described well enough to talk about and to act on.

    A record, not a string: :class:`ListWindows` feeds the macro editor and the
    LLM's tool calls, and both need the process and the handle apart from the
    caption. ``monitor`` is 1-based in the order the settings window numbers
    displays, with the primary one first.
    """

    hwnd: int
    title: str = ""
    class_name: str = ""
    process: str = ""
    pid: int = 0
    rect: Rect = field(default_factory=Rect)
    placement: WindowPlacement = WindowPlacement.NORMAL
    monitor: int = 0
    foreground: bool = False

    @property
    def process_stem(self) -> str:
        """Process name without ``.exe``, which is how people say it."""
        return Path(self.process).stem.casefold()

    @property
    def label(self) -> str:
        """Caption if there is one, process name otherwise. For speaking aloud."""
        return self.title or self.process or f"окно {self.hwnd}"

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for :attr:`ActionResult.data` and the audit trail."""
        return {
            "hwnd": self.hwnd,
            "title": self.title,
            "class_name": self.class_name,
            "process": self.process,
            "pid": self.pid,
            "monitor": self.monitor,
            "placement": str(self.placement),
            "foreground": self.foreground,
            "rect": self.rect.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class WindowQuery:
    """Which windows the user meant, as filters that can be combined.

    All of it is folded through :func:`~ayris.nlu.normalize.fold_letters`, so «ТГ»
    finds «тг» and «Ё» finds «е» — a spoken title never arrives in the case the
    application wrote it in.
    """

    title: str = ""
    regex: bool = False
    class_name: str = ""
    process: str = ""
    monitor: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether nothing was asked for, i.e. the foreground window is meant."""
        return not (self.title or self.class_name or self.process or self.monitor)

    @property
    def described_ru(self) -> str:
        """What was searched for, for an error message the user can act on."""
        parts = [part for part in (self.title, self.class_name, self.process) if part]
        return ", ".join(parts) if parts else "активное окно"

    def matches(self, record: WindowRecord) -> bool:
        """Whether one window satisfies every filter that was set."""
        if self.monitor and record.monitor != self.monitor:
            return False
        if self.process and not self._process_matches(record):
            return False
        if self.class_name and fold_letters(self.class_name) not in fold_letters(record.class_name):
            return False
        if not self.title:
            return True
        if self.regex:
            return self._regex_matches(record.title)
        return fold_letters(self.title) in fold_letters(record.title)

    def _process_matches(self, record: WindowRecord) -> bool:
        """«хром» does not name a process; ``chrome`` and ``chrome.exe`` both do."""
        wanted = fold_letters(self.process).removesuffix(".exe")
        return wanted == record.process_stem or wanted in record.process_stem

    def _regex_matches(self, title: str) -> bool:
        """A broken pattern is the user's typo, not a crash.

        Raises:
            ActionError: the regular expression does not compile. Said out loud,
                because the alternative is a command that silently matches nothing.
        """
        try:
            pattern = re.compile(self.title, re.IGNORECASE)
        except re.error as exc:
            raise ActionError(
                f"invalid window title pattern {self.title!r}: {exc}",
                user_message=f"Не поняла шаблон поиска окна: {self.title}",
            ) from exc
        return pattern.search(title) is not None


class FocusStep(StrEnum):
    """Which rung of the foreground-lock ladder finally worked."""

    ALREADY = "already"
    DIRECT = "direct"
    ATTACHED = "attached"
    ALT_TRICK = "alt_trick"
    RAISED = "raised"
    FAILED = "failed"

    @property
    def ok(self) -> bool:
        """Whether the window ended up in front."""
        return self is not FocusStep.FAILED


@dataclass(frozen=True, slots=True)
class FocusOutcome:
    """What happened while trying to bring a window to the front."""

    hwnd: int
    step: FocusStep
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.step.ok


class WindowBackend(Protocol):
    """Everything this module needs from Windows. Faked wholesale in the tests."""

    def list_windows(self) -> list[WindowRecord]:
        """Every window worth showing a user, front-most first."""
        ...

    def describe(self, hwnd: int) -> WindowRecord | None:
        """One window, or ``None`` when the handle is stale."""
        ...

    def foreground(self) -> int:
        """Handle of the window that has the focus, or ``0``."""
        ...

    def show(self, hwnd: int, command: int) -> bool:
        """``ShowWindow``."""
        ...

    def set_foreground(self, hwnd: int) -> bool:
        """``SetForegroundWindow``, which is allowed to refuse."""
        ...

    def thread_of(self, hwnd: int) -> int:
        """Id of the thread owning a window, for ``AttachThreadInput``."""
        ...

    def attach(self, thread_id: int, *, attach: bool) -> bool:
        """Attach our input queue to another thread, or detach it."""
        ...

    def tap_alt(self) -> None:
        """Press and release Alt — the trick that ends the activation block."""
        ...

    def raise_window(self, hwnd: int) -> bool:
        """Change the z-order without asking for the focus."""
        ...

    def move(self, hwnd: int, rect: Rect) -> None:
        """Move and resize a window."""
        ...

    def work_area(self, hwnd: int) -> Rect:
        """Usable area of the display the window is on, taskbar excluded."""
        ...


class WinApiWindows:
    """The real backend: :class:`WindowBackend` on top of :mod:`ayris.utils.winapi`.

    Deliberately without decisions of its own. The one thing it does beyond
    forwarding is deciding which handles are worth returning at all, because that
    needs four WinAPI reads per window and doing it anywhere else would mean
    crossing the boundary four hundred times per listing.
    """

    def list_windows(self) -> list[WindowRecord]:
        monitors = self._monitor_indexes()
        processes: dict[int, str] = {}
        active = winapi.foreground_window()
        found: list[WindowRecord] = []
        for hwnd in winapi.enum_windows():
            if not self._is_listable(hwnd):
                continue
            found.append(self._describe(hwnd, monitors, processes, active))
        return found

    def describe(self, hwnd: int) -> WindowRecord | None:
        if not hwnd or not winapi.is_window(hwnd):
            return None
        return self._describe(hwnd, self._monitor_indexes(), {}, winapi.foreground_window())

    def foreground(self) -> int:
        return winapi.foreground_window()

    def show(self, hwnd: int, command: int) -> bool:
        return winapi.show_window(hwnd, command)

    def set_foreground(self, hwnd: int) -> bool:
        return winapi.set_foreground_window(hwnd)

    def thread_of(self, hwnd: int) -> int:
        return winapi.window_thread_id(hwnd)

    def attach(self, thread_id: int, *, attach: bool) -> bool:
        if not thread_id:
            return False
        return winapi.attach_thread_input(thread_id, attach=attach)

    def tap_alt(self) -> None:
        winapi.press_chord([winapi.VK_MENU])

    def raise_window(self, hwnd: int) -> bool:
        raised = winapi.bring_window_to_top(hwnd)
        switched = winapi.switch_to_this_window(hwnd)
        return raised or switched

    def move(self, hwnd: int, rect: Rect) -> None:
        winapi.set_window_position(hwnd, rect)

    def work_area(self, hwnd: int) -> Rect:
        monitor = winapi.monitor_from_window(hwnd)
        if not monitor:
            raise WindowNotFound(
                f"window {hwnd} is not on any monitor",
                user_message="Не поняла, на каком экране это окно.",
            )
        return winapi.monitor_info(monitor).work

    def _is_listable(self, hwnd: int) -> bool:
        """Whether a handle from ``EnumWindows`` describes a window a user sees."""
        if not winapi.is_window_visible(hwnd):
            return False
        if not winapi.window_title(hwnd):
            return False
        styles = winapi.window_ex_style(hwnd)
        app_window = bool(styles & winapi.WS_EX_APPWINDOW)
        if styles & winapi.WS_EX_TOOLWINDOW and not app_window:
            return False
        if winapi.window_owner(hwnd) and not app_window:
            return False
        return not winapi.is_cloaked(hwnd)

    def _describe(
        self,
        hwnd: int,
        monitors: dict[int, int],
        processes: dict[int, str],
        active: int,
    ) -> WindowRecord:
        pid = winapi.window_pid(hwnd)
        if pid not in processes:
            processes[pid] = winapi.process_image_name(pid)
        placement = WindowPlacement.NORMAL
        if winapi.is_iconic(hwnd):
            placement = WindowPlacement.MINIMIZED
        elif winapi.is_zoomed(hwnd):
            placement = WindowPlacement.MAXIMIZED
        return WindowRecord(
            hwnd=hwnd,
            title=winapi.window_title(hwnd)[:MAX_TITLE],
            class_name=winapi.window_class_name(hwnd),
            process=processes[pid],
            pid=pid,
            rect=winapi.window_rect(hwnd),
            placement=placement,
            monitor=monitors.get(winapi.monitor_from_window(hwnd), 0),
            foreground=hwnd == active,
        )

    def _monitor_indexes(self) -> dict[int, int]:
        """Monitor handle to 1-based number, the primary display first."""
        found = [winapi.monitor_info(handle) for handle in winapi.enum_display_monitors()]
        found.sort(key=lambda info: (not info.primary, info.rect.left, info.rect.top))
        return {info.handle: number for number, info in enumerate(found, start=1)}


_backend: WindowBackend | None = None


def get_window_backend() -> WindowBackend:
    """The backend in force. Real WinAPI unless a test replaced it.

    Raises:
        ActionUnavailable: this is not Windows and nothing was injected. Said in
            Russian, because a macro shared between machines can reach here.
    """
    if _backend is not None:
        return _backend
    if sys.platform != "win32":
        raise ActionUnavailable(
            "window actions require Windows",
            user_message="Управление окнами работает только в Windows.",
        )
    return WinApiWindows()


def set_window_backend(backend: WindowBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    _backend = backend


def list_windows(
    query: WindowQuery | None = None,
    *,
    backend: WindowBackend | None = None,
    limit: int = MAX_LISTED,
) -> list[WindowRecord]:
    """Windows matching ``query``, front-most first."""
    active = backend if backend is not None else get_window_backend()
    found = active.list_windows()
    if query is not None and not query.is_empty:
        found = [record for record in found if query.matches(record)]
    return found[: max(1, limit)]


def select_window(
    records: Iterable[WindowRecord],
    query: WindowQuery,
) -> WindowRecord:
    """The one window a phrase meant.

    The first match wins, and the caller hands records in z-order, so «переключись
    на браузер» with three Chrome windows open picks the one that was in front
    last. That is the only tie-break that matches what a person expects.

    Raises:
        WindowNotFound: nothing matched.
    """
    for record in records:
        if query.matches(record):
            return record
    raise WindowNotFound(
        f"no window matches {query.described_ru!r}",
        user_message=f"Не нашла окно «{query.described_ru}».",
    )


def focus_window(
    record: WindowRecord,
    *,
    backend: WindowBackend | None = None,
) -> FocusOutcome:
    """Bring a window to the front, escalating until Windows allows it.

    Returns the outcome instead of raising: «не смогла переключиться» is a result
    the user should hear, and the action turns it into exactly that. Every rung is
    verified by reading the foreground window back — ``SetForegroundWindow``
    returning ``TRUE`` is not proof that anything moved.
    """
    active = backend if backend is not None else get_window_backend()
    hwnd = record.hwnd
    if active.foreground() == hwnd and record.placement is not WindowPlacement.MINIMIZED:
        return FocusOutcome(hwnd, FocusStep.ALREADY, attempts=0)

    if record.placement is WindowPlacement.MINIMIZED:
        active.show(hwnd, winapi.SW_RESTORE)

    attempts = 1
    if _took_foreground(active, hwnd):
        return FocusOutcome(hwnd, FocusStep.DIRECT, attempts=attempts)

    # Rung two: borrow the input queue of whoever holds the foreground. Detached in
    # a finally — a stale attachment ties two input queues together for good.
    thread_id = active.thread_of(active.foreground())
    attached = active.attach(thread_id, attach=True)
    try:
        attempts += 1
        if _took_foreground(active, hwnd):
            return FocusOutcome(hwnd, FocusStep.ATTACHED, attempts=attempts)
    finally:
        if attached:
            active.attach(thread_id, attach=False)

    # Rung three: a lone Alt tap makes the shell believe the user just acted.
    active.tap_alt()
    attempts += 1
    if _took_foreground(active, hwnd):
        return FocusOutcome(hwnd, FocusStep.ALT_TRICK, attempts=attempts)

    # Rung four: raise it without the focus. The window is at least visible, and
    # one click gives it the keyboard.
    active.raise_window(hwnd)
    attempts += 1
    if active.foreground() == hwnd:
        return FocusOutcome(hwnd, FocusStep.RAISED, attempts=attempts)

    _log.warning(
        "Windows отказала в переднем плане для окна %s «%s» после %d попыток",
        hwnd,
        record.label,
        attempts,
    )
    return FocusOutcome(hwnd, FocusStep.FAILED, attempts=attempts)


def _took_foreground(backend: WindowBackend, hwnd: int) -> bool:
    """Ask for the foreground and check whether we actually got it."""
    backend.set_foreground(hwnd)
    return backend.foreground() == hwnd


def snap_rect(work: Rect, side: SnapSide) -> Rect:
    """Half of a work area, rounded the way Windows rounds it.

    The left half gets the extra pixel of an odd width, which is what Snap does,
    so a window snapped left and one snapped right meet exactly and leave no gap.
    """
    half_width = (work.width + 1) // 2
    half_height = (work.height + 1) // 2
    if side is SnapSide.LEFT:
        return Rect(work.left, work.top, work.left + half_width, work.bottom)
    if side is SnapSide.RIGHT:
        return Rect(work.left + half_width, work.top, work.right, work.bottom)
    if side is SnapSide.TOP:
        return Rect(work.left, work.top, work.right, work.top + half_height)
    return Rect(work.left, work.top + half_height, work.right, work.bottom)


def _windows_ru(count: int) -> str:
    """«1 окно», «2 окна», «5 окон» — said aloud, so it has to decline."""
    tail = count % 100
    if 11 <= tail <= 14:
        return f"{count} окон"
    last = count % 10
    if last == 1:
        return f"{count} окно"
    if 2 <= last <= 4:
        return f"{count} окна"
    return f"{count} окон"


def _resolve_target(
    query: WindowQuery,
    backend: WindowBackend,
) -> WindowRecord:
    """The window an action was aimed at: the match, or the foreground one.

    An empty query means «this window», which is what «сверни» on its own means.

    Raises:
        WindowNotFound: nothing matched, or there is no foreground window at all.
    """
    if query.is_empty:
        record = backend.describe(backend.foreground())
        if record is None:
            raise WindowNotFound(
                "no foreground window to act on",
                user_message="Не вижу активного окна.",
            )
        return record
    return select_window(backend.list_windows(), query)


class _WindowSelector(ActionParams):
    """Parameters shared by the two actions that aim at one window."""

    title: str = Field(
        default="",
        max_length=200,
        description="Часть заголовка окна",
    )
    regex: bool = Field(
        default=False,
        description="Считать заголовок регулярным выражением",
    )
    class_name: str = Field(
        default="",
        max_length=120,
        description="Класс окна, например Chrome_WidgetWin_1",
    )
    process: str = Field(
        default="",
        max_length=120,
        description="Имя процесса, например chrome.exe",
    )
    monitor: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Номер монитора, 0 — любой",
    )

    def to_query(self) -> WindowQuery:
        return WindowQuery(
            title=self.title.strip(),
            regex=self.regex,
            class_name=self.class_name.strip(),
            process=self.process.strip(),
            monitor=self.monitor,
        )


@register
class ListWindows(Action):
    """Open windows, as records rather than as a sentence.

    The result value is a list of :class:`WindowRecord`, and ``data`` carries the
    same list in JSON. That is what lets a macro loop over windows and an LLM tool
    call reason about them; a formatted string would have to be parsed back.
    """

    meta: ClassVar = ActionMeta(
        name="ListWindows",
        category=ActionCategory.WINDOWS,
        title_ru="Список окон",
        description_ru="Перечислить открытые окна, при желании отфильтровав их",
        timeout_ms=5_000,
    )

    class Params(_WindowSelector):
        limit: int = Field(
            default=20,
            ge=1,
            le=MAX_LISTED,
            description="Сколько окон вернуть",
        )

    def run(self, params: Params) -> ActionResult[list[WindowRecord]]:
        backend = get_window_backend()
        found = list_windows(params.to_query(), backend=backend, limit=params.limit)
        return ActionResult.done(
            f"Нашла {_windows_ru(len(found))}." if found else "Открытых окон не нашла.",
            value=found,
            data={
                "count": len(found),
                "windows": [record.as_dict() for record in found],
            },
        )


@register
class FocusWindow(Action):
    """Bring a window to the front, and say so when Windows refuses."""

    meta: ClassVar = ActionMeta(
        name="FocusWindow",
        category=ActionCategory.WINDOWS,
        title_ru="Переключиться на окно",
        description_ru="Сделать окно активным, обходя запрет на смену переднего плана",
        timeout_ms=5_000,
    )

    class Params(_WindowSelector):
        @model_validator(mode="after")
        def _something_to_find(self) -> FocusWindow.Params:
            """Focusing «any window» is not a request anyone means."""
            if self.to_query().is_empty:
                raise ValueError("укажите заголовок, класс или процесс окна")
            return self

    def run(self, params: Params) -> ActionResult[WindowRecord]:
        backend = get_window_backend()
        record = select_window(backend.list_windows(), params.to_query())
        outcome = focus_window(record, backend=backend)
        data = {"window": record.as_dict(), "step": str(outcome.step)}
        if not outcome.ok:
            return ActionResult.failed(
                f"Windows не дала переключиться на «{record.label}».",
                detail=f"foreground denied after {outcome.attempts} attempts",
                value=record,
                data=data,
            )
        return ActionResult.done(
            f"Переключилась на «{record.label}».",
            value=record,
            data=data,
        )


@register
class WindowState(Action):
    """Minimise, maximise, restore or snap a window to half of its display."""

    meta: ClassVar = ActionMeta(
        name="WindowState",
        category=ActionCategory.WINDOWS,
        title_ru="Состояние окна",
        description_ru="Свернуть, развернуть, восстановить или прижать окно к краю экрана",
        timeout_ms=5_000,
    )

    class Params(_WindowSelector):
        command: WindowCommand = Field(
            default=WindowCommand.MINIMIZE,
            description="Что сделать с окном",
            json_schema_extra={
                "choices_ru": {str(command): command.title_ru for command in WindowCommand}
            },
        )

    def run(self, params: Params) -> ActionResult[WindowRecord]:
        backend = get_window_backend()
        record = _resolve_target(params.to_query(), backend)
        command = params.command
        backend.show(record.hwnd, command.show_command)

        side = command.snap_side
        target: Rect | None = None
        if side is not None:
            target = snap_rect(backend.work_area(record.hwnd), side)
            backend.move(record.hwnd, target)

        data: dict[str, Any] = {
            "window": record.as_dict(),
            "command": str(command),
            "show_command": command.show_command,
        }
        if target is not None:
            data["rect"] = target.as_dict()
        return ActionResult.done(
            f"{command.done_ru} окно «{record.label}».",
            value=record,
            data=data,
        )


def describe_windows(records: Sequence[WindowRecord]) -> str:
    """One line per window, for a log entry or a debug overlay."""
    return "\n".join(
        f"{record.hwnd:#010x} {record.process or '?'} [{record.placement}] {record.title}"
        for record in records
    )
