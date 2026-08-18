"""Task 20: окна, программы и рабочие столы — всё, что можно проверить без Windows.

Three modules are covered here — :mod:`ayris.actions.system.windows`,
:mod:`ayris.actions.system.apps` and :mod:`ayris.actions.system.desktops` — and
they share one design decision, which is what makes the tests possible at all:
each of them talks to Windows through a narrow protocol
(:class:`~…windows.WindowBackend`, :class:`~…apps.AppLauncher`,
:class:`~…desktops.DesktopBackend`) and keeps every decision on this side of it.
So the fakes below are the whole operating system as far as these tests are
concerned, and the assertions are about exact handles, exact ``ShowWindow``
constants and exact Russian phrases rather than about «it called something».

Three rules carry the most weight and each is asserted directly.

*The foreground ladder escalates in order and admits defeat out loud.* Windows
refuses ``SetForegroundWindow`` to a process the user did not just touch, and it
refuses it silently. Four rungs are tried, each verified by reading the foreground
window back, and the last resort is a warning in the log plus a result the user
hears — a silent failure looks like the assistant ignored the command.

*Closing is not killing.* ``WM_CLOSE`` goes to every window the program owns, and
``TerminateProcess`` is reached only when the caller passed ``force`` and only
after the grace period. The three outcomes have three different phrasings,
because «Закрыла Word» said over an unsaved document is a lie with consequences.

*A snapped pair leaves no gap.* The left half gets the odd pixel, exactly as Snap
does, so ``snap_left`` and ``snap_right`` meet on one column.

Groups:

* :class:`TestWindowRecord` — the process stem, the spoken label, the JSON form.
* :class:`TestWindowQuery` — every filter, folded through the normaliser.
* :class:`TestSelectWindow` — z-order decides, and what is said when nothing matches.
* :class:`TestListWindowsFunction` — filtering, the limit and its floor.
* :class:`TestFocusLadder` — all six outcomes, in order, on a fake.
* :class:`TestSnapRect` — four sides, odd sizes, halves that meet.
* :class:`TestWindowBackendSeam` — the injection point and the off-Windows refusal.
* :class:`TestListWindowsAction` — records rather than a sentence, and «окон/окна/окно».
* :class:`TestFocusWindowAction` — the empty request is refused, the denial is spoken.
* :class:`TestWindowStateAction` — the constants handed to Windows, and the snapped rect.
* :class:`TestDescribeWindows` — the debug line.
* :class:`TestLaunchKind` — three launch paths, told apart by the target.
* :class:`TestLaunchRequest` — what reaches the shell, including a Store moniker.
* :class:`TestRunApp` — the pid, the launch counter, the refusal to guess.
* :class:`TestAppWindows` — by process, and by caption for a Store app.
* :class:`TestCloseApp` — polite close, forced kill, and the three phrasings.
* :class:`TestLauncherSeam` — the injection point and the off-Windows refusal.
* :class:`TestGuidFromBytes` — the mixed-endian blob the shell writes.
* :class:`TestParseDesktopIds` — a torn read costs a desktop, not the command.
* :class:`TestDesktopState` — numbers, names, and «do we know where we are».
* :class:`TestSwitchPlan` — which way and how many taps.
* :class:`TestSwitchDirection` — the two shell chords and which way each counts.
* :class:`TestDesktopBackendSeam` — the injection point and the off-Windows refusal.
* :class:`TestSwitchDesktopAction` — one of three parameters, and the two edges.
* :class:`TestSchemas` — six actions as the macro editor sees them.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from ayris.actions.base import FieldKind, build_schema
from ayris.actions.system.app_index import (
    STORE_PREFIX,
    AppCandidate,
    AppIndex,
    AppNotFound,
    IndexedApp,
    IndexSource,
    set_app_index,
)
from ayris.actions.system.apps import (
    DEFAULT_CLOSE_TIMEOUT_MS,
    SHELL_BINARY,
    CloseApp,
    CloseOutcome,
    LaunchKind,
    LaunchRequest,
    RunApp,
    WinApiLauncher,
    app_windows,
    get_launcher,
    set_launcher,
)
from ayris.actions.system.desktops import (
    MAX_TAPS,
    DesktopInfo,
    DesktopState,
    DesktopUnavailable,
    SwitchDesktop,
    SwitchDirection,
    get_desktop_backend,
    guid_from_bytes,
    iter_desktop_labels,
    parse_desktop_ids,
    set_desktop_backend,
    switch_plan,
)
from ayris.actions.system.windows import (
    MAX_LISTED,
    FocusOutcome,
    FocusStep,
    FocusWindow,
    ListWindows,
    SnapSide,
    WinApiWindows,
    WindowCommand,
    WindowNotFound,
    WindowPlacement,
    WindowQuery,
    WindowRecord,
    WindowState,
    describe_windows,
    focus_window,
    get_window_backend,
    list_windows,
    select_window,
    set_window_backend,
    snap_rect,
)
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.nlu.apps import DEFAULT_ALIAS_THRESHOLD, load_apps
from ayris.utils import winapi
from ayris.utils.logger import ROOT_LOGGER_NAME
from ayris.utils.winapi import Rect

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from ayris.actions.result import ActionResult

pytestmark = pytest.mark.unit

#: The shipped dictionary, so the resolver under test is the one the application
#: builds rather than a fixture that happens to agree with it.
CATALOG = load_apps()

#: AUMID of «Параметры Windows», read from the dictionary: a rename there should
#: break these tests loudly instead of leaving them passing against a dead id.
SETTINGS_AUMID = next(entry.uwp for entry in CATALOG.apps if entry.id == "settings")

#: A plausible scan: two ordinary programs, a Start-menu shortcut and a Store app.
INDEX_APPS = (
    IndexedApp(
        name="Google Chrome",
        target=r"C:\Program Files\Google\Chrome\chrome.exe",
        source=IndexSource.START_MENU,
        executable="chrome.exe",
        catalog_id="chrome",
    ),
    IndexedApp(
        name="Visual Studio Code",
        target=r"C:\Users\u\AppData\Local\Programs\VS Code\Code.exe",
        source=IndexSource.START_MENU,
        executable="Code.exe",
        arguments="--new-window",
        working_dir=r"C:\work",
        catalog_id="vscode",
    ),
    IndexedApp(
        name="Telegram Desktop",
        target=r"C:\ProgramData\Start Menu\Telegram.lnk",
        source=IndexSource.START_MENU,
        executable="Telegram.exe",
        catalog_id="telegram",
    ),
    IndexedApp(
        name="Параметры",
        target=f"{STORE_PREFIX}{SETTINGS_AUMID}",
        source=IndexSource.UWP,
        aumid=SETTINGS_AUMID,
        catalog_id="settings",
    ),
)

CHROME = WindowRecord(
    hwnd=0x1001,
    title="Ayris — Google Chrome",
    class_name="Chrome_WidgetWin_1",
    process="chrome.exe",
    pid=4100,
    rect=Rect(0, 0, 1200, 800),
    monitor=1,
    foreground=True,
)

CHROME_DOCS = WindowRecord(
    hwnd=0x1002,
    title="Документы — Google Chrome",
    class_name="Chrome_WidgetWin_1",
    process="chrome.exe",
    pid=4100,
    rect=Rect(1920, 0, 3840, 1080),
    placement=WindowPlacement.MAXIMIZED,
    monitor=2,
)

CODE = WindowRecord(
    hwnd=0x2001,
    title="test_windows.py — Ayris — Visual Studio Code",
    class_name="Chrome_WidgetWin_1",
    process="Code.exe",
    pid=4200,
    rect=Rect(100, 100, 1400, 900),
    placement=WindowPlacement.MINIMIZED,
    monitor=1,
)

TELEGRAM = WindowRecord(
    hwnd=0x3001,
    title="Telegram",
    class_name="Qt51514QWindowIcon",
    process="Telegram.exe",
    pid=4300,
    rect=Rect(300, 200, 900, 800),
    monitor=1,
)

#: A Store window: the process is the shared frame host, so only the caption
#: identifies it — which is the case :func:`app_windows` has a second branch for.
SETTINGS = WindowRecord(
    hwnd=0x4001,
    title="Параметры Windows",
    class_name="ApplicationFrameWindow",
    process="ApplicationFrameHost.exe",
    pid=4400,
    rect=Rect(400, 300, 1300, 1000),
    monitor=1,
)

#: Front-most first, the order :func:`select_window` relies on.
DESKTOP = (CHROME, CHROME_DOCS, CODE, TELEGRAM, SETTINGS)


class FakeWindows:
    """The whole of Windows, as far as :mod:`…windows` can tell.

    ``grant_after`` is the number of ``SetForegroundWindow`` calls after which the
    request is honoured, which is how the ladder's rungs are addressed: the first
    call is the direct attempt, the second happens with the input queue attached,
    the third after the Alt tap. ``None`` means Windows never gives in.
    """

    def __init__(
        self,
        records: Sequence[WindowRecord] = DESKTOP,
        *,
        foreground_hwnd: int = CHROME.hwnd,
        grant_after: int | None = 1,
        raise_grants: bool = False,
        attach_ok: bool = True,
        work: Rect = Rect(0, 0, 1920, 1040),
    ) -> None:
        self.records = list(records)
        self.work = work
        self.grant_after = grant_after
        self.raise_grants = raise_grants
        self.attach_ok = attach_ok
        self._foreground = foreground_hwnd
        self.calls: list[str] = []
        self.set_foreground_calls = 0
        self.shown: list[tuple[int, int]] = []
        self.moved: list[tuple[int, Rect]] = []
        self.attached: list[tuple[int, bool]] = []
        self.raised: list[int] = []
        self.alt_taps = 0

    def list_windows(self) -> list[WindowRecord]:
        self.calls.append("list_windows")
        return list(self.records)

    def describe(self, hwnd: int) -> WindowRecord | None:
        self.calls.append("describe")
        return next((record for record in self.records if record.hwnd == hwnd), None)

    def foreground(self) -> int:
        return self._foreground

    def show(self, hwnd: int, command: int) -> bool:
        self.calls.append("show")
        self.shown.append((hwnd, command))
        return True

    def set_foreground(self, hwnd: int) -> bool:
        self.calls.append("set_foreground")
        self.set_foreground_calls += 1
        if self.grant_after is not None and self.set_foreground_calls >= self.grant_after:
            self._foreground = hwnd
            return True
        return False

    def thread_of(self, hwnd: int) -> int:
        self.calls.append("thread_of")
        return hwnd * 10

    def attach(self, thread_id: int, *, attach: bool) -> bool:
        self.calls.append("attach" if attach else "detach")
        self.attached.append((thread_id, attach))
        return self.attach_ok

    def tap_alt(self) -> None:
        self.calls.append("tap_alt")
        self.alt_taps += 1

    def raise_window(self, hwnd: int) -> bool:
        self.calls.append("raise_window")
        self.raised.append(hwnd)
        if self.raise_grants:
            self._foreground = hwnd
        return self.raise_grants

    def move(self, hwnd: int, rect: Rect) -> None:
        self.calls.append("move")
        self.moved.append((hwnd, rect))

    def work_area(self, hwnd: int) -> Rect:
        self.calls.append("work_area")
        return self.work


class FakeLauncher:
    """Starting and stopping programs, without starting or stopping anything."""

    def __init__(
        self,
        *,
        pid: int = 4242,
        alive: Sequence[int] = (),
        terminate_ok: bool = True,
    ) -> None:
        self.pid = pid
        self.alive = tuple(alive)
        self.terminate_ok = terminate_ok
        self.requests: list[LaunchRequest] = []
        self.closed: list[int] = []
        self.terminated: list[int] = []
        self.waited: list[tuple[tuple[int, ...], int]] = []

    def launch(self, request: LaunchRequest) -> int:
        self.requests.append(request)
        return self.pid

    def close_window(self, hwnd: int) -> bool:
        self.closed.append(hwnd)
        return True

    def running(self, pid: int) -> bool:
        return pid in self.alive

    def wait_exit(self, pids: Sequence[int], timeout_ms: int) -> tuple[int, ...]:
        self.waited.append((tuple(pids), timeout_ms))
        return tuple(pid for pid in pids if pid in self.alive)

    def terminate(self, pid: int) -> bool:
        self.terminated.append(pid)
        if not self.terminate_ok:
            return False
        self.alive = tuple(other for other in self.alive if other != pid)
        return True


class FakeDesktops:
    """The registry and the keyboard, replaced by a list and a counter."""

    def __init__(self, state: DesktopState, *, moves: bool = True, blind: bool = False) -> None:
        self._state = state
        self.moves = moves
        self.blind = blind
        self.taps: list[tuple[SwitchDirection, int]] = []
        self.reads = 0

    def state(self) -> DesktopState:
        self.reads += 1
        if self.blind:
            raise DesktopUnavailable(
                "registry is unreadable",
                user_message="Не нашла список рабочих столов в реестре.",
            )
        return self._state

    def tap(self, direction: SwitchDirection, times: int) -> None:
        self.taps.append((direction, times))
        if self.moves and self._state.known:
            landed = self._state.current + direction.step * times
            self._state = DesktopState(
                desktops=self._state.desktops,
                current=min(max(landed, 1), self._state.count),
            )

    def supported(self) -> bool:
        return True


def desktop_state(
    count: int = 3, current: int = 1, names: dict[int, str] | None = None
) -> DesktopState:
    """A desktop list with plausible GUIDs and optional user-given names."""
    labels = names or {}
    return DesktopState(
        desktops=tuple(
            DesktopInfo(
                number=number,
                guid=f"{{00000000-0000-0000-0000-0000000000{number:02d}}}",
                name=labels.get(number, ""),
            )
            for number in range(1, count + 1)
        ),
        current=current,
    )


@pytest.fixture
def windows_backend() -> Iterator[FakeWindows]:
    """A fake desktop installed as the process-wide window backend."""
    backend = FakeWindows()
    set_window_backend(backend)
    try:
        yield backend
    finally:
        set_window_backend(None)


@pytest.fixture
def launcher() -> Iterator[FakeLauncher]:
    backend = FakeLauncher()
    set_launcher(backend)
    try:
        yield backend
    finally:
        set_launcher(None)


@pytest.fixture
def index(tmp_path: Path) -> Iterator[AppIndex]:
    """The application index over :data:`INDEX_APPS`, wired to a temporary cache.

    ``entries`` and ``threshold`` are passed explicitly so nothing here reads the
    user's settings, and the clock is frozen so the TTL never fires mid-test.
    """
    built = AppIndex(
        cache_path=tmp_path / "apps_index.json",
        ttl_s=100_000.0,
        scanner=lambda: list(INDEX_APPS),
        entries=CATALOG.apps,
        threshold=DEFAULT_ALIAS_THRESHOLD,
        clock=lambda: 1_000.0,
    )
    built.ensure_ready()
    set_app_index(built)
    try:
        yield built
    finally:
        set_app_index(None)
        built.close()


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` there, so a plain ``caplog`` sees
    nothing once any earlier test in the run has configured logging.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


class TestWindowRecord:
    """A record rather than a caption, because callers need the pieces apart."""

    def test_the_process_stem_drops_the_extension_and_the_case(self) -> None:
        """«переключись на хром» names ``chrome``, not ``chrome.exe``."""
        assert CHROME.process_stem == "chrome"
        assert CODE.process_stem == "code"

    def test_the_stem_drops_a_path_too(self) -> None:
        """A macro may have been written with a full path in the process field.

        Spelled with a forward slash on purpose: ``Path`` only knows about the
        backslash on Windows, and this assertion has to mean the same thing in the
        Linux job of CI as it does in the two Windows ones.
        """
        record = WindowRecord(hwnd=1, process="C:/Program Files/Google/chrome.exe")
        assert record.process_stem == "chrome"

    def test_the_stem_is_folded(self) -> None:
        """``Code.exe`` on disk, ``code`` in a query — the comparison has to agree."""
        assert WindowRecord(hwnd=1, process="Telegram.EXE").process_stem == "telegram"

    def test_no_process_leaves_an_empty_stem(self) -> None:
        assert WindowRecord(hwnd=1).process_stem == ""

    def test_the_label_prefers_the_caption(self) -> None:
        assert CHROME.label == "Ayris — Google Chrome"

    def test_a_window_without_a_caption_is_called_by_its_process(self) -> None:
        record = WindowRecord(hwnd=7, process="explorer.exe")
        assert record.label == "explorer.exe"

    def test_a_nameless_window_is_called_by_its_handle(self) -> None:
        """Said aloud, so «окно» plus a number beats an empty pair of quotes."""
        assert WindowRecord(hwnd=4097).label == "окно 4097"

    def test_the_json_form_carries_every_field(self) -> None:
        """It reaches the audit trail and an LLM tool call, so nothing is dropped."""
        payload = CHROME_DOCS.as_dict()
        assert payload == {
            "hwnd": 0x1002,
            "title": "Документы — Google Chrome",
            "class_name": "Chrome_WidgetWin_1",
            "process": "chrome.exe",
            "pid": 4100,
            "monitor": 2,
            "placement": "maximized",
            "foreground": False,
            "rect": {"left": 1920, "top": 0, "width": 1920, "height": 1080},
        }

    def test_the_placement_is_a_string_in_json(self) -> None:
        """``json.dumps`` of an enum member fails; of ``str(enum)`` it does not."""
        assert CODE.as_dict()["placement"] == "minimized"

    @pytest.mark.parametrize(
        ("placement", "title_ru"),
        [
            (WindowPlacement.NORMAL, "обычное"),
            (WindowPlacement.MINIMIZED, "свёрнуто"),
            (WindowPlacement.MAXIMIZED, "развёрнуто"),
        ],
    )
    def test_every_placement_has_a_russian_title(
        self, placement: WindowPlacement, title_ru: str
    ) -> None:
        assert placement.title_ru == title_ru


class TestWindowQuery:
    """Which windows a phrase meant, as filters that combine."""

    def test_an_empty_query_means_the_active_window(self) -> None:
        assert WindowQuery().is_empty
        assert WindowQuery().described_ru == "активное окно"

    def test_regex_alone_does_not_make_a_query(self) -> None:
        """``regex`` is how to read the title, not something to search for."""
        assert WindowQuery(regex=True).is_empty

    @pytest.mark.parametrize(
        "query",
        [
            WindowQuery(title="chrome"),
            WindowQuery(class_name="Chrome_WidgetWin_1"),
            WindowQuery(process="chrome.exe"),
            WindowQuery(monitor=1),
        ],
    )
    def test_any_single_filter_makes_a_query(self, query: WindowQuery) -> None:
        assert not query.is_empty

    def test_a_title_matches_a_fragment(self) -> None:
        assert WindowQuery(title="Google Chrome").matches(CHROME)
        assert not WindowQuery(title="Firefox").matches(CHROME)

    def test_the_title_is_folded_for_case_and_yo(self) -> None:
        """A spoken caption never arrives in the case the application wrote it in."""
        assert WindowQuery(title="GOOGLE").matches(CHROME)
        assert WindowQuery(title="ПАРАМЕТРЫ").matches(SETTINGS)

    def test_a_class_matches_a_fragment_too(self) -> None:
        assert WindowQuery(class_name="widgetwin").matches(CHROME)
        assert not WindowQuery(class_name="Qt5").matches(CHROME)

    def test_a_process_matches_with_or_without_the_extension(self) -> None:
        """The dictionary stores ``chrome.exe``; a person says «хром»."""
        assert WindowQuery(process="chrome.exe").matches(CHROME)
        assert WindowQuery(process="chrome").matches(CHROME)
        assert WindowQuery(process="CHROME.EXE").matches(CHROME)

    def test_a_process_may_be_a_fragment_of_the_stem(self) -> None:
        assert WindowQuery(process="tele").matches(TELEGRAM)

    def test_a_process_does_not_match_a_different_program(self) -> None:
        assert not WindowQuery(process="firefox").matches(CHROME)

    def test_a_monitor_filter_is_exact(self) -> None:
        """«окна на втором экране» is a number, not a range."""
        assert WindowQuery(monitor=2).matches(CHROME_DOCS)
        assert not WindowQuery(monitor=2).matches(CHROME)

    def test_filters_combine_with_and(self) -> None:
        """«окно хрома с документами» is one window out of two Chrome ones."""
        query = WindowQuery(title="Документы", process="chrome")
        assert query.matches(CHROME_DOCS)
        assert not query.matches(CHROME)

    def test_a_process_filter_alone_matches_both_windows_of_a_program(self) -> None:
        matched = [record for record in DESKTOP if WindowQuery(process="chrome").matches(record)]
        assert matched == [CHROME, CHROME_DOCS]

    def test_a_regex_title_searches_rather_than_anchors(self) -> None:
        assert WindowQuery(title=r"Google\s+Chrome$", regex=True).matches(CHROME)

    def test_a_regex_title_is_case_insensitive(self) -> None:
        assert WindowQuery(title="google chrome", regex=True).matches(CHROME)

    def test_a_regex_that_matches_nothing_simply_does_not_match(self) -> None:
        assert not WindowQuery(title=r"^Firefox", regex=True).matches(CHROME)

    def test_a_broken_pattern_is_said_out_loud(self) -> None:
        """A typo in a macro must not look like «no such window»."""
        with pytest.raises(ActionError) as caught:
            WindowQuery(title="(незакрытая", regex=True).matches(CHROME)
        assert caught.value.user_message == "Не поняла шаблон поиска окна: (незакрытая"

    def test_a_regex_flag_without_a_title_never_compiles_anything(self) -> None:
        """The title branch is skipped entirely, so a monitor query stays cheap."""
        assert WindowQuery(monitor=1, regex=True).matches(CHROME)

    def test_the_description_lists_what_was_searched_for(self) -> None:
        query = WindowQuery(title="Документы", class_name="Chrome_WidgetWin_1", process="chrome")
        assert query.described_ru == "Документы, Chrome_WidgetWin_1, chrome"

    def test_the_description_leaves_the_monitor_out(self) -> None:
        """«Не нашла окно «2»» would be nonsense; the empty case covers it."""
        assert WindowQuery(monitor=3).described_ru == "активное окно"


class TestSelectWindow:
    """One window out of several, decided by z-order."""

    def test_the_front_most_match_wins(self) -> None:
        """Three Chrome windows and «переключись на хром» means the last one used."""
        assert select_window(DESKTOP, WindowQuery(process="chrome")) is CHROME

    def test_the_order_of_the_records_is_the_tie_break(self) -> None:
        reordered = (CHROME_DOCS, CHROME)
        assert select_window(reordered, WindowQuery(process="chrome")) is CHROME_DOCS

    def test_nothing_matching_is_reported_with_what_was_asked_for(self) -> None:
        with pytest.raises(WindowNotFound) as caught:
            select_window(DESKTOP, WindowQuery(title="Fire fox"))
        assert caught.value.user_message == "Не нашла окно «Fire fox»."

    def test_an_empty_list_raises_the_same_way(self) -> None:
        with pytest.raises(WindowNotFound):
            select_window((), WindowQuery(process="chrome"))


class TestListWindowsFunction:
    """The plain function: filter, then cut to the limit."""

    def test_without_a_query_everything_is_returned(self, windows_backend: FakeWindows) -> None:
        assert list_windows(backend=windows_backend) == list(DESKTOP)

    def test_an_empty_query_is_not_a_filter(self, windows_backend: FakeWindows) -> None:
        """``matches`` is never called, so a window with no caption still appears."""
        assert list_windows(WindowQuery(), backend=windows_backend) == list(DESKTOP)

    def test_a_query_filters(self, windows_backend: FakeWindows) -> None:
        found = list_windows(WindowQuery(process="chrome"), backend=windows_backend)
        assert found == [CHROME, CHROME_DOCS]

    def test_the_limit_cuts_from_the_front(self, windows_backend: FakeWindows) -> None:
        assert list_windows(backend=windows_backend, limit=2) == [CHROME, CHROME_DOCS]

    def test_a_limit_of_zero_still_returns_one(self, windows_backend: FakeWindows) -> None:
        """A caller that computed ``0`` wants the front window, not an empty answer."""
        assert list_windows(backend=windows_backend, limit=0) == [CHROME]

    def test_the_default_limit_is_the_module_maximum(self, windows_backend: FakeWindows) -> None:
        assert MAX_LISTED == 100
        assert len(list_windows(backend=windows_backend)) == len(DESKTOP)

    def test_the_installed_backend_is_used_when_none_is_passed(
        self, windows_backend: FakeWindows
    ) -> None:
        assert list_windows() == list(DESKTOP)
        assert "list_windows" in windows_backend.calls


class TestFocusLadder:
    """The four rungs of the foreground lock, and the admission of defeat.

    Every rung is addressed through ``grant_after``: the number of
    ``SetForegroundWindow`` calls after which the fake lets the request through.
    That is the only knob needed, because each rung is exactly one more call.
    """

    def test_a_window_already_in_front_costs_nothing(self) -> None:
        """No ``ShowWindow``, no ``SetForegroundWindow``: «переключись на хром»
        while Chrome is in front must not blink the window."""
        backend = FakeWindows(foreground_hwnd=CHROME.hwnd)
        outcome = focus_window(CHROME, backend=backend)
        assert outcome.step is FocusStep.ALREADY
        assert outcome.attempts == 0
        assert outcome.ok
        assert backend.calls == []

    def test_a_minimized_window_in_front_is_still_restored(self) -> None:
        """Windows keeps the foreground handle on a window it has just iconified,
        so ``ALREADY`` there would leave the user looking at the taskbar."""
        backend = FakeWindows(foreground_hwnd=CODE.hwnd)
        outcome = focus_window(CODE, backend=backend)
        assert outcome.step is FocusStep.DIRECT
        assert backend.shown == [(CODE.hwnd, winapi.SW_RESTORE)]

    def test_a_normal_window_is_not_shown_at_all(self) -> None:
        """``SW_RESTORE`` on a maximised window un-maximises it, which nobody asked for."""
        backend = FakeWindows()
        focus_window(CHROME_DOCS, backend=backend)
        assert backend.shown == []

    def test_the_first_rung_is_a_plain_request(self) -> None:
        backend = FakeWindows(grant_after=1)
        outcome = focus_window(TELEGRAM, backend=backend)
        assert outcome.step is FocusStep.DIRECT
        assert outcome.attempts == 1
        assert backend.calls == ["set_foreground"]

    def test_the_second_rung_attaches_the_input_queue(self) -> None:
        """The thread asked about is the one holding the foreground, not ours."""
        backend = FakeWindows(foreground_hwnd=CHROME.hwnd, grant_after=2)
        outcome = focus_window(TELEGRAM, backend=backend)
        assert outcome.step is FocusStep.ATTACHED
        assert outcome.attempts == 2
        assert backend.attached[0] == (CHROME.hwnd * 10, True)

    def test_the_attachment_is_always_undone(self) -> None:
        """A stale attachment ties two input queues together for the session."""
        backend = FakeWindows(grant_after=2)
        focus_window(TELEGRAM, backend=backend)
        assert backend.attached == [(CHROME.hwnd * 10, True), (CHROME.hwnd * 10, False)]

    def test_a_refused_attachment_is_not_detached(self) -> None:
        """``AttachThreadInput`` failed, so detaching would unhook someone else's."""
        backend = FakeWindows(grant_after=3, attach_ok=False)
        focus_window(TELEGRAM, backend=backend)
        assert backend.attached == [(CHROME.hwnd * 10, True)]

    def test_the_third_rung_taps_alt(self) -> None:
        backend = FakeWindows(grant_after=3)
        outcome = focus_window(TELEGRAM, backend=backend)
        assert outcome.step is FocusStep.ALT_TRICK
        assert outcome.attempts == 3
        assert backend.alt_taps == 1

    def test_the_fourth_rung_only_changes_the_z_order(self) -> None:
        """The window is visible and one click away from the keyboard — worth saying yes to."""
        backend = FakeWindows(grant_after=None, raise_grants=True)
        outcome = focus_window(TELEGRAM, backend=backend)
        assert outcome.step is FocusStep.RAISED
        assert outcome.attempts == 4
        assert backend.raised == [TELEGRAM.hwnd]

    def test_the_rungs_are_tried_in_order_and_only_once_each(self) -> None:
        """The order is the contract: each step is uglier than the one before it."""
        backend = FakeWindows(grant_after=None)
        focus_window(TELEGRAM, backend=backend)
        assert backend.calls == [
            "set_foreground",
            "thread_of",
            "attach",
            "set_foreground",
            "detach",
            "tap_alt",
            "set_foreground",
            "raise_window",
        ]

    def test_a_refusal_is_a_failure_rather_than_an_exception(self) -> None:
        """The action turns it into «Windows не дала переключиться», which the user hears."""
        backend = FakeWindows(grant_after=None)
        outcome = focus_window(TELEGRAM, backend=backend)
        assert outcome.step is FocusStep.FAILED
        assert not outcome.ok
        assert outcome.attempts == 4
        assert outcome.hwnd == TELEGRAM.hwnd

    def test_a_refusal_is_written_to_the_log(self, ayris_log: pytest.LogCaptureFixture) -> None:
        """The one thing this code must never do quietly."""
        focus_window(TELEGRAM, backend=FakeWindows(grant_after=None))
        assert "отказала в переднем плане" in ayris_log.text
        assert TELEGRAM.label in ayris_log.text

    def test_a_successful_rung_leaves_nothing_in_the_log(
        self, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        focus_window(TELEGRAM, backend=FakeWindows(grant_after=1))
        assert "отказала" not in ayris_log.text

    def test_a_true_return_from_windows_is_not_believed(self) -> None:
        """``SetForegroundWindow`` returning ``TRUE`` while nothing moved is the
        whole reason every rung reads the foreground window back."""

        class Liar(FakeWindows):
            def set_foreground(self, hwnd: int) -> bool:
                self.calls.append("set_foreground")
                self.set_foreground_calls += 1
                return True

        backend = Liar()
        assert focus_window(TELEGRAM, backend=backend).step is FocusStep.FAILED
        assert backend.set_foreground_calls == 3

    def test_the_installed_backend_is_used_when_none_is_passed(
        self, windows_backend: FakeWindows
    ) -> None:
        assert focus_window(TELEGRAM).step is FocusStep.DIRECT
        assert windows_backend.set_foreground_calls == 1

    @pytest.mark.parametrize(
        ("step", "ok"),
        [
            (FocusStep.ALREADY, True),
            (FocusStep.DIRECT, True),
            (FocusStep.ATTACHED, True),
            (FocusStep.ALT_TRICK, True),
            (FocusStep.RAISED, True),
            (FocusStep.FAILED, False),
        ],
    )
    def test_only_the_last_step_counts_as_a_failure(self, step: FocusStep, ok: bool) -> None:
        assert step.ok is ok
        assert FocusOutcome(hwnd=1, step=step).ok is ok


class TestSnapRect:
    """Halves of a work area, computed rather than emulated with Win+Left."""

    WORK = Rect(0, 0, 1920, 1040)

    def test_the_left_half(self) -> None:
        assert snap_rect(self.WORK, SnapSide.LEFT) == Rect(0, 0, 960, 1040)

    def test_the_right_half(self) -> None:
        assert snap_rect(self.WORK, SnapSide.RIGHT) == Rect(960, 0, 1920, 1040)

    def test_the_top_half(self) -> None:
        assert snap_rect(self.WORK, SnapSide.TOP) == Rect(0, 0, 1920, 520)

    def test_the_bottom_half(self) -> None:
        assert snap_rect(self.WORK, SnapSide.BOTTOM) == Rect(0, 520, 1920, 1040)

    def test_the_odd_pixel_goes_left(self) -> None:
        """Exactly what Snap does, and the reason the two halves meet."""
        work = Rect(0, 0, 1367, 767)
        assert snap_rect(work, SnapSide.LEFT).width == 684
        assert snap_rect(work, SnapSide.RIGHT).width == 683

    def test_the_halves_meet_without_a_gap(self) -> None:
        work = Rect(0, 0, 1367, 767)
        assert snap_rect(work, SnapSide.LEFT).right == snap_rect(work, SnapSide.RIGHT).left
        assert snap_rect(work, SnapSide.TOP).bottom == snap_rect(work, SnapSide.BOTTOM).top

    def test_a_work_area_that_does_not_start_at_zero(self) -> None:
        """A second monitor to the left of the primary one has negative coordinates,
        and a taskbar on top moves the origin down."""
        work = Rect(-1920, 40, 0, 1080)
        assert snap_rect(work, SnapSide.LEFT) == Rect(-1920, 40, -960, 1080)
        assert snap_rect(work, SnapSide.RIGHT) == Rect(-960, 40, 0, 1080)

    def test_every_side_stays_inside_the_work_area(self) -> None:
        work = Rect(10, 20, 1001, 777)
        for side in SnapSide:
            half = snap_rect(work, side)
            assert half.left >= work.left
            assert half.top >= work.top
            assert half.right <= work.right
            assert half.bottom <= work.bottom


class TestWindowBackendSeam:
    """The injection point, and what happens without it."""

    def test_an_injected_backend_is_returned(self, windows_backend: FakeWindows) -> None:
        assert get_window_backend() is windows_backend

    @pytest.mark.skipif(sys.platform == "win32", reason="проверяет отказ вне Windows")
    def test_off_windows_the_refusal_is_in_russian(self) -> None:
        """A macro shared between machines can reach here, and the text is spoken."""
        set_window_backend(None)
        with pytest.raises(ActionUnavailable) as caught:
            get_window_backend()
        assert caught.value.user_message == "Управление окнами работает только в Windows."

    @pytest.mark.skipif(sys.platform != "win32", reason="настоящий backend есть только в Windows")
    def test_on_windows_the_real_backend_is_built(self) -> None:
        set_window_backend(None)
        assert isinstance(get_window_backend(), WinApiWindows)


class TestListWindowsAction:
    """Records rather than a sentence, and a count that declines."""

    def test_the_value_is_the_records_themselves(self, windows_backend: FakeWindows) -> None:
        """A macro loops over them and an LLM tool call reasons about them; a
        formatted string would have to be parsed back."""
        result = ListWindows().run(ListWindows.Params())
        assert result.ok
        assert result.value == list(DESKTOP)

    def test_the_data_carries_the_same_list_in_json(self, windows_backend: FakeWindows) -> None:
        result = ListWindows().run(ListWindows.Params())
        assert result.data["count"] == len(DESKTOP)
        assert result.data["windows"][0] == CHROME.as_dict()

    def test_a_filter_reaches_the_query(self, windows_backend: FakeWindows) -> None:
        result = ListWindows().run(ListWindows.Params(process="chrome"))
        assert result.value == [CHROME, CHROME_DOCS]

    def test_the_limit_is_honoured(self, windows_backend: FakeWindows) -> None:
        result = ListWindows().run(ListWindows.Params(limit=2))
        assert result.data["count"] == 2

    def test_a_query_is_stripped_before_it_is_used(self, windows_backend: FakeWindows) -> None:
        """A recogniser leaves trailing spaces, and « chrome » must still match."""
        result = ListWindows().run(ListWindows.Params(process="  chrome  "))
        assert result.value == [CHROME, CHROME_DOCS]

    def test_nothing_open_is_said_as_a_sentence(self) -> None:
        backend = FakeWindows(records=())
        set_window_backend(backend)
        try:
            result = ListWindows().run(ListWindows.Params())
        finally:
            set_window_backend(None)
        assert result.ok
        assert result.message_ru == "Открытых окон не нашла."
        assert result.value == []

    @pytest.mark.parametrize(
        ("count", "message"),
        [
            (1, "Нашла 1 окно."),
            (2, "Нашла 2 окна."),
            (4, "Нашла 4 окна."),
            (5, "Нашла 5 окон."),
            (11, "Нашла 11 окон."),
            (14, "Нашла 14 окон."),
            (21, "Нашла 21 окно."),
            (22, "Нашла 22 окна."),
            (25, "Нашла 25 окон."),
            (100, "Нашла 100 окон."),
        ],
    )
    def test_the_count_declines(self, count: int, message: str) -> None:
        """Said aloud, so «нашла 21 окон» is a bug the user hears every time."""
        records = tuple(
            WindowRecord(hwnd=0x100 + number, title=f"окно {number}", process="app.exe")
            for number in range(count)
        )
        set_window_backend(FakeWindows(records=records))
        try:
            result = ListWindows().run(ListWindows.Params(limit=MAX_LISTED))
        finally:
            set_window_backend(None)
        assert result.message_ru == message

    def test_the_limit_has_bounds(self) -> None:
        """``ge=1`` and ``le=100``: the editor draws a spin box from them."""
        with pytest.raises(ValidationError):
            ListWindows.Params(limit=0)
        with pytest.raises(ValidationError):
            ListWindows.Params(limit=MAX_LISTED + 1)

    def test_an_unknown_parameter_is_refused(self) -> None:
        """``extra="forbid"``: a macro with a misspelled key is a broken macro."""
        with pytest.raises(ValidationError):
            ListWindows.Params(monitr=1)


class TestFocusWindowAction:
    """«переключись на телеграм», and what is said when Windows says no."""

    def test_a_request_without_any_filter_is_refused(self) -> None:
        """Focusing «any window» is not a request anyone means."""
        with pytest.raises(ValidationError) as caught:
            FocusWindow.Params()
        assert "укажите заголовок, класс или процесс окна" in str(caught.value)

    def test_a_monitor_alone_is_enough_to_aim(self) -> None:
        """«переключись на окно на втором экране» is a request with a meaning."""
        assert FocusWindow.Params(monitor=2).to_query().monitor == 2

    def test_whitespace_is_not_a_filter(self) -> None:
        with pytest.raises(ValidationError):
            FocusWindow.Params(title="   ")

    def test_a_successful_switch_names_the_window(self, windows_backend: FakeWindows) -> None:
        result = FocusWindow().run(FocusWindow.Params(process="telegram"))
        assert result.ok
        assert result.message_ru == "Переключилась на «Telegram»."
        assert result.data["step"] == "direct"
        assert result.data["window"]["hwnd"] == TELEGRAM.hwnd

    def test_a_window_already_in_front_is_reported_as_done(
        self, windows_backend: FakeWindows
    ) -> None:
        result = FocusWindow().run(FocusWindow.Params(title="Ayris — Google Chrome"))
        assert result.ok
        assert result.data["step"] == "already"

    def test_a_refusal_is_a_failed_result_the_user_hears(self) -> None:
        """Not an exception: «Windows не дала переключиться» is the honest answer."""
        set_window_backend(FakeWindows(grant_after=None))
        try:
            result = FocusWindow().run(FocusWindow.Params(process="telegram"))
        finally:
            set_window_backend(None)
        assert not result.ok
        assert result.message_ru == "Windows не дала переключиться на «Telegram»."
        assert result.detail == "foreground denied after 4 attempts"
        assert result.value == TELEGRAM

    def test_no_such_window_raises_rather_than_fails(self, windows_backend: FakeWindows) -> None:
        """A missing window is a different answer from a refused switch, and the
        dispatcher turns the exception into «Не нашла окно «…»»."""
        with pytest.raises(WindowNotFound) as caught:
            FocusWindow().run(FocusWindow.Params(process="firefox"))
        assert caught.value.user_message == "Не нашла окно «firefox»."

    def test_the_front_most_of_several_is_taken(self, windows_backend: FakeWindows) -> None:
        result = FocusWindow().run(FocusWindow.Params(process="chrome"))
        assert result.value == CHROME


class TestWindowStateAction:
    """The exact ``ShowWindow`` constant, and the exact rectangle."""

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            (WindowCommand.MINIMIZE, winapi.SW_MINIMIZE),
            (WindowCommand.MAXIMIZE, winapi.SW_MAXIMIZE),
            (WindowCommand.RESTORE, winapi.SW_RESTORE),
            (WindowCommand.SNAP_LEFT, winapi.SW_RESTORE),
            (WindowCommand.SNAP_RIGHT, winapi.SW_RESTORE),
            (WindowCommand.SNAP_TOP, winapi.SW_RESTORE),
            (WindowCommand.SNAP_BOTTOM, winapi.SW_RESTORE),
        ],
    )
    def test_the_constant_handed_to_windows(
        self, windows_backend: FakeWindows, command: WindowCommand, expected: int
    ) -> None:
        """Snapping restores first: ``SetWindowPos`` on a maximised window is ignored."""
        WindowState().run(WindowState.Params(process="telegram", command=command))
        assert windows_backend.shown == [(TELEGRAM.hwnd, expected)]

    def test_the_constant_is_also_in_the_data(self, windows_backend: FakeWindows) -> None:
        """The audit trail records what was actually asked of Windows."""
        result = WindowState().run(
            WindowState.Params(process="telegram", command=WindowCommand.MAXIMIZE)
        )
        assert result.data["show_command"] == winapi.SW_MAXIMIZE
        assert result.data["command"] == "maximize"

    def test_a_plain_command_moves_nothing(self, windows_backend: FakeWindows) -> None:
        result = WindowState().run(
            WindowState.Params(process="telegram", command=WindowCommand.MINIMIZE)
        )
        assert windows_backend.moved == []
        assert "rect" not in result.data

    def test_snapping_asks_for_the_work_area_of_that_window(
        self, windows_backend: FakeWindows
    ) -> None:
        """The window's own monitor, not the primary one: a window on the second
        display must snap to the second display."""
        WindowState().run(WindowState.Params(process="telegram", command=WindowCommand.SNAP_LEFT))
        assert "work_area" in windows_backend.calls
        assert windows_backend.moved == [(TELEGRAM.hwnd, Rect(0, 0, 960, 1040))]

    def test_the_snapped_rectangle_is_reported(self, windows_backend: FakeWindows) -> None:
        result = WindowState().run(
            WindowState.Params(process="telegram", command=WindowCommand.SNAP_RIGHT)
        )
        assert result.data["rect"] == {"left": 960, "top": 0, "width": 960, "height": 1040}

    @pytest.mark.parametrize(
        ("command", "message"),
        [
            (WindowCommand.MINIMIZE, "Свернула окно «Telegram»."),
            (WindowCommand.MAXIMIZE, "Развернула окно «Telegram»."),
            (WindowCommand.RESTORE, "Восстановила окно «Telegram»."),
            (WindowCommand.SNAP_LEFT, "Прижала влево окно «Telegram»."),
            (WindowCommand.SNAP_RIGHT, "Прижала вправо окно «Telegram»."),
            (WindowCommand.SNAP_TOP, "Прижала вверх окно «Telegram»."),
            (WindowCommand.SNAP_BOTTOM, "Прижала вниз окно «Telegram»."),
        ],
    )
    def test_every_command_has_its_own_past_tense(
        self, windows_backend: FakeWindows, command: WindowCommand, message: str
    ) -> None:
        result = WindowState().run(WindowState.Params(process="telegram", command=command))
        assert result.message_ru == message

    def test_an_empty_query_means_the_active_window(self, windows_backend: FakeWindows) -> None:
        """«сверни» on its own means «сверни это»."""
        result = WindowState().run(WindowState.Params())
        assert result.value == CHROME
        assert windows_backend.shown == [(CHROME.hwnd, winapi.SW_MINIMIZE)]

    def test_no_active_window_is_said_out_loud(self) -> None:
        """A locked screen or a closing shell: the handle describes nothing."""
        set_window_backend(FakeWindows(foreground_hwnd=0))
        try:
            with pytest.raises(WindowNotFound) as caught:
                WindowState().run(WindowState.Params())
        finally:
            set_window_backend(None)
        assert caught.value.user_message == "Не вижу активного окна."

    def test_the_default_command_is_the_common_one(self) -> None:
        assert WindowState.Params().command is WindowCommand.MINIMIZE

    def test_an_unknown_command_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            WindowState.Params(command="закрыть")

    @pytest.mark.parametrize("command", list(WindowCommand))
    def test_every_command_has_a_russian_title_and_past_tense(self, command: WindowCommand) -> None:
        assert command.title_ru
        assert command.done_ru
        assert not command.title_ru.isascii()

    def test_only_the_snap_commands_have_a_side(self) -> None:
        assert WindowCommand.SNAP_LEFT.snap_side is SnapSide.LEFT
        assert WindowCommand.SNAP_BOTTOM.snap_side is SnapSide.BOTTOM
        assert WindowCommand.MINIMIZE.snap_side is None
        assert WindowCommand.RESTORE.snap_side is None


class TestDescribeWindows:
    """The debug listing: one line per window, handle first."""

    def test_the_line_has_the_handle_the_process_and_the_placement(self) -> None:
        line = describe_windows([CHROME])
        assert line == "0x00001001 chrome.exe [normal] Ayris — Google Chrome"

    def test_a_window_without_a_process_is_marked(self) -> None:
        line = describe_windows([WindowRecord(hwnd=1, title="что-то")])
        assert line == "0x00000001 ? [normal] что-то"

    def test_one_line_per_window(self) -> None:
        assert len(describe_windows(DESKTOP).splitlines()) == len(DESKTOP)

    def test_nothing_to_describe_is_an_empty_string(self) -> None:
        assert describe_windows([]) == ""


class TestLaunchKind:
    """Three launch paths, told apart by the target alone."""

    @pytest.mark.parametrize(
        "target",
        [
            r"C:\Program Files\Google\Chrome\chrome.exe",
            "notepad.exe",
            r"C:\Windows\System32\cmd.exe",
        ],
    )
    def test_an_executable(self, target: str) -> None:
        assert LaunchKind.of(target) is LaunchKind.EXECUTABLE

    @pytest.mark.parametrize(
        "target",
        [
            r"C:\ProgramData\Start Menu\Telegram.lnk",
            r"C:\Users\u\Desktop\Ссылка.LNK",
            r"C:\Users\u\Favorites\сайт.url",
        ],
    )
    def test_a_shortcut(self, target: str) -> None:
        """``.lnk`` carries its own arguments and elevation flag, so the shell reads it."""
        assert LaunchKind.of(target) is LaunchKind.SHORTCUT

    @pytest.mark.parametrize(
        "target",
        [
            f"{STORE_PREFIX}{SETTINGS_AUMID}",
            "shell:AppsFolder\\Whatever!App",
            "SHELL:AppsFolder\\Whatever!App",
            "shell:::{20D04FE0-3AEA-1069-A2D8-08002B30309D}",
        ],
    )
    def test_a_shell_moniker(self, target: str) -> None:
        """A Store app has no path at all, and neither does «Мой компьютер»."""
        assert LaunchKind.of(target) is LaunchKind.STORE

    def test_surrounding_space_does_not_change_the_verdict(self) -> None:
        assert LaunchKind.of("  telegram.lnk  ") is LaunchKind.SHORTCUT


class TestLaunchRequest:
    """What actually reaches ``ShellExecuteW``."""

    def test_an_executable_goes_to_the_shell_as_itself(self) -> None:
        request = LaunchRequest(target="chrome.exe", arguments="--incognito")
        assert request.shell_file == "chrome.exe"
        assert request.shell_arguments == "--incognito"

    def test_a_store_app_goes_through_explorer(self) -> None:
        """The only supported way into a package: the same one a click on the tile takes."""
        request = LaunchRequest(
            target=f"{STORE_PREFIX}{SETTINGS_AUMID}",
            kind=LaunchKind.STORE,
        )
        assert request.shell_file == SHELL_BINARY
        assert request.shell_arguments == f"{STORE_PREFIX}{SETTINGS_AUMID}"

    def test_a_store_app_puts_the_moniker_before_the_arguments(self) -> None:
        """A packaged app receives extras through its activation contract, if it has one."""
        request = LaunchRequest(
            target="shell:AppsFolder\\X!App",
            kind=LaunchKind.STORE,
            arguments="--flag",
        )
        assert request.shell_arguments == "shell:AppsFolder\\X!App --flag"

    def test_the_shell_binary_is_never_a_full_path(self) -> None:
        """``App Paths`` resolves it, and ``C:\\Windows`` is wrong on some machines."""
        assert SHELL_BINARY == "explorer.exe"

    def test_the_command_line_is_one_readable_string(self) -> None:
        request = LaunchRequest(target="code.exe", arguments="--new-window")
        assert request.command_ru == "code.exe --new-window"

    def test_a_command_without_arguments_has_no_trailing_space(self) -> None:
        assert LaunchRequest(target="notepad.exe").command_ru == "notepad.exe"

    def test_a_candidate_becomes_a_request(self, index: AppIndex) -> None:
        request = LaunchRequest.for_candidate(index.resolve("хром"))
        assert request.target == r"C:\Program Files\Google\Chrome\chrome.exe"
        assert request.kind is LaunchKind.EXECUTABLE
        assert request.name == "Google Chrome"

    def test_a_shortcut_candidate_keeps_its_kind(self, index: AppIndex) -> None:
        request = LaunchRequest.for_candidate(index.resolve("телеграм"))
        assert request.kind is LaunchKind.SHORTCUT

    def test_a_store_candidate_keeps_its_kind(self, index: AppIndex) -> None:
        request = LaunchRequest.for_candidate(index.resolve("параметры"))
        assert request.kind is LaunchKind.STORE
        assert request.shell_file == SHELL_BINARY

    def test_the_scanned_arguments_are_inherited(self, index: AppIndex) -> None:
        """A Start-menu shortcut for VS Code carries ``--new-window``; dropping it
        would launch the program differently from a click on the same shortcut."""
        request = LaunchRequest.for_candidate(index.resolve("вс код"))
        assert request.arguments == "--new-window"
        assert request.working_dir == r"C:\work"

    def test_the_command_wins_over_the_shortcut(self, index: AppIndex) -> None:
        """«открой вс код с папкой проекта» is a deliberate override, and appending
        both would produce a command line neither side meant."""
        request = LaunchRequest.for_candidate(
            index.resolve("вс код"),
            arguments="--diff a b",
            working_dir=r"D:\proj",
        )
        assert request.arguments == "--diff a b"
        assert request.working_dir == r"D:\proj"

    def test_a_dictionary_only_program_falls_back_to_its_executable(self, index: AppIndex) -> None:
        """«открой блокнот» on a machine whose Start menu the scan could not read:
        the shell resolves ``notepad.exe`` through ``App Paths`` itself."""
        candidate = index.resolve("блокнот")
        assert not candidate.installed
        assert LaunchRequest.for_candidate(candidate).target == "notepad.exe"

    def test_nothing_launchable_is_said_out_loud(self) -> None:
        """A dictionary entry for a program that is not installed and has no bare
        executable name to try: the honest answer is «знаю, но не нашла»."""
        candidate = AppCandidate(app_id="ghost", name="Привидение", confidence=1.0)
        with pytest.raises(AppNotFound) as caught:
            LaunchRequest.for_candidate(candidate)
        assert caught.value.user_message == "Знаю «Привидение», но не нашла его на этом компьютере."


class TestRunApp:
    """«открой хром» — the pid, the counter, and nothing guessed."""

    def test_the_program_is_launched_and_the_pid_returned(
        self, index: AppIndex, launcher: FakeLauncher
    ) -> None:
        result = RunApp().run(RunApp.Params(app="хром"))
        assert result.ok
        assert result.value == launcher.pid
        assert result.message_ru == "Открываю «Google Chrome»."
        assert len(launcher.requests) == 1
        assert launcher.requests[0].target == r"C:\Program Files\Google\Chrome\chrome.exe"

    def test_the_data_describes_what_was_started(
        self, index: AppIndex, launcher: FakeLauncher
    ) -> None:
        result = RunApp().run(RunApp.Params(app="хром"))
        assert result.data["app_id"] == "chrome"
        assert result.data["kind"] == "executable"
        assert result.data["pid"] == launcher.pid
        assert result.data["confidence"] == 1.0

    def test_the_launch_is_counted(self, index: AppIndex, launcher: FakeLauncher) -> None:
        """The counter breaks the next ambiguity, so it has to be written on every run."""
        first = RunApp().run(RunApp.Params(app="хром"))
        second = RunApp().run(RunApp.Params(app="хром"))
        assert first.data["launches"] == 1
        assert second.data["launches"] == 2
        assert index.resolver().launches == {"chrome": 2}

    def test_arguments_reach_the_launcher(self, index: AppIndex, launcher: FakeLauncher) -> None:
        RunApp().run(RunApp.Params(app="хром", arguments="--incognito", working_dir=r"D:\tmp"))
        assert launcher.requests[0].shell_arguments == "--incognito"
        assert launcher.requests[0].working_dir == r"D:\tmp"

    def test_a_store_app_reaches_the_shell_as_a_moniker(
        self, index: AppIndex, launcher: FakeLauncher
    ) -> None:
        """The whole reason :class:`LaunchRequest` exists as a value object."""
        RunApp().run(RunApp.Params(app="параметры"))
        request = launcher.requests[0]
        assert request.shell_file == SHELL_BINARY
        assert request.shell_arguments == f"{STORE_PREFIX}{SETTINGS_AUMID}"

    def test_an_unknown_program_is_refused_rather_than_guessed(
        self, index: AppIndex, launcher: FakeLauncher
    ) -> None:
        with pytest.raises(AppNotFound):
            RunApp().run(RunApp.Params(app="непонятно что"))
        assert launcher.requests == []

    def test_an_empty_name_is_refused_by_the_parameters(self) -> None:
        with pytest.raises(ValidationError):
            RunApp.Params(app="")

    def test_the_name_has_a_length_limit(self) -> None:
        with pytest.raises(ValidationError):
            RunApp.Params(app="х" * 121)


class TestAppWindows:
    """Which windows belong to a program, given that Store apps share a process."""

    def test_by_process_name(self, index: AppIndex) -> None:
        found = app_windows(index.resolve("хром"), FakeWindows())
        assert found == [CHROME, CHROME_DOCS]

    def test_a_store_app_is_found_by_its_caption(self, index: AppIndex) -> None:
        """Every Store application runs inside ``ApplicationFrameHost.exe``, so the
        process name identifies nothing and the title is all there is."""
        found = app_windows(index.resolve("параметры"), FakeWindows())
        assert found == [SETTINGS]

    def test_a_shortcut_candidate_uses_the_executable_it_declares(self, index: AppIndex) -> None:
        """The target is ``Telegram.lnk``; the process is ``Telegram.exe``."""
        found = app_windows(index.resolve("телеграм"), FakeWindows())
        assert found == [TELEGRAM]

    def test_a_program_that_is_not_running_has_no_windows(self, index: AppIndex) -> None:
        assert app_windows(index.resolve("вс код"), FakeWindows(records=(CHROME,))) == []

    def test_a_dictionary_only_program_still_looks_by_process(self, index: AppIndex) -> None:
        """Not installed, but the dictionary knows the executable name — enough to
        find its windows, which is what «закрой блокнот» needs."""
        notepad = WindowRecord(hwnd=0x9001, title="Без имени — Блокнот", process="notepad.exe")
        found = app_windows(index.resolve("блокнот"), FakeWindows(records=(notepad,)))
        assert found == [notepad]


class TestCloseApp:
    """«закрой телеграм» — a request first, a kill only when asked."""

    def test_every_window_gets_a_close_request(
        self, index: AppIndex, launcher: FakeLauncher, windows_backend: FakeWindows
    ) -> None:
        """Chrome has two windows here, and leaving one open is not «закрыла»."""
        result = CloseApp().run(CloseApp.Params(app="хром"))
        assert result.ok
        assert launcher.closed == [CHROME.hwnd, CHROME_DOCS.hwnd]
        assert result.message_ru == "Закрыла «Google Chrome»."
        assert result.data["windows"] == 2

    def test_the_process_is_waited_for_once_per_pid(
        self, index: AppIndex, launcher: FakeLauncher, windows_backend: FakeWindows
    ) -> None:
        """Both Chrome windows share pid 4100; asking about it twice would double
        the wait for a program that closes on the second poll."""
        CloseApp().run(CloseApp.Params(app="хром"))
        assert launcher.waited == [((CHROME.pid,), DEFAULT_CLOSE_TIMEOUT_MS)]

    def test_the_timeout_is_passed_through(
        self, index: AppIndex, launcher: FakeLauncher, windows_backend: FakeWindows
    ) -> None:
        CloseApp().run(CloseApp.Params(app="хром", timeout_ms=800))
        assert launcher.waited == [((CHROME.pid,), 800)]

    def test_a_program_that_ignores_the_request_is_left_alone(
        self, index: AppIndex, windows_backend: FakeWindows
    ) -> None:
        """The default: an unsaved document is a real loss, and «не закрывается»
        gives the person the chance to look at the dialog themselves."""
        launcher = FakeLauncher(alive=(TELEGRAM.pid,))
        set_launcher(launcher)
        try:
            result = CloseApp().run(CloseApp.Params(app="телеграм"))
        finally:
            set_launcher(None)
        assert not result.ok
        assert (
            result.message_ru
            == "«Telegram» не закрывается — возможно, спрашивает про несохранённое."
        )
        assert launcher.terminated == []
        assert result.data["alive"] == [TELEGRAM.pid]

    def test_force_terminates_what_stayed(
        self, index: AppIndex, windows_backend: FakeWindows
    ) -> None:
        launcher = FakeLauncher(alive=(TELEGRAM.pid,))
        set_launcher(launcher)
        try:
            result = CloseApp().run(CloseApp.Params(app="телеграм", force=True))
        finally:
            set_launcher(None)
        assert result.ok
        assert result.message_ru == "«Telegram» не закрылся сам, завершила процесс."
        assert launcher.terminated == [TELEGRAM.pid]
        assert result.data["killed"] == [TELEGRAM.pid]
        assert result.data["alive"] == []

    def test_the_polite_request_comes_first_even_under_force(
        self, index: AppIndex, windows_backend: FakeWindows
    ) -> None:
        """``force`` is a fallback, not a shortcut: a program that saves on
        ``WM_CLOSE`` must get the chance to before anything is killed."""
        launcher = FakeLauncher(alive=(TELEGRAM.pid,))
        set_launcher(launcher)
        try:
            CloseApp().run(CloseApp.Params(app="телеграм", force=True))
        finally:
            set_launcher(None)
        assert launcher.closed == [TELEGRAM.hwnd]
        assert launcher.waited == [((TELEGRAM.pid,), DEFAULT_CLOSE_TIMEOUT_MS)]

    def test_a_refused_kill_is_not_reported_as_success(
        self, index: AppIndex, windows_backend: FakeWindows
    ) -> None:
        """``TerminateProcess`` fails on a process of another user or an elevated
        one, and «завершила» would then be a lie."""
        launcher = FakeLauncher(alive=(TELEGRAM.pid,), terminate_ok=False)
        set_launcher(launcher)
        try:
            result = CloseApp().run(CloseApp.Params(app="телеграм", force=True))
        finally:
            set_launcher(None)
        assert not result.ok
        assert result.data["killed"] == []
        assert result.data["alive"] == [TELEGRAM.pid]

    def test_a_program_that_is_not_running(self, index: AppIndex, launcher: FakeLauncher) -> None:
        set_window_backend(FakeWindows(records=(CHROME,)))
        try:
            result = CloseApp().run(CloseApp.Params(app="телеграм"))
        finally:
            set_window_backend(None)
        assert not result.ok
        assert result.message_ru == "«Telegram» и так не запущен."
        assert result.value == CloseOutcome()
        assert launcher.closed == []

    def test_a_window_without_a_pid_is_not_waited_for(
        self, index: AppIndex, launcher: FakeLauncher
    ) -> None:
        """``GetWindowThreadProcessId`` can fail; a zero pid names no process and
        would make ``wait_exit`` wait out the whole timeout for nothing."""
        ghost = WindowRecord(hwnd=0x9101, title="Блокнот", process="notepad.exe", pid=0)
        set_window_backend(FakeWindows(records=(ghost,)))
        try:
            CloseApp().run(CloseApp.Params(app="блокнот"))
        finally:
            set_window_backend(None)
        assert launcher.closed == [ghost.hwnd]
        assert launcher.waited == [((), DEFAULT_CLOSE_TIMEOUT_MS)]

    def test_closing_is_declared_dangerous(self) -> None:
        """Because of the ``force`` path, so section 14 asks before it runs."""
        assert CloseApp.meta.is_dangerous

    def test_the_outcome_knows_whether_it_worked(self) -> None:
        assert CloseOutcome(windows=1, closed=True).ok
        assert CloseOutcome(windows=1, killed=(10,)).ok
        assert not CloseOutcome(windows=1, alive=(10,)).ok
        assert not CloseOutcome().ok


class TestLauncherSeam:
    """The launcher is replaceable, and says so when it cannot exist."""

    def test_the_installed_launcher_is_returned(self, launcher: FakeLauncher) -> None:
        assert get_launcher() is launcher

    @pytest.mark.skipif(sys.platform == "win32", reason="на Windows launcher настоящий")
    def test_off_windows_the_answer_is_spoken_aloud(self) -> None:
        set_launcher(None)
        with pytest.raises(ActionUnavailable) as caught:
            get_launcher()
        assert caught.value.user_message == "Запуск программ работает только в Windows."

    @pytest.mark.skipif(sys.platform != "win32", reason="настоящий launcher есть только в Windows")
    def test_on_windows_the_real_launcher_appears(self) -> None:
        set_launcher(None)
        assert isinstance(get_launcher(), WinApiLauncher)


class TestGuidFromBytes:
    """The registry stores desktop ids packed; the name keys spell them out."""

    def test_the_windows_layout_is_honoured(self) -> None:
        """First three fields little-endian, the last eight as they lie. Reading the
        blob as plain big-endian bytes would produce a GUID that matches no key."""
        raw = bytes.fromhex("78563412" "3412" "7856" "1122334455667788")
        assert guid_from_bytes(raw) == "{12345678-1234-5678-1122-334455667788}"

    def test_the_spelling_matches_the_registry_keys(self) -> None:
        """Braces and upper case: the names live under
        ``…\\VirtualDesktops\\Desktops\\{GUID}`` spelled exactly this way."""
        guid = guid_from_bytes(bytes(range(16)))
        assert guid.startswith("{")
        assert guid.endswith("}")
        assert guid == guid.upper()

    @pytest.mark.parametrize("length", [0, 15, 17, 32])
    def test_a_blob_of_the_wrong_length_is_refused(self, length: int) -> None:
        with pytest.raises(ValueError, match="expected 16 bytes"):
            guid_from_bytes(bytes(length))


class TestParseDesktopIds:
    """``VirtualDesktopIDs`` is one blob of concatenated GUIDs."""

    def test_the_order_is_the_shell_order(self) -> None:
        """The list is what «второй рабочий стол» counts along, so it must not be
        sorted or de-duplicated on the way in."""
        blob = bytes(16) + bytes([9] * 16) + bytes([1] * 16)
        parsed = parse_desktop_ids(blob)
        assert len(parsed) == 3
        assert parsed[0] == "{00000000-0000-0000-0000-000000000000}"
        assert parsed[1] == guid_from_bytes(bytes([9] * 16))

    def test_an_empty_value_is_an_empty_list(self) -> None:
        assert parse_desktop_ids(b"") == ()

    def test_a_torn_read_costs_one_desktop_not_the_command(self) -> None:
        """The shell rewrites this value while desktops are created and closed."""
        assert len(parse_desktop_ids(bytes(16) + bytes(7))) == 1

    def test_a_value_shorter_than_one_guid_yields_nothing(self) -> None:
        assert parse_desktop_ids(bytes(15)) == ()


class TestDesktopState:
    """The list, the position, and what to call each desktop."""

    def test_the_count_is_the_length_of_the_list(self) -> None:
        assert desktop_state(count=4).count == 4

    def test_numbering_starts_at_one(self) -> None:
        """Because people say «на второй стол», not «на стол номер один»."""
        state = desktop_state(count=3)
        assert state.at(1) is state.desktops[0]
        assert state.at(3) is state.desktops[2]

    @pytest.mark.parametrize("number", [0, -1, 4, 99])
    def test_a_number_outside_the_list_is_nothing(self, number: int) -> None:
        assert desktop_state(count=3).at(number) is None

    def test_a_desktop_without_a_name_is_called_by_its_number(self) -> None:
        assert desktop_state(count=2).at(2) is not None
        desktop = desktop_state(count=2).at(2)
        assert desktop is not None
        assert desktop.label == "рабочий стол 2"

    def test_a_renamed_desktop_is_called_by_its_name(self) -> None:
        state = desktop_state(count=2, names={2: "Работа"})
        desktop = state.at(2)
        assert desktop is not None
        assert desktop.label == "Работа"

    @pytest.mark.parametrize("said", ["Работа", "работа", "  РАБОТА  "])
    def test_a_name_is_matched_however_it_was_said(self, said: str) -> None:
        """Speech recognition capitalises as it pleases, and the person said a word,
        not a spelling."""
        state = desktop_state(count=2, names={2: "Работа"})
        found = state.by_name(said)
        assert found is not None
        assert found.number == 2

    def test_an_unknown_name_is_nothing(self) -> None:
        assert desktop_state(count=2, names={2: "Работа"}).by_name("Отдых") is None

    @pytest.mark.parametrize("said", ["", "   "])
    def test_an_empty_name_matches_nothing_rather_than_the_first_unnamed(self, said: str) -> None:
        """Every other desktop has ``name == ""``, so a careless comparison would
        make «перейди на стол» land on desktop 1."""
        assert desktop_state(count=3).by_name(said) is None

    def test_a_readable_list_with_a_known_position(self) -> None:
        assert desktop_state(count=3, current=2).known

    def test_a_list_without_a_position_is_not_known(self) -> None:
        """``CurrentVirtualDesktop`` can hold a GUID that is not in the list at all
        right after a desktop is closed; the position is then unknown."""
        assert not desktop_state(count=3, current=0).known

    def test_no_list_at_all_is_not_known(self) -> None:
        assert not DesktopState().known

    def test_the_payload_describes_the_whole_list(self) -> None:
        data = desktop_state(count=2, current=2, names={1: "Дом"}).as_dict()
        assert data["count"] == 2
        assert data["current"] == 2
        assert data["desktops"][0] == {
            "number": 1,
            "guid": "{00000000-0000-0000-0000-000000000001}",
            "name": "Дом",
        }

    def test_the_debug_listing_marks_where_we_are(self) -> None:
        lines = list(iter_desktop_labels(desktop_state(count=3, current=2, names={3: "Игры"})))
        assert lines == ["  1. рабочий стол 1", "→ 2. рабочий стол 2", "  3. Игры"]


class TestSwitchPlan:
    """Desktops do not wrap, so the plan is a plain difference."""

    @pytest.mark.parametrize(
        ("current", "target", "direction", "taps"),
        [
            (1, 2, SwitchDirection.NEXT, 1),
            (1, 4, SwitchDirection.NEXT, 3),
            (4, 1, SwitchDirection.PREVIOUS, 3),
            (3, 2, SwitchDirection.PREVIOUS, 1),
        ],
    )
    def test_the_way_and_the_number_of_taps(
        self, current: int, target: int, direction: SwitchDirection, taps: int
    ) -> None:
        plan = switch_plan(desktop_state(count=4, current=current), target=target)
        assert plan == (direction, taps)

    def test_the_long_way_round_is_not_taken(self) -> None:
        """Win+Ctrl+Right on the last desktop does nothing, so «four taps right»
        from 4 to 1 in a list of four would simply not move."""
        direction, taps = switch_plan(desktop_state(count=4, current=4), target=1)
        assert direction is SwitchDirection.PREVIOUS
        assert taps == 3

    def test_a_desktop_that_does_not_exist_is_said_with_the_count(self) -> None:
        """«перейди на пятый стол» when there are three: the number is the useful
        part of the answer."""
        with pytest.raises(DesktopUnavailable) as caught:
            switch_plan(desktop_state(count=3), target=5)
        assert caught.value.user_message == "Рабочего стола 5 нет — их всего 3."

    def test_an_unreadable_list_is_a_different_sentence(self) -> None:
        """«их всего 0» would be nonsense: the desktops exist, the registry read
        did not work."""
        with pytest.raises(DesktopUnavailable) as caught:
            switch_plan(DesktopState(), target=2)
        assert caught.value.user_message == "Не вижу списка рабочих столов."

    def test_an_absurd_distance_is_refused_rather_than_tapped_out(self) -> None:
        """Thirty-three synthesised chords take seconds and look like a stuck key."""
        state = desktop_state(count=MAX_TAPS + 5, current=1)
        with pytest.raises(DesktopUnavailable) as caught:
            switch_plan(state, target=MAX_TAPS + 2)
        assert (
            caught.value.user_message
            == "Слишком далеко переключаться, сделаю только рядом стоящие."
        )

    def test_the_limit_itself_is_allowed(self) -> None:
        state = desktop_state(count=MAX_TAPS + 1, current=1)
        assert switch_plan(state, target=MAX_TAPS + 1) == (SwitchDirection.NEXT, MAX_TAPS)


class TestSwitchDirection:
    """The two chords, and which way each one counts."""

    def test_next_counts_forward(self) -> None:
        assert SwitchDirection.NEXT.step == 1
        assert SwitchDirection.PREVIOUS.step == -1

    def test_the_chords_are_the_shell_ones(self) -> None:
        """Win+Ctrl+Arrow — the same keys a person would press by hand, which is
        why this works without any undocumented COM interface."""
        assert SwitchDirection.NEXT.keys == (winapi.VK_LWIN, winapi.VK_CONTROL, winapi.VK_RIGHT)
        assert SwitchDirection.PREVIOUS.keys == (winapi.VK_LWIN, winapi.VK_CONTROL, winapi.VK_LEFT)

    def test_each_way_has_a_russian_name(self) -> None:
        assert SwitchDirection.NEXT.title_ru == "вправо"
        assert SwitchDirection.PREVIOUS.title_ru == "влево"


class TestDesktopBackendSeam:
    """The registry-and-keyboard backend is replaceable."""

    def test_the_installed_backend_is_returned(self) -> None:
        backend = FakeDesktops(desktop_state())
        set_desktop_backend(backend)
        try:
            assert get_desktop_backend() is backend
        finally:
            set_desktop_backend(None)

    @pytest.mark.skipif(sys.platform == "win32", reason="на Windows столы настоящие")
    def test_off_windows_the_answer_is_spoken_aloud(self) -> None:
        set_desktop_backend(None)
        with pytest.raises(ActionUnavailable) as caught:
            get_desktop_backend()
        assert (
            caught.value.user_message
            == "Виртуальные рабочие столы есть только в Windows 10 и новее."
        )


class TestSwitchDesktopAction:
    """«перейди на второй стол», «переключи вправо», and the edges of both."""

    @staticmethod
    def _run(backend: FakeDesktops, **kwargs: object) -> ActionResult[DesktopState]:
        set_desktop_backend(backend)
        try:
            return SwitchDesktop().run(SwitchDesktop.Params(**kwargs))
        finally:
            set_desktop_backend(None)

    def test_nothing_named_is_refused_by_the_parameters(self) -> None:
        with pytest.raises(ValidationError, match="укажите номер, имя или направление"):
            SwitchDesktop.Params()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"index": 2, "direction": SwitchDirection.NEXT},
            {"name": "Работа", "direction": SwitchDirection.PREVIOUS},
            {"index": 2, "name": "Работа"},
        ],
    )
    def test_two_things_at_once_are_refused(self, kwargs: dict[str, object]) -> None:
        """«на второй стол вправо» has no meaning, and honouring half of it would
        make the mistake invisible."""
        with pytest.raises(ValidationError, match="укажите что-то одно"):
            SwitchDesktop.Params(**kwargs)

    def test_a_blank_name_does_not_count_as_a_name(self) -> None:
        with pytest.raises(ValidationError, match="укажите номер, имя или направление"):
            SwitchDesktop.Params(name="   ")

    def test_switching_by_number(self) -> None:
        backend = FakeDesktops(desktop_state(count=3, current=1))
        result = self._run(backend, index=3)
        assert result.ok
        assert backend.taps == [(SwitchDirection.NEXT, 2)]
        assert result.message_ru == "Перешла на «рабочий стол 3»."
        assert result.data["current"] == 3
        assert result.data["requested"] == 3

    def test_switching_by_name(self) -> None:
        backend = FakeDesktops(desktop_state(count=3, current=3, names={1: "Дом"}))
        result = self._run(backend, name="дом")
        assert backend.taps == [(SwitchDirection.PREVIOUS, 2)]
        assert result.message_ru == "Перешла на «Дом»."

    def test_being_already_there_is_not_a_switch(self) -> None:
        """Tapping the chord anyway would animate a switch to the same desktop."""
        backend = FakeDesktops(desktop_state(count=3, current=2, names={2: "Работа"}))
        result = self._run(backend, index=2)
        assert result.ok
        assert result.message_ru == "Уже на «Работа»."
        assert backend.taps == []

    def test_an_unknown_name_is_said_out_loud(self) -> None:
        backend = FakeDesktops(desktop_state(count=2, names={1: "Дом"}))
        with pytest.raises(DesktopUnavailable) as caught:
            self._run(backend, name="Работа")
        assert caught.value.user_message == "Не нашла рабочий стол «Работа»."
        assert backend.taps == []

    def test_a_number_needs_a_starting_point(self) -> None:
        """An absolute jump is done with relative chords, so «на третий стол» is
        impossible when the current position could not be read."""
        backend = FakeDesktops(DesktopState(desktops=desktop_state(count=3).desktops, current=0))
        with pytest.raises(DesktopUnavailable) as caught:
            self._run(backend, index=3)
        assert caught.value.user_message == "Не поняла, на каком рабочем столе мы сейчас."
        assert backend.taps == []

    def test_a_number_outside_the_list_is_said_with_the_count(self) -> None:
        backend = FakeDesktops(desktop_state(count=3, current=1))
        with pytest.raises(DesktopUnavailable) as caught:
            self._run(backend, index=7)
        assert caught.value.user_message == "Рабочего стола 7 нет — их всего 3."

    @pytest.mark.parametrize(
        ("current", "direction", "landed"),
        [
            (1, SwitchDirection.NEXT, 2),
            (3, SwitchDirection.PREVIOUS, 2),
        ],
    )
    def test_switching_sideways(
        self, current: int, direction: SwitchDirection, landed: int
    ) -> None:
        backend = FakeDesktops(desktop_state(count=3, current=current))
        result = self._run(backend, direction=direction)
        assert backend.taps == [(direction, 1)]
        assert result.message_ru == f"Перешла на «рабочий стол {landed}»."

    @pytest.mark.parametrize(
        ("current", "direction", "edge"),
        [
            (3, SwitchDirection.NEXT, "последнем"),
            (1, SwitchDirection.PREVIOUS, "первом"),
        ],
    )
    def test_the_edge_is_reported_rather_than_tapped_into(
        self, current: int, direction: SwitchDirection, edge: str
    ) -> None:
        """The chord at the edge does nothing at all, and «переключила» would then
        be a lie about a desktop that never changed."""
        backend = FakeDesktops(desktop_state(count=3, current=current))
        result = self._run(backend, direction=direction)
        assert not result.ok
        assert result.message_ru == f"Это уже {edge} рабочий стол."
        assert backend.taps == []

    def test_sideways_works_even_when_the_list_is_unreadable(self) -> None:
        """The one case that needs no knowledge of where we are: the chord itself
        is relative, so «переключи вправо» is honoured blind."""
        backend = FakeDesktops(DesktopState())
        result = self._run(backend, direction=SwitchDirection.NEXT)
        assert result.ok
        assert result.message_ru == "Переключила рабочий стол вправо."
        assert result.detail == "blind switch next"
        assert backend.taps == [(SwitchDirection.NEXT, 1)]

    def test_the_blind_switch_taps_exactly_once(self) -> None:
        """Without a position there is nothing to count against, and a second tap
        would move a desktop the person did not ask for."""
        backend = FakeDesktops(DesktopState(current=2))
        self._run(backend, direction=SwitchDirection.PREVIOUS)
        assert backend.taps == [(SwitchDirection.PREVIOUS, 1)]

    def test_the_state_is_read_back_after_the_taps(self) -> None:
        """The result data should describe where we ended up, not where we were."""
        backend = FakeDesktops(desktop_state(count=4, current=1))
        result = self._run(backend, index=4)
        assert backend.reads == 2
        assert result.value.current == 4

    def test_a_slow_shell_animation_is_not_a_failure(self) -> None:
        """The chord is asynchronous: a read straight afterwards can still show the
        old desktop. The taps were sent, so the command succeeded, and the number
        that was asked for stays in the payload next to what the read saw."""
        backend = FakeDesktops(desktop_state(count=3, current=1), moves=False)
        result = self._run(backend, index=3)
        assert result.ok
        assert result.data["requested"] == 3
        assert result.data["current"] == 1

    def test_a_failed_read_back_does_not_undo_the_switch(self) -> None:
        """The registry read can start failing between the plan and the report. The
        taps were already sent, so this is reported as success with the last state
        that could be read, not as an error."""

        class Fickle(FakeDesktops):
            def state(self) -> DesktopState:
                if self.taps:
                    raise DesktopUnavailable("gone", user_message="Не вижу столов.")
                return super().state()

        backend = Fickle(desktop_state(count=3, current=1))
        result = self._run(backend, index=3)
        assert result.ok
        assert result.data["requested"] == 3

    def test_an_unknown_position_afterwards_names_the_desktop_asked_for(self) -> None:
        """``CurrentVirtualDesktop`` holds a GUID that is briefly absent from the
        list while the shell animates. The switch happened; the name to say is the
        one that was aimed at."""

        class Amnesiac(FakeDesktops):
            def state(self) -> DesktopState:
                read = super().state()
                return DesktopState(desktops=read.desktops, current=0) if self.taps else read

        backend = Amnesiac(desktop_state(count=3, current=1, names={3: "Игры"}))
        result = self._run(backend, index=3)
        assert result.ok
        assert result.message_ru == "Перешла на «Игры»."


class TestSchemas:
    """Every action of this task, as the settings UI and the LLM see it."""

    @pytest.mark.parametrize(
        "action",
        [ListWindows, FocusWindow, WindowState, RunApp, CloseApp, SwitchDesktop],
    )
    def test_the_schema_is_buildable_and_russian(self, action: type) -> None:
        """A field the UI cannot label is a field nobody can set: every label has to
        be a real Russian sentence, not a leftover field name."""
        schema = build_schema(action)
        assert schema.title_ru
        assert schema.fields
        for field in schema.fields:
            assert field.label_ru
            assert field.label_ru == field.label_ru.strip()

    def test_the_window_selector_is_the_same_everywhere(self) -> None:
        """The three window actions share one selector, so «сверни хром» and
        «переключись на хром» cannot disagree about what «хром» means."""
        selector = ("title", "regex", "class_name", "process", "monitor")
        for action in (ListWindows, FocusWindow, WindowState):
            names = tuple(field.name for field in build_schema(action).fields)
            assert names[: len(selector)] == selector

    def test_the_selector_bounds(self) -> None:
        fields = {field.name: field for field in build_schema(ListWindows).fields}
        assert fields["title"].kind is FieldKind.TEXT
        assert fields["title"].max_length == 200
        assert fields["class_name"].max_length == 120
        assert fields["process"].max_length == 120
        assert fields["regex"].kind is FieldKind.BOOLEAN
        assert (fields["monitor"].minimum, fields["monitor"].maximum) == (0.0, 16.0)

    def test_the_listing_limit_is_bounded_at_the_edge(self) -> None:
        """Unbounded, an LLM asking for every window would fill the answer with
        toolbars and hidden shells."""
        limit = next(f for f in build_schema(ListWindows).fields if f.name == "limit")
        assert (limit.minimum, limit.maximum) == (1.0, float(MAX_LISTED))
        assert limit.default == 20

    def test_the_window_commands_are_offered_as_a_list(self) -> None:
        """A free-text field here would invite «свернуть в трей», which no command
        can do."""
        command = next(f for f in build_schema(WindowState).fields if f.name == "command")
        assert command.kind is FieldKind.CHOICE
        assert [choice.value for choice in command.choices] == [str(item) for item in WindowCommand]
        assert [choice.label_ru for choice in command.choices][:3] == [
            "Свернуть",
            "Развернуть",
            "Восстановить",
        ]

    def test_the_program_name_is_the_only_required_field(self) -> None:
        for action in (RunApp, CloseApp):
            required = [f.name for f in build_schema(action).fields if f.required]
            assert required == ["app"]

    def test_the_launch_extras_have_room_but_not_infinite_room(self) -> None:
        fields = {field.name: field for field in build_schema(RunApp).fields}
        assert fields["arguments"].max_length == 1_000
        assert fields["working_dir"].max_length == 500

    def test_the_close_timeout_carries_its_unit(self) -> None:
        """«5000» in a settings box means nothing without «мс» next to it."""
        timeout = next(f for f in build_schema(CloseApp).fields if f.name == "timeout_ms")
        assert timeout.unit_ru == "мс"
        assert timeout.default == DEFAULT_CLOSE_TIMEOUT_MS
        assert (timeout.minimum, timeout.maximum) == (200.0, 30_000.0)

    def test_only_closing_is_marked_dangerous(self) -> None:
        """Focusing or minimising a window loses nothing; killing a process does."""
        assert build_schema(CloseApp).is_dangerous
        for action in (ListWindows, FocusWindow, WindowState, RunApp, SwitchDesktop):
            assert not build_schema(action).is_dangerous

    def test_the_desktop_directions_are_offered_as_a_list(self) -> None:
        direction = next(f for f in build_schema(SwitchDesktop).fields if f.name == "direction")
        assert direction.kind is FieldKind.CHOICE
        assert [(c.value, c.label_ru) for c in direction.choices] == [
            ("next", "вправо"),
            ("previous", "влево"),
        ]

    def test_the_desktop_number_is_bounded(self) -> None:
        """Sixty-four is already absurd; the bound is there so a hallucinated
        «перейди на стол 900000» is refused before any chord is synthesised."""
        index = next(f for f in build_schema(SwitchDesktop).fields if f.name == "index")
        assert (index.minimum, index.maximum) == (0.0, 64.0)

    def test_the_actions_land_in_sensible_categories(self) -> None:
        assert build_schema(RunApp).category == build_schema(CloseApp).category
        assert build_schema(SwitchDesktop).category == build_schema(ListWindows).category
