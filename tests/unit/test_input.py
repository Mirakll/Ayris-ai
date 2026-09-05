"""Задача 23: клавиатура и мышь — без клавиатуры и мыши.

Everything here runs on a machine with no input devices, and it still covers the
parts that actually break. That is possible because the layer has two seams and
all the arithmetic sits above them:

* :class:`~ayris.actions.input.backend.RecordingBackend` is Windows. Every
  assertion about *what was sent* — the order of down and up, the normalised
  coordinates, the interpolated steps of a drag, the release of a stuck modifier —
  is an assertion about the list of events it collected.
* :class:`~ayris.actions.input.mouse.ScreenBackend` is the desktop. A fake one
  describes a two-monitor mixed-DPI layout with the second display to the *left*
  of the primary, which is the configuration that exposes the sign error in
  coordinate normalisation. No second monitor required.

Four things carry the most weight.

*A combination is parsed, not guessed.* ``ctrl+shift+f5`` has to produce those
three keys in that order, ``tab+alt`` has to be reordered into Alt-then-Tab —
press Tab first and a window switch becomes a tab character — and a malformed
string has to be refused in Russian rather than silently pressing something.

*Coordinates are inclusive over the virtual desktop.* ``SendInput`` takes a
fraction of 65535, the desktop starts at a negative x whenever a monitor sits to
the left of the primary one, and the last pixel column must reach 65535 exactly.
Each of those is one test, because each of them puts a click on the wrong monitor
when it is wrong.

*Nothing stays held.* A :class:`~ayris.actions.input.keys.KeyDown` records what it
pressed, :func:`~ayris.actions.input.keys.release_held_keys` puts it back up in
reverse order, and a combination that fails halfway still releases what it managed
to press. On a real desktop the alternative is a stuck Ctrl.

*A missing driver is a fallback, not a crash.* ``backend = "interception"`` on a
machine without the kernel driver has to log the reason once and go on through
``SendInput`` — which is every CI runner and most users, so the test asserts it
rather than trusting it.

Groups:

* :class:`TestKeyTable` — names, aliases, the ``E0`` flag, unknown keys.
* :class:`TestComboParser` — order, duplicates, the plus key, malformed strings.
* :class:`TestBackendSelection` — the config choice, the fallback, the seam.
* :class:`TestSendInputBackend` — flags and counts handed to ``SendInput``.
* :class:`TestKeyActions` — KeyPress/KeyDown/KeyUp and the timings they read.
* :class:`TestHeldKeys` — the registry, reverse release, release on failure.
* :class:`TestTypeText` — Unicode, the layout mode, the clipboard threshold.
* :class:`TestNormalizePoint` — the 0..65535 conversion under three layouts.
* :class:`TestDragPath` — step count, spacing, the degenerate case.
* :class:`TestMousePoint` — desktop, monitor, window and cursor frames; DPI.
* :class:`TestMouseActions` — click, move, drag and wheel as event sequences.
* :class:`TestRealSendInput` — one Windows-only call that really injects.
* :class:`TestRealClipboardPaste` — the paste route against the real clipboard.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from ayris.actions.input import backend as backend_module
from ayris.actions.input import keys as keys_module
from ayris.actions.input import mouse as mouse_module
from ayris.actions.input.backend import (
    ABSOLUTE_MAX,
    BackendKind,
    InputDriverMissing,
    InterceptionBackend,
    KeyStroke,
    MouseButton,
    RecordingBackend,
    SendInputBackend,
    get_input_backend,
    reset_input_backend,
    set_input_backend,
)
from ayris.actions.input.keys import (
    ALIASES,
    KEYS,
    MODIFIERS,
    KeyDown,
    KeyPress,
    KeyUp,
    TypeMode,
    TypeText,
    UnknownKey,
    held_keys,
    parse_combo,
    release_held_keys,
    resolve_key,
)
from ayris.actions.input.mouse import (
    MouseClick,
    MouseDrag,
    MouseMove,
    MousePoint,
    MouseWheel,
    Origin,
    ScreenLayout,
    drag_path,
    normalize_point,
    set_screen_backend,
)
from ayris.actions.system import clipboard as clipboard_module
from ayris.actions.system.clipboard import (
    ClipboardKind,
    ClipboardSnapshot,
    FakeClipboard,
    reset_clipboard,
    set_clipboard,
)
from ayris.core.errors import ActionError
from ayris.utils.logger import ROOT_LOGGER_NAME
from ayris.utils.winapi import MonitorInfo, Rect

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from contextlib import AbstractContextManager

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

#: A single 1920×1080 display at the origin. The boring case.
SINGLE = ScreenLayout(
    virtual=Rect(0, 0, 1920, 1080),
    monitors=(
        MonitorInfo(
            handle=1,
            rect=Rect(0, 0, 1920, 1080),
            work=Rect(0, 0, 1920, 1040),
            device=r"\\.\DISPLAY1",
            primary=True,
        ),
    ),
    dpi=(96,),
)

#: Two displays, the second one to the *left* of the primary and scaled to 150 %.
#: The negative ``left`` is the whole point: it is what a normalisation that
#: forgets the offset gets wrong, and it is also the layout a laptop with an
#: external monitor plugged in on the left actually reports.
DUAL = ScreenLayout(
    virtual=Rect(-1600, 0, 1920, 1080),
    monitors=(
        MonitorInfo(
            handle=1,
            rect=Rect(0, 0, 1920, 1080),
            work=Rect(0, 0, 1920, 1040),
            device=r"\\.\DISPLAY1",
            primary=True,
        ),
        MonitorInfo(
            handle=2,
            rect=Rect(-1600, 0, 0, 900),
            work=Rect(-1600, 0, 0, 860),
            device=r"\\.\DISPLAY2",
        ),
    ),
    dpi=(96, 144),
)


class FakeScreen:
    """A desktop of whatever shape the test needs."""

    def __init__(
        self,
        layout: ScreenLayout = SINGLE,
        *,
        cursor: tuple[int, int] = (100, 100),
        window: Rect | None = None,
    ) -> None:
        self._layout = layout
        self._cursor = cursor
        self._window = window
        self.layout_calls = 0

    def layout(self) -> ScreenLayout:
        self.layout_calls += 1
        return self._layout

    def cursor(self) -> tuple[int, int]:
        return self._cursor

    def active_window(self) -> Rect | None:
        return self._window


class FakeInterception:
    """The ``interception`` wrapper's surface, as :class:`InterceptionBackend` uses it."""

    KEYBOARD_KEYS = ("ctrl", "shift", "w", "f5", "esc", "enter", "space")

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def key_down(self, key: str) -> None:
        self.calls.append(("key_down", key))

    def key_up(self, key: str) -> None:
        self.calls.append(("key_up", key))

    def mouse_down(self, button: str) -> None:
        self.calls.append(("mouse_down", button))

    def mouse_up(self, button: str) -> None:
        self.calls.append(("mouse_up", button))

    def scroll(self, direction: str) -> None:
        self.calls.append(("scroll", direction))


@pytest.fixture
def recorder() -> Iterator[RecordingBackend]:
    """Install a recording backend for the duration of one test."""
    fake = RecordingBackend()
    set_input_backend(fake)
    try:
        yield fake
    finally:
        set_input_backend(None)
        reset_input_backend()
        keys_module._held.clear()


@pytest.fixture
def clipboard() -> Iterator[FakeClipboard]:
    """Install a clipboard in a variable, because paste mode now needs a real one.

    ``TypeText`` in clipboard mode goes through the project's one clipboard
    wrapper, and that wrapper is win32-only. Without this seam every paste-mode
    test would be skipped outside Windows — and the part worth asserting, that
    whatever the user had copied comes back afterwards, has nothing to do with
    Windows. :class:`TestRealClipboardPaste` covers the win32 half.
    """
    fake = FakeClipboard()
    set_clipboard(fake)
    try:
        yield fake
    finally:
        set_clipboard(None)
        reset_clipboard()
        # The paste hides its own two writes from the history monitor; the marks
        # are consumed by a monitor that is not running here.
        clipboard_module._suppressed.clear()


@pytest.fixture
def screen() -> Iterator[FakeScreen]:
    """Install the single-monitor desktop; a test may replace the layout."""
    fake = FakeScreen()
    set_screen_backend(fake)
    try:
        yield fake
    finally:
        set_screen_backend(None)


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree, exactly once each.

    ``caplog`` listens on the interpreter root, and whether a record reaches it
    depends on test order: :func:`setup_logging` sets ``propagate = False`` on the
    ``ayris`` logger, so before it has run a record arrives twice — once through
    the handler attached here and once through the root — and afterwards not at
    all. Attaching the handler directly and pinning propagation off makes both
    «was it logged» and «was it logged once» mean what they say.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


@pytest.fixture(autouse=True)
def _no_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delays are configuration, and this suite is not testing the clock.

    The defaults are non-zero on purpose — a macro that types at machine speed
    loses characters — which would make the keyboard tests below take seconds of
    real time each. Nothing here asserts on duration, so the waits go.
    """
    monkeypatch.setattr(keys_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(mouse_module.time, "sleep", lambda _seconds: None)


class TestKeyTable:
    """Names into virtual keys, and the flag that decides Home from Numpad-7."""

    @pytest.mark.parametrize(
        ("name", "vk"),
        [
            ("ctrl", 0x11),
            ("shift", 0x10),
            ("alt", 0x12),
            ("win", 0x5B),
            ("a", 0x41),
            ("z", 0x5A),
            ("0", 0x30),
            ("9", 0x39),
            ("f1", 0x70),
            ("f24", 0x87),
            ("enter", 0x0D),
            ("escape", 0x1B),
            ("space", 0x20),
            ("left", 0x25),
            ("num5", 0x65),
            ("semicolon", 0xBA),
            ("volumeup", 0xAF),
        ],
    )
    def test_a_name_maps_to_its_virtual_key(self, name: str, vk: int) -> None:
        assert resolve_key(name).vk == vk

    @pytest.mark.parametrize(
        ("name", "extended"),
        [
            ("left", True),
            ("right", True),
            ("up", True),
            ("down", True),
            ("home", True),
            ("end", True),
            ("insert", True),
            ("delete", True),
            ("pageup", True),
            ("pagedown", True),
            ("ralt", True),
            ("rctrl", True),
            ("numdivide", True),
            ("numenter", True),
            ("ctrl", False),
            ("shift", False),
            ("a", False),
            ("num5", False),
            ("enter", False),
        ],
    )
    def test_the_e0_prefix_is_recorded(self, name: str, extended: bool) -> None:
        """A key sent without ``KEYEVENTF_EXTENDEDKEY`` arrives as its numpad twin."""
        assert resolve_key(name).extended is extended

    def test_numpad_enter_and_enter_share_a_virtual_key_but_not_the_flag(self) -> None:
        """The only thing that tells the two Enters apart is the extended flag."""
        assert KEYS["numenter"].vk == KEYS["enter"].vk
        assert KEYS["numenter"].extended and not KEYS["enter"].extended

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("ктрл", "ctrl"),
            ("контрол", "ctrl"),
            ("control", "ctrl"),
            ("шифт", "shift"),
            ("альт", "alt"),
            ("вин", "win"),
            ("винда", "win"),
            ("ввод", "enter"),
            ("энтер", "enter"),
            ("return", "enter"),
            ("esc", "escape"),
            ("пробел", "space"),
            ("таб", "tab"),
            ("вверх", "up"),
            ("вправо", "right"),
            ("удалить", "delete"),
            ("pgup", "pageup"),
        ],
    )
    def test_an_alias_resolves_to_the_canonical_key(self, alias: str, canonical: str) -> None:
        """Dictation produces «ктрл шифт», and it has to mean the same as ``ctrl+shift``."""
        assert resolve_key(alias) is KEYS[canonical]

    def test_case_and_spacing_do_not_matter(self) -> None:
        assert resolve_key("  CTRL ") is KEYS["ctrl"]
        assert resolve_key("PageUp") is KEYS["pageup"]

    @pytest.mark.parametrize("name", ["", "   ", "нажми", "f25", "ctrl-c", "й", "мышь"])
    def test_an_unknown_name_is_refused_in_russian(self, name: str) -> None:
        """Pressing a hopeful key is worse than saying the name was not understood."""
        with pytest.raises(UnknownKey) as info:
            resolve_key(name)
        assert info.value.user_message

    def test_every_modifier_is_in_the_table(self) -> None:
        assert set(KEYS) >= MODIFIERS

    def test_every_alias_points_at_a_real_key(self) -> None:
        """An alias for a key that does not exist is a typo nobody would notice."""
        assert set(ALIASES.values()) <= set(KEYS)

    def test_a_key_knows_its_own_name(self) -> None:
        """The name is carried along, because messages and the held-key registry use it."""
        assert all(name == key.name for name, key in KEYS.items())


class TestComboParser:
    """``ctrl+shift+f5`` into three keys, in the order they must be pressed."""

    def test_a_plain_combination_keeps_its_order(self) -> None:
        assert [key.name for key in parse_combo("ctrl+shift+f5")] == ["ctrl", "shift", "f5"]

    def test_a_single_key_is_a_combination_of_one(self) -> None:
        assert [key.name for key in parse_combo("f5")] == ["f5"]

    def test_modifiers_are_moved_to_the_front(self) -> None:
        """``tab+alt`` typed the wrong way round is still a window switch.

        Pressed in the written order it is a Tab character followed by an Alt that
        nothing is holding — not the same thing at all.
        """
        assert [key.name for key in parse_combo("tab+alt")] == ["alt", "tab"]

    def test_relative_order_within_the_modifiers_survives(self) -> None:
        assert [key.name for key in parse_combo("shift+ctrl+c")] == ["shift", "ctrl", "c"]

    def test_spaces_around_the_separator_are_ignored(self) -> None:
        assert [key.name for key in parse_combo(" ctrl + alt + delete ")] == [
            "ctrl",
            "alt",
            "delete",
        ]

    def test_a_repeated_key_is_pressed_once(self) -> None:
        assert [key.name for key in parse_combo("ctrl+ctrl+c")] == ["ctrl", "c"]

    def test_russian_modifiers_parse(self) -> None:
        assert [key.name for key in parse_combo("ктрл+шифт+эск")] == ["ctrl", "shift", "escape"]

    @pytest.mark.parametrize(
        ("combo", "names"),
        [
            ("ctrl++", ["ctrl", "equal"]),
            ("+", ["equal"]),
            ("ctrl+plus", ["ctrl", "equal"]),
            ("ctrl+=", ["ctrl", "equal"]),
        ],
    )
    def test_the_plus_key_can_itself_be_pressed(self, combo: str, names: list[str]) -> None:
        """Ctrl+Plus is «zoom in» in half the applications there are."""
        assert [key.name for key in parse_combo(combo)] == names

    @pytest.mark.parametrize(
        "combo",
        ["", "   ", "ctrl+", "ctrl+alt+", "ctrl+нажми", "хватит", "ctrl+f25"],
    )
    def test_a_malformed_combination_is_refused(self, combo: str) -> None:
        with pytest.raises(UnknownKey) as info:
            parse_combo(combo)
        assert info.value.user_message

    def test_the_refusal_names_the_combination(self) -> None:
        """«В комбинации … не хватает клавиши» beats «ошибка разбора»."""
        with pytest.raises(UnknownKey) as info:
            parse_combo("ctrl+")
        assert "ctrl+" in info.value.user_message


class TestBackendSelection:
    """Which implementation is in force, and what happens when it cannot be."""

    def test_the_default_is_sendinput(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_input_backend()
        assert isinstance(get_input_backend(), SendInputBackend)

    def test_the_configured_choice_is_read_from_the_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ayris.core import config as config_module

        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__BACKEND", "interception")
        config_module.reset_config_manager()
        assert backend_module.backend_kind() is BackendKind.INTERCEPTION

    def test_a_missing_driver_falls_back_to_sendinput(
        self, monkeypatch: pytest.MonkeyPatch, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """The point of the whole optional-backend design.

        No kernel driver on a CI runner, and none on the machine of anyone who
        never ran the installer. Asking for it must produce ordinary input and a
        line in the log — not a macro that refuses to run.
        """
        monkeypatch.setattr(backend_module, "backend_kind", lambda: BackendKind.INTERCEPTION)

        def no_driver() -> InterceptionBackend:
            raise InputDriverMissing("driver is not installed")

        monkeypatch.setattr(InterceptionBackend, "create", staticmethod(no_driver))
        reset_input_backend()
        chosen = get_input_backend()
        assert isinstance(chosen, SendInputBackend)
        assert "SendInput" in ayris_log.text

    def test_the_fallback_is_logged_once_however_many_calls(
        self, monkeypatch: pytest.MonkeyPatch, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        """A macro in a loop would otherwise fill the log with one repeated line."""
        monkeypatch.setattr(backend_module, "backend_kind", lambda: BackendKind.INTERCEPTION)
        monkeypatch.setattr(
            InterceptionBackend,
            "create",
            staticmethod(lambda: (_ for _ in ()).throw(InputDriverMissing("no driver"))),
        )
        reset_input_backend()
        get_input_backend()
        # Only the cached instance goes, not the record of what has been warned
        # about: reset_input_backend() clears both on purpose, because a user who
        # switched the setting deserves to hear the reason again.
        backend_module._state.backend = None
        get_input_backend()
        assert ayris_log.text.count("работаю через SendInput") == 1

    def test_the_user_is_told_about_the_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A line in the log is not something a user reads. The tray balloon is."""
        from ayris.core.events import EventBus, NotificationRequested

        bus = EventBus()
        seen: list[NotificationRequested] = []
        bus.subscribe(NotificationRequested, seen.append)
        monkeypatch.setattr(backend_module, "backend_kind", lambda: BackendKind.INTERCEPTION)
        monkeypatch.setattr(
            InterceptionBackend,
            "create",
            staticmethod(lambda: (_ for _ in ()).throw(InputDriverMissing("no driver"))),
        )
        backend_module.set_input_bus(bus)
        reset_input_backend()
        try:
            get_input_backend()
            backend_module._state.backend = None
            get_input_backend()
        finally:
            backend_module.set_input_bus(None)
        assert len(seen) == 1  # once per reason, not once per call
        assert "Interception" in seen[0].message
        assert seen[0].level == "warning"

    def test_without_a_bus_the_fallback_still_happens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing here may depend on a UI being wired up — the CLI has none."""
        monkeypatch.setattr(backend_module, "backend_kind", lambda: BackendKind.INTERCEPTION)
        monkeypatch.setattr(
            InterceptionBackend,
            "create",
            staticmethod(lambda: (_ for _ in ()).throw(InputDriverMissing("no driver"))),
        )
        backend_module.set_input_bus(None)
        reset_input_backend()
        assert isinstance(get_input_backend(), SendInputBackend)

    def test_the_instance_is_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Capturing the driver's devices is not free, so it happens once."""
        reset_input_backend()
        assert get_input_backend() is get_input_backend()

    def test_a_reset_makes_the_setting_reread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        reset_input_backend()
        first = get_input_backend()
        reset_input_backend()
        assert get_input_backend() is not first

    def test_an_installed_backend_wins_over_the_setting(self, recorder: RecordingBackend) -> None:
        """The test seam: no configuration reaches past an explicit override."""
        assert get_input_backend() is recorder

    def test_the_driver_is_windows_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(backend_module.sys, "platform", "linux")
        with pytest.raises(InputDriverMissing):
            InterceptionBackend.create()

    def test_nothing_here_installs_anything(self) -> None:
        """A driver install needs administrator rights and a reboot: never silently.

        Asserted against the parsed source rather than the prose, because the
        failure mode is a future edit that adds a convenient ``install_driver()``
        call and the review that would catch it is this test.
        """
        import ast
        from pathlib import Path

        tree = ast.parse(Path(backend_module.__file__).read_text(encoding="utf-8"))
        called = {
            node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert not {name for name in called if "install" in name.lower()}


class TestInterceptionBackend:
    """The driver wrapper, against a fake of its own surface."""

    def test_a_key_it_knows_goes_through_the_driver(self) -> None:
        module = FakeInterception()
        driver = InterceptionBackend(module)
        driver.key_down(KEYS["w"])
        driver.key_up(KEYS["w"])
        assert module.calls == [("key_down", "w"), ("key_up", "w")]

    def test_a_key_it_does_not_know_goes_through_sendinput(self) -> None:
        """A gap in the wrapper's key list must not fail the whole macro."""
        module = FakeInterception()
        fallback = RecordingBackend()
        driver = InterceptionBackend(module, fallback=fallback)
        driver.key_down(KEYS["f13"])
        assert module.calls == []
        assert fallback.keys == ("+f13",)

    def test_text_always_goes_through_sendinput(self) -> None:
        """The driver types through the layout, which cannot produce «привет»."""
        fallback = RecordingBackend()
        driver = InterceptionBackend(FakeInterception(), fallback=fallback)
        driver.type_text("привет")
        assert fallback.typed == "привет"

    def test_the_pointer_goes_through_sendinput(self) -> None:
        """The driver takes pixels; what arrives here is already normalised."""
        fallback = RecordingBackend()
        driver = InterceptionBackend(FakeInterception(), fallback=fallback)
        driver.mouse_move(100, 200)
        assert fallback.points == ((100, 200),)

    def test_buttons_go_through_the_driver(self) -> None:
        module = FakeInterception()
        driver = InterceptionBackend(module)
        driver.mouse_button(MouseButton.LEFT, pressed=True)
        driver.mouse_button(MouseButton.X1, pressed=False)
        assert module.calls == [("mouse_down", "left"), ("mouse_up", "mouse4")]

    def test_the_wheel_is_one_notch_per_call(self) -> None:
        """The driver takes a direction, not an amount."""
        module = FakeInterception()
        driver = InterceptionBackend(module)
        driver.mouse_wheel(-3)
        assert module.calls == [("scroll", "down")] * 3

    def test_a_horizontal_wheel_goes_through_sendinput(self) -> None:
        """The wrapper has no horizontal scroll at all."""
        fallback = RecordingBackend()
        driver = InterceptionBackend(FakeInterception(), fallback=fallback)
        driver.mouse_wheel(2, horizontal=True)
        assert module_events(fallback, "mouse_wheel")[0].horizontal is True


def module_events(recorder: RecordingBackend, kind: str) -> list[backend_module.InputEvent]:
    """Every recorded event of one kind, in order."""
    return [event for event in recorder.events if event.kind == kind]


class TestSendInputBackend:
    """Exactly which flags and values reach ``SendInput``, against a stubbed winapi."""

    @pytest.fixture
    def sent(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        """Collect the triples and mouse arguments instead of injecting them."""
        collected: list[Any] = []

        def key_events(events: Any) -> int:
            collected.extend(events)
            return len(list(events))

        def mouse_event(*, flags: int, dx: int = 0, dy: int = 0, data: int = 0) -> int:
            collected.append(("mouse", flags, dx, dy, data))
            return 1

        def unicode_text(text: str) -> int:
            collected.append(("text", text))
            return len(text) * 2

        monkeypatch.setattr(backend_module.winapi, "send_key_events", key_events)
        monkeypatch.setattr(backend_module.winapi, "send_mouse_event", mouse_event)
        monkeypatch.setattr(backend_module.winapi, "send_unicode_text", unicode_text)
        monkeypatch.setattr(backend_module.winapi, "map_virtual_key", lambda _vk: 0x3F)
        return collected

    def test_a_press_sends_the_virtual_key_without_the_keyup_flag(self, sent: list[Any]) -> None:
        SendInputBackend().key_down(KEYS["a"])
        assert sent == [(0x41, 0x3F, 0)]

    def test_a_release_sets_the_keyup_flag(self, sent: list[Any]) -> None:
        SendInputBackend().key_up(KEYS["a"])
        assert sent == [(0x41, 0x3F, backend_module.winapi.KEYEVENTF_KEYUP)]

    def test_an_extended_key_carries_its_flag(self, sent: list[Any]) -> None:
        SendInputBackend().key_down(KEYS["left"])
        assert sent == [(0x25, 0x3F, backend_module.winapi.KEYEVENTF_EXTENDEDKEY)]

    def test_the_scancode_mode_drops_the_virtual_key(self, sent: list[Any]) -> None:
        """DirectInput reads ``wScan`` and ignores ``wVk``; sending both confuses some games."""
        SendInputBackend().key_down(KEYS["w"], scancode=True)
        assert sent == [(0, 0x3F, backend_module.winapi.KEYEVENTF_SCANCODE)]

    def test_a_key_with_no_scancode_still_goes_as_a_virtual_key(
        self, sent: list[Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Media keys have no scan code at all — that is not an error."""
        monkeypatch.setattr(backend_module.winapi, "map_virtual_key", lambda _vk: 0)
        SendInputBackend().key_down(KEYS["volumeup"], scancode=True)
        assert sent == [(0xAF, 0, backend_module.winapi.KEYEVENTF_EXTENDEDKEY)]

    def test_an_explicit_scancode_is_not_looked_up(self, sent: list[Any]) -> None:
        SendInputBackend().key_down(KeyStroke(name="ф", vk=0x41, scan=0x1E), scancode=True)
        assert sent == [(0, 0x1E, backend_module.winapi.KEYEVENTF_SCANCODE)]

    def test_a_move_is_absolute_over_the_whole_virtual_desktop(self, sent: list[Any]) -> None:
        """Without ``VIRTUALDESK`` the coordinates are read against the primary monitor only."""
        SendInputBackend().mouse_move(1000, 2000)
        flags = sent[0][1]
        assert flags & backend_module.winapi.MOUSEEVENTF_ABSOLUTE
        assert flags & backend_module.winapi.MOUSEEVENTF_VIRTUALDESK
        assert flags & backend_module.winapi.MOUSEEVENTF_MOVE
        assert sent[0][2:4] == (1000, 2000)

    @pytest.mark.parametrize(
        ("button", "pressed", "flag_name", "data"),
        [
            (MouseButton.LEFT, True, "MOUSEEVENTF_LEFTDOWN", 0),
            (MouseButton.LEFT, False, "MOUSEEVENTF_LEFTUP", 0),
            (MouseButton.RIGHT, True, "MOUSEEVENTF_RIGHTDOWN", 0),
            (MouseButton.MIDDLE, False, "MOUSEEVENTF_MIDDLEUP", 0),
            (MouseButton.X1, True, "MOUSEEVENTF_XDOWN", 1),
            (MouseButton.X2, True, "MOUSEEVENTF_XDOWN", 2),
            (MouseButton.X2, False, "MOUSEEVENTF_XUP", 2),
        ],
    )
    def test_each_button_has_its_own_flag(
        self, sent: list[Any], button: MouseButton, pressed: bool, flag_name: str, data: int
    ) -> None:
        """The side buttons are one flag and a number, which is easy to get backwards."""
        SendInputBackend().mouse_button(button, pressed=pressed)
        assert sent == [("mouse", getattr(backend_module.winapi, flag_name), 0, 0, data)]

    def test_a_wheel_notch_is_multiplied_by_the_delta(self, sent: list[Any]) -> None:
        """A person means notches; the API counts 120ths of one."""
        SendInputBackend().mouse_wheel(-2)
        assert sent == [("mouse", backend_module.winapi.MOUSEEVENTF_WHEEL, 0, 0, -240)]

    def test_a_horizontal_wheel_has_its_own_flag(self, sent: list[Any]) -> None:
        SendInputBackend().mouse_wheel(1, horizontal=True)
        assert sent[0][1] == backend_module.winapi.MOUSEEVENTF_HWHEEL

    def test_zero_notches_sends_nothing(self, sent: list[Any]) -> None:
        SendInputBackend().mouse_wheel(0)
        assert sent == []

    def test_empty_text_sends_nothing(self, sent: list[Any]) -> None:
        SendInputBackend().type_text("")
        assert sent == []

    def test_an_injection_of_zero_events_is_reported_as_blocked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """UIPI's refusal is silent: ``SendInput`` returns 0 and Windows says nothing.

        Which means a macro aimed at an elevated window looks like it worked and
        did nothing at all. Saying so is the only way the user can act on it.
        """
        monkeypatch.setattr(backend_module.winapi, "send_key_events", lambda _events: 0)
        monkeypatch.setattr(backend_module.winapi, "map_virtual_key", lambda _vk: 0x1E)
        with pytest.raises(backend_module.InputBlocked) as info:
            SendInputBackend().key_down(KEYS["a"])
        assert "администратор" in info.value.user_message

    def test_a_winapi_failure_is_reported_as_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(events: Any) -> int:
            raise backend_module.winapi.WinApiError("SendInput failed")

        monkeypatch.setattr(backend_module.winapi, "send_key_events", boom)
        monkeypatch.setattr(backend_module.winapi, "map_virtual_key", lambda _vk: 0x1E)
        with pytest.raises(backend_module.InputBlocked):
            SendInputBackend().key_down(KEYS["a"])


class TestKeyActions:
    """KeyPress, KeyDown and KeyUp as the sequence of events they produce."""

    def test_a_press_goes_down_in_order_and_up_in_reverse(self, recorder: RecordingBackend) -> None:
        """Releasing Ctrl before F5 would leave the modifier applying to whatever came next."""
        KeyPress().run(KeyPress.Params(combo="ctrl+shift+f5"))
        assert recorder.keys == ("+ctrl", "+shift", "+f5", "-f5", "-shift", "-ctrl")

    def test_repeats_press_the_whole_combination_each_time(
        self, recorder: RecordingBackend
    ) -> None:
        KeyPress().run(KeyPress.Params(combo="f5", times=3))
        assert recorder.keys == ("+f5", "-f5") * 3

    def test_the_result_names_what_was_pressed(self, recorder: RecordingBackend) -> None:
        result = KeyPress().run(KeyPress.Params(combo="alt+tab"))
        assert result.ok
        assert result.value == "alt+tab"
        assert "alt+tab" in result.message_ru

    def test_the_result_counts_the_repeats(self, recorder: RecordingBackend) -> None:
        result = KeyPress().run(KeyPress.Params(combo="tab", times=4))
        assert "×4" in result.message_ru
        assert result.data["times"] == 4

    def test_the_scancode_flag_reaches_the_backend(self, recorder: RecordingBackend) -> None:
        KeyPress().run(KeyPress.Params(combo="w", scancode=True))
        assert all(event.scancode for event in module_events(recorder, "key_down"))

    def test_an_unknown_combination_never_touches_the_keyboard(
        self, recorder: RecordingBackend
    ) -> None:
        with pytest.raises(UnknownKey):
            KeyPress().run(KeyPress.Params(combo="ctrl+нажми"))
        assert recorder.events == []

    @pytest.mark.parametrize(
        "params",
        [
            {"combo": "", "times": 1},
            {"combo": "f5", "times": 0},
            {"combo": "f5", "times": 101},
            {"combo": "f5", "hold_ms": -1},
            {"combo": "f5", "hold_ms": 9999},
            {"combo": "f5", "delay_ms": 5000},
        ],
    )
    def test_impossible_parameters_are_refused_by_the_model(self, params: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            KeyPress.Params(**params)

    def test_the_timings_come_from_the_settings(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hold and delay are configuration, read per call rather than at import."""
        waits: list[float] = []
        monkeypatch.setattr(keys_module.time, "sleep", lambda seconds: waits.append(seconds))
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__KEY_HOLD_MS", "40")
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__KEY_DELAY_MS", "70")
        from ayris.core import config as config_module

        config_module.reset_config_manager()
        KeyPress().run(KeyPress.Params(combo="f5", times=2))
        assert waits == [0.040, 0.070, 0.040]

    def test_an_explicit_timing_overrides_the_settings(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A macro block may need a longer hold than the global default."""
        waits: list[float] = []
        monkeypatch.setattr(keys_module.time, "sleep", lambda seconds: waits.append(seconds))
        KeyPress().run(KeyPress.Params(combo="f5", hold_ms=250))
        assert waits == [0.250]

    def test_a_hold_of_zero_waits_not_at_all(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []
        monkeypatch.setattr(keys_module.time, "sleep", lambda seconds: waits.append(seconds))
        KeyPress().run(KeyPress.Params(combo="f5", hold_ms=0))
        assert waits == []

    def test_a_keydown_leaves_the_key_down(self, recorder: RecordingBackend) -> None:
        """The reason KeyDown exists: Shift held across the blocks that follow."""
        KeyDown().run(KeyDown.Params(combo="shift"))
        assert recorder.keys == ("+shift",)
        assert held_keys() == ("shift",)

    def test_a_keyup_releases_a_named_key(self, recorder: RecordingBackend) -> None:
        KeyDown().run(KeyDown.Params(combo="ctrl+shift"))
        recorder.clear()
        KeyUp().run(KeyUp.Params(combo="shift"))
        assert recorder.keys == ("-shift",)
        assert held_keys() == ("ctrl",)

    def test_a_keyup_with_no_combination_releases_everything(
        self, recorder: RecordingBackend
    ) -> None:
        KeyDown().run(KeyDown.Params(combo="ctrl+shift+w"))
        recorder.clear()
        result = KeyUp().run(KeyUp.Params())
        assert recorder.keys == ("-w", "-shift", "-ctrl")
        assert held_keys() == ()
        assert result.data["released"] == ["w", "shift", "ctrl"]

    def test_a_keyup_with_nothing_held_says_so(self, recorder: RecordingBackend) -> None:
        result = KeyUp().run(KeyUp.Params())
        assert result.ok
        assert recorder.events == []
        assert "не было" in result.message_ru


class TestHeldKeys:
    """The registry that keeps a stopped macro from leaving Ctrl down."""

    def test_release_returns_the_keys_in_reverse_order(self, recorder: RecordingBackend) -> None:
        """Reverse, because a modifier pressed first has to be the last one let go."""
        KeyDown().run(KeyDown.Params(combo="ctrl+alt+delete"))
        assert release_held_keys() == ("delete", "alt", "ctrl")

    def test_release_is_idempotent(self, recorder: RecordingBackend) -> None:
        KeyDown().run(KeyDown.Params(combo="shift"))
        release_held_keys()
        assert release_held_keys() == ()

    def test_a_failure_to_release_one_key_does_not_strand_the_others(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup continues: a stuck Shift must not keep Ctrl down as well."""

        class HalfBrokenBackend(RecordingBackend):
            def key_up(self, key: KeyStroke, *, scancode: bool = False) -> None:
                if key.name == "alt":
                    raise backend_module.InputBlocked("refused")
                super().key_up(key, scancode=scancode)

        broken = HalfBrokenBackend()
        set_input_backend(broken)
        try:
            KeyDown().run(KeyDown.Params(combo="ctrl+alt+shift"))
            broken.clear()
            released = release_held_keys()
        finally:
            set_input_backend(None)
            keys_module._held.clear()
        assert released == ("shift", "ctrl")
        assert held_keys() == ()

    def test_a_combination_that_fails_halfway_releases_what_it_pressed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guarantee that matters most: no exception leaves a modifier down."""

        class FailsOnTheThirdKey(RecordingBackend):
            def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
                if key.name == "f5":
                    raise backend_module.InputBlocked("refused")
                super().key_down(key, scancode=scancode)

        failing = FailsOnTheThirdKey()
        set_input_backend(failing)
        try:
            with pytest.raises(backend_module.InputBlocked):
                KeyPress().run(KeyPress.Params(combo="ctrl+shift+f5"))
        finally:
            set_input_backend(None)
        assert failing.keys == ("+ctrl", "+shift", "-shift", "-ctrl")

    def test_a_keydown_of_the_same_key_twice_is_held_once(self, recorder: RecordingBackend) -> None:
        KeyDown().run(KeyDown.Params(combo="shift"))
        KeyDown().run(KeyDown.Params(combo="shift"))
        assert held_keys() == ("shift",)

    def test_the_scancode_mode_is_remembered_for_the_release(
        self, recorder: RecordingBackend
    ) -> None:
        """A key pressed as a scan code has to be released as one, or a game holds it forever."""
        KeyDown().run(KeyDown.Params(combo="w", scancode=True))
        recorder.clear()
        release_held_keys()
        assert module_events(recorder, "key_up")[0].scancode is True


class TestTypeText:
    """Three routes to the same characters, and how the mode is chosen."""

    def test_cyrillic_goes_through_as_characters(self, recorder: RecordingBackend) -> None:
        """The default mode exists so «привет» arrives with a US layout in front."""
        result = TypeText().run(TypeText.Params(text="привет", mode=TypeMode.UNICODE))
        assert recorder.typed == "привет"
        assert result.value == 6

    def test_an_emoji_survives(self, recorder: RecordingBackend) -> None:
        """A surrogate pair is one character here and two code units at the API."""
        TypeText().run(TypeText.Params(text="🙂", mode=TypeMode.UNICODE))
        assert recorder.typed == "🙂"

    def test_characters_go_one_at_a_time_when_there_is_a_delay(
        self, recorder: RecordingBackend
    ) -> None:
        """One call per character is what makes the configured pause real."""
        TypeText().run(TypeText.Params(text="abc", mode=TypeMode.UNICODE, char_delay_ms=5))
        assert [event.text for event in module_events(recorder, "type_text")] == ["a", "b", "c"]

    def test_a_delay_of_zero_sends_the_whole_string_at_once(
        self, recorder: RecordingBackend
    ) -> None:
        """No pause to honour, so no reason to pay for one call per character."""
        TypeText().run(TypeText.Params(text="abc", mode=TypeMode.UNICODE, char_delay_ms=0))
        assert [event.text for event in module_events(recorder, "type_text")] == ["abc"]

    def test_the_character_delay_comes_from_the_settings(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []
        monkeypatch.setattr(keys_module.time, "sleep", lambda seconds: waits.append(seconds))
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__CHAR_DELAY_MS", "9")
        from ayris.core import config as config_module

        config_module.reset_config_manager()
        TypeText().run(TypeText.Params(text="abc", mode=TypeMode.UNICODE))
        assert waits == [0.009, 0.009]

    def test_empty_text_does_nothing_and_says_so(self, recorder: RecordingBackend) -> None:
        result = TypeText().run(TypeText.Params(text=""))
        assert recorder.events == []
        assert result.value == 0

    def test_the_layout_mode_sends_scancodes(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What a game needs: a real key of the active layout, with its scan code."""
        monkeypatch.setattr(keys_module.winapi, "vk_key_scan", lambda _char: 0x41)
        monkeypatch.setattr(keys_module.winapi, "map_virtual_key", lambda _vk: 0x1E)
        TypeText().run(TypeText.Params(text="a", mode=TypeMode.LAYOUT))
        assert recorder.keys == ("+a", "-a")
        assert all(event.scancode for event in recorder.events if event.kind != "type_text")

    def test_the_layout_mode_holds_shift_for_a_capital(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``VkKeyScanW`` reports the modifier state in the high byte; 0x0100 is Shift."""
        monkeypatch.setattr(keys_module.winapi, "vk_key_scan", lambda _char: 0x0141)
        monkeypatch.setattr(keys_module.winapi, "map_virtual_key", lambda _vk: 0x1E)
        TypeText().run(TypeText.Params(text="A", mode=TypeMode.LAYOUT))
        assert recorder.keys == ("+shift", "+A", "-A", "-shift")

    def test_the_layout_mode_falls_back_to_unicode_for_what_it_cannot_type(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A US layout cannot produce «ж»; dropping the character silently would be worse."""
        monkeypatch.setattr(keys_module.winapi, "vk_key_scan", lambda _char: -1)
        TypeText().run(TypeText.Params(text="ж", mode=TypeMode.LAYOUT))
        assert recorder.typed == "ж"

    def test_shift_is_released_even_when_the_key_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same guarantee as a combination: nothing leaves Shift down."""
        monkeypatch.setattr(keys_module.winapi, "vk_key_scan", lambda _char: 0x0141)
        monkeypatch.setattr(keys_module.winapi, "map_virtual_key", lambda _vk: 0x1E)

        class RefusesLetters(RecordingBackend):
            def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
                if key.name == "A":
                    raise backend_module.InputBlocked("refused")
                super().key_down(key, scancode=scancode)

        failing = RefusesLetters()
        set_input_backend(failing)
        try:
            with pytest.raises(backend_module.InputBlocked):
                TypeText().run(TypeText.Params(text="A", mode=TypeMode.LAYOUT))
        finally:
            set_input_backend(None)
        assert failing.keys == ("+shift", "-shift")

    def test_auto_stays_with_unicode_below_the_threshold(self, recorder: RecordingBackend) -> None:
        result = TypeText().run(TypeText.Params(text="привет"))
        assert result.data["mode"] == "unicode"

    def test_auto_switches_to_the_clipboard_above_the_threshold(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A thousand characters typed one at a time is half a minute of visible work."""
        pasted: list[str] = []

        def record(_self: TypeText, text: str, *, backend: object) -> None:
            pasted.append(text)

        monkeypatch.setattr(TypeText, "_paste", record)
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__CLIPBOARD_THRESHOLD", "10")
        from ayris.core import config as config_module

        config_module.reset_config_manager()
        result = TypeText().run(TypeText.Params(text="x" * 20))
        assert result.data["mode"] == "clipboard"
        assert pasted == ["x" * 20]

    def test_a_threshold_of_zero_never_uses_the_clipboard(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The setting doubles as an off switch, which a user on a shared clipboard wants."""
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__CLIPBOARD_THRESHOLD", "0")
        from ayris.core import config as config_module

        config_module.reset_config_manager()
        result = TypeText().run(TypeText.Params(text="x" * 5000))
        assert result.data["mode"] == "unicode"

    def test_the_clipboard_route_pastes_and_restores(
        self, recorder: RecordingBackend, clipboard: FakeClipboard
    ) -> None:
        """Losing what the user had copied would be a worse failure than a slow paste."""
        clipboard.put(ClipboardSnapshot(kind=ClipboardKind.TEXT, text="что-то важное"))
        TypeText().run(TypeText.Params(text="длинный текст", mode=TypeMode.CLIPBOARD))
        assert recorder.keys == ("+ctrl", "+v", "-v", "-ctrl")
        assert clipboard.writes == ["длинный текст", "что-то важное"]

    def test_neither_write_reaches_the_clipboard_history(
        self, recorder: RecordingBackend, clipboard: FakeClipboard
    ) -> None:
        """The user copied neither value: the clipboard is a transport here, not a copy."""
        clipboard.put(ClipboardSnapshot(kind=ClipboardKind.TEXT, text="что-то важное"))
        TypeText().run(TypeText.Params(text="длинный текст", mode=TypeMode.CLIPBOARD))
        for text in ("длинный текст", "что-то важное"):
            assert clipboard_module._claim_suppressed(text), text

    def test_a_picture_in_the_clipboard_is_not_restored_as_text(
        self, recorder: RecordingBackend, clipboard: FakeClipboard
    ) -> None:
        """A screenshot cannot be put back, and pretending otherwise loses it silently.

        It is already lost — win32 has no «give me back the bitmap I overwrote» — so
        the honest end state is an empty clipboard rather than the *word* for what
        used to be there.
        """
        clipboard.put(ClipboardSnapshot(kind=ClipboardKind.IMAGE))
        TypeText().run(TypeText.Params(text="длинный текст", mode=TypeMode.CLIPBOARD))
        assert clipboard.writes == ["длинный текст", ""]

    def test_the_clipboard_is_restored_even_when_the_paste_fails(
        self, clipboard: FakeClipboard
    ) -> None:
        clipboard.put(ClipboardSnapshot(kind=ClipboardKind.TEXT, text="исходное"))

        class RefusesEverything(RecordingBackend):
            def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
                raise backend_module.InputBlocked("refused")

        set_input_backend(RefusesEverything())
        try:
            with pytest.raises(backend_module.InputBlocked):
                TypeText().run(TypeText.Params(text="текст", mode=TypeMode.CLIPBOARD))
        finally:
            set_input_backend(None)
            reset_input_backend()
        assert clipboard.snapshot.text == "исходное"

    def test_a_clipboard_that_cannot_be_read_still_pastes(
        self, recorder: RecordingBackend, clipboard: FakeClipboard
    ) -> None:
        """Nothing readable means nothing worth putting back, not a refused paste."""
        clipboard.busy_reads = 99
        TypeText().run(TypeText.Params(text="длинный текст", mode=TypeMode.CLIPBOARD))
        assert recorder.keys == ("+ctrl", "+v", "-v", "-ctrl")
        assert clipboard.writes[0] == "длинный текст"

    def test_text_longer_than_the_field_allows_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            TypeText.Params(text="x" * 100_001)


class TestNormalizePoint:
    """The one piece of arithmetic that puts a click on the wrong monitor."""

    def test_the_origin_is_zero(self) -> None:
        assert normalize_point(0, 0, Rect(0, 0, 1920, 1080)) == (0, 0)

    def test_the_last_pixel_reaches_the_top_of_the_range(self) -> None:
        """Dividing by the width instead of ``width - 1`` leaves this one step short."""
        assert normalize_point(1919, 1079, Rect(0, 0, 1920, 1080)) == (ABSOLUTE_MAX, ABSOLUTE_MAX)

    def test_the_middle_is_the_middle(self) -> None:
        nx, ny = normalize_point(960, 540, Rect(0, 0, 1920, 1080))
        assert abs(nx - ABSOLUTE_MAX // 2) < 40
        assert abs(ny - ABSOLUTE_MAX // 2) < 40

    def test_a_monitor_to_the_left_shifts_the_origin(self) -> None:
        """``virtual.left`` is negative here; forget to subtract it and everything slides."""
        assert normalize_point(-1600, 0, DUAL.virtual) == (0, 0)

    def test_the_primary_origin_is_not_zero_on_a_left_hand_layout(self) -> None:
        """The point a user calls «0, 0» is a third of the way across the virtual desktop."""
        nx, _ = normalize_point(0, 0, DUAL.virtual)
        assert nx == (1600 * ABSOLUTE_MAX) // (3520 - 1)

    def test_the_far_right_of_a_left_hand_layout_still_reaches_the_top(self) -> None:
        assert normalize_point(1919, 0, DUAL.virtual)[0] == ABSOLUTE_MAX

    @pytest.mark.parametrize(
        ("x", "y"),
        [(-99_999, 0), (0, -99_999), (99_999, 0), (0, 99_999)],
    )
    def test_a_point_outside_the_desktop_is_clamped_not_wrapped(self, x: int, y: int) -> None:
        """Integer arithmetic on a negative offset would wrap to the far edge."""
        nx, ny = normalize_point(x, y, Rect(0, 0, 1920, 1080))
        assert 0 <= nx <= ABSOLUTE_MAX
        assert 0 <= ny <= ABSOLUTE_MAX

    def test_a_one_pixel_desktop_does_not_divide_by_zero(self) -> None:
        assert normalize_point(0, 0, Rect(0, 0, 1, 1)) == (0, 0)

    def test_an_empty_desktop_is_refused_in_russian(self) -> None:
        with pytest.raises(ActionError) as caught:
            normalize_point(10, 10, Rect(0, 0, 0, 0))
        assert "рабочего стола" in caught.value.user_message


class TestDragPath:
    """Interpolation: enough points, evenly spaced, ending exactly on target."""

    def test_the_path_ends_on_the_target(self) -> None:
        assert drag_path((0, 0), (100, 50), step_px=10)[-1] == (100, 50)

    def test_the_start_is_not_repeated(self) -> None:
        """The pointer is already there; sending it again is a wasted event."""
        assert drag_path((0, 0), (100, 0), step_px=10)[0] != (0, 0)

    def test_the_step_count_follows_the_step_length(self) -> None:
        assert len(drag_path((0, 0), (100, 0), step_px=25)) == 4
        assert len(drag_path((0, 0), (100, 0), step_px=10)) == 10

    def test_a_step_longer_than_the_distance_gives_one_jump(self) -> None:
        assert drag_path((0, 0), (30, 0), step_px=500) == ((30, 0),)

    def test_no_step_is_longer_than_asked(self) -> None:
        path = drag_path((0, 0), (317, 211), step_px=40)
        previous = (0, 0)
        for point in path:
            assert max(abs(point[0] - previous[0]), abs(point[1] - previous[1])) <= 40
            previous = point

    def test_the_diagonal_moves_on_both_axes(self) -> None:
        """A path that only interpolates x draws an L, and a canvas selection notices."""
        path = drag_path((0, 0), (100, 100), step_px=25)
        assert all(x == y for x, y in path)

    def test_a_negative_direction_works(self) -> None:
        path = drag_path((100, 100), (0, 0), step_px=25)
        assert path[-1] == (0, 0)
        assert all(x <= 100 and y <= 100 for x, y in path)

    def test_the_same_point_gives_one_event(self) -> None:
        """Degenerate but reachable: a click-and-hold expressed as a zero-length drag."""
        assert drag_path((7, 7), (7, 7), step_px=10) == ((7, 7),)

    def test_a_step_of_zero_does_not_hang(self) -> None:
        """``step_px`` is validated in the params, but the helper is called from elsewhere too."""
        assert drag_path((0, 0), (3, 0), step_px=0)[-1] == (3, 0)


class TestMousePoint:
    """Four frames of reference and two coordinate scales, resolved to real pixels."""

    def test_desktop_coordinates_are_taken_as_they_are(self) -> None:
        point = MousePoint(x=300, y=200)
        assert point.resolve(FakeScreen(), SINGLE) == (300, 200)

    def test_monitor_coordinates_are_relative_to_the_primary_by_default(self) -> None:
        point = MousePoint(x=10, y=20, origin=Origin.MONITOR)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (10, 20)

    def test_monitor_coordinates_follow_the_monitor_number(self) -> None:
        """The second display starts at ``x = -1600``, so «10, 20 on monitor 2» is negative."""
        point = MousePoint(x=10, y=20, origin=Origin.MONITOR, monitor=2)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (-1590, 20)

    def test_a_monitor_that_is_not_there_is_refused_in_russian(self) -> None:
        point = MousePoint(origin=Origin.MONITOR, monitor=5)
        with pytest.raises(ActionError) as caught:
            point.resolve(FakeScreen(DUAL), DUAL)
        assert "Монитора №5" in caught.value.user_message

    def test_an_empty_desktop_has_no_monitor_to_measure_from(self) -> None:
        empty = ScreenLayout(virtual=Rect(0, 0, 1920, 1080))
        point = MousePoint(origin=Origin.MONITOR)
        with pytest.raises(ActionError) as caught:
            point.resolve(FakeScreen(empty), empty)
        assert "ни одного монитора" in caught.value.user_message

    def test_window_coordinates_are_relative_to_the_foreground_window(self) -> None:
        screen = FakeScreen(window=Rect(400, 300, 1000, 800))
        point = MousePoint(x=50, y=60, origin=Origin.WINDOW)
        assert point.resolve(screen, SINGLE) == (450, 360)

    def test_no_foreground_window_is_refused_in_russian(self) -> None:
        point = MousePoint(x=1, y=1, origin=Origin.WINDOW)
        with pytest.raises(ActionError) as caught:
            point.resolve(FakeScreen(), SINGLE)
        assert "активного окна" in caught.value.user_message

    def test_cursor_coordinates_are_an_offset(self) -> None:
        screen = FakeScreen(cursor=(700, 400))
        point = MousePoint(x=-50, y=25, origin=Origin.CURSOR)
        assert point.resolve(screen, SINGLE) == (650, 425)

    def test_a_zero_cursor_offset_stays_put(self) -> None:
        screen = FakeScreen(cursor=(700, 400))
        assert MousePoint(origin=Origin.CURSOR).resolve(screen, SINGLE) == (700, 400)

    def test_physical_coordinates_ignore_scaling(self) -> None:
        """A 150 % monitor with ``logical=False``: pixels are pixels."""
        point = MousePoint(x=100, y=100, origin=Origin.MONITOR, monitor=2)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (-1500, 100)

    def test_logical_coordinates_are_scaled_by_the_monitor_dpi(self) -> None:
        """144 dpi is 150 %, so 100 logical points are 150 physical pixels."""
        point = MousePoint(x=100, y=100, origin=Origin.MONITOR, monitor=2, logical=True)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (-1450, 150)

    def test_logical_coordinates_on_an_unscaled_monitor_change_nothing(self) -> None:
        """Same params, monitor 1 at 96 dpi — this is the mixed-DPI case in one pair."""
        point = MousePoint(x=100, y=100, origin=Origin.MONITOR, monitor=1, logical=True)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (100, 100)

    def test_a_logical_window_offset_uses_the_dpi_of_the_window_monitor(self) -> None:
        """The window sits on the left-hand 150 % display, so its offsets scale too."""
        screen = FakeScreen(DUAL, window=Rect(-800, 100, -200, 600))
        point = MousePoint(x=100, y=0, origin=Origin.WINDOW, logical=True)
        assert point.resolve(screen, DUAL) == (-650, 100)

    def test_a_logical_desktop_point_uses_the_monitor_it_lands_on(self) -> None:
        point = MousePoint(x=-800, y=100, logical=True)
        assert point.resolve(FakeScreen(DUAL), DUAL) == (-1200, 150)

    def test_an_unknown_monitor_dpi_falls_back_to_no_scaling(self) -> None:
        """``GetDpiForMonitor`` can fail; treating 0 as 0 % would collapse every coordinate."""
        broken = ScreenLayout(virtual=DUAL.virtual, monitors=DUAL.monitors, dpi=(0, 0))
        point = MousePoint(x=100, y=100, origin=Origin.MONITOR, monitor=2, logical=True)
        assert point.resolve(FakeScreen(broken), broken) == (-1500, 100)

    @pytest.mark.parametrize(("field", "value"), [("x", 200_000), ("y", -200_000)])
    def test_coordinates_far_outside_any_desktop_are_refused(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError):
            MousePoint(**{field: value})

    @pytest.mark.parametrize("monitor", [0, 17])
    def test_an_impossible_monitor_number_is_refused(self, monitor: int) -> None:
        with pytest.raises(ValidationError):
            MousePoint(monitor=monitor)


class TestMouseActions:
    """The four pointer blocks, read off the list of events they produced."""

    def test_a_move_normalises_the_target(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        result = MouseMove().run(MouseMove.Params(x=1919, y=1079))
        assert recorder.points == ((ABSOLUTE_MAX, ABSOLUTE_MAX),)
        assert result.value == (1919, 1079)

    def test_a_move_reports_physical_pixels_not_the_normalised_pair(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        """The result is for a human and a следующий шаг макроса, not for ``SendInput``."""
        result = MouseMove().run(MouseMove.Params(x=300, y=200))
        assert result.data == {"x": 300, "y": 200}

    def test_a_move_onto_the_left_hand_monitor_lands_there(
        self, recorder: RecordingBackend
    ) -> None:
        """End to end through the negative origin: the failure mode is the other monitor."""
        set_screen_backend(FakeScreen(DUAL))
        try:
            MouseMove().run(MouseMove.Params(x=-1600, y=0))
        finally:
            set_screen_backend(None)
        assert recorder.points == ((0, 0),)

    def test_the_layout_is_read_once_per_action(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        """A drag makes dozens of moves; re-enumerating displays for each is both slow
        and inconsistent if something gets unplugged halfway."""
        MouseDrag().run(
            MouseDrag.Params(start=MousePoint(x=0, y=0), end=MousePoint(x=200, y=0), step_px=10)
        )
        assert screen.layout_calls == 1

    def test_a_click_presses_and_releases(self, recorder: RecordingBackend) -> None:
        MouseClick().run(MouseClick.Params())
        assert [(event.button, event.pressed) for event in recorder.events] == [
            (MouseButton.LEFT, True),
            (MouseButton.LEFT, False),
        ]

    def test_a_click_does_not_move_the_pointer_by_default(self, recorder: RecordingBackend) -> None:
        """Coordinates default to ``0, 0``; clicking at the corner instead of where the
        pointer is would be a nasty surprise, so a move is opt-in."""
        MouseClick().run(MouseClick.Params(x=500, y=500))
        assert recorder.points == ()

    def test_a_click_with_move_goes_to_the_point_first(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        MouseClick().run(MouseClick.Params(x=960, y=540, move=True))
        assert recorder.events[0].kind == "mouse_move"
        assert recorder.events[1].pressed is True

    def test_a_double_click_is_two_pairs(self, recorder: RecordingBackend) -> None:
        MouseClick().run(MouseClick.Params(clicks=2))
        assert [event.pressed for event in recorder.events] == [True, False, True, False]

    def test_the_right_button_is_the_right_button(self, recorder: RecordingBackend) -> None:
        result = MouseClick().run(MouseClick.Params(button=MouseButton.RIGHT))
        assert all(event.button is MouseButton.RIGHT for event in recorder.events)
        assert "правой" in result.message_ru

    def test_a_click_names_the_point_it_moved_to(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        result = MouseClick().run(MouseClick.Params(x=300, y=200, move=True))
        assert "300, 200" in result.message_ru

    @pytest.mark.parametrize("clicks", [0, 4])
    def test_an_impossible_click_count_is_refused(self, clicks: int) -> None:
        with pytest.raises(ValidationError):
            MouseClick.Params(clicks=clicks)

    def test_a_drag_moves_presses_travels_and_releases(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        MouseDrag().run(
            MouseDrag.Params(start=MousePoint(x=0, y=0), end=MousePoint(x=100, y=0), step_px=50)
        )
        kinds = [event.kind for event in recorder.events]
        assert kinds[0] == "mouse_move"
        assert kinds[1] == "mouse_button"
        assert kinds[-1] == "mouse_button"
        assert recorder.events[1].pressed is True
        assert recorder.events[-1].pressed is False

    def test_a_drag_travels_through_intermediate_points(
        self, recorder: RecordingBackend, screen: FakeScreen
    ) -> None:
        """A jump straight to the destination is ignored by half of what is worth dragging."""
        result = MouseDrag().run(
            MouseDrag.Params(start=MousePoint(x=0, y=0), end=MousePoint(x=200, y=0), step_px=50)
        )
        assert result.data["steps"] == 4
        assert len(recorder.points) == 5  # the initial positioning move plus four steps

    def test_a_drag_reports_both_ends(self, recorder: RecordingBackend, screen: FakeScreen) -> None:
        result = MouseDrag().run(
            MouseDrag.Params(start=MousePoint(x=10, y=20), end=MousePoint(x=30, y=40))
        )
        assert result.data["from"] == [10, 20]
        assert result.data["to"] == [30, 40]
        assert result.value == (30, 40)

    def test_the_button_comes_up_even_when_a_step_fails(self, screen: FakeScreen) -> None:
        """A drag left half-finished holds the left button down over the user's desktop."""

        class FailsMidTravel(RecordingBackend):
            def mouse_move(self, x: int, y: int) -> None:
                super().mouse_move(x, y)
                if len(self.points) > 2:
                    raise backend_module.InputBlocked("refused")

        failing = FailsMidTravel()
        set_input_backend(failing)
        try:
            with pytest.raises(backend_module.InputBlocked):
                MouseDrag().run(
                    MouseDrag.Params(
                        start=MousePoint(x=0, y=0), end=MousePoint(x=300, y=0), step_px=50
                    )
                )
        finally:
            set_input_backend(None)
        assert [event.pressed for event in failing.events if event.kind == "mouse_button"] == [
            True,
            False,
        ]

    def test_the_drag_step_comes_from_the_settings(
        self, recorder: RecordingBackend, screen: FakeScreen, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AYRIS_ACTIONS__INPUT__DRAG_STEP_PX", "25")
        from ayris.core import config as config_module

        config_module.reset_config_manager()
        result = MouseDrag().run(
            MouseDrag.Params(start=MousePoint(x=0, y=0), end=MousePoint(x=100, y=0))
        )
        assert result.data["steps"] == 4

    def test_a_wheel_notch_is_one_event(self, recorder: RecordingBackend) -> None:
        """Smooth-scrolling applications animate per notch; one big delta jumps."""
        MouseWheel().run(MouseWheel.Params(clicks=3))
        events = module_events(recorder, "mouse_wheel")
        assert [event.clicks for event in events] == [1, 1, 1]

    def test_scrolling_down_is_negative(self, recorder: RecordingBackend) -> None:
        MouseWheel().run(MouseWheel.Params(clicks=-2))
        assert [event.clicks for event in module_events(recorder, "mouse_wheel")] == [-1, -1]

    def test_a_horizontal_wheel_says_so(self, recorder: RecordingBackend) -> None:
        result = MouseWheel().run(MouseWheel.Params(clicks=1, horizontal=True))
        assert module_events(recorder, "mouse_wheel")[0].horizontal is True
        assert "по горизонтали" in result.message_ru

    def test_zero_notches_send_nothing(self, recorder: RecordingBackend) -> None:
        result = MouseWheel().run(MouseWheel.Params(clicks=0))
        assert recorder.events == []
        assert result.value == 0

    def test_the_wheel_pause_comes_from_the_settings(
        self, recorder: RecordingBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        waits: list[float] = []
        monkeypatch.setattr(mouse_module.time, "sleep", lambda seconds: waits.append(seconds))
        MouseWheel().run(MouseWheel.Params(clicks=3, delay_ms=7))
        assert waits == [0.007, 0.007]

    @pytest.mark.parametrize("clicks", [-1001, 1001])
    def test_an_absurd_wheel_count_is_refused(self, clicks: int) -> None:
        with pytest.raises(ValidationError):
            MouseWheel.Params(clicks=clicks)

    def test_every_mouse_action_is_in_the_registry(self) -> None:
        from ayris.actions.registry import ActionRegistry

        registry = ActionRegistry()
        registry.discover()
        names = set(registry.names)
        assert {"MouseClick", "MouseDrag", "MouseMove", "MouseWheel"} <= names
        assert {"KeyDown", "KeyPress", "KeyUp", "TypeText"} <= names


@pytest.mark.skipif(sys.platform != "win32", reason="настоящий SendInput есть только в Windows")
class TestRealSendInput:
    """The three tests that actually inject, because no fake can check a struct layout.

    A wrong ``INPUT`` size or a mis-declared union member makes ``SendInput``
    return 0 with no error anywhere else, and a recording backend would never
    notice. So this asks Windows itself — while taking care that nothing visible
    happens to whatever window has the focus, since the suite runs on a real
    desktop as often as on a runner: F24 does nothing in any application, U+200B is
    a zero-width space, and the pointer goes back where it was.
    """

    def test_windows_accepts_a_real_keystroke(self) -> None:
        backend = SendInputBackend()
        backend.key_down(KEYS["f24"])
        backend.key_up(KEYS["f24"])

    def test_windows_accepts_a_real_unicode_character(self) -> None:
        """The ``KEYEVENTF_UNICODE`` path: the code unit goes in ``wScan``, ``wVk`` is 0."""
        SendInputBackend().type_text("​")

    def test_the_cursor_really_moves_and_comes_back(self) -> None:
        """``MOUSEEVENTF_ABSOLUTE | VIRTUALDESK`` end to end, against the real pointer."""
        from ayris.actions.input.mouse import WinApiScreen

        screen = WinApiScreen()
        layout = screen.layout()
        origin = screen.cursor()
        target = (layout.virtual.left + 10, layout.virtual.top + 10)
        try:
            SendInputBackend().mouse_move(*normalize_point(*target, layout.virtual))
            moved = screen.cursor()
        finally:
            SendInputBackend().mouse_move(*normalize_point(*origin, layout.virtual))
        assert abs(moved[0] - target[0]) <= 2
        assert abs(moved[1] - target[1]) <= 2


@pytest.mark.xdist_group("clipboard")
@pytest.mark.skipif(sys.platform != "win32", reason="нужен настоящий буфер обмена Windows")
class TestRealClipboardPaste:
    """Paste mode against the real clipboard, since that path used to be a fake.

    Until task 27 this route went through ``pyperclip`` and the test replaced the
    module with four lines of Python — so on a Windows runner the code that talks to
    the clipboard was never executed at all, and the one thing that can actually go
    wrong here (a clipboard another program is holding open) could not happen. Now
    it is the project's own win32 wrapper, and this asks Windows.

    Input stays recorded: a real Ctrl+V would land in whatever window has the focus.
    What matters is that the text is genuinely on the clipboard at the moment the
    combination fires, which is what the watching backend checks.

    ``xdist_group`` is what keeps it green. The clipboard is one lock for the whole
    desktop, so this class and ``test_clipboard.py::TestRealClipboard`` cannot run at
    the same time on two workers — one of them then loses the lock race and reports
    ``OpenClipboard [5]``. The group name puts every real-clipboard test on the same
    worker; the run passes ``--dist=loadgroup`` for that.
    """

    def test_the_text_is_really_on_the_clipboard_when_ctrl_v_fires(
        self, clipboard_or_skip: Callable[[], AbstractContextManager[None]]
    ) -> None:
        set_clipboard(None)
        reset_clipboard()
        real = clipboard_module.get_clipboard()
        seen: list[str] = []

        class WatchingBackend(RecordingBackend):
            def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
                super().key_down(key, scancode=scancode)
                seen.append(real.read().text)

        set_input_backend(WatchingBackend())
        try:
            with clipboard_or_skip():
                real.write_text("прежнее значение")
                TypeText().run(TypeText.Params(text="вставляемый текст", mode=TypeMode.CLIPBOARD))
                assert "вставляемый текст" in seen
                assert real.read().text == "прежнее значение"
        finally:
            set_input_backend(None)
            reset_input_backend()
            clipboard_module._suppressed.clear()
