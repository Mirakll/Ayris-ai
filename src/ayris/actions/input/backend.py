"""Where a synthesised keystroke or click actually leaves Ayris.

Three backends implement one interface, and the reason there is an interface at
all is that the three are not interchangeable in the eyes of the receiver:

* :class:`SendInputBackend` — ``SendInput``, the Windows API. Works in every
  ordinary application and is the default. Two things it cannot do: reach a
  window belonging to an elevated process (UIPI refuses, and the refusal is
  silent unless somebody checks the return value — :class:`InputBlocked` is that
  check), and satisfy a game that reads scan codes through DirectInput and
  ignores anything without one.
* :class:`InterceptionBackend` — the ``interception`` kernel driver, which puts
  the event below the injection flag every low-level hook can see. Optional by
  construction: the driver needs administrator rights and a reboot to install,
  so a missing driver is a line in the log and a fallback to ``SendInput``, not
  a failure. Nothing here ever tries to install it.
* :class:`RecordingBackend` — writes the events down instead of sending them.
  This is what makes the whole layer testable off Windows, and it is also the
  honest way to dry-run a macro the user is editing.

**The interface speaks in already-converted units.** ``mouse_move`` takes
coordinates normalised to 0..65535 over the virtual desktop, not pixels, because
that conversion depends on the monitor layout and belongs in one place
(:mod:`ayris.actions.input.mouse`) rather than in each backend. ``mouse_wheel``
takes notches, not the multiple of 120 the API wants, because a notch is what a
person means. The split is deliberate: everything that requires knowing about
displays happens above this line, and everything below it is one call per event.

**Nothing here sleeps.** Delays between events are pacing, pacing comes from
``[actions.input]`` and from the individual macro block, and a backend that
imposed its own would make the configured value a lie.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.core.events import EventBus

__all__ = [
    "BackendKind",
    "InputBackend",
    "InputBlocked",
    "InputDriverMissing",
    "InputEvent",
    "InterceptionBackend",
    "KeyStroke",
    "MouseButton",
    "RecordingBackend",
    "SendInputBackend",
    "backend_kind",
    "get_input_backend",
    "reset_input_backend",
    "set_input_backend",
    "set_input_bus",
]

_log = get_logger(__name__)

#: Highest value of a normalised absolute coordinate, as ``SendInput`` counts.
ABSOLUTE_MAX: Final = 65_535


class InputBlocked(ActionError):
    """The injected input was refused, and UIPI is nearly always the reason.

    A process at medium integrity cannot send input to a window owned by an
    elevated one. Windows does not report this as an error a user would notice —
    ``SendInput`` returns fewer events than it was given and the keystroke simply
    does not arrive. Saying so out loud is the whole point of this class.
    """

    default_user_message = (
        "Не получилось отправить нажатие: активное окно запущено от администратора. "
        "Запустите Ayris от имени администратора или переключитесь на другое окно."
    )


class InputDriverMissing(ActionUnavailable):
    """The ``interception`` driver was asked for and is not installed."""

    default_user_message = "Драйвер Interception не установлен, работаю через обычный ввод Windows."


class BackendKind(StrEnum):
    """Which implementation is meant. Mirrors ``[actions.input] backend``."""

    SEND_INPUT = "sendinput"
    INTERCEPTION = "interception"

    @property
    def title_ru(self) -> str:
        if self is BackendKind.SEND_INPUT:
            return "обычный ввод Windows"
        return "драйвер Interception"


class MouseButton(StrEnum):
    """The five buttons a mouse event can be about."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"
    X1 = "x1"
    X2 = "x2"

    @property
    def title_ru(self) -> str:
        return _BUTTON_TITLES[self]

    @property
    def instrumental_ru(self) -> str:
        """The same name in the case a sentence needs: «щёлкнул правой кнопкой».

        Separate from :attr:`title_ru` because that one goes into a dropdown, where
        the nominative is what a list of options should read like.
        """
        return _BUTTON_INSTRUMENTAL[self]


_BUTTON_TITLES: Final[dict[MouseButton, str]] = {
    MouseButton.LEFT: "левая",
    MouseButton.RIGHT: "правая",
    MouseButton.MIDDLE: "средняя",
    MouseButton.X1: "боковая 1",
    MouseButton.X2: "боковая 2",
}

_BUTTON_INSTRUMENTAL: Final[dict[MouseButton, str]] = {
    MouseButton.LEFT: "левой кнопкой",
    MouseButton.RIGHT: "правой кнопкой",
    MouseButton.MIDDLE: "средней кнопкой",
    MouseButton.X1: "боковой кнопкой 1",
    MouseButton.X2: "боковой кнопкой 2",
}


@dataclass(frozen=True, slots=True)
class KeyStroke:
    """One key, described the way both injection paths need it.

    ``vk`` is the virtual key code and ``name`` the canonical Ayris spelling of
    it (``"ctrl"``, ``"f5"``, ``"left"``). ``extended`` marks the keys whose scan
    code carries the ``E0`` prefix — the arrows, Insert, Delete, the right Alt
    and Ctrl, the numpad Enter. Getting that flag wrong is what makes a game read
    Home as Numpad-7.

    ``scan`` is normally ``0``, meaning "ask ``MapVirtualKeyW`` when sending".
    A non-zero value overrides that: :func:`~ayris.actions.input.keys.char_key`
    fills it in for a character typed through the current layout.
    """

    name: str
    vk: int
    extended: bool = False
    scan: int = 0

    def with_scan(self, scan: int) -> KeyStroke:
        """Copy with an explicit scan code."""
        return KeyStroke(name=self.name, vk=self.vk, extended=self.extended, scan=scan)


@dataclass(frozen=True, slots=True)
class InputEvent:
    """One event as :class:`RecordingBackend` wrote it down.

    ``kind`` is the method that was called; the remaining fields are whichever of
    its arguments applied. Compared whole in the tests, so a stray argument is a
    failure rather than something the assertion happens not to look at.
    """

    kind: str
    key: str = ""
    vk: int = 0
    scan: int = 0
    text: str = ""
    x: int = 0
    y: int = 0
    button: MouseButton | None = None
    pressed: bool = False
    clicks: int = 0
    horizontal: bool = False
    scancode: bool = False


class InputBackend(ABC):
    """How Ayris sends keystrokes and mouse events.

    Implementations are stateless as far as the caller is concerned: which keys
    are currently held is tracked one level up, in
    :mod:`ayris.actions.input.keys`, so that the guarantee «everything held gets
    released» survives a backend being swapped mid-macro.
    """

    #: Which backend this is, for the log and for ``ActionResult.data``.
    kind: ClassVar[BackendKind]

    @abstractmethod
    def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
        """Press one key and leave it down.

        Args:
            key: Which key.
            scancode: Send the scan code instead of the virtual key. Required by
                applications that read the keyboard through DirectInput, which
                ignore an event whose ``wScan`` is empty.
        """

    @abstractmethod
    def key_up(self, key: KeyStroke, *, scancode: bool = False) -> None:
        """Release one key."""

    @abstractmethod
    def type_text(self, text: str) -> None:
        """Type ``text`` as characters, independently of the active layout.

        The Unicode path: every character goes as its UTF-16 code units, so
        Cyrillic and emoji arrive with a US layout active and nothing has to
        switch layouts behind the user's back.
        """

    @abstractmethod
    def mouse_move(self, x: int, y: int) -> None:
        """Move the pointer to an absolute position.

        Args:
            x: Horizontal position, already normalised to ``0..65535`` across the
                whole virtual desktop.
            y: Vertical position, same scale.
        """

    @abstractmethod
    def mouse_button(self, button: MouseButton, *, pressed: bool) -> None:
        """Press or release one mouse button at the current position."""

    @abstractmethod
    def mouse_wheel(self, clicks: int, *, horizontal: bool = False) -> None:
        """Turn the wheel by ``clicks`` notches; negative is down or left."""


class SendInputBackend(InputBackend):
    """The default: ``SendInput`` through :mod:`ayris.utils.winapi`.

    Every method is one API call, and every call checks how many events Windows
    accepted. Zero means the injection was refused — see :class:`InputBlocked`.
    """

    kind: ClassVar[BackendKind] = BackendKind.SEND_INPUT

    def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
        self._send_key(key, scancode=scancode, release=False)

    def key_up(self, key: KeyStroke, *, scancode: bool = False) -> None:
        self._send_key(key, scancode=scancode, release=True)

    def type_text(self, text: str) -> None:
        if not text:
            return
        self._guard(lambda: winapi.send_unicode_text(text), f"type {len(text)} chars")

    def mouse_move(self, x: int, y: int) -> None:
        flags = (
            winapi.MOUSEEVENTF_MOVE | winapi.MOUSEEVENTF_ABSOLUTE | winapi.MOUSEEVENTF_VIRTUALDESK
        )
        self._guard(
            lambda: winapi.send_mouse_event(flags=flags, dx=x, dy=y),
            f"move to {x},{y}",
        )

    def mouse_button(self, button: MouseButton, *, pressed: bool) -> None:
        flags, data = _button_flags(button, pressed=pressed)
        self._guard(
            lambda: winapi.send_mouse_event(flags=flags, data=data),
            f"{button} {'down' if pressed else 'up'}",
        )

    def mouse_wheel(self, clicks: int, *, horizontal: bool = False) -> None:
        if not clicks:
            return
        flags = winapi.MOUSEEVENTF_HWHEEL if horizontal else winapi.MOUSEEVENTF_WHEEL
        self._guard(
            lambda: winapi.send_mouse_event(flags=flags, data=clicks * winapi.WHEEL_DELTA),
            f"wheel {clicks}",
        )

    def _send_key(self, key: KeyStroke, *, scancode: bool, release: bool) -> None:
        """One half of a keystroke, as a virtual key or as a scan code."""
        flags = winapi.KEYEVENTF_KEYUP if release else 0
        scan = key.scan or self._scan_of(key)
        if key.extended:
            flags |= winapi.KEYEVENTF_EXTENDEDKEY
        if scancode and scan:
            flags |= winapi.KEYEVENTF_SCANCODE
            vk = 0
        else:
            vk = key.vk
        self._guard(
            lambda: winapi.send_key_events([(vk, scan, flags)]),
            f"{key.name} {'up' if release else 'down'}",
        )

    def _scan_of(self, key: KeyStroke) -> int:
        """Scan code of a virtual key, or ``0`` when Windows has none for it.

        Media and browser keys genuinely have no scan code; that is not an error,
        it only means the scan-code mode cannot express them and the virtual key
        is sent instead.
        """
        try:
            return winapi.map_virtual_key(key.vk)
        except winapi.WinApiError:
            return 0

    def _guard(self, call: Any, what: str) -> None:
        """Run one injection and turn a refusal into :class:`InputBlocked`."""
        try:
            sent = int(call())
        except winapi.WinApiError as exc:
            raise InputBlocked(f"SendInput refused {what}: {exc}") from exc
        if sent == 0:
            raise InputBlocked(f"SendInput injected nothing for {what}")


class InterceptionBackend(InputBackend):
    """The ``interception`` driver, for games that ignore injected input.

    The wrapper is imported inside :meth:`create` and nowhere else, so a machine
    without the package — every CI runner, and every user who never installed the
    driver — is a fallback rather than a collection error. :meth:`create` is also
    the only place that decides whether the driver is usable: it captures the
    devices, which is what fails when the kernel side is absent.

    The wrapper speaks in key *names* rather than virtual keys, and its names are
    pyautogui's. :data:`_INTERCEPTION_NAMES` maps the few Ayris spellings that
    differ; the rest match, and an unknown one falls back to ``SendInput`` for
    that keystroke instead of failing the macro.
    """

    kind: ClassVar[BackendKind] = BackendKind.INTERCEPTION

    def __init__(self, module: Any, *, fallback: InputBackend | None = None) -> None:
        self._module = module
        self._fallback = fallback or SendInputBackend()

    @classmethod
    def create(cls) -> InterceptionBackend:
        """Build the backend, or explain why the driver cannot be used.

        Raises:
            InputDriverMissing: the package is absent, the driver is not
                installed, or the devices could not be captured.
        """
        if sys.platform != "win32":
            raise InputDriverMissing("interception is Windows-only")
        try:
            import interception
        except ImportError as exc:
            raise InputDriverMissing(f"interception-python is not installed: {exc}") from exc
        capture = getattr(interception, "auto_capture_devices", None)
        if capture is None:  # pragma: no cover - a wrapper this old is not shipped
            raise InputDriverMissing("interception has no auto_capture_devices")
        try:
            capture(keyboard=True, mouse=True)
        except Exception as exc:
            raise InputDriverMissing(f"interception driver is unusable: {exc}") from exc
        return cls(interception)

    def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
        name = self._name_of(key)
        if name is None:
            self._fallback.key_down(key, scancode=scancode)
            return
        self._module.key_down(name)

    def key_up(self, key: KeyStroke, *, scancode: bool = False) -> None:
        name = self._name_of(key)
        if name is None:
            self._fallback.key_up(key, scancode=scancode)
            return
        self._module.key_up(name)

    def type_text(self, text: str) -> None:
        """Always the Unicode path.

        The driver types by scan code and therefore through the active layout,
        which cannot produce Cyrillic under a US layout — the exact thing
        :class:`~ayris.actions.input.keys.TypeText` promises. Text goes through
        ``SendInput``; the games that need the driver do not take dictation.
        """
        self._fallback.type_text(text)

    def mouse_move(self, x: int, y: int) -> None:
        """Move by way of the fallback: the driver speaks in pixels.

        ``interception.move_to`` takes screen pixels, and what arrives here is
        already normalised to 0..65535 for ``SendInput``. Converting back would
        mean re-deriving the monitor layout inside a backend, which is the one
        thing this interface keeps out. Pointer position is not what the driver
        is installed for.
        """
        self._fallback.mouse_move(x, y)

    def mouse_button(self, button: MouseButton, *, pressed: bool) -> None:
        name = _INTERCEPTION_BUTTONS.get(button)
        if name is None:
            self._fallback.mouse_button(button, pressed=pressed)
            return
        if pressed:
            self._module.mouse_down(name)
        else:
            self._module.mouse_up(name)

    def mouse_wheel(self, clicks: int, *, horizontal: bool = False) -> None:
        """One notch at a time; the driver takes a direction, not an amount."""
        if horizontal or not clicks:
            self._fallback.mouse_wheel(clicks, horizontal=horizontal)
            return
        direction = "up" if clicks > 0 else "down"
        for _ in range(abs(clicks)):
            self._module.scroll(direction)

    def _name_of(self, key: KeyStroke) -> str | None:
        """The wrapper's name for a key, or ``None`` when it has none."""
        name = _INTERCEPTION_NAMES.get(key.name, key.name)
        keys: Sequence[str] = getattr(self._module, "KEYBOARD_KEYS", ())
        if keys and name not in keys:
            _log.debug("interception не знает клавишу %s, отправляю через SendInput", key.name)
            return None
        return name


class RecordingBackend(InputBackend):
    """Writes events down instead of sending them.

    Two jobs. In the tests it is the whole of Windows: the order of down and up,
    the normalised coordinates, the interpolated steps of a drag and the release
    of held keys are all assertions about :attr:`events`. In the application it
    is the dry run — a macro can be stepped through in the editor without the
    pointer jumping across the user's screen.
    """

    kind: ClassVar[BackendKind] = BackendKind.SEND_INPUT

    def __init__(self) -> None:
        self.events: list[InputEvent] = []

    def key_down(self, key: KeyStroke, *, scancode: bool = False) -> None:
        self._add("key_down", key, scancode=scancode)

    def key_up(self, key: KeyStroke, *, scancode: bool = False) -> None:
        self._add("key_up", key, scancode=scancode)

    def type_text(self, text: str) -> None:
        self.events.append(InputEvent(kind="type_text", text=text))

    def mouse_move(self, x: int, y: int) -> None:
        self.events.append(InputEvent(kind="mouse_move", x=x, y=y))

    def mouse_button(self, button: MouseButton, *, pressed: bool) -> None:
        self.events.append(InputEvent(kind="mouse_button", button=button, pressed=pressed))

    def mouse_wheel(self, clicks: int, *, horizontal: bool = False) -> None:
        self.events.append(InputEvent(kind="mouse_wheel", clicks=clicks, horizontal=horizontal))

    # -- what the tests read ---------------------------------------------- #

    @property
    def keys(self) -> tuple[str, ...]:
        """Key events as ``"+ctrl"`` / ``"-ctrl"``, in order. For readable asserts."""
        return tuple(
            f"{'+' if event.kind == 'key_down' else '-'}{event.key}"
            for event in self.events
            if event.kind in {"key_down", "key_up"}
        )

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        """Every normalised position the pointer was moved to, in order."""
        return tuple((event.x, event.y) for event in self.events if event.kind == "mouse_move")

    @property
    def typed(self) -> str:
        """Everything :meth:`type_text` was given, concatenated."""
        return "".join(event.text for event in self.events if event.kind == "type_text")

    def clear(self) -> None:
        self.events.clear()

    def _add(self, kind: str, key: KeyStroke, *, scancode: bool) -> None:
        self.events.append(
            InputEvent(
                kind=kind,
                key=key.name,
                vk=key.vk,
                scan=key.scan,
                pressed=kind == "key_down",
                scancode=scancode,
            )
        )


@dataclass(slots=True)
class _Selection:
    """Which backend is in force, and why. Module state, reset by the tests."""

    backend: InputBackend | None = None
    override: InputBackend | None = None
    warned: set[str] = field(default_factory=set)
    bus: EventBus | None = None


_state = _Selection()


def set_input_bus(bus: EventBus | None) -> None:
    """Where to send the «no driver, using ordinary input» notice.

    A module-level setter rather than a constructor argument because the backend
    is chosen lazily, deep inside an action, and threading a bus down to there
    would mean an argument on every keyboard and mouse call. Without a bus the
    fallback is still logged — the notification is the part that needs a UI, so
    the console entry points can simply leave this alone. Called by whoever hands
    the action registry to the application; until then it is only the seam.
    """
    _state.bus = bus


def backend_kind() -> BackendKind:
    """What ``[actions.input] backend`` asks for, defaulting to ``SendInput``."""
    from ayris.core.config import get_settings

    try:
        configured = get_settings().actions.input.backend
    except Exception:
        return BackendKind.SEND_INPUT
    try:
        return BackendKind(configured)
    except ValueError:  # pragma: no cover - the Literal already refuses this
        return BackendKind.SEND_INPUT


def get_input_backend() -> InputBackend:
    """The backend in force, building it on first use.

    The configured choice is honoured when it can be: asking for
    ``interception`` on a machine without the driver logs the reason once and
    returns ``SendInput``, because a macro that types text is not improved by
    failing over a driver it does not need. The instance is cached — capturing
    the driver's devices is not free — and :func:`reset_input_backend` drops it
    when the setting changes.
    """
    if _state.override is not None:
        return _state.override
    if _state.backend is not None:
        return _state.backend
    wanted = backend_kind()
    backend: InputBackend = SendInputBackend()
    if wanted is BackendKind.INTERCEPTION:
        try:
            backend = InterceptionBackend.create()
        except InputDriverMissing as exc:
            _warn_once(f"interception: {exc.technical}", user_message=exc.user_message)
    _state.backend = backend
    return backend


def set_input_backend(backend: InputBackend | None) -> None:
    """Install a backend, or restore the configured one with ``None``. Test seam."""
    _state.override = backend


def reset_input_backend() -> None:
    """Forget the cached instance, so the next call re-reads the setting."""
    _state.backend = None
    _state.warned.clear()


def _warn_once(message: str, *, user_message: str = "") -> None:
    """Log a fallback once per reason, and tell the user once as well.

    A macro in a loop would otherwise fill the log with one repeated line and
    stack a dozen identical balloons — which is why the reason, not the call, is
    what gets counted.
    """
    if message in _state.warned:
        return
    _state.warned.add(message)
    _log.warning("%s — работаю через SendInput", message)
    bus = _state.bus
    if bus is None or not user_message:
        return
    from ayris.core.events import NotificationRequested

    try:
        bus.publish(
            NotificationRequested(
                title="Ввод",
                message=user_message,
                level="warning",
                timeout_ms=6000,
            )
        )
    except Exception as exc:  # pragma: no cover - the bus swallows its own
        _log.debug("не удалось показать уведомление о бэкенде ввода: %s", exc)


def _button_flags(button: MouseButton, *, pressed: bool) -> tuple[int, int]:
    """``SendInput`` flags and ``mouseData`` for one button change."""
    if button is MouseButton.LEFT:
        return (winapi.MOUSEEVENTF_LEFTDOWN if pressed else winapi.MOUSEEVENTF_LEFTUP), 0
    if button is MouseButton.RIGHT:
        return (winapi.MOUSEEVENTF_RIGHTDOWN if pressed else winapi.MOUSEEVENTF_RIGHTUP), 0
    if button is MouseButton.MIDDLE:
        return (winapi.MOUSEEVENTF_MIDDLEDOWN if pressed else winapi.MOUSEEVENTF_MIDDLEUP), 0
    data = 1 if button is MouseButton.X1 else 2
    return (winapi.MOUSEEVENTF_XDOWN if pressed else winapi.MOUSEEVENTF_XUP), data


#: Ayris spellings that differ from the wrapper's pyautogui-style names.
_INTERCEPTION_NAMES: Final[dict[str, str]] = {
    "win": "win",
    "escape": "esc",
    "return": "enter",
    "pageup": "pageup",
    "pagedown": "pagedown",
    "capslock": "capslock",
    "printscreen": "printscreen",
    "scrolllock": "scrolllock",
    "backquote": "`",
    "minus": "-",
    "equal": "=",
    "bracketleft": "[",
    "bracketright": "]",
    "backslash": "\\",
    "semicolon": ";",
    "quote": "'",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "space": " ",
}

#: Ayris buttons as the wrapper names them; ``None`` means it cannot express one.
_INTERCEPTION_BUTTONS: Final[dict[MouseButton, str]] = {
    MouseButton.LEFT: "left",
    MouseButton.RIGHT: "right",
    MouseButton.MIDDLE: "middle",
    MouseButton.X1: "mouse4",
    MouseButton.X2: "mouse5",
}
