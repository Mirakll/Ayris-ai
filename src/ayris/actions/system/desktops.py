"""Virtual desktops, over an API Microsoft never wrote down.

Windows has had virtual desktops since 10 and has never shipped a public API for
switching between them. What exists is ``IVirtualDesktopManager``, documented only
far enough to answer «is this window on the current desktop» — and behind it a set
of interfaces whose IIDs change from build to build and whose methods get inserted
in the middle of vtables between releases. Code written against the 21H2 layout
does not merely fail on 24H2, it calls the wrong slot.

So this module trusts three things, in order of how much:

1. **The registry.** ``HKCU\\…\\Explorer\\VirtualDesktops`` holds
   ``VirtualDesktopIDs`` — the GUIDs of every desktop, in order, packed as 16-byte
   blobs — and ``CurrentVirtualDesktop``, the GUID of the one in front. Names, when
   the user set any, sit under ``Desktops\\{guid}``. This is a read, it is stable
   across builds, and it is how :class:`DesktopState` is assembled.
2. **The keyboard.** Win+Ctrl+Left and Win+Ctrl+Right have moved a desktop since
   Windows 10 shipped and are the shell's own binding. Switching is done this way:
   *n* taps in the right direction. Unglamorous, and it survives every update.
3. **COM**, for nothing but a sanity check —
   ``IVirtualDesktopManager.GetWindowDesktopId`` on a real window confirms the
   feature is present at all. No vtable past the documented interface is touched.

The cost of the keyboard is honesty about limits: switching by number needs to know
where we are now, and only the registry can say. When the registry cannot be read,
:class:`SwitchDesktop` says so instead of tapping a hopeful number of times.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from pydantic import Field, model_validator

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils import winapi
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

__all__ = [
    "DesktopBackend",
    "DesktopInfo",
    "DesktopState",
    "DesktopUnavailable",
    "SwitchDesktop",
    "SwitchDirection",
    "WindowsDesktops",
    "get_desktop_backend",
    "guid_from_bytes",
    "parse_desktop_ids",
    "set_desktop_backend",
    "switch_plan",
]

_log = get_logger(__name__)

#: Where the shell keeps the desktop list, under ``HKEY_CURRENT_USER``.
DESKTOPS_KEY: Final = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\VirtualDesktops"

#: Subkey holding the per-desktop user-given names.
DESKTOP_NAMES_KEY: Final = rf"{DESKTOPS_KEY}\Desktops"

#: The first build with virtual desktops — Windows 10 RTM.
MIN_DESKTOP_BUILD: Final = 10_240

#: Sanity limit on how many hotkey taps one command may produce. A user with 30
#: desktops asking for the last one is plausible; 500 taps is a stuck loop.
MAX_TAPS: Final = 32

#: Documented CLSID/IID of the one interface Microsoft admits exists.
CLSID_VIRTUAL_DESKTOP_MANAGER: Final = "{AA509086-5CA9-4C25-8F95-589D3C07B48A}"
IID_VIRTUAL_DESKTOP_MANAGER: Final = "{A5CD92FF-29BE-454C-8D04-D82879FB3F1B}"

#: Length of a packed GUID in ``VirtualDesktopIDs``.
_GUID_BYTES: Final = 16


class DesktopUnavailable(ActionError):
    """Virtual desktops cannot be reached on this machine."""

    default_user_message = "Не получилось разобраться с рабочими столами."


class SwitchDirection(StrEnum):
    """Where to go relative to the desktop in front."""

    NEXT = "next"
    PREVIOUS = "previous"

    @property
    def step(self) -> int:
        """``+1`` or ``-1``, as an offset in the desktop list."""
        return 1 if self is SwitchDirection.NEXT else -1

    @property
    def keys(self) -> tuple[int, ...]:
        """The shell chord that moves one desktop this way."""
        arrow = winapi.VK_RIGHT if self is SwitchDirection.NEXT else winapi.VK_LEFT
        return (winapi.VK_LWIN, winapi.VK_CONTROL, arrow)

    @property
    def title_ru(self) -> str:
        return "вправо" if self is SwitchDirection.NEXT else "влево"


@dataclass(frozen=True, slots=True)
class DesktopInfo:
    """One virtual desktop.

    ``number`` is 1-based, the way the task view shows it and the way a person says
    it. ``name`` is empty unless the user renamed the desktop, in which case
    «переключись на рабочий стол игры» can find it by name.
    """

    number: int
    guid: str
    name: str = ""

    @property
    def label(self) -> str:
        """What to call it out loud."""
        return self.name or f"рабочий стол {self.number}"

    def as_dict(self) -> dict[str, Any]:
        return {"number": self.number, "guid": self.guid, "name": self.name}


@dataclass(frozen=True, slots=True)
class DesktopState:
    """The desktop list as the shell currently sees it."""

    desktops: tuple[DesktopInfo, ...] = ()
    current: int = 0

    @property
    def count(self) -> int:
        return len(self.desktops)

    @property
    def known(self) -> bool:
        """Whether both the list and our position in it were readable."""
        return bool(self.desktops) and self.current > 0

    def at(self, number: int) -> DesktopInfo | None:
        """The desktop with a given 1-based number."""
        if 1 <= number <= self.count:
            return self.desktops[number - 1]
        return None

    def by_name(self, name: str) -> DesktopInfo | None:
        """A desktop the user renamed, matched case-insensitively."""
        wanted = name.strip().casefold()
        if not wanted:
            return None
        for desktop in self.desktops:
            if desktop.name.casefold() == wanted:
                return desktop
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "current": self.current,
            "desktops": [desktop.as_dict() for desktop in self.desktops],
        }


def guid_from_bytes(raw: bytes) -> str:
    """Format one packed 16-byte GUID the way the registry keys spell it.

    The blob is in the Windows mixed-endian layout — first three fields
    little-endian, the last eight bytes as they lie — which is exactly what
    :class:`uuid.UUID`'s ``bytes_le`` means.

    Raises:
        ValueError: the blob is not sixteen bytes long.
    """
    if len(raw) != _GUID_BYTES:
        raise ValueError(f"expected {_GUID_BYTES} bytes, got {len(raw)}")
    return f"{{{uuid.UUID(bytes_le=bytes(raw))!s}}}".upper()


def parse_desktop_ids(blob: bytes) -> tuple[str, ...]:
    """Split ``VirtualDesktopIDs`` into GUIDs, in the shell's own order.

    A trailing partial GUID is dropped rather than raising: the value is written by
    the shell while desktops are being created and closed, and a torn read costs a
    desktop from the list, not the whole command.
    """
    found: list[str] = []
    for offset in range(0, len(blob) - _GUID_BYTES + 1, _GUID_BYTES):
        chunk = blob[offset : offset + _GUID_BYTES]
        try:
            found.append(guid_from_bytes(chunk))
        except ValueError:  # pragma: no cover - length is guaranteed by the range
            break
    return tuple(found)


def switch_plan(state: DesktopState, *, target: int) -> tuple[SwitchDirection, int]:
    """Which way to tap, and how many times, to reach ``target``.

    Desktops do not wrap: Win+Ctrl+Right on the last one does nothing, so the plan
    is a plain difference rather than the shorter way round a ring.

    Raises:
        DesktopUnavailable: the target is outside the list, or the plan would need
            more taps than :data:`MAX_TAPS`.
    """
    if state.at(target) is None:
        raise DesktopUnavailable(
            f"desktop {target} does not exist (have {state.count})",
            user_message=(
                f"Рабочего стола {target} нет — их всего {state.count}."
                if state.count
                else "Не вижу списка рабочих столов."
            ),
        )
    delta = target - state.current
    direction = SwitchDirection.NEXT if delta > 0 else SwitchDirection.PREVIOUS
    taps = abs(delta)
    if taps > MAX_TAPS:
        raise DesktopUnavailable(
            f"switching {taps} desktops at once is not sane",
            user_message="Слишком далеко переключаться, сделаю только рядом стоящие.",
        )
    return direction, taps


class DesktopBackend(Protocol):
    """What this module needs from Windows: read the list, tap a chord."""

    def state(self) -> DesktopState:
        """The desktop list and the current position."""
        ...

    def tap(self, direction: SwitchDirection, times: int) -> None:
        """Send the shell's switch chord ``times`` times."""
        ...

    def supported(self) -> bool:
        """Whether virtual desktops exist on this build at all."""
        ...


class WindowsDesktops:
    """The real backend: registry for reading, hotkeys for switching."""

    def supported(self) -> bool:
        """Whether the feature is present, checked twice over.

        The build number rules out Windows 8 and older; the COM probe rules out a
        stripped image (LTSC without the shell experience pack) where the build says
        yes and the manager is not registered. A ``False`` from the probe is not
        fatal on its own — the registry still answers — so the build check decides
        and the probe only writes to the log.
        """
        if sys.platform != "win32":
            return False
        if winapi.windows_build() < MIN_DESKTOP_BUILD:
            return False
        if not self._probe_com():
            _log.debug("IVirtualDesktopManager недоступен, работаю только через реестр")
        return True

    def state(self) -> DesktopState:
        """Read the desktop list out of ``HKCU``.

        Raises:
            DesktopUnavailable: the keys are not readable — a fresh profile that
                has never opened the task view has no ``VirtualDesktopIDs`` at all.
        """
        try:
            import winreg
        except ImportError as exc:  # pragma: no cover - win32 only
            raise DesktopUnavailable(
                "winreg is unavailable",
                user_message="Рабочие столы доступны только в Windows.",
            ) from exc
        try:
            raw = self._read_value(winreg, DESKTOPS_KEY, "VirtualDesktopIDs")
            current = self._read_value(winreg, DESKTOPS_KEY, "CurrentVirtualDesktop")
        except OSError as exc:
            raise DesktopUnavailable(
                f"cannot read {DESKTOPS_KEY}: {exc}",
                user_message="Не нашла список рабочих столов в реестре.",
            ) from exc
        if not isinstance(raw, bytes):
            raise DesktopUnavailable(
                "VirtualDesktopIDs is not binary",
                user_message="Не нашла список рабочих столов в реестре.",
            )
        guids = parse_desktop_ids(raw)
        names = self._read_names(winreg, guids)
        desktops = tuple(
            DesktopInfo(number=number, guid=guid, name=names.get(guid, ""))
            for number, guid in enumerate(guids, start=1)
        )
        active = guid_from_bytes(current) if isinstance(current, bytes) else ""
        position = next(
            (desktop.number for desktop in desktops if desktop.guid == active),
            0,
        )
        return DesktopState(desktops=desktops, current=position)

    def tap(self, direction: SwitchDirection, times: int) -> None:
        keys = direction.keys
        for _ in range(max(0, times)):
            winapi.press_chord(keys)

    def _read_names(self, winreg: ModuleType, guids: Sequence[str]) -> dict[str, str]:
        """User-given desktop names, for the ones that have any."""
        names: dict[str, str] = {}
        for guid in guids:
            try:
                value = self._read_value(winreg, rf"{DESKTOP_NAMES_KEY}\{guid}", "Name")
            except OSError:
                continue
            if isinstance(value, str) and value.strip():
                names[guid] = value.strip()
        return names

    def _read_value(self, winreg: ModuleType, subkey: str, name: str) -> object:
        """One value from ``HKCU``, with the key closed either way."""
        key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, subkey)
        try:
            value, _kind = winreg.QueryValueEx(key, name)
        finally:
            winreg.CloseKey(key)
        return value

    def _probe_com(self) -> bool:
        """Ask whether the documented interface can be created at all.

        Everything is swallowed: this is a capability check, and ``comtypes`` is an
        optional dependency that raises half a dozen different ways when the object
        is not registered. Only the documented CLSID is touched — no vtable
        spelunking, which is the part that breaks between builds.
        """
        try:
            import comtypes
            import comtypes.client
        except ImportError:
            return False
        try:
            comtypes.CoInitialize()
        except Exception:
            return False
        try:
            manager = comtypes.client.CreateObject(CLSID_VIRTUAL_DESKTOP_MANAGER)
        except Exception:
            return False
        else:
            return manager is not None
        finally:
            try:
                comtypes.CoUninitialize()
            except Exception:
                _log.debug("CoUninitialize не удался", exc_info=True)


_backend: DesktopBackend | None = None


def get_desktop_backend() -> DesktopBackend:
    """The backend in force. Real Windows unless a test replaced it.

    Raises:
        ActionUnavailable: not Windows, or a build without virtual desktops.
    """
    if _backend is not None:
        return _backend
    real = WindowsDesktops()
    if not real.supported():
        raise ActionUnavailable(
            "virtual desktops require Windows 10 or newer",
            user_message="Виртуальные рабочие столы есть только в Windows 10 и новее.",
        )
    return real


def set_desktop_backend(backend: DesktopBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    _backend = backend


@register
class SwitchDesktop(Action):
    """Move to another virtual desktop, by number or by direction."""

    meta: ClassVar = ActionMeta(
        name="SwitchDesktop",
        category=ActionCategory.WINDOWS,
        title_ru="Переключить рабочий стол",
        description_ru="Перейти на рабочий стол по номеру, имени или в сторону",
        timeout_ms=8_000,
    )

    class Params(ActionParams):
        index: int = Field(
            default=0,
            ge=0,
            le=64,
            description="Номер рабочего стола, считая с единицы",
        )
        name: str = Field(
            default="",
            max_length=120,
            title="Имя рабочего стола",
            description="Если пользователь переименовал стол в представлении задач",
        )
        direction: SwitchDirection | None = Field(
            default=None,
            description="Куда перейти относительно текущего",
            json_schema_extra={
                "choices_ru": {str(direction): direction.title_ru for direction in SwitchDirection}
            },
        )

        @model_validator(mode="after")
        def _exactly_one(self) -> SwitchDesktop.Params:
            """Number, name or direction — one of them, and not two.

            «на второй стол вправо» is not a command with a meaning, and guessing
            which half to honour would make the mistake invisible.
            """
            given = sum((self.index > 0, bool(self.name.strip()), self.direction is not None))
            if given == 0:
                raise ValueError("укажите номер, имя или направление")
            if given > 1:
                raise ValueError("укажите что-то одно: номер, имя или направление")
            return self

    def run(self, params: Params) -> ActionResult[DesktopState]:
        backend = get_desktop_backend()
        state = backend.state()

        if params.direction is not None:
            return self._relative(backend, state, params.direction)
        target = self._target_number(state, params)
        if target == state.current:
            desktop = state.at(target)
            label = desktop.label if desktop is not None else f"рабочий стол {target}"
            return ActionResult.done(
                f"Уже на «{label}».",
                value=state,
                data=state.as_dict(),
            )
        direction, taps = switch_plan(state, target=target)
        backend.tap(direction, taps)
        return self._done(backend, state, target=target)

    def _target_number(self, state: DesktopState, params: Params) -> int:
        """The desktop a number or a name points at.

        Raises:
            DesktopUnavailable: the name is unknown, or the current position could
                not be read and an absolute jump therefore has no starting point.
        """
        if params.name.strip():
            desktop = state.by_name(params.name)
            if desktop is None:
                raise DesktopUnavailable(
                    f"no desktop named {params.name!r}",
                    user_message=f"Не нашла рабочий стол «{params.name.strip()}».",
                )
            target = desktop.number
        else:
            target = params.index
        if not state.known:
            raise DesktopUnavailable(
                "current desktop is unknown, cannot switch by number",
                user_message="Не поняла, на каком рабочем столе мы сейчас.",
            )
        return target

    def _relative(
        self,
        backend: DesktopBackend,
        state: DesktopState,
        direction: SwitchDirection,
    ) -> ActionResult[DesktopState]:
        """One step left or right.

        The only case that works without knowing where we are: the chord itself is
        relative. When the list *is* readable, the edge is checked first, because
        the hotkey at the edge does nothing and reporting «переключила» would be a
        lie.
        """
        if state.known:
            target = state.current + direction.step
            if state.at(target) is None:
                edge = "последнем" if direction is SwitchDirection.NEXT else "первом"
                return ActionResult.failed(
                    f"Это уже {edge} рабочий стол.",
                    detail=f"desktop {state.current} has no neighbour {direction}",
                    value=state,
                    data=state.as_dict(),
                )
            backend.tap(direction, 1)
            return self._done(backend, state, target=target)
        backend.tap(direction, 1)
        return ActionResult.done(
            f"Переключила рабочий стол {direction.title_ru}.",
            value=state,
            detail=f"blind switch {direction}",
            data=state.as_dict(),
        )

    def _done(
        self,
        backend: DesktopBackend,
        before: DesktopState,
        *,
        target: int,
    ) -> ActionResult[DesktopState]:
        """Read the state back and report what actually happened.

        The hotkey is asynchronous — the shell animates the switch — so a read
        immediately afterwards can still show the old desktop. That is not treated
        as a failure: the taps were sent, and re-reading is a courtesy that makes
        the result data accurate when it can be.
        """
        try:
            after = backend.state()
        except DesktopUnavailable:
            after = before
        landed = after.at(after.current) or before.at(target)
        label = landed.label if landed is not None else f"рабочий стол {target}"
        data = dict(after.as_dict())
        data["requested"] = target
        return ActionResult.done(
            f"Перешла на «{label}».",
            value=after,
            data=data,
        )


def iter_desktop_labels(state: DesktopState) -> Iterator[str]:
    """Human labels of every desktop, for a debug listing."""
    for desktop in state.desktops:
        marker = "→" if desktop.number == state.current else " "
        yield f"{marker} {desktop.number}. {desktop.label}"
