"""Sound devices: listing them, and changing which one is «по умолчанию».

Two halves that Microsoft treats very differently. **Listing** is documented and
stable: ``IMMDeviceEnumerator`` has answered the same way since Vista, and every
endpoint carries an id, a friendly name, a state and a property store.
**Switching** is not documented at all. There is no public API for «make these
headphones the default» — the Sound applet does it through ``IPolicyConfig``, an
interface Microsoft never wrote down, whose CLSID differs between Windows
versions and whose vtable has methods inserted in the middle between releases.

So the two are kept apart on purpose. Enumeration is trusted and always
available. Switching is one method behind one CLSID, tried once, and when the
build does not answer to it the action says so in Russian instead of dying in a
``COMError`` — :class:`PolicyUnavailable` exists for exactly that. Nothing else
in Ayris touches ``IPolicyConfig``.

This module is also where the COM plumbing for the whole audio subsystem lives:
:func:`initialize_com` and :func:`device_enumerator` are what
:mod:`ayris.actions.system.audio` reaches for too, so that one thread-local cache
serves both and one call to :func:`invalidate_devices` empties it. Every ``except``
around a COM call here lists :data:`ayris.utils.winapi.COM_ERRORS` rather than
``OSError``, because ``COMError`` is not one — with a bare ``except OSError`` all
the Russian messages below were unreachable code.

**Nothing survives a hot-plug except the enumerator.** Endpoint pointers go stale
the moment a USB headset is unplugged, and a cached ``IMMDevice`` then fails in
ways that look like a bug in the action. Only the enumerator — which outlives any
device — is kept, and only per thread, because a COM pointer created on one
thread must not be used from another and the registry runs actions on a pool. On
top of that :func:`invalidate_devices` drops even that much, and
``OnDefaultDeviceChanged`` calls it: see :class:`DeviceWatcher`.

The device list on a real machine is mostly rubbish — this development box
reports 41 render endpoints, of which two are ``Active`` and the rest are
monitors that were plugged in once. :attr:`DeviceState.usable` is the filter, and
listing hides the rest unless asked.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final, Protocol

from pydantic import Field

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import register
from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils.logger import get_logger
from ayris.utils.winapi import COM_ERRORS

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "CLSID_POLICY_CONFIG",
    "MAX_LISTED_DEVICES",
    "AudioDevice",
    "DeviceBackend",
    "DeviceKind",
    "DeviceNotFound",
    "DeviceState",
    "DeviceUnavailable",
    "DeviceWatcher",
    "PolicyUnavailable",
    "SetAudioDevice",
    "WasapiDevices",
    "audio_utilities",
    "default_device",
    "device_enumerator",
    "endpoint_volume",
    "find_device",
    "get_device_backend",
    "initialize_com",
    "invalidate_devices",
    "list_audio_devices",
    "match_devices",
    "set_device_backend",
    "start_device_watcher",
    "stop_device_watcher",
]

_log = get_logger(__name__)

#: How many devices one listing returns. A machine with a docking station and a
#: history of monitors reports dozens; nobody reads past the first handful.
MAX_LISTED_DEVICES: Final = 32

#: The undocumented interface behind the Sound applet's «Set as default». This
#: CLSID answers on Windows 10 and 11; earlier ones used a different one, and a
#: future build may drop it entirely, which is why it is tried and not assumed.
CLSID_POLICY_CONFIG: Final = "{870af99c-171d-4f9e-af0d-e63df40c2bc9}"

#: ``IPolicyConfig`` itself. Only ``SetDefaultEndpoint`` is ever called, and it
#: is the eleventh slot — hence the ten placeholders in the declaration below.
IID_POLICY_CONFIG: Final = "{f8679f50-850a-41cf-9c72-430f290290c8}"

#: ``ERole`` values. All three are set together: Windows keeps a separate default
#: for playback, for multimedia and for calls, and moving only one of them is how
#: «переключила на наушники» ends with the ringtone still in the speakers.
ROLE_CONSOLE: Final = 0
ROLE_MULTIMEDIA: Final = 1
ROLE_COMMUNICATIONS: Final = 2
_ROLES: Final[tuple[int, ...]] = (ROLE_CONSOLE, ROLE_MULTIMEDIA, ROLE_COMMUNICATIONS)

#: ``DEVICE_STATE.MASK_ALL`` — enumerate everything and filter here, so that a
#: disabled device can still be named in an error message.
_STATE_MASK_ALL: Final = 15


class DeviceNotFound(ActionError):
    """No sound device matches the name that was said."""

    default_user_message = "Не нашла такое звуковое устройство."


class DeviceUnavailable(ActionUnavailable):
    """The device list cannot be read, or there is nothing in it.

    The normal case on a machine with no sound card at all — including the CI
    runner, where this is the expected answer rather than a failure. An
    :class:`~ayris.core.errors.ActionUnavailable` and not a plain action error
    because that is what it means: not «попробуйте иначе», but «здесь этого нет».
    """

    default_user_message = "Не нашла звуковые устройства."


class PolicyUnavailable(ActionUnavailable):
    """This Windows build does not answer to ``IPolicyConfig``."""

    default_user_message = (
        "Эта сборка Windows не даёт переключать устройство по умолчанию — "
        "поменяйте его в параметрах звука."
    )


class DeviceKind(StrEnum):
    """Which direction a device carries sound in."""

    OUTPUT = "output"
    INPUT = "input"

    @property
    def title_ru(self) -> str:
        """How the direction is named in the settings window."""
        return "Вывод звука" if self is DeviceKind.OUTPUT else "Запись звука"

    @property
    def noun_ru(self) -> str:
        """The direction as it appears mid-sentence: «устройство вывода»."""
        return "устройство вывода" if self is DeviceKind.OUTPUT else "устройство ввода"

    @property
    def flow(self) -> int:
        """``EDataFlow``: ``eRender`` is 0 and ``eCapture`` is 1."""
        return 0 if self is DeviceKind.OUTPUT else 1

    @property
    def missing_ru(self) -> str:
        """What to say when there is no such device on this machine at all."""
        return f"Не нашла {self.noun_ru}."


class DeviceState(StrEnum):
    """Whether a listed endpoint is something sound can actually go to."""

    ACTIVE = "active"
    DISABLED = "disabled"
    NOT_PRESENT = "not_present"
    UNPLUGGED = "unplugged"
    UNKNOWN = "unknown"

    @property
    def title_ru(self) -> str:
        """State as the settings window shows it."""
        return _STATE_TITLES[self]

    @property
    def usable(self) -> bool:
        """Whether this device can be made the default right now."""
        return self is DeviceState.ACTIVE

    @classmethod
    def from_wasapi(cls, value: int) -> DeviceState:
        """Map a ``DEVICE_STATE`` bit onto the enum, unknown bits included."""
        return _WASAPI_STATES.get(value, cls.UNKNOWN)


_STATE_TITLES: Final[dict[DeviceState, str]] = {
    DeviceState.ACTIVE: "работает",
    DeviceState.DISABLED: "отключено",
    DeviceState.NOT_PRESENT: "не подключено",
    DeviceState.UNPLUGGED: "вынуто из гнезда",
    DeviceState.UNKNOWN: "состояние неизвестно",
}

_WASAPI_STATES: Final[dict[int, DeviceState]] = {
    1: DeviceState.ACTIVE,
    2: DeviceState.DISABLED,
    4: DeviceState.NOT_PRESENT,
    8: DeviceState.UNPLUGGED,
}


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """One sound endpoint, described well enough to name it out loud."""

    device_id: str
    name: str = ""
    kind: DeviceKind = DeviceKind.OUTPUT
    state: DeviceState = DeviceState.ACTIVE
    is_default: bool = False

    @property
    def usable(self) -> bool:
        """Whether sound can be routed here right now."""
        return self.state.usable

    @property
    def label(self) -> str:
        """Friendly name if the driver gave one, the raw id otherwise."""
        return self.name or self.device_id

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready form, for :attr:`ActionResult.data` and the audit trail."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "kind": str(self.kind),
            "state": str(self.state),
            "is_default": self.is_default,
        }


def match_devices(
    devices: Iterable[AudioDevice],
    query: str,
    *,
    usable_only: bool = True,
) -> list[AudioDevice]:
    """Devices whose name answers to ``query``, best match first.

    Four grades of match, in order: the whole name, a name that starts with the
    query, a name that contains it, and a name that contains every word of it
    somewhere. The last one is what makes «наушники realtek» find «Наушники
    (Realtek(R) Audio)» — the driver puts the model in brackets and a plain
    substring search would miss it.

    Case-insensitive by :meth:`str.casefold`, which is the right fold for
    Russian: half of the endpoint names on a Russian Windows are Cyrillic.
    """
    needle = " ".join(query.casefold().split())
    if not needle:
        return []
    words = needle.split()
    scored: list[tuple[int, int, str, AudioDevice]] = []
    for device in devices:
        if usable_only and not device.usable:
            continue
        name = " ".join(device.name.casefold().split())
        score = _match_score(name, needle, words)
        if score:
            scored.append((-score, 0 if device.usable else 1, name, device))
    scored.sort(key=lambda item: item[:3])
    return [device for *_, device in scored]


def _match_score(name: str, needle: str, words: Sequence[str]) -> int:
    """How well one name answers a query: 4 exact … 1 all words, 0 no match."""
    if not name:
        return 0
    if name == needle:
        return 4
    if name.startswith(needle):
        return 3
    if needle in name:
        return 2
    if len(words) > 1 and all(word in name for word in words):
        return 1
    return 0


def find_device(
    devices: Iterable[AudioDevice],
    query: str,
    *,
    kind: DeviceKind = DeviceKind.OUTPUT,
) -> AudioDevice:
    """The one device ``query`` names.

    Raises:
        DeviceNotFound: nothing matches, or only devices that are unplugged or
            switched off do — and saying «нашла, но оно отключено» is more use
            than «не нашла».
    """
    listed = list(devices)
    found = match_devices(listed, query)
    if found:
        return found[0]
    unusable = match_devices(listed, query, usable_only=False)
    if unusable:
        device = unusable[0]
        raise DeviceNotFound(
            f"device {device.device_id!r} matched {query!r} but is {device.state}",
            user_message=f"Нашла «{device.label}», но оно сейчас {device.state.title_ru}.",
        )
    raise DeviceNotFound(
        f"no {kind} device matches {query!r}",
        user_message=f"Не нашла {kind.noun_ru} с названием «{query}».",
    )


# --------------------------------------------------------------------------- #
# COM plumbing, shared with ayris.actions.system.audio
# --------------------------------------------------------------------------- #

#: Per-thread COM objects. Thread-local and not process-wide because a pointer
#: obtained on one thread is not valid on another, and the registry runs actions
#: on a worker pool.
_com = threading.local()


def initialize_com() -> None:
    """``CoInitialize`` for the calling thread. Idempotent, reference-counted.

    Raises:
        ActionUnavailable: ``comtypes`` is not installed — that is, this is not
            Windows, since it is pinned with ``sys_platform == 'win32'``.
    """
    try:
        import comtypes
    except ImportError as exc:
        raise ActionUnavailable(
            f"comtypes is unavailable: {exc}",
            user_message="Управление звуком работает только в Windows.",
        ) from exc
    try:
        comtypes.CoInitialize()
    except COM_ERRORS as exc:  # pragma: no cover - only on a hostile apartment
        _log.debug("CoInitialize вернул %s", exc)


def audio_utilities() -> Any:
    """``pycaw.utils.AudioUtilities``, imported on first use.

    Every ``pycaw`` import in this package is inside a function. At module level
    it would break test collection in the sandbox and on the Linux CI job —
    ``pycaw`` is pinned with ``sys_platform == 'win32'`` and is deliberately
    absent from ``requirements-ci.txt`` — and a missing optional dependency has
    to look like :class:`ActionUnavailable`, not like a broken test suite.

    Raises:
        ActionUnavailable: ``pycaw`` is not installed.
    """
    try:
        from pycaw.utils import AudioUtilities
    except ImportError as exc:
        raise ActionUnavailable(
            f"pycaw is unavailable: {exc}",
            user_message="Управление звуком работает только в Windows.",
        ) from exc
    return AudioUtilities


def device_enumerator() -> Any:
    """This thread's ``IMMDeviceEnumerator``, created once and reused.

    The only COM object worth caching: it survives a hot-plug, while every
    ``IMMDevice`` behind it does not.

    Raises:
        ActionUnavailable: ``pycaw`` or ``comtypes`` is missing.
        DeviceUnavailable: the sound subsystem itself refused.
    """
    cached = getattr(_com, "enumerator", None)
    if cached is not None:
        return cached
    initialize_com()
    try:
        enumerator = audio_utilities().GetDeviceEnumerator()
    except COM_ERRORS as exc:
        raise DeviceUnavailable(
            f"cannot create IMMDeviceEnumerator: {exc}",
            user_message="Не смогла добраться до звуковой подсистемы Windows.",
        ) from exc
    if enumerator is None:  # pragma: no cover - pycaw raises instead
        raise DeviceUnavailable(
            "IMMDeviceEnumerator is unavailable",
            user_message="Не смогла добраться до звуковой подсистемы Windows.",
        )
    _com.enumerator = enumerator
    return enumerator


def endpoint_volume(
    kind: DeviceKind = DeviceKind.OUTPUT,
    device_id: str = "",
) -> tuple[Any, str]:
    """``(IAudioEndpointVolume, friendly name)`` for one endpoint.

    ``device_id`` empty means «whatever is the default right now», resolved on
    every call rather than remembered: between two actions the user may well have
    pulled the headphones out.

    Raises:
        DeviceUnavailable: there is no such endpoint — including the case of a
            machine with no sound card at all, which is what the CI runner is.
    """
    enumerator = device_enumerator()
    utilities = audio_utilities()
    try:
        raw = (
            enumerator.GetDevice(device_id)
            if device_id
            else enumerator.GetDefaultAudioEndpoint(kind.flow, ROLE_MULTIMEDIA)
        )
        described = utilities.CreateDevice(raw)
    except COM_ERRORS as exc:
        raise DeviceUnavailable(
            f"no {kind} endpoint {device_id or '(default)'}: {exc}",
            user_message=kind.missing_ru,
        ) from exc
    if described is None:  # pragma: no cover - pycaw raises instead
        raise DeviceUnavailable(
            f"no {kind} endpoint {device_id or '(default)'}",
            user_message=kind.missing_ru,
        )
    try:
        volume = described.EndpointVolume
    except COM_ERRORS as exc:
        raise DeviceUnavailable(
            f"cannot activate IAudioEndpointVolume on {described.id!r}: {exc}",
            user_message=f"Не смогла управлять громкостью: {kind.noun_ru} не отвечает.",
        ) from exc
    return volume, str(described.FriendlyName or "")


def invalidate_devices() -> None:
    """Drop cached COM objects. Idempotent and safe from any thread.

    Called from ``OnDefaultDeviceChanged`` — that is, from a COM callback on a
    thread we do not own — so it does nothing but clear a dictionary. It clears
    the calling thread's cache only, which is all a thread can safely do; other
    threads re-resolve their endpoints on every call anyway, so the worst a stale
    enumerator costs them is one failed call.
    """
    _com.__dict__.clear()


class DeviceBackend(Protocol):
    """Everything this module needs from WASAPI. Faked wholesale in the tests."""

    def list_devices(self, kind: DeviceKind) -> list[AudioDevice]:
        """Every endpoint of one direction, whatever state it is in."""
        ...

    def default_device(self, kind: DeviceKind) -> AudioDevice:
        """The endpoint sound currently goes to, or comes from."""
        ...

    def set_default(self, device_id: str, kind: DeviceKind) -> None:
        """Make one endpoint the default for all three roles."""
        ...


class WasapiDevices:
    """The real backend, over ``pycaw``."""

    def supported(self) -> bool:
        """Whether WASAPI can be reached at all in this process."""
        if sys.platform != "win32":
            return False
        try:
            import comtypes  # noqa: F401
            import pycaw.utils  # noqa: F401
        except ImportError:
            _log.debug("pycaw недоступен, звуковые устройства не читаются")
            return False
        return True

    def list_devices(self, kind: DeviceKind) -> list[AudioDevice]:
        """Enumerate one direction, marking which endpoint is the default.

        An endpoint that cannot be described is skipped rather than fatal: one
        broken driver in the list must not hide the other forty devices.

        Raises:
            ActionUnavailable: ``pycaw`` is missing or this is not Windows.
            DeviceUnavailable: the sound subsystem is not reachable.
        """
        utilities = audio_utilities()
        enumerator = device_enumerator()
        default_id = self._default_id(kind)
        try:
            collection = enumerator.EnumAudioEndpoints(kind.flow, _STATE_MASK_ALL)
            count = int(collection.GetCount())
        except COM_ERRORS as exc:
            _log.warning("не удалось перечислить %s: %s", kind.noun_ru, exc)
            return []
        devices: list[AudioDevice] = []
        for index in range(count):
            device = self._describe(utilities, collection, index, kind, default_id)
            if device is not None:
                devices.append(device)
        devices.sort(key=lambda item: (not item.is_default, not item.usable, item.name.casefold()))
        return devices

    def default_device(self, kind: DeviceKind) -> AudioDevice:
        """Read the current default endpoint.

        Raises:
            DeviceUnavailable: there is no default — the normal answer on a
                machine with no sound card, and the one the CI runner gives.
        """
        utilities = audio_utilities()
        enumerator = device_enumerator()
        try:
            described = utilities.CreateDevice(
                enumerator.GetDefaultAudioEndpoint(kind.flow, ROLE_MULTIMEDIA)
            )
        except COM_ERRORS as exc:
            raise DeviceUnavailable(
                f"no default {kind} endpoint: {exc}",
                user_message=kind.missing_ru,
            ) from exc
        if described is None:  # pragma: no cover - pycaw raises instead
            raise DeviceUnavailable(
                f"no default {kind} endpoint",
                user_message=kind.missing_ru,
            )
        return AudioDevice(
            device_id=str(described.id or ""),
            name=str(described.FriendlyName or ""),
            kind=kind,
            state=_state_of(described),
            is_default=True,
        )

    def set_default(self, device_id: str, kind: DeviceKind) -> None:
        """Point all three roles at one endpoint.

        Raises:
            PolicyUnavailable: the interface is not there, or refused the call.
                Both are the same thing to the user: this build will not do it.
        """
        policy = self._policy()
        for role in _ROLES:
            try:
                policy.SetDefaultEndpoint(device_id, role)
            except COM_ERRORS as exc:
                raise PolicyUnavailable(
                    f"SetDefaultEndpoint({device_id!r}, role={role}) failed: {exc}",
                ) from exc
        invalidate_devices()
        _log.info("устройство по умолчанию (%s): %s", kind, device_id)

    def _describe(
        self,
        utilities: Any,
        collection: Any,
        index: int,
        kind: DeviceKind,
        default_id: str,
    ) -> AudioDevice | None:
        """One endpoint out of the collection, or ``None`` when it cannot be read."""
        try:
            described = utilities.CreateDevice(collection.Item(index))
        except COM_ERRORS as exc:
            _log.debug("устройство %s не читается: %s", index, exc)
            return None
        if described is None:
            return None
        device_id = str(described.id or "")
        return AudioDevice(
            device_id=device_id,
            name=str(described.FriendlyName or ""),
            kind=kind,
            state=_state_of(described),
            is_default=bool(device_id) and device_id == default_id,
        )

    def _default_id(self, kind: DeviceKind) -> str:
        """Id of the current default, or ``""`` when there is none."""
        try:
            return self.default_device(kind).device_id
        except DeviceUnavailable:
            return ""

    def _policy(self) -> Any:
        """``IPolicyConfig``, declared by hand and created once per thread.

        The declaration lists ten placeholder methods before the one that is
        used. That is not decoration: ``SetDefaultEndpoint`` is the eleventh slot
        of the vtable, and ``comtypes`` locates a method by its position, so the
        ten have to be there and have to be in that order.

        Raises:
            PolicyUnavailable: the object cannot be created on this build.
        """
        cached = getattr(_com, "policy", None)
        if cached is not None:
            return cached
        initialize_com()
        try:
            import comtypes.client
            from comtypes import GUID
        except ImportError as exc:  # pragma: no cover - initialize_com raised first
            raise ActionUnavailable(
                f"comtypes is unavailable: {exc}",
                user_message="Управление звуком работает только в Windows.",
            ) from exc
        try:
            policy = comtypes.client.CreateObject(
                GUID(CLSID_POLICY_CONFIG),
                interface=_policy_interface(),
            )
        except COM_ERRORS as exc:
            raise PolicyUnavailable(
                f"IPolicyConfig {CLSID_POLICY_CONFIG} is not available: {exc}",
            ) from exc
        if policy is None:  # pragma: no cover - comtypes raises instead
            raise PolicyUnavailable(f"IPolicyConfig {CLSID_POLICY_CONFIG} returned nothing")
        _com.policy = policy
        return policy


def _state_of(described: Any) -> DeviceState:
    """Device state out of a ``pycaw`` device, whatever shape it came in.

    ``pycaw`` gives an ``AudioDeviceState`` enum member, but a device it failed
    to read leaves a plain int there — and an older ``pycaw`` gives the int
    always.
    """
    raw = getattr(described, "state", None)
    value: Any = getattr(raw, "value", raw)
    try:
        return DeviceState.from_wasapi(int(value))
    except (TypeError, ValueError):
        return DeviceState.UNKNOWN


def _policy_interface() -> Any:
    """Declare ``IPolicyConfig`` for ``comtypes``, once per process."""
    cached = getattr(_policy_interface, "_interface", None)
    if cached is not None:
        return cached

    import ctypes

    from comtypes import COMMETHOD, GUID, HRESULT, IUnknown

    class IPolicyConfig(IUnknown):  # type: ignore[misc]  # comtypes is untyped
        """Only the eleventh method is real; the rest hold their slots."""

        _iid_ = GUID(IID_POLICY_CONFIG)
        _methods_ = (
            COMMETHOD([], HRESULT, "GetMixFormat"),
            COMMETHOD([], HRESULT, "GetDeviceFormat"),
            COMMETHOD([], HRESULT, "ResetDeviceFormat"),
            COMMETHOD([], HRESULT, "SetDeviceFormat"),
            COMMETHOD([], HRESULT, "GetProcessingPeriod"),
            COMMETHOD([], HRESULT, "SetProcessingPeriod"),
            COMMETHOD([], HRESULT, "GetShareMode"),
            COMMETHOD([], HRESULT, "SetShareMode"),
            COMMETHOD([], HRESULT, "GetPropertyValue"),
            COMMETHOD([], HRESULT, "SetPropertyValue"),
            COMMETHOD(
                [],
                HRESULT,
                "SetDefaultEndpoint",
                (["in"], ctypes.c_wchar_p, "device_id"),
                (["in"], ctypes.c_uint, "role"),
            ),
            COMMETHOD([], HRESULT, "SetEndpointVisibility"),
        )

    _policy_interface._interface = IPolicyConfig  # type: ignore[attr-defined]
    return IPolicyConfig


#: What subscribing to — or unsubscribing from — hot-plug events can fail with.
#: COM refusing is one half; the other is an older ``pycaw`` whose callback class
#: does not have the shape this code expects, which arrives as an
#: ``AttributeError`` or a ``TypeError``. A constant rather than a starred tuple
#: in place because ``mypy`` does not accept the latter in an ``except``.
_WATCH_ERRORS: Final[tuple[type[Exception], ...]] = (
    ActionError,
    *COM_ERRORS,
    AttributeError,
    TypeError,
)


class DeviceWatcher:
    """Registers for ``OnDefaultDeviceChanged`` and clears the caches on it.

    The callback does not report the new device anywhere — nothing in Ayris
    caches which device that is. Its whole job is to drop the COM objects made
    before the change, so the next action builds fresh ones. Losing the
    notification therefore costs nothing: endpoints are re-resolved on every call
    anyway, and this only saves the enumerator from outliving the sound stack it
    came from.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self._enumerator: Any = None

    @property
    def watching(self) -> bool:
        """Whether the notification is registered right now."""
        return self._client is not None

    def start(self) -> bool:
        """Register the callback. ``False`` when this build cannot do it."""
        if self._client is not None:
            return True
        try:
            from pycaw.callbacks import MMNotificationClient
        except ImportError:
            _log.debug("pycaw.callbacks недоступен, hot-plug не отслеживается")
            return False

        class _Client(MMNotificationClient):  # type: ignore[misc]  # pycaw is untyped
            """Everything that changes the device list clears the cache."""

            def on_default_device_changed(
                self,
                flow: str,
                flow_id: int,  # noqa: ARG002 - pycaw's signature, unused by design
                role: str,
                role_id: int,  # noqa: ARG002
                default_device_id: str | None,  # noqa: ARG002
            ) -> None:
                _log.info("устройство по умолчанию сменилось (%s/%s)", flow, role)
                invalidate_devices()

            def on_device_added(self, added_device_id: str) -> None:  # noqa: ARG002
                invalidate_devices()

            def on_device_removed(self, removed_device_id: str) -> None:  # noqa: ARG002
                invalidate_devices()

        try:
            enumerator = device_enumerator()
            client = _Client()
            enumerator.RegisterEndpointNotificationCallback(client)
        except _WATCH_ERRORS as exc:
            _log.debug("не удалось подписаться на события устройств: %s", exc)
            return False
        self._client = client
        self._enumerator = enumerator
        return True

    def stop(self) -> None:
        """Unregister the callback. Safe to call when it was never registered."""
        if self._client is None:
            return
        try:
            self._enumerator.UnregisterEndpointNotificationCallback(self._client)
        except _WATCH_ERRORS as exc:  # pragma: no cover - teardown only
            _log.debug("не удалось отписаться от событий устройств: %s", exc)
        self._client = None
        self._enumerator = None


_backend: DeviceBackend | None = None
_real_backend: WasapiDevices | None = None
_watcher: DeviceWatcher | None = None
#: Re-entrant on purpose: :func:`get_device_backend` holds it while calling
#: :func:`start_device_watcher`, which takes it again.
_lock = threading.RLock()


def start_device_watcher() -> bool:
    """Subscribe to device changes once per process.

    Public because the volume actions need it too: a session that only ever says
    «громче» still has to drop its cached enumerator when the headphones are
    unplugged, and it has no reason to ask for a device backend otherwise.

    Returns:
        Whether the notification is registered — ``False`` on a build that cannot
        do it, which costs nothing but a stale cache.
    """
    global _watcher
    with _lock:
        if _watcher is None:
            _watcher = DeviceWatcher()
        if not _watcher.watching:
            _watcher.start()
        return _watcher.watching


def stop_device_watcher() -> None:
    """Unsubscribe and forget the watcher. For tests and for shutdown."""
    global _watcher
    with _lock:
        if _watcher is not None:
            _watcher.stop()
            _watcher = None


def get_device_backend() -> DeviceBackend:
    """The backend in force. Real WASAPI unless a test replaced it.

    The real one is built once so that the hot-plug watcher is registered once.

    Raises:
        ActionUnavailable: this is not Windows, or ``pycaw`` is missing.
    """
    if _backend is not None:
        return _backend
    global _real_backend
    with _lock:
        if _real_backend is None:
            real = WasapiDevices()
            if not real.supported():
                raise ActionUnavailable(
                    "audio device actions require Windows with pycaw",
                    user_message="Управление звуковыми устройствами работает только в Windows.",
                )
            _real_backend = real
            start_device_watcher()
        return _real_backend


def set_device_backend(backend: DeviceBackend | None) -> None:
    """Install a backend, or restore the real one with ``None``. Test seam."""
    global _backend
    _backend = backend


def list_audio_devices(
    kind: DeviceKind = DeviceKind.OUTPUT,
    *,
    backend: DeviceBackend | None = None,
    usable_only: bool = True,
    limit: int = MAX_LISTED_DEVICES,
) -> list[AudioDevice]:
    """Sound devices of one direction, default first.

    ``usable_only`` is on by default because the raw list is mostly noise: a
    machine that has ever had a monitor plugged in remembers every one of them
    as a ``NotPresent`` endpoint.
    """
    devices = (backend or get_device_backend()).list_devices(kind)
    if usable_only:
        devices = [device for device in devices if device.usable]
    return devices[: max(0, limit)]


def default_device(
    kind: DeviceKind = DeviceKind.OUTPUT,
    *,
    backend: DeviceBackend | None = None,
) -> AudioDevice:
    """The endpoint in use right now.

    Raises:
        DeviceUnavailable: nothing is connected.
    """
    return (backend or get_device_backend()).default_device(kind)


@register
class SetAudioDevice(Action):
    """Switch the default output or input device by part of its name.

    Not marked dangerous: it is loud rather than destructive, and it is undone by
    saying the other device's name.
    """

    meta: ClassVar = ActionMeta(
        name="SetAudioDevice",
        category=ActionCategory.AUDIO,
        title_ru="Переключить звуковое устройство",
        description_ru="Сделать устройство вывода или ввода тем, что по умолчанию",
        timeout_ms=10_000,
    )

    class Params(ActionParams):
        device: str = Field(
            min_length=1,
            max_length=200,
            title="Устройство",
            description="Название или его часть: «наушники», «колонки», «fifine»",
        )
        kind: DeviceKind = Field(
            default=DeviceKind.OUTPUT,
            title="Направление",
            description="Что переключать — вывод звука или запись",
            json_schema_extra={"choices_ru": {str(k): k.title_ru for k in DeviceKind}},
        )

    def run(self, params: Params) -> ActionResult[AudioDevice]:
        backend = get_device_backend()
        devices = backend.list_devices(params.kind)
        if not devices:
            raise DeviceUnavailable(
                f"no {params.kind} devices on this machine",
                user_message=params.kind.missing_ru,
            )
        target = find_device(devices, params.device, kind=params.kind)
        if target.is_default:
            return ActionResult.done(
                f"«{target.label}» и так выбрано.",
                value=target,
                detail=f"{target.device_id} is already default",
                data=target.as_dict(),
            )
        backend.set_default(target.device_id, params.kind)
        switched = AudioDevice(
            device_id=target.device_id,
            name=target.name,
            kind=target.kind,
            state=target.state,
            is_default=True,
        )
        return ActionResult.done(
            f"Переключила на «{target.label}».",
            value=switched,
            detail=f"default {params.kind} is now {target.device_id}",
            data=switched.as_dict(),
        )
