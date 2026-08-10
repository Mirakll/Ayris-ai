"""Audio device enumeration with identifiers that survive a hot-plug.

PortAudio addresses devices by index, and those indices are not stable: unplug
a USB microphone and every device after it shifts down one slot.  A settings
file that stored index 3 would silently start recording from the wrong device.
So Ayris derives its own identifier from the host API plus a normalised device
name and only uses the index for the moment it takes to open a stream.

The name is normalised because Windows renumbers duplicate devices by editing
their name: the same headset appears as ``Микрофон (USB Audio Device)`` in one
port and ``Микрофон (2- USB Audio Device)`` in the next.  Stripping that
counter makes the identifier survive replugging.

The rest of the module is the seam between Ayris and PortAudio.  Everything the
capture pipeline needs is expressed by :class:`AudioBackend`, implemented for
real by :class:`SoundDeviceBackend` and replaced by a fake in the tests - which
is what lets device loss, resampling and the buffer be tested on machines with
no sound card at all.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol

from ayris.core.errors import AudioError

__all__ = [
    "AudioBackend",
    "AudioDevice",
    "AudioStream",
    "DeviceChange",
    "DeviceDirection",
    "DeviceEnumerator",
    "OutputStream",
    "PlaybackBackend",
    "PlaybackRequest",
    "RawDevice",
    "SoundDeviceBackend",
    "StreamCallback",
    "StreamRequest",
    "clean_device_name",
    "default_device",
    "describe_devices",
    "device_id",
    "diff_devices",
    "find_device",
    "list_devices",
    "resolve_device",
]

_log = logging.getLogger(__name__)

#: Windows numbers duplicate devices inside the parenthesised part of the name:
#: ``Микрофон (2- USB Audio Device)``.  The counter is positional, not a
#: property of the hardware, so it must not reach the identifier.
_ENUMERATION_PREFIX: Final = re.compile(r"\(\s*\d+\s*-\s*")

_WHITESPACE: Final = re.compile(r"\s+")


class DeviceDirection(StrEnum):
    """Which side of the sound card a device sits on."""

    INPUT = "input"
    OUTPUT = "output"

    @property
    def role(self) -> str:
        """Russian genitive for log and error messages.

        Not ``title``: :class:`~enum.StrEnum` inherits :meth:`str.title`, and
        shadowing it with a property of a different shape is a type error.
        """
        return "записи" if self is DeviceDirection.INPUT else "воспроизведения"

    @property
    def missing_hint(self) -> str:
        """What to check when there is no device of this kind at all."""
        if self is DeviceDirection.INPUT:
            return "Проверьте, подключён ли микрофон."
        return "Проверьте, подключены ли наушники или колонки."


@dataclass(frozen=True, slots=True)
class RawDevice:
    """One device exactly as the backend reports it.

    This is the narrow shape :class:`AudioBackend` implementations produce;
    :func:`list_devices` turns it into :class:`AudioDevice`.
    """

    index: int
    name: str
    host_api: str = ""
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: float = 44100.0
    default_input: bool = False
    default_output: bool = False

    def channels(self, direction: DeviceDirection) -> int:
        """Channel count for ``direction``."""
        if direction is DeviceDirection.INPUT:
            return self.max_input_channels
        return self.max_output_channels

    def is_default(self, direction: DeviceDirection) -> bool:
        """Whether the system picked this device as the default one."""
        if direction is DeviceDirection.INPUT:
            return self.default_input
        return self.default_output


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """A device as the rest of Ayris sees it.

    Attributes:
        id: Stable identifier, safe to store in the config.  Survives a
            hot-plug and a reboot; changes only if the driver renames the
            device.
        name: Cleaned device name, shown in the settings.
        direction: Capture or playback.
        index: PortAudio index.  Valid only until the next enumeration - pass
            it to the backend right away, never store it.
        host_api: Windows audio API behind the device (``MME``, ``WASAPI``, ...).
        channels: Channel count in :attr:`direction`.
        default_sample_rate: Rate the driver prefers.  Opening a stream at
            16 kHz can fail on WASAPI, so capture falls back to this one and
            resamples.
        is_default: Whether this is the system default device.
    """

    id: str
    name: str
    direction: DeviceDirection
    index: int
    host_api: str = ""
    channels: int = 1
    default_sample_rate: float = 44100.0
    is_default: bool = False

    @property
    def label(self) -> str:
        """Text for the settings combo box."""
        if self.host_api:
            return f"{self.name} ({self.host_api})"
        return self.name

    def matches(self, spec: str) -> bool:
        """Whether ``spec`` refers to this device.

        Accepts the identifier, the cleaned name, the label or the raw name the
        driver reported - config files written by hand or by older versions
        keep working.
        """
        if not spec:
            return False
        # The identifier is compared as-is: it is already normalised, and
        # ``_normalise`` would eat the ``:`` separators it is built from.
        if spec.strip().casefold() == self.id:
            return True
        return _normalise(spec) in {_normalise(self.name), _normalise(self.label)}


@dataclass(frozen=True, slots=True)
class DeviceChange:
    """Difference between two enumerations."""

    added: tuple[AudioDevice, ...] = ()
    removed: tuple[AudioDevice, ...] = ()
    default_changed: bool = False

    def __bool__(self) -> bool:
        """True when anything actually changed."""
        return bool(self.added or self.removed or self.default_changed)


@dataclass(frozen=True, slots=True)
class StreamRequest:
    """Parameters for opening a capture stream.

    ``sample_rate`` and ``channels`` describe the *device* side.  When the
    device cannot deliver 16 kHz mono, capture opens the stream at whatever the
    driver accepts and converts afterwards.
    """

    device_index: int
    sample_rate: int
    channels: int
    block_frames: int


#: Called by the backend once per captured block, on the real-time audio
#: thread: ``(pcm, overflowed)``.  Implementations must copy the bytes and
#: return - see :mod:`ayris.audio.capture` for what is allowed in there.
StreamCallback = Callable[[bytes, bool], None]


class AudioStream(Protocol):
    """An open capture stream."""

    @property
    def sample_rate(self) -> int:
        """Rate the device actually delivers."""

    @property
    def channels(self) -> int:
        """Channel count the device actually delivers."""

    @property
    def active(self) -> bool:
        """Whether the device is still feeding the callback."""

    def start(self) -> None:
        """Begin delivering blocks to the callback."""

    def stop(self) -> None:
        """Stop delivering blocks; the stream stays open."""

    def close(self) -> None:
        """Release the device.  Must tolerate being called twice."""


@dataclass(frozen=True, slots=True)
class PlaybackRequest:
    """Parameters for opening a playback stream.

    Unlike capture, playback is pull-based: PortAudio asks for the next block
    and the player fills it.  ``block_frames`` therefore sets the latency floor -
    a stop cannot take effect faster than the block already handed to the driver,
    which is why the player keeps it small.
    """

    device_index: int
    sample_rate: int
    channels: int
    block_frames: int


class DeviceEnumerator(Protocol):
    """The part of a backend that only lists hardware.

    Split out from :class:`AudioBackend` so that :func:`list_devices` and
    :func:`resolve_device` work for playback too: the TTS player needs the same
    stable identifiers and the same "device vanished" handling, but has no use
    for a capture stream.
    """

    def raw_devices(self) -> Sequence[RawDevice]:
        """Enumerate devices.

        Returns an empty sequence on a machine with no sound card instead of
        raising.

        Raises:
            AudioError: If the audio library itself is unusable.
        """

    def refresh(self) -> None:
        """Re-scan the hardware.

        PortAudio caches the device list at initialisation, so newly plugged
        devices stay invisible until this is called.  It tears the library down
        and back up, which kills open streams - only legal while this backend
        has no stream open.
        """


class AudioBackend(DeviceEnumerator, Protocol):
    """Everything the capture pipeline needs from PortAudio."""

    def open_input_stream(self, request: StreamRequest, callback: StreamCallback) -> AudioStream:
        """Open a capture stream.

        Raises:
            AudioError: If the device rejects the requested format or vanished
                between enumeration and this call.
        """

    def supports_rate(self, request: StreamRequest) -> bool:
        """Whether the device accepts this exact format without conversion."""


class OutputStream(Protocol):
    """An open playback stream.

    Deliberately not :class:`AudioStream`: playback has no callback of its own
    and needs :meth:`write`, while capture needs neither.  Sharing one Protocol
    would leave both sides with methods that raise.
    """

    @property
    def sample_rate(self) -> int:
        """Rate the device accepted."""

    @property
    def channels(self) -> int:
        """Channel count the device accepted."""

    @property
    def active(self) -> bool:
        """Whether the device is still consuming samples."""

    def start(self) -> None:
        """Begin consuming written samples."""

    def write(self, pcm: bytes) -> None:
        """Queue ``int16`` samples, blocking while the buffer is full.

        Raises:
            AudioError: The device was lost mid-playback.
        """

    def stop(self) -> None:
        """Stop now, discarding anything already queued.

        Discarding is the point: «Айрис, стоп» must not be followed by half a
        second of buffered speech.  Implementations use PortAudio's ``abort``
        rather than ``stop``, which drains.
        """

    def close(self) -> None:
        """Release the device.  Must tolerate being called twice."""


class PlaybackBackend(DeviceEnumerator, Protocol):
    """What the TTS player needs from PortAudio."""

    def open_output_stream(self, request: PlaybackRequest) -> OutputStream:
        """Open a playback stream in blocking-write mode.

        Raises:
            AudioError: If the device rejects the format or vanished between
                enumeration and this call.
        """


# --------------------------------------------------------------------- naming


def clean_device_name(name: str) -> str:
    """Strip the positional duplicate counter and collapse whitespace."""
    cleaned = _ENUMERATION_PREFIX.sub("(", name)
    return _WHITESPACE.sub(" ", cleaned).strip()


def device_id(name: str, host_api: str, direction: DeviceDirection) -> str:
    """Build the stable identifier for a device.

    Readable on purpose - it ends up in ``config.toml``, and a user who edits it
    by hand should be able to tell which microphone a line refers to.
    """
    return f"{direction.value}:{_normalise(host_api)}:{_normalise(clean_device_name(name))}"


def _normalise(text: str) -> str:
    """Casefold, collapse whitespace and free the identifier separator."""
    return _WHITESPACE.sub(" ", text).strip().casefold().replace(":", "_")


# ---------------------------------------------------------------- enumeration


def list_devices(
    backend: DeviceEnumerator,
    direction: DeviceDirection = DeviceDirection.INPUT,
) -> tuple[AudioDevice, ...]:
    """Enumerate usable devices in one direction.

    Devices with no channels in ``direction`` are dropped: PortAudio lists
    every endpoint twice, once per direction, with a zero channel count on the
    side it cannot serve.

    Identifiers are made unique within the returned tuple.  Two physically
    identical devices normalise to the same name, so the second and later ones
    get a ``#N`` suffix; their order follows PortAudio's, which is stable as
    long as neither device is unplugged.

    Raises:
        AudioError: If the backend cannot enumerate at all.  A machine with no
            devices is not an error - it returns an empty tuple.
    """
    devices: list[AudioDevice] = []
    seen: dict[str, int] = {}
    for raw in backend.raw_devices():
        channels = raw.channels(direction)
        if channels <= 0:
            continue
        base = device_id(raw.name, raw.host_api, direction)
        count = seen.get(base, 0) + 1
        seen[base] = count
        devices.append(
            AudioDevice(
                id=base if count == 1 else f"{base}#{count}",
                name=clean_device_name(raw.name),
                direction=direction,
                index=raw.index,
                host_api=raw.host_api,
                channels=channels,
                default_sample_rate=raw.default_sample_rate,
                is_default=raw.is_default(direction),
            )
        )
    return tuple(devices)


def default_device(
    backend: DeviceEnumerator,
    direction: DeviceDirection = DeviceDirection.INPUT,
) -> AudioDevice | None:
    """Return the system default device, or the first one, or ``None``."""
    devices = list_devices(backend, direction)
    for device in devices:
        if device.is_default:
            return device
    return devices[0] if devices else None


def find_device(devices: Iterable[AudioDevice], spec: str) -> AudioDevice | None:
    """Find the device ``spec`` refers to, or ``None``."""
    if not spec:
        return None
    return next((device for device in devices if device.matches(spec)), None)


def resolve_device(
    backend: DeviceEnumerator,
    spec: str = "",
    *,
    direction: DeviceDirection = DeviceDirection.INPUT,
    fallback: bool = False,
) -> AudioDevice:
    """Pick the device to open.

    Args:
        backend: Source of the device list.
        spec: Identifier or name from the settings.  Empty means "system
            default".
        direction: Capture or playback.
        fallback: When the requested device is missing, use the default instead
            of failing.  Capture sets this on the first start so a stale entry
            in the settings cannot leave the user without a microphone; hot-plug
            handling leaves it off, because there the point is to notice that
            the device is gone.

    Raises:
        AudioError: If there are no devices at all, or ``spec`` does not match
            any of them and ``fallback`` is off.
    """
    devices = list_devices(backend, direction)
    if not devices:
        raise AudioError(
            f"no {direction.value} devices available",
            user_message=(
                f"Не найдено ни одного устройства {direction.role}. {direction.missing_hint}"
            ),
        )
    if not spec:
        return next((device for device in devices if device.is_default), devices[0])
    found = find_device(devices, spec)
    if found is not None:
        return found
    if not fallback:
        raise AudioError(
            f"{direction.value} device {spec!r} not found",
            user_message=f"Устройство {direction.role} «{spec}» не найдено. "
            "Выберите другое в настройках.",
        )
    chosen = next((device for device in devices if device.is_default), devices[0])
    _log.warning(
        "device %r is gone, falling back to %r", spec, chosen.id, extra={"device": chosen.id}
    )
    return chosen


def diff_devices(
    before: Sequence[AudioDevice],
    after: Sequence[AudioDevice],
) -> DeviceChange:
    """Compare two enumerations by identifier."""
    old = {device.id for device in before}
    new = {device.id for device in after}
    added = tuple(device for device in after if device.id not in old)
    removed = tuple(device for device in before if device.id not in new)
    old_default = next((device.id for device in before if device.is_default), None)
    new_default = next((device.id for device in after if device.is_default), None)
    return DeviceChange(
        added=added,
        removed=removed,
        default_changed=bool(before) and old_default != new_default,
    )


def describe_devices(devices: Sequence[AudioDevice]) -> str:
    """One-line summary for the log."""
    if not devices:
        return "<none>"
    return ", ".join(f"{device.label}{'*' if device.is_default else ''}" for device in devices)


# ------------------------------------------------------------------- backends


class _SoundDeviceStream:
    """:class:`AudioStream` over ``sounddevice.RawInputStream``."""

    __slots__ = ("_channels", "_closed", "_sample_rate", "_stream")

    def __init__(self, stream: Any, sample_rate: int, channels: int) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._channels = channels
        self._closed = False

    @property
    def sample_rate(self) -> int:
        """Rate the device delivers."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Channel count the device delivers."""
        return self._channels

    @property
    def active(self) -> bool:
        """Whether PortAudio still considers the stream alive."""
        if self._closed:
            return False
        try:
            return bool(self._stream.active)
        except Exception:  # a dead device raises PortAudioError here
            return False

    def start(self) -> None:
        """Start the device."""
        try:
            self._stream.start()
        except Exception as exc:  # PortAudioError et al.
            raise AudioError(f"cannot start audio stream: {exc}") from exc

    def stop(self) -> None:
        """Stop the device, keeping it open."""
        if self._closed:
            return
        try:
            self._stream.stop()
        except Exception as exc:  # unplugged devices fail to stop
            _log.debug("ignoring error while stopping audio stream: %s", exc)

    def close(self) -> None:
        """Release the device."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close(ignore_errors=True)
        except Exception as exc:  # see above
            _log.debug("ignoring error while closing audio stream: %s", exc)


class _SoundDeviceOutput:
    """:class:`OutputStream` over ``sounddevice.RawOutputStream``."""

    __slots__ = ("_channels", "_closed", "_sample_rate", "_stream")

    def __init__(self, stream: Any, sample_rate: int, channels: int) -> None:
        self._stream = stream
        self._sample_rate = sample_rate
        self._channels = channels
        self._closed = False

    @property
    def sample_rate(self) -> int:
        """Rate the device accepted."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Channel count the device accepted."""
        return self._channels

    def _is_closed(self) -> bool:
        """Whether :meth:`close` has run.

        A method rather than a bare attribute read because ``close()`` is called
        from the thread that cancels playback while :meth:`write` is blocked
        inside PortAudio.  Reading ``self._closed`` twice around that call looks
        redundant to a type checker - it narrows the second read to ``False`` and
        calls the branch unreachable - but the value genuinely changes in between.
        """
        return self._closed

    @property
    def active(self) -> bool:
        """Whether PortAudio still considers the stream alive."""
        if self._closed:
            return False
        try:
            return bool(self._stream.active)
        except Exception:  # a dead device raises PortAudioError here
            return False

    def start(self) -> None:
        """Start the device."""
        try:
            self._stream.start()
        except Exception as exc:  # PortAudioError et al.
            raise AudioError(
                f"cannot start output stream: {exc}",
                user_message="Не удалось начать воспроизведение.",
            ) from exc

    def write(self, pcm: bytes) -> None:
        """Hand samples to PortAudio, blocking while its buffer is full.

        Raises:
            AudioError: The device was lost.  Writing to a closed stream is not
                an error - a cancel racing the writer is the normal path, and
                the player checks its own flag right after.
        """
        if self._closed or not pcm:
            return
        try:
            self._stream.write(pcm)
        except Exception as exc:  # PortAudioError on unplug, ValueError on size
            if self._is_closed():
                return
            raise AudioError(
                f"cannot write to output stream: {exc}",
                user_message="Воспроизведение прервалось: устройство недоступно.",
            ) from exc

    def stop(self) -> None:
        """Abort playback, discarding what PortAudio still holds."""
        if self._closed:
            return
        try:
            # abort() drops the buffered tail; stop() would play it out.
            self._stream.abort(ignore_errors=True)
        except Exception as exc:  # unplugged devices fail to abort
            _log.debug("ignoring error while aborting output stream: %s", exc)

    def close(self) -> None:
        """Release the device."""
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close(ignore_errors=True)
        except Exception as exc:  # see above
            _log.debug("ignoring error while closing output stream: %s", exc)


class SoundDeviceBackend:
    """PortAudio through :mod:`sounddevice`.

    The import is deferred into the methods on purpose.  ``import sounddevice``
    loads the PortAudio shared library at import time, which fails on machines
    and CI runners without one; a module-level import would then break
    collection of every test that merely mentions audio.  Deferring it turns
    that into an :class:`~ayris.core.errors.AudioError` from the one call that
    needs the library.
    """

    __slots__ = ("_module",)

    def __init__(self) -> None:
        self._module: Any | None = None

    def _sd(self) -> Any:
        """Import :mod:`sounddevice` once.

        Raises:
            AudioError: If the module or PortAudio itself is missing.
        """
        if self._module is None:
            try:
                import sounddevice  # deferred on purpose, see the class docstring
            except (ImportError, OSError) as exc:
                raise AudioError(
                    f"sounddevice/PortAudio is unavailable: {exc}",
                    user_message="Звуковая подсистема недоступна: не удалось загрузить PortAudio.",
                ) from exc
            self._module = sounddevice
        return self._module

    def raw_devices(self) -> Sequence[RawDevice]:
        """Enumerate PortAudio devices."""
        sd = self._sd()
        try:
            entries = list(sd.query_devices())
            host_apis = list(sd.query_hostapis())
            default_in, default_out = self._defaults(sd)
        except Exception as exc:  # PortAudio raises library-specific types
            raise AudioError(
                f"cannot enumerate audio devices: {exc}",
                user_message="Не удалось получить список аудиоустройств.",
            ) from exc

        devices: list[RawDevice] = []
        for index, entry in enumerate(entries):
            api_index = int(entry.get("hostapi", -1))
            api_name = ""
            if 0 <= api_index < len(host_apis):
                api_name = str(host_apis[api_index].get("name", ""))
            devices.append(
                RawDevice(
                    index=index,
                    name=str(entry.get("name", f"device {index}")),
                    host_api=api_name,
                    max_input_channels=int(entry.get("max_input_channels", 0)),
                    max_output_channels=int(entry.get("max_output_channels", 0)),
                    default_sample_rate=float(entry.get("default_samplerate", 44100.0)),
                    default_input=index == default_in,
                    default_output=index == default_out,
                )
            )
        return tuple(devices)

    def refresh(self) -> None:
        """Re-initialise PortAudio so hot-plugged devices become visible."""
        sd = self._sd()
        try:
            sd._terminate()  # the documented way to re-scan devices
            sd._initialize()
        except Exception as exc:  # never let a rescan kill the worker
            _log.debug("PortAudio rescan failed: %s", exc)

    def open_input_stream(self, request: StreamRequest, callback: StreamCallback) -> AudioStream:
        """Open a raw ``int16`` capture stream."""
        sd = self._sd()

        def _shim(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            """Adapt PortAudio's callback to :data:`StreamCallback`.

            Runs on the audio thread: copy the driver's buffer (it is reused
            for the next block) and hand it over.  Nothing else belongs here.
            """
            callback(bytes(indata), bool(status))

        try:
            stream = sd.RawInputStream(
                device=request.device_index,
                samplerate=request.sample_rate,
                channels=request.channels,
                dtype="int16",
                blocksize=request.block_frames,
                callback=_shim,
            )
        except Exception as exc:  # PortAudioError, ValueError, OSError
            raise AudioError(
                f"cannot open device {request.device_index} at "
                f"{request.sample_rate} Hz / {request.channels}ch: {exc}",
                user_message="Не удалось открыть микрофон. "
                "Возможно, он занят другой программой.",
            ) from exc
        return _SoundDeviceStream(stream, request.sample_rate, request.channels)

    def supports_rate(self, request: StreamRequest) -> bool:
        """Ask PortAudio whether the device accepts this format."""
        sd = self._sd()
        try:
            sd.check_input_settings(
                device=request.device_index,
                samplerate=request.sample_rate,
                channels=request.channels,
                dtype="int16",
            )
        except Exception:  # "unsupported" is reported as an exception
            return False
        return True

    def open_output_stream(self, request: PlaybackRequest) -> OutputStream:
        """Open a raw ``int16`` playback stream in blocking-write mode.

        No callback: the player already owns a thread and writing from it is
        both simpler and safer than filling a driver buffer from a real-time
        callback that may not allocate.  PortAudio blocks the write when its
        buffer is full, which is exactly the pacing the player wants.
        """
        sd = self._sd()
        try:
            stream = sd.RawOutputStream(
                device=request.device_index,
                samplerate=request.sample_rate,
                channels=request.channels,
                dtype="int16",
                blocksize=request.block_frames,
            )
        except Exception as exc:  # PortAudioError, ValueError, OSError
            raise AudioError(
                f"cannot open output device {request.device_index} at "
                f"{request.sample_rate} Hz / {request.channels}ch: {exc}",
                user_message="Не удалось открыть устройство воспроизведения. "
                "Возможно, оно занято другой программой.",
            ) from exc
        return _SoundDeviceOutput(stream, request.sample_rate, request.channels)

    @staticmethod
    def _defaults(sd: Any) -> tuple[int, int]:
        """Read the default device indices, tolerating a deviceless machine."""
        try:
            default = sd.default.device
            return int(default[0]), int(default[1])
        except Exception as exc:  # raises when no device exists at all
            _log.debug("no default audio device: %s", exc)
            return -1, -1
