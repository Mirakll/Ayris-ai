"""The audio worker process: owns the microphone, publishes what it hears.

Capture lives in its own process for two reasons.  PortAudio's callback has a
hard deadline - a few milliseconds of GC pause in the GUI process would be
audible as a dropped block - and the process can be given a raised priority
without making the whole application aggressive about scheduling.  Section 12
of the specification asks for exactly that, and
:func:`~ayris.workers.base.apply_process_priority` applies
``performance.audio_priority`` before this class is even constructed.

The worker is a thin shell around :class:`~ayris.audio.capture.AudioCapture`:
it translates configuration into :class:`~ayris.audio.capture.CaptureSettings`,
exposes control over the wire, and turns the capture callbacks into worker
events.  Audio itself does *not* stream over the pipe frame by frame; consumers
ask for a slice of the ring buffer when they need one (task 08 for VAD, task 09
for the wake word), and only levels are pushed, at 20 Hz.

:func:`translate_audio_event` closes the loop on the parent's side by mapping
those events onto the bus.  The manager cannot know that ``level`` means
:class:`~ayris.core.events.AudioLevelChanged`, so the subsystem says so itself.
The bus classes are imported *inside* that function on purpose: it only ever
runs in the main process, and a top-level import would drag the settings model
- and with it pydantic - into every audio worker start.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.capture import (
    TARGET_SAMPLE_RATE,
    AudioCapture,
    AudioLevel,
    CaptureCallbacks,
    CaptureSettings,
    CaptureState,
)
from ayris.audio.devices import AudioBackend, AudioDevice, DeviceChange, SoundDeviceBackend
from ayris.audio.ring_buffer import DEFAULT_PRE_ROLL_MS
from ayris.core.errors import AudioError
from ayris.core.models import JsonObject
from ayris.core.paths import get_paths
from ayris.workers.base import Worker, method

if TYPE_CHECKING:
    from ayris.core.events import Event
    from ayris.workers.base import WorkerContext

__all__ = ["EVENT_TRANSLATOR", "AudioWorker", "translate_audio_event"]

#: Largest slice :meth:`AudioWorker.read` will put on the pipe.  Ten seconds of
#: 16 kHz mono is 320 KiB; anything longer belongs in a WAV file, not in a
#: pickled response.
MAX_READ_MS: Final = 10_000

#: Where debug recordings go when the caller does not name a file.
_DUMP_DIR: Final = "audio"


class AudioWorker(Worker):
    """Microphone capture, as seen from the supervisor.

    The methods mirror :class:`~ayris.audio.capture.AudioCapture` one for one.
    They return plain JSON-ish dictionaries so that DevTools can display a
    response without knowing anything about this module.
    """

    kind: ClassVar[str] = "audio"

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self._capture: AudioCapture | None = None

    # ------------------------------------------------------------ lifecycle

    def build_backend(self) -> AudioBackend:
        """Create the audio backend.

        A separate method so that tests can drive the whole worker against a
        fake sound card; production always gets PortAudio.
        """
        return SoundDeviceBackend()

    def on_start(self) -> None:
        """Open the microphone and start capturing.

        A missing or busy device does *not* fail the start.  Failing would make
        the supervisor restart the process, and restarting cannot conjure a
        microphone; instead capture enters ``device_lost`` and its monitor
        thread picks the device up as soon as it appears, which is what a user
        who plugs a headset in after launch expects.
        """
        capture = AudioCapture(
            settings=self._settings_from_params(self.params),
            backend=self.build_backend(),
            callbacks=CaptureCallbacks(
                on_level=self._on_level,
                on_state=self._on_state,
                on_devices=self._on_devices,
            ),
        )
        self._capture = capture
        try:
            capture.start()
        except AudioError as exc:
            self.log.warning("микрофон недоступен при запуске: %s", exc.technical)

    def on_stop(self) -> None:
        """Release the device."""
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

    def on_configure(self, params: JsonObject) -> None:
        """Apply new settings without dropping the stream when possible."""
        if self._capture is not None:
            self._capture.configure(self._settings_from_params(params))

    # --------------------------------------------------------------- events

    def _on_level(self, level: AudioLevel) -> None:
        """Forward a level measurement.

        Already throttled by capture - see
        :attr:`~ayris.audio.capture.CaptureSettings.level_interval_ms`.
        """
        self.emit(
            "level",
            {"rms": round(level.rms, 4), "peak": round(level.peak, 4), "clipped": level.clipped},
        )

    def _on_state(self, state: CaptureState, detail: str) -> None:
        """Forward a state transition."""
        self.emit("state", {"state": state.value, "detail": detail})

    def _on_devices(self, change: DeviceChange) -> None:
        """Forward an appeared/disappeared device."""
        self.emit(
            "devices",
            {
                "added": [device.label for device in change.added],
                "removed": [device.label for device in change.removed],
                "default_changed": change.default_changed,
            },
        )

    # -------------------------------------------------------------- control

    @method()
    def start(self, _params: JsonObject) -> JsonObject:
        """Open the device and begin capturing."""
        self._require().start()
        return self.status({})

    @method()
    def stop(self, _params: JsonObject) -> JsonObject:
        """Stop capturing and release the device."""
        self._require().stop()
        return self.status({})

    @method()
    def pause(self, params: JsonObject) -> JsonObject:
        """Release the device but keep the buffered audio."""
        self._require().pause(str(params.get("reason", "")))
        return self.status({})

    @method()
    def resume(self, _params: JsonObject) -> JsonObject:
        """Re-open the device after ``pause``."""
        self._require().resume()
        return self.status({})

    @method()
    def mute(self, params: JsonObject) -> JsonObject:
        """Mute or unmute the microphone in software.

        Args:
            params: ``muted`` (default ``True``).
        """
        self._require().mute(bool(params.get("muted", True)))
        return self.status({})

    @method()
    def set_device(self, params: JsonObject) -> JsonObject:
        """Switch to another microphone without restarting the worker.

        Args:
            params: ``device`` - identifier or name; empty means the system
                default.
        """
        self._require().set_device(str(params.get("device", "")))
        return self.status({})

    @method()
    def set_gain(self, params: JsonObject) -> JsonObject:
        """Change the software gain.

        Args:
            params: ``gain`` - linear factor, ``1.0`` for unity.
        """
        self._require().set_gain(_as_float(params.get("gain"), 1.0))
        return self.status({})

    # ----------------------------------------------------------------- data

    @method()
    def devices(self, params: JsonObject) -> JsonObject:
        """List input devices.

        Args:
            params: ``rescan`` - re-scan the hardware first.  Ignored while a
                stream is open, because re-initialising PortAudio would kill it.
        """
        capture = self._require()
        found = capture.devices(rescan=bool(params.get("rescan", False)))
        current = capture.device
        return {
            "devices": [_describe(device) for device in found],
            "current": current.id if current else "",
        }

    @method()
    def status(self, _params: JsonObject) -> JsonObject:
        """Everything DevTools shows about capture."""
        capture = self._capture
        if capture is None:
            return {"state": CaptureState.STOPPED.value, "running": False}
        stats = capture.stats()
        return {
            "state": stats.state.value,
            "running": stats.state is CaptureState.RUNNING,
            "device": stats.device_id,
            "device_name": stats.device_name,
            "device_sample_rate": stats.device_sample_rate,
            "device_channels": stats.device_channels,
            "sample_rate": capture.settings.sample_rate,
            "muted": stats.muted,
            "recording": stats.recording,
            "frames": stats.frames,
            "dropped_blocks": stats.dropped_blocks,
            "overflows": stats.overflows,
            "buffered_ms": stats.buffered_ms,
            "rms": round(stats.level.rms, 4),
            "peak": round(stats.level.peak, 4),
            "position": capture.position,
        }

    @method()
    def read(self, params: JsonObject) -> JsonObject:
        """Return captured audio.

        Two modes, because consumers come in two shapes.  A one-shot consumer
        asks for the last ``ms`` milliseconds; a streaming one passes the
        ``position`` from its previous answer and gets everything since, plus
        the count of frames it was too slow to collect.

        Args:
            params: ``ms`` (default :data:`~ayris.audio.ring_buffer.DEFAULT_PRE_ROLL_MS`)
                or ``position``.

        Returns:
            ``pcm`` as little-endian ``int16`` mono bytes, with the rate and the
            position to continue from.
        """
        capture = self._require()
        rate = capture.settings.sample_rate
        if "position" in params:
            limit = capture.buffer.frames_for_ms(MAX_READ_MS)
            read = capture.read_from(int(params["position"]), max_frames=limit)
            return {
                "pcm": read.pcm,
                "sample_rate": rate,
                "frames": read.frames,
                "position": read.position,
                "dropped": read.dropped,
            }
        ms = min(_as_float(params.get("ms"), float(DEFAULT_PRE_ROLL_MS)), float(MAX_READ_MS))
        pcm = capture.read_recent(ms)
        return {
            "pcm": pcm,
            "sample_rate": rate,
            "frames": len(pcm) // 2,
            "position": capture.position,
            "dropped": 0,
        }

    # ------------------------------------------------------------ debugging

    @method()
    def dump_wav(self, params: JsonObject) -> JsonObject:
        """Write the tail of the ring buffer to a WAV file.

        The DevTools button behind "что услышал микрофон": the fastest way to
        tell a dead microphone from a wake word that simply did not match.

        Args:
            params: ``path`` (default ``<cache>/audio/capture_<n>.wav``) and
                ``ms`` (default: the whole buffer).
        """
        capture = self._require()
        path = self._dump_path(params, "capture")
        ms = params.get("ms")
        written = capture.dump_wav(path, int(ms) if ms is not None else None)
        return {"path": str(written), "bytes": written.stat().st_size}

    @method()
    def record(self, params: JsonObject) -> JsonObject:
        """Start or stop recording captured audio to a WAV file.

        Unlike :meth:`dump_wav` this follows the stream, which is what
        collecting wake word samples needs.

        Args:
            params: ``enabled`` (default ``True``), ``path``, ``pre_roll_ms``.
        """
        capture = self._require()
        if not bool(params.get("enabled", True)):
            stopped = capture.stop_recording()
            return {"recording": False, "path": str(stopped) if stopped else ""}
        path = capture.start_recording(
            self._dump_path(params, "record"),
            pre_roll_ms=int(_as_float(params.get("pre_roll_ms"), 0.0)),
        )
        return {"recording": True, "path": str(path)}

    # -------------------------------------------------------------- helpers

    def _require(self) -> AudioCapture:
        """Return the capture pipeline.

        Raises:
            AudioError: If a method is called before ``on_start``, which means
                the worker is being driven out of order.
        """
        if self._capture is None:
            raise AudioError(
                "audio capture is not initialised",
                user_message="Захват звука ещё не запущен.",
            )
        return self._capture

    def _dump_path(self, params: JsonObject, prefix: str) -> Path:
        """Resolve the destination of a debug recording."""
        raw = params.get("path")
        if raw:
            return Path(str(raw)).expanduser()
        directory = get_paths().cache_dir / _DUMP_DIR
        directory.mkdir(parents=True, exist_ok=True)
        existing = len(list(directory.glob(f"{prefix}_*.wav")))
        return directory / f"{prefix}_{existing + 1:03d}.wav"

    @staticmethod
    def _settings_from_params(params: JsonObject) -> CaptureSettings:
        """Turn worker parameters into capture settings.

        The buffer and throttle values have no home in ``config.toml`` yet -
        the settings window is task 43 - so they are read from the parameters
        with the defaults from the specification, and a later task can start
        sending them without touching this code.
        """
        return CaptureSettings(
            device=str(params.get("device", "")),
            sample_rate=int(_as_float(params.get("sample_rate"), float(TARGET_SAMPLE_RATE))),
            frame_ms=int(_as_float(params.get("frame_ms"), 20.0)),
            gain=_as_float(params.get("gain"), 1.0),
            buffer_seconds=_as_float(params.get("buffer_seconds"), 30.0),
            pre_roll_ms=int(_as_float(params.get("pre_roll_ms"), float(DEFAULT_PRE_ROLL_MS))),
            level_interval_ms=int(_as_float(params.get("level_interval_ms"), 50.0)),
            mute_stops_stream=bool(params.get("mute_stops_stream", False)),
        )


def _describe(device: AudioDevice) -> JsonObject:
    """Flatten a device for the settings window."""
    return {
        "id": device.id,
        "name": device.name,
        "label": device.label,
        "host_api": device.host_api,
        "channels": device.channels,
        "sample_rate": int(device.default_sample_rate),
        "default": device.is_default,
    }


def _as_float(value: object, fallback: float) -> float:
    """Read a number out of the parameters, tolerating strings and ``None``."""
    if isinstance(value, bool) or value is None:
        return fallback
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return fallback


def translate_audio_event(kind: str, payload: JsonObject) -> Event | None:
    """Turn an audio worker event into a bus event.

    Registered by :func:`~ayris.workers.manager.install_workers`.  Returning
    ``None`` drops the event, which is the right answer for the transitions
    that only matter in the log.
    """
    # Imported here, never at module level: this function only ever runs in the
    # main process, where the events module is loaded anyway, while a top-level
    # import would drag the settings model - and with it pydantic - into every
    # audio worker start.
    from ayris.core.events import AudioLevelChanged

    if kind == "level":
        return AudioLevelChanged(
            rms=float(payload.get("rms", 0.0)),
            peak=float(payload.get("peak", 0.0)),
        )
    if kind == "state":
        return _state_notification(payload)
    if kind == "devices":
        return _devices_notification(payload)
    return None


def _state_notification(payload: JsonObject) -> Event | None:
    """Notify only about the transitions a user can act on."""
    from ayris.core.events import NotificationRequested

    state = str(payload.get("state", ""))
    detail = str(payload.get("detail", ""))
    if state == CaptureState.DEVICE_LOST.value:
        return NotificationRequested(
            title="Микрофон недоступен",
            message=detail or "Устройство записи отключено.",
            level="warning",
        )
    if state == CaptureState.RUNNING.value and detail:
        # Only the recovery path fills ``detail`` for a running stream, so this
        # cannot fire on an ordinary start.
        return NotificationRequested(title="Микрофон", message=detail, timeout_ms=3000)
    return None


def _devices_notification(payload: JsonObject) -> Event | None:
    """Tell the user when a microphone appears, stay quiet otherwise.

    A disappearing device is already covered by the ``device_lost`` transition
    when it was the one in use, and is noise when it was not.
    """
    from ayris.core.events import NotificationRequested

    added = payload.get("added")
    if not isinstance(added, list) or not added:
        return None
    names = ", ".join(str(item) for item in added)
    return NotificationRequested(
        title="Подключено устройство записи",
        message=names,
        timeout_ms=4000,
    )


#: What the supervisor registers for this worker.
EVENT_TRANSLATOR: Final = translate_audio_event
