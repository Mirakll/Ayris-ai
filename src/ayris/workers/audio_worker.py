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
ask for a slice of the ring buffer when they need one (task 09 for the wake
word), and only levels are pushed, at 20 Hz.

Detection rides along on the same callback.  Every frame capture produces goes
through :class:`~ayris.audio.denoise.DenoiseStream` and then
:class:`~ayris.audio.segmenter.Segmenter`, here in the audio process, because
the alternative - shipping raw frames to the parent and segmenting there - would
put a GIL-bound consumer between the microphone and the phrase boundary.  What
crosses the pipe is the *result*: ``speech_started`` and ``speech_ended`` with
positions and lengths.  The audio of a finished phrase stays here until somebody
asks for it with :meth:`AudioWorker.segment`, so a wake word that did not match
costs nothing to discard.

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

from ayris.audio.calibration import (
    DEFAULT_PHRASE_SEC,
    DEFAULT_SILENCE_SEC,
    calibrate_pcm,
)
from ayris.audio.capture import (
    TARGET_SAMPLE_RATE,
    AudioCapture,
    AudioLevel,
    CaptureCallbacks,
    CaptureSettings,
    CaptureState,
)
from ayris.audio.denoise import DenoiseMode, DenoiseSettings, DenoiseStream
from ayris.audio.devices import AudioBackend, AudioDevice, DeviceChange, SoundDeviceBackend
from ayris.audio.ring_buffer import DEFAULT_PRE_ROLL_MS
from ayris.audio.segmenter import (
    Segmenter,
    SegmenterCallbacks,
    SegmenterSettings,
    SpeechSegment,
    SpeechStart,
)
from ayris.audio.vad import VadSettings
from ayris.core.errors import AudioError
from ayris.core.models import JsonObject
from ayris.core.paths import get_paths
from ayris.workers.base import Worker, method

if TYPE_CHECKING:
    from ayris.audio.calibration import CalibrationReport
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
        self._denoise: DenoiseStream | None = None
        self._segmenter: Segmenter | None = None
        self._last_segment: SpeechSegment | None = None
        self._calibration_noise: bytes = b""

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
        self._build_detection(self.params)
        capture = AudioCapture(
            settings=self._settings_from_params(self.params),
            backend=self.build_backend(),
            callbacks=CaptureCallbacks(
                on_level=self._on_level,
                on_state=self._on_state,
                on_devices=self._on_devices,
                on_frames=self._on_frames,
            ),
        )
        self._capture = capture
        try:
            capture.start()
        except AudioError as exc:
            self.log.warning("микрофон недоступен при запуске: %s", exc.technical)

    def on_stop(self) -> None:
        """Release the device."""
        if self._segmenter is not None:
            self._segmenter.flush()
        if self._capture is not None:
            self._capture.stop()
            self._capture = None
        self._denoise = None
        self._segmenter = None

    def on_configure(self, params: JsonObject) -> None:
        """Apply new settings without dropping the stream when possible.

        Detection is reconfigured before capture: a phrase in progress is closed
        under the old thresholds, which is the only interpretation that does not
        lose audio, and the new denoiser is in place before the next frame
        arrives on the callback.
        """
        self._configure_detection(params)
        if self._capture is not None:
            self._capture.configure(self._settings_from_params(params))

    # --------------------------------------------------------------- events

    def _on_level(self, level: AudioLevel) -> None:
        """Forward a level measurement.

        Already throttled by capture - see
        :attr:`~ayris.audio.capture.CaptureSettings.level_interval_ms`.  The
        detector's verdict rides along because the overlay needs both numbers at
        once: a meter that moves while the sphere stays dark is precisely the
        picture of a gate set too high.
        """
        segmenter = self._segmenter
        self.emit(
            "level",
            {
                "rms": round(level.rms, 4),
                "peak": round(level.peak, 4),
                "clipped": level.clipped,
                "is_speech": segmenter.is_speech if segmenter is not None else False,
                "gate_db": round(segmenter.vad.gate_db, 1) if segmenter is not None else 0.0,
            },
        )

    def _on_frames(self, pcm: bytes) -> None:
        """Run captured audio through denoising and phrase detection.

        Called from capture's processing thread, so it must stay cheap: the gate
        costs a fraction of a millisecond per frame and the detector less than
        that, which :meth:`vad` reports so a slow machine can be spotted rather
        than guessed at.
        """
        segmenter = self._segmenter
        denoiser = self._denoise
        if segmenter is None or denoiser is None:
            return
        clean = denoiser.push(pcm)
        if clean:
            segmenter.push(clean)

    def _on_speech_started(self, start: SpeechStart) -> None:
        """Forward a confirmed onset."""
        capture = self._capture
        self.emit(
            "speech_started",
            {
                "frame_index": start.frame_index,
                "pre_roll_ms": start.pre_roll_ms,
                "position": capture.position if capture is not None else 0,
            },
        )

    def _on_speech_ended(self, segment: SpeechSegment) -> None:
        """Forward a finished phrase, keeping its audio here.

        The PCM is deliberately left out of the event: a rejected segment is
        never asked for, and an accepted one is fetched once by whoever is going
        to recognise it.  Pushing every phrase across the pipe would double the
        cost of a wake word that did not match.
        """
        if segment.accepted:
            self._last_segment = segment
        capture = self._capture
        self.emit(
            "speech_ended",
            {
                "frame_index": segment.frame_index,
                "duration_ms": segment.duration_ms,
                "speech_ms": segment.speech_ms,
                "pre_roll_ms": segment.pre_roll_ms,
                "reason": segment.reason.value,
                "accepted": segment.accepted,
                "frames": segment.frames,
                "position": capture.position if capture is not None else 0,
            },
        )

    def _on_state(self, state: CaptureState, detail: str) -> None:
        """Forward a state transition."""
        if state is not CaptureState.RUNNING and self._segmenter is not None:
            # A stream that stopped will not deliver the silence that would
            # normally close the phrase, so close it here rather than leave a
            # half-collected segment behind for the next start to inherit.
            self._segmenter.flush()
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

    # ------------------------------------------------------------ detection

    @method()
    def vad(self, _params: JsonObject) -> JsonObject:
        """Everything the settings window shows about detection.

        The two halves answer different questions.  ``gate_db`` versus the level
        meter explains *why* a phrase was or was not heard; ``avg_ms`` and
        ``fallback`` explain what the machine is actually running, which is the
        first thing to check when detection works on one computer and not on
        another.
        """
        segmenter = self._segmenter
        denoiser = self._denoise
        if segmenter is None or denoiser is None:
            return {"running": False}
        stats = segmenter.stats
        settings = segmenter.vad.settings
        denoise = denoiser.stats
        return {
            "running": True,
            "engine": segmenter.vad.engine,
            "aggressiveness": settings.aggressiveness,
            "threshold": settings.threshold,
            "noise_floor_db": settings.noise_floor_db,
            "gate_db": round(segmenter.vad.gate_db, 1),
            "frame_ms": settings.frame_ms,
            "state": stats.state.value,
            "is_speech": segmenter.is_speech,
            "frames": stats.frames,
            "speech_frames": stats.speech_frames,
            "speech_ratio": round(stats.speech_ratio, 3),
            "segments": stats.segments,
            "rejected": stats.rejected,
            "truncated": stats.truncated,
            "current_ms": stats.current_ms,
            "silence_ms": segmenter.settings.silence_ms,
            "min_speech_ms": segmenter.settings.min_speech_ms,
            "denoise": {
                "mode": denoise.mode.value,
                "engine": denoise.engine.value,
                "fallback": denoise.fallback,
                "latency_ms": round(denoise.latency_ms, 1),
                "avg_ms": round(denoise.avg_ms, 3),
                "max_ms": round(denoise.max_ms, 3),
                "realtime_factor": round(denoise.realtime_factor, 3),
                "reduction_db": round(denoise.reduction_db, 1),
            },
        }

    @method()
    def segment(self, params: JsonObject) -> JsonObject:
        """Return the audio of the last accepted phrase.

        Args:
            params: ``keep`` - leave the segment available for another caller.
                The default consumes it, so that a recogniser polling after a
                restart cannot pick up a phrase from before it.

        Returns:
            ``pcm`` with its metadata, or ``available: False`` when nothing has
            been detected yet.
        """
        segment = self._last_segment
        if segment is None:
            return {"available": False, "pcm": b"", "sample_rate": TARGET_SAMPLE_RATE}
        if not bool(params.get("keep", False)):
            self._last_segment = None
        return {
            "available": True,
            "pcm": segment.pcm,
            "sample_rate": segment.sample_rate,
            "frames": segment.frames,
            "frame_index": segment.frame_index,
            "duration_ms": segment.duration_ms,
            "speech_ms": segment.speech_ms,
            "pre_roll_ms": segment.pre_roll_ms,
            "reason": segment.reason.value,
        }

    @method()
    def calibrate(self, params: JsonObject) -> JsonObject:
        """Measure the room and propose settings, one stage per call.

        Nothing here blocks: the audio is already in the ring buffer, so a stage
        reads backwards over the last few seconds.  The caller drives the
        prompts - show "помолчите", wait, call ``silence``; show "скажите
        фразу", wait, call ``phrase`` - which keeps the GUI responsive and lets
        a user who coughed simply repeat a stage.

        Args:
            params: ``stage`` - ``silence``, ``phrase``, ``report`` or
                ``reset``; ``seconds`` - how far back to read.

        Returns:
            For ``silence``: the measured floor.  For ``phrase`` and ``report``:
            the full report from :func:`~ayris.audio.calibration.calibrate_pcm`.

        Raises:
            AudioError: When ``phrase`` or ``report`` is asked for before
                ``silence`` - the floor is what every threshold hangs off, and
                guessing it would make the report a fiction.
        """
        capture = self._require()
        stage = str(params.get("stage", "silence"))
        if stage == "reset":
            self._calibration_noise = b""
            return {"stage": stage, "ready": False}
        if stage == "silence":
            seconds = _as_float(params.get("seconds"), DEFAULT_SILENCE_SEC)
            self._calibration_noise = capture.read_recent(seconds * 1000.0)
            report = self._calibration_report(capture, None)
            return {"stage": stage, "ready": True, **report.as_dict()}
        if not self._calibration_noise:
            raise AudioError(
                "calibration: silence stage has not run",
                user_message="Сначала измерьте уровень тишины.",
            )
        if stage == "report":
            return {
                "stage": stage,
                "ready": True,
                **self._calibration_report(capture, None).as_dict(),
            }
        if stage != "phrase":
            raise AudioError(
                f"unknown calibration stage {stage!r}",
                user_message="Неизвестный этап калибровки.",
            )
        seconds = _as_float(params.get("seconds"), DEFAULT_PHRASE_SEC)
        spoken = capture.read_recent(seconds * 1000.0)
        return {
            "stage": stage,
            "ready": True,
            **self._calibration_report(capture, spoken).as_dict(),
        }

    def _calibration_report(self, capture: AudioCapture, phrase: bytes | None) -> CalibrationReport:
        """Analyse the recorded stages against the settings currently in force."""
        segmenter = self._segmenter
        denoiser = self._denoise
        return calibrate_pcm(
            self._calibration_noise,
            phrase,
            sample_rate=capture.settings.sample_rate,
            vad_settings=segmenter.vad.settings if segmenter is not None else None,
            segmenter_settings=segmenter.settings if segmenter is not None else None,
            base_gain=capture.settings.gain,
            base_denoise=denoiser.settings.mode if denoiser is not None else DenoiseMode.RNNOISE,
        )

    # -------------------------------------------------------------- helpers

    def _build_detection(self, params: JsonObject) -> None:
        """Create the denoiser and the segmenter for ``params``."""
        self._denoise = DenoiseStream(_denoise_settings_from_params(params))
        self._segmenter = Segmenter(
            _segmenter_settings_from_params(params),
            _vad_settings_from_params(params),
            callbacks=SegmenterCallbacks(
                on_speech_started=self._on_speech_started,
                on_speech_ended=self._on_speech_ended,
            ),
        )

    def _configure_detection(self, params: JsonObject) -> None:
        """Apply new thresholds to the running detector."""
        if self._denoise is None or self._segmenter is None:
            self._build_detection(params)
            return
        self._denoise.configure(_denoise_settings_from_params(params))
        self._segmenter.configure(
            _segmenter_settings_from_params(params),
            _vad_settings_from_params(params),
        )

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


def _vad_settings_from_params(params: JsonObject) -> VadSettings:
    """Turn worker parameters into detector settings.

    The names match ``voice.audio_input`` one for one, so the settings window
    can hand its section over unchanged.
    """
    return VadSettings(
        sample_rate=int(_as_float(params.get("sample_rate"), float(TARGET_SAMPLE_RATE))),
        frame_ms=int(_as_float(params.get("frame_ms"), 20.0)),
        aggressiveness=int(_as_float(params.get("vad_aggressiveness"), 2.0)),
        threshold=_as_float(params.get("vad_threshold"), 0.5),
        noise_floor_db=_as_float(params.get("noise_floor_db"), -45.0),
    )


def _segmenter_settings_from_params(params: JsonObject) -> SegmenterSettings:
    """Turn worker parameters into phrase thresholds.

    ``max_utterance_sec`` is seconds in the configuration and milliseconds
    everywhere below it; the conversion belongs here rather than in the
    segmenter, which should not have to know what a TOML file looks like.
    """
    return SegmenterSettings(
        silence_ms=int(_as_float(params.get("silence_ms"), 700.0)),
        max_utterance_ms=int(_as_float(params.get("max_utterance_sec"), 30.0) * 1000.0),
        pre_roll_ms=int(_as_float(params.get("pre_roll_ms"), float(DEFAULT_PRE_ROLL_MS))),
    )


def _denoise_settings_from_params(params: JsonObject) -> DenoiseSettings:
    """Turn worker parameters into suppression settings.

    An unknown mode falls back to ``rnnoise`` rather than raising: the value
    comes from a file a user can edit, and refusing to capture over a typo would
    be a poor trade.
    """
    raw = str(params.get("denoise", DenoiseMode.RNNOISE.value))
    try:
        mode = DenoiseMode(raw)
    except ValueError:
        mode = DenoiseMode.RNNOISE
    return DenoiseSettings(
        mode=mode,
        sample_rate=int(_as_float(params.get("sample_rate"), float(TARGET_SAMPLE_RATE))),
        frame_ms=int(_as_float(params.get("frame_ms"), 20.0)),
        noise_floor_db=_as_float(params.get("noise_floor_db"), -45.0),
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
            is_speech=bool(payload.get("is_speech", False)),
        )
    if kind == "speech_started":
        return _speech_started(payload)
    if kind == "speech_ended":
        return _speech_ended(payload)
    if kind == "state":
        return _state_notification(payload)
    if kind == "devices":
        return _devices_notification(payload)
    return None


def _speech_started(_payload: JsonObject) -> Event:
    """An onset became a bus event."""
    from ayris.core.events import SpeechStarted

    return SpeechStarted(source="vad")


def _speech_ended(payload: JsonObject) -> Event | None:
    """A finished phrase became a bus event.

    Rejected segments are published too, with ``reason="too_short"``.  They have
    to be: the onset was confirmed and ``SpeechStarted`` went out, so the
    overlay is sitting in its listening state and something must take it out of
    there.  Subscribers that only want real speech filter on the reason, which
    is one check in one place instead of a stuck animation.
    """
    from ayris.core.events import SpeechEnded

    return SpeechEnded(
        duration_ms=int(_as_float(payload.get("duration_ms"), 0.0)),
        reason=str(payload.get("reason", "silence")),
    )


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
