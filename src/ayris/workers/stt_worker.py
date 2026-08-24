"""The STT worker process: holds the recognition model, answers with text.

Recognition lives in its own process for the reason every heavy model does: a
Whisper decode holds the GIL for hundreds of milliseconds at a time, and doing
that in the GUI process would freeze the overlay exactly while it is supposed to
be animating.  Here it costs nothing - the parent is blocked on a pipe read and
free to paint.

**Audio arrives through shared memory.**  A phrase is a few hundred kilobytes;
pickling it into the request would copy it twice and stall the supervisor's
writer thread.  Instead the caller puts the PCM in a
:class:`~ayris.workers.protocol.SharedAudioBlock` and sends the descriptor, and
:func:`~ayris.workers.protocol.open_audio` attaches to it here.  The bytes are
copied out of the mapping immediately - the view dies with the ``with`` block -
and **resampled exactly once, at this boundary**.  Everything downstream of
:meth:`SttWorker.transcribe` may assume 16 kHz mono int16, which is what Vosk
requires and what Whisper's feature extractor wants anyway.

**The model is loaded lazily and dropped when it goes cold.**  Starting the
process is cheap; loading a model is not, and a user who talks to Ayris twice a
day should not be paying a gigabyte of resident memory for the other twenty-three
hours.  So :meth:`SttWorker.transcribe` brings the engine up on first use, and an
idle timer - ``performance.model_idle_sec``, halved in eco mode - unloads it
again after a silence, with a :func:`gc.collect` behind it so CTranslate2's
destructor actually runs and the VRAM comes back.  The next phrase pays the load
again.  That trade is the whole point of eco mode.

**Loading never blocks the caller.**  :meth:`SttWorker.load_model` returns as
soon as the background thread is running and reports the outcome as a
``model`` event; the dispatch loop in :class:`~ayris.workers.base.Worker` is
single-threaded, so a synchronous load would make the worker unresponsive to
``ping`` for the twenty seconds a cold Whisper model takes - and the supervisor
would kill it as hung.  A ``transcribe`` that arrives mid-load waits on the same
lock and then proceeds, which is what the caller wanted anyway.

**The RAM limit is checked before the load, not after.**  ``ram_limit_mb`` is a
soft cap the user picked in the settings; a model whose measured size times its
engine's :attr:`~ayris.audio.stt.base.SttEngine.memory_factor` exceeds it is
refused with a sentence naming both numbers.  An OOM kill in a worker process
looks to the user like the assistant randomly dying, and the model manager in
section 14 needs a real answer to show next to the download button.

Every call is timed and the numbers - load time, inference time, real-time
factor - go to the pipeline log under the caller's ``request_id`` and into
:meth:`SttWorker.status` for DevTools.  RTF is the one that matters: above 1.0
means recognition is slower than speech, and the user is going to notice.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, ClassVar, Final

from ayris.audio.stt.base import (
    STT_SAMPLE_RATE,
    AudioBuffer,
    SttOptions,
    TranscriptResult,
    create_engine,
    engine_names,
    estimate_model_bytes,
)
from ayris.core.errors import AyrisError, SttError
from ayris.core.models import JsonObject
from ayris.core.paths import get_paths
from ayris.utils.logger import get_pipeline_logger
from ayris.workers.base import Worker, method
from ayris.workers.protocol import AudioChunk, WorkerError, open_audio

if TYPE_CHECKING:
    from ayris.audio.stt.base import SttEngine
    from ayris.core.events import Event
    from ayris.workers.base import WorkerContext

__all__ = ["EVENT_TRANSLATOR", "SttWorker", "translate_stt_event"]

#: Bytes in a megabyte, as the settings mean it.
_MB: Final = 1024 * 1024

#: How often the idle watcher wakes up.  Five seconds is fine granularity for a
#: timeout measured in minutes and costs nothing; a thread that sleeps for the
#: whole timeout would ignore a shortened one after ``configure``.
_IDLE_POLL: Final = 5.0

#: Eco mode multiplies the configured idle timeout by this.  Half, because the
#: user asking for eco mode is saying they would rather wait than allocate, but
#: unloading after a handful of seconds would reload the model between two
#: sentences of the same conversation.
_ECO_IDLE_FACTOR: Final = 0.5

#: Idle unload is never scheduled sooner than this, whatever the settings say.
#: A five-second timeout with a twenty-second load is not a memory strategy.
_MIN_IDLE_SEC: Final = 30.0


class SttWorker(Worker):
    """Speech recognition in a worker process.

    Args:
        context: Supplied by the runtime.
    """

    kind: ClassVar[str] = "stt"

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self._engine: SttEngine | None = None
        self._engine_name = ""
        self._model_name = ""
        #: Guards the engine reference against the idle thread and the loader.
        #: Held across a load and across a transcription, which is what makes
        #: "unload while decoding" impossible rather than merely unlikely.
        self._lock = threading.RLock()
        self._loading: threading.Thread | None = None
        self._last_used = monotonic()
        self._idle_stop = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._load_ms = 0.0
        self._calls = 0
        self._total_inference_ms = 0.0
        self._total_audio_ms = 0.0
        self._pipeline = get_pipeline_logger()

    # -- lifecycle ------------------------------------------------------

    def on_start(self) -> None:
        """Start the idle watcher.  The model itself waits for a caller.

        Deliberately empty of model work: a worker that loaded on start would
        make ``eco_mode`` meaningless and would push a cold-disk Whisper load
        into the supervisor's start timeout.
        """
        self._idle_thread = threading.Thread(
            target=self._idle_loop, name="ayris-stt-idle", daemon=True
        )
        self._idle_thread.start()
        self.log.info(
            "stt: воркер готов, движок %s, модель %s, выгрузка после %.0f с простоя",
            self._configured_engine(),
            self._configured_model(),
            self._idle_timeout(),
        )

    def on_stop(self) -> None:
        """Stop the watcher and drop the model.  Must not raise."""
        self._idle_stop.set()
        thread = self._idle_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_IDLE_POLL + 1.0)
        self._idle_thread = None
        self._release("stop")

    def on_configure(self, params: JsonObject) -> None:
        """Unload when the engine or the model changed underneath us.

        Threads and language are read per call, so they need nothing here.  The
        engine and the model are baked into the loaded object, and the settings
        that pick them carry :attr:`~ayris.core.config.RestartScope.STT` anyway;
        this is the belt to that braces, for a configure that arrives without a
        restart.
        """
        with self._lock:
            if self._engine is None:
                return
            engine = str(params.get("offline_engine", "")) or self._engine_name
            model = str(params.get("offline_model", "")) or self._model_name
            if engine != self._engine_name or model != self._model_name:
                self.log.info(
                    "stt: настройки сменили модель %s/%s на %s/%s, выгружаю",
                    self._engine_name,
                    self._model_name,
                    engine,
                    model,
                )
                self._release("reconfigured")

    # -- wire methods ---------------------------------------------------

    @method()
    def load_model(self, params: JsonObject) -> JsonObject:
        """Bring the engine up in the background.

        Args:
            params: ``engine`` and ``model`` override the configured ones;
                ``wait`` blocks until the load finishes, which only the tests
                and an explicit "load now" button in the settings want.

        Returns:
            ``started`` is ``False`` when the model was already loaded or a load
            was already running - both are successes, not errors.  The outcome
            of a background load arrives as a ``model`` event.
        """
        engine_name = str(params.get("engine", "")) or self._configured_engine()
        model_name = str(params.get("model", "")) or self._configured_model()
        wait = bool(params.get("wait", False))

        with self._lock:
            if self._engine is not None and self._matches(engine_name, model_name):
                return self._loaded_reply(started=False, reason="already_loaded")
            running = self._loading
            if running is not None and running.is_alive():
                # Somebody already asked.  Waiting means waiting for *that*
                # load: starting a second one would load the same model twice.
                if wait:
                    pending = running
                else:
                    return self._loaded_reply(started=False, reason="in_progress")
            else:
                pending = threading.Thread(
                    target=self._load_in_background,
                    args=(engine_name, model_name),
                    name="ayris-stt-load",
                    daemon=True,
                )
                self._loading = pending
                pending.start()
            started = pending is not running

        if wait:
            pending.join()
            return self._loaded_reply(started=started, reason="loaded")
        return {"started": True, "engine": engine_name, "model": model_name, "loaded": False}

    @method()
    def transcribe(self, params: JsonObject) -> JsonObject:
        """Recognise one phrase.

        Args:
            params: An :class:`~ayris.workers.protocol.AudioChunk` descriptor
                under :data:`~ayris.workers.protocol.AUDIO_PARAM`, plus an
                optional ``request_id`` for the pipeline log and ``language``.

        Returns:
            :meth:`~ayris.audio.stt.base.TranscriptResult.to_params` - a plain
            dict, because the caller may be a different Python build in a
            different process and dataclasses over a pipe are a liability.

        Raises:
            SttError: No audio in the request, or the engine refused it.
                Silence is *not* an error: it comes back as an empty result.
        """
        chunk = AudioChunk.from_params(params)
        if chunk is None:
            raise SttError(
                "stt: transcribe called without an audio descriptor",
                user_message="Внутренняя ошибка: аудио не передано в распознавание.",
            )
        request_id = str(params.get("request_id", ""))
        buffer = self._read_audio(chunk)

        started = perf_counter()
        engine = self._ensure_loaded(
            str(params.get("engine", "")) or self._configured_engine(),
            str(params.get("model", "")) or self._configured_model(),
        )
        with self._lock:
            self.context.check_cancelled()
            result = engine.transcribe(buffer)
            self._last_used = monotonic()
        elapsed_ms = (perf_counter() - started) * 1000.0

        self._record(result, elapsed_ms, request_id)
        return result.to_params()

    @method()
    def unload(self, _params: JsonObject) -> JsonObject:
        """Drop the model now, without waiting for the idle timer.

        What the settings window calls when the user switches to online
        recognition, and what eco mode calls when the assistant is put to sleep.
        """
        with self._lock:
            was_loaded = self._engine is not None
            self._release("requested")
        return {"unloaded": was_loaded}

    @method()
    def status(self, _params: JsonObject) -> JsonObject:
        """Everything DevTools shows on the recognition row.

        The averages are cumulative over the worker's life rather than a moving
        window: the interesting question is "is this machine fast enough for this
        model", and that does not change between phrases.
        """
        with self._lock:
            engine = self._engine
            device = engine.device if engine is not None else ""
            loaded = engine is not None
            fallback = str(getattr(engine, "fallback_reason", "")) if engine is not None else ""
        idle_for = monotonic() - self._last_used
        return {
            "loaded": loaded,
            "loading": self._loading is not None and self._loading.is_alive(),
            "engine": self._engine_name or self._configured_engine(),
            "model": self._model_name or self._configured_model(),
            "device": device,
            "fallback_reason": fallback,
            "available_engines": list(engine_names(available_only=True)),
            "load_ms": round(self._load_ms, 1),
            "calls": self._calls,
            "avg_inference_ms": round(self._average_inference_ms(), 1),
            "real_time_factor": round(self._overall_rtf(), 3),
            "idle_sec": round(idle_for, 1),
            "idle_timeout_sec": round(self._idle_timeout(), 1),
            "ram_limit_mb": self._ram_limit_mb(),
        }

    # -- loading --------------------------------------------------------

    def _ensure_loaded(self, engine_name: str, model_name: str) -> SttEngine:
        """The engine, loading it first if this is the first phrase.

        Raises:
            SttError: The load failed.  Propagated to the caller rather than
                swallowed, because a transcription that cannot happen is an
                answer the pipeline has to handle.
        """
        with self._lock:
            engine = self._engine
            if engine is not None and self._matches(engine_name, model_name):
                return engine
            if engine is not None:
                self._release("model_switch")
            return self._load(engine_name, model_name)

    def _load(self, engine_name: str, model_name: str) -> SttEngine:
        """Construct, size-check and load an engine.  Caller holds the lock.

        Raises:
            SttError: Unknown engine, missing model, or a model that does not
                fit under ``ram_limit_mb``.
        """
        engine = create_engine(engine_name)
        path = self._model_path(model_name)
        self._check_memory(engine, path)

        started = perf_counter()
        engine.load(path, self._options())
        self._load_ms = (perf_counter() - started) * 1000.0
        self._engine = engine
        self._engine_name = engine_name
        self._model_name = model_name
        self._last_used = monotonic()
        self._pipeline.info(
            "stt: модель %s (%s) загружена за %.0f мс на %s",
            model_name,
            engine_name,
            self._load_ms,
            engine.device,
            extra={"request_id": ""},
        )
        self.emit(
            "model",
            {
                "state": "loaded",
                "engine": engine_name,
                "model": model_name,
                "device": engine.device,
                "load_ms": round(self._load_ms, 1),
            },
        )
        return engine

    def _load_in_background(self, engine_name: str, model_name: str) -> None:
        """Thread body for :meth:`load_model`.  Never raises out of the thread.

        A failure here has no caller to raise at - the request that started it
        returned long ago - so it becomes a ``model`` event with ``state:
        "failed"`` and the user-facing sentence from the exception.  Dropping it
        on the floor would leave the settings window spinning forever.
        """
        try:
            with self._lock:
                if self._engine is not None and self._matches(engine_name, model_name):
                    return
                if self._engine is not None:
                    self._release("model_switch")
                self._load(engine_name, model_name)
        except AyrisError as exc:
            self.log.error("stt: не удалось загрузить модель %s: %s", model_name, exc)
            self.emit(
                "model",
                {
                    "state": "failed",
                    "engine": engine_name,
                    "model": model_name,
                    "error": exc.user_message,
                },
            )
        except Exception as exc:  # pragma: no cover - a bug, not a usage error
            self.log.exception("stt: сбой загрузки модели %s", model_name)
            self.emit(
                "model",
                {
                    "state": "failed",
                    "engine": engine_name,
                    "model": model_name,
                    "error": f"Не удалось загрузить модель распознавания: {exc}",
                },
            )

    def _release(self, reason: str) -> None:
        """Unload the engine and let the allocator have the memory back.

        The :func:`gc.collect` is the part that matters.  CTranslate2 frees VRAM
        in a destructor, and a reference cycle inside the vendor object means
        dropping our reference alone leaves the card allocated until some
        unrelated allocation triggers a collection - by which time the user has
        looked at their GPU monitor and concluded eco mode does nothing.
        """
        engine = self._engine
        self._engine = None
        self._engine_name = ""
        self._model_name = ""
        self._load_ms = 0.0
        if engine is None:
            return
        try:
            engine.unload()
        except Exception:  # pragma: no cover - unload must never fail a stop
            self.log.exception("stt: ошибка выгрузки модели")
        gc.collect()
        self.log.info("stt: модель выгружена (%s)", reason)
        self.emit("model", {"state": "unloaded", "reason": reason})

    def _check_memory(self, engine: SttEngine, path: Path) -> None:
        """Refuse a model that will not fit under the configured cap.

        Measured from the files on disk rather than guessed from the name, so an
        unpacked ``large-v3`` is caught even when the folder is called
        ``whisper``.  ``ram_limit_mb`` of 0 means the user turned the cap off.

        Raises:
            SttError: The estimate exceeds the limit.
        """
        limit_mb = self._ram_limit_mb()
        if limit_mb <= 0:
            return
        needed_bytes = estimate_model_bytes(path) * engine.memory_factor
        if needed_bytes <= 0.0:
            return
        needed_mb = needed_bytes / _MB
        if needed_mb <= float(limit_mb):
            return
        raise SttError(
            f"stt: model {path.name} needs about {needed_mb:.0f} MB, limit is {limit_mb} MB",
            user_message=(
                f"Модель «{path.name}» требует примерно {needed_mb:.0f} МБ, "
                f"а лимит памяти — {limit_mb} МБ. Выберите модель поменьше "
                f"или поднимите лимит в настройках."
            ),
        )

    # -- audio ----------------------------------------------------------

    def _read_audio(self, chunk: AudioChunk) -> AudioBuffer:
        """Copy PCM out of shared memory and bring it to 16 kHz mono.

        The copy is not avoidable and not regrettable: the mapping is unlinked
        by the supervisor as soon as the call returns, and the resampler needs a
        contiguous buffer anyway.  Resampling happens here and nowhere else -
        see the module docstring.

        Raises:
            SttError: The block vanished, which means the supervisor released it
                early or the caller lied about its name.
        """
        try:
            with open_audio(chunk) as view:
                pcm = bytes(view)
        except (WorkerError, OSError) as exc:
            raise SttError(
                f"stt: shared audio block {chunk.block} is gone: {exc}",
                user_message="Внутренняя ошибка: аудио не дошло до распознавания.",
            ) from exc
        buffer = AudioBuffer(
            pcm=pcm,
            sample_rate=chunk.sample_rate or STT_SAMPLE_RATE,
            channels=max(1, chunk.channels),
        )
        return buffer.prepared_for(STT_SAMPLE_RATE)

    # -- idle -----------------------------------------------------------

    def _idle_loop(self) -> None:
        """Poll for an idle model until the worker stops."""
        while not self._idle_stop.wait(_IDLE_POLL):
            self._drop_if_idle()

    def _drop_if_idle(self) -> None:
        """Unload the model if it has been unused for the configured time.

        Split out of :meth:`_idle_loop` so that a test can ask the question
        directly instead of sleeping through a timeout that is five minutes long
        by default.
        """
        timeout = self._idle_timeout()
        if timeout <= 0.0:
            return
        with self._lock:
            if self._engine is None:
                return
            if monotonic() - self._last_used < timeout:
                return
            self._release("idle")

    def _idle_timeout(self) -> float:
        """Seconds of silence before the model goes away.  ``0`` disables it."""
        configured = _as_float(self.params.get("model_idle_sec"), 0.0)
        if configured <= 0.0:
            return 0.0
        if bool(self.params.get("eco_mode", False)):
            configured *= _ECO_IDLE_FACTOR
        return max(_MIN_IDLE_SEC, configured)

    # -- metrics --------------------------------------------------------

    def _record(self, result: TranscriptResult, elapsed_ms: float, request_id: str) -> None:
        """Log one call's timings and fold them into the running averages.

        ``elapsed_ms`` covers the lazy load too when this was the first phrase,
        which is why it is reported next to the engine's own ``inference_ms``
        rather than instead of it - a two-second first answer with a 90 ms
        inference is a cold start, not a slow machine.
        """
        inference_ms = result.inference_ms or elapsed_ms
        self._calls += 1
        self._total_inference_ms += inference_ms
        self._total_audio_ms += result.duration_ms
        self._pipeline.info(
            "stt: %s за %.0f мс (аудио %.0f мс, RTF %.2f, движок %s%s)%s",
            "пусто" if result.is_empty else f"{result.word_count} сл.",
            inference_ms,
            result.duration_ms,
            result.real_time_factor,
            result.engine,
            f"/{result.device}" if result.device else "",
            f", всего {elapsed_ms:.0f} мс" if elapsed_ms - inference_ms > 1.0 else "",
            extra={"request_id": request_id},
        )
        self.emit(
            "metrics",
            {
                "request_id": request_id,
                "inference_ms": round(inference_ms, 1),
                "total_ms": round(elapsed_ms, 1),
                "audio_ms": round(result.duration_ms, 1),
                "real_time_factor": round(result.real_time_factor, 3),
                "confidence": round(result.confidence, 3),
                "empty": result.is_empty,
                "engine": result.engine,
                "device": result.device,
            },
        )

    def _average_inference_ms(self) -> float:
        """Mean inference time over the worker's life, or zero before the first."""
        return self._total_inference_ms / self._calls if self._calls else 0.0

    def _overall_rtf(self) -> float:
        """Inference time over audio time.  Above 1.0 is slower than real time."""
        return self._total_inference_ms / self._total_audio_ms if self._total_audio_ms else 0.0

    # -- settings -------------------------------------------------------

    def _options(self) -> SttOptions:
        """Turn worker params into engine options."""
        return SttOptions.from_params(self.params)

    def _configured_engine(self) -> str:
        """The engine the settings picked, defaulting to Vosk.

        Deliberately not the settings' own default of GigaAM: this fallback only
        fires when the worker was started without params at all, and then the
        streaming engine that needs 46 MB is a better guess than the one that needs
        215 MB and may not be downloaded yet.
        """
        return str(self.params.get("offline_engine", "")) or "vosk"

    def _configured_model(self) -> str:
        """The model folder the settings picked."""
        return str(self.params.get("offline_model", ""))

    def _ram_limit_mb(self) -> int:
        """``performance.ram_limit_mb``; ``0`` means the user turned it off."""
        return int(_as_float(self.params.get("ram_limit_mb"), 0.0))

    def _model_path(self, model_name: str) -> Path:
        """Where a model named in the settings lives.

        An absolute path is honoured as-is so a user can keep a fifteen-gigabyte
        model on another drive; a bare name is resolved under the profile's
        ``models/stt``.

        Raises:
            SttError: The settings name no model at all.
        """
        if not model_name:
            raise SttError(
                "stt: no offline model configured",
                user_message="Не выбрана модель распознавания речи. Укажите её в настройках.",
            )
        candidate = Path(model_name)
        if candidate.is_absolute():
            return candidate
        return get_paths().stt_models_dir / model_name

    def _matches(self, engine_name: str, model_name: str) -> bool:
        """Whether the loaded engine is already the one being asked for."""
        return self._engine_name == engine_name and self._model_name == model_name

    def _loaded_reply(self, *, started: bool, reason: str) -> JsonObject:
        """The answer :meth:`load_model` gives when nothing had to be started."""
        engine = self._engine
        return {
            "started": started,
            "reason": reason,
            "loaded": engine is not None,
            "engine": self._engine_name,
            "model": self._model_name,
            "device": engine.device if engine is not None else "",
            "load_ms": round(self._load_ms, 1),
        }


def _as_float(value: object, default: float) -> float:
    """Read a params value as a float without trusting its type."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def translate_stt_event(kind: str, payload: JsonObject) -> Event | None:
    """Turn an STT worker event into a bus event.

    Registered by :func:`~ayris.workers.manager.install_workers`.  ``metrics``
    returns ``None`` on purpose: the numbers are already in the pipeline log and
    in :meth:`SttWorker.status`, and putting one bus event on every phrase in
    front of every subscriber buys nothing.
    """
    # Imported here rather than at module level for the same reason as in the
    # audio worker: this only ever runs in the main process, and a top-level
    # import would pull pydantic into every STT worker start.
    if kind == "model":
        return _model_notification(payload)
    return None


def _model_notification(payload: JsonObject) -> Event | None:
    """Tell the user when a model failed to load, stay quiet otherwise.

    A successful load and an idle unload are both invisible by design - the
    point of lazy loading is that nobody has to think about it.  A failure is
    the one case where the assistant is about to look broken and the reason is
    knowable.
    """
    from ayris.core.events import NotificationRequested

    if str(payload.get("state", "")) != "failed":
        return None
    return NotificationRequested(
        title="Распознавание речи",
        message=str(payload.get("error", "")) or "Не удалось загрузить модель.",
        level="error",
        timeout_ms=8000,
    )


#: What the supervisor registers for this worker.
EVENT_TRANSLATOR: Final = translate_stt_event
