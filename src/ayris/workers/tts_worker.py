"""The TTS worker process: holds the voice, answers with PCM.

Synthesis goes in its own process for the same reason recognition does. Piper's
ONNX session and Silero's TorchScript both hold the GIL for the length of a
sentence, and doing that in the GUI process would freeze the overlay exactly
while it is meant to be animating a mouth. Here the parent is blocked on a pipe
read and free to paint.

**Audio leaves through shared memory.** A sentence at 22 kHz is 100-200 KB;
pickling it into the reply would copy it twice on the way out. So
:meth:`TtsWorker.synthesize` allocates a
:class:`~ayris.workers.protocol.SharedAudioBlock`, and the reply carries only the
descriptor. The block is owned by *this* process and stays alive until the caller
says it is done - :meth:`TtsWorker.release` - or until the next synthesis
recycles it. That is the one place this worker differs from the STT one, where
the parent owns the block and the worker only reads.

**Streaming is a sequence of calls, not one long one.** ``synthesize_stream``
produces the first sentence and returns it together with a token; the caller asks
for the next piece with ``next_chunk`` until it comes back empty. A single call
that yielded over the pipe would hold the dispatch loop for the whole answer and
make ``ping`` time out, and the supervisor would kill the worker as hung. This
way the first sound is out in one round trip and each following sentence is
synthesized while the previous one plays.

**Cancellation is cooperative and checked between sentences.**
:meth:`TtsWorker.cancel` sets a flag that the streaming loop reads before each
sentence, and :meth:`~ayris.workers.base.WorkerContext.check_cancelled` covers
the supervisor's own cancel. A sentence already inside ONNX cannot be interrupted
- neither engine offers that - so the guarantee is "no new audio after cancel",
and the player's abort is what silences what is already queued. Together they are
what «Айрис, стоп» means.

**The voice is loaded lazily and dropped when it goes cold**, exactly as in
:mod:`ayris.workers.stt_worker`: a load costs 400 ms for Piper and several
seconds for Silero, and a user who talks to Ayris twice a day should not hold the
weights for the other twenty-three hours. ``performance.model_idle_sec`` drives
it, halved in eco mode.

**The cache lives here, not in the player.** A hit has to skip the synthesis, and
the synthesis is what happens in this process. It is keyed on text, voice, speed
and pitch, so the player receives the same bytes either way and cannot tell the
difference - which is the point.
"""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from time import monotonic, perf_counter
from typing import TYPE_CHECKING, ClassVar, Final
from uuid import uuid4

from ayris.audio.tts.base import (
    AudioChunk,
    TtsOptions,
    VoiceSpec,
    clamp_pitch,
    clamp_speed,
    concat_chunks,
    create_engine,
    engine_class,
    engine_names,
    estimate_voice_bytes,
)
from ayris.audio.tts.cache import PhraseCache
from ayris.audio.tts.sentence_split import split_sentences
from ayris.core.errors import AyrisError, TtsError
from ayris.core.models import JsonObject
from ayris.core.paths import get_paths
from ayris.utils.logger import get_pipeline_logger
from ayris.workers.base import Worker, method
from ayris.workers.protocol import AUDIO_PARAM, SharedAudioBlock

if TYPE_CHECKING:
    from ayris.audio.tts.base import TtsEngine
    from ayris.core.events import Event
    from ayris.workers.base import WorkerContext

__all__ = ["EVENT_TRANSLATOR", "TtsWorker", "translate_tts_event"]

#: Bytes in a megabyte, as the settings mean it.
_MB: Final = 1024 * 1024

#: How often the idle watcher wakes up.
_IDLE_POLL: Final = 5.0

#: Eco mode multiplies the configured idle timeout by this.
_ECO_IDLE_FACTOR: Final = 0.5

#: Idle unload is never scheduled sooner than this, however low the setting.
_MIN_IDLE_SEC: Final = 30.0

#: Streams older than this are forgotten by :meth:`TtsWorker.next_chunk`. A
#: caller that crashed mid-answer must not leave its sentences pinned forever.
_STREAM_TTL_SEC: Final = 300.0


class _Stream:
    """A partly-consumed multi-sentence answer.

    Not a generator: a generator would keep the engine call on the worker's
    dispatch thread across pipe round trips, and the sentences have to be
    produced one per call, when the caller asks.
    """

    __slots__ = ("created", "index", "pitch", "sentences", "speed", "text", "voice")

    def __init__(
        self,
        sentences: list[str],
        voice: VoiceSpec,
        speed: float,
        pitch: float,
        text: str,
    ) -> None:
        self.sentences = sentences
        self.voice = voice
        self.speed = speed
        self.pitch = pitch
        self.text = text
        self.index = 0
        self.created = monotonic()

    @property
    def done(self) -> bool:
        """Whether every sentence has been handed out."""
        return self.index >= len(self.sentences)

    def take(self) -> str:
        """The next sentence, or empty when there are none left."""
        if self.done:
            return ""
        sentence = self.sentences[self.index]
        self.index += 1
        return sentence


class TtsWorker(Worker):
    """Speech synthesis in a worker process.

    Args:
        context: Supplied by the runtime.
    """

    kind: ClassVar[str] = "tts"

    def __init__(self, context: WorkerContext) -> None:
        super().__init__(context)
        self._engine: TtsEngine | None = None
        self._engine_name = ""
        self._voice: VoiceSpec | None = None
        #: Guards the engine against the idle thread and the loader. Held across
        #: a load and across a synthesis, which is what makes "unload while
        #: speaking" impossible rather than merely unlikely.
        self._lock = threading.RLock()
        self._cancelled = threading.Event()
        self._last_used = monotonic()
        self._idle_stop = threading.Event()
        self._idle_thread: threading.Thread | None = None
        self._blocks: dict[str, SharedAudioBlock] = {}
        self._streams: dict[str, _Stream] = {}
        self._cache = PhraseCache(limit_bytes=0)
        self._load_ms = 0.0
        self._calls = 0
        self._total_synthesis_ms = 0.0
        self._total_audio_ms = 0.0
        self._pipeline = get_pipeline_logger()

    # -- lifecycle ------------------------------------------------------

    def on_start(self) -> None:
        """Start the idle watcher and size the cache. The voice waits for a caller."""
        self._cache.set_limit(self._cache_limit_bytes())
        self._idle_thread = threading.Thread(
            target=self._idle_loop, name="ayris-tts-idle", daemon=True
        )
        self._idle_thread.start()
        self.log.info(
            "tts: воркер готов, движок %s, голос %s, кэш %d МБ, " "выгрузка после %.0f с простоя",
            self._configured_engine(),
            self._configured_voice(),
            self._cache_limit_bytes() // _MB,
            self._idle_timeout(),
        )

    def on_stop(self) -> None:
        """Stop the watcher, drop the voice and free every shared block."""
        self._idle_stop.set()
        thread = self._idle_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=_IDLE_POLL + 1.0)
        self._idle_thread = None
        with self._lock:
            self._release("stop")
        self._release_blocks()
        self._streams.clear()

    def on_configure(self, params: JsonObject) -> None:
        """React to a settings change that the loaded voice cannot absorb.

        Speed, pitch and volume are per call and need nothing here. Engine and
        voice are baked into the loaded object; the cache limit is applied
        immediately so a user who set it to zero stops accumulating files at
        once rather than after a restart.
        """
        self._cache.set_limit(self._cache_limit_bytes())
        with self._lock:
            if self._engine is None:
                return
            engine = str(params.get("engine", "")) or self._engine_name
            voice = str(params.get("voice", "")) or self._voice_id()
            if engine != self._engine_name or voice != self._voice_id():
                self.log.info(
                    "tts: настройки сменили голос %s/%s на %s/%s, выгружаю",
                    self._engine_name,
                    self._voice_id(),
                    engine,
                    voice,
                )
                self._release("reconfigured")

    # -- wire methods ---------------------------------------------------

    @method()
    def load_voice(self, params: JsonObject) -> JsonObject:
        """Bring a voice into memory.

        Args:
            params: ``engine`` and ``voice`` override the configured ones.

        Returns:
            Which voice is loaded, on what device, and how long it took.

        Raises:
            TtsError: The library is missing, the voice is not found, or it does
                not fit under ``ram_limit_mb``.
        """
        engine_name = str(params.get("engine", "")) or self._configured_engine()
        voice_id = str(params.get("voice", "")) or self._configured_voice()
        with self._lock:
            engine = self._ensure_loaded(engine_name, voice_id)
            return {
                "loaded": True,
                "engine": self._engine_name,
                "voice": self._voice_id(),
                "device": engine.device,
                "sample_rate": engine.sample_rate,
                "load_ms": round(self._load_ms, 1),
            }

    @method()
    def voices(self, params: JsonObject) -> JsonObject:
        """Voices the settings window can offer, without loading anything.

        Args:
            params: ``engine`` limits the answer to one engine; by default every
                engine that is installed is asked.
        """
        wanted = str(params.get("engine", ""))
        names = (wanted,) if wanted else engine_names()
        found: list[JsonObject] = []
        for name in names:
            try:
                engine = engine_class(name)
            except TtsError:
                continue
            found.extend(spec.to_params() for spec in engine.voices())
        return {"voices": found, "engines": list(engine_names())}

    @method()
    def synthesize(self, params: JsonObject) -> JsonObject:
        """Speak one text in full, cache included.

        Args:
            params: ``text``, plus optional ``engine``, ``voice``, ``speed``,
                ``pitch`` and ``request_id``.

        Returns:
            A shared-memory descriptor under ``block`` with the format fields,
            or ``empty: True`` when there was nothing speakable. ``cached`` says
            whether synthesis ran at all, which is what DevTools shows.

        Raises:
            TtsError: No voice could be loaded, or synthesis failed.
        """
        text = str(params.get("text", ""))
        request_id = str(params.get("request_id", ""))
        sentences = self._sentences(text)
        if not sentences:
            return {"empty": True, "cached": False, "request_id": request_id}

        self._cancelled.clear()
        started = perf_counter()
        voice, speed, pitch = self._resolve_call(params)
        cached = self._cache.get(text, voice, speed, pitch)
        if cached is not None:
            self._record(text, cached, 0.0, request_id, cached_hit=True)
            return self._publish(cached, request_id, cached=True)

        with self._lock:
            self.context.check_cancelled()
            engine = self._ensure_loaded(self._engine_for(params), voice.voice_id)
            chunks = [
                engine.synthesize(sentence, voice, speed, pitch)
                for sentence in sentences
                if not self._stop_requested()
            ]
            self._last_used = monotonic()
        if self._stop_requested():
            raise self._cancelled_error()
        chunk = concat_chunks(chunks)
        elapsed_ms = (perf_counter() - started) * 1000.0
        self._cache.put(text, voice, speed, pitch, chunk)
        self._record(text, chunk, elapsed_ms, request_id, cached_hit=False)
        return self._publish(chunk, request_id, cached=False)

    @method()
    def synthesize_stream(self, params: JsonObject) -> JsonObject:
        """Speak the first sentence and open a stream for the rest.

        The point of the whole module: the caller gets audio after one sentence
        instead of after the whole answer, which is what puts the first sound
        inside the 500 ms budget.

        Args:
            params: As :meth:`synthesize`.

        Returns:
            The first chunk's descriptor plus ``stream`` - a token for
            :meth:`next_chunk` - and ``remaining``. ``stream`` is empty when the
            text was a single sentence and there is nothing more to ask for.

        Raises:
            TtsError: As :meth:`synthesize`.
        """
        text = str(params.get("text", ""))
        request_id = str(params.get("request_id", ""))
        sentences = self._sentences(text)
        if not sentences:
            return {"empty": True, "stream": "", "remaining": 0, "request_id": request_id}

        self._cancelled.clear()
        voice, speed, pitch = self._resolve_call(params)
        stream = _Stream(sentences, voice, speed, pitch, text)
        first = self._synthesize_sentence(stream, params, request_id)
        token = ""
        if not stream.done:
            token = uuid4().hex
            self._forget_stale_streams()
            self._streams[token] = stream
        reply = self._publish(first, request_id, cached=False)
        reply["stream"] = token
        reply["remaining"] = len(stream.sentences) - stream.index
        return reply

    @method()
    def next_chunk(self, params: JsonObject) -> JsonObject:
        """The next sentence of an open stream.

        Args:
            params: ``stream`` from :meth:`synthesize_stream`.

        Returns:
            The next descriptor, or ``empty: True`` with ``remaining: 0`` once
            the answer is finished - which is also what an unknown or cancelled
            token gets, because "there is no more audio" is the same fact from
            the player's point of view.
        """
        token = str(params.get("stream", ""))
        request_id = str(params.get("request_id", ""))
        stream = self._streams.get(token)
        if stream is None or self._stop_requested():
            self._streams.pop(token, None)
            return {"empty": True, "stream": "", "remaining": 0, "request_id": request_id}

        chunk = self._synthesize_sentence(stream, params, request_id)
        remaining = len(stream.sentences) - stream.index
        if stream.done:
            self._streams.pop(token, None)
        reply = self._publish(chunk, request_id, cached=False)
        reply["stream"] = "" if stream.done else token
        reply["remaining"] = remaining
        return reply

    @method()
    def cancel(self, params: JsonObject) -> JsonObject:
        """Stop producing audio for the current answer.

        What :class:`~ayris.core.events.CancelRequested` reaches. A sentence
        already inside the engine finishes - neither Piper nor Silero can be
        interrupted mid-inference - but nothing after it is synthesized, and the
        player throws away what it has queued. See the module docstring.
        """
        self._cancelled.set()
        dropped = len(self._streams)
        self._streams.clear()
        self._release_blocks()
        self.log.info("tts: синтез отменён, потоков сброшено %d", dropped)
        return {"cancelled": True, "streams_dropped": dropped, **_echo_request(params)}

    @method()
    def release(self, params: JsonObject) -> JsonObject:
        """Free a shared block the caller has finished reading.

        Args:
            params: ``block`` from a previous reply, or nothing to free them all.
        """
        name = str(params.get("block", ""))
        if not name:
            freed = len(self._blocks)
            self._release_blocks()
            return {"released": freed}
        block = self._blocks.pop(name, None)
        if block is not None:
            block.close()
        return {"released": 1 if block is not None else 0}

    @method()
    def unload(self, _params: JsonObject) -> JsonObject:
        """Drop the voice now, without waiting for the idle timer."""
        with self._lock:
            was_loaded = self._engine is not None
            self._release("requested")
        return {"unloaded": was_loaded}

    @method()
    def clear_cache(self, params: JsonObject) -> JsonObject:
        """Delete cached phrases.

        Args:
            params: ``voice`` keeps that voice's entries and drops the rest,
                which is what a voice change wants. Without it everything goes.
        """
        keep = params.get("voice")
        voice = VoiceSpec.from_params(keep) if isinstance(keep, dict) else None
        removed = self._cache.invalidate(voice)
        return {"removed": removed}

    @method()
    def status(self, _params: JsonObject) -> JsonObject:
        """Everything DevTools shows on the synthesis row."""
        with self._lock:
            engine = self._engine
            voice = self._voice
        stats = self._cache.stats()
        return {
            "loaded": engine is not None,
            "engine": self._engine_name or self._configured_engine(),
            "voice": self._voice_id() or self._configured_voice(),
            "voice_label": voice.label if voice is not None else "",
            "device": engine.device if engine is not None else "",
            "sample_rate": engine.sample_rate if engine is not None else 0,
            "available_engines": list(engine_names(available_only=True)),
            "load_ms": round(self._load_ms, 1),
            "calls": self._calls,
            "avg_synthesis_ms": round(self._average_synthesis_ms(), 1),
            "real_time_factor": round(self._overall_rtf(), 3),
            "idle_sec": round(monotonic() - self._last_used, 1),
            "idle_timeout_sec": round(self._idle_timeout(), 1),
            "open_streams": len(self._streams),
            "open_blocks": len(self._blocks),
            "cache": {
                "entries": stats.entries,
                "bytes_used": stats.bytes_used,
                "limit_bytes": stats.limit_bytes,
                "hits": stats.hits,
                "misses": stats.misses,
                "evictions": stats.evictions,
                "hit_rate": round(stats.hit_rate, 3),
            },
        }

    # -- synthesis ------------------------------------------------------

    def _synthesize_sentence(
        self,
        stream: _Stream,
        params: JsonObject,
        request_id: str,
    ) -> AudioChunk:
        """Synthesize one sentence of a stream, cache included.

        Raises:
            TtsError: Synthesis failed, or the answer was cancelled.
        """
        sentence = stream.take()
        if not sentence:
            return AudioChunk(b"")
        if self._stop_requested():
            raise self._cancelled_error()

        cached = self._cache.get(sentence, stream.voice, stream.speed, stream.pitch)
        if cached is not None:
            self._record(sentence, cached, 0.0, request_id, cached_hit=True)
            return cached

        started = perf_counter()
        with self._lock:
            self.context.check_cancelled()
            engine = self._ensure_loaded(self._engine_for(params), stream.voice.voice_id)
            chunk = engine.synthesize(sentence, stream.voice, stream.speed, stream.pitch)
            self._last_used = monotonic()
        elapsed_ms = (perf_counter() - started) * 1000.0
        self._cache.put(sentence, stream.voice, stream.speed, stream.pitch, chunk)
        self._record(sentence, chunk, elapsed_ms, request_id, cached_hit=False)
        return chunk

    def _sentences(self, text: str) -> list[str]:
        """Split an answer into speakable pieces."""
        return split_sentences(text)

    def _publish(self, chunk: AudioChunk, request_id: str, *, cached: bool) -> JsonObject:
        """Put PCM in shared memory and describe it for the reply.

        The block is kept in :attr:`_blocks` so it outlives this call: the caller
        reads it after the reply arrives. :meth:`release` frees it, and
        :meth:`cancel` and :meth:`on_stop` free whatever is left - a caller that
        crashed must not leak a mapping.
        """
        if chunk.empty:
            return {
                "empty": True,
                "cached": cached,
                "request_id": request_id,
                **chunk.metadata(),
            }
        block = SharedAudioBlock.create(
            chunk.pcm,
            sample_rate=chunk.sample_rate,
            channels=max(1, chunk.channels),
            sample_format="int16",
        )
        self._blocks[block.chunk.block] = block
        return {
            "empty": False,
            "cached": cached,
            "request_id": request_id,
            AUDIO_PARAM: block.chunk,
            "block": block.chunk.block,
            **chunk.metadata(),
        }

    def _release_blocks(self) -> None:
        """Free every shared block this worker still owns."""
        for block in self._blocks.values():
            block.close()
        self._blocks.clear()

    def _forget_stale_streams(self) -> None:
        """Drop streams nobody came back for."""
        cutoff = monotonic() - _STREAM_TTL_SEC
        stale = [token for token, stream in self._streams.items() if stream.created < cutoff]
        for token in stale:
            self._streams.pop(token, None)
        if stale:
            self.log.debug("tts: забыто %d брошенных потоков синтеза", len(stale))

    def _stop_requested(self) -> bool:
        """Whether this answer should stop producing audio."""
        return self._cancelled.is_set() or self.context.cancelled or self.context.stopping

    def _cancelled_error(self) -> TtsError:
        """The exception a cancelled synthesis raises."""
        return TtsError(
            "tts: synthesis cancelled",
            user_message="Озвучка прервана.",
        )

    # -- loading --------------------------------------------------------

    def _ensure_loaded(self, engine_name: str, voice_id: str) -> TtsEngine:
        """The engine, loading the voice first if needed. Caller holds the lock.

        Raises:
            TtsError: The load failed.
        """
        engine = self._engine
        same = self._engine_name == engine_name and self._voice_id() == voice_id
        if engine is not None and same:
            return engine
        if engine is not None and self._engine_name != engine_name:
            self._release("engine_switch")
        return self._load(engine_name, voice_id)

    def _load(self, engine_name: str, voice_id: str) -> TtsEngine:
        """Construct, size-check and load. Caller holds the lock.

        Raises:
            TtsError: Unknown engine, missing voice, or one that does not fit
                under ``ram_limit_mb``.
        """
        engine = self._engine
        if engine is None or self._engine_name != engine_name:
            engine = create_engine(engine_name)
        voice = self._voice_spec(engine_name, voice_id, type(engine))
        self._check_memory(engine, voice)

        started = perf_counter()
        try:
            engine.load(voice, self._options())
        except Exception as error:
            # The supervisor turns this into a notification: a voice that will not
            # load leaves the assistant mute, and silence with nothing on screen
            # reads as a hang.
            self.emit(
                "voice",
                {
                    "state": "failed",
                    "engine": engine_name,
                    "voice": voice_id,
                    "error": _failure_text(error),
                },
            )
            raise
        self._load_ms = (perf_counter() - started) * 1000.0
        self._engine = engine
        self._engine_name = engine_name
        self._voice = engine.voice or voice
        self._last_used = monotonic()
        # A different voice makes every cached phrase unreachable but still
        # counted against the limit, so the cache is trimmed to the new voice
        # rather than left to evict them one at a time.
        self._cache.invalidate(self._voice)
        self._pipeline.info(
            "tts: голос %s (%s) загружен за %.0f мс на %s, %d Гц",
            voice_id,
            engine_name,
            self._load_ms,
            engine.device,
            engine.sample_rate,
            extra={"request_id": ""},
        )
        self.emit(
            "voice",
            {
                "state": "loaded",
                "engine": engine_name,
                "voice": voice_id,
                "device": engine.device,
                "sample_rate": engine.sample_rate,
                "load_ms": round(self._load_ms, 1),
            },
        )
        return engine

    def _release(self, reason: str) -> None:
        """Unload the voice and give the memory back. Caller holds the lock.

        The :func:`gc.collect` is here for the same reason as in the STT worker:
        TorchScript and ONNX Runtime free their arenas in destructors, and a
        reference cycle inside the vendor object means dropping our reference
        alone leaves the memory allocated until something unrelated triggers a
        collection.
        """
        engine = self._engine
        self._engine = None
        self._engine_name = ""
        self._voice = None
        self._load_ms = 0.0
        if engine is None:
            return
        try:
            engine.unload()
        except Exception:  # unload must never fail a stop
            self.log.exception("tts: ошибка выгрузки голоса")
        gc.collect()
        self.log.info("tts: голос выгружен (%s)", reason)
        self.emit("voice", {"state": "unloaded", "reason": reason})

    def _check_memory(self, engine: TtsEngine, voice: VoiceSpec) -> None:
        """Refuse a voice that will not fit under the configured cap.

        Raises:
            TtsError: The estimate exceeds ``ram_limit_mb``.
        """
        limit_mb = self._ram_limit_mb()
        if limit_mb <= 0 or not voice.path:
            return
        needed_bytes = estimate_voice_bytes(Path(voice.path)) * engine.memory_factor
        if needed_bytes <= 0.0:
            return
        needed_mb = needed_bytes / _MB
        if needed_mb <= float(limit_mb):
            return
        raise TtsError(
            f"tts: voice {voice.voice_id} needs about {needed_mb:.0f} MB, limit is {limit_mb} MB",
            user_message=(
                f"Голос «{voice.label}» требует примерно {needed_mb:.0f} МБ, "
                f"а лимит памяти — {limit_mb} МБ. Выберите голос полегче "
                f"или поднимите лимит в настройках."
            ),
        )

    def _voice_spec(self, engine_name: str, voice_id: str, engine: type[TtsEngine]) -> VoiceSpec:
        """Build the spec for a voice named in the settings.

        An entry the engine can enumerate is used as enumerated - that carries
        the real path, language and sample rate. Anything else becomes a bare
        spec and the engine resolves it by its own conventions, which is how a
        hand-written config naming just ``ru_RU-irina-medium`` keeps working.

        ``engine`` is the class rather than an instance because
        :meth:`~ayris.audio.tts.base.TtsEngine.voices` is a classmethod: asking
        what voices exist must not require constructing anything, or the settings
        window could not list them without loading a library.

        Raises:
            TtsError: The settings name no voice at all.
        """
        if not voice_id:
            raise TtsError(
                "tts: no voice configured",
                user_message="Не выбран голос синтеза речи. Укажите его в настройках.",
            )
        directory = get_paths().tts_models_dir
        candidate = Path(voice_id)
        for spec in engine.voices(directory):
            if voice_id in {spec.voice_id, spec.path, spec.display_name}:
                return spec
        return VoiceSpec(
            engine=engine_name,
            voice_id=candidate.stem if candidate.is_absolute() else voice_id,
            path=str(candidate) if candidate.is_absolute() else "",
        )

    # -- idle -----------------------------------------------------------

    def _idle_loop(self) -> None:
        """Poll for an idle voice until the worker stops."""
        while not self._idle_stop.wait(_IDLE_POLL):
            self._drop_if_idle()

    def _drop_if_idle(self) -> None:
        """Unload the voice if it has been unused for the configured time.

        Split out of :meth:`_idle_loop` so a test can ask directly instead of
        sleeping through a timeout measured in minutes.
        """
        timeout = self._idle_timeout()
        if timeout <= 0.0:
            return
        with self._lock:
            if self._engine is None:
                return
            if monotonic() - self._last_used < timeout:
                return
            if self._streams:
                # An answer is still being handed out sentence by sentence.
                return
            self._release("idle")

    def _idle_timeout(self) -> float:
        """Seconds of silence before the voice goes away. ``0`` disables it."""
        configured = _as_float(self.params.get("model_idle_sec"), 0.0)
        if configured <= 0.0:
            return 0.0
        if bool(self.params.get("eco_mode", False)):
            configured *= _ECO_IDLE_FACTOR
        return max(_MIN_IDLE_SEC, configured)

    # -- metrics --------------------------------------------------------

    def _record(
        self,
        text: str,
        chunk: AudioChunk,
        elapsed_ms: float,
        request_id: str,
        *,
        cached_hit: bool,
    ) -> None:
        """Log one call and fold it into the running averages.

        A cache hit is counted as a call with zero synthesis time on purpose: the
        interesting number in DevTools is what the user waited for, and the hit
        rate next to it explains the rest.
        """
        self._calls += 1
        self._total_synthesis_ms += elapsed_ms
        self._total_audio_ms += chunk.duration_ms
        self._pipeline.info(
            "tts: %d симв. → %.0f мс аудио %s (движок %s)",
            len(text),
            chunk.duration_ms,
            "из кэша" if cached_hit else f"за {elapsed_ms:.0f} мс",
            self._engine_name or self._configured_engine(),
            extra={"request_id": request_id},
        )
        self.emit(
            "metrics",
            {
                "request_id": request_id,
                "chars": len(text),
                "synthesis_ms": round(elapsed_ms, 1),
                "audio_ms": round(chunk.duration_ms, 1),
                "cached": cached_hit,
                "engine": self._engine_name,
            },
        )

    def _average_synthesis_ms(self) -> float:
        """Mean synthesis time over the worker's life."""
        return self._total_synthesis_ms / self._calls if self._calls else 0.0

    def _overall_rtf(self) -> float:
        """Synthesis time over audio produced. Below 1.0 is faster than real time."""
        return self._total_synthesis_ms / self._total_audio_ms if self._total_audio_ms else 0.0

    # -- settings -------------------------------------------------------

    def _resolve_call(self, params: JsonObject) -> tuple[VoiceSpec, float, float]:
        """Voice, speed and pitch for one request.

        Raises:
            TtsError: The engine is unknown or no voice is configured.
        """
        engine_name = self._engine_for(params)
        voice_id = str(params.get("voice", "")) or self._configured_voice()
        with self._lock:
            loaded = self._voice
            if (
                loaded is not None
                and self._engine_name == engine_name
                and loaded.voice_id == voice_id
            ):
                voice = loaded
            else:
                voice = self._voice_spec(engine_name, voice_id, engine_class(engine_name))
        options = self._options()
        speed = clamp_speed(_as_float(params.get("speed"), options.speed))
        pitch = clamp_pitch(_as_float(params.get("pitch"), options.pitch))
        return voice, speed, pitch

    def _options(self) -> TtsOptions:
        """Turn worker params into engine options."""
        return TtsOptions.from_params(self.params)

    def _engine_for(self, params: JsonObject) -> str:
        """Engine named in the request, or the configured one."""
        return str(params.get("engine", "")) or self._configured_engine()

    def _configured_engine(self) -> str:
        """The engine the settings picked, defaulting to Piper."""
        return str(self.params.get("engine", "")) or "piper"

    def _configured_voice(self) -> str:
        """The voice the settings picked."""
        return str(self.params.get("voice", ""))

    def _voice_id(self) -> str:
        """Identifier of the loaded voice, empty when nothing is loaded."""
        return self._voice.voice_id if self._voice is not None else ""

    def _ram_limit_mb(self) -> int:
        """``performance.ram_limit_mb``; ``0`` means the user turned it off."""
        return int(_as_float(self.params.get("ram_limit_mb"), 0.0))

    def _cache_limit_bytes(self) -> int:
        """``voice.tts.cache_size_mb`` in bytes; ``0`` disables the cache."""
        return max(0, int(_as_float(self.params.get("cache_size_mb"), 0.0))) * _MB


def _as_float(value: object, default: float) -> float:
    """Read a params value as a float without trusting its type."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    return float(value)


def _echo_request(params: JsonObject) -> JsonObject:
    """Carry the caller's ``request_id`` into a reply that has nothing else."""
    request_id = str(params.get("request_id", ""))
    return {"request_id": request_id} if request_id else {}


def _failure_text(error: Exception) -> str:
    """What to show the user when a voice will not load.

    :class:`AyrisError` already carries a sentence written for a person, so it is
    used as is. Anything else - a vendor ``RuntimeError``, a missing DLL - has a
    message meant for a log, and putting it in a toast on its own explains
    nothing, so it is prefixed.
    """
    if isinstance(error, AyrisError):
        return error.user_message
    return f"Не удалось загрузить голос: {error}"


def translate_tts_event(kind: str, payload: JsonObject) -> Event | None:
    """Turn a TTS worker event into a bus event.

    ``metrics`` returns ``None``: the numbers are already in the pipeline log and
    in :meth:`TtsWorker.status`, and one bus event per sentence in front of every
    subscriber buys nothing. Speaking itself is announced by the *player*, which
    is the only thing that knows when sound actually starts.
    """
    if kind == "voice":
        return _voice_notification(payload)
    return None


def _voice_notification(payload: JsonObject) -> Event | None:
    """Tell the user when a voice failed to load, stay quiet otherwise."""
    from ayris.core.events import NotificationRequested

    if str(payload.get("state", "")) != "failed":
        return None
    return NotificationRequested(
        title="Синтез речи",
        message=str(payload.get("error", "")) or "Не удалось загрузить голос.",
        level="error",
        timeout_ms=8000,
    )


#: What the supervisor registers for this worker.
EVENT_TRANSLATOR: Final = translate_tts_event
