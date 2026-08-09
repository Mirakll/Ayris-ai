"""Everything about wake word detection that does not depend on the engine.

An engine answers one question about one frame.  Between the microphone and
that question there is a surprising amount of work, and all of it lives here:

*Framing and rate.*  Capture delivers whatever the driver produced - 20 ms
blocks normally, an odd number of samples when a 44.1 kHz device is being
resampled.  Engines accept exactly one block length and exactly one rate.  So
the detector holds a :class:`~ayris.audio.capture.Resampler` and a
:class:`~ayris.audio.vad.FrameSplitter` and hands the engine only frames it can
take.  Doing this once here rather than three times in the engines is the reason
none of them contains a ring buffer.

*Thresholds.*  Engines return raw scores; the detector compares each score
against the threshold of the specific variant it belongs to.  That is what makes
an unlimited list of variants with individual sensitivities work at all - one
global threshold could not express "fire on ``айрис`` readily but on ``ирис``
only when it is unmistakable", and that distinction is the whole point of having
variants.

*Debounce.*  A wake word takes half a second to say and the engine scores every
80 ms, so one utterance produces a run of detections, not one.  After a trigger
the detector ignores everything for :attr:`WakeWordSettings.debounce_sec`,
counted **in audio time**: the position in the stream, not the wall clock.  A
machine that stalls for a second under load must not have its window slide with
it, and a test feeding a file must not have to sleep in real time.

*The thread.*  Inference is tens of milliseconds of matrix arithmetic and it
absolutely may not run in the sounddevice callback - a callback that overruns
its block period drops audio, and dropped audio is exactly what the wake word
needs to hear.  :meth:`WakeWordDetector.push` therefore only puts bytes in a
queue; a dedicated thread does the work.  If the queue fills - a machine too
slow to keep up - blocks are dropped and counted rather than allowed to grow
into unbounded latency.

*The counters.*  Every rejection is counted separately by reason: below the
phrase's threshold, or inside the debounce window.  Those two numbers are what a
user tuning sensitivity actually needs; "it fires too often" and "it fires twice
per phrase" have different fixes and look identical from the outside.
"""

from __future__ import annotations

import queue
import threading
from contextlib import suppress
from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import TYPE_CHECKING, Any, Final

from ayris.audio.capture import Resampler
from ayris.audio.vad import FrameSplitter
from ayris.audio.wake_word.base import (
    PTT_PHRASE,
    WAKE_SAMPLE_RATE,
    ModelSpec,
    WakeDetection,
    WakePhrase,
    WakeWordEngine,
    create_engine,
    normalise_phrase,
)
from ayris.core.errors import AyrisError, WakeWordError
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

__all__ = [
    "DEFAULT_DEBOUNCE_MS",
    "MAX_DEBOUNCE_MS",
    "MIN_DEBOUNCE_MS",
    "WakeStats",
    "WakeWordCallbacks",
    "WakeWordDetector",
    "WakeWordSettings",
]

_log: Final = get_logger(__name__)

#: Recommended debounce, and the default in the settings model.  Long enough to
#: cover one utterance of a two-syllable word plus the tail the engine scores
#: after it, short enough that a user correcting themselves is not ignored.
DEFAULT_DEBOUNCE_MS: Final = 1500

#: Below this the window stops covering a single utterance and the same phrase
#: fires twice.
MIN_DEBOUNCE_MS: Final = 200

#: Above this a user who was not heard the first time waits long enough to
#: assume the assistant is broken.
MAX_DEBOUNCE_MS: Final = 10000

#: Blocks the queue holds before the detector starts dropping them.  About a
#: second of audio at the usual block size: enough to ride out a scheduling
#: hiccup, short enough that a machine which cannot keep up stays responsive
#: instead of falling further behind with every block.
DEFAULT_QUEUE_BLOCKS: Final = 64

#: How long :meth:`WakeWordDetector.stop` waits for the thread to notice.
_JOIN_TIMEOUT_SEC: Final = 2.0

#: Source of a manual trigger when the caller does not name one.
_MANUAL_SOURCE: Final = "hotkey"


@dataclass(frozen=True, slots=True)
class WakeWordSettings:
    """Everything the detector needs, in one immutable object.

    Attributes:
        enabled: Whether to score audio at all.  ``False`` is the Push-to-Talk
            microphone mode: the engine is never loaded, :meth:`push` costs
            nothing, and :meth:`WakeWordDetector.trigger_manual` still works.
        engine: Which implementation to load - see
            :data:`~ayris.audio.wake_word.base.ENGINE_ENTRYPOINTS`.
        phrases: Variants of the wake word.  Unlimited in number; each carries
            its own sensitivity.
        debounce_ms: How long to ignore detections after a trigger, clamped to
            :data:`MIN_DEBOUNCE_MS` - :data:`MAX_DEBOUNCE_MS`.
        source_rate: Rate the audio arrives at.  Equal to
            :data:`~ayris.audio.wake_word.base.WAKE_SAMPLE_RATE` in Ayris,
            because capture normalises before the detector sees anything, but
            the resampler is here so that a caller feeding a file does not have
            to.
        models_dir: Where engine weights live, normally
            :attr:`~ayris.core.paths.AppPaths.wake_models_dir`.
        access_key: Vendor credential for an engine that needs one.  Resolved by
            the caller from the credential store, never read from a settings
            file.
        options: Engine-specific extras, passed through to
            :class:`~ayris.audio.wake_word.base.ModelSpec`.
        queue_blocks: Backlog the processing thread may accumulate before blocks
            start being dropped.
    """

    enabled: bool = True
    engine: str = "openwakeword"
    phrases: tuple[WakePhrase, ...] = ()
    debounce_ms: int = DEFAULT_DEBOUNCE_MS
    source_rate: int = WAKE_SAMPLE_RATE
    models_dir: Path | None = None
    access_key: str = ""
    options: Mapping[str, Any] = field(default_factory=dict)
    queue_blocks: int = DEFAULT_QUEUE_BLOCKS

    @property
    def debounce_sec(self) -> float:
        """:attr:`debounce_ms` in seconds, clamped to the usable range."""
        clamped = max(MIN_DEBOUNCE_MS, min(MAX_DEBOUNCE_MS, self.debounce_ms))
        return clamped / 1000.0

    def spec(self, phrases: Sequence[WakePhrase] | None = None) -> ModelSpec:
        """The specification to hand :meth:`WakeWordEngine.load`."""
        return ModelSpec(
            phrases=tuple(self.phrases if phrases is None else phrases),
            models_dir=self.models_dir,
            access_key=self.access_key,
            options=self.options,
        )


@dataclass(frozen=True, slots=True)
class WakeStats:
    """Counters accumulated since the detector last started.

    The three rejection counters are the point of this object.  ``candidates``
    is how often the engine thought it heard something; ``fired`` is how often
    that turned into an activation; the difference is split between
    ``below_threshold`` (the variant's sensitivity was too low) and ``debounced``
    (it was the same utterance again).  A user complaining about false positives
    is looking at a high ``fired``; a user complaining that nothing happens is
    looking at a high ``below_threshold``, and those need opposite fixes.

    Attributes:
        engine: Name of the selected engine.
        running: Whether the processing thread is alive.
        loaded: Whether a model is in memory.  ``False`` with ``running`` true
            means loading failed - see :attr:`error`.
        error: Why the engine is not available, in Russian, or an empty string.
        frames: Frames handed to the engine.
        audio_sec: Audio scored, in seconds.  The detector's clock.
        candidates: Frames the engine returned a detection for.
        fired: Activations published, manual triggers included.
        manual: Activations that came from Push-to-Talk rather than the voice.
        below_threshold: Candidates rejected because no variant's threshold was
            reached.
        debounced: Candidates rejected because one had just fired.
        dropped: Blocks discarded because the queue was full.
        errors: Frames the engine raised on.
        last_phrase: Variant that last fired.
        last_score: Its score.
        last_fired_sec: Position in the stream where it fired.
        avg_ms: Mean time one frame spends in the engine.
        max_ms: Worst such time.  Both matter: the mean says whether the machine
            can keep up at all, the maximum says whether it stutters.
        fired_by_phrase: Activations per variant.
        rejected_by_phrase: Best-scoring variant of each rejected candidate.
            Read together, these two say which variant is carrying the load and
            which one is only ever *almost* triggering.
    """

    engine: str = ""
    running: bool = False
    loaded: bool = False
    error: str = ""
    frames: int = 0
    audio_sec: float = 0.0
    candidates: int = 0
    fired: int = 0
    manual: int = 0
    below_threshold: int = 0
    debounced: int = 0
    dropped: int = 0
    errors: int = 0
    last_phrase: str = ""
    last_score: float = 0.0
    last_fired_sec: float = 0.0
    avg_ms: float = 0.0
    max_ms: float = 0.0
    fired_by_phrase: Mapping[str, int] = field(default_factory=dict)
    rejected_by_phrase: Mapping[str, int] = field(default_factory=dict)

    @property
    def rejected(self) -> int:
        """Candidates that did not become an activation."""
        return self.below_threshold + self.debounced

    @property
    def false_positive_rate(self) -> float:
        """Rejections per minute of audio.

        Expressed per minute rather than per frame because that is the unit a
        user can feel: "it lit up four times while I was on a call" is a rate,
        not a ratio.
        """
        if self.audio_sec <= 0.0:
            return 0.0
        return self.rejected * 60.0 / self.audio_sec


def _ignore_detection(_detection: WakeDetection) -> None:
    """Default :attr:`WakeWordCallbacks.on_detected`: drop it."""


def _ignore_error(_error: AyrisError) -> None:
    """Default :attr:`WakeWordCallbacks.on_error`: the detector already logged."""


@dataclass(frozen=True, slots=True)
class WakeWordCallbacks:
    """What the detector reports, and where.

    Both run on the processing thread and must not block: whatever they do
    happens between two frames of live audio.  In Ayris they only put a message
    on the worker's pipe.

    Attributes:
        on_detected: An activation, voice or manual.  Fired after the thresholds
            and the debounce, so every call is a real one.
        on_error: The engine failed - at load time, or mid-stream.  The detector
            keeps running and keeps counting; the caller decides whether to tell
            the user.
    """

    on_detected: Callable[[WakeDetection], None] = _ignore_detection
    on_error: Callable[[AyrisError], None] = _ignore_error


class WakeWordDetector:
    """Runs an engine over a live stream and decides when the assistant wakes.

    Lifecycle is ``start`` - ``push`` many times - ``stop``.  The phrase list can
    be changed at any point in between, from any thread, without interrupting
    the audio: :meth:`add_phrase`, :meth:`remove_phrase` and
    :meth:`set_sensitivity` record the change and the processing thread applies
    it before the next frame.  That indirection is not decoration - engines are
    not thread-safe, and reloading a model from the GUI thread while the audio
    thread is calling into it is a segfault, not an exception.

    Args:
        settings: Initial configuration.
        callbacks: Where activations and failures go.
        engine: A ready engine, instead of one built from
            :attr:`WakeWordSettings.engine`.  Used by the tests, which need to
            drive the decision logic with known scores rather than a model.
    """

    __slots__ = (
        "_audio_sec",
        "_below_threshold",
        "_callbacks",
        "_candidates",
        "_debounced",
        "_dropped",
        "_engine",
        "_engine_ms",
        "_error",
        "_errors",
        "_fired",
        "_fired_counts",
        "_frames",
        "_last_fire_at",
        "_last_fired",
        "_last_phrase",
        "_last_score",
        "_lock",
        "_manual",
        "_max_ms",
        "_owned_engine",
        "_pending",
        "_phrases",
        "_queue",
        "_rejected_counts",
        "_resampler",
        "_settings",
        "_splitter",
        "_started",
        "_stop",
        "_thread",
    )

    def __init__(
        self,
        settings: WakeWordSettings,
        callbacks: WakeWordCallbacks | None = None,
        engine: WakeWordEngine | None = None,
    ) -> None:
        self._settings = settings
        self._callbacks = callbacks or WakeWordCallbacks()
        self._engine = engine
        # An injected engine belongs to the caller: stopping the detector must
        # not unload a model somebody else is still holding.
        self._owned_engine = engine is None
        self._lock = threading.Lock()
        self._phrases: tuple[WakePhrase, ...] = settings.phrases
        self._pending: tuple[WakePhrase, ...] | None = None
        self._queue: queue.Queue[bytes | None] = queue.Queue(maxsize=settings.queue_blocks)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._started = False
        self._error = ""
        self._resampler = Resampler(settings.source_rate, WAKE_SAMPLE_RATE)
        self._splitter: FrameSplitter | None = None
        # The audio clock, and the debounce windows measured against it.
        # ``-1.0`` rather than zero: at position zero the stream has not started,
        # and a window anchored there would swallow the first utterance.
        self._audio_sec = 0.0
        self._last_fire_at = -1.0
        self._last_fired: dict[str, float] = {}
        # Session counters.  Written only under the lock, so that the snapshot
        # in :attr:`stats` is internally consistent rather than a mix of two
        # moments.
        self._frames = 0
        self._candidates = 0
        self._fired = 0
        self._manual = 0
        self._below_threshold = 0
        self._debounced = 0
        self._dropped = 0
        self._errors = 0
        self._engine_ms = 0.0
        self._max_ms = 0.0
        self._last_phrase = ""
        self._last_score = 0.0
        self._fired_counts: dict[str, int] = {}
        self._rejected_counts: dict[str, int] = {}

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Load the engine and start the processing thread.

        A load failure is not raised: the audio worker must keep capturing even
        when no wake model is installed, or a user whose model failed to
        download loses the level meter and dictation along with the wake word.
        The failure is logged, reported through
        :attr:`WakeWordCallbacks.on_error`, and left visible in
        :attr:`WakeStats.error`.
        """
        if self._started:
            return
        self._started = True
        self._error = ""
        if not self._settings.enabled:
            # Push-to-Talk only.  No engine, no thread, no CPU - but
            # trigger_manual still works, which is the entire point.
            _log.info("wake word detection disabled by the microphone mode")
            return
        if not self._load_engine():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ayris-wake", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and release the model.  Safe to call twice."""
        if not self._started:
            return
        self._started = False
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None:
            # The sentinel is what actually wakes the thread; the event only
            # tells it not to start another iteration.
            with suppress(queue.Full):
                self._queue.put_nowait(None)
            thread.join(timeout=_JOIN_TIMEOUT_SEC)
            if thread.is_alive():  # pragma: no cover - a wedged engine
                _log.warning("wake word thread did not stop within %.1fs", _JOIN_TIMEOUT_SEC)
        engine = self._engine
        if engine is not None and self._owned_engine:
            engine.unload()
            self._engine = None
        self._drain()

    def reset(self) -> None:
        """Forget the stream: buffers, the audio clock and the debounce window.

        Called when capture restarts.  The counters survive on purpose - they
        are per session, and a user tuning sensitivity switches devices while
        doing it.
        """
        with self._lock:
            self._audio_sec = 0.0
            self._last_fire_at = -1.0
            self._last_fired.clear()
        self._resampler.reset()
        if self._splitter is not None:
            self._splitter.reset()
        engine = self._engine
        if engine is not None:
            engine.reset()
        self._drain()

    def configure(self, settings: WakeWordSettings) -> None:
        """Apply new settings, restarting the engine only if it has to.

        A changed phrase list or sensitivity goes through the cheap path - the
        engine is told about it between frames.  A changed engine, model
        directory or source rate cannot: those are constructor arguments, so the
        detector stops and starts.
        """
        previous = self._settings
        self._settings = settings
        heavy = (
            settings.enabled != previous.enabled
            or settings.engine != previous.engine
            or settings.models_dir != previous.models_dir
            or settings.source_rate != previous.source_rate
            or settings.access_key != previous.access_key
            or dict(settings.options) != dict(previous.options)
        )
        if heavy and self._started:
            self.stop()
            self.set_phrases(settings.phrases)
            self.start()
            return
        self.set_phrases(settings.phrases)

    # -------------------------------------------------------------------- audio

    def push(self, pcm: bytes | bytearray | memoryview) -> None:
        """Hand captured audio to the detector.

        Called from the capture thread, so it does the least it possibly can:
        one queue insert.  Everything else - resampling, framing, inference -
        happens on the processing thread, because a callback that overruns its
        block period costs the recording the very audio the wake word is in.

        A full queue means the machine cannot keep up.  The block is dropped and
        counted rather than queued: a backlog that grows makes the wake word
        respond to something the user said seconds ago, which is worse than
        missing it.
        """
        if not self._started or self._thread is None or not pcm:
            return
        try:
            self._queue.put_nowait(bytes(pcm))
        except queue.Full:
            with self._lock:
                self._dropped += 1

    def trigger_manual(self, source: str = _MANUAL_SOURCE) -> WakeDetection:
        """Activate without the microphone: the Push-to-Talk entry point.

        The global hotkey that calls this is task 37; what lives here is the
        path it will use, so that a manual activation is indistinguishable
        downstream from a spoken one - same event, same segmenter, same
        counters.

        Deliberately **not** debounced.  The window exists to collapse one
        utterance into one activation; a user pressing a key twice meant it
        twice.  The window is still reset, so that pressing the key and then
        saying the word does not fire again.

        Args:
            source: What triggered it, for the log and for
                :attr:`WakeDetection.engine`.

        Returns:
            The published detection, so a caller can log what it produced.
        """
        with self._lock:
            position = self._audio_sec
            self._last_fire_at = position
            self._last_fired[PTT_PHRASE] = position
            self._fired += 1
            self._manual += 1
            self._fired_counts[PTT_PHRASE] = self._fired_counts.get(PTT_PHRASE, 0) + 1
            self._last_phrase = PTT_PHRASE
            self._last_score = 1.0
        detection = WakeDetection(
            phrase=PTT_PHRASE,
            score=1.0,
            timestamp=position,
            engine=source,
            scores={PTT_PHRASE: 1.0},
        )
        _log.info("wake word triggered manually from %s", source)
        self._emit(detection)
        return detection

    # ------------------------------------------------------------------ phrases

    @property
    def settings(self) -> WakeWordSettings:
        """The configuration in force.

        Read rather than stored by the caller because :meth:`configure` may have
        replaced it since, and the debounce window a user is looking at in the
        settings window has to be the one the detector is actually applying.
        """
        return self._settings

    @property
    def phrases(self) -> tuple[WakePhrase, ...]:
        """The variants currently configured."""
        with self._lock:
            return self._phrases

    def set_phrases(self, phrases: Iterable[WakePhrase]) -> None:
        """Replace the whole list.

        Takes effect on the next frame.  The engine is only rebuilt if it is one
        that bakes thresholds in - see
        :attr:`~ayris.audio.wake_word.base.WakeWordEngine.reload_on_sensitivity`.
        """
        updated = tuple(phrases)
        with self._lock:
            self._phrases = updated
            self._pending = updated

    def add_phrase(self, phrase: WakePhrase) -> None:
        """Add a variant, or replace one with the same text.

        The list is unlimited by design: a user whose name for the assistant is
        misheard adds the misheard form rather than being told to pronounce it
        differently.
        """
        with self._lock:
            kept = tuple(item for item in self._phrases if item.text != phrase.text)
            self._phrases = (*kept, phrase)
            self._pending = self._phrases

    def remove_phrase(self, text: str) -> bool:
        """Drop a variant.

        Returns:
            Whether it was there.
        """
        folded = normalise_phrase(text)
        with self._lock:
            kept = tuple(item for item in self._phrases if item.text != folded)
            if len(kept) == len(self._phrases):
                return False
            self._phrases = kept
            self._pending = kept
            self._last_fired.pop(folded, None)
        return True

    def set_sensitivity(self, text: str, sensitivity: float) -> bool:
        """Move one variant's sensitivity without touching the others.

        This is the settings slider, and it is expected to work while the
        microphone is live - a user tunes sensitivity by saying the word and
        watching what happens, which is impossible if every change restarts
        capture.

        Returns:
            Whether the variant exists.

        Raises:
            WakeWordError: If ``sensitivity`` is outside 0.0-1.0.
        """
        folded = normalise_phrase(text)
        with self._lock:
            found = False
            updated: list[WakePhrase] = []
            for item in self._phrases:
                if item.text == folded:
                    updated.append(item.with_sensitivity(sensitivity))
                    found = True
                else:
                    updated.append(item)
            if not found:
                return False
            self._phrases = tuple(updated)
            self._pending = self._phrases
        return True

    # -------------------------------------------------------------------- stats

    @property
    def stats(self) -> WakeStats:
        """A snapshot of the counters, safe to read from any thread."""
        engine = self._engine
        thread = self._thread
        with self._lock:
            frames = self._frames
            average = self._engine_ms / frames if frames else 0.0
            return WakeStats(
                engine=self._settings.engine,
                running=thread is not None and thread.is_alive(),
                loaded=engine is not None and engine.loaded,
                error=self._error,
                frames=frames,
                audio_sec=round(self._audio_sec, 3),
                candidates=self._candidates,
                fired=self._fired,
                manual=self._manual,
                below_threshold=self._below_threshold,
                debounced=self._debounced,
                dropped=self._dropped,
                errors=self._errors,
                last_phrase=self._last_phrase,
                last_score=round(self._last_score, 4),
                last_fired_sec=round(max(self._last_fire_at, 0.0), 3),
                avg_ms=round(average, 3),
                max_ms=round(self._max_ms, 3),
                fired_by_phrase=dict(self._fired_counts),
                rejected_by_phrase=dict(self._rejected_counts),
            )

    # ------------------------------------------------------------------ internals

    def _load_engine(self) -> bool:
        """Build and load the engine, converting failure into a report.

        Returns:
            Whether the engine is ready.
        """
        settings = self._settings
        try:
            if self._engine is None:
                self._engine = create_engine(settings.engine)
            engine = self._engine
            if not engine.loaded:
                engine.load(settings.spec(self.phrases))
        except AyrisError as exc:
            self._error = exc.user_message
            _log.warning("wake word engine %s unavailable: %s", settings.engine, exc.technical)
            self._callbacks.on_error(exc)
            return False
        except Exception as exc:  # pragma: no cover - a vendor raising something new
            error = WakeWordError(
                f"wake word engine {settings.engine} failed to load: {exc}",
                user_message="Не удалось запустить распознавание слова активации.",
            )
            self._error = error.user_message
            _log.exception("wake word engine %s failed to load", settings.engine)
            self._callbacks.on_error(error)
            return False
        self._splitter = FrameSplitter(self._engine.frame_bytes)
        return True

    def _run(self) -> None:
        """The processing thread: dequeue, frame, score, decide."""
        while not self._stop.is_set():
            block = self._queue.get()
            if block is None:
                break
            try:
                self._consume(block)
            except Exception:  # pragma: no cover - defensive
                # A thread that dies takes the wake word with it silently, and
                # the user sees an assistant that simply stopped answering.
                _log.exception("wake word processing failed")
                with self._lock:
                    self._errors += 1

    def _consume(self, block: bytes) -> None:
        """Resample one block, cut it into frames and score each one."""
        splitter = self._splitter
        engine = self._engine
        if splitter is None or engine is None:
            return
        self._apply_pending(engine)
        pcm = self._resampler.process(block)
        if not pcm:
            return
        for frame in splitter.push(pcm):
            self._score(engine, frame)

    def _score(self, engine: WakeWordEngine, frame: bytes) -> None:
        """Run one frame through the engine and act on what comes back."""
        started = perf_counter()
        try:
            detection = engine.process(frame)
        except AyrisError as exc:
            with self._lock:
                self._errors += 1
            _log.warning("wake word frame rejected: %s", exc.technical)
            self._callbacks.on_error(exc)
            return
        elapsed_ms = (perf_counter() - started) * 1000.0

        duration = engine.frame_samples / engine.sample_rate
        with self._lock:
            self._frames += 1
            self._audio_sec += duration
            self._engine_ms += elapsed_ms
            self._max_ms = max(self._max_ms, elapsed_ms)
            position = self._audio_sec
            if detection is None:
                return
            self._candidates += 1
            accepted = self._decide(detection, position)
        if accepted is not None:
            self._emit(accepted)

    def _decide(self, detection: WakeDetection, position: float) -> WakeDetection | None:
        """Turn a raw candidate into an activation, or count why it is not.

        Runs with the lock held: it reads the phrase list and writes every
        counter.  Cheap enough for that - a handful of comparisons over a list
        that is three entries long in practice.
        """
        scores = detection.all_scores()
        winner: WakePhrase | None = None
        best_margin = 0.0
        best_score = 0.0
        for phrase in self._phrases:
            if not phrase.enabled:
                continue
            score = scores.get(phrase.text)
            if score is None:
                continue
            margin = score - phrase.threshold
            if margin >= 0.0 and (winner is None or margin > best_margin):
                winner, best_margin, best_score = phrase, margin, score

        if winner is None:
            # The engine heard something; no variant was configured loosely
            # enough to act on it.  This is the counter a user watches while
            # deciding whether to raise a sensitivity.
            self._below_threshold += 1
            self._note_rejection(detection.phrase)
            self._last_score = detection.score
            return None

        if self._is_debounced(winner.text, position):
            self._debounced += 1
            self._note_rejection(winner.text)
            return None

        self._last_fire_at = position
        self._last_fired[winner.text] = position
        self._fired += 1
        self._fired_counts[winner.text] = self._fired_counts.get(winner.text, 0) + 1
        self._last_phrase = winner.text
        self._last_score = best_score
        return replace(detection, phrase=winner.text, score=best_score, timestamp=position)

    def _is_debounced(self, phrase: str, position: float) -> bool:
        """Whether an activation this close to the last one should be ignored.

        Two windows, both in audio time.  The global one collapses an utterance
        that scored under several variants at once - "айрис" and "аирис" are
        near-identical to a model, and firing twice would start two
        conversations.  The per-phrase one is what survives a user who added a
        variant so loose it triggers on the tail of its own activation.
        """
        window = self._settings.debounce_sec
        if self._last_fire_at >= 0.0 and position - self._last_fire_at < window:
            return True
        previous = self._last_fired.get(phrase)
        return previous is not None and position - previous < window

    def _note_rejection(self, phrase: str) -> None:
        """Record which variant was closest when a candidate was refused."""
        if not phrase:
            return
        self._rejected_counts[phrase] = self._rejected_counts.get(phrase, 0) + 1

    def _apply_pending(self, engine: WakeWordEngine) -> None:
        """Hand a changed phrase list to the engine, between frames.

        Only ever called from the processing thread, which is the only thread
        allowed to touch the engine at all.
        """
        with self._lock:
            pending = self._pending
            self._pending = None
        if pending is None:
            return
        if not engine.reload_on_sensitivity and self._same_models(engine, pending):
            # Nothing the engine cares about changed - only the numbers the
            # detector itself compares against.  Reloading a model to change a
            # threshold would drop half a second of audio for nothing.
            return
        try:
            engine.update_phrases(pending)
        except AyrisError as exc:
            self._error = exc.user_message
            _log.warning("wake word phrases not applied: %s", exc.technical)
            self._callbacks.on_error(exc)

    @staticmethod
    def _same_models(engine: WakeWordEngine, phrases: Sequence[WakePhrase]) -> bool:
        """Whether the engine is already loaded with exactly these variants."""
        loaded = {(item.text, item.engine_model) for item in engine.phrases}
        wanted = {(item.text, item.engine_model) for item in phrases if item.enabled}
        return loaded == wanted

    def _emit(self, detection: WakeDetection) -> None:
        """Publish an activation, never letting a consumer kill the thread."""
        try:
            self._callbacks.on_detected(detection)
        except Exception:  # pragma: no cover - defensive
            _log.exception("wake word consumer failed")

    def _drain(self) -> None:
        """Empty the queue so a restart does not score stale audio."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    def __repr__(self) -> str:
        stats = self.stats
        return (
            f"WakeWordDetector(engine={stats.engine!r}, phrases={len(self.phrases)}, "
            f"fired={stats.fired}, rejected={stats.rejected})"
        )
