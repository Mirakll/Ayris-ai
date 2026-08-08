"""Turning a stream of speech/silence frames into whole phrases.

:mod:`ayris.audio.vad` answers "is this 20 ms frame speech".  That answer alone
is unusable: it flickers on plosives, it goes quiet between syllables, and it
says nothing about where a phrase begins.  The segmenter is the state machine
that turns it into the two events the rest of Ayris cares about -
``SpeechStarted`` and ``SpeechEnded`` with a finished chunk of audio.

Four rules, each of which exists because of a specific failure:

**Onset needs confirmation.**  A single speech frame is a mouse click or a chair
creaking.  :attr:`SegmenterSettings.start_frames` frames in a row are required
before a phrase is declared, which costs ~60 ms of latency and removes almost
every false start.

**The phrase starts before the detector notices.**  By the time the onset is
confirmed the speaker is already 60 ms in, and the first consonant is usually
below the gate anyway.  The segmenter therefore keeps a rolling pre-roll of the
last :attr:`SegmenterSettings.pre_roll_ms` and prepends it, so recognition sees
"айрис открой браузер" and not "рис открой браузер".

**Pauses inside a sentence are not the end of it.**  People stop for a third of
a second mid-thought.  Only :attr:`SegmenterSettings.silence_ms` of continuous
silence - 700 ms by default - closes a phrase, and any speech frame resets that
counter.

**Nothing records forever.**  A microphone left next to a television would
otherwise grow one unbounded segment until memory runs out.  At
:attr:`SegmenterSettings.max_utterance_ms` the segment is cut and handed over
with ``reason="max_length"``, and collection continues immediately so the
speaker loses nothing but the seam.

The segmenter owns its audio: it works on the frames pushed into it and needs no
access to the capture ring buffer, which is what makes it testable against a WAV
file in a few lines and identical in production and in tests.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.audio.ring_buffer import DEFAULT_PRE_ROLL_MS, SAMPLE_WIDTH
from ayris.audio.vad import VadFrame, VadSettings, VadStream
from ayris.core.errors import AudioError

if TYPE_CHECKING:
    from collections.abc import Callable

    from ayris.audio.vad import Vad

__all__ = [
    "DEFAULT_MAX_UTTERANCE_MS",
    "DEFAULT_MIN_SPEECH_MS",
    "DEFAULT_SILENCE_MS",
    "EndReason",
    "SegmentState",
    "Segmenter",
    "SegmenterCallbacks",
    "SegmenterSettings",
    "SegmenterStats",
    "SpeechSegment",
    "SpeechStart",
    "segment_pcm",
]

_log = logging.getLogger(__name__)

#: Continuous silence that ends a phrase.  Section 8.1 offers 200-5000 ms in the
#: settings; 700 ms is long enough for a mid-sentence pause and short enough
#: that the assistant does not feel sluggish.
DEFAULT_SILENCE_MS: Final = 700

#: Below this much *speech* a segment is discarded as a click or a cough.
DEFAULT_MIN_SPEECH_MS: Final = 300

#: Hard cap on one segment.  Matches ``voice.audio_input.max_utterance_sec``.
DEFAULT_MAX_UTTERANCE_MS: Final = 30_000

#: Speech frames in a row required to declare a phrase started.
DEFAULT_START_FRAMES: Final = 3

#: Trailing silence kept in the handed-over segment.  Recognisers behave better
#: with a little room after the last word than with a hard cut, but keeping the
#: full ``silence_ms`` would pad every phrase with most of a second of nothing.
DEFAULT_TAIL_MS: Final = 200


class SegmentState(StrEnum):
    """What the state machine is doing."""

    #: Waiting for an onset; audio goes into the pre-roll ring and nowhere else.
    IDLE = "idle"
    #: Collecting a phrase.
    SPEECH = "speech"


class EndReason(StrEnum):
    """Why a segment was closed."""

    #: Enough continuous silence followed the speech.  The normal path.
    SILENCE = "silence"
    #: The segment hit :attr:`SegmenterSettings.max_utterance_ms`.
    MAX_LENGTH = "max_length"
    #: It held less speech than :attr:`SegmenterSettings.min_speech_ms`.
    TOO_SHORT = "too_short"
    #: Capture stopped, the microphone was muted or settings changed mid-phrase.
    FLUSH = "flush"


@dataclass(frozen=True, slots=True)
class SegmenterSettings:
    """Thresholds of the state machine.

    Attributes:
        start_frames: Consecutive speech frames that declare an onset.
        silence_ms: Continuous silence that closes a phrase.
        min_speech_ms: Segments with less speech than this are rejected.
        max_utterance_ms: Hard cap on one segment.
        pre_roll_ms: Audio kept before the onset and prepended to the segment.
        tail_ms: Trailing silence kept when closing on silence.

    Raises:
        AudioError: If a value would make the machine unable to ever emit a
            segment - a minimum longer than the maximum, or a non-positive
            silence window.
    """

    start_frames: int = DEFAULT_START_FRAMES
    silence_ms: int = DEFAULT_SILENCE_MS
    min_speech_ms: int = DEFAULT_MIN_SPEECH_MS
    max_utterance_ms: int = DEFAULT_MAX_UTTERANCE_MS
    pre_roll_ms: int = DEFAULT_PRE_ROLL_MS
    tail_ms: int = DEFAULT_TAIL_MS

    def __post_init__(self) -> None:
        if self.start_frames < 1:
            raise AudioError(
                f"start_frames must be at least 1, got {self.start_frames}",
                user_message="Некорректные настройки определения речи.",
            )
        if self.silence_ms <= 0:
            raise AudioError(
                f"silence_ms must be positive, got {self.silence_ms}",
                user_message="Длительность паузы для конца фразы должна быть больше нуля.",
            )
        if self.min_speech_ms < 0 or self.pre_roll_ms < 0 or self.tail_ms < 0:
            raise AudioError(
                f"min_speech_ms/pre_roll_ms/tail_ms must not be negative, got "
                f"{self.min_speech_ms}/{self.pre_roll_ms}/{self.tail_ms}",
                user_message="Некорректные настройки определения речи.",
            )
        if self.max_utterance_ms <= self.min_speech_ms:
            raise AudioError(
                f"max_utterance_ms ({self.max_utterance_ms}) must exceed "
                f"min_speech_ms ({self.min_speech_ms})",
                user_message="Максимальная длина фразы меньше минимальной.",
            )


@dataclass(frozen=True, slots=True)
class SpeechStart:
    """Published the moment an onset is confirmed.

    Attributes:
        frame_index: Absolute VAD frame the segment begins at, pre-roll
            included.  Comparable with :attr:`ayris.audio.vad.VadFrame.index`.
        pre_roll_ms: How much audio ahead of the onset the segment will carry.
    """

    frame_index: int
    pre_roll_ms: int = 0


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """One finished phrase.

    Attributes:
        pcm: The audio, little-endian ``int16`` mono, pre-roll first.
        sample_rate: Rate of :attr:`pcm`.
        frame_index: Absolute VAD frame the segment starts at.
        pre_roll_ms: Audio in front of the confirmed onset.
        speech_ms: How much of the segment the detector called speech.  This,
            not :attr:`duration_ms`, is what the minimum length is checked
            against.
        reason: Why the segment was closed.
        accepted: Whether the segment is worth recognising.  A rejected one is
            still reported so the UI can drop back out of "listening", but it
            must not be sent to speech-to-text.
    """

    pcm: bytes
    sample_rate: int
    frame_index: int
    pre_roll_ms: int
    speech_ms: int
    reason: EndReason
    accepted: bool = True

    @property
    def frames(self) -> int:
        """Sample frames in :attr:`pcm`."""
        return len(self.pcm) // SAMPLE_WIDTH

    @property
    def duration_ms(self) -> int:
        """Length of the whole segment, pre-roll and tail included."""
        return self.frames * 1000 // self.sample_rate if self.sample_rate else 0

    def __bool__(self) -> bool:
        """True for a segment that should be recognised."""
        return self.accepted and bool(self.pcm)


@dataclass(frozen=True, slots=True)
class SegmenterStats:
    """Counters for the status panel and the calibration report."""

    state: SegmentState = SegmentState.IDLE
    frames: int = 0
    speech_frames: int = 0
    segments: int = 0
    rejected: int = 0
    truncated: int = 0
    current_ms: int = 0

    @property
    def speech_ratio(self) -> float:
        """Share of frames the detector called speech, 0.0-1.0."""
        return self.speech_frames / self.frames if self.frames else 0.0


def _ignore_start(_start: SpeechStart) -> None:
    """Default :attr:`SegmenterCallbacks.on_speech_started`."""


def _ignore_segment(_segment: SpeechSegment) -> None:
    """Default :attr:`SegmenterCallbacks.on_speech_ended`."""


def _ignore_frame(_frame: VadFrame) -> None:
    """Default :attr:`SegmenterCallbacks.on_frame`."""


@dataclass(frozen=True, slots=True)
class SegmenterCallbacks:
    """Where the segmenter reports to.

    All three run on the thread that called :meth:`Segmenter.push` - in
    production the audio worker's processing thread - so they must not block.
    Handing a segment to a recogniser means queueing it, not decoding it here.
    """

    #: An onset was confirmed.  Becomes ``SpeechStarted`` on the bus.
    on_speech_started: Callable[[SpeechStart], None] = _ignore_start
    #: A phrase finished, accepted or not.  Becomes ``SpeechEnded``.
    on_speech_ended: Callable[[SpeechSegment], None] = _ignore_segment
    #: Every classified frame, for the level meter and the debug view.
    on_frame: Callable[[VadFrame], None] = _ignore_frame


@dataclass(slots=True)
class _Collector:
    """Audio and counters of the segment being built."""

    frames: list[bytes] = field(default_factory=list)
    frame_index: int = 0
    pre_roll_frames: int = 0
    speech_frames: int = 0
    silence_frames: int = 0
    #: Speech frames seen in a row while idle, counting towards the onset.
    onset_run: int = 0

    def clear(self, frame_index: int) -> None:
        """Start over at ``frame_index`` with nothing collected."""
        self.frames = []
        self.frame_index = frame_index
        self.pre_roll_frames = 0
        self.speech_frames = 0
        self.silence_frames = 0
        self.onset_run = 0


class Segmenter:
    """Phrase boundaries from classified frames.

    Args:
        settings: Thresholds of the state machine.
        vad_settings: Detector configuration, used when ``stream`` is omitted.
        stream: Pre-built :class:`~ayris.audio.vad.VadStream`, for tests and for
            sharing one detector across a reconfiguration.
        callbacks: Where onsets, segments and frames are reported.

    Not thread-safe: one instance belongs to one audio pipeline, and in Ayris
    that is the audio worker's processing thread.
    """

    __slots__ = ("_callbacks", "_collector", "_pre_roll", "_settings", "_state", "_stats", "_vad")

    def __init__(
        self,
        settings: SegmenterSettings | None = None,
        vad_settings: VadSettings | None = None,
        *,
        stream: VadStream | None = None,
        callbacks: SegmenterCallbacks | None = None,
    ) -> None:
        self._settings = settings or SegmenterSettings()
        self._vad = stream if stream is not None else VadStream(vad_settings or VadSettings())
        self._callbacks = callbacks or SegmenterCallbacks()
        self._pre_roll: deque[bytes] = deque(maxlen=self._pre_roll_capacity())
        self._collector = _Collector()
        self._state = SegmentState.IDLE
        self._stats = SegmenterStats()

    # -------------------------------------------------------------- read-only

    @property
    def settings(self) -> SegmenterSettings:
        """Thresholds in force."""
        return self._settings

    @property
    def vad(self) -> VadStream:
        """The detector feeding the state machine."""
        return self._vad

    @property
    def state(self) -> SegmentState:
        """Whether a phrase is being collected right now."""
        return self._state

    @property
    def is_speech(self) -> bool:
        """Whether the assistant currently believes somebody is talking.

        This is the flag the overlay animates on, so it follows the state
        machine rather than the raw frame verdict - one dropped frame in the
        middle of a word must not make the sphere flicker.
        """
        return self._state is SegmentState.SPEECH

    @property
    def stats(self) -> SegmenterStats:
        """Counters since the last :meth:`reset`, including the live segment."""
        collector = self._collector
        current = (
            len(collector.frames) * self._frame_ms if self._state is SegmentState.SPEECH else 0
        )
        stats = self._stats
        return SegmenterStats(
            state=self._state,
            frames=stats.frames,
            speech_frames=stats.speech_frames,
            segments=stats.segments,
            rejected=stats.rejected,
            truncated=stats.truncated,
            current_ms=current,
        )

    # ------------------------------------------------------------------ input

    def push(self, pcm: bytes | bytearray | memoryview) -> tuple[SpeechSegment, ...]:
        """Feed captured audio in and collect whatever phrases it completed.

        Args:
            pcm: Little-endian ``int16`` mono at the detector's sample rate.
                Any length: framing is handled inside.

        Returns:
            Segments closed by this block, in order.  Usually empty - a phrase
            takes many blocks - and occasionally holds two when a maximum-length
            cut is followed by the real end.
        """
        segments: list[SpeechSegment] = []
        for frame in self._vad.push(pcm):
            self._callbacks.on_frame(frame)
            segments.extend(self._consume(frame))
        return tuple(segments)

    def flush(self) -> tuple[SpeechSegment, ...]:
        """Close an in-progress phrase.

        Called when capture stops, the microphone is muted or the settings
        change: whatever was collected is better handed over than dropped, but
        it is marked ``reason="flush"`` so a consumer can treat it as partial.

        Returns:
            Everything the padded remainder and the forced close produced -
            empty when nothing was being collected.
        """
        segments: list[SpeechSegment] = []
        for frame in self._vad.flush():
            self._callbacks.on_frame(frame)
            segments.extend(self._consume(frame))
        forced = self._force_close(EndReason.FLUSH)
        if forced is not None:
            segments.append(forced)
        return tuple(segments)

    def reset(self) -> None:
        """Drop everything: the phrase in progress, the pre-roll and the stats."""
        self._vad.reset()
        self._pre_roll.clear()
        self._collector.clear(0)
        self._state = SegmentState.IDLE
        self._stats = SegmenterStats()

    def configure(
        self,
        settings: SegmenterSettings | None = None,
        vad_settings: VadSettings | None = None,
    ) -> None:
        """Apply new thresholds, closing any phrase in progress.

        A phrase collected under the old thresholds cannot be finished under the
        new ones without ambiguity, so it is flushed first - which also means
        the caller gets the audio instead of losing it.
        """
        if self._state is SegmentState.SPEECH:
            self._force_close(EndReason.FLUSH)
        if settings is not None:
            self._settings = settings
            self._pre_roll = deque(self._pre_roll, maxlen=self._pre_roll_capacity())
        if vad_settings is not None:
            self._vad.configure(vad_settings)
            self._pre_roll = deque(maxlen=self._pre_roll_capacity())

    # --------------------------------------------------------------- internal

    @property
    def _frame_ms(self) -> int:
        """Frame length of the detector."""
        return self._vad.settings.frame_ms

    def _pre_roll_capacity(self) -> int:
        """Frames of pre-roll to keep, at least one."""
        return max(1, -(-self._settings.pre_roll_ms // max(1, self._frame_ms)))

    def _consume(self, frame: VadFrame) -> list[SpeechSegment]:
        """Advance the state machine by one frame."""
        self._stats = _bump(self._stats, frames=1, speech_frames=int(frame.speech))
        if self._state is SegmentState.IDLE:
            return self._while_idle(frame)
        return self._while_speaking(frame)

    def _while_idle(self, frame: VadFrame) -> list[SpeechSegment]:
        """Watch for an onset, keeping the pre-roll fed."""
        self._pre_roll.append(frame.pcm)
        if not frame.speech:
            self._collector.onset_run = 0
            return []
        run = self._collector.onset_run + 1
        self._collector.onset_run = run
        if run < self._settings.start_frames:
            return []
        self._begin(frame)
        return []

    def _begin(self, frame: VadFrame) -> None:
        """Open a segment from the pre-roll ring."""
        collected = list(self._pre_roll)
        # The onset frames are already in the pre-roll; everything before them
        # is the lead-in the speaker's first consonant lives in.
        pre_roll_frames = max(0, len(collected) - self._settings.start_frames)
        self._collector.frames = collected
        self._collector.frame_index = frame.index - len(collected) + 1
        self._collector.pre_roll_frames = pre_roll_frames
        self._collector.speech_frames = self._settings.start_frames
        self._collector.silence_frames = 0
        self._collector.onset_run = 0
        self._pre_roll.clear()
        self._state = SegmentState.SPEECH
        self._callbacks.on_speech_started(
            SpeechStart(
                frame_index=self._collector.frame_index,
                pre_roll_ms=pre_roll_frames * self._frame_ms,
            )
        )

    def _while_speaking(self, frame: VadFrame) -> list[SpeechSegment]:
        """Collect the phrase and watch for its end."""
        collector = self._collector
        collector.frames.append(frame.pcm)
        if frame.speech:
            collector.speech_frames += 1
            collector.silence_frames = 0
        else:
            collector.silence_frames += 1

        if collector.silence_frames * self._frame_ms >= self._settings.silence_ms:
            segment = self._close(EndReason.SILENCE)
            if segment is None:
                return []
            self._callbacks.on_speech_ended(segment)
            return [segment]

        if len(collector.frames) * self._frame_ms >= self._settings.max_utterance_ms:
            segment = self._cut()
            self._callbacks.on_speech_ended(segment)
            return [segment]
        return []

    def _force_close(self, reason: EndReason) -> SpeechSegment | None:
        """Close a phrase from outside the frame loop, if there is one.

        Returns ``None`` when nothing is being collected - see :meth:`_close`.
        """
        if self._state is not SegmentState.SPEECH:
            return None
        segment = self._close(reason)
        if segment is None:
            return None
        self._callbacks.on_speech_ended(segment)
        return segment

    def _cut(self) -> SpeechSegment:
        """Hand over an over-long segment and keep collecting.

        The state stays :attr:`SegmentState.SPEECH` on purpose: the speaker has
        not stopped, and going back to idle would swallow the next
        ``start_frames`` frames while the onset is re-confirmed.
        """
        collector = self._collector
        segment = self._build(collector.frames, EndReason.MAX_LENGTH, collector.speech_frames)
        next_index = collector.frame_index + len(collector.frames)
        silence = collector.silence_frames
        collector.clear(next_index)
        # A silence run that was already under way must survive the seam,
        # otherwise a phrase ending right at the cap would never close.
        collector.silence_frames = silence
        self._stats = _bump(self._stats, segments=1, truncated=1)
        _log.debug("phrase reached %d ms and was cut", segment.duration_ms)
        return segment

    def _close(self, reason: EndReason) -> SpeechSegment | None:
        """Finish the current segment and go back to idle.

        Returns ``None`` when there is nothing left to hand over.  That happens
        after a maximum-length cut: the cut leaves the state at
        :attr:`SegmentState.SPEECH` with an empty collector and the silence run
        carried across the seam, so the close that follows can be holding
        nothing but that silence.  An empty segment there would be a second
        ``SpeechEnded`` for audio nobody spoke, and a rejection in the stats
        that never happened.
        """
        collector = self._collector
        frames = collector.frames
        if reason is EndReason.SILENCE:
            frames = self._trim_tail(frames, collector.silence_frames)
        consumed = len(collector.frames)
        segment = self._build(frames, reason, collector.speech_frames) if frames else None
        self._state = SegmentState.IDLE
        self._pre_roll.clear()
        collector.clear(collector.frame_index + consumed)
        if segment is None:
            return None
        self._stats = _bump(
            self._stats,
            segments=int(segment.accepted),
            rejected=int(not segment.accepted),
        )
        return segment

    def _trim_tail(self, frames: list[bytes], silence_frames: int) -> list[bytes]:
        """Drop the silence that closed the phrase, keeping ``tail_ms`` of it.

        ``silence_frames`` can exceed what is collected - it survives a
        maximum-length cut - so the drop is clamped, otherwise the negative
        slice bound would quietly cut from the wrong end.
        """
        keep = max(0, -(-self._settings.tail_ms // max(1, self._frame_ms)))
        drop = min(len(frames), max(0, silence_frames - keep))
        return frames[: len(frames) - drop] if drop else frames

    def _build(self, frames: list[bytes], reason: EndReason, speech_frames: int) -> SpeechSegment:
        """Assemble the segment and decide whether it is worth keeping."""
        speech_ms = speech_frames * self._frame_ms
        accepted = speech_ms >= self._settings.min_speech_ms
        return SpeechSegment(
            pcm=b"".join(frames),
            sample_rate=self._vad.settings.sample_rate,
            frame_index=self._collector.frame_index,
            pre_roll_ms=self._collector.pre_roll_frames * self._frame_ms,
            speech_ms=speech_ms,
            reason=reason if accepted else EndReason.TOO_SHORT,
            accepted=accepted,
        )

    def __repr__(self) -> str:
        return f"Segmenter(state={self._state.value}, engine={self._vad.engine})"


def _bump(
    stats: SegmenterStats,
    *,
    frames: int = 0,
    speech_frames: int = 0,
    segments: int = 0,
    rejected: int = 0,
    truncated: int = 0,
) -> SegmenterStats:
    """Add to the counters of an immutable snapshot."""
    return SegmenterStats(
        state=stats.state,
        frames=stats.frames + frames,
        speech_frames=stats.speech_frames + speech_frames,
        segments=stats.segments + segments,
        rejected=stats.rejected + rejected,
        truncated=stats.truncated + truncated,
        current_ms=stats.current_ms,
    )


def segment_pcm(
    pcm: bytes | bytearray | memoryview,
    *,
    settings: SegmenterSettings | None = None,
    vad_settings: VadSettings | None = None,
    vad: Vad | None = None,
) -> tuple[SpeechSegment, ...]:
    """Run a finished recording through the segmenter in one call.

    The offline counterpart of :meth:`Segmenter.push`, used by calibration to
    check a test phrase and by the tests to work on WAV fixtures.  The trailing
    phrase is flushed, so a recording that ends mid-word still produces a
    segment.

    Args:
        pcm: The whole recording, mono ``int16``.
        settings: Segmenter thresholds.
        vad_settings: Detector configuration.
        vad: Pre-built detector, when the caller wants a specific engine.

    Returns:
        Every segment found, rejected ones included - the caller decides what
        "too short" means for its purpose.
    """
    resolved = vad_settings or (vad.settings if vad is not None else VadSettings())
    segmenter = Segmenter(settings, stream=VadStream(resolved, vad=vad))
    found = list(segmenter.push(pcm))
    found.extend(segmenter.flush())
    return tuple(found)
