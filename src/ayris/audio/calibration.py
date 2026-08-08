"""Measuring the room and proposing settings that suit it.

The default thresholds in ``config.toml`` are a guess about a room nobody has
heard.  In a quiet flat with a headset they are too cautious; next to a desktop
fan they cut speech off mid-word.  Calibration replaces the guess with a
measurement, and it is the answer to the most common support question there is -
"почему Ayris меня не слышит".

Two steps, in the order the settings window walks the user through them:

**Silence.**  Three seconds of the room saying nothing.  That gives the noise
floor, and the floor is what every other threshold hangs off: the VAD gate sits
a margin above it, and so does the noise gate.

**A phrase.**  The user says "айрис открой браузер" and the recording goes
through the very same :class:`~ayris.audio.segmenter.Segmenter` that will run in
production.  This is the part that makes the report trustworthy: it does not
predict that the settings will work, it demonstrates that they did, on this
microphone, in this room.  If the phrase comes back in two pieces or clipped at
the start, the recommendation is adjusted until it comes back whole.

Everything here is pure analysis over PCM.  Recording is somebody else's job -
:class:`AudioSource` is the whole interface to it - which is what lets the tests
calibrate against WAV fixtures and keeps the microphone out of CI.
"""

from __future__ import annotations

import logging
import math
from array import array
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol

from ayris.audio.capture import MIN_DBFS, TARGET_SAMPLE_RATE, apply_gain, pcm_level
from ayris.audio.denoise import DenoiseMode
from ayris.audio.ring_buffer import SAMPLE_WIDTH
from ayris.audio.segmenter import Segmenter, SegmenterSettings
from ayris.audio.vad import MAX_MARGIN_DB, MIN_MARGIN_DB, VadSettings
from ayris.core.errors import AudioError

if TYPE_CHECKING:
    from ayris.audio.segmenter import SpeechSegment
    from ayris.core.models import JsonObject

__all__ = [
    "CALIBRATION_PHRASE",
    "DEFAULT_PHRASE_SEC",
    "DEFAULT_SILENCE_SEC",
    "AudioSource",
    "CalibrationReport",
    "NoiseProfile",
    "PhraseCheck",
    "Recommendation",
    "Verdict",
    "analyse_noise",
    "analyse_phrase",
    "build_report",
    "calibrate_pcm",
    "recommend",
    "run_calibration",
]

_log = logging.getLogger(__name__)

#: What the user is asked to say.  The wake word plus a short command, because
#: the thing worth proving is that the *beginning* survives - a phrase that
#: starts with a vowel would hide exactly the pre-roll bug we are checking for.
CALIBRATION_PHRASE: Final = "айрис открой браузер"

#: Silence recorded to measure the floor.  Long enough to catch a fan cycling,
#: short enough that nobody gets bored holding still.
DEFAULT_SILENCE_SEC: Final = 3.0

#: Time given to the test phrase.
DEFAULT_PHRASE_SEC: Final = 5.0

#: Where speech should land after gain.  Recognition models are trained on audio
#: around this level; louder invites clipping, quieter loses consonants.
TARGET_SPEECH_DBFS: Final = -23.0

#: Peak the loudest syllable must stay under, leaving room for a shout.
TARGET_PEAK_DBFS: Final = -6.0

#: Bounds on the recommendation.  Software gain is not an amplifier: past ~8x it
#: raises the hiss along with the voice, and below 0.5x the problem is the
#: system mixer, not Ayris.
MIN_GAIN: Final = 0.5
MAX_GAIN: Final = 8.0

#: Noise floor at or below this is a quiet room.
QUIET_FLOOR_DB: Final = -55.0

#: Above this the room is loud enough that denoising should be on and the user
#: should be told why.
NOISY_FLOOR_DB: Final = -40.0

#: Speech this quiet will not survive recognition even with gain.
WEAK_SPEECH_DB: Final = -45.0

#: Share of clipped samples that makes a recording unusable.
CLIPPING_RATIO: Final = 0.002

#: Where the gate is placed between the floor and the speech level.  Low enough
#: to catch a trailing consonant, high enough to ignore the room.
_GATE_SHARE: Final = 0.35

#: Fallback margin when there is no phrase to measure against.
_DEFAULT_MARGIN_DB: Final = 9.0

#: Extra silence allowed when a phrase came back in pieces.
_SILENCE_STEP_MS: Final = 300
_MAX_SILENCE_MS: Final = 2000


class Verdict(StrEnum):
    """The one-word answer, which decides the colour of the banner."""

    #: Everything measured well; the recommendation is a refinement.
    GOOD = "good"
    #: Usable, but the room is loud enough to need denoising.
    NOISY = "noisy"
    #: Speech is far too quiet even after the proposed gain.
    QUIET = "quiet"
    #: The recording is clipped; no threshold can fix that.
    CLIPPING = "clipping"
    #: Nothing that looks like speech was found.
    NO_SPEECH = "no_speech"


@dataclass(frozen=True, slots=True)
class NoiseProfile:
    """What the silent recording contained.

    Attributes:
        rms_db: Average level, the number quoted to the user as "уровень шума".
        peak_db: Loudest sample.  A peak far above the average means something
            knocked or clicked, and the average understates the room.
        clipped_ratio: Share of samples at full scale.  Non-zero during silence
            means the input gain in Windows is set absurdly high.
        duration_ms: How much was actually analysed.
    """

    rms_db: float = MIN_DBFS
    peak_db: float = MIN_DBFS
    clipped_ratio: float = 0.0
    duration_ms: int = 0

    @property
    def quiet(self) -> bool:
        """Whether this is a quiet room by :data:`QUIET_FLOOR_DB`."""
        return self.rms_db <= QUIET_FLOOR_DB

    @property
    def noisy(self) -> bool:
        """Whether the room is loud enough to warrant denoising."""
        return self.rms_db >= NOISY_FLOOR_DB


@dataclass(frozen=True, slots=True)
class PhraseCheck:
    """What the segmenter made of the test phrase.

    Attributes:
        detected: Whether any segment was accepted at all.
        whole: Whether the phrase arrived as exactly one segment.  Two segments
            mean the silence threshold is too short for this speaker; zero means
            the gate is too high.
        segments: How many accepted segments came out.
        speech_ms: Speech in the longest segment.
        duration_ms: Length of the longest segment, pre-roll included.
        lead_ms: Audio kept in front of the onset.  Zero here is the bug that
            eats the first syllable of the wake word.
        rms_db: Level of the speech itself, measured over the longest segment.
        peak_db: Loudest sample of that segment.
        clipped_ratio: Share of clipped samples in it.
    """

    detected: bool = False
    whole: bool = False
    segments: int = 0
    speech_ms: int = 0
    duration_ms: int = 0
    lead_ms: int = 0
    rms_db: float = MIN_DBFS
    peak_db: float = MIN_DBFS
    clipped_ratio: float = 0.0

    @property
    def clipping(self) -> bool:
        """Whether the phrase is clipped beyond use."""
        return self.clipped_ratio >= CLIPPING_RATIO


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Values to write into ``voice.audio_input``.

    Attributes:
        gain: Software gain that puts speech at :data:`TARGET_SPEECH_DBFS`.
        vad_threshold: ``0.0-1.0``, converted from the margin the measurement
            asks for - see :attr:`ayris.audio.vad.VadSettings.margin_db`.
        noise_floor_db: The measured floor, which anchors every gate.
        silence_ms: End-of-phrase pause, raised when the test phrase split.
        denoise: Suppression mode suited to the room.
        gate_db: Absolute cut-off the recommendation works out to.  Not a
            setting - it is what the level meter draws its line at, and what
            makes the other numbers make sense to somebody looking at the meter.
    """

    gain: float = 1.0
    vad_threshold: float = 0.5
    noise_floor_db: float = -45.0
    silence_ms: int = 700
    denoise: DenoiseMode = DenoiseMode.RNNOISE
    gate_db: float = MIN_DBFS

    def as_dict(self) -> JsonObject:
        """Flatten for the settings window and the worker pipe."""
        return {
            "gain": round(self.gain, 2),
            "vad_threshold": round(self.vad_threshold, 2),
            "noise_floor_db": round(self.noise_floor_db, 1),
            "silence_ms": self.silence_ms,
            "denoise": self.denoise.value,
            "gate_db": round(self.gate_db, 1),
        }

    def vad_settings(self, base: VadSettings | None = None) -> VadSettings:
        """Apply the recommendation to a detector configuration."""
        source = base or VadSettings()
        return VadSettings(
            sample_rate=source.sample_rate,
            frame_ms=source.frame_ms,
            aggressiveness=source.aggressiveness,
            threshold=self.vad_threshold,
            noise_floor_db=self.noise_floor_db,
            engine=source.engine,
        )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Everything the settings window shows after calibration.

    Attributes:
        verdict: The headline.
        noise: Measurements from the silent recording.
        phrase: Measurements from the test phrase, all zeros when it was
            skipped.
        recommended: Proposed settings.
        messages: Russian sentences explaining the verdict, in the order they
            should be shown.  Ready for display - the UI adds no wording of its
            own, so that the same text can be logged and pasted into a bug
            report.
        phrase_checked: Whether a phrase was recorded at all.
    """

    verdict: Verdict = Verdict.GOOD
    noise: NoiseProfile = field(default_factory=NoiseProfile)
    phrase: PhraseCheck = field(default_factory=PhraseCheck)
    recommended: Recommendation = field(default_factory=Recommendation)
    messages: tuple[str, ...] = ()
    phrase_checked: bool = False

    @property
    def ok(self) -> bool:
        """Whether the microphone can be used as it is."""
        return self.verdict in (Verdict.GOOD, Verdict.NOISY)

    @property
    def summary(self) -> str:
        """One line for the banner."""
        return _SUMMARY[self.verdict]

    def as_dict(self) -> JsonObject:
        """Flatten for the worker pipe and DevTools."""
        return {
            "verdict": self.verdict.value,
            "ok": self.ok,
            "summary": self.summary,
            "messages": list(self.messages),
            "noise": {
                "rms_db": round(self.noise.rms_db, 1),
                "peak_db": round(self.noise.peak_db, 1),
                "clipped_ratio": round(self.noise.clipped_ratio, 4),
                "duration_ms": self.noise.duration_ms,
            },
            "phrase": {
                "checked": self.phrase_checked,
                "detected": self.phrase.detected,
                "whole": self.phrase.whole,
                "segments": self.phrase.segments,
                "speech_ms": self.phrase.speech_ms,
                "duration_ms": self.phrase.duration_ms,
                "lead_ms": self.phrase.lead_ms,
                "rms_db": round(self.phrase.rms_db, 1),
                "peak_db": round(self.phrase.peak_db, 1),
            },
            "recommended": self.recommended.as_dict(),
        }


#: Banner text per verdict.
_SUMMARY: Final[dict[Verdict, str]] = {
    Verdict.GOOD: "Микрофон настроен, речь распознаётся уверенно.",
    Verdict.NOISY: "Микрофон работает, но в помещении шумно.",
    Verdict.QUIET: "Голос слишком тихий — Ayris будет часто вас не слышать.",
    Verdict.CLIPPING: "Запись перегружена: звук искажён.",
    Verdict.NO_SPEECH: "Речь не распознана.",
}


class AudioSource(Protocol):
    """Where calibration gets its audio.

    Deliberately tiny.  Production passes an adapter over
    :class:`~ayris.audio.capture.AudioCapture`; tests pass an object that reads
    a WAV file; neither this module nor its tests need to know the difference.
    """

    @property
    def sample_rate(self) -> int:
        """Rate of everything :meth:`record` returns."""
        ...

    def record(self, seconds: float) -> bytes:
        """Block for ``seconds`` and return what was heard, ``int16`` mono."""
        ...


def analyse_noise(pcm: bytes | bytearray | memoryview, sample_rate: int) -> NoiseProfile:
    """Measure a recording of an empty room.

    Args:
        pcm: The silent recording, ``int16`` mono.
        sample_rate: Its rate, needed only to report the duration.

    Returns:
        The profile.  An empty recording gives :data:`~ayris.audio.capture.MIN_DBFS`,
        which reads downstream as "impossibly quiet" and produces the
        :attr:`Verdict.NO_SPEECH` path rather than a division by zero.
    """
    data = bytes(pcm)
    level = pcm_level(data)
    frames = len(data) // SAMPLE_WIDTH
    return NoiseProfile(
        rms_db=level.rms_db,
        peak_db=level.peak_db,
        clipped_ratio=_clipped_ratio(data),
        duration_ms=frames * 1000 // sample_rate if sample_rate else 0,
    )


def _clipped_ratio(pcm: bytes) -> float:
    """Share of samples sitting at the rail.

    :func:`~ayris.audio.capture.pcm_level` only answers "did anything clip",
    which is the right question for a level meter and the wrong one here: a
    single clipped sample in three seconds is a door closing, while one in five
    hundred is a gain problem the user has to fix in Windows.
    """
    samples = array("h")
    samples.frombytes(pcm)
    if not samples:
        return 0.0
    clipped = sum(1 for sample in samples if sample >= 32767 or sample <= -32768)
    return clipped / len(samples)


def analyse_phrase(
    pcm: bytes | bytearray | memoryview,
    *,
    vad_settings: VadSettings,
    segmenter_settings: SegmenterSettings | None = None,
) -> PhraseCheck:
    """Run the test phrase through the production segmenter.

    Args:
        pcm: The recording, ``int16`` mono at ``vad_settings.sample_rate``.
        vad_settings: Detector configuration to test, normally the one just
            derived from the noise floor.
        segmenter_settings: Thresholds to test with.

    Returns:
        What came out.  The *longest* accepted segment is measured, because a
        user who coughed before speaking should not be told their phrase was
        split - the count in :attr:`PhraseCheck.segments` says that instead.
    """
    segmenter = Segmenter(segmenter_settings, vad_settings)
    accepted = [segment for segment in _run(segmenter, pcm) if segment.accepted]
    if not accepted:
        return PhraseCheck()
    best = max(accepted, key=lambda segment: segment.speech_ms)
    level = pcm_level(best.pcm)
    return PhraseCheck(
        detected=True,
        whole=len(accepted) == 1,
        segments=len(accepted),
        speech_ms=best.speech_ms,
        duration_ms=best.duration_ms,
        lead_ms=best.pre_roll_ms,
        rms_db=level.rms_db,
        peak_db=level.peak_db,
        clipped_ratio=_clipped_ratio(best.pcm),
    )


def _run(segmenter: Segmenter, pcm: bytes | bytearray | memoryview) -> list[SpeechSegment]:
    """Push a whole recording through and collect everything, tail included."""
    found = list(segmenter.push(pcm))
    found.extend(segmenter.flush())
    return found


def recommend(
    noise: NoiseProfile,
    phrase: PhraseCheck,
    *,
    base_gain: float = 1.0,
    base_silence_ms: int = 700,
    base_denoise: DenoiseMode = DenoiseMode.RNNOISE,
) -> Recommendation:
    """Turn measurements into settings.

    Args:
        noise: The silent recording.
        phrase: The test phrase; an empty one means only the floor is known.
        base_gain: Gain in force, kept when there is no phrase to judge by.
        base_silence_ms: End-of-phrase pause in force.
        base_denoise: Suppression mode in force, kept unless the room asks for
            more.

    Returns:
        The recommendation, every value clamped to what the settings window will
        accept.
    """
    floor = min(noise.rms_db, -20.0)
    gain = _recommend_gain(phrase, base_gain)
    margin = _recommend_margin(floor, phrase, gain)
    threshold = (margin - MIN_MARGIN_DB) / (MAX_MARGIN_DB - MIN_MARGIN_DB)
    silence_ms = base_silence_ms
    if phrase.detected and not phrase.whole:
        # The phrase came back in pieces: the speaker pauses longer than the
        # threshold allows.  Give them room rather than blaming the microphone.
        silence_ms = min(_MAX_SILENCE_MS, base_silence_ms + _SILENCE_STEP_MS)
    denoise = base_denoise
    if noise.noisy and denoise is DenoiseMode.OFF:
        denoise = DenoiseMode.SPECTRAL
    return Recommendation(
        gain=gain,
        vad_threshold=min(1.0, max(0.0, round(threshold, 2))),
        noise_floor_db=round(floor, 1),
        silence_ms=silence_ms,
        denoise=denoise,
        gate_db=round(floor + margin, 1),
    )


def _recommend_gain(phrase: PhraseCheck, base_gain: float) -> float:
    """Put speech at :data:`TARGET_SPEECH_DBFS` without clipping its peaks.

    Both constraints are applied and the smaller wins: raising the average to
    target is pointless if it drives the loudest syllable into the ceiling.
    """
    if not phrase.detected or phrase.rms_db <= MIN_DBFS:
        return round(base_gain, 2)
    by_rms = float(10.0 ** ((TARGET_SPEECH_DBFS - phrase.rms_db) / 20.0))
    by_peak = float(10.0 ** ((TARGET_PEAK_DBFS - phrase.peak_db) / 20.0))
    wanted = base_gain * min(by_rms, by_peak)
    return round(min(MAX_GAIN, max(MIN_GAIN, wanted)), 2)


def _recommend_margin(floor_db: float, phrase: PhraseCheck, gain: float) -> float:
    """Place the gate between the floor and the speech, in dB above the floor.

    Without a phrase there is nothing to place it against, so the default
    margin is kept: guessing from the floor alone puts the gate on top of a
    quiet speaker in a quiet room.
    """
    if not phrase.detected or phrase.rms_db <= MIN_DBFS:
        return _DEFAULT_MARGIN_DB
    speech_db = phrase.rms_db + 20.0 * math.log10(max(gain, 1e-6))
    span = speech_db - floor_db
    if span <= 0.0:
        return MIN_MARGIN_DB
    return min(MAX_MARGIN_DB, max(MIN_MARGIN_DB, span * _GATE_SHARE))


def build_report(
    noise: NoiseProfile,
    phrase: PhraseCheck,
    recommended: Recommendation,
    *,
    phrase_checked: bool,
) -> CalibrationReport:
    """Assemble the verdict and the wording around it.

    The order of the checks is the order of severity: a clipped recording makes
    every other measurement meaningless, so it is reported first and alone.
    """
    messages: list[str] = [
        f"Уровень шума: {noise.rms_db:.0f} дБ, пик {noise.peak_db:.0f} дБ.",
    ]
    verdict = Verdict.GOOD

    if noise.clipped_ratio >= CLIPPING_RATIO or phrase.clipping:
        verdict = Verdict.CLIPPING
        messages.append(
            "Запись достигает максимума громкости. Уменьшите усиление микрофона "
            "в параметрах звука Windows."
        )
    elif phrase_checked and not phrase.detected:
        verdict = Verdict.NO_SPEECH
        messages.append(
            f"Фраза «{CALIBRATION_PHRASE}» не распознана как речь. "
            "Проверьте, что говорите в нужный микрофон и он не выключен."
        )
    elif phrase_checked and phrase.rms_db <= WEAK_SPEECH_DB:
        verdict = Verdict.QUIET
        messages.append(
            f"Речь записана на уровне {phrase.rms_db:.0f} дБ — это очень тихо. "
            "Придвиньте микрофон ближе или увеличьте его громкость в Windows."
        )
    elif noise.noisy:
        verdict = Verdict.NOISY
        messages.append(
            "Фоновый шум высокий. Включено шумоподавление, но в тишине "
            "распознавание будет заметно точнее."
        )
    elif noise.quiet:
        messages.append("Помещение тихое — распознавание должно работать уверенно.")

    if phrase_checked and phrase.detected:
        messages.append(
            f"Фраза записана целиком: {phrase.duration_ms} мс, из них речи "
            f"{phrase.speech_ms} мс, запас перед началом {phrase.lead_ms} мс."
            if phrase.whole
            else f"Фраза разделилась на {phrase.segments} части — пауза для конца "
            f"фразы увеличена до {recommended.silence_ms} мс."
        )
    elif not phrase_checked:
        messages.append("Тестовая фраза не записывалась, порог рассчитан только по шуму.")

    messages.append(
        f"Рекомендуемые значения: усиление {recommended.gain:.2f}, "
        f"порог VAD {recommended.vad_threshold:.2f}, "
        f"порог срабатывания {recommended.gate_db:.0f} дБ."
    )
    return CalibrationReport(
        verdict=verdict,
        noise=noise,
        phrase=phrase,
        recommended=recommended,
        messages=tuple(messages),
        phrase_checked=phrase_checked,
    )


def calibrate_pcm(
    silence: bytes | bytearray | memoryview,
    phrase: bytes | bytearray | memoryview | None = None,
    *,
    sample_rate: int = TARGET_SAMPLE_RATE,
    vad_settings: VadSettings | None = None,
    segmenter_settings: SegmenterSettings | None = None,
    base_gain: float = 1.0,
    base_denoise: DenoiseMode = DenoiseMode.RNNOISE,
) -> CalibrationReport:
    """Calibrate from two recordings that already exist.

    The core of the procedure and the only part the tests need.  The phrase is
    analysed with the settings derived from the *measured* floor rather than the
    ones currently in force, so the report answers "will the new values work",
    which is the question the user is about to be asked to agree to.

    Args:
        silence: Recording of the empty room.
        phrase: Recording of :data:`CALIBRATION_PHRASE`; may be omitted.
        sample_rate: Rate of both recordings.
        vad_settings: Detector configuration to start from.
        segmenter_settings: Segmenter thresholds to start from.
        base_gain: Gain currently applied by capture.
        base_denoise: Suppression mode currently selected.

    Returns:
        The report.

    Raises:
        AudioError: When the silent recording is empty - there is nothing to
            measure, and silently reporting a floor of -100 dB would send the
            user chasing a threshold that was never computed.
    """
    if not bytes(silence):
        raise AudioError(
            "calibration needs a recording of silence",
            user_message="Не удалось записать тишину для калибровки.",
        )
    base_vad = vad_settings or VadSettings(sample_rate=sample_rate)
    base_segmenter = segmenter_settings or SegmenterSettings()
    noise = analyse_noise(silence, sample_rate)

    # Two passes.  The first derives thresholds from the floor alone; the second
    # re-checks the phrase under the gain the first pass proposes, because a
    # phrase that was too quiet to detect at gain 1.0 may well be found at 2.5.
    draft = recommend(
        noise,
        PhraseCheck(),
        base_gain=base_gain,
        base_silence_ms=base_segmenter.silence_ms,
        base_denoise=base_denoise,
    )
    if phrase is None:
        return build_report(noise, PhraseCheck(), draft, phrase_checked=False)

    amplified = apply_gain(phrase, draft.gain) if draft.gain != 1.0 else bytes(phrase)
    check = analyse_phrase(
        amplified,
        vad_settings=draft.vad_settings(base_vad),
        segmenter_settings=base_segmenter,
    )
    final = recommend(
        noise,
        check,
        base_gain=draft.gain,
        base_silence_ms=base_segmenter.silence_ms,
        base_denoise=base_denoise,
    )
    return build_report(noise, check, final, phrase_checked=True)


def run_calibration(
    source: AudioSource,
    *,
    silence_sec: float = DEFAULT_SILENCE_SEC,
    phrase_sec: float | None = DEFAULT_PHRASE_SEC,
    vad_settings: VadSettings | None = None,
    segmenter_settings: SegmenterSettings | None = None,
    base_gain: float = 1.0,
    base_denoise: DenoiseMode = DenoiseMode.RNNOISE,
) -> CalibrationReport:
    """Record and calibrate.

    Blocks for ``silence_sec + phrase_sec``: the settings window runs this on a
    worker, not on the GUI thread.  The prompts ("помолчите", "скажите фразу")
    belong to the caller - this function has no way to draw them and no business
    guessing when the user is ready.

    Args:
        source: Where to record from.
        silence_sec: Length of the silent recording.
        phrase_sec: Length of the phrase recording; ``None`` skips it.
        vad_settings: Detector configuration to start from.
        segmenter_settings: Segmenter thresholds to start from.
        base_gain: Gain currently applied by capture.
        base_denoise: Suppression mode currently selected.

    Returns:
        The report.
    """
    rate = source.sample_rate
    _log.info("калибровка: запись тишины, %.1f с", silence_sec)
    quiet = source.record(silence_sec)
    spoken: bytes | None = None
    if phrase_sec is not None:
        _log.info("калибровка: запись фразы «%s», %.1f с", CALIBRATION_PHRASE, phrase_sec)
        spoken = source.record(phrase_sec)
    return calibrate_pcm(
        quiet,
        spoken,
        sample_rate=rate,
        vad_settings=vad_settings,
        segmenter_settings=segmenter_settings,
        base_gain=base_gain,
        base_denoise=base_denoise,
    )
