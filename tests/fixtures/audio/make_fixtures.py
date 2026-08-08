"""Generates the WAV fixtures the task 08 tests run on.

Committed alongside the files it produces, so that anybody can regenerate them
and see exactly what they contain.  Run it from the project root::

    _tools/python/bin/python3 tests/fixtures/audio/make_fixtures.py

Why synthesise instead of recording.  A real recording of "айрис открой браузер"
would be a voice - somebody's, identifiable, in a public repository - and it
would be a megabyte per take.  These are a few hundred kilobytes in total, they
are byte-identical on every machine, and they exercise the property the tests
actually assert on: WebRTC's classifier looks for a periodic glottal excitation
shaped by formants, and that is what :func:`voiced` produces.  Verified against
``webrtcvad`` at every aggressiveness level - the synthetic phrase is called
speech in 50 frames out of 50, the dither in 0 out of 50.

The synthesis is a source-filter model, which is the standard way to make a
vowel: an impulse train at the fundamental (120 Hz, with a little vibrato and
dither so the spectrum is not a comb of perfect harmonics) is passed through
three two-pole resonators at 700, 1220 and 2600 Hz - roughly the formants of a
Russian "а".  The result sounds like a kazoo and classifies like speech.
"""

from __future__ import annotations

import math
import random
import sys
import wave
from array import array
from pathlib import Path
from typing import Final

#: Everything downstream of capture works at this rate.
SAMPLE_RATE: Final = 16000

#: Fixed so that a regenerated fixture is byte-identical to the committed one.
SEED: Final = 20250808

#: Formants of a Russian "а": centre frequency and bandwidth, in Hz.
_FORMANTS: Final = ((700.0, 80.0), (1220.0, 90.0), (2600.0, 120.0))

#: Fundamental of the synthetic voice.
_F0: Final = 120.0

#: Vibrato, without which the excitation is too regular to be plausible.
_VIBRATO_HZ: Final = 3.0
_VIBRATO_DEPTH: Final = 0.05

#: Amplitude of a normally-spoken phrase, in int16 counts.  About -23 dBFS RMS,
#: which is where a headset at a sane input level lands.
SPEECH_AMPLITUDE: Final = 9000

#: Room tone of a quiet flat.
QUIET_AMPLITUDE: Final = 12

#: Room tone next to a desktop fan.
NOISY_AMPLITUDE: Final = 900


def silence(ms: int, amplitude: int = QUIET_AMPLITUDE, *, rng: random.Random) -> array[int]:
    """Room tone: uniform dither at ``amplitude`` counts.

    Not digital zero.  A file of zeros is not silence, it is a disconnected
    microphone, and it would let a broken noise-floor estimate pass the tests.
    """
    count = SAMPLE_RATE * ms // 1000
    return array("h", (rng.randint(-amplitude, amplitude) for _ in range(count)))


def voiced(ms: int, amplitude: int = SPEECH_AMPLITUDE, *, rng: random.Random) -> array[int]:
    """A sustained vowel, normalised to ``amplitude`` peak."""
    count = SAMPLE_RATE * ms // 1000
    if count <= 0:
        return array("h")
    coefficients = [_resonator(centre, width) for centre, width in _FORMANTS]
    previous1 = [0.0] * len(coefficients)
    previous2 = [0.0] * len(coefficients)
    phase = 0.0
    raw: list[float] = []
    for index in range(count):
        vibrato = math.sin(2.0 * math.pi * _VIBRATO_HZ * index / SAMPLE_RATE)
        frequency = _F0 * (1.0 + _VIBRATO_DEPTH * vibrato)
        phase += frequency / SAMPLE_RATE
        excitation = rng.uniform(-0.02, 0.02)
        if phase >= 1.0:
            phase -= 1.0
            excitation += 1.0
        total = 0.0
        for band, (first, second) in enumerate(coefficients):
            value = excitation + first * previous1[band] + second * previous2[band]
            previous2[band] = previous1[band]
            previous1[band] = value
            total += value
        raw.append(total)
    peak = max(abs(value) for value in raw) or 1.0
    scale = amplitude / peak
    return array("h", (_clip(value * scale) for value in raw))


def mix(*parts: array[int]) -> array[int]:
    """Add signals sample by sample, truncating to the shortest."""
    length = min(len(part) for part in parts)
    return array("h", (_clip(sum(part[index] for part in parts)) for index in range(length)))


def overdrive(samples: array[int], factor: float) -> array[int]:
    """Amplify past full scale so the peaks saturate.

    This is what clipping actually looks like: not one sample touching 32767,
    but flat tops across a good share of the waveform.  Scaling a normalised
    signal *to* 32767 does not clip it, and calibration is right to shrug at
    that - so the fixture has to overdrive properly for the check to mean
    anything.
    """
    return array("h", (_clip(sample * factor) for sample in samples))


def concat(*parts: array[int]) -> array[int]:
    """Join signals end to end."""
    joined = array("h")
    for part in parts:
        joined.extend(part)
    return joined


def write(path: Path, samples: array[int]) -> Path:
    """Write mono 16-bit PCM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return path


def build(directory: Path) -> list[Path]:
    """Write every fixture and return the paths, for the log line."""
    rng = random.Random(SEED)
    written: list[Path] = []

    # An empty room.  Calibration measures its floor; the segmenter must find
    # nothing at all in it.
    written.append(write(directory / "silence.wav", silence(2000, rng=rng)))

    # One phrase with room on both sides.  The onset is 400 ms in, so a
    # segmenter that forgets the pre-roll produces a visibly shorter segment.
    phrase = concat(
        silence(400, rng=rng),
        voiced(900, rng=rng),
        silence(900, rng=rng),
    )
    written.append(write(directory / "phrase.wav", phrase))

    # The same phrase with a 300 ms breath in the middle - shorter than the
    # 700 ms end-of-phrase threshold, so it must not split.
    written.append(
        write(
            directory / "phrase_with_pause.wav",
            concat(
                silence(400, rng=rng),
                voiced(500, rng=rng),
                silence(300, rng=rng),
                voiced(600, rng=rng),
                silence(900, rng=rng),
            ),
        )
    )

    # Two phrases separated by a full second: this one *must* split.
    written.append(
        write(
            directory / "two_utterances.wav",
            concat(
                silence(300, rng=rng),
                voiced(600, rng=rng),
                silence(1100, rng=rng),
                voiced(600, rng=rng),
                silence(900, rng=rng),
            ),
        )
    )

    # A 60 ms burst: long enough to confirm an onset, far too short to be a
    # phrase.  This is the click the minimum-length rule exists for.
    written.append(
        write(
            directory / "click.wav",
            concat(silence(400, rng=rng), voiced(60, rng=rng), silence(900, rng=rng)),
        )
    )

    # The phrase again, this time over fan noise at about -30 dBFS.  Detection
    # has to survive it, and denoising has to reduce it.
    speech_in_noise = concat(
        silence(400, NOISY_AMPLITUDE, rng=rng),
        mix(voiced(900, rng=rng), silence(900, NOISY_AMPLITUDE, rng=rng)),
        silence(900, NOISY_AMPLITUDE, rng=rng),
    )
    written.append(write(directory / "phrase_in_noise.wav", speech_in_noise))

    # Two seconds of the same fan without a voice: the noisy-room branch of
    # calibration, and the input the noise gate is measured on.
    written.append(write(directory / "noise.wav", silence(2000, NOISY_AMPLITUDE, rng=rng)))

    # A phrase shouted into the microphone from two centimetres away.  The
    # voice is driven four times past full scale, so the peaks come back
    # flattened and calibration has to say so rather than recommend a gain.
    written.append(
        write(
            directory / "clipped.wav",
            concat(
                silence(300, rng=rng),
                overdrive(voiced(800, rng=rng), 14.0),
                silence(900, rng=rng),
            ),
        )
    )

    # A phrase from somebody sitting too far away: detectable, but the level is
    # hopeless and gain alone will not save it.
    written.append(
        write(
            directory / "quiet_phrase.wav",
            concat(
                silence(400, rng=rng),
                voiced(900, 260, rng=rng),
                silence(900, rng=rng),
            ),
        )
    )
    return written


def _resonator(centre_hz: float, bandwidth_hz: float) -> tuple[float, float]:
    """Two-pole resonator coefficients for a formant."""
    radius = math.exp(-math.pi * bandwidth_hz / SAMPLE_RATE)
    theta = 2.0 * math.pi * centre_hz / SAMPLE_RATE
    return 2.0 * radius * math.cos(theta), -radius * radius


def _clip(value: float) -> int:
    """Round to int16, saturating."""
    return max(-32768, min(32767, int(round(value))))


def main() -> int:
    """Regenerate every fixture next to this script."""
    written = build(Path(__file__).parent)
    total = sum(path.stat().st_size for path in written)
    # A script, not library code: this is its entire output.  Written through
    # sys.stdout because the T20 ban on print() applies repository-wide.
    sys.stdout.write(f"{len(written)} файлов, {total // 1024} КиБ\n")
    for path in written:
        sys.stdout.write(f"  {path.name:24s} {path.stat().st_size // 1024:4d} КиБ\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
