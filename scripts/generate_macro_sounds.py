"""Generate the deterministic built-in tones shipped by task 34."""

from __future__ import annotations

import math
import wave
from array import array
from pathlib import Path

RATE = 48_000
SPECS = {
    "start": ((720, 980), 260),
    "success": ((660, 880, 1100), 360),
    "error": ((360, 260), 420),
    "cancel": ((620, 420), 300),
    "timer": ((880, 880, 1175), 520),
    "volume_changed": ((520, 780), 280),
    "wake_word": ((600, 900, 1200), 400),
}


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "resources" / "sounds"
    for name, (frequencies, duration_ms) in SPECS.items():
        frames = RATE * duration_ms // 1000
        segment = max(1, frames // len(frequencies))
        samples = array("h")
        for index in range(frames):
            frequency = frequencies[min(len(frequencies) - 1, index // segment)]
            envelope = min(1.0, index / (RATE * 0.012)) * min(
                1.0, (frames - index) / (RATE * 0.045)
            )
            phase = 2 * math.pi * frequency * index / RATE
            value = int(9_200 * envelope * (math.sin(phase) + 0.22 * math.sin(phase * 2.01)))
            samples.append(max(-32768, min(32767, value)))
        with wave.open(str(root / f"{name}.wav"), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(RATE)
            output.writeframes(samples.tobytes())


if __name__ == "__main__":
    main()
