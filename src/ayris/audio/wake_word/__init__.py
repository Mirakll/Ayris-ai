"""Wake word detection: "Айрис" and whatever else the user calls the assistant.

The package splits along one line.  :mod:`~ayris.audio.wake_word.base` and the
three engine modules answer "does this frame sound like the word", and nothing
else.  :mod:`~ayris.audio.wake_word.manager` owns everything that is the same
whichever engine is selected - resampling, framing, per-phrase thresholds, the
debounce window, the false-positive counters, and the thread the inference runs
on.

Engine modules are imported lazily by
:func:`~ayris.audio.wake_word.base.create_engine`, so selecting openWakeWord
never costs a user of Porcupine an ONNX runtime in memory, and a missing vendor
package produces a sentence in the settings window instead of an ImportError at
startup.
"""

from __future__ import annotations

from ayris.audio.wake_word.base import (
    DEFAULT_SENSITIVITY,
    MAX_THRESHOLD,
    MIN_THRESHOLD,
    PTT_PHRASE,
    WAKE_SAMPLE_RATE,
    ModelSpec,
    WakeDetection,
    WakePhrase,
    WakeWordEngine,
    create_engine,
    engine_names,
    normalise_phrase,
    phrases_from,
)
from ayris.audio.wake_word.manager import (
    WakeStats,
    WakeWordCallbacks,
    WakeWordDetector,
    WakeWordSettings,
)

__all__ = [
    "DEFAULT_SENSITIVITY",
    "MAX_THRESHOLD",
    "MIN_THRESHOLD",
    "PTT_PHRASE",
    "WAKE_SAMPLE_RATE",
    "ModelSpec",
    "WakeDetection",
    "WakePhrase",
    "WakeStats",
    "WakeWordCallbacks",
    "WakeWordDetector",
    "WakeWordEngine",
    "WakeWordSettings",
    "create_engine",
    "engine_names",
    "normalise_phrase",
    "phrases_from",
]
