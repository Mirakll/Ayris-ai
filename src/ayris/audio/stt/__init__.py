"""Speech-to-text engines: offline (Vosk, faster-whisper) and cloud providers.

Every engine implements :class:`~ayris.audio.stt.base.SttEngine`, so the code
above this package - :mod:`ayris.workers.stt_worker` and
:class:`~ayris.audio.stt.router.SttRouter` - never names one.  Which engine runs
is ``voice.stt.offline_engine`` or ``voice.stt.online_provider``, resolved to a
class by :func:`~ayris.audio.stt.base.create_engine` and
:func:`~ayris.audio.stt.cloud_base.create_cloud_engine`.

Only :mod:`~ayris.audio.stt.base` is re-exported here.  The engine modules import
their vendor libraries lazily inside ``load()``, and importing them from this
``__init__`` would undo that: a user on Vosk would pay for CTranslate2 being on
disk, and the test suite would stop being collectable on a runner without it.
The cloud modules are left out for the same reason in reverse - they import httpx
at module level, which offline mode has no reason to pay for.
"""

from __future__ import annotations

from ayris.audio.stt.base import (
    DEFAULT_LANGUAGE,
    ENGINE_ENTRYPOINTS,
    MIN_SPEECH_MS,
    SILENCE_DBFS,
    STT_SAMPLE_RATE,
    AudioBuffer,
    SttEngine,
    SttOptions,
    TranscriptResult,
    TranscriptSegment,
    create_engine,
    engine_class,
    engine_names,
    estimate_model_bytes,
)

__all__ = [
    "DEFAULT_LANGUAGE",
    "ENGINE_ENTRYPOINTS",
    "MIN_SPEECH_MS",
    "SILENCE_DBFS",
    "STT_SAMPLE_RATE",
    "AudioBuffer",
    "SttEngine",
    "SttOptions",
    "TranscriptResult",
    "TranscriptSegment",
    "create_engine",
    "engine_class",
    "engine_names",
    "estimate_model_bytes",
]
