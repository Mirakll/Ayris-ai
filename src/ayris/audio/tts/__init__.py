"""Text-to-speech engines: local (Piper, Silero, Coqui) and cloud providers.

Each engine implements the common ABC and returns audio to the shared player.

The pieces fit together in one direction: :mod:`~ayris.audio.tts.sentence_split`
cuts an answer into speakable sentences, an engine turns each one into an
:class:`~ayris.audio.tts.base.AudioChunk`,
:mod:`~ayris.audio.tts.cache` remembers the ones that recur, and
:class:`~ayris.audio.tts.player.TtsPlayer` puts them on a device in order. Only
the player touches ``sounddevice``; everything above it is pure bytes, which is
why splitting, caching and synthesis are all testable without an audio card.

Engines are reached through :func:`~ayris.audio.tts.base.create_engine` rather
than imported directly - a vendor library is imported only when its engine is
actually loaded, so a missing ``piper`` or ``torch`` costs a clear error at load
time instead of an :class:`ImportError` at startup. Cloud voices work the same
way through :func:`~ayris.audio.tts.cloud_base.create_cloud_engine`.

Nothing above this package chooses an engine either: everything that speaks calls
:meth:`~ayris.audio.tts.router.TtsRouter.say`, and the router decides between the
local voice and the paid one. It is imported from its own module rather than from
here, as :class:`~ayris.audio.stt.router.SttRouter` is, and for the same reason
the cloud modules are left out of this ``__init__``: they import ``httpx`` at
module level, which offline mode has no reason to pay for.
"""

from __future__ import annotations

from ayris.audio.tts.base import (
    DEFAULT_SAMPLE_RATE,
    MAX_PITCH,
    MAX_SPEED,
    MIN_PITCH,
    MIN_SPEED,
    AudioChunk,
    TtsEngine,
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
from ayris.audio.tts.cache import CacheStats, PhraseCache, phrase_key
from ayris.audio.tts.player import (
    PlaybackReason,
    PlayerStats,
    SpeechRequest,
    TtsPlayer,
    chunks_from,
)
from ayris.audio.tts.sentence_split import normalize_whitespace, split_sentences

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "MAX_PITCH",
    "MAX_SPEED",
    "MIN_PITCH",
    "MIN_SPEED",
    "AudioChunk",
    "CacheStats",
    "PhraseCache",
    "PlaybackReason",
    "PlayerStats",
    "SpeechRequest",
    "TtsEngine",
    "TtsOptions",
    "TtsPlayer",
    "VoiceSpec",
    "chunks_from",
    "clamp_pitch",
    "clamp_speed",
    "concat_chunks",
    "create_engine",
    "engine_class",
    "engine_names",
    "estimate_voice_bytes",
    "normalize_whitespace",
    "phrase_key",
    "split_sentences",
]
