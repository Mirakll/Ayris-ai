"""Disk cache for phrases Ayris says over and over.

«Готово», «Не расслышала», «Слушаю» — a handful of confirmations account for most
of what the assistant ever speaks, and synthesizing them again every time costs
80–200 ms of CPU for a result that is byte-for-byte identical. Caching them turns
the common answer into a file read, which is the difference between the overlay
animating immediately and the overlay animating after a visible pause.

**What the key is made of.** The text, the voice, the speed and the pitch - the
four inputs that determine the samples. Anything else about the engine's state
either does not affect the output or already lives in the voice: two different
Piper models are two different :attr:`~ayris.audio.tts.base.VoiceSpec.path`
values and therefore two different keys.

**Why WAV and not raw PCM.** The cache directory is inside the user's profile,
and a user who opens it should find something they can double-click. ``wave`` is
in the standard library, carries the sample rate and channel count that a raw
dump would need a sidecar file for, and costs a few microseconds per read against
the tens of milliseconds it saves.

**LRU without an index.** Recency is the file's modification time, refreshed on
every hit. An index file would be faster to consult but would have to survive
crashes, concurrent workers and a user deleting files by hand; the filesystem
already tracks this, and a cache that heals itself by being re-derivable is worth
more here than one that is a millisecond quicker to query.

**The voice is in the file name.** Entries are named ``<voice>_<phrase>.wav``,
where the first part hashes the voice. Switching voices then invalidates by
prefix - one directory scan, no reads - instead of leaving a cache full of
entries that will never be hit again and still count against the limit.

Nothing in here imports ``sounddevice`` or any engine, so the tests run anywhere.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ayris.audio.tts.base import SAMPLE_WIDTH, AudioChunk, VoiceSpec
from ayris.core.paths import get_paths

__all__ = [
    "CACHE_DIR_NAME",
    "MAX_TEXT_CHARS",
    "CacheStats",
    "PhraseCache",
    "phrase_key",
    "voice_key",
]

_log = logging.getLogger(__name__)

#: Subdirectory of the profile cache the entries live in.
CACHE_DIR_NAME: Final = "tts"

#: Longer texts are not cached. The cache exists for short repeated phrases; a
#: three-paragraph answer is unlikely to be asked for twice, and storing it would
#: evict fifty confirmations that would have been.
MAX_TEXT_CHARS: Final = 240

#: A single entry may not take more than this share of the budget, however large
#: the budget is. Without it one long phrase could evict everything else and then
#: sit there alone.
_MAX_ENTRY_SHARE: Final = 0.25

#: Hex characters kept from each hash. 8 for the voice (collisions there only
#: cost a stale-looking invalidation) and 24 for the phrase, which is 96 bits -
#: far past the point where a collision in a few thousand files is conceivable.
_VOICE_HASH_CHARS: Final = 8
_PHRASE_HASH_CHARS: Final = 24

_SUFFIX: Final = ".wav"


def voice_key(voice: VoiceSpec) -> str:
    """Short stable hash of a voice, used as the file-name prefix."""
    digest = hashlib.sha256(voice.key.encode("utf-8")).hexdigest()
    return digest[:_VOICE_HASH_CHARS]


def phrase_key(text: str, voice: VoiceSpec, speed: float, pitch: float) -> str:
    """File name for one cache entry.

    Speed and pitch are rounded to two decimals before hashing: the settings
    slider moves in steps of 0.05, and letting floating-point noise through would
    turn a repeated phrase into a permanent miss.
    """
    payload = "\x00".join(
        (
            text.strip(),
            voice.key,
            f"{round(speed, 2):.2f}",
            f"{round(pitch, 2):.2f}",
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{voice_key(voice)}_{digest[:_PHRASE_HASH_CHARS]}{_SUFFIX}"


@dataclass(frozen=True, slots=True)
class CacheStats:
    """What DevTools shows about the cache.

    Attributes:
        entries: Files currently stored.
        bytes_used: Their total size.
        limit_bytes: The configured cap. ``0`` means caching is off.
        hits: Reads served from disk since the process started.
        misses: Reads that had to be synthesized.
        evictions: Entries deleted to stay under the cap.
    """

    entries: int = 0
    bytes_used: int = 0
    limit_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Share of reads that were hits, 0.0 when nothing has been read."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class PhraseCache:
    """LRU cache of synthesized phrases on disk.

    Args:
        directory: Where to keep the entries. Defaults to ``cache/tts`` in the
            profile.
        limit_bytes: Size cap. ``0`` disables the cache entirely - every
            :meth:`get` misses and every :meth:`put` is a no-op, which is what
            ``voice.tts.cache_size_mb = 0`` means.

    Thread-safe: the worker's synthesis thread writes while the player's thread
    may read. The lock is coarse - one per cache, held across the file I/O - on
    the grounds that the operations take microseconds and the contention window
    is smaller than the cost of anything finer.
    """

    __slots__ = ("_directory", "_evictions", "_hits", "_limit", "_lock", "_misses")

    def __init__(self, directory: Path | None = None, limit_bytes: int = 0) -> None:
        self._directory = directory if directory is not None else default_cache_dir()
        self._limit = max(0, limit_bytes)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def directory(self) -> Path:
        """Where the entries live."""
        return self._directory

    @property
    def limit_bytes(self) -> int:
        """Current size cap."""
        return self._limit

    @property
    def enabled(self) -> bool:
        """Whether anything is cached at all."""
        return self._limit > 0

    def set_limit(self, limit_bytes: int) -> None:
        """Change the cap, trimming immediately if it shrank.

        Called when the user moves the slider in the settings: leaving 200 MB on
        disk after they asked for 50 would make the setting a lie.
        """
        with self._lock:
            self._limit = max(0, limit_bytes)
            if self._limit == 0:
                self._clear_locked()
            else:
                self._trim_locked()

    # ------------------------------------------------------------------ reads

    def get(self, text: str, voice: VoiceSpec, speed: float, pitch: float) -> AudioChunk | None:
        """Look up a phrase.

        Returns:
            The stored audio, or ``None`` on a miss. A corrupt or truncated file
            is a miss too, and is deleted on the way out - a half-written entry
            from a crash must not make a phrase permanently unspeakable.
        """
        if not self.enabled or not text.strip():
            return None
        path = self._directory / phrase_key(text, voice, speed, pitch)
        with self._lock:
            chunk = _read_wav(path)
            if chunk is None:
                self._misses += 1
                _unlink(path)
                return None
            self._hits += 1
            _touch(path)
            return chunk

    def put(
        self,
        text: str,
        voice: VoiceSpec,
        speed: float,
        pitch: float,
        chunk: AudioChunk,
    ) -> bool:
        """Store a phrase, evicting older ones if the cap is reached.

        Returns:
            Whether it was stored. ``False`` for a disabled cache, an empty or
            over-long phrase, an entry too large for the budget, or a write that
            failed - none of which are errors the caller should act on, because
            the audio is already in hand either way.
        """
        if not self.enabled or chunk.empty:
            return False
        stripped = text.strip()
        if not stripped or len(stripped) > MAX_TEXT_CHARS:
            return False
        if len(chunk.pcm) > self._limit * _MAX_ENTRY_SHARE:
            return False

        path = self._directory / phrase_key(stripped, voice, speed, pitch)
        with self._lock:
            if not _write_wav(path, chunk):
                return False
            self._trim_locked()
        return True

    # ------------------------------------------------------------ maintenance

    def invalidate(self, voice: VoiceSpec | None = None) -> int:
        """Drop entries that no longer apply.

        Args:
            voice: When given, keep this voice and delete everything else -
                what a voice change calls. When ``None``, delete everything.

        Returns:
            Number of files removed.
        """
        with self._lock:
            if voice is None:
                return self._clear_locked()
            keep = f"{voice_key(voice)}_"
            removed = 0
            for entry in self._entries_locked():
                if not entry.name.startswith(keep):
                    _unlink(entry)
                    removed += 1
            if removed:
                _log.info("tts: кэш очищен от чужих голосов, удалено %d файлов", removed)
            return removed

    def stats(self) -> CacheStats:
        """Current counters, for DevTools and the tests."""
        with self._lock:
            entries = self._entries_locked()
            return CacheStats(
                entries=len(entries),
                bytes_used=sum(_size(entry) for entry in entries),
                limit_bytes=self._limit,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    # ---------------------------------------------------------------- internals

    def _entries_locked(self) -> list[Path]:
        """Cache files, or an empty list when the directory is unusable."""
        try:
            return [item for item in self._directory.glob(f"*{_SUFFIX}") if item.is_file()]
        except OSError as exc:
            _log.debug("tts: не удалось прочитать каталог кэша %s: %s", self._directory, exc)
            return []

    def _clear_locked(self) -> int:
        """Delete every entry. Caller holds the lock."""
        removed = 0
        for entry in self._entries_locked():
            _unlink(entry)
            removed += 1
        return removed

    def _trim_locked(self) -> None:
        """Evict least-recently-used entries until the cache fits.

        Sorted by modification time, which :meth:`get` refreshes on every hit,
        so "oldest mtime" means "longest since anyone asked for it". Entries
        that vanish mid-loop - another process, or the user cleaning up - are
        skipped rather than raised on.
        """
        entries = self._entries_locked()
        total = sum(_size(entry) for entry in entries)
        if total <= self._limit:
            return
        entries.sort(key=_mtime)
        for entry in entries:
            if total <= self._limit:
                break
            size = _size(entry)
            if _unlink(entry):
                total -= size
                self._evictions += 1
        _log.debug(
            "tts: кэш подрезан до %d байт при лимите %d",
            total,
            self._limit,
        )


def default_cache_dir() -> Path:
    """``cache/tts`` inside the active profile."""
    return get_paths().cache_dir / CACHE_DIR_NAME


# --------------------------------------------------------------- file helpers


def _read_wav(path: Path) -> AudioChunk | None:
    """Read one entry, or ``None`` if it is missing or unreadable."""
    try:
        with wave.open(str(path), "rb") as source:
            if source.getsampwidth() != SAMPLE_WIDTH:
                return None
            frames = source.readframes(source.getnframes())
            if not frames:
                return None
            return AudioChunk(
                pcm=frames,
                sample_rate=source.getframerate(),
                channels=source.getnchannels(),
            )
    except (OSError, wave.Error, EOFError) as exc:
        _log.debug("tts: запись кэша %s нечитаема: %s", path.name, exc)
        return None


def _write_wav(path: Path, chunk: AudioChunk) -> bool:
    """Write one entry atomically.

    Through a temporary file and a rename, because a reader hitting a half-
    written WAV would see a valid header with a wrong frame count and play
    silence - a failure mode that would be blamed on the engine, not the cache.
    """
    temporary = path.with_suffix(f".{time.monotonic_ns():x}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(temporary), "wb") as target:
            target.setnchannels(max(1, chunk.channels))
            target.setsampwidth(SAMPLE_WIDTH)
            target.setframerate(chunk.sample_rate)
            target.writeframes(chunk.pcm)
        temporary.replace(path)
    except (OSError, wave.Error) as exc:
        _log.debug("tts: не удалось записать кэш %s: %s", path.name, exc)
        _unlink(temporary)
        return False
    return True


def _unlink(path: Path) -> bool:
    """Delete a file, reporting whether it is gone now."""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        _log.debug("tts: не удалось удалить %s: %s", path.name, exc)
        return False
    return True


def _touch(path: Path) -> None:
    """Mark an entry as just used. A failure here only costs LRU accuracy."""
    with contextlib.suppress(OSError):
        path.touch()


def _size(path: Path) -> int:
    """File size, 0 if it vanished."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _mtime(path: Path) -> float:
    """Modification time, 0.0 if it vanished - which sorts it first for eviction."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
