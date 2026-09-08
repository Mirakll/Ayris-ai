"""Import user audio into portable mono PCM WAV."""

from __future__ import annotations

import math
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ayris.audio.tts.base import AudioChunk
from ayris.core.errors import AudioError

INTERNAL_SAMPLE_RATE = 48_000
SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg"})


class SoundImportError(AudioError):
    """A sound file cannot be decoded or converted."""

    default_user_message = "Не удалось импортировать звук."


class SoundDecoder(Protocol):
    def decode(self, source: Path) -> AudioChunk: ...


@dataclass(frozen=True, slots=True)
class ImportResult:
    path: Path
    duration_s: float
    peak_dbfs: float
    warning: str = ""


def import_sound(
    source: Path,
    sounds_dir: Path,
    *,
    name: str = "",
    decoder: SoundDecoder | None = None,
    sample_rate: int = INTERNAL_SAMPLE_RATE,
    target_dbfs: float = -3.0,
    long_sound_s: float = 15.0,
) -> ImportResult:
    """Decode, resample, peak-normalise and atomically save one sound."""
    source = Path(source)
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise SoundImportError(
            "unsupported container", user_message="Поддерживаются только WAV, MP3 и OGG."
        )
    if not source.is_file():
        raise SoundImportError(
            "sound file missing", user_message=f"Файл звука «{source.name}» не найден."
        )
    try:
        audio = load_wav(source) if suffix == ".wav" else _decode(source, decoder)
        values = _mono(audio)
        values = _resample(values, audio.sample_rate, sample_rate)
        values = _normalise(values, target_dbfs)
    except SoundImportError:
        raise
    except Exception as exc:
        raise SoundImportError(
            f"cannot decode {source}: {exc}",
            user_message=f"Файл «{source.name}» повреждён или имеет неподдерживаемый формат.",
        ) from exc
    stem = _safe_stem(name or source.stem)
    sounds_dir.mkdir(parents=True, exist_ok=True)
    target = _unique_path(sounds_dir, stem)
    with tempfile.NamedTemporaryFile(dir=sounds_dir, suffix=".tmp", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        write_wav(temporary_path, AudioChunk(values.astype("<i2").tobytes(), sample_rate, 1))
        temporary_path.replace(target)
    finally:
        temporary_path.unlink(missing_ok=True)
    duration = len(values) / sample_rate
    warning = (
        f"Звук длится {duration:.1f} с — лучше короче {long_sound_s:g} с."
        if duration > long_sound_s
        else ""
    )
    return ImportResult(target, duration, _peak_dbfs(values), warning)


def load_wav(path: Path) -> AudioChunk:
    """Read integer PCM WAV and convert its samples to signed 16-bit."""
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            width = stream.getsampwidth()
            rate = stream.getframerate()
            raw = stream.readframes(stream.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise SoundImportError(
            "invalid wav", user_message=f"Файл «{path.name}» не является исправным WAV."
        ) from exc
    if channels < 1 or rate < 1 or width not in (1, 2, 3, 4):
        raise SoundImportError(
            "unsupported wav layout", user_message=f"Формат WAV «{path.name}» не поддерживается."
        )
    return AudioChunk(_pcm16(raw, width).tobytes(), rate, channels)


def write_wav(path: Path, audio: AudioChunk) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.channels)
        stream.setsampwidth(2)
        stream.setframerate(audio.sample_rate)
        stream.writeframes(audio.pcm)


def _decode(source: Path, decoder: SoundDecoder | None) -> AudioChunk:
    if decoder is not None:
        return decoder.decode(source)
    try:
        import av
    except ImportError as exc:
        raise SoundImportError(
            "compressed decoder unavailable",
            user_message="Для импорта MP3 и OGG нужен декодер PyAV из полного комплекта Ayris.",
        ) from exc
    try:
        with av.open(str(source)) as container:
            frames = list(container.decode(audio=0))
        if not frames:
            raise ValueError("no audio stream")
        rate = int(frames[0].sample_rate)
        arrays = [
            np.asarray(frame.to_ndarray(format="s16"), dtype=np.int16)  # type: ignore[call-arg]
            for frame in frames
        ]
        data = np.concatenate(arrays, axis=-1)
    except Exception as exc:
        raise SoundImportError(
            "compressed decode failed", user_message=f"Не удалось декодировать «{source.name}»."
        ) from exc
    channels = int(data.shape[0]) if data.ndim == 2 else 1
    return AudioChunk(data.T.astype("<i2").tobytes(), rate, channels)


def _pcm16(raw: bytes, width: int) -> NDArray[np.int16]:
    if width == 1:
        return ((np.frombuffer(raw, np.uint8).astype(np.int16) - 128) << 8).astype(np.int16)
    if width == 2:
        return np.frombuffer(raw, "<i2").copy()
    if width == 3:
        data = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        values = data[:, 0] | data[:, 1] << 8 | data[:, 2] << 16
        return (np.where(values & 0x800000, values - 0x1000000, values) >> 8).astype(np.int16)
    return (np.frombuffer(raw, "<i4") >> 16).astype(np.int16)


def _mono(audio: AudioChunk) -> NDArray[np.float64]:
    values = np.frombuffer(audio.pcm, "<i2").astype(np.float64)
    if audio.channels > 1:
        values = (
            values[: len(values) - len(values) % audio.channels].reshape(-1, audio.channels).mean(1)
        )
    return values


def _resample(values: NDArray[np.float64], source: int, target: int) -> NDArray[np.float64]:
    if source == target or not values.size:
        return values
    count = max(1, round(values.size * target / source))
    return np.interp(np.linspace(0, values.size - 1, count), np.arange(values.size), values)


def _normalise(values: NDArray[np.float64], target_dbfs: float) -> NDArray[np.float64]:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    if peak <= 0:
        return values
    wanted = 32767.0 * 10.0 ** (min(0.0, target_dbfs) / 20.0)
    return np.asarray(np.clip(values * wanted / peak, -32768, 32767), dtype=np.float64)


def _peak_dbfs(values: NDArray[np.float64]) -> float:
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return 20.0 * math.log10(peak / 32767.0) if peak else -math.inf


def _safe_stem(value: str) -> str:
    cleaned = " ".join(value.strip().split()).rstrip(". ")
    if not cleaned or any(char in cleaned for char in '<>:"/\\|?*'):
        raise SoundImportError(
            "unsafe sound name", user_message="Имя звука содержит недопустимые символы."
        )
    return cleaned[:120]


def _unique_path(directory: Path, stem: str) -> Path:
    target = directory / f"{stem}.wav"
    number = 2
    while target.exists():
        target = directory / f"{stem} ({number}).wav"
        number += 1
    return target
