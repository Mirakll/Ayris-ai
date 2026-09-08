"""Manifest-backed sound library, TTS cache and user catalog operations."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ayris.actions.macros.schema import CommandModel, SoundBinding, SoundSource
from ayris.actions.macros.sounds.importer import load_wav, write_wav
from ayris.actions.macros.sounds.mixer import SoundHandle, SoundMixer
from ayris.audio.tts.base import AudioChunk
from ayris.audio.tts.router import TtsRouter
from ayris.core.errors import AudioError
from ayris.core.paths import executable_dir


class SoundLibraryError(AudioError):
    default_user_message = "Не удалось открыть библиотеку звуков."


class TtsSoundSynthesizer(Protocol):
    @property
    def cache_identity(self) -> str: ...

    def synthesize(self, text: str) -> AudioChunk: ...


class RouterSoundSynthesizer:
    """Use the configured TTS router for cached macro phrases."""

    def __init__(self, router: TtsRouter) -> None:
        self._router = router

    @property
    def cache_identity(self) -> str:
        params = self._router.params
        voice = params.voice.key if params.voice is not None else "default"
        return f"{self._router.mode}:{voice}:{params.speed:.4f}:{params.pitch:.4f}"

    def synthesize(self, text: str) -> AudioChunk:
        return self._router.synthesize(text)


class SoundEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=120)
    file: str = Field(pattern=r"^[^/]+[.]wav$")
    duration_ms: int = Field(gt=0, le=60_000)
    category: str = Field(min_length=1, max_length=60)


class SoundManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: int = Field(ge=1, le=1)
    sounds: tuple[SoundEntry, ...]

    @model_validator(mode="after")
    def unique_ids(self) -> SoundManifest:
        ids = [entry.id for entry in self.sounds]
        if len(ids) != len(set(ids)):
            raise ValueError("sound ids are not unique")
        return self


@dataclass(frozen=True, slots=True)
class CatalogSound:
    reference: str
    name: str
    category: str
    duration_ms: int
    builtin: bool
    path: Path
    used_by: int = 0


class SoundLibrary:
    def __init__(
        self,
        sounds_dir: Path,
        cache_dir: Path,
        mixer: SoundMixer,
        *,
        resources_dir: Path | None = None,
        synthesizer: TtsSoundSynthesizer | None = None,
        commands: Iterable[CommandModel] | None = None,
    ) -> None:
        self.sounds_dir = sounds_dir
        self.cache_dir = cache_dir / "macro_sounds"
        self.resources_dir = resources_dir or executable_dir() / "resources" / "sounds"
        self.mixer = mixer
        self.synthesizer = synthesizer
        self.commands = tuple(commands or ())
        self._manifest = self._load_manifest()

    def resolve(self, binding: SoundBinding) -> AudioChunk:
        if binding.source is SoundSource.BUILTIN:
            entry = next((item for item in self._manifest.sounds if item.id == binding.value), None)
            if entry is None:
                raise SoundLibraryError(
                    f"unknown builtin sound: {binding.value}",
                    user_message=f"Встроенный звук «{binding.value}» не найден в манифесте.",
                )
            return load_wav(self.resources_dir / entry.file)
        if binding.source is SoundSource.FILE:
            path = self.sounds_dir / binding.value
            if not path.is_file():
                raise SoundLibraryError(
                    f"custom sound missing: {path}",
                    user_message=f"Пользовательский звук «{binding.value}» не найден.",
                )
            return load_wav(path)
        return self._tts(binding.value)

    def play_binding(self, binding: SoundBinding, *, owner: str = "") -> SoundHandle:
        audio = self.resolve(binding)
        volume = (binding.volume if binding.volume is not None else 100) / 100
        return self.mixer.play(audio, volume=volume, owner=owner, wait=binding.wait)

    def preview(self, reference: str) -> SoundHandle:
        return self.play_binding(SoundBinding(value=reference), owner="preview")

    def list(self, *, search: str = "", category: str = "") -> tuple[CatalogSound, ...]:
        result = [
            CatalogSound(
                f"builtin:{entry.id}",
                entry.name,
                entry.category,
                entry.duration_ms,
                True,
                self.resources_dir / entry.file,
                self.usage_count(f"builtin:{entry.id}"),
            )
            for entry in self._manifest.sounds
        ]
        if self.sounds_dir.is_dir():
            for path in sorted(
                self.sounds_dir.glob("*.wav"), key=lambda item: item.name.casefold()
            ):
                audio = load_wav(path)
                reference = f"custom:{path.name}"
                result.append(
                    CatalogSound(
                        reference,
                        path.stem,
                        "Пользовательские",
                        int(audio.duration_ms),
                        False,
                        path,
                        self.usage_count(reference),
                    )
                )
        needle = search.strip().casefold()
        return tuple(
            item
            for item in result
            if (not category or item.category == category)
            and (
                not needle or needle in item.name.casefold() or needle in item.reference.casefold()
            )
        )

    def rename(self, filename: str, new_name: str) -> Path:
        filename = _custom_filename(filename)
        new_stem = _custom_stem(new_name)
        source = self.sounds_dir / filename
        target = self.sounds_dir / f"{new_stem}.wav"
        if not source.is_file():
            raise SoundLibraryError(
                "cannot rename custom sound",
                user_message="Пользовательский звук не найден.",
            )
        if target.exists():
            raise SoundLibraryError(
                "target sound exists", user_message=f"Звук «{target.name}» уже существует."
            )
        source.rename(target)
        return target

    def delete(self, filename: str, *, force: bool = False) -> int:
        filename = _custom_filename(filename)
        reference = f"custom:{filename}"
        used = self.usage_count(reference)
        if used and not force:
            raise SoundLibraryError(
                f"sound is used by {used} commands",
                user_message=f"Звук используется в {used} командах. Подтвердите удаление.",
            )
        path = self.sounds_dir / filename
        if not path.is_file():
            raise SoundLibraryError(
                "custom sound missing", user_message=f"Звук «{filename}» не найден."
            )
        path.unlink()
        return used

    def usage_count(self, reference: str) -> int:
        return sum(_command_uses(command, reference) for command in self.commands)

    def _tts(self, text: str) -> AudioChunk:
        if self.synthesizer is None:
            raise SoundLibraryError(
                "tts synthesizer unavailable", user_message="Синтез звука не настроен."
            )
        identity = self.synthesizer.cache_identity
        digest = hashlib.sha256(f"{identity}\0{text}".encode()).hexdigest()
        target = self.cache_dir / f"{digest}.wav"
        if target.is_file():
            return load_wav(target)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.cache_dir.glob("*.voice"):
            if stale.read_text(encoding="utf-8") != identity:
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                break
        audio = self.synthesizer.synthesize(text)
        write_wav(target, audio)
        target.with_suffix(".voice").write_text(identity, encoding="utf-8")
        return audio

    def _load_manifest(self) -> SoundManifest:
        path = self.resources_dir / "sounds.json"
        try:
            manifest = SoundManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            raise SoundLibraryError(
                "invalid sound manifest", user_message="Манифест встроенных звуков повреждён."
            ) from exc
        missing = [
            entry.file
            for entry in manifest.sounds
            if not (self.resources_dir / entry.file).is_file()
        ]
        if missing:
            raise SoundLibraryError(
                "builtin sound files missing",
                user_message=f"Нет встроенных файлов: {', '.join(missing)}.",
            )
        return manifest


def _command_uses(command: CommandModel, reference: str) -> int:
    bindings = list(command.sounds)
    bindings.extend(
        location.block.sound for location in command.blocks() if location.block.sound is not None
    )
    return int(any(binding.reference == reference for binding in bindings))


def _custom_filename(value: str) -> str:
    name = Path(value).name
    if name != value or name in {"", ".", ".."} or Path(name).suffix.lower() != ".wav":
        raise SoundLibraryError(
            "unsafe custom sound name",
            user_message="Имя пользовательского звука недопустимо.",
        )
    return name


def _custom_stem(value: str) -> str:
    stem = value.strip().removesuffix(".wav").rstrip(". ")
    if not stem or len(stem) > 120 or any(char in stem for char in '<>:"/\\|?*'):
        raise SoundLibraryError(
            "unsafe custom sound name",
            user_message="Имя пользовательского звука недопустимо.",
        )
    return stem
