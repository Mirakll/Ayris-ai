"""Единый владелец вывода звука команд и общий хендл (превью редактора + рантайм).

Ничего не звучит и ни один поток не крутится: превью и остановка проверяются через
фейковый :class:`~ayris.actions.macros.sounds.mixer.SoundOutput` со счётчиками
(как в ``test_macro_sounds``). Фабрика строит настоящий плеер, но устройство не
открывает (плеер открывает его лениво на первой фразе), поэтому PortAudio не нужен.
Проверки: общий хендл (set/get, очистка), фабрика собирает библиотеку и даёт
безопасный ``stop``, адаптер «Прослушать» режет громкость, не ждёт окончания и
играет под владельцем «preview», а ``Стоп`` отменяет только его.
"""

from __future__ import annotations

import wave
from array import array
from pathlib import Path

import pytest

from ayris.actions.macros.schema import SoundBinding
from ayris.actions.macros.sounds import (
    SoundLibrary,
    SoundMixer,
    active_sound_library,
    build_sound_library,
    set_active_sound_library,
)
from ayris.audio.tts.base import AudioChunk
from ayris.audio.tts.player import SpeechRequest
from ayris.gui.tabs.commands import _build_sound_preview, _LibrarySoundPreview

pytestmark = pytest.mark.unit


def _tone(*, rate: int = 16_000, frames: int = 320, level: int = 4_000) -> AudioChunk:
    return AudioChunk(array("h", [level] * frames).tobytes(), rate)


def _write_wave(path: Path, audio: AudioChunk) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.channels)
        stream.setsampwidth(2)
        stream.setframerate(audio.sample_rate)
        stream.writeframes(audio.pcm)


def _manifest(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "sounds.json").write_text(
        '{"schema_version":1,"sounds":['
        '{"id":"start","name":"Запуск","file":"start.wav",'
        '"duration_ms":20,"category":"Команды"}]}',
        encoding="utf-8",
    )
    _write_wave(directory / "start.wav", _tone())


class _Output:
    """A :class:`SoundOutput` that records requests instead of playing them."""

    def __init__(self) -> None:
        self.requests: list[SpeechRequest] = []
        self.cancelled: list[str] = []

    @property
    def speaking(self) -> bool:
        return False

    def submit(self, request: SpeechRequest) -> None:
        self.requests.append(request)

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True


def _library(tmp_path: Path, output: _Output | None = None) -> SoundLibrary:
    resources = tmp_path / "resources"
    _manifest(resources)
    return SoundLibrary(
        tmp_path / "custom",
        tmp_path / "cache",
        SoundMixer(output or _Output()),
        resources_dir=resources,
    )


@pytest.fixture(autouse=True)
def _clear_active() -> object:
    set_active_sound_library(None)
    yield
    set_active_sound_library(None)


def test_active_handle_round_trips(tmp_path: Path) -> None:
    assert active_sound_library() is None
    library = _library(tmp_path)
    set_active_sound_library(library)
    assert active_sound_library() is library
    set_active_sound_library(None)
    assert active_sound_library() is None


def test_build_returns_library_and_stop(tmp_path: Path) -> None:
    # The builtin manifest lives at the real resources dir, which the library finds
    # itself. The player opens no device until the first sound, so this touches no
    # PortAudio; the stop callback must be callable, quiet and idempotent.
    built = build_sound_library(sounds_dir=tmp_path / "sounds", cache_dir=tmp_path / "cache")
    assert built is not None
    library, stop = built
    assert isinstance(library, SoundLibrary)
    stop()
    stop()


def test_preview_plays_builtin_without_waiting(tmp_path: Path) -> None:
    output = _Output()
    preview = _LibrarySoundPreview(_library(tmp_path, output))

    handle = preview.preview_binding(SoundBinding(value="builtin:start", volume=50))

    assert handle.request_id.startswith("macro-sound:")
    # One request, under the «preview» owner, at half gain and not waiting.
    (request,) = output.requests
    assert request.gain == pytest.approx(0.8 * 0.5)
    assert preview.duration_ms(SoundBinding(value="builtin:start")) == 20


def test_stop_cancels_only_preview(tmp_path: Path) -> None:
    output = _Output()
    library = _library(tmp_path, output)
    preview = _LibrarySoundPreview(library)
    # A command's own sound plays under a different owner and must survive Стоп.
    run_handle = library.play_binding(SoundBinding(value="builtin:start"), owner="run")
    preview_handle = preview.preview_binding(SoundBinding(value="builtin:start"))

    preview.stop()

    assert output.cancelled == [preview_handle.request_id]
    assert run_handle.request_id not in output.cancelled


def test_build_preview_follows_active_handle(tmp_path: Path) -> None:
    assert _build_sound_preview() is None
    set_active_sound_library(_library(tmp_path))
    built = _build_sound_preview()
    assert isinstance(built, _LibrarySoundPreview)
