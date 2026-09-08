"""Task 34: macro sound bindings, library, import and shared-output policy."""

from __future__ import annotations

import threading
import wave
from array import array
from pathlib import Path
from typing import Any

import pytest

from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.schema import ActionBlock, CommandModel, SoundBinding, SoundStage
from ayris.actions.macros.sounds import (
    MixPolicy,
    SoundImportError,
    SoundLibrary,
    SoundLibraryError,
    SoundMixer,
    bindings_for_stage,
    import_sound,
)
from ayris.actions.result import ActionResult
from ayris.audio.tts.base import AudioChunk
from ayris.audio.tts.player import SpeechRequest
from ayris.core.errors import AudioError

pytestmark = pytest.mark.unit


def tone(*, rate: int = 16_000, frames: int = 160, level: int = 2_000) -> AudioChunk:
    return AudioChunk(array("h", [level] * frames).tobytes(), rate)


def manifest(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "sounds.json").write_text(
        '{"schema_version":1,"sounds":['
        '{"id":"start","name":"Запуск","file":"start.wav",'
        '"duration_ms":10,"category":"Команды"}]}',
        encoding="utf-8",
    )
    write_wave(directory / "start.wav", tone())


def write_wave(path: Path, audio: AudioChunk) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.channels)
        stream.setsampwidth(2)
        stream.setframerate(audio.sample_rate)
        stream.writeframes(audio.pcm)


class Output:
    def __init__(self, *, speaking: bool = False) -> None:
        self._speaking = speaking
        self.requests: list[SpeechRequest] = []
        self.cancelled: list[str] = []

    @property
    def speaking(self) -> bool:
        return self._speaking

    def submit(self, request: SpeechRequest) -> None:
        self.requests.append(request)

    def cancel(self, request_id: str) -> bool:
        self.cancelled.append(request_id)
        return True


class Synthesizer:
    def __init__(self) -> None:
        self.cache_identity = "piper:irina:1.0"
        self.calls = 0

    def synthesize(self, text: str) -> AudioChunk:
        self.calls += 1
        return tone(level=len(text) * 100)


def library(tmp_path: Path, output: Output | None = None, **kwargs: Any) -> SoundLibrary:
    resources = tmp_path / "resources"
    manifest(resources)
    return SoundLibrary(
        tmp_path / "custom",
        tmp_path / "cache",
        SoundMixer(output or Output()),
        resources_dir=resources,
        **kwargs,
    )


def test_block_binding_replaces_command_at_its_stage() -> None:
    command = CommandModel(
        name="Звук",
        sounds=["builtin:start", {"stage": "on_error", "value": "builtin:error"}],
    )
    block = ActionBlock(
        type="Run",
        sound={"stage": "on_error", "value": "tts:Не вышло"},
    )

    assert [item.reference for item in bindings_for_stage(command, SoundStage.ON_SUCCESS)] == [
        "builtin:start"
    ]
    assert [item.reference for item in bindings_for_stage(command, SoundStage.ON_ERROR, block)] == [
        "tts:Не вышло"
    ]


def test_library_resolves_builtin_custom_and_tts_cache(tmp_path: Path) -> None:
    synth = Synthesizer()
    sounds = library(tmp_path, synthesizer=synth)
    sounds.sounds_dir.mkdir()
    write_wave(sounds.sounds_dir / "mine.wav", tone(level=4_000))

    assert sounds.resolve(SoundBinding(value="builtin:start")).pcm
    assert sounds.resolve(SoundBinding(value="custom:mine.wav")).pcm
    first = sounds.resolve(SoundBinding(value="tts:Готово"))
    second = sounds.resolve(SoundBinding(value="tts:Готово"))

    assert first == second
    assert synth.calls == 1


def test_tts_cache_is_invalidated_when_voice_changes(tmp_path: Path) -> None:
    synth = Synthesizer()
    sounds = library(tmp_path, synthesizer=synth)
    sounds.resolve(SoundBinding(value="tts:Готово"))
    synth.cache_identity = "piper:dmitri:1.0"
    sounds.resolve(SoundBinding(value="tts:Готово"))

    assert synth.calls == 2
    assert len(list(sounds.cache_dir.glob("*.wav"))) == 1


def test_missing_sound_has_readable_error(tmp_path: Path) -> None:
    sounds = library(tmp_path)

    with pytest.raises(SoundLibraryError, match="missing") as caught:
        sounds.resolve(SoundBinding(value="custom:none.wav"))

    assert "не найден" in caught.value.user_message


def test_import_converts_resamples_and_normalises_wav(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    stereo = array("h", [1_000, -1_000, 2_000, 2_000] * 100)
    write_wave(source, AudioChunk(stereo.tobytes(), 8_000, 2))

    result = import_sound(source, tmp_path / "sounds", sample_rate=16_000)

    with wave.open(str(result.path), "rb") as stream:
        pcm = array("h")
        pcm.frombytes(stream.readframes(stream.getnframes()))
        assert stream.getnchannels() == 1
        assert stream.getframerate() == 16_000
    assert max(abs(sample) for sample in pcm) == pytest.approx(32767 * 10 ** (-3 / 20), abs=2)
    assert result.peak_dbfs == pytest.approx(-3.0, abs=0.01)


def test_mp3_uses_decoder_and_bad_files_are_explained(tmp_path: Path) -> None:
    source = tmp_path / "source.mp3"
    source.write_bytes(b"not really mp3")

    class Decoder:
        def decode(self, _source: Path) -> AudioChunk:
            return tone()

    assert import_sound(source, tmp_path / "sounds", decoder=Decoder()).path.suffix == ".wav"
    broken = tmp_path / "broken.wav"
    broken.write_bytes(b"broken")
    with pytest.raises(SoundImportError) as caught:
        import_sound(broken, tmp_path / "sounds")
    assert "исправным WAV" in caught.value.user_message


def test_queue_and_duck_policy_keep_independent_gain() -> None:
    queue_output = Output(speaking=True)
    duck_output = Output(speaking=True)
    SoundMixer(queue_output, policy=MixPolicy.QUEUE, volume=0.5).play(tone(), volume=0.4)
    SoundMixer(duck_output, policy=MixPolicy.DUCK, volume=0.5, duck_db=-6).play(tone(), volume=0.4)

    assert queue_output.requests[0].gain == pytest.approx(0.2)
    assert duck_output.requests[0].gain == pytest.approx(0.2)
    assert duck_output.requests[0].duck_gain == pytest.approx(10 ** (-6 / 20))
    assert queue_output.requests[0].defer_while_speaking
    assert not duck_output.requests[0].defer_while_speaking
    assert queue_output.requests[0].mix
    assert not queue_output.requests[0].priority


def test_voice_limit_and_stop_only_cancel_macro_requests() -> None:
    output = Output()
    mixer = SoundMixer(output, max_voices=1)
    handle = mixer.play(tone(), owner="run")

    with pytest.raises(AudioError, match="voice limit"):
        mixer.play(tone(), owner="other")
    assert mixer.stop("run") == 1
    assert output.cancelled == [handle.request_id]


def test_catalog_preview_rename_delete_and_usage(tmp_path: Path) -> None:
    output = Output()
    command = CommandModel(name="Звук", sounds=["custom:mine.wav"])
    sounds = library(tmp_path, output, commands=[command])
    sounds.sounds_dir.mkdir()
    write_wave(sounds.sounds_dir / "mine.wav", tone())

    assert [item.reference for item in sounds.list(search="зап")] == ["builtin:start"]
    assert sounds.usage_count("custom:mine.wav") == 1
    assert sounds.preview("builtin:start").request_id.startswith("macro-sound:")
    renamed = sounds.rename("mine.wav", "мой звук")
    assert renamed.name == "мой звук.wav"
    with pytest.raises(SoundLibraryError, match="unsafe"):
        sounds.delete("../мой звук.wav", force=True)
    assert sounds.delete("мой звук.wav", force=True) == 0


class Registry:
    def has(self, _name: str) -> bool:
        return True

    def execute(self, name: str, params: dict[str, Any] | None = None, **_kwargs: Any):
        if name == "Fail":
            raise RuntimeError("boom")
        return ActionResult.done("ok")


class BindingPlayer:
    def __init__(self) -> None:
        self.played: list[tuple[str, SoundStage]] = []
        self.lock = threading.Lock()

    def play_binding(self, binding: SoundBinding, *, owner: str = "") -> object:
        with self.lock:
            self.played.append((binding.reference, binding.stage))
        return object()


def test_engine_plays_on_error_for_failed_block() -> None:
    player = BindingPlayer()
    command = CommandModel.model_validate(
        {
            "name": "Ошибка",
            "actions": [
                {
                    "type": "Fail",
                    "sound": {"stage": "on_error", "value": "builtin:error"},
                }
            ],
        }
    )

    with MacroEngine(Registry(), sounds=player) as engine:
        assert engine.run(command).failed

    assert ("builtin:error", SoundStage.ON_ERROR) in player.played
