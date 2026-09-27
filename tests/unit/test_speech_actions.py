"""The «Голос/Звук» macro actions, with the runtime faked out.

:mod:`ayris.actions.system.speech` reaches the speech and sound runtime through
two process-global handles — :func:`~ayris.audio.tts.app_router.active_tts_router`
and :func:`~ayris.actions.macros.sounds.runtime.active_sound_library` — that the
dispatcher wires at start-up. That is exactly the seam these tests stand on: a
fake router and a fake library are installed through the same setters, so every
assertion here is about what the action *asked the runtime to do* — the text it
spoke, the binding it played, the voice it set — without a sound card, a network,
or a Qt loop.

Four behaviours carry the weight:

* speaking passes the text straight through and, by default, waits for it;
* playing turns a reference string into the right :class:`SoundBinding` and hands
  it to the mixer under its own owner, so «Остановить звук» can find it;
* stopping asks the mixer to stop *macro* sounds and reports how many;
* a runtime that never came up (the handle is ``None``) is a handled failure with
  a Russian sentence, not an exception — the one path a real machine without an
  audio stack actually takes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ayris.actions.macros.sounds.runtime import set_active_sound_library
from ayris.actions.system.speech import PlaySound, Say, SetTTSVoice, StopSound
from ayris.audio.tts.app_router import set_active_tts_router
from ayris.audio.tts.base import VoiceSpec
from ayris.audio.tts.router import VoiceParams

if TYPE_CHECKING:
    from collections.abc import Iterator


class FakeHandle:
    """What a faked ``say`` hands back: it records whether it was waited on."""

    def __init__(self, engine: str = "piper") -> None:
        self.engine = engine
        self.waited = False

    def wait(self, timeout: float | None = None) -> bool:
        self.waited = True
        return True


class FakeRouter:
    """A stand-in for :class:`~ayris.audio.tts.router.TtsRouter`."""

    def __init__(self, voice: VoiceSpec | None = None) -> None:
        self.params = VoiceParams(voice=voice)
        self.said: list[str] = []
        self.handles: list[FakeHandle] = []
        self.set_params_calls: list[VoiceParams] = []

    def say(self, text: str) -> FakeHandle:
        self.said.append(text)
        handle = FakeHandle()
        self.handles.append(handle)
        return handle

    def set_params(self, params: VoiceParams) -> None:
        self.set_params_calls.append(params)
        self.params = params


class FakeMixer:
    def __init__(self, stopped: int = 0) -> None:
        self._stopped = stopped
        self.stop_calls = 0

    def stop(self, owner: str = "") -> int:
        self.stop_calls += 1
        return self._stopped


class FakeLibrary:
    """A stand-in for :class:`~ayris.actions.macros.sounds.library.SoundLibrary`."""

    def __init__(self, stopped: int = 0) -> None:
        self.mixer = FakeMixer(stopped)
        self.played: list[tuple[str, str]] = []

    def play_binding(self, binding: object, *, owner: str = "") -> None:
        self.played.append((binding.reference, owner))  # type: ignore[attr-defined]


@pytest.fixture
def router() -> Iterator[FakeRouter]:
    fake = FakeRouter()
    set_active_tts_router(fake)  # type: ignore[arg-type]
    try:
        yield fake
    finally:
        set_active_tts_router(None)


@pytest.fixture
def no_router() -> Iterator[None]:
    set_active_tts_router(None)
    yield


@pytest.fixture
def library() -> Iterator[FakeLibrary]:
    fake = FakeLibrary()
    set_active_sound_library(fake)  # type: ignore[arg-type]
    try:
        yield fake
    finally:
        set_active_sound_library(None)


@pytest.fixture
def no_library() -> Iterator[None]:
    set_active_sound_library(None)
    yield


class TestSay:
    def test_speaks_the_text_and_waits_by_default(self, router: FakeRouter) -> None:
        result = Say().run(Say.Params(text="Привет"))
        assert result.ok
        assert router.said == ["Привет"]
        assert router.handles[0].waited is True

    def test_wait_false_does_not_block(self, router: FakeRouter) -> None:
        result = Say().run(Say.Params(text="Привет", wait=False))
        assert result.ok
        assert router.handles[0].waited is False

    def test_no_router_is_a_handled_failure(self, no_router: None) -> None:
        result = Say().run(Say.Params(text="Привет"))
        assert result.ok is False
        assert "недоступен" in result.message_ru


class TestPlaySound:
    def test_builtin_reference_is_played_under_the_macro_owner(self, library: FakeLibrary) -> None:
        result = PlaySound().run(PlaySound.Params(sound="notification"))
        assert result.ok
        assert library.played == [("builtin:notification", "macro:play")]

    def test_tts_prefix_becomes_a_tts_binding(self, library: FakeLibrary) -> None:
        PlaySound().run(PlaySound.Params(sound="tts:Готово"))
        assert library.played == [("tts:Готово", "macro:play")]

    def test_no_library_is_a_handled_failure(self, no_library: None) -> None:
        result = PlaySound().run(PlaySound.Params(sound="notification"))
        assert result.ok is False
        assert "недоступен" in result.message_ru


class TestStopSound:
    def test_stops_macro_sounds_and_reports_the_count(self, library: FakeLibrary) -> None:
        library.mixer = FakeMixer(stopped=2)
        result = StopSound().run(StopSound.Params())
        assert result.ok
        assert library.mixer.stop_calls == 1
        assert result.message_ru == "Звук остановлен."

    def test_nothing_playing_says_nothing(self, library: FakeLibrary) -> None:
        result = StopSound().run(StopSound.Params())
        assert result.ok
        assert result.message_ru == ""

    def test_no_library_is_a_quiet_success(self, no_library: None) -> None:
        result = StopSound().run(StopSound.Params())
        assert result.ok
        assert result.message_ru == ""


class TestSetTTSVoice:
    def test_sets_the_voice_on_the_running_router(self) -> None:
        fake = FakeRouter(voice=VoiceSpec(engine="piper", voice_id="old"))
        set_active_tts_router(fake)  # type: ignore[arg-type]
        try:
            result = SetTTSVoice().run(SetTTSVoice.Params(voice="несуществующий-голос"))
        finally:
            set_active_tts_router(None)
        assert result.ok
        assert fake.set_params_calls, "set_params was not called"
        applied = fake.set_params_calls[-1].voice
        assert applied is not None
        assert applied.engine == "piper"
        assert applied.voice_id == "несуществующий-голос"

    def test_no_router_is_a_handled_failure(self, no_router: None) -> None:
        result = SetTTSVoice().run(SetTTSVoice.Params(voice="irina"))
        assert result.ok is False
        assert "недоступен" in result.message_ru
