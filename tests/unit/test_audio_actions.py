"""Task 21: громкость, микшер и звуковые устройства — без звуковой карты.

Two modules are covered here — :mod:`ayris.actions.system.audio` and
:mod:`ayris.actions.system.audio_devices` — and they are testable for the same
reason the window actions are: every COM call is behind a narrow protocol
(:class:`~…audio.AudioBackend`, :class:`~…audio_devices.DeviceBackend`) and every
decision is on this side of it. The fakes below are the whole Windows audio stack
as far as these tests are concerned, so the assertions are about the exact scalar
that reached WASAPI and the exact Russian sentence that reached the user.

Three things carry the most weight, and each is asserted directly.

*Percent goes in, a scalar comes out, and the conversion happens in one place.*
WASAPI wants ``0.6`` where a person says «шестьдесят», and the two are easy to
confuse in a way that sets the volume to maximum. Every test that changes the
volume checks what the backend was handed, not just that it was called — a
regression that passes ``60`` down is a wrong number, not a missing call.

*Out of range means «as far as it goes», not an error.* «Громче» at 95 is a
request for 100, and :func:`~…audio.clamp_volume` is what makes it one. The
pydantic bounds still refuse a macro that literally asks for 120, because that is
a mistake in the macro rather than something a person said.

*A machine with no sound card answers in Russian.* This is the one behaviour the
CI runner checks for real: it has no audio endpoints at all, so the listing is
empty there and the actions must say «не нашла» rather than raise a ``COMError``.
:class:`~…audio_devices.DeviceUnavailable` and :class:`~…audio.MixerUnavailable`
are :class:`~ayris.core.errors.ActionUnavailable` subclasses for exactly that.

Groups:

* :class:`TestConversion` — percent ↔ scalar, clamping, and the read-back rounding.
* :class:`TestVolumeStateShape` — the copies a setter reports back, and the JSON form.
* :class:`TestAudioSessionShape` — the resource-string display name, the spoken label.
* :class:`TestSessionMatching` — five gradations, active first, «хром» to ``chrome``.
* :class:`TestFindSessions` — one program means all its sessions; the two-pass order.
* :class:`TestSetVolumeAction` — the scalar that reached WASAPI, mute lifted, undo.
* :class:`TestAdjustVolumeAction` — the configured step, an explicit amount, the stops.
* :class:`TestMuteToggleAction` — three modes, and the microphone's own phrasing.
* :class:`TestSetAppVolumeAction` — every session of one program, and the refusal.
* :class:`TestSetMicVolumeAction` — the capture endpoint, and what is never published.
* :class:`TestVolumeStep` — the step comes from the config and is re-read every time.
* :class:`TestDeviceMatching` — four gradations over device names, in Russian.
* :class:`TestDeviceListing` — the default first, and the rubbish filtered out.
* :class:`TestSetAudioDeviceAction` — all three roles moved, and the two refusals.
* :class:`TestBackendSeams` — the injection points and the off-Windows refusals.
* :class:`TestSchemas` — six actions as the macro editor sees them.
* :class:`TestLiveAudio` — read-only checks against the real WASAPI, on Windows only.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from ayris.actions.base import FieldKind, build_schema
from ayris.actions.system.audio import (
    AdjustVolume,
    AudioSession,
    MuteMode,
    MuteToggle,
    SessionNotFound,
    SetAppVolume,
    SetMicVolume,
    SetVolume,
    VolumeState,
    clamp_volume,
    find_sessions,
    get_audio_backend,
    match_sessions,
    percent_to_scalar,
    scalar_to_percent,
    set_audio_backend,
    volume_step,
)
from ayris.actions.system.audio_devices import (
    AudioDevice,
    DeviceKind,
    DeviceNotFound,
    DeviceState,
    DeviceUnavailable,
    SetAudioDevice,
    default_device,
    find_device,
    get_device_backend,
    list_audio_devices,
    match_devices,
    set_device_backend,
)
from ayris.core.errors import ActionUnavailable

if TYPE_CHECKING:
    from collections.abc import Iterator

#: A real session on a real machine: the system sounds, whose «name» is a
#: reference into a DLL's resource table rather than anything a person says.
SYSTEM_SOUNDS_NAME = "@%SystemRoot%\\System32\\AudioSrv.Dll,-202"


# --------------------------------------------------------------------------- #
# Fakes: the whole audio stack, as far as these tests are concerned
# --------------------------------------------------------------------------- #


class FakeAudio:
    """An :class:`AudioBackend` that remembers every scalar it was handed.

    ``volume_calls`` holds floats and not percentages on purpose: the one thing
    these tests exist to catch is a percentage leaking down to WASAPI, and a fake
    that quietly converted would hide it.
    """

    def __init__(
        self,
        *,
        level: int = 50,
        muted: bool = False,
        mic_level: int = 70,
        mic_muted: bool = False,
        sessions: list[AudioSession] | None = None,
        device_name: str = "Динамики (USB Audio Device)",
    ) -> None:
        self.levels: dict[DeviceKind, int] = {
            DeviceKind.OUTPUT: level,
            DeviceKind.INPUT: mic_level,
        }
        self.mutes: dict[DeviceKind, bool] = {
            DeviceKind.OUTPUT: muted,
            DeviceKind.INPUT: mic_muted,
        }
        self.device_name = device_name
        self.sessions = list(sessions or ())
        self.volume_calls: list[tuple[float, DeviceKind, str]] = []
        self.mute_calls: list[tuple[bool, DeviceKind, str]] = []
        self.session_volume_calls: list[tuple[int, float]] = []
        self.session_mute_calls: list[tuple[int, bool]] = []
        self.reads = 0

    def get_master_volume(
        self,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> VolumeState:
        self.reads += 1
        return VolumeState(
            level=self.levels[kind],
            muted=self.mutes[kind],
            kind=kind,
            device=self.device_name,
        )

    def set_master_volume(
        self,
        scalar: float,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        self.volume_calls.append((scalar, kind, device_id))
        self.levels[kind] = round(scalar * 100)

    def set_master_mute(
        self,
        muted: bool,
        kind: DeviceKind = DeviceKind.OUTPUT,
        device_id: str = "",
    ) -> None:
        self.mute_calls.append((muted, kind, device_id))
        self.mutes[kind] = muted

    def list_sessions(self) -> list[AudioSession]:
        return list(self.sessions)

    def set_session_volume(self, pid: int, scalar: float) -> None:
        self.session_volume_calls.append((pid, scalar))

    def set_session_mute(self, pid: int, muted: bool) -> None:
        self.session_mute_calls.append((pid, muted))


class FakeDevices:
    """A :class:`DeviceBackend` over a fixed list, remembering what was switched."""

    def __init__(self, devices: list[AudioDevice] | None = None) -> None:
        self.devices = list(devices if devices is not None else _default_devices())
        self.switched: list[tuple[str, DeviceKind]] = []
        self.fail_switch: Exception | None = None

    def list_devices(self, kind: DeviceKind = DeviceKind.OUTPUT) -> list[AudioDevice]:
        found = [device for device in self.devices if device.kind is kind]
        found.sort(key=lambda item: (not item.is_default, not item.usable, item.name.casefold()))
        return found

    def default_device(self, kind: DeviceKind = DeviceKind.OUTPUT) -> AudioDevice:
        for device in self.list_devices(kind):
            if device.is_default:
                return device
        raise DeviceUnavailable(f"no default {kind} endpoint", user_message=kind.missing_ru)

    def set_default(self, device_id: str, kind: DeviceKind = DeviceKind.OUTPUT) -> None:
        if self.fail_switch is not None:
            raise self.fail_switch
        self.switched.append((device_id, kind))
        self.devices = [
            AudioDevice(
                device_id=device.device_id,
                name=device.name,
                kind=device.kind,
                state=device.state,
                is_default=(
                    device.device_id == device_id if device.kind is kind else device.is_default
                ),
            )
            for device in self.devices
        ]


def _default_devices() -> list[AudioDevice]:
    """A plausible machine: two working outputs, one dead, one microphone."""
    return [
        AudioDevice(
            device_id="{spk}",
            name="Динамики (USB Audio Device)",
            kind=DeviceKind.OUTPUT,
            state=DeviceState.ACTIVE,
            is_default=True,
        ),
        AudioDevice(
            device_id="{hdmi}",
            name="Цифровое аудио (HDMI)",
            kind=DeviceKind.OUTPUT,
            state=DeviceState.ACTIVE,
        ),
        AudioDevice(
            device_id="{gone}",
            name="Наушники (Realtek Audio)",
            kind=DeviceKind.OUTPUT,
            state=DeviceState.NOT_PRESENT,
        ),
        AudioDevice(
            device_id="{mic}",
            name="Микрофон (USB Audio Device)",
            kind=DeviceKind.INPUT,
            state=DeviceState.ACTIVE,
            is_default=True,
        ),
    ]


def _sessions() -> list[AudioSession]:
    """A busy mixer: two Chrome renderers, Spotify, and the system sounds."""
    return [
        AudioSession(pid=100, process="chrome", display_name="Google Chrome", level=80),
        AudioSession(
            pid=101,
            process="chrome",
            display_name="Google Chrome",
            level=80,
            active=False,
        ),
        AudioSession(pid=200, process="spotify", display_name="Spotify", level=55),
        AudioSession(
            pid=0,
            process="",
            display_name=SYSTEM_SOUNDS_NAME,
            level=100,
        ),
    ]


@pytest.fixture
def audio() -> Iterator[FakeAudio]:
    """Install a fake audio backend for one test and take it away after."""
    backend = FakeAudio(sessions=_sessions())
    set_audio_backend(backend)
    try:
        yield backend
    finally:
        set_audio_backend(None)


@pytest.fixture
def devices() -> Iterator[FakeDevices]:
    """Install a fake device backend for one test and take it away after."""
    backend = FakeDevices()
    set_device_backend(backend)
    try:
        yield backend
    finally:
        set_device_backend(None)


# --------------------------------------------------------------------------- #
# Percent ↔ scalar
# --------------------------------------------------------------------------- #


class TestConversion:
    """The one place that knows WASAPI counts to one and people count to a hundred."""

    @pytest.mark.parametrize(
        ("level", "expected"),
        [(0, 0.0), (1, 0.01), (50, 0.5), (60, 0.6), (100, 1.0)],
    )
    def test_percent_becomes_a_scalar(self, level: int, expected: float) -> None:
        assert percent_to_scalar(level) == pytest.approx(expected)

    @pytest.mark.parametrize(("level", "expected"), [(-40, 0), (0, 0), (100, 100), (250, 100)])
    def test_clamping_takes_the_nearest_end(self, level: int, expected: int) -> None:
        assert clamp_volume(level) == expected

    def test_out_of_range_percent_clamps_before_wasapi_sees_it(self) -> None:
        """The last gate: a scalar above one is a ``COMError`` and not a loud sound."""
        assert percent_to_scalar(140) == pytest.approx(1.0)
        assert percent_to_scalar(-5) == pytest.approx(0.0)

    def test_a_driver_scalar_rounds_back_to_the_percent_it_was_set_from(self) -> None:
        """``0.6`` comes back as ``0.6000000238418579`` and must still read 60."""
        assert scalar_to_percent(0.6000000238418579) == 60
        assert scalar_to_percent(0.006) == 1
        assert scalar_to_percent(0.0) == 0

    def test_the_round_trip_is_stable_across_the_whole_scale(self) -> None:
        for level in range(0, 101):
            assert scalar_to_percent(percent_to_scalar(level)) == level


class TestVolumeStateShape:
    """What a setter reports back, and what lands in the audit trail."""

    def test_with_level_clamps_and_keeps_everything_else(self) -> None:
        state = VolumeState(level=40, muted=True, kind=DeviceKind.INPUT, device="Микрофон")
        moved = state.with_level(180)
        assert moved.level == 100
        assert moved.muted is True
        assert moved.kind is DeviceKind.INPUT
        assert moved.device == "Микрофон"

    def test_with_mute_leaves_the_level_alone(self) -> None:
        state = VolumeState(level=40).with_mute(True)
        assert (state.level, state.muted) == (40, True)

    def test_spoken_ru_says_muted_rather_than_a_number(self) -> None:
        assert VolumeState(level=40, muted=True).spoken_ru == "без звука"
        assert VolumeState(level=40).spoken_ru == "громкость 40"

    def test_as_dict_is_json_ready(self) -> None:
        payload = VolumeState(level=30, muted=True, kind=DeviceKind.INPUT, device="Мик").as_dict()
        assert payload == {
            "level": 30,
            "muted": True,
            "kind": "input",
            "device": "Мик",
        }


class TestAudioSessionShape:
    """The mixer's own idea of a name, cleaned up enough to say out loud."""

    def test_a_resource_string_display_name_is_not_a_name(self) -> None:
        """A name like ``@%SystemRoot%…,-202`` is an id: saying it is worse than not."""
        session = AudioSession(pid=0, display_name=SYSTEM_SOUNDS_NAME)
        assert session.spoken_name == ""
        assert session.label == "системные звуки"

    def test_the_display_name_wins_when_there_is_a_real_one(self) -> None:
        session = AudioSession(pid=100, process="chrome", display_name="Google Chrome")
        assert session.label == "Google Chrome"

    def test_the_stem_carries_a_session_with_no_display_name(self) -> None:
        assert AudioSession(pid=100, process="spotify").label == "spotify"

    def test_as_dict_is_json_ready(self) -> None:
        payload = AudioSession(pid=7, process="vlc", level=30, muted=True, active=False).as_dict()
        assert payload == {
            "pid": 7,
            "process": "vlc",
            "display_name": "",
            "level": 30,
            "muted": True,
            "active": False,
        }


# --------------------------------------------------------------------------- #
# Which session did they mean
# --------------------------------------------------------------------------- #


class TestSessionMatching:
    """Five gradations over the mixer, folded the way the ``{app}`` resolver folds."""

    def test_the_process_stem_wins_over_the_marketing_name(self) -> None:
        """«chrome» is what the program is called; «Google Chrome» is what it says."""
        sessions = [
            AudioSession(pid=1, process="steam", display_name="Chrome Remote Desktop"),
            AudioSession(pid=2, process="chrome", display_name="Google Chrome"),
        ]
        assert [session.pid for session in match_sessions(sessions, "chrome")] == [2, 1]

    def test_the_display_name_matches_when_the_stem_does_not(self) -> None:
        sessions = [AudioSession(pid=3, process="msedge", display_name="Microsoft Edge")]
        assert match_sessions(sessions, "microsoft edge")[0].pid == 3

    def test_a_prefix_is_enough(self) -> None:
        sessions = [AudioSession(pid=4, process="spotify", display_name="Spotify")]
        assert match_sessions(sessions, "спот") == []
        assert match_sessions(sessions, "spot")[0].pid == 4

    def test_a_substring_is_enough(self) -> None:
        sessions = [AudioSession(pid=5, process="vlc", display_name="VLC media player")]
        assert match_sessions(sessions, "media")[0].pid == 5

    def test_every_word_of_a_multiword_query_has_to_land(self) -> None:
        sessions = [AudioSession(pid=6, process="teams", display_name="Microsoft Teams (work)")]
        assert match_sessions(sessions, "microsoft work")[0].pid == 6
        assert match_sessions(sessions, "microsoft outlook") == []

    def test_a_playing_session_comes_before_an_idle_one(self) -> None:
        """«Сделай хром тише» is about the tab that is making noise right now."""
        sessions = [
            AudioSession(pid=11, process="chrome", display_name="Google Chrome", active=False),
            AudioSession(pid=12, process="chrome", display_name="Google Chrome"),
        ]
        assert [session.pid for session in match_sessions(sessions, "chrome")] == [12, 11]

    def test_the_system_sounds_are_not_matched_by_a_stray_word(self) -> None:
        """Its display name is a DLL path, and «звук» must not reach into it."""
        sessions = [AudioSession(pid=0, display_name=SYSTEM_SOUNDS_NAME)]
        assert match_sessions(sessions, "audiosrv") == []
        assert match_sessions(sessions, "system32") == []

    def test_an_empty_query_matches_nothing(self) -> None:
        assert match_sessions(_sessions(), "   ") == []

    def test_a_russian_query_folds_the_same_way_both_sides_do(self) -> None:
        sessions = [AudioSession(pid=7, process="яндекс браузер", display_name="Яндекс Браузер")]
        assert match_sessions(sessions, "Яндекс браузер")[0].pid == 7


class TestFindSessions:
    """One program means every session it owns, and the search order is cheap first."""

    def test_all_of_a_programs_sessions_come_back(self) -> None:
        """Chrome answers with one session per renderer, and «потише» means all of them."""
        found = find_sessions(_sessions(), "chrome")
        assert sorted(session.pid for session in found) == [100, 101]

    def test_a_session_with_no_stem_is_grouped_by_pid_alone(self) -> None:
        """Grouping unnameable sessions by an empty stem would collect all of them."""
        sessions = [
            AudioSession(pid=0, display_name="Системные звуки"),
            AudioSession(pid=9, display_name="Что-то ещё"),
        ]
        found = find_sessions(sessions, "системные звуки")
        assert [session.pid for session in found] == [0]

    def test_nothing_matching_says_so_in_russian(self) -> None:
        with pytest.raises(SessionNotFound) as excinfo:
            find_sessions(_sessions(), "телеграм")
        assert excinfo.value.user_message == "Не нашла приложение «телеграм» в микшере."

    def test_the_app_index_is_not_consulted_when_the_mixer_already_knows(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``get_app_index().resolve()`` can block on a registry scan. «Хром тише» must not."""
        from ayris.actions.system import audio as audio_module

        def _explode(_query: str) -> str:
            raise AssertionError("индекс программ не должен был понадобиться")

        monkeypatch.setattr(audio_module, "_resolved_stem", _explode)
        assert len(find_sessions(_sessions(), "chrome")) == 2

    def test_the_app_index_resolves_a_name_the_mixer_spells_differently(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The mixer says ``msedge`` and the user said «эдж»: only the index knows."""
        from ayris.actions.system import audio as audio_module

        monkeypatch.setattr(audio_module, "_resolved_stem", lambda _query: "msedge")
        sessions = [AudioSession(pid=42, process="msedge", display_name="")]
        assert find_sessions(sessions, "эдж")[0].pid == 42


# --------------------------------------------------------------------------- #
# The master-volume actions
# --------------------------------------------------------------------------- #


class TestSetVolumeAction:
    """An absolute level: the scalar that reached WASAPI, and what «отмени» restores."""

    def test_the_backend_is_handed_a_scalar_and_not_a_percentage(self, audio: FakeAudio) -> None:
        """The whole point of the conversion living in one place."""
        result = SetVolume().run(SetVolume.Params(level=60))
        assert audio.volume_calls == [(0.6, DeviceKind.OUTPUT, "")]
        assert result.ok
        assert result.message_ru == "Громкость 60%."

    @pytest.mark.parametrize(("level", "scalar"), [(0, 0.0), (100, 1.0)])
    def test_the_ends_of_the_scale_are_ordinary_values(
        self,
        audio: FakeAudio,
        level: int,
        scalar: float,
    ) -> None:
        SetVolume().run(SetVolume.Params(level=level))
        assert audio.volume_calls[0][0] == pytest.approx(scalar)

    def test_a_level_outside_the_scale_is_refused_by_the_parameters(self) -> None:
        """Clamping is for «громче» at 95, not for a macro that literally asks for 120."""
        with pytest.raises(ValidationError):
            SetVolume.Params(level=120)
        with pytest.raises(ValidationError):
            SetVolume.Params(level=-1)

    def test_setting_a_level_lifts_mute(self, audio: FakeAudio) -> None:
        """«Сделай громкость 40» to a muted machine means «и включи звук»."""
        audio.mutes[DeviceKind.OUTPUT] = True
        SetVolume().run(SetVolume.Params(level=40))
        assert audio.volume_calls == [(0.4, DeviceKind.OUTPUT, "")]
        assert audio.mute_calls == [(False, DeviceKind.OUTPUT, "")]

    def test_asking_for_the_level_it_is_already_at_touches_nothing(
        self,
        audio: FakeAudio,
    ) -> None:
        result = SetVolume().run(SetVolume.Params(level=50))
        assert audio.volume_calls == []
        assert result.message_ru == "Громкость уже 50%."
        assert result.undo_token is None

    def test_the_undo_token_remembers_the_level_and_the_mute(self, audio: FakeAudio) -> None:
        audio.mutes[DeviceKind.OUTPUT] = True
        result = SetVolume().run(SetVolume.Params(level=90))
        assert result.undoable
        assert result.undo_token == "50|1|"

    def test_undo_puts_both_of_them_back(self, audio: FakeAudio) -> None:
        result = SetVolume().run(SetVolume.Params(level=90))
        assert result.undo_token is not None
        audio.volume_calls.clear()
        undone = SetVolume().undo(result.undo_token)
        assert audio.volume_calls == [(0.5, DeviceKind.OUTPUT, "")]
        assert undone.message_ru == "Вернула 50%."

    def test_a_token_from_somewhere_else_is_refused_in_russian(self, audio: FakeAudio) -> None:
        from ayris.core.errors import ActionError

        with pytest.raises(ActionError) as excinfo:
            SetVolume().undo("не-токен")
        assert excinfo.value.user_message == "Не помню, какая громкость была до этого."
        assert audio.volume_calls == []

    def test_a_named_device_is_resolved_and_passed_down(
        self,
        audio: FakeAudio,
        devices: FakeDevices,
    ) -> None:
        SetVolume().run(SetVolume.Params(level=30, device="hdmi"))
        assert audio.volume_calls == [(0.3, DeviceKind.OUTPUT, "{hdmi}")]

    def test_an_unknown_device_name_is_refused_before_anything_moves(
        self,
        audio: FakeAudio,
        devices: FakeDevices,
    ) -> None:
        with pytest.raises(DeviceNotFound):
            SetVolume().run(SetVolume.Params(level=30, device="телевизор"))
        assert audio.volume_calls == []


class TestAdjustVolumeAction:
    """«Громче» and «на десять процентов тише», and both stops of the slider."""

    def test_the_default_step_comes_from_the_config(self, audio: FakeAudio) -> None:
        """Ten percent by default, applied to the scale and not to the current value."""
        result = AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert audio.volume_calls == [(0.6, DeviceKind.OUTPUT, "")]
        assert result.message_ru == "Громкость 60%."

    def test_down_goes_the_other_way(self, audio: FakeAudio) -> None:
        AdjustVolume().run(AdjustVolume.Params(direction="down"))
        assert audio.volume_calls == [(0.4, DeviceKind.OUTPUT, "")]

    def test_an_explicit_amount_replaces_the_step(self, audio: FakeAudio) -> None:
        AdjustVolume().run(AdjustVolume.Params(direction="up", amount=25))
        assert audio.volume_calls == [(0.75, DeviceKind.OUTPUT, "")]

    def test_a_percent_is_a_percent_of_the_scale(self, audio: FakeAudio) -> None:
        """«На 10% тише» at 20 gives 10, not 18 — that is what a 0..100 slider means."""
        audio.levels[DeviceKind.OUTPUT] = 20
        AdjustVolume().run(AdjustVolume.Params(direction="down", amount=10))
        assert audio.volume_calls == [(0.1, DeviceKind.OUTPUT, "")]

    def test_the_step_is_configurable(
        self,
        audio: FakeAudio,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ayris.core import config as config_module

        monkeypatch.setenv("AYRIS_ACTIONS__AUDIO__VOLUME_STEP", "5")
        config_module.reset_config_manager()
        AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert audio.volume_calls == [(0.55, DeviceKind.OUTPUT, "")]

    def test_louder_at_the_top_clamps_instead_of_failing(self, audio: FakeAudio) -> None:
        """«Громче» at 95 asks for as loud as it goes, and 105 would be refused."""
        audio.levels[DeviceKind.OUTPUT] = 95
        AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert audio.volume_calls == [(1.0, DeviceKind.OUTPUT, "")]

    def test_quieter_at_the_bottom_clamps_too(self, audio: FakeAudio) -> None:
        audio.levels[DeviceKind.OUTPUT] = 5
        AdjustVolume().run(AdjustVolume.Params(direction="down"))
        assert audio.volume_calls == [(0.0, DeviceKind.OUTPUT, "")]

    def test_already_at_the_top_says_so_and_touches_nothing(self, audio: FakeAudio) -> None:
        audio.levels[DeviceKind.OUTPUT] = 100
        result = AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert audio.volume_calls == []
        assert result.message_ru == "Громкость уже на максимуме."

    def test_already_at_the_bottom_says_so_too(self, audio: FakeAudio) -> None:
        audio.levels[DeviceKind.OUTPUT] = 0
        result = AdjustVolume().run(AdjustVolume.Params(direction="down"))
        assert audio.volume_calls == []
        assert result.message_ru == "Тише уже некуда."

    def test_louder_on_a_muted_machine_unmutes_it(self, audio: FakeAudio) -> None:
        audio.mutes[DeviceKind.OUTPUT] = True
        AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert audio.mute_calls == [(False, DeviceKind.OUTPUT, "")]

    def test_undo_restores_the_level_it_moved_from(self, audio: FakeAudio) -> None:
        result = AdjustVolume().run(AdjustVolume.Params(direction="up"))
        assert result.undo_token == "50|0|"
        audio.volume_calls.clear()
        AdjustVolume().undo(result.undo_token)
        assert audio.volume_calls == [(0.5, DeviceKind.OUTPUT, "")]

    def test_a_zero_amount_is_refused(self) -> None:
        """«Сделай громче на ноль процентов» is a broken macro, not a request."""
        with pytest.raises(ValidationError):
            AdjustVolume.Params(direction="up", amount=0)


class TestMuteToggleAction:
    """Three modes over two directions, each with its own phrasing."""

    def test_toggle_flips_what_is_there(self, audio: FakeAudio) -> None:
        result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.TOGGLE))
        assert audio.mute_calls == [(True, DeviceKind.OUTPUT, "")]
        assert result.message_ru == "Выключила звук."

    def test_toggle_the_other_way_reports_the_level_it_came_back_to(
        self,
        audio: FakeAudio,
    ) -> None:
        """After «включи звук» the level matters: a machine at 5% is still silent."""
        audio.mutes[DeviceKind.OUTPUT] = True
        result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.TOGGLE))
        assert audio.mute_calls == [(False, DeviceKind.OUTPUT, "")]
        assert result.message_ru == "Включила звук, громкость 50%."

    def test_on_is_idempotent_and_says_so(self, audio: FakeAudio) -> None:
        audio.mutes[DeviceKind.OUTPUT] = True
        result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.ON))
        assert audio.mute_calls == []
        assert result.message_ru == "Звук и так выключен."

    def test_off_is_idempotent_too(self, audio: FakeAudio) -> None:
        result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.OFF))
        assert audio.mute_calls == []
        assert result.message_ru == "Звук и так включён."

    def test_the_microphone_gets_its_own_words(self, audio: FakeAudio) -> None:
        result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.ON, kind=DeviceKind.INPUT))
        assert audio.mute_calls == [(True, DeviceKind.INPUT, "")]
        assert result.message_ru == "Выключила микрофон."

    def test_the_default_mode_is_toggle(self) -> None:
        assert MuteToggle.Params().mode is MuteMode.TOGGLE

    @pytest.mark.parametrize(
        ("mode", "muted", "expected"),
        [
            (MuteMode.ON, False, True),
            (MuteMode.ON, True, True),
            (MuteMode.OFF, True, False),
            (MuteMode.TOGGLE, False, True),
            (MuteMode.TOGGLE, True, False),
        ],
    )
    def test_the_mode_decides_the_target_state(
        self,
        mode: MuteMode,
        muted: bool,
        expected: bool,
    ) -> None:
        assert mode.applies(muted) is expected


# --------------------------------------------------------------------------- #
# One program in the mixer, and the microphone
# --------------------------------------------------------------------------- #


class TestSetAppVolumeAction:
    """One program means every session it owns, and a scalar reaches each of them."""

    def test_every_session_of_the_program_is_moved(self, audio: FakeAudio) -> None:
        """Chrome owns one session per renderer, and «потише» is about the program."""
        result = SetAppVolume().run(SetAppVolume.Params(app="chrome", level=30))
        assert sorted(audio.session_volume_calls) == [(100, 0.3), (101, 0.3)]
        assert result.message_ru == "Громкость «Google Chrome» — 30%."

    def test_the_scalar_and_not_the_percentage_reaches_the_mixer(self, audio: FakeAudio) -> None:
        SetAppVolume().run(SetAppVolume.Params(app="spotify", level=60))
        assert audio.session_volume_calls == [(200, 0.6)]

    def test_muting_one_program_leaves_the_others_alone(self, audio: FakeAudio) -> None:
        result = SetAppVolume().run(SetAppVolume.Params(app="spotify", mute=True))
        assert audio.session_mute_calls == [(200, True)]
        assert audio.session_volume_calls == []
        assert result.message_ru == "Заглушила «Spotify»."

    def test_both_halves_of_one_request_are_reported_together(self, audio: FakeAudio) -> None:
        result = SetAppVolume().run(SetAppVolume.Params(app="spotify", level=40, mute=False))
        assert audio.session_volume_calls == [(200, 0.4)]
        assert audio.session_mute_calls == [(200, False)]
        assert result.message_ru == "Включила звук у «Spotify», громкость 40%."

    def test_a_call_that_would_change_nothing_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SetAppVolume.Params(app="chrome")

    def test_a_level_outside_the_scale_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SetAppVolume.Params(app="chrome", level=140)

    def test_an_unknown_program_is_refused_in_russian_before_anything_moves(
        self,
        audio: FakeAudio,
    ) -> None:
        with pytest.raises(SessionNotFound) as excinfo:
            SetAppVolume().run(SetAppVolume.Params(app="телеграм", level=20))
        assert excinfo.value.user_message == "Не нашла приложение «телеграм» в микшере."
        assert audio.session_volume_calls == []

    def test_the_result_carries_the_sessions_it_touched(self, audio: FakeAudio) -> None:
        result = SetAppVolume().run(SetAppVolume.Params(app="chrome", level=30, mute=True))
        assert result.value is not None
        assert [session.level for session in result.value] == [30, 30]
        assert all(session.muted for session in result.value)
        assert result.data["sessions"][0]["pid"] in {100, 101}

    def test_more_than_one_session_is_noted_in_the_detail(self, audio: FakeAudio) -> None:
        result = SetAppVolume().run(SetAppVolume.Params(app="chrome", level=30))
        assert "задето 2 сессии" in result.detail

    def test_one_session_needs_no_detail(self, audio: FakeAudio) -> None:
        result = SetAppVolume().run(SetAppVolume.Params(app="spotify", level=30))
        assert result.detail == ""

    def test_an_empty_program_name_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SetAppVolume.Params(app="", level=30)


class TestSetMicVolumeAction:
    """The capture endpoint, and the two things this action deliberately does not do."""

    def test_the_capture_endpoint_is_the_one_that_moves(self, audio: FakeAudio) -> None:
        result = SetMicVolume().run(SetMicVolume.Params(level=80))
        assert audio.volume_calls == [(0.8, DeviceKind.INPUT, "")]
        assert result.message_ru == "Чувствительность микрофона — 80%."

    def test_the_output_is_never_touched(self, audio: FakeAudio) -> None:
        SetMicVolume().run(SetMicVolume.Params(level=80, mute=False))
        assert all(kind is DeviceKind.INPUT for _, kind, _ in audio.volume_calls)
        assert all(kind is DeviceKind.INPUT for _, kind, _ in audio.mute_calls)

    def test_muting_the_device_says_it_is_the_device(self, audio: FakeAudio) -> None:
        result = SetMicVolume().run(SetMicVolume.Params(mute=True))
        assert audio.mute_calls == [(True, DeviceKind.INPUT, "")]
        assert result.message_ru == "Выключила микрофон."

    def test_unmuting_reports_the_sensitivity_it_came_back_to(self, audio: FakeAudio) -> None:
        audio.mutes[DeviceKind.INPUT] = True
        result = SetMicVolume().run(SetMicVolume.Params(mute=False))
        assert result.message_ru == "Включила микрофон, чувствительность 70%."

    def test_nothing_about_the_assistant_is_published(self, audio: FakeAudio) -> None:
        """A muted capture endpoint is not the assistant's «перестань слушать».

        Windows-wide capture mute and ``MicToggled`` are different things, and the
        task is explicit that this action must not conflate them: the bus sees the
        ordinary action lifecycle and nothing else.
        """
        from ayris.actions.registry import ActionRegistry
        from ayris.core.events import Event, EventBus, MicToggled

        seen: list[Event] = []
        bus = EventBus()
        unsubscribe = bus.subscribe(Event, seen.append, weak=False)
        registry = ActionRegistry(bus=bus)
        name = registry.add(SetMicVolume)
        try:
            registry.execute(name, {"mute": True})
            bus.drain(0)
        finally:
            unsubscribe()
        assert audio.mute_calls == [(True, DeviceKind.INPUT, "")]
        assert not any(isinstance(event, MicToggled) for event in seen)
        assert {type(event).__name__ for event in seen} == {"ActionStarted", "ActionFinished"}

    def test_a_named_microphone_is_resolved_and_named_back(
        self,
        audio: FakeAudio,
        devices: FakeDevices,
    ) -> None:
        result = SetMicVolume().run(SetMicVolume.Params(level=50, device="usb"))
        assert audio.volume_calls == [(0.5, DeviceKind.INPUT, "{mic}")]
        assert "«Динамики (USB Audio Device)»" in result.message_ru

    def test_a_call_that_would_change_nothing_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SetMicVolume.Params()


class TestVolumeStep:
    """Where one «громче» gets its size from."""

    def test_the_default_is_ten_percent(self) -> None:
        assert volume_step() == 10

    def test_the_configured_step_replaces_the_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ayris.core import config as config_module

        monkeypatch.setenv("AYRIS_ACTIONS__AUDIO__VOLUME_STEP", "25")
        config_module.reset_config_manager()
        assert volume_step() == 25

    def test_it_is_read_from_the_settings_every_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Settings are hot-reloadable: a step cached at import time is a stale step."""
        from ayris.core import config as config_module

        steps = iter([25, 15])
        base = config_module.get_settings()

        def one_more_step() -> config_module.Settings:
            audio = base.actions.audio.model_copy(update={"volume_step": next(steps)})
            actions = base.actions.model_copy(update={"audio": audio})
            return base.model_copy(update={"actions": actions})

        monkeypatch.setattr(config_module, "get_settings", one_more_step)
        assert volume_step() == 25
        assert volume_step() == 15

    @pytest.mark.parametrize("value", ["0", "51"])
    def test_an_impossible_step_is_refused_by_the_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        value: str,
    ) -> None:
        """A step of zero is a volume control that does nothing; 60 skips half the scale."""
        from ayris.core.config import AudioActionsConfig

        monkeypatch.setenv("AYRIS_ACTIONS__AUDIO__VOLUME_STEP", value)
        with pytest.raises(ValidationError):
            AudioActionsConfig(volume_step=int(value))


# --------------------------------------------------------------------------- #
# Devices: listing, matching, switching
# --------------------------------------------------------------------------- #


class TestDeviceMatching:
    """Four grades of match, on the Russian names a real driver produces."""

    def test_the_whole_name_wins(self) -> None:
        found = match_devices(_default_devices(), "Цифровое аудио (HDMI)")
        assert [device.device_id for device in found] == ["{hdmi}"]

    def test_a_prefix_matches(self) -> None:
        found = match_devices(_default_devices(), "динамики")
        assert [device.device_id for device in found] == ["{spk}"]

    def test_a_substring_matches(self) -> None:
        found = match_devices(_default_devices(), "hdmi")
        assert [device.device_id for device in found] == ["{hdmi}"]

    def test_words_in_any_order_match_what_a_substring_cannot(self) -> None:
        """«наушники realtek» is how a person says «Наушники (Realtek Audio)»."""
        found = match_devices(_default_devices(), "наушники realtek", usable_only=False)
        assert [device.device_id for device in found] == ["{gone}"]

    def test_the_case_of_cyrillic_does_not_matter(self) -> None:
        found = match_devices(_default_devices(), "ДИНАМИКИ")
        assert [device.device_id for device in found] == ["{spk}"]

    def test_extra_spaces_do_not_matter(self) -> None:
        found = match_devices(_default_devices(), "  цифровое   аудио  ")
        assert [device.device_id for device in found] == ["{hdmi}"]

    def test_an_exact_name_outranks_a_substring(self) -> None:
        devices = [
            AudioDevice(device_id="{long}", name="USB Audio Device (2)", kind=DeviceKind.OUTPUT),
            AudioDevice(device_id="{short}", name="USB Audio Device", kind=DeviceKind.OUTPUT),
        ]
        found = match_devices(devices, "usb audio device")
        assert [device.device_id for device in found] == ["{short}", "{long}"]

    def test_an_unplugged_device_is_left_out_by_default(self) -> None:
        assert match_devices(_default_devices(), "наушники") == []

    def test_an_empty_query_matches_nothing(self) -> None:
        assert match_devices(_default_devices(), "   ") == []

    def test_nothing_matching_is_refused_by_name(self) -> None:
        with pytest.raises(DeviceNotFound) as excinfo:
            find_device(_default_devices(), "телевизор")
        assert excinfo.value.user_message == "Не нашла устройство вывода с названием «телевизор»."

    def test_a_device_that_is_there_but_unplugged_says_so(self) -> None:
        """«Нашла, но оно отключено» is more use to the user than «не нашла»."""
        with pytest.raises(DeviceNotFound) as excinfo:
            find_device(_default_devices(), "наушники")
        assert excinfo.value.user_message == (
            "Нашла «Наушники (Realtek Audio)», но оно сейчас не подключено."
        )

    def test_the_direction_is_named_in_the_refusal(self) -> None:
        with pytest.raises(DeviceNotFound) as excinfo:
            find_device(_default_devices(), "телевизор", kind=DeviceKind.INPUT)
        assert "устройство ввода" in excinfo.value.user_message


class TestDeviceListing:
    """The default endpoint comes first, and the forty dead ones stay hidden."""

    def test_only_the_asked_direction_is_listed(self, devices: FakeDevices) -> None:
        assert [device.device_id for device in list_audio_devices()] == ["{spk}", "{hdmi}"]
        assert [device.device_id for device in list_audio_devices(DeviceKind.INPUT)] == ["{mic}"]

    def test_devices_that_are_not_present_are_hidden(self, devices: FakeDevices) -> None:
        """A machine remembers every monitor it ever had as a NotPresent endpoint."""
        listed = list_audio_devices()
        assert "{gone}" not in {device.device_id for device in listed}
        assert "{gone}" in {device.device_id for device in list_audio_devices(usable_only=False)}

    def test_the_limit_is_honoured(self, devices: FakeDevices) -> None:
        assert len(list_audio_devices(limit=1)) == 1
        assert list_audio_devices(limit=0) == []

    def test_the_default_device_is_read_per_direction(self, devices: FakeDevices) -> None:
        assert default_device().device_id == "{spk}"
        assert default_device(DeviceKind.INPUT).device_id == "{mic}"

    def test_a_machine_with_no_sound_card_is_refused_in_russian(
        self,
        devices: FakeDevices,
    ) -> None:
        """This is the CI runner, and it must answer rather than crash."""
        devices.devices = []
        assert list_audio_devices() == []
        with pytest.raises(DeviceUnavailable) as excinfo:
            default_device()
        assert excinfo.value.user_message == "Не нашла устройство вывода."
        assert isinstance(excinfo.value, ActionUnavailable)


class TestSetAudioDeviceAction:
    """Switching the default endpoint, and the two ways it can refuse."""

    def test_the_named_device_becomes_the_default(self, devices: FakeDevices) -> None:
        result = SetAudioDevice().run(SetAudioDevice.Params(device="hdmi"))
        assert devices.switched == [("{hdmi}", DeviceKind.OUTPUT)]
        assert result.message_ru == "Переключила на «Цифровое аудио (HDMI)»."
        assert result.value is not None
        assert result.value.is_default is True

    def test_the_microphone_is_switched_by_direction(self, devices: FakeDevices) -> None:
        devices.devices.append(
            AudioDevice(
                device_id="{mic2}",
                name="Микрофон (FIFINE)",
                kind=DeviceKind.INPUT,
            )
        )
        SetAudioDevice().run(SetAudioDevice.Params(device="fifine", kind=DeviceKind.INPUT))
        assert devices.switched == [("{mic2}", DeviceKind.INPUT)]

    def test_the_device_already_in_use_is_left_alone(self, devices: FakeDevices) -> None:
        result = SetAudioDevice().run(SetAudioDevice.Params(device="динамики"))
        assert devices.switched == []
        assert result.message_ru == "«Динамики (USB Audio Device)» и так выбрано."

    def test_an_unknown_name_is_refused_before_anything_switches(
        self,
        devices: FakeDevices,
    ) -> None:
        with pytest.raises(DeviceNotFound):
            SetAudioDevice().run(SetAudioDevice.Params(device="телевизор"))
        assert devices.switched == []

    def test_a_machine_with_no_devices_is_refused_in_russian(
        self,
        devices: FakeDevices,
    ) -> None:
        devices.devices = []
        with pytest.raises(DeviceUnavailable) as excinfo:
            SetAudioDevice().run(SetAudioDevice.Params(device="динамики"))
        assert excinfo.value.user_message == "Не нашла устройство вывода."

    def test_a_build_that_refuses_the_switch_is_reported_as_such(
        self,
        devices: FakeDevices,
    ) -> None:
        """``IPolicyConfig`` is undocumented; a Windows build may simply say no."""
        from ayris.actions.system.audio_devices import PolicyUnavailable

        devices.fail_switch = PolicyUnavailable("SetDefaultEndpoint failed")
        with pytest.raises(PolicyUnavailable) as excinfo:
            SetAudioDevice().run(SetAudioDevice.Params(device="hdmi"))
        assert excinfo.value.user_message
        assert isinstance(excinfo.value, ActionUnavailable)

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            SetAudioDevice.Params(device="")


# --------------------------------------------------------------------------- #
# The seams, the schemas, and the real thing
# --------------------------------------------------------------------------- #


class TestBackendSeams:
    """Both modules keep exactly one injection point, and it restores cleanly."""

    def test_the_installed_backend_is_the_one_used(self) -> None:
        fake = FakeAudio()
        set_audio_backend(fake)
        try:
            assert get_audio_backend() is fake
        finally:
            set_audio_backend(None)

    def test_the_installed_device_backend_is_the_one_used(self) -> None:
        fake = FakeDevices()
        set_device_backend(fake)
        try:
            assert get_device_backend() is fake
        finally:
            set_device_backend(None)

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has the real backend")
    def test_off_windows_volume_is_refused_in_russian(self) -> None:
        """The Linux CI job runs this: an honest refusal, not an ImportError."""
        set_audio_backend(None)
        with pytest.raises(ActionUnavailable) as excinfo:
            get_audio_backend()
        assert excinfo.value.user_message == "Управление громкостью работает только в Windows."

    @pytest.mark.skipif(sys.platform == "win32", reason="Windows has the real backend")
    def test_off_windows_devices_are_refused_in_russian(self) -> None:
        set_device_backend(None)
        with pytest.raises(ActionUnavailable) as excinfo:
            get_device_backend()
        assert excinfo.value.user_message

    @pytest.mark.skipif(sys.platform != "win32", reason="needs WASAPI")
    def test_on_windows_the_real_backend_is_the_default(self) -> None:
        from ayris.actions.system.audio import WasapiAudio
        from ayris.actions.system.audio_devices import WasapiDevices

        set_audio_backend(None)
        set_device_backend(None)
        assert isinstance(get_audio_backend(), WasapiAudio)
        assert isinstance(get_device_backend(), WasapiDevices)

    def test_every_fake_satisfies_the_protocol(self) -> None:
        """A fake that drifts from the Protocol makes the whole suite a lie."""
        from ayris.actions.system.audio import AudioBackend
        from ayris.actions.system.audio_devices import DeviceBackend

        audio: AudioBackend = FakeAudio()
        devices: DeviceBackend = FakeDevices()
        assert audio.get_master_volume().kind is DeviceKind.OUTPUT
        assert devices.list_devices(DeviceKind.OUTPUT)


class TestSchemas:
    """What the macro editor draws for the six actions."""

    def test_all_six_actions_are_registered(self) -> None:
        from ayris.actions.registry import ActionRegistry

        registry = ActionRegistry()
        registry.discover()
        assert {
            "SetVolume",
            "AdjustVolume",
            "MuteToggle",
            "SetAppVolume",
            "SetMicVolume",
            "SetAudioDevice",
        } <= set(registry.names)

    @pytest.mark.parametrize(
        "action_class",
        [SetVolume, AdjustVolume, MuteToggle, SetAppVolume, SetMicVolume, SetAudioDevice],
    )
    def test_every_action_describes_itself_in_russian(self, action_class: Any) -> None:
        schema = build_schema(action_class)
        assert schema.title_ru
        assert schema.description_ru
        assert schema.category_title_ru == "Звук и голос"
        assert all(field.label_ru for field in schema.fields)

    def test_a_volume_is_a_bounded_integer_in_percent(self) -> None:
        field = build_schema(SetVolume).field_by_name("level")
        assert field is not None
        assert field.kind is FieldKind.INTEGER
        assert (field.minimum, field.maximum) == (0, 100)
        assert field.unit_ru == "%"
        assert field.required is True

    def test_a_direction_is_a_choice_of_two(self) -> None:
        field = build_schema(AdjustVolume).field_by_name("direction")
        assert field is not None
        assert field.kind is FieldKind.CHOICE
        assert {choice.value for choice in field.choices} == {"up", "down"}
        assert all(choice.label_ru for choice in field.choices)

    def test_the_step_is_optional_so_the_config_can_decide(self) -> None:
        field = build_schema(AdjustVolume).field_by_name("amount")
        assert field is not None
        assert field.required is False
        assert field.default is None

    def test_muting_offers_three_modes(self) -> None:
        field = build_schema(MuteToggle).field_by_name("mode")
        assert field is not None
        assert {choice.value for choice in field.choices} == {"on", "off", "toggle"}
        assert field.default == "toggle"

    def test_undo_is_advertised_only_where_it_exists(self) -> None:
        assert build_schema(SetVolume).supports_undo is True
        assert build_schema(AdjustVolume).supports_undo is True
        assert build_schema(SetAppVolume).supports_undo is False
        assert build_schema(SetAudioDevice).supports_undo is False

    def test_switching_a_device_is_not_marked_dangerous(self) -> None:
        """Loud, not destructive: it is undone by naming the other device."""
        assert build_schema(SetAudioDevice).is_dangerous is False
        assert build_schema(SetAudioDevice).require_admin is False


@pytest.mark.hardware
@pytest.mark.skipif(sys.platform != "win32", reason="needs a real sound stack")
class TestLiveAudio:
    """Read-only against the machine's own WASAPI. Nothing here changes a setting.

    Marked ``hardware`` because that is what it is: a runner has no sound card at
    all, so «прочитал громкость» there means «наткнулся на отсутствие устройства».
    The pycaw code path itself *is* checked in CI — by
    :meth:`TestBackendSeams.test_on_windows_the_real_backend_is_the_default`,
    which needs the library imported and not a speaker plugged in.
    """

    def test_the_master_volume_reads_back_as_a_percentage(self) -> None:
        set_audio_backend(None)
        state = get_audio_backend().get_master_volume()
        assert 0 <= state.level <= 100
        assert isinstance(state.muted, bool)
        assert state.device

    def test_the_microphone_reads_back_too(self) -> None:
        set_audio_backend(None)
        state = get_audio_backend().get_master_volume(DeviceKind.INPUT)
        assert 0 <= state.level <= 100
        assert state.kind is DeviceKind.INPUT

    def test_the_mixer_lists_sessions_with_names_and_levels(self) -> None:
        set_audio_backend(None)
        for session in get_audio_backend().list_sessions():
            assert 0 <= session.level <= 100
            assert session.label
            assert session.pid >= 0

    def test_the_default_output_is_active_and_listed_first(self) -> None:
        set_device_backend(None)
        listed = list_audio_devices()
        assert listed, "the machine has speakers"
        assert listed[0].is_default is True
        assert all(device.state is DeviceState.ACTIVE for device in listed)
        assert default_device().device_id == listed[0].device_id

    def test_the_dead_endpoints_a_real_machine_remembers_are_filtered_out(self) -> None:
        """Forty NotPresent endpoints is normal; the user must not hear about them."""
        set_device_backend(None)
        raw = get_device_backend().list_devices(DeviceKind.OUTPUT)
        assert len(list_audio_devices(limit=1000)) <= len(raw)

    def test_matching_finds_the_default_device_by_its_own_name(self) -> None:
        set_device_backend(None)
        current = default_device()
        assert find_device(list_audio_devices(), current.name).device_id == current.device_id
