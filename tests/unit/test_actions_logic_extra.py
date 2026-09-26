"""Логика системных действий без обращения к железу.

Здесь добираются ветки чистой логики, разбора параметров и обработки ошибок в
модулях, чьи основные пути уходят в WinAPI/COM/подпроцессы. Всё достигается через
подмену backend-шва (``set_*_backend``, ``get_netsh``/``get_radio``, ленивые
импорты ``comtypes``/``pycaw``), поэтому ни одна проверка не трогает Windows, звук,
мониторы или ``netsh``. То, что физически недостижимо без оборудования (реальный
WMI/DDC, живой WASAPI, WinRT, запуск ``netsh``), намеренно не покрывается.
"""

from __future__ import annotations

import subprocess
import sys
import types
import warnings
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest

from ayris.actions.system import audio, audio_devices, brightness, network
from ayris.actions.system.app_index import AppNotFound
from ayris.actions.system.audio import (
    AudioSession,
    MuteMode,
    MuteToggle,
    VolumeState,
)
from ayris.actions.system.audio_devices import (
    AudioDevice,
    DeviceKind,
    DeviceState,
    DeviceWatcher,
    WasapiDevices,
)
from ayris.actions.system.brightness import (
    AdjustBrightness,
    BrightnessCapability,
    BrightnessMethod,
    BrightnessState,
    ListMonitors,
    SetBrightness,
    WindowsBrightnessBackend,
)
from ayris.actions.system.network import (
    NetshResult,
    RadioKind,
    RadioMode,
    RadioState,
    RecordingRadio,
    ScriptedNetsh,
    SubprocessNetsh,
    UnavailableRadio,
    switch_radio,
)
from ayris.core.errors import ActionError, ActionUnavailable, SecretsError
from ayris.nlu.slot_types import Direction
from ayris.utils import admin, winapi
from ayris.utils.admin import ElevationDeclined, ElevationUnavailable
from ayris.utils.monitors import MonitorInfo

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_seams() -> Iterator[None]:
    """Каждая проверка возвращает все швы к «нет подмены» после себя."""
    yield
    brightness.set_brightness_backend(None)
    audio.set_audio_backend(None)
    audio_devices.set_device_backend(None)
    network.set_netsh(None)
    network.set_radio(None)


def _monitor(index: int) -> MonitorInfo:
    left = index * 1920
    return MonitorInfo(
        handle=index + 10,
        index=index,
        rect=winapi.Rect(left, 0, left + 1920, 1080),
        work=winapi.Rect(left, 0, left + 1920, 1040),
        device=rf"\\.\DISPLAY{index + 1}",
        name=f"Panel {index + 1}",
        device_id=rf"MONITOR\PANEL{index + 1}",
        primary=index == 0,
        external_index=index - 1,
    )


class _FakeBrightness:
    """Backend яркости без WMI/DDC: карта «handle → уровень» и набор неподдержанных."""

    def __init__(self, levels: dict[int, int], unsupported: set[int] | None = None) -> None:
        self.levels = dict(levels)
        self.unsupported = set(unsupported or set())
        self.writes: list[tuple[int, int]] = []

    def capabilities(self, monitor: MonitorInfo) -> BrightnessCapability:
        if monitor.handle in self.unsupported:
            return BrightnessCapability(False, None, reason="no ddc")
        return BrightnessCapability(True, BrightnessMethod.DDC, 0, 100)

    def read(self, monitor: MonitorInfo) -> BrightnessState:
        if monitor.handle in self.unsupported:
            raise WindowsBrightnessBackend._unsupported(monitor, "no ddc")
        return BrightnessState(monitor, self.levels[monitor.handle], BrightnessMethod.DDC)

    def write(self, monitor: MonitorInfo, percent: int) -> None:
        self.writes.append((monitor.handle, percent))
        self.levels[monitor.handle] = percent


# --------------------------------------------------------------------------- #
# brightness: чистая логика, разбор целей, токены, диспетчер backend           #
# --------------------------------------------------------------------------- #


def test_brightness_pure_helpers() -> None:
    assert brightness._normalise_device(r"MONITOR\Panel-1") == "monitorpanel1"
    assert brightness._normalise_device("") == ""
    assert WindowsBrightnessBackend._native(50, 0, 200) == 100
    assert WindowsBrightnessBackend._native(150, 10, 20) == 20  # клампится сверху
    assert WindowsBrightnessBackend._percent(50, 0, 200) == 25
    assert WindowsBrightnessBackend._percent(5, 10, 10) == 0  # maximum <= minimum
    error = WindowsBrightnessBackend._unsupported(_monitor(0), "no ddc")
    assert isinstance(error, ActionUnavailable)
    assert "не поддерживает управление яркостью" in error.user_message
    assert brightness._token([]) is None


def test_targets_reports_no_monitors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brightness, "list_monitors", lambda: [])
    with pytest.raises(ActionUnavailable, match="no attached monitors"):
        brightness._targets("")


def test_targets_reports_unknown_address(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brightness, "list_monitors", lambda: [_monitor(0)])
    with pytest.raises(ActionError, match="no monitor matches") as info:
        brightness._targets("Нет такого")
    assert "«Нет такого» не найден" in info.value.user_message


def test_set_brightness_noop_when_already_at_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monitors = [_monitor(0)]
    monkeypatch.setattr(brightness, "list_monitors", lambda: monitors)
    brightness.set_brightness_backend(_FakeBrightness({10: 60}))
    result = SetBrightness().run(SetBrightness.Params(monitor="1", level=60))
    assert result.ok is True
    assert result.message_ru == "Яркость уже 60%."
    assert result.undo_token is None  # нечего откатывать


def test_set_brightness_unexpected_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monitors = [_monitor(0)]
    monkeypatch.setattr(brightness, "list_monitors", lambda: monitors)
    backend = _FakeBrightness({10: 40})

    def boom(monitor: MonitorInfo) -> BrightnessState:
        raise RuntimeError("не COM и не ActionError")

    backend.read = boom  # type: ignore[method-assign]
    brightness.set_brightness_backend(backend)
    with pytest.raises(RuntimeError, match="не COM"):
        SetBrightness().run(SetBrightness.Params(monitor="1", level=90))


def test_set_brightness_group_all_unsupported_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    monitors = [_monitor(0), _monitor(1)]
    monkeypatch.setattr(brightness, "list_monitors", lambda: monitors)
    brightness.set_brightness_backend(_FakeBrightness({10: 50, 11: 50}, unsupported={10, 11}))
    result = SetBrightness().run(SetBrightness.Params(monitor="", level=70))
    assert result.ok is False
    assert result.message_ru == "Яркость изменена на 0 из 2 мониторов."
    assert result.undo_token is None


def test_adjust_group_mixed_then_undo(monkeypatch: pytest.MonkeyPatch) -> None:
    monitors = [_monitor(0), _monitor(1), _monitor(2)]
    monkeypatch.setattr(brightness, "list_monitors", lambda: monitors)
    monkeypatch.setattr(brightness, "brightness_step", lambda: 10)
    backend = _FakeBrightness({10: 50, 11: 5, 12: 40}, unsupported={10})
    brightness.set_brightness_backend(backend)
    action = AdjustBrightness()
    result = action.run(AdjustBrightness.Params(monitor="", direction=Direction.UP, amount=None))
    assert result.ok is True
    assert result.message_ru == "Яркость изменена на 2 из 3 мониторов."
    assert backend.levels == {10: 50, 11: 15, 12: 50}
    assert result.undo_token is not None
    restored = action.undo(result.undo_token)
    assert restored.ok is True
    assert backend.levels == {10: 50, 11: 5, 12: 40}


def test_undo_rejects_malformed_token() -> None:
    with pytest.raises(ActionError, match="malformed brightness undo token") as info:
        brightness._undo('{"v": 2, "monitors": []}')
    assert "какая яркость была" in info.value.user_message


def test_undo_skips_vanished_monitor_and_reports_total_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(brightness, "list_monitors", lambda: [])
    brightness.set_brightness_backend(_FakeBrightness({}))
    token = brightness._token([BrightnessState(_monitor(0), 30, BrightnessMethod.DDC)])
    assert token is not None
    result = brightness._undo(token)
    assert result.ok is False
    assert result.message_ru == "Не удалось вернуть прежнюю яркость."


def test_list_monitors_action_reports_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(brightness, "list_monitors", lambda: [])
    with pytest.raises(ActionUnavailable, match="no attached monitors"):
        ListMonitors().run(ListMonitors.Params())


def test_windows_backend_dispatch_wmi(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = WindowsBrightnessBackend()
    monitor = _monitor(0)
    cached = brightness._CachedCapability(
        BrightnessCapability(True, BrightnessMethod.WMI), wmi_instance="INST"
    )
    written: list[tuple[str, int]] = []
    monkeypatch.setattr(backend, "_probe", lambda _m: cached)
    monkeypatch.setattr(backend, "_wmi_read", lambda _inst: 42)
    monkeypatch.setattr(backend, "_wmi_write", lambda inst, level: written.append((inst, level)))
    monkeypatch.setattr(brightness, "list_monitors", lambda: [monitor])
    assert backend.capabilities(monitor).method is BrightnessMethod.WMI
    assert backend.read(monitor).level == 42
    backend.write(monitor, 130)  # клампится до 100
    assert written == [("INST", 100)]


def test_windows_backend_dispatch_ddc(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = WindowsBrightnessBackend()
    monitor = _monitor(1)
    cached = brightness._CachedCapability(
        BrightnessCapability(True, BrightnessMethod.DDC, 0, 100), physical_index=0
    )
    seen: list[int] = []
    monkeypatch.setattr(backend, "_probe", lambda _m: cached)
    monkeypatch.setattr(backend, "_ddc_read", lambda _m, _c: 77)
    monkeypatch.setattr(backend, "_ddc_write", lambda _m, _c, level: seen.append(level))
    monkeypatch.setattr(brightness, "list_monitors", lambda: [monitor])
    assert backend.read(monitor).level == 77
    backend.write(monitor, 33)
    assert seen == [33]


def test_windows_backend_dispatch_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = WindowsBrightnessBackend()
    monitor = _monitor(2)
    cached = brightness._CachedCapability(BrightnessCapability(False, None, reason="nope"))
    monkeypatch.setattr(backend, "_probe", lambda _m: cached)
    monkeypatch.setattr(brightness, "list_monitors", lambda: [monitor])
    with pytest.raises(ActionUnavailable, match="brightness unsupported"):
        backend.read(monitor)
    with pytest.raises(ActionUnavailable, match="brightness unsupported"):
        backend.write(monitor, 50)


# --------------------------------------------------------------------------- #
# audio: разрешение имён, чистые хелперы, WASAPI-шов через endpoint_volume      #
# --------------------------------------------------------------------------- #


class _FakeAudio:
    """Backend громкости: отдаёт заранее заданное состояние, пишет вызовы."""

    def __init__(self, state: VolumeState, *, read_error: Exception | None = None) -> None:
        self.state = state
        self.read_error = read_error
        self.volumes: list[tuple[float, DeviceKind, str]] = []
        self.mutes: list[tuple[bool, DeviceKind, str]] = []
        self.sessions: list[AudioSession] = []

    def get_master_volume(
        self, kind: DeviceKind = DeviceKind.OUTPUT, device_id: str = ""
    ) -> VolumeState:
        if self.read_error is not None:
            raise self.read_error
        return replace(self.state, kind=kind)

    def set_master_volume(
        self, scalar: float, kind: DeviceKind = DeviceKind.OUTPUT, device_id: str = ""
    ) -> None:
        self.volumes.append((scalar, kind, device_id))

    def set_master_mute(
        self, muted: bool, kind: DeviceKind = DeviceKind.OUTPUT, device_id: str = ""
    ) -> None:
        self.mutes.append((muted, kind, device_id))

    def list_sessions(self) -> list[AudioSession]:
        return list(self.sessions)

    def set_session_volume(self, pid: int, scalar: float) -> None:
        self.volumes.append((scalar, DeviceKind.OUTPUT, str(pid)))

    def set_session_mute(self, pid: int, muted: bool) -> None:
        self.mutes.append((muted, DeviceKind.OUTPUT, str(pid)))


class _FakeVolume:
    """``IAudioEndpointVolume``-заглушка: читает/пишет скаляр, может «отвалиться»."""

    def __init__(self, *, scalar: float = 0.0, muted: bool = False, fail: bool = False) -> None:
        self.scalar = scalar
        self.muted = muted
        self.fail = fail
        self.set_scalars: list[tuple[float, Any]] = []
        self.set_mutes: list[tuple[bool, Any]] = []

    def GetMasterVolumeLevelScalar(self) -> float:  # noqa: N802 - имя из COM
        if self.fail:
            raise OSError("endpoint gone")
        return self.scalar

    def GetMute(self) -> bool:  # noqa: N802 - имя из COM
        if self.fail:
            raise OSError("endpoint gone")
        return self.muted

    def SetMasterVolumeLevelScalar(self, scalar: float, ctx: Any) -> None:  # noqa: N802
        if self.fail:
            raise OSError("endpoint gone")
        self.set_scalars.append((scalar, ctx))

    def SetMute(self, muted: bool, ctx: Any) -> None:  # noqa: N802 - имя из COM
        if self.fail:
            raise OSError("endpoint gone")
        self.set_mutes.append((muted, ctx))


class _FakeSimple:
    """``ISimpleAudioVolume``-заглушка для одной сессии микшера."""

    def __init__(self, *, level: float = 0.0, muted: bool = False, fail: bool = False) -> None:
        self.level = level
        self.muted = muted
        self.fail = fail
        self.set_vol: list[tuple[float, Any]] = []
        self.set_mute: list[tuple[bool, Any]] = []

    def GetMasterVolume(self) -> float:  # noqa: N802 - имя из COM
        if self.fail:
            raise OSError("session gone")
        return self.level

    def GetMute(self) -> bool:  # noqa: N802 - имя из COM
        return self.muted

    def SetMasterVolume(self, value: float, ctx: Any) -> None:  # noqa: N802 - имя из COM
        self.set_vol.append((value, ctx))

    def SetMute(self, value: bool, ctx: Any) -> None:  # noqa: N802 - имя из COM
        self.set_mute.append((value, ctx))


class _FakeRaw:
    """``pycaw``-подобная сессия. ``simple=None`` — доступ к тому падает COM-ошибкой."""

    def __init__(
        self, *, pid: int = 0, display: str = "", state: int = 1, simple: _FakeSimple | None = None
    ) -> None:
        self.ProcessId = pid
        self.DisplayName = display
        self.State = state
        self._simple = simple

    @property
    def SimpleAudioVolume(self) -> _FakeSimple:  # noqa: N802 - имя из COM
        if self._simple is None:
            raise OSError("no simple audio volume")
        return self._simple


def test_pure_audio_helpers() -> None:
    assert audio._clamp_scalar(1.5) == 1.0
    assert audio._clamp_scalar(-0.2) == 0.0
    assert audio._clamp_scalar(0.5) == 0.5
    failed = audio._endpoint_failed(DeviceKind.OUTPUT, OSError("gone"))
    assert isinstance(failed, ActionError)
    assert "не отвечает" in failed.user_message
    assert audio._volume_said(VolumeState(level=0, muted=True)) == "Выключила звук."
    assert audio._volume_said(VolumeState(level=40, muted=False)) == "Громкость 40%."


def test_mixer_said_covers_every_branch() -> None:
    assert audio._mixer_said("Chrome", None, True) == "Заглушила «Chrome»."
    assert audio._mixer_said("Chrome", None, False) == "Включила звук у «Chrome»."
    assert audio._mixer_said("Chrome", 40, False) == "Включила звук у «Chrome», громкость 40%."
    assert audio._mixer_said("", 30, None) == "Громкость приложения — 30%."


def test_process_stem_prefers_winapi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(winapi, "process_image_name", lambda _pid: r"C:\Apps\Chrome.EXE")
    assert audio._process_stem(object(), 42) == "chrome"


def test_process_stem_falls_back_to_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(winapi, "process_image_name", lambda _pid: "")

    class _Named:
        class Process:
            @staticmethod
            def name() -> str:
                return "Spotify.exe"

    assert audio._process_stem(_Named(), 0) == "spotify"
    assert audio._process_stem(object(), 0) == ""  # ни winapi, ни psutil не назвали


def test_psutil_name_handles_missing_and_failing() -> None:
    assert audio._psutil_name(object()) == ""

    class _Boom:
        class Process:
            @staticmethod
            def name() -> str:
                raise RuntimeError("psutil упал")

    assert audio._psutil_name(_Boom()) == ""


def test_wasapi_get_master_volume_reads_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        audio, "endpoint_volume", lambda _kind, _device: (_FakeVolume(scalar=0.6), "Speakers")
    )
    state = audio.WasapiAudio().get_master_volume(DeviceKind.OUTPUT, "")
    assert state.level == 60
    assert state.muted is False
    assert state.device == "Speakers"
    monkeypatch.setattr(
        audio, "endpoint_volume", lambda _kind, _device: (_FakeVolume(fail=True), "Mic")
    )
    with pytest.raises(ActionError, match="endpoint volume call failed"):
        audio.WasapiAudio().get_master_volume(DeviceKind.INPUT, "")


def test_wasapi_set_master_volume_clamps_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = _FakeVolume()
    monkeypatch.setattr(audio, "endpoint_volume", lambda _kind, _device: (volume, "Speakers"))
    audio.WasapiAudio().set_master_volume(1.5)  # клампится до 1.0
    assert volume.set_scalars == [(1.0, None)]
    monkeypatch.setattr(
        audio, "endpoint_volume", lambda _kind, _device: (_FakeVolume(fail=True), "Speakers")
    )
    with pytest.raises(ActionError, match="endpoint volume call failed"):
        audio.WasapiAudio().set_master_volume(0.5)


def test_wasapi_set_master_mute_writes_and_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    volume = _FakeVolume()
    monkeypatch.setattr(audio, "endpoint_volume", lambda _kind, _device: (volume, "Speakers"))
    audio.WasapiAudio().set_master_mute(True)
    assert volume.set_mutes == [(True, None)]
    monkeypatch.setattr(
        audio, "endpoint_volume", lambda _kind, _device: (_FakeVolume(fail=True), "Speakers")
    )
    with pytest.raises(ActionError, match="endpoint volume call failed"):
        audio.WasapiAudio().set_master_mute(False)


def test_wasapi_list_sessions_skips_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    good = _FakeRaw(pid=0, display="Music", state=1, simple=_FakeSimple(level=0.5))
    broken = _FakeRaw(pid=7, display="Broken", state=1, simple=None)
    backend = audio.WasapiAudio()
    monkeypatch.setattr(backend, "_raw_sessions", lambda: [good, broken])
    sessions = backend.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].display_name == "Music"
    assert sessions[0].level == 50
    assert sessions[0].active is True
    assert sessions[0].process == ""  # pid 0 и без psutil


def test_wasapi_each_session_touches_only_matching_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    mine = _FakeSimple()
    other = _FakeSimple()
    backend = audio.WasapiAudio()
    raws = [
        _FakeRaw(pid=5, simple=mine),
        _FakeRaw(pid=9, simple=other),
        _FakeRaw(pid=5, simple=None),  # совпадает по pid, но том падает
    ]
    monkeypatch.setattr(backend, "_raw_sessions", lambda: raws)
    backend.set_session_volume(5, 1.5)  # клампится до 1.0
    assert mine.set_vol == [(1.0, None)]
    assert other.set_vol == []  # чужой pid пропущен
    backend.set_session_mute(5, True)
    assert mine.set_mute == [(True, None)]
    backend.set_session_volume(999, 0.5)  # ни одной сессии — просто debug-строка


def test_raw_sessions_reads_and_reports_manager_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "initialize_com", lambda: None)

    class _Utils:
        def GetAllSessions(self) -> list[str]:  # noqa: N802 - имя из pycaw
            return ["a", "b"]

    monkeypatch.setattr(audio, "audio_utilities", lambda: _Utils())
    assert audio.WasapiAudio()._raw_sessions() == ["a", "b"]

    class _BadUtils:
        def GetAllSessions(self) -> list[str]:  # noqa: N802 - имя из pycaw
            raise OSError("no session manager")

    monkeypatch.setattr(audio, "audio_utilities", lambda: _BadUtils())
    with pytest.raises(audio.MixerUnavailable, match="IAudioSessionManager2"):
        audio.WasapiAudio()._raw_sessions()


def test_find_sessions_raises_when_nothing_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "_resolved_stem", lambda _query: "")
    with pytest.raises(audio.SessionNotFound, match="no mixer session matches"):
        audio.find_sessions([], "хром")
    monkeypatch.setattr(audio, "_resolved_stem", lambda _query: "ghost")
    with pytest.raises(audio.SessionNotFound) as info:
        audio.find_sessions([], "хром")
    assert "«хром»" in info.value.user_message


def test_resolved_stem_swallows_index_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Index:
        def resolve(self, phrase: str) -> Any:
            raise AppNotFound("nothing installed")

    monkeypatch.setattr(audio, "get_app_index", lambda: _Index())
    assert audio._resolved_stem("что-то") == ""


def test_wasapi_supported_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert audio.WasapiAudio().supported() is False


def test_wasapi_supported_without_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pycaw.utils", None)
    assert audio.WasapiAudio().supported() is False


def test_read_state_falls_back_when_readback_fails() -> None:
    fallback = VolumeState(level=42, muted=True, kind=DeviceKind.OUTPUT)
    backend = _FakeAudio(VolumeState(), read_error=ActionError("readback boom"))
    assert audio._read_state(backend, DeviceKind.OUTPUT, "", fallback) is fallback


def test_mute_toggle_reports_microphone_already_muted() -> None:
    audio.set_audio_backend(_FakeAudio(VolumeState(level=0, muted=True, kind=DeviceKind.INPUT)))
    result = MuteToggle().run(MuteToggle.Params(mode=MuteMode.ON, kind=DeviceKind.INPUT))
    assert result.ok is True
    assert result.message_ru == "Микрофон и так выключен."


def test_get_audio_backend_requires_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    audio.set_audio_backend(None)
    monkeypatch.setattr(audio, "_real_backend", None)
    monkeypatch.setattr(audio.WasapiAudio, "supported", lambda _self: False)
    with pytest.raises(ActionUnavailable, match="require Windows with pycaw"):
        audio.get_audio_backend()


# --------------------------------------------------------------------------- #
# audio_devices: сопоставление, состояние, ленивые импорты, наблюдатель          #
# --------------------------------------------------------------------------- #


def test_match_devices_ignores_nameless_device() -> None:
    devices = [AudioDevice(device_id="id-1", name="", state=DeviceState.ACTIVE)]
    assert audio_devices.match_devices(devices, "динамики") == []


def test_state_of_maps_every_shape() -> None:
    class _Described:
        def __init__(self, state: Any) -> None:
            self.state = state

    class _EnumLike:
        value = 2

    assert audio_devices._state_of(_Described(1)) is DeviceState.ACTIVE
    assert audio_devices._state_of(_Described(16)) is DeviceState.UNKNOWN  # неизвестный бит
    assert audio_devices._state_of(_Described("bad")) is DeviceState.UNKNOWN  # int() падает
    assert audio_devices._state_of(_Described(_EnumLike())) is DeviceState.DISABLED
    assert DeviceState.from_wasapi(8) is DeviceState.UNPLUGGED


def test_initialize_com_reports_missing_comtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "comtypes", None)
    with pytest.raises(ActionUnavailable, match="comtypes is unavailable") as info:
        audio_devices.initialize_com()
    assert "только в Windows" in info.value.user_message


def test_audio_utilities_reports_missing_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pycaw.utils", None)
    with pytest.raises(ActionUnavailable, match="pycaw is unavailable"):
        audio_devices.audio_utilities()


def test_create_device_swallows_pycaw_warnings() -> None:
    class _Utils:
        def CreateDevice(self, endpoint: object) -> str:  # noqa: N802 - имя из pycaw
            warnings.warn("pycaw property 21 missing", stacklevel=2)
            return "device"

    assert audio_devices._create_device(_Utils(), object()) == "device"


def test_wasapi_devices_supported_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert WasapiDevices().supported() is False


def test_wasapi_devices_supported_without_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pycaw.utils", None)
    assert WasapiDevices().supported() is False


def test_device_watcher_start_is_idempotent() -> None:
    watcher = DeviceWatcher()
    watcher._client = object()
    assert watcher.start() is True
    assert watcher.watching is True


def test_device_watcher_start_without_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pycaw.callbacks", None)
    assert DeviceWatcher().start() is False


def test_device_watcher_start_handles_subscribe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("pycaw.callbacks")

    class MMNotificationClient:  # pycaw's base class, faked
        pass

    module.MMNotificationClient = MMNotificationClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pycaw.callbacks", module)

    def _boom() -> Any:
        raise OSError("no enumerator")

    monkeypatch.setattr(audio_devices, "device_enumerator", _boom)
    assert DeviceWatcher().start() is False


def test_device_watcher_stop_unregisters_and_early_returns() -> None:
    DeviceWatcher().stop()  # никогда не запускался — ранний выход

    calls: list[object] = []

    class _Enum:
        def UnregisterEndpointNotificationCallback(self, client: object) -> None:  # noqa: N802
            calls.append(client)

    watcher = DeviceWatcher()
    client = object()
    watcher._client = client
    watcher._enumerator = _Enum()
    watcher.stop()
    assert calls == [client]
    assert watcher.watching is False


def test_stop_device_watcher_stops_and_forgets(monkeypatch: pytest.MonkeyPatch) -> None:
    stopped: list[bool] = []

    class _FakeWatcher:
        watching = False

        def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr(audio_devices, "_watcher", _FakeWatcher())
    audio_devices.stop_device_watcher()
    assert stopped == [True]
    assert audio_devices._watcher is None


def test_get_device_backend_requires_pycaw(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_devices.set_device_backend(None)
    monkeypatch.setattr(audio_devices, "_real_backend", None)
    monkeypatch.setattr(WasapiDevices, "supported", lambda _self: False)
    with pytest.raises(ActionUnavailable, match="require Windows with pycaw"):
        audio_devices.get_device_backend()


# --------------------------------------------------------------------------- #
# network: чистая логика радио, кодировка консоли, netsh-шов, секреты Wi-Fi      #
# --------------------------------------------------------------------------- #


_INTERFACES = """
    Имя                 : Wi-Fi
    Описание            : Intel Dual Band Wireless
    Идентификатор GUID  : {11111111-2222-3333-4444-555555555555}
    Физический адрес    : aa:bb:cc:dd:ee:ff
    Состояние           : подключен
    SSID                : HomeNet
    BSSID               : 00:11:22:33:44:55
    Профиль             : HomeNet
"""


def test_radio_enum_titles_and_target() -> None:
    assert RadioKind.WIFI.title_ru == "Wi-Fi"
    assert RadioKind.BLUETOOTH.title_ru == "Bluetooth"
    assert RadioState.ON.title_ru == "включён"
    assert RadioState.OFF.title_ru == "выключен"
    assert RadioState.DISABLED.title_ru == "отключён в системе"
    assert RadioState.UNKNOWN.title_ru == "неизвестно"
    assert RadioMode.ON.title_ru == "Включить"
    assert RadioMode.OFF.title_ru == "Выключить"
    assert RadioMode.TOGGLE.title_ru == "Переключить"
    assert RadioMode.ON.target(RadioState.OFF) is True
    assert RadioMode.OFF.target(RadioState.ON) is False
    assert RadioMode.TOGGLE.target(RadioState.ON) is False
    assert RadioMode.TOGGLE.target(RadioState.OFF) is True
    assert RadioMode.TOGGLE.target(RadioState.UNKNOWN) is True


def test_console_encoding_prefers_console_then_oem_then_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(winapi, "console_output_codepage", lambda: 437)
    monkeypatch.setattr(winapi, "oem_codepage", lambda: 866)
    assert network.console_encoding() == "cp437"
    monkeypatch.setattr(winapi, "console_output_codepage", lambda: 0)
    assert network.console_encoding() == "cp866"
    monkeypatch.setattr(winapi, "oem_codepage", lambda: 0)
    assert network.console_encoding() == "utf-8"


def test_netsh_result_and_scripted_runner() -> None:
    assert NetshResult(0, "hi", "").ok is True
    assert NetshResult(1).ok is False
    assert NetshResult(0, "", "из stderr").text == "из stderr"  # stdout пуст → берём stderr
    runner = ScriptedNetsh({"wlan show profiles": "Профиль: HomeNet"})
    assert runner.run(("wlan", "show", "profiles")).stdout == "Профиль: HomeNet"
    empty = runner.run(("wlan", "show", "interfaces"))  # не заскриптовано → пусто, код 1
    assert empty.returncode == 1
    assert empty.stdout == ""
    assert runner.called("wlan show profiles") is True
    assert runner.called("wlan show interfaces") is True


def test_subprocess_netsh_timeout_reports_action_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _timeout(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="netsh", timeout=kwargs.get("timeout", 1.0))

    monkeypatch.setattr(subprocess, "run", _timeout)
    with pytest.raises(ActionError, match="timed out") as info:
        SubprocessNetsh().run(("wlan", "show", "interfaces"))
    assert "не ответила" in info.value.user_message


def test_subprocess_netsh_os_error_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise OSError("netsh missing")

    monkeypatch.setattr(subprocess, "run", _boom)
    with pytest.raises(ActionUnavailable, match="could not be started") as info:
        SubprocessNetsh().run(("wlan", "show", "interfaces"))
    assert "Не удалось запустить netsh" in info.value.user_message


def test_subprocess_netsh_decodes_with_console_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(winapi, "console_output_codepage", lambda: 437)
    monkeypatch.setattr(winapi, "oem_codepage", lambda: 0)

    def _ok(*args: Any, **kwargs: Any) -> Any:
        return types.SimpleNamespace(returncode=0, stdout=b"HomeNet", stderr=b"")

    monkeypatch.setattr(subprocess, "run", _ok)
    result = SubprocessNetsh().run(("wlan", "show", "profiles"))
    assert result.ok is True
    assert result.stdout == "HomeNet"


def test_radio_state_maps_by_last_component() -> None:
    assert network._radio_state("RadioState.On") is RadioState.ON
    assert network._radio_state("winrt.RadioState.Off") is RadioState.OFF
    assert network._radio_state("RadioState.Disabled") is RadioState.DISABLED
    assert network._radio_state("ns.On_") is RadioState.ON  # хвостовые «_» снимаются
    assert network._radio_state("RadioState.Whatever") is RadioState.UNKNOWN


def test_prop_reads_dict_and_swallows_failures() -> None:
    good = types.SimpleNamespace(properties={"k": "v"})
    assert network._prop(good, "k") == "v"
    assert network._prop(good, "missing") is None  # KeyError
    assert network._prop(object(), "k") is None  # нет .properties → AttributeError
    assert network._prop(types.SimpleNamespace(properties=5), "k") is None  # не индексируется


def test_unavailable_radio_answers_unknown_and_refuses() -> None:
    radio = UnavailableRadio()
    assert radio.available is False
    assert radio.state(RadioKind.WIFI) is RadioState.UNKNOWN
    assert radio.devices() == ()
    with pytest.raises(ActionUnavailable, match="no radio backend"):
        radio.switch(RadioKind.BLUETOOTH, on=True)


def test_switch_radio_disabled_refuses_without_touching(monkeypatch: pytest.MonkeyPatch) -> None:
    radio = RecordingRadio(states={RadioKind.WIFI: RadioState.DISABLED})
    network.set_radio(radio)
    result = switch_radio(RadioKind.WIFI, RadioMode.ON)
    assert result.ok is False
    assert "отключён в системе" in result.message_ru
    assert result.value is RadioState.DISABLED
    assert radio.switches == []  # ничего не переключали


def test_switch_radio_already_in_target_state_is_noop() -> None:
    radio = RecordingRadio(states={RadioKind.WIFI: RadioState.ON})
    network.set_radio(radio)
    result = switch_radio(RadioKind.WIFI, RadioMode.ON)
    assert result.ok is True
    assert result.message_ru == "Wi-Fi и так включён"
    assert result.data["changed"] is False
    assert radio.switches == []


def test_switch_radio_flips_via_backend() -> None:
    radio = RecordingRadio(states={RadioKind.BLUETOOTH: RadioState.OFF})
    network.set_radio(radio)
    result = switch_radio(RadioKind.BLUETOOTH, RadioMode.TOGGLE)
    assert result.ok is True
    assert result.value is RadioState.ON
    assert result.data["changed"] is True
    assert radio.switches == [(RadioKind.BLUETOOTH, True)]


def test_switch_radio_bluetooth_without_winrt_refuses() -> None:
    network.set_radio(UnavailableRadio())
    with pytest.raises(ActionUnavailable, match="bluetooth needs WinRT") as info:
        switch_radio(RadioKind.BLUETOOTH, RadioMode.ON)
    assert "Bluetooth нельзя переключить без WinRT" in info.value.user_message


def test_switch_radio_wifi_via_netsh_when_elevated(monkeypatch: pytest.MonkeyPatch) -> None:
    network.set_radio(UnavailableRadio())
    runner = ScriptedNetsh({"wlan show interfaces": _INTERFACES, "interface set interface": ""})
    network.set_netsh(runner)
    monkeypatch.setattr(admin, "is_elevated", lambda: True)
    result = switch_radio(RadioKind.WIFI, RadioMode.ON)
    assert result.ok is True
    assert result.value is RadioState.ON
    assert "включён" in result.message_ru
    assert runner.called("interface set interface") is True


def test_switch_wifi_via_netsh_elevated_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = ScriptedNetsh({"wlan show interfaces": _INTERFACES, "interface set interface": ""})
    runner.codes["interface set interface"] = 5
    network.set_netsh(runner)
    monkeypatch.setattr(admin, "is_elevated", lambda: True)
    with pytest.raises(ActionError, match="exited 5") as info:
        network._switch_wifi_via_netsh(on=True)
    assert "включить Wi-Fi" in info.value.user_message


def test_switch_wifi_via_netsh_requires_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    network.set_netsh(ScriptedNetsh())  # «wlan show interfaces» → пусто, адаптера нет
    monkeypatch.setattr(admin, "is_elevated", lambda: True)
    with pytest.raises(ActionUnavailable, match="no wireless interface"):
        network._switch_wifi_via_netsh(on=False)


def test_switch_wifi_via_netsh_elevation_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    network.set_netsh(ScriptedNetsh({"wlan show interfaces": _INTERFACES}))
    monkeypatch.setattr(admin, "is_elevated", lambda: False)

    monkeypatch.setattr(admin, "run_elevated", lambda *_a, **_k: types.SimpleNamespace(exit_code=0))
    assert network._switch_wifi_via_netsh(on=True) is RadioState.ON

    def _declined(*a: Any, **k: Any) -> Any:
        raise ElevationDeclined("пользователь нажал «Нет»")

    monkeypatch.setattr(admin, "run_elevated", _declined)
    with pytest.raises(ElevationDeclined):
        network._switch_wifi_via_netsh(on=True)

    def _unavailable(*a: Any, **k: Any) -> Any:
        raise ElevationUnavailable("нет UAC")

    monkeypatch.setattr(admin, "run_elevated", _unavailable)
    with pytest.raises(ActionUnavailable, match="cannot elevate"):
        network._switch_wifi_via_netsh(on=True)

    monkeypatch.setattr(admin, "run_elevated", lambda *_a, **_k: types.SimpleNamespace(exit_code=3))
    with pytest.raises(ActionError, match="elevated netsh exited 3"):
        network._switch_wifi_via_netsh(on=True)


def test_secret_ref_is_stable_and_prefixed() -> None:
    ref = network.secret_ref_for_ssid("HomeNet")
    assert ref.startswith("wifi.")
    assert ref == network.secret_ref_for_ssid("HomeNet")
    assert ref != network.secret_ref_for_ssid("OtherNet")


def test_store_and_read_password_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict[str, str] = {}

    class _Secrets:
        def save(self, ref: str, value: str) -> None:
            saved[ref] = value

        def get(self, ref: str) -> str | None:
            return saved.get(ref)

    monkeypatch.setattr(network, "get_secrets", lambda: _Secrets())
    ref = network._store_password("HomeNet", "hunter2")
    assert ref == network.secret_ref_for_ssid("HomeNet")
    assert network._stored_password("HomeNet") == "hunter2"
    assert network._stored_password("Unknown") == ""


def test_password_helpers_swallow_secrets_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _BrokenSecrets:
        def save(self, ref: str, value: str) -> None:
            raise SecretsError("хранилище недоступно")

        def get(self, ref: str) -> str | None:
            raise SecretsError("хранилище недоступно")

    monkeypatch.setattr(network, "get_secrets", lambda: _BrokenSecrets())
    assert network._store_password("HomeNet", "hunter2") == ""  # запись не удалась
    assert network._stored_password("HomeNet") == ""  # чтение не удалось


def test_require_interface_parses_first_adapter() -> None:
    network.set_netsh(ScriptedNetsh({"wlan show interfaces": _INTERFACES}))
    interface = network.require_interface()
    assert interface.name == "Wi-Fi"
    assert interface.ssid == "HomeNet"
    assert interface.guid.strip("{}").startswith("11111111")


def test_require_interface_refuses_when_none(monkeypatch: pytest.MonkeyPatch) -> None:
    network.set_netsh(ScriptedNetsh())  # netsh молчит про адаптеры
    with pytest.raises(ActionUnavailable, match="no wireless interface"):
        network.require_interface()
