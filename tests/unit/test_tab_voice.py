"""Tab «Голос» (task 49): binding, dynamic lists, restart plaques, no key leak.

Offscreen, with fake services and a fake supervisor, every widget closed in
teardown — an un-closed settings page keeps a config subscription and hangs CI.
The live parts (recognition, listening, wake detection, the real signal) are
checked by hand; here the checks are structural: sections build, engine and
slider choices reach the config, test buttons are disabled until a service is
provided, a vanished device does not crash the page, and a saved cloud key never
reaches the config file.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from PySide6.QtWidgets import QApplication

from ayris.audio.devices import DeviceDirection, RawDevice, list_devices
from ayris.audio.tts.player import PlaybackReason
from ayris.core.config import ConfigManager, RestartScope, dump_settings
from ayris.core.errors import TtsError
from ayris.core.secrets import SecretsStore
from ayris.gui.tabs import tab_spec
from ayris.gui.tabs.voice import VoiceServices, VoiceTab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import set_active_worker_control
from ayris.models.catalog import ModelCatalog, ModelEntry
from ayris.workers.manager import WorkerStatus, WorkerSummary

pytestmark = pytest.mark.unit

_SHA = "0" * 64


class _FakeKeyring:
    """An in-memory credential store, so no test touches Windows."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.entries.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.entries[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.entries.pop((service, username), None)


class _FakeEnumerator:
    """A device source with a fixed list, replaceable to simulate hot-plug."""

    def __init__(self, devices: tuple[RawDevice, ...]) -> None:
        self.devices = devices
        self.refreshed = 0

    def raw_devices(self) -> tuple[RawDevice, ...]:
        return self.devices

    def refresh(self) -> None:
        self.refreshed += 1


class _FakeControl:
    """Supervisor stand-in that records restarts instead of doing them."""

    def __init__(self) -> None:
        self.restarted: list[tuple[RestartScope, str]] = []

    def status(self) -> tuple[WorkerSummary, ...]:
        return ()

    def restart_scope(self, scope: RestartScope, settings_reason: str = "") -> int:
        self.restarted.append((scope, settings_reason))
        return 1


def _catalog() -> ModelCatalog:
    # model_validate, not the constructor: ModelEntry's validator rebuilds the
    # instance to fill derived targets, which pydantic only allows off __init__.
    stt = ModelEntry.model_validate(
        {
            "url": "https://example.com/gigaam.zip",
            "sha256": _SHA,
            "size_bytes": 1,
            "id": "gigaam-v3-ctc",
            "name": "GigaAM v3 CTC",
            "kind": "stt",
            "engine": "gigaam",
            "archive": "zip",
        }
    )
    tts = ModelEntry.model_validate(
        {
            "url": "https://example.com/irina.onnx",
            "sha256": _SHA,
            "size_bytes": 1,
            "id": "ru-irina",
            "name": "Ирина",
            "kind": "tts",
            "engine": "piper",
        }
    )
    return ModelCatalog(entries=(stt, tts))


def _microphone() -> RawDevice:
    return RawDevice(
        index=0,
        name="Микрофон",
        host_api="MME",
        max_input_channels=1,
        default_input=True,
    )


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def manager(tmp_path: Path) -> ConfigManager:
    result = ConfigManager(tmp_path / "config.toml")
    result.load()
    return result


@pytest.fixture(autouse=True)
def _clear_control() -> Iterator[None]:
    yield
    set_active_worker_control(None)


def _services(**overrides: object) -> VoiceServices:
    defaults: dict[str, object] = {
        "secrets": SecretsStore(backend=_FakeKeyring()),
        "devices": _FakeEnumerator((_microphone(),)),
        "catalog": _catalog(),
        "installed": lambda _kind: frozenset(),
    }
    defaults.update(overrides)
    return VoiceServices(**defaults)  # type: ignore[arg-type]


def _make_tab(manager: ConfigManager, theme: ThemeManager, **overrides: object) -> VoiceTab:
    tab = VoiceTab(manager, theme, None, services=_services(**overrides))
    tab.load_from_config()
    return tab


def test_tab_registers_itself_as_the_voice_factory() -> None:
    assert tab_spec("voice").factory is VoiceTab


def test_content_fits_without_a_horizontal_scrollbar(
    app: QApplication, manager: ConfigManager
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QScrollArea

    tab = _make_tab(manager, ThemeManager(app))
    scroll = tab.findChild(QScrollArea)
    assert scroll is not None
    assert scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    content = scroll.widget()
    # No card may push the page wider than the narrowest settings window, or a
    # combo gets clipped on the right (regression guard for the layout fix).
    assert content.minimumSizeHint().width() <= 600
    tab.dispose()
    tab.close()


def test_tab_assembles_and_loads_defaults(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    stt = tab._sections[0]
    assert stt._mode_combo.currentData() == "auto"
    assert stt._engine_combo.currentData() == "gigaam"
    assert stt._model_combo.currentData() == "gigaam-v3-ctc"
    tab.dispose()
    tab.close()


def test_engine_mode_and_sliders_reach_config(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    stt, tts, _wake, audio = tab._sections

    stt._mode_combo.setCurrentIndex(stt._mode_combo.findData("offline"))
    tts._speed.setValue(150)  # 1.5x
    audio._gain.setValue(200)  # 2.0x
    audio._vad.setValue(70)  # 0.70
    audio._denoise_combo.setCurrentIndex(audio._denoise_combo.findData("off"))

    tab.flush_pending()
    settings = manager.settings.voice
    assert settings.stt.mode == "offline"
    assert settings.tts.speed == pytest.approx(1.5)
    assert settings.audio_input.gain == pytest.approx(2.0)
    assert settings.audio_input.vad_threshold == pytest.approx(0.70)
    assert settings.audio_input.denoise == "off"
    tab.dispose()
    tab.close()


def test_offline_piper_sliders_reach_config(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tts = tab._sections[1]

    tts._expressiveness.setValue(80)  # 0.80
    tts._pause.setValue(250)  # 0.25 s

    tab.flush_pending()
    settings = manager.settings.voice.tts
    assert settings.expressiveness == pytest.approx(0.80)
    assert settings.sentence_pause == pytest.approx(0.25)
    tab.dispose()
    tab.close()


def test_expressiveness_card_follows_the_engine(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    tts = tab._sections[1]

    # Piper is the default engine and the only one that reads noise_scale.
    assert not tts._expr_card.isHidden()

    # Another offline engine (Silero) does not — the card goes away.
    tts._engine_combo.setCurrentIndex(tts._engine_combo.findData("silero"))
    assert tts._expr_card.isHidden()

    # A cloud engine has no expressiveness knob either.
    tts._engine_combo.setCurrentIndex(tts._engine_combo.findData("openai"))
    assert tts._expr_card.isHidden()

    # Back to Piper and it returns.
    tts._engine_combo.setCurrentIndex(tts._engine_combo.findData("piper"))
    assert not tts._expr_card.isHidden()
    tab.dispose()
    tab.close()


def test_vad_slider_moves_the_meter_threshold(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    audio = tab._sections[3]
    audio._vad.setValue(80)
    assert audio._meter.threshold() == pytest.approx(0.80)
    tab.dispose()
    tab.close()


def test_wake_phrase_added_and_removed(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    wake = tab._sections[2]
    before = len(manager.settings.voice.wake.phrases)

    wake._new_phrase.setText("слушай айрис")
    wake._add_phrase()
    phrases = [item.phrase for item in manager.settings.voice.wake.phrases]
    assert "слушай айрис" in phrases
    assert len(phrases) == before + 1

    wake._remove_phrase("слушай айрис")
    assert "слушай айрис" not in [item.phrase for item in manager.settings.voice.wake.phrases]
    tab.dispose()
    tab.close()


def test_wake_sensitivity_saved_per_variant(app: QApplication, manager: ConfigManager) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    wake = tab._sections[2]
    wake._rows[0].sensitivity.setValue(90)
    wake._flush_live()
    assert manager.settings.voice.wake.phrases[0].sensitivity == pytest.approx(0.90)
    tab.dispose()
    tab.close()


def test_test_buttons_disabled_until_a_service_is_ready(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    assert not tab._sections[0]._test_button.isEnabled()
    assert not tab._sections[1]._listen_button.isEnabled()
    assert not tab._sections[3]._calibrate_button.isEnabled()
    tab.dispose()
    tab.close()


def test_test_buttons_enabled_when_services_provided(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(
        manager,
        ThemeManager(app),
        transcribe_once=lambda: ("привет", 120),
        speak_sample=lambda: None,
        audio_source=lambda: object(),
    )
    assert tab._sections[0]._test_button.isEnabled()
    assert tab._sections[1]._listen_button.isEnabled()
    assert tab._sections[3]._calibrate_button.isEnabled()
    tab.dispose()
    tab.close()


def test_calibrate_enabled_by_the_live_worker(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In the running app nobody injects audio_source; «Калибровать» then lights from
    # the live audio worker and records through its ring buffer, not by opening the
    # microphone a second time behind the worker that already owns it.
    monkeypatch.setattr("ayris.audio.worker_source.time.sleep", lambda *_a, **_k: None)

    class _AudioControl:
        def __init__(self) -> None:
            self.reads: list[float] = []

        def status(self) -> tuple[WorkerSummary, ...]:
            return ()

        def restart_scope(self, scope: RestartScope, settings_reason: str = "") -> int:
            return 0

        def is_ready(self, name: str) -> bool:
            return name == "audio"

        def call_sync(
            self, worker: str, method: str, params: dict[str, float] | None = None
        ) -> object:
            if method == "status":
                return {"sample_rate": 16000}
            self.reads.append((params or {})["ms"])
            return {"pcm": b"\x00\x00" * 1600, "sample_rate": 16000}

    control = _AudioControl()
    set_active_worker_control(control)

    tab = _make_tab(manager, ThemeManager(app))  # no audio_source injected
    audio = tab._sections[3]
    assert audio._calibrate_button.isEnabled()

    factory = audio._calibration_factory()
    assert factory is not None
    from ayris.audio.calibration import CalibrationReport, run_calibration

    report = run_calibration(factory(), base_gain=1.0)
    assert isinstance(report, CalibrationReport)
    # Тишина ~3000 мс, затем фраза ~5000 мс — два окна из буфера воркера.
    assert [round(ms) for ms in control.reads] == [3000, 5000]
    tab.dispose()
    tab.close()


def test_listen_falls_back_to_the_runtime_router(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No speak_sample is injected in the running app; «Прослушать» then speaks the
    # fixed sample through the live router the dispatcher registered. Its presence
    # enables the button, and pressing it previews once and waits off the UI thread.
    waited: list[float | None] = []

    class _Handle:
        reason = PlaybackReason.COMPLETED

        def wait(self, timeout: float | None = None) -> bool:
            waited.append(timeout)
            return True

    class _Router:
        def __init__(self) -> None:
            self.previews = 0

        def preview(self) -> _Handle:
            self.previews += 1
            return _Handle()

    router = _Router()
    monkeypatch.setattr("ayris.gui.tabs.voice_sections.tts.active_tts_router", lambda: router)

    tab = _make_tab(manager, ThemeManager(app))  # no speak_sample injected
    tts = tab._sections[1]
    assert tts._listen_button.isEnabled()

    service = tts._sample_service()
    assert service is not None
    service()  # call the resolved sample directly, skipping the worker thread
    assert router.previews == 1
    assert waited == [pytest.approx(30.0)]
    tab.dispose()
    tab.close()


def test_listen_button_stays_live_after_a_preview(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # In the running app speak_sample is not injected, so the preview goes through
    # the runtime router. _finish_listen must re-enable the button from that same
    # source; gating it on services.speak_sample latched «Прослушать» off after the
    # first play — it answered once, then went dead.
    class _Router:
        def preview(self) -> object:
            return object()

    monkeypatch.setattr("ayris.gui.tabs.voice_sections.tts.active_tts_router", lambda: _Router())

    tab = _make_tab(manager, ThemeManager(app))  # no speak_sample injected
    tts = tab._sections[1]
    assert tts._listen_button.isEnabled()

    tts._listen_button.setEnabled(False)  # as _listen() does while it speaks
    tts._finish_listen()  # the worker finished; the button must come back
    assert tts._listen_button.isEnabled()
    tab.dispose()
    tab.close()


def test_preview_reports_a_failed_synthesis(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A missing or half-downloaded voice fails inside the player thread, where the
    # error only reaches the log; the handle then reports ERROR. The preview must
    # raise so «Прослушать» shows «Не удалось озвучить: …» instead of falling silent.
    class _Handle:
        reason = PlaybackReason.ERROR

        def wait(self, timeout: float | None = None) -> bool:
            return True

    class _Router:
        def preview(self) -> _Handle:
            return _Handle()

    monkeypatch.setattr("ayris.gui.tabs.voice_sections.tts.active_tts_router", lambda: _Router())
    tab = _make_tab(manager, ThemeManager(app))
    tts = tab._sections[1]
    with pytest.raises(TtsError) as excinfo:
        tts._preview_through_router()
    assert excinfo.value.user_message
    tab.dispose()
    tab.close()


def test_preview_times_out_when_synthesis_hangs(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # wait() returning False means the synth never answered within the timeout; the
    # preview must raise rather than report a success that never sounded.
    class _Handle:
        reason = PlaybackReason.COMPLETED

        def wait(self, timeout: float | None = None) -> bool:
            return False

    class _Router:
        def preview(self) -> _Handle:
            return _Handle()

    monkeypatch.setattr("ayris.gui.tabs.voice_sections.tts.active_tts_router", lambda: _Router())
    tab = _make_tab(manager, ThemeManager(app))
    tts = tab._sections[1]
    with pytest.raises(TtsError) as excinfo:
        tts._preview_through_router()
    assert excinfo.value.user_message
    tab.dispose()
    tab.close()


def test_preview_stays_quiet_when_cancelled(
    app: QApplication, manager: ConfigManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Clicking another voice cancels the running preview; a cancellation is not a
    # failure, so the preview returns without raising and «Прослушать» stays calm.
    class _Handle:
        reason = PlaybackReason.CANCELLED

        def wait(self, timeout: float | None = None) -> bool:
            return True

    class _Router:
        def preview(self) -> _Handle:
            return _Handle()

    monkeypatch.setattr("ayris.gui.tabs.voice_sections.tts.active_tts_router", lambda: _Router())
    tab = _make_tab(manager, ThemeManager(app))
    tts = tab._sections[1]
    tts._preview_through_router()  # must not raise
    tab.dispose()
    tab.close()


def test_missing_model_shows_a_warning(app: QApplication, manager: ConfigManager) -> None:
    manager.apply({"voice.stt.mode": "offline"})
    tab = _make_tab(manager, ThemeManager(app), installed=lambda _kind: frozenset({"other-model"}))
    assert not tab._sections[0]._model_notice.isHidden()
    tab.dispose()
    tab.close()


def test_vanished_device_does_not_crash(app: QApplication, manager: ConfigManager) -> None:
    manager.apply({"voice.audio_input.device": "input:mme:отключённый"})
    tab = _make_tab(manager, ThemeManager(app))
    audio = tab._sections[3]
    # The stored device is not in the fake enumerator, so it appears as unavailable.
    assert audio._device_combo.findData("input:mme:отключённый") >= 0
    assert audio._device_notice.text()

    # A rescan that returns nothing must also be survivable.
    tab.services.devices = _FakeEnumerator(())
    audio._enumerator_cache = tab.services.devices
    audio._rescan_devices()
    assert audio._device_combo.count() >= 1
    tab.dispose()
    tab.close()


def test_present_device_is_selected(app: QApplication, manager: ConfigManager) -> None:
    device = list_devices(_FakeEnumerator((_microphone(),)), DeviceDirection.INPUT)[0]
    manager.apply({"voice.audio_input.device": device.id})
    tab = _make_tab(manager, ThemeManager(app))
    assert tab._sections[3]._device_combo.currentData() == device.id
    assert not tab._sections[3]._device_notice.text()
    tab.dispose()
    tab.close()


def test_restart_plaque_appears_then_clears(app: QApplication, manager: ConfigManager) -> None:
    control = _FakeControl()
    set_active_worker_control(control)
    tab = _make_tab(manager, ThemeManager(app))
    bar = tab._restart_bars[RestartScope.AUDIO]
    assert bar.isHidden()

    manager.apply({"voice.audio_input.sample_rate": 48000})
    app.processEvents()
    assert not bar.isHidden()
    assert bar.button.isEnabled()

    bar.button.click()
    assert control.restarted == [(RestartScope.AUDIO, "перезапуск из настроек")]
    assert RestartScope.AUDIO not in manager.pending_restarts
    assert bar.isHidden()
    tab.dispose()
    tab.close()


def test_restart_button_disabled_without_a_supervisor(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    manager.apply({"voice.stt.offline_engine": "vosk"})
    app.processEvents()
    bar = tab._restart_bars[RestartScope.STT]
    assert not bar.isHidden()
    assert not bar.button.isEnabled()
    tab.dispose()
    tab.close()


def test_cloud_key_never_reaches_the_config(
    app: QApplication, manager: ConfigManager, tmp_path: Path
) -> None:
    keyring = _FakeKeyring()
    tab = _make_tab(manager, ThemeManager(app), secrets=SecretsStore(backend=keyring))
    secret = tab._sections[0]._secret

    key = "supersecretcloudkey1234567890"
    secret.edit.setText(key)
    secret._save()

    # Stored in the credential manager, and only there.
    assert keyring.entries[("Ayris", "yandex")] == key
    assert secret.edit.text() == ""
    assert manager.settings.voice.stt.credential_ref == "yandex"
    assert key not in str(dump_settings(manager.settings))
    assert key not in (tmp_path / "config.toml").read_text(encoding="utf-8")
    assert key not in repr(tab.services.secrets)
    tab.dispose()
    tab.close()


def test_stt_cloud_endpoint_and_model_shown_and_saved_for_openai(
    app: QApplication, manager: ConfigManager
) -> None:
    tab = _make_tab(manager, ThemeManager(app))
    stt = tab._sections[0]

    # Default provider is Yandex: the paste-your-own-service fields stay hidden.
    assert stt._endpoint_card.isHidden()
    assert stt._model_card.isHidden()

    # Choosing the OpenAI-compatible provider reveals them...
    stt._provider_combo.setCurrentIndex(stt._provider_combo.findData("openai"))
    assert not stt._endpoint_card.isHidden()
    assert not stt._model_card.isHidden()

    # ...and a custom endpoint and model typed there reach the config.
    stt._endpoint_edit.setText("https://groq.example/openai/v1/audio/transcriptions")
    stt._model_edit.setText("whisper-large-v3")
    tab.flush_pending()
    settings = manager.settings.voice.stt
    assert settings.online_provider == "openai"
    assert settings.online_endpoint == "https://groq.example/openai/v1/audio/transcriptions"
    assert settings.online_model == "whisper-large-v3"
    tab.dispose()
    tab.close()


def test_calibration_recommendation_applied(app: QApplication, manager: ConfigManager) -> None:
    from ayris.audio.calibration import Recommendation
    from ayris.audio.denoise import DenoiseMode

    tab = _make_tab(manager, ThemeManager(app))
    audio = tab._sections[3]
    audio.apply_recommendation(
        Recommendation(
            gain=2.5,
            vad_threshold=0.6,
            noise_floor_db=-42.0,
            silence_ms=900,
            denoise=DenoiseMode.SPECTRAL,
        )
    )
    settings = manager.settings.voice.audio_input
    assert settings.gain == pytest.approx(2.5)
    assert settings.vad_threshold == pytest.approx(0.6)
    assert settings.silence_ms == 900
    assert settings.denoise == "spectral"
    tab.dispose()
    tab.close()


def test_worker_status_summary_is_ignored_gracefully(
    app: QApplication, manager: ConfigManager
) -> None:
    control = _FakeControl()
    set_active_worker_control(control)
    tab = _make_tab(manager, ThemeManager(app))
    # A tab built with a live-but-empty supervisor still refreshes cleanly.
    tab._refresh_dynamic()
    assert WorkerStatus.READY  # sanity: the import is used
    tab.dispose()
    tab.close()
