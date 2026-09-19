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
from ayris.core.config import ConfigManager, RestartScope, dump_settings
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
