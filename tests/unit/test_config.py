"""Task 02: settings schema, TOML persistence, recovery, hot reload, secrets.

The acceptance criteria of ``tasks/02_config.md`` map onto the classes below:
defaults on first run, boundary validation, save→load round trip, behaviour on a
corrupted file, detection of "requires restart" fields, and — the one that is a
security property rather than a convenience — an API key that appears neither in
``config.toml`` nor in a log record.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import tomlkit

from ayris.core.config import (
    RAM_LIMIT_CHOICES,
    SCHEMA_VERSION,
    ConfigManager,
    RestartScope,
    Settings,
    diff_settings,
    dump_settings,
    get_config_manager,
    get_settings,
    init_config,
    load_settings,
    restart_scope,
    save_settings,
)
from ayris.core.errors import ConfigError, SecretsError
from ayris.core.paths import AppPaths
from ayris.core.secrets import (
    KNOWN_SLOTS,
    KeyringBackend,
    SecretsStore,
    is_valid_ref,
    mask,
    reset_secrets,
    resolve_secrets,
)
from ayris.utils.logger import ROOT_LOGGER_NAME

pytestmark = pytest.mark.unit


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    """A settings file inside an otherwise empty profile directory."""
    root = tmp_path / "profile"
    root.mkdir(parents=True, exist_ok=True)
    return root / "config.toml"


@pytest.fixture
def manager(config_path: Path) -> Iterator[ConfigManager]:
    """A loaded manager whose file already exists on disk."""
    instance = ConfigManager(config_path)
    instance.load()
    yield instance
    instance.stop_watching()


@pytest.fixture
def ayris_log(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Capture records from the ``ayris`` logger tree.

    ``setup_logging`` sets ``propagate = False`` on that logger so nothing leaks
    into the interpreter root — which is exactly where ``caplog`` normally
    listens. Attaching its handler directly keeps these assertions independent of
    whether an earlier test happened to configure logging.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        logger.removeHandler(caplog.handler)
        logger.setLevel(previous)


def read_toml(path: Path) -> dict[str, Any]:
    """The file as plain Python data, with tomlkit's wrappers removed."""
    return tomlkit.parse(path.read_text(encoding="utf-8")).unwrap()


def messages(records: list[logging.LogRecord]) -> str:
    """Captured records rendered into one blob, for ``in`` assertions."""
    return "\n".join(record.getMessage() for record in records)


class TestDefaults:
    """«Готово когда»: the first run writes every section with sane defaults."""

    def test_construction_without_a_file_succeeds(self) -> None:
        settings = Settings()

        assert settings.schema_version == SCHEMA_VERSION
        assert settings.general.language == "ru"
        assert settings.general.theme == "dark_purple"
        assert settings.general.autostart is False
        assert settings.voice.stt.mode == "auto"
        assert settings.voice.audio_input.vad_threshold == pytest.approx(0.5)
        assert settings.performance.audio_priority == "high"
        assert settings.performance.ram_limit_mb in RAM_LIMIT_CHOICES

    def test_telemetry_and_audio_storage_are_off(self) -> None:
        """Section 17 of the specification: zero telemetry, no recordings on disk."""
        settings = Settings()

        assert settings.privacy.telemetry is False
        assert settings.privacy.store_audio is False

    def test_first_load_creates_the_file_with_every_section(self, config_path: Path) -> None:
        ConfigManager(config_path).load()

        assert config_path.is_file()
        written = read_toml(config_path)
        assert written["schema_version"] == SCHEMA_VERSION
        for section in (
            "general",
            "voice",
            "commands",
            "ai",
            "hotkeys",
            "overlay",
            "plugins",
            "privacy",
            "performance",
            "updates",
            "devtools",
        ):
            assert section in written, section
        assert {"stt", "tts", "wake", "audio_input"} <= set(written["voice"])

    def test_the_generated_file_explains_itself_in_russian(self, config_path: Path) -> None:
        ConfigManager(config_path).load()

        text = config_path.read_text(encoding="utf-8")
        assert "Настройки Ayris" in text
        assert "требуется перезапуск" in text
        assert "API-ключи здесь НЕ хранятся" in text

    def test_defaults_survive_a_round_trip(self, config_path: Path) -> None:
        original = Settings()
        save_settings(original, config_path)

        restored, dropped = load_settings(config_path)

        assert dropped == ()
        assert restored == original

    def test_a_missing_file_yields_defaults(self, tmp_path: Path) -> None:
        settings, dropped = load_settings(tmp_path / "nothing.toml")

        assert settings == Settings()
        assert dropped == ()


class TestValidation:
    """The ranges named in the task: TTS speed, wake sensitivity, volume, RAM."""

    @pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
    def test_tts_speed_accepts_the_range(self, speed: float) -> None:
        settings = Settings.model_validate({"voice": {"tts": {"speed": speed}}})

        assert settings.voice.tts.speed == pytest.approx(speed)

    @pytest.mark.parametrize("speed", [0.49, 2.01, -1.0, 100.0])
    def test_tts_speed_rejects_values_outside_the_range(self, speed: float) -> None:
        with pytest.raises(ValueError, match="speed"):
            Settings.model_validate({"voice": {"tts": {"speed": speed}}})

    @pytest.mark.parametrize("sensitivity", [0.0, 0.5, 1.0])
    def test_wake_sensitivity_accepts_the_range(self, sensitivity: float) -> None:
        settings = Settings.model_validate(
            {"voice": {"wake": {"phrases": [{"phrase": "айрис", "sensitivity": sensitivity}]}}}
        )

        assert settings.voice.wake.phrases[0].sensitivity == pytest.approx(sensitivity)

    @pytest.mark.parametrize("sensitivity", [-0.01, 1.01])
    def test_wake_sensitivity_rejects_values_outside_the_range(self, sensitivity: float) -> None:
        with pytest.raises(ValueError, match="sensitivity"):
            Settings.model_validate(
                {"voice": {"wake": {"phrases": [{"phrase": "айрис", "sensitivity": sensitivity}]}}}
            )

    @pytest.mark.parametrize("volume", [0, 50, 100])
    def test_volume_accepts_the_range(self, volume: int) -> None:
        settings = Settings.model_validate({"voice": {"tts": {"volume": volume}}})

        assert settings.voice.tts.volume == volume

    @pytest.mark.parametrize("volume", [-1, 101])
    def test_volume_rejects_values_outside_the_range(self, volume: int) -> None:
        with pytest.raises(ValueError, match="volume"):
            Settings.model_validate({"voice": {"tts": {"volume": volume}}})

    @pytest.mark.parametrize("limit", RAM_LIMIT_CHOICES)
    def test_ram_limit_accepts_the_allowed_set(self, limit: int) -> None:
        settings = Settings.model_validate({"performance": {"ram_limit_mb": limit}})

        assert settings.performance.ram_limit_mb == limit

    @pytest.mark.parametrize("limit", [1, 3000, 999999])
    def test_ram_limit_rejects_anything_else(self, limit: int) -> None:
        """The settings tab offers a combo box; a free-form value would desync it."""
        with pytest.raises(ValueError, match="лимита памяти"):
            Settings.model_validate({"performance": {"ram_limit_mb": limit}})

    def test_wake_phrases_are_folded_and_deduplicated(self) -> None:
        settings = Settings.model_validate(
            {
                "voice": {
                    "wake": {
                        "phrases": [
                            {"phrase": "  Айрис  "},
                            {"phrase": "АЙРИС"},
                            {"phrase": "Ирис"},
                        ]
                    }
                }
            }
        )

        assert [item.phrase for item in settings.voice.wake.phrases] == ["айрис", "ирис"]

    def test_hotkeys_are_normalised(self) -> None:
        settings = Settings.model_validate({"hotkeys": {"push_to_talk": " Ctrl + Shift + Space "}})

        assert settings.hotkeys.push_to_talk == "ctrl+shift+space"

    def test_two_actions_cannot_share_one_combo(self) -> None:
        with pytest.raises(ValueError, match="уже занято"):
            Settings.model_validate(
                {"hotkeys": {"push_to_talk": "ctrl+shift+a", "toggle_wake": "Ctrl+Shift+A"}}
            )

    def test_history_cannot_be_summarised_before_it_is_dropped(self) -> None:
        with pytest.raises(ValueError, match="сворачивать историю"):
            Settings.model_validate({"ai": {"history_turns": 20, "summarize_after_turns": 5}})

    def test_settings_are_immutable(self) -> None:
        """Readers rely on a snapshot never changing under them."""
        settings = Settings()
        attribute = "theme"

        with pytest.raises(ValueError, match="frozen"):
            setattr(settings.general, attribute, "light")


class TestRecovery:
    """«Готово когда»: a broken TOML never stops Ayris from starting."""

    def test_an_unparsable_file_is_backed_up_and_defaults_are_used(
        self, config_path: Path, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        config_path.write_text("this is [not valid toml", encoding="utf-8")

        settings = ConfigManager(config_path).load()

        broken = config_path.with_name("config.toml.broken")
        assert broken.is_file()
        assert "not valid toml" in broken.read_text(encoding="utf-8")
        assert settings == Settings()
        assert config_path.is_file(), "a fresh file must replace the broken one"
        assert "broken" in messages(ayris_log.records)

    def test_one_invalid_value_does_not_discard_the_rest(self, config_path: Path) -> None:
        config_path.write_text(
            '[general]\ntheme = "dark"\n\n[voice.tts]\nspeed = 99.0\nvolume = 42\n',
            encoding="utf-8",
        )

        settings, dropped = load_settings(config_path)

        assert settings.general.theme == "dark", "valid neighbours must survive"
        assert settings.voice.tts.volume == 42
        assert settings.voice.tts.speed == Settings().voice.tts.speed
        assert dropped == ("voice.tts.speed",)

    def test_an_unknown_key_is_ignored(self, config_path: Path) -> None:
        """A file written by a newer Ayris must still load in an older one."""
        config_path.write_text('[general]\ntheme = "dark"\nfuture_option = 5\n', encoding="utf-8")

        settings, dropped = load_settings(config_path)

        assert settings.general.theme == "dark"
        assert dropped == ()

    def test_an_empty_section_uses_defaults(self, config_path: Path) -> None:
        config_path.write_text("[general]\n", encoding="utf-8")

        settings, dropped = load_settings(config_path)

        assert settings == Settings()
        assert dropped == ()

    def test_a_bom_from_notepad_is_tolerated(self, config_path: Path) -> None:
        config_path.write_text('[general]\ntheme = "light"\n', encoding="utf-8-sig")

        settings, _dropped = load_settings(config_path)

        assert settings.general.theme == "light"

    def test_reload_keeps_the_current_settings_while_the_file_is_broken(
        self, manager: ConfigManager
    ) -> None:
        """Garbage seen mid-edit must not wipe a running session's settings."""
        manager.apply({"general.theme": "light"})
        manager.path.write_text("[[[", encoding="utf-8")

        assert manager.reload() is None
        assert manager.settings.general.theme == "light"
        assert not manager.path.with_name("config.toml.broken").exists()


class TestPersistence:
    def test_round_trip_after_changing_several_sections(self, config_path: Path) -> None:
        settings = Settings.model_validate(
            {
                "general": {"theme": "light", "autostart": True},
                "voice": {"tts": {"speed": 1.35, "volume": 55}},
                "plugins": {"disabled": ["demo"]},
            }
        )

        save_settings(settings, config_path)
        restored, dropped = load_settings(config_path)

        assert dropped == ()
        assert restored == settings

    def test_saving_preserves_a_comment_the_user_wrote(self, config_path: Path) -> None:
        ConfigManager(config_path).load()
        text = config_path.read_text(encoding="utf-8")
        assert "[general]" in text
        config_path.write_text(
            text.replace("[general]", "# моя пометка\n[general]"),
            encoding="utf-8",
        )

        second = ConfigManager(config_path)
        second.load()
        second.apply({"overlay.opacity": 0.5})

        assert "# моя пометка" in config_path.read_text(encoding="utf-8")

    def test_a_save_leaves_no_temporary_file_behind(self, config_path: Path) -> None:
        """A stray ``.tmp`` would confuse both the watcher and the user."""
        save_settings(Settings(), config_path)

        assert list(config_path.parent.iterdir()) == [config_path]

    def test_tuples_are_written_as_arrays(self, config_path: Path) -> None:
        settings = Settings.model_validate({"plugins": {"disabled": ["alpha", "beta"]}})
        save_settings(settings, config_path)

        assert read_toml(config_path)["plugins"]["disabled"] == ["alpha", "beta"]
        assert load_settings(config_path)[0].plugins.disabled == ("alpha", "beta")


class TestRestartScopes:
    """«Готово когда»: fields that cannot be applied live are flagged as such."""

    @pytest.mark.parametrize(
        ("dotted", "expected"),
        [
            ("voice.audio_input.device", RestartScope.AUDIO),
            ("voice.audio_input.sample_rate", RestartScope.AUDIO),
            ("voice.stt.offline_model", RestartScope.STT),
            ("voice.tts.engine", RestartScope.TTS),
            ("voice.wake.engine", RestartScope.WAKE),
            ("ai.model", RestartScope.LLM),
            ("plugins.sandbox", RestartScope.APP),
            ("voice.tts.speed", RestartScope.NONE),
            ("overlay.opacity", RestartScope.NONE),
            ("privacy.telemetry", RestartScope.NONE),
        ],
    )
    def test_the_scope_of_a_known_field(self, dotted: str, expected: RestartScope) -> None:
        assert restart_scope(dotted) is expected

    def test_a_change_inside_a_tagged_container_inherits_its_scope(self) -> None:
        assert restart_scope("voice.wake.phrases") is RestartScope.WAKE
        assert restart_scope("voice.wake.phrases.0.sensitivity") is RestartScope.WAKE

    def test_an_unknown_path_needs_no_restart(self) -> None:
        assert restart_scope("nothing.like.this") is RestartScope.NONE

    def test_every_scope_has_a_russian_label(self) -> None:
        for scope in RestartScope:
            assert scope.label.strip()

    def test_a_diff_separates_live_changes_from_restart_ones(self) -> None:
        old = Settings()
        new = Settings.model_validate(
            {"voice": {"tts": {"speed": 1.5}, "audio_input": {"device": "Микрофон USB"}}}
        )

        change = diff_settings(old, new)

        assert change
        assert change.paths == ("voice.audio_input.device", "voice.tts.speed")
        assert [item.path for item in change.live] == ["voice.tts.speed"]
        assert [item.path for item in change.restart_required] == ["voice.audio_input.device"]
        assert change.restart_scopes == frozenset({RestartScope.AUDIO})
        assert change.touches("voice.audio_input")
        assert not change.touches("overlay")
        assert "voice.tts.speed" in change.summary()

    def test_identical_settings_produce_an_empty_diff(self) -> None:
        change = diff_settings(Settings(), Settings())

        assert not change
        assert len(change) == 0
        assert change.summary() == "настройки не изменились"

    def test_pending_restarts_accumulate_until_acknowledged(self, manager: ConfigManager) -> None:
        manager.apply({"voice.audio_input.device": "Микрофон USB"})

        assert manager.pending_restarts == frozenset({RestartScope.AUDIO})

        manager.acknowledge_restart(RestartScope.AUDIO)
        assert manager.pending_restarts == frozenset()

    def test_a_live_change_creates_no_pending_restart(self, manager: ConfigManager) -> None:
        manager.apply({"voice.tts.speed": 1.2})

        assert manager.pending_restarts == frozenset()


class TestApply:
    def test_apply_validates_saves_and_notifies(self, manager: ConfigManager) -> None:
        seen: list[tuple[str, ...]] = []
        manager.subscribe(lambda change: seen.append(change.paths))

        change = manager.apply({"general.theme": "light", "voice.tts.speed": 1.25})

        assert manager.settings.general.theme == "light"
        assert change.paths == ("general.theme", "voice.tts.speed")
        assert seen == [("general.theme", "voice.tts.speed")]
        assert read_toml(manager.path)["general"]["theme"] == "light"

    def test_apply_rejects_an_invalid_value_and_changes_nothing(
        self, manager: ConfigManager
    ) -> None:
        before = manager.settings
        text_before = manager.path.read_text(encoding="utf-8")

        with pytest.raises(ConfigError) as caught:
            manager.apply({"voice.tts.speed": 9.0})

        assert manager.settings is before
        assert manager.path.read_text(encoding="utf-8") == text_before
        assert "voice.tts.speed" in caught.value.user_message

    def test_applying_an_unchanged_value_is_a_no_op(self, manager: ConfigManager) -> None:
        seen: list[object] = []
        manager.subscribe(seen.append)

        change = manager.apply({"general.theme": manager.settings.general.theme})

        assert not change
        assert seen == []

    def test_unsubscribing_stops_the_callbacks(self, manager: ConfigManager) -> None:
        seen: list[object] = []
        unsubscribe = manager.subscribe(seen.append)

        manager.apply({"overlay.opacity": 0.7})
        unsubscribe()
        manager.apply({"overlay.opacity": 0.6})

        assert len(seen) == 1

    def test_a_failing_listener_does_not_stop_the_others(
        self, manager: ConfigManager, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        def explode(_change: object) -> None:
            raise RuntimeError("listener bug")

        seen: list[object] = []
        manager.subscribe(explode)
        manager.subscribe(seen.append)

        manager.apply({"overlay.opacity": 0.55})

        assert len(seen) == 1
        assert "listener" in messages(ayris_log.records)


class TestHotReload:
    """«Готово когда»: an external edit applies without restarting Ayris."""

    def test_reload_picks_up_an_edit_made_in_a_text_editor(self, manager: ConfigManager) -> None:
        seen: list[tuple[str, ...]] = []
        manager.subscribe(lambda change: seen.append(change.paths))
        text = manager.path.read_text(encoding="utf-8")
        assert 'theme = "dark_purple"' in text
        manager.path.write_text(
            text.replace('theme = "dark_purple"', 'theme = "light"'),
            encoding="utf-8",
        )

        change = manager.reload()

        assert change is not None
        assert change.paths == ("general.theme",)
        assert manager.settings.general.theme == "light"
        assert seen == [("general.theme",)]

    def test_reload_without_a_change_returns_none(self, manager: ConfigManager) -> None:
        assert manager.reload() is None

    def test_a_restart_field_is_reported_instead_of_applied_silently(
        self, manager: ConfigManager
    ) -> None:
        """The value is stored, but nobody may assume the worker adopted it."""
        edited = Settings.model_validate({"voice": {"audio_input": {"device": "Микрофон USB"}}})
        save_settings(edited, manager.path)

        change = manager.reload()

        assert change is not None
        assert change.restart_scopes == frozenset({RestartScope.AUDIO})
        assert manager.pending_restarts == frozenset({RestartScope.AUDIO})
        assert manager.settings.voice.audio_input.device == "Микрофон USB"

    def test_the_watcher_notices_an_edit_made_while_ayris_runs(
        self, manager: ConfigManager
    ) -> None:
        applied = threading.Event()
        manager.subscribe(lambda _change: applied.set())
        manager.start_watching(interval=0.02)

        text = manager.path.read_text(encoding="utf-8")
        assert "opacity = 0.92" in text
        manager.path.write_text(
            text.replace("opacity = 0.92", "opacity = 0.4"),
            encoding="utf-8",
        )

        assert applied.wait(timeout=15.0), "the watcher never noticed the edit"
        assert manager.settings.overlay.opacity == pytest.approx(0.4)

    def test_the_watcher_ignores_the_managers_own_writes(self, manager: ConfigManager) -> None:
        seen: list[object] = []
        manager.subscribe(seen.append)
        manager.start_watching(interval=0.02)

        manager.apply({"overlay.opacity": 0.44})
        time.sleep(0.3)

        assert len(seen) == 1, "a write by Ayris itself must not fire a second event"

    def test_stopping_a_watcher_that_never_started_is_safe(self, manager: ConfigManager) -> None:
        manager.stop_watching()
        manager.stop_watching()


class TestModuleSingleton:
    def test_init_config_installs_the_manager(self, config_path: Path) -> None:
        manager = init_config(config_path)

        assert get_config_manager() is manager
        assert get_settings() == manager.settings

    def test_the_manager_loads_itself_on_first_use(self, profile_paths: AppPaths) -> None:
        settings = get_settings()

        assert settings.general.language == "ru"
        assert profile_paths.config_file.is_file()


class TestEnvironmentOverrides:
    """``AYRIS_VOICE__TTS__SPEED=1.4`` pins one field for one launch."""

    def test_a_nested_variable_wins_over_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AYRIS_VOICE__TTS__SPEED", "1.4")

        settings = Settings()

        assert settings.voice.tts.speed == pytest.approx(1.4)
        assert settings.voice.tts.volume == 80, "the rest of the section keeps its defaults"

    def test_an_invalid_variable_is_reported_rather_than_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AYRIS_VOICE__TTS__SPEED", "9.0")

        with pytest.raises(ValueError, match="speed"):
            Settings()


class FakeKeyring:
    """In-memory stand-in for the Windows Credential Manager."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], str] = {}
        self.fail = False

    def _check(self) -> None:
        if self.fail:
            raise RuntimeError("credential store is locked")

    def get_password(self, service_name: str, username: str) -> str | None:
        self._check()
        return self.entries.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self._check()
        self.entries[(service_name, username)] = password

    def delete_password(self, service_name: str, username: str) -> None:
        self._check()
        del self.entries[(service_name, username)]


@pytest.fixture
def fake_keyring() -> FakeKeyring:
    return FakeKeyring()


@pytest.fixture
def store(fake_keyring: FakeKeyring) -> SecretsStore:
    """A store wired to the fake backend and installed as the process-wide one."""
    backend: KeyringBackend = fake_keyring
    instance = SecretsStore("Ayris-test", backend=backend)
    reset_secrets(instance)
    return instance


class TestSecrets:
    """«Готово когда»: the key lives in keyring, not in config.toml and not in a log."""

    def test_a_key_can_be_stored_and_read_back(self, store: SecretsStore) -> None:
        store.save("openai", "sk-secret-value-1234")

        assert store.get("openai") == "sk-secret-value-1234"
        assert store.has("openai")
        assert store.stored_refs(["openai", "anthropic"]) == ("openai",)

    def test_a_missing_reference_reads_as_none(self, store: SecretsStore) -> None:
        assert store.get("anthropic") is None
        assert store.has("anthropic") is False

    def test_delete_reports_whether_anything_was_removed(self, store: SecretsStore) -> None:
        store.save("openai", "sk-secret-value-1234")

        assert store.delete("openai") is True
        assert store.delete("openai") is False

    def test_an_empty_value_is_refused(self, store: SecretsStore) -> None:
        with pytest.raises(SecretsError, match="empty credential"):
            store.save("openai", "   ")

    @pytest.mark.parametrize("ref", ["sk-abcdefghijklmnop", "Openai", "", "a" * 40, "два слова"])
    def test_a_key_pasted_into_the_name_field_is_refused(
        self, ref: str, store: SecretsStore
    ) -> None:
        """Otherwise the key itself would end up in ``config.toml`` in clear text."""
        assert not is_valid_ref(ref)
        with pytest.raises(SecretsError):
            store.get(ref)

    def test_a_backend_failure_becomes_a_secrets_error(
        self, store: SecretsStore, fake_keyring: FakeKeyring
    ) -> None:
        fake_keyring.fail = True

        with pytest.raises(SecretsError, match="cannot read"):
            store.get("openai")
        assert store.is_available() is False

    def test_masking_never_reveals_more_than_the_tail(self) -> None:
        assert mask("sk-abcdefghijkl").endswith("ijkl")
        assert "abcdefgh" not in mask("sk-abcdefghijkl")
        assert set(mask("short")) == {"•"}, "a short value must not show a tail at all"
        assert mask("") == mask(None)
        assert "•" not in mask("")

    def test_the_store_never_reveals_a_secret_in_its_repr(self, store: SecretsStore) -> None:
        store.save("openai", "sk-super-secret-value")

        assert "sk-super-secret-value" not in repr(store)

    def test_a_saved_key_never_reaches_the_log(
        self, store: SecretsStore, ayris_log: pytest.LogCaptureFixture
    ) -> None:
        store.save("openai", "sk-super-secret-value")
        store.delete("openai")

        written = messages(ayris_log.records)
        assert "sk-super-secret-value" not in written
        assert "openai" in written, "the reference name is safe, and useful, to log"

    def test_the_config_file_holds_only_the_reference(
        self, config_path: Path, store: SecretsStore
    ) -> None:
        store.save("anthropic", "sk-super-secret-value")
        manager = ConfigManager(config_path)
        manager.load()
        manager.apply({"ai.credential_ref": "anthropic"})

        text = config_path.read_text(encoding="utf-8")
        assert 'credential_ref = "anthropic"' in text
        assert "sk-super-secret-value" not in text
        assert "sk-super-secret-value" not in str(dump_settings(manager.settings))

    def test_secret_refs_feed_the_resolver(self, store: SecretsStore) -> None:
        store.save("openai", "sk-super-secret-value")
        settings = Settings.model_validate({"ai": {"credential_ref": "openai"}})

        resolved = dict(resolve_secrets(settings.secret_refs()))

        assert resolved == {"ai": "sk-super-secret-value"}

    def test_prune_removes_keys_that_are_no_longer_referenced(self, store: SecretsStore) -> None:
        store.save("openai", "sk-one-secret-value")
        store.save("anthropic", "sk-two-secret-value")

        removed = store.prune(keep=["openai"])

        assert removed == ("anthropic",)
        assert store.has("openai")
        assert not store.has("anthropic")

    def test_every_known_slot_is_described_in_russian(self) -> None:
        for ref, slot in KNOWN_SLOTS.items():
            assert slot.ref == ref
            assert slot.title.strip()
            assert slot.hint.strip()

    def test_a_status_reads_naturally_in_the_settings_tab(self, store: SecretsStore) -> None:
        store.save("openai", "sk-super-secret-value")

        assert store.status("openai").label == "OpenAI: ключ сохранён"
        assert store.status("anthropic").label == "Anthropic: ключ не задан"
