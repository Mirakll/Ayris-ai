"""Task 06: profiles — lifecycle, the portable ``.zip``, backups, moving the root.

The centre of gravity here is the export→import round trip, because that is the
one operation whose failure modes are invisible: a bundle that loses a trigger,
mangles a Cyrillic name or quietly imports half of itself all look like success
from the outside. Every fixture is therefore named in Russian — an ASCII-only
fixture would pass even if the archive lost its UTF-8 flag, which is exactly the
bug the flag exists to prevent.

Groups:

* :class:`TestNames` — the one pure function worth testing on its own.
* :class:`TestLifecycle` — create, copy, rename, delete, switch, notify.
* :class:`TestRoundTrip` — export → import onto an empty profile, verbatim.
* :class:`TestSecrets` — credentials must not be in the archive, in any form.
* :class:`TestConflicts` — overwrite / rename / skip, for commands and sounds.
* :class:`TestManifest` — schema-version gates and hostile archives.
* :class:`TestRollback` — a bundle that fails half way changes nothing.
* :class:`TestBackups` — backup before import and reset, pruning.
* :class:`TestRoot` — moving the installation and living to tell about it.
* :class:`TestWindowsOnly` — behaviour that only exists on the target platform.
"""

from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from ayris.core import paths as paths_module
from ayris.core.config import ConfigManager, dump_settings
from ayris.core.database import Database, reset_database
from ayris.core.errors import ProfileError
from ayris.core.events import EventBus
from ayris.core.models import (
    Command,
    CommandFolder,
    ModelRecord,
    Profile,
    Trigger,
    TriggerType,
    VariableScope,
    VariableType,
)
from ayris.core.portable_profile import (
    BUNDLE_FORMAT,
    BUNDLE_SCHEMA_VERSION,
    COMMANDS_NAME,
    CONFIG_NAME,
    MANIFEST_NAME,
    MODELS_NAME,
    SOUNDS_PREFIX,
    VARIABLES_NAME,
    BundleManifest,
    ConflictPolicy,
    export_bundle,
    import_bundle,
    preview_bundle,
    read_manifest,
    strip_secrets,
)
from ayris.core.profile import (
    DEFAULT_PROFILE_NAME,
    MAX_BACKUPS,
    MAX_NAME_LENGTH,
    ProfileManager,
    ProfilesChanged,
    ProfileSwitched,
    _clean_name,
    _file_stem,
)
from ayris.core.repositories import Repositories

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.unit

WINDOWS_ONLY = pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific behaviour")

#: Sound payloads used by the conflict and rollback tests. Written as encoded
#: text rather than byte literals because Python only allows ASCII in the
#: latter, and an ASCII fixture would not exercise the UTF-8 paths at all.
FROM_BUNDLE = "из архива".encode()
FROM_DISK = "локальный".encode()


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path: Path) -> paths_module.AppPaths:
    """An initialised installation root, process-wide for the duration."""
    return paths_module.init_paths(profile=tmp_path / "root")


@pytest.fixture
def database(paths: paths_module.AppPaths) -> Iterator[Database]:
    """The installation's database, open and migrated."""
    handle = Database.open(paths.database_file)
    yield handle
    handle.close()
    reset_database()


@pytest.fixture
def repos(database: Database) -> Repositories:
    return Repositories(database)


@pytest.fixture
def bus() -> EventBus:
    """A bus that delivers on the calling thread, so publishes are synchronous."""
    return EventBus(thread_id=None)


@pytest.fixture
def manager(
    repos: Repositories,
    paths: paths_module.AppPaths,
    bus: EventBus,
) -> ProfileManager:
    return ProfileManager(repos, paths=paths, bus=bus)


def seed(repos: Repositories, profile_id: int) -> None:
    """Fill a profile with the kind of content a real one has, named in Russian.

    Two nested folders, a command in the child folder with two triggers and a
    non-trivial action list, a loose command, and one variable of each exported
    scope. Enough that a round trip has something to lose.
    """
    outer = repos.folders.create(CommandFolder(name="Свет", profile_id=profile_id, sort_order=1))
    assert outer.id is not None
    inner = repos.folders.create(
        CommandFolder(name="Кухня", profile_id=profile_id, parent_id=outer.id, sort_order=2)
    )
    assert inner.id is not None

    lamp = repos.commands.create(
        Command(
            name="Включи свет",
            profile_id=profile_id,
            folder_id=inner.id,
            description="Свет на кухне",
            tags=("дом", "свет"),
            priority=7,
            cooldown_ms=250,
            require_admin=True,
            actions=(
                {"type": "run", "path": "C:/Windows/System32/notepad.exe"},
                {"type": "say", "text": "Готово, свет включён"},
            ),
        )
    )
    assert lamp.id is not None
    repos.triggers.add_voice(lamp.id, "включи свет на кухне")
    repos.triggers.add(
        Trigger(
            command_id=lamp.id,
            type=TriggerType.HOTKEY,
            payload={"combo": "Ctrl+Alt+Л"},
            fuzzy=False,
            priority=3,
        )
    )

    note = repos.commands.create(
        Command(name="Заметка", profile_id=profile_id, description="Без папки", enabled=False)
    )
    assert note.id is not None
    repos.triggers.add_voice(note.id, "запиши заметку", fuzzy=False)

    repos.variables.set(
        "город", "Москва", scope=VariableScope.PROFILE, profile_id=profile_id, persistent=True
    )
    repos.variables.set(
        "порог", 42, scope=VariableScope.GLOBAL, var_type=VariableType.INT, persistent=True
    )
    repos.variables.set(
        "черновик",
        "не сохранять",
        scope=VariableScope.PROFILE,
        profile_id=profile_id,
        persistent=False,
    )


def read_entry(archive: Path, name: str) -> dict[str, Any]:
    """Decode one JSON member of a bundle."""
    with zipfile.ZipFile(archive) as bundle:
        payload = json.loads(bundle.read(name).decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def rewrite(archive: Path, destination: Path, replacements: dict[str, bytes]) -> Path:
    """Copy a bundle, substituting the named members. Used to forge bad archives."""
    with (
        zipfile.ZipFile(archive) as source,
        zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for info in source.infolist():
            data = replacements.get(info.filename, source.read(info.filename))
            new_info = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            new_info.compress_type = zipfile.ZIP_DEFLATED
            new_info.flag_bits |= 0x800
            target.writestr(new_info, data)
    return destination


def command_names(repos: Repositories, profile_id: int) -> set[str]:
    return {command.name for command in repos.commands.list_for_profile(profile_id)}


# ----------------------------------------------------------------------
# names
# ----------------------------------------------------------------------


class TestNames:
    """Name validation, which every mutating method funnels through."""

    def test_collapses_whitespace(self) -> None:
        assert _clean_name("  Мой   профиль \n") == "Мой профиль"

    @pytest.mark.parametrize("raw", ["", "   ", "\t\n"])
    def test_rejects_blank(self, raw: str) -> None:
        with pytest.raises(ProfileError):
            _clean_name(raw)

    def test_rejects_too_long(self) -> None:
        with pytest.raises(ProfileError):
            _clean_name("я" * (MAX_NAME_LENGTH + 1))

    def test_accepts_the_limit_exactly(self) -> None:
        name = "я" * MAX_NAME_LENGTH
        assert _clean_name(name) == name

    def test_file_stem_strips_windows_reserved_characters(self) -> None:
        assert _file_stem("Дом: свет/тепло?") == "Дом_ свет_тепло_"

    def test_file_stem_never_returns_empty(self) -> None:
        assert _file_stem("...") == "profile"


# ----------------------------------------------------------------------
# lifecycle
# ----------------------------------------------------------------------


class TestLifecycle:
    """Create, copy, rename, delete, switch."""

    def test_creates_the_default_profile_on_an_empty_database(
        self, repos: Repositories, paths: paths_module.AppPaths
    ) -> None:
        manager = ProfileManager(repos, paths=paths)
        assert manager.active.name == DEFAULT_PROFILE_NAME
        assert manager.active.is_active is True

    def test_adopts_an_existing_profile_that_is_not_marked_active(
        self, repos: Repositories, paths: paths_module.AppPaths
    ) -> None:
        repos.profiles.create("Работа")
        manager = ProfileManager(repos, paths=paths)
        assert manager.active.name == "Работа"
        stored = repos.profiles.active()
        assert stored is not None and stored.name == "Работа"

    def test_create_rejects_a_duplicate_name(self, manager: ProfileManager) -> None:
        manager.create("Игры")
        with pytest.raises(ProfileError, match="already exists"):
            manager.create("Игры")

    def test_create_can_activate_immediately(self, manager: ProfileManager) -> None:
        created = manager.create("Игры", activate=True)
        assert manager.active.id == created.id

    def test_rename_follows_the_active_profile(self, manager: ProfileManager) -> None:
        renamed = manager.rename(manager.active, "Основной")
        assert renamed.name == "Основной"
        assert manager.active.name == "Основной"

    def test_rename_to_the_same_name_is_a_no_op(self, manager: ProfileManager) -> None:
        before = manager.active
        assert manager.rename(before, before.name) == before

    def test_rename_rejects_a_name_another_profile_holds(self, manager: ProfileManager) -> None:
        manager.create("Игры")
        with pytest.raises(ProfileError):
            manager.rename(manager.active, "Игры")

    def test_copy_duplicates_contents_without_linking_them(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)

        clone = manager.copy(manager.active)
        assert clone.id is not None
        assert clone.name == f"{manager.active.name} — копия"
        assert command_names(repos, clone.id) == {"Включи свет", "Заметка"}

        # Folder tree, not just a flat list of names.
        folders = {
            folder.name: folder
            for folder in repos.folders.list_for_profile(clone.id)
            if folder.profile_id == clone.id
        }
        assert set(folders) == {"Свет", "Кухня"}
        assert folders["Кухня"].parent_id == folders["Свет"].id

        # Triggers came along.
        lamp = repos.commands.get_by_name(clone.id, "Включи свет")
        assert lamp is not None and lamp.id is not None
        assert {trigger.type for trigger in repos.triggers.list_for_command(lamp.id)} == {
            TriggerType.VOICE,
            TriggerType.HOTKEY,
        }

        # And editing the copy leaves the original alone.
        repos.commands.delete(lamp.id)
        assert command_names(repos, source_id) == {"Включи свет", "Заметка"}

    def test_copy_picks_a_free_name_when_called_twice(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        first = manager.copy(manager.active)
        second = manager.copy(manager.active)
        assert first.name != second.name
        assert second.name.endswith("(2)")
        assert len(repos.profiles.list_all()) == 3

    def test_copy_rejects_an_explicit_name_that_is_taken(self, manager: ProfileManager) -> None:
        manager.create("Игры")
        with pytest.raises(ProfileError):
            manager.copy(manager.active, "Игры")

    def test_delete_refuses_the_last_profile(self, manager: ProfileManager) -> None:
        with pytest.raises(ProfileError, match="last profile"):
            manager.delete(manager.active)

    def test_delete_moves_the_active_flag_to_a_survivor(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        doomed = manager.active
        manager.create("Работа")
        survivor = manager.delete(doomed)
        assert survivor.name == "Работа"
        assert manager.active.name == "Работа"
        stored = repos.profiles.active()
        assert stored is not None and stored.name == "Работа"

    def test_delete_of_an_inactive_profile_leaves_the_active_one_alone(
        self, manager: ProfileManager
    ) -> None:
        spare = manager.create("Работа")
        active_before = manager.active
        manager.delete(spare)
        assert manager.active.id == active_before.id

    def test_switch_notifies_listeners_and_the_bus(
        self, manager: ProfileManager, bus: EventBus
    ) -> None:
        seen: list[str] = []
        events: list[ProfileSwitched] = []
        manager.subscribe(lambda profile: seen.append(profile.name))
        bus.subscribe(ProfileSwitched, events.append)

        target = manager.create("Игры")
        manager.switch(target)

        assert seen == ["Игры"]
        assert [event.profile.name for event in events] == ["Игры"]
        assert events[0].previous is not None
        assert events[0].previous.name == DEFAULT_PROFILE_NAME

    def test_switch_to_the_active_profile_notifies_nobody(self, manager: ProfileManager) -> None:
        seen: list[str] = []
        manager.subscribe(lambda profile: seen.append(profile.name))
        manager.switch(manager.active)
        assert seen == []

    def test_switch_drops_transient_variables(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        repos.variables.set("буфер", "мусор", scope=VariableScope.GLOBAL, persistent=False)
        repos.variables.set("город", "Москва", scope=VariableScope.GLOBAL, persistent=True)
        manager.switch(manager.create("Игры"))
        assert repos.variables.get("буфер", scope=VariableScope.GLOBAL) is None
        assert repos.variables.get("город", scope=VariableScope.GLOBAL) is not None

    def test_a_broken_subscriber_does_not_stop_the_others(self, manager: ProfileManager) -> None:
        reached: list[str] = []

        def explode(_profile: Profile) -> None:
            raise RuntimeError("подписчик сломан")

        manager.subscribe(explode)
        manager.subscribe(lambda profile: reached.append(profile.name))
        manager.switch(manager.create("Игры"))
        assert reached == ["Игры"]

    def test_unsubscribe_stops_delivery(self, manager: ProfileManager) -> None:
        seen: list[str] = []
        cancel = manager.subscribe(lambda profile: seen.append(profile.name))
        cancel()
        manager.switch(manager.create("Игры"))
        assert seen == []

    def test_list_change_events_carry_the_new_list(
        self, manager: ProfileManager, bus: EventBus
    ) -> None:
        events: list[ProfilesChanged] = []
        bus.subscribe(ProfilesChanged, events.append)
        manager.create("Игры")
        assert events
        assert {profile.name for profile in events[-1].profiles} == {
            DEFAULT_PROFILE_NAME,
            "Игры",
        }

    def test_operations_on_an_unsaved_profile_are_refused(self, manager: ProfileManager) -> None:
        with pytest.raises(ProfileError, match="no id"):
            manager.rename(Profile(name="Ниоткуда"), "Куда-то")


# ----------------------------------------------------------------------
# the round trip
# ----------------------------------------------------------------------


class TestRoundTrip:
    """Export a full profile, import it into an empty one, compare."""

    def test_export_import_preserves_everything(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)

        archive = tmp_path / "профиль.zip"
        manifest = manager.export(archive)
        assert archive.is_file()
        assert manifest.counts["commands"] == 2
        assert manifest.counts["folders"] == 2
        # The transient variable is not exported; the other two are.
        assert manifest.counts["variables"] == 2

        empty = manager.create("Пустой")
        assert empty.id is not None
        report = manager.import_bundle(archive, profile=empty, backup=False)

        assert set(report.added_commands) == {"Включи свет", "Заметка"}
        assert report.renamed_commands == ()
        assert report.skipped_commands == ()
        assert report.total_commands == 2
        assert set(report.added_folders) == {"Свет", "Свет / Кухня"}

        lamp = repos.commands.get_by_name(empty.id, "Включи свет")
        original = repos.commands.get_by_name(source_id, "Включи свет")
        assert lamp is not None and lamp.id is not None
        assert original is not None
        assert lamp.description == original.description
        assert lamp.tags == original.tags
        assert lamp.priority == original.priority
        assert lamp.cooldown_ms == original.cooldown_ms
        assert lamp.require_admin == original.require_admin
        assert lamp.actions == original.actions
        assert lamp.enabled is True

        note = repos.commands.get_by_name(empty.id, "Заметка")
        assert note is not None
        assert note.enabled is False
        assert note.folder_id is None

        # Folder identity travels as a name path, so the tree is rebuilt.
        folders = {
            folder.name: folder
            for folder in repos.folders.list_for_profile(empty.id)
            if folder.profile_id == empty.id
        }
        assert set(folders) == {"Свет", "Кухня"}
        assert folders["Кухня"].parent_id == folders["Свет"].id
        assert lamp.folder_id == folders["Кухня"].id

        triggers = {trigger.type: trigger for trigger in repos.triggers.list_for_command(lamp.id)}
        assert set(triggers) == {TriggerType.VOICE, TriggerType.HOTKEY}
        assert triggers[TriggerType.VOICE].phrase == "включи свет на кухне"
        assert triggers[TriggerType.VOICE].fuzzy is True
        assert triggers[TriggerType.HOTKEY].payload == {"combo": "Ctrl+Alt+Л"}
        assert triggers[TriggerType.HOTKEY].fuzzy is False
        assert triggers[TriggerType.HOTKEY].priority == 3

        # Global variables belong to the installation, not the profile, so the
        # one the source profile created is already visible here and is left
        # alone; only the profile-scoped one is actually imported.
        assert report.added_variables == ("город",)
        assert report.skipped_variables == ("порог",)
        city = repos.variables.get("город", scope=VariableScope.PROFILE, profile_id=empty.id)
        assert city is not None
        assert city.value == "Москва"

    def test_cyrillic_entry_names_carry_the_utf8_flag(
        self,
        manager: ProfileManager,
        repos: Repositories,
        paths: paths_module.AppPaths,
        tmp_path: Path,
    ) -> None:
        """Bit 11 must be set on every Cyrillic name, or Explorer shows mojibake.

        Only the non-ASCII names need it, and that is what :mod:`zipfile`
        does — the point of the test is that the flag is present where it
        matters and that the name survives a re-read.
        """
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        (paths.sounds_dir / "звук готовности.wav").write_bytes(b"RIFF")

        archive = tmp_path / "профиль.zip"
        manager.export(archive)

        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            flags = {info.filename: bool(info.flag_bits & 0x800) for info in bundle.infolist()}
        sound = f"{SOUNDS_PREFIX}звук готовности.wav"
        assert sound in names
        assert flags[sound] is True
        assert all(flagged for name, flagged in flags.items() if not name.isascii()), flags

        # And the Cyrillic content inside the entries survived as well.
        payload = read_entry(archive, COMMANDS_NAME)
        assert {entry["name"] for entry in payload["commands"]} == {"Включи свет", "Заметка"}
        assert ["Свет", "Кухня"] in [entry["path"] for entry in payload["folders"]]

    def test_sounds_come_back_byte_for_byte(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        (paths.sounds_dir / "звук.wav").write_bytes(b"\x00\x01RIFF\xff")

        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        (paths.sounds_dir / "звук.wav").unlink()

        report = manager.import_bundle(archive, backup=False)
        assert report.added_sounds == ("звук.wav",)
        assert (paths.sounds_dir / "звук.wav").read_bytes() == b"\x00\x01RIFF\xff"

    def test_import_reports_models_it_cannot_find(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        repos.models.add(ModelRecord(kind="stt", name="vosk-ru-0.42", version="0.42"))
        archive = tmp_path / "профиль.zip"
        manager.export(archive)

        installed = repos.models.list_all()
        assert installed and installed[0].id is not None
        repos.models.delete(installed[0].id)

        report = manager.import_bundle(archive, backup=False)
        assert any("vosk-ru-0.42" in item for item in report.missing_models)
        assert "vosk-ru-0.42" in report.describe()

    def test_models_travel_as_manifests_not_weights(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        repos.models.add(
            ModelRecord(kind="stt", name="vosk-ru-0.42", path="D:/models/vosk", sha256="ab" * 32)
        )
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        payload = read_entry(archive, MODELS_NAME)
        entry = payload["models"][0]
        assert entry["name"] == "vosk-ru-0.42"
        assert entry["sha256"] == "ab" * 32
        # An absolute path on someone else's machine is worse than useless.
        assert "path" not in entry

    def test_preview_changes_nothing_and_names_the_conflicts(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        archive = tmp_path / "профиль.zip"
        manager.export(archive)

        preview = manager.preview_import(archive)
        assert set(preview.commands) == {"Включи свет", "Заметка"}
        assert set(preview.conflicts) == {"Включи свет", "Заметка"}
        assert set(preview.folders) == {"Свет", "Свет / Кухня"}
        assert set(preview.variables) == {"город", "порог"}
        assert "Включи свет" in preview.describe() or "Команд: 2" in preview.describe()
        assert len(repos.commands.list_for_profile(source_id)) == 2

    def test_preview_without_a_target_reports_no_conflicts(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        assert preview_bundle(archive).conflicts == ()

    def test_export_of_a_profile_without_an_id_is_refused(
        self, repos: Repositories, tmp_path: Path
    ) -> None:
        with pytest.raises(ProfileError, match="no id"):
            export_bundle(repos, tmp_path / "нет.zip", profile=Profile(name="Ниоткуда"))

    def test_import_into_a_profile_without_an_id_is_refused(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        with pytest.raises(ProfileError, match="no id"):
            import_bundle(archive, repos, profile=Profile(name="Ниоткуда"))

    def test_a_half_written_export_never_replaces_a_good_one(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        assert not archive.with_name(f"{archive.name}.part").exists()


# ----------------------------------------------------------------------
# secrets
# ----------------------------------------------------------------------


class TestSecrets:
    """«Секреты в экспорт не включать» — verified against the archive bytes."""

    def test_strip_secrets_removes_credential_fields_at_any_depth(self) -> None:
        cleaned, removed = strip_secrets(
            {
                "ai": {"provider": "openai", "credential_ref": "ayris/ai", "temperature": 0.5},
                "voice": {"stt": {"api_key": "sk-живой-ключ", "engine": "vosk"}},
                "general": {"theme": "dark"},
            }
        )
        assert cleaned == {
            "ai": {"provider": "openai", "temperature": 0.5},
            "voice": {"stt": {"engine": "vosk"}},
            "general": {"theme": "dark"},
        }
        assert set(removed) == {"ai.credential_ref", "voice.stt.api_key"}

    @pytest.mark.parametrize(
        "key",
        ["credential_ref", "api_key", "apiKey", "access_key", "password", "SECRET", "auth_token"],
    )
    def test_every_credential_shaped_name_is_dropped(self, key: str) -> None:
        cleaned, removed = strip_secrets({key: "значение", "keep": 1})
        assert cleaned == {"keep": 1}
        assert removed == (key,)

    def test_the_archive_does_not_contain_a_credential_reference(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        config = ConfigManager(paths.config_file)
        config.apply({"ai.credential_ref": "ayris/openai", "general.theme": "dark"})
        assert config.settings.ai.credential_ref == "ayris/openai"

        with_config = ProfileManager(manager.repositories, paths=paths, config=config)
        archive = tmp_path / "профиль.zip"
        with_config.export(archive)

        with zipfile.ZipFile(archive) as bundle:
            assert CONFIG_NAME in bundle.namelist()
            text = bundle.read(CONFIG_NAME).decode("utf-8")
            raw = b"".join(bundle.read(name) for name in bundle.namelist())

        assert "ayris/openai" not in text
        assert b"ayris/openai" not in raw
        # The setting that is not a secret survived, so this is not a blanket drop.
        assert "dark" in text

    def test_importing_settings_cannot_repoint_a_local_credential(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        config = ConfigManager(paths.config_file)
        config.apply({"ai.credential_ref": "ayris/мой-ключ"})

        archive = tmp_path / "профиль.zip"
        ProfileManager(manager.repositories, paths=paths, config=config).export(archive)

        # Forge an archive whose config tries to point at someone else's entry.
        forged = rewrite(
            archive,
            tmp_path / "поддельный.zip",
            {CONFIG_NAME: '[ai]\ncredential_ref = "чужой-ключ"\n'.encode()},
        )
        report = manager.import_bundle(forged, backup=False, apply_config=True)
        assert report.config_applied is True
        assert "чужой-ключ" not in paths.config_file.read_text("utf-8")

    def test_settings_are_not_applied_unless_asked(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        config = ConfigManager(paths.config_file)
        config.apply({"general.theme": "light"})
        archive = tmp_path / "профиль.zip"
        ProfileManager(manager.repositories, paths=paths, config=config).export(archive)

        config.apply({"general.theme": "dark"})
        report = manager.import_bundle(archive, backup=False)
        assert report.config_applied is False
        assert dump_settings(config.load())["general"]["theme"] == "dark"

    def test_export_without_a_config_manager_ships_no_settings(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        with zipfile.ZipFile(archive) as bundle:
            assert CONFIG_NAME not in bundle.namelist()


# ----------------------------------------------------------------------
# conflicts
# ----------------------------------------------------------------------


class TestConflicts:
    """What happens when a name in the bundle is already taken."""

    @pytest.fixture
    def archive(self, manager: ProfileManager, repos: Repositories, tmp_path: Path) -> Path:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        destination = tmp_path / "профиль.zip"
        manager.export(destination)
        return destination

    def test_rename_keeps_both_copies(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        report = manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)

        assert dict(report.renamed_commands) == {
            "Включи свет": "Включи свет (2)",
            "Заметка": "Заметка (2)",
        }
        assert command_names(repos, profile_id) == {
            "Включи свет",
            "Включи свет (2)",
            "Заметка",
            "Заметка (2)",
        }

    def test_rename_twice_walks_the_counter(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)
        manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)
        assert "Включи свет (3)" in command_names(repos, profile_id)

    def test_overwrite_replaces_in_place(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        lamp = repos.commands.get_by_name(profile_id, "Включи свет")
        assert lamp is not None and lamp.id is not None
        repos.commands.update(
            Command(
                id=lamp.id,
                name="Включи свет",
                profile_id=profile_id,
                description="испорчено",
                actions=(),
            )
        )

        report = manager.import_bundle(archive, policy=ConflictPolicy.OVERWRITE, backup=False)
        assert set(report.replaced_commands) == {"Включи свет", "Заметка"}
        assert command_names(repos, profile_id) == {"Включи свет", "Заметка"}
        restored = repos.commands.get_by_name(profile_id, "Включи свет")
        assert restored is not None
        assert restored.description == "Свет на кухне"
        assert len(restored.actions) == 2
        # The replaced row is gone, triggers and all — not two rows for one name.
        assert len(repos.commands.list_for_profile(profile_id)) == 2

    def test_skip_leaves_the_existing_commands_untouched(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        report = manager.import_bundle(archive, policy=ConflictPolicy.SKIP, backup=False)
        assert set(report.skipped_commands) == {"Включи свет", "Заметка"}
        assert report.added_commands == ()
        assert len(repos.commands.list_for_profile(profile_id)) == 2

    def test_existing_folders_are_reused_rather_than_duplicated(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        before = len(repos.folders.list_for_profile(profile_id))
        report = manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)
        assert report.added_folders == ()
        assert len(repos.folders.list_for_profile(profile_id)) == before

    def test_variables_are_kept_unless_overwrite_is_asked(
        self, manager: ProfileManager, repos: Repositories, archive: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        repos.variables.set("город", "Тверь", scope=VariableScope.PROFILE, profile_id=profile_id)

        report = manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)
        assert "город" in report.skipped_variables
        kept = repos.variables.get("город", scope=VariableScope.PROFILE, profile_id=profile_id)
        assert kept is not None and kept.value == "Тверь"

        report = manager.import_bundle(archive, policy=ConflictPolicy.OVERWRITE, backup=False)
        assert "город" in report.added_variables
        replaced = repos.variables.get("город", scope=VariableScope.PROFILE, profile_id=profile_id)
        assert replaced is not None and replaced.value == "Москва"

    def test_sound_conflicts_follow_the_policy(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        target = paths.sounds_dir / "звук.wav"
        target.write_bytes(FROM_BUNDLE)
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        target.write_bytes(FROM_DISK)

        skipped = manager.import_bundle(archive, policy=ConflictPolicy.SKIP, backup=False)
        assert skipped.skipped_sounds == ("звук.wav",)
        assert target.read_bytes() == FROM_DISK

        renamed = manager.import_bundle(archive, policy=ConflictPolicy.RENAME, backup=False)
        assert renamed.added_sounds == ("звук (2).wav",)
        assert target.read_bytes() == FROM_DISK
        assert (paths.sounds_dir / "звук (2).wav").read_bytes() == FROM_BUNDLE

        overwritten = manager.import_bundle(archive, policy=ConflictPolicy.OVERWRITE, backup=False)
        assert overwritten.added_sounds == ("звук.wav",)
        assert target.read_bytes() == FROM_BUNDLE

    def test_sounds_can_be_left_out_of_an_import(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        (paths.sounds_dir / "звук.wav").write_bytes(b"RIFF")
        archive = tmp_path / "профиль.zip"
        manager.export(archive)
        (paths.sounds_dir / "звук.wav").unlink()

        report = manager.import_bundle(archive, backup=False, include_sounds=False)
        assert report.added_sounds == ()
        assert not (paths.sounds_dir / "звук.wav").exists()


# ----------------------------------------------------------------------
# manifest and hostile archives
# ----------------------------------------------------------------------


class TestManifest:
    """The header is the only thing standing between us and someone's zip bomb."""

    @pytest.fixture
    def archive(self, manager: ProfileManager, tmp_path: Path) -> Path:
        destination = tmp_path / "профиль.zip"
        manager.export(destination)
        return destination

    def test_manifest_carries_the_versions_and_the_date(
        self, manager: ProfileManager, archive: Path
    ) -> None:
        manifest = read_manifest(archive)
        assert manifest.schema_version == BUNDLE_SCHEMA_VERSION
        assert manifest.profile_name == manager.active.name
        assert manifest.app_version
        assert manifest.db_schema_version > 0
        assert manifest.created_at.tzinfo is not None
        assert read_entry(archive, MANIFEST_NAME)["format"] == BUNDLE_FORMAT

    def test_a_newer_schema_version_is_refused(self, archive: Path, tmp_path: Path) -> None:
        payload = read_entry(archive, MANIFEST_NAME)
        payload["schema_version"] = BUNDLE_SCHEMA_VERSION + 1
        forged = rewrite(
            archive,
            tmp_path / "новый.zip",
            {MANIFEST_NAME: json.dumps(payload, ensure_ascii=False).encode("utf-8")},
        )
        with pytest.raises(ProfileError, match="newer than supported") as caught:
            read_manifest(forged)
        assert caught.value.recoverable is False
        assert "Обновите приложение" in caught.value.user_message

    def test_a_newer_schema_version_is_refused_before_anything_is_written(
        self, manager: ProfileManager, repos: Repositories, archive: Path, tmp_path: Path
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        payload = read_entry(archive, MANIFEST_NAME)
        payload["schema_version"] = 99
        forged = rewrite(
            archive,
            tmp_path / "новый.zip",
            {MANIFEST_NAME: json.dumps(payload, ensure_ascii=False).encode("utf-8")},
        )
        with pytest.raises(ProfileError):
            manager.import_bundle(forged, backup=False)
        assert repos.commands.list_for_profile(profile_id) == []

    def test_an_older_schema_version_is_refused(self, archive: Path, tmp_path: Path) -> None:
        payload = read_entry(archive, MANIFEST_NAME)
        payload["schema_version"] = 0
        forged = rewrite(
            archive,
            tmp_path / "старый.zip",
            {MANIFEST_NAME: json.dumps(payload, ensure_ascii=False).encode("utf-8")},
        )
        with pytest.raises(ProfileError, match="older than supported"):
            read_manifest(forged)

    def test_a_foreign_zip_is_not_a_profile(self, tmp_path: Path) -> None:
        alien = tmp_path / "чужой.zip"
        with zipfile.ZipFile(alien, "w") as bundle:
            bundle.writestr("readme.txt", "не профиль")
        with pytest.raises(ProfileError, match="no manifest.json"):
            read_manifest(alien)

    def test_a_manifest_with_a_wrong_format_marker_is_refused(
        self, archive: Path, tmp_path: Path
    ) -> None:
        forged = rewrite(
            archive,
            tmp_path / "не-тот.zip",
            {MANIFEST_NAME: b'{"format": "something-else", "schema_version": 1}'},
        )
        with pytest.raises(ProfileError, match="not an Ayris profile bundle"):
            read_manifest(forged)

    def test_a_manifest_without_a_version_is_refused(self, archive: Path, tmp_path: Path) -> None:
        forged = rewrite(
            archive,
            tmp_path / "без-версии.zip",
            {MANIFEST_NAME: json.dumps({"format": BUNDLE_FORMAT}).encode("utf-8")},
        )
        with pytest.raises(ProfileError, match="no usable schema_version"):
            read_manifest(forged)

    def test_a_corrupt_file_is_refused(self, tmp_path: Path) -> None:
        broken = tmp_path / "битый.zip"
        broken.write_bytes(b"PK\x03\x04" + "и дальше мусор".encode())
        with pytest.raises(ProfileError, match="cannot open bundle"):
            read_manifest(broken)

    def test_a_missing_file_is_reported_as_such(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="bundle not found"):
            read_manifest(tmp_path / "нет-такого.zip")

    def test_a_path_traversal_entry_is_refused(self, archive: Path, tmp_path: Path) -> None:
        hostile = tmp_path / "побег.zip"
        with zipfile.ZipFile(archive) as source, zipfile.ZipFile(hostile, "w") as target:
            for name in source.namelist():
                target.writestr(name, source.read(name))
            target.writestr("../../снаружи.txt", "побег")
        with pytest.raises(ProfileError, match="unsafe entry name") as caught:
            read_manifest(hostile)
        assert caught.value.recoverable is False

    def test_malformed_json_inside_a_valid_zip_is_refused(
        self, manager: ProfileManager, archive: Path, tmp_path: Path
    ) -> None:
        forged = rewrite(archive, tmp_path / "мусор.zip", {COMMANDS_NAME: "{не json".encode()})
        with pytest.raises(ProfileError, match="not valid JSON"):
            manager.import_bundle(forged, backup=False)

    def test_a_json_array_where_an_object_belongs_is_refused(
        self, manager: ProfileManager, archive: Path, tmp_path: Path
    ) -> None:
        forged = rewrite(archive, tmp_path / "массив.zip", {VARIABLES_NAME: b"[1, 2, 3]"})
        with pytest.raises(ProfileError, match="expected an object"):
            manager.import_bundle(forged, backup=False)

    def test_a_bundle_from_a_newer_database_only_warns(self, archive: Path, tmp_path: Path) -> None:
        payload = read_entry(archive, MANIFEST_NAME)
        payload["db_schema_version"] = 999
        forged = rewrite(
            archive,
            tmp_path / "новая-бд.zip",
            {MANIFEST_NAME: json.dumps(payload, ensure_ascii=False).encode("utf-8")},
        )
        warnings = read_manifest(forged).compatibility_warnings()
        assert warnings and "новой версией базы данных" in warnings[0]
        assert preview_bundle(forged).warnings == warnings


# ----------------------------------------------------------------------
# rollback
# ----------------------------------------------------------------------


class TestRollback:
    """A bundle that fails half way must leave the profile exactly as it was."""

    @pytest.fixture
    def broken(
        self,
        manager: ProfileManager,
        repos: Repositories,
        paths: paths_module.AppPaths,
        tmp_path: Path,
    ) -> Path:
        """A bundle whose second command has no name, and which carries a sound.

        The nameless entry is deliberately *second*: the importer inserts the
        first one before it fails, so a missing rollback shows up as a leftover
        row rather than as nothing happening at all.
        """
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        (paths.sounds_dir / "звук.wav").write_bytes(FROM_BUNDLE)

        good = tmp_path / "профиль.zip"
        manager.export(good)
        payload = read_entry(good, COMMANDS_NAME)
        payload["commands"].append({"name": "   ", "actions": []})
        payload["commands"].append({"name": "Хвост", "actions": []})
        return rewrite(
            good,
            tmp_path / "битый-профиль.zip",
            {COMMANDS_NAME: json.dumps(payload, ensure_ascii=False).encode("utf-8")},
        )

    def test_database_rows_are_rolled_back(
        self, manager: ProfileManager, repos: Repositories, broken: Path
    ) -> None:
        empty = manager.create("Пустой")
        assert empty.id is not None

        with pytest.raises(ProfileError, match="has no name") as caught:
            manager.import_bundle(broken, profile=empty, backup=False)
        assert "импорт отменён" in caught.value.user_message

        assert repos.commands.list_for_profile(empty.id) == []
        assert [
            folder
            for folder in repos.folders.list_for_profile(empty.id)
            if folder.profile_id == empty.id
        ] == []
        assert repos.variables.list_all(scope=VariableScope.PROFILE, profile_id=empty.id) == []

    def test_files_are_rolled_back(
        self, manager: ProfileManager, paths: paths_module.AppPaths, broken: Path
    ) -> None:
        target = paths.sounds_dir / "звук.wav"
        target.write_bytes(FROM_DISK)

        with pytest.raises(ProfileError):
            manager.import_bundle(broken, policy=ConflictPolicy.OVERWRITE, backup=False)

        assert target.read_bytes() == FROM_DISK

    def test_a_file_the_bundle_added_is_removed_again(
        self, manager: ProfileManager, paths: paths_module.AppPaths, broken: Path
    ) -> None:
        (paths.sounds_dir / "звук.wav").unlink()

        with pytest.raises(ProfileError):
            manager.import_bundle(broken, policy=ConflictPolicy.OVERWRITE, backup=False)

        assert not (paths.sounds_dir / "звук.wav").exists()

    def test_settings_are_rolled_back_too(
        self, manager: ProfileManager, paths: paths_module.AppPaths, broken: Path
    ) -> None:
        paths.config_file.parent.mkdir(parents=True, exist_ok=True)
        paths.config_file.write_bytes(b"# local settings\n")

        with pytest.raises(ProfileError):
            manager.import_bundle(
                broken, policy=ConflictPolicy.OVERWRITE, backup=False, apply_config=True
            )

        assert paths.config_file.read_bytes() == b"# local settings\n"

    def test_the_staging_area_does_not_survive_a_rollback(
        self, manager: ProfileManager, paths: paths_module.AppPaths, broken: Path
    ) -> None:
        with pytest.raises(ProfileError):
            manager.import_bundle(broken, policy=ConflictPolicy.OVERWRITE, backup=False)
        leftovers = list(paths.cache_dir.glob("import_*")) if paths.cache_dir.is_dir() else []
        assert leftovers == []

    def test_a_failed_import_still_leaves_its_backup(
        self, manager: ProfileManager, broken: Path
    ) -> None:
        with pytest.raises(ProfileError):
            manager.import_bundle(broken, backup=True)
        assert manager.list_backups()

    def test_the_untouched_source_profile_is_unaffected(
        self, manager: ProfileManager, repos: Repositories, broken: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        empty = manager.create("Пустой")
        with pytest.raises(ProfileError):
            manager.import_bundle(broken, profile=empty, backup=False)
        assert command_names(repos, source_id) == {"Включи свет", "Заметка"}


# ----------------------------------------------------------------------
# backups and reset
# ----------------------------------------------------------------------


class TestBackups:
    """Backups are ordinary bundles, which is what makes them worth having."""

    def test_a_backup_is_written_before_an_import(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        archive = tmp_path / "профиль.zip"
        manager.export(archive)

        report = manager.import_bundle(archive, policy=ConflictPolicy.OVERWRITE)
        assert report.backup is not None
        assert report.backup.is_file()
        assert report.backup.parent == manager.backups_dir
        assert "import" in report.backup.name
        assert report.backup.name in report.describe()
        # And it is a real bundle, not a placeholder.
        assert set(preview_bundle(report.backup).commands) == {"Включи свет", "Заметка"}

    def test_a_backup_can_be_restored_over_a_reset_profile(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        seed(repos, profile_id)

        snapshot = manager.backup(reason="ручной")
        manager.reset(backup=False)
        assert repos.commands.list_for_profile(profile_id) == []

        manager.import_bundle(snapshot, backup=False)
        assert command_names(repos, profile_id) == {"Включи свет", "Заметка"}
        city = repos.variables.get("город", scope=VariableScope.PROFILE, profile_id=profile_id)
        assert city is not None and city.value == "Москва"

    def test_reset_backs_up_first_and_keeps_the_profile_row(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        seed(repos, profile_id)

        same = manager.reset()
        assert same.id == profile_id
        assert manager.active.id == profile_id
        assert repos.profiles.get(profile_id) is not None
        assert repos.commands.list_for_profile(profile_id) == []
        assert [
            folder
            for folder in repos.folders.list_for_profile(profile_id)
            if folder.profile_id == profile_id
        ] == []
        assert any("reset" in item.name for item in manager.list_backups())

    def test_reset_leaves_other_profiles_and_global_variables_alone(
        self, manager: ProfileManager, repos: Repositories
    ) -> None:
        profile_id = manager.active.id
        assert profile_id is not None
        seed(repos, profile_id)
        other = manager.copy(manager.active, "Второй")
        assert other.id is not None

        manager.reset(backup=False)
        assert command_names(repos, other.id) == {"Включи свет", "Заметка"}
        assert repos.variables.get("порог", scope=VariableScope.GLOBAL) is not None

    def test_backups_are_pruned_to_the_limit(self, manager: ProfileManager) -> None:
        for index in range(MAX_BACKUPS + 3):
            manager.backup(reason=f"прогон{index:02d}")
        assert len(manager.list_backups()) == MAX_BACKUPS

    def test_backups_are_listed_newest_first(self, manager: ProfileManager) -> None:
        first = manager.backup(reason="раз")
        second = manager.backup(reason="два")
        # Same-second timestamps would make mtime order ambiguous, so nudge them.
        import os as _os

        _os.utime(first, (1_000_000, 1_000_000))
        _os.utime(second, (2_000_000, 2_000_000))
        assert manager.list_backups()[0] == second

    def test_listing_backups_before_any_exist_is_empty(self, manager: ProfileManager) -> None:
        assert manager.list_backups() == []

    def test_a_backup_name_survives_a_profile_named_like_a_path(
        self, manager: ProfileManager
    ) -> None:
        renamed = manager.rename(manager.active, "Дом / Работа")
        archive = manager.backup(profile=renamed)
        assert archive.is_file()
        assert "/" not in archive.name


# ----------------------------------------------------------------------
# the profile root
# ----------------------------------------------------------------------


class TestRoot:
    """Moving the installation, which is how the Syncthing use case works."""

    def test_change_root_moves_the_data_and_keeps_working(
        self, manager: ProfileManager, repos: Repositories, tmp_path: Path
    ) -> None:
        source_id = manager.active.id
        assert source_id is not None
        seed(repos, source_id)
        old_root = manager.paths.root

        target = tmp_path / "sync" / "Ayris"
        paths = manager.change_root(target)

        assert paths.root == target.resolve()
        assert manager.paths.root == target.resolve()
        assert paths.database_file.is_file()
        assert not old_root.exists()
        assert paths_module.get_paths().root == target.resolve()

        # The data came along and the manager is usable on the new handle.
        profile_id = manager.active.id
        assert profile_id is not None
        assert command_names(manager.repositories, profile_id) == {"Включи свет", "Заметка"}
        created = manager.create("После переезда")
        assert created.id is not None

    def test_change_root_writes_a_pointer_and_rebinds_the_caller(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        rebound: list[Path] = []
        manager_with_hook = ProfileManager(
            manager.repositories,
            paths=manager.paths,
            on_rebind=lambda paths, _repos: rebound.append(paths.root),
        )
        target = tmp_path / "sync" / "Ayris"
        manager_with_hook.change_root(target)

        assert rebound == [target.resolve()]
        assert paths_module.read_configured_root() == target.resolve()
        assert paths_module.get_paths().source is paths_module.RootSource.EXPLICIT

    def test_change_root_can_keep_the_old_copy(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        old_root = manager.paths.root
        manager.change_root(tmp_path / "sync", remove_source=False)
        assert old_root.is_dir()
        assert (old_root / "ayris.db").is_file()

    def test_change_root_to_the_same_place_is_a_no_op(self, manager: ProfileManager) -> None:
        before = manager.paths
        assert manager.change_root(before.root) is before

    def test_change_root_refuses_a_folder_inside_the_current_one(
        self, manager: ProfileManager
    ) -> None:
        with pytest.raises(ProfileError, match="is inside"):
            manager.change_root(manager.paths.root / "внутри")

    def test_change_root_refuses_a_non_empty_folder(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        occupied = tmp_path / "занято"
        occupied.mkdir()
        (occupied / "чужое.txt").write_text("не трогать", encoding="utf-8")
        with pytest.raises(ProfileError, match="not empty"):
            manager.change_root(occupied)
        assert (occupied / "чужое.txt").is_file()

    def test_change_root_refuses_a_file(self, manager: ProfileManager, tmp_path: Path) -> None:
        blocker = tmp_path / "файл"
        blocker.write_text("я не папка", encoding="utf-8")
        with pytest.raises(ProfileError, match="not a directory"):
            manager.change_root(blocker)

    def test_a_refused_move_leaves_the_database_usable(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        occupied = tmp_path / "занято"
        occupied.mkdir()
        (occupied / "чужое.txt").write_text("не трогать", encoding="utf-8")
        with pytest.raises(ProfileError):
            manager.change_root(occupied)
        assert manager.create("Всё ещё работает").id is not None

    def test_caches_and_logs_stay_behind(self, manager: ProfileManager, tmp_path: Path) -> None:
        manager.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        (manager.paths.logs_dir / "ayris.log").write_text("шум", encoding="utf-8")
        manager.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        (manager.paths.cache_dir / "мусор.tmp").write_text("шум", encoding="utf-8")
        (manager.paths.root / "ayris.lock").write_text("1234", encoding="utf-8")

        paths = manager.change_root(tmp_path / "sync")
        assert not (paths.logs_dir / "ayris.log").exists()
        assert not (paths.cache_dir / "мусор.tmp").exists()
        assert not (paths.root / "ayris.lock").exists()

    def test_moving_back_to_the_default_root_clears_the_pointer(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        """Returning home must work even though the pointer file lives there."""
        manager.change_root(tmp_path / "sync")
        assert paths_module.read_configured_root() is not None

        manager.change_root(paths_module.default_root())
        assert paths_module.read_configured_root() is None
        assert paths_module.resolve_root_with_source() == (
            paths_module.default_root(),
            paths_module.RootSource.DEFAULT,
        )
        assert manager.paths.database_file.is_file()

    def test_open_folder_refuses_a_path_that_is_not_there(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        with pytest.raises(ProfileError, match="not a directory"):
            manager.open_folder(tmp_path / "нет-такой-папки")

    def test_backups_dir_hangs_off_the_current_root(
        self, manager: ProfileManager, tmp_path: Path
    ) -> None:
        manager.backup()
        paths = manager.change_root(tmp_path / "sync")
        assert manager.backups_dir.parent == paths.root
        assert manager.list_backups(), "backups should travel with the root"


# ----------------------------------------------------------------------
# Windows only
# ----------------------------------------------------------------------


@WINDOWS_ONLY
class TestWindowsOnly:
    """Behaviour that has no Linux equivalent, so the sandbox run skips it."""

    def test_open_folder_hands_the_root_to_the_shell(
        self, manager: ProfileManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        opened: list[Path] = []
        monkeypatch.setattr(
            "os.startfile", lambda target: opened.append(Path(target)), raising=False
        )
        assert manager.open_folder() == manager.paths.root
        assert opened == [manager.paths.root]

    def test_open_folder_reports_a_shell_refusal(
        self, manager: ProfileManager, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(_target: object) -> None:
            raise OSError("нет обработчика")

        monkeypatch.setattr("os.startfile", refuse, raising=False)
        with pytest.raises(ProfileError, match="cannot open"):
            manager.open_folder()

    def test_a_file_held_open_cannot_be_replaced_and_the_import_says_so(
        self, manager: ProfileManager, paths: paths_module.AppPaths, tmp_path: Path
    ) -> None:
        paths.sounds_dir.mkdir(parents=True, exist_ok=True)
        target = paths.sounds_dir / "звук.wav"
        target.write_bytes(FROM_BUNDLE)
        archive = tmp_path / "профиль.zip"
        manager.export(archive)

        with target.open("rb"):
            with pytest.raises(ProfileError, match="cannot move") as caught:
                manager.import_bundle(archive, policy=ConflictPolicy.OVERWRITE, backup=False)
            assert "занят другой программой" in caught.value.user_message
        assert target.read_bytes() == FROM_BUNDLE


def test_events_are_frozen_like_the_rest_of_the_bus() -> None:
    """The bus assumes events cannot change after publication; check ours cannot."""
    event = ProfileSwitched(profile=Profile(name="Игры", id=1))
    with pytest.raises((AttributeError, TypeError)):
        event.profile = Profile(name="Другой")  # type: ignore[misc]


def test_manifest_json_round_trips() -> None:
    original = BundleManifest(profile_name="Дом / Работа", counts={"commands": 3})
    restored = BundleManifest.from_json(original.to_json())
    assert restored.profile_name == original.profile_name
    assert restored.schema_version == original.schema_version
    assert restored.counts == original.counts
    assert restored.created_at == original.created_at
