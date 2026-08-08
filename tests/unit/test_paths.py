"""Task 02: profile root resolution, portable mode and the directory layout.

The smoke tests from task 01 already cover the happy path; these cover the rules
that decide *where* the profile lives, because getting that wrong silently
splits a user's settings across two folders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.core.errors import ConfigError
from ayris.core.paths import (
    POINTER_FILE_NAME,
    PORTABLE_ENV_VAR,
    PORTABLE_MARKERS,
    PROFILE_ENV_VAR,
    AppPaths,
    ModelKind,
    RootSource,
    clear_configured_root,
    default_root,
    init_paths,
    read_configured_root,
    resolve_root,
    resolve_root_with_source,
    write_configured_root,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def fake_exe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pretend the frozen executable lives in its own directory."""
    exe_dir = tmp_path / "app"
    exe_dir.mkdir()
    monkeypatch.setattr("ayris.core.paths.executable_dir", lambda: exe_dir)
    return exe_dir


class TestPrecedence:
    """The order in the module docstring, one test per rule that can win."""

    def test_explicit_profile_beats_everything(
        self, tmp_path: Path, fake_exe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (fake_exe / PORTABLE_MARKERS[0]).write_text("", encoding="utf-8")
        monkeypatch.setenv(PROFILE_ENV_VAR, str(tmp_path / "from_env"))

        root, source = resolve_root_with_source(portable=True, profile=tmp_path / "chosen")

        assert root == (tmp_path / "chosen").resolve()
        assert source is RootSource.EXPLICIT

    def test_portable_flag_beats_the_environment(
        self, fake_exe: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PROFILE_ENV_VAR, str(tmp_path / "from_env"))

        root, source = resolve_root_with_source(portable=True)

        assert root == (fake_exe / "profile").resolve()
        assert source is RootSource.PORTABLE_FLAG

    def test_profile_environment_variable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PROFILE_ENV_VAR, str(tmp_path / "from_env"))

        root, source = resolve_root_with_source()

        assert root == (tmp_path / "from_env").resolve()
        assert source is RootSource.ENVIRONMENT

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on", "да"])
    def test_portable_environment_variable(
        self, value: str, fake_exe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PORTABLE_ENV_VAR, value)

        root, source = resolve_root_with_source()

        assert root == (fake_exe / "profile").resolve()
        assert source is RootSource.PORTABLE_ENV

    def test_portable_variable_set_to_zero_is_ignored(
        self, fake_exe: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PORTABLE_ENV_VAR, "0")

        _root, source = resolve_root_with_source()

        assert source is RootSource.DEFAULT

    @pytest.mark.parametrize("marker", PORTABLE_MARKERS)
    def test_marker_file_switches_on_portable_mode(self, marker: str, fake_exe: Path) -> None:
        (fake_exe / marker).write_text("", encoding="utf-8")

        root, source = resolve_root_with_source()

        assert root == (fake_exe / "profile").resolve()
        assert source is RootSource.PORTABLE_MARKER

    def test_marker_file_can_redirect_the_root(self, fake_exe: Path, tmp_path: Path) -> None:
        target = tmp_path / "stick" / "ayris"
        (fake_exe / PORTABLE_MARKERS[0]).write_text(str(target), encoding="utf-8")

        root, source = resolve_root_with_source()

        assert root == target.resolve()
        assert source is RootSource.PORTABLE_MARKER

    def test_marker_redirect_is_relative_to_the_executable(self, fake_exe: Path) -> None:
        """A USB stick gets a different drive letter every time it is plugged in."""
        (fake_exe / PORTABLE_MARKERS[0]).write_text("data/ayris", encoding="utf-8")

        root, _source = resolve_root_with_source()

        assert root == (fake_exe / "data" / "ayris").resolve()

    def test_marker_written_by_notepad_with_a_bom(self, fake_exe: Path, tmp_path: Path) -> None:
        target = tmp_path / "from_notepad"
        (fake_exe / PORTABLE_MARKERS[0]).write_text(str(target), encoding="utf-8-sig")

        root, _source = resolve_root_with_source()

        assert root == target.resolve()

    def test_configured_pointer_is_used_when_nothing_else_applies(
        self, tmp_path: Path, fake_exe: Path
    ) -> None:
        del fake_exe
        chosen = tmp_path / "elsewhere" / "Ayris"
        write_configured_root(chosen)

        root, source = resolve_root_with_source()

        assert root == chosen.resolve()
        assert source is RootSource.CONFIGURED

    def test_default_is_appdata(self, tmp_path: Path, fake_exe: Path) -> None:
        del fake_exe
        root, source = resolve_root_with_source()

        assert root == default_root()
        assert root.parent == tmp_path / "AppData"
        assert source is RootSource.DEFAULT


class TestConfiguredRoot:
    def test_write_read_clear_round_trip(self, tmp_path: Path) -> None:
        chosen = tmp_path / "profiles" / "work"

        written = write_configured_root(chosen)

        assert written == chosen.resolve()
        assert read_configured_root() == chosen.resolve()
        assert (default_root() / POINTER_FILE_NAME).is_file()

        clear_configured_root()
        assert read_configured_root() is None

    def test_clearing_twice_is_not_an_error(self) -> None:
        clear_configured_root()
        clear_configured_root()

    def test_an_empty_pointer_file_is_ignored(self, fake_exe: Path) -> None:
        del fake_exe
        pointer = default_root() / POINTER_FILE_NAME
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text("   \n", encoding="utf-8")

        assert read_configured_root() is None
        assert resolve_root_with_source()[1] is RootSource.DEFAULT


class TestLayout:
    def test_every_directory_is_created_under_the_root(self, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "profile")

        for directory in paths.all_directories():
            assert directory.is_dir()
            assert directory == paths.root or paths.root in directory.parents

    def test_file_locations(self, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "profile")

        assert paths.config_file == paths.root / "config.toml"
        assert paths.broken_config_file == paths.root / "config.toml.broken"
        assert paths.database_file == paths.root / "ayris.db"
        assert paths.log_file("20260808") == paths.logs_dir / "ayris_20260808.log"

    @pytest.mark.parametrize("kind", ["stt", "tts", "wake", "llm"])
    def test_model_directories(self, kind: ModelKind, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "profile")

        assert paths.model_dir(kind) == paths.models_dir / kind
        assert paths.model_dir(kind).is_dir()

    def test_init_is_idempotent(self, tmp_path: Path) -> None:
        first = init_paths(profile=tmp_path / "profile")
        (first.config_file).write_text("keep me", encoding="utf-8")

        second = init_paths(profile=tmp_path / "profile")

        assert second.root == first.root
        assert second.config_file.read_text(encoding="utf-8") == "keep me"

    def test_creation_can_be_skipped(self, tmp_path: Path) -> None:
        paths = init_paths(profile=tmp_path / "dry", create=False)

        assert not paths.root.exists()


class TestPortableFlag:
    def test_is_portable_is_true_only_for_portable_sources(self, tmp_path: Path) -> None:
        assert AppPaths(root=tmp_path, source=RootSource.PORTABLE_FLAG).is_portable
        assert AppPaths(root=tmp_path, source=RootSource.PORTABLE_ENV).is_portable
        assert AppPaths(root=tmp_path, source=RootSource.PORTABLE_MARKER).is_portable
        assert not AppPaths(root=tmp_path, source=RootSource.DEFAULT).is_portable
        assert not AppPaths(root=tmp_path, source=RootSource.EXPLICIT).is_portable

    def test_every_source_has_a_russian_label(self, tmp_path: Path) -> None:
        for source in RootSource:
            label = AppPaths(root=tmp_path, source=source).source_label
            assert label
            assert label == label.strip()

    def test_paths_are_hashable_and_comparable(self, tmp_path: Path) -> None:
        """Frozen dataclass: safe to stash in a set or compare in a test."""
        first = AppPaths(root=tmp_path)
        second = AppPaths(root=tmp_path)

        assert first == second
        assert len({first, second}) == 1


class TestFailures:
    def test_resolve_root_matches_resolve_root_with_source(self, tmp_path: Path) -> None:
        assert (
            resolve_root(profile=tmp_path / "p")
            == resolve_root_with_source(profile=tmp_path / "p")[0]
        )

    def test_write_configured_root_rejects_a_bad_path(self) -> None:
        with pytest.raises(ConfigError):
            write_configured_root("\0invalid")

    def test_unwritable_root_reports_the_directory(self, tmp_path: Path) -> None:
        def deny(*_args: object, **_kwargs: object) -> None:
            raise OSError("access denied")

        paths = AppPaths(root=tmp_path / "denied")
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "mkdir", deny)
            with pytest.raises(ConfigError) as caught:
                paths.ensure_directories()

        assert caught.value.recoverable is False
        assert str(paths.root) in caught.value.user_message
