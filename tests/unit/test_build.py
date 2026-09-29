"""Tests for the task 71 distribution build (``build/*``).

The whole point of the ``build`` package is that it is ordinary, importable Python
with the one slow, environment-heavy step — the Nuitka compile — injected as a
``runner``. So these tests exercise every decision the build makes (which packages
are included, which extras are cut, the version resource, the portable marker, the
size report and its diff) by handing ``run_build`` a fake runner that records the
command lines and simulates Nuitka, ``mt.exe`` and the ``--version`` smoke test —
without ever spawning a compiler.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from build import build_report, portable_spec, version_info
from build.nuitka_build import (
    INCLUDE_PACKAGES,
    NOFOLLOW_PACKAGES,
    BuildOptions,
    build_command,
    clean_outputs,
    describe_build,
    embed_manifest,
    main,
    nuitka_arguments,
    run_build,
    smoke_test,
)


class FakeRunner:
    """Records every command and fakes Nuitka / mt.exe / the smoke test.

    Nuitka "produces" the exe and a couple of data files under ``dist_dir`` so the
    post-build steps (report, portable marker, smoke test) have a real tree to walk.
    """

    def __init__(
        self,
        dist_dir: Path,
        *,
        build_rc: int = 0,
        smoke_rc: int = 0,
        smoke_out: str | None = None,
        make_exe: bool = True,
    ) -> None:
        self.dist_dir = dist_dir
        self.build_rc = build_rc
        self.smoke_rc = smoke_rc
        self.smoke_out = smoke_out
        self.make_exe = make_exe
        self.calls: list[list[str]] = []

    # PLACEHOLDER_CALL
    def __call__(self, args, **kwargs: object) -> subprocess.CompletedProcess[str]:
        argv = list(args)
        self.calls.append(argv)
        if "nuitka" in argv:
            if self.build_rc == 0 and self.make_exe:
                self.dist_dir.mkdir(parents=True, exist_ok=True)
                (self.dist_dir / version_info.EXECUTABLE_NAME).write_bytes(b"MZ" + b"\0" * 4094)
                (self.dist_dir / "python312.dll").write_bytes(b"\0" * 2048)
                (self.dist_dir / "resources").mkdir(exist_ok=True)
                (self.dist_dir / "resources" / "themes.json").write_text("{}", encoding="utf-8")
            return subprocess.CompletedProcess(argv, self.build_rc)
        if argv and argv[-1] == "--version":
            out = self.smoke_out if self.smoke_out is not None else f"Ayris {version_info.VERSION}"
            return subprocess.CompletedProcess(argv, self.smoke_rc, stdout=out, stderr="")
        # mt.exe manifest embed, or anything else.
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


@pytest.fixture
def options(tmp_path: Path) -> BuildOptions:
    """A build writing into an isolated temp dir, smoke test on, icon absent."""
    return BuildOptions(output_dir=tmp_path / "out", icon=tmp_path / "missing.ico")


# PLACEHOLDER_TESTS
@pytest.mark.unit
def test_build_is_standalone_never_onefile(options: BuildOptions) -> None:
    args = nuitka_arguments(options)
    assert "--standalone" in args
    assert not any(a.startswith("--onefile") for a in args), (
        "onefile распаковывается во временную папку — executable_dir() уедет от неё, "
        "и ресурсы/модели/профиль рядом с exe перестанут находиться"
    )


@pytest.mark.unit
def test_console_is_disabled(options: BuildOptions) -> None:
    assert "--windows-console-mode=disable" in nuitka_arguments(options)


@pytest.mark.unit
def test_pyside6_plugin_and_qt_plugins_enabled(options: BuildOptions) -> None:
    args = nuitka_arguments(options)
    assert "--enable-plugin=pyside6" in args
    assert "--include-qt-plugins=sensible" in args


@pytest.mark.unit
def test_resources_are_shipped_beside_exe(options: BuildOptions) -> None:
    data_dirs = [a for a in nuitka_arguments(options) if a.startswith("--include-data-dir=")]
    assert len(data_dirs) == 1
    assert data_dirs[0].endswith("=resources"), data_dirs[0]


@pytest.mark.unit
def test_version_resource_is_stamped(options: BuildOptions) -> None:
    joined = " ".join(nuitka_arguments(options))
    meta = version_info.metadata()
    assert f"--product-name={meta['app_name']}" in joined
    assert f"--company-name={meta['company']}" in joined
    assert f"--file-version={meta['file_version']}" in joined
    assert f"--product-version={meta['product_version']}" in joined or (
        f"--product-version={meta['file_version']}" in joined
    )
    assert f"--copyright={meta['copyright']}" in joined


@pytest.mark.unit
def test_entry_is_ayris_py_and_last(options: BuildOptions) -> None:
    args = nuitka_arguments(options)
    assert args[-1].replace("\\", "/").endswith("build/ayris.py")


@pytest.mark.unit
def test_explicit_includes_present(options: BuildOptions) -> None:
    args = nuitka_arguments(options)
    for package in INCLUDE_PACKAGES:
        assert f"--include-package={package}" in args
    assert "--include-package=keyring.backends" in args, "keyring находит бэкенд по entry point"
    assert "--include-package-data=sounddevice" in args, "в sounddevice лежит PortAudio DLL"


# PLACEHOLDER_TESTS2
@pytest.mark.unit
@pytest.mark.parametrize("extra", ["llama_cpp", "playwright", "paddleocr", "torch", "TTS"])
def test_heavy_optional_extras_do_not_leak(options: BuildOptions, extra: str) -> None:
    assert f"--nofollow-import-to={extra}" in nuitka_arguments(options)


@pytest.mark.unit
@pytest.mark.parametrize("runtime_dep", ["scipy", "sklearn"])
def test_wake_word_runtime_deps_are_not_excluded(runtime_dep: str) -> None:
    """openwakeword's ``__init__`` imports scipy+sklearn on every load, so cutting
    them would break the wake word — a regression easy to introduce while trimming
    size."""
    assert runtime_dep not in NOFOLLOW_PACKAGES


@pytest.mark.unit
def test_build_command_runs_nuitka_in_this_interpreter(options: BuildOptions) -> None:
    command = build_command(options)
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "nuitka"]


@pytest.mark.unit
def test_icon_flag_only_when_the_file_exists(tmp_path: Path) -> None:
    without = BuildOptions(output_dir=tmp_path, icon=tmp_path / "nope.ico")
    assert not any(a.startswith("--windows-icon-from-ico=") for a in nuitka_arguments(without))

    icon = tmp_path / "ayris.ico"
    icon.write_bytes(b"\0")
    with_icon = BuildOptions(output_dir=tmp_path, icon=icon)
    assert any(a.startswith("--windows-icon-from-ico=") for a in nuitka_arguments(with_icon))


@pytest.mark.unit
def test_jobs_flag_is_passed_when_set(tmp_path: Path) -> None:
    with_jobs = nuitka_arguments(BuildOptions(output_dir=tmp_path, jobs=3))
    assert "--jobs=3" in with_jobs
    without_jobs = nuitka_arguments(BuildOptions(output_dir=tmp_path))
    assert not any(a.startswith("--jobs=") for a in without_jobs)


@pytest.mark.unit
def test_clean_outputs_removes_previous_artifacts(options: BuildOptions) -> None:
    for name in ("ayris.dist", "ayris.build", "ayris.onefile-build"):
        (options.output_dir / name).mkdir(parents=True)
        (options.output_dir / name / "stale.txt").write_text("x", encoding="utf-8")
    removed = clean_outputs(options)
    assert len(removed) == 3
    assert not options.dist_dir.exists()


# PLACEHOLDER_TESTS3
@pytest.mark.unit
def test_run_build_happy_path_produces_report_and_passes_smoke(options: BuildOptions) -> None:
    runner = FakeRunner(options.dist_dir)
    result = run_build(options, runner)
    assert result.exe_path.is_file()
    assert result.report.file_count >= 2
    assert result.smoke_ok
    assert result.diff is None  # первая сборка — сравнивать не с чем
    # ровно один вызов Nuitka и один smoke test
    assert sum("nuitka" in call for call in runner.calls) == 1
    assert any(call and call[-1] == "--version" for call in runner.calls)


@pytest.mark.unit
def test_run_build_writes_report_json_that_diffs_next_time(options: BuildOptions) -> None:
    run_build(options, FakeRunner(options.dist_dir))
    assert (options.output_dir / "build-report.json").is_file()
    # вторая сборка в тот же каталог обязана сравниться с прошлым манифестом
    second = run_build(options, FakeRunner(options.dist_dir))
    assert second.diff is not None


@pytest.mark.unit
def test_run_build_raises_when_nuitka_fails(options: BuildOptions) -> None:
    runner = FakeRunner(options.dist_dir, build_rc=1)
    with pytest.raises(RuntimeError, match="кодом 1"):
        run_build(options, runner)


@pytest.mark.unit
def test_run_build_raises_when_exe_is_missing(options: BuildOptions) -> None:
    runner = FakeRunner(options.dist_dir, make_exe=False)
    with pytest.raises(RuntimeError, match="не найден"):
        run_build(options, runner)


@pytest.mark.unit
def test_smoke_test_fails_on_wrong_version(tmp_path: Path) -> None:
    exe = tmp_path / "ayris.exe"
    exe.write_bytes(b"\0")
    runner = FakeRunner(tmp_path, smoke_out="Ayris 9.9.9")
    ok, output = smoke_test(exe, runner)
    assert not ok
    assert "9.9.9" in output


@pytest.mark.unit
def test_embed_manifest_invokes_mt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exe = tmp_path / "ayris.exe"
    exe.write_bytes(b"\0")
    manifest = tmp_path / "ayris.manifest"
    manifest.write_text("<assembly/>", encoding="utf-8")
    monkeypatch.setattr("build.nuitka_build.shutil.which", lambda _tool: "mt.exe")
    runner = FakeRunner(tmp_path)
    assert embed_manifest(exe, manifest, runner)
    mt_calls = [call for call in runner.calls if any("-outputresource" in a for a in call)]
    assert len(mt_calls) == 1


@pytest.mark.unit
def test_embed_manifest_is_soft_when_mt_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "ayris.exe"
    exe.write_bytes(b"\0")
    manifest = tmp_path / "ayris.manifest"
    manifest.write_text("<assembly/>", encoding="utf-8")
    monkeypatch.setattr("build.nuitka_build.shutil.which", lambda _tool: None)
    assert not embed_manifest(exe, manifest, FakeRunner(tmp_path))


# PLACEHOLDER_TESTS4
@pytest.mark.unit
def test_portable_build_lays_down_marker_and_models(tmp_path: Path) -> None:
    options = BuildOptions(
        variant="portable", output_dir=tmp_path / "out", icon=tmp_path / "none.ico"
    )
    result = run_build(options, FakeRunner(options.dist_dir))
    marker = options.dist_dir / portable_spec.PORTABLE_MARKER_NAME
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == portable_spec.PORTABLE_MARKER_CONTENT
    for kind in portable_spec.MODEL_SUBDIRS:
        assert (options.dist_dir / "models" / kind).is_dir()
    assert result.portable_paths


@pytest.mark.unit
def test_standalone_build_has_no_marker(options: BuildOptions) -> None:
    run_build(options, FakeRunner(options.dist_dir))
    assert not (options.dist_dir / portable_spec.PORTABLE_MARKER_NAME).exists()


@pytest.mark.unit
def test_main_dry_run_does_not_touch_the_runner(tmp_path: Path) -> None:
    runner = FakeRunner(tmp_path)
    code = main(["--dry-run", "--output-dir", str(tmp_path)], runner=runner)
    assert code == 0
    assert runner.calls == []


@pytest.mark.unit
def test_main_reports_missing_nuitka(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("build.nuitka_build.nuitka_available", lambda: False)
    runner = FakeRunner(tmp_path)
    code = main(["--output-dir", str(tmp_path)], runner=runner)
    assert code == 2
    assert runner.calls == []


@pytest.mark.unit
def test_main_happy_path_returns_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    monkeypatch.setattr("build.nuitka_build.nuitka_available", lambda: True)
    runner = FakeRunner(out / "ayris.dist")
    code = main(["--output-dir", str(out)], runner=runner)
    assert code == 0


@pytest.mark.unit
def test_describe_build_shows_command_and_excludes(options: BuildOptions) -> None:
    text = describe_build(options)
    assert "Команда:" in text
    assert "playwright" in text


# PLACEHOLDER_TESTS5
@pytest.mark.unit
def test_version_is_read_from_the_single_source() -> None:
    from ayris import __app_name__, __version__

    assert __version__ == version_info.VERSION
    assert __app_name__ == version_info.APP_NAME


@pytest.mark.unit
def test_version_tuple_pads_to_four_windows_fields() -> None:
    assert len(version_info.version_tuple()) == 4
    assert version_info.numeric_version().count(".") == 3


@pytest.mark.unit
def test_metadata_has_every_resource_field() -> None:
    meta = version_info.metadata()
    assert set(meta) == {
        "app_name",
        "company",
        "product_version",
        "file_version",
        "description",
        "copyright",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("num", "text"),
    [(0, "0 B"), (512, "512 B"), (1536, "1.5 KiB"), (1024 * 1024, "1.0 MiB")],
)
def test_human_size(num: int, text: str) -> None:
    assert build_report.human_size(num) == text


@pytest.mark.unit
def test_report_roundtrips_through_json(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\0" * 10)
    report = build_report.make_report(tmp_path)
    assert report.file_count == 2
    assert report.total_bytes == 15

    path = tmp_path / "report.json"
    build_report.write_report(report, path)
    assert build_report.read_report(path) == report


@pytest.mark.unit
def test_diff_reports_flags_added_removed_and_changed() -> None:
    old = build_report.Report(
        total_bytes=30,
        file_count=2,
        files=(
            build_report.FileEntry("keep.txt", 10),
            build_report.FileEntry("gone.txt", 20),
        ),
    )
    new = build_report.Report(
        total_bytes=40,
        file_count=2,
        files=(
            build_report.FileEntry("keep.txt", 15),
            build_report.FileEntry("new.txt", 25),
        ),
    )
    diff = build_report.diff_reports(old, new)
    assert [entry.path for entry in diff.added] == ["new.txt"]
    assert [entry.path for entry in diff.removed] == ["gone.txt"]
    assert diff.changed == (("keep.txt", 10, 15),)
    assert diff.total_delta == 10


@pytest.mark.unit
def test_portable_paths_match_what_make_portable_creates(tmp_path: Path) -> None:
    expected = portable_spec.portable_paths(tmp_path)
    created = portable_spec.make_portable(tmp_path)
    assert set(created) == set(expected)
    assert all(path.exists() for path in created)


@pytest.mark.unit
def test_make_portable_is_idempotent(tmp_path: Path) -> None:
    portable_spec.make_portable(tmp_path)
    (tmp_path / "models" / "stt" / "user.onnx").write_bytes(b"\0")
    portable_spec.make_portable(tmp_path)
    assert (
        tmp_path / "models" / "stt" / "user.onnx"
    ).is_file(), "второй прогон не должен стирать данные"


@pytest.mark.unit
def test_make_portable_needs_an_existing_dist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        portable_spec.make_portable(tmp_path / "does-not-exist")
