"""Nuitka standalone build for Ayris (task 71, §16.1–16.2).

This composes the exact Nuitka command line that turns the source tree into a
windowed ``ayris.exe`` with its dependencies beside it, then runs the post-build
steps the task asks for: embed the DPI/UAC manifest, optionally lay down the
portable marker, and write an artifact size report (diffed against the previous
build when one is around). A ``--version`` smoke test confirms the binary at least
starts.

Two design decisions drive everything here:

* **Standalone, never onefile.** Onefile unpacks to a temp directory at launch, so
  ``ayris.core.paths.executable_dir()`` — the single seam every resource, model and
  profile path is resolved against — would point at the temp dir, not the folder
  the user sees. Beside-exe resources and the portable variant both depend on the
  executable's *real* directory, which only ``--standalone`` gives.
* **Explicit includes and excludes, no guessing.** Nuitka follows imports, but the
  ones that matter here are invisible to static analysis: engines loaded through a
  string (``openwakeword.model``), backends discovered by entry point
  (``keyring.backends``), and native data shipped beside a wheel's Python
  (PortAudio inside ``sounddevice``). Those are listed by hand below, and so are
  the heavy optional extras (``torch``, ``llama_cpp``, ``playwright``,
  ``paddleocr`` …) that must *not* leak into the shipped dist.

The actual Nuitka call goes through an injected ``runner`` (defaulting to
:func:`subprocess.run`) so the whole composition — arguments, post-steps, report —
is exercised by ``tests/unit/test_build.py`` without ever paying for a half-hour
compile. Nuitka itself is imported lazily (only its presence is checked, via
:func:`importlib.util.find_spec`) so importing this module costs nothing.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from build import build_report, portable_spec, version_info

__all__ = [
    "INCLUDE_PACKAGES",
    "INCLUDE_PACKAGE_DATA",
    "NOFOLLOW_PACKAGES",
    "BuildOptions",
    "BuildResult",
    "build_command",
    "clean_outputs",
    "describe_build",
    "embed_manifest",
    "main",
    "nuitka_arguments",
    "nuitka_available",
    "run_build",
    "smoke_test",
]

#: Repository root — this file is ``<root>/build/nuitka_build.py``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The DPI/UAC manifest embedded after Nuitka (see ``build/README.md``).
MANIFEST = PROJECT_ROOT / "build" / "ayris.manifest"

#: Default place an ``.ico`` is picked up from, if one has been added. The build
#: works without it (Nuitka then uses its default icon); the flag is only passed
#: when the file actually exists, so a missing icon never fails the build.
DEFAULT_ICON = PROJECT_ROOT / "resources" / "icons" / "ayris.ico"

#: Nuitka names its output tree after the compiled file; ``build/ayris.py`` yields
#: these two folders under ``--output-dir``.
_ARTIFACT_DIRS = ("ayris.dist", "ayris.build", "ayris.onefile-build")

#: Qt plugin set. "sensible" is the pyside6 plugin's curated group — platforms
#: (qwindows), styles, imageformats, iconengines, platformthemes — everything a
#: windowed Qt app needs to draw. QtWebEngine (the sphere) is pulled in by the
#: import of ``QtWebEngineWidgets`` and shipped by the plugin, not listed here.
QT_PLUGINS = "sensible"

#: Packages Nuitka would miss because nothing imports them by a static name:
#: the app tree itself (much of it reached through strings and plugins), the
#: keyring Windows backend (found via entry points), its ``win32ctypes`` helper,
#: and ``tzdata`` (zoneinfo finds it only by the distribution being present).
INCLUDE_PACKAGES = (
    "ayris",
    "keyring.backends",
    "win32ctypes",
    "tzdata",
)

#: Packages whose *non-code* payload — a bundled DLL or model — sits beside the
#: Python and would otherwise be left behind. sounddevice carries PortAudio, vosk
#: its libvosk, openwakeword its melspectrogram/embedding ONNX, tzdata the zoneinfo
#: database, pyrnnoise its RNNoise library; the ONNX/CTranslate2/av stacks ship
#: their runtimes as data too.
INCLUDE_PACKAGE_DATA = (
    "sounddevice",
    "vosk",
    "openwakeword",
    "onnxruntime",
    "onnx_asr",
    "ctranslate2",
    "av",
    "piper",
    "tzdata",
    "pyrnnoise",
    "webrtcvad",
)

#: Heavy optional extras that must never reach the main dist. Each is an extra in
#: ``pyproject.toml`` (llm-local, web, ocr, tts-extra, wake-extra, games, cuda) or
#: a training-only transitive of a shipped engine (``torch`` via openwakeword's
#: ``data``/``train``; ``matplotlib`` via the pyrnnoise wrapper). The app imports
#: them lazily, so cutting them here just makes the unshipped feature raise a clear
#: ImportError instead of bloating the download by hundreds of megabytes. NB:
#: ``scipy``/``sklearn`` are deliberately absent — openwakeword's ``__init__`` pulls
#: them on every wake-word load, so they are genuinely runtime, not optional.
NOFOLLOW_PACKAGES = (
    "torch",
    "torchaudio",
    "torchvision",
    "TTS",
    "llama_cpp",
    "playwright",
    "paddle",
    "paddleocr",
    "paddlex",
    "pytesseract",
    "pvporcupine",
    "interception",
    "matplotlib",
    "pandas",
    "nvidia",
    "IPython",
    "tkinter",
    "pytest",
    "_pytest",
)


class Runner(Protocol):
    """The subset of :func:`subprocess.run` the build depends on.

    Injected so tests can capture the composed command lines and simulate Nuitka,
    ``mt.exe`` and the smoke test without spawning anything.
    """

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True, slots=True)
class BuildOptions:
    """Everything the build needs to know, with sane defaults for a local run."""

    #: ``"standalone"`` ships the plain dist; ``"portable"`` also writes the marker
    #: and pre-creates ``models/`` so the folder runs from a flash drive.
    variant: str = "standalone"
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
    #: Where Nuitka writes ``ayris.dist``/``ayris.build``. Under ``dist/`` so a
    #: build never litters the tracked tree (``.gitignore`` already covers it).
    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "dist")
    icon: Path | None = field(default_factory=lambda: DEFAULT_ICON)
    #: ``None`` lets Nuitka pick its own parallelism; a number pins it.
    jobs: int | None = None
    #: Wipe any previous ``ayris.dist``/``ayris.build`` first — on by default so a
    #: build is reproducible and never mixes files from an older compile.
    clean: bool = True
    smoke_test: bool = True

    @property
    def is_portable(self) -> bool:
        return self.variant == "portable"

    @property
    def entry(self) -> Path:
        """The file Nuitka compiles — its name decides the exe's name."""
        return self.project_root / "build" / "ayris.py"

    @property
    def resources_dir(self) -> Path:
        return self.project_root / "resources"

    @property
    def dist_dir(self) -> Path:
        return self.output_dir / "ayris.dist"

    @property
    def exe_path(self) -> Path:
        return self.dist_dir / version_info.EXECUTABLE_NAME


@dataclass(frozen=True, slots=True)
class BuildResult:
    """What a completed build produced, for the summary and the tests."""

    variant: str
    exe_path: Path
    report: build_report.Report
    manifest_embedded: bool
    smoke_ok: bool
    smoke_output: str
    portable_paths: tuple[Path, ...] = ()
    diff: build_report.ReportDiff | None = None

    def summary(self) -> str:
        """Human summary for the build log."""
        lines = [
            f"Собран вариант: {self.variant}",
            f"Исполняемый файл: {self.exe_path}",
            build_report.render_report(self.report),
        ]
        if self.diff is not None:
            lines.append(build_report.render_diff(self.diff))
        lines.append(
            "Манифест внедрён." if self.manifest_embedded else "Манифест не внедрён (нет mt.exe)."
        )
        if self.portable_paths:
            lines.append(portable_spec.describe_layout())
        if self.smoke_output:
            verdict = "ок" if self.smoke_ok else "ПРОВАЛ"
            lines.append(f"Проверка --version: {verdict} — {self.smoke_output}")
        return "\n".join(lines)


def nuitka_arguments(options: BuildOptions) -> list[str]:
    """Compose the full, ordered Nuitka argument list (flags then the entry file).

    The order is fixed on purpose: a stable command line is part of a reproducible
    build. The icon flag is appended only when the ``.ico`` exists, so the list is
    otherwise identical run to run.
    """
    args = [
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        f"--include-qt-plugins={QT_PLUGINS}",
        "--windows-console-mode=disable",
        *version_info.nuitka_version_arguments(),
        f"--output-dir={options.output_dir}",
        f"--output-filename={version_info.EXECUTABLE_NAME}",
        f"--include-data-dir={options.resources_dir}=resources",
        *(f"--include-package={name}" for name in INCLUDE_PACKAGES),
        *(f"--include-package-data={name}" for name in INCLUDE_PACKAGE_DATA),
        *(f"--nofollow-import-to={name}" for name in NOFOLLOW_PACKAGES),
    ]
    if options.icon is not None and options.icon.is_file():
        args.append(f"--windows-icon-from-ico={options.icon}")
    if options.jobs is not None:
        args.append(f"--jobs={options.jobs}")
    args.append(str(options.entry))
    return args


def build_command(options: BuildOptions) -> list[str]:
    """The process invocation: the current interpreter running Nuitka as a module.

    Using ``sys.executable -m nuitka`` (not a bare ``nuitka``) guarantees the build
    runs against the very environment whose pinned dependencies it is packaging.
    """
    return [sys.executable, "-m", "nuitka", *nuitka_arguments(options)]


def nuitka_available() -> bool:
    """Whether Nuitka is importable, without importing it. Guards the real build."""
    return importlib.util.find_spec("nuitka") is not None


def clean_outputs(options: BuildOptions) -> list[Path]:
    """Remove any previous Nuitka output under ``output_dir``. Returns what it removed."""
    removed: list[Path] = []
    for name in _ARTIFACT_DIRS:
        target = options.output_dir / name
        if target.exists():
            shutil.rmtree(target)
            removed.append(target)
    return removed


def embed_manifest(exe: Path, manifest: Path, runner: Runner) -> bool:
    """Embed the DPI/UAC manifest into ``exe`` with ``mt.exe`` (Windows SDK).

    Best-effort by design: ``mt.exe`` is not always on PATH (it ships with the
    Windows SDK), and ``utils/dpi.py`` sets per-monitor awareness at runtime as a
    fallback. So a missing tool is a warning, not a failed build.
    """
    tool = shutil.which("mt.exe") or shutil.which("mt")
    if tool is None or not manifest.is_file():
        return False
    result = runner(
        [tool, "-nologo", "-manifest", str(manifest), f"-outputresource:{exe};#1"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def smoke_test(exe: Path, runner: Runner) -> tuple[bool, str]:
    """Run ``exe --version`` and confirm it prints the expected version.

    ``--version`` returns before any Qt import or disk touch, so this is a cheap
    "does the binary even start" check — the one the release workflow runs on the
    fresh runner.
    """
    result = runner([str(exe), "--version"], check=False, capture_output=True, text=True)
    output = f"{result.stdout or ''}{result.stderr or ''}".strip()
    ok = result.returncode == 0 and version_info.VERSION in output
    return ok, output


def describe_build(options: BuildOptions) -> str:
    """A dry-run description: what will be built and the exact command line."""
    icon = (
        str(options.icon)
        if options.icon is not None and options.icon.is_file()
        else "по умолчанию Nuitka"
    )
    lines = [
        f"Вариант: {options.variant}",
        f"Точка входа: {options.entry}",
        f"Каталог вывода: {options.output_dir}",
        f"Ресурсы: {options.resources_dir} → resources/",
        f"Иконка: {icon}",
        f"Включаемые пакеты: {', '.join(INCLUDE_PACKAGES)}",
        f"Данные пакетов: {', '.join(INCLUDE_PACKAGE_DATA)}",
        f"Исключаемые (не в дистрибутив): {', '.join(NOFOLLOW_PACKAGES)}",
        "",
        "Команда:",
        "  " + " ".join(build_command(options)),
    ]
    return "\n".join(lines)


def _report_path(options: BuildOptions) -> Path:
    return options.output_dir / "build-report.json"


def run_build(
    options: BuildOptions,
    runner: Runner = subprocess.run,
    *,
    previous_report: Path | None = None,
) -> BuildResult:
    """Run the whole build and its post-steps, returning a :class:`BuildResult`.

    Raises:
        RuntimeError: Nuitka exited non-zero, or finished without producing the exe.
    """
    # Read the previous manifest before we overwrite it, so the size diff compares
    # against the last build. An explicit path wins; otherwise the last report here.
    prior = previous_report if previous_report is not None else _report_path(options)
    previous = build_report.read_report(prior) if prior.is_file() else None

    if options.clean:
        clean_outputs(options)
    options.output_dir.mkdir(parents=True, exist_ok=True)

    result = runner(build_command(options), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Nuitka завершился с кодом {result.returncode}")
    if not options.exe_path.is_file():
        raise RuntimeError(f"Nuitka отработал, но исполняемый файл не найден: {options.exe_path}")

    manifest_embedded = embed_manifest(options.exe_path, MANIFEST, runner)

    portable_paths: tuple[Path, ...] = ()
    if options.is_portable:
        portable_paths = tuple(portable_spec.make_portable(options.dist_dir))

    report = build_report.make_report(options.dist_dir)
    build_report.write_report(report, _report_path(options))
    diff = build_report.diff_reports(previous, report) if previous is not None else None

    smoke_ok, smoke_output = (True, "")
    if options.smoke_test:
        smoke_ok, smoke_output = smoke_test(options.exe_path, runner)

    return BuildResult(
        variant=options.variant,
        exe_path=options.exe_path,
        report=report,
        manifest_embedded=manifest_embedded,
        smoke_ok=smoke_ok,
        smoke_output=smoke_output,
        portable_paths=portable_paths,
        diff=diff,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build.nuitka_build",
        description="Сборка ayris.exe через Nuitka (standalone / портабельный вариант).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="портабельный вариант: маркер и папка models/ рядом с exe",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="куда писать ayris.dist/ayris.build (по умолчанию dist/)",
    )
    parser.add_argument(
        "--icon",
        type=Path,
        default=None,
        metavar="ICO",
        help="иконка exe (по умолчанию resources/icons/ayris.ico, если есть)",
    )
    parser.add_argument("--jobs", type=int, default=None, help="число параллельных заданий Nuitka")
    parser.add_argument("--no-clean", action="store_true", help="не удалять прошлую сборку")
    parser.add_argument("--skip-smoke-test", action="store_true", help="не запускать exe --version")
    parser.add_argument(
        "--previous-report",
        type=Path,
        default=None,
        metavar="JSON",
        help="манифест прошлой сборки для диффа размера",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="только показать команду и состав, ничего не собирать",
    )
    return parser


def _options_from_args(namespace: argparse.Namespace) -> BuildOptions:
    kwargs: dict[str, object] = {
        "variant": "portable" if namespace.portable else "standalone",
        "jobs": namespace.jobs,
        "clean": not namespace.no_clean,
        "smoke_test": not namespace.skip_smoke_test,
    }
    if namespace.output_dir is not None:
        kwargs["output_dir"] = namespace.output_dir
    if namespace.icon is not None:
        kwargs["icon"] = namespace.icon
    return BuildOptions(**kwargs)  # type: ignore[arg-type]


def main(argv: list[str] | None = None, runner: Runner = subprocess.run) -> int:
    """CLI entry: ``python -m build.nuitka_build [--portable] [--dry-run] …``."""
    namespace = _build_parser().parse_args(argv)
    options = _options_from_args(namespace)

    if namespace.dry_run:
        sys.stdout.write(describe_build(options) + "\n")
        return 0

    if not nuitka_available():
        sys.stderr.write(
            "Nuitka не установлен. Поставьте сборочную зависимость: pip install -e .[build]\n"
        )
        return 2

    try:
        result = run_build(options, runner, previous_report=namespace.previous_report)
    except RuntimeError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    sys.stdout.write(result.summary() + "\n")
    return 0 if result.smoke_ok else 1


if __name__ == "__main__":
    sys.exit(main())
