"""Single source of truth for every filesystem path Ayris uses.

Architecture invariant: no module hardcodes ``%APPDATA%`` or builds profile paths
by hand. Everything goes through :func:`get_paths`, which makes the portable
build, the ``--profile`` override and the user-configurable profile location
work without touching call sites.

Layout of a profile root::

    <root>/
      config.toml
      config.toml.broken      # only after a corrupted file was rescued
      ayris.db
      cache/
      logs/
      models/{stt,tts,wake,llm}/
      plugins/
      sounds/
      screenshots/
      themes/

Where that root lives is decided by :func:`resolve_root_with_source`, in
descending priority:

1. ``profile`` argument — the ``--profile`` command line option.
2. ``portable`` argument — the ``--portable`` command line option.
3. ``AYRIS_PROFILE`` environment variable.
4. ``AYRIS_PORTABLE`` environment variable.
5. A portable marker file next to the executable (how the portable build ships).
6. The pointer file written by :func:`write_configured_root` — section 17 of the
   specification requires the profile location to be user-configurable.
7. ``%APPDATA%\\Ayris``.

The pointer file lives in the default root rather than in ``config.toml``:
the profile location has to be known *before* the configuration file can be
found, so it cannot be stored inside it.

This module deliberately imports nothing from :mod:`ayris.utils.logger` —
logging resolves its own directory through :func:`get_paths`, so a dependency in
that direction would be circular. Failures are reported through
:class:`~ayris.core.errors.ConfigError` and the :class:`RootSource` returned
alongside the root, which the caller logs once at startup.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

from ayris.core.errors import ConfigError

__all__ = [
    "APP_DIR_NAME",
    "POINTER_FILE_NAME",
    "PORTABLE_ENV_VAR",
    "PORTABLE_MARKERS",
    "PROFILE_ENV_VAR",
    "ROOT_SOURCE_LABELS",
    "AppPaths",
    "ModelKind",
    "RootSource",
    "clear_configured_root",
    "default_root",
    "executable_dir",
    "get_paths",
    "init_paths",
    "read_configured_root",
    "reset_paths",
    "resolve_root",
    "resolve_root_with_source",
    "write_configured_root",
]

APP_DIR_NAME: Final = "Ayris"

PROFILE_ENV_VAR: Final = "AYRIS_PROFILE"
PORTABLE_ENV_VAR: Final = "AYRIS_PORTABLE"

#: Presence of any of these next to the executable switches on portable mode.
#: An empty marker means "profile beside the executable"; a marker containing a
#: path (absolute, or relative to the executable) redirects the root there.
PORTABLE_MARKERS: Final = ("ayris.portable", "portable.txt")

#: Written into the *default* root to redirect the profile somewhere else.
POINTER_FILE_NAME: Final = "profile_path.txt"

PORTABLE_DIR_NAME: Final = "profile"

_TRUTHY: Final = frozenset({"1", "true", "yes", "on", "да"})

ModelKind = Literal["stt", "tts", "wake", "llm"]

_paths: AppPaths | None = None


class RootSource(StrEnum):
    """Why the profile root ended up where it did. Logged once at startup."""

    EXPLICIT = "explicit"
    PORTABLE_FLAG = "portable_flag"
    ENVIRONMENT = "environment"
    PORTABLE_ENV = "portable_env"
    PORTABLE_MARKER = "portable_marker"
    CONFIGURED = "configured"
    DEFAULT = "default"


#: Russian labels for the profile tab and the startup log line.
ROOT_SOURCE_LABELS: Final[MappingProxyType[RootSource, str]] = MappingProxyType(
    {
        RootSource.EXPLICIT: "указан ключом --profile",
        RootSource.PORTABLE_FLAG: "портабельный режим (--portable)",
        RootSource.ENVIRONMENT: f"переменная окружения {PROFILE_ENV_VAR}",
        RootSource.PORTABLE_ENV: f"переменная окружения {PORTABLE_ENV_VAR}",
        RootSource.PORTABLE_MARKER: "портабельный режим (файл-маркер рядом с программой)",
        RootSource.CONFIGURED: "путь выбран в настройках",
        RootSource.DEFAULT: "по умолчанию (%APPDATA%\\Ayris)",
    }
)

_PORTABLE_SOURCES: Final = frozenset(
    {RootSource.PORTABLE_FLAG, RootSource.PORTABLE_ENV, RootSource.PORTABLE_MARKER}
)


@dataclass(frozen=True, slots=True)
class AppPaths:
    """Resolved absolute paths for one profile root."""

    root: Path
    source: RootSource = RootSource.DEFAULT

    @property
    def is_portable(self) -> bool:
        """Whether the profile sits next to the executable."""
        return self.source in _PORTABLE_SOURCES

    @property
    def source_label(self) -> str:
        """Russian explanation of where the root came from."""
        return ROOT_SOURCE_LABELS[self.source]

    @property
    def config_file(self) -> Path:
        return self.root / "config.toml"

    @property
    def broken_config_file(self) -> Path:
        return self.root / "config.toml.broken"

    @property
    def database_file(self) -> Path:
        return self.root / "ayris.db"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def stt_models_dir(self) -> Path:
        return self.models_dir / "stt"

    @property
    def tts_models_dir(self) -> Path:
        return self.models_dir / "tts"

    @property
    def wake_models_dir(self) -> Path:
        return self.models_dir / "wake"

    @property
    def llm_models_dir(self) -> Path:
        return self.models_dir / "llm"

    @property
    def plugins_dir(self) -> Path:
        return self.root / "plugins"

    @property
    def sounds_dir(self) -> Path:
        return self.root / "sounds"

    @property
    def screenshots_dir(self) -> Path:
        return self.root / "screenshots"

    @property
    def themes_dir(self) -> Path:
        return self.root / "themes"

    def model_dir(self, kind: ModelKind) -> Path:
        """Directory holding models of one kind. Used by the model manager."""
        return self.models_dir / kind

    def log_file(self, stamp: str) -> Path:
        """Path of the log file for a ``YYYYMMDD`` stamp."""
        return self.logs_dir / f"ayris_{stamp}.log"

    def all_directories(self) -> tuple[Path, ...]:
        """Directories that must exist before the application starts."""
        return (
            self.root,
            self.cache_dir,
            self.logs_dir,
            self.models_dir,
            self.stt_models_dir,
            self.tts_models_dir,
            self.wake_models_dir,
            self.llm_models_dir,
            self.plugins_dir,
            self.sounds_dir,
            self.screenshots_dir,
            self.themes_dir,
        )

    def ensure_directories(self) -> None:
        """Create missing directories.

        Raises:
            ConfigError: The profile root is not writable.
        """
        for directory in self.all_directories():
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(
                    f"cannot create profile directory {directory}: {exc}",
                    user_message=(
                        f"Не удалось создать папку профиля:\n{directory}\n"
                        "Проверьте права доступа или укажите другой путь через --profile."
                    ),
                    recoverable=False,
                ) from exc


def _truthy(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


def _safe_resolve(value: str | Path) -> Path | None:
    """``Path.resolve`` that returns ``None`` instead of raising on garbage input."""
    try:
        return Path(value).expanduser().resolve()
    except (OSError, ValueError):
        return None


def executable_dir() -> Path:
    """Directory of the running binary, or the repository root in development."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    # src/ayris/core/paths.py -> repository root
    return Path(__file__).resolve().parents[3]


def default_root() -> Path:
    """Per-user profile root: ``%APPDATA%\\Ayris`` on Windows."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / APP_DIR_NAME
    if sys.platform != "win32":
        # Non-Windows is only used for running the test suite and linters. The
        # check is written this way round so the branch sits inside a
        # sys.platform guard: mypy runs with platform = "win32" and would
        # otherwise report the fallback as unreachable code.
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_DIR_NAME
    return Path.home() / "AppData" / "Roaming" / APP_DIR_NAME


def _portable_root(exe_dir: Path) -> Path:
    """Root for portable mode, honouring a redirect inside the marker file."""
    redirected = _portable_marker_root(exe_dir)
    if redirected is not None:
        return redirected
    return (exe_dir / PORTABLE_DIR_NAME).resolve()


def _portable_marker_root(exe_dir: Path) -> Path | None:
    """Read the portable marker next to the executable, if there is one.

    An empty marker means ``<exe_dir>/profile``. A marker holding a path
    redirects the root; relative paths are resolved against the executable, so a
    USB stick can keep the profile on a different drive letter every time it is
    plugged in.
    """
    for name in PORTABLE_MARKERS:
        marker = exe_dir / name
        if not marker.is_file():
            continue
        try:
            raw = marker.read_text(encoding="utf-8-sig").strip()
        except OSError:
            raw = ""
        if not raw:
            return (exe_dir / PORTABLE_DIR_NAME).resolve()
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = exe_dir / target
        return _safe_resolve(target) or (exe_dir / PORTABLE_DIR_NAME).resolve()
    return None


def read_configured_root() -> Path | None:
    """Profile root chosen by the user in the settings, if any."""
    pointer = default_root() / POINTER_FILE_NAME
    try:
        raw = pointer.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None
    if not raw:
        return None
    return _safe_resolve(raw)


def write_configured_root(root: Path | str) -> Path:
    """Point future launches at ``root``. Takes effect on the next start.

    Returns:
        The absolute root that was recorded.

    Raises:
        ConfigError: The pointer file could not be written.
    """
    resolved = _safe_resolve(root)
    if resolved is None:
        raise ConfigError(
            f"invalid profile root: {root!r}",
            user_message=f"Недопустимый путь к папке профиля:\n{root}",
        )
    pointer = default_root() / POINTER_FILE_NAME
    try:
        pointer.parent.mkdir(parents=True, exist_ok=True)
        pointer.write_text(f"{resolved}\n", encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"cannot write profile pointer {pointer}: {exc}",
            user_message=(
                "Не удалось запомнить новый путь к профилю.\n"
                f"Проверьте права доступа к папке:\n{pointer.parent}"
            ),
        ) from exc
    return resolved


def clear_configured_root() -> None:
    """Forget a previously configured root and fall back to ``%APPDATA%\\Ayris``."""
    pointer = default_root() / POINTER_FILE_NAME
    try:
        pointer.unlink(missing_ok=True)
    except OSError as exc:
        raise ConfigError(
            f"cannot remove profile pointer {pointer}: {exc}",
            user_message="Не удалось сбросить путь к профилю.",
        ) from exc


def resolve_root_with_source(
    *,
    portable: bool = False,
    profile: Path | str | None = None,
) -> tuple[Path, RootSource]:
    """Pick the profile root and report which rule decided it.

    See the module docstring for the full precedence list.
    """
    if profile is not None:
        return Path(profile).expanduser().resolve(), RootSource.EXPLICIT

    exe_dir = executable_dir()
    if portable:
        return _portable_root(exe_dir), RootSource.PORTABLE_FLAG

    env_profile = os.environ.get(PROFILE_ENV_VAR, "").strip()
    if env_profile:
        resolved = _safe_resolve(env_profile)
        if resolved is not None:
            return resolved, RootSource.ENVIRONMENT

    if _truthy(os.environ.get(PORTABLE_ENV_VAR, "")):
        return _portable_root(exe_dir), RootSource.PORTABLE_ENV

    marker_root = _portable_marker_root(exe_dir)
    if marker_root is not None:
        return marker_root, RootSource.PORTABLE_MARKER

    configured = read_configured_root()
    if configured is not None:
        return configured, RootSource.CONFIGURED

    return default_root(), RootSource.DEFAULT


def resolve_root(*, portable: bool = False, profile: Path | str | None = None) -> Path:
    """Pick the profile root. Thin wrapper over :func:`resolve_root_with_source`."""
    root, _source = resolve_root_with_source(portable=portable, profile=profile)
    return root


def init_paths(
    *,
    portable: bool = False,
    profile: Path | str | None = None,
    create: bool = True,
) -> AppPaths:
    """Resolve and install the process-wide :class:`AppPaths`.

    Called once from :mod:`ayris.__main__` before anything touches disk.
    """
    global _paths
    root, source = resolve_root_with_source(portable=portable, profile=profile)
    paths = AppPaths(root=root, source=source)
    if create:
        paths.ensure_directories()
    _paths = paths
    return paths


def get_paths() -> AppPaths:
    """Return the installed paths, resolving defaults on first use."""
    if _paths is None:
        return init_paths()
    return _paths


def reset_paths() -> None:
    """Drop the cached paths. Test helper."""
    global _paths
    _paths = None
