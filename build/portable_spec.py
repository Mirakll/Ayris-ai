"""Portable-variant layout (task 71, §16.2).

A portable Ayris is the standalone build with one file added: the marker
:data:`ayris.core.paths.PORTABLE_MARKERS` looks for next to the executable. With
it present the whole profile — ``config.toml``, ``ayris.db``, ``logs/``,
``models/``, downloaded plugins — lives beside ``ayris.exe`` instead of in
``%APPDATA%\\Ayris``, so the folder can be copied to a USB stick and run from any
machine. All of that switching happens inside ``core/paths.py`` (the single place
that decides installed-vs-portable); this module only lays the marker down and
pre-creates the ``models/`` folder the model manager downloads into.

The marker content is ``"."``: an empty marker would put the profile in a
``profile/`` subdirectory, but §16.2 asks for the profile and ``models/`` to sit
*beside* the exe, which is what a ``"."`` redirect (resolved against the
executable's own directory) produces.

Marker names, and the fact that a ``"."`` redirect resolves to the executable's
directory, are imported from :mod:`ayris.core.paths` rather than re-typed, so the
build and the runtime can never disagree about what makes a build portable.
"""

from __future__ import annotations

from pathlib import Path

from ayris.core.paths import PORTABLE_MARKERS

__all__ = [
    "MODEL_SUBDIRS",
    "PORTABLE_MARKER_CONTENT",
    "PORTABLE_MARKER_NAME",
    "describe_layout",
    "make_portable",
    "portable_paths",
]

#: The marker Ayris writes. The runtime accepts any name in ``PORTABLE_MARKERS``;
#: the build writes the first (canonical) one.
PORTABLE_MARKER_NAME = PORTABLE_MARKERS[0]

#: ``"."`` redirects the profile root to the executable's own directory (see the
#: module docstring), which is how the profile and ``models/`` end up beside the exe.
PORTABLE_MARKER_CONTENT = "."

#: The model kinds the manager downloads into, mirroring
#: :meth:`ayris.core.paths.AppPaths.model_dir`. Pre-created so the folders are
#: visible on a fresh stick before anything is downloaded.
MODEL_SUBDIRS: tuple[str, ...] = ("stt", "tts", "wake", "llm")


def portable_paths(dist_dir: Path) -> list[Path]:
    """The paths :func:`make_portable` will create under *dist_dir*, in order.

    Exposed separately from the side-effecting :func:`make_portable` so a dry run
    and the unit tests can assert the layout without writing anything.
    """
    marker = dist_dir / PORTABLE_MARKER_NAME
    models = dist_dir / "models"
    return [marker, models, *(models / kind for kind in MODEL_SUBDIRS)]


def make_portable(dist_dir: Path) -> list[Path]:
    """Turn the standalone artifact at *dist_dir* into a portable one.

    Writes the portable marker and creates an empty ``models/`` tree. Idempotent:
    running it twice leaves the same result and does not clobber anything a user
    has already put in ``models/``.

    Returns the paths it created or ensured, so the caller can report them.
    """
    if not dist_dir.is_dir():
        raise FileNotFoundError(f"нет папки сборки для портабельного варианта: {dist_dir}")

    marker = dist_dir / PORTABLE_MARKER_NAME
    marker.write_text(f"{PORTABLE_MARKER_CONTENT}\n", encoding="utf-8")

    created = [marker]
    models = dist_dir / "models"
    for directory in (models, *(models / kind for kind in MODEL_SUBDIRS)):
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return created


def describe_layout() -> str:
    """Human description of the shipped portable folder, for docs and the log."""
    lines = [
        "Портабельная раскладка (всё рядом с ayris.exe):",
        f"  {PORTABLE_MARKER_NAME}   маркер портабельного режима (содержит «.»)",
        "  resources/         темы, звуки, промпты, манифесты моделей, сертификаты",
        "  models/            веса, которые докачивает менеджер моделей:",
    ]
    lines.extend(f"    {kind}/" for kind in MODEL_SUBDIRS)
    lines.extend(
        [
            "  config.toml        настройки (создаётся при первом запуске)",
            "  ayris.db           команды, переменные, история",
            "  logs/              журналы",
        ]
    )
    return "\n".join(lines)
