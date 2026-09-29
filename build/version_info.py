"""Single source of the distribution's version metadata (task 71, §16.1).

The application version and name are not re-typed here: they live in
``src/ayris/__init__.py`` (``__version__``/``__app_name__``), which is also where
``pyproject.toml`` reads the version from via ``setuptools.dynamic``. This module
parses that one file — importing the package would drag Qt-adjacent imports into a
build tool that must stay light — and adds the fields a Windows binary needs but a
Python package does not: company, copyright, file description.

Everything a compiled ``ayris.exe`` shows in its Properties → Details tab, and the
version the running application reports through ``--version`` and in crash reports,
therefore traces back to a single place: raise ``__version__`` once and the binary,
the installer and the UI all move together.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

__all__ = [
    "APP_NAME",
    "COMPANY",
    "COPYRIGHT",
    "COPYRIGHT_YEAR",
    "DESCRIPTION",
    "EXECUTABLE_NAME",
    "VERSION",
    "metadata",
    "nuitka_version_arguments",
    "numeric_version",
    "product_version",
    "version_tuple",
]

_REPO_ROOT: Final = Path(__file__).resolve().parents[1]
_INIT_FILE: Final = _REPO_ROOT / "src" / "ayris" / "__init__.py"


def _read_dunder(name: str) -> str:
    """Pull a ``__dunder__ = "value"`` string out of ``ayris/__init__.py``.

    Parsing rather than importing keeps this module free of the application's
    import graph, which matters when the build runs in an environment where Qt or
    an audio backend is not importable.
    """
    text = _INIT_FILE.read_text(encoding="utf-8")
    match = re.search(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:  # pragma: no cover - only if __init__.py loses the constant
        raise RuntimeError(f"{name} не найден в {_INIT_FILE}")
    return match.group(1)


#: Read once at import from the single source of truth.
APP_NAME: Final = _read_dunder("__app_name__")
VERSION: Final = _read_dunder("__version__")

#: Fields a Windows binary carries but a Python package has no place for. Kept
#: beside the parsed version so the whole version resource has one origin.
COMPANY: Final = "Ayris contributors"
#: Fixed rather than ``date.today().year``: a reproducible build must not change
#: its embedded metadata from one day to the next. Bump by hand at year's end.
COPYRIGHT_YEAR: Final = 2026
COPYRIGHT: Final = f"© {COPYRIGHT_YEAR} {COMPANY}"
DESCRIPTION: Final = "Ayris — голосовой помощник для Windows 11"
EXECUTABLE_NAME: Final = "ayris.exe"


def version_tuple() -> tuple[int, int, int, int]:
    """The version as four integers, padding or truncating to Windows' shape.

    A Windows ``FILEVERSION``/``PRODUCTVERSION`` resource is always four 16-bit
    numbers. ``"0.1.0"`` becomes ``(0, 1, 0, 0)``; a pre-release suffix such as
    ``"1.2.3rc1"`` keeps only the leading digits of each component.
    """
    parts: list[int] = []
    for chunk in VERSION.split("."):
        digits = re.match(r"\d+", chunk)
        parts.append(int(digits.group()) if digits else 0)
    parts = [*parts, 0, 0, 0, 0][:4]
    return parts[0], parts[1], parts[2], parts[3]


def numeric_version() -> str:
    """Dotted four-number form Nuitka wants for ``--file-version`` etc."""
    return ".".join(str(part) for part in version_tuple())


def product_version() -> str:
    """The human product version — the same string the application reports."""
    return VERSION


def metadata() -> dict[str, str]:
    """Every version-resource field, as one dictionary. Handy for tests and logs."""
    return {
        "app_name": APP_NAME,
        "company": COMPANY,
        "product_version": product_version(),
        "file_version": numeric_version(),
        "description": DESCRIPTION,
        "copyright": COPYRIGHT,
    }


def nuitka_version_arguments() -> list[str]:
    """The Nuitka flags that stamp the Windows version resource onto the binary.

    These are the generic (cross-platform) metadata options Nuitka 1.7+ exposes;
    on Windows they land in the ``VS_VERSION_INFO`` resource that Explorer shows.
    """
    return [
        f"--product-name={APP_NAME}",
        f"--company-name={COMPANY}",
        f"--file-version={numeric_version()}",
        f"--product-version={numeric_version()}",
        f"--file-description={DESCRIPTION}",
        f"--copyright={COPYRIGHT}",
    ]
