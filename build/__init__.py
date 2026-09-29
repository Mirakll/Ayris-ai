"""Build configuration for Ayris (task 71).

This package holds the *configuration* of the distribution build — the Nuitka
argument composition, the Windows version resource, the portable-mode layout and
the artifact report. It is deliberately plain, importable Python so the sandbox
can exercise every decision without ever invoking a half-hour Nuitka compile: the
tests in ``tests/unit/test_build.py`` import these modules and mock the compiler.

Nothing here is imported by the shipping application; ``ayris`` never depends on
``build``. The only cross-import is the other way round — :mod:`build.version_info`
reads the single source of the application version from ``src/ayris/__init__.py``.
"""

from __future__ import annotations

__all__: list[str] = []
