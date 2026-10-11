"""Nuitka compilation entry point (task 71, §16.1).

Nuitka names the executable after the file it compiles, so the entry lives here —
``build/ayris.py`` → ``ayris.exe`` — rather than pointing Nuitka at
``src/ayris/__main__.py`` (which would yield ``__main__.exe``) or at the package
(``-m`` style, unsupported for standalone). It does nothing but hand control to
the real entry point, so every startup decision stays in one place.

Keep this file import-light: it must not touch Qt or any engine at module scope,
so that ``ayris.exe --version`` stays as cheap as ``python -m ayris --version``.
"""

from __future__ import annotations

import sys

from ayris.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
