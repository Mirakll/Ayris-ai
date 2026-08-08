"""Ayris — voice assistant for Windows 11.

Public package metadata only. Importing this package must stay cheap: no Qt,
no audio devices, no filesystem work at import time, because ``pyproject.toml``
reads ``__version__`` from here during the build.
"""

from __future__ import annotations

__all__ = ["__app_name__", "__version__"]

__version__ = "0.1.0"
__app_name__ = "Ayris"
