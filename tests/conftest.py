"""Shared pytest fixtures.

Every test that touches paths, settings or logging must be isolated from the
real ``%APPDATA%\\Ayris`` profile, otherwise running the suite would pollute the
developer's own installation — and, worse, a test that writes a credential would
reach the real Windows Credential Manager.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from ayris.core import config as config_module
from ayris.core import paths as paths_module
from ayris.core import secrets as secrets_module
from ayris.utils import logger as logger_module


@pytest.fixture(autouse=True)
def _isolated_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the profile root at ``tmp_path`` and reset module-level state."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    # A developer with a portable build or a --profile override exported in their
    # shell would otherwise resolve a different root than CI does.
    monkeypatch.delenv(paths_module.PROFILE_ENV_VAR, raising=False)
    monkeypatch.delenv(paths_module.PORTABLE_ENV_VAR, raising=False)
    # Settings read AYRIS_* variables; a stray one would silently override the
    # default a test is asserting on.
    for name in [key for key in os.environ if key.startswith(config_module.ENV_PREFIX)]:
        monkeypatch.delenv(name, raising=False)

    config_module.reset_config_manager()
    secrets_module.reset_secrets()
    paths_module.reset_paths()
    logger_module.shutdown_logging()
    yield
    logger_module.shutdown_logging()
    config_module.reset_config_manager()
    secrets_module.reset_secrets()
    paths_module.reset_paths()


@pytest.fixture
def profile_paths(tmp_path: Path) -> paths_module.AppPaths:
    """An initialised profile root inside ``tmp_path``."""
    return paths_module.init_paths(profile=tmp_path / "profile")
