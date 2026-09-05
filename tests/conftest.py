"""Shared pytest fixtures.

Every test that touches paths, settings or logging must be isolated from the
real ``%APPDATA%\\Ayris`` profile, otherwise running the suite would pollute the
developer's own installation — and, worse, a test that writes a credential would
reach the real Windows Credential Manager.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ayris.core import config as config_module
from ayris.core import paths as paths_module
from ayris.core import secrets as secrets_module
from ayris.utils import logger as logger_module

#: Variables that point the ``models`` tests at locally downloaded weights.
#: They share the settings prefix but are not settings, so the isolation fixture
#: below leaves them alone.
HARNESS_ENV_PREFIX = f"{config_module.ENV_PREFIX}TEST_"


@pytest.fixture(autouse=True)
def _isolated_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the profile root at ``tmp_path`` and reset module-level state."""
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData"))
    # A developer with a portable build or a --profile override exported in their
    # shell would otherwise resolve a different root than CI does.
    monkeypatch.delenv(paths_module.PROFILE_ENV_VAR, raising=False)
    monkeypatch.delenv(paths_module.PORTABLE_ENV_VAR, raising=False)
    # Settings read AYRIS_* variables; a stray one would silently override the
    # default a test is asserting on.  AYRIS_TEST_* is the exception: those name
    # model directories for the ``models`` tests, no settings field is called
    # ``test_*``, and wiping them made every one of those tests skip itself even
    # on a machine where the weights were sitting right there.
    for name in [key for key in os.environ if key.startswith(config_module.ENV_PREFIX)]:
        if name.startswith(HARNESS_ENV_PREFIX):
            continue
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


@pytest.fixture(scope="session")
def ascii_weights() -> Iterator[Callable[[Path], Path]]:
    """Give a native library an ASCII path to model weights it must open.

    Vosk and CTranslate2 hand the path to a C++ library as UTF-8 bytes, which
    Windows then reads back through the ANSI code page - so a checkout in a
    folder like ``E:\\мистер бит ест рис`` is unopenable, and the ``models``
    tests that use real weights could only ever be skipped here.  Production
    handles this by asking Windows for the 8.3 spelling (see
    :func:`ayris.core.paths.native_path`) and refusing with a clear message when
    there is none; a test harness cannot refuse, so it copies instead.

    The copy lands once per session under the temporary directory, which has a
    short name on the system disk, and is removed afterwards.  Weights whose own
    path is already safe are handed back untouched, so nothing is copied on a
    machine where the checkout has a Latin path - CI included.
    """
    root: Path | None = None

    def prepared(source: Path) -> Path:
        nonlocal root
        safe = paths_module.native_path(source)
        if safe is not None:
            return Path(safe)
        if root is None:
            base = paths_module.native_path(tempfile.gettempdir())
            if base is None:  # pragma: no cover - a temp dir without a short name
                pytest.skip(f"нет ASCII-пути ни для {source}, ни для временной папки")
            root = Path(tempfile.mkdtemp(prefix="ayris-weights-", dir=base))
        destination = root / source.name
        if not destination.exists():
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        return destination

    yield prepared
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def _clipboard_or_skip() -> Iterator[None]:
    """Skip, rather than fail, when another program is holding the clipboard.

    The clipboard is one lock for the whole desktop and nothing can reserve it:
    a clipboard manager, Windows' own history service or a program that has just
    copied something can hold it past every retry in ``winapi._clipboard_open``.
    A test that never gets the lock has not found a defect in Ayris, it has found
    a busy desktop — and a red run that means «кто-то что-то скопировал» teaches
    everyone to stop reading red runs.

    The reason carries the holder's file name, so a skip that says ``python.exe``
    is a signal that this one is ours after all and worth chasing.
    """
    from ayris.actions.system.clipboard import ClipboardBusy

    try:
        yield
    except ClipboardBusy as busy:  # pragma: no cover - зависит от чужой программы
        pytest.skip(f"буфер обмена занят другой программой: {busy}")


@pytest.fixture
def clipboard_or_skip() -> Callable[[], contextlib.AbstractContextManager[None]]:
    """The guard above, as a fixture: tests never import from ``conftest`` here."""
    return _clipboard_or_skip
