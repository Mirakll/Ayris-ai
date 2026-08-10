"""Model management: what is available, what is installed, and getting between.

Four modules, in the order a model passes through them:

* :mod:`~ayris.models.catalog` — what Ayris knows how to install. Four JSON
  files under ``resources/models``, validated by pydantic at load time.
* :mod:`~ayris.models.downloader` — bytes over HTTP, with resume, cancellation,
  progress and a SHA-256 computed while writing.
* :mod:`~ayris.models.installer` — verified downloads into ``models/<kind>``,
  unpacking archives member by member so nothing escapes the target directory.
* :mod:`~ayris.models.registry` — the database side: which model is active for
  each kind, how much disk everything takes, whether it is still intact.

Almost every caller wants :class:`ModelRegistry` alone; it owns the other three.
The rest is exported for tests and for the settings window, which shows catalog
entries that are not installed yet and therefore has no record to hand around.

There is no UI here — the model manager page is task 50.
"""

from __future__ import annotations

from ayris.models.catalog import (
    CATALOG_SCHEMA_VERSION,
    CatalogError,
    CatalogFile,
    ModelCatalog,
    ModelEntry,
    ModelFile,
    catalog_dir,
    load_catalog,
    load_catalog_file,
)
from ayris.models.downloader import (
    DownloadCancelled,
    Downloader,
    DownloadError,
    DownloadHandle,
    DownloadResult,
    IntegrityError,
    NotEnoughSpaceError,
    ProgressCallback,
    human_size,
    sha256_file,
)
from ayris.models.installer import (
    ArchiveError,
    InstallError,
    Installer,
    InstallResult,
    Manifest,
    ManifestEntry,
    read_manifest,
)
from ayris.models.registry import (
    DiskUsage,
    IntegrityReport,
    IntegrityStatus,
    ModelInUseError,
    ModelRegistry,
    NotInstalledError,
)

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "ArchiveError",
    "CatalogError",
    "CatalogFile",
    "DiskUsage",
    "DownloadCancelled",
    "DownloadError",
    "DownloadHandle",
    "DownloadResult",
    "Downloader",
    "InstallError",
    "InstallResult",
    "Installer",
    "IntegrityError",
    "IntegrityReport",
    "IntegrityStatus",
    "Manifest",
    "ManifestEntry",
    "ModelCatalog",
    "ModelEntry",
    "ModelFile",
    "ModelInUseError",
    "ModelRegistry",
    "NotEnoughSpaceError",
    "NotInstalledError",
    "ProgressCallback",
    "catalog_dir",
    "human_size",
    "load_catalog",
    "load_catalog_file",
    "read_manifest",
    "sha256_file",
]
