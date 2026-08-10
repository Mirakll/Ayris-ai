"""The model manager: install, select, verify, remove.

This is the piece the settings window and the workers talk to. The catalog says
what exists, the downloader fetches bytes, the installer puts them on disk — and
this module owns the consequences: which model is the active one for each kind,
what that costs in disk space, and whether what is on disk is still what was
downloaded.

**Selecting a model is an event, not a setting.** The STT, TTS and wake-word
workers already have a model loaded when the user picks a different one, and they
cannot poll the database. So :meth:`ModelRegistry.set_active` publishes
:class:`~ayris.core.events.ActiveModelChanged` carrying the name *and* the path,
which is everything a reload needs. The database keeps the invariant — a partial
unique index allows one active model per kind — and the event carries the news.

**Deleting a model is the dangerous operation.** Files an engine has open cannot
be deleted on Windows, and a model that vanishes while it is the active one takes
recognition down with it. So deletion clears the active flag first (publishing
the change, so whoever holds the model can let go), refuses by default to touch a
model that is in use, and only then removes files and the row. The order matters:
row last, because a file that fails to delete with the row already gone leaves an
orphan nothing knows about.

**Integrity checking has two speeds.** The quick pass compares sizes and is
instant; the full pass re-hashes and takes as long as reading the model. A
single-file model is checked against the ``sha256`` in its own database row. A
model that arrived as an archive cannot be — the archive's hash says nothing
about the extracted tree — so the installer leaves a manifest inside the
directory and the full check verifies against that. Either way a failure names
the catalog entry, so the UI can offer «скачать заново» rather than just
«повреждена».
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Final

from ayris.core.errors import ModelError
from ayris.core.events import (
    ActiveModelChanged,
    ModelDownloadFinished,
    ModelDownloadStarted,
    ModelRemoved,
)
from ayris.core.models import MODEL_KINDS, ModelRecord
from ayris.models.catalog import CatalogError, ModelCatalog, load_catalog
from ayris.models.downloader import (
    DownloadHandle,
    Downloader,
    human_size,
    sha256_file,
)
from ayris.models.installer import Installer, read_manifest
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from ayris.core.events import Event, EventBus
    from ayris.core.paths import AppPaths, ModelKind
    from ayris.core.repositories import ModelRepository
    from ayris.models.catalog import ModelEntry
    from ayris.models.downloader import ProgressCallback

__all__ = [
    "DiskUsage",
    "IntegrityReport",
    "IntegrityStatus",
    "ModelInUseError",
    "ModelRegistry",
    "NotInstalledError",
]

_log = get_logger(__name__)

#: Subdirectory of the cache where partial downloads live. Under ``cache`` and
#: not under ``models`` on purpose: everything here is disposable, which is what
#: makes "очистить кэш моделей" safe to offer as a single button.
DOWNLOAD_DIR_NAME: Final = "downloads"


class NotInstalledError(ModelError):
    """The model is not in the database, or its files are gone."""

    default_user_message = "Модель не установлена."


class ModelInUseError(ModelError):
    """Refused to delete a model something is still using."""

    default_user_message = "Модель используется прямо сейчас."


class IntegrityStatus(StrEnum):
    """Outcome of checking an installed model."""

    OK = "ok"
    #: The row exists, the files do not. The usual cause is a profile restored
    #: without its models directory, or a user cleaning up by hand.
    MISSING = "missing"
    #: Files are there and do not match what was installed.
    CORRUPTED = "corrupted"
    #: Nothing to compare against — installed before manifests existed, or copied
    #: in by hand. Not a failure, and must not be presented as one.
    UNVERIFIED = "unverified"


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """What a check found, and whether anything can be done about it."""

    record: ModelRecord
    status: IntegrityStatus
    detail: str = ""
    checked_files: int = 0
    full: bool = False

    @property
    def ok(self) -> bool:
        return self.status in {IntegrityStatus.OK, IntegrityStatus.UNVERIFIED}

    @property
    def can_redownload(self) -> bool:
        """Whether the catalog still knows where this model came from."""
        return bool(self.record.catalog_id)


@dataclass(frozen=True, slots=True)
class DiskUsage:
    """How much room models take, per kind and in total."""

    by_kind: Mapping[str, int] = field(default_factory=dict)
    total: int = 0
    #: Partial downloads and staged files. Reported separately because it is the
    #: part the privacy page can free without uninstalling anything.
    cache_bytes: int = 0

    @property
    def human_total(self) -> str:
        return human_size(self.total)


class ModelRegistry:
    """Installs, selects, checks and removes models.

    Args:
        repository: The ``models`` table.
        paths: Profile paths, used for ``models/<kind>`` and for the cache.
        catalog: Loaded catalog. Read from ``resources/models`` when omitted.
        bus: Event bus. Optional, but without it nothing reloads on a change.
        downloader: Injected by tests; built from ``paths`` otherwise.
        installer: Injected by tests; built from ``paths`` otherwise.
        is_busy: Asked before deleting a model whether something still has it
            open. The composition root wires this to the worker supervisor; on
            its own the registry can only see that a model is *active*, which is
            not the same thing as loaded.
    """

    __slots__ = (
        "_bus",
        "_catalog",
        "_downloader",
        "_installer",
        "_is_busy",
        "_paths",
        "_repository",
    )

    def __init__(
        self,
        repository: ModelRepository,
        paths: AppPaths,
        *,
        catalog: ModelCatalog | None = None,
        bus: EventBus | None = None,
        downloader: Downloader | None = None,
        installer: Installer | None = None,
        is_busy: Callable[[ModelRecord], bool] | None = None,
    ) -> None:
        self._repository = repository
        self._paths = paths
        self._bus = bus
        self._is_busy = is_busy
        self._catalog = catalog if catalog is not None else _load_or_empty()
        self._downloader = (
            downloader if downloader is not None else Downloader(self.download_dir, bus=bus)
        )
        self._installer = installer if installer is not None else Installer(paths.models_dir)

    # ------------------------------------------------------------------
    # wiring
    # ------------------------------------------------------------------

    @property
    def catalog(self) -> ModelCatalog:
        return self._catalog

    @property
    def download_dir(self) -> Path:
        """Where partial downloads live."""
        return self._paths.cache_dir / "models" / DOWNLOAD_DIR_NAME

    def close(self) -> None:
        """Release the HTTP client."""
        self._downloader.close()

    # ------------------------------------------------------------------
    # installing
    # ------------------------------------------------------------------

    def install(
        self,
        model: str | ModelEntry,
        *,
        activate: bool | None = None,
        handle: DownloadHandle | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> ModelRecord:
        """Download, verify, unpack and register a catalog model.

        Args:
            model: Catalog id, or the entry itself.
            activate: Make it the active model for its kind. ``None`` means
                "only if the kind has none", which is what a first install
                wants: downloading a recogniser and then having to select it is
                a step nobody wants to take.
            handle: Cancellation token, so the UI can offer «отмена».
            on_progress: Called on this thread with the task's progress tuple.

        Returns:
            The registered row.

        Raises:
            NotEnoughSpaceError: Checked before the download starts.
            DownloadCancelled: The user cancelled; partial files are kept.
            IntegrityError: Checksum mismatch; the file is deleted.
            InstallError: The files could not be put in place.
        """
        entry = self._catalog.require(model) if isinstance(model, str) else model
        token = handle if handle is not None else DownloadHandle()
        self._downloader.check_space(entry, self._paths.model_dir(entry.kind))

        pending = sum(self._downloader.pending_bytes(entry.id, file) for file in entry.files)
        self._publish(
            ModelDownloadStarted(
                model_id=entry.id,
                kind=entry.kind,
                name=entry.name,
                total=entry.total_bytes,
                downloaded=pending,
                resumed=pending > 0,
            )
        )
        downloads = self._downloader.fetch_all(entry, handle=token, on_progress=on_progress)
        result = self._installer.install(entry, downloads)

        record = self._register(entry, result.path, result.size_bytes, downloads[0].sha256)
        wanted = activate if activate is not None else self.active(entry.kind) is None
        if wanted:
            record = self.set_active(record)

        self._publish(
            ModelDownloadFinished(
                model_id=entry.id,
                kind=entry.kind,
                path=str(result.path),
                size_bytes=result.size_bytes,
                duration_ms=sum(item.duration_ms for item in downloads),
            )
        )
        return record

    def _register(
        self,
        entry: ModelEntry,
        path: Path,
        size_bytes: int,
        sha256: str,
    ) -> ModelRecord:
        """Insert or refresh the row describing an installed model.

        Reinstalling has to update rather than insert: ``(kind, name, version)``
        is unique, and so is a non-empty ``catalog_id``.
        """
        existing = self._repository.by_catalog_id(entry.id) or self._repository.find(
            entry.kind, entry.target, entry.version
        )
        record = ModelRecord(
            kind=entry.kind,
            name=entry.target,
            id=existing.id if existing is not None else None,
            engine=entry.engine,
            version=entry.version,
            path=str(path),
            sha256=sha256,
            size_bytes=size_bytes,
            catalog_id=entry.id,
            is_active=existing.is_active if existing is not None else False,
        )
        if existing is None:
            return self._repository.add(record)
        return self._repository.update(record)

    def register_local(
        self,
        kind: ModelKind,
        name: str,
        *,
        engine: str = "",
        version: str = "",
        activate: bool = False,
    ) -> ModelRecord:
        """Register a model the user put into ``models/<kind>`` by hand.

        There is no requirement to use the catalog: someone with a Vosk model
        already on disk should be able to point Ayris at it. Such a row has no
        ``catalog_id``, which is what makes its integrity check fall back to
        "unverified" instead of claiming damage.

        Raises:
            NotInstalledError: There is nothing by that name on disk.
        """
        path = self._paths.model_dir(kind) / name
        if not path.exists():
            raise NotInstalledError(
                f"{path} does not exist",
                user_message=f"Файл модели не найден:\n{path}",
            )
        existing = self._repository.find(kind, name, version)
        record = ModelRecord(
            kind=kind,
            name=name,
            id=existing.id if existing is not None else None,
            engine=engine,
            version=version,
            path=str(path),
            size_bytes=_size_on_disk(path),
            is_active=existing.is_active if existing is not None else False,
        )
        stored = self._repository.update(record) if existing else self._repository.add(record)
        if activate:
            stored = self.set_active(stored)
        return stored

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def installed(self, kind: ModelKind | None = None) -> list[ModelRecord]:
        """Rows for installed models, for one kind or all of them."""
        if kind is None:
            return self._repository.list_all()
        return self._repository.list_by_kind(kind)

    def active(self, kind: ModelKind) -> ModelRecord | None:
        """The model a kind is currently using."""
        return self._repository.active(kind)

    def set_active(self, model: int | ModelRecord) -> ModelRecord:
        """Select a model for its kind and tell everyone.

        Idempotent: selecting the already-active model still publishes, because
        a worker that failed to load it may be waiting for exactly that nudge.

        Raises:
            NotInstalledError: No such row.
        """
        record = self._resolve(model)
        if record.id is None:  # pragma: no cover - _resolve guarantees an id
            raise NotInstalledError(
                "cannot activate a model without an id",
                user_message="Модель не установлена.",
            )
        self._repository.set_active(record.id)
        active = replace(record, is_active=True)
        _log.info("активная модель %s: %s", active.kind, active.name)
        self._publish(
            ActiveModelChanged(
                kind=active.kind,
                name=active.name,
                engine=active.engine,
                path=active.path,
                model_id=active.id,
                catalog_id=active.catalog_id,
            )
        )
        return active

    def clear_active(self, kind: ModelKind) -> bool:
        """Leave a kind with no model, telling its subsystem to unload."""
        if not self._repository.clear_active(kind):
            return False
        _log.info("активная модель %s снята", kind)
        self._publish(ActiveModelChanged(kind=kind))
        return True

    # ------------------------------------------------------------------
    # removal
    # ------------------------------------------------------------------

    def remove(self, model: int | ModelRecord, *, force: bool = False) -> int:
        """Uninstall a model: deselect it, delete its files, drop its row.

        Args:
            model: Row id or record.
            force: Delete even if the model is the active one. The settings
                window asks first; a scripted cleanup may not want to.

        Returns:
            Bytes freed.

        Raises:
            ModelInUseError: The model is active and ``force`` is not set, or the
                supervisor says a worker still has it open.
            InstallError: The files could not be deleted — on Windows, usually
                because an engine still holds them.
        """
        record = self._resolve(model)
        if self._is_busy is not None and self._is_busy(record):
            raise ModelInUseError(
                f"model {record.name} is loaded by a worker",
                user_message=(
                    f"Модель «{record.name}» сейчас используется.\n"
                    "Выберите другую модель и попробуйте снова."
                ),
            )
        if record.is_active and not force:
            raise ModelInUseError(
                f"model {record.name} is the active {record.kind} model",
                user_message=(
                    f"Модель «{record.name}» выбрана как активная для «{record.kind}».\n"
                    "Сначала выберите другую или подтвердите удаление."
                ),
            )
        if record.is_active:
            # Deselect before the files go, so whoever holds the model hears
            # about it while it is still readable.
            self.clear_active(record.kind)

        path = self._path_of(record)
        freed = self._installer.uninstall(path, extra=self._companions(record, path))
        if record.id is not None:
            self._repository.delete(record.id)
        _log.info("модель %s удалена, освобождено %s", record.name, human_size(freed))
        self._publish(
            ModelRemoved(
                model_id=record.catalog_id or record.name,
                kind=record.kind,
                name=record.name,
                freed_bytes=freed,
            )
        )
        return freed

    def _companions(self, record: ModelRecord, path: Path) -> tuple[Path, ...]:
        """Extra files installed alongside ``path``, per the catalog.

        A Piper voice is an ``.onnx`` and an ``.onnx.json``; deleting only the
        first leaves an orphan the settings list would keep showing.
        """
        entry = self._catalog.get(record.catalog_id) if record.catalog_id else None
        if entry is None or entry.is_archive:
            return ()
        return tuple(
            path.parent / extra.target for extra in entry.extra_files if extra.target != path.name
        )

    def purge_downloads(self) -> int:
        """Delete every partial and staged download.

        The «очистить кэш моделей» button. Installed models are untouched; what
        goes is only work in progress, which is why this needs no confirmation.

        Returns:
            Bytes freed.
        """
        root = self.download_dir
        if not root.is_dir():
            return 0
        freed = 0
        for item in sorted(root.rglob("*"), reverse=True):
            if item.is_file():
                freed += item.stat().st_size
                item.unlink(missing_ok=True)
        _log.info("кэш загрузок очищен, освобождено %s", human_size(freed))
        return freed

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------

    def disk_usage(self, *, measure: bool = False) -> DiskUsage:
        """Space taken by models, per kind and in total.

        Args:
            measure: Walk the directories instead of trusting the recorded
                sizes. Accurate and slow; the settings window uses the recorded
                numbers and offers this behind a refresh.
        """
        by_kind: dict[str, int] = {}
        for kind in MODEL_KINDS:
            if measure:
                by_kind[kind] = _size_on_disk(self._paths.model_dir(kind))
            else:
                by_kind[kind] = self._repository.total_size(kind)
        return DiskUsage(
            by_kind=by_kind,
            total=sum(by_kind.values()),
            cache_bytes=_size_on_disk(self.download_dir),
        )

    # ------------------------------------------------------------------
    # integrity
    # ------------------------------------------------------------------

    def verify(self, model: int | ModelRecord, *, full: bool = False) -> IntegrityReport:
        """Check that an installed model is still what was installed.

        Args:
            model: Row id or record.
            full: Re-hash the contents. Off by default: a quick size comparison
                catches a truncated or half-deleted model, which is what actually
                happens, and costs nothing.
        """
        record = self._resolve(model)
        path = self._path_of(record)
        if not path.exists():
            return IntegrityReport(
                record=record,
                status=IntegrityStatus.MISSING,
                detail=f"файл не найден: {path}",
                full=full,
            )
        if path.is_dir():
            return self._verify_directory(record, path, full=full)
        return self._verify_file(record, path, full=full)

    def verify_all(self, *, full: bool = False) -> list[IntegrityReport]:
        """Check every installed model. Used by the settings window's button."""
        return [self.verify(record, full=full) for record in self._repository.list_all()]

    def missing(self) -> list[ModelRecord]:
        """Rows whose files are gone — the database and the disk disagreeing."""
        return [
            record for record in self._repository.list_all() if not self._path_of(record).exists()
        ]

    def _verify_file(self, record: ModelRecord, path: Path, *, full: bool) -> IntegrityReport:
        size = path.stat().st_size
        if record.size_bytes and size != record.size_bytes:
            return IntegrityReport(
                record=record,
                status=IntegrityStatus.CORRUPTED,
                detail=f"размер {size} вместо {record.size_bytes}",
                checked_files=1,
                full=full,
            )
        if not full:
            status = IntegrityStatus.OK if record.size_bytes else IntegrityStatus.UNVERIFIED
            return IntegrityReport(record=record, status=status, checked_files=1)
        if not record.sha256:
            return IntegrityReport(
                record=record,
                status=IntegrityStatus.UNVERIFIED,
                detail="контрольная сумма не записана",
                checked_files=1,
                full=True,
            )
        actual = sha256_file(path)
        if actual != record.sha256:
            return IntegrityReport(
                record=record,
                status=IntegrityStatus.CORRUPTED,
                detail=f"sha256 {actual[:12]}… вместо {record.sha256[:12]}…",
                checked_files=1,
                full=True,
            )
        return IntegrityReport(record=record, status=IntegrityStatus.OK, checked_files=1, full=True)

    def _verify_directory(self, record: ModelRecord, path: Path, *, full: bool) -> IntegrityReport:
        manifest = read_manifest(path)
        if manifest is None:
            size = _size_on_disk(path)
            if record.size_bytes and size != record.size_bytes:
                return IntegrityReport(
                    record=record,
                    status=IntegrityStatus.CORRUPTED,
                    detail=f"размер {human_size(size)} вместо {human_size(record.size_bytes)}",
                    full=full,
                )
            return IntegrityReport(
                record=record,
                status=IntegrityStatus.UNVERIFIED,
                detail="нет описи файлов",
                full=full,
            )

        checked = 0
        for item in manifest.files:
            member = path / item.name
            if not member.is_file():
                return IntegrityReport(
                    record=record,
                    status=IntegrityStatus.MISSING,
                    detail=f"не хватает файла {item.name}",
                    checked_files=checked,
                    full=full,
                )
            if member.stat().st_size != item.size_bytes:
                return IntegrityReport(
                    record=record,
                    status=IntegrityStatus.CORRUPTED,
                    detail=f"размер файла {item.name} изменился",
                    checked_files=checked,
                    full=full,
                )
            if full and sha256_file(member) != item.sha256:
                return IntegrityReport(
                    record=record,
                    status=IntegrityStatus.CORRUPTED,
                    detail=f"контрольная сумма файла {item.name} не совпала",
                    checked_files=checked,
                    full=True,
                )
            checked += 1
        return IntegrityReport(
            record=record,
            status=IntegrityStatus.OK,
            checked_files=checked,
            full=full,
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _path_of(self, record: ModelRecord) -> Path:
        """Where a row's files live.

        ``path`` is absolute and authoritative, but a row written before the
        profile directory moved — or by an older build — may have none, so the
        name inside ``models/<kind>`` is the fallback.
        """
        if record.path:
            return Path(record.path)
        return self._paths.model_dir(record.kind) / record.name

    def _resolve(self, model: int | ModelRecord) -> ModelRecord:
        """Re-read a row from the database, by id or by record.

        Always a fresh read: a caller holding a ``ModelRecord`` from a listing
        may have missed an activation, and acting on a stale ``is_active`` is
        how a model gets deleted out from under a running worker.
        """
        if isinstance(model, ModelRecord):
            if model.id is None:
                raise NotInstalledError(
                    "model record has no id",
                    user_message="Модель не установлена.",
                )
            record = self._repository.get(model.id)
        else:
            record = self._repository.get(model)
        if record is None:
            raise NotInstalledError(
                f"model {model if isinstance(model, int) else model.name} is not registered",
                user_message="Модель не найдена в списке установленных.",
            )
        return record

    def _publish(self, event: Event) -> None:
        if self._bus is not None:
            self._bus.publish(event)


def _load_or_empty() -> ModelCatalog:
    """Load the shipped catalog, degrading to an empty one if it is broken.

    A malformed catalog must not stop Ayris from starting: the user still has
    their installed models, and an empty "доступно для загрузки" list plus a log
    line is a far better failure than a window that will not open.
    """
    try:
        return load_catalog()
    except CatalogError as exc:
        _log.error("каталог моделей не загружен: %s", exc.technical)
        return ModelCatalog()


def _size_on_disk(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
