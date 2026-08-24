"""Putting a downloaded model where the engines will find it.

The downloader hands over verified bytes in a work directory; this module turns
them into something ``models/<kind>/`` can be pointed at. That means unpacking
archives, naming the result what the engines expect, and doing both without
letting an archive decide where on the disk it lands.

**Archives are extracted member by member.** ``ZipFile.extractall`` and
``TarFile.extractall`` are one call each, and both were rejected here. A tar
member can name ``../../../etc/passwd``, a zip entry can carry a Windows
absolute path or a backslash separator that ``zipfile`` will not treat as a
separator at all, and either can be a symlink pointing anywhere. Extracting one
member at a time means every destination is resolved and checked against the
target directory before a byte is written. It also sidesteps a Python-version
problem: ``tarfile.extractall`` without ``filter=`` raises a ``DeprecationWarning``
on 3.12 — an error under this project's ``filterwarnings`` — while ``filter=``
does not exist before 3.11.4, so no single call works on both interpreters CI
runs.

**A Vosk archive contains one top-level directory and the model is that
directory.** Extracting it as-is under a target named after the model would give
``models/stt/vosk-model-small-ru-0.22/vosk-model-small-ru-0.22/``, and every
engine would fail to find anything. So a single common top-level directory is
stripped, and the catalog's ``target`` names the result — which is what lets the
default setting ``offline_model = "vosk-model-small-ru-0.22"`` resolve without
the user touching it.

**Installation is staged, then swapped.** Everything lands in a
``<target>.incomplete`` directory beside the destination, and only a complete,
verified tree is moved into place. A crash halfway through leaves the previous
model working and a junk directory to clean up, rather than half a model where a
whole one used to be.

**A directory install leaves a manifest.** ``.ayris-model.json`` inside the
extracted directory lists every file with its size and hash, because after
extraction the archive's own checksum can no longer be recomputed from what is on
disk — and without it "проверить целостность" could only ever compare sizes. A
single-file install needs no manifest: the catalog's ``sha256`` is the hash of
the file itself.

**A record may ask for a subdirectory even without an archive.** Some models are
several loose files with names their loader globs for — a GigaAM export is
weights plus a vocabulary, found by ``v?_ctc*.onnx`` and ``v?_vocab.txt`` — so
flat in ``models/stt`` a second variant would collide with the first on the same
glob. :attr:`~ayris.models.catalog.ModelEntry.directory` names the folder, and
then those files are staged, described and swapped exactly like an archive's,
manifest included. Everything downstream sees a directory install and needs no
special case.
"""

from __future__ import annotations

import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, TYPE_CHECKING, Final

from ayris.core.errors import ModelError
from ayris.models.downloader import human_size, sha256_file
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ayris.models.catalog import ModelEntry
    from ayris.models.downloader import DownloadResult

__all__ = [
    "MANIFEST_NAME",
    "ArchiveError",
    "InstallError",
    "InstallResult",
    "Installer",
    "Manifest",
    "ManifestEntry",
    "read_manifest",
]

_log = get_logger(__name__)

#: Written inside a directory install, next to the model's own files. The name
#: starts with a dot so the engines' globs (``*.onnx``, ``model.bin``) ignore it,
#: and carries the project name so it is obviously ours if a user finds it.
MANIFEST_NAME: Final = ".ayris-model.json"

#: Suffix of the staging directory. Deliberately not ``.tmp``: cleanup tools
#: delete ``.tmp`` directories, and this one holds an in-progress install.
STAGING_SUFFIX: Final = ".incomplete"

#: Refused outright in an archive member's name.
_UNSAFE_PARTS: Final = ("..",)

#: Bytes read at a time while extracting, matching the downloader's chunk.
_COPY_CHUNK: Final = 64 * 1024


class InstallError(ModelError):
    """A model could not be installed."""

    default_user_message = "Не удалось установить модель."


class ArchiveError(InstallError):
    """The archive is malformed, or is trying to write outside the target."""

    default_user_message = "Архив модели повреждён или небезопасен."


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One file inside an installed directory."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Manifest:
    """What was written when a directory model was installed.

    Exists so :meth:`ayris.models.registry.ModelRegistry.verify` can answer "is
    this still intact" for an unpacked model. Without it the only honest answer
    would be "the directory is still there and roughly the right size".
    """

    catalog_id: str
    archive_sha256: str
    total_bytes: int
    files: tuple[ManifestEntry, ...] = ()

    def to_json(self) -> str:
        payload = {
            "catalog_id": self.catalog_id,
            "archive_sha256": self.archive_sha256,
            "total_bytes": self.total_bytes,
            "files": [
                {"name": item.name, "size_bytes": item.size_bytes, "sha256": item.sha256}
                for item in self.files
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=1)


@dataclass(frozen=True, slots=True)
class InstallResult:
    """Where a model ended up and how much room it takes."""

    catalog_id: str
    path: Path
    name: str
    size_bytes: int
    files: tuple[Path, ...] = ()

    @property
    def is_directory(self) -> bool:
        return self.path.is_dir()


class Installer:
    """Moves verified downloads into ``models/<kind>``.

    Args:
        models_root: Directory holding one subdirectory per kind. Taken as an
            argument rather than read from :func:`~ayris.core.paths.get_paths`
            so a test — and a future "install to another drive" — needs no global
            state.
    """

    __slots__ = ("_root",)

    def __init__(self, models_root: Path) -> None:
        self._root = models_root

    def destination(self, entry: ModelEntry) -> Path:
        """Where this model will live once installed."""
        return self._root / entry.kind / entry.install_name

    def is_installed(self, entry: ModelEntry) -> bool:
        return self.destination(entry).exists()

    # ------------------------------------------------------------------
    # installing
    # ------------------------------------------------------------------

    def install(self, entry: ModelEntry, downloads: Sequence[DownloadResult]) -> InstallResult:
        """Place a downloaded model into its kind directory.

        Args:
            entry: The catalog record the files came from.
            downloads: Verified files from :meth:`Downloader.fetch_all`, in the
                same order as :attr:`ModelEntry.files`.

        Returns:
            Where the model landed and what it occupies.

        Raises:
            ArchiveError: The archive is unreadable or tries to escape.
            InstallError: The files could not be moved into place.
        """
        if not downloads:
            raise InstallError(
                f"nothing downloaded for {entry.id}",
                user_message=f"Нечего устанавливать для «{entry.name}».",
            )
        kind_dir = self._root / entry.kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        if entry.is_archive:
            return self._install_archive(entry, downloads[0], kind_dir)
        return self._install_files(entry, downloads, kind_dir)

    def _install_archive(
        self,
        entry: ModelEntry,
        download: DownloadResult,
        kind_dir: Path,
    ) -> InstallResult:
        destination = kind_dir / entry.target
        staging = kind_dir / f"{entry.target}{STAGING_SUFFIX}"
        _remove(staging)
        staging.mkdir(parents=True)
        try:
            _extract(download.path, staging, entry.archive)
            _strip_single_root(staging)
            manifest = _describe(staging, catalog_id=entry.id, archive_sha256=download.sha256)
            (staging / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
            _swap(staging, destination)
        except OSError as exc:
            _remove(staging)
            raise InstallError(
                f"cannot install {entry.id} into {destination}: {exc}",
                user_message=(
                    f"Не удалось установить «{entry.name}».\n"
                    "Проверьте, что файлы модели не открыты другой программой."
                ),
            ) from exc
        except ModelError:
            _remove(staging)
            raise

        download.path.unlink(missing_ok=True)
        size = _directory_size(destination)
        _log.info("модель %s установлена в %s (%s)", entry.id, destination, human_size(size))
        return InstallResult(
            catalog_id=entry.id,
            path=destination,
            name=entry.target,
            size_bytes=size,
            files=tuple(sorted(path for path in destination.rglob("*") if path.is_file())),
        )

    def _install_files(
        self,
        entry: ModelEntry,
        downloads: Sequence[DownloadResult],
        kind_dir: Path,
    ) -> InstallResult:
        """Move bare files into the kind directory.

        All of them or none: a Piper voice whose ``.onnx`` arrived and whose
        ``.onnx.json`` did not is not a voice, it is a file that makes the engine
        raise on first use.
        """
        if entry.directory:
            return self._install_directory(entry, downloads, kind_dir)
        moved: list[Path] = []
        try:
            for download in downloads:
                target = kind_dir / download.target
                target.parent.mkdir(parents=True, exist_ok=True)
                _remove(target)
                download.path.replace(target)
                moved.append(target)
        except OSError as exc:
            for path in moved:
                _remove(path)
            raise InstallError(
                f"cannot install {entry.id} into {kind_dir}: {exc}",
                user_message=(
                    f"Не удалось установить «{entry.name}».\n"
                    "Проверьте, что файл модели не открыт другой программой."
                ),
            ) from exc

        primary = kind_dir / entry.target
        size = sum(path.stat().st_size for path in moved)
        _log.info("модель %s установлена в %s (%s)", entry.id, primary, human_size(size))
        return InstallResult(
            catalog_id=entry.id,
            path=primary,
            name=entry.target,
            size_bytes=size,
            files=tuple(moved),
        )

    def _install_directory(
        self,
        entry: ModelEntry,
        downloads: Sequence[DownloadResult],
        kind_dir: Path,
    ) -> InstallResult:
        """Move bare files into a subdirectory of their own, staged like an archive.

        A model that arrives as several loose files — a GigaAM export is weights
        plus a vocabulary — has to keep the names its loader globs for, so two
        variants of it cannot share ``models/<kind>``. Staged and swapped for the
        same reason an archive is: a half-moved set of files would look installed
        and then fail on load.
        """
        destination = kind_dir / entry.directory
        staging = kind_dir / f"{entry.directory}{STAGING_SUFFIX}"
        _remove(staging)
        try:
            staging.mkdir(parents=True)
            for download in downloads:
                download.path.replace(staging / download.target)
            manifest = _describe(staging, catalog_id=entry.id, archive_sha256=downloads[0].sha256)
            (staging / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
            _swap(staging, destination)
        except OSError as exc:
            _remove(staging)
            raise InstallError(
                f"cannot install {entry.id} into {destination}: {exc}",
                user_message=(
                    f"Не удалось установить «{entry.name}».\n"
                    "Проверьте, что файлы модели не открыты другой программой."
                ),
            ) from exc

        size = _directory_size(destination)
        _log.info("модель %s установлена в %s (%s)", entry.id, destination, human_size(size))
        return InstallResult(
            catalog_id=entry.id,
            path=destination,
            name=entry.directory,
            size_bytes=size,
            files=tuple(sorted(path for path in destination.rglob("*") if path.is_file())),
        )

    # ------------------------------------------------------------------
    # removing
    # ------------------------------------------------------------------

    def uninstall(self, path: Path, *, extra: Iterable[Path] = ()) -> int:
        """Delete an installed model and report how much it freed.

        Args:
            path: The file or directory named by the model record.
            extra: Companion files — a Piper ``.onnx.json``, say — that live
                beside ``path`` rather than inside it.

        Raises:
            InstallError: Something is holding the files open. On Windows that is
                the usual outcome of deleting a model an engine still has loaded,
                and the message says so.
        """
        freed = 0
        try:
            for item in (path, *extra):
                if not item.exists():
                    continue
                freed += _directory_size(item) if item.is_dir() else item.stat().st_size
                _remove(item)
        except OSError as exc:
            raise InstallError(
                f"cannot delete {path}: {exc}",
                user_message=(
                    "Не удалось удалить файлы модели: они заняты другой программой.\n"
                    "Закройте её или перезапустите Ayris и попробуйте снова."
                ),
            ) from exc
        return freed


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------


def _extract(archive: Path, target: Path, kind: str) -> None:
    """Unpack ``archive`` into ``target``, refusing anything that escapes it."""
    if kind == "zip":
        _extract_zip(archive, target)
        return
    if kind == "tar":
        _extract_tar(archive, target)
        return
    raise ArchiveError(  # pragma: no cover - the catalog validates ``archive``
        f"unsupported archive type {kind!r}",
        user_message=f"Неизвестный формат архива: {kind}.",
    )


def _extract_zip(archive: Path, target: Path) -> None:
    try:
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                destination = _safe_destination(info.filename, target)
                if destination is None:
                    continue
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, destination.open("wb") as sink:
                    _copy(source, sink)
    except (zipfile.BadZipFile, OSError) as exc:
        raise ArchiveError(
            f"cannot read zip {archive}: {exc}",
            user_message="Архив модели повреждён. Скачайте модель заново.",
        ) from exc


def _extract_tar(archive: Path, target: Path) -> None:
    try:
        with tarfile.open(archive) as bundle:
            for member in bundle:
                if member.issym() or member.islnk():
                    # A link's target is resolved by the filesystem, not by us,
                    # so a "safe" name can still point at /etc. Models do not
                    # need links; refusing them costs nothing.
                    _log.warning("пропущена ссылка в архиве: %s", member.name)
                    continue
                if not (member.isfile() or member.isdir()):
                    continue
                destination = _safe_destination(member.name, target)
                if destination is None:
                    continue
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                source = bundle.extractfile(member)
                if source is None:  # pragma: no cover - isfile() implies a stream
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as sink:
                    _copy(source, sink)
    except (tarfile.TarError, OSError) as exc:
        raise ArchiveError(
            f"cannot read tar {archive}: {exc}",
            user_message="Архив модели повреждён. Скачайте модель заново.",
        ) from exc


def _copy(source: IO[bytes], sink: IO[bytes]) -> int:
    """Stream one archive member out, without materialising it."""
    total = 0
    while True:
        block = source.read(_COPY_CHUNK)
        if not block:
            break
        sink.write(block)
        total += len(block)
    return total


def _safe_destination(name: str, target: Path) -> Path | None:
    """Resolve an archive member name inside ``target``, or ``None`` to skip it.

    Rejects, in order: Windows separators (``zipfile`` does not treat a backslash
    as one, so ``a\\..\\..\\b`` would reach the filesystem intact), absolute
    paths, and any ``..`` component. Drive letters and UNC prefixes survive
    :class:`PurePosixPath` as ordinary path components, so the resolved
    destination is compared against the resolved target as well — which also
    catches whatever this list did not think of.

    Raises:
        ArchiveError: The name is an escape attempt. Skipping it silently would
            leave the user with a model that is quietly missing files.
    """
    raw = name.replace("\\", "/")
    if not raw or raw in {".", "./"}:
        return None
    pure = PurePosixPath(raw)
    escapes = pure.is_absolute() or any(part in _UNSAFE_PARTS for part in pure.parts)
    destination = target / pure
    root = target.resolve()
    if not escapes:
        try:
            resolved = destination.resolve()
        except OSError as exc:  # pragma: no cover - only on a broken mount
            raise ArchiveError(
                f"cannot resolve archive member {name!r}: {exc}",
                user_message="Архив модели повреждён. Скачайте модель заново.",
            ) from exc
        escapes = resolved != root and root not in resolved.parents
    if escapes:
        raise ArchiveError(
            f"archive member escapes the target directory: {name!r}",
            user_message=(
                "Архив модели пытается записать файлы за пределы папки моделей.\n"
                "Установка отменена."
            ),
        )
    return destination


def _strip_single_root(staging: Path) -> None:
    """Hoist the contents of a lone top-level directory up one level.

    ``vosk-model-small-ru-0.22.zip`` contains exactly one directory of that name;
    without this the model would install to a path with the name in it twice.
    Only done when the archive has a single directory entry and nothing else, so
    an archive that legitimately puts files at the root is left alone.
    """
    children = list(staging.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        return
    inner = children[0]
    # Move aside first: renaming ``x/y`` to ``x`` directly is not something every
    # filesystem allows, and on Windows it is an error.
    holding = staging.parent / f"{staging.name}.root"
    _remove(holding)
    inner.replace(holding)
    _remove(staging)
    holding.replace(staging)


# ----------------------------------------------------------------------
# manifests and files
# ----------------------------------------------------------------------


def _describe(root: Path, *, catalog_id: str, archive_sha256: str) -> Manifest:
    """Hash everything under ``root`` into a manifest.

    Done once, at install time, while the files are still in the page cache from
    being written — which is the cheapest moment it will ever be.
    """
    files = sorted(path for path in root.rglob("*") if path.is_file())
    entries = tuple(
        ManifestEntry(
            name=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in files
    )
    return Manifest(
        catalog_id=catalog_id,
        archive_sha256=archive_sha256,
        total_bytes=sum(item.size_bytes for item in entries),
        files=entries,
    )


def read_manifest(path: Path) -> Manifest | None:
    """Read the manifest of an installed directory, or ``None`` if there is none.

    A model installed by an older build, or copied in by hand, has no manifest;
    that is not an error, it only means the integrity check falls back to sizes.
    """
    source = path / MANIFEST_NAME if path.is_dir() else path
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    raw_files = payload.get("files")
    files: list[ManifestEntry] = []
    if isinstance(raw_files, list):
        for item in raw_files:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            digest = item.get("sha256")
            size = item.get("size_bytes")
            if isinstance(name, str) and isinstance(digest, str) and isinstance(size, int):
                files.append(ManifestEntry(name=name, size_bytes=size, sha256=digest))
    total = payload.get("total_bytes")
    return Manifest(
        catalog_id=str(payload.get("catalog_id", "")),
        archive_sha256=str(payload.get("archive_sha256", "")),
        total_bytes=total if isinstance(total, int) else 0,
        files=tuple(files),
    )


def _swap(staging: Path, destination: Path) -> None:
    """Replace ``destination`` with ``staging``, keeping the old one until it works.

    ``Path.replace`` onto a non-empty directory fails on every platform, so the
    previous install is moved out of the way first and only deleted once the new
    one is in place. If the move fails, the old model is put back.
    """
    if not destination.exists():
        staging.replace(destination)
        return
    backup = destination.parent / f"{destination.name}.old"
    _remove(backup)
    destination.replace(backup)
    try:
        staging.replace(destination)
    except OSError:
        backup.replace(destination)
        raise
    _remove(backup)


def _remove(path: Path) -> None:
    """Delete a file or a whole directory. Missing is fine."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _directory_size(path: Path) -> int:
    """Bytes occupied by a file, or by every file under a directory."""
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
