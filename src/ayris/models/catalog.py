"""The catalog of downloadable models.

``resources/models/{stt,tts,wake,llm}.json`` lists what Ayris knows how to fetch:
one file per model kind, so a new voice can be added without touching the file
that describes recognisers, and a broken edit takes down one kind instead of the
whole catalog.

Every record is validated by pydantic before anything is downloaded. That is the
point of doing it here rather than at the download site: a typo in ``sha256`` or a
``size_bytes`` of ``-1`` is a packaging mistake, and it should surface as one
readable Russian message at startup instead of as a checksum failure after the
user has waited out a gigabyte.

**A model is a set of files, not always one.** Vosk ships a zip that unpacks to a
directory; a whisper.cpp model is a single ``.bin``; a Piper voice is an ``.onnx``
*and* the ``.onnx.json`` beside it that carries the sample rate and the phoneme
map, and either one alone is useless. So the record keeps the task's shape - one
``url``, one ``sha256``, one ``size_bytes``, one ``archive`` - and adds
:attr:`ModelEntry.extra_files` for the companions. :attr:`ModelEntry.files` hands
the installer the whole set as one uniform sequence, primary first.

**``target`` decides the name on disk, and it matters.** Settings name models the
way the engines find them: ``voice.stt.offline_model`` is a folder name inside
``models/stt``, ``voice.tts.voice`` is a file name inside ``models/tts``. So the
catalog spells the resulting name out instead of deriving it from ``id``, which
lets ``vosk-model-small-ru-0.22.zip`` land as the folder the default settings
already point at.

The file-level ``schema_version`` is the format version, checked against
:data:`CATALOG_SCHEMA_VERSION`. It is stamped onto every entry that does not
carry its own, so a record always knows which format it was written in - a newer
build reading an older catalog can branch on it without re-reading the file.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from ayris.core.errors import ModelError
from ayris.core.models import MODEL_KINDS
from ayris.core.paths import ModelKind, executable_dir
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "RESOURCE_DIR_NAME",
    "ArchiveKind",
    "CatalogError",
    "CatalogFile",
    "ModelCatalog",
    "ModelEntry",
    "ModelFile",
    "catalog_dir",
    "catalog_file_name",
    "load_catalog",
    "load_catalog_file",
]

_log = get_logger(__name__)

#: Format version of the JSON files in ``resources/models``. Bumped only when a
#: field changes meaning; adding an optional field does not need it, because an
#: older catalog still validates against the newer model.
CATALOG_SCHEMA_VERSION: Final = 1

#: Where the catalog sits relative to the installation root.
RESOURCE_DIR_NAME: Final = "resources"

#: How a model arrives. ``none`` is a bare file — a ``.bin``, an ``.onnx``, a
#: ``.gguf`` — which is the common case for everything except Vosk.
ArchiveKind = Literal["zip", "tar", "none"]

#: 64 lowercase hex digits. Written out rather than assembled, so the error
#: pydantic produces names the pattern the author has to match.
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"

#: Slug rules for ``id``: it becomes a lock key, a ``.part`` file name and a row
#: in the database, so no spaces, no slashes, no case games.
_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]*$"

#: Path separators rejected in ``target``. A catalog is data that ships with the
#: application, but it is still data: a ``target`` of ``../../evil`` would write
#: outside the profile, and the check costs one line.
_TARGET_FORBIDDEN: Final = ("..", "/", "\\", ":")


class CatalogError(ModelError):
    """A catalog file is missing, malformed or fails validation."""

    default_user_message = "Каталог моделей повреждён."


def catalog_file_name(kind: str) -> str:
    """Name of the catalog file for one kind, e.g. ``stt.json``."""
    return f"{kind}.json"


def catalog_dir() -> Path:
    """Directory holding the catalog files.

    ``resources`` ships next to the executable in a built distribution and sits
    at the repository root in development; :func:`~ayris.core.paths.executable_dir`
    already resolves that difference. Deliberately not part of
    :class:`~ayris.core.paths.AppPaths`: the catalog belongs to the installation,
    not to a profile, and two profiles must not see two different catalogs.
    """
    return executable_dir() / RESOURCE_DIR_NAME / "models"


class ModelFile(BaseModel):
    """One downloadable artifact: where it comes from and what it must hash to.

    Args:
        url: HTTP(S) source. Redirects are followed by the downloader.
        sha256: Expected digest of the downloaded bytes, 64 lowercase hex digits.
        size_bytes: Expected length. Used for the free-space check before the
            first byte is fetched, and as the quick integrity check afterwards.
        target: Name the file or directory gets inside ``models/<kind>``. Empty
            means "derive it": the last path segment of the URL for a bare file,
            the entry's ``id`` for an archive.
        archive: Whether the artifact needs unpacking.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size_bytes: int = Field(ge=0)
    target: str = ""
    archive: ArchiveKind = "none"

    @field_validator("sha256", mode="before")
    @classmethod
    def _fold_digest(cls, value: Any) -> Any:
        """Accept the uppercase form ``sha256sum`` prints on some platforms."""
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("нужен абсолютный http(s)-адрес")
        return value

    @field_validator("target")
    @classmethod
    def _check_target(cls, value: str) -> str:
        if any(token in value for token in _TARGET_FORBIDDEN):
            raise ValueError("имя на диске не может содержать путь")
        return value.strip()

    @property
    def url_name(self) -> str:
        """Last segment of the URL, percent-decoded and stripped of a query."""
        return Path(unquote(urlsplit(self.url).path)).name

    @property
    def is_archive(self) -> bool:
        return self.archive != "none"


class ModelEntry(ModelFile):
    """One model in the catalog.

    Inherits the artifact fields, because the primary file *is* the model in
    every case except a multi-file voice, and adds the description the settings
    window shows.
    """

    id: str = Field(pattern=_ID_PATTERN)
    name: str = Field(min_length=1)
    kind: ModelKind
    engine: str = Field(min_length=1)
    description: str = ""
    language: str = ""
    version: str = ""
    schema_version: int = Field(default=CATALOG_SCHEMA_VERSION, ge=1)
    extra_files: tuple[ModelFile, ...] = ()

    @field_validator("engine")
    @classmethod
    def _fold_engine(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _fill_targets(self) -> Self:
        """Derive missing ``target`` names and reject companions to an archive.

        An archive already decides its own layout, so a companion file next to it
        has nowhere sensible to go; that combination is a catalog mistake rather
        than something to guess at.
        """
        if self.extra_files and self.is_archive:
            raise ValueError("к архиву нельзя добавить extra_files")

        default = self.id if self.is_archive else self.url_name
        if not default:
            raise ValueError("не удалось определить имя файла: задайте target")
        updates: dict[str, Any] = {}
        if not self.target:
            updates["target"] = default
        extras = tuple(
            extra if extra.target else extra.model_copy(update={"target": extra.url_name})
            for extra in self.extra_files
        )
        if any(not extra.target for extra in extras):
            raise ValueError("не удалось определить имя дополнительного файла: задайте target")
        if extras != self.extra_files:
            updates["extra_files"] = extras
        if not updates:
            return self
        # Rebuilt through the validator, so a derived target is checked by the
        # same rules as one written out by hand.
        return type(self).model_validate({**self.model_dump(), **updates})

    @property
    def files(self) -> tuple[ModelFile, ...]:
        """Every artifact of this model, primary first."""
        primary = ModelFile(
            url=self.url,
            sha256=self.sha256,
            size_bytes=self.size_bytes,
            target=self.target,
            archive=self.archive,
        )
        return (primary, *self.extra_files)

    @property
    def total_bytes(self) -> int:
        """Bytes to download for the whole model, companions included."""
        return self.size_bytes + sum(extra.size_bytes for extra in self.extra_files)

    @property
    def label(self) -> str:
        """``name`` with the language in brackets, for a log line."""
        return f"{self.name} [{self.language}]" if self.language else self.name


class CatalogFile(BaseModel):
    """One ``resources/models/<kind>.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(ge=1)
    kind: ModelKind
    models: tuple[ModelEntry, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _stamp_version(cls, data: Any) -> Any:
        """Fill each entry's ``schema_version`` and ``kind`` from the file.

        The shipped files do not repeat ``"kind": "stt"`` on all eight entries —
        the file is already named ``stt.json``. An entry that states a kind keeps
        it, so ``_check_consistency`` can still catch one filed in the wrong file.
        """
        if not isinstance(data, dict):
            return data
        version = data.get("schema_version")
        kind = data.get("kind")
        entries = data.get("models")
        if version is None or not isinstance(entries, list):
            return data
        defaults: dict[str, Any] = {"schema_version": version}
        if kind is not None:
            defaults["kind"] = kind
        stamped = [{**defaults, **entry} if isinstance(entry, dict) else entry for entry in entries]
        return {**data, "models": stamped}

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.schema_version > CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"формат каталога v{self.schema_version} новее поддерживаемого "
                f"v{CATALOG_SCHEMA_VERSION}"
            )
        wrong = [entry.id for entry in self.models if entry.kind != self.kind]
        if wrong:
            raise ValueError(f"записи не того вида ({self.kind}): {', '.join(wrong)}")
        counts = Counter(entry.id for entry in self.models)
        duplicates = sorted(model_id for model_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"повторяющиеся id: {', '.join(duplicates)}")
        return self


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Every catalog file, merged and indexed by ``id``.

    Immutable on purpose: the settings window, the downloader and the registry
    all read it, and a catalog that could be edited underneath them would make
    "which model is this" depend on timing.
    """

    entries: tuple[ModelEntry, ...] = ()

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[ModelEntry]:
        return iter(self.entries)

    def __contains__(self, model_id: object) -> bool:
        return any(entry.id == model_id for entry in self.entries)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(entry.id for entry in self.entries)

    def get(self, model_id: str) -> ModelEntry | None:
        """The entry with this ``id``, or ``None``."""
        for entry in self.entries:
            if entry.id == model_id:
                return entry
        return None

    def require(self, model_id: str) -> ModelEntry:
        """The entry with this ``id``.

        Raises:
            CatalogError: No such model. Happens when a database row survives a
                catalog entry being renamed, which is exactly when the caller
                needs to be told rather than handed ``None``.
        """
        entry = self.get(model_id)
        if entry is None:
            raise CatalogError(
                f"model {model_id!r} is not in the catalog",
                user_message=f"Модель «{model_id}» отсутствует в каталоге.",
            )
        return entry

    def for_kind(self, kind: ModelKind) -> tuple[ModelEntry, ...]:
        return tuple(entry for entry in self.entries if entry.kind == kind)

    def for_engine(self, kind: ModelKind, engine: str) -> tuple[ModelEntry, ...]:
        wanted = engine.strip().lower()
        return tuple(entry for entry in self.for_kind(kind) if entry.engine == wanted)

    def engines(self, kind: ModelKind) -> tuple[str, ...]:
        """Engine names present for a kind, in catalog order."""
        found: list[str] = []
        for entry in self.for_kind(kind):
            if entry.engine not in found:
                found.append(entry.engine)
        return tuple(found)


def load_catalog_file(path: Path) -> CatalogFile:
    """Read and validate one catalog file.

    Raises:
        CatalogError: The file is unreadable, is not JSON, or a record is
            invalid. The Russian message names the file and the first few
            problems, because a catalog is edited by hand and "invalid" alone
            would mean re-reading the whole thing.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise CatalogError(
            f"cannot read model catalog {path}: {exc}",
            user_message=f"Не удалось прочитать каталог моделей:\n{path}",
        ) from exc

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise CatalogError(
            f"model catalog {path} is not valid JSON: {exc}",
            user_message=(f"Каталог моделей повреждён:\n{path}\nОшибка разбора JSON: {exc}"),
        ) from exc

    try:
        return CatalogFile.model_validate(payload)
    except ValidationError as exc:
        raise CatalogError(
            f"model catalog {path} failed validation: {exc.error_count()} problem(s)",
            user_message=f"Каталог моделей {path.name} заполнен неверно:\n{_describe(exc)}",
        ) from exc


def load_catalog(
    directory: Path | None = None,
    *,
    kinds: Sequence[str] = MODEL_KINDS,
    strict: bool = False,
) -> ModelCatalog:
    """Read every catalog file in ``directory``.

    Args:
        directory: Where the files are. Defaults to :func:`catalog_dir`.
        kinds: Which files to read. Defaults to every model kind.
        strict: Treat a missing file as an error. Off by default so a build that
            ships without, say, ``llm.json`` still starts: the settings window
            shows an empty list, which is honest, instead of the application
            refusing to launch over a catalog it may never need.

    Raises:
        CatalogError: A file is malformed, or is missing while ``strict``.
    """
    root = directory if directory is not None else catalog_dir()
    entries: list[ModelEntry] = []
    seen: dict[str, str] = {}
    for kind in kinds:
        path = root / catalog_file_name(kind)
        if not path.is_file():
            if strict:
                raise CatalogError(
                    f"model catalog {path} is missing",
                    user_message=f"Файл каталога моделей не найден:\n{path}",
                )
            _log.warning("каталог моделей не найден: %s", path)
            continue
        for entry in load_catalog_file(path).models:
            previous = seen.get(entry.id)
            if previous is not None:
                raise CatalogError(
                    f"duplicate model id {entry.id!r} in {path.name} and {previous}",
                    user_message=(
                        f"Модель «{entry.id}» описана дважды: " f"в {previous} и в {path.name}."
                    ),
                )
            seen[entry.id] = path.name
            entries.append(entry)
    _log.debug("каталог моделей: %d записей из %s", len(entries), root)
    return ModelCatalog(entries=tuple(entries))


def _describe(exc: ValidationError) -> str:
    """Turn a pydantic failure into something a catalog author can act on."""
    lines: list[str] = []
    for error in exc.errors()[:5]:
        where = ".".join(str(part) for part in error["loc"]) or "файл"
        lines.append(f"• {where}: {error['msg']}")
    if exc.error_count() > 5:
        lines.append(f"…и ещё {exc.error_count() - 5}")
    return "\n".join(lines)
