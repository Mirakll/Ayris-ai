"""Portable profile bundles: export a profile to ``.zip`` and read one back.

Section 17 of the specification rules out cloud sync: what Ayris offers instead
is a single file the user can hand to another machine, drop into a synced
folder, or keep as a backup. This module owns that file format; the profile
lifecycle around it lives in :mod:`ayris.core.profile`.

Layout of a bundle::

    manifest.json     format, schema version, app version, date, counts
    config.toml       settings with every secret-bearing field removed
    commands.ayris    folders, commands and their triggers, as JSON
    variables.json    persistent profile- and global-scope variables
    models.json       which models the profile expects - manifests, not files
    sounds/...        user sound files copied verbatim

Three properties the format is built around:

*No secrets.* Settings only ever hold the *name* of a credential entry, but the
export strips those names too, together with anything that looks like a key or a
token. A bundle is safe to attach to a bug report.

*No absolute paths.* Folders are identified by their name path, models by kind
and name. Nothing in the archive points at the machine that produced it.

*Recoverable failure.* An import that dies half way puts the profile back the
way it was: file writes go through :class:`_FileGuard`, database writes through
one transaction, and neither is committed until both have succeeded.

Names inside the archive are written as UTF-8 with the language-encoding flag
set, because command and sound names are routinely Cyrillic and a bundle
produced here has to unpack correctly under any unzip tool.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Final, cast
from uuid import uuid4

from ayris import __app_name__, __version__
from ayris.core.config import SCHEMA_VERSION as CONFIG_SCHEMA_VERSION
from ayris.core.config import (
    Settings,
    dump_settings,
    load_settings,
    save_settings,
    settings_from_mapping,
)
from ayris.core.errors import ConfigError, ProfileError
from ayris.core.migrations import SCHEMA_VERSION as DB_SCHEMA_VERSION
from ayris.core.models import (
    Command,
    CommandFolder,
    JsonObject,
    Trigger,
    TriggerType,
    VariableScope,
    VariableType,
    utc_now,
)
from ayris.core.paths import get_paths
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from ayris.core.models import Profile
    from ayris.core.paths import AppPaths
    from ayris.core.repositories import Repositories

_log = get_logger(__name__)

#: Marker in ``manifest.json``. A zip without it is not ours, however it looks.
BUNDLE_FORMAT: Final = "ayris-profile"

#: Version of *this* layout. Bumped when the archive gains or renames a file.
BUNDLE_SCHEMA_VERSION: Final = 1

#: Oldest layout still readable. Kept separate so dropping support is a decision
#: rather than a side effect of bumping the version above.
MIN_BUNDLE_SCHEMA_VERSION: Final = 1

#: Suggested extension. The GUI file dialog uses it; nothing enforces it.
BUNDLE_SUFFIX: Final = ".zip"

MANIFEST_NAME: Final = "manifest.json"
CONFIG_NAME: Final = "config.toml"
COMMANDS_NAME: Final = "commands.ayris"
VARIABLES_NAME: Final = "variables.json"
MODELS_NAME: Final = "models.json"
SOUNDS_PREFIX: Final = "sounds/"

#: Refuse archives that would unpack to more than this. A profile is text plus a
#: handful of short sounds; anything larger is a mistake or a zip bomb.
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ENTRY_BYTES: Final = 64 * 1024 * 1024
MAX_ENTRIES: Final = 5000

#: Fields dropped from the exported settings. Matched as substrings of the key,
#: lower-cased, so ``openai_api_key`` and ``credential_ref`` both go.
_SECRET_KEY_PARTS: Final[tuple[str, ...]] = (
    "credential_ref",
    "api_key",
    "apikey",
    "access_key",
    "password",
    "secret",
    "token",
)

#: Scopes worth carrying between machines. ``local`` variables belong to a
#: single macro run and are meaningless after it.
_EXPORTED_SCOPES: Final[tuple[VariableScope, ...]] = (
    VariableScope.PROFILE,
    VariableScope.GLOBAL,
)


class ConflictPolicy(StrEnum):
    """What to do when an imported name already exists in the target profile."""

    #: Replace the existing command, variable or sound file.
    OVERWRITE = "overwrite"
    #: Keep both: the incoming one gets a numbered suffix.
    RENAME = "rename"
    #: Keep the existing one and drop the incoming one.
    SKIP = "skip"


#: Russian labels for the conflict radio buttons in the import dialog.
CONFLICT_LABELS: Final[dict[ConflictPolicy, str]] = {
    ConflictPolicy.OVERWRITE: "Перезаписать существующие",
    ConflictPolicy.RENAME: "Переименовать новые",
    ConflictPolicy.SKIP: "Пропустить существующие",
}

# ----------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """Header of a bundle: what produced it, and how much is inside.

    ``db_schema_version`` and ``config_schema_version`` are recorded next to the
    bundle's own version so a future reader can tell "written by a newer Ayris"
    apart from "written in a layout I do not understand" - the first is a
    warning, the second is a refusal.
    """

    profile_name: str
    schema_version: int = BUNDLE_SCHEMA_VERSION
    app_name: str = __app_name__
    app_version: str = __version__
    created_at: datetime = field(default_factory=utc_now)
    db_schema_version: int = DB_SCHEMA_VERSION
    config_schema_version: int = CONFIG_SCHEMA_VERSION
    counts: JsonObject = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "format": BUNDLE_FORMAT,
            "schema_version": self.schema_version,
            "app_name": self.app_name,
            "app_version": self.app_version,
            "created_at": self.created_at.isoformat(),
            "db_schema_version": self.db_schema_version,
            "config_schema_version": self.config_schema_version,
            "profile_name": self.profile_name,
            "counts": dict(self.counts),
        }

    @classmethod
    def from_json(cls, data: JsonObject) -> BundleManifest:
        """Validate a manifest read from an archive.

        Raises:
            ProfileError: The file is not an Ayris bundle, or its layout is
                newer or older than this build can read.
        """
        if data.get("format") != BUNDLE_FORMAT:
            raise ProfileError(
                f"not an Ayris profile bundle: format={data.get('format')!r}",
                user_message="Этот файл не похож на профиль Ayris.",
            )
        version = _as_int(data.get("schema_version"))
        if version is None:
            raise ProfileError(
                "manifest has no usable schema_version",
                user_message="Файл профиля повреждён: не указана версия формата.",
            )
        if version > BUNDLE_SCHEMA_VERSION:
            raise ProfileError(
                f"bundle schema {version} is newer than supported {BUNDLE_SCHEMA_VERSION}",
                user_message=(
                    "Файл создан более новой версией Ayris "
                    f"(формат {version}, поддерживается {BUNDLE_SCHEMA_VERSION}).\n"
                    "Обновите приложение и повторите импорт."
                ),
                recoverable=False,
            )
        if version < MIN_BUNDLE_SCHEMA_VERSION:
            raise ProfileError(
                f"bundle schema {version} is older than supported {MIN_BUNDLE_SCHEMA_VERSION}",
                user_message=(
                    f"Формат файла ({version}) слишком старый и больше не поддерживается."
                ),
                recoverable=False,
            )
        counts = data.get("counts")
        return cls(
            profile_name=_as_text(data.get("profile_name")) or "Импортированный профиль",
            schema_version=version,
            app_name=_as_text(data.get("app_name")) or __app_name__,
            app_version=_as_text(data.get("app_version")),
            created_at=_as_datetime(data.get("created_at")),
            db_schema_version=_as_int(data.get("db_schema_version")) or 0,
            config_schema_version=_as_int(data.get("config_schema_version")) or 0,
            counts=dict(counts) if isinstance(counts, dict) else {},
        )

    def compatibility_warnings(self) -> tuple[str, ...]:
        """Russian notes worth showing before applying an otherwise valid bundle."""
        notes: list[str] = []
        if self.db_schema_version > DB_SCHEMA_VERSION:
            notes.append(
                "Профиль создан более новой версией базы данных "
                f"({self.db_schema_version} > {DB_SCHEMA_VERSION}): "
                "часть данных может не примениться."
            )
        if self.config_schema_version > CONFIG_SCHEMA_VERSION:
            notes.append(
                "Настройки созданы более новой версией приложения: "
                "неизвестные параметры будут пропущены."
            )
        return tuple(notes)


@dataclass(frozen=True, slots=True)
class BundlePreview:
    """What an import would touch, computed without changing anything."""

    manifest: BundleManifest
    commands: tuple[str, ...] = ()
    folders: tuple[str, ...] = ()
    variables: tuple[str, ...] = ()
    models: tuple[str, ...] = ()
    sounds: tuple[str, ...] = ()
    has_config: bool = False
    #: Command names already present in the target profile.
    conflicts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        """Multi-line Russian summary for the confirmation dialog."""
        lines = [
            f"Профиль «{self.manifest.profile_name}»",
            f"Создан: {self.manifest.created_at:%d.%m.%Y %H:%M} "
            f"(Ayris {self.manifest.app_version or '?'})",
            f"Команд: {len(self.commands)}, папок: {len(self.folders)}, "
            f"переменных: {len(self.variables)}, звуков: {len(self.sounds)}",
        ]
        if self.models:
            lines.append(f"Ожидаемые модели: {len(self.models)}")
        if self.conflicts:
            lines.append(f"Совпадают имена команд: {len(self.conflicts)}")
        lines.extend(self.warnings)
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ImportReport:
    """Outcome of an import: what landed, what did not, and what is missing."""

    profile_name: str
    added_commands: tuple[str, ...] = ()
    replaced_commands: tuple[str, ...] = ()
    #: ``(original name, name it was stored under)`` for renamed conflicts.
    renamed_commands: tuple[tuple[str, str], ...] = ()
    skipped_commands: tuple[str, ...] = ()
    added_folders: tuple[str, ...] = ()
    added_variables: tuple[str, ...] = ()
    skipped_variables: tuple[str, ...] = ()
    added_sounds: tuple[str, ...] = ()
    skipped_sounds: tuple[str, ...] = ()
    #: Models the bundle expects that are not installed here.
    missing_models: tuple[str, ...] = ()
    config_applied: bool = False
    #: Settings that did not survive validation, by dotted name.
    dropped_settings: tuple[str, ...] = ()
    #: Backup written before the import, when the caller asked for one.
    backup: Path | None = None

    @property
    def total_commands(self) -> int:
        return len(self.added_commands) + len(self.replaced_commands) + len(self.renamed_commands)

    def describe(self) -> str:
        """Multi-line Russian report for the dialog that follows an import."""
        lines = [f"Импортировано из профиля «{self.profile_name}»."]
        if self.added_commands:
            lines.append(f"Добавлено команд: {len(self.added_commands)}")
        if self.replaced_commands:
            lines.append(f"Перезаписано команд: {len(self.replaced_commands)}")
        if self.renamed_commands:
            renamed = ", ".join(f"{old} → {new}" for old, new in self.renamed_commands[:5])
            lines.append(f"Переименовано команд: {len(self.renamed_commands)} ({renamed})")
        if self.skipped_commands:
            lines.append(f"Пропущено команд: {len(self.skipped_commands)}")
        if self.added_folders:
            lines.append(f"Создано папок: {len(self.added_folders)}")
        if self.added_variables:
            lines.append(f"Добавлено переменных: {len(self.added_variables)}")
        if self.added_sounds:
            lines.append(f"Добавлено звуков: {len(self.added_sounds)}")
        if self.missing_models:
            lines.append("Не хватает моделей: " + ", ".join(self.missing_models))
        if self.dropped_settings:
            lines.append(
                "Настройки, которые не удалось применить: " + ", ".join(self.dropped_settings)
            )
        if self.backup is not None:
            lines.append(f"Резервная копия: {self.backup.name}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# coercion helpers
# ----------------------------------------------------------------------
#
# Everything read out of an archive is untrusted: a hand-edited ``.ayris`` or a
# bundle from a future version may hold anything at all. These helpers turn
# "whatever was in the JSON" into the type the model layer expects, so the
# import code below can stay readable.


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _as_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        with suppress(ValueError):
            return int(value.strip())
    return None


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    return default


def _as_datetime(value: object) -> datetime:
    if isinstance(value, str):
        with suppress(ValueError):
            return datetime.fromisoformat(value)
    return utc_now()


def _as_object(value: object) -> JsonObject:
    return cast("JsonObject", value) if isinstance(value, dict) else {}


def _as_array(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _as_objects(value: object) -> list[JsonObject]:
    return [cast("JsonObject", item) for item in _as_array(value) if isinstance(item, dict)]


def _as_name_path(value: object) -> tuple[str, ...]:
    """Folder identity: the chain of folder names from the root."""
    parts = [_as_text(item).strip() for item in _as_array(value)]
    return tuple(part for part in parts if part)


def _folder_label(path: Sequence[str]) -> str:
    return " / ".join(path)


def _unique_name(name: str, taken: Iterable[str]) -> str:
    """``Свет`` -> ``Свет (2)``, ``Свет (3)``... Used by :attr:`ConflictPolicy.RENAME`."""
    existing = set(taken)
    if name not in existing:
        return name
    for index in range(2, 1000):
        candidate = f"{name} ({index})"
        if candidate not in existing:
            return candidate
    # A thousand collisions means something is generating names in a loop; a
    # timestamp is ugly but always terminates.
    return f"{name} ({utc_now():%Y%m%d%H%M%S})"


# ----------------------------------------------------------------------
# archive plumbing
# ----------------------------------------------------------------------


def _zip_time(when: datetime) -> tuple[int, int, int, int, int, int]:
    """Zip timestamps cannot predate 1980; clamp instead of failing on a bad clock."""
    return (max(1980, when.year), when.month, when.day, when.hour, when.minute, when.second)


def _zip_info(name: str, when: datetime) -> zipfile.ZipInfo:
    """Entry header with a fixed timestamp, so exports are reproducible.

    Encoding is left to :mod:`zipfile`, which is the only place that can get it
    right: it writes the name as UTF-8 and sets bit 11 of the general purpose
    flag whenever the name is not pure ASCII. Setting the bit here would be
    theatre - :meth:`zipfile.ZipFile.open` resets ``flag_bits`` before writing -
    so the guarantee the Cyrillic command and folder names depend on is
    asserted in the tests instead.
    """
    info = zipfile.ZipInfo(name, date_time=_zip_time(when))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def _is_unsafe_name(name: str) -> bool:
    """Reject anything that could escape the directory we unpack into."""
    if not name or name.startswith(("/", "\\")) or ":" in name:
        return True
    if "\\" in name:  # zip separators are always "/"; a backslash is a red flag
        return True
    return any(part in {"..", "."} for part in name.split("/"))


def _entries(bundle: zipfile.ZipFile, archive: Path) -> dict[str, zipfile.ZipInfo]:
    """Index the archive, refusing anything hostile or absurdly large.

    Raises:
        ProfileError: The archive is malformed, escapes its directory, or would
            unpack to more than :data:`MAX_TOTAL_BYTES`.
    """
    infos = bundle.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ProfileError(
            f"{archive}: {len(infos)} entries exceeds the {MAX_ENTRIES} limit",
            user_message="В файле слишком много элементов — он не похож на профиль.",
            recoverable=False,
        )
    total = 0
    found: dict[str, zipfile.ZipInfo] = {}
    for info in infos:
        if info.is_dir():
            continue
        if _is_unsafe_name(info.filename):
            raise ProfileError(
                f"{archive}: unsafe entry name {info.filename!r}",
                user_message="Файл профиля содержит недопустимые пути и не будет распакован.",
                recoverable=False,
            )
        if info.file_size > MAX_ENTRY_BYTES:
            raise ProfileError(
                f"{archive}: entry {info.filename!r} is {info.file_size} bytes",
                user_message=f"Элемент «{info.filename}» слишком большой для профиля.",
                recoverable=False,
            )
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ProfileError(
                f"{archive}: uncompressed size exceeds {MAX_TOTAL_BYTES} bytes",
                user_message="Распакованный профиль занял бы слишком много места.",
                recoverable=False,
            )
        found[info.filename] = info
    return found


def _read_bytes(bundle: zipfile.ZipFile, name: str, archive: Path) -> bytes:
    try:
        return bundle.read(name)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise ProfileError(
            f"{archive}: cannot read {name}: {exc}",
            user_message=f"Не удалось прочитать «{name}» из файла профиля.",
        ) from exc


def _read_json(bundle: zipfile.ZipFile, name: str, archive: Path) -> JsonObject:
    raw = _read_bytes(bundle, name, archive)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(
            f"{archive}: {name} is not valid JSON: {exc}",
            user_message=f"Файл «{name}» внутри профиля повреждён.",
        ) from exc
    if not isinstance(parsed, dict):
        raise ProfileError(
            f"{archive}: {name} is {type(parsed).__name__}, expected an object",
            user_message=f"Файл «{name}» внутри профиля имеет неожиданную структуру.",
        )
    return cast("JsonObject", parsed)


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------


def strip_secrets(data: Mapping[str, Any]) -> tuple[JsonObject, tuple[str, ...]]:
    """Copy ``data`` without any field whose name suggests a credential.

    Settings never hold a key to begin with - :mod:`ayris.core.secrets` keeps
    those in the Windows Credential Manager - so this is belt and braces. It
    also drops the ``credential_ref`` *names*, which are useless on another
    machine and needlessly describe the owner's accounts.

    Returns:
        The cleaned mapping and the dotted names of what was removed.
    """
    removed: list[str] = []

    def walk(node: Mapping[str, Any], prefix: str) -> JsonObject:
        clean: JsonObject = {}
        for key, value in node.items():
            dotted = f"{prefix}{key}"
            lowered = key.lower()
            if any(part in lowered for part in _SECRET_KEY_PARTS):
                removed.append(dotted)
                continue
            if isinstance(value, dict):
                clean[key] = walk(cast("Mapping[str, Any]", value), f"{dotted}.")
            else:
                clean[key] = value
        return clean

    return walk(data, ""), tuple(removed)


def _folder_paths(folders: Sequence[CommandFolder]) -> dict[int, tuple[str, ...]]:
    """Map folder id to its chain of names, which is how bundles refer to folders.

    Numeric ids mean nothing on the receiving machine, and a name path survives
    the round trip while staying readable to anyone who opens the archive.
    """
    by_id = {folder.id: folder for folder in folders if folder.id is not None}
    paths: dict[int, tuple[str, ...]] = {}
    for folder_id, folder in by_id.items():
        chain: list[str] = []
        current: CommandFolder | None = folder
        seen: set[int] = set()
        while current is not None and current.id is not None and current.id not in seen:
            seen.add(current.id)
            chain.append(current.name)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
        paths[folder_id] = tuple(reversed(chain))
    return paths


def _commands_payload(repositories: Repositories, profile_id: int) -> JsonObject:
    folders = repositories.folders.list_for_profile(profile_id)
    paths = _folder_paths(folders)
    folder_entries = [
        {"path": list(paths[folder.id]), "sort_order": folder.sort_order}
        for folder in folders
        if folder.id is not None and paths.get(folder.id)
    ]
    command_entries: list[JsonObject] = []
    for command in repositories.commands.list_for_profile(profile_id):
        if command.id is None:  # pragma: no cover - rows always carry an id
            continue
        triggers = [
            {
                "type": str(trigger.type),
                "payload": dict(trigger.payload),
                "fuzzy": trigger.fuzzy,
                "priority": trigger.priority,
            }
            for trigger in repositories.triggers.list_for_command(command.id)
        ]
        folder_path = paths.get(command.folder_id) if command.folder_id is not None else None
        command_entries.append(
            {
                "name": command.name,
                "folder": list(folder_path) if folder_path else None,
                "description": command.description,
                "tags": list(command.tags),
                "enabled": command.enabled,
                "priority": command.priority,
                "cooldown_ms": command.cooldown_ms,
                "require_admin": command.require_admin,
                "actions": [dict(action) for action in command.actions],
                "triggers": triggers,
            }
        )
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "folders": folder_entries,
        "commands": command_entries,
    }


def _variables_payload(repositories: Repositories, profile_id: int) -> JsonObject:
    entries: list[JsonObject] = []
    for scope in _EXPORTED_SCOPES:
        owner = profile_id if scope is VariableScope.PROFILE else None
        for variable in repositories.variables.list_all(scope=scope, profile_id=owner):
            if not variable.persistent:
                continue
            entries.append(
                {
                    "name": variable.name,
                    "scope": str(variable.scope),
                    "type": str(variable.type),
                    "value": variable.value,
                }
            )
    return {"schema_version": BUNDLE_SCHEMA_VERSION, "variables": entries}


def _models_payload(repositories: Repositories) -> JsonObject:
    """Model manifests only - the weights stay where they are.

    A Vosk model is hundreds of megabytes; shipping it inside a profile would
    turn a 30 KB file into an unusable one. The receiving side reports what is
    missing and the user downloads it through the models tab.
    """
    entries = [
        {
            "kind": str(model.kind),
            "name": model.name,
            "version": model.version,
            "sha256": model.sha256,
            "is_active": model.is_active,
        }
        for model in repositories.models.list_all()
    ]
    return {"schema_version": BUNDLE_SCHEMA_VERSION, "models": entries}


def _config_bytes(settings: Settings) -> tuple[bytes, tuple[str, ...]]:
    """Render settings to TOML with the secret-bearing fields removed.

    Round-tripping through :func:`settings_from_mapping` rather than editing the
    text keeps the exported file a *valid* config: the stripped fields come back
    as their defaults instead of leaving holes for the reader to trip over.

    Returns:
        The file contents and the dotted names of the removed fields.
    """
    cleaned, removed = strip_secrets(dump_settings(settings))
    stripped, dropped = settings_from_mapping(cleaned)
    if dropped:  # pragma: no cover - a valid dump should always revalidate
        _log.warning("export dropped settings during round-trip: %s", ", ".join(dropped))
    with TemporaryDirectory(prefix="ayris_export_") as workspace:
        staged = Path(workspace) / CONFIG_NAME
        save_settings(stripped, staged)
        return staged.read_bytes(), removed


def _sound_files(sounds_dir: Path) -> list[tuple[str, Path]]:
    """User sound files as ``(archive name, source)``, sorted for a stable archive."""
    if not sounds_dir.is_dir():
        return []
    found: list[tuple[str, Path]] = []
    for source in sorted(sounds_dir.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(sounds_dir).as_posix()
        if _is_unsafe_name(relative):  # pragma: no cover - defensive
            _log.warning("skipping sound with unusable name: %s", source)
            continue
        found.append((f"{SOUNDS_PREFIX}{relative}", source))
    return found


def export_bundle(
    repositories: Repositories,
    destination: Path,
    *,
    profile: Profile | None = None,
    settings: Settings | None = None,
    paths: AppPaths | None = None,
    include_sounds: bool = True,
) -> BundleManifest:
    """Write ``profile`` to ``destination`` as a portable ``.zip``.

    Args:
        repositories: Storage to read the profile out of.
        destination: Target file. Overwritten if it exists.
        profile: Profile to export. Defaults to the active one.
        settings: Settings to export. Defaults to reading the profile's
            ``config.toml``; when there is none, the archive simply has no
            settings and the importer keeps its own.
        paths: Path set to read sounds from. Defaults to the process-wide one.
        include_sounds: Copy user sounds into the archive.

    Returns:
        The manifest that was written, counts included.

    Raises:
        ProfileError: The profile has no id, or the file could not be written.
    """
    resolved_paths = paths if paths is not None else get_paths()
    source = profile if profile is not None else repositories.profiles.active()
    if source is None or source.id is None:
        raise ProfileError(
            "cannot export a profile that has no id",
            user_message="Нечего экспортировать: активный профиль не выбран.",
        )

    commands = _commands_payload(repositories, source.id)
    variables = _variables_payload(repositories, source.id)
    models = _models_payload(repositories)
    sounds = _sound_files(resolved_paths.sounds_dir) if include_sounds else []

    config_payload: bytes | None = None
    if settings is not None:
        config_payload, removed = _config_bytes(settings)
        if removed:
            _log.debug("export stripped %d secret-bearing fields", len(removed))
    elif resolved_paths.config_file.is_file():
        loaded, _dropped = load_settings(resolved_paths.config_file)
        config_payload, _removed = _config_bytes(loaded)

    manifest = BundleManifest(
        profile_name=source.name,
        counts={
            "commands": len(_as_objects(commands.get("commands"))),
            "folders": len(_as_objects(commands.get("folders"))),
            "variables": len(_as_objects(variables.get("variables"))),
            "models": len(_as_objects(models.get("models"))),
            "sounds": len(sounds),
        },
    )
    _write_archive(destination, manifest, commands, variables, models, sounds, config_payload)
    _log.info(
        "exported profile %r to %s (%d commands, %d sounds)",
        source.name,
        destination,
        manifest.counts.get("commands"),
        len(sounds),
    )
    return manifest


def _write_archive(
    destination: Path,
    manifest: BundleManifest,
    commands: JsonObject,
    variables: JsonObject,
    models: JsonObject,
    sounds: Sequence[tuple[str, Path]],
    config_payload: bytes | None,
) -> None:
    """Build the archive beside ``destination``, then move it into place.

    A half-written export must not replace a good one, so the file the user sees
    only ever appears complete.
    """
    when = manifest.created_at
    temporary = destination.with_name(f"{destination.name}.part")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr(_zip_info(MANIFEST_NAME, when), _json_bytes(manifest.to_json()))
            if config_payload is not None:
                bundle.writestr(_zip_info(CONFIG_NAME, when), config_payload)
            bundle.writestr(_zip_info(COMMANDS_NAME, when), _json_bytes(commands))
            bundle.writestr(_zip_info(VARIABLES_NAME, when), _json_bytes(variables))
            bundle.writestr(_zip_info(MODELS_NAME, when), _json_bytes(models))
            for name, source in sounds:
                bundle.writestr(_zip_info(name, when), source.read_bytes())
        temporary.replace(destination)
    except OSError as exc:
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ProfileError(
            f"cannot write bundle {destination}: {exc}",
            user_message=(
                f"Не удалось сохранить файл профиля:\n{destination}\nПроверьте права доступа."
            ),
        ) from exc


def _json_bytes(payload: JsonObject) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


# ----------------------------------------------------------------------
# reading a bundle
# ----------------------------------------------------------------------


@contextmanager
def _open_bundle(archive: Path) -> Iterator[zipfile.ZipFile]:
    """Open an archive, turning every way it can fail into a :class:`ProfileError`."""
    try:
        bundle = zipfile.ZipFile(archive)
    except FileNotFoundError as exc:
        raise ProfileError(
            f"bundle not found: {archive}",
            user_message=f"Файл профиля не найден:\n{archive}",
        ) from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise ProfileError(
            f"cannot open bundle {archive}: {exc}",
            user_message=f"Не удалось открыть файл профиля:\n{archive}\nВозможно, он повреждён.",
        ) from exc
    try:
        yield bundle
    finally:
        bundle.close()


def read_manifest(archive: Path) -> BundleManifest:
    """Read and validate just the header of a bundle.

    Raises:
        ProfileError: The file is unreadable, not a bundle, or written in a
            layout this build cannot handle.
    """
    with _open_bundle(archive) as bundle:
        entries = _entries(bundle, archive)
        if MANIFEST_NAME not in entries:
            raise ProfileError(
                f"{archive}: no {MANIFEST_NAME}",
                user_message="Это не файл профиля Ayris: внутри нет манифеста.",
                recoverable=False,
            )
        return BundleManifest.from_json(_read_json(bundle, MANIFEST_NAME, archive))


def preview_bundle(
    archive: Path,
    *,
    repositories: Repositories | None = None,
    profile_id: int | None = None,
) -> BundlePreview:
    """Describe what a bundle holds without touching anything.

    Args:
        archive: Bundle to inspect.
        repositories: Optional storage used to spot name conflicts.
        profile_id: Profile the conflicts would be checked against.

    Raises:
        ProfileError: Same conditions as :func:`read_manifest`.
    """
    with _open_bundle(archive) as bundle:
        entries = _entries(bundle, archive)
        if MANIFEST_NAME not in entries:
            raise ProfileError(
                f"{archive}: no {MANIFEST_NAME}",
                user_message="Это не файл профиля Ayris: внутри нет манифеста.",
                recoverable=False,
            )
        manifest = BundleManifest.from_json(_read_json(bundle, MANIFEST_NAME, archive))
        commands: tuple[str, ...] = ()
        folders: tuple[str, ...] = ()
        if COMMANDS_NAME in entries:
            payload = _read_json(bundle, COMMANDS_NAME, archive)
            commands = tuple(
                _as_text(entry.get("name"))
                for entry in _as_objects(payload.get("commands"))
                if _as_text(entry.get("name"))
            )
            folders = tuple(
                _folder_label(_as_name_path(entry.get("path")))
                for entry in _as_objects(payload.get("folders"))
                if _as_name_path(entry.get("path"))
            )
        variables: tuple[str, ...] = ()
        if VARIABLES_NAME in entries:
            payload = _read_json(bundle, VARIABLES_NAME, archive)
            variables = tuple(
                _as_text(entry.get("name"))
                for entry in _as_objects(payload.get("variables"))
                if _as_text(entry.get("name"))
            )
        models: tuple[str, ...] = ()
        if MODELS_NAME in entries:
            payload = _read_json(bundle, MODELS_NAME, archive)
            models = tuple(_model_label(entry) for entry in _as_objects(payload.get("models")))
        sounds = tuple(
            name[len(SOUNDS_PREFIX) :] for name in sorted(entries) if name.startswith(SOUNDS_PREFIX)
        )

    conflicts: tuple[str, ...] = ()
    if repositories is not None and profile_id is not None:
        taken = {command.name for command in repositories.commands.list_for_profile(profile_id)}
        conflicts = tuple(name for name in commands if name in taken)
    return BundlePreview(
        manifest=manifest,
        commands=commands,
        folders=folders,
        variables=variables,
        models=models,
        sounds=sounds,
        has_config=CONFIG_NAME in entries,
        conflicts=conflicts,
        warnings=manifest.compatibility_warnings(),
    )


def _model_label(entry: JsonObject) -> str:
    kind = _as_text(entry.get("kind")) or "?"
    name = _as_text(entry.get("name")) or "?"
    version = _as_text(entry.get("version"))
    return f"{kind}/{name} {version}".strip()


# ----------------------------------------------------------------------
# import
# ----------------------------------------------------------------------


class _FileGuard:
    """Stage file writes so a failure anywhere can put the originals back.

    Every replaced file is first moved into ``workspace``, which lives inside the
    profile root on purpose: :meth:`Path.replace` is only atomic within one
    filesystem, and a temp directory could easily be on another drive.

    On Windows a file another process has open cannot be moved. That surfaces
    here as a :class:`ProfileError` naming the file, which is the honest outcome:
    better to refuse the import than to half-apply it.
    """

    __slots__ = ("_counter", "_restore", "_workspace")

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        self._restore: list[tuple[Path, Path | None]] = []
        self._counter = 0

    def write(self, target: Path, data: bytes) -> None:
        """Replace ``target`` with ``data``, remembering how to undo it."""
        self._stash(target)
        staged = self._slot("new")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            staged.replace(target)
        except OSError as exc:
            raise ProfileError(
                f"cannot write {target}: {exc}",
                user_message=f"Не удалось записать файл:\n{target}\nПроверьте права доступа.",
            ) from exc

    def _stash(self, target: Path) -> None:
        if not target.exists():
            self._restore.append((target, None))
            return
        backup = self._slot("old")
        try:
            target.replace(backup)
        except OSError as exc:
            raise ProfileError(
                f"cannot move {target} aside: {exc}",
                user_message=(
                    f"Файл занят другой программой и не может быть заменён:\n{target}\n"
                    "Закройте программу и повторите импорт."
                ),
            ) from exc
        self._restore.append((target, backup))

    def _slot(self, prefix: str) -> Path:
        self._counter += 1
        self._workspace.mkdir(parents=True, exist_ok=True)
        return self._workspace / f"{prefix}_{self._counter:04d}"

    def rollback(self) -> None:
        """Undo every write, newest first. Never raises: it runs while handling an error."""
        for target, backup in reversed(self._restore):
            try:
                if backup is not None:
                    backup.replace(target)
                else:
                    target.unlink(missing_ok=True)
            except OSError:
                _log.exception("cannot roll back %s", target)
        self._restore.clear()
        self._cleanup()

    def commit(self) -> None:
        """Drop the staged originals. The writes stand."""
        self._restore.clear()
        self._cleanup()

    def _cleanup(self) -> None:
        shutil.rmtree(self._workspace, ignore_errors=True)


def _resolve_conflict(
    name: str,
    *,
    exists: bool,
    policy: ConflictPolicy,
    taken: Iterable[str],
) -> str | None:
    """Final name for an incoming item, or ``None`` when it should be skipped."""
    if not exists:
        return name
    if policy is ConflictPolicy.SKIP:
        return None
    if policy is ConflictPolicy.RENAME:
        return _unique_name(name, taken)
    return name


@dataclass(slots=True)
class _CommandOutcome:
    """Mutable tally filled in while commands are applied."""

    added: list[str] = field(default_factory=list)
    replaced: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    folders: list[str] = field(default_factory=list)


def _folder_index(repositories: Repositories, profile_id: int) -> dict[tuple[str, ...], int]:
    """Existing folders of a profile, keyed by their name path."""
    folders = repositories.folders.list_for_profile(profile_id)
    paths = _folder_paths(folders)
    return {path: folder_id for folder_id, path in paths.items() if path}


def _ensure_folder(
    path: Sequence[str],
    *,
    repositories: Repositories,
    profile_id: int,
    index: dict[tuple[str, ...], int],
    created: list[str],
) -> int | None:
    """Return the id of ``path``, creating the missing links of the chain."""
    parent: int | None = None
    for depth in range(len(path)):
        key = tuple(path[: depth + 1])
        known = index.get(key)
        if known is None:
            folder = repositories.folders.create(
                CommandFolder(name=key[-1], profile_id=profile_id, parent_id=parent)
            )
            if folder.id is None:  # pragma: no cover - insert always yields an id
                raise ProfileError(
                    f"folder {_folder_label(key)!r} was created without an id",
                    user_message="Не удалось создать папку команд при импорте.",
                )
            index[key] = folder.id
            created.append(_folder_label(key))
            known = folder.id
        parent = known
    return parent


def _trigger_from_json(entry: JsonObject, command_id: int) -> Trigger:
    raw_type = _as_text(entry.get("type"))
    try:
        trigger_type = TriggerType(raw_type)
    except ValueError:
        trigger_type = TriggerType.VOICE
    return Trigger(
        command_id=command_id,
        type=trigger_type,
        payload=_as_object(entry.get("payload")),
        fuzzy=_as_bool(entry.get("fuzzy"), default=True),
        priority=_as_int(entry.get("priority")) or 0,
    )


def _apply_commands(
    payload: JsonObject,
    *,
    repositories: Repositories,
    profile_id: int,
    policy: ConflictPolicy,
) -> _CommandOutcome:
    """Insert the bundle's folders and commands into ``profile_id``.

    Raises:
        ProfileError: An entry has no name. Reading and applying are
            deliberately interleaved, so a bundle that goes bad half way aborts
            the surrounding transaction instead of silently importing a prefix.
    """
    outcome = _CommandOutcome()
    index = _folder_index(repositories, profile_id)
    taken = {command.name for command in repositories.commands.list_for_profile(profile_id)}

    for entry in _as_objects(payload.get("folders")):
        path = _as_name_path(entry.get("path"))
        if path:
            _ensure_folder(
                path,
                repositories=repositories,
                profile_id=profile_id,
                index=index,
                created=outcome.folders,
            )

    for position, entry in enumerate(_as_objects(payload.get("commands")), start=1):
        original = _as_text(entry.get("name")).strip()
        if not original:
            raise ProfileError(
                f"command #{position} in the bundle has no name",
                user_message=f"Команда №{position} в файле профиля без имени — импорт отменён.",
            )
        target = _resolve_conflict(original, exists=original in taken, policy=policy, taken=taken)
        if target is None:
            outcome.skipped.append(original)
            continue
        if original in taken:
            if target == original:
                existing = repositories.commands.get_by_name(profile_id, original)
                if existing is not None and existing.id is not None:
                    repositories.commands.delete(existing.id)
                outcome.replaced.append(original)
            else:
                outcome.renamed.append((original, target))
        else:
            outcome.added.append(target)
        taken.add(target)

        folder_path = _as_name_path(entry.get("folder"))
        folder_id = (
            _ensure_folder(
                folder_path,
                repositories=repositories,
                profile_id=profile_id,
                index=index,
                created=outcome.folders,
            )
            if folder_path
            else None
        )
        command = repositories.commands.create(
            Command(
                name=target,
                profile_id=profile_id,
                folder_id=folder_id,
                description=_as_text(entry.get("description")),
                tags=tuple(_as_text(tag) for tag in _as_array(entry.get("tags")) if _as_text(tag)),
                enabled=_as_bool(entry.get("enabled"), default=True),
                priority=_as_int(entry.get("priority")) or 0,
                cooldown_ms=_as_int(entry.get("cooldown_ms")) or 0,
                require_admin=_as_bool(entry.get("require_admin"), default=False),
                actions=tuple(_as_objects(entry.get("actions"))),
            )
        )
        if command.id is None:  # pragma: no cover - insert always yields an id
            raise ProfileError(
                f"command {target!r} was created without an id",
                user_message="Не удалось сохранить команду при импорте.",
            )
        for trigger in _as_objects(entry.get("triggers")):
            repositories.triggers.add(_trigger_from_json(trigger, command.id))
    return outcome


def _apply_variables(
    payload: JsonObject,
    *,
    repositories: Repositories,
    profile_id: int,
    policy: ConflictPolicy,
) -> tuple[list[str], list[str]]:
    """Insert the bundle's variables. Returns ``(applied, skipped)`` names.

    :attr:`ConflictPolicy.RENAME` behaves like ``SKIP`` here: the imported
    commands refer to variables *by name*, so inventing a new name would give
    them a variable nothing reads.
    """
    applied: list[str] = []
    skipped: list[str] = []
    for entry in _as_objects(payload.get("variables")):
        name = _as_text(entry.get("name")).strip()
        if not name:
            continue
        try:
            scope = VariableScope(_as_text(entry.get("scope")))
        except ValueError:
            scope = VariableScope.PROFILE
        if scope not in _EXPORTED_SCOPES:
            skipped.append(name)
            continue
        owner = profile_id if scope is VariableScope.PROFILE else None
        if (
            policy is not ConflictPolicy.OVERWRITE
            and repositories.variables.get(name, scope=scope, profile_id=owner) is not None
        ):
            skipped.append(name)
            continue
        try:
            var_type = VariableType(_as_text(entry.get("type")))
        except ValueError:
            var_type = None
        repositories.variables.set(
            name,
            entry.get("value"),
            scope=scope,
            profile_id=owner,
            var_type=var_type,
            persistent=True,
        )
        applied.append(name)
    return applied, skipped


def _missing_models(payload: JsonObject, repositories: Repositories) -> list[str]:
    """Which of the bundle's models are not installed here.

    Weights are never shipped inside a profile, so this is the list the import
    report tells the user to download.
    """
    installed = {(model.kind, model.name) for model in repositories.models.list_all()}
    missing: list[str] = []
    for entry in _as_objects(payload.get("models")):
        kind = _as_text(entry.get("kind"))
        name = _as_text(entry.get("name"))
        if kind and name and (kind, name) not in installed:
            missing.append(_model_label(entry))
    return missing


def _apply_sounds(
    bundle: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    *,
    archive: Path,
    sounds_dir: Path,
    guard: _FileGuard,
    policy: ConflictPolicy,
) -> tuple[list[str], list[str]]:
    """Copy the bundle's sound files into the profile. Returns ``(added, skipped)``."""
    added: list[str] = []
    skipped: list[str] = []
    for name in sorted(entries):
        if not name.startswith(SOUNDS_PREFIX):
            continue
        relative = name[len(SOUNDS_PREFIX) :]
        if not relative or _is_unsafe_name(relative):
            continue
        target = sounds_dir / PurePosixPath(relative)
        if target.exists():
            if policy is ConflictPolicy.SKIP:
                skipped.append(relative)
                continue
            if policy is ConflictPolicy.RENAME:
                siblings = (
                    {item.stem for item in target.parent.iterdir()}
                    if target.parent.is_dir()
                    else set()
                )
                target = target.with_stem(_unique_name(target.stem, siblings))
        guard.write(target, _read_bytes(bundle, name, archive))
        added.append(target.name)
    return added, skipped


def _apply_config(payload: bytes, *, config_file: Path, guard: _FileGuard) -> tuple[str, ...]:
    """Write the bundle's settings over the local ones. Returns the dropped fields.

    The incoming file is parsed and re-serialised rather than copied: that
    validates it, fills in whatever the sending version did not have, and - by
    stripping secrets a second time - guarantees a hand-edited archive cannot
    repoint the local credential references at entries of its choosing.
    """
    with TemporaryDirectory(prefix="ayris_import_") as workspace:
        staged = Path(workspace) / CONFIG_NAME
        staged.write_bytes(payload)
        try:
            incoming, dropped = load_settings(staged)
        except ConfigError as exc:
            raise ProfileError(
                f"bundle config is unusable: {exc}",
                user_message="Настройки внутри файла профиля повреждены — импорт отменён.",
            ) from exc
        cleaned, _removed = strip_secrets(dump_settings(incoming))
        normalised, _dropped = settings_from_mapping(cleaned)
        target = Path(workspace) / "normalised.toml"
        save_settings(normalised, target)
        guard.write(config_file, target.read_bytes())
    return dropped


def import_bundle(
    archive: Path,
    repositories: Repositories,
    *,
    profile: Profile | None = None,
    policy: ConflictPolicy = ConflictPolicy.RENAME,
    paths: AppPaths | None = None,
    include_sounds: bool = True,
    apply_config: bool = False,
) -> ImportReport:
    """Apply a bundle to a profile, all of it or none of it.

    Files are written first, through a :class:`_FileGuard`, and the database
    work runs inside one transaction. A failure in the second phase rolls back
    the first, so a bundle that turns out to be broken half way through leaves
    the profile exactly as it was.

    Args:
        archive: Bundle to read.
        repositories: Storage to write into.
        profile: Target profile. Defaults to the active one.
        policy: What to do about names that already exist.
        paths: Path set for sounds and settings. Defaults to the process-wide one.
        include_sounds: Copy the bundle's sound files.
        apply_config: Also overwrite the local settings. Off by default:
            ``config.toml`` belongs to the installation, not to one profile, so
            importing a colleague's commands should not silently change the
            microphone or the theme.

    Returns:
        A report of what was added, replaced, renamed, skipped and missing.

    Raises:
        ProfileError: The bundle is invalid or unreadable, the target profile
            has no id, or a file could not be written.
    """
    resolved_paths = paths if paths is not None else get_paths()
    target = profile if profile is not None else repositories.profiles.active()
    if target is None or target.id is None:
        raise ProfileError(
            "cannot import into a profile that has no id",
            user_message="Не выбран профиль, в который импортировать.",
        )

    guard = _FileGuard(resolved_paths.cache_dir / f"import_{uuid4().hex}")
    with _open_bundle(archive) as bundle:
        entries = _entries(bundle, archive)
        if MANIFEST_NAME not in entries:
            raise ProfileError(
                f"{archive}: no {MANIFEST_NAME}",
                user_message="Это не файл профиля Ayris: внутри нет манифеста.",
                recoverable=False,
            )
        manifest = BundleManifest.from_json(_read_json(bundle, MANIFEST_NAME, archive))
        try:
            report = _apply_bundle(
                bundle,
                entries,
                archive=archive,
                manifest=manifest,
                repositories=repositories,
                profile_id=target.id,
                policy=policy,
                paths=resolved_paths,
                guard=guard,
                include_sounds=include_sounds,
                apply_config=apply_config,
            )
        except Exception:
            guard.rollback()
            _log.exception("import of %s failed and was rolled back", archive)
            raise
        guard.commit()

    _log.info(
        "imported %s into profile %r: %d commands, %d variables",
        archive.name,
        target.name,
        report.total_commands,
        len(report.added_variables),
    )
    return report


def _apply_bundle(
    bundle: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    *,
    archive: Path,
    manifest: BundleManifest,
    repositories: Repositories,
    profile_id: int,
    policy: ConflictPolicy,
    paths: AppPaths,
    guard: _FileGuard,
    include_sounds: bool,
    apply_config: bool,
) -> ImportReport:
    """Phase two of :func:`import_bundle`, with the rollback left to the caller."""
    dropped_settings: tuple[str, ...] = ()
    config_applied = False
    if apply_config and CONFIG_NAME in entries:
        dropped_settings = _apply_config(
            _read_bytes(bundle, CONFIG_NAME, archive),
            config_file=paths.config_file,
            guard=guard,
        )
        config_applied = True

    added_sounds: list[str] = []
    skipped_sounds: list[str] = []
    if include_sounds:
        added_sounds, skipped_sounds = _apply_sounds(
            bundle,
            entries,
            archive=archive,
            sounds_dir=paths.sounds_dir,
            guard=guard,
            policy=policy,
        )

    commands = _CommandOutcome()
    variables: list[str] = []
    skipped_variables: list[str] = []
    missing_models: list[str] = []
    with repositories.database.transaction():
        if COMMANDS_NAME in entries:
            commands = _apply_commands(
                _read_json(bundle, COMMANDS_NAME, archive),
                repositories=repositories,
                profile_id=profile_id,
                policy=policy,
            )
        if VARIABLES_NAME in entries:
            variables, skipped_variables = _apply_variables(
                _read_json(bundle, VARIABLES_NAME, archive),
                repositories=repositories,
                profile_id=profile_id,
                policy=policy,
            )
        if MODELS_NAME in entries:
            missing_models = _missing_models(_read_json(bundle, MODELS_NAME, archive), repositories)

    return ImportReport(
        profile_name=manifest.profile_name,
        added_commands=tuple(commands.added),
        replaced_commands=tuple(commands.replaced),
        renamed_commands=tuple(commands.renamed),
        skipped_commands=tuple(commands.skipped),
        added_folders=tuple(dict.fromkeys(commands.folders)),
        added_variables=tuple(variables),
        skipped_variables=tuple(skipped_variables),
        added_sounds=tuple(added_sounds),
        skipped_sounds=tuple(skipped_sounds),
        missing_models=tuple(missing_models),
        config_applied=config_applied,
        dropped_settings=dropped_settings,
    )


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SUFFIX",
    "CONFLICT_LABELS",
    "MIN_BUNDLE_SCHEMA_VERSION",
    "BundleManifest",
    "BundlePreview",
    "ConflictPolicy",
    "ImportReport",
    "export_bundle",
    "import_bundle",
    "preview_bundle",
    "read_manifest",
    "strip_secrets",
]
