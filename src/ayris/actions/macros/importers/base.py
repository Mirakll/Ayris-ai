"""Shared importer contract, preview state and atomic database application."""

from __future__ import annotations

import logging
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from ayris.actions.macros.schema import CommandModel, VariableModel
from ayris.actions.macros.serializer import (
    command_to_row,
    initial_variables,
    triggers_to_rows,
)
from ayris.actions.macros.validator import ensure_valid
from ayris.core.database import Database
from ayris.core.models import CommandFolder
from ayris.core.repositories import (
    CommandRepository,
    FolderRepository,
    TriggerRepository,
    VariableRepository,
)

_log = logging.getLogger(__name__)


class ConflictStrategy(StrEnum):
    """How an import handles a command whose name already exists."""

    RENAME = "rename"
    REPLACE = "replace"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class ImportNotice:
    """A recoverable problem tied to a source location or command."""

    message: str
    command: str = ""
    source: str = ""

    @property
    def text(self) -> str:
        prefix = f"{self.command}: " if self.command else ""
        suffix = f" ({self.source})" if self.source else ""
        return f"{prefix}{self.message}{suffix}"


@dataclass(frozen=True, slots=True)
class UnsupportedItem:
    """One source construct for which Ayris has no safe equivalent."""

    command: str
    kind: str
    source: str = ""
    reason: str = "нет соответствующего блока Ayris"

    @property
    def text(self) -> str:
        source = f": {self.source}" if self.source else ""
        return f"{self.command}: {self.kind}{source} — {self.reason}"


@dataclass(frozen=True, slots=True)
class ImportedSound:
    """An external sound that must become a portable profile asset."""

    source: Path
    target_name: str
    command: str = ""


@dataclass(slots=True)
class ImportResult:
    """Everything learned from a file, before any persistent change."""

    source: Path
    profile_name: str = ""
    commands: list[CommandModel] = field(default_factory=list)
    folders: set[tuple[str, ...]] = field(default_factory=set)
    variables: list[VariableModel] = field(default_factory=list)
    sounds: list[ImportedSound] = field(default_factory=list)
    warnings: list[ImportNotice] = field(default_factory=list)
    unsupported: list[UnsupportedItem] = field(default_factory=list)
    total_commands: int = 0
    skipped: Counter[str] = field(default_factory=Counter)

    @property
    def imported_commands(self) -> int:
        return len(self.commands)

    @property
    def summary(self) -> str:
        return f"Импортировано {self.imported_commands} из {self.total_commands} команд"


@dataclass(slots=True)
class ImportPreview:
    """UI-neutral preview: selection, destination and conflict choice."""

    result: ImportResult
    target_folder: tuple[str, ...] = ()
    conflict_strategy: ConflictStrategy = ConflictStrategy.RENAME
    selected: set[int] | None = None

    def __post_init__(self) -> None:
        if self.selected is None:
            self.selected = set(range(len(self.result.commands)))

    @property
    def commands(self) -> tuple[CommandModel, ...]:
        selected = self.selected or set()
        return tuple(
            command for index, command in enumerate(self.result.commands) if index in selected
        )

    def set_selected(self, index: int, selected: bool) -> None:
        if not 0 <= index < len(self.result.commands):
            raise IndexError(index)
        if self.selected is None:
            self.selected = set(range(len(self.result.commands)))
        if selected:
            self.selected.add(index)
        else:
            self.selected.discard(index)


@dataclass(frozen=True, slots=True)
class ApplyReport:
    """Outcome shown after applying a preview."""

    imported: int
    total: int
    skipped: dict[str, int] = field(default_factory=dict)
    renamed: int = 0
    replaced: int = 0

    @property
    def summary(self) -> str:
        return f"Импортировано {self.imported} из {self.total} команд"


class Importer(ABC):
    """A parser that never executes or persists anything it reads."""

    @abstractmethod
    def parse(self, path: Path) -> ImportResult:
        """Read one foreign file into checked Ayris command models."""

    def apply(
        self,
        result: ImportResult,
        target_folder: str | Sequence[str],
        *,
        database: Database,
        profile_id: int,
        sounds_dir: Path,
        selected: Iterable[int] | None = None,
        conflicts: ConflictStrategy = ConflictStrategy.RENAME,
    ) -> ApplyReport:
        """Validate and apply selected commands in one database transaction."""
        return _apply(
            result,
            _folder_parts(target_folder),
            database=database,
            profile_id=profile_id,
            sounds_dir=sounds_dir,
            selected=selected,
            conflicts=conflicts,
        )


def _folder_parts(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace("\\", "/").split("/") if part.strip())
    return tuple(part.strip() for part in value if part.strip())


def _apply(
    result: ImportResult,
    target_folder: tuple[str, ...],
    *,
    database: Database,
    profile_id: int,
    sounds_dir: Path,
    selected: Iterable[int] | None,
    conflicts: ConflictStrategy,
) -> ApplyReport:
    selected_indices = set(selected) if selected is not None else set(range(len(result.commands)))
    chosen = [command for index, command in enumerate(result.commands) if index in selected_indices]
    library = {command.name: command for command in chosen}
    for command in chosen:
        ensure_valid(command, library=library)

    command_repo = CommandRepository(database)
    trigger_repo = TriggerRepository(database)
    variable_repo = VariableRepository(database)
    folder_repo = FolderRepository(database)
    skipped: Counter[str] = Counter(result.skipped)
    imported = renamed = replaced_count = 0

    with database.transaction():
        for command in chosen:
            folder_id = _ensure_folder(folder_repo, profile_id, (*target_folder, *command.folder))
            name = command.name
            existing = command_repo.get_by_name(profile_id, name)
            if existing is not None and conflicts is ConflictStrategy.SKIP:
                skipped["конфликт имени"] += 1
                continue
            if existing is not None and conflicts is ConflictStrategy.RENAME:
                name = _free_name(command_repo, profile_id, name)
                renamed += 1
            elif existing is not None:
                assert existing.id is not None
                command_repo.delete(existing.id)
                replaced_count += 1

            ready = command.model_copy(update={"name": name, "folder_id": folder_id})
            stored = command_repo.create(command_to_row(ready, profile_id=profile_id))
            assert stored.id is not None
            for trigger in triggers_to_rows(ready, command_id=stored.id):
                trigger_repo.add(trigger)
            for variable in initial_variables(ready, profile_id=profile_id):
                variable_repo.set(
                    variable.name,
                    variable.value,
                    scope=variable.scope,
                    profile_id=variable.profile_id,
                    var_type=variable.type,
                    persistent=variable.persistent,
                )
            imported += 1

        _copy_sounds(result.sounds, sounds_dir, result.warnings)

    report = ApplyReport(
        imported=imported,
        total=result.total_commands,
        skipped=dict(skipped),
        renamed=renamed,
        replaced=replaced_count,
    )
    _log.info(report.summary)
    return report


def _ensure_folder(repo: FolderRepository, profile_id: int, path: Sequence[str]) -> int | None:
    parent: int | None = None
    folders = repo.list_for_profile(profile_id)
    for name in path:
        found = next(
            (
                folder
                for folder in folders
                if folder.profile_id == profile_id
                and folder.parent_id == parent
                and folder.name.casefold() == name.casefold()
            ),
            None,
        )
        if found is None:
            found = repo.create(CommandFolder(name=name, profile_id=profile_id, parent_id=parent))
            folders.append(found)
        parent = found.id
    return parent


def _free_name(repo: CommandRepository, profile_id: int, original: str) -> str:
    suffix = 2
    while True:
        candidate = f"{original} (импорт {suffix})"
        if repo.get_by_name(profile_id, candidate) is None:
            return candidate
        suffix += 1


def _copy_sounds(
    sounds: Iterable[ImportedSound],
    sounds_dir: Path,
    warnings: list[ImportNotice],
) -> None:
    sounds_dir.mkdir(parents=True, exist_ok=True)
    for sound in sounds:
        if not sound.source.is_file():
            warnings.append(
                ImportNotice(
                    "файл звука не найден",
                    command=sound.command,
                    source=str(sound.source),
                )
            )
            continue
        target = sounds_dir / sound.target_name
        if sound.source.resolve() != target.resolve():
            shutil.copy2(sound.source, target)
