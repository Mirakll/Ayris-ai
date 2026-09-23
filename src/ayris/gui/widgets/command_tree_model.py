"""The command library as a tree: a repository-backed data store and a Qt model.

Two layers live here, split so the interesting half is testable without a window.

:class:`CommandTreeStore` is pure Python over the command, folder and trigger
repositories of task 03. It reads the tree, counts commands per folder, finds the
trigger conflicts of tasks 37/38, and performs every mutation the library offers —
create, rename, move, reorder, duplicate, delete, enable, tag — plus import and
export through the ``.ayris`` serializer of task 30. It never touches Qt, so a test
builds one over an in-memory database and asserts on the result in the database.

:class:`CommandTreeModel` is a :class:`QAbstractItemModel` over one store. It builds
an internal node tree from a store snapshot, applies the current
:class:`TreeFilter` by pruning to matched commands with their folder path kept, and
exposes semantic roles (kind, id, enabled, conflicts) plus a tooltip listing the
commands a trigger clashes with. Structure-changing operations reload the branch;
enabling or disabling one command only refreshes that row, never the whole model.

The ordering rule matches the schema: folders carry ``sort_order`` and are dragged
into any order, commands have only ``priority`` and sort by it then by name. So a
command dragged onto a folder changes its ``folder_id``; reordering is a folder
affair, and both survive a restart because both are columns, not view state.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import TYPE_CHECKING, Final

from PySide6.QtCore import (
    QAbstractItemModel,
    QByteArray,
    QDataStream,
    QIODevice,
    QMimeData,
    QModelIndex,
    QObject,
    QPersistentModelIndex,
    Qt,
    Signal,
)

from ayris.actions.macros.debug_session import DebugSessionStore
from ayris.actions.macros.serializer import (
    FolderEntry,
    command_from_rows,
    command_from_snapshot,
    command_to_row,
    dump_command,
    dump_commands,
    load_document,
    triggers_to_rows,
)
from ayris.core.errors import AyrisError
from ayris.core.models import (
    Command,
    CommandFolder,
    CommandVersion,
    Trigger,
    TriggerType,
    VariableScope,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.macros.schema import CommandModel
    from ayris.core.repositories import Repositories

__all__ = [
    "CONFLICT_ROLE",
    "COUNT_ROLE",
    "ENABLED_ROLE",
    "ENTITY_ID_ROLE",
    "KIND_ROLE",
    "TREE_MIME_TYPE",
    "CommandTreeModel",
    "CommandTreeStore",
    "ConflictStrategy",
    "ImportOutcome",
    "NodeKind",
    "StatusFilter",
    "TreeFilter",
]

_log = get_logger(__name__)

#: Mime type carrying a drag of tree rows. Local to Ayris, never leaves the app.
TREE_MIME_TYPE: Final = "application/x-ayris-command-tree"

#: The copy suffix a duplicate gets, per the task. A second copy becomes «— копия 2».
_COPY_SUFFIX: Final = "— копия"


class _Role(IntEnum):
    KIND = int(Qt.ItemDataRole.UserRole) + 1
    ENTITY_ID = int(Qt.ItemDataRole.UserRole) + 2
    ENABLED = int(Qt.ItemDataRole.UserRole) + 3
    CONFLICT = int(Qt.ItemDataRole.UserRole) + 4
    COUNT = int(Qt.ItemDataRole.UserRole) + 5


#: ``"folder"`` or ``"command"`` for the node at an index.
KIND_ROLE: Final = int(_Role.KIND)
#: The database id of the folder or command at an index.
ENTITY_ID_ROLE: Final = int(_Role.ENTITY_ID)
#: ``bool`` — whether a command is enabled. Always ``True`` for a folder.
ENABLED_ROLE: Final = int(_Role.ENABLED)
#: ``tuple[str, ...]`` — names of commands this one's triggers clash with.
CONFLICT_ROLE: Final = int(_Role.CONFLICT)
#: ``int`` — commands in a folder's whole subtree. ``0`` for a command node.
COUNT_ROLE: Final = int(_Role.COUNT)


class NodeKind(StrEnum):
    """What a tree node stands for."""

    FOLDER = "folder"
    COMMAND = "command"


class StatusFilter(StrEnum):
    """The enabled/disabled filter over commands."""

    ALL = "all"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class TreeFilter:
    """The search and filter state applied to the tree.

    ``text`` matches a command by name, by any tag or by any voice-trigger phrase;
    ``tag`` narrows to one exact tag; ``status`` to enabled or disabled commands;
    ``only_conflicts`` to commands whose triggers clash with another's.
    """

    text: str = ""
    status: StatusFilter = StatusFilter.ALL
    tag: str = ""
    only_conflicts: bool = False

    @property
    def active(self) -> bool:
        return bool(
            self.text.strip()
            or self.tag
            or self.status is not StatusFilter.ALL
            or self.only_conflicts
        )


@dataclass(frozen=True, slots=True)
class ImportOutcome:
    """What applying an imported ``.ayris`` document changed."""

    imported: int = 0
    skipped: int = 0
    renamed: int = 0
    replaced: int = 0
    folders: int = 0

    @property
    def summary(self) -> str:
        parts = [f"добавлено команд: {self.imported}"]
        if self.renamed:
            parts.append(f"переименовано: {self.renamed}")
        if self.replaced:
            parts.append(f"заменено: {self.replaced}")
        if self.skipped:
            parts.append(f"пропущено: {self.skipped}")
        if self.folders:
            parts.append(f"новых папок: {self.folders}")
        return ", ".join(parts)


class ConflictStrategy(StrEnum):
    """How import resolves a command whose name is already taken."""

    RENAME = "rename"
    REPLACE = "replace"
    SKIP = "skip"


def _now() -> datetime:
    return datetime.now(UTC)


def _trigger_key(trigger: Trigger) -> tuple[str, str] | None:
    """The comparable identity of a trigger, or ``None`` if it cannot clash.

    Only voice phrases and hotkey combinations can be shared by two commands; an
    event name or a schedule is not a conflict the library warns about.
    """
    if trigger.type is TriggerType.VOICE:
        phrase = trigger.payload.get("phrase") or trigger.payload.get("pattern") or ""
        text = str(phrase).casefold().strip()
        return ("voice", text) if text else None
    if trigger.type is TriggerType.HOTKEY:
        combo = str(trigger.payload.get("combo", "")).casefold().strip()
        return ("hotkey", combo) if combo else None
    return None


class CommandTreeStore:
    """Pure data access and mutation over the command library of one profile."""

    def __init__(
        self, repositories: Repositories, profile_id: int, *, version_limit: int = 20
    ) -> None:
        self._repos = repositories
        self._profile_id = profile_id
        self._version_limit = version_limit
        self._debug_store: DebugSessionStore | None = None

    @property
    def profile_id(self) -> int:
        return self._profile_id

    @property
    def debug_store(self) -> DebugSessionStore:
        """The per-command debugger session store (breakpoints, watches, slots).

        Built over the same database as the library, so breakpoints set in the node
        editor land in ``macro_debug_sessions`` — the very table
        :class:`~ayris.actions.macros.debugger.MacroDebugger` restores from when a
        debug run starts. Cached: the store is stateless glue over the database.
        """
        if self._debug_store is None:
            self._debug_store = DebugSessionStore(self._repos.database)
        return self._debug_store

    @property
    def version_limit(self) -> int:
        """How many versions per command survive a prune, pins aside (task 54)."""
        return self._version_limit

    # -- reads --------------------------------------------------------------

    def folders(self) -> list[CommandFolder]:
        return self._repos.folders.list_for_profile(self._profile_id)

    def commands(self) -> list[Command]:
        return self._repos.commands.list_for_profile(self._profile_id)

    def triggers(self) -> list[Trigger]:
        return self._repos.triggers.list_for_profile(self._profile_id, enabled_only=False)

    def phrases_by_command(self) -> dict[int, tuple[str, ...]]:
        """Every voice phrase of each command, lower-cased, for the text search."""
        result: dict[int, list[str]] = defaultdict(list)
        for trigger in self.triggers():
            key = _trigger_key(trigger)
            if key is not None and key[0] == "voice":
                result[trigger.command_id].append(key[1])
        return {cid: tuple(phrases) for cid, phrases in result.items()}

    def conflicts(self) -> dict[int, tuple[str, ...]]:
        """For each command in a conflict, the names of the commands it clashes with."""
        names = {c.id: c.name for c in self.commands() if c.id is not None}
        by_key: dict[tuple[str, str], set[int]] = defaultdict(set)
        for trigger in self.triggers():
            key = _trigger_key(trigger)
            if key is not None:
                by_key[key].add(trigger.command_id)
        clashing: dict[int, set[int]] = defaultdict(set)
        for members in by_key.values():
            if len(members) < 2:
                continue
            for command_id in members:
                clashing[command_id].update(members - {command_id})
        return {
            command_id: tuple(sorted(names[other] for other in others if other in names))
            for command_id, others in clashing.items()
        }

    def folder_path(self, folder_id: int | None) -> list[str]:
        """The named path of a folder, root first. Empty for the tree root."""
        if folder_id is None:
            return []
        by_id = {f.id: f for f in self.folders()}
        path: list[str] = []
        current = by_id.get(folder_id)
        guard = 0
        while current is not None and guard < 1000:
            path.append(current.name)
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
            guard += 1
        path.reverse()
        return path

    # -- structural mutations ----------------------------------------------

    def create_folder(self, parent_id: int | None, name: str) -> CommandFolder:
        siblings = self._repos.folders.children(parent_id)
        order = max((f.sort_order for f in siblings), default=-1) + 1
        return self._repos.folders.create(
            CommandFolder(
                name=name, profile_id=self._profile_id, parent_id=parent_id, sort_order=order
            )
        )

    def create_command(self, folder_id: int | None, name: str) -> Command:
        return self._repos.commands.create(
            Command(
                name=self._unique_command_name(name),
                profile_id=self._profile_id,
                folder_id=folder_id,
                created_at=_now(),
                updated_at=_now(),
            )
        )

    def rename_folder(self, folder_id: int, name: str) -> None:
        self._repos.folders.rename(folder_id, name)

    def rename_command(self, command_id: int, name: str) -> None:
        command = self._repos.commands.get(command_id)
        if command is None:
            return
        from dataclasses import replace

        self._repos.commands.update(replace(command, name=name))

    def set_enabled(self, command_id: int, *, enabled: bool) -> None:
        self._repos.commands.set_enabled(command_id, enabled=enabled)

    def move_command(self, command_id: int, folder_id: int | None) -> None:
        self._repos.commands.move_to_folder(command_id, folder_id)

    def is_descendant(self, folder_id: int, ancestor_id: int) -> bool:
        """Whether ``folder_id`` sits inside ``ancestor_id`` (or is it)."""
        if folder_id == ancestor_id:
            return True
        by_id = {f.id: f for f in self.folders()}
        current = by_id.get(folder_id)
        guard = 0
        while current is not None and guard < 1000:
            if current.parent_id == ancestor_id:
                return True
            current = by_id.get(current.parent_id) if current.parent_id is not None else None
            guard += 1
        return False

    def move_folder(self, folder_id: int, parent_id: int | None) -> None:
        """Reparent a folder, refusing to drop it inside itself or its own subtree."""
        if parent_id is not None and self.is_descendant(parent_id, folder_id):
            raise AyrisError(
                f"folder {folder_id} cannot move under its own descendant {parent_id}",
                user_message="Папку нельзя перенести внутрь самой себя.",
            )
        folder = self._repos.folders.get(folder_id)
        if folder is None:
            return
        from dataclasses import replace

        siblings = self._repos.folders.children(parent_id)
        order = max((f.sort_order for f in siblings), default=-1) + 1
        self._repos.folders.update(replace(folder, parent_id=parent_id, sort_order=order))

    def reorder_folders(self, ordered_ids: Sequence[int]) -> None:
        self._repos.folders.reorder(ordered_ids)

    def delete_folder(self, folder_id: int) -> None:
        """Delete a folder and its subfolders; commands inside move to the root.

        This is the ``ON DELETE SET NULL`` behaviour of ``commands.folder_id`` — the
        single, tested rule: a deleted folder never orphans or loses a command.
        """
        self._repos.folders.delete(folder_id)

    def commands_in_subtree(self, folder_id: int) -> list[Command]:
        """Every command in a folder and its descendants — for the delete warning."""
        by_parent: dict[int | None, list[CommandFolder]] = defaultdict(list)
        for folder in self.folders():
            by_parent[folder.parent_id].append(folder)
        wanted: set[int] = set()
        stack = [folder_id]
        while stack:
            current = stack.pop()
            wanted.add(current)
            for child in by_parent.get(current, ()):
                if child.id is not None:
                    stack.append(child.id)
        return [c for c in self.commands() if c.folder_id in wanted]

    def delete_command(self, command_id: int) -> None:
        self._repos.commands.delete(command_id)

    def duplicate_command(self, command_id: int) -> Command | None:
        """Copy a command under a «— копия» name, dropping its hotkey triggers.

        A voice phrase may be shared (the copy simply shows as a conflict), but a
        hotkey cannot fire two commands, so the busy combinations are not carried
        over — the task's «без дублирования занятых хоткеев».
        """
        from dataclasses import replace

        source = self._repos.commands.get(command_id)
        if source is None:
            return None
        copy = replace(
            source,
            id=None,
            name=self._unique_command_name(f"{source.name} {_COPY_SUFFIX}"),
            created_at=_now(),
            updated_at=_now(),
        )
        created = self._repos.commands.create(copy)
        if created.id is None:
            return created
        for trigger in self._repos.triggers.list_for_command(command_id):
            if trigger.type is TriggerType.HOTKEY:
                continue
            self._repos.triggers.add(replace(trigger, id=None, command_id=created.id))
        return created

    def assign_tag(self, command_ids: Iterable[int], tag: str) -> int:
        """Add ``tag`` to each command that does not already carry it."""
        from dataclasses import replace

        tag = tag.strip()
        if not tag:
            return 0
        changed = 0
        for command_id in command_ids:
            command = self._repos.commands.get(command_id)
            if command is None or tag in command.tags:
                continue
            self._repos.commands.update(replace(command, tags=(*command.tags, tag)))
            changed += 1
        return changed

    # -- editor load / save (task 52) --------------------------------------

    def command_model(self, command_id: int) -> CommandModel:
        """The full :class:`CommandModel` of one command, for the editor to open.

        This is the same read the export path uses: the command row, its triggers,
        and its folder path, back into the one model the editor and the node view of
        task 53 both work on. The variables the command declares travel inside
        ``actions_json`` (the declaration header of the serializer), so the model
        comes back whole.
        """
        return self._command_model(command_id)

    def sibling_names(self, command_id: int) -> set[str]:
        """Names of the other commands in the same folder, for duplicate checking.

        The editor flags a name that clashes before it lets the command be saved;
        the comparison is case-folded so «Свет» and «свет» count as the same name.
        """
        target = self._repos.commands.get(command_id)
        if target is None:
            return set()
        return {
            command.name.casefold()
            for command in self.commands()
            if command.id != command_id and command.folder_id == target.folder_id
        }

    def trigger_conflicts(self, command_id: int) -> dict[tuple[str, str], tuple[str, ...]]:
        """Every voice phrase / hotkey combo used by *other* commands and by whom.

        Keyed by the same ``(kind, text)`` identity :func:`_trigger_key` builds, so the
        editor can look a trigger up as the user types it and name the command it would
        collide with — the task's «конфликт показывается до сохранения».
        """
        names = {c.id: c.name for c in self.commands() if c.id is not None}
        owners: dict[tuple[str, str], set[str]] = defaultdict(set)
        for trigger in self.triggers():
            if trigger.command_id == command_id:
                continue
            key = _trigger_key(trigger)
            if key is not None and trigger.command_id in names:
                owners[key].add(names[trigger.command_id])
        return {key: tuple(sorted(who)) for key, who in owners.items()}

    def save_command(self, model: CommandModel) -> CommandModel:
        """Persist an edited command: its row, its triggers, its variable declarations.

        The command keeps its id and folder — the editor does not move it — so this is
        an update of the ``commands`` row, a full replacement of the ``triggers`` rows
        (the editor is the authority on the trigger set now), and a merge of the profile
        and global variable declarations into the ``variables`` table without stepping
        on a value a running macro has since stored. Returns the model as it reads back,
        so the editor shows exactly what was written.

        Raises:
            AyrisError: the model has no id — a command must exist in the tree before
                the editor can save it.
        """
        if model.id is None:
            raise AyrisError(
                "cannot save a command without an id",
                user_message="Команду нельзя сохранить: она ещё не создана.",
            )
        from dataclasses import replace

        command_id = model.id
        current = self._repos.commands.get(command_id)
        folder_id = current.folder_id if current is not None else model.folder_id
        placed = model.model_copy(
            update={"folder_id": folder_id, "folder": [], "updated_at": _now()}
        )
        row = replace(command_to_row(placed, profile_id=self._profile_id), folder_id=folder_id)
        # One transaction for the whole save: the row, its version snapshot, its
        # triggers and its declarations commit together or not at all. Task 54: a
        # failure at any step must roll the lot back and leave the previously
        # registered version working, never a half-written command. update() saves
        # the *previous* state as a version first (triggers still un-replaced, so
        # the snapshot's triggers are the old set), then writes the new row.
        with self._repos.database.transaction():
            self._repos.commands.update(row, save_version=True, comment="редактор")
            self._repos.triggers.replace_for_command(
                command_id, triggers_to_rows(placed, command_id=command_id)
            )
            self._sync_declarations(placed)
            self._repos.commands.prune_versions(command_id, keep=self._version_limit)
        return self._command_model(command_id)

    def _sync_declarations(self, model: CommandModel) -> None:
        """Write the profile/global declarations that are new, leaving values alone."""
        for declared in model.variables:
            if declared.scope is VariableScope.LOCAL:
                continue
            profile_id = self._profile_id if declared.scope is VariableScope.PROFILE else None
            existing = self._repos.variables.get(
                declared.name, scope=declared.scope, profile_id=profile_id
            )
            if existing is not None:
                continue
            self._repos.variables.set(
                declared.name,
                declared.default,
                scope=declared.scope,
                profile_id=profile_id,
                var_type=declared.type,
                persistent=declared.persistent,
            )

    # -- versions -----------------------------------------------------------

    def versions(self, command_id: int) -> list[CommandVersion]:
        """The stored versions of a command, newest first (task 54 history)."""
        return self._repos.commands.list_versions(command_id, limit=self._version_limit + 50)

    def version_model(self, command_id: int, version: int) -> CommandModel:
        """One stored version rebuilt into a :class:`CommandModel`, for diff/preview.

        Raises:
            AyrisError: the version does not exist.
        """
        stored = self._repos.commands.get_version(command_id, version)
        if stored is None:
            raise AyrisError(
                f"version {version} of command {command_id} not found",
                user_message=f"Версия {version} не найдена.",
            )
        current = self._repos.commands.get(command_id)
        folder = self.folder_path(current.folder_id) if current is not None else ()
        return command_from_snapshot(stored.snapshot, command_id=command_id, folder=folder)

    def mark_version_important(
        self, command_id: int, version: int, *, important: bool = True
    ) -> None:
        """Pin or unpin a version so the prune keeps it (task 54)."""
        self._repos.commands.mark_version_important(command_id, version, important=important)

    def export_version(self, command_id: int, version: int) -> str:
        """One stored version as a ``.ayris`` document (task 54)."""
        return dump_command(self.version_model(command_id, version))

    # -- export / import ----------------------------------------------------

    def export_command(self, command_id: int) -> str:
        return dump_command(self._command_model(command_id))

    def export_commands(self, command_ids: Iterable[int]) -> str:
        models = [self._command_model(cid) for cid in command_ids]
        return dump_commands(models, folders=self._folder_entries(models))

    def export_folder(self, folder_id: int) -> str:
        models = [
            self._command_model(c.id)
            for c in self.commands_in_subtree(folder_id)
            if c.id is not None
        ]
        return dump_commands(models, folders=self._folder_entries(models))

    def apply_import(
        self,
        text: str,
        *,
        target_folder_id: int | None,
        strategy: ConflictStrategy,
    ) -> ImportOutcome:
        """Write an imported ``.ayris`` document into this profile.

        Folders named in the document are recreated under ``target_folder_id``; each
        command is placed at the end of its named path and its name collision is
        resolved by ``strategy``. Everything runs in one transaction.
        """
        document = load_document(text)
        imported = renamed = replaced = skipped = folders_made = 0
        with self._repos.database.transaction():
            # The chosen folder is the base: a command's own path is created
            # underneath it, so importing into «Игры» a command from «Работа»
            # yields «Игры/Работа», not a second «Игры».
            path_cache: dict[tuple[str, ...], int | None] = {(): target_folder_id}
            for command in document.commands:
                folder_id, made = self._ensure_path(tuple(command.folder), path_cache)
                folders_made += made
                existing = self._repos.commands.get_by_name(self._profile_id, command.name)
                final_name = command.name
                if existing is not None:
                    if strategy is ConflictStrategy.SKIP:
                        skipped += 1
                        continue
                    if strategy is ConflictStrategy.REPLACE:
                        if existing.id is not None:
                            self._repos.commands.delete(existing.id)
                        replaced += 1
                    else:
                        final_name = self._unique_command_name(command.name)
                        renamed += 1
                self._write_command(command, folder_id=folder_id, name=final_name)
                imported += 1
        return ImportOutcome(
            imported=imported,
            skipped=skipped,
            renamed=renamed,
            replaced=replaced,
            folders=folders_made,
        )

    # -- helpers ------------------------------------------------------------

    def _command_model(self, command_id: int) -> CommandModel:
        command = self._repos.commands.get(command_id)
        if command is None:
            raise AyrisError(
                f"command {command_id} not found",
                user_message="Команда не найдена.",
            )
        triggers = self._repos.triggers.list_for_command(command_id)
        return command_from_rows(command, triggers, folder=self.folder_path(command.folder_id))

    @staticmethod
    def _folder_entries(models: Sequence[CommandModel]) -> list[FolderEntry]:
        seen: dict[tuple[str, ...], None] = {}
        for model in models:
            for depth in range(1, len(model.folder) + 1):
                seen.setdefault(tuple(model.folder[:depth]), None)
        return [FolderEntry(path=list(path)) for path in seen]

    def _ensure_path(
        self, path: tuple[str, ...], cache: dict[tuple[str, ...], int | None]
    ) -> tuple[int | None, int]:
        """Resolve or create the folder for a path, returning its id and how many were made."""
        if path in cache:
            return cache[path], 0
        parent_id, made = self._ensure_path(path[:-1], cache)
        name = path[-1]
        for child in self._repos.folders.children(parent_id):
            if child.name == name and child.id is not None:
                cache[path] = child.id
                return child.id, made
        created = self.create_folder(parent_id, name)
        cache[path] = created.id
        return created.id, made + 1

    def _write_command(self, model: CommandModel, *, folder_id: int | None, name: str) -> None:
        from dataclasses import replace

        placed = model.model_copy(update={"name": name, "folder_id": folder_id, "folder": []})
        row = replace(command_to_row(placed, profile_id=self._profile_id), folder_id=folder_id)
        created = self._repos.commands.create(row)
        if created.id is None:
            return
        for trigger in triggers_to_rows(placed, command_id=created.id):
            self._repos.triggers.add(trigger)

    def _unique_command_name(self, base: str) -> str:
        base = base.strip() or "Команда"
        if self._repos.commands.get_by_name(self._profile_id, base) is None:
            return base
        index = 2
        while self._repos.commands.get_by_name(self._profile_id, f"{base} {index}") is not None:
            index += 1
        return f"{base} {index}"


# ----------------------------------------------------------------------
# the Qt model
# ----------------------------------------------------------------------


@dataclass(slots=True)
class _Node:
    kind: NodeKind
    entity_id: int
    name: str
    enabled: bool
    sort_key: tuple[object, ...]
    parent: _Node | None = None
    children: list[_Node] = field(default_factory=list)
    visible: list[_Node] = field(default_factory=list)
    count: int = 0
    conflicts: tuple[str, ...] = ()
    #: ``(haystack, tags)`` for the text and tag search; empty for a folder.
    search_terms: tuple[str, tuple[str, ...]] = ("", ())


class CommandTreeModel(QAbstractItemModel):
    """A tree of folders and commands over a :class:`CommandTreeStore`."""

    #: Emitted with a Russian message when a drop is refused (e.g. into a subtree).
    drop_rejected = Signal(str)
    #: Emitted after a mutation the model made itself (drop, rename, enable).
    changed = Signal()

    def __init__(self, store: CommandTreeStore, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._store = store
        self._filter = TreeFilter()
        self._root = _Node(NodeKind.FOLDER, -1, "", True, ())
        self._index_by_key: dict[tuple[str, int], _Node] = {}
        #: The command open in the editor with unsaved edits, marked in the tree
        #: (task 54); ``None`` when nothing is dirty.
        self._dirty_command: int | None = None
        self.reload()

    @property
    def store(self) -> CommandTreeStore:
        return self._store

    @property
    def tree_filter(self) -> TreeFilter:
        return self._filter

    def set_store(self, store: CommandTreeStore) -> None:
        self._store = store
        self.reload()

    # -- building -----------------------------------------------------------

    def reload(self) -> None:
        self.beginResetModel()
        self._build()
        self.endResetModel()

    def _build(self) -> None:
        folders = self._store.folders()
        commands = self._store.commands()
        conflicts = self._store.conflicts()
        phrases = self._store.phrases_by_command()

        root = _Node(NodeKind.FOLDER, -1, "", True, ())
        nodes: dict[int, _Node] = {}
        for folder in folders:
            if folder.id is None:
                continue
            nodes[folder.id] = _Node(
                NodeKind.FOLDER,
                folder.id,
                folder.name,
                True,
                (0, folder.sort_order, folder.name.casefold()),
            )
        for folder in folders:
            if folder.id is None:
                continue
            node = nodes[folder.id]
            parent = nodes.get(folder.parent_id) if folder.parent_id is not None else root
            node.parent = parent or root
            (parent or root).children.append(node)

        self._index_by_key = {("folder", fid): node for fid, node in nodes.items()}
        for command in commands:
            if command.id is None:
                continue
            node = _Node(
                NodeKind.COMMAND,
                command.id,
                command.name,
                command.enabled,
                (1, -command.priority, command.name.casefold()),
                conflicts=conflicts.get(command.id, ()),
                search_terms=_search_terms(command, phrases.get(command.id, ())),
            )
            parent = nodes.get(command.folder_id) if command.folder_id is not None else root
            node.parent = parent or root
            (parent or root).children.append(node)
            self._index_by_key[("command", command.id)] = node

        for node in (root, *nodes.values()):
            node.children.sort(key=lambda item: item.sort_key)
        self._compute_counts(root)
        self._root = root
        self._apply_filter(root)

    def _compute_counts(self, node: _Node) -> int:
        total = 0
        for child in node.children:
            if child.kind is NodeKind.COMMAND:
                total += 1
            else:
                total += self._compute_counts(child)
        node.count = total
        return total

    def _apply_filter(self, node: _Node) -> bool:
        """Recompute ``visible`` children, returning whether ``node`` has a match."""
        flt = self._filter
        node.visible = []
        matched = False
        for child in node.children:
            if child.kind is NodeKind.COMMAND:
                if self._command_matches(child):
                    node.visible.append(child)
                    matched = True
            else:
                child_has = self._apply_filter(child)
                if child_has or (not flt.active):
                    node.visible.append(child)
                matched = matched or child_has
        node.visible.sort(key=lambda item: item.sort_key)
        return matched

    def _command_matches(self, node: _Node) -> bool:
        flt = self._filter
        if flt.status is StatusFilter.ENABLED and not node.enabled:
            return False
        if flt.status is StatusFilter.DISABLED and node.enabled:
            return False
        if flt.only_conflicts and not node.conflicts:
            return False
        name_term, tags = node.search_terms
        if flt.tag and flt.tag.casefold() not in {t.casefold() for t in tags}:
            return False
        text = flt.text.strip().casefold()
        return not (text and text not in name_term)

    # -- filter API ---------------------------------------------------------

    def set_filter(self, tree_filter: TreeFilter) -> None:
        self._filter = tree_filter
        self.beginResetModel()
        self._apply_filter(self._root)
        self.endResetModel()

    def all_tags(self) -> list[str]:
        tags: set[str] = set()
        for command in self._store.commands():
            tags.update(command.tags)
        return sorted(tags, key=str.casefold)

    # -- Qt model interface -------------------------------------------------

    def index(
        self, row: int, column: int, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = self._node(parent)
        if 0 <= row < len(parent_node.visible):
            return self.createIndex(row, column, parent_node.visible[row])
        return QModelIndex()

    def parent(self, index: QModelIndex = QModelIndex()) -> QModelIndex:  # type: ignore[override]
        if not index.isValid():
            return QModelIndex()
        node = self._node(index)
        parent = node.parent
        if parent is None or parent is self._root:
            return QModelIndex()
        grand = parent.parent or self._root
        row = grand.visible.index(parent) if parent in grand.visible else 0
        return self.createIndex(row, 0, parent)

    def rowCount(  # noqa: N802
        self, parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return len(self._node(parent).visible)

    def columnCount(  # noqa: N802
        self, _parent: QModelIndex | QPersistentModelIndex = QModelIndex()
    ) -> int:
        return 1

    def flags(self, index: QModelIndex | QPersistentModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.ItemIsDropEnabled
        node = self._node(index)
        base = (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEditable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        if node.kind is NodeKind.FOLDER:
            base |= Qt.ItemFlag.ItemIsDropEnabled
        return base

    def data(
        self,
        index: QModelIndex | QPersistentModelIndex,
        role: int = int(Qt.ItemDataRole.DisplayRole),
    ) -> object:
        if not index.isValid():
            return None
        node = self._node(index)
        if role == int(Qt.ItemDataRole.DisplayRole):
            if node.kind is NodeKind.FOLDER:
                return f"{node.name} ({node.count})"
            if node.kind is NodeKind.COMMAND and node.entity_id == self._dirty_command:
                # A bullet marks the command being edited with unsaved changes.
                return f"● {node.name}"
            return node.name
        if role == int(Qt.ItemDataRole.EditRole):
            return node.name
        if role == KIND_ROLE:
            return str(node.kind)
        if role == ENTITY_ID_ROLE:
            return node.entity_id
        if role == ENABLED_ROLE:
            return node.enabled
        if role == CONFLICT_ROLE:
            return node.conflicts
        if role == COUNT_ROLE:
            return node.count
        if role == int(Qt.ItemDataRole.ToolTipRole):
            if node.conflicts:
                joined = ", ".join(node.conflicts)
                return f"Триггер конфликтует с командами: {joined}"
            return None
        return None

    def setData(  # noqa: N802
        self,
        index: QModelIndex | QPersistentModelIndex,
        value: object,
        role: int = int(Qt.ItemDataRole.EditRole),
    ) -> bool:
        if not index.isValid() or role != int(Qt.ItemDataRole.EditRole):
            return False
        name = str(value).strip()
        node = self._node(index)
        if not name or name == node.name:
            return False
        try:
            if node.kind is NodeKind.FOLDER:
                self._store.rename_folder(node.entity_id, name)
            else:
                self._store.rename_command(node.entity_id, name)
        except AyrisError:
            _log.exception("не удалось переименовать узел дерева")
            return False
        self.reload()
        self.changed.emit()
        return True

    # -- drag and drop ------------------------------------------------------

    def supportedDropActions(self) -> Qt.DropAction:  # noqa: N802
        return Qt.DropAction.MoveAction

    def mimeTypes(self) -> list[str]:  # noqa: N802
        return [TREE_MIME_TYPE]

    def mimeData(self, indexes: Iterable[QModelIndex]) -> QMimeData:  # noqa: N802
        payload = QByteArray()
        stream = QDataStream(payload, QIODevice.OpenModeFlag.WriteOnly)
        rows = [idx for idx in indexes if idx.isValid() and idx.column() == 0]
        stream.writeInt32(len(rows))
        for idx in rows:
            node = self._node(idx)
            stream.writeQString(str(node.kind))
            stream.writeInt64(node.entity_id)
        mime = QMimeData()
        mime.setData(TREE_MIME_TYPE, payload)
        return mime

    def dropMimeData(  # noqa: N802
        self,
        data: QMimeData,
        action: Qt.DropAction,
        row: int,
        column: int,  # noqa: ARG002
        parent: QModelIndex | QPersistentModelIndex,
    ) -> bool:
        if action == Qt.DropAction.IgnoreAction:
            return True
        if not data.hasFormat(TREE_MIME_TYPE):
            return False
        dragged = _decode_mime(data)
        target = self._node(parent)
        target_id = None if target is self._root else target.entity_id
        try:
            reordered = self._perform_drop(dragged, target_id, row)
        except AyrisError as exc:
            _log.warning("перемещение отменено: %s", exc.technical)
            self.drop_rejected.emit(exc.user_message)
            return False
        if reordered:
            self.reload()
            self.changed.emit()
        return reordered

    def _perform_drop(
        self, dragged: list[tuple[str, int]], target_id: int | None, row: int
    ) -> bool:
        moved = False
        folder_moves = [eid for kind, eid in dragged if kind == "folder"]
        command_moves = [eid for kind, eid in dragged if kind == "command"]
        for folder_id in folder_moves:
            if target_id is not None and self._store.is_descendant(target_id, folder_id):
                raise AyrisError(
                    f"folder {folder_id} into its own subtree",
                    user_message="Папку нельзя перетащить внутрь самой себя.",
                )
        for folder_id in folder_moves:
            current = self._folder_parent(folder_id)
            if current != target_id:
                self._store.move_folder(folder_id, target_id)
                moved = True
        if folder_moves and row >= 0:
            moved = self._reorder_folder_siblings(target_id, folder_moves, row) or moved
        for command_id in command_moves:
            self._store.move_command(command_id, target_id)
            moved = True
        return moved

    def _folder_parent(self, folder_id: int) -> int | None:
        node = self._index_by_key.get(("folder", folder_id))
        if node is None or node.parent is None or node.parent is self._root:
            return None
        return node.parent.entity_id

    def _reorder_folder_siblings(
        self, parent_id: int | None, moved_ids: list[int], row: int
    ) -> bool:
        parent = self._root if parent_id is None else self._index_by_key.get(("folder", parent_id))
        if parent is None:
            return False
        order = [
            child.entity_id
            for child in parent.visible
            if child.kind is NodeKind.FOLDER and child.entity_id not in moved_ids
        ]
        insert_at = min(max(row, 0), len(order))
        order[insert_at:insert_at] = moved_ids
        self._store.reorder_folders(order)
        return True

    # -- lazy single-command refresh ---------------------------------------

    def refresh_command(self, command_id: int) -> None:
        """Re-read one command's row and refresh its cell — no structural reset."""
        node = self._index_by_key.get(("command", command_id))
        if node is None:
            self.reload()
            return
        command = self._store._repos.commands.get(command_id)
        if command is None:
            self.reload()
            return
        node.enabled = command.enabled
        node.name = command.name
        index = self._index_of(node)
        if index.isValid():
            self.dataChanged.emit(index, index)

    def set_enabled(self, command_id: int, *, enabled: bool) -> None:
        self._store.set_enabled(command_id, enabled=enabled)
        self.refresh_command(command_id)
        self.changed.emit()

    # -- lookup helpers -----------------------------------------------------

    def _node(self, index: QModelIndex | QPersistentModelIndex) -> _Node:
        if not index.isValid():
            return self._root
        pointer = index.internalPointer()
        return pointer if isinstance(pointer, _Node) else self._root

    def _index_of(self, node: _Node) -> QModelIndex:
        parent = node.parent or self._root
        if node not in parent.visible:
            return QModelIndex()
        return self.createIndex(parent.visible.index(node), 0, node)

    def index_for(self, kind: NodeKind, entity_id: int) -> QModelIndex:
        node = self._index_by_key.get((str(kind), entity_id))
        return self._index_of(node) if node is not None else QModelIndex()

    def set_dirty_command(self, command_id: int | None) -> None:
        """Mark one command as having unsaved edits, or clear the mark (task 54).

        Repaints only the two rows that change — the one losing the mark and the one
        gaining it — so the dirty bullet appears and clears without a tree rebuild.
        """
        if command_id == self._dirty_command:
            return
        previous = self._dirty_command
        self._dirty_command = command_id
        for entity_id in (previous, command_id):
            if entity_id is None:
                continue
            index = self.index_for(NodeKind.COMMAND, entity_id)
            if index.isValid():
                self.dataChanged.emit(index, index, [int(Qt.ItemDataRole.DisplayRole)])


def _search_terms(command: Command, phrases: tuple[str, ...]) -> tuple[str, tuple[str, ...]]:
    """The lower-cased haystack for the text search, and the command's tags."""
    parts = [command.name.casefold(), *(t.casefold() for t in command.tags), *phrases]
    return (" \n".join(parts), command.tags)


def _decode_mime(data: QMimeData) -> list[tuple[str, int]]:
    stream = QDataStream(data.data(TREE_MIME_TYPE))
    count = stream.readInt32()
    result: list[tuple[str, int]] = []
    for _ in range(count):
        kind = stream.readQString()
        entity_id = stream.readInt64()
        result.append((kind, entity_id))
    return result
