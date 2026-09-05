"""Commands as files and as database rows.

Two directions out of :mod:`ayris.actions.macros.schema`, kept in one module
because they are the same question asked twice — what of a command is portable, and
what only means something here.

**A file is a document, not a command.** Even the export of a single command is
wrapped in :class:`AyrisDocument`: one envelope with ``schema_version``, the folders
the commands live in, and the commands themselves. Importing a folder and importing
one command are then the same code path with a different ``kind``, and the version
that :mod:`ayris.actions.macros.format_migrations` needs has exactly one place to
sit.

**Two fields never leave the machine.** ``id`` and ``folder_id`` are row numbers;
on the machine that imports the file they belong to something else entirely, so the
dump drops them and the import looks folders up by name. Secrets do not leave
either: given the action registry, :func:`mask_secrets` replaces every parameter an
action marked secret with :data:`ayris.actions.base.SECRET_MASK` — a macro that
connects to Wi-Fi holds a password, and an exported file is something users mail to
each other.

**The database has three tables and the model has more fields than that.** Triggers
map onto rows one for one, the block tree is ``actions_json``, and the columns of
``commands`` cover the rest — except the variable declarations and the command's
stage sounds, which task 3 did not give a column. Rather than lose them on save,
they travel as a declaration header at the front of ``actions_json``
(:data:`DECLARATIONS_TYPE`), which is not a block and is named so it cannot be
mistaken for one. When the schema grows the two columns, that header becomes a
schema migration and this note goes away.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field, field_validator

from ayris.actions.base import SECRET_MASK, secret_fields
from ayris.actions.macros.format_migrations import (
    CURRENT_FORMAT_VERSION,
    MacroFormatError,
    migrate_document,
    trigger_from_row,
)
from ayris.actions.macros.schema import (
    ActionBlock,
    CommandModel,
    EventTrigger,
    HotkeyTrigger,
    MacroModel,
    TimerTrigger,
    VoiceTrigger,
)
from ayris.core.models import (
    Command,
    Trigger,
    TriggerType,
    Variable,
    VariableScope,
    to_db_timestamp,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from ayris.actions.macros.schema import TriggerModel
    from ayris.actions.registry import ActionRegistry
    from ayris.core.models import JsonObject

__all__ = [
    "AYRIS_SUFFIX",
    "DECLARATIONS_TYPE",
    "AyrisDocument",
    "FolderEntry",
    "command_from_rows",
    "command_to_row",
    "command_to_rows",
    "dump_command",
    "dump_commands",
    "dump_document",
    "initial_variables",
    "load_command",
    "load_document",
    "mask_secrets",
    "read_document",
    "triggers_to_rows",
    "write_document",
]

#: Extension of an exported file. The same one profile bundles use for
#: ``commands.ayris``, because it is the same format at a different version.
AYRIS_SUFFIX: Final = ".ayris"

#: ``type`` of the declaration header inside ``actions_json``. Starts with a
#: character :class:`~ayris.actions.macros.schema.ActionBlock` rejects in a block
#: name, so no action and no logic block can ever collide with it.
DECLARATIONS_TYPE: Final = "#declarations"

_INDENT: Final = 2


class FolderEntry(MacroModel):
    """One folder of the tree a document carries, named rather than numbered."""

    path: list[str] = Field(min_length=1)
    sort_order: int = 0

    @field_validator("path")
    @classmethod
    def _check_path(cls, value: list[str]) -> list[str]:
        cleaned = [part.strip() for part in value]
        if not all(cleaned):
            raise ValueError("в пути папки есть пустое имя")
        return cleaned


class AyrisDocument(MacroModel):
    """What a ``.ayris`` file holds: a version, some folders, some commands.

    ``exported_at`` is optional on purpose. A file written by the application
    stamps it; the fixtures of this task do not, so that dumping them again
    reproduces the file byte for byte and a round-trip test can compare text.
    """

    schema_version: int = CURRENT_FORMAT_VERSION
    kind: Literal["command", "collection"] = "collection"
    exported_at: datetime | None = None
    folders: list[FolderEntry] = Field(default_factory=list)
    commands: list[CommandModel] = Field(default_factory=list)


def dump_document(document: AyrisDocument, *, registry: ActionRegistry | None = None) -> str:
    """Serialize a document to the JSON text of a ``.ayris`` file.

    Pass ``registry`` and secret parameters are masked on the way out; without it
    nothing here can know which parameter of which action is a password, and the
    caller carries that decision.
    """
    prepared = document
    if registry is not None:
        prepared = document.model_copy(
            update={"commands": [mask_secrets(command, registry) for command in document.commands]}
        )
    payload = prepared.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
        exclude={"commands": {"__all__": {"id", "folder_id"}}},
    )
    return json.dumps(payload, ensure_ascii=False, indent=_INDENT) + "\n"


def dump_command(
    command: CommandModel,
    *,
    exported_at: datetime | None = None,
    registry: ActionRegistry | None = None,
) -> str:
    """One command as a ``.ayris`` file, folders included when it sits in one."""
    folders = [FolderEntry(path=list(command.folder))] if command.folder else []
    document = AyrisDocument(
        kind="command",
        exported_at=exported_at,
        folders=folders,
        commands=[command],
    )
    return dump_document(document, registry=registry)


def dump_commands(
    commands: Iterable[CommandModel],
    *,
    folders: Iterable[FolderEntry] | None = None,
    exported_at: datetime | None = None,
    registry: ActionRegistry | None = None,
) -> str:
    """A whole folder as one ``.ayris`` file.

    Folder entries are collected from the commands themselves when not given, so
    exporting a subtree does not require walking the folder table twice.
    """
    listed = list(commands)
    entries = list(folders) if folders is not None else _folders_of(listed)
    document = AyrisDocument(
        kind="collection",
        exported_at=exported_at,
        folders=entries,
        commands=listed,
    )
    return dump_document(document, registry=registry)


def _folders_of(commands: Sequence[CommandModel]) -> list[FolderEntry]:
    """Every folder path mentioned by a command, parents included, in order."""
    seen: dict[tuple[str, ...], None] = {}
    for command in commands:
        for depth in range(1, len(command.folder) + 1):
            seen.setdefault(tuple(command.folder[:depth]), None)
    return [FolderEntry(path=list(path)) for path in seen]


def load_document(source: str | bytes | JsonObject) -> AyrisDocument:
    """Read a ``.ayris`` document, migrating an older format on the way in.

    Raises:
        MacroFormatError: the text is not JSON, not an object, or carries a version
            this build cannot read.
        pydantic.ValidationError: the document is the right format but the wrong
            shape — a broken parameter, an unknown trigger type, a bad combination.
    """
    raw = _as_object(source)
    return AyrisDocument.model_validate(migrate_document(raw))


def load_command(source: str | bytes | JsonObject) -> CommandModel:
    """Read a file holding exactly one command.

    Raises:
        MacroFormatError: the file holds no commands or more than one — an import
            that silently took the first would lose the rest.
    """
    document = load_document(source)
    if len(document.commands) != 1:
        raise MacroFormatError(
            f"expected one command, found {len(document.commands)}",
            user_message=(
                f"В файле {len(document.commands)} команд, а нужна одна. "
                "Импортируйте его как папку."
            ),
        )
    return document.commands[0]


def _as_object(source: str | bytes | JsonObject) -> JsonObject:
    if isinstance(source, dict):
        return source
    try:
        loaded = json.loads(source)
    except (TypeError, ValueError) as exc:
        raise MacroFormatError(
            f"not valid JSON: {exc}",
            user_message="Файл повреждён: это не JSON.",
        ) from exc
    if not isinstance(loaded, dict):
        raise MacroFormatError(
            f"document is {type(loaded).__name__}, not an object",
            user_message="Файл команд должен быть JSON-объектом.",
        )
    return loaded


def read_document(path: Path) -> AyrisDocument:
    """Load a ``.ayris`` file from disk.

    Raises:
        MacroFormatError: the file cannot be read or is not a readable document.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MacroFormatError(
            f"cannot read {path}: {exc}",
            user_message=f"Не могу прочитать файл «{path.name}».",
        ) from exc
    except UnicodeDecodeError as exc:
        raise MacroFormatError(
            f"{path} is not UTF-8: {exc}",
            user_message=f"Файл «{path.name}» не в кодировке UTF-8.",
        ) from exc
    return load_document(text)


def write_document(
    path: Path,
    document: AyrisDocument,
    *,
    registry: ActionRegistry | None = None,
) -> None:
    """Write a document to disk as UTF-8 with ``\\n`` line endings.

    The newline is pinned rather than left to the platform: the same file is read on
    Windows and in the Linux CI job, and a file whose bytes depend on where it was
    written cannot be compared in a test or diffed in a review.

    Raises:
        MacroFormatError: the file cannot be written.
    """
    text = dump_document(document, registry=registry)
    try:
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise MacroFormatError(
            f"cannot write {path}: {exc}",
            user_message=f"Не могу записать файл «{path.name}».",
        ) from exc


def mask_secrets(command: CommandModel, registry: ActionRegistry) -> CommandModel:
    """A copy of the command with every secret parameter replaced by the mask.

    Which parameters those are is the action's own declaration
    (``json_schema_extra={"secret": True}``), read through
    :func:`ayris.actions.base.secret_fields`, so a new action with a token field is
    covered the day it is written and nothing here has to keep a list.
    """
    copy = command.model_copy(deep=True)
    for location in copy.blocks():
        block = location.block
        if not registry.has(block.type):
            continue
        fields = secret_fields(registry.get(block.type).params_model())
        for name in fields & block.params.keys():
            block.params[name] = SECRET_MASK
    return copy


def _dump_blocks(blocks: Sequence[ActionBlock]) -> list[JsonObject]:
    """The block tree as the JSON array the ``actions_json`` column holds."""
    return [block.model_dump(mode="json", by_alias=True, exclude_none=True) for block in blocks]


def _declarations(command: CommandModel) -> JsonObject | None:
    """The header that carries what ``commands`` has no column for, or ``None``.

    ``None`` when there is nothing to carry, and that is the common case: a command
    without its own variables and without stage sounds writes the plain array of
    blocks that task 3 expects and that every other reader of the column already
    understands.
    """
    if not command.variables and not command.sounds:
        return None
    header: JsonObject = {"type": DECLARATIONS_TYPE}
    if command.variables:
        header["variables"] = [
            declared.model_dump(mode="json", exclude_none=True) for declared in command.variables
        ]
    if command.sounds:
        header["sounds"] = [
            sound.model_dump(mode="json", exclude_none=True) for sound in command.sounds
        ]
    return header


def _split_declarations(actions: Iterable[JsonObject]) -> tuple[JsonObject, list[JsonObject]]:
    """The declaration header and the blocks, out of one ``actions_json`` array."""
    header: JsonObject = {}
    blocks: list[JsonObject] = []
    for entry in actions:
        if entry.get("type") == DECLARATIONS_TYPE:
            header = header or entry
            continue
        blocks.append(entry)
    return header, blocks


def _listed(header: JsonObject, key: str) -> list[Any]:
    """One list out of the header, tolerating a column edited by hand."""
    value = header.get(key)
    return value if isinstance(value, list) else []


def command_to_row(command: CommandModel, *, profile_id: int) -> Command:
    """The model as the ``commands`` row, block tree included.

    ``folder`` does not travel: the row points at a folder by id, and turning a named
    path into an id is the folder repository's job, not this module's. The id already
    on the model is passed through, so this is both an insert and an update.
    """
    header = _declarations(command)
    blocks = _dump_blocks(command.actions)
    return Command(
        name=command.name,
        profile_id=profile_id,
        id=command.id,
        folder_id=command.folder_id,
        description=command.description,
        tags=tuple(command.tags),
        enabled=command.enabled,
        priority=command.priority,
        cooldown_ms=command.cooldown_ms,
        require_admin=command.require_admin,
        actions=tuple(blocks) if header is None else (header, *blocks),
        created_at=command.created_at,
        updated_at=command.updated_at,
    )


def triggers_to_rows(command: CommandModel, *, command_id: int) -> tuple[Trigger, ...]:
    """The command's triggers as ``triggers`` rows pointing at ``command_id``.

    Payload keys are the ones :func:`ayris.nlu.matcher.trigger_from_db` reads, so a
    voice trigger saved here is a trigger the matcher can fire; ``fuzzy`` and
    ``priority`` sit in their own columns because the matcher's index needs them
    there.
    """
    return tuple(
        Trigger(
            command_id=command_id,
            type=TriggerType(trigger.type),
            payload=_trigger_payload(trigger),
            fuzzy=trigger.fuzzy if isinstance(trigger, VoiceTrigger) else True,
            priority=trigger.priority if isinstance(trigger, VoiceTrigger) else 0,
        )
        for trigger in command.triggers
    )


def command_to_rows(
    command: CommandModel,
    *,
    profile_id: int,
    command_id: int | None = None,
) -> tuple[Command, tuple[Trigger, ...]]:
    """Both halves of a save: the ``commands`` row and the ``triggers`` rows.

    A command that has never been saved has no id, and a trigger row has nowhere to
    point: the pair then comes back with no triggers, and the caller writes the
    command, takes the id the insert returned and calls :func:`triggers_to_rows` with
    it. Passing a zero down instead would look like a real row number.
    """
    row = command_to_row(command, profile_id=profile_id)
    identifier = command_id if command_id is not None else command.id
    triggers = () if identifier is None else triggers_to_rows(command, command_id=identifier)
    return row, triggers


def _trigger_payload(trigger: TriggerModel) -> JsonObject:
    """One trigger's ``payload_json``, in the spelling the matcher reads."""
    if isinstance(trigger, VoiceTrigger):
        payload: JsonObject = {
            trigger.payload_key: trigger.phrase,
            "enabled": trigger.enabled,
            **trigger.conditions,
        }
        if trigger.fuzzy_threshold is not None:
            payload["threshold"] = trigger.fuzzy_threshold
        return payload
    if isinstance(trigger, HotkeyTrigger):
        return {"combo": trigger.combo, "enabled": trigger.enabled}
    if isinstance(trigger, EventTrigger):
        return {
            "event_name": trigger.event_name,
            "filter_json": trigger.filter_json,
            "enabled": trigger.enabled,
        }
    return _timer_payload(trigger)


def _timer_payload(trigger: TimerTrigger) -> JsonObject:
    payload: JsonObject = {"enabled": trigger.enabled}
    if trigger.cron is not None:
        payload["cron"] = trigger.cron
    else:
        payload["fire_at"] = to_db_timestamp(trigger.fire_at)
    return payload


def command_from_rows(
    row: Command,
    triggers: Iterable[Trigger] = (),
    *,
    folder: Sequence[str] = (),
) -> CommandModel:
    """The rows of one command back into the model, validated on the way.

    ``folder`` is the named path of ``row.folder_id``, which only the folder table can
    resolve; without it the command reads back at the root of the tree while the
    numeric id still travels in ``folder_id``, so a save puts it back where it was.

    Raises:
        pydantic.ValidationError: the rows do not make a valid command — a hand-edited
            ``actions_json``, a payload from a build that stored something else.
        MacroFormatError: a trigger row carries a type this build does not know.
    """
    header, blocks = _split_declarations(row.actions)
    payload: JsonObject = {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "tags": list(row.tags),
        "folder_id": row.folder_id,
        "folder": list(folder),
        "enabled": row.enabled,
        "priority": row.priority,
        "cooldown_ms": row.cooldown_ms,
        "require_admin": row.require_admin,
        "triggers": [
            trigger_from_row(
                str(trigger.type),
                dict(trigger.payload),
                fuzzy=trigger.fuzzy,
                priority=trigger.priority,
            )
            for trigger in triggers
        ],
        "actions": blocks,
        "variables": _listed(header, "variables"),
        "sounds": _listed(header, "sounds"),
    }
    if row.created_at is not None:
        payload["created_at"] = row.created_at
    if row.updated_at is not None:
        payload["updated_at"] = row.updated_at
    return CommandModel.model_validate(payload)


def initial_variables(
    command: CommandModel,
    *,
    profile_id: int | None = None,
) -> tuple[Variable, ...]:
    """The command's declarations as ``variables`` rows — for a first save only.

    Local variables are left out: they live for one invocation of the command, and the
    table is for what outlives it. What comes back are the profile and global
    declarations with their defaults, so a command that reads ``{music_volume}`` finds
    the row on its first run instead of an unknown name.

    Called when the command is created and not on every save. A declaration's
    ``default`` is where a variable *starts*; writing it again later would step on the
    value the user's macro has since put there, and that is exactly the persistence
    the declaration asked for.
    """
    return tuple(
        Variable(
            name=declared.name,
            scope=declared.scope,
            profile_id=profile_id if declared.scope is VariableScope.PROFILE else None,
            type=declared.type,
            value=declared.default,
            persistent=declared.persistent,
        )
        for declared in command.variables
        if declared.scope is not VariableScope.LOCAL
    )
