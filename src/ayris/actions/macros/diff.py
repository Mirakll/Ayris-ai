"""Structural diff of two commands, for the version history of task 54.

The diff is computed on the :class:`~ayris.actions.macros.schema.CommandModel`,
never on the JSON text: a re-indented file or a reordered key is not a change,
and a moved block is one thing that moved, not a wall of line noise. Four things
are compared — the block tree, the triggers, the variables and the stage sounds,
plus the plain header fields — and each difference comes out as one
:class:`Change` with a path a person can read (``actions[1].then[0]``) and the
old and the new value.

Block matching is positional with alignment: :class:`difflib.SequenceMatcher`
lines the two block lists up by type, so a block inserted in the middle is
reported as *added* at its place instead of turning every block after it into a
change. Triggers, variables and sounds match by identity — a phrase, a name, a
stage — because order carries no meaning there.

The module has no Qt and no database import: it is a pure function over two
models, so the version view, the undo stack and the tests all call the same code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

from ayris.actions.macros.schema import (
    ActionBlock,
    EventTrigger,
    HotkeyTrigger,
    SoundBinding,
    VariableModel,
    VoiceTrigger,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ayris.actions.macros.schema import CommandModel, TriggerModel

__all__ = [
    "CHANGE_ADDED",
    "CHANGE_CHANGED",
    "CHANGE_REMOVED",
    "Change",
    "CommandDiff",
    "diff_commands",
]

#: The three shapes a difference takes. Strings, not an enum, because they end up
#: in a table cell and a test assertion far more often than in a ``match``.
CHANGE_ADDED = "added"
CHANGE_REMOVED = "removed"
CHANGE_CHANGED = "changed"

#: Categories a change belongs to, used to group the list beside the version tree.
_CATEGORY_FIELD = "field"
_CATEGORY_BLOCK = "block"
_CATEGORY_TRIGGER = "trigger"
_CATEGORY_VARIABLE = "variable"
_CATEGORY_SOUND = "sound"


@dataclass(frozen=True, slots=True)
class Change:
    """One difference between two commands.

    ``path`` locates it — a block path like ``actions[1].then[0]`` or the name of
    a field, a trigger, a variable or a sound. ``old`` and ``new`` are already
    rendered to text, so the view and a test read the same thing; a value that
    does not apply (an addition has no old value) is ``None``.
    """

    kind: str
    category: str
    path: str
    label: str
    old: str | None = None
    new: str | None = None


@dataclass(frozen=True, slots=True)
class CommandDiff:
    """Every :class:`Change` between two commands, in reading order.

    Header fields first, then the block tree top-down, then triggers, variables
    and sounds — the order the editor lays the command out in, so the list reads
    like a walk through the command rather than a dump of a dictionary.
    """

    changes: tuple[Change, ...]

    @property
    def is_empty(self) -> bool:
        """Whether the two commands are structurally identical."""
        return not self.changes

    def of_category(self, category: str) -> tuple[Change, ...]:
        """The changes in one category, e.g. ``"block"`` or ``"trigger"``."""
        return tuple(change for change in self.changes if change.category == category)

    def summary(self) -> str:
        """A short Russian count: «+2 −1 ~3» or «без изменений»."""
        added = sum(1 for change in self.changes if change.kind == CHANGE_ADDED)
        removed = sum(1 for change in self.changes if change.kind == CHANGE_REMOVED)
        changed = sum(1 for change in self.changes if change.kind == CHANGE_CHANGED)
        if not self.changes:
            return "без изменений"
        parts: list[str] = []
        if added:
            parts.append(f"+{added}")
        if removed:
            parts.append(f"−{removed}")
        if changed:
            parts.append(f"~{changed}")
        return " ".join(parts)


def diff_commands(old: CommandModel, new: CommandModel) -> CommandDiff:
    """Compare two commands and return every structural difference."""
    changes: list[Change] = []
    _diff_fields(old, new, changes)
    _diff_block_lists(list(old.actions), list(new.actions), ("actions",), ("actions",), changes)
    _diff_triggers(old.triggers, new.triggers, changes)
    _diff_variables(old.variables, new.variables, changes)
    _diff_sounds(old.sounds, new.sounds, changes)
    return CommandDiff(changes=tuple(changes))


# ----------------------------------------------------------------------
# header fields
# ----------------------------------------------------------------------

#: Plain scalar fields of a command, with the label the editor shows them under.
_FIELDS: tuple[tuple[str, str], ...] = (
    ("name", "Имя"),
    ("description", "Описание"),
    ("enabled", "Включена"),
    ("priority", "Приоритет"),
    ("cooldown_ms", "Пауза, мс"),
    ("require_admin", "Требует прав администратора"),
)


def _diff_fields(old: CommandModel, new: CommandModel, changes: list[Change]) -> None:
    for attr, label in _FIELDS:
        before = getattr(old, attr)
        after = getattr(new, attr)
        if before != after:
            changes.append(
                Change(
                    kind=CHANGE_CHANGED,
                    category=_CATEGORY_FIELD,
                    path=label,
                    label=f"Изменено поле «{label}»",
                    old=_fmt(before),
                    new=_fmt(after),
                )
            )
    if list(old.tags) != list(new.tags):
        changes.append(
            Change(
                kind=CHANGE_CHANGED,
                category=_CATEGORY_FIELD,
                path="Метки",
                label="Изменены метки",
                old=", ".join(old.tags) or "—",
                new=", ".join(new.tags) or "—",
            )
        )


# ----------------------------------------------------------------------
# blocks
# ----------------------------------------------------------------------


def _diff_block_lists(
    old: list[ActionBlock],
    new: list[ActionBlock],
    old_base: tuple[str | int, ...],
    new_base: tuple[str | int, ...],
    changes: list[Change],
) -> None:
    """Align two sibling block lists by type and recurse into the matches.

    ``autojunk`` is off: block-type lists are short, and its heuristic would treat
    a type that repeats a lot — a body full of ``Say`` — as junk and mis-align it.
    """
    matcher = SequenceMatcher(a=[b.type for b in old], b=[b.type for b in new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for oi, nj in zip(range(i1, i2), range(j1, j2), strict=True):
                _diff_block(old[oi], new[nj], (*old_base, oi), (*new_base, nj), changes)
        elif tag == "delete":
            for oi in range(i1, i2):
                changes.append(_block_change(CHANGE_REMOVED, old[oi], (*old_base, oi)))
        elif tag == "insert":
            for nj in range(j1, j2):
                changes.append(_block_change(CHANGE_ADDED, new[nj], (*new_base, nj)))
        else:  # replace — a type changed here; report the swap as remove + add.
            for oi in range(i1, i2):
                changes.append(_block_change(CHANGE_REMOVED, old[oi], (*old_base, oi)))
            for nj in range(j1, j2):
                changes.append(_block_change(CHANGE_ADDED, new[nj], (*new_base, nj)))


def _block_change(kind: str, block: ActionBlock, path: tuple[str | int, ...]) -> Change:
    verb = "Добавлен блок" if kind == CHANGE_ADDED else "Удалён блок"
    return Change(
        kind=kind,
        category=_CATEGORY_BLOCK,
        path=_path_text(path),
        label=f"{verb} {block.type}",
        old=None if kind == CHANGE_ADDED else block.type,
        new=block.type if kind == CHANGE_ADDED else None,
    )


def _diff_block(
    old: ActionBlock,
    new: ActionBlock,
    old_path: tuple[str | int, ...],
    new_path: tuple[str | int, ...],
    changes: list[Change],
) -> None:
    """Two blocks of the same type at aligned positions: compare and recurse."""
    where = _path_text(new_path)
    _diff_params(old.params, new.params, where, changes)
    if old.enabled != new.enabled:
        changes.append(
            Change(
                kind=CHANGE_CHANGED,
                category=_CATEGORY_BLOCK,
                path=where,
                label="Блок включён" if new.enabled else "Блок отключён",
                old=_fmt(old.enabled),
                new=_fmt(new.enabled),
            )
        )
    if old.comment != new.comment:
        changes.append(
            Change(
                kind=CHANGE_CHANGED,
                category=_CATEGORY_BLOCK,
                path=where,
                label="Изменён комментарий блока",
                old=old.comment or "—",
                new=new.comment or "—",
            )
        )
    if old.on_error != new.on_error:
        changes.append(
            Change(
                kind=CHANGE_CHANGED,
                category=_CATEGORY_BLOCK,
                path=where,
                label="Изменена реакция на ошибку",
                old=str(old.on_error),
                new=str(new.on_error),
            )
        )
    _diff_block_sound(old.sound, new.sound, where, changes)
    for (wire, old_children), (_, new_children) in zip(old.branches(), new.branches(), strict=True):
        _diff_block_lists(old_children, new_children, (*old_path, wire), (*new_path, wire), changes)


def _diff_params(
    old: dict[str, Any], new: dict[str, Any], where: str, changes: list[Change]
) -> None:
    for key in sorted(set(old) | set(new)):
        in_old, in_new = key in old, key in new
        if in_old and not in_new:
            changes.append(
                Change(
                    CHANGE_REMOVED,
                    _CATEGORY_BLOCK,
                    where,
                    f"Удалён параметр {key}",
                    _fmt(old[key]),
                    None,
                )
            )
        elif in_new and not in_old:
            changes.append(
                Change(
                    CHANGE_ADDED,
                    _CATEGORY_BLOCK,
                    where,
                    f"Добавлен параметр {key}",
                    None,
                    _fmt(new[key]),
                )
            )
        elif old[key] != new[key]:
            changes.append(
                Change(
                    CHANGE_CHANGED,
                    _CATEGORY_BLOCK,
                    where,
                    f"Изменён параметр {key}",
                    _fmt(old[key]),
                    _fmt(new[key]),
                )
            )


def _diff_block_sound(
    old: SoundBinding | None, new: SoundBinding | None, where: str, changes: list[Change]
) -> None:
    if old is None and new is None:
        return
    old_text = _fmt(old.model_dump(mode="json", exclude_none=True)) if old is not None else None
    new_text = _fmt(new.model_dump(mode="json", exclude_none=True)) if new is not None else None
    if old_text == new_text:
        return
    if old is None:
        label, kind = "Добавлен звук блока", CHANGE_ADDED
    elif new is None:
        label, kind = "Удалён звук блока", CHANGE_REMOVED
    else:
        label, kind = "Изменён звук блока", CHANGE_CHANGED
    changes.append(Change(kind, _CATEGORY_BLOCK, where, label, old_text, new_text))


# ----------------------------------------------------------------------
# triggers
# ----------------------------------------------------------------------


def _diff_triggers(
    old: Sequence[TriggerModel], new: Sequence[TriggerModel], changes: list[Change]
) -> None:
    old_by_id = {_trigger_identity(t): t for t in old}
    new_by_id = {_trigger_identity(t): t for t in new}
    for identity in _ordered_union(old_by_id, new_by_id):
        before = old_by_id.get(identity)
        after = new_by_id.get(identity)
        if before is None and after is not None:
            changes.append(
                Change(
                    CHANGE_ADDED,
                    _CATEGORY_TRIGGER,
                    _trigger_desc(after),
                    f"Добавлен триггер: {_trigger_desc(after)}",
                    None,
                    _trigger_desc(after),
                )
            )
        elif after is None and before is not None:
            changes.append(
                Change(
                    CHANGE_REMOVED,
                    _CATEGORY_TRIGGER,
                    _trigger_desc(before),
                    f"Удалён триггер: {_trigger_desc(before)}",
                    _trigger_desc(before),
                    None,
                )
            )
        elif before is not None and after is not None:
            old_dump = before.model_dump(mode="json", exclude_none=True)
            new_dump = after.model_dump(mode="json", exclude_none=True)
            if old_dump != new_dump:
                changes.append(
                    Change(
                        CHANGE_CHANGED,
                        _CATEGORY_TRIGGER,
                        _trigger_desc(after),
                        f"Изменён триггер: {_trigger_desc(after)}",
                        _fmt(old_dump),
                        _fmt(new_dump),
                    )
                )


def _trigger_identity(trigger: TriggerModel) -> tuple[str, str]:
    if isinstance(trigger, VoiceTrigger):
        return ("voice", trigger.phrase.casefold())
    if isinstance(trigger, HotkeyTrigger):
        return ("hotkey", trigger.combo.casefold())
    if isinstance(trigger, EventTrigger):
        return ("event", trigger.event_name.casefold())
    return ("timer", trigger.cron or (trigger.fire_at.isoformat() if trigger.fire_at else ""))


def _trigger_desc(trigger: TriggerModel) -> str:
    if isinstance(trigger, VoiceTrigger):
        return f"голос «{trigger.phrase}»"
    if isinstance(trigger, HotkeyTrigger):
        return f"хоткей {trigger.combo}"
    if isinstance(trigger, EventTrigger):
        return f"событие {trigger.event_name}"
    if trigger.cron:
        return f"расписание {trigger.cron}"
    return f"таймер {trigger.fire_at.isoformat()}" if trigger.fire_at else "таймер"


# ----------------------------------------------------------------------
# variables
# ----------------------------------------------------------------------


def _diff_variables(
    old: Sequence[VariableModel], new: Sequence[VariableModel], changes: list[Change]
) -> None:
    old_by_name = {v.name: v for v in old}
    new_by_name = {v.name: v for v in new}
    for name in _ordered_union(old_by_name, new_by_name):
        before = old_by_name.get(name)
        after = new_by_name.get(name)
        if before is None and after is not None:
            changes.append(
                Change(
                    CHANGE_ADDED,
                    _CATEGORY_VARIABLE,
                    name,
                    f"Добавлена переменная {name}",
                    None,
                    _variable_desc(after),
                )
            )
        elif after is None and before is not None:
            changes.append(
                Change(
                    CHANGE_REMOVED,
                    _CATEGORY_VARIABLE,
                    name,
                    f"Удалена переменная {name}",
                    _variable_desc(before),
                    None,
                )
            )
        elif (
            before is not None
            and after is not None
            and _variable_desc(before) != _variable_desc(after)
        ):
            changes.append(
                Change(
                    CHANGE_CHANGED,
                    _CATEGORY_VARIABLE,
                    name,
                    f"Изменена переменная {name}",
                    _variable_desc(before),
                    _variable_desc(after),
                )
            )


def _variable_desc(variable: VariableModel) -> str:
    return _fmt(
        {
            "type": str(variable.type),
            "scope": str(variable.scope),
            "default": variable.default,
            "persistent": variable.persistent,
        }
    )


# ----------------------------------------------------------------------
# sounds
# ----------------------------------------------------------------------


def _diff_sounds(
    old: Sequence[SoundBinding], new: Sequence[SoundBinding], changes: list[Change]
) -> None:
    old_by_id = {_sound_identity(s): s for s in old}
    new_by_id = {_sound_identity(s): s for s in new}
    for identity in _ordered_union(old_by_id, new_by_id):
        before = old_by_id.get(identity)
        after = new_by_id.get(identity)
        where = f"{identity[0]}: {identity[1]}"
        if before is None and after is not None:
            changes.append(
                Change(
                    CHANGE_ADDED,
                    _CATEGORY_SOUND,
                    where,
                    f"Добавлен звук ({where})",
                    None,
                    _fmt(after.model_dump(mode="json", exclude_none=True)),
                )
            )
        elif after is None and before is not None:
            changes.append(
                Change(
                    CHANGE_REMOVED,
                    _CATEGORY_SOUND,
                    where,
                    f"Удалён звук ({where})",
                    _fmt(before.model_dump(mode="json", exclude_none=True)),
                    None,
                )
            )
        elif before is not None and after is not None:
            old_dump = before.model_dump(mode="json", exclude_none=True)
            new_dump = after.model_dump(mode="json", exclude_none=True)
            if old_dump != new_dump:
                changes.append(
                    Change(
                        CHANGE_CHANGED,
                        _CATEGORY_SOUND,
                        where,
                        f"Изменён звук ({where})",
                        _fmt(old_dump),
                        _fmt(new_dump),
                    )
                )


def _sound_identity(sound: SoundBinding) -> tuple[str, str]:
    return (str(sound.stage), sound.value)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _ordered_union(old: dict[Any, Any], new: dict[Any, Any]) -> list[Any]:
    """Keys of both maps, old ones first in their order, then new-only ones."""
    order = list(old)
    order.extend(key for key in new if key not in old)
    return order


def _path_text(path: tuple[str | int, ...]) -> str:
    """``actions[1].then[0]`` — the same rendering as ``BlockLocation.path_text``."""
    parts: list[str] = []
    for step in path:
        if isinstance(step, int):
            parts.append(f"[{step}]")
        elif parts:
            parts.append(f".{step}")
        else:
            parts.append(step)
    return "".join(parts)


def _fmt(value: Any) -> str:
    """Render a value for a table cell: booleans in Russian, structures as JSON."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)
