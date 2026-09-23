"""Undo/redo for the command editor, task 54.

The editor of task 52 and the node view of task 53 are two views over one
:class:`~ayris.actions.macros.schema.CommandModel`; the undo stack is therefore
over the *model*, not over either widget, and switching «Список ↔ Ноды» carries
the same history. Each entry is a whole-model snapshot before and after one edit,
so undo is a swap, not a replay — a design that cannot drift out of sync with the
model the way a stack of inverse operations would, at the cost of holding two
copies per step (a command is small; a hundred steps is kilobytes).

The description each step carries — «Добавлен блок Say», «Изменён параметр level»,
«Перемещён блок» — is not tracked by the caller. It is *derived* from the
structural diff of the two snapshots (:mod:`ayris.actions.macros.diff`), so one
`record` call after any edit, however the edit was made, is labelled correctly.

Rapid edits of one field coalesce: typing in a text parameter records on every
keystroke, but within :attr:`coalesce_seconds` a second edit of the *same* field
folds into the first — one «Изменён параметр text» on the stack, undoable in one
step, not one per character. The signature that decides «same field» comes from
the diff too: same single change, same path, same field.

This module has no Qt import. The widget layer drives it — records after an edit,
calls :meth:`undo` / :meth:`redo` on the shortcuts, reads :meth:`descriptions`
for the dropdown — and applies the returned model to its views.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ayris.actions.macros.diff import diff_commands

if TYPE_CHECKING:
    from ayris.actions.macros.schema import CommandModel

__all__ = ["UndoEntry", "UndoStack"]

#: Default window for folding successive edits of one field into one step.
_DEFAULT_COALESCE_SECONDS = 0.8


@dataclass(slots=True)
class UndoEntry:
    """One undoable step: the model before and after, and how to describe it."""

    description: str
    before: CommandModel
    after: CommandModel
    signature: str
    at: float


class UndoStack:
    """A shared, model-level undo/redo history with human-readable steps.

    Time is injected (``clock``) so a test can drive coalescing without sleeping.
    """

    def __init__(
        self,
        *,
        coalesce_seconds: float = _DEFAULT_COALESCE_SECONDS,
        limit: int = 200,
        clock: object | None = None,
    ) -> None:
        import time

        self._coalesce = coalesce_seconds
        self._limit = limit
        self._clock = clock if callable(clock) else time.monotonic
        self._baseline: CommandModel | None = None
        self._undo: list[UndoEntry] = []
        self._redo: list[UndoEntry] = []

    # -- lifecycle ----------------------------------------------------------

    def reset(self, model: CommandModel | None) -> None:
        """Start over on a new command: clear both stacks, remember the baseline.

        The baseline is the model as loaded — the state an undo of the first
        recorded step returns to. Called when the editor opens a command; the task
        requires the stack reset on command switch.
        """
        self._baseline = None if model is None else model.model_copy(deep=True)
        self._undo.clear()
        self._redo.clear()

    def record(self, model: CommandModel) -> bool:
        """Note that the model changed since the last recorded state.

        Diffs the new model against the previous top-of-stack (or the baseline) to
        derive a description and a coalescing signature. Returns ``False`` — and
        records nothing — when the diff is empty, so a rebuild that changed nothing
        does not push a no-op step. Recording clears the redo stack, as an edit
        after an undo always does.
        """
        if self._baseline is None:
            self._baseline = model.model_copy(deep=True)
            return False
        previous = self._current()
        diff = diff_commands(previous, model)
        if diff.is_empty:
            return False

        snapshot = model.model_copy(deep=True)
        description = _describe(diff)
        signature = _signature(diff)
        now = float(self._clock())

        if self._can_coalesce(signature, now):
            top = self._undo[-1]
            top.after = snapshot
            top.description = description
            top.at = now
            return True

        self._undo.append(
            UndoEntry(
                description=description,
                before=previous.model_copy(deep=True),
                after=snapshot,
                signature=signature,
                at=now,
            )
        )
        del self._undo[: max(0, len(self._undo) - self._limit)]
        self._redo.clear()
        return True

    # -- navigation ---------------------------------------------------------

    def undo(self) -> CommandModel | None:
        """Step back one operation, returning the model to apply, or ``None``."""
        if not self._undo:
            return None
        entry = self._undo.pop()
        self._redo.append(entry)
        return entry.before.model_copy(deep=True)

    def redo(self) -> CommandModel | None:
        """Step forward one operation, returning the model to apply, or ``None``."""
        if not self._redo:
            return None
        entry = self._redo.pop()
        self._undo.append(entry)
        return entry.after.model_copy(deep=True)

    # -- queries ------------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def descriptions(self) -> list[str]:
        """The undoable steps, newest last — for the «последние операции» dropdown."""
        return [entry.description for entry in self._undo]

    def undo_description(self) -> str | None:
        """What Ctrl+Z would undo, for the menu label, or ``None``."""
        return self._undo[-1].description if self._undo else None

    def redo_description(self) -> str | None:
        """What Ctrl+Shift+Z would redo, for the menu label, or ``None``."""
        return self._redo[-1].description if self._redo else None

    # -- internals ----------------------------------------------------------

    def _current(self) -> CommandModel:
        if self._undo:
            return self._undo[-1].after
        assert self._baseline is not None
        return self._baseline

    def _can_coalesce(self, signature: str, now: float) -> bool:
        if not self._undo or not signature:
            return False
        top = self._undo[-1]
        return top.signature == signature and (now - top.at) <= self._coalesce


def _describe(diff: object) -> str:
    """A short Russian label for a step, from its diff.

    One change speaks for itself — its own label. Several changes in one step
    (a paste, a move that touched two lists) get a count, because naming one of
    them would mislead.
    """
    from ayris.actions.macros.diff import CommandDiff

    assert isinstance(diff, CommandDiff)
    changes = diff.changes
    if len(changes) == 1:
        return changes[0].label
    return f"Изменений: {len(changes)}"


def _signature(diff: object) -> str:
    """The coalescing key: non-empty only for a single-field, in-place change.

    Adding or removing a block, or any multi-change step, gets an empty signature
    and never coalesces — only the repeated tweak of one parameter or one field
    of one block folds together.
    """
    from ayris.actions.macros.diff import CHANGE_CHANGED, CommandDiff

    assert isinstance(diff, CommandDiff)
    if len(diff.changes) != 1:
        return ""
    change = diff.changes[0]
    if change.kind != CHANGE_CHANGED:
        return ""
    return f"{change.category}:{change.path}:{change.label}"
