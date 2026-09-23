"""Applying a saved command without a restart, task 54.

A command edited in the editor has to start working the moment it is saved: a new
hotkey bound, an old one released, the voice phrase in the NLU index, the schedule
and the event subscriptions rebuilt. None of that is done here by hand — the
trigger subsystems (:class:`~ayris.triggers.dispatcher.TriggerDispatcher`, the
hotkey manager of task 37, the NLU :class:`~ayris.nlu.index.TriggerIndex`) already
listen for :class:`~ayris.core.events.CommandsChanged` and re-register the one
command it names. This module owns the *order* around that signal, which is the
part that has to be right:

    validate → write the row, its triggers and a version in one transaction →
    publish ``CommandsChanged`` so the subsystems re-register → publish
    ``CommandReloaded`` for the tree and the overlay.

The guarantee of step one is the reason it is a module and not three lines in the
editor: **an error leaves the previously registered version running.** Validation
fails, or the write raises — either way nothing is published, so no subsystem
re-registers, and the command keeps firing as it did before the save. The command
is never disabled as a side effect of a failed save.

Re-registration is idempotent by construction: every subsystem rebuilds *its* view
of the command from the rows on a ``CommandsChanged``, so saving twice leaves one
subscription per phrase and one hotkey binding, never two. A command already
running when the save lands is not touched — the engine holds its own model and
plays it to the end; the new version is what the *next* trigger starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ayris.actions.macros.validator import (
    MacroValidationError,
    ValidationReport,
    validate_command,
)
from ayris.core.events import (
    COMMANDS_CHANGE_DELETED,
    COMMANDS_CHANGE_SAVED,
    CommandReloaded,
    CommandsChanged,
)
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.macros.schema import CommandModel
    from ayris.actions.registry import ActionRegistry
    from ayris.core.events import EventBus

__all__ = ["CommandSaver", "HotReloader", "ReloadResult"]

_log = get_logger(__name__)


class CommandSaver(Protocol):
    """What :class:`HotReloader` needs of the store: an atomic, versioning save.

    :meth:`~ayris.gui.widgets.command_tree_model.CommandTreeStore.save_command`
    satisfies it. A test can pass anything with the same one method.
    """

    def save_command(self, model: CommandModel) -> CommandModel:
        """Persist the whole command — row, triggers, declarations, version — atomically."""
        ...


@dataclass(frozen=True, slots=True)
class ReloadResult:
    """The outcome of a successful apply: what was saved and how it validated."""

    command: CommandModel
    report: ValidationReport


class HotReloader:
    """Applies a saved command and re-registers it live, in the right order."""

    def __init__(
        self,
        saver: CommandSaver,
        bus: EventBus,
        *,
        registry: ActionRegistry | None = None,
    ) -> None:
        self._saver = saver
        self._bus = bus
        self._registry = registry

    def apply_command(self, model: CommandModel) -> ReloadResult:
        """Validate, persist and re-register one command.

        The steps run in the order task 54 fixes, and the first failing one stops
        the rest: a command that does not validate is never written, and a write
        that raises is never published. In both cases the previously registered
        version keeps running — this method changes nothing that a subsystem can
        see until the save has already succeeded.

        Raises:
            MacroValidationError: the command has validation errors (warnings do
                not stop a save). Nothing was written or published.
            AyrisError / DatabaseError: the write failed. The transaction rolled
                back; nothing was published.
        """
        report = validate_command(model, registry=self._registry)
        if not report.ok:
            # Nothing written, nothing published — the old version is still live.
            raise MacroValidationError(report)

        saved = self._saver.save_command(model)
        self._republish(saved, COMMANDS_CHANGE_SAVED)
        return ReloadResult(command=saved, report=report)

    def retire_command(self, command_id: int, *, deleted: bool = False) -> None:
        """Tear a command's registrations down after it was disabled or deleted.

        Publishes the same two events with a ``deleted`` change, so the subsystems
        drop the command's hotkey, its phrase and its schedule, and the tree and
        overlay stop showing it as live. Idempotent: retiring an already-retired
        command re-publishes to an empty registration and changes nothing.
        """
        change = COMMANDS_CHANGE_DELETED if deleted else COMMANDS_CHANGE_SAVED
        self._bus.publish(CommandsChanged(command_id=command_id, change=change))
        self._bus.publish(CommandReloaded(command_id=command_id, change=change))

    def _republish(self, saved: CommandModel, change: str) -> None:
        """Drive re-registration, then announce the reload to the UI listeners.

        ``CommandsChanged`` first: the trigger subsystems reload their view of the
        command from it, so by the time ``CommandReloaded`` reaches the tree the
        command is already live in its new form and the tree can trust the state.
        A command with no id was never persisted and cannot be re-registered — that
        is a programming error upstream, logged rather than published.
        """
        if saved.id is None:
            _log.error("apply_command produced a command without an id; not republishing")
            return
        self._bus.publish(CommandsChanged(command_id=saved.id, change=change))
        self._bus.publish(CommandReloaded(command_id=saved.id, change=change))
