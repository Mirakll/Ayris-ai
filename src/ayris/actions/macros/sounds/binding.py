"""Resolution of command- and block-level sound bindings."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ayris.actions.macros.schema import SoundBinding, SoundStage

if TYPE_CHECKING:
    from ayris.actions.macros.schema import ActionBlock, CommandModel

__all__ = ["SoundBinding", "SoundBindingPlayer", "bindings_for_stage"]


class SoundBindingPlayer(Protocol):
    """The small part of the sound service used by the macro engine."""

    def play_binding(self, binding: SoundBinding, *, owner: str = "") -> object:
        """Resolve and play one binding, optionally waiting as requested."""
        ...


def bindings_for_stage(
    command: CommandModel,
    stage: SoundStage,
    block: ActionBlock | None = None,
) -> tuple[SoundBinding, ...]:
    """Return effective bindings for a stage; a block binding replaces the command."""
    if block is not None and block.sound is not None and block.sound.stage is stage:
        return (block.sound,)
    return tuple(binding for binding in command.sounds if binding.stage is stage)
