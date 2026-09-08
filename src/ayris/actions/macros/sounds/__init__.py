"""Sound accompaniment for macro stages."""

from ayris.actions.macros.sounds.binding import SoundBindingPlayer, bindings_for_stage
from ayris.actions.macros.sounds.importer import ImportResult, SoundImportError, import_sound
from ayris.actions.macros.sounds.library import (
    CatalogSound,
    RouterSoundSynthesizer,
    SoundLibrary,
    SoundLibraryError,
)
from ayris.actions.macros.sounds.mixer import MixPolicy, PlayerOutput, SoundHandle, SoundMixer

__all__ = [
    "CatalogSound",
    "ImportResult",
    "MixPolicy",
    "PlayerOutput",
    "RouterSoundSynthesizer",
    "SoundBindingPlayer",
    "SoundHandle",
    "SoundImportError",
    "SoundLibrary",
    "SoundLibraryError",
    "SoundMixer",
    "bindings_for_stage",
    "import_sound",
]
