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
from ayris.actions.macros.sounds.runtime import (
    active_sound_library,
    build_sound_library,
    set_active_sound_library,
)

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
    "active_sound_library",
    "bindings_for_stage",
    "build_sound_library",
    "import_sound",
    "set_active_sound_library",
]
