"""The one audio-output owner for macro sounds, and the handle that shares it.

Both places that play a command's sounds need the same thing: a
:class:`~ayris.actions.macros.sounds.library.SoundLibrary` sitting on top of the
single device owner. The macro engine plays a stage's binding when a command
fires; the command editor's «Прослушать» button previews a binding while it is
being edited. Nothing else in the running application builds this path — the TTS
worker only synthesises PCM, the audio worker only captures the microphone — so
it is built here, once, and handed to both.

:func:`build_sound_library` assembles
``TtsPlayer`` → :class:`~ayris.actions.macros.sounds.mixer.PlayerOutput` →
:class:`~ayris.actions.macros.sounds.mixer.SoundMixer` →
:class:`~ayris.actions.macros.sounds.library.SoundLibrary`. The player opens no
device until the first sound, so constructing it at start-up costs nothing and
cannot fail for want of a speaker. A build that still fails — no PortAudio at
all, a broken builtin manifest — degrades to ``None`` rather than aborting the
launch, exactly as the command store and the sound importer do.

The synthesiser is left ``None``: the TTS→speaker path is not wired at runtime
yet (see :mod:`ayris.core.pipeline_app`), so a ``tts:`` binding raises the
library's «Синтез звука не настроен» and the engine logs it per binding. File
and builtin sounds need no synthesiser and play immediately.

:func:`set_active_sound_library` / :func:`active_sound_library` are the shared
handle, the same shape as
:func:`~ayris.gui.widgets.resource_monitor.set_active_worker_control`: the
dispatcher registers the library it built so the settings window's command
editor can preview through the very same instance and device.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Final

from ayris.actions.macros.sounds.library import SoundLibrary
from ayris.actions.macros.sounds.mixer import PlayerOutput, SoundMixer
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from ayris.actions.macros.schema import CommandModel

__all__ = [
    "active_sound_library",
    "build_sound_library",
    "set_active_sound_library",
]

_log = get_logger(__name__)


def build_sound_library(
    *,
    sounds_dir: Path,
    cache_dir: Path,
    commands: Iterable[CommandModel] | None = None,
) -> tuple[SoundLibrary, Callable[[], None]] | None:
    """Build the sound library over a fresh device owner, or ``None`` on failure.

    Args:
        sounds_dir: The profile's folder of imported («Файл») sounds.
        cache_dir: Where synthesised phrases would be cached; unused until a
            synthesiser is wired, but the library wants the path.
        commands: Library commands, for the catalog's usage counts. Optional; the
            engine and the preview button do not need them.

    Returns:
        The library and a callback that stops its player and releases the device,
        or ``None`` when the path could not be built (no PortAudio, or a broken
        builtin manifest). A ``None`` return disables sounds rather than crashing,
        the same degradation as :func:`~ayris.gui.tabs.commands.build_store`.

    The player opens no device here — it does that lazily at the first sound (see
    :meth:`~ayris.audio.tts.player.TtsPlayer.start`) — so building this at start-up
    costs nothing and cannot fail for want of a speaker.
    """
    try:
        from ayris.audio.tts.player import TtsPlayer

        player = TtsPlayer()
        library = SoundLibrary(
            sounds_dir,
            cache_dir,
            SoundMixer(PlayerOutput(player)),
            commands=commands,
        )
    except Exception:
        _log.exception("не удалось собрать вывод звука команд")
        return None
    return library, player.stop


_ACTIVE_LIBRARY: SoundLibrary | None = None
_LIBRARY_LOCK: Final = threading.Lock()


def set_active_sound_library(library: SoundLibrary | None) -> None:
    """Register (or clear) the library the command editor previews through."""
    global _ACTIVE_LIBRARY
    with _LIBRARY_LOCK:
        _ACTIVE_LIBRARY = library


def active_sound_library() -> SoundLibrary | None:
    """The library registered with :func:`set_active_sound_library`, if any."""
    with _LIBRARY_LOCK:
        return _ACTIVE_LIBRARY
