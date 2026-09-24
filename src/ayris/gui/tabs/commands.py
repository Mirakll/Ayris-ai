"""Tab «Команды»: the command library, with a full-width list and a separate editor.

Two screens live in one :class:`~PySide6.QtWidgets.QStackedWidget`. The library screen
is the :class:`~ayris.gui.widgets.command_tree.CommandTree` (task 51) given the whole
width — search, filters and the command list. Creating a command, or asking to edit an
existing one, switches to the editor screen: the
:class:`~ayris.gui.widgets.macro_editor.MacroEditor` (tasks 52–54, opening on the node
canvas of task 53) under a «← К списку команд» bar that guards unsaved edits on the
way back.

The store is built lazily over the live database, the same shape the «Обновления» tab
uses: constructing every settings page must stay cheap, and a page the user may never
open should not touch storage. The tab bridges the tree to the rest of Ayris through
the event bus — it publishes :class:`CommandsChanged` after a change here, and rebuilds
the tree on a :class:`CommandsChanged` or :class:`ProfileSwitched` from elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QStackedWidget

from ayris.actions.macros.schema import SoundBinding
from ayris.actions.macros.sounds import (
    SoundHandle,
    SoundLibrary,
    active_sound_library,
    import_sound,
)
from ayris.core.config import ConfigManager
from ayris.core.events import CommandReloaded, CommandsChanged, EventBus
from ayris.core.profile import ProfileSwitched
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.command_tree import CommandTree
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.macro_editor import MacroEditor, MacroEditorServices
from ayris.utils.logger import get_logger

__all__ = ["CommandsTab", "build_store"]

_log = get_logger(__name__)


class _ProfileSoundImporter:
    """Copy a picked file into the profile's sounds folder as a portable WAV.

    The editor's «Файл» binding stores only a filename inside that folder, so a sound
    the user picks anywhere on disk is decoded, resampled and saved here first. MP3 and
    OGG need the PyAV decoder from the full package; :func:`import_sound` raises a
    Russian :class:`~ayris.actions.macros.sounds.importer.SoundImportError` when it is
    missing, which the section shows next to the row.
    """

    def __init__(self, sounds_dir: Path) -> None:
        self._sounds_dir = sounds_dir

    def import_file(self, source: Path) -> str:
        return import_sound(source, self._sounds_dir).path.name


def build_store(*, version_limit: int = 20) -> CommandTreeStore | None:
    """A store over the live database and active profile, or ``None`` if unavailable.

    Mirrors the lazy backend of the «Обновления» tab: the tab must build even when
    storage is not ready yet, degrading to a disabled panel rather than crashing.
    """
    try:
        from ayris.core.database import get_database
        from ayris.core.repositories import Repositories

        repositories = Repositories(get_database())
        active = repositories.profiles.active()
        if active is None or active.id is None:
            return None
        return CommandTreeStore(repositories, active.id, version_limit=version_limit)
    except Exception:
        _log.exception("не удалось построить хранилище дерева команд")
        return None


def _build_sound_importer() -> _ProfileSoundImporter | None:
    """An importer over the active profile's sounds folder, or ``None`` if unavailable.

    A missing folder must degrade to a disabled «Выбрать файл…» button, not a crash,
    the same way :func:`build_store` degrades when storage is not ready.
    """
    try:
        from ayris.core.paths import get_paths

        return _ProfileSoundImporter(get_paths().sounds_dir)
    except Exception:
        _log.exception("не удалось подготовить импорт звуков")
        return None


class _LibrarySoundPreview:
    """Play a binding for the editor's «Прослушать», through the shared library.

    The widget never touches an audio library itself (see
    :class:`~ayris.gui.widgets.sound_binding.SoundPreview`); it calls this facade,
    which routes to the one :class:`~ayris.actions.macros.sounds.library.SoundLibrary`
    the dispatcher built and the macro engine plays through. Preview never waits on
    playback — a long sound must not freeze the editor — so it goes straight through
    the mixer with ``wait=False`` under its own «preview» owner, which
    :meth:`stop` then cancels without touching a command's own sounds.
    """

    def __init__(self, library: SoundLibrary) -> None:
        self._library = library

    def preview_binding(self, binding: SoundBinding) -> SoundHandle:
        audio = self._library.resolve(binding)
        volume = (binding.volume if binding.volume is not None else 100) / 100
        return self._library.mixer.play(audio, volume=volume, owner="preview", wait=False)

    def stop(self) -> None:
        self._library.mixer.stop("preview")

    def duration_ms(self, binding: SoundBinding) -> int | None:
        return int(self._library.resolve(binding).duration_ms)


def _build_sound_preview() -> _LibrarySoundPreview | None:
    """A preview over the running sound library, or ``None`` when none is up.

    The dispatcher registers the library at start-up; without it (storage not ready,
    no PortAudio) the «Прослушать» button stays disabled rather than the editor
    building a second device owner of its own.
    """
    library = active_sound_library()
    return _LibrarySoundPreview(library) if library is not None else None


class CommandsTab(SettingsTab):
    """The «Команды» settings page: full-width library, editor on a second screen."""

    #: Fired ``True`` when the editor screen opens and ``False`` when the library
    #: returns, so the window can collapse the settings nav + search and give the
    #: node canvas the whole layer while a command is being edited.
    immersive_changed = Signal(bool)

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        store: CommandTreeStore | None = None,
    ) -> None:
        super().__init__("commands", "Команды", ("commands", "actions"), manager, theme, bus)
        # No config field is bound here, so the reset/dirty chrome would only
        # confuse a library page.
        self.reset_button.hide()
        self.dirty_label.hide()
        # This page is not a form of narrow settings rows but a full-width editor:
        # give the tree and the node canvas the whole layer. Pull the «Команды»
        # heading up almost to the search field (tiny top margin) and shrink the
        # page padding on every side, so the editor breathes out to the edges.
        self._layout.setContentsMargins(
            theme.metric("spacing_xs"),
            theme.metric("spacing_xs"),
            theme.metric("spacing_xs"),
            theme.metric("spacing_xs"),
        )
        self._layout.setSpacing(theme.metric("spacing_xs"))
        self._bus = bus
        # Defaults set before any early return, so dispose() is always safe even
        # when the library could not be opened.
        self._tree: CommandTree | None = None
        self._editor: MacroEditor | None = None
        self._store: CommandTreeStore | None = None
        self._unsub_commands: Callable[[], None] = lambda: None
        self._unsub_reloaded: Callable[[], None] = lambda: None
        self._unsub_profile: Callable[[], None] = lambda: None
        self._suppress_reload = False

        commands_config = manager.settings.commands
        resolved = (
            store
            if store is not None
            else build_store(version_limit=commands_config.version_history_limit)
        )

        self._stack = QStackedWidget()
        self.body.addWidget(self._stack)

        if resolved is None:
            notice = QLabel(
                "Библиотека команд недоступна: не удалось открыть профиль. "
                "Откройте вкладку позже или перезапустите Ayris."
            )
            notice.setProperty("role", "secondary")
            notice.setWordWrap(True)
            self._stack.addWidget(notice)
            return

        self._store = resolved
        self._tree = CommandTree(resolved, theme)
        self._tree.command_edit_requested.connect(self._open_editor)
        self._tree.tree_changed.connect(self._on_tree_changed)
        # Screen 0 — the library, given the whole width of the tab.
        self._stack.addWidget(self._tree)

        # A bus turns a save into a live re-registration (task 54): the editor's
        # HotReloader publishes CommandsChanged + CommandReloaded itself, so the tab
        # must not also publish on command_saved — that is what _on_command_saved
        # reconciles below.
        services = MacroEditorServices(
            bus=bus,
            sound_importer=_build_sound_importer(),
            sound_preview=_build_sound_preview(),
            draft_autosave_s=commands_config.draft_autosave_s,
            action_view=commands_config.action_view,
            on_action_view_changed=self._save_action_view,
        )
        self._editor = MacroEditor(resolved, theme, services=services)
        self._editor.command_saved.connect(self._on_command_saved)
        self._editor.dirty_changed.connect(self._on_editor_dirty)
        # The editor now carries its own browser-style top row (crumb · name · section
        # tabs, the mockup's `.topbar`), so the «← К списку команд» crumb lives there and
        # only asks us to leave — the unsaved-edit guard stays here.
        self._editor.back_requested.connect(self._back_to_library)

        # Screen 1 — the editor itself; its top row is the only chrome above the canvas.
        self._stack.addWidget(self._editor)

        if bus is not None:
            self._unsub_commands = bus.subscribe(CommandsChanged, self._on_commands_changed)
            self._unsub_reloaded = bus.subscribe(CommandReloaded, self._on_command_reloaded)
            self._unsub_profile = bus.subscribe(ProfileSwitched, self._on_profile_switched)

    # -- event wiring -------------------------------------------------------

    def _set_screen(self, index: int) -> None:
        """Switch library ↔ editor and announce immersion so the nav can collapse."""
        self._stack.setCurrentIndex(index)
        # The library keeps the «Команды» page heading; the editor hides it so its own
        # browser top row (crumb · name · tabs) is the only chrome above the canvas —
        # the mockup's single line, which lets the canvas rise.
        self.title_label.setVisible(index == 0)
        self.immersive_changed.emit(index == 1)

    def _open_editor(self, command_id: int) -> None:
        """Load a command into the editor and switch to the editor screen."""
        if self._editor is None:
            return
        self._editor.load_command(command_id)
        self._set_screen(1)

    def reveal_command(self, command_id: int) -> None:
        """Show a command in the library, selected — the «Открыть команду» target.

        Task 55's «Горячие клавиши» tab links each command hotkey here. Any unsaved
        editor edits are guarded first (task 54); if the user chooses to keep
        editing, the jump is abandoned rather than dropping their changes silently.
        """
        if self._tree is None:
            return
        if (
            self._stack.currentIndex() == 1
            and self._editor is not None
            and not self._editor.guard_unsaved()
        ):
            return
        self._set_screen(0)
        self._tree.refresh()
        self._tree.select_command(command_id)

    def _back_to_library(self) -> None:
        """Return to the list, guarding unsaved edits (task 54) before leaving.

        Cancelling the «Сохранить / Не сохранять / Отмена» prompt keeps the editor
        open on its command; otherwise the list is shown again and refreshed so a
        rename or a new command is reflected.
        """
        if self._editor is not None and not self._editor.guard_unsaved():
            return
        self._set_screen(0)
        if self._tree is not None:
            self._tree.refresh()

    def _on_editor_dirty(self, dirty: bool) -> None:
        if self._tree is not None and self._editor is not None:
            self._tree.set_dirty_command(self._editor.command_id if dirty else None)

    def _on_command_saved(self, _command_id: int) -> None:
        # With a bus, the editor's HotReloader already published a targeted
        # CommandsChanged (re-registering the command) and a CommandReloaded (which
        # refreshes the tree below), so the tab must stay silent or it would fire a
        # bare, full-reload CommandsChanged on top. Without a bus there is nothing
        # else to refresh the tree, so fall back to the direct refresh.
        if self._bus is None and self._tree is not None:
            self._tree.refresh()

    def _save_action_view(self, view: str) -> None:
        # Persist the user's «Список ↔ Ноды» choice so the next command and the next
        # launch open in the same view. A rejected write must not break the editor —
        # the toggle already switched, it just would not be remembered.
        if view == self._manager.settings.commands.action_view:
            return
        try:
            self._manager.apply({"commands.action_view": view})
        except Exception:
            _log.exception("не удалось сохранить режим редактора действий")

    def _on_tree_changed(self) -> None:
        if self._bus is not None:
            # Guard the echo: our own publish comes straight back as CommandsChanged.
            self._suppress_reload = True
            try:
                self._bus.publish(CommandsChanged())
            finally:
                self._suppress_reload = False

    def _on_commands_changed(self, _event: CommandsChanged) -> None:
        if self._suppress_reload or self._tree is None:
            return
        self._tree.refresh()

    def _on_command_reloaded(self, _event: CommandReloaded) -> None:
        # The editor re-registered one command; its tree label, priority glyph and
        # conflict marks may have changed. Refresh from the same event the trigger
        # subsystems reloaded on, so the tree agrees with what is now live.
        if self._tree is not None:
            self._tree.refresh()

    def _on_profile_switched(self, _event: ProfileSwitched) -> None:
        if self._tree is None:
            return
        # The profile is already switching elsewhere; we cannot veto it, but we can
        # still offer to save the open command's edits before its library is dropped.
        if self._editor is not None:
            self._editor.guard_unsaved()
        store = build_store(version_limit=self._manager.settings.commands.version_history_limit)
        if store is not None:
            self._store = store
            self._tree.set_store(store)
            if self._editor is not None:
                self._editor.set_store(store)
        # The open command belonged to the old profile's library; fall back to the
        # list so the user is not left staring at a now-cleared editor.
        self._set_screen(0)

    # -- lifecycle ----------------------------------------------------------

    def dispose(self) -> None:
        self._unsub_commands()
        self._unsub_reloaded()
        self._unsub_profile()
        if self._editor is not None:
            self._editor.stop_autosave()
        super().dispose()


register_tab("commands", CommandsTab)
