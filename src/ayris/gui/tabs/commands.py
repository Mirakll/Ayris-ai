"""Tab «Команды»: the command library — a tree on the left, the editor on the right.

Task 51 builds the left panel: the :class:`~ayris.gui.widgets.command_tree.CommandTree`
over the command, folder and trigger repositories of the active profile. The right
side is the editor of task 52; until it lands the tab shows a placeholder that names
the selected command, and :attr:`CommandTree.command_activated` already carries the
id the editor will consume.

The store is built lazily over the live database, the same shape the «Обновления» tab
uses: constructing every settings page must stay cheap, and a page the user may never
open should not touch storage. The tab bridges the tree to the rest of Ayris through
the event bus — it publishes :class:`CommandsChanged` after a change here, and rebuilds
the tree on a :class:`CommandsChanged` or :class:`ProfileSwitched` from elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSplitter

from ayris.core.config import ConfigManager
from ayris.core.events import CommandsChanged, EventBus
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


def build_store() -> CommandTreeStore | None:
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
        return CommandTreeStore(repositories, active.id)
    except Exception:
        _log.exception("не удалось построить хранилище дерева команд")
        return None


class CommandsTab(SettingsTab):
    """The «Команды» settings page (task 51 — the library tree; editor is task 52)."""

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
        self._bus = bus
        # Defaults set before any early return, so dispose() is always safe even
        # when the library could not be opened.
        self._tree: CommandTree | None = None
        self._editor: MacroEditor | None = None
        self._store: CommandTreeStore | None = None
        self._unsub_commands: Callable[[], None] = lambda: None
        self._unsub_profile: Callable[[], None] = lambda: None
        self._suppress_reload = False

        resolved = store if store is not None else build_store()

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body.addWidget(self._splitter)

        if resolved is None:
            notice = QLabel(
                "Библиотека команд недоступна: не удалось открыть профиль. "
                "Откройте вкладку позже или перезапустите Ayris."
            )
            notice.setProperty("role", "secondary")
            notice.setWordWrap(True)
            self._splitter.addWidget(notice)
            return

        self._store = resolved
        self._tree = CommandTree(resolved, theme)
        self._tree.command_activated.connect(self._on_command_activated)
        self._tree.tree_changed.connect(self._on_tree_changed)
        self._splitter.addWidget(self._tree)

        self._editor = MacroEditor(resolved, theme, services=MacroEditorServices())
        self._editor.command_saved.connect(self._on_command_saved)
        self._splitter.addWidget(self._editor)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 3)

        if bus is not None:
            self._unsub_commands = bus.subscribe(CommandsChanged, self._on_commands_changed)
            self._unsub_profile = bus.subscribe(ProfileSwitched, self._on_profile_switched)

    # -- event wiring -------------------------------------------------------

    def _on_command_activated(self, command_id: int) -> None:
        if self._editor is not None:
            self._editor.load_command(command_id)

    def _on_command_saved(self, _command_id: int) -> None:
        # A save changed the command's name, tags or triggers; the tree labels and
        # conflict marks are now stale. Refresh through the same echo guard as a tree
        # edit, so our own CommandsChanged does not bounce back into a reload.
        self._on_tree_changed()

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

    def _on_profile_switched(self, _event: ProfileSwitched) -> None:
        if self._tree is None:
            return
        store = build_store()
        if store is not None:
            self._store = store
            self._tree.set_store(store)
            if self._editor is not None:
                self._editor.set_store(store)

    # -- lifecycle ----------------------------------------------------------

    def dispose(self) -> None:
        self._unsub_commands()
        self._unsub_profile()
        super().dispose()


register_tab("commands", CommandsTab)
