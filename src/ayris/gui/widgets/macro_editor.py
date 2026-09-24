"""The list-mode command editor: the whole of task 52 assembled on one model.

:class:`MacroEditor` is the right panel of the «Команды» tab. It opens a command by id
through :class:`~ayris.gui.widgets.command_tree_model.CommandTreeStore`, edits the one
working :class:`CommandModel` every section shares — header, triggers, action list,
palette, parameter form, variables, sounds — and saves it back through the store. The
node view of task 53 will hang on the same model, so no section keeps a second copy.

Heavy work stays off the UI thread. Validation (a walk of the whole block tree through
the task-30 validator) and the «Тест» run (through the task-35 debugger) each go to a
small daemon-thread runner and come back as a queued signal, so a large command never
freezes the interface. The parameter form for a block is built from that block's schema
alone (task 33 catalog / :func:`ayris.actions.base.build_schema`), never hand-written.

What the editor cannot do without a service is disabled, not faked: no test runner and
the «Тест» button is off; no sound preview and the preview buttons are off. Those
services are injected (:class:`MacroEditorServices`) so the tests drive the editor
without an audio device or a live worker.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QCloseEvent, QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.base import ParamField
from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.debug_session import Breakpoint, DebugSessionSnapshot
from ayris.actions.macros.diff import diff_commands
from ayris.actions.macros.hot_reload import HotReloader
from ayris.actions.macros.schema import CommandModel
from ayris.actions.macros.validator import Severity, ValidationReport, validate_command
from ayris.core.errors import AyrisError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.action_list import ActionListModel, ActionListView, BlockPath
from ayris.gui.widgets.block_palette import BlockPalette
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.confirm_dialog import ConfirmDialog
from ayris.gui.widgets.draft_store import DraftStore
from ayris.gui.widgets.editor_header import EditorHeader
from ayris.gui.widgets.editor_undo import UndoStack
from ayris.gui.widgets.node_editor import NodeEditor
from ayris.gui.widgets.param_form import ParamForm
from ayris.gui.widgets.sound_binding import SoundBindingSection, SoundImporter, SoundPreview
from ayris.gui.widgets.trigger_list import HotkeyCapture, TriggerList
from ayris.gui.widgets.variable_table import VariableTable
from ayris.gui.widgets.version_history import VersionHistory
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.actions.registry import ActionRegistry
    from ayris.core.events import EventBus

__all__ = [
    "MacroEditor",
    "MacroEditorServices",
    "MacroTestResult",
    "MacroTestRunner",
    "UnsavedChoice",
]

_log = get_logger(__name__)

#: The inspector's empty state when no block is selected — a prompt, not a blank
#: panel, so the parameter panel reads as present and its purpose is clear.
_SELECT_HINT = "Выберите ноду, чтобы изменить её параметры."

#: The status badge is capped this wide while it rides the floating capsule on the node
#: canvas, so a long validation note cannot stretch the no-wrap capsule across the graph;
#: in the footer row it is uncapped (Qt's default max) and free to take the spare width.
_STATUS_MAX_NODE = 240
_STATUS_MAX_FREE = 16_777_215


@dataclass(frozen=True, slots=True)
class _TestStage:
    """One stage of a test run, as the log panel shows it."""

    path: str
    status: str
    duration_ms: float
    message: str = ""


@dataclass(frozen=True, slots=True)
class MacroTestResult:
    """The outcome of a «Тест» run: a headline and the per-stage log."""

    outcome: str
    message: str
    stages: tuple[_TestStage, ...] = ()
    dry_run: bool = False


class MacroTestRunner(Protocol):
    """Runs a command for the «Тест» button, off the UI thread.

    The editor hands over the command, the slot values to stand in for a real trigger,
    and whether to run dry (dangerous blocks described, not performed). Implemented over
    the task-35 debugger in the application; a test supplies a fake.
    """

    def run_test(
        self,
        command: CommandModel,
        *,
        slots: Mapping[str, Any],
        dry_run: bool,
    ) -> MacroTestResult: ...


@dataclass(slots=True)
class MacroEditorServices:
    """The optional collaborators the editor uses; each absent one disables a feature."""

    catalog: BlockCatalog | None = None
    hotkey_capture: HotkeyCapture | None = None
    event_names: Sequence[str] = field(default_factory=tuple)
    sound_preview: SoundPreview | None = None
    #: Brings a picked file into the profile's sounds folder for a «Файл» binding.
    #: Absent disables the «Выбрать файл…» button rather than copying files elsewhere.
    sound_importer: SoundImporter | None = None
    test_runner: MacroTestRunner | None = None
    #: The event bus. Given one, a save re-registers the command live (task 54)
    #: through :class:`~ayris.actions.macros.hot_reload.HotReloader`; without one
    #: the editor still saves, but nothing re-registers — the mode a test uses.
    bus: EventBus | None = None
    #: Passed to the validator so a save is gated on the same rules as the editor.
    registry: ActionRegistry | None = None
    #: Seconds between draft autosaves of an unsaved command; ``0`` turns it off.
    draft_autosave_s: float = 5.0
    #: A pre-built draft store; when absent one is opened under the profile cache.
    draft_store: DraftStore | None = None
    #: Which action view opens first — the node canvas or the list. Task 53's canvas
    #: is the default; the tab feeds the user's remembered choice here.
    action_view: str = "nodes"
    #: Called with the new choice (``"nodes"`` / ``"list"``) whenever the user flips
    #: the toggle, so the tab can persist it. Absent means the choice is not remembered.
    on_action_view_changed: Callable[[str], None] | None = None


class UnsavedChoice:
    """What the user picked in the unsaved-changes guard, or what a caller may force."""

    SAVE = "save"
    DISCARD = "discard"
    CANCEL = "cancel"


class _AsyncRunner(QObject):
    """Runs a blocking call off the UI thread, delivering the result as a queued signal."""

    finished = Signal(object)
    failed = Signal(str)

    def run(self, work: Callable[[], object]) -> None:
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work: Callable[[], object]) -> None:
        try:
            result = work()
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
        except Exception as exc:
            _log.exception("фоновая операция редактора команды упала")
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class MacroEditor(QWidget):
    """Full list-mode editor for one command, over a :class:`CommandTreeStore`."""

    #: Emitted after a successful save, so the tree can refresh its labels.
    command_saved = Signal(int)
    #: Emitted with the dirty flag whenever the working model changes or is saved.
    dirty_changed = Signal(bool)
    #: The open command's name and enabled flag, whenever either changes (load, header
    #: edit, save, undo/redo). The browser top bar shows them beside «← К списку команд».
    title_changed = Signal(str, bool)
    #: The «← К списку команд» crumb in the browser top bar was clicked. The host
    #: («Команды» tab) returns to the library, guarding unsaved edits first — the crumb
    #: itself only asks to leave, so the editor owns the row but not the navigation.
    back_requested = Signal()

    def __init__(
        self,
        store: CommandTreeStore,
        theme: ThemeManager,
        *,
        services: MacroEditorServices | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._theme = theme
        self._services = services if services is not None else MacroEditorServices()
        self._catalog = self._services.catalog or BlockCatalog()
        self._model: CommandModel | None = None
        self._dirty = False
        self._applying_undo = False
        # Set while an in-form param edit is driving a view rebuild. The rebuild
        # re-selects the same node and re-emits ``block_selected``; rebuilding the
        # inspector form then would delete the field widget still mid-``textChanged``
        # — a use-after-free crash. The form already shows this block, so skip it.
        self._syncing_params = False

        # Task 54 collaborators. The undo stack is model-level and shared by both
        # action views; the reloader turns a save into a live re-registration when
        # a bus is present; the draft store autosaves the unsaved command.
        self._undo = UndoStack()
        # Validate a save with the same registry the editor builds its forms from,
        # so a command that looks valid in the editor is not refused on save.
        self._reloader = (
            HotReloader(
                self._store,
                self._services.bus,
                registry=self._services.registry or self._catalog.registry,
            )
            if self._services.bus is not None
            else None
        )
        self._drafts = self._services.draft_store or _default_draft_store()

        self._validator = _AsyncRunner(self)
        self._validator.finished.connect(self._on_validation)
        self._tester = _AsyncRunner(self)
        self._tester.finished.connect(self._on_test_finished)
        self._tester.failed.connect(self._on_test_failed)

        self._draft_timer = QTimer(self)
        self._draft_timer.setSingleShot(False)
        self._draft_timer.timeout.connect(self._autosave_draft)
        if self._services.draft_autosave_s > 0:
            self._draft_timer.setInterval(int(self._services.draft_autosave_s * 1000))

        outer = QVBoxLayout(self)
        # No outer padding: the «Команды» tab already frames the editor, and the
        # node canvas should reach the page edges on every side. The tab bar's own
        # margins keep the content off the very border.
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_xs"))

        self._placeholder = QLabel("Выберите команду в дереве слева.")
        self._placeholder.setProperty("role", "secondary")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        outer.addWidget(self._placeholder)

        self._content = QWidget()
        self._content.setProperty("transparent", True)
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(theme.metric("spacing_sm"))
        outer.addWidget(self._content, 1)

        self._tabs = QTabWidget()
        # Flat, document-style tab bar: no boxed frame around the pages, so the node
        # canvas below reads as one continuous surface rather than a panel inside a panel.
        self._tabs.setDocumentMode(True)
        self._content_layout = content_layout
        content_layout.addWidget(self._tabs, 1)
        self._build_overview_tab()
        self._build_actions_tab()
        self._build_variables_tab()
        self._build_sounds_tab()
        self._build_versions_tab()
        # The stock tab strip is replaced by the browser top row built below: its
        # buttons drive the same QTabWidget, so the pages stay reachable while the row
        # also carries the «← К списку команд» crumb and the command's name.
        tab_bar = self._tabs.tabBar()
        if tab_bar is not None:
            tab_bar.hide()

        self._footer = self._build_footer()
        content_layout.addWidget(self._footer)
        self._content.hide()

        # One browser-style row above the pages (mockup's `.topbar`): crumb + name on the
        # left, the section tabs on the right. It lives in the outer layout, not inside
        # `_content`, so the «← К списку команд» crumb stays reachable even when a command
        # fails to open and the placeholder shows in place of the tabs.
        self._build_topbar()
        outer.insertWidget(0, self._topbar)
        # A hairline under the row seals it off as the mockup's `.topbar` (its
        # border-bottom), separating the chrome from the canvas that rises to meet it.
        topbar_rule = QFrame()
        topbar_rule.setProperty("rule", True)
        topbar_rule.setFrameShape(QFrame.Shape.HLine)
        outer.insertWidget(1, topbar_rule)

        # The canvas owns the footer while «Ноды» is the shown mode (task 53 default),
        # so it reaches the window's bottom edge; every other tab keeps it under the tabs.
        self._tabs.currentChanged.connect(self._sync_footer_placement)
        self._sync_footer_placement()

        # Ctrl+Z / Ctrl+Shift+Z on the editor — active in both action modes, since
        # the stack is over the model, not either view.
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)
        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)

    # -- tabs ---------------------------------------------------------------

    def _build_overview_tab(self) -> None:
        tags = sorted({tag for command in self._store.commands() for tag in command.tags})
        self._header = EditorHeader(self._theme, tag_suggestions=tags)
        self._header.changed.connect(self._on_model_changed)

        self._triggers = TriggerList(
            self._theme,
            hotkey_capture=self._services.hotkey_capture,
            event_names=self._services.event_names,
        )
        self._triggers.changed.connect(self._on_triggers_changed)

        page = _ScrollPage(self._theme)
        page.add(_section("Команда", self._header, self._theme))
        page.add(_section("Триггеры", self._triggers, self._theme))
        self._tabs.addTab(page, "Обзор")

    def _build_actions_tab(self) -> None:
        self._action_model = ActionListModel()
        self._action_view = ActionListView(self._action_model, self._theme, catalog=self._catalog)
        self._action_view.changed.connect(self._on_actions_changed)
        self._action_view.block_selected.connect(self._on_block_selected)

        # The node editor of task 53 is a second view over the SAME model, so switching
        # «Список ↔ Ноды» keeps every unsaved edit — there is no second copy of the tree.
        self._node_editor = NodeEditor(self._action_model, self._theme, catalog=self._catalog)
        self._node_editor.changed.connect(self._on_actions_changed)
        self._node_editor.block_selected.connect(self._on_block_selected)
        # A breakpoint toggled on the canvas is persisted to the debugger session
        # store, where the MacroDebugger reads it from when a debug run starts.
        self._node_editor.breakpoints_changed.connect(self._persist_breakpoints)

        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._action_view)  # index 0 — список
        self._mode_stack.addWidget(self._node_editor)  # index 1 — ноды

        self._list_button = QPushButton("Список")
        self._list_button.setCheckable(True)
        self._nodes_button = QPushButton("Ноды")
        self._nodes_button.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self._list_button)
        mode_group.addButton(self._nodes_button)
        # The remembered choice (task 53's node canvas by default) picks the opening
        # view; set the stack and toggle directly so this initial pick is not counted
        # as a user flip and re-persisted.
        nodes_first = self._services.action_view != "list"
        self._mode_stack.setCurrentIndex(1 if nodes_first else 0)
        self._nodes_button.setChecked(nodes_first)
        self._list_button.setChecked(not nodes_first)
        self._list_button.clicked.connect(lambda: self._on_mode_clicked(0))
        self._nodes_button.clicked.connect(lambda: self._on_mode_clicked(1))
        # «Список ↔ Ноды» as one segmented pill (a themed «островок», the same toolgroup
        # frame the history segment uses). It is placed in the browser-style top row
        # (:meth:`_build_topbar`), just left of the section tabs, so the mode switch shares
        # that one row instead of standing on its own strip above the canvas — one row less
        # of chrome, so the canvas rises. Shown only while the «Действия» tab is current
        # (toggled in :meth:`_sync_footer_placement`).
        self._list_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._nodes_button.setCursor(Qt.CursorShape.PointingHandCursor)
        segmented = QFrame()
        segmented.setProperty("toolgroup", True)
        segmented_layout = QHBoxLayout(segmented)
        segmented_layout.setContentsMargins(2, 2, 2, 2)
        segmented_layout.setSpacing(2)
        segmented_layout.addWidget(self._list_button)
        segmented_layout.addWidget(self._nodes_button)
        self._segmented = segmented

        # Undo/redo, shared across list and node modes. The middle button drops a
        # menu of the recent operations — task 54's «выпадающий список последних».
        # The three live inside one themed «островок» (QFrame[toolgroup]); on the node
        # canvas the island is mounted into the LEFT slot of the bottom command capsule
        # («вперёд/назад … как на фото»), and in every other view it sits at the left of
        # the footer row under the tabs.
        square = self._theme.metric("control_height")
        self._undo_button = QToolButton()
        self._undo_button.setText("↶")
        self._undo_button.setFixedWidth(square)
        self._undo_button.setToolTip("Отменить (Ctrl+Z)")
        self._undo_button.clicked.connect(self.undo)
        self._history_button = QToolButton()
        self._history_button.setText("История ▾")
        self._history_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._history_menu = QMenu(self._history_button)
        self._history_menu.aboutToShow.connect(self._fill_history_menu)
        self._history_button.setMenu(self._history_menu)
        self._redo_button = QToolButton()
        self._redo_button.setText("↷")
        self._redo_button.setFixedWidth(square)
        self._redo_button.setToolTip("Повторить (Ctrl+Shift+Z)")
        self._redo_button.clicked.connect(self.redo)

        history_group = QFrame()
        history_group.setProperty("toolgroup", True)
        group_layout = QHBoxLayout(history_group)
        group_layout.setContentsMargins(2, 2, 2, 2)
        group_layout.setSpacing(2)
        for button in (self._undo_button, self._history_button, self._redo_button):
            button.setAutoRaise(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            group_layout.addWidget(button)
        self._history_group = history_group
        self._update_undo_buttons()

        self._palette = BlockPalette(self._theme, catalog=self._catalog)
        self._palette.block_chosen.connect(self._on_palette_choice)

        # The inspector is a solid card (mockup's `.inspector`), not a transparent
        # strip: with nothing selected it used to be invisible floating text on the
        # canvas grid — you could not tell the panel was there. A `surface` fill with
        # a border reads as a panel in every state; a titled header row over a rule
        # (`.insp-head`) sits above the schema-built form.
        param_panel = QWidget()
        param_panel.setProperty("card", True)
        param_layout = QVBoxLayout(param_panel)
        pad = self._theme.metric("spacing_md")
        param_layout.setContentsMargins(pad, pad, pad, pad)
        param_layout.setSpacing(self._theme.metric("spacing_sm"))

        self._param_title = QLabel("Параметры")
        self._param_title.setProperty("role", "h2")
        param_layout.addWidget(self._param_title)
        header_rule = QFrame()
        header_rule.setProperty("rule", True)
        header_rule.setFrameShape(QFrame.Shape.HLine)
        param_layout.addWidget(header_rule)

        self._param_form = ParamForm(self._theme)
        self._param_form.changed.connect(self._on_params_changed)
        param_layout.addWidget(self._param_form, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._palette)
        splitter.addWidget(self._mode_stack)
        splitter.addWidget(param_panel)
        # The canvas is the point of this tab, so it takes every spare pixel while the
        # palette and parameter panels stay narrow (stretch 0) and are capped so they
        # never grow at the canvas's expense on a wide window. Both side panels may be
        # dragged shut to give the graph the whole width; the canvas never collapses.
        self._palette.setMinimumWidth(140)
        self._palette.setMaximumWidth(240)
        param_panel.setMinimumWidth(150)
        param_panel.setMaximumWidth(280)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, False)
        splitter.setCollapsible(2, True)
        splitter.setSizes([150, 1200, 170])
        self._actions_page = splitter
        self._tabs.addTab(splitter, "Действия")
        # `_segmented` is placed in the browser top row (:meth:`_build_topbar`), just left
        # of the section tabs; its visibility follows the current tab
        # (:meth:`_sync_footer_placement`), so it appears only over «Действия».

    def _on_mode_clicked(self, target: int) -> None:
        """Handle a click on the «Список»/«Ноды» segment.

        The node canvas is the home view (task 53). A first click on «Список» opens the
        list; a second click on the already-active «Список» collapses back to the nodes,
        so the toggle reads as open/close rather than a dead press. Re-clicking the
        active «Ноды» stays put — the home view has nowhere to fall back to.
        """
        current = self._mode_stack.currentIndex()
        new_index = 1 if target == current else target
        self._list_button.setChecked(new_index == 0)
        self._nodes_button.setChecked(new_index == 1)
        if new_index != current:
            self._set_action_mode(new_index)

    def _set_action_mode(self, index: int) -> None:
        """Switch «Список ↔ Ноды», carrying the selection and rebuilding the shown view.

        A user flip is remembered through the injected callback, so the next command
        and the next launch open in the same view (task 53's canvas by default).
        """
        selected = self._current_action_view().selected_path()
        self._mode_stack.setCurrentIndex(index)
        view = self._current_action_view()
        view.rebuild()
        if selected:
            view.select_path(selected)
        self._sync_footer_placement()
        callback = self._services.on_action_view_changed
        if callback is not None:
            callback("nodes" if index == 1 else "list")

    def _sync_footer_placement(self, *_args: object) -> None:
        """Fold the bottom bar into the canvas capsule in «Ноды», else keep it below.

        On the node canvas the history island and the статус · Тест · Сохранить group are
        mounted into the floating «Кинематограф» capsule, so the whole bottom bar reads as
        one pult and the canvas reaches the window edge; on every other tab and in the list
        view they sit under the tabs as an ordinary footer row. The same widgets move either
        way, so status text and button states carry across untouched. It also keeps the top
        row's tab buttons in step with the current page and shows the «Список / Ноды» mode
        pill on «Действия» alone.
        """
        if not hasattr(self, "_footer"):
            return
        current = self._tabs.currentIndex()
        if hasattr(self, "_tab_buttons"):
            for index, button in enumerate(self._tab_buttons):
                button.setChecked(index == current)
        on_actions = self._tabs.currentWidget() is self._actions_page
        if hasattr(self, "_segmented"):
            self._segmented.setVisible(on_actions)
        on_nodes = on_actions and self._mode_stack.currentIndex() == 1
        if on_nodes:
            self._status.setMaximumWidth(_STATUS_MAX_NODE)
            self._node_editor.mount_controls(self._history_group, self._footer_actions)
            self._footer.hide()
        else:
            self._node_editor.mount_controls(None, None)
            self._status.setMaximumWidth(_STATUS_MAX_FREE)
            self._layout_footer_row()
            self._footer.show()

    def _current_action_view(self) -> ActionListView | NodeEditor:
        if self._mode_stack.currentIndex() == 1:
            return self._node_editor
        return self._action_view

    def _rebuild_action_views(self) -> None:
        """Redraw whichever action view is showing after a model change."""
        self._current_action_view().rebuild()

    def _build_variables_tab(self) -> None:
        self._variables = VariableTable(self._theme)
        self._variables.changed.connect(self._on_variables_changed)
        page = _ScrollPage(self._theme)
        page.add(_section("Переменные", self._variables, self._theme))
        self._tabs.addTab(page, "Переменные")

    def _build_sounds_tab(self) -> None:
        self._sounds = SoundBindingSection(
            self._theme,
            preview=self._services.sound_preview,
            importer=self._services.sound_importer,
        )
        self._sounds.changed.connect(self._on_model_changed)
        page = _ScrollPage(self._theme)
        page.add(_section("Звуки стадий", self._sounds, self._theme))
        self._tabs.addTab(page, "Звуки")

    def _build_versions_tab(self) -> None:
        self._versions = VersionHistory(self._store, self._theme)
        self._versions.rollback_requested.connect(self._on_rollback_requested)
        self._versions.status.connect(self._on_version_status)
        self._tabs.addTab(self._versions, "История")

    def _build_topbar(self) -> None:
        """The browser-style top row (mockup's `.topbar`): one line of chrome over the pages.

        Left to right: the «← К списку команд» crumb (a flat link that only emits
        :attr:`back_requested`), a muted «/», the open command's name and an «включена /
        выключена» pill, then a stretch, then the «Список / Ноды» mode pill (shown only on
        «Действия») and the five section tabs pushed hard to the right. The tab buttons
        drive the same hidden :class:`QTabWidget`, so they replace its stock strip while
        folding the old separate crumb row into this single line — the canvas rises to it.
        """
        bar = QWidget()
        bar.setProperty("browserTopbar", True)
        bar.setProperty("transparent", True)
        layout = QHBoxLayout(bar)
        pad = self._theme.metric("spacing_xs")
        layout.setContentsMargins(self._theme.metric("spacing_sm"), pad, pad, pad)
        layout.setSpacing(self._theme.metric("spacing_sm"))

        self._back_button = QPushButton("← К списку команд")
        self._back_button.setProperty("link", True)
        self._back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._back_button.clicked.connect(lambda: self.back_requested.emit())
        layout.addWidget(self._back_button)

        self._title_sep = QLabel("/")
        self._title_sep.setProperty("role", "secondary")
        self._title_name = QLabel("")
        self._title_name.setProperty("role", "h3")
        self._title_badge = QLabel("")
        for label in (self._title_sep, self._title_name, self._title_badge):
            layout.addWidget(label)
            label.hide()

        layout.addStretch(1)

        # The mode pill sits just left of the tabs so, when it hides off «Действия», the
        # stretch keeps the tabs flush right with no sideways jump.
        layout.addWidget(self._segmented)

        self._tab_buttons: list[QPushButton] = []
        group = QButtonGroup(self)
        group.setExclusive(True)
        for index in range(self._tabs.count()):
            button = QPushButton(self._tabs.tabText(index))
            button.setProperty("navTab", True)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, i=index: self._tabs.setCurrentIndex(i))
            group.addButton(button)
            layout.addWidget(button)
            self._tab_buttons.append(button)

        self.title_changed.connect(self._update_title)
        self._topbar = bar

    def _update_title(self, name: str, enabled: bool) -> None:
        """Reflect the open command's name and enabled state in the browser top row.

        With no command open the «/», name and pill stay hidden, leaving just the crumb —
        a lone «/» with an empty name would read as chrome noise.
        """
        has_name = bool(name)
        for label in (self._title_sep, self._title_name, self._title_badge):
            label.setVisible(has_name)
        self._title_name.setText(name)
        self._title_badge.setText("включена" if enabled else "выключена")
        state = "on" if enabled else "off"
        if self._title_badge.property("statePill") != state:
            self._title_badge.setProperty("statePill", state)
            style = self._title_badge.style()
            if style is not None:
                style.unpolish(self._title_badge)
                style.polish(self._title_badge)

    def _build_footer(self) -> QWidget:
        # The validation note is a compact badge now, not a full-width strip that ate a
        # band of the canvas (the user's «убрал бы эту полосу … сделай меньше»). It is
        # transparent when empty, so no empty pill floats, and coloured by severity in
        # :meth:`_on_validation`. The text it shows is unchanged — only its chrome shrank.
        self._status = _StatusLabel()
        self._status.setProperty("badge", "muted")
        self._status.setWordWrap(False)

        self._test_button = QPushButton("Тест")
        self._test_button.setProperty("textButton", True)
        self._test_button.setEnabled(self._services.test_runner is not None)
        self._test_button.clicked.connect(self._on_test)
        self._save_button = QPushButton("Сохранить")
        self._save_button.setProperty("kind", "primary")
        self._save_button.setProperty("textButton", True)
        self._save_button.clicked.connect(self.save)

        # статус · Тест · Сохранить as one movable group. On the node canvas it is mounted
        # into the RIGHT slot of the floating capsule (with the history island on the left,
        # «как на фото»); in every other view it sits at the right of the footer row. Kept
        # as its own widget so the whole group reparents in a single move.
        actions = QWidget()
        actions.setProperty("transparent", True)
        actions_layout = QHBoxLayout(actions)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        actions_layout.setSpacing(self._theme.metric("spacing_sm"))
        actions_layout.addWidget(self._status, 1)
        actions_layout.addWidget(self._test_button)
        actions_layout.addWidget(self._save_button)
        self._footer_actions = actions

        footer = QWidget()
        footer.setProperty("transparent", True)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(self._theme.metric("spacing_sm"))
        self._footer_layout = layout
        self._layout_footer_row()

        # The test log is kept as an off-screen sink so test output still has somewhere
        # to land, but it no longer eats the editor's vertical space — the headline
        # outcome shows in the status line. Parented and hidden, never added to a layout.
        self._log = QPlainTextEdit(self)
        self._log.setReadOnly(True)
        self._log.hide()
        return footer

    def _layout_footer_row(self) -> None:
        """Assemble the under-tabs footer row: history island · left, actions · right.

        Also used to pull both groups back out of the floating capsule when leaving the
        node canvas — each is first detached from its current parent layout (the capsule
        host) before being re-added here, mirroring the reparent order that avoids the
        editor's known teardown crashes.
        """
        for widget in (self._history_group, self._footer_actions):
            parent = widget.parentWidget()
            parent_layout = parent.layout() if parent is not None else None
            if parent_layout is not None:
                parent_layout.removeWidget(widget)
        self._footer_layout.addWidget(self._history_group)
        self._footer_layout.addWidget(self._footer_actions, 1)
        self._history_group.show()
        self._footer_actions.show()

    # -- loading / saving ---------------------------------------------------

    def load_command(self, command_id: int) -> None:
        """Open a command by id, filling every section from its model.

        A draft left by an earlier session (a crash, a «потом допишу») is offered
        for restore before the model is shown; declining it, or none existing, opens
        the saved model. The undo history and the version list are reset to the
        command being opened — task 54 requires the stack cleared on command switch —
        and the draft autosave timer is (re)started for the open command.
        """
        try:
            model = self._store.command_model(command_id)
        except AyrisError as exc:
            self._show_placeholder(exc.user_message)
            return
        model, restored = self._restore_draft(command_id, model)
        self._model = model
        self._apply_model(model, sibling_names=self._store.sibling_names(command_id))
        self._node_editor.reset_layout()
        self._load_breakpoints(command_id)
        self._node_editor.rebuild()
        self._triggers.set_conflicts(self._store.trigger_conflicts(command_id))
        self._param_form.set_schema((), empty_text=_SELECT_HINT)
        self._param_title.setText("Параметры")
        self._content.show()
        self._placeholder.hide()
        # Open straight on «Действия» — the action editor (node canvas by default) is
        # what a user opens a command to see, not the name/trigger overview.
        self._tabs.setCurrentWidget(self._actions_page)
        self._undo.reset(model)
        self._update_undo_buttons()
        self._versions.load(command_id, model)
        self._set_dirty(restored)
        if restored:
            self._set_status("Восстановлен несохранённый черновик.", "info")
        else:
            self._set_status("", "muted")
        self._log.clear()
        self._restart_draft_timer()
        self._schedule_validation()

    def save(self) -> bool:
        """Validate the name, then persist the whole model — live-reloading it if wired.

        The order is task 54's: name check, then validate-and-write. With a bus the
        save goes through :class:`~ayris.actions.macros.hot_reload.HotReloader`, which
        validates, writes row + triggers + version in one transaction, then republishes
        so the trigger subsystems re-register and the tree updates. Any failure — an
        invalid command, a write that raises — leaves the previously registered version
        working: nothing is published, the command is not disabled, and the editor keeps
        the unsaved model so the user can fix and retry.

        Returns ``True`` when the command was written, ``False`` when the save was
        refused (an empty/duplicate name, or a block that fails validation). A refused
        save keeps the command dirty on purpose — the mark stays lit because the edit
        genuinely is not saved — and :meth:`_report_save_failure` makes the reason
        visible instead of leaving «Сохранить» looking inert.
        """
        if self._model is None:
            return False
        if not self._header.is_name_valid():
            self._set_status("Исправьте имя команды: оно пустое или уже занято.", "error")
            self._tabs.setCurrentIndex(0)
            return False
        try:
            if self._reloader is not None:
                saved = self._reloader.apply_command(self._model).command
            else:
                saved = self._store.save_command(self._model)
        except AyrisError as exc:
            self._report_save_failure(exc)
            return False
        self._model = saved
        self._apply_model(saved, sibling_names=self._siblings_of(saved))
        # The saved model is the new baseline; a later undo must not walk back to a
        # pre-save snapshot that still carried no id.
        self._undo.reset(saved)
        self._update_undo_buttons()
        self._set_dirty(False)
        self._set_status("Команда сохранена.", "success")
        if saved.id is not None:
            self._drafts.discard(saved.id)
            self._versions.load(saved.id, saved)
            # A command saved for the first time now has an id: flush any breakpoints
            # set while it was still unsaved (they could not be persisted before).
            self._persist_breakpoints()
            self.command_saved.emit(saved.id)
        return True

    def _report_save_failure(self, exc: AyrisError) -> None:
        """Surface a refused save and take the user to the block that blocked it.

        A validation error otherwise hides in the footer status line: pressing
        «Сохранить» looks like it did nothing, the unsaved mark stays lit, and the
        exit guard keeps re-prompting — which reads as a stuck dirty flag rather than
        «этот блок не заполнен». So we jump to «Действия» and select the first block
        the validator rejected, turning an invisible refusal into a pointed «вот что
        не так». The command stays dirty (its old version is still what runs live).
        """
        self._set_status(exc.user_message, "error")
        report = getattr(exc, "report", None)
        errors = report.errors if report is not None else ()
        if not errors:
            return
        self._tabs.setCurrentWidget(self._actions_page)
        target = errors[0].path
        for row in self._action_model.rows():
            if row.path_text == target:
                self._current_action_view().select_path(row.path)
                break

    def set_store(self, store: CommandTreeStore) -> None:
        """Point the editor at a new profile's library and clear the open command."""
        self._store = store
        self._model = None
        self._draft_timer.stop()
        self._undo.reset(None)
        self._update_undo_buttons()
        self._versions.set_store(store)
        self._show_placeholder("Выберите команду в дереве слева.")

    @property
    def command_id(self) -> int | None:
        return None if self._model is None else self._model.id

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    # -- change handling ----------------------------------------------------

    def _on_model_changed(self) -> None:
        self._set_dirty(True)
        self._record_undo()
        self._schedule_validation()
        if self._model is not None:
            # A header edit may have renamed the command or flipped «включена»; keep the
            # screen title beside «← К списку команд» in step with it.
            self.title_changed.emit(self._model.name, self._model.enabled)

    def _on_triggers_changed(self) -> None:
        self._set_dirty(True)
        self._record_undo()
        if self._model is not None and self._model.id is not None:
            self._triggers.set_conflicts(self._store.trigger_conflicts(self._model.id))
        self._refresh_completions()
        self._schedule_validation()

    def _on_actions_changed(self) -> None:
        # A node-canvas connect/disconnect rebuilds the command through the model and
        # REPLACES the action model's command object (NodeScene.request_connect /
        # disconnect_edge call ``set_command(rebuilt)``). ``self._model`` still points at
        # the pre-edit object, so without re-syncing here the edit is orphaned: save,
        # undo and validation would all run on the stale model and the canvas change is
        # silently lost. Re-adopt the action model's current command as the source of
        # truth. (List-view edits mutate in place, so this is a no-op for them.)
        current = self._action_model.command
        if current is not None:
            self._model = current
        self._set_dirty(True)
        self._record_undo()
        self._schedule_validation()

    # -- breakpoints --------------------------------------------------------

    def _load_breakpoints(self, command_id: int) -> None:
        """Show the command's saved breakpoints on the node canvas, if any."""
        try:
            snapshot = self._store.debug_store.load(command_id)
        except Exception:
            _log.exception("не удалось загрузить точки останова команды %s", command_id)
            self._node_editor.set_breakpoints(set())
            return
        paths = {bp.path for bp in snapshot.breakpoints} if snapshot is not None else set()
        self._node_editor.set_breakpoints(paths)

    def _persist_breakpoints(self) -> None:
        """Save the canvas breakpoints to the debugger session store.

        Keyed by the saved command's id, so a debug run's
        :class:`~ayris.actions.macros.debugger.MacroDebugger` restores them. Only
        the breakpoint set is rewritten: any watches, slot overrides, conditions or
        last report the store already holds are preserved. An unsaved command
        (no id) keeps its breakpoints on the canvas until the first save flushes them.
        """
        command_id = self.command_id
        if command_id is None:
            return
        paths = self._node_editor.breakpoints()
        try:
            existing = self._store.debug_store.load(command_id)
            prior = {bp.path: bp for bp in existing.breakpoints} if existing else {}
            self._store.debug_store.save(
                DebugSessionSnapshot(
                    command_id=command_id,
                    breakpoints=[prior.get(path, Breakpoint(path)) for path in sorted(paths)],
                    watches=list(existing.watches) if existing else [],
                    slots_override=dict(existing.slots_override) if existing else {},
                    report=existing.report if existing else None,
                )
            )
        except Exception:
            _log.exception("не удалось сохранить точки останова команды %s", command_id)

    def _on_variables_changed(self) -> None:
        self._set_dirty(True)
        self._record_undo()
        self._refresh_completions()
        self._schedule_validation()

    def _on_params_changed(self) -> None:
        block = self._selected_block()
        if block is not None:
            block.params = self._param_form.values()
            self._set_dirty(True)
            self._record_undo()
            self._syncing_params = True
            try:
                self._current_action_view().rebuild()
            finally:
                self._syncing_params = False
            self._schedule_validation()

    def _on_palette_choice(self, block_type: str) -> None:
        path = self._insertion_path()
        new_path = self._action_model.insert_type(block_type, path[0], path[1])
        if new_path is None:
            # Refused — the insertion point is already at the depth ceiling.
            self._set_status("Слишком глубокая вложенность блоков.", "error")
            return
        view = self._current_action_view()
        if view is self._node_editor:
            # On the node canvas a block is added FREE — no auto-wire. The flow is the wires
            # the user drags, not list order, so a palette block drops in detached like the
            # canvas's own «＋ Нода». In the list view order *is* the flow, so it stays wired.
            new_block = self._action_model.block_at(new_path)
            if new_block is not None:
                new_block.detached = True
        view.rebuild()
        view.select_path(new_path)
        self._on_actions_changed()

    def _insertion_path(self) -> tuple[BlockPath, int]:
        """Where a palette double-click inserts: after the selection, else at the end."""
        selected = self._current_action_view().selected_path()
        if selected:
            container, index = selected[:-1], selected[-1]
            assert isinstance(index, int)
            return container, index + 1
        root: BlockPath = ("actions",)
        return root, len(self._model.actions) if self._model is not None else 0

    def _on_block_selected(self, path: tuple[object, ...]) -> None:
        if self._syncing_params:
            # The rebuild triggered by an in-form edit re-selects this very block and
            # lands here mid-``textChanged``; rebuilding the form now would delete the
            # field widget the user is typing into. It already shows this block — leave
            # it be. See ``_on_params_changed`` and [[node-scene-rebuild-use-after-free]].
            return
        block = self._action_model.block_at(path) if path else None
        if block is None:
            self._param_form.set_schema((), empty_text=_SELECT_HINT)
            self._param_title.setText("Параметры")
            return
        meta = self._catalog.try_get(block.type)
        title = meta.title_ru if meta is not None else block.type
        self._param_title.setText(f"Параметры — {title}")
        self._param_form.set_schema(self._fields_for(block.type))
        self._param_form.set_values(block.params)
        self._refresh_completions()

    def _fields_for(self, block_type: str) -> Sequence[ParamField]:
        meta = self._catalog.try_get(block_type)
        return meta.fields if meta is not None else ()

    def _selected_block(self) -> Any:
        return self._action_model.block_at(self._current_action_view().selected_path())

    def _refresh_completions(self) -> None:
        if self._model is None:
            return
        names = list(self._variables.declared_names())
        names.extend(self._model.slot_names)
        self._param_form.set_completions(names)

    # -- validation ---------------------------------------------------------

    def _schedule_validation(self) -> None:
        if self._model is None:
            return
        snapshot = self._model.model_copy(deep=True)
        registry = self._catalog.registry
        self._validator.run(lambda: validate_command(snapshot, registry=registry))

    def _on_validation(self, report: object) -> None:
        if not isinstance(report, ValidationReport):
            return
        declared = {v.name for v in self._model.variables} if self._model else set()
        used = _referenced_names(self._model) if self._model else set()
        self._variables.set_diagnostics(
            unused=declared - used,
            undeclared=used - declared,
        )
        errors = sum(1 for problem in report.problems if problem.severity is Severity.ERROR)
        warnings = sum(1 for problem in report.problems if problem.severity is Severity.WARNING)
        if errors:
            self._set_status(f"Ошибок: {errors}, предупреждений: {warnings}.", "error")
        elif warnings:
            self._set_status(f"Предупреждений: {warnings}.", "warning")
        else:
            self._set_status("Проверка пройдена.", "success")

    # -- test run -----------------------------------------------------------

    def _on_test(self) -> None:
        runner = self._services.test_runner
        if runner is None or self._model is None:
            return
        if not self._header.is_name_valid():
            self._set_status("Исправьте имя команды перед тестом.", "error")
            return
        try:
            self._store.save_command(self._model)
        except AyrisError as exc:
            self._set_status(exc.user_message, "error")
            return
        self._set_dirty(False)
        snapshot = self._model.model_copy(deep=True)
        slots = {name: f"<{name}>" for name in snapshot.slot_names}
        dry_run = any(self._is_dangerous(block.block.type) for block in snapshot.blocks())
        self._set_status("Тест запущен…", "info")
        self._log.clear()
        self._test_button.setEnabled(False)
        self._tester.run(lambda: runner.run_test(snapshot, slots=slots, dry_run=dry_run))

    def _is_dangerous(self, block_type: str) -> bool:
        meta = self._catalog.try_get(block_type)
        return bool(meta is not None and meta.is_dangerous)

    def _on_test_finished(self, result: object) -> None:
        self._test_button.setEnabled(self._services.test_runner is not None)
        if not isinstance(result, MacroTestResult):
            return
        lines = [
            f"{_status_glyph(stage.status)} {stage.path} · {stage.duration_ms:.0f} мс"
            + (f" — {stage.message}" if stage.message else "")
            for stage in result.stages
        ]
        self._log.setPlainText("\n".join(lines))
        prefix = "Сухой прогон. " if result.dry_run else ""
        kind = {
            "ok": "success",
            "failed": "error",
            "timeout": "error",
            "error": "error",
            "cancelled": "muted",
        }.get(result.outcome, "info")
        self._set_status(f"{prefix}{result.message}", kind)

    def _on_test_failed(self, message: str) -> None:
        self._test_button.setEnabled(self._services.test_runner is not None)
        self._set_status(f"Тест не выполнен: {message}", "error")

    # -- helpers ------------------------------------------------------------

    def _set_status(self, message: str, kind: str = "muted") -> None:
        """Set the status badge's text and severity colour, re-polishing so it takes.

        Routing every status line through one setter keeps the badge's colour in step
        with its text — a fresh «Команда сохранена.» is never left tinted red from a
        prior error, and an empty note reads as muted. The message is shown verbatim.
        """
        self._status.setText(message)
        if self._status.property("badge") != kind:
            self._status.setProperty("badge", kind)
            style = self._status.style()
            if style is not None:
                style.unpolish(self._status)
                style.polish(self._status)

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)

    def _show_placeholder(self, message: str) -> None:
        self._placeholder.setText(message)
        self._placeholder.show()
        self._content.hide()

    def _apply_model(self, model: CommandModel, *, sibling_names: set[str]) -> None:
        """Push *model* into every section, without touching dirty/undo/version state.

        The one place that fills the header, triggers, action list, variables and
        sounds from a model — used by load, save (with the read-back model), undo/redo
        and rollback, so no section keeps a second copy and every path shows the same
        thing.
        """
        self._header.set_command(model, sibling_names=sibling_names)
        self._triggers.set_command(model)
        self._action_model.set_command(model)
        self._current_action_view().rebuild()
        self._variables.set_command(model)
        self._sounds.set_command(model)
        self._refresh_completions()
        self.title_changed.emit(model.name, model.enabled)

    def _siblings_of(self, model: CommandModel) -> set[str]:
        """Sibling names for the header's duplicate check; empty for an id-less model."""
        return self._store.sibling_names(model.id) if model.id is not None else set()

    # -- undo / redo (task 54) ----------------------------------------------

    def undo(self) -> None:
        """Step the model back one operation and show it in every section."""
        restored = self._undo.undo()
        if restored is not None:
            self._apply_undo_model(restored)

    def redo(self) -> None:
        """Step the model forward one operation and show it in every section."""
        restored = self._undo.redo()
        if restored is not None:
            self._apply_undo_model(restored)

    def _apply_undo_model(self, model: CommandModel) -> None:
        """Adopt a model returned by the undo stack, guarding against re-recording.

        The section widgets emit ``changed`` as they are refilled; the guard stops
        those from pushing the undo itself back onto the stack as a fresh edit.
        """
        self._applying_undo = True
        try:
            self._model = model
            self._apply_model(model, sibling_names=self._siblings_of(model))
        finally:
            self._applying_undo = False
        self._set_dirty(True)
        self._update_undo_buttons()
        self._schedule_validation()

    def _record_undo(self) -> None:
        """Snapshot the current model onto the undo stack after an edit."""
        if self._applying_undo or self._model is None:
            return
        if self._undo.record(self._model):
            self._update_undo_buttons()

    def _update_undo_buttons(self) -> None:
        can_undo = self._undo.can_undo
        can_redo = self._undo.can_redo
        self._undo_button.setEnabled(can_undo)
        self._redo_button.setEnabled(can_redo)
        self._history_button.setEnabled(can_undo or can_redo)
        undo_desc = self._undo.undo_description()
        redo_desc = self._undo.redo_description()
        self._undo_button.setToolTip(
            f"Отменить: {undo_desc} (Ctrl+Z)" if undo_desc else "Отменить (Ctrl+Z)"
        )
        self._redo_button.setToolTip(
            f"Повторить: {redo_desc} (Ctrl+Shift+Z)" if redo_desc else "Повторить (Ctrl+Shift+Z)"
        )

    def _fill_history_menu(self) -> None:
        """Build the «последние операции» dropdown: newest step first, undo to it."""
        self._history_menu.clear()
        descriptions = self._undo.descriptions()
        if not descriptions:
            action = self._history_menu.addAction("Нет операций")
            action.setEnabled(False)
            return
        # Newest first; choosing the Nth undoes back through it (N+1 steps from top).
        for offset, description in enumerate(reversed(descriptions)):
            steps = offset + 1
            action = self._history_menu.addAction(description)
            action.triggered.connect(lambda _checked=False, n=steps: self._undo_steps(n))

    def _undo_steps(self, count: int) -> None:
        for _ in range(count):
            if not self._undo.can_undo:
                break
            restored = self._undo.undo()
            if restored is None:
                break
            self._apply_undo_model(restored)

    # -- drafts (task 54) ---------------------------------------------------

    def _restore_draft(self, command_id: int, model: CommandModel) -> tuple[CommandModel, bool]:
        """Offer a leftover draft for the command; return the model to open and if restored.

        Returns the draft's model (carrying the live id) when the user restores it,
        otherwise the saved model with the draft discarded. A draft that does not load
        is dropped inside the store and treated as absent.
        """
        record = self._drafts.load(command_id)
        if record is None:
            return model, False
        dialog = ConfirmDialog(
            "Черновик команды",
            "Для этой команды остался несохранённый черновик от "
            f"{record.saved_at.astimezone():%d.%m.%Y %H:%M}. Восстановить его?",
            self._theme,
            confirm_text="Восстановить",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self._drafts.discard(command_id)
            return model, False
        return record.command.model_copy(update={"id": command_id}), True

    def _autosave_draft(self) -> None:
        """Timer tick: write the working command as a draft while it is unsaved."""
        if self._model is not None and self._dirty:
            self._drafts.save(self._model)

    def _restart_draft_timer(self) -> None:
        self._draft_timer.stop()
        if self._services.draft_autosave_s > 0:
            self._draft_timer.start()

    # -- version history (task 54) ------------------------------------------

    def _on_rollback_requested(self, model: object) -> None:
        """Apply a rolled-back version as a normal save, so it becomes a new version.

        The version widget has already confirmed the rollback through the task-42
        dialog and rebuilt the model with the live id. Adopting it, recording an undo
        step and saving keeps the current state in history (the save versions it first)
        and re-registers the command through the same path any save uses.
        """
        if not isinstance(model, CommandModel) or self._model is None:
            return
        self._model = model
        self._apply_model(model, sibling_names=self._siblings_of(model))
        self._record_undo()
        self._set_dirty(True)
        self.save()

    def _on_version_status(self, message: str) -> None:
        self._set_status(message, "info")

    # -- unsaved-changes guard (task 54) ------------------------------------

    def guard_unsaved(self) -> bool:
        """Ask about unsaved edits before leaving the command. ``True`` to proceed.

        Returns ``True`` at once when there is nothing to lose. Otherwise prompts
        «Сохранить / Не сохранять / Отмена»; a chosen save that fails (an invalid
        command) keeps the user on the command by returning ``False``.
        """
        if not self._dirty or self._model is None:
            return True
        return self.prompt_unsaved() != UnsavedChoice.CANCEL

    def prompt_unsaved(self) -> str:
        """Run the unsaved-changes dialog and act on the choice; return what was chosen."""
        dialog = _UnsavedDialog(self._unsaved_summary(), self._theme, parent=self)
        dialog.exec()
        choice = dialog.choice
        if choice == UnsavedChoice.SAVE and not self.save():
            # The save was refused (an invalid name or a block that fails validation);
            # _report_save_failure has shown why and jumped to the offending block, so
            # keep the user on the command rather than leaving with unsaved edits.
            return UnsavedChoice.CANCEL
        return choice

    def _unsaved_summary(self) -> str:
        """A short line naming what changed since the command was last saved."""
        if self._model is None:
            return ""
        if self._model.id is None:
            return "Новая команда ещё не сохранена."
        try:
            saved = self._store.command_model(self._model.id)
        except AyrisError:
            return "В команде есть несохранённые изменения."
        result = diff_commands(saved, self._model)
        if result.is_empty:
            return "В команде есть несохранённые изменения."
        return f"Несохранённые изменения: {result.summary()}."

    # -- lifecycle ----------------------------------------------------------

    def stop_autosave(self) -> None:
        """Stop the draft autosave timer. The tests call this; so does closing."""
        self._draft_timer.stop()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override
        """Stop the autosave timer so it cannot outlive the widget and hang a run."""
        self._draft_timer.stop()
        super().closeEvent(event)


class _StatusLabel(QLabel):
    """One-line status note that elides with «…» instead of being chopped mid-word.

    The validation / save status shares the cramped floating capsule with the toolbar and
    the «Тест / Сохранить» buttons; a long message used to be hard-clipped to a stump like
    «Предупр» («текст статуса сжеван»). This keeps it to a single line, elided on the right
    with the full text in the tooltip, and — being width-agnostic (:attr:`Ignored`) — it
    never widens the capsule enough to shove the buttons off the canvas edge. :meth:`text`
    still returns the whole message, so callers (and the tests that pin it) read the real
    status, not the elided form.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def setText(self, text: str) -> None:  # noqa: N802 — Qt override.
        self._full = text
        self.setToolTip(text)
        self._relayout()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override.
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        elided = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, max(self.width(), 0)
        )
        super().setText(elided)


class _ScrollPage(QScrollArea):
    """A vertically scrolling column of sections, so a tall tab stays usable."""

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self._host = QWidget()
        self._host.setProperty("transparent", True)
        self._layout = QVBoxLayout(self._host)
        self._layout.setContentsMargins(0, 0, theme.metric("spacing_sm"), 0)
        self._layout.setSpacing(theme.metric("spacing_md"))
        self._layout.addStretch(1)
        self.setWidget(self._host)

    def add(self, widget: QWidget) -> None:
        self._layout.insertWidget(self._layout.count() - 1, widget)


def _section(title: str, body: QWidget, theme: ThemeManager) -> QWidget:
    """A titled block: a heading over the body, on the themed card surface."""
    card = QWidget()
    card.setProperty("card", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(*(theme.metric("spacing_md"),) * 4)
    layout.setSpacing(theme.metric("spacing_sm"))
    heading = QLabel(title)
    heading.setProperty("role", "h2")
    layout.addWidget(heading)
    layout.addWidget(body)
    return card


def _referenced_names(command: CommandModel) -> set[str]:
    """Every ``{name}`` mentioned in the command's block parameters."""
    names: set[str] = set()
    for location in command.blocks():
        for value in location.block.params.values():
            names.update(re.findall(r"\{(\w+)\}", value)) if isinstance(value, str) else None
    return names


def _status_glyph(status: str) -> str:
    return {"ok": "✓", "failed": "✗", "skipped": "⊘", "cancelled": "■"}.get(status, "•")


def _default_draft_store() -> DraftStore:
    """A draft store under the active profile's cache, used when none was injected.

    Resolving the profile path can, in principle, fail before the app has installed
    one (a bare test that builds an editor without services). A temp directory is a
    safe fallback: a draft written there is simply never recovered, which is exactly
    the guarantee a draft carries anyway.
    """
    from ayris.core.paths import get_paths

    try:
        directory = get_paths().command_drafts_dir
    except Exception:
        import tempfile
        from pathlib import Path

        directory = Path(tempfile.gettempdir()) / "ayris-command-drafts"
    return DraftStore(directory)


class _UnsavedDialog(QDialog):
    """«Сохранить / Не сохранять / Отмена» for the unsaved-changes guard, task 54.

    Three outcomes, so it is not the two-button :class:`ConfirmDialog`. The choice is
    read from :attr:`choice` after :meth:`exec`; closing the dialog any other way
    leaves it :data:`UnsavedChoice.CANCEL`, the safe default that keeps the edit.
    """

    def __init__(self, summary: str, theme: ThemeManager, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.choice = UnsavedChoice.CANCEL
        self.setWindowTitle("Несохранённые изменения")
        self.setModal(True)
        pad = theme.metric("spacing_xl")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(pad, pad, pad, pad)
        layout.setSpacing(theme.metric("spacing_lg"))
        heading = QLabel("Сохранить изменения перед выходом?")
        heading.setProperty("role", "h2")
        body = QLabel(summary or "В команде есть несохранённые изменения.")
        body.setProperty("role", "secondary")
        body.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(body)

        buttons = QDialogButtonBox()
        save_button = QPushButton("Сохранить")
        save_button.setProperty("kind", "primary")
        discard_button = QPushButton("Не сохранять")
        discard_button.setProperty("kind", "danger")
        cancel_button = QPushButton("Отмена")
        save_button.clicked.connect(self._chose_save)
        discard_button.clicked.connect(self._chose_discard)
        cancel_button.clicked.connect(self.reject)
        buttons.addButton(cancel_button, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(discard_button, QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.addButton(save_button, QDialogButtonBox.ButtonRole.AcceptRole)
        layout.addWidget(buttons)
        self.setMinimumWidth(theme.metric("dialog_width"))

    def _chose_save(self) -> None:
        self.choice = UnsavedChoice.SAVE
        self.accept()

    def _chose_discard(self) -> None:
        self.choice = UnsavedChoice.DISCARD
        self.accept()
