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
from typing import Any, Protocol

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.base import ParamField
from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import CommandModel
from ayris.actions.macros.validator import Severity, ValidationReport, validate_command
from ayris.core.errors import AyrisError
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.action_list import ActionListModel, ActionListView, BlockPath
from ayris.gui.widgets.block_palette import BlockPalette
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.editor_header import EditorHeader
from ayris.gui.widgets.param_form import ParamForm
from ayris.gui.widgets.sound_binding import SoundBindingSection, SoundPreview
from ayris.gui.widgets.trigger_list import HotkeyCapture, TriggerList
from ayris.gui.widgets.variable_table import VariableTable
from ayris.utils.logger import get_logger

__all__ = ["MacroEditor", "MacroEditorServices", "MacroTestResult", "MacroTestRunner"]

_log = get_logger(__name__)


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
    test_runner: MacroTestRunner | None = None


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

        self._validator = _AsyncRunner(self)
        self._validator.finished.connect(self._on_validation)
        self._tester = _AsyncRunner(self)
        self._tester.finished.connect(self._on_test_finished)
        self._tester.failed.connect(self._on_test_failed)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*(theme.metric("spacing_md"),) * 4)
        outer.setSpacing(theme.metric("spacing_sm"))

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
        content_layout.addWidget(self._tabs, 1)
        self._build_overview_tab()
        self._build_actions_tab()
        self._build_variables_tab()
        self._build_sounds_tab()

        content_layout.addWidget(self._build_footer())
        self._content.hide()

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

        self._palette = BlockPalette(self._theme, catalog=self._catalog)
        self._palette.block_chosen.connect(self._on_palette_choice)

        self._param_title = QLabel("Параметры")
        self._param_title.setProperty("role", "h2")
        self._param_form = ParamForm(self._theme)
        self._param_form.changed.connect(self._on_params_changed)
        param_panel = QWidget()
        param_panel.setProperty("transparent", True)
        param_layout = QVBoxLayout(param_panel)
        param_layout.setContentsMargins(0, 0, 0, 0)
        param_layout.setSpacing(self._theme.metric("spacing_xs"))
        param_layout.addWidget(self._param_title)
        param_layout.addWidget(self._param_form, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._palette)
        splitter.addWidget(self._action_view)
        splitter.addWidget(param_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 2)
        self._tabs.addTab(splitter, "Действия")

    def _build_variables_tab(self) -> None:
        self._variables = VariableTable(self._theme)
        self._variables.changed.connect(self._on_variables_changed)
        page = _ScrollPage(self._theme)
        page.add(_section("Переменные", self._variables, self._theme))
        self._tabs.addTab(page, "Переменные")

    def _build_sounds_tab(self) -> None:
        self._sounds = SoundBindingSection(self._theme, preview=self._services.sound_preview)
        self._sounds.changed.connect(self._on_model_changed)
        page = _ScrollPage(self._theme)
        page.add(_section("Звуки стадий", self._sounds, self._theme))
        self._tabs.addTab(page, "Звуки")

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setProperty("transparent", True)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        self._status.setWordWrap(True)
        layout.addWidget(self._status, 1)

        self._test_button = QPushButton("Тест")
        self._test_button.setEnabled(self._services.test_runner is not None)
        self._test_button.clicked.connect(self._on_test)
        self._save_button = QPushButton("Сохранить")
        self._save_button.setProperty("kind", "primary")
        self._save_button.clicked.connect(self.save)
        layout.addWidget(self._test_button)
        layout.addWidget(self._save_button)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Лог теста появится здесь.")
        self._log.setMaximumHeight(self._theme.metric("spacing_xl") * 6)

        container = QWidget()
        container.setProperty("transparent", True)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(self._theme.metric("spacing_xs"))
        outer.addWidget(footer)
        outer.addWidget(self._log)
        return container

    # -- loading / saving ---------------------------------------------------

    def load_command(self, command_id: int) -> None:
        """Open a command by id, filling every section from its model."""
        try:
            model = self._store.command_model(command_id)
        except AyrisError as exc:
            self._show_placeholder(exc.user_message)
            return
        self._model = model
        self._header.set_command(model, sibling_names=self._store.sibling_names(command_id))
        self._triggers.set_command(model)
        self._triggers.set_conflicts(self._store.trigger_conflicts(command_id))
        self._action_model.set_command(model)
        self._action_view.rebuild()
        self._variables.set_command(model)
        self._sounds.set_command(model)
        self._param_form.set_schema(())
        self._param_title.setText("Параметры")
        self._refresh_completions()
        self._content.show()
        self._placeholder.hide()
        self._set_dirty(False)
        self._status.clear()
        self._log.clear()
        self._schedule_validation()

    def save(self) -> None:
        """Validate the name, then persist the whole model through the store."""
        if self._model is None:
            return
        if not self._header.is_name_valid():
            self._status.setText("Исправьте имя команды: оно пустое или уже занято.")
            self._tabs.setCurrentIndex(0)
            return
        try:
            saved = self._store.save_command(self._model)
        except AyrisError as exc:
            self._status.setText(exc.user_message)
            return
        self._model = saved
        self._action_model.set_command(saved)
        self._action_view.rebuild()
        self._set_dirty(False)
        self._status.setText("Команда сохранена.")
        if saved.id is not None:
            self.command_saved.emit(saved.id)

    def set_store(self, store: CommandTreeStore) -> None:
        """Point the editor at a new profile's library and clear the open command."""
        self._store = store
        self._model = None
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
        self._schedule_validation()

    def _on_triggers_changed(self) -> None:
        self._set_dirty(True)
        if self._model is not None and self._model.id is not None:
            self._triggers.set_conflicts(self._store.trigger_conflicts(self._model.id))
        self._refresh_completions()
        self._schedule_validation()

    def _on_actions_changed(self) -> None:
        self._set_dirty(True)
        self._schedule_validation()

    def _on_variables_changed(self) -> None:
        self._set_dirty(True)
        self._refresh_completions()
        self._schedule_validation()

    def _on_params_changed(self) -> None:
        block = self._selected_block()
        if block is not None:
            block.params = self._param_form.values()
            self._set_dirty(True)
            self._action_view.rebuild()
            self._schedule_validation()

    def _on_palette_choice(self, block_type: str) -> None:
        path = self._insertion_path()
        new_path = self._action_model.insert_type(block_type, path[0], path[1])
        if new_path is None:
            # Refused — the insertion point is already at the depth ceiling.
            self._status.setText("Слишком глубокая вложенность блоков.")
            return
        self._action_view.rebuild()
        self._action_view.select_path(new_path)
        self._on_actions_changed()

    def _insertion_path(self) -> tuple[BlockPath, int]:
        """Where a palette double-click inserts: after the selection, else at the end."""
        selected = self._action_view.selected_path()
        if selected:
            container, index = selected[:-1], selected[-1]
            assert isinstance(index, int)
            return container, index + 1
        root: BlockPath = ("actions",)
        return root, len(self._model.actions) if self._model is not None else 0

    def _on_block_selected(self, path: tuple[object, ...]) -> None:
        block = self._action_model.block_at(path) if path else None
        if block is None:
            self._param_form.set_schema(())
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
        return self._action_model.block_at(self._action_view.selected_path())

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
            self._status.setText(f"Ошибок: {errors}, предупреждений: {warnings}.")
        elif warnings:
            self._status.setText(f"Предупреждений: {warnings}.")
        else:
            self._status.setText("Проверка пройдена.")

    # -- test run -----------------------------------------------------------

    def _on_test(self) -> None:
        runner = self._services.test_runner
        if runner is None or self._model is None:
            return
        if not self._header.is_name_valid():
            self._status.setText("Исправьте имя команды перед тестом.")
            return
        try:
            self._store.save_command(self._model)
        except AyrisError as exc:
            self._status.setText(exc.user_message)
            return
        self._set_dirty(False)
        snapshot = self._model.model_copy(deep=True)
        slots = {name: f"<{name}>" for name in snapshot.slot_names}
        dry_run = any(self._is_dangerous(block.block.type) for block in snapshot.blocks())
        self._status.setText("Тест запущен…")
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
        self._status.setText(f"{prefix}{result.message}")

    def _on_test_failed(self, message: str) -> None:
        self._test_button.setEnabled(self._services.test_runner is not None)
        self._status.setText(f"Тест не выполнен: {message}")

    # -- helpers ------------------------------------------------------------

    def _set_dirty(self, dirty: bool) -> None:
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)

    def _show_placeholder(self, message: str) -> None:
        self._placeholder.setText(message)
        self._placeholder.show()
        self._content.hide()


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
