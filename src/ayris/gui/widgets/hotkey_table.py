"""Rows and tables for the «Горячие клавиши» tab (task 55).

The tab shows two tables — the assistant's own system hotkeys and the hotkeys of
user commands — and both need the same three things done to every combination:
parse it *once* through task 37 so a combo here can never disagree with one stored
in the config or a command trigger, work out whether it is actually registered, and
mark it when two owners fight over it or Windows refuses to hand it over.

That decision-making lives in the pure builders here (:func:`build_system_rows`,
:func:`build_command_rows`), which take plain values and return frozen rows — no Qt,
no database, so the offscreen tests of task 55 exercise the model directly. The two
:class:`~PySide6.QtWidgets.QWidget` tables below only render those rows and emit a
signal when a button is pressed; the tab owns the capture dialog and the writes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.theme import ThemeManager
from ayris.utils.hotkey_manager import HotkeyBinding, detect_conflicts
from ayris.utils.hotkeys import Hotkey, try_parse_hotkey

__all__ = [
    "SYSTEM_HOTKEY_ORDER",
    "SYSTEM_HOTKEY_SPECS",
    "CommandHotkeyEntry",
    "CommandHotkeyRow",
    "CommandHotkeyStatus",
    "CommandHotkeyTable",
    "HotkeyRows",
    "SystemHotkeyRow",
    "SystemHotkeySpec",
    "SystemHotkeyStatus",
    "SystemHotkeyTable",
    "build_rows",
]


class SystemHotkeyStatus(StrEnum):
    """Registration state shown in a system hotkey row."""

    ACTIVE = "active"
    UNREGISTERED = "unregistered"
    EMPTY = "empty"


class CommandHotkeyStatus(StrEnum):
    """Registration state shown in a command hotkey row (task 55, item 3)."""

    ACTIVE = "active"
    DISABLED = "disabled"
    UNREGISTERED = "unregistered"


@dataclass(frozen=True, slots=True)
class SystemHotkeySpec:
    """A system hotkey the tab can show: its config field and Russian captions.

    The set of fields must stay equal to what
    :class:`~ayris.utils.hotkey_manager.HotkeyManager` actually registers; a unit
    test pins that so the two never drift.
    """

    field: str
    title: str
    description: str


#: The five system hotkeys, in the order task 55 lists them. Kept in step with
#: ``ayris.utils.hotkey_manager._SYSTEM_LABELS`` by ``test_hotkeys_tab``.
SYSTEM_HOTKEY_SPECS: tuple[SystemHotkeySpec, ...] = (
    SystemHotkeySpec("push_to_talk", "Push-to-Talk", "Говорить, пока сочетание зажато."),
    SystemHotkeySpec(
        "toggle_wake", "Режим Wake Word", "Включить или выключить активацию по слову."
    ),
    SystemHotkeySpec("toggle_overlay", "Оверлей", "Показать или скрыть плавающий оверлей."),
    SystemHotkeySpec("toggle_mute", "Mute микрофона", "Отключить или включить микрофон."),
    SystemHotkeySpec("cancel", "Отмена действия", "Прервать текущее выполнение и озвучку."),
)

SYSTEM_HOTKEY_ORDER: tuple[str, ...] = tuple(spec.field for spec in SYSTEM_HOTKEY_SPECS)


@dataclass(frozen=True, slots=True)
class SystemHotkeyRow:
    """A rendered system hotkey row: what to show and how it is faring."""

    spec: SystemHotkeySpec
    combo: str
    hotkey: Hotkey | None
    status: SystemHotkeyStatus
    conflict_with: tuple[str, ...] = ()
    registration_error: str = ""
    is_default: bool = True

    @property
    def has_problem(self) -> bool:
        return (
            bool(self.conflict_with)
            or bool(self.registration_error)
            or (self.status is SystemHotkeyStatus.UNREGISTERED)
        )


@dataclass(frozen=True, slots=True)
class CommandHotkeyEntry:
    """Raw facts about one command hotkey, gathered from storage by the tab.

    ``combo`` is the canonical string task 37 stored on the trigger, so the builder
    only has to parse it, never re-spell it. ``folder`` is the named path of the
    command's folder, root first (empty at the tree root).
    """

    command_id: int
    name: str
    folder: tuple[str, ...]
    combo: str
    command_enabled: bool
    trigger_enabled: bool
    require_admin: bool


@dataclass(frozen=True, slots=True)
class CommandHotkeyRow:
    """A rendered command hotkey row and its resolved status."""

    command_id: int
    name: str
    folder: tuple[str, ...]
    combo: str
    hotkey: Hotkey | None
    require_admin: bool
    status: CommandHotkeyStatus
    conflict_with: tuple[str, ...] = ()
    registration_error: str = ""

    @property
    def has_problem(self) -> bool:
        return (
            bool(self.conflict_with)
            or bool(self.registration_error)
            or (self.status is CommandHotkeyStatus.UNREGISTERED)
        )


@dataclass(frozen=True, slots=True)
class HotkeyRows:
    """The two tables' rows and the shared problem count for the tab header."""

    system: tuple[SystemHotkeyRow, ...]
    commands: tuple[CommandHotkeyRow, ...]

    @property
    def problem_count(self) -> int:
        return sum(row.has_problem for row in self.system) + sum(
            row.has_problem for row in self.commands
        )


def _command_owner(name: str) -> str:
    return f"команда «{name}»"


def build_rows(
    system_combos: Mapping[str, str],
    system_defaults: Mapping[str, str],
    command_entries: Sequence[CommandHotkeyEntry],
    *,
    registration_errors: Mapping[str, str] | None = None,
) -> HotkeyRows:
    """Resolve every row's status and cross-mark conflicts, without touching Qt.

    Combinations are parsed only through task 37 (:func:`try_parse_hotkey`); an
    unparseable stored value degrades to «not registered» rather than raising.
    Conflicts and the registered set are computed over the bindings that *would*
    be registered — every non-empty system combo and every enabled command combo —
    mirroring :meth:`~ayris.utils.hotkey_manager.HotkeyManager.reload`, which keeps
    the first claimant of a duplicate (system first) and skips the rest.
    """
    errors = dict(registration_errors or {})

    parsed_system: dict[str, Hotkey | None] = {
        spec.field: (
            (try_parse_hotkey(system_combos.get(spec.field, "")) or None)
            if system_combos.get(spec.field, "")
            else None
        )
        for spec in SYSTEM_HOTKEY_SPECS
    }
    parsed_commands: list[tuple[CommandHotkeyEntry, Hotkey | None]] = [
        (entry, try_parse_hotkey(entry.combo) if entry.combo else None) for entry in command_entries
    ]

    bindings: list[HotkeyBinding] = []
    for spec in SYSTEM_HOTKEY_SPECS:
        hotkey = parsed_system[spec.field]
        if hotkey is not None:
            bindings.append(HotkeyBinding(hotkey, spec.title, spec.field))
    for entry, hotkey in parsed_commands:
        if hotkey is not None and entry.command_enabled and entry.trigger_enabled:
            bindings.append(
                HotkeyBinding(hotkey, _command_owner(entry.name), "command", entry.command_id)
            )

    conflicts = {conflict.hotkey: conflict.owners for conflict in detect_conflicts(bindings)}

    system_rows: list[SystemHotkeyRow] = []
    for spec in SYSTEM_HOTKEY_SPECS:
        hotkey = parsed_system[spec.field]
        combo = system_combos.get(spec.field, "")
        others = tuple(o for o in conflicts.get(hotkey, ()) if o != spec.title) if hotkey else ()
        reg_error = errors.get(hotkey.canonical, "") if hotkey is not None else ""
        if hotkey is None:
            status = SystemHotkeyStatus.EMPTY
        elif reg_error:
            status = SystemHotkeyStatus.UNREGISTERED
        else:
            status = SystemHotkeyStatus.ACTIVE
        system_rows.append(
            SystemHotkeyRow(
                spec=spec,
                combo=combo,
                hotkey=hotkey,
                status=status,
                conflict_with=others,
                registration_error=reg_error,
                is_default=combo == system_defaults.get(spec.field, ""),
            )
        )

    command_rows: list[CommandHotkeyRow] = []
    for entry, hotkey in parsed_commands:
        owner = _command_owner(entry.name)
        others = tuple(o for o in conflicts.get(hotkey, ()) if o != owner) if hotkey else ()
        reg_error = errors.get(hotkey.canonical, "") if hotkey is not None else ""
        if not (entry.command_enabled and entry.trigger_enabled):
            command_status = CommandHotkeyStatus.DISABLED
        elif hotkey is None or reg_error or bool(others):
            # A live conflict means the manager skipped this claimant, so it is not
            # actually registered — say so, and the conflict text explains by whom.
            command_status = CommandHotkeyStatus.UNREGISTERED
        else:
            command_status = CommandHotkeyStatus.ACTIVE
        command_rows.append(
            CommandHotkeyRow(
                command_id=entry.command_id,
                name=entry.name,
                folder=entry.folder,
                combo=entry.combo,
                hotkey=hotkey,
                require_admin=entry.require_admin,
                status=command_status,
                conflict_with=others,
                registration_error=reg_error,
            )
        )

    return HotkeyRows(tuple(system_rows), tuple(command_rows))


# ----------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------


def _clear_grid(grid: QGridLayout) -> None:
    while grid.count():
        item = grid.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _status_label(text: str, severity: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setProperty("role", "secondary" if severity in {"", "secondary"} else None)
    if severity in {"warning", "error", "success"}:
        label.setProperty("status", severity)
    return label


def _column_caption(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "secondary")
    return label


# __APPEND_TABLES__


class SystemHotkeyTable(QWidget):
    """The system hotkeys table: purpose, combo, status and the row buttons."""

    assign_requested = Signal(str)
    clear_requested = Signal(str)
    reset_requested = Signal(str)

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._capture_available = True
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setColumnStretch(0, 1)
        theme.theme_changed.connect(lambda *_: self._apply_metrics())
        self._apply_metrics()

    def set_capture_available(self, available: bool) -> None:
        self._capture_available = available

    def _apply_metrics(self) -> None:
        gap = self._theme.metric("spacing_md")
        self._grid.setHorizontalSpacing(self._theme.metric("spacing_lg"))
        self._grid.setVerticalSpacing(gap)

    def set_rows(self, rows: Sequence[SystemHotkeyRow]) -> None:
        _clear_grid(self._grid)
        for column, caption in enumerate(("Действие", "Сочетание", "Статус", "")):
            self._grid.addWidget(_column_caption(caption), 0, column)
        for index, row in enumerate(rows, start=1):
            self._grid.addWidget(self._purpose_cell(row), index, 0)
            self._grid.addWidget(self._combo_cell(row.hotkey, row.has_problem), index, 1)
            self._grid.addWidget(_status_label(*_system_status(row)), index, 2)
            self._grid.addLayout(self._buttons(row), index, 3)

    def _purpose_cell(self, row: SystemHotkeyRow) -> QWidget:
        cell = QWidget()
        box = QVBoxLayout(cell)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(self._theme.metric("spacing_xs"))
        title = QLabel(row.spec.title)
        title.setProperty("role", "h3")
        description = QLabel(row.spec.description)
        description.setProperty("role", "secondary")
        description.setWordWrap(True)
        box.addWidget(title)
        box.addWidget(description)
        return cell

    def _combo_cell(self, hotkey: Hotkey | None, problem: bool) -> QLabel:
        label = QLabel(hotkey.label_ru if hotkey is not None else "—")
        label.setProperty("role", "code")
        if problem:
            label.setProperty("status", "warning")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _buttons(self, row: SystemHotkeyRow) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(self._theme.metric("spacing_xs"))
        assign = QPushButton("Назначить")
        assign.setEnabled(self._capture_available)
        if not self._capture_available:
            assign.setToolTip("Обработчик горячих клавиш не запущен.")
        assign.clicked.connect(lambda: self.assign_requested.emit(row.spec.field))
        clear = QPushButton("Очистить")
        clear.setEnabled(row.hotkey is not None)
        clear.clicked.connect(lambda: self.clear_requested.emit(row.spec.field))
        reset = QPushButton("Сброс")
        reset.setEnabled(not row.is_default)
        reset.setToolTip("Вернуть комбинацию по умолчанию для этого хоткея.")
        reset.clicked.connect(lambda: self.reset_requested.emit(row.spec.field))
        for button in (assign, clear, reset):
            box.addWidget(button)
        return box


def _system_status(row: SystemHotkeyRow) -> tuple[str, str]:
    if row.registration_error:
        return row.registration_error, "error"
    if row.conflict_with:
        return f"Конфликт: {', '.join(row.conflict_with)}", "warning"
    if row.status is SystemHotkeyStatus.EMPTY:
        return "Не назначена", "secondary"
    if row.status is SystemHotkeyStatus.ACTIVE:
        return "Зарегистрирована", "success"
    return "Не зарегистрирована", "warning"


def _command_status(row: CommandHotkeyRow) -> tuple[str, str]:
    if row.status is CommandHotkeyStatus.DISABLED:
        return "Команда выключена", "secondary"
    if row.registration_error:
        return row.registration_error, "error"
    if row.conflict_with:
        return f"Конфликт: {', '.join(row.conflict_with)}", "warning"
    if row.status is CommandHotkeyStatus.ACTIVE:
        return "Активна", "success"
    return "Не зарегистрирована", "warning"


class CommandHotkeyTable(QWidget):
    """The command hotkeys table: combo, command, folder, status and buttons."""

    assign_requested = Signal(int)
    clear_requested = Signal(int)
    open_requested = Signal(int)

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._capture_available = True
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setColumnStretch(1, 1)
        self._grid.setColumnStretch(2, 1)
        theme.theme_changed.connect(lambda *_: self._apply_metrics())
        self._apply_metrics()

    def set_capture_available(self, available: bool) -> None:
        self._capture_available = available

    def _apply_metrics(self) -> None:
        self._grid.setHorizontalSpacing(self._theme.metric("spacing_lg"))
        self._grid.setVerticalSpacing(self._theme.metric("spacing_md"))

    def set_rows(self, rows: Sequence[CommandHotkeyRow]) -> None:
        _clear_grid(self._grid)
        if not rows:
            empty = QLabel("Ни одной команде пока не назначен хоткей.")
            empty.setProperty("role", "secondary")
            empty.setWordWrap(True)
            self._grid.addWidget(empty, 0, 0, 1, 5)
            return
        for column, caption in enumerate(("Сочетание", "Команда", "Папка", "Статус", "")):
            self._grid.addWidget(_column_caption(caption), 0, column)
        for index, row in enumerate(rows, start=1):
            self._grid.addWidget(self._combo_cell(row), index, 0)
            self._grid.addWidget(self._command_cell(row), index, 1)
            self._grid.addWidget(self._folder_cell(row.folder), index, 2)
            self._grid.addWidget(_status_label(*_command_status(row)), index, 3)
            self._grid.addLayout(self._buttons(row), index, 4)

    def _combo_cell(self, row: CommandHotkeyRow) -> QLabel:
        label = QLabel(row.hotkey.label_ru if row.hotkey is not None else row.combo or "—")
        label.setProperty("role", "code")
        if row.has_problem:
            label.setProperty("status", "warning")
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    def _command_cell(self, row: CommandHotkeyRow) -> QWidget:
        cell = QWidget()
        box = QVBoxLayout(cell)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(self._theme.metric("spacing_xs"))
        name = QLabel(row.name)
        name.setProperty("role", "h3")
        name.setWordWrap(True)
        box.addWidget(name)
        if row.require_admin:
            badge = QLabel("🛡 нужны права администратора")
            badge.setProperty("role", "secondary")
            badge.setToolTip(
                "Команда запускается с повышением прав. Хоткей не сработает поверх окон "
                "с более высокими правами, если Ayris запущен без администратора."
            )
            box.addWidget(badge)
        return cell

    def _folder_cell(self, folder: tuple[str, ...]) -> QLabel:
        label = QLabel(" / ".join(folder) if folder else "— корень —")
        label.setProperty("role", "secondary")
        label.setWordWrap(True)
        return label

    def _buttons(self, row: CommandHotkeyRow) -> QVBoxLayout:
        box = QVBoxLayout()
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(self._theme.metric("spacing_xs"))
        open_link = QPushButton("Открыть команду")
        open_link.setProperty("link", True)
        open_link.setCursor(Qt.CursorShape.PointingHandCursor)
        open_link.clicked.connect(lambda: self.open_requested.emit(row.command_id))
        assign = QPushButton("Назначить")
        assign.setEnabled(self._capture_available)
        if not self._capture_available:
            assign.setToolTip("Обработчик горячих клавиш не запущен.")
        assign.clicked.connect(lambda: self.assign_requested.emit(row.command_id))
        clear = QPushButton("Очистить")
        clear.setEnabled(bool(row.combo))
        clear.clicked.connect(lambda: self.clear_requested.emit(row.command_id))
        for button in (open_link, assign, clear):
            box.addWidget(button)
        return box
