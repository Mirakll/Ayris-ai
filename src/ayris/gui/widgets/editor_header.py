"""The head of a command in the editor: its name, description, tags and switches.

Every control here writes back into the working :class:`CommandModel` and emits
:attr:`EditorHeader.changed`, so the editor never keeps a second copy of a field.
Two things are checked on the spot, because a name is what the tree and the trigger
matcher key a command by: an empty name and a name already used by a sibling command
are marked before the command can be saved. The list of sibling names is handed in
(the store knows the folder), so this widget stays a pure view over the model.

Tags are chips with autocompletion from the tags already in the library, added by the
completer's line edit and removed by the chip's own button; nothing here touches the
database.
"""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCompleter,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.schema import CommandModel
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.toggle import ToggleSwitch

__all__ = ["EditorHeader", "TagChips"]


class TagChips(QWidget):
    """A row of tag chips with a completing input, editing a list of strings."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        suggestions: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._tags: list[str] = []
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(theme.metric("spacing_xs"))
        self._chip_host = QWidget()
        self._chip_host.setProperty("transparent", True)
        self._chip_layout = QHBoxLayout(self._chip_host)
        self._chip_layout.setContentsMargins(0, 0, 0, 0)
        self._chip_layout.setSpacing(theme.metric("spacing_xs"))
        self._input = QLineEdit()
        self._input.setPlaceholderText("Добавить тег…")
        self._input.setClearButtonEnabled(True)
        self._input.returnPressed.connect(self._commit_input)
        self.set_suggestions(suggestions)
        self._layout.addWidget(self._chip_host)
        self._layout.addWidget(self._input, 1)

    def set_suggestions(self, suggestions: Sequence[str]) -> None:
        completer = QCompleter(sorted({s for s in suggestions if s}), self._input)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self._input.setCompleter(completer)

    def set_tags(self, tags: Sequence[str]) -> None:
        self._tags = list(dict.fromkeys(tag for tag in tags if tag))
        self._rebuild()

    def tags(self) -> list[str]:
        return list(self._tags)

    def _commit_input(self) -> None:
        text = self._input.text().strip()
        self._input.clear()
        if text and text not in self._tags:
            self._tags.append(text)
            self._rebuild()
            self.changed.emit()

    def _remove(self, tag: str) -> None:
        if tag in self._tags:
            self._tags.remove(tag)
            self._rebuild()
            self.changed.emit()

    def _rebuild(self) -> None:
        while self._chip_layout.count():
            item = self._chip_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for tag in self._tags:
            self._chip_layout.addWidget(self._make_chip(tag))

    def _make_chip(self, tag: str) -> QWidget:
        chip = QPushButton(f"{tag}  ✕")
        chip.setProperty("chip", True)
        chip.setCursor(Qt.CursorShape.PointingHandCursor)
        chip.setToolTip("Убрать тег")
        chip.clicked.connect(lambda: self._remove(tag))
        return chip


class EditorHeader(QWidget):
    """Name, description, tags and the command's switches, over a ``CommandModel``."""

    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        tag_suggestions: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._model: CommandModel | None = None
        self._sibling_names: set[str] = set()
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_sm"))

        self._name = QLineEdit()
        self._name.setPlaceholderText("Имя команды")
        self._name.textChanged.connect(self._on_name)
        outer.addWidget(self._name)

        self._name_error = QLabel("")
        self._name_error.setProperty("role", "muted")
        self._name_error.setProperty("badge", "error")
        self._name_error.setWordWrap(True)
        self._name_error.hide()
        outer.addWidget(self._name_error)

        self._description = QPlainTextEdit()
        self._description.setPlaceholderText("Описание команды")
        self._description.setFixedHeight(theme.metric("spacing_xl") * 3)
        self._description.textChanged.connect(self._on_field_changed)
        outer.addWidget(self._description)

        self._tags = TagChips(theme, suggestions=tag_suggestions)
        self._tags.changed.connect(self._on_tags)
        outer.addWidget(_labelled("Теги", self._tags, theme))

        self._enabled = ToggleSwitch(theme, label="Команда включена", checked=True)
        self._enabled.toggled.connect(self._on_field_changed)
        outer.addWidget(
            _switch_row(
                "Команда включена",
                "Выключенная команда остаётся в списке, но не срабатывает по триггерам.",
                self._enabled,
                theme,
            )
        )

        self._require_admin = ToggleSwitch(theme, label="Права администратора")
        self._require_admin.setToolTip(
            "Команда запросит повышение прав (UAC) перед запуском действий, "
            "которым нужны права администратора."
        )
        self._require_admin.toggled.connect(self._on_field_changed)
        outer.addWidget(
            _switch_row(
                "Права администратора",
                "Перед запуском команда запросит повышение прав через UAC.",
                self._require_admin,
                theme,
            )
        )

        numbers = QHBoxLayout()
        numbers.setSpacing(theme.metric("spacing_lg"))
        self._priority = QSpinBox()
        self._priority.setRange(-1000, 1000)
        self._priority.setToolTip("Чем больше приоритет, тем раньше срабатывает при совпадении.")
        self._priority.valueChanged.connect(self._on_field_changed)
        self._cooldown = QSpinBox()
        self._cooldown.setRange(0, 3_600_000)
        self._cooldown.setSingleStep(100)
        self._cooldown.setSuffix(" мс")
        self._cooldown.setToolTip("Минимальный промежуток между двумя запусками команды.")
        self._cooldown.valueChanged.connect(self._on_field_changed)
        numbers.addWidget(_labelled("Приоритет", self._priority, theme))
        numbers.addWidget(_labelled("Кулдаун", self._cooldown, theme))
        numbers.addStretch(1)
        outer.addLayout(numbers)

    # -- public API ---------------------------------------------------------

    def set_command(self, model: CommandModel, *, sibling_names: set[str]) -> None:
        """Load the model into the fields; no ``changed`` fires during the load."""
        self._loading = True
        try:
            self._model = model
            self._sibling_names = {name.casefold() for name in sibling_names}
            self._name.setText(model.name)
            self._description.setPlainText(model.description)
            self._tags.set_tags(model.tags)
            self._enabled.setChecked(model.enabled)
            self._require_admin.setChecked(model.require_admin)
            self._priority.setValue(model.priority)
            self._cooldown.setValue(model.cooldown_ms)
        finally:
            self._loading = False
        self._validate_name()

    def is_name_valid(self) -> bool:
        return not self._name_error.isVisibleTo(self)

    # -- editing ------------------------------------------------------------

    def _on_name(self, text: str) -> None:
        self._validate_name()
        if self._model is not None and not self._loading:
            name = text.strip()
            # Пустое имя в модель не пишем: CommandModel.name требует min_length=1 и
            # кинул бы ValidationError прямо в слоте. Ошибку показывает _validate_name,
            # а save()/_on_test() сами упрутся в is_name_valid(). changed всё равно
            # шлём — «очистил имя» делает форму грязной.
            if name:
                self._model.name = name
            self.changed.emit()

    def _on_tags(self) -> None:
        if self._model is not None and not self._loading:
            self._model.tags = self._tags.tags()
            self.changed.emit()

    def _on_field_changed(self, *_args: object) -> None:
        if self._model is None or self._loading:
            return
        self._model.description = self._description.toPlainText().strip()
        self._model.enabled = self._enabled.isChecked()
        self._model.require_admin = self._require_admin.isChecked()
        self._model.priority = self._priority.value()
        self._model.cooldown_ms = self._cooldown.value()
        self.changed.emit()

    def _validate_name(self) -> None:
        name = self._name.text().strip()
        if not name:
            self._show_name_error("Имя команды не может быть пустым.")
        elif name.casefold() in self._sibling_names:
            self._show_name_error("В этой папке уже есть команда с таким именем.")
        else:
            self._name_error.hide()
            self._name.setProperty("invalid", False)
            _repolish(self._name)

    def _show_name_error(self, message: str) -> None:
        self._name_error.setText(message)
        self._name_error.show()
        self._name.setProperty("invalid", True)
        _repolish(self._name)


def _labelled(text: str, widget: QWidget, theme: ThemeManager) -> QWidget:
    box = QWidget()
    box.setProperty("transparent", True)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(theme.metric("spacing_xs"))
    caption = QLabel(text)
    caption.setProperty("role", "muted")
    layout.addWidget(caption)
    layout.addWidget(widget)
    return box


def _switch_row(title: str, description: str, toggle: QWidget, theme: ThemeManager) -> QWidget:
    """A titled row for a toggle: title over description on the left, switch on the right.

    Mirrors :class:`~ayris.gui.widgets.setting_card.SettingCard` — the shape every other
    page gives a switch — so the command's switches read the same, but stays transparent
    because the «Команда» section is already a themed card and a card-in-card would double
    the surface.
    """
    box = QWidget()
    box.setProperty("transparent", True)
    row = QHBoxLayout(box)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(theme.metric("spacing_md"))

    texts = QVBoxLayout()
    texts.setContentsMargins(0, 0, 0, 0)
    texts.setSpacing(theme.metric("spacing_xs"))
    heading = QLabel(title)
    heading.setProperty("role", "h2")
    heading.setWordWrap(True)
    caption = QLabel(description)
    caption.setProperty("role", "secondary")
    caption.setWordWrap(True)
    texts.addWidget(heading)
    texts.addWidget(caption)
    row.addLayout(texts, 1)
    row.addWidget(toggle, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return box


def _repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
