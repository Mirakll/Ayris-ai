"""A parameter form generated from a block's schema, never hand-written per block.

The macro editor draws the parameters of the selected block here. The form is built
from the :class:`~ayris.actions.base.ParamField` list an action or native block
publishes (task 33 catalog / task 30 :func:`ayris.actions.base.build_schema`), so a
new action grows an editable form the day it is written and nothing in this module
knows that ``SetVolume`` has a ``level``.

Each :class:`~ayris.actions.base.FieldKind` maps to one input widget:

* ``TEXT`` — a line edit, or a text area when the field is ``multiline``; a masked
  line edit when it is ``secret`` so a token never shows in the open.
* ``INTEGER`` — a slider with a spin box when the field has both bounds, a plain spin
  box otherwise; ``NUMBER`` — a double spin box.
* ``BOOLEAN`` — the themed toggle.
* ``CHOICE`` — a themed combo, values kept as their schema type through ``currentData``.
* ``LIST`` / ``OBJECT`` — a line edit parsed as JSON, with the raw text kept when it
  does not parse yet, so half-typed input is not thrown away.

An unknown kind falls back to a text field with a warning caption rather than raising:
opening a command written by a newer build must never crash the editor.

Text fields autocomplete ``{placeholder}`` tokens — the command's variables and the
slots its voice triggers declare — through a completer fed by :meth:`set_completions`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from PySide6.QtCore import QStringListModel, Qt, Signal
from PySide6.QtWidgets import (
    QCompleter,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.base import FieldKind, ParamField
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.slider_field import SliderField
from ayris.gui.widgets.toggle import ToggleSwitch

__all__ = ["ParamForm"]

#: Qt's signed-int ceiling; a spin box cannot show a wider range than this.
_INT_LIMIT = 2_147_483_647
#: Widest span still worth a slider; beyond it a spin box is the honest control.
_SLIDER_SPAN = 100_000


class _Row:
    """One built field: its description and the reader/writer that move its value."""

    __slots__ = ("field", "getter", "setter")

    def __init__(
        self,
        field: ParamField,
        getter: object,
        setter: object,
    ) -> None:
        self.field = field
        self.getter = getter
        self.setter = setter


class ParamForm(QWidget):
    """Editable form for one block's parameters, built from its schema."""

    #: Emitted whenever any field changes, so the editor can mark the command dirty.
    changed = Signal()

    def __init__(
        self,
        theme: ThemeManager,
        *,
        fields: Sequence[ParamField] = (),
        completions: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._completions = _tokens(completions)
        self._completer_model = QStringListModel(list(self._completions))
        self._rows: list[_Row] = []
        #: Suppresses ``changed`` while values are loaded programmatically, so filling
        #: a block's fields never looks like the user editing them (which would re-enter
        #: the editor's change handler and loop through selection).
        self._loading = False
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(0, 0, 0, 0)
        self._outer.setSpacing(theme.metric("spacing_xs"))
        self._form_host = QWidget()
        self._form_host.setProperty("transparent", True)
        self._form = QFormLayout(self._form_host)
        self._form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setHorizontalSpacing(theme.metric("spacing_md"))
        self._form.setVerticalSpacing(theme.metric("spacing_sm"))
        self._empty = QLabel("У блока нет параметров.")
        self._empty.setProperty("role", "secondary")
        self._empty.setWordWrap(True)
        self._outer.addWidget(self._empty)
        self._outer.addWidget(self._form_host)
        if fields:
            self.set_schema(fields)
        else:
            self._form_host.hide()

    # -- public API ---------------------------------------------------------

    def set_completions(self, names: Sequence[str]) -> None:
        """Set the ``{placeholder}`` tokens text fields suggest."""
        self._completions = _tokens(names)
        self._completer_model.setStringList(list(self._completions))

    def set_schema(self, fields: Sequence[ParamField]) -> None:
        """Rebuild the whole form for a new block, discarding the old widgets."""
        self._loading = True
        self._clear()
        listed = tuple(fields)
        self._empty.setVisible(not listed)
        self._form_host.setVisible(bool(listed))
        for field in listed:
            control, row = self._build_field(field)
            self._rows.append(row)
            self._form.addRow(_caption(field), control)
            if field.description_ru and field.description_ru != field.label_ru:
                hint = QLabel(field.description_ru)
                hint.setProperty("role", "muted")
                hint.setWordWrap(True)
                self._form.addRow("", hint)
        self._loading = False

    def set_values(self, params: Mapping[str, Any]) -> None:
        """Load ``params`` into the built fields; missing keys keep their default."""
        self._loading = True
        try:
            for row in self._rows:
                name = row.field.name
                value = params.get(name, row.field.default)
                setter = row.setter
                assert callable(setter)
                setter(value)
        finally:
            self._loading = False

    def values(self) -> dict[str, Any]:
        """Collect the current fields into a params dict.

        An optional text, list or object left empty is dropped rather than sent as
        an empty string: the action's ``extra="forbid"`` model would otherwise take
        ``""`` where it wanted a real value or nothing at all.
        """
        result: dict[str, Any] = {}
        for row in self._rows:
            getter = row.getter
            assert callable(getter)
            value = getter()
            if value is None and not row.field.required:
                continue
            if (
                not row.field.required
                and row.field.kind in (FieldKind.TEXT, FieldKind.LIST, FieldKind.OBJECT)
                and value in ("", [], {})
            ):
                continue
            result[row.field.name] = value
        return result

    # -- construction -------------------------------------------------------

    def _clear(self) -> None:
        self._rows.clear()
        while self._form.rowCount():
            self._form.removeRow(0)

    def _build_field(self, field: ParamField) -> tuple[QWidget, _Row]:
        if field.choices:
            return self._build_choice(field)
        if field.kind is FieldKind.BOOLEAN:
            return self._build_bool(field)
        if field.kind is FieldKind.INTEGER:
            return self._build_int(field)
        if field.kind is FieldKind.NUMBER:
            return self._build_number(field)
        if field.kind in (FieldKind.LIST, FieldKind.OBJECT):
            return self._build_json(field)
        return self._build_text(field)

    def _build_choice(self, field: ParamField) -> tuple[QWidget, _Row]:
        combo = ThemedComboBox()
        for choice in field.choices:
            combo.addItem(choice.label_ru, choice.value)
        combo.currentIndexChanged.connect(self._on_changed)

        def getter() -> Any:
            return combo.currentData()

        def setter(value: Any) -> None:
            index = combo.findData(value)
            combo.setCurrentIndex(index if index >= 0 else 0)

        if field.default is not None:
            setter(field.default)
        return combo, _Row(field, getter, setter)

    def _build_bool(self, field: ParamField) -> tuple[QWidget, _Row]:
        toggle = ToggleSwitch(self._theme, label=field.label_ru, checked=bool(field.default))
        toggle.toggled.connect(self._on_changed)

        def getter() -> Any:
            return toggle.isChecked()

        def setter(value: Any) -> None:
            toggle.setChecked(bool(value))

        return toggle, _Row(field, getter, setter)

    def _build_int(self, field: ParamField) -> tuple[QWidget, _Row]:
        low = int(field.minimum) if field.minimum is not None else -_INT_LIMIT
        high = int(field.maximum) if field.maximum is not None else _INT_LIMIT
        bounded = field.minimum is not None and field.maximum is not None
        if bounded and 0 <= high - low <= _SLIDER_SPAN:
            slider = SliderField(
                self._theme,
                minimum=low,
                maximum=high,
                value=int(field.default) if field.default is not None else low,
                unit=field.unit_ru,
                label=field.label_ru,
            )
            slider.value_changed.connect(self._on_changed)

            def slider_get() -> Any:
                return slider.value()

            def slider_set(value: Any) -> None:
                slider.setValue(int(value) if value is not None else low)

            return slider, _Row(field, slider_get, slider_set)

        spin = QSpinBox()
        spin.setRange(low, high)
        if field.unit_ru:
            spin.setSuffix(f" {field.unit_ru}")
        if field.default is not None:
            spin.setValue(int(field.default))
        spin.valueChanged.connect(self._on_changed)

        def spin_get() -> Any:
            return spin.value()

        def spin_set(value: Any) -> None:
            spin.setValue(int(value) if value is not None else low)

        return spin, _Row(field, spin_get, spin_set)

    def _build_number(self, field: ParamField) -> tuple[QWidget, _Row]:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setSingleStep(0.1)
        spin.setRange(
            float(field.minimum) if field.minimum is not None else -1e9,
            float(field.maximum) if field.maximum is not None else 1e9,
        )
        if field.unit_ru:
            spin.setSuffix(f" {field.unit_ru}")
        if field.default is not None:
            spin.setValue(float(field.default))
        spin.valueChanged.connect(self._on_changed)

        def getter() -> Any:
            return round(spin.value(), 3)

        def setter(value: Any) -> None:
            spin.setValue(float(value) if value is not None else spin.minimum())

        return spin, _Row(field, getter, setter)

    def _build_text(self, field: ParamField) -> tuple[QWidget, _Row]:
        if field.multiline:
            editor = QPlainTextEdit()
            editor.setPlaceholderText(field.description_ru)
            editor.textChanged.connect(self._on_changed)

            def area_get() -> Any:
                return editor.toPlainText()

            def area_set(value: Any) -> None:
                editor.setPlainText("" if value is None else str(value))

            return editor, _Row(field, area_get, area_set)

        line = QLineEdit()
        if field.secret:
            line.setEchoMode(QLineEdit.EchoMode.Password)
        else:
            self._attach_completer(line)
        if field.max_length:
            line.setMaxLength(field.max_length)
        line.setPlaceholderText(field.description_ru)
        line.textChanged.connect(self._on_changed)

        def getter() -> Any:
            return line.text()

        def setter(value: Any) -> None:
            line.setText("" if value is None else str(value))

        return line, _Row(field, getter, setter)

    def _build_json(self, field: ParamField) -> tuple[QWidget, _Row]:
        line = QLineEdit()
        self._attach_completer(line)
        line.setPlaceholderText(
            "Список JSON, например [1, 2]" if field.kind is FieldKind.LIST else "Объект JSON"
        )
        line.textChanged.connect(self._on_changed)

        def getter() -> Any:
            text = line.text().strip()
            if not text:
                return [] if field.kind is FieldKind.LIST else {}
            try:
                return json.loads(text)
            except ValueError:
                # Keep the half-typed text rather than lose it; the validator flags it.
                return text

        def setter(value: Any) -> None:
            if value in (None, "", [], {}):
                line.setText("")
            elif isinstance(value, str):
                line.setText(value)
            else:
                line.setText(json.dumps(value, ensure_ascii=False))

        return line, _Row(field, getter, setter)

    def _attach_completer(self, line: QLineEdit) -> None:
        completer = QCompleter(self._completer_model, line)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchContains)
        line.setCompleter(completer)

    def _on_changed(self, *_args: object) -> None:
        if not self._loading:
            self.changed.emit()


def _caption(field: ParamField) -> str:
    """The label a required field shows with a marker, so a blank one is obvious."""
    return f"{field.label_ru} *" if field.required else field.label_ru


def _tokens(names: Sequence[str]) -> list[str]:
    """Format bare variable and slot names as the ``{name}`` tokens a field accepts."""
    seen: dict[str, None] = {}
    for name in names:
        text = name if name.startswith("{") else "{" + name + "}"
        seen.setdefault(text, None)
    return list(seen)
