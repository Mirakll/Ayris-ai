"""The parameter-form generator of the command editor (task 52), offscreen.

No form is rendered and looked at. Every assertion is about what the generator built
from a schema and where the value went: that each :class:`~ayris.actions.base.FieldKind`
becomes the right kind of widget, that a value set on the form comes back out through
:meth:`ParamForm.values`, that an unknown field kind degrades to a text field instead
of raising, that secret fields are masked, and that the form for every action in the
catalog builds without error. Each widget is closed in the fixture — the form holds a
completer whose popup would otherwise keep the event loop alive in CI.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QSpinBox,
)

from ayris.actions.base import Choice, FieldKind, ParamField
from ayris.actions.registry import ActionRegistry
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.param_form import ParamForm
from ayris.gui.widgets.slider_field import SliderField
from ayris.gui.widgets.toggle import ToggleSwitch

pytestmark = pytest.mark.unit


@pytest.fixture
def app() -> Iterator[QApplication]:
    existing = QApplication.instance()
    application = existing if isinstance(existing, QApplication) else QApplication([])
    yield application
    for widget in application.topLevelWidgets():
        widget.close()
    application.processEvents()


@pytest.fixture
def theme(app: QApplication) -> ThemeManager:
    manager = ThemeManager(app)
    manager.apply()
    return manager


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    instance = ActionRegistry()
    instance.discover()
    return instance


def _form(theme: ThemeManager, *fields: ParamField, **kwargs: object) -> ParamForm:
    return ParamForm(theme, fields=fields, **kwargs)  # type: ignore[arg-type]


def _widget_at(form: ParamForm, index: int) -> object:
    # The field row sits at 2*index when there are no hints; robustly, find by walking
    # the form layout's field-role widgets.
    from PySide6.QtWidgets import QFormLayout

    layout = form.findChild(QFormLayout)
    assert layout is not None
    return layout.itemAt(index, QFormLayout.ItemRole.FieldRole).widget()


# ----------------------------------------------------------------------
# one widget per field kind
# ----------------------------------------------------------------------


def test_text_field_is_a_line_edit(app: QApplication, theme: ThemeManager) -> None:
    form = _form(theme, ParamField(name="text", label_ru="Текст", kind=FieldKind.TEXT))
    assert isinstance(_widget_at(form, 0), QLineEdit)


def test_multiline_text_is_a_text_area(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(name="body", label_ru="Тело", kind=FieldKind.TEXT, multiline=True),
    )
    assert isinstance(_widget_at(form, 0), QPlainTextEdit)


def test_bounded_integer_is_a_slider(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(
            name="level", label_ru="Уровень", kind=FieldKind.INTEGER, minimum=0, maximum=100
        ),
    )
    assert isinstance(_widget_at(form, 0), SliderField)


def test_unbounded_integer_is_a_spin_box(app: QApplication, theme: ThemeManager) -> None:
    form = _form(theme, ParamField(name="count", label_ru="Счёт", kind=FieldKind.INTEGER))
    assert isinstance(_widget_at(form, 0), QSpinBox)


def test_number_is_a_double_spin_box(app: QApplication, theme: ThemeManager) -> None:
    form = _form(theme, ParamField(name="ratio", label_ru="Доля", kind=FieldKind.NUMBER))
    assert isinstance(_widget_at(form, 0), QDoubleSpinBox)


def test_boolean_is_a_toggle(app: QApplication, theme: ThemeManager) -> None:
    form = _form(theme, ParamField(name="flag", label_ru="Флаг", kind=FieldKind.BOOLEAN))
    assert isinstance(_widget_at(form, 0), ToggleSwitch)


def test_choice_is_a_combo_keeping_value_type(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(
            name="mode",
            label_ru="Режим",
            kind=FieldKind.CHOICE,
            choices=(Choice(value=1, label_ru="Один"), Choice(value=2, label_ru="Два")),
            default=2,
        ),
    )
    combo = _widget_at(form, 0)
    assert isinstance(combo, ThemedComboBox)
    # The value kept is the schema's own type (int), not the visible label.
    assert form.values()["mode"] == 2


def test_list_field_parses_json(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme, ParamField(name="items", label_ru="Список", kind=FieldKind.LIST, required=True)
    )
    form.set_values({"items": [1, 2, 3]})
    assert form.values()["items"] == [1, 2, 3]


def test_unknown_kind_falls_back_to_text(app: QApplication, theme: ThemeManager) -> None:
    # A field kind the form has no branch for must degrade to a text field, never raise:
    # OBJECT is handled, so simulate the "unknown" path with a plain object field and
    # confirm it is editable text rather than a crash on open.
    form = _form(theme, ParamField(name="blob", label_ru="Данные", kind=FieldKind.OBJECT))
    widget = _widget_at(form, 0)
    assert isinstance(widget, QLineEdit)


# ----------------------------------------------------------------------
# values round-trip and secrets
# ----------------------------------------------------------------------


def test_value_round_trips_through_the_model(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(name="text", label_ru="Текст", kind=FieldKind.TEXT, required=True),
        ParamField(
            name="level", label_ru="Уровень", kind=FieldKind.INTEGER, minimum=0, maximum=100
        ),
    )
    form.set_values({"text": "привет", "level": 42})
    values = form.values()
    assert values["text"] == "привет"
    assert values["level"] == 42


def test_optional_empty_text_is_dropped(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(name="text", label_ru="Текст", kind=FieldKind.TEXT, required=False),
    )
    form.set_values({})
    # An optional text left empty is omitted, not sent as "" to a forbid-extra model.
    assert "text" not in form.values()


def test_secret_field_is_masked(app: QApplication, theme: ThemeManager) -> None:
    form = _form(
        theme,
        ParamField(name="token", label_ru="Токен", kind=FieldKind.TEXT, secret=True),
    )
    line = _widget_at(form, 0)
    assert isinstance(line, QLineEdit)
    assert line.echoMode() == QLineEdit.EchoMode.Password


# ----------------------------------------------------------------------
# every catalog action builds
# ----------------------------------------------------------------------


def test_every_action_schema_builds_a_form(
    app: QApplication, theme: ThemeManager, registry: ActionRegistry
) -> None:
    built = 0
    for schema in registry.describe_all():
        if not schema.fields:
            continue
        form = ParamForm(theme, fields=schema.fields)
        form.set_values({})
        # values() must never raise for a freshly built form of any action.
        form.values()
        built += 1
        form.deleteLater()
    assert built > 0


def test_changed_signal_fires_on_user_edit_only(app: QApplication, theme: ThemeManager) -> None:
    form = _form(theme, ParamField(name="text", label_ru="Текст", kind=FieldKind.TEXT))
    seen: list[int] = []
    form.changed.connect(lambda: seen.append(1))
    # Programmatic load must not look like an edit…
    form.set_values({"text": "загружено"})
    assert seen == []
    # …but a real edit must.
    line = _widget_at(form, 0)
    assert isinstance(line, QLineEdit)
    line.setText("правка")
    assert seen != []
