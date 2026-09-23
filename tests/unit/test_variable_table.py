"""Таблица объявленных переменных команды (задача 52), offscreen.

Ничего не отрисовывается и не разглядывается: каждая проверка — про состояние рабочей
:class:`CommandModel` и про сигнал :attr:`VariableTable.changed`. Проверяем, что
:meth:`set_command` строит строки из ``model.variables``; что добавление и удаление
строки меняют модель и шлют ``changed``; что смена типа, области, значения по умолчанию
и флага «хранить» летит обратно в модель; что :func:`_coerce_default` приводит текст к
типу и на плохом значении не роняет коммит, а красит строку; что
:meth:`set_diagnostics` подсвечивает строки по данным (флаги/фон/подсказка), а не по
пикселям; и что :meth:`declared_names` отдаёт имена. Все виджеты закрываются в
фикстуре ``app``.

Про подсказки. ``VariableScope``/``VariableType`` — ``StrEnum``, и Qt хранит их в
``userData`` как голую строку: ``QComboBox.currentData()`` возвращает ``"int"``, а не
``VariableType.INT``. Поэтому сравниваем данные комбобоксов через ``==`` (у ``StrEnum``
равенство со строкой истинно), а не через ``is``. Это же поведение — корень бага
``_coerce_default`` для int/float/bool, описанного в отчёте: тесты проверяют то, что
виджет делает на самом деле, а не то, как задумывалось.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QLabel,
    QPushButton,
    QTableWidget,
)

from ayris.actions.macros.schema import CommandModel, VariableModel
from ayris.core.models import VariableScope, VariableType
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.variable_table import VariableTable, _coerce_default

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


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _command(*variables: VariableModel) -> CommandModel:
    return CommandModel(name="Тест", variables=list(variables))


def _table(widget: VariableTable) -> QTableWidget:
    table = widget.findChild(QTableWidget)
    assert table is not None
    return table


def _button(widget: VariableTable, text: str) -> QPushButton:
    for button in widget.findChildren(QPushButton):
        if button.text() == text:
            return button
    raise AssertionError(f"нет кнопки «{text}»")


def _type_combo(widget: VariableTable, row: int) -> QComboBox:
    combo = _table(widget).cellWidget(row, 1)
    assert isinstance(combo, QComboBox)
    return combo


def _scope_combo(widget: VariableTable, row: int) -> QComboBox:
    combo = _table(widget).cellWidget(row, 2)
    assert isinstance(combo, QComboBox)
    return combo


# ----------------------------------------------------------------------
# set_command / declared_names
# ----------------------------------------------------------------------


def test_set_command_fills_rows_from_variables(app: QApplication, theme: ThemeManager) -> None:
    model = _command(
        VariableModel(name="имя", type=VariableType.STRING),
        VariableModel(name="счёт", type=VariableType.INT, default=5),
        VariableModel(
            name="флаг",
            type=VariableType.BOOL,
            scope=VariableScope.GLOBAL,
            default=True,
            persistent=True,
        ),
    )
    widget = VariableTable(theme)
    widget.set_command(model)
    table = _table(widget)

    assert table.rowCount() == 3
    assert table.item(0, 0).text() == "имя"
    assert table.item(1, 0).text() == "счёт"
    # тип и область читаются из данных комбобокса, не из подписи (StrEnum → сравнение ==)
    assert _type_combo(widget, 1).currentData() == VariableType.INT
    assert _scope_combo(widget, 2).currentData() == VariableScope.GLOBAL
    # None рисуется пустой ячейкой, значение — своей строкой
    assert table.item(0, 3).text() == ""
    assert table.item(1, 3).text() == "5"
    # флаг «хранить» отражает persistent
    assert table.item(0, 4).checkState() == Qt.CheckState.Unchecked
    assert table.item(2, 4).checkState() == Qt.CheckState.Checked


def test_declared_names_empty_then_filled(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    assert widget.declared_names() == []  # без модели — пусто, не падение

    widget.set_command(_command(VariableModel(name="альфа"), VariableModel(name="бета")))
    assert widget.declared_names() == ["альфа", "бета"]


def test_set_command_rebuilds_on_second_call(app: QApplication, theme: ThemeManager) -> None:
    # Повторная загрузка полностью перестраивает таблицу под новую модель.
    widget = VariableTable(theme)
    widget.set_command(_command(VariableModel(name="старая")))
    assert _table(widget).rowCount() == 1

    widget.set_command(_command(VariableModel(name="одна"), VariableModel(name="две")))
    assert _table(widget).rowCount() == 2
    assert widget.declared_names() == ["одна", "две"]


# ----------------------------------------------------------------------
# добавление / удаление строки
# ----------------------------------------------------------------------


@pytest.mark.qt_no_exception_capture
def test_add_row_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    # Добавление имени в новую строку срабатывает до появления комбобоксов и роняет
    # ассерт в _commit (см. отчёт, баг №2). Qt глотает исключение в слоте, и финальный
    # _commit всё-таки пишет модель — это и проверяем, отключив перехват pytest-qt.
    widget = VariableTable(theme)
    model = _command()
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))

    _button(widget, "Добавить").click()

    assert [v.name for v in model.variables] == ["переменная"]
    assert model.variables[0].type is VariableType.STRING
    assert model.variables[0].scope is VariableScope.LOCAL
    assert seen != []


@pytest.mark.qt_no_exception_capture
def test_add_row_avoids_name_collision(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    widget.set_command(_command(VariableModel(name="переменная")))

    _button(widget, "Добавить").click()
    _button(widget, "Добавить").click()

    assert widget.declared_names() == ["переменная", "переменная2", "переменная3"]


def test_add_row_without_model_is_noop(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)  # set_command не вызывали
    _button(widget, "Добавить").click()
    assert _table(widget).rowCount() == 0


def test_remove_selected_updates_model(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    model = _command(VariableModel(name="первая"), VariableModel(name="вторая"))
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))

    _table(widget).selectRow(0)
    _button(widget, "Удалить").click()

    assert [v.name for v in model.variables] == ["вторая"]
    assert seen != []


def test_remove_without_model_is_noop(app: QApplication, theme: ThemeManager) -> None:
    # _remove_selected всё равно зовёт _commit — тот обязан выйти при model=None.
    widget = VariableTable(theme)
    _button(widget, "Удалить").click()
    assert _table(widget).rowCount() == 0


# ----------------------------------------------------------------------
# правка ячеек: тип / область / значение / хранить
# ----------------------------------------------------------------------


def test_change_type_writes_to_model(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    model = _command(VariableModel(name="имя", type=VariableType.STRING))
    widget.set_command(model)
    seen: list[int] = []
    widget.changed.connect(lambda: seen.append(1))

    combo = _type_combo(widget, 0)
    combo.setCurrentIndex(combo.findData(VariableType.INT))

    assert model.variables[0].type is VariableType.INT
    assert seen != []


def test_change_scope_writes_to_model(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    model = _command(VariableModel(name="имя", scope=VariableScope.LOCAL))
    widget.set_command(model)

    combo = _scope_combo(widget, 0)
    combo.setCurrentIndex(combo.findData(VariableScope.PROFILE))

    assert model.variables[0].scope is VariableScope.PROFILE


def test_edit_string_default_writes_to_model(app: QApplication, theme: ThemeManager) -> None:
    # У строкового типа _coerce_default отдаёт текст как есть — значение доходит до модели.
    widget = VariableTable(theme)
    model = _command(VariableModel(name="имя", type=VariableType.STRING))
    widget.set_command(model)

    _table(widget).item(0, 3).setText("привет")

    assert model.variables[0].default == "привет"


def test_toggle_persistent_writes_to_model(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    # область global — иначе persistent недопустим и строка стала бы невалидной
    model = _command(VariableModel(name="флаг", scope=VariableScope.GLOBAL))
    widget.set_command(model)

    _table(widget).item(0, 4).setCheckState(Qt.CheckState.Checked)

    assert model.variables[0].persistent is True


def test_empty_name_row_is_skipped(app: QApplication, theme: ThemeManager) -> None:
    widget = VariableTable(theme)
    model = _command(VariableModel(name="имя"))
    widget.set_command(model)

    _table(widget).item(0, 0).setText("   ")  # пустое после strip

    assert model.variables == []


# ----------------------------------------------------------------------
# значение по умолчанию: невалидное красит строку, правка снимает подсветку
# ----------------------------------------------------------------------


def test_bad_default_marks_row_then_fix_clears_it(app: QApplication, theme: ThemeManager) -> None:
    # Тип list: _coerce_default идёт через json и ветку ARRAY/DICT (сравнение ==),
    # поэтому и падение на плохом значении, и восстановление на хорошем реально видны.
    widget = VariableTable(theme)
    model = _command(VariableModel(name="список", type=VariableType.ARRAY))
    widget.set_command(model)
    table = _table(widget)

    table.item(0, 3).setText("не json")

    assert model.variables == []  # строка выпала, но коммит не упал
    assert table.item(0, 3).toolTip() == "Значение не соответствует типу переменной."
    assert table.item(0, 3).background().color().alpha() == 64  # тон ошибки, полупрозрачный

    # исправление возвращает строку и снимает подсветку
    table.item(0, 3).setText("[1, 2]")
    assert model.variables[0].default == [1, 2]
    assert table.item(0, 3).toolTip() == ""
    assert table.item(0, 3).background().color().alpha() == 0


# ----------------------------------------------------------------------
# диагностика: неиспользуемые красятся, необъявленные перечисляются
# ----------------------------------------------------------------------


def test_diagnostics_paints_unused_and_lists_undeclared(
    app: QApplication, theme: ThemeManager
) -> None:
    widget = VariableTable(theme)
    widget.set_command(_command(VariableModel(name="альфа"), VariableModel(name="бета")))
    table = _table(widget)

    widget.set_diagnostics(unused=["альфа"], undeclared=["гамма"])

    assert table.item(0, 0).background().color().alpha() == 64  # альфа подсвечена
    assert table.item(0, 0).toolTip() == "На эту переменную никто не ссылается."
    assert table.item(1, 0).background().color().alpha() == 0  # бета чистая
    assert table.item(1, 0).toolTip() == ""

    diagnostics = widget.findChild(QLabel)
    assert diagnostics is not None
    # offscreen окно не показано, поэтому isVisible() всегда False — читаем флаг isHidden().
    assert not diagnostics.isHidden()
    assert "гамма" in diagnostics.text()

    # сброс: подсветка снята, ярлык спрятан
    widget.set_diagnostics(unused=[], undeclared=[])
    assert table.item(0, 0).background().color().alpha() == 0
    assert diagnostics.isHidden()


# ----------------------------------------------------------------------
# _coerce_default — приведение текста к типу
# ----------------------------------------------------------------------


def test_coerce_default_scalars() -> None:
    assert _coerce_default("", VariableType.STRING) is None
    assert _coerce_default("  ", VariableType.INT) is None
    assert _coerce_default("10", VariableType.INT) == 10
    assert _coerce_default("1.5", VariableType.FLOAT) == 1.5
    assert _coerce_default("привет", VariableType.STRING) == "привет"


@pytest.mark.parametrize("text", ["да", "true", "1", "yes", "ДА", "TRUE"])
def test_coerce_default_bool_true(text: str) -> None:
    assert _coerce_default(text, VariableType.BOOL) is True


@pytest.mark.parametrize("text", ["нет", "false", "0", "no", "НЕТ"])
def test_coerce_default_bool_false(text: str) -> None:
    assert _coerce_default(text, VariableType.BOOL) is False


def test_coerce_default_bool_bad_raises() -> None:
    with pytest.raises(ValueError):
        _coerce_default("может быть", VariableType.BOOL)


def test_coerce_default_int_bad_raises() -> None:
    with pytest.raises(ValueError):
        _coerce_default("не число", VariableType.INT)


def test_coerce_default_array_and_dict() -> None:
    assert _coerce_default("[1, 2, 3]", VariableType.ARRAY) == [1, 2, 3]
    assert _coerce_default('{"a": 1}', VariableType.DICT) == {"a": 1}


def test_coerce_default_array_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        _coerce_default('{"a": 1}', VariableType.ARRAY)  # словарь, а ждали список


def test_coerce_default_dict_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        _coerce_default("[1, 2]", VariableType.DICT)  # список, а ждали словарь


def test_coerce_default_bad_json_raises() -> None:
    with pytest.raises(ValueError):
        _coerce_default("не json", VariableType.ARRAY)
