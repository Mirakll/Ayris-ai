"""Шапка команды в редакторе (задача 52), offscreen.

Ничего не отрисовывается и не разглядывается: все проверки — про состояние
:class:`CommandModel` за виджетом и про сигнал :attr:`EditorHeader.changed`.
Проверяем, что :meth:`EditorHeader.set_command` раскладывает модель по полям и при
загрузке не шлёт ``changed``; что правка имени, описания, приоритета, кулдауна и
переключателей уходит в модель и поднимает ``changed``; что имя валидируется на
пустоту и на дубль (по casefold среди имён-соседей) и это видно через
:meth:`EditorHeader.is_name_valid`; и что чипы-теги добавляются и убираются, меняя
``model.tags``. У :class:`TagChips` внутри живёт :class:`QCompleter`, поэтому каждый
виджет закрывается фикстурой ``app`` — иначе его поповер держал бы цикл событий в CI.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtWidgets import QApplication, QPushButton

from ayris.actions.macros.schema import CommandModel
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.editor_header import EditorHeader, TagChips

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


def _counter(header: EditorHeader) -> list[int]:
    """Список, куда падает единица на каждый ``changed`` заголовка."""
    seen: list[int] = []
    header.changed.connect(lambda: seen.append(1))
    return seen


# ----------------------------------------------------------------------
# TagChips: набор, добавление, удаление, подсказки
# ----------------------------------------------------------------------


def test_tag_chips_set_tags_dedupes_and_drops_empty(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["альфа", "альфа", "бета", ""])
    # Дубли схлопнуты, пустая строка отброшена, порядок сохранён.
    assert chips.tags() == ["альфа", "бета"]


def test_tag_chips_tags_returns_a_copy(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["один"])
    snapshot = chips.tags()
    snapshot.append("подделка")
    # Возврат — копия: правка списка снаружи не трогает виджет.
    assert chips.tags() == ["один"]


def test_tag_chips_commit_adds_new_tag_and_emits(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    fired: list[int] = []
    chips.changed.connect(lambda: fired.append(1))
    chips._input.setText("  свежий  ")
    chips._commit_input()
    assert chips.tags() == ["свежий"]  # текст обрезан по краям
    assert chips._input.text() == ""  # поле очищено после ввода
    assert fired == [1]


def test_tag_chips_commit_ignores_empty(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    fired: list[int] = []
    chips.changed.connect(lambda: fired.append(1))
    chips._input.setText("   ")
    chips._commit_input()
    assert chips.tags() == []
    assert fired == []


def test_tag_chips_commit_ignores_duplicate(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["готово"])
    fired: list[int] = []
    chips.changed.connect(lambda: fired.append(1))
    chips._input.setText("готово")
    chips._commit_input()
    assert chips.tags() == ["готово"]
    assert fired == []


def test_tag_chips_remove_present_tag_emits(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["раз", "два"])
    fired: list[int] = []
    chips.changed.connect(lambda: fired.append(1))
    chips._remove("раз")
    assert chips.tags() == ["два"]
    assert fired == [1]


def test_tag_chips_remove_missing_tag_is_a_noop(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["раз"])
    fired: list[int] = []
    chips.changed.connect(lambda: fired.append(1))
    chips._remove("нет-такого")
    assert chips.tags() == ["раз"]
    assert fired == []


def test_tag_chips_chip_button_removes_its_tag(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["кино"])
    button = chips._chip_host.findChild(QPushButton)
    assert button is not None
    button.click()  # крестик на чипе зовёт _remove
    assert chips.tags() == []


def test_tag_chips_rebuild_clears_old_chips(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme)
    chips.set_tags(["а", "б", "в"])
    assert len(chips._chip_host.findChildren(QPushButton)) == 3
    chips.set_tags(["один"])
    # Старые чипы сняты (deleteLater), список перестроен под новый набор.
    app.processEvents()
    assert chips.tags() == ["один"]
    assert chips._chip_layout.count() == 1


def test_tag_chips_suggestions_filter_empty(app: QApplication, theme: ThemeManager) -> None:
    chips = TagChips(theme, suggestions=("альфа", "", "бета"))
    completer = chips._input.completer()
    assert completer is not None
    # Пустая подсказка отфильтрована, остались только осмысленные.
    assert completer.model().rowCount() == 2
    chips.set_suggestions(("гамма",))
    assert chips._input.completer().model().rowCount() == 1


# ----------------------------------------------------------------------
# EditorHeader: загрузка модели
# ----------------------------------------------------------------------


def test_fresh_header_reports_name_valid(app: QApplication, theme: ThemeManager) -> None:
    header = EditorHeader(theme)
    # Без загруженной команды подсказка об ошибке скрыта — имя считается валидным.
    assert header.is_name_valid() is True


def test_set_command_fills_every_field(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(
        name="Открыть браузер",
        description="Печатает привет",
        tags=["дом", "работа"],
        enabled=False,
        require_admin=True,
        priority=50,
        cooldown_ms=500,
    )
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    assert header._name.text() == "Открыть браузер"
    assert header._description.toPlainText() == "Печатает привет"
    assert header._tags.tags() == ["дом", "работа"]
    assert header._enabled.isChecked() is False
    assert header._require_admin.isChecked() is True
    assert header._priority.value() == 50
    assert header._cooldown.value() == 500
    assert header.is_name_valid() is True


def test_set_command_does_not_emit_changed(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(
        name="Тихая загрузка",
        description="описание",
        tags=["метка"],
        enabled=False,
        require_admin=True,
        priority=7,
        cooldown_ms=300,
    )
    header = EditorHeader(theme)
    seen = _counter(header)
    header.set_command(model, sibling_names=set())
    # Программная загрузка не должна выглядеть как правка пользователя.
    assert seen == []


# ----------------------------------------------------------------------
# EditorHeader: правка полей уходит в модель и шлёт changed
# ----------------------------------------------------------------------


def test_editing_name_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Старое имя")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._name.setText("  Новое имя  ")
    assert model.name == "Новое имя"  # имя обрезано по краям
    assert seen == [1]
    assert header.is_name_valid() is True


def test_editing_description_updates_model_and_emits(
    app: QApplication, theme: ThemeManager
) -> None:
    model = CommandModel(name="Команда")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._description.setPlainText("Новое описание")
    assert model.description == "Новое описание"
    assert seen != []


def test_editing_priority_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Команда")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._priority.setValue(123)
    assert model.priority == 123
    assert seen != []


def test_editing_cooldown_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Команда")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._cooldown.setValue(2500)
    assert model.cooldown_ms == 2500
    assert seen != []


def test_editing_enabled_updates_model_and_emits(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Команда", enabled=True)
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._enabled.setChecked(False)
    assert model.enabled is False
    assert seen != []


def test_editing_require_admin_updates_model_and_emits(
    app: QApplication, theme: ThemeManager
) -> None:
    model = CommandModel(name="Команда", require_admin=False)
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._require_admin.setChecked(True)
    assert model.require_admin is True
    assert seen != []


def test_require_admin_tooltip_mentions_uac(app: QApplication, theme: ThemeManager) -> None:
    header = EditorHeader(theme)
    # Пометка объясняет, что команда запросит повышение прав через UAC.
    assert "UAC" in header._require_admin.toolTip()


# ----------------------------------------------------------------------
# EditorHeader: правки без модели не падают и ничего не шлют
# ----------------------------------------------------------------------


def test_name_edit_without_model_does_not_emit(app: QApplication, theme: ThemeManager) -> None:
    header = EditorHeader(theme)
    seen = _counter(header)
    header._name.setText("что-то")  # модель ещё не задана
    # Без модели правка имени только валидирует, но не шлёт changed.
    assert seen == []


def test_field_edit_without_model_does_not_emit(app: QApplication, theme: ThemeManager) -> None:
    header = EditorHeader(theme)
    seen = _counter(header)
    header._priority.setValue(10)
    header._cooldown.setValue(10)
    header._require_admin.setChecked(True)
    assert seen == []


def test_tag_edit_without_model_does_not_emit(app: QApplication, theme: ThemeManager) -> None:
    header = EditorHeader(theme)
    seen = _counter(header)
    header._tags._input.setText("тег")
    header._tags._commit_input()  # tags.changed -> _on_tags, но модели нет
    assert seen == []


# ----------------------------------------------------------------------
# EditorHeader: теги через модель
# ----------------------------------------------------------------------


def test_adding_tag_updates_model_tags(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Команда")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._tags._input.setText("новый")
    header._tags._commit_input()
    assert model.tags == ["новый"]
    assert seen == [1]


def test_removing_tag_updates_model_tags(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Команда", tags=["раз", "два"])
    header = EditorHeader(theme)
    header.set_command(model, sibling_names=set())
    seen = _counter(header)
    header._tags._remove("раз")
    assert model.tags == ["два"]
    assert seen == [1]


# ----------------------------------------------------------------------
# EditorHeader: валидация имени
# ----------------------------------------------------------------------


def test_empty_name_is_marked_invalid(app: QApplication, theme: ThemeManager) -> None:
    # Пустое имя проверяем без модели: логика валидации срабатывает и подсвечивает
    # поле, а присвоения пустой строки в модель (которая его бы отвергла) не будет.
    header = EditorHeader(theme)
    header._name.setText("Имя")
    assert header.is_name_valid() is True
    header._name.setText("")
    assert header.is_name_valid() is False
    assert header._name.property("invalid") is True


def test_duplicate_sibling_name_is_marked_invalid(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Уникальное")
    header = EditorHeader(theme)
    # Сосед задан в другом регистре — сверка идёт по casefold.
    header.set_command(model, sibling_names={"Дубль"})
    assert header.is_name_valid() is True
    header._name.setText("дУбЛь")
    assert header.is_name_valid() is False
    assert model.name == "дУбЛь"  # непустое имя всё же ушло в модель


def test_valid_name_clears_the_error(app: QApplication, theme: ThemeManager) -> None:
    model = CommandModel(name="Имя")
    header = EditorHeader(theme)
    header.set_command(model, sibling_names={"занято"})
    header._name.setText("занято")
    assert header.is_name_valid() is False
    header._name.setText("свободно")
    assert header.is_name_valid() is True
    assert header._name.property("invalid") is False
