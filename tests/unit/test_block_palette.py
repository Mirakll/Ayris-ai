"""Палитра блоков редактора команд (задача 52), offscreen.

Ничего не рендерится и не смотрится глазами. Проверяется состояние дерева и сигналы:
что дерево сгруппировано по категориям каталога с русскими заголовками, что поиск
фильтрует по названию, описанию и типу, что опасные блоки помечены глифом, а недоступные
выключены и несут причину в подсказке, что двойной клик по доступному блоку эмитит
:attr:`BlockPalette.block_chosen`, а по заголовку категории или выключенному блоку —
нет, и что перетаскивание кладёт в mime тип блока под :data:`BLOCK_MIME`. Каждый виджет
закрывается в фикстуре ``app``. Реальный :class:`QDrag` подменяется фейком — настоящий
запустил бы модальный цикл перетаскивания.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from ayris.actions.macros.blocks.catalog import (
    BlockCatalog,
    BlockCategory,
    BlockMeta,
    CategoryMeta,
)
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import block_palette as bp_module
from ayris.gui.widgets.block_palette import BLOCK_MIME, BlockPalette

pytestmark = pytest.mark.unit

_TYPE_ROLE = Qt.ItemDataRole.UserRole


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
def catalog() -> BlockCatalog:
    # Каталог сам делает discover; строим один раз на модуль.
    return BlockCatalog()


def _palette(theme: ThemeManager, catalog: BlockCatalog) -> BlockPalette:
    return BlockPalette(theme, catalog=catalog)


def _leaf_items(palette: BlockPalette) -> list[QTreeWidgetItem]:
    """Все дочерние (блочные) элементы дерева, поверх заголовков категорий."""
    tree = palette._tree
    items: list[QTreeWidgetItem] = []
    for i in range(tree.topLevelItemCount()):
        category = tree.topLevelItem(i)
        for j in range(category.childCount()):
            items.append(category.child(j))
    return items


def _item_for(palette: BlockPalette, block_type: str) -> QTreeWidgetItem:
    for item in _leaf_items(palette):
        if item.data(0, _TYPE_ROLE) == block_type:
            return item
    raise AssertionError(f"Блок {block_type!r} не найден в дереве")


def _category_item(palette: BlockPalette, index: int = 0) -> QTreeWidgetItem:
    item = palette._tree.topLevelItem(index)
    assert item is not None
    return item


# ----------------------------------------------------------------------
# структура дерева: категории и русские заголовки
# ----------------------------------------------------------------------


def test_tree_is_grouped_by_catalog_categories(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    tree = palette._tree
    categories = catalog.list_categories()
    assert tree.topLevelItemCount() == len(categories)
    # Каждый верхний узел несёт русский заголовок своей категории.
    for i, meta in enumerate(categories):
        header = tree.topLevelItem(i).text(0)
        assert meta.title_ru in header


def test_every_catalog_block_has_a_tree_item(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    types_in_tree = {item.data(0, _TYPE_ROLE) for item in _leaf_items(palette)}
    for category in catalog.list_categories():
        for block in catalog.list_blocks(category.type):
            assert block.type in types_in_tree


def test_block_rows_show_russian_titles(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    wait = _item_for(palette, "Wait")
    assert "Ожидание" in wait.text(0)


# ----------------------------------------------------------------------
# опасные и недоступные блоки
# ----------------------------------------------------------------------


def test_dangerous_block_is_marked_and_tooltip_warns(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # CloseApp опасен и доступен: глиф ⚠ в конце строки, подсказка про опасность.
    item = _item_for(palette, "CloseApp")
    assert item.text(0).rstrip().endswith("⚠")
    assert item.toolTip(0).startswith("Опасный блок.")
    assert bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)


def test_unavailable_block_is_disabled_with_reason(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # Say не подключён в этой сборке: строка выключена, причина в подсказке.
    item = _item_for(palette, "Say")
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert bool(item.flags() & Qt.ItemFlag.ItemIsSelectable)
    assert "ещё не подключено" in item.toolTip(0)
    reason = catalog.get("Say").unavailable_reason
    assert item.toolTip(0) == reason


def test_unavailable_dangerous_block_keeps_warning_marker(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # RunShell опасен и недоступен: глиф ⚠ в строке, но подсказка — причина недоступности.
    item = _item_for(palette, "RunShell")
    assert item.text(0).rstrip().endswith("⚠")
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "ещё не подключено" in item.toolTip(0)


def test_unavailable_without_reason_falls_back_to_default_tooltip(
    app: QApplication, theme: ThemeManager
) -> None:
    # Каталог всегда даёт причину для недоступного блока, поэтому ветку дефолтной
    # подсказки покрываем фейковым каталогом с пустым unavailable_reason.
    palette = BlockPalette(theme, catalog=_FakeCatalog())
    item = _item_for(palette, "NoReason")
    assert not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert item.toolTip(0) == "Действие недоступно в этой сборке."


def test_available_plain_block_has_no_warning(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # SetVolume доступен и не опасен: без ⚠ и с обычной подсказкой-описанием.
    item = _item_for(palette, "SetVolume")
    assert "⚠" not in item.text(0)
    assert bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert item.toolTip(0) == catalog.get("SetVolume").description_ru


# ----------------------------------------------------------------------
# поиск: фильтрация по названию, описанию и типу
# ----------------------------------------------------------------------


def test_search_by_title_hides_non_matches(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # Ввод текста в поле идёт через реальный сигнал textChanged -> _apply_filter.
    palette._search.setText("ожида")
    assert not _item_for(palette, "Wait").isHidden()
    assert _item_for(palette, "SetVolume").isHidden()
    # Категория без совпадений (Голос/Звук) целиком спрятана.
    audio_index = [c.type for c in catalog.list_categories()].index(BlockCategory.AUDIO)
    assert _category_item(palette, audio_index).isHidden()


def test_search_by_description_matches(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # «макрокоманд» встречается только в описаниях нативных блоков, не в их названиях.
    palette._search.setText("макрокоманд")
    assert not _item_for(palette, "Wait").isHidden()


def test_search_by_type_matches_latin_name(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # «wait» нет в русском заголовке/описании — совпадение идёт по типу блока.
    palette._search.setText("wait")
    assert not _item_for(palette, "Wait").isHidden()


def test_clearing_search_reveals_everything(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    palette._search.setText("ожида")
    assert _item_for(palette, "SetVolume").isHidden()
    palette._search.setText("")
    # Пустой запрос показывает все блоки и все категории снова.
    for item in _leaf_items(palette):
        assert not item.isHidden()
    for i in range(palette._tree.topLevelItemCount()):
        assert not _category_item(palette, i).isHidden()


# ----------------------------------------------------------------------
# двойной клик -> block_chosen
# ----------------------------------------------------------------------


def test_double_click_available_block_emits_type(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    item = _item_for(palette, "SetVolume")
    # Выбор идёт через itemActivated — его же поднимает и двойной клик, и Enter.
    palette._tree.itemActivated.emit(item, 0)
    assert seen == ["SetVolume"]


def test_double_click_emits_type_once_not_twice(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Регрессия: реальный двойной клик поднимает СРАЗУ два сигнала —
    # itemDoubleClicked и itemActivated. Когда оба были подключены к выбору,
    # один двойной клик вставлял два блока, и дубль всплывал стопкой на том же
    # месте — «вторая нода появляется после перемещения». Выбор должен сработать
    # ровно один раз.
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    item = _item_for(palette, "SetVolume")
    palette._tree.itemDoubleClicked.emit(item, 0)
    palette._tree.itemActivated.emit(item, 0)
    assert seen == ["SetVolume"]


def test_double_click_category_header_emits_nothing(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    # У заголовка категории нет типа блока — обработчик выходит молча.
    palette._tree.itemActivated.emit(_category_item(palette, 0), 0)
    assert seen == []


def test_double_click_disabled_block_emits_nothing(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    # Say недоступен: тип есть, но флага ItemIsEnabled нет — сигнала быть не должно.
    palette._tree.itemActivated.emit(_item_for(palette, "Say"), 0)
    assert seen == []


# ----------------------------------------------------------------------
# одиночный клик по заголовку категории сворачивает и разворачивает
# ----------------------------------------------------------------------


def test_single_click_category_toggles_expansion(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    category = _category_item(palette, 0)
    # Категории строятся раскрытыми: первый клик сворачивает, второй — разворачивает.
    assert category.isExpanded()
    palette._tree.itemClicked.emit(category, 0)
    assert not category.isExpanded()
    assert category.text(0).lstrip().startswith(bp_module._CHEVRON_SHUT)
    palette._tree.itemClicked.emit(category, 0)
    assert category.isExpanded()
    assert category.text(0).lstrip().startswith(bp_module._CHEVRON_OPEN)


def test_single_click_block_does_not_toggle_or_emit(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    category = _category_item(palette, 0)
    item = _item_for(palette, "SetVolume")
    # Клик по блоку не сворачивает его категорию и не эмитит выбор (это делает двойной).
    palette._tree.itemClicked.emit(item, 0)
    assert category.isExpanded()
    assert seen == []


# ----------------------------------------------------------------------
# перетаскивание: mime несёт тип блока
# ----------------------------------------------------------------------


class _FakeDrag:
    """Заглушка QDrag: запоминает mime, а exec ничего не запускает."""

    instances: ClassVar[list[_FakeDrag]] = []

    def __init__(self, _source: object) -> None:
        self._mime: object | None = None
        _FakeDrag.instances.append(self)

    def setMimeData(self, mime: object) -> None:  # noqa: N802 — Qt-совместимое имя.
        self._mime = mime

    def exec(self, _action: object = None) -> None:
        return None


def test_start_drag_carries_block_type_in_mime(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bp_module, "QDrag", _FakeDrag)
    _FakeDrag.instances.clear()
    palette = _palette(theme, catalog)
    tree = palette._tree
    tree.setCurrentItem(_item_for(palette, "SetVolume"))
    tree.startDrag(Qt.DropAction.CopyAction)
    assert len(_FakeDrag.instances) == 1
    mime = _FakeDrag.instances[0]._mime
    assert mime is not None
    assert bytes(mime.data(BLOCK_MIME)).decode("utf-8") == "SetVolume"
    assert mime.text() == "SetVolume"


def test_start_drag_on_category_header_is_noop(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bp_module, "QDrag", _FakeDrag)
    _FakeDrag.instances.clear()
    palette = _palette(theme, catalog)
    tree = palette._tree
    # У заголовка категории нет типа — startDrag выходит, QDrag не создаётся.
    tree.setCurrentItem(_category_item(palette, 0))
    tree.startDrag(Qt.DropAction.CopyAction)
    assert _FakeDrag.instances == []


# ----------------------------------------------------------------------
# фейковый каталог для ветки «недоступно без причины»
# ----------------------------------------------------------------------


def _fake_block(block_type: str, *, available: bool, reason: str = "") -> BlockMeta:
    return BlockMeta(
        type=block_type,
        category=BlockCategory.SYSTEM,
        title_ru=block_type,
        description_ru="описание",
        icon="box",
        fields=(),
        json_schema={},
        example={},
        available=available,
        unavailable_reason=reason,
    )


class _FakeCatalog:
    """Минимальный каталог: одна категория с недоступным блоком без причины."""

    def list_categories(self) -> tuple[CategoryMeta, ...]:
        return (CategoryMeta(BlockCategory.SYSTEM, "Система", "monitor-cog"),)

    def list_blocks(self, _category: object) -> tuple[BlockMeta, ...]:
        return (_fake_block("NoReason", available=False, reason=""),)
