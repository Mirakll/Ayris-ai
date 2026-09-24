"""Палитра блоков редактора команд карточками (задача 52 + редизайн), offscreen.

Ничего не рендерится глазами: проверяются состав карточек и сигналы. Что каталог разложен
по категориям с русскими заголовками капсом, что у каждого блока есть карточка, что поиск
прячет несовпадающие карточки и опустевшие категории, что опасный блок помечен ⚠ и
подсказкой, а недоступный — не интерактивен и несёт причину в подсказке, что клик по
доступной карточке эмитит :attr:`BlockPalette.block_chosen`, а по недоступной — нет, и что
перетаскивание кладёт тип блока в mime под :data:`BLOCK_MIME`. Реальный :class:`QDrag`
подменяется фейком — настоящий запустил бы модальный цикл перетаскивания.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QLabel

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


def _cards(palette: BlockPalette) -> list[object]:
    """Все карточки блоков палитры, поверх заголовков категорий."""
    return [card for card, _hay in palette._cards]


def _card_for(palette: BlockPalette, block_type: str) -> object:
    for card, _hay in palette._cards:
        if card._block.type == block_type:
            return card
    raise AssertionError(f"Блок {block_type!r} не найден в палитре")


def _header_for(palette: BlockPalette, title_ru: str) -> QLabel:
    for header, _ in palette._headers:
        if header.text() == title_ru.upper():
            return header
    raise AssertionError(f"Заголовок категории {title_ru!r} не найден")


def _name_text(card: object) -> str:
    for label in card.findChildren(QLabel):
        if label.property("ciName"):
            return label.text()
    raise AssertionError("У карточки нет метки названия")


def _mouse(
    kind: QEvent.Type,
    pos: tuple[float, float],
    button: Qt.MouseButton,
    buttons: Qt.MouseButton,
) -> QMouseEvent:
    point = QPointF(*pos)
    return QMouseEvent(kind, point, point, button, buttons, Qt.KeyboardModifier.NoModifier)


def _click(card: object) -> None:
    """Полный клик по карточке: зажать и отпустить на месте (без сдвига — не drag)."""
    card.mousePressEvent(
        _mouse(
            QEvent.Type.MouseButtonPress,
            (4, 4),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
    )
    card.mouseReleaseEvent(
        _mouse(
            QEvent.Type.MouseButtonRelease,
            (4, 4),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        )
    )


# ----------------------------------------------------------------------
# структура: категории капсом и русские названия карточек
# ----------------------------------------------------------------------


def test_cards_grouped_by_catalog_categories(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    non_empty = [c for c in catalog.list_categories() if catalog.list_blocks(c.type)]
    headers = [header.text() for header, _cards_ in palette._headers]
    # Заголовок — русское имя категории капсом, в порядке каталога; пустых категорий нет.
    assert headers == [c.title_ru.upper() for c in non_empty]


def test_every_catalog_block_has_a_card(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    types_in_palette = {card._block.type for card in _cards(palette)}
    for category in catalog.list_categories():
        for block in catalog.list_blocks(category.type):
            assert block.type in types_in_palette


def test_card_shows_russian_title(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    assert "Ожидание" in _name_text(_card_for(palette, "Wait"))


# ----------------------------------------------------------------------
# опасные и недоступные блоки
# ----------------------------------------------------------------------


def test_dangerous_block_is_marked_and_tooltip_warns(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # CloseApp опасен и доступен: ⚠ в конце названия, подсказка про опасность.
    card = _card_for(palette, "CloseApp")
    assert _name_text(card).rstrip().endswith("⚠")
    assert card.toolTip().startswith("Опасный блок.")
    assert card._available


def test_unavailable_block_is_not_interactive_with_reason(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # Say не подключён в этой сборке: карточка не интерактивна, причина в подсказке.
    card = _card_for(palette, "Say")
    assert not card._available
    assert card.property("interactive") is False
    assert "ещё не подключено" in card.toolTip()
    assert card.toolTip() == catalog.get("Say").unavailable_reason


def test_unavailable_dangerous_block_keeps_warning_marker(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # RunShell опасен и недоступен: ⚠ в названии, но подсказка — причина недоступности.
    card = _card_for(palette, "RunShell")
    assert _name_text(card).rstrip().endswith("⚠")
    assert not card._available
    assert "ещё не подключено" in card.toolTip()


def test_unavailable_without_reason_falls_back_to_default_tooltip(
    app: QApplication, theme: ThemeManager
) -> None:
    # Каталог всегда даёт причину; ветку дефолтной подсказки берём фейковым каталогом.
    palette = BlockPalette(theme, catalog=_FakeCatalog())
    card = _card_for(palette, "NoReason")
    assert not card._available
    assert card.toolTip() == "Действие недоступно в этой сборке."


def test_available_plain_block_has_no_warning(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # SetVolume доступен и не опасен: без ⚠, подсказка — описание.
    card = _card_for(palette, "SetVolume")
    assert "⚠" not in _name_text(card)
    assert card._available
    assert card.toolTip() == catalog.get("SetVolume").description_ru


# ----------------------------------------------------------------------
# поиск: фильтрация карточек по названию, описанию и типу
# ----------------------------------------------------------------------


def test_search_by_title_hides_non_matches(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # Ввод идёт через реальный сигнал textChanged -> _apply_filter.
    palette._search.setText("ожида")
    assert not _card_for(palette, "Wait").isHidden()
    assert _card_for(palette, "SetVolume").isHidden()
    # Категория без совпадений (Голос/Звук) целиком спрятана вместе с заголовком.
    audio_title = next(
        c.title_ru for c in catalog.list_categories() if c.type is BlockCategory.AUDIO
    )
    assert _header_for(palette, audio_title).isHidden()


def test_search_by_description_matches(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # «макрокоманд» встречается только в описаниях, не в названиях.
    palette._search.setText("макрокоманд")
    assert not _card_for(palette, "Wait").isHidden()


def test_search_by_type_matches_latin_name(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    # «wait» нет в русском заголовке/описании — совпадение по типу блока.
    palette._search.setText("wait")
    assert not _card_for(palette, "Wait").isHidden()


def test_clearing_search_reveals_everything(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    palette._search.setText("ожида")
    assert _card_for(palette, "SetVolume").isHidden()
    palette._search.setText("")
    # Пустой запрос показывает все карточки и все категории снова.
    for card in _cards(palette):
        assert not card.isHidden()
    for header, _ in palette._headers:
        assert not header.isHidden()


# ----------------------------------------------------------------------
# клик по карточке -> block_chosen
# ----------------------------------------------------------------------


def test_click_available_card_emits_type(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    _click(_card_for(palette, "SetVolume"))
    assert seen == ["SetVolume"]


def test_click_unavailable_card_emits_nothing(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    palette = _palette(theme, catalog)
    seen: list[str] = []
    palette.block_chosen.connect(seen.append)
    # Say недоступен: клик по нему не выбирает блок.
    _click(_card_for(palette, "Say"))
    assert seen == []


# ----------------------------------------------------------------------
# перетаскивание: mime несёт тип блока, недоступный не тянется
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
    _card_for(palette, "SetVolume")._start_drag()
    assert len(_FakeDrag.instances) == 1
    mime = _FakeDrag.instances[0]._mime
    assert mime is not None
    assert bytes(mime.data(BLOCK_MIME)).decode("utf-8") == "SetVolume"
    assert mime.text() == "SetVolume"


def test_drag_past_threshold_starts_on_available_card(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bp_module, "QDrag", _FakeDrag)
    _FakeDrag.instances.clear()
    palette = _palette(theme, catalog)
    card = _card_for(palette, "SetVolume")
    card.mousePressEvent(
        _mouse(
            QEvent.Type.MouseButtonPress,
            (4, 4),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
    )
    card.mouseMoveEvent(
        _mouse(QEvent.Type.MouseMove, (40, 40), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    )
    assert len(_FakeDrag.instances) == 1


def test_unavailable_card_does_not_start_drag(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bp_module, "QDrag", _FakeDrag)
    _FakeDrag.instances.clear()
    palette = _palette(theme, catalog)
    card = _card_for(palette, "Say")
    card.mousePressEvent(
        _mouse(
            QEvent.Type.MouseButtonPress,
            (4, 4),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
        )
    )
    card.mouseMoveEvent(
        _mouse(QEvent.Type.MouseMove, (40, 40), Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton)
    )
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

    def try_get(self, block_type: str) -> BlockMeta | None:
        return _fake_block("NoReason", available=False) if block_type == "NoReason" else None
