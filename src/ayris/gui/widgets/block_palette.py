"""Палитра блоков: каталог задачи 33 карточками, как в браузерном макете.

Блоки сгруппированы по категориям каталога; каждая — заголовок капсом, под ним карточки
«значок + название + описание» (``.cat-item`` из ``design/mockups/command_redesign``).
Поиск фильтрует по названию, описанию и типу во всех категориях. Блок добавляется кликом
(палитра шлёт :attr:`block_chosen`) или перетаскиванием на холст/список — drag несёт тип
блока тем же mime, что читает список действий. Значок и цвет карточки берутся от РОЛИ блока
(:func:`role_of`), поэтому иконка в каталоге совпадает с иконкой ноды после вставки. Опасные
блоки помечены ⚠ и подсказкой; недоступные в сборке — приглушены, без drag и клика, с
причиной в подсказке. Палитра ничего не строит и не трогает БД — только называет блоки.
"""

from __future__ import annotations

from PySide6.QtCore import QMimeData, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QResizeEvent,
)
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.blocks.catalog import BlockCatalog, BlockMeta
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.node_editor.bridge import ROLE_TOKENS, NodeRole, role_of
from ayris.gui.widgets.node_editor.icons import paint_block_glyph
from ayris.gui.widgets.search_field import SearchField

__all__ = ["BLOCK_MIME", "BlockPalette"]

#: The mime type a dragged palette block carries; the action list drop reads it.
BLOCK_MIME = "application/x-ayris-block-type"

#: How far the pointer must travel with the button down before a press becomes a drag
#: rather than a click — below this a release counts as «choose this block».
_DRAG_THRESHOLD = 8


def _mix(a: QColor, b: QColor, ratio: float) -> QColor:
    """``ratio`` of ``a`` over ``b`` — the mockup's ``color-mix`` for the icon tile tint."""
    inverse = 1.0 - ratio
    return QColor(
        round(a.red() * ratio + b.red() * inverse),
        round(a.green() * ratio + b.green() * inverse),
        round(a.blue() * ratio + b.blue() * inverse),
    )


class _GlyphIcon(QWidget):
    """28×28 плитка роли с глифом внутри — ``.ci-ico`` макета (фон роли @26 %, глиф ролью).

    Цвет плитки и обводки берётся от РОЛИ блока, а сам глиф — от ТИПА блока (клавиатура у
    «Нажать клавиши», часы у «Ожидание»…), с откатом к глифу роли для незнакомого типа.
    """

    def __init__(
        self,
        block_type: str,
        role: NodeRole,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._block_type = block_type
        self._role = role
        self._theme = theme
        self.setProperty("transparent", True)
        self.setFixedSize(28, 28)

    def paintEvent(self, _event: object) -> None:  # noqa: N802 — Qt override.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colour = QColor(self._theme.theme.color(ROLE_TOKENS[self._role]))
        surface = QColor(self._theme.theme.color("surface"))
        square = QRectF(0, 0, 28, 28)
        tile = QPainterPath()
        tile.addRoundedRect(square, 8, 8)
        painter.fillPath(tile, QBrush(_mix(colour, surface, 0.26)))
        paint_block_glyph(painter, square, self._block_type, self._role, colour, side=15.0)
        painter.end()


class _ElidedLabel(QLabel):
    """Однострочная метка, обрезающая текст многоточием по ширине, а не рубящая его.

    Это ``.cat-item small`` из макета с ``text-overflow: ellipsis``: описание блока всегда
    помещается в одну строку, а не «съедается» краем узкой панели. Политика ширины —
    :attr:`Ignored`, чтобы длинное описание не растягивало карточку и не диктовало ширину
    колонки; полный текст остаётся в подсказке карточки.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = text
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._relayout()

    def setText(self, text: str) -> None:  # noqa: N802 — Qt override.
        self._full = text
        self._relayout()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 — Qt override.
        super().resizeEvent(event)
        self._relayout()

    def _relayout(self) -> None:
        elided = self.fontMetrics().elidedText(
            self._full, Qt.TextElideMode.ElideRight, max(self.width(), 0)
        )
        super().setText(elided)


class _BlockCard(QFrame):
    """Строка каталога: значок + название + описание, перетаскиваемая и кликабельная.

    Повторяет ``.cat-item`` макета. Левый зажим, сдвинутый дальше :data:`_DRAG_THRESHOLD`,
    начинает drag типа блока (тем же mime, что читает список действий); отпускание на месте
    считается выбором блока. Недоступный блок не делает ни того, ни другого — лишь показывает
    причину в подсказке.
    """

    chosen = Signal(str)

    def __init__(
        self,
        block: BlockMeta,
        role: NodeRole,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._block = block
        self._available = block.available
        self._press: QPoint | None = None
        self._dragging = False
        self.setProperty("catItem", True)
        self.setProperty("interactive", block.available)
        if block.available:
            self.setCursor(Qt.CursorShape.PointingHandCursor)

        gap = theme.metric("spacing_sm")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(gap, gap, gap, gap)
        layout.setSpacing(gap)
        self._icon = _GlyphIcon(block.type, role, theme)
        layout.addWidget(self._icon)

        meta = QVBoxLayout()
        meta.setContentsMargins(0, 0, 0, 0)
        meta.setSpacing(1)
        name = QLabel(block.title_ru + ("  ⚠" if block.is_dangerous else ""))
        name.setProperty("ciName", True)
        name.setWordWrap(False)
        desc = _ElidedLabel(block.description_ru)
        desc.setProperty("ciDesc", True)
        meta.addWidget(name)
        meta.addWidget(desc)
        layout.addLayout(meta, 1)

        if not block.available:
            self.setToolTip(block.unavailable_reason or "Действие недоступно в этой сборке.")
        elif block.is_dangerous:
            self.setToolTip(f"Опасный блок. {block.description_ru}".strip())
        else:
            self.setToolTip(block.description_ru)

    def refresh_icon(self) -> None:
        self._icon.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if event.button() == Qt.MouseButton.LeftButton and self._available:
            self._press = event.position().toPoint()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if (
            self._press is not None
            and bool(event.buttons() & Qt.MouseButton.LeftButton)
            and not self._dragging
            and (event.position().toPoint() - self._press).manhattanLength() > _DRAG_THRESHOLD
        ):
            self._dragging = True
            self._start_drag()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 — Qt override.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._available
            and self._press is not None
            and not self._dragging
        ):
            self.chosen.emit(self._block.type)
        self._press = None
        self._dragging = False
        super().mouseReleaseEvent(event)

    def _start_drag(self) -> None:
        mime = QMimeData()
        mime.setData(BLOCK_MIME, self._block.type.encode("utf-8"))
        mime.setText(self._block.type)
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class BlockPalette(QWidget):
    """Каталог блоков карточками по категориям — панель сплиттера или всплывающее меню.

    Один и тот же виджет служит и боковой панелью редактора макросов, и всплывающим окном
    кнопки «＋» нодового редактора. Наружу отдаёт единственный сигнал :attr:`block_chosen`
    с типом блока; перетаскивание несёт тот же тип через :data:`BLOCK_MIME`.
    """

    #: Emitted with a block type when a card is clicked (drag uses :data:`BLOCK_MIME` instead).
    block_chosen = Signal(str)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        catalog: BlockCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("blockPalette", True)
        self._theme = theme
        self._catalog = catalog if catalog is not None else BlockCatalog()
        self._cards: list[tuple[_BlockCard, str]] = []
        self._headers: list[tuple[QLabel, list[_BlockCard]]] = []

        gap = theme.metric("spacing_sm")
        outer = QVBoxLayout(self)
        # No right margin: the scroll bar hugs the panel's right edge instead of leaving
        # a band of dead panel to its right («справа от ползунка осталось ненужное
        # пространство»). The scroll area shrinks its content off the bar when it shows,
        # so cards never hide under it.
        outer.setContentsMargins(gap, gap, 0, gap)
        outer.setSpacing(gap)

        self._search = SearchField(placeholder="Поиск блока", theme=theme)
        self._search.textChanged.connect(self._apply_filter)
        outer.addWidget(self._search)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # The global thumb keeps a large top/bottom margin so it stays clear of a card's
        # rounded corner; this frameless, transparent palette has no such corner, so that
        # margin only made the thumb look short and «кривой». A geometry-only override
        # runs it near the full track height (2 px all round); the gradient, radius and
        # min-height stay inherited from the global scroll-bar rule.
        self._scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar::handle:vertical { margin: 2px; }"
        )
        self._scroll.viewport().setAutoFillBackground(False)
        content = QWidget()
        content.setProperty("transparent", True)
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(2)
        self._scroll.setWidget(content)
        outer.addWidget(self._scroll, 1)

        self._build()
        theme.theme_changed.connect(self._on_theme_changed)

    def _build(self) -> None:
        for category in self._catalog.list_categories():
            blocks = self._catalog.list_blocks(category.type)
            if not blocks:
                continue
            header = QLabel(category.title_ru.upper())
            header.setProperty("catTitle", True)
            self._content_layout.addWidget(header)
            group_cards: list[_BlockCard] = []
            for block in blocks:
                card = _BlockCard(block, role_of(block.type, self._catalog), self._theme)
                card.chosen.connect(self.block_chosen)
                self._content_layout.addWidget(card)
                haystack = f"{block.title_ru}\n{block.description_ru}\n{block.type}".casefold()
                self._cards.append((card, haystack))
                group_cards.append(card)
            self._headers.append((header, group_cards))
        self._content_layout.addStretch(1)

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().casefold()
        matches = {card: (not needle or needle in hay) for card, hay in self._cards}
        for card, _hay in self._cards:
            card.setVisible(matches[card])
        for header, cards in self._headers:
            header.setVisible(any(matches[card] for card in cards))

    def _on_theme_changed(self, _theme: object) -> None:
        for card, _hay in self._cards:
            card.refresh_icon()
