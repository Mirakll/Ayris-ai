"""The block palette: the task-33 catalog as a searchable, draggable tree.

Blocks are grouped by the catalog's own categories, each with its Russian title and
icon glyph. A search box filters by title and description across all categories. A
block is added by double-click (the palette emits :attr:`block_chosen`) or dragged
onto the action list — the drag carries the block type as text, the same mime the
action list reads, so a drop lands a new block of that type.

Dangerous blocks (``is_dangerous``) and unavailable actions (``available`` false in
this build) are marked in the row: the danger glyph and a muted, disabled row with the
reason in its tooltip. The palette never builds a command or touches the database; it
only names what the editor may insert.
"""

from __future__ import annotations

from PySide6.QtCore import QMimeData, Qt, Signal
from PySide6.QtGui import QColor, QDrag
from PySide6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ayris.actions.macros.blocks.catalog import BlockCatalog, BlockMeta
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.search_field import SearchField

__all__ = ["BLOCK_MIME", "BlockPalette"]

#: The mime type a dragged palette block carries; the action list drop reads it.
BLOCK_MIME = "application/x-ayris-block-type"

#: Roles the tree items carry, above the display text.
_TYPE_ROLE = Qt.ItemDataRole.UserRole
_HAYSTACK_ROLE = Qt.ItemDataRole.UserRole + 1


class _PaletteTree(QTreeWidget):
    """A tree that starts a text drag of the selected block's type."""

    def startDrag(self, _actions: Qt.DropAction) -> None:  # noqa: N802 — Qt override.
        item = self.currentItem()
        block_type = item.data(0, _TYPE_ROLE)
        if not block_type:
            return
        mime = QMimeData()
        mime.setData(BLOCK_MIME, str(block_type).encode("utf-8"))
        mime.setText(str(block_type))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class BlockPalette(QWidget):
    """Searchable, draggable catalog of blocks the editor can insert."""

    #: Emitted with the block type when a block is chosen (double-click / Enter).
    block_chosen = Signal(str)

    def __init__(
        self,
        theme: ThemeManager,
        *,
        catalog: BlockCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setProperty("transparent", True)
        self._theme = theme
        self._catalog = catalog if catalog is not None else BlockCatalog()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.metric("spacing_sm"))

        self._search = SearchField(placeholder="Поиск блока", theme=theme)
        self._search.textChanged.connect(self._apply_filter)
        outer.addWidget(self._search)

        self._tree = _PaletteTree()
        self._tree.setHeaderHidden(True)
        self._tree.setDragEnabled(True)
        self._tree.setDragDropMode(QTreeWidget.DragDropMode.DragOnly)
        self._tree.itemActivated.connect(self._on_activated)
        self._tree.itemDoubleClicked.connect(self._on_activated)
        outer.addWidget(self._tree)

        self._build()

    # -- building -----------------------------------------------------------

    def _build(self) -> None:
        self._tree.clear()
        for category in self._catalog.list_categories():
            parent = QTreeWidgetItem([f"{category.icon}  {category.title_ru}"])
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._tree.addTopLevelItem(parent)
            parent.setExpanded(True)
            for block in self._catalog.list_blocks(category.type):
                parent.addChild(self._make_item(block))

    def _make_item(self, block: BlockMeta) -> QTreeWidgetItem:
        label = f"{block.icon}  {block.title_ru}"
        if block.is_dangerous:
            label = f"⚠ {label}"
        item = QTreeWidgetItem([label])
        item.setData(0, _TYPE_ROLE, block.type)
        haystack = f"{block.title_ru}\n{block.description_ru}\n{block.type}".casefold()
        item.setData(0, _HAYSTACK_ROLE, haystack)
        tooltip = block.description_ru
        if not block.available:
            item.setFlags(Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(0, self._muted())
            tooltip = block.unavailable_reason or "Действие недоступно в этой сборке."
        elif block.is_dangerous:
            tooltip = f"Опасный блок. {tooltip}".strip()
        item.setToolTip(0, tooltip)
        return item

    def _muted(self) -> QColor:
        return QColor(self._theme.theme.color("text_muted"))

    # -- filtering ----------------------------------------------------------

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().casefold()
        for i in range(self._tree.topLevelItemCount()):
            category = self._tree.topLevelItem(i)
            visible_children = 0
            for j in range(category.childCount()):
                child = category.child(j)
                haystack = child.data(0, _HAYSTACK_ROLE) or ""
                match = not needle or needle in haystack
                child.setHidden(not match)
                visible_children += int(match)
            category.setHidden(visible_children == 0)
            category.setExpanded(bool(needle) or visible_children > 0)

    # -- events -------------------------------------------------------------

    def _on_activated(self, item: QTreeWidgetItem, _column: int = 0) -> None:
        block_type = item.data(0, _TYPE_ROLE)
        if not block_type:
            return
        if not (item.flags() & Qt.ItemFlag.ItemIsEnabled):
            return
        self.block_chosen.emit(str(block_type))
