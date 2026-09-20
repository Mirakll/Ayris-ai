"""The block tree of a command as an ordered, nestable, drag-and-drop list.

The heart of the editor. :class:`ActionListModel` is the pure part — it owns the
working :class:`CommandModel` and edits ``model.actions`` in place: it flattens the
tree into display rows, inserts a block from the palette, moves a block into or out of
a branch, removes with a one-step undo, duplicates, and toggles a block's ``enabled``
and ``comment``. Every position is the task's block path — ``("actions", 1, "then",
0)`` — so a move recomputes the path the way the validator and the node view of task
53 read it, and no second representation of the tree exists.

A path here is split in two: the *container* (where a list of sibling blocks lives,
e.g. ``("actions",)`` or ``("actions", 1, "then")``) and the *index* within it. That
split is what makes a move unambiguous — drop onto row *n* of a branch is «this
container, this index», and dragging a block down its own list adjusts for the hole it
leaves behind.

:class:`ActionListView` is the thin Qt tree over the model: indentation and connector
lines show nesting, a dragged row or a dropped palette block calls the model, and the
row widgets carry the enable toggle, comment and per-row menu. The view holds no tree
state of its own; it rebuilds from the model's rows after every change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QInputDialog,
    QMenu,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import LOGIC_BLOCKS, ActionBlock, CommandModel
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.block_palette import BLOCK_MIME

__all__ = [
    "BRANCH_LABELS",
    "ActionListModel",
    "ActionListView",
    "ActionRow",
    "BlockPath",
]

#: A container path: where a list of sibling blocks lives. ``("actions",)`` is the top
#: level; ``("actions", 1, "then")`` is the ``then`` branch of the second block.
BlockPath = tuple[object, ...]

#: Russian labels for the branch a nested list belongs to, shown as a group heading.
BRANCH_LABELS: Final[dict[str, str]] = {
    "then": "Тогда",
    "else": "Иначе",
    "body": "Тело",
    "catch": "При ошибке",
}

#: The schema field behind each wire branch name (``else`` is stored as ``else_``).
_BRANCH_FIELDS: Final[dict[str, str]] = {
    "then": "then",
    "else": "else_",
    "body": "body",
    "catch": "catch",
}

_MAX_DEPTH: Final = 16


@dataclass(frozen=True, slots=True)
class ActionRow:
    """One row of the flattened tree: a block, where it sits, and how deep."""

    block: ActionBlock
    container: BlockPath
    index: int
    depth: int
    #: The wire branch name this row's container is (``then``/``else``/…), or ``""``
    #: for a top-level row; set on the first child of a branch so the view can head it.
    branch: str = ""
    #: Whether this row is the first child of its branch — the view draws the heading.
    branch_head: bool = False

    @property
    def path(self) -> BlockPath:
        """The full block path, ``container + (index,)`` — ``actions[1].then[0]``."""
        return (*self.container, self.index)

    @property
    def path_text(self) -> str:
        parts: list[str] = []
        for step in self.path:
            if isinstance(step, int):
                parts.append(f"[{step}]")
            elif parts:
                parts.append(f".{step}")
            else:
                parts.append(str(step))
        return "".join(parts)


@dataclass(slots=True)
class _Undo:
    """One removed block and where it was, so a single delete can be taken back."""

    block: ActionBlock
    container: BlockPath
    index: int


class ActionListModel:
    """Ordered, nestable list of action blocks over a command's ``actions``.

    Not a ``QAbstractItemModel``: the tree is small, the view rebuilds cheaply, and a
    plain object is what the tests drive without a widget. The view listens to nothing
    here — it calls a mutator and then reads :meth:`rows` again.
    """

    def __init__(self, command: CommandModel | None = None) -> None:
        self._command: CommandModel | None = command
        self._undo: _Undo | None = None
        self._clipboard: ActionBlock | None = None

    # -- loading ------------------------------------------------------------

    def set_command(self, command: CommandModel) -> None:
        self._command = command
        self._undo = None

    @property
    def command(self) -> CommandModel | None:
        return self._command

    # -- reading ------------------------------------------------------------

    def rows(self) -> list[ActionRow]:
        """The whole tree flattened depth-first, branch headings marked.

        Order is exactly the order the view draws: a block, then each of its declared
        branches in schema order, each branch's blocks recursively. Empty branches of a
        logic block still emit a heading row's marker through ``branch_head`` on... —
        rather, an empty branch emits nothing, and the view shows the branch as a drop
        target by the parent's presence.
        """
        if self._command is None:
            return []
        rows: list[ActionRow] = []
        self._walk(self._command.actions, ("actions",), 0, rows)
        return rows

    def _walk(
        self,
        blocks: Sequence[ActionBlock],
        container: BlockPath,
        depth: int,
        rows: list[ActionRow],
    ) -> None:
        for index, block in enumerate(blocks):
            rows.append(ActionRow(block=block, container=container, index=index, depth=depth))
            spec = LOGIC_BLOCKS.get(block.type)
            if spec is None:
                continue
            for wire in spec.branches:
                child_container = (*container, index, wire)
                children = self._branch_list(block, wire)
                for child_index, child in enumerate(children):
                    child_rows: list[ActionRow] = []
                    self._walk([child], child_container, depth + 1, child_rows)
                    # Re-key the single child at its real index, mark the branch head.
                    head = child_rows[0]
                    child_rows[0] = ActionRow(
                        block=head.block,
                        container=child_container,
                        index=child_index,
                        depth=depth + 1,
                        branch=wire,
                        branch_head=child_index == 0,
                    )
                    rows.extend(child_rows)

    def branches_of(self, block: ActionBlock) -> tuple[str, ...]:
        """The wire branch names a block actually declares, in schema order."""
        spec = LOGIC_BLOCKS.get(block.type)
        return spec.branches if spec is not None else ()

    # -- mutation -----------------------------------------------------------

    def insert(self, block: ActionBlock, container: BlockPath, index: int) -> BlockPath | None:
        """Insert a block at ``container[index]``; returns its new path, or ``None``.

        Refused (``None``, no mutation) when the block's own branches would push the
        tree past :data:`MAX_BLOCK_DEPTH` — the same ceiling the schema enforces, so a
        drop that could not be saved is stopped here instead of raising on the next
        edit through :meth:`_touch`.
        """
        if self._depth_of(container) + self._subtree_height(block) > _MAX_DEPTH:
            return None
        target = self._resolve(container)
        clamped = max(0, min(index, len(target)))
        target.insert(clamped, block)
        self._touch()
        return (*container, clamped)

    def insert_type(self, block_type: str, container: BlockPath, index: int) -> BlockPath | None:
        """Insert a fresh default block of ``block_type`` (the palette's drop)."""
        return self.insert(ActionBlock(type=block_type), container, index)

    def remove(self, container: BlockPath, index: int) -> ActionBlock | None:
        """Remove and return the block at a position, remembering it for undo."""
        target = self._resolve(container)
        if not 0 <= index < len(target):
            return None
        block = target.pop(index)
        self._undo = _Undo(block=block, container=container, index=index)
        self._touch()
        return block

    def undo_remove(self) -> bool:
        """Put the last removed block back where it was, if nothing moved since."""
        if self._undo is None:
            return False
        target = self._resolve(self._undo.container)
        index = max(0, min(self._undo.index, len(target)))
        target.insert(index, self._undo.block)
        self._undo = None
        self._touch()
        return True

    def can_undo(self) -> bool:
        return self._undo is not None

    def move(
        self,
        source_container: BlockPath,
        source_index: int,
        dest_container: BlockPath,
        dest_index: int,
    ) -> BlockPath | None:
        """Move a block to another position, possibly into or out of a branch.

        Refuses to drop a block inside its own subtree — that would detach it from the
        tree — and refuses a move that would nest deeper than the schema allows. The
        destination index is corrected when the block is dragged further down its own
        container, because removing it first shifts every later sibling up by one.
        Returns the block's new path, or ``None`` when the move was refused.
        """
        if self._is_inside(source_container, source_index, dest_container):
            return None
        block = self._peek(source_container, source_index)
        if block is None:
            return None
        if self._depth_of(dest_container) + self._subtree_height(block) > _MAX_DEPTH:
            return None
        same_container = source_container == dest_container
        adjusted_index = (
            dest_index - 1 if same_container and dest_index > source_index else dest_index
        )
        # Removing the source shifts every later sibling — and any destination path
        # that descends through one of them — up by one. Correct the destination
        # container before resolving it against the mutated tree.
        target_container = self._shift_after_removal(dest_container, source_container, source_index)
        self._resolve(source_container).pop(source_index)
        dest = self._resolve(target_container)
        clamped = max(0, min(adjusted_index, len(dest)))
        dest.insert(clamped, block)
        self._undo = None
        self._touch()
        return (*target_container, clamped)

    @staticmethod
    def _shift_after_removal(
        dest_container: BlockPath, source_container: BlockPath, source_index: int
    ) -> BlockPath:
        """The destination container as it reads after the source is popped."""
        cut = len(source_container)
        if len(dest_container) <= cut or dest_container[:cut] != source_container:
            return dest_container
        step = dest_container[cut]
        if isinstance(step, int) and step > source_index:
            return (*dest_container[:cut], step - 1, *dest_container[cut + 1 :])
        return dest_container

    def duplicate(self, container: BlockPath, index: int) -> BlockPath | None:
        """Insert a deep copy of a block right after it."""
        block = self._peek(container, index)
        if block is None:
            return None
        return self.insert(block.model_copy(deep=True), container, index + 1)

    def copy(self, container: BlockPath, index: int) -> bool:
        """Hold a deep copy of a block for a later :meth:`paste`."""
        block = self._peek(container, index)
        if block is None:
            return False
        self._clipboard = block.model_copy(deep=True)
        return True

    def paste(self, container: BlockPath, index: int) -> BlockPath | None:
        """Insert the held block, if any — the copy-paste-between-commands path."""
        if self._clipboard is None:
            return None
        return self.insert(self._clipboard.model_copy(deep=True), container, index)

    def has_clipboard(self) -> bool:
        return self._clipboard is not None

    def set_enabled(self, container: BlockPath, index: int, *, enabled: bool) -> None:
        block = self._peek(container, index)
        if block is not None:
            block.enabled = enabled
            self._touch()

    def set_comment(self, container: BlockPath, index: int, comment: str) -> None:
        block = self._peek(container, index)
        if block is not None:
            block.comment = comment.strip()
            self._touch()

    def block_at(self, path: BlockPath) -> ActionBlock | None:
        """The block at a full path (``container + index``), or ``None``."""
        if not path:
            return None
        container, index = path[:-1], path[-1]
        if not isinstance(index, int):
            return None
        return self._peek(container, index)

    # -- internals ----------------------------------------------------------

    def _resolve(self, container: BlockPath) -> list[ActionBlock]:
        """The mutable list of blocks a container path points at."""
        if self._command is None:
            raise RuntimeError("action list model has no command")
        if container == ("actions",):
            return self._command.actions
        blocks = self._command.actions
        current: ActionBlock | None = None
        # Skip the leading "actions"; walk index/branch pairs.
        steps = container[1:]
        i = 0
        while i < len(steps):
            index = steps[i]
            assert isinstance(index, int)
            current = blocks[index]
            wire = steps[i + 1]
            assert isinstance(wire, str)
            blocks = self._branch_list(current, wire)
            i += 2
        return blocks

    @staticmethod
    def _branch_list(block: ActionBlock, wire: str) -> list[ActionBlock]:
        branch: list[ActionBlock] = getattr(block, _BRANCH_FIELDS[wire])
        return branch

    def _peek(self, container: BlockPath, index: int) -> ActionBlock | None:
        target = self._resolve(container)
        return target[index] if 0 <= index < len(target) else None

    def _is_inside(
        self, source_container: BlockPath, source_index: int, dest_container: BlockPath
    ) -> bool:
        """Whether ``dest_container`` lies within the moved block's own subtree."""
        block_path = (*source_container, source_index)
        return dest_container[: len(block_path)] == block_path

    @staticmethod
    def _depth_of(container: BlockPath) -> int:
        """How deep a container is: ``("actions",)`` is 0, each branch adds one."""
        return sum(1 for step in container if isinstance(step, str) and step != "actions")

    def _subtree_height(self, block: ActionBlock) -> int:
        """The number of branch levels below a block, plus one for itself."""
        spec = LOGIC_BLOCKS.get(block.type)
        if spec is None:
            return 1
        deepest = 0
        for wire in spec.branches:
            for child in self._branch_list(block, wire):
                deepest = max(deepest, self._subtree_height(child))
        return 1 + deepest

    def _touch(self) -> None:
        """Re-key ``actions`` through the model so validation runs on every edit.

        Assigning the list back to the validating field turns a mutation the editor
        made directly on a nested list into a real ``CommandModel`` assignment, so a
        block put somewhere the schema forbids is caught here and not at save.
        """
        if self._command is not None:
            self._command.actions = self._command.actions


# ----------------------------------------------------------------------
# the Qt view
# ----------------------------------------------------------------------

_CONTAINER_ROLE = Qt.ItemDataRole.UserRole
_INDEX_ROLE = Qt.ItemDataRole.UserRole + 1


class ActionListView(QTreeWidget):
    """Drag-and-drop tree of a command's action blocks over an :class:`ActionListModel`.

    A thin view: it keeps no tree state of its own and rebuilds its rows from
    :meth:`ActionListModel.rows` after every change, mapping a Qt item back to a block
    by the container path and index it stores. Indentation shows nesting; a branch's
    first child carries its heading («Тогда», «Иначе»…). A row is dragged to reorder or
    to move into/out of a branch, and a block dragged from the palette (the
    :data:`~ayris.gui.widgets.block_palette.BLOCK_MIME` mime) drops in as a new block.
    The per-row menu covers comment, enable/disable, duplicate, copy, paste and
    delete-with-undo — all delegated to the model.
    """

    #: Emitted with the selected block's full path, or an empty tuple on deselect.
    block_selected = Signal(tuple)
    #: Emitted after any structural or in-place change the view made through the model.
    changed = Signal()

    def __init__(
        self,
        model: ActionListModel,
        theme: ThemeManager,
        *,
        catalog: BlockCatalog | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._theme = theme
        self._catalog = catalog if catalog is not None else BlockCatalog()
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setExpandsOnDoubleClick(False)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)
        self.itemSelectionChanged.connect(self._on_selection)
        self.itemChanged.connect(self._on_item_changed)
        self._suppress_item_changed = False
        self.rebuild()

    # -- building -----------------------------------------------------------

    def rebuild(self) -> None:
        """Redraw the whole tree from the model, keeping the selected path."""
        selected = self.selected_path()
        self._suppress_item_changed = True
        self.clear()
        parents: dict[BlockPath, QTreeWidgetItem] = {}
        for row in self._model.rows():
            item = self._make_item(row)
            owner = self._owner_item(row.container, parents)
            if owner is None:
                self.addTopLevelItem(item)
            else:
                owner.addChild(item)
            parents[row.path] = item
            item.setExpanded(True)
        self._suppress_item_changed = False
        if selected:
            self._select_path(selected)

    def _owner_item(
        self, container: BlockPath, parents: dict[BlockPath, QTreeWidgetItem]
    ) -> QTreeWidgetItem | None:
        """The item a row hangs under: the block that owns its branch, or the root."""
        if container == ("actions",):
            return None
        # container == (*block_path, wire) → parent item is the block at block_path.
        return parents.get(container[:-1])

    def _make_item(self, row: ActionRow) -> QTreeWidgetItem:
        meta = self._catalog.try_get(row.block.type)
        title = meta.title_ru if meta is not None else row.block.type
        prefix = f"[{BRANCH_LABELS.get(row.branch, row.branch)}] " if row.branch_head else ""
        label = f"{prefix}{title}"
        if not row.block.enabled:
            label = f"{label} (выкл.)"
        if row.block.comment:
            label = f"{label} — {row.block.comment}"
        item = QTreeWidgetItem([label])
        item.setData(0, _CONTAINER_ROLE, row.container)
        item.setData(0, _INDEX_ROLE, row.index)
        item.setFlags(
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsDragEnabled
            | Qt.ItemFlag.ItemIsDropEnabled
            | Qt.ItemFlag.ItemIsUserCheckable
        )
        item.setCheckState(
            0, Qt.CheckState.Checked if row.block.enabled else Qt.CheckState.Unchecked
        )
        tooltip = row.path_text
        if meta is not None and meta.is_dangerous:
            tooltip = f"⚠ Опасный блок. {tooltip}"
        item.setToolTip(0, tooltip)
        return item

    # -- selection ----------------------------------------------------------

    def selected_path(self) -> BlockPath:
        items = self.selectedItems()
        return self._path_of(items[0]) if items else ()

    def _path_of(self, item: QTreeWidgetItem) -> BlockPath:
        container = item.data(0, _CONTAINER_ROLE)
        index = item.data(0, _INDEX_ROLE)
        if container is None or index is None:
            return ()
        return (*container, index)

    def select_path(self, path: BlockPath) -> None:
        """Select the row at a block path, if it is still there after a rebuild."""
        self._select_path(path)

    def _select_path(self, path: BlockPath) -> None:
        match = self._find_item(path)
        if match is not None:
            self.setCurrentItem(match)

    def _find_item(self, path: BlockPath) -> QTreeWidgetItem | None:
        stack = [self.topLevelItem(i) for i in range(self.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if self._path_of(item) == path:
                return item
            stack.extend(item.child(i) for i in range(item.childCount()))
        return None

    def _on_selection(self) -> None:
        self.block_selected.emit(self.selected_path())

    def _on_item_changed(self, item: QTreeWidgetItem, _column: int) -> None:
        if self._suppress_item_changed:
            return
        path = self._path_of(item)
        if not path:
            return
        container, index = path[:-1], path[-1]
        assert isinstance(index, int)
        enabled = item.checkState(0) == Qt.CheckState.Checked
        self._model.set_enabled(container, index, enabled=enabled)
        self.changed.emit()

    # -- drag and drop ------------------------------------------------------

    def dragEnterEvent(self, event: QDropEvent) -> None:  # noqa: N802 — Qt override.
        if event.mimeData().hasFormat(BLOCK_MIME):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)  # type: ignore[arg-type]

    def dragMoveEvent(self, event: QDropEvent) -> None:  # noqa: N802 — Qt override.
        if event.mimeData().hasFormat(BLOCK_MIME):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)  # type: ignore[arg-type]

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 — Qt override.
        target = self.itemAt(event.position().toPoint())
        container, index = self._drop_target(target)
        mime = event.mimeData()
        if mime.hasFormat(BLOCK_MIME):
            block_type = bytes(mime.data(BLOCK_MIME).data()).decode("utf-8")
            new_path = self._model.insert_type(block_type, container, index)
            self._finish_drop(event, new_path)
            return
        source_items = self.selectedItems()
        if not source_items:
            event.ignore()
            return
        source_path = self._path_of(source_items[0])
        if not source_path:
            event.ignore()
            return
        src_container, src_index = source_path[:-1], source_path[-1]
        assert isinstance(src_index, int)
        new_path = self._model.move(src_container, src_index, container, index)
        self._finish_drop(event, new_path)

    def _finish_drop(self, event: QDropEvent, new_path: BlockPath | None) -> None:
        if new_path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.rebuild()
        self._select_path(new_path)
        self.changed.emit()

    def _drop_target(self, item: QTreeWidgetItem | None) -> tuple[BlockPath, int]:
        """Where a drop lands: a container path and an index within it.

        Dropping onto a logic block's row targets the first branch's start, so a block
        can be dragged straight into an ``If``. Dropping below a row lands after it in
        that row's own container; onto or above, before it.
        """
        if item is None:
            root: BlockPath = ("actions",)
            return root, self._root_len()
        path = self._path_of(item)
        container, index = path[:-1], path[-1]
        assert isinstance(index, int)
        block = self._model.block_at(path)
        indicator = self.dropIndicatorPosition()
        if indicator == QAbstractItemView.DropIndicatorPosition.OnItem and block is not None:
            branches = self._model.branches_of(block)
            if branches:
                return (*path, branches[0]), 0
        if indicator == QAbstractItemView.DropIndicatorPosition.BelowItem:
            return container, index + 1
        return container, index

    # -- context menu -------------------------------------------------------

    def _show_menu(self, point: QPoint) -> None:
        item = self.itemAt(point)
        menu = QMenu(self)
        if item is not None:
            path = self._path_of(item)
            container, index = path[:-1], path[-1]
            assert isinstance(index, int)
            block = self._model.block_at(path)
            menu.addAction("Комментарий…", lambda: self._edit_comment(container, index))
            menu.addAction(
                "Дублировать", lambda: self._act(self._model.duplicate, container, index)
            )
            menu.addAction("Копировать", lambda: self._model.copy(container, index))
            menu.addAction(
                "Вставить после", lambda: self._act(self._model.paste, container, index + 1)
            )
            enable_label = "Выключить" if block is not None and block.enabled else "Включить"
            menu.addAction(enable_label, lambda: self._toggle_enabled(container, index, block))
            menu.addSeparator()
            menu.addAction("Удалить", lambda: self._delete(container, index))
        if self._model.has_clipboard():
            menu.addAction(
                "Вставить в конец",
                lambda: self._act(self._model.paste, ("actions",), self._root_len()),
            )
        if self._model.can_undo():
            menu.addAction("Отменить удаление", self._undo)
        menu.exec(self.viewport().mapToGlobal(point))

    def _edit_comment(self, container: BlockPath, index: int) -> None:
        block = self._model.block_at((*container, index))
        current = block.comment if block is not None else ""
        text, ok = QInputDialog.getText(self, "Комментарий блока", "Комментарий:", text=current)
        if ok:
            self._model.set_comment(container, index, text)
            self._after_change((*container, index))

    def _toggle_enabled(self, container: BlockPath, index: int, block: ActionBlock | None) -> None:
        if block is not None:
            self._model.set_enabled(container, index, enabled=not block.enabled)
            self._after_change((*container, index))

    def _delete(self, container: BlockPath, index: int) -> None:
        self._model.remove(container, index)
        self._after_change(())

    def _undo(self) -> None:
        self._model.undo_remove()
        self._after_change(())

    def _act(self, method: object, container: BlockPath, index: int) -> None:
        assert callable(method)
        new_path = method(container, index)
        self._after_change(new_path if isinstance(new_path, tuple) else ())

    def _after_change(self, select: BlockPath) -> None:
        self.rebuild()
        if select:
            self._select_path(select)
        self.changed.emit()

    def _root_len(self) -> int:
        return len(self._model._resolve(("actions",)))
