"""The blocks that are not actions, and the interface the interpreter runs them behind.

Two files, split the way section 7 splits them: :mod:`~ayris.actions.macros.blocks.logic`
decides what runs next, :mod:`~ayris.actions.macros.blocks.variables` reads and writes the
command's own data. :data:`BLOCK_HANDLERS` is both of them in one table, which is what the
engine dispatches on — and what a test compares against
:data:`~ayris.actions.macros.schema.LOGIC_BLOCKS`, so a block added to the language without
an implementation fails a test instead of being quietly taken for the name of an action.

Adding a block is a function in one of the two files and a line in its table. Nothing in
the engine changes, because the engine no longer knows what the blocks are — only that they
take a :class:`~ayris.actions.macros.blocks.logic.BlockRuntime` and answer with a
:class:`~ayris.actions.macros.blocks.logic.Flow`.
"""

from __future__ import annotations

from typing import Final

from ayris.actions.macros.blocks.logic import (
    LOGIC_HANDLERS,
    BlockHandler,
    BlockRuntime,
    Flow,
    as_dict,
    as_int,
    as_list,
    error_text,
    matches,
    sequence,
    short,
    split_items,
)
from ayris.actions.macros.blocks.variables import (
    VARIABLE_HANDLERS,
    as_scope,
    element,
    member,
    store_read,
    variable_name,
)

__all__ = [
    "BLOCK_HANDLERS",
    "LOGIC_HANDLERS",
    "VARIABLE_HANDLERS",
    "BlockHandler",
    "BlockRuntime",
    "Flow",
    "as_dict",
    "as_int",
    "as_list",
    "as_scope",
    "element",
    "error_text",
    "matches",
    "member",
    "sequence",
    "short",
    "split_items",
    "store_read",
    "variable_name",
]

#: Which function runs which block. A table and not a chain of ``elif`` for one reason: the
#: keys have to be exactly :data:`~ayris.actions.macros.schema.LOGIC_BLOCKS`, and a table
#: can be compared with it.
BLOCK_HANDLERS: Final[dict[str, BlockHandler]] = {**LOGIC_HANDLERS, **VARIABLE_HANDLERS}
