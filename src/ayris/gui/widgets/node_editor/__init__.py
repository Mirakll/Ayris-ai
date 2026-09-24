"""Нодовый редактор команды (задача 53).

Второе представление той же :class:`~ayris.actions.macros.schema.CommandModel`, что и
список-режим задачи 52: блоки-ноды на :class:`QGraphicsView`, связи кубическими кривыми
Безье, ветвление ``If``/``Switch``/``While``/``Try`` с русскими подписями портов,
панорамирование и зум к курсору, авто-раскладка слева направо.

Слои разделены так, чтобы интересная половина проверялась без окна:

* :mod:`bridge` — чистая конвертация «дерево ↔ граф» (ноды и связи), роль→цвет, сводка
  блока, валидация связей и формат сохранения координат. Ни строчки Qt: round-trip и
  запрет невалидных связей покрываются offscreen-тестом.
* :mod:`layout` — послойная авто-раскладка слева направо без наложений и привязка к сетке.
* :mod:`scene`, :mod:`node_item`, :mod:`edge_item`, :mod:`view` — рисующая
  часть на ``QGraphicsScene``/``QGraphicsView``; offscreen-тест проверяет геометрию.
* :class:`NodeEditor` — виджет, который вешается на тот же ``ActionListModel``, что и
  список-режим, и переключается с ним без потери данных.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ayris.gui.widgets.node_editor.bridge import (
    GraphEdge,
    GraphNode,
    NodeGraph,
    NodeRole,
    can_connect,
    command_from_graph,
    graph_from_command,
    layout_from_json,
    layout_to_json,
    role_of,
)
from ayris.gui.widgets.node_editor.layout import auto_layout, snap_to_grid

if TYPE_CHECKING:
    # Give the type checker the real class; the runtime keeps the lazy import below so
    # the pure bridge/layout tests never drag in Qt just to touch this package.
    from ayris.gui.widgets.node_editor.editor import NodeEditor as NodeEditor

__all__ = [
    "GraphEdge",
    "GraphNode",
    "NodeEditor",
    "NodeGraph",
    "NodeRole",
    "auto_layout",
    "can_connect",
    "command_from_graph",
    "graph_from_command",
    "layout_from_json",
    "layout_to_json",
    "role_of",
    "snap_to_grid",
]


def __getattr__(name: str) -> object:
    """Lazily expose the Qt widget so importing the pure layer needs no display.

    :class:`NodeEditor` drags in ``QGraphicsView`` and the whole scene; the bridge and
    layout tests import this package for the pure functions and must not pay for Qt.
    """
    if name == "NodeEditor":
        from ayris.gui.widgets.node_editor.editor import NodeEditor

        return NodeEditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
