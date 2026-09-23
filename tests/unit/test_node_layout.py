"""Авто-раскладка графа нод (задача 53), без Qt.

Раскладка чистая — она пишет ``x``/``y`` в ноды, окна не нужно. Главное
утверждение: после :func:`auto_layout` никакие две ноды не перекрываются, какой бы
формы ни был граф (цепочка, ветвление, орфаны). Отдельно проверяется поток
слева-направо (у детей слой больше родителя) и привязка к сетке.
"""

from __future__ import annotations

import pytest

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.gui.widgets.node_editor.bridge import (
    MAIN_PORT,
    GraphEdge,
    GraphNode,
    NodeGraph,
    NodeRole,
    graph_from_command,
)
from ayris.gui.widgets.node_editor.layout import (
    DEFAULT_ORIGIN,
    GRID,
    NODE_HEIGHT,
    NODE_WIDTH,
    auto_layout,
    free_slot,
    snap_to_grid,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def catalog() -> BlockCatalog:
    return BlockCatalog()


def _overlap(a: GraphNode, b: GraphNode) -> bool:
    """Пересекаются ли карточки двух нод как прямоугольники ``NODE_WIDTH×NODE_HEIGHT``."""
    return (
        a.x < b.x + NODE_WIDTH
        and b.x < a.x + NODE_WIDTH
        and a.y < b.y + NODE_HEIGHT
        and b.y < a.y + NODE_HEIGHT
    )


def _assert_no_overlaps(graph: NodeGraph) -> None:
    nodes = graph.nodes
    for i, first in enumerate(nodes):
        for second in nodes[i + 1 :]:
            assert not _overlap(first, second), f"{first.id} перекрывает {second.id}"


def _chain(count: int) -> NodeGraph:
    nodes = [
        GraphNode(id=str(i), block=ActionBlock(type="Say"), role=NodeRole.ACTION)
        for i in range(count)
    ]
    edges = [GraphEdge(str(i), MAIN_PORT, str(i + 1)) for i in range(count - 1)]
    return NodeGraph(nodes=nodes, edges=edges, root_id="0" if count else None)


def _wide_command() -> CommandModel:
    return CommandModel(
        name="Широкая",
        actions=[
            ActionBlock(
                type="If",
                params={"condition": "{x}"},
                then=[
                    ActionBlock(type="Say", params={"text": "1"}),
                    ActionBlock(type="Say", params={"text": "2"}),
                ],
                else_=[
                    ActionBlock(type="Say", params={"text": "3"}),
                    ActionBlock(type="Say", params={"text": "4"}),
                ],
            ),
            ActionBlock(
                type="Try",
                body=[ActionBlock(type="Say", params={"text": "5"})],
                catch=[ActionBlock(type="Say", params={"text": "6"})],
            ),
        ],
    )


# ----------------------------------------------------------------------
# нет перекрытий на разных формах графа
# ----------------------------------------------------------------------


def test_empty_graph_is_a_noop() -> None:
    graph = NodeGraph()
    auto_layout(graph)  # не должно падать на пустом графе
    assert graph.nodes == []


def test_chain_has_no_overlaps() -> None:
    graph = _chain(6)
    auto_layout(graph)
    _assert_no_overlaps(graph)


def test_branching_command_has_no_overlaps(catalog: BlockCatalog) -> None:
    graph = graph_from_command(_wide_command(), catalog=catalog)
    auto_layout(graph)
    _assert_no_overlaps(graph)


def test_disconnected_nodes_have_no_overlaps() -> None:
    # Три ноды без единой связи — все в слое ноль, но всё равно не наезжают друг на друга.
    nodes = [
        GraphNode(id=str(i), block=ActionBlock(type="Say"), role=NodeRole.ACTION) for i in range(3)
    ]
    graph = NodeGraph(nodes=nodes, edges=[], root_id="0")
    auto_layout(graph)
    _assert_no_overlaps(graph)


# ----------------------------------------------------------------------
# поток слева-направо: слой ребёнка больше слоя родителя
# ----------------------------------------------------------------------


def test_flow_is_left_to_right() -> None:
    graph = _chain(4)
    auto_layout(graph)
    xs = [graph.node_by_id(str(i)).x for i in range(4)]
    # Каждый следующий блок строго правее предыдущего.
    assert xs == sorted(xs)
    assert len(set(xs)) == 4


def test_branch_children_are_right_of_parent(catalog: BlockCatalog) -> None:
    graph = graph_from_command(_wide_command(), catalog=catalog)
    auto_layout(graph)
    parent = graph.node_by_id("actions[0]")
    then_child = graph.node_by_id("actions[0].then[0]")
    else_child = graph.node_by_id("actions[0].else[0]")
    assert then_child.x > parent.x
    assert else_child.x > parent.x
    # Ветки то и иначе расходятся по вертикали, а не садятся друг на друга.
    assert then_child.y != else_child.y


# ----------------------------------------------------------------------
# привязка к сетке
# ----------------------------------------------------------------------


def test_snap_to_grid_rounds_to_nearest_intersection() -> None:
    x, y = snap_to_grid(GRID * 2 + 3, GRID * 5 - 2)
    assert x == GRID * 2
    assert y == GRID * 5


def test_snap_to_grid_custom_step() -> None:
    assert snap_to_grid(23, 57, grid=10) == (20, 60)


# ----------------------------------------------------------------------
# free_slot: место для новой ноды — в потоке рядом с якорем, не поверх других
# ----------------------------------------------------------------------


def _card_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) < NODE_WIDTH and abs(a[1] - b[1]) < NODE_HEIGHT


def test_free_slot_without_anchor_is_the_default_origin() -> None:
    # Первый блок пустой команды: якоря нет — нода садится в начало сетки, не в (0,0)
    # поверх воображаемого центра вьюпорта.
    assert free_slot(None, []) == DEFAULT_ORIGIN


def test_free_slot_places_one_step_right_of_a_free_anchor() -> None:
    # Якорь на месте, справа пусто — новая нода продолжает поток вправо.
    slot = free_slot((60.0, 120.0), [(60.0, 120.0)])
    assert slot[0] > 60.0 and slot[1] == 120.0


def test_free_slot_nudges_down_when_the_next_cell_is_taken() -> None:
    # Классический баг: вставка между двумя нодами. Место справа от якоря занято
    # следующим блоком — новая нода уходит вниз, а не садится ему на голову.
    anchor = (60.0, 120.0)
    neighbour = (330.0, 120.0)  # тот, кто уже занял «следующую» клетку
    slot = free_slot(anchor, [anchor, neighbour])
    assert not _card_overlap(slot, anchor)
    assert not _card_overlap(slot, neighbour)


def test_free_slot_never_lands_on_any_occupied_card() -> None:
    # Плотная колонка справа от якоря — слот всё равно находит свободное место.
    anchor = (60.0, 120.0)
    occupied = [anchor] + [(330.0, 120.0 + i * 170.0) for i in range(4)]
    slot = free_slot(anchor, occupied)
    assert all(not _card_overlap(slot, spot) for spot in occupied)
