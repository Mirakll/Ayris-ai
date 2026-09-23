"""Авто-раскладка графа слева направо и привязка к сетке — без Qt.

Раскладка послойная: слой ноды — самый длинный путь к ней от корня по связям, так что
поток идёт слева направо, ветви расходятся по вертикали. Внутри слоя порядок берётся по
барицентру родителей, чтобы поменьше пересечений. Шаги по осям заведомо больше размера
ноды, поэтому две ноды никогда не встают друг на друга — это и проверяет offscreen-тест
по геометрии сцены. Привязка к сетке — опциональная помощь: по умолчанию ноды двигаются
свободно.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from ayris.gui.widgets.node_editor.bridge import NodeGraph

__all__ = [
    "DEFAULT_ORIGIN",
    "GRID",
    "NODE_HEIGHT",
    "NODE_WIDTH",
    "auto_layout",
    "free_slot",
    "snap_to_grid",
]

#: Node card size, mirrored from the mockup (`design/mockups/node_editor_mockup.html`).
NODE_WIDTH: Final = 210.0
NODE_HEIGHT: Final = 84.0

#: Grid step, the mockup's dotted grid; snapping rounds to it.
GRID: Final = 26.0

#: Layout spacing — both strictly larger than the node size, so laid-out cards never
#: overlap: distinct (layer, slot) pairs map to distinct, non-touching boxes.
_X_GAP: Final = 270.0
_Y_GAP: Final = 170.0
_X0: Final = 60.0
_Y0: Final = 120.0

#: Where the first node of an empty command lands — the top-left of the auto-layout grid,
#: so a hand-placed first block sits exactly where an auto pass would have put it.
DEFAULT_ORIGIN: Final[tuple[float, float]] = (_X0, _Y0)


def snap_to_grid(x: float, y: float, *, grid: float = GRID) -> tuple[float, float]:
    """Round a point to the nearest grid intersection."""
    return (round(x / grid) * grid, round(y / grid) * grid)


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Whether two node cards placed at these top-left corners would touch or overlap."""
    return abs(a[0] - b[0]) < NODE_WIDTH and abs(a[1] - b[1]) < NODE_HEIGHT


def free_slot(
    anchor: tuple[float, float] | None,
    occupied: list[tuple[float, float]],
    *,
    x_gap: float = _X_GAP,
    y_gap: float = _Y_GAP,
) -> tuple[float, float]:
    """A free coordinate for a new node, in flow next to ``anchor``, never on another card.

    Placement follows the left-to-right flow the auto-layout draws: a new node lands one
    ``x_gap`` to the right of its anchor (the block it was inserted after), the natural
    «next block» spot. If that cell is taken — a later sibling already sits there — it is
    nudged straight down by ``y_gap`` until it clears every occupied card, so a fresh node
    is always visible beside its anchor instead of stacking on the origin or on a neighbour.

    ``anchor`` is ``None`` only for the very first block of an empty command; it then lands
    at :data:`DEFAULT_ORIGIN`. Coordinates are node top-left corners, matching the scene.
    """
    candidate = DEFAULT_ORIGIN if anchor is None else (anchor[0] + x_gap, anchor[1])
    # Nudge down until the card touches nothing; the loop is bounded by the finite node
    # count (each step clears at least the highest blocker it passed), so it terminates.
    guard = 0
    while any(_overlaps(candidate, spot) for spot in occupied) and guard <= len(occupied):
        candidate = (candidate[0], candidate[1] + y_gap)
        guard += 1
    return candidate


def auto_layout(
    graph: NodeGraph,
    *,
    x_gap: float = _X_GAP,
    y_gap: float = _Y_GAP,
    x0: float = _X0,
    y0: float = _Y0,
) -> None:
    """Place every node in a left-to-right layered grid, writing ``x``/``y`` in place.

    Layers come from the longest path from the root along the edges; a node no edge
    reaches (a disconnected block) lands in layer zero. Within a layer nodes are ordered by
    the mean position of their parents to keep branch wires from crossing. The gaps are
    wider than a node, so the result has no overlaps regardless of graph shape.
    """
    if not graph.nodes:
        return
    layer = _layers(graph)
    order = {node.id: index for index, node in enumerate(graph.nodes)}
    by_layer: dict[int, list[str]] = defaultdict(list)
    for node in graph.nodes:
        by_layer[layer[node.id]].append(node.id)

    parents: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        parents[edge.target_id].append(edge.source_id)

    slot: dict[str, int] = {}
    for depth in sorted(by_layer):
        ids = by_layer[depth]
        if depth == 0:
            ids.sort(key=lambda nid: order[nid])
        else:
            ids.sort(key=lambda nid: (_barycentre(nid, parents, slot, order), order[nid]))
        for index, node_id in enumerate(ids):
            slot[node_id] = index

    for node in graph.nodes:
        depth = layer[node.id]
        node.x = x0 + depth * x_gap
        node.y = y0 + slot[node.id] * y_gap


def _layers(graph: NodeGraph) -> dict[str, int]:
    """Longest-path depth of each node from the root, following edges to a fixpoint."""
    depth: dict[str, int] = {node.id: 0 for node in graph.nodes}
    for _ in range(len(graph.nodes)):
        changed = False
        for edge in graph.edges:
            candidate = depth[edge.source_id] + 1
            if candidate > depth[edge.target_id]:
                depth[edge.target_id] = candidate
                changed = True
        if not changed:
            break
    return depth


def _barycentre(
    node_id: str,
    parents: dict[str, list[str]],
    slot: dict[str, int],
    order: dict[str, int],
) -> float:
    """The mean slot of a node's already-placed parents, for crossing-light ordering."""
    placed = [slot[pid] for pid in parents.get(node_id, ()) if pid in slot]
    if placed:
        return sum(placed) / len(placed)
    return float(order[node_id])
