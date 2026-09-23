"""Дерево команды ↔ граф нод: одна модель, два представления, без потери данных.

Нодовый вид и список-режим — это два взгляда на одну
:class:`~ayris.actions.macros.schema.CommandModel`. Здесь живёт чистая (без Qt)
конвертация между деревом блоков и графом нод и обратно, так что «список → граф →
список» можно проверить на равенство модели, а не на глаз. Любое расхождение
round-trip — это потеря пользовательских данных, поэтому конвертация держится за
несколько правил:

* **Нода на блок.** Каждый :class:`ActionBlock` — одна нода; её ``id`` — путь блока
  (``actions[1].then[0]``), уникальный в пределах снимка дерева.
* **Связи = смежность дерева.** В списке соседей идёт «главная» связь
  (:data:`MAIN_PORT`) от блока к следующему; у логического блока порт ветки
  (``then``/``else``/``body``/``catch``) ведёт к первому блоку ветки, дальше её блоки
  сцеплены главными связями.
* **Обратная сборка канонична.** :func:`command_from_graph` идёт по связям от корня,
  собирая вложенные списки заново; неполный граф (нода без входа) не теряется — её
  блок дописывается в конец корневого списка, а :meth:`NodeGraph.orphan_ids` называет
  такие ноды для предупреждения.

Задержка на проводе — это блок ``Пауза`` (``Sleep``) между двумя нодами: конвертация
держит его обычной нодой (round-trip не страдает), а рисующий слой волен показать его
компактным чипом на связи.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Final

from ayris.actions.macros.blocks.catalog import BlockCatalog, BlockCategory
from ayris.actions.macros.schema import LOGIC_BLOCKS, ActionBlock, CommandModel

if TYPE_CHECKING:
    from ayris.actions.base import ParamField

#: A chain builder: given a start node id, the list of blocks it heads (see below).
_ChainFn = Callable[[str | None], "list[ActionBlock]"]

__all__ = [
    "BRANCH_LABELS",
    "DELAY_BLOCK_TYPE",
    "MAIN_PORT",
    "DisplayEdge",
    "GraphEdge",
    "GraphNode",
    "NodeGraph",
    "NodeRole",
    "can_connect",
    "collapse_delays",
    "command_from_graph",
    "delay_ms_of",
    "format_delay",
    "graph_from_command",
    "is_delay_block",
    "layout_from_json",
    "layout_to_json",
    "make_delay_block",
    "role_of",
    "summary_of",
]

#: The port name of the sequential «next block» connection between two siblings.
MAIN_PORT: Final = "main"

#: The block type that carries a delay between two nodes — «Пауза». A linear one is drawn
#: collapsed onto the wire as a chip (see :func:`collapse_delays`), never as its own card,
#: but it stays an ordinary block in the model, the palette and the ``.ayris`` file.
DELAY_BLOCK_TYPE: Final = "Sleep"

#: Wire branch name → the schema field that holds it (``else`` is a keyword).
_BRANCH_FIELD: Final[dict[str, str]] = {
    "then": "then",
    "else": "else_",
    "body": "body",
    "catch": "catch",
}

#: Russian labels for the branch ports drawn on a logic node.
BRANCH_LABELS: Final[dict[str, str]] = {
    "then": "то",
    "else": "иначе",
    "body": "тело",
    "catch": "ошибка",
}

#: The TTS-facing audio blocks that read as an «ответ», not a «звук».
_RESPONSE_BLOCKS: Final[frozenset[str]] = frozenset({"Say", "SetTTSVoice"})

#: Blocks that read as the danger role even without a catalog flag.
_MAX_SUMMARY: Final = 64


class NodeRole(StrEnum):
    """The visual role of a node: colour, icon and label come from it.

    Triggers are not action blocks (they live on the command, not in ``actions``), so
    the action node editor only ever assigns the last four; ``TRIGGER`` is kept for the
    palette and for symmetry with the mockup.
    """

    TRIGGER = "trigger"
    RESPONSE = "response"
    ACTION = "action"
    CONDITION = "condition"
    SOUND = "sound"


#: Role → the :class:`~ayris.gui.theme.tokens.ColorTokens` field that colours it. The two
#: new tokens of task 53 (``role_action``/``role_sound``) keep «действие» and «звук» from
#: collapsing onto «триггер»/«ответ» by hue.
ROLE_TOKENS: Final[dict[NodeRole, str]] = {
    NodeRole.TRIGGER: "info",
    NodeRole.RESPONSE: "accent",
    NodeRole.ACTION: "role_action",
    NodeRole.CONDITION: "success",
    NodeRole.SOUND: "role_sound",
}

#: Role → its capitalised Russian label, drawn under the node title.
ROLE_LABELS: Final[dict[NodeRole, str]] = {
    NodeRole.TRIGGER: "триггер",
    NodeRole.RESPONSE: "ответ",
    NodeRole.ACTION: "действие",
    NodeRole.CONDITION: "условие",
    NodeRole.SOUND: "звук",
}

#: Role → the icon glyph name (lucide) drawn in the node header.
ROLE_ICONS: Final[dict[NodeRole, str]] = {
    NodeRole.TRIGGER: "mic",
    NodeRole.RESPONSE: "message-square",
    NodeRole.ACTION: "zap",
    NodeRole.CONDITION: "git-branch",
    NodeRole.SOUND: "volume-2",
}


def role_of(block_type: str, catalog: BlockCatalog) -> NodeRole:
    """The node role a block type reads as, from the task-33 catalog and logic table.

    A branching logic block (``If``/``Switch``/``While``/``For``/``Try`` and the
    ``Case``/``Default`` arms) is a condition; a TTS block is a response; the rest of the
    audio category is a sound; everything else is an action. An unknown type — a command
    from a newer build — falls back to action rather than raising.
    """
    spec = LOGIC_BLOCKS.get(block_type)
    if spec is not None and spec.branches:
        return NodeRole.CONDITION
    meta = catalog.try_get(block_type)
    if meta is not None and meta.category is BlockCategory.AUDIO:
        return NodeRole.RESPONSE if block_type in _RESPONSE_BLOCKS else NodeRole.SOUND
    return NodeRole.ACTION


def summary_of(block: ActionBlock, catalog: BlockCatalog) -> str:
    """A single-line summary of a block's key parameters, ellipsised.

    The node body shows only this; the full parameters live in the inspector. Built from
    the block's own filled-in parameters in catalog field order, so the most meaningful
    values lead. Empty for a block with nothing set yet.
    """
    meta = catalog.try_get(block.type)
    fields: tuple[ParamField, ...] = meta.fields if meta is not None else ()
    parts: list[str] = []
    for spec in fields:
        value = block.params.get(spec.name)
        if value is None or value == "" or value == [] or value == {}:
            continue
        parts.append(f"{spec.label_ru}: {value}")
        if len(parts) >= 2:
            break
    if not parts:
        # Fall back to any set parameter, then to the catalog description.
        for name, value in block.params.items():
            if value not in (None, "", [], {}):
                parts.append(f"{name}: {value}")
                break
    text = " · ".join(parts)
    if not text and meta is not None:
        text = meta.description_ru
    if len(text) > _MAX_SUMMARY:
        text = text[: _MAX_SUMMARY - 1].rstrip() + "…"
    return text


@dataclass(slots=True)
class GraphNode:
    """One block as a node: identity, its block, role, title and free position."""

    id: str
    block: ActionBlock
    role: NodeRole
    title: str = ""
    x: float = 0.0
    y: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.block.enabled

    @property
    def branch_ports(self) -> tuple[str, ...]:
        """The wire branch names this node exposes as output ports, schema order."""
        spec = LOGIC_BLOCKS.get(self.block.type)
        return spec.branches if spec is not None else ()


@dataclass(slots=True)
class GraphEdge:
    """A connection from a node's output port to another node's input.

    ``source_port`` is :data:`MAIN_PORT` for the sequential «next block» wire, or a branch
    name (``then``/``else``/``body``/``catch``) for a branch's first child.
    """

    source_id: str
    source_port: str
    target_id: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.source_port, self.target_id)


@dataclass(slots=True)
class NodeGraph:
    """Nodes and edges derived from a command, plus which node the flow starts at."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    root_id: str | None = None

    def node_by_id(self, node_id: str) -> GraphNode | None:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def out_edge(self, node_id: str, port: str) -> GraphEdge | None:
        """The single edge leaving ``node_id`` on ``port`` — a port holds at most one."""
        for edge in self.edges:
            if edge.source_id == node_id and edge.source_port == port:
                return edge
        return None

    def in_edge(self, node_id: str) -> GraphEdge | None:
        """The single edge entering ``node_id`` — an input holds at most one."""
        for edge in self.edges:
            if edge.target_id == node_id:
                return edge
        return None

    def orphan_ids(self) -> list[str]:
        """Nodes with no incoming edge that are not the root — the disconnected ones."""
        return [
            node.id
            for node in self.nodes
            if node.id != self.root_id and self.in_edge(node.id) is None
        ]

    def positions(self) -> dict[str, tuple[float, float]]:
        return {node.id: (node.x, node.y) for node in self.nodes}


def _path_text(path: tuple[str | int, ...]) -> str:
    """``actions[1].then[0]`` — the block path a person and the debugger both read."""
    parts: list[str] = []
    for step in path:
        if isinstance(step, int):
            parts.append(f"[{step}]")
        elif parts:
            parts.append(f".{step}")
        else:
            parts.append(str(step))
    return "".join(parts)


def graph_from_command(
    command: CommandModel,
    *,
    catalog: BlockCatalog,
    positions: Mapping[str, tuple[float, float]] | None = None,
) -> NodeGraph:
    """Build the node graph for a command's action tree, left-to-right by nature.

    Every block becomes a node keyed by its path; consecutive siblings are joined by a
    :data:`MAIN_PORT` edge and each branch's first child by a branch-port edge. Positions
    are taken from ``positions`` (a saved layout) when present, else left at the origin for
    :func:`~ayris.gui.widgets.node_editor.layout.auto_layout` to place.
    """
    graph = NodeGraph()
    positions = positions or {}
    _build_list(command.actions, ("actions",), graph, catalog, positions)
    # The flow starts at the first *wired* root block, not simply the first: a command
    # may open with free (detached) draft nodes sitting at the top, and the walk that
    # decides what runs must not treat one of them as the start.
    root_index = next(
        (index for index, block in enumerate(command.actions) if not block.detached), None
    )
    graph.root_id = _path_text(("actions", root_index)) if root_index is not None else None
    return graph


def _build_list(
    blocks: list[ActionBlock],
    container: tuple[str | int, ...],
    graph: NodeGraph,
    catalog: BlockCatalog,
    positions: Mapping[str, tuple[float, float]],
) -> None:
    previous_id: str | None = None
    for index, block in enumerate(blocks):
        node_id = _path_text((*container, index))
        meta = catalog.try_get(block.type)
        node = GraphNode(
            id=node_id,
            block=block,
            role=role_of(block.type, catalog),
            title=meta.title_ru if meta is not None else block.type,
        )
        if node_id in positions:
            node.x, node.y = positions[node_id]
        graph.nodes.append(node)
        # A detached block is a free node: no incoming main wire and never a predecessor,
        # so the connected chain skips straight over it and it reads as an orphan. Its
        # branch wires are still drawn below, so a detached logic block keeps its subtree
        # attached to it as one movable, re-wireable unit.
        if not block.detached:
            if previous_id is not None:
                graph.edges.append(GraphEdge(previous_id, MAIN_PORT, node_id))
            previous_id = node_id
        spec = LOGIC_BLOCKS.get(block.type)
        if spec is None:
            continue
        for wire in spec.branches:
            children: list[ActionBlock] = getattr(block, _BRANCH_FIELD[wire])
            if not children:
                continue
            child_container = (*container, index, wire)
            first_child_id = _path_text((*child_container, 0))
            graph.edges.append(GraphEdge(node_id, wire, first_child_id))
            _build_list(children, child_container, graph, catalog, positions)


def command_from_graph(
    graph: NodeGraph,
    base: CommandModel,
    *,
    id_remap: dict[str, str] | None = None,
) -> CommandModel:
    """Rebuild a command from a graph, keeping everything but the action tree from ``base``.

    Walks the flow from the *head* of :attr:`NodeGraph.root_id`'s chain along
    :data:`MAIN_PORT` edges, filling each logic node's branches from its branch-port edges. A
    chain the user builds purely from free nodes has no live root to start from, so any head (a
    node with no incoming edge) wired forward along the main port also seeds a live chain — this
    is what lets two free nodes, once connected, become a running flow whose wire survives the
    next rebuild. A head with no forward wire is a lone free node; it and any node no flow
    reaches are appended detached rather than lost — :meth:`NodeGraph.orphan_ids` names them.

    When ``id_remap`` is given it is filled with ``{old node id: new node id}``: the rebuild
    reorders ``actions``, which renumbers the positional paths node coordinates are keyed by,
    and the scene rides this map to keep every node's position across the edit.
    """
    fresh: dict[str, ActionBlock] = {}
    origin: dict[int, str] = {}
    for node in graph.nodes:
        block = _fresh_block(node.block)
        fresh[node.id] = block
        origin[id(block)] = node.id
    visited: set[str] = set()

    def chain(start_id: str | None) -> list[ActionBlock]:
        result: list[ActionBlock] = []
        current = start_id
        while current is not None and current in fresh and current not in visited:
            visited.add(current)
            block = fresh[current]
            _fill_branches(block, current, graph, chain)
            result.append(block)
            nxt = graph.out_edge(current, MAIN_PORT)
            current = nxt.target_id if nxt is not None else None
        return result

    # Reachability from a flow entry is the source of truth for «detached»: a block a flow
    # reaches is wired (and runs), a block no flow reaches is a free node. The primary flow
    # starts at the *head* of ``root_id``'s main chain: usually ``root_id`` itself, but if the
    # user wired a fresh node into the loaded root's input, that node is the new head — begin
    # there so a prepend keeps its place at the front instead of being stranded at the tail
    # with its wire dropped. A flow built purely from free nodes has no live root; its heads
    # seed live chains below. Marking stays here, so connecting clears the flag and
    # disconnecting sets it with no extra bookkeeping in the scene.
    connected = chain(_main_chain_head(graph, graph.root_id))
    _mark_detached(connected, detached=False)
    wired: list[ActionBlock] = []
    detached: list[ActionBlock] = []
    for node in graph.nodes:
        if node.id in visited:
            continue
        if graph.in_edge(node.id) is not None:
            continue  # reached mid-chain or as a branch child — placed by its head's walk
        members = chain(node.id)
        # A head the user has wired forward is a live chain even without a live root above it
        # (a flow built entirely from free nodes); a head with no forward wire is a lone free
        # node. Either way the whole subtree inherits the flag.
        live = graph.out_edge(node.id, MAIN_PORT) is not None
        _mark_detached(members, detached=not live)
        (wired if live else detached).extend(members)
    for node in graph.nodes:  # safety net: a node no head reached (only via a forbidden cycle)
        if node.id not in visited:
            tail = chain(node.id)
            _mark_detached(tail, detached=True)
            detached.extend(tail)
    new_actions = connected + wired + detached
    if id_remap is not None:
        _index_paths(new_actions, ("actions",), origin, id_remap)
    return base.model_copy(update={"actions": new_actions})


def _main_chain_head(graph: NodeGraph, node_id: str | None) -> str | None:
    """Walk back along incoming main edges to the head of ``node_id``'s chain.

    Follows only :data:`MAIN_PORT` edges — a branch edge means the node is a branch's child,
    not a chain head — and a ``seen`` guard keeps a stray cycle from looping forever. Returns
    ``node_id`` unchanged when nothing is wired into it, and ``None`` for ``None``.
    """
    if node_id is None:
        return None
    head = node_id
    seen: set[str] = set()
    while head not in seen:
        seen.add(head)
        edge = graph.in_edge(head)
        if edge is None or edge.source_port != MAIN_PORT:
            break
        head = edge.source_id
    return head


def _index_paths(
    blocks: list[ActionBlock],
    container: tuple[str | int, ...],
    origin: dict[int, str],
    remap: dict[str, str],
) -> None:
    """Record ``old node id → new path`` for each rebuilt block, mirroring :func:`_build_list`.

    The rebuild reorders ``actions``, so a block's positional path — the id its canvas
    coordinate is keyed by — changes. Walking the fresh tree the same way it was built maps
    each block (found by its identity in ``origin``) from its old id to its new one.
    """
    for index, block in enumerate(blocks):
        new_id = _path_text((*container, index))
        old_id = origin.get(id(block))
        if old_id is not None:
            remap[old_id] = new_id
        spec = LOGIC_BLOCKS.get(block.type)
        if spec is None:
            continue
        for wire in spec.branches:
            children: list[ActionBlock] = getattr(block, _BRANCH_FIELD[wire])
            _index_paths(children, (*container, index, wire), origin, remap)


def _mark_detached(blocks: list[ActionBlock], *, detached: bool) -> None:
    """Set ``detached`` on each block and its whole subtree, so a free node's branches
    carry the flag too and never run when their unwired parent does not."""
    for block in blocks:
        block.detached = detached
        for _wire, children in block.branches():
            _mark_detached(children, detached=detached)


def _fresh_block(block: ActionBlock) -> ActionBlock:
    """A copy of a block with empty branches — the branches come back from the edges."""
    return ActionBlock(
        type=block.type,
        params=dict(block.params),
        sound=block.sound.model_copy(deep=True) if block.sound is not None else None,
        enabled=block.enabled,
        comment=block.comment,
        on_error=block.on_error,
    )


def _fill_branches(block: ActionBlock, node_id: str, graph: NodeGraph, chain: _ChainFn) -> None:
    spec = LOGIC_BLOCKS.get(block.type)
    if spec is None:
        return
    for wire in spec.branches:
        edge = graph.out_edge(node_id, wire)
        if edge is None:
            continue
        children: list[ActionBlock] = chain(edge.target_id)
        setattr(block, _BRANCH_FIELD[wire], children)


# ----------------------------------------------------------------------
# Delay on the wire — a linear «Пауза» (Sleep) drawn collapsed as a chip
# ----------------------------------------------------------------------


def is_delay_block(block: ActionBlock) -> bool:
    """Whether a block is a delay — the :data:`DELAY_BLOCK_TYPE` («Пауза»)."""
    return block.type == DELAY_BLOCK_TYPE


def delay_ms_of(block: ActionBlock) -> int:
    """A delay block's duration in milliseconds, from ``ms`` or ``seconds`` (never < 0).

    ``Sleep`` accepts either spelling; the chip shows one number, so both are normalised
    to whole milliseconds here. A non-numeric or missing value reads as ``0``.
    """
    params = block.params
    ms = params.get("ms")
    if isinstance(ms, int | float) and not isinstance(ms, bool):
        return max(0, int(ms))
    seconds = params.get("seconds")
    if isinstance(seconds, int | float) and not isinstance(seconds, bool):
        return max(0, round(float(seconds) * 1000))
    return 0


def make_delay_block(ms: int) -> ActionBlock:
    """A fresh «Пауза» block of ``ms`` milliseconds, for inserting on a wire."""
    return ActionBlock(type=DELAY_BLOCK_TYPE, params={"ms": max(0, int(ms))})


def format_delay(ms: int) -> str:
    """The chip label for a delay: ``«0 мс»`` when none, ``«500 мс»``, ``«1,5 с»``.

    Under a second the value reads in milliseconds; from a second up it reads in seconds
    with a Russian decimal comma and no trailing zeros, so a round delay stays terse.
    """
    if ms <= 0:
        return "0 мс"
    if ms < 1000:
        return f"{ms} мс"
    seconds = ms / 1000
    text = f"{seconds:.2f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', ',')} с"


@dataclass(slots=True)
class DisplayEdge:
    """A wire as the canvas draws it, with the delay collapsed onto it as a chip.

    A run of linear :data:`DELAY_BLOCK_TYPE` nodes between two visible nodes is not drawn as
    its own cards; instead this one edge carries their summed :attr:`delay_ms`, and
    :attr:`sleep_ids` names the collapsed nodes so an edit can update or drop them in the
    model. An ordinary wire (no delay) is still a :class:`DisplayEdge` with ``delay_ms == 0``
    and an empty :attr:`sleep_ids`, so every wire shows a chip.
    """

    source_id: str
    source_port: str
    target_id: str
    delay_ms: int = 0
    sleep_ids: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.source_id, self.source_port, self.target_id)


def _is_collapsible_delay(graph: NodeGraph, node: GraphNode) -> bool:
    """Whether a node is a delay to fold onto its wire rather than draw as a card.

    A delay folds only when it sits *between* two nodes in the flow: it must be a delay
    block, be wired in (an incoming edge), lead somewhere (a main outgoing edge), not be the
    flow's root, and be enabled — a disabled or dangling delay stays a visible card so its
    off/end state is not silently hidden behind a chip.
    """
    if not is_delay_block(node.block):
        return False
    if node.id == graph.root_id:
        return False
    if not node.block.enabled:
        return False
    if graph.in_edge(node.id) is None:
        return False
    return graph.out_edge(node.id, MAIN_PORT) is not None


def collapse_delays(graph: NodeGraph) -> tuple[set[str], list[DisplayEdge]]:
    """Fold linear delay runs onto wires, returning the hidden node ids and display edges.

    Every wire leaving a *visible* node becomes one :class:`DisplayEdge`: the walk follows
    the wire through any collapsible delay nodes to the first visible node, summing their
    durations. Hidden ids are the delays folded away — the scene draws no card for them. A
    delay that is the root, disabled or has no successor is not collapsible and stays a
    visible node, so it never vanishes without a place to show it.
    """
    hidden = {node.id for node in graph.nodes if _is_collapsible_delay(graph, node)}
    display: list[DisplayEdge] = []
    for node in graph.nodes:
        if node.id in hidden:
            continue
        for port in (MAIN_PORT, *node.branch_ports):
            edge = graph.out_edge(node.id, port)
            if edge is None:
                continue
            delay = 0
            sleeps: list[str] = []
            cur: str | None = edge.target_id
            while cur is not None and cur in hidden:
                sleeps.append(cur)
                folded = graph.node_by_id(cur)
                if folded is not None:
                    delay += delay_ms_of(folded.block)
                nxt = graph.out_edge(cur, MAIN_PORT)
                cur = nxt.target_id if nxt is not None else None
            if cur is None:
                continue
            display.append(DisplayEdge(node.id, port, cur, delay, sleeps))
    return hidden, display


def can_connect(
    graph: NodeGraph, source_id: str, source_port: str, target_id: str
) -> tuple[bool, str]:
    """Whether an edge ``source[port] → target`` is allowed, and why not otherwise.

    Refuses a self-loop, a second wire into an occupied input, a second wire out of an
    occupied port, and any connection that would close a cycle in what must stay a tree.
    The message is Russian and shown next to the live wire when a drop is rejected.
    """
    if source_id == target_id:
        return False, "Нельзя соединить ноду с собой."
    if graph.node_by_id(source_id) is None or graph.node_by_id(target_id) is None:
        return False, "Одна из нод не найдена."
    if graph.in_edge(target_id) is not None:
        return False, "У блока уже есть входящая связь."
    if graph.out_edge(source_id, source_port) is not None:
        return False, "У этого выхода уже есть связь."
    if _reaches(graph, target_id, source_id):
        return False, "Связь образует цикл."
    return True, ""


def _reaches(graph: NodeGraph, start_id: str, goal_id: str) -> bool:
    """Whether ``goal_id`` is reachable from ``start_id`` following outgoing edges."""
    stack = [start_id]
    seen: set[str] = set()
    while stack:
        current = stack.pop()
        if current == goal_id:
            return True
        if current in seen:
            continue
        seen.add(current)
        stack.extend(edge.target_id for edge in graph.edges if edge.source_id == current)
    return False


def layout_to_json(graph: NodeGraph) -> dict[str, list[float]]:
    """The node positions keyed by block path — the UI metadata saved with a command.

    A plain ``{path: [x, y]}`` mapping, kept out of the ``.ayris`` schema on purpose: its
    absence or corruption must only cost an auto-layout, never the ability to open a
    command. Returned as data (JSON-ready), serialised by the caller.
    """
    return {node.id: [round(node.x, 1), round(node.y, 1)] for node in graph.nodes}


def layout_from_json(data: object) -> dict[str, tuple[float, float]]:
    """Read a saved layout back into ``{path: (x, y)}``, tolerating a corrupt blob.

    Anything that is not a well-formed pair of numbers is skipped rather than raised on:
    stale or hand-edited UI metadata should degrade to a partial layout, not a crash.
    """
    result: dict[str, tuple[float, float]] = {}
    if not isinstance(data, dict):
        return result
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, list | tuple) and len(value) == 2:
            x, y = value
            if isinstance(x, int | float) and isinstance(y, int | float):
                result[key] = (float(x), float(y))
    return result
