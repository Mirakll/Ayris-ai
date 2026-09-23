"""Мост «дерево команды ↔ граф нод» (задача 53), без Qt.

Ни одного виджета: всё проверяется на чистой конвертации из
:mod:`ayris.gui.widgets.node_editor.bridge`. Главное утверждение — round-trip
«список → граф → список» не теряет ничего: ``model_dump`` исходной команды равен
``model_dump`` пересобранной, включая вложенные ``If``/``While``/``Try``/``Switch``.
Отдельно проверяются запреты связей (:func:`can_connect`), сохранение координат
(:func:`layout_to_json`/:func:`layout_from_json`) и роли нод (:func:`role_of`).
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
    can_connect,
    command_from_graph,
    graph_from_command,
    layout_from_json,
    layout_to_json,
    role_of,
    summary_of,
)

pytestmark = pytest.mark.unit


# Реестр действий поднимается один раз на модуль: BlockCatalog() без реестра
# зовёт discover(), а это дорого повторять.
@pytest.fixture(scope="module")
def catalog() -> BlockCatalog:
    return BlockCatalog()


def _nested_command() -> CommandModel:
    """Команда со всеми видами ветвления: If (то/иначе), While, Try, Switch."""
    return CommandModel(
        name="Сложная",
        description="round-trip",
        actions=[
            ActionBlock(type="Say", params={"text": "старт"}),
            ActionBlock(
                type="If",
                params={"condition": "{x}"},
                then=[
                    ActionBlock(type="Say", params={"text": "да"}),
                    ActionBlock(type="Sleep", params={"ms": "100"}),
                ],
                else_=[ActionBlock(type="Say", params={"text": "нет"})],
            ),
            ActionBlock(
                type="While",
                params={"condition": "{y}"},
                body=[
                    ActionBlock(
                        type="Try",
                        body=[ActionBlock(type="Say", params={"text": "в цикле"})],
                        catch=[ActionBlock(type="Say", params={"text": "поймал"})],
                    ),
                ],
            ),
            ActionBlock(
                type="Switch",
                params={"value": "{z}"},
                body=[
                    ActionBlock(
                        type="Case",
                        params={"value": "1"},
                        body=[ActionBlock(type="Say", params={"text": "один"})],
                    ),
                    ActionBlock(
                        type="Default",
                        body=[ActionBlock(type="Say", params={"text": "иначе"})],
                    ),
                ],
            ),
        ],
    )


def _manual_graph() -> NodeGraph:
    """Три ноды-заглушки a→b→c, чтобы проверять связи без каталога."""
    nodes = [
        GraphNode(id="a", block=ActionBlock(type="Say"), role=NodeRole.RESPONSE),
        GraphNode(id="b", block=ActionBlock(type="Say"), role=NodeRole.RESPONSE),
        GraphNode(id="c", block=ActionBlock(type="Say"), role=NodeRole.RESPONSE),
    ]
    return NodeGraph(nodes=nodes, edges=[], root_id="a")


# ----------------------------------------------------------------------
# round-trip: список → граф → список без потери данных
# ----------------------------------------------------------------------


def test_round_trip_preserves_nested_command(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    rebuilt = command_from_graph(graph, command)
    # Побайтовое равенство модели — любое расхождение это потеря данных пользователя.
    assert rebuilt.model_dump() == command.model_dump()


def test_round_trip_keeps_block_flags(catalog: BlockCatalog) -> None:
    command = CommandModel(
        name="Флаги",
        actions=[
            ActionBlock(
                type="RunShell",
                params={"cmd": "echo"},
                enabled=False,
                comment="осторожно",
            ),
            ActionBlock(type="Say", params={"text": "готово"}),
        ],
    )
    rebuilt = command_from_graph(graph_from_command(command, catalog=catalog), command)
    first = rebuilt.actions[0]
    assert first.enabled is False
    assert first.comment == "осторожно"
    assert rebuilt.model_dump() == command.model_dump()


def test_round_trip_empty_command(catalog: BlockCatalog) -> None:
    command = CommandModel(name="Пусто")
    graph = graph_from_command(command, catalog=catalog)
    assert graph.nodes == []
    assert graph.root_id is None
    rebuilt = command_from_graph(graph, command)
    assert rebuilt.model_dump() == command.model_dump()


def test_round_trip_stable_on_second_pass(catalog: BlockCatalog) -> None:
    command = _nested_command()
    once = command_from_graph(graph_from_command(command, catalog=catalog), command)
    twice = command_from_graph(graph_from_command(once, catalog=catalog), once)
    assert twice.model_dump() == command.model_dump()


# ----------------------------------------------------------------------
# структура графа: нода на блок, связи = смежность дерева
# ----------------------------------------------------------------------


def test_graph_node_ids_are_block_paths(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    ids = {node.id for node in graph.nodes}
    # Пути читаются как у отладчика: actions[i], ветки — actions[i].then[j] и т.п.
    assert "actions[0]" in ids
    assert "actions[1].then[0]" in ids
    assert "actions[1].else[0]" in ids
    assert "actions[2].body[0].catch[0]" in ids
    assert "actions[3].body[1].body[0]" in ids


def test_main_edges_chain_siblings(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    # Корневые соседи сцеплены главной связью по порядку.
    assert graph.out_edge("actions[0]", MAIN_PORT) is not None
    assert graph.out_edge("actions[0]", MAIN_PORT).target_id == "actions[1]"
    # Первый блок ветки то входит по branch-порту then, а не по main.
    then_edge = graph.out_edge("actions[1]", "then")
    assert then_edge is not None and then_edge.target_id == "actions[1].then[0]"
    # Внутри ветки то два блока — они сцеплены главной связью.
    inner = graph.out_edge("actions[1].then[0]", MAIN_PORT)
    assert inner is not None and inner.target_id == "actions[1].then[1]"


def test_root_is_first_action(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    assert graph.root_id == "actions[0]"


def test_branch_ports_follow_schema(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    if_node = graph.node_by_id("actions[1]")
    assert if_node is not None
    assert if_node.branch_ports == ("then", "else")
    try_node = graph.node_by_id("actions[2].body[0]")
    assert try_node is not None
    assert try_node.branch_ports == ("body", "catch")


# ----------------------------------------------------------------------
# орфаны: несвязанная нода не теряется, а дописывается в конец
# ----------------------------------------------------------------------


def test_orphan_node_is_appended_not_lost(catalog: BlockCatalog) -> None:
    command = CommandModel(
        name="Разрыв",
        actions=[
            ActionBlock(type="Say", params={"text": "раз"}),
            ActionBlock(type="Say", params={"text": "два"}),
        ],
    )
    graph = graph_from_command(command, catalog=catalog)
    # Убираем главную связь — второй блок остаётся без входа (орфан).
    graph.edges = [edge for edge in graph.edges if edge.source_id != "actions[0]"]
    assert graph.orphan_ids() == ["actions[1]"]
    rebuilt = command_from_graph(graph, command)
    # Оба блока целы: разрыв не удаляет блок, он просто дописан в конец.
    assert [b.params.get("text") for b in rebuilt.actions] == ["раз", "два"]


def test_wiring_two_free_nodes_makes_them_a_live_chain(catalog: BlockCatalog) -> None:
    # Регресс BUG 1: две свободные (detached) ноды соединяли проводом, но связь пропадала
    # на первом же rebuild — обе оставались detached, а detached-блоки не получают главной
    # связи в graph_from_command. Провод между свободными нодами обязан оживить их цепочку и
    # пережить повторный разбор.
    command = CommandModel(
        name="Свободные",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}, detached=True),
            ActionBlock(type="Say", params={"text": "B"}, detached=True),
        ],
    )
    graph = graph_from_command(command, catalog=catalog)
    assert graph.edges == []  # свободные ноды не соединяются автоматически
    assert graph.root_id is None
    graph.edges.append(GraphEdge("actions[0]", MAIN_PORT, "actions[1]"))
    rebuilt = command_from_graph(graph, command)
    # Соединение оживило обе ноды — теперь это поток, а не два черновика.
    assert [b.detached for b in rebuilt.actions] == [False, False]
    # И главное: провод переживает повторный разбор (раньше здесь он исчезал).
    regraph = graph_from_command(rebuilt, catalog=catalog)
    edge = regraph.out_edge("actions[0]", MAIN_PORT)
    assert edge is not None and edge.target_id == "actions[1]"


def test_prepend_free_node_to_live_chain_keeps_topology(catalog: BlockCatalog) -> None:
    # Регресс BUG 2: подключаем свободную X в ГОЛОВУ живой цепочки A→B→C. Раньше обход стартовал
    # от root_id=A, съедал A,B,C, а X с потерянным ребром уезжал в хвост (получалось A→B→C→X).
    # Обратный проход к истинной голове по входящим main-рёбрам обязан дать поток X→A→B→C.
    command = CommandModel(
        name="Голова",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
            ActionBlock(type="Say", params={"text": "X"}, detached=True),
        ],
    )
    graph = graph_from_command(command, catalog=catalog)
    assert graph.root_id == "actions[0]"
    graph.edges.append(GraphEdge("actions[3]", MAIN_PORT, "actions[0]"))  # X → A
    rebuilt = command_from_graph(graph, command)
    assert [b.params["text"] for b in rebuilt.actions] == ["X", "A", "B", "C"]
    assert all(b.detached is False for b in rebuilt.actions)
    # Поток переживает повторный разбор именно как X→A→B→C, а не в обратную сторону.
    regraph = graph_from_command(rebuilt, catalog=catalog)
    order = ["X", "A", "B", "C"]
    for index, _text in enumerate(order[:-1]):
        edge = regraph.out_edge(f"actions[{index}]", MAIN_PORT)
        assert edge is not None
        nxt = regraph.node_by_id(edge.target_id)
        assert nxt is not None and nxt.block.params["text"] == order[index + 1]


def test_command_from_graph_reports_id_remap_after_reorder(catalog: BlockCatalog) -> None:
    # Регресс BUG 3 (механизм): связывание переставляет actions, а координаты нод ключуются
    # путём. command_from_graph обязан сообщить старый→новый id, чтобы сцена перенесла координаты
    # и карточки не поменялись местами.
    command = CommandModel(
        name="Перестановка",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}, detached=True),
            ActionBlock(type="Say", params={"text": "B"}, detached=True),
            ActionBlock(type="Say", params={"text": "C"}, detached=True),
        ],
    )
    graph = graph_from_command(command, catalog=catalog)
    graph.edges.append(GraphEdge("actions[1]", MAIN_PORT, "actions[0]"))  # B → A
    remap: dict[str, str] = {}
    rebuilt = command_from_graph(graph, command, id_remap=remap)
    # B встал в голову: список стал [B, A, C].
    assert [b.params["text"] for b in rebuilt.actions] == ["B", "A", "C"]
    # Старый actions[1] (B) → новый actions[0]; старый actions[0] (A) → actions[1]; C на месте.
    assert remap["actions[1]"] == "actions[0]"
    assert remap["actions[0]"] == "actions[1]"
    assert remap["actions[2]"] == "actions[2]"


# ----------------------------------------------------------------------
# can_connect: запреты — сам-с-собой, занятый вход/выход, цикл
# ----------------------------------------------------------------------


def test_can_connect_allows_free_edge() -> None:
    graph = _manual_graph()
    ok, reason = can_connect(graph, "a", MAIN_PORT, "b")
    assert ok is True
    assert reason == ""


def test_can_connect_rejects_self_loop() -> None:
    graph = _manual_graph()
    ok, reason = can_connect(graph, "a", MAIN_PORT, "a")
    assert ok is False
    assert "с собой" in reason


def test_can_connect_rejects_unknown_node() -> None:
    graph = _manual_graph()
    ok, reason = can_connect(graph, "a", MAIN_PORT, "нет")
    assert ok is False
    assert "не найдена" in reason


def test_can_connect_rejects_occupied_input() -> None:
    graph = _manual_graph()
    graph.edges.append(GraphEdge("a", MAIN_PORT, "b"))
    # У b уже есть входящая связь от a; c войти не может.
    ok, reason = can_connect(graph, "c", MAIN_PORT, "b")
    assert ok is False
    assert "входящая связь" in reason


def test_can_connect_rejects_occupied_output_port() -> None:
    graph = _manual_graph()
    graph.edges.append(GraphEdge("a", MAIN_PORT, "b"))
    # main-выход a уже занят; второй провод с того же порта нельзя.
    ok, reason = can_connect(graph, "a", MAIN_PORT, "c")
    assert ok is False
    assert "этого выхода" in reason


def test_can_connect_rejects_cycle() -> None:
    graph = _manual_graph()
    graph.edges.append(GraphEdge("a", MAIN_PORT, "b"))
    graph.edges.append(GraphEdge("b", MAIN_PORT, "c"))
    # c→a замкнуло бы дерево в цикл (a достижима из c через… из a достижима c).
    ok, reason = can_connect(graph, "c", MAIN_PORT, "a")
    assert ok is False
    assert "цикл" in reason


# ----------------------------------------------------------------------
# сохранение координат: layout_to_json ↔ layout_from_json и восстановление
# ----------------------------------------------------------------------


def test_layout_round_trips_through_json(catalog: BlockCatalog) -> None:
    command = _nested_command()
    graph = graph_from_command(command, catalog=catalog)
    for index, node in enumerate(graph.nodes):
        node.x = float(index * 100)
        node.y = float(index * 50 + 7)
    data = layout_to_json(graph)
    restored = layout_from_json(data)
    assert restored == {node.id: (node.x, node.y) for node in graph.nodes}


def test_saved_positions_applied_on_build(catalog: BlockCatalog) -> None:
    command = _nested_command()
    positions = {"actions[0]": (321.0, 123.0), "actions[1]": (500.0, 400.0)}
    graph = graph_from_command(command, catalog=catalog, positions=positions)
    assert (graph.node_by_id("actions[0]").x, graph.node_by_id("actions[0]").y) == (321.0, 123.0)
    assert (graph.node_by_id("actions[1]").x, graph.node_by_id("actions[1]").y) == (500.0, 400.0)


def test_layout_from_json_tolerates_garbage() -> None:
    # Битые метаданные UI не должны ронять открытие команды: мусор просто пропускается.
    assert layout_from_json("не словарь") == {}
    assert layout_from_json(None) == {}
    assert layout_from_json({"ok": [1, 2], "bad_len": [1], "bad_type": ["x", 2], 7: [1, 2]}) == {
        "ok": (1.0, 2.0)
    }


def test_layout_to_json_rounds_coordinates() -> None:
    node = GraphNode(id="a", block=ActionBlock(type="Say"), role=NodeRole.ACTION, x=1.239, y=2.0)
    graph = NodeGraph(nodes=[node])
    assert layout_to_json(graph) == {"a": [1.2, 2.0]}


# ----------------------------------------------------------------------
# роли и сводка блока
# ----------------------------------------------------------------------


def test_role_of_logic_block_is_condition(catalog: BlockCatalog) -> None:
    assert role_of("If", catalog) is NodeRole.CONDITION
    assert role_of("While", catalog) is NodeRole.CONDITION
    assert role_of("Try", catalog) is NodeRole.CONDITION


def test_role_of_tts_is_response_audio_is_sound(catalog: BlockCatalog) -> None:
    assert role_of("Say", catalog) is NodeRole.RESPONSE
    assert role_of("PlaySound", catalog) is NodeRole.SOUND


def test_role_of_unknown_falls_back_to_action(catalog: BlockCatalog) -> None:
    assert role_of("НетТакогоБлока", catalog) is NodeRole.ACTION


def test_summary_of_lists_filled_params(catalog: BlockCatalog) -> None:
    block = ActionBlock(type="Say", params={"text": "привет мир"})
    summary = summary_of(block, catalog)
    assert "привет мир" in summary


def test_summary_of_empty_block_is_short(catalog: BlockCatalog) -> None:
    # Ничего не настроено: сводка либо пустая, либо короткое описание из каталога.
    summary = summary_of(ActionBlock(type="Say"), catalog)
    assert len(summary) <= 64
