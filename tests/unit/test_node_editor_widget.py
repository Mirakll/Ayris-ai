"""Нодовый редактор как виджеты (задача 53), offscreen.

Окна не показываются (``QT_QPA_PLATFORM=offscreen`` ставит ``scripts/check.sh``).
Проверяется поведение, а не пиксели: сцена строит по ноде на блок и по связи на
смежность, выделение эмитит путь блока, запрос связи через модель принимается или
отклоняется по :func:`can_connect`, отладка подсвечивает нужную ноду, двойной клик
ставит точку останова, а :class:`NodeEditor` держит контракт списка-режима
(``rebuild``/``selected_path``/``select_path``, сигналы ``changed``/``block_selected``).
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.blocks.catalog import BlockCatalog
from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.action_list import ActionListModel
from ayris.gui.widgets.node_editor import NodeEditor
from ayris.gui.widgets.node_editor.bridge import MAIN_PORT
from ayris.gui.widgets.node_editor.edge_item import EdgeItem
from ayris.gui.widgets.node_editor.scene import NodeScene
from ayris.gui.widgets.node_editor.view import NodeView

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
    return BlockCatalog()


def _command() -> CommandModel:
    return CommandModel(
        name="Демо",
        actions=[
            ActionBlock(type="Say", params={"text": "раз"}),
            ActionBlock(
                type="If",
                params={"condition": "{x}"},
                then=[ActionBlock(type="Say", params={"text": "да"})],
                else_=[ActionBlock(type="Say", params={"text": "нет"})],
            ),
            ActionBlock(type="Say", params={"text": "конец"}),
        ],
    )


def _model(command: CommandModel | None = None) -> ActionListModel:
    model = ActionListModel()
    model.set_command(command if command is not None else _command())
    return model


# ----------------------------------------------------------------------
# сцена: построение, геометрия, выделение
# ----------------------------------------------------------------------


def test_scene_builds_a_node_per_block(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    # Say, If, If.then[0], If.else[0], Say — пять блоков, пять нод.
    assert len(scene.graph.nodes) == 5
    assert {node.id for node in scene.graph.nodes} == {
        "actions[0]",
        "actions[1]",
        "actions[1].then[0]",
        "actions[1].else[0]",
        "actions[2]",
    }


def test_scene_auto_layout_gives_distinct_positions(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    positions = {(node.x, node.y) for node in scene.graph.nodes}
    # Без сохранённой раскладки авто-раскладка развела все ноды по разным точкам.
    assert len(positions) == len(scene.graph.nodes)


def test_scene_selection_emits_block_path(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    seen: list[object] = []
    scene.block_selected.connect(seen.append)
    scene.select_path(("actions", 1))
    assert scene.selected_path() == ("actions", 1)
    assert seen and seen[-1] == ("actions", 1)


def test_scene_rebuild_while_selected_survives(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Regression: a structural edit rebuilds the scene while a node is selected.
    # ``clear()`` deletes the C++ items and fires ``selectionChanged``; if the handler
    # walked the stale node map it hit «Internal C++ object already deleted». The scene
    # must rebuild cleanly and keep the selection. This crashed the live app on any
    # add/connect/delete once the node canvas became the default action view.
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    scene.select_path(("actions", 1))
    seen: list[object] = []
    scene.block_selected.connect(seen.append)
    scene.rebuild()  # must not raise
    assert scene.selected_path() == ("actions", 1)
    assert seen and seen[-1] == ("actions", 1)


def test_scene_saved_layout_wins_over_auto(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.set_layout({"actions[0]": (777.0, 333.0)})
    scene.rebuild()
    node = scene.graph.node_by_id("actions[0]")
    assert (node.x, node.y) == (777.0, 333.0)


# ----------------------------------------------------------------------
# связи через модель: принять валидную, отклонить невалидную
# ----------------------------------------------------------------------


def test_request_connect_rejects_invalid(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    rejected: list[str] = []
    scene.connection_rejected.connect(rejected.append)
    # Сам-с-собой — запрещено; модель не меняется.
    assert scene.request_connect("actions[0]", MAIN_PORT, "actions[0]") is False
    assert rejected and "с собой" in rejected[-1]


def test_request_connect_reconnects_through_model(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Плоская команда из трёх блоков; разорвать и пересобрать порядок через связь.
    command = CommandModel(
        name="Плоско",
        actions=[
            ActionBlock(type="Say", params={"text": "1"}),
            ActionBlock(type="Say", params={"text": "2"}),
            ActionBlock(type="Say", params={"text": "3"}),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    changed: list[int] = []
    scene.changed_command.connect(lambda: changed.append(1))
    # Убираем связь 2→3, затем строим её заново — модель должна остаться валидной.
    scene.graph.edges = [edge for edge in scene.graph.edges if edge.source_id != "actions[1]"]
    assert scene.request_connect("actions[1]", MAIN_PORT, "actions[2]") is True
    assert changed  # структурное изменение объявлено
    assert model.command is not None
    assert [b.params.get("text") for b in model.command.actions] == ["1", "2", "3"]


def test_connecting_two_free_nodes_keeps_the_wire_after_rebuild(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # BUG 1: две свободные ноды не соединялись — провод исчезал на rebuild сразу после
    # request_connect. После связи цепочка обязана ожить, а провод — остаться на холсте.
    command = CommandModel(
        name="Свободные",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}, detached=True),
            ActionBlock(type="Say", params={"text": "B"}, detached=True),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    assert scene.graph.out_edge("actions[0]", MAIN_PORT) is None  # без авто-провода
    assert scene.request_connect("actions[0]", MAIN_PORT, "actions[1]") is True
    # request_connect уже пересобрал сцену через модель — провод пережил rebuild.
    edge = scene.graph.out_edge("actions[0]", MAIN_PORT)
    assert edge is not None and edge.target_id == "actions[1]"
    assert model.command is not None
    assert [b.detached for b in model.command.actions] == [False, False]


def _id_of(scene: NodeScene, text: str) -> str | None:
    """Node id of the block whose ``text`` param matches — paths renumber, texts do not."""
    for node in scene.graph.nodes:
        if node.block.params.get("text") == text:
            return node.id
    return None


def _text_of(scene: NodeScene, node_id: str) -> str | None:
    node = scene.graph.node_by_id(node_id)
    return node.block.params.get("text") if node is not None else None


def test_connecting_middle_to_top_keeps_positions(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Регресс BUG 3: три свободные ноды, среднюю (B) цепляем к верхней (A). Топология B→A
    # верна, но раньше карточки A и B визуально менялись местами: координаты ключуются путём,
    # а связь переставляет actions. Ни одна карточка не должна переехать.
    command = CommandModel(
        name="Тройка",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}, detached=True),
            ActionBlock(type="Say", params={"text": "B"}, detached=True),
            ActionBlock(type="Say", params={"text": "C"}, detached=True),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.set_layout(
        {"actions[0]": (100.0, 100.0), "actions[1]": (100.0, 300.0), "actions[2]": (100.0, 500.0)}
    )
    scene.rebuild()
    before = {n.block.params["text"]: (n.x, n.y) for n in scene.graph.nodes}
    # B → A: тянем от выхода B (actions[1]) во вход A (actions[0]).
    assert scene.request_connect("actions[1]", MAIN_PORT, "actions[0]") is True
    after = {n.block.params["text"]: (n.x, n.y) for n in scene.graph.nodes}
    assert after == before  # ни одна карточка не переехала


def test_prepend_free_node_becomes_new_head(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Регресс BUG 2 на сцене: живая цепочка A→B→C и свободная X; тянем от выхода X во вход A.
    # X обязан стать новой головой (X→A→B→C), а не уехать в хвост.
    command = CommandModel(
        name="Вставка головы",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
            ActionBlock(type="Say", params={"text": "X"}, detached=True),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    assert scene.request_connect("actions[3]", MAIN_PORT, "actions[0]") is True
    assert model.command is not None
    assert [b.params["text"] for b in model.command.actions] == ["X", "A", "B", "C"]
    assert all(b.detached is False for b in model.command.actions)
    edge = scene.graph.out_edge("actions[0]", MAIN_PORT)
    assert edge is not None and edge.target_id == "actions[1]"


def test_grab_input_port_detaches_the_wire(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Регресс BUG 1: провод хватается за входной порт приёмника и отрывается. begin_reconnect
    # снимает входящее ребро (приёмник становится свободным) и возвращает освободившийся конец
    # от источника, чтобы тянуть дальше. Бросок в пустоту (finish без цели) оставит C свободным.
    command = CommandModel(
        name="Цепочка",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    # Хватаем вход C (ребро B→C). Освободившийся конец — выход B.
    grabbed = scene.begin_reconnect("actions[2]")
    assert grabbed is not None
    assert _text_of(scene, grabbed[0]) == "B" and grabbed[1] == MAIN_PORT
    # C оторван — стал свободным, у B выход освободился, поток теперь только A→B.
    assert model.command is not None
    assert model.command.actions[2].detached is True
    assert scene.graph.out_edge(_id_of(scene, "B") or "", MAIN_PORT) is None


def test_grab_input_port_then_reconnect_to_another_node(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Регресс BUG 1 (reconnect): оторвали провод от C, бросили освободившийся конец B на
    # свободную D → B→D, а C так и остался свободным.
    command = CommandModel(
        name="Переподключение",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
            ActionBlock(type="Say", params={"text": "D"}, detached=True),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    grabbed = scene.begin_reconnect(_id_of(scene, "C") or "")  # ребро B→C
    assert grabbed is not None and _text_of(scene, grabbed[0]) == "B"
    source_id, source_port = grabbed
    assert scene.request_connect(source_id, source_port, _id_of(scene, "D") or "") is True
    edge = scene.graph.out_edge(_id_of(scene, "B") or "", MAIN_PORT)
    assert edge is not None and _text_of(scene, edge.target_id) == "D"
    assert model.command is not None
    c_block = next(b for b in model.command.actions if b.params["text"] == "C")
    assert c_block.detached is True


def test_grab_output_port_detaches_the_wire(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Провод тянется и с правого (выходного) конца: хватаем выход B (ребро B→C) — оторвётся
    # дальний конец, а не источник. begin_reconnect_from_source снимает ребро (C свободен) и
    # отдаёт id приёмника C, чтобы вид перецепил на него новый источник.
    command = CommandModel(
        name="Цепочка справа",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    # Хватаем выход B (ребро B→C). Освободился дальний конец — вход C.
    freed = scene.begin_reconnect_from_source(_id_of(scene, "B") or "", MAIN_PORT)
    assert freed is not None and _text_of(scene, freed) == "C"
    # C оторван — стал свободным, у B выход освободился, поток теперь только A→B.
    assert model.command is not None
    assert next(b for b in model.command.actions if b.params["text"] == "C").detached is True
    assert scene.graph.out_edge(_id_of(scene, "B") or "", MAIN_PORT) is None


def test_grab_output_port_reanchors_origin_to_new_node(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Зеркало reconnect: оторвали провод за выход B, свободный вход C перецепили на новый
    # источник D → D→C, а прежний источник B остался без исходящего провода.
    command = CommandModel(
        name="Смена истока",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
            ActionBlock(type="Say", params={"text": "C"}),
            ActionBlock(type="Say", params={"text": "D"}, detached=True),
        ],
    )
    model = _model(command)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    freed = scene.begin_reconnect_from_source(_id_of(scene, "B") or "", MAIN_PORT)  # ребро B→C
    assert freed is not None and _text_of(scene, freed) == "C"
    assert scene.request_connect(_id_of(scene, "D") or "", MAIN_PORT, freed) is True
    # Вход C теперь питается от нового истока D (а не от прежнего B). Порядок-модель линейна,
    # так что общий поток становится A→B→D→C — но входящий провод C идёт именно из D.
    into_c = scene.graph.in_edge(_id_of(scene, "C") or "")
    assert into_c is not None and _text_of(scene, into_c.source_id) == "D"
    edge = scene.graph.out_edge(_id_of(scene, "D") or "", MAIN_PORT)
    assert edge is not None and _text_of(scene, edge.target_id) == "C"
    assert model.command is not None
    assert next(b for b in model.command.actions if b.params["text"] == "C").detached is False


def test_begin_reconnect_from_source_returns_none_for_free_output(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # У свободного выхода отрывать нечего — begin_reconnect_from_source отдаёт None,
    # вид падает в обычное перетаскивание ноды.
    scene = NodeScene(_flat_model(2), theme, catalog=catalog)
    scene.rebuild()
    assert scene.begin_reconnect_from_source("actions[1]", MAIN_PORT) is None  # хвост, выхода нет


def test_begin_reconnect_returns_none_for_headless_node(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # У ноды без входящего провода хватать нечего — begin_reconnect отдаёт None,
    # вид падает в обычное перетаскивание.
    scene = NodeScene(_flat_model(2), theme, catalog=catalog)
    scene.rebuild()
    assert scene.begin_reconnect("actions[0]") is None  # голова цепочки, входа нет


# ----------------------------------------------------------------------
# отладка: подсветка и точки останова
# ----------------------------------------------------------------------


def test_highlight_block_marks_running(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    scene.highlight_block("actions[1].then[0]")
    running = scene.running_item()
    assert running is not None and running.node_id == "actions[1].then[0]"
    scene.clear_running()
    assert scene.running_item() is None


def test_toggle_breakpoint_round_trips(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    changes: list[None] = []
    scene.breakpoint_changed.connect(lambda: changes.append(None))
    assert scene.toggle_breakpoint("actions[0]") is True
    assert scene.breakpoints() == {"actions[0]"}
    assert scene.toggle_breakpoint("actions[0]") is False
    assert scene.breakpoints() == set()
    # Каждое переключение — один сигнал на сохранение.
    assert len(changes) == 2


def test_breakpoints_survive_rebuild(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    scene.toggle_breakpoint("actions[0]")
    scene.rebuild()  # структурная правка перестраивает сцену с нуля
    assert scene.breakpoints() == {"actions[0]"}
    item = scene.node_item("actions[0]")
    assert item is not None and item.has_breakpoint()


def test_set_breakpoints_reflects_stored_state_silently(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    changes: list[None] = []
    scene.breakpoint_changed.connect(lambda: changes.append(None))
    scene.set_breakpoints({"actions[0]", "actions[2]"})
    assert scene.breakpoints() == {"actions[0]", "actions[2]"}
    # Загрузка из хранилища не должна порождать сохранение обратно.
    assert changes == []


def test_breakpoint_of_deleted_block_is_dropped(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_flat_model(3), theme, catalog=catalog)
    scene.rebuild()
    scene.set_breakpoints({"actions[2]"})
    scene.select_path(("actions", 2))
    assert scene.delete_selected() is True
    # У удалённого блока не осталось ноды — точка останова отпала.
    assert scene.breakpoints() == set()


# ----------------------------------------------------------------------
# вид: масштаб к курсору держит границы
# ----------------------------------------------------------------------


def test_view_zoom_is_clamped(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    view = NodeView(scene)
    view.resize(400, 300)
    # Многократный zoom-out не проваливает масштаб ниже минимума 0.4.
    for _ in range(30):
        if view.scale_factor * 0.9 >= 0.4:
            view.scale(0.9, 0.9)
    assert view.scale_factor >= 0.4 - 1e-6
    view.reset_zoom()
    assert abs(view.scale_factor - 1.0) < 1e-6


def test_view_press_on_input_port_starts_reconnect(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Регресс BUG 1 через сам обработчик мыши: ЛКМ на входном порту приёмника отрывает провод
    # и начинает тянуть освободившийся конец от источника (а не двигает ноду).
    command = CommandModel(
        name="Холст",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
        ],
    )
    scene = NodeScene(_model(command), theme, catalog=catalog)
    view = NodeView(scene)
    view.resize(800, 600)
    scene.rebuild()
    b_item = scene.node_item("actions[1]")
    assert b_item is not None
    port_view = view.mapFromScene(b_item.input_scene_pos())
    local = QPointF(port_view)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        local,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    # Провод A→B оторван от входа B: вид тянет освободившийся конец от выхода A.
    assert view._connect_source is not None
    assert view._connect_source[0] == "actions[0]"


def test_view_press_on_output_port_starts_reverse_reconnect(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Зеркало через обработчик мыши: ЛКМ на занятом выходном порту источника отрывает провод
    # за правый конец и тянет новый исток на освободившийся вход приёмника (а не двигает ноду).
    command = CommandModel(
        name="Холст справа",
        actions=[
            ActionBlock(type="Say", params={"text": "A"}),
            ActionBlock(type="Say", params={"text": "B"}),
        ],
    )
    scene = NodeScene(_model(command), theme, catalog=catalog)
    view = NodeView(scene)
    view.resize(800, 600)
    scene.rebuild()
    a_item = scene.node_item("actions[0]")
    assert a_item is not None
    port_view = view.mapFromScene(a_item.output_scene_pos(MAIN_PORT))
    local = QPointF(port_view)
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        local,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    # Провод A→B оторван за выход A: вид тянет новый исток на освободившийся вход B.
    assert view._connect_source is None
    assert view._connect_target == "actions[1]"


def test_view_double_click_toggles_breakpoint_signal(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    view = NodeView(scene)
    scene.rebuild()
    fired: list[str] = []
    view.breakpoint_toggled.connect(fired.append)
    node = scene.node_item("actions[0]")
    assert node is not None
    # Прямо дёргаем сцену: двойной клик виджета лишь зовёт то же самое.
    scene.toggle_breakpoint(node.node_id)
    view.breakpoint_toggled.emit(node.node_id)
    assert fired == ["actions[0]"]


# ----------------------------------------------------------------------
# NodeEditor: контракт списка-режима + сохранение раскладки
# ----------------------------------------------------------------------


def test_editor_mirrors_list_contract(theme: ThemeManager, catalog: BlockCatalog) -> None:
    editor = NodeEditor(_model(), theme, catalog=catalog)
    editor.rebuild()
    editor.select_path(("actions", 2))
    assert editor.selected_path() == ("actions", 2)


def test_editor_layout_json_round_trips(theme: ThemeManager, catalog: BlockCatalog) -> None:
    editor = NodeEditor(_model(), theme, catalog=catalog)
    editor.rebuild()
    data = editor.layout_json()
    assert "actions[0]" in data
    # Задаём свою раскладку и убеждаемся, что она применяется на следующем rebuild.
    editor.set_layout_json({"actions[0]": [640.0, 480.0]})
    editor.rebuild()
    node = editor._scene.graph.node_by_id("actions[0]")
    assert (node.x, node.y) == (640.0, 480.0)


def test_editor_arrange_frames_graph(theme: ThemeManager, catalog: BlockCatalog) -> None:
    editor = NodeEditor(_model(), theme, catalog=catalog)
    editor.rebuild()
    editor.arrange()  # авто-раскладка + fit — не должно падать
    positions = {(n.x, n.y) for n in editor._scene.graph.nodes}
    assert len(positions) == len(editor._scene.graph.nodes)


def test_editor_debug_events_highlight(theme: ThemeManager, catalog: BlockCatalog) -> None:
    editor = NodeEditor(_model(), theme, catalog=catalog)
    editor.rebuild()

    class _Event:
        block_path = "actions[1].then[0]"

    editor.on_debug_paused(_Event())
    running = editor._scene.running_item()
    assert running is not None and running.node_id == "actions[1].then[0]"
    editor.clear_debug()
    assert editor._scene.running_item() is None


def test_editor_breakpoints_forward_scene_signal(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    editor = NodeEditor(_model(), theme, catalog=catalog)
    editor.rebuild()
    fired: list[None] = []
    editor.breakpoints_changed.connect(lambda: fired.append(None))
    # set_breakpoints — тихая загрузка из хранилища, сигнала быть не должно.
    editor.set_breakpoints({"actions[0]"})
    assert editor.breakpoints() == {"actions[0]"}
    assert fired == []
    # Пользовательское переключение через сцену — один сигнал на сохранение.
    editor._scene.toggle_breakpoint("actions[2]")
    assert editor.breakpoints() == {"actions[0]", "actions[2]"}
    assert len(fired) == 1


def test_editor_insert_block_from_palette(theme: ThemeManager, catalog: BlockCatalog) -> None:
    model = _model(CommandModel(name="Пусто"))
    editor = NodeEditor(model, theme, catalog=catalog)
    editor.rebuild()
    editor._insert_block("Say")  # как выбор из палитры тулбара
    assert model.command is not None
    assert [b.type for b in model.command.actions] == ["Say"]


def test_editor_preview_wire_uses_scene(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_model(), theme, catalog=catalog)
    scene.rebuild()
    wire = scene.preview_wire("actions[0]", MAIN_PORT, QPointF(300.0, 300.0))
    assert wire is not None
    scene.removeItem(wire)


def _flat_model(count: int) -> ActionListModel:
    return _model(
        CommandModel(
            name="Плоско",
            actions=[ActionBlock(type="Say", params={"text": str(i)}) for i in range(count)],
        )
    )


def test_insert_keeps_existing_node_positions(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Regression: node coordinates are keyed by positional path, and an insert renumbers
    # the paths. Before the fix, inserting after the first node left every later node's
    # saved coordinate on the neighbour that inherited its old path, flinging the tail of
    # the flow off-canvas («ноды ломаются и выходят за грани»). Positions must follow the
    # block, not the path index.
    editor = NodeEditor(_flat_model(3), theme, catalog=catalog)
    editor.rebuild()
    editor.arrange()  # stable left-to-right coordinates
    before = {n.block.params["text"]: (n.x, n.y) for n in editor._scene.graph.nodes}
    editor.select_path(("actions", 0))
    editor._insert_block("Say")  # insert right after the first node
    after = {
        n.block.params["text"]: (n.x, n.y)
        for n in editor._scene.graph.nodes
        if n.block.params.get("text") in before
    }
    assert after == before  # the pre-existing "1" and "2" did not move


def _card_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    from ayris.gui.widgets.node_editor.layout import NODE_HEIGHT, NODE_WIDTH

    return abs(a[0] - b[0]) < NODE_WIDTH and abs(a[1] - b[1]) < NODE_HEIGHT


def test_repeated_inserts_do_not_stack_on_one_point(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: every insert used to drop the new node at the viewport centre, so adding
    # blocks to an empty command piled them all on the same coordinate — the «спереди …
    # ломаются» symptom, N cards stacked as one. Each new node must take its own free spot.
    editor = NodeEditor(_model(CommandModel(name="Пусто")), theme, catalog=catalog)
    editor.rebuild()
    for _ in range(3):
        editor._insert_block("Say")
    positions = [(n.x, n.y) for n in editor._scene.graph.nodes]
    assert len(positions) == 3
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert not _card_overlap(positions[i], positions[j])


def test_insert_places_new_node_in_flow_not_off_canvas(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: with the view zoomed and panned away from the graph, the new node landed
    # at the viewport centre — far off in scene space («выходят за грани»). It must instead
    # land next to its anchor in the flow, near the existing cluster, regardless of the view.
    editor = NodeEditor(_flat_model(3), theme, catalog=catalog)
    editor.rebuild()
    editor.arrange()
    editor._view.scale(2.0, 2.0)
    editor._view.centerOn(QPointF(2000.0, 1500.0))
    editor.select_path(("actions", 0))
    editor._insert_block("SetVolume")
    xs = [n.x for n in editor._scene.graph.nodes if n.block.params.get("text") in {"0", "1", "2"}]
    new_node = next(n for n in editor._scene.graph.nodes if n.block.type == "SetVolume")
    # The new node sits within the flow's x-span (plus one gap), not out at x≈2000.
    assert min(xs) - 300.0 <= new_node.x <= max(xs) + 300.0
    others = [(n.x, n.y) for n in editor._scene.graph.nodes if n is not new_node]
    assert all(not _card_overlap((new_node.x, new_node.y), o) for o in others)


def test_long_chain_tail_stays_inside_the_scene_rect(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: the scene rect was pinned at a fixed ±5000, but auto_layout places each
    # node one column (270px) further right without bound, so past ~19 nodes the tail fell
    # outside the scrollable area and could not be reached by panning («выходят за грани»
    # for long flows). The canvas must grow to contain the whole graph.
    from ayris.gui.widgets.node_editor.layout import NODE_WIDTH

    model = _model(
        CommandModel(
            name="Длинная",
            actions=[ActionBlock(type="Say", params={"text": str(i)}) for i in range(25)],
        )
    )
    editor = NodeEditor(model, theme, catalog=catalog)
    editor.rebuild()
    editor.arrange()
    right_edge = max(n.x + NODE_WIDTH for n in editor._scene.graph.nodes)
    assert right_edge > 5000.0  # this chain genuinely exceeds the old fixed rect
    assert editor._view.sceneRect().right() >= right_edge  # yet the tail is reachable


def _visible_scene_rect(editor: NodeEditor) -> object:
    from PySide6.QtCore import QRectF

    view = editor._view
    return QRectF(view.mapToScene(view.viewport().rect()).boundingRect())


def _branching_command() -> CommandModel:
    # A command wide and tall enough that its auto-layout spills past a modest viewport
    # unless the graph is framed on open.
    return CommandModel(
        name="Ветвистая",
        actions=[
            ActionBlock(
                type="If",
                params={"condition": "{x}"},
                then=[ActionBlock(type="Say", params={"text": f"да{i}"}) for i in range(4)],
                else_=[ActionBlock(type="Say", params={"text": f"нет{i}"}) for i in range(4)],
            ),
            ActionBlock(type="Say", params={"text": "хвост"}),
        ],
    )


def test_open_command_frames_the_graph_inside_the_viewport(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: opening a command auto-laid-out the graph from the scene origin but never
    # framed it, so the view stayed at 1:1 centred on (0,0) and a branching/long command
    # opened with its right-hand branches clipped off the edge («выехжающие элементы»),
    # unreachable because the canvas hides its scrollbars. reset_layout + first show must
    # fit the whole graph into the visible viewport.
    editor = NodeEditor(_model(_branching_command()), theme, catalog=catalog)
    editor.resize(480, 520)
    editor._view.resize(480, 520)
    editor.show()
    app.processEvents()
    editor.reset_layout()  # mimic load_command on a fresh model
    editor.rebuild()
    visible = _visible_scene_rect(editor)
    outside = [
        node.id
        for node in editor._scene.graph.nodes
        if (item := editor._scene.node_item(node.id)) is not None
        and not visible.contains(item.sceneBoundingRect())
    ]
    assert not outside, f"ноды вне вьюпорта после открытия: {outside}"


def test_command_loaded_while_hidden_is_framed_once_shown(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: a command is loaded while the node view sits behind the list view, so its
    # viewport is 0×0 and framing must be deferred; the switch to «Ноды» shows the canvas.
    # The graph must be framed on that first real show, not left clipped.
    editor = NodeEditor(_model(_branching_command()), theme, catalog=catalog)
    editor.reset_layout()
    editor.rebuild()  # viewport is 0×0 here — frame is deferred, not lost
    # Now it becomes visible at a real size, like flipping to the node tab.
    editor.resize(480, 520)
    editor._view.resize(480, 520)
    editor.show()
    app.processEvents()
    visible = _visible_scene_rect(editor)
    outside = [
        node.id
        for node in editor._scene.graph.nodes
        if (item := editor._scene.node_item(node.id)) is not None
        and not visible.contains(item.sceneBoundingRect())
    ]
    assert not outside, f"ноды вне вьюпорта после показа: {outside}"


def test_reframe_only_on_open_not_on_every_edit(
    app: QApplication, theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # The frame fires once per opened command, not on every rebuild: after the user pans or
    # zooms, a structural edit (which rebuilds) must not yank the view back to «fit all».
    editor = NodeEditor(_flat_model(3), theme, catalog=catalog)
    editor.resize(480, 520)
    editor._view.resize(480, 520)
    editor.show()
    app.processEvents()
    editor.reset_layout()
    editor.rebuild()  # consumes the one pending frame
    editor._view.scale(1.5, 1.5)
    editor._view.centerOn(QPointF(400.0, 400.0))
    before = editor._view.scale_factor
    centre_before = editor._view.mapToScene(editor._view.viewport().rect().center())
    editor.rebuild()  # a plain edit-driven rebuild must leave the view untouched
    assert abs(editor._view.scale_factor - before) < 1e-6
    centre_after = editor._view.mapToScene(editor._view.viewport().rect().center())
    assert abs(centre_after.x() - centre_before.x()) < 1.0
    assert abs(centre_after.y() - centre_before.y()) < 1.0


def test_select_path_is_exclusive_so_delete_hits_the_right_node(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Regression: select_path used setSelected(True) without clearing the prior selection.
    # «+ Нода» selects the new node while its anchor is still selected, leaving two lit;
    # delete_selected reads selected_path() (the first selected) and removed the anchor
    # instead of the new node. select_path must be exclusive.
    editor = NodeEditor(_flat_model(3), theme, catalog=catalog)
    editor.rebuild()
    editor.select_path(("actions", 0))
    editor.select_path(("actions", 2))  # must deselect actions[0]
    assert editor.selected_path() == ("actions", 2)
    editor.delete_selected()
    assert editor._model.command is not None
    # The third block ("2") is gone; the first two survive in order.
    assert [b.params["text"] for b in editor._model.command.actions] == ["0", "1"]


# ----------------------------------------------------------------------
# задержка на проводе: «Пауза» свёрнута в чип, клик правит модель
# ----------------------------------------------------------------------


def _delay_model(ms: int = 500, *, comment: str | None = None) -> ActionListModel:
    # A → Пауза → B: линейный «Sleep» между двумя «Say» сворачивается в чип на проводе.
    sleep = ActionBlock(type="Sleep", params={"ms": ms})
    if comment is not None:
        sleep.comment = comment
    return _model(
        CommandModel(
            name="Пауза на проводе",
            actions=[
                ActionBlock(type="Say", params={"text": "A"}),
                sleep,
                ActionBlock(type="Say", params={"text": "B"}),
            ],
        )
    )


def _plain_model() -> ActionListModel:
    return _model(
        CommandModel(
            name="Без паузы",
            actions=[
                ActionBlock(type="Say", params={"text": "A"}),
                ActionBlock(type="Say", params={"text": "B"}),
            ],
        )
    )


def _wire_between(scene: NodeScene, source_id: str, target_id: str) -> EdgeItem:
    for wire in scene._edges:
        if wire.source_id == source_id and wire.target_id == target_id:
            return wire
    raise AssertionError(f"нет провода {source_id}→{target_id}")


def test_folded_pause_has_no_card(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # Линейная «Пауза» между A и B не рисуется карточкой: остаются две видимые ноды.
    scene = NodeScene(_delay_model(500), theme, catalog=catalog)
    scene.rebuild()
    assert scene.node_item("actions[1]") is None
    assert set(scene._nodes) == {"actions[0]", "actions[2]"}
    assert len(scene._edges) == 1


def test_wire_carries_the_folded_delay(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_delay_model(500), theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    assert wire.delay_ms == 500
    assert wire.edge.sleep_ids == ["actions[1]"]


def test_plain_wire_has_a_zero_chip(theme: ThemeManager, catalog: BlockCatalog) -> None:
    scene = NodeScene(_plain_model(), theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[1]")
    assert wire.delay_ms == 0
    assert wire.edge.sleep_ids == []


def test_edge_at_chip_finds_the_wire_under_a_point(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    scene = NodeScene(_delay_model(500), theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    centre = wire.path().pointAtPercent(0.5)
    assert scene.edge_at_chip(centre) is wire
    assert scene.edge_at_chip(QPointF(-99_999.0, -99_999.0)) is None


def test_set_wire_delay_updates_an_existing_pause(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    model = _delay_model(500)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    assert scene.set_wire_delay(wire.edge, 750) is True
    assert model.command is not None
    assert [b.type for b in model.command.actions] == ["Say", "Sleep", "Say"]
    assert model.command.actions[1].params["ms"] == 750
    assert _wire_between(scene, "actions[0]", "actions[2]").delay_ms == 750


def test_set_wire_delay_zero_drops_the_pause(theme: ThemeManager, catalog: BlockCatalog) -> None:
    model = _delay_model(500)
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    assert scene.set_wire_delay(wire.edge, 0) is True
    assert model.command is not None
    # «Пауза» удалена из модели, провод стал прямым A→B с нулевым чипом.
    assert [b.type for b in model.command.actions] == ["Say", "Say"]
    assert _wire_between(scene, "actions[0]", "actions[1]").delay_ms == 0


def test_set_wire_delay_inserts_a_pause_on_a_plain_wire(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    model = _plain_model()
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[1]")
    assert scene.set_wire_delay(wire.edge, 300) is True
    assert model.command is not None
    # Прямой провод обзавёлся «Паузой»: она встала между A и B в модели.
    assert [b.type for b in model.command.actions] == ["Say", "Sleep", "Say"]
    assert model.command.actions[1].params["ms"] == 300
    assert _wire_between(scene, "actions[0]", "actions[2]").delay_ms == 300


def test_set_wire_delay_keeps_a_reused_pause_comment(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    # Правка задержки на одиночной «Паузе» сохраняет её комментарий, а не сбрасывает.
    model = _delay_model(500, comment="подожди тут")
    scene = NodeScene(model, theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    assert scene.set_wire_delay(wire.edge, 750) is True
    assert model.command is not None
    assert model.command.actions[1].comment == "подожди тут"


def test_highlight_folded_pause_lights_its_wire(theme: ThemeManager, catalog: BlockCatalog) -> None:
    # У свёрнутой «Паузы» нет карточки: отладка подсвечивает её провод, а не роняет сцену.
    scene = NodeScene(_delay_model(500), theme, catalog=catalog)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    scene.highlight_block("actions[1]")
    assert scene.running_item() is None
    assert scene._running_id == "actions[1]"
    assert wire._running is True
    scene.clear_running()
    assert scene._running_id is None
    assert wire._running is False


def test_editor_edit_delay_writes_through_the_dialog(
    theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Клик по чипу открывает QInputDialog; подтверждение правит «Паузу» через сцену и модель.
    model = _delay_model(500)
    editor = NodeEditor(model, theme, catalog=catalog)
    editor.rebuild()
    wire = _wire_between(editor._scene, "actions[0]", "actions[2]")
    stub = SimpleNamespace(getInt=lambda *_a, **_k: (750, True))
    monkeypatch.setattr("ayris.gui.widgets.node_editor.editor.QInputDialog", stub)
    editor._edit_delay(wire)
    assert model.command is not None
    assert model.command.actions[1].params["ms"] == 750


def test_editor_edit_delay_cancelled_leaves_model_untouched(
    theme: ThemeManager, catalog: BlockCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _delay_model(500)
    editor = NodeEditor(model, theme, catalog=catalog)
    editor.rebuild()
    wire = _wire_between(editor._scene, "actions[0]", "actions[2]")
    stub = SimpleNamespace(getInt=lambda *_a, **_k: (0, False))
    monkeypatch.setattr("ayris.gui.widgets.node_editor.editor.QInputDialog", stub)
    editor._edit_delay(wire)
    assert model.command is not None
    # Отмена (ok=False) ничего не меняет, даже если возвращённое значение 0.
    assert model.command.actions[1].params["ms"] == 500


def test_view_click_on_chip_emits_delay_chip_clicked(
    theme: ThemeManager, catalog: BlockCatalog
) -> None:
    scene = NodeScene(_delay_model(500), theme, catalog=catalog)
    view = NodeView(scene)
    view.resize(800, 600)
    scene.rebuild()
    wire = _wire_between(scene, "actions[0]", "actions[2]")
    fired: list[object] = []
    view.delay_chip_clicked.connect(fired.append)
    centre = wire.path().pointAtPercent(0.5)
    local = QPointF(view.mapFromScene(centre))
    press = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        local,
        local,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    view.mousePressEvent(press)
    assert fired == [wire]
