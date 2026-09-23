"""Проба двух багов нодового редактора: связь орфанов и авто-провода при вставке."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ayris.actions.macros.blocks.catalog import BlockCatalog  # noqa: E402
from ayris.actions.macros.schema import ActionBlock, CommandModel  # noqa: E402
from ayris.gui.theme import ThemeManager  # noqa: E402
from ayris.gui.widgets.action_list import ActionListModel  # noqa: E402
from ayris.gui.widgets.node_editor.bridge import MAIN_PORT  # noqa: E402
from ayris.gui.widgets.node_editor.editor import NodeEditor  # noqa: E402

app = QApplication.instance() or QApplication([])
theme = ThemeManager(app)
theme.apply()
catalog = BlockCatalog()


def _edges(scene: object) -> list[tuple[str, str, str]]:
    return [(e.source_id, e.source_port, e.target_id) for e in scene._graph.edges]


def _detached(model: ActionListModel) -> dict[str, bool]:
    cmd = model.command
    assert cmd is not None
    return {f"actions[{i}]": b.detached for i, b in enumerate(cmd.actions)}


print("=== Сценарий 1: пустая команда, две ноды, связать их ===")
model = ActionListModel(CommandModel(name="t", actions=[]))
editor = NodeEditor(model, theme, catalog=catalog)
editor.reset_layout()
editor.rebuild()
editor._insert_block("Say")
editor._insert_block("SetVolume")
print("после вставки 2 нод:")
print("  detached:", _detached(model))
print("  рёбра:", _edges(editor._scene))
print("  root_id:", editor._scene._graph.root_id)
ids = [n.id for n in editor._scene._graph.nodes]
print("  ноды:", ids)
ok = editor._scene.request_connect(ids[0], MAIN_PORT, ids[1])
print(f"request_connect({ids[0]} -> {ids[1]}) = {ok}")
print("  detached после связи:", _detached(model))
print("  рёбра после связи:", _edges(editor._scene))
print("  root_id после связи:", editor._scene._graph.root_id)
editor.deleteLater()

print()
print("=== Сценарий 2: команда с 1 существующим блоком, добавить ноду ===")
model2 = ActionListModel(CommandModel(name="t2", actions=[ActionBlock(type="Say", params={"text": "уже был"})]))
editor2 = NodeEditor(model2, theme, catalog=catalog)
editor2.reset_layout()
editor2.rebuild()
print("до вставки:")
print("  detached:", _detached(model2))
print("  рёбра:", _edges(editor2._scene))
editor2._insert_block("SetVolume")
print("после вставки 1 ноды:")
print("  detached:", _detached(model2))
print("  рёбра:", _edges(editor2._scene))
print("  root_id:", editor2._scene._graph.root_id)
editor2.deleteLater()

print()
print("=== Сценарий 3: два существующих блока (как из списка), связать 0->1 уже есть ===")
model3 = ActionListModel(
    CommandModel(
        name="t3",
        actions=[
            ActionBlock(type="Say", params={"text": "a"}),
            ActionBlock(type="Say", params={"text": "b"}),
        ],
    )
)
editor3 = NodeEditor(model3, theme, catalog=catalog)
editor3.reset_layout()
editor3.rebuild()
print("  detached:", _detached(model3))
print("  рёбра:", _edges(editor3._scene))
editor3.deleteLater()
