"""Проба: воспроизвести два бага нодового редактора (авто-провода, не соединяется)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ayris.actions.macros.schema import CommandModel  # noqa: E402
from ayris.gui.theme import ThemeManager  # noqa: E402
from ayris.gui.widgets.action_list import ActionListModel  # noqa: E402
from ayris.gui.widgets.node_editor import NodeEditor  # noqa: E402


def dump(tag: str, model: ActionListModel) -> None:
    cmd = model.command
    assert cmd is not None
    print(f"--- {tag} ---")
    for i, b in enumerate(cmd.actions):
        print(f"  actions[{i}] type={b.type!r} detached={b.detached}")


def main() -> None:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    theme = ThemeManager(app)
    theme.apply()

    model = ActionListModel()
    model.set_command(CommandModel(name="Проба", actions=[]))
    editor = NodeEditor(model, theme)

    # Пользователь добавляет три ноды через палитру («+ Нода»).
    editor._insert_block("KeyDown")
    editor._insert_block("KeyUp")
    editor._insert_block("SetVolume")
    dump("после трёх вставок", model)
    graph = editor._scene.graph
    print("  root_id =", graph.root_id)
    print("  edges   =", [e.key for e in graph.edges])

    # Берём id первых двух нод и пробуем соединить их проводом.
    ids = [n.id for n in graph.nodes]
    print("  node ids=", ids)
    src, dst = ids[0], ids[1]
    ok = editor._scene.request_connect(src, MAIN_PORT_GET(), dst)
    print(f"request_connect({src!r} -> {dst!r}) = {ok}")
    dump("после соединения", model)
    graph2 = editor._scene.graph
    print("  root_id =", graph2.root_id)
    print("  edges   =", [e.key for e in graph2.edges])

    editor.close()


def MAIN_PORT_GET() -> str:
    from ayris.gui.widgets.node_editor.bridge import MAIN_PORT

    return MAIN_PORT


if __name__ == "__main__":
    main()
