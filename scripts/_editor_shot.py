"""Offscreen grab: render the command editor's Обзор and Звуки tabs to PNG."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from ayris.core.database import Database  # noqa: E402
from ayris.core.models import Command  # noqa: E402
from ayris.core.repositories import Repositories  # noqa: E402
from ayris.gui.theme import ThemeManager  # noqa: E402
from ayris.gui.widgets.command_tree_model import CommandTreeStore  # noqa: E402
from ayris.gui.widgets.macro_editor import MacroEditor  # noqa: E402


RAMP = " .:-=+*#%@"


def ascii_art(img: QImage, cols: int = 120) -> str:
    w, h = img.width(), img.height()
    step = max(1, w // cols)
    rows = []
    for y in range(0, h, step * 2):
        line = []
        for x in range(0, w, step):
            c = img.pixelColor(x, y)
            lum = (c.red() * 30 + c.green() * 59 + c.blue() * 11) // 100
            line.append(RAMP[min(len(RAMP) - 1, lum * len(RAMP) // 256)])
        rows.append("".join(line))
    return "\n".join(rows)


def grab(widget: QWidget, path: str) -> None:
    from PySide6.QtGui import QColor

    img = QImage(widget.size(), QImage.Format.Format_ARGB32)
    img.fill(QColor("#FFFFFF"))
    painter = QPainter(img)
    widget.render(painter, QPoint(0, 0))
    painter.end()
    img.save(path)
    print(f"saved {path}: {img.width()}x{img.height()}")
    _ = ascii_art


def main() -> None:
    app = QApplication.instance() or QApplication([])
    theme = ThemeManager(app)
    theme.set_mode("light")
    theme.apply()

    db = Database.open(":memory:")
    repos = Repositories(db)
    profile = repos.profiles.create("Основной", activate=True)
    assert profile.id is not None
    cmd = repos.commands.create(Command(name="привет", profile_id=profile.id))
    assert cmd.id is not None
    store = CommandTreeStore(repos, profile.id)

    editor = MacroEditor(store, theme)
    editor.resize(760, 620)
    editor.load_command(cmd.id)
    editor.show()
    app.processEvents()

    tabs = editor._tabs
    for i in range(tabs.count()):
        if tabs.tabText(i) in ("Обзор", "Звуки"):
            tabs.setCurrentIndex(i)
            app.processEvents()
            grab(editor, f"scripts/_editor_{tabs.tabText(i)}.png")

    editor.stop_autosave()
    editor.close()
    db.close()


if __name__ == "__main__":
    main()
