"""Offscreen probe: walk the editor's Обзор/Звуки tabs, print visible widgets."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ayris.core.database import Database  # noqa: E402
from ayris.core.models import Command  # noqa: E402
from ayris.core.repositories import Repositories  # noqa: E402
from ayris.gui.theme import ThemeManager  # noqa: E402
from ayris.gui.widgets.command_tree_model import CommandTreeStore  # noqa: E402
from ayris.gui.widgets.macro_editor import MacroEditor  # noqa: E402
from ayris.gui.widgets.toggle import ToggleSwitch  # noqa: E402


def describe(w: QWidget, root: QWidget) -> str | None:
    if not w.isVisibleTo(root):
        return None
    kind = type(w).__name__
    if isinstance(w, ToggleSwitch):
        return f"TOGGLE checked={w.isChecked()} accName='{w.accessibleName()}'"
    if isinstance(w, QComboBox):
        return f"COMBO '{w.currentText()}'"
    if isinstance(w, QLabel):
        t = w.text().strip()
        return f"LABEL '{t}'" if t else None
    if isinstance(w, QPushButton):
        return f"BUTTON '{w.text()}'"
    if isinstance(w, QLineEdit):
        return f"LINEEDIT ph='{w.placeholderText()}'"
    if isinstance(w, QPlainTextEdit):
        return f"TEXT ph='{w.placeholderText()}'"
    if isinstance(w, QSpinBox):
        return f"SPIN suffix='{w.suffix()}'"
    return None


def dump(root: QWidget) -> None:
    for w in sorted(
        root.findChildren(QWidget),
        key=lambda x: (x.mapTo(root, x.rect().topLeft()).y(), x.mapTo(root, x.rect().topLeft()).x()),
    ):
        line = describe(w, root)
        if line:
            p = w.mapTo(root, w.rect().topLeft())
            print(f"  y={p.y():>3} x={p.x():>3} w={w.width():>3} h={w.height():>2}  {line}")


def main() -> None:
    app = QApplication.instance() or QApplication([])
    theme = ThemeManager(app)
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
            print(f"\n===== вкладка «{tabs.tabText(i)}» =====")
            dump(tabs.currentWidget())

    editor.stop_autosave()
    editor.close()
    db.close()


if __name__ == "__main__":
    main()
