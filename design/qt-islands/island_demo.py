"""Демо «QML-острова внутри QWidget-окна».

Показывает выбранный путь реализации редизайна (вариант 2): каркас окна и
настройки остаются QWidget, а «богатые» сцены (стекло, неон, будущий 3D-орб)
живут как QQuickWidget-острова в том же окне. Токены дизайн-системы приходят в
QML через общий объект `theme` (см. theme_bridge.py) — смена темы нативной
кнопкой мгновенно перекрашивает оба острова через биндинги.

Запуск живого окна:
    _tools/venv/Scripts/python.exe design/qt-islands/island_demo.py

Снимок без окна (доказательство блюра/свечения на GPU):
    _tools/venv/Scripts/python.exe design/qt-islands/island_demo.py --shot design/qt-islands/out
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QUrl, Qt, QTimer
from PySide6.QtQuickWidgets import QQuickWidget
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from theme_bridge import ThemeBridge, available_themes  # noqa: E402


def _island(bridge: ThemeBridge, qml_name: str) -> QQuickWidget:
    w = QQuickWidget()
    w.setResizeMode(QQuickWidget.ResizeMode.SizeRootObjectToView)
    w.rootContext().setContextProperty("theme", bridge)
    w.setSource(QUrl.fromLocalFile(str(_HERE / "qml" / qml_name)))
    if w.status() == QQuickWidget.Status.Error:
        for e in w.errors():
            print("QML error:", e.toString(), file=sys.stderr)
    w.setMinimumSize(420, 320)
    return w


def build_window(bridge: ThemeBridge) -> QWidget:
    win = QWidget()
    win.setWindowTitle("Ayris · QML-острова в QWidget-окне")
    root = QVBoxLayout(win)

    islands = QHBoxLayout()
    islands.addWidget(_island(bridge, "GlassIsland.qml"))
    islands.addWidget(_island(bridge, "NeonIsland.qml"))
    root.addLayout(islands)

    bar = QHBoxLayout()
    bar.addWidget(QLabel("Тема (нативный QWidget-контрол):"))
    combo = QComboBox()
    combo.addItems(available_themes())
    combo.setCurrentText(bridge.property("name"))
    combo.currentTextChanged.connect(bridge.set_theme)
    bar.addWidget(combo)
    bar.addStretch(1)
    root.addLayout(bar)

    win.resize(980, 460)
    return win


def _shot(bridge: ThemeBridge, out_stem: str) -> int:
    """Отрендерить оба острова без показа окна и сохранить PNG."""
    win = build_window(bridge)
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.show()
    app = QApplication.instance()
    for _ in range(8):  # дать Quick-сцене отрисовать несколько кадров
        assert app is not None
        app.processEvents()
        QTimer.singleShot(60, lambda: None)
        app.processEvents()
    ok = 0
    for qw in win.findChildren(QQuickWidget):
        img = qw.grabFramebuffer()
        idx = win.findChildren(QQuickWidget).index(qw)
        path = f"{out_stem}-{idx}.png"
        if not img.isNull() and img.save(path):
            print("сохранён", path, img.width(), "x", img.height())
            ok += 1
        else:
            print("пустой кадр (нет GPU-контекста?):", path, file=sys.stderr)
    return 0 if ok else 2


def main() -> int:
    app = QApplication(sys.argv)
    bridge = ThemeBridge("dark_purple")
    if "--shot" in sys.argv:
        stem = sys.argv[sys.argv.index("--shot") + 1]
        return _shot(bridge, stem)
    win = build_window(bridge)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
