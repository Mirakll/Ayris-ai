"""Offscreen render of one SoundBindingRow to inspect the spinbox arrows."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSize
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.schema import SoundStage
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sound_binding import SoundBindingRow

app = QApplication([])
theme = ThemeManager(app)
theme.apply()

row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
row.resize(640, 110)
row.show()
app.processEvents()
row.adjustSize()
row.resize(640, max(110, row.sizeHint().height()))
app.processEvents()

from PySide6.QtCore import Qt

pixmap = row.grab()
scaled = pixmap.scaled(
    pixmap.width() * 2,
    pixmap.height() * 2,
    Qt.AspectRatioMode.IgnoreAspectRatio,
    Qt.TransformationMode.SmoothTransformation,
)
out = "_backups/sound_row.png"
scaled.save(out)
print("saved", out, scaled.size())
