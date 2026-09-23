"""Render the volume ThemedSpinBox at high scale to eyeball the chip and chevrons."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.schema import SoundStage
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sound_binding import SoundBindingRow

app = QApplication([])
theme = ThemeManager(app)
theme.apply()

row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
row._enabled.setCurrentIndex(1)
row._value.setText("привет")
row._volume.setValue(80)
row.resize(640, 120)
row.show()
app.processEvents()

spin = row._volume
# Render the spin box onto a surface tinted like the app background so the crop
# reads the same as on screen (the chip's field colour differs from pure black).
scale = 8
img = QImage(spin.size() * scale, QImage.Format.Format_ARGB32)
img.fill(0xFF120E1E)
painter = QPainter(img)
painter.scale(scale, scale)
spin.render(painter)
painter.end()
out = "_backups/spinbox_crop.png"
img.save(out)
print("saved", out, img.size())
