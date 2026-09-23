"""Confirm two chevrons render in the chip: count arrow-coloured pixels per half."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from ayris.actions.macros.schema import SoundStage
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sound_binding import SoundBindingRow

app = QApplication([])
theme = ThemeManager(app)
theme.apply()

row = SoundBindingRow(theme, SoundStage.ON_SUCCESS)
row._enabled.setCurrentIndex(1)
row.resize(640, 120)
row.show()
app.processEvents()

spin = row._volume
chip = spin._chip_rect()
arrow = QColor(spin.property("arrowColor"))
chip_c = QColor(spin.property("chipColor"))
print("arrow colour", arrow.name(), "chip colour", chip_c.name())

img = QImage(spin.size(), QImage.Format.Format_ARGB32)
img.fill(0)
spin.render(img)


def is_arrowish(c: QColor) -> bool:
    # Arrow stroke is anti-aliased over the chip, so match "closer to arrow than chip".
    da = abs(c.red() - arrow.red()) + abs(c.green() - arrow.green()) + abs(c.blue() - arrow.blue())
    dc = (
        abs(c.red() - chip_c.red())
        + abs(c.green() - chip_c.green())
        + abs(c.blue() - chip_c.blue())
    )
    return c.alpha() > 0 and da < dc


mid = chip.center().y()
upper = lower = 0
for x in range(int(chip.left()), int(chip.right())):
    for y in range(int(chip.top()), int(chip.bottom())):
        if is_arrowish(img.pixelColor(x, y)):
            if y < mid:
                upper += 1
            else:
                lower += 1
print("arrow pixels — upper half:", upper, "lower half:", lower)
