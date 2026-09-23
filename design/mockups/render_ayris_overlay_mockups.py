"""Render high-fidelity Ayris main-overlay concept variants.

The geometry and labels are deliberately deterministic so the mockups keep
Russian text and control states exact.  This is a design artifact, not runtime
application code.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication


W, H = 1920, 1200
OUT = Path(__file__).resolve().parent

BG = "#090712"
SURFACE = "#141022"
SURFACE_2 = "#211A35"
BORDER = "#3A3154"
ACCENT = "#9B6DFF"
ACCENT_HOVER = "#AE8AFF"
TEXT = "#F7F3FF"
SECONDARY = "#C9BFE0"
MUTED = "#9287AA"
SUCCESS = "#45D6A1"
WARNING = "#F6C85F"
ERROR = "#FF6685"
INFO = "#68B8FF"
CYAN = "#59E1EE"


@dataclass(frozen=True)
class Variant:
    filename: str
    eyebrow: str
    title: str
    state: str
    state_note: str
    sphere_color: str
    network: str
    network_color: str
    profile: str
    mode: str
    mic_on: bool
    transcript: str
    response: str
    footer_hint: str
    macro: bool = False
    error: bool = False


VARIANTS = (
    Variant(
        "ayris-main-idle.png",
        "ОСНОВНОЙ ОВЕРЛЕЙ · ГОТОВА",
        "Спокойный режим",
        "Готова",
        "Скажите «Айрис» или введите команду",
        ACCENT,
        "Онлайн",
        SUCCESS,
        "Дом",
        "Hybrid",
        True,
        "Айрис, какая погода сегодня?",
        "В Санкт-Петербурге +18°, облачно. Дождь ожидается после 19:00.",
        "Enter — отправить  ·  ↑↓ — история команд",
    ),
    Variant(
        "ayris-main-listening.png",
        "ОСНОВНОЙ ОВЕРЛЕЙ · СЛУШАЮ",
        "Активное распознавание",
        "Слушаю",
        "Говорите — уровень голоса управляет сферой",
        CYAN,
        "Онлайн",
        SUCCESS,
        "Дом",
        "PTT",
        True,
        "Поставь таймер на двадцать минут…",
        "Распознаю речь локально",
        "Удерживайте Ctrl + Space  ·  отпустите, чтобы выполнить",
    ),
    Variant(
        "ayris-main-working.png",
        "ОСНОВНОЙ ОВЕРЛЕЙ · ВЫПОЛНЯЮ",
        "Команда и сценарий",
        "Выполняю",
        "Сценарий «Начать рабочий день» · шаг 3 из 4",
        ACCENT_HOVER,
        "Локально",
        INFO,
        "Работа",
        "Always",
        False,
        "Айрис, начни рабочий день",
        "Открываю приложения и восстанавливаю рабочее пространство.",
        "Голосовые команды временно выключены · текстовый ввод доступен",
        macro=True,
    ),
)


def c(value: str, alpha: int | None = None) -> QColor:
    color = QColor(value)
    if alpha is not None:
        color.setAlpha(alpha)
    return color


class Canvas:
    def __init__(self, variant: Variant) -> None:
        self.variant = variant
        self.image = QImage(W, H, QImage.Format.Format_ARGB32_Premultiplied)
        self.image.fill(c(BG))
        self.p = QPainter(self.image)
        self.p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.family = "Inter" if "Inter" in QFontDatabase.families() else "Segoe UI"

    def finish(self) -> None:
        self.p.end()
        self.image.save(str(OUT / self.variant.filename), "PNG", 96)

    def font(self, size: int, weight: int = 400) -> QFont:
        f = QFont(self.family)
        f.setPixelSize(size)
        f.setWeight(QFont.Weight(weight))
        return f

    def rr(self, rect: QRectF, radius: float, fill: str | QColor, stroke: str | QColor | None = None, width: float = 1.0) -> None:
        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        self.p.fillPath(path, c(fill) if isinstance(fill, str) else fill)
        if stroke is not None:
            self.p.setPen(QPen(c(stroke) if isinstance(stroke, str) else stroke, width))
            self.p.drawPath(path)
        self.p.setPen(Qt.PenStyle.NoPen)

    def text(self, x: float, y: float, w: float, h: float, value: str, size: int, color: str = TEXT, weight: int = 400, align: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter) -> None:
        self.p.setFont(self.font(size, weight))
        self.p.setPen(c(color))
        self.p.drawText(QRectF(x, y, w, h), align | Qt.TextFlag.TextWordWrap, value)

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 2.0) -> None:
        self.p.setPen(QPen(c(color), width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        self.p.drawLine(QPointF(x1, y1), QPointF(x2, y2))

    def circle(self, x: float, y: float, r: float, fill: str | QColor, stroke: str | None = None, width: float = 1.0) -> None:
        self.p.setBrush(c(fill) if isinstance(fill, str) else fill)
        self.p.setPen(Qt.PenStyle.NoPen if stroke is None else QPen(c(stroke), width))
        self.p.drawEllipse(QPointF(x, y), r, r)

    def icon(self, kind: str, cx: float, cy: float, color: str = SECONDARY, scale: float = 1.0) -> None:
        pen = QPen(c(color), 2.1 * scale, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        self.p.setPen(pen)
        self.p.setBrush(Qt.BrushStyle.NoBrush)
        if kind == "settings":
            self.p.drawEllipse(QPointF(cx, cy), 4 * scale, 4 * scale)
            self.p.drawEllipse(QPointF(cx, cy), 9 * scale, 9 * scale)
            for i in range(8):
                a = i * math.pi / 4
                self.p.drawLine(QPointF(cx + math.cos(a) * 10 * scale, cy + math.sin(a) * 10 * scale), QPointF(cx + math.cos(a) * 13 * scale, cy + math.sin(a) * 13 * scale))
        elif kind == "hide":
            self.p.drawLine(QPointF(cx - 9 * scale, cy), QPointF(cx + 9 * scale, cy))
        elif kind == "mic":
            self.p.drawRoundedRect(QRectF(cx - 5 * scale, cy - 10 * scale, 10 * scale, 16 * scale), 5 * scale, 5 * scale)
            self.p.drawArc(QRectF(cx - 10 * scale, cy - 4 * scale, 20 * scale, 16 * scale), 180 * 16, 180 * 16)
            self.p.drawLine(QPointF(cx, cy + 12 * scale), QPointF(cx, cy + 16 * scale))
            self.p.drawLine(QPointF(cx - 6 * scale, cy + 16 * scale), QPointF(cx + 6 * scale, cy + 16 * scale))
        elif kind == "send":
            path = QPainterPath(QPointF(cx - 10 * scale, cy - 8 * scale))
            path.lineTo(QPointF(cx + 11 * scale, cy))
            path.lineTo(QPointF(cx - 10 * scale, cy + 8 * scale))
            path.lineTo(QPointF(cx - 5 * scale, cy))
            path.closeSubpath()
            self.p.drawPath(path)
        elif kind == "timer":
            self.p.drawEllipse(QPointF(cx, cy + 1 * scale), 10 * scale, 10 * scale)
            self.p.drawLine(QPointF(cx, cy - 9 * scale), QPointF(cx, cy - 14 * scale))
            self.p.drawLine(QPointF(cx - 4 * scale, cy - 14 * scale), QPointF(cx + 4 * scale, cy - 14 * scale))
            self.p.drawLine(QPointF(cx, cy + 1 * scale), QPointF(cx + 5 * scale, cy - 3 * scale))
        elif kind == "copy":
            self.p.drawRoundedRect(QRectF(cx - 8 * scale, cy - 8 * scale, 13 * scale, 13 * scale), 2, 2)
            self.p.drawRoundedRect(QRectF(cx - 3 * scale, cy - 3 * scale, 13 * scale, 13 * scale), 2, 2)
        elif kind == "chevron":
            self.p.drawLine(QPointF(cx - 5 * scale, cy - 3 * scale), QPointF(cx, cy + 2 * scale))
            self.p.drawLine(QPointF(cx, cy + 2 * scale), QPointF(cx + 5 * scale, cy - 3 * scale))
        elif kind == "close":
            self.p.drawLine(QPointF(cx - 6 * scale, cy - 6 * scale), QPointF(cx + 6 * scale, cy + 6 * scale))
            self.p.drawLine(QPointF(cx + 6 * scale, cy - 6 * scale), QPointF(cx - 6 * scale, cy + 6 * scale))
        self.p.setPen(Qt.PenStyle.NoPen)

    def background(self) -> None:
        # Soft, asymmetric ambient light keeps the overlay legible without a generic wallpaper.
        for x, y, radius, color, alpha in ((250, 170, 520, ACCENT, 25), (1690, 1020, 580, INFO, 18), (1300, 110, 340, CYAN, 10)):
            grad = QRadialGradient(QPointF(x, y), radius)
            grad.setColorAt(0, c(color, alpha))
            grad.setColorAt(1, c(color, 0))
            self.p.fillRect(self.image.rect(), grad)
        self.p.setPen(QPen(c(BORDER, 28), 1))
        for x in range(0, W, 64):
            self.p.drawLine(x, 0, x, H)
        for y in range(0, H, 64):
            self.p.drawLine(0, y, W, y)
        self.p.setPen(Qt.PenStyle.NoPen)
        self.text(94, 42, 1200, 26, self.variant.eyebrow, 13, MUTED, 500)
        self.text(92, 69, 1000, 48, self.variant.title, 30, TEXT, 700)
        self.text(1440, 72, 388, 34, "Ayris · проектная визуализация", 13, MUTED, 400, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def shell(self) -> QRectF:
        shell = QRectF(92, 138, 1736, 986)
        shadow = QColor(0, 0, 0, 95)
        self.rr(shell.translated(0, 18), 30, shadow)
        grad = QLinearGradient(shell.topLeft(), shell.bottomRight())
        grad.setColorAt(0, c("#171126"))
        grad.setColorAt(0.6, c("#0E0B18"))
        grad.setColorAt(1, c("#151020"))
        self.rr(shell, 30, grad, c(BORDER, 210), 1)
        self.line(92, 232, 1828, 232, BORDER, 1)
        return shell

    def header(self) -> None:
        self.circle(136, 185, 17, ACCENT)
        self.circle(136, 185, 6, TEXT)
        self.text(167, 158, 220, 34, "Ayris", 24, TEXT, 700)
        self.text(167, 188, 300, 25, "голосовой помощник", 12, MUTED, 400)
        # Network status
        self.rr(QRectF(1120, 163, 118, 44), 22, c(SURFACE_2, 180), BORDER)
        self.circle(1144, 185, 5, self.variant.network_color)
        self.text(1158, 163, 66, 44, self.variant.network, 13, SECONDARY, 500)
        # Profile selector
        self.rr(QRectF(1252, 163, 176, 44), 12, SURFACE_2, BORDER)
        self.circle(1278, 185, 11, ACCENT)
        self.text(1300, 163, 92, 44, self.variant.profile, 13, TEXT, 500)
        self.icon("chevron", 1405, 186, MUTED, .85)
        # Settings / hide
        for x, icon in ((1460, "settings"), (1518, "hide")):
            self.rr(QRectF(x, 163, 44, 44), 12, SURFACE_2, BORDER)
            self.icon(icon, x + 22, 185, SECONDARY, .8)
        self.text(1588, 163, 190, 44, "Поверх всех окон", 12, MUTED, 400, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def sphere(self) -> None:
        cx, cy, radius = 437, 451, 154
        color = self.variant.sphere_color
        glow = QRadialGradient(QPointF(cx, cy), 235)
        glow.setColorAt(0, c(color, 38))
        glow.setColorAt(.62, c(color, 13))
        glow.setColorAt(1, c(color, 0))
        self.p.setBrush(glow)
        self.p.setPen(Qt.PenStyle.NoPen)
        self.p.drawEllipse(QPointF(cx, cy), 235, 235)
        # Perspective Fibonacci point cloud.
        count = 330
        golden = math.pi * (3 - math.sqrt(5))
        phase = {"Готова": .2, "Слушаю": .75, "Выполняю": 1.35}.get(self.variant.state, .2)
        points = []
        for i in range(count):
            yy = 1 - 2 * (i + .5) / count
            rr = math.sqrt(max(0, 1 - yy * yy))
            a = golden * i + phase
            xx, zz = math.cos(a) * rr, math.sin(a) * rr
            # Rotate in two axes.
            xr = xx * math.cos(.42) + zz * math.sin(.42)
            zr = -xx * math.sin(.42) + zz * math.cos(.42)
            yr = yy * math.cos(-.18) - zr * math.sin(-.18)
            zr2 = yy * math.sin(-.18) + zr * math.cos(-.18)
            if self.variant.state == "Выполняю":
                xr += math.sin(i * 1.71) * .025
                yr += math.cos(i * 1.37) * .025
            pulse = 1.0 + (.035 * math.sin(i * .22) if self.variant.state == "Слушаю" else 0)
            perspective = 1 / (1.15 - zr2 * .16)
            sx = cx + xr * radius * pulse * perspective
            sy = cy + yr * radius * pulse * perspective
            points.append((zr2, sx, sy))
        for depth, x, y in sorted(points):
            alpha = int(55 + (depth + 1) * 82)
            size = 1.15 + (depth + 1) * .82
            self.circle(x, y, size, c(color, alpha))
        if self.variant.state == "Слушаю":
            self.p.setBrush(Qt.BrushStyle.NoBrush)
            for offset, alpha in ((18, 65), (34, 30)):
                self.p.setPen(QPen(c(CYAN, alpha), 2))
                self.p.drawEllipse(QPointF(cx, cy), radius + offset, radius + offset)
            self.p.setPen(Qt.PenStyle.NoPen)
        self.text(248, 634, 378, 38, self.variant.state, 25, TEXT, 700, Qt.AlignmentFlag.AlignCenter)
        self.text(235, 674, 404, 52, self.variant.state_note, 13, MUTED, 400, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.waveform(270, 735, 334, 54, color)

    def waveform(self, x: float, y: float, w: float, h: float, color: str) -> None:
        self.line(x, y + h / 2, x + w, y + h / 2, BORDER, 1)
        values = [5, 9, 14, 8, 20, 28, 17, 33, 45, 25, 19, 38, 51, 31, 22, 39, 29, 17, 23, 13, 9, 6]
        if self.variant.state == "Готова":
            values = [max(3, v // 4) for v in values]
        if self.variant.state == "Выполняю":
            values = [max(3, v // 6) for v in values]
        step = w / len(values)
        for i, value in enumerate(values):
            xx = x + step * (i + .5)
            self.line(xx, y + h / 2 - value / 2, xx, y + h / 2 + value / 2, color, 3)

    def mode_controls(self) -> None:
        self.text(216, 817, 210, 26, "Режим прослушивания", 13, SECONDARY, 500)
        rect = QRectF(216, 852, 442, 54)
        self.rr(rect, 14, SURFACE_2, BORDER)
        labels = ("Always", "Hybrid", "PTT")
        seg_w = rect.width() / 3
        for i, label in enumerate(labels):
            selected = label == self.variant.mode
            if selected:
                self.rr(QRectF(rect.x() + i * seg_w + 4, rect.y() + 4, seg_w - 8, rect.height() - 8), 11, ACCENT)
            self.text(rect.x() + i * seg_w, rect.y(), seg_w, rect.height(), label, 13, "#110B1D" if selected else SECONDARY, 600 if selected else 400, Qt.AlignmentFlag.AlignCenter)
        # Mic switch has a semantic label, not color alone.
        self.rr(QRectF(216, 927, 442, 62), 14, SURFACE, BORDER)
        self.circle(248, 958, 18, c(SUCCESS if self.variant.mic_on else ERROR, 35))
        self.icon("mic", 248, 956, SUCCESS if self.variant.mic_on else ERROR, .68)
        if not self.variant.mic_on:
            self.line(237, 945, 259, 967, ERROR, 2.2)
        self.text(280, 937, 240, 24, "Микрофон", 13, TEXT, 600)
        self.text(280, 960, 250, 21, "Включён" if self.variant.mic_on else "Выключен", 12, SUCCESS if self.variant.mic_on else ERROR, 500)
        tx, ty = 584, 958
        self.rr(QRectF(tx - 27, ty - 14, 54, 28), 14, ACCENT if self.variant.mic_on else BORDER)
        self.circle(tx + (13 if self.variant.mic_on else -13), ty, 10, TEXT)

    def dialog(self) -> None:
        x, y, w = 708, 266, 1060
        self.text(x, y, 250, 30, "Диалог", 18, TEXT, 700)
        self.text(x + 760, y, 260, 30, "Последние 8 сообщений", 12, MUTED, 400, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        # User message
        self.rr(QRectF(x + 242, y + 50, 776, 92), 18, SURFACE_2, BORDER)
        self.text(x + 270, y + 65, 644, 24, self.variant.transcript, 14, TEXT, 500)
        self.text(x + 270, y + 100, 190, 20, "Вы · 14:32", 11, MUTED, 400)
        self.icon("copy", x + 985, y + 96, MUTED, .68)
        # Ayris response
        response_color = self.variant.sphere_color
        self.rr(QRectF(x, y + 164, 848, 112), 18, c(SURFACE, 245), BORDER)
        self.circle(x + 31, y + 194, 13, response_color)
        self.circle(x + 31, y + 194, 4.5, TEXT)
        self.text(x + 58, y + 178, 734, 52, self.variant.response, 14, TEXT, 400)
        self.text(x + 58, y + 236, 190, 20, "Ayris · 14:32", 11, MUTED, 400)
        self.icon("copy", x + 814, y + 243, MUTED, .68)
        if self.variant.state == "Слушаю":
            self.rr(QRectF(x, y + 300, 1018, 58), 14, c(CYAN, 14), c(CYAN, 100))
            self.circle(x + 28, y + 329, 5, CYAN)
            self.text(x + 48, y + 300, 940, 58, "Транскрипт обновляется в реальном времени · обработка на устройстве", 12, SECONDARY, 500)
        elif self.variant.macro:
            self.macro_panel(x, y + 298, 1018)
        else:
            self.rr(QRectF(x, y + 300, 1018, 58), 14, c(ACCENT, 12), c(ACCENT, 70))
            self.text(x + 24, y + 300, 970, 58, "История хранится в профиле «Дом» · в оверлее показаны только последние строки", 12, SECONDARY, 400)

    def macro_panel(self, x: float, y: float, w: float) -> None:
        self.rr(QRectF(x, y, w, 150), 16, c(ACCENT, 13), c(ACCENT, 100))
        self.circle(x + 30, y + 31, 6, ACCENT)
        self.text(x + 48, y + 16, 500, 30, "Начать рабочий день", 14, TEXT, 600)
        self.text(x + 760, y + 16, 224, 30, "3 из 4 · 6 сек", 12, MUTED, 500, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        steps = (("Открыть почту", True), ("Запустить календарь", True), ("Развернуть рабочие окна", False), ("Включить фокус-режим", False))
        sx = x + 28
        for i, (label, done) in enumerate(steps):
            px = sx + i * 238
            self.circle(px, y + 76, 10, SUCCESS if done else (ACCENT if i == 2 else BORDER))
            if done:
                self.line(px - 4, y + 76, px - 1, y + 80, "#110B1D", 2)
                self.line(px - 1, y + 80, px + 5, y + 72, "#110B1D", 2)
            if i < 3:
                self.line(px + 13, y + 76, px + 54, y + 76, BORDER, 2)
            self.text(px - 10, y + 94, 210, 38, label, 11, SECONDARY if i <= 2 else MUTED, 400)
        self.rr(QRectF(x + w - 122, y + 96, 94, 36), 10, SURFACE_2, BORDER)
        self.text(x + w - 122, y + 96, 94, 36, "Отменить", 12, ERROR, 500, Qt.AlignmentFlag.AlignCenter)

    def timers(self) -> None:
        x, y, w = 708, 751 if not self.variant.macro else 843, 1060
        # In working mode this becomes a compact lower rail under the macro.
        if self.variant.macro:
            y = 786
        self.text(x, y, 250, 28, "Таймеры и напоминания", 15, TEXT, 600)
        self.text(x + 810, y, 208, 28, "2 активных", 12, MUTED, 400, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        cards_y = y + 40
        card_w = 510
        data = (("Перерыв", "12:46", "через 12 минут", INFO), ("Выключить духовку", "18:30", "сегодня", WARNING))
        for i, (name, value, note, color) in enumerate(data):
            xx = x + i * (card_w + 20)
            self.rr(QRectF(xx, cards_y, card_w, 78), 14, SURFACE, BORDER)
            self.circle(xx + 32, cards_y + 39, 18, c(color, 30))
            self.icon("timer", xx + 32, cards_y + 39, color, .72)
            self.text(xx + 62, cards_y + 12, 250, 28, name, 13, TEXT, 500)
            self.text(xx + 62, cards_y + 39, 220, 24, note, 11, MUTED, 400)
            self.text(xx + 328, cards_y + 14, 120, 28, value, 18, color, 600, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.icon("close", xx + 479, cards_y + 39, MUTED, .68)

    def command(self) -> None:
        x, y, w = 708, 1011, 1060
        if self.variant.macro:
            y = 943
        self.rr(QRectF(x, y, w, 62), 16, SURFACE_2, BORDER)
        self.text(x + 22, y, w - 104, 62, "Введите команду…", 14, MUTED, 400)
        self.rr(QRectF(x + w - 54, y + 9, 44, 44), 12, ACCENT)
        self.icon("send", x + w - 32, y + 31, "#110B1D", .78)
        self.text(x, y + 68, w, 28, self.variant.footer_hint, 11, MUTED, 400)

    def render(self) -> None:
        self.background()
        self.shell()
        self.header()
        self.sphere()
        self.mode_controls()
        self.dialog()
        self.timers()
        self.command()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    del app
    for variant in VARIANTS:
        canvas = Canvas(variant)
        canvas.render()
        canvas.finish()
        print(OUT / variant.filename)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
