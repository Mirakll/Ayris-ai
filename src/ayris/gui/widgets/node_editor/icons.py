"""Лаконичные глифы ролей для нодового редактора — общие для холста и палитры.

И карточка-нода (:mod:`node_item`), и каталог блоков (:mod:`block_palette`) рисуют один и
тот же маленький значок роли — микрофон, реплика, молния, ромб, динамик, — поэтому он задан
здесь один раз как :class:`QPainterPath` в поле 24×24 и обводится там, где нужен. Данные
путей перенесены 1:1 из макета (``design/mockups/command_redesign/engine.js``, карта
``ICONS``). Модуль намеренно вынесен из :mod:`bridge`: тот слой держат Qt-free ради чистых
round-trip тестов, а тут уже нужен ``QPainterPath``.
"""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen

from ayris.gui.widgets.node_editor.bridge import NodeRole

__all__ = [
    "block_glyph_path",
    "paint_block_glyph",
    "paint_role_glyph",
    "role_glyph_path",
]


def role_glyph_path(role: NodeRole) -> QPainterPath:
    """Глиф роли в системе координат 24×24, готовый к обводке вызывающим кодом.

    Две дуги (стойка микрофона, звуковая волна) приближены кривыми Безье — на значке 14 px
    отличие от настоящей SVG-дуги незаметно.
    """
    path = QPainterPath()
    if role is NodeRole.TRIGGER:
        path.addRoundedRect(QRectF(9, 3, 6, 11), 3, 3)
        path.moveTo(5, 11)
        path.cubicTo(5, 20.3, 19, 20.3, 19, 11)
        path.moveTo(12, 18)
        path.lineTo(12, 21)
    elif role is NodeRole.RESPONSE:
        path.moveTo(4, 5)
        path.lineTo(20, 5)
        path.lineTo(20, 16)
        path.lineTo(8, 16)
        path.lineTo(4, 20)
        path.closeSubpath()
    elif role is NodeRole.ACTION:
        path.moveTo(13, 2)
        path.lineTo(4, 14)
        path.lineTo(11, 14)
        path.lineTo(10, 22)
        path.lineTo(19, 10)
        path.lineTo(12, 10)
        path.closeSubpath()
    elif role is NodeRole.CONDITION:
        path.moveTo(12, 3)
        path.lineTo(21, 12)
        path.lineTo(12, 21)
        path.lineTo(3, 12)
        path.closeSubpath()
    else:  # NodeRole.SOUND
        path.moveTo(4, 9)
        path.lineTo(4, 15)
        path.lineTo(8, 15)
        path.lineTo(13, 19)
        path.lineTo(13, 5)
        path.lineTo(8, 9)
        path.closeSubpath()
        path.moveTo(17, 8)
        path.quadTo(21, 12, 17, 16)
    return path


def paint_role_glyph(
    painter: QPainter,
    square: QRectF,
    role: NodeRole,
    colour: QColor,
    *,
    side: float = 14.0,
) -> None:
    """Обвести глиф роли по центру ``square`` цветом ``colour``.

    Глиф автора в поле 24×24 масштабируется до ``side`` пикселей и центрируется в квадрате,
    как ``.ico svg`` / ``.ci-ico svg`` в макете. Косметическое перо держит штрих ~1.6 px при
    любом зуме холста, поэтому обводка не толстеет и не истончается вместе с нодой.
    """
    _stroke_glyph(painter, square, role_glyph_path(role), colour, side=side)


def paint_block_glyph(
    painter: QPainter,
    square: QRectF,
    block_type: str,
    role: NodeRole,
    colour: QColor,
    *,
    side: float = 14.0,
) -> None:
    """Обвести глиф КОНКРЕТНОГО блока (по его типу), с откатом к глифу роли.

    В отличие от :func:`paint_role_glyph`, значок передаёт смысл именно этого блока —
    клавиатура у «Нажать клавиши», мышь у «Щёлкнуть», часы у «Ожидание» и т. д., — а не одну
    иконку на всю роль. Неизвестный тип (команда из более новой сборки) откатывается к глифу
    роли, поэтому пустых значков не бывает.
    """
    path = block_glyph_path(block_type)
    if path is None:
        path = role_glyph_path(role)
    _stroke_glyph(painter, square, path, colour, side=side)


def block_glyph_path(block_type: str) -> QPainterPath | None:
    """Глиф блока по его типу в поле 24×24, либо ``None`` для неизвестного типа."""
    ops = _GLYPHS.get(_BLOCK_GLYPHS.get(block_type, ""))
    return _build(ops) if ops is not None else None


def _stroke_glyph(
    painter: QPainter,
    square: QRectF,
    path: QPainterPath,
    colour: QColor,
    *,
    side: float,
) -> None:
    painter.save()
    painter.translate(square.center().x() - side / 2, square.center().y() - side / 2)
    painter.scale(side / 24.0, side / 24.0)
    pen = QPen(colour, 1.6)
    pen.setCosmetic(True)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawPath(path)
    painter.restore()


# ----------------------------------------------------------------------------------------
# Per-block glyph library. A glyph is authored as a compact op list in the 24×24 box (the
# same field as the role glyphs) and interpreted by :func:`_build`. Keeping glyphs as data
# rather than one function each keeps the module small and mypy-clean. Icons are simplified
# line-art — at 15 px on a card or ~16 px on a node the outline reads, not the detail.
# ----------------------------------------------------------------------------------------

#: One drawing op: a command and its float arguments (``m``/``l`` = move/line-to, ``q``/``c``
#: = quad/cubic bézier, ``z`` = close, ``rr`` = rounded rect, ``el`` = ellipse, ``arc`` =
#: arc-move + arc-to on a bounding box).
_Op = tuple[str, tuple[float, ...]]


def _build(ops: tuple[_Op, ...]) -> QPainterPath:
    path = QPainterPath()
    for cmd, a in ops:
        if cmd == "m":
            path.moveTo(a[0], a[1])
        elif cmd == "l":
            path.lineTo(a[0], a[1])
        elif cmd == "q":
            path.quadTo(a[0], a[1], a[2], a[3])
        elif cmd == "c":
            path.cubicTo(a[0], a[1], a[2], a[3], a[4], a[5])
        elif cmd == "z":
            path.closeSubpath()
        elif cmd == "rr":
            path.addRoundedRect(QRectF(a[0], a[1], a[2], a[3]), a[4], a[5])
        elif cmd == "el":
            path.addEllipse(QRectF(a[0], a[1], a[2], a[3]))
        elif cmd == "arc":
            path.arcMoveTo(a[0], a[1], a[2], a[3], a[4])
            path.arcTo(a[0], a[1], a[2], a[3], a[4], a[5])
    return path


#: Block glyphs authored in the 24×24 box, keyed by a short glyph name. A pair of related
#: blocks may share one glyph (all variable ops → ``tag``, all dict ops → ``braces``); that
#: is meaning-preserving and still far more varied than one icon per role.
_GLYPHS: dict[str, tuple[_Op, ...]] = {
    "speech": (("rr", (3, 4, 18, 12, 3, 3)), ("m", (9, 16)), ("l", (9, 20)), ("l", (14, 16))),
    "play": (("m", (8, 6)), ("l", (18, 12)), ("l", (8, 18)), ("z", ())),
    "pause": (("rr", (7.5, 6, 3, 12, 1.5, 1.5)), ("rr", (13.5, 6, 3, 12, 1.5, 1.5))),
    "stop": (("rr", (6.5, 6.5, 11, 11, 2, 2)),),
    "volume": (
        ("m", (4, 9)),
        ("l", (4, 15)),
        ("l", (8, 15)),
        ("l", (13, 19)),
        ("l", (13, 5)),
        ("l", (8, 9)),
        ("z", ()),
        ("m", (16, 9)),
        ("q", (19, 12, 16, 15)),
    ),
    "mic": (
        ("rr", (9, 3, 6, 10, 3, 3)),
        ("m", (6, 11)),
        ("c", (6, 17.5, 18, 17.5, 18, 11)),
        ("m", (12, 17.5)),
        ("l", (12, 21)),
    ),
    "keyboard": (
        ("rr", (2.5, 7, 19, 10, 2, 2)),
        ("m", (6, 11)),
        ("l", (6.4, 11)),
        ("m", (9, 11)),
        ("l", (9.4, 11)),
        ("m", (12, 11)),
        ("l", (12.4, 11)),
        ("m", (15, 11)),
        ("l", (15.4, 11)),
        ("m", (8, 14)),
        ("l", (16, 14)),
    ),
    "key-down": (
        ("rr", (5, 5, 14, 14, 3, 3)),
        ("m", (12, 8)),
        ("l", (12, 16)),
        ("m", (9, 13)),
        ("l", (12, 16)),
        ("l", (15, 13)),
    ),
    "key-up": (
        ("rr", (5, 5, 14, 14, 3, 3)),
        ("m", (12, 16)),
        ("l", (12, 8)),
        ("m", (9, 11)),
        ("l", (12, 8)),
        ("l", (15, 11)),
    ),
    "type": (
        ("m", (9, 5)),
        ("l", (15, 5)),
        ("m", (9, 19)),
        ("l", (15, 19)),
        ("m", (12, 5)),
        ("l", (12, 19)),
    ),
    "mouse": (("rr", (7, 3, 10, 18, 5, 5)), ("m", (12, 3)), ("l", (12, 9))),
    "move": (
        ("m", (12, 3)),
        ("l", (12, 21)),
        ("m", (3, 12)),
        ("l", (21, 12)),
        ("m", (9, 6)),
        ("l", (12, 3)),
        ("l", (15, 6)),
        ("m", (9, 18)),
        ("l", (12, 21)),
        ("l", (15, 18)),
        ("m", (6, 9)),
        ("l", (3, 12)),
        ("l", (6, 15)),
        ("m", (18, 9)),
        ("l", (21, 12)),
        ("l", (18, 15)),
    ),
    "drag": (
        ("rr", (6, 3, 9, 15, 4.5, 4.5)),
        ("m", (10.5, 3)),
        ("l", (10.5, 8)),
        ("m", (14, 14)),
        ("l", (20, 20)),
        ("m", (20, 16)),
        ("l", (20, 20)),
        ("l", (16, 20)),
    ),
    "scroll": (
        ("rr", (7, 3, 10, 18, 5, 5)),
        ("m", (10.5, 7)),
        ("l", (12, 5.5)),
        ("l", (13.5, 7)),
        ("m", (10.5, 12)),
        ("l", (12, 13.5)),
        ("l", (13.5, 12)),
    ),
    "app-window": (
        ("rr", (3, 5, 18, 14, 2, 2)),
        ("m", (3, 9)),
        ("l", (21, 9)),
        ("m", (6, 7)),
        ("l", (6.3, 7)),
    ),
    "terminal": (
        ("rr", (3, 5, 18, 14, 2, 2)),
        ("m", (7, 10)),
        ("l", (10, 12.5)),
        ("l", (7, 15)),
        ("m", (12, 15)),
        ("l", (16, 15)),
    ),
    "x-square": (
        ("rr", (3, 5, 18, 14, 2, 2)),
        ("m", (9, 10)),
        ("l", (15, 16)),
        ("m", (15, 10)),
        ("l", (9, 16)),
    ),
    "target": (
        ("el", (5, 5, 14, 14)),
        ("el", (9, 9, 6, 6)),
        ("m", (12, 2)),
        ("l", (12, 5)),
        ("m", (12, 19)),
        ("l", (12, 22)),
        ("m", (2, 12)),
        ("l", (5, 12)),
        ("m", (19, 12)),
        ("l", (22, 12)),
    ),
    "layout": (
        ("rr", (3, 4, 18, 16, 2, 2)),
        ("m", (3, 9)),
        ("l", (21, 9)),
        ("m", (10, 9)),
        ("l", (10, 20)),
    ),
    "sun": (
        ("el", (8, 8, 8, 8)),
        ("m", (12, 2)),
        ("l", (12, 4.5)),
        ("m", (12, 19.5)),
        ("l", (12, 22)),
        ("m", (2, 12)),
        ("l", (4.5, 12)),
        ("m", (19.5, 12)),
        ("l", (22, 12)),
        ("m", (5, 5)),
        ("l", (6.8, 6.8)),
        ("m", (17.2, 17.2)),
        ("l", (19, 19)),
        ("m", (19, 5)),
        ("l", (17.2, 6.8)),
        ("m", (6.8, 17.2)),
        ("l", (5, 19)),
    ),
    "wifi": (
        ("m", (4, 10)),
        ("q", (12, 3, 20, 10)),
        ("m", (7, 13)),
        ("q", (12, 8.5, 17, 13)),
        ("m", (9.5, 16)),
        ("q", (12, 13.8, 14.5, 16)),
        ("m", (12, 19)),
        ("l", (12.2, 19)),
    ),
    "bluetooth": (
        ("m", (7, 8)),
        ("l", (17, 16)),
        ("l", (12, 20)),
        ("l", (12, 4)),
        ("l", (17, 8)),
        ("l", (7, 16)),
    ),
    "power": (("m", (12, 3)), ("l", (12, 11)), ("arc", (5, 5, 14, 14, 65, 290))),
    "camera": (
        ("rr", (3, 7, 18, 12, 2, 2)),
        ("el", (9, 10, 6, 6)),
        ("m", (8, 7)),
        ("l", (9.5, 4.5)),
        ("l", (14.5, 4.5)),
        ("l", (16, 7)),
    ),
    "clipboard-in": (
        ("rr", (5, 5, 14, 16, 2, 2)),
        ("rr", (9, 3, 6, 4, 1, 1)),
        ("m", (12, 10)),
        ("l", (12, 16)),
        ("m", (9, 13)),
        ("l", (12, 16)),
        ("l", (15, 13)),
    ),
    "clipboard-out": (
        ("rr", (5, 5, 14, 16, 2, 2)),
        ("rr", (9, 3, 6, 4, 1, 1)),
        ("m", (12, 16)),
        ("l", (12, 10)),
        ("m", (9, 13)),
        ("l", (12, 10)),
        ("l", (15, 13)),
    ),
    "diamond": (("m", (12, 3)), ("l", (21, 12)), ("l", (12, 21)), ("l", (3, 12)), ("z", ())),
    "switch": (("rr", (3, 8, 18, 8, 4, 4)), ("el", (12.5, 8.5, 7, 7))),
    "repeat": (
        ("m", (5, 11)),
        ("q", (12, 3, 19, 11)),
        ("m", (19, 7)),
        ("l", (19, 11)),
        ("l", (15, 11)),
        ("m", (19, 13)),
        ("q", (12, 21, 5, 13)),
        ("m", (5, 17)),
        ("l", (5, 13)),
        ("l", (9, 13)),
    ),
    "shield": (
        ("m", (12, 3)),
        ("l", (19, 6)),
        ("l", (19, 12)),
        ("c", (19, 17, 15, 20, 12, 21)),
        ("c", (9, 20, 5, 17, 5, 12)),
        ("l", (5, 6)),
        ("z", ()),
    ),
    "braces": (
        ("m", (9, 4)),
        ("c", (6, 4, 7.5, 10, 5.5, 12)),
        ("c", (7.5, 14, 6, 20, 9, 20)),
        ("m", (15, 4)),
        ("c", (18, 4, 16.5, 10, 18.5, 12)),
        ("c", (16.5, 14, 18, 20, 15, 20)),
    ),
    "brackets": (
        ("m", (9, 4)),
        ("l", (6, 4)),
        ("l", (6, 20)),
        ("l", (9, 20)),
        ("m", (15, 4)),
        ("l", (18, 4)),
        ("l", (18, 20)),
        ("l", (15, 20)),
    ),
    "tag": (
        ("m", (11, 4)),
        ("l", (20, 13)),
        ("l", (13, 20)),
        ("l", (4, 11)),
        ("l", (4, 4)),
        ("l", (11, 4)),
        ("z", ()),
        ("el", (6.5, 6.5, 3, 3)),
    ),
    "clock": (("el", (3, 3, 18, 18)), ("m", (12, 7)), ("l", (12, 12)), ("l", (15.5, 14))),
    "sleep": (
        ("m", (18, 14.5)),
        ("c", (16.5, 15, 15, 15, 13.5, 14.5)),
        ("c", (9, 13, 8, 7.5, 10.5, 4)),
        ("c", (5, 5.5, 3, 11.5, 6, 16)),
        ("c", (9, 20, 15, 20, 18, 14.5)),
        ("z", ()),
    ),
    "call": (
        ("m", (6, 4)),
        ("l", (6, 13)),
        ("l", (18, 13)),
        ("m", (14, 9)),
        ("l", (18, 13)),
        ("l", (14, 17)),
    ),
    "return": (
        ("m", (18, 4)),
        ("l", (18, 13)),
        ("l", (6, 13)),
        ("m", (10, 9)),
        ("l", (6, 13)),
        ("l", (10, 17)),
    ),
    "break": (("m", (6, 6)), ("l", (18, 18)), ("m", (18, 6)), ("l", (6, 18))),
    "skip-forward": (
        ("m", (5, 6)),
        ("l", (12, 12)),
        ("l", (5, 18)),
        ("z", ()),
        ("m", (12, 6)),
        ("l", (18, 12)),
        ("l", (12, 18)),
        ("z", ()),
        ("m", (19, 6)),
        ("l", (19, 18)),
    ),
    "skip-back": (
        ("m", (19, 6)),
        ("l", (12, 12)),
        ("l", (19, 18)),
        ("z", ()),
        ("m", (12, 6)),
        ("l", (6, 12)),
        ("l", (12, 18)),
        ("z", ()),
        ("m", (5, 6)),
        ("l", (5, 18)),
    ),
    "globe": (
        ("el", (3, 3, 18, 18)),
        ("m", (3, 12)),
        ("l", (21, 12)),
        ("el", (8, 3, 8, 18)),
        ("m", (12, 3)),
        ("l", (12, 21)),
    ),
    "external-link": (
        ("m", (11, 5)),
        ("l", (5, 5)),
        ("l", (5, 19)),
        ("l", (19, 19)),
        ("l", (19, 13)),
        ("m", (20, 4)),
        ("l", (12, 12)),
        ("m", (14, 4)),
        ("l", (20, 4)),
        ("l", (20, 10)),
    ),
    "search": (("el", (5, 5, 11, 11)), ("m", (14, 14)), ("l", (20, 20))),
    "bell": (
        ("m", (8, 16)),
        ("c", (8, 9, 8, 6, 12, 6)),
        ("c", (16, 6, 16, 9, 16, 16)),
        ("z", ()),
        ("m", (6, 16.5)),
        ("l", (18, 16.5)),
        ("m", (10.5, 17)),
        ("q", (12, 20, 13.5, 17)),
        ("m", (12, 4)),
        ("l", (12, 6)),
    ),
    "message": (
        ("rr", (3, 4, 18, 13, 3, 3)),
        ("m", (8, 17)),
        ("l", (8, 21)),
        ("l", (13, 17)),
        ("m", (7, 9)),
        ("l", (17, 9)),
        ("m", (7, 12)),
        ("l", (14, 12)),
    ),
    "heart": (
        ("m", (12, 20)),
        ("c", (4, 14, 4, 7, 8.5, 7)),
        ("c", (11, 7, 12, 9, 12, 9)),
        ("c", (12, 9, 13, 7, 15.5, 7)),
        ("c", (20, 7, 20, 14, 12, 20)),
        ("z", ()),
    ),
    "list-music": (
        ("m", (4, 7)),
        ("l", (14, 7)),
        ("m", (4, 12)),
        ("l", (12, 12)),
        ("m", (4, 17)),
        ("l", (10, 17)),
        ("m", (18, 6)),
        ("l", (18, 16)),
        ("el", (15, 15.5, 3.3, 3.3)),
        ("m", (18, 6)),
        ("l", (21, 7)),
    ),
}


#: Block type → glyph name. A type with no entry (or an unknown glyph name) falls back to the
#: role glyph in :func:`paint_block_glyph`, so an unmapped block is never blank.
_BLOCK_GLYPHS: dict[str, str] = {
    "Say": "speech",
    "PlaySound": "play",
    "StopSound": "stop",
    "SetVolume": "volume",
    "SetTTSVoice": "mic",
    "KeyPress": "keyboard",
    "KeyDown": "key-down",
    "KeyUp": "key-up",
    "TypeText": "type",
    "MouseClick": "mouse",
    "MouseMove": "move",
    "MouseDrag": "drag",
    "MouseWheel": "scroll",
    "RunApp": "app-window",
    "RunShell": "terminal",
    "CloseApp": "x-square",
    "FocusWindow": "target",
    "WindowState": "layout",
    "SetBrightness": "sun",
    "SetWifi": "wifi",
    "SetBluetooth": "bluetooth",
    "PowerAction": "power",
    "Screenshot": "camera",
    "ClipboardSet": "clipboard-in",
    "ClipboardGet": "clipboard-out",
    "If": "diamond",
    "Switch": "switch",
    "While": "repeat",
    "For": "repeat",
    "Try": "shield",
    "SetVar": "tag",
    "GetVar": "tag",
    "ArrayPush": "brackets",
    "ArrayGet": "brackets",
    "DictSet": "braces",
    "DictGet": "braces",
    "Wait": "clock",
    "Sleep": "sleep",
    "CallCommand": "call",
    "Return": "return",
    "Break": "break",
    "Continue": "skip-forward",
    "WebRequest": "globe",
    "OpenURL": "external-link",
    "SearchWeb": "search",
    "ToastNotify": "bell",
    "OverlayLog": "message",
    "YMPlay": "play",
    "YMPause": "pause",
    "YMNext": "skip-forward",
    "YMPrev": "skip-back",
    "YMLike": "heart",
    "YMSearch": "search",
    "YMPlaylist": "list-music",
}
