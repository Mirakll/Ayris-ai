"""Theme-driven QPainter renderers, optionally backed by QOpenGLWidget."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QLineF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QOpenGLContext,
    QPainter,
    QPaintEvent,
    QPen,
    QRadialGradient,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import AnimationParams


@dataclass(frozen=True, slots=True)
class RenderScene:
    """Everything the painter needs for one frame of the wire sphere.

    ``segments`` is ``(S, 4)`` screen-space ``x1, y1, x2, y2``; ``depth`` and
    ``vertical`` are ``(S,)`` in ``0..1`` (far→near, bottom→top).  Segments are
    pre-sorted far→near so nearer lines paint over the dim back hemisphere.
    """

    segments: NDArray[np.float64]
    depth: NDArray[np.float64]
    vertical: NDArray[np.float64]
    params: AnimationParams
    wave_progresses: tuple[float, ...]
    glow: bool
    shimmer: float = 0.0
    #: Overrides the theme accent for the sphere gradient/glow when set
    #: (``overlay.sphere_accent``); ``None`` keeps the palette accent.
    accent: QColor | None = None


class SceneProvider(Protocol):
    def __call__(self) -> RenderScene: ...


def _mix(first: QColor, second: QColor, amount: float) -> QColor:
    bounded = max(0.0, min(1.0, amount))
    return QColor(
        round(first.red() + (second.red() - first.red()) * bounded),
        round(first.green() + (second.green() - first.green()) * bounded),
        round(first.blue() + (second.blue() - first.blue()) * bounded),
    )


def _lighten(colour: QColor, amount: float) -> QColor:
    return _mix(colour, QColor(255, 255, 255), amount)


def paint_scene(
    painter: QPainter, widget: QWidget, scene: RenderScene, theme: ThemeManager
) -> None:
    # QOpenGLWidget keeps its previous framebuffer between QPainter passes on some
    # drivers.  Replacing every pixel first is essential: without it the moving
    # grid accumulates into the pink rectangular trails seen on real Windows GPUs.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    painter.fillRect(widget.rect(), Qt.GlobalColor.transparent)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    colours = theme.theme.colors
    err = max(0.0, min(1.0, scene.params.error))
    # Accent drives the bottom of the gradient, the shimmer tint and the glow;
    # an override colour (overlay.sphere_accent) replaces the palette accent.
    accent = scene.accent if scene.accent is not None else QColor(colours.accent)
    # Gradient endpoints: violet at the bottom, cyan at the top (matches the
    # prototype).  Under an error the whole grid shifts to red→amber.
    bottom = _mix(accent, QColor(colours.error), err)
    top = _mix(QColor(colours.info), QColor(colours.warning), err)
    shimmer_col = _lighten(accent, 0.35)

    count = int(scene.segments.shape[0]) if scene.segments.size else 0
    if scene.glow and count:
        glow_radius = min(widget.width(), widget.height()) * 0.4
        glow = QRadialGradient(widget.rect().center(), glow_radius)
        inner = _mix(QColor(colours.info), accent, 0.5)
        if err > 0.0:
            inner = _mix(inner, QColor(colours.error), err)
        inner.setAlpha(34)
        outer = QColor(inner)
        outer.setAlpha(0)
        glow.setColorAt(0.0, inner)
        glow.setColorAt(1.0, outer)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(glow)
        painter.drawEllipse(QRectF(widget.rect()))

    painter.setBrush(Qt.BrushStyle.NoBrush)
    shimmer = max(0.0, min(1.0, scene.shimmer))
    # Colour depends on latitude, alpha/width on depth.  Drawing 5000 lines with
    # a fresh pen each is too slow for the overlay, so we quantise into a small
    # grid of (latitude x depth) bands and issue one drawLines() call per band —
    # a couple dozen pen switches instead of thousands.
    bands_v, bands_d = 5, 6
    vertical_band = np.clip((scene.vertical * bands_v).astype(np.int64), 0, bands_v - 1)
    depth_band = np.clip((scene.depth * bands_d).astype(np.int64), 0, bands_d - 1)
    pen = QPen()
    for di in range(bands_d):  # far bands first so nearer lines paint on top
        near = (di + 0.5) / bands_d
        for vi in range(bands_v):
            mask = (depth_band == di) & (vertical_band == vi)
            rows = scene.segments[mask]
            if not rows.shape[0]:
                continue
            vertical = (vi + 0.5) / bands_v
            line = _mix(bottom, top, vertical)
            if shimmer > 0.0:
                line = _mix(line, shimmer_col, shimmer * (0.4 + 0.6 * (1.0 - vertical)))
            line.setAlphaF(max(0.05, min(0.85, 0.14 + 0.7 * near * near)))
            pen.setColor(line)
            pen.setWidthF(0.7 + 0.9 * near)
            painter.setPen(pen)
            painter.drawLines([QLineF(x1, y1, x2, y2) for x1, y1, x2, y2 in rows])

    _paint_waves(painter, widget, scene, theme)


def _paint_waves(
    painter: QPainter, widget: QWidget, scene: RenderScene, theme: ThemeManager
) -> None:
    if not scene.wave_progresses or scene.params.waves <= 0.0:
        return
    base = QColor(theme.theme.colors.on_accent)
    travel = min(widget.width(), widget.height()) / 2.0
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for progress in scene.wave_progresses:
        inset = progress * travel
        wave = QColor(base)
        wave.setAlphaF(max(0.0, (1.0 - progress / (2.0 / 3.0)) * 0.42 * scene.params.waves))
        painter.setPen(wave)
        painter.drawRoundedRect(
            QRectF(inset, inset, widget.width() - 2.0 * inset, widget.height() - 2.0 * inset),
            theme.metric("radius_xl"),
            theme.metric("radius_xl"),
        )


class PainterSphereRenderer(QWidget):
    backend_name = "QPainter"
    frame_rendered = Signal(float)

    def __init__(
        self, provider: SceneProvider, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._theme = theme
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        started = time.perf_counter()
        painter = QPainter(self)
        paint_scene(painter, self, self._provider(), self._theme)
        painter.end()
        self.frame_rendered.emit((time.perf_counter() - started) * 1000.0)


class OpenGLSphereRenderer(QOpenGLWidget):
    backend_name = "QOpenGLWidget"
    context_failed = Signal()
    frame_rendered = Signal(float)

    def __init__(
        self, provider: SceneProvider, theme: ThemeManager, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._provider = provider
        self._theme = theme
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.NoPartialUpdate)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def initializeGL(self) -> None:  # noqa: N802
        context = self.context()
        if context is None or not context.isValid():
            self.context_failed.emit()

    def paintGL(self) -> None:  # noqa: N802
        started = time.perf_counter()
        painter = QPainter(self)
        paint_scene(painter, self, self._provider(), self._theme)
        painter.end()
        self.frame_rendered.emit((time.perf_counter() - started) * 1000.0)


def opengl_available() -> bool:
    application = QApplication.instance()
    if (
        not isinstance(application, QApplication)
        or application.platformName().lower() in {"offscreen", "minimal"}
        or os.environ.get("AYRIS_DISABLE_OPENGL") == "1"
    ):
        return False
    context = QOpenGLContext()
    return context.create() and context.isValid()


def create_renderer(
    provider: SceneProvider,
    theme: ThemeManager,
    parent: QWidget,
    *,
    prefer_opengl: bool = False,
) -> PainterSphereRenderer | OpenGLSphereRenderer:
    if prefer_opengl and opengl_available():
        return OpenGLSphereRenderer(provider, theme, parent)
    return PainterSphereRenderer(provider, theme, parent)
