"""Theme-aware QPainter renderer for the AEGIS point cloud.

Particles are drawn as soft bokeh dots (a faint halo behind a bright core) over
a central glow, so the flat dot field reads as a self-luminous sphere.  The
palette supplies the cool/warm tints; the alarm red is taken from the theme so
an error looks like an error whatever skin the app wears.  An optional
:class:`QOpenGLWidget` backend is available but the QPainter path is the visual
reference, exactly as in the original sphere widget.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QOpenGLContext, QPainter, QPaintEvent, QRadialGradient
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.geosphere.geometry import ProjectedDot
from ayris.gui.widgets.geosphere.palette import SpherePalette
from ayris.gui.widgets.geosphere.states import AnimationParams


@dataclass(frozen=True, slots=True)
class RenderScene:
    dots: tuple[ProjectedDot, ...]
    params: AnimationParams
    palette: SpherePalette
    glow: bool
    core_glow: float = 0.0
    shake_x: float = 0.0
    shake_y: float = 0.0


class SceneProvider(Protocol):
    def __call__(self) -> RenderScene: ...


def _mix(first: QColor, second: QColor, amount: float) -> QColor:
    bounded = max(0.0, min(1.0, amount))
    return QColor(
        round(first.red() + (second.red() - first.red()) * bounded),
        round(first.green() + (second.green() - first.green()) * bounded),
        round(first.blue() + (second.blue() - first.blue()) * bounded),
    )


def paint_scene(
    painter: QPainter, widget: QWidget, scene: RenderScene, theme: ThemeManager
) -> None:
    # QOpenGLWidget can keep its previous framebuffer between passes on some
    # drivers; replacing every pixel first avoids the accumulated smear the
    # original sphere hit on real Windows GPUs.
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
    painter.fillRect(widget.rect(), Qt.GlobalColor.transparent)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
    painter.translate(scene.shake_x, scene.shake_y)
    painter.setPen(Qt.PenStyle.NoPen)

    base = scene.palette.base_color()
    warm = scene.palette.warm_color()
    # Deep crimson: pull the theme's error toward black so the alarm reads as a
    # dense red shell rather than a flat swatch.
    alarm = _mix(QColor(theme.theme.colors.error), QColor(0, 0, 0), 0.22)
    error = scene.params.error

    _paint_core(painter, widget, scene, theme, base, warm, alarm, error)

    min_side = float(min(widget.width(), widget.height()))
    for dot in scene.dots:
        brightness = dot.brightness
        if brightness <= 0.01:
            continue
        warmth = max(0.0, min(1.0, (brightness - 0.6) / 0.6))
        colour = _mix(base, warm, warmth)
        if error > _MIN_ERROR:
            colour = _mix(colour, alarm, error)
        core_alpha = max(0.0, min(0.92, brightness * 0.75))
        size = dot.size
        # Faint halo first, bright core on top — cheap, ordered bokeh.
        halo = QColor(colour)
        halo.setAlphaF(core_alpha * 0.26)
        painter.setBrush(halo)
        halo_size = size * (2.6 if scene.glow else 1.8)
        painter.drawEllipse(
            QRectF(dot.x - halo_size / 2.0, dot.y - halo_size / 2.0, halo_size, halo_size)
        )
        core = QColor(colour)
        core.setAlphaF(core_alpha)
        painter.setBrush(core)
        painter.drawEllipse(QRectF(dot.x - size / 2.0, dot.y - size / 2.0, size, size))

    if scene.params.flash > _MIN_ERROR:
        radius = min_side * 0.46
        flash = QRadialGradient(widget.rect().center(), radius)
        centre = QColor(alarm)
        centre.setAlphaF(min(0.24, scene.params.flash * 0.24))
        edge = QColor(centre)
        edge.setAlpha(0)
        flash.setColorAt(0.4, centre)
        flash.setColorAt(1.0, edge)
        painter.setBrush(flash)
        painter.drawEllipse(QRectF(widget.rect()))


_MIN_ERROR = 0.001


def _paint_core(
    painter: QPainter,
    widget: QWidget,
    scene: RenderScene,
    theme: ThemeManager,  # noqa: ARG001
    base: QColor,
    warm: QColor,
    alarm: QColor,  # noqa: ARG001
    error: float,
) -> None:
    if not scene.glow:
        return
    radius = min(widget.width(), widget.height()) * 0.4
    if radius <= 0.0:
        return
    gradient = QRadialGradient(widget.rect().center(), radius)
    inner = _mix(scene.palette.core_color(), warm, 0.2)
    # The core brightens with the speaking pulse and goes dark under an error,
    # leaving only the red rim the dots draw.
    strength = (0.16 + 0.5 * max(0.0, min(1.0, scene.core_glow))) * (1.0 - error)
    inner.setAlphaF(max(0.0, min(0.55, strength)))
    outer = QColor(base)
    outer.setAlpha(0)
    gradient.setColorAt(0.0, inner)
    gradient.setColorAt(1.0, outer)
    painter.setBrush(gradient)
    painter.drawEllipse(QRectF(widget.rect()))


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
