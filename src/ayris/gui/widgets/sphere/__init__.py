"""Animated assistant sphere used by the Ayris overlay.

``SphereWidget`` is the offline WebGL neural sphere (pixel-identical to the
``design/sphere_ref`` prototype).  ``PainterSphereWidget`` is a lightweight
QPainter fallback for GPU-/WebEngine-less environments.
"""

from ayris.gui.widgets.sphere.sphere_widget import SphereWidget as PainterSphereWidget
from ayris.gui.widgets.sphere.states import SphereState
from ayris.gui.widgets.sphere.web_widget import SphereWidget

__all__ = ["PainterSphereWidget", "SphereState", "SphereWidget"]
