"""Pick the best available state sphere, with a graceful fallback.

The offline WebGL sphere (:mod:`…sphere.web_widget`) is the intended widget, but
it needs Qt WebEngine. Where WebEngine is missing, the lightweight QPainter
sphere (:class:`…sphere.sphere_widget.SphereWidget`) is used instead. Both expose
the same ``set_state`` / ``set_level`` API, so callers do not care which they got.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from PySide6.QtWidgets import QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import SphereState
from ayris.utils.logger import get_logger

__all__ = ["SphereLike", "make_sphere"]

_log = get_logger(__name__)


@runtime_checkable
class SphereLike(Protocol):
    """The slice of the sphere widget the dashboard drives."""

    @property
    def state(self) -> SphereState: ...

    def set_state(self, state: SphereState | str) -> None: ...

    def set_level(self, level: float) -> None: ...


def make_sphere(theme: ThemeManager, parent: QWidget | None = None) -> QWidget:
    """Return the WebGL sphere if WebEngine is available, else the painter one.

    The returned widget is always a :class:`QWidget` and always satisfies
    :class:`SphereLike`. The accessible name is set here so every call site gets
    it for free.
    """
    widget: QWidget
    try:
        from ayris.gui.widgets.sphere.web_widget import SphereWidget as WebSphere

        widget = WebSphere(parent)
    except Exception as exc:  # pragma: no cover - only without WebEngine
        _log.info("WebGL-сфера недоступна (%s), беру QPainter-фолбэк", exc)
        from ayris.gui.widgets.sphere.sphere_widget import SphereWidget as PainterSphere

        widget = PainterSphere(theme, parent=parent)
    widget.setAccessibleName("Сфера состояния Айрис")
    return widget
