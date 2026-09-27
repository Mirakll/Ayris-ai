"""Pick the best available state sphere, with a graceful fallback.

The offline WebGL sphere (:mod:`…sphere.web_widget`) is the intended widget, but
it needs Qt WebEngine. Where WebEngine is missing, the lightweight QPainter
sphere (:class:`…sphere.sphere_widget.SphereWidget`) is used instead. Both expose
the same ``set_state`` / ``set_level`` API, so callers do not care which they got.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from PySide6.QtWidgets import QWidget

from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import AnimationProfile, SphereState
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from ayris.core.config import OverlayConfig

__all__ = ["SphereLike", "apply_overlay_appearance", "make_sphere"]

_log = get_logger(__name__)


@runtime_checkable
class SphereLike(Protocol):
    """The slice of the sphere widget the dashboard drives."""

    @property
    def state(self) -> SphereState: ...

    def set_state(self, state: SphereState | str) -> None: ...

    def set_level(self, level: float) -> None: ...

    # Appearance/economy — honoured live from OverlayConfig (task 56).
    def set_point_count(self, count: int) -> None: ...

    def set_target_fps(self, target_fps: int) -> None: ...

    def set_animations_enabled(self, enabled: bool) -> None: ...

    def set_stop_when_hidden(self, stop: bool) -> None: ...

    def set_profile(self, profile: AnimationProfile) -> None: ...

    def set_accent(self, colour: str | None) -> None: ...

    # Avatar switch + icosahedron-only knobs («Панель / Сфера», режим icosa).
    def set_avatar(self, kind: str) -> None: ...

    def set_icosa_config(self, cfg: dict[str, object]) -> None: ...


def apply_overlay_appearance(sphere: SphereLike, overlay: OverlayConfig) -> None:
    """Push every live appearance/economy setting from ``overlay`` onto a sphere.

    Used by the dashboard's showcase sphere and by the settings preview so a
    single mapping keeps them identical. Order is deliberate: the avatar is
    switched first, then shape/economy, then the motion profile, so a paused
    figure still repaints once; icosahedron-only knobs go last and only when the
    icosahedron is the active avatar.
    """
    sphere.set_avatar(overlay.avatar)
    sphere.set_point_count(overlay.sphere_points)
    sphere.set_target_fps(overlay.target_fps)
    sphere.set_stop_when_hidden(overlay.stop_when_hidden)
    sphere.set_animations_enabled(overlay.animations)
    sphere.set_accent(overlay.sphere_accent or None)
    sphere.set_profile(
        AnimationProfile(
            rotation_speed=overlay.rotation_speed,
            pulse_amplitude=overlay.pulse_amplitude,
            wave_intensity=overlay.wave_intensity,
            rotation=overlay.rotation,
            pulsation=overlay.pulsation,
            waves=overlay.waves,
            error_flash=overlay.error_flash,
        )
    )
    if overlay.avatar == "icosa":
        sphere.set_icosa_config(
            {
                "opacity": overlay.icosa_opacity,
                "calmSpin": overlay.icosa_calm_spin,
                "flip": {
                    "listen": {"h": overlay.icosa_listen_rot_h, "v": overlay.icosa_listen_rot_v},
                    "think": {"h": overlay.icosa_think_rot_h, "v": overlay.icosa_think_rot_v},
                },
                "twist": {
                    "calm": overlay.icosa_calm_twist,
                    "listen": overlay.icosa_listen_twist,
                    "think": overlay.icosa_think_twist,
                    "speak": overlay.icosa_speak_twist,
                },
                "dockVariant": overlay.icosa_dock_variant,
            }
        )


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
