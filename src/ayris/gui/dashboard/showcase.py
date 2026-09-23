"""The left «showcase» column: brand, state sphere and a quiet caption.

The panel is the darkest surface in the window. The sphere sits centred in a
square that scales with the panel; it animates itself according to its state.
Empty areas drag the frameless window.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTime, QTimer, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ayris.gui.dashboard.sphere_host import SphereLike, make_sphere
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.sphere.states import SphereState

__all__ = ["ShowcasePanel"]


class _SphereHost(QWidget):
    """Lets the sphere canvas fill the whole area; the sphere centres itself.

    The canvas must cover the entire host (not a centred square): the WebGL
    frame is opaque — filled with the panel colour — and bloom lifts the pixels
    around the sphere. A smaller square would clip that glow at its edge and show
    a lighter box behind the sphere. Full-bleed lets the glow fade to zero inside
    the frame, so its border matches the panel and no box is visible. The sphere's
    camera auto-fits any aspect ratio, so a non-square canvas never clips it.
    """

    def __init__(self, child: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._child = child
        child.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def resizeEvent(self, event: object) -> None:  # noqa: N802, ARG002
        self._child.setGeometry(0, 0, self.width(), self.height())


class ShowcasePanel(QFrame):
    """Logo, centred sphere and caption; empty space moves the window."""

    drag_started = Signal()
    fullscreen_toggle_requested = Signal()

    #: Glyphs for the corner button: outward arrows enter, inward arrows exit.
    _ENTER_GLYPH = "⤢"
    _EXIT_GLYPH = "⤡"

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._flat = False
        self.setObjectName("showcasePanel")

        self._layout = QVBoxLayout(self)

        # Logo and clock live in one tight header so the time sits right under
        # the brand, unaffected by the panel's larger inter-widget spacing.
        self._header = QWidget(self)
        header_layout = QVBoxLayout(self._header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.logo = QLabel("AYRIS", self._header)
        self.logo.setObjectName("showcaseLogo")
        self.logo.setAccessibleName("Айрис")

        self.clock = QLabel(self._header)
        self.clock.setObjectName("showcaseClock")
        self.clock.setAccessibleName("Текущее время")

        self.logo.setAlignment(Qt.AlignmentFlag.AlignLeft)
        # Header shrinks to the widest child (the logo), so centring the clock
        # inside it lands the time under the middle of the AYRIS text.
        self.clock.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(self.logo)
        header_layout.addWidget(self.clock)

        # Tick every second, aligned so the display tracks wall-clock time.
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._refresh_clock)
        self._clock_timer.start()
        self._refresh_clock()

        sphere_widget = make_sphere(theme, self)
        self._sphere: SphereLike = sphere_widget  # type: ignore[assignment]
        self._sphere_host = _SphereHost(sphere_widget, self)

        self.caption = QLabel("A Y R I S   A S S I S T A N T", self)
        self.caption.setObjectName("showcaseCaption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Bottom-left corner control: a quiet ghost button that toggles the
        # window's full-screen mode. A matching spacer on the right keeps the
        # caption centred despite the button eating space on the left.
        self.fullscreen_button = QPushButton(self._ENTER_GLYPH, self)
        self.fullscreen_button.setObjectName("showcaseGhost")
        self.fullscreen_button.setAccessibleName("Полноэкранный режим")
        self.fullscreen_button.setToolTip("Во весь экран (F11)")
        self.fullscreen_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.fullscreen_button.clicked.connect(self.fullscreen_toggle_requested.emit)

        self._footer_spacer = QWidget(self)
        self._footer_spacer.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        self._footer = QHBoxLayout()
        self._footer.setContentsMargins(0, 0, 0, 0)
        self._footer.addWidget(self.fullscreen_button, 0, Qt.AlignmentFlag.AlignLeft)
        self._footer.addStretch(1)
        self._footer.addWidget(self.caption, 0, Qt.AlignmentFlag.AlignHCenter)
        self._footer.addStretch(1)
        self._footer.addWidget(self._footer_spacer, 0, Qt.AlignmentFlag.AlignRight)

        self._layout.addWidget(
            self._header, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._layout.addWidget(self._sphere_host, 1)
        self._layout.addLayout(self._footer)

        theme.theme_changed.connect(self._refresh_theme)
        self._refresh_theme()

    @property
    def sphere(self) -> SphereLike:
        return self._sphere

    def set_state(self, state: SphereState | str) -> None:
        self._sphere.set_state(state)

    def set_level(self, level: float) -> None:
        self._sphere.set_level(level)

    def set_sphere_visible(self, visible: bool) -> None:
        """Hide the sphere while a full-window layer (settings) is up.

        The WebGL sphere is a ``QWebEngineView`` whose native surface on Windows
        composites above sibling widgets regardless of ``raise_()``. A slide-down
        settings layer therefore cannot cover it, so we drop that surface by
        hiding the host and bring it back when the layer closes.
        """
        self._sphere_host.setVisible(visible)

    def set_fullscreen(self, active: bool) -> None:
        """Reflect the window's full-screen state on the corner button.

        Flat corners follow along: on a full screen the panel's rounded left
        edge would let the desktop show through the window's translucent
        background, so the radius is dropped while full-screen.
        """
        self.fullscreen_button.setText(self._EXIT_GLYPH if active else self._ENTER_GLYPH)
        self.fullscreen_button.setToolTip(
            "Выйти из полноэкранного режима (F11)" if active else "Во весь экран (F11)"
        )
        self.fullscreen_button.setAccessibleName(
            "Выйти из полноэкранного режима" if active else "Полноэкранный режим"
        )
        if active != self._flat:
            self._flat = active
            self._refresh_theme()

    # Empty areas drag the frameless window.
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_started.emit()
        super().mousePressEvent(event)

    def _refresh_clock(self) -> None:
        self.clock.setText(QTime.currentTime().toString("HH:mm"))

    def _refresh_theme(self, _theme: object | None = None) -> None:
        color = self._theme.theme.color
        metric = self._theme.metric
        typography = self._theme.theme.typography

        background = color("background")
        text = color("text_primary")
        secondary = color("text_secondary")
        muted = color("text_muted")
        surface_high = color("surface_highlight")
        pad = metric("spacing_xl")
        radius = 0 if self._flat else metric("radius_lg")

        # QWebEngineView won't composite transparently on Windows, so blend the
        # sphere in by painting its page the same colour as this panel.
        set_background = getattr(self._sphere, "set_background", None)
        if callable(set_background):
            set_background(background)

        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(metric("spacing_lg"))
        # Round only the outer (left) corners so the panel follows the window's
        # rounded card; the right edge butts against the seam. Qt does not clip
        # children to the root's border-radius, so each corner-touching panel
        # must round itself, or the square fill covers the rounded frame.
        self.setStyleSheet(
            f"#showcasePanel {{ background: {background};"
            f" border-top-left-radius: {radius}px;"
            f" border-bottom-left-radius: {radius}px; }}"
            f"#showcaseLogo {{ color: {text}; font-size: {typography.h1_size}px;"
            f" font-weight: {typography.weight_bold}; letter-spacing: 2px; }}"
            f"#showcaseClock {{ color: {secondary}; font-size: {typography.body_size}px;"
            f" font-weight: {typography.weight_medium}; letter-spacing: 1px; }}"
            f"#showcaseCaption {{ color: {muted}; font-size: {typography.caption_size}px;"
            f" letter-spacing: 4px; }}"
            f"#showcaseGhost {{ background: transparent; border: none; color: {muted};"
            f" font-size: {typography.h2_size}px; border-radius: {metric('radius_md')}px; }}"
            f"#showcaseGhost:hover {{ color: {text}; background: {surface_high}; }}"
        )

        control = metric("control_height")
        self.fullscreen_button.setFixedSize(QSize(control, control))
        self._footer_spacer.setFixedSize(QSize(control, control))
