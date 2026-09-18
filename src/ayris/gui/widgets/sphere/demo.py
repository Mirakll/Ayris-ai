"""Default sphere showcase: ``python -m ayris.gui.widgets.sphere.demo``.

The default sphere is the offline WebGL widget — this module re-exports the
``web_demo`` window so the canonical demo command shows the production visual.
For the lightweight QPainter fallback use ``sphere.painter_demo``.
"""

from __future__ import annotations

from ayris.gui.widgets.sphere.web_demo import WebSphereDemo as SphereDemoWindow
from ayris.gui.widgets.sphere.web_demo import main

__all__ = ["SphereDemoWindow", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
