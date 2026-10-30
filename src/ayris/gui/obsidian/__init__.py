"""Obsidian 2c desktop interface integration."""

from __future__ import annotations

from ayris.gui.obsidian.server import ObsidianServer, ensure_obsidian_server
from ayris.gui.obsidian.view import ObsidianView

__all__ = ["ObsidianServer", "ObsidianView", "ensure_obsidian_server"]
