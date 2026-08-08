"""Executable actions and the macro engine.

Every action is published through :mod:`ayris.actions.registry` with a parameter
schema. NLU and macros call actions only through the registry, never directly.
"""

from __future__ import annotations
