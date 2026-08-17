"""Executable actions and the macro engine.

Every action is published through :mod:`ayris.actions.registry` with a parameter
schema. NLU and macros call actions only through the registry, never directly:
that is what lets a macro name a block by string, survive an export to another
machine, and be described in the editor without importing WinAPI.

Re-exported here are the pieces an action author touches — :class:`Action`,
:class:`ActionMeta`, :class:`ActionResult`, :func:`register` — so a new action
module needs one import line instead of four.
"""

from __future__ import annotations

from ayris.actions.base import (
    DEFAULT_TIMEOUT_MS,
    Action,
    ActionCategory,
    ActionMeta,
    ActionParams,
    ActionSchema,
    Choice,
    FieldKind,
    ParamField,
    build_schema,
    mask_params,
)
from ayris.actions.registry import (
    ActionRegistry,
    RegisteredAction,
    register,
    registered_actions,
)
from ayris.actions.result import ActionResult

__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "Action",
    "ActionCategory",
    "ActionMeta",
    "ActionParams",
    "ActionRegistry",
    "ActionResult",
    "ActionSchema",
    "Choice",
    "FieldKind",
    "ParamField",
    "RegisteredAction",
    "build_schema",
    "mask_params",
    "register",
    "registered_actions",
]
