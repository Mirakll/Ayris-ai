"""System actions: applications, windows, audio, display, input, power, web.

Every WinAPI call is wrapped in error handling that logs the failure and raises
a typed :class:`ayris.core.errors.ActionError` with a user-facing message.
"""

from __future__ import annotations
