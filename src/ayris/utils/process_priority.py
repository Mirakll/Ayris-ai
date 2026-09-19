"""Apply the main assistant process's Windows scheduling priority, live.

Task 48's «Производительность» section lets the user raise or lower how the
operating system schedules Ayris itself. Unlike the audio worker — whose priority
is applied inside its own process at spawn and changed by restarting it — the main
process can have its priority class changed in place, with no restart, which is
what :func:`apply_priority` does.

The audio worker keeps its own copy of this logic in :mod:`ayris.workers.base`: a
worker process must never import the GUI-adjacent ``utils`` that pull in Qt, so
the small WinAPI call is deliberately written twice rather than shared across that
boundary. The mappings are kept identical on purpose.
"""

from __future__ import annotations

import ctypes
import os
import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from ayris.utils.logger import get_logger

__all__ = ["PRIORITY_CLASSES", "PRIORITY_LABELS", "apply_priority"]

_log = get_logger(__name__)

#: Windows priority classes, keyed by the values ``performance.process_priority``
#: and ``performance.audio_priority`` accept. Same numbers as the worker copy.
PRIORITY_CLASSES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "idle": 0x00000040,
        "below_normal": 0x00004000,
        "normal": 0x00000020,
        "above_normal": 0x00008000,
        "high": 0x00000080,
        "realtime": 0x00000100,
    }
)

#: Russian wording for the priority selectors in the «Общие» tab.
PRIORITY_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "idle": "Низкий",
        "below_normal": "Ниже обычного",
        "normal": "Обычный",
        "above_normal": "Выше обычного",
        "high": "Высокий",
        "realtime": "Реальное время",
    }
)

#: POSIX fallback, so the setting is exercised on the Linux CI runner too.
#: Negative values need privileges the runner lacks; failing to get them is fine.
_NICE_VALUES: Final[Mapping[str, int]] = MappingProxyType(
    {
        "idle": 10,
        "below_normal": 5,
        "normal": 0,
        "above_normal": -5,
        "high": -10,
        "realtime": -15,
    }
)


def _windows_dll(name: str) -> Any | None:
    """Open a Windows DLL with last-error tracking, or ``None`` off Windows."""
    factory = getattr(ctypes, "WinDLL", None)
    if factory is None:
        return None
    try:
        return factory(name, use_last_error=True)
    except OSError:
        return None


def apply_priority(priority: str) -> bool:
    """Set this (the main) process's scheduling priority class in place.

    ``realtime`` needs a privilege a normal account does not have; it is attempted
    and then quietly downgraded to «high» rather than refused, matching the audio
    worker so the two never disagree about what a value means.

    Returns:
        Whether the exact requested class was applied.
    """
    if sys.platform != "win32":
        wanted = _NICE_VALUES.get(priority, 0)
        if wanted == 0:
            return True
        try:
            os.nice(wanted)
        except (OSError, PermissionError):
            _log.debug("не удалось изменить nice до %d", wanted)
            return False
        return True

    kernel32 = _windows_dll("kernel32")
    wanted_class = PRIORITY_CLASSES.get(priority)
    if kernel32 is None or wanted_class is None:
        return False
    for candidate in (wanted_class, PRIORITY_CLASSES["high"]):
        try:
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            handle = kernel32.GetCurrentProcess()
            if kernel32.SetPriorityClass(ctypes.c_void_p(handle), candidate):
                return candidate == wanted_class
        except (AttributeError, OSError):
            break
    _log.warning("приоритет главного процесса «%s» не применён", priority)
    return False
