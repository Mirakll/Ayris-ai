"""Optional two-way sync of reminders with external calendars.

Off by default and isolated: a missing or misconfigured provider never touches
the local scheduler. Every provider implements :class:`CalendarSync`; its heavy
third-party dependency is imported lazily inside the provider, not at module
import, so walking this package for action discovery costs nothing and works on a
machine where none of the SDKs are installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from ayris.core.errors import AyrisError
from ayris.utils.logger import get_logger

__all__ = [
    "CalendarEvent",
    "CalendarSync",
    "CalendarSyncError",
    "SyncProvider",
    "available_providers",
    "make_sync",
]

_log = get_logger(__name__)


class CalendarSyncError(AyrisError):
    """A calendar provider could not be reached or configured."""


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """One remote reminder, normalised across providers."""

    uid: str
    title: str
    start: datetime
    all_day: bool = False


class SyncProvider:
    """The provider identifiers the user may enable in settings."""

    MS_TODO: Final = "ms_todo"
    GOOGLE: Final = "google"
    CALDAV: Final = "caldav"


class CalendarSync(ABC):
    """The common contract every calendar backend implements.

    Concrete providers import their SDK lazily in :meth:`connect`, so this module
    imports on any machine. A provider that cannot connect raises
    :class:`CalendarSyncError`; the scheduler treats that as «sync unavailable»
    and carries on with local timers.
    """

    provider: str = ""

    @abstractmethod
    def connect(self) -> None:
        """Acquire credentials and a client. Raises on failure."""

    @abstractmethod
    def pull(self, *, since: datetime) -> Sequence[CalendarEvent]:
        """Remote reminders changed since ``since``."""

    @abstractmethod
    def push(self, event: CalendarEvent) -> str:
        """Create or update a remote reminder; return its remote uid."""


@dataclass(frozen=True, slots=True)
class _MissingDependency(CalendarSync):
    """Placeholder returned for a provider whose SDK is not installed.

    It answers :meth:`connect` with a clear, localised error rather than an
    ``ImportError`` deep in a stack trace, and never silently pretends to sync.
    """

    provider: str = ""
    package: str = ""
    _events: tuple[CalendarEvent, ...] = field(default_factory=tuple)

    def connect(self) -> None:
        raise CalendarSyncError(
            f"Синхронизация «{self.provider}» недоступна: не установлен пакет {self.package}"
        )

    def pull(self, *, since: datetime) -> Sequence[CalendarEvent]:
        raise CalendarSyncError(
            f"Провайдер «{self.provider}» не подключён (запрос с {since:%Y-%m-%d})"
        )

    def push(self, event: CalendarEvent) -> str:
        raise CalendarSyncError(f"Провайдер «{self.provider}» не подключён (событие {event.uid})")


_PROVIDER_PACKAGE: Final[dict[str, str]] = {
    SyncProvider.MS_TODO: "msal",
    SyncProvider.GOOGLE: "google-api-python-client",
    SyncProvider.CALDAV: "caldav",
}


def available_providers() -> tuple[str, ...]:
    """Provider ids whose SDK can be imported right now."""
    import importlib.util

    ready: list[str] = []
    for provider, package in _PROVIDER_PACKAGE.items():
        module = package.split("-")[0].replace("google", "googleapiclient")
        if importlib.util.find_spec(module) is not None:
            ready.append(provider)
    return tuple(ready)


def make_sync(provider: str) -> CalendarSync:
    """A sync backend for ``provider``.

    Returns a placeholder that fails cleanly on :meth:`~CalendarSync.connect`
    when the provider's SDK is absent, so enabling sync on a machine without the
    dependency degrades to a clear message instead of a crash.
    """
    package = _PROVIDER_PACKAGE.get(provider)
    if package is None:
        raise CalendarSyncError(f"Неизвестный провайдер синхронизации: {provider!r}")
    _log.debug("запрошена синхронизация календаря: %s", provider)
    return _MissingDependency(provider=provider, package=package)
