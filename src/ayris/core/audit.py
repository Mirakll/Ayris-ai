"""Read-only audit queries shaped for the DevTools table."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ayris.core.models import AuditEntry, ExecutionResult
from ayris.core.repositories import AuditRepository

__all__ = ["AuditFilter", "AuditPage", "AuditReader"]


@dataclass(frozen=True, slots=True)
class AuditFilter:
    since: datetime | None = None
    until: datetime | None = None
    command: str = ""
    result: ExecutionResult | None = None
    require_admin: bool | None = None
    elevated: bool | None = None
    confirmed: bool | None = None


@dataclass(frozen=True, slots=True)
class AuditPage:
    items: tuple[AuditEntry, ...]
    page: int
    page_size: int
    total: int


class AuditReader:
    """Read-only facade; the action registry remains the only writer."""

    def __init__(self, repository: AuditRepository) -> None:
        self._repository = repository

    def page(
        self, filters: AuditFilter = AuditFilter(), *, page: int = 1, page_size: int = 100
    ) -> AuditPage:
        number = max(1, page)
        size = max(1, min(page_size, 1000))
        items = self._repository.query(
            since=filters.since,
            until=filters.until,
            command=filters.command,
            result=filters.result,
            require_admin=filters.require_admin,
            elevated=filters.elevated,
            confirmed=filters.confirmed,
            limit=size,
            offset=(number - 1) * size,
        )
        total = self._repository.count_filtered(
            since=filters.since,
            until=filters.until,
            command=filters.command,
            result=filters.result,
            require_admin=filters.require_admin,
            elevated=filters.elevated,
            confirmed=filters.confirmed,
        )
        return AuditPage(tuple(items), number, size, total)

    def top_commands(
        self, *, since: datetime | None = None, limit: int = 10
    ) -> tuple[tuple[str, int], ...]:
        return tuple(self._repository.top_commands(since=since, limit=limit))
