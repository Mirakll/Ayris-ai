"""Task 41: read-only, filtered and paged audit access for DevTools."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from ayris.core.audit import AuditFilter, AuditReader
from ayris.core.database import Database
from ayris.core.models import AuditEntry, ExecutionResult, utc_now
from ayris.core.repositories import AuditRepository

pytestmark = pytest.mark.unit


def test_filtered_pages_and_aggregates(tmp_path: Path) -> None:
    database = Database.open(tmp_path / "audit.db")
    repository = AuditRepository(database)
    now = utc_now()
    repository.add(AuditEntry(command_name="volume.set", ts=now, result=ExecutionResult.OK))
    repository.add(
        AuditEntry(
            command_name="system.shutdown",
            ts=now - timedelta(minutes=1),
            result=ExecutionResult.DENIED,
            require_admin=True,
            elevated=True,
            confirmed=True,
        )
    )
    repository.add(AuditEntry(command_name="volume.set", ts=now - timedelta(days=30)))
    reader = AuditReader(repository)

    page = reader.page(
        AuditFilter(
            since=now - timedelta(days=1),
            command="shutdown",
            result=ExecutionResult.DENIED,
            require_admin=True,
            elevated=True,
            confirmed=True,
        ),
        page_size=1,
    )
    assert page.total == 1
    assert page.items[0].command_name == "system.shutdown"
    assert reader.top_commands(limit=1) == (("volume.set", 2),)
    database.close()


def test_repository_redacts_pattern_secrets(tmp_path: Path) -> None:
    database = Database.open(tmp_path / "audit-redaction.db")
    repository = AuditRepository(database)
    api_key = "sk-" + "abcdefghijklmnopqrstuvwxyz"
    repository.add(AuditEntry(command_name="probe", params={"note": api_key}))

    entry = repository.recent(1)[0]
    assert api_key not in str(entry.params)
    assert entry.params["note"] == "[скрыто]"
    database.close()
