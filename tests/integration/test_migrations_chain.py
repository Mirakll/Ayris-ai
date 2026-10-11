"""Сквозная проверка цепочки миграций на РЕАЛЬНОМ файле БД.

Отдельные миграции и свежая схема покрыты в ``tests/unit/test_database.py``,
но там база всегда рождается сразу целиком (``migrate=True`` с нуля) — а это не
тот путь, которым проходит база у пользователя. У него файл был создан старой
версией Ayris, лежит на диске со старыми строками, и новая версия обязана
доехать по ``MIGRATIONS`` от ранней схемы до текущей, ничего не потеряв и
проставив дефолты новых столбцов на уже существующих строках.

Поэтому здесь мы: открываем файл, откатываем схему к v1, наполняем его данными
ТОЛЬКО из словаря v1 (сырым SQL — репозитории писать нельзя, их INSERT'ы знают о
поздних столбцах), закрываем, а затем открываем заново обычным путём и проверяем,
что схема доехала до :data:`SCHEMA_VERSION`, новые таблицы/столбцы/индексы
появились, дефолты проставлены на старых строках, а данные v1 выжили.

Метки времени — строковые литералы ISO-8601: адаптеры ``datetime`` в sqlite3
объявлены устаревшими, а ``filterwarnings=error`` превратил бы это в падение.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.core.database import Database
from ayris.core.migrations import (
    SCHEMA_VERSION,
    apply_migrations,
    current_version,
    schema_version_row,
)

pytestmark = pytest.mark.integration

#: Литерал ISO-8601 UTC — вместо datetime, чтобы не будить адаптеры sqlite3.
TS = "2024-01-01T00:00:00+00:00"


def _table_exists(db: Database, name: str) -> bool:
    """Есть ли таблица ``name`` в схеме."""
    row = db.query_one("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,))
    return row is not None


def _index_exists(db: Database, name: str) -> bool:
    """Есть ли индекс ``name`` в схеме."""
    row = db.query_one("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?", (name,))
    return row is not None


def _columns(db: Database, table: str) -> set[str]:
    """Имена столбцов таблицы через ``PRAGMA table_info``."""
    return {str(row["name"]) for row in db.query_all(f"PRAGMA table_info({table})")}


def _seed_v1(db: Database) -> None:
    """Наполнить базу строками ровно по схеме v1 (сырой SQL, без репозиториев)."""
    with db.transaction():
        db.execute(
            "INSERT INTO profiles (name, created_at, is_active) VALUES (?, ?, 1)",
            ("Старый профиль", TS),
        )
        profile_id = int(
            db.query_value("SELECT id FROM profiles WHERE name = ?", ("Старый профиль",))
        )
        db.execute(
            """
            INSERT INTO commands (profile_id, name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (profile_id, "старая команда", TS, TS),
        )
        command_id = int(
            db.query_value("SELECT id FROM commands WHERE name = ?", ("старая команда",))
        )
        db.execute(
            """
            INSERT INTO command_versions (command_id, version, snapshot_json, created_at)
            VALUES (?, 1, ?, ?)
            """,
            (command_id, '{"name": "старая команда"}', TS),
        )
        db.execute(
            """
            INSERT INTO models (kind, name, version, path, installed_at, is_active)
            VALUES ('stt', 'vosk-small', '0.22', 'models/vosk', ?, 1)
            """,
            (TS,),
        )
        db.execute(
            """
            INSERT INTO audit (ts, command_name, result, require_admin, elevated)
            VALUES (?, 'старая команда', 'ok', 0, 0)
            """,
            (TS,),
        )


def test_fresh_v1_lacks_later_schema(tmp_path: Path) -> None:
    """На v1 поздних таблиц/столбцов ещё нет — иначе проверка ниже ничего не докажет."""
    db_path = tmp_path / "ayris.db"
    with Database.open(db_path, migrate=False) as db:
        assert current_version(db) == 0
        assert apply_migrations(db, target=1) == 1
        assert current_version(db) == 1

        # Таблиц поздних миграций ещё нет.
        assert not _table_exists(db, "tts_usage")  # v2
        assert not _table_exists(db, "installed_apps")  # v4
        assert not _table_exists(db, "app_aliases")  # v4
        assert not _table_exists(db, "macro_debug_sessions")  # v6
        # Столбцов, добавленных ALTER'ами, тоже нет.
        assert "engine" not in _columns(db, "models")  # v3
        assert "size_bytes" not in _columns(db, "models")  # v3
        assert "catalog_id" not in _columns(db, "models")  # v3
        assert "confirmed" not in _columns(db, "audit")  # v5
        assert "important" not in _columns(db, "command_versions")  # v8


def test_chain_upgrades_v1_file_to_current(tmp_path: Path) -> None:
    """v1-файл со старыми строками доезжает до текущей схемы без потерь."""
    db_path = tmp_path / "ayris.db"

    # 1. Родить файл на v1 и наполнить данными той эпохи.
    with Database.open(db_path, migrate=False) as db:
        apply_migrations(db, target=1)
        _seed_v1(db)

    # 2. Открыть обычным путём — миграции доедут сами.
    with Database.open(db_path) as db:
        assert current_version(db) == SCHEMA_VERSION
        row = schema_version_row(db)
        assert row is not None
        assert row[0] == SCHEMA_VERSION

        # Новые таблицы появились.
        assert _table_exists(db, "tts_usage")  # v2
        assert _table_exists(db, "installed_apps")  # v4
        assert _table_exists(db, "app_aliases")  # v4
        assert _table_exists(db, "macro_debug_sessions")  # v6

        # Новые индексы появились.
        assert _index_exists(db, "idx_tts_usage_period")  # v2
        assert _index_exists(db, "idx_models_catalog")  # v3
        assert _index_exists(db, "idx_audit_command_ts")  # v7
        assert _index_exists(db, "idx_audit_result_ts")  # v7
        assert _index_exists(db, "idx_versions_important")  # v8

        # Данные v1 выжили.
        assert db.query_value("SELECT COUNT(*) FROM profiles") == 1
        assert db.query_value("SELECT name FROM profiles") == "Старый профиль"
        assert db.query_value("SELECT name FROM commands") == "старая команда"

        # Дефолты новых столбцов проставлены на СТАРЫХ строках.
        model = db.query_one("SELECT engine, size_bytes, catalog_id FROM models")
        assert model is not None
        assert model["engine"] == ""  # v3 default
        assert model["size_bytes"] == 0  # v3 default
        assert model["catalog_id"] == ""  # v3 default
        assert db.query_value("SELECT confirmed FROM audit") == 0  # v5 default
        assert db.query_value("SELECT important FROM command_versions") == 0  # v8 default
