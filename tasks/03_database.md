# Задача 03 — База данных SQLite

**Этап:** A — Фундамент | **Зависит от:** 02 | **Разделы ТЗ:** 1.3, 7.1, 11, 18

## Цель
Слой доступа к SQLite: схема, миграции, репозитории для команд, переменных, истории, таймеров и аудита.

## Что сделать
1. `core/database.py`: обёртка над sqlite3 — connection pool или потокобезопасное соединение, WAL-режим, `foreign_keys=ON`, контекст-менеджер транзакций, безопасное закрытие.
2. Система миграций: таблица `schema_version`, список миграций в коде, применение по возрастанию при старте, идемпотентность. Только вперёд, ничего не удалять деструктивно.
3. Схема таблиц:
   - `profiles` — id, name, created_at, is_active
   - `command_folders` — дерево категорий (id, parent_id, name, order)
   - `commands` — id, profile_id, folder_id, name, description, tags, enabled, priority, cooldown_ms, require_admin, actions_json, created_at, updated_at
   - `command_versions` — история версий команды для undo/redo и экспорта
   - `triggers` — id, command_id, type (voice/hotkey/event/timer), payload_json, fuzzy, priority
   - `variables` — id, scope (local/profile/global), profile_id, name, type, value_json, persistent
   - `history` — id, ts, stt_raw, matched_command_id, intent, params_json, result, error, duration_ms
   - `audit` — id, ts, command_name, params_json, result, require_admin, elevated
   - `timers` — id, kind (timer/reminder/alarm), label, fire_at, cron, enabled, sound, payload_json
   - `clipboard_history` — id, ts, content, pinned
   - `models` — id, kind, name, version, path, sha256, installed_at, is_active
4. Индексы под частые запросы: триггеры по profile_id, история по ts, таймеры по fire_at.
5. Репозитории (по файлу или один модуль `core/repositories.py`) с типизированными методами: CRUD команд, папок, триггеров, переменных; запись/чтение истории и аудита; операции с таймерами.
6. Хелперы обслуживания: очистка истории старше N дней, VACUUM, полная очистка по категориям (для вкладки Приватность), бэкап БД в файл.
7. Тесты на временной БД: миграции с нуля, миграция поверх старой версии, CRUD, каскадное удаление, очистка.

## Файлы
`src/ayris/core/database.py`, `src/ayris/core/migrations.py`, `src/ayris/core/models.py` (dataclass/pydantic сущности), `src/ayris/core/repositories.py`, `tests/unit/test_database.py`

## Готово когда
- [ ] Чистый запуск создаёт БД со всеми таблицами и `schema_version`.
- [ ] Повторный запуск не выполняет миграции заново.
- [ ] CRUD команд с триггерами и каскадным удалением работает.
- [ ] Очистка истории и бэкап работают.
- [ ] Тесты зелёные.

## Промпт для нового чата
> Прочитай `AYRIS_SPEC.md`, `AYRIS_CONTEXT.md` и код проекта, затем выполни Задачу 03 из `tasks/03_database.md`: реализуй `core/database.py` (WAL, транзакции), систему миграций с версионированием схемы, все таблицы из задачи, типизированные сущности и репозитории, хелперы очистки и бэкапа. Напиши тесты на временной БД. Прогони ruff, mypy, pytest.
