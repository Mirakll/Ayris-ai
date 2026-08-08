# Задача 01 — Скелет проекта и tooling

**Этап:** A — Фундамент | **Зависит от:** — | **Разделы ТЗ:** 21, 24

## Цель
Создать структуру репозитория, настроить зависимости и инструменты качества, поднять минимальное PySide6-окно, которое запускается командой `python -m ayris`.

## Что сделать
1. Полное дерево каталогов по разделу 24 ТЗ. В каждом пакете — `__init__.py`.
2. `pyproject.toml`: метаданные проекта, зависимости из раздела 21 (закрепи точные версии), опциональные группы `dev`, `cuda`, `ocr`, `games`, конфиги ruff / black / mypy (strict) / pytest.
3. `src/ayris/utils/logger.py`: настройка логирования, ротация по 10 МБ и 7 дней, уровни DEBUG/INFO/WARNING/ERROR, форматтер с модулем и временем, отдельный логгер для пайплайна.
4. `src/ayris/core/errors.py`: базовое `AyrisError` и подклассы под будущие домены (`ConfigError`, `AudioError`, `SttError`, `TtsError`, `ActionError`, `MacroError`, `PluginError`).
5. `src/ayris/__main__.py`: разбор аргументов (`--minimized`, `--profile`, `--log-level`, `--portable`), инициализация логгера, создание `QApplication`, per-monitor DPI awareness, запуск пустого окна и выход по закрытию.
6. `.pre-commit-config.yaml`, `.gitignore`, `README.md` (кратко: что это, как запустить, как собрать), `LICENSE` (MIT).
7. `tests/unit/test_smoke.py`: проверка импорта пакета и создания логгера.

## Файлы
`pyproject.toml`, `.pre-commit-config.yaml`, `.gitignore`, `README.md`, `LICENSE`, всё дерево `src/ayris/`, `tests/unit/test_smoke.py`

## Решения, которые нужно зафиксировать
- Python 3.11 как минимальная версия.
- Layout `src/`, установка через `pip install -e .`.
- Опциональные зависимости не тянутся по умолчанию (interception, paddleocr, coqui, playwright — только по группам).

## Готово когда
- [ ] `pip install -e ".[dev]"` проходит без ошибок.
- [ ] `python -m ayris` открывает пустое окно, `--minimized` не показывает окно.
- [ ] `ruff check .`, `black --check .`, `mypy src` — чисто.
- [ ] `pytest` — зелёный.
- [ ] Лог-файл появляется в `%APPDATA%\Ayris\logs\ayris_YYYYMMDD.log`.

## Промпт для нового чата
> Ты — старший Python-разработчик. Прочитай `AYRIS_SPEC.md` и `AYRIS_CONTEXT.md`, затем выполни Задачу 01 из `tasks/01_skeleton.md`: создай структуру проекта Ayris, `pyproject.toml` с закреплёнными версиями зависимостей, настройку ruff/black/mypy(strict)/pytest/pre-commit, модуль логирования с ротацией, иерархию исключений и точку входа `__main__.py`, поднимающую пустое PySide6-окно с per-monitor DPI awareness. Пиши рабочий типизированный код, без заглушек. В конце прогони линтеры и тесты, покажи результат.
