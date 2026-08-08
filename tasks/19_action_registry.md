# Задача 19 — Реестр действий

**Этап:** D — Системные действия | **Зависит от:** 04 | **Разделы ТЗ:** 6, 7.2

## Цель
Единая точка регистрации и вызова всех системных действий: базовый класс с pydantic-схемой параметров, реестр с поиском и валидацией, типизированный результат. NLU, макросы и плагины вызывают действия только через реестр.

## Что сделать
1. `actions/base.py`: ABC `Action` — вложенный `Params(BaseModel)` с типизированными полями и описаниями (`Field(description=...)`), метаданные в `ActionMeta`: `name`, `category`, `title_ru`, `description_ru`, `require_admin`, `is_dangerous`, `supports_undo`, `timeout_ms`. Методы `run(params) -> ActionResult` (sync) и `arun(params)` (async) — базовый класс сводит оба к одному пути вызова. Опциональный `undo(token)` для действий с `supports_undo=True`.
2. `actions/result.py`: `ActionResult` — `ok`, `value` (типизированный payload конкретного действия), `message_ru` для озвучивания и лога, `duration_ms`, `undo_token`. `ActionError` в `core/errors.py` с подклассами `ActionNotFound`, `ActionParamsInvalid`, `ActionTimeout`, `ActionRequiresAdmin`, `ActionUnavailable` и полем `user_message_ru`.
3. `actions/registry.py` — `ActionRegistry`: декоратор `@register` плюс автодискавери подпакетов `actions/system/*` через `pkgutil.walk_packages`, методы `get(name)`, `find(category=None, query=None)`, `list_categories()`, защита от дублей имён. Действия плагинов регистрируются с префиксом имени плагина.
4. Единый вход `execute(name, params: Mapping) -> ActionResult`: поиск действия → валидация параметров в `Params` (ошибки pydantic → `ActionParamsInvalid` с человекочитаемым списком полей) → проверка `require_admin` → выполнение с таймаутом (async — `asyncio.wait_for`, sync — в пуле потоков) → сборка `ActionResult`. Любое исключение внутри действия оборачивается в `ActionError`, наружу голых исключений не летит.
5. События и аудит: публикация `ActionStarted(name, params)`, `ActionFinished(name, result, duration_ms)`, `ActionFailed(name, error)` в event bus, запись в таблицу `audit` через репозиторий. Чувствительные параметры маскируются по признаку в схеме (`Field(json_schema_extra={"secret": True})`).
6. Интроспекция для UI редактора макросов: `describe(name) -> ActionSchema` — JSON Schema параметров плюс UI-подсказки (тип поля, диапазон, enum-варианты, дефолт, обязательность, русская подпись); `describe_all()` для дерева блоков. UI рисует поля по схеме, без хардкода параметров под каждое действие.
7. Тесты: регистрация и автодискавери, конфликт имён, валидация корректных и битых параметров, таймаут, состав и порядок событий, маскирование секретов, интроспекция схемы, sync и async действия.

## Файлы
`src/ayris/actions/__init__.py`, `src/ayris/actions/base.py`, `src/ayris/actions/result.py`, `src/ayris/actions/registry.py`, `src/ayris/actions/system/__init__.py`, `src/ayris/core/errors.py` (обновить), `tests/unit/test_action_registry.py`

## Осторожно
- Реестр — единственный путь вызова: макросы и плагины не должны импортировать классы действий напрямую.
- Схема параметров — контракт с UI и с форматом экспорта `.ayris`: переименование поля требует миграции, а не правки на месте.
- Таймаут не должен оставлять висящий поток: длинные WinAPI-вызовы делать отменяемыми или уносить в воркер.

## Готово когда
- [ ] `registry.execute("SetVolume", {"level": 50})` выполняется, битые параметры дают `ActionParamsInvalid` с понятным сообщением.
- [ ] Новое действие подхватывается автодискавери без правок вызывающего кода.
- [ ] Каждый вызов даёт пару событий Started/Finished либо Started/Failed и запись в `audit`.
- [ ] `describe_all()` отдаёт схему, достаточную для отрисовки полей всех блоков в редакторе.
- [ ] Секретные параметры не попадают ни в лог, ни в аудит.
- [ ] Тесты зелёные.

## Промпт для нового чата
> Прочитай `AYRIS_SPEC.md`, `AYRIS_CONTEXT.md` и код проекта, затем выполни Задачу 19 из `tasks/19_action_registry.md`: реализуй базовый класс `Action` с pydantic-схемой параметров и метаданными, типизированный `ActionResult` и иерархию `ActionError`, реестр с декоратором регистрации и автодискавери, единый `execute(name, params)` с валидацией, проверкой прав и таймаутом, публикацию событий ActionStarted/Finished/Failed с записью в аудит и маскированием секретов, интроспекцию схемы для UI редактора макросов. Напиши тесты. Прогони ruff, mypy, pytest.
