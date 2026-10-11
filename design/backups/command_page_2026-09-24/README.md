# Бэкап вкладки «Команды» — 2026-09-24

Снимок файлов вкладки «Команды» (список + нодовый холст) **до** редизайна под
вариант «Кинематограф». Сделан по просьбе пользователя перед портированием
макета в приложение.

## Что внутри (снято 1:1 из `src/ayris/gui/`)

- `commands.py` — `src/ayris/gui/tabs/commands.py`
- `macro_editor.py` — `src/ayris/gui/widgets/macro_editor.py`
- `block_palette.py` — `src/ayris/gui/widgets/block_palette.py`
- `param_form.py` — `src/ayris/gui/widgets/param_form.py`
- `qss.py` — `src/ayris/gui/theme/qss.py`
- `node_editor/*.py` — `src/ayris/gui/widgets/node_editor/`

## Как откатить

Скопировать нужный файл обратно на его место в `src/ayris/gui/…`, затем
прогнать `bash scripts/check.sh`.
