# Ayris

Голосовой помощник для Windows 11. Русскоязычный интерфейс, офлайн-first
распознавание и синтез речи с облачными фоллбеками, движок макросов в стиле
VoiceAttack, система плагинов и оверлей.

Проект в разработке. Полное описание — в `AYRIS_SPEC.md`, карта работ — в
`AYRIS_ROADMAP.md`.

## Требования

- Windows 10 1909 или новее (целевая платформа — Windows 11 22H2+)
- Python 3.11 или 3.12
- 4 ГБ RAM минимум, 16 ГБ для локальных языковых моделей

## Установка для разработки

```powershell
git clone https://github.com/ayris-app/ayris.git
cd ayris
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
move build\pre-commit-config.yaml .pre-commit-config.yaml
pre-commit install
```

> Файл конфигурации pre-commit лежит в `build\pre-commit-config.yaml` и его нужно
> один раз перенести в корень под именем `.pre-commit-config.yaml` — среда,
> в которой создавался скелет, не позволяла записывать файлы, начинающиеся с точки.

Тяжёлые зависимости не тянутся по умолчанию. Ставьте их группами по мере
необходимости:

| Группа | Что даёт |
| --- | --- |
| `cuda` | GPU-ускорение faster-whisper и llama.cpp |
| `ocr` | распознавание текста на скриншотах |
| `games` | глобальные хоткеи в полноэкранных играх (нужен драйвер Interception) |
| `web` | браузерная автоматизация Яндекс Музыки |
| `llm-local` | локальные языковые модели без Ollama |
| `tts-extra` | клонирование голоса Coqui XTTS v2 |
| `wake-extra` | слово активации через Porcupine |
| `build` | сборка дистрибутива Nuitka |

Пример: `pip install -e ".[dev,cuda]"`.

## Запуск

```powershell
python -m ayris
```

Аргументы:

| Аргумент | Назначение |
| --- | --- |
| `--minimized` | запуск без показа окна настроек |
| `--profile PATH` | папка профиля вместо `%APPDATA%\Ayris` |
| `--log-level LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `--portable` | хранить профиль рядом с исполняемым файлом |
| `--no-console-log` | не дублировать логи в консоль |

Данные пользователя лежат в `%APPDATA%\Ayris`: `config.toml`, `ayris.db`,
`logs/`, `models/`, `plugins/`. Логи пишутся в
`%APPDATA%\Ayris\logs\ayris_ГГГГММДД.log` с ротацией по 10 МБ и хранением
7 дней; трассировка пайплайна — в отдельный `pipeline_ГГГГММДД.log`.

## Проверки

```powershell
ruff check .
black --check .
mypy src
pytest
```

Те же проверки навешаны на pre-commit хук.

## Сборка

Дистрибутив собирается Nuitka, инсталлятор — Inno Setup. Скрипты появятся в
`build/` (задачи 71–72).

```powershell
pip install -e ".[build]"
python build\nuitka_build.py
```

## Лицензия

MIT — см. `LICENSE`.
