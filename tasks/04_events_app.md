# Задача 04 — Event bus и жизненный цикл приложения

**Этап:** A — Фундамент | **Зависит от:** 02, 03 | **Разделы ТЗ:** 1.2, 6, 14

## Цель
Типизированная шина событий и класс приложения, который собирает все подсистемы, управляет запуском и корректным завершением.

## Что сделать
1. `core/events.py`:
   - Dataclass-события: `WakeWordDetected`, `AudioLevelChanged`, `SpeechStarted`, `SpeechEnded`, `TranscriptReady`, `IntentMatched`, `ActionStarted`, `ActionFinished`, `ActionFailed`, `TtsStarted`, `TtsFinished`, `CancelRequested`, `ModeChanged`, `MicToggled`, `OnlineStatusChanged`, `WorkerCrashed`, `WorkerRestarted`, `ConfigChanged`, `TimerFired`, `NotificationRequested`, `LogLine`.
   - `EventBus`: `subscribe(event_type, handler)`, `unsubscribe`, `publish(event)`. Потокобезопасность (события приходят из воркеров и QThread), доставка в UI-поток через Qt-сигнал или очередь + таймер.
   - Слабые ссылки на подписчиков либо явная отписка, чтобы не течь.
2. `core/app.py` — класс `AyrisApp`:
   - Порядок инициализации: логгер → пути → конфиг → БД → миграции → профиль → event bus → реестр действий → менеджер воркеров → NLU → GUI/трей/оверлей → плагины.
   - Единая точка доступа к подсистемам (внутренний контейнер, без глобальных синглтонов по коду).
   - Реакция на `ConfigChanged`: применение живых настроек, запрос перезапуска нужных воркеров.
   - Корректный shutdown: остановка воркеров с таймаутом и жёстким убийством, сохранение переменных с `persistent=True`, флаш логов, закрытие БД.
   - Обработчик необработанных исключений и `faulthandler` в лог.
   - Защита от второго экземпляра (named mutex); повторный запуск показывает окно уже работающего.
3. `core/state.py`: текущее состояние помощника (`idle / listening / thinking / speaking / error`), режим микрофона (`always / ptt`), online/offline, mic on/off. Изменение состояния публикует событие — на это подпишутся оверлей и трей.
4. Тесты: публикация/подписка, отсутствие утечек после отписки, порядок shutdown, single-instance.

## Файлы
`src/ayris/core/events.py`, `src/ayris/core/app.py`, `src/ayris/core/state.py`, `src/ayris/__main__.py` (обновить), `tests/unit/test_events.py`

## Готово когда
- [ ] События из фонового потока доходят до подписчика в UI-потоке.
- [ ] `python -m ayris` поднимает и корректно гасит все подсистемы, в логе виден порядок.
- [ ] Второй запуск не создаёт второй экземпляр.
- [ ] Смена состояния публикует событие.
- [ ] Тесты зелёные.

## Промпт для нового чата
> Прочитай `AYRIS_SPEC.md`, `AYRIS_CONTEXT.md` и код проекта, затем выполни Задачу 04 из `tasks/04_events_app.md`: реализуй типизированный event bus с потокобезопасной доставкой в UI-поток, машину состояний помощника и класс `AyrisApp` с полным жизненным циклом (инициализация в правильном порядке, реакция на смену конфига, корректный shutdown, single-instance mutex, глобальный обработчик исключений). Обнови `__main__.py`. Напиши тесты. Прогони ruff, mypy, pytest.
