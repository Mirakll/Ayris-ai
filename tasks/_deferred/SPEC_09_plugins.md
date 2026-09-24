# Раздел 9 ТЗ — Плагины / расширяемость (отложено)

Этот раздел вырезан из `AYRIS_SPEC.md` при снятии плагинов с текущего плана.
Сохранён здесь целиком, чтобы вернуть его позже без потери деталей. Нумерация
раздела «9» в ТЗ оставлена как заглушка со ссылкой сюда, чтобы разделы 10–25 не
сдвинулись и ссылки «Разделы ТЗ: NN» в задачах не поехали.

---

## 9. ПЛАГИНЫ / РАСШИРЯЕМОСТЬ
### 9.1 Архитектура
- Папка `%APPDATA%\Ayris\plugins\` — каждый плагин = Python пакет с `manifest.json`:
```json
{
  "name": "home_assistant",
  "version": "1.0.0",
  "entry_point": "plugin:HomeAssistantPlugin",
  "permissions": ["network", "settings_read", "commands_register"],
  "api_version": 1
}
```
- **Песочница**: плагины запускаются в отдельном процессе (опционально), ограниченные права, запрос разрешений при установке.

### 9.2 SDK (API для плагинов)
- `register_command(trigger, action)` — регистрация своей команды.
- `say(text)`, `play_sound(path)`, `notify(title, body)`.
- `get_setting(key)`, `set_setting(key, value)`.
- `execute_action(action_name, params)` — вызов встроенного действия.
- `on_event(event_name, callback)` — хуки: `on_startup`, `on_shutdown`, `on_command`, `on_wake_word`, `on_audio_level`, `on_timer`.
- `register_hotkey(combo, callback)`.
- `variables.get/set(scope, name, value)`.

### 9.3 Официальные плагины (планируемые)
- Home Assistant / MQTT
- Spotify / Яндекс Музыка (продвинутое)
- Discord (RPC, команды)
- Obsidian / Notion
- Steam (запуск игр, статус)
- Browser automation (Playwright-based)

### 9.4 Пользовательские плагины
- Кнопка "+ Добавить кастомный плагин" → выбор папки с `manifest.json` → валидация → установка.

---

## Прочие упоминания, вырезанные из ТЗ

Хранилище (раздел 1.3):

```
| Плагины | Python пакеты | `%APPDATA%\Ayris\plugins\` |
```

UI, вкладка (раздел 8.1):

```
7. **Плагины** — список установленных, "+ Добавить кастомный плагин" (указание пути к Python пакету), маркетплейс (позже).
```

Безопасность (раздел 11):

```
- **Песочница плагинов** — изоляция процессов, запрос разрешений (network, fs, admin, audio, input).
```

Документация (раздел 19): в списке разделов были «Плагины» и «API для плагинов».

Родмап (раздел 20): «Этап 4: Плагины и Расширяемость» целиком и пункт
«Песочница плагинов (процессы + разрешения)» из Этапа 5.

Зависимости (раздел 21):

```toml
# Plugins
importlib-metadata >= 6.0
pluggy >= 1.3
```

Структура каталогов (раздел 24): пакет `src/ayris/plugins/{manager,sandbox,sdk}.py`,
вкладка `gui/tabs/plugins.py` и каталог верхнего уровня `plugins/` для официальных
плагинов (submodules).
