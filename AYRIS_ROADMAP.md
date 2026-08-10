# AYRIS — карта задач

74 задачи, 9 этапов. Один чат = одна задача.

**Как использовать:** открой новый чат, прикрепи `AYRIS_SPEC.md`, `AYRIS_CONTEXT.md` и файл задачи из `tasks/`, вставь промпт из конца файла задачи. По завершении отметь `[x]` здесь.

---

## Этап A — Фундамент
Без этого ничего не работает. Строго по порядку.

- [x] **01** — [Скелет проекта и tooling](tasks/01_skeleton.md) — зависит от: —
- [x] **02** — [Конфигурация и пути](tasks/02_config.md) — зависит от: 01
  - Появилось: `core/paths.py` (`get_paths()`/`init_paths()` — профиль, portable-режим, все каталоги), `core/config.py` (`get_settings()`, `ConfigManager.apply/reload/subscribe`, `RestartScope`/`restart_scope()` для полей «требует перезапуск»), `core/secrets.py` (`get_secrets()` — API-ключи в keyring, в TOML только `credential_ref`).
- [x] **03** — [База данных SQLite](tasks/03_database.md) — зависит от: 02
  - Появилось: `core/database.py` (`Database`/`init_database()`/`get_database()` — WAL, `foreign_keys=ON`, вложенные транзакции через SAVEPOINT, `backup()`/`restore()`/`vacuum()`), `core/migrations.py` (`apply_migrations()`, `SCHEMA_VERSION`, версионирование через `PRAGMA user_version` + таблица `schema_version`, только вперёд), `core/models.py` (замороженные датаклассы всех 11 таблиц + `utc_now()`/`to_db_timestamp()`), `core/repositories.py` (`Repositories` — `profiles`/`folders`/`commands`/`triggers`/`variables`/`history`/`audit`/`timers`/`clipboard`/`models`/`maintenance`, очистка по категориям `CleanupCategory` и бэкап).
- [x] **04** — [Event bus и жизненный цикл](tasks/04_events_app.md) — зависит от: 02, 03
  - Появилось: `core/events.py` (21 событие-датакласс + `EventBus.subscribe/unsubscribe/publish` — потокобезопасная очередь с доставкой в UI-поток через `wakeup`/`drain()`, слабые ссылки на методы, подписка на базовый класс), `core/state.py` (`StateMachine`/`StatusSnapshot`, `AssistantState` idle/listening/thinking/speaking/error, `MicMode`, онлайн-флаг — любое изменение публикует событие), `core/app.py` (`AyrisApp` — 14 этапов запуска в фиксированном порядке и остановка в обратном, `Component` для подключения подсистем, реакция на `ConfigChanged` + `register_restart_handler()`, single-instance через мьютекс, глобальные обработчики исключений и `faulthandler`), `__main__.py` (мост шины в очередь Qt).
- [x] **05** — [Воркеры и IPC](tasks/05_workers_ipc.md) — зависит от: 04
  - Появилось: `workers/protocol.py` (сообщения `Call`/`Result`/`Event`/`Heartbeat`/`Shutdown` поверх анонимного pipe, `AudioChunk`/`open_audio`/`share_audio` — передача PCM через shared memory без копий), `workers/base.py` (`Worker` + декоратор `@method()`, `WorkerContext` с `emit`/`check_cancelled`, поток heartbeat, проброс логов и типизированных ошибок в родителя, отсутствие импорта PySide6), `workers/manager.py` (`WorkerManager` — spawn-запуск, `call()`/`call_sync()` через `Future`, health-check по пропущенным ударам, авто-перезапуск с экспоненциальной задержкой и событиями шины, Job Object + watchdog против процессов-сирот), `workers/registry.py` (`WorkerKind`/`WorkerSpec`, `plan_workers()` — какие воркеры нужны при текущих настройках, режим экономии откладывает запуск до первого вызова).
- [x] **06** — [Профили и экспорт/импорт](tasks/06_profiles.md) — зависит от: 03
  - Появилось: `core/profile.py` (`ProfileManager` — `list_all`/`create`/`copy`/`rename`/`delete`/`switch` с защитой последнего профиля и переключением на лету, `subscribe()` для подписчиков + события `ProfileSwitched`/`ProfilesChanged`, `export`/`preview_import`/`import_bundle` с бэкапом перед импортом, `backup`/`list_backups`/`reset` (ротация до `MAX_BACKUPS`), `open_folder`, `change_root()` — перенос корня установки на другой диск/в облако с миграцией данных и откатом), `core/portable_profile.py` (формат `.zip` `ayris-profile` схемы v1: `export_bundle`/`preview_bundle`/`import_bundle`/`read_manifest`/`strip_secrets`, `BundleManifest`/`BundlePreview`/`ImportReport`, `ConflictPolicy` overwrite/rename/skip, секреты вырезаются и при экспорте и при импорте, защита от zip-slip и zip-bomb, атомарность — файлы через staging + БД одной транзакцией).

## Этап B — Аудио и речь
- [x] **07** — [Аудио захват и устройства](tasks/07_audio_capture.md) — зависит от: 05
- [x] **08** — [VAD, шумоподавление, калибровка](tasks/08_vad_denoise.md) — зависит от: 07
  - Появилось: `audio/vad.py` (`VadSettings` — один пользовательский порог 0..1 превращается в `gate_db` над измеренным уровнем шума, `Vad`/`WebRtcVad`/`EnergyVad` + `create_vad()` с автовыбором и запасным детектором по энергии, если колесо webrtcvad не собралось, `FrameSplitter`/`VadStream`/`frames_of()` — нарезка на кадры 10/20/30 мс независимо от того, какими блоками отдаёт драйвер), `audio/segmenter.py` (`Segmenter`/`segment_pcm()` — состояния idle/speech, подтверждение начала по `start_frames`, конец по `silence_ms`, pre-roll из кольцевого буфера прикладывается к сегменту, `min_speech_ms` отбрасывает щелчки, `max_utterance_ms` режет бесконечную фразу без потери непрерывности, `SpeechStart`/`SpeechSegment`/`SegmenterStats`/`SegmenterCallbacks`), `audio/denoise.py` (`DenoiseStream`/`denoise_pcm()`, `DenoiseMode` off/rnnoise/spectral — RNNoise через ctypes с ресемплом 16↔48 кГц, при её отсутствии молча включается `NoiseGate` — двухполосный экспандер на stdlib, `DenoiseStats` меряет добавленную задержку и `realtime_factor`), `audio/calibration.py` (`calibrate_pcm()`/`run_calibration()` — запись тишины и фразы «айрис открой браузер», `analyse_noise`/`analyse_phrase`/`recommend`, `CalibrationReport` с вердиктом good/noisy/quiet/clipping/no_speech, русскими подсказками и `as_dict()` для окна настроек). В аудио-воркере: методы `vad` (полная телеметрия детекции и подавления), `segment` (PCM последней фразы, по трубе не едет в событии), `calibrate` (этапы silence/phrase/report/reset, читают назад по кольцевому буферу и не блокируют); события `speech_started`/`speech_ended` → `SpeechStarted`/`SpeechEnded` на шине, `AudioLevelChanged` получил `is_speech`.
- [x] **09** — [Wake word](tasks/09_wake_word.md) — зависит от: 08
  - Появилось: `audio/wake_word/base.py` (ABC `WakeWordEngine` — контракт `load`/`process`/`unload`/`reset` и `sample_rate`/`frame_samples`, `WakePhrase` с чувствительностью 0.0–1.0 и производным порогом `threshold`, `WakeDetection`, `ModelSpec`, реестр `ENGINE_ENTRYPOINTS`/`engine_names()`/`engine_class()`/`create_engine()` с ленивым импортом движка, `PTT_PHRASE` для Push-to-Talk), `audio/wake_word/openwakeword_engine.py` (движок по умолчанию, бесплатный, ONNX, свои модели из каталога), `audio/wake_word/porcupine_engine.py` (Picovoice, AccessKey читается из keyring по `credential_ref`, ключ не покидает процесс-хозяин в настройках воркера), `audio/wake_word/vosk_engine.py` (KWS по грамматике, переиспользует модель офлайн-STT), `audio/wake_word/manager.py` (`WakeWordDetector` — неограниченный список вариантов слова с индивидуальной чувствительностью, ресемпл и нарезка кадров под требования движка, debounce 1.5–2 с по звуковым часам (общий и на фразу), поток инференса и очередь блоков, `add_phrase`/`remove_phrase`/`set_sensitivity`/`set_phrases` на лету без остановки захвата, `trigger_manual()` — точка входа Push-to-Talk (хоткей — задача 37), `WakeStats` с метрикой ложных срабатываний `false_positive_rate` (отказы в минуту) и разбивкой по причинам, `WakeWordSettings`/`WakeWordCallbacks`, сбой загрузки движка не роняет захват — причина остаётся в `stats.error`). В аудио-воркере: wake слушает тот же поток кадров, что и VAD (сырой блок — детектору, подавленный — сегментатору), методы `wake` (телеметрия и метрики), `wake_phrases` (add/remove/sensitivity/list), `trigger_manual`; событие `wake_word` → `WakeWordDetected` на шине, окно прослушивания `listen_window_sec` для режимов `always`/`ptt`/`hybrid`.
- [x] **10** — [STT офлайн](tasks/10_stt_offline.md) — зависит от: 05, 08
  - Появилось: `audio/stt/base.py` (ABC `SttEngine` — `load`/`transcribe`/`unload`, `AudioBuffer` с ресемплингом на stdlib, `TranscriptResult`/`TranscriptSegment` с таймингами и `real_time_factor`, `SttOptions`, ленивый реестр движков `create_engine()`), `audio/stt/vosk_engine.py` (потоковый, строго 16 кГц моно), `audio/stt/faster_whisper_engine.py` (автоопределение CUDA с откатом на CPU, фильтры галлюцинаций), `audio/stt/whispercpp_engine.py` (опционально), `workers/stt_worker.py` (аудио через разделяемую память, ресемплинг ровно один раз на входе, ленивая загрузка в отдельном потоке — диспетчер однопоточный, выгрузка по таймауту простоя, проверка лимита RAM до загрузки, тайминги в лог пайплайна и в `metrics`).
- [x] **11** — [STT онлайн и роутер Auto](tasks/11_stt_online.md) — зависит от: 10
  - Появилось: `audio/stt/cloud_base.py` (`CloudSttEngine` — общий httpx-клиент с раздельными таймаутами connect/read, ретраи с экспоненциальной задержкой и джиттером, `NetworkError` повторяется, `QuotaError` — никогда, ключ читается только из keyring по `credential_ref` и маскируется в логе, тело с аудио в лог не попадает, `as_wav()`, реестр `create_cloud_engine()`), четыре провайдера: `yandex_engine.py` (LPCM + `folderId`, API-ключ или IAM-токен), `google_engine.py` (base64 LINEAR16, ключ в заголовке `X-Goog-Api-Key`), `azure_engine.py` (WAV на региональный хост, `RecognitionStatus`/`NBest`), `openai_engine.py` (multipart собирается вручную, `verbose_json`, уверенность из `exp(avg_logprob)`); `core/connectivity.py` (`ConnectivityMonitor` — состояние ведут в первую очередь реальные отказы запросов, фоновая проба редкая и ходит только по адресу из конфига, возврат в онлайн после двух подряд удачных проб, публикует `OnlineStatusChanged`); `audio/stt/router.py` (`SttMode` offline/online/auto, автооткат на локальную модель при любом отказе облака и автовозврат без перезапуска по событию монитора, `preload()` греет офлайн-модель в фоне, чтобы переключение уложилось в 2 с, отсутствие локальной модели проговаривается уведомлением и в `user_message`). В конфиг добавлены `online_endpoint`/`online_region`/`online_folder_id`/`online_auth_scheme`/`online_model`/`online_retries`/`probe_url`/`probe_interval_sec`.
- [x] **12** — [TTS локальный и плеер](tasks/12_tts_local.md) — зависит от: 05
  - Появилось: `audio/tts/base.py` (ABC `TtsEngine` — `load`/`synthesize`/`synthesize_stream`/`unload` и классметод `voices()`, `AudioChunk` с `duration_ms`, `VoiceSpec` с `key`/`same_voice` для инвалидации кэша, `TtsOptions`, ленивый реестр `ENGINE_ENTRYPOINTS`/`engine_names()`/`engine_class()`/`create_engine()`, `estimate_voice_bytes()` для проверки лимита RAM до загрузки), `audio/tts/sentence_split.py` (разбивка с таблицей русских и английских сокращений — «т. е.», «ул. Ленина, д. 5», «рис. 3» не считаются концом предложения, как и инициалы, десятичные дроби и номера версий; слишком длинное предложение режется по запятой или тире, короткий хвост подклеивается к предыдущему куску, `is_speakable()` отсекает «...» и эмодзи, которые стоят вызова синтеза и возвращаются щелчком), `audio/tts/cache.py` (`PhraseCache` — LRU на диске, ключ по hash(text, voice, speed, pitch), лимит из конфига, атомарная запись `.wav`, запись отклоняется, если фраза больше четверти бюджета или длиннее 240 символов, `invalidate(voice)` при смене голоса, `CacheStats.hit_rate`), три движка: `piper_engine.py` (`.onnx` + `.json`, пользовательский каталог моделей, скорость через `length_scale`), `silero_engine.py` (русские голоса, `torch.package`, строго CPU), `coqui_engine.py` (опциональный XTTS v2 — регистрируется только когда пакет установлен, клонирование по референсной записи), `audio/tts/player.py` (`TtsPlayer` — две очереди с приоритетом для системных фраз, один поток записи, блоки по 20 мс, громкость 0.0–1.0 применяется к сэмплам и меняется внутри фразы, согласование частоты с устройством и ресемплинг при отказе, переоткрытие потока при выдёргивании наушников, `PlayerStats`), `workers/tts_worker.py` (`load_voice`/`synthesize`/`synthesize_stream`/`next_chunk`/`cancel`, PCM через разделяемую память, ленивая загрузка, выгрузка по простою, кооперативная отмена внутри самого синтеза). В `audio/devices.py` добавлен выходной поток (`OutputStream`/`PlaybackBackend`/`PlaybackRequest`), на шину — `TtsStarted(text, duration_estimate)` и `TtsFinished(reason: completed|cancelled|error)`, в конфиг — `voice.tts.cache_size_mb`.
- [ ] **13** — [TTS облачный и роутер](tasks/13_tts_cloud.md) — зависит от: 12
- [ ] **14** — [Менеджер моделей (бэкенд)](tasks/14_model_manager.md) — зависит от: 02

## Этап C — NLU
- [ ] **15** — [Matcher: exact / fuzzy / regex](tasks/15_nlu_matcher.md) — зависит от: 03
- [ ] **16** — [Слоты и нормализация](tasks/16_nlu_slots.md) — зависит от: 15
- [ ] **17** — [Контекст и follow-up](tasks/17_nlu_context.md) — зависит от: 16
- [ ] **18** — [Пайплайн-диспетчер](tasks/18_pipeline.md) — зависит от: 09, 11, 13, 17

## Этап D — Системные действия
- [ ] **19** — [Реестр действий](tasks/19_action_registry.md) — зависит от: 04
- [ ] **20** — [Программы и окна](tasks/20_apps_windows.md) — зависит от: 19
- [ ] **21** — [Аудио-действия](tasks/21_audio_actions.md) — зависит от: 19
- [ ] **22** — [Яркость и мониторы](tasks/22_display.md) — зависит от: 19
- [ ] **23** — [Эмуляция ввода](tasks/23_input.md) — зависит от: 19
- [ ] **24** — [Сеть и питание](tasks/24_network_power.md) — зависит от: 19
- [ ] **25** — [Браузер, поиск, мгновенные ответы](tasks/25_browser_search.md) — зависит от: 19
- [ ] **26** — [Скриншоты и OCR](tasks/26_screenshots_ocr.md) — зависит от: 19
- [ ] **27** — [Буфер обмена и автозаполнение](tasks/27_clipboard_autofill.md) — зависит от: 19
- [ ] **28** — [Таймеры, напоминания, будильники](tasks/28_timers.md) — зависит от: 03, 19
- [ ] **29** — [Яндекс Музыка](tasks/29_yandex_music.md) — зависит от: 19

## Этап E — Макросы
- [ ] **30** — [Схема .ayris и модель команды](tasks/30_macro_schema.md) — зависит от: 03, 19
- [ ] **31** — [Интерпретатор макросов](tasks/31_macro_engine.md) — зависит от: 30
- [ ] **32** — [Логика и переменные](tasks/32_macro_logic_vars.md) — зависит от: 31
- [ ] **33** — [Библиотека блоков](tasks/33_macro_blocks.md) — зависит от: 32, этап D
- [ ] **34** — [Звуковое сопровождение](tasks/34_macro_sounds.md) — зависит от: 13, 31
- [ ] **35** — [Отладчик макросов](tasks/35_macro_debugger.md) — зависит от: 32
- [ ] **36** — [Импорт VoiceAttack и AHK](tasks/36_import_vap_ahk.md) — зависит от: 33

## Этап F — Хоткеи и интеграция ядра
- [ ] **37** — [Глобальные хоткеи](tasks/37_hotkeys.md) — зависит от: 04
- [ ] **38** — [Триггеры: события и расписание](tasks/38_triggers.md) — зависит от: 31, 37
- [ ] **39** — [Права администратора и UAC](tasks/39_admin_uac.md) — зависит от: 19
- [ ] **40** — [Подтверждение опасных команд](tasks/40_confirmations.md) — зависит от: 39, 13
- [ ] **41** — [Логирование и аудит](tasks/41_logging_audit.md) — зависит от: 03

## Этап G — Интерфейс
- [ ] **42** — [Тема, токены, базовые виджеты](tasks/42_theme.md) — зависит от: 01
- [ ] **43** — [Каркас главного окна](tasks/43_main_window.md) — зависит от: 42, 02
- [ ] **44** — [Системный трей и автозапуск](tasks/44_tray_autostart.md) — зависит от: 43
- [ ] **45** — [Виджет сферы](tasks/45_sphere_widget.md) — зависит от: 42
- [ ] **46** — [Оверлей мини](tasks/46_overlay_mini.md) — зависит от: 45, 04
- [ ] **47** — [Оверлей расширенный](tasks/47_overlay_expanded.md) — зависит от: 46, 28
- [ ] **48** — [Вкладка Общие](tasks/48_tab_general.md) — зависит от: 43, 44
- [ ] **49** — [Вкладка Голос](tasks/49_tab_voice.md) — зависит от: 43, 09, 11, 13
- [ ] **50** — [UI менеджера моделей](tasks/50_ui_model_manager.md) — зависит от: 43, 14
- [ ] **51** — [Дерево команд](tasks/51_command_tree.md) — зависит от: 43, 30
- [ ] **52** — [Редактор команды: списочный режим](tasks/52_editor_list.md) — зависит от: 51, 33
- [ ] **53** — [Редактор команды: нодовый режим](tasks/53_editor_nodes.md) — зависит от: 52
- [ ] **54** — [Hot reload и история версий](tasks/54_hot_reload_versions.md) — зависит от: 52
- [ ] **55** — [Вкладка Горячие клавиши](tasks/55_tab_hotkeys.md) — зависит от: 43, 37
- [ ] **56** — [Вкладка Оверлей](tasks/56_tab_overlay.md) — зависит от: 43, 47
- [ ] **57** — [Вкладка Профили](tasks/57_tab_profiles.md) — зависит от: 43, 06
- [ ] **58** — [Вкладка Логи / DevTools](tasks/58_tab_devtools.md) — зависит от: 43, 41, 18
- [ ] **59** — [Вкладка Приватность](tasks/59_tab_privacy.md) — зависит от: 43, 03
- [ ] **60** — [Онбординг](tasks/60_onboarding.md) — зависит от: 49, 50, 06

## Этап H — LLM
- [ ] **61** — [Абстракция LLM и облачные провайдеры](tasks/61_llm_cloud.md) — зависит от: 05
- [ ] **62** — [Локальные LLM](tasks/62_llm_local.md) — зависит от: 61
- [ ] **63** — [Режимы NLU, промпты, память](tasks/63_llm_modes.md) — зависит от: 62, 18
- [ ] **64** — [Вкладка ИИ / LLM](tasks/64_tab_ai.md) — зависит от: 43, 63

## Этап I — Плагины, сборка, релиз
- [ ] **65** — [Ядро системы плагинов](tasks/65_plugin_core.md) — зависит от: 19, 04
- [ ] **66** — [SDK и песочница плагинов](tasks/66_plugin_sdk_sandbox.md) — зависит от: 65
- [ ] **67** — [Вкладка Плагины](tasks/67_tab_plugins.md) — зависит от: 43, 66
- [ ] **68** — [Официальные плагины](tasks/68_official_plugins.md) — зависит от: 66
- [ ] **69** — [Устойчивость и восстановление](tasks/69_resilience.md) — зависит от: 05, 07
- [ ] **70** — [Тесты и CI](tasks/70_tests_ci.md) — зависит от: этапы A–H
- [ ] **71** — [Сборка Nuitka и Portable](tasks/71_build.md) — зависит от: 70
- [ ] **72** — [Инсталлер и автообновления](tasks/72_installer_updates.md) — зависит от: 71
- [ ] **73** — [Документация MkDocs](tasks/73_docs.md) — зависит от: 72
- [ ] **74** — [Приёмка релиза](tasks/74_release_check.md) — зависит от: всё

---

## Минимальный рабочий срез
Если хочется увидеть живого помощника раньше всего остального, достаточно этой цепочки:
**01 → 02 → 03 → 04 → 05 → 07 → 08 → 09 → 10 → 12 → 15 → 16 → 18 → 19 → 20 → 21 → 42 → 43 → 44 → 45 → 46**
Получится: слышит «Айрис», распознаёт офлайн, отвечает голосом, открывает программы и крутит громкость, показывает оверлей со сферой.

## Что можно делать параллельно в разных чатах
- После задачи 19 — все задачи 20–29 независимы друг от друга.
- После задачи 43 — вкладки 48, 55, 56, 57, 59 независимы.
- Задачи 14, 41, 42 можно брать почти в любой момент после этапа A.

