"""Typed settings: schema, TOML persistence, hot reload, restart scopes.

The schema mirrors the settings window one-to-one, so a tab is a section and a
control is a field: :class:`GeneralConfig`, :class:`VoiceConfig` (with ``stt``,
``tts``, ``wake`` and ``audio_input`` inside it), :class:`AiConfig`,
:class:`ActionsConfig`, :class:`HotkeysConfig`, :class:`OverlayConfig`,
:class:`PluginsConfig`, :class:`PrivacyConfig`, :class:`PerformanceConfig`,
:class:`UpdatesConfig` and :class:`DevtoolsConfig`. Adding a control means adding
a field, and the TOML file, the environment overrides and the change diff follow
for free.

Three properties are worth knowing before touching this module.

**Settings are immutable.** Every model is frozen. A change produces a new
:class:`Settings` instance which :class:`ConfigManager` swaps in under a lock, so
a worker thread reading ``get_settings()`` can never observe a half-applied
update.

**Not everything can be applied live.** An audio device or a model path cannot
change under a running worker. Those fields are tagged in the schema with a
:class:`RestartScope`; :meth:`ConfigChanged.restart_scopes` reports which workers
the supervisor has to restart, and the live part of the diff is applied
immediately. Nothing in this module restarts anything by itself.

**Secrets are not here.** Fields named ``credential_ref`` hold the *name* of a
credential entry; the key itself lives in the Windows Credential Manager through
:mod:`ayris.core.secrets`. Dumping settings to TOML or to a log can therefore
never leak a key.

Failure handling follows section 17 of the specification. A file that does not
parse is moved to ``config.toml.broken`` and the application starts on defaults.
A file that parses but fails validation is recovered field by field: the invalid
leaves are dropped, the rest is kept, and the names of the dropped fields are
reported through :attr:`ConfigManager.dropped_fields` so the GUI can tell the
user which controls were reset.

Hot reload uses a stdlib polling thread rather than ``QFileSystemWatcher``:
``ayris.core`` must not import Qt, and the worker processes need the same
mechanism without an event loop. Listeners run on that thread — a GUI listener
has to marshal to the UI thread itself.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, Final, Literal, TypeGuard

import tomlkit
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import Table
from tomlkit.toml_document import TOMLDocument

from ayris.core.errors import ConfigError
from ayris.core.paths import get_paths
from ayris.utils.logger import LogLevel, get_logger

__all__ = [
    "DEFAULT_SEARCH_PROVIDERS",
    "ENV_PREFIX",
    "RAM_LIMIT_CHOICES",
    "SCHEMA_VERSION",
    "ActionsConfig",
    "AiConfig",
    "AudioActionsConfig",
    "AudioInputConfig",
    "AutofillActionsConfig",
    "BrowserActionsConfig",
    "ClipboardActionsConfig",
    "CommandsConfig",
    "ConfigChange",
    "ConfigChanged",
    "ConfigListener",
    "ConfigManager",
    "ConfigSection",
    "DevtoolsConfig",
    "GeneralConfig",
    "HotkeysConfig",
    "InputActionsConfig",
    "InstantActionsConfig",
    "MediaActionsConfig",
    "OcrActionsConfig",
    "OverlayConfig",
    "PerformanceConfig",
    "PluginsConfig",
    "PrivacyConfig",
    "RestartScope",
    "ScreenshotActionsConfig",
    "Settings",
    "SttConfig",
    "TtsConfig",
    "UpdatesConfig",
    "VoiceConfig",
    "WakeConfig",
    "WakePhrase",
    "diff_settings",
    "dump_settings",
    "get_config_manager",
    "get_settings",
    "init_config",
    "load_settings",
    "reset_config_manager",
    "restart_scope",
    "save_settings",
]

_log = get_logger("core.config")

#: Bumped whenever a migration is needed. Written into every file.
SCHEMA_VERSION: Final = 1

#: ``AYRIS_VOICE__TTS__SPEED=1.4`` overrides one field for one launch.
ENV_PREFIX: Final = "AYRIS_"

#: Allowed values for :attr:`PerformanceConfig.ram_limit_mb`. ``0`` means no cap.
RAM_LIMIT_CHOICES: Final[tuple[int, ...]] = (0, 2048, 4096, 8192, 16384)

_SETTLE_POLLS: Final = 2
_DEFAULT_POLL_INTERVAL: Final = 1.0
_MAX_RECOVERY_ROUNDS: Final = 8


class RestartScope(StrEnum):
    """Which subsystem has to be restarted for a field to take effect.

    :attr:`NONE` is the common case: the new value is picked up on the next read.
    Anything else names a worker the supervisor must recycle, and
    :attr:`APP` means the whole application.
    """

    NONE = "none"
    AUDIO = "audio"
    WAKE = "wake"
    STT = "stt"
    TTS = "tts"
    LLM = "llm"
    APP = "app"

    @property
    def label(self) -> str:
        """Russian text for the "требуется перезапуск" hint in the settings tab."""
        return _RESTART_LABELS[self]


_RESTART_LABELS: Final[Mapping[RestartScope, str]] = {
    RestartScope.NONE: "применяется сразу",
    RestartScope.AUDIO: "требуется перезапуск захвата звука",
    RestartScope.WAKE: "требуется перезапуск распознавания слова активации",
    RestartScope.STT: "требуется перезапуск распознавания речи",
    RestartScope.TTS: "требуется перезапуск синтеза речи",
    RestartScope.LLM: "требуется перезапуск языковой модели",
    RestartScope.APP: "требуется перезапуск Ayris",
}

_RESTART_KEY: Final = "restart"


def _restart(scope: RestartScope) -> dict[str, Any]:
    """Tag a field with the restart it needs. Goes into ``json_schema_extra``.

    The return type is ``dict[str, Any]`` rather than the narrower
    ``dict[str, str]`` because pydantic declares the parameter as its own
    ``JsonDict`` alias, which is ``dict[str, Any]``. ``dict`` is invariant in
    its value type, so a ``dict[str, str]`` is not accepted there.
    """
    return {_RESTART_KEY: scope.value}


def _scope_of(info: FieldInfo) -> RestartScope:
    """Read the restart tag off one field, defaulting to :attr:`RestartScope.NONE`."""
    extra = info.json_schema_extra
    if not isinstance(extra, dict):
        return RestartScope.NONE
    raw = extra.get(_RESTART_KEY)
    if not isinstance(raw, str):
        return RestartScope.NONE
    try:
        return RestartScope(raw)
    except ValueError:  # pragma: no cover - only reachable via a typo in a tag
        return RestartScope.NONE


@cache
def restart_scope(dotted: str) -> RestartScope:
    """Restart scope of the field at ``dotted``, e.g. ``"voice.audio_input.device"``.

    Resolution walks as deep into the schema as the path goes and returns the
    scope of the deepest field it could resolve, so a change *inside* a tagged
    container (a dict of hotkeys, a tuple of wake phrases) inherits the tag of the
    container itself.
    """
    fields: Mapping[str, FieldInfo] = Settings.model_fields
    scope = RestartScope.NONE
    for part in dotted.split("."):
        info = fields.get(part)
        if info is None:
            break
        scope = _scope_of(info)
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields = annotation.model_fields
        else:
            break
    return scope


class ConfigSection(BaseModel):
    """Base for every settings section.

    Frozen, so a section can be shared between threads without copying.
    ``extra="ignore"`` keeps an unknown key written by a newer Ayris from breaking
    an older one; ``validate_assignment`` is pointless on a frozen model and is
    left off.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )


class GeneralConfig(ConfigSection):
    """Tab «Общие» — language, theme, startup and tray behaviour."""

    language: Literal["ru"] = Field(
        default="ru",
        description="Язык интерфейса и распознавания. Пока поддерживается только русский",
    )
    theme: Literal["dark_purple", "dark", "light", "system"] = Field(
        default="dark_purple",
        description="Оформление окна и оверлея",
    )
    autostart: bool = Field(
        default=False,
        description="Запускать Ayris вместе с Windows",
    )
    start_minimized: bool = Field(
        default=False,
        description="Стартовать свёрнутым в трей",
    )
    minimize_to_tray: bool = Field(
        default=True,
        description="Кнопка «свернуть» убирает окно в трей",
    )
    close_to_tray: bool = Field(
        default=True,
        description="Кнопка «закрыть» убирает окно в трей, а не выходит из программы",
    )
    show_tray_notifications: bool = Field(
        default=True,
        description="Всплывающие уведомления из трея",
    )
    show_onboarding: bool = Field(
        default=True,
        description="Показать мастер первого запуска. Снимается автоматически после него",
    )
    single_instance: bool = Field(
        default=True,
        description="Не давать запустить вторую копию Ayris",
        json_schema_extra=_restart(RestartScope.APP),
    )


class PerformanceConfig(ConfigSection):
    """Tab «Общие», lower half — process priority, memory cap, thread pools."""

    process_priority: Literal["normal", "above_normal", "high"] = Field(
        default="normal",
        description="Приоритет главного процесса",
        json_schema_extra=_restart(RestartScope.APP),
    )
    audio_priority: Literal["normal", "above_normal", "high", "realtime"] = Field(
        default="high",
        description="Приоритет процесса захвата звука. Ниже «high» возможны пропуски",
        json_schema_extra=_restart(RestartScope.AUDIO),
    )
    ram_limit_mb: int = Field(
        default=4096,
        description="Мягкий лимит памяти в МБ. 0 — без ограничения",
    )
    stt_threads: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Потоков распознавания речи",
        json_schema_extra=_restart(RestartScope.STT),
    )
    tts_threads: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Потоков синтеза речи",
        json_schema_extra=_restart(RestartScope.TTS),
    )
    llm_threads: int = Field(
        default=1,
        ge=1,
        le=2,
        description="Потоков языковой модели",
        json_schema_extra=_restart(RestartScope.LLM),
    )
    macro_threads: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Потоков одновременного выполнения макросов",
    )
    eco_mode: bool = Field(
        default=False,
        description="Экономный режим: выгружать простаивающие модели из памяти",
        json_schema_extra=_restart(RestartScope.APP),
    )
    model_idle_sec: float = Field(
        default=300.0,
        ge=0.0,
        le=3600.0,
        description="Через сколько секунд простоя выгружать модель. 0 — держать всегда",
    )
    gpu: Literal["auto", "cuda", "cpu"] = Field(
        default="auto",
        description="Ускорение распознавания на видеокарте",
        json_schema_extra=_restart(RestartScope.STT),
    )

    @field_validator("ram_limit_mb")
    @classmethod
    def _known_ram_limit(cls, value: int) -> int:
        """Only the values the settings tab offers, so the combo box never desyncs."""
        if value not in RAM_LIMIT_CHOICES:
            allowed = ", ".join(str(choice) for choice in RAM_LIMIT_CHOICES)
            raise ValueError(f"допустимые значения лимита памяти: {allowed}")
        return value


class SttConfig(ConfigSection):
    """Tab «Голос» → speech recognition."""

    mode: Literal["offline", "online", "auto"] = Field(
        default="auto",
        description="offline — только локально, online — только облако, auto — облако с откатом",
    )
    offline_engine: Literal["gigaam", "vosk", "whisper"] = Field(
        default="gigaam",
        description="Локальный движок распознавания",
        json_schema_extra=_restart(RestartScope.STT),
    )
    offline_model: str = Field(
        default="gigaam-v3-ctc",
        description="Имя папки модели в models/stt",
        json_schema_extra=_restart(RestartScope.STT),
    )
    online_provider: Literal["yandex", "google", "azure", "openai"] = Field(
        default="yandex",
        description="Облачный провайдер распознавания",
    )
    credential_ref: str = Field(
        default="yandex",
        description="Имя записи с ключом в хранилище Windows. Сам ключ здесь не хранится",
    )
    online_timeout_sec: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Сколько ждать ответ облака перед откатом на офлайн",
    )
    online_endpoint: str = Field(
        default="",
        description="Свой адрес облачного сервиса. Пусто — адрес по умолчанию у провайдера",
    )
    online_region: str = Field(
        default="",
        description="Регион Azure Speech, например westeurope. Другим провайдерам не нужен",
    )
    online_folder_id: str = Field(
        default="",
        description="Идентификатор каталога Яндекс Облака. Не секрет, но без него запрос не идёт",
    )
    online_auth_scheme: Literal["api-key", "iam"] = Field(
        default="api-key",
        description="Способ авторизации Яндекса: долгий API-ключ или IAM-токен на 12 часов",
    )
    online_model: str = Field(
        default="",
        description="Модель у провайдера, если он даёт выбор. Пусто — модель по умолчанию",
    )
    online_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Сколько раз повторить запрос при сетевой ошибке. Квоту не повторяем никогда",
    )
    probe_url: str = Field(
        default="https://www.gstatic.com/generate_204",
        description="Единственный адрес фоновой проверки связи. Пусто — не проверять в фоне",
    )
    probe_interval_sec: float = Field(
        default=120.0,
        ge=15.0,
        le=3600.0,
        description="Как часто проверять связь в фоне. Реже — тише сеть, дольше возврат в онлайн",
    )
    punctuation: bool = Field(
        default=True,
        description="Расставлять знаки препинания, если движок умеет",
    )
    partial_results: bool = Field(
        default=True,
        description="Показывать промежуточный текст в оверлее по мере распознавания",
    )
    min_confidence: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Ниже этой уверенности фраза считается нераспознанной",
    )


class TtsConfig(ConfigSection):
    """Tab «Голос» → speech synthesis."""

    engine: Literal["piper", "silero", "xtts", "sapi", "yandex", "elevenlabs"] = Field(
        default="piper",
        description="Движок синтеза речи",
        json_schema_extra=_restart(RestartScope.TTS),
    )
    voice: str = Field(
        default="ru_RU-irina-medium",
        description="Имя голоса или файла модели в models/tts",
        json_schema_extra=_restart(RestartScope.TTS),
    )
    speed: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Скорость речи: 1.0 — обычная",
    )
    pitch: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Высота голоса: 1.0 — без изменения",
    )
    volume: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Громкость озвучки в процентах",
    )
    output_device: str = Field(
        default="",
        description="Устройство вывода. Пусто — системное по умолчанию",
        json_schema_extra=_restart(RestartScope.TTS),
    )
    cloud_fallback: bool = Field(
        default=False,
        description="Если локальный синтез не справился, озвучить через облако",
    )
    credential_ref: str = Field(
        default="yandex",
        description="Имя записи с ключом облачного синтеза в хранилище Windows",
    )
    duck_other_audio: bool = Field(
        default=True,
        description="Приглушать музыку и игры во время ответа",
    )
    interrupt_on_speech: bool = Field(
        default=True,
        description="Прерывать озвучку, если пользователь начал говорить",
    )
    cache_size_mb: int = Field(
        default=64,
        ge=0,
        le=4096,
        description="Лимит кэша озвученных фраз на диске в МБ. 0 — не кэшировать",
    )


class WakePhrase(ConfigSection):
    """One wake word variant with its own sensitivity."""

    phrase: str = Field(min_length=2, max_length=40, description="Слово активации")
    sensitivity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог срабатывания: выше — чаще ложные, ниже — чаще пропуски",
    )
    enabled: bool = Field(default=True, description="Учитывать этот вариант")

    @field_validator("phrase")
    @classmethod
    def _normalise(cls, value: str) -> str:
        """Wake words are matched case-insensitively, so store them folded."""
        folded = " ".join(value.split()).lower()
        if not folded:
            raise ValueError("слово активации не может быть пустым")
        return folded


class WakeConfig(ConfigSection):
    """Tab «Голос» → wake word."""

    enabled: bool = Field(default=True, description="Реагировать на слово активации")
    engine: Literal["openwakeword", "porcupine", "vosk"] = Field(
        default="openwakeword",
        description="Движок распознавания слова активации",
        json_schema_extra=_restart(RestartScope.WAKE),
    )
    phrases: tuple[WakePhrase, ...] = Field(
        default=(
            WakePhrase(phrase="айрис"),
            WakePhrase(phrase="аирис"),
            WakePhrase(phrase="ирис", sensitivity=0.65),
        ),
        description="Варианты произношения. Пустой список отключает активацию голосом",
        json_schema_extra=_restart(RestartScope.WAKE),
    )
    debounce_ms: int = Field(
        default=1500,
        ge=200,
        le=10000,
        description="Не срабатывать повторно в течение этого времени",
    )
    mic_mode: Literal["always", "ptt", "hybrid"] = Field(
        default="hybrid",
        description="always — микрофон всегда открыт, ptt — по клавише, hybrid — оба способа",
    )
    listen_window_sec: float = Field(
        default=6.0,
        ge=1.0,
        le=30.0,
        description="Сколько слушать команду после активации",
    )
    credential_ref: str = Field(
        default="porcupine",
        description="Имя записи с AccessKey Porcupine в хранилище Windows",
    )

    @field_validator("phrases")
    @classmethod
    def _unique(cls, value: tuple[WakePhrase, ...]) -> tuple[WakePhrase, ...]:
        """Duplicates would make the engine fire twice for one utterance."""
        seen: set[str] = set()
        unique: list[WakePhrase] = []
        for item in value:
            if item.phrase in seen:
                continue
            seen.add(item.phrase)
            unique.append(item)
        return tuple(unique)


class AudioInputConfig(ConfigSection):
    """Tab «Голос» → microphone, VAD and noise suppression."""

    device: str = Field(
        default="",
        description="Имя устройства записи. Пусто — системное по умолчанию",
        json_schema_extra=_restart(RestartScope.AUDIO),
    )
    sample_rate: Literal[8000, 16000, 32000, 48000] = Field(
        default=16000,
        description="Частота дискретизации в Гц. Движки распознавания ждут 16000",
        json_schema_extra=_restart(RestartScope.AUDIO),
    )
    frame_ms: Literal[10, 20, 30] = Field(
        default=20,
        description="Длина кадра для VAD в миллисекундах",
        json_schema_extra=_restart(RestartScope.AUDIO),
    )
    gain: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Программное усиление сигнала",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Порог отделения речи от тишины. Настраивается калибровкой",
    )
    vad_aggressiveness: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Агрессивность WebRTC VAD: 0 — мягко, 3 — жёстко",
    )
    silence_ms: int = Field(
        default=800,
        ge=200,
        le=5000,
        description="Тишина такой длины считается концом фразы",
    )
    max_utterance_sec: float = Field(
        default=30.0,
        ge=2.0,
        le=120.0,
        description="Максимальная длина одной фразы",
    )
    denoise: Literal["off", "rnnoise", "spectral"] = Field(
        default="rnnoise",
        description="Шумоподавление на входе",
        json_schema_extra=_restart(RestartScope.AUDIO),
    )
    noise_floor_db: float = Field(
        default=-45.0,
        ge=-90.0,
        le=0.0,
        description="Уровень шума комнаты, определяется калибровкой микрофона",
    )


class VoiceConfig(ConfigSection):
    """Tab «Голос» as a whole."""

    stt: SttConfig = Field(default_factory=SttConfig, description="Распознавание речи")
    tts: TtsConfig = Field(default_factory=TtsConfig, description="Синтез речи")
    wake: WakeConfig = Field(default_factory=WakeConfig, description="Слово активации")
    audio_input: AudioInputConfig = Field(
        default_factory=AudioInputConfig,
        description="Микрофон и обработка входного сигнала",
    )


class AiConfig(ConfigSection):
    """Tab «ИИ / LLM» — the three toggles, providers, prompts and memory."""

    fallback_to_llm: bool = Field(
        default=True,
        description="Если команда не распознана точным совпадением, спросить модель",
    )
    llm_understanding: bool = Field(
        default=False,
        description="Разбирать смысл фразы моделью, а не только по шаблонам",
    )
    free_chat: bool = Field(
        default=False,
        description="Свободный разговор с моделью, а не только команды",
    )
    provider: Literal[
        "ollama",
        "lmstudio",
        "llamacpp",
        "openai",
        "anthropic",
        "openrouter",
        "deepseek",
        "gigachat",
    ] = Field(
        default="ollama",
        description="Поставщик языковой модели",
        json_schema_extra=_restart(RestartScope.LLM),
    )
    model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Название модели у выбранного поставщика",
        json_schema_extra=_restart(RestartScope.LLM),
    )
    host: str = Field(
        default="http://127.0.0.1:11434",
        description="Адрес локального сервера моделей (Ollama, LM Studio)",
        json_schema_extra=_restart(RestartScope.LLM),
    )
    credential_ref: str = Field(
        default="openai",
        description="Имя записи с ключом в хранилище Windows. Сам ключ здесь не хранится",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Разброс ответов: 0 — предсказуемо, 2 — творчески",
    )
    max_tokens: int = Field(
        default=1024,
        ge=64,
        le=32768,
        description="Максимальная длина ответа модели",
    )
    request_timeout_sec: float = Field(
        default=30.0,
        ge=2.0,
        le=300.0,
        description="Сколько ждать ответ модели",
    )
    history_turns: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Сколько последних реплик держать в контексте",
    )
    summarize_after_turns: int = Field(
        default=30,
        ge=0,
        le=500,
        description="После скольких реплик сворачивать историю в краткий пересказ. 0 — никогда",
    )
    chat_system_prompt: str = Field(
        default=(
            "Ты Айрис — голосовой помощник на компьютере пользователя. "
            "Отвечай кратко и по-русски: ответ будет озвучен вслух."
        ),
        description="Системный промпт для свободного разговора",
    )
    nlu_system_prompt: str = Field(
        default=(
            "Определи, какую команду просит выполнить пользователь, и верни JSON "
            "с полями intent и slots. Если подходящей команды нет, верни intent: null."
        ),
        description="Системный промпт для разбора команд",
    )

    @model_validator(mode="after")
    def _summary_after_history(self) -> AiConfig:
        """Summarising sooner than the window is dropped would lose the recent turns."""
        if 0 < self.summarize_after_turns < self.history_turns:
            raise ValueError(
                "сворачивать историю нужно позже, чем заканчивается окно последних реплик "
                f"({self.history_turns})"
            )
        return self


class CommandsConfig(ConfigSection):
    """Tab «Команды» — matching behaviour of the command library."""

    fuzzy_threshold: float = Field(
        default=0.82,
        ge=0.5,
        le=1.0,
        description="Насколько похожей должна быть фраза, чтобы сработала команда",
    )
    default_cooldown_ms: int = Field(
        default=0,
        ge=0,
        le=600000,
        description="Пауза между повторами команды, если у неё не задана своя",
    )
    max_parallel: int = Field(
        default=3,
        ge=1,
        le=16,
        description="Сколько команд может выполняться одновременно",
    )
    stop_word: str = Field(
        default="отмена",
        description="Слово, прерывающее выполнение и озвучку",
    )
    announce_start: bool = Field(
        default=False,
        description="Проговаривать начало выполнения долгой команды",
    )
    followup_ttl_s: float = Field(
        default=30.0,
        ge=0.0,
        le=600.0,
        description="Сколько секунд можно продолжать разговор без обращения по имени",
    )
    object_ttl_s: float = Field(
        default=60.0,
        ge=0.0,
        le=3600.0,
        description="Сколько секунд «его» и «это» указывают на последний объект",
    )
    answer_ttl_s: float = Field(
        default=600.0,
        ge=0.0,
        le=86400.0,
        description="Сколько секунд «повтори» ещё возвращает последний ответ",
    )
    clarify_timeout_s: float = Field(
        default=20.0,
        ge=0.0,
        le=300.0,
        description="Сколько ждать ответа на уточняющий вопрос",
    )


class AudioActionsConfig(ConfigSection):
    """Sub-section ``[actions.audio]`` — how far one «громче» moves the volume.

    Ten percent is what the media keys on most keyboards do; five is inaudible
    over a video and twenty overshoots. The same step serves the microphone: a
    user who says «микрофон тише» means the same size of change they mean for
    the speakers.
    """

    volume_step: int = Field(
        default=10,
        ge=1,
        le=50,
        description="На сколько процентов «громче» и «тише» меняют громкость",
    )


class InputActionsConfig(ConfigSection):
    """Sub-section ``[actions.input]`` — the pacing of synthesised input.

    Every number here is a delay, and each of them exists because zero does not
    work. Applications read the input queue on their own schedule: a chord sent
    with no gap between the modifier and the key arrives as two unrelated
    presses in some, and a text of two hundred characters typed with no pause
    loses the middle of itself in others. The defaults are the smallest values
    that survived a browser, a terminal and Notepad; a macro that needs to be
    slower says so in its own block.

    ``backend`` picks how the events are injected. ``sendinput`` is the Windows
    API and works everywhere except against a window of an elevated process;
    ``interception`` is a kernel driver that also reaches full-screen games, and
    Ayris falls back to ``sendinput`` with a line in the log when it is not
    installed — it is never installed silently, that needs administrator rights
    and a reboot.
    """

    backend: Literal["sendinput", "interception"] = Field(
        default="sendinput",
        description="Чем отправлять ввод: SendInput или драйвер Interception для игр",
        json_schema_extra=_restart(RestartScope.APP),
    )
    key_delay_ms: int = Field(
        default=12,
        ge=0,
        le=2000,
        description="Пауза между нажатиями клавиш",
    )
    key_hold_ms: int = Field(
        default=30,
        ge=0,
        le=5000,
        description="Сколько держать клавишу нажатой",
    )
    char_delay_ms: int = Field(
        default=6,
        ge=0,
        le=1000,
        description="Пауза между символами при наборе текста",
    )
    clipboard_threshold: int = Field(
        default=200,
        ge=0,
        le=100_000,
        description="С какой длины текст вставлять через буфер обмена; 0 — никогда",
    )
    drag_step_px: int = Field(
        default=24,
        ge=1,
        le=500,
        description="Длина одного шага при перетаскивании мышью, в пикселях",
    )
    drag_step_delay_ms: int = Field(
        default=8,
        ge=0,
        le=1000,
        description="Пауза между шагами перетаскивания",
    )
    mouse_settle_ms: int = Field(
        default=16,
        ge=0,
        le=1000,
        description="Пауза после перемещения курсора перед нажатием кнопки",
    )


#: Search providers shipped with Ayris, as ``{query}`` templates.
#:
#: The templates live in the settings rather than in the code because a search
#: engine is a matter of taste and because the list is never complete: a user
#: who searches a wiki, a torrent tracker or an internal help desk adds a line
#: here and says «найди в <имя>» without touching Python. The placeholder is
#: filled with the percent-encoded query, so a template may put it in the path
#: as well as in the query string.
DEFAULT_SEARCH_PROVIDERS: Final[Mapping[str, str]] = {
    "google": "https://www.google.com/search?q={query}",
    "yandex": "https://yandex.ru/search/?text={query}",
    "youtube": "https://www.youtube.com/results?search_query={query}",
    "duckduckgo": "https://duckduckgo.com/?q={query}",
    "wikipedia": "https://ru.wikipedia.org/w/index.php?search={query}",
}


class BrowserActionsConfig(ConfigSection):
    """Sub-section ``[actions.browser]`` — where links and searches go.

    ``default_provider`` is the engine used when the phrase names none, and
    ``providers`` is the whole table of them: five come with Ayris, and anything
    a user adds is merged on top, so a template can be both added and
    overridden. Yandex is the default because the assistant answers in Russian
    and a Russian query returns better results there; every field here is a
    matter of preference, not of correctness.

    ``browser`` names the browser to open links in — empty means the one Windows
    considers default, and anything else is resolved through the application
    index by the same name the user would say out loud («хром», «фаерфокс»).
    """

    default_provider: str = Field(
        default="yandex",
        min_length=1,
        max_length=64,
        description="Поисковик по умолчанию, если во фразе он не назван",
    )
    providers: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_SEARCH_PROVIDERS),
        description="Шаблоны поиска: имя = URL с {query}; свой добавляется строкой",
    )
    browser: str = Field(
        default="",
        max_length=200,
        description="В каком браузере открывать ссылки; пусто — браузер по умолчанию",
    )
    private_by_default: bool = Field(
        default=False,
        description="Всегда открывать в приватном окне",
    )


class InstantActionsConfig(ConfigSection):
    """Sub-section ``[actions.instant]`` — the short spoken answers.

    Every number here exists to keep Ayris inside the limits of free public
    APIs. The three time-to-live values are the rates at which the data itself
    changes: a forecast is recomputed about every ten minutes, the Central Bank
    publishes rates once a business day, and an encyclopedia article does not
    move at all — an hour and a day of caching cost nothing and turn a repeated
    question into zero requests.

    ``city`` answers «какая погода» with no city named. It is a plain string and
    not a coordinate pair on purpose: the user types the name they would say,
    and the geocoder resolves it once and keeps the answer in the same cache.

    ``stale_hours`` is how old a cached answer may be before Ayris refuses to
    read it out at all when the network is gone. Yesterday's weather with the
    date said out loud is useful; a forecast from last week is not.

    ``offline`` is the switch that says «do not go out at all». With it on Ayris
    answers from what it already has and otherwise says «нет подключения к сети»,
    exactly as it does with the cable pulled — which is the point: a metered
    connection or a flight should behave like no connection, not like a slow one.
    """

    city: str = Field(
        default="Москва",
        min_length=1,
        max_length=120,
        description="Город по умолчанию для погоды и времени",
    )
    offline: bool = Field(
        default=False,
        description="Офлайн-режим: не выходить в сеть за ответом ни при каких условиях",
    )
    weather_ttl_min: int = Field(
        default=10,
        ge=1,
        le=1440,
        description="Сколько минут держать прогноз в кэше",
    )
    rates_ttl_min: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Сколько минут держать курс валют в кэше",
    )
    facts_ttl_min: int = Field(
        default=1440,
        ge=1,
        le=20_160,
        description="Сколько минут держать справки и определения в кэше",
    )
    timeout_s: float = Field(
        default=6.0,
        ge=0.5,
        le=60.0,
        description="Сколько ждать ответа от сервиса",
    )
    retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Сколько раз повторить запрос при сетевой ошибке",
    )
    stale_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="До какого возраста озвучивать устаревший кэш офлайн",
    )


class ScreenshotActionsConfig(ConfigSection):
    """Sub-section ``[actions.screenshot]`` — where a capture goes and under what name.

    ``output`` is the whole difference between the two ways people use
    screenshots: ``file`` for the ones that are kept, ``clipboard`` for the ones
    that are pasted into a chat and never wanted on disk, ``both`` when it is not
    known in advance. Nothing else in the section changes behaviour that much.

    ``directory`` empty means :attr:`ayris.core.paths.AppPaths.screenshots_dir` —
    a folder inside the profile, created on first use. A custom path is taken as
    given, and it is the one place a screenshot can end up outside the profile.

    ``filename`` is a template rather than a fixed pattern because the useful
    name depends on the habit: ``{date}_{time}`` sorts chronologically in any file
    manager, ``{window}_{n}`` groups a series taken from one program. The
    placeholders are ``{date}``, ``{time}``, ``{monitor}``, ``{window}`` and
    ``{n}``; an unknown one is left alone rather than raising, because a typo in
    the settings must not break the action.

    ``jpeg_quality`` only matters when ``format`` is ``jpeg``. PNG is the default:
    a screenshot is mostly flat colour and sharp text, which is what PNG is good
    at and JPEG is worst at — the ringing around letters is visible at any
    quality below about 90.
    """

    output: Literal["file", "clipboard", "both"] = Field(
        default="both",
        description="Куда девать снимок: файл, буфер обмена или и то и другое",
    )
    directory: str = Field(
        default="",
        max_length=400,
        description="Папка для снимков; пусто — папка «Скриншоты» в профиле",
    )
    filename: str = Field(
        default="ayris_{date}_{time}",
        min_length=1,
        max_length=200,
        description="Шаблон имени файла: {date}, {time}, {monitor}, {window}, {n}",
    )
    format: Literal["png", "jpeg"] = Field(
        default="png",
        description="Формат файла: png (без потерь) или jpeg",
    )
    jpeg_quality: int = Field(
        default=92,
        ge=40,
        le=100,
        description="Качество JPEG, если формат jpeg",
    )
    dim_opacity: float = Field(
        default=0.45,
        ge=0.0,
        le=0.9,
        description="Насколько затемнять экран при выделении области",
    )
    selection_timeout_s: float = Field(
        default=60.0,
        ge=2.0,
        le=600.0,
        description="Сколько ждать выделения области, прежде чем отменить",
    )


class OcrActionsConfig(ConfigSection):
    """Sub-section ``[actions.ocr]`` — reading text off the screen.

    ``engine`` is a preference, not a promise: ``auto`` takes the first available
    of Windows OCR, Tesseract and PaddleOCR, and naming one explicitly still falls
    back if it turns out not to be installed. Windows OCR comes first because it
    is the only one already present on every Windows 10/11 — offline, with Russian
    and English in the box and nothing to install.

    ``languages`` is what the engine is told to expect, best first. Order matters
    more than it looks: Windows OCR picks one language and one only, and asking it
    for English on Russian text returns confident-looking garbage rather than
    nothing.

    ``upscale_to_dpi`` is the one preprocessing knob that reliably changes the
    result. Screen text is 12–14 px tall at 96 DPI and every engine is trained on
    scans around 300; scaling the crop up before recognition is what turns
    «Настройки» from unreadable into read. ``binarize`` is off by default because
    it helps a photographed page and hurts anti-aliased screen text.

    ``speak_limit`` caps how much of a recognised wall of text is read out. The
    whole of it still goes to the clipboard; only the spoken part is cut, at a
    sentence boundary where there is one.
    """

    engine: Literal["auto", "windows", "tesseract", "paddle"] = Field(
        default="auto",
        description="Движок распознавания; auto — первый доступный",
    )
    languages: list[str] = Field(
        default_factory=lambda: ["ru", "en"],
        min_length=1,
        max_length=8,
        description="Языки распознавания, важнейший первым",
    )
    output: Literal["clipboard", "speak", "both"] = Field(
        default="both",
        description="Куда девать распознанный текст",
    )
    speak_limit: int = Field(
        default=400,
        ge=40,
        le=4000,
        description="Сколько символов озвучивать; остальное только в буфер",
    )
    upscale_to_dpi: int = Field(
        default=300,
        ge=96,
        le=600,
        description="До какого DPI увеличивать снимок перед распознаванием",
    )
    binarize: bool = Field(
        default=False,
        description="Приводить снимок к чёрно-белому перед распознаванием",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Отбрасывать блоки с уверенностью ниже (0 — не отбрасывать)",
    )
    tesseract_path: str = Field(
        default="",
        max_length=400,
        description="Путь к tesseract.exe, если он не в PATH",
    )


class ClipboardActionsConfig(ConfigSection):
    """Sub-section ``[actions.clipboard]`` — the clipboard history.

    ``monitor`` is the master switch. It is on by default because a history
    nobody enabled is a history nobody has, and the whole point is that «вставь
    третий» works for something copied ten minutes ago. Turning it off stops the
    listener and leaves the recorded entries alone: the setting governs what gets
    written from now on, not what is already there.

    ``limit`` is the number of *unpinned* entries kept. Pinned ones are outside
    the count entirely — a person who pinned an address did so precisely so that a
    day of copying would not push it out.

    ``max_length`` skips over-long entries instead of truncating them. A truncated
    entry looks fine in the list and pastes half a document, which is worse than
    not offering it: a copied file, a base64 blob or a whole page has no place in
    a spoken «вставь второй» anyway.

    ``preview_length`` is only about the list the user hears or sees; the stored
    text is never shortened.

    ``skip_password_managers`` honours the two clipboard formats managers set to
    ask monitors to look away — ``Clipboard Viewer Ignore`` and
    ``ExcludeClipboardContentFromMonitorProcessing``. It is the difference between
    a history and a plaintext password file, so turning it off is a deliberate act
    and is left possible only because a manager can set the marker on something
    harmless.
    """

    monitor: bool = Field(
        default=True,
        description="Записывать копирования в историю буфера",
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Сколько незакреплённых записей хранить",
    )
    max_length: int = Field(
        default=20_000,
        ge=16,
        le=1_000_000,
        description="Длиннее этого — в историю не пишется",
    )
    preview_length: int = Field(
        default=80,
        ge=8,
        le=400,
        description="Длина превью записи в списке",
    )
    skip_password_managers: bool = Field(
        default=True,
        description="Не сохранять то, что менеджер паролей просил не сохранять",
    )
    clear_after_secret: bool = Field(
        default=True,
        description="Очищать буфер сразу после вставки пароля или карты",
    )
    warn_windows_history: bool = Field(
        default=True,
        description="Предупреждать, если включена история буфера Windows (Win+V)",
    )


class AutofillActionsConfig(ConfigSection):
    """Sub-section ``[actions.autofill]`` — templates and password managers.

    Only the non-secret half of a template lives here. A value is either written
    down in ``config.toml`` — a city, a street, an email — or it is a reference to
    the Credential Manager or to a password manager, and then the settings file
    holds the reference and never the value. That split is what makes it safe to
    put ``config.toml`` in a backup.

    ``provider`` picks where a reference with no explicit source is looked up.
    ``keyring`` is the Windows Credential Manager and needs nothing installed;
    the other two shell out to ``keepassxc-cli`` and ``bw``, which is why they are
    opt-in and why the paths below exist for a CLI that is not in ``PATH``.

    ``session_ttl_s`` is how long an unlocked vault stays usable. The session
    token lives in memory only — never on disk, never in the log — so quitting
    Ayris always relocks. Ten minutes is long enough to fill a form and short
    enough that a walk away from an unlocked machine is not a whole afternoon.

    ``clear_clipboard`` and ``paste_mode`` decide how a value reaches the field.
    Typing it avoids the clipboard entirely and is the default for that reason;
    the clipboard route exists for the fields that refuse synthesised keystrokes,
    and there the clipboard is wiped the moment the paste is done.
    """

    provider: Literal["keyring", "keepass", "bitwarden"] = Field(
        default="keyring",
        description="Откуда брать секретные значения по умолчанию",
    )
    paste_mode: Literal["type", "clipboard"] = Field(
        default="type",
        description="Как подставлять значение: печатать или через буфер",
    )
    clear_clipboard: bool = Field(
        default=True,
        description="Очищать буфер после подстановки через буфер",
    )
    session_ttl_s: int = Field(
        default=600,
        ge=30,
        le=7200,
        description="Сколько секунд держать хранилище разблокированным",
    )
    keepass_cli: str = Field(
        default="",
        max_length=400,
        description="Путь к keepassxc-cli.exe, если он не в PATH",
    )
    keepass_database: str = Field(
        default="",
        max_length=400,
        description="Путь к базе KeePass (.kdbx)",
    )
    keepass_key_file: str = Field(
        default="",
        max_length=400,
        description="Файл-ключ к базе KeePass, если он используется",
    )
    bitwarden_cli: str = Field(
        default="",
        max_length=400,
        description="Путь к bw.exe, если он не в PATH",
    )
    templates: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Шаблоны: имя → поле → значение или ссылка secret:имя",
    )


class MediaActionsConfig(ConfigSection):
    """Sub-section ``[actions.media]`` — the player Ayris controls.

    Two levels of control, and most of these settings are about the second one.
    Pause, play, next, previous and «что играет» go through Windows' own media
    transport and need nothing configured; starting a named artist, a playlist or
    Моя волна means driving Яндекс Музыка's own interface, and that needs the app
    to be running with a DevTools port open.

    ``player_app_id`` is how the app is recognised in Windows' list of players —
    ``SourceAppUserModelId``, not a process name. Clearing it makes the transport
    actions address whichever player is currently playing, which is what somebody
    who does not use Яндекс Музыка wants.

    ``launch_app`` lets Ayris start the app itself, with the debug flags, when it
    is not running. It has to: Яндекс Музыка has no autostart entry to add a flag
    to, so an app started by hand never has the port. An app already running
    *without* the port is never restarted — that would cut the music off — and the
    advanced actions say so instead.

    ``media_keys_fallback`` is off, and that is deliberate rather than cautious.
    A synthesised key press goes to whatever has focus, and the person using this
    plays games where keys are binds; the transport above sends the order straight
    to the player's session object instead, so nothing is broadcast. The fallback
    exists only for a player that publishes no session at all.

    ``selectors`` overrides individual entries of
    :data:`ayris.actions.media.yandex_music.SELECTORS`. The app's ``data-test-id``
    names are its internal business and an update can rename one; the override
    makes that a line in ``config.toml`` instead of a wait for a new Ayris.
    """

    player_app_id: str = Field(
        default="ru.yandex.desktop.music",
        max_length=200,
        description="Идентификатор плеера в Windows; пусто — любой играющий",
    )
    player_name: str = Field(
        default="Яндекс Музыка",
        max_length=120,
        description="Под каким названием искать программу, чтобы запустить",
    )
    player_path: str = Field(
        default="",
        max_length=400,
        description="Путь к exe плеера, если его не находит поиск программ",
    )
    debug_port: int = Field(
        default=9222,
        ge=1024,
        le=65535,
        description="Порт отладки, через который Ayris управляет приложением",
    )
    launch_app: bool = Field(
        default=True,
        description="Запускать Яндекс Музыку самой, если она закрыта",
    )
    launch_timeout_s: float = Field(
        default=20.0,
        ge=1.0,
        le=120.0,
        description="Сколько секунд ждать, пока приложение откроет порт",
    )
    command_timeout_s: float = Field(
        default=15.0,
        ge=1.0,
        le=120.0,
        description="Сколько секунд ждать ответа от приложения на команду",
    )
    render_timeout_ms: int = Field(
        default=8000,
        ge=500,
        le=60_000,
        description="Сколько миллисекунд ждать, пока страница нарисует нужное",
    )
    media_keys_fallback: bool = Field(
        default=False,
        description="Если плеер не виден Windows — жать медиа-клавиши (в играх нежелательно)",
    )
    selectors: dict[str, str] = Field(
        default_factory=dict,
        description="Переопределения селекторов интерфейса: имя → CSS-селектор",
    )


class ActionsConfig(ConfigSection):
    """Tab «Действия» — what the action library reads out of the settings."""

    audio: AudioActionsConfig = Field(
        default_factory=AudioActionsConfig,
        description="Громкость и звуковые устройства",
    )
    input: InputActionsConfig = Field(
        default_factory=InputActionsConfig,
        description="Клавиатура и мышь",
    )
    browser: BrowserActionsConfig = Field(
        default_factory=BrowserActionsConfig,
        description="Браузер и поисковые провайдеры",
    )
    instant: InstantActionsConfig = Field(
        default_factory=InstantActionsConfig,
        description="Мгновенные ответы: погода, курс, время, справки",
    )
    screenshot: ScreenshotActionsConfig = Field(
        default_factory=ScreenshotActionsConfig,
        description="Снимки экрана: куда сохранять, формат, имя файла",
    )
    ocr: OcrActionsConfig = Field(
        default_factory=OcrActionsConfig,
        description="Распознавание текста с экрана",
    )
    clipboard: ClipboardActionsConfig = Field(
        default_factory=ClipboardActionsConfig,
        description="История буфера обмена: монитор, лимиты, закрепление",
    )
    autofill: AutofillActionsConfig = Field(
        default_factory=AutofillActionsConfig,
        description="Шаблоны автозаполнения и менеджеры паролей",
    )
    media: MediaActionsConfig = Field(
        default_factory=MediaActionsConfig,
        description="Плеер: Яндекс Музыка и управление воспроизведением",
    )


#: Fields of :class:`HotkeysConfig` that hold a combo, in settings-tab order.
_HOTKEY_FIELDS: Final[tuple[str, ...]] = (
    "push_to_talk",
    "toggle_wake",
    "toggle_overlay",
    "toggle_mute",
    "cancel",
    "open_settings",
)


class HotkeysConfig(ConfigSection):
    """Tab «Горячие клавиши» — the assistant's own global shortcuts."""

    push_to_talk: str = Field(default="ctrl+shift+space", description="Говорить, пока зажата")
    toggle_wake: str = Field(default="ctrl+shift+a", description="Включить или выключить активацию")
    toggle_overlay: str = Field(default="ctrl+shift+o", description="Показать или скрыть оверлей")
    toggle_mute: str = Field(default="ctrl+shift+m", description="Отключить микрофон")
    cancel: str = Field(default="esc", description="Прервать выполнение и озвучку")
    open_settings: str = Field(default="ctrl+shift+s", description="Открыть окно настроек")
    use_interception: bool = Field(
        default=False,
        description="Драйвер Interception — хоткеи работают в полноэкранных играх",
        json_schema_extra=_restart(RestartScope.APP),
    )
    suppress_in_games: bool = Field(
        default=True,
        description="Не передавать нажатие дальше, если хоткей сработал",
    )

    @field_validator(
        "push_to_talk",
        "toggle_wake",
        "toggle_overlay",
        "toggle_mute",
        "cancel",
        "open_settings",
    )
    @classmethod
    def _normalise_combo(cls, value: str) -> str:
        """Store combos folded, so ``Ctrl+Shift+A`` and ``ctrl+shift+a`` collide."""
        parts = [part.strip().lower() for part in value.split("+") if part.strip()]
        return "+".join(parts)

    @model_validator(mode="after")
    def _no_conflicts(self) -> HotkeysConfig:
        """Two actions on one combo — the settings tab shows this as a conflict."""
        assigned: dict[str, str] = {}
        for name in _HOTKEY_FIELDS:
            combo = getattr(self, name)
            if not combo:
                continue
            previous = assigned.get(combo)
            if previous is not None:
                raise ValueError(f"сочетание «{combo}» уже занято действием «{previous}»")
            assigned[combo] = name
        return self


class OverlayConfig(ConfigSection):
    """Tab «Оверлей» — the always-on-top window with the dotted sphere."""

    enabled: bool = Field(default=True, description="Показывать оверлей")
    mode: Literal["mini", "expanded"] = Field(
        default="mini",
        description="Режим при запуске: капсула со сферой или развёрнутая панель",
    )
    position: Literal[
        "top_left", "top_right", "bottom_left", "bottom_right", "center", "custom"
    ] = Field(default="bottom_right", description="Где показывать окно")
    custom_x: int = Field(default=0, description="Координата X при позиции «custom»")
    custom_y: int = Field(default=0, description="Координата Y при позиции «custom»")
    monitor: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Номер монитора: 0 — основной",
    )
    opacity: float = Field(
        default=0.92,
        ge=0.2,
        le=1.0,
        description="Прозрачность окна",
    )
    scale: float = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Масштаб оверлея независимо от масштаба системы",
    )
    click_through: bool = Field(
        default=False,
        description="Пропускать клики сквозь оверлей",
    )
    hide_when_idle: bool = Field(
        default=False,
        description="Прятать оверлей, пока помощник не слушает",
    )
    idle_hide_sec: float = Field(
        default=10.0,
        ge=1.0,
        le=600.0,
        description="Через сколько секунд бездействия прятать",
    )
    animations: bool = Field(default=True, description="Анимации сферы")
    sphere_points: int = Field(
        default=600,
        ge=100,
        le=3000,
        description="Точек в сфере. Меньше — легче для слабых видеокарт",
    )
    target_fps: int = Field(
        default=60,
        ge=15,
        le=144,
        description="Частота кадров анимации",
    )
    show_transcript: bool = Field(
        default=True,
        description="Показывать распознанный текст в развёрнутом режиме",
    )
    follow_system_theme: bool = Field(
        default=False,
        description="Брать тему из Windows вместо темы из «Общих»",
    )


class PluginsConfig(ConfigSection):
    """Tab «Плагины»."""

    enabled: bool = Field(
        default=True,
        description="Загружать плагины при старте",
        json_schema_extra=_restart(RestartScope.APP),
    )
    sandbox: bool = Field(
        default=True,
        description="Запускать плагины в отдельном процессе с урезанными правами",
        json_schema_extra=_restart(RestartScope.APP),
    )
    disabled: tuple[str, ...] = Field(
        default=(),
        description="Имена плагинов, которые не загружать",
    )
    extra_dirs: tuple[str, ...] = Field(
        default=(),
        description="Дополнительные папки с плагинами помимо профиля",
        json_schema_extra=_restart(RestartScope.APP),
    )
    allow_network: bool = Field(
        default=False,
        description="Разрешать плагинам сетевые запросы без отдельного подтверждения",
    )


class PrivacyConfig(ConfigSection):
    """Tab «Приватность». Telemetry is off and stays off unless asked."""

    telemetry: bool = Field(
        default=False,
        description="Отправка обезличенной статистики. По умолчанию выключено",
    )
    store_history: bool = Field(
        default=True,
        description="Хранить историю распознанных фраз и ответов",
    )
    history_limit: int = Field(
        default=1000,
        ge=0,
        le=100000,
        description="Сколько записей истории держать. 0 — не ограничивать",
    )
    store_audio: bool = Field(
        default=False,
        description="Сохранять записи голоса на диск. По умолчанию выключено",
    )
    audit_commands: bool = Field(
        default=True,
        description="Вести журнал выполненных команд",
    )
    require_confirmation: bool = Field(
        default=True,
        description="Спрашивать подтверждение перед опасными действиями",
    )
    confirmation_method: Literal["voice", "dialog", "both"] = Field(
        default="both",
        description="Как спрашивать подтверждение",
    )
    clear_on_exit: bool = Field(
        default=False,
        description="Стирать историю и кэш при выходе",
    )


class UpdatesConfig(ConfigSection):
    """Tab «Обновления» — application and model updates."""

    check_on_start: bool = Field(
        default=True,
        description="Проверять обновления при запуске",
    )
    channel: Literal["stable", "beta"] = Field(
        default="stable",
        description="Канал обновлений",
    )
    auto_download: bool = Field(
        default=False,
        description="Скачивать обновление автоматически",
    )
    auto_install: bool = Field(
        default=False,
        description="Устанавливать обновление без вопроса при следующем запуске",
    )
    check_models: bool = Field(
        default=True,
        description="Проверять обновления моделей вместе с программой",
    )
    interval_hours: int = Field(
        default=24,
        ge=1,
        le=720,
        description="Как часто проверять обновления",
    )


class DevtoolsConfig(ConfigSection):
    """Tab «Логи / DevTools»."""

    log_level: LogLevel = Field(default="INFO", description="Подробность журнала")
    console_log: bool = Field(
        default=False,
        description="Дублировать журнал в консоль. Нужно при запуске из терминала",
        json_schema_extra=_restart(RestartScope.APP),
    )
    pipeline_log: bool = Field(
        default=False,
        description="Подробный журнал пайплайна с таймингами каждого шага",
    )
    log_retention_days: int = Field(
        default=7,
        ge=0,
        le=365,
        description="Сколько дней хранить журналы. 0 — не удалять",
    )
    log_max_mb: int = Field(
        default=10,
        ge=1,
        le=500,
        description="Размер файла журнала, после которого он делится на части",
    )
    repl_enabled: bool = Field(
        default=False,
        description="Встроенная консоль Python в окне настроек",
        json_schema_extra=_restart(RestartScope.APP),
    )
    text_input: bool = Field(
        default=True,
        description="Поле для ввода команд текстом вместо голоса",
    )


class Settings(BaseSettings):
    """Every Ayris setting, one attribute per settings tab.

    Frozen: :class:`ConfigManager` replaces the whole object instead of mutating
    it, so readers never see a partially applied change.

    Environment variables override the file for one launch, which is how the test
    suite and the crash-recovery run pin a value without touching disk::

        AYRIS_VOICE__TTS__SPEED=1.4
        AYRIS_DEVTOOLS__LOG_LEVEL=DEBUG
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
    )

    schema_version: int = Field(
        default=SCHEMA_VERSION,
        description="Версия схемы настроек. Меняется только при миграциях",
    )
    general: GeneralConfig = Field(default_factory=GeneralConfig, description="Общие")
    voice: VoiceConfig = Field(default_factory=VoiceConfig, description="Голос")
    commands: CommandsConfig = Field(default_factory=CommandsConfig, description="Команды")
    actions: ActionsConfig = Field(default_factory=ActionsConfig, description="Действия")
    ai: AiConfig = Field(default_factory=AiConfig, description="ИИ / LLM")
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig, description="Горячие клавиши")
    overlay: OverlayConfig = Field(default_factory=OverlayConfig, description="Оверлей")
    plugins: PluginsConfig = Field(default_factory=PluginsConfig, description="Плагины")
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig, description="Приватность")
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Производительность",
    )
    updates: UpdatesConfig = Field(default_factory=UpdatesConfig, description="Обновления")
    devtools: DevtoolsConfig = Field(default_factory=DevtoolsConfig, description="Логи / DevTools")

    def secret_refs(self) -> dict[str, str]:
        """Credential reference per subsystem, skipping the empty ones.

        Feeds :func:`ayris.core.secrets.resolve_secrets`, so a worker receives
        only the key it needs.
        """
        candidates = {
            "stt": self.voice.stt.credential_ref,
            "tts": self.voice.tts.credential_ref,
            "wake": self.voice.wake.credential_ref,
            "ai": self.ai.credential_ref,
        }
        return {slot: ref for slot, ref in candidates.items() if ref}


#: Section order in the generated file, and the Russian header above each one.
_SECTION_TITLES: Final[tuple[tuple[str, str], ...]] = (
    ("general", "Общие: язык, тема, автозапуск, поведение окна"),
    ("voice", "Голос: распознавание, синтез, слово активации, микрофон"),
    ("commands", "Команды: сопоставление фраз и ограничения выполнения"),
    ("actions", "Действия: шаг громкости и прочие настройки системных действий"),
    ("ai", "ИИ / LLM: режимы, поставщик, промпты, память"),
    ("hotkeys", "Горячие клавиши помощника"),
    ("overlay", "Оверлей: положение, вид, анимации"),
    ("plugins", "Плагины"),
    ("privacy", "Приватность: телеметрия выключена по умолчанию"),
    ("performance", "Производительность: приоритеты, память, потоки"),
    ("updates", "Обновления программы и моделей"),
    ("devtools", "Логи и инструменты разработчика"),
)

_FILE_HEADER: Final[tuple[str, ...]] = (
    "Настройки Ayris.",
    "",
    "Файл можно править вручную — изменения применяются на ходу,",
    "кроме полей, помеченных «требуется перезапуск».",
    "",
    "API-ключи здесь НЕ хранятся: в полях credential_ref указано только имя",
    "записи в диспетчере учётных данных Windows.",
    "",
    "Если файл окажется повреждён, Ayris переименует его в config.toml.broken",
    "и запустится со значениями по умолчанию.",
)


def _read_document(path: Path) -> TOMLDocument:
    """Parse ``path`` into a tomlkit document, preserving comments.

    Raises:
        ConfigError: The file cannot be read or is not valid TOML.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(
            f"cannot read {path}: {exc}",
            user_message=f"Не удалось прочитать файл настроек:\n{path}",
        ) from exc
    try:
        # utf-8-sig: Notepad writes a BOM, and tomlkit refuses to parse it.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid UTF-8: {exc}",
            user_message="Файл настроек повреждён: неверная кодировка.",
        ) from exc
    try:
        return tomlkit.parse(text)
    except TOMLKitError as exc:
        raise ConfigError(
            f"{path} is not valid TOML: {exc}",
            user_message=f"Файл настроек повреждён и не читается:\n{exc}",
        ) from exc


def _removable_path(payload: Mapping[str, Any], loc: tuple[int | str, ...]) -> tuple[str, ...]:
    """Longest prefix of ``loc`` that actually exists in ``payload``.

    A validation error points at a leaf; dropping that leaf is enough when it is
    a plain key, but for a bad element inside a list the whole list has to go.
    Walking the payload tells us which of the two this is.
    """
    node: Any = payload
    resolved: list[str] = []
    for part in loc:
        if not isinstance(part, str) or not isinstance(node, Mapping) or part not in node:
            break
        resolved.append(part)
        node = node[part]
    return tuple(resolved)


def _drop(payload: MutableMapping[str, Any], path: tuple[str, ...]) -> bool:
    """Remove ``path`` from ``payload``. Returns whether anything was removed."""
    node: Any = payload
    for part in path[:-1]:
        if not isinstance(node, MutableMapping) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, MutableMapping) or path[-1] not in node:
        return False
    del node[path[-1]]
    return True


def settings_from_mapping(data: Mapping[str, Any]) -> tuple[Settings, tuple[str, ...]]:
    """Validate ``data``, dropping the fields that refuse to validate.

    A single bad value must not cost the user every other setting they changed,
    so invalid leaves are removed one round at a time and validation is retried.
    The loop is bounded — a payload that keeps failing falls back to defaults.

    Returns:
        The settings and the dotted names of the fields that were dropped.
    """
    payload: dict[str, Any] = dict(data)
    dropped: list[str] = []
    for _attempt in range(_MAX_RECOVERY_ROUNDS):
        try:
            return Settings.model_validate(payload), tuple(dropped)
        except ValidationError as exc:
            removed_this_round = False
            for error in exc.errors():
                target = _removable_path(payload, error["loc"])
                if target and _drop(payload, target):
                    dropped.append(".".join(target))
                    removed_this_round = True
            if not removed_this_round:
                break
    _log.warning("Configuration could not be repaired; falling back to defaults")
    return Settings(), (*dropped, "*")


def load_settings(path: Path) -> tuple[Settings, tuple[str, ...]]:
    """Read one settings file.

    Returns:
        The settings and the dotted names of any fields that had to be dropped.

    Raises:
        ConfigError: The file exists but cannot be parsed. The caller decides
            whether to rescue it — see :meth:`ConfigManager.load`.
    """
    if not path.is_file():
        return Settings(), ()
    document = _read_document(path)
    return settings_from_mapping(document.unwrap())


def dump_settings(settings: Settings) -> dict[str, Any]:
    """Plain JSON-compatible mapping of every setting. Contains no secrets."""
    return settings.model_dump(mode="json")


def _atomic_write(path: Path, text: str) -> None:
    """Write via a temporary file, so a crash cannot truncate the settings.

    Raises:
        ConfigError: The file could not be written.
    """
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(text, encoding="utf-8", newline="\n")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ConfigError(
            f"cannot write {path}: {exc}",
            user_message=(
                f"Не удалось сохранить настройки:\n{path}\nПроверьте права доступа к папке."
            ),
        ) from exc


def _field_note(dotted: str, info: FieldInfo | None) -> str:
    """Trailing comment for one field: its description plus the restart warning."""
    parts: list[str] = []
    if info is not None and info.description:
        parts.append(info.description)
    scope = restart_scope(dotted)
    if scope is not RestartScope.NONE:
        parts.append(scope.label)
    return ". ".join(parts)


def _field_info(dotted: str) -> FieldInfo | None:
    """Look up one field in the schema by its dotted path."""
    fields: Mapping[str, FieldInfo] = Settings.model_fields
    info: FieldInfo | None = None
    for part in dotted.split("."):
        info = fields.get(part)
        if info is None:
            return None
        annotation = info.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            fields = annotation.model_fields
        else:
            fields = {}
    return info


def _is_table(value: Any) -> TypeGuard[Mapping[str, Any]]:
    """Whether a dumped value should be written as a TOML table.

    A ``TypeGuard`` rather than a plain ``bool`` so that callers get the
    narrowing: the values come out of ``model_dump()`` as ``Any``, and the
    branch guarded by this check then passes them where a mapping is required.
    """
    return isinstance(value, Mapping)


def _to_toml_value(value: Any) -> Any:
    """Convert a dumped value into something tomlkit can serialise.

    Tuples become arrays; ``None`` has no TOML representation, so a field that is
    ``None`` is written as an empty string, which every optional field in the
    schema also accepts on the way back in.
    """
    if value is None:
        return ""
    if isinstance(value, tuple):
        return [_to_toml_value(item) for item in value]
    if isinstance(value, list):
        return [_to_toml_value(item) for item in value]
    return value


def _build_table(data: Mapping[str, Any], prefix: str) -> Table:
    """Render one section, annotating each scalar with its description."""
    table = tomlkit.table()
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if _is_table(value):
            table.append(key, _build_table(value, dotted))
            continue
        item = tomlkit.item(_to_toml_value(value))
        note = _field_note(dotted, _field_info(dotted))
        if note:
            item.comment(note)
        table.append(key, item)
    return table


def _build_document(settings: Settings) -> TOMLDocument:
    """Create a fully commented file from scratch."""
    document = tomlkit.document()
    for line in _FILE_HEADER:
        document.add(tomlkit.comment(line) if line else tomlkit.nl())
    document.add(tomlkit.nl())

    data = dump_settings(settings)
    version = tomlkit.item(data["schema_version"])
    version.comment("Версия схемы. Не меняйте вручную")
    document.add("schema_version", version)
    document.add(tomlkit.nl())

    for name, title in _SECTION_TITLES:
        section = data.get(name)
        if not _is_table(section):
            continue
        document.add(tomlkit.comment(title))
        document.add(name, _build_table(section, name))
        document.add(tomlkit.nl())
    return document


def _merge_into_table(
    target: MutableMapping[str, Any],
    data: Mapping[str, Any],
    prefix: str,
) -> None:
    """Write ``data`` into an existing table, leaving untouched keys as they are.

    Keys whose value did not change are skipped entirely, so the user's own
    comments and formatting survive a save. New keys — fields added by a newer
    Ayris — arrive with the generated description attached.
    """
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if _is_table(value):
            existing = target.get(key)
            if isinstance(existing, MutableMapping):
                _merge_into_table(existing, value, dotted)
            else:
                target[key] = _build_table(value, dotted)
            continue

        converted = _to_toml_value(value)
        if key in target:
            current = target[key]
            if _unwrapped(current) == converted:
                continue
            item = tomlkit.item(converted)
            comment = _existing_comment(current)
            if comment:
                item.comment(comment)
            target[key] = item
            continue

        item = tomlkit.item(converted)
        note = _field_note(dotted, _field_info(dotted))
        if note:
            item.comment(note)
        target[key] = item


def _unwrapped(item: Any) -> Any:
    """Plain Python value behind a tomlkit item."""
    unwrap = getattr(item, "unwrap", None)
    return unwrap() if callable(unwrap) else item


def _existing_comment(item: Any) -> str:
    """The user's own trailing comment on a value, if there is one."""
    trivia = getattr(item, "trivia", None)
    comment = getattr(trivia, "comment", "") if trivia is not None else ""
    return comment.lstrip("#").strip() if comment else ""


def save_settings(settings: Settings, path: Path) -> None:
    """Write ``settings`` to ``path``, preserving comments in an existing file.

    Raises:
        ConfigError: The file could not be written.
    """
    document: TOMLDocument | None = None
    if path.is_file():
        try:
            document = _read_document(path)
        except ConfigError:
            # A broken file is about to be replaced wholesale; the caller has
            # already backed it up.
            document = None

    if document is None:
        document = _build_document(settings)
    else:
        _merge_into_table(document, dump_settings(settings), "")

    _atomic_write(path, tomlkit.dumps(document))


@dataclass(frozen=True, slots=True)
class ConfigChange:
    """One field that changed, with the restart it implies."""

    path: str
    old: Any
    new: Any
    restart: RestartScope = RestartScope.NONE

    @property
    def live(self) -> bool:
        """Whether the new value takes effect without restarting anything."""
        return self.restart is RestartScope.NONE

    def __str__(self) -> str:
        return f"{self.path}: {self.old!r} -> {self.new!r}"


@dataclass(frozen=True, slots=True)
class ConfigChanged:
    """The full diff between two :class:`Settings`.

    Handed to every listener. Subsystems filter it by prefix — the audio worker
    looks at ``voice.audio_input``, the overlay at ``overlay`` — instead of
    reloading everything on any change.
    """

    changes: tuple[ConfigChange, ...]
    settings: Settings

    def __bool__(self) -> bool:
        return bool(self.changes)

    def __iter__(self) -> Iterator[ConfigChange]:
        return iter(self.changes)

    def __len__(self) -> int:
        return len(self.changes)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(change.path for change in self.changes)

    @property
    def live(self) -> tuple[ConfigChange, ...]:
        """Changes that are already in effect."""
        return tuple(change for change in self.changes if change.live)

    @property
    def restart_required(self) -> tuple[ConfigChange, ...]:
        """Changes that need a worker or the application restarted."""
        return tuple(change for change in self.changes if not change.live)

    @property
    def restart_scopes(self) -> frozenset[RestartScope]:
        """Which workers the supervisor has to recycle."""
        return frozenset(change.restart for change in self.changes if not change.live)

    def touches(self, prefix: str) -> bool:
        """Whether anything under ``prefix`` changed, e.g. ``"voice.tts"``."""
        return any(
            change.path == prefix or change.path.startswith(f"{prefix}.") for change in self.changes
        )

    def summary(self) -> str:
        """One-line Russian summary for the log and the toast."""
        if not self.changes:
            return "настройки не изменились"
        names = ", ".join(self.paths[:5])
        tail = f" и ещё {len(self.changes) - 5}" if len(self.changes) > 5 else ""
        return f"изменено: {names}{tail}"


#: A listener may not raise; :class:`ConfigManager` logs and keeps going if it does.
ConfigListener = Callable[[ConfigChanged], None]


def _walk(prefix: str, old: Any, new: Any, sink: list[ConfigChange]) -> None:
    """Collect leaf differences between two dumped trees."""
    if isinstance(old, Mapping) and isinstance(new, Mapping):
        for key in {*old, *new}:
            path = f"{prefix}.{key}" if prefix else str(key)
            _walk(path, old.get(key), new.get(key), sink)
        return
    if old != new:
        sink.append(ConfigChange(path=prefix, old=old, new=new, restart=restart_scope(prefix)))


def diff_settings(old: Settings, new: Settings) -> ConfigChanged:
    """Compare two settings objects field by field."""
    sink: list[ConfigChange] = []
    _walk("", dump_settings(old), dump_settings(new), sink)
    sink.sort(key=lambda change: change.path)
    return ConfigChanged(changes=tuple(sink), settings=new)


class ConfigManager:
    """Owns the current :class:`Settings`, the file behind them, and the watcher.

    Thread safety: the settings object is replaced, never mutated, and the swap
    happens under an :class:`~threading.RLock`. Readers get a consistent snapshot
    without holding anything.

    Listeners run **outside** the lock, on the watcher thread — a listener that
    touches Qt must marshal to the UI thread itself, and a slow listener delays
    the other listeners but never blocks a reader.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path if path is not None else get_paths().config_file
        self._lock = threading.RLock()
        self._settings = Settings()
        self._listeners: list[ConfigListener] = []
        self._dropped: tuple[str, ...] = ()
        self._pending: set[RestartScope] = set()
        self._fingerprint: tuple[int, int] | None = None
        self._digest: str = ""
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self._loaded = False

    @property
    def path(self) -> Path:
        """The settings file this manager reads and writes."""
        return self._path

    @property
    def settings(self) -> Settings:
        """Current settings. Safe to call from any thread."""
        with self._lock:
            return self._settings

    @property
    def dropped_fields(self) -> tuple[str, ...]:
        """Fields the last load had to reset, for the "часть настроек сброшена" banner."""
        with self._lock:
            return self._dropped

    @property
    def pending_restarts(self) -> frozenset[RestartScope]:
        """Workers waiting to be restarted for an already saved change."""
        with self._lock:
            return frozenset(self._pending)

    def acknowledge_restart(self, scope: RestartScope) -> None:
        """Clear ``scope`` once the supervisor has actually restarted that worker."""
        with self._lock:
            self._pending.discard(scope)

    def load(self) -> Settings:
        """Read the file, creating it on first run and rescuing it if broken.

        A file that cannot be parsed is moved to ``config.toml.broken`` and
        replaced with defaults, so a bad edit costs the user their settings but
        never the ability to start Ayris.
        """
        with self._lock:
            try:
                settings, dropped = load_settings(self._path)
            except ConfigError as exc:
                self._quarantine(exc)
                settings, dropped = Settings(), ("*",)

            self._settings = settings
            self._dropped = dropped
            self._loaded = True

            if dropped:
                _log.warning("Reset invalid settings: %s", ", ".join(dropped))
            if not self._path.is_file():
                _log.info("Creating a settings file at %s", self._path)
                save_settings(settings, self._path)
            self._remember_state()
            return settings

    def _quarantine(self, exc: ConfigError) -> None:
        """Move an unparsable file aside so the next save starts from a clean one.

        The rescued copy keeps the user's edits, which is the only way to get
        them back after a typo — Ayris itself never reads it again.
        """
        broken = self._path.with_name(f"{self._path.name}.broken")
        _log.error("Settings file is broken (%s); saving it as %s", exc.technical, broken.name)
        try:
            broken.unlink(missing_ok=True)
            self._path.replace(broken)
        except OSError as move_error:
            _log.error("Could not move the broken settings file aside: %s", move_error)

    def reload(self) -> ConfigChanged | None:
        """Re-read the file and apply what changed.

        Returns:
            The diff, or ``None`` if nothing changed or the file is momentarily
            unreadable. Unlike :meth:`load`, a broken file during a reload is not
            quarantined — the user is probably mid-edit, so the current settings
            are kept and the next poll tries again.
        """
        with self._lock:
            try:
                settings, dropped = load_settings(self._path)
            except ConfigError as exc:
                _log.warning("Ignoring an unreadable settings file: %s", exc.technical)
                self._remember_state()
                return None

            previous = self._settings
            change = diff_settings(previous, settings)
            self._remember_state()
            if not change:
                return None

            self._settings = settings
            self._dropped = dropped
            self._pending.update(change.restart_scopes)

        _log.info("Settings reloaded: %s", change.summary())
        if dropped:
            _log.warning("Reset invalid settings: %s", ", ".join(dropped))
        self._notify(change)
        return change

    def save(self, settings: Settings | None = None) -> None:
        """Persist ``settings`` (or the current ones) without firing a reload.

        Raises:
            ConfigError: The file could not be written.
        """
        with self._lock:
            target = settings if settings is not None else self._settings
            self._settings = target
            save_settings(target, self._path)
            self._remember_state()

    def apply(self, values: Mapping[str, Any]) -> ConfigChanged:
        """Change fields by dotted path, validate, save, and notify.

        This is what the settings window calls on «Применить»::

            manager.apply({"voice.tts.speed": 1.2, "overlay.opacity": 0.8})

        Raises:
            ConfigError: The result would not validate. Nothing is saved and the
                current settings are untouched.

        Returns:
            The diff that was applied. Empty if the values matched what was
            already there.
        """
        with self._lock:
            payload = dump_settings(self._settings)
            for dotted, value in values.items():
                _assign(payload, dotted.split("."), value)
            try:
                candidate = Settings.model_validate(payload)
            except ValidationError as exc:
                raise ConfigError(
                    f"rejected settings update: {exc.error_count()} invalid field(s)",
                    user_message=_describe_validation(exc),
                ) from exc

            previous = self._settings
            change = diff_settings(previous, candidate)
            if not change:
                return change

            self._settings = candidate
            self._pending.update(change.restart_scopes)
            save_settings(candidate, self._path)
            self._remember_state()

        _log.info("Settings changed: %s", change.summary())
        self._notify(change)
        return change

    def subscribe(self, listener: ConfigListener) -> Callable[[], None]:
        """Register ``listener``.

        Returns:
            A callable that unsubscribes. Long-lived objects should keep it and
            call it on teardown, or they will be called after they are closed.
        """
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _notify(self, change: ConfigChanged) -> None:
        """Call every listener outside the lock, surviving a listener that raises."""
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(change)
            except Exception:
                _log.exception("A settings listener failed; the rest still run")

    def _remember_state(self) -> None:
        """Record the file's fingerprint so the watcher can tell a real edit apart.

        Called after every read *and* every write, including failed reads, so a
        file that stays broken is reported once rather than every poll.
        """
        self._fingerprint = _fingerprint(self._path)
        self._digest = _digest(self._path)

    def _changed_on_disk(self) -> bool:
        """Whether the file differs from what this manager last read or wrote.

        Cheap ``mtime``/``size`` check first, hash only when that fires: editors
        rewrite a file byte-for-byte often enough that hashing every poll would
        be wasteful, and comparing only mtime would fire on every touch.
        """
        current = _fingerprint(self._path)
        if current == self._fingerprint:
            return False
        return _digest(self._path) != self._digest

    def start_watching(self, interval: float = _DEFAULT_POLL_INTERVAL) -> None:
        """Watch the file for external edits.

        Polling, not ``QFileSystemWatcher``: ``ayris.core`` must not import Qt,
        and worker processes need the same mechanism without an event loop. The
        thread is a daemon, so it never keeps a dying process alive.
        """
        with self._lock:
            if self._watcher is not None:
                return
            self._stop.clear()
            self._watcher = threading.Thread(
                target=self._watch_loop,
                args=(interval,),
                name="ayris-config-watch",
                daemon=True,
            )
            self._watcher.start()
        _log.debug("Watching %s for changes", self._path)

    def stop_watching(self, timeout: float = 2.0) -> None:
        """Stop the watcher thread and wait for it to finish."""
        with self._lock:
            watcher = self._watcher
            self._watcher = None
        if watcher is None:
            return
        self._stop.set()
        watcher.join(timeout)

    def _watch_loop(self, interval: float) -> None:
        """Poll for changes, waiting for the file to settle before reading it.

        An editor writes in several steps — truncate, write, flush — and reading
        between them yields a half-file. Requiring the fingerprint to hold still
        for a couple of polls avoids a spurious "broken file" warning on what is
        really a normal save.
        """
        stable = 0
        while not self._stop.wait(interval):
            try:
                if not self._changed_on_disk():
                    stable = 0
                    continue
                stable += 1
                if stable < _SETTLE_POLLS:
                    continue
                stable = 0
                self.reload()
            except Exception:
                _log.exception("The settings watcher hit an error; it keeps running")

    def __repr__(self) -> str:
        return f"ConfigManager(path={str(self._path)!r}, loaded={self._loaded})"


def _assign(payload: MutableMapping[str, Any], parts: list[str], value: Any) -> None:
    """Set a dotted path inside a dumped settings tree, creating tables as needed."""
    node: MutableMapping[str, Any] = payload
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def _describe_validation(exc: ValidationError) -> str:
    """Turn a pydantic error into something worth showing a user."""
    lines: list[str] = []
    for error in exc.errors()[:5]:
        where = ".".join(str(part) for part in error["loc"])
        lines.append(f"• {where}: {error['msg']}")
    if exc.error_count() > 5:
        lines.append(f"…и ещё {exc.error_count() - 5}")
    return "Настройки не сохранены — неверные значения:\n" + "\n".join(lines)


def _fingerprint(path: Path) -> tuple[int, int] | None:
    """Modification time and size, or ``None`` if the file is gone."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def _digest(path: Path) -> str:
    """SHA-256 of the file, or an empty string if it cannot be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


_manager: ConfigManager | None = None
_manager_lock: Final = threading.RLock()


def init_config(path: Path | None = None, *, watch: bool = False) -> ConfigManager:
    """Create the process-wide manager and load the file.

    Called once from the application entry point, before anything reads a
    setting. ``watch=True`` also starts the file watcher — the main process wants
    it, worker processes usually do not.
    """
    global _manager
    with _manager_lock:
        manager = ConfigManager(path)
        manager.load()
        if watch:
            manager.start_watching()
        _manager = manager
        return manager


def get_config_manager() -> ConfigManager:
    """The process-wide manager, initialised on first use.

    Falling back to a lazy load keeps tests and one-off scripts from having to
    call :func:`init_config` themselves.
    """
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = ConfigManager()
            _manager.load()
        return _manager


def get_settings() -> Settings:
    """Current settings. The usual way for the rest of Ayris to read a value."""
    return get_config_manager().settings


def reset_config_manager() -> None:
    """Drop the process-wide manager, stopping its watcher. Test helper."""
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.stop_watching()
        _manager = None
    restart_scope.cache_clear()
