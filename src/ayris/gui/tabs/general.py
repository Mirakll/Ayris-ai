"""Tab «Общие»: language, theme, startup, and the performance half.

The page is two halves of one settings section. The top is interface and window
behaviour — language (Russian only, locked), theme (applied the instant it
changes), autostart, start-minimized and close-to-tray. The bottom is
performance — process and audio priority, the memory cap, the thread pools, eco
mode, and a live RAM/CPU panel.

Three things here are not the usual bind-and-forget:

*Theme applies live.* Nothing in the app listens for a theme *config* change, so
the combo drives :meth:`ThemeManager.set_mode` itself the moment it changes, and
also re-applies on an external edit or a section reset.

*Some fields need a worker restart.* Audio priority and the STT/TTS/LLM pools are
tagged with a :class:`RestartScope` in the schema. When one is changed and saved,
:attr:`ConfigManager.pending_restarts` gains that scope; the card grows a plaque
with a restart button that recycles exactly that worker through the supervisor
registered in :mod:`ayris.gui.widgets.resource_monitor`. Process priority, by
contrast, is applied to the running process in place with no restart.

*«Реальное время» is guarded.* It can starve the system on a weak machine, so the
audio-priority combo asks for confirmation before accepting it and reverts if the
user declines.
"""

from __future__ import annotations

import sys
from typing import Final, get_args

from PySide6.QtCore import QSignalBlocker, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import (
    RAM_LIMIT_CHOICES,
    ConfigManager,
    PerformanceConfig,
    RestartScope,
)
from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.events import EventBus
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    ConfirmDialog,
    ResourceMonitor,
    ResourceRow,
    Sampler,
    SettingCard,
    SliderField,
    ThemedComboBox,
    ToggleSwitch,
    active_worker_control,
)
from ayris.utils.logger import get_logger
from ayris.utils.process_priority import PRIORITY_LABELS, apply_priority

__all__ = ["GeneralTab"]

_log = get_logger(__name__)

#: Russian names for the worker kinds shown in the resource panel.
_WORKER_LABELS: Final[dict[str, str]] = {
    "audio": "Захват звука",
    "stt": "Распознавание речи",
    "tts": "Синтез речи",
    "llm": "Языковая модель",
}

#: Thread-pool sliders: config path → (title, description, minimum, maximum).
_POOLS: Final[tuple[tuple[str, str, str, int, int, RestartScope], ...]] = (
    (
        "performance.stt_threads",
        "Потоки распознавания",
        "Больше — быстрее на длинных фразах",
        1,
        4,
        RestartScope.STT,
    ),
    (
        "performance.tts_threads",
        "Потоки синтеза",
        "Параллельная озвучка нескольких ответов",
        1,
        2,
        RestartScope.TTS,
    ),
    (
        "performance.llm_threads",
        "Потоки языковой модели",
        "Параллельные запросы к модели",
        1,
        2,
        RestartScope.LLM,
    ),
    (
        "performance.macro_threads",
        "Потоки макросов",
        "Сколько макросов выполняется одновременно",
        1,
        16,
        RestartScope.NONE,
    ),
)


def _priority_options(field: str) -> list[tuple[str, str]]:
    """Combo entries (value, label) for a priority field, straight from the schema."""
    annotation = PerformanceConfig.model_fields[field].annotation
    options: list[tuple[str, str]] = []
    for value in get_args(annotation):
        key = str(value)
        options.append((key, PRIORITY_LABELS.get(key, key)))
    return options


def _ram_options() -> list[tuple[int, str]]:
    """Combo entries for the memory cap, matching :data:`RAM_LIMIT_CHOICES`."""
    options: list[tuple[int, str]] = []
    for choice in RAM_LIMIT_CHOICES:
        options.append((choice, "Без ограничения" if choice == 0 else f"{choice // 1024} ГБ"))
    return options


class _RestartBar(QFrame):
    """The «Применится после перезапуска воркера» plaque with a restart button."""

    restart_requested = Signal()

    def __init__(self, scope_hint: str, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("notice", True)
        self.setProperty("status", "warning")
        self._layout = QHBoxLayout(self)
        self.label = QLabel("Применится после перезапуска воркера")
        self.label.setWordWrap(True)
        self.label.setToolTip(scope_hint)
        self._layout.addWidget(self.label, 1)
        self.button = QPushButton("Перезапустить")
        self.button.clicked.connect(self.restart_requested)
        self._layout.addWidget(self.button)
        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self.hide()

    def set_pending(self, pending: bool, *, has_control: bool) -> None:
        self.button.setEnabled(has_control)
        self.button.setToolTip("" if has_control else "Воркеры сейчас не запущены")
        self.setVisible(pending)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_sm")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(pad)
        self.setMinimumHeight(self._theme.metric("notice_min_height"))


class GeneralTab(SettingsTab):
    """The «Общие» settings page (task 48)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        sampler: Sampler | None = None,
    ) -> None:
        super().__init__("general", "Общие", ("general", "performance"), manager, theme, bus)
        self._sampler = sampler
        self._restart_bars: dict[RestartScope, _RestartBar] = {}
        self._audio_prev: str = manager.settings.performance.audio_priority

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._build_interface()
        self._build_startup()
        self._build_performance()
        self._build_eco()
        self._build_resources()
        self._content.addStretch(1)

    # -- section builders ---------------------------------------------------

    def _add_header(self, text: str) -> None:
        header = QLabel(text)
        header.setProperty("role", "h2")
        self._content.addWidget(header)

    def _add_caption(self, text: str) -> QLabel:
        caption = QLabel(text)
        caption.setProperty("role", "muted")
        caption.setWordWrap(True)
        self._content.addWidget(caption)
        return caption

    def _build_interface(self) -> None:
        self._add_header("Интерфейс")

        self._language_combo = ThemedComboBox()
        self._language_combo.addItem("Русский", "ru")
        self._language_combo.setEnabled(False)
        self.bind_combo(self._language_combo, "general.language", "Язык интерфейса")
        self._content.addWidget(
            SettingCard(
                "Язык",
                "Язык интерфейса и распознавания. Другие языки пока не поддерживаются.",
                self._language_combo,
                self._theme,
            )
        )

        self._theme_combo = ThemedComboBox()
        for value, text in (
            ("dark_purple", "Тёмная фиолетовая"),
            ("light", "Светлая"),
            ("system", "Системная"),
        ):
            self._theme_combo.addItem(text, value)
        self.bind_combo(self._theme_combo, "general.theme", "Тема оформления")
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        self._content.addWidget(
            SettingCard(
                "Тема",
                "Оформление окна и оверлея. Применяется сразу.",
                self._theme_combo,
                self._theme,
            )
        )

    def _build_startup(self) -> None:
        self._add_header("Запуск и закрытие")

        self._autostart_toggle = ToggleSwitch(self._theme, label="Автозапуск")
        self.bind_toggle(self._autostart_toggle, "general.autostart", "Автозапуск с Windows")
        self._content.addWidget(
            SettingCard(
                "Запускать вместе с Windows",
                "Добавляет Ayris в автозагрузку текущего пользователя.",
                self._autostart_toggle,
                self._theme,
            )
        )

        start_min = ToggleSwitch(self._theme, label="Старт свёрнутым")
        self.bind_toggle(start_min, "general.start_minimized", "Стартовать свёрнутым")
        self._content.addWidget(
            SettingCard(
                "Стартовать свёрнутым",
                "При запуске окно не показывается — Ayris ждёт в трее.",
                start_min,
                self._theme,
            )
        )

        minimize = ToggleSwitch(self._theme, label="Свернуть в трей")
        self.bind_toggle(minimize, "general.minimize_to_tray", "Сворачивать в трей")
        self._content.addWidget(
            SettingCard(
                "Кнопка «свернуть» — в трей",
                "Свёрнутое окно убирается в трей, а не на панель задач.",
                minimize,
                self._theme,
            )
        )

        close_tray = ToggleSwitch(self._theme, label="Закрытие в трей")
        self.bind_toggle(close_tray, "general.close_to_tray", "Закрытие сворачивает в трей")
        self._content.addWidget(
            SettingCard(
                "Кнопка «закрыть» — в трей",
                "Закрытие окна прячет Ayris в трей вместо выхода из программы.",
                close_tray,
                self._theme,
            )
        )

    def _build_performance(self) -> None:
        self._add_header("Производительность")

        self._process_combo = ThemedComboBox()
        for value, label in _priority_options("process_priority"):
            self._process_combo.addItem(label, value)
        self.bind_combo(self._process_combo, "performance.process_priority", "Приоритет помощника")
        self._process_combo.currentIndexChanged.connect(self._on_process_priority_changed)
        self._content.addWidget(
            SettingCard(
                "Приоритет помощника",
                "Приоритет главного процесса. Применяется сразу; фоновые воркеры "
                "подхватят его при следующем перезапуске.",
                self._process_combo,
                self._theme,
            )
        )

        self._audio_combo = ThemedComboBox()
        for value, label in _priority_options("audio_priority"):
            self._audio_combo.addItem(label, value)
        self.bind_combo(self._audio_combo, "performance.audio_priority", "Приоритет захвата звука")
        self._audio_combo.currentIndexChanged.connect(self._on_audio_priority_changed)
        self._card_with_restart(
            "Приоритет захвата звука",
            "Ниже «Высокого» возможны пропуски звука. «Реальное время» — с осторожностью.",
            self._audio_combo,
            RestartScope.AUDIO,
        )

        self._ram_combo = ThemedComboBox()
        for ram_value, ram_label in _ram_options():
            self._ram_combo.addItem(ram_label, ram_value)
        self.bind_combo(self._ram_combo, "performance.ram_limit_mb", "Лимит памяти моделей")
        self._ram_combo.currentIndexChanged.connect(self._on_ram_changed)
        self._content.addWidget(
            SettingCard(
                "Лимит памяти для моделей",
                "Мягкий предел, выше которого локальные модели не загружаются.",
                self._ram_combo,
                self._theme,
            )
        )

        for path, title, desc, minimum, maximum, scope in _POOLS:
            field = SliderField(
                self._theme,
                minimum=minimum,
                maximum=maximum,
                value=minimum,
                label=title,
            )
            self._bind_slider_field(field, path, title)
            if scope is RestartScope.NONE:
                self._content.addWidget(SettingCard(title, desc, field, self._theme))
            else:
                self._card_with_restart(title, desc, field, scope)

    def _build_eco(self) -> None:
        self._add_header("Экономный режим")
        self._eco_toggle = ToggleSwitch(self._theme, label="Экономный режим")
        self.bind_toggle(self._eco_toggle, "performance.eco_mode", "Экономный режим")
        self._content.addWidget(
            SettingCard(
                "Экономить память",
                "Не держать в памяти локальные модели, которые сейчас не на основном пути.",
                self._eco_toggle,
                self._theme,
            )
        )
        self._eco_note = self._add_caption("")

    def _build_resources(self) -> None:
        self._add_header("Ресурсы")
        self._resources = ResourceMonitor(
            self._theme,
            sampler=self._sampler,
            sources=self._worker_rows,
            ram_limit_mb=self._manager.settings.performance.ram_limit_mb,
        )
        self._content.addWidget(self._resources)

    # -- helpers ------------------------------------------------------------

    def _card_with_restart(
        self, title: str, desc: str, control: QWidget, scope: RestartScope
    ) -> None:
        self._content.addWidget(SettingCard(title, desc, control, self._theme))
        bar = _RestartBar(scope.label, self._theme)
        bar.restart_requested.connect(lambda: self._restart_scope(scope))
        self._restart_bars[scope] = bar
        self._content.addWidget(bar)

    def _bind_slider_field(self, field: SliderField, path: str, label: str) -> None:
        self._bind(field, path, label, field.value, field.setValue, field.value_changed)

    def _worker_rows(self) -> list[ResourceRow]:
        control = active_worker_control()
        if control is None:
            return []
        rows: list[ResourceRow] = []
        for summary in control.status():
            name = _WORKER_LABELS.get(summary.kind, summary.kind)
            rows.append(
                ResourceRow(
                    key=f"worker:{summary.name}",
                    label=f"{name} ({summary.name})",
                    pid=summary.pid if summary.alive else None,
                    note="" if summary.alive else summary.status.label,
                )
            )
        return rows

    def _eco_explanation(self) -> str:
        settings = self._manager.settings
        deferred: list[str] = []
        if settings.voice.stt.mode == "online":
            deferred.append("распознавание речи (офлайн-движок остаётся запасным)")
        if settings.voice.tts.engine in {"yandex", "elevenlabs"}:
            deferred.append("синтез речи (локальный голос остаётся запасным)")
        ai = settings.ai
        needs_llm = ai.fallback_to_llm or ai.llm_understanding or ai.free_chat
        if needs_llm and not (ai.llm_understanding or ai.free_chat):
            deferred.append("языковая модель (только на случай нераспознанной команды)")

        if not settings.performance.eco_mode:
            return "Выключено: все нужные локальные модели загружаются при старте."
        if deferred:
            return (
                "При текущих движках не будут запущены сразу: "
                + "; ".join(deferred)
                + ". Они поднимутся при первом обращении — первый вызов будет с задержкой."
            )
        return (
            "При текущих движках всё уже на основном пути: откладывать нечего, "
            "но простаивающие модели будут выгружаться из памяти."
        )

    # -- live application ---------------------------------------------------

    def load_from_config(self) -> None:
        super().load_from_config()
        self._audio_prev = self._manager.settings.performance.audio_priority
        self._sync_autostart_from_registry()
        self._refresh_dynamic()

    def _sync_autostart_from_registry(self) -> None:
        """Show the real registry state — a third-party tool may have removed it."""
        if sys.platform != "win32":
            return
        from ayris.utils.autostart import is_enabled

        blocker = QSignalBlocker(self._autostart_toggle)
        self._autostart_toggle.setChecked(is_enabled())
        del blocker

    def _on_theme_changed(self) -> None:
        value = self._theme_combo.currentData()
        if isinstance(value, str):
            self._apply_theme(value)

    def _apply_theme(self, value: str) -> None:
        if value == "light":
            self._theme.set_mode("light")
        elif value == "system":
            self._theme.set_mode("system")
        else:
            self._theme.set_mode("dark")

    def _on_process_priority_changed(self) -> None:
        value = self._process_combo.currentData()
        if isinstance(value, str):
            apply_priority(value)

    def _on_audio_priority_changed(self) -> None:
        value = self._audio_combo.currentData()
        if value == "realtime" and value != self._audio_prev and not self._confirm_realtime():
            blocker = QSignalBlocker(self._audio_combo)
            index = self._audio_combo.findData(self._audio_prev)
            if index >= 0:
                self._audio_combo.setCurrentIndex(index)
            del blocker
            self._pending.pop("performance.audio_priority", None)
            self.dirty_label.setVisible(bool(self._pending))
            return
        if isinstance(value, str):
            self._audio_prev = value

    def _confirm_realtime(self) -> bool:
        dialog = ConfirmDialog(
            "Приоритет «Реальное время»?",
            "На слабой машине это может подвесить систему: захват звука получит приоритет "
            "выше системных служб. Обычно достаточно «Высокого».",
            self._theme,
            confirm_text="Всё равно включить",
            dangerous=True,
            parent=self,
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    def _on_ram_changed(self) -> None:
        value = self._ram_combo.currentData()
        if isinstance(value, int):
            self._resources.set_ram_limit(value)

    def _restart_scope(self, scope: RestartScope) -> None:
        control = active_worker_control()
        if control is None:
            return
        try:
            control.restart_scope(scope, "перезапуск из настроек")
        except Exception:
            _log.exception("не удалось перезапустить воркеры области %s", scope.value)
            return
        self._manager.acknowledge_restart(scope)
        self._refresh_dynamic()

    def _refresh_dynamic(self) -> None:
        pending = self._manager.pending_restarts
        has_control = active_worker_control() is not None
        for scope, bar in self._restart_bars.items():
            bar.set_pending(scope in pending, has_control=has_control)
        self._eco_note.setText(self._eco_explanation())

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        super()._on_config_changed(diff)
        paths = set(diff.paths)
        if "general.theme" in paths:
            self._apply_theme(diff.settings.general.theme)
        if "performance.process_priority" in paths:
            apply_priority(diff.settings.performance.process_priority)
        if "performance.ram_limit_mb" in paths:
            self._resources.set_ram_limit(diff.settings.performance.ram_limit_mb)
        self._refresh_dynamic()


register_tab("general", GeneralTab)
