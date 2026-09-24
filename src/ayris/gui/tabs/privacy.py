"""Task 59: the «Приватность» settings page.

Brings together the privacy controls whose engine already lives in the core:
telemetry (off, and not implemented — the switch says so and stays disabled),
what the history and audit trail record, per-category auto-cleanup, one-shot
deletion with counts and confirmation, an explanation of where API keys live, the
audit journal and an on-disk data overview. Every destructive button confirms with
concrete numbers first, and the full reset makes a backup before it runs.
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris.core.audit import AuditReader
from ayris.core.config import ConfigManager
from ayris.core.errors import AyrisError
from ayris.core.events import EventBus
from ayris.core.models import utc_now
from ayris.core.paths import AppPaths, get_paths
from ayris.core.repositories import (
    CleanupCategory,
    CleanupReport,
    Repositories,
)
from ayris.core.secrets import KNOWN_SLOTS, SecretsStore, get_secrets
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import (
    BusyIndicator,
    ConfirmDialog,
    InlineNotice,
    SettingCard,
    ThemedComboBox,
    ToggleSwitch,
)
from ayris.gui.widgets.audit_view import AuditView
from ayris.gui.widgets.data_inventory import DataInventory, human_size
from ayris.gui.widgets.notice import NoticeKind
from ayris.utils.logger import get_logger

__all__ = ["PrivacyServices", "PrivacyTab"]

_log = get_logger(__name__)

#: Retention presets offered by the auto-cleanup combos, label → days (0 = never).
_RETENTION: tuple[tuple[str, int], ...] = (
    ("Никогда", 0),
    ("7 дней", 7),
    ("30 дней", 30),
    ("90 дней", 90),
)


@dataclass(slots=True)
class PrivacyServices:
    """Everything the tab needs from the core, all optional so it degrades safely.

    A missing repository disables the deletion and cleanup controls rather than
    crashing the page; a test injects fakes or a temp-database ``Repositories``.
    """

    repositories: Repositories | None = None
    secrets: SecretsStore = field(default_factory=get_secrets)
    paths: AppPaths | None = None
    audit_reader: AuditReader | None = None
    open_folder: Callable[[Path], None] | None = None


def _default_repositories() -> Repositories | None:
    try:
        from ayris.core.database import get_database

        return Repositories(get_database())
    except Exception:
        _log.exception("не удалось открыть хранилище для вкладки «Приватность»")
        return None


class _DbRunner(QObject):
    """Runs a blocking database call off the UI thread, result back on it.

    Cleanup, backup and a full reset can each take a while (a big ``VACUUM`` most
    of all); routing them through here keeps the window responsive and guarantees
    only one runs at a time per runner.
    """

    finished = Signal(object)
    failed = Signal(str)

    def run(self, work: Callable[[], object]) -> None:
        threading.Thread(target=self._run, args=(work,), daemon=True).start()

    def _run(self, work: Callable[[], object]) -> None:
        try:
            result = work()
        except AyrisError as exc:
            self.failed.emit(exc.user_message)
        except Exception as exc:  # keep the UI alive whatever the store throws
            _log.exception("операция с данными на вкладке «Приватность» упала")
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)


class _TypedConfirmDialog(QDialog):
    """Confirmation that only unlocks once the user types an exact word.

    Used for the full profile reset: a plain «да/нет» is too easy to click, so the
    destructive button stays disabled until the confirmation word is typed back.
    """

    def __init__(
        self,
        title: str,
        text: str,
        word: str,
        theme: ThemeManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._word = word
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)
        heading = QLabel(title)
        heading.setProperty("role", "h2")
        body = QLabel(text)
        body.setProperty("role", "secondary")
        body.setWordWrap(True)
        self._field = QLineEdit()
        self._field.setPlaceholderText(word)
        self._field.setAccessibleName("Слово подтверждения")
        layout.addWidget(heading)
        layout.addWidget(body)
        layout.addWidget(QLabel(f"Введите «{word}», чтобы подтвердить:"))
        layout.addWidget(self._field)
        buttons = QDialogButtonBox()
        self.confirm_button = QPushButton("Сбросить всё")
        self.confirm_button.setProperty("kind", "danger")
        self.confirm_button.setEnabled(False)
        cancel = QPushButton("Отмена")
        buttons.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(self.confirm_button, QDialogButtonBox.ButtonRole.AcceptRole)
        layout.addWidget(buttons)
        self.confirm_button.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        self._field.textChanged.connect(self._validate)

    def _validate(self, text: str) -> None:
        self.confirm_button.setEnabled(text.strip() == self._word)


#: Model kinds shown when warning what a cache wipe will delete, dir → label.
_MODEL_KINDS: tuple[tuple[str, str], ...] = (
    ("stt", "распознавание"),
    ("tts", "синтез речи"),
    ("wake", "активацию"),
    ("llm", "языковые модели"),
)


def _dir_size(root: Path) -> int:
    """Total on-disk size under ``root``; ``0`` if it is missing. Metadata only."""
    total = 0
    try:
        for entry in root.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        return total
    return total


def _clear_directory(root: Path) -> int:
    """Erase everything inside ``root`` but keep ``root`` itself.

    Returns bytes freed. The folder stays so the app keeps writing there (logs,
    cache) without recreating it. A locked file — the log Ayris is writing right
    now — is skipped rather than fatal.
    """
    if not root.is_dir():
        return 0
    freed = _dir_size(root)
    for entry in root.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            _log.warning("не удалось удалить %s", entry)
    return max(0, freed - _dir_size(root))


def _as_int(value: object) -> int:
    """Coerce a background worker's result to bytes freed, defaulting to ``0``."""
    return value if isinstance(value, int) else 0


class _StatusRow(QFrame):
    """A spinner plus a status line for one background database operation.

    Cleanup, backup and a full reset report no percentage, so «прогресс» is an
    honest indeterminate spinner beside a message rather than a fake bar.
    """

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._busy = BusyIndicator(theme, active=False)
        self._busy.hide()
        self._message = QLabel("")
        self._message.setProperty("role", "secondary")
        self._message.setWordWrap(True)
        self._message.hide()
        self._layout.addWidget(self._busy)
        self._layout.addWidget(self._message, 1)

    def set_busy(self, busy: bool, text: str = "") -> None:
        self._busy.setVisible(busy)
        self._busy.setActive(busy)
        self._message.setText(text)
        self._message.setVisible(busy and bool(text))

    def show_message(self, text: str, kind: str) -> None:
        self._busy.setActive(False)
        self._busy.hide()
        self._message.setProperty("badge", kind)
        self._message.setText(text)
        style = self._message.style()
        if style is not None:
            style.unpolish(self._message)
            style.polish(self._message)
        self._message.setVisible(bool(text))


class PrivacyTab(SettingsTab):
    """The «Приватность» settings page (task 59)."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        services: PrivacyServices | None = None,
    ) -> None:
        super().__init__("privacy", "Приватность", ("privacy",), manager, theme, bus)
        self.services = services if services is not None else PrivacyServices()
        if self.services.repositories is None:
            self.services.repositories = _default_repositories()
        if self.services.paths is None:
            self.services.paths = get_paths()
        repos = self.services.repositories
        if self.services.audit_reader is None and repos is not None:
            self.services.audit_reader = AuditReader(repos.audit)

        self._runners: list[_DbRunner] = []
        self._inventory: DataInventory | None = None
        self._audit_view: AuditView | None = None
        self._secret_rows: QVBoxLayout | None = None
        self._last_cleanup: QLabel | None = None

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._status = _StatusRow(theme)
        self._build_telemetry()
        self._build_composition()
        self._build_retention()
        self._build_deletion()
        self._build_audit()
        self._build_inventory()
        self._build_secrets()
        self._content.addWidget(self._status)
        self._content.addStretch(1)
        self._update_last_cleanup_label()

    # -- section chrome ---------------------------------------------------

    def _heading(self, text: str) -> None:
        label = QLabel(text)
        label.setProperty("role", "h2")
        self._content.addWidget(label)

    def _note(self, text: str, kind: NoticeKind = "info") -> None:
        self._content.addWidget(InlineNotice(text, self._theme, kind=kind))

    # -- 1. telemetry -----------------------------------------------------

    def _build_telemetry(self) -> None:
        self._heading("Телеметрия")
        toggle = ToggleSwitch(self._theme, checked=False, label="Телеметрия")
        toggle.setEnabled(False)
        self._content.addWidget(
            SettingCard(
                "Сбор данных выключен",
                "Ayris не собирает статистику и не отправляет её. Функции сбора в приложении "
                "нет вообще — включать нечего, поэтому переключатель недоступен.",
                toggle,
                self._theme,
            )
        )
        self._note(
            "Проверить можно самому: запустите сетевой сниффер (Wireshark или монитор во "
            "вкладке «Логи») и убедитесь, что без вашего действия Ayris не открывает ни одного "
            "исходящего соединения.",
        )

    # -- 6. what history records -----------------------------------------

    def _build_composition(self) -> None:
        self._heading("Что записывать")
        transcript = ToggleSwitch(self._theme, label="Записывать распознанный текст")
        self.bind_toggle(transcript, "privacy.record_transcript", "Записывать распознанный текст")
        self._content.addWidget(
            SettingCard(
                "Распознанный текст в истории",
                "Выключено — запись о команде остаётся, но сама фраза в «history» не сохраняется. "
                "Команда «повтори» и трассы во вкладке «Логи» останутся без текста.",
                transcript,
                self._theme,
            )
        )
        clipboard = ToggleSwitch(self._theme, label="Записывать содержимое буфера обмена")
        self.bind_toggle(clipboard, "actions.clipboard.monitor", "Записывать буфер обмена")
        self._content.addWidget(
            SettingCard(
                "История буфера обмена",
                "Выключено — копирования перестают попадать в историю буфера немедленно.",
                clipboard,
                self._theme,
            )
        )
        params = ToggleSwitch(self._theme, label="Записывать параметры действий в аудит")
        self.bind_toggle(params, "privacy.audit_params", "Параметры действий в аудите")
        self._content.addWidget(
            SettingCard(
                "Параметры действий в аудите",
                "Выключено — запись в журнале аудита остаётся, но поле параметров пустое.",
                params,
                self._theme,
            )
        )
        self._note(
            "Переключатели действуют только на новые записи. Уже сохранённое удаляется кнопками "
            "ниже в разделе «Удаление данных».",
            "warning",
        )

    # -- 3. auto-cleanup retention ---------------------------------------

    def _build_retention(self) -> None:
        self._heading("Автоочистка по сроку")
        self._retention_row(
            "История команд",
            "Удалять записи истории старше выбранного срока.",
            "privacy.retention_history_days",
        )
        self._retention_row(
            "История буфера обмена",
            "Незакреплённые записи буфера старше срока удаляются автоматически.",
            "privacy.retention_clipboard_days",
        )
        self._retention_row(
            "Журнал аудита",
            "Записи аудита старше срока удаляются. По умолчанию аудит не чистится.",
            "privacy.retention_audit_days",
        )
        row = QHBoxLayout()
        self._last_cleanup = QLabel("")
        self._last_cleanup.setProperty("role", "secondary")
        self._last_cleanup.setWordWrap(True)
        run_now = QPushButton("Выполнить очистку сейчас")
        run_now.clicked.connect(self._run_cleanup_now)
        row.addWidget(self._last_cleanup, 1)
        row.addWidget(run_now)
        self._content.addLayout(row)
        self._note(
            "Очистка выполняется при старте и раз в сутки. Здесь можно запустить её вручную.",
        )

    def _retention_row(self, title: str, description: str, path: str) -> None:
        combo = ThemedComboBox()
        for label, days in _RETENTION:
            combo.addItem(label, days)
        self._content.addWidget(SettingCard(title, description, combo, self._theme))
        self.bind_combo(combo, path, title)

    def _update_last_cleanup_label(self) -> None:
        if self._last_cleanup is None:
            return
        privacy = self._manager.settings.privacy
        when = privacy.last_cleanup_at
        if not when:
            self._last_cleanup.setText("Автоочистка ещё не выполнялась.")
            return
        try:
            stamp = datetime.fromisoformat(when).strftime("%Y-%m-%d %H:%M")
        except ValueError:
            stamp = when
        self._last_cleanup.setText(
            f"Последняя очистка: {stamp}, удалено записей: {privacy.last_cleanup_removed}."
        )

    # -- background plumbing ---------------------------------------------

    def _run_bg(
        self,
        work: Callable[[], object],
        busy_text: str,
        done: Callable[[object], str],
    ) -> None:
        runner = _DbRunner(self)
        self._runners.append(runner)
        runner.finished.connect(lambda result, d=done: self._finish_bg(d, result))
        runner.failed.connect(self._on_bg_failed)
        self._status.set_busy(True, busy_text)
        runner.run(work)

    def _finish_bg(self, done: Callable[[object], str], result: object) -> None:
        try:
            message = done(result)
        except Exception:  # a failed refresh must not swallow the success
            _log.exception("постобработка операции с данными упала")
            message = "Готово."
        self._status.show_message(message, "success")
        self._refresh_after_change()

    def _on_bg_failed(self, message: str) -> None:
        self._status.show_message(f"Не удалось выполнить: {message}", "error")

    def _refresh_after_change(self) -> None:
        self._update_last_cleanup_label()
        if self._inventory is not None:
            self._inventory.reload()
        if self._audit_view is not None:
            self._audit_view.refresh()

    def _run_cleanup_now(self) -> None:
        repos = self.services.repositories
        if repos is None:
            self._status.show_message("Хранилище недоступно.", "error")
            return
        privacy = self._manager.settings.privacy
        history_days = privacy.retention_history_days
        history_limit = privacy.history_limit
        clipboard_days = privacy.retention_clipboard_days
        audit_days = privacy.retention_audit_days

        def work() -> object:
            return repos.maintenance.apply_retention(
                history_days=history_days,
                history_limit=history_limit,
                clipboard_days=clipboard_days,
                audit_days=audit_days,
            )

        def done(result: object) -> str:
            removed = int(result) if isinstance(result, int) else 0
            self._manager.apply(
                {
                    "privacy.last_cleanup_at": utc_now().isoformat(),
                    "privacy.last_cleanup_removed": removed,
                }
            )
            return f"Очистка выполнена, удалено записей: {removed}."

        self._run_bg(work, "Выполняется очистка…", done)

    # -- 2. deletion buttons ---------------------------------------------

    def _build_deletion(self) -> None:
        self._heading("Удаление данных")
        self._delete_card(
            "История команд",
            "Удаляет все записи распознанных фраз и результатов из «history».",
            "Удалить историю",
            self._delete_history,
        )
        self._delete_card(
            "История буфера обмена",
            "Удаляет всю историю буфера обмена, включая закреплённые записи.",
            "Очистить буфер",
            self._delete_clipboard,
        )
        self._delete_card(
            "Кэш моделей",
            "Удаляет скачанные модели распознавания, синтеза речи, активации и языковые. "
            "Это гигабайты, и после удаления их придётся скачать заново.",
            "Очистить кэш моделей",
            self._delete_model_cache,
        )
        self._delete_card(
            "Логи",
            "Удаляет файлы журналов из папки logs. Текущий занятый файл может остаться.",
            "Удалить логи",
            self._delete_logs,
        )
        self._delete_card(
            "Полный сброс профиля",
            "Удаляет историю, аудит, буфер, переменные, таймеры и версии команд. Перед "
            "сбросом создаётся резервная копия базы. Сами команды остаются.",
            "Сбросить профиль…",
            self._full_reset,
        )

    def _delete_card(
        self, title: str, description: str, button_text: str, handler: Callable[[], None]
    ) -> None:
        button = QPushButton(button_text)
        button.setProperty("kind", "danger")
        button.clicked.connect(handler)
        self._content.addWidget(SettingCard(title, description, button, self._theme))

    def _confirm(self, title: str, text: str, confirm_text: str) -> bool:
        dialog = ConfirmDialog(
            title, text, self._theme, confirm_text=confirm_text, dangerous=True, parent=self
        )
        return dialog.exec() == QDialog.DialogCode.Accepted

    @staticmethod
    def _report_message(prefix: str, result: object) -> str:
        if isinstance(result, CleanupReport):
            return (
                f"{prefix}. Удалено записей: {result.total}, "
                f"освобождено {human_size(result.freed_bytes)}."
            )
        return f"{prefix}."

    def _repos_or_warn(self) -> Repositories | None:
        repos = self.services.repositories
        if repos is None:
            self._status.show_message("Хранилище недоступно.", "error")
        return repos

    def _delete_history(self) -> None:
        repos = self._repos_or_warn()
        if repos is None:
            return
        count = repos.history.count()
        if not self._confirm(
            "Удалить историю команд?",
            f"Будет удалено записей: {count}. Действие необратимо.",
            "Удалить",
        ):
            return
        self._run_bg(
            lambda: repos.maintenance.clear([CleanupCategory.HISTORY]),
            "Удаление истории…",
            lambda result: self._report_message("История команд удалена", result),
        )

    def _delete_clipboard(self) -> None:
        repos = self._repos_or_warn()
        if repos is None:
            return
        count = repos.clipboard.count()
        if not self._confirm(
            "Очистить историю буфера обмена?",
            f"Будет удалено записей: {count} (включая закреплённые). Действие необратимо.",
            "Очистить",
        ):
            return
        self._run_bg(
            lambda: repos.maintenance.clear([CleanupCategory.CLIPBOARD]),
            "Очистка буфера обмена…",
            lambda result: self._report_message("История буфера очищена", result),
        )

    def _delete_model_cache(self) -> None:
        paths = self.services.paths
        if paths is None:
            self._status.show_message("Пути профиля недоступны.", "error")
            return
        root = paths.models_dir
        size = _dir_size(root)
        kinds = ", ".join(label for _key, label in _MODEL_KINDS)
        if not self._confirm(
            "Очистить кэш моделей?",
            f"Будут удалены модели ({kinds}) — {human_size(size)}. После удаления их придётся "
            "скачать заново. Действие необратимо.",
            "Очистить",
        ):
            return
        self._run_bg(
            lambda: _clear_directory(root),
            "Очистка кэша моделей…",
            lambda result: f"Кэш моделей очищен, освобождено {human_size(_as_int(result))}.",
        )

    def _delete_logs(self) -> None:
        paths = self.services.paths
        if paths is None:
            self._status.show_message("Пути профиля недоступны.", "error")
            return
        root = paths.logs_dir
        size = _dir_size(root)
        if not self._confirm(
            "Удалить логи?",
            f"Будет удалено {human_size(size)} журналов. Текущий занятый файл может остаться. "
            "Действие необратимо.",
            "Удалить",
        ):
            return
        self._run_bg(
            lambda: _clear_directory(root),
            "Удаление логов…",
            lambda result: f"Логи удалены, освобождено {human_size(_as_int(result))}.",
        )

    def _full_reset(self) -> None:
        repos = self._repos_or_warn()
        if repos is None:
            return
        stats = repos.maintenance.statistics()
        total = sum(
            int(stats.get(table, 0))
            for table in (
                "history",
                "audit",
                "clipboard_history",
                "variables",
                "timers",
                "command_versions",
            )
        )
        dialog = _TypedConfirmDialog(
            "Полный сброс профиля",
            f"Будет удалено записей: {total} (история, аудит, буфер, переменные, таймеры, версии "
            "команд). Команды сохранятся. Перед сбросом создаётся резервная копия базы. "
            "Действие необратимо.",
            "СБРОС",
            self._theme,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        def work() -> object:
            backup = repos.maintenance.backup()
            report = repos.maintenance.clear_all()
            return (backup, report)

        def done(result: object) -> str:
            if isinstance(result, tuple) and len(result) == 2:
                backup, report = result
                name = backup.name if isinstance(backup, Path) else str(backup)
                removed = report.total if isinstance(report, CleanupReport) else 0
                return f"Профиль сброшен. Резервная копия: {name}. Удалено записей: {removed}."
            return "Профиль сброшен."

        self._run_bg(work, "Сброс профиля…", done)

    # -- 4. audit journal -------------------------------------------------

    def _build_audit(self) -> None:
        self._heading("Журнал аудита")
        reader = self.services.audit_reader
        if reader is None:
            self._note("Журнал аудита недоступен: хранилище не открыто.", "warning")
            return
        self._audit_view = AuditView(reader, self._theme)
        self._content.addWidget(self._audit_view)

    # -- 5. data inventory ------------------------------------------------

    def _build_inventory(self) -> None:
        self._heading("Что и где хранится")
        paths = self.services.paths
        if paths is None:
            self._note("Обзор данных недоступен: пути профиля не определены.", "warning")
            return
        repos = self.services.repositories
        statistics = repos.maintenance.statistics if repos is not None else None
        self._inventory = DataInventory(
            paths,
            self._theme,
            statistics=statistics,
            open_folder=self.services.open_folder,
        )
        self._content.addWidget(self._inventory)

    # -- 7. secrets -------------------------------------------------------

    def _build_secrets(self) -> None:
        self._heading("API-ключи")
        self._note(
            "Ключи и токены хранятся в Диспетчере учётных данных Windows, а не в config.toml и "
            "не в базе. В config.toml лежит только имя записи (credential_ref). Ключи не "
            "попадают в экспорт профиля, в логи и в архив баг-репорта.",
        )
        container = QWidget()
        self._secret_rows = QVBoxLayout(container)
        self._secret_rows.setContentsMargins(0, 0, 0, 0)
        self._content.addWidget(container)
        button = QPushButton("Удалить сохранённые ключи")
        button.setProperty("kind", "danger")
        button.clicked.connect(self._delete_secrets)
        self._content.addWidget(button)
        self._refresh_secret_rows()

    def _refresh_secret_rows(self) -> None:
        layout = self._secret_rows
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        store = self.services.secrets
        if not store.is_available():
            layout.addWidget(QLabel("Хранилище ключей Windows недоступно."))
            return
        stored = [status for status in store.statuses() if status.stored]
        if not stored:
            empty = QLabel("Сохранённых ключей нет.")
            empty.setProperty("role", "secondary")
            layout.addWidget(empty)
            return
        for status in stored:
            layout.addWidget(QLabel(f"• {status.title or status.ref}: ключ сохранён"))

    def _delete_secrets(self) -> None:
        store = self.services.secrets
        if not store.is_available():
            self._status.show_message("Хранилище ключей Windows недоступно.", "error")
            return
        refs = store.stored_refs()
        if not refs:
            self._status.show_message("Сохранённых ключей нет.", "info")
            return
        names = ", ".join(KNOWN_SLOTS[ref].title if ref in KNOWN_SLOTS else ref for ref in refs)
        if not self._confirm(
            "Удалить сохранённые ключи?",
            f"Будут удалены ключи из Диспетчера учётных данных Windows: {names}. "
            "Действие необратимо.",
            "Удалить",
        ):
            return
        removed = 0
        for ref in refs:
            try:
                if store.delete(ref):
                    removed += 1
            except AyrisError as exc:
                self._status.show_message(exc.user_message, "error")
                self._refresh_secret_rows()
                return
        self._status.show_message(f"Удалено ключей: {removed}.", "success")
        self._refresh_secret_rows()

    # -- teardown ---------------------------------------------------------

    def dispose(self) -> None:
        if self._inventory is not None:
            self._inventory.dispose()
        super().dispose()


register_tab("privacy", PrivacyTab)
