"""Tab «Горячие клавиши»: the system hotkeys and the command hotkeys (task 55).

Two tables share one page. The top one is the assistant's own five system hotkeys
(push-to-talk, wake toggle, overlay, mute, cancel), read from and written back to
``config.toml`` through task 02 so a change here re-registers live. The bottom one
lists every command that owns a hotkey trigger, and editing a combination here
rewrites the command's trigger and lets task 54 hot-reload it.

The page itself holds no combination logic: the pure builders in
:mod:`ayris.gui.widgets.hotkey_table` resolve every row's status and cross-mark
conflicts from plain values, and the capture goes through
:class:`~ayris.gui.widgets.hotkey_capture.HotkeyCaptureDialog`, which speaks to the
live :class:`~ayris.utils.hotkey_manager.HotkeyManager`. That keeps the tab a thin
shell over task 37 — combinations are parsed in exactly one place — and lets the
offscreen tests of task 55 drive :meth:`HotkeysTab.assign_system`,
:meth:`~HotkeysTab.assign_command` and :meth:`~HotkeysTab.set_interception`
directly, without a keyboard or a modal dialog.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
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

from ayris.actions.macros.schema import HotkeyTrigger
from ayris.core.config import ConfigChanged as SettingsDiff
from ayris.core.config import ConfigManager, Settings
from ayris.core.events import CommandReloaded, CommandsChanged, EventBus, OpenCommandRequested
from ayris.core.models import TriggerType
from ayris.core.profile import ProfileSwitched
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import ConfirmDialog, ToggleSwitch
from ayris.gui.widgets.command_tree_model import CommandTreeStore
from ayris.gui.widgets.hotkey_capture import CaptureResult, HotkeyCaptureDialog
from ayris.gui.widgets.hotkey_table import (
    SYSTEM_HOTKEY_SPECS,
    CommandHotkeyEntry,
    CommandHotkeyRow,
    CommandHotkeyTable,
    HotkeyRows,
    SystemHotkeyRow,
    SystemHotkeyTable,
    build_rows,
)
from ayris.gui.widgets.notice import InlineNotice, NoticeKind
from ayris.utils.hotkey_backends.interception import probe_interception
from ayris.utils.hotkey_manager import active_hotkey_manager
from ayris.utils.hotkeys import Hotkey, try_parse_hotkey
from ayris.utils.logger import get_logger

__all__ = ["HotkeysTab"]

_log = get_logger(__name__)

#: Where the optional Interception driver is downloaded from — the «установить»
#: link of task 55, item 6. Opened in the user's browser, never fetched here.
_INTERCEPTION_URL = "https://github.com/oblitum/Interception/releases"


class _SectionHeader(QWidget):
    """Заголовок раздела страницы: подпись h2 и тонкая линия-разделитель под ней.

    На этой вкладке разделы — это просто таблицы без карточек, поэтому голый h2
    сливался с их строками: подзаголовок было не отличить от текста. Линия под
    заголовком явно отбивает начало раздела. Необязательный ``trailing`` встаёт
    в одну строку с заголовком справа (кнопка «Сбросить все системные»).
    """

    def __init__(
        self,
        text: str,
        theme: ThemeManager,
        *,
        trailing: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(theme.metric("spacing_xs"))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.metric("spacing_md"))
        title = QLabel(text)
        title.setProperty("role", "h2")
        row.addWidget(title)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing)
        box.addLayout(row)
        rule = QFrame()
        rule.setProperty("rule", True)
        box.addWidget(rule)


class HotkeysTab(SettingsTab):
    """The «Горячие клавиши» page: system table, command table, games toggle."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
        *,
        store: CommandTreeStore | None = None,
    ) -> None:
        super().__init__("hotkeys", "Горячие клавиши", ("hotkeys",), manager, theme, bus)
        # This page writes single fields through explicit «Назначить»/«Сброс»
        # buttons, so the base section-wide reset and dirty chrome would only
        # confuse it.
        self.reset_button.hide()
        self.dirty_label.hide()

        self._bus = bus
        self._defaults = Settings().hotkeys
        self._rows: HotkeyRows = HotkeyRows((), ())
        self._suppress_reload = False
        self._syncing_interception = False
        self._interception_notice: InlineNotice | None = None
        self._unsub_commands: Callable[[], None] = lambda: None
        self._unsub_reloaded: Callable[[], None] = lambda: None
        self._unsub_profile: Callable[[], None] = lambda: None

        self._store = store if store is not None else self._build_store()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_lg"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._build_controls()
        self._build_system_section()
        self._build_interception_section()
        self._build_command_section()
        self._content.addStretch(1)

        if bus is not None:
            self._unsub_commands = bus.subscribe(CommandsChanged, self._on_commands_changed)
            self._unsub_reloaded = bus.subscribe(CommandReloaded, self._on_command_reloaded)
            self._unsub_profile = bus.subscribe(ProfileSwitched, self._on_profile_switched)

    # -- construction -------------------------------------------------------

    def _build_controls(self) -> None:
        """The top strip: transient notices, the problem counter, the «only» filter."""
        # Transient feedback (a refused clash, a failed write) is dropped in here so
        # it appears above the tables without shifting the section headers.
        self._notice_holder = QVBoxLayout()
        self._notice_holder.setContentsMargins(0, 0, 0, 0)
        self._content.addLayout(self._notice_holder)

        row = QHBoxLayout()
        self._problem_label = QLabel()
        self._problem_label.setProperty("role", "secondary")
        row.addWidget(self._problem_label)
        row.addStretch(1)
        filter_caption = QLabel("Только проблемные")
        filter_caption.setProperty("role", "secondary")
        row.addWidget(filter_caption)
        self._conflicts_toggle = ToggleSwitch(
            self._theme, label="Показывать только конфликты и ошибки"
        )
        self._conflicts_toggle.toggled.connect(lambda *_: self._rebuild())
        row.addWidget(self._conflicts_toggle)
        self._content.addLayout(row)

    def _build_system_section(self) -> None:
        self._reset_all_button = QPushButton("Сбросить все системные")
        self._reset_all_button.clicked.connect(self.reset_all_system)
        self._content.addWidget(
            _SectionHeader(
                "Системные горячие клавиши", self._theme, trailing=self._reset_all_button
            )
        )

        self._system_table = SystemHotkeyTable(self._theme)
        self._system_table.assign_requested.connect(self._capture_system)
        self._system_table.clear_requested.connect(self.clear_system)
        self._system_table.reset_requested.connect(self.reset_system)
        self._content.addWidget(self._system_table)

    def _build_interception_section(self) -> None:
        self._content.addWidget(_SectionHeader("Работа в играх", self._theme))

        row = QHBoxLayout()
        caption = QLabel(
            "Перехватывать клавиши драйвером Interception — чтобы хоткеи срабатывали "
            "в полноэкранных играх, которые прячут обычные горячие клавиши Windows."
        )
        caption.setProperty("role", "secondary")
        caption.setWordWrap(True)
        row.addWidget(caption, 1)
        self._interception_toggle = ToggleSwitch(
            self._theme, label="Работа в играх через Interception"
        )
        self._interception_toggle.toggled.connect(self._on_interception_toggled)
        row.addWidget(self._interception_toggle)
        self._content.addLayout(row)

        # A slot the driver-status notice is dropped into and taken out of, so the
        # rest of the section never shifts when there is nothing to say.
        self._interception_holder = QVBoxLayout()
        self._interception_holder.setContentsMargins(0, 0, 0, 0)
        self._content.addLayout(self._interception_holder)

    def _build_command_section(self) -> None:
        self._content.addWidget(_SectionHeader("Горячие клавиши команд", self._theme))

        self._command_table = CommandHotkeyTable(self._theme)
        self._command_table.assign_requested.connect(self._capture_command)
        self._command_table.clear_requested.connect(self.clear_command)
        self._command_table.open_requested.connect(self._open_command)
        self._content.addWidget(self._command_table)

        admin_hint = QLabel(
            "Хоткеям, помеченным 🛡, нужны права администратора. Если Ayris запущен "
            "без прав администратора, такой хоткей не сработает поверх окон с более "
            "высокими правами. Постоянное повышение включается в разделе «Основные», "
            "параметром «Всегда запускать от администратора»."
        )
        admin_hint.setProperty("role", "secondary")
        admin_hint.setWordWrap(True)
        self._content.addWidget(admin_hint)

    # -- data + refresh -----------------------------------------------------

    def _build_store(self) -> CommandTreeStore | None:
        """A store over the live database and active profile, or ``None``.

        Mirrors the lazy backend of the «Команды» and «Обновления» tabs: the page
        must build even when storage is not ready, degrading to an empty command
        table rather than crashing.
        """
        try:
            from ayris.core.database import get_database
            from ayris.core.repositories import Repositories

            repositories = Repositories(get_database())
            active = repositories.profiles.active()
            if active is None or active.id is None:
                return None
            limit = self._manager.settings.commands.version_history_limit
            return CommandTreeStore(repositories, active.id, version_limit=limit)
        except Exception:
            _log.exception("не удалось построить хранилище команд для хоткеев")
            return None

    def _command_entries(self) -> list[CommandHotkeyEntry]:
        if self._store is None:
            return []
        try:
            commands = {c.id: c for c in self._store.commands() if c.id is not None}
            entries: list[CommandHotkeyEntry] = []
            for trigger in self._store.triggers():
                if trigger.type is not TriggerType.HOTKEY:
                    continue
                command = commands.get(trigger.command_id)
                if command is None:
                    continue
                entries.append(
                    CommandHotkeyEntry(
                        command_id=trigger.command_id,
                        name=command.name,
                        folder=tuple(self._store.folder_path(command.folder_id)),
                        combo=str(trigger.payload.get("combo", "")),
                        command_enabled=command.enabled,
                        trigger_enabled=bool(trigger.payload.get("enabled", True)),
                        require_admin=command.require_admin,
                    )
                )
            return entries
        except Exception:
            _log.exception("не удалось собрать хоткеи команд")
            return []

    def _build_hotkey_rows(self) -> HotkeyRows:
        hotkeys = self._manager.settings.hotkeys
        system_combos = {
            spec.field: getattr(hotkeys, spec.field, "") for spec in SYSTEM_HOTKEY_SPECS
        }
        system_defaults = {
            spec.field: getattr(self._defaults, spec.field, "") for spec in SYSTEM_HOTKEY_SPECS
        }
        manager = active_hotkey_manager()
        errors = dict(manager.registration_errors) if manager is not None else {}
        return build_rows(
            system_combos, system_defaults, self._command_entries(), registration_errors=errors
        )

    def _rebuild(self) -> None:
        self._rows = self._build_hotkey_rows()
        only = self._conflicts_toggle.isChecked()
        system = (
            [r for r in self._rows.system if r.has_problem] if only else list(self._rows.system)
        )
        commands = (
            [r for r in self._rows.commands if r.has_problem] if only else list(self._rows.commands)
        )
        available = active_hotkey_manager() is not None
        self._system_table.set_capture_available(available)
        self._command_table.set_capture_available(available)
        self._system_table.set_rows(system)
        self._command_table.set_rows(commands)
        self._update_problem_label(self._rows.problem_count)
        self._syncing_interception = True
        try:
            self._interception_toggle.setChecked(self._manager.settings.hotkeys.use_interception)
        finally:
            self._syncing_interception = False

    def _update_problem_label(self, count: int) -> None:
        if count:
            self._problem_label.setText(f"Проблемы: {count}")
            self._problem_label.setProperty("status", "warning")
        else:
            self._problem_label.setText("Проблем нет")
            self._problem_label.setProperty("status", None)
        self._problem_label.style().unpolish(self._problem_label)
        self._problem_label.style().polish(self._problem_label)

    # -- system hotkeys -----------------------------------------------------

    def _capture_system(self, field: str) -> None:
        result = self._capture()
        if result is None:
            return
        if result.cleared:
            self.clear_system(field)
        elif result.hotkey is not None:
            self.assign_system(field, result.hotkey)

    def assign_system(self, field: str, hotkey: Hotkey) -> bool:
        """Write one system combination, refusing a clash with another system hotkey.

        The config layer's own validator would reject a duplicate too, but catching
        it here lets the message name the hotkey it collides with instead of showing
        a raw validation error.
        """
        clash = self._system_clash(field, hotkey.canonical)
        if clash is not None:
            self._notify(
                f"Сочетание {hotkey.label_ru} уже занято системным хоткеем «{clash}».",
                kind="warning",
            )
            return False
        return self._write_hotkey(field, hotkey.canonical)

    def clear_system(self, field: str) -> None:
        self._write_hotkey(field, "")

    def reset_system(self, field: str) -> None:
        self._write_hotkey(field, getattr(self._defaults, field, ""))

    def reset_all_system(self) -> None:
        changes = self._system_reset_changes()
        if not changes:
            self._notify("Системные горячие клавиши уже сброшены к значениям по умолчанию.")
            return
        listing = "\n".join(f"• {title}: {old} → {new}" for title, old, new in changes)
        dialog = ConfirmDialog(
            "Сбросить системные горячие клавиши?",
            f"Вернутся значения по умолчанию:\n{listing}\n\nХоткеи команд не затрагиваются.",
            self._theme,
            confirm_text="Сбросить",
            dangerous=True,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._apply_system_defaults()

    def _apply_system_defaults(self) -> None:
        values = {
            f"hotkeys.{spec.field}": getattr(self._defaults, spec.field, "")
            for spec in SYSTEM_HOTKEY_SPECS
        }
        try:
            self._manager.apply(values)
        except Exception:
            _log.exception("не удалось сбросить системные хоткеи")
            self._notify("Не удалось сбросить системные горячие клавиши.", kind="error")

    def _system_reset_changes(self) -> list[tuple[str, str, str]]:
        hotkeys = self._manager.settings.hotkeys
        changes: list[tuple[str, str, str]] = []
        for spec in SYSTEM_HOTKEY_SPECS:
            current = getattr(hotkeys, spec.field, "")
            default = getattr(self._defaults, spec.field, "")
            if current != default:
                changes.append((spec.title, _combo_label(current), _combo_label(default)))
        return changes

    def _system_clash(self, field: str, canonical: str) -> str | None:
        hotkeys = self._manager.settings.hotkeys
        for spec in SYSTEM_HOTKEY_SPECS:
            if spec.field == field:
                continue
            other = getattr(hotkeys, spec.field, "")
            parsed = try_parse_hotkey(other) if other else None
            if parsed is not None and parsed.canonical == canonical:
                return spec.title
        return None

    def _write_hotkey(self, field: str, combo: str) -> bool:
        try:
            self._manager.apply({f"hotkeys.{field}": combo})
        except Exception:
            _log.exception("не удалось записать системный хоткей %s", field)
            self._notify("Не удалось сохранить сочетание.", kind="error")
            return False
        return True

    # -- command hotkeys ----------------------------------------------------

    def _capture(self) -> CaptureResult | None:
        """Open the capture dialog over the live manager; ``None`` if unavailable."""
        manager = active_hotkey_manager()
        if manager is None:
            self._notify(
                "Обработчик горячих клавиш не запущен — назначить сочетание сейчас нельзя.",
                kind="warning",
            )
            return None
        dialog = HotkeyCaptureDialog(self._theme, manager, parent=self)
        return dialog.run()

    def _capture_command(self, command_id: int) -> None:
        result = self._capture()
        if result is None:
            return
        if result.cleared:
            self.clear_command(command_id)
        elif result.hotkey is not None:
            self.assign_command(command_id, result.hotkey)

    def assign_command(self, command_id: int, hotkey: Hotkey) -> bool:
        """Rewrite a command's hotkey trigger to ``hotkey`` and hot-reload it.

        A command in the UI carries one hotkey, so every hotkey trigger is replaced
        by the single new one; the command's other triggers (voice, timer, event)
        are kept untouched. A clash with another command or a system hotkey is *not*
        refused — it is saved and shown as a conflict in both rows, matching how the
        manager keeps the first claimant and skips the rest.
        """
        if self._store is None:
            self._notify("Библиотека команд недоступна.", kind="warning")
            return False
        try:
            model = self._store.command_model(command_id)
            keep = [t for t in model.triggers if not isinstance(t, HotkeyTrigger)]
            model.triggers = [*keep, HotkeyTrigger(combo=hotkey.canonical, enabled=True)]
            self._store.save_command(model)
        except Exception:
            _log.exception("не удалось назначить хоткей команде %s", command_id)
            self._notify("Не удалось назначить сочетание команде.", kind="error")
            return False
        self._commands_changed()
        return True

    def clear_command(self, command_id: int) -> bool:
        if self._store is None:
            self._notify("Библиотека команд недоступна.", kind="warning")
            return False
        try:
            model = self._store.command_model(command_id)
            keep = [t for t in model.triggers if not isinstance(t, HotkeyTrigger)]
            if len(keep) == len(model.triggers):
                return True
            model.triggers[:] = keep
            self._store.save_command(model)
        except Exception:
            _log.exception("не удалось очистить хоткей команды %s", command_id)
            self._notify("Не удалось очистить сочетание команды.", kind="error")
            return False
        self._commands_changed()
        return True

    def _open_command(self, command_id: int) -> None:
        """Ask the window to switch to «Команды» and select this command (task 51)."""
        if self._bus is not None:
            self._bus.publish(OpenCommandRequested(command_id))

    def _commands_changed(self) -> None:
        # Publishing re-registers the command through task 54; guard the echo so our
        # own CommandsChanged does not rebuild twice, then rebuild once by hand.
        if self._bus is not None:
            self._suppress_reload = True
            try:
                self._bus.publish(CommandsChanged())
            finally:
                self._suppress_reload = False
        self._rebuild()

    # -- games (interception) ----------------------------------------------

    def set_interception(self, enabled: bool) -> bool:
        """Toggle the games backend, refusing to enable it without a live driver.

        Task 55 forbids silently switching to Interception when the driver is not
        there: probe it, and if it is missing or not running, keep the WinAPI
        fallback, explain why with an inline notice, and snap the switch back off.
        """
        if not enabled:
            self._clear_interception_notice()
            return self._write_interception(False)
        status = probe_interception()
        if not status.ready:
            self._show_interception_notice(status.message)
            self._write_interception(False)
            self._sync_interception_toggle(False)
            return False
        self._clear_interception_notice()
        return self._write_interception(True)

    def _on_interception_toggled(self, checked: bool) -> None:
        if self._syncing_interception:
            return
        self.set_interception(checked)

    def _write_interception(self, value: bool) -> bool:
        if self._manager.settings.hotkeys.use_interception == value:
            return True
        try:
            self._manager.apply({"hotkeys.use_interception": value})
        except Exception:
            _log.exception("не удалось переключить режим Interception")
            self._notify("Не удалось сохранить настройку работы в играх.", kind="error")
            return False
        return True

    def _sync_interception_toggle(self, value: bool) -> None:
        self._syncing_interception = True
        try:
            self._interception_toggle.setChecked(value)
        finally:
            self._syncing_interception = False

    def _show_interception_notice(self, message: str) -> None:
        self._clear_interception_notice()
        notice = InlineNotice(message, self._theme, kind="warning")
        notice.closed.connect(self._clear_interception_notice)
        self._interception_notice = notice
        self._interception_holder.addWidget(notice)
        link = QPushButton("Скачать драйвер Interception")
        link.setProperty("link", True)
        link.setCursor(Qt.CursorShape.PointingHandCursor)
        link.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(_INTERCEPTION_URL)))
        self._interception_holder.addWidget(link)

    def _clear_interception_notice(self) -> None:
        self._interception_notice = None
        while self._interception_holder.count():
            item = self._interception_holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    # -- notices ------------------------------------------------------------

    def _notify(self, message: str, *, kind: NoticeKind = "info") -> None:
        while self._notice_holder.count():
            item = self._notice_holder.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._notice_holder.addWidget(
            InlineNotice(message, self._theme, kind=kind, auto_hide_ms=6000)
        )

    # -- state exposed to tests --------------------------------------------

    @property
    def problem_count(self) -> int:
        return self._rows.problem_count

    def system_rows(self) -> tuple[SystemHotkeyRow, ...]:
        return self._rows.system

    def command_rows(self) -> tuple[CommandHotkeyRow, ...]:
        return self._rows.commands

    # -- event wiring + lifecycle ------------------------------------------

    def load_from_config(self) -> None:
        super().load_from_config()
        self._rebuild()

    def _on_config_changed(self, diff: SettingsDiff) -> None:
        super()._on_config_changed(diff)
        if any(path.startswith("hotkeys.") for path in diff.paths):
            self._rebuild()

    def _on_commands_changed(self, _event: CommandsChanged) -> None:
        if not self._suppress_reload:
            self._rebuild()

    def _on_command_reloaded(self, _event: CommandReloaded) -> None:
        self._rebuild()

    def _on_profile_switched(self, _event: ProfileSwitched) -> None:
        self._store = self._build_store()
        self._rebuild()

    def dispose(self) -> None:
        self._unsub_commands()
        self._unsub_reloaded()
        self._unsub_profile()
        super().dispose()


def _combo_label(combo: str) -> str:
    """A stored combination as the user reads it, «—» when unset or unparseable."""
    hotkey = try_parse_hotkey(combo) if combo else None
    return hotkey.label_ru if hotkey is not None else "—"


register_tab("hotkeys", HotkeysTab)
