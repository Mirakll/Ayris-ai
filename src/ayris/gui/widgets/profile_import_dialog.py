"""Preview a profile ``.zip`` before importing, and collect how to apply it.

The dialog is built from a :class:`~ayris.core.portable_profile.BundlePreview`, which
:meth:`ProfileManager.preview_import` computes without touching anything. It shows
what the archive holds — commands, folders, variables, referenced models — flags the
names that already exist in the target, and warns about a schema written by a newer
build. It then collects three decisions and applies none of them:

* import into a **new profile** or merge into the **current** one,
* how to resolve a name clash — rename the newcomers, overwrite, or skip,
* whether to also apply the archive's **settings** (never its secrets — there are none
  in the archive to begin with).

An archive whose schema the current build cannot read never reaches this dialog: the
tab catches that from :meth:`ProfileManager.preview_import` and shows the message.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ayris.core.portable_profile import CONFLICT_LABELS, BundlePreview, ConflictPolicy
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets.combo_box import ThemedComboBox
from ayris.gui.widgets.notice import InlineNotice

__all__ = ["ProfileImportDialog"]


class ProfileImportDialog(QDialog):
    """Import preview: destination profile and conflict resolution."""

    def __init__(
        self,
        preview: BundlePreview,
        theme: ThemeManager,
        *,
        active_profile_name: str,
        missing_models: tuple[str, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._theme = theme
        self._preview = preview
        self.setWindowTitle("Импорт профиля")
        self.setAccessibleName("Импорт профиля")
        self.setModal(True)

        self._layout = QVBoxLayout(self)
        heading = QLabel(f"Импорт профиля «{preview.manifest.profile_name}»")
        heading.setProperty("role", "h2")
        self._layout.addWidget(heading)

        summary = QLabel(preview.describe())
        summary.setProperty("role", "secondary")
        summary.setWordWrap(True)
        self._layout.addWidget(summary)

        for warning in preview.warnings:
            self._layout.addWidget(InlineNotice(warning, theme, kind="warning"))
        if missing_models:
            names = ", ".join(missing_models)
            self._layout.addWidget(
                InlineNotice(
                    f"Не хватает моделей: {names}. Их можно скачать во вкладке «Обновления».",
                    theme,
                    kind="warning",
                )
            )

        self._contents = QListWidget()
        self._contents.setAccessibleName("Содержимое архива")
        self._fill_contents(preview)
        self._layout.addWidget(self._contents, 1)

        self._build_target(preview, active_profile_name)
        self._build_conflict(preview)
        self._build_config(preview)
        self._build_buttons()

        theme.theme_changed.connect(self._refresh_metrics)
        self._refresh_metrics()
        self._sync_enabled()

    # -- assembled properties ----------------------------------------------

    @property
    def target_new(self) -> bool:
        return self._new_radio.isChecked()

    @property
    def new_profile_name(self) -> str:
        return self._new_name.text().strip()

    @property
    def policy(self) -> ConflictPolicy:
        data = self._policy_combo.currentData()
        return ConflictPolicy(data) if isinstance(data, str) else ConflictPolicy.RENAME

    @property
    def apply_config(self) -> bool:
        return self._apply_config.isChecked() and self._apply_config.isEnabled()

    # -- construction -------------------------------------------------------

    def _fill_contents(self, preview: BundlePreview) -> None:
        conflicts = set(preview.conflicts)
        for name in preview.folders:
            self._contents.addItem(f"📁 {name}")
        for name in preview.commands:
            label = f"⚙ {name}"
            if name in conflicts:
                label += "  ⚠ уже есть"
            item = QListWidgetItem(label)
            self._contents.addItem(item)
        for name in preview.variables:
            self._contents.addItem(f"= {name}")
        for name in preview.models:
            self._contents.addItem(f"◇ модель: {name}")

    def _build_target(self, preview: BundlePreview, active_profile_name: str) -> None:
        self._target_group = QButtonGroup(self)
        self._new_radio = QRadioButton("Импортировать в новый профиль")
        self._new_radio.setChecked(True)
        self._merge_radio = QRadioButton(f"Влить в текущий профиль «{active_profile_name}»")
        self._target_group.addButton(self._new_radio)
        self._target_group.addButton(self._merge_radio)
        self._layout.addWidget(self._new_radio)

        form = QFormLayout()
        self._new_name = QLineEdit(preview.manifest.profile_name)
        self._new_name.setAccessibleName("Имя нового профиля")
        form.addRow("Имя нового профиля:", self._new_name)
        self._layout.addLayout(form)
        self._layout.addWidget(self._merge_radio)

        self._new_radio.toggled.connect(self._sync_enabled)

    def _build_conflict(self, preview: BundlePreview) -> None:
        self._conflict_form = QFormLayout()
        self._policy_combo = ThemedComboBox()
        for policy in (ConflictPolicy.RENAME, ConflictPolicy.OVERWRITE, ConflictPolicy.SKIP):
            self._policy_combo.addItem(CONFLICT_LABELS[policy], str(policy))
        self._conflict_label = QLabel("При совпадении имени:")
        self._conflict_form.addRow(self._conflict_label, self._policy_combo)
        self._layout.addLayout(self._conflict_form)
        if preview.conflicts:
            note = QLabel(f"Совпадают имена: {', '.join(preview.conflicts)}.")
            note.setProperty("role", "secondary")
            note.setWordWrap(True)
            self._layout.addWidget(note)

    def _build_config(self, preview: BundlePreview) -> None:
        self._apply_config = QCheckBox("Применить настройки из архива")
        self._apply_config.setToolTip(
            "Настройки из архива заменят текущие. Секретов в архиве нет — ключи "
            "останутся вашими."
        )
        if not preview.has_config:
            self._apply_config.setEnabled(False)
            self._apply_config.setToolTip("В этом архиве настроек нет")
        self._layout.addWidget(self._apply_config)

    def _build_buttons(self) -> None:
        buttons = QDialogButtonBox()
        self._import_button = QPushButton("Импортировать")
        self._import_button.setProperty("kind", "primary")
        cancel = QPushButton("Отмена")
        self._import_button.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        buttons.addButton(cancel, QDialogButtonBox.ButtonRole.RejectRole)
        buttons.addButton(self._import_button, QDialogButtonBox.ButtonRole.AcceptRole)
        self._layout.addWidget(buttons)

    def _sync_enabled(self, _checked: bool | None = None) -> None:
        new = self._new_radio.isChecked()
        self._new_name.setEnabled(new)
        # A brand-new profile starts empty, so a name clash cannot arise: the policy
        # only has meaning when merging into a profile that already has content.
        merging_with_conflicts = not new and bool(self._preview.conflicts)
        self._policy_combo.setEnabled(merging_with_conflicts)
        self._conflict_label.setEnabled(merging_with_conflicts)

    def _refresh_metrics(self, _theme: object | None = None) -> None:
        pad = self._theme.metric("spacing_xl")
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(self._theme.metric("spacing_md"))
        self.setMinimumWidth(self._theme.metric("dialog_width"))
        self._contents.setMinimumHeight(self._theme.metric("control_height") * 4)
