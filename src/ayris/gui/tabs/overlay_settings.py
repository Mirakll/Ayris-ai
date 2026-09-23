"""Tab «Панель / Сфера» (задача 56): облик окна-дашборда и его сферы.

Плавающего оверлея больше нет, поэтому здесь ничего не позиционирует окно —
только то, что настраивает его вид и нагрузку: форма и движение сферы, тумблеры
экономии анимаций, состав столбца диалога, видимость блоков управления и палитра
сферы. Всё применяется на лету через :class:`ConfigChanged`; перезапуск не нужен.

Встроенный предпросмотр (:class:`SpherePreview`) показывает ту же сферу задачи 45,
что и главное окно, со всеми пятью состояниями. Он обновляется немедленно из
«ожидающих» правок — до отложенного сохранения, — через переопределённый хук
:meth:`_pending_changed`, так что ползунок виден в сфере сразу.
"""

from __future__ import annotations

from typing import Any, Final

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ayris.core.config import ConfigManager, OverlayConfig
from ayris.core.events import EventBus
from ayris.gui.tabs.base import SettingsTab
from ayris.gui.tabs.registry import register_tab
from ayris.gui.theme import ThemeManager
from ayris.gui.widgets import SettingCard, SliderField, ThemedComboBox, ToggleSwitch
from ayris.gui.widgets.sphere_preview import SpherePreview

__all__ = ["OVERLAY_PRESETS", "OverlayTab"]

#: Множитель для float-полей 0..3, показанных целым процентом на слайдере.
_SCALE: Final = 100.0

#: Пункты комбобокса частоты кадров: подпись → значение (Авто = потолок le=144).
_FPS_OPTIONS: Final[tuple[tuple[str, int], ...]] = (("30", 30), ("60", 60), ("Авто", 144))

#: Пресеты «экономии»: каждый выставляет связанный набор полей сферы разом. Ключи —
#: короткие имена внутри секции ``overlay`` (без префикса), значения — числа/флаги.
OVERLAY_PRESETS: Final[dict[str, dict[str, Any]]] = {
    "beautiful": {
        "sphere_points": 1500,
        "target_fps": 60,
        "animations": True,
        "rotation": True,
        "pulsation": True,
        "waves": True,
        "error_flash": True,
        "stop_when_hidden": True,
    },
    "balanced": {
        "sphere_points": 600,
        "target_fps": 60,
        "animations": True,
        "rotation": True,
        "pulsation": True,
        "waves": True,
        "error_flash": True,
        "stop_when_hidden": True,
    },
    "economy": {
        "sphere_points": 250,
        "target_fps": 30,
        "animations": True,
        "rotation": True,
        "pulsation": True,
        "waves": False,
        "error_flash": True,
        "stop_when_hidden": True,
    },
}

#: Подписи кнопок пресетов в порядке слева направо.
_PRESET_LABELS: Final[tuple[tuple[str, str], ...]] = (
    ("beautiful", "Красиво"),
    ("balanced", "Сбалансированно"),
    ("economy", "Экономно"),
)


class _PalettePreview(QFrame):
    """Ряд образцов текущей темы — чтобы «следовать системе» показывало эффект.

    Сама тема не настраивается здесь (её выбор — на вкладке «Общие»); эти образцы
    лишь отражают активный :class:`ThemeManager`, который главное окно переключает,
    когда тумблер «следовать системе» меняется. Обновляются на ``theme_changed``.
    """

    _TOKENS: Final[tuple[tuple[str, str], ...]] = (
        ("Фон", "background"),
        ("Панель", "surface"),
        ("Акцент", "accent"),
        ("Текст", "text_primary"),
    )

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self.setProperty("transparent", True)
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._swatches: list[tuple[QFrame, str]] = []
        for label, token in self._TOKENS:
            column = QVBoxLayout()
            column.setSpacing(theme.metric("spacing_xs"))
            swatch = QFrame()
            swatch.setObjectName("paletteSwatch")
            caption = QLabel(label)
            caption.setProperty("role", "muted")
            caption.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            column.addWidget(swatch)
            column.addWidget(caption)
            self._row.addLayout(column)
            self._swatches.append((swatch, token))
        self._row.addStretch(1)
        theme.theme_changed.connect(self._refresh)
        self._refresh()

    def _refresh(self, _theme: object | None = None) -> None:
        size = self._theme.metric("control_height")
        radius = self._theme.metric("radius_md")
        border = self._theme.theme.color("border")
        self._row.setSpacing(self._theme.metric("spacing_md"))
        for swatch, token in self._swatches:
            swatch.setFixedSize(size, size)
            swatch.setStyleSheet(
                f"#paletteSwatch {{ background: {self._theme.theme.color(token)};"
                f" border: 1px solid {border}; border-radius: {radius}px; }}"
            )


class _AccentField(QWidget):
    """Выбор акцента сферы: образец + «Выбрать цвет» (диалог) + «Из темы» (сброс).

    Держит нормализованное ``#rrggbb`` в нижнем регистре — ровно то, что принимает
    валидатор :class:`OverlayConfig`; пустая строка означает «наследовать акцент
    темы». Отдельный виджет (а не строковое поле) выбран нарочно: правка вручную
    по символу отправляла бы валидатору незавершённый ``#rr`` и падала.
    """

    changed = Signal(str)

    def __init__(self, theme: ThemeManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._value = ""
        self.setProperty("transparent", True)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(theme.metric("spacing_sm"))
        self._swatch = QFrame()
        self._swatch.setObjectName("accentSwatch")
        self._pick = QPushButton("Выбрать цвет")
        self._pick.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pick.clicked.connect(self._choose)
        self._clear = QPushButton("Из темы")
        self._clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear.clicked.connect(lambda: self.set_value("", emit=True))
        row.addWidget(self._swatch)
        row.addWidget(self._pick)
        row.addWidget(self._clear)
        row.addStretch(1)
        theme.theme_changed.connect(self._refresh)
        self._refresh()

    def value(self) -> str:
        return self._value

    def set_value(self, value: str, *, emit: bool = False) -> None:
        self._value = (value or "").strip().lower()
        self._refresh()
        if emit:
            self.changed.emit(self._value)

    def _choose(self) -> None:
        initial = QColor(self._value) if self._value else QColor(self._theme.theme.color("accent"))
        chosen = QColorDialog.getColor(initial, self, "Акцент сферы")
        if chosen.isValid():
            self.set_value(chosen.name(), emit=True)

    def _refresh(self, _theme: object | None = None) -> None:
        radius = self._theme.metric("radius_md")
        size = self._theme.metric("control_height")
        border = self._theme.theme.color("border")
        shown = self._value or self._theme.theme.color("accent")
        self._swatch.setFixedSize(size, size)
        self._swatch.setStyleSheet(
            f"#accentSwatch {{ background: {shown}; border: 1px solid {border};"
            f" border-radius: {radius}px; }}"
        )
        self._clear.setEnabled(bool(self._value))


class OverlayTab(SettingsTab):
    """Вкладка «Панель / Сфера» (задача 56): облик окна-дашборда и его сферы."""

    def __init__(
        self,
        manager: ConfigManager,
        theme: ThemeManager,
        bus: EventBus | None = None,
    ) -> None:
        super().__init__("overlay", "Панель / Сфера", ("overlay",), manager, theme, bus)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        container = QWidget()
        self._content = QVBoxLayout(container)
        self._content.setSpacing(theme.metric("spacing_md"))
        scroll.setWidget(container)
        self.body.addWidget(scroll)

        self._build_launch()
        self._build_sphere()
        self._build_economy()
        self._build_dialogue()
        self._build_theme()
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

    def _toggle_card(self, title: str, desc: str, path: str, label: str) -> ToggleSwitch:
        toggle = ToggleSwitch(self._theme, label=label)
        self.bind_toggle(toggle, path, label)
        self._content.addWidget(SettingCard(title, desc, toggle, self._theme))
        return toggle

    def _build_launch(self) -> None:
        self._add_header("Запуск")
        self._toggle_card(
            "Показывать окно при запуске",
            "Открывать панель Ayris при старте. Выключено — программа ждёт в трее.",
            "overlay.enabled",
            "Показывать окно при запуске",
        )

    def _build_sphere(self) -> None:
        self._add_header("Сфера")
        self.preview = SpherePreview(self._theme)
        self._content.addWidget(self.preview)
        self._add_caption("Предпросмотр отражает правки сразу — переключите состояние кнопками.")

        points = SliderField(
            self._theme,
            minimum=100,
            maximum=3000,
            value=600,
            unit=" точек",
            label="Количество точек",
        )
        points.slider.setSingleStep(50)
        points.slider.setPageStep(250)
        self._bind_int_slider(points, "overlay.sphere_points", "Количество точек")
        self._content.addWidget(
            SettingCard(
                "Количество точек",
                "Плотность сетки сферы. Больше — красивее, но дороже в отрисовке.",
                points,
                self._theme,
            )
        )

        self._scaled_card(
            "Скорость вращения",
            "Как быстро сфера вращается вокруг оси.",
            "overlay.rotation_speed",
            "Скорость вращения",
        )
        self._scaled_card(
            "Пульсация в «Слушаю»",
            "Насколько сильно сфера дышит, пока слушает вас.",
            "overlay.pulse_amplitude",
            "Пульсация в «Слушаю»",
        )
        self._scaled_card(
            "Волны в «Говорю»",
            "Интенсивность волн по поверхности во время ответа.",
            "overlay.wave_intensity",
            "Волны в «Говорю»",
        )

    def _build_economy(self) -> None:
        self._add_header("Экономия анимаций")
        self._toggle_card(
            "Анимации сферы",
            "Общий выключатель. Выключено — сфера замирает и не тратит кадры.",
            "overlay.animations",
            "Анимации сферы",
        )
        self._toggle_card(
            "Вращение", "Медленный поворот сферы вокруг оси.", "overlay.rotation", "Вращение"
        )
        self._toggle_card(
            "Пульсация", "Дыхание сферы в состоянии «Слушаю».", "overlay.pulsation", "Пульсация"
        )
        self._toggle_card(
            "Волны", "Волны по поверхности в состоянии «Говорю».", "overlay.waves", "Волны"
        )
        self._toggle_card(
            "Вспышка при ошибке",
            "Резкая вспышка сферы, когда что-то пошло не так.",
            "overlay.error_flash",
            "Вспышка при ошибке",
        )

        self._fps_combo = ThemedComboBox()
        for label, value in _FPS_OPTIONS:
            self._fps_combo.addItem(label, value)
        self.bind_combo(self._fps_combo, "overlay.target_fps", "Частота кадров")
        self._content.addWidget(
            SettingCard(
                "Частота кадров",
                "Потолок кадров в секунду для сферы. «Авто» — по частоте дисплея.",
                self._fps_combo,
                self._theme,
            )
        )

        self._toggle_card(
            "Замирать при скрытии окна",
            "Не рисовать сферу, пока окно свёрнуто или спрятано в трей.",
            "overlay.stop_when_hidden",
            "Замирать при скрытии окна",
        )

        self._add_caption("Пресеты выставляют связанные параметры сферы разом:")
        row = QWidget()
        buttons = QHBoxLayout(row)
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(self._theme.metric("spacing_sm"))
        for name, label in _PRESET_LABELS:
            button = QPushButton(label)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _checked=False, key=name: self._apply_preset(key))
            buttons.addWidget(button)
        buttons.addStretch(1)
        self._content.addWidget(row)

    def _build_dialogue(self) -> None:
        self._add_header("Диалог")
        self._toggle_card(
            "Показывать лог диалога",
            "Столбец реплик под сферой. Выключено — остаётся только сфера.",
            "overlay.show_log",
            "Показывать лог диалога",
        )

        lines = SliderField(
            self._theme, minimum=1, maximum=50, value=6, unit=" строк", label="Строк в логе"
        )
        self._bind_int_slider(lines, "overlay.log_lines", "Строк в логе")
        self._content.addWidget(
            SettingCard(
                "Строк в логе",
                "Сколько последних реплик держать в столбце диалога.",
                lines,
                self._theme,
            )
        )

        self._toggle_card(
            "Показывать распознанную речь",
            "Добавлять в лог то, что вы сказали.",
            "overlay.show_transcript",
            "Показывать распознанную речь",
        )
        self._toggle_card(
            "Показывать ответы",
            "Добавлять в лог ответы Ayris.",
            "overlay.show_answers",
            "Показывать ответы",
        )
        self._toggle_card(
            "Показывать ошибки",
            "Добавлять в лог сбои действий и макросов.",
            "overlay.show_errors",
            "Показывать ошибки",
        )

        self._add_header("Блоки управления")
        self._toggle_card(
            "Кнопка микрофона",
            "Кнопка включения прослушивания.",
            "overlay.show_mic",
            "Кнопка микрофона",
        )
        self._toggle_card(
            "Строка таймеров",
            "Активные таймеры и напоминания.",
            "overlay.show_timers",
            "Строка таймеров",
        )
        self._toggle_card(
            "Селектор профиля",
            "Переключатель профиля голоса.",
            "overlay.show_profile",
            "Селектор профиля",
        )
        self._toggle_card(
            "Кнопка настроек",
            "Кнопка-гамбургер, открывающая настройки.",
            "overlay.show_settings_button",
            "Кнопка настроек",
        )
        self._toggle_card(
            "Поле текстового ввода",
            "Строка ввода команд текстом рядом с микрофоном.",
            "overlay.show_text_input",
            "Поле текстового ввода",
        )

    def _build_theme(self) -> None:
        self._add_header("Тема сферы")
        self._toggle_card(
            "Следовать системной теме",
            "Переключать светлую и тёмную тему вслед за Windows.",
            "overlay.follow_system_theme",
            "Следовать системной теме",
        )
        self._content.addWidget(_PalettePreview(self._theme))

        self._accent = _AccentField(self._theme)
        self._bind(
            self._accent,
            "overlay.sphere_accent",
            "Акцент сферы",
            self._accent.value,
            self._accent.set_value,
            self._accent.changed,
        )
        self._content.addWidget(
            SettingCard(
                "Акцент сферы",
                "Свой цвет свечения сферы. «Из темы» — наследовать акцент оформления.",
                self._accent,
                self._theme,
            )
        )

    # -- helpers ------------------------------------------------------------

    def _bind_int_slider(self, field: SliderField, path: str, label: str) -> None:
        self._bind(field, path, label, field.value, field.setValue, field.value_changed)

    def _scaled_card(self, title: str, desc: str, path: str, label: str) -> None:
        """Слайдер 0..300 % поверх float-поля 0..3 (делим/умножаем на :data:`_SCALE`)."""
        field = SliderField(self._theme, minimum=0, maximum=300, value=100, unit=" %", label=label)
        field.slider.setSingleStep(5)
        field.slider.setPageStep(25)

        def getter() -> float:
            return field.value() / _SCALE

        def setter(value: Any) -> None:
            field.setValue(round(float(value) * _SCALE))

        self._bind(field, path, label, getter, setter, field.value_changed)
        self._content.addWidget(SettingCard(title, desc, field, self._theme))

    # -- live application ---------------------------------------------------

    def _pending_changed(self) -> None:
        """Толкнуть облик (сохранённое + несохранённые правки) в предпросмотр сразу."""
        preview = getattr(self, "preview", None)
        if preview is not None:
            preview.apply_overlay(self._effective_overlay())

    def _effective_overlay(self) -> OverlayConfig:
        """Секция ``overlay`` с наложенными «ожидающими» правками — без валидации.

        ``model_copy`` минует валидатор, поэтому промежуточное значение (например,
        пустой акцент в момент правки) никогда не роняет предпросмотр.
        """
        overlay = self._manager.settings.overlay
        updates = {
            path.split(".", 1)[1]: value
            for path, value in self._pending.items()
            if path.startswith("overlay.")
        }
        if not updates:
            return overlay
        return overlay.model_copy(update=updates)

    def _apply_preset(self, name: str) -> None:
        """Выставить связанный набор полей разом (сразу сохранить, минуя дебаунс)."""
        preset = OVERLAY_PRESETS.get(name)
        if preset is None:
            return
        self._manager.apply({f"overlay.{key}": value for key, value in preset.items()})


register_tab("overlay", OverlayTab)
