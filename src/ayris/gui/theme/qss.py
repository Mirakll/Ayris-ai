"""QSS generation and live theme application."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QStyleFactory, QWidget

from ayris.gui.theme.system import SystemThemeWatcher
from ayris.gui.theme.tokens import Theme, bundled_theme_path, load_theme

__all__ = [
    "DEFAULT_QSS_TEMPLATE",
    "ThemeManager",
    "render_qss",
    "resolve_font_family",
]

ThemeMode = Literal["dark", "light", "system"]
_TOKEN = re.compile(r"{{\s*(color|metric|typography)\.([a-zA-Z0-9_]+)\s*}}")

DEFAULT_QSS_TEMPLATE = """
* {
    color: {{color.text_primary}};
    font-family: "{{typography.family}}";
    font-size: {{typography.body_size}}px;
}
QWidget { background-color: {{color.background}}; }
QLabel, QSlider, QRadioButton, QCheckBox, SliderField, ToggleSwitch, DownloadProgress,
ModelManager { background: transparent; }
/* Layout-only containers opt out of the window fill so they don't paint a dark
   rectangle on top of a card. */
QWidget[transparent="true"], QFrame[transparent="true"] { background: transparent; }
QWidget[card="true"], QFrame[card="true"] {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
}
QLabel[role="secondary"] { color: {{color.text_secondary}}; }
QLabel[role="muted"] {
    color: {{color.text_muted}};
    font-size: {{typography.caption_size}}px;
}
QLabel[role="h1"] {
    font-size: {{typography.h1_size}}px;
    font-weight: {{typography.weight_bold}};
}
QLabel[role="h2"] {
    font-size: {{typography.h2_size}}px;
    font-weight: {{typography.weight_bold}};
}
/* Заголовок третьего уровня (название строки/карточки, например действие в
   таблице хоткеев): размер body, но насыщеннее — отделяется от описания под ним
   не только цветом, но и весом. Завершает шкалу h1 → h2 → h3, которой раньше не
   хватало последней ступени. */
QLabel[role="h3"] {
    font-weight: {{typography.weight_medium}};
}
QLabel[status="warning"] { color: {{color.warning}}; }
QListWidget#settingsSidebar, QListWidget {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
    outline: none;
    padding: {{metric.spacing_xs}}px;
}
QListWidget#settingsSidebar::item {
    border-radius: {{metric.radius_md}}px;
    padding-left: {{metric.spacing_md}}px;
}
QListWidget#settingsSidebar::item:hover { background-color: {{color.surface_highlight}}; }
QListWidget#settingsSidebar::item:selected {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
}
/* Дерево команд: та же поверхность-карта, что и сайдбар, без системной рамки
   выделения и без квадрата за узлом. Ветви и отступ рисует Qt, цвет наследуем. */
QTreeView {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
    outline: none;
    padding: {{metric.spacing_xs}}px;
    show-decoration-selected: 1;
}
QTreeView::item {
    border-radius: {{metric.radius_md}}px;
    padding: {{metric.spacing_xs}}px {{metric.spacing_sm}}px;
    min-height: {{metric.control_height}}px;
}
QTreeView::item:hover { background-color: {{color.surface_highlight}}; }
QTreeView::item:selected {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
}
/* Ветви. У листа-команды колонку раскрытия обнуляем полностью: ни фона, ни
   «ниточек»-линий, ни картинки-заглушки — именно её Fusion на Windows оставляет
   серым квадратом слева от команды. У папки (has-children) стрелку раскрытия не
   трогаем: её по-прежнему рисует стиль, иначе дерево не свернуть. Выделение и
   ховер в колонке ветви тоже прозрачны, чтобы за узлом не появлялся прямоугольник. */
QTreeView::branch { background: transparent; }
QTreeView::branch:!has-children,
QTreeView::branch:has-siblings:!adjoins-item,
QTreeView::branch:has-siblings:adjoins-item,
QTreeView::branch:!has-children:!has-siblings:adjoins-item {
    background: transparent;
    border-image: none;
    image: none;
}
QTreeView::branch:selected,
QTreeView::branch:hover {
    background: transparent;
}
/* Вкладки редактора команд («Обзор», «Действия», …). Панель контента без рамки и
   фона: иначе штатный QTabWidget обводил бы прямоугольником всю область под
   вкладками (палитра + холст + параметры) — та самая «квадратная рамка вокруг окна».
   Полоса вкладок — плоская, активная вкладка подчёркнута акцентом. */
QTabWidget::pane {
    border: none;
    background: transparent;
    top: 0;
}
QTabBar { background: transparent; }
QTabBar::tab {
    background: transparent;
    color: {{color.text_secondary}};
    border: none;
    border-bottom: 2px solid transparent;
    padding: {{metric.spacing_sm}}px {{metric.spacing_md}}px;
    margin-right: {{metric.spacing_xs}}px;
    font-weight: {{typography.weight_medium}};
}
QTabBar::tab:hover { color: {{color.text_primary}}; }
QTabBar::tab:selected {
    color: {{color.text_primary}};
    border-bottom: 2px solid {{color.accent}};
}
QTabBar::tab:focus { outline: none; }
QPushButton, QLineEdit, QSpinBox, QComboBox {
    min-height: {{metric.control_height}}px;
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
    background-color: {{color.surface}};
    padding-left: {{metric.spacing_md}}px;
    padding-right: {{metric.spacing_md}}px;
}
QPushButton:hover, QLineEdit:hover, QSpinBox:hover, QComboBox:hover {
    border-color: {{color.accent_hover}};
    background-color: {{color.surface_highlight}};
}
QPushButton:pressed { border-color: {{color.accent_pressed}}; }
QPushButton:focus, QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: {{metric.focus_width}}px solid {{color.focus}};
}
QPushButton:disabled, QLineEdit:disabled, QSpinBox:disabled {
    color: {{color.text_muted}};
    border-color: {{color.accent_disabled}};
}
/* Combobox: шеврон рисует сам ThemedComboBox (widgets/combo_box.py) в чипе
   справа. Стилю запрещаем рисовать любую стрелку и дроп-даун — иначе на дробном
   DPI (125 %, 150 %) QStyleSheetStyle всё равно нарисует свою стрелку по центру
   поля поверх нашей: те самые «две стрелки». Ноль ширины и image: none гасят её
   для любого QComboBox, в том числе на будущих страницах. */
QComboBox {
    min-height: {{metric.control_height_lg}}px;
    padding-right: {{metric.spacing_xs}}px;
}
QComboBox::drop-down {
    width: 0;
    border: none;
    background: transparent;
}
QComboBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
/* ThemedComboBox — единственный, кто рисует стрелку; кормим ему цвета и размеры
   Qt-свойствами, чтобы он оставался темозависимым. */
ThemedComboBox {
    qproperty-chipColor: {{color.surface_highlight}};
    qproperty-chipColorActive: {{color.accent}};
    qproperty-arrowColor: {{color.text_secondary}};
    qproperty-arrowColorActive: {{color.on_accent}};
    qproperty-arrowColorDisabled: {{color.text_muted}};
    qproperty-chipSize: {{metric.control_height}};
    qproperty-chipInset: {{metric.spacing_xs}};
    qproperty-chipRadius: {{metric.radius_sm}};
    qproperty-arrowSize: {{metric.icon_sm}};
}
/* Спинбокс: штатный стиль рисует справа два крохотных стрелка-кнопки друг над
   другом — они читаются как «двойной переключатель» и, как и стрелка комбобокса,
   двоятся на дробном DPI. ThemedSpinBox (widgets/spin_box.py) убирает обе кнопки
   и сам рисует один вертикальный чип ▴/▾. Тушим кнопки на любом QSpinBox, чтобы
   не проступали под чипом. */
QSpinBox::up-button, QSpinBox::down-button {
    width: 0;
    border: none;
    background: transparent;
}
QSpinBox::up-arrow, QSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
}
/* ThemedSpinBox — единственный, кто рисует стрелки; кормим ему цвета и размеры
   Qt-свойствами, как ThemedComboBox. Чип уже вертикальной (два шеврона друг над
   другом), поэтому он и уже комбобоксового. */
ThemedSpinBox {
    qproperty-chipColor: {{color.surface_highlight}};
    qproperty-chipColorActive: {{color.accent}};
    qproperty-arrowColor: {{color.text_secondary}};
    qproperty-arrowColorActive: {{color.on_accent}};
    qproperty-arrowColorDisabled: {{color.text_muted}};
    qproperty-chipSize: {{metric.icon_md}};
    qproperty-chipInset: {{metric.spacing_xs}};
    qproperty-chipRadius: {{metric.radius_sm}};
    qproperty-arrowSize: {{metric.icon_sm}};
}
/* Всплывающий список комбобокса. Углы прямые (radius 0): скруглённый список
   лежит внутри отдельного окна-контейнера, которое Qt рисует квадратным и
   непрозрачным, и за скруглением проступали серые уголки этого контейнера.
   Квадратный список закрывает контейнер целиком — стыковаться нечему. Контейнер
   вдобавок translucent+frameless (widgets/combo_box.py) как защита от рамки и
   тени. Обводка — акцентный фиолетовый, в меру яркий. */
QComboBox QAbstractItemView {
    background: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.accent}};
    border-radius: 0px;
    padding: {{metric.spacing_xs}}px;
    outline: none;
    selection-background-color: {{color.accent}};
    selection-color: {{color.on_accent}};
}
QPushButton[kind="primary"] {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
    border-color: {{color.accent}};
    font-weight: {{typography.weight_medium}};
}
QPushButton[kind="primary"]:hover {
    background-color: {{color.accent_hover}};
    border-color: {{color.accent_hover}};
}
QPushButton[kind="primary"]:pressed {
    background-color: {{color.accent_pressed}};
    border-color: {{color.accent_pressed}};
}
QPushButton[kind="danger"] {
    color: {{color.on_accent}};
    background-color: {{color.error}};
    border-color: {{color.error}};
    font-weight: {{typography.weight_medium}};
}
QPushButton[iconButton="true"] {
    min-width: {{metric.control_height}}px;
    max-width: {{metric.control_height}}px;
    padding: 0;
}
/* Сегмент истории редактора (отменить · версии · повторить): три кнопки слиты
   в один «островок» на поверхности, прикреплённый к верху холста, вместо
   разрозненных системных QToolButton без темы. */
QFrame[toolgroup="true"] {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
}
QFrame[toolgroup="true"] QToolButton {
    background: transparent;
    border: none;
    border-radius: {{metric.radius_sm}}px;
    padding: 0 {{metric.spacing_sm}}px;
    min-height: {{metric.control_height}}px;
    color: {{color.text_secondary}};
    font-weight: {{typography.weight_medium}};
}
QFrame[toolgroup="true"] QToolButton:hover {
    background-color: {{color.surface_highlight}};
    color: {{color.text_primary}};
}
QFrame[toolgroup="true"] QToolButton:pressed {
    background-color: {{color.accent}};
    color: {{color.on_accent}};
}
QFrame[toolgroup="true"] QToolButton:disabled {
    background: transparent;
    color: {{color.text_muted}};
}
QFrame[toolgroup="true"] QToolButton::menu-indicator {
    image: none;
    width: 0;
    height: 0;
}
/* Плавающая командная капсула нодового холста (вариант «Кинематограф»): «пульт»
   инструментов, парящий у нижнего края холста поверх нод. Фон и обводку несёт сама
   капсула; кнопки внутри — плоские, без своей рамки, с квадратной иконкой (iconButton).
   Разделители — тонкие вертикальные линии цвета границы. */
QFrame[capsule="true"] {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
}
QFrame[capsule="true"] QPushButton {
    background-color: transparent;
    border: none;
    border-radius: {{metric.radius_md}}px;
    color: {{color.text_secondary}};
    padding: 0;
}
QFrame[capsule="true"] QPushButton:hover {
    background-color: {{color.surface_highlight}};
    border: none;
    color: {{color.text_primary}};
}
QFrame[capsule="true"] QPushButton:pressed,
QFrame[capsule="true"] QPushButton:checked {
    background-color: {{color.accent}};
    border: none;
    color: {{color.on_accent}};
}
QFrame[capsule="true"] QPushButton:disabled {
    background-color: transparent;
    border: none;
    color: {{color.text_muted}};
}
QFrame[capsule="true"] QFrame[vline="true"] {
    background-color: {{color.border}};
    border: none;
}
/* Капсула теперь несёт и мостик редактора: слева — «островок» истории, справа —
   статус · «Тест» · «Сохранить». Базовое правило капсулы гасит любые QPushButton до
   плоских иконок (padding:0), поэтому текстовым кнопкам и акцентному «Сохранить»
   нужны точечные, более специфичные переопределения — иначе «Тест»/«Сохранить»
   схлопнутся без полей, а «Сохранить» потеряет заливку. */
QFrame[capsule="true"] QPushButton[textButton="true"] {
    padding: 0 {{metric.spacing_md}}px;
    min-height: {{metric.control_height}}px;
}
QFrame[capsule="true"] QPushButton[kind="primary"] {
    color: {{color.on_accent}};
    background-color: {{color.accent}};
}
QFrame[capsule="true"] QPushButton[kind="primary"]:hover {
    background-color: {{color.accent_hover}};
    color: {{color.on_accent}};
}
QFrame[capsule="true"] QPushButton[kind="primary"]:pressed {
    background-color: {{color.accent_pressed}};
    color: {{color.on_accent}};
}
/* Вложенный «островок» истории внутри капсулы читается плоско: своя поверхность и
   рамка убираются, чтобы не было «коробки в коробке» на фоне капсулы. Кнопки внутри
   островка сохраняют своё поведение — их задают отдельные toolgroup-правила. */
QFrame[capsule="true"] QFrame[toolgroup="true"] {
    background: transparent;
    border: none;
}
/* Плоская ссылка (например «← К списку команд»): без рамки и фона, левый край
   без отступа — так стрелка встаёт вровень с левым краем панели под кнопкой. */
QPushButton[link="true"] {
    background: transparent;
    border: none;
    padding: 0;
    min-height: 0;
    color: {{color.text_secondary}};
    text-align: left;
}
QPushButton[link="true"]:hover {
    background: transparent;
    border: none;
    color: {{color.accent}};
}
QPushButton[link="true"]:pressed { border: none; }
QPushButton[link="true"]:focus { border: none; }
/* Вкладки браузерной верхней панели («Обзор … История») и переключатель «Список /
   Ноды» рядом: плоская «пилюля» вместо рамки-кнопки, чтобы весь ряд читался как один
   браузерный таб-бар. Активная вкладка залита подсветкой с акцентным текстом —
   приближение к `.tab.active` макета (там accent 22 %; отдельного мягкого токена нет). */
QPushButton[navTab="true"] {
    background: transparent;
    border: none;
    border-radius: {{metric.radius_md}}px;
    padding: {{metric.spacing_xs}}px {{metric.spacing_md}}px;
    min-height: 0;
    color: {{color.text_secondary}};
    font-weight: {{typography.weight_medium}};
}
QPushButton[navTab="true"]:hover {
    background: {{color.surface_highlight}};
    color: {{color.text_primary}};
}
QPushButton[navTab="true"]:checked {
    background: {{color.surface_highlight}};
    color: {{color.accent}};
}
QPushButton[navTab="true"]:focus { border: none; }
/* «Пилюля» состояния команды в верхней панели: обведённая капсула цветом состояния
   (включена — успех, выключена — приглушённая), как зелёный `.badge` в макете, а не
   просто цветной текст. */
QLabel[statePill="on"], QLabel[statePill="off"] {
    font-size: {{typography.caption_size}}px;
    font-weight: {{typography.weight_medium}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_sm}}px;
    padding: 1px {{metric.spacing_xs}}px;
}
QLabel[statePill="on"] { color: {{color.success}}; border-color: {{color.success}}; }
QLabel[statePill="off"] { color: {{color.text_muted}}; }
/* Тонкая разделительная линия (например под заголовком инспектора параметров):
   один горизонтальный штрих цветом границы, без рамки-рельефа QFrame. */
QFrame[rule="true"] {
    border: none;
    background: {{color.border}};
    max-height: {{metric.border_width}}px;
    min-height: {{metric.border_width}}px;
}
QFrame[notice="true"] {
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
}
QFrame[status="info"] { border-color: {{color.info}}; }
QFrame[status="warning"] { border-color: {{color.warning}}; }
QFrame[status="error"] { border-color: {{color.error}}; }
QFrame[status="success"] { border-color: {{color.success}}; }
/* Прогресс-бар загрузки модели: тема, не системный серый квадрат. */
QProgressBar {
    min-height: {{metric.spacing_md}}px;
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_sm}}px;
    background: {{color.surface_highlight}};
    text-align: center;
    color: {{color.text_primary}};
}
QProgressBar::chunk {
    background: {{color.accent}};
    border-radius: {{metric.radius_sm}}px;
}
/* Статус модели: цвет несёт смысл, фон прозрачный — никакого квадрата за текстом. */
QLabel[badge="true"] {
    background: transparent;
    color: {{color.text_secondary}};
    font-size: {{typography.caption_size}}px;
    font-weight: {{typography.weight_medium}};
}
QLabel[badge="success"] { color: {{color.success}}; }
QLabel[badge="warning"] { color: {{color.warning}}; }
QLabel[badge="error"] { color: {{color.error}}; }
QLabel[badge="info"] { color: {{color.info}}; }
QLabel[badge="muted"] { color: {{color.text_muted}}; }
/* Чип тега в редакторе команды: скруглённая «пилюля» на подсвеченной поверхности,
   без квадрата за текстом. */
QPushButton[chip="true"] {
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_lg}}px;
    padding-left: {{metric.spacing_md}}px;
    padding-right: {{metric.spacing_md}}px;
    color: {{color.text_secondary}};
    min-height: {{metric.control_height}}px;
}
QPushButton[chip="true"]:hover {
    border-color: {{color.error}};
    color: {{color.text_primary}};
}
/* Поле с ошибкой ввода (пустое или занятое имя команды): рамка цветом ошибки. */
QLineEdit[invalid="true"], QSpinBox[invalid="true"] {
    border-color: {{color.error}};
}
QSlider::groove:horizontal {
    height: {{metric.spacing_xs}}px;
    background: {{color.surface_highlight}};
    border-radius: {{metric.radius_sm}}px;
}
QSlider::sub-page:horizontal {
    background: {{color.accent}};
    border-radius: {{metric.radius_sm}}px;
}
QSlider::handle:horizontal {
    width: {{metric.spacing_md}}px;
    margin: -6px 0;
    border-radius: 6px;
    background: {{color.accent}};
}
/* Полосы прокрутки живут внутри скруглённых карт-контейнеров (дерево, список
   настроек — radius_lg). Трек красим цветом карты (surface), а не тёмным фоном
   окна: иначе за бегунком и в его торцах проступала тёмная («чёрная») колонка —
   это универсальное правило QWidget заливало полосу фоном окна. Отступ сверху и
   снизу держит бегунок в прямой части борта, подальше от скруглённого угла. */
QScrollBar:vertical {
    width: {{metric.spacing_md}}px;
    background: {{color.surface}};
    border-radius: 6px;
    margin: 0;
}
QScrollBar:horizontal {
    height: {{metric.spacing_md}}px;
    background: {{color.surface}};
    border-radius: 6px;
    margin: 0;
}
/* Бегунок отступает от концов через СВОЙ margin, а не через margin полосы: зазор
   бегунка заливается фоном полосы (surface), тогда как margin самой полосы
   пропускал бы тёмный фон окна («чёрный» торец). Отступ ≈ радиусу угла карты,
   чтобы бегунок держался в прямой части борта и не заезжал на скругление. */
QScrollBar::handle:vertical {
    min-height: {{metric.control_height_lg}}px;
    margin: {{metric.radius_lg}}px 2px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent_hover}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::handle:horizontal {
    min-width: {{metric.control_height_lg}}px;
    margin: 2px {{metric.radius_lg}}px;
    border-radius: 6px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent_hover}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 {{color.accent}}, stop:1 {{color.accent_pressed}});
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
    height: 0;
    background: transparent;
    border: none;
}
/* Дорожка над и под бегунком — цветом карты, а не прозрачная: прозрачная
   пропускала тёмный фон окна из-под полосы, и он читался «чёрной» колонкой за
   бегунком и в его торцах. */
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: {{color.surface}};
}
QToolTip {
    color: {{color.text_primary}};
    background-color: {{color.surface_highlight}};
    border: {{metric.border_width}}px solid {{color.border}};
    padding: {{metric.spacing_sm}}px;
}
QMenu {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
    padding: {{metric.spacing_xs}}px;
    font-weight: {{typography.weight_medium}};
}
QMenu::item {
    padding: 9px 18px 9px 12px;
    border: 1px solid transparent;
    border-left: 3px solid transparent;
    border-radius: {{metric.radius_sm}}px;
    color: {{color.text_primary}};
}
QMenu::item:selected {
    background-color: {{color.surface_highlight}};
}
QMenu::item:checked {
    color: {{color.accent}};
    border-left: 3px solid {{color.accent}};
}
QMenu::item:checked:selected {
    background-color: {{color.surface_highlight}};
}
QMenu::item:disabled {
    color: {{color.text_muted}};
}
QMenu::icon {
    padding-left: {{metric.spacing_sm}}px;
}
QMenu::indicator {
    width: 0px;
    height: 0px;
}
QMenu::separator {
    height: {{metric.border_width}}px;
    background: {{color.border}};
    margin: {{metric.spacing_sm}}px {{metric.spacing_sm}}px;
}
QMenu::right-arrow {
    width: {{metric.spacing_sm}}px;
    height: {{metric.spacing_sm}}px;
    margin-right: {{metric.spacing_md}}px;
}
/* Палитра блоков (каталог задачи 33) — карточками, как в браузерном макете: панель с
   поверхностью и рамкой, заголовки категорий капсом и строки «значок + название +
   описание». Наводка подсвечивает только доступные карточки; недоступные приглушены. */
QWidget[blockPalette="true"] {
    background-color: {{color.surface}};
    border: {{metric.border_width}}px solid {{color.border}};
    border-radius: {{metric.radius_md}}px;
}
QLabel[catTitle="true"] {
    color: {{color.text_muted}};
    font-size: 10px;
    font-weight: {{typography.weight_medium}};
    padding: {{metric.spacing_sm}}px {{metric.spacing_xs}}px {{metric.spacing_xs}}px;
}
QFrame[catItem="true"] {
    border: none;
    border-radius: {{metric.radius_md}}px;
}
QFrame[catItem="true"][interactive="true"]:hover {
    background-color: {{color.surface_highlight}};
}
QFrame[catItem="true"] QLabel[ciName="true"] {
    color: {{color.text_primary}};
    font-size: 13px;
    font-weight: {{typography.weight_medium}};
}
QFrame[catItem="true"] QLabel[ciDesc="true"] {
    color: {{color.text_secondary}};
    font-size: 11px;
}
QFrame[catItem="true"][interactive="false"] QLabel[ciName="true"],
QFrame[catItem="true"][interactive="false"] QLabel[ciDesc="true"] {
    color: {{color.text_muted}};
}
""".strip()


def resolve_font_family(theme: Theme) -> str:
    """Choose the first installed family, always ending at a system fallback."""
    installed = {family.casefold(): family for family in QFontDatabase.families()}
    for candidate in (theme.typography.family, *theme.typography.fallbacks, "Segoe UI"):
        if candidate.casefold() in installed:
            return installed[candidate.casefold()]
    return QFont().defaultFamily()


def render_qss(theme: Theme, template: str = DEFAULT_QSS_TEMPLATE, *, scale: float = 1.0) -> str:
    """Substitute semantic tokens into QSS and reject unknown placeholders."""
    font_family = resolve_font_family(theme)

    def replace(match: re.Match[str]) -> str:
        group, name = match.groups()
        if group == "color":
            return theme.color(name)
        if group == "metric":
            return str(theme.metric(name, scale=scale))
        value = theme.type_value(name)
        if name == "family":
            return font_family
        if isinstance(value, tuple):
            return ", ".join(value)
        if isinstance(value, int | float) and name.endswith("_size"):
            return str(max(1, round(value * scale)))
        return str(value)

    rendered = _TOKEN.sub(replace, template)
    unresolved = _TOKEN.search(rendered)
    if unresolved is not None:
        raise KeyError(f"Не удалось подставить токен {unresolved.group(0)}")
    return rendered


class ThemeManager(QObject):
    """Own the current theme, stylesheet, system mode and DPI preview scale."""

    theme_changed = Signal(object)

    def __init__(
        self,
        application: QApplication,
        *,
        theme_dir: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._application = application
        # Base every widget on Fusion. The native Windows 11 style paints its own
        # combobox arrow that QSS cannot suppress, so a themed chevron leaves the
        # native one behind — two arrows on every combo. Fusion is a QSS-friendly
        # base the stylesheet fully owns, so the chevron is the only arrow and the
        # rest of the dark theme applies cleanly too.
        fusion = QStyleFactory.create("Fusion")
        if fusion is not None:
            application.setStyle(fusion)
        self._theme_dir = theme_dir
        self._mode: ThemeMode = "dark"
        self._scale = 1.0
        self._theme = self._load_named("dark_purple")
        self._watcher = SystemThemeWatcher(parent=self)
        self._watcher.changed.connect(self._system_changed)

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def scale(self) -> float:
        return self._scale

    def metric(self, name: str) -> int:
        return self._theme.metric(name, scale=self._scale)

    def set_scale(self, scale: float) -> None:
        if scale <= 0:
            raise ValueError("Масштаб интерфейса должен быть больше нуля")
        if scale == self._scale:
            return
        self._scale = scale
        self._apply()

    def set_mode(self, mode: ThemeMode) -> None:
        if mode not in ("dark", "light", "system"):
            raise ValueError(f"Неизвестный режим темы: {mode}")
        self._mode = mode
        if mode == "system":
            self._watcher.start()
            selected = self._watcher.current
        else:
            self._watcher.stop()
            selected = mode
        self._theme = self._load_named("light" if selected == "light" else "dark_purple")
        self._apply()

    def apply(self) -> None:
        self._apply()

    def _load_named(self, name: str) -> Theme:
        path = self._theme_dir / f"{name}.json" if self._theme_dir else bundled_theme_path(name)
        return load_theme(path)

    def _system_changed(self, mode: str) -> None:
        if self._mode != "system":
            return
        self._theme = self._load_named("light" if mode == "light" else "dark_purple")
        self._apply()

    def _apply(self) -> None:
        qss = render_qss(self._theme, scale=self._scale)
        self._application.setStyleSheet(qss)
        font = QFont(resolve_font_family(self._theme))
        font.setPixelSize(max(1, round(self._theme.typography.body_size * self._scale)))
        self._application.setFont(font)
        style = self._application.style()
        if style is not None:
            for widget in self._application.allWidgets():
                style.unpolish(widget)
                style.polish(widget)
                widget.repaint()
        self.theme_changed.emit(self._theme)


def screen_scale(widget: QWidget) -> float:
    """DPI ratio for manual painting; Qt scales QSS logical units itself."""
    screen = widget.screen()
    return screen.logicalDotsPerInch() / 96.0 if screen is not None else 1.0
