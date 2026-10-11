# QML-острова внутри QWidget-окна

Доказательство и заготовка для переноса редизайна в реальное приложение.
Выбранный путь — **вариант 2: Qt Quick/QML-острова** (`QQuickWidget` в том же
QWidget-окне), а НЕ эмуляция CSS через QSS. Причина: QML нативно даёт то, ради
чего в вебе тянутся к «CSS-классу возможностей».

## Что здесь лежит

- `theme_bridge.py` — мост «дизайн-токены → QML». Читает те же
  `resources/themes/*.json` тем же загрузчиком (`ayris.gui.theme.tokens`), что и
  прод, и отдаёт токены в QML как свойства одного объекта `theme`
  (`theme.accent`, `theme.radiusXl`, `theme.animNormal`, …). Смена темы →
  сигнал `changed` → QML сам перепривязывает все биндинги.
- `qml/GlassIsland.qml` — стекло/aurora: живой фон → **настоящий блюр на GPU**
  (`MultiEffect`, Qt 6.5+) → полупрозрачная карточка; наведение поднимает её
  через `Behavior on scale`.
- `qml/NeonIsland.qml` — неон: сфера с halo-свечением (блюр яркой копии под
  чётким орбом), дыхание, клик = bounce + вспышка (`Behavior`/анимации).
- `island_demo.py` — QWidget-окно с двумя островами и нативным QComboBox смены
  темы (доказывает синхронизацию QWidget↔QML через общий `theme`).
- `out-0.png` / `out-1.png` — снимки безоконного рендера (proof, что блюр и
  свечение считаются на GPU, а не эмулируются).

## Запуск

Живое окно:

```bash
_tools/venv/Scripts/python.exe design/qt-islands/island_demo.py
```

Снимок без окна (для CI/скриншотов):

```bash
_tools/venv/Scripts/python.exe design/qt-islands/island_demo.py --shot design/qt-islands/out
```

Проверено на PySide6 6.8.1 (QtQuick + Quick3D доступны). Оба острова
отрисовались офскрин в PNG — блюр и свечение настоящие.

## Что нативно даёт QML (и чего нет в QSS)

| Нужно для редизайна | QML | QSS |
| --- | --- | --- |
| Токены как переменные | `property color accent` + биндинги | нет (только подстановка при сборке) |
| Transitions | `Behavior on x { NumberAnimation {} }` | нет (только QPropertyAnimation вручную) |
| Blur / drop-shadow / glow | `MultiEffect` на GPU | грубо и дорого (QGraphicsEffect на CPU) |
| Произвольный GLSL (aurora, глитч) | `ShaderEffect` | нет |
| 3D | `Quick3D` | нет |

`:has()` / `oklch()` в QML тоже нет (это не CSS), но эффекты несоизмеримо богаче.

## Карта токен → свойство

`theme_bridge.py` уже мапит схему `ColorTokens`/`MetricTokens`:
`background, surface, surfaceHi, border, accent, accentHover, accentPressed,
textPrimary, textSecondary, textMuted, success, warning, error, info,
roleAction, roleSound, onAccent` (цвета) и `radiusMd/Lg/Xl`,
`animFast/Normal/Slow` (метрики). Добавить недостающие — одна строка `Property`.

## Границы (важно)

- **Острова — точечно.** Стекло-панели (Стекло/Aurora) и неон-сцену — как
  `QQuickWidget`-острова в QWidget-дашборде. Остальной каркас (навигация,
  карточки-настройки, формы) может остаться QWidget + QSS — это дёшево и уже есть.
- **Нодовый холст «Команды» не трогаем.** Он остаётся `QGraphicsView`, только
  перекрас под стиль (разметка и элементы 1:1). Не переносить в QML.
- **3D-аватар (Quick3D) — только возможная будущая функция.** Сейчас нигде не
  применять; герой сцены — абстрактная сфера (в варианте AiRI — в тёплом
  коралловом ключе). Quick3D-остров встанет сюда же, если появится.
- **`ShaderEffect` в Qt 6 требует скомпилированный `.qsb`** (RHI, не инлайновый
  GLSL как в Qt 5): `.frag` → `pyside6-qsb`. В этой заготовке неон сделан на
  `MultiEffect`, чтобы работать из коробки без шаг-сборки шейдеров; aurora на
  `ShaderEffect` — следующий шаг, когда добавим `.qsb`.

## Грабли (из памяти проекта)

- Прозрачные/translucent-острова и offscreen-грэб — рендерить через кадр Quick,
  не через `grab()` виджета; здесь `QQuickWidget.grabFramebuffer()`.
- Пути с кириллицей ломают некоторые нативные либы — здесь не задействованы,
  но QML-файлы грузим `QUrl.fromLocalFile` (без ручной сборки URL).
