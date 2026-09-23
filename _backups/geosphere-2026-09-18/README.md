# Бэкап: AEGIS geo-сфера (point cloud) — 2026-09-18

Снимок первого готового варианта точечной сферы-ассистента (плотное облако,
bokeh-точки, приподнятая камера, палитры циан/золото, пять состояний).

Соответствует рабочей версии в `src/ayris/gui/widgets/geosphere/` на 2026-09-18.

## Что внутри
- `src/geosphere/` — полный пакет виджета (`palette`, `geometry`, `states`,
  `field`, `renderer`, `widget`, `demo`, `__init__`).
- `tests/test_geosphere.py` — тесты (12 passed).

## Как вернуться к этому варианту
Из корня проекта:

```bash
rm -rf src/ayris/gui/widgets/geosphere
cp -r _backups/geosphere-2026-09-18/src/geosphere src/ayris/gui/widgets/geosphere
cp _backups/geosphere-2026-09-18/tests/test_geosphere.py tests/unit/test_geosphere.py
```

Демо:

```bash
_tools/venv/Scripts/python.exe -m ayris.gui.widgets.geosphere.demo
```
