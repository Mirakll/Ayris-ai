# Сборка Windows

`ayris.manifest` фиксирует безопасный запуск `asInvoker`, `uiAccess=false` и
Per-Monitor DPI Awareness v2. При сборке Nuitka его нужно передать линкеру как
ресурс приложения (resource type `RT_MANIFEST`, id `1`), либо внедрить в готовый
`ayris.exe` штатным `mt.exe -manifest build\ayris.manifest
-outputresource:dist\ayris.exe;#1`. Задача 71 должна выполнять этот шаг после
Nuitka и до подписи бинарника; `utils/dpi.py` остаётся страховкой запуска из исходников.
