# Сборка Windows

`ayris.manifest` фиксирует безопасный запуск `asInvoker`, `uiAccess=false` и
Per-Monitor DPI Awareness v2. При сборке Nuitka его нужно передать линкеру как
ресурс приложения (resource type `RT_MANIFEST`, id `1`), либо внедрить в готовый
`ayris.exe` штатным `mt.exe -manifest build\ayris.manifest
-outputresource:dist\ayris.exe;#1`. Задача 71 должна выполнять этот шаг после
Nuitka и до подписи бинарника; `utils/dpi.py` остаётся страховкой запуска из исходников.

## Локальная сборка

Движок llama.cpp вшивается в дистрибутив — локальная модель работает из коробки,
без Ollama и отдельного `llama-server.exe`. Значит, сборочное окружение обязано
его нести. CPU-колеса `llama-cpp-python` нет на PyPI (там только sdist, который
пришлось бы компилировать), поэтому ставим его с готового индекса:

```
pip install -e ".[build]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m build.nuitka_build --output-dir dist
```

`build.nuitka_build` до запуска Nuitka проверяет, что и Nuitka, и движок
`llama_cpp` на месте, и без них останавливается с понятным сообщением (коды
возврата 2 и 3). Веса моделей (GGUF) в сборку не входят — они всегда качаются по
требованию через менеджер моделей. То же самое делает `.github/workflows/release.yml`.
