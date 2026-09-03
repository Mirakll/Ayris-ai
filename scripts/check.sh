#!/usr/bin/env bash
# Ayris — все проверки одним вызовом, на любой из двух машин.
#
# Что здесь происходит и почему именно так:
#
# * Тесты уходят в фон, линтеры идут поверх них. Полный прогон — самая долгая
#   часть, и ждёт он не процессор, а порождённые процессы и таймеры, поэтому
#   ruff, black и mypy успевают закончиться внутри него бесплатно.
# * Тесты про время (маркер `slow`, 13 штук) идут вторым вызовом и
#   последовательно, когда всё остальное уже стихло. Под `-n` они делят ядро с
#   пятью процессами, и «уложился в N мс» становится броском монеты — красный
#   цвет должен означать баг, а не занятую машину.
# * Кроме тестов проверяется то, что тесты проверить не могут: окружение
#   совпадает с пинами (`verify_pins.py`), приложение отвечает на `--version`,
#   в коде нет секретов. Первые два ловят ровно те поломки, которые тесты
#   проходят молча: подмену версии библиотеки и упавший импорт в `__main__`.
# * Вывод сжат до строки на проверку; разворачивается только упавшая.
#
# Использование:
#   scripts/check.sh                        всё: линтеры + тесты + окружение
#   scripts/check.sh tests/unit/test_x.py   линтеры + только эти тесты
#   scripts/check.sh -k "wake word"         любые аргументы уходят в pytest
#   scripts/check.sh --lint                 только ruff/black/mypy и окружение
#   scripts/check.sh --tests                только pytest
#   scripts/check.sh --fmt                  переформатировать black-ом на месте
#   scripts/check.sh --seq                  тесты последовательно, как в CI
#
# Параллель и разделение на два прогона включаются только на полном прогоне: на
# одном тест-файле старт воркеров дороже самих тестов.
#
# Переменные: AYRIS_PY (питон проекта, если он не там, где скрипт его ищет),
# AYRIS_CHECK_JOBS (процессов pytest, по умолчанию ядра минус одно),
# AYRIS_CHECK_DIST (load или loadfile, по умолчанию load),
# AYRIS_CHECK_TAIL (строк от упавшей проверки, 120).
#
# Про маркеры: `hardware` (микрофон и колонки) требует человека рядом и здесь не
# запускается — `pytest -m hardware` руками. `network` и `models`, наоборот,
# запускаются: на этой машине есть и сеть, и скачанные веса, а в CI network не
# идёт нигде, чтобы падение чужого сайта не красило четыре джоба. То есть
# единственный прогон тестов, которые правда ходят в интернет, — вот этот.
# Чего прогон не покрыл, он говорит в конце сам, а не оставляет это в «skipped».
#
# Живых облачных запросов здесь нет и быть не должно: тесты облачных STT/TTS
# работают на httpx.MockTransport, ключи им не нужны. Настоящие ключи проверяет
# scripts/check-keys.ps1 на машине пользователя, и в этот скрипт он не входит.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1

# Windows под Git Bash или linux-песочница: отличаются они не только питоном, а
# и тем, нужны ли обходные пути для временных каталогов (см. pt_common).
case "$(uname -s)" in
    MINGW* | MSYS* | CYGWIN*) WINDOWS=1 ;;
    *) WINDOWS=0 ;;
esac

# Питон проекта, в порядке «задан руками → venv машины → портативный из
# песочницы». Системный питон не берётся: в нём нет ни PySide6, ни движков, и
# прогон в нём проверяет не то окружение, в котором работает приложение.
PY=""
for candidate in \
    ${AYRIS_PY:+"$AYRIS_PY"} \
    "$ROOT/_tools/venv/Scripts/python.exe" \
    "$ROOT/_tools/venv/bin/python3" \
    "$ROOT/_tools/python/bin/python3"; do
    if [[ -x $candidate ]]; then
        PY=$candidate
        break
    fi
done
if [[ -z $PY ]]; then
    echo "не нашёл питон проекта. Искал:" >&2
    echo "  _tools/venv/Scripts/python.exe   (машина разработчика)" >&2
    echo "  _tools/python/bin/python3        (песочница)" >&2
    echo "Задай свой: AYRIS_PY=/путь/к/python scripts/check.sh" >&2
    exit 2
fi

# Кириллица в выводе: без этого python на windows пишет в cp1251 и падает с
# UnicodeEncodeError на первом же сообщении об ошибке вместо того, чтобы его
# показать.
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
# Ни одного настоящего окна, как и в CI: иначе прогон разбрасывает по экрану
# окна Qt и крадёт фокус посреди работы.
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}

RNNOISE="$ROOT/_tools/rnnoise/librnnoise.so"
# Наследие песочницы: там библиотеку приходилось показывать переменной. На
# windows её находит сам `rnnoise_library()` внутри пакета pyrnnoise, и файла по
# этому пути просто нет.
[[ -f $RNNOISE ]] && export AYRIS_RNNOISE_LIB="$RNNOISE"

cpus=$(nproc 2>/dev/null || echo 4)
JOBS=${AYRIS_CHECK_JOBS:-$((cpus > 2 ? cpus - 1 : 1))}
DIST=${AYRIS_CHECK_DIST:-load}
TAIL=${AYRIS_CHECK_TAIL:-120}
RUN_LINT=1
RUN_TESTS=1
FMT=0
PYTEST_ARGS=()

for arg in "$@"; do
    case $arg in
        --lint) RUN_TESTS=0 ;;
        --tests) RUN_LINT=0 ;;
        --fmt) FMT=1 ;;
        --seq) JOBS=1 ;;
        -h | --help)
            sed -n '19,29p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

# Полный прогон — это прогон без аргументов. Только на нём имеет смысл и
# параллель, и отдельный последовательный проход по тестам про время.
FULL=$((${#PYTEST_ARGS[@]} == 0 ? 1 : 0))
((FULL)) || JOBS=1

LOGDIR=$(mktemp -d) || exit 2
trap 'rm -rf "$LOGDIR"' EXIT

FAILED=()
STARTED=$(date +%s)

# Прогнать проверку, показать одну строку при успехе и хвост лога при падении.
# Последняя строка вывода у всех инструментов — это как раз итог ("All checks
# passed!", "Success: no issues...", "4907 passed, 12 skipped", "окружение то,
# что заявлено").
#
# Имена проверок латиницей не из вкусовщины: printf выравнивает по байтам, и
# любое русское имя в этой колонке съезжает на длину своей кириллицы.
STAGE_N=0
stage() {
    local name=$1
    shift
    local log t0 spent
    STAGE_N=$((STAGE_N + 1))
    log="$LOGDIR/stage-$STAGE_N.log"
    t0=$(date +%s)
    if "$@" >"$log" 2>&1; then
        spent=$(($(date +%s) - t0))
        printf '%-8s ok      %4ds  %s\n' "$name" "$spent" "$(tail -n 1 "$log")"
    else
        spent=$(($(date +%s) - t0))
        printf '%-8s ПАДАЕТ  %4ds\n' "$name" "$spent"
        tail -n "$TAIL" "$log" | sed 's/^/  | /'
        FAILED+=("$name")
    fi
}

# Общие флаги pytest. Функцией, а не переменной: `--basetemp` нельзя
# переиспользовать между вызовами (второй падает с FileExistsError), поэтому
# каждому прогону нужен свой.
pt_common() {
    PT=(-q --tb=short -p no:cacheprovider)
    # Только для песочницы: там pytest падает в RecursionError на удалении
    # tmp_path в смонтированной папке проекта. На windows каталог по умолчанию
    # нормальный, и подсовывать ему msys-путь — только портить.
    ((WINDOWS)) || PT+=("--basetemp=/tmp/pt_${RANDOM}${RANDOM}_$$")
}

# Главный прогон уходит в фон: пока он идёт, линтеры успевают закончиться.
PYTEST_LOG=""
PYTEST_PID=""
PYTEST_START=0
pytest_start() {
    pt_common
    if ((FULL)); then
        # hardware — микрофон и колонки: нужен человек рядом и работающее
        # устройство. Здесь они не идут не из осторожности, а по факту: на
        # машине без подключённого микрофона оба падают COMError-ом «элемент не
        # найден», то есть красный цвет означал бы «микрофон отключён», а не
        # «код сломан». Руками: `pytest -m hardware`.
        # slow — тесты про время, вторым проходом ниже.
        PT+=(-m "not hardware and not slow")
        if ((JOBS > 1)); then
            if "$PY" -c 'import xdist' 2>/dev/null; then
                PT+=(-n "$JOBS" "--dist=$DIST")
            else
                echo "pytest-xdist не установлен, прогон последовательный:"
                echo "  $PY -m pip install pytest-xdist==3.8.0"
            fi
        fi
    fi
    PYTEST_LOG="$LOGDIR/pytest.log"
    PYTEST_START=$(date +%s)
    "$PY" -m pytest "${PT[@]}" "${PYTEST_ARGS[@]}" >"$PYTEST_LOG" 2>&1 &
    PYTEST_PID=$!
}

pytest_wait() {
    local spent code
    wait "$PYTEST_PID"
    code=$?
    spent=$(($(date +%s) - PYTEST_START))
    if ((code == 0)); then
        printf '%-8s ok      %4ds  %s\n' pytest "$spent" "$(tail -n 1 "$PYTEST_LOG")"
    else
        printf '%-8s ПАДАЕТ  %4ds\n' pytest "$spent"
        tail -n "$TAIL" "$PYTEST_LOG" | sed 's/^/  | /'
        FAILED+=(pytest)
    fi
}

((RUN_TESTS)) && pytest_start

if ((RUN_LINT)); then
    # Ровно те же цели, что в CI: src и tests. Расширять список нельзя, иначе
    # локальный прогон краснеет там, где CI зелёный, и наоборот.
    stage ruff "$PY" -m ruff check src tests
    if ((FMT)); then
        stage black "$PY" -m black src tests
    else
        stage black "$PY" -m black --check src tests
    fi
    if ((WINDOWS)); then
        stage mypy "$PY" -m mypy src
    else
        # Вне windows `.mypy_cache` в смонтированной папке не пишется:
        # PermissionError вместо проверки типов.
        stage mypy "$PY" -m mypy --cache-dir=/tmp/mypycache src
    fi
fi

((RUN_TESTS)) && pytest_wait

# Тесты про время — отдельно и в тишине: к этому моменту ни линтеров, ни
# воркеров pytest уже нет, поэтому их таймингам никто не мешает.
if ((RUN_TESTS)) && ((FULL)); then
    pt_common
    stage slow "$PY" -m pytest "${PT[@]}" -m slow
fi

if ((RUN_LINT)); then
    # Что установлено — то, что заявлено в requirements-ci*.txt. Тесты этого не
    # видят: они одинаково зелены и на numpy 1.26, и на 2.3, а приложение — нет.
    stage pins "$PY" scripts/verify_pins.py
    # Самый дешёвый способ поймать сломанный импорт в точке входа: тесты зовут
    # внутренности напрямую и `python -m ayris` не проверяют вообще. PYTHONPATH
    # — чтобы проверка работала и там, где пакет не установлен editable: она про
    # импорт `__main__`, а не про способ установки.
    stage smoke env PYTHONPATH=src "$PY" -m ayris --version
fi

# Куски склеены, чтобы git grep не нашёл сам этот файл.
SECRET_PAT='githu''b_pat_|gh''p_|sk-[A-Za-z0-9]{20}|BEGIN (RSA|OPENSSH|PRIVATE)'
FOUND=$(git grep -nIE "$SECRET_PAT" 2>/dev/null)
if [[ -n $FOUND ]]; then
    printf '%-8s ПАДАЕТ        похоже на секрет в коде\n' secrets
    echo "$FOUND" | sed 's/^/  | /'
    FAILED+=(secrets)
else
    printf '%-8s ok         0s  секретов в трекаемых файлах нет\n' secrets
fi

# Оставленный lock-файл не даст пользователю сделать коммит.
mapfile -t LOCKS < <(find .git -name '*.lock' 2>/dev/null)
if ((${#LOCKS[@]})); then
    rm -f "${LOCKS[@]}"
    echo "убрано lock-файлов git: ${#LOCKS[@]}"
fi

TOTAL=$(($(date +%s) - STARTED))

# Тесты на настоящих весах ищут пути в AYRIS_TEST_*, и без них тихо
# пропускаются. Сказать об этом один раз честнее, чем показывать «12 skipped» и
# оставлять человека гадать, что именно не проверено. Сам скрипт весов не
# качает: он ходит на huggingface за faster-whisper при каждом запуске и стоит
# полторы минуты, а в CI эти тесты и так идут отдельным джобом.
if ((RUN_TESTS)) && ((FULL)); then
    echo "не проверено здесь: hardware (микрофон и колонки) — pytest -m hardware"
    if [[ -z ${AYRIS_TEST_STT_MODEL:-} ]]; then
        echo "не проверено здесь: models (веса) — включить в этой оболочке:"
        echo "  eval \"\$($PY scripts/fetch_models.py --root _tools/models-ci --quiet)\""
    fi
fi

if ((${#FAILED[@]})); then
    echo "красное: ${FAILED[*]} (всего ${TOTAL}s)"
    [[ " ${FAILED[*]} " == *" black "* ]] && echo "форматирование чинится само: scripts/check.sh --fmt"
    [[ " ${FAILED[*]} " == *" pins "* ]] &&
        echo "окружение чинится так: $PY -m pip install -r requirements-ci.txt"
    exit 1
fi
echo "всё зелёное за ${TOTAL}s"
