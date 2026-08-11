#!/usr/bin/env bash
# Ayris - все проверки одним вызовом шелла.
#
# Зачем: четыре команды из CLAUDE.md - это четыре вызова, а каждый вызов
# обрывается примерно на 178-й секунде. Один полный pytest занимал 157 из них,
# поэтому запас был нулевой. Здесь тесты идут параллельно (xdist, ~117 с), и всё
# вместе укладывается в один вызов примерно за 135 с. Вывод сжат до строки на
# проверку; разворачивается только то, что упало.
#
# Использование:
#   scripts/check.sh                        линтеры + все тесты
#   scripts/check.sh tests/unit/test_x.py   линтеры + только эти тесты (~25 с)
#   scripts/check.sh -k "wake word"         любые аргументы уходят в pytest
#   scripts/check.sh --lint                 только ruff/black/mypy (~16 с)
#   scripts/check.sh --tests                только pytest
#   scripts/check.sh --fmt                  переформатировать black-ом на месте
#   scripts/check.sh --seq                  тесты последовательно, как в CI
#
# Параллель включается только на полном прогоне: восемь процессов стартуют
# около 25 с, и на одном тест-файле это дороже самих тестов.
#
# Переменные: AYRIS_CHECK_JOBS (число процессов pytest, по умолчанию 8),
# AYRIS_CHECK_TAIL (сколько строк показывать от упавшей проверки, 120).
#
# Живых облачных запросов здесь нет и быть не должно: тесты облачных STT/TTS
# работают на httpx.MockTransport, ключи им не нужны. Настоящие ключи проверяет
# scripts/check-keys.ps1 на машине пользователя, и в этот скрипт он не входит.

set -uo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT" || exit 1

PY="$ROOT/_tools/python/bin/python3"
RUFF="$ROOT/_tools/python/bin/ruff"
RNNOISE="$ROOT/_tools/rnnoise/librnnoise.so"

if [[ ! -x $PY ]]; then
    echo "нет портативного питона: $PY" >&2
    echo "тулчейн живёт в папке проекта, системный python не подходит" >&2
    exit 2
fi

JOBS=${AYRIS_CHECK_JOBS:-8}
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
            sed -n '9,21p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

# На частичном прогоне старт восьми процессов дороже самих тестов.
((${#PYTEST_ARGS[@]})) && JOBS=1

FAILED=()
STARTED=$(date +%s)

# Прогнать проверку, показать одну строку при успехе и хвост лога при падении.
# Последняя строка вывода у всех четырёх инструментов - это как раз итог
# ("All checks passed!", "Success: no issues...", "1628 passed, 5 skipped").
stage() {
    local name=$1
    shift
    local log t0 spent
    log=$(mktemp "/tmp/ayris-check-$name-XXXXXX.log")
    t0=$(date +%s)
    if "$@" >"$log" 2>&1; then
        spent=$(($(date +%s) - t0))
        printf '%-7s ok      %4ds  %s\n' "$name" "$spent" "$(tail -n 1 "$log")"
    else
        spent=$(($(date +%s) - t0))
        printf '%-7s ПАДАЕТ  %4ds\n' "$name" "$spent"
        tail -n "$TAIL" "$log" | sed 's/^/  | /'
        FAILED+=("$name")
    fi
    rm -f "$log"
}

# Тесты - самая долгая проверка, и они ждут не процессор, а порождённые процессы.
# Поэтому pytest уходит в фон, линтеры идут поверх него, и полный прогон
# укладывается в ~120 с вместо ~155: без этого холодный mypy (31 с на пустом
# /tmp) съедал весь запас до обрыва вызова на 178-й секунде.
PYTEST_LOG=""
PYTEST_PID=""
pytest_start() {
    PT=(-q --tb=short "--basetemp=/tmp/pt_${RANDOM}${RANDOM}_$$" -p no:cacheprovider)
    if ((JOBS > 1)) && "$PY" -c 'import xdist' 2>/dev/null; then
        PT+=(-n "$JOBS" --dist=load)
    fi
    # Без этой переменной rnnoise_available() отдаёт False, весь путь RNNoise
    # молча уходит в noise gate, а TestRnnoise пропускается.
    [[ -f $RNNOISE ]] && export AYRIS_RNNOISE_LIB="$RNNOISE"
    PYTEST_LOG=$(mktemp /tmp/ayris-check-pytest-XXXXXX.log)
    PYTEST_START=$(date +%s)
    "$PY" -m pytest "${PT[@]}" "${PYTEST_ARGS[@]}" >"$PYTEST_LOG" 2>&1 &
    PYTEST_PID=$!
}

pytest_wait() {
    local spent
    wait "$PYTEST_PID"
    local code=$?
    spent=$(($(date +%s) - PYTEST_START))
    if ((code == 0)); then
        printf '%-7s ok      %4ds  %s\n' pytest "$spent" "$(tail -n 1 "$PYTEST_LOG")"
    else
        printf '%-7s ПАДАЕТ  %4ds\n' pytest "$spent"
        tail -n "$TAIL" "$PYTEST_LOG" | sed 's/^/  | /'
        FAILED+=(pytest)
    fi
    rm -f "$PYTEST_LOG"
}

((RUN_TESTS)) && pytest_start

if ((RUN_LINT)); then
    # Ровно те же цели, что в CI: src и tests. Расширять список нельзя, иначе
    # песочница краснеет там, где CI зелёный.
    stage ruff "$RUFF" check src tests
    if ((FMT)); then
        stage black "$PY" -m black src tests
    else
        stage black "$PY" -m black --check src tests
    fi
    # Без --cache-dir вне проекта mypy падает на PermissionError.
    stage mypy "$PY" -m mypy --cache-dir=/tmp/mypycache src
fi

((RUN_TESTS)) && pytest_wait

# Куски склеены, чтобы git grep не нашёл сам этот файл.
SECRET_PAT='githu''b_pat_|gh''p_|sk-[A-Za-z0-9]{20}|BEGIN (RSA|OPENSSH|PRIVATE)'
FOUND=$(git grep -nIE "$SECRET_PAT" 2>/dev/null)
if [[ -n $FOUND ]]; then
    printf '%-7s ПАДАЕТ        похоже на секрет в коде\n' секреты
    echo "$FOUND" | sed 's/^/  | /'
    FAILED+=(секреты)
else
    printf '%-7s ok         0s  секретов в трекаемых файлах нет\n' секреты
fi

# Оставленный lock-файл не даст пользователю сделать коммит.
mapfile -t LOCKS < <(find .git -name '*.lock' 2>/dev/null)
if ((${#LOCKS[@]})); then
    rm -f "${LOCKS[@]}"
    echo "убрано lock-файлов git: ${#LOCKS[@]}"
fi

TOTAL=$(($(date +%s) - STARTED))
if ((${#FAILED[@]})); then
    echo "красное: ${FAILED[*]} (всего ${TOTAL}s)"
    [[ " ${FAILED[*]} " == *" black "* ]] && echo "форматирование чинится само: scripts/check.sh --fmt"
    exit 1
fi
echo "всё зелёное за ${TOTAL}s"
