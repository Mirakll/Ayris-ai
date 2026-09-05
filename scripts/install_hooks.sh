#!/usr/bin/env bash
# Ayris — поставить хуки git.
#
# Хуков два, и оба зовут один и тот же scripts/check.sh:
#
#   pre-commit  →  scripts/check.sh --lint   (~7 с)
#   pre-push    →  scripts/check.sh          (~65 с, с тестами, slow и hardware)
#
# Почему не `pre-commit install`, хотя .pre-commit-config.yaml в проекте есть:
# генератор из пакета pre-commit вписывает в хук абсолютный путь к питону, а путь
# этого проекта — кириллица, и записывается он в кодировке консоли (cp1251), то
# есть в файл попадает «E:\??????? ??? ??? ???\_tools\venv\Scripts\python.exe».
# Такой путь не существует, `[ -x ]` даёт ложь, а самого `pre-commit` в PATH нет
# — хук отвечает «`pre-commit` not found» и не пускает ни один коммит. Проверено
# 05.09.2026 на этой машине. Здешние хуки абсолютных путей не содержат вовсе:
# git зовёт их из корня рабочей копии, поэтому хватает относительного пути.
#
# Конфиг pre-commit остаётся для `pre-commit run --all-files` и для машин с
# латинским путём. Набор проверок и там, и здесь один — сам скрипт check.sh.
#
# Запуск (идемпотентно, можно сколько угодно раз):
#   bash scripts/install_hooks.sh
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
HOOKS=$(git rev-parse --git-path hooks) || exit 1
mkdir -p "$HOOKS" || exit 1

write_hook() {
    local name=$1 flag=$2 what=$3
    cat >"$HOOKS/$name" <<HOOK
#!/bin/sh
# Поставлен scripts/install_hooks.sh — не редактировать руками.
# $what
exec bash scripts/check.sh $flag
HOOK
    chmod +x "$HOOKS/$name"
    echo "  $name → scripts/check.sh ${flag:-(весь набор)}"
}

echo "хуки git:"
write_hook pre-commit --lint "ruff, black, mypy, сверка пинов, запуск — ~7 с."
write_hook pre-push "" "Весь набор: тесты, slow, hardware, секреты — ~65 с."
echo "готово. Обойти в исключительном случае: git commit --no-verify"
