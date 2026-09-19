#!/bin/sh
# Пересборка графа кода graphify перед началом сессии.
#
# Вызывается хуком SessionStart из .claude/settings.local.json. Ничего не качает
# и не тратит токенов: tree-sitter разбирает исходники локально. Если graphify не
# установлен (CI, чистый клон, другая машина) — молча выходит с нулём, чтобы хук
# ничего не ломал.
#
# Почему полный extract, а не `graphify update`: update ориентируется на свой
# манифест и в одном из прогонов записал пустой graph.json (0 узлов), после чего
# на повторный вызов отвечал «No code-graph changes detected» и не восстанавливал
# граф. Полная сборка детерминирована: 42 с, 14.4k узлов каждый раз.
#
# Собирается в _tools/graph.new и подменяется одним mv, чтобы запрос к графу в
# этот момент не поймал недописанный файл. Свежесть — в _tools/graph/BUILT.txt.

root="${CLAUDE_PROJECT_DIR:-.}"
# Claude Code передаёт путь в windows-виде (E:\...), для sh его надо перевести.
command -v cygpath >/dev/null 2>&1 && root=$(cygpath "$root")
cd "$root" || exit 0

GRAPHIFY=_tools/venv/Scripts/graphify.exe
STAMP=_tools/graph/BUILT.txt
GRAPH=_tools/graph/graphify-out/graph.json
HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?')

if [ "${1:-}" = "--check" ]; then
  [ -x "$GRAPHIFY" ] || { echo "graphify не установлен" >&2; exit 1; }
  [ -f "$STAMP" ] || { echo "нет $STAMP" >&2; exit 1; }
  [ -f "$GRAPH" ] || { echo "нет $GRAPH" >&2; exit 1; }
  size=$(wc -c <"$GRAPH")
  [ "$size" -gt 1000000 ] || { echo "graph.json подозрительно мал: $size байт" >&2; exit 1; }
  grep -qx "commit $HEAD" "$STAMP" || { echo "граф собран не для HEAD $HEAD" >&2; exit 1; }
  nodes=$(sed -n 's/^nodes //p' "$STAMP" | head -1)
  case "$nodes" in
    ''|*[!0-9]*) echo "в штампе нет корректного числа узлов" >&2; exit 1 ;;
  esac
  [ "$nodes" -gt 1000 ] || { echo "в графе подозрительно мало узлов: $nodes" >&2; exit 1; }
  newer=$(find src tests scripts -name '*.py' -newer "$STAMP" -print -quit 2>/dev/null)
  [ -z "$newer" ] || { echo "граф старше $newer" >&2; exit 1; }
  echo "graphify: актуален для $HEAD, $nodes узлов"
  exit 0
fi

[ -x "$GRAPHIFY" ] || exit 0

# Пересобирать только если что-то изменилось: сменился коммит, пропал штамп или
# какой-нибудь .py новее последней сборки. Иначе 42 с процессорного времени уходят
# впустую на каждый новый чат.
if [ -f "$STAMP" ] && grep -qx "commit $HEAD" "$STAMP"; then
  newer=$(find src tests scripts -name '*.py' -newer "$STAMP" -print -quit 2>/dev/null)
  [ -n "$newer" ] || exit 0
fi

rm -rf _tools/graph.new
out=$("$GRAPHIFY" extract . --code-only --out _tools/graph.new 2>&1) || exit 0

NEW=_tools/graph.new/graphify-out/graph.json
[ -f "$NEW" ] || exit 0
# Пустой граф — 125 байт; настоящий — 18 МБ. Ниже 1 МБ подменять нечем.
size=$(wc -c <"$NEW")
[ "$size" -gt 1000000 ] || exit 0

nodes=$(printf '%s\n' "$out" | sed -n 's/.*graph\.json: \([0-9]*\) nodes.*/\1/p' | head -1)
printf 'commit %s\nnodes %s\nbuilt %s\n' "$HEAD" "${nodes:-?}" "$(date '+%Y-%m-%d %H:%M')" \
  >_tools/graph.new/BUILT.txt

rm -rf _tools/graph.old
[ -d _tools/graph ] && mv _tools/graph _tools/graph.old
mv _tools/graph.new _tools/graph && rm -rf _tools/graph.old
