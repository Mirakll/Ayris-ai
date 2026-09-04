"""Ожидание прогона CI для коммита — одной командой, без гадания.

Почему не `gh run watch`: `gh` на этой машине не установлен, а ставить его ради
одной команды — лишняя зависимость. Всё нужное REST API отдаёт и анонимно, кроме
логов упавшего джоба: за ними нужен токен, и он берётся из `.git/.credentials`,
куда его положил credential helper. Значение токена никуда не печатается.

Что скрипт делает такого, чего не делает `curl` руками:

* ждёт появления прогона (после `git push` он создаётся не мгновенно) и отличает
  «прогона ещё нет» от «на этой ветке workflow не запускается»;
* печатает строку на джоб и перепечатывает только при изменении: полный json
  прогона — это несколько сотен строк, из которых нужны четыре;
* по завершении сам находит упавший шаг и показывает хвост его лога, вместо того
  чтобы предлагать человеку сходить за `job_id` и сделать второй запрос. Хвост
  режется по последнему `##[error]`, а не по концу файла: после упавшего шага в
  логе идёт выгрузка артефактов (`if: always()`), и последние строки — про неё;
* укладывается в бюджет времени: вызов шелла обрывается примерно на 178-й
  секунде, поэтому по умолчанию скрипт выходит через 150 с и говорит, что зайти
  надо ещё раз. Прогон целиком — около шести минут, то есть заходов будет три.

    python scripts/ci_wait.py                 # прогон для HEAD, до 150 с
    python scripts/ci_wait.py --budget 20     # один взгляд и выход
    python scripts/ci_wait.py --sha <40 hex>  # прогон для другого коммита
    python scripts/ci_wait.py --no-logs       # не тянуть логи упавших джобов

Коды возврата: 0 — всё зелёное, 1 — есть упавшие джобы, 2 — прогон ещё идёт
(бюджет вышел), 3 — прогона для этого коммита нет, 4 — API недоступен.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Final

_ROOT: Final = Path(__file__).resolve().parents[1]
_API: Final = "https://api.github.com"
_CREDENTIALS: Final = _ROOT / ".git" / ".credentials"

#: `https://<логин>:<токен>@host` — формат, в котором пишет credential helper.
_CRED: Final = re.compile(r"^https://[^:/@]+:(?P<token>[^@]+)@")

#: Метка времени в начале каждой строки лога Actions.
_STAMP: Final = re.compile(r"^\S+Z\s")

#: Заключение джоба человеческими словами. Ширина колонки — как в check.sh.
_DONE: Final[dict[str, str]] = {
    "success": "ok",
    "failure": "ПАДАЕТ",
    "cancelled": "отменён",
    "skipped": "пропущен",
    "timed_out": "таймаут",
    "action_required": "ждёт кнопки",
    "neutral": "нейтрально",
}


class _DropAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Редирект без токена в заголовке.

    За логами джоба API отвечает не логом, а 302 на подписанный адрес в
    хранилище: подпись лежит в самом адресе. `urllib` при переходе тащит
    `Authorization` за собой, а хранилище на запрос с чужим заголовком отвечает
    400 — то есть без этого обработчика единственная команда, которой нужен
    токен, как раз и не работает. Заодно токен не уезжает на сторонний хост.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        following = super().redirect_request(req, fp, code, msg, headers, newurl)
        if following is None:
            return None
        if urllib.parse.urlsplit(newurl).netloc != urllib.parse.urlsplit(req.full_url).netloc:
            for name in ("Authorization", "authorization"):
                following.headers.pop(name, None)
                following.unredirected_hdrs.pop(name, None)
        return following


#: Свой opener вместо `urlopen`: тому обработчик редиректов не подменить.
_OPENER: Final = urllib.request.build_opener(_DropAuthOnRedirect)


def _git(*args: str) -> str:
    """Вывод git-команды. Ошибка git — это конец работы: без sha и адреса
    origin делать здесь нечего."""
    done = subprocess.run(["git", *args], cwd=_ROOT, capture_output=True, check=False, timeout=30)
    if done.returncode != 0:
        raise SystemExit(f"git {' '.join(args)}: {done.stderr.decode('utf-8', 'replace').strip()}")
    return done.stdout.decode("utf-8", "replace").strip()


def repo_slug() -> str:
    """`owner/repo` из адреса origin, с отброшенными логином и токеном в адресе."""
    url = re.sub(r"^https://[^@/]+@", "https://", _git("remote", "get-url", "origin"))
    match = re.search(r"github\.com[:/](?P<slug>[^/]+/[^/]+?)(?:\.git)?/?$", url)
    if match is None:
        raise SystemExit(f"не понял, какой это репозиторий: {url}")
    return match["slug"]


def read_token() -> str | None:
    """Токен из окружения или из `.git/.credentials`. Только для логов: всё
    остальное API отдаёт анонимно, а анонимный лимит — 60 запросов в час."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value.strip()
    if not _CREDENTIALS.is_file():
        return None
    for line in _CREDENTIALS.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CRED.match(line.strip())
        if match is not None:
            return match["token"]
    return None


def api(path: str, token: str | None, *, text: bool = False) -> Any:
    """Запрос к API. Возвращает разобранный json или текст; None — если API
    ответил ошибкой (её текст печатается)."""
    request = urllib.request.Request(f"{_API}{path}")
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("User-Agent", "ayris-ci-wait")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with _OPENER.open(request, timeout=30) as answer:
            raw = answer.read()
    except urllib.error.HTTPError as exc:
        print(f"API ответил {exc.code} на {path}")
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"API недоступен: {exc}")
        return None
    return raw.decode("utf-8", "replace") if text else json.loads(raw)


def human(seconds: float) -> str:
    """`48s`, `2m10s` — ширина предсказуемая, в отличие от «2 минуты 10 секунд»."""
    whole = int(seconds)
    return f"{whole}s" if whole < 60 else f"{whole // 60}m{whole % 60:02d}s"


def _moment(value: str | None) -> datetime | None:
    """Время из json Actions. Формат — ISO с `Z`, его разбирает сам datetime."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def spent(job: dict[str, Any]) -> str:
    """Сколько джоб идёт (или шёл). Пусто, если он ещё не начинался."""
    start = _moment(job.get("started_at"))
    if start is None:
        return ""
    end = _moment(job.get("completed_at")) or datetime.now(tz=start.tzinfo)
    return human((end - start).total_seconds())


def job_line(job: dict[str, Any]) -> str:
    """Строка на джоб: имя, состояние, время. Русские имена джобов выравниваются
    по символам — `str.ljust`, в отличие от `printf` в shell, считает не байты."""
    if job.get("status") != "completed":
        mark = "идёт" if job.get("status") == "in_progress" else "в очереди"
    else:
        done = job.get("conclusion") or "?"
        mark = _DONE.get(done, done)
    return f"  {job.get('name', '?'):<30.30} {mark:<11} {spent(job):>6}"


def failure_tail(slug: str, job: dict[str, Any], token: str | None, tail: int) -> None:
    """Упавший шаг и хвост его лога.

    Хвост отсчитывается от последнего `##[error]`, а не от конца файла: в
    workflow есть шаг выгрузки артефактов с `if: always()`, и последние строки
    лога — про него, а не про падение.
    """
    steps = [step for step in job.get("steps") or [] if step.get("conclusion") == "failure"]
    where = ", ".join(f"«{step.get('name')}»" for step in steps) or "шаг не назван"
    print(f"  {job.get('name')}: упал шаг {where}")
    print(f"  {job.get('html_url')}")
    if tail <= 0:
        return
    raw = api(f"/repos/{slug}/actions/jobs/{job['id']}/logs", token, text=True)
    if raw is None:
        print("  лога нет: анонимно за ним не пускают, нужен токен в .git/.credentials")
        return
    lines = [_STAMP.sub("", line).rstrip() for line in raw.splitlines()]
    marks = [number for number, line in enumerate(lines) if "##[error]" in line]
    stop = marks[-1] + 1 if marks else len(lines)
    for line in lines[max(0, stop - tail) : stop]:
        print(f"  | {line}")


def find_run(slug: str, sha: str, token: str | None, deadline: float) -> dict[str, Any] | int:
    """Прогон для коммита или код возврата: 3 — прогона нет, 4 — API молчит.

    Ждём появления: после `git push` GitHub создаёт прогон не мгновенно, и
    первый запрос честно отвечает `total_count: 0`. Но ждать этого долго нельзя —
    ровно так же отвечает ветка, на которой workflow не запускается вообще.
    """
    while True:
        answer = api(f"/repos/{slug}/actions/runs?head_sha={sha}&per_page=1", token)
        if answer is None:
            return 4
        runs = answer.get("workflow_runs") or []
        if runs:
            first: dict[str, Any] = runs[0]
            return first
        if time.monotonic() > deadline:
            print(f"прогона для {sha[:12]} нет: коммит не запушен или workflow тут не идёт")
            return 3
        time.sleep(5)


def wall_clock(jobs: list[dict[str, Any]]) -> str:
    """Время прогона по джобам: они идут параллельно, поэтому от самого раннего
    старта до самого позднего конца."""
    starts = [moment for job in jobs if (moment := _moment(job.get("started_at")))]
    ends = [moment for job in jobs if (moment := _moment(job.get("completed_at")))]
    if not starts or not ends:
        return "?"
    return human((max(ends) - min(starts)).total_seconds())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ждать прогон CI для коммита.")
    parser.add_argument("--sha", help="коммит; по умолчанию HEAD")
    parser.add_argument("--repo", help="owner/repo; по умолчанию из адреса origin")
    parser.add_argument("--budget", type=int, default=150, help="секунд на этот заход (150)")
    parser.add_argument("--interval", type=int, default=15, help="секунд между опросами (15)")
    parser.add_argument("--tail", type=int, default=40, help="строк лога от упавшего шага (40)")
    parser.add_argument("--no-logs", action="store_true", help="не тянуть логи упавших джобов")
    args = parser.parse_args(argv)

    slug = args.repo or repo_slug()
    # Полные 40 символов обязательны: на сокращённом sha API молча отдаёт пустой
    # список, и это не отличить от «прогона ещё нет».
    sha = _git("rev-parse", args.sha or "HEAD")
    token = read_token()
    interval = args.interval if token else max(args.interval, 30)
    if token is None:
        print("токена нет: опрос реже (анонимно 60 запросов в час), логи недоступны")

    started = time.monotonic()
    run = find_run(slug, sha, token, started + min(args.budget, 45))
    if isinstance(run, int):
        return run
    print(f"{run.get('head_branch')} {sha[:12]} — {run.get('html_url')}")

    seen = ""
    jobs: list[dict[str, Any]] = []
    while True:
        answer = api(f"/repos/{slug}/actions/runs/{run['id']}/jobs?per_page=50", token)
        if answer is None:
            return 4
        jobs = answer.get("jobs") or []
        # Перепечатываем только изменения: иначе каждые пятнадцать секунд в
        # контекст уезжает одна и та же таблица из пяти строк.
        snapshot = "|".join(f"{job.get('status')}{job.get('conclusion')}" for job in jobs)
        if snapshot != seen:
            seen = snapshot
            print(f"{human(time.monotonic() - started)} ожидания:")
            for job in jobs:
                print(job_line(job))
        if jobs and all(job.get("status") == "completed" for job in jobs):
            break
        left = args.budget - (time.monotonic() - started)
        if left <= 0:
            print("прогон ещё идёт — позови ещё раз: python scripts/ci_wait.py")
            return 2
        time.sleep(min(interval, left))

    failed = [job for job in jobs if job.get("conclusion") == "failure"]
    for job in failed:
        failure_tail(slug, job, token, 0 if args.no_logs else args.tail)
    strange = [
        job for job in jobs if job.get("conclusion") not in {"success", "skipped", "failure"}
    ]
    if failed or strange:
        names = [str(job.get("name")) for job in failed + strange]
        print(f"красное: {', '.join(names)} (прогон {wall_clock(jobs)})")
        return 1
    print(f"всё зелёное: {len(jobs)} джобов за {wall_clock(jobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
