#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Авто-«продолжай»: следит за окном Claude и при появлении ошибки шлёт сообщение.

Идея: раз в N секунд читает текст окна Claude, и если в хвосте видит фразу-ошибку
(«Server error», «Connection error», «API Error», «Ошибка соединения» и т.п.),
активирует окно, вставляет «продолжай» в поле ввода и жмёт Enter.

ВАЖНО:
  * Запускать в ОТДЕЛЬНОМ терминале (не внутри Claude Code).
  * Скрипт при отправке ПЕРЕХВАТЫВАЕТ фокус на окно Claude — не работай в этот
    момент в других приложениях.
  * Сначала прогнать с --dry-run: он ничего не печатает в чат, только показывает,
    что бы он словил. По хвосту подстрой ERROR_PHRASES под реальный текст ошибки.
  * Требуется: pip install uiautomation

Только Windows.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from ctypes import wintypes

try:
    import uiautomation as auto
except ImportError:  # pragma: no cover - подсказка при отсутствии зависимости
    print("Нет модуля uiautomation. Установи:  pip install uiautomation", file=sys.stderr)
    sys.exit(1)


# --- фразы, по которым считаем, что в чате ошибка -------------------------
# Держи их достаточно конкретными, иначе поймаешь обычный текст беседы.
# Подстрой по выводу --dry-run под реальные баннеры.
ERROR_PHRASES = [
    "server error",
    "connection error",
    "api error",
    "internal server error",
    "request timed out",
    "fetch failed",
    "network error",
    "overloaded",
    "something went wrong",
    "please try again",
    "ошибка соединения",
    "ошибка сервера",
    "произошла ошибка",
    "попробуйте ещё раз",
    "попробуйте еще раз",
]

MESSAGE = "продолжай"
INTERVAL = 30          # секунд между проверками/отправками
TAIL_CHARS = 1800      # смотрим только «хвост» текста окна (низ = свежее)
MAX_SENDS = 40         # предохранитель от бесконечного зацикливания
PROCESS = "Claude.exe"  # имя процесса окна по умолчанию
MAX_NODES = 6000       # ограничение обхода дерева UI


# --- winapi: pid -> имя exe ------------------------------------------------
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _pid_to_name(pid: int) -> str:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        k32.CloseHandle(h)


# --- поиск окна ------------------------------------------------------------
def _top_windows():
    root = auto.GetRootControl()
    return list(root.GetChildren())


def list_windows() -> None:
    print("Верхнеуровневые окна (заголовок | процесс | pid):")
    for w in _top_windows():
        try:
            name = (w.Name or "").strip()
            pid = w.ProcessId
            proc = _pid_to_name(pid)
        except Exception:  # noqa: BLE001
            continue
        if not name and not proc:
            continue
        print(f"  {name!r:40}  {proc:20}  {pid}")


def find_window(process: str, title_sub: str | None):
    best = None
    best_area = -1
    for w in _top_windows():
        try:
            pid = w.ProcessId
            name = w.Name or ""
            r = w.BoundingRectangle
        except Exception:  # noqa: BLE001
            continue
        if title_sub:
            if title_sub.lower() not in name.lower():
                continue
        else:
            if _pid_to_name(pid).lower() != process.lower():
                continue
        try:
            area = max(0, (r.right - r.left)) * max(0, (r.bottom - r.top))
        except Exception:  # noqa: BLE001
            area = 0
        if area > best_area:
            best, best_area = w, area
    return best


# --- чтение текста окна ----------------------------------------------------
def collect_text(control) -> str:
    parts: list[str] = []
    stack = [control]
    count = 0
    while stack and count < MAX_NODES:
        c = stack.pop()
        count += 1
        try:
            name = c.Name
            if name:
                parts.append(name)
        except Exception:  # noqa: BLE001
            pass
        try:
            # в обратном порядке, чтобы DFS шёл сверху вниз по документу
            stack.extend(reversed(c.GetChildren()))
        except Exception:  # noqa: BLE001
            pass
    return "\n".join(parts)


def matched_phrase(text: str) -> str | None:
    low = text.lower()
    for p in ERROR_PHRASES:
        if p in low:
            return p
    return None


# --- ввод сообщения --------------------------------------------------------
def find_input(win):
    """Пытается найти самое нижнее поле ввода (Edit/Document)."""
    candidates = []
    stack = [win]
    count = 0
    while stack and count < MAX_NODES:
        c = stack.pop()
        count += 1
        try:
            if c.ControlTypeName in ("EditControl", "DocumentControl"):
                r = c.BoundingRectangle
                if r and (r.right - r.left) > 40 and (r.bottom - r.top) > 8:
                    candidates.append((r.top, c, r))
        except Exception:  # noqa: BLE001
            pass
        try:
            stack.extend(c.GetChildren())
        except Exception:  # noqa: BLE001
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])  # по верхней кромке
    return candidates[-1]  # самое нижнее — это композер


def send_message(win, text: str) -> None:
    win.SetActive()
    time.sleep(0.25)
    hit = find_input(win)
    if hit is not None:
        _, _edit, r = hit
        x = (r.left + r.right) // 2
        y = r.bottom - 10
    else:
        r = win.BoundingRectangle
        x = (r.left + r.right) // 2
        y = r.bottom - 60
    auto.Click(x, y)
    time.sleep(0.15)
    auto.SetClipboardText(text)
    time.sleep(0.05)
    auto.SendKeys("{Ctrl}v", waitTime=0.05)
    time.sleep(0.15)
    auto.SendKeys("{Enter}", waitTime=0.05)


# --- главный цикл ----------------------------------------------------------
def ts() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="Авто-«продолжай» при ошибке в окне Claude")
    ap.add_argument("--dry-run", action="store_true", help="ничего не печатать в чат, только показывать что словил")
    ap.add_argument("--interval", type=float, default=INTERVAL, help="секунд между проверками")
    ap.add_argument("--message", default=MESSAGE, help="что отправлять")
    ap.add_argument("--process", default=PROCESS, help="имя процесса окна (по умолчанию Claude.exe)")
    ap.add_argument("--window-title", default=None, help="искать окно по подстроке заголовка вместо процесса")
    ap.add_argument("--max-sends", type=int, default=MAX_SENDS, help="предохранитель: макс. отправок подряд")
    ap.add_argument("--tail", type=int, default=TAIL_CHARS, help="сколько символов хвоста анализировать")
    ap.add_argument("--list", action="store_true", help="показать окна и выйти")
    args = ap.parse_args()

    if args.list:
        list_windows()
        return 0

    print(f"[{ts()}] старт. dry_run={args.dry_run} interval={args.interval}s "
          f"target={'title:'+args.window_title if args.window_title else args.process} "
          f"max_sends={args.max_sends}")
    print("Ctrl+C — выход.\n")

    sends = 0
    try:
        while True:
            win = find_window(args.process, args.window_title)
            if win is None:
                print(f"[{ts()}] окно не найдено (проверь --process/--window-title, см. --list)")
                time.sleep(args.interval)
                continue

            text = collect_text(win)
            tail = text[-args.tail:]
            hit = matched_phrase(tail)

            if hit:
                if args.dry_run:
                    print(f"[{ts()}] DRY-RUN: словил '{hit}' — отправил бы «{args.message}»")
                    print("  --- хвост окна ---")
                    print("  " + tail.replace("\n", "\n  ")[-800:])
                    print("  ------------------")
                else:
                    send_message(win, args.message)
                    sends += 1
                    print(f"[{ts()}] ошибка ('{hit}') → отправлено «{args.message}» "
                          f"({sends}/{args.max_sends})")
                    if sends >= args.max_sends:
                        print(f"[{ts()}] достигнут лимит отправок — стоп (защита от зацикливания)")
                        return 0
            else:
                print(f"[{ts()}] ошибок не видно")

            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n[{ts()}] остановлено пользователем")
        return 0


if __name__ == "__main__":
    sys.exit(main())
