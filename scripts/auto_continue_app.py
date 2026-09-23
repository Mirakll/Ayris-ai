#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Мини-приложение: следит за окном(ами) Claude и при ошибке шлёт «продолжай».

Окно с кнопками Старт/Стоп, галочками режима и живым логом. Раз в N секунд читает
текст окна Claude; если видит фразу-ошибку в хвосте чата — отправляет сообщение.
Пока ошибка не пропала, повторяет каждые N секунд (с предохранителем от зацикливания).

Два ключевых режима:
  * «Фоновый режим» — шлёт ввод через PostMessage прямо в окно, НЕ воруя фокус и НЕ
    трогая буфер обмена. Можно спокойно печатать в других приложениях. Best-effort:
    работает, если поле ввода в том окне было в фокусе последним и окно НЕ свёрнуто.
    Если снять галку — старое поведение: активирует окно, вставляет через буфер, Enter
    (надёжнее попадает, но перехватывает фокус).
  * «Следить за всеми окнами» — обрабатывает сразу все окна Claude (каждый чат должен
    быть ОТДЕЛЬНЫМ окном ОС, не вкладкой: у неактивной вкладки контент не читается).

Запуск:
    pip install uiautomation
    python scripts/auto_continue_app.py

Только Windows. Запускать в ОТДЕЛЬНОМ процессе (не внутри Claude Code).
Свёрнутые окна не отслеживаются — держи их видимыми (можно сбоку/на другом мониторе).
Сначала погоняй с галкой «Тест», чтобы увидеть, что ловится, и подправь список фраз.
"""

from __future__ import annotations

import ctypes
import os
import queue
import threading
import time
from ctypes import wintypes

import tkinter as tk
from tkinter import ttk

try:
    import uiautomation as auto
except ImportError:
    import tkinter.messagebox as mb

    r = tk.Tk()
    r.withdraw()
    mb.showerror("Нет зависимости",
                 "Не установлен модуль uiautomation.\n\nОткрой терминал и выполни:\n"
                 "pip install uiautomation")
    raise SystemExit(1)


# --- дефолты ---------------------------------------------------------------
DEFAULT_PHRASES = [
    "server error", "connection error", "api error", "internal server error",
    "request timed out", "fetch failed", "network error", "overloaded",
    "something went wrong", "please try again",
    "ошибка соединения", "ошибка сервера", "произошла ошибка",
    "попробуйте ещё раз", "попробуйте еще раз",
]
DEFAULT_MESSAGE = "продолжай"
DEFAULT_INTERVAL = 30
DEFAULT_PROCESS = "Claude.exe"
TAIL_CHARS = 1800
MAX_NODES = 6000
MAX_SENDS = 40


# --- winapi ----------------------------------------------------------------
_user32 = ctypes.windll.user32
_k32 = ctypes.windll.kernel32
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_CHAR = 0x0102
VK_RETURN = 0x0D
MAPVK_VK_TO_VSC = 0


def _pid_to_name(pid: int) -> str:
    h = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if _k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return os.path.basename(buf.value)
        return ""
    finally:
        _k32.CloseHandle(h)


def _top_windows():
    return list(auto.GetRootControl().GetChildren())


def find_windows(process: str, title_sub: str, all_windows: bool):
    """Список окон-целей. Если all_windows=False — только одно (самое большое)."""
    found = []
    for w in _top_windows():
        try:
            pid, name, r = w.ProcessId, (w.Name or ""), w.BoundingRectangle
        except Exception:
            continue
        if title_sub:
            if title_sub.lower() not in name.lower():
                continue
        else:
            if _pid_to_name(pid).lower() != process.lower():
                continue
        try:
            area = max(0, r.right - r.left) * max(0, r.bottom - r.top)
        except Exception:
            area = 0
        if area <= 0:
            continue  # свёрнутое/невидимое пропускаем
        found.append((area, w))
    if not found:
        return []
    found.sort(key=lambda t: t[0], reverse=True)
    if all_windows:
        return [w for _, w in found]
    return [found[0][1]]


def collect_text(control) -> str:
    parts, stack, count = [], [control], 0
    while stack and count < MAX_NODES:
        c = stack.pop()
        count += 1
        try:
            if c.Name:
                parts.append(c.Name)
        except Exception:
            pass
        try:
            stack.extend(reversed(c.GetChildren()))
        except Exception:
            pass
    return "\n".join(parts)


def matched_phrase(text: str, phrases) -> str | None:
    low = text.lower()
    for p in phrases:
        if p and p in low:
            return p
    return None


def find_input(win):
    candidates, stack, count = [], [win], 0
    while stack and count < MAX_NODES:
        c = stack.pop()
        count += 1
        try:
            if c.ControlTypeName in ("EditControl", "DocumentControl"):
                r = c.BoundingRectangle
                if r and (r.right - r.left) > 40 and (r.bottom - r.top) > 8:
                    candidates.append((r.top, r))
        except Exception:
            pass
        try:
            stack.extend(c.GetChildren())
        except Exception:
            pass
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0])
    return candidates[-1][1]


# --- отправка: фон (без фокуса) через PostMessage --------------------------
def _hwnd_of(win) -> int:
    try:
        return int(win.NativeWindowHandle)
    except Exception:
        return 0


def _find_render_widget(hwnd: int) -> int:
    """Ищет дочернее окно рендерера Chromium — туда шлём ввод в фоне."""
    target = {"h": 0}

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(child, _lparam):
        buf = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(child, buf, 256)
        cls = buf.value or ""
        if "Chrome_RenderWidgetHostHWND" in cls:
            target["h"] = child
            return False
        if "Intermediate" in cls or "Chrome_WidgetWin" in cls:
            # спускаемся глубже
            _user32.EnumChildWindows(child, cb, 0)
            if target["h"]:
                return False
        return True

    _user32.EnumChildWindows(hwnd, cb, 0)
    return target["h"]


def _post_char(hwnd: int, ch: str) -> None:
    _user32.PostMessageW(hwnd, WM_CHAR, ord(ch), 0)


def _post_enter(hwnd: int) -> None:
    scan = _user32.MapVirtualKeyW(VK_RETURN, MAPVK_VK_TO_VSC)
    down = 0x00000001 | (scan << 16)
    up = 0x00000001 | (scan << 16) | 0xC0000000
    _user32.PostMessageW(hwnd, WM_KEYDOWN, VK_RETURN, down)
    _user32.PostMessageW(hwnd, WM_CHAR, VK_RETURN, down)
    _user32.PostMessageW(hwnd, WM_KEYUP, VK_RETURN, up)


def send_background(win, text: str) -> bool:
    """Шлёт текст+Enter в окно, не воруя фокус. True, если нашли куда слать."""
    hwnd = _hwnd_of(win)
    if not hwnd:
        return False
    target = _find_render_widget(hwnd) or hwnd
    for ch in text:
        _post_char(target, ch)
        time.sleep(0.005)
    time.sleep(0.05)
    _post_enter(target)
    return True


# --- отправка: передний план (с фокусом) через буфер -----------------------
def send_foreground(win, text: str) -> None:
    win.SetActive()
    time.sleep(0.25)
    r = find_input(win)
    if r is not None:
        x, y = (r.left + r.right) // 2, r.bottom - 10
    else:
        b = win.BoundingRectangle
        x, y = (b.left + b.right) // 2, b.bottom - 60
    auto.Click(x, y)
    time.sleep(0.15)
    auto.SetClipboardText(text)
    time.sleep(0.05)
    auto.SendKeys("{Ctrl}v", waitTime=0.05)
    time.sleep(0.15)
    auto.SendKeys("{Enter}", waitTime=0.05)


# --- рабочий поток ---------------------------------------------------------
class Watcher(threading.Thread):
    def __init__(self, cfg: dict, log_q: queue.Queue):
        super().__init__(daemon=True)
        self.cfg = cfg
        self.log = log_q
        self._stop = threading.Event()
        self.sends: dict[int, int] = {}  # hwnd -> счётчик отправок

    def stop(self):
        self._stop.set()

    def emit(self, msg: str):
        self.log.put(time.strftime("[%H:%M:%S] ") + msg)

    def _handle_window(self, win, cfg) -> None:
        hwnd = _hwnd_of(win)
        try:
            title = (win.Name or "")[:30]
        except Exception:
            title = ""
        tag = f"[{title}]" if cfg["all"] else ""
        text = collect_text(win)
        tail = text[-TAIL_CHARS:]
        hit = matched_phrase(tail, cfg["phrases"])
        if not hit:
            self.emit(f"{tag} ошибок не видно")
            return
        if cfg["dry"]:
            self.emit(f"{tag} ТЕСТ: словил '{hit}' — отправил бы «{cfg['message']}»")
            return
        cnt = self.sends.get(hwnd, 0)
        if cnt >= cfg["max_sends"]:
            self.emit(f"{tag} лимит отправок достигнут — не шлю (защита от зацикливания)")
            return
        ok = True
        if cfg["background"]:
            ok = send_background(win, cfg["message"])
            if not ok:
                self.emit(f"{tag} не нашёл куда слать в фоне — пропуск (окно свёрнуто?)")
        else:
            send_foreground(win, cfg["message"])
        if ok:
            self.sends[hwnd] = cnt + 1
            mode = "фон" if cfg["background"] else "фокус"
            self.emit(f"{tag} ошибка '{hit}' → отправлено «{cfg['message']}» "
                      f"({mode}, {self.sends[hwnd]}/{cfg['max_sends']})")

    def run(self):
        cfg = self.cfg
        self.emit(f"старт. тест={cfg['dry']} фон={cfg['background']} все_окна={cfg['all']} "
                  f"интервал={cfg['interval']}с "
                  f"цель={'title:'+cfg['title'] if cfg['title'] else cfg['process']}")
        # COM надо инициализировать в ЭТОМ потоке, иначе uiautomation падает
        # с OSError «Не был произведён вызов CoInitialize».
        auto.InitializeUIAutomationInCurrentThread()
        try:
            while not self._stop.is_set():
                try:
                    wins = find_windows(cfg["process"], cfg["title"], cfg["all"])
                    if not wins:
                        self.emit("окно не найдено (свёрнуто? проверь процесс/заголовок)")
                    else:
                        for w in wins:
                            if self._stop.is_set():
                                break
                            try:
                                self._handle_window(w, cfg)
                            except Exception as e:
                                self.emit(f"сбой по окну: {e!r}")
                except Exception as e:
                    self.emit(f"сбой цикла: {e!r}")
                for _ in range(int(cfg["interval"] * 10)):
                    if self._stop.is_set():
                        break
                    time.sleep(0.1)
        finally:
            auto.UninitializeUIAutomationInCurrentThread()
        self.emit("остановлено")
        self.log.put("__STOPPED__")


# --- GUI -------------------------------------------------------------------
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.log_q: queue.Queue = queue.Queue()
        self.watcher: Watcher | None = None

        root.title("Авто-продолжай для Claude")
        root.geometry("580x600")
        root.minsize(500, 500)

        frm = ttk.Frame(root, padding=10)
        frm.pack(fill="both", expand=True)

        row = 0
        ttk.Label(frm, text="Сообщение:").grid(row=row, column=0, sticky="w")
        self.msg = tk.StringVar(value=DEFAULT_MESSAGE)
        ttk.Entry(frm, textvariable=self.msg).grid(row=row, column=1, sticky="ew", pady=2)

        row += 1
        ttk.Label(frm, text="Интервал, сек:").grid(row=row, column=0, sticky="w")
        self.interval = tk.StringVar(value=str(DEFAULT_INTERVAL))
        ttk.Entry(frm, textvariable=self.interval, width=8).grid(row=row, column=1, sticky="w", pady=2)

        row += 1
        ttk.Label(frm, text="Процесс окна:").grid(row=row, column=0, sticky="w")
        self.process = tk.StringVar(value=DEFAULT_PROCESS)
        ttk.Entry(frm, textvariable=self.process).grid(row=row, column=1, sticky="ew", pady=2)

        row += 1
        ttk.Label(frm, text="…или заголовок:").grid(row=row, column=0, sticky="w")
        self.title = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=self.title).grid(row=row, column=1, sticky="ew", pady=2)

        row += 1
        self.max_sends = tk.StringVar(value=str(MAX_SENDS))
        ttk.Label(frm, text="Лимит отправок:").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.max_sends, width=8).grid(row=row, column=1, sticky="w", pady=2)

        row += 1
        self.background = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Фоновый режим (не воровать фокус — можно печатать в других окнах)",
                        variable=self.background).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

        row += 1
        self.all_windows = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Следить за всеми окнами Claude сразу",
                        variable=self.all_windows).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

        row += 1
        self.dry = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Тест (не отправлять, только показывать что ловит)",
                        variable=self.dry).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

        row += 1
        ttk.Label(frm, text="Фразы-ошибки (по одной в строке):").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        row += 1
        self.phrases = tk.Text(frm, height=6, wrap="none")
        self.phrases.insert("1.0", "\n".join(DEFAULT_PHRASES))
        self.phrases.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)

        row += 1
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        self.start_btn = ttk.Button(btns, text="Старт", command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(btns, text="Стоп", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=6)
        ttk.Button(btns, text="Окна…", command=self.list_windows).pack(side="left")
        self.status = ttk.Label(btns, text="остановлено")
        self.status.pack(side="right")

        row += 1
        ttk.Label(frm, text="Лог:").grid(row=row, column=0, sticky="w", pady=(6, 0))
        row += 1
        self.logbox = tk.Text(frm, height=10, state="disabled", wrap="word")
        self.logbox.grid(row=row, column=0, columnspan=2, sticky="nsew")

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(row, weight=1)

        self.root.after(150, self._drain_log)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _append(self, line: str):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", line + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _drain_log(self):
        try:
            while True:
                line = self.log_q.get_nowait()
                if line == "__STOPPED__":
                    self._set_running(False)
                    continue
                self._append(line)
        except queue.Empty:
            pass
        self.root.after(150, self._drain_log)

    def _set_running(self, running: bool):
        self.start_btn.configure(state="disabled" if running else "normal")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.status.configure(text="работает" if running else "остановлено")

    def start(self):
        if self.watcher and self.watcher.is_alive():
            return
        try:
            interval = max(3.0, float(self.interval.get().replace(",", ".")))
        except ValueError:
            interval = DEFAULT_INTERVAL
        try:
            max_sends = max(1, int(self.max_sends.get()))
        except ValueError:
            max_sends = MAX_SENDS
        phrases = [p.strip().lower() for p in self.phrases.get("1.0", "end").splitlines() if p.strip()]
        cfg = {
            "message": self.msg.get() or DEFAULT_MESSAGE,
            "interval": interval,
            "process": self.process.get().strip() or DEFAULT_PROCESS,
            "title": self.title.get().strip(),
            "max_sends": max_sends,
            "dry": self.dry.get(),
            "background": self.background.get(),
            "all": self.all_windows.get(),
            "phrases": phrases or DEFAULT_PHRASES,
        }
        self.watcher = Watcher(cfg, self.log_q)
        self.watcher.start()
        self._set_running(True)

    def stop(self):
        if self.watcher:
            self.watcher.stop()

    def list_windows(self):
        self._append("--- окна (заголовок | процесс) ---")
        for w in _top_windows():
            try:
                name = (w.Name or "").strip()
                proc = _pid_to_name(w.ProcessId)
            except Exception:
                continue
            if name or proc:
                self._append(f"  {name!r}  |  {proc}")
        self._append("-----------------------------------")

    def _on_close(self):
        if self.watcher:
            self.watcher.stop()
        self.root.after(200, self.root.destroy)


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
