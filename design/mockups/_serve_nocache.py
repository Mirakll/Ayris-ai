# Статик-сервер БЕЗ кэша для превью макетов.
# python.exe _serve_nocache.py [port] [directory]
# python.exe _serve_nocache.py --all
#
# Штатный `python -m http.server` не шлёт ETag/Cache-Control, поэтому браузер
# по эвристике держит _sphere.html и страницы в кэше и не видит правок
# (симптом: «поменял, а всё как было»). Здесь каждый ответ помечен no-store —
# каждый reload тянет свежие файлы.
# Поддерживает режим --all для одновременного запуска всех серверов из .claude/launch.json,
# fallback к корню проекта для бесшовной загрузки ресурсов и CORS-заголовки.
import os
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Серверы по умолчанию (соответствуют .claude/launch.json)
DEFAULT_SERVERS = [
    (8777, "design/mockups", "Каталог мокапов / Трей"),
    (8778, "design/mockups/home", "Домашний экран / Сфера"),
    (8779, "design/mockups", "Redesign 2 (Astra / Obsidian)"),
    (8780, "design/mockups/avatar", "3D VRM Аватар АйРИ"),
    (8790, "design/mockups", "Икосаэдр (Icosahedron Core)"),
    (8791, "src/ayris/gui/widgets/sphere/assets", "Sphere Assets"),
    (8812, "src/ayris/gui/widgets/sphere/assets", "Sphere Assets Verify"),
]


class NoCacheHandler(SimpleHTTPRequestHandler):
    extensions_map = SimpleHTTPRequestHandler.extensions_map.copy()
    extensions_map.update({
        ".vrm": "model/gltf-binary",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".wasm": "application/wasm",
        ".mjs": "application/javascript",
        ".js": "application/javascript",
        ".json": "application/json",
        ".css": "text/css",
    })

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def translate_path(self, path: str) -> str:
        normal = super().translate_path(path)
        if os.path.exists(normal):
            return normal
        # Fallback к корню репозитория, если ресурс запрашивается по полному пути (/design/...)
        clean = unquote(urlsplit(path).path).lstrip("/\\")
        cand = os.path.abspath(os.path.join(PROJECT_ROOT, clean))
        if os.path.exists(cand) and cand.startswith(PROJECT_ROOT):
            return cand
        return normal

    def log_message(self, format: str, *args) -> None:
        # Лаконичный лог без засорения консоли
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")


def run_single(port: int, directory: str, label: str = "") -> None:
    abs_dir = os.path.abspath(os.path.join(PROJECT_ROOT, directory)) if not os.path.isabs(directory) else directory
    handler = partial(NoCacheHandler, directory=abs_dir)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    except OSError as e:
        print(f"[!] Порт {port} ({label or directory}) уже занят или недоступен: {e}", file=sys.stderr)
        return
    lbl_str = f" [{label}]" if label else ""
    print(f"serving {abs_dir} on http://127.0.0.1:{port}{lbl_str} (no-store)")
    httpd.serve_forever()


def run_all() -> None:
    print(f"=== Запуск всех серверов мокапов Ayris (корень: {PROJECT_ROOT}) ===")
    threads = []
    for port, rel_dir, label in DEFAULT_SERVERS:
        t = threading.Thread(
            target=run_single,
            args=(port, rel_dir, label),
            daemon=True,
            name=f"MockupServer-{port}",
        )
        t.start()
        threads.append(t)
        time.sleep(0.05)

    print("\nВсе серверы запущены:")
    for port, rel_dir, label in DEFAULT_SERVERS:
        print(f"  * http://127.0.0.1:{port}/ -> {label} ({rel_dir})")
    print("\nНажмите Ctrl+C для завершения.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nОстановка серверов...")


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "--all":
        run_all()
    else:
        port = int(sys.argv[1])
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        run_single(port, directory)


if __name__ == "__main__":
    main()
