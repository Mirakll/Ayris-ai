# Статик-сервер БЕЗ кэша для превью макетов.
# python.exe _serve_nocache.py <port> <directory>
#
# Штатный `python -m http.server` не шлёт ETag/Cache-Control, поэтому браузер
# по эвристике держит _sphere.html и страницы в кэше и не видит правок
# (симптом: «поменял, а всё как было»). Здесь каждый ответ помечен no-store —
# каждый reload тянет свежие файлы.
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8778
    directory = sys.argv[2] if len(sys.argv) > 2 else "."
    handler = partial(NoCacheHandler, directory=directory)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"serving {directory} on http://127.0.0.1:{port} (no-store)")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
