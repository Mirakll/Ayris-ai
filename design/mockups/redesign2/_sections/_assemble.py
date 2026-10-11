"""Собрать партиалы разделов настроек в 2b-obsidian-violet.html.
Заменяет каждый маркер <!--SP:key--> содержимым _sections/key.html.
Запуск: python _sections/_assemble.py
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "2b-obsidian-violet.html"
KEYS = ["voice", "commands", "ai", "hotkeys", "overlay",
        "profiles", "updates", "devtools", "privacy"]


def main() -> None:
    html = MAIN.read_text(encoding="utf-8")
    missing, injected = [], []
    for key in KEYS:
        part = ROOT / "_sections" / f"{key}.html"
        marker = f"<!--SP:{key}-->"
        if not part.exists():
            missing.append(key)
            continue
        if marker not in html:
            print(f"SKIP {key}: marker not present (already filled?)")
            continue
        frag = part.read_text(encoding="utf-8").strip()
        html = html.replace(marker, "\n" + frag + "\n")
        injected.append(f"{key}:{len(frag)}B")
    MAIN.write_text(html, encoding="utf-8")
    print("INJECTED:", ", ".join(injected) if injected else "none")
    print("MISSING:", ", ".join(missing) if missing else "none")


if __name__ == "__main__":
    main()
