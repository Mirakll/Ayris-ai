"""Системные промпты чата и NLU: шаблон из ресурсов + персона из настроек.

Промпт собирается из трёх слоёв, чтобы каждый можно было менять отдельно:

* **персона** — короткая строка из «ИИ» (``ai.chat_system_prompt`` /
  ``ai.nlu_system_prompt``), которую правит пользователь: роль, тон, язык;
* **шаблон** — подробные правила (структура ответа, контракт JSON для NLU,
  место для списка команд). Лежит в ``resources/prompts/{chat,nlu}.txt`` и
  переопределяется файлом с тем же именем в каталоге профиля;
* **подстановки** — список доступных команд для NLU, который собирает
  :mod:`ayris.nlu.llm.tools` по релевантности к фразе.

Сборка вынесена из пайплайна сюда, чтобы задача 64 могла показать итоговый
промпт в предпросмотре, не поднимая модель. Подстановка идёт через
:meth:`str.replace` по маркеру :data:`COMMANDS_PLACEHOLDER`, а не через
:meth:`str.format`: шаблон NLU полон фигурных скобок JSON, и ``format`` на нём
падал бы на каждой из них.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Literal

from ayris.core.paths import executable_dir

__all__ = [
    "COMMANDS_PLACEHOLDER",
    "PROMPTS_SUBDIR",
    "PromptKind",
    "build_chat_prompt",
    "build_nlu_prompt",
    "load_template",
    "prompts_dir",
]

_log = logging.getLogger("ayris.nlu.llm.prompts")

PromptKind = Literal["chat", "nlu"]

#: Куда шаблон NLU вставляет список команд. Двойные скобки не встречаются в
#: обычном тексте промпта, поэтому замена не заденет ничего лишнего.
COMMANDS_PLACEHOLDER: Final = "{{COMMANDS}}"

#: Подкаталог ресурсов и профиля, где лежат ``chat.txt`` и ``nlu.txt``.
PROMPTS_SUBDIR: Final = "prompts"

# Встроенный запасной вариант на случай, когда файла ресурса нет (сломанная
# сборка, обрезанный портейбл). Он короче ресурсного, но самодостаточен:
# лучше урезанный промпт, чем пустой system и модель без правил.
_FALLBACK_CHAT: Final = (
    "Ты отвечаешь голосом. Пиши только по-русски, коротко, без markdown и эмодзи. "
    "Не выдумывай факты; если нужны свежие данные, скажи об этом честно."
)
_FALLBACK_NLU: Final = (
    "Определи команду из списка и верни строго JSON "
    '{"command": <имя или null>, "params": {}, "confidence": <0..1>} '
    "без текста вокруг. Ничего не придумывай.\n\nДоступные команды:\n" + COMMANDS_PLACEHOLDER
)

_FALLBACKS: Final[dict[str, str]] = {"chat": _FALLBACK_CHAT, "nlu": _FALLBACK_NLU}


def prompts_dir() -> Path:
    """Каталог с шаблонами промптов в ресурсах приложения."""
    return executable_dir() / "resources" / PROMPTS_SUBDIR


def load_template(
    kind: PromptKind,
    *,
    resources_dir: Path | None = None,
    override_dir: Path | None = None,
) -> str:
    """Текст шаблона ``kind``: сперва переопределение пользователя, затем ресурс.

    ``override_dir`` — каталог профиля, где пользователь может положить свой
    ``chat.txt`` или ``nlu.txt`` и заменить встроенный. Если там ничего нет,
    берётся файл из ``resources``; если и его нет — встроенная строка-заглушка,
    чтобы система никогда не осталась с пустым промптом.
    """
    for directory in (override_dir, resources_dir or prompts_dir()):
        if directory is None:
            continue
        text = _read(directory / f"{kind}.txt")
        if text is not None:
            return text
    return _FALLBACKS[kind]


def build_chat_prompt(
    persona: str,
    *,
    resources_dir: Path | None = None,
    override_dir: Path | None = None,
) -> str:
    """Системный промпт чата: персона пользователя над шаблоном поведения."""
    template = load_template("chat", resources_dir=resources_dir, override_dir=override_dir)
    return _join(persona, template)


def build_nlu_prompt(
    persona: str,
    commands: str,
    *,
    resources_dir: Path | None = None,
    override_dir: Path | None = None,
) -> str:
    """Системный промпт NLU с подставленным списком команд.

    ``commands`` — уже отрендеренный блок из :mod:`ayris.nlu.llm.tools`. Пустой
    список подставляется честной пометкой, а не пустотой, чтобы модель поняла,
    что подходящих команд нет, и вернула ``null`` вместо выдумки.
    """
    template = load_template("nlu", resources_dir=resources_dir, override_dir=override_dir)
    block = commands.strip() or "(команд нет)"
    if COMMANDS_PLACEHOLDER in template:
        template = template.replace(COMMANDS_PLACEHOLDER, block)
    else:
        template = f"{template}\n\nДоступные команды:\n{block}"
    return _join(persona, template)


def _join(persona: str, template: str) -> str:
    head = persona.strip()
    body = template.strip()
    if head and body:
        return f"{head}\n\n{body}"
    return head or body


def _read(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8-sig").strip()
    except OSError as exc:
        _log.warning("не прочитать шаблон промпта %s: %s", path, exc)
        return None
    return text or None
