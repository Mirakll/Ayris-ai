"""Conservative, non-executing import of basic AutoHotkey scripts."""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path

from ayris.actions.macros.importers.base import (
    Importer,
    ImportNotice,
    ImportResult,
    UnsupportedItem,
)
from ayris.actions.macros.schema import (
    ActionBlock,
    CommandModel,
    HotkeyTrigger,
    VariableModel,
)
from ayris.actions.macros.validator import MacroValidationError, ensure_valid
from ayris.core.models import VariableType
from ayris.utils.hotkeys import HotkeyNotationError, canonical_hotkey

_HOTKEY = re.compile(r"^(?P<hotkey>[^:;]+?)::(?P<inline>.*)$")
_ASSIGN = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*(?::=|=(?!=))\s*(?P<value>.*)$")
_SEND_TOKEN = re.compile(r"\{([^{}]+)\}")


class UnknownLinePolicy(StrEnum):
    """How source lines outside the supported AHK subset are retained."""

    DISABLED_RUN_SHELL = "disabled_run_shell"
    SKIP = "skip"


class AutoHotkeyImporter(Importer):
    """Parse hotkey labels and a deliberately small safe command subset."""

    def __init__(
        self, unknown_policy: UnknownLinePolicy = UnknownLinePolicy.DISABLED_RUN_SHELL
    ) -> None:
        self.unknown_policy = unknown_policy

    def parse(self, path: Path) -> ImportResult:
        path = Path(path)
        result = ImportResult(source=path, profile_name=path.stem)
        lines = _read_script(path).splitlines()
        index = 0
        while index < len(lines):
            stripped = lines[index].strip()
            match = _HOTKEY.match(stripped)
            if match is None:
                index += 1
                continue
            result.total_commands += 1
            source_hotkey = match.group("hotkey").strip()
            inline = match.group("inline").strip()
            body: list[tuple[int, str]] = []
            if inline:
                body.append((index + 1, inline))
            else:
                cursor = index + 1
                while cursor < len(lines):
                    line = lines[cursor].strip()
                    if line.casefold() == "return":
                        break
                    if _HOTKEY.match(line):
                        break
                    body.append((cursor + 1, line))
                    cursor += 1
                index = cursor
            command = self._command(path, source_hotkey, body, result)
            if command is not None:
                result.commands.append(command)
                result.folders.add((path.stem,))
            index += 1
        return result

    def _command(
        self,
        path: Path,
        source_hotkey: str,
        body: list[tuple[int, str]],
        result: ImportResult,
    ) -> CommandModel | None:
        try:
            hotkey = canonical_hotkey(source_hotkey)
            trigger = HotkeyTrigger(combo=hotkey)
        except (HotkeyNotationError, ValueError) as exc:
            result.skipped["некорректный хоткей"] += 1
            result.warnings.append(ImportNotice(str(exc), command=source_hotkey, source=str(path)))
            return None

        actions: list[ActionBlock] = []
        variables: dict[str, VariableModel] = {}
        for line_number, raw in body:
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            parsed = _parse_line(line, variables)
            if parsed is not None:
                actions.extend(parsed)
                continue
            source = f"{path.name}:{line_number}"
            result.unsupported.append(UnsupportedItem(source_hotkey, _instruction_name(line), line))
            if self.unknown_policy is UnknownLinePolicy.SKIP:
                result.warnings.append(
                    ImportNotice("строка пропущена", command=source_hotkey, source=source)
                )
            else:
                actions.append(
                    ActionBlock(
                        type="RunShell",
                        params={"command": line},
                        enabled=False,
                        comment="Импортировано из AHK без выполнения",
                    )
                )
        try:
            command = CommandModel(
                name=f"AHK {hotkey}",
                description=f"Импортировано из {path.name}: {source_hotkey}",
                folder=[path.stem],
                triggers=[trigger],
                actions=actions,
                variables=list(variables.values()),
            )
            ensure_valid(command)
        except (ValueError, MacroValidationError) as exc:
            result.skipped["команда не прошла проверку"] += 1
            result.warnings.append(ImportNotice(str(exc), command=source_hotkey, source=str(path)))
            return None
        return command


def _read_script(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ("utf-8-sig", "utf-16", "cp1251")
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Не удалось определить кодировку AHK-файла: {path}")


def _parse_line(line: str, variables: dict[str, VariableModel]) -> list[ActionBlock] | None:
    command, separator, argument = line.partition(",")
    keyword = command.strip().casefold()
    argument = argument.strip() if separator else ""
    if keyword == "run":
        return [_run(argument)]
    if keyword in {"send", "sendinput", "sendraw"}:
        return _send(argument, raw=keyword == "sendraw")
    if keyword == "winactivate":
        return [ActionBlock(type="FocusWindow", params={"title": argument})]
    if keyword == "sleep":
        try:
            milliseconds = max(0, int(argument))
        except ValueError:
            return None
        return [ActionBlock(type="Wait", params={"ms": milliseconds})]
    assignment = _ASSIGN.match(line)
    if assignment is not None:
        name = assignment.group("name")
        value = _literal(assignment.group("value"))
        variables[name] = VariableModel(name=name, type=_variable_type(value), default=value)
        return [ActionBlock(type="SetVar", params={"name": name, "value": value})]
    return None


def _run(argument: str) -> ActionBlock:
    target, separator, arguments = argument.partition(",")
    target = target.strip()
    arguments = arguments.strip() if separator else ""
    shell = Path(target.strip('"')).suffix.casefold() in {".bat", ".cmd", ".ps1"}
    if shell:
        return ActionBlock(
            type="RunShell",
            params={"command": " ".join(filter(None, (target, arguments)))},
            enabled=False,
            comment="Сценарий отключён после импорта AHK",
        )
    return ActionBlock(
        type="RunApp",
        params={"app": target.strip('"'), "arguments": arguments},
    )


def _send(text: str, *, raw: bool) -> list[ActionBlock]:
    if raw:
        return (
            [ActionBlock(type="TypeText", params={"text": _escape_template(text)})] if text else []
        )
    actions: list[ActionBlock] = []
    position = 0
    for match in _SEND_TOKEN.finditer(text):
        if match.start() > position:
            actions.append(
                ActionBlock(type="TypeText", params={"text": text[position : match.start()]})
            )
        token = match.group(1).strip()
        lowered = token.casefold()
        if lowered.endswith(" down"):
            actions.append(
                ActionBlock(type="KeyDown", params={"combo": token[:-5].strip().casefold()})
            )
        elif lowered.endswith(" up"):
            actions.append(
                ActionBlock(type="KeyUp", params={"combo": token[:-3].strip().casefold()})
            )
        else:
            actions.append(ActionBlock(type="KeyPress", params={"combo": token.casefold()}))
        position = match.end()
    if position < len(text):
        actions.append(ActionBlock(type="TypeText", params={"text": text[position:]}))
    return actions


def _escape_template(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def _literal(source: str) -> str | int | float | bool:
    value = source.strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1].replace('""', '"')
    if value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value.replace(",", "."))
        except ValueError:
            return value


def _variable_type(value: object) -> VariableType:
    if isinstance(value, bool):
        return VariableType.BOOL
    if isinstance(value, int):
        return VariableType.INT
    if isinstance(value, float):
        return VariableType.FLOAT
    return VariableType.STRING


def _instruction_name(line: str) -> str:
    return re.split(r"[,(\s]", line, maxsplit=1)[0] or "неизвестная строка"
