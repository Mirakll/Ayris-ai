"""Safe, tolerant import of VoiceAttack XML profiles."""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ayris.actions.macros.importers.base import (
    ImportedSound,
    Importer,
    ImportNotice,
    ImportResult,
    UnsupportedItem,
)
from ayris.actions.macros.importers.vap_mapping import map_vap_action
from ayris.actions.macros.schema import (
    ActionBlock,
    CommandModel,
    HotkeyTrigger,
    SoundBinding,
    SoundSource,
    SoundStage,
    TriggerModel,
    VariableModel,
    VoiceTrigger,
)
from ayris.actions.macros.validator import MacroValidationError, ensure_valid
from ayris.core.models import VariableType

_MAX_XML_BYTES = 32 * 1024 * 1024
_UNSAFE_XML = re.compile(rb"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_COMMAND_TAGS = {"command", "commanditem", "profilecommand"}
_ACTION_CONTAINERS = {"actions", "actionsequence", "commandactions"}
_ACTION_TYPE_FIELDS = ("actiontype", "type", "name")
_VAR_PATTERN = re.compile(r"[^\w]", re.UNICODE)


class VoiceAttackImporter(Importer):
    """Read VoiceAttack data without importing code or resolving entities."""

    def parse(self, path: Path) -> ImportResult:
        path = Path(path)
        root = _safe_root(path)
        result = ImportResult(
            source=path,
            profile_name=_first_text(root, "profilename", "name") or path.stem,
        )
        command_nodes = [node for node in root.iter() if _tag(node) in _COMMAND_TAGS]
        result.total_commands = len(command_nodes)
        for index, node in enumerate(command_nodes, 1):
            command = self._parse_command(node, index, path, result)
            if command is not None:
                result.commands.append(command)
                result.folders.add(tuple(command.folder))
        return result

    def _parse_command(
        self,
        node: ET.Element,
        index: int,
        source_path: Path,
        result: ImportResult,
    ) -> CommandModel | None:
        name = _first_text(node, "commandstring", "caption", "name") or f"Команда {index}"
        phrase = _first_text(node, "spokenphrase", "phrase", "commandstring")
        prefix = _first_text(node, "prefix", "commandprefix")
        category = _first_text(node, "category", "group")
        folder = [part.strip() for part in re.split(r"[\/]", category) if part.strip()]
        enabled = _bool(_first_text(node, "enabled", "isactive", "active"), True)
        triggers: list[TriggerModel] = []
        if phrase:
            for expanded in expand_voice_phrases(phrase):
                full_phrase = " ".join(part for part in (prefix, expanded) if part).strip()
                if full_phrase:
                    triggers.append(VoiceTrigger(phrase=full_phrase, enabled=enabled))
        hotkey = _first_text(node, "hotkey", "shortcut", "keybind")
        if hotkey:
            try:
                triggers.append(HotkeyTrigger(combo=hotkey, enabled=enabled))
            except ValueError as exc:
                result.warnings.append(ImportNotice(str(exc), command=name, source=hotkey))

        actions: list[ActionBlock] = []
        variables: dict[str, VariableModel] = {}
        for action_node in _action_nodes(node):
            fields = _fields(action_node)
            kind = _action_kind(action_node, fields)
            block = map_vap_action(kind, fields)
            if block is None:
                result.unsupported.append(
                    UnsupportedItem(name, kind or _tag(action_node), _short_xml(action_node))
                )
                continue
            if block.type == "SetVar":
                variable = _variable(fields)
                if variable is None:
                    result.warnings.append(
                        ImportNotice("некорректное имя переменной", command=name, source=kind)
                    )
                    continue
                variables[variable.name] = variable
                block = block.model_copy(
                    update={"params": {"name": variable.name, "value": variable.default}}
                )
            if block.type == "PlaySound":
                original = str(block.params.get("sound", ""))
                sound_path = _external_path(source_path, original)
                if original and sound_path.name:
                    block = block.model_copy(update={"params": {"sound": sound_path.name}})
                    result.sounds.append(ImportedSound(sound_path, sound_path.name, name))
            actions.append(block)

        sounds = _bound_sounds(node, source_path, name, result)
        try:
            command = CommandModel(
                name=name,
                description=f"Импортировано из VoiceAttack: {source_path.name}",
                folder=folder,
                enabled=enabled,
                triggers=triggers,
                actions=actions,
                variables=list(variables.values()),
                sounds=sounds,
            )
            ensure_valid(command)
        except (ValueError, MacroValidationError) as exc:
            result.skipped["команда не прошла проверку"] += 1
            result.warnings.append(ImportNotice(str(exc), command=name, source=source_path.name))
            return None
        return command


def _safe_root(path: Path) -> ET.Element:
    raw = path.read_bytes()
    if len(raw) > _MAX_XML_BYTES:
        raise ValueError(f"VAP-файл больше {_MAX_XML_BYTES // 1024 // 1024} МиБ")
    if _UNSAFE_XML.search(raw):
        raise ValueError("VAP-файл содержит запрещённые DTD или XML-сущности")
    try:
        return ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"Повреждённый VAP/XML: {exc}") from exc


def _tag(node: ET.Element) -> str:
    return str(node.tag).rsplit("}", 1)[-1].casefold()


def _first_text(node: ET.Element, *names: str) -> str:
    for name in names:
        wanted = name.casefold()
        for child in node.iter():
            if child is not node and _tag(child) in _COMMAND_TAGS:
                continue
            if _tag(child) == wanted and child.text and child.text.strip():
                return child.text.strip()
    return ""


def _action_nodes(command: ET.Element) -> list[ET.Element]:
    nodes: list[ET.Element] = []
    for container in command.iter():
        if _tag(container) not in _ACTION_CONTAINERS:
            continue
        for child in list(container):
            if isinstance(child.tag, str):
                nodes.append(child)
    return nodes


def _fields(node: ET.Element) -> dict[str, str]:
    fields: dict[str, str] = {}
    for child in node.iter():
        if child is node or not child.text or not child.text.strip():
            continue
        fields[str(child.tag).rsplit("}", 1)[-1]] = child.text.strip()
    for key, value in node.attrib.items():
        fields.setdefault(key, value)
    return fields


def _action_kind(node: ET.Element, fields: dict[str, str]) -> str:
    lowered = {key.casefold(): value for key, value in fields.items()}
    for name in _ACTION_TYPE_FIELDS:
        if lowered.get(name):
            return lowered[name]
    return str(node.tag).rsplit("}", 1)[-1]


def expand_voice_phrases(source: str, *, limit: int = 256) -> list[str]:
    """Expand VoiceAttack semicolon variants and optional bracket groups."""
    expanded: list[str] = []
    for variant in source.split(";"):
        parts: list[list[str]] = []
        position = 0
        for match in re.finditer(r"\[([^]]*)]", variant):
            if match.start() > position:
                parts.append([variant[position : match.start()]])
            alternatives = [item.strip() for item in re.split(r"[;|]", match.group(1))]
            parts.append(["", *(item for item in alternatives if item)])
            position = match.end()
        parts.append([variant[position:]])
        for combination in itertools.product(*parts):
            phrase = re.sub(r"\s+", " ", "".join(combination)).strip()
            if phrase and phrase not in expanded:
                expanded.append(phrase)
                if len(expanded) >= limit:
                    return expanded
    return expanded


def _variable(fields: dict[str, str]) -> VariableModel | None:
    lowered = {key.casefold(): value for key, value in fields.items()}
    source_name = lowered.get("name") or lowered.get("variable") or ""
    name = _VAR_PATTERN.sub("_", source_name.strip())
    if not name or name[0].isdigit():
        return None
    raw = lowered.get("value", "")
    kind = (lowered.get("vartype") or lowered.get("valuetype") or "string").casefold()
    if kind in {"int", "integer"}:
        try:
            value: str | int | float | bool = int(raw)
        except ValueError:
            value = 0
        variable_type = VariableType.INT
    elif kind in {"float", "double", "decimal"}:
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            value = 0.0
        variable_type = VariableType.FLOAT
    elif kind in {"bool", "boolean"}:
        value = _bool(raw, False)
        variable_type = VariableType.BOOL
    else:
        value = raw
        variable_type = VariableType.STRING
    return VariableModel(name=name, type=variable_type, default=value)


def _bound_sounds(
    node: ET.Element,
    source_path: Path,
    command: str,
    result: ImportResult,
) -> list[SoundBinding]:
    tags = {
        "startsound": SoundStage.ON_START,
        "successsound": SoundStage.ON_SUCCESS,
        "errorsound": SoundStage.ON_ERROR,
    }
    sounds: list[SoundBinding] = []
    for child in node.iter():
        stage = tags.get(_tag(child))
        if stage is None or not child.text or not child.text.strip():
            continue
        external = _external_path(source_path, child.text.strip())
        try:
            sounds.append(SoundBinding(stage=stage, source=SoundSource.FILE, value=external.name))
        except ValueError as exc:
            result.warnings.append(ImportNotice(str(exc), command=command, source=str(external)))
            continue
        result.sounds.append(ImportedSound(external, external.name, command))
    return sounds


def _external_path(vap_path: Path, source: str) -> Path:
    candidate = Path(source.strip().strip('"'))
    return candidate if candidate.is_absolute() else vap_path.parent / candidate


def _bool(source: str, default: bool) -> bool:
    if not source:
        return default
    return source.strip().casefold() not in {"0", "false", "no", "off"}


def _short_xml(node: ET.Element) -> str:
    return ET.tostring(node, encoding="unicode", short_empty_elements=True)[:300]
