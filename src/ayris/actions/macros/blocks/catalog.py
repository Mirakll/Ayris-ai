"""The macro editor palette, built from action introspection and native blocks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from ayris.actions.base import ActionSchema, FieldKind, ParamField
from ayris.actions.registry import ActionRegistry

__all__ = [
    "BLOCKS_BY_CATEGORY",
    "BlockCatalog",
    "BlockCategory",
    "BlockMeta",
    "CategoryMeta",
    "list_blocks",
    "list_categories",
    "search",
]


class BlockCategory(StrEnum):
    AUDIO = "audio"
    INPUT = "input"
    SYSTEM = "system"
    LOGIC = "logic"
    FLOW = "flow"
    WEB = "web"
    NOTIFY = "notify"
    YANDEX = "yandex"


@dataclass(frozen=True, slots=True)
class CategoryMeta:
    type: BlockCategory
    title_ru: str
    icon: str


@dataclass(frozen=True, slots=True)
class BlockMeta:
    type: str
    category: BlockCategory
    title_ru: str
    description_ru: str
    icon: str
    fields: tuple[ParamField, ...]
    json_schema: dict[str, Any]
    example: dict[str, Any]
    action_name: str = ""
    available: bool = True
    unavailable_reason: str = ""
    is_dangerous: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        """The one JSON-schema vocabulary the UI uses for every block."""
        return self.json_schema


CATEGORIES: Final = (
    CategoryMeta(BlockCategory.AUDIO, "Голос/Звук", "volume-2"),
    CategoryMeta(BlockCategory.INPUT, "Ввод", "mouse-pointer-2"),
    CategoryMeta(BlockCategory.SYSTEM, "Система", "monitor-cog"),
    CategoryMeta(BlockCategory.LOGIC, "Логика", "git-branch"),
    CategoryMeta(BlockCategory.FLOW, "Поток", "route"),
    CategoryMeta(BlockCategory.WEB, "Сеть/Веб", "globe"),
    CategoryMeta(BlockCategory.NOTIFY, "Уведомления", "bell"),
    CategoryMeta(BlockCategory.YANDEX, "Яндекс Музыка", "music-2"),
)

BLOCKS_BY_CATEGORY: Final[dict[BlockCategory, tuple[str, ...]]] = {
    BlockCategory.AUDIO: ("Say", "PlaySound", "StopSound", "SetVolume", "SetTTSVoice"),
    BlockCategory.INPUT: (
        "KeyPress",
        "KeyDown",
        "KeyUp",
        "TypeText",
        "MouseClick",
        "MouseMove",
        "MouseDrag",
        "MouseWheel",
    ),
    BlockCategory.SYSTEM: (
        "RunApp",
        "RunShell",
        "CloseApp",
        "FocusWindow",
        "WindowState",
        "SetBrightness",
        "SetWifi",
        "SetBluetooth",
        "PowerAction",
        "Screenshot",
        "ClipboardSet",
        "ClipboardGet",
    ),
    BlockCategory.LOGIC: (
        "If",
        "Else",
        "Switch",
        "While",
        "For",
        "Try",
        "Catch",
        "SetVar",
        "GetVar",
        "ArrayPush",
        "ArrayGet",
        "DictSet",
        "DictGet",
    ),
    BlockCategory.FLOW: ("Wait", "CallCommand", "Return", "Break", "Continue", "Sleep"),
    BlockCategory.WEB: ("WebRequest", "OpenURL", "SearchWeb"),
    BlockCategory.NOTIFY: ("ToastNotify", "OverlayLog"),
    BlockCategory.YANDEX: (
        "YMPlay",
        "YMPause",
        "YMNext",
        "YMPrev",
        "YMLike",
        "YMSearch",
        "YMPlaylist",
    ),
}

_TITLES: Final = {name: name for names in BLOCKS_BY_CATEGORY.values() for name in names}
_TITLES.update(
    {
        "Wait": "Ожидание",
        "Sleep": "Пауза",
        "WebRequest": "Веб-запрос",
        "ToastNotify": "Показать уведомление",
        "OverlayLog": "Строка в оверлее",
        "If": "Если",
        "Else": "Иначе",
        "While": "Пока",
        "For": "Цикл",
        "Try": "Попытка",
        "Catch": "Обработка ошибки",
        "Return": "Вернуть",
    }
)
_EXAMPLES: Final[dict[str, dict[str, Any]]] = {
    "Say": {"text": "Готово"},
    "PlaySound": {"sound": "notification"},
    "SetVolume": {"level": 50},
    "SetTTSVoice": {"voice": "irina"},
    "KeyPress": {"combo": "ctrl+s"},
    "KeyDown": {"combo": "shift"},
    "TypeText": {"text": "Привет"},
    "MouseWheel": {"clicks": -3},
    "RunApp": {"app": "notepad"},
    "RunShell": {"command": "echo hello"},
    "CloseApp": {"app": "notepad"},
    "FocusWindow": {"title": "Блокнот"},
    "SetBrightness": {"level": 70},
    "PowerAction": {"operation": "sleep"},
    "ClipboardSet": {"text": "текст"},
    "OpenURL": {"url": "https://example.com"},
    "SearchWeb": {"query": "погода"},
    "YMSearch": {"query": "джаз"},
    "YMPlaylist": {"name": "Мне нравится"},
}


def _field(
    name: str,
    kind: FieldKind = FieldKind.TEXT,
    *,
    required: bool = True,
    default: Any = None,
    choices: tuple[str, ...] = (),
) -> ParamField:
    from ayris.actions.base import Choice

    return ParamField(
        name,
        name,
        FieldKind.CHOICE if choices else kind,
        required=required,
        default=default,
        choices=tuple(Choice(item, item) for item in choices),
    )


_NATIVE_FIELDS: Final[dict[str, tuple[ParamField, ...]]] = {
    "If": (_field("condition"),),
    "Switch": (_field("value"),),
    "While": (_field("condition"), _field("max_iterations", FieldKind.INTEGER, required=False)),
    "For": (_field("var"), _field("items", FieldKind.LIST, required=False)),
    "Try": (_field("error_var", required=False, default="error"),),
    "SetVar": (_field("name"), _field("value")),
    "GetVar": (_field("name"), _field("into", required=False)),
    "ArrayPush": (_field("name"), _field("value")),
    "ArrayGet": (_field("name"), _field("index", FieldKind.INTEGER)),
    "DictSet": (_field("name"), _field("key"), _field("value")),
    "DictGet": (_field("name"), _field("key")),
    "Wait": (
        _field("ms", FieldKind.INTEGER, required=False),
        _field("seconds", FieldKind.NUMBER, required=False),
        _field("condition", required=False),
        _field("timeout_ms", FieldKind.INTEGER, required=False, default=30000),
        _field("poll_ms", FieldKind.INTEGER, required=False, default=50),
    ),
    "Sleep": (
        _field("ms", FieldKind.INTEGER, required=False),
        _field("seconds", FieldKind.NUMBER, required=False),
    ),
    "CallCommand": (_field("command"), _field("args", FieldKind.OBJECT, required=False)),
    "Return": (_field("value", required=False),),
    "WebRequest": (
        _field("method", choices=("GET", "POST"), required=False, default="GET"),
        _field("url"),
        _field("headers", FieldKind.OBJECT, required=False),
        _field("body", FieldKind.OBJECT, required=False),
        _field("timeout_ms", FieldKind.INTEGER, required=False, default=10000),
        _field("json", FieldKind.BOOLEAN, required=False, default=False),
        _field("json_path", required=False),
        _field("into", required=False),
        _field("max_bytes", FieldKind.INTEGER, required=False, default=1000000),
    ),
    "ToastNotify": (
        _field("title"),
        _field("message", required=False),
        _field("icon", required=False),
        _field("action", required=False),
        _field("level", choices=("info", "warning", "error"), required=False, default="info"),
    ),
    "OverlayLog": (
        _field("message"),
        _field(
            "level", choices=("debug", "info", "warning", "error"), required=False, default="info"
        ),
    ),
}
_NATIVE_EXAMPLES: Final[dict[str, dict[str, Any]]] = {
    "If": {"condition": "{ready}"},
    "Switch": {"value": "{mode}"},
    "While": {"condition": "{attempt} < 3"},
    "For": {"var": "item", "items": [1, 2]},
    "Try": {"error_var": "error"},
    "SetVar": {"name": "answer", "value": 42},
    "GetVar": {"name": "answer"},
    "ArrayPush": {"name": "items", "value": "новое"},
    "ArrayGet": {"name": "items", "index": 0},
    "DictSet": {"name": "data", "key": "status", "value": "ok"},
    "DictGet": {"name": "data", "key": "status"},
    "Wait": {"seconds": 1},
    "Sleep": {"ms": 500},
    "CallCommand": {"command": "Рабочий режим"},
    "Return": {"value": "готово"},
    "Break": {},
    "Continue": {},
    "WebRequest": {
        "method": "GET",
        "url": "https://example.com/api",
        "json": True,
        "json_path": "$.result",
        "into": "result",
    },
    "ToastNotify": {"title": "Готово", "message": "Команда выполнена"},
    "OverlayLog": {"message": "Шаг выполнен", "level": "info"},
    "Else": {},
    "Catch": {},
}


def _json_schema(fields: tuple[ParamField, ...]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    types = {
        FieldKind.TEXT: "string",
        FieldKind.INTEGER: "integer",
        FieldKind.NUMBER: "number",
        FieldKind.BOOLEAN: "boolean",
        FieldKind.LIST: "array",
        FieldKind.OBJECT: "object",
    }
    for item in fields:
        node: dict[str, Any] = {"type": types.get(item.kind, "string"), "title": item.label_ru}
        if item.choices:
            node["enum"] = [choice.value for choice in item.choices]
        if item.default is not None:
            node["default"] = item.default
        properties[item.name] = node
        if item.required:
            required.append(item.name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _example(schema: ActionSchema) -> dict[str, Any]:
    values = dict(_EXAMPLES.get(schema.name, {}))
    for item in schema.fields:
        if item.name in values or not item.required:
            continue
        if item.choices:
            values[item.name] = item.choices[0].value
        elif item.kind is FieldKind.TEXT:
            values[item.name] = "пример"
        elif item.kind is FieldKind.INTEGER:
            values[item.name] = int(item.minimum or 1)
        elif item.kind is FieldKind.NUMBER:
            values[item.name] = float(item.minimum or 1)
        elif item.kind is FieldKind.BOOLEAN:
            values[item.name] = False
        elif item.kind is FieldKind.LIST:
            values[item.name] = []
        else:
            values[item.name] = {}
    return values


class BlockCatalog:
    def __init__(self, registry: ActionRegistry | None = None) -> None:
        self.registry = registry if registry is not None else ActionRegistry()
        if registry is None:
            self.registry.discover()
        self._blocks = self._build()

    def _build(self) -> dict[str, BlockMeta]:
        result: dict[str, BlockMeta] = {}
        native = set(_NATIVE_EXAMPLES)
        for category, names in BLOCKS_BY_CATEGORY.items():
            for name in names:
                if self.registry.has(name):
                    schema = self.registry.describe(name)
                    result[name] = BlockMeta(
                        name,
                        category,
                        schema.title_ru,
                        schema.description_ru or f"Выполняет действие {name}.",
                        "box",
                        schema.fields,
                        dict(schema.json_schema),
                        _example(schema),
                        action_name=name,
                        is_dangerous=schema.is_dangerous,
                    )
                    continue
                fields = _NATIVE_FIELDS.get(name, ())
                is_native = name in native
                reason = "" if is_native else f"Действие {name} ещё не подключено в этой сборке."
                result[name] = BlockMeta(
                    name,
                    category,
                    _TITLES[name],
                    f"Блок «{_TITLES[name]}» для макрокоманд.",
                    "box",
                    fields,
                    _json_schema(fields),
                    dict(_NATIVE_EXAMPLES.get(name, _EXAMPLES.get(name, {}))),
                    available=is_native,
                    unavailable_reason=reason,
                    is_dangerous=name in {"RunShell", "WebRequest"},
                )
        return result

    def get(self, block_type: str) -> BlockMeta:
        return self._blocks[block_type]

    def list_categories(self) -> tuple[CategoryMeta, ...]:
        return CATEGORIES

    def list_blocks(self, category: BlockCategory | str) -> tuple[BlockMeta, ...]:
        kind = BlockCategory(category)
        return tuple(self._blocks[name] for name in BLOCKS_BY_CATEGORY[kind])

    def search(self, query: str) -> tuple[BlockMeta, ...]:
        needle = query.strip().casefold()
        return tuple(
            block
            for block in self._blocks.values()
            if not needle
            or needle in " ".join((block.type, block.title_ru, block.description_ru)).casefold()
        )


_DEFAULT: BlockCatalog | None = None


def _default() -> BlockCatalog:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = BlockCatalog()
    return _DEFAULT


def list_categories() -> tuple[CategoryMeta, ...]:
    return _default().list_categories()


def list_blocks(category: BlockCategory | str) -> tuple[BlockMeta, ...]:
    return _default().list_blocks(category)


def search(query: str) -> tuple[BlockMeta, ...]:
    return _default().search(query)
