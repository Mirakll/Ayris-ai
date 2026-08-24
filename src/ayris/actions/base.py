"""What an action is: its parameters, its metadata, and how it describes itself.

An action is the smallest thing Ayris can be asked to do — ``RunApp``,
``SetVolume``, ``Screenshot``. Sections 6 and 7.2 list about seventy of them, and
they are written by nine different task chats plus plugin authors, so the shape
has to be worth repeating seventy times.

**Parameters are a pydantic model, not ``**kwargs``.** The same nested
:class:`Action.Params` is the validation rule at call time, the contract with the
``.ayris`` export format, and the source the macro editor draws its form from. One
declaration, three consumers — which is why renaming a field is a migration and
not an edit (task 30 owns that migration).

**Metadata is data, not overrides.** ``require_admin``, ``is_dangerous`` and
``timeout_ms`` live in :class:`ActionMeta` where the registry, the confirmation
dialog (task 40) and the elevation check (task 39) can read them *before* running
anything. A flag expressed as an overridden method would only be discoverable by
calling it.

**Sync and async are one call path.** Most actions are a blocking WinAPI call and
want :meth:`Action.run`; a few — a web request, a WinRT await — want
:meth:`Action.arun`. An action implements whichever fits and the base class
supplies the other, so the registry has exactly one way to invoke anything:
:meth:`Action.run` off the UI thread, :meth:`Action.arun` inside a loop.

**Introspection is generated, never hand-written.** :func:`build_schema` turns a
``Params`` model into :class:`ActionSchema`: field kinds, ranges, enum choices,
defaults, Russian labels. The editor renders forms from that. Nothing in the UI
knows that ``WindowState`` has a ``state`` parameter, which is the only way a
seventy-block editor stays maintainable.

Field extras the schema understands, passed as
``Field(json_schema_extra={...})``:

``secret``
    Value is masked in the log, in the audit trail and on the event bus.
``multiline``
    Render as a text area rather than a single line.
``choices_ru``
    Mapping of enum value to Russian label, for readable dropdowns.
``unit_ru``
    Unit suffix shown next to a number field, e.g. ``"мс"``.
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Final

from pydantic import BaseModel, ConfigDict

from ayris.actions.result import ActionResult
from ayris.core.errors import ActionError, ActionUnavailable
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "Action",
    "ActionCategory",
    "ActionMeta",
    "ActionParams",
    "ActionSchema",
    "Choice",
    "FieldKind",
    "ParamField",
    "build_schema",
    "mask_params",
    "params_to_json",
]

_log = get_logger(__name__)

#: Ceiling for one action, in milliseconds. Ten seconds is generous for a WinAPI
#: call and short enough that a wedged action does not look like a hung
#: assistant. ``0`` means "no limit" and is for the blocks that legitimately
#: wait: ``Wait``, ``CallCommand``, a download.
DEFAULT_TIMEOUT_MS: Final = 10_000

#: ``RunApp`` or ``myplugin.RunApp``. PascalCase because these names are the block
#: names the user sees in the editor and the strings inside exported ``.ayris``
#: files; the plugin prefix is a lowercase slug so the two halves never blur.
_NAME_PATTERN: Final = re.compile(
    r"^(?:(?P<plugin>[a-z][a-z0-9_-]*)\.)?(?P<action>[A-Z][A-Za-z0-9]*)$"
)

#: Placeholder written wherever a secret parameter would have been.
SECRET_MASK: Final = "***"


class ActionCategory(StrEnum):
    """Top level of the block tree in the macro editor (section 7.2).

    Declaration order is the order the editor shows, so the groups a user reaches
    for most often come first.
    """

    APPS = "apps"
    WINDOWS = "windows"
    AUDIO = "audio"
    DISPLAY = "display"
    INPUT = "input"
    SYSTEM = "system"
    WEB = "web"
    MEDIA = "media"
    CAPTURE = "capture"
    CLIPBOARD = "clipboard"
    TIMERS = "timers"
    NOTIFY = "notify"
    LOGIC = "logic"
    FLOW = "flow"
    PLUGIN = "plugin"

    @property
    def title_ru(self) -> str:
        """Russian heading for this group in the editor."""
        return _CATEGORY_TITLES[self]


_CATEGORY_TITLES: Final[Mapping[ActionCategory, str]] = {
    ActionCategory.APPS: "Программы",
    ActionCategory.WINDOWS: "Окна и рабочие столы",
    ActionCategory.AUDIO: "Звук и голос",
    ActionCategory.DISPLAY: "Экран",
    ActionCategory.INPUT: "Клавиатура и мышь",
    ActionCategory.SYSTEM: "Система, сеть и питание",
    ActionCategory.WEB: "Браузер и веб",
    ActionCategory.MEDIA: "Музыка и медиа",
    ActionCategory.CAPTURE: "Скриншоты и распознавание",
    ActionCategory.CLIPBOARD: "Буфер обмена",
    ActionCategory.TIMERS: "Таймеры и напоминания",
    ActionCategory.NOTIFY: "Уведомления",
    ActionCategory.LOGIC: "Логика и переменные",
    ActionCategory.FLOW: "Поток выполнения",
    ActionCategory.PLUGIN: "Плагины",
}


@dataclass(frozen=True, slots=True)
class ActionMeta:
    """Everything about an action that is knowable without running it.

    Args:
        name: Block name, ``PascalCase``. Unique across the registry.
        category: Group in the editor's block tree.
        title_ru: Short Russian label, e.g. «Запустить программу».
        description_ru: One Russian sentence for the tooltip and the docs.
        require_admin: Needs elevation. Checked by the registry before the call.
        is_dangerous: Worth confirming out loud (section 11): shutdown, delete,
            anything the user cannot take back.
        supports_undo: The action can reverse a call given its ``undo_token``.
        timeout_ms: Ceiling for one call. ``0`` disables the limit.

    Raises:
        ValueError: The metadata itself is malformed — a bug in the module that
            declared it, caught at import rather than at the first call.
    """

    name: str
    category: ActionCategory
    title_ru: str
    description_ru: str = ""
    require_admin: bool = False
    is_dangerous: bool = False
    supports_undo: bool = False
    timeout_ms: int = DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.match(self.name):
            raise ValueError(
                f"action name {self.name!r} must be PascalCase, optionally prefixed "
                "with a lowercase plugin slug (e.g. 'RunApp' or 'myplugin.RunApp')"
            )
        if not self.title_ru.strip():
            raise ValueError(f"action {self.name} needs a Russian title")
        if self.timeout_ms < 0:
            raise ValueError(f"action {self.name} has a negative timeout: {self.timeout_ms}")

    @property
    def plugin(self) -> str:
        """Plugin slug this action came from, or ``""`` for a built-in one."""
        match = _NAME_PATTERN.match(self.name)
        return match.group("plugin") or "" if match else ""

    @property
    def short_name(self) -> str:
        """Block name without the plugin prefix."""
        match = _NAME_PATTERN.match(self.name)
        return match.group("action") if match else self.name

    @property
    def timeout_s(self) -> float | None:
        """Timeout as seconds for ``wait_for``, or ``None`` when unlimited."""
        return self.timeout_ms / 1000.0 if self.timeout_ms > 0 else None

    def with_prefix(self, plugin: str) -> ActionMeta:
        """Copy renamed for a plugin: ``RunApp`` becomes ``myplugin.RunApp``."""
        slug = plugin.strip().lower()
        if not slug:
            return self
        return replace(self, name=f"{slug}.{self.short_name}")


class ActionParams(BaseModel):
    """Base for every action's parameter model.

    ``extra="forbid"`` is deliberate: a macro exported with a misspelled key is a
    broken macro, and silently dropping the key would make it fail later and
    somewhere else. ``frozen=True`` keeps an action from editing its own input,
    which matters once the same params are handed to the audit trail.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def secret_now(self) -> frozenset[str]:
        """Extra field names to mask for *this* call, on top of the schema ones.

        ``json_schema_extra={"secret": True}`` marks a field that is always a
        secret, which is what the editor draws as a password box. Some fields are
        only sometimes one: the text handed to ``ClipboardSet`` is an ordinary
        note when a macro copies a link and a password when it copies a password,
        and the caller is the only one who knows which. Overriding this makes the
        decision per call, and :func:`mask_params` honours both sources — so the
        audit row keeps the readable value in the first case and hides it in the
        second.
        """
        return frozenset()


class Action(ABC):
    """One executable action.

    A subclass declares :attr:`meta`, a nested ``Params`` model, and exactly one
    of :meth:`run` or :meth:`arun`::

        @register
        class RunApp(Action):
            meta = ActionMeta(
                name="RunApp",
                category=ActionCategory.APPS,
                title_ru="Запустить программу",
            )

            class Params(ActionParams):
                app: str = Field(description="Название программы")

            def run(self, params: RunApp.Params) -> ActionResult[int]:
                ...

    Instances are created once, when the registry picks the class up, and reused
    for every call. An action therefore must not keep per-call state on ``self``:
    two macros can run at the same time (``commands.max_parallel``).
    """

    #: Set by every concrete subclass. Read by the registry before instantiating.
    meta: ClassVar[ActionMeta]

    class Params(ActionParams):
        """Parameters this action accepts. Subclasses override with typed fields."""

    @classmethod
    def params_model(cls) -> type[ActionParams]:
        """The ``Params`` model in force for this class, inherited or its own."""
        return cls.Params

    @classmethod
    def implements_sync(cls) -> bool:
        """Whether this class overrides :meth:`run`."""
        return cls.run is not Action.run

    @classmethod
    def implements_async(cls) -> bool:
        """Whether this class overrides :meth:`arun`."""
        return cls.arun is not Action.arun

    def run(self, params: Any) -> ActionResult[Any]:
        """Execute synchronously. Called on a worker thread, never on the UI one.

        The default drives an async-only action to completion, so that a macro —
        which is synchronous — can call anything in the registry.
        """
        if not type(self).implements_async():
            raise NotImplementedError(f"{type(self).__name__} implements neither run() nor arun()")
        if _loop_is_running():
            raise ActionError(
                f"{self.meta.name} is async-only and run() was called from a running "
                "event loop; await registry.aexecute() instead",
                user_message=self.meta.title_ru + ": нельзя выполнить здесь.",
            )
        return asyncio.run(self.arun(params))

    async def arun(self, params: Any) -> ActionResult[Any]:
        """Execute inside an event loop.

        The default moves a synchronous action onto a worker thread rather than
        calling it inline: most of them block on WinAPI, and blocking the loop
        would stall everything else awaiting on it.
        """
        if not type(self).implements_sync():
            raise NotImplementedError(f"{type(self).__name__} implements neither run() nor arun()")
        return await asyncio.to_thread(self.run, params)

    def undo(self, token: str) -> ActionResult[Any]:
        """Reverse an earlier call described by ``token``.

        Only meaningful when :attr:`ActionMeta.supports_undo` is set; the default
        refuses rather than pretending to succeed.
        """
        raise ActionUnavailable(
            f"{self.meta.name} does not support undo (token={token!r})",
            user_message="Это действие нельзя отменить.",
        )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.meta.name}>"


def _loop_is_running() -> bool:
    """Whether the calling thread is inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #


def secret_fields(model: type[BaseModel]) -> frozenset[str]:
    """Field names marked ``json_schema_extra={"secret": True}``."""
    names = set()
    for name, info in model.model_fields.items():
        extra = info.json_schema_extra
        if isinstance(extra, dict) and extra.get("secret"):
            names.add(name)
    return frozenset(names)


def params_to_json(params: BaseModel) -> dict[str, Any]:
    """JSON-ready dump of ``params``, unmasked. For the action itself."""
    return params.model_dump(mode="json")


def mask_params(params: BaseModel) -> dict[str, Any]:
    """JSON-ready dump with secret fields replaced by :data:`SECRET_MASK`.

    Everything that leaves the action layer goes through this: the log line, the
    ``audit`` row, :class:`~ayris.core.events.ActionStarted`. Nesting is walked by
    value rather than by annotation, so a ``Params`` that carries a sub-model
    still gets its inner secrets masked.

    Fields named by :meth:`ActionParams.secret_now` are masked as well, which is
    how a value that is only sometimes a secret stays out of the audit.
    """
    dumped = params.model_dump(mode="json")
    hidden = secret_fields(type(params))
    if isinstance(params, ActionParams):
        hidden |= params.secret_now()
    for name in dumped:
        if name in hidden:
            dumped[name] = SECRET_MASK
            continue
        value = getattr(params, name, None)
        if isinstance(value, BaseModel):
            dumped[name] = mask_params(value)
    return dumped


# --------------------------------------------------------------------------- #
# Introspection for the editor
# --------------------------------------------------------------------------- #


class FieldKind(StrEnum):
    """Which widget the editor should draw for a parameter."""

    TEXT = "text"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CHOICE = "choice"
    LIST = "list"
    OBJECT = "object"


@dataclass(frozen=True, slots=True)
class Choice:
    """One option of a :attr:`FieldKind.CHOICE` parameter."""

    value: Any
    label_ru: str


@dataclass(frozen=True, slots=True)
class ParamField:
    """One parameter, described well enough to render an input for it."""

    name: str
    label_ru: str
    kind: FieldKind
    required: bool = True
    default: Any = None
    description_ru: str = ""
    minimum: float | None = None
    maximum: float | None = None
    max_length: int | None = None
    pattern: str = ""
    choices: tuple[Choice, ...] = ()
    item_kind: FieldKind | None = None
    secret: bool = False
    multiline: bool = False
    unit_ru: str = ""

    @property
    def has_range(self) -> bool:
        """Whether a slider or a spin box has bounds to respect."""
        return self.minimum is not None or self.maximum is not None


@dataclass(frozen=True, slots=True)
class ActionSchema:
    """An action as the editor and the docs see it."""

    name: str
    category: ActionCategory
    title_ru: str
    description_ru: str = ""
    require_admin: bool = False
    is_dangerous: bool = False
    supports_undo: bool = False
    timeout_ms: int = DEFAULT_TIMEOUT_MS
    is_async: bool = False
    fields: tuple[ParamField, ...] = ()
    json_schema: Mapping[str, Any] = field(default_factory=dict)

    @property
    def category_title_ru(self) -> str:
        """Russian heading of the group this action belongs to."""
        return self.category.title_ru

    def field_by_name(self, name: str) -> ParamField | None:
        """One parameter description, or ``None`` when there is no such field."""
        return next((item for item in self.fields if item.name == name), None)

    @property
    def secret_fields(self) -> tuple[str, ...]:
        """Names of parameters whose values must never be shown or logged."""
        return tuple(item.name for item in self.fields if item.secret)


#: Longest description that may double as a field label. Beyond it the text is a
#: hint, not a caption, and the editor would clip it.
_LABEL_LIMIT: Final = 48

_JSON_KIND: Final[Mapping[str, FieldKind]] = {
    "string": FieldKind.TEXT,
    "integer": FieldKind.INTEGER,
    "number": FieldKind.NUMBER,
    "boolean": FieldKind.BOOLEAN,
    "array": FieldKind.LIST,
    "object": FieldKind.OBJECT,
}


def build_schema(action: Action | type[Action]) -> ActionSchema:
    """Describe ``action`` — metadata plus one :class:`ParamField` per parameter.

    Built from the pydantic JSON schema rather than from the annotations: pydantic
    has already resolved ``Literal``, enums, ``| None`` and every constraint into
    one flat vocabulary, and duplicating that resolution here would mean two
    places to keep in step.
    """
    cls = action if isinstance(action, type) else type(action)
    model = cls.params_model()
    schema: dict[str, Any] = model.model_json_schema(mode="validation")
    defs: Mapping[str, Any] = schema.get("$defs", {})
    required = set(schema.get("required", ()))
    properties: Mapping[str, Any] = schema.get("properties", {})

    fields = tuple(
        _build_field(
            name=name,
            node=properties.get(name, {}),
            defs=defs,
            required=name in required,
            title=model.model_fields[name].title,
            extra=_field_extra(model, name),
        )
        for name in model.model_fields
    )
    return ActionSchema(
        name=cls.meta.name,
        category=cls.meta.category,
        title_ru=cls.meta.title_ru,
        description_ru=cls.meta.description_ru,
        require_admin=cls.meta.require_admin,
        is_dangerous=cls.meta.is_dangerous,
        supports_undo=cls.meta.supports_undo,
        timeout_ms=cls.meta.timeout_ms,
        is_async=cls.implements_async() and not cls.implements_sync(),
        fields=fields,
        json_schema=schema,
    )


def _field_extra(model: type[BaseModel], name: str) -> Mapping[str, Any]:
    """UI hints attached to a field, or an empty mapping."""
    extra = model.model_fields[name].json_schema_extra
    return extra if isinstance(extra, dict) else {}


def _build_field(
    *,
    name: str,
    node: Mapping[str, Any],
    defs: Mapping[str, Any],
    required: bool,
    title: str | None,
    extra: Mapping[str, Any],
) -> ParamField:
    resolved, optional = _resolve(node, defs)
    kind = _kind_of(resolved)
    choices = tuple(
        Choice(value=value, label_ru=str(_labels(extra).get(str(value), value)))
        for value in _enum_values(resolved)
    )
    if choices:
        kind = FieldKind.CHOICE
    item_kind: FieldKind | None = None
    if kind is FieldKind.LIST:
        items, _ = _resolve(resolved.get("items", {}), defs)
        item_kind = _kind_of(items)
    minimum, maximum = _bounds(resolved)
    label, description = _caption(
        title=title,
        description=str(node.get("description", resolved.get("description", ""))),
        name=name,
    )
    return ParamField(
        name=name,
        label_ru=label,
        kind=kind,
        required=required and not optional,
        default=node.get("default", resolved.get("default")),
        description_ru=description,
        minimum=minimum,
        maximum=maximum,
        max_length=_int_or_none(resolved.get("maxLength")),
        pattern=str(resolved.get("pattern", "")),
        choices=choices,
        item_kind=item_kind,
        secret=bool(extra.get("secret")),
        multiline=bool(extra.get("multiline")),
        unit_ru=str(extra.get("unit_ru", "")),
    )


def _caption(*, title: str | None, description: str, name: str) -> tuple[str, str]:
    """Pick the Russian label for a parameter, and what is left as its hint.

    An author writes ``Field(title="Громкость")`` when the label and the hint say
    different things, and ``Field(description="Громкость в процентах")`` when one
    sentence is the whole story. In the second case the sentence is promoted to
    the label instead of being shown twice — as a caption above the input and as
    the same text underneath it.

    The field name is the last resort. It is English, so it reads badly in the
    editor, which is the point: it is visible enough that the missing Russian
    caption gets noticed.
    """
    if title:
        return title, description
    single_line = description.strip()
    if single_line and "\n" not in single_line and len(single_line) <= _LABEL_LIMIT:
        return single_line, ""
    return name.replace("_", " "), description


def _labels(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    """Russian labels for enum values, keyed by the value as a string."""
    labels = extra.get("choices_ru")
    return labels if isinstance(labels, dict) else {}


def _resolve(node: Mapping[str, Any], defs: Mapping[str, Any]) -> tuple[Mapping[str, Any], bool]:
    """Follow ``$ref`` and unwrap ``X | None``.

    Returns the meaningful half of the node and whether ``None`` was one of the
    alternatives — an optional parameter is one the editor may leave empty even
    when pydantic lists it as required.
    """
    optional = False
    current = node
    for _ in range(_RESOLVE_DEPTH):
        ref = current.get("$ref")
        if isinstance(ref, str):
            current = _deref(ref, defs)
            continue
        branches = current.get("anyOf") or current.get("oneOf")
        if isinstance(branches, list):
            nulls = [item for item in branches if _is_null(item)]
            rest = [item for item in branches if not _is_null(item)]
            optional = optional or bool(nulls)
            if len(rest) == 1:
                merged = dict(rest[0])
                # ``default`` and ``description`` sit on the wrapper, not on the
                # branch, and losing them would drop the field's Russian hint.
                for key in ("default", "description", "title"):
                    if key in current and key not in merged:
                        merged[key] = current[key]
                current = merged
                continue
        break
    return current, optional


#: ``$ref`` chains in a pydantic schema are shallow; the bound only keeps a
#: hand-written recursive model from spinning here.
_RESOLVE_DEPTH: Final = 8


def _deref(ref: str, defs: Mapping[str, Any]) -> Mapping[str, Any]:
    """Look one ``#/$defs/Name`` pointer up. Unknown pointers resolve to ``{}``."""
    name = ref.rsplit("/", 1)[-1]
    node = defs.get(name)
    return node if isinstance(node, dict) else {}


def _is_null(node: Any) -> bool:
    return isinstance(node, dict) and node.get("type") == "null"


def _kind_of(node: Mapping[str, Any]) -> FieldKind:
    declared = node.get("type")
    if isinstance(declared, str):
        return _JSON_KIND.get(declared, FieldKind.TEXT)
    if isinstance(declared, list):
        for item in declared:
            if isinstance(item, str) and item != "null":
                return _JSON_KIND.get(item, FieldKind.TEXT)
    if "enum" in node or "const" in node:
        return FieldKind.CHOICE
    if "properties" in node:
        return FieldKind.OBJECT
    return FieldKind.TEXT


def _enum_values(node: Mapping[str, Any]) -> Iterator[Any]:
    values = node.get("enum")
    if isinstance(values, list):
        yield from values
        return
    if "const" in node:
        yield node["const"]


def _bounds(node: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Inclusive bounds, folding the exclusive forms in.

    The editor draws a spin box, and a spin box has a lowest value it can show.
    ``exclusiveMinimum: 0`` on a float is therefore reported as ``0`` with the
    validator still rejecting it — better a slider that reaches an invalid end
    than a slider with no floor at all.
    """
    minimum = _float_or_none(node.get("minimum"))
    if minimum is None:
        minimum = _float_or_none(node.get("exclusiveMinimum"))
    maximum = _float_or_none(node.get("maximum"))
    if maximum is None:
        maximum = _float_or_none(node.get("exclusiveMaximum"))
    return minimum, maximum


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
