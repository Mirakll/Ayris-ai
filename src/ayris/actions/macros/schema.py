"""The command model and the ``.ayris`` contract: triggers, blocks, variables, sounds.

Everything downstream of task 30 speaks these types. The interpreter walks
:class:`ActionBlock`, the editor edits :class:`CommandModel`, the ``.vap`` and
``.ahk`` importers build them, the database stores them column by column, and a
``.ayris`` file is one of them written out. That makes this module a *contract*
rather than an implementation detail: a renamed field is a format migration
(:mod:`ayris.actions.macros.format_migrations`), not an edit.

Three decisions are worth knowing before reading on.

**Blocks are a tree, branches are fields.** ``then``, ``else``, ``body`` and
``catch`` are lists of blocks on the block itself, so a nested ``If`` inside a
``While`` inside a ``Try`` is one value with no side table. Section 7.2 of the
specification lists ``Else`` and ``Catch`` as blocks; here they are branches of
``If`` and ``Try``, which is the same language with one less way to write it
wrong — an ``Else`` cannot end up orphaned. ``Switch`` keeps its arms as ``Case``
and ``Default`` blocks inside ``body`` for the same reason: fixed branch names, no
open-ended keys.

**Sounds are bound, not played.** Section 7.1 asks for a sound per stage — on
start, on success, on error — so a stage-bound sound is a :class:`SoundBinding` on
the block (or on the command as a whole), not a ``PlaySound`` block with a
``stage`` parameter that the interpreter would have to reorder. The two examples of
section 22 are written that way in ``tests/fixtures/macros``.

**Models validate, they do not resolve.** ``{volume}`` in a parameter is left
exactly as written: substitution happens at run time from variables and trigger
slots. Which is why parameter *types* cannot be checked here and are checked
against the action's own ``Params`` by :mod:`ayris.actions.macros.validator`,
skipping the values that are still placeholders.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from ayris.core.models import TriggerType, VariableScope, VariableType, utc_now
from ayris.nlu.slots import template_slot_names, validate_template
from ayris.utils.hotkeys import Hotkey, HotkeyNotationError, canonical_hotkey, parse_hotkey

__all__ = [
    "DECLARED_BLOCKS",
    "LOGIC_BLOCKS",
    "MAX_BLOCKS",
    "MAX_BLOCK_DEPTH",
    "SOUND_EXTENSIONS",
    "ActionBlock",
    "BlockLocation",
    "CommandModel",
    "EventTrigger",
    "HotkeyTrigger",
    "LogicBlockSpec",
    "MacroModel",
    "OnError",
    "SoundBinding",
    "SoundSource",
    "SoundStage",
    "TimerTrigger",
    "TriggerModel",
    "VariableModel",
    "VoiceTrigger",
    "walk_blocks",
]

#: How deep branches may nest. Sixteen is far past anything a person writes in the
#: editor and short enough that a hand-edited (or hostile) ``.ayris`` cannot make
#: the interpreter or the recursive validator run out of stack.
MAX_BLOCK_DEPTH: Final = 16

#: How many blocks one command may hold in total, branches included. A guard on the
#: same class of file as the depth limit: a flat list costs no recursion but still
#: has to be walked, validated and drawn.
MAX_BLOCKS: Final = 2000

_IDENTIFIER: Final = re.compile(r"^[^\W\d]\w*$", re.UNICODE)

#: A five- or six-field cron line. Shape only: the timer subsystem owns the meaning
#: of the fields, this just refuses prose in a field that has to be a schedule.
_CRON_FIELD: Final = r"[*\d,/\-A-Za-z?#]+"
_CRON: Final = re.compile(rf"^{_CRON_FIELD}(?:\s+{_CRON_FIELD}){{4,5}}$")


class SoundStage(StrEnum):
    """When a bound sound plays, per section 7.1."""

    ON_START = "on_start"
    ON_SUCCESS = "on_success"
    ON_ERROR = "on_error"


class SoundSource(StrEnum):
    """Where the sound comes from.

    ``builtin`` is a name in the shipped set, ``file`` a name inside the profile's
    own sounds folder, ``tts`` a phrase to speak. Written in a file as one string —
    ``builtin:volume_changed``, ``custom:work_mode_start.wav`` — which is the
    notation section 22 uses and :class:`SoundBinding` accepts.
    """

    BUILTIN = "builtin"
    FILE = "file"
    TTS = "tts"


class OnError(StrEnum):
    """What the interpreter does when a block raises."""

    CONTINUE = "continue"
    STOP = "stop"


class MacroModel(BaseModel):
    """Shared configuration for every model in the contract.

    ``extra="forbid"`` is the point: a typo in a hand-written ``.ayris`` (or a field
    dropped by a format migration that forgot to rename it) has to be an error at
    load time, not a value silently missing at run time. Not frozen — the editor of
    task 33 mutates the tree in place — but ``validate_assignment`` keeps that
    mutation as checked as construction.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        use_enum_values=False,
    )


#: The prefixes a sound reference may carry. ``custom:`` is section 22's spelling of
#: "a file the user imported", which is :attr:`SoundSource.FILE`.
_SOUND_PREFIXES: Final[dict[str, SoundSource]] = {
    "builtin": SoundSource.BUILTIN,
    "custom": SoundSource.FILE,
    "file": SoundSource.FILE,
    "tts": SoundSource.TTS,
    "say": SoundSource.TTS,
}

#: Container formats section 7.1 allows for an imported sound.
SOUND_EXTENSIONS: Final[frozenset[str]] = frozenset({".wav", ".mp3", ".ogg"})

_BUILTIN_NAME: Final = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_DRIVE: Final = re.compile(r"^[A-Za-z]:")


class SoundBinding(MacroModel):
    """A sound bound to a stage of a command or of one block.

    Accepts the one-string form used in files and in the section 22 examples —
    ``builtin:volume_changed``, ``custom:work_mode_start.wav``, ``tts:Готово`` — as
    well as the spelled-out mapping. Both end up as the same three fields.

    A ``file`` reference is a *name inside the profile's sounds folder*, and this is
    where that is enforced: separators, ``..``, a drive letter and a leading ``~``
    are all refused, so an exported ``.ayris`` cannot carry ``C:\\Users\\…`` from the
    machine it was written on to the machine that imports it.
    """

    stage: SoundStage = SoundStage.ON_SUCCESS
    source: SoundSource = SoundSource.BUILTIN
    value: str = Field(min_length=1, max_length=500)
    volume: int | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="before")
    @classmethod
    def _split_reference(cls, data: Any) -> Any:
        """``builtin:volume_changed`` to ``source`` plus ``value``."""
        if isinstance(data, str):
            data = {"value": data}
        if not isinstance(data, dict):
            return data
        value = data.get("value")
        if not isinstance(value, str) or data.get("source") is not None:
            return data
        prefix, separator, rest = value.partition(":")
        source = _SOUND_PREFIXES.get(prefix.strip().lower()) if separator else None
        if source is None:
            return data
        return {**data, "source": source, "value": rest.strip()}

    @model_validator(mode="after")
    def _check_value(self) -> Self:
        if self.source is SoundSource.BUILTIN and not _BUILTIN_NAME.match(self.value):
            raise ValueError(f"«{self.value}» не похоже на имя встроенного звука")
        if self.source is SoundSource.FILE:
            _check_portable_name(self.value)
            if not any(self.value.lower().endswith(suffix) for suffix in SOUND_EXTENSIONS):
                allowed = ", ".join(sorted(SOUND_EXTENSIONS))
                raise ValueError(f"звук «{self.value}» должен быть файлом {allowed}")
        return self

    @property
    def reference(self) -> str:
        """The one-string spelling, the way a file carries it."""
        prefix = "custom" if self.source is SoundSource.FILE else self.source.value
        return f"{prefix}:{self.value}"


def _check_portable_name(name: str) -> None:
    """Refuse anything that only means something on the machine it was written on.

    Task 30 forbids absolute paths in ``.ayris`` outright; a relative path with
    ``..`` is the same problem one step removed, so both are rejected here rather
    than at export time, where a hand-written file would slip past.
    """
    if "\\" in name or "/" in name:
        raise ValueError(f"«{name}» — путь; в .ayris хранится только имя файла")
    if _DRIVE.match(name) or name.startswith("~"):
        raise ValueError(f"«{name}» — абсолютный путь; в .ayris хранится только имя файла")
    if name.startswith(".."):
        raise ValueError(f"«{name}» выходит за папку звуков профиля")


#: ``str`` is what task 30 and the editor say, ``string`` is what the database CHECK
#: stores; both are read, the canonical one is written.
_TYPE_ALIASES: Final[dict[str, VariableType]] = {
    "str": VariableType.STRING,
    "text": VariableType.STRING,
    "integer": VariableType.INT,
    "number": VariableType.FLOAT,
    "double": VariableType.FLOAT,
    "boolean": VariableType.BOOL,
    "list": VariableType.ARRAY,
    "object": VariableType.DICT,
    "map": VariableType.DICT,
}


class VariableModel(MacroModel):
    """One variable a command declares: name, type, lifetime, starting value.

    The type is checked against ``default`` here, which is the one place it *can* be
    checked — at run time a variable holds whatever the last assignment put in it,
    and the interpreter of task 31 coerces on write against this declaration.
    """

    name: str = Field(min_length=1, max_length=64)
    type: VariableType = VariableType.STRING
    scope: VariableScope = VariableScope.LOCAL
    default: Any = None
    persistent: bool = False

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _IDENTIFIER.match(value):
            raise ValueError(
                f"имя переменной «{value}» должно начинаться с буквы "
                "и состоять из букв, цифр и подчёркиваний"
            )
        return value

    @field_validator("type", mode="before")
    @classmethod
    def _read_type_alias(cls, value: Any) -> Any:
        if isinstance(value, str):
            return _TYPE_ALIASES.get(value.strip().lower(), value)
        return value

    @model_validator(mode="after")
    def _check_default(self) -> Self:
        """``None`` means "no starting value"; anything else has to match the type."""
        if self.default is None:
            return self
        if not _matches_type(self.default, self.type):
            raise ValueError(
                f"значение по умолчанию переменной «{self.name}» не подходит "
                f"объявленному типу {self.type.value}"
            )
        if self.type is VariableType.FLOAT and isinstance(self.default, int):
            # Written straight into ``__dict__``: assignment would re-run this
            # validator, and ``validate_assignment`` plus a coercing validator is a
            # loop. The value is already validated, only its Python type changes.
            self.__dict__["default"] = float(self.default)
        return self

    @model_validator(mode="after")
    def _check_persistence(self) -> Self:
        """A local variable dies with the invocation, so it cannot be persistent."""
        if self.persistent and self.scope is VariableScope.LOCAL:
            raise ValueError(
                f"переменная «{self.name}» с областью local не может быть постоянной: "
                "она живёт один вызов команды"
            )
        return self


def _matches_type(value: Any, declared: VariableType) -> bool:
    """Whether a literal fits a declared type, keeping ``bool`` out of the numbers.

    ``isinstance(True, int)`` is true in Python, so a ``bool`` would quietly pass as
    an ``int`` default and then be written to the database as ``1``.
    """
    if isinstance(value, bool):
        return declared is VariableType.BOOL
    match declared:
        case VariableType.STRING:
            return isinstance(value, str)
        case VariableType.INT:
            return isinstance(value, int)
        case VariableType.FLOAT:
            return isinstance(value, int | float)
        case VariableType.ARRAY:
            return isinstance(value, list)
        case VariableType.DICT:
            return isinstance(value, dict)
    return False


class VoiceTrigger(MacroModel):
    """A phrase, fuzzy or exact, plain or with slots, or a regular expression.

    The pattern is written once, in ``phrase``, and *how* it is read follows from
    the text: ``regex`` says it is a regular expression, braces say it is a template
    with slots, neither says it is a plain phrase. :attr:`payload_key` is that
    decision written down, because the matcher of task 16 reads the three cases from
    three different payload keys and this is the only place that chooses between
    them.

    Both are checked at save time, which is the promise of task 30: a regex that
    does not compile and a slot type that does not exist are errors in the editor,
    not silence at the moment the user speaks.
    """

    type: Literal[TriggerType.VOICE] = TriggerType.VOICE
    phrase: str = Field(min_length=1, max_length=500)
    fuzzy: bool = True
    fuzzy_threshold: float | None = Field(default=None, gt=0.0, le=1.0)
    regex: bool = False
    priority: int = Field(default=0, ge=-1000, le=1000)
    enabled: bool = True
    #: The matcher's ``when_*`` payload keys, carried through untouched. Nothing here
    #: writes them yet; they exist so a command loaded from the database and saved
    #: back keeps conditions this task did not invent.
    conditions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_pattern(self) -> Self:
        if self.regex:
            try:
                re.compile(self.phrase)
            except re.error as exc:
                raise ValueError(f"регулярное выражение не компилируется: {exc}") from exc
            if self.fuzzy:
                # A regular expression either matches or it does not; there is no
                # near miss to score, so the flag is silently dropped rather than
                # kept as a setting that does nothing.
                self.__dict__["fuzzy"] = False
            return self
        if self.slot_names:
            message = validate_template(self.phrase)
            if message:
                raise ValueError(message)
        return self

    @field_validator("conditions")
    @classmethod
    def _check_conditions(cls, value: dict[str, Any]) -> dict[str, Any]:
        unknown = sorted(key for key in value if not key.startswith("when_"))
        if unknown:
            raise ValueError(f"условия триггера начинаются с when_, а не {', '.join(unknown)}")
        return value

    @property
    def slot_names(self) -> tuple[str, ...]:
        """Slots the phrase declares, empty for a regex or a plain phrase."""
        if self.regex:
            return ()
        return template_slot_names(self.phrase)

    @property
    def payload_key(self) -> str:
        """Which key of the database payload carries the pattern."""
        if self.regex:
            return "regex"
        return "template" if self.slot_names else "phrase"


class HotkeyTrigger(MacroModel):
    """A key combination, stored in the one canonical spelling.

    Whatever notation was typed or imported, ``combo`` comes out as
    ``ctrl+alt+v``: :mod:`ayris.utils.hotkeys` folds the notations, and normalising
    at save time is what lets task 37 detect two commands claiming one combination
    by comparing strings.
    """

    type: Literal[TriggerType.HOTKEY] = TriggerType.HOTKEY
    combo: str = Field(min_length=1, max_length=100)
    enabled: bool = True

    @field_validator("combo")
    @classmethod
    def _canonicalize(cls, value: str) -> str:
        try:
            return canonical_hotkey(value)
        except HotkeyNotationError as exc:
            # The Russian text of the parser is better than anything phrasable
            # here — it names the token it choked on.
            raise ValueError(exc.user_message) from exc

    @property
    def hotkey(self) -> Hotkey:
        """The combination as a value object, for registration and for the UI."""
        return parse_hotkey(self.combo)


class EventTrigger(MacroModel):
    """An event from the bus or from a plugin, optionally filtered."""

    type: Literal[TriggerType.EVENT] = TriggerType.EVENT
    event_name: str = Field(min_length=1, max_length=120)
    #: Field-to-value pairs the event has to carry. Kept as data, not code: a
    #: condition language here would be a second one next to ``If``.
    filter_json: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("event_name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not re.match(r"^[^\W\d][\w.]*$", value, re.UNICODE):
            raise ValueError(f"«{value}» не похоже на имя события")
        return value


class TimerTrigger(MacroModel):
    """A moment or a schedule: exactly one of ``fire_at`` and ``cron``.

    ``fire_at`` without a timezone means local time on the machine that runs the
    command — which is what a person setting an alarm means, and why an exported
    file does not carry an offset unless the author put one there.
    """

    type: Literal[TriggerType.TIMER] = TriggerType.TIMER
    fire_at: datetime | None = None
    cron: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def _check_schedule(self) -> Self:
        if (self.fire_at is None) == (self.cron is None):
            raise ValueError("у таймера должно быть либо время fire_at, либо расписание cron")
        if self.cron is not None and not _CRON.match(self.cron):
            raise ValueError(f"«{self.cron}» не похоже на расписание cron из пяти или шести полей")
        return self


#: A trigger, read by its ``type``. Pydantic picks the model from the tag instead of
#: trying each in turn, so an unknown type is one clear error rather than four.
TriggerModel = Annotated[
    VoiceTrigger | HotkeyTrigger | EventTrigger | TimerTrigger,
    Field(discriminator="type"),
]


@dataclass(frozen=True, slots=True)
class LogicBlockSpec:
    """What a logic block needs: its parameters and which branches it may hold.

    A table rather than a class per block. The blocks of section 7.2 differ only in
    those four facts, and both the validator and the editor's palette read them from
    here, so ``If`` gaining an optional parameter is one line in one place.
    """

    name: str
    required_params: tuple[str, ...] = ()
    optional_params: tuple[str, ...] = ()
    branches: tuple[str, ...] = ()
    required_branches: tuple[str, ...] = ()
    #: ``True`` for the block that runs another command — the one the validator has
    #: to follow to find a cycle.
    is_call: bool = False

    @property
    def params(self) -> frozenset[str]:
        """Every parameter name the block accepts."""
        return frozenset(self.required_params) | frozenset(self.optional_params)


def _logic_blocks() -> dict[str, LogicBlockSpec]:
    """The logic and flow blocks of section 7.2, spelled out once."""
    specs = (
        LogicBlockSpec(
            "If",
            required_params=("condition",),
            branches=("then", "else"),
            required_branches=("then",),
        ),
        LogicBlockSpec(
            "Switch",
            required_params=("value",),
            branches=("body",),
            required_branches=("body",),
        ),
        LogicBlockSpec(
            "Case", required_params=("value",), branches=("body",), required_branches=("body",)
        ),
        LogicBlockSpec("Default", branches=("body",), required_branches=("body",)),
        LogicBlockSpec(
            "While",
            required_params=("condition",),
            optional_params=("max_iterations",),
            branches=("body",),
            required_branches=("body",),
        ),
        LogicBlockSpec(
            "For",
            required_params=("var",),
            optional_params=("items", "from", "to", "step"),
            branches=("body",),
            required_branches=("body",),
        ),
        LogicBlockSpec(
            "Try",
            optional_params=("error_var",),
            branches=("body", "catch"),
            required_branches=("body",),
        ),
        LogicBlockSpec("SetVar", required_params=("name", "value"), optional_params=("scope",)),
        LogicBlockSpec("GetVar", required_params=("name",), optional_params=("into",)),
        LogicBlockSpec("ArrayPush", required_params=("name", "value")),
        LogicBlockSpec("ArrayGet", required_params=("name", "index"), optional_params=("into",)),
        LogicBlockSpec("DictSet", required_params=("name", "key", "value")),
        LogicBlockSpec("DictGet", required_params=("name", "key"), optional_params=("into",)),
        LogicBlockSpec("Wait", required_params=("ms",)),
        LogicBlockSpec("Sleep", required_params=("ms",)),
        LogicBlockSpec(
            "CallCommand",
            required_params=("command",),
            optional_params=("args", "wait"),
            is_call=True,
        ),
        LogicBlockSpec("Return", optional_params=("value",)),
        LogicBlockSpec("Break"),
        LogicBlockSpec("Continue"),
    )
    return {spec.name: spec for spec in specs}


#: Blocks the macro language itself provides — everything the action registry does
#: not, because control flow has no side effect to register.
LOGIC_BLOCKS: Final[dict[str, LogicBlockSpec]] = _logic_blocks()

#: Action blocks section 7.2 declares that this build does not register yet. The
#: validator warns about them instead of calling them unknown: a macro imported from
#: a VoiceAttack profile may well use one, and the file should survive until the
#: task that implements it lands. Checked against the registry first, so a name
#: leaves this list by being implemented, not by being deleted from it.
DECLARED_BLOCKS: Final[frozenset[str]] = frozenset(
    {
        "Say",
        "PlaySound",
        "StopSound",
        "SetTTSVoice",
        "SetBrightness",
        "RunShell",
        "WebRequest",
        "ToastNotify",
        "OverlayLog",
    }
)

_BLOCK_NAME: Final = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: Branch fields in the order the editor draws them, paired with the name a file
#: writes: ``else`` is a Python keyword, so the field cannot be called that.
_BRANCHES: Final[tuple[tuple[str, str], ...]] = (
    ("then", "then"),
    ("else_", "else"),
    ("body", "body"),
    ("catch", "catch"),
)


class ActionBlock(MacroModel):
    """One step of a command: an action to run or a logic block, with its branches.

    ``type`` is either a name from the action registry of task 19 or one of
    :data:`LOGIC_BLOCKS`. Which of the two it is decides everything else — whether
    ``params`` are checked against an action's ``Params``, and which branches may
    hold anything — and that decision is the validator's, not the model's, because
    the registry is built at run time and a model has to load a file without one.

    What *is* enforced here is shape: a branch a block cannot have must be empty, and
    the arms of a ``Switch`` are ``Case`` and ``Default`` blocks. Emptiness of a
    branch that must not be empty is a validator problem instead, so the editor can
    save a half-written ``If`` without the file becoming unloadable.
    """

    type: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)
    then: list[ActionBlock] = Field(default_factory=list)
    else_: list[ActionBlock] = Field(default_factory=list, alias="else")
    body: list[ActionBlock] = Field(default_factory=list)
    catch: list[ActionBlock] = Field(default_factory=list)
    sound: SoundBinding | None = None
    enabled: bool = True
    comment: str = Field(default="", max_length=500)
    on_error: OnError = OnError.STOP

    #: Branch field names, for code that walks the tree generically.
    BRANCH_FIELDS: ClassVar[tuple[str, ...]] = tuple(field for field, _ in _BRANCHES)

    @field_validator("type")
    @classmethod
    def _check_type(cls, value: str) -> str:
        if not _BLOCK_NAME.match(value):
            raise ValueError(f"«{value}» не похоже на имя блока")
        return value

    @model_validator(mode="after")
    def _check_branches(self) -> Self:
        spec = LOGIC_BLOCKS.get(self.type)
        allowed = spec.branches if spec is not None else ()
        for name, blocks in self.branches():
            if blocks and name not in allowed:
                raise ValueError(f"у блока «{self.type}» не бывает ветки {name}")
        if self.type == "Switch":
            wrong = sorted({child.type for child in self.body} - {"Case", "Default"})
            if wrong:
                raise ValueError(
                    f"внутри Switch бывают только Case и Default, а не {', '.join(wrong)}"
                )
        return self

    @model_serializer(mode="wrap")
    def _drop_empty(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave out the branches and the parameters this block does not use.

        Every block carries all four branch fields and almost every block leaves
        three of them empty, so writing them out would double an exported file and
        the ``actions_json`` column with nothing gained: an absent branch and an
        empty one read back the same. Dropped afterwards rather than with
        ``exclude_defaults``, which would also drop the ``type`` of a trigger — and
        that one is the discriminator the whole union hangs on.
        """
        data: dict[str, Any] = handler(self)
        for field_name, wire_name in (*_BRANCHES, ("params", "params")):
            for key in (field_name, wire_name):
                if not data.get(key, True):
                    del data[key]
        return data

    def branches(self) -> Iterator[tuple[str, list[ActionBlock]]]:
        """Every branch as ``(name a file writes, blocks)``, empty ones included."""
        for field_name, wire_name in _BRANCHES:
            yield wire_name, getattr(self, field_name)

    @property
    def spec(self) -> LogicBlockSpec | None:
        """The logic block description, or ``None`` when this is an action."""
        return LOGIC_BLOCKS.get(self.type)

    @property
    def is_logic(self) -> bool:
        """Whether the macro language runs this block itself."""
        return self.type in LOGIC_BLOCKS


ActionBlock.model_rebuild()


@dataclass(frozen=True, slots=True)
class BlockLocation:
    """A block together with where it sits, so a problem can point at it.

    The path is the vocabulary the whole task speaks: the validator puts it in every
    problem it reports, and the editor of task 33 uses it to select the offending
    block. Written as ``actions[1].then[0]``.
    """

    block: ActionBlock
    path: tuple[str | int, ...]
    depth: int

    @property
    def path_text(self) -> str:
        """``actions[1].then[0]`` — the path as a person reads it."""
        parts: list[str] = []
        for step in self.path:
            if isinstance(step, int):
                parts.append(f"[{step}]")
            elif parts:
                parts.append(f".{step}")
            else:
                parts.append(step)
        return "".join(parts)


def walk_blocks(
    blocks: list[ActionBlock] | tuple[ActionBlock, ...],
    *,
    root: str = "actions",
    depth: int = 0,
    path: tuple[str | int, ...] = (),
) -> Iterator[BlockLocation]:
    """Walk a block tree depth-first, parents before children.

    One walker for the validator, the serializer, the interpreter and the editor, so
    "every block, including the ones inside branches" cannot be spelled four ways
    with three of them forgetting ``catch``.
    """
    base = path if path else (root,)
    for index, block in enumerate(blocks):
        here = (*base, index)
        yield BlockLocation(block=block, path=here, depth=depth)
        for name, children in block.branches():
            if children:
                yield from walk_blocks(children, depth=depth + 1, path=(*here, name))


class CommandModel(MacroModel):
    """One command: what fires it, what it does, what it remembers, how it sounds.

    The unit of everything downstream — a row in the command list, a file on disk,
    three tables in the database, one entry in the matcher's index. ``id`` and
    ``folder_id`` are database keys and are the only fields an exported file leaves
    out: they mean nothing on another machine, where the same command gets other
    numbers. The portable equivalent of ``folder_id`` is :attr:`folder`, the path
    written as names, exactly as :mod:`ayris.core.portable_profile` writes it.

    Both forms of section 22 are accepted for ``variables`` and ``sounds`` — the
    mapping a person writes by hand and the list a dump produces — because a file
    the user edited in a text editor should load, and the specification's own
    examples are written the first way.
    """

    id: int | None = None
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list)
    folder_id: int | None = None
    #: Folder path as names, root being the empty list. The portable half of
    #: ``folder_id``: an export carries this, an import looks the folders up by name.
    folder: list[str] = Field(default_factory=list)
    enabled: bool = True
    priority: int = Field(default=0, ge=-1000, le=1000)
    cooldown_ms: int = Field(default=0, ge=0, le=3_600_000)
    require_admin: bool = False
    triggers: list[TriggerModel] = Field(default_factory=list)
    actions: list[ActionBlock] = Field(default_factory=list)
    variables: list[VariableModel] = Field(default_factory=list)
    #: Sounds for the command as a whole, at most one per stage. A block's own sound
    #: is :attr:`ActionBlock.sound`.
    sounds: list[SoundBinding] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, value: list[str]) -> list[str]:
        seen: dict[str, None] = {}
        for tag in value:
            cleaned = tag.strip()
            if not cleaned:
                continue
            if len(cleaned) > 40:
                raise ValueError(f"метка «{cleaned}» длиннее 40 символов")
            seen.setdefault(cleaned, None)
        return list(seen)

    @field_validator("folder")
    @classmethod
    def _check_folder(cls, value: list[str]) -> list[str]:
        for part in value:
            if not part.strip():
                raise ValueError("в пути папки есть пустое имя")
            _check_portable_name(part)
        return [part.strip() for part in value]

    @model_validator(mode="before")
    @classmethod
    def _read_mappings(cls, data: Any) -> Any:
        """Accept the section 22 shapes for ``variables`` and ``sounds``."""
        if not isinstance(data, dict):
            return data
        changed = dict(data)
        variables = changed.get("variables")
        if isinstance(variables, dict):
            changed["variables"] = [
                {"name": name, **body} if isinstance(body, dict) else {"name": name, "type": body}
                for name, body in variables.items()
            ]
        sounds = changed.get("sounds")
        if isinstance(sounds, dict):
            changed["sounds"] = [
                (
                    {"stage": stage, "value": body}
                    if isinstance(body, str)
                    else {"stage": stage, **body}
                )
                for stage, body in sounds.items()
            ]
        return changed

    @model_validator(mode="after")
    def _check_unique(self) -> Self:
        names = [variable.name for variable in self.variables]
        duplicate = _first_duplicate(names)
        if duplicate is not None:
            raise ValueError(f"переменная «{duplicate}» объявлена дважды")
        # One sound *and* one phrase per stage: section 22's second example both
        # speaks and plays a file on success, so the stage alone cannot be the key,
        # while two builtin sounds racing on one stage is nothing anybody meant.
        bindings = [f"{binding.stage.value}/{binding.source.value}" for binding in self.sounds]
        duplicate = _first_duplicate(bindings)
        if duplicate is not None:
            stage, _, source = duplicate.partition("/")
            raise ValueError(f"для стадии {stage} задано два звука вида {source}")
        return self

    @model_validator(mode="after")
    def _check_size(self) -> Self:
        """Depth and count, checked once for the whole tree rather than per block."""
        total = 0
        for location in self.blocks():
            total += 1
            if location.depth >= MAX_BLOCK_DEPTH:
                raise ValueError(
                    f"блок {location.path_text} вложен глубже {MAX_BLOCK_DEPTH} уровней"
                )
        if total > MAX_BLOCKS:
            raise ValueError(f"в команде {total} блоков, больше допустимых {MAX_BLOCKS}")
        return self

    def blocks(self) -> Iterator[BlockLocation]:
        """Every block of the command, parents before children, with its path."""
        return walk_blocks(self.actions)

    @property
    def voice_triggers(self) -> tuple[VoiceTrigger, ...]:
        """Voice triggers only — what the matcher index of task 16 is built from."""
        return tuple(trigger for trigger in self.triggers if isinstance(trigger, VoiceTrigger))

    @property
    def hotkey_triggers(self) -> tuple[HotkeyTrigger, ...]:
        """Hotkey triggers only — what task 37 registers."""
        return tuple(trigger for trigger in self.triggers if isinstance(trigger, HotkeyTrigger))

    @property
    def variable_names(self) -> frozenset[str]:
        """Names a ``{placeholder}`` may refer to, declarations only."""
        return frozenset(variable.name for variable in self.variables)

    @property
    def slot_names(self) -> frozenset[str]:
        """Names a ``{placeholder}`` may refer to, filled by a voice trigger."""
        return frozenset(name for trigger in self.voice_triggers for name in trigger.slot_names)


def _first_duplicate(values: list[str]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
