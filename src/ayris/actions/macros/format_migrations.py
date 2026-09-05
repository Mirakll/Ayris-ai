"""Reading a ``.ayris`` written by an older Ayris.

The version lives on the document, not on the command, and it is a single integer:
one file is one format, whatever it holds. Loading walks a chain of one-step
migrations from the version in the file up to :data:`CURRENT_FORMAT_VERSION`, so a
new format costs one function and one entry in :data:`_MIGRATIONS` — never a branch
inside the models, which are the current shape and nothing else.

**Version 1 is not invented.** It is the shape
:mod:`ayris.core.portable_profile` already writes into ``commands.ayris`` inside a
profile bundle: folders as paths, and every trigger as the database row it came
from — ``{"type": …, "payload": {…}, "fuzzy": …, "priority": …}``. Version 2 is the
contract of task 30, where a trigger is a typed model with named fields and a
command also carries its variables and sounds. So the chain here converts files
that exist rather than describing a hypothetical past.

A version from the future is refused, in Russian, naming both numbers: a file
written by a newer Ayris can hold fields this build would silently drop, and
dropping them on import and then saving is how a user loses work.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final

from ayris.core.errors import MacroError
from ayris.core.models import JsonObject

__all__ = [
    "CURRENT_FORMAT_VERSION",
    "MIN_FORMAT_VERSION",
    "MacroFormatError",
    "document_version",
    "migrate_document",
    "trigger_from_row",
]

#: One step of the chain: a whole document in, the next version's document out.
FormatMigration = Callable[[JsonObject], JsonObject]

#: Format version this build writes.
CURRENT_FORMAT_VERSION: Final = 2

#: Oldest format version that can still be read.
MIN_FORMAT_VERSION: Final = 1


class MacroFormatError(MacroError):
    """A ``.ayris`` file cannot be read: no version, a broken one, or one too new.

    Separate from the validation errors of :mod:`ayris.actions.macros.validator`
    because there is nothing to point at in the editor — the file did not become a
    command at all.
    """

    default_user_message = "Не могу прочитать файл команд Ayris."


def document_version(document: JsonObject) -> int:
    """The ``schema_version`` of a loaded document, checked for sanity.

    Raises:
        MacroFormatError: the field is missing, not a whole number, older than
            :data:`MIN_FORMAT_VERSION` or newer than
            :data:`CURRENT_FORMAT_VERSION`.
    """
    if "schema_version" not in document:
        raise MacroFormatError(
            "document has no schema_version",
            user_message="В файле нет поля schema_version — это не файл команд Ayris.",
        )
    raw: object = document["schema_version"]
    # ``bool`` is an ``int`` in Python, and ``True`` is not a version.
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise MacroFormatError(
            f"schema_version is not an integer: {raw!r}",
            user_message=f"Версия формата «{raw}» не похожа на число.",
        )
    if raw < MIN_FORMAT_VERSION:
        raise MacroFormatError(
            f"schema_version {raw} is below the oldest supported {MIN_FORMAT_VERSION}",
            user_message=(
                f"Версия формата {raw} слишком старая: я понимаю формат "
                f"с {MIN_FORMAT_VERSION}-го."
            ),
        )
    if raw > CURRENT_FORMAT_VERSION:
        raise MacroFormatError(
            f"schema_version {raw} is newer than supported {CURRENT_FORMAT_VERSION}",
            user_message=(
                f"Файл сделан более новой версией Ayris: формат {raw}, "
                f"а я понимаю до {CURRENT_FORMAT_VERSION}-го. Обновите Ayris."
            ),
        )
    return raw


def migrate_document(document: JsonObject) -> JsonObject:
    """Bring a loaded document up to :data:`CURRENT_FORMAT_VERSION`.

    Returns a new dictionary; the argument is left alone, because the caller may
    still want to show the user what was in the file.

    Raises:
        MacroFormatError: the version is unusable, or a step cannot convert
            something it must not drop.
    """
    version = document_version(document)
    migrated = dict(document)
    while version < CURRENT_FORMAT_VERSION:
        step = _MIGRATIONS.get(version)
        if step is None:  # pragma: no cover - a gap in the chain is a programming error
            raise MacroFormatError(
                f"no migration from format {version}",
                user_message=f"Не умею обновлять формат {version}.",
            )
        migrated = step(migrated)
        version += 1
        migrated["schema_version"] = version
    return migrated


def _v1_to_v2(document: JsonObject) -> JsonObject:
    """Profile-bundle shape to the task 30 contract: triggers stop being rows.

    Everything else survives untouched. ``variables`` and ``sounds`` are simply
    absent in version 1, and absent is what the models default to, so nothing is
    invented here — including timestamps, which the model fills with the moment of
    import rather than a made-up past.
    """
    commands = document.get("commands")
    if not isinstance(commands, list):
        return {**document, "commands": []}
    return {**document, "commands": [_v1_command(command) for command in commands]}


def _v1_command(command: Any) -> Any:
    if not isinstance(command, dict):
        return command
    converted: JsonObject = dict(command)
    if converted.get("folder") is None:
        converted["folder"] = []
    triggers = converted.get("triggers")
    if isinstance(triggers, list):
        converted["triggers"] = [_v1_trigger(trigger) for trigger in triggers]
    return converted


def _v1_trigger(trigger: Any) -> Any:
    """One row of a version 1 file to one typed trigger."""
    if not isinstance(trigger, dict) or "payload" not in trigger:
        return trigger
    payload = trigger.get("payload")
    return trigger_from_row(
        str(trigger.get("type", "voice")),
        payload if isinstance(payload, dict) else {},
        fuzzy=bool(trigger.get("fuzzy", True)),
        priority=int(trigger.get("priority", 0) or 0),
    )


def trigger_from_row(
    kind: str,
    payload: JsonObject,
    *,
    fuzzy: bool = True,
    priority: int = 0,
) -> JsonObject:
    """A ``triggers`` row's ``type`` and ``payload`` as one typed trigger's fields.

    Shared with :mod:`ayris.actions.macros.serializer`, which reads rows written by
    the repositories of task 3 rather than a file. The payload keys are the same ones
    :func:`ayris.nlu.matcher.trigger_from_db` reads, so a row and a version 1 file
    are one conversion instead of two that drift apart.

    Raises:
        MacroFormatError: ``kind`` is not a trigger type this build knows.
    """
    match kind:
        case "voice":
            return _v1_voice(payload, fuzzy=fuzzy, priority=priority)
        case "hotkey":
            combo = _first_string(payload, ("combo", "hotkey", "keys", "key"))
            return {"type": "hotkey", "combo": combo, "enabled": _enabled(payload)}
        case "event":
            event = _first_string(payload, ("event_name", "event", "name"))
            filters = payload.get("filter_json") or payload.get("filter") or {}
            return {
                "type": "event",
                "event_name": event,
                "filter_json": filters if isinstance(filters, dict) else {},
                "enabled": _enabled(payload),
            }
        case "timer":
            return _v1_timer(payload)
    raise MacroFormatError(
        f"unknown trigger type {kind!r}",
        user_message=f"Триггер неизвестного вида «{kind}».",
    )


def _v1_voice(payload: JsonObject, *, fuzzy: bool, priority: int) -> JsonObject:
    """The matcher's three pattern keys back into one phrase plus a flag.

    ``regex`` is both a key and a flag in version 1: the pattern sitting under that
    key *is* the regular expression, which is how :func:`ayris.nlu.matcher.
    trigger_from_db` reads it.
    """
    is_regex = isinstance(payload.get("regex"), str) and bool(payload["regex"])
    phrase = _first_string(payload, ("template", "regex", "phrase"))
    converted: JsonObject = {
        "type": "voice",
        "phrase": phrase,
        "fuzzy": fuzzy and not is_regex,
        "regex": is_regex,
        "priority": priority,
        "enabled": _enabled(payload),
    }
    threshold = payload.get("threshold")
    if isinstance(threshold, int | float) and not isinstance(threshold, bool):
        converted["fuzzy_threshold"] = float(threshold)
    conditions = {key: value for key, value in payload.items() if key.startswith("when_")}
    if conditions:
        converted["conditions"] = conditions
    return converted


def _v1_timer(payload: JsonObject) -> JsonObject:
    fire_at = _first_string(payload, ("fire_at", "at", "when"))
    cron = _first_string(payload, ("cron", "schedule"))
    converted: JsonObject = {"type": "timer", "enabled": _enabled(payload)}
    if cron:
        converted["cron"] = cron
    else:
        converted["fire_at"] = fire_at
    return converted


def _first_string(payload: JsonObject, keys: tuple[str, ...]) -> str:
    """The first of ``keys`` holding a non-empty string, or an empty string.

    An empty result is left to the models to refuse: the migration's job is to move
    a value into its new place, and a bundle with an empty combination was already
    broken when it was written.
    """
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _enabled(payload: JsonObject) -> bool:
    return bool(payload.get("enabled", True))


#: One entry per readable version, keyed by the version it upgrades *from*.
_MIGRATIONS: Final[dict[int, FormatMigration]] = {1: _v1_to_v2}
