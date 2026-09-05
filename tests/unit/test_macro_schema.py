"""Задача 30: команда как модель, как файл и как строки в таблицах.

Everything downstream of this task speaks these types, so the tests are written
against the *contract* rather than against the implementation. Four things carry the
weight.

*Both examples of section 22 are files on disk.* ``tests/fixtures/macros`` holds
them, and :class:`TestSectionTwentyTwoExamples` reads each one back to the byte,
through the database rows and past the validator. A round trip compared on models
would pass while the dump quietly drifted; comparing text is also what keeps the
fixture a readable diff in a review.

*A file is portable or it is not a file.* No row numbers, no absolute paths, no
``\\`` and no passwords: an exported ``.ayris`` is something users mail to each
other, which is what :class:`TestSecretsNeverLeave` and the checks on the fixture
text are for.

*An older format is read, a newer one is refused.* Version 1 is not invented — it is
the shape :mod:`ayris.core.portable_profile` already writes — so
:class:`TestFormatVersions` migrates a real bundle and pins that a file from a newer
Ayris says so instead of silently dropping the fields this build does not know.

*Every problem points at a block.* The editor of task 33 selects the offending block
by the path in the problem, so :class:`TestValidation` checks paths as carefully as
messages.

Groups:

* :class:`TestSectionTwentyTwoExamples` — the two fixtures, five ways.
* :class:`TestTriggers` — voice with slots and regex, hotkey, event, timer.
* :class:`TestBlocks` — the tree: branches, limits, paths, what a dump leaves out.
* :class:`TestVariablesAndSounds` — declarations, stages and the mapping form.
* :class:`TestFormatVersions` — migration from a bundle, refusal of the future.
* :class:`TestDatabaseMapping` — three tables and the declaration header.
* :class:`TestSecretsNeverLeave` — masking on the way out.
* :class:`TestValidation` — broken parameters, unknown references, call cycles.
* :class:`TestFilesOnDisk` — reading, writing, and what is not a document at all.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ayris.actions.base import SECRET_MASK
from ayris.actions.macros.format_migrations import (
    CURRENT_FORMAT_VERSION,
    MacroFormatError,
    document_version,
    migrate_document,
)
from ayris.actions.macros.schema import (
    MAX_BLOCK_DEPTH,
    MAX_BLOCKS,
    ActionBlock,
    CommandModel,
    HotkeyTrigger,
    SoundBinding,
    SoundSource,
    TimerTrigger,
    VariableModel,
    VoiceTrigger,
)
from ayris.actions.macros.serializer import (
    DECLARATIONS_TYPE,
    AyrisDocument,
    command_from_rows,
    command_to_row,
    command_to_rows,
    dump_command,
    dump_commands,
    dump_document,
    initial_variables,
    load_command,
    load_document,
    mask_secrets,
    read_document,
    triggers_to_rows,
    write_document,
)
from ayris.actions.macros.validator import (
    MacroValidationError,
    Severity,
    ensure_valid,
    validate_command,
    validate_library,
)
from ayris.actions.registry import ActionRegistry
from ayris.core.models import Command, TriggerType, VariableScope, VariableType

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "macros"
EXAMPLES = ("volume_50.ayris", "work_mode.ayris")

#: A profile bundle as :mod:`ayris.core.portable_profile` writes it: triggers are the
#: database rows they came from, and there are no variables and no sounds at all.
V1_BUNDLE: dict[str, Any] = {
    "schema_version": 1,
    "kind": "collection",
    "folders": [{"path": ["Режимы"], "sort_order": 0}],
    "commands": [
        {
            "name": "Свет",
            "folder": None,
            "triggers": [
                {
                    "type": "voice",
                    "payload": {
                        "template": "айрис свет {level:int}",
                        "enabled": True,
                        "when_app": "code",
                    },
                    "fuzzy": True,
                    "priority": 40,
                },
                {
                    "type": "voice",
                    "payload": {"regex": "^айрис (свет|лампа)$", "threshold": 0.8},
                    "fuzzy": True,
                },
                {"type": "hotkey", "payload": {"keys": "Ctrl + Alt + L"}},
                {"type": "event", "payload": {"event": "app.started", "filter": {"name": "code"}}},
                {"type": "timer", "payload": {"cron": "0 9 * * 1-5"}},
            ],
            "actions": [{"type": "SetVar", "params": {"name": "light", "value": "{level}"}}],
        }
    ],
}


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    """The real registry: parameters are only worth checking against real actions."""
    built = ActionRegistry()
    built.discover()
    return built


def _text(name: str) -> str:
    """One fixture exactly as it sits on disk."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def _example(name: str) -> CommandModel:
    return load_command(_text(name))


class TestSectionTwentyTwoExamples:
    """The two commands of section 22, as the fixtures they were turned into."""

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_the_file_reads_back_byte_for_byte(self, name: str) -> None:
        """Round trip on the text, not on the model: a dump nobody can diff is not one."""
        text = _text(name)
        assert dump_command(load_command(text)) == text

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_the_document_around_the_command_is_the_file(self, name: str) -> None:
        document = read_document(FIXTURES / name)
        assert document.schema_version == CURRENT_FORMAT_VERSION
        assert document.kind == "command"
        assert document.exported_at is None
        assert dump_document(document) == _text(name)

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_nothing_in_the_file_belongs_to_one_machine(self, name: str) -> None:
        """No row numbers, no drive letters, no ``\\``: the file is meant to travel."""
        text = _text(name)
        assert "\\" not in text
        assert re.search(r"[A-Za-z]:[\\/]", text) is None
        for command in json.loads(text)["commands"]:
            assert "id" not in command
            assert "folder_id" not in command

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_the_database_keeps_every_field(self, name: str) -> None:
        """Save and read back: three tables have to hold what one model does."""
        command = _example(name)
        row, triggers = command_to_rows(command, profile_id=1, command_id=7)
        restored = command_from_rows(row, triggers, folder=command.folder)
        assert restored.model_dump() == command.model_dump()

    @pytest.mark.parametrize("name", EXAMPLES)
    def test_both_examples_pass_the_validator(self, name: str, registry: ActionRegistry) -> None:
        report = validate_command(_example(name), registry=registry)
        assert report.ok, report.user_message

    def test_the_first_example_says_what_section_22_says(self) -> None:
        command = _example("volume_50.ayris")
        voice = command.voice_triggers[0]
        assert voice.phrase == "айрис громкость {volume:int}"
        assert voice.fuzzy is True
        assert voice.slot_names == ("volume",)
        assert command.hotkey_triggers[0].combo == "ctrl+alt+v"
        assert [location.block.type for location in command.blocks()] == ["SetVolume"]
        assert command.actions[0].params == {"level": "{volume}"}
        assert [sound.reference for sound in command.sounds] == ["builtin:volume_changed"]

    def test_the_second_example_says_what_section_22_says(self) -> None:
        command = _example("work_mode.ayris")
        assert command.folder == ["Режимы"]
        declared = command.variables[0]
        assert declared.name == "work_monitor_brightness"
        assert declared.type is VariableType.INT
        assert declared.scope is VariableScope.PROFILE
        assert (declared.default, declared.persistent) == (70, True)
        assert [location.path_text for location in command.blocks()] == [
            "actions[0]",
            "actions[1]",
            "actions[2]",
            "actions[3]",
            "actions[3].then[0]",
        ]
        assert [sound.reference for sound in command.sounds] == [
            "tts:Рабочий режим активирован. Хорошего кода!",
            "custom:work_mode_start.wav",
        ]

    def test_row_numbers_do_not_travel_even_when_the_command_has_them(self) -> None:
        command = _example("volume_50.ayris").model_copy(update={"id": 7, "folder_id": 3})
        payload = json.loads(dump_command(command))["commands"][0]
        assert "id" not in payload
        assert "folder_id" not in payload


class TestTriggers:
    """Four ways to fire a command, each with the part of itself that can be checked."""

    @pytest.mark.parametrize(
        ("phrase", "regex", "key", "slots"),
        [
            ("айрис привет", False, "phrase", ()),
            ("айрис громкость {volume:int}", False, "template", ("volume",)),
            ("^айрис (свет|лампа)$", True, "regex", ()),
        ],
    )
    def test_the_payload_key_follows_the_pattern(
        self, phrase: str, regex: bool, key: str, slots: tuple[str, ...]
    ) -> None:
        """The matcher reads three kinds of pattern from three keys, and this chooses."""
        trigger = VoiceTrigger(phrase=phrase, regex=regex)
        assert trigger.payload_key == key
        assert trigger.slot_names == slots

    def test_a_regular_expression_is_never_fuzzy(self) -> None:
        """There is no near miss to score, so the flag is dropped instead of ignored."""
        assert VoiceTrigger(phrase="^айрис$", regex=True, fuzzy=True).fuzzy is False

    def test_a_pattern_that_cannot_work_is_refused_at_save_time(self) -> None:
        """Task 30's promise: an error in the editor, not silence when the user speaks."""
        with pytest.raises(ValidationError, match="не компилируется"):
            VoiceTrigger(phrase="айрис (", regex=True)
        with pytest.raises(ValidationError, match="Неизвестный тип слота"):
            VoiceTrigger(phrase="айрис свет {level:кот}")

    def test_conditions_are_the_matcher_keys_and_nothing_else(self) -> None:
        assert VoiceTrigger(phrase="айрис свет", conditions={"when_app": "code"}).conditions
        with pytest.raises(ValidationError, match="when_"):
            VoiceTrigger(phrase="айрис свет", conditions={"mode": "работа"})

    @pytest.mark.parametrize(
        "text", ["Ctrl + Alt + V", "[LCONTROL][LMENU][V]", "^!v", "CTRL-ALT-V"]
    )
    def test_a_combination_is_stored_in_one_spelling(self, text: str) -> None:
        """Task 37 finds two commands claiming one combination by comparing strings."""
        assert HotkeyTrigger(combo=text).combo == "ctrl+alt+v"
        assert HotkeyTrigger(combo=text).hotkey.label_ru == "Ctrl + Alt + V"

    def test_an_unreadable_combination_names_the_token(self) -> None:
        with pytest.raises(ValidationError, match="Не понимаю «бубубо»"):
            HotkeyTrigger(combo="ctrl+бубубо")

    def test_a_timer_is_a_moment_or_a_schedule_but_not_both(self) -> None:
        assert TimerTrigger(cron="0 9 * * 1-5").cron == "0 9 * * 1-5"
        assert TimerTrigger(fire_at=datetime(2026, 1, 1, tzinfo=UTC)).cron is None
        with pytest.raises(ValidationError, match="либо время fire_at"):
            TimerTrigger()
        with pytest.raises(ValidationError, match="либо время fire_at"):
            TimerTrigger(cron="0 9 * * *", fire_at=datetime(2026, 1, 1, tzinfo=UTC))

    def test_a_schedule_has_to_look_like_a_schedule(self) -> None:
        with pytest.raises(ValidationError, match="не похоже на расписание cron"):
            TimerTrigger(cron="каждое утро")

    def test_an_unknown_kind_of_trigger_is_one_error_and_not_four(self) -> None:
        """The union is read by its tag, so a typo does not produce four failed guesses."""
        with pytest.raises(ValidationError) as raised:
            CommandModel.model_validate({"name": "К", "triggers": [{"type": "смс"}]})
        assert raised.value.error_count() == 1
        assert raised.value.errors()[0]["type"] == "union_tag_invalid"

    def test_the_two_kinds_downstream_asks_for_are_told_apart(self) -> None:
        command = _example("volume_50.ayris")
        assert len(command.voice_triggers) == 1
        assert len(command.hotkey_triggers) == 1
        assert len(command.triggers) == 2


class TestBlocks:
    """The block tree: what shape the model itself enforces, and what it writes out."""

    def test_a_branch_a_block_cannot_have_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="не бывает ветки then"):
            ActionBlock(type="SetVolume", params={"level": 50}, then=[ActionBlock(type="Break")])

    def test_switch_holds_only_its_own_arms(self) -> None:
        """``Else`` cannot end up orphaned and a ``Switch`` cannot hold loose blocks."""
        with pytest.raises(ValidationError, match="только Case и Default"):
            ActionBlock(type="Switch", params={"value": "{x}"}, body=[ActionBlock(type="Break")])

    def test_else_is_read_and_written_under_the_name_a_file_uses(self) -> None:
        block = ActionBlock.model_validate(
            {
                "type": "If",
                "params": {"condition": "1"},
                "then": [{"type": "Break"}],
                "else": [{"type": "Continue"}],
            }
        )
        assert [child.type for child in block.else_] == ["Continue"]
        assert block.model_dump(by_alias=True)["else"][0]["type"] == "Continue"

    def test_an_empty_branch_is_not_written_at_all(self) -> None:
        """Four branch fields on every block would double a file for nothing."""
        written = ActionBlock(type="Break").model_dump(by_alias=True, exclude_none=True)
        assert set(written) == {"type", "enabled", "comment", "on_error"}

    def test_the_tree_is_walked_parents_first_with_the_path_of_each_block(self) -> None:
        command = CommandModel.model_validate(
            {
                "name": "К",
                "actions": [
                    {
                        "type": "Try",
                        "body": [
                            {
                                "type": "While",
                                "params": {"condition": "1"},
                                "body": [{"type": "Break"}],
                            }
                        ],
                        "catch": [{"type": "Return"}],
                    }
                ],
            }
        )
        assert [(location.path_text, location.depth) for location in command.blocks()] == [
            ("actions[0]", 0),
            ("actions[0].body[0]", 1),
            ("actions[0].body[0].body[0]", 2),
            ("actions[0].catch[0]", 1),
        ]

    def test_a_tree_deeper_than_the_limit_is_refused_with_the_path(self) -> None:
        """A hand-edited file must not be able to run the interpreter out of stack."""
        nested: dict[str, Any] = {"type": "Break"}
        for _ in range(MAX_BLOCK_DEPTH - 1):
            nested = {"type": "If", "params": {"condition": "1"}, "then": [nested]}
        assert CommandModel.model_validate({"name": "К", "actions": [nested]}).actions
        too_deep = {"type": "If", "params": {"condition": "1"}, "then": [nested]}
        with pytest.raises(ValidationError, match="вложен глубже"):
            CommandModel.model_validate({"name": "К", "actions": [too_deep]})

    def test_more_blocks_than_the_limit_is_refused(self) -> None:
        blocks = [{"type": "Break"}] * (MAX_BLOCKS + 1)
        with pytest.raises(ValidationError, match=f"больше допустимых {MAX_BLOCKS}"):
            CommandModel.model_validate({"name": "К", "actions": blocks})

    def test_a_block_carries_its_own_sound(self) -> None:
        block = ActionBlock.model_validate({"type": "Break", "sound": "builtin:click"})
        assert block.sound is not None
        assert block.sound.reference == "builtin:click"

    def test_a_block_knows_whether_the_language_runs_it_itself(self) -> None:
        logic = ActionBlock(type="For", params={"var": "i", "items": "{список}"})
        assert logic.is_logic is True
        assert logic.spec is not None
        assert logic.spec.required_params == ("var",)
        assert ActionBlock(type="SetVolume", params={"level": 50}).is_logic is False
        assert ActionBlock(type="SetVolume", params={"level": 50}).spec is None

    def test_the_declaration_header_can_never_be_a_block(self) -> None:
        """What the serializer's header relies on: ``#`` is not a block name."""
        with pytest.raises(ValidationError, match="не похоже на имя блока"):
            ActionBlock(type=DECLARATIONS_TYPE)


class TestVariablesAndSounds:
    """Declarations and stage sounds, in both spellings section 22 uses."""

    @pytest.mark.parametrize(
        ("written", "declared"),
        [
            ("str", VariableType.STRING),
            ("integer", VariableType.INT),
            ("boolean", VariableType.BOOL),
            ("list", VariableType.ARRAY),
            ("map", VariableType.DICT),
        ],
    )
    def test_the_spelling_of_a_type_does_not_matter(
        self, written: str, declared: VariableType
    ) -> None:
        """The editor says ``str``, the database CHECK says ``string``; both are read."""
        assert VariableModel(name="x", type=written).type is declared

    def test_a_starting_value_has_to_fit_the_declared_type(self) -> None:
        """The one place a type *can* be checked: later the variable holds what it holds."""
        assert VariableModel(name="x", type="float", default=1).default == 1.0
        with pytest.raises(ValidationError, match="не подходит"):
            VariableModel(name="x", type="int", default="громко")
        with pytest.raises(ValidationError, match="не подходит"):
            VariableModel(name="x", type="int", default=True)

    def test_a_local_variable_cannot_outlive_the_command(self) -> None:
        with pytest.raises(ValidationError, match="не может быть постоянной"):
            VariableModel(name="x", persistent=True)

    def test_a_name_that_is_not_an_identifier_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="имя переменной"):
            VariableModel(name="2 режима")

    def test_one_variable_cannot_be_declared_twice(self) -> None:
        with pytest.raises(ValidationError, match="объявлена дважды"):
            CommandModel.model_validate(
                {"name": "К", "variables": [{"name": "x"}, {"name": "x", "type": "int"}]}
            )

    def test_the_mapping_form_of_section_22_is_read(self) -> None:
        """A file a person edited by hand loads, and the specification's own examples do."""
        command = CommandModel.model_validate(
            {
                "name": "К",
                "variables": {"work_mode": {"type": "bool", "scope": "global", "default": False}},
                "sounds": {
                    "on_success": "builtin:volume_changed",
                    "on_error": {"source": "tts", "value": "Не вышло"},
                },
            }
        )
        assert command.variables[0].name == "work_mode"
        assert command.variables[0].scope is VariableScope.GLOBAL
        assert [sound.reference for sound in command.sounds] == [
            "builtin:volume_changed",
            "tts:Не вышло",
        ]

    @pytest.mark.parametrize(
        ("reference", "source", "value"),
        [
            ("builtin:volume_changed", SoundSource.BUILTIN, "volume_changed"),
            ("custom:work_mode_start.wav", SoundSource.FILE, "work_mode_start.wav"),
            ("tts:Готово", SoundSource.TTS, "Готово"),
        ],
    )
    def test_a_sound_is_one_string_or_three_fields(
        self, reference: str, source: SoundSource, value: str
    ) -> None:
        binding = SoundBinding.model_validate(reference)
        assert binding.source is source
        assert binding.value == value
        assert binding.reference == reference

    @pytest.mark.parametrize(
        "value",
        ["C:/Users/Артем/старт.wav", "C:\\Users\\Артем\\старт.wav", "звуки/старт.wav", "~/с.wav"],
    )
    def test_a_file_sound_is_a_name_and_never_a_path(self, value: str) -> None:
        """Задача 30 forbids absolute paths outright, and this is where that lands."""
        with pytest.raises(ValidationError):
            SoundBinding(source="file", value=value)

    def test_a_file_sound_has_to_be_a_sound(self) -> None:
        with pytest.raises(ValidationError, match="должен быть файлом"):
            SoundBinding(source="file", value="старт.flac")

    def test_a_stage_takes_a_phrase_and_a_file_but_not_two_of_a_kind(self) -> None:
        """Section 22's second example both speaks and plays a file on success."""
        both = CommandModel.model_validate(
            {
                "name": "К",
                "sounds": [
                    {"stage": "on_success", "source": "tts", "value": "Готово"},
                    {"stage": "on_success", "source": "file", "value": "готово.wav"},
                ],
            }
        )
        assert len(both.sounds) == 2
        with pytest.raises(ValidationError, match="два звука"):
            CommandModel.model_validate(
                {"name": "К", "sounds": [{"value": "click"}, {"value": "clack"}]}
            )

    def test_only_variables_that_outlive_the_command_get_a_row(self) -> None:
        """``initial_variables`` is for the first save; a local variable has no row to be in."""
        command = CommandModel.model_validate(
            {
                "name": "К",
                "variables": [
                    {"name": "tmp", "type": "int", "default": 1},
                    {
                        "name": "brightness",
                        "type": "int",
                        "scope": "profile",
                        "default": 70,
                        "persistent": True,
                    },
                    {"name": "mode", "type": "bool", "scope": "global", "default": False},
                ],
            }
        )
        rows = initial_variables(command, profile_id=3)
        assert [row.name for row in rows] == ["brightness", "mode"]
        assert rows[0].scope is VariableScope.PROFILE
        assert rows[0].profile_id == 3
        assert rows[0].value == 70
        assert rows[1].profile_id is None


class TestFormatVersions:
    """An older file is read, a newer one is refused, and both say so in Russian."""

    def test_a_profile_bundle_becomes_the_current_contract(self) -> None:
        document = load_document(V1_BUNDLE)
        assert document.schema_version == CURRENT_FORMAT_VERSION
        voice, regex, hotkey, event, timer = document.commands[0].triggers
        assert isinstance(voice, VoiceTrigger)
        assert voice.phrase == "айрис свет {level:int}"
        assert voice.priority == 40
        assert voice.fuzzy is True
        assert voice.conditions == {"when_app": "code"}
        assert isinstance(regex, VoiceTrigger)
        assert regex.regex is True
        assert regex.fuzzy is False
        assert regex.fuzzy_threshold == 0.8
        assert isinstance(hotkey, HotkeyTrigger)
        assert hotkey.combo == "ctrl+alt+l"
        assert getattr(event, "event_name", "") == "app.started"
        assert getattr(event, "filter_json", {}) == {"name": "code"}
        assert getattr(timer, "cron", "") == "0 9 * * 1-5"

    def test_a_bundle_without_a_folder_lands_at_the_root(self) -> None:
        assert load_document(V1_BUNDLE).commands[0].folder == []

    def test_the_file_that_was_read_is_left_alone(self) -> None:
        """The caller may still want to show the user what was in it."""
        migrated = migrate_document(V1_BUNDLE)
        assert migrated["schema_version"] == CURRENT_FORMAT_VERSION
        assert V1_BUNDLE["schema_version"] == 1
        assert V1_BUNDLE["commands"][0]["triggers"][0]["payload"]["template"]

    @pytest.mark.parametrize(
        ("document", "expected"),
        [
            ({}, "нет поля schema_version"),
            ({"schema_version": "2"}, "не похожа на число"),
            ({"schema_version": True}, "не похожа на число"),
            ({"schema_version": 0}, "слишком старая"),
            ({"schema_version": 99}, "более новой версией"),
        ],
    )
    def test_a_version_that_cannot_be_read_says_which(
        self, document: dict[str, Any], expected: str
    ) -> None:
        with pytest.raises(MacroFormatError) as raised:
            document_version(document)
        assert expected in raised.value.user_message

    def test_a_file_from_a_newer_ayris_names_both_versions(self) -> None:
        """Dropping fields on import and saving is how a user loses work, so it is refused."""
        newer = CURRENT_FORMAT_VERSION + 1
        with pytest.raises(MacroFormatError) as raised:
            load_document({"schema_version": newer, "commands": []})
        message = raised.value.user_message
        assert f"формат {newer}" in message
        assert f"до {CURRENT_FORMAT_VERSION}-го" in message

    def test_a_trigger_of_an_unknown_kind_stops_the_import(self) -> None:
        bundle = {
            "schema_version": 1,
            "commands": [{"name": "К", "triggers": [{"type": "смс", "payload": {}}]}],
        }
        with pytest.raises(MacroFormatError) as raised:
            load_document(bundle)
        assert "неизвестного вида «смс»" in raised.value.user_message


class TestDatabaseMapping:
    """Three tables for a model with more fields than three tables have columns."""

    def test_a_command_without_declarations_writes_the_plain_array(self) -> None:
        """What task 3 expects and what every other reader of the column understands."""
        command = CommandModel(name="К", actions=[ActionBlock(type="Break")])
        row = command_to_row(command, profile_id=1)
        assert [entry["type"] for entry in row.actions] == ["Break"]

    def test_variables_and_sounds_travel_in_a_header_in_front_of_the_blocks(self) -> None:
        command = _example("work_mode.ayris")
        row = command_to_row(command, profile_id=1)
        header, *blocks = row.actions
        assert header["type"] == DECLARATIONS_TYPE
        assert [declared["name"] for declared in header["variables"]] == ["work_monitor_brightness"]
        assert len(header["sounds"]) == 2
        assert [entry["type"] for entry in blocks] == ["SetVar", "RunApp", "RunApp", "If"]

    def test_a_command_with_no_id_gets_no_trigger_rows(self) -> None:
        """A zero passed down instead would look like a real row number."""
        command = _example("volume_50.ayris")
        row, triggers = command_to_rows(command, profile_id=1)
        assert row.id is None
        assert triggers == ()
        assert triggers_to_rows(command, command_id=7)[0].command_id == 7

    def test_the_payload_keys_are_the_ones_the_matcher_reads(self) -> None:
        """Task 3 writes the row, the matcher of task 12 reads it: one spelling for both."""
        command = _example("volume_50.ayris")
        voice, hotkey = triggers_to_rows(command, command_id=3)
        assert voice.type is TriggerType.VOICE
        assert voice.payload["template"] == "айрис громкость {volume:int}"
        assert voice.fuzzy is True
        assert voice.priority == 100
        assert hotkey.type is TriggerType.HOTKEY
        assert hotkey.payload["combo"] == "ctrl+alt+v"

    def test_a_folder_is_a_number_in_the_row_and_a_name_in_the_model(self) -> None:
        """Naming a ``folder_id`` is the folder table's job, so the caller passes it in."""
        command = _example("work_mode.ayris")
        row = command_to_row(command, profile_id=1)
        assert row.folder_id is None
        restored = command_from_rows(row, folder=["Режимы"])
        assert restored.folder == ["Режимы"]
        assert [declared.name for declared in restored.variables] == ["work_monitor_brightness"]

    def test_a_hand_edited_actions_column_names_the_block_that_broke(self) -> None:
        """The column is JSON someone may have edited, so reading it back validates."""
        row = Command(name="К", profile_id=1, actions=({"type": "Break", "оп": 1},))
        with pytest.raises(ValidationError) as raised:
            command_from_rows(row)
        assert "actions.0.оп" in str(raised.value)


class TestSecretsNeverLeave:
    """A password typed into an action stays on the machine it was typed on."""

    def _with_password(self) -> CommandModel:
        return CommandModel(
            name="Домашний Wi-Fi",
            actions=[
                ActionBlock(
                    type="If",
                    params={"condition": "{home} == true"},
                    then=[
                        ActionBlock(
                            type="ConnectWifi",
                            params={"ssid": "Ayris", "password": "очень-секретно"},
                        )
                    ],
                )
            ],
        )

    def test_a_secret_parameter_is_masked_wherever_the_block_sits(
        self, registry: ActionRegistry
    ) -> None:
        """Nested branches are walked too — a secret does not become safe inside an ``If``."""
        masked = mask_secrets(self._with_password(), registry)
        inner = masked.actions[0].then[0]
        assert inner.params["password"] == SECRET_MASK
        assert inner.params["ssid"] == "Ayris"

    def test_masking_leaves_the_command_in_memory_alone(self, registry: ActionRegistry) -> None:
        """The running command still needs the real password; only the copy is masked."""
        command = self._with_password()
        mask_secrets(command, registry)
        assert command.actions[0].then[0].params["password"] == "очень-секретно"

    def test_the_file_carries_the_mask_when_the_registry_is_there_to_ask(
        self, registry: ActionRegistry
    ) -> None:
        text = dump_command(self._with_password(), registry=registry)
        assert "очень-секретно" not in text
        assert SECRET_MASK in text

    def test_without_a_registry_the_caller_owns_the_decision(self) -> None:
        """Nothing knows which parameters are secret, so nothing is quietly dropped."""
        text = dump_command(self._with_password())
        assert "очень-секретно" in text


class TestValidation:
    """Everything wrong with a command, each thing with the path to its block."""

    def test_a_missing_and_an_unknown_parameter_are_two_problems_at_one_path(
        self, registry: ActionRegistry
    ) -> None:
        """A typo in a name loses the parameter and gains a stranger, and says both."""
        command = CommandModel(name="К", actions=[ActionBlock(type="SetVolume", params={"lvl": 5})])
        report = validate_command(command, registry=registry)
        assert [problem.path for problem in report.errors] == ["actions[0]", "actions[0]"]
        assert "нужен параметр «level»" in report.errors[0].message
        assert "нет параметра «lvl»" in report.errors[1].message

    def test_a_parameter_of_the_wrong_type_says_what_was_expected(
        self, registry: ActionRegistry
    ) -> None:
        command = CommandModel(
            name="К", actions=[ActionBlock(type="SetVolume", params={"level": "громко"})]
        )
        report = validate_command(command, registry=registry)
        assert "параметр «level»: нужно целое число" in report.user_message

    def test_problems_come_in_the_order_the_editor_draws_the_blocks(
        self, registry: ActionRegistry
    ) -> None:
        """The list is read next to the blocks, so it follows them, not their severity."""
        command = CommandModel(
            name="К",
            actions=[
                ActionBlock(type="SetVolume", params={}),
                ActionBlock(type="Say", params={"text": "готово"}),
                ActionBlock(type="SetVolume", params={"level": "тихо"}),
            ],
        )
        report = validate_command(command, registry=registry)
        assert [problem.path for problem in report.problems] == [
            "triggers",
            "actions[0]",
            "actions[1]",
            "actions[2]",
        ]

    def test_a_placeholder_declared_by_a_trigger_slot_is_left_alone(
        self, registry: ActionRegistry
    ) -> None:
        """``{volume}`` is a number at runtime; the slot in the phrase is the declaration."""
        command = _example("volume_50.ayris")
        assert validate_command(command, registry=registry).problems == ()

    def test_a_variable_the_caller_knows_about_is_not_a_broken_reference(
        self, registry: ActionRegistry
    ) -> None:
        """Profile and global variables live in the database, not in the command."""
        command = CommandModel(
            name="К", actions=[ActionBlock(type="SetVolume", params={"level": "{loud}"})]
        )
        assert validate_command(command, registry=registry).errors
        assert validate_command(command, registry=registry, known_variables=["loud"]).ok

    def test_a_reference_to_nothing_points_at_the_parameter_it_sits_in(
        self, registry: ActionRegistry
    ) -> None:
        """A block can hold ten parameters, so the path goes down to the one that broke."""
        command = CommandModel(
            name="К", actions=[ActionBlock(type="Say", params={"text": "здравствуй {кто}"})]
        )
        report = validate_command(command, registry=registry)
        problem = report.errors[0]
        assert problem.path == "actions[0].params.text"
        assert problem.message == "ссылка на «кто»: нет ни такой переменной, ни такого слота"

    def test_an_action_the_build_does_not_have_yet_is_a_warning_not_an_error(
        self, registry: ActionRegistry
    ) -> None:
        """A file made in a newer build still opens, with a note next to the block."""
        command = CommandModel(name="К", actions=[ActionBlock(type="Say", params={"text": "да"})])
        report = validate_command(command, registry=registry)
        assert [problem.message for problem in report.warnings if problem.block == "Say"] == [
            "действие «Say» описано в ТЗ, но эта сборка его ещё не умеет"
        ]

    def test_without_a_registry_nothing_is_said_about_block_types(self) -> None:
        """Silence beats guessing: the parameters of an unknown action are unknowable."""
        command = CommandModel(name="К", actions=[ActionBlock(type="SetVolume", params={"lvl": 5})])
        report = validate_command(command)
        assert report.ok
        assert [problem.severity for problem in report.problems] == [Severity.WARNING]

    def test_a_command_with_no_trigger_saves_with_a_note(self, registry: ActionRegistry) -> None:
        """Warnings do not stop a save — a command run from the list is a real command."""
        command = CommandModel(
            name="К", actions=[ActionBlock(type="SetVolume", params={"level": 50})]
        )
        report = validate_command(command, registry=registry)
        assert report.ok
        assert report.warnings[0].path == "triggers"
        assert "запустить её можно будет только вручную" in report.warnings[0].message

    @pytest.mark.parametrize(
        ("block", "expected"),
        [
            (ActionBlock(type="If", params={"condition": "1 == 1"}), "пустая ветка then"),
            (ActionBlock(type="Break"), "стоит вне цикла"),
            (ActionBlock(type="Continue"), "стоит вне цикла"),
            (ActionBlock(type="Case", params={"value": 1}), "бывает только внутри Switch"),
            (ActionBlock(type="Default"), "бывает только внутри Switch"),
        ],
    )
    def test_a_block_in_the_wrong_place_says_where_it_belongs(
        self, registry: ActionRegistry, block: ActionBlock, expected: str
    ) -> None:
        """Structure is checked without a registry's help — these blocks are the language."""
        command = CommandModel(name="К", actions=[block])
        messages = " ".join(
            problem.message for problem in validate_command(command, registry=registry).errors
        )
        assert expected in messages

    def test_a_break_inside_a_loop_is_fine(self, registry: ActionRegistry) -> None:
        """The check is about the blocks above, not about the block itself."""
        command = CommandModel(
            name="К",
            actions=[
                ActionBlock(
                    type="While",
                    params={"condition": "1 == 1"},
                    body=[ActionBlock(type="Break")],
                )
            ],
        )
        assert validate_command(command, registry=registry).ok

    def test_a_command_that_calls_itself_says_so_without_a_library(
        self, registry: ActionRegistry
    ) -> None:
        """The shortest cycle needs nothing but the command itself to be found."""
        command = CommandModel(
            name="Свет", actions=[ActionBlock(type="CallCommand", params={"command": "Свет"})]
        )
        report = validate_command(command, registry=registry)
        assert report.errors[0].message == "команда «Свет» вызывает саму себя"
        assert report.errors[0].path == "actions[0]"

    @staticmethod
    def _ring() -> dict[str, CommandModel]:
        """``A → B → C → A``: the cycle no single command can see on its own."""
        return {
            name: CommandModel(
                name=name,
                actions=[ActionBlock(type="CallCommand", params={"command": nxt})],
            )
            for name, nxt in (("A", "B"), ("B", "C"), ("C", "A"))
        }

    def test_a_cycle_through_other_commands_is_shown_as_the_whole_ring(
        self, registry: ActionRegistry
    ) -> None:
        ring = self._ring()
        report = validate_command(ring["A"], registry=registry, library=ring)
        assert report.errors[0].message == "вызовы ходят по кругу: A → B → C → A"

    def test_every_command_in_the_ring_reports_it_from_its_own_side(
        self, registry: ActionRegistry
    ) -> None:
        """Whichever command the editor has open, the cycle is named starting from it."""
        reports = validate_library(self._ring().values(), registry=registry)
        assert {name: report.user_message for name, report in reports.items()} == {
            "A": "actions[0]: вызовы ходят по кругу: A → B → C → A",
            "B": "actions[0]: вызовы ходят по кругу: B → C → A → B",
            "C": "actions[0]: вызовы ходят по кругу: C → A → B → C",
        }

    def test_a_call_into_nothing_is_only_a_problem_when_the_library_is_known(
        self, registry: ActionRegistry
    ) -> None:
        """Without the library the command may well exist; guessing would cry wolf."""
        command = CommandModel(
            name="К", actions=[ActionBlock(type="CallCommand", params={"command": "Свет"})]
        )
        assert validate_command(command, registry=registry).ok
        report = validate_command(command, registry=registry, library=self._ring())
        assert report.errors[0].message == "команды «Свет» нет: вызывать нечего"

    def test_ensure_valid_raises_with_the_whole_report_attached(
        self, registry: ActionRegistry
    ) -> None:
        """The window shows every problem at once, so the exception carries them all."""
        command = CommandModel(
            name="К",
            actions=[
                ActionBlock(type="SetVolume", params={}),
                ActionBlock(type="SetVolume", params={"level": "тихо"}),
            ],
        )
        with pytest.raises(MacroValidationError) as raised:
            ensure_valid(command, registry=registry)
        assert len(raised.value.report.errors) == 2
        assert raised.value.user_message.startswith("actions[0]: ")

    def test_ensure_valid_returns_the_report_when_only_warnings_were_found(
        self, registry: ActionRegistry
    ) -> None:
        report = ensure_valid(_example("work_mode.ayris"), registry=registry)
        assert report.warnings
        assert report.ok


class TestFilesOnDisk:
    """The file is the unit of sharing, so its bytes and its refusals are pinned."""

    def test_a_written_file_reads_back_the_same_and_has_unix_line_endings(
        self, tmp_path: Path
    ) -> None:
        """The same file is written on Windows and read in the Linux CI job."""
        document = load_document(_text("work_mode.ayris"))
        path = tmp_path / "экспорт.ayris"
        write_document(path, document)
        assert b"\r\n" not in path.read_bytes()
        assert read_document(path) == document

    def test_a_file_that_is_not_json_says_so_before_anything_else(self, tmp_path: Path) -> None:
        path = tmp_path / "битый.ayris"
        path.write_text("не json", encoding="utf-8")
        with pytest.raises(MacroFormatError) as raised:
            read_document(path)
        assert raised.value.user_message == "Файл повреждён: это не JSON."

    def test_a_json_array_is_not_a_document(self) -> None:
        with pytest.raises(MacroFormatError) as raised:
            load_document("[]")
        assert raised.value.user_message == "Файл команд должен быть JSON-объектом."

    def test_a_missing_file_names_the_file_and_not_the_whole_path(self, tmp_path: Path) -> None:
        """The path may hold the user's name; the window shows the file name."""
        with pytest.raises(MacroFormatError) as raised:
            read_document(tmp_path / "нет.ayris")
        assert raised.value.user_message == "Не могу прочитать файл «нет.ayris»."

    def test_a_folder_export_is_not_imported_as_one_command(self) -> None:
        """Taking the first command and dropping the rest is the failure being prevented."""
        text = dump_commands([_example(name) for name in EXAMPLES])
        assert load_document(text).commands
        with pytest.raises(MacroFormatError) as raised:
            load_command(text)
        assert raised.value.user_message == (
            "В файле 2 команд, а нужна одна. Импортируйте его как папку."
        )

    def test_a_document_built_by_hand_is_written_in_the_current_format(self) -> None:
        """Nothing has to remember the number — an export made today is a current export."""
        document = AyrisDocument(commands=[_example("volume_50.ayris")])
        assert document.schema_version == CURRENT_FORMAT_VERSION
        written = json.loads(dump_document(document))
        assert written["schema_version"] == CURRENT_FORMAT_VERSION
        assert written["kind"] == "collection"

    def test_a_folder_export_carries_the_folders_of_its_commands(self) -> None:
        """A subtree arrives with its tree, so an import does not flatten it."""
        document = load_document(dump_commands([_example(name) for name in EXAMPLES]))
        assert [entry.path for entry in document.folders] == [["Режимы"]]
