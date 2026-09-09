from __future__ import annotations

from pathlib import Path

import pytest

from ayris.actions.macros.importers import (
    ConflictStrategy,
    ImportPreview,
    VoiceAttackImporter,
)
from ayris.actions.macros.importers.report import render_report, write_report
from ayris.actions.macros.importers.voiceattack import expand_voice_phrases
from ayris.actions.macros.validator import validate_library
from ayris.core.database import Database
from ayris.core.repositories import CommandRepository, ProfileRepository

FIXTURES = Path(__file__).parents[1] / "fixtures" / "imports"


def test_vap_imports_profile_phrases_hotkey_and_actions() -> None:
    result = VoiceAttackImporter().parse(FIXTURES / "simple.vap")

    assert result.profile_name == "Космос"
    assert result.summary == "Импортировано 1 из 1 команд"
    command = result.commands[0]
    assert command.folder == ["Навигация", "Карты"]
    assert [trigger.phrase for trigger in command.triggers if trigger.type == "voice"] == [
        "Альфа открой карту",
        "Альфа открой звёздную карту",
        "Альфа покажи карту",
    ]
    assert next(trigger.combo for trigger in command.triggers if trigger.type == "hotkey") == (
        "ctrl+alt+m"
    )
    assert [block.type for block in command.actions] == ["RunApp", "Wait", "TypeText"]
    assert all(report.ok for report in validate_library(result.commands).values())


def test_vap_imports_variables_conditions_and_portable_sounds() -> None:
    result = VoiceAttackImporter().parse(FIXTURES / "conditions.vap")
    command = result.commands[0]

    assert command.variables[0].name == "level"
    assert command.variables[0].default == 3
    assert [block.type for block in command.actions] == ["SetVar", "If", "PlaySound"]
    assert command.actions[-1].params == {"sound": "ready.wav"}
    assert command.sounds[0].reference == "custom:ready.wav"
    assert {sound.target_name for sound in result.sounds} == {"ready.wav"}


def test_vap_reports_unknown_action_and_rejects_unsafe_xml(tmp_path: Path) -> None:
    result = VoiceAttackImporter().parse(FIXTURES / "unknown.vap")
    assert result.imported_commands == 1
    assert result.unsupported[0].command == "Телепорт"
    assert result.unsupported[0].kind == "QuantumTeleport"

    unsafe = tmp_path / "unsafe.vap"
    unsafe.write_text(
        '<!DOCTYPE x [<!ENTITY leak SYSTEM "file:///etc/passwd">]><Profile>&leak;</Profile>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="DTD|сущност"):
        VoiceAttackImporter().parse(unsafe)


def test_preview_apply_conflicts_and_report(tmp_path: Path) -> None:
    importer = VoiceAttackImporter()
    result = importer.parse(FIXTURES / "simple.vap")
    preview = ImportPreview(result, selected=set())
    assert preview.commands == ()
    preview.set_selected(0, True)

    with Database.open(tmp_path / "ayris.db") as database:
        profile = ProfileRepository(database).create("Импорт")
        assert profile.id is not None
        first = importer.apply(
            result,
            ("Перенесённое",),
            database=database,
            profile_id=profile.id,
            sounds_dir=tmp_path / "sounds",
            conflicts=ConflictStrategy.RENAME,
        )
        second = importer.apply(
            result,
            ("Перенесённое",),
            database=database,
            profile_id=profile.id,
            sounds_dir=tmp_path / "sounds",
            conflicts=ConflictStrategy.RENAME,
        )
        commands = CommandRepository(database).list_for_profile(profile.id)

    assert first.summary == "Импортировано 1 из 1 команд"
    assert second.renamed == 1
    assert {command.name for command in commands} == {
        "Открыть карту",
        "Открыть карту (импорт 2)",
    }
    assert (tmp_path / "sounds" / "ready.wav").exists() is False
    report_path = tmp_path / "report.txt"
    write_report(report_path, result, second)
    assert report_path.read_text(encoding="utf-8") == render_report(result, second)


def test_expand_voice_phrases_is_bounded() -> None:
    assert expand_voice_phrases("включи [быстро|тихо] свет") == [
        "включи свет",
        "включи быстро свет",
        "включи тихо свет",
    ]
    assert len(expand_voice_phrases("[a|b][c|d][e|f]", limit=4)) == 4
