from __future__ import annotations

from pathlib import Path

from ayris.actions.macros.importers import AutoHotkeyImporter, UnknownLinePolicy
from ayris.actions.macros.validator import validate_library

FIXTURES = Path(__file__).parents[1] / "fixtures" / "imports"


def test_ahk_imports_hotkeys_actions_and_variable() -> None:
    result = AutoHotkeyImporter().parse(FIXTURES / "hotkeys.ahk")

    assert result.summary == "Импортировано 2 из 2 команд"
    first, second = result.commands
    assert first.triggers[0].combo == "ctrl+alt+v"
    assert [block.type for block in first.actions] == [
        "RunApp",
        "FocusWindow",
        "Wait",
        "SetVar",
    ]
    assert first.variables[0].name == "count"
    assert first.variables[0].default == 3
    assert second.triggers[0].combo == "win+n"
    assert [block.type for block in second.actions] == ["TypeText", "KeyPress"]
    assert all(report.ok for report in validate_library(result.commands).values())


def test_ahk_send_braces_become_keyboard_and_text_blocks() -> None:
    command = AutoHotkeyImporter().parse(FIXTURES / "send_braces.ahk").commands[0]

    assert [block.type for block in command.actions] == [
        "TypeText",
        "KeyDown",
        "TypeText",
        "KeyUp",
        "KeyPress",
        "TypeText",
        "TypeText",
    ]
    assert command.actions[-1].params["text"] == "{{literal}}"


def test_ahk_unknown_lines_are_disabled_or_skipped() -> None:
    retained = AutoHotkeyImporter().parse(FIXTURES / "unknown.ahk")
    assert retained.unsupported
    assert all(
        block.type == "RunShell" and not block.enabled for block in retained.commands[0].actions
    )

    skipped = AutoHotkeyImporter(UnknownLinePolicy.SKIP).parse(FIXTURES / "unknown.ahk")
    assert skipped.unsupported
    assert skipped.warnings
    assert skipped.commands[0].actions == []
