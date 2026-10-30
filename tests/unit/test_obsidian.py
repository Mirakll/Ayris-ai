"""Unit tests for the Obsidian 2c interface server and bridge."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from ayris.core.config import ConfigManager
from ayris.gui.obsidian.server import ensure_obsidian_server


class _MockWindow:
    def __init__(self) -> None:
        self._manager = ConfigManager()
        self._current_state = "idle"
        self._current_mode_name = "hybrid"
        self._current_profile_name = "По умолчанию"
        self._profiles = lambda: []


def _fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, data: dict[str, Any]) -> dict[str, Any]:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))


def test_obsidian_server_endpoints(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # Ping endpoint
    ping_res = _fetch_json(f"{base_url}/api/ping")
    assert ping_res == {"status": "ok"}

    # Status endpoint
    status_res = _fetch_json(f"{base_url}/api/status")
    assert status_res["state"] == "idle"
    assert status_res["mode"] == "hybrid"
    assert status_res["profile"] == "По умолчанию"

    # Settings endpoint
    settings_res = _fetch_json(f"{base_url}/api/settings")
    assert "general" in settings_res
    assert "voice" in settings_res
    assert "ai" in settings_res


def test_obsidian_bridge_signals(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    signals = server.signals

    received: list[str] = []

    def on_command(cmd: str) -> None:
        received.append(f"cmd:{cmd}")

    def on_voice() -> None:
        received.append("voice")

    def on_mode(mode: str) -> None:
        received.append(f"mode:{mode}")

    def on_profile(prof: str) -> None:
        received.append(f"profile:{prof}")

    def on_action(act: str) -> None:
        received.append(f"action:{act}")

    from PySide6.QtCore import Qt

    signals.command_requested.connect(on_command, Qt.ConnectionType.DirectConnection)
    signals.voice_toggled.connect(on_voice, Qt.ConnectionType.DirectConnection)
    signals.mode_requested.connect(on_mode, Qt.ConnectionType.DirectConnection)
    signals.profile_requested.connect(on_profile, Qt.ConnectionType.DirectConnection)
    signals.window_action_requested.connect(on_action, Qt.ConnectionType.DirectConnection)

    assert _post_json(f"{base_url}/api/command", {"text": "привет"}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/toggle-mic", {}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/set-mode", {"mode": "chat"}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/switch-profile", {"name": "Рабочий"}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/window-action", {"action": "minimize"}) == {"status": "ok"}

    qtbot.waitUntil(lambda: len(received) == 5, timeout=2000)

    assert "cmd:привет" in received
    assert "voice" in received
    assert "mode:chat" in received
    assert "profile:Рабочий" in received
    assert "action:minimize" in received


def test_obsidian_update_setting(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    updated: list[tuple[str, Any]] = []
    from PySide6.QtCore import Qt

    server.signals.setting_update_requested.connect(
        lambda p, v: updated.append((p, v)), Qt.ConnectionType.DirectConnection
    )

    assert _post_json(
        f"{base_url}/api/update-setting",
        {"path": "general.start_minimized", "value": True},
    ) == {"status": "ok"}

    qtbot.waitUntil(lambda: len(updated) == 1, timeout=2000)
    assert updated[0] == ("general.start_minimized", True)


def test_obsidian_voice_endpoints(qtbot: Any) -> None:
    mock_window = _MockWindow()
    mock_window._wake_detections = 5
    mock_window._wake_false_positives = 1
    mock_window._last_wake_phrase = "айрис"
    mock_window._last_wake_confidence = 0.95

    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    signals = server.signals

    events: list[str] = []
    from PySide6.QtCore import Qt

    signals.voice_triggered.connect(
        lambda: events.append("triggered"), Qt.ConnectionType.DirectConnection
    )
    signals.pick_model_requested.connect(
        lambda: events.append("pick_model"), Qt.ConnectionType.DirectConnection
    )
    signals.wake_reset_requested.connect(
        lambda: events.append("reset_wake"), Qt.ConnectionType.DirectConnection
    )
    signals.wake_mark_false_requested.connect(
        lambda: events.append("mark_false"), Qt.ConnectionType.DirectConnection
    )

    # Test trigger-voice
    assert _post_json(f"{base_url}/api/trigger-voice", {}) == {"status": "ok"}
    # Test pick-model
    assert _post_json(f"{base_url}/api/voice/pick-model", {}) == {"status": "ok"}
    # Test reset-wake
    assert _post_json(f"{base_url}/api/voice/reset-wake", {})["status"] == "ok"
    # Test mark-false-wake
    assert _post_json(f"{base_url}/api/voice/mark-false-wake", {}) == {"status": "ok"}

    qtbot.waitUntil(lambda: len(events) == 4, timeout=2000)
    assert events == ["triggered", "pick_model", "reset_wake", "mark_false"]

    # Test wake-status
    status = _fetch_json(f"{base_url}/api/voice/wake-status")
    assert status["detections"] == 5
    assert status["false_positives"] == 1
    assert status["last_phrase"] == "айрис"
    assert status["last_confidence"] == 0.95

    # Test devices endpoint
    devs = _fetch_json(f"{base_url}/api/voice/devices")
    assert "devices" in devs
    assert isinstance(devs["devices"], list)


def test_obsidian_settings_actions(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"
    signals = server.signals

    events: list[str] = []
    from PySide6.QtCore import Qt

    signals.pick_color_requested.connect(
        lambda: events.append("pick_color"), Qt.ConnectionType.DirectConnection
    )
    signals.pick_gguf_requested.connect(
        lambda: events.append("pick_gguf"), Qt.ConnectionType.DirectConnection
    )
    signals.pick_folder_requested.connect(
        lambda target: events.append(f"pick_folder:{target}"), Qt.ConnectionType.DirectConnection
    )

    # System folders
    folders = _fetch_json(f"{base_url}/api/system/folders")
    assert "root" in folders
    assert "logs" in folders

    # Secret save, status, delete
    save_res = _post_json(
        f"{base_url}/api/secret/save",
        {"ref": "test_ref_key", "value": "test_val_123"},
    )
    assert save_res["status"] == "ok"

    status_res = _fetch_json(f"{base_url}/api/secret/status?ref=test_ref_key")
    assert status_res["ref"] == "test_ref_key"

    del_res = _post_json(f"{base_url}/api/secret/delete", {"ref": "test_ref_key"})
    assert del_res["status"] == "ok"

    # Pick color & gguf signals
    assert _post_json(f"{base_url}/api/overlay/pick-color", {}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/ai/pick-gguf", {}) == {"status": "ok"}
    assert _post_json(
        f"{base_url}/api/ai/pick-folder",
        {"target": "actions.files.allowed_dirs"},
    ) == {"status": "ok"}

    qtbot.waitUntil(lambda: len(events) == 3, timeout=2000)
    assert events == ["pick_color", "pick_gguf", "pick_folder:actions.files.allowed_dirs"]

    # AI reset-prompt
    prompt_res = _post_json(f"{base_url}/api/ai/reset-prompt", {})
    assert prompt_res["status"] == "ok"
    assert "prompt" in prompt_res

    # Voice calibrate
    calib_res = _post_json(f"{base_url}/api/voice/calibrate", {})
    assert calib_res["status"] == "ok"
    assert calib_res["gain"] == 1.2

    # General rerun & reset
    assert _post_json(f"{base_url}/api/general/rerun-onboarding", {}) == {"status": "ok"}
    assert _post_json(f"{base_url}/api/general/reset-defaults", {}) == {"status": "ok"}


def test_obsidian_profiles_tab_endpoints(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # Profiles list
    profs_res = _fetch_json(f"{base_url}/api/profiles/list")
    assert "profiles" in profs_res
    assert isinstance(profs_res["profiles"], list)
    assert "data_dir" in profs_res
    assert "backups" in profs_res

    # Create profile
    created = _post_json(f"{base_url}/api/profiles/create", {"name": "ТестПрофиль"})
    assert created["status"] == "ok"
    assert created["profile"]["name"] == "ТестПрофиль"

    # Rename profile
    renamed = _post_json(
        f"{base_url}/api/profiles/rename",
        {"name": "ТестПрофиль", "new_name": "ТестПрофиль2"},
    )
    assert renamed["status"] == "ok"

    # Duplicate profile
    duplicated = _post_json(f"{base_url}/api/profiles/duplicate", {"name": "ТестПрофиль2"})
    assert duplicated["status"] == "ok"

    # Delete profiles created
    del_res1 = _post_json(f"{base_url}/api/profiles/delete", {"name": "ТестПрофиль2"})
    assert del_res1["status"] == "ok"
    if duplicated.get("profile"):
        dup_name = duplicated["profile"]["name"]
        _post_json(f"{base_url}/api/profiles/delete", {"name": dup_name})

    # Backup & Export
    bk_res = _post_json(f"{base_url}/api/profiles/backup", {})
    assert bk_res["status"] == "ok"
    assert "filename" in bk_res

    # Delete backup
    del_bk = _post_json(f"{base_url}/api/profiles/delete-backup", {"filename": bk_res["filename"]})
    assert del_bk["status"] == "ok"


def test_obsidian_devtools_and_models_endpoints(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # Devtools state
    dt_state = _fetch_json(f"{base_url}/api/devtools/state")
    assert "common_level" in dt_state
    assert "module_levels" in dt_state

    # Module level set and reset
    set_mod = _post_json(
        f"{base_url}/api/devtools/module-level",
        {"module": "ayris.test.mod", "level": "DEBUG"},
    )
    assert set_mod["status"] == "ok"

    reset_mod = _post_json(
        f"{base_url}/api/devtools/module-reset",
        {"module": "ayris.test.mod"},
    )
    assert reset_mod["status"] == "ok"

    # REPL
    repl_res = _post_json(f"{base_url}/api/devtools/repl", {"code": "1 + 1"})
    assert repl_res["status"] == "ok"
    assert "2" in repl_res["output"]

    # Models list
    models_res = _fetch_json(f"{base_url}/api/models/list")
    assert "installed" in models_res
    assert "catalog" in models_res
    assert "disk" in models_res

    # Updates check and install
    up_check = _post_json(f"{base_url}/api/updates/check", {})
    assert up_check["status"] == "ok"
    assert "latest_version" in up_check

    up_inst = _post_json(f"{base_url}/api/updates/install", {})
    assert up_inst["status"] == "ok"

    # Hotkeys reset
    hk_res = _post_json(f"{base_url}/api/hotkeys/reset", {})
    assert hk_res["status"] == "ok"

    # Commands list
    cmds_res = _fetch_json(f"{base_url}/api/commands/list")
    assert "commands" in cmds_res

    # AI preview prompt
    preview_res = _post_json(
        f"{base_url}/api/ai/preview-prompt",
        {"kind": "chat", "persona": "Тестовая персона"},
    )
    assert preview_res["status"] == "ok"
    assert "Тестовая персона" in preview_res["preview"]
    assert preview_res["tokens"] > 0

    # AI reset prompt
    reset_prompt_res = _post_json(
        f"{base_url}/api/ai/reset-prompt",
        {"kind": "chat"},
    )
    assert reset_prompt_res["status"] == "ok"
    assert "Ты Айрис" in reset_prompt_res["prompt"]
    assert reset_prompt_res["tokens"] > 0

    # AI allowed dirs
    add_dir_res = _post_json(
        f"{base_url}/api/ai/allowed-dirs",
        {"action": "add", "dir": "C:\\test_ai_dir"},
    )
    assert add_dir_res["status"] == "ok"
    assert "C:\\test_ai_dir" in add_dir_res["allowed_dirs"]

    remove_dir_res = _post_json(
        f"{base_url}/api/ai/allowed-dirs",
        {"action": "remove", "dir": "C:\\test_ai_dir"},
    )
    assert remove_dir_res["status"] == "ok"
    assert "C:\\test_ai_dir" not in remove_dir_res["allowed_dirs"]

    # AI clear history
    clear_hist_res = _post_json(f"{base_url}/api/ai/clear-history", {})
    assert clear_hist_res["status"] == "ok"


def test_obsidian_privacy_and_commands_extended(qtbot: Any) -> None:
    mock_window = _MockWindow()
    server = ensure_obsidian_server(window=mock_window, port=0)  # type: ignore[arg-type]
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # Secret check
    sec_check = _fetch_json(f"{base_url}/api/secret/check?ref=gemini")
    assert sec_check["status"] == "ok"
    assert "stored" in sec_check

    # Privacy endpoints
    clean_res = _post_json(f"{base_url}/api/privacy/cleanup", {})
    assert clean_res["status"] == "ok"

    clear_hist = _post_json(f"{base_url}/api/privacy/clear-history", {})
    assert clear_hist["status"] == "ok"

    clear_clip = _post_json(f"{base_url}/api/privacy/clear-clipboard", {})
    assert clear_clip["status"] == "ok"

    clear_cache = _post_json(f"{base_url}/api/privacy/clear-cache", {})
    assert clear_cache["status"] == "ok"
    assert "removed" in clear_cache

    del_logs = _post_json(f"{base_url}/api/privacy/delete-logs", {})
    assert del_logs["status"] == "ok"
    assert "deleted" in del_logs

    reset_prof = _post_json(f"{base_url}/api/privacy/reset-profile", {})
    assert reset_prof["status"] == "ok"

    audit_export = _post_json(f"{base_url}/api/privacy/export-audit", {"format": "json"})
    assert audit_export["status"] == "ok"
    assert "records" in audit_export

    dt_clear_logs = _post_json(f"{base_url}/api/devtools/clear-logs", {})
    assert dt_clear_logs["status"] == "ok"
