"""Task 39: UAC state, elevation helpers, and reversible always-admin mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ayris.actions.base import Action, ActionCategory, ActionMeta, ActionParams
from ayris.actions.registry import ActionRegistry
from ayris.actions.result import ActionResult
from ayris.core.database import Database, reset_database
from ayris.core.errors import ActionNotConfirmed
from ayris.core.models import ExecutionResult
from ayris.core.paths import init_paths, reset_paths
from ayris.core.repositories import Repositories
from ayris.utils import admin, winapi

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean_caches() -> None:
    admin.reset_elevation_cache()
    reset_database()
    reset_paths()
    yield
    admin.reset_elevation_cache()
    reset_database()
    reset_paths()


def status(monkeypatch: pytest.MonkeyPatch, *, elevated: bool, uac: bool, member: bool):
    monkeypatch.setattr(admin, "is_elevated", lambda: elevated)
    monkeypatch.setattr(admin, "uac_enabled", lambda: uac)
    monkeypatch.setattr(admin, "user_is_admin", lambda: member)
    return admin.admin_status()


class TestAdminStatus:
    def test_uac_disabled_has_its_own_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answer = status(monkeypatch, elevated=False, uac=False, member=True)
        assert answer.state is admin.AdminCapability.UAC_DISABLED
        assert "UAC выключен" in answer.message_ru

    def test_plain_user_has_its_own_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answer = status(monkeypatch, elevated=False, uac=True, member=False)
        assert answer.state is admin.AdminCapability.NOT_ADMIN
        assert "не входит в группу" in answer.message_ru

    def test_filtered_admin_can_show_uac(self, monkeypatch: pytest.MonkeyPatch) -> None:
        answer = status(monkeypatch, elevated=False, uac=True, member=True)
        assert answer.state is admin.AdminCapability.CAN_ELEVATE
        assert "диалог UAC" in answer.message_ru

    def test_group_membership_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = 0

        def member() -> bool:
            nonlocal calls
            calls += 1
            return True

        monkeypatch.setattr(winapi, "available", lambda: True)
        monkeypatch.setattr(winapi, "current_user_is_admin", member)
        assert admin.user_is_admin() is True
        assert admin.user_is_admin() is True
        assert calls == 1
        admin.reset_elevation_cache()
        assert admin.user_is_admin() is True
        assert calls == 2

    def test_filtered_token_counts_as_an_administrator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeFunction:
            restype: object = None
            argtypes: object = None

            def __init__(self, callback) -> None:
                self.callback = callback

            def __call__(self, *args: object) -> bool:
                return bool(self.callback(*args))

        create_sid = FakeFunction(lambda *_args: True)

        def check(_token: object, _sid: object, result: object) -> bool:
            import ctypes

            ctypes.cast(result, ctypes.POINTER(ctypes.c_bool)).contents.value = False
            return True

        check_membership = FakeFunction(check)

        def require(_library: str, function: str) -> FakeFunction:
            return create_sid if function == "CreateWellKnownSid" else check_membership

        monkeypatch.setattr(winapi, "_require", require)
        monkeypatch.setattr(
            winapi,
            "process_elevation",
            lambda: winapi.ElevationInfo(elevation_type=winapi.ELEVATION_TYPE_LIMITED),
        )
        assert winapi.current_user_is_admin() is True

    @pytest.mark.skipif(sys.platform != "win32", reason="CheckTokenMembership is Windows-only")
    def test_live_windows_admin_status_is_consistent(self) -> None:
        answer = admin.admin_status()
        assert isinstance(answer.elevated, bool)
        assert isinstance(answer.uac_enabled, bool)
        assert isinstance(answer.user_is_admin, bool)
        assert answer.state in admin.AdminCapability


class TestElevatedRun:
    def test_cancelled_uac_is_distinct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi, "available", lambda: True)

        def decline(*_args: object, **_kwargs: object) -> winapi.ProcessRun:
            raise winapi.WinApiError("cancelled", code=winapi.ERROR_CANCELLED)

        monkeypatch.setattr(winapi, "shell_execute_ex", decline)
        with pytest.raises(admin.ElevationDeclined):
            admin.run_elevated("helper.exe")

    def test_other_launch_error_is_not_a_decline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(winapi, "available", lambda: True)

        def broken(*_args: object, **_kwargs: object) -> winapi.ProcessRun:
            raise winapi.WinApiError("missing", code=2)

        monkeypatch.setattr(winapi, "shell_execute_ex", broken)
        with pytest.raises(winapi.WinApiError) as caught:
            admin.run_elevated("missing.exe")
        assert not isinstance(caught.value, admin.ElevationDeclined)

    def test_declined_uac_is_written_to_audit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class BorrowRights(Action):
            meta = ActionMeta(
                name="BorrowRights",
                category=ActionCategory.SYSTEM,
                title_ru="Операция с повышением",
            )

            def run(self, params: ActionParams) -> ActionResult[None]:
                raise admin.ElevationDeclined("UAC declined")

        monkeypatch.setattr("ayris.actions.registry._process_is_elevated", lambda: False)
        database = Database.open(tmp_path / "audit.db")
        repos = Repositories(database)
        registry = ActionRegistry(audit=repos.audit, audit_enabled=lambda: True)
        registry.add(BorrowRights)
        try:
            with pytest.raises(ActionNotConfirmed):
                registry.execute("BorrowRights")
            entry = repos.audit.recent(1)[0]
        finally:
            registry.shutdown()
            database.close()
        assert entry.result is ExecutionResult.CANCELLED
        assert entry.elevated is False

    def test_output_is_read_and_temp_file_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        init_paths(profile=tmp_path)

        def fake_run(_exe: str, args: list[str], **_kwargs: object) -> winapi.ProcessRun:
            output = Path(args[2])
            output.write_text(
                json.dumps({"stdout": "готово", "stderr": "", "returncode": 0}),
                encoding="utf-8",
            )
            return winapi.ProcessRun(pid=7, exit_code=0)

        monkeypatch.setattr(admin, "run_elevated", fake_run)
        answer = admin.run_elevated_output("tool.exe", ["--value", "два слова"])
        assert answer.stdout == "готово"
        assert answer.run.exit_code == 0
        assert list((tmp_path / "cache" / "elevated").iterdir()) == []


def test_always_admin_creates_and_removes_one_task(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, list[str]]] = []

    def run(executable: str, arguments: list[str]) -> winapi.ProcessRun:
        calls.append((executable, arguments))
        return winapi.ProcessRun(exit_code=0)

    monkeypatch.setattr(admin, "run_elevated", run)
    admin.configure_always_admin(True, executable=r"C:\Ayris\ayris.exe", arguments=["--minimized"])
    admin.configure_always_admin(False)
    assert calls[0][0] == "schtasks.exe"
    assert "/Create" in calls[0][1] and "/RL" in calls[0][1] and "HIGHEST" in calls[0][1]
    assert calls[1][1] == ["/Delete", "/TN", admin.ALWAYS_ADMIN_TASK, "/F"]
    assert "плагин" in admin.ALWAYS_ADMIN_WARNING_RU


def test_manifest_has_safe_defaults() -> None:
    manifest = Path("build/ayris.manifest").read_text(encoding="utf-8")
    assert 'level="asInvoker"' in manifest
    assert 'uiAccess="false"' in manifest
    assert "PerMonitorV2" in manifest
    assert "8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a" in manifest
