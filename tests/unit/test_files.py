"""Файловые действия: песочница по разрешённым папкам, корзина, подтверждение.

Как и у питания (``test_power.py``), всё, что меняет диск, идёт через шов
:class:`~ayris.actions.system.files.FileSystem`, а в тестах стоит
:class:`RecordingFileSystem` — он пишет вызов в ``calls`` вместо того, чтобы его
делать. Поэтому ни один обычный тест не трогает настоящий диск; единственное
исключение — windows-only ``TestRealDelete`` под ``tmp_path``. Чтение безвредно
и идёт прямо по ``tmp_path``.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from ayris.actions.registry import ActionRegistry
from ayris.actions.system import files
from ayris.actions.system.files import (
    CopyFile,
    CreateFolder,
    DeleteFile,
    FileInfo,
    FindFiles,
    ListDir,
    MoveFile,
    ReadTextFile,
)
from ayris.core.errors import ActionError, ActionNotConfirmed

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """``tmp_path`` as the one allowed root; ``_config`` patched to point at it."""
    from ayris.core.config import FilesActionsConfig

    monkeypatch.setattr(files, "_config", lambda: FilesActionsConfig(allowed_dirs=[str(tmp_path)]))
    return tmp_path


@pytest.fixture(autouse=True)
def recorder() -> Iterator[files.RecordingFileSystem]:
    """Recording backend installed for every test; no real mutation can happen."""
    rec = files.RecordingFileSystem()
    files.set_file_system(rec)
    try:
        yield rec
    finally:
        files.set_file_system(None)


class TestSandbox:
    def test_no_allowed_dirs_denies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from ayris.core.config import FilesActionsConfig

        monkeypatch.setattr(files, "_config", lambda: FilesActionsConfig(allowed_dirs=[]))
        with pytest.raises(ActionError) as info:
            files._resolve_inside(str(tmp_path / "a.txt"))
        assert "не настроен" in info.value.user_message.casefold()

    def test_inside_is_accepted(self, root: Path) -> None:
        resolved = files._resolve_inside(str(root / "note.txt"))
        assert resolved == root / "note.txt"

    def test_outside_is_rejected(self, root: Path) -> None:
        with pytest.raises(ActionError):
            files._resolve_inside(str(root.parent / "secret.txt"))

    def test_traversal_is_rejected(self, root: Path) -> None:
        with pytest.raises(ActionError):
            files._resolve_inside(str(root / ".." / "secret.txt"))


class TestMetadata:
    def test_delete_and_move_are_dangerous(self) -> None:
        assert DeleteFile.meta.is_dangerous is True
        assert MoveFile.meta.is_dangerous is True

    def test_reads_and_benign_writes_are_not_dangerous(self) -> None:
        assert ListDir.meta.is_dangerous is False
        assert CopyFile.meta.is_dangerous is False
        assert CreateFolder.meta.is_dangerous is False

    def test_category_is_files(self) -> None:
        from ayris.actions.base import ActionCategory

        assert DeleteFile.meta.category is ActionCategory.FILES


class TestReads:
    def test_list_dir_counts_files_and_folders(self, root: Path) -> None:
        (root / "a.txt").write_text("x", encoding="utf-8")
        (root / "b.txt").write_text("y", encoding="utf-8")
        (root / "sub").mkdir()
        result = ListDir().run(ListDir.Params(path=str(root)))
        assert result.ok
        assert result.value is not None
        assert len(result.value) == 3
        assert result.data["dirs"] == 1
        assert result.data["files"] == 2

    def test_find_files_by_pattern(self, root: Path) -> None:
        (root / "report.pdf").write_text("x", encoding="utf-8")
        (root / "notes.txt").write_text("y", encoding="utf-8")
        (root / "deep").mkdir()
        (root / "deep" / "old.pdf").write_text("z", encoding="utf-8")
        result = FindFiles().run(FindFiles.Params(root=str(root), pattern="*.pdf"))
        assert result.ok
        assert result.data["count"] == 2

    def test_read_text_file(self, root: Path) -> None:
        target = root / "hello.txt"
        target.write_text("привет мир", encoding="utf-8")
        result = ReadTextFile().run(ReadTextFile.Params(path=str(target)))
        assert result.ok
        assert result.value == "привет мир"
        assert result.data["truncated"] is False

    def test_read_text_file_truncates_at_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ayris.core.config import FilesActionsConfig

        monkeypatch.setattr(
            files,
            "_config",
            lambda: FilesActionsConfig(allowed_dirs=[str(tmp_path)], max_read_bytes=1024),
        )
        target = tmp_path / "big.txt"
        target.write_text("a" * 4096, encoding="utf-8")
        result = ReadTextFile().run(ReadTextFile.Params(path=str(target)))
        assert result.ok
        assert result.data["truncated"] is True
        assert len(result.value or "") == 1024

    def test_file_info_missing(self, root: Path) -> None:
        result = FileInfo().run(FileInfo.Params(path=str(root / "nope.txt")))
        assert not result.ok
        assert result.value is not None
        assert result.value.exists is False


class TestMutations:
    def test_delete_goes_to_trash_by_default(
        self, root: Path, recorder: files.RecordingFileSystem
    ) -> None:
        target = root / "junk.txt"
        target.write_text("x", encoding="utf-8")
        result = DeleteFile().run(DeleteFile.Params(path=str(target)))
        assert result.ok
        assert recorder.calls == [("trash", str(target))]
        assert target.exists()  # рекордер не трогал диск

    def test_delete_permanent_removes(
        self, root: Path, recorder: files.RecordingFileSystem
    ) -> None:
        target = root / "junk.txt"
        target.write_text("x", encoding="utf-8")
        result = DeleteFile().run(DeleteFile.Params(path=str(target), permanent=True))
        assert result.ok
        assert recorder.calls == [("remove", str(target), False)]

    def test_move_refuses_overwrite(self, root: Path, recorder: files.RecordingFileSystem) -> None:
        src = root / "a.txt"
        dst = root / "b.txt"
        src.write_text("x", encoding="utf-8")
        dst.write_text("y", encoding="utf-8")
        result = MoveFile().run(MoveFile.Params(source=str(src), destination=str(dst)))
        assert not result.ok
        assert recorder.calls == []

    def test_move_records_when_clear(self, root: Path, recorder: files.RecordingFileSystem) -> None:
        src = root / "a.txt"
        src.write_text("x", encoding="utf-8")
        dst = root / "moved.txt"
        result = MoveFile().run(MoveFile.Params(source=str(src), destination=str(dst)))
        assert result.ok
        assert recorder.calls == [("move", str(src), str(dst))]

    def test_copy_refuses_directory_without_recursive(
        self, root: Path, recorder: files.RecordingFileSystem
    ) -> None:
        src = root / "folder"
        src.mkdir()
        result = CopyFile().run(CopyFile.Params(source=str(src), destination=str(root / "copy")))
        assert not result.ok
        assert recorder.calls == []

    def test_create_folder_refuses_existing(
        self, root: Path, recorder: files.RecordingFileSystem
    ) -> None:
        existing = root / "there"
        existing.mkdir()
        result = CreateFolder().run(CreateFolder.Params(path=str(existing)))
        assert not result.ok
        assert recorder.calls == []


class TestConfirmation:
    def test_refusal_stops_delete(self, root: Path, recorder: files.RecordingFileSystem) -> None:
        target = root / "x.txt"
        target.write_text("x", encoding="utf-8")
        registry = ActionRegistry(audit_enabled=lambda: False, confirm=lambda _request: False)
        registry.add(DeleteFile)
        try:
            with pytest.raises(ActionNotConfirmed):
                registry.execute("DeleteFile", {"path": str(target)})
        finally:
            registry.shutdown()
        assert recorder.calls == []

    def test_confirmation_lets_delete_through(
        self, root: Path, recorder: files.RecordingFileSystem
    ) -> None:
        target = root / "x.txt"
        target.write_text("x", encoding="utf-8")
        registry = ActionRegistry(audit_enabled=lambda: False, confirm=lambda _request: True)
        registry.add(DeleteFile)
        try:
            result = registry.execute("DeleteFile", {"path": str(target)})
        finally:
            registry.shutdown()
        assert result.ok
        assert recorder.calls == [("trash", str(target))]


@pytest.mark.hardware
@pytest.mark.skipif(sys.platform != "win32", reason="удаляет настоящий файл в корзину")
class TestRealDelete:
    def test_removes_to_trash(self, root: Path) -> None:
        pytest.importorskip("send2trash")
        files.set_file_system(files.RealFileSystem())
        try:
            target = root / "real-delete.txt"
            target.write_text("x", encoding="utf-8")
            result = DeleteFile().run(DeleteFile.Params(path=str(target)))
            assert result.ok
            assert not target.exists()
        finally:
            files.set_file_system(None)
