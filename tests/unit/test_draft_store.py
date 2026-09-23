"""Автосохранение несохранённых правок команды (задача 54).

Черновик — не база: он несохранён по определению, не попадает в экспорт профиля и
не стоит записи на каждый символ. Хранилище лишь читает, пишет и удаляет — таймер
живёт в редакторе, — поэтому тут проверяется файловый цикл без событийного цикла:
после «краха» (записали черновик и забыли про экземпляр) следующий разбор читает
ту же команду; битый черновик считается отсутствующим и удаляется; успешное
сохранение стирает черновик.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.actions.macros.schema import ActionBlock, CommandModel
from ayris.gui.widgets.draft_store import DraftStore

pytestmark = pytest.mark.unit


def _cmd(command_id: int | None, text: str) -> CommandModel:
    return CommandModel(
        id=command_id,
        name="Свет",
        actions=[ActionBlock(type="Say", params={"text": text})],
    )


def test_no_draft_returns_none(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts")
    assert store.load(7) is None
    assert store.has_draft(7) is False


def test_saved_draft_is_restored_after_crash(tmp_path: Path) -> None:
    directory = tmp_path / "drafts"
    # «Крах»: одно хранилище записало черновик и пропало.
    DraftStore(directory).save(_cmd(7, "недописанное"))

    # Новый экземпляр читает тот же файл — как редактор при следующем открытии.
    record = DraftStore(directory).load(7)
    assert record is not None
    assert record.command.actions[0].params["text"] == "недописанное"
    assert record.saved_at is not None


def test_save_creates_directory_lazily(tmp_path: Path) -> None:
    directory = tmp_path / "not_yet"
    assert not directory.exists()
    DraftStore(directory).save(_cmd(7, "текст"))
    assert directory.exists()
    assert DraftStore(directory).has_draft(7) is True


def test_discard_removes_draft_and_stamp(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts")
    store.save(_cmd(7, "текст"))
    assert store.has_draft(7) is True
    store.discard(7)
    assert store.has_draft(7) is False
    assert store.load(7) is None


def test_discard_absent_draft_is_safe(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts")
    store.discard(999)  # ничего нет — не должно падать


def test_unreadable_draft_is_dropped_and_reported_absent(tmp_path: Path) -> None:
    directory = tmp_path / "drafts"
    directory.mkdir()
    # Битый черновик: файл есть, но это не .ayris-документ.
    (directory / "7.ayris").write_text("{ не json", encoding="utf-8")
    store = DraftStore(directory)
    assert store.load(7) is None  # прочитать нельзя — считаем, что черновика нет
    assert store.has_draft(7) is False  # и удалён, чтобы не звать восстановление снова


def test_new_command_uses_zero_key(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts")
    store.save(_cmd(None, "новая"))  # у новой команды нет id — ключ 0
    assert store.has_draft(0) is True
    record = store.load(0)
    assert record is not None and record.command.actions[0].params["text"] == "новая"


def test_save_overwrites_previous_draft(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts")
    store.save(_cmd(7, "первое"))
    store.save(_cmd(7, "второе"))
    record = store.load(7)
    assert record is not None and record.command.actions[0].params["text"] == "второе"
