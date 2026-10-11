"""Поиск локальных файлов моделей: что попадает в список и как разводятся имена."""

from __future__ import annotations

from pathlib import Path

import pytest

from ayris.nlu.llm.local_models import (
    MAX_LOCAL_MODELS,
    LocalModel,
    discover_local_models,
    local_model_dirs,
)

pytestmark = pytest.mark.unit


def _gguf(directory: Path, name: str, size: int = 16) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{name}.gguf"
    file.write_bytes(b"\0" * size)
    return file


def test_finds_files_in_the_folder_and_its_direct_subfolders(tmp_path: Path) -> None:
    _gguf(tmp_path, "alpha")
    _gguf(tmp_path / "downloaded", "beta")
    _gguf(tmp_path / "deep" / "deeper", "gamma")  # глубже двух уровней не ищем

    found = discover_local_models([tmp_path])

    assert [model.name for model in found] == ["alpha", "beta"]


def test_ignores_everything_that_is_not_a_gguf(tmp_path: Path) -> None:
    _gguf(tmp_path, "model")
    (tmp_path / "readme.txt").write_text("не модель", encoding="utf-8")
    (tmp_path / "model.gguf.part").write_bytes(b"\0")

    found = discover_local_models([tmp_path])

    assert [model.name for model in found] == ["model"]


def test_same_file_seen_through_two_folders_appears_once(tmp_path: Path) -> None:
    _gguf(tmp_path, "solo")

    found = discover_local_models([tmp_path, tmp_path])

    assert len(found) == 1


def test_same_stem_in_two_folders_gets_a_folder_suffix(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _gguf(left, "twin")
    _gguf(right, "twin")

    found = discover_local_models([left, right])

    assert sorted(model.name for model in found) == ["twin", "twin (right)"]
    assert len({model.path for model in found}) == 2


def test_missing_folder_is_simply_empty(tmp_path: Path) -> None:
    assert discover_local_models([tmp_path / "нет такой"]) == ()


def test_the_list_is_capped(tmp_path: Path) -> None:
    for index in range(MAX_LOCAL_MODELS + 5):
        _gguf(tmp_path, f"m{index:03d}")

    assert len(discover_local_models([tmp_path])) == MAX_LOCAL_MODELS


def test_dirs_include_the_profile_folder_and_the_current_file_folder(tmp_path: Path) -> None:
    profile = tmp_path / "profile" / "llm"
    mine = tmp_path / "D" / "weights"
    profile.mkdir(parents=True)
    current = _gguf(mine, "mine")

    dirs = local_model_dirs(models_dir=profile, current_path=current)

    assert dirs == (profile, mine)


def test_dirs_drop_duplicates_and_paths_that_are_not_folders(tmp_path: Path) -> None:
    folder = tmp_path / "llm"
    folder.mkdir()
    file = _gguf(folder, "x")

    dirs = local_model_dirs(models_dir=folder, current_path=file, extra=[tmp_path / "ghost"])

    assert dirs == (folder,)


def test_dirs_without_anything_configured_are_empty() -> None:
    assert local_model_dirs() == ()


def test_human_size_reads_in_gigabytes_above_one() -> None:
    assert LocalModel("x", Path("x.gguf"), 2 * 1024**3).human_size == "≈2.0 ГБ"
    assert LocalModel("x", Path("x.gguf"), 800 * 1024**2).human_size == "≈800 МБ"
