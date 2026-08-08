"""Consistency checks for packaging metadata.

Dependency versions live in two places: ``pyproject.toml`` (the real install) and
``requirements-ci.txt`` (the trimmed set CI installs, without speech engines and
WinAPI wrappers that have no wheels on a runner). A version raised in one file and
forgotten in the other means the sandbox, CI and a developer machine each type-check
and test against a different library — the kind of drift that surfaces as an
unreproducible failure weeks later.

These tests are cheap and run everywhere, so the drift is caught at commit time.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS_CI = PROJECT_ROOT / "requirements-ci.txt"

#: ``name==version`` with the extras/markers tail ignored.
_PIN = re.compile(r"^(?P<name>[A-Za-z0-9._-]+)\s*==\s*(?P<version>[^\s;#]+)")


def _normalize(name: str) -> str:
    """PEP 503 name normalisation: ``PySide6`` and ``pyside6`` are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_pins(lines: list[str]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _PIN.match(line)
        if match is not None:
            pins[_normalize(match["name"])] = match["version"]
    return pins


def _pyproject_pins() -> dict[str, str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    lines: list[str] = list(project["dependencies"])
    for extra in project.get("optional-dependencies", {}).values():
        lines.extend(extra)
    return _parse_pins(lines)


def _requirements_pins() -> dict[str, str]:
    return _parse_pins(REQUIREMENTS_CI.read_text(encoding="utf-8").splitlines())


@pytest.mark.unit
def test_requirements_ci_exists() -> None:
    assert REQUIREMENTS_CI.is_file(), "requirements-ci.txt пропал — CI не соберётся"


@pytest.mark.unit
def test_ci_requirements_match_pyproject() -> None:
    """Every pin in requirements-ci.txt must equal the one in pyproject.toml."""
    project_pins = _pyproject_pins()
    ci_pins = _requirements_pins()

    unknown = sorted(set(ci_pins) - set(project_pins))
    assert not unknown, (
        f"в requirements-ci.txt есть пакеты, которых нет в pyproject.toml: {unknown}. "
        "Добавь их в dependencies или в optional-dependencies."
    )

    mismatched = {
        name: (project_pins[name], ci_pins[name])
        for name in sorted(ci_pins)
        if project_pins[name] != ci_pins[name]
    }
    assert not mismatched, (
        "версии разошлись между pyproject.toml и requirements-ci.txt "
        f"(pyproject, requirements-ci): {mismatched}"
    )


@pytest.mark.unit
def test_every_dependency_is_pinned() -> None:
    """Loose bounds make a build non-reproducible; the project pins exactly."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    unpinned = [
        spec for spec in data["project"]["dependencies"] if _PIN.match(spec.strip()) is None
    ]
    assert not unpinned, f"зависимости без точной версии: {unpinned}"


@pytest.mark.unit
def test_runtime_imports_are_covered_by_ci_requirements() -> None:
    """Anything ``src/ayris`` imports at module level must be installable in CI.

    A task that adds ``import sounddevice`` to a module without adding the package
    to requirements-ci.txt turns every CI run red on collection, which looks like a
    broken test suite rather than a missing pin.
    """
    ci_pins = _requirements_pins()
    # stdlib and first-party imports are irrelevant here; this is the list of
    # third-party names the code may reach for once a stage lands. Extend it in the
    # same commit that adds the import.
    known_third_party = {
        "PySide6": "pyside6",
        "pydantic": "pydantic",
        "pydantic_settings": "pydantic-settings",
        "tomlkit": "tomlkit",
        "keyring": "keyring",
        "numpy": "numpy",
    }
    sources = (PROJECT_ROOT / "src" / "ayris").rglob("*.py")
    pattern = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)

    missing: dict[str, str] = {}
    for path in sources:
        for module in pattern.findall(path.read_text(encoding="utf-8")):
            distribution = known_third_party.get(module)
            if distribution is not None and distribution not in ci_pins:
                missing[module] = str(path.relative_to(PROJECT_ROOT))
    assert not missing, (
        "модуль импортируется, но пакета нет в requirements-ci.txt — CI упадёт "
        f"на сборе тестов: {missing}"
    )
