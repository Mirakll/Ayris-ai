"""Consistency checks for packaging metadata.

Dependency versions live in two places: ``pyproject.toml`` (the real install) and
``requirements-ci.txt`` (the trimmed set CI installs, without speech engines and
WinAPI wrappers that have no wheels on a runner). A version raised in one file and
forgotten in the other means the sandbox, CI and a developer machine each type-check
and test against a different library — the kind of drift that surfaces as an
unreproducible failure weeks later.

Two more files are the same list under different install flags, so their pins are
checked alongside the others:

* ``requirements-ci-nodeps.txt`` — installed with ``--no-deps``, and the workflow
  is checked for actually passing that flag, because dropping it silently pulls a
  hundred megabytes of transitive dependencies onto three runners.
* ``requirements-ci-models.txt`` — the speech engines, installed in the one job
  that downloads weights, and checked for being installed with
  ``-c requirements-ci.txt``: without the constraint pip raises numpy past the
  project's pin to satisfy scipy, and that job silently stops testing the
  environment the application actually runs in.

These tests are cheap and run everywhere, so the drift is caught at commit time.
The workflow itself is read as text for the same reason: yaml has no way to say
"this flag matters", and the two flags that matter here (``--no-deps``, ``-c``) are
invisible when lost. The pytest marker expressions are checked in the same spirit —
a marker excluded by every job and run by none is indistinguishable from a passing
suite.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
REQUIREMENTS_CI = PROJECT_ROOT / "requirements-ci.txt"
REQUIREMENTS_CI_NODEPS = PROJECT_ROOT / "requirements-ci-nodeps.txt"
REQUIREMENTS_CI_MODELS = PROJECT_ROOT / "requirements-ci-models.txt"
WORKFLOW = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"

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
    """Every pin CI installs, from all three requirements files at once."""
    pins = _parse_pins(REQUIREMENTS_CI.read_text(encoding="utf-8").splitlines())
    pins.update(_parse_pins(REQUIREMENTS_CI_NODEPS.read_text(encoding="utf-8").splitlines()))
    pins.update(_parse_pins(REQUIREMENTS_CI_MODELS.read_text(encoding="utf-8").splitlines()))
    return pins


@pytest.mark.unit
def test_requirements_ci_exists() -> None:
    assert REQUIREMENTS_CI.is_file(), "requirements-ci.txt пропал — CI не соберётся"
    assert REQUIREMENTS_CI_NODEPS.is_file(), "requirements-ci-nodeps.txt пропал — CI не соберётся"
    assert REQUIREMENTS_CI_MODELS.is_file(), "requirements-ci-models.txt пропал — CI не соберётся"


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
def test_the_nodeps_requirements_are_installed_without_dependencies() -> None:
    """Every job that installs requirements must install both files, the second
    one with ``--no-deps``.

    Without the flag pip resolves pyrnnoise's ``Requires-Dist`` and drags
    matplotlib, audiolab and av onto all three runners — and, worse, gets a say in
    the numpy version the tests then run against. The flag is easy to lose while
    editing yaml, and nothing else would notice: the suite would still be green,
    just slower and pinned differently.
    """
    workflow = WORKFLOW.read_text(encoding="utf-8")
    installs = [line.strip() for line in workflow.splitlines() if "pip install" in line]

    plain = [line for line in installs if "-r requirements-ci.txt" in line]
    nodeps = [line for line in installs if "-r requirements-ci-nodeps.txt" in line]
    assert len(nodeps) == len(plain) > 0, (
        f"requirements-ci.txt ставится в {len(plain)} местах, "
        f"requirements-ci-nodeps.txt — в {len(nodeps)}"
    )
    without_flag = [line for line in nodeps if "--no-deps" not in line]
    assert not without_flag, f"без --no-deps: {without_flag}"


@pytest.mark.unit
def test_the_engine_requirements_are_installed_under_constraints() -> None:
    """The weights job must install the engines with ``-c requirements-ci.txt``.

    openwakeword asks for scipy and scikit-learn — it imports them only in its
    training code, but pip does not know that — and their current releases require
    ``numpy>=2.3``. Without a constraints file pip resolves that by quietly
    raising numpy past the project's own pin of 1.26.4, and the job stays green
    while testing an environment nobody ships. With it, pip either finds versions
    that fit or fails loudly, which is the outcome we want.

    Only the weights job installs this file, so unlike the other two the count is
    not compared against anything: one place is correct.
    """
    workflow = WORKFLOW.read_text(encoding="utf-8")
    installs = [
        line.strip()
        for line in workflow.splitlines()
        if "pip install" in line and "-r requirements-ci-models.txt" in line
    ]

    assert installs, "requirements-ci-models.txt не ставится ни в одном джобе"
    without_constraint = [line for line in installs if "-c requirements-ci.txt" not in line]
    assert not without_constraint, f"без -c requirements-ci.txt: {without_constraint}"


@pytest.mark.unit
def test_the_special_markers_are_excluded_everywhere_and_run_somewhere() -> None:
    """Each of ``hardware``, ``models``, ``network`` is excluded by every ordinary
    pytest job, and ``models`` is actually run by one of them.

    Both halves have already gone wrong. The markers were introduced one at a
    time, and adding ``network`` to the windows job left the linux job running the
    twenty-two ``HEAD`` requests it was meant to be spared — a job that goes red
    when somebody else's website is down. The other direction is worse and quieter:
    a marker excluded everywhere and run nowhere looks exactly like a green suite,
    which is how eight tests on real weights sat unrun for months.

    ``hardware`` is deliberately absent from the second half: a runner has no
    microphone, and those three tests are for a human at a real machine.
    """
    expressions = [
        line.split('"')[1]
        for line in WORKFLOW.read_text(encoding="utf-8").splitlines()
        if "pytest -m " in line and '"' in line
    ]
    assert expressions, "ни один джоб не запускает pytest с маркерами"

    excluding = [text for text in expressions if text.startswith("not ")]
    assert excluding, "ни один джоб не исключает маркеры"
    for marker in ("hardware", "models", "network"):
        missing = [text for text in excluding if f"not {marker}" not in text]
        assert not missing, f"{marker} не исключён в: {missing}"

    assert any(
        text == "models" for text in expressions
    ), f"маркер models исключён везде и не запускается нигде: {expressions}"


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
        "sounddevice": "sounddevice",
        "webrtcvad": "webrtcvad-wheels",
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
