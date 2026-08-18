"""What is installed on this machine, and which of it the user just named.

:mod:`ayris.nlu.apps` answers «which program does this phrase mean» from a
dictionary of 34 programs. That is the half that can be reasoned about offline.
This module is the other half: it finds what is *actually* installed, keeps the
answer on disk so the next start does not pay for the walk again, and turns a
resolved name into something a launcher can be handed.

**Five sources, because no single one is complete.** ``App Paths`` is the closest
thing Windows has to a list of launchable executables, but only well-behaved
installers write to it. The uninstall keys know display names — «Яндекс Браузер»
— that no executable carries. The Start menu catches everything else, including
programs installed per-user. ``PATH`` is what makes «запусти ffmpeg» work at all.
And a Store application has no path: it exists only as an AUMID behind
``shell:AppsFolder``, so it is enumerated through the shell namespace or not at
all. Each record remembers which source it came from, and the source decides ties
— see :class:`IndexSource`.

**The cache is JSON, not the database.** Schema v4 has an ``installed_apps``
table, but it has nowhere to put an icon or a launch counter, and this index needs
both. A single JSON file in the cache directory also survives a corrupt write
better: the worst case is one slow scan, not a failed migration. TTL is twelve
hours, refreshed in the background — the user installing a program mid-session is
handled by :meth:`AppIndex.refresh_now`, which is what the settings window's
«Обновить список» button calls.

**Scanning never happens while someone is waiting.** The walk takes seconds; a
slot is filled with the user mid-sentence. So :meth:`AppIndex.ensure_ready`
answers from the cache even when it is stale, and starts a background refresh
instead of blocking. Only the very first run, with no cache at all, waits.

**Ambiguity is a question, not a guess.** When two programs are equally close to
what was said, and neither has been launched more often than the other, the
resolver raises :class:`AppAmbiguous` carrying both names. The pipeline turns that
into «Chrome или Firefox?» — a clarifying question is cheap, launching the wrong
program is not.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Self

from ayris.core.errors import ActionError
from ayris.core.paths import get_paths
from ayris.nlu.apps import (
    DEFAULT_ALIAS_THRESHOLD,
    AppEntry,
    AppResolver,
    AppsError,
    InstalledApp,
    load_apps,
)
from ayris.nlu.matcher import similarity
from ayris.nlu.normalize import normalize_text
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
    from types import ModuleType

__all__ = [
    "AMBIGUITY_GAP",
    "DEFAULT_TTL_S",
    "GENERIC_ALIASES",
    "INDEX_FILE_NAME",
    "INDEX_SCHEMA_VERSION",
    "MAX_INDEXED_APPS",
    "STORE_FOLDER",
    "STORE_PREFIX",
    "AppAmbiguous",
    "AppCandidate",
    "AppIndex",
    "AppIndexResolver",
    "AppIndexSnapshot",
    "AppNotFound",
    "IndexSource",
    "IndexedApp",
    "dedupe",
    "get_app_index",
    "link_catalog",
    "parse_app_paths_entry",
    "parse_path_executable",
    "parse_shortcut",
    "parse_uninstall_entry",
    "parse_uwp_entry",
    "phrase_key",
    "scan_installed",
    "set_app_index",
]

_log = get_logger(__name__)

#: Format version of the cache file. Bumped when a field changes meaning; an
#: older or newer file is discarded and rescanned rather than guessed at.
INDEX_SCHEMA_VERSION: Final = 1

#: Cache file name inside :attr:`ayris.core.paths.AppPaths.cache_dir`.
INDEX_FILE_NAME: Final = "app_index.json"

#: How long a scan stays fresh. Twelve hours: long enough that a normal day costs
#: one scan, short enough that yesterday's installation is found by itself.
DEFAULT_TTL_S: Final = 12 * 3600

#: Ceiling on the whole index. A machine with more launchable things than this has
#: a broken Start menu, and every fuzzy miss walks this list.
MAX_INDEXED_APPS: Final = 4000

#: Ceiling per ``PATH`` directory. ``System32`` alone holds several hundred
#: executables, almost none of which anyone says out loud.
MAX_PATH_EXECUTABLES: Final = 200

#: How close a runner-up has to be to the best match before the resolver stops
#: guessing and asks. Two spellings of the same word differ by less than this.
AMBIGUITY_GAP: Final = 0.06

#: The shell namespace that lists Store applications, and the prefix of the only
#: moniker one of them can be launched by.
STORE_FOLDER: Final = "shell:AppsFolder"
STORE_PREFIX: Final = f"{STORE_FOLDER}\\"

#: Words dropped from the front of a phrase before it is scored. Kept in step
#: with the same list in :mod:`ayris.nlu.apps`, which owns the resolution itself;
#: here it only has to make «открой хром» and «хром» score alike.
_LEADING_WORDS: Final = frozenset(
    {
        "в",
        "во",
        "на",
        "из",
        "с",
        "со",
        "к",
        "ко",
        "у",
        "открой",
        "открыть",
        "запусти",
        "запустить",
        "включи",
        "включить",
        "покажи",
    }
)

#: Category words that name no particular program. Resolved against what is
#: installed: «браузер» is Chrome on one machine and Firefox on the next, and the
#: answer follows the launch history rather than a hardcoded preference.
GENERIC_ALIASES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "браузер": "browser",
        "браузере": "browser",
        "browser": "browser",
        "интернет": "browser",
        "мессенджер": "messenger",
        "редактор": "editor",
        "редакторе": "editor",
        "editor": "editor",
        "плеер": "media",
        "проигрыватель": "media",
    }
)

#: Shortest alias worth a fuzzy comparison, and the tolerance used. Both mirror
#: :mod:`ayris.nlu.apps`: this scan only ranks rivals, the answer itself still
#: comes from :meth:`ayris.nlu.apps.AppResolver.resolve`.
_MIN_RIVAL_LENGTH: Final = 3


class AppNotFound(ActionError):
    """Nothing installed answers to that name."""

    default_user_message = "Не нашла такое приложение."


class AppAmbiguous(ActionError):
    """Several programs answer to that name and nothing separates them.

    ``options`` holds the display names, in the order they should be offered, so
    the follow-up machinery can build the question without re-resolving anything.
    """

    default_user_message = "Не поняла, какое приложение открыть."

    def __init__(
        self,
        technical: str,
        *,
        options: Sequence[str] = (),
        user_message: str | None = None,
    ) -> None:
        super().__init__(technical, user_message=user_message)
        self.options: tuple[str, ...] = tuple(options)


class IndexSource(StrEnum):
    """Where a record came from, and therefore how much it is trusted.

    The weight settles a tie between two records of the same program. The Start
    menu wins because its label is the one the user reads; ``App Paths`` is next
    because it is the authoritative launch path; a Store entry is as good but only
    exists for Store apps; the uninstall keys point at an installer's icon as
    often as at the program; and ``PATH`` is last, being a directory of tools
    rather than of applications.
    """

    START_MENU = "start-menu"
    APP_PATHS = "app-paths"
    UWP = "uwp"
    UNINSTALL = "uninstall"
    PATH = "path"

    @property
    def weight(self) -> int:
        """Trust score, higher wins."""
        return _SOURCE_WEIGHTS[self]

    @property
    def title_ru(self) -> str:
        """How the source is named in the settings window."""
        return _SOURCE_TITLES[self]


_SOURCE_WEIGHTS: Final[Mapping[IndexSource, int]] = MappingProxyType(
    {
        IndexSource.START_MENU: 50,
        IndexSource.APP_PATHS: 45,
        IndexSource.UWP: 40,
        IndexSource.UNINSTALL: 30,
        IndexSource.PATH: 10,
    }
)

_SOURCE_TITLES: Final[Mapping[IndexSource, str]] = MappingProxyType(
    {
        IndexSource.START_MENU: "Меню «Пуск»",
        IndexSource.APP_PATHS: "Реестр, App Paths",
        IndexSource.UWP: "Приложения Microsoft Store",
        IndexSource.UNINSTALL: "Реестр, установленные программы",
        IndexSource.PATH: "Переменная PATH",
    }
)


@dataclass(frozen=True, slots=True)
class IndexedApp:
    """One launchable thing found on this machine.

    ``target`` is what a launcher receives: an executable path, a ``.lnk`` path or
    a ``shell:AppsFolder\\<AUMID>`` moniker. ``icon`` is a Windows icon reference
    — a file, optionally with an index after a comma — and is empty for Store
    applications, whose icon only the shell knows how to draw.
    """

    name: str
    target: str
    source: IndexSource
    executable: str = ""
    aumid: str = ""
    arguments: str = ""
    working_dir: str = ""
    icon: str = ""
    catalog_id: str = ""

    @property
    def is_store(self) -> bool:
        """Whether this can only be started through the shell namespace."""
        return bool(self.aumid)

    @property
    def is_shortcut(self) -> bool:
        """Whether the target is a Start-menu ``.lnk``."""
        return self.target.lower().endswith(".lnk")

    @property
    def key(self) -> str:
        """Identity in the resolver's id space.

        The catalogue id when the scan recognised the program, its folded name
        otherwise — the same rule :meth:`ayris.nlu.apps.AppResolver.with_installed`
        uses, so launch counters and resolved matches agree on what to call a
        program.
        """
        return self.catalog_id or normalize_text(self.name)

    @property
    def dedupe_key(self) -> str:
        """What makes two records the same program for deduplication."""
        return (self.aumid or self.target).lower()

    def as_installed(self) -> InstalledApp:
        """The record in the shape :mod:`ayris.nlu.apps` merges into a resolver."""
        return InstalledApp(
            name=self.name,
            target=self.target,
            executable=self.executable,
            arguments=self.arguments,
            source=str(self.source),
            catalog_id=self.catalog_id,
        )

    def to_json(self) -> dict[str, Any]:
        """Cache representation. Empty fields are dropped to keep the file small."""
        payload: dict[str, Any] = {
            "name": self.name,
            "target": self.target,
            "source": str(self.source),
        }
        for name in ("executable", "aumid", "arguments", "working_dir", "icon", "catalog_id"):
            value = getattr(self, name)
            if value:
                payload[name] = value
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Self | None:
        """Read one cached record, or ``None`` when it is not usable.

        A record that lost its name, its target or its source is dropped rather
        than repaired: the cache is regenerated from the machine in seconds.
        """
        name = str(payload.get("name", "")).strip()
        target = str(payload.get("target", "")).strip()
        raw_source = str(payload.get("source", ""))
        if not name or not target:
            return None
        try:
            source = IndexSource(raw_source)
        except ValueError:
            return None
        return cls(
            name=name,
            target=target,
            source=source,
            executable=str(payload.get("executable", "")),
            aumid=str(payload.get("aumid", "")),
            arguments=str(payload.get("arguments", "")),
            working_dir=str(payload.get("working_dir", "")),
            icon=str(payload.get("icon", "")),
            catalog_id=str(payload.get("catalog_id", "")),
        )


@dataclass(frozen=True, slots=True)
class AppIndexSnapshot:
    """The index as of one scan, plus what has been launched from it.

    Immutable on purpose: a background refresh replaces the snapshot a resolver
    was built from instead of mutating it, so a match in flight keeps answering
    from a consistent picture.
    """

    apps: tuple[IndexedApp, ...] = ()
    scanned_at: float = 0.0
    launches: Mapping[str, int] = field(default_factory=dict)

    def age_s(self, now: float) -> float:
        """Seconds since the scan. Negative clock jumps count as «just now»."""
        return max(0.0, now - self.scanned_at)

    def is_stale(self, ttl_s: float, now: float) -> bool:
        """Whether the scan is old enough to be worth repeating."""
        return self.scanned_at <= 0.0 or self.age_s(now) >= ttl_s

    def with_launches(self, launches: Mapping[str, int]) -> AppIndexSnapshot:
        """Copy carrying a different launch history."""
        return replace(self, launches=dict(launches))

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "scanned_at": self.scanned_at,
            "apps": [app.to_json() for app in self.apps],
            "launches": dict(self.launches),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Self | None:
        """Read a cache file, or ``None`` when it is from another version."""
        if int(payload.get("schema_version", 0)) != INDEX_SCHEMA_VERSION:
            return None
        raw_apps = payload.get("apps", [])
        if not isinstance(raw_apps, list):
            return None
        apps: list[IndexedApp] = []
        for item in raw_apps:
            if isinstance(item, dict):
                app = IndexedApp.from_json(item)
                if app is not None:
                    apps.append(app)
        raw_launches = payload.get("launches", {})
        launches: dict[str, int] = {}
        if isinstance(raw_launches, dict):
            for app_id, count in raw_launches.items():
                if isinstance(count, int) and count > 0:
                    launches[str(app_id)] = count
        return cls(
            apps=tuple(apps),
            scanned_at=float(payload.get("scanned_at", 0.0)),
            launches=launches,
        )


# --------------------------------------------------------------------------- #
# Parsers. Every source is read as raw rows and turned into records here, so the
# interesting part — what counts as launchable — is testable without Windows.
# --------------------------------------------------------------------------- #


def parse_app_paths_entry(key_name: str, values: Mapping[str, object]) -> IndexedApp | None:
    """One ``App Paths`` subkey: the key is the executable, the default value its path.

    The optional ``Path`` value is the working directory the program expects, and
    a few programs genuinely misbehave without it.
    """
    executable = key_name.strip()
    target = str(values.get("", "") or "").strip().strip('"')
    if not executable or not target:
        return None
    return IndexedApp(
        name=Path(executable).stem,
        target=target,
        source=IndexSource.APP_PATHS,
        executable=Path(executable).name,
        working_dir=str(values.get("Path", "") or "").strip().strip('"'),
        icon=target,
    )


def parse_uninstall_entry(values: Mapping[str, object]) -> IndexedApp | None:
    """One uninstall key, kept only when it points at something launchable.

    Most of these keys describe an installation, not a program: runtimes, driver
    packages, update bundles. Three filters cut them out — a ``SystemComponent``
    flag, a missing ``DisplayName``, and a ``DisplayIcon`` that is not an
    executable. What survives is a display name Windows shows in «Программы и
    компоненты» together with the binary that owns its icon, which is very nearly
    always the program itself.
    """
    if _as_int(values.get("SystemComponent")) == 1:
        return None
    display = str(values.get("DisplayName", "") or "").strip()
    if not display:
        return None
    raw_icon = str(values.get("DisplayIcon", "") or "").strip()
    target = raw_icon.split(",")[0].strip('" ')
    if not target.lower().endswith(".exe"):
        return None
    return IndexedApp(
        name=display,
        target=target,
        source=IndexSource.UNINSTALL,
        executable=Path(target).name,
        working_dir=str(values.get("InstallLocation", "") or "").strip().strip('"'),
        icon=raw_icon or target,
    )


def parse_shortcut(path: Path) -> IndexedApp | None:
    """One Start-menu ``.lnk``, labelled by its file name.

    The file name is the label because that is what the Start menu shows and
    therefore what the user says. Where the shortcut points is deliberately not
    resolved: reading a ``.lnk`` needs COM, and the shortcut itself is a perfectly
    good thing to hand ``ShellExecuteW`` — it even carries the arguments and the
    working directory the installer chose.

    Uninstallers are dropped. Every vendor puts one next to the program, they all
    match «удали …» far better than anything the user wants launched by voice, and
    a wrong hit here is destructive.
    """
    if path.suffix.lower() != ".lnk":
        return None
    label = path.stem.strip()
    if not label or _is_uninstaller(label):
        return None
    return IndexedApp(
        name=label,
        target=str(path),
        source=IndexSource.START_MENU,
        icon=str(path),
    )


def parse_path_executable(path: Path) -> IndexedApp | None:
    """One executable found in a ``PATH`` directory."""
    if path.suffix.lower() != ".exe":
        return None
    stem = path.stem.strip()
    if not stem:
        return None
    return IndexedApp(
        name=stem,
        target=str(path),
        source=IndexSource.PATH,
        executable=path.name,
        icon=str(path),
    )


def parse_uwp_entry(name: str, app_id: str) -> IndexedApp | None:
    """One ``shell:AppsFolder`` item, kept only when it is a Store application.

    The shell namespace lists desktop programs alongside Store ones, and for a
    desktop program it hands back a moniker that means nothing outside the Start
    menu. A Store application is recognisable: its id is
    ``PackageFamilyName!ApplicationId``, and the ``!`` is what tells them apart.
    Desktop entries are dropped because the Start-menu scan already has them, with
    a path that can be launched directly.
    """
    label = name.strip()
    aumid = app_id.strip()
    if not label or "!" not in aumid:
        return None
    if aumid.lower().startswith(STORE_PREFIX.lower()):
        aumid = aumid[len(STORE_PREFIX) :]
    return IndexedApp(
        name=label,
        target=f"{STORE_PREFIX}{aumid}",
        source=IndexSource.UWP,
        aumid=aumid,
    )


def link_catalog(apps: Iterable[IndexedApp], entries: Sequence[AppEntry]) -> list[IndexedApp]:
    """Attach a catalogue id to every record the dictionary recognises.

    By AUMID first, then by executable, then by an exact folded name. Nothing
    fuzzy: the scan runs unattended, and a wrong link would quietly point «открой
    хром» at whatever the guess landed on.
    """
    by_uwp = {entry.uwp.lower(): entry.id for entry in entries if entry.uwp}
    by_exe: dict[str, str] = {}
    by_name: dict[str, str] = {}
    for entry in entries:
        for executable in entry.executables:
            by_exe.setdefault(Path(executable).stem.lower(), entry.id)
        by_name.setdefault(normalize_text(entry.name), entry.id)
        for alias in entry.aliases:
            by_name.setdefault(normalize_text(alias), entry.id)

    linked: list[IndexedApp] = []
    for app in apps:
        if app.catalog_id:
            linked.append(app)
            continue
        stem = Path(app.executable or app.target).stem.lower()
        catalog_id = (
            by_uwp.get(app.aumid.lower(), "")
            or by_exe.get(stem, "")
            or by_name.get(normalize_text(app.name), "")
        )
        linked.append(replace(app, catalog_id=catalog_id) if catalog_id else app)
    return linked


def dedupe(apps: Iterable[IndexedApp], *, limit: int = MAX_INDEXED_APPS) -> tuple[IndexedApp, ...]:
    """Collapse records that describe the same target, keeping the best source.

    Also collapses two records of the same catalogued program — Chrome reached
    through the Start menu and through ``App Paths`` is one entry in a list the
    user is offered, and it should be the one with the better label.
    """
    by_target: dict[str, IndexedApp] = {}
    for app in apps:
        key = app.dedupe_key
        if not key:
            continue
        current = by_target.get(key)
        if current is None or app.source.weight > current.source.weight:
            by_target[key] = app

    by_program: dict[str, IndexedApp] = {}
    for app in by_target.values():
        if not app.catalog_id:
            by_program[f"target:{app.dedupe_key}"] = app
            continue
        current = by_program.get(app.catalog_id)
        if current is None or app.source.weight > current.source.weight:
            by_program[app.catalog_id] = app

    result = sorted(by_program.values(), key=lambda app: (-app.source.weight, app.name.lower()))
    if len(result) > limit:
        _log.warning("индекс приложений урезан до %d записей", limit)
        del result[limit:]
    return tuple(result)


def _as_int(value: object) -> int:
    """Registry values arrive as ``int`` or as text. Neither may raise here."""
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _is_uninstaller(label: str) -> bool:
    """Whether a shortcut label names an uninstaller rather than a program."""
    folded = label.casefold()
    return any(word in folded for word in ("uninstall", "удалить", "деинсталл", "remove "))


# --------------------------------------------------------------------------- #
# Scanners. Windows-only, guarded, and each one yields into the parsers above.
# --------------------------------------------------------------------------- #


def scan_installed(*, limit: int = MAX_INDEXED_APPS) -> list[IndexedApp]:
    """Everything this machine can launch, from all five sources.

    Returns an empty list off Windows: every caller runs on both platforms and a
    scanner that raised would push the guard into each of them. Takes seconds —
    call it from :class:`AppIndex`, which owns the thread it runs on.
    """
    if sys.platform != "win32":
        return []
    started = time.monotonic()
    found: list[IndexedApp] = []
    for scanner in (_scan_app_paths, _scan_uninstall, _scan_start_menu, _scan_path, _scan_uwp):
        try:
            found.extend(scanner())
        except OSError as exc:
            _log.warning("источник %s не прочитан: %s", scanner.__name__, exc)
    entries = _catalog_entries()
    apps = dedupe(link_catalog(found, entries), limit=limit)
    _log.info(
        "индекс приложений собран: %d записей из %d найденных за %.1f с",
        len(apps),
        len(found),
        time.monotonic() - started,
    )
    return list(apps)


def _catalog_entries() -> tuple[AppEntry, ...]:
    """The shipped dictionary, or nothing when it is missing or broken.

    A packaging problem must not cost the user the whole index: without the
    dictionary the scan still finds programs, they just lose their Russian
    aliases.
    """
    try:
        return load_apps().apps
    except AppsError as exc:
        _log.warning("словарь приложений недоступен, индекс без псевдонимов: %s", exc)
        return ()


def _scan_app_paths() -> Iterator[IndexedApp]:
    """``SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths`` in both hives."""
    import winreg

    subkey = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for name, values in _iter_registry(winreg, hive, subkey):
            app = parse_app_paths_entry(name, values)
            if app is not None:
                yield app


def _scan_uninstall() -> Iterator[IndexedApp]:
    """Both uninstall keys, in both registry views."""
    import winreg

    subkey = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for _name, values in _iter_registry(winreg, hive, subkey):
            app = parse_uninstall_entry(values)
            if app is not None:
                yield app


def _scan_start_menu() -> Iterator[IndexedApp]:
    """Every ``.lnk`` under the machine's and the user's Start menu."""
    for root in _start_menu_dirs():
        try:
            shortcuts = sorted(root.rglob("*.lnk"))
        except OSError as exc:
            _log.debug("меню «Пуск» %s не прочитано: %s", root, exc)
            continue
        for path in shortcuts:
            app = parse_shortcut(path)
            if app is not None:
                yield app


def _scan_path() -> Iterator[IndexedApp]:
    """Executables in the directories of ``PATH``, capped per directory."""
    seen: set[str] = set()
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        directory = raw.strip().strip('"')
        if not directory:
            continue
        folded = directory.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        root = Path(directory)
        try:
            children = sorted(root.iterdir())
        except OSError:
            continue
        taken = 0
        for path in children:
            app = parse_path_executable(path)
            if app is None:
                continue
            yield app
            taken += 1
            if taken >= MAX_PATH_EXECUTABLES:
                _log.debug("каталог PATH %s урезан до %d файлов", root, taken)
                break


def _scan_uwp() -> Iterator[IndexedApp]:
    """Store applications, through the ``shell:AppsFolder`` namespace.

    The only way to enumerate them: a Store app has no path on disk to find and
    no registry key that names it in a stable place. COM is initialised for this
    thread explicitly — the scan runs on a background thread of ours, where
    nothing has done it yet — and any failure downgrades to a warning, because a
    machine without Store apps is perfectly normal and so is one where the shell
    refuses to answer.
    """
    try:
        import comtypes
        import comtypes.client
    except ImportError as exc:  # pragma: no cover - comtypes ships on Windows
        _log.debug("перечисление приложений Store недоступно: %s", exc)
        return
    try:
        comtypes.CoInitialize()
    except OSError as exc:  # pragma: no cover - depends on the thread's COM state
        _log.debug("COM не инициализирован: %s", exc)
        return
    try:
        shell = comtypes.client.CreateObject("Shell.Application")
        folder = shell.NameSpace(STORE_FOLDER)
        if folder is None:
            return
        for item in folder.Items():
            app = parse_uwp_entry(str(item.Name), str(item.Path))
            if app is not None:
                yield app
    except (OSError, AttributeError, ValueError) as exc:
        _log.warning("приложения Store не перечислены: %s", exc)
    finally:
        comtypes.CoUninitialize()


def _iter_registry(
    winreg: ModuleType,
    hive: int,
    subkey: str,
) -> Iterator[tuple[str, dict[str, object]]]:
    """Subkeys of one registry key with their values, from both WOW64 views.

    A program installed as 32-bit is invisible to the 64-bit view and users have
    no idea which they have, so both are walked. A missing key is normal —
    ``HKEY_CURRENT_USER`` often has neither of these — and ends the iteration.
    """
    for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
        access = winreg.KEY_READ | view
        try:
            root = winreg.OpenKey(hive, subkey, 0, access)
        except OSError:
            continue
        with root:
            index = 0
            while True:
                try:
                    name = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                try:
                    with winreg.OpenKey(root, name, 0, access) as child:
                        yield name, _read_values(winreg, child)
                except OSError as exc:
                    _log.debug("пропущен раздел реестра %s\\%s: %s", subkey, name, exc)


def _read_values(winreg: ModuleType, key: object) -> dict[str, object]:
    """Every value of one registry key as a dict. The default value is ``""``."""
    values: dict[str, object] = {}
    index = 0
    while True:
        try:
            name, value, _kind = winreg.EnumValue(key, index)
        except OSError:
            break
        values[name] = value
        index += 1
    return values


def _start_menu_dirs() -> tuple[Path, ...]:
    """Start-menu folders to walk: the machine's and the current user's."""
    candidates = (
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    )
    return tuple(path for path in candidates if path.parts and path.is_dir())


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #


def phrase_key(text: str) -> str:
    """Fold a spoken phrase to the shape aliases are stored in.

    Normalisation plus the leading verb or preposition: «Открой Хром!», «в хроме»
    and «хром» have to score the same, or a phrase that came out of a slot at the
    end of a template would never reach an alias.
    """
    words = normalize_text(text).split()
    index = 0
    while index < len(words) - 1 and words[index] in _LEADING_WORDS:
        index += 1
    return " ".join(words[index:])


@dataclass(frozen=True, slots=True)
class AppCandidate:
    """One answer to «which program did they mean», ranked against the others."""

    app_id: str
    name: str
    confidence: float
    app: IndexedApp | None = None
    entry: AppEntry | None = None
    launches: int = 0

    @property
    def installed(self) -> bool:
        """Whether a scan actually found this program on the machine."""
        return self.app is not None

    @property
    def source_weight(self) -> int:
        """Trust of the source this candidate was found in, ``0`` when unfound."""
        return self.app.source.weight if self.app is not None else 0

    @property
    def target(self) -> str:
        """What to launch: the scanned target, or the dictionary's own guess.

        A dictionary entry falls back to a bare executable name — ``calc.exe`` —
        which the shell resolves through ``App Paths`` itself. That is what makes
        «открой калькулятор» work on a machine whose Start menu the scan could
        not read.
        """
        if self.app is not None:
            return self.app.target
        if self.entry is None:
            return ""
        if self.entry.uwp:
            return f"{STORE_PREFIX}{self.entry.uwp}"
        return self.entry.primary_executable

    @property
    def rank(self) -> tuple[float, int, int]:
        """Sort key: how sure, how often used, how trusted the source."""
        return (self.confidence, self.launches, self.source_weight)


@dataclass(frozen=True, slots=True)
class AppIndexResolver:
    """Turns what was said into one program, or into a question.

    Built by :meth:`AppIndex.resolver` from a snapshot and the shipped dictionary.
    Immutable and free of I/O, so every rule below — the generic aliases, the
    launch-frequency tie-break, the ambiguity threshold — is tested on any
    platform.
    """

    catalog: AppResolver
    apps: Mapping[str, IndexedApp] = field(default_factory=dict)
    kinds: Mapping[str, str] = field(default_factory=dict)
    launches: Mapping[str, int] = field(default_factory=dict)
    threshold: float = DEFAULT_ALIAS_THRESHOLD

    @classmethod
    def build(
        cls,
        snapshot: AppIndexSnapshot,
        *,
        entries: Iterable[AppEntry] = (),
        threshold: float = DEFAULT_ALIAS_THRESHOLD,
        user_aliases: Mapping[str, str] | None = None,
    ) -> Self:
        """Index a snapshot together with the dictionary that names its programs."""
        catalog_entries = tuple(entries)
        catalog = AppResolver.from_apps(catalog_entries, threshold=threshold)
        catalog = catalog.with_installed(app.as_installed() for app in snapshot.apps)
        if user_aliases:
            catalog = catalog.with_user_aliases(user_aliases)
        apps: dict[str, IndexedApp] = {}
        for app in snapshot.apps:
            current = apps.get(app.key)
            if current is None or app.source.weight > current.source.weight:
                apps[app.key] = app
        kinds = {entry.id: entry.kind for entry in catalog_entries if entry.kind}
        return cls(
            catalog=catalog,
            apps=apps,
            kinds=kinds,
            launches=dict(snapshot.launches),
            threshold=threshold,
        )

    def resolve(self, phrase: str) -> AppCandidate:
        """The program a phrase names.

        Raises:
            AppNotFound: nothing is close enough to what was said.
            AppAmbiguous: two programs are equally close and the launch history
                does not separate them. Carries both names for the question.
        """
        key = phrase_key(phrase)
        if not key:
            raise AppNotFound(
                "empty application phrase",
                user_message="Не поняла, какое приложение открыть.",
            )
        # The user said this phrase means this program. Asking anyway — or reading
        # it as a category word — would make the setting look ignored: «почта»
        # ships pointing at Outlook and «браузер» at whatever is installed, and
        # someone who bound either of them has already answered the question the
        # two branches below would otherwise raise.
        mine = key in self.catalog.user_aliases
        if not mine:
            kind = GENERIC_ALIASES.get(key)
            if kind is not None:
                return self._resolve_kind(kind, phrase)

        match = self.catalog.resolve(phrase)
        if match is None:
            raise AppNotFound(
                f"no application matches {phrase!r}",
                user_message=f"Не нашла приложение «{phrase.strip()}».",
            )
        if mine:
            return self._candidate(match.app_id, match.name, match.confidence)
        rivals = self._rivals(key, match.app_id)
        if rivals:
            options = tuple(self._display_name(app_id) for app_id in (match.app_id, *rivals))
            raise AppAmbiguous(
                f"{phrase!r} matches {len(options)} applications: {', '.join(options)}",
                options=options,
                user_message=f"Какое приложение открыть: {_enumerate_ru(options)}?",
            )
        return self._candidate(match.app_id, match.name, match.confidence)

    def candidates(self, phrase: str) -> list[AppCandidate]:
        """Everything that could be meant, best first. For the settings window."""
        key = phrase_key(phrase)
        if not key:
            return []
        if key not in self.catalog.user_aliases:
            kind = GENERIC_ALIASES.get(key)
            if kind is not None:
                return self._of_kind(kind)
        scored = self._scores(key)
        found = [self._candidate(app_id, "", score) for app_id, score in scored.items()]
        return sorted(found, key=lambda item: item.rank, reverse=True)

    def app(self, app_id: str) -> IndexedApp | None:
        """The indexed record for an id, or ``None`` when nothing was found."""
        return self.apps.get(app_id)

    def _resolve_kind(self, kind: str, phrase: str) -> AppCandidate:
        """«браузер» — whatever browser this machine has, or a question."""
        found = self._of_kind(kind)
        if not found:
            raise AppNotFound(
                f"nothing installed of kind {kind!r}",
                user_message=f"Не нашла ни одного приложения: «{phrase.strip()}».",
            )
        if len(found) > 1 and found[0].launches == found[1].launches:
            options = tuple(item.name for item in found[:3])
            raise AppAmbiguous(
                f"{len(found)} applications of kind {kind!r}, none used more than the others",
                options=options,
                user_message=f"Какое приложение открыть: {_enumerate_ru(options)}?",
            )
        return found[0]

    def _of_kind(self, kind: str) -> list[AppCandidate]:
        """Installed programs of one dictionary category, most-used first."""
        found = [
            self._candidate(app_id, "", 1.0)
            for app_id, entry_kind in self.kinds.items()
            if entry_kind == kind and app_id in self.apps
        ]
        return sorted(found, key=lambda item: item.rank, reverse=True)

    def _rivals(self, key: str, chosen: str) -> tuple[str, ...]:
        """Other programs close enough to the phrase to be worth asking about.

        The winner is not re-derived here — :meth:`ayris.nlu.apps.AppResolver.resolve`
        already decided, with rules this scan deliberately does not repeat. This
        only answers «was it a close call», and a rival is dropped when it has been
        launched less often than the winner: the user who opens Chrome every day
        should not be asked about Chromium.
        """
        scored = self._scores(key)
        best = scored.get(chosen, 1.0)
        chosen_launches = self.launches.get(chosen, 0)
        rivals = [
            app_id
            for app_id, score in scored.items()
            if app_id != chosen
            and score >= best - AMBIGUITY_GAP
            and self.launches.get(app_id, 0) >= chosen_launches
        ]
        rivals.sort(key=lambda app_id: (-scored[app_id], -self.launches.get(app_id, 0), app_id))
        return tuple(rivals[:2])

    def _scores(self, key: str) -> dict[str, float]:
        """Best similarity between the phrase and each program's aliases.

        A coarse sweep: exact match or edit distance, no stemming. It exists to
        spot a tie, so it has to be cheap and it has to treat every program the
        same way.
        """
        scored: dict[str, float] = {}
        for alias, app_id in self._aliases():
            if alias == key:
                score = 1.0
            elif len(alias) < _MIN_RIVAL_LENGTH or len(key) < _MIN_RIVAL_LENGTH:
                continue
            else:
                score = similarity(key, alias, floor=self.threshold)
                if score < self.threshold:
                    continue
            if score > scored.get(app_id, 0.0):
                scored[app_id] = score
        return scored

    def _aliases(self) -> Iterator[tuple[str, str]]:
        """Every alias known to the catalogue, the user's own first."""
        yield from self.catalog.user_aliases.items()
        yield from self.catalog.aliases.items()

    def _candidate(self, app_id: str, name: str, confidence: float) -> AppCandidate:
        """Assemble a candidate from everything known about one program id."""
        entry = self.catalog.entry(app_id)
        app = self.apps.get(app_id)
        display = name or self._display_name(app_id)
        return AppCandidate(
            app_id=app_id,
            name=display,
            confidence=confidence,
            app=app,
            entry=entry,
            launches=self.launches.get(app_id, 0),
        )

    def _display_name(self, app_id: str) -> str:
        """The name to say out loud: the dictionary's, the scan's, or the id."""
        entry = self.catalog.entry(app_id)
        if entry is not None:
            return entry.name
        app = self.apps.get(app_id)
        return app.name if app is not None else app_id


def _enumerate_ru(names: Sequence[str]) -> str:
    """«Chrome, Firefox или Opera» — a list a person can answer."""
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} или {names[-1]}"


# --------------------------------------------------------------------------- #
# The index itself: cache, TTL, background refresh
# --------------------------------------------------------------------------- #


class AppIndex:
    """The application index with its cache, its TTL and its refresh thread.

    One instance per process, reached through :func:`get_app_index`. Everything
    here is about *when* to scan; what a scan finds and what a phrase means are
    both elsewhere in this module, and both are pure.

    Args:
        cache_path: Where the JSON cache lives. Defaults to the profile's cache
            directory, resolved lazily so importing this module touches no disk.
        ttl_s: How long a scan stays fresh.
        scanner: What to call to scan. Injected by the tests, which have no
            registry to walk.
        entries: The shipped dictionary. Loaded on first use when omitted.
        threshold: Fuzzy tolerance for the resolver. Defaults to the «Команды»
            setting when omitted.
        clock: Wall-clock source, injected so a test can age a cache without
            waiting twelve hours.
    """

    def __init__(
        self,
        *,
        cache_path: Path | None = None,
        ttl_s: float = DEFAULT_TTL_S,
        scanner: Callable[[], list[IndexedApp]] | None = None,
        entries: Iterable[AppEntry] | None = None,
        threshold: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._cache_path = cache_path
        self._ttl_s = max(0.0, ttl_s)
        self._scanner = scanner if scanner is not None else scan_installed
        self._entries: tuple[AppEntry, ...] | None = None if entries is None else tuple(entries)
        self._threshold = threshold
        self._clock = clock
        self._lock = threading.RLock()
        self._snapshot: AppIndexSnapshot | None = None
        self._resolver: AppIndexResolver | None = None
        self._thread: threading.Thread | None = None
        self._loaded = False

    # -- state ------------------------------------------------------------- #

    @property
    def cache_path(self) -> Path:
        """Where the cache is kept."""
        if self._cache_path is None:
            self._cache_path = get_paths().cache_dir / INDEX_FILE_NAME
        return self._cache_path

    @property
    def snapshot(self) -> AppIndexSnapshot | None:
        """What the index knows right now, without loading or scanning anything."""
        with self._lock:
            return self._snapshot

    @property
    def refreshing(self) -> bool:
        """Whether a background scan is running."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def is_stale(self) -> bool:
        """Whether what we know is old enough to be worth rescanning."""
        with self._lock:
            snapshot = self._snapshot
        return snapshot is None or snapshot.is_stale(self._ttl_s, self._clock())

    # -- reading ----------------------------------------------------------- #

    def ensure_ready(self, *, wait: bool = False) -> AppIndexSnapshot:
        """The best snapshot available right now.

        Reads the cache once per process. A stale cache is returned as it is and a
        background refresh is started behind it — an index that is twelve hours
        old still knows where Chrome lives, and making the user wait for a
        registry walk to hear «Открываю Chrome» is the wrong trade. Only the first
        run ever, with no cache at all, scans synchronously.

        Args:
            wait: Scan synchronously even when a stale cache exists. What the
                «Обновить список» button and the tests use.
        """
        with self._lock:
            snapshot = self._snapshot
            if snapshot is None and not self._loaded:
                self._loaded = True
                snapshot = self._read_cache()
                if snapshot is not None:
                    self._install(snapshot)
        if snapshot is None:
            return self.refresh_now()
        if snapshot.is_stale(self._ttl_s, self._clock()):
            if wait:
                return self.refresh_now()
            self.refresh()
        return snapshot

    def resolver(self) -> AppIndexResolver:
        """A resolver over the current snapshot, rebuilt only when it changes."""
        snapshot = self.ensure_ready()
        with self._lock:
            if self._resolver is None:
                self._resolver = AppIndexResolver.build(
                    snapshot,
                    entries=self._catalog(),
                    threshold=self._fuzzy_threshold(),
                )
            return self._resolver

    def resolve(self, phrase: str) -> AppCandidate:
        """Shorthand for ``resolver().resolve(phrase)``, which is the common case."""
        return self.resolver().resolve(phrase)

    # -- writing ----------------------------------------------------------- #

    def refresh(self) -> bool:
        """Start a background scan. Returns ``False`` when one is already running.

        The thread is a daemon: a scan half-way through a registry walk must never
        be the reason the application does not exit.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            thread = threading.Thread(
                target=self._scan_and_install,
                name="ayris-app-index",
                daemon=True,
            )
            self._thread = thread
        thread.start()
        return True

    def refresh_now(self) -> AppIndexSnapshot:
        """Scan on this thread and return the result. Never call it from the UI."""
        return self._scan_and_install()

    def note_launch(self, app_id: str) -> int:
        """Remember that a program was started, and return its new count.

        The counter is what breaks a tie between two programs the user could have
        meant, so it has to survive a restart — it is written back into the cache
        file immediately. A launch is a rare event; the write is a few kilobytes.
        """
        if not app_id:
            return 0
        with self._lock:
            snapshot = self._snapshot or AppIndexSnapshot()
            launches = dict(snapshot.launches)
            count = launches.get(app_id, 0) + 1
            launches[app_id] = count
            self._install(snapshot.with_launches(launches))
            self._write_cache()
            return count

    def close(self) -> None:
        """Wait briefly for a running scan. Called from the application teardown."""
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    # -- internals --------------------------------------------------------- #

    def _scan_and_install(self) -> AppIndexSnapshot:
        """Run the scanner and publish what it found."""
        try:
            apps = tuple(self._scanner())
        except Exception:
            # A scan is best-effort by nature: five sources, any of which can be
            # broken on a given machine. Whatever went wrong, the previous
            # snapshot is still better than no index at all.
            _log.exception("сканирование приложений не удалось")
            with self._lock:
                return self._snapshot or AppIndexSnapshot(scanned_at=self._clock())
        with self._lock:
            previous = self._snapshot
            snapshot = AppIndexSnapshot(
                apps=apps,
                scanned_at=self._clock(),
                launches=dict(previous.launches) if previous is not None else {},
            )
            self._install(snapshot)
            self._write_cache()
            return snapshot

    def _install(self, snapshot: AppIndexSnapshot) -> None:
        """Publish a snapshot and drop the resolver built from the old one."""
        self._snapshot = snapshot
        self._resolver = None

    def _catalog(self) -> tuple[AppEntry, ...]:
        """The shipped dictionary, read once per index."""
        if self._entries is None:
            self._entries = _catalog_entries()
        return self._entries

    def _fuzzy_threshold(self) -> float:
        """Fuzzy tolerance: the constructor's, or the «Команды» setting.

        Imported where it is used rather than at module level: the settings
        machinery pulls in a good part of the core, and this module is imported by
        the action registry's discovery pass on every start.
        """
        if self._threshold is not None:
            return self._threshold
        from ayris.core.config import get_settings

        return get_settings().commands.fuzzy_threshold

    def _read_cache(self) -> AppIndexSnapshot | None:
        """Load the cache file, or ``None`` when there is nothing usable in it."""
        path = self.cache_path
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            _log.warning("кэш приложений повреждён, будет пересобран: %s", exc)
            return None
        if not isinstance(payload, dict):
            return None
        snapshot = AppIndexSnapshot.from_json(payload)
        if snapshot is None:
            _log.info("кэш приложений другой версии, будет пересобран")
            return None
        _log.debug("кэш приложений прочитан: %d записей", len(snapshot.apps))
        return snapshot

    def _write_cache(self) -> None:
        """Save the snapshot through a temporary file, so a crash cannot truncate it."""
        snapshot = self._snapshot
        if snapshot is None:
            return
        path = self.cache_path
        temporary = path.with_suffix(".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(snapshot.to_json(), ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            _log.warning("кэш приложений не сохранён: %s", exc)


_index: AppIndex | None = None
_index_lock: Final = threading.RLock()


def get_app_index() -> AppIndex:
    """The process-wide index, created on first use."""
    global _index
    with _index_lock:
        if _index is None:
            _index = AppIndex()
        return _index


def set_app_index(index: AppIndex | None) -> None:
    """Install an index, or drop the current one. Application wiring and tests."""
    global _index
    with _index_lock:
        if _index is not None and _index is not index:
            _index.close()
        _index = index
