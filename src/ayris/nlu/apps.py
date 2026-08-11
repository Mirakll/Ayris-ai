"""Turning «открой гугл хром» into something that can be launched.

The names people say and the names Windows uses have almost nothing in common.
A user says «хром», «гуглхром», «хроме» or, because the recogniser dropped a
vowel, «хрм»; Windows knows ``chrome.exe`` at a path that depends on how it was
installed. This module bridges the two, and the bridge is deliberately built in
two halves:

**The resolver is a pure function of a dictionary.** :class:`AppResolver` takes
aliases and answers questions — nothing here reads the registry, the Start menu
or the disk. That is what makes the interesting half testable everywhere: the
Russian cases, the fuzzy tolerance for a misheard alias, the priority of the
user's own vocabulary over the shipped one, all of it runs on Linux in CI and on
the developer's machine.

**The scanner is the half that has to touch Windows.** :func:`scan_installed_apps`
reads ``App Paths``, the uninstall keys and the Start menu, and it exists behind
a ``sys.platform`` guard so importing this module on any other system stays
harmless. Its output is a list of :class:`InstalledApp` records — data the
resolver merges in like any other source, which is why the merge is tested
without a scan ever running.

**Scanning is not something a match may wait for.** Walking the registry and the
Start menu takes seconds, and a slot is filled while the user waits for a
command to fire. So the scan belongs in the background, its result in the
``installed_apps`` table added by schema v4, and the resolver reads the table.
:meth:`AppResolver.with_installed` is how a scan — fresh or from cache — gets
folded into an index; the resolver itself never blocks.

**Aliases match by stem, not by string.** «в хроме», «хрома», «хромом» are the
same program, and listing every case of every alias would be a table nobody
maintains. So a lookup that fails exactly is retried against the longest alias
whose stem the phrase starts with, and only then against a fuzzy comparison.
Order matters: exact beats stem, stem beats fuzzy, and a user alias beats all
three, because «открой почту» meaning Thunderbird is a decision the user made
and the shipped dictionary does not get a vote.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ayris.core.errors import AyrisError
from ayris.core.paths import executable_dir
from ayris.nlu.matcher import similarity
from ayris.nlu.normalize import normalize_text
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from types import ModuleType

__all__ = [
    "APPS_FILE_NAME",
    "APPS_SCHEMA_VERSION",
    "DEFAULT_ALIAS_THRESHOLD",
    "MAX_SCANNED_APPS",
    "MIN_FUZZY_LENGTH",
    "MIN_STEM_RATIO",
    "AppEntry",
    "AppMatch",
    "AppResolver",
    "AppSource",
    "AppsError",
    "AppsFile",
    "InstalledApp",
    "apps_file",
    "load_apps",
    "scan_installed_apps",
]

_log = get_logger(__name__)

#: Format version of ``resources/nlu/apps.json``.
APPS_SCHEMA_VERSION: Final = 1

#: Name of the dictionary file inside ``resources/nlu``.
APPS_FILE_NAME: Final = "apps.json"

#: How close a misheard alias has to be to still resolve. Higher than the
#: matcher's default: an app name is short, and a single edit on «хром» would
#: otherwise reach «хлеб».
DEFAULT_ALIAS_THRESHOLD: Final = 0.75

#: The shortest alias a fuzzy comparison is attempted on. Below this a single
#: edit changes too large a share of the word to mean anything: «тг» and «дс»
#: only ever match exactly. Three is the floor because «хрм» — a recogniser that
#: swallowed a vowel — is exactly this case and has to reach «хром».
MIN_FUZZY_LENGTH: Final = 3

#: Prepositions dropped from the front of a phrase before it is looked up.
#: «в хроме», «с ютуба» — a slot that captured the object of a preposition keeps
#: it, and no alias list is going to enumerate «в хроме» separately.
_LEADING_PREPOSITIONS: Final = frozenset({"в", "во", "на", "из", "с", "со", "к", "ко", "у"})

#: Verbs that introduce a program name. Stripped for the same reason: a template
#: whose slot is the whole phrase hands over «открой хром», not «хром».
_LEADING_VERBS: Final = frozenset(
    {"открой", "открыть", "запусти", "запустить", "включи", "включить", "покажи"}
)

#: How much of an alias has to be present for a stem match. «хром» inside
#: «хроме» is four of five letters; «про» inside «проводник» is three of nine,
#: and matching that would make every phrase starting with «про» a launch.
MIN_STEM_RATIO: Final = 0.7

#: Ceiling on a single scan. A Start menu with thousands of shortcuts is a broken
#: machine, not a use case, and the resolver walks this list on every fuzzy miss.
MAX_SCANNED_APPS: Final = 2000

#: Slug rules for ``id``: it is stored in ``app_aliases.app_id`` and named in
#: command templates, so no spaces and no case games.
_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._+-]*$"


class AppsError(AyrisError):
    """The application dictionary is missing or malformed."""

    default_user_message = "Словарь приложений повреждён."


class AppSource(BaseModel):
    """Marker base so the two record types share a config. Not instantiated."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AppEntry(AppSource):
    """One program in the shipped dictionary.

    ``executables`` is a list because the same program ships under different
    names — ``obs64.exe`` and ``obs32.exe`` — and the first one found on disk
    wins. ``uwp`` carries the AUMID for a Store application, which has no
    executable path to launch at all.
    """

    id: str = Field(pattern=_ID_PATTERN, min_length=2, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="", max_length=32)
    executables: tuple[str, ...] = ()
    uwp: str = Field(default="", max_length=200)
    aliases: tuple[str, ...] = ()

    @field_validator("aliases", "executables", mode="after")
    @classmethod
    def _no_blanks(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned = tuple(item.strip() for item in value if item.strip())
        if len(cleaned) != len(value):
            raise ValueError("пустые строки в списке")
        return cleaned

    @property
    def primary_executable(self) -> str:
        """The executable to try first, or ``""`` for a Store application."""
        return self.executables[0] if self.executables else ""


class AppsFile(AppSource):
    """The dictionary file as it sits on disk."""

    schema_version: int = Field(default=APPS_SCHEMA_VERSION, ge=1)
    apps: tuple[AppEntry, ...] = ()

    @field_validator("apps", mode="after")
    @classmethod
    def _unique_ids(cls, value: tuple[AppEntry, ...]) -> tuple[AppEntry, ...]:
        counts = Counter(entry.id for entry in value)
        duplicates = sorted(app_id for app_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"повторяющиеся id: {', '.join(duplicates)}")
        return value


@dataclass(frozen=True, slots=True)
class InstalledApp:
    """A program the scanner found, or the cache remembers finding.

    ``target`` is what a launcher is handed: a path to an executable, a path to
    a ``.lnk``, or a ``shell:AppsFolder\\…`` moniker for a Store application.
    ``catalog_id`` links the record to an :class:`AppEntry` when the scan
    recognised one, which is what lets «открой хром» reach a Chrome that lives
    somewhere the dictionary never guessed.
    """

    name: str
    target: str
    executable: str = ""
    arguments: str = ""
    source: str = ""
    catalog_id: str = ""


@dataclass(frozen=True, slots=True)
class AppMatch:
    """The program a phrase named, and how sure the resolver is.

    ``confidence`` is 1.0 for an exact alias, a little less for a stem match and
    the similarity ratio itself for a fuzzy one, so a caller can refuse a guess
    it does not like. ``installed`` is ``None`` when the dictionary knows the
    program but no scan has confirmed it is on this machine — enough to answer
    «what did the user mean», not enough to launch.
    """

    app_id: str
    name: str
    alias: str
    confidence: float
    entry: AppEntry | None = None
    installed: InstalledApp | None = None

    @property
    def target(self) -> str:
        """What to launch: the scanned target, or the dictionary's executable."""
        if self.installed is not None:
            return self.installed.target
        if self.entry is None:
            return ""
        return self.entry.uwp or self.entry.primary_executable


def apps_file(directory: Path | None = None) -> Path:
    """Path of the dictionary file.

    ``resources`` ships beside the executable and sits at the repository root in
    development; :func:`~ayris.core.paths.executable_dir` already resolves that.
    """
    base = directory if directory is not None else executable_dir() / "resources" / "nlu"
    return base / APPS_FILE_NAME


def load_apps(path: Path | None = None) -> AppsFile:
    """Read and validate the shipped dictionary.

    Raises:
        AppsError: The file is unreadable, is not JSON, or a record is invalid.
            A dictionary is data that ships with the application, so a problem
            here is a packaging mistake and should say so in one Russian
            sentence rather than fail later as a command that does nothing.
    """
    target = path if path is not None else apps_file()
    try:
        raw = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise AppsError(
            f"cannot read app dictionary {target}: {exc}",
            user_message=f"Не удалось прочитать словарь приложений:\n{target}",
        ) from exc

    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise AppsError(
            f"app dictionary {target} is not valid JSON: {exc}",
            user_message=f"Словарь приложений повреждён:\n{target}\nОшибка разбора JSON: {exc}",
        ) from exc

    try:
        parsed = AppsFile.model_validate(payload)
    except ValidationError as exc:
        raise AppsError(
            f"app dictionary {target} failed validation: {exc.error_count()} problem(s)",
            user_message=f"Словарь приложений заполнен неверно:\n{target}",
        ) from exc

    if parsed.schema_version > APPS_SCHEMA_VERSION:
        raise AppsError(
            f"app dictionary {target} is v{parsed.schema_version}, "
            f"newer than supported v{APPS_SCHEMA_VERSION}",
            user_message="Словарь приложений создан более новой версией Ayris.",
        )
    return parsed


def _alias_key(text: str) -> str:
    """Fold an alias the same way a spoken phrase is folded.

    Through :func:`~ayris.nlu.normalize.normalize_text` rather than ``lower()``
    so «Гугл Хром!» and «гугл хром» collapse to one key, and so a difference
    that survives is a real one — the same guarantee the matcher relies on.
    """
    return normalize_text(text)


def _executable_key(name: str) -> str:
    """The bare name of an executable, lowercase and without its suffix."""
    return Path(name).stem.lower()


def _strip_lead(key: str) -> str:
    """Drop leading verbs and prepositions from an already folded phrase.

    «в хроме» and «открой хром» are what a slot actually hands over, because a
    template whose slot sits at the end captures everything after the verb. The
    last word is never dropped: «в» alone is not a phrase, and a program called
    «Из» would otherwise be unreachable.
    """
    words = key.split()
    index = 0
    while index < len(words) - 1 and words[index] in _LEADING_PREPOSITIONS | _LEADING_VERBS:
        index += 1
    return " ".join(words[index:])


@dataclass(frozen=True, slots=True)
class AppResolver:
    """Answers «which program is this» from a dictionary, without touching disk.

    Build it with :meth:`from_apps` and layer on what the machine knows with
    :meth:`with_installed` and :meth:`with_user_aliases`. Every layer returns a
    new resolver: a background scan finishing must not change the answer a match
    already in flight is computing, and an immutable resolver makes that
    impossible rather than unlikely.
    """

    entries: tuple[AppEntry, ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    user_aliases: Mapping[str, str] = field(default_factory=dict)
    installed: Mapping[str, InstalledApp] = field(default_factory=dict)
    threshold: float = DEFAULT_ALIAS_THRESHOLD

    @classmethod
    def from_apps(
        cls,
        apps: Iterable[AppEntry],
        *,
        threshold: float = DEFAULT_ALIAS_THRESHOLD,
    ) -> Self:
        """Index a dictionary by every alias, name and executable stem it names.

        The name and the executable are indexed alongside the aliases because
        both are things users say: «открой visual studio code» is the name, and
        «запусти notepad» is the executable. An alias already claimed by an
        earlier entry is left with its first owner, so the order in the file
        decides ties — «опера» reaching Opera rather than a later program that
        listed it as an afterthought.
        """
        entries = tuple(apps)
        aliases: dict[str, str] = {}
        for entry in entries:
            spellings = [entry.name, *entry.aliases, *map(_executable_key, entry.executables)]
            for spelling in spellings:
                key = _alias_key(spelling)
                if key and key not in aliases:
                    aliases[key] = entry.id
        return cls(entries=entries, aliases=aliases, threshold=threshold)

    def with_installed(self, apps: Iterable[InstalledApp]) -> Self:
        """A resolver that also knows what a scan found on this machine.

        A record carrying a ``catalog_id`` attaches to that dictionary entry;
        one without gets indexed under its own name, which is how a program the
        dictionary has never heard of still answers to «открой …». Dictionary
        aliases keep priority: a stray Start-menu shortcut called «Хром» must not
        outrank the real Chrome.
        """
        found = dict(self.installed)
        aliases = dict(self.aliases)
        known = {entry.id for entry in self.entries}
        for app in apps:
            app_id = app.catalog_id if app.catalog_id in known else _alias_key(app.name)
            if not app_id:
                continue
            found.setdefault(app_id, app)
            for spelling in (app.name, _executable_key(app.executable or app.target)):
                key = _alias_key(spelling)
                if key and key not in aliases:
                    aliases[key] = app_id
        return type(self)(
            entries=self.entries,
            aliases=aliases,
            user_aliases=self.user_aliases,
            installed=found,
            threshold=self.threshold,
        )

    def with_user_aliases(self, aliases: Mapping[str, str]) -> Self:
        """A resolver that prefers the user's own names for programs.

        Kept in their own mapping rather than merged, because they are consulted
        first: «открой почту» pointing at Thunderbird is a decision the shipped
        dictionary does not get to override on the next update.
        """
        folded = {_alias_key(alias): app_id for alias, app_id in aliases.items() if alias.strip()}
        return type(self)(
            entries=self.entries,
            aliases=self.aliases,
            user_aliases=folded,
            installed=self.installed,
            threshold=self.threshold,
        )

    def resolve(self, text: str) -> AppMatch | None:
        """The program a phrase names, or ``None`` when nothing is close enough.

        Three passes in order of trust. An exact alias scores 1.0. A stem match —
        the phrase is «хроме» and the alias is «хром» — scores by how much of the
        longer word the shared prefix covers, so a near-complete word scores near
        1.0 and a bare prefix does not qualify at all. A fuzzy match scores its
        similarity ratio. User aliases are checked exactly before any of it.
        """
        key = _strip_lead(_alias_key(text))
        if not key:
            return None
        user_id = self.user_aliases.get(key)
        if user_id is not None:
            return self._build(user_id, key, 1.0)
        app_id = self.aliases.get(key)
        if app_id is not None:
            return self._build(app_id, key, 1.0)
        return self._resolve_stem(key) or self._resolve_fuzzy(key)

    def resolve_prefix(self, text: str) -> tuple[AppMatch, str] | None:
        """The program a phrase *starts* with, and the words left over.

        For «открой хром и найди погоду», where the app slot has to stop
        somewhere. Longest first: «яндекс браузер» must win over «яндекс», or the
        slot swallows one word and leaves «браузер» as a stray tail.
        """
        words = _strip_lead(_alias_key(text)).split()
        for size in range(len(words), 0, -1):
            match = self.resolve(" ".join(words[:size]))
            if match is not None:
                return match, " ".join(words[size:])
        return None

    def entry(self, app_id: str) -> AppEntry | None:
        """Dictionary record for an id, or ``None`` for a scan-only program."""
        return next((item for item in self.entries if item.id == app_id), None)

    def _resolve_stem(self, key: str) -> AppMatch | None:
        """Best alias that shares a long enough prefix with the phrase.

        This is what handles Russian cases without a table of them: «хроме» and
        «хром» share four letters of five. The ratio is measured against the
        longer of the two, so a short alias cannot claim a long phrase — «про» is
        not a licence to launch «проводник».
        """
        best: tuple[float, str, str] | None = None
        for alias, app_id in self._all_aliases():
            shared = _common_prefix(key, alias)
            if not shared:
                continue
            ratio = shared / max(len(key), len(alias))
            if ratio < MIN_STEM_RATIO:
                continue
            if best is None or ratio > best[0]:
                best = (ratio, alias, app_id)
        if best is None:
            return None
        ratio, alias, app_id = best
        return self._build(app_id, alias, round(ratio, 4))

    def _resolve_fuzzy(self, key: str) -> AppMatch | None:
        """Best alias within the edit-distance tolerance, for a misheard word.

        Short aliases are skipped: «тг» and «дс» are two letters, and one edit
        away from half the alphabet. Length is compared before the distance is
        computed, because an alias that cannot possibly score high enough is not
        worth the comparison.
        """
        if len(key) < MIN_FUZZY_LENGTH:
            return None
        best: tuple[float, str, str] | None = None
        for alias, app_id in self._all_aliases():
            if len(alias) < MIN_FUZZY_LENGTH:
                continue
            score = similarity(key, alias, floor=self.threshold)
            if score < self.threshold:
                continue
            if best is None or score > best[0]:
                best = (score, alias, app_id)
        if best is None:
            return None
        score, alias, app_id = best
        return self._build(app_id, alias, round(score, 4))

    def _all_aliases(self) -> Iterator[tuple[str, str]]:
        """Every alias known, the user's first so ties resolve in their favour."""
        yield from self.user_aliases.items()
        yield from self.aliases.items()

    def _build(self, app_id: str, alias: str, confidence: float) -> AppMatch:
        """Assemble a match, filling in whatever the dictionary and scan know."""
        entry = self.entry(app_id)
        installed = self.installed.get(app_id)
        name = entry.name if entry is not None else (installed.name if installed else app_id)
        return AppMatch(
            app_id=app_id,
            name=name,
            alias=alias,
            confidence=confidence,
            entry=entry,
            installed=installed,
        )


def _common_prefix(left: str, right: str) -> int:
    """How many leading characters two strings share."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _match_catalog(entries: Iterable[AppEntry], executable: str, name: str) -> str:
    """Which dictionary entry a scanned program is, or ``""`` for none.

    By executable first, because ``chrome.exe`` is the same program wherever it
    was installed from, and only then by an exact folded name. Nothing fuzzy
    happens here: the scan runs unattended, and a wrong link would quietly point
    «открой хром» at whatever the guess landed on.
    """
    exe_key = _executable_key(executable)
    if exe_key:
        for entry in entries:
            if any(exe_key == _executable_key(item) for item in entry.executables):
                return entry.id
    name_key = _alias_key(name)
    if name_key:
        for entry in entries:
            if name_key == _alias_key(entry.name) or name_key in map(_alias_key, entry.aliases):
                return entry.id
    return ""


def _shortcut_targets(root: Path) -> Iterator[tuple[str, Path]]:
    """Every ``.lnk`` under a Start-menu folder, as ``(label, path)``.

    The file name is the label because that is what the Start menu shows and
    therefore what the user reads out loud. Where the shortcut points is not read
    here — resolving a ``.lnk`` needs COM, and a path good enough to launch is
    already in the shortcut itself.
    """
    try:
        found = sorted(root.rglob("*.lnk"))
    except OSError as exc:
        _log.debug("не удалось прочитать меню «Пуск» %s: %s", root, exc)
        return
    for path in found:
        yield path.stem, path


def scan_installed_apps(*, limit: int = MAX_SCANNED_APPS) -> list[InstalledApp]:
    """Everything installed on this machine that could plausibly be launched.

    Three sources, cheapest first: the ``App Paths`` registry key, which is the
    closest thing Windows has to a list of launchable executables; the uninstall
    keys, which know display names the registry key does not; and the Start menu,
    which catches everything the other two miss. Records are deduplicated by
    target, first source winning.

    Returns an empty list off Windows — every caller runs on both, and a scanner
    that raises would push the guard into each of them. The work belongs on a
    background thread: this walks the registry and a directory tree and takes
    seconds, while a slot is filled with the user waiting.
    """
    if sys.platform != "win32":
        return []
    entries = load_apps().apps if apps_file().exists() else ()
    found: dict[str, InstalledApp] = {}
    for name, target, executable, source in _scan_windows_sources():
        key = target.lower()
        if not key or key in found:
            continue
        found[key] = InstalledApp(
            name=name,
            target=target,
            executable=executable,
            source=source,
            catalog_id=_match_catalog(entries, executable, name),
        )
        if len(found) >= limit:
            _log.warning("сканирование остановлено на пределе в %d программ", limit)
            break
    _log.info("найдено установленных программ: %d", len(found))
    return list(found.values())


def _scan_windows_sources() -> Iterator[tuple[str, str, str, str]]:
    """Raw ``(name, target, executable, source)`` rows from every Windows source.

    Separate from :func:`scan_installed_apps` so deduplication, catalogue linking
    and the limit are written once and this stays a plain enumeration. Import of
    ``winreg`` is local: the module is missing everywhere else, and this file is
    imported on Linux by every test that touches the resolver.
    """
    import winreg

    app_paths = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for name, values in _iter_registry(winreg, hive, app_paths):
            target = str(values.get("", "")).strip('"')
            if target:
                yield Path(name).stem, target, name, "app-paths"

    uninstall = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for _key, values in _iter_registry(winreg, hive, uninstall):
            display = str(values.get("DisplayName", "")).strip()
            icon = str(values.get("DisplayIcon", "")).split(",")[0].strip('" ')
            if display and icon.lower().endswith(".exe"):
                yield display, icon, Path(icon).name, "uninstall"

    for root in _start_menu_dirs():
        for label, path in _shortcut_targets(root):
            yield label, str(path), "", "start-menu"


def _iter_registry(
    winreg: ModuleType,
    hive: int,
    subkey: str,
) -> Iterator[tuple[str, dict[str, object]]]:
    """Subkeys of a registry key, each with its values already read out.

    A missing key is normal — ``HKEY_CURRENT_USER`` often has neither of the two
    this module reads — so it ends the iteration instead of raising. Both 32- and
    64-bit views are consulted, because a program installed as one is invisible
    to the other and users do not know which they have.
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
