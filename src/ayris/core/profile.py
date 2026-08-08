"""Profiles: several sets of commands in one installation, switched live.

A "profile" here is a row in the ``profiles`` table plus everything that hangs
off it — folders, commands, triggers and profile-scope variables. All profiles
share one installation directory, so the database, the settings file and the
sounds folder are common ground; only the contents of the profile move when the
user switches. That is what makes switching cheap enough to do without a
restart, which is the whole point of section 8.1's profiles tab.

:class:`ProfileManager` owns the lifecycle: create, copy, rename, delete, switch,
back up, reset. The archive format used by :meth:`ProfileManager.export` and
:meth:`ProfileManager.import_bundle` lives in
:mod:`ayris.core.portable_profile`; this module only decides *when* to call it
and what to do around it — take a backup first, refuse to delete the last
profile, tell the rest of the application that the active profile moved.

Two notes on the shape of the API:

*Switching notifies twice.* Subscribers registered through
:meth:`ProfileManager.subscribe` are called synchronously, and a
:class:`ProfileSwitched` event goes onto the bus for everything that is only
loosely coupled. The subsystems that must reload — the matcher, the hotkey
listener, the command tree — take the first; the overlay and the tray take the
second.

*Settings do not follow the profile.* ``config.toml`` belongs to the
installation, so switching a profile changes commands and variables and leaves
the microphone, the theme and the wake word alone. Importing a bundle can
overwrite the settings, but only when the caller asks for it explicitly.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, TypeAlias

from ayris.core.database import Database, init_database
from ayris.core.errors import ProfileError
from ayris.core.events import Event
from ayris.core.models import Profile, VariableScope, utc_now
from ayris.core.paths import (
    POINTER_FILE_NAME,
    clear_configured_root,
    default_root,
    get_paths,
    init_paths,
    write_configured_root,
)
from ayris.core.portable_profile import (
    BundleManifest,
    BundlePreview,
    ConflictPolicy,
    ImportReport,
    export_bundle,
    import_bundle,
    preview_bundle,
)
from ayris.core.repositories import Repositories
from ayris.utils.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ayris.core.config import ConfigManager
    from ayris.core.events import EventBus, Unsubscribe
    from ayris.core.paths import AppPaths

_log = get_logger(__name__)

#: Subdirectory of the profile root holding automatic backups.
BACKUP_DIR_NAME: str = "backups"

#: How many automatic backups to keep. Older ones are pruned after each new one.
MAX_BACKUPS: int = 10

#: Name of the profile created on first run, mirrored from the lifecycle so this
#: module does not have to import it and create a cycle.
DEFAULT_PROFILE_NAME: str = "По умолчанию"

#: Longest profile name accepted. Long enough for a sentence, short enough that
#: it still fits a backup file name.
MAX_NAME_LENGTH: int = 64

#: Entries left behind when the profile root moves. The lock and the WAL
#: sidecars belong to the *running* installation, the caches and logs are
#: reproducible, and copying a log file that logging still holds open is exactly
#: the kind of thing that fails halfway on Windows.
_SKIPPED_ON_MOVE: frozenset[str] = frozenset({"ayris.lock", "cache", "logs"})

#: Characters Windows refuses in a file name, plus the ones that only cause
#: confusion in one.
_UNSAFE_IN_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


@dataclass(frozen=True, slots=True)
class ProfileSwitched(Event):
    """The active profile changed. Everything profile-scoped must reload."""

    profile: Profile
    previous: Profile | None = None


@dataclass(frozen=True, slots=True)
class ProfilesChanged(Event):
    """The set of profiles changed: one was created, copied, renamed or deleted."""

    profiles: tuple[Profile, ...]
    active: Profile | None = None


#: Called synchronously on every switch, with the profile that is now active.
ProfileListener: TypeAlias = "Callable[[Profile], None]"


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _clean_name(name: str) -> str:
    """Validate a profile name.

    Raises:
        ProfileError: The name is blank or longer than :data:`MAX_NAME_LENGTH`.
    """
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ProfileError(
            "profile name is empty",
            user_message="Введите название профиля.",
        )
    if len(cleaned) > MAX_NAME_LENGTH:
        raise ProfileError(
            f"profile name is {len(cleaned)} characters, limit is {MAX_NAME_LENGTH}",
            user_message=f"Название профиля не длиннее {MAX_NAME_LENGTH} символов.",
        )
    return cleaned


def _file_stem(name: str) -> str:
    """Turn a profile name into something Windows will accept as a file name."""
    stem = _UNSAFE_IN_FILENAME.sub("_", name).strip(" .")
    return stem or "profile"


def _copy_filter(directory: str, names: list[str]) -> set[str]:
    """``shutil.copytree`` ignore callback for :meth:`ProfileManager.change_root`."""
    skipped = {name for name in names if name in _SKIPPED_ON_MOVE}
    skipped.update(name for name in names if name.endswith((".db-wal", ".db-shm")))
    if skipped:
        _log.debug("не переносим из %s: %s", directory, ", ".join(sorted(skipped)))
    return skipped


# ----------------------------------------------------------------------
# manager
# ----------------------------------------------------------------------


class ProfileManager:
    """The profiles tab, without the widgets.

    Args:
        repositories: Storage the profiles live in.
        paths: Path set of the running installation. Defaults to the
            process-wide one.
        bus: Event bus to announce switches on. Optional: a headless caller or a
            test can do without.
        config: Settings manager, reloaded after an import that overwrites
            ``config.toml`` and re-pointed after the root moves.
        on_rebind: Called after :meth:`change_root` swapped the database, with
            the new paths and repositories. This is how the composition root
            replaces the handles it cached at startup.

    Thread safety: none. Everything here is called from the UI thread, and the
    operations are either instantaneous or explicitly modal (export, import,
    moving the profile root).
    """

    __slots__ = (
        "_active",
        "_bus",
        "_config",
        "_listeners",
        "_on_rebind",
        "_paths",
        "_repositories",
    )

    def __init__(
        self,
        repositories: Repositories,
        *,
        paths: AppPaths | None = None,
        bus: EventBus | None = None,
        config: ConfigManager | None = None,
        on_rebind: Callable[[AppPaths, Repositories], None] | None = None,
    ) -> None:
        self._repositories = repositories
        self._paths = paths if paths is not None else get_paths()
        self._bus = bus
        self._config = config
        self._on_rebind = on_rebind
        self._listeners: list[ProfileListener] = []
        self._active = self._resolve_active()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    @property
    def repositories(self) -> Repositories:
        """Current storage. Replaced by :meth:`change_root`, so read it, do not cache it."""
        return self._repositories

    @property
    def paths(self) -> AppPaths:
        """Current path set. Replaced by :meth:`change_root`."""
        return self._paths

    @property
    def active(self) -> Profile:
        """The profile everything profile-scoped currently reads and writes."""
        return self._active

    @property
    def backups_dir(self) -> Path:
        """Where automatic backups are written."""
        return self._paths.root / BACKUP_DIR_NAME

    def list_all(self) -> list[Profile]:
        """Every profile, by name. What the profiles list widget shows."""
        return self._repositories.profiles.list_all()

    def refresh(self) -> Profile:
        """Re-read the active profile from storage, e.g. after an external change."""
        self._active = self._resolve_active()
        return self._active

    def _resolve_active(self) -> Profile:
        """The active profile, creating the default one if the table is empty."""
        profile = self._repositories.profiles.active()
        if profile is None:
            existing = self._repositories.profiles.list_all()
            if existing:
                profile = existing[0]
                if profile.id is not None:
                    self._repositories.profiles.set_active(profile.id)
                profile = replace(profile, is_active=True)
            else:
                profile = self._repositories.profiles.create(DEFAULT_PROFILE_NAME, activate=True)
                _log.info("создан профиль «%s»", profile.name)
        return profile

    def _require_id(self, profile: Profile) -> int:
        if profile.id is None:
            raise ProfileError(
                f"profile {profile.name!r} has no id",
                user_message="Профиль не сохранён — операция невозможна.",
            )
        return profile.id

    def _reload(self, profile_id: int) -> Profile:
        stored = self._repositories.profiles.get(profile_id)
        if stored is None:
            raise ProfileError(
                f"profile {profile_id} disappeared",
                user_message="Профиль не найден — возможно, он удалён в другом окне.",
            )
        return stored

    # ------------------------------------------------------------------
    # notification
    # ------------------------------------------------------------------

    def subscribe(self, listener: ProfileListener) -> Unsubscribe:
        """Register ``listener``, called on every switch with the new profile.

        Returns:
            A callable that unsubscribes. Long-lived objects should keep it and
            call it on teardown, or they will be called after they are closed.
        """
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify_switch(self, profile: Profile, previous: Profile | None) -> None:
        """Tell everyone, in a way one broken subscriber cannot stop."""
        for listener in list(self._listeners):
            try:
                listener(profile)
            except Exception:
                _log.exception("подписчик профиля упал на «%s»", profile.name)
        if self._bus is not None:
            self._bus.publish(ProfileSwitched(profile=profile, previous=previous))

    def _notify_list_changed(self) -> None:
        if self._bus is not None:
            self._bus.publish(ProfilesChanged(profiles=tuple(self.list_all()), active=self._active))

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, name: str, *, activate: bool = False) -> Profile:
        """Create an empty profile.

        Raises:
            ProfileError: The name is unusable or already taken.
        """
        cleaned = _clean_name(name)
        self._reject_duplicate(cleaned)
        profile = self._repositories.profiles.create(cleaned, activate=activate)
        _log.info("создан профиль «%s»", cleaned)
        if activate:
            previous = self._active
            self._active = profile
            self._after_switch(previous)
        self._notify_list_changed()
        return profile

    def copy(self, source: Profile, name: str | None = None) -> Profile:
        """Duplicate a profile, contents and all.

        The copy is a snapshot: folders, commands with their triggers, and the
        profile's variables. Nothing links the two afterwards.

        Args:
            source: Profile to copy.
            name: Name for the copy. Defaults to ``«source» — копия``.

        Raises:
            ProfileError: The name is unusable or already taken.
        """
        source_id = self._require_id(source)
        proposed = name if name is not None else f"{source.name} — копия"
        cleaned = _clean_name(proposed)
        if name is None:
            cleaned = self._free_name(cleaned)
        else:
            self._reject_duplicate(cleaned)

        with self._repositories.database.transaction():
            created = self._repositories.profiles.create(cleaned)
            self._copy_contents(source_id, self._require_id(created))
        _log.info("профиль «%s» скопирован в «%s»", source.name, cleaned)
        self._notify_list_changed()
        return created

    def _copy_contents(self, source_id: int, target_id: int) -> None:
        """Deep-copy folders, commands, triggers and variables between profiles.

        Folders are rebuilt parents-first so a child never points at an id that
        does not exist yet; ``sorted`` on the source ids is enough for that,
        because a folder is always inserted after its parent.
        """
        repositories = self._repositories
        folder_map: dict[int, int] = {}
        for folder in sorted(
            repositories.folders.list_for_profile(source_id),
            key=lambda item: item.id or 0,
        ):
            if folder.id is None or folder.profile_id != source_id:
                continue
            parent = folder_map.get(folder.parent_id) if folder.parent_id is not None else None
            copied_folder = repositories.folders.create(
                replace(folder, id=None, profile_id=target_id, parent_id=parent)
            )
            if copied_folder.id is not None:
                folder_map[folder.id] = copied_folder.id

        for command in repositories.commands.list_for_profile(source_id):
            if command.id is None:
                continue
            triggers = repositories.triggers.list_for_command(command.id)
            copied_command = repositories.commands.create(
                replace(
                    command,
                    id=None,
                    profile_id=target_id,
                    folder_id=(
                        folder_map.get(command.folder_id) if command.folder_id is not None else None
                    ),
                    created_at=None,
                    updated_at=None,
                )
            )
            if copied_command.id is None:  # pragma: no cover - insert always yields an id
                continue
            for trigger in triggers:
                repositories.triggers.add(replace(trigger, id=None, command_id=copied_command.id))

        for variable in repositories.variables.list_all(
            scope=VariableScope.PROFILE, profile_id=source_id
        ):
            repositories.variables.set(
                variable.name,
                variable.value,
                scope=VariableScope.PROFILE,
                profile_id=target_id,
                var_type=variable.type,
                persistent=variable.persistent,
            )

    def rename(self, profile: Profile, name: str) -> Profile:
        """Rename a profile.

        Raises:
            ProfileError: The name is unusable or already taken by another profile.
        """
        profile_id = self._require_id(profile)
        cleaned = _clean_name(name)
        if cleaned == profile.name:
            return profile
        self._reject_duplicate(cleaned)
        self._repositories.profiles.rename(profile_id, cleaned)
        renamed = self._reload(profile_id)
        if self._active.id == profile_id:
            self._active = renamed
        _log.info("профиль «%s» переименован в «%s»", profile.name, cleaned)
        self._notify_list_changed()
        return renamed

    def delete(self, profile: Profile) -> Profile:
        """Delete a profile and everything in it.

        The last profile is protected: an installation without one has nowhere
        to put a command, and the next start would silently create a new empty
        default, which looks exactly like data loss.

        Returns:
            The profile that is active afterwards — the same one, unless the
            deleted profile *was* the active one.

        Raises:
            ProfileError: This is the last profile, or it has no id.
        """
        profile_id = self._require_id(profile)
        if self._repositories.profiles.count() <= 1:
            raise ProfileError(
                "refusing to delete the last profile",
                user_message="Нельзя удалить единственный профиль.\nСначала создайте другой.",
            )
        was_active = self._active.id == profile_id
        successor: Profile | None = None
        if was_active:
            successor = next(
                (item for item in self.list_all() if item.id != profile_id),
                None,
            )
            if successor is None:  # pragma: no cover - count() above rules this out
                raise ProfileError(
                    "no profile left to switch to",
                    user_message="Нельзя удалить единственный профиль.",
                )

        with self._repositories.database.transaction():
            if successor is not None:
                self._repositories.profiles.set_active(self._require_id(successor))
            self._repositories.profiles.delete(profile_id)

        _log.info("профиль «%s» удалён", profile.name)
        if successor is not None:
            previous = self._active
            self._active = self._reload(self._require_id(successor))
            self._after_switch(previous)
        self._notify_list_changed()
        return self._active

    def switch(self, profile: Profile) -> Profile:
        """Make ``profile`` active without restarting anything.

        Transient variables are dropped on the way out: they describe a run of
        the *previous* profile's macros and would otherwise leak into the new
        one under the same names.

        Raises:
            ProfileError: The profile has no id or no longer exists.
        """
        profile_id = self._require_id(profile)
        if profile_id == self._active.id:
            return self._active
        target = self._reload(profile_id)

        previous = self._active
        with self._repositories.database.transaction():
            self._repositories.variables.clear_transient()
            self._repositories.profiles.set_active(profile_id)
        self._active = replace(target, is_active=True)
        _log.info("активный профиль: «%s» (был «%s»)", target.name, previous.name)
        self._after_switch(previous)
        return self._active

    def _after_switch(self, previous: Profile) -> None:
        if previous.id == self._active.id:
            return
        self._notify_switch(self._active, previous)

    def _reject_duplicate(self, name: str) -> None:
        if self._repositories.profiles.get_by_name(name) is not None:
            raise ProfileError(
                f"profile {name!r} already exists",
                user_message=f"Профиль «{name}» уже существует.",
            )

    def _free_name(self, name: str) -> str:
        """First unused variation of ``name``. Used when the caller did not pick one."""
        taken = {item.name for item in self.list_all()}
        if name not in taken:
            return name
        for index in range(2, 1000):
            candidate = f"{name} ({index})"
            if candidate not in taken:
                return candidate
        return f"{name} ({utc_now():%Y%m%d%H%M%S})"

    # ------------------------------------------------------------------
    # export, import, backup
    # ------------------------------------------------------------------

    def export(
        self,
        destination: Path,
        *,
        profile: Profile | None = None,
        include_sounds: bool = True,
        include_settings: bool = True,
    ) -> BundleManifest:
        """Write a profile to a portable ``.zip``.

        Secrets never travel: the settings in the archive have every
        credential-bearing field removed, and API keys were never in the
        settings to begin with.

        Raises:
            ProfileError: The file could not be written.
        """
        settings = self._config.settings if include_settings and self._config is not None else None
        return export_bundle(
            self._repositories,
            destination,
            profile=profile if profile is not None else self._active,
            settings=settings,
            paths=self._paths,
            include_sounds=include_sounds,
        )

    def preview_import(self, archive: Path) -> BundlePreview:
        """Describe a bundle and the conflicts it would cause, changing nothing.

        Raises:
            ProfileError: The file is not a readable Ayris bundle.
        """
        return preview_bundle(
            archive,
            repositories=self._repositories,
            profile_id=self._active.id,
        )

    def import_bundle(
        self,
        archive: Path,
        *,
        profile: Profile | None = None,
        policy: ConflictPolicy = ConflictPolicy.RENAME,
        backup: bool = True,
        include_sounds: bool = True,
        apply_config: bool = False,
    ) -> ImportReport:
        """Apply a bundle to a profile, taking a backup first.

        The import itself is all-or-nothing; the backup exists for the case the
        import *succeeds* and the user then decides they preferred what they had.

        Args:
            archive: Bundle to read.
            profile: Target profile. Defaults to the active one.
            policy: What to do about names that already exist.
            backup: Export the target profile before touching it.
            include_sounds: Copy the bundle's sound files into the installation.
            apply_config: Overwrite ``config.toml`` from the bundle. Off by
                default — settings belong to the installation, not the profile.

        Raises:
            ProfileError: The bundle is invalid, or a file could not be written.
        """
        target = profile if profile is not None else self._active
        snapshot = self.backup(profile=target, reason="import") if backup else None
        report = import_bundle(
            archive,
            self._repositories,
            profile=target,
            policy=policy,
            paths=self._paths,
            include_sounds=include_sounds,
            apply_config=apply_config,
        )
        if report.config_applied and self._config is not None:
            self._config.reload()
        if target.id == self._active.id:
            self._notify_switch(self._active, None)
        return replace(report, backup=snapshot)

    def backup(self, *, profile: Profile | None = None, reason: str = "manual") -> Path:
        """Export a profile into the backups folder and prune the old ones.

        Returns:
            The archive that was written.

        Raises:
            ProfileError: The backup could not be written.
        """
        target = profile if profile is not None else self._active
        stamp = utc_now().strftime("%Y%m%d_%H%M%S")
        destination = self.backups_dir / f"{_file_stem(target.name)}_{reason}_{stamp}.zip"
        settings = self._config.settings if self._config is not None else None
        export_bundle(
            self._repositories,
            destination,
            profile=target,
            settings=settings,
            paths=self._paths,
            include_sounds=False,
        )
        self._prune_backups()
        _log.info("резервная копия профиля «%s»: %s", target.name, destination.name)
        return destination

    def list_backups(self) -> list[Path]:
        """Backup archives, newest first."""
        if not self.backups_dir.is_dir():
            return []
        archives = [item for item in self.backups_dir.glob("*.zip") if item.is_file()]
        return sorted(archives, key=lambda item: item.stat().st_mtime, reverse=True)

    def _prune_backups(self) -> None:
        for stale in self.list_backups()[MAX_BACKUPS:]:
            try:
                stale.unlink()
            except OSError:
                _log.warning("не удалось удалить старую резервную копию %s", stale.name)

    def reset(self, *, profile: Profile | None = None, backup: bool = True) -> Profile:
        """Empty a profile: commands, folders and its own variables all go.

        The profile row itself stays, so the user keeps their name, and the
        active profile does not change under them.

        Returns:
            The profile that was reset.

        Raises:
            ProfileError: The profile has no id.
        """
        target = profile if profile is not None else self._active
        profile_id = self._require_id(target)
        if backup:
            self.backup(profile=target, reason="reset")

        with self._repositories.database.transaction():
            for command in self._repositories.commands.list_for_profile(profile_id):
                if command.id is not None:
                    self._repositories.commands.delete(command.id)
            for folder in self._repositories.folders.list_for_profile(profile_id):
                if folder.id is not None and folder.profile_id == profile_id:
                    self._repositories.folders.delete(folder.id)
            for variable in self._repositories.variables.list_all(
                scope=VariableScope.PROFILE, profile_id=profile_id
            ):
                self._repositories.variables.delete(
                    variable.name, scope=VariableScope.PROFILE, profile_id=profile_id
                )

        _log.info("профиль «%s» сброшен", target.name)
        if profile_id == self._active.id:
            self._notify_switch(self._active, None)
        return target

    # ------------------------------------------------------------------
    # profile root
    # ------------------------------------------------------------------

    def open_folder(self, target: Path | None = None) -> Path:
        """Show a folder of the profile in the file manager.

        Args:
            target: Folder to open. Defaults to the profile root.

        Returns:
            The folder that was opened.

        Raises:
            ProfileError: The folder does not exist or the shell refused to open it.
        """
        folder = target if target is not None else self._paths.root
        if not folder.is_dir():
            raise ProfileError(
                f"not a directory: {folder}",
                user_message=f"Папка не найдена:\n{folder}",
            )
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                # Only reached by the test suite; Ayris itself is a Windows app.
                self._open_folder_posix(folder)
        except OSError as exc:
            raise ProfileError(
                f"cannot open {folder}: {exc}",
                user_message=f"Не удалось открыть папку:\n{folder}",
            ) from exc
        return folder

    @staticmethod
    def _open_folder_posix(folder: Path) -> None:
        """``xdg-open`` fallback, kept out of the Windows path above."""
        opener = shutil.which("xdg-open")
        if opener is None:
            raise ProfileError(
                "no xdg-open available",
                user_message=f"Откройте папку вручную:\n{folder}",
            )
        subprocess.Popen(
            [opener, str(folder)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def change_root(self, target: Path, *, remove_source: bool = True) -> AppPaths:
        """Move the whole installation to ``target`` and keep running.

        This is what makes syncing through Syncthing or OneDrive possible
        (section 17): the user points Ayris at a folder inside the synced tree
        and everything follows. The data is copied, verified, and only then does
        the pointer file change — a failure leaves the original in place and the
        application still running on it.

        Caches, logs and the lock file are deliberately left behind: they
        describe this run, not this profile.

        Args:
            target: New root. Must be empty or not exist.
            remove_source: Delete the old root once the copy is verified.
                A file another process holds open only produces a warning.

        Returns:
            The new path set, already installed process-wide.

        Raises:
            ProfileError: The target is unusable, the copy failed, or the copied
                database did not pass an integrity check.
        """
        destination = target.expanduser().resolve()
        source = self._paths.root
        if destination == source:
            return self._paths
        self._check_destination(destination, source)

        database = self._repositories.database
        database.checkpoint()
        database.close()

        try:
            shutil.copytree(source, destination, ignore=_copy_filter, dirs_exist_ok=True)
        except (OSError, shutil.Error) as exc:
            init_database(self._paths.database_file, migrate=False)
            raise ProfileError(
                f"cannot copy profile {source} -> {destination}: {exc}",
                user_message=(
                    f"Не удалось перенести данные профиля в:\n{destination}\n"
                    "Папка осталась на прежнем месте."
                ),
            ) from exc

        return self._adopt_root(destination, source, remove_source=remove_source)

    def _check_destination(self, destination: Path, source: Path) -> None:
        """Refuse a target that would nest inside the source or overwrite data."""
        if destination.is_relative_to(source):
            raise ProfileError(
                f"{destination} is inside {source}",
                user_message="Нельзя перенести профиль внутрь его собственной папки.",
            )
        if destination.exists():
            if not destination.is_dir():
                raise ProfileError(
                    f"{destination} is not a directory",
                    user_message=f"Указанный путь занят файлом:\n{destination}",
                )
            # The pointer file lives in the default root and names the folder we
            # are currently running from, so a move *back* to the default root
            # always finds it there. Ignoring it is what makes returning home
            # possible; anything else in the folder is still someone's data.
            leftovers = [item for item in destination.iterdir() if item.name != POINTER_FILE_NAME]
            if leftovers:
                raise ProfileError(
                    f"{destination} is not empty",
                    user_message=(
                        f"Папка не пуста:\n{destination}\nВыберите пустую или новую папку."
                    ),
                )
        try:
            destination.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ProfileError(
                f"cannot create {destination}: {exc}",
                user_message=(
                    f"Не удалось создать папку:\n{destination}\nПроверьте права доступа."
                ),
            ) from exc

    def _adopt_root(self, destination: Path, source: Path, *, remove_source: bool) -> AppPaths:
        """Point the process at the copied root, verifying it before committing."""
        moved = Database.open(destination / self._paths.database_file.name, migrate=False)
        if not moved.integrity_check():
            moved.close()
            shutil.rmtree(destination, ignore_errors=True)
            init_database(self._paths.database_file, migrate=False)
            raise ProfileError(
                f"copied database at {destination} failed its integrity check",
                user_message=(
                    "Перенесённая база данных повреждена. " "Профиль остался на прежнем месте."
                ),
            )
        moved.close()

        if destination == default_root():
            # Back to the standard location: the pointer would be redundant, and
            # a stale one is how a later move ends up somewhere unexpected.
            clear_configured_root()
        else:
            write_configured_root(destination)

        paths = init_paths(profile=destination)
        database = init_database(paths.database_file, migrate=False)
        self._paths = paths
        self._repositories = Repositories(database)
        self._active = self._resolve_active()
        _log.info("папка профиля перенесена: %s -> %s", source, destination)

        if remove_source:
            self._discard_old_root(source)
        if self._on_rebind is not None:
            self._on_rebind(paths, self._repositories)
        self._notify_switch(self._active, None)
        return paths

    def _discard_old_root(self, source: Path) -> None:
        """Delete the previous root, tolerating files Windows still holds open."""
        shutil.rmtree(source, ignore_errors=True)
        if source.exists():
            _log.warning("старая папка профиля удалена не полностью: %s", source)


__all__ = [
    "BACKUP_DIR_NAME",
    "DEFAULT_PROFILE_NAME",
    "MAX_BACKUPS",
    "MAX_NAME_LENGTH",
    "ProfileListener",
    "ProfileManager",
    "ProfileSwitched",
    "ProfilesChanged",
]
