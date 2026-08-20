"""Exception hierarchy for Ayris.

Every subsystem raises a typed subclass of :class:`AyrisError` instead of a bare
``Exception``, so callers can react to a category of failure and the GUI always
has a Russian message to show the user.

Two message channels are kept apart on purpose:

* ``args[0]`` / :attr:`AyrisError.technical` — English, for the log file.
* :attr:`AyrisError.user_message` — Russian, for the overlay, toasts and dialogs.

If ``user_message`` is not supplied, the class-level
:attr:`AyrisError.default_user_message` is used, so no code path can end up
showing an English traceback fragment to the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ActionError",
    "ActionNotConfirmed",
    "ActionNotFound",
    "ActionParamsInvalid",
    "ActionRequiresAdmin",
    "ActionTimeout",
    "ActionUnavailable",
    "AudioError",
    "AyrisError",
    "ConfigError",
    "DatabaseError",
    "HotkeyError",
    "LlmError",
    "MacroError",
    "ModelError",
    "ParamProblem",
    "PermissionDeniedError",
    "PluginError",
    "ProfileError",
    "SecretsError",
    "SttError",
    "TtsError",
    "WakeWordError",
]


class AyrisError(Exception):
    """Base class for every error raised by Ayris itself.

    Args:
        technical: English description written to the log.
        user_message: Russian text shown to the user. Falls back to
            :attr:`default_user_message` when omitted.
        recoverable: Whether the application can keep running. ``False`` means
            the caller should tear down the affected subsystem.
    """

    default_user_message = "Произошла внутренняя ошибка Ayris."

    def __init__(
        self,
        technical: str,
        *,
        user_message: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(technical)
        self.technical = technical
        self.user_message = user_message or self.default_user_message
        self.recoverable = recoverable

    def __str__(self) -> str:
        return self.technical

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.technical!r}, "
            f"user_message={self.user_message!r}, recoverable={self.recoverable!r})"
        )


class ConfigError(AyrisError):
    """Configuration file is missing, malformed or fails validation."""

    default_user_message = "Не удалось прочитать настройки. Применены значения по умолчанию."


class SecretsError(AyrisError):
    """Credential store is unavailable or rejected a read/write.

    Raised by :mod:`ayris.core.secrets`. Never carries the secret value itself,
    only the name of the entry, so the message is safe to log.
    """

    default_user_message = (
        "Не удалось обратиться к хранилищу ключей Windows. Введите ключ ещё раз в настройках."
    )


class ProfileError(AyrisError):
    """Profile cannot be created, switched, exported or imported."""

    default_user_message = "Не удалось выполнить операцию с профилем."


class DatabaseError(AyrisError):
    """SQLite storage is unavailable, locked or failed a migration."""

    default_user_message = "Ошибка базы данных Ayris."


class AudioError(AyrisError):
    """Audio device or capture stream failure."""

    default_user_message = "Проблема с аудиоустройством. Проверьте микрофон в настройках."


class WakeWordError(AyrisError):
    """Wake word engine failed to load a model or to process audio."""

    default_user_message = "Не удалось запустить распознавание слова активации."


class SttError(AyrisError):
    """Speech recognition failed — engine, model or network."""

    default_user_message = "Не удалось распознать речь."


class TtsError(AyrisError):
    """Speech synthesis failed — engine, voice or playback."""

    default_user_message = "Не удалось озвучить ответ."


class LlmError(AyrisError):
    """LLM provider is unreachable, rejected the request or returned garbage."""

    default_user_message = "Языковая модель недоступна."


class ModelError(AyrisError):
    """Model download, checksum verification or installation failed."""

    default_user_message = "Не удалось загрузить или проверить модель."


class ActionError(AyrisError):
    """A registered action failed during execution.

    The action layer is the one place where a typed error is not a convenience
    but a requirement: :meth:`ayris.actions.registry.ActionRegistry.execute`
    wraps *everything* an action can raise into a subclass of this, so a caller
    - the pipeline, a macro, a plugin - never has to catch bare ``Exception`` to
    stay alive.

    The subclasses below exist because the caller reacts differently to each:
    a missing action is a library problem to show in the editor, invalid
    parameters are a form to fix, a timeout is worth retrying, and missing rights
    are worth an elevation prompt (task 39).
    """

    default_user_message = "Не удалось выполнить команду."

    @property
    def user_message_ru(self) -> str:
        """Alias of :attr:`AyrisError.user_message`.

        The action layer and the plugin SDK spell the Russian channel out in the
        field name, because a plugin author has no reason to know that
        ``user_message`` is Russian by convention. Same string, one storage.
        """
        return self.user_message


class ActionNotFound(ActionError):
    """No action is registered under that name."""

    default_user_message = "Не знаю такого действия."


@dataclass(frozen=True, slots=True)
class ParamProblem:
    """One rejected parameter: which field, and what is wrong with it in Russian.

    Kept structured rather than pre-formatted because the same list feeds three
    places: the log line, the phrase Ayris says, and the editor, which highlights
    the offending field and needs its name apart from the complaint.
    """

    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.field}: {self.message}" if self.field else self.message


class ActionParamsInvalid(ActionError):
    """Parameters failed validation against the action's schema."""

    default_user_message = "Не поняла параметры команды."

    def __init__(
        self,
        technical: str,
        *,
        problems: Sequence[ParamProblem] = (),
        user_message: str | None = None,
        recoverable: bool = True,
    ) -> None:
        super().__init__(technical, user_message=user_message, recoverable=recoverable)
        self.problems: tuple[ParamProblem, ...] = tuple(problems)

    @property
    def fields(self) -> tuple[str, ...]:
        """Names of the fields that were rejected, in the order pydantic saw them."""
        return tuple(problem.field for problem in self.problems if problem.field)


class ActionTimeout(ActionError):
    """The action did not finish inside :attr:`ActionMeta.timeout_ms`."""

    default_user_message = "Действие не успело выполниться."


class ActionRequiresAdmin(ActionError):
    """The action is marked ``require_admin`` and Ayris is not elevated."""

    default_user_message = "Для этого нужны права администратора."


class ActionUnavailable(ActionError):
    """The action cannot run here: wrong OS, missing dependency, no hardware."""

    default_user_message = "Это действие сейчас недоступно."


class ActionNotConfirmed(ActionError):
    """A dangerous action was not confirmed, so it did not run.

    Covers every way a confirmation can fail to arrive: the user said «нет», the
    prompt timed out, or there was no way to ask in the first place. The action
    itself was never called — that is the whole point of the type, and the reason
    the registry raises it before touching the backend.
    """

    default_user_message = "Не выполняю без подтверждения."


class MacroError(AyrisError):
    """Macro definition is invalid or its execution failed."""

    default_user_message = "Ошибка при выполнении макроса."


class HotkeyError(AyrisError):
    """Global hotkey could not be registered, usually a conflict."""

    default_user_message = "Не удалось зарегистрировать горячую клавишу."


class PluginError(AyrisError):
    """Plugin manifest is invalid or the plugin crashed while loading."""

    default_user_message = "Ошибка плагина."


class PermissionDeniedError(AyrisError):
    """Operation requires elevation or a permission the user did not grant."""

    default_user_message = "Недостаточно прав для выполнения этого действия."
