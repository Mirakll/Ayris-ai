"""The confirmation callable installed into the action registry."""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from difflib import SequenceMatcher
from typing import Protocol, TypeVar

from ayris.actions.base import Action
from ayris.actions.registry import ConfirmationRequest, ConfirmationVerdict
from ayris.core.config import PrivacyConfig
from ayris.security.hello import WindowsHello
from ayris.security.pin import PinManager
from ayris.security.policy import ConfirmationMethod, ConfirmationPolicy

__all__ = ["ConfirmationManager", "VoicePrompt", "windows_dialog_prompt"]

_YES = ("да", "подтверждаю", "подтвердить", "выполняй", "согласен")
_NO = ("нет", "не надо", "отмена", "отменить", "стоп")
_T = TypeVar("_T")


class VoicePrompt(Protocol):
    def __call__(self, question: str, timeout: float) -> str | None: ...


class ConfirmationManager:
    """Resolve policy and return a verdict; never execute or audit an action."""

    def __init__(
        self,
        privacy: PrivacyConfig | Callable[[], PrivacyConfig],
        *,
        action: Callable[[str], Action | None] | None = None,
        voice: VoicePrompt | None = None,
        pause_pipeline: Callable[[], AbstractContextManager[None]] | None = None,
        pin: PinManager | None = None,
        ask_pin: Callable[[int], str | None] | None = None,
        dialog: Callable[[ConfirmationRequest, float], bool | None] | None = None,
        hello: WindowsHello | None = None,
    ) -> None:
        self._privacy = privacy
        self._action = action
        self._voice = voice
        self._pause = pause_pipeline
        self._pin = pin
        self._ask_pin = ask_pin
        self._dialog = dialog
        self._hello = hello or WindowsHello()

    def __call__(self, request: ConfirmationRequest) -> ConfirmationVerdict:
        settings = self._settings()
        if not settings.require_confirmation:
            return ConfirmationVerdict.yes()
        subject = self._action(request.action) if self._action is not None else None
        method = ConfirmationPolicy(settings).method_for(subject)
        return self._confirm(method, request, settings)

    def requires_confirmation(self, action: Action) -> bool:
        """Let the registry extend its existing barrier with user-picked actions."""
        return action.meta.name in self._settings().confirmation_actions

    def _settings(self) -> PrivacyConfig:
        if callable(self._privacy):
            return self._privacy()
        return self._privacy

    def _confirm(
        self, method: ConfirmationMethod, request: ConfirmationRequest, settings: PrivacyConfig
    ) -> ConfirmationVerdict:
        if method is ConfirmationMethod.VOICE:
            return self._voice_verdict(request, settings)
        if method is ConfirmationMethod.PIN:
            return self._pin_verdict(settings)
        if method is ConfirmationMethod.DIALOG:
            return self._dialog_verdict(request, settings.confirmation_timeout_sec)
        if method is ConfirmationMethod.BOTH:
            first = self._voice_verdict(request, settings)
            if first.confirmed or first.reason not in {"unavailable", "timeout"}:
                return first
            return self._dialog_verdict(request, settings.confirmation_timeout_sec)
        hello = self._hello.verify(request.question_ru, timeout=settings.confirmation_timeout_sec)
        if hello.confirmed:
            return ConfirmationVerdict.yes()
        if hello.available:
            message = "Подтверждение Windows Hello отклонено."
            if "timeout" in hello.reason:
                message = "Время ожидания Windows Hello истекло."
            return ConfirmationVerdict.no(hello.reason, user_message=message)
        fallback = ConfirmationMethod(settings.confirmation_fallback)
        result = self._confirm(fallback, request, settings)
        if not result.confirmed and result.reason == "unavailable":
            return ConfirmationVerdict.no(
                "hello and fallback unavailable",
                user_message="Windows Hello и резервный способ подтверждения недоступны.",
            )
        return result

    def _voice_verdict(
        self, request: ConfirmationRequest, settings: PrivacyConfig
    ) -> ConfirmationVerdict:
        if self._voice is None:
            return ConfirmationVerdict.no(
                "unavailable", user_message="Голосовое подтверждение сейчас недоступно."
            )
        voice = self._voice
        guard = self._pause() if self._pause is not None else nullcontext()
        with guard:
            completed, answer = _bounded(
                lambda: voice(
                    f"{request.question_ru} Вы уверены? Скажите да или нет.",
                    settings.confirmation_timeout_sec,
                ),
                settings.confirmation_timeout_sec,
            )
        if not completed or answer is None:
            return ConfirmationVerdict.no(
                "timeout", user_message="Время ожидания подтверждения истекло."
            )
        decision = _voice_decision(answer, settings.confirmation_fuzzy_threshold)
        if decision is True:
            return ConfirmationVerdict.yes()
        if decision is False:
            return ConfirmationVerdict.no("rejected", user_message="Действие отменено.")
        return ConfirmationVerdict.no(
            "unclear", user_message="Подтверждение не распознано, действие отменено."
        )

    def _pin_verdict(self, settings: PrivacyConfig) -> ConfirmationVerdict:
        if self._pin is None or self._ask_pin is None or not self._pin.configured():
            return ConfirmationVerdict.no("unavailable", user_message="ПИН-код не настроен.")
        pin = self._pin
        ask_pin = self._ask_pin
        completed, accepted = _bounded(
            lambda: pin.prompt(
                ask_pin,
                attempts=settings.confirmation_pin_attempts,
                delay_sec=settings.confirmation_pin_delay_sec,
            ),
            settings.confirmation_timeout_sec,
        )
        if not completed:
            return ConfirmationVerdict.no(
                "timeout", user_message="Время ожидания ПИН-кода истекло."
            )
        if accepted:
            return ConfirmationVerdict.yes()
        return ConfirmationVerdict.no(
            "pin rejected", user_message="Неверный ПИН-код. Действие отменено."
        )

    def _dialog_verdict(self, request: ConfirmationRequest, timeout: float) -> ConfirmationVerdict:
        if self._dialog is None:
            return ConfirmationVerdict.no(
                "unavailable", user_message="Окно подтверждения недоступно."
            )
        dialog = self._dialog
        completed, answer = _bounded(lambda: dialog(request, timeout), timeout)
        if not completed:
            answer = None
        if answer is True:
            return ConfirmationVerdict.yes()
        reason = "timeout" if answer is None else "rejected"
        message = (
            "Время ожидания подтверждения истекло." if answer is None else "Действие отменено."
        )
        return ConfirmationVerdict.no(reason, user_message=message)


def _voice_decision(text: str, threshold: float) -> bool | None:
    normalized = " ".join(re.findall(r"[а-яё]+", text.casefold()))
    if not normalized:
        return None
    scored: list[tuple[float, bool]] = []
    for phrase in _YES:
        scored.append((SequenceMatcher(None, normalized, phrase).ratio(), True))
    for phrase in _NO:
        scored.append((SequenceMatcher(None, normalized, phrase).ratio(), False))
    score, decision = max(scored, key=lambda item: item[0])
    return decision if score >= threshold else None


def windows_dialog_prompt(request: ConfirmationRequest, timeout: float) -> bool | None:
    """Show a bounded native Yes/No dialog without importing the GUI package."""
    if sys.platform != "win32":
        return None
    from ctypes import WINFUNCTYPE, WinDLL, c_int, c_uint, c_void_p, wintypes

    user32 = WinDLL("user32", use_last_error=True)
    prototype = WINFUNCTYPE(
        c_int,
        wintypes.HWND,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        c_uint,
        wintypes.WORD,
        c_uint,
    )
    try:
        message_box_timeout = prototype(("MessageBoxTimeoutW", user32))
    except AttributeError:
        return None
    result = message_box_timeout(
        c_void_p(),
        request.question_ru,
        "Подтверждение Ayris",
        0x00000004 | 0x00000020 | 0x00000100,
        0,
        max(1, round(timeout * 1000)),
    )
    if result == 6:
        return True
    if result == 7:
        return False
    return None


def _bounded(call: Callable[[], _T], timeout: float) -> tuple[bool, _T | None]:
    """Bound an integration callback even when its implementation wedges."""
    values: list[_T] = []
    failed: list[Exception] = []

    def run() -> None:
        try:
            values.append(call())
        except Exception as exc:
            failed.append(exc)

    worker = threading.Thread(target=run, name="ayris-confirmation", daemon=True)
    worker.start()
    worker.join(max(0.0, timeout))
    if worker.is_alive():
        return False, None
    if failed:
        return True, None
    return True, values[0] if values else None
