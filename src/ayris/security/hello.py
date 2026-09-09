"""Fail-closed Windows Hello adapter."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["HelloResult", "WindowsHello"]


class HelloBackend(Protocol):
    def check_availability_async(self) -> Any: ...

    def request_verification_async(self, message: str) -> Any: ...


@dataclass(frozen=True, slots=True)
class HelloResult:
    confirmed: bool
    available: bool
    reason: str = ""


def _await(operation: Any, timeout: float) -> Any:
    done: list[Any] = []
    failed: list[BaseException] = []

    def wait() -> None:
        try:
            done.append(operation.get())
        except BaseException as exc:
            failed.append(exc)

    thread = threading.Thread(target=wait, name="ayris-windows-hello", daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        cancel = getattr(operation, "cancel", None)
        if callable(cancel):
            cancel()
        raise TimeoutError("Windows Hello timed out")
    if failed:
        raise failed[0]
    return done[0]


class WindowsHello:
    """Check availability before every request and classify every result."""

    def __init__(self, backend: HelloBackend | None = None) -> None:
        self._backend = backend

    def _verifier(self) -> HelloBackend:
        if self._backend is None:
            from winrt.windows.security.credentials.ui import UserConsentVerifier

            self._backend = UserConsentVerifier
        return self._backend

    def verify(self, message: str, *, timeout: float = 10.0) -> HelloResult:
        try:
            verifier = self._verifier()
            deadline = time.monotonic() + timeout
            availability = _await(verifier.check_availability_async(), timeout)
            available_name = str(getattr(availability, "name", availability)).casefold()
            if "available" not in available_name or "not" in available_name:
                return HelloResult(False, False, f"hello unavailable: {available_name}")
            remaining = max(0.0, deadline - time.monotonic())
            result = _await(verifier.request_verification_async(message), remaining)
            name = str(getattr(result, "name", result)).casefold()
            if name in {"verified", "0"} or name.endswith(".verified"):
                return HelloResult(True, True, "hello verified")
            return HelloResult(False, True, f"hello rejected: {name}")
        except TimeoutError:
            return HelloResult(False, True, "hello timeout")
        except Exception as exc:
            return HelloResult(False, False, f"hello unavailable: {type(exc).__name__}")
