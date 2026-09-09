"""Map action metadata onto a configured confirmation method."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ayris.actions.base import Action
    from ayris.core.config import PrivacyConfig

__all__ = ["ConfirmationMethod", "ConfirmationPolicy"]


class ConfirmationMethod(StrEnum):
    VOICE = "voice"
    DIALOG = "dialog"
    BOTH = "both"
    PIN = "pin"
    HELLO = "hello"


@dataclass(frozen=True, slots=True)
class ConfirmationPolicy:
    privacy: PrivacyConfig

    def method_for(self, action: Action | None) -> ConfirmationMethod:
        default = ConfirmationMethod(self.privacy.confirmation_method)
        if action is None:
            return default
        category = self.category_of(action)
        configured = self.privacy.confirmation_by_category.get(category, default.value)
        try:
            return ConfirmationMethod(configured)
        except ValueError:
            return default

    @staticmethod
    def category_of(action: Action) -> str:
        meta = action.meta
        module = type(action).__module__.casefold()
        if meta.require_admin:
            return "admin"
        if ".power" in module:
            return "power"
        if any(part in module for part in (".file", ".clipboard")):
            return "files"
        if ".network" in module:
            return "network"
        if meta.plugin or ".macros" in module:
            return "scripts"
        return meta.category.value
