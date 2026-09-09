"""Pattern-based last-resort redaction for logs and support archives."""

from __future__ import annotations

import re
from typing import Final

from ayris.utils.logger import SECRET_PLACEHOLDER

__all__ = ["redact_patterns", "redact_text"]

_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{48,}={0,2}(?![A-Za-z0-9+/=])"),
    re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
)


def _luhn(candidate: str) -> bool:
    digits = [int(char) for char in candidate if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_patterns(text: str) -> str:
    """Redact token and card-shaped strings without consulting the registry."""
    cleaned = text
    for index, pattern in enumerate(_PATTERNS):
        if index == len(_PATTERNS) - 1:
            cleaned = pattern.sub(
                lambda match: SECRET_PLACEHOLDER if _luhn(match.group(0)) else match.group(0),
                cleaned,
            )
        else:
            cleaned = pattern.sub(SECRET_PLACEHOLDER, cleaned)
    return cleaned


def redact_text(text: str) -> str:
    """Redact known values first, then token and card-shaped strings."""
    from ayris.utils.logger import redact

    return redact(text)
