"""Typed theme tokens loaded from the shared JSON resources."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ayris.core.paths import executable_dir

__all__ = [
    "ColorTokens",
    "MetricTokens",
    "Theme",
    "ThemeLoadError",
    "TypographyTokens",
    "bundled_theme_path",
    "load_theme",
]

_COLOUR_RE = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")


class ThemeLoadError(ValueError):
    """A theme file is unreadable or does not match the common schema."""


class _Tokens(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ColorTokens(_Tokens):
    background: str
    surface: str
    surface_highlight: str
    border: str
    accent: str
    accent_hover: str
    accent_pressed: str
    accent_disabled: str
    text_primary: str
    text_secondary: str
    text_muted: str
    success: str
    warning: str
    error: str
    info: str
    overlay: str
    focus: str
    on_accent: str

    @field_validator("*")
    @classmethod
    def valid_colour(cls, value: str) -> str:
        if not _COLOUR_RE.fullmatch(value):
            raise ValueError("ожидается цвет #RRGGBB или #RRGGBBAA")
        return value.upper()


class TypographyTokens(_Tokens):
    family: str = Field(min_length=1)
    fallbacks: tuple[str, ...] = Field(min_length=1)
    h1_size: int = Field(gt=0)
    h2_size: int = Field(gt=0)
    body_size: int = Field(gt=0)
    caption_size: int = Field(gt=0)
    weight_regular: int = Field(ge=100, le=900)
    weight_medium: int = Field(ge=100, le=900)
    weight_bold: int = Field(ge=100, le=900)
    line_height_tight: float = Field(gt=0)
    line_height_normal: float = Field(gt=0)


class MetricTokens(_Tokens):
    radius_sm: int = Field(ge=0)
    radius_md: int = Field(ge=0)
    radius_lg: int = Field(ge=0)
    radius_xl: int = Field(ge=0)
    spacing_xs: int = Field(ge=0)
    spacing_sm: int = Field(ge=0)
    spacing_md: int = Field(ge=0)
    spacing_lg: int = Field(ge=0)
    spacing_xl: int = Field(ge=0)
    spacing_2xl: int = Field(ge=0)
    border_width: int = Field(gt=0)
    focus_width: int = Field(gt=0)
    control_height: int = Field(gt=0)
    control_height_lg: int = Field(gt=0)
    icon_sm: int = Field(gt=0)
    icon_md: int = Field(gt=0)
    icon_lg: int = Field(gt=0)
    toggle_width: int = Field(gt=0)
    toggle_height: int = Field(gt=0)
    toggle_knob_margin: int = Field(ge=0)
    slider_field_width: int = Field(gt=0)
    notice_min_height: int = Field(gt=0)
    toast_width: int = Field(gt=0)
    dialog_width: int = Field(gt=0)
    busy_size: int = Field(gt=0)
    animation_fast: int = Field(ge=0)
    animation_normal: int = Field(ge=0)
    animation_slow: int = Field(ge=0)
    window_min_width: int = Field(gt=0)
    window_min_height: int = Field(gt=0)
    content_width: int = Field(gt=0)


class Theme(_Tokens):
    name: str = Field(min_length=1)
    mode: Literal["dark", "light"]
    colors: ColorTokens
    typography: TypographyTokens
    metrics: MetricTokens

    def color(self, name: str) -> str:
        """Return a semantic colour or raise a useful lookup error."""
        if name not in ColorTokens.model_fields:
            raise KeyError(f"В теме «{self.name}» нет цвета color.{name}")
        return str(getattr(self.colors, name))

    def metric(self, name: str, *, scale: float = 1.0) -> int:
        """Return a logical metric, optionally scaled for a DPI preview."""
        if name not in MetricTokens.model_fields:
            raise KeyError(f"В теме «{self.name}» нет метрики metric.{name}")
        return max(0, round(int(getattr(self.metrics, name)) * scale))

    def type_value(self, name: str) -> str | int | float | tuple[str, ...]:
        if name not in TypographyTokens.model_fields:
            raise KeyError(f"В теме «{self.name}» нет токена typography.{name}")
        value = getattr(self.typography, name)
        if isinstance(value, str | int | float | tuple):
            return value
        raise TypeError(f"Неподдерживаемый токен typography.{name}")


def bundled_theme_path(name: str) -> Path:
    """Path to a theme shipped with the application."""
    return executable_dir() / "resources" / "themes" / f"{name}.json"


def _format_validation_error(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_url=False):
        key = ".".join(str(part) for part in item["loc"]) or "<корень>"
        message = "отсутствует обязательный ключ" if item["type"] == "missing" else item["msg"]
        details.append(f"{key}: {message}")
    return "; ".join(details)


def load_theme(path: Path | str) -> Theme:
    """Read and validate one theme against the schema shared by all themes."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise ThemeLoadError(f"Не удалось прочитать тему {source}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ThemeLoadError(
            f"Тема {source} содержит неверный JSON: строка {exc.lineno}, столбец {exc.colno}"
        ) from exc
    try:
        return Theme.model_validate(data)
    except ValidationError as exc:
        raise ThemeLoadError(
            f"Тема {source} не прошла проверку: {_format_validation_error(exc)}"
        ) from exc
