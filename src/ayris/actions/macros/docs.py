"""Generate macro-block documentation from the palette catalog."""

from __future__ import annotations

import json
from pathlib import Path

from ayris.actions.macros.blocks.catalog import BlockCatalog

__all__ = ["generate_block_docs", "write_block_docs"]


def generate_block_docs(catalog: BlockCatalog | None = None) -> str:
    catalog = catalog or BlockCatalog()
    lines = ["# Блоки макросов", "", "Страница сгенерирована из каталога блоков.", ""]
    for category in catalog.list_categories():
        lines.extend((f"## {category.title_ru}", ""))
        for block in catalog.list_blocks(category.type):
            lines.extend((f"### {block.title_ru} (`{block.type}`)", "", block.description_ru, ""))
            if not block.available:
                lines.extend((f"> {block.unavailable_reason}", ""))
            lines.extend(
                ("```json", json.dumps(block.example, ensure_ascii=False, indent=2), "```", "")
            )
    return "\n".join(lines)


def write_block_docs(path: str | Path, catalog: BlockCatalog | None = None) -> Path:
    target = Path(path)
    target.write_text(generate_block_docs(catalog), encoding="utf-8")
    return target
