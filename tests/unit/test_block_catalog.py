"""Task 33: complete, introspected macro-block palette."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from ayris.actions.macros.blocks.catalog import BLOCKS_BY_CATEGORY, BlockCatalog, BlockCategory
from ayris.actions.macros.blocks.web import set_transport
from ayris.actions.macros.docs import generate_block_docs
from ayris.actions.macros.engine import MacroEngine
from ayris.actions.macros.errors import MacroTimeoutError
from ayris.actions.macros.schema import CommandModel
from ayris.actions.registry import ActionRegistry
from ayris.actions.result import ActionResult
from ayris.core.events import EventBus, LogLine, NotificationRequested

pytestmark = pytest.mark.unit


class FakeRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def has(self, name: str) -> bool:
        return True

    def execute(
        self, name: str, params: dict[str, Any] | None = None, **_options: Any
    ) -> ActionResult[Any]:
        self.calls.append((name, dict(params or {})))
        return ActionResult.done()


def command(*actions: dict[str, Any]) -> CommandModel:
    return CommandModel(name="Каталог", actions=list(actions))


@pytest.fixture(scope="module")
def registry() -> ActionRegistry:
    result = ActionRegistry()
    result.discover()
    yield result
    result.shutdown()


@pytest.fixture(scope="module")
def catalog(registry: ActionRegistry) -> BlockCatalog:
    return BlockCatalog(registry)


def test_every_spec_block_has_complete_palette_metadata(catalog: BlockCatalog) -> None:
    expected = {name for names in BLOCKS_BY_CATEGORY.values() for name in names}
    actual = {
        block.type
        for category in catalog.list_categories()
        for block in catalog.list_blocks(category.type)
    }
    assert (
        actual == expected
    ), f"Пропущены: {sorted(expected - actual)}; лишние: {sorted(actual - expected)}"
    for name in expected:
        block = catalog.get(name)
        assert block.title_ru and block.description_ru and block.icon
        assert block.schema.get("type") == "object"
        assert isinstance(block.example, dict)
        if not block.available:
            assert block.unavailable_reason


def test_action_wrappers_reuse_registry_schema_and_examples_validate(
    catalog: BlockCatalog, registry: ActionRegistry
) -> None:
    for category in catalog.list_categories():
        for block in catalog.list_blocks(category.type):
            if not block.action_name:
                continue
            schema = registry.describe(block.action_name)
            assert block.fields == schema.fields
            registry.get(block.action_name).params_model().model_validate(block.example)


def test_wrapper_is_dispatched_to_exactly_the_named_action(catalog: BlockCatalog) -> None:
    fake = FakeRegistry()
    wrappers = [
        block
        for category in catalog.list_categories()
        for block in catalog.list_blocks(category.type)
        if block.action_name
    ]
    with MacroEngine(fake) as engine:
        for wrapper in wrappers:
            engine.run(command({"type": wrapper.type, "params": wrapper.example}))
    assert [name for name, _params in fake.calls] == [block.action_name for block in wrappers]


def test_palette_order_and_search(catalog: BlockCatalog) -> None:
    assert [item.type for item in catalog.list_categories()] == list(BlockCategory)
    assert catalog.search("веб-запрос")[0].type == "WebRequest"
    assert catalog.search("WebRequest")[0].type == "WebRequest"


def test_wait_supports_seconds_and_condition_timeout() -> None:
    with MacroEngine(FakeRegistry()) as engine:
        assert engine.run(command({"type": "Wait", "params": {"seconds": 0.001}})).ok
        report = engine.run(
            command(
                {"type": "Wait", "params": {"condition": "False", "timeout_ms": 2, "poll_ms": 1}}
            )
        )
    assert not report.ok
    assert isinstance(report.error, MacroTimeoutError)


def test_wait_is_cancelled_cooperatively() -> None:
    with MacroEngine(FakeRegistry()) as engine:
        run = engine.start(command({"type": "Wait", "params": {"seconds": 10}}))
        assert engine.cancel(run.run_id, reason="тест") == 1
        report = run.wait(1)
    assert report.cancelled


def test_web_request_parses_json_path_into_variable() -> None:
    set_transport(
        httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": [{"value": 42}]}))
    )
    fake = FakeRegistry()
    try:
        with MacroEngine(fake) as engine:
            report = engine.run(
                command(
                    {
                        "type": "WebRequest",
                        "params": {
                            "url": "https://example.test/api",
                            "json_path": "$.data[0].value",
                            "into": "answer",
                        },
                    },
                    {"type": "Echo", "params": {"value": "{answer}"}},
                )
            )
    finally:
        set_transport(None)
    assert report.ok
    assert fake.calls == [("Echo", {"value": 42})]


def test_notification_blocks_publish_ui_events() -> None:
    bus = EventBus(thread_id=None)
    notifications: list[NotificationRequested] = []
    logs: list[LogLine] = []
    bus.subscribe(NotificationRequested, notifications.append)
    bus.subscribe(LogLine, logs.append)
    with MacroEngine(FakeRegistry(), bus=bus) as engine:
        report = engine.run(
            command(
                {
                    "type": "ToastNotify",
                    "params": {"title": "Готово", "message": "Да", "icon": "ok", "action": "Open"},
                },
                {"type": "OverlayLog", "params": {"message": "строка", "level": "warning"}},
            )
        )
    assert report.ok
    assert notifications[0].icon == "ok" and notifications[0].action == "Open"
    assert logs[0].message == "строка" and logs[0].level == "warning"


def test_docs_are_generated_from_every_catalog_entry(catalog: BlockCatalog) -> None:
    page = generate_block_docs(catalog)
    for names in BLOCKS_BY_CATEGORY.values():
        for name in names:
            assert f"`{name}`" in page
    json.loads(page.split("```json\n", 1)[1].split("\n```", 1)[0])
