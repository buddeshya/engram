from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from memory.types import MemoryType
from services.memory_service import MemoryService


@pytest.mark.asyncio
async def test_process_turn_noop_skips_resolve(monkeypatch, settings_stub, mock_openai):
    repo = SimpleNamespace(get_active=AsyncMock(return_value=[]))
    settings_stub.memory_md_path = "/tmp/engram-memory-test.md"
    service = MemoryService(repo, mock_openai, settings_stub)

    async def fake_extract(*args, **kwargs):
        return []

    resolve_mock = AsyncMock()
    monkeypatch.setattr("services.memory_service.extract", fake_extract)
    monkeypatch.setattr("services.memory_service.resolve", resolve_mock)

    await service.process_turn("u", "a", uuid4(), uuid4())
    assert resolve_mock.await_count == 0


@pytest.mark.asyncio
async def test_update_content_rejects_episodic(settings_stub, mock_openai):
    memory = SimpleNamespace(id=uuid4(), user_id=uuid4(), type=MemoryType.EPISODIC)
    repo = SimpleNamespace(get_by_id=AsyncMock(return_value=memory))
    service = MemoryService(repo, mock_openai, settings_stub)
    with pytest.raises(ValueError, match="read-only"):
        await service.update_content(memory.id, memory.user_id, "new text")


@pytest.mark.asyncio
async def test_list_active_groups_by_type(settings_stub, mock_openai):
    memories = [
        SimpleNamespace(type=SimpleNamespace(value="fact")),
        SimpleNamespace(type=SimpleNamespace(value="fact")),
        SimpleNamespace(type=SimpleNamespace(value="preference")),
    ]
    repo = SimpleNamespace(get_active=AsyncMock(return_value=memories))
    service = MemoryService(repo, mock_openai, settings_stub)

    result = await service.list_active(uuid4())
    assert result["total"] == 3
    assert result["by_type"]["fact"] == 2
    assert result["by_type"]["preference"] == 1
