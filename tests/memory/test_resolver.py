from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from memory.resolver import resolve
from memory.types import MemoryCandidate, MemoryType


@pytest.mark.asyncio
async def test_resolver_exact_duplicate_short_circuits_llm(mock_openai):
    repo = SimpleNamespace(
        get_active_by_type=AsyncMock(return_value=[SimpleNamespace(id=uuid4(), content="Prefers tabs", confidence=0.8)]),
        search_by_vector=AsyncMock(),
        update_confidence=AsyncMock(),
        insert=AsyncMock(),
    )
    candidate = MemoryCandidate(type=MemoryType.PREFERENCE, content="   prefers   tabs   ", confidence=0.7)
    await resolve(repo, mock_openai, candidate, [0.1, 0.2], uuid4(), uuid4(), "gpt-4o-mini")
    repo.update_confidence.assert_awaited_once()
    repo.search_by_vector.assert_not_called()
    repo.insert.assert_not_called()


@pytest.mark.asyncio
async def test_resolver_inserts_when_no_match(mock_openai):
    repo = SimpleNamespace(
        get_active_by_type=AsyncMock(return_value=[]),
        search_by_vector=AsyncMock(return_value=[]),
        update_confidence=AsyncMock(),
        insert=AsyncMock(),
    )
    candidate = MemoryCandidate(type=MemoryType.FACT, content="Uses Python", confidence=0.75)
    await resolve(repo, mock_openai, candidate, [0.3, 0.4], uuid4(), uuid4(), "gpt-4o-mini")
    repo.insert.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolver_refinement_updates_content(mock_openai):
    existing = SimpleNamespace(id=uuid4(), content="Uses Python", confidence=0.8)
    repo = SimpleNamespace(
        get_active_by_type=AsyncMock(return_value=[]),
        search_by_vector=AsyncMock(return_value=[(existing, 0.80)]),
        update_confidence=AsyncMock(),
        insert=AsyncMock(),
        supersede=AsyncMock(),
        set_conflict=AsyncMock(),
        update_content=AsyncMock(),
    )
    mock_openai.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"classification":"REFINEMENT","merged_content":"Uses Python and FastAPI"}'
                )
            )
        ]
    )
    candidate = MemoryCandidate(type=MemoryType.FACT, content="Uses FastAPI", confidence=0.75)
    await resolve(repo, mock_openai, candidate, [0.5, 0.6], uuid4(), uuid4(), "gpt-4o-mini")
    repo.update_content.assert_awaited_once()
