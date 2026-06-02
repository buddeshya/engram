from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from memory.retriever import retrieve_dual


@pytest.mark.asyncio
async def test_retrieve_dual_uses_single_embedding_call(mock_openai):
    fake_vector = [0.1] * 1536
    mock_openai.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=fake_vector)]
    )
    semantic_memory = SimpleNamespace(id=uuid4(), type=SimpleNamespace(value="fact"))
    episodic_memory = SimpleNamespace(id=uuid4(), type=SimpleNamespace(value="episodic"))
    repo = SimpleNamespace(
        search_by_vector=AsyncMock(
            side_effect=[
                [(semantic_memory, 0.8)],
                [(episodic_memory, 0.9)],
            ]
        )
    )

    semantic, episodic = await retrieve_dual(
        memory_repo=repo,
        openai_client=mock_openai,
        user_id=uuid4(),
        query="help with api",
        embedding_model="text-embedding-3-small",
        semantic_top_k=5,
        episodic_top_k=3,
    )

    assert len(semantic) == 1
    assert len(episodic) == 1
    assert mock_openai.embeddings.create.await_count == 1
    assert repo.search_by_vector.await_count == 2
