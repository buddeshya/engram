import logging

from openai import AsyncOpenAI

from memory.types import MemoryType
from repositories.memory_repository import MemoryRepository

log = logging.getLogger("engram.retriever")


async def embed(text: str, openai_client: AsyncOpenAI, embedding_model: str) -> list[float]:
    """Convert text to a 1536-dim vector using OpenAI embeddings."""
    response = await openai_client.embeddings.create(model=embedding_model, input=text)
    return response.data[0].embedding


async def retrieve_dual(
    memory_repo: MemoryRepository,
    openai_client: AsyncOpenAI,
    user_id,
    query: str,
    embedding_model: str,
    semantic_top_k: int,
    episodic_top_k: int,
) -> tuple[list, list]:
    """
    Single-embed retrieval for both semantic and episodic memories.

    We embed the query once and reuse the same vector for two separate
    pgvector HNSW index scans — one for semantic types (preference/fact/
    decision/correction) and one for episodic summaries. This keeps the
    embedding cost constant regardless of memory store size.

    Returns: (semantic_memories, episodic_memories)
    """
    query_embedding = await embed(query, openai_client, embedding_model)

    # Semantic search: preferences, facts, decisions, corrections.
    semantic_rows = await memory_repo.search_by_vector(
        user_id=user_id,
        embedding=query_embedding,
        types=[
            MemoryType.CORRECTION,
            MemoryType.PREFERENCE,
            MemoryType.FACT,
            MemoryType.DECISION,
        ],
        top_k=semantic_top_k,
        threshold=0.60,
    )

    # Episodic search: past session summaries.
    episodic_rows = await memory_repo.search_by_vector(
        user_id=user_id,
        embedding=query_embedding,
        types=[MemoryType.EPISODIC],
        top_k=episodic_top_k,
        threshold=0.60,
    )

    semantic = [m for m, _ in semantic_rows]
    episodic = [m for m, _ in episodic_rows]
    log.info("Retrieved  semantic=%d  episodic=%d", len(semantic), len(episodic))
    if semantic:
        for m in semantic:
            content = getattr(m, "content", "")[:70]
            log.info("  [%s] %s", m.type.value, content)
    return semantic, episodic
