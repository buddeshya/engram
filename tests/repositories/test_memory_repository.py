from uuid import uuid4

import pytest

from memory.types import MemoryCandidate, MemoryType
from repositories.memory_repository import MemoryRepository
from tests.conftest import unit_vector


async def _create_user(conn):
    async with conn.cursor() as cur:
        await cur.execute("INSERT INTO users DEFAULT VALUES RETURNING id")
        row = await cur.fetchone()
    await conn.commit()
    return row[0]


async def _cleanup_user(conn, user_id):
    async with conn.cursor() as cur:
        await cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
    await conn.commit()


@pytest.mark.asyncio
async def test_insert_and_get_active(db_conn):
    repo = MemoryRepository(db_conn)
    user_id = await _create_user(db_conn)
    try:
        candidate = MemoryCandidate(type=MemoryType.FACT, content="Uses Python", confidence=0.8)
        inserted = await repo.insert(candidate, unit_vector(11), user_id=user_id, session_id=None)
        active = await repo.get_active(user_id)
        assert inserted.id is not None
        assert any(m.id == inserted.id and m.status.value == "active" for m in active)
    finally:
        await _cleanup_user(db_conn, user_id)


@pytest.mark.asyncio
async def test_supersede_and_search_filters(db_conn):
    repo = MemoryRepository(db_conn)
    user_id = await _create_user(db_conn)
    try:
        vec = unit_vector(19)
        base = await repo.insert(
            MemoryCandidate(type=MemoryType.PREFERENCE, content="Prefers tabs", confidence=0.7),
            vec,
            user_id=user_id,
            session_id=None,
        )
        await repo.supersede(base.id)
        matches = await repo.search_by_vector(
            user_id=user_id,
            embedding=vec,
            types=[MemoryType.PREFERENCE],
            top_k=5,
            threshold=0.1,
        )
        assert matches == []
    finally:
        await _cleanup_user(db_conn, user_id)
