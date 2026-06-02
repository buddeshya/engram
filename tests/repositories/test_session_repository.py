import pytest

from repositories.session_repository import SessionRepository


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
async def test_create_and_list_sessions(db_conn):
    repo = SessionRepository(db_conn)
    user_id = await _create_user(db_conn)
    try:
        created = await repo.create(user_id=user_id, title="First session")
        sessions = await repo.list_by_user(user_id)
        assert created["id"] is not None
        assert any(s["id"] == created["id"] for s in sessions)
    finally:
        await _cleanup_user(db_conn, user_id)


@pytest.mark.asyncio
async def test_update_summary_and_title_once(db_conn):
    repo = SessionRepository(db_conn)
    user_id = await _create_user(db_conn)
    try:
        created = await repo.create(user_id=user_id, title=None)
        await repo.update_summary(created["id"], "Summary v1", 10)
        await repo.update_title_if_null(created["id"], "Auto title")
        await repo.update_title_if_null(created["id"], "Should not overwrite")
        saved = await repo.get_by_id(created["id"])
        assert saved["summary"] == "Summary v1"
        assert saved["summary_turn_count"] == 10
        assert saved["title"] == "Auto title"
    finally:
        await _cleanup_user(db_conn, user_id)
