from repositories.base import BaseRepository


class SessionRepository(BaseRepository):
    async def create(self, user_id, title: str | None = None):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO sessions (user_id, title)
                VALUES (%s, %s)
                RETURNING id, user_id, title, summary, summary_turn_count, created_at, updated_at
                """,
                (user_id, title),
            )
            row = await cur.fetchone()
        await self.conn.commit()
        return self._to_dict(row)

    async def get_by_id(self, session_id):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, user_id, title, summary, summary_turn_count, created_at, updated_at
                FROM sessions WHERE id = %s
                """,
                (session_id,),
            )
            row = await cur.fetchone()
        return self._to_dict(row) if row else None

    async def list_by_user(self, user_id):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, user_id, title, summary, summary_turn_count, created_at, updated_at
                FROM sessions WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            rows = await cur.fetchall()
        return [self._to_dict(r) for r in rows]

    async def touch(self, session_id):
        async with self.conn.cursor() as cur:
            await cur.execute("UPDATE sessions SET updated_at = now() WHERE id = %s", (session_id,))
        await self.conn.commit()

    async def update_summary(self, session_id, summary: str, summary_turn_count: int):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE sessions
                SET summary = %s, summary_turn_count = %s, summary_updated_at = now(), updated_at = now()
                WHERE id = %s
                """,
                (summary, summary_turn_count, session_id),
            )
        await self.conn.commit()

    async def update_title_if_null(self, session_id, title: str) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE sessions
                SET title = %s, updated_at = now()
                WHERE id = %s AND title IS NULL
                """,
                (title, session_id),
            )
        await self.conn.commit()

    def _to_dict(self, row):
        return {
            "id": row[0],
            "user_id": row[1],
            "title": row[2],
            "summary": row[3],
            "summary_turn_count": row[4],
            "created_at": row[5],
            "updated_at": row[6],
        }
