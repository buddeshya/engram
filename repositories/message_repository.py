from repositories.base import BaseRepository


class MessageRepository(BaseRepository):
    async def insert(self, session_id, role: str, content: str):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (%s, %s, %s)
                RETURNING id, session_id, role, content, created_at
                """,
                (session_id, role, content),
            )
            row = await cur.fetchone()
        await self.conn.commit()
        return self._to_dict(row)

    async def get_by_session(self, session_id):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            rows = await cur.fetchall()
        return [self._to_dict(r) for r in rows]

    async def get_recent_n(self, session_id, n: int):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, n),
            )
            rows = await cur.fetchall()
        rows.reverse()
        return [self._to_dict(r) for r in rows]

    async def count_by_session(self, session_id) -> int:
        async with self.conn.cursor() as cur:
            await cur.execute("SELECT COUNT(*) FROM messages WHERE session_id = %s", (session_id,))
            row = await cur.fetchone()
        return int(row[0])

    def _to_dict(self, row):
        return {
            "id": row[0],
            "session_id": row[1],
            "role": row[2],
            "content": row[3],
            "created_at": row[4],
        }
