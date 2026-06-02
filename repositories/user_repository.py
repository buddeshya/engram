from repositories.base import BaseRepository


class UserRepository(BaseRepository):
    async def create(self):
        async with self.conn.cursor() as cur:
            await cur.execute("INSERT INTO users DEFAULT VALUES RETURNING id, created_at")
            row = await cur.fetchone()
        await self.conn.commit()
        return {"id": row[0], "created_at": row[1]}
