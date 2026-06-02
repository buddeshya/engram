import json

from memory.types import Memory, MemoryStatus, MemoryType
from repositories.base import BaseRepository


class MemoryRepository(BaseRepository):
    async def insert(self, candidate, embedding, user_id, session_id=None, supersedes_id=None) -> Memory:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                INSERT INTO memories
                (user_id, source_session_id, supersedes_id, type, content, embedding, confidence, metadata)
                VALUES (%s, %s, %s, %s::memory_type, %s, %s::vector, %s, %s::jsonb)
                RETURNING id, user_id, type, content, embedding, confidence, status,
                          source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at
                """,
                (
                    user_id,
                    session_id,
                    supersedes_id,
                    candidate.type.value,
                    candidate.content,
                    self._vector_literal(embedding),
                    candidate.confidence,
                    json.dumps(candidate.metadata or {}),
                ),
            )
            row = await cur.fetchone()
        await self.conn.commit()
        return self._to_memory(row)

    async def get_active(self, user_id, exclude_types: list[MemoryType] | None = None) -> list[Memory]:
        query = """
            SELECT id, user_id, type, content, embedding, confidence, status,
                   source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at
            FROM memories
            WHERE user_id = %s AND status = 'active'
        """
        params = [user_id]
        if exclude_types:
            query += " AND NOT (type = ANY(%s::memory_type[]))"
            params.append([t.value for t in exclude_types])
        query += " ORDER BY confidence DESC, created_at DESC"
        async with self.conn.cursor() as cur:
            await cur.execute(query, tuple(params))
            rows = await cur.fetchall()
        return [self._to_memory(r) for r in rows]

    async def get_active_by_type(self, user_id, memory_type: MemoryType) -> list[Memory]:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, user_id, type, content, embedding, confidence, status,
                       source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at
                FROM memories
                WHERE user_id = %s AND type = %s::memory_type AND status = 'active'
                ORDER BY confidence DESC
                """,
                (user_id, memory_type.value),
            )
            rows = await cur.fetchall()
        return [self._to_memory(r) for r in rows]

    async def search_by_vector(self, user_id, embedding, types: list[MemoryType], top_k: int, threshold: float):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, user_id, type, content, embedding, confidence, status,
                       source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM memories
                WHERE user_id = %s
                  AND status = 'active'
                  AND type = ANY(%s::memory_type[])
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (self._vector_literal(embedding), user_id, [t.value for t in types], self._vector_literal(embedding), top_k * 2),
            )
            rows = await cur.fetchall()
        out = []
        for row in rows:
            memory = self._to_memory(row[:-1])
            similarity = float(row[-1])
            if similarity >= threshold:
                out.append((memory, similarity))
            if len(out) >= top_k:
                break
        return out

    async def update_content(self, memory_id, new_content: str, new_embedding: list[float]) -> Memory:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE memories
                SET content = %s, embedding = %s::vector
                WHERE id = %s
                RETURNING id, user_id, type, content, embedding, confidence, status,
                          source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at
                """,
                (new_content, self._vector_literal(new_embedding), memory_id),
            )
            row = await cur.fetchone()
        await self.conn.commit()
        return self._to_memory(row)

    async def supersede(self, old_id) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute("UPDATE memories SET status = 'superseded' WHERE id = %s", (old_id,))
        await self.conn.commit()

    async def forget(self, memory_id) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute("UPDATE memories SET status = 'forgotten' WHERE id = %s", (memory_id,))
        await self.conn.commit()

    async def update_confidence(self, memory_id, delta: float) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE memories
                SET confidence = GREATEST(0, LEAST(1, confidence + %s))
                WHERE id = %s
                """,
                (delta, memory_id),
            )
        await self.conn.commit()

    async def bump_access(self, memory_ids) -> None:
        if not memory_ids:
            return
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1, last_accessed_at = now()
                WHERE id = ANY(%s::uuid[])
                """,
                (memory_ids,),
            )
        await self.conn.commit()

    async def get_by_id(self, memory_id):
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, user_id, type, content, embedding, confidence, status,
                       source_session_id, supersedes_id, metadata, access_count, last_accessed_at, created_at
                FROM memories WHERE id = %s
                """,
                (memory_id,),
            )
            row = await cur.fetchone()
        return self._to_memory(row) if row else None

    async def count_active(self, user_id) -> dict[str, int]:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                SELECT type, COUNT(*) FROM memories
                WHERE user_id = %s AND status = 'active'
                GROUP BY type
                """,
                (user_id,),
            )
            rows = await cur.fetchall()
        return {row[0]: row[1] for row in rows}

    async def set_conflict(self, left_id, right_id, reason: str) -> None:
        async with self.conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE memories
                SET metadata = metadata || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps({"conflict_with": str(right_id), "conflict_reason": reason}), left_id),
            )
            await cur.execute(
                """
                UPDATE memories
                SET metadata = metadata || %s::jsonb
                WHERE id = %s
                """,
                (json.dumps({"conflict_with": str(left_id), "conflict_reason": reason}), right_id),
            )
        await self.conn.commit()

    def _vector_literal(self, embedding: list[float]) -> str:
        return "[" + ",".join(str(x) for x in embedding) + "]"

    def _to_memory(self, row) -> Memory:
        metadata = row[9] if isinstance(row[9], dict) else {}
        return Memory(
            id=row[0],
            user_id=row[1],
            type=MemoryType(row[2]),
            content=row[3],
            embedding=list(row[4]),
            confidence=float(row[5]),
            status=MemoryStatus(row[6]),
            source_session_id=row[7],
            supersedes_id=row[8],
            metadata=metadata,
            access_count=int(row[10]),
            last_accessed_at=row[11],
            created_at=row[12],
        )
