import asyncio
import statistics
import time
import uuid
from dataclasses import dataclass

from openai import AsyncOpenAI

from config import settings
from db.connection import close_pool, get_conn, init_pool
from memory.types import MemoryCandidate, MemoryType
from repositories.memory_repository import MemoryRepository
from services.chat_service import ChatService
from repositories.session_repository import SessionRepository
from repositories.message_repository import MessageRepository


@dataclass
class TimingResult:
    p50_ms: float
    samples_ms: list[float]


def _unit_vector(seed: int, dim: int = 1536) -> list[float]:
    # Lightweight deterministic pseudo-random vector.
    values = [((seed * (i + 17)) % 1000) / 1000.0 for i in range(dim)]
    norm = sum(v * v for v in values) ** 0.5
    if norm == 0:
        return [0.0] * dim
    return [v / norm for v in values]


async def _measure_context_latency(chat_service: ChatService, user_id, session_id, n: int = 20) -> TimingResult:
    samples = []
    for i in range(n):
        t0 = time.perf_counter()
        await chat_service.retrieve_context(user_id, session_id, f"benchmark-query-{i}")
        samples.append((time.perf_counter() - t0) * 1000)
    return TimingResult(p50_ms=statistics.median(samples), samples_ms=samples)


async def main():
    await init_pool(settings.database_url)
    openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async with get_conn() as conn:
        memory_repo = MemoryRepository(conn)
        session_repo = SessionRepository(conn)
        message_repo = MessageRepository(conn)

        # Single-user benchmark scope.
        if not settings.user_id:
            raise RuntimeError("USER_ID not set. Run scripts/seed_user.py first.")
        user_id = uuid.UUID(settings.user_id)
        session = await session_repo.create(user_id=user_id, title="benchmark")
        session_id = session["id"]
        chat_service = ChatService(memory_repo, session_repo, message_repo, openai, settings)

        baseline = await _measure_context_latency(chat_service, user_id, session_id, n=20)
        print(f"Baseline p50 retrieve_context: {baseline.p50_ms:.2f} ms")

        # Seed 1000 semantic memories without LLM writes.
        for i in range(1000):
            candidate = MemoryCandidate(
                type=MemoryType.FACT,
                content=f"Benchmark memory #{i}",
                confidence=0.75,
            )
            await memory_repo.insert(candidate, _unit_vector(i), user_id=user_id, session_id=session_id)

        loaded = await _measure_context_latency(chat_service, user_id, session_id, n=20)
        delta = loaded.p50_ms - baseline.p50_ms
        print(f"Loaded p50 retrieve_context (1000 memories): {loaded.p50_ms:.2f} ms")
        print(f"Delta: {delta:.2f} ms")
        print("PASS" if delta <= 200 else "FAIL")

    await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
