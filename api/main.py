import logging
import sys
from contextlib import asynccontextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI

from api.dependencies import get_user_id
from api.routers.chat import router as chat_router
from api.routers.memories import router as memories_router
from api.routers.sessions import router as sessions_router
from api.schemas import HealthResponse
from config import settings
from db import connection
from repositories.memory_repository import MemoryRepository
from services.memory_service import MemoryService, apply_md_edits


@asynccontextmanager
async def lifespan(app: FastAPI):
    await connection.init_pool(settings.database_url)
    app.state.openai = AsyncOpenAI(api_key=settings.openai_api_key)

    if settings.user_id:
        async with connection.get_conn() as conn:
            memory_repo = MemoryRepository(conn)
            service = MemoryService(memory_repo, app.state.openai, settings)
            await apply_md_edits(memory_repo, service, get_user_id(), settings.memory_md_path)
            await service.sync_memory_md(get_user_id())
    yield
    await connection.close_pool()


app = FastAPI(title="Engram", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router, prefix="/sessions", tags=["sessions"])
app.include_router(chat_router, prefix="/sessions", tags=["chat"])
app.include_router(memories_router, prefix="/memories", tags=["memories"])


@app.get("/health", response_model=HealthResponse)
async def health():
    async with connection.get_conn() as conn:
        repo = MemoryRepository(conn)
        try:
            user_id = get_user_id()
        except Exception:
            return HealthResponse(status="ok", database="connected", memory_count=0)
        counts = await repo.count_active(user_id)
        return HealthResponse(status="ok", database="connected", memory_count=sum(counts.values()))
