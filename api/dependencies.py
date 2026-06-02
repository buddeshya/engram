from uuid import UUID

from fastapi import Depends, HTTPException, Request
from openai import AsyncOpenAI

from config import settings
from db import connection
from repositories.memory_repository import MemoryRepository
from repositories.message_repository import MessageRepository
from repositories.session_repository import SessionRepository
from services.chat_service import ChatService
from services.memory_service import MemoryService
from services.session_service import SessionService


def get_user_id() -> UUID:
    if not settings.user_id:
        raise HTTPException(status_code=503, detail="USER_ID not configured, run scripts/seed_user.py")
    return UUID(settings.user_id)


def get_openai_client(request: Request) -> AsyncOpenAI:
    return request.app.state.openai


async def get_db_conn():
    async with connection.get_conn() as conn:
        yield conn


async def get_memory_repo(conn=Depends(get_db_conn)):
    return MemoryRepository(conn)


async def get_message_repo(conn=Depends(get_db_conn)):
    return MessageRepository(conn)


async def get_session_repo(conn=Depends(get_db_conn)):
    return SessionRepository(conn)


async def get_chat_service(
    memory_repo=Depends(get_memory_repo),
    session_repo=Depends(get_session_repo),
    message_repo=Depends(get_message_repo),
    openai_client=Depends(get_openai_client),
):
    return ChatService(memory_repo, session_repo, message_repo, openai_client, settings)


async def get_memory_service(
    memory_repo=Depends(get_memory_repo),
    openai_client=Depends(get_openai_client),
):
    return MemoryService(memory_repo, openai_client, settings)


async def get_session_service(
    session_repo=Depends(get_session_repo),
    message_repo=Depends(get_message_repo),
    memory_repo=Depends(get_memory_repo),
    openai_client=Depends(get_openai_client),
):
    return SessionService(session_repo, message_repo, memory_repo, openai_client, settings)
