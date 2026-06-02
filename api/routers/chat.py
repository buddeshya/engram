from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from agent.orchestrator import Orchestrator
from api.dependencies import (
    get_chat_service,
    get_memory_service,
    get_message_repo,
    get_session_repo,
    get_session_service,
    get_user_id,
)
from api.schemas import ChatRequest

router = APIRouter()


@router.post("/{session_id}/chat")
async def chat(
    session_id: UUID,
    body: ChatRequest,
    chat_service=Depends(get_chat_service),
    memory_service=Depends(get_memory_service),
    session_service=Depends(get_session_service),
    message_repo=Depends(get_message_repo),
    session_repo=Depends(get_session_repo),
    user_id: UUID = Depends(get_user_id),
):
    session = await session_repo.get_by_id(session_id)
    if session is None or session["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = Orchestrator(
        chat_service=chat_service,
        memory_service=memory_service,
        session_service=session_service,
        message_repo=message_repo,
        session_repo=session_repo,
    )

    async def event_generator():
        async for delta in orchestrator.handle_chat_stream(
            session_id=session_id,
            user_id=user_id,
            user_message=body.message,
        ):
            yield {"data": delta}
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())
