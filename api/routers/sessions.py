from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.dependencies import get_message_repo, get_session_repo, get_session_service, get_user_id
from api.schemas import CreateSessionRequest, EndSessionRequest, MessageResponse, SessionResponse

router = APIRouter()


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body: CreateSessionRequest,
    session_repo=Depends(get_session_repo),
    user_id: UUID = Depends(get_user_id),
):
    session = await session_repo.create(user_id=user_id, title=body.title)
    return SessionResponse(**session)


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    session_repo=Depends(get_session_repo),
    user_id: UUID = Depends(get_user_id),
):
    rows = await session_repo.list_by_user(user_id)
    return [SessionResponse(**row) for row in rows]


@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_session_messages(
    session_id: UUID,
    session_repo=Depends(get_session_repo),
    message_repo=Depends(get_message_repo),
    user_id: UUID = Depends(get_user_id),
):
    session = await session_repo.get_by_id(session_id)
    if session is None or session["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    rows = await message_repo.get_by_session(session_id)
    return [MessageResponse(**row) for row in rows]


@router.post("/{session_id}/end", status_code=202)
async def end_session(
    session_id: UUID,
    body: EndSessionRequest,
    session_repo=Depends(get_session_repo),
    session_service=Depends(get_session_service),
    user_id: UUID = Depends(get_user_id),
):
    session = await session_repo.get_by_id(session_id)
    if session is None or session["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Session not found")
    if body.generate_episodic:
        await session_service.generate_episodic_memory(session_id, user_id)
    return Response(status_code=202)
