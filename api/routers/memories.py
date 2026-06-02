from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from api.dependencies import get_memory_service, get_user_id
from api.schemas import MemoryListResponse, MemoryResponse, UpdateMemoryRequest

router = APIRouter()


def _memory_to_response(memory):
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        type=memory.type.value,
        content=memory.content,
        confidence=memory.confidence,
        status=memory.status.value,
        source_session_id=memory.source_session_id,
        supersedes_id=memory.supersedes_id,
        metadata=memory.metadata,
        access_count=memory.access_count,
        last_accessed_at=memory.last_accessed_at,
        created_at=memory.created_at,
    )


@router.get("", response_model=MemoryListResponse)
async def list_memories(memory_service=Depends(get_memory_service), user_id: UUID = Depends(get_user_id)):
    result = await memory_service.list_active(user_id)
    return MemoryListResponse(
        memories=[_memory_to_response(m) for m in result["memories"]],
        total=result["total"],
        by_type=result["by_type"],
    )


@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    body: UpdateMemoryRequest,
    memory_service=Depends(get_memory_service),
    user_id: UUID = Depends(get_user_id),
):
    try:
        updated = await memory_service.update_content(memory_id, user_id, body.content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return _memory_to_response(updated)


@router.delete("/{memory_id}", status_code=204)
async def delete_memory(
    memory_id: UUID,
    memory_service=Depends(get_memory_service),
    user_id: UUID = Depends(get_user_id),
):
    try:
        await memory_service.forget(memory_id, user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)
