from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str
    memory_count: int


class CreateSessionRequest(BaseModel):
    title: str | None = None


class SessionResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str | None
    summary: str | None
    summary_turn_count: int
    created_at: datetime
    updated_at: datetime


class ChatRequest(BaseModel):
    message: str


class EndSessionRequest(BaseModel):
    generate_episodic: bool = True


class MessageResponse(BaseModel):
    id: UUID
    session_id: UUID
    role: str
    content: str
    created_at: datetime


class MemoryResponse(BaseModel):
    id: UUID
    user_id: UUID
    type: str
    content: str
    confidence: float
    status: str
    source_session_id: UUID | None
    supersedes_id: UUID | None
    metadata: dict
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime


class MemoryListResponse(BaseModel):
    memories: list[MemoryResponse]
    total: int
    by_type: dict[str, int]


class UpdateMemoryRequest(BaseModel):
    content: str
