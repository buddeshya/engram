from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID


class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT = "fact"
    DECISION = "decision"
    CORRECTION = "correction"
    EPISODIC = "episodic"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


@dataclass
class MemoryCandidate:
    type: MemoryType
    content: str
    confidence: float = 0.7
    metadata: dict = field(default_factory=dict)


@dataclass
class Memory:
    id: UUID
    user_id: UUID
    type: MemoryType
    content: str
    embedding: list[float]
    confidence: float
    status: MemoryStatus
    source_session_id: UUID | None
    supersedes_id: UUID | None
    metadata: dict
    access_count: int
    last_accessed_at: datetime | None
    created_at: datetime
