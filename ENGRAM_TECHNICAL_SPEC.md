# Engram — Technical Specification (v2)
> Conversational agent with a four-tier persistent memory architecture.
> Pass this document to Cursor as the single source of truth for implementation.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack & Dependencies](#2-tech-stack--dependencies)
3. [Project Structure](#3-project-structure)
4. [Environment Configuration](#4-environment-configuration)
5. [Docker Setup](#5-docker-setup)
6. [Database Schema & Migrations](#6-database-schema--migrations)
7. [Architecture: Four-Tier Memory](#7-architecture-four-tier-memory)
8. [Layer Responsibilities](#8-layer-responsibilities)
9. [Module Specifications — Repositories](#9-module-specifications--repositories)
10. [Module Specifications — Services](#10-module-specifications--services)
11. [Module Specifications — Memory](#11-module-specifications--memory)
12. [Module Specifications — Agent](#12-module-specifications--agent)
13. [Module Specifications — API](#13-module-specifications--api)
14. [Module Specifications — UI & Scripts](#14-module-specifications--ui--scripts)
15. [Test Specifications](#15-test-specifications)
16. [API Reference](#16-api-reference)
17. [Key Design Decisions & Tradeoffs](#17-key-design-decisions--tradeoffs)
18. [Setup & Run Instructions](#18-setup--run-instructions)

---

## 1. Project Overview

Engram is a conversational AI agent with **four-tier persistent memory** across sessions. A user can have multiple conversation sessions; all memory is user-scoped and shared across sessions. Starting a new session never resets what was learned.

**Memory tiers (bounded independently — this is the whole design):**
- **Working memory** — last N turns verbatim (in-session, resets per session)
- **Session summary** — rolling compressed summary of current session (~200 tok cap)
- **Semantic memory** — preferences, facts, decisions, corrections (cross-session, top-5)
- **Episodic memory** — one-paragraph summaries of past sessions (cross-session, top-3)

**Core thesis:** Intelligence lives on the *write path*. Selectivity at extraction time keeps the store clean so retrieval stays precise. Every tier is independently bounded so the prompt never grows, first-token latency stays flat, and retrieval stays accurate — simultaneously, with no tradeoff.

**Latency constraint:** p50 first-token latency at turn 1,000 must be within 200ms of p50 at turn 1. Proved by `scripts/benchmark.py`.

**Stack:** Python 3.11 · FastAPI · Streamlit · PostgreSQL + pgvector · OpenAI API

---

## 2. Tech Stack & Dependencies

### pyproject.toml

```toml
[project]
name = "engram"
version = "2.0.0"
description = "Conversational agent with four-tier persistent memory"
requires-python = ">=3.11"

dependencies = [
    # API
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.29.0",

    # Streamlit UI
    "streamlit>=1.35.0",
    "httpx>=0.27.0",

    # Database
    "psycopg[async,binary]>=3.1.19",
    "psycopg-pool>=3.2.2",
    "pgvector>=0.3.2",
    "alembic>=1.13.1",

    # Config & validation
    "pydantic>=2.7.0",
    "pydantic-settings>=2.2.1",
    "python-dotenv>=1.0.1",

    # LLM
    "openai>=1.30.0",
]

[project.optional-dependencies]
test = [
    "pytest>=8.2.0",
    "pytest-asyncio>=0.23.6",
    "pytest-cov>=5.0.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

---

## 3. Project Structure

```
engram/
├── docker-compose.yml
├── pyproject.toml
├── .env.example
├── .env                              # gitignored
├── README.md
├── config.py                         # Settings (pydantic-settings), single source of truth
│
├── alembic/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/
│       ├── 001_initial_schema.py
│       └── 002_episodic_and_session_summary.py
│
├── db/
│   ├── __init__.py
│   └── connection.py                 # async pool init, pgvector registration
│
├── repositories/                     # DATA ACCESS LAYER — pure SQL, zero business logic
│   ├── __init__.py
│   ├── base.py                       # BaseRepository: execute/fetchrow/fetchall helpers
│   ├── user_repository.py
│   ├── session_repository.py
│   ├── message_repository.py
│   └── memory_repository.py          # replaces memory/store.py
│
├── services/                         # BUSINESS LOGIC LAYER — orchestrates repos + memory modules
│   ├── __init__.py
│   ├── chat_service.py               # four-tier context retrieval + prompt assembly
│   ├── memory_service.py             # extract → resolve → persist + md sync
│   └── session_service.py            # session lifecycle, summary refresh, episodic generation
│
├── memory/                           # LLM-POWERED MEMORY OPS — pure LLM logic, no SQL
│   ├── __init__.py
│   ├── types.py                      # Memory, MemoryCandidate, all enums
│   ├── retriever.py                  # embed + dual pgvector search (semantic + episodic)
│   ├── extractor.py                  # LLM: (user, assistant) → list[MemoryCandidate]
│   ├── resolver.py                   # conflict classification + resolution
│   └── md_sync.py                    # pure formatter: list[Memory] → markdown string
│
├── agent/
│   ├── __init__.py
│   ├── prompt.py                     # four-tier prompt templates + assembly
│   └── orchestrator.py              # read_turn() stream + process_turn_bg() background
│
├── api/
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, lifespan, router mount
│   ├── dependencies.py               # Depends() for conn, openai, all services
│   ├── schemas.py                    # all Pydantic request + response models
│   └── routers/
│       ├── __init__.py
│       ├── sessions.py
│       ├── chat.py                   # SSE streaming endpoint
│       └── memories.py
│
├── ui/
│   ├── app.py
│   └── components/
│       ├── sidebar.py
│       ├── chat.py
│       └── memory_viewer.py
│
├── scripts/
│   ├── seed_user.py
│   └── benchmark.py
│
└── tests/
    ├── conftest.py
    ├── repositories/
    │   ├── test_memory_repository.py
    │   ├── test_session_repository.py
    │   └── test_message_repository.py
    ├── services/
    │   ├── test_memory_service.py
    │   ├── test_session_service.py
    │   └── test_chat_service.py
    └── memory/
        ├── test_extractor.py
        ├── test_retriever.py
        └── test_resolver.py
```

---

## 4. Environment Configuration

### config.py (root level, not inside any package)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # OpenAI
    openai_api_key: str
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str

    # User
    user_id: str = ""

    # Memory tuning
    memory_top_k: int = 5              # semantic memories injected per turn
    episodic_top_k: int = 3            # episodic memories injected per turn
    memory_similarity_threshold: float = 0.30
    context_window_turns: int = 10     # verbatim working memory turns
    summary_refresh_every_k: int = 10  # turns between rolling summary refreshes

    # Storage
    memory_md_path: str = "./memory.md"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_base_url: str = "http://localhost:8000"

    class Config:
        env_file = ".env"

settings = Settings()
```

### .env.example

```env
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
DATABASE_URL=postgresql+psycopg://agent:agent@localhost:5432/memory_agent
USER_ID=
MEMORY_TOP_K=5
EPISODIC_TOP_K=3
MEMORY_SIMILARITY_THRESHOLD=0.30
CONTEXT_WINDOW_TURNS=10
SUMMARY_REFRESH_EVERY_K=10
MEMORY_MD_PATH=./memory.md
API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=http://localhost:8000
```

---

## 5. Docker Setup

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent
      POSTGRES_DB: memory_agent
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d memory_agent"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

---

## 6. Database Schema & Migrations

### Migration 001 — initial_schema.py

```python
def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.execute("""
        CREATE TYPE memory_type AS ENUM
        ('preference', 'fact', 'decision', 'correction', 'episodic')
    """)
    op.execute("""
        CREATE TYPE memory_status AS ENUM ('active', 'superseded', 'forgotten')
    """)

    op.execute("""
        CREATE TABLE users (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    op.execute("""
        CREATE TABLE sessions (
            id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title      TEXT,
            summary    TEXT,                        -- rolling session summary
            summary_turn_count INTEGER NOT NULL DEFAULT 0,
            summary_updated_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_sessions_user_updated ON sessions(user_id, updated_at DESC)")

    op.execute("""
        CREATE TABLE messages (
            id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id  UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        VARCHAR(16) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content     TEXT NOT NULL,
            token_count INTEGER,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("CREATE INDEX idx_messages_session_time ON messages(session_id, created_at ASC)")

    op.execute("""
        CREATE TABLE memories (
            id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            source_session_id UUID REFERENCES sessions(id) ON DELETE SET NULL,
            supersedes_id     UUID REFERENCES memories(id) ON DELETE SET NULL,
            type              memory_type NOT NULL,
            content           TEXT NOT NULL,
            embedding         vector(1536) NOT NULL,
            confidence        FLOAT NOT NULL DEFAULT 0.7
                                  CHECK (confidence BETWEEN 0 AND 1),
            status            memory_status NOT NULL DEFAULT 'active',
            access_count      INTEGER NOT NULL DEFAULT 0,
            last_accessed_at  TIMESTAMPTZ,
            metadata          JSONB NOT NULL DEFAULT '{}',
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE INDEX idx_memories_embedding
        ON memories USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    op.execute("CREATE INDEX idx_memories_user_status ON memories(user_id, status)")
    op.execute("CREATE INDEX idx_memories_user_type_status ON memories(user_id, type, status)")
    op.execute("CREATE INDEX idx_memories_source_session ON memories(source_session_id) WHERE source_session_id IS NOT NULL")
    op.execute("CREATE INDEX idx_memories_supersedes ON memories(supersedes_id) WHERE supersedes_id IS NOT NULL")

def downgrade():
    op.execute("DROP TABLE IF EXISTS memories CASCADE")
    op.execute("DROP TABLE IF EXISTS messages CASCADE")
    op.execute("DROP TABLE IF EXISTS sessions CASCADE")
    op.execute("DROP TABLE IF EXISTS users CASCADE")
    op.execute("DROP TYPE IF EXISTS memory_status")
    op.execute("DROP TYPE IF EXISTS memory_type")
```

**Note:** The `episodic` memory type and `summary` columns on `sessions` are included in migration 001 from the start. There is no migration 002 — design them in from day one rather than adding later.

The five memory types:
- `preference` — sticky, latest-wins on conflict
- `fact` — updatable, temporal supersession
- `decision` — project-scoped, sticky while project active
- `correction` — highest injection priority, never decays
- `episodic` — one-paragraph session summaries, cross-session recall

---

## 7. Architecture: Four-Tier Memory

Every turn assembles a bounded prompt from exactly four sources. Each tier has an independent cap so the total never grows with conversation length, store size, or session count.

```
┌─────────────────────────────────────────────────────┐
│                  BOUNDED PROMPT (~1450 tok max)      │
│                                                      │
│  System prompt           ~150 tok  (fixed)           │
│  Semantic memories       ≤100 tok  (top-5 × ~20)    │
│  Episodic memories       ≤150 tok  (top-3 × ~50)    │
│  Session summary         ≤200 tok  (rolling, capped) │
│  Working memory          ≤800 tok  (last 10 turns)   │
│  Current message          ~50 tok  (current)         │
└─────────────────────────────────────────────────────┘
```

**Read path (synchronous — critical path to first token):**
1. Embed user query (one OpenAI call, constant cost)
2. pgvector scan → semantic top-5 (HNSW, sub-ms)
3. pgvector scan → episodic top-3 (same index, sub-ms)
4. Fetch session summary (single indexed row)
5. Fetch last N messages (working memory)
6. Assemble bounded prompt
7. Stream from OpenAI

**Write path (async — never touches first-token latency):**
1. Save user + assistant messages
2. Touch `sessions.updated_at`
3. Auto-title session (first turn only)
4. Refresh rolling summary (every K turns)
5. Extract candidate memories (LLM call — most turns return [])
6. Resolve each candidate (dedup/supersede/merge)
7. Persist new memories
8. Sync memory.md
9. On session end: generate episodic memory

### Prompt format (four-tier injection)

```
[SYSTEM]
You are a helpful, concise AI assistant with memory across conversations.
Use stored context naturally — do not mention that you are using stored memory.

<semantic_context>
Preferences: prefers tabs; concise answers without preamble
Facts: works at Google; uses macOS; building a REST API in Python
Decisions: Atlas project uses PostgreSQL
Corrections: do not suggest switching to JavaScript
</semantic_context>

<past_sessions>
Session 2024-01-10: Implemented the auth module. Set JWT expiry to 30 min.
  Decided to store refresh tokens in Redis with 7-day TTL. Hit a bug
  where tokens weren't invalidated on logout — resolved with a blocklist.
Session 2024-01-08: Scaffolded the project. Chose FastAPI + PostgreSQL.
  Discussed DB schema for users, sessions, tokens.
</past_sessions>

<current_session>
Earlier in this session: Reviewed the login endpoint. Discussed bcrypt
  cost factor — settled on 12. Started looking at the token refresh flow.
</current_session>

[CONVERSATION HISTORY — last 10 turns verbatim]
User: ...
Assistant: ...
...

User: Can you review this endpoint for auth issues?
```

### Rolling session summary

The rolling summary compresses turns that fall outside the working memory window. Updated every K turns (default: 10). Uses the previous summary + the K newest turns as LLM input. Output is capped in the prompt to ~200 tokens. Never expands beyond that cap.

Prompt for summary refresh:
```
Summarize the key topics, decisions, and context from this conversation so far.
Be concise — your output will be used as context for future turns.
Max 200 words.

Previous summary:
{previous_summary or "None"}

New turns to incorporate:
{last_k_turns_formatted}
```

### Episodic memory generation

Generated once per session, on session end or inactivity (triggered by the client or a background check). LLM generates a 2-3 sentence paragraph capturing the session's work, decisions, and outcomes. This is embedded and stored as a memory with `type='episodic'`. Retrieved in future sessions via the same pgvector index.

Prompt for episodic generation:
```
Write a 2-3 sentence summary of this conversation session for future reference.
Focus on: what was built or decided, key outcomes, open questions.
Write in third person past tense. Be specific — include names, values, and decisions.

Session summary: {session.summary}
Message count: {count}
```

---

## 8. Layer Responsibilities

| Layer | Location | Knows about | Does NOT know about |
|-------|----------|-------------|---------------------|
| Repository | `repositories/` | SQL, psycopg3, table schemas | Business logic, OpenAI |
| Service | `services/` | Repositories, memory modules | Raw SQL, HTTP |
| Memory | `memory/` | OpenAI SDK, pgvector | Repositories, HTTP |
| Agent | `agent/` | Services, OpenAI streaming | SQL, HTTP routing |
| API | `api/` | Services, FastAPI, Pydantic | SQL, OpenAI directly |

This boundary is strict. If a repository method contains business logic, move it to a service. If a service contains raw SQL, move it to a repository.

---

## 9. Module Specifications — Repositories

### 9.1 `repositories/base.py`

**Purpose:** Thin base class providing typed helpers over psycopg3's cursor API. All repositories inherit from this.

```python
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from uuid import UUID

class BaseRepository:
    def __init__(self, conn: AsyncConnection):
        self.conn = conn

    async def fetchrow(self, query: str, *args) -> dict | None:
        """Execute query, return first row as dict or None."""

    async def fetchall(self, query: str, *args) -> list[dict]:
        """Execute query, return all rows as list of dicts."""

    async def execute(self, query: str, *args) -> None:
        """Execute a statement with no return value."""

    async def fetchval(self, query: str, *args) -> any:
        """Execute query, return first column of first row."""
```

Always use `row_factory=dict_row` on the cursor so all results are dicts, not tuples.

---

### 9.2 `repositories/user_repository.py`

```python
class UserRepository(BaseRepository):

    async def create(self) -> dict:
        """INSERT INTO users DEFAULT VALUES RETURNING *"""

    async def get_by_id(self, user_id: UUID) -> dict | None:
        """SELECT * FROM users WHERE id = $1"""
```

---

### 9.3 `repositories/session_repository.py`

```python
class SessionRepository(BaseRepository):

    async def create(self, user_id: UUID, title: str | None = None) -> dict:
        """INSERT INTO sessions (user_id, title) VALUES ($1, $2) RETURNING *"""

    async def get_by_id(self, session_id: UUID) -> dict | None:
        """SELECT * FROM sessions WHERE id = $1"""

    async def list_by_user(self, user_id: UUID) -> list[dict]:
        """
        SELECT s.*, COUNT(m.id) as message_count
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        WHERE s.user_id = $1
        GROUP BY s.id
        ORDER BY s.updated_at DESC
        """

    async def touch(self, session_id: UUID) -> None:
        """UPDATE sessions SET updated_at = now() WHERE id = $1"""

    async def update_title(self, session_id: UUID, title: str) -> None:
        """UPDATE sessions SET title = $2 WHERE id = $1 AND title IS NULL"""
        # Only updates if title is still NULL — first message sets it, never overwritten.

    async def update_summary(
        self,
        session_id: UUID,
        summary: str,
        turn_count: int,
    ) -> None:
        """
        UPDATE sessions SET
            summary = $2,
            summary_turn_count = $3,
            summary_updated_at = now()
        WHERE id = $1
        """

    async def delete(self, session_id: UUID) -> None:
        """DELETE FROM sessions WHERE id = $1"""
```

---

### 9.4 `repositories/message_repository.py`

```python
class MessageRepository(BaseRepository):

    async def insert(
        self,
        session_id: UUID,
        role: str,
        content: str,
    ) -> dict:
        """INSERT INTO messages (session_id, role, content) VALUES ($1, $2, $3) RETURNING *"""

    async def get_by_session(self, session_id: UUID) -> list[dict]:
        """SELECT * FROM messages WHERE session_id = $1 ORDER BY created_at ASC"""

    async def get_recent_n(self, session_id: UUID, n: int) -> list[dict]:
        """
        SELECT * FROM (
          SELECT * FROM messages WHERE session_id = $1
          ORDER BY created_at DESC LIMIT $2
        ) sub ORDER BY created_at ASC
        """
        # Returns in chronological order (oldest first within the window).

    async def count_by_session(self, session_id: UUID) -> int:
        """SELECT COUNT(*) FROM messages WHERE session_id = $1"""
```

---

### 9.5 `repositories/memory_repository.py`

This is the largest repository. All SQL for the `memories` table lives here.

```python
from memory.types import Memory, MemoryCandidate, MemoryType, MemoryStatus

class MemoryRepository(BaseRepository):

    async def insert(
        self,
        candidate: MemoryCandidate,
        embedding: list[float],
        user_id: UUID,
        session_id: UUID | None,
        supersedes_id: UUID | None = None,
    ) -> Memory:
        """
        INSERT INTO memories
          (user_id, source_session_id, supersedes_id, type, content,
           embedding, confidence, metadata)
        VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
        RETURNING *
        """
        # Convert returned row to Memory dataclass.
        # embedding column returns numpy array from pgvector — call .tolist().

    async def get_active(
        self,
        user_id: UUID,
        exclude_types: list[MemoryType] | None = None,
    ) -> list[Memory]:
        """
        SELECT * FROM memories
        WHERE user_id = $1 AND status = 'active'
        [AND type != ALL($2)]
        ORDER BY confidence DESC, created_at DESC
        """
        # exclude_types useful for getting only semantic (exclude episodic) or vice versa.

    async def search_by_vector(
        self,
        user_id: UUID,
        embedding: list[float],
        types: list[MemoryType],
        top_k: int,
        threshold: float,
    ) -> list[tuple[Memory, float]]:
        """
        SELECT *, 1 - (embedding <=> $1::vector) AS similarity
        FROM memories
        WHERE user_id = $2
          AND status = 'active'
          AND type = ANY($3::memory_type[])
        ORDER BY embedding <=> $1::vector
        LIMIT $4
        """
        # Filter client-side: keep rows where similarity >= threshold.
        # Returns list of (Memory, similarity_score) tuples.
        # Over-fetch by 2x before threshold filter to avoid under-returning.

    async def get_active_by_type(
        self,
        user_id: UUID,
        memory_type: MemoryType,
    ) -> list[Memory]:
        """
        SELECT * FROM memories
        WHERE user_id = $1 AND type = $2 AND status = 'active'
        ORDER BY confidence DESC
        """

    async def get_by_id(self, memory_id: UUID) -> Memory | None:
        """SELECT * FROM memories WHERE id = $1"""

    async def supersede(self, old_id: UUID) -> None:
        """UPDATE memories SET status = 'superseded' WHERE id = $1"""

    async def forget(self, memory_id: UUID) -> None:
        """UPDATE memories SET status = 'forgotten' WHERE id = $1"""

    async def bump_access(self, memory_ids: list[UUID]) -> None:
        """
        UPDATE memories
        SET access_count = access_count + 1, last_accessed_at = now()
        WHERE id = ANY($1)
        """
        # Called async after retrieval — never on the critical path.

    async def update_confidence(self, memory_id: UUID, delta: float) -> None:
        """
        UPDATE memories
        SET confidence = GREATEST(0, LEAST(1, confidence + $2))
        WHERE id = $1
        """

    async def update_content(
        self,
        memory_id: UUID,
        new_content: str,
        new_embedding: list[float],
    ) -> Memory:
        """
        UPDATE memories
        SET content = $2, embedding = $3::vector
        WHERE id = $1
        RETURNING *
        """

    async def count_active(self, user_id: UUID) -> dict[str, int]:
        """
        SELECT type, COUNT(*) as count
        FROM memories
        WHERE user_id = $1 AND status = 'active'
        GROUP BY type
        """
        # Returns dict keyed by type name, e.g. {"preference": 3, "fact": 7, ...}
```

---

## 10. Module Specifications — Services

Services contain all business logic. They take repositories and memory module clients as constructor dependencies (injected via FastAPI's `Depends()`).

### 10.1 `services/chat_service.py`

**Purpose:** Assembles all four tiers of context for each turn. Provides the `retrieve_context()` and `build_prompt()` methods used by the orchestrator on the read path.

```python
from dataclasses import dataclass
from memory.types import Memory
from repositories.memory_repository import MemoryRepository
from repositories.session_repository import SessionRepository
from repositories.message_repository import MessageRepository
from memory.retriever import retrieve_dual
from agent.prompt import build_messages

@dataclass
class TurnContext:
    semantic_memories: list[Memory]
    episodic_memories: list[Memory]
    session_summary: str | None
    recent_messages: list[dict]     # [{"role": str, "content": str}, ...]

class ChatService:
    def __init__(
        self,
        memory_repo: MemoryRepository,
        session_repo: SessionRepository,
        message_repo: MessageRepository,
        openai_client: AsyncOpenAI,
        settings: Settings,
    ):
        ...

    async def retrieve_context(
        self,
        user_id: UUID,
        session_id: UUID,
        query: str,
    ) -> TurnContext:
        """
        Retrieves all four tiers of memory context for a single turn.

        Steps:
        1. Call retriever.retrieve_dual(query) → (semantic_memories, episodic_memories)
        2. session_repo.get_by_id(session_id) → extract .summary field
        3. message_repo.get_recent_n(session_id, settings.context_window_turns * 2)
           → format as [{"role": ..., "content": ...}]
        4. Fire-and-forget: bump_access on retrieved memory IDs
        5. Return TurnContext

        Never raises — if embedding fails, returns TurnContext with empty memory lists.
        """

    async def build_prompt(
        self,
        query: str,
        context: TurnContext,
    ) -> list[dict]:
        """
        Calls agent.prompt.build_messages() with all four tiers.
        Returns the full messages list ready for the OpenAI chat call.
        """
```

---

### 10.2 `services/memory_service.py`

**Purpose:** Runs the full write-path pipeline after each turn. Called from a FastAPI `BackgroundTask` — never blocks the read path.

```python
from memory.extractor import extract
from memory.resolver import resolve
from memory.retriever import embed
from memory.md_sync import format_memory_md
from repositories.memory_repository import MemoryRepository

class MemoryService:
    def __init__(
        self,
        memory_repo: MemoryRepository,
        openai_client: AsyncOpenAI,
        settings: Settings,
    ):
        ...

    async def process_turn(
        self,
        user_message: str,
        assistant_message: str,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """
        The full extraction + resolution + persistence pipeline.
        Called async after each turn. Most invocations do nothing (extractor returns []).

        Steps:
        1. extract(user_message, assistant_message, openai_client, settings.chat_model)
           → list[MemoryCandidate] (usually [])
        2. For each candidate:
           a. embed(candidate.content, ...) → list[float]
           b. resolve(conn, candidate, embedding, user_id, session_id, ...) → None
              (resolve handles inserting/updating/ignoring internally)
        3. sync_memory_md(user_id)
        """

    async def sync_memory_md(self, user_id: UUID) -> None:
        """
        Fetches all active memories, formats to markdown via md_sync.format_memory_md(),
        writes to settings.memory_md_path.
        """

    async def forget(self, memory_id: UUID, user_id: UUID) -> None:
        """
        Sets memory status = 'forgotten'.
        Validates memory belongs to user_id before forgetting.
        Calls sync_memory_md after.
        """

    async def update_content(
        self,
        memory_id: UUID,
        user_id: UUID,
        new_content: str,
    ) -> Memory:
        """
        Re-embeds new_content, calls memory_repo.update_content().
        Validates ownership. Calls sync_memory_md after.
        Returns updated Memory.
        """

    async def list_active(self, user_id: UUID) -> dict:
        """
        Returns all active memories grouped by type.
        Used by the GET /memories endpoint.
        Returns: {"memories": list[Memory], "total": int, "by_type": dict[str, int]}
        """
```

---

### 10.3 `services/session_service.py`

**Purpose:** Manages the session lifecycle — creation, rolling summary refresh, auto-titling, and episodic memory generation on session end.

```python
from repositories.session_repository import SessionRepository
from repositories.message_repository import MessageRepository
from repositories.memory_repository import MemoryRepository
from memory.retriever import embed
from memory.types import MemoryCandidate, MemoryType

class SessionService:
    def __init__(
        self,
        session_repo: SessionRepository,
        message_repo: MessageRepository,
        memory_repo: MemoryRepository,
        openai_client: AsyncOpenAI,
        settings: Settings,
    ):
        ...

    async def create(self, user_id: UUID, title: str | None = None) -> dict:
        """Creates a new session row. Returns the full session dict."""

    async def auto_title(self, session_id: UUID, first_message: str) -> None:
        """
        Generates a 4-6 word title from the first user message via LLM.
        Only fires if session.title IS NULL (first turn only).

        LLM prompt:
          "Summarize this message in 4-6 words as a conversation title.
           Return only the title, no quotes, no punctuation at the end.
           Message: {first_message}"

        Calls session_repo.update_title() — which only updates if title is still NULL.
        """

    async def refresh_summary_if_needed(self, session_id: UUID) -> None:
        """
        Called from the write-path background task after every turn.

        Logic:
        1. session_repo.get_by_id(session_id) → get current summary_turn_count
        2. message_repo.count_by_session(session_id) → current total
        3. If (total - summary_turn_count) < settings.summary_refresh_every_k * 2: return early
        4. Fetch last (summary_refresh_every_k * 2) messages
        5. LLM call to generate new summary (see prompt below)
        6. session_repo.update_summary(session_id, new_summary, total)

        Summary refresh LLM prompt:
          SYSTEM: "You produce concise conversation summaries for an AI agent's memory."
          USER:
            "Previous summary:
             {session.summary or 'None — this is the first summary.'}

             New conversation turns to incorporate:
             {formatted_recent_turns}

             Write a new summary (max 200 words) capturing key topics,
             decisions, and context. Include specific names, values, and choices."
        """

    async def generate_episodic_memory(
        self,
        session_id: UUID,
        user_id: UUID,
    ) -> None:
        """
        Called when a session ends (explicit close or inactivity timeout).
        Generates a 2-3 sentence episodic paragraph, embeds it,
        stores as memory with type='episodic'.

        Steps:
        1. session_repo.get_by_id(session_id) → get summary + metadata
        2. message_repo.count_by_session(session_id) → message count
        3. LLM call to generate episodic paragraph:
           SYSTEM: "You produce episodic memory summaries for an AI agent."
           USER:
             "Session summary: {summary}
              Message count: {count}
              Session date: {created_at}

              Write 2-3 sentences capturing what happened, what was decided,
              and any open questions. Third person, past tense. Be specific."
        4. embedding = await embed(episodic_text, openai_client, settings.embedding_model)
        5. candidate = MemoryCandidate(type=MemoryType.EPISODIC, content=episodic_text, confidence=1.0)
        6. memory_repo.insert(candidate, embedding, user_id, session_id)

        If session has fewer than 4 messages: skip (not enough signal).
        """

    async def list_for_user(self, user_id: UUID) -> list[dict]:
        """Returns session_repo.list_by_user(user_id)."""

    async def delete(self, session_id: UUID, user_id: UUID) -> None:
        """Validates ownership, then session_repo.delete(). Cascade handles messages."""
```

---

## 11. Module Specifications — Memory

The `memory/` package contains LLM-powered logic only. No raw SQL. All database operations go through repositories, injected as parameters.

### 11.1 `memory/types.py`

```python
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID
from datetime import datetime

class MemoryType(str, Enum):
    PREFERENCE = "preference"
    FACT       = "fact"
    DECISION   = "decision"
    CORRECTION = "correction"
    EPISODIC   = "episodic"     # session-level summaries

class MemoryStatus(str, Enum):
    ACTIVE     = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN  = "forgotten"

class ResolutionLabel(str, Enum):
    NEW          = "NEW"
    DUPLICATE    = "DUPLICATE"
    UPDATE       = "UPDATE"
    CONTRADICTION = "CONTRADICTION"
    REFINEMENT   = "REFINEMENT"
    NOOP         = "NOOP"

SEMANTIC_TYPES = {MemoryType.PREFERENCE, MemoryType.FACT,
                  MemoryType.DECISION, MemoryType.CORRECTION}
EPISODIC_TYPES = {MemoryType.EPISODIC}

@dataclass
class Memory:
    id:                UUID
    user_id:           UUID
    type:              MemoryType
    content:           str
    embedding:         list[float]
    confidence:        float
    status:            MemoryStatus
    supersedes_id:     UUID | None
    source_session_id: UUID | None
    access_count:      int
    metadata:          dict
    last_accessed_at:  datetime | None
    created_at:        datetime

@dataclass
class MemoryCandidate:
    """Output of the extractor — not yet persisted."""
    type:       MemoryType
    content:    str
    confidence: float = 0.7
    metadata:   dict  = field(default_factory=dict)

@dataclass
class ResolutionResult:
    label:          ResolutionLabel
    reasoning:      str
    merged_content: str | None = None
```

---

### 11.2 `memory/retriever.py`

**Purpose:** Embeds a query string and runs two separate pgvector searches — one for semantic memories, one for episodic memories. Returns both lists independently so the prompt can inject them in separate sections.

```python
async def embed(
    text: str,
    openai_client: AsyncOpenAI,
    model: str,
) -> list[float]:
    """
    Calls openai_client.embeddings.create(input=text, model=model).
    Returns response.data[0].embedding as list[float].
    On failure: logs error, returns None. Callers must handle None gracefully.
    """

async def retrieve_dual(
    memory_repo: MemoryRepository,
    openai_client: AsyncOpenAI,
    user_id: UUID,
    query: str,
    semantic_top_k: int = 5,
    episodic_top_k: int = 3,
    threshold: float = 0.30,
    embedding_model: str = "text-embedding-3-small",
) -> tuple[list[Memory], list[Memory]]:
    """
    Single embed call, two pgvector scans.
    Returns (semantic_memories, episodic_memories).

    Steps:
    1. embedding = await embed(query, openai_client, embedding_model)
    2. If embedding is None: return ([], [])
    3. semantic = await memory_repo.search_by_vector(
           user_id, embedding,
           types=list(SEMANTIC_TYPES),
           top_k=semantic_top_k,
           threshold=threshold
       )
    4. episodic = await memory_repo.search_by_vector(
           user_id, embedding,
           types=[MemoryType.EPISODIC],
           top_k=episodic_top_k,
           threshold=threshold
       )
    5. return (semantic, episodic)

    The single embed call is the key efficiency: both scans reuse the same vector.
    Both scans use the HNSW index — sub-millisecond each.
    """
```

---

### 11.3 `memory/extractor.py`

**Purpose:** Given a single conversation turn, uses the LLM to extract zero or more durable memory candidates. Most turns produce an empty list.

**Extraction system prompt (use verbatim):**

```
You are a memory extraction system for a conversational AI assistant.

Given a conversation turn, identify durable facts about the user worth remembering.

STORE ONLY if the fact would:
  1. Still be true and relevant in a DIFFERENT conversation THREE MONTHS from now
  2. Meaningfully change how the assistant should behave in future conversations

DO NOT store:
  - The current question, task, or topic being discussed
  - Transient debugging context or specific error messages
  - Information derivable from common knowledge
  - Credentials, API keys, passwords, or any personal identifiers
  - Vague or uncertain impressions

NORMALIZE to a clean declarative third-person statement.
  Good: "Prefers tabs over spaces for Python indentation"
  Bad:  "User said tabs when I asked about indentation"

NEVER extract episodic memories — those are generated separately.

OUTPUT FORMAT: Return ONLY a valid JSON array. No markdown, no explanation.
Each element: {"type": "preference|fact|decision|correction", "content": "...", "confidence": 0.5-1.0}
Return [] if nothing is worth storing.

MEMORY TYPES:
  preference  — how the user likes things done (style, tools, format)
  fact        — stable info about the user (role, stack, project names)
  decision    — a choice made about a specific project or situation
  correction  — something the assistant should avoid (highest priority)
```

**User message for extraction:**
```
CONVERSATION TURN:
User: {user_message}
Assistant: {assistant_message}

Extract durable memories. Return [].
```

**Implementation:**
```python
async def extract(
    user_message: str,
    assistant_message: str,
    openai_client: AsyncOpenAI,
    model: str,
) -> list[MemoryCandidate]:
    """
    Returns a list of MemoryCandidate objects. Returns [] on any error.
    Parse response with json.loads(). On parse failure: log and return [].
    Validate each item has type, content, confidence.
    Skip any item with an unknown type (e.g. 'episodic' — never extracted here).
    Clamp confidence to [0.5, 1.0].
    """
```

---

### 11.4 `memory/resolver.py`

**Purpose:** For each `MemoryCandidate`, determines whether to insert, ignore, supersede, or merge against existing memories of the same type.

**Resolution algorithm:**
```python
async def resolve(
    memory_repo: MemoryRepository,
    openai_client: AsyncOpenAI,
    candidate: MemoryCandidate,
    candidate_embedding: list[float],
    user_id: UUID,
    session_id: UUID | None,
    model: str,
) -> None:
    """
    1. memory_repo.search_by_vector(user_id, candidate_embedding,
           types=[candidate.type], top_k=3, threshold=0.60)
    2. If no similar memories found: INSERT as NEW → return
    3. Take the most similar match. Make ONE LLM classification call.
    4. Act on result:
       - DUPLICATE:     memory_repo.update_confidence(existing.id, +0.05)
       - UPDATE:        new = memory_repo.insert(..., supersedes_id=existing.id)
                        memory_repo.supersede(existing.id)
       - CONTRADICTION: keep higher-confidence active; supersede the lower.
                        Write {"conflict_with": str(other.id)} to both metadata.
       - REFINEMENT:    memory_repo.update_content(existing.id, merged_content, new_embedding)
       - NOOP:          do nothing
    """
```

**Classification LLM system prompt:**
```
Classify the relationship between two memory statements about the same person.
Return ONLY valid JSON. No markdown.
```

**Classification user message:**
```
EXISTING: "{existing_content}"
NEW:      "{candidate_content}"

Classify as: DUPLICATE | UPDATE | CONTRADICTION | REFINEMENT | NOOP

- DUPLICATE:     essentially the same information
- UPDATE:        new supersedes old temporally (job change, preference change)
- CONTRADICTION: logically incompatible, no temporal explanation
- REFINEMENT:    new adds specificity to old without replacing it
- NOOP:          related but neither updates nor contradicts

Return: {"classification": "...", "reasoning": "one sentence",
         "merged_content": "if REFINEMENT: merged statement, else null"}
```

---

### 11.5 `memory/md_sync.py`

**Purpose:** Pure formatter — takes a list of Memory objects and returns a markdown string. No I/O. The caller (MemoryService) handles file writing.

```python
def format_memory_md(memories: list[Memory], updated_at: datetime) -> str:
    """
    Groups active memories by type. Returns markdown string.

    Format:
    # Memory — last updated: {updated_at} UTC

    ## Preferences
    - Prefers tabs over spaces for Python indentation  <!-- id:{id} confidence:{c:.2f} -->

    ## Facts
    - Works at Google on the Search infrastructure team  <!-- id:{id} confidence:{c:.2f} -->

    ## Decisions
    - Atlas project uses PostgreSQL for persistence  <!-- id:{id} confidence:{c:.2f} -->

    ## Corrections
    - Do not suggest switching to JavaScript  <!-- id:{id} confidence:{c:.2f} -->

    ## Past sessions (episodic)
    - {episodic_content}  <!-- id:{id} -->

    The HTML comments with id: are machine-readable for sync-back.
    Episodic memories are shown but never sync'd back — they are read-only.
    """
```

---

## 12. Module Specifications — Agent

### 12.1 `agent/prompt.py`

**Purpose:** Pure string functions. No I/O, no imports from repositories or services.

```python
SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant with memory across conversations. "
    "You remember things users tell you. Use that context naturally — "
    "do not mention that you are using stored memory. "
    "Be concise unless asked for detail."
)

SEMANTIC_TYPES_ORDER = ["correction", "preference", "fact", "decision"]

def format_semantic_context(memories: list[Memory]) -> str:
    """
    Groups by type in SEMANTIC_TYPES_ORDER priority (corrections first).
    Returns <semantic_context>...</semantic_context> block or "" if empty.

    Format:
    <semantic_context>
    Corrections: do not suggest switching to JavaScript
    Preferences: prefers tabs; concise answers
    Facts: works at Google; uses macOS
    Decisions: Atlas project uses PostgreSQL
    </semantic_context>
    """

def format_episodic_context(memories: list[Memory]) -> str:
    """
    Returns <past_sessions>...</past_sessions> block or "" if empty.
    One memory per line, prefixed with date extracted from metadata or created_at.

    Format:
    <past_sessions>
    2024-01-10: Implemented auth module. JWT 30min expiry. Refresh tokens in Redis.
    2024-01-08: Project scaffolded. FastAPI + PostgreSQL chosen.
    </past_sessions>
    """

def format_session_context(summary: str | None) -> str:
    """
    Returns <current_session>...</current_session> block or "" if summary is None/empty.
    """

def build_messages(
    semantic_context: str,
    episodic_context: str,
    session_context: str,
    history: list[dict],
    user_message: str,
) -> list[dict]:
    """
    Assembles the full messages list for the OpenAI chat call.

    Order:
    1. {"role": "system", "content": SYSTEM_PROMPT + assembled context blocks}
       Context blocks appended only if non-empty, in this order:
       semantic_context, episodic_context, session_context
    2. history items (the recent N turns)
    3. {"role": "user", "content": user_message}
    """
```

---

### 12.2 `agent/orchestrator.py`

**Purpose:** Coordinates a single conversational turn. Two functions: the synchronous read path (streaming generator) and the async write path (background task). Delegates all context assembly to `ChatService` and all memory pipeline work to `MemoryService` and `SessionService`.

```python
async def read_turn(
    user_message: str,
    session_id: UUID,
    user_id: UUID,
    chat_service: ChatService,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> AsyncGenerator[str, None]:
    """
    CRITICAL PATH — every line here affects first-token latency.

    Steps:
    1. context = await chat_service.retrieve_context(user_id, session_id, user_message)
    2. messages = await chat_service.build_prompt(user_message, context)
    3. stream = await openai_client.chat.completions.create(
           model=settings.chat_model, messages=messages, stream=True
       )
    4. async for chunk in stream:
           delta = chunk.choices[0].delta.content
           if delta: yield delta
    """

async def process_turn_bg(
    user_message: str,
    assistant_message: str,
    session_id: UUID,
    user_id: UUID,
    message_repo: MessageRepository,
    session_service: SessionService,
    memory_service: MemoryService,
) -> None:
    """
    WRITE PATH — runs after stream completes, never touches first-token latency.

    Steps:
    1. message_repo.insert(session_id, 'user', user_message)
    2. message_repo.insert(session_id, 'assistant', assistant_message)
    3. session_repo.touch(session_id)
    4. await session_service.auto_title(session_id, user_message)
    5. await session_service.refresh_summary_if_needed(session_id)
    6. await memory_service.process_turn(user_message, assistant_message, session_id, user_id)
    """
```

---

## 13. Module Specifications — API

### 13.1 `api/schemas.py`

All request and response Pydantic v2 models. Every API endpoint uses types from this file exclusively.

```python
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID
from datetime import datetime

# ── Requests ──────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    title: str | None = Field(None, max_length=200)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=32_000)

class UpdateMemoryRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=2_000)

class EndSessionRequest(BaseModel):
    """Signals the session has ended — triggers episodic memory generation."""
    generate_episodic: bool = True

# ── Responses ─────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    id: UUID
    created_at: datetime

class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:         UUID
    user_id:    UUID
    title:      str | None
    created_at: datetime
    updated_at: datetime

class SessionDetailResponse(SessionResponse):
    summary:       str | None     # current rolling summary
    message_count: int
    memory_count:  int            # total active memories for this user

class SessionListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:            UUID
    title:         str | None
    created_at:    datetime
    updated_at:    datetime
    message_count: int

class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:         UUID
    session_id: UUID
    role:       str
    content:    str
    created_at: datetime

class MemoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id:                UUID
    user_id:           UUID
    type:              str       # preference | fact | decision | correction | episodic
    content:           str
    confidence:        float
    status:            str
    source_session_id: UUID | None
    supersedes_id:     UUID | None
    access_count:      int
    last_accessed_at:  datetime | None
    created_at:        datetime

class MemoryListResponse(BaseModel):
    memories: list[MemoryResponse]
    total:    int
    by_type:  dict[str, int]     # {"preference": 3, "fact": 7, "episodic": 12, ...}

class HealthResponse(BaseModel):
    status:       str            # "ok"
    database:     str            # "connected"
    memory_count: int
```

---

### 13.2 `api/dependencies.py`

```python
from fastapi import Request, Depends, HTTPException
from psycopg import AsyncConnection
from openai import AsyncOpenAI
from uuid import UUID

from db.connection import get_conn as _get_conn
from repositories.user_repository import UserRepository
from repositories.session_repository import SessionRepository
from repositories.message_repository import MessageRepository
from repositories.memory_repository import MemoryRepository
from services.chat_service import ChatService
from services.memory_service import MemoryService
from services.session_service import SessionService
from config import settings

# ── Database ──────────────────────────────────────────────────────────────────

async def get_conn(request: Request) -> AsyncGenerator[AsyncConnection, None]:
    async with _get_conn() as conn:
        yield conn

# ── OpenAI ────────────────────────────────────────────────────────────────────

def get_openai(request: Request) -> AsyncOpenAI:
    return request.app.state.openai

# ── Repositories ──────────────────────────────────────────────────────────────

def get_memory_repo(conn = Depends(get_conn)) -> MemoryRepository:
    return MemoryRepository(conn)

def get_session_repo(conn = Depends(get_conn)) -> SessionRepository:
    return SessionRepository(conn)

def get_message_repo(conn = Depends(get_conn)) -> MessageRepository:
    return MessageRepository(conn)

def get_user_repo(conn = Depends(get_conn)) -> UserRepository:
    return UserRepository(conn)

# ── Services ──────────────────────────────────────────────────────────────────

def get_chat_service(
    memory_repo:  MemoryRepository  = Depends(get_memory_repo),
    session_repo: SessionRepository = Depends(get_session_repo),
    message_repo: MessageRepository = Depends(get_message_repo),
    openai:       AsyncOpenAI       = Depends(get_openai),
) -> ChatService:
    return ChatService(memory_repo, session_repo, message_repo, openai, settings)

def get_memory_service(
    memory_repo: MemoryRepository = Depends(get_memory_repo),
    openai:      AsyncOpenAI      = Depends(get_openai),
) -> MemoryService:
    return MemoryService(memory_repo, openai, settings)

def get_session_service(
    session_repo: SessionRepository = Depends(get_session_repo),
    message_repo: MessageRepository = Depends(get_message_repo),
    memory_repo:  MemoryRepository  = Depends(get_memory_repo),
    openai:       AsyncOpenAI       = Depends(get_openai),
) -> SessionService:
    return SessionService(session_repo, message_repo, memory_repo, openai, settings)

# ── Auth (single-user) ────────────────────────────────────────────────────────

def get_user_id() -> UUID:
    if not settings.user_id:
        raise HTTPException(503, "USER_ID not set. Run: python -m scripts.seed_user")
    return UUID(settings.user_id)
```

---

### 13.3 `api/main.py`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connection.init_pool(settings.database_url)
    app.state.openai = AsyncOpenAI(api_key=settings.openai_api_key)

    # On startup: sync any manual edits made to memory.md between sessions
    if settings.user_id:
        async with connection.get_conn() as conn:
            memory_repo = MemoryRepository(conn)
            # Load and apply any edits from memory.md (non-episodic memories only)
            await _apply_md_edits(memory_repo, settings)

    yield

    await connection.close_pool()

app = FastAPI(title="Engram", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(sessions_router, prefix="/sessions",  tags=["sessions"])
app.include_router(chat_router,     prefix="/sessions",  tags=["chat"])
app.include_router(memories_router, prefix="/memories",  tags=["memories"])

@app.get("/health", response_model=HealthResponse)
async def health(memory_repo = Depends(get_memory_repo), user_id = Depends(get_user_id)):
    counts = await memory_repo.count_active(user_id)
    return HealthResponse(status="ok", database="connected", memory_count=sum(counts.values()))
```

---

### 13.4 `api/routers/sessions.py`

```python
router = APIRouter()

@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(
    body:            CreateSessionRequest,
    session_service: SessionService = Depends(get_session_service),
    user_id:         UUID           = Depends(get_user_id),
): ...

@router.get("", response_model=list[SessionListItem])
async def list_sessions(
    session_service: SessionService = Depends(get_session_service),
    user_id:         UUID           = Depends(get_user_id),
): ...

@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id:      UUID,
    session_service: SessionService = Depends(get_session_service),
    memory_service:  MemoryService  = Depends(get_memory_service),
    user_id:         UUID           = Depends(get_user_id),
): ...

@router.get("/{session_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    session_id:  UUID,
    message_repo: MessageRepository = Depends(get_message_repo),
    user_id:      UUID              = Depends(get_user_id),
): ...

@router.post("/{session_id}/end", status_code=202)
async def end_session(
    session_id:      UUID,
    body:            EndSessionRequest,
    background_tasks: BackgroundTasks,
    session_service: SessionService = Depends(get_session_service),
    user_id:         UUID           = Depends(get_user_id),
):
    """
    Signals session end. If body.generate_episodic=True, schedules episodic
    memory generation as a background task.
    Returns 202 Accepted immediately.
    """

@router.delete("/{session_id}", status_code=204)
async def delete_session(
    session_id:      UUID,
    session_service: SessionService = Depends(get_session_service),
    user_id:         UUID           = Depends(get_user_id),
): ...
```

---

### 13.5 `api/routers/chat.py`

```python
@router.post("/{session_id}/chat")
async def chat(
    session_id:       UUID,
    body:             ChatRequest,
    background_tasks: BackgroundTasks,
    chat_service:     ChatService     = Depends(get_chat_service),
    memory_service:   MemoryService   = Depends(get_memory_service),
    session_service:  SessionService  = Depends(get_session_service),
    message_repo:     MessageRepository = Depends(get_message_repo),
    openai_client:    AsyncOpenAI     = Depends(get_openai),
    user_id:          UUID            = Depends(get_user_id),
):
    collected_tokens: list[str] = []

    async def event_generator():
        async for token in orchestrator.read_turn(
            body.message, session_id, user_id,
            chat_service, openai_client, settings,
        ):
            collected_tokens.append(token)
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    async def write_after_stream():
        assistant_message = "".join(collected_tokens)
        await orchestrator.process_turn_bg(
            body.message, assistant_message,
            session_id, user_id,
            message_repo, session_service, memory_service,
        )

    background_tasks.add_task(write_after_stream)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

---

### 13.6 `api/routers/memories.py`

```python
@router.get("", response_model=MemoryListResponse)
async def list_memories(
    memory_service: MemoryService = Depends(get_memory_service),
    user_id: UUID = Depends(get_user_id),
): ...

@router.patch("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id:      UUID,
    body:           UpdateMemoryRequest,
    memory_service: MemoryService = Depends(get_memory_service),
    user_id:        UUID          = Depends(get_user_id),
): ...

@router.delete("/{memory_id}", status_code=204)
async def forget_memory(
    memory_id:      UUID,
    memory_service: MemoryService = Depends(get_memory_service),
    user_id:        UUID          = Depends(get_user_id),
): ...
```

---

## 14. Module Specifications — UI & Scripts

### 14.1 `db/connection.py`

```python
# Pool init, pgvector registration, get_conn() context manager.
# Same as v1 spec — unchanged.
# Key: register_vector_async on every new connection via configure callback.
# Pool: min_size=2, max_size=10, dict_row factory.
```

### 14.2 `ui/components/chat.py` — SSE consumer

```python
def stream_chat(session_id: str, user_message: str) -> str:
    """
    POST /sessions/{session_id}/chat with Accept: text/event-stream.
    Iterates lines with httpx client.stream().
    Parses data: {token} lines, stops on data: [DONE].
    Updates st.empty() placeholder progressively.
    Returns full response string.
    """
```

### 14.3 `ui/components/sidebar.py`

Shows session list + New Session button. On session click: calls `GET /sessions/{id}/messages` to load history into `st.session_state.messages`. On end session: calls `POST /sessions/{id}/end` to trigger episodic generation.

### 14.4 `ui/components/memory_viewer.py`

Shows `GET /memories` grouped by type. Edit/forget via PATCH/DELETE. Calls `st.rerun()` after mutations. Episodic memories shown as read-only (no edit button).

### 14.5 `scripts/seed_user.py`

Creates user, writes `USER_ID=<uuid>` to `.env`. Unchanged from v1 spec.

### 14.6 `scripts/benchmark.py`

Seeds 1,000 random unit-vector memories via direct `memory_repo.insert()` calls (no LLM). Measures p50 first-token latency at N=30 samples, at baseline (empty store) and at 1,000 memories. Asserts delta ≤ 200ms. Unchanged methodology from v1 spec.

---

## 15. Test Specifications

### 15.1 `tests/conftest.py`

```python
@pytest_asyncio.fixture
async def conn():
    # Real psycopg3 async connection to TEST database.
    # Wraps each test in a transaction rolled back on teardown.
    # register_vector_async on the connection.

@pytest.fixture
def mock_openai():
    # AsyncMock client.
    # embeddings.create: returns deterministic unit vector seeded by hash(input).
    # chat.completions.create: returns configurable JSON string.
    # Tests set chat response via mock_openai.chat.completions.create.return_value.

@pytest_asyncio.fixture
async def user_id(conn): ...

@pytest_asyncio.fixture
async def session_id(conn, user_id): ...

@pytest_asyncio.fixture
def memory_repo(conn): return MemoryRepository(conn)

@pytest_asyncio.fixture
def session_repo(conn): return SessionRepository(conn)

@pytest_asyncio.fixture
def message_repo(conn): return MessageRepository(conn)
```

### 15.2 `tests/repositories/test_memory_repository.py`

| Test | What it verifies |
|------|-----------------|
| `test_insert_returns_memory` | Inserted memory has correct fields |
| `test_active_by_default` | Inserted memory has status=active |
| `test_search_excludes_superseded` | Superseded memory not returned by search |
| `test_search_excludes_forgotten` | Forgotten memory not returned |
| `test_search_type_filter` | search_by_vector with types=[episodic] returns only episodic |
| `test_supersession_chain` | new.supersedes_id = old.id; old.status = superseded |
| `test_bump_access_increments` | access_count 0 → 1 after bump_access |
| `test_update_confidence_clamps` | Never exceeds 1.0 or goes below 0.0 |
| `test_count_active_by_type` | Returns correct counts per type |

### 15.3 `tests/repositories/test_session_repository.py`

| Test | What it verifies |
|------|-----------------|
| `test_create_session` | Returns session with correct user_id |
| `test_update_summary` | summary, summary_turn_count, summary_updated_at all updated |
| `test_update_title_once` | Second call to update_title does NOT overwrite first |
| `test_list_by_user_ordered` | Most recently updated session comes first |
| `test_delete_cascades_messages` | Messages deleted when session deleted |

### 15.4 `tests/services/test_memory_service.py`

Mock the extractor and resolver. Verify the pipeline calls them in order and persists correctly.

| Test | Setup | Expected |
|------|-------|----------|
| `test_process_turn_noop` | extractor returns [] | No insert calls |
| `test_process_turn_new_memory` | extractor returns 1 candidate | 1 insert call |
| `test_process_turn_calls_md_sync` | Any extraction | md_sync called after persist |
| `test_forget_validates_ownership` | Wrong user_id | Raises 403 |
| `test_update_re_embeds` | update_content called | embed called, content updated |

### 15.5 `tests/services/test_session_service.py`

| Test | What it verifies |
|------|-----------------|
| `test_refresh_summary_skipped_below_k` | refresh_summary_if_needed returns early if too few new turns |
| `test_refresh_summary_calls_llm` | After K turns, LLM called, summary updated |
| `test_auto_title_only_once` | Second call does not overwrite title |
| `test_episodic_skipped_short_session` | < 4 messages → no episodic memory created |
| `test_episodic_memory_inserted` | Session with sufficient messages → episodic memory inserted with type=episodic |

### 15.6 `tests/memory/test_extractor.py`

| Test | Mock returns | Expected |
|------|-------------|----------|
| `test_extracts_preference` | Valid JSON with preference | 1 MemoryCandidate of type PREFERENCE |
| `test_returns_empty` | `[]` | Empty list, no exception |
| `test_invalid_json_returns_empty` | `"not json"` | Empty list |
| `test_ignores_episodic_type` | `[{"type":"episodic", ...}]` | Skipped → returns [] |
| `test_clamps_confidence` | confidence=99 | Clamped to 1.0 |

### 15.7 `tests/memory/test_retriever.py`

| Test | What it verifies |
|------|-----------------|
| `test_retrieve_dual_returns_two_lists` | Returns tuple of (semantic, episodic) |
| `test_single_embed_call` | embed called exactly once per retrieve_dual call |
| `test_episodic_separated` | episodic type memories only in second list |
| `test_similarity_floor` | Nothing returned when similarity < threshold |
| `test_empty_store_no_error` | Empty store returns ([], []) cleanly |

### 15.8 `tests/memory/test_resolver.py`

| Test | Mock classification | Expected |
|------|---------------------|----------|
| `test_new_below_threshold` | N/A (no LLM call) | INSERT called once |
| `test_duplicate_bumps_confidence` | DUPLICATE | No insert; update_confidence called |
| `test_update_supersedes` | UPDATE | New row inserted; old row superseded |
| `test_contradiction_keeps_higher` | CONTRADICTION | Lower-confidence row superseded |
| `test_refinement_updates_content` | REFINEMENT with merged_content | update_content called; no insert |

---

## 16. API Reference

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| GET | `/health` | Health check | — | `HealthResponse` |
| POST | `/sessions` | Create session | `CreateSessionRequest` | `SessionResponse` 201 |
| GET | `/sessions` | List sessions | — | `list[SessionListItem]` |
| GET | `/sessions/{id}` | Get session detail | — | `SessionDetailResponse` |
| GET | `/sessions/{id}/messages` | Get message history | — | `list[MessageResponse]` |
| POST | `/sessions/{id}/end` | End session (triggers episodic) | `EndSessionRequest` | 202 |
| DELETE | `/sessions/{id}` | Delete session | — | 204 |
| POST | `/sessions/{id}/chat` | **Stream chat (SSE)** | `ChatRequest` | `text/event-stream` |
| GET | `/memories` | List active memories | — | `MemoryListResponse` |
| PATCH | `/memories/{id}` | Edit memory content | `UpdateMemoryRequest` | `MemoryResponse` |
| DELETE | `/memories/{id}` | Forget a memory | — | 204 |

**SSE protocol:**
```
POST /sessions/{id}/chat
{"message": "user text"}

→ text/event-stream
data: Hello
data: ,
data:  how can I help?
data: [DONE]
```

---

## 17. Key Design Decisions & Tradeoffs

### Four-tier memory — the core architecture decision

The prompt has a hard ceiling (~1,450 tokens) because every tier is independently capped:
- Semantic top-K: caps growth as the *store* grows
- Episodic top-K: caps growth as the number of *past sessions* grows
- Rolling summary: caps growth as the *current session* runs long
- Working memory window: caps verbatim history

This simultaneously prevents prompt inflation, keeps first-token latency flat (bounded prompt → constant LLM TTFT), and protects accuracy (beyond a small K, more retrieved memories reduce precision by introducing noise).

### Service / Repository separation

Repositories are pure SQL — they can be tested against a real database with no mocking needed. Services contain business logic — they can be tested with mock repositories. The boundary is strict: if SQL appears in a service, move it to the repository. This split also means the LLM-powered memory logic (extraction, resolution, retrieval) lives in `memory/` and is independently testable with a mock OpenAI client.

### Why `episodic` is a memory type, not a separate table

Episodic memories are stored in the same `memories` table with `type='episodic'`. The HNSW index covers them automatically. The `search_by_vector` method filters by type, so semantic and episodic searches are two fast index scans over the same structure. Adding a separate table would require a second index, a second pool of connections, and a separate retrieval path — with no benefit at this scale.

### Soft-delete only — supersedes_id as the recovery mechanism

Nothing is ever hard-deleted. `supersedes_id` forms an audit chain: to recover from a wrong supersession, flip statuses and follow the chain. This directly satisfies the requirement to "recover when a memory turns out to be wrong or stale."

### Async write path — why BackgroundTasks

FastAPI's `BackgroundTasks.add_task()` runs after the response is fully sent. Extraction + conflict resolution are LLM calls. Doing them before responding would add latency proportional to extraction complexity. Running them in the background means they never contribute to first-token latency — which is the measured quantity in the benchmark.

### Summary refresh threshold

Refresh every K=10 turns (20 messages — 10 user + 10 assistant). Fewer refreshes = less LLM cost but staler context in the extractor. More refreshes = fresher context but higher cost. K=10 is the default; it's configurable via `SUMMARY_REFRESH_EVERY_K`.

### What was cut and why

- Two-hop associative retrieval: adds recall for session-linked decisions, but introduces a new unbounded growth path unless carefully capped. Deferred to post-MVP.
- Graph memory (Mem0g): relational recall across entities requires Neo4j, adds significant infrastructure cost with no measurable benefit for single-user at this scale.
- Agent-decided memory (LangMem pattern): unpredictable write frequency, adds latency when the agent reasons about whether to store. Always-on background extraction is more reliable.
- FAISS / external vector index: pgvector HNSW is sub-millisecond at this scale. Swap in FAISS if active memories exceed ~500k.

---

## 18. Setup & Run Instructions

```bash
# 1. Clone and install
git clone https://github.com/you/engram && cd engram
pip install uv && uv sync

# 2. Start Postgres
docker compose up -d && docker compose ps   # wait for "healthy"

# 3. Configure environment
cp .env.example .env
# Edit .env: add OPENAI_API_KEY

# 4. Run migrations
alembic upgrade head

# 5. Create the user
python -m scripts.seed_user
# → writes USER_ID to .env automatically

# 6. Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Start the UI (separate terminal)
streamlit run ui/app.py

# 8. Run tests
pytest tests/ -v --cov=repositories --cov=services --cov=memory --cov=agent

# 9. Run latency benchmark
python -m scripts.benchmark
# → prints p50 baseline, p50 at 1k memories, delta vs 200ms budget

# 10. Trigger episodic generation for a session (from Streamlit or curl)
curl -X POST http://localhost:8000/sessions/{session_id}/end \
  -H "Content-Type: application/json" \
  -d '{"generate_episodic": true}'
```
