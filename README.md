# Engram

> A conversational AI assistant with persistent, selective memory across sessions.

Engram remembers facts, preferences, decisions, and corrections across chat sessions using a four-tier memory architecture backed by PostgreSQL + pgvector. It extracts only durable information from each turn — most turns produce no stored memory — and resolves conflicts when information changes.

---

## Architecture Overview

```
User message
     │
     ▼
┌─────────────────────────────────────────┐
│            Read Path (sync)             │
│  Embed query → vector search → build   │
│  bounded prompt → stream LLM response  │
└─────────────────────────────────────────┘
     │
     ▼ (after response)
┌─────────────────────────────────────────┐
│           Write Path (async)            │
│  Extract candidates → resolve against  │
│  existing → insert/update/supersede    │
│  → sync memory.md                      │
└─────────────────────────────────────────┘
```

**Four-tier bounded prompt** (total ≤ ~1450 tokens):
| Tier | Source | Cap |
|---|---|---|
| Semantic memories | pgvector HNSW scan | top-5 |
| Episodic memories | pgvector HNSW scan | top-3 |
| Session summary | Rolling LLM summary | ≤ 200 words |
| Working memory | Last N message pairs | configurable window |

---

## Requirements

- Python **3.11+**
- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- [uv](https://github.com/astral-sh/uv) — Python package manager
- OpenAI API key (`gpt-4o-mini` + `text-embedding-3-small`)

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/engram.git
cd engram
```

### 2. Install uv (if not already installed)

```bash
pip install uv
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://agent:agent@localhost:5433/memory_agent
```

> The default DB port is `5433` to avoid conflicts with any local Postgres.
> Adjust `DATABASE_URL` and `docker-compose.yml` if you change it.

### 5. Start the database

```bash
docker compose up -d
```

Wait for healthy status:

```bash
docker compose ps   # STATUS should show "(healthy)"
```

### 6. Run database migrations

```bash
uv run python -m scripts.init_db
```

This runs `alembic upgrade head` to create all tables, enums, and indexes (including the HNSW vector index).

### 7. Seed single user (Uddeshya)

```bash
uv run python -m scripts.seed_user
```

This creates the user row and writes `USER_ID=<uuid>` to your `.env` automatically.

### 8. Start the API server

```bash
uv run uvicorn api.main:app --reload
```

API is live at: [http://127.0.0.1:8000](http://127.0.0.1:8000)
Swagger docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 9. Start the UI (new terminal)

```bash
uv run streamlit run ui/app.py
```

UI is live at: [http://localhost:8501](http://localhost:8501)

---

## Project Structure

```
engram/
├── api/                    # FastAPI app — routes, schemas, dependencies
│   ├── main.py             # App startup, lifespan, CORS
│   ├── dependencies.py     # Dependency injection (repos, services, user auth)
│   ├── schemas.py          # Pydantic request/response models
│   └── routers/
│       ├── sessions.py     # /sessions — create, list, messages, end
│       ├── chat.py         # /sessions/{id}/chat — SSE streaming
│       └── memories.py     # /memories — list, edit, forget, sync-md
├── agent/
│   ├── orchestrator.py     # Read-path coordinator + async write-path tasks
│   └── prompt.py           # Pure prompt formatting (no I/O)
├── memory/
│   ├── types.py            # Dataclasses: Memory, MemoryCandidate, enums
│   ├── retriever.py        # Single-embed dual retrieval (semantic + episodic)
│   ├── extractor.py        # LLM-powered extraction from conversation turns
│   ├── resolver.py         # Conflict resolution: duplicate/update/refinement
│   └── md_sync.py          # Format active memories to memory.md
├── services/
│   ├── chat_service.py     # Read path: retrieve context + build prompt
│   ├── memory_service.py   # Write path: extract → resolve → sync
│   └── session_service.py  # Auto-title, rolling summary, episodic generation
├── repositories/           # Pure SQL — no business logic
│   ├── memory_repository.py
│   ├── session_repository.py
│   ├── message_repository.py
│   └── user_repository.py
├── db/
│   └── connection.py       # Async psycopg pool management
├── ui/
│   ├── app.py              # Streamlit app entry point + styles
│   ├── api_client.py       # HTTP client for all API calls
│   └── components/
│       ├── chat.py         # Chat message view + SSE streaming
│       ├── sidebar.py      # Sessions list + inline memory viewer
│       └── memory_viewer.py# Full memory viewer with edit/forget
├── scripts/
│   ├── init_db.py          # Run Alembic migrations
│   ├── seed_user.py        # Create user, write USER_ID to .env
│   ├── benchmark.py        # Latency benchmark: baseline vs 1k memories
│   └── e2e_test.py         # 200-message 5-session functional test
├── tests/
│   ├── conftest.py         # Fixtures: mock_openai, settings_stub, db_conn
│   ├── api/                # Route-level tests with dependency overrides
│   ├── memory/             # Unit tests: extractor, retriever, resolver
│   ├── repositories/       # DB-backed repository tests
│   └── services/           # Service-layer unit tests
├── alembic/                # Alembic migration environment
│   └── versions/
│       └── 001_initial_schema.py
├── config.py               # Pydantic Settings (reads from .env)
├── pyproject.toml
├── docker-compose.yml
├── alembic.ini
└── memory.md               # Human-readable mirror of stored memories
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + memory count |
| POST | `/sessions` | Create a new session |
| GET | `/sessions` | List all sessions |
| GET | `/sessions/{id}/messages` | Get message history |
| POST | `/sessions/{id}/chat` | **Stream chat (SSE)** |
| POST | `/sessions/{id}/end` | End session + generate episodic memory |
| GET | `/memories` | List all active memories |
| PATCH | `/memories/{id}` | Edit memory content |
| DELETE | `/memories/{id}` | Forget a memory (soft-delete) |
| POST | `/memories/sync-md` | Apply manual `memory.md` edits to DB |

---

## Running Tests

### Unit + API tests (no DB required)

```bash
uv run --with pytest --with pytest-asyncio python -m pytest tests -q
```

### Repository tests (requires running DB)

```bash
uv run --with pytest --with pytest-asyncio python -m pytest tests/repositories -q
```

### Full end-to-end test (requires API + DB running)

Runs 200 messages across 5 sessions testing extraction, cross-session recall,
conflict resolution, corrections, episodic generation, and latency.

```bash
uv run python -m scripts.e2e_test
```

### Latency benchmark

Seeds 1,000 memories directly, measures p50 `retrieve_context` latency at
baseline vs 1k memories. Asserts delta ≤ 200ms.

```bash
uv run python -m scripts.benchmark
```

---

## Configuration Reference

All values live in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI key |
| `DATABASE_URL` | `postgresql://agent:agent@localhost:5433/memory_agent` | Postgres connection string |
| `USER_ID` | — | Written by `seed_user.py` |
| `USER_NAME` | `Uddeshya` | Display name |
| `CHAT_MODEL` | `gpt-4o-mini` | LLM used for chat + extraction |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MEMORY_MD_PATH` | `./memory.md` | Path to human-readable memory file |
| `WORKING_MEMORY_WINDOW` | `10` | Number of recent turn pairs to include |
| `SEMANTIC_TOP_K` | `5` | Max semantic memories per turn |
| `EPISODIC_TOP_K` | `3` | Max episodic memories per turn |
| `SUMMARY_REFRESH_EVERY_K` | `10` | Refresh rolling summary every K turns |

---

## Memory File (`memory.md`)

After each turn, Engram rewrites `memory.md` as a human-readable mirror of the DB:

```markdown
# Memory — last updated: 2026-06-02T09:45:12 UTC

## Preferences
- Prefers tabs over spaces  <!-- id:abc... confidence:0.90 -->

## Facts
- Works at a startup on AI infrastructure  <!-- id:def... confidence:0.85 -->

## Corrections
- Never suggest TypeScript  <!-- id:ghi... confidence:0.95 -->

## Past sessions (episodic, read-only)
- Uddeshya discussed building a memory pipeline ...  <!-- id:jkl... read_only:true -->
```

You can **edit this file by hand** and then sync changes back to the DB:

```bash
curl -X POST http://127.0.0.1:8000/memories/sync-md
```

Or use the **Sync memory.md** button in the Memories view.
