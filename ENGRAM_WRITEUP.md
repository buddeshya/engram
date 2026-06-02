# Engram
### A conversational AI agent with four-tier persistent memory across sessions.

---

## The Problem

Every time you start a new conversation with an AI assistant, you start from zero.

You re-explain your stack. You re-state your preferences. You remind it what project you're working on, what decisions you've already made, what you tried last time. The assistant is capable — but it has no memory of you.

Engram solves this — not by storing everything, but by remembering the right things, in the right structure, with guarantees on both speed and accuracy.

---

## What I'm Building

Engram is a conversational AI agent with **persistent, selective memory across sessions**.

A user can have multiple conversation sessions. Between sessions — and within long ones — the agent remembers facts, preferences, decisions, and the history of past conversations. Start a new session tomorrow and the agent already knows your preferences, your tech stack, and the decisions made last week. It also knows what you were working on last session, with enough specificity to pick up where you left off.

The key word is *selective*. The hard part of this problem isn't remembering — it's knowing what *not* to remember, when to update, and when to forget. An agent that stores everything becomes noisy and confused. Engram is designed to extract only durable, high-value facts, reconcile them against what it already knows, and stay fast as memory grows.

---

## The Memory Architecture

Engram uses **four independent memory tiers**. Each tier answers a different question. Each tier is independently bounded so the prompt never grows with conversation length, store size, or number of past sessions.

```
┌─────────────────────────────────────────────────────┐
│               BOUNDED PROMPT  (~1,450 tokens max)    │
│                                                      │
│  Working memory        last 10 turns · verbatim      │
│  Session summary       rolling · ~200 tok cap        │
│  Semantic memory       top-5 by similarity           │
│  Episodic memory       top-3 past sessions           │
└─────────────────────────────────────────────────────┘
```

### Tier 1 — Working memory (in-session)
The last 10 turns of the current conversation, injected verbatim. Keeps immediate back-and-forth coherent. Resets when a new session starts.

### Tier 2 — Session summary (in-session)
A rolling compressed summary of the current conversation. As the conversation grows beyond 10 turns, older turns don't vanish — they're compressed into a ~200-token summary that's refreshed every 10 turns. This means no matter how long a conversation runs, the context stays bounded.

### Tier 3 — Semantic memory (cross-session)
The durable user model: preferences, facts, decisions, and corrections extracted from all past conversations. Stored as vector embeddings in PostgreSQL with a pgvector HNSW index. Retrieved by similarity — top-5 per turn, with a similarity floor that prevents irrelevant facts from being injected.

Five memory types with different rules:
- `preference` — sticky, latest-wins on conflict
- `fact` — updatable, temporal supersession
- `decision` — project-scoped, sticky while active
- `correction` — highest injection priority, never decays
- `episodic` — past session summaries (read-only in retrieval)

### Tier 4 — Episodic memory (cross-session)
One-paragraph summaries of past sessions, generated when a session ends. Unlike semantic memory (which captures what is currently true about the user), episodic memory captures *what happened* in past sessions. Retrieved by similarity to the current query — so a question about auth surfaces "Session 12: implemented auth module, JWT 30min, Redis refresh tokens" even without explicit keywords.

---

## How the Write Path Works

After each turn streams to the user, a background pipeline runs without affecting response latency:

1. Save messages and update session metadata
2. Auto-generate a session title from the first message (one-time)
3. Refresh the rolling session summary (every 10 turns)
4. Run the extractor: most turns produce nothing; occasional turns produce 1-3 candidates
5. For each candidate: resolve against existing memories (duplicate / update / supersede / merge)
6. Persist and sync memory.md (human-readable mirror)
7. On session end: generate and store the episodic memory for this session

The key: none of this touches first-token latency. The user already has their answer before any of this runs.

---

## How the Read Path Works

On each turn, the read path has a hard latency ceiling:

1. Embed the user's message (one network call — constant cost)
2. pgvector HNSW scan → semantic top-5 (~sub-millisecond)
3. pgvector HNSW scan → episodic top-3 (~sub-millisecond, reuses same embedding)
4. Fetch session summary (single indexed row)
5. Fetch last 10 turns (working memory)
6. Assemble bounded prompt and stream

At turn 1,000, the only thing that grew compared to turn 1 is the vector scan — by under a millisecond. The embedding call is constant. The prompt is bounded, so the LLM's own time-to-first-token is constant. The latency budget holds.

---

## A Concrete Example

**Session 1, turn 1:**
> Maya: "I'm building a REST API in Python. I always use tabs, and please never suggest switching to JavaScript."
>
> *Stored: `preference: prefers tabs`, `fact: building REST API in Python`, `correction: no JavaScript`*

**Session 1, turn 25:**
> Maya: "How should I structure the error handling?"
>
> *Retrieved: the Python fact (relevant). Tabs preference is below the similarity floor for this query — correctly not injected. Session summary covers turns 1-14. Working memory has turns 15-24.*

**Session 40, turn 1 (months later):**
> Maya: "Can you review this endpoint for auth issues?"
>
> *Retrieved semantics: no-JS correction, tabs preference, PostgreSQL fact, JWT decision.*
> *Retrieved episodic: "Session 12: implemented auth module, JWT 30min expiry, Redis refresh tokens, resolved logout bug." — a past session directly relevant to "auth issues."*
> *The agent walks in already knowing the history of her auth work. Prompt: ~1,450 tokens. First-token latency: same as turn 1.*

---

## What Success Looks Like

- A new session picks up context from previous sessions without re-prompting.
- The rolling summary keeps long in-session conversations coherent without growing the prompt.
- Episodic memory surfaces relevant past-session work — not just facts about the user.
- The agent correctly handles a changed preference (update, not contradiction).
- The memory.md file accurately reflects all stored memories and can be edited by hand.
- The latency benchmark proves p50 first-token latency at 1,000 memories is within 200ms of turn 1.
- Tests cover the full lifecycle: extraction, storage, retrieval, conflict resolution, forgetting, and session summary.

---

## Technical Snapshot

| Layer | Choice | Why |
|-------|--------|-----|
| Backend | FastAPI + Python 3.11 | Async-native, clean SSE streaming |
| Frontend | Streamlit | Focuses the demo on the memory system |
| Database | PostgreSQL + pgvector | HNSW index for sub-ms vector search |
| LLM | OpenAI (gpt-4o-mini) | Direct API, no frameworks |
| Architecture | Repository + Service layers | Clean separation of SQL from business logic |
| Memory format | Postgres rows + memory.md | Machine-fast reads; human-readable and editable |

Built without agent frameworks (no LangChain, LlamaIndex, etc.) and without off-the-shelf memory libraries. The memory layer — extraction, conflict resolution, rolling summary, episodic generation — is designed and implemented from scratch.

---

*Engram — from neuroscience: the physical trace a memory leaves in the brain.*
