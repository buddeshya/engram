"""
Engram — Demo Test (50 messages, 3 sessions)
=============================================
Demonstrates the core memory system working end-to-end:

  Session 1 (20 turns) — Establish preferences, facts, corrections
  Session 2 (20 turns) — Cross-session recall without re-introduction
  Session 3 (10 turns) — Conflict resolution + episodic generation

Results are printed to stdout and saved to demo_test_results.txt.

Run:
  uv run python -m scripts.demo_test
"""

import sys
import time
import textwrap
from datetime import datetime, timezone

import requests

API_BASE = "http://127.0.0.1:8000"

# ── ANSI colours (stripped from file output) ──────────────────────────────────
GREEN   = "\033[92m"
RED     = "\033[91m"
BLUE    = "\033[94m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

OUTPUT_LINES: list[str] = []   # accumulates plain-text lines for the file


def emit(line: str = "", colour: str = ""):
    """Print coloured to terminal, plain to file buffer."""
    if line:
        print(colour + line + RESET if colour else line)
    else:
        print()
    OUTPUT_LINES.append(_strip_ansi(line))


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", text)


def section(title: str):
    emit()
    emit("─" * 58, BOLD)
    emit(f"  {title}", BOLD)
    emit("─" * 58, BOLD)


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = f"{GREEN}  PASS{RESET}" if condition else f"{RED}  FAIL{RESET}"
    suffix = f"  ({detail})" if detail else ""
    plain  = ("  PASS" if condition else "  FAIL") + f"  {label}" + suffix
    print(marker + f"  {label}" + suffix)
    OUTPUT_LINES.append(_strip_ansi(plain))
    return condition


RESULTS: list[tuple[str, bool, str]] = []


def record(label: str, condition: bool, detail: str = ""):
    RESULTS.append((label, condition, detail))
    return check(label, condition, detail)


# ── Display helpers ───────────────────────────────────────────────────────────

TYPE_ICON = {
    "correction": "🚫",
    "preference": "⚙️ ",
    "fact":       "📌",
    "decision":   "✅",
    "episodic":   "📖",
}


def show_memory_store(label: str = "Current memory store"):
    """Print all active memories grouped by type with source session."""
    data = requests.get(f"{API_BASE}/memories", timeout=15).json()
    mems = data.get("memories", [])
    total = data.get("total", 0)

    emit()
    emit(f"  ┌── {label} ({total} active memories) ──", BLUE)
    if not mems:
        emit("  │   (empty)", BLUE)
    else:
        grouped: dict[str, list[dict]] = {}
        for m in mems:
            grouped.setdefault(m["type"], []).append(m)
        for mtype in ["correction", "preference", "fact", "decision", "episodic"]:
            rows = grouped.get(mtype, [])
            if not rows:
                continue
            icon = TYPE_ICON.get(mtype, "•")
            emit(f"  │   {icon} {mtype.upper()} ({len(rows)})", BLUE)
            for m in rows:
                src = f"session={m['source_session_id'][:8]}…" if m.get("source_session_id") else "source=unknown"
                conf = f"conf={m['confidence']:.2f}"
                content = m["content"][:65] + ("…" if len(m["content"]) > 65 else "")
                emit(f"  │       [{conf}  {src}]  {content}", BLUE)
    emit("  └─────────────────────────────────────────────────────", BLUE)
    emit()


def show_retrieved_for(query: str, session_id: str):
    """
    Show what the API currently holds in memory for context —
    a client-side preview of what the agent will inject into its prompt.
    (Actual retrieval is server-side via HNSW; this shows the store state.)
    """
    mems = requests.get(f"{API_BASE}/memories", timeout=15).json().get("memories", [])
    emit(f"  ▶ Query: \"{query}\"", BOLD)
    emit(f"  Memory available to agent (from DB, injected into prompt):", BLUE)
    semantic_types = {"correction", "preference", "fact", "decision"}
    for m in mems:
        if m["type"] in semantic_types:
            icon = TYPE_ICON.get(m["type"], "•")
            content = m["content"][:70] + ("…" if len(m["content"]) > 70 else "")
            emit(f"      {icon} [{m['type']}] {content}", BLUE)
    if not any(m["type"] in semantic_types for m in mems):
        emit("      (no semantic memories in store)", BLUE)
    emit()


# ── API helpers ───────────────────────────────────────────────────────────────

def new_session(title: str | None = None) -> str:
    r = requests.post(f"{API_BASE}/sessions", json={"title": title} if title else {}, timeout=15)
    r.raise_for_status()
    return r.json()["id"]


def chat(session_id: str, message: str) -> str:
    """Stream SSE response and collect full text."""
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{API_BASE}/sessions/{session_id}/chat",
                json={"message": message},
                stream=True,
                timeout=90,
            )
            resp.raise_for_status()
            chunks = []
            for raw in resp.iter_lines(chunk_size=None, decode_unicode=True):
                if not raw:
                    continue
                raw = raw.strip()
                if not raw.startswith("data:"):
                    continue
                chunk = raw[len("data:"):]
                if chunk.startswith(" "):
                    chunk = chunk[1:]
                if chunk == "[DONE]":
                    resp.close()
                    break
                chunks.append(chunk)
            return "".join(chunks).strip()
        except Exception as exc:
            if attempt < 2:
                time.sleep(2)
            else:
                return f"[error: {exc}]"
    return ""


def end_session(session_id: str):
    requests.post(
        f"{API_BASE}/sessions/{session_id}/end",
        json={"generate_episodic": True},
        timeout=30,
    ).raise_for_status()


def memories() -> list[dict]:
    return requests.get(f"{API_BASE}/memories", timeout=15).json().get("memories", [])


def memories_of(mtype: str) -> list[dict]:
    return [m for m in memories() if m["type"] == mtype]


def total_memories() -> int:
    return requests.get(f"{API_BASE}/memories", timeout=15).json().get("total", 0)


def wait(s: float = 0.5):
    time.sleep(s)


def show_turn(idx: int, user: str, reply: str):
    emit(f"\n  Turn {idx:02d}")
    emit(f"  User   : {user}")
    wrapped = textwrap.fill(reply, width=72, initial_indent="  Agent  : ",
                            subsequent_indent="           ")
    emit(wrapped)


# ── Sessions ──────────────────────────────────────────────────────────────────

def session_1_establish(sid: str):
    section("Session 1 — Establishing identity, preferences & corrections (20 turns)")
    emit(f"  Session ID: {sid}", BLUE)
    emit()

    turns = [
        # Durable — should be extracted
        ("My name is Uddeshya. I build backend systems professionally.", True),
        ("I always use Python for new projects. I never use JavaScript.", True),
        ("CORRECTION: Never suggest JavaScript or TypeScript to me under any circumstances.", True),
        ("I work at a startup focused on AI infrastructure.", True),
        ("I prefer concise answers. Don't pad responses.", True),
        ("I use PostgreSQL for all persistent storage.", True),
        ("My preferred framework is FastAPI.", True),
        ("I always use tabs for indentation, not spaces.", True),
        ("I'm building a project called Engram — a memory agent.", True),
        ("I follow PEP 8 strictly.", True),
        # Filler — should mostly not be extracted
        ("What is the HTTP status code for 'Not Found'?", False),
        ("Explain what a foreign key constraint does.", False),
        ("What's the difference between PUT and PATCH?", False),
        ("How does an HNSW index work?", False),
        ("What is connection pooling?", False),
        ("How does async/await work in Python?", False),
        ("Briefly explain REST vs GraphQL.", False),
        ("What is a UUID?", False),
        ("What is idempotency?", False),
        ("How does DNS resolution work?", False),
    ]

    for i, (msg, _durable) in enumerate(turns, 1):
        reply = chat(sid, msg)
        show_turn(i, msg, reply)
        wait(0.4)

    wait(2)  # allow background extraction to complete
    mem_count = total_memories()
    prefs = len(memories_of("preference"))
    facts = len(memories_of("fact"))
    corrections = len(memories_of("correction"))

    record("Preferences stored",  prefs >= 1,       f"count={prefs}")
    record("Facts stored",        facts >= 1,       f"count={facts}")
    record("Corrections stored",  corrections >= 1, f"count={corrections}")
    record("Filler didn't flood memory store", mem_count < 30, f"total={mem_count}")

    show_memory_store("Memory extracted from Session 1")


def session_2_recall(sid: str):
    section("Session 2 — Cross-session recall without re-introduction (20 turns)")
    emit(f"  Session ID: {sid}", BLUE)
    emit("  (No preferences stated — agent must recall from memory)")
    emit()

    # Show what the agent has access to before any probe
    show_memory_store("Memory available at start of Session 2 (carried over from Session 1)")

    probe_turns = [
        ("What programming language do I prefer?",
         lambda r: any(w in r.lower() for w in ["python", "fastapi"]),
         "language recalled"),
        ("Do you know my name?",
         lambda r: "uddeshya" in r.lower(),
         "name recalled"),
        ("What database do I use?",
         lambda r: "postgresql" in r.lower() or "postgres" in r.lower(),
         "db recalled"),
        ("What is my preferred web framework?",
         lambda r: "fastapi" in r.lower(),
         "framework recalled"),
        ("What project am I building?",
         lambda r: "engram" in r.lower(),
         "project recalled"),
    ]

    for i, (msg, validator, label) in enumerate(probe_turns, 1):
        show_retrieved_for(msg, sid)
        reply = chat(sid, msg)
        show_turn(i, msg, reply)
        record(f"Cross-session recall: {label}", validator(reply), f"reply={reply[:90]}")
        wait(0.5)

    # 15 general turns
    general = [
        "Help me write a FastAPI endpoint that returns a list of users.",
        "What are best practices for database migrations?",
        "How should I structure a Python monorepo?",
        "What's a good pattern for handling background tasks in FastAPI?",
        "Explain the difference between INNER JOIN and LEFT JOIN.",
        "What is a vector embedding?",
        "How does cosine similarity work?",
        "What is pgvector?",
        "How do I handle 429 rate limit errors from OpenAI?",
        "What is an async generator in Python?",
        "Explain dependency injection.",
        "What is the repository pattern?",
        "How does connection pooling work in psycopg3?",
        "What is SSE (Server-Sent Events)?",
        "Explain soft deletes in databases.",
    ]
    for i, msg in enumerate(general, 6):
        reply = chat(sid, msg)
        show_turn(i, msg, reply)
        wait(0.4)


def session_3_conflict_episodic(sid: str):
    section("Session 3 — Conflict resolution + episodic memory (10 turns)")
    emit(f"  Session ID: {sid}", BLUE)
    emit()

    show_memory_store("Memory store entering Session 3")
    before = total_memories()
    before_episodic = len(memories_of("episodic"))

    conflict_turns = [
        "I've changed my mind — I now prefer spaces over tabs. 4 spaces specifically.",
        "I moved to a larger company recently, no longer at a startup.",
        "Help me design an API endpoint for semantic search.",
        "What are the tradeoffs of storing embeddings in Postgres vs a dedicated vector DB?",
        "How do I structure a prompt for memory extraction?",
        "What is the best way to handle concurrent writes to a shared resource?",
        "Explain optimistic vs pessimistic locking.",
        "How does pgvector's HNSW index compare to IVFFlat?",
        "What is the difference between episodic and semantic memory?",
        "How would you design a memory system that gets smarter over time?",
    ]

    for i, msg in enumerate(conflict_turns, 1):
        reply = chat(sid, msg)
        show_turn(i, msg, reply)
        wait(0.4)

    wait(2)
    after = total_memories()
    show_memory_store("Memory store after conflict resolution turns")
    record("Conflict resolution: memory store stable", after <= before + 5,
           f"before={before} after={after}")

    # Probe that updated preference is reflected
    reply = chat(sid, "What indentation style do I currently use?")
    show_turn(11, "What indentation style do I currently use?", reply)
    record("Conflict resolution: updated preference recalled",
           any(w in reply.lower() for w in ["space", "4", "four"]),
           f"reply={reply[:90]}")

    # End session → trigger episodic memory generation
    emit("\n  [Ending session — generating episodic memory…]", BLUE)
    end_session(sid)
    wait(3)

    after_episodic = len(memories_of("episodic"))
    show_memory_store("Memory store after session end (includes new episodic)")
    record("Episodic memory created on session end",
           after_episodic > before_episodic,
           f"before={before_episodic} after={after_episodic}")

    sessions_resp = requests.get(f"{API_BASE}/sessions", timeout=15).json()
    titled   = [s for s in sessions_resp if s.get("title")]
    summarised = [s for s in sessions_resp if s.get("summary")]
    record("Sessions have auto-generated titles", len(titled) >= 2,
           f"titled={len(titled)}/{len(sessions_resp)}")
    record("Sessions have rolling summaries", len(summarised) >= 1,
           f"summarised={len(summarised)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    emit("=" * 58, BOLD)
    emit("  ENGRAM — Demo Test  (50 messages, 3 sessions)", BOLD)
    emit(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", BOLD)
    emit("=" * 58, BOLD)

    # Health check
    section("Health check")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=10)
        record("API reachable", r.status_code == 200, f"status={r.status_code}")
        record("DB connected", r.json().get("database") == "connected")
    except Exception as exc:
        emit(f"  ERROR: {exc}", RED)
        emit("  Ensure API is running: uv run uvicorn api.main:app --reload")
        sys.exit(1)

    # Create 3 sessions
    s1 = new_session()
    s2 = new_session()
    s3 = new_session()
    emit(f"\n  Session 1: {s1}", BLUE)
    emit(f"  Session 2: {s2}", BLUE)
    emit(f"  Session 3: {s3}", BLUE)

    # Run test sections
    session_1_establish(s1)
    end_session(s1)
    wait(2)

    session_2_recall(s2)
    end_session(s2)
    wait(2)

    session_3_conflict_episodic(s3)

    # Final summary
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = sum(1 for _, ok, _ in RESULTS if not ok)
    total  = len(RESULTS)

    section("FINAL RESULTS")
    emit(f"  Total checks : {total}")
    emit(f"  Passed       : {passed}", GREEN)
    if failed:
        emit(f"  Failed       : {failed}", RED)
        emit()
        emit("  Failed checks:", RED)
        for label, ok, detail in RESULTS:
            if not ok:
                d = f"  ({detail})" if detail else ""
                emit(f"    ✗  {label}{d}", RED)

    emit()
    emit(f"  Final active memory count: {total_memories()}", BLUE)
    emit()
    emit("=" * 58, BOLD)

    # Write plain-text file
    out_path = "demo_test_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(OUTPUT_LINES) + "\n")
    emit(f"\n  Results saved to: {out_path}", BLUE)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
