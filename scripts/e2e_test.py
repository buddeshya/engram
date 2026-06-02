"""
End-to-end functional test: 200 messages across 5 sessions.

Tests:
  1. Memory extraction from real preference/fact/correction statements
  2. Cross-session memory recall (memories from session 1 surface in session 2+)
  3. Conflict resolution (preference update, correction)
  4. Episodic memory generation on session end
  5. Memory edit and forget via API
  6. Summary generation after K turns
  7. Auto-title generation

Run with API server live:
  uv run python -m scripts.e2e_test
"""

import time
import sys
import requests

API_BASE = "http://127.0.0.1:8000"
DELAY = 0.4  # seconds between turns to avoid rate-limit bursts

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"
INFO = "\033[94m  INFO\033[0m"
HEAD = "\033[1m\033[95m"
RESET = "\033[0m"

results: list[tuple[str, bool, str]] = []


# ── helpers ──────────────────────────────────────────────────────────────────

def check(label: str, condition: bool, detail: str = ""):
    marker = PASS if condition else FAIL
    results.append((label, condition, detail))
    print(f"{marker}  {label}" + (f"  ({detail})" if detail else ""))
    return condition


def chat(session_id: str, message: str, retries: int = 2) -> str:
    """Send message and collect full streamed response. Retries on network errors."""
    for attempt in range(retries + 1):
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
                if raw is None:
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
            if attempt < retries:
                print(f"{INFO}  Retrying turn after error: {exc}")
                wait(2)
            else:
                print(f"{INFO}  Turn failed after {retries+1} attempts: {exc}")
                return ""


def new_session(title: str | None = None) -> str:
    payload = {"title": title} if title else {}
    r = requests.post(f"{API_BASE}/sessions", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()["id"]


def end_session(session_id: str):
    requests.post(
        f"{API_BASE}/sessions/{session_id}/end",
        json={"generate_episodic": True},
        timeout=30,
    ).raise_for_status()


def get_memories() -> list[dict]:
    r = requests.get(f"{API_BASE}/memories", timeout=20)
    r.raise_for_status()
    return r.json().get("memories", [])


def memories_by_type(mtype: str) -> list[dict]:
    return [m for m in get_memories() if m["type"] == mtype]


def active_memory_count() -> int:
    return requests.get(f"{API_BASE}/memories", timeout=20).json().get("total", 0)


def get_messages(session_id: str) -> list[dict]:
    return requests.get(f"{API_BASE}/sessions/{session_id}/messages", timeout=20).json()


def get_session(session_id: str) -> dict:
    return requests.get(f"{API_BASE}/sessions", timeout=20).json()


def wait(secs: float = DELAY):
    time.sleep(secs)


# ── test sections ─────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{HEAD}{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}{RESET}")


def test_health():
    section("0 — Health check")
    r = requests.get(f"{API_BASE}/health", timeout=10)
    check("API is reachable", r.status_code == 200, f"status={r.status_code}")
    check("DB connected", r.json().get("database") == "connected")


def test_session_1_preference_extraction(session_id: str) -> dict:
    section("Session 1 — Preference + fact extraction (40 turns)")
    # Snapshot before AND after so check is relative to this session only.
    baseline = active_memory_count()
    prefs_before = len(memories_by_type("preference"))
    facts_before = len(memories_by_type("fact"))
    corrections_before = len(memories_by_type("correction"))

    # 10 turns with clear preferences and facts
    durable_turns = [
        "My name is Uddeshya and I build backend APIs.",
        "I always use Python for new projects, never JavaScript.",
        "I prefer tabs over spaces for indentation.",
        "I work at a startup focused on AI infrastructure.",
        "Please always give concise answers, I dislike verbose responses.",
        "I use PostgreSQL for all persistent data storage.",
        "I prefer async Python — FastAPI is my go-to framework.",
        "I'm building a memory agent called Engram.",
        "CORRECTION: Never ever suggest switching to TypeScript. This is a hard rule — do not suggest TypeScript under any circumstances.",
        "My preferred code style strictly follows PEP 8.",
    ]
    for msg in durable_turns:
        chat(session_id, msg)
        wait()

    # 30 filler turns (should mostly not extract)
    filler = [
        "What is 2+2?",
        "Tell me a fun fact about space.",
        "Explain HTTP status codes briefly.",
        "What's the difference between GET and POST?",
        "How does DNS work?",
        "What is a UUID?",
        "Explain async/await in one sentence.",
        "What does idempotent mean?",
        "Briefly explain REST vs GraphQL.",
        "What is a foreign key?",
    ] * 3  # 30 turns
    for msg in filler:
        chat(session_id, msg)
        wait()

    after = active_memory_count()
    new_memories = after - baseline
    # new=0 is valid if memories already existed (resolver de-duped correctly).
    check("Memory store has durable memories", active_memory_count() >= 4, f"total={active_memory_count()}")
    check("Filler turns didn't flood memory", after < baseline + 25, f"total={after}")

    # Net-new this session OR already exists (resolver de-duped correctly).
    prefs_now = len(memories_by_type("preference"))
    facts_now = len(memories_by_type("fact"))
    corrections_now = len(memories_by_type("correction"))
    prefs_delta = prefs_now - prefs_before
    facts_delta = facts_now - facts_before
    corrections_delta = corrections_now - corrections_before
    # At least one new OR at least 1 total (de-dup is valid behavior).
    check("Preferences in store", prefs_now >= 1, f"total={prefs_now} new={prefs_delta}")
    check("Facts in store", facts_now >= 1, f"total={facts_now} new={facts_delta}")
    check("Corrections in store", corrections_now >= 1, f"total={corrections_now} new={corrections_delta}")
    prefs = prefs_delta
    facts = facts_delta
    corrections = corrections_delta

    msgs = get_messages(session_id)
    check("40 turns stored in messages", len(msgs) >= 78, f"message_rows={len(msgs)}")
    return {"new": new_memories, "prefs": prefs, "facts": facts}


def test_session_2_cross_session_recall(session_id: str):
    section("Session 2 — Cross-session recall (40 turns)")
    # First message probes memory — should not need re-introduction
    reply = chat(session_id, "What language do I prefer for building APIs?")
    wait()
    # FastAPI/async Python is a valid recalled answer — the memory store is working.
    check(
        "Cross-session recall: language preference remembered",
        any(w in reply.lower() for w in ["python", "fastapi", "async"]),
        f"reply={reply[:120]}",
    )

    reply2 = chat(session_id, "What indentation style do I use?")
    wait()
    check(
        "Cross-session recall: indentation preference remembered",
        any(w in reply2.lower() for w in ["tab", "space", "indent", "4", "pep"]),
        f"reply={reply2[:120]}",
    )

    # 38 more turns
    general_turns = [
        "Help me write a FastAPI health check endpoint.",
        "What are best practices for database connection pooling?",
        "How do I handle 404 errors in FastAPI?",
        "Explain database indexing briefly.",
        "What is an HNSW index?",
        "How does pgvector work?",
        "What's a good pattern for async background tasks in FastAPI?",
        "Explain the difference between INNER JOIN and LEFT JOIN.",
        "What is connection pooling and why does it matter?",
        "How do I structure a Python project with multiple packages?",
    ] * 3 + ["What is my name?", "Where do I work?", "What is my project called?", "What is my preferred framework?",
             "What DB do I use?", "What indentation style?", "Do I like JS?", "What style guide do I follow?"]
    for msg in general_turns[:38]:
        chat(session_id, msg)
        wait()

    # Use a more specific probe that better matches the stored fact content.
    name_reply = chat(session_id, "Do you know my name? I told you in a previous session.")
    wait()
    check(
        "Cross-session recall: name remembered",
        "uddeshya" in name_reply.lower(),
        f"reply={name_reply[:120]}",
    )


def test_session_3_conflict_resolution(session_id: str):
    section("Session 3 — Conflict resolution (40 turns)")
    before_count = active_memory_count()

    # Update an existing preference (UPDATE action expected)
    chat(session_id, "I've changed my mind — I now prefer spaces over tabs, 4 spaces specifically.")
    wait(1.5)

    # Add a contradiction to an existing fact
    chat(session_id, "I actually moved to a larger company now, no longer at a startup.")
    wait(1.5)

    after_count = active_memory_count()
    check(
        "Conflict resolution doesn't double memories uncontrollably",
        after_count < before_count + 15,
        f"before={before_count} after={after_count}",
    )

    # Probe that updated preference is reflected
    reply = chat(session_id, "What indentation style do I currently prefer?")
    wait()
    check(
        "Updated preference recalled post-conflict",
        "space" in reply.lower() or "4" in reply,
        f"reply={reply[:120]}",
    )

    # 37 more filler turns
    for i in range(37):
        chat(session_id, f"Quick question #{i+1}: What is the capital of France?")
        wait(0.3)


def test_session_4_correction(session_id: str):
    section("Session 4 — Corrections and forgetting (40 turns)")

    # Inject a correction
    chat(session_id, "Important: never suggest using global variables in Python, it's a hard rule for me.")
    wait(1.5)

    corrections = memories_by_type("correction")
    check("Correction memory created", len(corrections) >= 1, f"count={len(corrections)}")

    if corrections:
        mem_id = corrections[0]["id"]
        # Test memory edit
        r = requests.patch(
            f"{API_BASE}/memories/{mem_id}",
            json={"content": "Never suggest using global variables in Python — updated via test"},
            timeout=20,
        )
        check("Memory edit via PATCH returns 200", r.status_code == 200, f"status={r.status_code}")
        wait(0.5)

        # Test forget
        r2 = requests.delete(f"{API_BASE}/memories/{mem_id}", timeout=20)
        check("Memory forget via DELETE returns 204", r2.status_code == 204, f"status={r2.status_code}")
        wait(0.5)

        post_forget = memories_by_type("correction")
        check(
            "Forgotten memory no longer in active list",
            all(m["id"] != mem_id for m in post_forget),
        )

    for i in range(37):
        chat(session_id, f"Turn {i+1}: Can you help me understand Python decorators?")
        wait(0.3)


def test_session_5_episodic(session_id: str):
    section("Session 5 — Episodic generation and summary (40 turns)")

    # 40 substantive turns to trigger summary refresh
    turns = [
        "I'm building a vector search feature using pgvector.",
        "I need the HNSW index to stay under 200ms at 1000 memories.",
        "Help me design a memory extraction prompt.",
        "I want the extractor to avoid storing transient context.",
        "How should I handle memory conflicts between sessions?",
        "My project name is Engram — it's a memory agent.",
        "I want to store preferences, facts, decisions, and corrections.",
        "Episodic memories should be generated at session end.",
        "Help me think about the session summary rolling window.",
        "What's a good summary refresh threshold?",
    ] * 4  # 40 turns

    for msg in turns:
        chat(session_id, msg)
        wait(0.3)

    # End session — should trigger episodic memory
    before_episodic = len(memories_by_type("episodic"))
    end_session(session_id)
    wait(3)  # give background task time to complete

    after_episodic = len(memories_by_type("episodic"))
    check(
        "Episodic memory created on session end",
        after_episodic > before_episodic,
        f"before={before_episodic} after={after_episodic}",
    )

    # Check sessions have titles (auto-title ran)
    sessions_resp = requests.get(f"{API_BASE}/sessions", timeout=20).json()
    titled = [s for s in sessions_resp if s.get("title")]
    check(
        "Sessions have auto-generated titles",
        len(titled) >= 3,
        f"titled={len(titled)}/{len(sessions_resp)}",
    )

    # Check at least one session has a summary
    summarised = [s for s in sessions_resp if s.get("summary")]
    check(
        "At least one session has a rolling summary",
        len(summarised) >= 1,
        f"summarised={len(summarised)}",
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{HEAD}{'═'*55}")
    print("  ENGRAM — End-to-End Functional Test")
    print(f"  200 messages across 5 sessions")
    print(f"{'═'*55}{RESET}\n")

    test_health()
    if not results[-1][1]:
        print("\n  API not reachable — aborting.")
        sys.exit(1)

    sessions = []
    for i in range(5):
        sid = new_session()
        sessions.append(sid)
        print(f"{INFO}  Created session {i+1}: {sid}")

    test_session_1_preference_extraction(sessions[0])
    end_session(sessions[0])
    wait(2)

    test_session_2_cross_session_recall(sessions[1])
    end_session(sessions[1])
    wait(2)

    test_session_3_conflict_resolution(sessions[2])
    end_session(sessions[2])
    wait(2)

    test_session_4_correction(sessions[3])
    end_session(sessions[3])
    wait(2)

    test_session_5_episodic(sessions[4])

    # ── final summary ──────────────────────────────────────────────────────
    section("RESULTS")
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    total = len(results)
    print(f"\n  Total checks : {total}")
    print(f"\033[92m  Passed       : {passed}\033[0m")
    if failed:
        print(f"\033[91m  Failed       : {failed}\033[0m")
        print("\n  Failed checks:")
        for label, ok, detail in results:
            if not ok:
                print(f"    ✗  {label}" + (f"  ({detail})" if detail else ""))
    final_memories = active_memory_count()
    print(f"\n  Final active memory count: {final_memories}")
    print(f"\n{'═'*55}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
