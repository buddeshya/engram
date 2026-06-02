import json
import logging

from openai import AsyncOpenAI

from memory.types import MemoryCandidate
from repositories.memory_repository import MemoryRepository

log = logging.getLogger("engram.resolver")

CLASSIFY_PROMPT = """Classify the relationship between two memory statements about the same person.
Return ONLY valid JSON:
{"classification":"DUPLICATE|UPDATE|CONTRADICTION|REFINEMENT|NOOP","merged_content":null}
"""


def _norm(text: str) -> str:
    """Normalise text for exact-match comparison (lowercase, collapsed whitespace)."""
    return " ".join(text.lower().strip().split())


async def _classify(openai_client: AsyncOpenAI, model: str, existing: str, new: str) -> dict:
    """
    Ask the LLM to classify the relationship between two memory statements.
    Falls back to NOOP on any parse failure — we'd rather do nothing than
    corrupt the memory store with a bad classification.
    """
    response = await openai_client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": CLASSIFY_PROMPT},
            {"role": "user", "content": f'EXISTING: "{existing}"\nNEW: "{new}"'},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"classification": "NOOP", "merged_content": None}


async def resolve(
    memory_repo: MemoryRepository,
    openai_client: AsyncOpenAI,
    candidate: MemoryCandidate,
    candidate_embedding: list[float],
    user_id,
    session_id,
    model: str,
) -> None:
    """
    Decide what to do with a newly extracted memory candidate.

    Resolution order (two deterministic guardrails first, then LLM):
      1. Exact text match → DUPLICATE (no LLM call, cheapest path)
      2. Vector similarity ≥ 0.94 → DUPLICATE (near-identical phrasing)
      3. No similar match found → INSERT as NEW
      4. LLM classification for ambiguous cases:
           DUPLICATE   → bump confidence +0.05
           UPDATE      → insert new, supersede old (soft-delete via status)
           CONTRADICTION → keep higher-confidence version, mark both with metadata
           REFINEMENT  → merge content into existing row (no new row)
           NOOP        → discard candidate silently

    Superseded rows are never hard-deleted — they form an audit chain
    recoverable via the supersedes_id foreign key.
    """
    log.info("Resolving [%s] \"%s\"", candidate.type.value, candidate.content[:80])

    # Guardrail 1: exact normalized text → instant duplicate, no LLM call.
    active_same_type = await memory_repo.get_active_by_type(user_id=user_id, memory_type=candidate.type)
    for existing in active_same_type:
        if _norm(existing.content) == _norm(candidate.content):
            log.info("  → DUPLICATE (exact match) — confidence bumped")
            await memory_repo.update_confidence(existing.id, +0.05)
            return

    matches = await memory_repo.search_by_vector(
        user_id=user_id,
        embedding=candidate_embedding,
        types=[candidate.type],
        top_k=3,
        threshold=0.60,
    )
    if not matches:
        log.info("  → NEW (no similar memories found) — inserting")
        await memory_repo.insert(candidate, candidate_embedding, user_id, session_id)
        return

    existing, similarity = matches[0]
    log.info("  similar match found (sim=%.3f): \"%s\"", similarity, existing.content[:60])

    # Guardrail 2: very high similarity → treat as duplicate without LLM.
    if similarity >= 0.94:
        log.info("  → DUPLICATE (sim≥0.94) — confidence bumped")
        await memory_repo.update_confidence(existing.id, +0.05)
        return

    # LLM classification for the ambiguous middle range (0.60–0.94).
    label = await _classify(openai_client, model, existing.content, candidate.content)
    action = (label.get("classification") or "NOOP").upper()
    log.info("  → LLM classified as: %s", action)

    if action == "DUPLICATE":
        await memory_repo.update_confidence(existing.id, +0.05)

    elif action == "UPDATE":
        # New fact supersedes old — insert new row linked to old via supersedes_id.
        new_row = await memory_repo.insert(candidate, candidate_embedding, user_id, session_id, supersedes_id=existing.id)
        await memory_repo.supersede(existing.id)
        await memory_repo.set_conflict(existing.id, new_row.id, "update")

    elif action == "CONTRADICTION":
        # Logically incompatible — keep whichever has higher confidence.
        if candidate.confidence >= existing.confidence:
            log.info("  candidate wins contradiction — superseding old")
            new_row = await memory_repo.insert(candidate, candidate_embedding, user_id, session_id, supersedes_id=existing.id)
            await memory_repo.supersede(existing.id)
            await memory_repo.set_conflict(existing.id, new_row.id, "contradiction")
        else:
            log.info("  existing wins contradiction — confidence bumped")
            await memory_repo.update_confidence(existing.id, +0.02)

    elif action == "REFINEMENT":
        # New content adds detail to old — merge and update in-place (no new row).
        merged = label.get("merged_content") or candidate.content
        log.info("  merging into: \"%s\"", merged[:80])
        await memory_repo.update_content(existing.id, merged, candidate_embedding)

    else:
        log.info("  → NOOP — nothing stored")
