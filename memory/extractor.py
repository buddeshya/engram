import json
import logging

from openai import AsyncOpenAI

from memory.types import MemoryCandidate, MemoryType

log = logging.getLogger("engram.extractor")

# The extraction prompt is strict by design: most turns should return [].
# Only facts that would still be true and useful months from now are stored.
EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction system for a conversational AI assistant.

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

NEVER extract episodic memories.
Return ONLY valid JSON array:
[{"type":"preference|fact|decision|correction","content":"...","confidence":0.5}]
Return [] if nothing is worth storing.
"""


def _to_candidate(raw: dict) -> MemoryCandidate | None:
    """
    Validate and normalise a single raw extraction dict into a MemoryCandidate.
    Returns None if the type is unknown or content is empty, so callers can
    filter silently rather than raising exceptions.
    """
    kind = raw.get("type")
    content = (raw.get("content") or "").strip()
    confidence = raw.get("confidence", 0.7)

    # Reject episodic — those are generated separately by SessionService.
    if kind not in {t.value for t in MemoryType if t is not MemoryType.EPISODIC}:
        return None
    if not content:
        return None
    try:
        # Clamp confidence to [0.5, 1.0] — below 0.5 is too uncertain to store.
        conf = max(0.5, min(1.0, float(confidence)))
    except (ValueError, TypeError):
        conf = 0.7
    return MemoryCandidate(type=MemoryType(kind), content=content, confidence=conf)


async def extract(
    user_message: str,
    assistant_message: str,
    openai_client: AsyncOpenAI,
    model: str,
) -> list[MemoryCandidate]:
    """
    Run one LLM call to extract zero or more durable memory candidates
    from a single conversation turn.

    Returns [] on any error — extraction failures must never break chat.
    The extractor is intentionally conservative: most turns return [].
    """
    log.info("Extracting memories from turn…")
    try:
        response = await openai_client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"CONVERSATION TURN:\nUser: {user_message}\n"
                        f"Assistant: {assistant_message}\n\nExtract durable memories. Return []."
                    ),
                },
            ],
        )
        raw_text = response.choices[0].message.content or "[]"
        parsed = json.loads(raw_text)
        if not isinstance(parsed, list):
            log.warning("Extractor returned non-list, skipping")
            return []
        out = []
        for item in parsed:
            if isinstance(item, dict):
                c = _to_candidate(item)
                if c is not None:
                    out.append(c)
        if out:
            for c in out:
                log.info("  + extracted [%s] %s (conf=%.2f)", c.type.value, c.content[:80], c.confidence)
        else:
            log.info("  → no durable memories found in this turn")
        return out
    except Exception as exc:
        log.error("Extractor error: %s", exc)
        return []
