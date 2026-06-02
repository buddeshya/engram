from memory.types import Memory

SYSTEM_PROMPT = (
    "You are Engram, a helpful and concise assistant for Uddeshya. "
    "Use remembered context naturally without mentioning memory internals."
)


def _semantic_line(memories: list[Memory], memory_type: str) -> str:
    rows = [m.content for m in memories if m.type.value == memory_type]
    return "; ".join(rows)


def format_semantic_context(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = ["<semantic_context>"]
    for label, key in [
        ("Corrections", "correction"),
        ("Preferences", "preference"),
        ("Facts", "fact"),
        ("Decisions", "decision"),
    ]:
        content = _semantic_line(memories, key)
        if content:
            lines.append(f"{label}: {content}")
    lines.append("</semantic_context>")
    return "\n".join(lines)


def format_episodic_context(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = ["<past_sessions>"]
    for m in memories:
        lines.append(f"- {m.content}")
    lines.append("</past_sessions>")
    return "\n".join(lines)


def build_messages(
    query: str,
    semantic_memories: list[Memory],
    episodic_memories: list[Memory],
    session_summary: str | None,
    recent_messages: list[dict],
) -> list[dict]:
    context_parts = []
    sem = format_semantic_context(semantic_memories)
    epi = format_episodic_context(episodic_memories)
    if sem:
        context_parts.append(sem)
    if epi:
        context_parts.append(epi)
    if session_summary:
        context_parts.append(f"<session_summary>\n{session_summary}\n</session_summary>")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_parts:
        messages.append({"role": "system", "content": "\n\n".join(context_parts)})
    messages.extend(recent_messages)
    messages.append({"role": "user", "content": query})
    return messages
