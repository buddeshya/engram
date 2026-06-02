from datetime import datetime, timezone

from memory.types import Memory, MemoryType


def format_memory_md(memories: list[Memory], updated_at: datetime | None = None) -> str:
    updated_at = updated_at or datetime.now(timezone.utc)
    grouped = {t: [] for t in MemoryType}
    for memory in memories:
        grouped[memory.type].append(memory)

    lines = [f"# Memory — last updated: {updated_at.isoformat()} UTC", ""]
    mapping = [
        (MemoryType.PREFERENCE, "Preferences"),
        (MemoryType.FACT, "Facts"),
        (MemoryType.DECISION, "Decisions"),
        (MemoryType.CORRECTION, "Corrections"),
        (MemoryType.EPISODIC, "Past sessions (episodic, read-only)"),
    ]
    for memory_type, heading in mapping:
        lines.append(f"## {heading}")
        rows = grouped[memory_type]
        if not rows:
            lines.append("- (none)")
        else:
            for row in rows:
                if memory_type is MemoryType.EPISODIC:
                    lines.append(f"- {row.content}  <!-- id:{row.id} read_only:true -->")
                else:
                    lines.append(f"- {row.content}  <!-- id:{row.id} confidence:{row.confidence:.2f} -->")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
