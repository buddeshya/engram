import re
from datetime import datetime, timezone

from memory.extractor import extract
from memory.md_sync import format_memory_md
from memory.resolver import resolve
from memory.retriever import embed
from memory.types import MemoryType


class MemoryService:
    """
    Owns the full write-path memory pipeline and user-facing memory mutations.

    Write path (called from background task after each chat turn):
      extract → for each candidate: embed → resolve → sync_memory_md

    User-facing mutations (called directly from API routes):
      forget → marks a memory row as 'forgotten' (soft-delete)
      update_content → re-embeds new content and updates the row in-place
      sync_from_md → reads memory.md, applies edits back to DB
    """

    def __init__(self, memory_repo, openai_client, settings):
        self.memory_repo = memory_repo
        self.openai = openai_client
        self.settings = settings

    async def process_turn(self, user_message: str, assistant_message: str, session_id, user_id) -> None:
        """
        Full extraction + resolution pipeline for one conversation turn.
        Most turns return [] from the extractor and exit early with no DB writes.
        sync_memory_md is always called at the end to keep the file in sync.
        """
        candidates = await extract(
            user_message=user_message,
            assistant_message=assistant_message,
            openai_client=self.openai,
            model=self.settings.chat_model,
        )
        for candidate in candidates:
            # Each candidate is embedded individually because contents differ.
            vector = await embed(candidate.content, self.openai, self.settings.embedding_model)
            await resolve(
                memory_repo=self.memory_repo,
                openai_client=self.openai,
                candidate=candidate,
                candidate_embedding=vector,
                user_id=user_id,
                session_id=session_id,
                model=self.settings.chat_model,
            )
        await self.sync_memory_md(user_id)

    async def sync_memory_md(self, user_id) -> None:
        """
        Rewrite memory.md from the current DB state.
        Direction: DB → file. Called after every mutation so the file is always
        a faithful mirror of active memories.
        """
        memories = await self.memory_repo.get_active(user_id=user_id)
        content = format_memory_md(memories, updated_at=datetime.now(timezone.utc))
        with open(self.settings.memory_md_path, "w", encoding="utf-8") as f:
            f.write(content)

    async def forget(self, memory_id, user_id) -> None:
        """
        Soft-delete a memory by setting status = 'forgotten'.
        Row is preserved in DB for audit; it will never appear in retrieval.
        """
        memory = await self.memory_repo.get_by_id(memory_id)
        if memory is None or memory.user_id != user_id:
            raise ValueError("Memory not found")
        await self.memory_repo.forget(memory_id)
        await self.sync_memory_md(user_id)

    async def update_content(self, memory_id, user_id, new_content: str):
        """
        Edit the content of an existing memory and re-embed it so vector
        search stays accurate. Episodic memories are read-only.
        """
        memory = await self.memory_repo.get_by_id(memory_id)
        if memory is None or memory.user_id != user_id:
            raise ValueError("Memory not found")
        if memory.type == MemoryType.EPISODIC:
            raise ValueError("Episodic memory is read-only")
        vector = await embed(new_content, self.openai, self.settings.embedding_model)
        updated = await self.memory_repo.update_content(memory_id, new_content, vector)
        await self.sync_memory_md(user_id)
        return updated

    async def list_active(self, user_id):
        """Return all active memories grouped by type for the memories API."""
        memories = await self.memory_repo.get_active(user_id=user_id)
        by_type: dict[str, int] = {}
        for m in memories:
            by_type[m.type.value] = by_type.get(m.type.value, 0) + 1
        return {"memories": memories, "total": len(memories), "by_type": by_type}

    async def sync_from_md(self, user_id) -> None:
        """
        Apply manual edits made to memory.md back into the DB.
        Direction: file → DB (reverse of sync_memory_md).
        After applying edits, rewrites the file from DB to restore canonical format.
        """
        await apply_md_edits(
            memory_repo=self.memory_repo,
            memory_service=self,
            user_id=user_id,
            path=self.settings.memory_md_path,
        )
        await self.sync_memory_md(user_id)


async def apply_md_edits(memory_repo, memory_service: MemoryService, user_id, path: str) -> None:
    """
    Parse memory.md and apply any content changes back to the DB.

    Contract:
      - Each bullet line must carry an HTML comment with the memory id:
          - Some fact  <!-- id:uuid confidence:0.90 -->
      - If the content differs from the DB row → update + re-embed.
      - If a line is removed → we do NOT auto-forget (user must use the UI/API).
      - Episodic lines are read-only and are silently skipped.
      - Unknown or malformed ids are skipped and logged.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = f.read()
    except FileNotFoundError:
        return

    bullet_pattern = re.compile(r"^- (.*?)\s+<!--\s*id:([a-f0-9\-]+).*?-->\s*$", re.MULTILINE)
    found = bullet_pattern.findall(data)
    for content, raw_id in found:
        memory = await memory_repo.get_by_id(raw_id)
        if memory is None or memory.user_id != user_id:
            continue
        if memory.type == MemoryType.EPISODIC:
            continue
        if memory.content.strip() != content.strip():
            vector = await embed(content.strip(), memory_service.openai, memory_service.settings.embedding_model)
            await memory_repo.update_content(raw_id, content.strip(), vector)
