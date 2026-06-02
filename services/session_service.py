from memory.retriever import embed
from memory.types import MemoryCandidate, MemoryType


class SessionService:
    """
    Manages session lifecycle events that run off the critical path:
      - auto_title               → called on the first turn of each session
      - refresh_summary_if_needed → called after every turn; only fires every K turns
      - generate_episodic_memory  → called explicitly when a session ends
    """

    def __init__(self, session_repo, message_repo, memory_repo, openai_client, settings):
        self.session_repo = session_repo
        self.message_repo = message_repo
        self.memory_repo = memory_repo
        self.openai = openai_client
        self.settings = settings

    async def auto_title(self, session_id, first_message: str) -> None:
        """
        Generate a 4-6 word title from the first user message and write it
        to the session row — but only if the title is still NULL.

        The repository's update_title_if_null uses a conditional UPDATE so
        concurrent calls are safe (second call is a no-op at the SQL level).
        """
        session = await self.session_repo.get_by_id(session_id)
        if session is None or session.get("title"):
            return
        response = await self.openai.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the message in 4-6 words. Return only title text.",
                },
                {"role": "user", "content": first_message},
            ],
        )
        title = (response.choices[0].message.content or "").strip().strip('"')
        if title:
            await self.session_repo.update_title_if_null(session_id, title[:80])

    async def refresh_summary_if_needed(self, session_id) -> None:
        """
        Rolling summary refresh — fires only when at least K*2 new messages
        have arrived since the last summary was written.

        The summary is fed back into itself on each refresh (incremental style),
        so it never grows beyond ~200 words while always reflecting recent turns.
        This keeps the session_summary prompt tier bounded regardless of session length.
        """
        session = await self.session_repo.get_by_id(session_id)
        if session is None:
            return
        total = await self.message_repo.count_by_session(session_id)
        k = self.settings.summary_refresh_every_k * 2  # *2 because user+assistant per turn
        if (total - int(session["summary_turn_count"])) < k:
            return

        recent = await self.message_repo.get_recent_n(session_id, k)
        formatted = "\n".join([f'{m["role"]}: {m["content"]}' for m in recent])
        response = await self.openai.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You produce concise conversation summaries for AI memory."},
                {
                    "role": "user",
                    "content": (
                        f"Previous summary:\n{session.get('summary') or 'None'}\n\n"
                        f"New turns:\n{formatted}\n\n"
                        "Write a new summary in <= 200 words."
                    ),
                },
            ],
        )
        summary = (response.choices[0].message.content or "").strip()
        if summary:
            await self.session_repo.update_summary(session_id, summary, total)

    async def generate_episodic_memory(self, session_id, user_id) -> None:
        """
        Convert the session into a 2-3 sentence episodic memory paragraph and
        store it as a memory row with type='episodic'.

        Episodic memories are retrieved at the start of future sessions via
        vector similarity — so the agent can recall "what happened last time"
        even without seeing the raw message history.

        Skipped if the session has fewer than 4 messages (not enough signal).
        """
        session = await self.session_repo.get_by_id(session_id)
        if session is None:
            return
        count = await self.message_repo.count_by_session(session_id)
        if count < 4:
            return
        response = await self.openai.chat.completions.create(
            model=self.settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You produce episodic memory summaries for an AI agent."},
                {
                    "role": "user",
                    "content": (
                        f"Session summary: {session.get('summary') or 'No summary'}\n"
                        f"Message count: {count}\n"
                        "Write 2-3 sentences in third-person past tense with key decisions and context."
                    ),
                },
            ],
        )
        episodic_text = (response.choices[0].message.content or "").strip()
        if not episodic_text:
            return
        vector = await embed(episodic_text, self.openai, self.settings.embedding_model)
        candidate = MemoryCandidate(
            type=MemoryType.EPISODIC,
            content=episodic_text,
            confidence=1.0,
            metadata={"kind": "session_end"},
        )
        await self.memory_repo.insert(candidate, vector, user_id=user_id, session_id=session_id)
