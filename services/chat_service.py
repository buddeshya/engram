from dataclasses import dataclass

from agent.prompt import build_messages
from memory.retriever import retrieve_dual
from repositories.memory_repository import MemoryRepository
from repositories.message_repository import MessageRepository
from repositories.session_repository import SessionRepository


@dataclass
class TurnContext:
    """
    All context assembled for a single chat turn.
    Each field maps to one tier of the four-tier bounded prompt:
      semantic_memories → corrections/preferences/facts/decisions (top-K)
      episodic_memories → past session summaries (top-K)
      session_summary   → rolling summary of current session
      recent_messages   → verbatim last N message pairs (working memory)
    """
    semantic_memories: list
    episodic_memories: list
    session_summary: str | None
    recent_messages: list[dict]


class ChatService:
    """
    Responsible for the read path only — assembling context and building prompts.
    No writes happen here; all persistence is handled by MemoryService and
    SessionService via background tasks after the response is streamed.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        session_repo: SessionRepository,
        message_repo: MessageRepository,
        openai_client,
        settings,
    ):
        self.memory_repo = memory_repo
        self.session_repo = session_repo
        self.message_repo = message_repo
        self.openai = openai_client
        self.settings = settings

    async def retrieve_context(self, user_id, session_id, query: str) -> TurnContext:
        """
        Assemble all four context tiers for the current turn.

        Embedding failure is silently caught — if the vector DB is unreachable,
        the agent falls back to summary + working memory only rather than failing
        the whole request.
        """
        try:
            semantic, episodic = await retrieve_dual(
                memory_repo=self.memory_repo,
                openai_client=self.openai,
                user_id=user_id,
                query=query,
                embedding_model=self.settings.embedding_model,
                semantic_top_k=self.settings.semantic_top_k,
                episodic_top_k=self.settings.episodic_top_k,
            )
        except Exception:
            semantic, episodic = [], []

        session = await self.session_repo.get_by_id(session_id)
        # working_memory_window * 2 because each turn = 1 user + 1 assistant message.
        recent = await self.message_repo.get_recent_n(session_id, self.settings.working_memory_window * 2)
        recent_messages = [{"role": m["role"], "content": m["content"]} for m in recent]
        return TurnContext(
            semantic_memories=semantic,
            episodic_memories=episodic,
            session_summary=(session or {}).get("summary"),
            recent_messages=recent_messages,
        )

    async def build_prompt(self, query: str, context: TurnContext) -> list[dict]:
        """Delegate prompt construction to the pure agent.prompt module."""
        return build_messages(
            query=query,
            semantic_memories=context.semantic_memories,
            episodic_memories=context.episodic_memories,
            session_summary=context.session_summary,
            recent_messages=context.recent_messages,
        )
