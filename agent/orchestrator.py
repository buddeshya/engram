import asyncio
import contextlib
import logging

log = logging.getLogger("engram.orchestrator")


class Orchestrator:
    """
    Central coordinator for a single chat turn.

    Read path (latency-critical, blocks until first token):
      retrieve context → build prompt → stream LLM response

    Write path (fire-and-forget, never blocks the response):
      memory extraction → conflict resolution → md sync
      + auto-title + rolling summary refresh
    """

    def __init__(self, chat_service, memory_service, session_service, message_repo, session_repo):
        self.chat_service = chat_service
        self.memory_service = memory_service
        self.session_service = session_service
        self.message_repo = message_repo
        self.session_repo = session_repo

    async def _run_background(self, coro):
        # Swallow errors so a failing background task never crashes the server.
        try:
            await coro
        except Exception as exc:
            log.error("Background task failed: %s", exc)

    async def handle_chat_stream(self, session_id, user_id, user_message: str):
        """
        Async generator — yields LLM token deltas for SSE streaming.

        After the full response is collected, three background tasks are
        launched via asyncio.create_task so they run concurrently without
        blocking the next request:
          1. memory_service.process_turn  — extract + resolve memories
          2. session_service.auto_title   — set title on first turn only
          3. session_service.refresh_summary_if_needed — rolling summary
        """
        log.info("── TURN START ─────────────────────────────")
        log.info("User: %s", user_message[:120])

        await self.message_repo.insert(session_id, "user", user_message)
        await self.session_repo.touch(session_id)

        log.info("Retrieving context (embedding + vector search)…")
        context = await self.chat_service.retrieve_context(user_id, session_id, user_message)
        log.info(
            "Context ready  semantic=%d  episodic=%d  summary=%s",
            len(context.semantic_memories),
            len(context.episodic_memories),
            "yes" if context.session_summary else "no",
        )
        prompt = await self.chat_service.build_prompt(user_message, context)

        chunks: list[str] = []
        stream = await self.chat_service.openai.chat.completions.create(
            model=self.chat_service.settings.chat_model,
            messages=prompt,
            temperature=0.4,
            stream=True,
        )
        async for part in stream:
            delta = part.choices[0].delta.content if part.choices else None
            if delta:
                chunks.append(delta)
                yield delta

        final_text = "".join(chunks).strip()
        log.info("Stream complete  tokens≈%d", len(final_text.split()))
        if final_text:
            await self.message_repo.insert(session_id, "assistant", final_text)
            await self.session_repo.touch(session_id)
            log.info("Scheduling background tasks: memory_extraction, auto_title, summary_refresh")
            asyncio.create_task(
                self._run_background(
                    self.memory_service.process_turn(
                        user_message=user_message,
                        assistant_message=final_text,
                        session_id=session_id,
                        user_id=user_id,
                    )
                )
            )
            asyncio.create_task(self._run_background(self.session_service.auto_title(session_id, user_message)))
            asyncio.create_task(self._run_background(self.session_service.refresh_summary_if_needed(session_id)))
