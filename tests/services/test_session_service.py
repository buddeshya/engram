from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from services.session_service import SessionService


def _chat_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_auto_title_only_when_missing(settings_stub, mock_openai):
    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value={"id": uuid4(), "title": None}),
        update_title_if_null=AsyncMock(),
    )
    service = SessionService(session_repo, SimpleNamespace(), SimpleNamespace(), mock_openai, settings_stub)
    mock_openai.chat.completions.create.return_value = _chat_response("Build memory pipeline")
    await service.auto_title(uuid4(), "please help build memory pipeline")
    session_repo.update_title_if_null.assert_awaited_once()


@pytest.mark.asyncio
async def test_refresh_summary_skips_below_threshold(settings_stub, mock_openai):
    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(return_value={"id": uuid4(), "summary_turn_count": 8, "summary": None}),
        update_summary=AsyncMock(),
    )
    message_repo = SimpleNamespace(
        count_by_session=AsyncMock(return_value=10),
        get_recent_n=AsyncMock(),
    )
    service = SessionService(session_repo, message_repo, SimpleNamespace(), mock_openai, settings_stub)
    await service.refresh_summary_if_needed(uuid4())
    message_repo.get_recent_n.assert_not_called()
    session_repo.update_summary.assert_not_called()


@pytest.mark.asyncio
async def test_generate_episodic_inserts_memory(settings_stub, mock_openai, monkeypatch):
    session_id = uuid4()
    user_id = uuid4()
    session_repo = SimpleNamespace(get_by_id=AsyncMock(return_value={"id": session_id, "summary": "Worked on API"}))
    message_repo = SimpleNamespace(count_by_session=AsyncMock(return_value=8))
    memory_repo = SimpleNamespace(insert=AsyncMock())
    service = SessionService(session_repo, message_repo, memory_repo, mock_openai, settings_stub)

    mock_openai.chat.completions.create.return_value = _chat_response("He finalized the API and next steps.")
    async def fake_embed(*args, **kwargs):
        return [0.1] * 1536
    monkeypatch.setattr("services.session_service.embed", fake_embed)

    await service.generate_episodic_memory(session_id, user_id)
    memory_repo.insert.assert_awaited_once()
