import json
from types import SimpleNamespace

import pytest

from memory.extractor import extract
from memory.types import MemoryType


def _chat_response(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.asyncio
async def test_extract_returns_preference_candidate(mock_openai):
    mock_openai.chat.completions.create.return_value = _chat_response(
        json.dumps([{"type": "preference", "content": "Prefers tabs", "confidence": 0.9}])
    )
    out = await extract("u", "a", mock_openai, "gpt-4o-mini")
    assert len(out) == 1
    assert out[0].type == MemoryType.PREFERENCE
    assert out[0].content == "Prefers tabs"
    assert out[0].confidence == 0.9


@pytest.mark.asyncio
async def test_extract_ignores_invalid_and_episodic_types(mock_openai):
    mock_openai.chat.completions.create.return_value = _chat_response(
        json.dumps(
            [
                {"type": "episodic", "content": "bad", "confidence": 1.0},
                {"type": "unknown", "content": "bad", "confidence": 0.8},
            ]
        )
    )
    out = await extract("u", "a", mock_openai, "gpt-4o-mini")
    assert out == []


@pytest.mark.asyncio
async def test_extract_handles_invalid_json(mock_openai):
    mock_openai.chat.completions.create.return_value = _chat_response("not-json")
    out = await extract("u", "a", mock_openai, "gpt-4o-mini")
    assert out == []
